"""
train_tokenizer.py

Standalone script to train a byte-level BPE tokenizer from scratch.
Runs once to produce tokenizer/vocab.json, tokenizer/merges.txt,
and tokenizer/special_tokens.json.

Currently implemented:
    - Part 1: Pre-tokenizer (GPT-2 regex)
    - Part 2: Byte-level encoding (256-byte base vocabulary)

To be added:
    - Part 3: BPE training algorithm
    - Part 4: Saving artifacts
    - Part 5: Encoder/decoder methods
"""

import regex as re
from functools import lru_cache


# Part 1: Pre-Tokenizer

# GPT-2 / GPT-3 / Llama style pre-tokenizer pattern.
# Splits text into 'word-like' chunks before BPE so merges happen within words.
PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pre_tokenize(text: str) -> list[str]:
    """
    Split raw text into 'word-like' chunks before BPE.

    Each chunk is one of:
        - English contraction (e.g. "'s", "'ll")
        - Letter run with optional leading space (e.g. " hello")
        - Digit run with optional leading space (e.g. " 123")
        - Punctuation/symbol run with optional leading space (e.g. " ...")
        - Whitespace run

    Roundtrip property: "".join(pre_tokenize(text)) == text always.
    """
    return PATTERN.findall(text)



# Part 2: Byte-Level Encoding

@lru_cache(maxsize=None)
def bytes_to_unicode() -> dict[int, str]:
    """
    Build a reversible mapping: byte (0-255) → printable unicode character.

    Why: BPE operates on bytes (256 possibilities cover all UTF-8 text), but
    raw bytes include unprintable control characters that would corrupt
    vocab.json. So we map every byte to a printable unicode character.

    Returns:
        Dict mapping each byte value (0-255) to a unicode string of length 1.
    """
    # Bytes that are already printable (visible) in standard encodings.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]   # printable bytes map to themselves

    # For every byte NOT already printable, assign an unused codepoint
    # starting at 256.
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1

    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


@lru_cache(maxsize=None)
def unicode_to_bytes() -> dict[str, int]:
    """Inverse of bytes_to_unicode — for decoding tokens back to bytes."""
    return {v: k for k, v in bytes_to_unicode().items()}


def chunk_to_byte_string(chunk: str) -> str:
    """
    Convert a text chunk to a byte-level string suitable for BPE.

    Steps:
        1. Encode chunk as UTF-8 bytes.
        2. Map each byte to its printable unicode representative.
        3. Join into a single string.

    Example:
        ' café' → b' caf\\xc3\\xa9' → 6 bytes → 6 chars → 'ĠcafÃ©'
    """
    byte_to_uni = bytes_to_unicode()
    return "".join(byte_to_uni[b] for b in chunk.encode("utf-8"))


def byte_string_to_text(byte_string: str) -> str:
    """
    Convert a byte-level string back to original text.

    Inverse of chunk_to_byte_string. Used at decoding time.
    `errors="replace"` is a safety net for malformed sequences.
    """
    uni_to_byte = unicode_to_bytes()
    byte_array = bytearray(uni_to_byte[c] for c in byte_string)
    return byte_array.decode("utf-8", errors="replace")


# Part 3: BPE Training Algorithm

from collections import Counter, defaultdict
from typing import Iterable


def build_word_counts(text_iterable: Iterable[str]) -> dict[tuple[str, ...], int]:
    """
    Pre-tokenize all text and count unique 'words' (chunks).

    Each word is stored as a tuple of byte-level characters
    (the output of chunk_to_byte_string, split into individual chars).

    Args:
        text_iterable: Yields text chunks (e.g. lines from a file).

    Returns:
        Dict mapping (char1, char2, ...) tuples to their count in the corpus.

    Example:
        "the cat" appears 100 times →
            {('t','h','e'): 100, ('Ġ','c','a','t'): 100}
    """
    word_counts: Counter = Counter()

    for text in text_iterable:
        for chunk in pre_tokenize(text):
            byte_str = chunk_to_byte_string(chunk)
            # Convert string to tuple of chars (immutable, hashable)
            word_counts[tuple(byte_str)] += 1

    return dict(word_counts)


def get_pair_counts(
    word_counts: dict[tuple[str, ...], int]
) -> Counter:
    """
    Count all adjacent character pairs across the corpus.

    Pairs are weighted by word frequency:
        if "the" appears 100 times, ('t','h') gets +100 and ('h','e') gets +100.

    Returns:
        Counter mapping (char1, char2) tuples to total count.
    """
    pair_counts: Counter = Counter()

    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += count

    return pair_counts


def merge_pair_in_word(
    word: tuple[str, ...],
    pair: tuple[str, str],
    new_token: str,
) -> tuple[str, ...]:
    """
    Apply a single merge rule to one word.

    Example:
        word = ('l','o','w','e','r')
        pair = ('l','o')
        new_token = 'lo'
        →  ('lo','w','e','r')

    Returns:
        New word tuple with all occurrences of `pair` replaced by `new_token`.
    """
    new_word = []
    i = 0
    while i < len(word):
        # Check if current position starts the pair
        if (
            i < len(word) - 1
            and word[i] == pair[0]
            and word[i + 1] == pair[1]
        ):
            new_word.append(new_token)
            i += 2  # skip both merged chars
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def train_bpe(
    word_counts: dict[tuple[str, ...], int],
    num_merges: int,
    verbose: bool = True,
) -> list[tuple[str, str]]:
    """
    Train BPE merges on the corpus.

    Args:
        word_counts: Output of build_word_counts.
        num_merges: How many merge rules to learn.
        verbose: Print progress every 500 merges.

    Returns:
        Ordered list of merge rules. Each merge is (left, right).
        Apply them in order at encoding time.
    """
    # Make a mutable copy we'll update during training
    word_counts = dict(word_counts)
    merges: list[tuple[str, str]] = []

    # Initial pair counts across entire corpus
    pair_counts = get_pair_counts(word_counts)

    for merge_idx in range(num_merges):
        if not pair_counts:
            print(f"No more pairs to merge at iteration {merge_idx}.")
            break

        # Find the most frequent pair
        # (Counter.most_common(1) returns [((pair), count)])
        best_pair, best_count = pair_counts.most_common(1)[0]

        if best_count < 2:
            # No pair appears more than once — vocabulary saturated
            print(f"All remaining pairs appear < 2 times. Stopping at {merge_idx}.")
            break

        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # Update word_counts: replace best_pair with new_token in every word
        # Also incrementally update pair_counts.
        new_word_counts: dict[tuple[str, ...], int] = {}

        for word, count in word_counts.items():
            # Skip words that don't contain the pair (fast check)
            if best_pair[0] not in word:
                new_word_counts[word] = count
                continue

            # Apply the merge to this word
            new_word = merge_pair_in_word(word, best_pair, new_token)

            if new_word == word:
                # Pair didn't actually appear (the 'in' check was a false positive)
                new_word_counts[word] = count
                continue

            # Subtract old pairs in `word`, add new pairs in `new_word`
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] -= count
            for i in range(len(new_word) - 1):
                pair_counts[(new_word[i], new_word[i + 1])] += count

            # Clean up zero/negative counts
            # (Counter doesn't auto-remove these)
            new_word_counts[new_word] = new_word_counts.get(new_word, 0) + count

        # Remove the merged pair itself (count is now 0 or negative)
        del pair_counts[best_pair]
        # Drop any other pairs that have hit zero
        pair_counts = Counter({p: c for p, c in pair_counts.items() if c > 0})

        word_counts = new_word_counts

        if verbose and (merge_idx + 1) % 500 == 0:
            print(
                f"Merge {merge_idx + 1:>5}/{num_merges} "
                f"| {best_pair[0]!r}+{best_pair[1]!r} → {new_token!r} "
                f"(count={best_count:,})"
            )

    return merges


# Part 4: Loading Corpus + Saving Artifacts

import json
import os
from typing import Iterator
from src.config import TransformerConfig


# Special tokens — order matters, defines their token IDs
SPECIAL_TOKENS = [
    "<|pad|>",         # ID 0
    "<|endoftext|>",   # ID 1
    "<|bos|>",         # ID 2
    "<|im_start|>",    # ID 3 (chat: start of turn)
    "<|im_end|>",      # ID 4 (chat: end of turn)
]


def iter_corpus(
    path: str,
    max_lines: int | None = None,
    strip_eot: bool = True,
) -> Iterator[str]:
    """
    Stream lines from a text file.

    Args:
        path: Path to the text file (.txt).
        max_lines: If set, stop after this many lines.
        strip_eot: If True, remove '<|endoftext|>' markers
                   so they aren't seen by BPE.

    Yields:
        Each line as a string.
    """
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            if strip_eot:
                line = line.replace("<|endoftext|>", "")
            yield line


def build_vocab(
    merges: list[tuple[str, str]],
    special_tokens: list[str],
) -> dict[str, int]:
    """
    Build the final token→ID vocabulary.

    Layout:
        IDs 0 to N-1            → special tokens
        IDs N to N+255          → single-byte tokens (printable representatives)
        IDs N+256 onward        → BPE merge results, in learning order

    Args:
        merges: Ordered list of BPE merges.
        special_tokens: List of special token strings.

    Returns:
        Dict mapping token string → int ID.
    """
    vocab: dict[str, int] = {}

    # 1. Special tokens
    for tok in special_tokens:
        vocab[tok] = len(vocab)

    # 2. All 256 single-byte tokens (printable representatives)
    byte_to_uni = bytes_to_unicode()
    for b in range(256):
        char = byte_to_uni[b]
        vocab[char] = len(vocab)

    # 3. BPE merges (each merge produces a new token = concatenation)
    for left, right in merges:
        merged = left + right
        if merged not in vocab:
            vocab[merged] = len(vocab)

    return vocab


def save_tokenizer(
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
    special_tokens: list[str],
    output_dir: str,
) -> None:
    """
    Save tokenizer artifacts: vocab.json, merges.txt, special_tokens.json.

    Args:
        vocab: token → id mapping.
        merges: Ordered list of BPE merges.
        special_tokens: List of special token strings.
        output_dir: Directory to save files in (created if missing).
    """
    os.makedirs(output_dir, exist_ok=True)

    # vocab.json
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  Saved {vocab_path}  ({len(vocab):,} tokens)")

    # merges.txt
    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for left, right in merges:
            f.write(f"{left} {right}\n")
    print(f"  Saved {merges_path}  ({len(merges):,} merges)")

    # special_tokens.json
    specials_path = os.path.join(output_dir, "special_tokens.json")
    specials_data = {
        "special_tokens": special_tokens,
        "pad_token_id": vocab[special_tokens[0]],
        "eos_token_id": vocab[special_tokens[1]],
        "bos_token_id": vocab[special_tokens[2]],
    }
    with open(specials_path, "w", encoding="utf-8") as f:
        json.dump(specials_data, f, indent=2)
    print(f"  Saved {specials_path}")


def train_tokenizer_from_file(
    train_file: str,
    output_dir: str,
    vocab_size: int,
    special_tokens: list[str] = SPECIAL_TOKENS,
    max_lines: int | None = None,
) -> None:
    """
    End-to-end tokenizer training: read corpus → train BPE → save artifacts.

    Args:
        train_file: Path to the training text file.
        output_dir: Where to save vocab.json, merges.txt, special_tokens.json.
        vocab_size: Target vocabulary size.
        special_tokens: Special token strings (reserved IDs at the start).
        max_lines: If set, only use this many lines (for quick testing).
    """
    # How many BPE merges we need:
    # vocab_size = len(specials) + 256 (bytes) + num_merges
    num_merges = vocab_size - len(special_tokens) - 256
    if num_merges <= 0:
        raise ValueError(
            f"vocab_size ({vocab_size}) too small. "
            f"Need at least {len(special_tokens) + 256 + 1}."
        )

    print(f"Training BPE tokenizer:")
    print(f"  corpus      : {train_file}")
    print(f"  vocab_size  : {vocab_size:,}")
    print(f"  num_merges  : {num_merges:,}")
    print(f"  specials    : {len(special_tokens)}")
    if max_lines:
        print(f"  max_lines   : {max_lines:,}  (subset for testing)")
    print()

    # Step 1: count words
    print("[1/3] Counting unique words from corpus...")
    word_counts = build_word_counts(iter_corpus(train_file, max_lines=max_lines))
    print(f"  {len(word_counts):,} unique words found")
    total_word_occurrences = sum(word_counts.values())
    print(f"  {total_word_occurrences:,} total word occurrences")
    print()

    # Step 2: train BPE
    print(f"[2/3] Training {num_merges:,} BPE merges...")
    merges = train_bpe(word_counts, num_merges=num_merges, verbose=True)
    print(f"  Learned {len(merges):,} merges")
    print()

    # Step 3: build vocab and save
    print("[3/3] Building vocabulary and saving artifacts...")
    vocab = build_vocab(merges, special_tokens)
    save_tokenizer(vocab, merges, special_tokens, output_dir)
    print()
    print(f"Done. Tokenizer saved to {output_dir}/")

# Tests

if __name__ == "__main__":
    # ---- Part 1: Pre-tokenizer tests ----
    print("--- Pre-tokenizer test ---")
    test_cases = [
        "Hello, world!",
        "Don't you love AI?",
        "Once upon a time   there was   a cat.",
        "Numbers: 3.14 and 2024.",
        "Émojis café résumé",
        "",
        "   ",
        "Tabs\tand\nnewlines",
    ]

    for text in test_cases:
        chunks = pre_tokenize(text)
        roundtrip = "".join(chunks)
        ok = "✓" if roundtrip == text else "✗"
        print(f"{ok} {chunks!r}")
        assert roundtrip == text, f"Roundtrip failed for {text!r}"

    # ---- Part 2: Byte-level encoding tests ----
    print("\n--- Byte-level encoding test ---")
    test_chunks = [
        " hello",
        " café",
        " 你好",
        " 🚀",
        "\n",
        " ",
    ]

    for chunk in test_chunks:
        byte_str = chunk_to_byte_string(chunk)
        recovered = byte_string_to_text(byte_str)
        ok = "✓" if recovered == chunk else "✗"
        print(f"{ok} {chunk!r:15} → {byte_str!r:20} → {recovered!r}")
        assert recovered == chunk, f"Roundtrip failed for {chunk!r}"

    # Sanity check: mapping covers all 256 bytes
    mapping = bytes_to_unicode()
    assert len(mapping) == 256, f"Expected 256 byte mappings, got {len(mapping)}"
    assert len(set(mapping.values())) == 256, "Mapping not invertible!"
    print("\n✓ Byte mapping covers all 256 bytes and is invertible")

    print("\nAll tests passed.")
    
    # Part 3 tests


    print("\n" + "=" * 60)
    print("Part 3: BPE training tests")
    print("=" * 60)


    # ---- Test 3.1: word counting ----
    print("\n[3.1] Word counting")
    sample_text = ["the cat sat", "the cat ran"]
    word_counts = build_word_counts(sample_text)

    # 'the' appears 2x, ' cat' appears 2x, ' sat' 1x, ' ran' 1x
    expected_min_words = 4   # at least these unique chunks
    assert len(word_counts) >= expected_min_words, \
        f"Expected ≥{expected_min_words} unique words, got {len(word_counts)}"

    # Check that 'the' (as byte-string tuple) has count 2
    the_tuple = tuple(chunk_to_byte_string("the"))
    assert word_counts.get(the_tuple) == 2, \
        f"'the' should appear 2x, got {word_counts.get(the_tuple)}"

    # Check that ' cat' (with leading space → 'Ġcat') has count 2
    cat_tuple = tuple(chunk_to_byte_string(" cat"))
    assert word_counts.get(cat_tuple) == 2, \
        f"' cat' should appear 2x, got {word_counts.get(cat_tuple)}"

    print(f"✓ {len(word_counts)} unique words, frequencies correct")


    # ---- Test 3.2: pair counting weights by word frequency ----
    print("\n[3.2] Pair counting respects word frequency")
    # Word ('a','b','c') with count 5 should contribute:
    #   ('a','b'): 5,  ('b','c'): 5
    test_word_counts = {
        ('a', 'b', 'c'): 5,
        ('a', 'b'): 3,
    }
    pair_counts = get_pair_counts(test_word_counts)

    assert pair_counts[('a', 'b')] == 5 + 3, \
        f"('a','b') should be 8, got {pair_counts[('a','b')]}"
    assert pair_counts[('b', 'c')] == 5, \
        f"('b','c') should be 5, got {pair_counts[('b','c')]}"
    print(f"✓ pair counts correctly weighted by word frequency")


    # ---- Test 3.3: merge_pair_in_word correctness ----
    print("\n[3.3] Merge application")

    # Basic merge
    result = merge_pair_in_word(('l', 'o', 'w', 'e', 'r'), ('l', 'o'), 'lo')
    assert result == ('lo', 'w', 'e', 'r'), f"Got {result}"

    # Merge at end
    result = merge_pair_in_word(('l', 'o', 'w'), ('o', 'w'), 'ow')
    assert result == ('l', 'ow'), f"Got {result}"

    # Multiple non-overlapping occurrences
    result = merge_pair_in_word(('a', 'b', 'a', 'b', 'a', 'b'), ('a', 'b'), 'ab')
    assert result == ('ab', 'ab', 'ab'), f"Got {result}"

    # Pair not present — word unchanged
    result = merge_pair_in_word(('x', 'y', 'z'), ('a', 'b'), 'ab')
    assert result == ('x', 'y', 'z'), f"Got {result}"

    # Edge case: single-char word
    result = merge_pair_in_word(('x',), ('a', 'b'), 'ab')
    assert result == ('x',), f"Got {result}"

    # Edge case: overlapping pattern — should NOT double-merge.
    # ('a','a','a') with merge ('a','a') → ('aa','a'), not ('aa','aa')
    result = merge_pair_in_word(('a', 'a', 'a'), ('a', 'a'), 'aa')
    assert result == ('aa', 'a'), f"Got {result} — overlapping merges broken!"

    print(f"✓ merge correctly handles all positions, including overlaps")


    # ---- Test 3.4: BPE picks most frequent pair first ----
    print("\n[3.4] BPE picks most frequent pair first")

    # Construct a corpus where ('t','h') is unambiguously most frequent
    # 'the' appears 100 times, 'cat' appears 5 times
    controlled_counts = {
        ('t', 'h', 'e'): 100,
        ('c', 'a', 't'): 5,
    }
    merges = train_bpe(controlled_counts, num_merges=1, verbose=False)
    assert merges[0] == ('t', 'h'), \
        f"First merge should be ('t','h'), got {merges[0]}"
    print(f"✓ first merge = {merges[0]} (most frequent pair)")


    # ---- Test 3.5: BPE produces correct vocabulary growth ----
    print("\n[3.5] BPE merges grow vocabulary as expected")

    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog",
    ] * 50

    word_counts = build_word_counts(corpus)
    merges = train_bpe(word_counts, num_merges=20, verbose=False)

    assert len(merges) == 20, f"Should have 20 merges, got {len(merges)}"

    # Every merge must be unique
    assert len(set(merges)) == 20, "Duplicate merges produced!"

    # Common pattern check: 'th' merge should appear in the first 10 merges
    # (since 'the' appears in every input line)
    merge_strings = [a + b for a, b in merges]
    assert "th" in merge_strings[:10], \
        f"'th' merge expected in first 10, got {merge_strings[:10]}"

    print(f"✓ 20 unique merges, common patterns learned early")
    print(f"  first 5 merges: {merges[:5]}")


    # ---- Test 3.6: BPE stops cleanly when no merges are useful ----
    print("\n[3.6] BPE stops early on saturated corpus")

    # Tiny corpus with very few unique pairs
    saturated = {
        ('a', 'b'): 1,   # only one occurrence each — nothing to merge meaningfully
        ('c', 'd'): 1,
    }
    merges = train_bpe(saturated, num_merges=100, verbose=False)
    # Will print "All remaining pairs appear < 2 times" — that's expected output
    assert len(merges) == 0, f"Should learn 0 merges (no pair > 1), got {len(merges)}"
    print(f"✓ correctly stops when no pair appears ≥ 2 times")


    # ---- Test 3.7: Merges can be applied to encode a new word ----
    print("\n[3.7] Learned merges can re-encode words")

    corpus = ["the cat", "the cat", "the cat"] * 10
    word_counts = build_word_counts(corpus)
    merges = train_bpe(word_counts, num_merges=10, verbose=False)

    # Apply learned merges to "the" from scratch
    test_word = tuple(chunk_to_byte_string("the"))
    for left, right in merges:
        test_word = merge_pair_in_word(test_word, (left, right), left + right)

    # After merging, 'the' should compress to fewer tokens than its 3 chars
    assert len(test_word) < 3, \
        f"'the' should compress, still has {len(test_word)} tokens: {test_word}"
    print(f"✓ 'the' encodes to {test_word} after applying merges (compressed from 3 chars)")


    print("\n" + "=" * 60)
    print("All Part 3 tests passed.")
    print("=" * 60)

    
    # Part 4: smoke test on small corpus subset

    print("\n" + "=" * 60)
    print("Part 4: smoke test (small subset)")
    print("=" * 60)

    cfg = TransformerConfig()
    train_file = os.path.join(cfg.data_dir, "TinyStoriesV2-GPT4-train.txt")

    if os.path.exists(train_file):
        # Train on first 5,000 lines with a small vocab — should take ~30 sec
        train_tokenizer_from_file(
            train_file=train_file,
            output_dir="tokenizer_smoketest",   # separate dir so it doesn't overwrite real one
            vocab_size=2000,                     # tiny vocab for quick test
            max_lines=5_000,
        )

        # Verify files were created
        for fname in ["vocab.json", "merges.txt", "special_tokens.json"]:
            path = os.path.join("tokenizer_smoketest", fname)
            assert os.path.exists(path), f"Missing {path}"
            size = os.path.getsize(path)
            print(f"  ✓ {path}  ({size:,} bytes)")

        # Peek at first 10 merges
        with open("tokenizer_smoketest/merges.txt") as f:
            lines = f.readlines()
        print("\n  First 10 merges:")
        for line in lines[1:11]:
            print(f"    {line.rstrip()}")

        # Cleanup
        import shutil
        shutil.rmtree("tokenizer_smoketest")
        print("\n  ✓ smoke test passed (cleanup done)")
    else:
        print(f"  ⚠ {train_file} not found — skipping smoke test")


# Main entry point

def main():
    cfg = TransformerConfig()
    train_file = os.path.join(cfg.data_dir, "TinyStoriesV2-GPT4-train.txt")

    train_tokenizer_from_file(
        train_file=train_file,
        output_dir=cfg.tokenizer_dir,
        vocab_size=cfg.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        max_lines=None,  # full corpus; set to e.g. 100_000 for a quick test
    )


# To actually train the tokenizer, run:  python -c "from train_tokenizer import main; main()"
# The existing if __name__ == "__main__" block runs the test suite.