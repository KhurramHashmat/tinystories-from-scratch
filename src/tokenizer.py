"""
src/tokenizer.py

Runtime BPE tokenizer. Loads vocab.json + merges.txt produced by
train_tokenizer.py and provides encode() / decode() methods.
"""

import json
import os
import regex as re
from functools import lru_cache


# ============================================================================
# Helpers (duplicated from train_tokenizer.py to keep src/ self-contained)
# ============================================================================

GPT2_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@lru_cache(maxsize=None)
def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
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
    return {v: k for k, v in bytes_to_unicode().items()}


# ============================================================================
# BPE Tokenizer
# ============================================================================

class BPETokenizer:
    """
    Byte-level BPE tokenizer compatible with GPT-2 / Llama style.
    Loads from vocab.json + merges.txt + special_tokens.json.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: list[str],
    ):
        self.vocab = vocab                                      # token → id
        self.inverse_vocab = {v: k for k, v in vocab.items()}   # id → token
        self.special_tokens = special_tokens

        # Merge ranks: merge → priority (lower = applied first)
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        # Special token IDs (assumes special tokens are first in vocab)
        self.pad_token_id = vocab[special_tokens[0]]
        self.eos_token_id = vocab[special_tokens[1]]
        self.bos_token_id = vocab[special_tokens[2]]

        # Regex to match any special token (used during encoding)
        if special_tokens:
            escaped = [re.escape(t) for t in special_tokens]
            self.special_pattern = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_pattern = None

        # Cache for encoded chunks (huge speedup — same words repeat constantly)
        self._encode_cache: dict[str, tuple[int, ...]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(cls, tokenizer_dir: str) -> "BPETokenizer":
        """Load tokenizer from a directory containing the artifact files."""
        vocab_path = os.path.join(tokenizer_dir, "vocab.json")
        merges_path = os.path.join(tokenizer_dir, "merges.txt")
        specials_path = os.path.join(tokenizer_dir, "special_tokens.json")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if lines and lines[0].startswith("#"):
            lines = lines[1:]
        merges = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(" ")
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

        with open(specials_path, "r", encoding="utf-8") as f:
            specials_data = json.load(f)
        special_tokens = specials_data["special_tokens"]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # ------------------------------------------------------------------
    # Core BPE
    # ------------------------------------------------------------------

    def _bpe_encode_chunk(self, chunk: str) -> tuple[str, ...]:
        """Apply BPE merges to a single pre-tokenized chunk."""
        byte_to_uni = bytes_to_unicode()
        word = tuple(byte_to_uni[b] for b in chunk.encode("utf-8"))

        if len(word) == 1:
            return word

        # Greedy merge: at each step, find the lowest-rank applicable merge
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]

            best_pair = None
            best_rank = float("inf")
            for pair in pairs:
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)

            if len(word) == 1:
                break

        return word

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        Encode text to a list of token IDs.

        Args:
            text: Input text.
            add_special_tokens: If True, prepend BOS and append EOS.

        Returns:
            List of integer token IDs.
        """
        ids: list[int] = []

        # Split text on special tokens (so they don't get BPE'd)
        if self.special_pattern is not None:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self.vocab and part in self.special_tokens:
                ids.append(self.vocab[part])
            else:
                for chunk in GPT2_PATTERN.findall(part):
                    if chunk in self._encode_cache:
                        ids.extend(self._encode_cache[chunk])
                        continue
                    tokens = self._bpe_encode_chunk(chunk)
                    chunk_ids = tuple(self.vocab[t] for t in tokens)
                    self._encode_cache[chunk] = chunk_ids
                    ids.extend(chunk_ids)

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """
        Decode a list of token IDs back to text.

        Args:
            ids: List of token IDs.
            skip_special_tokens: If True, omit special tokens from output.

        Returns:
            Decoded string.
        """
        tokens: list[str] = []
        for tid in ids:
            tok = self.inverse_vocab.get(tid)
            if tok is None:
                continue
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)

        uni_to_byte = unicode_to_bytes()
        result = ""
        buffer = bytearray()

        for tok in tokens:
            if tok in self.special_tokens:
                # Flush buffered bytes first
                if buffer:
                    result += buffer.decode("utf-8", errors="replace")
                    buffer = bytearray()
                result += tok
            else:
                # Convert each char in the token back to its byte
                for c in tok:
                    if c in uni_to_byte:
                        buffer.append(uni_to_byte[c])

        if buffer:
            result += buffer.decode("utf-8", errors="replace")

        return result

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"specials={len(self.special_tokens)}, "
            f"cache_size={len(self._encode_cache)})"
        )