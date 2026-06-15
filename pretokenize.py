"""
pretokenize.py

One-time script: encode the entire training and validation corpora into
binary files of uint16 token IDs. Run after the tokenizer is trained.

Output:
    data/train.bin  — all training tokens as raw uint16
    data/val.bin    — all validation tokens as raw uint16
    data/meta.json  — metadata (count, dtype, vocab_size)

Run:
    python pretokenize.py
"""

import os
import json
import time
import numpy as np
from tqdm import tqdm

from src.config import TransformerConfig
from src.tokenizer import BPETokenizer


# How many tokens to flush to disk at a time. 1M = ~2MB per flush.
# Larger = fewer disk writes but more RAM. 1M is a good balance.
FLUSH_INTERVAL = 1_000_000


def encode_file_to_bin(
    tokenizer: BPETokenizer,
    input_path: str,
    output_path: str,
    label: str,
) -> int:
    """
    Stream a text file, encode it line-by-line, write tokens to a .bin file.

    Args:
        tokenizer: Trained BPE tokenizer.
        input_path: Path to input .txt file.
        output_path: Path to output .bin file.
        label: 'train' or 'val' (for progress bar).

    Returns:
        Total number of tokens written.
    """
    file_size = os.path.getsize(input_path)
    total_tokens = 0
    buffer: list[int] = []

    # We'll write in append-binary mode. Open output once, keep flushing.
    with open(output_path, "wb") as out_f:
        with open(input_path, "r", encoding="utf-8") as in_f:
            pbar = tqdm(
                total=file_size,
                unit="B",
                unit_scale=True,
                desc=f"Encoding {label}",
            )

            for line in in_f:
                pbar.update(len(line.encode("utf-8")))

                # Encode the line — tokenizer handles <|endoftext|> as ID 1
                ids = tokenizer.encode(line)
                buffer.extend(ids)

                # Flush periodically
                if len(buffer) >= FLUSH_INTERVAL:
                    arr = np.array(buffer, dtype=np.uint16)
                    arr.tofile(out_f)
                    total_tokens += len(buffer)
                    buffer.clear()

            # Final flush
            if buffer:
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(out_f)
                total_tokens += len(buffer)

            pbar.close()

    return total_tokens


def main():
    cfg = TransformerConfig()

    # Paths
    train_txt = os.path.join(cfg.data_dir, "TinyStoriesV2-GPT4-train.txt")
    val_txt = os.path.join(cfg.data_dir, "TinyStoriesV2-GPT4-valid.txt")
    train_bin = os.path.join(cfg.data_dir, "train.bin")
    val_bin = os.path.join(cfg.data_dir, "val.bin")
    meta_path = os.path.join(cfg.data_dir, "meta.json")

    # Sanity checks
    for path in [train_txt, val_txt]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input: {path}")

    # Verify vocab fits in uint16 (max 65535)
    if cfg.vocab_size > 65_535:
        raise ValueError(
            f"vocab_size ({cfg.vocab_size}) exceeds uint16 limit. "
            f"Use uint32 instead."
        )

    print("Loading tokenizer...")
    tokenizer = BPETokenizer.from_dir(cfg.tokenizer_dir)
    print(f"  {tokenizer}")
    print()

    # Encode train
    t0 = time.time()
    n_train = encode_file_to_bin(tokenizer, train_txt, train_bin, "train")
    print(f"  Train: {n_train:,} tokens in {time.time() - t0:.1f}s")
    print(f"  Saved {train_bin} ({os.path.getsize(train_bin) / 1e9:.2f} GB)")
    print()

    # Encode val
    t0 = time.time()
    n_val = encode_file_to_bin(tokenizer, val_txt, val_bin, "val  ")
    print(f"  Val:   {n_val:,} tokens in {time.time() - t0:.1f}s")
    print(f"  Saved {val_bin} ({os.path.getsize(val_bin) / 1e6:.1f} MB)")
    print()

    # Save metadata
    meta = {
        "train_tokens": n_train,
        "val_tokens": n_val,
        "dtype": "uint16",
        "vocab_size": cfg.vocab_size,
        "tokenizer_cache_size": len(tokenizer._encode_cache),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved {meta_path}")
    print()

    # Summary
    print("=" * 50)
    print("Pre-tokenization complete.")
    print(f"  Train tokens:  {n_train:>14,}")
    print(f"  Val tokens:    {n_val:>14,}")
    print(f"  Total:         {n_train + n_val:>14,}")
    print("=" * 50)


if __name__ == "__main__":
    main()