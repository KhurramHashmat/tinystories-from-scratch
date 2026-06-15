"""
src/dataset.py

PyTorch Dataset for Phase 1 (TinyStories pretraining).

Loads a pre-tokenized binary file via memory mapping. Each __getitem__
returns a randomly sampled (input_ids, labels) pair for next-token
prediction. Labels are input_ids shifted by 1.

Phase 2 (chat fine-tuning with loss masking) will live in this file too
once we get there.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TokenStreamDataset(Dataset):
    """
    Random-access dataset over a pre-tokenized binary file.

    Each __getitem__ samples a random starting position and returns
    a sequence of length seq_len for input plus the next token for
    each position (labels shifted by 1).

    Stories are NOT separated — sequences may cross story boundaries
    (marked by <|endoftext|> = token ID 1). The model learns this.
    """

    def __init__(
        self,
        bin_path: str,
        seq_len: int,
        epoch_size: int | None = None,
    ):
        """
        Args:
            bin_path: Path to the .bin file of uint16 token IDs.
            seq_len: Length of each input sequence.
            epoch_size: Samples per epoch. Defaults to total_tokens // seq_len.
        """
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Token file not found: {bin_path}")

        self.bin_path = bin_path
        self.seq_len = seq_len

        # Memory-map the file: instant, no RAM cost.
        # OS pages bytes from disk on access.
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

        self.total_tokens = len(self.data)

        # Need at least seq_len + 1 tokens for one sample
        # (seq_len for input, +1 for the final label position).
        if self.total_tokens < self.seq_len + 1:
            raise ValueError(
                f"File has {self.total_tokens} tokens, "
                f"need ≥ {self.seq_len + 1} for seq_len={self.seq_len}"
            )

        # Valid starting positions: 0 to (total - seq_len - 1) inclusive.
        self.max_start = self.total_tokens - self.seq_len - 1

        # Default "epoch" size: approximate number of non-overlapping windows
        self.epoch_size = (
            epoch_size if epoch_size is not None
            else self.total_tokens // self.seq_len
        )

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one random (input_ids, labels) sample.

        Note: `idx` is ignored. Each call returns a fresh random sample.
        This is intentional — random sampling gives more data diversity
        than fixed-slice indexing.
        """
        # Pick a random starting position.
        # np.random is fine here (each worker process reseeds independently).
        start = np.random.randint(0, self.max_start + 1)

        # Read seq_len + 1 tokens (extra one for the shifted label)
        # .astype(np.int64) → PyTorch's default integer dtype for embedding lookup
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)

        # Convert to tensor and split
        chunk_tensor = torch.from_numpy(chunk)
        input_ids = chunk_tensor[:-1]   # tokens [0 .. seq_len-1]
        labels = chunk_tensor[1:]       # tokens [1 .. seq_len]   (shifted by 1)

        return input_ids, labels


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with sensible defaults for training.
    
    Defaults num_workers=0 on Windows (faster for memmap-backed datasets)
    and 2 on Linux/Mac (where fork() makes workers nearly free).
    """
    if num_workers is None:
        num_workers = 0 if sys.platform == "win32" else 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )