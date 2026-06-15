from dataclasses import dataclass, asdict
import json
import os


@dataclass
class TransformerConfig:

    # Vocabulary
    vocab_size: int = 32000
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2

    # Model Architecture
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072  # 4 × d_model
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_base: int = 10000  # RoPE theta

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 16          # effective batch = 8 × 16 = 128
    learning_rate: float = 3e-4
    min_lr: float = 3e-5               # cosine decay floor (10% of peak)
    warmup_steps: int = 200
    max_steps_phase1: int = 5000    # ~Approx. 6 hours on TinyStories
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95            # 0.95 is standard for transformer pretraining; see nanoGPT, GPT-3, LLaMA

    # Validation & Checkpointing
    val_interval: int = 500            # running val every N steps
    val_batches: int = 50              # how many val batches per val run
    save_interval: int = 500           # saving periodic checkpoint every N steps
    keep_last_k: int = 3               # how many periodic checkpoints to keep

    # Resumable training
    max_train_hours: float = 6.0       # auto-stop after this many hours per session
    seed: int = 1337                   # reproducibility
    log_interval: int = 10             # log metrics to MLflow every N steps

    # Paths
    data_dir: str = "data"
    tokenizer_dir: str = "tokenizer"
    checkpoint_dir: str = "checkpoints"
    train_bin: str = "train.bin"       # pre-tokenized training data
    val_bin: str = "val.bin"           # pre-tokenized validation data

    # Precision
    dtype: str = "bfloat16"            # bfloat16 | float16 | float32

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.dtype in ("bfloat16", "float16", "float32"), \
            f"dtype must be bfloat16 | float16 | float32, got {self.dtype}"
        assert self.min_lr < self.learning_rate, \
            "min_lr must be less than learning_rate"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    def save(self, path: str) -> None:
        """Save config to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TransformerConfig":
        """Load config from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self) -> str:
        lines = ["TransformerConfig("]
        for k, v in asdict(self).items():
            lines.append(f"  {k}: {v}")
        lines.append(")")
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = TransformerConfig()
    print(cfg)
    print(f"\nhead_dim       : {cfg.head_dim}")
    print(f"effective_batch: {cfg.effective_batch_size}")

    # Rough param count estimate
    embedding_params = cfg.vocab_size * cfg.d_model
    attn_per_layer   = 4 * cfg.d_model ** 2
    ffn_per_layer    = 2 * cfg.d_model * cfg.d_ff
    per_layer        = attn_per_layer + ffn_per_layer
    total            = embedding_params + cfg.n_layers * per_layer
    print(f"~param count   : {total / 1e6:.1f}M")

    # Test save/load roundtrip
    cfg.save("checkpoints/test_config.json")
    cfg2 = TransformerConfig.load("checkpoints/test_config.json")
    assert cfg == cfg2, "save/load roundtrip failed!"
    print("✓ save/load roundtrip works")

    os.remove("checkpoints/test_config.json")