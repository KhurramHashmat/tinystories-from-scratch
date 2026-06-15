"""
train.py

Training script for the transformer.
Phase 1: pretraining on TinyStories.
Phase 2: chat fine-tuning (added later).

Supports resumable training in time-limited sessions.

Run from project root:
    python train.py --phase 1
    python train.py --phase 1 --resume checkpoints/latest.pt
"""

import os
import math
import time
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import mlflow

from src.config import TransformerConfig
from src.model import TransformerLM
from src.dataset import TokenStreamDataset, make_dataloader


# ============================================================================
# Part 1: Setup helpers
# ============================================================================

def get_device() -> torch.device:
    """Auto-select CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Memory: {total_mem:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available — using CPU (this will be slow)")
    return device


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across torch, cuda, and numpy."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def make_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    Split parameters into decay (2D+) and no-decay (1D) groups.
    Deduplicates tied weights so a shared tensor isn't added to the
    optimizer twice (which would double its weight decay).
    """
    decay_params = []
    no_decay_params = []
    seen = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print("Parameter groups:")
    print(f"  with weight decay:    {len(decay_params):>4} tensors, {n_decay:>12,} params")
    print(f"  without weight decay: {len(no_decay_params):>4} tensors, {n_no_decay:>12,} params")

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine learning-rate schedule with linear warmup.

    Phases:
        step < warmup_steps   → linear ramp from ~0 to peak_lr
        warmup ≤ step ≤ max   → cosine decay from peak_lr to min_lr
        step > max_steps      → constant min_lr (floor)
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


# ============================================================================
# Part 2: Checkpoint Management
# ============================================================================

def save_checkpoint(
    path: str,
    step: int,
    phase: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    config: TransformerConfig,
) -> None:
    """Save complete training state atomically (temp file + rename)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "step": step,
        "phase": phase,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": asdict(config),
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load checkpoint and restore model (+ optionally optimizer) state."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"].cpu())
    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())

    return {
        "step": checkpoint.get("step", 0),
        "phase": checkpoint.get("phase", 1),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
    }


def prune_checkpoints(checkpoint_dir: str, keep_last_k: int) -> None:
    """Delete old periodic step_*.pt checkpoints, keeping the most recent K."""
    if not os.path.isdir(checkpoint_dir):
        return

    step_ckpts = []
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("step_") and fname.endswith(".pt"):
            try:
                step_num = int(fname[len("step_"):-len(".pt")])
                step_ckpts.append((step_num, fname))
            except ValueError:
                continue

    step_ckpts.sort()
    to_delete = step_ckpts[:-keep_last_k] if keep_last_k > 0 else step_ckpts
    for _, fname in to_delete:
        os.remove(os.path.join(checkpoint_dir, fname))


# ============================================================================
# Part 3: Validation Loop
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    n_batches: int,
    dtype: torch.dtype,
) -> dict:
    """Run validation and return average loss + perplexity."""
    model.eval()

    total_loss = 0.0
    count = 0

    val_iter = iter(val_loader)
    for _ in range(n_batches):
        try:
            input_ids, labels = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            input_ids, labels = next(val_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        total_loss += loss.item()
        count += 1

    model.train()

    avg_loss = total_loss / max(count, 1)
    perplexity = math.exp(min(avg_loss, 20))

    return {"loss": avg_loss, "perplexity": perplexity}


# ============================================================================
# Part 4: Main Training Loop
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train the transformer.")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Training phase: 1=pretraining, 2=chat fine-tuning")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def dtype_from_str(dtype_str: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]


def setup_training(args) -> dict:
    """Build config, model, optimizer, data loaders. Resolve phase hyperparameters."""
    cfg = TransformerConfig()
    set_seed(cfg.seed)
    device = get_device()

    # Phase-specific hyperparameters
    if args.phase == 1:
        peak_lr = cfg.learning_rate
        min_lr = cfg.min_lr
        warmup_steps = cfg.warmup_steps
        max_steps = cfg.max_steps_phase1
        train_bin = os.path.join(cfg.data_dir, cfg.train_bin)
        val_bin = os.path.join(cfg.data_dir, cfg.val_bin)
    else:  # phase 2
        peak_lr = cfg.learning_rate_phase2
        min_lr = cfg.min_lr_phase2
        warmup_steps = cfg.warmup_steps_phase2
        max_steps = cfg.max_steps_phase2
        train_bin = os.path.join(cfg.data_dir, "chat_train.bin")
        val_bin = os.path.join(cfg.data_dir, "chat_val.bin")

    print(f"\n=== Phase {args.phase} ===")
    print(f"  peak_lr:    {peak_lr}")
    print(f"  min_lr:     {min_lr}")
    print(f"  warmup:     {warmup_steps}")
    print(f"  max_steps:  {max_steps:,}")
    print(f"  train_bin:  {train_bin}")

    if not os.path.exists(train_bin):
        raise FileNotFoundError(
            f"Training data not found: {train_bin}\n"
            f"(Phase {args.phase} requires this file. "
            f"For Phase 2, build the chat dataset first.)"
        )

    # Build model
    model = TransformerLM(cfg).to(device)
    print(f"\n{model}")

    # Optimizer
    param_groups = make_param_groups(model, cfg.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=peak_lr,
        betas=(cfg.beta1, cfg.beta2),
    )

    # Data
    train_ds = TokenStreamDataset(bin_path=train_bin, seq_len=cfg.max_seq_len)
    val_ds = TokenStreamDataset(bin_path=val_bin, seq_len=cfg.max_seq_len)
    train_loader = make_dataloader(train_ds, batch_size=cfg.batch_size)
    val_loader = make_dataloader(val_ds, batch_size=cfg.batch_size)
    print(f"\nDatasets:")
    print(f"  train: {train_ds.total_tokens:,} tokens")
    print(f"  val:   {val_ds.total_tokens:,} tokens")

    # Resume if requested
    start_step = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        print(f"\nResuming from {args.resume}")
        meta = load_checkpoint(args.resume, model, optimizer, device=device)
        start_step = meta["step"]
        best_val_loss = meta["best_val_loss"]
        print(f"  resumed at step {start_step}, best_val_loss={best_val_loss:.4f}")

    return {
        "cfg": cfg,
        "device": device,
        "model": model,
        "optimizer": optimizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "phase": args.phase,
        "peak_lr": peak_lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "start_step": start_step,
        "best_val_loss": best_val_loss,
        "dtype": dtype_from_str(cfg.dtype),
    }


def train_loop(state: dict) -> None:
    """The main training loop."""
    cfg = state["cfg"]
    device = state["device"]
    model = state["model"]
    optimizer = state["optimizer"]
    train_loader = state["train_loader"]
    val_loader = state["val_loader"]
    phase = state["phase"]
    peak_lr = state["peak_lr"]
    min_lr = state["min_lr"]
    warmup_steps = state["warmup_steps"]
    max_steps = state["max_steps"]
    dtype = state["dtype"]

    start_step = state["start_step"]
    best_val_loss = state["best_val_loss"]
    ckpt_dir = cfg.checkpoint_dir

    # MLflow setup
    mlflow.set_experiment("tiny-transformer")
    mlflow.start_run(run_name=f"phase{phase}_step{start_step}")
    mlflow.log_params({
        "phase": phase,
        "peak_lr": peak_lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "effective_batch": cfg.effective_batch_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "weight_decay": cfg.weight_decay,
        "params": model.num_parameters(),
    })

    model.train()
    train_iter = iter(train_loader)
    start_time = time.time()
    step = start_step

    tokens_per_step = cfg.effective_batch_size * cfg.max_seq_len
    last_log_time = time.time()

    print(f"\n=== Starting training from step {step} ===")
    print(f"Time limit: {cfg.max_train_hours} hours\n")

    while step < max_steps:
        # Set LR for this step
        lr = get_lr(step, warmup_steps, max_steps, peak_lr, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(cfg.grad_accum_steps):
            try:
                input_ids, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, labels = next(train_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=dtype):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                loss = loss / cfg.grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        )

        # Optimizer step
        optimizer.step()
        step += 1

        # Logging
        if step % cfg.log_interval == 0:
            now = time.time()
            elapsed = now - last_log_time
            toks_per_sec = (tokens_per_step * cfg.log_interval) / elapsed
            last_log_time = now

            mlflow.log_metrics({
                "train/loss": accum_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item(),
                "train/tokens_per_sec": toks_per_sec,
            }, step=step)

            elapsed_total = (now - start_time) / 3600
            print(f"step {step:>6}/{max_steps} | loss {accum_loss:.4f} | "
                  f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                  f"{toks_per_sec:,.0f} tok/s | {elapsed_total:.2f}h")

        # Validation
        if step % cfg.val_interval == 0:
            val_metrics = evaluate(
                model, val_loader, device,
                n_batches=cfg.val_batches, dtype=dtype,
            )
            mlflow.log_metrics({
                "val/loss": val_metrics["loss"],
                "val/perplexity": val_metrics["perplexity"],
            }, step=step)
            print(f"  >>> val loss {val_metrics['loss']:.4f} | "
                  f"perplexity {val_metrics['perplexity']:.2f}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"),
                    step, phase, model, optimizer, best_val_loss, cfg,
                )
                print(f"  >>> new best! saved best.pt")

        # Periodic checkpoint
        if step % cfg.save_interval == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"step_{step}.pt"),
                step, phase, model, optimizer, best_val_loss, cfg,
            )
            save_checkpoint(
                os.path.join(ckpt_dir, "latest.pt"),
                step, phase, model, optimizer, best_val_loss, cfg,
            )
            prune_checkpoints(ckpt_dir, cfg.keep_last_k)

        # Time limit check (clean stop for resumable sessions)
        if (time.time() - start_time) > cfg.max_train_hours * 3600:
            print(f"\n=== Time limit ({cfg.max_train_hours}h) reached at step {step} ===")
            save_checkpoint(
                os.path.join(ckpt_dir, "latest.pt"),
                step, phase, model, optimizer, best_val_loss, cfg,
            )
            print(f"Saved latest.pt. Resume with: "
                  f"python train.py --phase {phase} --resume {ckpt_dir}/latest.pt")
            break
    else:
        # Loop completed naturally
        print(f"\n=== Training complete: reached max_steps ({max_steps}) ===")
        save_checkpoint(
            os.path.join(ckpt_dir, "latest.pt"),
            step, phase, model, optimizer, best_val_loss, cfg,
        )

    mlflow.end_run()
    print(f"\nFinal step: {step}, best val loss: {best_val_loss:.4f}")


def main():
    args = parse_args()
    state = setup_training(args)
    train_loop(state)


if __name__ == "__main__":
    main()


