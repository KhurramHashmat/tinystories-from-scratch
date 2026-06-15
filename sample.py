"""sample_with_metrics.py — Generate text with confidence metrics + interactive mode."""
import math
import torch
import torch.nn.functional as F
import mlflow

from src.config import TransformerConfig
from src.model import TransformerLM
from src.tokenizer import BPETokenizer


@torch.no_grad()
def generate_with_metrics(
    model, input_ids, max_new_tokens, temperature, top_p, eos_token_id,
):
    """Generate text and capture per-token confidence metrics."""
    model.eval()
    max_seq_len = model.config.max_seq_len

    chosen_probs = []
    entropies = []
    log_probs = []

    for _ in range(max_new_tokens):
        cropped = input_ids[:, -max_seq_len:] if input_ids.size(1) > max_seq_len else input_ids
        logits = model(cropped)[:, -1, :]

        full_probs = F.softmax(logits, dim=-1)
        entropy = -(full_probs * (full_probs + 1e-12).log()).sum(dim=-1).item()

        if temperature == 0.0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                to_remove = remove.scatter(-1, sorted_idx, remove)
                scaled = scaled.masked_fill(to_remove, float("-inf"))
            probs = F.softmax(scaled, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        chosen_prob = full_probs.gather(-1, next_token).item()
        log_prob = math.log(max(chosen_prob, 1e-12))

        chosen_probs.append(chosen_prob)
        entropies.append(entropy)
        log_probs.append(log_prob)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return input_ids, {
        "chosen_probs": chosen_probs,
        "entropies": entropies,
        "log_probs": log_probs,
    }


def compute_aggregate_metrics(metrics: dict) -> dict:
    """Roll per-token metrics up to summary statistics."""
    chosen = metrics["chosen_probs"]
    entropies = metrics["entropies"]
    log_probs = metrics["log_probs"]
    n = len(chosen)

    if n == 0:
        return {}

    return {
        "n_tokens": n,
        "avg_confidence": sum(chosen) / n,
        "min_confidence": min(chosen),
        "avg_entropy": sum(entropies) / n,
        "max_entropy": max(entropies),
        "avg_log_prob": sum(log_probs) / n,
        "perplexity": math.exp(-sum(log_probs) / n),
    }


def run_prompt(model, tok, device, prompt, temperature, top_p, max_tokens, mlflow_key=None):
    """Run a single prompt and print results. Optionally log to MLflow."""
    print(f"\n--- Prompt: {prompt!r} ---")
    print(f"  (temperature={temperature}, top_p={top_p}, max_tokens={max_tokens})")

    input_ids = torch.tensor([tok.encode(prompt)], device=device)

    output, raw_metrics = generate_with_metrics(
        model, input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
    )

    text = tok.decode(output[0].tolist())
    print(f"\n{text}")

    agg = compute_aggregate_metrics(raw_metrics)
    print(f"\n  📊 Metrics:")
    print(f"     Generated tokens:  {agg['n_tokens']}")
    print(f"     Avg confidence:    {agg['avg_confidence']:.3f}")
    print(f"     Min confidence:    {agg['min_confidence']:.3f}")
    print(f"     Avg entropy:       {agg['avg_entropy']:.3f}")
    print(f"     Gen perplexity:    {agg['perplexity']:.2f}")

    if mlflow_key is not None:
        for k, v in agg.items():
            mlflow.log_metric(f"{mlflow_key}/{k}", v)
        mlflow.log_text(text, f"{mlflow_key}_output.txt")

    return text, agg


def parse_inline_overrides(raw_input: str, default_temp: float, default_top_p: float):
    """
    Parse 'temp=0.3: prompt text' or 'temp=0.3, top_p=0.95: prompt text'.

    Returns (prompt, temperature, top_p).
    If no overrides found, returns (raw_input, default_temp, default_top_p).
    """
    if ":" not in raw_input:
        return raw_input, default_temp, default_top_p

    head, prompt = raw_input.split(":", 1)
    head = head.strip()
    prompt = prompt.strip()

    # The head must look like 'key=value' (optionally comma-separated)
    # to be treated as overrides. Otherwise it's part of the prompt.
    if "=" not in head:
        return raw_input, default_temp, default_top_p

    temperature = default_temp
    top_p = default_top_p

    for pair in head.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        try:
            if k == "temp" or k == "temperature":
                temperature = float(v)
            elif k == "top_p":
                top_p = float(v)
        except ValueError:
            print(f"  ⚠ Invalid value for {k}: {v!r} (using default)")

    return prompt, temperature, top_p


def get_user_settings():
    """Interactive settings menu."""
    print("\nGeneration settings (press Enter for defaults):")
    try:
        temp_in = input("  Temperature (default 0.8, range 0.0-1.5): ").strip()
        temperature = float(temp_in) if temp_in else 0.8

        top_p_in = input("  Top-p (default 0.9, range 0.0-1.0): ").strip()
        top_p = float(top_p_in) if top_p_in else 0.9

        max_in = input("  Max new tokens (default 200): ").strip()
        max_tokens = int(max_in) if max_in else 200

        return temperature, top_p, max_tokens
    except ValueError:
        print("  ⚠ Invalid input, using defaults")
        return 0.8, 0.9, 200


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TransformerConfig()

    # --- Load model ---
    print("Loading model...")
    model = TransformerLM(cfg).to(device)
    ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"  ✓ best.pt from step {ckpt['step']} (val loss: {ckpt['best_val_loss']:.4f})")

    tok = BPETokenizer.from_dir(cfg.tokenizer_dir)
    print(f"  ✓ {tok}")

    # --- Start MLflow ---
    mlflow.set_experiment("tiny-transformer-inference")
    with mlflow.start_run(run_name=f"sampling_step{ckpt['step']}"):
        mlflow.log_params({
            "checkpoint_step": ckpt["step"],
            "training_val_loss": ckpt["best_val_loss"],
        })

        # --- Preset prompts ---
        print("\n" + "=" * 60)
        print("PRESET PROMPTS")
        print("=" * 60)
        presets = [
            "Once upon a time, there was a little girl named Lily",
            "Tom and his dog went to the park",
            "The big red ball rolled",
        ]
        for i, prompt in enumerate(presets):
            run_prompt(model, tok, device, prompt,
                       temperature=0.8, top_p=0.9, max_tokens=200,
                       mlflow_key=f"preset_{i}")

        # --- Interactive mode ---
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Commands:")
        print("  settings  — change default temperature/top_p/max_tokens")
        print("  show      — display current defaults")
        print("  quit      — exit")
        print()
        print("Inline overrides (one-time, just for that prompt):")
        print("  temp=0.3: Once upon a time")
        print("  temp=1.2, top_p=0.95: The wizard waved")
        print()

        temperature = 0.8
        top_p = 0.9
        max_tokens = 200
        user_prompt_count = 0

        print(f"Current defaults: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")

        while True:
            try:
                print()
                raw = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not raw:
                continue

            cmd = raw.lower()

            if cmd in ("quit", "exit", "q"):
                print("Exiting.")
                break

            if cmd == "settings":
                temperature, top_p, max_tokens = get_user_settings()
                print(f"  → updated defaults: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
                continue

            if cmd == "show":
                print(f"  Current defaults: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
                continue

            # Parse possible inline overrides
            actual_prompt, prompt_temp, prompt_top_p = parse_inline_overrides(
                raw, default_temp=temperature, default_top_p=top_p,
            )

            if not actual_prompt:
                print("  ⚠ Empty prompt after parsing overrides.")
                continue

            run_prompt(model, tok, device, actual_prompt,
                       temperature=prompt_temp, top_p=prompt_top_p, max_tokens=max_tokens,
                       mlflow_key=f"user_{user_prompt_count}")
            user_prompt_count += 1

        total = len(presets) + user_prompt_count
        mlflow.log_metric("total_prompts", total)
        print(f"\n✓ Session complete. Generated {total} samples ({user_prompt_count} interactive).")
        print(f"  View in MLflow: experiment 'tiny-transformer-inference'")


if __name__ == "__main__":
    main()