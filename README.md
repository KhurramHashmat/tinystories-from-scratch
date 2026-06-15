# TinyStories from Scratch

A 137M parameter LLaMA-style decoder-only transformer, trained from scratch on TinyStories — no HuggingFace `transformers`, no `tokenizers`. Every layer, the tokenizer, the dataset pipeline, and the training loop implemented from PyTorch primitives.

Trained for 5,000 steps on a single RTX 4060 Laptop GPU (8GB) in ~5 hours. Final validation perplexity: **3.34**.

## What It Does

Given any prompt, generates a short children's story in the TinyStories style:

> **Prompt:** *"Once upon a time, there was a little girl named Lily"*
>
> Once upon a time, there was a little girl named Lily. She loved to play with her toys and draw pictures. One day, she found a big box in her room. Lily was very excited and opened the box. Inside the box, there were many colorful pictures.
>
> Lily showed the pictures to her friend, Tom. Tom said, "Wow! I like the pictures too!" They decided to draw pictures together. Lily drew a big sun, and Tom drew a funny cat. They laughed and had a lot of fun. [...]

See [`samples/examples.md`](samples/examples.md) for more outputs across in-domain, out-of-domain, and stress-test prompts, with confidence and perplexity metrics for each.

## Architecture

Modern LLaMA-style decoder-only transformer:

- **RMSNorm** with fp32-cast for numerical stability
- **Rotary Position Embeddings (RoPE)** with precomputed cos/sin cache
- **Multi-head causal self-attention** using `F.scaled_dot_product_attention` (FlashAttention when available)
- **SwiGLU feed-forward** (3 projections: gate, up, down)
- **Pre-norm** residual connections
- **Weight tying** between embedding and LM head
- **GPT-2 style initialization** (std=0.02)

| Hyperparameter | Value |
|---|---|
| Parameters | 137.8M |
| d_model | 768 |
| n_layers | 12 |
| n_heads | 12 |
| d_ff | 3,072 |
| max_seq_len | 512 |
| vocab_size | 32,000 |
| Precision | bfloat16 (mixed precision via autocast) |

## Training

| Setting | Value |
|---|---|
| Effective batch size | 128 (8 micro-batches × 16 grad accum) |
| Peak learning rate | 3e-4 |
| Schedule | Cosine decay with 200-step warmup |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Weight decay | 0.1 (on 2D+ tensors only; norms and biases excluded) |
| Gradient clipping | 1.0 |
| Steps | 5,000 (stopped at diminishing returns) |
| Throughput | ~17,800 tokens/sec on RTX 4060 |

## Results

| Metric | Value |
|---|---|
| Final train loss | 1.20 |
| Final val loss | 1.20 |
| Final val perplexity | **3.34** |
| Train/val gap | 0.005 (no overfitting) |

The model converged cleanly with train and val losses nearly identical throughout — it genuinely learned the distribution rather than memorizing.

## Project Structure

```
tinystories-from-scratch/
├── src/
│   ├── config.py        # All hyperparameters (single source of truth)
│   ├── tokenizer.py     # Byte-level BPE encoder/decoder
│   ├── model.py         # Full transformer (RMSNorm, RoPE, SwiGLU, etc.)
│   └── dataset.py       # Memory-mapped binary loader
│
├── tokenizer/           # Trained 32K vocabulary
│   ├── vocab.json
│   ├── merges.txt
│   └── special_tokens.json
│
├── samples/
│   └── examples.md      # Generated outputs with metrics
│
├── train_tokenizer.py   # One-time: trains BPE vocab
├── pretokenize.py       # One-time: text corpus → uint16 binary
├── train.py             # Training loop (resumable, MLflow logging)
├── sample.py            # Interactive generation with metrics
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/KhurramHashmat/tinystories-from-scratch.git
cd tinystories-from-scratch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
# .venv\Scripts\activate      # Windows

# Install PyTorch first (choose appropriate variant)
# CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
# Or CPU-only:
# pip install torch

# Install remaining dependencies
pip install -r requirements.txt
```

## Reproducing the Training

```bash
# 1. Download TinyStories dataset
# Place TinyStoriesV2-GPT4-train.txt and TinyStoriesV2-GPT4-valid.txt in data/

# 2. Train the BPE tokenizer (~30 minutes)
python -c "from train_tokenizer import main; main()"

# 3. Pre-tokenize the corpus to binary (~8 minutes)
python pretokenize.py

# 4. Train the model (~5 hours on RTX 4060)
python train.py --phase 1

# 5. (Optional) Resume from a checkpoint
python train.py --phase 1 --resume checkpoints/latest.pt
```

Training auto-stops at the time limit set in `src/config.py` (default 6 hours) and saves a resumable checkpoint.

## Using the Trained Model

The trained checkpoint is too large for GitHub (~550MB). It will be hosted on Hugging Face — link coming soon.

Once downloaded:

```bash
# Place best.pt in checkpoints/
python sample.py
```

This launches an interactive generation session with confidence metrics:

```
Loading model...
  ✓ best.pt from step 5000 (val loss: 1.2048)
  ✓ BPETokenizer(vocab_size=32000, ...)

PRESET PROMPTS
--- Prompt: 'Once upon a time, there was a little girl named Lily' ---
[generates a story]
  📊 Metrics:
     Generated tokens:  141
     Avg confidence:    0.680
     Avg entropy:       0.959
     Gen perplexity:    1.74

INTERACTIVE MODE
>>> The little dragon
[generates]

>>> temp=0.3: The little dragon
[same prompt at lower temperature — more conservative]

>>> settings
[change defaults]

>>> quit
```

Per-generation metrics are also logged to MLflow (`tiny-transformer-inference` experiment) for later analysis.

## What This Model Can and Can't Do

**What it can do well:**

- Generate coherent multi-character short stories in the TinyStories style
- Handle dialogue with proper quoting and attribution
- Produce grammatically correct English
- Stop naturally at story boundaries (learned to emit `<|endoftext|>`)
- Signal its own uncertainty — confidence drops on out-of-domain prompts

**What it can't do:**

- Answer factual questions ("the capital of France is..." → becomes a story)
- Reason about technical or specialized topics
- Hold information across long contexts beyond ~500 tokens
- Generate non-English text (tokenizer supports it, but no training data)

The model learned exactly what was in its training data: simple stories with a small vocabulary, a narrow set of characters (Lily, Tom, dogs, cats, princesses), and predictable narrative patterns. Out-of-domain prompts get silently transformed into the in-domain register — a CEO becomes someone's mother, a Python programmer becomes a child dancing.

The architecture and training pipeline can handle any text data. The model's limits are a function of TinyStories, not the implementation.

## Hardware

Trained entirely on a single laptop:

- NVIDIA RTX 4060 Laptop GPU, 8GB VRAM
- bfloat16 mixed precision via `torch.autocast`
- Peak memory: ~5.5 GB
- Throughput: 17,800 tokens/sec
- Total wall-clock training time: 5.14 hours

## Acknowledgments

- **TinyStories** dataset and methodology: [Eldan & Li (2023)](https://arxiv.org/abs/2305.07759)
- **LLaMA** architecture: [Touvron et al. (2023)](https://arxiv.org/abs/2302.13971)
- **RoPE**: [Su et al. (2021)](https://arxiv.org/abs/2104.09864)
- **SwiGLU**: [Shazeer (2020)](https://arxiv.org/abs/2002.05202)
- **Byte-level BPE**: [GPT-2 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## License

MIT — see [LICENSE](LICENSE).