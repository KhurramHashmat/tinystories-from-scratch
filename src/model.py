"""
src/model.py

Transformer model architecture, built from scratch.

Currently implemented:
    - Part 1: RMSNorm
    - Part 2: RoPE (Rotary Positional Embeddings)
    - Part 3: Multi-Head Attention
    - Part 4: FeedForward (SwiGLU)
    - Part 5: TransformerBlock
    - Part 6: Full Model
    - Part 7: generate()
    
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from src.config import TransformerConfig

# RMSNorm 
class RMSNorm(nn.Module):

    """
    Root Mean Square Layer Normalization.

    Normalizes a vector by its RMS magnitude and applies a learned per-dim gain.

    Formula:
        rms(x) = sqrt(mean(x²) + eps)
        out    = (x / rms(x)) * gamma

    Args:
        dim: Size of the dimension to normalize (typically d_model).
        eps: Small constant for numerical stability. 1e-6 is standard.
    """

    def __init__(self, dim: int, eps: float = 1e-6): 
        super().__init__()
        self.eps = eps 
        # `gamma` is the learnable per-dim gain. Initialized to ones,
        # so RMSNorm is the identity-times-normalize at the start of training.
        # nn.Parameter registers this as a learnable parameter PyTorch will track.
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor: 

        """
        Args:
            x: Input tensor of shape (..., dim). The last dim is normalized.

        Returns:
            Same shape as input.
        """
        # Cast to float32 for the RMS computation to avoid bfloat16 overflow.
        # The actual scaling happens back in the original dtype.
        x_fp32 = x.float() 

        # Mean of squares along the last dimension.
        # keepdim=True so we can broadcast against x's shape.
        mean_sq = x_fp32.pow(2).mean(dim=-1, keepdim=True) 

        # rsqrt(mean_sq + eps) = 1 / sqrt(mean_sq + eps)
        # Slightly faster than computing sqrt then dividing.
        inv_rms = torch.rsqrt(mean_sq + self.eps)

        # Normalize (still in fp32) then cast back to input dtype.
        x_normed = (x_fp32 * inv_rms).to(x.dtype)

        # Apply learned gain.
        return x_normed * self.gamma 
    

# RoPE (Rotary Positional Embeddings)
def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    base: int = 10_000,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin rotation tables for RoPE.

    Args:
        head_dim: Dimension per attention head (must be even).
        max_seq_len: Maximum positions to support.
        base: RoPE θ base. 10,000 is the original RoPE paper value.
        device: Where to place the tensors.

    Returns:
        cos_cache, sin_cache — each of shape (max_seq_len, head_dim).
        Cached values are FP32 for precision.
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    # Step 1: compute the per-pair frequencies θ_i
    #   θ_i = 1 / base^(2i / head_dim)   for i = 0, 1, ..., head_dim/2 - 1
    #
    # arange(0, head_dim, 2) gives [0, 2, 4, ..., head_dim-2]
    # dividing by head_dim gives [0/d, 2/d, 4/d, ...]
    # that's 2i/d for i in [0, 1, 2, ...]
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    # inv_freq shape: (head_dim/2,)

    # Step 2: positions 0, 1, 2, ..., max_seq_len - 1
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    # positions shape: (max_seq_len,)

    # Step 3: outer product → angles for every (position, pair) combination
    #   angles[p, i] = positions[p] × inv_freq[i] = p × θ_i
    angles = torch.outer(positions, inv_freq)
    # angles shape: (max_seq_len, head_dim/2)

    # Step 4: duplicate each pair so cos/sin have full head_dim
    # We use the half-split convention:
    #   first half of head_dim uses angles[..., 0:head_dim/2]
    #   second half uses the SAME angles (duplicated)
    # This is what enables the LLaMA-style rotation in apply_rope.
    angles_full = torch.cat([angles, angles], dim=-1)
    # angles_full shape: (max_seq_len, head_dim)

    cos_cache = angles_full.cos()
    sin_cache = angles_full.sin()

    if device is not None:
        cos_cache = cos_cache.to(device)
        sin_cache = sin_cache.to(device)

    return cos_cache, sin_cache

def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding to a tensor.

    Uses the LLaMA-style half-split rotation:
        x1 = x[..., :head_dim/2]
        x2 = x[..., head_dim/2:]
        rotated = concat([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos])

    Args:
        x: Tensor of shape (..., seq_len, head_dim). The last dim is rotated.
        cos: Precomputed cos cache, shape (seq_len, head_dim).
        sin: Precomputed sin cache, shape (seq_len, head_dim).

    Returns:
        Rotated tensor of the same shape and dtype as x.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2

    # Split last dim in half
    x1 = x[..., :half]
    x2 = x[..., half:]

    # The cached cos/sin are duplicated across both halves.
    # We only need the first half of each (the second half is identical).
    cos_half = cos[..., :half]  # shape: (seq_len, head_dim/2)
    sin_half = sin[..., :half]

    # Rotate using the half-split formula
    # First half: x1·cos - x2·sin
    # Second half: x1·sin + x2·cos
    rotated_x1 = x1 * cos_half - x2 * sin_half
    rotated_x2 = x1 * sin_half + x2 * cos_half

    out = torch.cat([rotated_x1, rotated_x2], dim=-1)
    return out.type_as(x)


# Multi-Head Attention 
class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE position encoding.

    Uses separate Q, K, V projections (modern LLM convention) and
    torch's SDPA for the attention computation, which auto-selects
    FlashAttention when available.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_base: int = 10_000,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: Total embedding dimension (e.g. 768).
            n_heads: Number of parallel attention heads (e.g. 12).
            max_seq_len: Maximum sequence length the cache supports.
            rope_base: RoPE θ base. 10,000 is the standard value.
            dropout: Attention dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # Separate Q, K, V projections.
        # No bias — modern LLM convention (LLaMA, Mistral).
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection (after concatenating all heads).
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        # Precompute RoPE cache once. Buffers move with .to(device)
        # but are NOT learnable parameters.
        cos, sin = precompute_rope_cache(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Step 1: project x into Q, K, V — three separate linears
        # Each produces shape (batch, seq_len, d_model)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # Step 2: reshape for multi-head attention
        # (batch, seq_len, d_model) → (batch, seq_len, n_heads, head_dim)
        # → (batch, n_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Step 3: apply RoPE to Q and K (NOT V)
        # Slice cache to current seq_len (allows seq_len < max_seq_len).
        cos = self.cos_cache[:seq_len]
        sin = self.sin_cache[:seq_len]
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        # Step 4: scaled dot-product attention with causal mask
        # SDPA auto-selects FlashAttention or memory-efficient backend.
        # is_causal=True applies the upper-triangular mask internally.
        # Dropout is zeroed at inference.
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        # out shape: (batch, n_heads, seq_len, head_dim)

        # Step 5: undo the head split
        # (batch, n_heads, seq_len, head_dim) → (batch, seq_len, n_heads, head_dim)
        # → (batch, seq_len, d_model)
        # .contiguous() needed because transpose produces non-contiguous memory.
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # Step 6: output projection
        out = self.w_out(out)

        return out
    

# FeedForward (SwiGLU)
class FeedForward(nn.Module):
    """
    SwiGLU feed-forward network.

    Three linear projections instead of the standard two:
        gate  = SiLU(W_gate · x)
        value = W_up · x
        out   = W_down · (gate ⊙ value)

    The multiplicative gate gives the model learned, per-dimension control
    over what flows through. Used by LLaMA, Mistral, Qwen, etc.
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: Input/output dimension.
            d_ff: Hidden dimension (typically 4 × d_model for standard FFN,
                  or (8/3) × d_model for LLaMA-style SwiGLU).
        """
        super().__init__()

        # Two parallel projections up to d_ff:
        # one for the gate, one for the value.
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)

        # Single projection back down to d_model.
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (..., d_model).

        Returns:
            Output of shape (..., d_model).
        """
        # Compute gate and value in parallel.
        # F.silu is the SiLU activation, also known as Swish:
        #     SiLU(x) = x * sigmoid(x)
        # Smooth, non-monotonic, near-identity for positive values.
        gate = F.silu(self.w_gate(x))
        value = self.w_up(x)

        # Elementwise multiplication: gate controls per-dim flow of value.
        hidden = gate * value

        # Project back down to d_model.
        return self.w_down(hidden)
    

# TransformerBlock
class TransformerBlock(nn.Module):
    """
    A single transformer block: pre-norm attention + pre-norm FFN,
    each wrapped in a residual connection.

    Structure:
        x = x + Attention(RMSNorm(x))
        x = x + FeedForward(RMSNorm(x))

    Input and output shapes are identical: (batch, seq_len, d_model).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_base: int = 10_000,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        """
        Args:
            d_model: Model embedding dimension.
            n_heads: Number of attention heads.
            d_ff: FFN hidden dimension.
            max_seq_len: Maximum sequence length (for RoPE cache).
            rope_base: RoPE θ base. 10,000 standard.
            dropout: Attention dropout probability.
            norm_eps: RMSNorm ε.
        """
        super().__init__()

        # Pre-norm for the attention sublayer
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
            dropout=dropout,
        )

        # Pre-norm for the FFN sublayer
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, d_model).

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        # Sublayer 1: pre-norm attention + residual
        x = x + self.attn(self.attn_norm(x))

        # Sublayer 2: pre-norm FFN + residual
        x = x + self.ffn(self.ffn_norm(x))

        return x
    

# Full Model
class TransformerLM(nn.Module):
    """
    Decoder-only transformer language model.

    Architecture (modern LLaMA-style):
        - Token embedding
        - N transformer blocks (pre-norm, RoPE, SwiGLU)
        - Final RMSNorm
        - LM head (weight-tied to embedding)

    forward(input_ids) returns logits of shape (batch, seq_len, vocab_size).
    Loss is computed externally (in the training loop).
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embedding: (vocab_size, d_model)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of N transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                rope_base=config.rope_base,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Final normalization before LM head
        self.final_norm = RMSNorm(config.d_model)

        # LM head: projects (d_model) → (vocab_size)
        # bias=False to enable clean weight tying with embedding
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying — share the embedding matrix with the LM head.
        # After this assignment, both layers literally use the same tensor.
        # Saves ~24.5M parameters for our config.
        self.lm_head.weight = self.embedding.weight

        # Apply GPT-2 style initialization to all submodules
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights using the GPT-2 standard:
            - Linear and Embedding: N(0, 0.02²)
            - Biases: zero (we mostly have bias=False anyway)
            - RMSNorm: gamma is already 1.0 from RMSNorm.__init__
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        # Embed tokens: (batch, seq_len) → (batch, seq_len, d_model)
        x = self.embedding(input_ids)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary: (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Count parameters, accounting for weight tying (shared tensor counted once)."""
        if only_trainable:
            params = [p for p in self.parameters() if p.requires_grad]
        else:
            params = list(self.parameters())

        # Deduplicate by tensor identity (weight tying shares one tensor)
        seen = set()
        total = 0
        for p in params:
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    def __repr__(self) -> str:
        n_params = self.num_parameters()
        return (
            f"TransformerLM(\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  d_model={self.config.d_model},\n"
            f"  n_layers={self.config.n_layers},\n"
            f"  n_heads={self.config.n_heads},\n"
            f"  d_ff={self.config.d_ff},\n"
            f"  max_seq_len={self.config.max_seq_len},\n"
            f"  parameters={n_params:,} ({n_params / 1e6:.1f}M)\n"
            f")"
        )

    # Generation method with temperature and top-p sampling, plus optional EOS stopping.
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens.

        Args:
            input_ids: Starting prompt, shape (batch, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
                0.0 → greedy (argmax, deterministic).
                1.0 → original distribution.
                <1 → sharper, more confident.
                >1 → flatter, more random.
            top_p: Nucleus sampling threshold (0 < top_p ≤ 1).
                1.0 → no filtering (full distribution).
                0.9 → keep smallest set with cumulative prob ≥ 0.9.
            eos_token_id: If set, stop early when ALL sequences in the batch
                            have generated this token.

        Returns:
            Tensor of shape (batch, seq_len + new_tokens). Includes the prompt.
        """
        self.eval()   # ensure dropout is disabled

        max_seq_len = self.config.max_seq_len
        device = input_ids.device

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if input is too long
            # (sliding window — keep the most recent tokens)
            if input_ids.size(1) > max_seq_len:
                input_ids_cropped = input_ids[:, -max_seq_len:]
            else:
                input_ids_cropped = input_ids

            # Forward pass to get logits
            logits = self(input_ids_cropped)
            # logits shape: (batch, seq_len, vocab_size)

            # We only care about predictions for the LAST position
            # (predicting what comes after the current end-of-sequence)
            next_token_logits = logits[:, -1, :]   # shape: (batch, vocab_size)

            # Branch on sampling strategy
            if temperature == 0.0:
                # Greedy: pick the highest-probability token
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                # shape: (batch, 1)
            else:
                # Apply temperature
                scaled_logits = next_token_logits / temperature

                # Optional top-p (nucleus) filtering
                if top_p < 1.0:
                    scaled_logits = self._top_p_filter(scaled_logits, top_p)

                # Convert to probabilities and sample
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # shape: (batch, 1)

            # Append the new token to the running sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Early stop if every batch element has emitted EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        return input_ids

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Apply top-p (nucleus) filtering to logits.

        Tokens whose cumulative probability exceeds top_p are zeroed out
        (set to -inf so softmax gives them probability 0).

        Args:
            logits: Shape (batch, vocab_size).
            top_p: Cumulative probability threshold.

        Returns:
            Filtered logits of same shape, with low-probability tokens set to -inf.
        """
        # Sort logits in descending order along vocab dim
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Mask: True where we should DROP this token
        # (cumulative prob has already exceeded top_p before reaching this token)
        # Shift right by 1 so we always keep at least one token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter the mask back to original vocab positions
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )

        # Set removed positions to -infinity (softmax → 0)
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits