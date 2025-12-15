import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    """Rotary Position Embedding."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        # Shape: (seq_len, dim/2) -> (seq_len, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    # q, k: (batch, num_heads, seq_len, head_dim)
    # cos, sin: (seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(seq_len)
        q, k = apply_rope(q, k, cos, sin)

        # Convert boolean mask to float mask with dtype-safe -inf
        # True -> 0.0 (attend), False -> -inf (mask out)
        float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        float_mask.masked_fill_(~attn_mask, torch.finfo(q.dtype).min)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, d_model: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 4

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(xW1) * (xW3)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 512):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, max_seq_len)
        self.feed_forward = FeedForward(d_model)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x), attn_mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class MathTransformer(nn.Module):
    """
    Decoder-only transformer for math operations.

    Architecture (from specification):
    - 16 layers
    - d_model: 512
    - 4 attention heads
    - context: 512
    - RoPE positional encoding
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 16,
        num_heads: int = 4,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, max_seq_len) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between embedding and output
        self.output.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

        # Create causal mask (1, 1, S, S) - True means attend
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token indices of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # Token embedding (no positional embedding - using RoPE)
        h = self.token_embedding(x)

        # Causal mask for current sequence length: (1, 1, S, S)
        causal = self.causal_mask[:, :, :seq_len, :seq_len]

        # Auto-build pad_mask from input: (B, S)
        pad_mask = x != self.pad_token_id

        # pad_mask: (B, S) -> (B, 1, 1, S) for key positions
        # Combined mask: causal AND pad (only attend to non-PAD positions in the past)
        pad_mask_expanded = pad_mask.view(batch_size, 1, 1, seq_len)
        attn_mask = causal & pad_mask_expanded  # (B, 1, S, S)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, attn_mask)

        h = self.norm(h)
        logits = self.output(h)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively. Stops when all sequences emit EOS.

        Args:
            idx: Input token indices of shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token indices of shape (batch_size, seq_len + generated).
            Sequences are padded with pad_token_id after EOS.
        """
        batch_size = idx.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]

            # Get logits
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Replace with PAD for finished sequences
            idx_next = torch.where(
                finished.unsqueeze(1),
                torch.full_like(idx_next, self.pad_token_id),
                idx_next,
            )
            idx = torch.cat([idx, idx_next], dim=1)

            # Update finished status
            finished = finished | (idx_next.squeeze(1) == self.eos_token_id)
            if finished.all():
                break

        return idx
