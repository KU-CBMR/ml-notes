+++
title = "Transformer From Scratch in Python: Source Code Walkthrough (PyTorch)"
date = 2026-02-10
draft = false
tags = ["transformer", "pytorch", "attention", "deep-learning", "source-code"]
categories = ["Deep Learning"]
description = "A readable Transformer implementation in Python (PyTorch), with a line-by-line walkthrough of Multi-Head Attention, positional encoding, and Transformer blocks."
+++

This post walks through a _minimal but complete_ Transformer implementation in **Python (PyTorch)** and explains how each part works.

The goal is clarity: you should be able to copy this into a single file, run it, and recognize every tensor shape.

---

## What we’re building

A standard Transformer block (encoder-style) is basically:

1. **Token embedding** + **positional encoding**
2. **Multi-Head Self-Attention (MHA)**
3. **Feed-Forward Network (FFN)**
4. **Residual connections + LayerNorm**
5. (Optionally) **dropout** and **attention masking**

We’ll implement:

- `PositionalEncoding` (sin/cos)
- `MultiHeadAttention` (scaled dot-product attention)
- `TransformerBlock` (MHA + FFN with residuals)
- `TinyTransformerEncoder` (stack of blocks)

---

## Full source code (single file)

> Requirements: `pip install torch`

```python
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding from "Attention Is All You Need".
    Adds a deterministic position signal to token embeddings.

    Input:  x  -> (B, T, D)
    Output: x + pe[:T] -> (B, T, D)
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                 # (max_len, D)
        position = torch.arange(0, max_len).unsqueeze(1)   # (max_len, 1)

        # Compute the div_term (frequency terms)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)       # even dims
        pe[:, 1::2] = torch.cos(position * div_term)       # odd dims

        self.register_buffer("pe", pe)  # buffer: saved with model, not trained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x + self.pe[:T].unsqueeze(0)                # (1, T, D) broadcast to (B, T, D)


# -----------------------------
# 2) Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention (or cross-attention if you pass different kv).

    Shapes:
      x_q: (B, Tq, D)
      x_kv: (B, Tk, D)  (optional; if None, self-attention uses x_q)

    Returns:
      out: (B, Tq, D)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Project input to Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Project concatenated heads back to model dim
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, T, D) -> (B, H, T, Hd)
        """
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim)      # (B, T, H, Hd)
        return x.transpose(1, 2)                           # (B, H, T, Hd)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, H, T, Hd) -> (B, T, D)
        """
        B, H, T, Hd = x.shape
        x = x.transpose(1, 2).contiguous()                 # (B, T, H, Hd)
        return x.view(B, T, H * Hd)                        # (B, T, D)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        attn_mask: optional boolean mask, shape broadcastable to (B, H, Tq, Tk)
                   True means "keep", False means "mask out".
        """
        if x_kv is None:
            x_kv = x_q

        # 1) Linear projections
        Q = self.q_proj(x_q)  # (B, Tq, D)
        K = self.k_proj(x_kv) # (B, Tk, D)
        V = self.v_proj(x_kv) # (B, Tk, D)

        # 2) Split into heads
        Q = self._split_heads(Q)  # (B, H, Tq, Hd)
        K = self._split_heads(K)  # (B, H, Tk, Hd)
        V = self._split_heads(V)  # (B, H, Tk, Hd)

        # 3) Scaled dot-product attention
        # scores = QK^T / sqrt(Hd)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, Tq, Tk)

        # 4) Apply mask (if any)
        if attn_mask is not None:
            # Masked positions get -inf so softmax -> 0 probability
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        # 5) Softmax -> attention weights
        attn = F.softmax(scores, dim=-1)                   # (B, H, Tq, Tk)
        attn = self.dropout(attn)

        # 6) Weighted sum of values
        ctx = torch.matmul(attn, V)                        # (B, H, Tq, Hd)

        # 7) Merge heads + final projection
        ctx = self._merge_heads(ctx)                       # (B, Tq, D)
        out = self.out_proj(ctx)                           # (B, Tq, D)
        return out


# -----------------------------
# 3) Feed-Forward Network (FFN)
# -----------------------------
class FeedForward(nn.Module):
    """
    Position-wise MLP:
      (B, T, D) -> (B, T, D_ff) -> (B, T, D)

    Many implementations use GELU.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# -----------------------------
# 4) Transformer Block (Encoder-style)
# -----------------------------
class TransformerBlock(nn.Module):
    """
    A minimal Encoder block:

      x -> LN -> MHA -> +residual
        -> LN -> FFN -> +residual
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN style (common in modern models)
        h = self.ln1(x)
        h = self.mha(h, attn_mask=attn_mask)
        x = x + self.drop(h)

        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.drop(h)
        return x


# -----------------------------
# 5) Tiny Transformer Encoder
# -----------------------------
@dataclass
class TinyConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 4
    max_len: int = 512
    dropout: float = 0.1


class TinyTransformerEncoder(nn.Module):
    """
    Turns token ids into contextualized representations.

    Input:
      token_ids: (B, T) int64

    Output:
      h: (B, T, D)
    """
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, dropout=cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        padding_mask: optional boolean mask (B, T), True for real tokens, False for padding.
        We'll convert it to (B, 1, 1, T) so it can broadcast over heads and query positions.
        """
        x = self.tok_emb(token_ids)                        # (B, T, D)
        x = self.pos_enc(x)                                # (B, T, D)
        x = self.drop(x)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask[:, None, None, :]     # (B, 1, 1, T)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        return self.ln_f(x)


# -----------------------------
# 6) Quick sanity test
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = TinyConfig(vocab_size=1000, d_model=128, n_heads=8, d_ff=512, n_layers=2, max_len=64, dropout=0.0)
    model = TinyTransformerEncoder(cfg)

    B, T = 2, 10
    token_ids = torch.randint(0, cfg.vocab_size, (B, T))
    padding_mask = torch.ones(B, T, dtype=torch.bool)
    padding_mask[0, -2:] = False  # pretend last 2 are padding for sample 0

    h = model(token_ids, padding_mask=padding_mask)
    print("output:", h.shape)  # (B, T, D)
```

---

## Walkthrough: how the tensors flow

Let:

- `B` = batch size
- `T` = sequence length
- `D` = `d_model`
- `H` = number of heads
- `Hd = D / H` = per-head dimension

### 1) Token embedding + positional encoding

- `token_ids` is `(B, T)` integers.
- `nn.Embedding` returns `(B, T, D)`.
- Positional encoding adds `(1, T, D)` (broadcast over batch).

So after this stage:

- `x` is `(B, T, D)`.

### 2) Multi-Head Attention (MHA)

We build queries/keys/values:

- `Q = W_q x` -> `(B, Tq, D)`
- `K = W_k x` -> `(B, Tk, D)`
- `V = W_v x` -> `(B, Tk, D)`

In self-attention, `Tq = Tk = T`.

Then we split heads:

- `(B, T, D)` -> `(B, H, T, Hd)`

Now attention scores:

- `scores = Q @ K^T / sqrt(Hd)`  
  `(B, H, T, Hd) @ (B, H, Hd, T)` -> `(B, H, T, T)`

Softmax along the last dimension gives attention weights per query token over all key tokens:

- `attn = softmax(scores, dim=-1)` -> `(B, H, T, T)`

Context (weighted sum of values):

- `ctx = attn @ V`  
  `(B, H, T, T) @ (B, H, T, Hd)` -> `(B, H, T, Hd)`

Merge heads:

- `(B, H, T, Hd)` -> `(B, T, D)`

Final output projection:

- `(B, T, D)` -> `(B, T, D)`

### 3) Masking: ignoring padding tokens

If your batch has padding, you typically don’t want other tokens to attend to padding. We pass a boolean mask:

- `padding_mask`: `(B, T)` with **True** for real tokens, **False** for padding

We reshape it to:

- `(B, 1, 1, T)` so it broadcasts across heads and query positions.

Then we do:

```python
scores = scores.masked_fill(~attn_mask, -inf)
```

Meaning: wherever the mask is False, set the score to `-inf`, so after softmax, attention weight becomes ~0.

### 4) Residual + LayerNorm

The Transformer’s stability comes from:

- **Residual connections**: `x = x + f(x)`
- **LayerNorm**: keeps activations well-scaled

This code uses **Pre-LN** blocks:

```python
h = LN(x)
x = x + MHA(h)
h = LN(x)
x = x + FFN(h)
```

Pre-LN often trains more stably for deeper stacks.

### 5) Feed-Forward Network (FFN)

This is a position-wise MLP applied to every token independently:

- `D -> D_ff -> D`

You can think of attention as “mix tokens” and FFN as “mix channels”.

---

## Common gotchas when reading Transformer code

1. **Transpose and view**: head splitting/merging is mostly `view + transpose`.
2. **Mask conventions**: some code uses `1/0`, others use `True/False`. Always check what “masked” means.
3. **Pre-LN vs Post-LN**: both exist; Pre-LN is common in modern LLMs.
4. **Cross-attention**: same module, but keys/values come from a different sequence (e.g., encoder outputs).

---

## Small extension ideas (turn this into an LLM)

- Add a **causal mask** (lower-triangular) for decoder-only language modeling
- Add an output head: `lm_logits = Linear(D, vocab_size)`
- Implement **KV cache** for fast autoregressive decoding
- Swap sinusoidal positions for **learned** or **RoPE** positions
