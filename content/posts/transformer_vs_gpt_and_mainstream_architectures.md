+++
title = "Transformer vs GPT: What’s the Difference? A Practical Architecture Guide (with Code)"
date = 2026-02-11
draft = false
tags = ["transformer", "gpt", "bert", "t5", "architecture", "attention", "pytorch"]
categories = ["ML Systems"]
description = "Clear differences between Transformers and GPT-style models, plus a guided tour of major Transformer architectures with small PyTorch examples."
+++

This post clarifies a common confusion:

- **Transformer** is a _model family / building block_ (attention + MLP + residuals + normalization).
- **GPT** is a _specific Transformer configuration_ (decoder-only, causal attention) trained with an autoregressive objective.

In other words: **GPT is a Transformer**, but not all Transformers are GPT.

---

## 1) The Transformer block (what stays the same)

Most modern Transformer variants repeat a layer that looks like:

1. Normalization (LayerNorm / RMSNorm)
2. **Multi-Head Attention**
3. Residual add
4. Normalization
5. **MLP / Feedforward** (GELU or SwiGLU variants are common)
6. Residual add

Key variations include:

- **Pre-norm** vs **post-norm** (modern large LMs typically use pre-norm for training stability)
- MLP type (GELU vs **SwiGLU**)
- Positional encoding (absolute, sinusoidal, **RoPE**, ALiBi, etc.)
- Attention implementation (standard, FlashAttention kernels, sparse patterns, etc.)

---

## 2) GPT vs other Transformers: the practical differences

### A) Attention mask (direction of information flow)

- **GPT-style (decoder-only)** uses **causal self-attention**  
  Token _t_ can only attend to tokens `<= t`.  
  This enables left-to-right generation.

- **Encoder-only (BERT-like)** uses **bidirectional self-attention**  
  Token _t_ can attend to all tokens in the sequence (both left and right).  
  This is great for “understanding” representations.

- **Encoder–decoder (seq2seq)** uses:
  - encoder: bidirectional self-attention
  - decoder: causal self-attention
  - decoder also uses **cross-attention** to read encoder outputs

### B) Training objective (what the model is optimized for)

Common objectives by family:

- **GPT**: autoregressive next-token prediction  
  \
  Given tokens `x_1..x_t`, predict `x_{t+1}`.

- **BERT**: masked language modeling  
  \
  Mask some tokens; predict the masked tokens from context.

- **T5**: span corruption (text-to-text)  
  \
  Mask spans in the input; generate the missing spans.

- **BART**: denoising autoencoder  
  \
  Corrupt the input (mask, permute, delete); reconstruct the original.

---

## 3) Code: causal vs bidirectional attention masks

The Transformer block is the same — the **mask** changes whether the model is GPT-like or BERT-like.

> This snippet uses `torch.nn.MultiheadAttention` for clarity (not for peak performance).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_mask(seq_len: int, device=None):
    # Float mask with -inf above diagonal: prevents attention to future tokens.
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)

class TinySelfAttentionBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, attn_mask=None):
        # Pre-norm
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

# Demo
B, T, D = 2, 6, 128
x = torch.randn(B, T, D)
block = TinySelfAttentionBlock(d_model=D)

# Bidirectional (encoder-style): no causal mask
y_bi = block(x, attn_mask=None)

# Causal (GPT-style): prevent attending to future tokens
mask = causal_mask(T, device=x.device)
y_causal = block(x, attn_mask=mask)

print("bidirectional:", y_bi.shape, "causal:", y_causal.shape)
```

**Takeaway:** GPT vs BERT is not “different attention,” it’s **the same attention with different masking and objectives**.

---

## 4) The three major Transformer “shapes”

### A) Encoder-only (BERT family)

**Best for:** embeddings, classification, reranking, retrieval.

- Stack of **bidirectional self-attention** layers
- Produces contextual embeddings for each token
- Often used by adding a task head (classification, token labeling, etc.)

Examples: **BERT, RoBERTa, DeBERTa, DistilBERT**

**Strengths**

- Strong representations for NLU tasks
- Efficient for embedding extraction

**Limits**

- Not naturally an autoregressive generator

---

### B) Decoder-only (GPT family)

**Best for:** generation, chat, code completion, instruction following.

- Stack of **causal self-attention** layers
- Predicts the next token repeatedly (generation loop)

Examples: **GPT-style LMs**, and many open LMs in the GPT family

**Strengths**

- Simple and scalable
- One model can handle many tasks via prompting

**Limits**

- Autoregressive decoding is sequential (generation latency scales with output length)

---

### C) Encoder–Decoder (Seq2Seq)

**Best for:** translation, summarization, structured transformations.

- Encoder reads input bidirectionally
- Decoder generates output autoregressively
- Decoder uses **cross-attention** over encoder outputs

Examples: **Original Transformer, T5, BART, mBART, FLAN-T5**

**Strengths**

- Natural for conditional generation (input → output)
- Can be efficient when input is long but output is moderate

**Limits**

- More components than decoder-only
- Cross-attention adds compute/memory

---

## 5) Code: tiny Encoder–Decoder sketch (structure-focused)

This minimal example highlights _where_ cross-attention sits.

```python
import torch
import torch.nn as nn

class TinySeq2Seq(nn.Module):
    def __init__(self, vocab_size=32000, d_model=256, n_heads=4, n_layers=2):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, src_ids, tgt_ids):
        src = self.tok(src_ids)  # [B, S, D]
        tgt = self.tok(tgt_ids)  # [B, T, D]

        # Encoder: bidirectional by default
        memory = self.encoder(src)  # [B, S, D]

        # Decoder: causal mask for autoregressive output
        T = tgt.size(1)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=tgt.device), diagonal=1)

        out = self.decoder(tgt, memory, tgt_mask=causal)  # includes cross-attention
        logits = self.lm_head(out)
        return logits
```

---

## 6) Mainstream Transformer architecture variants (what differs)

Below is a practical map of what changes across popular families.

### Original Transformer (encoder–decoder)

- Shape: encoder–decoder
- Positional encoding: sinusoidal (classic)
- Norm: post-norm in the original paper; many modern models use pre-norm

### BERT / RoBERTa (encoder-only)

- Shape: encoder-only
- Attention: bidirectional
- Objective: masked language modeling
- RoBERTa: improved training recipe (data/steps/hyperparams)

### GPT-style (decoder-only)

- Shape: decoder-only
- Attention: causal
- Objective: next-token prediction
- Common modern upgrades: pre-norm, improved tokenizers, better kernels

### T5 / FLAN-T5 (encoder–decoder)

- Shape: encoder–decoder
- Objective: span corruption (text-to-text)
- Design: unified interface (everything is text in/out)

### BART (encoder–decoder)

- Denoising objective: corrupt input, reconstruct original
- Strong for summarization and seq2seq-style tasks

---

## 7) Modern decoder-only design choices (commonly seen)

Many recent GPT-like LMs share these architectural choices:

- **Pre-norm** (stability at scale)
- **RMSNorm** (a common alternative to LayerNorm)
- **RoPE** (rotary positional embeddings) or similar long-context-friendly position methods
- **SwiGLU** MLP (often improves quality/efficiency vs GELU)
- Highly optimized attention kernels (e.g., FlashAttention-class techniques)

These are _implementation and architecture_ refinements on the same decoder-only template.

---

## 8) Efficient / long-context Transformers (two directions)

There are two broad ways to handle long sequences:

1. **Keep full attention graph, optimize kernels**
   - Example approach: FlashAttention-style kernels to reduce memory traffic and improve speed.
2. **Change the attention pattern (sparse / structured)**
   - Examples: Longformer/BigBird-style sparse attention, windowed attention, etc.

Both exist; many production systems prioritize kernel optimizations because they preserve full attention semantics.

---

## 9) Mixture-of-Experts (MoE) Transformers

MoE increases parameter count without proportional compute by activating only a subset of experts per token.

- Router picks top-k experts per token
- Only those experts run

Tradeoffs:

- Great scaling for quality/cost
- Harder serving (routing, load balancing, memory footprint)

---

## 10) Quick selection guide (what to pick)

Choose **decoder-only (GPT-like)** if you want:

- chat / general generation
- code completion
- prompt-driven multi-task behavior

Choose **encoder-only (BERT-like)** if you want:

- embeddings for retrieval
- classification / reranking
- efficient representation learning

Choose **encoder–decoder (T5/BART-like)** if you want:

- translation/summarization
- explicit input→output transformation tasks

---

## Appendix: tiny “GPT-like” forward (causal LM head)

```python
import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size=32000, d_model=256, n_heads=4, n_layers=2, max_len=2048):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(input_ids) + self.pos(pos_ids)

        causal = torch.triu(torch.full((T, T), float("-inf"), device=input_ids.device), diagonal=1)
        h = self.blocks(x, mask=causal)
        return self.lm_head(h)  # [B, T, V]
```

---
