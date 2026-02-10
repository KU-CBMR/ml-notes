+++
title = "Optimize Transformers: Save GPU Memory & Reduce Tokens (with Practical Code)"
date = 2026-02-10
draft = false
tags = ["transformer", "llm", "optimization", "memory", "kv-cache", "tokens", "pytorch", "inference"]
categories = ["ML Systems"]
description = "A detailed, practical guide to reducing GPU memory usage and token cost in Transformer/LLM training and inference, with concrete strategies, rules of thumb, and runnable Python snippets."
+++

This post is a practical, engineering-first guide to answering two questions:

1. **How do I save GPU memory when training or serving Transformers/LLMs?**
2. **How do I reduce tokens (cost + latency) without hurting output quality?**

You’ll get **actionable checklists**, **rules of thumb**, and **Python code** you can copy into your stack.

---

## 0) The mental model: where memory goes

Transformer memory usage is dominated by three buckets:

### A) Weight memory (parameters)

- The model weights themselves (plus any quantization metadata).
- In training, you also pay for **gradients** and **optimizer states** (often the biggest part).

### B) Activation memory (training heavy)

- Intermediate tensors from forward pass needed for backward.
- Usually scales with **batch size** and **sequence length**.

### C) KV cache (inference heavy)

- For autoregressive decoding, each generated token appends K/V tensors per layer.
- Scales with **(batch or concurrency) × (prompt length + generated length)**.

> A useful shortcut:
>
> - Training OOM → often **activations** or **optimizer states**
> - Inference OOM (after generating for a while) → often **KV cache**

---

## 1) Token reduction is the highest ROI optimization

If you only do one thing: **reduce input tokens + reduce output tokens**.

Why it works:

- Less compute (attention scales badly with long sequences)
- Less KV cache growth during generation
- Lower latency and lower API cost

### 1.1 Control output length aggressively (and politely)

**Do this in your prompt**:

- Ask for a fixed format (bullet list / JSON keys)
- Set explicit limits (“≤ 150 words”, “exactly 8 bullets”)

**Do this in decoding**:

- `max_new_tokens`
- `stop` sequences
- lower `temperature` if the model rambles

**Example prompt fragment**

```text
Output format:
- Exactly 6 bullet points.
- Each bullet <= 18 words.
- No intro, no outro.
Stop after the 6th bullet.
```

### 1.2 Don’t paste the entire world into the prompt

Common mistakes:

- Stuffing full docs into context
- Including long chat history when only last turns matter
- Repeating policies/instructions verbosely every time

Better approach:

- Keep a small **“system rules”** section
- Keep the last **N turns** (or last **K tokens**)
- Summarize older history into a short memory

### 1.3 RAG: retrieve small, relevant chunks instead of full documents

RAG (Retrieval Augmented Generation) reduces tokens by **injecting only top-k snippets**.

Practical tips:

- Chunk size: ~200–500 tokens
- Keep `k` small (e.g. 3–6)
- Clean chunks (remove boilerplate, headers, repeated navigation text)

### 1.4 Lightweight “compression” before sending to the LLM

For long user inputs, you can pre-process:

- Extract key fields (title, constraints, dates)
- Remove duplicate lines
- Normalize lists

**Example: a tiny, safe “prompt compressor”**

```python
import re

def compress_prompt(text: str, max_chars: int = 6000) -> str:
    # Remove repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Remove repeated blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Truncate hard (char-based), you can replace with token-based later
    return text[:max_chars]
```

---

## 2) Inference memory: KV cache is usually the culprit

### 2.1 Why KV cache grows

Decoder-only generation works like:

- Prompt → compute K/V for each layer
- Each new token → append new K/V for each layer
- Next token attends to **all previous tokens** (unless you do special tricks)

This means:

- Long prompts + long outputs + high concurrency = huge KV cache

### 2.2 Practical levers to reduce KV cache memory

#### Lever A: reduce total tokens (prompt + output)

- Shorter prompt (RAG + summarization)
- Smaller `max_new_tokens`
- “Stop early” with `stop` sequences

#### Lever B: reduce concurrency or batch smarter

If you serve many users:

- Use **continuous batching**
- Use **paged attention** (vLLM-style) to reduce fragmentation and reuse memory

#### Lever C: quantize KV cache (advanced)

Some stacks support storing KV in lower precision/bit-width to save memory.
You trade a bit of quality for capacity.

#### Lever D: Multi-query / grouped-query attention (model architecture)

Some models reduce KV size by having fewer KV heads than Q heads.
You can’t change this at runtime, but it’s a selection criterion when choosing a model.

---

## 3) Inference compute & memory: use better attention kernels

### 3.1 Use PyTorch SDPA (scaled_dot_product_attention)

Modern PyTorch can automatically pick optimized kernels that are faster and more memory-efficient.

**Example: replacing manual attention with SDPA**

```python
import torch
import torch.nn.functional as F

def sdpa_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    # q,k,v: (B, H, T, Hd)
    # attn_mask: broadcastable to (B, H, T, T) or None
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
```

### 3.2 FlashAttention (when available)

If your environment supports FlashAttention, it typically provides:

- Lower memory (does not materialize large attention matrices the same way)
- Higher speed for long sequences

---

## 4) Weight memory: precision and quantization

### 4.1 Mixed precision (BF16/FP16) is baseline

For inference:

- Most deployments run weights in BF16/FP16
- Often enough to cut memory roughly in half vs FP32

### 4.2 INT8 / INT4 quantization

If weight memory is your bottleneck:

- INT8 often has minimal quality loss and good speed
- INT4/NF4 compress more but may require careful validation

> Note: Quantizing weights reduces **weight memory**,  
> but if your bottleneck is **KV cache**, you still need token reduction and/or KV strategies.

---

## 5) Training memory: activations + optimizer states dominate

### 5.1 Activation checkpointing (recompute instead of store)

Big win for deep models and long sequences.

**PyTorch example**

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x)

class CheckpointedStack(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.blocks:
            # checkpoint requires a function; lambda captures blk
            x = checkpoint(lambda t: blk(t), x)
        return x
```

Trade-off:

- Lower activation memory
- Higher compute (extra forward replays)

### 5.2 Gradient accumulation

Useful when batch size OOMs:

- Split batch into micro-batches
- Accumulate gradients
- Step optimizer once

**Example**

```python
def train_with_accum(model, optimizer, loss_fn, dataloader, device, accum_steps=4):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

### 5.3 Optimizer state (AdamW is expensive)

AdamW stores two extra tensors per parameter (m, v), often FP32.
Options:

- 8-bit optimizers (if you can use them)
- ZeRO/FSDP to shard optimizer states across GPUs

### 5.4 Sequence packing + dynamic padding

Padding is wasted attention compute and wasted activation memory.

Two simple rules:

- Pad to **batch max length**, not global max.
- Pack multiple short samples into one sequence when possible.

---

## 6) A simple token + memory estimator (rule-of-thumb calculator)

You can’t optimize what you can’t estimate.  
Below is a small helper that estimates:

- total tokens per request
- rough KV cache size (very approximate, but useful for capacity planning)

**KV cache rough estimate**
For decoder-only inference, KV cache per token roughly scales with:

- `layers × heads_kv × head_dim × dtype_bytes × 2` (K and V)
- multiplied by `batch_size × sequence_length`

Different implementations store slightly different layouts, so treat this as a ballpark.

```python
def estimate_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,  # 2 for fp16/bf16, 1 for int8-like, 4 for fp32
) -> int:
    # K and V => factor 2
    return batch_size * seq_len * n_layers * n_kv_heads * head_dim * dtype_bytes * 2

def pretty_gb(n_bytes: int) -> str:
    return f"{n_bytes / (1024**3):.2f} GB"

# Example usage:
if __name__ == "__main__":
    # Example: 32 layers, 8 KV heads, head_dim 128, BF16, batch=4, seq=2048
    kv = estimate_kv_cache_bytes(
        batch_size=4,
        seq_len=2048,
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
    )
    print("Estimated KV cache:", pretty_gb(kv))
```

How to use it:

- Plug in your model config
- Compare different `seq_len` and concurrency values
- Decide if you need shorter prompts, smaller max output, or lower concurrency

---

## 7) A practical optimization playbook (what to do first)

### Step 1 — Stabilize token length (biggest ROI)

- Put hard caps on prompt size
- Cap output with `max_new_tokens`
- Enforce short output formats
- Use windowed history + summary memory

### Step 2 — Fix inference OOM (usually KV cache)

- Shorten context (RAG top-k, summarization)
- Reduce concurrency or implement continuous batching
- Prefer paged attention / vLLM-like serving if you do high throughput
- Consider KV cache quantization if supported

### Step 3 — Fix training OOM (activations + optimizer)

- BF16 + activation checkpointing
- Gradient accumulation
- Dynamic padding + packing
- ZeRO/FSDP if multi-GPU

### Step 4 — Squeeze weights

- FP16/BF16 inference baseline
- INT8 / INT4 if weight memory is the limiter

---

## 8) Quick diagnosis: which bucket is killing you?

### Case A: OOM immediately (before generating)

Likely:

- weights too big
- activation spikes during prompt prefill

Try:

- lower precision / quantization
- reduce batch
- shorter prompts

### Case B: OOM after generating some tokens

Likely:

- KV cache growth

Try:

- reduce prompt + max output tokens
- reduce concurrency
- paged attention / better serving engine

### Case C: Sometimes OOM, sometimes not

Likely:

- token length variance
- memory fragmentation

Try:

- enforce strict input/output limits
- use batching strategies that avoid pathological long sequences
- keep model loaded (avoid frequent unload/reload)

---

## Closing thoughts

“Transformer optimization” is often less about exotic kernels and more about discipline:

- **Feed fewer tokens** (token trimming + RAG + summaries)
- **Store less** (checkpointing, quantization, KV management)
- **Serve smarter** (continuous batching, paged attention)
