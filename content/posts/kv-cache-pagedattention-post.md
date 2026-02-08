+++
title = "KV Cache & PagedAttention: Why Long Context Gets Slow (and How to Fix It)"
date = 2026-02-08
draft = false
tags = ["llm", "inference", "kv-cache", "pagedattention", "vllm", "performance"]
categories = ["ML Systems"]
description = "A practical guide to KV cache, why long prompts hurt latency/throughput, and what to do about it (chunked prefill, prefix cache, eviction, and vLLM/PagedAttention basics)."
+++

- “Why does a slightly longer prompt explode my latency?”
- “Why do I OOM even though the model weights fit?”
- “Why does throughput collapse with many concurrent chats?”

This post is a compact, practical guide.

---

## What is the KV cache (in one paragraph)

During autoregressive decoding, each new token attends to _all previous tokens_. To avoid recomputing attention keys/values for past tokens every step, we store them in GPU memory as the **KV cache**. Great for speed—but it scales with:

- **sequence length** (more tokens ⇒ more KV)
- **batch/concurrency** (more requests ⇒ more KV)
- **layers / hidden size / heads** (bigger model ⇒ more KV)
- **dtype** (fp16/bf16 vs fp8/int8 KV)

So KV cache is often the _real_ bottleneck, not the model weights.

---

## Why long context makes everything slower

Two different phases hurt differently:

### 1) Prefill (processing the prompt)

You pay an up-front compute cost to encode the prompt and populate KV.

- Symptoms: **TTFT increases** with prompt length.
- Typical bottleneck: compute (and sometimes CPU tokenization).

### 2) Decode (generating tokens)

Each step attends over the entire cached history.

- Symptoms: **tokens/sec drops** as context grows, especially under concurrency.
- Typical bottleneck: memory bandwidth + KV cache pressure.

Rule of thumb: if you see **TTFT spikes** it’s usually prefill; if you see **steady tokens/sec degradation** over the conversation, it’s usually decode + KV.

---

## The 4 metrics that reveal KV cache problems

Track these per request (and aggregate p50/p95):

1. **input_tokens** (prompt length)
2. **output_tokens**
3. **TTFT** (time-to-first-token)
4. **decode_tokens_per_second** (or per-token latency)

Plus one system metric:

- **GPU memory usage** and _OOM/evictions_ count (if available)

A dead giveaway:

- TTFT grows ~linearly with input tokens (prefill cost),
- tokens/sec drops as “effective context” grows (decode cost),
- OOM happens first when concurrency rises.

---

## Why vLLM’s PagedAttention matters

The KV cache is basically “a lot of memory blocks”. A naive layout can fragment memory and waste space, especially with many requests of different lengths.

**PagedAttention** (vLLM’s key idea) manages KV cache in _paged blocks_, similar to virtual memory paging:

- reduces fragmentation,
- improves utilization under high concurrency,
- makes long-running serving more stable.

If you’re serving multiple chats concurrently, this is often a bigger win than micro-optimizing kernels.

---

## Practical fixes (ordered by impact)

### 1) Cap context (yes, really)

If your product doesn’t need 32k context, don’t pay for it.

- Set a hard `max_model_len` (or equivalent).
- Enforce truncation and summarize older history.

**Best ROI**: big reduction in KV memory + better throughput immediately.

### 2) Chunked prefill (reduce TTFT variance)

Instead of prefilling huge prompts as one big job, chunk it so the system can interleave work across requests.

- Helps tail latency (p95 TTFT) under load.
- Improves fairness when a few giant prompts would otherwise block the queue.

### 3) Prefix caching (cheap wins for repeated prompts)

If users share a common system prompt or template (RAG scaffolding, policies, etc.):

- cache the prefix KV once,
- reuse across requests.

This reduces prefill compute and lowers TTFT for repeated structures.

### 4) Control concurrency deliberately

High concurrency is great until KV cache becomes the limiter.

- Add an admission controller based on **available KV blocks** (not just request count).
- Consider separate pools/queues for “long-context” vs “short-context” traffic.

### 5) Reduce KV size

Depending on your stack/model:

- use lower precision KV (if supported),
- use quantized KV cache modes,
- use smaller model variants for long-context routes.

Even small reductions compound under concurrency.

---

## A simple “KV cache diagnosis” checklist

- [ ] Does TTFT correlate strongly with `input_tokens`? (prefill)
- [ ] Does tokens/sec degrade as conversation length grows? (decode + KV)
- [ ] Does throughput collapse primarily when concurrency increases? (KV memory pressure)
- [ ] Are OOMs happening with many sessions, not just big prompts? (KV fragmentation/utilization)
- [ ] Can you reproduce by replaying traffic with controlled input/output lengths?

If you answer “yes” to 3+ items, prioritize KV strategies over general “GPU utilization” tuning.

---

## Minimal logging snippet (copy/paste)

Add these fields to every request log:

```json
{
  "ts": "ISO8601",
  "model": "…",
  "input_tokens": 1234,
  "output_tokens": 256,
  "ttft_ms": 420,
  "total_latency_ms": 3120,
  "decode_tokens_per_s": 85.3,
  "concurrency": 12,
  "gpu_mem_used_mb": 38750
}
```

Then slice by:

- input token buckets (0–512, 512–2k, 2k–8k, 8k+)
- concurrency buckets (1–4, 5–16, 17+)

You’ll see immediately whether it’s prefill, decode, or KV pressure.

---

## Closing thought

You can’t “optimize away” KV cache physics: long context + high concurrency costs memory.  
The winning move is usually **product + systems design**:

- cap context,
- reuse prefixes,
- chunk prefill,
- and use paging-aware KV management (like vLLM’s PagedAttention).
