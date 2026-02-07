+++
title = "LLM Inference Latency: A Practical Debugging Checklist"
date = 2026-02-06
draft = false
tags = ["llm", "inference", "latency", "profiling", "debugging"]
categories = ["ML Systems"]
description = "A step-by-step checklist to diagnose and reduce LLM inference latency (p50/p95)."
+++

This post is a compact checklist to debug LLM inference latency (p50/p95).
It helps you quickly identify whether you are CPU-bound, GPU-bound, or queueing-bound.

## Checklist

- Measure p50/p95/p99 latency and tokens/s
- Separate queueing time vs model compute time
- Track token length distributions (input/output)
- Check GPU utilization + VRAM pressure
- Verify batching and concurrency settings

---

## Download (paid)

I’ve packaged this checklist + a benchmark template into a downloadable bundle:

- 👉 Buy the bundle: YOUR_GUMROAD_OR_LEMONSQUEEZY_LINK

> Disclosure: The link above is a paid product link. If you purchase, I earn revenue.

<!-- +++
title = "LLM Inference Latency: A Practical Debugging Checklist"
date = 2026-02-06
draft = false
tags = ["llm", "inference", "latency", "profiling", "debugging"]
categories = ["ML Systems"]
description = "A practical, step-by-step checklist to diagnose and reduce LLM inference latency (p50/p95), from input pipeline to GPU kernels."
+++

This post is a compact checklist I use to debug LLM inference latency issues (p50/p95).
The goal is to quickly locate the bottleneck before making any code or infrastructure changes.

## What to measure first
- Throughput: tokens/s (or requests/s)
- Latency: p50 / p95 / p99
- GPU utilization: SM%, memory bandwidth, and VRAM usage

## Common bottleneck map
- Input / preprocessing → tokenization, I/O, batching
- Model execution → GPU kernels, KV cache, attention, GEMMs
- Postprocessing → detokenization, streaming, formatting
- Serving layer → queueing, timeouts, concurrency limits

## Quick checks (high-signal)
1. Is the GPU idle? If utilization is low while latency is high, you are likely CPU-bound or under-batched.
2. Separate queueing time from compute time.
3. Track token length distributions (not just averages).
4. Watch VRAM pressure: paging/evictions often cause p95 spikes.

Inline math: $x^2 + y^2$

$$
\mathcal{L} = -\sum_i y_i \log \hat{y}_i
$$ -->
