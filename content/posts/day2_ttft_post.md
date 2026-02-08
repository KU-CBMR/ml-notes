+++
title = "Reduce LLM Time-To-First-Token (TTFT): A Practical Debugging Guide"
date = 2026-02-07
draft = false
tags = ["llm", "latency", "ttft", "streaming", "profiling", "debugging"]
categories = ["ML Systems"]
description = "A step-by-step guide to diagnose and reduce Time-To-First-Token (TTFT) with concrete measurements and fixes."
+++

Many “LLM latency” issues are actually **TTFT** issues: the model is fine, but users still _feel_ the system is slow because the first token arrives late.

This post shows how to break TTFT into measurable parts and fix the biggest offenders.

## What TTFT really includes

TTFT is end-to-end and often dominated by non-model time:

- Client-side overhead (DNS, TLS handshake, connection reuse)
- Gateway / proxy time (routing, auth, rate limits)
- Queueing time (concurrency limits, provider throttling)
- Request prep time (prompt building, serialization)
- Model warm-up / cache misses
- Tool calls / RAG before generation starts (if applicable)

## Measure TTFT the right way (3 timestamps)

Add these three timestamps to every request:

- **t0**: client sends request
- **t1**: server/provider acknowledges / request accepted
- **t2**: first token received (stream) or first byte of completion

Then compute:

- **Network + handshake** ≈ (t0 → t1)
- **Server + queueing** ≈ (t1 → t2) _(a good proxy even if you can’t log model start)_
- **TTFT** = (t0 → t2)

> If you can log `model_start`, split (t1 → t2) into `queueing` and `server/model prep`. If you can’t, (t1 → t2) is still enough to tell “not network” vs “network”.

## Quick diagnosis: where is your TTFT coming from?

### 1) TTFT spikes only at peak traffic → queueing

Common causes:

- Too many concurrent requests
- Provider rate limits / throttling
- Batch/worker pool saturation
- Retries amplifying load during incidents

Fixes:

- Cap concurrency and add backpressure (reject fast vs slow timeouts)
- Separate “interactive” vs “batch” traffic (dedicated limits/workers)
- Add a request queue with priority lanes (VIP/interactive first)
- Use jittered retries + circuit breakers (avoid retry storms)

### 2) TTFT is consistently high even at low traffic → connection / overhead

Common causes:

- No keep-alive / no connection pooling
- TLS handshake every request
- Heavy request serialization (large JSON payloads, extra fields)
- Synchronous logging on the hot path

Fixes:

- Enable keep-alive + connection pooling (client + gateway)
- Reuse TLS sessions where possible
- Reduce payload size (avoid sending full chat history when unnecessary)
- Move logging/metrics off the critical path (async, sampling)

### 3) TTFT is high only when RAG/tools are enabled → pre-generation work

Common causes:

- Slow vector DB queries
- Reranking too expensive for interactive requests
- Serial tool calls before any tokens are streamed

Fixes:

- Stream _after_ minimal context is ready (don’t block on non-critical tool calls)
- Cache retrieval results for repeated queries (per-user + global cache)
- Parallelize retrieval steps (query + fetch + rerank where possible)
- Use smaller rerank models or skip rerank when confidence is high

## A minimal TTFT instrumenting checklist

Add these spans (even if you start with coarse timing):

- `dns_lookup`
- `tcp_connect`
- `tls_handshake`
- `request_write`
- `gateway_auth`
- `rate_limit_check`
- `queue_wait`
- `prompt_build`
- `retrieval`
- `rerank`
- `provider_request`
- `first_token_received`

If you can only pick **3**, pick:

1. `t0 client_send`
2. `t1 provider_accept`
3. `t2 first_token_received`

## 5 changes that usually improve TTFT immediately

1. **Connection reuse**: keep-alive + pooling end-to-end
2. **Prompt build caching**: cache static system + templates, avoid repeated string work
3. **RAG fast path**: cache, skip rerank, or cap docs for interactive requests
4. **Traffic separation**: interactive vs batch concurrency limits
5. **Fail fast**: tight timeouts + circuit breakers (better UX than slow hangs)

---

<!--
## Buy & download

If you want a ready-to-use **TTFT + throughput benchmarking template** and a one-page debugging checklist:

- 👉 [Buy & download](https://mlnotes.gumroad.com/l/LLMLatencyDebuggingChecklist)

> Disclosure: The link above is a paid product link. If you purchase, I earn revenue. -->
