---
title: "vLLM: What It Is, Why It Matters, and a Tiny Local PDF Summarizer Tool"
date: 2026-02-08
draft: false
tags: ["vllm", "llm", "inference", "openai-api", "pdf", "automation"]
categories: ["ML Systems"]
description: "A practical introduction to vLLM and a small example tool: start a local OpenAI-compatible server, summarize PDFs in batches."
---

## What is vLLM?

**vLLM** is a high-performance LLM inference engine designed to serve large language models efficiently, especially when you have **many concurrent requests** or long prompts/outputs.

A key idea behind vLLM is efficient GPU memory management (often discussed as _paged attention_), which helps reduce waste and improves throughput for real-world serving workloads.

In practice, vLLM is popular because it can run an **OpenAI-compatible API server**, meaning your application can call endpoints like:

- `POST /v1/chat/completions`

…while the model is running locally or on your own machine/cluster.

## Why use vLLM?

Here are the most common reasons:

- **OpenAI-compatible interface**: You can reuse existing OpenAI-style client code with minimal changes.
- **Higher throughput**: Better batching + memory efficiency can improve requests/sec.
- **Lower operational friction**: Spin up a server, send requests, stop it when done.
- **Local/private workloads**: Useful when you want to process internal documents without sending them to external services.

## A small example tool: “Start vLLM → Summarize PDFs → Stop”

This page walks through a compact pattern for building a practical tool:

1. **Start** a local vLLM OpenAI server (only if one isn’t already running)
2. **Wait** until the server is ready
3. **Slice** PDFs into page chunks
4. **Map step**: extract structured info from each chunk
5. **Reduce step**: merge chunk-level extractions into a single final summary
6. **Write outputs** to JSON files

This design is great for:

- batch document processing,
- repeatable research workflows,
- lightweight “one-file utilities” you can run from the command line.

---

## Architecture overview

### 1) Start an OpenAI-compatible local server

The tool first checks whether a server is already running by calling:

- `GET /v1/models`

If it’s not running, it launches the vLLM server process.

### 2) Wait for readiness

Cold starts can take time (model download + load). A simple readiness loop retries `GET /v1/models` until it succeeds or times out.

### 3) PDF slicing (page chunks)

Instead of sending an entire PDF at once, the tool:

- reads text page-by-page,
- groups pages into chunks (e.g., 2 pages per chunk),
- and further splits overly long chunks by character limits.

This keeps prompts stable and avoids hitting context limits.

### 4) Map/Reduce summarization

**Map**: for each chunk, ask the model for a structured extraction (e.g., JSON fields).  
**Reduce**: merge chunk results into one consolidated JSON.

This scales better than “summarize the whole document at once” and is easier to debug.

---

## Minimal example code (sanitized)

### Configuration (safe defaults)

```python
from pathlib import Path
import os

# Safe, non-sensitive placeholders
WORK_DIR = Path("./runtime")                 # local folder for logs
OUTPUT_DIR = Path("./outputs")               # local folder for results
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOST = os.environ.get("VLLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BASE_URL_V1 = f"http://{HOST}:{PORT}/v1"

# Use a public model name or local directory path you control
MODEL = os.environ.get("VLLM_MODEL", "org-or-user/model-name")

# Example vLLM args (tune to your GPU / needs)
VLLM_ARGS = os.environ.get("VLLM_ARGS", "--dtype auto --max-model-len 4096")
```

### Start server and wait until ready

```python
import sys, time, subprocess, requests

def start_vllm_server():
    # Check if already running
    try:
        r = requests.get(f"{BASE_URL_V1}/models", timeout=1)
        if r.ok:
            print("vLLM server already running.")
            return None, False
    except Exception:
        pass

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--host", HOST,
        "--port", str(PORT),
        *VLLM_ARGS.split()
    ]

    log_path = WORK_DIR / "vllm_server.log"
    log_f = open(log_path, "ab", buffering=0)

    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
    return proc, True

def wait_ready(timeout_sec=600):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL_V1}/models", timeout=3)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(1.5)
    return False
```

### Call the OpenAI-compatible endpoint

```python
from openai import OpenAI

def make_client():
    # Dummy key: often required by client libraries even for local servers
    return OpenAI(base_url=BASE_URL_V1, api_key="na")

def chat_once(client, user_text, max_tokens=400):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_text}],
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=600,
    )
    return (resp.choices[0].message.content or "").strip()
```

### PDF slicing (simple version)

```python
import pdfplumber
import math

def iter_pdf_chunks(pdf_path, chunk_pages=2, max_chars=1500):
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for i in range(0, n_pages, chunk_pages):
            pages = pdf.pages[i : min(i + chunk_pages, n_pages)]
            text = "\n".join((p.extract_text() or "") for p in pages).strip()
            if not text:
                continue

            # further split too-long chunks
            if len(text) > max_chars:
                parts = math.ceil(len(text) / max_chars)
                span = len(text) // parts
                for k in range(parts):
                    yield (i+1, min(i+chunk_pages, n_pages), text[k*span : (k+1)*span])
            else:
                yield (i+1, min(i+chunk_pages, n_pages), text)
```

### Map/Reduce pattern (structured outputs)

You can do “chunk-level extraction” first, then merge.

```python
import json

MAP_PROMPT = """You are a research assistant. Extract ONLY what is present in the provided slice.
Return strict JSON with these fields:

{
  "paper_id": "",
  "span_pages": "start-end",
  "method": "",
  "dataset": "",
  "metrics": "",
  "code_link": "",
  "notes": ""
}

Slice:
{slice_text}
"""

REDUCE_PROMPT = """You will receive a list of JSON objects extracted from slices of the same paper.
Merge them into one strict JSON:

{
  "paper_id": "",
  "method": "",
  "dataset": "",
  "metrics": "",
  "code_link": "",
  "notes": "",
  "page_refs": []
}

Rules:
- Use only the provided inputs.
- Prefer more specific statements.
- If information is missing, use "unknown".
Return JSON only.

Inputs:
{items}
"""

def safe_json_loads(s):
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        # best-effort: extract a JSON object/array substring
        for a, b in [("{", "}"), ("[", "]")]:
            i, j = s.find(a), s.rfind(b)
            if i != -1 and j != -1 and j > i:
                return json.loads(s[i:j+1])
        raise

def map_extract(client, paper_id, start_page, end_page, chunk_text):
    prompt = MAP_PROMPT.format(
        slice_text=f"[DOC={paper_id}][PAGE={start_page}-{end_page}]\n{chunk_text}"
    )
    raw = chat_once(client, prompt, max_tokens=600)
    obj = safe_json_loads(raw)
    obj.setdefault("paper_id", paper_id)
    obj.setdefault("span_pages", f"{start_page}-{end_page}")
    return obj

def reduce_merge(client, paper_id, items):
    prompt = REDUCE_PROMPT.format(items=json.dumps(items, ensure_ascii=False))
    raw = chat_once(client, prompt, max_tokens=800)
    merged = safe_json_loads(raw)
    merged.setdefault("paper_id", paper_id)
    return merged
```

### Putting it together (batch processing)

```python
from pathlib import Path

def summarize_pdf(client, pdf_path):
    pdf_path = Path(pdf_path)
    paper_id = pdf_path.stem
    partials = []

    for (s, e, txt) in iter_pdf_chunks(str(pdf_path), chunk_pages=2, max_chars=1500):
        partials.append(map_extract(client, paper_id, s, e, txt))

    merged = reduce_merge(client, paper_id, partials)

    out = OUTPUT_DIR / f"{paper_id}.json"
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)
```

## Where to go next

If you want to turn this into a more polished “mini app”, you can add:

- CLI flags (chunk size, output tokens, temperature, model name)
- retries + backoff on failed generations
- structured validation (e.g., JSON schema)
- streaming progress UI
- parallel PDF processing (careful with GPU concurrency)

---

## Summary

vLLM makes it straightforward to run a **fast local LLM server** behind an **OpenAI-compatible API**.  
By combining that with a **Map/Reduce PDF summarization pipeline**, you can build small, reliable batch tools that are easy to debug, scale, and keep private.
