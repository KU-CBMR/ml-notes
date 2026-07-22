---
title: "Local LLM Deployment Tools Explained: Transformers, Ollama, vLLM, TGI, LMDeploy, and SGLang"
date: 2026-02-19
draft: false
tags: ["LLM", "Inference", "Deployment", "RAG", "Embedding"]
categories: ["LLM Systems"]
---

When I first started building a local LLM application, I kept seeing the same names:

```text
Transformers
Ollama
vLLM
Text Generation Inference
LMDeploy
SGLang
```

At first, they looked like six competing tools that did the same thing.

They do not.

Some tools load a model directly inside Python. Some run the model as a separate HTTP service. Some prioritize easy local use, while others prioritize GPU utilization, concurrency, and production serving.

This post explains what each tool does, how local deployment differs from server deployment, and why a project may use:

```text
Qwen2.5-3B-Instruct
-> generate the final answer through HTTP

all-MiniLM-L6-v2
-> generate vectors directly inside Python
```

The goal is not to memorize six product names. The goal is to understand which layer each tool belongs to.

---

## 1. First separate the model from the tool

A model and a deployment tool are not the same thing.

For example:

```text
Qwen2.5-3B-Instruct
```

is a **language model**. It contains learned parameters and predicts the next token. It is the part that actually writes an answer.

By contrast:

```text
Transformers
Ollama
vLLM
TGI
LMDeploy
SGLang
```

are software tools used to load, run, optimize, or serve models.

A simple mental model is:

```text
model weights       = the brain
Transformers        = a Python toolkit for using the brain
inference engine    = an optimized machine that runs the brain
HTTP API            = an interface used by other programs
application         = the website, chatbot, or RAG system
```

The same Qwen model can therefore be run through Transformers, Ollama, vLLM, TGI, LMDeploy, or SGLang.

The brain may stay the same. The surrounding execution system changes.

---

## 2. The four layers of an LLM application

A typical LLM application can be divided into four layers.

### Layer 1: The model

Examples:

```text
Qwen
Llama
Mistral
Gemma
```

The model performs the actual neural-network computation.

### Layer 2: The Python model library

The main example here is:

```text
Transformers
```

It downloads model files, builds the network, loads weights, tokenizes text, and calls methods such as `generate()`.

### Layer 3: The inference and serving system

Examples:

```text
Ollama
vLLM
TGI
LMDeploy
SGLang
```

This layer keeps the model running and may handle:

- GPU memory management;
- request batching;
- streaming output;
- quantization;
- multi-GPU execution;
- concurrency;
- HTTP APIs.

### Layer 4: The application

Examples:

```text
website
chatbot
RAG system
research tool
API backend
```

The application sends a prompt and receives an answer. It does not always need to know how the model is loaded internally.

This separation is useful because the application may later switch from Ollama to vLLM without rewriting the whole retrieval pipeline.

---

## 3. Transformers: load and control models inside Python

[Transformers](https://huggingface.co/docs/transformers/index) is the most general-purpose tool in this list.

A simplified example looks like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

inputs = tokenizer("Explain gradient descent.", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

Here, the model is loaded directly into the current Python process.

Transformers is a good choice when I want to:

- study how a model works;
- write custom inference code;
- access hidden states;
- fine-tune a model;
- test a model in a notebook;
- control preprocessing and generation in detail.

However, Transformers is not primarily a production request server.

If many users send requests at the same time, I may need to build several things myself:

```text
request queue
batching
streaming
API layer
worker management
GPU memory coordination
failure recovery
```

So the simplest mental model is:

```text
Transformers
= maximum flexibility inside Python
```

It is excellent for research, fine-tuning, and custom logic. It is not always the easiest way to serve many users efficiently.

---

## 4. Ollama: the easiest local model service

[Ollama](https://docs.ollama.com/) is designed to make local model usage simple.

After installation, it can download a model, run it, and expose a local API. Its default API address is usually:

```text
http://localhost:11434
```

The architecture becomes:

```text
Python application
        |
        | HTTP request
        v
Ollama service
        |
        v
Qwen / Llama / Gemma model
```

Ollama hides many setup details:

- model download;
- model file management;
- local API creation;
- common quantized model formats;
- starting and stopping the runtime.

This makes it especially useful for:

```text
personal projects
local chatbots
prototypes
demonstrations
laptop or workstation development
```

Its main priority is convenience.

For a large production system with many simultaneous users, I may want more control over batching, GPU scheduling, parallelism, and distributed deployment. That is where vLLM, TGI, LMDeploy, and SGLang become more relevant.

```text
Ollama
= easy local model runner + local HTTP service
```

---

## 5. Production-oriented inference servers

The next four tools solve a similar high-level problem:

```text
Keep a large model loaded,
receive requests,
run inference efficiently,
and return generated tokens.
```

They overlap, but each has a different emphasis.

### 5.1 vLLM: high-throughput general-purpose serving

[vLLM](https://docs.vllm.ai/) is a high-performance inference and serving engine for large language models.

Its job is not to train the model. Its job is to run an already trained model efficiently.

vLLM provides an OpenAI-compatible HTTP server, so an application can call endpoints such as:

```text
/v1/chat/completions
```

while the actual model runs on my own GPU.

```text
application
    |
    | OpenAI-compatible HTTP request
    v
vLLM server
    |
    v
Qwen2.5-3B-Instruct
```

vLLM focuses on features such as:

- continuous batching;
- efficient KV-cache management;
- streaming generation;
- tensor parallelism;
- high-throughput GPU serving.

It is a strong choice when one GPU server receives many requests or when several applications need to share one model.

```text
vLLM
= an optimized model server for high request throughput
```

### 5.2 TGI: serving in the Hugging Face ecosystem

[Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index), usually called **TGI**, is Hugging Face's toolkit for deploying and serving language models.

Like vLLM, it runs the model as a dedicated service and supports production features such as:

- token streaming;
- continuous batching;
- tensor parallelism;
- metrics and tracing integrations;
- HTTP serving;
- OpenAI-compatible chat endpoints.

TGI fits naturally into a workflow already centered on:

```text
Hugging Face Hub
Transformers
Hugging Face model repositories
container-based deployment
```

The difference from Transformers is:

```text
Transformers
-> load the model inside my Python program

TGI
-> run the model in a dedicated server process
```

```text
TGI
= production model serving in the Hugging Face ecosystem
```

### 5.3 LMDeploy: deployment, acceleration, and quantization

[LMDeploy](https://lmdeploy.readthedocs.io/) is a toolkit for compressing, deploying, and serving large language models and vision-language models.

It supports both offline inference and API serving, with two main engine paths:

```text
TurboMind engine
PyTorch engine
```

LMDeploy includes features such as:

- OpenAI-compatible serving;
- quantized-model inference;
- multi-GPU deployment;
- offline batch inference;
- LLM and VLM support.

It is especially interesting when model compression and deployment are part of the same problem.

For example, if a model is too large for the available GPU memory, quantization can reduce its memory requirement, while an optimized engine can make the quantized model practical to serve.

```text
LMDeploy
= deployment plus optimized and quantized inference
```

### 5.4 SGLang: structured and prefix-heavy workloads

[SGLang](https://docs.sglang.ai/) is a high-performance serving framework for language and multimodal models.

It is designed for low-latency and high-throughput inference, from a single GPU to distributed clusters.

Its features include:

- prefix caching;
- RadixAttention;
- continuous batching;
- multi-GPU parallelism;
- structured generation;
- OpenAI-compatible APIs.

Prefix reuse matters when many requests share the same long beginning.

For example:

```text
You are a scientific research assistant...
[large shared instruction block]
[user-specific question]
```

Recomputing the shared prefix from the beginning for every request is wasteful. A serving system with strong prefix caching can reuse part of the previous computation.

This is useful in:

```text
RAG systems with repeated templates
agents with long shared instructions
structured-output pipelines
multi-step LLM programs
```

```text
SGLang
= high-performance serving for structured and reusable LLM workflows
```

---

## 6. A practical comparison

| Tool         | Main role                          | Typical use                                                      |
| ------------ | ---------------------------------- | ---------------------------------------------------------------- |
| Transformers | Python model library               | Research, fine-tuning, custom inference                          |
| Ollama       | Easy model runner and local API    | Personal use, prototypes, simple local deployment                |
| vLLM         | Optimized inference server         | High-throughput OpenAI-compatible serving                        |
| TGI          | Hugging Face serving toolkit       | Hugging Face-centered production deployment                      |
| LMDeploy     | Optimized and quantized deployment | LLM/VLM serving, quantization, TurboMind/PyTorch engines         |
| SGLang       | High-performance serving framework | Prefix-heavy, structured, agentic, and high-throughput workloads |

This table describes each tool's main design priority. It is not a strict boundary.

For example:

- vLLM can run on a local GPU workstation;
- Ollama can be exposed to another machine;
- Transformers can be placed behind a custom API;
- LMDeploy and SGLang can run locally during development.

---

## 7. Local machine versus server

The word **local** can mean two different things.

```text
Question 1:
Is the model running on my own hardware or on a third-party cloud API?

Question 2:
Is the model running inside the same Python process or in a separate service?
```

These are not the same distinction.

### Same machine, same Python process

```text
Python application
    |
    v
Transformers model
```

This is local and in-process.

### Same machine, separate HTTP service

```text
Python application
    |
    | localhost HTTP
    v
Ollama / vLLM / LMDeploy / SGLang
```

This is still local deployment.

`localhost` simply means that two programs on the same computer communicate through a network interface.

### Different machines

```text
website or laptop
       |
       | network request
       v
GPU server
       |
       v
model service
```

This is remote server deployment.

The application code may remain almost identical. The main change is the server address.

---

## 8. Advantages of local deployment

Running a model locally gives me:

### Privacy

Prompts and documents can stay on my own machine or private network.

### Offline availability

The model can work without an external API connection.

### Cost control

There is no per-token API bill. For a small workload, existing hardware may be enough.

### Full control

I can choose the model, quantization level, software version, logging behavior, and update schedule.

The main limitation is hardware.

A laptop may have limited:

```text
RAM or VRAM
model size
generation speed
concurrency
cooling and power
```

Local deployment is excellent for privacy and development, but it is not automatically the fastest option.

---

## 9. Advantages of a dedicated server

A server becomes useful when the workload grows.

It may provide:

- one or more stronger GPUs;
- shared access for several users;
- a model that stays loaded while the web app restarts;
- request queues and multiple workers;
- monitoring and load balancing;
- easier multi-GPU scaling.

However, it also introduces:

```text
hardware or rental cost
system administration
network latency
authentication
monitoring
maintenance
```

So the real question is not:

```text
Is local or server better?
```

It is:

```text
How many users do I have?
How large is the model?
Where should the data stay?
How much maintenance do I want?
```

---

## 10. Why Qwen generates answers through HTTP

Consider this design:

| Task               | Model                 | How it is called                    |
| ------------------ | --------------------- | ----------------------------------- |
| Generate an answer | `Qwen2.5-3B-Instruct` | Call a local HTTP model service     |
| Generate vectors   | `all-MiniLM-L6-v2`    | Load directly in the Python process |

This is a sensible architecture because the two models do different jobs.

`Qwen2.5-3B-Instruct` is an instruction-tuned causal language model. It reads a prompt and generates new tokens as the answer.

Compared with an embedding model, it is relatively large and expensive to run. Keeping it in a separate service gives several benefits:

- the model is loaded only once;
- the application process stays smaller;
- several app components can share it;
- the model runtime can restart independently;
- the serving engine can batch requests;
- generated tokens can be streamed.

The request flow is:

```text
Python application
    |
    | prompt
    v
local HTTP model service
    |
    v
Qwen2.5-3B-Instruct
    |
    | generated text
    v
Python application
```

HTTP does not mean that the model is on the public internet.

The service may run on the same machine at an address such as:

```text
http://127.0.0.1:8000
```

---

## 11. What all-MiniLM-L6-v2 actually does

[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is not used to write the final answer.

It is an **embedding model**.

Its job is to convert a sentence or paragraph into a fixed-length vector.

For example:

```text
"How does gradient descent work?"
```

becomes something conceptually like:

```text
[0.12, -0.03, 0.44, ..., 0.08]
```

For `all-MiniLM-L6-v2`, the output is a 384-dimensional dense vector.

The exact numbers are not meaningful to a human. What matters is the distance between vectors.

Texts with similar meanings should have vectors that are closer together.

```text
"How does gradient descent work?"

and

"Explain how an optimizer reduces the loss."
```

should be closer than:

```text
"How do I bake a chocolate cake?"
```

This allows the application to perform semantic search.

---

## 12. How the two models work together in RAG

A language model cannot efficiently read every document in a large knowledge base for every question.

RAG first retrieves a small number of relevant passages, then gives those passages to the answer model.

### Step 1: Index the documents

```text
documents
    |
    v
split into chunks
    |
    v
all-MiniLM-L6-v2
    |
    v
embedding vectors
    |
    v
vector database or vector index
```

Each document chunk is converted into a vector and stored.

### Step 2: Process the question

```text
user question
    |
    v
all-MiniLM-L6-v2
    |
    v
question vector
    |
    v
find the nearest document vectors
    |
    v
retrieve relevant text chunks
```

### Step 3: Generate the answer

```text
question + retrieved chunks
    |
    v
Qwen2.5-3B-Instruct
    |
    v
final answer
```

The embedding model answers:

```text
Which stored passages are semantically related to this question?
```

The language model answers:

```text
Given the question and retrieved passages, what should I say?
```

These are different tasks, so using two different models is normal.

---

## 13. Why the embedding model can stay inside Python

An embedding model such as `all-MiniLM-L6-v2` is much smaller than a 3-billion-parameter generative model.

It is often practical to load it directly:

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

vectors = embedder.encode([
    "Gradient descent reduces the loss.",
    "The model updates its parameters iteratively."
])
```

This is convenient because:

- the model is relatively small;
- embedding inference is fast;
- it often runs acceptably on CPU;
- there is no HTTP serialization step;
- the application can call `encode()` directly.

The architecture becomes:

```text
Python process
├── application logic
├── all-MiniLM-L6-v2
├── vector search
└── HTTP client
        |
        v
   Qwen model service
```

For a small or medium project, my default choice would be:

```text
embedding model
-> load locally inside Python

generative model
-> run as a separate service
```

A separate embedding server becomes useful only when many applications share the model, indexing volume is very large, or high concurrent embedding throughput is required.

---

## 14. One important language warning

The `all-MiniLM-L6-v2` model card identifies it as an English model.

If the documents and questions are mainly English, it is a practical lightweight choice.

If the knowledge base is mainly Chinese or multilingual, I should not assume that an English embedding model will provide the best retrieval quality. I should select a multilingual or Chinese-oriented embedding model and evaluate it using real project queries.

This matters because a powerful answer model cannot repair passages that were never retrieved.

```text
bad retrieval
    -> irrelevant context
    -> weak final answer
```

In many RAG systems, improving retrieval helps more than replacing the answer model with a slightly larger LLM.

---

## 15. Which tool should I choose?

A practical decision guide is:

### Simplest local setup

```text
Ollama + local embedding model
```

Good for a personal RAG application, prototype, or demonstration.

### Maximum control inside Python

```text
Transformers
```

Good for research, fine-tuning, model inspection, and custom inference.

### NVIDIA GPU with higher concurrency

```text
vLLM
```

A strong general-purpose choice for an OpenAI-compatible model server.

### Hugging Face-centered deployment

```text
TGI
```

A natural choice when the surrounding workflow already uses Hugging Face tools.

### Quantization or LLM/VLM deployment

```text
LMDeploy
```

Useful when compression and optimized deployment are both important.

### Repeated prefixes or structured LLM programs

```text
SGLang
```

Especially relevant for prefix-heavy, structured, and multi-step workloads.

---

## 16. Recommended architecture for your project

For my own project, I would start with:

```text
                           ┌──────────────────────────┐
                           │ Qwen2.5-3B-Instruct      │
                           │ Ollama or vLLM service   │
                           └────────────▲─────────────┘
                                        │ HTTP
                                        │
┌───────────────────────────────────────┴─────────────┐
│ Python application                                 │
│                                                     │
│  1. Receive the user question                       │
│  2. Encode it with all-MiniLM-L6-v2                 │
│  3. Search the vector index                         │
│  4. Build a prompt with retrieved passages          │
│  5. Send the prompt to the Qwen HTTP service        │
│  6. Return the generated answer                     │
└─────────────────────────────────────────────────────┘
```

For one local user, this is a simple starting point:

```text
Ollama + Qwen2.5-3B-Instruct
SentenceTransformers + all-MiniLM-L6-v2
```

If the project later has more concurrent users and a dedicated NVIDIA GPU server, the Qwen service can move to:

```text
vLLM
```

without changing the basic RAG logic.

If embedding traffic later becomes large, the embedding model can also move into its own service.

A natural growth path is:

```text
Stage 1:
all components on one computer

Stage 2:
separate the large language model into a model server

Stage 3:
separate embedding, vector search, and generation services

Stage 4:
add replicas, queues, monitoring, and load balancing
```

Do not start at Stage 4 unless the workload actually requires it.

---

## 17. Final mental model

```text
Qwen2.5-3B-Instruct
= generates the answer

all-MiniLM-L6-v2
= converts text into vectors for semantic retrieval

Transformers
= loads and controls models directly in Python

Ollama
= makes local model execution easy

vLLM
= serves LLMs efficiently at high throughput

TGI
= serves models in the Hugging Face ecosystem

LMDeploy
= combines deployment with optimized and quantized inference

SGLang
= serves structured and prefix-heavy workloads efficiently
```

The current mixed architecture is not inconsistent.

It is a useful separation:

```text
small retrieval model
-> direct local Python call

large generation model
-> independent HTTP model service
```

Once this separation is clear, the tool names become much easier to understand.

They are not six different brains.

They are different ways to load, operate, and serve the brain.

---

## References

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index)
- [vLLM documentation](https://docs.vllm.ai/)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [Text Generation Inference documentation](https://huggingface.co/docs/text-generation-inference/index)
- [LMDeploy documentation](https://lmdeploy.readthedocs.io/)
- [SGLang documentation](https://docs.sglang.ai/)
- [Ollama documentation](https://docs.ollama.com/)
- [Sentence Transformers documentation](https://www.sbert.net/)
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Qwen2.5-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
