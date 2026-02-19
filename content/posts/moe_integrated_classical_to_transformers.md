---
title: "Mixture of Experts (MoE): From Classical ML to Sparse Transformers (with Code)"
date: 2026-02-19
tags:
  [
    "moe",
    "mixture-of-experts",
    "transformers",
    "routing",
    "pytorch",
    "cifar10",
    "ensemble-learning",
  ]
---

Mixture of Experts (MoE) is a modeling idea where **multiple specialized sub-models (“experts”)** are combined using a **gating / routing function**. The _same_ high-level concept appears in:

1. **Traditional ML MoE** (dense mixing: use _all_ experts for every input)
2. **Deep-learning MoE** (sparse mixing / conditional computation: use only _a few_ experts per input)
3. **Transformer MoE** (sparse MoE replaces the Transformer FFN to scale model capacity efficiently)

This post explains all three, shows simple formulas, and includes two runnable code examples:

- a **classical ML MoE** (logistic experts + softmax gate, EM-style training)
- a **PyTorch CIFAR-10 sparse MoE** (top-1 routing + load balancing)

---

## 1) Core idea and notation

You have:

- experts: $\(f_1(x), f_2(x), \dots, f_E(x)\)$
- a gate/router that outputs non-negative weights \(w_i(x)\)

A common “canonical” MoE combination is the weighted sum:

$
f(x) = \sum\_{i=1}^{E} w_i(x)\, f_i(x), \quad w_i(x)\ge 0
$

This basic structure (experts + gating function + combination) is described in standard references. [1]

---

## 2) Traditional ML MoE (dense mixing)

### What it is

In **classical MoE**, each input uses **all experts**. The gate produces weights for _every_ expert, and you compute a full weighted sum for each query. [1] [3]

This is great when:

- you want **interpretable specialization** (“expert 2 handles region A of the feature space”)
- compute cost is manageable (small number of experts)

### How it’s trained (typical approach: EM)

Wikipedia notes MoE can be trained using an **Expectation–Maximization (EM)**-style procedure, similar to other mixture models:

- **E-step**: assign “responsibility” of each data point to experts
- **M-step**: update experts + gate to better fit those responsibilities [1]

---

## 3) Deep-learning MoE (sparse routing / conditional computation)

### What changes vs classical MoE?

Modern deep-learning MoE is often used for **conditional computation**: to reduce compute, each input routes to **only a small subset of experts** (often top-1 or top-2). [1] [2] [3]

So instead of summing over all $E$ experts, we sum only over a selected set $S(x)$ with $|S(x)| = k \ll E$:

$
f(x) = \sum\_{i\in S(x)} \tilde{w}\_i(x)\, f_i(x), \quad |S(x)| = k
$

This is exactly the point we emphasized: **routing becomes the key design choice** in deep-learning MoE—how to route a batch of queries to the best experts while controlling cost and avoiding bottlenecks. [1]

### Why load balancing matters

Sparse routing can collapse: the router may send most inputs to a few “popular” experts, leaving others idle.

As a result, modern MoE designs often add **auxiliary load-balancing losses** (or routing constraints) to keep expert usage more uniform. Wikipedia explicitly calls out load-balancing issues in deep-learning MoE and discusses auxiliary losses used to address them. [1]

---

## 4) Transformer MoE: why replace the FFN?

### Where MoE goes in the Transformer

In many Transformer blocks, the dense **feed-forward network (FFN)** is a large compute + parameter hotspot. MoE usually replaces _that_ FFN with:

- a router (gate)
- a bank of FFN experts

Wikipedia notes that MoE layers are typically used to select the Transformer feed-forward layers and are commonly sparsely gated (e.g., k=1 or k=2). [1]

### Why Transformers use MoE (the motivation)

**Goal:** increase total model capacity (parameters) while keeping _per-token_ compute roughly similar.

Benefits:

- **Better “capacity per FLOP”**: big total parameter count, but only a few experts active per token
- **Specialization**: experts can learn to handle different patterns (e.g., code vs prose), though this is emergent and not guaranteed
- **Faster training at scale** (in the right infra): Switch Transformer reports large speedups with simplified routing. [6]

Trade-offs:

- **Engineering complexity** (routing + token dispatch + all-to-all communication)
- **Load balancing + stability** issues
- **Serving latency** can be worse for small batches

---

## 5) Representative MoE architectures (classic → modern)

Below are “standout” MoE representatives commonly cited in the modern literature and ecosystem:

- **Sparsely-Gated MoE (Shazeer et al., 2017)**: introduced sparse top-k gating for conditional computation at large scale. [4]
- **GShard (Lepikhin et al., 2020)**: combined conditional computation (MoE) with large-scale sharding/parallelism tooling for giant Transformers. [5]
- **Switch Transformer (Fedus et al., 2021/2022)**: simplified to **top-1** routing to reduce compute/communication overhead and improve practicality. [6]
- **V-MoE (Riquelme et al., 2021)**: brought sparse MoE ideas to vision Transformers with strong scaling and compute trade-offs. [7]
- **Mixtral (Mistral, 2023)**: a high-profile open sparse MoE LLM (top-2 per token) that popularized MoE in the open model community. [8]

For a practical, system-level discussion of MoE building blocks and serving trade-offs, the Hugging Face MoE overview is also a helpful companion. [9]

---

## 6) Code 1 — Traditional ML MoE (logistic experts + softmax gate, EM-like training)

This is a compact educational implementation for **binary classification**:

- each expert is logistic regression
- the gate is softmax over experts
- training alternates:
  - **E-step** responsibilities
  - **M-step** gradient steps for experts + gate

```python
import numpy as np

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


class LogisticExpert:
    def __init__(self, d):
        self.w = 0.01 * np.random.randn(d)

    def prob1(self, X):
        return sigmoid(X @ self.w)

    def grad(self, X, y, r):
        # Weighted logistic regression gradient (negative log-likelihood)
        p = self.prob1(X)
        g = (r * (p - y)) @ X
        return g / (X.shape[0] + 1e-12)


class SoftmaxGate:
    def __init__(self, d, K):
        self.W = 0.01 * np.random.randn(d, K)

    def probs(self, X):
        return softmax(X @ self.W)

    def grad(self, X, R):
        # Gradient of negative log-likelihood: X^T (P - R)
        P = self.probs(X)
        G = X.T @ (P - R)
        return G / (X.shape[0] + 1e-12)


class ClassicalMoE_Binary:
    # p(y=1|x) = sum_k gate_k(x) * expert_k(x)
    # EM-like alternating updates.
    def __init__(self, d, K=3, lr=0.5, steps_m=50, seed=0):
        np.random.seed(seed)
        self.K = K
        self.experts = [LogisticExpert(d) for _ in range(K)]
        self.gate = SoftmaxGate(d, K)
        self.lr = lr
        self.steps_m = steps_m

    def e_step(self, X, y):
        P_gate = self.gate.probs(X)          # [N, K]
        N = X.shape[0]
        like = np.zeros((N, self.K))

        for k, ex in enumerate(self.experts):
            pk = ex.prob1(X)
            like[:, k] = (pk ** y) * ((1 - pk) ** (1 - y))  # Bernoulli likelihood

        numer = P_gate * like
        R = numer / (numer.sum(axis=1, keepdims=True) + 1e-12)
        return R

    def m_step(self, X, y, R):
        for _ in range(self.steps_m):
            for k, ex in enumerate(self.experts):
                ex.w -= self.lr * ex.grad(X, y, R[:, k])
            self.gate.W -= self.lr * self.gate.grad(X, R)

    def fit(self, X, y, iters=30):
        y = y.astype(float)
        for t in range(1, iters + 1):
            R = self.e_step(X, y)
            self.m_step(X, y, R)
            print(f"iter {t:02d} | loglike {self.log_likelihood(X, y):.4f}")

    def predict_proba(self, X):
        P_gate = self.gate.probs(X)
        P = np.column_stack([ex.prob1(X) for ex in self.experts])  # [N, K]
        return (P_gate * P).sum(axis=1)

    def log_likelihood(self, X, y):
        p = np.clip(self.predict_proba(X), 1e-9, 1 - 1e-9)
        return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


if __name__ == "__main__":
    np.random.seed(0)
    N, d = 2000, 5
    X = np.random.randn(N, d)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    y[X[:, 2] > 1.0] = 1 - y[X[:, 2] > 1.0]  # flip in a region

    model = ClassicalMoE_Binary(d=d, K=3, lr=0.8, steps_m=30)
    model.fit(X, y, iters=20)
    pred = (model.predict_proba(X) >= 0.5).astype(int)
    print("train acc:", (pred == y).mean())
```

**Key property:** every input uses all experts (dense mixing), which is the “classical” MoE flavor. [1] [3]

---

## 7) Code 2 — CIFAR-10 Sparse MoE in PyTorch (top-1 routing + balancing loss)

This example:

- creates **multiple CNN experts**
- learns a **router** that picks **one expert per image**
- adds a small **load-balancing loss** to avoid collapse

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNNExpert(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4 * width, num_classes)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.head(h)


class RouterBackbone(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 4 * width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.feat(x).flatten(1)


class Top1MoECIFAR(nn.Module):
    def __init__(self, num_experts: int = 4, num_classes: int = 10, width: int = 64):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes

        self.backbone = RouterBackbone(width=width)
        d = 4 * width
        self.router = nn.Linear(d, num_experts, bias=False)
        self.experts = nn.ModuleList([CNNExpert(num_classes=num_classes, width=width) for _ in range(num_experts)])

    def forward(self, x):
        feats = self.backbone(x)
        router_logits = self.router(feats)

        top1_idx = torch.argmax(router_logits, dim=-1)
        top1_weight = F.softmax(router_logits, dim=-1).gather(1, top1_idx[:, None])

        B = x.size(0)
        out = x.new_zeros((B, self.num_classes))

        for e, expert in enumerate(self.experts):
            mask = (top1_idx == e)
            if not mask.any():
                continue
            out[mask] = top1_weight[mask] * expert(x[mask])

        return out, router_logits, top1_idx


def load_balancing_loss(router_logits: torch.Tensor, top1_idx: torch.Tensor, num_experts: int):
    probs = F.softmax(router_logits, dim=-1)
    importance = probs.sum(dim=0)
    load = torch.bincount(top1_idx, minlength=num_experts).float()

    importance = importance / (importance.sum() + 1e-9)
    load = load / (load.sum() + 1e-9)

    return (importance * load).sum() * (num_experts ** 2)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = Top1MoECIFAR(num_experts=4, num_classes=10, width=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    aux_weight = 0.01
    epochs = 10

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, router_logits, top1_idx = model(x)

            ce = F.cross_entropy(logits, y)
            lb = load_balancing_loss(router_logits, top1_idx, model.num_experts)
            loss = ce + aux_weight * lb

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()

        acc = evaluate(model, test_loader, device)
        print(f"epoch {ep:02d} | train_loss {running/len(train_loader):.4f} | test_acc {acc:.4f}")


if __name__ == "__main__":
    main()
```

This “top-1 routing + balancing” pattern matches the modern sparse MoE theme: **activate only a small subset of experts per query** and use auxiliary signals to keep routing stable. [1] [6]

---

## 8) Practical checklist: when MoE is a good idea

MoE is attractive if you have:

- large-scale training infra (multi-GPU / multi-node)
- big batch sizes (routing overhead amortized)
- a need for bigger _capacity_ under a compute budget

Dense models are often simpler if you care about:

- small-batch latency
- minimal engineering complexity
- deployment on limited hardware

---

## References

<!-- [1] Wikipedia, “Mixture of experts”. https://en.wikipedia.org/wiki/Mixture_of_experts
[2] IBM Think, “What is mixture of experts?”. https://www.ibm.com/think/topics/mixture-of-experts
[3] DataCamp, “What Is Mixture of Experts (MoE)?”. https://www.datacamp.com/blog/mixture-of-experts-moe
[4] Shazeer et al. (2017): “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”
[5] Lepikhin et al. (2020): “GShard”.
[6] Fedus et al. (2021/2022): “Switch Transformers”.
[7] Riquelme et al. (2021): “Scaling Vision with Sparse Mixture of Experts (V-MoE)”.
[8] Mistral (2023) Mixtral.
[9] Hugging Face MoE explainer. -->

## References

- [1] Wikipedia, “Mixture of experts”. https://en.wikipedia.org/wiki/Mixture_of_experts
- [2] IBM Think, “What is mixture of experts?”. https://www.ibm.com/think/topics/mixture-of-experts
- [3] DataCamp, “What Is Mixture of Experts (MoE)?”. https://www.datacamp.com/blog/mixture-of-experts-moe
- [4] Shazeer et al. (2017): “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”.
- [5] Lepikhin et al. (2020): “GShard”.
- [6] Fedus et al. (2021/2022): “Switch Transformers”.
- [7] Riquelme et al. (2021): “Scaling Vision with Sparse Mixture of Experts (V-MoE)”.
- [8] Mistral (2023): “Mixtral”.
- [9] Hugging Face: “MoE explainer”.
