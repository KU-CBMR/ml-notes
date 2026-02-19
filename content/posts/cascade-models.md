---
title: "Cascade Models: Multi-Stage Learning for Accuracy and Efficiency"
date: 2026-02-16
tags:
  ["ml", "ensemble", "efficiency", "computer-vision", "transformers", "cascade"]
categories: ["ML Systems Notes"]
---

Cascade models (or **model cascades**) are a simple idea with a big impact:

> **Don’t spend the full compute budget on every example.**  
> Use a _sequence_ of stages. Easy cases exit early; hard cases go deeper.

You’ve probably seen cascades in face detection, object detection, ranking/retrieval, and “early-exit” Transformers. This post explains **where cascades came from**, **why they work**, **how to train them**, and **how inference differs**—with code you can adapt.

---

## 1) What is a cascade?

A cascade is a **pipeline of models** (or model heads) indexed by stage $k = 1..K$.

- Stage 1 is **cheap** and tries to accept/reject easy examples.
- Later stages are **more expensive** and handle ambiguous cases.
- A **gate** decides whether to **exit** or **continue**.

A common abstraction:

$
\hat{y}(x) = f\_{k^_(x)}(x), \quad k^_(x) = \min \{k \;|\; g_k(x) = \text{exit}\}
$

where:

- $f_k$ is the predictor at stage $k$
- $g_k$ is a gating rule (often confidence-based)

### Why does this help?

The expected inference cost becomes:

$
\mathbb{E}[\text{cost}] = \sum\_{k=1}^{K} \Pr(\text{reach stage } k)\cdot \text{cost}\_k
$

If many samples exit early, **average latency and FLOPs drop a lot**—without sacrificing much accuracy.

---

## 2) Where cascades came from (traditional ML)

### 2.1 The classic

The breakthrough “rapid object detection” system (popularly known for face detection) used:

- **A cascade of stages**
- Each stage is a **strong classifier** built by AdaBoost over simple features
- Early stages are tuned for **very high recall**, rejecting obvious negatives fast
- Later stages focus on **hard negatives**

This is the canonical “real-time cascade” story.

**Key training idea:** _stage-wise hard negative mining_  
After training stage $k$, run it on background images, collect false positives, and train stage \$k+1$ to reject them.

(Reference: Viola & Jones, 2001) [1]

### 2.2 Cascade forests (gcForest / deep forest)

Cascades aren’t just linear “reject/accept” chains. You can build **deep-ish models** by stacking non-neural blocks.

The **gcForest (multi-grained cascade forest)** builds layers of tree ensembles where each layer transforms features and feeds the next. Model depth can be chosen based on validation performance.

(References: Zhou & Feng, 2017; Zhou, 2019) [2] [3]

---

## 3) Cascades in modern deep learning (DNNs)

### 3.1 Multi-stage detectors: Cascade R-CNN

In object detection, there’s a subtle issue: if you train with a low IoU threshold, you get noisy positives; but training with high IoU reduces positives and can overfit.

**Cascade R-CNN** addresses this by training a _sequence_ of detectors with **increasing IoU thresholds**, using the outputs of earlier stages to create better training samples for later ones.

- Stage 1: learns to handle coarse proposals
- Stage 2/3: increasingly selective, higher-quality localization

(References: Cai & Vasconcelos, 2018; 2019) [4] [5]

### 3.2 Early-exit networks (“anytime prediction”)

Instead of multiple separate models, you can put **exit heads** at intermediate layers:

- If confidence is high → exit early
- Otherwise → compute deeper layers

This idea is used in vision and NLP to trade accuracy for latency on a per-example basis.

---

## 4) Cascades for Transformers

Transformers are accurate but expensive. Cascades show up in two common forms:

### 4.1 Early exiting for BERT-style models (DeeBERT)

**DeeBERT** adds intermediate classifier heads to BERT and lets samples exit early based on confidence, saving inference time with minimal quality loss.

(References: Xin et al., 2020; ACL Anthology / arXiv) [6] [7]

### 4.2 Cascades for ranking / retrieval

Ranking often uses a _candidate generation_ stage (cheap) and a _re-ranking_ stage (expensive). Some systems further cascade Transformers by dropping/early-exiting items progressively (e.g., “Cascade Transformer” style approaches discussed in the IR literature).

(Reference: early-exit ranking work that cites “Cascade Transformer”, e.g., Xin et al., 2020) [6] [7]

---

## 5) How do you decide when to exit?

The gate $g_k(x)$ is everything. Common strategies:

1. **Confidence threshold**
   - Exit if $\max_c p_k(c|x) \ge \tau_k$
2. **Margin**
   - Exit if $p_1 - p_2 \ge \tau_k$ (top-1 minus top-2)
3. **Learned gate**
   - Train a small gating network to predict whether going deeper will change the decision
4. **Budget-aware**
   - Adjust thresholds $\tau_k$ to match a target latency/FLOPs budget

A practical tip: **calibrate** probabilities (temperature scaling) before setting thresholds, otherwise your “confidence” can be misleading.

---

## 6) Training cascades: what changes vs. a single model?

### 6.1 Stage-wise training (classic cascade)

Common in boosted cascades and many multi-stage pipelines:

1. Train stage 1 to achieve high recall.
2. Run stage 1, collect mistakes (hard negatives / hard cases).
3. Train stage 2 on the filtered, harder distribution.
4. Repeat.

**Pros**

- Simple, stable, good when “hard negative mining” matters

**Cons**

- Later stages depend on earlier stages’ behavior
- Training can be slow if you repeatedly re-run inference to mine hard cases

### 6.2 Joint training (multi-exit deep nets)

For early-exit DNNs/Transformers, you often train with **multiple losses**:

$
\mathcal{L} = \sum\_{k=1}^{K} \lambda_k \, \mathcal{L}\_k
$

- $\mathcal{L}\_k$ is the loss at exit $k$
- $\lambda_k$ balances how much you care about early vs. late exits

You can also use **knowledge distillation**:

- Train a full model (“teacher”)
- Train intermediate exits (“students”) to match teacher logits

### 6.3 Threshold tuning (deployment step)

Even after training, you typically choose thresholds $\tau_k$ on a validation set to satisfy:

- Target accuracy drop ≤ ε
- Target average latency/FLOPs ≤ budget

This is often where most “systems” wins come from.

---

## 7) Inference: how it actually runs

A typical online inference loop:

1. Run stage 1
2. If confident → return
3. Else run stage 2
4. …

This gives two key operational properties:

- **Average latency** improves if most cases are easy
- **Tail latency** can worsen if hard cases queue up (you still have a worst-case path)

In production, you often monitor:

- exit distribution (how many samples exit at each stage)
- accuracy by stage
- “hard-case rate” drift (a sign of data shift)

---

## 8) Code: a simple ML cascade with early exit (scikit-learn)

Below is a minimal two-stage cascade:

- Stage 1: very fast linear model
- Stage 2: stronger model for uncertain cases

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# toy data
X, y = make_classification(n_samples=5000, n_features=30, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# stage 1 (fast)
m1 = LogisticRegression(max_iter=2000).fit(X_train, y_train)

# stage 2 (stronger)
m2 = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_train, y_train)

def cascade_predict(X, tau=0.90):
    # If stage-1 confidence >= tau, exit early; otherwise defer to stage-2.
    p1 = m1.predict_proba(X)
    conf = p1.max(axis=1)
    y1 = p1.argmax(axis=1)

    use_stage2 = conf < tau
    y = y1.copy()
    if np.any(use_stage2):
        y[use_stage2] = m2.predict(X[use_stage2])
    return y, use_stage2.mean()

y_pred, frac_stage2 = cascade_predict(X_test, tau=0.90)
print("accuracy:", accuracy_score(y_test, y_pred))
print("fraction sent to stage-2:", frac_stage2)
```

**What to tune**

- `tau` controls the speed/quality trade-off
- choose `tau` by sweeping it on a validation set

---

## 9) Code: early-exit Transformer idea (PyTorch sketch)

A simplified pattern for early exit in a Transformer encoder:

- Add classifier heads at intermediate layers
- Exit when confidence is high

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyExitEncoder(nn.Module):
    def __init__(self, encoder_layers, hidden_dim, num_classes, exit_layers=(3, 6, 9, 12)):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layers)
        self.exit_layers = set(exit_layers)
        self.heads = nn.ModuleDict({str(i): nn.Linear(hidden_dim, num_classes) for i in exit_layers})

    @torch.no_grad()
    def forward(self, x, tau=0.9):
        # x: [B, T, H]
        # Returns: (logits, exit_layer_index)
        for i, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if i in self.exit_layers:
                logits = self.heads[str(i)](x[:, 0])  # e.g., [CLS]
                probs = F.softmax(logits, dim=-1)
                conf, _ = probs.max(dim=-1)

                # exit if ALL samples are confident (batch-wise gating for simplicity)
                if torch.all(conf >= tau):
                    return logits, i

        last = max(self.exit_layers)
        logits = self.heads[str(last)](x[:, 0])
        return logits, last
```

**How to train**

- attach losses at each exit head (multi-loss training)
- optionally distill intermediate heads from the final head

---

## 10) Strengths and weaknesses

### Strengths

- **Lower average latency / cost** (big win in production)
- **Anytime behavior** (useful under dynamic budgets)
- **Better specialization** (later stages focus on hard cases)
- **Modular** (swap stages, retrain one stage, etc.)

### Weaknesses

- **Thresholding can be brittle** under distribution shift (exit rates drift)
- **Worst-case latency** remains (tail can be worse)
- **More moving parts** (monitoring, calibration, stage compatibility)
- **Train/deploy mismatch risk** (if the gate differs between training and deployment)

---

## 11) Notable cascade architectures (quick list)

- **Viola–Jones boosted cascade** (real-time object/face detection) [1]
- **gcForest / deep forest** (cascade of tree ensembles)
- **Cascade R-CNN / Cascade Mask R-CNN** (multi-stage detectors)
- **Early-exit Transformers (e.g., DeeBERT)**

---

## References

- [1] P. Viola, M. Jones. _Rapid Object Detection using a Boosted Cascade of Simple Features._ CVPR 2001. PDF: https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
- [2] Z.-H. Zhou, J. Feng. _Deep Forest._ arXiv:1702.08835, 2017. https://arxiv.org/abs/1702.08835
- [3] Z.-H. Zhou. _Deep forest._ National Science Review, 2019. https://academic.oup.com/nsr/article/6/1/74/5123737
- [4] Z. Cai, N. Vasconcelos. _Cascade R-CNN: Delving into High Quality Object Detection._ CVPR 2018. PDF: https://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf
- [5] Z. Cai, N. Vasconcelos. _Cascade R-CNN: High Quality Object Detection and Instance Segmentation._ arXiv:1906.09756, 2019. https://arxiv.org/abs/1906.09756
- [6] J. Xin et al. _DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference._ ACL 2020. https://aclanthology.org/2020.acl-main.204/
- [7] J. Xin et al. _Early Exiting BERT for Efficient Document Ranking._ PDF: https://cs.uwaterloo.ca/~y328yu/publication/xin-nyl-20/xin-nyl-20.pdf
