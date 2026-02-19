---
title: "Ensemble Learning: From Classic ML to Deep Nets and Transformers"
date: 2026-02-19
tags:
  [
    "machine-learning",
    "ensembles",
    "bagging",
    "boosting",
    "random-forests",
    "deep-learning",
    "transformers",
  ]
---

Ensemble learning is the idea of **combining multiple models** to get a predictor that is usually **more accurate, more robust, and better calibrated** than any single model.

If you’ve ever averaged multiple exam scores to reduce randomness, you already understand the intuition: **errors from different models can cancel out**—especially when the models make _different_ mistakes.

## 1) Where ensemble learning came from (a short history)

Ensembles grew out of a simple question:

> _If one model is noisy, can we train several noisy-but-reasonable models and aggregate them?_

This led to:

- **Bagging** (bootstrap aggregating): train many models on bootstrap-resampled data and average/vote. Popularized by Breiman. [1]
- **Boosting**: train models _sequentially_, focusing later models on earlier mistakes (e.g., AdaBoost). [2]
- **Random Forests**: bagging + randomized feature selection for decision trees. [3]
- **Stacking**: learn a **meta-model** that combines the outputs of base models. [4]

In modern practice, ensembles are still everywhere: Kaggle competitions, production ranking systems, safety-critical prediction pipelines—and increasingly in deep learning for **uncertainty estimation**.

---

## 2) Why ensembles work (the core idea)

A useful mental model is the **bias–variance tradeoff**:

- **High variance** models (e.g., deep decision trees) can change a lot if the training data changes slightly.  
  → Ensembles like **bagging** reduce variance by averaging many variants.

- **High bias** models (e.g., shallow trees / weak learners) underfit.  
  → **Boosting** can reduce bias by building a strong learner from many weak ones.

### The one-line math

For regression, the simplest ensemble is an **average**:

$
\hat{y}(x) \;=\; \frac{1}{M}\sum_{m=1}^{M} f_m(x)
$

For classification, a common approach is **average probabilities** (or **vote**):

$
p(y\mid x) \;=\; \frac{1}{M}\sum_{m=1}^{M} p_m(y\mid x)
$

The practical lesson: **ensembles help most when models are accurate _and diverse_.**

---

## 3) The “big three” ensemble families

### A) Bagging (parallel training)

**Training idea:**

1. sample $M$ bootstrap datasets from the training set
2. train one model per dataset
3. average/vote at inference

**Strengths**

- Great for high-variance learners (decision trees).
- Easy to parallelize.
- Often very strong out-of-the-box.

**Weaknesses**

- Can be expensive (many models).
- Doesn’t fix systematic bias as effectively as boosting.

**Classic representative:** **Bagging** (Breiman) [1]
**Practical superstar:** **Random Forests** (Breiman) [3]

### B) Boosting (sequential training)

**Training idea:**  
Train models in sequence; later models focus on examples the earlier ones got wrong. In AdaBoost, you maintain weights over training examples and update them each round. [2]
**Strengths**

- Often achieves very high accuracy.
- Can reduce bias.
- Works surprisingly well with weak learners.

**Weaknesses**

- More sensitive to noisy labels / outliers.
- Less parallelizable (because it’s sequential).
- Can overfit if pushed too far (though modern variants are much more stable).

**Classic representative:** **AdaBoost** (Freund & Schapire). [2]

### C) Stacking (learned combination)

**Training idea:**  
Train base models, then use their predictions (often from cross-validation) as features for a **meta-model**. This is “learning how to combine learners.” [4]

**Strengths**

- Can combine very different model families (trees + linear + neural nets).
- Often strong when you have heterogeneous signals.

**Weaknesses**

- Easy to leak information if you don’t do stacking properly (you must use out-of-fold predictions).
- More engineering complexity.

---

## 4) Ensemble learning in traditional ML (scikit-learn examples)

Below are practical examples with **bagging**, **random forests**, **boosting**, and **stacking**.

> Tip: In classic ML, ensembles often beat single models with **minimal tuning**.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Toy data
X, y = make_classification(
    n_samples=5000, n_features=30, n_informative=10, n_redundant=5,
    class_sep=1.2, random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 1) Bagging (variance reduction)
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None, random_state=0),
    n_estimators=200,
    bootstrap=True,
    n_jobs=-1,
    random_state=0,
)
bag.fit(X_train, y_train)
print("Bagging acc:", accuracy_score(y_test, bag.predict(X_test)))

# 2) Random Forest (bagging + feature randomness)
rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    n_jobs=-1,
    random_state=0
)
rf.fit(X_train, y_train)
print("Random Forest acc:", accuracy_score(y_test, rf.predict(X_test)))

# 3) Gradient Boosting (sequential)
gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)
print("Gradient Boosting acc:", accuracy_score(y_test, gb.predict(X_test)))

# 4) Stacking (heterogeneous combination)
estimators = [
    ("rf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)),
    ("gb", GradientBoostingClassifier(random_state=0)),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    n_jobs=-1
)
stack.fit(X_train, y_train)
print("Stacking acc:", accuracy_score(y_test, stack.predict(X_test)))
```

---

## 5) Ensemble learning in deep neural networks (DNNs)

In deep learning, ensembles are still valuable, but **training cost** is the main barrier.

### A) Deep Ensembles (the modern default)

A **deep ensemble** is usually:

- same architecture
- different random seeds
- (often) different data order or augmentation
- average the predictive probabilities

This surprisingly simple approach provides strong accuracy and **high-quality uncertainty estimates**. [5]

#### Minimal PyTorch-style training sketch

```python
import copy
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 256, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)

def train_one(model, train_loader, epochs=5, lr=1e-3, device="cpu"):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def predict_proba_ensemble(models, x, device="cpu"):
    probs = []
    for m in models:
        m.eval()
        logits = m(x.to(device))
        probs.append(torch.softmax(logits, dim=-1))
    return torch.stack(probs, dim=0).mean(dim=0)  # (B, C)

# Example usage:
# base = MLP(d_in=30, n_classes=2)
# models = [train_one(copy.deepcopy(base), loader, epochs=10, device="cuda") for _ in range(5)]
# proba = predict_proba_ensemble(models, x_batch, device="cuda")
```

**Training:** parallelizable across GPUs/hosts (each model independent).  
**Inference:** you pay ~$M\times$ compute unless you distill/compress (see below).

### B) Snapshot Ensembles (cheaper training)

Snapshot ensembling saves multiple “checkpoints” from one training run using cyclic learning rates. You get multiple diverse models at almost the cost of one run. [6]

### C) Monte-Carlo Dropout (approximate uncertainty)

At inference, keep dropout **on** and run multiple forward passes; average predictions. This connects dropout to approximate Bayesian inference. [7]

---

## 6) Ensembles with Transformers (why, how, tradeoffs)

Transformers are powerful but can be:

- **overconfident** (poor calibration)
- sensitive to data shifts
- expensive to run

Ensembling helps by:

- improving accuracy a bit
- improving robustness
- improving uncertainty estimates

Transformers were introduced in the “Attention Is All You Need” paper. [8]

### A) How transformer ensembling usually works

Most commonly: **logit / probability averaging**.

For a classification model with logits $z_m(x)$ from model $m$:

$
p(y\mid x) \;=\; \frac{1}{M}\sum_{m=1}^{M} \text{softmax}(z_m(x))
$

For text generation (language modeling), you can ensemble next-token distributions:

$
p(t_{k}\mid t_{<k}) \;=\; \frac{1}{M}\sum_{m=1}^{M} p_m(t_{k}\mid t_{<k})
$

This is simple and effective—but costs $M$ forward passes per decoding step, so it can get expensive.

### B) Minimal Hugging Face-style inference sketch (logit averaging)

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@torch.no_grad()
def ensemble_predict(texts, model_names, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_names[0])
    models = [
        AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        for name in model_names
    ]

    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

    probs = []
    for m in models:
        logits = m(**batch).logits
        probs.append(torch.softmax(logits, dim=-1))

    return torch.stack(probs, dim=0).mean(dim=0)  # (B, C)

# Example:
# proba = ensemble_predict(
#   ["This movie was great!", "I hated the ending."],
#   ["distilbert-base-uncased-finetuned-sst-2-english",
#    "textattack/bert-base-uncased-SST-2"],
# )
```

> Note: For **generation**, ensembling is similar but more involved because you ensemble _at every decoding step_.

### C) Making transformer ensembles practical

Because transformer inference is expensive, people often use:

1. **Distillation**: train a _single student model_ to match the ensemble’s outputs.
2. **Speculative / cascaded serving**: cheap model first, expensive ensemble only when uncertain.
3. **Mixture-of-Experts (MoE)** (related concept): route tokens to a subset of experts instead of running all models.

---

## 7) Training vs inference: what actually changes?

### Training-time ensembles (you train multiple models)

- Bagging / Random Forests
- Deep Ensembles
- Snapshot Ensembles (one run, many checkpoints)
- Some stacking setups

**Pros:** often best accuracy/uncertainty  
**Cons:** training cost & storage

### Inference-time ensembles (one trained model, many runs)

- MC Dropout
- Test-time augmentation (TTA) for vision
- Multi-sample decoding tricks

**Pros:** no need to train many models  
**Cons:** inference latency increases, sometimes noisy gains

---

## 8) Strengths and weaknesses (honest tradeoffs)

### Strengths

- **Accuracy**: often improves metrics, especially when the baseline is unstable.
- **Robustness**: less sensitive to quirks of a single training run.
- **Uncertainty / calibration**: deep ensembles are widely used for better uncertainty. [5]
- **Safety**: reduces the risk of a single model’s failure mode dominating.

### Weaknesses

- **Cost**: training and/or inference can multiply by $M$.
- **Latency**: critical for real-time systems (especially transformer generation).
- **Complexity**: deployment, monitoring, and debugging are harder.
- **Diminishing returns**: after a point, adding more similar models helps less.

---

## 9) Practical recipe (what to do in real projects)

1. Start with a strong single model baseline.
2. If variance is high (unstable results), try:
   - bagging / random forests (classic ML),
   - deep ensembles (DNNs),
   - snapshot ensembles (cheaper DNN ensembles).
3. Force diversity:
   - different seeds, augmentations, architectures,
   - different feature sets, different training data slices.
4. If inference cost is too high:
   - distill the ensemble into one model,
   - use a cascade (ensemble only for uncertain cases).

---

## References (high-signal starting points)

- [1] Breiman, **“Bagging Predictors”** (1996).
- [2] Freund & Schapire, **AdaBoost / boosting foundations** (mid-1990s; widely cited 1996/1997 versions).
- [3] Breiman, **“Random Forests”** (2001).
- [4] Wolpert, **“Stacked Generalization”** (1992).
- [5] Lakshminarayanan et al., **Deep Ensembles for predictive uncertainty** (NeurIPS 2017).
- [6] Huang et al., **“Snapshot Ensembles”** (2017).
- [7] Gal & Ghahramani, **Dropout as Bayesian approximation** (2016).
- [8] Vaswani et al., **“Attention Is All You Need”** (Transformer, 2017).

---
