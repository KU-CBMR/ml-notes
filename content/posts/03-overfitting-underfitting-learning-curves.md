---
title: "03. Overfitting and Underfitting: Read the Learning Curves, Not a Single Score"
date: 2026-05-03
draft: false
categories: ["Machine Learning · LLM · Agent Full-Stack Roadmap"]
tags: ["Overfitting", "Underfitting", "Learning Curves"]
summary: "Diagnose underfitting, overfitting, and optimization failure from training and validation curves instead of labeling a model from one accuracy number."
---

# 03. Overfitting and Underfitting: Read the Learning Curves, Not a Single Score

> **Question this article answers:** Diagnose underfitting, overfitting, and optimization failure from training and validation curves instead of labeling a model from one accuracy number.

“Training accuracy is high, so the model overfits” is an incomplete diagnosis. Overfitting is a relationship among training behavior, validation behavior, capacity, data volume, noise, and distribution—not a fixed accuracy threshold.

## 0. What you should be able to do

- Recognize common training/validation curve patterns.
- Separate underfitting from failed optimization.
- Choose remedies based on evidence rather than habit.
- Use a controlled polynomial example to visualize capacity.

## 1. Build the mental model first

```text
capacity + data + noise + optimization
        ↓
training curve and validation curve
        ↓
underfit / useful fit / overfit / unstable training
```

Keep this data flow in mind. Whenever you meet a formula, library, or new model name, first locate it in the flow instead of memorizing it in isolation.

## 2. Core ideas: from intuition to mechanism

### 2.1 Underfitting means the training structure was not learned

Both training and validation performance remain poor. Possible causes include insufficient capacity, weak features, too much regularization, too few optimization steps, a wrong loss, or implementation bugs.

### 2.2 Overfitting is a widening generalization gap

Training performance continues to improve while validation performance plateaus or deteriorates. The model is fitting details that do not transfer to the evaluation distribution.

### 2.3 Optimization failure can mimic underfitting

A powerful model with a bad learning rate, broken gradient flow, incorrect labels, or numerical instability may perform badly on both splits. Before increasing capacity, verify that a tiny batch can be memorized.

### 2.4 Learning curves help choose the next action

Curves against epochs reveal dynamics; curves against training-set size reveal whether more data is likely to help. Remedies should target the diagnosed cause: data, capacity, regularization, optimization, or split design.

### 2.5 Distribution shift is not ordinary overfitting

A small train–validation gap on a random split does not guarantee future performance. Time, user, device, or domain shifts require evaluation that mirrors deployment.

## 3. Runnable code

Companion file: [`code/03_learning_curve.py`](../code/03_learning_curve.py)

The example is intentionally small. Run it first, inspect every shape and metric, and then change one variable at a time.

```python
"""Generate a small learning-curve table without plotting."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression


def run(seed: int = 42) -> list[dict[str, float]]:
    X, y = make_moons(n_samples=1400, noise=0.28, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X_train))
    rows: list[dict[str, float]] = []
    for n in [40, 80, 160, 320, 640, len(X_train)]:
        idx = order[:n]
        model = make_pipeline(
            PolynomialFeatures(degree=5, include_bias=False),
            StandardScaler(),
            LogisticRegression(C=20.0, max_iter=2000),
        )
        model.fit(X_train[idx], y_train[idx])
        rows.append({
            'n_train': float(n),
            'train_acc': float(accuracy_score(y_train[idx], model.predict(X_train[idx]))),
            'valid_acc': float(accuracy_score(y_valid, model.predict(X_valid))),
        })
    return rows


if __name__ == '__main__':
    for row in run():
        print(row)
```

### Run it

```bash
python code/03_learning_curve.py
```

## 4. Common mistakes and better practice

| Common mistake                                    | Better practice                                                                                                                       |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Using one final accuracy to diagnose fit          | Plot training and validation loss/metrics over time and, when possible, over dataset size.                                            |
| Adding regularization whenever validation is poor | First check leakage, split mismatch, label quality, optimization, and whether the task is learnable.                                  |
| Assuming more epochs always help underfitting     | More epochs help only when optimization is incomplete; they cannot fix bad features, wrong labels, or an inadequate hypothesis space. |

## 5. Required experiments / exercises

- [ ] Run the polynomial example with several degrees and identify underfit, reasonable fit, and overfit cases.
- [ ] Add label noise and observe how the best validation degree changes.
- [ ] Create a learning curve over training-set size and explain whether more data is likely to help.

<!-- ## 6. Interview follow-ups: a stable answer structure

### Q: How do you distinguish underfitting from overfitting?

**Answer:** Underfitting shows poor training and validation performance; overfitting shows strong training performance with worse validation performance. Curves, capacity, data volume, and optimization evidence are considered together.

### Q: What should you inspect before saying a neural network lacks capacity?

**Answer:** Verify input and label correctness, the loss, gradients, parameter updates, learning rate, and whether the network can overfit a very small batch.

### Q: Can a model have high training accuracy without overfitting?

**Answer:** Yes. If validation performance is also strong and representative of deployment, high training accuracy alone is not a problem.

A reliable interview structure is: **one-sentence conclusion → mechanism or data flow → one concrete experiment/project example → limitations and trade-offs**.

## 7. Self-check

- [ ] I can draw the data flow from memory.
- [ ] I can explain the key tensor shapes or data structures.
- [ ] I can name at least two failure modes and how to detect them.
- [ ] I can answer the interview questions in 90 seconds each.
- [ ] I recorded the experiment result and one failed attempt in the README. -->

## 8. References

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

**Previous/next reading:** follow the order in the root `SUMMARY.md`; see Articles 68–70 for the study plans.
