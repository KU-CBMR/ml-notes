---
title: "01. What Machine Learning Is Really Doing: The Complete Loop from Data to Generalization"
date: 2026-05-01
draft: false
categories: ["Machine Learning · LLM · Agent Full-Stack Roadmap"]
tags: ["Machine Learning Foundations", "Generalization", "Experiment Design"]
summary: "Use a runnable binary-classification example to connect data, hypothesis space, loss, optimization, validation, and business decisions."
---

# 01. What Machine Learning Is Really Doing: The Complete Loop from Data to Generalization

> **Question this article answers:** Use a runnable binary-classification example to connect data, hypothesis space, loss, optimization, validation, and business decisions.

It is tempting to describe machine learning as “give a table to a model and call `fit`.” Most serious mistakes happen before and after that call: the samples may not represent the future, the label may encode leaked information, and the optimized loss may not match the real decision cost.

## 0. What you should be able to do

- Draw the end-to-end data flow of a machine-learning project.
- Separate the training objective from the evaluation metric and the business decision.
- Explain empirical risk, expected risk, hypothesis space, and generalization.
- Write and inspect a minimal reproducible training program.

## 1. Build the mental model first

```text
Real world → data-generating process → samples and labels → train/validation/test split
          → preprocessing → model fθ(x) → loss → optimization
          → offline evaluation → threshold/cost decision → production feedback
```

Keep this data flow in mind. Whenever you meet a formula, library, or new model name, first locate it in the flow instead of memorizing it in isolation.

## 2. Core ideas: from intuition to mechanism

### 2.1 Define the task before choosing the model

Write down the input available at prediction time, the required output, the user of that output, and the cost of each error. A model cannot repair a poorly defined target or use information that will not exist when the prediction is made.

### 2.2 Empirical risk is only a proxy

Training minimizes average loss on a finite sample. What we really care about is expected loss on future data. Because that distribution is not directly observable, trustworthy splitting and evaluation are part of the learning problem, not administrative details.

### 2.3 The hypothesis space controls what can be learned

Linear models, trees, kernels, and neural networks represent different families of functions. Their inductive biases affect sample efficiency, robustness, interpretability, and the kinds of relationships they can express.

### 2.4 Prediction and decision are different layers

A model may output a score or probability. A downstream policy converts it into an action using thresholds, capacity constraints, risk tolerance, and costs. A threshold of 0.5 is a convention, not a law.

### 2.5 A complete loop includes failure analysis

A credible project stores data versions, random seeds, configuration, metrics, failure cases, and scope conditions. A single best score is not enough evidence that the system works.

## 3. Runnable code

Companion file: [`code/01_ml_closed_loop.py`](../code/01_ml_closed_loop.py)

The example is intentionally small. Run it first, inspect every shape and metric, and then change one variable at a time.

```python
"""A minimal, reproducible supervised-learning loop."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Result:
    train_accuracy: float
    valid_accuracy: float
    valid_log_loss: float


def run(seed: int = 42) -> Result:
    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        class_sep=1.2,
        random_state=seed,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    valid_prob = model.predict_proba(X_valid)[:, 1]
    return Result(
        train_accuracy=accuracy_score(y_train, train_prob >= 0.5),
        valid_accuracy=accuracy_score(y_valid, valid_prob >= 0.5),
        valid_log_loss=log_loss(y_valid, valid_prob),
    )


if __name__ == '__main__':
    print(run())
```

### Run it

```bash
python code/01_ml_closed_loop.py
```

## 4. Common mistakes and better practice

| Common mistake                            | Better practice                                                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Starting with “Which model should I use?” | Start with how the data is generated, when the label becomes available, and what information exists at prediction time. |
| Treating loss as business value           | Track training loss, offline metrics, and decision cost separately; document why they are related but not identical.    |
| Tuning repeatedly on the test set         | Use validation data for choices and keep the test set untouched until the design is frozen.                             |

## 5. Required experiments / exercises

- [ ] Draw your current project as a data–model–decision diagram and mark every possible leakage path.
- [ ] Add the data version, random seed, preprocessing parameters, and metrics to the example program.
- [ ] Write one sentence explaining how your project loss differs from the real business objective.

<!-- ## 6. Interview follow-ups: a stable answer structure

### Q: What is the difference between empirical risk and expected risk?

**Answer:** Empirical risk is average loss on the observed training sample. Expected risk is average loss under the true data distribution. We optimize the former and use data design, regularization, and independent evaluation to obtain a low value of the latter.

### Q: Why does low training loss not prove the model is good?

**Answer:** The model may overfit, exploit leakage, face a shifted distribution, or optimize a loss that is poorly aligned with the decision cost. Independent evaluation and error analysis are required.

### Q: What should be written first in a machine-learning project?

**Answer:** Write the task definition and data-generating process first, followed by the split, baseline, metrics, and success criteria. Model selection comes later.

A reliable interview structure is: **one-sentence conclusion → mechanism or data flow → one concrete experiment/project example → limitations and trade-offs**. -->

## 7. Self-check

- [ ] I can draw the data flow from memory.
- [ ] I can explain the key tensor shapes or data structures.
- [ ] I can name at least two failure modes and how to detect them.
- [ ] I can answer the interview questions in 90 seconds each.
- [ ] I recorded the experiment result and one failed attempt in the README.

## 8. References

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

**Previous/next reading:** follow the order in the root `SUMMARY.md`; see Articles 68–70 for the study plans.
