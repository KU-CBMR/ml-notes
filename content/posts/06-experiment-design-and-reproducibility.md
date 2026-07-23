---
title: "06. Experiment Design and Reproducibility: Change One Variable at a Time"
date: 2026-05-06
draft: false
categories: ["Machine Learning · LLM · Agent Full-Stack Roadmap"]
tags: ["Reproducibility", "Ablation", "Experiment Tracking"]
summary: "Turn model experiments into auditable evidence by freezing data, code, environment, seeds, metrics, and comparison rules."
---

# 06. Experiment Design and Reproducibility: Change One Variable at a Time

> **Question this article answers:** Turn model experiments into auditable evidence by freezing data, code, environment, seeds, metrics, and comparison rules.

An experiment is not reproducible merely because a notebook ran once. Another person must be able to reconstruct the data, configuration, software environment, and decision rule—and obtain results within the expected statistical variation.

## 0. What you should be able to do

- Define a controlled comparison and a valid baseline.
- Record the minimum metadata needed to reproduce a run.
- Report variation instead of a single lucky seed.
- Design ablations that support causal claims.

## 1. Build the mental model first

```text
question → hypothesis → fixed controls → one changed factor
        → repeated runs → metrics + uncertainty → failure analysis → conclusion
```

Keep this data flow in mind. Whenever you meet a formula, library, or new model name, first locate it in the flow instead of memorizing it in isolation.

## 2. Core ideas: from intuition to mechanism

### 2.1 Write the hypothesis before running the experiment

State what is changed, what remains fixed, which metric should move, and what result would falsify the claim. This prevents post-hoc storytelling.

### 2.2 Reproducibility needs full provenance

Record data version and split, code commit, environment, hardware, model/tokenizer/configuration, random seeds, and exact evaluation script. A model name and score are insufficient.

### 2.3 A baseline defines the value of complexity

Use a simple, credible system under the same data and evaluation conditions. Comparing against a weak or differently tuned baseline does not isolate the contribution.

### 2.4 Repeated runs reveal statistical noise

Random initialization, batch order, sampling, and nondeterministic kernels create variation. Report mean, standard deviation, confidence intervals, and paired differences when appropriate.

### 2.5 Ablation is controlled subtraction

Remove or replace one component while keeping the rest constant. If multiple things change together, the result cannot tell which factor caused the difference.

## 3. Runnable code

Companion file: [`code/23_reproducible_experiment.py`](../code/23_reproducible_experiment.py)

The example is intentionally small. Run it first, inspect every shape and metric, and then change one variable at a time.

```python
"""Capture enough metadata to reproduce a small experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import platform
import subprocess
import sys
import numpy as np
import torch


@dataclass(frozen=True)
class ExperimentRecord:
    seed: int
    python: str
    platform: str
    numpy: str
    torch: str
    git_commit: str | None
    metrics: dict[str, float]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def save_record(path: Path, seed: int, metrics: dict[str, float]) -> None:
    record = ExperimentRecord(
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        numpy=np.__version__,
        torch=torch.__version__,
        git_commit=git_commit(),
        metrics=metrics,
    )
    path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    output = Path(__file__).with_name('experiment_record.example.json')
    save_record(output, 42, {'valid_accuracy': 0.91})
    print(output)
```

### Run it

```bash
python code/23_reproducible_experiment.py
```

## 4. Common mistakes and better practice

| Common mistake                               | Better practice                                                             |
| -------------------------------------------- | --------------------------------------------------------------------------- |
| Saving only the best checkpoint and score    | Store all runs, seeds, failures, configuration, and selection criteria.     |
| Changing model, data, and metric together    | Change one factor or use a factorial design that can separate interactions. |
| Calling a one-seed difference an improvement | Repeat runs and quantify uncertainty before claiming a reliable gain.       |

## 5. Required experiments / exercises

- [ ] Use the experiment script to produce two runs that differ in exactly one parameter.
- [ ] Create an experiment record containing commit, environment, data hash, split, seed, and metrics.
- [ ] Design an ablation table for one project and state what each row tests.

<!-- ## 6. Interview follow-ups: a stable answer structure

### Q: What makes an ML experiment reproducible?

**Answer:** Another person can reconstruct the data, split, code, dependencies, configuration, hardware assumptions, seed, and evaluation procedure, and obtain a result within expected variation.

### Q: Why should you change one variable at a time?

**Answer:** It isolates the effect of the tested factor. When multiple variables change, improvement cannot be attributed to a specific cause.

### Q: How should random-seed variation be reported?

**Answer:** Use several predefined seeds and report the distribution—typically mean, standard deviation, and a confidence interval or paired comparison—not only the maximum.

A reliable interview structure is: **one-sentence conclusion → mechanism or data flow → one concrete experiment/project example → limitations and trade-offs**.

## 7. Self-check

- [ ] I can draw the data flow from memory.
- [ ] I can explain the key tensor shapes or data structures.
- [ ] I can name at least two failure modes and how to detect them.
- [ ] I can answer the interview questions in 90 seconds each.
- [ ] I recorded the experiment result and one failed attempt in the README.

## 8. References

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

**Previous/next reading:** follow the order in the root `SUMMARY.md`; see Articles 68–70 for the study plans. -->
