+++
title = "Clustering Accuracy (ACC) with the Hungarian Algorithm"
date = 2026-02-12
draft = false
tags = ["clustering", "evaluation", "hungarian", "unsupervised"]
categories = ["Machine Learning"]
description = "Compute clustering accuracy (ACC) by optimally matching cluster IDs to ground-truth class IDs using the Hungarian algorithm."
+++

Clustering algorithms output **arbitrary cluster IDs**.  
That makes “accuracy” non-trivial: cluster `0` is not guaranteed to correspond to class `0`, etc.

This post shows how to compute **Clustering Accuracy (ACC)** for unsupervised clustering **when ground-truth labels are available for evaluation** (e.g., benchmarks and datasets with labels).

---

## Requirements

Given:

- `y_true`: ground-truth class labels (shape `(n,)`)
- `y_pred`: predicted cluster assignments (shape `(n,)`)

We want an accuracy-like score in `[0, 1]` that:

- is **invariant to permutations** of cluster IDs,
- works when labels are **not contiguous** or **do not start at 0**,
- avoids memory blow-ups when label values are large (e.g., `100, 200, 300`),
- can optionally handle “noise/unassigned” labels like `-1` (common in DBSCAN-like methods).

---

## Why Naive Accuracy Fails

Suppose we have:

- `y_true = [0, 0, 1, 1]`
- `y_pred = [1, 1, 0, 0]`

This is a perfect clustering (just swapped IDs), but naive accuracy compares IDs directly and returns `0.0`.

So we must first find the **best mapping** from cluster IDs → class IDs.

---

## Core Idea: Hungarian Algorithm on a Count Matrix

1. Build a **count matrix** `W`:

- rows = clusters
- columns = true classes
- `W[i, j]` = number of samples assigned to cluster `i` and belonging to class `j`

2. Find a one-to-one assignment that **maximizes matches**:

$$
ACC = \frac{\max_{\pi} \sum_i W[i, \pi(i)]}{n}
$$

This is a **linear assignment problem**, solvable with the **Hungarian algorithm**.
In Python, use `scipy.optimize.linear_sum_assignment`, which returns the optimal assignment.

---

## A Robust Implementation (Recommended)

This version fixes common pitfalls in “ACC” snippets found online:

- label values may be non-contiguous or start at -1,
- label values may be very large (wastes memory if used directly as indices),
- inputs may come from lists, pandas, or torch (we convert via NumPy).

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(y_true, y_pred, digits=4, ignore_noise=False, noise_label=-1):
    """
    Clustering ACC via optimal assignment (Hungarian algorithm).

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth labels (only used for evaluation).
    y_pred : array-like, shape (n,)
        Cluster assignments.
    digits : int
        Rounding digits for the returned score.
    ignore_noise : bool
        If True, drops samples where y_true or y_pred equals noise_label.
    noise_label : int
        The label value used for noise/unassigned points (commonly -1).

    Returns
    -------
    float
        Clustering accuracy in [0, 1], rounded to `digits`.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.size != y_pred.size:
        raise ValueError(f"y_true and y_pred must have the same length, got {y_true.size} vs {y_pred.size}")

    # Optional: ignore noise / unassigned points (e.g., DBSCAN uses -1)
    if ignore_noise:
        mask = (y_true != noise_label) & (y_pred != noise_label)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if y_true.size == 0:
        return 0.0

    # Compress labels to 0..K-1 to avoid huge sparse matrices
    _, y_true_c = np.unique(y_true, return_inverse=True)
    _, y_pred_c = np.unique(y_pred, return_inverse=True)

    n_true = int(y_true_c.max() + 1)
    n_pred = int(y_pred_c.max() + 1)
    D = max(n_true, n_pred)

    # Build count matrix
    W = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred_c.size):
        W[y_pred_c[i], y_true_c[i]] += 1

    # Hungarian solves min-cost; convert to cost by subtracting from max
    row_ind, col_ind = linear_sum_assignment(W.max() - W)
    acc = W[row_ind, col_ind].sum() / y_pred_c.size
    return round(float(acc), digits)
```

---

## Minimal Example

```python
y_true = [0, 0, 1, 1, 2, 2]
y_pred = [2, 2, 0, 0, 1, 1]  # perfect clustering but permuted IDs

print(clustering_accuracy(y_true, y_pred))  # 1.0
```

---

## Using ACC with Other Clustering Metrics

ACC is often reported with:

- **ARI**: Adjusted Rand Index
- **NMI**: Normalized Mutual Information
- **Silhouette**: uses only `X` and `y_pred` (does not need `y_true`)

Example evaluation helper:

```python
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def evaluate_clustering(X, y_true, y_pred):
    sil = round(silhouette_score(X, y_pred), 4)
    ari = round(adjusted_rand_score(y_true, y_pred), 4)
    nmi = round(normalized_mutual_info_score(y_true, y_pred), 4)
    acc = clustering_accuracy(y_true, y_pred, digits=4)
    return {"sil": sil, "ari": ari, "nmi": nmi, "acc": acc}
```

---

## Common Pitfalls

1. **Noise label `-1` (DBSCAN/HDBSCAN)**
   - Decide whether to *keep* noise points or *drop* them (`ignore_noise=True`).

2. **Huge / sparse label values**
   - Always compress labels (the function above does this).

3. **ACC is not “purely unsupervised”**
   - It uses `y_true` for evaluation only.
   - That’s fine for benchmarks but not available in truly unlabeled deployments.

---


---

## Should These Metrics Be Maximized or Minimized?

Here’s the “direction” for the metrics commonly reported together:

- **ACC (Clustering Accuracy)**: **higher is better**, range **[0, 1]**.  
  `1.0` means a perfect clustering *after* the optimal cluster→class matching.

- **ARI (Adjusted Rand Index)**: **higher is better**, typically in **[-1, 1]** (often **[0, 1]** in practice).  
  `1.0` is best; values near `0` indicate random-like agreement; negative values can mean worse-than-random agreement.

- **NMI (Normalized Mutual Information)**: **higher is better**, range **[0, 1]**.  
  `1.0` indicates perfect agreement; `0` indicates little to no mutual information.

- **Silhouette Score**: **higher is better**, range **[-1, 1]**.  
  - close to **1**: points are well matched to their own cluster and far from others (good separation)  
  - around **0**: clusters overlap / are not well separated  
  - **negative**: many points may be assigned to the “wrong” cluster (poor clustering)

### Practical Notes

- **Silhouette does not require ground-truth labels**; it uses only features `X` and predicted clusters `y_pred`.  
  **ACC / ARI / NMI require `y_true`** and are therefore evaluation metrics for labeled benchmarks.

- Avoid comparing **Silhouette** across runs where the **feature scaling** differs (e.g., standardized vs. raw features), because the score depends on distances.


## Summary

To compute clustering accuracy (ACC) correctly:

- Build a cluster-vs-class count matrix.
- Use the Hungarian algorithm to find the best cluster→class mapping.
- Divide matched counts by `n`.

This makes ACC invariant to cluster ID permutations and robust to real-world label formats.
