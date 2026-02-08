+++
title = "Why Higher-Degree Polynomials Fit the Training Data Better"
date = 2026-02-08
draft = false
categories = ["Classical ML", "Theory"]
tags = ["polynomial-regression", "overfitting", "bias-variance", "hypothesis-space"]
description = "Why increasing polynomial degree enlarges the hypothesis space, making training error non-increasing (but potentially hurting generalization)."
+++

## Why Higher-Degree Polynomials Fit the Training Data Better

Consider a training set

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n.
$$

In polynomial regression, we approximate the relationship between $x$ and $y$ with a polynomial of degree $m$:

$$
\mathcal{H}_m = \{ f_m(x) = \sum_{k=0}^{m} w_k x^k \mid w_0,\ldots,w_m \in \mathbb{R} \}.
$$


where the coefficients $w_0, \dots, w_m$ are parameters learned from data.

---

## Hypothesis spaces grow with the degree

Let $\mathcal{H}_m$ denote the hypothesis space of all degree-$m$ polynomials:

$$
\mathcal{H}_m =
\left\{ f_m(x) = \sum_{k=0}^{m} w_k x^k \;\big|\; w_0,\ldots,w_m \in \mathbb{R} \right\}.
$$

If we compare $\mathcal{H}_1$ and $\mathcal{H}_2$, we see that every linear function

$$
f_1(x) = w_0 + w_1 x
$$

is also a quadratic function with $w_2 = 0$:

$$
f_1(x) = w_0 + w_1 x + 0 \cdot x^2 \in \mathcal{H}_2.
$$

More generally,

$$
\mathcal{H}_1 \subset \mathcal{H}_2 \subset \mathcal{H}_3 \subset \dots
$$

because any lower-degree polynomial can be represented as a higher-degree polynomial by setting the extra coefficients to zero.

This means that as we increase the degree $m$, we enlarge the set of functions the model can choose from. In other words, the model becomes more *flexible*: it can bend and twist in more ways to follow the data.

---

## Training error cannot increase when the degree increases

To make this precise, define the empirical (training) loss for a model $f$ using mean squared error (MSE):

$$
L_{\text{emp}}(f) = \frac{1}{n} \sum_{i=1}^{n} \left(f(x_i) - y_i\right)^2.
$$

For degree $m$, the best possible training loss we can achieve within $\mathcal{H}_m$ is

$$
L_{\text{emp}}^{*}(m) = \min_{f \in \mathcal{H}_m} L_{\text{emp}}(f).
$$

Because $\mathcal{H}_m \subset \mathcal{H}_{m+1}$, when we move from degree $m$ to degree $m+1$, we are minimizing over a *larger* set of functions:

$$
L_{\text{emp}}^{*}(m+1)
= \min_{f \in \mathcal{H}_{m+1}} L_{\text{emp}}(f)
\le
\min_{f \in \mathcal{H}_m} L_{\text{emp}}(f)
= L_{\text{emp}}^{*}(m).
$$

So the optimal training loss as a function of degree is *non-increasing*:

$$
L_{\text{emp}}^{*}(1) \ge L_{\text{emp}}^{*}(2) \ge L_{\text{emp}}^{*}(3) \ge \dots
$$

Intuitively, a higher-degree polynomial can always imitate the best lower-degree solution (by setting extra coefficients to zero), and in addition it has more freedom to adjust the curve to reduce the training error even further. Therefore, higher-degree polynomials can usually fit the training data better.

---

## Flexibility vs. generalization

However, fitting the training data better does *not* necessarily mean better performance on unseen data. If the degree is too high, the model may start fitting noise and outliers in the training set, a phenomenon known as *overfitting*. In practice, we must balance:

- **Bias:** too-low degree $\Rightarrow$ model too simple, underfits.
- **Variance:** too-high degree $\Rightarrow$ model too flexible, overfits.

In summary: increasing the polynomial degree enlarges the hypothesis space, making the model more flexible. This guarantees that the *best possible* training error cannot get worse and typically gets better, but it also increases the risk of overfitting and poor generalization.
