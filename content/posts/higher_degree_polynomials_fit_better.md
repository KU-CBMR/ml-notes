+++
title = "Why Higher-Degree Polynomials Fit the Training Data Better"
date = 2026-02-08
draft = false
categories = ["Classical ML", "Theory"]
tags = ["polynomial-regression", "overfitting", "bias-variance", "hypothesis-space"]
description = "Why increasing polynomial degree enlarges the hypothesis space, making training error non-increasing (but potentially hurting generalization)."
+++

## Why Higher-Degree Polynomials Fit the Training Data Better

Consider a training set:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{n}.
$$

In polynomial regression, we approximate the relationship between $x$ and $y$ with a polynomial of degree $m$:

$$
f_m(x) = w_0 + w_1 x + w_2 x^2 + \cdots + w_m x^m
      = \sum_{k=0}^{m} w_k x^k.
$$

where the coefficients $w_0,\ldots,w_m$ are parameters learned from data.

---

## Hypothesis spaces grow with the degree

Let $\mathcal{H}(m)$ denote the hypothesis space of all polynomials of degree $m$:

$$
\mathcal{H}(m) = \{ f_m(x) = \sum_{k=0}^{m} w_k x^k \mid w_0,\ldots,w_m \in \mathbb{R} \}.
$$

If we compare $\mathcal{H}(1)$ and $\mathcal{H}(2)$, we see that every linear function

$$
f_1(x) = w_0 + w_1 x
$$

is also a quadratic function with $w_2 = 0$:

$$
f_1(x) = w_0 + w_1 x + 0 \cdot x^2 \in \mathcal{H}(2).
$$

More generally,

$$
\mathcal{H}(1) \subset \mathcal{H}(2) \subset \mathcal{H}(3) \subset \cdots
$$

because any lower-degree polynomial can be represented as a higher-degree polynomial by setting the extra coefficients to zero.

This means that as we increase the degree $m$, we enlarge the set of functions the model can choose from. In other words, the model becomes more *flexible*: it can bend and twist in more ways to follow the data.

---

## Training error cannot increase when the degree increases

To make this precise, define the empirical (training) loss for a model $f$ using mean squared error (MSE):

$$
L_{\mathrm{emp}}(f) = \frac{1}{n} \sum_{i=1}^{n} \left(f(x^{(i)}) - y^{(i)}\right)^2.
$$

For degree $m$, the best possible training loss we can achieve within $\mathcal{H}(m)$ is

$$
L_{\mathrm{emp}}^{\star}(m) = \min_{f \in \mathcal{H}(m)} L_{\mathrm{emp}}(f).
$$

Because $\mathcal{H}(m) \subset \mathcal{H}(m+1)$, when we move from degree $m$ to degree $m+1$, we are minimizing over a *larger* set of functions:

$$
L_{\mathrm{emp}}^{\star}(m+1)
= \min_{f \in \mathcal{H}(m+1)} L_{\mathrm{emp}}(f)
\le \min_{f \in \mathcal{H}(m)} L_{\mathrm{emp}}(f)
= L_{\mathrm{emp}}^{\star}(m).
$$

So the optimal training loss as a function of degree is *non-increasing*:

$$
L_{\mathrm{emp}}^{\star}(1) \ge L_{\mathrm{emp}}^{\star}(2) \ge L_{\mathrm{emp}}^{\star}(3) \ge \cdots
$$

Intuitively, a higher-degree polynomial can always imitate the best lower-degree solution (by setting extra coefficients to zero), and in addition it has more freedom to adjust the curve to reduce the training error even further. Therefore, higher-degree polynomials can usually fit the training data better.

---

## Flexibility vs. generalization

However, fitting the training data better does *not* necessarily mean better performance on unseen data. If the degree is too high, the model may start fitting noise and outliers in the training set, a phenomenon known as *overfitting*. In practice, we must balance:

- **Bias:** too-low degree $\Rightarrow$ model too simple, underfits.
- **Variance:** too-high degree $\Rightarrow$ model too flexible, overfits.

In summary: increasing the polynomial degree enlarges the hypothesis space, making the model more flexible. This guarantees that the *best possible* training error cannot get worse and typically gets better, but it also increases the risk of overfitting and poor generalization.
