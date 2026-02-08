---
title: "Softmax + Cross-Entropy: Gradient Derivation"
date: 2026-01-14
draft: false
tags: ["softmax", "cross-entropy", "gradient", "backprop", "ml"]
categories: ["ML Basics"]
description: "A step-by-step derivation of the Softmax + Cross-Entropy gradient."
---

# Softmax + Cross-Entropy: Gradient Derivation

This post is a continuation of the previous note on Softmax and cross-entropy.  
Here we derive the gradient of the single-example loss with respect to each logit $z_i$.


---

## The Goal

We want the gradient of the loss $L$ with respect to the raw score (logit) $z_i$:

- $\frac{\partial L}{\partial z_i} = ?$

---

## Setup and Definitions

- **Logits ($z$):** the raw outputs from the neural network layer.
- **Probabilities ($a$):** the output after Softmax.
  - $a_k = \frac{e^{z_k}}{\Sigma}, \quad \Sigma = \sum_{j=1}^{N} e^{z_j}$
- **Loss function ($L$):** cross-entropy loss for a single example where $y$ is the index of the true class.
  - $L = -\log(a_y)$

---

## Step 1: The Chain Rule

Since $L$ is a function of $a$, and $a$ is a function of $z$, apply the chain rule:

- $\frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial a_y}\cdot\frac{\partial a_y}{\partial z_i}$

> Note: Strictly speaking, $L$ depends on all $a_k$, but with the simplified cross-entropy form $L=-\log(a_y)$, only the term for the correct class remains.

### Part A: Derivative of Loss w.r.t. Probability

- $\frac{\partial L}{\partial a_y} = \frac{\partial (-\log a_y)}{\partial a_y} = -\frac{1}{a_y}$

---

## Step 2: Derivative of Softmax (The Tricky Part)

We need $\frac{\partial a_k}{\partial z_i}$.

Recall:

- $a_k = \frac{e^{z_k}}{\Sigma}$
- $\Sigma = \sum_{j=1}^{N} e^{z_j}$

First compute the derivative of $\Sigma$ with respect to $z_i$:

- $\frac{\partial \Sigma}{\partial z_i} = e^{z_i}$

Now consider two cases.

### Case 1: k equals i

(That is, $k=i$.)

- $\frac{\partial a_i}{\partial z_i}
= \frac{e^{z_i}\Sigma - e^{z_i}e^{z_i}}{\Sigma^2}
= \frac{e^{z_i}}{\Sigma}\cdot\frac{\Sigma - e^{z_i}}{\Sigma}
= a_i(1-a_i)$

### Case 2: k not equal to i

(That is, $k\neq i$.)

- $\frac{\partial a_k}{\partial z_i}
= \frac{0\cdot\Sigma - e^{z_k}e^{z_i}}{\Sigma^2}
= -\frac{e^{z_k}}{\Sigma}\cdot\frac{e^{z_i}}{\Sigma}
= -a_k a_i$

---

## Step 3: Combine Everything

Plug the Softmax derivative into the chain rule:

- $\frac{\partial L}{\partial z_i} = -\frac{1}{a_y}\cdot\frac{\partial a_y}{\partial z_i}$

We evaluate two scenarios.

### Scenario A: i equals y (correct class)

(That is, $i=y$.)

- $\frac{\partial L}{\partial z_y}
= \left(-\frac{1}{a_y}\right)\cdot a_y(1-a_y)
= -(1-a_y)
= a_y - 1$

### Scenario B: i not equal to y (incorrect classes)

(That is, $i\neq y$.)

- $\frac{\partial L}{\partial z_i}
= \left(-\frac{1}{a_y}\right)\cdot(-a_y a_i)
= a_i$

---

## Final Conclusion

Unify the two cases using the one-hot label $y_i$ (where $y_i=1$ if $i=y$, else $0$):

- $\frac{\partial L}{\partial z_i} = a_i - y_i$

### Interpretation

- If prediction matches truth ($a_i \approx y_i$), then gradient $\approx 0$ (no change needed).
- If prediction is far from truth, the gradient magnitude is large (bigger update needed).
