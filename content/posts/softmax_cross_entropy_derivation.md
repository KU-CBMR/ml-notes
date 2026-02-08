---
title: "Derivation of Softmax Regression and Cross-Entropy Loss"
date: 2026-01-12
draft: false
tags: ["softmax", "cross-entropy", "logits", "backprop", "ml"]
categories: ["ML Basics"]
description: "A clean derivation of Softmax, cross-entropy (MLE + information theory views), and the gradient."
---

# Derivation of Softmax Regression and Cross-Entropy Loss

## The Softmax Function

In multi-class classification (where we have $N$ classes), the neural network outputs a vector of raw scores, known as **logits**, denoted as $z$. Since these scores range from $(-\infty, +\infty)$, they are difficult to interpret. We use the Softmax function to convert them into a probability distribution.

For a specific class $k$, the predicted probability $a_k$ is defined as:

- $a_k = \operatorname{Softmax}(z)_k = \dfrac{e^{z_k}}{\sum_{j=1}^{N} e^{z_j}}$


**Properties:**

- $a_k \in (0, 1)$
- $\sum_{k=1}^{N} a_k = 1$

---

## Derivation of the Loss Function

Why do we define the loss function as $L = -\log(a_y)$? There are two primary mathematical perspectives.

### Perspective 1: Maximum Likelihood Estimation (MLE)

This is the statistical perspective.

1. **Assumption:** We assume that the samples in our dataset are **Independent and Identically Distributed (i.i.d.)**.
2. **Goal:** We want to find the model parameters that maximize the probability of the observed data (the ground truth labels).
3. **Likelihood:** For a single sample with the correct label $y$, the likelihood is simply the predicted probability of that class:

   $$
   \text{Likelihood} = a_y.
   $$

   For the entire dataset, the joint probability is the product of individual probabilities:

   $$
   \mathcal{L}(\theta) = \prod_{i=1}^{m} a_{y^{(i)}}.
   $$

4. **Log-Likelihood:** Maximizing a product is numerically unstable (prone to underflow). Since $\log(x)$ is monotonically increasing, maximizing the likelihood is equivalent to maximizing the log-likelihood:

   $$
   \log \mathcal{L}(\theta) = \sum_{i=1}^{m} \log(a_{y^{(i)}}).
   $$

5. **Minimizing Loss:** In optimization, we conventionally minimize a “loss” rather than maximizing a “score”. Therefore, we take the negative:

   $$
   J(\theta) = - \sum_{i=1}^{m} \log(a_{y^{(i)}}).
   $$

   For a single sample, this simplifies to:

   $$
   L = -\log(a_y).
   $$

### Perspective 2: Information Theory (Cross-Entropy)

This perspective views classification as measuring the difference between two probability distributions.

- **True Distribution $P$:** This is the one-hot encoded vector of the ground truth. If the correct class is $y$, then $P(y)=1$ and $P(k)=0$ for all $k \neq y$.
- **Predicted Distribution $Q$:** This is the output of the Softmax function, where $Q(k) = a_k$.

The **Cross-Entropy** measures the distance between $P$ and $Q$:

$$
H(P, Q) = - \sum_{k=1}^{N} P(k) \log Q(k).
$$

Since $P(k)$ is non-zero (equal to 1) only when $k$ is the correct class $y$:

$$
H(P, Q) = - \left[ 1 \cdot \log(a_y) + \sum_{k \neq y} 0 \cdot \log(a_k) \right] = -\log(a_y).
$$

---

## Gradient Derivation (Backpropagation)

To train the model, we need the derivative of the loss $L$ with respect to the logits $z_i$.

Using the Chain Rule:

$$
\frac{\partial L}{\partial z_i} = \frac{\partial L}{\partial a_y} \cdot \frac{\partial a_y}{\partial z_i}.
$$

After differentiating the Softmax function and the Log function, the result is elegantly simple:

$$
\frac{\partial L}{\partial z_i} = a_i - y_i,
$$

where $y_i$ is the one-hot label (1 if $i$ is the correct class, 0 otherwise).

- For the **correct class** ($i=y$): gradient is $a_y - 1$ (negative, pushes $z_y$ up).
- For **incorrect classes** ($i \neq y$): gradient is $a_i - 0 = a_i$ (positive, pushes $z_i$ down).

---

## Why Do We Ignore Incorrect Classes in the Loss Formula?

A common question is:

> “Why does the formula $L = -\log(a_y)$ only look at the correct class? Doesn’t the model need to be penalized for assigning high probabilities to wrong classes?”

The answer lies in the **zero-sum** nature of the Softmax function:

$$
\sum_{k=1}^{N} a_k = 1.
$$

Since the total probability mass is fixed at 1.0, probability works like a see-saw (or a zero-sum game). You cannot increase the probability of the correct class ($a_y$) without strictly decreasing the probabilities of the incorrect classes ($a_{\text{wrong}}$).

Therefore:

- By explicitly maximizing $a_y$, we are **implicitly** minimizing all other $a_k$ (where $k \neq y$).
- The gradient derivation confirms this: the derivative for incorrect classes is $a_i$. If an incorrect class has a high probability (e.g., $a_{\text{dog}}=0.4$ when it is actually a cat), the gradient is positive ($0.4$), which directs the optimizer to reduce that score.
