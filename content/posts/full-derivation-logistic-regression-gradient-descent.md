+++
title = "Logistic Regression Gradients (Derivation)"
date = 2026-02-08
draft = false
categories = ["Classical ML", "Theory"]
tags = ["logistic-regression", "gradient-descent", "derivation"]
description = "Derive the gradients of logistic regression (log loss) w.r.t. weights and bias, and write the gradient descent updates."
+++

## Model and Notation

Training set:

$$
\{(x^{(i)}, y^{(i)}) \mid i = 1,2,\dots,m\}
$$


where

$$
x^{(i)} =
\begin{bmatrix}
x^{(i)}_1\\
x^{(i)}_2\\
\vdots\\
x^{(i)}_n
\end{bmatrix}
\in \mathbb{R}^n,
\qquad
y^{(i)} \in \{0,1\}.
$$

Parameters:

$$
w =
\begin{bmatrix}
w_1\\
w_2\\
\vdots\\
w_n
\end{bmatrix}
\in \mathbb{R}^n,
\qquad
b \in \mathbb{R}.
$$

Logistic regression model:

$$
z^{(i)} = w^\top x^{(i)} + b,
\qquad
f_{w,b}\bigl(x^{(i)}\bigr) = g\!\left(z^{(i)}\right),
$$

where the sigmoid function is

$$
g(z) = \frac{1}{1 + e^{-z}}.
$$

For brevity:

$$
f^{(i)} \triangleq f_{w,b}\bigl(x^{(i)}\bigr),
\qquad
z^{(i)} = w^\top x^{(i)} + b.
$$

---

## Cost Function

Single-example loss (log loss) for $(x^{(i)}, y^{(i)})$: (1)

$$
\ell^{(i)}(w,b) = -\left[y^{(i)}\log f^{(i)} + (1-y^{(i)})\log(1-f^{(i)})\right].
$$


Overall (average) cost:

$$
J(w,b)
= \frac{1}{m} \sum_{i=1}^m \ell^{(i)}(w,b)
= - \frac{1}{m} \sum_{i=1}^m
\Bigl[
y^{(i)} \log f^{(i)} + (1 - y^{(i)}) \log (1 - f^{(i)})
\Bigr].
\qquad (2)
$$

Goal: derive

$$
\frac{\partial J}{\partial w_j}
\quad\text{and}\quad
\frac{\partial J}{\partial b},
$$

and then write down the gradient descent update rules.

---

## Sigmoid Derivative: $g'(z) = g(z)\bigl(1-g(z)\bigr)$

### Step 1: Differentiate the definition

Start from:

$$
g(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}.
$$


Differentiate:

$$
g'(z)
= -1 \cdot (1 + e^{-z})^{-2}
   \cdot \frac{d}{dz}(1 + e^{-z}).
$$


Since

$$
\frac{\mathrm{d}}{\mathrm{d}z}\bigl(1 + e^{-z}\bigr)= -e^{-z},
$$

we get

$$
g'(z)= \frac{e^{-z}}{\bigl(1 + e^{-z}\bigr)^2}.
$$

### Step 2: Rewrite in terms of \(g(z)\)

From

$$
g(z) = \frac{1}{1 + e^{-z}},
$$

we have

$$
1 - g(z)= \frac{e^{-z}}{1 + e^{-z}}.
$$

Thus,

$$
g(z)\bigl(1 - g(z)\bigr)
= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}
= \frac{e^{-z}}{\bigl(1 + e^{-z}\bigr)^2}.
$$

Therefore,

$$
\boxed{g'(z) = g(z)\bigl(1 - g(z)\bigr)}.
\qquad (3)
$$

---

## Gradient of a Single Example w.r.t. $w_j$: $\partial \ell^{(i)}/\partial w_j$

Start from (1):

$$
\ell^{(i)}(w,b)
= - \Bigl[
   y^{(i)} \log f^{(i)}
   + (1 - y^{(i)}) \log (1 - f^{(i)})
  \Bigr].
$$

### Step 1: \(\partial \ell^{(i)}/\partial f^{(i)}\)

Treat \(f^{(i)}\) as scalar \(f\):

$$
\ell(f) = - \bigl[ y \log f + (1-y)\log(1-f) \bigr].
$$

Differentiate:

$$
\frac{\mathrm{d}\ell}{\mathrm{d}f}
= -\frac{y}{f} + \frac{1-y}{1-f}
= \frac{f-y}{f(1-f)}.
$$

So,

$$
\frac{\partial \ell^{(i)}}{\partial f^{(i)}}
= \frac{f^{(i)} - y^{(i)}}{f^{(i)}\bigl(1 - f^{(i)}\bigr)}.
\qquad (4)
$$

### Step 2: \(\partial f^{(i)}/\partial z^{(i)}\)

Since \(f^{(i)} = g(z^{(i)})\), using (3):

$$
\frac{\partial f^{(i)}}{\partial z^{(i)}}
= f^{(i)}\bigl(1 - f^{(i)}\bigr).
\qquad (5)
$$

### Step 3: \(\partial z^{(i)}/\partial w_j\)

$$
z^{(i)} = w^\top x^{(i)} + b = \sum_{k=1}^n w_k x_k^{(i)} + b,
$$

thus,

$$
\frac{\partial z^{(i)}}{\partial w_j} = x_j^{(i)}.
\qquad (6)
$$

### Step 4: Chain rule

By the chain rule,

$$
\frac{\partial \ell^{(i)}}{\partial w_j}
= \frac{\partial \ell^{(i)}}{\partial f^{(i)}}
  \cdot
  \frac{\partial f^{(i)}}{\partial z^{(i)}}
  \cdot
  \frac{\partial z^{(i)}}{\partial w_j}.
$$

Substitute (4)(5)(6):

$$
\frac{\partial \ell^{(i)}}{\partial w_j}
= \frac{f^{(i)} - y^{(i)}}{f^{(i)}(1-f^{(i)})}
   \cdot f^{(i)}(1-f^{(i)})
   \cdot x_j^{(i)}
= (f^{(i)} - y^{(i)}) x_j^{(i)}.
$$

So,

$$
\boxed{
\frac{\partial \ell^{(i)}}{\partial w_j}
= \bigl(f^{(i)} - y^{(i)}\bigr) x_j^{(i)}
}.
\qquad (7)
$$

---

## Gradient of the Overall Cost w.r.t. \(w_j\): \(\partial J/\partial w_j\)

Because

$$
J(w,b) = \frac{1}{m} \sum_{i=1}^m \ell^{(i)}(w,b),
$$

we have

$$
\frac{\partial J}{\partial w_j}
= \frac{1}{m} \sum_{i=1}^m
   \frac{\partial \ell^{(i)}}{\partial w_j}.
$$

Substitute (7):

$$
\boxed{
\frac{\partial J}{\partial w_j}
= \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr) x_j^{(i)}
}.
\qquad (8)
$$

---

## Gradient w.r.t. Bias \(b\): \(\partial J/\partial b\)

### Step 1: Single-example gradient \(\partial \ell^{(i)}/\partial b\)

Again by the chain rule,

$$
\frac{\partial \ell^{(i)}}{\partial b}
= \frac{\partial \ell^{(i)}}{\partial f^{(i)}}
  \cdot
  \frac{\partial f^{(i)}}{\partial z^{(i)}}
  \cdot
  \frac{\partial z^{(i)}}{\partial b}.
$$

Since

$$
\frac{\partial z^{(i)}}{\partial b} = 1,
\qquad (9)
$$

we get

$$
\frac{\partial \ell^{(i)}}{\partial b}
= \frac{f^{(i)} - y^{(i)}}{f^{(i)}(1-f^{(i)})}
   \cdot f^{(i)}(1-f^{(i)})
   \cdot 1
= f^{(i)} - y^{(i)}.
$$

Thus,

$$
\boxed{
\frac{\partial \ell^{(i)}}{\partial b}
= f^{(i)} - y^{(i)}
}.
\qquad (10)
$$

### Step 2: Overall gradient

$$
\boxed{
\frac{\partial J}{\partial b}
= \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr)
}.
\qquad (11)
$$

---

## Gradient Descent Update Rules

Generic update:

$$
\theta := \theta - \alpha \frac{\partial J}{\partial \theta},
$$

where \(\alpha\) is the learning rate.

So,

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j},
\qquad
b   := b   - \alpha \frac{\partial J}{\partial b}.
\qquad (12)
$$

Substitute (8) and (11):

$$
w_j := w_j
 - \alpha \left[
   \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr) x_j^{(i)}
 \right],
$$

$$
b := b
 - \alpha \left[
   \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr)
 \right].
$$

Vector form:

$$
w := w - \alpha \cdot \frac{1}{m}
      \sum_{i=1}^m
      \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr) x^{(i)},
$$

$$
b := b - \alpha \cdot \frac{1}{m}
      \sum_{i=1}^m
      \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr).
$$
