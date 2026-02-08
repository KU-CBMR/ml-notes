+++
title = "Full Derivation of Logistic Regression Gradient Descent"
date = 2026-02-08
draft = false
math = true
categories = ["Classical ML", "Optimization"]
tags = ["logistic-regression", "gradient-descent", "derivation", "ml-math"]
description = "A full, step-by-step derivation of logistic regression gradients and gradient descent updates."
+++

## Model and Notation

The training set is

$$
\left\{ \bigl(x^{(i)}, y^{(i)}\bigr) \mid i = 1,2,\dots,m \right\}
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
y^{(i)} \in \\{0,1\\}.
$$

The parameters are

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

The logistic regression model is

$$
z^{(i)} = w^\top x^{(i)} + b,
\qquad
f_{w,b}\bigl(x^{(i)}\bigr) = g\!\left(z^{(i)}\right),
$$

where $g(z)$ is the sigmoid function:

$$
g(z) = \frac{1}{1 + e^{-z}}.
$$

For brevity, we will often write

$$
f^{(i)} \triangleq f_{w,b}\bigl(x^{(i)}\bigr),
\qquad
z^{(i)} = w^\top x^{(i)} + b.
$$

## Cost Function

The loss (log loss) for a single training example $(x^{(i)}, y^{(i)})$ is defined as

$$
\ell^{(i)}(w,b)
= - \Bigl[
   y^{(i)} \log f^{(i)}
   + (1 - y^{(i)}) \log (1 - f^{(i)})
  \Bigr].
$$

The overall (average) cost function is

$$
J(w,b)
= \frac{1}{m} \sum_{i=1}^m \ell^{(i)}(w,b)
= - \frac{1}{m} \sum_{i=1}^m
\Bigl[
y^{(i)} \log f^{(i)} + (1 - y^{(i)}) \log (1 - f^{(i)})
\Bigr].
$$

Our goal is to derive

$$
\frac{\partial J}{\partial w_j}
\quad\text{and}\quad
\frac{\partial J}{\partial b},
$$

and then write down the gradient descent update rules.

## Sigmoid Derivative: $g'(z) = g(z)\bigl(1-g(z)\bigr)$

Start from the definition:

$$
g(z) = \frac{1}{1 + e^{-z}}.
$$

Rewrite it as a power:

$$
g(z) = \bigl(1 + e^{-z}\bigr)^{-1}.
$$

Differentiate with respect to $z$:

$$
g'(z)
= -1 \cdot \bigl(1 + e^{-z}\bigr)^{-2}
   \cdot \frac{\mathrm{d}}{\mathrm{d}z}\bigl(1 + e^{-z}\bigr).
$$

Note that

$$
\frac{\mathrm{d}}{\mathrm{d}z}\bigl(1 + e^{-z}\bigr) = - e^{-z}.
$$

Substituting:

$$
\begin{aligned}
g'(z)
&= - \bigl(1 + e^{-z}\bigr)^{-2} \cdot (-e^{-z})\\[3pt]
&= \frac{e^{-z}}{\bigl(1 + e^{-z}\bigr)^2}.
\end{aligned}
$$

Rewrite using $g(z)$:

$$
1 - g(z)
= 1 - \frac{1}{1 + e^{-z}}
= \frac{e^{-z}}{1 + e^{-z}}.
$$

Therefore,

$$
\begin{aligned}
g(z)\bigl(1 - g(z)\bigr)
&= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}\\[3pt]
&= \frac{e^{-z}}{\bigl(1 + e^{-z}\bigr)^2}.
\end{aligned}
$$

Hence

$$
\boxed{g'(z) = g(z)\bigl(1 - g(z)\bigr)}.
$$

## Gradient of a Single Example w.r.t. $w_j$

Start from the single-example loss:

$$
\ell^{(i)}(w,b)
= - \Bigl[
   y^{(i)} \log f^{(i)}
   + (1 - y^{(i)}) \log (1 - f^{(i)})
  \Bigr].
$$

### Step 1: Differentiate w.r.t. $f^{(i)}$

Temporarily treat $f^{(i)}$ as a scalar $f$:

$$
\ell(f) = - \bigl[ y \log f + (1-y)\log(1-f) \bigr].
$$

Differentiate:

$$
\begin{aligned}
\frac{\mathrm{d}\ell}{\mathrm{d}f}
&= -\left[
    y \cdot \frac{1}{f}
    + (1-y)\cdot \frac{1}{1-f} \cdot \frac{\mathrm{d}}{\mathrm{d}f}(1-f)
   \right]\\[3pt]
&= -\left[
    \frac{y}{f}
    + (1-y)\cdot \frac{1}{1-f} \cdot (-1)
   \right]\\[3pt]
&= -\left[
    \frac{y}{f}
    - \frac{1-y}{1-f}
   \right]\\[3pt]
&= -\frac{y}{f} + \frac{1-y}{1-f}.
\end{aligned}
$$

Combine:

$$
\begin{aligned}
\frac{\mathrm{d}\ell}{\mathrm{d}f}
&= \frac{1-y}{1-f} - \frac{y}{f}\\[3pt]
&= \frac{(1-y)f - y(1-f)}{f(1-f)}\\[3pt]
&= \frac{f - y}{f(1-f)}.
\end{aligned}
$$

Switch back:

$$
\frac{\partial \ell^{(i)}}{\partial f^{(i)}}
= \frac{f^{(i)} - y^{(i)}}{f^{(i)}\bigl(1 - f^{(i)}\bigr)}.
$$

### Step 2: $∂ f^{(i)}/∂ z^{(i)}$

$$
\frac{\partial f^{(i)}}{\partial z^{(i)}}
= f^{(i)}\bigl(1 - f^{(i)}\bigr).
$$

### Step 3: $∂ z^{(i)}/∂ w_j$

$$
\frac{\partial z^{(i)}}{\partial w_j}
= x_j^{(i)}.
$$

### Step 4: Chain rule

$$
\begin{aligned}
\frac{\partial \ell^{(i)}}{\partial w_j}
&= \frac{\partial \ell^{(i)}}{\partial f^{(i)}}
   \cdot \frac{\partial f^{(i)}}{\partial z^{(i)}}
   \cdot \frac{\partial z^{(i)}}{\partial w_j}\\[3pt]
&= \frac{f^{(i)} - y^{(i)}}{f^{(i)}\bigl(1 - f^{(i)}\bigr)}
   \cdot f^{(i)}\bigl(1 - f^{(i)}\bigr)
   \cdot x_j^{(i)}\\[3pt]
&= (f^{(i)} - y^{(i)}) x_j^{(i)}.
\end{aligned}
$$

So

$$
\boxed{
\frac{\partial \ell^{(i)}}{\partial w_j}
= \bigl(f^{(i)} - y^{(i)}\bigr) x_j^{(i)}
}.
$$

## Gradient of the Overall Cost w.r.t. $w_j$

$$
\frac{\partial J}{\partial w_j}
= \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr) x_j^{(i)}.
$$

## Gradient w.r.t. the Bias $b$

Single example:

$$
\frac{\partial \ell^{(i)}}{\partial b}
= f^{(i)} - y^{(i)}.
$$

Overall:

$$
\frac{\partial J}{\partial b}
= \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr).
$$

## Gradient Descent Update Rules

For learning rate $\alpha$:

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j},
\qquad
b := b - \alpha \frac{\partial J}{\partial b}.
$$

Substituting the gradients:

$$
w_j := w_j
 - \alpha \left[
   \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr) x_j^{(i)}
 \right]
$$

$$
b := b
 - \alpha \left[
   \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr)
 \right].
$$

In vector form:

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
