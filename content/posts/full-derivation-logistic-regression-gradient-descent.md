+++
title = "Full Derivation of Logistic Regression Gradient Descent"
date = 2026-02-08
draft = false
categories = ["Classical ML", "Optimization"]
tags = ["logistic-regression", "gradient-descent", "derivation", "ml-math"]
description = "A full, step-by-step derivation of the logistic regression gradients and gradient descent update rules."
+++

## Model and Notation

The training set is

$$
\left\{ \bigl(x^{(i)}, y^{(i)}\bigr) \mid i = 1,2,\dots,m \right\},
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
\tag{eq:single-loss}
$$

The overall (average) cost function is

$$
J(w,b)
= \frac{1}{m} \sum_{i=1}^m \ell^{(i)}(w,b)
= - \frac{1}{m} \sum_{i=1}^m
\Bigl[
y^{(i)} \log f^{(i)} + (1 - y^{(i)}) \log (1 - f^{(i)})
\Bigr].
\tag{eq:J}
$$

Our goal is to derive

$$
\frac{\partial J}{\partial w_j}
\quad\text{and}\quad
\frac{\partial J}{\partial b},
$$

and then write down the gradient descent update rules.

## Sigmoid Derivative: $g'(z) = g(z)\bigl(1-g(z)\bigr)$

### Step 1: Differentiate the definition

Start from the definition:

$$
g(z) = \frac{1}{1 + e^{-z}}.
$$

Rewrite it as a power to apply the chain rule easily:

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
\frac{\mathrm{d}}{\mathrm{d}z}\bigl(1 + e^{-z}\bigr)
= 0 + \frac{\mathrm{d}}{\mathrm{d}z} e^{-z}
= - e^{-z}.
$$

Substituting, we obtain

$$
\begin{align*}
g'(z)
&= - \bigl(1 + e^{-z}\bigr)^{-2} \cdot (-e^{-z})\\[3pt]
&= \frac{e^{-z}}{\bigl(1 + e^{-z}\bigr)^2}.
\end{align*}
$$

### Step 2: Rewrite the derivative using $g(z)$

From the definition of $g(z)$,

$$
g(z) = \frac{1}{1 + e^{-z}},
$$

we get

$$
1 - g(z)
= 1 - \frac{1}{1 + e^{-z}}
= \frac{1 + e^{-z} - 1}{1 + e^{-z}}
= \frac{e^{-z}}{1 + e^{-z}}.
$$

Therefore,

$$
\begin{align*}
g(z)\bigl(1 - g(z)\bigr)
&= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}\\[3pt]
&= \frac{e^{-z}}{\bigl(1 + e^{-z}\bigr)^2}.
\end{align*}
$$

This is exactly the same expression as we obtained for $g'(z)$, hence

$$
\boxed{g'(z) = g(z)\bigl(1 - g(z)\bigr)}.
\tag{eq:gprime}
$$

## Gradient of a Single Example w.r.t. $w_j$: $\partial \ell^{(i)}/\partial w_j$

Start from the single-example loss:

$$
\ell^{(i)}(w,b)
= - \Bigl[
   y^{(i)} \log f^{(i)}
   + (1 - y^{(i)}) \log (1 - f^{(i)})
  \Bigr].
$$

### Step 1: Differentiate $\ell^{(i)}$ w.r.t. $f^{(i)}$

Temporarily treat $f^{(i)}$ as a scalar variable $f$ and write

$$
\ell(f) = - \bigl[ y \log f + (1-y)\log(1-f) \bigr].
$$

Differentiate with respect to $f$:

$$
\begin{align*}
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
\end{align*}
$$

Combine the two fractions:

$$
\begin{align*}
\frac{\mathrm{d}\ell}{\mathrm{d}f}
&= \frac{1-y}{1-f} - \frac{y}{f}\\[3pt]
&= \frac{(1-y)f - y(1-f)}{f(1-f)}\\[3pt]
&= \frac{f - yf - y + yf}{f(1-f)}\\[3pt]
&= \frac{f - y}{f(1-f)}.
\end{align*}
$$

Switch back to $f^{(i)}$ and $\ell^{(i)}$:

$$
\frac{\partial \ell^{(i)}}{\partial f^{(i)}}
= \frac{f^{(i)} - y^{(i)}}{f^{(i)}\bigl(1 - f^{(i)}\bigr)}.
\tag{eq:dldf}
$$

### Step 2: $\partial f^{(i)}/\partial z^{(i)}$

Recall that

$$
f^{(i)} = g\!\left(z^{(i)}\right).
$$

Using the sigmoid derivative,

$$
\frac{\partial f^{(i)}}{\partial z^{(i)}}
= f^{(i)}\bigl(1 - f^{(i)}\bigr).
\tag{eq:dfdz}
$$

### Step 3: $\partial z^{(i)}/\partial w_j$

We have

$$
z^{(i)} = w^\top x^{(i)} + b
= \sum_{k=1}^n w_k x_k^{(i)} + b.
$$

When differentiating with respect to $w_j$, only the term $w_j x_j^{(i)}$ depends on $w_j$, hence

$$
\frac{\partial z^{(i)}}{\partial w_j}
= x_j^{(i)}.
\tag{eq:dzdw}
$$

### Step 4: Chain rule to combine the three steps

By the chain rule,

$$
\frac{\partial \ell^{(i)}}{\partial w_j}
= \frac{\partial \ell^{(i)}}{\partial f^{(i)}}
  \cdot
  \frac{\partial f^{(i)}}{\partial z^{(i)}}
  \cdot
  \frac{\partial z^{(i)}}{\partial w_j}.
$$

Substituting gives

$$
\begin{align*}
\frac{\partial \ell^{(i)}}{\partial w_j}
&= \frac{f^{(i)} - y^{(i)}}{f^{(i)}\bigl(1 - f^{(i)}\bigr)}
   \cdot f^{(i)}\bigl(1 - f^{(i)}\bigr)
   \cdot x_j^{(i)}\\[3pt]
&= (f^{(i)} - y^{(i)}) x_j^{(i)}.
\end{align*}
$$

Thus

$$
\boxed{
\frac{\partial \ell^{(i)}}{\partial w_j}
= \bigl(f^{(i)} - y^{(i)}\bigr) x_j^{(i)}
}.
\tag{eq:dldwj}
$$

## Gradient of the Overall Cost w.r.t. $w_j$: $\partial J/\partial w_j$

From the definition of $J(w,b)$:

$$
J(w,b) = \frac{1}{m} \sum_{i=1}^m \ell^{(i)}(w,b),
$$

differentiating with respect to $w_j$ gives

$$
\frac{\partial J}{\partial w_j}
= \frac{1}{m} \sum_{i=1}^m
   \frac{\partial \ell^{(i)}}{\partial w_j}.
$$

Substitute the single-example result:

$$
\begin{align*}
\frac{\partial J}{\partial w_j}
&= \frac{1}{m} \sum_{i=1}^m
   \bigl(f^{(i)} - y^{(i)}\bigr) x_j^{(i)}\\[3pt]
&= \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr) x_j^{(i)}.
\end{align*}
$$

Therefore,

$$
\boxed{
\frac{\partial J}{\partial w_j}
= \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr) x_j^{(i)}
}.
\tag{eq:dJdwj}
$$

## Gradient w.r.t. the Bias $b$: $\partial J/\partial b$

The derivation for $b$ is analogous to the derivation for $w_j$.

### Step 1: Single-example gradient $\partial \ell^{(i)}/\partial b$

Again,

$$
\ell^{(i)}(w,b)
= \ell\bigl(f^{(i)}(z^{(i)}(w,b))\bigr),
$$

so by the chain rule,

$$
\frac{\partial \ell^{(i)}}{\partial b}
= \frac{\partial \ell^{(i)}}{\partial f^{(i)}}
  \cdot
  \frac{\partial f^{(i)}}{\partial z^{(i)}}
  \cdot
  \frac{\partial z^{(i)}}{\partial b}.
$$

We only need

$$
\frac{\partial z^{(i)}}{\partial b}
= 1.
\tag{eq:dzdb}
$$

Substituting gives

$$
\begin{align*}
\frac{\partial \ell^{(i)}}{\partial b}
&= \frac{f^{(i)} - y^{(i)}}{f^{(i)}\bigl(1 - f^{(i)}\bigr)}
   \cdot f^{(i)}\bigl(1 - f^{(i)}\bigr)
   \cdot 1\\[3pt]
&= f^{(i)} - y^{(i)}.
\end{align*}
$$

Hence

$$
\boxed{
\frac{\partial \ell^{(i)}}{\partial b}
= f^{(i)} - y^{(i)}
}.
\tag{eq:dldb}
$$

### Step 2: Overall cost gradient

From

$$
J(w,b) = \frac{1}{m} \sum_{i=1}^m \ell^{(i)}(w,b),
$$

we get

$$
\frac{\partial J}{\partial b}
= \frac{1}{m} \sum_{i=1}^m
   \frac{\partial \ell^{(i)}}{\partial b}.
$$

Substituting gives

$$
\begin{align*}
\frac{\partial J}{\partial b}
&= \frac{1}{m} \sum_{i=1}^m
   \bigl(f^{(i)} - y^{(i)}\bigr)\\[3pt]
&= \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr).
\end{align*}
$$

Thus

$$
\boxed{
\frac{\partial J}{\partial b}
= \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr)
}.
\tag{eq:dJdb}
$$

## Gradient Descent Update Rules

The basic gradient descent update rule is

$$
\theta := \theta - \alpha \frac{\partial J}{\partial \theta},
$$

where $\alpha$ is the learning rate.

For $w_j$ and $b$ we have

$$
\begin{align}
w_j &:= w_j - \alpha \frac{\partial J}{\partial w_j},
\tag{eq:gd-wj}\\[3pt]
b   &:= b   - \alpha \frac{\partial J}{\partial b}.
\tag{eq:gd-b}
\end{align}
$$

Substituting gives

$$
\begin{align*}
w_j &:= w_j
 - \alpha \left[
   \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr) x_j^{(i)}
 \right],\\[6pt]
b   &:= b
 - \alpha \left[
   \frac{1}{m} \sum_{i=1}^m
   \bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr)
 \right].
\end{align*}
$$

In vector form, the updates can be written as

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

These are the full gradients and gradient descent update rules for logistic regression.
