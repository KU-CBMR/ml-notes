---
title: "07. Linear Algebra for Machine Learning: Shapes, Broadcasting, Projection, and Batched Matrix Multiplication"
date: 2026-05-07
draft: false
categories: ["Machine Learning · LLM · Agent Full-Stack Roadmap"]
tags: ["Linear Algebra", "Tensor Shapes", "Embeddings"]
summary: "Learn only the linear algebra needed to read tensor code, reason about embeddings, and debug modern neural networks."
---

# 07. Linear Algebra for Machine Learning: Shapes, Broadcasting, Projection, and Batched Matrix Multiplication

> **Question this article answers:** What linear algebra do you actually need to read tensor code, reason about embeddings, and debug modern neural networks?

In practice, many “linear algebra errors” are really **meaning errors expressed through shapes**. A program may run without raising an exception, yet still combine the wrong axes and train the wrong model.

Before manipulating symbols, state the meaning and size of every axis: batch, sequence, channel, feature, head, height, and width. For example, do not describe a tensor only as `[32, 128, 768]`. Write:

```text
[batch=32, sequence=128, hidden=768]
```

That one habit prevents a surprising number of bugs.

## 0. What you should be able to do

By the end of this article, you should be able to:

- Track vector, matrix, and tensor shapes through elementwise operations, transposes, reductions, and matrix multiplication.
- Explain what a dot product measures and why inner product, cosine similarity, and Euclidean distance can rank the same embeddings differently.
- Understand projection as “keeping the part explained by a direction” and connect that idea to least squares and PCA.
- Explain eigenvectors, SVD, rank, and low-rank approximation without relying only on formulas.
- Read the shapes in a multi-head-attention computation and identify which dimensions are contracted.
- Use batched matrix multiplication intentionally instead of depending on accidental broadcasting.
- Interpret Jacobians and Hessians as local sensitivity and curvature, even when the full matrices are too large to construct.

You do **not** need to become a pure mathematician first. The goal is to build enough geometric and shape intuition that tensor code stops looking like unexplained symbol manipulation.

## 1. Build the mental model first

```text
objects → vectors/tensors → linear maps → similarities/projections
       → decompositions (eigen/SVD) → compression and learned representations
```

A useful way to remember the flow is:

1. **Real objects become numbers.** A word, image, user, or product becomes a vector or tensor.
2. **Linear maps reorganize those numbers.** A weight matrix mixes input features into new features.
3. **Similarities compare representations.** Dot products and distances answer questions such as “Which document is closest to this query?”
4. **Projections keep useful directions.** We retain the part of a vector that lies in a chosen direction or subspace.
5. **Decompositions reveal structure.** Eigenvalue decomposition and SVD identify strong, weak, repeated, or redundant directions.
6. **Models learn representations.** Neural networks repeatedly apply linear maps and nonlinear functions to create useful coordinates.

Consider a sentence-classification example:

```text
sentence
  ↓ tokenization
[token ids]
  ↓ embedding lookup
[sequence, hidden]
  ↓ linear layers / attention
[new sequence representations]
  ↓ pooling
[one sentence vector]
  ↓ classifier matrix
[class scores]
```

Linear algebra appears at every arrow, but each operation has a different semantic role. The embedding table converts discrete token IDs into vectors. Attention uses dot products to compare tokens. The classifier matrix converts a hidden representation into class scores.

Whenever you meet a new formula or API, first ask:

- What real object does each axis represent?
- Which dimensions are being combined?
- Which dimensions survive in the output?
- Is this operation comparing vectors, transforming vectors, or compressing vectors?

Locating an operation in this flow is more memorable than memorizing its formula in isolation.

## 2. Core ideas: from intuition to mechanism

### 2.1 Shape is part of the mathematical meaning

A tensor is not just a rectangular container of numbers. Each axis carries meaning.

For example, these two tensors have the same shape:

```text
A: [batch=32, sequence=128, hidden=768]
B: [batch=32, height=128, width=768]
```

They contain the same number of values, but the axes mean different things. Swapping or reducing an axis in `A` changes token structure; doing the same thing in `B` changes image structure. Equal element counts do not imply equal mathematical meaning.

#### 2.1.1 Matrix multiplication contracts one dimension

For ordinary matrix multiplication:

$$
A \in \mathbb{R}^{m \times n}, \qquad B \in \mathbb{R}^{n \times p}
$$

then:

$$
AB \in \mathbb{R}^{m \times p}
$$

The shared dimension $n$ is **contracted**: it is used to compute dot products and disappears from the output.

A small example makes this concrete:

```text
X: [2 samples, 3 input features]
W: [3 input features, 4 output features]
Y = X @ W
Y: [2 samples, 4 output features]
```

Each row of `X` is one sample. Each column of `W` describes how the three input features contribute to one output feature. The number `3` must match because both tensors refer to the same input-feature space.

The output keeps:

- the sample axis from `X`;
- the output-feature axis from `W`.

This is more useful than remembering only the mechanical rule “inside dimensions must match.” The semantic rule is:

> The contracted dimensions must describe the same kind of thing.

#### 2.1.2 A valid shape can still represent the wrong computation

Suppose token embeddings have shape:

```text
X: [batch=8, sequence=100, hidden=100]
```

A linear layer uses:

```text
W: [hidden=100, output=32]
```

Then:

```text
X @ W → [batch=8, sequence=100, output=32]
```

The same weight matrix is applied independently to every token. This is intended.

But imagine transposing `X` incorrectly and obtaining:

```text
[batch=8, hidden=100, sequence=100]
```

A later operation may still run if another dimension happens to equal `100`. The program sees compatible integers; it does not know that “sequence” was matched with “feature.” This is why semantic shape comments matter.

#### 2.1.3 Reduction also changes meaning

For a tensor:

```text
X: [batch, sequence, hidden]
```

these operations answer different questions:

```text
X.mean(axis=1) → [batch, hidden]
```

This averages over tokens and produces one vector per sequence.

```text
X.mean(axis=2) → [batch, sequence]
```

This averages over hidden features and produces one scalar per token.

Both are legal. Only one may match your intention.

#### 2.1.4 A practical debugging rule

At every important line, annotate the tensor:

```python
# x: [batch, sequence, hidden]
# weight: [hidden, output]
y = x @ weight
# y: [batch, sequence, output]
```

Then assert the shape when possible:

```python
assert y.shape == (batch_size, sequence_length, output_size)
```

Do not wait until the loss becomes strange. Catch the semantic mismatch near the operation that created it.

---

### 2.2 Dot products combine magnitude and alignment

For two vectors $x$ and $y$, the dot product is:

$$
x^\top y = \sum_i x_i y_i
$$

Geometrically:

$$
x^\top y = \| x \|_2 \| y \|_2 \cos \theta
$$

This equation says that a dot product becomes large for two different reasons:

1. the vectors have large magnitudes;
2. the vectors point in similar directions.

#### 2.2.1 A small numeric example

Let:

$$
x = [1, 2], \qquad y = [3, 4]
$$

Then:

$$
x^\top y = 1 \cdot 3 + 2 \cdot 4 = 11
$$

Now compare these vectors:

```text
query      = [1, 0]
item A     = [1, 0]
item B     = [10, 0]
item C     = [0, 1]
item D     = [-1, 0]
```

Their dot products with the query are:

```text
query · A = 1
query · B = 10
query · C = 0
query · D = -1
```

Interpretation:

- `A` points in the same direction.
- `B` also points in the same direction, but has much larger magnitude.
- `C` is orthogonal, so the dot product is zero.
- `D` points in the opposite direction, so the dot product is negative.

The dot product does not separate “direction” from “strength.” Sometimes that is exactly what the model wants. Sometimes it is not.

#### 2.2.2 Cosine similarity removes magnitude

Cosine similarity is:

$$
\operatorname{cos}(x,y) = \frac{x^\top y}{\| x \|_2 \| y \|_2}
$$

For the previous example, `item A` and `item B` both have cosine similarity `1` with the query because they point in the same direction.

**This is useful when vector direction carries semantics and vector length is not meant to affect similarity. Sentence embeddings are often compared this way, especially when they are explicitly normalized.**

**But cosine similarity is not automatically “better.” If embedding magnitude contains confidence, popularity, frequency, or another learned signal, normalization removes that information.**

#### 2.2.3 Euclidean distance measures absolute displacement

Euclidean distance is:

$$
\|x-y\|_2
$$

It asks: “How far must I move through the coordinate space to go from one point to the other?”

Compare:

```text
x = [1, 0]
y = [2, 0]
z = [100, 0]
```

All three point in the same direction. Their cosine similarity is `1`, but their Euclidean distances are very different:

```text
||x - y|| = 1
||x - z|| = 99
```

**Cosine similarity sees identical direction. Euclidean distance sees different locations.**

#### 2.2.4 Normalization connects the metrics

Dot product, cosine similarity, and Euclidean distance look like three different ways to compare vectors:

- the dot product depends on both vector length and directional alignment;
- cosine similarity compares direction after removing vector length;
- Euclidean distance measures how far apart two vectors are.

However, after both vectors are normalized to unit length, their magnitudes are fixed. Only their directions can still differ. Under this condition, the three metrics become different numerical expressions of the same geometric relationship.

**Step 1: Expand the squared Euclidean distance**

Assume that both vectors are normalized:

$$
\|x\|_2 = 1
$$

and

$$
\|y\|_2 = 1.
$$

Start from the squared Euclidean distance:

$$
\|x-y\|_2^2 = (x-y)^\top(x-y).
$$

Expanding the product gives

$$
\|x-y\|_2^2 = x^\top x - x^\top y - y^\top x + y^\top y.
$$

For real-valued vectors, the dot product is symmetric:

$$
x^\top y = y^\top x.
$$

Therefore,

$$
\|x-y\|_2^2 = x^\top x + y^\top y - 2x^\top y.
$$

A vector dotted with itself is its squared norm:

$$
x^\top x = \|x\|_2^2
$$

and

$$
y^\top y = \|y\|_2^2.
$$

So, for arbitrary vectors,

$$
\|x-y\|_2^2 = \|x\|_2^2 + \|y\|_2^2 - 2x^\top y.
$$

Because both vectors have unit length,

$$
\|x\|_2^2 = 1
$$

and

$$
\|y\|_2^2 = 1.
$$

Substituting these values gives

$$
\|x-y\|_2^2 = 1 + 1 - 2x^\top y.
$$

Therefore,

$$
\|x-y\|_2^2 = 2 - 2x^\top y.
$$

This is the direct mathematical connection between Euclidean distance and the dot product for normalized vectors.

**Step 2: Connect the dot product to cosine similarity**

Cosine similarity is defined as

$$
\operatorname{cos}(x,y) = \frac{x^\top y} {\|x\|_2\|y\|_2}.
$$

For unit vectors,

$$
\|x\|_2\|y\|_2 = 1.
$$

Therefore,

$$
\operatorname{cos}(x,y) = x^\top y.
$$

For normalized vectors, cosine similarity and dot product are exactly equal. The distance formula can therefore also be written as

$$
\|x-y\|_2^2 = 2 - 2\operatorname{cos}(x,y).
$$

**Why do they produce the same ranking?**

Suppose a normalized query vector $q$ is compared with many normalized candidate vectors $x_i$.

A larger dot product means a larger cosine similarity because

$$
\operatorname{cos}(q,x_i) = q^\top x_i.
$$

At the same time, a larger dot product gives a smaller squared Euclidean distance because

$$
\|q-x_i\|_2^2 = 2 - 2q^\top x_i.
$$

For example, suppose two candidates have dot-product scores

$$
q^\top x_1 = 0.9
$$

and

$$
q^\top x_2 = 0.6.
$$

Their squared Euclidean distances are

$$
\|q-x_1\|_2^2 = 2 - 2(0.9) = 0.2
$$

and

$$
\|q-x_2\|_2^2 = 2 - 2(0.6) = 0.8.
$$

The first candidate has the larger dot product and the smaller Euclidean distance. Therefore, both metrics rank it as the better match.

In general, for normalized vectors,

$$
\text{maximum dot product} \quad\Longleftrightarrow\quad \text{maximum cosine similarity} \quad\Longleftrightarrow\quad \text{minimum Euclidean distance}.
$$

The scores are not numerically identical, but the ranking is identical because one score is a monotonic transformation of the other.

**A concrete two-dimensional example**

Consider the normalized query vector

$$
q = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

and two normalized candidates

$$
a = \begin{bmatrix} 0.8 \\ 0.6 \end{bmatrix}
$$

and

$$
b = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}.
$$

Both candidate vectors have unit length because

$$
0.8^2 + 0.6^2 = 1.
$$

Their dot products with the query are

$$
q^\top a = 0.8
$$

and

$$
q^\top b = 0.6.
$$

Since the vectors are normalized, these values are also their cosine similarities.

Now calculate their squared Euclidean distances:

$$
\|q-a\|_2^2 = 2 - 2(0.8) = 0.4
$$

and

$$
\|q-b\|_2^2 = 2 - 2(0.6) = 0.8.
$$

All three metrics agree that $a$ is more similar to $q$:

- $a$ has the larger dot product;
- $a$ has the larger cosine similarity;
- $a$ has the smaller Euclidean distance.

**Geometric intuition**

Before normalization, two vectors can differ in both length and direction. After normalization, every nonzero vector is moved onto the unit sphere. All vectors now have the same length, so magnitude can no longer influence the comparison.

The dot product can be written as

$$
x^\top y = \|x\|_2\|y\|_2\cos\theta,
$$

where $\theta$ is the angle between the vectors.

For unit vectors,

$$
x^\top y = \cos\theta.
$$

Substituting this into the distance formula gives

$$
\|x-y\|_2^2 = 2 - 2\cos\theta.
$$

This explains the geometry:

- a small angle gives a large cosine value and a small Euclidean distance;
- a large angle gives a small cosine value and a large Euclidean distance.

On the unit sphere, the three metrics are therefore measuring the same angular separation in different ways.

**Why the equivalence fails without normalization**

Without normalization, vector length can change the result.

Consider

$$
q = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \qquad a = \begin{bmatrix} 10 \\ 10 \end{bmatrix}, \qquad b = \begin{bmatrix} 2 \\ 0 \end{bmatrix}.
$$

The raw dot products are

$$
q^\top a = 10
$$

and

$$
q^\top b = 2.
$$

Raw dot product prefers $a$ because $a$ has a much larger norm.

However, $b$ points in exactly the same direction as $q$, so

$$
\operatorname{cos}(q,b) = 1.
$$

For $a$,

$$
\operatorname{cos}(q,a) = \frac{10}{\sqrt{10^2+10^2}} \approx 0.707.
$$

Cosine similarity therefore prefers $b$.

This disagreement occurs because raw dot product contains both magnitude and alignment:

$$
x^\top y = \|x\|_2\|y\|_2\cos\theta.
$$

A large norm can produce a large dot product even when the direction is not the closest match. Normalization removes this magnitude term.

**Practical implication for embedding retrieval**

Suppose both the query embedding and all document embeddings are normalized before retrieval:

```python
query = query / np.linalg.norm(query)

documents = documents / np.linalg.norm(
    documents,
    axis=1,
    keepdims=True,
)
```

The dot-product scores are

```python
dot_scores = documents @ query
```

Because all vectors have unit length, these are also cosine-similarity scores.

The Euclidean distances are

```python
euclidean_distances = np.linalg.norm(
    documents - query,
    axis=1,
)
```

Sorting `dot_scores` from largest to smallest gives the same ranking as sorting `euclidean_distances` from smallest to largest.

This is why a vector database can sometimes use an inner-product index for cosine-similarity retrieval: normalize all embeddings first, then maximum inner product search and maximum cosine similarity search become mathematically equivalent.

This equivalence has several conditions:

- both query and candidate vectors must be normalized;
- zero vectors cannot be normalized because their norm is zero;
- normalization removes any information stored in vector magnitude;
- approximate search, quantization, and floating-point error may still create small implementation differences.

The main idea is:

> Before normalization, both magnitude and direction can affect similarity. After normalization, magnitude is fixed, so dot product, cosine similarity, and Euclidean distance become different transformations of the same angular relationship.

#### 2.2.5 The metric must match training

Suppose a model was trained with raw inner products and learned to encode useful information in vector norms. Replacing inner-product search with cosine search changes the scoring function.

Conversely, if the model was trained using normalized embeddings and cosine-style contrastive learning, raw dot products may introduce unwanted magnitude effects.

A reliable rule is:

> Use the same normalization and similarity definition during indexing, retrieval, evaluation, and training unless you have measured a reason to change them.

---

<!-- ### 2.3 Projection asks how much lies in a direction or subspace

A projection keeps the part of a vector that can be explained by a chosen direction.

Imagine shining a light perpendicular to a line. The shadow of a point on that line is its projection. The original vector can then be separated into:

```text
original vector = projected part + residual part
```

The projected part lies in the chosen direction. The residual is perpendicular to it.

#### 2.3.1 Projection onto one vector

To project $x$ onto a nonzero vector $u$:

$$
\operatorname{proj}_u(x) = \frac{x^\top u}{u^\top u}u
$$

If $u$ is already a unit vector, the denominator is `1`, so:

$$
\operatorname{proj}_u(x) = (x^\top u)u
$$

Consider:

$$
x = [3,2], \qquad u = [1,0]
$$

The vector $u$ represents the horizontal direction. The projection is:

$$
\operatorname{proj}_u(x) = [3,0]
$$

The residual is:

$$
r = x - \operatorname{proj}_u(x) = [0,2]
$$

So the original vector is decomposed as:

$$
[3,2] = [3,0] + [0,2]
$$

The first part is explained by the horizontal direction; the second part is what remains unexplained.

#### 2.3.2 Projection onto a subspace

A single direction is often not enough. Suppose the columns of matrix $Q$ are orthonormal basis vectors for a subspace. Then:

$$
\operatorname{proj}_Q(x) = QQ^\top x
$$

The operation has two conceptual steps:

1. $Q^\top x$ asks how much of $x$ lies along each basis direction;
2. $Q(Q^\top x)$ reconstructs the explained part in the original coordinate system.

#### 2.3.3 Least squares is a projection problem

In linear regression, we seek coefficients $\beta$ such that:

$$
X\beta \approx y
$$

The possible predictions $X\beta$ lie in the column space of $X$. Usually, the target vector $y$ does not lie exactly in that space. Least squares chooses the prediction $\hat{y}=X\hat{\beta}$ that is closest to $y$.

Geometrically:

```text
y = fitted component + residual
```

The fitted component is the projection of $y$ onto the column space of $X$. The residual is orthogonal to every column of $X$.

This gives the normal-equation condition:

$$
X^\top(y-X\hat{\beta}) = 0
$$

You do not need to memorize this as an isolated algebra trick. It simply states:

> After the best projection, no feature direction can explain any more of the residual.

#### 2.3.4 PCA is also projection

PCA finds directions that preserve as much variation as possible. After selecting the top principal directions, each sample is projected onto the subspace they span.

For example, height and weight may be strongly correlated. A two-dimensional cloud of people may mostly lie near a diagonal line. Projecting onto that line compresses two coordinates into one while preserving much of the important variation.

The discarded perpendicular direction contains less variation. It may represent noise, measurement error, or a weaker pattern.

#### 2.3.5 Projection always depends on the chosen geometry

“Perpendicular” and “closest” are defined by an inner product. Standard orthogonal projection uses the usual Euclidean geometry. If features have different scales or a different metric is used, the meaning of distance and orthogonality changes.

This is why preprocessing matters. Projecting raw income and age without scaling may let the numerically larger feature dominate the geometry.

---

### 2.4 SVD exposes important directions and rank

For a matrix $A \in \mathbb{R}^{m \times n}$, the singular value decomposition is:

$$
A = U\Sigma V^\top
$$

A memorable interpretation is:

```text
input coordinates
  ↓ rotate or reflect with Vᵀ
independent directions
  ↓ stretch or shrink with Σ
scaled directions
  ↓ rotate or reflect with U
output coordinates
```

In other words, every linear map can be understood as **rotate → scale → rotate**.

#### 2.4.1 What the pieces mean

- The columns of $V$ are important directions in the input space.
- The singular values in $\Sigma$ tell us how strongly $A$ acts on those directions.
- The columns of $U$ are the corresponding directions in the output space.

If a singular value is large, the matrix strongly preserves or amplifies that direction. If it is tiny, changes along that input direction barely affect the output.

#### 2.4.2 A simple geometric picture

Imagine applying a matrix to every point on the unit circle.

- A general matrix turns the circle into an ellipse.
- The right singular vectors identify the input directions that become the ellipse's principal axes.
- The singular values are the lengths of those axes.
- The left singular vectors describe their final orientation.

This picture explains why singular values measure the strength of a linear transformation.

#### 2.4.3 Rank counts active directions

The rank of a matrix is the number of linearly independent directions it preserves.

In SVD language, rank is the number of nonzero singular values.

For example, consider a dataset matrix whose columns are:

```text
feature 1: temperature in Celsius
feature 2: temperature in Fahrenheit
```

The second feature is an exact linear transformation of the first:

```text
F = 1.8C + 32
```

After centering, these two columns lie on one line. Although the matrix has two columns, it contains only one independent direction of variation. Its effective rank is one.

Real data is rarely exactly low rank, but it is often approximately low rank: a small number of directions explain most of the signal, while many directions contain weak structure or noise.

#### 2.4.4 Low-rank approximation keeps the strongest directions

If the singular values are ordered:

$$
\sigma_1 \ge \sigma_2 \ge \cdots
$$

we can keep only the first $k$ directions:

$$
A_k = U_k \Sigma_k V_k^\top
$$

This creates a rank-$k$ approximation.

A concrete image example helps:

- An image with shape `[height, width]` is a matrix.
- Full SVD represents it using many singular directions.
- Keeping only the largest singular values preserves broad shapes and smooth patterns.
- Removing smaller singular values discards fine details.

At a very low rank, the image looks blurry but recognizable. As rank increases, edges and texture return.

This is compression because storing $U_k$, $\Sigma_k$, and $V_k$ may require far fewer numbers than storing the full matrix.

#### 2.4.5 Why low rank appears in machine learning

Low-rank structure appears in many places:

- PCA compresses data into a small number of principal directions.
- Recommender systems approximate a user-item matrix with low-dimensional user and item factors.
- Low-rank adapters update a large weight matrix through two much smaller matrices.
- Model compression approximates large matrices while trying to preserve important transformations.

The shared idea is:

> Many useful transformations are dominated by a smaller number of directions than the raw matrix size suggests.

#### 2.4.6 Important cautions

SVD finds strong linear structure, not automatically meaningful concepts. A dominant singular direction may reflect lighting, document length, background frequency, or another nuisance factor.

Also, signs are not unique. If one singular vector is multiplied by `-1` and its paired vector is also multiplied by `-1`, the reconstructed matrix is unchanged. Do not attach meaning to the sign alone.

--- -->

### 2.3 Projection asks: how much of this vector belongs to a chosen direction?

Projection sounds abstract, but the idea is simple:

> Choose a direction, then keep only the part of the vector that points along that direction.

Imagine that a vector is an arrow. Now imagine placing a lamp directly above it and shining light onto a line. The shadow of the arrow on that line is its projection.

The original vector can always be split into two parts:

```text
original vector = projected part + leftover part
```

The projected part follows the chosen direction. The leftover part, usually called the **residual**, points away from that direction.

A useful mental picture is:

```text
What the chosen direction can explain
                    +
What the chosen direction cannot explain
                    =
The original vector
```

Projection is therefore not merely “dropping a point onto a line.” It is a way to separate **explained structure** from **unexplained structure**.

---

#### 2.3.1 Start with the easiest case: projection onto the horizontal axis

Consider the vector

$$
x = [3,2].
$$

It means:

- move 3 units horizontally;
- move 2 units vertically.

Now choose the horizontal direction

$$
u = [1,0].
$$

If we keep only the horizontal part of $x$, we get

$$
\operatorname{proj}_u(x) = [3,0].
$$

The part that remains is

$$
r = x - \operatorname{proj}_u(x) = [0,2].
$$

Therefore,

$$
[3,2] = [3,0] + [0,2].
$$

This example contains the whole idea of projection:

- $[3,0]$ is the part explained by the horizontal direction;
- $[0,2]$ is the part the horizontal direction cannot explain.

The residual is vertical, so it is perpendicular to the horizontal direction.

```text
x = [3, 2]
     ↘
      ↘ original vector
       ●
       │
       │ residual = [0, 2]
       │
-------●----------------→ horizontal direction
   projection = [3, 0]
```

The chosen direction acts like a question:

> “How much of $x$ is horizontal?”

The projection is the answer.

---

#### 2.3.2 The dot product first measures how much points along the direction

For a unit direction vector $u$, the dot product

$$
x^\top u
$$

measures the signed amount of $x$ that points along $u$.

Suppose

$$
x = [3,2]
$$

and

$$
u = [1,0].
$$

Then

$$
x^\top u = 3(1) + 2(0) = 3.
$$

The result is the number $3$, not yet a vector.

It tells us:

> The vector $x$ contains 3 units in the direction $u$.

To turn that scalar amount back into a vector, multiply it by the direction vector:

$$
\operatorname{proj}_u(x) = (x^\top u)u.
$$

Substituting the numbers gives

$$
\operatorname{proj}_u(x) = 3[1,0] = [3,0].
$$

So the operation has two steps:

```text
xᵀu        → measure how much lies along u
(xᵀu)u     → rebuild that amount as a vector
```

This is worth remembering:

> The dot product gives the amount; multiplying by the direction gives the projected vector.

---

#### 2.3.3 Why does the general projection formula divide by $u^\top u$?

The simple formula

$$
\operatorname{proj}_u(x) = (x^\top u)u
$$

works only when $u$ has unit length.

A unit vector has norm $1$:

$$
\|u\|_2 = 1.
$$

But what happens if the same direction is written using a longer vector?

For example, both of these vectors point horizontally:

$$
u_1 = [1,0]
$$

and

$$
u_2 = [10,0].
$$

They represent the same direction, but $u_2$ is ten times longer.

If we incorrectly use the unit-vector formula with $u_2$, then

$$
x^\top u_2 = 3(10) + 2(0) = 30.
$$

Multiplying again by $u_2$ gives

$$
30[10,0] = [300,0].
$$

That is obviously not the horizontal part of $[3,2]$. The direction vector was counted twice:

- once inside the dot product;
- once again when multiplying by $u$.

The general formula corrects for the length of $u$:

$$
\operatorname{proj}_u(x) = \frac{x^\top u}{u^\top u}u.
$$

For

$$
x = [3,2]
$$

and

$$
u = [10,0],
$$

we have

$$
x^\top u = 30
$$

and

$$
u^\top u = 10^2 = 100.
$$

Therefore,

$$
\operatorname{proj}_u(x) = \frac{30}{100}[10,0] = [3,0].
$$

The result is correct again.

The denominator

$$
u^\top u = \|u\|_2^2
$$

removes the arbitrary scale of the direction vector.

This gives an important rule:

> Projection depends on the direction of $u$, not on how long we happened to draw $u$.

---

#### 2.3.4 The residual must be perpendicular to the chosen direction

After projection, define the residual as

$$
r = x - \operatorname{proj}_u(x).
$$

For an orthogonal projection, the residual satisfies

$$
u^\top r = 0.
$$

A dot product of zero means the two vectors are perpendicular.

Using the previous example,

$$
u = [1,0]
$$

and

$$
r = [0,2].
$$

Then

$$
u^\top r = 1(0) + 0(2) = 0.
$$

Why is this property important?

Because if the residual still contained some component along $u$, then the projection would not have captured everything available in that direction. We could move a little farther along $u$ and get even closer to $x$.

So perpendicularity means:

> The chosen direction has already explained as much as it possibly can.

This idea reappears in linear regression, least squares, PCA, and many optimization problems.

---

#### 2.3.5 Projection onto a subspace means keeping several directions at once

A single direction may be too restrictive.

For example:

- one line is a one-dimensional subspace;
- one plane is a two-dimensional subspace;
- a collection of several feature directions forms a higher-dimensional subspace.

Imagine a three-dimensional vector floating above the floor. Projecting it onto the floor keeps its horizontal $x$- and $y$-components but removes its vertical component.

```text
3D vector
   ↓ projection onto the floor
2D shadow on the floor
```

Now suppose the columns of a matrix $Q$ are orthonormal direction vectors:

$$
Q = [q_1\ q_2\ \cdots\ q_k].
$$

“Orthonormal” means:

- every column has unit length;
- different columns are perpendicular.

The projection onto the subspace spanned by these columns is

$$
\operatorname{proj}_Q(x) = QQ^\top x.
$$

This formula looks compact, but it performs two understandable steps.

First,

$$
Q^\top x
$$

computes how much of $x$ lies along each basis direction.

The result is a list of coordinates:

```text
amount along q₁
amount along q₂
...
amount along qₖ
```

Second,

$$
Q(Q^\top x)
$$

uses those coordinates to rebuild the explained part in the original space.

So the data flow is:

```text
original vector x
      ↓ Qᵀ
coordinates inside the subspace
      ↓ Q
projected vector in the original space
```

A useful memory aid is:

> $Q^\top$ compresses into subspace coordinates; $Q$ reconstructs from those coordinates.

---

#### 2.3.6 A concrete subspace example

Suppose a vector lives in three dimensions:

$$
x = [3,4,5].
$$

We want to project it onto the horizontal $xy$-plane.

Choose the two basis directions

$$
q_1 = [1,0,0]
$$

and

$$
q_2 = [0,1,0].
$$

These directions form the matrix

$$
Q = [q_1\ q_2].
$$

The coordinates inside the plane are

$$
Q^\top x = [3,4].
$$

These numbers say:

- 3 units along the $x$-axis;
- 4 units along the $y$-axis.

Reconstructing gives

$$
QQ^\top x = [3,4,0].
$$

The vertical component $5$ has been removed.

The residual is

$$
r = [3,4,5] - [3,4,0] = [0,0,5].
$$

So

```text
original vector     = [3, 4, 5]
projected part      = [3, 4, 0]
residual            = [0, 0, 5]
```

The projection keeps everything the plane can represent and discards everything perpendicular to the plane.

---

#### 2.3.7 Least squares is projection disguised as regression

In linear regression, we try to predict a target vector $y$ using features stored in a matrix $X$:

$$
X\beta \approx y.
$$

The coefficient vector $\beta$ chooses how to combine the columns of $X$.

Every possible prediction has the form

$$
\hat{y} = X\beta.
$$

Therefore, all possible predictions live inside the **column space of $X$**.

Think of the columns of $X$ as building blocks. By changing $\beta$, we can mix those columns in different amounts, but we cannot leave the space they span.

Usually, the true target $y$ does not lie exactly inside that space.

```text
target y
   ●
   │\
   │ \
   │  \ residual
   │   \
---●---------------- prediction space
  ŷ
```

Least squares chooses the point $\hat{y}$ in the prediction space that is closest to $y$.

That is exactly an orthogonal projection:

$$
\hat{y} = X\hat{\beta}.
$$

The residual is

$$
r = y - X\hat{\beta}.
$$

At the best solution, the residual is perpendicular to every column of $X$:

$$
X^\top r = 0.
$$

Substituting the definition of $r$ gives

$$
X^\top(y-X\hat{\beta}) = 0.
$$

This is the normal-equation condition.

It does not need to be memorized as mysterious algebra. It says:

> After finding the best prediction, none of the feature directions can explain any remaining part of the error.

If a feature direction still aligned with the residual, changing its coefficient would reduce the error further. Therefore, at the least-squares solution, no such alignment remains.

---

#### 2.3.8 A tiny least-squares example

Suppose we want to approximate

$$
y = [2,3]
$$

using only multiples of

$$
x = [1,1].
$$

The possible predictions are

$$
\hat{y} = \beta[1,1].
$$

This means every possible prediction lies on the diagonal line

```text
[0,0], [1,1], [2,2], [3,3], ...
```

The target $[2,3]$ is not on that line.

Its projection onto the line is

$$
\hat{y} = [2.5,2.5].
$$

The residual is

$$
r = [2,3] - [2.5,2.5] = [-0.5,0.5].
$$

Check perpendicularity:

$$
[1,1]^\top[-0.5,0.5] = -0.5 + 0.5 = 0.
$$

This is the geometric meaning of least squares:

- choose the closest point the model is capable of producing;
- accept the perpendicular remainder as unexplained error.

---

#### 2.3.9 PCA is projection onto directions that preserve the most variation

PCA also uses projection, but it asks a different question.

Least squares asks:

> Which point in the model's prediction space is closest to the target?

PCA asks:

> Which lower-dimensional directions preserve the most variation in the data?

Imagine plotting height and weight for many people.

Because taller people often weigh more, the points may form an elongated diagonal cloud:

```text
weight
  ↑
  |              ●
  |          ●
  |       ●
  |    ●
  | ●
  +--------------------→ height
```

The data uses two coordinates:

```text
[height, weight]
```

But most of the variation follows one diagonal direction.

PCA finds that direction and projects every point onto it.

After projection, each person can be represented by one coordinate instead of two:

```text
[height, weight]
        ↓ project
[position along the main direction]
```

This is compression.

The first principal direction preserves as much variation as possible. A second perpendicular direction captures the remaining variation. If that second direction contains little variation, discarding it loses relatively little information.

A useful interpretation is:

```text
main direction       → dominant shared pattern
perpendicular part   → smaller deviations, noise, or weaker structure
```

However, PCA does not know which variation is useful for your task. It only knows which variation is large.

A direction with high variance may represent:

- useful signal;
- lighting differences;
- document length;
- sensor scale;
- background noise.

So PCA preserves large variation, not automatically meaningful variation.

---

#### 2.3.10 Projection depends on how distance is defined

Projection uses words such as:

- closest;
- perpendicular;
- length;
- angle.

These ideas depend on the geometry used to measure distance.

In ordinary Euclidean geometry, we use the standard dot product. But feature scales can distort this geometry.

Suppose each person is represented by

```text
[age, annual income]
```

Typical values might be

```text
age:       18 to 80
income:    20,000 to 200,000
```

Income has a much larger numerical scale than age.

Without scaling, Euclidean distance may be dominated by income. A difference of 10 years can look tiny compared with a difference of 10,000 currency units.

As a result, projection or PCA may mostly follow the income axis, not because income is inherently more important, but because its numerical values are larger.

This is why preprocessing matters:

```text
raw features
    ↓ center and possibly scale
comparable geometry
    ↓ projection or PCA
meaningful directions
```

The main lesson is:

> Projection is only as meaningful as the distance and feature scaling used to define “closest.”

---

#### 2.3.11 Projection: the complete mental model

Keep this summary in mind:

```text
choose a direction or subspace
              ↓
measure how much of x lies inside it
              ↓
reconstruct the explained part
              ↓
subtract from x
              ↓
obtain a perpendicular residual
```

In symbols,

$$
x = \operatorname{proj}(x) + r.
$$

The projected part is what the chosen space can represent. The residual is what it cannot represent.

This one idea connects:

- vector decomposition;
- least-squares regression;
- PCA;
- dimensionality reduction;
- orthogonal bases;
- low-rank approximation.

---

<!-- ### 2.4 SVD finds the strongest directions inside a matrix

Projection assumes that we have already chosen a useful direction or subspace.

SVD helps answer the next question:

> Where do the important directions come from?

A matrix can be viewed as a machine that receives a vector and produces another vector:

$$
y = Ax.
$$

Different input directions may be treated very differently by this machine:

- one direction may be stretched strongly;
- another may be stretched only a little;
- another may be collapsed almost to zero;
- directions may also be rotated or reflected.

SVD separates this complicated behavior into simple pieces.

For a matrix

$$
A \in \mathbb{R}^{m\times n},
$$

the singular value decomposition is

$$
A = U\Sigma V^\top.
$$

A good mental model is:

```text
input vector
    ↓ Vᵀ
express it in the matrix's preferred input directions
    ↓ Σ
stretch or shrink each direction independently
    ↓ U
place the result into the output space
```

This is often summarized as:

> rotate → scale → rotate

More precisely, the rotations may also include reflections, but “rotate → scale → rotate” is the useful intuition.

---

#### 2.4.1 First understand SVD as a machine with preferred directions

Imagine pushing an object made of soft rubber.

If you push it in one direction, it may stretch a lot. If you push it in another direction, it may barely move.

A matrix behaves similarly. It has preferred input directions.

For each preferred direction $v_i$,

$$
Av_i = \sigma_i u_i.
$$

This equation says:

1. start with the input direction $v_i$;
2. apply the matrix $A$;
3. the result points along $u_i$;
4. its length is multiplied by $\sigma_i$.

The three objects have distinct roles:

```text
vᵢ        preferred input direction
σᵢ        strength of that direction
uᵢ        resulting output direction
```

The singular value $\sigma_i$ is always nonnegative.

If

$$
\sigma_i = 10,
$$

the matrix strongly stretches that direction.

If

$$
\sigma_i = 0.01,
$$

the matrix almost removes that direction.

If

$$
\sigma_i = 0,
$$

the direction disappears completely.

---

#### 2.4.2 What $V^\top$, $\Sigma$, and $U$ actually do

The matrix $V$ contains the preferred input directions:

$$
V = [v_1\ v_2\ \cdots].
$$

Applying $V^\top$ to an input vector asks:

> How much of the input lies along each preferred direction?

This is a projection-like coordinate change.

```text
ordinary input coordinates
          ↓ Vᵀ
coordinates along preferred directions
```

The diagonal matrix $\Sigma$ then scales each coordinate separately:

```text
direction 1 × σ₁
direction 2 × σ₂
direction 3 × σ₃
...
```

Finally, $U$ converts those scaled coordinates into the output directions:

```text
scaled preferred coordinates
          ↓ U
ordinary output coordinates
```

So the full operation

$$
Ax = U\Sigma V^\top x
$$

can be read from right to left:

1. $V^\top x$: describe $x$ using important input directions;
2. $\Sigma V^\top x$: scale each direction by its strength;
3. $U\Sigma V^\top x$: reconstruct the final output.

This is not merely a factorization trick. It reveals how the matrix acts geometrically.

---

#### 2.4.3 A very simple SVD example

Consider the matrix

$$
A = \begin{bmatrix}3&0\\0&1\end{bmatrix}.
$$

This matrix transforms

$$
[x_1,x_2]
$$

into

$$
[3x_1,x_2].
$$

It stretches the horizontal direction by $3$ and leaves the vertical direction unchanged.

The important input directions are already the coordinate axes:

```text
v₁ = horizontal direction
v₂ = vertical direction
```

The singular values are

$$
\sigma_1 = 3
$$

and

$$
\sigma_2 = 1.
$$

The output directions are also the coordinate axes.

In this special case, the rotations do nothing, and SVD is mostly the scaling step:

```text
horizontal component × 3
vertical component   × 1
```

A more general matrix first rotates the preferred directions away from the coordinate axes, but the principle remains the same.

---

#### 2.4.4 Why a unit circle becomes an ellipse

Imagine drawing every unit vector in two dimensions. Their endpoints form a unit circle.

Now apply the same matrix $A$ to every point on the circle.

A general linear transformation turns the circle into an ellipse.

```text
unit circle
    ↓ apply A
ellipse
```

Why an ellipse?

Because SVD says the matrix:

1. rotates the circle;
2. stretches different perpendicular directions by different amounts;
3. rotates the result again.

Rotating a circle does not change its shape. Stretching it differently along two perpendicular directions creates an ellipse. The final rotation changes only the ellipse's orientation.

The singular values are the lengths of the ellipse's principal semi-axes.

If

$$
\sigma_1 = 5
$$

and

$$
\sigma_2 = 1,
$$

the ellipse is long in one direction and narrow in the other.

If

$$
\sigma_1 = \sigma_2,
$$

the circle is scaled equally in all directions and remains a circle.

If

$$
\sigma_2 = 0,
$$

the ellipse collapses into a line.

This picture makes the singular values easy to interpret:

> Singular values tell us how much the matrix stretches its preferred directions.

---

#### 2.4.5 Rank counts how many independent directions survive

The rank of a matrix is the number of independent directions that remain after the matrix acts.

In SVD language:

> Rank is the number of nonzero singular values.

Consider the matrix

$$
A = \begin{bmatrix}1&2\\2&4\end{bmatrix}.
$$

The second row is exactly twice the first. The matrix contains repeated information.

Its two columns are also dependent:

$$
\begin{bmatrix}2\\4\end{bmatrix} = 2\begin{bmatrix}1\\2\end{bmatrix}.
$$

Although the matrix is $2\times2$, it contains only one independent direction.

Therefore, its rank is $1$.

SVD reveals this with one positive singular value and one zero singular value.

The zero singular value means:

> There is an input direction that the matrix completely collapses.

---

#### 2.4.6 A dataset can have many columns but few real directions

Suppose a dataset contains two temperature features:

```text
feature 1: Celsius
feature 2: Fahrenheit
```

They are related by

$$
F = 1.8C + 32.
$$

After subtracting the mean from each feature, the constant $32$ disappears, leaving

$$
F_{\text{centered}} = 1.8C_{\text{centered}}.
$$

The two centered columns contain the same information up to scaling.

So although the dataset has two columns, it has only one true direction of variation.

Another example:

```text
feature 1: distance in meters
feature 2: distance in centimeters
feature 3: distance in millimeters
```

There are three columns, but only one underlying quantity.

This is what low rank means in data:

> The number of stored features is larger than the number of genuinely independent patterns.

Real datasets are rarely exactly low rank. More often, they are approximately low rank:

```text
a few strong directions       → main signal
many weak directions          → fine detail, noise, or small effects
```

SVD measures this through the sizes of the singular values.

---

#### 2.4.7 Singular values form an importance ladder

SVD usually orders singular values from largest to smallest:

$$
\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge \cdots \ge 0.
$$

You can imagine them as an importance ladder:

```text
σ₁   strongest direction
σ₂   second strongest direction
σ₃   third strongest direction
...
```

A steep drop might look like

```text
100, 48, 20, 2, 0.8, 0.1, ...
```

The first few directions dominate the transformation.

A flat sequence might look like

```text
10, 9.5, 9.1, 8.8, 8.4, ...
```

Many directions have comparable strength, so aggressive compression will lose more information.

This is why people often inspect a singular-value plot. It helps answer:

> Is the matrix approximately controlled by only a few directions?

---

#### 2.4.8 Low-rank approximation keeps only the strongest directions

The full SVD is

$$
A = U\Sigma V^\top.
$$

Suppose we keep only the first $k$ singular directions:

$$
A_k = U_k\Sigma_kV_k^\top.
$$

This is a rank-$k$ approximation.

Conceptually, we discard the weakest directions:

```text
keep:
σ₁, σ₂, ..., σₖ

discard:
σₖ₊₁, σₖ₊₂, ...
```

The approximation keeps the transformations that matter most and ignores weaker ones.

This is closely related to projection:

1. $V_k^\top$ projects the input onto the top $k$ input directions;
2. $\Sigma_k$ scales those retained directions;
3. $U_k$ reconstructs the result in the output space.

So low-rank SVD is not unrelated to projection. It is a projection onto the matrix's strongest directions, followed by scaling and reconstruction.

---

#### 2.4.9 Why truncated SVD is a good approximation

Suppose the matrix contains one strong pattern plus a small amount of noise.

For example, imagine a grayscale image of a smooth sky with a bird:

```text
large smooth regions      → strong low-frequency structure
edges and small textures  → weaker fine detail
sensor noise              → very weak directions
```

The largest singular values often capture broad structure. Smaller singular values capture increasingly fine detail.

Keeping only the largest values may preserve:

- large shapes;
- overall lighting;
- smooth gradients;
- dominant repeated patterns.

Discarding smaller values may remove:

- fine texture;
- tiny edges;
- noise.

The approximation becomes blurrier, but the main content can remain recognizable.

SVD has a special mathematical guarantee: among all rank-$k$ matrices, truncated SVD gives the closest approximation under common matrix-distance measures.

The intuitive reason is:

> If only $k$ directions may be kept, the best choice is to keep the $k$ strongest ones.

---

#### 2.4.10 A concrete image-compression example

Suppose a grayscale image has shape

```text
1000 × 1000
```

Storing the full matrix requires

```text
1,000,000 numbers
```

A rank-$k$ SVD stores:

- $U_k$: $1000\times k$ numbers;
- $\Sigma_k$: $k$ numbers;
- $V_k^\top$: $k\times1000$ numbers.

The total is

$$
1000k + k + 1000k = 2001k.
$$

For $k=20$, this is

$$
2001(20) = 40020.
$$

Instead of one million numbers, we store about forty thousand.

That is roughly $4\%$ of the original amount.

The trade-off is that the reconstructed image is approximate.

```text
small k     → strong compression, blurrier image
large k     → weaker compression, sharper image
full rank   → nearly exact reconstruction
```

This makes rank a controllable compression knob.

---

#### 2.4.11 Why low-rank structure appears in machine learning

Low-rank structure appears whenever many observed values are driven by a smaller number of hidden factors.

**Recommender systems**

A user-item rating matrix may contain millions of entries, but preferences may be influenced by a smaller number of factors:

```text
action preference
comedy preference
price sensitivity
brand loyalty
difficulty preference
```

A low-rank model represents each user and item using a short factor vector.

**PCA**

PCA projects high-dimensional data onto a smaller set of strong variation directions.

**Language and embedding models**

Large embedding spaces may contain correlated or redundant directions. Low-rank approximations can sometimes preserve much of the useful transformation with fewer parameters.

**LoRA**

A large weight update is approximated using two small matrices:

$$
\Delta W = BA.
$$

If $B$ and $A$ have a small inner dimension, then $\Delta W$ is low rank.

Instead of training every entry in a huge matrix, LoRA trains a limited number of update directions.

**Model compression**

A large weight matrix can be approximated by a lower-rank product to reduce memory and computation.

The common assumption is:

> The model may contain many parameters, but the useful change or dominant behavior may lie in a much smaller subspace.

---

#### 2.4.12 SVD and PCA are closely related but not identical

PCA operates on centered data and asks:

> Which directions explain the most variance among samples?

SVD asks more generally:

> What are the strongest input-output directions of this matrix?

If $X$ is a centered data matrix, applying SVD gives

$$
X = U\Sigma V^\top.
$$

The columns of $V$ are the principal directions used by PCA.

The singular values are related to how much variance each principal direction explains.

So PCA can be computed using SVD, but their viewpoints differ:

```text
PCA   → interpret directions of data variation
SVD   → decompose any matrix transformation
```

PCA is one important application of SVD.

---

#### 2.4.13 Strong directions are not automatically useful directions

A large singular value means a direction is strong in the matrix. It does not mean that direction is useful for the task.

For images, the strongest direction might represent:

- overall brightness;
- background color;
- camera exposure.

For text, it might represent:

- document length;
- punctuation frequency;
- common formatting.

For user data, it might represent:

- activity level;
- popularity;
- missing-value patterns.

SVD detects strong linear structure. It does not understand meaning.

Therefore, after finding dominant directions, ask:

- What real examples have large positive coordinates?
- What examples have large negative coordinates?
- Does the direction help the downstream task?
- Is it signal, bias, or nuisance variation?

---

#### 2.4.14 Singular-vector signs are arbitrary

Suppose one SVD component uses vectors $u_i$ and $v_i$.

Its contribution is

$$
\sigma_i u_iv_i^\top.
$$

Now flip both signs:

$$
u_i' = -u_i
$$

and

$$
v_i' = -v_i.
$$

Then

$$
\sigma_i u_i'(v_i')^\top = \sigma_i(-u_i)(-v_i)^\top = \sigma_i u_iv_i^\top.
$$

The reconstructed matrix is unchanged.

Therefore, two software runs may return singular vectors with opposite signs while representing the same solution.

Do not interpret the sign alone as inherently meaningful.

What matters is:

- the direction as a line;
- the subspace spanned;
- the reconstructed matrix;
- relative coordinates used consistently.

---

#### 2.4.15 Numerical rank is not always exactly zero or nonzero

In exact mathematics, rank counts nonzero singular values.

In floating-point computation, a theoretically zero singular value may appear as

```text
0.0000000000003
```

because of numerical error.

Real data also contains noise, so weak directions are often small rather than exactly zero.

Therefore, practical code uses a tolerance:

```text
large singular value       → active direction
tiny singular value        → effectively inactive direction
```

This is called **numerical rank**.

The threshold should depend on:

- matrix scale;
- floating-point precision;
- noise level;
- downstream tolerance for approximation error.

Rank is therefore sometimes a modeling decision, not only a literal count.

---

#### 2.4.16 Projection and SVD fit together

Projection and SVD are easier to remember when connected.

Projection says:

> Given a direction or subspace, keep the part of the vector inside it.

SVD says:

> Given a matrix, discover the directions that the matrix treats most strongly.

Low-rank approximation combines both ideas:

```text
discover strong directions with SVD
                ↓
project onto the top k directions
                ↓
discard weak directions
                ↓
reconstruct an approximation
```

This connection appears in PCA, image compression, recommender systems, LoRA, and many representation analyses.

---

#### 2.4.17 SVD: the complete mental model

Remember SVD as a story rather than as three unexplained letters:

```text
A = UΣVᵀ

Vᵀ:
find how much of the input lies along preferred input directions

Σ:
scale each preferred direction by its singular value

U:
place the scaled directions into the output space
```

The singular values tell us which directions matter most.

```text
large σᵢ   → strong direction
small σᵢ   → weak direction
zero σᵢ    → collapsed direction
```

Keeping only the largest singular values gives a low-rank approximation:

```text
full matrix
    ↓ keep dominant directions
smaller representation
    ↓ reconstruct
approximate matrix
```

The main idea is:

> A large matrix may look complicated in ordinary coordinates, while becoming simple when expressed in the right directions. -->

### 2.4 SVD decomposes a matrix into ranked patterns

Before studying the formula, first understand what problem SVD solves.

A matrix may contain thousands or millions of numbers, but those numbers are often highly related. An image contains neighboring pixels with similar values. A user-item rating table contains groups of users with similar tastes. A neural-network weight matrix may perform most of its useful work through a relatively small number of directions.

SVD asks:

> Can this large, complicated matrix be explained using a smaller number of simple patterns?

It then answers three practical questions:

```text
What are the main patterns inside the matrix?
How strong is each pattern?
How much information is lost if weak patterns are removed?
```

The most useful mental model is:

```text
complicated matrix
        ↓ SVD
simple patterns ordered from strongest to weakest
        ↓ keep only the strongest patterns
compression, dimensionality reduction, denoising, or analysis
```

SVD is therefore not mainly about memorizing “rotate, scale, rotate.” Its central purpose is to reveal the hidden structure of a matrix.

---

#### 2.4.1 A matrix can contain many numbers but few real patterns

Suppose a grayscale image has shape

```text
1000 × 1000
```

It contains one million pixel values.

However, those one million values are not one million completely independent facts:

- neighboring pixels are often similar;
- large areas may share the same background;
- edges create repeated horizontal or vertical structure;
- lighting changes many pixels together;
- symmetric objects repeat similar patterns.

The same idea appears in a dataset.

Suppose three columns store the same distance in different units:

```text
distance in meters
distance in centimeters
distance in millimeters
```

There are three columns, but only one underlying quantity. The columns contain repeated information.

SVD tries to discover this kind of redundancy.

It asks:

> How many genuinely independent patterns are needed to explain most of this matrix?

If the answer is much smaller than the raw number of rows or columns, the matrix is approximately low rank.

---

#### 2.4.2 SVD separates a matrix into simple pattern layers

The standard SVD formula is

$$
A = U\Sigma V^\top.
$$

However, another equivalent form is often easier to understand:

$$
A = \sigma_1u_1v_1^\top + \sigma_2u_2v_2^\top + \sigma_3u_3v_3^\top + \cdots.
$$

This says that the original matrix is built by adding many simple pattern matrices:

```text
original matrix
=
first pattern
+
second pattern
+
third pattern
+
...
```

Each pattern has the form

$$
\sigma_i u_i v_i^\top.
$$

Its three parts have different roles:

```text
uᵢ      how the pattern varies across rows
vᵢ      how the pattern varies across columns
σᵢ      how strong the pattern is
```

The vectors $u_i$ and $v_i$ describe the shape of one pattern. The singular value $\sigma_i$ tells us how much that pattern contributes to the original matrix.

SVD orders the singular values from largest to smallest:

$$
\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge \cdots \ge 0.
$$

Therefore:

```text
σ₁u₁v₁ᵀ   strongest pattern
σ₂u₂v₂ᵀ   second strongest pattern
σ₃u₃v₃ᵀ   third strongest pattern
...
```

This ordering is the key reason SVD is useful. It does not merely break a matrix into arbitrary pieces. It places the most important pieces first.

---

#### 2.4.3 What does one simple pattern look like?

Consider two column vectors:

$$
u = \begin{bmatrix}1\\2\\3\end{bmatrix}
$$

and

$$
v = \begin{bmatrix}2\\1\end{bmatrix}.
$$

Their outer product is

$$
uv^\top = \begin{bmatrix}2&1\\4&2\\6&3\end{bmatrix}.
$$

Notice the structure:

- the second column is always one half of the first;
- every row follows the same column pattern;
- the whole matrix is controlled by only two short vectors.

This is a rank-one pattern.

A rank-one matrix is very simple because every row and column follows one shared relationship.

SVD represents a complicated matrix as a sum of such simple rank-one patterns:

```text
complex matrix
=
rank-one pattern
+
rank-one pattern
+
rank-one pattern
+
...
```

The first few patterns often capture broad, repeated structure. Later patterns capture smaller details.

---

#### 2.4.4 A tiny numerical example

Consider the matrix

$$
A = \begin{bmatrix}3&0\\0&1\end{bmatrix}.
$$

It can be written as the sum of two simple pattern matrices:

$$
A = \begin{bmatrix}3&0\\0&0\end{bmatrix} + \begin{bmatrix}0&0\\0&1\end{bmatrix}.
$$

The first part can be written as

$$
\begin{bmatrix}3&0\\0&0\end{bmatrix} = 3\begin{bmatrix}1\\0\end{bmatrix}\begin{bmatrix}1&0\end{bmatrix}.
$$

The second part can be written as

$$
\begin{bmatrix}0&0\\0&1\end{bmatrix} = 1\begin{bmatrix}0\\1\end{bmatrix}\begin{bmatrix}0&1\end{bmatrix}.
$$

Therefore,

$$
A = 3u_1v_1^\top + 1u_2v_2^\top.
$$

SVD has found two independent patterns:

```text
pattern 1: horizontal structure, strength 3
pattern 2: vertical structure, strength 1
```

The first pattern is three times stronger than the second.

If we are allowed to keep only one pattern, SVD keeps the strongest one:

$$
A_1 = \begin{bmatrix}3&0\\0&0\end{bmatrix}.
$$

This is no longer exactly equal to $A$, but it preserves the dominant behavior.

That is the basic idea of low-rank approximation:

> Keep the strongest patterns and discard weaker ones.

---

#### 2.4.5 Why keeping the strongest patterns is useful

Real data often looks like

```text
main structure
+
smaller details
+
noise
```

SVD sorts the patterns by strength, so we can choose how much structure to retain.

Keeping only the strongest patterns may provide:

**Compression**

A large matrix can be stored using a few shorter vectors instead of every original entry.

**Dimensionality reduction**

Many correlated features can be replaced by a smaller number of combined directions.

**Denoising**

Weak, irregular patterns may contain noise. Removing them can sometimes reveal cleaner structure.

**Matrix analysis**

The singular values show whether the matrix has many independent directions or only a few dominant ones.

**Numerically stable computation**

SVD is also useful for least squares, pseudoinverses, and diagnosing nearly redundant features.

All of these uses come from one ability:

> SVD separates strong matrix structure from weak matrix structure.

---

#### 2.4.6 Low-rank approximation is a controlled loss of information

The full decomposition is

$$
A = U\Sigma V^\top.
$$

Suppose we keep only the first $k$ singular patterns:

$$
A_k = U_k\Sigma_kV_k^\top.
$$

Equivalently,

$$
A_k = \sum_{i=1}^{k}\sigma_i u_i v_i^\top.
$$

This is called a rank-$k$ approximation.

Conceptually:

```text
keep:
σ₁, σ₂, ..., σₖ

discard:
σₖ₊₁, σₖ₊₂, ...
```

The value $k$ controls the trade-off:

```text
small k    → smaller representation, more information loss
large k    → larger representation, less information loss
full rank  → nearly exact reconstruction
```

The important fact is that truncated SVD is not an arbitrary approximation.

Among all matrices of rank at most $k$, truncated SVD gives the closest approximation to $A$ under common matrix-error measures.

The intuition is straightforward:

> If only $k$ patterns may be kept, keep the $k$ strongest patterns.

---

#### 2.4.7 Image compression makes the purpose concrete

A grayscale image is a matrix:

```text
rows       → vertical pixel positions
columns    → horizontal pixel positions
values     → pixel brightness
```

SVD decomposes the image into pattern layers:

```text
image
=
strongest pattern
+
second strongest pattern
+
third strongest pattern
+
...
```

The interpretation is not exact for every image, but typically:

```text
early patterns     → broad shapes, large bright and dark regions
middle patterns    → edges and medium-sized details
later patterns     → fine texture, tiny variations, and some noise
```

Suppose the image has shape

```text
1000 × 1000
```

The original image stores

```text
1,000,000 numbers
```

A rank-$k$ approximation stores:

- $U_k$: $1000k$ numbers;
- $\Sigma_k$: $k$ numbers;
- $V_k^\top$: $1000k$ numbers.

The total is

$$
1000k + k + 1000k = 2001k.
$$

For $k=20$,

$$
2001(20) = 40020.
$$

Instead of one million numbers, we store about forty thousand.

The reconstructed image will not be exact, but its main shapes may remain visible.

```text
rank 1       → extremely blurry, only the strongest structure
rank 10      → broad shapes become visible
rank 50      → more edges and textures return
full rank    → original image up to numerical error
```

SVD turns rank into a compression knob.

---

#### 2.4.8 Singular values are an importance ladder

The singular values tell us how much each pattern contributes:

$$
\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge \cdots.
$$

Imagine these singular values:

```text
100, 52, 21, 3, 0.8, 0.1
```

The first three directions dominate the matrix. Keeping only them may preserve most of the structure.

Now imagine:

```text
10, 9.7, 9.3, 8.9, 8.5, 8.1
```

There is no sharp drop. Many directions have similar strength, so aggressive compression will lose more information.

A singular-value plot helps answer:

> Is this matrix controlled mostly by a few patterns, or are many patterns equally important?

A steep drop suggests approximate low rank. A slow decline suggests that many directions matter.

---

#### 2.4.9 Rank counts independent patterns

The rank of a matrix is the number of independent directions or patterns it contains.

In exact SVD language:

> Rank is the number of nonzero singular values.

Consider

$$
A = \begin{bmatrix}1&2\\2&4\end{bmatrix}.
$$

The second column is twice the first:

$$
\begin{bmatrix}2\\4\end{bmatrix} = 2\begin{bmatrix}1\\2\end{bmatrix}.
$$

The matrix has two columns, but they do not carry two independent pieces of information.

It contains only one independent pattern, so its rank is $1$.

SVD reveals this through:

```text
one positive singular value
one zero singular value
```

The zero singular value means that one input direction is completely lost by the matrix.

---

#### 2.4.10 Exact rank and numerical rank are different in practice

In exact mathematics, a singular value is either zero or nonzero.

In floating-point computation, a theoretically zero value may appear as

```text
0.0000000000003
```

because of numerical error.

Real data also contains noise, so redundant directions may produce tiny singular values rather than exact zeros.

Therefore, practical code often uses a tolerance:

```text
large singular value   → active direction
tiny singular value    → effectively inactive direction
```

This produces the **numerical rank**.

Numerical rank depends on:

- floating-point precision;
- matrix scale;
- noise level;
- how much approximation error the application can tolerate.

So rank is sometimes not just a count. It is also a modeling decision about which directions are meaningful enough to keep.

---

#### 2.4.11 What $U$, $\Sigma$, and $V^\top$ mean

Now return to the compact formula:

$$
A = U\Sigma V^\top.
$$

For

$$
A \in \mathbb{R}^{m\times n},
$$

the components have the following roles:

```text
V       important patterns in the input or column space
Σ       strengths of those patterns
U       corresponding patterns in the output or row space
```

If we use the matrix as a transformation

$$
y = Ax,
$$

then

$$
Ax = U\Sigma V^\top x.
$$

Read the operations from right to left.

First,

$$
V^\top x
$$

asks how much of the input $x$ lies along each preferred input direction.

Then,

$$
\Sigma V^\top x
$$

scales each of those directions according to its singular value.

Finally,

$$
U\Sigma V^\top x
$$

reconstructs the result using the corresponding output directions.

The data flow is:

```text
input vector
    ↓ Vᵀ
coordinates along preferred input directions
    ↓ Σ
each coordinate scaled by its singular value
    ↓ U
output vector reconstructed in the output space
```

This is the origin of the phrase:

```text
rotate or reflect
        ↓
scale
        ↓
rotate or reflect again
```

That geometric description explains how a matrix transforms vectors. The pattern description explains why SVD is useful. They are two views of the same decomposition.

---

<!-- #### 2.4.12 Why a unit circle becomes an ellipse

Imagine all unit vectors in two dimensions. Their endpoints form a circle.

Now apply the same matrix $A$ to every vector.

SVD says that the matrix:

1. chooses special perpendicular input directions;
2. scales each direction by a singular value;
3. places the scaled directions into the output space.

As a result, the unit circle becomes an ellipse.

```text
unit circle
    ↓ apply A
ellipse
```

The ellipse's principal directions correspond to the singular vectors. Its principal semi-axis lengths are the singular values.

For example:

```text
σ₁ = 5   → one direction is stretched to length 5
σ₂ = 1   → the perpendicular direction keeps length 1
```

The result is a long, narrow ellipse.

If

$$
\sigma_1 = \sigma_2,
$$

every direction is scaled equally, so the circle remains a circle.

If

$$
\sigma_2 = 0,
$$

one direction disappears, so the ellipse collapses into a line.

This geometric picture gives a second interpretation:

> Singular values measure how strongly the matrix acts along its preferred directions.

---

#### 2.4.13 SVD and projection solve different parts of the problem

Projection assumes that a useful direction has already been chosen:

```text
given a direction
        ↓
keep the part of the vector inside it
```

SVD discovers useful directions from the matrix:

```text
given a matrix
      ↓
find its strongest independent directions
      ↓
order them by importance
```

Low-rank SVD combines the two ideas:

```text
use SVD to discover strong directions
                  ↓
project onto the top k directions
                  ↓
discard weak directions
                  ↓
reconstruct an approximation
```

A useful distinction is:

> Projection uses directions; SVD discovers and ranks directions.

---

#### 2.4.14 SVD and PCA are closely related

PCA asks:

> Which directions preserve the most variation among centered data samples?

SVD asks more generally:

> What are the strongest patterns or input-output directions of this matrix?

For a centered data matrix $X$,

$$
X = U\Sigma V^\top.
$$

The columns of $V$ are the principal directions used by PCA.

Projecting samples onto the first $k$ columns of $V$ produces a lower-dimensional representation.

Therefore, PCA can be computed using SVD.

The viewpoints are slightly different:

```text
PCA   → analyze variation in centered data
SVD   → decompose any matrix into ranked patterns
```

PCA is an important application of SVD. -->

#### 2.4.12 Why a unit circle becomes an ellipse

This geometric picture is not really about circles. It is a way to see what a matrix does to **every possible direction**.

All two-dimensional unit vectors have length $1$. They can be written as

$$
x(\theta)=\begin{bmatrix}\cos\theta\\\sin\theta\end{bmatrix}.
$$

As $\theta$ moves from $0$ to $2\pi$, the endpoints of these vectors form the unit circle.

Now apply the same matrix to every unit vector. Consider

$$
A=\begin{bmatrix}3&0\\0&1\end{bmatrix}.
$$

For an arbitrary vector

$$
x=\begin{bmatrix}x_1\\x_2\end{bmatrix},
$$

the transformed vector is

$$
Ax=\begin{bmatrix}3x_1\\x_2\end{bmatrix}.
$$

So this matrix performs a simple operation:

```text
horizontal component × 3
vertical component   × 1
```

For example:

```text
[ 1,  0]  →  [ 3,  0]
[-1,  0]  →  [-3,  0]
[ 0,  1]  →  [ 0,  1]
[ 0, -1]  →  [ 0, -1]
```

The original circle extends one unit left, right, up, and down. After the transformation, it extends three units left and right but still only one unit up and down. The circle therefore becomes a horizontal ellipse.

We can verify this algebraically. A point on the unit circle is

$$
x(\theta)=\begin{bmatrix}\cos\theta\\\sin\theta\end{bmatrix}.
$$

After applying $A$,

$$
Ax(\theta)=\begin{bmatrix}3\cos\theta\\\sin\theta\end{bmatrix}.
$$

Call the new coordinates $X$ and $Y$:

$$
X=3\cos\theta,\qquad Y=\sin\theta.
$$

Then

$$
\frac{X^2}{9}+Y^2=1.
$$

This is an ellipse whose semi-axis lengths are $3$ and $1$.

Those two lengths are exactly the singular values:

$$
\sigma_1=3,\qquad \sigma_2=1.
$$

In this example, the important directions happen to be horizontal and vertical. For a general matrix, they may be diagonal or rotated. SVD discovers those perpendicular directions automatically.

The decomposition

$$
A=U\Sigma V^\top
$$

can be read from right to left:

```text
Vᵀ  → rotate or reflect the input into the matrix's preferred directions
Σ   → stretch or shrink each preferred direction
U   → rotate or reflect the result into its final orientation
```

Only $\Sigma$ changes the axis lengths. $V^\top$ and $U$ mainly change orientation.

This gives three important cases.

**Different singular values**

If

$$
\sigma_1=5,\qquad \sigma_2=1,
$$

one direction is stretched much more than the other, so the circle becomes a long, narrow ellipse.

**Equal singular values**

If

$$
\sigma_1=\sigma_2=5,
$$

every direction is scaled equally, so the unit circle becomes a larger circle.

**A zero singular value**

If

$$
\sigma_1=5,\qquad \sigma_2=0,
$$

one direction survives while the perpendicular direction is completely flattened. The ellipse collapses into a line segment.

```text
circle
  ↓ one direction scaled to zero
line segment
```

That means the matrix has destroyed one dimension of information.

The geometric meaning is:

> Singular values tell us how much of each preferred direction survives after the matrix transformation.

---

#### 2.4.13 Projection and SVD solve different problems

Projection and SVD are related, but they answer different questions.

Projection asks:

> Given a direction or subspace, how much of this vector lies inside it?

SVD asks:

> Given a matrix, which directions or patterns are strongest?

Suppose a two-dimensional dataset forms a long diagonal cloud:

```text
y
↑
|              ●
|          ●
|       ●
|    ●
| ●
+--------------------→ x
```

The data varies mostly along one diagonal direction.

**Projection uses a direction that is already known**

Suppose someone gives us the diagonal unit vector $q$.

Projection measures how much of each point $x$ lies along $q$:

$$
\operatorname{proj}_q(x)=(q^\top x)q.
$$

Projection does not decide whether $q$ is useful. It simply uses the direction provided to it.

```text
given direction q
        ↓
measure each point along q
```

**SVD discovers the direction**

Now suppose nobody gives us $q$. We only have the data matrix $X$.

SVD examines $X$ and discovers:

- the strongest direction;
- the next strongest perpendicular direction;
- the strength associated with each direction.

```text
given data matrix X
        ↓ SVD
discover important directions
        ↓
rank them by singular value
```

The distinction is:

> Projection uses directions; SVD discovers and ranks directions.

Low-rank SVD combines the two ideas.

First, SVD finds the strongest directions:

$$
v_1,v_2,\ldots,v_k.
$$

Then the input is represented only through those directions.

The rank-$k$ approximation is

$$
A_k=U_k\Sigma_kV_k^\top.
$$

This can be read as:

1. $V_k^\top$ measures the input along the top $k$ directions;
2. $\Sigma_k$ scales the retained directions;
3. $U_k$ reconstructs the output using only those directions.

The complete flow is:

```text
matrix
  ↓ SVD
discover the top k directions
  ↓ projection
keep only the part inside those directions
  ↓ reconstruction
build a lower-rank approximation
```

The discarded directions are not necessarily meaningless. They are simply weaker according to the matrix.

A useful memory aid is:

```text
projection:    use directions
SVD:           discover and rank directions
low-rank SVD:  discover directions, keep the strongest ones, discard the rest
```

---

#### 2.4.14 PCA uses SVD to find directions of maximum variation

PCA and SVD are closely related, but their questions are slightly different.

PCA asks:

> Along which directions does a centered dataset vary the most?

SVD asks:

> What are the strongest patterns or input-output directions of this matrix?

When the matrix is a centered data matrix, the two viewpoints meet.

Suppose

$$
X\in\mathbb{R}^{n\times d},
$$

where:

```text
n  = number of samples
d  = number of features
```

Each row is one sample, and each column is one feature.

Before applying PCA, subtract the mean of each feature. Then compute

$$
X=U\Sigma V^\top.
$$

The columns of $V$ are the principal directions:

```text
v₁  → direction of greatest variation
v₂  → next greatest perpendicular direction
v₃  → third greatest perpendicular direction
...
```

The singular values tell us how strongly the data varies along these directions. The variance explained by component $i$ is proportional to

$$
\sigma_i^2.
$$

Return to the height-weight example. After centering, the points may form a long diagonal cloud.

PCA finds:

```text
first principal direction   → follows the long diagonal cloud
second principal direction  → perpendicular to the cloud
```

The first direction captures most of the variation. The second captures smaller deviations from the main trend.

To reduce the data from $d$ dimensions to $k$ dimensions, keep the first $k$ columns of $V$:

$$
V_k=[v_1\ v_2\ \cdots\ v_k].
$$

Project the centered data onto those directions:

$$
Z=XV_k.
$$

Using the SVD relation,

$$
XV_k=U_k\Sigma_k.
$$

So $Z$ contains the lower-dimensional coordinates.

The complete data flow is:

```text
raw data
   ↓ subtract feature means
centered matrix X
   ↓ SVD
principal directions V
   ↓ keep the first k columns
Vₖ
   ↓ project
Z = XVₖ
```

For example:

```text
original sample:    [height, weight]
compressed sample:  [position along the main diagonal direction]
```

PCA is therefore one important application of SVD.

The distinction is:

```text
SVD:
decompose any matrix into ranked patterns

PCA:
apply SVD to centered data and interpret the top directions
as maximum-variance directions
```

One caution matters:

> PCA preserves directions with large variance, not necessarily directions that are most useful for prediction.

A high-variance direction may represent useful signal, but it may also represent lighting, scale, document length, or another nuisance factor.

---

#### 2.4.15 Singular matrices and rank: when information disappears

The words **singular**, **rank**, and **zero singular value** describe the same underlying event from different viewpoints:

> A matrix has lost one or more independent directions of information.

For a square matrix $A$, the matrix is called **singular** when it has no inverse.

That means no matrix $A^{-1}$ can satisfy

$$
A^{-1}A=I.
$$

Why might an inverse fail to exist?

Because the matrix may map different inputs to the same output.

Consider

$$
A=\begin{bmatrix}1&2\\2&4\end{bmatrix}.
$$

Its second column is twice the first:

$$
\begin{bmatrix}2\\4\end{bmatrix}=2\begin{bmatrix}1\\2\end{bmatrix}.
$$

The two columns do not describe two independent directions. Both lie on the same line.

This matrix maps the two-dimensional input plane into a one-dimensional output line:

```text
2D input plane
      ↓ A
1D output line
```

For example,

$$
A\begin{bmatrix}2\\0\end{bmatrix}=\begin{bmatrix}2\\4\end{bmatrix}
$$

and

$$
A\begin{bmatrix}0\\1\end{bmatrix}=\begin{bmatrix}2\\4\end{bmatrix}.
$$

Two different inputs produce the same output.

An inverse would need to recover the original input from the output, but the output $[2,4]$ does not tell us which of those inputs was used. The information has been lost.

That is why the matrix cannot be inverted.

---

**Rank counts how many independent directions survive**

The rank of a matrix is the number of linearly independent directions it preserves.

For the previous matrix, the rank is $1$, even though the matrix is $2\times2$.

```text
matrix size:  2 × 2
rank:         1
```

It accepts two-dimensional inputs but preserves only one independent dimension.

For a square $n\times n$ matrix:

```text
rank = n   → full rank, no direction is completely lost
rank < n   → rank deficient, at least one direction is lost
```

A square matrix is invertible exactly when it has full rank.

---

**SVD makes rank visible**

Suppose

$$
A=U\Sigma V^\top.
$$

The diagonal entries of $\Sigma$ are the singular values.

The rank is the number of nonzero singular values:

$$
\operatorname{rank}(A)=\#\{i:\sigma_i>0\}.
$$

For a singular $2\times2$ matrix, the singular values might be

$$
\sigma_1=5,\qquad \sigma_2=0.
$$

The first singular value means one direction survives.

The zero singular value means the perpendicular direction is completely flattened.

Geometrically:

```text
unit circle
   ↓ A
line segment
```

This gives the complete connection:

```text
zero singular value
        ↓
one direction is destroyed
        ↓
rank decreases
        ↓
square matrix becomes singular
        ↓
ordinary inverse does not exist
```

---

**Why rank matters in data and machine learning**

Suppose a dataset contains:

```text
feature 1: temperature in Celsius
feature 2: temperature in Fahrenheit
```

After centering, the two features satisfy

$$
F_{\text{centered}}=1.8C_{\text{centered}}.
$$

The second feature contains no new independent information.

The data matrix has two columns but only one real feature direction.

This can cause several problems:

- regression coefficients may not be unique;
- direct matrix inversion may fail;
- numerical algorithms may become unstable;
- storage and computation are wasted on redundant features.

Rank tells us whether the apparent dimensionality matches the actual independent dimensionality.

---

**Singular and nearly singular are different**

A matrix can be invertible but still be dangerously close to singular.

Suppose its singular values are

$$
\sigma_1=100,\qquad \sigma_2=0.000001.
$$

The second direction is not exactly zero, but it is extremely weak.

Inversion must divide by the singular values. Along the weak direction,

$$
\frac{1}{\sigma_2}=1000000.
$$

This means tiny measurement errors or floating-point errors can be amplified one million times.

So the matrix is invertible in theory but unstable in practice.

For a full-rank matrix, the condition number is

$$
\kappa(A)=\frac{\sigma_{\max}}{\sigma_{\min}}.
$$

Interpretation:

```text
condition number near 1      → stable
very large condition number  → nearly singular and noise-sensitive
infinite condition number    → singular
```

This matters in least squares, optimization, regression, and scientific computing.

---

**Why the pseudoinverse is useful**

A singular or rectangular matrix may not have an ordinary inverse.

SVD still lets us define the pseudoinverse:

$$
A^+=V\Sigma^+U^\top.
$$

To construct $\Sigma^+$:

```text
nonzero singular value σᵢ  → replace with 1/σᵢ
zero singular value        → keep as 0
```

The pseudoinverse does not recover information that has already been destroyed.

Instead, it returns a best-fit solution, often the solution with the smallest norm.

It is useful for:

- least squares;
- underdetermined systems;
- rank-deficient regression;
- projection onto column spaces;
- stable numerical solutions.

---

**The complete mental model**

Think of a matrix as acting differently on different directions:

```text
large σᵢ   → direction survives strongly
small σᵢ   → direction is weak and noise-sensitive
zero σᵢ    → direction disappears completely
```

Rank counts how many directions survive.

For a square matrix:

```text
all directions survive    → full rank → invertible
some direction disappears → rank deficient → singular
```

The most important sentence is:

> A singular matrix cannot be inverted because it maps different inputs to the same output, so part of the original information has been permanently lost.

---

#### 2.4.16 Why low rank appears in machine learning

Low-rank structure appears when many observations are controlled by a smaller number of hidden factors.

**Recommender systems**

A user-item rating matrix may have millions of entries, but preferences may depend on fewer latent factors:

```text
action preference
comedy preference
price sensitivity
brand loyalty
difficulty preference
```

A low-rank model represents each user and item using short factor vectors.

**PCA and dimensionality reduction**

Hundreds of correlated features can sometimes be summarized by a much smaller number of principal directions.

**Image and model compression**

Large matrices can be approximated using a smaller number of singular patterns.

**LoRA**

A large weight update is represented as

$$
\Delta W = BA.
$$

When the inner dimension is small, $\Delta W$ is low rank.

The assumption is that fine-tuning does not need to change a large model independently in every possible direction. A smaller set of update directions may be enough.

**Least squares and pseudoinverses**

SVD can detect weak or redundant feature directions and avoid unstable division by extremely small values.

All of these examples use the same idea:

> The raw matrix may be large, while its useful behavior may be controlled by a much smaller subspace.

---

#### 2.4.17 Strong patterns are not automatically meaningful patterns

A large singular value means a pattern is strong in the matrix.

It does not mean the pattern is automatically useful, causal, or understandable.

For images, a strong pattern might represent:

- overall brightness;
- background color;
- camera exposure.

For text, it might represent:

- document length;
- punctuation frequency;
- formatting conventions.

For user data, it might represent:

- activity level;
- popularity;
- missing-value patterns.

SVD finds strong linear structure. It does not understand the task.

After finding dominant directions, ask:

- What real examples score highly on this direction?
- What examples score negatively?
- Does this pattern help prediction or retrieval?
- Is it useful signal, bias, or nuisance variation?

---

#### 2.4.18 The signs of singular vectors are arbitrary

One SVD pattern is

$$
\sigma_i u_i v_i^\top.
$$

If both singular vectors change sign,

$$
u_i' = -u_i
$$

and

$$
v_i' = -v_i,
$$

then

$$
\sigma_i u_i'(v_i')^\top = \sigma_i u_i v_i^\top.
$$

The reconstructed matrix does not change.

Therefore, two SVD implementations may return singular vectors with opposite signs while representing the same solution.

Do not attach meaning to the sign alone.

What matters is:

- the line or subspace represented by the vector;
- the singular value;
- the reconstructed matrix;
- coordinates interpreted consistently.

---

#### 2.4.19 The complete SVD mental model

Start with the practical purpose:

```text
a matrix contains many mixed patterns
                  ↓
SVD separates those patterns
                  ↓
orders them from strongest to weakest
                  ↓
allows us to keep only the strongest ones
```

Then remember the pattern formula:

$$
A = \sigma_1u_1v_1^\top + \sigma_2u_2v_2^\top + \cdots.
$$

Each term is one simple rank-one pattern.

```text
uᵢ      row-side shape
vᵢ      column-side shape
σᵢ      pattern strength
```

Finally, remember the transformation view:

```text
Vᵀ   → measure the input along preferred directions
Σ    → scale those directions
U    → reconstruct the output
```

The most important sentence is:

> SVD takes a complicated matrix, separates it into simple independent patterns, ranks those patterns by strength, and lets us keep only the most important ones when an approximation is acceptable.

### 2.5 Eigenvectors and PCA identify stable directions

An eigenvector of a square matrix $A$ is a nonzero vector $v$ whose direction does not change when multiplied by $A$:

$$
Av = \lambda v
$$

The vector may be stretched, shrunk, or reversed, but it stays on the same line. The scalar $\lambda$ is the eigenvalue.

#### 2.5.1 A concrete example

Consider:

$$
A = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}
$$

Then:

```text
[1, 0] is stretched by 2
[0, 1] is stretched by 1
```

Both coordinate axes are eigenvector directions. A general vector such as `[1, 1]` does not keep its direction because it becomes `[2, 1]`.

Eigenvectors answer:

> Which directions behave simply under repeated application of this transformation?

That makes them useful for studying dynamical systems, covariance structure, graph processes, and local optimization behavior.

#### 2.5.2 PCA uses eigenvectors of covariance

Suppose data matrix $X$ contains centered samples. Its covariance matrix describes how features vary together.

The principal components are eigenvectors of the covariance matrix. The eigenvalues tell us how much variance lies along each component.

A practical PCA flow is:

```text
raw data
  ↓ center each feature
centered data
  ↓ find dominant covariance directions
principal axes
  ↓ project samples
lower-dimensional coordinates
```

Imagine measurements of height and weight. Taller people often weigh more, so the data cloud stretches diagonally. The first principal component follows that long direction. The second component is perpendicular and captures smaller deviations from the height-weight trend.

Keeping only the first component compresses the data from two values to one while preserving much of its variation.

#### 2.5.3 PCA and SVD are closely connected

Instead of explicitly constructing the covariance matrix, we can apply SVD directly to the centered data matrix. The right singular vectors give the principal directions.

This connection is useful because forming a large covariance matrix may be expensive or numerically less stable.

#### 2.5.4 Centering is not optional

Without centering, PCA may mostly capture the direction from the origin to the average sample rather than the directions of variation around the mean.

For example, if every image has a large positive average pixel value, the first uncentered component may represent overall brightness. That may be useful in some applications, but it is not the standard PCA interpretation.

Always record the mean used during fitting and apply the same centering to validation, test, and future data.

---

### 2.6 Jacobians and Hessians describe local sensitivity

A derivative tells us how a small input change affects an output. A Jacobian extends this idea to vector inputs and vector outputs. A Hessian describes how the slope itself changes.

#### 2.6.1 Start with the one-dimensional idea

For a scalar function $f(x)$, the derivative $f'(x)$ is the local slope.

If:

$$
f(x) = x^2
$$

then:

$$
f'(x) = 2x
$$

At $x=3$, a small increase of `0.01` changes the output by approximately:

$$
2 \cdot 3 \cdot 0.01 = 0.06
$$

The derivative is a local linear approximation.

#### 2.6.2 A Jacobian stores many local slopes

For:

$$
f: \mathbb{R}^n \rightarrow \mathbb{R}^m
$$

its Jacobian has shape:

```text
[output dimensions, input dimensions]
```

Each entry asks:

> How does output component $i$ change when input component $j$ changes slightly?

Consider:

$$
f(x_1,x_2) = \begin{bmatrix} x_1+x_2 \\ x_1x_2 \end{bmatrix}
$$

The Jacobian is:

$$
J_f(x) = \begin{bmatrix} 1 & 1 \\ x_2 & x_1 \end{bmatrix}
$$

At $(x_1,x_2)=(2,3)$:

$$
J_f(2,3) = \begin{bmatrix} 1 & 1 \\ 3 & 2 \end{bmatrix}
$$

Interpretation:

- Increasing either input by a small amount increases the first output equally.
- The second output is more sensitive to $x_1$ when $x_2$ is large, and more sensitive to $x_2$ when $x_1$ is large.

#### 2.6.3 Neural networks multiply local sensitivities

A neural network is a composition of functions:

```text
input → layer 1 → layer 2 → ... → loss
```

The chain rule combines their local derivatives. Backpropagation efficiently computes how the final loss changes with respect to many parameters.

In modern models, the full Jacobian may contain billions of entries. Frameworks therefore compute products such as:

```text
vector × Jacobian
Jacobian × vector
```

without materializing the entire matrix.

This is why automatic differentiation can train large networks even though constructing every derivative matrix explicitly would be impossible.

#### 2.6.4 The Hessian describes curvature

For a scalar loss $L(\theta)$ with parameter vector $\theta$, the Hessian is:

$$
H_{ij} = \frac{\partial^2 L} {\partial \theta_i \partial \theta_j}
$$

The gradient tells us which direction goes downhill. The Hessian tells us how the slope changes in different directions.

A useful picture is:

- In a round bowl, curvature is positive in every direction.
- In a long narrow valley, some directions have strong curvature and others are nearly flat.
- At a saddle point, some directions curve upward and others downward.

This matters for optimization. A learning rate that is safe in a flat direction may be too large in a sharply curved direction.

#### 2.6.5 Why full Hessians are rarely formed

If a model has $n$ parameters, the Hessian has $n^2$ entries. For one billion parameters, storing it is not remotely practical.

Deep-learning methods instead use approximations, diagonal statistics, low-rank structure, or Hessian-vector products. The concept remains useful even when the matrix is implicit.

The practical takeaway is:

> Gradients describe first-order direction; curvature explains why optimization speed and stability differ across directions.

---

### 2.7 Batched matrix multiplication applies the same idea many times

Machine-learning code rarely multiplies only two two-dimensional matrices. We usually process many samples, tokens, or attention heads at once.

For batched matrix multiplication:

```text
A: [batch, m, k]
B: [batch, k, n]
A @ B: [batch, m, n]
```

Each batch item performs its own matrix multiplication:

```text
output[b] = A[b] @ B[b]
```

The batch dimension is preserved. The `k` dimension is contracted.

#### 2.7.1 Example: attention scores

Suppose:

```text
Q: [batch, heads, query_length, head_dim]
K: [batch, heads, key_length, head_dim]
```

To compare each query with each key, transpose the last two axes of `K`:

```text
Kᵀ: [batch, heads, head_dim, key_length]
```

Then:

```text
Q @ Kᵀ
```

produces:

```text
[batch, heads, query_length, key_length]
```

The contracted dimension is `head_dim`. Each output value is the dot product between one query vector and one key vector.

For self-attention, `query_length` and `key_length` are usually both the sequence length, so the score matrix is:

```text
[batch, heads, sequence, sequence]
```

The final two axes answer:

> For each query token, how strongly does it match every key token?

#### 2.7.2 Shared weights are also a form of broadcasting

Consider:

```text
X: [batch, sequence, hidden]
W: [hidden, output]
```

Then:

```text
X @ W → [batch, sequence, output]
```

The same matrix `W` is used for every batch item and token. This is intended broadcasting of the leading dimensions.

#### 2.7.3 Broadcasting can silently create extra pairwise computations

Suppose:

```text
A: [batch, 1, m, k]
B: [1, heads, k, n]
```

A general `matmul` operation may broadcast the leading dimensions and return:

```text
[batch, heads, m, n]
```

That may be exactly what you want: every batch item combined with every head. But it may also be an accidental Cartesian expansion that greatly increases memory and computes the wrong relationships.

The operation is mathematically valid. The danger is that the library cannot infer your intended semantics.

#### 2.7.4 Choose an API that communicates intent

Common options include:

- `matmul` or `@`: flexible and supports broadcasting;
- `bmm`: usually requires explicit three-dimensional batches and prevents some accidental broadcasting;
- `einsum`: makes axis relationships explicit, but incorrect labels can still encode the wrong computation.

For example, attention scores can be written conceptually as:

```python
scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
```

The labels say:

- preserve `b` for batch;
- preserve `h` for head;
- preserve `q` for query position;
- preserve `k` for key position;
- contract `d` for head dimension.

Do not use `einsum` merely because it looks advanced. Use it when the labels make the intended axis relationship easier to verify.

## 3. Minimal implementation plan

The fastest way to make these ideas memorable is to implement small examples and assert every shape.

### 3.1 Create arrays with semantic axis comments

```python
import numpy as np

batch = 2
sequence = 3
hidden = 4
output = 5

# x: [batch, sequence, hidden]
x = np.random.randn(batch, sequence, hidden)

# weight: [hidden, output]
weight = np.random.randn(hidden, output)

# y: [batch, sequence, output]
y = x @ weight

assert x.shape == (2, 3, 4)
assert weight.shape == (4, 5)
assert y.shape == (2, 3, 5)
```

Before running the code, predict the output shape on paper. The prediction step is where learning happens.

### 3.2 Implement cosine similarity

```python
def cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(numerator / max(denominator, eps))

x = np.array([1.0, 0.0])
y = np.array([10.0, 0.0])

assert np.isclose(cosine_similarity(x, y), 1.0)
assert np.dot(x, y) == 10.0
```

The vectors have perfect directional agreement, but the raw dot product still reflects magnitude.

### 3.3 Implement projection onto a vector

```python
def project_onto(x: np.ndarray, u: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denominator = np.dot(u, u)
    if denominator < eps:
        raise ValueError("Cannot project onto the zero vector.")
    return (np.dot(x, u) / denominator) * u

x = np.array([3.0, 2.0])
u = np.array([1.0, 0.0])

projection = project_onto(x, u)
residual = x - projection

assert np.allclose(projection, [3.0, 0.0])
assert np.allclose(residual, [0.0, 2.0])
assert np.isclose(np.dot(residual, u), 0.0)
```

The final assertion verifies the key geometric property: the residual is orthogonal to the projection direction.

### 3.4 Implement batched matrix multiplication

```python
batch = 2
m = 3
k = 4
n = 5

# a: [batch, m, k]
a = np.random.randn(batch, m, k)

# b: [batch, k, n]
b = np.random.randn(batch, k, n)

# c: [batch, m, n]
c = a @ b

assert c.shape == (batch, m, n)
assert np.allclose(c[0], a[0] @ b[0])
assert np.allclose(c[1], a[1] @ b[1])
```

This confirms that the batch axis is not contracted. It selects independent matrix multiplications.

### 3.5 Reconstruct a matrix with different SVD ranks

```python
matrix = np.random.randn(20, 10)

u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)

for rank in [1, 2, 5, 10]:
    reconstruction = (
        u[:, :rank]
        @ np.diag(singular_values[:rank])
        @ vt[:rank, :]
    )

    error = np.linalg.norm(matrix - reconstruction, ord="fro")
    print(f"rank={rank:2d}, reconstruction_error={error:.4f}")
```

You should observe that reconstruction error decreases as rank increases. At full rank, the reconstruction should be nearly exact up to floating-point error.

### 3.6 Deliberately create one broadcasting bug

Create two tensors whose leading dimensions are broadcast-compatible but semantically wrong. Let the code run, inspect the unexpectedly large output, and explain which pairwise combinations were created.

A failed experiment is useful here because the dangerous part of broadcasting is precisely that it may not fail.

## 4. Common mistakes and better practice

| Common mistake                                                                      | Why it is dangerous                                                                                              | Better practice                                                                                  |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Reasoning from element counts instead of axis meaning                               | Two tensors can contain the same number of elements while representing completely different structures.          | Annotate tensors as`[batch, sequence, hidden]` and name the contracted axes.                     |
| Confusing elementwise multiplication with matrix multiplication                     | `A * B` multiplies matching positions; `A @ B` computes dot products across a contracted dimension.              | Write the expected output shape before choosing the operator.                                    |
| Calling a generic transpose on a tensor with more than two axes                     | In NumPy,`.T` reverses all axes, which may move batch and sequence dimensions unexpectedly.                      | Use an explicit permutation such as`transpose(0, 2, 1)` or `permute(...)`.                       |
| Using`squeeze()` without specifying an axis                                         | A batch dimension of size one may disappear and cause code to behave differently between training and inference. | Squeeze only the intended axis and assert the result.                                            |
| Depending on implicit broadcasting without checking the expanded shape              | The operation may create unintended pairwise combinations and large intermediate tensors.                        | Use explicit`unsqueeze`, `expand`, or a stricter batched API; inspect the complete output shape. |
| Using cosine similarity when the model was trained for raw inner product            | Normalization removes magnitude information that the objective may have learned to use.                          | Align retrieval normalization and metric with the training objective.                            |
| Comparing unnormalized vectors with Euclidean distance when norms vary greatly      | Distance may be dominated by magnitude rather than semantic direction.                                           | Plot norm distributions and evaluate the metric on the actual task.                              |
| Interpreting PCA components without centering                                       | The first component may mostly point toward the mean instead of explaining variation around it.                  | Fit the mean on training data and reuse it for every later transform.                            |
| Treating a large singular value as automatically meaningful                         | Dominant variation may reflect nuisance factors such as brightness, document length, or frequency.               | Inspect examples at both extremes of each component and test downstream relevance.               |
| Computing an explicit matrix inverse for least squares                              | Inversion can be slower and numerically less stable.                                                             | Use`lstsq`, `solve`, QR, or SVD-based methods.                                                   |
| Assuming rank from matrix dimensions alone                                          | A large matrix can have highly redundant rows or columns.                                                        | Inspect singular values and use a tolerance when estimating numerical rank.                      |
| Forgetting that normalized dot product and Euclidean distance may become equivalent | You may compare retrieval systems that are mathematically producing the same ranking.                            | Derive the relation after normalization before attributing gains to the metric name.             |

## 5. Required experiments / exercises

### 5.1 Annotate a multi-head-attention forward pass

- [ ] Start with `X: [batch, sequence, hidden]`.
- [ ] Derive the shapes of `Q`, `K`, and `V` after projection.
- [ ] Reshape them into `[batch, heads, sequence, head_dim]`.
- [ ] Derive the score shape after `Q @ K.transpose(-2, -1)`.
- [ ] Derive the context shape after multiplying attention weights by `V`.
- [ ] Merge the head and head-dimension axes back into `hidden`.

Expected observation: most attention bugs become obvious once you identify the two sequence axes and the contracted `head_dim` axis.

### 5.2 Compare three embedding metrics

- [ ] Generate or collect several vectors with different directions and norms.
- [ ] Rank them against one query using raw inner product.
- [ ] Rank them using cosine similarity.
- [ ] Rank them using Euclidean distance.
- [ ] Normalize every vector and repeat the experiment.

Expected observation: rankings can differ before normalization. After unit normalization, the three metrics become closely related, and dot product, cosine similarity, and Euclidean distance should produce equivalent rankings.

### 5.3 Visualize projection and residual

- [ ] Choose a two-dimensional vector $x$ and a direction $u$.
- [ ] Plot $x$, its projection onto $u$, and the residual.
- [ ] Verify numerically that the residual is orthogonal to $u$.
- [ ] Change the scale of $u$ and confirm that the projection stays the same.

Expected observation: projection depends on the direction of $u$, not its arbitrary length.

### 5.4 Reconstruct a matrix at several ranks

- [ ] Compute the SVD of an image or data matrix.
- [ ] Reconstruct it using ranks `1`, `5`, `10`, and full rank.
- [ ] Plot reconstruction error against rank.
- [ ] Record how visual quality changes as smaller singular directions are restored.

Expected observation: the first few directions often capture broad structure, while later directions restore finer details.

### 5.5 Create and diagnose an accidental broadcast

- [ ] Construct tensors with leading dimensions `[batch, 1, ...]` and `[1, heads, ...]`.
- [ ] Run matrix multiplication and inspect the output shape.
- [ ] Estimate the intermediate memory use.
- [ ] Rewrite the computation so the intended relationship is explicit.

Expected observation: a result can be shape-valid and still be semantically wrong or unnecessarily expensive.

### 5.6 Connect least squares to projection

- [ ] Create a small design matrix $X$ and target vector $y$.
- [ ] Solve for $\hat{\beta}$ using `np.linalg.lstsq`.
- [ ] Compute the residual $r=y-X\hat{\beta}$.
- [ ] Verify that $X^\top r$ is approximately zero.

Expected observation: the residual is orthogonal to every feature column because the fitted vector is the projection of $y$ onto the column space of $X$.

<!-- ## 6. Interview follow-ups: a stable answer structure

### Q: Why is cosine similarity common for embedding retrieval?

**Answer:** Cosine similarity compares direction while removing vector magnitude. This is useful when the training objective or explicit normalization makes direction carry semantic similarity. For example, `[1, 0]` and `[10, 0]` have the same cosine similarity even though their norms differ by a factor of ten. The limitation is that normalization discards magnitude, so cosine similarity is inappropriate when norm contains useful information such as confidence or frequency. I would therefore match the retrieval metric to the model's training objective and validate it on the downstream ranking task.

### Q: What is the intuition behind SVD?

**Answer:** SVD rewrites a matrix as rotate, scale, then rotate again. The singular vectors identify important input and output directions, while singular values measure how strongly the matrix acts along each direction. If only a few singular values are large, the matrix is approximately low rank, so we can keep those directions for compression. For example, a low-rank image reconstruction preserves broad shapes before fine texture. The limitation is that dominant directions capture variance or transformation strength, not automatically human-interpretable meaning.

### Q: What is broadcasting, and why can it be dangerous?

**Answer:** Broadcasting virtually expands compatible dimensions so an operation can be applied without explicitly copying data. It is useful for applying one bias vector or weight matrix across many samples. The danger is that two leading dimensions may be compatible even when they represent different concepts, creating an unintended pairwise computation. I check the semantic axis names, predict the complete output shape, and use assertions or a stricter API when the intended batching should be one-to-one.

### Q: What is the difference between a dot product and cosine similarity?

**Answer:** A dot product combines vector magnitude and directional alignment, while cosine similarity divides by both norms and keeps only angular alignment. Two vectors in the same direction can have very different dot products but identical cosine similarity. Raw dot products are appropriate when magnitude is part of the learned score; cosine similarity is appropriate when vectors are normalized or only direction should matter.

### Q: How is PCA related to projection?

**Answer:** PCA finds orthogonal directions that explain the most variance in centered data, then projects samples onto the subspace spanned by the top directions. Keeping fewer components is a form of lossy compression. For height and weight data, the first component may follow the main diagonal trend, while the second captures smaller deviations. PCA is linear and variance-based, so high variance does not always mean high task relevance.

### Q: Why do tensor-shape bugs sometimes avoid runtime errors?

**Answer:** Libraries check numeric compatibility, not semantic meaning. If two dimensions happen to have the same size or can be broadcast, an operation may run even though it matches sequence positions with features or combines every batch item with every head. I prevent this by naming axes, using explicit permutations, and asserting intermediate shapes near the operation that changes them.

A reliable interview structure is:

```text
one-sentence conclusion
  → mechanism or data flow
  → one concrete numeric or project example
  → limitation and trade-off
```

Do not stop after giving a definition. A strong answer shows that you can connect the formula to implementation behavior.

## 7. Self-check

- [ ] I can draw the data flow from real objects to vectors, transformations, similarities, projections, and decompositions.
- [ ] Given `A: [m, n]` and `B: [n, p]`, I can explain why the output is `[m, p]` in both numeric and semantic terms.
- [ ] I can identify which axis is contracted in an attention score computation.
- [ ] I can explain why `[1, 0]` and `[10, 0]` have the same cosine similarity but different dot products.
- [ ] I can decompose a vector into a projected component and an orthogonal residual.
- [ ] I can explain SVD using the phrase “rotate, scale, rotate” and connect truncation to low-rank compression.
- [ ] I can explain why PCA requires centering.
- [ ] I can state the shape of a Jacobian for $f:\mathbb{R}^n \to \mathbb{R}^m$.
- [ ] I can name at least three broadcasting or shape failure modes and describe how to detect them.
- [ ] I can answer each interview question in about 90 seconds with one example and one limitation.
- [ ] I recorded the result of each experiment and at least one failed attempt in the README.

A final test is to open a real model implementation and annotate every major tensor for one forward pass. If you can predict shapes before running the code, the linear algebra is becoming operational knowledge rather than memorized vocabulary.

## 8. References

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Linear Algebra Documentation](https://pytorch.org/docs/stable/linalg.html)
- [NumPy Linear Algebra Documentation](https://numpy.org/doc/stable/reference/routines.linalg.html)

---

**Previous/next reading:** follow the order in the root `SUMMARY.md`; see Articles 68–70 for the study plans. -->
