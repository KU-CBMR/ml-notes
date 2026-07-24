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

Suppose a normalized query vector \(q\) is compared with many normalized candidate vectors \(x_i\).

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

All three metrics agree that \(a\) is more similar to \(q\):

- \(a\) has the larger dot product;
- \(a\) has the larger cosine similarity;
- \(a\) has the smaller Euclidean distance.

**Geometric intuition**

Before normalization, two vectors can differ in both length and direction. After normalization, every nonzero vector is moved onto the unit sphere. All vectors now have the same length, so magnitude can no longer influence the comparison.

The dot product can be written as

$$
x^\top y = \|x\|_2\|y\|_2\cos\theta,
$$

where \(\theta\) is the angle between the vectors.

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

Raw dot product prefers \(a\) because \(a\) has a much larger norm.

However, \(b\) points in exactly the same direction as \(q\), so

$$
\operatorname{cos}(q,b) = 1.
$$

For \(a\),

$$
\operatorname{cos}(q,a) = \frac{10}{\sqrt{10^2+10^2}} \approx 0.707.
$$

Cosine similarity therefore prefers \(b\).

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

### 2.3 Projection asks how much lies in a direction or subspace

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

---

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
