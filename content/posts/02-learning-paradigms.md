---
title: "02. Supervised, Unsupervised, Self-Supervised, and Reinforcement Learning: Where the Learning Signal Comes From"
date: 2026-05-02
draft: false
categories: ["Machine Learning · LLM · Agent Full-Stack Roadmap"]
tags:
  ["Supervised Learning", "Self-Supervised Learning", "Reinforcement Learning"]
summary: "Understand four learning paradigms by tracing how the supervision signal is created rather than memorizing category names."
---

# 02. Supervised, Unsupervised, Self-Supervised, and Reinforcement Learning: Where the Learning Signal Comes From

> **Question this article answers:** Understand four learning paradigms by tracing how the supervision signal is created rather than memorizing category names.

These paradigms can look like four unrelated fields. A more durable question is: what signal tells the system that one prediction or action is better than another, and who or what produced that signal?

## 0. What you should be able to do

- Classify a task by the origin of its learning signal.
- Explain why self-supervised learning still has targets.
- Distinguish a reinforcement-learning reward from a supervised label.
- Design multiple objectives for the same raw dataset.

## 1. Build the mental model first

```text
Supervised: external label y
Unsupervised: structure inside x
Self-supervised: construct target y~=g(x) from x
Reinforcement learning: interaction produces reward r and long-term return
```

Keep this data flow in mind. Whenever you meet a formula, library, or new model name, first locate it in the flow instead of memorizing it in isolation.

## 2. Core ideas: from intuition to mechanism

### 2.1 Supervised learning depends on a labeling protocol

Labels are not natural truth by default. They are produced by annotators, devices, experts, or business rules. Ambiguous definitions, delayed labels, and label drift can set a hard ceiling on performance.

### 2.2 Unsupervised structure still needs interpretation

Clustering, density estimation, and dimensionality reduction reveal patterns under a chosen representation and distance. Cluster IDs have no inherent meaning; stability and usefulness must be validated externally.

### 2.3 Self-supervision is automatic question generation

Masked-token prediction, next-token prediction, image reconstruction, and contrastive learning create targets from the raw input. Human labeling is reduced, but the usefulness of the representation still depends on the objective and data.

### 2.4 Reinforcement learning optimizes sequential consequences

Actions affect future states, rewards can be delayed or sparse, and the policy changes the data it observes. The objective is expected cumulative return, not independent one-step accuracy.

### 2.5 Real systems often combine paradigms

A foundation model may be self-supervised during pretraining, supervised during instruction tuning, and preference-optimized afterward. An agent may additionally learn from environment feedback.

### 2.6 Why models such as VAEs may be called both unsupervised and self-supervised

The distinction depends on which classification question is being asked.

Traditionally, a variational autoencoder is called an **unsupervised generative model** because it is trained on observations (x) without human-provided labels (y). It learns a latent-variable model by maximizing an evidence lower bound:

$$
\mathcal{L}(x) =
\mathbb{E}_{q_\phi(z \mid x)}
\left[
\log p_\theta(x \mid z)
\right]
-
D_{\mathrm{KL}}
\left(
q_\phi(z \mid x)
\,\|\,
p(z)
\right)
$$

From the perspective of the learning signal, however, the reconstruction term has a **self-supervised structure**. The input data provides its own prediction target:

```text
input x → latent representation z → reconstruction x̂
target: the original x
```

No annotator supplies the target. It is constructed automatically from the observation itself. This is similar to masked-token prediction, next-token prediction, image denoising, and ordinary autoencoding.

The two labels therefore emphasize different properties:

| Classification question                             | Description of a VAE                                     |
| --------------------------------------------------- | -------------------------------------------------------- |
| Does training require human labels?                 | No, so it is unsupervised.                               |
| Is there an explicit target derived from the input? | Yes, so its reconstruction objective is self-supervised. |
| Does it learn a probability model of the data?      | Yes, so it is a latent-variable generative model.        |

A precise description is:

> A VAE is an unsupervised latent-variable generative model trained partly through a self-supervised reconstruction signal.

Terminology is not completely standardized. Some authors use **self-supervised learning** broadly for any task whose targets are generated from the data, while others reserve the term mainly for representation-learning objectives such as masking, contrastive prediction, or temporal prediction. The underlying mechanism is less ambiguous than the label: always inspect how the target and loss are produced.

## 3. Minimal implementation plan

Take one dataset and write three objective functions: a supervised target, a self-supervised prediction task, and an interaction-based reward. Compare what information each objective requires and what behavior it encourages.

## 4. Common mistakes and better practice

| Common mistake                                           | Better practice                                                                                      |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| “No human labels” means unsupervised                     | Self-supervised learning uses explicit prediction targets generated from the data itself.            |
| Treating reward as an ordinary label                     | Reward evaluates action consequences, may arrive late, and depends on the policy and trajectory.     |
| Calling a visually pleasing clustering result successful | Check stability across seeds, scaling, number of clusters, and usefulness for a downstream decision. |

## 5. Required experiments / exercises

- [ ] Design supervised, self-supervised, and reinforcement-learning objectives for a recommender system.
- [ ] Specify how positive and negative pairs would be constructed for a contrastive-learning task.
- [ ] Find one label in your project that looks objective but is actually defined by a workflow or policy.

<!-- ## 6. Interview follow-ups: a stable answer structure

### Q: How is self-supervised learning different from unsupervised learning?

**Answer:** Self-supervised learning constructs a specific prediction target from the input and optimizes a supervised-style loss. Unsupervised learning more broadly models structure, density, similarity, or latent factors without a fixed external target.

### Q: Why do large language models rely heavily on self-supervision?

**Answer:** Raw text is vastly more abundant than manually labeled instruction data, and next-token prediction turns almost every position into a training example.

### Q: What is the central difference between reinforcement learning and supervised learning?

**Answer:** In reinforcement learning, the policy influences future states and the data distribution, rewards may be delayed, and the objective is long-term return. Supervised examples are usually treated as fixed input–target pairs.

A reliable interview structure is: **one-sentence conclusion → mechanism or data flow → one concrete experiment/project example → limitations and trade-offs**.

## 7. Self-check

- [ ] I can draw the data flow from memory.
- [ ] I can explain the key tensor shapes or data structures.
- [ ] I can name at least two failure modes and how to detect them.
- [ ] I can answer the interview questions in 90 seconds each.
- [ ] I recorded the experiment result and one failed attempt in the README. -->

## 8. References

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

**Previous/next reading:** follow the order in the root `SUMMARY.md`; see Articles 68–70 for the study plans.
