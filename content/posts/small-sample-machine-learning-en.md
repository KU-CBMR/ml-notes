---
title: "Small-Sample Machine Learning: Can We Still Do ML When We Have Very Few Samples?"
date: 2026-02-18
tags: ["machine learning", "small data", "model selection", "practical ML"]
categories: ["Machine Learning"]
draft: false
---

When people hear “machine learning”, they often think we need thousands or millions of samples.

That is helpful, of course. But it is not always required.

In real research and real industry projects, we often do not have big data. In medicine, biology, lab experiments, industrial inspection, questionnaires, and many scientific studies, a dataset may only contain dozens of samples. Sometimes even fewer.

So the question is not simply:

> Can we train a model?

The more important question is:

> Can we train a model that still works on new data?

Small-sample machine learning is possible. But we need to treat it differently from large-scale machine learning. We should not think of it as a “small version of big data”. We should think of it as a high-uncertainty problem where overfitting is always nearby.

In this note, I will explain three things:

1. Can we do machine learning with very few samples?
2. What algorithms are usually suitable?
3. If we have many datasets with different sample sizes, how should we choose a strategy?

---

## 1. Can we do machine learning with very few samples?

Yes, we can.

But we need to lower our expectations and be more careful.

The most common mistake in small-sample machine learning is this:

> We have very little data, but we use a very complex model.

For example:

- 30 samples, but we train a deep neural network from scratch.
- 50 samples, but we have 20,000 features and directly run XGBoost, random forest, or deep learning.
- A tiny validation set, but we try many models and many hyperparameters until one looks good.

This can easily produce a model with high accuracy on paper. But the model may not have learned a real pattern. It may have simply memorized the training data.

In small-sample settings, the main challenge is not whether a model can fit the data. Almost any flexible model can fit a small dataset.

The real challenge is:

> Will the model generalize to new samples?

So in small-sample machine learning, the key ideas are:

- control overfitting
- use simple baselines
- use regularization
- estimate uncertainty
- validate carefully
- avoid data leakage
- be honest about instability

---

## 2. The first principle: do not start with a complex model

When the sample size is small, a simple model is often more reliable than a complex model.

Here is an intuitive way to think about it.

A complex model is like a student with a very strong memory. If there are only a few exam questions, the student can memorize the answers without understanding the subject.

A simple model is more restricted. It cannot memorize every detail. It is forced to capture only the strongest and most general patterns.

That is why simple models often work surprisingly well in small datasets.

A practical order is:

1. **Start with simple baselines**: logistic regression, linear regression, ridge, lasso.
2. **Then try moderately complex models**: linear SVM, kernel SVM, random forest, gradient boosting.
3. **Only consider deep learning carefully**: usually with pretraining, transfer learning, or feature extraction.

Do not start by asking:

> Which algorithm is the most powerful?

A better question is:

> Given my small sample size, which model is least likely to fool me?

---

## 3. What does “small sample size” mean?

There is no universal cutoff.

Whether a dataset is “small” depends on at least three things.

### 3.1 Number of samples: n

If you only have tens of samples, it is clearly a small-sample problem.

If you have a few hundred samples, it depends. For a simple tabular problem, a few hundred samples may be usable. But if the features are high-dimensional, the labels are noisy, or the classes are imbalanced, a few hundred samples can still be small.

### 3.2 Number of features: p

Many scientific datasets are not just small. They are small and high-dimensional.

For example:

- 50 patients
- 20,000 gene expression features

This is a classic **p much larger than n** problem.

In this setting, overfitting is very easy. With enough features, we can always find patterns that look meaningful but are actually random noise.

### 3.3 Signal strength

If the signal is very strong, small samples can still be useful.

For example, if two groups are clearly different, even 20 samples may show a pattern.

But if the signal is weak, the noise is high, and the labels are uncertain, even hundreds of samples may not be enough.

So we should not judge only by sample size. We should also ask:

- How many features do we have?
- Is the signal strong or weak?
- Are the labels reliable?
- Are the classes balanced?
- Are the datasets consistent with each other?
- Is there dataset shift or batch effect?

---

## 4. Which algorithms are suitable for small samples?

Here is a practical guide.

| Situation                                                 | Good first choices                                             | Why                                           |
| --------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------- |
| Classification, small sample, not too many features       | Logistic regression, linear SVM                                | Simple, stable, interpretable                 |
| Regression, small sample                                  | Ridge, lasso, elastic net                                      | Regularization helps reduce overfitting       |
| Many features, few samples                                | Lasso, elastic net, linear SVM                                 | Useful for high-dimensional data              |
| Possible non-linear pattern, but sample size is not large | RBF SVM, random forest                                         | Can capture non-linearity, but tune carefully |
| Many small related datasets                               | Pooled model, multi-task learning, hierarchical model          | Can share information across datasets         |
| Image, text, or sequence data with few samples            | Pretrained model + feature extraction or fine-tuning           | Avoid learning everything from scratch        |
| Need uncertainty estimates                                | Bayesian model, Gaussian process                               | Can express uncertainty more naturally        |
| Need interpretability                                     | Logistic regression, lasso, elastic net, shallow decision tree | Easier to explain                             |

My general rule is:

> With small samples, prefer constrained models before flexible models.

A constrained model is not always the most accurate model on the training data. But it is often more trustworthy.

---

## 5. How to understand common algorithms in small-sample settings

### 5.1 Logistic regression and linear regression

These are excellent starting points.

Many people think they are “too simple”. But in small datasets, simplicity is often a strength.

If the task is binary classification, start with logistic regression.

If the task is continuous prediction, start with linear regression, preferably with regularization such as ridge or lasso.

Why they are useful:

- stable
- interpretable
- fast
- good as baselines
- less likely to overfit compared with very flexible models

If a complex model only slightly outperforms logistic regression, be careful. That small improvement may just be random variation in cross-validation.

---

### 5.2 Ridge, lasso, and elastic net

These are very useful for small-sample problems, especially when the number of features is large.

A simple explanation:

- **Ridge** shrinks model coefficients so they do not become too large.
- **Lasso** can shrink some coefficients exactly to zero, so it performs a kind of feature selection.
- **Elastic net** combines ridge and lasso. It is often a good choice for high-dimensional biological or scientific data.

If your data looks like gene expression, proteomics, imaging-derived features, omics data, or other high-dimensional tables, elastic net should almost always be considered.

---

### 5.3 Support vector machine

SVM can work well with small samples, especially when the feature dimension is high.

A **linear SVM** is often a good choice when:

- sample size is small
- feature dimension is high
- we want a relatively stable classifier

An **RBF kernel SVM** can capture non-linear patterns, but it needs careful tuning. If the parameters are too aggressive, it can overfit badly.

A simple rule:

- Small n, large p: try linear SVM first.
- Suspected non-linearity: try RBF SVM, but tune carefully.
- Always evaluate with cross-validation.

---

### 5.4 Random forest

Random forest is popular and useful, but it is not always the best first choice for very small datasets.

Its strengths:

- handles non-linear relationships
- handles interactions between features
- less sensitive to feature scaling
- often works well with tabular data

Its weaknesses in small samples:

- each tree sees only part of the data
- results can be unstable
- feature importance can be misleading
- it may look good in random splits but fail on external data

So random forest is worth trying, but I would not blindly trust it in very small datasets.

For very small n, I usually look at regularized linear models and SVM before relying on random forest.

---

### 5.5 XGBoost, LightGBM, and CatBoost

Gradient boosting models are very strong for many tabular data problems.

But in small-sample settings, they can be risky.

Two common problems:

1. We tune too many hyperparameters and accidentally overfit the validation set.
2. The model looks strong in one split but performs poorly on a new dataset.

If you use boosting in small datasets, I would suggest:

- use shallow trees
- use strong regularization
- use a conservative learning rate
- limit the hyperparameter search space
- use nested cross-validation if possible
- do not keep tuning until the validation score looks good

Boosting can be a candidate model. But it should not be the only answer.

---

### 5.6 Deep learning

If the sample size is very small, training a deep neural network from scratch is usually a bad idea.

But this does not mean deep learning is impossible.

In small-sample settings, deep learning is more reasonable when we use existing knowledge:

- pretrained models
- transfer learning
- frozen feature extractors
- fine-tuning only the last layers
- self-supervised pretraining
- data augmentation

For example, in image analysis, training a CNN from scratch with 100 images is risky. But using a pretrained model to extract embeddings, and then training a simple classifier on top, can be reasonable.

The core idea is:

> Do not ask a deep model to learn the whole world from your tiny dataset. Let it bring prior knowledge, and only adapt it slightly to your task.

---

## 6. What if we have many datasets with different sizes?

This is very common.

Suppose we have several datasets:

- Dataset A: 30 samples
- Dataset B: 80 samples
- Dataset C: 300 samples
- Dataset D: 1000 samples

A tempting idea is to combine all samples and randomly split them into train and test sets.

But this can be dangerous.

Different datasets may differ in many ways:

- sample source
- patient population
- experimental batch
- measurement platform
- label definition
- feature distribution
- preprocessing pipeline

If we ignore these differences, the model may not learn the biological, medical, or business signal we care about.

It may learn dataset identity.

For example, the model may learn:

> This sample looks like it came from Dataset A.

instead of:

> This sample has the disease pattern.

This is called dataset shift, domain shift, or batch effect depending on the context.

---

## 7. Three strategies for multiple datasets

### 7.1 Build a separate model for each dataset

This is the most conservative approach.

It is useful when:

- datasets are very different
- labels are not exactly the same
- features are not fully aligned
- you want to know whether each dataset contains a signal on its own

The advantage is that the analysis is clean.

The disadvantage is that each dataset becomes even smaller. If every dataset is tiny, separate models may be unstable.

---

### 7.2 Combine datasets into one model

This is the most direct approach.

It is reasonable when:

- features are aligned
- labels have the same meaning
- measurement methods are similar
- dataset distributions are not too different

Before combining datasets, we should:

1. align features
2. standardize carefully
3. check batch effects
4. keep the dataset ID
5. test generalization with leave-one-dataset-out validation

The last point is extremely important.

Do not rely only on a random train/test split.

In a random split, samples from the same dataset can appear in both training and testing. This can make performance look better than it really is.

A better test is:

> Leave out an entire dataset as the test set.

For example:

- Train on Dataset A + B + C
- Test on Dataset D

This tells us whether the model can generalize to a new data source.

---

### 7.3 Use hierarchical models or multi-task learning

If datasets are related but not identical, we can consider more advanced methods:

- hierarchical models
- mixed-effect models
- multi-task learning
- domain adaptation
- meta-learning

The intuition is:

> Each dataset has its own characteristics, but the datasets may also share a common signal.

These methods are more flexible than fully pooling all data, but they also share information better than training every dataset separately.

However, they are more complex to implement and explain.

So in practice, I would usually start with:

1. single-dataset baselines
2. pooled baseline model
3. leave-one-dataset-out validation
4. only then consider hierarchical or multi-task methods

---

## 8. How do we decide which algorithm to use?

I usually follow this process.

### Step 1: Define the task

First, identify the type of problem:

- classification
- regression
- survival analysis
- clustering
- anomaly detection
- ranking

Different tasks need different models.

For example:

- classification: logistic regression, SVM, random forest, XGBoost
- regression: ridge, lasso, random forest regressor, XGBoost regressor
- survival analysis: Cox model, regularized Cox model, random survival forest

Do not choose the algorithm before understanding the task.

---

### Step 2: Look at the relationship between n and p

This is one of the most important steps.

If:

> n is small and p is large

then first consider:

- lasso
- elastic net
- linear SVM
- PCA plus a simple model
- feature selection plus a simple model

If:

> n is moderate and p is not too large

then you can try:

- logistic regression
- SVM
- random forest
- gradient boosting

If:

> n is large

then more complex models and deep learning become more realistic.

---

### Step 3: Decide how much interpretability you need

If the goal is scientific understanding, biomarker discovery, clinical explanation, or a publication, interpretability matters.

In that case, do not only chase the highest accuracy.

Prefer models such as:

- logistic regression
- linear regression
- lasso
- elastic net
- Cox model
- shallow decision tree

If the goal is mainly prediction and interpretability is less important, you can include:

- random forest
- XGBoost
- LightGBM
- neural networks

But with small samples, even prediction-focused projects need strict validation.

---

### Step 4: Check whether the validation design is reliable

In small-sample machine learning, validation is often more important than the algorithm.

A common mistake is:

> I tried 20 models, tuned many hyperparameters, and reported the best validation result.

This can overfit the validation set.

Better practice:

- use cross-validation
- use nested cross-validation for tuning
- repeat cross-validation with different random seeds
- use leave-one-dataset-out validation when there are multiple datasets
- report mean performance and uncertainty, not only the best score

In small datasets, do not only ask:

> What is the highest accuracy?

Ask:

> If I change the data split, does the result remain stable?

---

## 9. A practical workflow for small-sample machine learning

### 9.1 Inspect the data first

Before modeling, check:

- number of samples in each dataset
- number of samples per class
- number of features
- missing values
- outliers
- class imbalance
- differences between datasets

This step is not optional. In small datasets, one unusual sample can change the result.

---

### 9.2 Build simple baselines

Do not start with the most complex model.

Start with:

- majority-class baseline
- logistic regression
- ridge or lasso
- elastic net
- linear SVM

If a simple baseline already works well, the signal may be strong.

If a simple baseline completely fails, a complex model may not magically solve the problem. It may only overfit better.

---

### 9.3 Handle features carefully

Feature processing is very important in small samples.

Possible steps:

- remove clearly uninformative features
- standardize features
- reduce dimensionality
- select features using domain knowledge
- combine highly correlated features
- use PCA carefully

One warning is very important:

> Feature selection must happen inside cross-validation.

Do not select features using the full dataset and then run cross-validation. That causes data leakage.

All preprocessing steps that learn from data should be fitted only on the training fold and then applied to the validation fold.

---

### 9.4 Compare only a small number of models

Do not compare 50 models in a tiny dataset.

A reasonable first set might be:

- logistic regression with regularization
- elastic net
- linear SVM
- random forest
- XGBoost or LightGBM with strong regularization

If you try too many models, you increase the chance of selecting a model that looks good only by luck.

---

### 9.5 Use the right validation strategy

If you have only one dataset:

- use repeated k-fold cross-validation
- if the sample size is extremely small, leave-one-out cross-validation is possible, but its variance can be high

If you have multiple datasets:

- use leave-one-dataset-out validation
- or at least split by dataset group, not by random individual samples

This is often the difference between a result that looks good and a result that is actually useful.

---

### 9.6 Report uncertainty

With small samples, a single number is not enough.

Report things like:

- mean performance
- standard deviation
- confidence interval
- performance for each dataset
- results across different random seeds

This helps readers understand whether the model is stable.

A model with 0.85 accuracy in one split and 0.55 accuracy in another split is not a reliable model, even if the best number looks impressive.

---

## 10. Common traps in small-sample machine learning

### Trap 1: Data leakage

Examples:

- standardizing using the full dataset before cross-validation
- selecting features using the full dataset before cross-validation
- running PCA on the full dataset before cross-validation
- using information from the test set during preprocessing

Correct approach:

> Fit preprocessing only on the training fold. Then apply it to the validation or test fold.

---

### Trap 2: Too much hyperparameter tuning

When the validation set is small, it is noisy.

If we tune too many parameters, we may tune to noise rather than signal.

So in small datasets:

- keep the search space small
- use simple models first
- use nested cross-validation if tuning is important
- avoid repeatedly looking at the test set

---

### Trap 3: Reporting only the best result

If we try many models and only report the best one, the result is usually optimistic.

A more honest report includes:

- which models were tried
- how hyperparameters were tuned
- how validation was done
- whether the result was stable
- how much uncertainty there is

---

### Trap 4: Confusing dataset effects with real signal

When combining datasets, the model may learn dataset identity instead of the target signal.

Useful checks include:

- Can a model predict dataset ID easily?
- Do PCA or UMAP plots cluster by dataset?
- Does performance drop when testing on a held-out dataset?
- Does random split look good but leave-one-dataset-out look poor?

If yes, the model may not be learning a generalizable pattern.

---

## 11. What I would do in practice

If I had many datasets with different sizes, I would do this.

### Layer 1: Analyze each dataset separately

For each dataset, check:

- sample size
- class balance
- feature count
- missing values
- simple baseline performance

The goal is not to get the highest score. The goal is to understand whether each dataset contains useful signal.

---

### Layer 2: Build a pooled model

If features and labels are compatible, combine the datasets.

But keep the dataset ID.

Start with simple models:

- logistic regression with regularization
- elastic net
- linear SVM
- random forest
- XGBoost with strong regularization

Also check whether dataset effects are strong.

---

### Layer 3: Use leave-one-dataset-out validation

Suppose we have five datasets.

We can train on four datasets and test on the remaining one. Then repeat this so every dataset becomes the test set once.

This answers a very important question:

> Is the model learning a general pattern, or is it only adapting to specific datasets?

If the model performs well in random split but poorly in leave-one-dataset-out validation, I would not trust it as a general model.

---

### Layer 4: Only then consider more complex methods

If simple models are stable, we may not need a complex model.

If performance varies a lot across datasets, then we can consider:

- adding dataset indicators
- batch correction
- domain adaptation
- multi-task learning
- hierarchical models

But these should come after the basic checks, not before.

---

## 12. A simple decision tree

Here is a quick rule of thumb.

```text
Do we have very few samples?
  Yes -> start with simple models + regularization + strict validation
  No  -> more complex models may be reasonable

Is the number of features much larger than the number of samples?
  Yes -> lasso / elastic net / linear SVM / dimensionality reduction
  No  -> logistic regression / SVM / random forest / boosting can all be tried

Do we have multiple datasets?
  Yes -> do not rely only on random splits; use leave-one-dataset-out validation
  No  -> use repeated cross-validation

Do we need interpretability?
  Yes -> linear models / lasso / elastic net / shallow trees
  No  -> random forest / XGBoost / LightGBM can be included

Is the data image, text, or sequence data?
  Yes -> use pretrained models and light fine-tuning or feature extraction
  No  -> start with traditional ML baselines
```

---

## 13. Summary

Small-sample machine learning is not impossible.

But it should not be done like big-data machine learning.

When the sample size is small, the goal is not to find the most powerful model. The goal is to build the most honest and reliable analysis possible with limited data.

My main suggestions are:

1. **Start with simple baselines.** Do not jump directly to complex models.
2. **Use regularized models.** Ridge, lasso, elastic net, and linear SVM are often strong choices.
3. **Be careful with high-dimensional data.** When p is much larger than n, overfitting is easy.
4. **Do not randomly mix multiple datasets without checking dataset effects.**
5. **Use leave-one-dataset-out validation when possible.** This is essential for testing generalization across datasets.
6. **Report uncertainty.** A single best score is not enough.
7. **Use deep learning carefully.** Prefer pretrained models, feature extraction, and transfer learning.

One sentence summary:

> In small-sample machine learning, the goal is not to fit the training data as well as possible. The goal is to honestly estimate how much reliable signal we can learn from very limited data.
