---
title: "05. Data Splitting and Leakage: The Most Common Source of Unrealistically High Scores"
date: 2026-05-05
draft: false
categories: ["Machine Learning · LLM · Agent Full-Stack Roadmap"]
tags: ["Data Leakage", "Train Validation Test", "Cross-Validation"]
summary: "Choose random, grouped, temporal, or nested evaluation from the dependency structure of entities, time, sources, and labels."
---

# 05. Data Splitting and Leakage: The Most Common Source of Unrealistically High Scores

> **Question this article answers:** Choose random, grouped, temporal, or nested evaluation from the dependency structure of entities, time, sources, and labels.

Many impressive offline results are produced by an invalid split rather than a better model. The split must block every path by which information about the evaluation examples can influence training or preprocessing.

## 0. What you should be able to do

- Choose a split that matches the data-generating process.
- Recognize target, feature, entity, temporal, and preprocessing leakage.
- Explain the separate roles of training, validation, and test sets.
- Build a deliberately leaked example and then repair it.

<!-- ## 1. Build the mental model first

```text
entities + time + source + duplicates + label availability
                         ↓
          dependency graph and leakage paths
                         ↓
 random / stratified / grouped / temporal / nested split
```

Keep this data flow in mind. Whenever you meet a formula, library, or new model name, first locate it in the flow instead of memorizing it in isolation.

## 2. Core ideas: from intuition to mechanism

### 2.1 Start from dependencies, not a default API call

Ask which rows share a user, patient, device, document template, event, or future outcome. Samples linked by the same latent entity are not independent just because they occupy different rows.

### 2.2 The split should simulate deployment

Random stratified splitting is suitable only when future observations are exchangeable with current ones. User systems need group splits; forecasting needs forward time splits; model-selection studies may need nested cross-validation.

### 2.3 Preprocessing can leak too

Scaling, imputation, feature selection, target encoding, deduplication decisions, and vocabulary construction must be fitted inside the training fold. A pipeline prevents accidental use of validation statistics.

### 2.4 Duplicates and benchmark contamination matter

Near-duplicate records, mirrored web pages, template documents, and memorized benchmark questions can cross the boundary even when IDs differ. Exact and approximate deduplication are part of evaluation design.

### 2.5 The test set loses its meaning when repeatedly consulted

Each decision made after viewing test results adapts the system to that set. Eventually the test set becomes another validation set, and a new independent evaluation is needed. -->

## 1. Build the mental model first

Before choosing `train_test_split`, `GroupKFold`, or any other API, first ask a more important question:

> **What information should the model not be able to access when it makes a prediction in the real world?**

A data split is not merely a way to divide rows. It is a boundary between what the model is allowed to know during development and what must remain genuinely unseen during evaluation.

Use the following flow:

```text
entities + time + source + duplicates + label availability
                         ↓
       which samples are connected to each other?
                         ↓
       which information crosses the split boundary?
                         ↓
 random / stratified / grouped / temporal / nested split
```

The first line describes the structure of the dataset.

- **Entities:** Do several rows belong to the same user, patient, machine, company, document, or conversation?
- **Time:** Was one row created before or after another row?
- **Source:** Do multiple samples come from the same website, hospital, sensor, template, or data collection process?
- **Duplicates:** Are some samples exact copies or slightly modified versions of each other?
- **Label availability:** Was a feature actually available at the moment the prediction would have been made?

These questions tell us whether rows are truly independent.

For example, imagine a medical dataset with ten visits per patient. A random row split may place eight visits from one patient in the training set and two visits from the same patient in the test set.

```text
Patient A:
    Visit 1 → training
    Visit 2 → training
    Visit 3 → test
    Visit 4 → training
```

The model may learn patient-specific patterns such as age, chronic conditions, medication history, or measurement habits. The test visit is technically a different row, but it is not a genuinely new patient.

The resulting score answers:

> How well can the model predict another visit from a patient it has already seen?

It does not answer:

> How well can the model predict for a completely new patient?

This distinction is the foundation of data splitting.

Whenever you encounter a new splitting method, do not memorize its name in isolation. Locate it in the data flow:

- Random split blocks only direct row reuse.
- Stratified split also preserves class proportions.
- Group split blocks entity overlap.
- Temporal split blocks future-to-past information flow.
- Nested cross-validation separates model selection from final performance estimation.

The correct method is the one that most closely reproduces the information boundary that will exist after deployment.

## 2. Core ideas: from intuition to mechanism

### 2.1 Start from dependencies, not a default API call

A common beginner workflow is:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

The code is valid, but the experiment may still be invalid.

The problem is that `train_test_split` treats every row as an independent sample. Real datasets often contain hidden relationships between rows.

Consider a fraud detection dataset:

```text
Row 1: card 8231, transaction at 10:01
Row 2: card 8231, transaction at 10:04
Row 3: card 8231, transaction at 10:08
```

If the first two transactions are used for training and the third is used for testing, the model has already seen the spending habits, location patterns, merchant preferences, and device information associated with that card.

This may be appropriate if the deployed system predicts future transactions for existing customers. It is not appropriate if the goal is to evaluate performance on completely new customers.

The same problem appears in many domains:

| Dataset                | Hidden dependency                            |
| ---------------------- | -------------------------------------------- |
| Medical records        | Multiple visits from the same patient        |
| Recommendation systems | Multiple interactions from the same user     |
| Predictive maintenance | Multiple measurements from the same machine  |
| NLP datasets           | Sentences from the same document or template |
| Image datasets         | Multiple frames from the same video          |
| Customer analytics     | Multiple purchases from the same account     |
| Speech recognition     | Multiple recordings from the same speaker    |

The important question is not:

> Are these rows different?

The important question is:

> Do these rows share information that the model could recognize?

A useful method is to draw a small dependency graph:

```text
User 17
 ├── session A
 │    ├── sample 1
 │    └── sample 2
 └── session B
      ├── sample 3
      └── sample 4
```

If the deployment scenario involves new users, all samples under `User 17` must remain in the same split.

```text
Correct:
User 17 → entirely training
User 42 → entirely validation
User 91 → entirely test
```

This is what grouped splitting is designed to enforce.

A practical rule is:

> Split at the highest level of dependency that must remain unseen during deployment.

If the model must generalize to new users, split by user. If it must generalize to new hospitals, split by hospital. If it must generalize to new documents, split by document rather than sentence.

---

### 2.2 The split should simulate deployment

A good evaluation is a simulation of future use.

Suppose you are building a model on data collected between 2022 and 2025. The model will be deployed in 2026.

A random split might produce this:

```text
Training:
2022, 2023, 2024, 2025

Test:
2022, 2023, 2024, 2025
```

The training and test sets come from almost identical time periods. This assumes that future data is exchangeable with past data and that the order of events does not matter.

However, the real deployment looks like this:

```text
Training:
2022 → 2023 → 2024 → 2025

Test:
2026
```

A temporal split better represents this situation:

```text
Training:   January 2022 – December 2024
Validation: January 2025 – June 2025
Test:       July 2025 – December 2025
```

The model always learns from the past and predicts the future.

This matters because data changes over time:

- Customer behavior changes.
- Prices and economic conditions change.
- Fraud strategies evolve.
- Medical treatment protocols change.
- Product catalogs change.
- Language and popular topics change.
- Sensors age or are recalibrated.

A random split can hide this drift because training samples may come from dates later than some test samples.

#### Example: predicting customer churn

Assume a customer cancels their subscription on December 20.

The dataset contains these features:

```text
last_login_date
number_of_support_tickets
account_status
final_refund_amount
```

If the prediction is supposed to happen on December 1, `final_refund_amount` does not yet exist. It becomes available only after cancellation.

Even if the feature is stored in the same database table, it is future information from the perspective of the prediction.

The correct question is:

> What information would the production system know at prediction time?

A feature should be included only if it would already exist at that moment.

Different deployment goals require different splits:

| Deployment goal                                               | Suitable evaluation          |
| ------------------------------------------------------------- | ---------------------------- |
| Predict another random sample from the same stable population | Random split                 |
| Preserve the positive/negative class ratio                    | Stratified split             |
| Predict for new users, patients, or devices                   | Group split                  |
| Predict future events                                         | Temporal split               |
| Predict for a new hospital, country, or website               | Source-based or domain split |
| Select hyperparameters and estimate final performance         | Nested cross-validation      |

There is no universally best split. There is only a split that is more or less faithful to the deployment scenario.

---

### 2.3 Preprocessing can leak too

Leakage does not require the target column to appear directly in the features. Information can enter the training process through preprocessing.

Consider standardization:

```python
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2
)
```

This looks harmless, but the scaler was fitted on the entire dataset. Therefore, its mean and standard deviation contain information from the test set.

The test labels were not used, but the test feature distribution influenced the training representation.

The correct order is:

```text
1. Split the data.
2. Fit preprocessing on the training set.
3. Apply the fitted preprocessing to validation and test sets.
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

A pipeline makes this safer:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ]
)

model.fit(X_train, y_train)
```

During cross-validation, the scaler is fitted separately inside each training fold.

The same rule applies to every preprocessing step that learns something from data:

- Mean or median imputation
- Standardization and normalization
- PCA
- Feature selection
- Target encoding
- Vocabulary construction
- TF-IDF calculation
- Outlier thresholds
- Class balancing
- Learned embeddings
- Deduplication thresholds

#### Example: feature selection leakage

Suppose a dataset has 10,000 features. You calculate the correlation between every feature and the target using the entire dataset, keep the top 100 features, and then perform cross-validation.

```text
Entire dataset
      ↓
select features using all labels
      ↓
cross-validation
```

The validation labels have already influenced which features were selected. Cross-validation is no longer evaluating a completely unseen validation fold.

The correct process is:

```text
Cross-validation fold
      ↓
training portion
      ↓
feature selection
      ↓
fit model
      ↓
evaluate on untouched validation portion
```

Feature selection must happen independently inside every fold.

#### Example: target encoding leakage

Suppose a categorical feature contains city names:

```text
Stockholm
London
Paris
Berlin
```

Target encoding replaces each city with its average target value.

```text
Stockholm → 0.72
London    → 0.31
Paris     → 0.55
```

If these averages are calculated using the entire dataset, each validation sample contributes to its own encoded feature. In small categories, the encoded value may almost reveal the label directly.

A strong practical rule is:

> Any operation that calls `fit`, calculates statistics, selects features, learns a vocabulary, or estimates a mapping belongs inside the training fold.

---

### 2.4 Duplicates and benchmark contamination matter

A random split assumes that training and test examples are separate pieces of information. Duplicates break this assumption.

The simplest case is an exact duplicate:

```text
Training:
"The package arrived late and damaged."

Test:
"The package arrived late and damaged."
```

A model can perform well by memorizing the training sentence. The test result does not measure generalization.

However, duplicates are not always identical.

```text
Training:
"The package arrived late and damaged."

Test:
"My package was delivered late and was damaged."
```

These are near duplicates. They express the same example with minor wording changes.

Other forms include:

- A resized or cropped copy of the same image
- Multiple frames from the same video
- A translated version of the same question
- A news article copied by several websites
- Product descriptions generated from the same template
- Different files containing the same source code
- A benchmark question appearing in public training data
- A question with different formatting but the same answer

IDs alone do not solve the problem.

```text
sample_001 → training
sample_847 → test
```

The IDs are different, but the content may still be nearly identical.

#### Example: document classification

Imagine a dataset of insurance documents. Every document begins with a company-specific template:

```text
ABC Insurance Claim Form
Policy Number:
Customer Name:
Date of Incident:
```

A random split puts documents from the same company in both training and test sets. The model may learn the company template rather than the actual semantic differences between claim types.

If deployment involves documents from new companies, evaluation should hold out entire companies or templates.

#### Benchmark contamination

Contamination is especially important for large language models.

Suppose a model is evaluated on a public question-answer benchmark. If the benchmark questions, answer keys, explanations, or close paraphrases appeared in the model's training data, a high score may reflect memorization.

The test set is still formally separate during evaluation, but it was not separate during pretraining.

Useful checks include:

- Exact string matching
- Normalized text matching
- Hash-based duplicate detection
- Character or token n-gram similarity
- Embedding similarity
- Image perceptual hashes
- Source URL comparison
- Template and metadata inspection

Deduplication must also happen before the split.

Incorrect order:

```text
random split
    ↓
deduplicate training and test separately
```

A duplicate can remain once in training and once in test because each subset is cleaned independently.

Better order:

```text
identify duplicate groups
          ↓
keep each group on one side of the boundary
          ↓
split the data
```

Sometimes it is better not to delete duplicates. Instead, assign a duplicate-group ID and use grouped splitting so that all versions remain in the same fold.

---

### 2.5 The test set loses its meaning when repeatedly consulted

The training, validation, and test sets serve different purposes:

```text
Training set
    ↓
fit model parameters

Validation set
    ↓
choose architecture, features, threshold, and hyperparameters

Test set
    ↓
estimate performance after all choices are frozen
```

The test set is valuable because it is supposed to represent an independent experiment.

However, consider the following workflow:

```text
Model A → test accuracy: 82%
Change features

Model B → test accuracy: 84%
Change learning rate

Model C → test accuracy: 83%
Change model architecture

Model D → test accuracy: 86%
Publish Model D
```

Although the model was never trained directly on the test samples, the test results influenced every decision.

The developer learned:

- which features improved test performance,
- which hyperparameters worked better,
- which architecture matched the test set,
- which random seed produced a stronger result,
- and which model should be reported.

The final system has therefore been indirectly optimized for the test set.

The test set has become a validation set.

This is a form of adaptive overfitting:

```text
test result
    ↓
human decision
    ↓
new model
    ↓
test result
    ↓
another human decision
```

Information flows from the test set into model development through the researcher.

#### A simple analogy

Imagine a student practicing with the same final exam every week.

At first, the exam measures the student's general understanding. After seeing the questions repeatedly, the student may learn the exact wording, common traps, and answer patterns.

A high score no longer tells us how well the student would perform on a genuinely new exam.

The same thing happens when a team repeatedly checks the same test set.

#### Better workflow

Use the validation set for development:

```text
Training set:
fit models

Validation set:
compare models
choose features
tune hyperparameters
select thresholds
perform error analysis

Test set:
evaluate once after decisions are frozen
```

After the final test result is obtained, there are two legitimate choices:

1. Report the result without further adaptation.
2. Continue improving the model, but treat the old test set as validation data and create a new independent test set.

For small datasets, nested cross-validation can help.

```text
Outer fold:
estimates generalization performance

Inner fold:
selects models and hyperparameters
```

The inner loop answers:

> Which model should we choose?

The outer loop answers:

> How well does the complete model-selection procedure generalize?

The key principle is:

> A dataset remains a test set only while its results do not influence the system being evaluated.

Even seemingly small decisions count as adaptation:

- Choosing the best random seed
- Changing a classification threshold
- Removing difficult examples
- Adding special rules for common test errors
- Selecting the best checkpoint
- Reporting only the best-performing model
- Rewriting prompts after inspecting benchmark failures

The test set is not protected merely because its labels are hidden from the training algorithm. It must also be protected from repeated human-guided optimization.

## 3. Runnable code

Companion file: [`code/02_data_leakage_demo.py`](../code/02_data_leakage_demo.py)

The example is intentionally small. Run it first, inspect every shape and metric, and then change one variable at a time.

```python
"""Demonstrate target leakage and its repair."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def evaluate(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


def run(seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = 2000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (1.2 * x1 - 0.8 * x2 + rng.normal(scale=1.2, size=n) > 0).astype(int)

    clean_X = np.column_stack([x1, x2])
    leaked_feature = y + rng.normal(scale=0.03, size=n)
    leaked_X = np.column_stack([x1, x2, leaked_feature])
    return evaluate(clean_X, y, seed), evaluate(leaked_X, y, seed)


if __name__ == '__main__':
    clean_auc, leaked_auc = run()
    print({'clean_auc': round(clean_auc, 4), 'leaked_auc': round(leaked_auc, 4)})
```

### Run it

```bash
python code/02_data_leakage_demo.py
```

## 4. Common mistakes and better practice

| Common mistake                                           | Better practice                                                                               |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Using random splitting for user histories or time series | Split by user, entity, or time so the evaluation mirrors what will be unknown in production.  |
| Fitting the scaler before the split                      | Fit every learned preprocessing step on the training fold only, preferably inside a pipeline. |
| Checking only exact duplicate rows                       | Also search for near duplicates, shared templates, derived records, and answer contamination. |

## 5. Required experiments / exercises

- [ ] Run the leakage demo and explain why the leaked metric looks plausible rather than obviously impossible.
- [ ] Convert a random split to a group or temporal split for a dataset you use.
- [ ] Draw the label-creation timeline and mark which features are unavailable at prediction time.

<!-- ## 6. Interview follow-ups: a stable answer structure

### Q: How do you detect data leakage?

**Answer:** Draw the data-generation graph, inspect entity and time overlap, search for duplicates, verify label availability, and ensure all learned preprocessing is fitted inside each training fold.

### Q: Why is a validation set needed?

**Answer:** Training data fits parameters, validation data selects models, thresholds, and hyperparameters, and the test set estimates performance only after choices are frozen.

### Q: When is random splitting unreliable?

**Answer:** When rows share entities, future information, sources, sessions, documents, or other dependencies that would not be available across the real deployment boundary.

A reliable interview structure is: **one-sentence conclusion → mechanism or data flow → one concrete experiment/project example → limitations and trade-offs**.

## 7. Self-check

- [ ] I can draw the data flow from memory.
- [ ] I can explain the key tensor shapes or data structures.
- [ ] I can name at least two failure modes and how to detect them.
- [ ] I can answer the interview questions in 90 seconds each.
- [ ] I recorded the experiment result and one failed attempt in the README.

## 8. References

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

**Previous/next reading:** follow the order in the root `SUMMARY.md`; see Articles 68–70 for the study plans. -->
