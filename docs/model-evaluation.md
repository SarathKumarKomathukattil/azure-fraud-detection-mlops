# Model Evaluation

## Evaluation Objective

Fraud detection is a highly imbalanced classification problem.

For that reason, overall accuracy is not used as the primary release metric.

The model is evaluated using:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC
- False Positive Rate

---

## Candidate Model Comparison & Selection Rationale

Model development evaluated three candidate algorithms before selecting the model family used in the production MLOps lifecycle:

- **Logistic Regression** — interpretable baseline
- **LightGBM** — gradient-boosting candidate
- **XGBoost** — gradient-boosting candidate ultimately retained for production

Because the dataset is severely imbalanced, model comparison emphasized **PR-AUC, ROC-AUC, precision, recall, F1, false-positive behavior, and threshold performance** rather than overall accuracy.

### Validation Comparison

| Model | Validation PR-AUC | Validation ROC-AUC | Role |
|---|---:|---:|---|
| Logistic Regression | **0.081011** | **0.906861** | Baseline |
| LightGBM | **0.269387** | **0.920936** | Boosted-tree challenger |
| XGBoost | **0.267363** | **0.921279** | Productionized model |

The boosted-tree models substantially improved PR-AUC over the Logistic Regression baseline.

LightGBM and XGBoost were then compared under the same operating requirement:

> **Recall >= 0.70, then maximize Precision**

### Operating-Point Comparison

| Metric | XGBoost | LightGBM |
|---|---:|---:|
| Threshold | **0.43953168** | **0.00174157** |
| Precision | 0.048607 | **0.049770** |
| Recall | **0.707912** | 0.701178 |
| F1 | 0.090968 | **0.092943** |
| True Positives | **841** | 833 |
| False Positives | 16,461 | **15,904** |
| True Negatives | 173,506 | **174,063** |
| False Negatives | **347** | 355 |

The comparison was close.

LightGBM achieved a small advantage in validation PR-AUC, precision, F1, and false-positive count, while XGBoost achieved a small advantage in ROC-AUC and recall.

For the final productionized implementation, **XGBoost was retained as the production model family** and carried through:

```text
validation-only threshold selection
        ↓
held-out test evaluation
        ↓
model quality gate
        ↓
Azure ML Model Registry
        ↓
managed online serving
        ↓
model monitoring
        ↓
Continuous Training
        ↓
blue/green deployment
```

The production choice therefore reflects an **engineering trade-off between comparable predictive performance and operational lifecycle maturity**, rather than a claim that XGBoost was superior on every offline metric.

This distinction is important:

> **Model-family selection happens before production model qualification.**

The quality gate determines whether a particular trained XGBoost candidate is eligible for release; it does not choose between Logistic Regression, LightGBM, and XGBoost.

---

## Threshold Selection Policy

The classification threshold is selected using validation data only.

```text
Generate validation probabilities
        ↓
Evaluate candidate thresholds
        ↓
Retain thresholds where Recall >= 0.70
        ↓
Select threshold with maximum Precision
        ↓
Lock threshold
        ↓
Evaluate once on held-out test data
```

The policy is stored as:

```text
threshold_policy = recall_70_max_precision
```

The held-out test set is therefore not used to tune the operating threshold.

---

## Model Quality Gate

A candidate becomes release eligible only when all required criteria pass.

| Metric | Release Rule | Effective Minimum |
|---|---|---:|
| Recall | `Recall >= 0.70` | **0.700000** |
| PR-AUC | `Test PR-AUC >= 90% of reference validation PR-AUC` | **0.240627** |
| ROC-AUC | `Test ROC-AUC >= reference validation ROC-AUC - 0.03` | **0.891279** |

### Gate Calculations

Reference validation PR-AUC:

```text
0.2673629652
```

Minimum test PR-AUC:

```text
0.2673629652 × 0.90
= 0.2406266687
```

Reference validation ROC-AUC:

```text
0.9212790418
```

Minimum test ROC-AUC:

```text
0.9212790418 - 0.03
= 0.8912790418
```

Recall floor:

```text
0.70
```

---

## Representative Approved Challenger

One approved challenger produced:

| Metric | Candidate | Required |
|---|---:|---:|
| Recall | **0.706869** | >= 0.700000 |
| PR-AUC | **0.333139** | >= 0.240627 |
| ROC-AUC | **0.916218** | >= 0.891279 |
| Operating threshold | **0.430271** | selected on validation |

This release passed all configured quality-gate conditions.

---

## Separate Reference Held-Out Evaluation

A separate validated reference release produced:

| Metric | Value |
|---|---:|
| PR-AUC | **0.347941** |
| ROC-AUC | **0.918942** |
| Precision | **0.098103** |
| Recall | **0.706592** |
| F1 | **0.172285** |
| False Positive Rate | **0.091607** |

These values are intentionally documented as a separate validated release and should not be interpreted as the same model version as the representative challenger above.

---

## Confusion Matrix

| | Predicted Fraud | Predicted Normal |
|---|---:|---:|
| **Actual Fraud** | TP = **879** | FN = **365** |
| **Actual Normal** | FP = **8,081** | TN = **80,133** |

---

## Derived Metrics

### Precision

```text
Precision
= TP / (TP + FP)

= 879 / (879 + 8081)

= 0.098103
```

### Recall

```text
Recall
= TP / (TP + FN)

= 879 / (879 + 365)

= 0.706592
```

### F1

```text
F1
= 2 × Precision × Recall
  / (Precision + Recall)

≈ 0.172285
```

### False Positive Rate

```text
FPR
= FP / (FP + TN)

= 8081 / (8081 + 80133)

≈ 0.091607
```

---

## Interpretation

The operating policy intentionally constrains recall to approximately 70%.

Among thresholds that satisfy that recall requirement, the workflow selects the threshold that maximizes precision.

This makes the fraud-detection trade-off explicit rather than optimizing for misleading overall accuracy on an imbalanced dataset.

---

## Release Principle

> **A successfully trained model is not automatically a production-ready model.**

Only candidates that satisfy the release gate are registered as deployment challengers.
