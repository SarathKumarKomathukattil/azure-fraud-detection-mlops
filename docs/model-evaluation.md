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