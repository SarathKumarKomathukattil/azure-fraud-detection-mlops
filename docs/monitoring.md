# Model Monitoring

## Objective

Production monitoring answers two separate questions:

```text
Has the input distribution changed?
```

and:

```text
Has predictive performance degraded?
```

These signals are intentionally monitored independently.

---

## Feature Drift Monitoring

Feature drift is measured using Population Stability Index (PSI).

Conceptually:

```text
PSI
= Σ (current proportion - reference proportion)
  × ln(current proportion / reference proportion)
```

Monitoring configuration:

| Parameter | Value |
|---|---:|
| Historical reference rows | **6,082,007** |
| Current monitoring window | **10,000 rows** |
| Monitored features | **12** |
| CT drift trigger | **PSI >= 0.20** |

Drift results are persisted in:

```text
fraud_mlops.gold.feature_drift_metrics
```

---

## PSI Interpretation

Reporting bands:

```text
PSI < 0.10
→ LOW

0.10 <= PSI < 0.25
→ MODERATE

PSI >= 0.25
→ HIGH
```

The Continuous Training policy intentionally uses:

```text
PSI >= 0.20
```

as its retraining condition.

---

## Production Predictions

Production prediction records are persisted in:

```text
fraud_mlops.gold.production_predictions
```

This provides an auditable production-scoring history.

---

## Delayed Ground Truth

Fraud labels are not assumed to be available immediately after prediction.

Matured labels are persisted in:

```text
fraud_mlops.gold.production_ground_truth
```

The monitoring workflow therefore distinguishes:

```text
feature-distribution change
```

from:

```text
actual predictive-performance degradation
```

---

## Performance Monitoring

Once labels become available, the workflow calculates:

```text
Precision
Recall
F1
PR-AUC
ROC-AUC
```

Results are persisted in:

```text
fraud_mlops.gold.model_performance_metrics
```

---

## Continuous Training Conditions

Retraining becomes eligible when at least one condition is breached:

| Signal | Condition |
|---|---:|
| Feature drift | **PSI >= 0.20** |
| Recall | **< 0.70** |
| PR-AUC | **< 0.240627** |
| ROC-AUC | **< 0.891279** |

The trigger result and trigger reason are persisted for traceability.

---

## CT Workflow

Production workflow:

```text
fraud-mlops-ct-workflow
```

Execution:

```text
feature_drift_monitoring ─────────┐
                                  │
performance_monitoring ───────────┤
                                  ↓
                              ct_trigger
                                  ↓
                           should_retrain
                                  ↓
                    build_retraining_dataset
                                  ↓
                    export_retraining_snapshot
                                  ↓
                   trigger_azureml_retraining
                                  ↓
                    evaluation + quality gate
                                  ↓
                       register challenger
```

---

## Operational Monitoring

Model monitoring is separate from endpoint infrastructure monitoring.

Azure Monitor is responsible for:

```text
HTTP request failures
P95 inference latency
endpoint service health
```

Databricks is responsible for:

```text
drift
model performance
retraining policy
CT workflow state
```

---

## Alerting Policy

| Signal | Condition | Platform |
|---|---|---|
| Endpoint failures | `4xx or 5xx > 0` | Azure Monitor |
| P95 latency | `> 2000 ms` initial guardrail | Azure Monitor |
| Feature drift | `PSI >= 0.20` | Databricks |
| Recall degradation | `< 0.70` | Databricks |
| PR-AUC degradation | `< 0.240627` | Databricks |
| ROC-AUC degradation | `< 0.891279` | Databricks |
| CT job failure | Job failure | Databricks |
| CD workflow failure | Workflow failure | GitHub Actions |

The latency threshold is currently an initial operational guardrail rather than a formal production SLO.

A production threshold should be calibrated against representative traffic, concurrency, business latency requirements and load testing.