# Operations Runbook

## Purpose

This runbook describes the first response to failures in the fraud-detection MLOps platform.

The primary operational goal is:

> **Preserve the last known-good production model whenever a new release cannot be proven safe.**

---

## 1. CI Failure

### Signal

GitHub Actions CI reports failure.

### Check

```text
compile errors
dependency installation
pytest failures
shared MLOps logic
```

### Response

Do not treat the code as release-ready.

Fix the failing test or code issue before continuing.

---

## 2. Databricks CT Workflow Failure

### Signal

```text
fraud-mlops-ct-workflow
```

fails and Databricks sends a failure notification.

### Check

Inspect the failed task:

```text
feature_drift_monitoring
performance_monitoring
ct_trigger
build_retraining_dataset
export_retraining_snapshot
trigger_azureml_retraining
```

### Response

Do not manually promote a model created from an incomplete CT workflow.

Resolve the failed stage and rerun the workflow.

---

## 3. Azure ML Retraining Failure

### Check

Inspect:

```text
training data asset
training environment
compute
training logs
MLflow run
evaluation output
```

### Response

A failed training run must not produce a release-eligible challenger.

Existing production remains unchanged.

---

## 4. Model Quality-Gate Failure

A quality-gate failure is not necessarily an infrastructure incident.

It means the candidate failed the release policy.

Check:

```text
Recall
PR-AUC
ROC-AUC
threshold selection
data distribution
```

### Response

Do not register or promote the model as an approved challenger.

Investigate data or model behavior before starting another release cycle.

---

## 5. Candidate Deployment Failure

### Signal

Inactive blue/green deployment fails provisioning.

### Response

Do not modify production traffic.

Keep:

```text
current production slot = 100%
candidate slot = 0%
```

Inspect deployment provisioning logs before retrying.

---

## 6. Candidate Smoke-Test Failure

### Signal

The candidate deployment does not successfully answer the direct scoring request.

### Response

Stop the release.

Do not allocate live traffic to the candidate.

Production remains on the current champion.

---

## 7. Canary Failure

### Signal

Validation fails after the 90/10 traffic shift.

### Response

Immediately restore:

```text
previous champion = 100%
candidate = 0%
```

Investigate the candidate before another release attempt.

Azure Machine Learning supports staged blue/green traffic allocation specifically so a new deployment can be tested before full promotion. :chatgpt-content-reference{index="5"}

---

## 8. Endpoint Request-Failure Alert

### Signal

Azure Monitor reports:

```text
4xx / 5xx > 0
```

### Check

```text
HTTP status class
request payload
active deployment
deployment logs
endpoint health
recent release activity
```

Determine whether the issue is:

```text
client request problem
deployment problem
scoring problem
endpoint infrastructure problem
```

---

## 9. High-Latency Alert

### Signal

```text
P95 > 2000 ms
```

### Check

```text
Azure Monitor latency
CPU utilization
memory utilization
disk utilization
recent deployment activity
request volume
```

The current threshold is an initial operational guardrail rather than a formal production SLO.

---

## 10. CT Workflow Identity Failure

The Databricks workflow runs as:

```text
github-fraud-mlops-cd
```

Check:

```text
job Run As identity
Unity Catalog privileges
volume permissions
service credential ACCESS permission
workspace notebook permissions
```

Databricks jobs configured to Run As a service principal execute using that principal's permissions. :chatgpt-content-reference{index="6"}

---

## 11. GitHub → Databricks Authentication Failure

Check:

```text
DATABRICKS_HOST
DATABRICKS_CLIENT_ID
GitHub OIDC permissions
Databricks federation policy
repository / branch subject
```

Databricks supports exchanging GitHub OIDC tokens for Databricks OAuth tokens through workload identity federation. :chatgpt-content-reference{index="7"}

---

## 12. GitHub → Azure Authentication Failure

Check:

```text
GitHub id-token permission
Azure client ID
tenant ID
subscription ID
federated identity subject
Azure RBAC assignment
```

Do not introduce a long-lived client secret merely to bypass an OIDC configuration problem.

---

## 13. Rollback Principle

The previous champion remains deployed in the inactive blue/green slot.

Rollback therefore consists primarily of restoring endpoint traffic to the previous known-good deployment rather than rebuilding the old model.

---

## 14. Recovery Verification

After recovery:

1. Confirm the endpoint returns successful predictions.
2. Confirm production traffic is on the expected deployment.
3. Confirm Azure Monitor metrics are healthy.
4. Confirm no failed CT/CD run is still active.
5. Record the failure cause before another release attempt.