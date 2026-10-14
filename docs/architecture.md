# Fraud Detection MLOps Architecture

## Overview

This document describes the technical architecture of the fraud-detection MLOps platform.

The system separates:

- data engineering
- model training
- model qualification
- model registry governance
- production monitoring
- Continuous Training
- Continuous Delivery
- online serving
- operational monitoring
- identity and access control

A central design principle is:

> **Training success is not release approval, and release approval is not production promotion.**

---

## End-to-End Architecture

```mermaid
flowchart TB

    subgraph DATA["1 — Data Platform"]
        PAYSIM["PaySim Historical Transactions"]
        ADLS["ADLS Gen2"]
        BRONZE["Bronze Delta"]
        SILVER["Silver Delta"]
        GOLD["Gold / Model Features"]

        PAYSIM --> ADLS
        ADLS --> BRONZE
        BRONZE --> SILVER
        SILVER --> GOLD
    end

    subgraph AML["2 — Azure Machine Learning"]
        DATAASSET["Versioned Azure ML Data Asset"]
        TRAIN["XGBoost Training"]
        MLFLOW["MLflow Experiment Tracking"]
        EVAL["Held-Out Test Evaluation"]
        GATE{"Model Quality Gate"}
        REG["Azure ML Model Registry<br/>release_status = challenger"]

        DATAASSET --> TRAIN
        TRAIN --> MLFLOW
        TRAIN --> EVAL
        EVAL --> GATE
        GATE -->|"PASS"| REG
    end

    GOLD --> DATAASSET

    subgraph MON["3 — Databricks Model Monitoring"]
        DBJOB["fraud-mlops-ct-workflow"]
        SCORE["Production Scoring"]
        PRED["production_predictions"]
        LABELS["Delayed Ground Truth"]
        DRIFT["Feature Drift Monitoring<br/>PSI"]
        PERF["Performance Monitoring<br/>Precision / Recall / F1 / PR-AUC / ROC-AUC"]
        CT{"Continuous Training Decision"}

        SCORE --> PRED
        PRED --> PERF
        LABELS --> PERF

        GOLD --> DRIFT

        DBJOB --> DRIFT
        DBJOB --> PERF

        DRIFT --> CT
        PERF --> CT
    end

    subgraph RETRAIN["4 — Reproducible Retraining"]
        SNAP["Versioned Retraining Snapshot"]
        VOLUME["Unity Catalog External Volume"]
        SNAPADLS["ADLS Training Snapshot"]
        NEWDATA["Versioned Azure ML Data Asset"]

        SNAP --> VOLUME
        VOLUME --> SNAPADLS
        SNAPADLS --> NEWDATA
    end

    CT -->|"retraining required"| SNAP
    NEWDATA --> TRAIN

    subgraph GITHUB["5 — GitHub & GitHub Actions"]
        REPO["GitHub Repository"]
        CI["CI Workflow<br/>compileall + pytest"]
        CD["CT / CD Workflow"]
        RELEASE["azureml/release_model.py"]

        REPO --> CI
        REPO --> CD
        CD --> RELEASE
    end

    CD -->|"OIDC-authenticated run-now"| DBJOB

    REG -->|"approved challenger"| RELEASE

    subgraph SERVING["6 — Safe Online Serving"]
        CANDIDATE["Deploy Challenger<br/>to Inactive Slot"]
        SMOKE["Direct Candidate<br/>Smoke Test"]
        CANARY["90 / 10<br/>Canary Traffic"]
        PROMOTE["100% Promotion"]
        ROLLBACK["Rollback to<br/>Previous Champion"]

        ENDPOINT["fraud-identify-endpoint<br/>Azure ML Managed Online Endpoint"]
        BLUE["Blue Deployment"]
        GREEN["Green Deployment"]

        CANDIDATE --> SMOKE
        SMOKE --> CANARY
        CANARY -->|"success"| PROMOTE
        CANARY -->|"failure"| ROLLBACK

        PROMOTE --> ENDPOINT
        ROLLBACK --> ENDPOINT

        ENDPOINT --- BLUE
        ENDPOINT --- GREEN
    end

    RELEASE --> CANDIDATE

    SCORE -->|"invoke"| ENDPOINT
    ENDPOINT -->|"prediction response"| SCORE

    subgraph OPS["7 — Operational Monitoring & Alerting"]
        AZMON["Azure Monitor"]
        ERRALERT["4xx / 5xx Request Alert"]
        LATALERT["P95 Latency Alert"]
        DBALERT["Databricks CT Failure Notification"]
        GHALERT["GitHub CD Failure Notification"]

        AZMON --> ERRALERT
        AZMON --> LATALERT
    end

    ENDPOINT --> AZMON
    DBJOB --> DBALERT
    CD --> GHALERT

    subgraph SECURITY["8 — Identity & Security"]
        GHOIDC["GitHub Actions<br/>OIDC Token"]

        ENTRA["Microsoft Entra ID<br/>Federated Identity"]
        AZSP["github-fraud-mlops-cd<br/>Service Principal"]
        RBAC["Azure RBAC"]

        DBFED["Databricks Workload<br/>Identity Federation"]
        DBSP["github-fraud-mlops-cd<br/>Databricks Service Principal"]

        UC["Unity Catalog ACLs"]
        SERVICECRED["fraud_aml_managed_identity<br/>Service Credential"]

        GHOIDC --> ENTRA
        ENTRA --> AZSP
        AZSP --> RBAC

        GHOIDC --> DBFED
        DBFED --> DBSP

        DBSP --> UC
        UC --> SERVICECRED
    end

    CD -. "GitHub → Azure authentication" .-> GHOIDC

    AZSP -. "credential used by release controller" .-> RELEASE
    RBAC -. "model registry access" .-> REG
    RBAC -. "endpoint deployment access" .-> ENDPOINT

    DBSP -. "Run As identity" .-> DBJOB

    SERVICECRED -. "managed identity access to Azure ML" .-> TRAIN
```

---

## Data Plane

Historical PaySim transaction data enters the platform through ADLS Gen2 and is processed through Bronze, Silver and Gold Delta layers.

The Gold layer contains model-ready features used by:

- initial model training
- drift monitoring
- production monitoring
- retraining

---

## Model Training and Qualification

Azure Machine Learning owns the model-development boundary.

The lifecycle is:

```text
Versioned data
      ↓
XGBoost training
      ↓
MLflow experiment tracking
      ↓
validation-based threshold selection
      ↓
held-out test evaluation
      ↓
quality gate
      ↓
model registration
```

A training run cannot become deployment eligible unless it passes the configured model-quality gate.

---

## CT/CD Boundary

Continuous Training and Continuous Delivery are intentionally decoupled.

```text
Continuous Training
        ↓
qualified challenger
        ↓
Azure ML Model Registry
        ↓
Continuous Delivery
        ↓
production
```

The registry is the formal contract between the two systems.

---

## Continuous Training

Databricks owns model-health monitoring and retraining orchestration.

The production workflow is:

```text
fraud-mlops-ct-workflow
```

The workflow combines:

- feature drift
- delayed-label performance
- retraining policy
- versioned dataset creation
- Azure ML retraining trigger

Retraining is triggered only when an explicit monitoring condition is breached.

---

## Safe Production Release

The production endpoint uses two deployment slots:

```text
blue
green
```

A challenger is first deployed to the inactive slot.

Release sequence:

```text
inactive deployment
        ↓
direct smoke test
        ↓
90 / 10 traffic split
        ↓
candidate validation
        ↓
100% promotion
```

If validation fails after traffic shifting starts, production traffic is restored to the previous champion.

Azure Machine Learning documents this blue/green staged-traffic pattern as a safe rollout strategy. :chatgpt-content-reference{index="1"}

---

## Operational Monitoring

Operational serving health is monitored independently from model quality.

Azure Monitor tracks:

```text
HTTP failures
P95 request latency
endpoint operational health
```

Databricks tracks:

```text
feature drift
delayed-label model performance
CT state
workflow health
```

---

## Identity Boundaries

### GitHub → Azure

```text
GitHub Actions
      ↓
OIDC
      ↓
Microsoft Entra ID
      ↓
Service Principal
      ↓
Azure RBAC
      ↓
Azure ML
```

### GitHub → Databricks

```text
GitHub Actions
      ↓
GitHub OIDC
      ↓
Databricks Workload Identity Federation
      ↓
Databricks Service Principal
      ↓
fraud-mlops-ct-workflow
```

Databricks supports GitHub OIDC workload identity federation so external CI/CD workloads can authenticate without storing Databricks secrets. :chatgpt-content-reference{index="2"}

### Databricks Runtime

The production workflow executes as:

```text
github-fraud-mlops-cd
```

A Databricks job configured to Run As a service principal executes using that principal's identity and permissions regardless of who triggers the run. :chatgpt-content-reference{index="3"}

---

## Failure Containment

The architecture is designed to prevent one lifecycle failure from silently propagating into production.

Examples:

```text
CI failure
→ code is not release-ready

monitoring failure
→ CT stops

training failure
→ no challenger

quality-gate failure
→ no release-eligible model

candidate deployment failure
→ production remains unchanged

smoke-test failure
→ promotion stops

canary failure
→ traffic returns to previous champion
```