# Security Model

## Objective

The platform is designed to avoid long-lived CI/CD credentials and reduce dependency on personal user identities.

The implementation uses:

- OpenID Connect federation
- workload identity federation
- service principals
- Azure RBAC
- Unity Catalog privileges
- managed identity-backed service access

---

## GitHub Actions → Azure

GitHub Actions authenticates to Microsoft Azure using OIDC federation.

```text
GitHub Actions
      ↓
OIDC token
      ↓
Microsoft Entra ID
      ↓
github-fraud-mlops-cd
      ↓
Azure RBAC
      ↓
Azure Machine Learning
```

No Azure client secret is stored in the GitHub Actions workflow.

Azure access is controlled through RBAC.

---

## GitHub Actions → Azure Databricks

GitHub Actions authenticates to Databricks using workload identity federation.

```text
GitHub Actions
      ↓
GitHub OIDC
      ↓
Databricks federation policy
      ↓
github-fraud-mlops-cd
      ↓
Databricks workspace
```

No Databricks Personal Access Token is required by the GitHub Actions workflow.

Databricks supports GitHub OIDC federation specifically to let automated workloads authenticate without Databricks secrets. :chatgpt-content-reference{index="8"}

---

## Databricks Production Runtime

The production CT workflow executes as:

```text
github-fraud-mlops-cd
```

rather than a personal user.

This removes runtime dependency on an individual user's continued workspace access.

Databricks documents that a job configured to Run As a service principal executes with that principal's identity and permissions regardless of who triggers the run. :chatgpt-content-reference{index="9"}

---

## Unity Catalog Permissions

The production service principal is granted only the permissions required by the workflow.

Examples include:

```text
USE CATALOG
USE SCHEMA
SELECT
MODIFY
CREATE TABLE
READ VOLUME
WRITE VOLUME
ACCESS service credential
CAN MANAGE RUN
```

Permissions are scoped to the required catalog, schema, volume, workflow and service credential.

---

## Azure ML Service Credential

The Databricks service credential used for Azure ML access is:

```text
fraud_aml_managed_identity
```

Unity Catalog service credentials are securable objects used to govern access to external cloud services. Access can be granted through the `ACCESS` privilege. :chatgpt-content-reference{index="10"}

This avoids embedding cloud credentials directly into notebook source code.

---

## GitHub Repository Security

The repository must not contain:

```text
.env files
client secrets
PAT tokens
endpoint keys
storage-account keys
private certificates
Azure access tokens
Databricks access tokens
```

Workspace configuration files may be committed only when they contain resource identifiers/configuration and no authentication secret.

---

## Least-Privilege Principle

Production automation receives only the access necessary to execute its responsibilities.

The project separates:

```text
GitHub source-control identity
Databricks runtime identity
Azure resource authorization
Unity Catalog data authorization
service credential access
```

This limits the effect of a compromised or misconfigured identity.

---

## Secret Handling

Secrets and temporary credentials must never be committed to source control.

Before every public release:

```text
review git diff
review GitHub Actions variables/secrets
review committed configuration files
confirm .env is ignored
confirm tokens/keys are absent
```

---

## Security Principles

The platform follows these principles:

- non-human identities for production automation
- federation over stored long-lived credentials
- explicit runtime identity
- least privilege
- controlled model promotion
- separation of CI, CT and CD responsibilities
- no production secrets committed to Git