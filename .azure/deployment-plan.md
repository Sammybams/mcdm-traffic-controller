# Azure Deployment Plan

> **Status:** Planning

Generated: 2026-08-31

---

## 1. Project Overview

**Goal:** Deploy the existing MCDM Traffic Vision FastAPI service to Azure App
Service while preserving single-road and four-road inference, model checksum
verification, health checks, and API-key protection.

**Path:** Modernize Existing

The existing Render configuration will remain available unless the user later
asks to remove it. Azure support will be added alongside it.

---

## 2. Requirements

| Attribute | Value |
|---|---|
| Classification | Pending user confirmation; recommended: POC |
| Scale | Pending user confirmation; recommended: Small |
| Budget | Pending user confirmation; recommended: Cost-Optimized |
| Compliance | Pending user confirmation; no requirement stated yet |
| Subscription | Pending Azure authentication and user confirmation |
| Location | Pending user confirmation; South Africa North is the latency-oriented recommendation for Lagos |

Azure CLI and Azure Developer CLI are not currently installed on the
workstation. Subscription policies and regional quotas cannot be checked until
the user approves installation and signs in.

---

## 3. Components Detected

| Component | Type | Technology | Path |
|---|---|---|---|
| traffic-vision-api | API service | Python 3.11+, FastAPI, Uvicorn | `src/traffic_vision/api.py` |
| inference runtime | ML inference | Ultralytics YOLO11n, PyTorch | `src/traffic_vision/detector.py` |
| geometry pipeline | Deterministic processing | Python | `src/traffic_vision/road_processor.py` |
| model artifact | Binary deployment asset | YOLO `.pt`, SHA-256 verified | `artifacts/research/toy-vehicle-prelabel.pt` |

### Dependencies

| Component | Depends On | Type |
|---|---|---|
| traffic-vision-api | inference runtime | In-process Python dependency |
| inference runtime | model artifact | Startup dependency |
| geometry pipeline | four-road configuration | Versioned JSON configuration |

### Existing Infrastructure

| Item | Status |
|---|---|
| FastAPI health and inference endpoints | Implemented and tested |
| `render.yaml` | Present; non-Azure deployment target |
| `azure.yaml` | Not found |
| `infra/*.bicep` | Not found |
| Dockerfile | Not found; native Linux App Service is preferred |
| Specialized Copilot/Azure Functions SDKs | Not detected |

---

## 4. Recipe Selection

**Selected:** AZD with Bicep, pending user approval

**Rationale:** This is an Azure-only, single-service modernization with no
existing infrastructure-as-code. Azure Developer CLI provides repeatable
environment management and deployment while Bicep is the default Azure-native
infrastructure language.

---

## 5. Architecture

**Stack:** Native Linux Azure App Service

### Service Mapping

| Component | Azure Service | Proposed SKU |
|---|---|---|
| FastAPI and YOLO inference | Linux Azure App Service | Basic B1, one instance |
| Compute plan | Azure App Service Plan | Basic B1 |
| Model artifact | Azure Blob Storage, private container | Standard LRS |
| API key | Azure Key Vault | Standard |
| Logs and traces | Application Insights | Workspace-based |
| Central logs | Log Analytics workspace | Pay-as-you-go |

### Security and Runtime Decisions

- Enable a system-assigned managed identity on the web app.
- Grant only `Storage Blob Data Reader` for the private model container.
- Grant only `Key Vault Secrets User` for the API-key secret.
- Disable storage shared-key access and keep the model container private.
- Resolve the API key through an App Service Key Vault reference.
- Verify the model SHA-256 before loading it.
- Run one Uvicorn worker because each worker loads a separate model copy.
- Configure `/health` as the App Service health-check path.
- Use HTTPS-only, TLS 1.2 or newer, FTPS disabled, and managed identity instead
  of embedded Azure credentials.
- Keep the existing application-level `X-API-Key` check for the prototype.

### Model Flow

```text
versioned local model
  -> deployment uploads model to private Blob Storage
  -> App Service managed identity downloads it at startup
  -> SHA-256 verification
  -> one in-memory YOLO detector
  -> temporary request images deleted after inference
```

### Sizing Rationale

Free/shared compute is not recommended for PyTorch inference. Basic B1 provides
dedicated compute and 1.75 GB RAM, which is the cost-oriented starting point.
Memory and latency will be measured after deployment; scaling up requires user
approval because it changes cost.

---

## 6. Provisioning Limit Checklist

Resource inventory is prepared. Usage, quota, and policy checks are blocked
until Azure CLI authentication, subscription confirmation, and region
confirmation are complete.

| Resource Type | Number to Deploy | Total After Deployment | Limit/Quota | Notes |
|---|---:|---:|---:|---|
| `Microsoft.Web/serverfarms` | 1 | Pending context | Pending quota check | Linux Basic B1 plan |
| `Microsoft.Web/sites` | 1 | Pending context | Pending quota check | FastAPI web app |
| `Microsoft.Storage/storageAccounts` | 1 | Pending context | Pending quota check | Private model artifact |
| `Microsoft.KeyVault/vaults` | 1 | Pending context | Pending quota check | API key |
| `Microsoft.OperationalInsights/workspaces` | 1 | Pending context | Pending quota check | Central logs |
| `Microsoft.Insights/components` | 1 | Pending context | Pending quota check | Application Insights |

**Status:** Blocked on Azure context; this plan must not be approved or executed
until all quota cells contain checked values.

---

## 7. Execution Checklist

### Phase 1: Planning

- [x] Create deployment-plan skeleton before Azure work.
- [x] Check prompt and codebase for specialized technology routing.
- [x] Analyze workspace in MODERNIZE mode.
- [x] Scan codebase and current deployment configuration.
- [x] Select the proposed AZD/Bicep recipe.
- [x] Draft the App Service architecture.
- [ ] Confirm classification, scale, budget, and compliance.
- [ ] Install/authenticate Azure CLI and Azure Developer CLI with approval.
- [ ] Confirm subscription and location with the user.
- [ ] Check Azure Policy assignments.
- [ ] Invoke `azure-quotas` and complete provisioning-limit checks.
- [ ] Finalize and obtain user approval for this plan.

### Phase 2: Execution

- [ ] Research all selected Azure components.
- [ ] Generate `azure.yaml` and Bicep infrastructure.
- [ ] Add managed-identity model download and Azure dependencies.
- [ ] Add Azure startup/deployment configuration.
- [ ] Apply security hardening.
- [ ] Run local functional and unit tests.
- [ ] Update status to `Ready for Validation`.

### Phase 3: Validation

- [ ] Invoke `azure-validate`.
- [ ] Record validation proof.
- [ ] Update status to `Validated`.

### Phase 4: Deployment

- [ ] Invoke `azure-deploy` only after validation.
- [ ] Verify `/health`, `/docs`, authenticated single-road inference, and
  authenticated four-road inference.
- [ ] Record endpoint and update status to `Deployed`.

---

## 8. Validation Proof

No Azure validation has run. The plan is still in planning status.

---

## 9. Files to Generate After Approval

| File | Purpose | Status |
|---|---|---|
| `.azure/deployment-plan.md` | Deployment source of truth | In progress |
| `azure.yaml` | Azure Developer CLI service definition | Waiting for approval |
| `infra/main.bicep` | Subscription/resource-group deployment entry point | Waiting for approval |
| `infra/resources.bicep` | App Service and supporting resources | Waiting for approval |
| `scripts/download_model_from_azure.py` | Managed-identity model bootstrap | Waiting for approval |
| `requirements.txt` | App Service/Oryx production dependencies | Waiting for approval |

---

## 10. Next Steps

1. Obtain the missing workload and Azure context decisions from the user.
2. Install/authenticate Azure tooling with permission.
3. Check policy and quota, then present the completed plan for approval.
