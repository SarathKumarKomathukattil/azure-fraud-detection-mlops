# Online Serving Benchmark

## Endpoint

The fraud model is served through:

```text
fraud-identify-endpoint
```

using an Azure Machine Learning Managed Online Endpoint.

Deployment configuration during the observation window:

| Property | Value |
|---|---|
| Endpoint | `fraud-identify-endpoint` |
| Serving platform | Azure ML Managed Online Endpoint |
| Compute SKU | `Standard_DS2_v2` |

---

## Observed Metrics

Azure Monitor measurements observed during validation:

| Metric | Observed Value |
|---|---:|
| P50 request latency | **≈ 8 ms** |
| P95 request latency | **≈ 12 ms** |
| P99 request latency | **≈ 14 ms** |
| CPU utilization | **≈ 0.3%** |
| Memory utilization | **≈ 4.3%** |
| Disk utilization | **≈ 29%** |

---

## Resource Headroom

### CPU

```text
100% - 0.3%
= 99.7%
```

Observed CPU headroom:

```text
≈ 99.7%
```

### Memory

```text
100% - 4.3%
= 95.7%
```

Observed memory headroom:

```text
≈ 95.7%
```

### Disk

```text
100% - 29%
= 71%
```

Observed remaining disk percentage:

```text
≈ 71%
```

---

## Tail Latency

### P95 vs P50

```text
12 ms - 8 ms
= 4 ms
```

### P99 vs P95

```text
14 ms - 12 ms
= 2 ms
```

### P99 / P50

```text
14 / 8
≈ 1.75×
```

During the observed validation window, the endpoint showed relatively low resource pressure and a compact latency distribution.

---

## Important Limitation

These measurements represent an **observed validation workload**.

They are not:

- a production SLA
- a production SLO
- a formal load-test result
- a capacity guarantee

Production sizing should be based on:

```text
representative request volume
concurrency
availability requirements
business latency requirements
formal load testing
```

Azure Machine Learning Managed Online Endpoints support real-time inference and staged production deployment. :chatgpt-content-reference{index="4"}