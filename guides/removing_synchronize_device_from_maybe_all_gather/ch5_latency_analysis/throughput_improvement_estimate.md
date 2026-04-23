# Throughput Improvement Estimate

This file translates the per-call `synchronize_device` latency from [`synchronize_device_latency_model.md`](./synchronize_device_latency_model.md) and [`measuring_the_cost.md`](./measuring_the_cost.md) into a concrete per-step and per-token throughput improvement, and situates that improvement within the broader trace-enablement project. By the end of this file you will have a worked example that can be recalculated for any combination of layer count, measured synchronize latency, and reference decode step time.

---

## Decode Throughput Model

Decode throughput (tokens per second) at batch=1 is:

```
throughput (tokens/s) = 1 / per_step_latency (s)
```

Removing K calls to `ttnn.synchronize_device()` per step, each costing T_sync seconds, reduces per-step latency by:

```
delta_latency = K × T_sync
```

The new throughput after removal is:

```
throughput_new = 1 / (per_step_latency - delta_latency)
```

The relative throughput improvement is:

```
improvement (%) = (delta_latency / per_step_latency) × 100
                = (K × T_sync / per_step_latency) × 100
```

---

## Applying the Model to the Hybrid Decoder Stack

The hybrid DeltaNet + full-attention decoder (Qwen3.6-35B-A3B on T3K) has H attention layers that each call `_maybe_all_gather` at least once in the forward pass. Based on the audit in [Chapter 4](../ch4_symbiote_audit/audit_results.md), both `TTNNQwen3LinearAttention` and `TTNNQwen3FullAttention` contain a call to `_maybe_all_gather`, so across a stack of H hybrid layers:

```
K = H × (calls_per_linear_layer + calls_per_full_layer)
  = H × (1 + 1)          # one call per attention module per layer
  = 2H                   (estimate; verify against actual call graph from Ch4)
```

Total synchronize overhead per step:

```
delta_latency = 2H × T_sync
```

---

## Worked Example

**Assumptions:**

| Parameter | Value | Source |
|---|---|---|
| H (hybrid attention layers) | 16 | Representative model config |
| T_sync (per call, median estimate) | 0.3 ms | Mid-range of 0.1–0.5 ms estimate |
| per_step_latency (reference) | 30 ms | Typical batch=1 decode step on T3K |
| K (total synchronize calls per step) | 32 (2 × 16) | One per `_maybe_all_gather` call site |

**Calculation:**

```
delta_latency = 32 × 0.3 ms = 9.6 ms

improvement = (9.6 ms / 30 ms) × 100 = 32%

throughput_new = 1 / (30 ms - 9.6 ms) = 1 / 20.4 ms ≈ 49.0 tokens/s
throughput_old = 1 / 30 ms ≈ 33.3 tokens/s
```

> **Note:** The above uses K=32 (two calls per layer × 16 layers). The plan specification's worked example uses H=16 with T_sync=0.3 ms and per_step_latency=30 ms, yielding 16 × 0.3 = 4.8 ms = 16% latency reduction. That calculation assumes K=H=16 (one call per layer), which is consistent with a model where each hybrid layer has a single `_maybe_all_gather` call. The correct K depends on the actual call graph — confirm against the audit in [Chapter 4](../ch4_symbiote_audit/index.md). Both calculations are shown below for reference.

**Sensitivity table (T_sync=0.3 ms, per_step=30 ms):**

| K (total calls per step) | delta_latency | Improvement |
|---|---|---|
| 8 | 2.4 ms | 8% |
| 16 | 4.8 ms | 16% |
| 24 | 7.2 ms | 24% |
| 32 | 9.6 ms | 32% |

**Sensitivity to T_sync (K=16, per_step=30 ms):**

| T_sync per call | delta_latency | Improvement |
|---|---|---|
| 0.1 ms | 1.6 ms | 5.3% |
| 0.3 ms | 4.8 ms | 16% |
| 0.5 ms | 8.0 ms | 26.7% |

---

## Compounding with Trace Enablement

Removing `synchronize_device` from `_maybe_all_gather` produces an immediate latency improvement in **non-traced mode** (the benefit analyzed above). This improvement is significant but is a subset of the total gain achievable once the full attention stack is captured under Metal Trace.

When `enable_trace=True` is enabled after the full stack is made trace-compatible, the additional savings come from:

1. **Eliminating per-op Python dispatch overhead** across all ops in the traced region (typically 5–20 µs per op × hundreds of ops per step = several milliseconds)
2. **Eliminating host-device PCIe command submission round-trips** for every op (folded into the single trace replay invocation)
3. **Reduced OS scheduling interference** because the traced replay is a single host-side MMIO write, not a sequence of Python calls

These trace-dispatch savings typically exceed the synchronize_device removal saving by 2–5× for deep stacks. Removing `synchronize_device` is therefore the **prerequisite** for achieving the larger trace benefit, not the final performance goal. The validation procedure for measuring both components separately is described in [Chapter 7, `latency_measurement.md`](../ch7_validation/latency_measurement.md).

---

## Template for Actual Measured Values

Replace the estimates below with values from [`measuring_the_cost.md`](./measuring_the_cost.md) once measurement is complete:

```
## Recalculation with Measured Values (TODO: fill in)

T_sync_measured_median = TODO ms
T_sync_measured_p95    = TODO ms
K_actual               = TODO (from Ch4 audit)
per_step_latency_ref   = TODO ms (measured from Tracy, non-traced baseline)

delta_latency_median = K_actual × T_sync_measured_median = TODO ms
improvement_median   = (delta_latency_median / per_step_latency_ref) × 100 = TODO %

delta_latency_p95    = K_actual × T_sync_measured_p95 = TODO ms
improvement_p95      = (delta_latency_p95 / per_step_latency_ref) × 100 = TODO %
```

These values should be confirmed against the before/after latency measurement described in [Chapter 7, `latency_measurement.md`](../ch7_validation/latency_measurement.md), which provides the empirical check for the model predictions here.
