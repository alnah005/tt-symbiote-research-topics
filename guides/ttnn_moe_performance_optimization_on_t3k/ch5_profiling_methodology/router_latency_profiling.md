# Router Latency Profiling

## Context

This file addresses **Q7**: What is the latency cost of the 3-pass BF16 centering trick in `TTNNMoERouterDecode.forward` (`moe.py:L891–L1024`), and what is the precision benefit (routing decision agreement rate) compared to a single-pass `topk` baseline?

Source range: `moe.py:L891–L1024` (`TTNNMoERouterDecode.forward`), with focus on the `n_group <= r.topk_group` branch at `moe.py:L925–L957`.

---

## What the 3-Pass BF16 Centering Trick Does

BF16 has 7 mantissa bits, giving it a minimum step size of approximately 0.0078 near 1.0. When two expert scores are separated by less than one BF16 step, a single-pass `ttnn.topk` in BF16 may compare two values that round to the same BF16 representation, and will then select based on index rather than value — producing incorrect routing decisions.

The centering trick cancels the common offset from all expert scores before each `topk`, so that the surviving comparisons happen near 0.0, where BF16's absolute resolution is at its finest (BF16 values near 0 have step size ~6e-5, roughly 130× finer than near 1.0).

### The Three Passes (moe.py:L925–L957)

```python
# Pass 1: rough BF16 topk(k+1) to find coarse threshold
scores_bf16_p1 = ttnn.typecast(scores_with_bias_f32, ttnn.bfloat16)
rough_vals, _ = ttnn.topk(scores_bf16_p1, k=top_k + 1, dim=3, largest=True, sorted=True)
rough_thr_bf16 = ttnn.slice(rough_vals, [0,0,0,top_k], [1,1,T,top_k+1])
rough_thr_f32 = ttnn.typecast(rough_thr_bf16, ttnn.float32)
# Center scores around decision boundary
scores_c1 = ttnn.sub(scores_with_bias_f32, rough_thr_f32)

# Pass 2: refined BF16 topk(k+1) on centered scores
scores_bf16_p2 = ttnn.typecast(scores_c1, ttnn.bfloat16)
refined_vals, _ = ttnn.topk(scores_bf16_p2, k=top_k + 1, dim=3, largest=True, sorted=True)
refined_thr_bf16 = ttnn.slice(refined_vals, [0,0,0,top_k], [1,1,T,top_k+1])
refined_thr_f32 = ttnn.typecast(refined_thr_bf16, ttnn.float32)
scores_c2 = ttnn.sub(scores_c1, refined_thr_f32)

# Pass 3: final topk(k) on doubly-centered scores
scores_bf16_final = ttnn.typecast(scores_c2, ttnn.bfloat16)
_, topk_expert_idx = ttnn.topk(scores_bf16_final, k=top_k, dim=3, largest=True, sorted=True)
```

**Why `topk(k+1)` in passes 1 and 2:** The threshold is extracted at position `top_k` (the `(k+1)`-th largest value, 0-indexed). This is the score just below the decision boundary — the largest score that will NOT be selected. Centering on this value places the boundary near 0.0 while keeping all top-k scores positive and all non-top-k scores negative after centering.

**Why two centering passes instead of one:** After pass 1's centering, the threshold is a BF16-rounded estimate of the true F32 threshold. There may still be a small residual offset (up to one BF16 step of the coarse threshold value). Pass 2 refines this by re-centering on the BF16-precise threshold of the already-centered scores. After two centerings, the residual is at most one BF16 step of a value near 0 — typically < 1e-4 — which is below the score separation of any expert pair that genuinely differs.

**Which path does NOT use 3-pass centering:** The group-based path (`n_group > topk_group`, executed when the number of expert groups exceeds the group selection limit) does not use this trick. It performs group-level selection first, then expert-level `topk` within selected groups. The score range after group selection is narrower, reducing (but not eliminating) the BF16 precision risk.

---

## Isolation Harness

The harness below runs `TTNNMoERouterDecode.forward` in isolation, without executing `TTNNExperts.forward`. This eliminates the expert compute time from the measurement, making the router's contribution measurable.

```python
# profile_router_latency.py
"""
Isolates TTNNMoERouterDecode.forward to measure:
  1. Latency of the 3-pass BF16 centering trick (production path)
  2. Latency of a single-pass topk baseline (ablation)
  3. Routing decision agreement rate between the two paths
"""
import os
import statistics
import time
from typing import Tuple, List

import ttnn
import torch


DEVICE_CLOCK_MHZ = 1000  # Wormhole B0


def build_dummy_scores(batch: int, n_experts: int, device, dtype=ttnn.float32) -> ttnn.Tensor:
    """
    Build a (1, 1, batch, n_experts) F32 tensor of dummy router logits on device.
    Values are drawn from N(0, 0.15) to approximate the distribution of real logits
    after the bias addition step (moe.py:L921–L924).
    """
    scores_torch = torch.randn(1, 1, batch, n_experts) * 0.15
    return ttnn.from_torch(
        scores_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def three_pass_centering_topk(scores_f32: ttnn.Tensor, top_k: int) -> Tuple[ttnn.Tensor, float]:
    """
    Runs the production 3-pass BF16 centering trick and returns
    (expert_indices, elapsed_us).
    Uses device-side cycle counters via TT_METAL_DEVICE_PROFILER.
    """
    t0 = time.perf_counter()

    # Pass 1
    scores_bf16_p1 = ttnn.typecast(scores_f32, ttnn.bfloat16)
    rough_vals, _ = ttnn.topk(scores_bf16_p1, k=top_k + 1, dim=3, largest=True, sorted=True)
    rough_thr_bf16 = ttnn.slice(rough_vals,
                                begins=[0, 0, 0, top_k],
                                ends=[1, 1, scores_f32.shape[2], top_k + 1])
    rough_thr_f32 = ttnn.typecast(rough_thr_bf16, ttnn.float32)
    scores_c1 = ttnn.sub(scores_f32, rough_thr_f32)

    # Pass 2
    scores_bf16_p2 = ttnn.typecast(scores_c1, ttnn.bfloat16)
    refined_vals, _ = ttnn.topk(scores_bf16_p2, k=top_k + 1, dim=3, largest=True, sorted=True)
    refined_thr_bf16 = ttnn.slice(refined_vals,
                                  begins=[0, 0, 0, top_k],
                                  ends=[1, 1, scores_f32.shape[2], top_k + 1])
    refined_thr_f32 = ttnn.typecast(refined_thr_bf16, ttnn.float32)
    scores_c2 = ttnn.sub(scores_c1, refined_thr_f32)

    # Pass 3
    scores_bf16_final = ttnn.typecast(scores_c2, ttnn.bfloat16)
    _, topk_idx = ttnn.topk(scores_bf16_final, k=top_k, dim=3, largest=True, sorted=True)

    ttnn.synchronize_device(scores_f32.device())
    elapsed_us = (time.perf_counter() - t0) * 1e6

    return topk_idx, elapsed_us


def single_pass_topk(scores_f32: ttnn.Tensor, top_k: int) -> Tuple[ttnn.Tensor, float]:
    """
    Runs a single-pass BF16 topk (the baseline without centering) and returns
    (expert_indices, elapsed_us).
    """
    t0 = time.perf_counter()

    scores_bf16 = ttnn.typecast(scores_f32, ttnn.bfloat16)
    _, topk_idx = ttnn.topk(scores_bf16, k=top_k, dim=3, largest=True, sorted=True)

    ttnn.synchronize_device(scores_f32.device())
    elapsed_us = (time.perf_counter() - t0) * 1e6

    return topk_idx, elapsed_us


def routing_agreement_rate(
    idx_ref: ttnn.Tensor,
    idx_test: ttnn.Tensor,
) -> float:
    """
    Compute the fraction of (token, slot) pairs where the selected expert index
    matches between idx_ref and idx_test.

    idx_ref, idx_test: shape (1, 1, T, top_k) on device.
    Returns a float in [0, 1].
    """
    ref_torch  = ttnn.to_torch(idx_ref).squeeze(0).squeeze(0)   # (T, top_k)
    test_torch = ttnn.to_torch(idx_test).squeeze(0).squeeze(0)  # (T, top_k)

    # Sort each token's selections before comparing (order within top_k is arbitrary)
    ref_sorted,  _ = ref_torch.sort(dim=1)
    test_sorted, _ = test_torch.sort(dim=1)

    matches = (ref_sorted == test_sorted).float().mean().item()
    return matches


def run_router_latency_sweep(
    device,
    n_experts: int = 128,
    top_k: int = 8,
    batch_sizes: List[int] = [1, 4, 8, 16],
    n_warmup: int = 5,
    n_profile: int = 20,
):
    """
    Sweeps over batch sizes and measures:
      - 3-pass centering latency (median over n_profile runs)
      - Single-pass topk latency (median)
      - Routing agreement rate between the two paths
    """
    print(f"\n{'Batch':>6} | {'3-pass µs':>10} | {'1-pass µs':>10} | {'Overhead %':>11} | {'Agreement':>10}")
    print("-" * 58)

    for batch in batch_sizes:
        scores = build_dummy_scores(batch, n_experts, device)

        # Warmup
        for _ in range(n_warmup):
            _, _ = three_pass_centering_topk(scores, top_k)
            _, _ = single_pass_topk(scores, top_k)

        # Profile
        times_3pass = []
        times_1pass = []
        agreements  = []

        for _ in range(n_profile):
            idx_3pass, t3 = three_pass_centering_topk(scores, top_k)
            idx_1pass, t1 = single_pass_topk(scores, top_k)
            times_3pass.append(t3)
            times_1pass.append(t1)
            # Use F32 topk as the ground-truth reference for agreement
            _, idx_f32_ref = torch.topk(
                ttnn.to_torch(scores).squeeze(0).squeeze(0),
                k=top_k,
                dim=1,
                largest=True,
            )
            idx_f32_ttnn = ttnn.from_torch(
                idx_f32_ref.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            agreements.append(routing_agreement_rate(idx_f32_ttnn, idx_3pass))

        med_3pass = statistics.median(times_3pass)
        med_1pass = statistics.median(times_1pass)
        overhead_pct = (med_3pass - med_1pass) / med_1pass * 100
        mean_agreement = statistics.mean(agreements)

        print(
            f"{batch:>6} | {med_3pass:>10.1f} | {med_1pass:>10.1f} | "
            f"{overhead_pct:>10.1f}% | {mean_agreement:>9.4f}"
        )

        ttnn.deallocate(scores)
```

---

## Step-by-Step: Running the Isolation Harness

```bash
# 1. Set environment
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_DEVICE_PROFILER_OUTPUT=/tmp/router_profiler
mkdir -p /tmp/router_profiler

# 2. Run the sweep
python -c "
import ttnn
device = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 8),
    dispatch_core_type=ttnn.DispatchCoreType.WORKER,
)
from profile_router_latency import run_router_latency_sweep
run_router_latency_sweep(
    device,
    n_experts=128,
    top_k=8,
    batch_sizes=[1, 4, 8, 16],
    n_warmup=5,
    n_profile=20,
)
ttnn.close_mesh_device(device)
"
```

---

## Precision Measurement: Routing Decision Agreement Rate

The agreement rate is the fraction of (token, selected-expert-slot) pairs where the 3-pass result matches the F32 ground truth. The F32 ground truth is obtained by running `torch.topk` on the logit tensor after converting it back to F32 on CPU.

A value of 1.0000 means every token chose the same set of experts as the F32 reference. A value of 0.9900 means 1% of (token, slot) pairs chose a different expert — at `top_k=8` and batch=1, this implies roughly 1 in 12.5 tokens has at least one misrouted expert slot.

### Expected Results

At batch=1 decode with `n_experts=128`, `top_k=8`, and logits drawn from N(0, 0.15):

| Method | Expected agreement vs F32 | Notes |
|---|---|---|
| 3-pass BF16 centering | ~0.9990–1.0000 | Misrouting occurs only when two experts are within ~1e-4 of each other |
| Single-pass BF16 topk | ~0.9700–0.9950 | Misrouting when two experts are within one BF16 step (~0.0078) of each other |

The gap between the two agreement rates narrows at larger batch sizes because the statistical likelihood of a near-tie between two experts is independent of batch size — but the per-token routing error rate remains the same.

### Latency Overhead of the 3-Pass Trick

Each additional pass adds:
- 2× `ttnn.typecast` (F32 → BF16)
- 1× `ttnn.topk`
- 1× `ttnn.slice`
- 1× `ttnn.typecast` (BF16 → F32)
- 1× `ttnn.sub`

At batch=1 decode on T3K, each pass executes on the full (1, 1, 1, 128) logit tensor. The tensor is trivially small — 128 elements — so device kernel launch overhead dominates over compute time. Each additional pass is expected to add ~10–15 µs of device-side latency (mostly command-queue dispatch and kernel launch overhead, not arithmetic).

The production 3-pass path thus costs approximately **~2.3×–3× the latency of a single-pass topk**. At batch=1, if a single-pass topk takes ~15 µs, the 3-pass variant takes ~35–45 µs. As a fraction of the total `TTNNMoE.forward` latency (~1.5–2 ms), this is roughly 2–3% — a small but measurable cost.

---

## Device-Side Cycle Counter Measurement

For sub-microsecond resolution on the individual `typecast` and `topk` ops within the router, use the TTNN op timer CSV approach from `ttnn_op_timer_profiling.md` with `TT_METAL_DEVICE_PROFILER=1`. Filter the CSV for rows whose `op_name` matches `topk` and whose dispatch timestamp falls within the router's wall-clock window:

```python
def extract_router_ops(rows, router_start_us, router_end_us):
    """
    Filter op timer CSV rows to those dispatched during the router's execution window.
    router_start_us, router_end_us: wall-clock timestamps from time.perf_counter() × 1e6.
    """
    router_ops = []
    for row in rows:
        host_ts = float(row.get("host_start_time_us", 0))
        if router_start_us <= host_ts <= router_end_us:
            router_ops.append(row)
    return router_ops
```

Wrap the `three_pass_centering_topk` call with `time.perf_counter()` before and after, convert to µs, and pass the bounds to `extract_router_ops` after reading the profiler CSV. The resulting subset will contain exactly the `typecast`, `topk`, `slice`, and `sub` ops belonging to the 3-pass centering computation.

---

## Interpreting the Results: Go / No-Go for Removing the Centering Trick

The centering trick can be safely removed (replaced with a single-pass BF16 topk) only if the agreement rate of single-pass BF16 topk against F32 is ≥ 0.9999 on the actual model's logit distribution. The N(0, 0.15) synthetic distribution used in the harness above is an approximation; for a definitive measurement, substitute real logits captured from a production inference run.

If the routing agreement rate with single-pass topk is < 0.9999, removing the centering trick will degrade routing quality. Whether the latency saving (~20–30 µs per forward pass) justifies a small degradation in routing quality is a model-quality question outside the scope of this profiling guide.

---

Next: [Chapter 6 — CPU Fallback Elimination](../ch6_cpu_fallback_elimination/index.md)
