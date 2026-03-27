# Plan: Speed Up MoE Module for Ling-mini-2.0 Decode

**Date:** 2026-03-27 (updated)
**Model:** Ling-mini-2.0 (inclusionAI/Ling-mini-2.0)
**Module:** `TTNNBailingMoE` (subclass of `TTNNMoE`) in `models/experimental/tt_symbiote/modules/moe.py`
**Test:** `pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py --timeout=0`
**Environment:** MESH_DEVICE=T3K (1x8), Wormhole B0
**Success Criteria:** Correct text generation + measurable decode speedup

---

## 1. Implementation Status Summary

| Step | Description | Status | Notes |
|------|-------------|--------|-------|
| 1.1 | Simplify weight application | **DONE** | Broadcast multiply replaced 7-op repeat pattern |
| 1.2 | Eliminate pre-gate typecast | **DONE** | `ttnn.linear` with `dtype=ttnn.float32` output |
| 1.3 | Reduce layout conversions | **PARTIAL** | Some eliminated; 6 remain (unavoidable for dispatch/sparse_matmul) |
| 2.1 | Shared expert overlap | **DONE** | `shared_experts(residual)` called before `experts.forward()` |
| 3.1 | Router optimization | **NO CHANGE NEEDED** | Single-pass topk path correct for n_group=1, topk_group=1 |
| 3.2 | Remove router CPU sort | **NO CHANGE NEEDED** | CPU sort only in GLM-4 path, not Bailing path |
| 4.1 | Trace capture for MoE | **REMAINING** | High impact, high complexity |
| 5.1 | Sweep sparse_matmul config | **REMAINING** | Low risk, low-medium impact |
| 5.2 | LoFi math fidelity | **REMAINING** | Low risk, medium impact |
| 6.1 | CCL parameter sweep | **REMAINING** | Low risk, low impact |

**Completed optimizations saved an estimated 0.9-2.1ms/token across 19 MoE layers.**

---

## 2. Remaining Optimizations (Prioritized)

### Priority 1: Trace Capture for TTNNBailingMoE (HIGH IMPACT, HIGH COMPLEXITY)

**Why:** At batch=1 decode, host dispatch overhead dominates device compute time. The MoE forward contains ~40-50 TTNN ops per layer. At ~5-10us host dispatch per op, that is 200-500us per layer, or 3.8-9.5ms across 19 layers. Trace capture eliminates this entirely.

**File:** `modules/moe.py`

**What to change:**

1. Add `@trace_enabled` decorator to `TTNNBailingMoE`:

```python
from models.experimental.tt_symbiote.core.run_config import trace_enabled

@trace_enabled
class TTNNBailingMoE(TTNNMoE):
    ...
```

2. In the test, enable traced execution mode:

```python
from models.experimental.tt_symbiote.core.run_config import set_run_mode, TracedRun
set_run_mode("TRACED")
TracedRun.configure(device=mesh_device)
```

**Challenges and mitigations:**

- **Dynamic control flow:** All branches in `TTNNExperts.forward` are shape-dependent or dtype-dependent, not data-dependent. For fixed batch=1 decode, branches are deterministic after the first call. Trace capture should work.
  - `topk_experts_indices.dtype != ttnn.uint16` (L1161) -- constant after first call
  - `num_tokens % SPARSITY_BLOCK_SIZE != 0` (L1172) -- constant for fixed batch
  - `pad_amount > 0` (L1312) -- constant for fixed batch
  - `T == 1` in router (L910, L986) -- constant for decode

- **Dynamic routing metadata:** `all_to_all_dispatch` produces `all_to_all_dispatch_metadata` that varies with expert selection each token. This metadata is consumed by `all_to_all_combine` and `moe_expert_token_remap`. **This is the key risk.** If these ops store metadata in trace-captured command buffers, replay will use stale metadata.

  **Mitigation:** Check whether `all_to_all_dispatch`, `all_to_all_combine`, and `moe_expert_token_remap` are trace-compatible. If not, the trace boundary must be split:
  - Trace A: all_gather -> gate linear -> router (deterministic ops)
  - Non-traced: all_to_all_dispatch (produces dynamic metadata)
  - Trace B: sparse_matmul pipeline (deterministic given fixed shapes)
  - Non-traced: all_to_all_combine (consumes dynamic metadata)
  - Trace C: weight application -> reduce_scatter -> add shared

  This partial trace approach still eliminates dispatch overhead for the majority of ops.

- **Shared expert overlap:** With trace capture, the shared expert call at L1438 may no longer overlap with routed experts since trace replays ops in a fixed order. Verify that the shared expert call is included in the trace and that async dispatch semantics are preserved within the trace.

**Expected savings:** 2-5ms/token (eliminates host dispatch for 40-50 ops x 19 layers)
**Risk:** High. Requires verification that all_to_all ops are trace-compatible.
**Test:** Run 128 tokens, verify output matches non-traced output exactly.

---

### Priority 2: Sweep sparse_matmul Program Config (LOW RISK, LOW-MEDIUM IMPACT)

**Why:** The current `in0_block_w=min(4, hidden_tiles)` and `per_core_M=1` were chosen as safe defaults, not profiled for Ling-mini-2.0's specific dimensions. Different block widths change L1 usage, data reuse, and compute efficiency.

**File:** `modules/moe.py`, lines 1118-1129

**Current config (Ling-mini-2.0 specific):**
```
Gate/Up: (32, 3072) x (3072, 1024) -> hidden_tiles=96, in0_block_w=4
Down:    (32, 1024) x (1024, 3072) -> hidden_tiles=32, in0_block_w=4
```

**Sweep grid:**
```python
# Gate/Up projections (in_features=3072, out_features=1024)
gate_up_configs = [
    {"in0_block_w": 2, "per_core_M": 1},
    {"in0_block_w": 4, "per_core_M": 1},  # current
    {"in0_block_w": 8, "per_core_M": 1},
    {"in0_block_w": 16, "per_core_M": 1},
    {"in0_block_w": 4, "per_core_M": 2},
]

# Down projection (in_features=1024, out_features=3072)
down_configs = [
    {"in0_block_w": 2, "per_core_M": 1},
    {"in0_block_w": 4, "per_core_M": 1},  # current
    {"in0_block_w": 8, "per_core_M": 1},
    {"in0_block_w": 16, "per_core_M": 1},
    {"in0_block_w": 4, "per_core_M": 2},
]
```

**How to implement:** Parameterize `_gate_up_program_config` and `_down_program_config` in `TTNNExperts.move_weights_to_device_impl` (L1118-1129). Run the test with each config and measure per-token decode latency from the CSV output. Select the config with lowest latency.

**Expected savings:** 0.1-0.5ms/token (10-30% on sparse_matmul time x 3 matmuls x 19 layers)
**Risk:** Low. Program config affects performance only; does not change numerical results.
**Test:** Compare output tensors with assert_allclose for each config.

---

### Priority 3: Evaluate LoFi Math Fidelity for Expert Matmuls (LOW RISK, MEDIUM IMPACT)

**Why:** Expert matmuls currently use `HiFi2` (L1130-1135). LoFi uses approximate multiply-accumulate which is faster but less precise. With 64 experts and top-6 routing, individual expert precision may matter less than routing precision (which uses HiFi4 and must not be changed).

**File:** `modules/moe.py`, lines 1130-1135

**Current:**
```python
self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

**Proposed change:**
```python
self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=True,  # keep fp32 accumulation for stability
    packer_l1_acc=True,
)
```

**Validation protocol:**
1. Run 128 decode tokens with HiFi2 (baseline) and LoFi
2. Compare final hidden states at each layer exit with cosine similarity
3. Acceptance: cosine_similarity >= 0.999 AND max_absolute_error <= 0.5% of output norm
4. Compare generated text quality (must produce coherent English)

**Expected savings:** 0.05-0.2ms/token (10-20% on sparse_matmul compute time)
**Risk:** Medium. Quality regression possible -- must validate per-model. Keep HiFi2 as fallback.
**Test:** Side-by-side text generation comparison.

---

### Priority 4: CCL Parameter Sweep (LOW RISK, LOW IMPACT)

**Why:** `reduce_scatter_minimal_async` at L1450-1462 uses default params that may not be optimal for Ling-mini-2.0's specific tensor size (1, 1, 1, 3072 -> 384 per device after scatter).

**File:** `modules/moe.py`, lines 1450-1462

**Current:**
```python
ttnn.experimental.reduce_scatter_minimal_async(
    routed_out,
    ...
    topology=ttnn.Topology.Ring,
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
```

**Sweep grid:**
```
chunks_per_sync:       {1, 5, 10, 20}
num_workers_per_link:  {1, 2}
num_buffers_per_channel: {1, 2}
```

That is 16 combinations. Run each with the standard decode test and measure reduce_scatter latency.

**Expected savings:** 0.05-0.1ms/token (5-15% on reduce_scatter latency, which is already small for batch=1)
**Risk:** Low. CCL params are performance-only knobs.
**Test:** Same correctness test with each config.

---

### Priority 5: Further Layout Conversion Reduction (LOW RISK, LOW IMPACT)

**Why:** `TTNNExperts.forward` still has 6 layout conversions per call:
1. L1193: x TILE -> ROW_MAJOR (for all_to_all_dispatch)
2. L1197: topk_experts_indices TILE -> ROW_MAJOR (for dispatch)
3. L1212: post_dispatch ROW_MAJOR -> TILE (for sparse_matmul)
4. L1279: expert_output TILE -> ROW_MAJOR (for all_to_all_combine)
5. L1296: combined_output ROW_MAJOR -> TILE (for weight application)
6. L1302-1303: w layout check + possible to_layout (for weight application)

Conversions 1, 2 are required (all_to_all_dispatch requires ROW_MAJOR input).
Conversion 3 is required (sparse_matmul requires TILE input).
Conversion 4 is required (all_to_all_combine requires ROW_MAJOR input).

**Potentially eliminable:**
- Conversion 5 (L1296): If weight application were done in ROW_MAJOR, this could be avoided. But `ttnn.mul` with broadcast is more efficient in TILE layout. Net benefit is unclear.
- Conversion 6 (L1302-1303): The weight tensor `w` comes from `topk_experts_weights` which is in TILE layout from the router. The permute at L1301 may change layout. Could pre-compute the weight tensor in the correct layout during routing to avoid this conversion.

**Expected savings:** 0.04-0.1ms/token (2-4us per avoided conversion x 1-2 conversions x 19 layers)
**Risk:** Low.

---

## 3. Total Remaining Savings Estimate

| Priority | Step | Expected Savings | Risk |
|----------|------|-----------------|------|
| 1 | Trace capture | 2-5ms | High |
| 2 | sparse_matmul config sweep | 0.1-0.5ms | Low |
| 3 | LoFi math fidelity | 0.05-0.2ms | Medium |
| 4 | CCL parameter sweep | 0.05-0.1ms | Low |
| 5 | Layout conversion reduction | 0.04-0.1ms | Low |
| **Total** | | **2.24-5.9ms/token** | |

**Recommended execution order:** Priority 2 and 4 (parameter sweeps) are independent and low-risk -- do them first to establish a better baseline. Priority 3 (LoFi) is also independent. Priority 1 (trace capture) is the highest payoff but requires more investigation into all_to_all trace compatibility.

---

## 4. What Was Already Done (Completed Steps)

### Step 1.1: Weight Application Simplification -- DONE

**moe.py L1298-1309:** The old 7-op pattern (`to_layout` -> `unsqueeze` -> `unsqueeze` -> `repeat(3072,1,1,1)` -> `permute` -> `to_layout` -> `mul`) was replaced with a 4-op broadcast pattern:
```python
w = ttnn.reshape(topk_experts_weights, ...)  # (1, 1, num_tokens, num_experts_per_tok)
w = ttnn.permute(w, (3, 1, 2, 0))           # (num_experts_per_tok, 1, num_tokens, 1)
w = ttnn.to_layout(w, ttnn.TILE_LAYOUT)      # ensure tile layout
weighted_output = ttnn.mul(combined_output, w) # broadcast multiply
final_output = ttnn.sum(weighted_output, dim=0, keepdim=True)
```

This eliminates the expensive `repeat(3072, 1, 1, 1)` which was copying 3072x more data than needed.

### Step 1.2: Pre-Gate Typecast Elimination -- DONE

**moe.py L1415-1426:** The `ttnn.typecast(x, ttnn.float32)` before the gate linear was removed. Instead, `ttnn.linear` is called with `dtype=ttnn.float32` output parameter on bf16 input:
```python
router_logits = ttnn.linear(
    x,                          # bf16 input (no typecast)
    self._gate_weight_tt,       # bf16 weight
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dtype=ttnn.float32,         # f32 output via accumulation
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    ),
)
```

### Step 2.1: Shared Expert Overlap -- DONE

**moe.py L1435-1438:** `shared_output = self.shared_experts(residual)` is now called BEFORE `routed_output = self.experts(x, ...)`, allowing TTNN's async dispatch to overlap shared expert matmuls with the routed expert all_to_all + sparse_matmul pipeline.

### Step 3.1/3.2: Router -- NO CHANGE NEEDED

For Ling-mini-2.0 (`n_group=1, topk_group=1`), the router takes the single-pass topk branch (L927-932) which is already optimal. The CPU sort fallback is only in `TTNNGlm4MoeRouteTokenToExperts` (L806-813), not in `TTNNMoERouterDecode` which is the active router for `TTNNBailingMoE`.

---

## 5. Key Source Locations (Updated)

| Component | File | Lines |
|-----------|------|-------|
| TTNNBailingMoE class | `modules/moe.py` | L1470-1624 |
| TTNNMoE.forward | `modules/moe.py` | L1385-1467 |
| TTNNExperts.forward | `modules/moe.py` | L1137-1317 |
| TTNNMoERouterDecode.forward | `modules/moe.py` | L891-1002 |
| Weight application (DONE) | `modules/moe.py` | L1298-1309 |
| Gate linear (DONE, no typecast) | `modules/moe.py` | L1415-1426 |
| Shared experts call (DONE, early) | `modules/moe.py` | L1435-1438 |
| sparse_matmul config (Priority 2) | `modules/moe.py` | L1118-1135 |
| Expert compute config (Priority 3) | `modules/moe.py` | L1130-1135 |
| reduce_scatter params (Priority 4) | `modules/moe.py` | L1450-1462 |
| Test file | `tests/test_ling_mini_2_0.py` | Full file |

---

## 6. How to Collect Measurements

```bash
cd /home/ttuser/salnahari
source tt_bashrc
cd tt-metal
MESH_DEVICE=T3K pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py --timeout=0 -xvs
```

The test saves timing to `ling_mini_2_0_paged_attention_timing_stats.csv`. For per-op profiling, use Tracy:

```bash
cd /home/ttuser/salnahari
source tt_bashrc
cd tt-metal
TT_METAL_DEVICE_PROFILER=1 MESH_DEVICE=T3K pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py --timeout=0 -xvs
```

Then analyze with `tt_metal/tools/profiler/process_ops_logs.py` to get per-op CSV.

---

## 7. Dependencies

- **PLAN_wrapper_optimizations.md** (P1-P5): Wrapper overhead savings (~440ms) dwarf MoE-specific savings. Apply in parallel.
- **TTNNBailingMoEDecoderLayer** (from wrapper optimizations P1): Enables full decoder layer trace capture (Step 4.2 from original plan), which is strictly better than MoE-only trace capture.
- **Profiling baseline:** Priorities 2, 3, 4 require per-op timing data. Collect baseline with Tracy or DispatchManager CSV before sweeping.
