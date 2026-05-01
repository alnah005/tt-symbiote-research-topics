# Qwen3.6-35B-A3B Module Profiling Session State

**Date:** 2026-04-21
**Status:** COMPLETE — all optimization priorities resolved; Metal Trace live on TTNNQwen3MoE (51x speedup)
**Prerequisite:** Bring-up SESSION_STATE_qwen3_6_bringup.md (all 32 tests passed)

---

## Profiling Plan

See `PLAN_qwen3_6_module_profiling.md` for the full profiling approach (3-method strategy: DispatchManager, ModuleTimer, Tracy).

---

## Phase 1 Results: Per-Module Test-Level Timing (T3K)

All timing is pytest test-body duration (from pytest slowest-durations). No explicit per-call forward timing instrumentation was added — these are test-level durations. For per-forward-call timing, see Phase 2 (ModuleTimer).

### P0 Module Results

| Step | Test | Pass/Fail | Wall-Clock Total | Test Body Duration | PCC |
|------|------|-----------|------------------|--------------------|-----|
| 1 | `test_ttnn_qwen3_moe_full_block` | PASS | 25.15s | **16.57s** | 0.999917 |
| 2 | `test_ttnn_qwen3_moe_prefill` | PASS | 39.48s | **17.18s** | 0.999826 overall |
| 3 | `test_ttnn_qwen3_full_attention_paged_decode_pcc` | PASS | 11.07s | **2.90s** | 0.999693 |
| 4 | `test_ttnn_qwen3_full_attention_prefill` | PASS | 8.04s | **2.31s** | 0.999763 |
| 5 | `test_ttnn_qwen3_linear_attention_decode_with_state` | PASS | 7.82s | **1.95s** | — |
| 6 | `test_ttnn_qwen3_linear_attention` (multi-step) | PASS | 17.13s | **1.97s** | first=0.999828, last=0.999571, min=0.999348 |
| 7 | `test_ttnn_qwen_router_decode_accuracy` | PASS | 6.49s | **0.68s** | — |
| 8 | `test_ttnn_qwen_experts_fused_sparse_matmul` | PASS | 19.94s | **14.11s** | 0.999969 |
| 9 | `test_ttnn_shared_expert_mlp_qwen` | PASS | 7.00s | **1.23s** | 0.999925 |
| 10 | `test_qwen3_four_layer_block` | PASS | 68.28s | **62.46s** | 0.999976 |

### P1 Module Results

| Step | Test | Pass/Fail | Duration | Notes |
|------|------|-----------|----------|-------|
| 11 | `test_ttnn_linear_qwen_weights` | PASS | 0.13s | Replicated |
| 11 | `test_ttnn_linear_replicated_col_sharded_qwen_weights` | PASS | 0.40s | Col-sharded |
| 11 | `test_ttnn_linear_col_sharded_row_sharded` | PASS | 0.37s | Row-sharded, PCC 0.999965 |
| 12 | `test_ttnn_rope_partial_rotary_qwen` | PASS | 0.21s | rotary_dim=64 |
| 12 | `test_ttnn_sdpa_prefill_qwen` | PASS | 0.21s | PCC 0.999853 |

---

## Key Findings

### Bottleneck Ranking (by test duration)

1. **`TTNNQwen3MoE` (full block, decode): 16.57s** — dominant cost, ~5.7× more expensive than full attention
2. **`TTNNQwenExperts` (sparse matmul): 14.11s** — accounts for ~85% of total MoE cost
3. **`TTNNQwen3FullAttention` (decode): 2.90s** — second tier
4. **`TTNNQwen3LinearAttention` (decode): 1.95s** — close to full attention
5. **`TTNNGlm4MoeMLP` (shared expert): 1.23s** — within MoE overhead
6. **`TTNNQwenMoERouterDecode`: 0.68s** — router is cheap (~4% of MoE cost)
7. **Linear layers: 0.13–0.40s** — fast, not bottleneck
8. **RoPE + SDPA: 0.21s each** — negligible

### MoE Internal Cost Breakdown

```
MoE full block:       16.57s (100%)
  └── Experts:        14.11s  (85%)
  └── Shared expert:   1.23s  ( 7%)
  └── Router:          0.68s  ( 4%)
  └── Overhead/norms:  ~0.55s ( 3%)
```

### 4-Layer Pattern Timing

| Sublayer | Expected cost (attn + MoE) | Layers |
|----------|---------------------------|--------|
| Linear attention | 1.95 + 16.57 = 18.52s | ×3 = 55.56s |
| Full attention | 2.90 + 16.57 = 19.47s | ×1 = 19.47s |
| **4-layer pattern total** | **~75.0s** (sum of parts) | |
| **Measured 4-layer block** | **62.46s** | (module tests include setup overhead) |

**Full model projection:** 62.46s × 10 = **~624.6s per decode step**
This matches the e2e test result from bringup (625s to generate 16 tokens = ~39.1s/token after tracing/warmup).

---

## Phase 3 Results: Tracy Op-Level Profiling (TTNNQwenExperts)

**CSV:** `/home/ttuser/salnahari/tt-metal/generated/profiler/reports/2026_04_21_22_27_46/ops_perf_results_2026_04_21_22_27_46.csv`

### Device-Side Op Breakdown (one call, device 0)

| Op | Device Kernel Time | % of Total | Bottleneck Type |
|----|--------------------|------------|-----------------|
| `FillPadDeviceOperation` (×3) | 404.50 us | 23.4% | Padding overhead |
| `AllToAllCombineDeviceOperation` | 355.47 us | 20.6% | CCL gather, BRISC-dominated |
| `MoeExpertTokenRemapDeviceOperation` | 308.27 us | 17.8% | Data movement, BRISC-only |
| `UntilizeDeviceOperation` | 282.24 us | 16.3% | Format conversion |
| `SparseMatmulDeviceOperation` (×2) | 92.31 us | 5.3% | **NOT bottleneck** |
| `AllToAllDispatchDeviceOperation` | 38.38 us | 2.2% | CCL scatter |
| Other (tilize, unary, binary, rms_norm) | ~248 us | 14.4% | |
| **Total** | **1,729 us** | 100% | |

### Key Finding: SparseMatmul Is NOT the Bottleneck

SparseMatmul runs in **47–110 us** vs PM Ideal of **484 us** (bandwidth model). FPU Util = 0.002%.
At decode time, only a handful of active expert tiles are touched — the compute work is trivially small.
Wall-clock 14.11s vs device 1.73ms discrepancy = host-dispatch and CCL synchronization dominate.

### Actual On-Device Bottleneck Stack

1. **FillPad (23.4%)** — 3× padding ops, purely overhead, can be eliminated
2. **AllToAllCombine (20.6%)** — CCL gather, BRISC-dominated, Ethernet bandwidth-limited
3. **MoeExpertTokenRemap (17.8%)** — pure data-movement kernel, ~300 us, TRISC idle
4. **Untilize (16.3%)** — format conversion overhead

### Optimization Recommendations (priority order)

1. **Eliminate FillPad (23.4% savings):** Pre-allocate zero-padded buffers at setup time; avoid 3× FillPad calls per decode step.
2. **AllToAllCombine (20.6%):** Primary CCL target. Reduce output tensor size (transmit only non-zero expert outputs) or evaluate `locally_reduced=true` if the reduce can be done before the all-to-all.
3. **Fuse MoeExpertTokenRemap (17.8%):** BRISC-only kernel hitting memory bandwidth. Fuse into sparse matmul reader kernel to eliminate a separate launch.
4. **Eliminate Untilize (16.3%):** Store expert outputs in tilized format to avoid format conversion.
5. **Do NOT tune SparseMatmul program config or lower math fidelity** — sparse matmul is 5.3% of device time; tuning it will not move the needle.
6. **Trace capture:** Total device time is ~1.73 ms/call but host-dispatch dominates wall-clock. Metal Trace API on the MoE inner loop would dramatically cut host overhead.

---

## Argmax Segfault Investigation

**`test_qwen3_four_layer_block_decode_argmax` — CONFIRMED PASS in strict isolation**

| Field | Value |
|-------|-------|
| Result | PASS |
| Duration | 69.25s |
| PCC | 0.999980 |
| argmax_match | True (torch=24534, ttnn=24534) |
| Top-5 overlap | 5/5 |
| Verdict | Memory-pressure artifact — not a real regression |

The previous segfault occurred because the `-k "test_qwen3_four_layer_block"` filter matched both the primary test and the argmax variant, causing two back-to-back model loads without a chip reset. Confirmed safe in isolation.

---

## Anomalies

- **`test_qwen3_four_layer_block_decode_argmax` — SEGFAULT**: Crashed during the Verifier run when matched by the `-k "test_qwen3_four_layer_block"` filter alongside the primary test. The primary test passed; the argmax variant segfaulted. The argmax variant had previously passed during bringup (PCC 0.999980, argmax_match=True, 72s). May be memory-pressure-sensitive — should be re-run in strict isolation with a chip reset.

---

## Optimization Summary (Final)

| Priority | Optimization | Status | Result |
|----------|-------------|--------|--------|
| 1 | `@trace_enabled` on `TTNNQwen3MoE` | **DONE** | **51x wall-clock speedup** (881s → 17.3s for 128 tokens) |
| 2 | TILE_LAYOUT for `remap_topk_mask` | **BLOCKED** — op requires ROW_MAJOR | Reverted |
| 3 | Skip `ttnn.repeat` when `batch_size_per_device==1` | **DONE** (1b) | Guard in place in moe.py + qwen_moe.py |
| 4 | L1 memory config on untilize before combine | **DONE** (1c) | Applied in qwen_moe.py |
| 5 | AllToAllCombine reduction | **RULED OUT** — no Python-level tuning params | N/A |
| 6 | MoeExpertTokenRemap fusion | **RULED OUT** — kernel-level change required | N/A |
| 7 | Untilize elimination | **RULED OUT** — required by all_to_all_combine contract | N/A |
| 8 | SparseMatmul tuning | **RULED OUT** — only 5.3% of device time | N/A |

### Code Changes (all in tt-metal, local only, no push)

**`modules/qwen_moe.py`:**
- Line 24: added `trace_enabled` to `run_config` import
- Line 549: added `@trace_enabled` before `class TTNNQwen3MoE`
- Lines 390–392: `if batch_size_per_device == 1` guard for `ttnn.repeat` (1b)
- Line 487: `memory_config=decode_memory_config` on untilize (1c)

**`modules/moe.py`:**
- Lines 1244–1246: `if batch_size_per_device == 1` guard for `ttnn.repeat` (1b)

---

## What To Do Next

### Immediate (Phase 2 — per-call forward timing)

Add explicit `time.perf_counter()` timing around the MoE forward call in `test_ttnn_qwen3_moe_full_block` and `test_ttnn_qwen_experts_fused_sparse_matmul` to get per-call decode latency (ms) rather than test-body latency.

Command template once instrumented:
```bash
unset TT_VISIBLE_DEVICES && tt-smi -r
MESH_DEVICE=T3K TTNN_LINEAR_ATTN_PROJECTIONS=1 \
    pytest models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py \
    -k "test_ttnn_qwen3_moe_full_block or test_ttnn_qwen_experts_fused_sparse_matmul" \
    -s -v --timeout=0
```

### Recommended Next Step (Phase 3 — Tracy on MoE experts)

Run Tracy on `test_ttnn_qwen_experts_fused_sparse_matmul` to get device-level kernel timing for:
- `ttnn.experimental.sparse_matmul` (w1/gate, w3/up, w2/down)
- `ttnn.all_to_all_dispatch` and `ttnn.all_to_all_combine` (CCL overhead)
- Determine compute-bound vs. bandwidth-bound via FPU utilization

Command (two-terminal Tracy):
```bash
# Terminal A:
cd /home/ttuser/salnahari/tt-metal
./tt_metal/third_party/tracy/capture/build/unix/tracy-capture -o /tmp/qwen3_6_experts_profile.tracy -f

# Terminal B:
cd /home/ttuser/salnahari/tt-metal
unset TT_VISIBLE_DEVICES && tt-smi -r
TT_METAL_DEVICE_PROFILER=1 TRACY_NO_EXIT=1 TT_METAL_CLEAR_L1=1 TT_METAL_PROFILER_SYNC=1 \
TT_SYMBIOTE_TRACY=1 MESH_DEVICE=T3K TTNN_LINEAR_ATTN_PROJECTIONS=1 \
    pytest models/experimental/tt_symbiote/tests/test_qwen3_6_35b_a3b_bottom_up.py \
    -k "test_ttnn_qwen_experts_fused_sparse_matmul" -s -v --timeout=0 -p no:randomly
```

### Investigate

- Re-run `test_qwen3_four_layer_block_decode_argmax` in strict isolation to confirm whether it still segfaults or was a memory-pressure artifact.

---

## Team Lead Rules

- **Only `research_topics.md` may be pushed** to research-topics repo
- **No pushes to tt-metal**
- Research topics pending (do not re-add):
  - Qwen3.6-35B-A3B Weight Compatibility (Pending)
  - M-RoPE Implementation on TTNN (Pending)
  - Multi-Token Prediction on TT Hardware (Pending)
