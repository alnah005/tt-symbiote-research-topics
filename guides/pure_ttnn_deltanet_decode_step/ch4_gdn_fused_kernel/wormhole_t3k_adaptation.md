# Wormhole T3K Adaptation for `gdn_full_fused_inplace`

`[REUSABLE — port and tune]`

This file specifies the required changes to adapt `gdn_full_fused_inplace` from Blackhole to Wormhole T3K. Given the `REUSABLE_WITH_TUNING` classification from `gdn_full_fused_inplace_analysis.md`, no rewrite of compute logic is required. The work is a set of targeted constant and configuration changes, followed by a verification test to confirm numerical correctness.

> **Key Finding:** Per-head state on T3K is `[128, 128]` BF16 = 32 KB. With 4 heads per device (post head-parallel sharding), total CB usage per core is approximately 40 KB — well within Wormhole's 1.5 MB L1. The kernel's DMA streaming pattern and 6-op compute structure are portable. The changes required are: (1) CB size constant audit, (2) FPU tile dimension check, (3) FP32_DEST_ACC path review, (4) core grid reconfiguration from Blackhole grid to a (1,4) Wormhole grid.

---

## 1. Required Constant Changes

Work through the following checklist in order. For each item, locate the relevant source line, record the current Blackhole value, and apply the Wormhole value. The file paths below are expected locations — verify against the actual source during the port (see `gdn_full_fused_inplace_analysis.md` Section 2 for the source location search procedure).

### 1.1 FPU Tile Dimensions

| Parameter | Blackhole B0 value | Wormhole value | Where to check |
|---|---|---|---|
| `TILE_WIDTH` | 64 (if hardcoded) | 32 | Compute kernel source; CB size calculations |
| `TILE_HEIGHT` | 64 (if hardcoded) | 32 | Compute kernel source; loop bounds over state tiles |
| Tiles per state row | 2 (if 64-wide) | 4 (if 32-wide) | Loop bounds for the 4×4 tile grid covering `[128, 128]` state |

If `TILE_WIDTH` and `TILE_HEIGHT` are already pulled from TT-Metalium compile-time constants (e.g., `tt::constants::TILE_WIDTH`) rather than hardcoded literals, this change is a no-op — the constants resolve to 32 on Wormhole automatically.

### 1.2 CB Size Constants

All `CreateCircularBuffer` calls must be audited. Extract the `total_size` for each CB, sum them, and confirm the total is below 1.5 MB (1,572,864 bytes).

**Expected CB layout for Wormhole T3K (one core, one head):**

| CB | Role | Elements | BF16 bytes | Notes |
|---|---|---|---|---|
| CB0 | State S `[d_k, d_v]` | 128 × 128 = 16,384 | 32,768 (32 KB) | Loaded from DRAM once; held throughout all 6 ops; written back at end |
| CB1 | k̃ input `[d_k, 1]` | 128 elements padded to 32×32 tile | 2,048 (2 KB) | Read from L1 (passed from projection ops) |
| CB2 | v input `[d_v, 1]` | 128 elements padded to 32×32 tile | 2,048 (2 KB) | Read from L1 |
| CB3 | g, β scalars broadcast | One 32×32 tile (scalar broadcast over all elements) | 2,048 (2 KB) | g_t and β_t broadcast to tile size |
| CBOUT | Output o_t `[d_v, 1]` | 128 elements padded to 32×32 tile | 2,048 (2 KB) | Written to output buffer after op 6 |

**Total CB usage:** 32,768 + 2,048 + 2,048 + 2,048 + 2,048 = **40,960 bytes (40 KB)**

This is 2.7% of Wormhole's 1.5 MB L1. There is no memory pressure risk. If the Blackhole implementation added double-buffering for input CBs (allocating 2× the CB size to pipeline DMA and compute), that optimization can be retained on Wormhole without issue.

> **Warning:** If the Blackhole implementation allocated CB0 assuming 64×64 tiles (where one state tile covers 4,096 elements × 2 bytes = 8 KB, so the 128×128 state requires 4 tiles × 8 KB = 32 KB), the total CB0 size is the same 32 KB. But if it allocated scratch CBs or double-buffered CB0 at `2 × 32 KB = 64 KB`, that is still well within 1.5 MB. Record the actual value during the port; the risk is if unusual extra allocations were made for Blackhole-specific optimizations (e.g., prefetching a second head's state while computing the first).

### 1.3 FP32_DEST_ACC Accumulation Path

Search the compute kernel for any of the following:

```
FP32_DEST_ACC
DEST_ACCUM_EN
fp32_dest_acc_en
```

If found:

1. Confirm whether Wormhole supports the same flag (it does, but the path may differ from Blackhole B0's native FP32 accumulation).
2. Run the kernel with the flag both enabled and disabled; compare PCC against the PyTorch BF16 reference.
3. If PCC > 0.999 without `FP32_DEST_ACC`, drop the flag — BF16 accumulation on Wormhole is sufficient for the DeltaNet state update (see Chapter 6, `pcc_accuracy_thresholds.md` for the error decay argument).
4. If PCC falls below 0.999 without the flag, retain it and verify that Wormhole's FP32 accumulation path is correctly activated (consult `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/` for the relevant API).

### 1.4 NOC Routing and Core Coordinates

Confirm that core coordinates in the reader and writer RISCV programs are computed from parameters passed at dispatch time (e.g., `get_arg_val<uint32_t>(0)` patterns), not hardcoded. If hardcoded core coordinates exist (e.g., `noc_xy_encoding(3, 5)` or similar), they must be replaced with runtime-computed coordinates based on the actual core assignment on the Wormhole NOC grid.

---

## 2. Core Grid for Wormhole T3K

### 2.1 Available Core Grid

Wormhole has a (8, 8) = 64 Tensix core grid available per chip. Under head-parallel sharding on T3K (8 devices), each device processes `num_v_heads / 8 = 4` attention heads per DeltaNet layer.

### 2.2 Core Assignment

Assign one Tensix core per head:

```
Core grid: (1, 4)  — a single column of 4 cores
  Core (0, 0) → head 0 on this device
  Core (0, 1) → head 1 on this device
  Core (0, 2) → head 2 on this device
  Core (0, 3) → head 3 on this device
```

Alternatively, `(4, 1)` — a single row of 4 cores — is equally valid. Either layout leaves 60 cores idle per DeltaNet layer dispatch (64 available − 4 used), which is acceptable: the decode step is bandwidth-bound on the DRAM state read/write, not compute-bound. Adding more cores per head would not reduce latency; it would only increase dispatch complexity.

### 2.3 State Sharding Under This Grid

Each core's state tensor is stored in a DRAM buffer corresponding to its assigned head. Under head-parallel sharding:

- State per head: `[B, 1, d_k, d_v] = [1, 1, 128, 128]` BF16 = 32 KB
- 4 heads × 32 KB = 128 KB of state DRAM per device per DeltaNet layer
- 30 DeltaNet layers × 128 KB = 3.84 MB per device for all state — negligible in the 12 GB DRAM budget

Each core independently reads its own 32 KB state, runs the 6 fused ops, and writes its updated state back. No cross-core or cross-NOC communication is required within a single device.

---

## 3. Multi-Device Sharding

Under head-parallel sharding across 8 Wormhole chips in the T3K mesh:

- Total heads (Qwen3.6-35B-A3B): `num_v_heads = 32` heads per DeltaNet layer (global)
- Per device: 4 heads
- Each device runs the fused kernel independently on its 4 heads
- No inter-chip communication is required for the recurrent state update

The all-gather for the attention output (`o_t`) happens downstream of the fused kernel, in the existing output projection path — the same all-gather that already follows the recurrent step in the current (non-fused) implementation. The fused kernel does not change the all-gather boundary.

> **Key Finding:** The fused kernel's per-device independence is a simplification over the all-reduce patterns used in the full-attention layers. Each T3K device runs its copy of the fused kernel on its 4 heads; there is no dependency between devices during the recurrent update step.

---

## 4. Verification Test

Run the following test after completing the constant changes and before declaring the port complete.

### 4.1 Correctness Test

1. Generate random BF16 inputs: `S_prev [1, 4, 128, 128]`, `k_tilde [1, 4, 128, 1]`, `v [1, 4, 128, 1]`, `q_tilde [1, 4, 128, 1]`, `g [1, 4, 1, 1]`, `beta [1, 4, 1, 1]` (shapes reflect 4 heads per device).
2. Run the PyTorch reference implementation of the 6 ops (from Chapter 2, `recurrence_math_and_tensor_ops.md`) in FP32.
3. Convert reference outputs to BF16.
4. Run the Wormhole fused kernel with the same inputs.
5. Compute PCC between the kernel's `S_new` output and the reference `S_new`; compute PCC between the kernel's `o_t` and the reference `o_t`.
6. Assert both PCCs exceed 0.999.

### 4.2 State Drift Test

Run 200 sequential decode steps using the fused kernel and the PyTorch reference in parallel with the same random input sequence:

1. At each step t, feed the same k̃, v, q̃, g, β to both implementations.
2. After each step, compute PCC(S_kernel_t, S_ref_t) and record it.
3. Compute the L2 norm of the difference: `||S_kernel_t - S_ref_t||_2`.
4. Assert that per-step PCC remains > 0.999 across all 200 steps.
5. Assert that the L2 norm of the state difference is bounded (does not grow unboundedly with step count). Given the decay gate g_t < 1, errors from earlier steps are exponentially suppressed — the drift should remain stable or decrease over long runs.

### 4.3 Trace Compatibility Test

1. Wrap the fused kernel call inside `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`.
2. Execute the trace for 10 decode steps.
3. Compare outputs of the traced run against the non-traced run; assert PCC > 0.999 at each step.
4. Confirm no `ttnn.from_torch` or `ttnn.to_torch` is invoked during the trace capture bracket.

---

## 5. Summary

| Item | Wormhole T3K value |
|---|---|
| L1 per Tensix core | 1.5 MB |
| CB total per core (all 5 CBs) | ~40 KB |
| Heads per device (post T3K sharding) | 4 |
| Cores used per layer dispatch | 4 (one per head) |
| Core grid | (1, 4) or (4, 1) |
| State per head | 32 KB BF16 `[128, 128]` |
| DRAM state per device per layer | 128 KB |
| FPU tile size | 32×32 (Wormhole standard) |
| FP32_DEST_ACC | Verify; drop if PCC > 0.999 without it |
| Cross-device communication | None (each device processes its heads independently) |
| Availability tag | `[REUSABLE — port and tune]` |
| PCC acceptance threshold | > 0.999 per step, state and output |
| State drift test | 200 steps; L2 norm bounded |
