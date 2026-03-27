# QK Normalization Latency

## Overview

When `use_qk_norm=True`, each decode step applies `TTNNRMSNorm` independently to the Q and K tensors immediately after rotary positional embedding. This operation is not free: the norm kernel imposes input constraints that require the tensors to be reshaped and moved into a specific memory configuration before the norm can execute, and then moved back afterward. At decode batch=1, the compute work inside the norm is trivial — one token's worth of values per head — but the surrounding data-movement infrastructure adds approximately **64 µs** of DRAM↔L1 transition overhead per step [ESTIMATE], not counting the reshape dispatch cost.

This file establishes the complete code path, derives the per-component latency breakdown, compares the aggregate cost against the fused QKV matmul for perspective, determines whether the L1 move is required by the norm kernel's input constraints or is merely a precaution, and evaluates options for reducing or eliminating the overhead.

All cost estimates are marked `[ESTIMATE]` and derive from the cost model in Chapter 4, `transition_cost_model.md`.

## Code Path When `use_qk_norm=True`

After RoPE completes, Q and K are in L1 HEIGHT_SHARDED layout. The QK norm path begins immediately after the post-RoPE L1→DRAM eviction (transitions T2a and T2b). The following sequence executes for **each of Q and K independently** on every decode step:

### Step 1 — Post-RoPE Eviction to DRAM (T2a / T2b)

```python
# T2a: Q leaves RoPE kernel, L1/HS → DRAM/ITVL
q_post_rope = ttnn.to_memory_config(q_rope_out, ttnn.DRAM_MEMORY_CONFIG)
# shape: (1, 16, 32, 128) tile-padded, BF16, DRAM INTERLEAVED

# T2b: K leaves RoPE kernel, L1/HS → DRAM/ITVL
k_post_rope = ttnn.to_memory_config(k_rope_out, ttnn.DRAM_MEMORY_CONFIG)
# shape: (1, 4, 32, 128) tile-padded, BF16, DRAM INTERLEAVED
```

T2a and T2b are defined and costed in Chapter 4. They are listed here because they are the proximate cause of why Q and K must make another DRAM→L1 trip for the norm: by the time the norm is reached, both tensors are in DRAM.

### Step 2 — DRAM→L1 Move for Norm Input (T_norm_in)

`TTNNRMSNorm` requires its input in L1. The tensors must be reloaded from DRAM:

```python
# T_norm_in_Q: Q DRAM→L1 for norm input
# Uses a 2D-compatible L1 INTERLEAVED config, not the HEIGHT_SHARDED RoPE config.
norm_l1_config_q = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.L1,
)
q_norm_in_raw = ttnn.to_memory_config(q_post_rope, norm_l1_config_q)
# shape: (1, 16, 32, 128) tile-padded, BF16, L1 INTERLEAVED

# T_norm_in_K: K DRAM→L1 for norm input
k_norm_in_raw = ttnn.to_memory_config(k_post_rope, norm_l1_config_q)
# shape: (1, 4, 32, 128) tile-padded, BF16, L1 INTERLEAVED
```

Cost: Q-side ≈ 21 µs, K-side ≈ 11 µs, total T_norm_in ≈ **32 µs** [ESTIMATE].

### Step 3 — 3D→2D Reshape Before Norm

`TTNNRMSNorm` expects a 2D input tensor. The current 3D shape `(N_heads, T, H)` — or equivalently the 4D tile-padded `(1, N_heads, 32, 128)` — must be collapsed to `(N_heads, H)` in logical terms (tile-padded: `(N_heads, 128)`) before the norm call:

```python
# For Q: collapse (1, 16, 32, 128) → (16, 128) in logical dimensions
# Tile-padded form: effectively (16*32, 128) = (512, 128) due to tile layout
q_norm_2d = ttnn.reshape(q_norm_in_raw, (16, 128))   # logical: (N_q, H)
# Tile-padded effective shape seen by kernel: (512, 128) = (N_q * T, H)

# For K: collapse (1, 4, 32, 128) → (4, 128) in logical dimensions
k_norm_2d = ttnn.reshape(k_norm_in_raw, (4, 128))    # logical: (N_kv, H)
# Tile-padded effective shape: (128, 128) = (N_kv * T, H)
```

This reshape collapses the batch and seq_len leading dimensions together with the heads dimension into a single row dimension. The L1 INTERLEAVED tensor is contiguous, so this reshape is a **metadata-only operation** (zero data movement) when tile boundaries align. For Q with tile-padded shape `(1, 16, 32, 128)`, the reshape to `(16, 128)` is tile-aligned: the 16 rows of 128 elements each map cleanly onto 16 tile rows. No re-tiling is required and no data copy occurs.

### Step 4 — TTNNRMSNorm Execution

```python
# RMSNorm on Q: output shape matches input shape (2D)
q_norm_out_2d = TTNNRMSNorm(
    q_norm_2d,
    weight=self.q_norm_weight,   # learned scale, shape (H,) = (128,)
    epsilon=self.norm_eps,       # typically 1e-6
)
# q_norm_out_2d: (16, 128) logical, L1 INTERLEAVED, BF16

# RMSNorm on K
k_norm_out_2d = TTNNRMSNorm(
    k_norm_2d,
    weight=self.k_norm_weight,
    epsilon=self.norm_eps,
)
# k_norm_out_2d: (4, 128) logical, L1 INTERLEAVED, BF16
```

RMS normalization computes, for each row vector `x` of dimension `H`:

```
RMSNorm(x) = x / sqrt( (1/H) * sum(x_i^2) + eps ) * weight
```

At decode batch=1 with a single token, the 2D Q tensor is `(16, 128)` — 16 rows of 128 elements. The norm executes independently per row. Each row requires: one sum-of-squares reduction over 128 elements, one reciprocal sqrt, and 128 element-wise multiplications. This is 16 × (128 + 1 + 128) = 4,112 floating-point operations for Q, and 4 × 257 = 1,028 for K.

At Wormhole's peak FP throughput, these operations complete in well under 1 µs [ESTIMATE]. The norm kernel cost is dominated by its dispatch overhead, not its arithmetic.

### Step 5 — 2D→3D Reshape After Norm

The 2D output must be restored to the 4D shape expected by downstream consumers:

```python
# Restore Q: (16, 128) → (1, 16, 1, 128) logical (tile-padded: (1, 16, 32, 128))
q_norm_3d = ttnn.reshape(q_norm_out_2d, (1, 16, 1, 128))

# Restore K: (4, 128) → (1, 4, 1, 128) logical (tile-padded: (1, 4, 32, 128))
k_norm_3d = ttnn.reshape(k_norm_out_2d, (1, 4, 1, 128))
```

Same logic as Step 3: this is a metadata-only operation on a contiguous L1 INTERLEAVED tensor. No data copy.

### Step 6 — L1→DRAM Eviction After Norm (T_norm_out)

The normalized Q and K are written back to DRAM for downstream consumers (SDPA and the paged KV-cache update):

```python
# T_norm_out_Q: Q post-norm, L1→DRAM
q_normed = ttnn.to_memory_config(q_norm_3d, ttnn.DRAM_MEMORY_CONFIG)
# shape: (1, 16, 32, 128) tile-padded, BF16, DRAM INTERLEAVED

# T_norm_out_K: K post-norm, L1→DRAM
k_normed = ttnn.to_memory_config(k_norm_3d, ttnn.DRAM_MEMORY_CONFIG)
# shape: (1, 4, 32, 128) tile-padded, BF16, DRAM INTERLEAVED
```

Cost: Q-side ≈ 21 µs, K-side ≈ 11 µs, total T_norm_out ≈ **32 µs** [ESTIMATE].

### Summary Diagram

```
DRAM/ITVL                L1/ITVL                     L1/ITVL                DRAM/ITVL
──────────               ──────────                  ──────────             ──────────
q_post_rope  →[T_norm_in_Q]→  q_norm_in_raw  →[reshape]→  q_norm_2d
(1,16,32,128)              (1,16,32,128)              (16,128)
                                                         ↓ TTNNRMSNorm
                                                      q_norm_out_2d
                                                         ↓ reshape
                                                      q_norm_3d        →[T_norm_out_Q]→  q_normed
                                                      (1,16,32,128)                      (1,16,32,128)

k_post_rope  →[T_norm_in_K]→  k_norm_in_raw  →[reshape]→  k_norm_2d
(1,4,32,128)               (1,4,32,128)               (4,128)
                                                         ↓ TTNNRMSNorm
                                                      k_norm_out_2d
                                                         ↓ reshape
                                                      k_norm_3d        →[T_norm_out_K]→  k_normed
                                                      (1,4,32,128)                       (1,4,32,128)
```

## Latency Breakdown

### Component Costs

Table: Per-component latency for the QK norm path per decode step [all ESTIMATE]

| Component | Q cost | K cost | Notes |
|---|---|---|---|
| T_norm_in (DRAM→L1 move) | ≈ 21 µs | ≈ 11 µs | Dominated by 8 µs fixed kernel dispatch overhead; transfer adds 13.1/3.3 µs for 128/32 KB |
| Reshape 3D→2D (Step 3) | < 1 µs | < 1 µs | Metadata-only; no data movement on contiguous L1 tensor |
| `TTNNRMSNorm` kernel (Step 4) | ≈ 3–8 µs | ≈ 3–8 µs | Arithmetic trivial; cost is kernel dispatch + launch overhead [ESTIMATE] |
| Reshape 2D→3D (Step 5) | < 1 µs | < 1 µs | Metadata-only |
| T_norm_out (L1→DRAM eviction) | ≈ 21 µs | ≈ 11 µs | Same structure as T_norm_in, reverse direction |
| **Subtotal** | **≈ 45–50 µs** | **≈ 25–30 µs** | Per tensor |
| **Total QK norm overhead** | | **≈ 70–80 µs** | Both Q and K combined [ESTIMATE] |

> **Note:** This table excludes reshape dispatch overhead (the four `ttnn.reshape` calls in Steps 3 and 5). Each reshape crosses the pybind11 layer and incurs ≈ 1–3 µs of Python dispatch cost; the aggregate is ≈ 4–12 µs. The complete estimate including reshape dispatch is in the second table below (≈ 74–92 µs).

The cost model from Chapter 4 reported T_norm_in = 32 µs and T_norm_out = 32 µs as aggregate figures. Including the norm kernel dispatch overhead (3–8 µs each for Q and K, executed sequentially), the true total lands in the **70–80 µs** range per decode step [ESTIMATE]. The 64 µs figure cited in Chapter 4 covers only the DRAM↔L1 transitions; the full accounting including norm dispatch is higher.

### Comparison to Fused QKV Matmul

The fused QKV matmul costs approximately **10–40 µs** per decode step [ESTIMATE] (see Chapter 2, `fusion_mechanics.md`). The QK norm overhead at 70–80 µs [ESTIMATE] therefore exceeds the prior compute stage by 2–8×. This ratio is a diagnostic indicator: at decode batch=1, the data-movement and kernel-dispatch overhead of QK norm is not a second-order correction — it is the dominant latency component of the post-QKV processing pipeline.

For comparison, the SDPA kernel itself (computing scaled dot-product attention over all cached tokens) has a compute cost that scales with KV cache length `S`:

```
T_SDPA ≈ f(S, N_q, N_kv, H)
```

For short KV contexts (S < 512 tokens), `T_SDPA` may be comparable to or less than the QK norm overhead, making the norm cost a primary target. For long contexts (S > 4096 tokens), SDPA compute dominates and the norm cost becomes relatively less important.

### Reshape Kernel Cost vs. Actual Measurement

The reshape steps (Steps 3 and 5) are stated above as metadata-only because the tensors are contiguous in L1 and the tile boundaries are compatible. This claim depends on two conditions being true:

1. The L1 INTERLEAVED tensor produced by T_norm_in is physically contiguous in L1 memory.
2. The target shape `(N_heads, 128)` is tile-aligned with the source shape `(1, N_heads, 32, 128)`.

Condition 1 holds by definition for INTERLEAVED layout (tiles are laid out sequentially by address). Condition 2 holds because `N_heads * 32 * 128 = N_heads * 4096` elements; reshaping to `(N_heads, 128)` tile-padded is `(N_heads * T, 128) = (N_heads * 32, 128)` which maps to the same set of tiles. No re-tiling is required and the reshape is zero-copy.

However, `ttnn.reshape` still incurs **Python dispatch overhead** (the call crosses the pybind11 layer and the TTNN runtime validates metadata). This is estimated at 1–3 µs per call [ESTIMATE], which is small relative to the transition costs but non-zero. With four reshape calls total (3D→2D and 2D→3D for each of Q and K), the aggregate reshape dispatch overhead is approximately **4–12 µs** [ESTIMATE].

Including this in the full accounting:

Table: Complete QK norm overhead per decode step [ESTIMATE]

| Item | Cost |
|---|---|
| T_norm_in_Q + T_norm_in_K | ≈ 32 µs |
| Reshape dispatch overhead × 4 | ≈ 4–12 µs |
| TTNNRMSNorm dispatch × 2 (Q + K) | ≈ 6–16 µs |
| T_norm_out_Q + T_norm_out_K | ≈ 32 µs |
| **Total** | **≈ 74–92 µs** |

## Is the L1 Move Required by `TTNNRMSNorm`'s Input Constraints?

The L1 move (T_norm_in) moves Q and K from DRAM INTERLEAVED to L1 INTERLEAVED before the norm call. The relevant question is whether this move is **required by the norm kernel** or whether it is a **precautionary software pattern** that could be eliminated.

### Evidence That the Move Is Kernel-Required

`TTNNRMSNorm` in the TTNN library processes its input using Tensix cores. Like all TTNN compute kernels, it must read input data from L1 — Tensix cores cannot directly address DRAM. For small tensors at decode batch=1, the kernel dispatch infrastructure does not automatically stage data from DRAM to L1 before execution; that staging is the caller's responsibility, expressed via `ttnn.to_memory_config`.

Therefore, the requirement to have input data in L1 before calling `TTNNRMSNorm` is a **hardware constraint**, not merely a software convention. The L1 move is required, not precautionary.

### Whether the Move Could Be Absorbed Into the Preceding Stage

The distinction with "precautionary" arises when considering whether a different arrangement of the preceding operations could **avoid creating the situation** where a DRAM→L1 move is needed:

- If Q arrived at the norm call already in L1 (i.e., if the post-RoPE eviction T2a were not performed), T_norm_in_Q would not be needed. This is the kernel-fusion strategy described in Chapter 4, Priority 1.
- If Q were produced directly in the 2D L1 format required by the norm (bypassing the HEIGHT_SHARDED RoPE path entirely), T_norm_in would similarly be unnecessary.

Neither of these is achievable without modifying the RoPE kernel or inserting a kernel that bridges the HEIGHT_SHARDED RoPE output directly to the 2D L1 norm input. The L1 move as a standalone operation is genuinely required given current kernel constraints; it is not a conservatism that can be removed by reordering Python calls.

### Can `TTNNRMSNorm` Accept HEIGHT_SHARDED Input?

The 3D→2D reshape requirement (Step 3) implies that `TTNNRMSNorm` does not currently accept the HEIGHT_SHARDED `(32, 128)` per-head shard format produced by the RoPE kernel. If it did, the reshape step would be unnecessary and the norm could be called on `q_rope_out` directly (after T_norm_in, or ideally without T_norm_in if the norm kernel can also accept DRAM input).

Based on the analysis in Chapter 4, `optimization_opportunities.md`, whether `TTNNRMSNorm` can be configured to accept HEIGHT_SHARDED L1 input with shard `(32, 128)` is an open implementation question. The norm kernel would need to compute the RMS reduction over each core's shard independently — which is correct per-head if the shard represents exactly one head's token vector. For Ling's `(32, 128)` shard shape at decode seq_len=1, each shard contains one tile row (32 elements) padded to tile size, with the actual token data in the first row. The RMS reduction would need to operate over the active (non-padded) elements only, requiring mask awareness or a kernel that handles the tile-row padding correctly.

This is **feasible in principle** but requires a modification to `TTNNRMSNorm` or a separate kernel variant. It is not available in the baseline implementation.

## Options for Reducing Overhead

### Option A — Fuse RoPE and Norm into a Single L1-Resident Pass

The highest-impact option (Priority 1 from Chapter 4) eliminates T2a, T2b, T_norm_in_Q, and T_norm_in_K simultaneously by keeping Q and K in L1 from the RoPE output through the norm computation. This requires:

1. `TTNNRMSNorm` accepting HEIGHT_SHARDED L1 input (or a new kernel variant that does).
2. The norm output remaining in L1 in a layout compatible with subsequent consumers (SDPA, paged KV update).

If successful, the T_norm_in costs (32 µs) and the T2a/T2b costs (32 µs) are both eliminated. T_norm_out remains (32 µs, non-eliminable without a norm output config change). Net saving: ≈ **64 µs** [ESTIMATE] per decode step.

```python
# Target state (if TTNNRMSNorm accepts HEIGHT_SHARDED L1 input):
# q_rope_out is already in L1/HS after the RoPE kernel.
# No T2a, no T_norm_in_Q.
q_norm_out = TTNNRMSNorm(q_rope_out, weight=self.q_norm_weight, epsilon=self.norm_eps)
# q_norm_out: L1/HS or L1/ITVL (depending on norm kernel output config)
# T_norm_out_Q still required if norm writes DRAM INTERLEAVED output
```

### Option B — Fuse Reshape Into the Norm Kernel

If a norm kernel variant were available that accepted 4D HEIGHT_SHARDED input and internally handled the per-head reduction, the four reshape calls (Steps 3 and 5 for Q and K) would be eliminated. The aggregate reshape dispatch saving is small (≈ 4–12 µs [ESTIMATE]) relative to the transition costs, so this option has a lower priority. It is only meaningful if pursued in conjunction with Option A (which addresses the larger T_norm_in costs).

### Option C — In-Place Norm to Avoid T_norm_out

T_norm_out (32 µs) is labeled non-eliminable in Chapter 4 because the norm kernel writes its output to L1 and the caller then evicts it to DRAM. If the norm kernel supported a configurable `output_memory_config` argument that wrote directly to DRAM, T_norm_out would be absorbed into the kernel's own write path. This is a kernel interface change, not a Python-level change.

An alternative formulation: if `paged_sdpa_decode` and `paged_update_on_device` can both accept L1-resident tensors for Q and K respectively, T_norm_out could be eliminated entirely by keeping Q and K in L1 after the norm and passing the L1 tensors directly to the downstream kernels. This requires verifying the input constraints of both downstream kernels — a task covered in Chapter 4 (SDPA side) and partially in the V-path analysis.

### Option D — Accept the Cost as Negligible Relative to SDPA

For long KV context lengths (S >> 512 tokens), `T_SDPA` dominates attention decode latency and the 70–90 µs norm overhead becomes a smaller fraction of total step time. In production deployments where S is typically 2,000–8,000 tokens, the norm overhead may represent only 5–15% of total attention latency. In that regime, Options A–C may not be worth the engineering investment relative to other optimizations.

For Ling's intended use case, whether Option D is acceptable depends on the target deployment S distribution and latency budget. For S < 512, Options A–C have high return on investment.

Table: Option comparison for reducing QK norm overhead

| Option | Mechanism | Potential saving | Requires kernel change? | Risk |
|---|---|---|---|---|
| A — Fuse RoPE→Norm (keep L1) | Eliminate T2a+T2b+T_norm_in | ≈ 64 µs | Yes — `TTNNRMSNorm` must accept HEIGHT_SHARDED input | Medium; norm correctness with tile padding |
| B — Reshape-fused norm kernel | Eliminate reshape dispatch overhead | ≈ 4–12 µs | Yes — need reshape-aware norm variant | Low; minor saving only |
| C — In-place norm (no T_norm_out) | Eliminate T_norm_out | ≈ 32 µs | Yes — norm needs output config arg or downstream L1 acceptance | Medium; changes downstream tensor locations |
| D — Accept as negligible | No change | 0 µs | No | None (for long-context serving) |

The recommended path is **Option A** for short- to medium-context deployments, combined with **Option C** if both downstream kernels can be confirmed to accept L1 input.

The combined saving of Options A+C spans both the norm path and the preceding post-RoPE evictions that exist specifically to feed the norm. The full eliminable cost is:

| Eliminated component | Cost |
|---|---|
| T2a + T2b (post-RoPE evictions, from Ch4) | ≈ 32 µs |
| T_norm_in_Q + T_norm_in_K (DRAM→L1 moves) | ≈ 32 µs |
| T_norm_out_Q + T_norm_out_K (L1→DRAM evictions, Option C) | ≈ 32 µs |
| Reshape dispatch overhead × 4 | ≈ 4–12 µs |
| **Total eliminable (Options A+C, including reshape)** | **≈ 100–108 µs** |

If reshape dispatch overhead is excluded (e.g. reshapes are retained in a fused kernel), the saving from T2a+T2b+T_norm_in+T_norm_out alone is ≈ **96 µs** [ESTIMATE] — this is the figure referenced in earlier summaries, and it represents the four DRAM↔L1 transition costs combined, not the full 74–92 µs norm-path overhead alone. Adding reshape elimination raises the total to 100–108 µs [ESTIMATE].

---

**Next:** [Partial Rotary RoPE](./partial_rotary_rope.md)
