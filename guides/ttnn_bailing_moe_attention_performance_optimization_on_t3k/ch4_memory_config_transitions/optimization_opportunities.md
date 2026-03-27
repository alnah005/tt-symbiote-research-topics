# Optimization Opportunities for Memory-Config Transitions

## Overview

This file classifies each memory-config transition from `decode_tensor_lifecycle.md` as either **eliminable** (removable through changes to surrounding code or memory config choices) or **kernel-mandated** (required by the current implementation of the RoPE, paged update, or SDPA kernel and not removable without changing the kernel itself). For eliminable transitions, concrete change locations in `TTNNBailingMoEAttention` are provided.

The cost model from `transition_cost_model.md` quantifies the stakes: the total estimated transition overhead per decode step is approximately **150–171 µs** [ESTIMATE], with individual transitions costing 11–21 µs each. Eliminating even two or three transitions would reduce per-step attention latency by 20–40 µs [ESTIMATE], which at 1000 decode steps would reclaim 20–40 ms of total generation time.

## Classification of Each Transition

### T1a, T1b: Q and K DRAM → L1 (RoPE Input)

**Class: Kernel-mandated (currently). Partially eliminable with upstream change.**

`TTNNRotaryPositionEmbedding` requires its input in HEIGHT_SHARDED L1. T1a and T1b are required by this constraint. They cannot be removed without modifying the RoPE kernel or replacing it with a variant that accepts DRAM INTERLEAVED input.

The mandatory steps leading up to T1a are: `ttnn.slice` produces `q_slice` in shape `(1,1,32,2048)` (DRAM INTERLEAVED), followed by `ttnn.reshape` to `(1,16,32,128)` (zero-copy on contiguous DRAM), followed by `ttnn.to_memory_config` into `rope_shard_mem_q` (T1a — the actual HEIGHT_SHARDED shard copy). The reshape is mandatory: the flat `(1,1,32,2048)` geometry is incompatible with the `(1,16,32,128)` shard spec required by `rope_shard_mem_q`, and the reshape must occur on the contiguous DRAM tensor to remain zero-copy. The correct three-step sequence is:

```python
# Current pattern (three operations: slice → DRAM, reshape → DRAM, to_memory_config → L1)
q_slice = ttnn.slice(qkv_replicated, begins, ends)              # DRAM INTERLEAVED: (1,1,32,2048)
# q_slice exits the slice as (1,1,32,2048) — flat, 32 tile-rows.
# rope_shard_mem_q requires (1,16,32,128): 16 cores × 32 rows = 512 tile-rows.
# The flat (1,1,32,2048) geometry is incompatible with the shard spec;
# a reshape to (1,16,32,128) is mandatory before the HEIGHT_SHARDED config can be applied.
q_reshaped = ttnn.reshape(q_slice, (1, 16, 32, 128))            # zero-copy: contiguous DRAM tensor
q_rope_in = ttnn.to_memory_config(q_reshaped, rope_shard_mem_q) # T1a — one explicit shard copy

# T1a is kernel-mandated; it cannot be removed by passing memory_config=rope_shard_mem_q
# directly to ttnn.slice, because the slice output is (1,1,32,2048) and the reshape to
# (1,16,32,128) must occur on a contiguous DRAM tensor to be zero-copy.
# Attempting to reshape a HEIGHT_SHARDED tensor requires redistributing tiles across L1
# banks and is not zero-copy.
```

**Feasibility:** T1a and T1b are currently kernel-mandated. The mandatory `ttnn.reshape` from `(1,1,32,2048)` to `(1,16,32,128)` must occur before the HEIGHT_SHARDED `to_memory_config` and must operate on a contiguous DRAM tensor to be zero-copy. This means the slice and reshape must remain on DRAM, and the HEIGHT_SHARDED shard copy (T1a/T1b) cannot be bypassed. The transitions are not eliminable without modifying the RoPE kernel to accept DRAM INTERLEAVED input (removing the HEIGHT_SHARDED requirement entirely).

**Code location:**
```
models/tt_transformers/tt/attention.py
  → TTNNBailingMoEAttention.forward()
  → Ensure the Q path follows the mandatory sequence:
      ttnn.slice → DRAM INTERLEAVED (1,1,32,2048)
      ttnn.reshape to (1,16,32,128) — zero-copy on contiguous DRAM
      ttnn.to_memory_config → rope_shard_mem_q (T1a — required shard copy)
  → Analogously for K: reshape to (1,4,32,128) before rope_shard_mem_k (T1b).
```

**Status:** Kernel-mandated. T1a (~21 µs) and T1b (~11 µs) cannot be eliminated without a RoPE kernel change. The reshape is a required, zero-copy intermediate step on DRAM.

---

### T2a, T2b: Q and K L1 → DRAM (Post-RoPE Eviction)

**Class: Conditionally kernel-mandated. Eliminable if downstream kernels accept L1-sharded input.**

T2a and T2b evict the post-RoPE Q and K tensors from L1 to DRAM. These transitions exist because the next consumers of Q and K (the QK norm kernel and `paged_update_on_device`) are assumed to require DRAM INTERLEAVED input. However, this is not necessarily true for all consumers:

**Case 1 — `paged_update_on_device` accepts L1 HEIGHT_SHARDED input (same shard spec as RoPE output):**
If the `paged_update_on_device` kernel can consume K (and V) directly from L1 in `rope_shard_mem` format, then T2b (K eviction) and T3a (K reload) can both be eliminated. The post-RoPE K tensor in L1 can be passed directly to `paged_update_on_device` without an intermediate DRAM round-trip:

```python
# Current: K goes L1 → DRAM (T2b) then DRAM → L1 (T3a)
k_post_rope = ttnn.to_memory_config(k_rope_out, ttnn.DRAM_MEMORY_CONFIG)   # T2b
k_update_in = ttnn.to_memory_config(k_post_rope, kv_update_mem)            # T3a

# Proposed: K stays in L1 throughout, if kv_update_mem == rope_shard_mem_k
# (or paged_update_on_device can accept rope_shard_mem_k directly)
k_update_in = k_rope_out   # no transition needed — reuse L1 buffer
```

This requires verifying that `rope_shard_mem_k` and `kv_update_mem` specify the same shard shape, grid, and orientation. From `decode_tensor_lifecycle.md`, both use `shard=(32,128)` on a 4-core grid for K — they may already be identical or easily made identical by aligning the grid specification.

**Potential saving:** Eliminates T2b (~11 µs) and T3a (~11 µs) = ~22 µs per decode step [ESTIMATE].

**Case 2 — QK norm requires DRAM INTERLEAVED (kernel constraint):**
If `TTNNRMSNorm` requires DRAM INTERLEAVED input (as analysed in Chapter 6), then T2a (Q eviction) cannot be eliminated — Q must leave L1 before entering the norm path. In this case T2a remains kernel-mandated. However, T2b (K eviction) may still be eliminable via Case 1 above.

**Code location for the change:**
```
models/tt_transformers/tt/attention.py
  → TTNNBailingMoEAttention.forward()
  → The ttnn.to_memory_config calls that produce k_post_rope (T2b)
    and v_post_rope (not listed separately but analogous).
  Change: remove the to_memory_config call(s) and pass k_rope_out
          directly to paged_update_on_device, conditioned on the
          kernel accepting the same shard spec.
```

---

### T3a, T3b: K and V DRAM → L1 (paged_update_on_device Input)

**Class: Kernel-mandated if K and V arrive from DRAM. Eliminable if upstream transitions are fused (see T2b analysis above).**

If T2b is not eliminated (K must go to DRAM post-RoPE), then T3a is unavoidable — K must come back from DRAM to L1 for `paged_update_on_device`. The same logic applies to V (T3b): V is never processed by RoPE and arrives in DRAM from the initial split.

**T3b is not eliminated.** The final `to_memory_config` call that copies V from DRAM to L1 HEIGHT_SHARDED (`kv_update_mem`) still executes. What can be eliminated is the redundant intermediate `to_memory_config` staging call that existed in the baseline between the reshape and the final shard copy. Passing `memory_config=kv_update_mem` directly to `ttnn.slice` would produce a HEIGHT_SHARDED L1 tensor immediately after the slice — but calling `ttnn.reshape` on that HEIGHT_SHARDED tensor is not a zero-copy operation (reshaping a HEIGHT_SHARDED tensor requires redistributing tiles across L1 banks, which is a data movement operation). The correct optimized sequence keeps the slice and reshape in DRAM, then performs one explicit shard copy:

```python
# Correct three-step sequence for V (Priority 4 approach):
v_slice = ttnn.slice(qkv_replicated, ...)            # DRAM INTERLEAVED — no memory_config arg
v_reshaped = ttnn.reshape(v_slice, (1, 4, 32, 128))  # zero-copy on contiguous DRAM tensor
v_update = ttnn.to_memory_config(v_reshaped, kv_update_mem)  # one explicit shard copy (T3b)

# Baseline had four steps:
#   slice → DRAM, reshape → DRAM, to_memory_config → DRAM (redundant staging), to_memory_config → L1
# Optimized has three steps:
#   slice → DRAM, reshape → DRAM, to_memory_config → L1
# One to_memory_config dispatch overhead (~8–11 µs) is eliminated; T3b (the final shard copy) still runs.
```

**Potential saving for the eliminated intermediate staging call:** ~8–11 µs per decode step [ESTIMATE] (one dispatch overhead removed; the final DRAM→L1 shard copy for T3b still occurs and is not saved).
**Potential saving for T3a:** ~11 µs (if T2b is also eliminated) [ESTIMATE].

**Code location:**
```
models/tt_transformers/tt/attention.py
  → TTNNBailingMoEAttention.forward()
  → The V path: ttnn.slice → ttnn.reshape → (redundant ttnn.to_memory_config → DRAM) → ttnn.to_memory_config → L1.
  Change: remove the redundant intermediate to_memory_config staging call.
          Do NOT pass memory_config=kv_update_mem to ttnn.slice — keep the slice
          and reshape operating on DRAM INTERLEAVED tensors so that reshape
          remains a zero-copy metadata operation.
```

---

### T4: Q DRAM → L1 (Conditional SDPA Input)

**Class: Kernel-mandated if `paged_sdpa_decode` requires L1-sharded Q. Eliminable if the kernel accepts DRAM INTERLEAVED.**

T4 is conditional: it exists only if `paged_sdpa_decode` requires Q in a specific L1-sharded layout. The `paged_sdpa_decode` kernel (see Chapter 5, `paged_sdpa_chunk_sizes.md`) may already accept DRAM INTERLEAVED Q input, in which case T4 does not exist in the current implementation.

If T4 does exist, it can potentially be fused with the end of the QK norm path: instead of emitting Q to DRAM after norm, the norm kernel's output could target the SDPA-required shard spec directly.

**Code location (if T4 exists):**
```
models/tt_transformers/tt/attention.py
  → TTNNBailingMoEAttention.forward()
  → The ttnn.to_memory_config call immediately before the
    paged_sdpa_decode invocation for Q.
  Change: investigate whether paged_sdpa_decode accepts
          DRAM_MEMORY_CONFIG for Q input; if yes, remove T4.
```

---

### T_norm: QK Norm Transitions

**Class: Kernel-mandated for the norm itself. The round-trip structure is eliminable with kernel fusion.**

`TTNNRMSNorm` requires a specific input memory config (typically 2D INTERLEAVED in L1 — see Chapter 6). The transitions that move Q and K into and out of that format (T_norm_in and T_norm_out) are currently separate data movement kernels. This round-trip structure (DRAM→L1 for norm, then L1→DRAM after) is the largest aggregate overhead block at ~64 µs [ESTIMATE].

The round-trip is not inherently kernel-mandated at the norm kernel level — it is a consequence of how the surrounding code sequences the operations. Two elimination strategies exist:

**Strategy A — Fuse reshape into the slice (upstream):**
If Q can be produced in the 2D L1 format required by `TTNNRMSNorm` at the point of the initial split, T_norm_in (the DRAM→L1 move before norm) is eliminated. This requires aligning the norm input memory config with whatever layout is most efficient for the upstream slice.

**Strategy B — Fuse RoPE and norm into a single L1-resident pass:**
If the norm is applied after RoPE (and the product of `use_qk_norm=True` and `partial_rotary_factor < 1.0` makes this correct), Q and K can remain in L1 from the RoPE kernel through the norm kernel, avoiding both T2a (L1→DRAM eviction post-RoPE) and T_norm_in (DRAM→L1 reload for norm). This requires that the norm kernel can accept HEIGHT_SHARDED input in the same shard spec produced by RoPE.

**Code location:**
```
models/tt_transformers/tt/attention.py
  → TTNNBailingMoEAttention.forward()
  → The sequence: to_memory_config → reshape → TTNNRMSNorm → reshape
    that wraps Q and K normalisation.
  Change (Strategy A): align the Q-slice (and K-slice, if use_qk_norm applies to K as well) memory config to norm input requirements.
  Change (Strategy B): investigate TTNNRMSNorm accepting HEIGHT_SHARDED input;
    if feasible, remove T2a and T_norm_in for both Q and K.

models/tt_transformers/tt/norm.py
  → TTNNRMSNorm.__call__()
  → Input validation and memory config assertions.
  Examine: whether HEIGHT_SHARDED input with shard=(32,128) is supported or
    can be added without changing the kernel binary.
```

---

## Summary Table: Classification and Opportunities

Table: Memory-config transition classification and optimisation potential

| ID | Transition | Class | Eliminable? | Strategy | Estimated saving [ESTIMATE] |
|---|---|---|---|---|---|
| T1a | Q DRAM→L1 (RoPE) | Kernel-mandated (input format) | No — mandatory reshape `(1,1,32,2048)`→`(1,16,32,128)` must precede the shard copy on a contiguous DRAM tensor; T1a cannot be bypassed without a RoPE kernel change | Ensure correct three-step sequence: slice→DRAM, reshape→DRAM (zero-copy), `to_memory_config`→L1 | Not eliminable without kernel change |
| T1b | K DRAM→L1 (RoPE) | Kernel-mandated (input format) | No — same reshape constraint applies; `(1,1,32,512)` must be reshaped to `(1,4,32,128)` before `rope_shard_mem_k` can be applied | Same three-step sequence as T1a | Not eliminable without kernel change |
| T2a | Q L1→DRAM (post-RoPE) | Conditionally mandated (norm input) | Partial — eliminable if norm accepts L1-sharded input | Investigate `TTNNRMSNorm` input constraints (Chapter 6) | ~21 µs |
| T2b | K L1→DRAM (post-RoPE) | Eliminable — if `paged_update_on_device` accepts same shard spec | Yes | Reuse `k_rope_out` directly for paged update | ~11 µs (included in Priority 1 — do not add separately) |
| T3a | K DRAM→L1 (paged update) | Eliminable — if T2b is eliminated | Yes (dependent on T2b) | Depends on T2b change | ~11 µs |
| T3b | V DRAM→L1 (paged update) | Not eliminated — still executes; the final DRAM→L1 shard copy to `kv_update_mem` remains required | Priority 4 removes the redundant intermediate `to_memory_config` staging call before the final shard copy, not T3b itself | ~8–11 µs (dispatch overhead of eliminated intermediate staging call) |
| T4 | Q DRAM→L1 (SDPA, conditional) | Kernel-mandated if present | Yes — if `paged_sdpa_decode` accepts DRAM input | Verify kernel input spec; may not exist | ~21 µs |
| T_norm_in (Q+K) | QK norm Q+K DRAM→L1 (in, before norm) | Kernel-mandated (norm input format) | Partially — eliminable via Priority 1 if norm accepts HEIGHT_SHARDED input | Strategy A or B above (see Chapter 6) | ~32 µs |
| T_norm_out (Q+K) | QK norm Q+K L1→DRAM (out, after norm) | Kernel-mandated (norm output format) | No — non-eliminable without norm output config change | Modify `TTNNRMSNorm` to support configurable output memory config | Not eliminable without kernel change |

## Prioritised Action List

Given the cost model, the highest-value changes in order of estimated impact are:

**Priority 1 — Investigate and remove QK norm round-trips (~64 µs potential saving):**

*Comprises: T2a (21 µs) + T2b (11 µs) + T_norm_in_Q (21 µs) + T_norm_in_K (11 µs) = 64 µs total.*

The QK norm transitions are the largest aggregate overhead. The key question is whether `TTNNRMSNorm` can be modified (or an alternative norm kernel selected) to accept HEIGHT_SHARDED L1 input. If yes, Q and K can remain in L1 from the RoPE output through the norm computation, eliminating T2a, T2b, T_norm_in for both Q and K, and saving approximately **64 µs** [ESTIMATE].

```python
# Target state (if TTNNRMSNorm accepts HEIGHT_SHARDED L1):
q_norm_out = TTNNRMSNorm(q_rope_out)   # q_rope_out stays in L1/HS, no eviction
k_norm_out = TTNNRMSNorm(k_rope_out)   # k_rope_out stays in L1/HS, no eviction
# T2a, T2b, T_norm_in (Q), T_norm_in (K) all eliminated
```

### Priority 2 — T1a/T1b: Q and K HEIGHT_SHARDED shard setup

See the [T1a/T1b analysis above](#t1a-t1b-q-and-k-dram--l1-rope-input) — these transitions are kernel-mandated and not eliminable without a RoPE kernel change. No additional action required beyond what is already described there.

**Priority 3 — Reuse `k_rope_out` for `paged_update_on_device` (~11 µs saving):**

If the `rope_shard_mem_k` and `kv_update_mem` configurations are identical (same shard shape, same grid), K can be passed directly from the RoPE output to `paged_update_on_device` without reloading from DRAM (T3a). T2b (K eviction, ~11 µs) is already counted in Priority 1 as part of the QK norm round-trip elimination. The non-overlapping saving here is T3a only (~11 µs), and requires removing the `ttnn.to_memory_config` reload call and confirming the shard specs match.

**Priority 4 — Eliminate the intermediate `v_raw` staging tensor in the V path (~8–11 µs saving):**

V is never processed by RoPE or QK norm. In the baseline path, V passes through four steps: `ttnn.slice` → DRAM INTERLEAVED, `ttnn.reshape` → DRAM INTERLEAVED, `ttnn.to_memory_config` → DRAM INTERLEAVED (a no-op staging step), `ttnn.to_memory_config` → L1 HEIGHT_SHARDED (`kv_update_mem`). The optimized path collapses this to three steps by removing the redundant intermediate `to_memory_config` call.

The correct optimized sequence is:

1. `ttnn.slice(qkv_replicated, ...)` — output is DRAM INTERLEAVED (no `memory_config` argument here)
2. `ttnn.reshape(v_slice, (1, 4, 32, 128))` — zero-copy metadata operation on the contiguous DRAM tensor
3. `ttnn.to_memory_config(v_reshaped, kv_update_mem)` — one explicit shard copy from DRAM to L1

Note: passing `memory_config=kv_update_mem` directly to `ttnn.slice` would produce a HEIGHT_SHARDED L1 tensor immediately after the slice. Calling `ttnn.reshape` on that sharded tensor is not a zero-copy operation — reshaping a HEIGHT_SHARDED tensor requires redistributing tiles across L1 banks, which is a data movement operation. That sequence would therefore not be functionally equivalent and the zero-copy reshape property would not apply.

What this optimization eliminates: the redundant intermediate `to_memory_config` call that existed in the baseline between the reshape and the final shard copy. The final `to_memory_config` to `kv_update_mem` still executes — it is not eliminated.

What this optimization does not eliminate: the `ttnn.slice`, the `ttnn.reshape`, or the final `ttnn.to_memory_config` call to `kv_update_mem`. The shard copy itself still occurs; only one dispatch overhead is saved.

```python
# Optimized: correct three-step sequence for V
v_slice = ttnn.slice(qkv_replicated, ...)            # DRAM INTERLEAVED — no memory_config arg
v_reshaped = ttnn.reshape(v_slice, (1, 4, 32, 128))  # zero-copy on contiguous DRAM tensor
v_update = ttnn.to_memory_config(v_reshaped, kv_update_mem)  # one explicit shard copy

# Baseline had four steps:
#   slice → DRAM, reshape → DRAM, to_memory_config → DRAM (redundant staging), to_memory_config → L1
# Optimized has three steps:
#   slice → DRAM, reshape → DRAM, to_memory_config → L1
# One to_memory_config dispatch overhead (~8–11 µs) is eliminated.
```

**Combined potential saving (Priorities 3–4, no kernel changes required):**

```
T3a + intermediate V staging call = 11 + ~8–11 = ~19–22 µs [ESTIMATE]
```

Note: T1a and T1b are kernel-mandated and cannot be eliminated without a RoPE kernel change — the mandatory reshape from `(1,1,32,2048)` to `(1,16,32,128)` must occur on a contiguous DRAM tensor before the HEIGHT_SHARDED shard copy; T1a and T1b are therefore not counted here. T3b (the final `to_memory_config` to `kv_update_mem`) still executes and is not counted here — Priority 4 eliminates only the redundant intermediate staging call before it. T2b (~11 µs) is not included here because it is already counted within Priority 1's QK norm round-trip elimination; including it again would double-count that saving.

Approximately **19–22 µs** can be recovered per decode step using only Python-level code changes in `TTNNBailingMoEAttention`, without any modifications to TTNN kernels. Combined with Priority 1 (if `TTNNRMSNorm` can accept HEIGHT_SHARDED input), the total recoverable overhead approaches **83–86 µs** [ESTIMATE] out of the estimated **150–171 µs** total transition cost.

## Non-Eliminable Transitions

After the above optimisations, the transitions that cannot be removed without deeper kernel changes are:

1. **T1a, T1b (Q and K DRAM→L1 for RoPE):** The mandatory `ttnn.reshape` from `(1,1,32,2048)` to `(1,16,32,128)` (for Q) and from `(1,1,32,512)` to `(1,4,32,128)` (for K) must occur on contiguous DRAM tensors before the HEIGHT_SHARDED `to_memory_config` can be applied. This means the shard copy itself (T1a/T1b) cannot be bypassed. Elimination requires modifying the RoPE kernel to accept DRAM INTERLEAVED input.

2. **T_norm_out (Q and K, L1 → DRAM after norm):** As long as the norm kernel writes its output to DRAM INTERLEAVED, the post-norm tensors must be in DRAM before the next stage. Removing this requires modifying `TTNNRMSNorm` to support a configurable output memory config.

3. **T4 (if present and kernel-mandated):** If `paged_sdpa_decode` requires L1-sharded Q and there is no version of the kernel that accepts DRAM INTERLEAVED Q, T4 remains. This is investigated in Chapter 5, `paged_sdpa_chunk_sizes.md`.

4. **DRAM-to-L1 within `paged_update_on_device`:** The paged update kernel internally reads from DRAM (the KV cache pool) and writes to DRAM (updating pages). This internal DRAM access is part of the kernel's function and is not exposed as a separate `to_memory_config` call.

---

**Next:** [Chapter 5 — SDPA and Compute Config](../ch5_sdpa_and_compute_config/index.md)
