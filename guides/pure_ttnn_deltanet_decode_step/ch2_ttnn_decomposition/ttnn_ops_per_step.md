# TTNN Ops Per Decode Step

This file enumerates all 12 TTNN operations required for one complete DeltaNet recurrent decode step, counting from after the linear projections and any all-gather collective. For each operation the table provides the TTNN API call, input shapes, output shape, recommended memory configuration, and availability status.

All shapes use the concrete Qwen3.6-35B-A3B dimensions: B=1, nH=32, d_k=128, d_v=128. `nH_local` refers to the per-device head count under 8-way head-parallel sharding on T3K (32 / 8 = 4).

## Op Table

| Step | Name | TTNN API | Input shapes | Output shape | Memory config | Availability |
|------|------|----------|--------------|--------------|---------------|--------------|
| 1 | QKV split | `ttnn.split` or `ttnn.slice` | `[B, seq=1, hidden]` | 3× `[B, 1, proj_dim]` | L1 interleaved | [AVAILABLE — needs wiring] |
| 2 | Q/K head expand | `ttnn.reshape` + `ttnn.repeat` | `[B, 1, nH_local * d_k]` | `[B, nH_local, d_k, 1]` | L1 interleaved | [AVAILABLE — needs wiring] |
| 3 | Decay gate g_t | `ttnn.exp(ttnn.mul(-ttnn.exp(A_log), ttnn.softplus(ttnn.add(a_t, dt_bias))))` | `[B, nH_local, 1, 1]` each | `[B, nH_local, 1, 1]` | L1 interleaved | [AVAILABLE — needs wiring] |
| 4 | Update rate β_t | `ttnn.sigmoid(b_t)` | `[B, nH_local, 1, 1]` | `[B, nH_local, 1, 1]` | L1 interleaved | [AVAILABLE — needs wiring] |
| 5 | L2 normalize K̃, Q̃ | `ttnn.normalize` (per-head) | `[B, nH_local, d_k, 1]` | `[B, nH_local, d_k, 1]` | L1 interleaved | [AVAILABLE — needs wiring] |
| 6 | Decay state | `ttnn.mul(g_broadcast, S_prev)` | `[B, nH_local, 1, 1]`, `[B, nH_local, d_k, d_v]` | `[B, nH_local, d_k, d_v]` | DRAM → L1 | [AVAILABLE — needs wiring] |
| 7 | Retrieval | `ttnn.matmul(S_prev, k_tilde, transpose_a=True)` | `[B, nH_local, d_k, d_v]`, `[B, nH_local, d_k, 1]` | `[B, nH_local, d_v, 1]` | L1 interleaved | [AVAILABLE — needs wiring] |
| 8 | Error | `ttnn.mul(beta_broadcast, ttnn.sub(v_t, retrieval))` | `[B, nH_local, d_v, 1]`, `[B, nH_local, 1, 1]` | `[B, nH_local, d_v, 1]` | L1 interleaved | [AVAILABLE — needs wiring] |
| 9 | Write (outer product) | `ttnn.matmul(k_tilde, ttnn.transpose(error, -2, -1))` | `[B, nH_local, d_k, 1]`, `[B, nH_local, 1, d_v]` | `[B, nH_local, d_k, d_v]` | L1 interleaved | [AVAILABLE — needs wiring] |
| 10 | New state | `ttnn.add(S_decayed, write)` | 2× `[B, nH_local, d_k, d_v]` | `[B, nH_local, d_k, d_v]` | L1 → DRAM | [AVAILABLE — needs wiring] |
| 11 | Output readout | `ttnn.matmul(S_new, q_tilde, transpose_a=True)` | `[B, nH_local, d_k, d_v]`, `[B, nH_local, d_k, 1]` | `[B, nH_local, d_v, 1]` | L1 interleaved | [AVAILABLE — needs wiring] |
| 12 | Flatten output | `ttnn.reshape(o_t, [B, 1, nH_local * d_v])` | `[B, nH_local, d_v, 1]` | `[B, 1, nH_local * d_v]` | L1 interleaved | [AVAILABLE — needs wiring] |

## Annotated Code

The 12 operations as Python using the TTNN API, with `# why:` comments on non-obvious choices:

```python
# ── Step 1: QKV split ──────────────────────────────────────────────────────
q_proj, k_proj, v_proj = ttnn.split(qkv_packed, split_size, dim=-1)
# why: projections arrive as a single concatenated tensor from the linear layer

# ── Step 2: Q/K head expand ───────────────────────────────────────────────
q_heads = ttnn.reshape(q_proj, [B, nH_local, d_k, 1])
k_heads = ttnn.reshape(k_proj, [B, nH_local, d_k, 1])
v_heads = ttnn.reshape(v_proj, [B, nH_local, d_v, 1])
# why: the recurrence ops expect the head dimension as dim=1
# In grouped-query attention (GQA), K/V have fewer heads than Q and must be
# repeated to match nH_local before per-head recurrence ops can broadcast correctly.
q_heads = ttnn.repeat(q_heads, ttnn.Shape([1, gqa_repeat_factor, 1, 1]))
k_heads = ttnn.repeat(k_heads, ttnn.Shape([1, gqa_repeat_factor, 1, 1]))
# why: repeat expands K and Q from nKV_local heads to nH_local heads along dim=1
# TODO: verify — if the downstream matmuls (ops 5–11) handle GQA broadcasting
#       internally, this ttnn.repeat may not be needed and can be removed.

# ── Step 3: Decay gate g_t ─────────────────────────────────────────────────
dt = ttnn.add(a_t, dt_bias)          # [B, nH_local, 1, 1]
g_t = ttnn.exp(
    ttnn.mul(
        -ttnn.exp(A_log),            # [nH_local, 1, 1] — head-wise log decay
        ttnn.softplus(dt)            # softplus ensures positivity before negation
    )
)
# why: this is the standard discretization of the continuous decay parameter A

# ── Step 4: Update rate β_t ────────────────────────────────────────────────
beta_t = ttnn.sigmoid(b_t)           # [B, nH_local, 1, 1], range (0, 1)
# why: sigmoid bounds the update rate to prevent runaway writes

# ── Step 5: L2-normalize K̃ and Q̃ ──────────────────────────────────────────
k_tilde = ttnn.normalize(k_heads, dim=-2)   # normalize along d_k dimension
q_tilde = ttnn.normalize(q_heads, dim=-2)
# why: DeltaNet correctness requires unit-norm keys and queries

# ── Step 6: Decay state (reads S_prev from DRAM) ───────────────────────────
g_broadcast = ttnn.reshape(g_t, [B, nH_local, 1, 1])
S_decayed = ttnn.mul(g_broadcast, S_prev)
# why: S_prev is in DRAM (persists between decode steps); result stays in L1
#      during this decode step computation

# ── Step 7: Retrieval (MUST use S_prev, not S_decayed) ─────────────────────
retrieval = ttnn.matmul(S_prev, k_tilde, transpose_a=True)
# why: transpose_a makes ttnn treat S_prev as [d_v, d_k]:
#      [d_v, d_k] × [d_k, 1] → [d_v, 1]
# why S_prev not S_decayed: delta rule requires pre-decay state for retrieval

# ── Step 8: Error ──────────────────────────────────────────────────────────
beta_broadcast = ttnn.reshape(beta_t, [B, nH_local, 1, 1])   # mirror g_broadcast pattern
error_raw = ttnn.sub(v_heads, retrieval)     # [B, nH_local, d_v, 1]
error = ttnn.mul(beta_broadcast, error_raw)  # scale by update rate
# why: beta_broadcast [B, nH_local, 1, 1] broadcasts over d_v

# ── Step 9: Write (outer product k̃ ⊗ error) ───────────────────────────────
error_T = ttnn.transpose(error, -2, -1)      # [B, nH_local, 1, d_v]
write = ttnn.matmul(k_tilde, error_T)
# why: [d_k, 1] × [1, d_v] → [d_k, d_v] is the outer product in matrix form

# ── Step 10: New state (write result back to DRAM) ─────────────────────────
S_new = ttnn.add(S_decayed, write)
# why: S_new must be explicitly copied to DRAM; the buffer at the S_prev
#      address is updated in-place or via double-buffer swap (see Ch. 3)

# ── Step 11: Output readout ────────────────────────────────────────────────
o_t = ttnn.matmul(S_new, q_tilde, transpose_a=True)
# why: same transpose reasoning as retrieval — q̃ is in d_k-space;
#      S^T [d_v, d_k] × [d_k, 1] → [d_v, 1]

# ── Step 12: Flatten for downstream projection ─────────────────────────────
o_flat = ttnn.reshape(o_t, [B, 1, nH_local * d_v])
# why: downstream o_proj linear layer expects [B, seq, hidden_dim]
```

## Notes on Memory Config Column

The "Memory config" column in the op table uses shorthand:

- **L1 interleaved** — `ttnn.L1_MEMORY_CONFIG`; layout depends on tensor shape:
  - The main state tensor S (shape [d_k, d_v] = [128, 128], tile-aligned) uses `ttnn.TILE_LAYOUT`.
  - Tensor intermediates where either of the innermost two dimensions (last or second-to-last) is not a multiple of 32 must use `ttnn.ROW_MAJOR_LAYOUT`. This covers two cases: (1) column vectors with last dim = 1 (k_tilde, q_tilde, v_t, retrieval, error, o_t — shapes [B, nH_local, d_k, 1] or [B, nH_local, d_v, 1]); and (2) row vectors with second-to-last dim = 1, specifically `error_T` (shape [B, nH_local, 1, d_v]) produced by transposing `error` for the outer-product matmul in op 9. In both cases a dimension of 1 is not a multiple of 32; applying TILE_LAYOUT would either fail or silently pad to 32 (32× memory overhead).
- **DRAM → L1** — state is read from `ttnn.DRAM_MEMORY_CONFIG` into L1 for computation; the `ttnn.mul` output is an L1 tensor
- **L1 → DRAM** — the result of `ttnn.add` (S_new) must be written back to `ttnn.DRAM_MEMORY_CONFIG` so it persists for the next decode step

See [state_tensor_memory_config.md](state_tensor_memory_config.md) for the full DRAM layout and sizing.

## Op Count

- Total ops tabulated: **12**
- This count excludes linear projections (Q/K/V/O weight matmuls) and any all-gather collective
- Ops 3 and 8 each decompose into 2–3 fused-eligible sub-calls; the 12-op count treats them as logical units

> **Key Finding:** All 12 ops are available in TTNN. No new kernel development is required for the composed form. The latency gain from pure-TTNN vs. host roundtrip is analyzed in Chapter 4.
