# Development Roadmap

This file specifies a prioritized sequence of kernel development work required to move the Gated Delta Net implementation from its current state — TTNN for projections, PyTorch/CUDA for the recurrent core — to a fully on-device forward pass on T3K.

Each priority item is classified by:
- **Impact:** what decode or prefill metric improves when this item is complete
- **Complexity:** estimated engineering effort (Low / Medium / High)
- **Dependency:** whether this item must precede another

---

## Priority 1 — Fused Recurrent Delta Rule Decode Kernel

**Target operation:** `recurrent_gated_delta_rule` → custom TT-Metalium fused kernel

**Impact:** Eliminates the host round-trip on the decode critical path. Enables L1-resident state matrix, converting the dominant DRAM bottleneck (7.36 µs/layer state I/O) to an L1-resident operation at roughly half the byte traffic (~3.7 µs/layer).

**Why it is the highest priority:** Every decode token in every Gated Delta Net layer currently requires:
1. Copy state from host PyTorch tensor to CUDA device
2. Run `recurrent_gated_delta_rule` on CUDA GPU
3. Copy result back to host
4. Copy to TTNN device for the next projection step

This round-trip introduces latency that is independent of sequence length and cannot be hidden by any pipelining at the current implementation level.

**Tensor shapes (per device, after head-parallel sharding):**
```
State S:      [B, 4, 128, 128]  BF16  (4 heads per device)
k̃, q̃:        [B, 4, 128]       BF16
v, error:     [B, 4, 128]       BF16
g (scalar):   [B, 4]            BF16
β (scalar):   [B, 4]            BF16
```

**Per-device state size:** 4 × 128 × 128 × 2 = 131,072 bytes = 128 KiB — fits in a single Tensix core's 1.5 MiB L1.

**Kernel structure:** A single Metalium kernel that:
1. Loads `S` from DRAM into L1 (or keeps it L1-resident across calls)
2. Executes all 6 recurrent operations (decay, retrieval, error, outer-product write, update, output) in registers
3. Writes `S_new` to DRAM and outputs `o` to the output buffer

This is analogous to a fused matrix-vector kernel with an in-place state accumulation step.

**Complexity:** Medium. The arithmetic is straightforward (no WY-decomposition); the challenge is the per-head `[d_k, d_v]` state tiling across Tensix cores and ensuring the state can remain L1-resident between calls.

**Prerequisite for:** Priority 2 (conv state must also be on-device to avoid round-trip), Priority 3.

---

## Priority 2 — Causal Conv1D Decode Update

**Target operation:** `causal_conv1d_update` → on-device TTNN or Metalium kernel

**Impact:** Removes the last host-side round-trip in the decode path. After Priority 1, the decode step still requires a host round-trip for the conv1d state update. Completing Priority 2 makes the entire Gated Delta Net decode step on-device.

**Operation description:** At each decode step, the conv state `[B, 8192, 4]` (a sliding window of the last 4 inputs) is updated:
1. Shift the window: `conv_state[:, :, :-1] = conv_state[:, :, 1:]`
2. Insert the new input: `conv_state[:, :, -1] = new_input`
3. Apply depthwise 1D conv with weight `[8192, 1, 4]`: `output = sum(conv_state * weight, dim=-1)`

After head-parallel sharding, the per-device conv state is `[B, 1024, 4]`.

**Simplest TTNN implementation:** This can be expressed as a sequence of `ttnn.roll`, `ttnn.slice`, `ttnn.concat`, and `ttnn.mul` + `ttnn.sum` operations. The shift-insert-convolve pattern is regular enough for a composable TTNN solution without a custom Metalium kernel, though a fused kernel would be more efficient.

**Complexity:** Low (composable from TTNN primitives) to Medium (if fused for efficiency).

---

## Priority 3 — FusedRMSNormSwishGate

**Target operation:** `FusedRMSNormSwishGate` → composable TTNN or fused Metalium

**Impact:** Removes the last PyTorch fallback in the output normalization step. After Priorities 1–2, the gated RMSNorm step is the remaining off-device operation.

**Operation:**
```
FusedRMSNormSwishGate(x, z):
  x_norm = RMSNorm(x, weight)
  gate   = silu(z)          # σ(z) × z
  return x_norm × gate
```

Input shapes (per device, after head-parallel sharding):
```
x (core_attn_out): [B, 4, 128]   (4 heads × d_v = 128)
z (gate input):    [B, 4, 128]
weight:            [128]          (per-head RMSNorm weight)
```

**Composable TTNN implementation:**
```python
x_norm = ttnn.rms_norm(x, weight=rms_weight)
gate   = ttnn.silu(z)
out    = ttnn.mul(x_norm, gate)
```

This requires no custom kernel — only wiring the existing TTNN ops correctly and verifying that numerics match `FusedRMSNormSwishGate`. The unfused form allocates intermediate tensors (`x_norm`, `gate`) to DRAM; a fused single-kernel implementation would keep intermediates in L1.

**Complexity:** Low (composable, no new kernel required); Medium if fused kernel is desired.

---

## Priority 4 — Small Gate Projections (`in_proj_a`, `in_proj_b`)

**Target operation:** `nn.Linear` for `in_proj_a`, `in_proj_b` → `ttnn.linear` with replicated weights

**Impact:** Removes the last `nn.Linear` host-side projections. After Priorities 1–3, these are the only operations still on the host.

**Operation:**
```
in_proj_a: [B, T, 2048] → [B, T, 32]   (α scalars, one per V head)
in_proj_b: [B, T, 2048] → [B, T, 32]   (β scalars, one per V head)
```

These have output dimension 32 (= H_v), which is too small to benefit from column sharding. The appropriate strategy is **replicated weights**: each device holds the full `[2048, 32]` weight matrix and computes the full `[B, T, 32]` output, then each device takes its 4-head slice `[B, T, 4]`.

**Weight memory:** `[2048, 32]` × 2 bytes = 131,072 bytes = 128 KiB per projection — negligible.

**Complexity:** Low. This is a wiring change (switch from `nn.Linear` to replicated `ttnn.linear`) with no new kernel needed.

---

## Priority 5 — Chunkwise Delta Rule Prefill

**Target operation:** `chunk_gated_delta_rule` → Python chunk loop with TTNN primitives or custom Metalium kernel

**Impact:** Moves prefill from CUDA GPU to T3K Wormhole chips. This affects prefill throughput but not decode latency (which is addressed by Priorities 1–4).

**Phase 1 — Python loop with TTNN primitives:**

A loop over T/C = 128 chunks (for T=8192, C=64), with each chunk processed via TTNN calls:

```python
for c in range(num_chunks):
    Q_chunk = Q_tiled[:, c*C:(c+1)*C, :, :]   # [B, C, H_v, d_k]
    K_chunk = K_tiled[:, c*C:(c+1)*C, :, :]
    V_chunk = V_tiled[:, c*C:(c+1)*C, :, :]
    g_chunk = g_tiled[:, c*C:(c+1)*C, :]

    # Inner-chunk attention: [C, C] attention matrix
    A_chunk = ttnn.matmul(Q_chunk, K_chunk.T)   # [B, H_v, C, C]
    O_intra = ttnn.matmul(A_chunk, V_chunk)      # [B, H_v, C, d_v]

    # Cross-chunk: O_cross = D · (Q̃_chunk @ S_in)
    O_cross = ttnn.matmul(Q_chunk, S)            # [B, H_v, C, d_v]; diagonal D applied elementwise
    O_chunk = O_intra + O_cross

    # State update: S_new = g_C · S + U_C W_C^T (WY-decomposition output)
    S = update_state(S, K_chunk, V_chunk, g_chunk)  # custom logic
```

This is feasible as a first-step port. TTNN dispatch overhead for 128 chunks × 3 matmuls = 384 kernel launches adds latency, but eliminates the CUDA dependency.

**Phase 2 — Custom Metalium kernel:**

A fully fused single-kernel implementation processes all chunks in one dispatch, tiling K, Q, V through L1 and streaming the state update. This is the long-term goal and matches the approach taken by `chunk_gated_delta_rule` in flash-linear-attention for CUDA.

**Complexity:** Medium (Python loop) to High (fully fused Metalium kernel).

---

## Summary Table

| Priority | Operation | Gap type | Complexity | Impact |
|---|---|---|---|---|
| 1 | Fused recurrent delta rule (decode) | Custom kernel | Medium | Eliminates host round-trip; enables L1-resident state |
| 2 | Causal conv1d update (decode) | Wiring or custom kernel | Low–Medium | Removes last host op in decode path |
| 3 | FusedRMSNormSwishGate | Wiring (composable) | Low | Removes output norm fallback |
| 4 | `in_proj_a`, `in_proj_b` | Wiring (replicated TTNN linear) | Low | Removes last `nn.Linear` host projection |
| 5 | Chunkwise delta rule (prefill) | Custom kernel | Medium–High | Enables on-device prefill |

## No New Development Needed

The following operations are already complete in TTNN and require no further kernel work:
- SDPA for Gated Attention (both prefill and decode)
- Paged KV cache management
- Q/K RMSNorm and Q gating for Gated Attention
- All input/output projections for both attention types (except `in_proj_a/b`)
- All-gather CCL for tensor parallelism

---

**Previous:** [`tt_transformers_review.md`](./tt_transformers_review.md)
