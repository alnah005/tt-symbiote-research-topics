# Existing TTNN Primitives Survey

This file audits every TTNN and tt-transformers primitive relevant to the hybrid Gated Delta Net + Gated Attention forward pass. Status tags:

- `[AVAILABLE]` — exists in TTNN and is already used in the current implementation
- `[AVAILABLE — not yet connected]` — exists in TTNN but not yet wired into the DeltaNet code path; closing this gap requires Python-level changes only
- `[GAP — requires custom kernel]` — no adequate TTNN equivalent; requires new TT-Metalium kernel or significant new development

---

## 1. Linear Projections

| Operation | Input → Output | Status | Notes |
|---|---|---|---|
| `in_proj_qkv` | `[B, T, 2048] → [B, T, 8192]` | `[AVAILABLE]` | `ttnn.linear` with column sharding; `TTNNLinearIReplicatedWColSharded` |
| `in_proj_z` | `[B, T, 2048] → [B, T, 4096]` | `[AVAILABLE]` | Same pattern as `in_proj_qkv` |
| `in_proj_a` | `[B, T, 2048] → [B, T, 32]` | `[AVAILABLE — not yet connected]` | Output dim too small to shard; use replicated `ttnn.linear`; currently `nn.Linear` on host |
| `in_proj_b` | `[B, T, 2048] → [B, T, 32]` | `[AVAILABLE — not yet connected]` | Same as `in_proj_a` |
| `out_proj` (DeltaNet) | `[B, T, 4096] → [B, T, 2048]` | `[AVAILABLE]` | `ttnn.linear` row-parallel + all-gather |
| Q+gate projection (Gated Attention) | `[B, T, 2048] → [B, T, 8192]` | `[AVAILABLE]` | Column-sharded; 2 Q heads + 2 gate vectors per device |
| KV projection (Gated Attention) | `[B, T, 2048] → [B, T, 1024]` | `[AVAILABLE]` | Replicated weights; `n_kv_h × d_h × 2 = 2 × 256 × 2 = 1024` |
| `out_proj` (Gated Attention) | `[B, T, 4096] → [B, T, 2048]` | `[AVAILABLE]` | Same structure as DeltaNet out_proj |

---

## 2. Elementwise and Activation Operations

| Operation | Shape | Status | Use |
|---|---|---|---|
| `ttnn.sigmoid` | `[B, T, n_q_h, d_h]` | `[AVAILABLE]` | Gated Attention: gate_sigmoid = σ(gate) |
| `ttnn.mul` | various | `[AVAILABLE]` | Gated Attention: Q ⊙ gate_sigmoid; DeltaNet: g · S decay |
| `ttnn.exp` | `[B, T, H_v]` | `[AVAILABLE — not yet connected]` | DeltaNet: g = exp(α); available but not wired |
| `ttnn.softplus` | `[B, T, H_v]` | `[AVAILABLE — not yet connected]` | DeltaNet: softplus(a + dt_bias); available but not wired |
| `ttnn.sub` | `[B, H_v, d_v]` | `[AVAILABLE — not yet connected]` | DeltaNet: error = v − retrieval |
| `ttnn.add` | `[B, H_v, d_k, d_v]` | `[AVAILABLE — not yet connected]` | DeltaNet: S_new = S_decayed + write |
| `ttnn.silu` | `[B, T, 4096]` | `[AVAILABLE — not yet connected]` | DeltaNet: component of FusedRMSNormSwishGate |

---

## 3. Normalization

| Operation | Weight shape | Status | Use |
|---|---|---|---|
| `ttnn.rms_norm` | `[d_h]` per head | `[AVAILABLE]` | Gated Attention: Q_norm and K_norm; weight shape `[n_q_h, d_h]` broadcast per head |
| `ttnn.rms_norm` | `[d_v]` per head | `[AVAILABLE — not yet connected]` | DeltaNet: output normalization in FusedRMSNormSwishGate; available but not wired as fused composite |
| FusedRMSNormSwishGate | `core_attn_out [B, H_v, d_v]`, `z [B, H_v, d_v]` | `[GAP — requires custom kernel]` | Composite of RMSNorm + SiLU/Swish gate; `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul` can compose it but the fused single-dispatch form does not exist |

---

## 4. Matrix Operations (DeltaNet Core)

| Operation | Shapes | Status | Notes |
|---|---|---|---|
| `g · S` — scalar decay | `g: [B, H_v]`, `S: [B, H_v, d_k, d_v]` | `[AVAILABLE — not yet connected]` | `ttnn.mul` with broadcast; g is per-head scalar |
| `S^T k̃` — state retrieval | `S: [B, H_v, d_k, d_v]`, `k̃: [B, H_v, d_k]` → `[B, H_v, d_v]` | `[AVAILABLE — not yet connected]` | `ttnn.matmul` per head; transpose S along k/v dims |
| `k̃ ⊗ error` — outer product write | `k̃: [B, H_v, d_k]`, `error: [B, H_v, d_v]` → `[B, H_v, d_k, d_v]` | `[AVAILABLE — not yet connected]` | `ttnn.matmul` with reshape: treat `k̃` as `[d_k, 1]` and `error` as `[1, d_v]` |
| `S_new^T q̃` — output retrieval | `S: [B, H_v, d_k, d_v]`, `q̃: [B, H_v, d_k]` → `[B, H_v, d_v]` | `[AVAILABLE — not yet connected]` | `ttnn.matmul` per head |
| Fused recurrent delta rule | all 6 ops above as one dispatch | `[GAP — requires custom kernel]` | The 6 operations individually expressible in TTNN; the fused form (state resident in L1 across all steps) requires a Metalium kernel |

---

## 5. Causal Conv1D

| Operation | Shapes | Status | Notes |
|---|---|---|---|
| `causal_conv1d_fn` (prefill) | input `[B, 8192, T]`, state `[B, 8192, 4]` | `[GAP — requires custom kernel]` | Sequential 1D causal convolution over time axis; `ttnn.conv` targets 2D spatial convolution; a 1D causal variant needs a custom implementation |
| `causal_conv1d_update` (decode) | input `[B, 8192, 1]`, state `[B, 8192, 4]` | `[GAP — requires custom kernel]` | Single-token conv state sliding update; simpler than prefill variant but still has no TTNN equivalent |

The conv kernel size is 4 (depthwise, no cross-channel mixing). The decode update is: drop the oldest slot, shift, insert the new input. This is expressible as elementwise shift + write, but not as a named TTNN op.

---

## 6. Chunkwise Delta Rule (Prefill)

| Operation | Status | Notes |
|---|---|---|
| `chunk_gated_delta_rule` | `[GAP — requires custom kernel]` | Full WY-decomposition chunkwise algorithm; includes inter-chunk state transfer (`S_{c+1} = g_C · S_c + U_C W_C^T`) and within-chunk QK/AV matmuls; no TTNN equivalent; currently implemented in `flash-linear-attention` (Triton) |

The within-chunk operations are individually expressible with `ttnn.matmul` and `ttnn.mul`. A Python loop over C=64 chunks calling TTNN primitives is feasible as a first-step port; a fully fused single-kernel implementation is the long-term goal.

---

## 7. Attention and CCL (Gated Attention — all available)

| Operation | Status | Notes |
|---|---|---|
| RoPE | `[AVAILABLE]` | `ttnn.experimental.rotary_embedding`; `rotary_dim = 64` for Gated Attention |
| GQA KV repeat | `[AVAILABLE]` | `ttnn.repeat_interleave(K, 8, dim=1)` to expand 2 KV heads to 16 |
| SDPA prefill | `[AVAILABLE]` | `ttnn.transformer.scaled_dot_product_attention`; Q/K/V `[1, n_q_h, T, d_h]` |
| SDPA decode | `[AVAILABLE]` | `ttnn.transformer.scaled_dot_product_attention_decode` with `cur_pos` |
| Paged SDPA decode | `[AVAILABLE]` | `ttnn.transformer.paged_scaled_dot_product_attention_decode` |
| KV cache write | `[AVAILABLE]` | `TTNNQwenPagedAttentionKVCache` manages paged cache slots |
| All-gather | `[AVAILABLE]` | `ttnn.experimental.all_gather_async`; topology=Linear for T3K 1×8 mesh |

---

## 8. Summary Count

| Status | Count |
|---|---|
| `[AVAILABLE]` (already used) | 14 |
| `[AVAILABLE — not yet connected]` | 9 |
| `[GAP — requires custom kernel]` | 4 |

The four custom-kernel gaps are: `recurrent_gated_delta_rule` (fused), `causal_conv1d_fn` (prefill), `causal_conv1d_update` (decode), and `chunk_gated_delta_rule`. `FusedRMSNormSwishGate` is an additional gap if the composite `rms_norm + silu + mul` formulation is deemed insufficiently performant without fusion.

---

**Next:** [`tt_transformers_review.md`](./tt_transformers_review.md)
