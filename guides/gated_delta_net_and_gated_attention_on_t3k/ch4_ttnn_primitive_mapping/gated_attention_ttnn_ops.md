# Gated Attention — TTNN Ops (Prefill and Decode)

This section catalogs every operation in the Gated Attention forward pass and maps each to its TTNN primitive. The central finding is that Gated Attention requires **no new kernel development**: every operation in the forward pass — including the gating mechanism, normalization, RoPE, paged KV cache, GQA repeat, and scaled dot-product attention — is covered by existing TTNN primitives.

Symbols: H=2048 (model dimension), d_h=256 (attention head dimension), n_q_h=16 (query heads), n_kv_h=2 (KV heads), B=batch, T=sequence length. GQA repeat factor: n_q_h / n_kv_h = 16 / 2 = 8.

---

## Input Projections

`[AVAILABLE]` — col-sharded across 8 devices

Gated Attention projects the input `x` `[B, T, 2048]` into Q, a Q-gate, K, and V. The Q and gate are fused into a single large projection; K and V are fused together.

**Q + gate projection:**
- Output dims: Q contributes n_q_h × d_h = 16 × 256 = 4096; gate contributes the same 4096.
- Total output: 4096 + 4096 = 8192.
- Weight: `[8192, 2048]`.
- Operation: `ttnn.linear(x, weight_qg, bias=None, core_grid=...)`.
- Output: Q_raw + gate `[B, T, 8192]`, col-sharded (each device: `[B, T, 1024]`).

**KV projection:**
- Output dims: K contributes n_kv_h × d_h = 2 × 256 = 512; V contributes 512. Total: 1024.
- Weight: `[1024, 2048]`.
- Operation: `ttnn.linear(x, weight_kv, bias=None, core_grid=...)`.
- Output: KV `[B, T, 1024]`, col-sharded (each device: `[B, T, 128]`).

---

## Gate Sigmoid

`[AVAILABLE]`

The gate tensor (second half of the Q+gate projection output) is passed through sigmoid to produce values in (0, 1).

- Input: gate `[B, T, 4096]` (split from the `[B, T, 8192]` projection output).
- Operation: `ttnn.sigmoid(gate)`.
- Output: gate_sigmoid `[B, T, 4096]` → reshape → `[B, T, n_q_h, d_h]` = `[B, T, 16, 256]`.

---

## Q Gating

`[AVAILABLE]`

The raw Q is multiplied element-wise with the sigmoid gate to produce a gated query tensor.

- Input Q_raw: `[B, T, 4096]` → reshape → `[B, T, 16, 256]`.
- Input gate_sigmoid: `[B, T, 16, 256]`.
- Operation: `ttnn.mul(Q_raw, gate_sigmoid)`.
- Output: Q_gated `[B, T, 16, 256]`.

Shape check: n_q_h × d_h = 16 × 256 = 4096 elements per (B, T) position — consistent with the projection output.

---

## Q Normalization

`[AVAILABLE]`

Each query head is L2-normalized via RMSNorm. The normalization is applied per head, so the norm weight has shape `[d_h]` = `[256]`.

- Input: Q_gated `[B, T, 16, 256]`.
- Operation: `ttnn.rms_norm(Q_gated, weight=q_norm_weight, epsilon=1e-6)`, applied over the last dimension (d_h=256) independently per head.
- Output: Q_normed `[B, T, 16, 256]`.

---

## K Normalization

`[AVAILABLE]`

Key heads are normalized in the same way before being written to the KV cache.

- Input: K `[B, T, n_kv_h, d_h]` = `[B, T, 2, 256]` (split from KV projection).
- Operation: `ttnn.rms_norm(K, weight=k_norm_weight, epsilon=1e-6)`, over d_h=256.
- Output: K_normed `[B, T, 2, 256]`.

---

## RoPE (Rotary Positional Embedding)

`[AVAILABLE]`

Rotary embeddings are applied to the first 64 dimensions of each Q and K head. The remaining d_h − 64 = 192 dimensions are left unchanged (a partial-RoPE variant used in some models to preserve non-positional head capacity).

- Input Q_normed: `[B, T, 16, 256]`.
- Input K_normed: `[B, T, 2, 256]`.
- Operation: `ttnn.experimental.rotary_embedding(tensor, cos_cached, sin_cached, token_idx)`, applied to `tensor[..., :64]`.
- cos/sin caches: precomputed `[1, 1, T_max, 64]` for the rope_dim=64 rotation.
- Outputs: Q_rope `[B, T, 16, 256]`, K_rope `[B, T, 2, 256]`.

---

## KV Cache Write

`[AVAILABLE]`

K and V are written into a paged KV cache. The cache management is handled by `TTNNQwenPagedAttentionKVCache`, which wraps TTNN page-table operations.

- K_rope `[B, T, 2, 256]` and V `[B, T, 2, 256]` are written to the appropriate cache pages based on the current sequence position and batch page table.
- At decode time (T=1), a single slot is updated. At prefill time, T consecutive slots are written.
- The paged cache layout allows variable-length sequences and in-place updates without reallocation.

---

## GQA KV Repeat

`[AVAILABLE]`

To support grouped query attention (n_q_h=16 query heads against n_kv_h=2 KV heads), K and V are repeated 8× along the head dimension before the attention computation.

- Input K: `[B, T, 2, 256]`.
- Operation: `ttnn.repeat_interleave(K, repeats=8, dim=2)`.
- Output K_expanded: `[B, T, 16, 256]`.

Shape check: n_kv_h × repeat = 2 × 8 = 16 = n_q_h. Each KV head is replicated to serve 8 query heads.

- Same operation applied to V: V_expanded `[B, T, 16, 256]`.

---

## SDPA — Prefill

`[AVAILABLE]`

For prefill (T > 1), scaled dot-product attention is computed over the full query sequence against the accumulated KV cache using the Flash Attention algorithm.

- Inputs after transposing to TTNN's expected layout:
  - Q: `[B, n_q_h, T, d_h]` = `[B, 16, T, 256]`.
  - K: `[B, n_q_h, T, d_h]` = `[B, 16, T, 256]` (after GQA expand).
  - V: `[B, n_q_h, T, d_h]` = `[B, 16, T, 256]`.

**Important — materialized GQA expansion:** `ttnn.transformer.scaled_dot_product_attention` receives the **materialized 8× expanded** K/V tensors `[B, 16, T, 256]`. The `ttnn.repeat_interleave` call from the GQA KV Repeat section above is performed first; the SDPA op itself does not handle the GQA grouping internally and does not accept a GQA group count argument. This means the forward pass incurs an 8× memory cost for K/V during the SDPA computation (the KV cache itself stores only the 2 unexpanded heads — the expansion is a transient compute-time allocation).

- Scale: $1/\sqrt{d_h} = 1/\sqrt{256} = 1/16$.
- Operation: `ttnn.transformer.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale, program_config=SDPAProgramConfig(...))`.
- `SDPAProgramConfig` specifies the block size and compute grid for the Flash Attention tiling on Wormhole cores.
- Output: attn_out `[B, 16, T, 256]`.

---

## SDPA — Decode

`[AVAILABLE]`

For decode (T=1), the single query token attends over all cached K/V positions up to `cur_pos`.

- Q: `[B, n_q_h, 1, d_h]` = `[B, 16, 1, 256]`.
- K_cache (read from cache): `[B, n_kv_h, S, d_h]` = `[B, 2, S, 256]` — the cache stores only the 2 unexpanded KV heads.
- V_cache (read from cache): `[B, n_kv_h, S, d_h]` = `[B, 2, S, 256]`.

The same GQA expansion applies: read K/V from cache at `[B, 2, S, 256]`, expand to `[B, 16, S, 256]` via `ttnn.repeat_interleave` (see Prefill section above for details).

- Operation: `ttnn.transformer.scaled_dot_product_attention_decode(Q, K_exp, V_exp, cur_pos=cur_pos, scale=scale, program_config=SDPADecodeConfig(...))`.
- Output: attn_out `[B, 16, 1, 256]`.

The decode SDPA path is optimized for the T=1 case and avoids materializing the full attention matrix.

---

## Output Reshape + Projection

`[AVAILABLE]`

The multi-head attention output is reshaped from head layout back to model dimension, then projected.

**Reshape:**
- attn_out `[B, n_q_h, T, d_h]` = `[B, 16, T, 256]` → transpose → `[B, T, 16, 256]` → reshape → `[B, T, 4096]`.
- Total output elements per (B, T): n_q_h × d_h = 16 × 256 = 4096.

**Output projection:**
- Weight: `[2048, 4096]` (out=H=2048, in=4096), row-sharded across 8 devices.
- Operation: `ttnn.linear(attn_out_flat, weight_out, bias=None, core_grid=...)`.
- Output: `[B, T, 256]` per device (row shard; 2048/8 = 256 dims per device).

---

## All-Gather

`[AVAILABLE]`

The row-sharded output projection is gathered across all 8 devices to restore the full `[B, T, 2048]` output.

- Operation: `ttnn.experimental.all_gather_async(out_shard, dim=-1, num_links=1, topology=ttnn.Topology.Ring)`.
- Output: `[B, T, 2048]` replicated across all 8 devices.

---

## Key Finding

Gated Attention is **substantially TTNN-accelerated** across both prefill and decode. Every operation — gating, normalization, RoPE, KV cache management, GQA repeat, scaled dot-product attention, and output projection — maps directly to an existing TTNN primitive. There is **no PyTorch fallback** in the core attention forward path. No new kernel development is required to run Gated Attention on T3K.

This stands in sharp contrast with Gated Delta Net, where four distinct kernel gaps require custom Metalium kernel development before the recurrent core can be TTNN-accelerated (see `kernel_gap_summary.md`).

---

**Next:** [`kernel_gap_summary.md`](./kernel_gap_summary.md)
