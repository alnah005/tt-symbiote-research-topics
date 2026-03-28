# Gated Delta Net — Decode Step (B=1, T=1)

This section traces every operation in a single Gated Delta Net decode step, annotating each with its tensor shape and TTNN status. Decode is the latency-critical path: the sequence length is 1, the recurrent state `S` carries all context, and there is no parallelism over the time dimension.

Symbols used throughout: H=2048 (model dimension), d_k=128 (key/query head dimension), d_v=128 (value head dimension), H_k=16 (number of key heads), H_v=32 (number of value heads — twice H_k due to the GDN head expansion), B=1, T=1.

---

## Step 1 — Input Projections

`[AVAILABLE]` — col-sharded across 8 devices

The input hidden state `x` has shape `[1, 1, 2048]`. Four linear projections fan out from this input.

**`in_proj_qkv`** — fused Q, K, V projection:
- Weight shape: `[8192, 2048]` (out=8192, in=2048).
  - Breakdown: Q contributes H_k × d_k = 16 × 128 = 2048 dims; K contributes H_k × d_k = 2048 dims; V contributes H_v × d_v = 32 × 128 = 4096 dims. Total: 2048 + 2048 + 4096 = 8192.
- Operation: `ttnn.linear(x, weight_qkv, bias=None, core_grid=...)`.
- Output: `[1, 1, 8192]`, col-sharded across 8 devices (each device holds `[1, 1, 1024]`).

**`in_proj_z`** — gating projection fed into the output RMSNorm gate:
- Weight shape: `[4096, 2048]` (H_v × d_v = 32 × 128 = 4096 output dims).
- Operation: `ttnn.linear(x, weight_z, bias=None, core_grid=...)`.
- Output: `[1, 1, 4096]`, col-sharded (each device holds `[1, 1, 512]`).

**`in_proj_a`** — log-decay projection, one scalar per key-head group:
- Weight shape: `[4, 2048]` (output size equals the number of GDN groups, H_k/4 = 4).
- `[GAP — requires wiring]`: output is too small to benefit from col-sharding in practice; currently dispatched as PyTorch `nn.Linear`. Recommended fix: run as a replicated `ttnn.linear` on a single device with no sharding.
- Output: `[1, 1, 4]`.

**`in_proj_b`** — β (write-gate strength) projection:
- Weight shape: `[4, 2048]`.
- `[GAP — requires wiring]`: same situation as `in_proj_a`.
- Output: `[1, 1, 4]`.

---

## Step 2 — All-Gather

`[AVAILABLE]`

The sharded outputs of `in_proj_qkv` and `in_proj_z` must be gathered before the conv1d and recurrent steps, which operate over the full channel dimension on each device.

- Operation: `ttnn.experimental.all_gather_async(tensor, dim=-1, num_links=1, topology=ttnn.Topology.Ring)`.
- Input: `[1, 1, 1024]` per device (for qkv), `[1, 1, 512]` per device (for z).
- Output: `[1, 1, 8192]` replicated; `[1, 1, 4096]` replicated.

---

## Step 3 — Causal Conv1d Update

`[GAP — requires custom kernel]`

The conv1d maintains a rolling 4-element history over the **time (sequence) axis** (dim=2), not the channel axis. The conv state shape is `[1, 8192, 4]` where 8192 is the channel dimension (dim=1) and 4 is the time/sequence axis (dim=2, equal to `kernel_size`). On each decode step, the oldest time slot (dim=2, index 0) is evicted and the new token's `mixed_qkv` is inserted at the latest time slot (dim=2, index 3). No sliding occurs along the channel axis. At decode time this is a single-step state update rather than a full convolution.

- Input `mixed_qkv` tensor shape (reshaped for conv): `[1, 8192, 1]` (batch, channels, time).
- Input conv state shape: `[1, 8192, 4]` (batch, channels, time axis of length kernel_size=4).
- Operation: `causal_conv1d_update(mixed_qkv, conv_state, weight, bias)` — PyTorch C extension (from the `causal-conv1d` package).
- Output `mixed_qkv`: `[1, 8192, 1]` (convolved output for this step).
- Output conv state: `[1, 8192, 4]` (time axis shifted: new token appended at dim=2 index 3, oldest time slot at dim=2 index 0 dropped).

Currently there is no TTNN primitive for a stateful sliding-window 1D convolution update. Porting requires either a custom Metalium kernel or expressing the update as a batched dot product (`ttnn.matmul` of the 4-element window against the kernel weights) — the latter is functionally correct but incurs per-step launch overhead for a very small compute tile.

---

## Step 4 — Split, Reshape, and Gate Scalars

After the conv1d output `mixed_qkv` `[1, 1, 8192]` is reshaped and split into Q, K, V and then gating scalars α, β, g are computed.

**Tensor split:**
- Q: `[1, 1, 8192]` slice `[..., :2048]` → reshape → `[1, 1, 16, 128]`.
- K: `[1, 1, 8192]` slice `[..., 2048:4096]` → reshape → `[1, 1, 16, 128]`.
- V: `[1, 1, 8192]` slice `[..., 4096:]` → reshape → `[1, 1, 32, 128]`.

**β (write-gate strength)** — `[GAP — requires wiring]`:
- Input: `b` (output of `in_proj_b`), shape `[1, 1, 4]`.
- Operation: `ttnn.sigmoid(b)` → β `[1, 1, 4]`.
- One β scalar per group (4 groups, each covering 8 value heads).
- `ttnn.sigmoid` is available; the gap is wiring the tensor from the PyTorch `in_proj_b` path into the TTNN graph.

**α (log-decay scalar)** — `[GAP — requires wiring]`:
- Input: `A_log` (learned parameter, shape `[4]`) and `a` (output of `in_proj_a`, shape `[1, 1, 4]`).
- `dt_bias`: learned parameter, shape `[4]`.
- Computation: `α = −exp(A_log) · softplus(a + dt_bias)`.
- TTNN primitives: `ttnn.exp`, `ttnn.softplus`, `ttnn.mul`, `ttnn.add` — all available individually.
- Gap: wiring the mixed PyTorch-tensor parameters into a TTNN tensor graph.

**g (state decay factor)** — `[GAP — requires wiring]`:
- Input: α `[1, 1, 4]`.
- Operation: `g = exp(α)` via `ttnn.exp`.
- Output: g `[1, 1, 4]`, one per group, broadcast to each group's 8 heads in Step 5.

**K/Q head repeat (GDN head expansion):**
- K `[1, 1, 16, 128]` and Q `[1, 1, 16, 128]` are each repeated interleave ×2 along the head dim to match H_v=32.
- K̃: `[1, 1, 32, 128]`; Q̃: `[1, 1, 32, 128]`.
- TTNN: `ttnn.repeat_interleave(K, repeats=2, dim=2)` — `[AVAILABLE]`.

---

## Step 5 — Recurrent Delta Rule Step

`[GAP — requires custom kernel]`

This is the core state update of the Gated Delta Net. It is currently dispatched as a single PyTorch call to `recurrent_gated_delta_rule`.

**Inputs:**
- Q̃: `[1, 1, 32, 128]`
- K̃: `[1, 1, 32, 128]`
- V: `[1, 1, 32, 128]`
- g: `[1, 1, 4]` (one per group; 4 groups × 8 heads = 32 heads)
- β: `[1, 1, 4]`
- S_prev: `[1, 32, 128, 128]` (recurrent state; d_k × d_v per head)

**TTNN decomposition** (not fused — each dispatched separately):

1. **State decay** — `S_decayed = g · S_prev`:
   - g has shape `[1, 1, 4]` (4 group scalars). The 4 groups map to consecutive blocks of 8 heads along the head axis (axis 1): group 0 → heads 0–7, group 1 → heads 8–15, group 2 → heads 16–23, group 3 → heads 24–31. To broadcast correctly, g must first be reshaped to `[1, 4, 1, 1]` and then expanded along axis 1 with `repeat_interleave(repeats=8, dim=1)` to produce `[1, 32, 1, 1]`. This `[1, 32, 1, 1]` tensor then broadcasts element-wise against S_prev `[1, 32, 128, 128]`.
   - `ttnn.mul(g_broadcast, S_prev)` → S_decayed `[1, 32, 128, 128]`. `[AVAILABLE]`

2. **Retrieval** — `retrieval = S_decayed^T @ k̃`:
   - S_decayed `[1, 32, 128, 128]` transposed to `[1, 32, 128, 128]` (swap last two dims → `[d_v, d_k]` per head).
   - k̃ reshaped to `[1, 32, 128, 1]`.
   - `ttnn.matmul(S_decayed_T, k̃)` → retrieval `[1, 32, 128, 1]` → `[d_v]` per head. `[AVAILABLE]`

3. **Error** — `error = β · (v − retrieval)`:
   - `ttnn.sub(V, retrieval)` → `[1, 32, 128, 1]`. `[AVAILABLE]`
   - Reshape and expand β identically to g (see sub-step 1): `[1, 1, 4]` → `[1, 32, 1, 1]`. Then `error = ttnn.mul(β_expanded, sub_result)` where `sub_result = V − retrieval ∈ R^{d_v}` per head (the output of the preceding `ttnn.sub` call).
   - `ttnn.mul(beta_broadcast, sub_result)` → error `[1, 32, 128, 1]`. `[AVAILABLE]`

4. **Write** — `write = k̃ ⊗ error^T` (outer product, `[d_k, d_v]` per head):
   - k̃ `[1, 32, 128, 1]`, error^T `[1, 32, 1, 128]`.
   - `ttnn.matmul(k̃, error^T)` → write `[1, 32, 128, 128]`. `[AVAILABLE]`

5. **State update** — `S_new = S_decayed + write`:
   - `ttnn.add(S_decayed, write)` → S_new `[1, 32, 128, 128]`. `[AVAILABLE]`

6. **Output** — `o = S_new^T @ q̃`:
   - S_new^T `[1, 32, 128, 128]` (swap last two dims).
   - q̃ `[1, 32, 128, 1]`.
   - `ttnn.matmul(S_new_T, q̃)` → o `[1, 32, 128, 1]` → core_attn_out `[1, 1, 32, 128]`. `[AVAILABLE]`

While each of the 6 sub-operations is individually expressible in TTNN, dispatching them separately incurs kernel-launch overhead for very small tensors (`[1, 32, 128, 128]` state). A fused Metalium kernel that executes the full recurrence in a single dispatch is the recommended path for decode latency (see `kernel_gap_summary.md`).

**Outputs:**
- core_attn_out: `[1, 1, 32, 128]`
- S_new: `[1, 32, 128, 128]` (written back to `recurrent_states[layer_idx]`)

---

## Step 6 — Gated RMSNorm

`[GAP — requires custom kernel]`

After the recurrent step, the output is normalized and gated with the `z` projection from Step 1.

**Inputs:**
- core_attn_out: `[1, 32, 128]` (reshaped from `[1, 1, 32, 128]` — batch and time dims collapsed).
- z: `[1, 32, 128]` (from `in_proj_z`, reshaped to match head layout).

**Current implementation:** PyTorch `FusedRMSNormSwishGate` (a fused CUDA kernel from the `fla` library).

**Gating function — SiLU / Swish (confirmed):** The "Swish" in `FusedRMSNormSwishGate` names the activation function. Qwen3.5's implementation uses **SiLU** (also called Swish): `gate(z) = z * sigmoid(z)`. This is distinct from plain sigmoid — `sigmoid(z)` returns a value in (0, 1), while `silu(z) = z * sigmoid(z)` is unbounded above 0 for positive z. Using `ttnn.sigmoid` instead of `ttnn.silu` here would produce a numerically wrong result. The correct TTNN decomposition is therefore:

**TTNN composable equivalent:**
1. `ttnn.rms_norm(core_attn_out, weight=norm_weight)` → normed `[1, 32, 128]`. `[AVAILABLE]`
2. `ttnn.silu(z)` → z_silu `[1, 32, 128]`. Computes `z * sigmoid(z)` per element. `[AVAILABLE]`
3. `ttnn.mul(normed, z_silu)` → gated_out `[1, 32, 128]`. `[AVAILABLE]`

Full expression: `gated_out = rms_norm(core_attn_out) * silu(z)` = `rms_norm(core_attn_out) * (z * sigmoid(z))`.

The gap is that these three calls are not fused: the fused CUDA version reads `core_attn_out` and `z` once and writes once, while the TTNN decomposition performs three separate read-write passes. For decode (where tensors are small), a custom fused kernel is preferred but the composable version is functionally sufficient as an initial port.

**Output:** gated_out `[1, 32, 128]` → reshaped to `[1, 1, 4096]` for projection.

---

## Step 7 — Output Projection

`[AVAILABLE]`

The gated output is projected back to the model dimension and all-gathered to restore a replicated output tensor.

**`out_proj`:**
- Input: `[1, 1, 4096]` (H_v × d_v = 32 × 128 = 4096).
- Weight shape: `[2048, 4096]` (out=H=2048, in=4096), row-sharded across 8 devices.
- Operation: `ttnn.linear(gated_out, weight_out, bias=None, core_grid=...)`.
- Output: `[1, 1, 256]` per device (each device holds a row shard, `2048/8 = 256` dims).

**All-gather to restore full output:**
- Operation: `ttnn.experimental.all_gather_async(out_shard, dim=-1, num_links=1, topology=ttnn.Topology.Ring)`.
- Output: `[1, 1, 2048]` replicated across all 8 devices.

---

**Next:** [`gated_delta_net_prefill_pass.md`](./gated_delta_net_prefill_pass.md)
