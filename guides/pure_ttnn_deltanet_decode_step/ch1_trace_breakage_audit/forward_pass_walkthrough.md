# Forward Pass Walkthrough

This file traces the execution of `TTNNQwen3LinearAttention.forward` at decode time (B=1, T=1) and annotates each logical step as either on-device TTNN (trace-compatible) or host-side (trace-breaking). By the end of this file the reader will know which operations stay on the Wormhole device, which operations cross the device-host boundary, and exactly which tensors are involved in each crossing.

---

## Configuration Reference

All shapes use the Qwen3.6-35B-A3B configuration unless otherwise noted.

| Symbol | Value |
|---|---|
| B | 1 (decode batch size) |
| T | 1 (decode step) |
| H | 2048 (model hidden dimension) |
| d_k | 128 (key/query head dimension) |
| d_v | 128 (value head dimension) |
| num_v_heads | 32 (value heads per DeltaNet layer) |
| num_k_heads | 16 (key/query heads; GQA-repeated to 32 to match num_v_heads) |
| mixed_dim | 8192 (= 2 × num_k_heads × d_k + num_v_heads × d_v = 2 × 2048 + 4096) |
| value_dim | 4096 (= num_v_heads × d_v) |
| K | 4 (causal conv1d kernel size) |

Source file: `models/experimental/tt_symbiote/modules/qwen_attention.py:TTNNQwen3LinearAttention`

---

## Step 1 — Input Projections `[AVAILABLE]`

**Classification: on-device TTNN — trace-compatible**

The forward pass begins with four linear projections of the incoming hidden state `x` [B, 1, H]:

```python
mixed_qkv = ttnn.linear(x, in_proj_qkv_weight, ...)   # [B, 1, mixed_dim]
z         = ttnn.linear(x, in_proj_z_weight,   ...)   # [B, 1, value_dim]
a_t       = ttnn.linear(x, in_proj_a_weight,   ...)   # [B, 1, num_v_heads]
b_t       = ttnn.linear(x, in_proj_b_weight,   ...)   # [B, 1, num_v_heads]
```

All four calls dispatch to `ttnn.linear`, which compiles down to a `ttnn.matmul` with a fused bias add. Under T3K tensor parallelism, `in_proj_qkv` and `in_proj_z` use column-sharded weights split across 8 devices; the output activations are column-sharded and are followed by an asynchronous all-gather over the ring to restore the full [B, 1, mixed_dim] and [B, 1, value_dim] tensors on every device.

`in_proj_a` and `in_proj_b` have very small output dimensions (num_v_heads = 32 for 35B-A3B; just 32 scalars per token). These projections may use replicated weights with plain `ttnn.linear` rather than column-sharded weights because sharding 32 output features across 8 devices yields only 4 features per device, which does not tile efficiently. In either case the output tensors remain on-device.

> **Key insight:** All four projections are on-device TTNN operations. No host interaction occurs. The all-gather CCL call is also dispatched on-device and is Metal Trace-compatible. Steps 1 is fully trace-compatible as-is.

---

## Step 2 — Causal Conv1D Update `[HOST_KERNEL_LAUNCH]` `[TO_TORCH]` `[FROM_TORCH]`

**Classification: host-crossing — trace-breaking**

After the QKV projection, the `mixed_qkv` tensor [B, 1, mixed_dim] passes through a causal sliding-window convolution. At decode time this is a single-step state update (not a full sequence convolution). The update is handled by the `causal_conv1d_update` function from the `causal-conv1d` C extension library:

```python
mixed_qkv_torch = ttnn.to_torch(mixed_qkv)              # device -> host: [B, mixed_dim, 1]
conv_state_torch = ttnn.to_torch(conv_state)             # device -> host: [B, mixed_dim, K]

mixed_qkv_out, conv_state_new = causal_conv1d_update(
    mixed_qkv_torch,
    conv_state_torch,
    conv_weight,                                          # [mixed_dim, K], on CPU
    conv_bias,                                            # [mixed_dim], on CPU
)

mixed_qkv    = ttnn.from_torch(mixed_qkv_out, ...)       # host -> device: [B, mixed_dim, 1]
conv_state   = ttnn.from_torch(conv_state_new, ...)      # host -> device: [B, mixed_dim, K]
```

The mathematical operation is a depthwise convolution decode step. For a rolling buffer `conv_state` [B, channels, K] and new input `x` [B, channels, 1]:

$$\text{conv\_state}[:, :, 0{:}K-1] \leftarrow \text{conv\_state}[:, :, 1{:}K]$$

$$\text{conv\_state}[:, :, K-1] \leftarrow x$$

$$\text{output} = \sum_{i=0}^{K-1} w_i \odot \text{conv\_state}[:, :, i]$$

where $w_i$ [channels] is the learned per-channel convolution weight at position $i$.

**Tensors crossing the device-host boundary:**

| Direction | Tensor | Shape | BF16 bytes |
|---|---|---|---|
| device → host | `mixed_qkv` | [1, 8192, 1] | 16,384 |
| device → host | `conv_state` (prev) | [1, 8192, 4] | 65,536 |
| host → device | `mixed_qkv` (updated) | [1, 8192, 1] | 16,384 |
| host → device | `conv_state` (new) | [1, 8192, 4] | 65,536 |

**Trace-break mechanisms:**
- `TO_TORCH`: `ttnn.to_torch` forces a device-to-host DMA transfer and an implicit `synchronize_device` call. Metal Trace cannot insert a blocking host sync into a pre-recorded command stream.
- `HOST_KERNEL_LAUNCH`: `causal_conv1d_update` dispatches a C extension (CUDA or CPU) kernel outside of the TTNN dispatch path. The trace recorder has no knowledge of this dispatch and cannot replay it.
- `FROM_TORCH`: `ttnn.from_torch` allocates a new device buffer at runtime. Metal Trace requires that all device buffer addresses be fixed at capture time; dynamic buffer allocation inside the trace bracket breaks the static address assumption.

> **Note:** See [device_state_persistence.md](./device_state_persistence.md) for why device state persistence is necessary but not sufficient for trace safety.

---

## Step 3 — Decay Gate and Update Rate `[TO_TORCH]` or `[AVAILABLE — needs wiring]`

**Classification: currently host-crossing (conditional) — can become trace-compatible**

The decay gate $g_t$ and the update rate $\beta_t$ are small scalar tensors computed from the projected `a_t` [B, 1, num_v_heads] and `b_t` [B, 1, num_v_heads] outputs:

$$\alpha_t = -\exp(A_{\log}) \cdot \text{softplus}(a_t + \Delta_{\text{bias}})$$

$$g_t = \exp(\alpha_t) \qquad g_t \in (0, 1]^{\text{num\_v\_heads}}$$

$$\beta_t = \sigma(b_t) \qquad \beta_t \in (0, 1)^{\text{num\_v\_heads}}$$

where $A_{\log}$ and $\Delta_{\text{bias}}$ are learned weight tensors (one scalar per head, fixed after training).

**Current implementation status:** The current code applies these computations using Python-level `torch.exp`, `F.softplus`, and `torch.sigmoid` calls. This requires `a_t` and `b_t` to be moved to host via `ttnn.to_torch` before the scalar ops execute, or it requires that `a_t` and `b_t` were never placed on-device in the first place (produced as plain PyTorch tensors from a host-side linear projection).

```python
# Current pattern (host-side, trace-breaking):
a_t_torch = ttnn.to_torch(a_t)                             # [B, 1, num_v_heads]
alpha_t   = -torch.exp(A_log) * F.softplus(a_t_torch + dt_bias)
g_t       = torch.exp(alpha_t)                             # [B, 1, num_v_heads]
b_t_torch = ttnn.to_torch(b_t)                             # [B, 1, num_v_heads]
beta_t    = torch.sigmoid(b_t_torch)                       # [B, 1, num_v_heads]
```

All of these ops have direct TTNN equivalents (`ttnn.exp`, `ttnn.softplus`, `ttnn.sigmoid`, `ttnn.mul`, `ttnn.add`), making this step `[AVAILABLE — needs wiring]` once the `TO_TORCH` calls are removed.

**Tensors crossing the device-host boundary (current):**

| Direction | Tensor | Shape | BF16 bytes |
|---|---|---|---|
| device → host | `a_t` | [1, 1, 32] | 64 |
| device → host | `b_t` | [1, 1, 32] | 64 |

`g_t` and `beta_t` are computed on host from `a_t` and `b_t` and remain as host-side PyTorch tensors. They are consumed directly by the host `recurrent_gated_delta_rule` kernel in Step 4 — no `ttnn.from_torch` is called for them in Step 3.

The tensors are tiny (64 bytes each), so the raw PCIe transfer cost is negligible. However, each `ttnn.to_torch` still triggers an implicit `synchronize_device` that stalls the entire device command queue — the latency impact is dominated by synchronization overhead, not data volume.

**Trace-break mechanism:** `TO_TORCH` on `a_t` and `b_t`.

> **Key insight:** This is the lowest-effort host crossing to eliminate. All required TTNN ops exist. The fix is purely a code-wiring change with no kernel development required.

---

## Step 4 — Recurrent Gated Delta Rule Step `[HOST_KERNEL_LAUNCH]` `[TO_TORCH]` `[FROM_TORCH]`

**Classification: host-crossing — trace-breaking — primary bottleneck**

This step implements the core DeltaNet recurrence. For each head $h$ at decode step $t$:

$$S_{\text{decayed}} = g_t^{(h)} \cdot S_{t-1}^{(h)}$$

$$\text{retrieval} = (S_{t-1}^{(h)})^\top \tilde{k}_t^{(h)}$$

$$\text{error} = \beta_t^{(h)} \cdot (v_t^{(h)} - \text{retrieval})$$

$$\text{write} = \tilde{k}_t^{(h)} \otimes \text{error}$$

$$S_t^{(h)} = S_{\text{decayed}}^{(h)} + \text{write}$$

$$o_t^{(h)} = (S_t^{(h)})^\top \tilde{q}_t^{(h)}$$

where $\otimes$ denotes the outer product and $\tilde{k}_t^{(h)}, \tilde{q}_t^{(h)}$ are L2-normalized keys and queries. Across all heads: $S$ [B, num_v_heads, d_k, d_v], $\tilde{k}_t$, $\tilde{q}_t$, $v_t$ [B, 1, num_v_heads, d_k or d_v].

The current implementation calls `recurrent_gated_delta_rule` from the `flash-linear-attention` library, which dispatches a Triton CUDA kernel or falls back to pure PyTorch on CPU:

```python
# Current pattern (host-side, trace-breaking):
q_tilde_torch = ttnn.to_torch(q_tilde)      # [B, 1, num_v_heads, d_k]
k_tilde_torch = ttnn.to_torch(k_tilde)      # [B, 1, num_v_heads, d_k]
v_torch       = ttnn.to_torch(v)            # [B, 1, num_v_heads, d_v]
g_torch       = ttnn.to_torch(g_t)          # [B, 1, num_v_heads]   (or already on host from Step 3)
beta_torch    = ttnn.to_torch(beta_t)       # [B, 1, num_v_heads]   (or already on host from Step 3)
S_prev_torch  = ttnn.to_torch(S_prev)       # [B, num_v_heads, d_k, d_v]  -- largest transfer

o_t_torch, S_new_torch = recurrent_gated_delta_rule(
    q_tilde_torch, k_tilde_torch, v_torch,
    g_torch, beta_torch,
    S_prev_torch,
)

o_t   = ttnn.from_torch(o_t_torch, ...)     # [B, 1, num_v_heads, d_v]
S_new = ttnn.from_torch(S_new_torch, ...)   # [B, num_v_heads, d_k, d_v]
```

**Tensors crossing the device-host boundary:**

| Direction | Tensor | Shape | BF16 bytes |
|---|---|---|---|
| device → host | `q_tilde` | [1, 1, 32, 128] | 8,192 |
| device → host | `k_tilde` | [1, 1, 32, 128] | 8,192 |
| device → host | `v` | [1, 1, 32, 128] | 8,192 |
| device → host | `g_t` | [1, 1, 32] | 64 |
| device → host | `beta_t` | [1, 1, 32] | 64 |
| device → host | `S_prev` | [1, 32, 128, 128] | 1,048,576 |
| host → device | `o_t` | [1, 1, 32, 128] | 8,192 |
| host → device | `S_new` | [1, 32, 128, 128] | 1,048,576 |

The dominant transfer is the state matrix S: 1 MB from device to host and 1 MB back to device on every decode step, repeated for every DeltaNet layer (30 layers in the 35B-A3B model).

**Trace-break mechanisms:**
- `TO_TORCH`: six `ttnn.to_torch` calls, each forcing a device sync. The S_prev readback (1 MB) is the latency-dominant transfer.
- `HOST_KERNEL_LAUNCH`: `recurrent_gated_delta_rule` dispatches a Triton kernel (on CUDA) or PyTorch ops (on CPU), neither of which is part of the TTNN dispatch path.
- `FROM_TORCH`: two `ttnn.from_torch` calls allocating new device buffers at runtime.

> **Note:** See [device_state_persistence.md](./device_state_persistence.md) for why device state persistence is necessary but not sufficient for trace safety.

> **Key insight:** This step is the primary bottleneck. It accounts for over 2 MB of PCIe round-trip traffic per layer per decode step, plus multiple synchronization stalls. Eliminating this host crossing is the highest-priority task in the implementation roadmap.

---

## Step 5 — Gated RMSNorm `[TO_TORCH]` or `[AVAILABLE — needs wiring]`

**Classification: currently host-crossing (conditional) — can become trace-compatible**

After the recurrent step, the attention output `o_t` [B, 1, num_v_heads × d_v] = [B, 1, 4096] is passed through a fused gated RMSNorm operation:

$$x_{\text{normed}} = \text{RMSNorm}(o_t; w_{\text{norm}})$$

$$\text{gate} = \text{SiLU}(z) = z \cdot \sigma(z)$$

$$\text{output} = x_{\text{normed}} \odot \text{gate}$$

where $z$ [B, 1, value\_dim] is the gate projection from Step 1, $w_{\text{norm}}$ [value\_dim] is the learned RMSNorm weight, and $\odot$ is element-wise multiplication.

**Current implementation status:** `FusedRMSNormSwishGate` is a PyTorch `nn.Module`. If it is executed as a host-side PyTorch module (the default), it requires `o_t` and `z` to be host tensors:

```python
# Current pattern (host-side, trace-breaking if o_t / z are on-device):
o_t_torch = ttnn.to_torch(o_t)     # [B, 1, 4096]
z_torch   = ttnn.to_torch(z)       # [B, 1, 4096]

output_torch = fused_rmsnorm_swish_gate(o_t_torch, z_torch)

output = ttnn.from_torch(output_torch, ...)   # [B, 1, 4096]
```

All three underlying operations (`ttnn.rms_norm`, `ttnn.silu`, `ttnn.mul`) are available in TTNN. The fix is a pure wiring change:

```python
# Target pattern (on-device, trace-compatible):
x_normed = ttnn.rms_norm(o_t, weight=w_norm, eps=1e-6)   # [B, 1, 4096]
gate_act = ttnn.silu(z)                                   # [B, 1, 4096]
output   = ttnn.mul(x_normed, gate_act)                   # [B, 1, 4096]
```

**Tensors crossing the device-host boundary (current):**

| Direction | Tensor | Shape | BF16 bytes |
|---|---|---|---|
| device → host | `o_t` | [1, 1, 4096] | 8,192 |
| device → host | `z` | [1, 1, 4096] | 8,192 |
| host → device | `output` | [1, 1, 4096] | 8,192 |

**Trace-break mechanism:** `TO_TORCH` on `o_t` and `z`, `FROM_TORCH` on `output`.

> **Note:** If step 4's host crossing is eliminated first (i.e., `o_t` is now produced on-device by the TTNN recurrence), then step 5's host crossing becomes the next barrier between the recurrence output and the output projection. Both must be eliminated for end-to-end trace compatibility.

---

## Step 6 — Output Projection `[AVAILABLE]`

**Classification: on-device TTNN — trace-compatible**

The gated RMSNorm output [B, 1, value_dim] is projected to the model hidden dimension via a row-sharded `ttnn.linear`:

```python
hidden_out = ttnn.linear(output, out_proj_weight, ...)   # [B, 1, H]
hidden_out = ttnn.all_gather(hidden_out, ...)            # row-shard -> replicated [B, 1, H]
```

The output projection uses row-sharded weights: each device holds `out_proj_weight` [value_dim/8, H], computes its partial sum, and an all-reduce or all-gather CCL op accumulates the result. The all-gather dispatch is on-device and trace-compatible.

> **Key insight:** Step 6 is fully trace-compatible as-is. Like step 1, it requires no changes.

---

## Summary

The table below classifies each step at a glance. The two trace-breaking steps in bold are the primary targets.

| Step | Operation | On-device? | Trace-compatible? |
|---|---|---|---|
| 1 | Input projections (`ttnn.linear` × 4) + all-gather | Yes | Yes |
| 2 | Causal conv1d update (`causal_conv1d_update`) | No | No |
| 3 | Decay gate and update rate (`torch.exp`, `softplus`, `sigmoid`) | No (currently) | No (currently) |
| **4** | **Recurrent gated delta rule (`recurrent_gated_delta_rule`)** | **No** | **No** |
| 5 | Gated RMSNorm (`FusedRMSNormSwishGate`) | No (currently) | No (currently) |
| 6 | Output projection (`ttnn.linear`) + all-gather | Yes | Yes |

The complete trace-break analysis for steps 2–5, including exact source lines and tensor sizes, is in `host_crossing_summary_table.md`.

---

**Next:** [`host_crossing_summary_table.md`](./host_crossing_summary_table.md)

---

