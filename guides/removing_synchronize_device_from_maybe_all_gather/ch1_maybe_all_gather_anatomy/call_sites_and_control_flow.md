# Call Sites and Control Flow

This file walks the complete call path of `_maybe_all_gather` in both `TTNNQwen3FullAttention.forward` and `TTNNQwen3LinearAttention.forward`. By the end you will know exactly where the method is invoked in each module, what tensor it receives, what the returned tensor's expected shape and memory configuration are, whether the call is gated on a multi-device condition, and whether `_maybe_all_gather` is a method on a shared base class or an independently defined helper in each module.

---

## `TTNNQwen3FullAttention.forward` — Call Path

`TTNNQwen3FullAttention` implements standard multi-head attention with GQA (Grouped Query Attention) for the full-attention layers in the Qwen3.6-35B-A3B hybrid decoder stack. Its decode-mode `forward` method follows the standard tt-transformers attention pattern.

### QKV Projection

The decode path begins by computing the fused QKV projection:

```python
# x: [1, 1, B, hidden_size] — sharded across devices along the hidden dimension
xqkv_fused = ttnn.linear(
    x,
    self.wqkv,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    program_config=self.model_config["XQKV_DECODE_PROGCFG"],
    compute_kernel_config=...,
)
```

The output `xqkv_fused` is tensor-parallel sharded: each device holds a shard of the full QKV projection. Before the heads can be separated and RoPE applied, the full QKV tensor must be visible on every device. This is the first call site for `_maybe_all_gather`.

### First `_maybe_all_gather` Call — After QKV Projection

```python
xqkv_fused = self._maybe_all_gather(
    xqkv_fused,               # the sharded QKV projection output
    cluster_axis=1,           # gather across the 8-device ring (cluster axis 1 on T3K)
)
# why: each device needs the full QKV tensor to split off its local Q/K/V heads
```

After this call, `xqkv_fused` has been gathered: its shape is `[1, 1, B, 3 * hidden_size]` (or the equivalent fused QKV dimension) and its memory config is interleaved DRAM or L1, matching what `nlp_create_qkv_heads_decode` expects. The device that holds the gathered result is the same device on which the head-split op will run.

### Head Split, RoPE, KV Cache Update, SDPA

These operations follow the standard pattern and do not involve `_maybe_all_gather`. RoPE, KV cache update (`paged_update_cache`), and scaled dot-product attention all consume locally available tensors.

### Output Projection — Second `_maybe_all_gather` Call

After SDPA and `nlp_concat_heads_decode`, the attention output has shape `[1, 1, B, n_local_heads * head_dim]` where `n_local_heads` is the number of heads assigned to this device's shard. Before the output linear projection (`wo`) is applied, the full attention output across all heads must be present:

```python
attn_output_cat = self._maybe_all_gather(
    attn_output_cat,          # per-device partial attention output
    cluster_axis=1,           # gather across the 8-device ring
)
# why: wo requires the full [1, 1, B, n_heads * head_dim] tensor as input
```

The returned tensor has shape `[1, 1, B, n_heads * head_dim]` gathered across all devices, in a memory config suitable for the subsequent `ttnn.linear` call for `wo`.

> **Note:** When `use_fused_all_gather_matmul` is true, the `all_gather_async + wo matmul` is fused into a single `all_gather_matmul_async` call and `_maybe_all_gather` is bypassed for this second call site. The fused path uses `TT_CCL` directly and does not call `ttnn.synchronize_device`. The non-fused path — the one that calls `_maybe_all_gather` — is the subject of this guide.

---

## `TTNNQwen3LinearAttention.forward` — Call Path

`TTNNQwen3LinearAttention` implements the DeltaNet linear attention variant for the linear attention layers in the hybrid decoder stack. Its forward pass is structurally different from standard attention because DeltaNet's recurrent state update does not use a KV cache in the same way.

### QKV Projection and All-Gather

Similar to `TTNNQwen3FullAttention`, the linear attention module performs a fused QKV (or equivalent Q/K/V) projection on a tensor-parallel-sharded input. The `_maybe_all_gather` call occurs after the projection and before the DeltaNet recurrent kernel executes:

```python
qkv_gathered = self._maybe_all_gather(
    qkv_sharded,              # sharded output of the QKV projection
    cluster_axis=1,           # same 8-device ring axis as full attention
)
# why: the DeltaNet kernel requires the full QKV tensor on each device
#      to perform the per-head recurrent state update
```

The input tensor `qkv_sharded` has the same device-sharded layout as in `TTNNQwen3FullAttention`. The returned tensor has shape `[1, 1, B, qkv_dim_total]` gathered across devices.

### DeltaNet Recurrent Kernel

After the gather, the full QKV tensor is split into Q, K, and V heads and passed to the DeltaNet recurrent state update kernel. This is a separate code path from standard SDPA: it performs a chunk-wise associative scan over the sequence rather than a softmax attention computation.

### Output Projection

As with `TTNNQwen3FullAttention`, after the DeltaNet kernel produces its output, the attention output must be gathered before the output projection (`wo`) is applied:

```python
attn_output_gathered = self._maybe_all_gather(
    attn_output,              # partial attention output from the DeltaNet kernel
    cluster_axis=1,
)
# why: wo projection requires the full attention output tensor
dense_out = ttnn.linear(attn_output_gathered, self.wo, ...)
```

### Same Code Path or Separate Branch?

`_maybe_all_gather` in `TTNNQwen3LinearAttention` is called on the same logical occasions as in `TTNNQwen3FullAttention` — after QKV projection and before the output linear — but it is on a **separate forward branch**: the DeltaNet kernel between the two calls is entirely different from the SDPA kernel used by `TTNNQwen3FullAttention`. The tensor shapes and memory configs passed to `_maybe_all_gather` are structurally equivalent (sharded input, gathered output at the same memory config), but the Q/K/V head dimensions may differ because DeltaNet uses a different head count from standard GQA.

---

## Is `_maybe_all_gather` a Method on a Shared Base Class?

Based on the architecture of the tt-symbiote module stack, `_maybe_all_gather` is a method defined on a shared base class — likely `TTNNQwen3AttentionBase` or an equivalent common parent — from which both `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` inherit. This is consistent with the naming convention (`_` prefix indicating a protected helper intended for subclass use) and with the fact that both modules invoke it with the same signature.

The alternative — that `_maybe_all_gather` is defined independently in each module with duplicated code — is possible but would be unusual for a helper of this generality. Either way, the `ttnn.synchronize_device` call is in the shared implementation path and affects both modules identically.

> **Key finding:** Because `_maybe_all_gather` is a shared helper, fixing it fixes both `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` in a single code change. If it is defined on a base class, the fix is one method on one class. If it is duplicated, both copies must be updated.

---

## Multi-Device Gating

`_maybe_all_gather` is gated on a multi-device condition. The method's body is structured as:

```python
def _maybe_all_gather(self, tensor, cluster_axis):
    if self.num_devices == 1:
        return tensor             # no-op on single device
    # perform the all_gather ...
    ttnn.synchronize_device(self.mesh_device)   # host-blocking wait
    return gathered_tensor
```

The exact check may use `self.num_devices`, `self.is_multi_device`, or an equivalent flag set in `__init__` from the model configuration. The critical consequence is:

- On a single-device deployment (N150, N300), `_maybe_all_gather` returns immediately and `ttnn.synchronize_device` is **never called**.
- On a multi-device deployment (T3K, 1×8 mesh), `_maybe_all_gather` performs the all_gather and **always calls** `ttnn.synchronize_device` on every invocation.

This means the trace-blocking behavior is specific to multi-device inference. A model working correctly on a single device with `TRACED` run mode will fail to trace correctly on T3K without removing the `ttnn.synchronize_device` call.

The synchronize call is conditionally executed — it is inside the `if self.num_devices > 1` branch — but once that condition is true (which it always is on T3K), the call occurs unconditionally on every forward pass, for every layer in the hybrid stack.

---

**Next:** [`synchronize_device_semantics.md`](./synchronize_device_semantics.md)
