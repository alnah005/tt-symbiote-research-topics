# Output Gate Mechanism

## What the Gate Does

After computing the standard attention output but before the WO projection, Qwen3.5 multiplies the result by a sigmoid-gated linear transform of the original layer input $x$:

```math
\text{gated output} = \text{attn output} \odot \sigma\!\left(x \, W_\text{gate}\right)
```

where $\sigma$ is the element-wise sigmoid function and $W_\text{gate} \in \mathbb{R}^{d_\text{hidden} \times (n_\text{heads} \cdot \text{head dim})}$ is the post-transpose gate weight (raw checkpoint weight transposed before upload).

This gate is architecturally distinct from the Q/K gate seen in DeltaNet layers. It is a full-dimension gating of the attention output — every element of the post-softmax attention result is scaled by a learned value in $(0, 1)$ before the WO projection.

## Where the Gate Weight Comes From: `q_proj_gate` Split

In the raw HuggingFace checkpoint, the Q projection weight has shape `(n_heads * head_dim * 2, hidden_size)` — twice the expected size. The first half encodes the query, the second half encodes the gate. They are interleaved per head: for each head $h$, positions `[h * 2 * head_dim : h * 2 * head_dim + head_dim]` are query and positions `[h * 2 * head_dim + head_dim : (h+1) * 2 * head_dim]` are gate.

For the 27B model:
```
q_proj.weight shape in HF checkpoint: (12288, 5120)
  = (n_heads=24 * head_dim=256 * 2, hidden_size=5120)
```

The weight conversion in `qwen35_utils.py` (or equivalent checkpoint loading) splits this into two separate weights:
- `q_proj.weight`  — shape `(n_heads * head_dim, hidden_size)` = `(6144, 5120)`
- `q_proj_gate.weight` — shape `(n_heads * head_dim, hidden_size)` = `(6144, 5120)`

From the reference test (`test_attention_pcc.py`), the HF-side split is:

```python
q_out = x @ q_proj_w.T                         # (1, 1, 12288)
q_out = q_out.view(1, 1, n_heads, head_dim * 2)
query, gate = torch.chunk(q_out, 2, dim=-1)     # each (1, 1, 24, 256)
gate = gate.reshape(1, 1, -1)                   # (1, 1, 6144)
```

And the gate application after attention:

```python
attn_output = attn_output * torch.sigmoid(gate)
```

## Loading `gate_weight` in `GatedAttention.__init__`

`GatedAttention` looks up the split gate weight by the key `q_proj_gate.weight` under the layer's state dict prefix:

```python
layer_prefix = configuration.get_state_dict_prefix("GatedAttention", layer_num)
gate_key = f"{layer_prefix}.q_proj_gate.weight"
if gate_key in state_dict:
    gate_w = state_dict[gate_key].float().T  # transpose: (hidden_size, q_size)
    self.gate_weight = ttnn.as_tensor(
        gate_w.unsqueeze(0).unsqueeze(0),    # (1, 1, hidden_size, n_heads * head_dim)
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_name,
    )
else:
    self.gate_weight = None
```

The weight is transposed before upload so that the gate linear can be computed as `x @ gate_weight` where `x` has shape `[1, 1, batch, hidden_size]` and `gate_weight` has shape `[1, 1, hidden_size, n_heads * head_dim]`. This matches the `ttnn.linear` convention.

When `gate_weight` is `None` (e.g., running without a full checkpoint during testing), the `pre_wo_hook` is never registered and the gate is silently skipped.

## The `pre_wo_hook` Mechanism

The base `Attention` class exposes a `pre_wo_hook` attribute. If set to a callable, `Attention.forward` invokes it on the attention output — after softmax attention and before the WO projection — and uses the returned tensor in place of the original. This hook point exists specifically to support the Qwen3.5 gate without changing the base class's forward logic.

`GatedAttention.__init__` registers its gate function into this hook at construction time:

```python
if self.gate_weight is not None:
    self.pre_wo_hook = self._apply_gate
```

When the hook is invoked by `Attention.forward`, control transfers to `_apply_gate`.

## `_apply_gate`: Gate Computation

```python
def _apply_gate(self, attn_output):
    """Pre-WO hook: multiply attn_output by sigmoid(x @ gate_weight).

    attn_output shape: (1, 1, batch, n_heads * head_dim) in L1
    _gate_input shape: (1, 1, batch, hidden_size) in DRAM
    gate_weight shape: (1, 1, hidden_size, n_heads * head_dim) in DRAM
    """
    gate = ttnn.linear(
        self._gate_input,
        self.gate_weight,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gate = ttnn.sigmoid(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Ensure compatible memory configs for multiply
    attn_output = ttnn.to_memory_config(attn_output, ttnn.DRAM_MEMORY_CONFIG)
    result = ttnn.mul(attn_output, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn.deallocate(gate)
    ttnn.deallocate(self._gate_input)
    self._gate_input = None
    return result
```

All operations use `DRAM_MEMORY_CONFIG` to avoid conflicts with the L1-resident attention output tensor.

## Memory Config Handling

The attention output produced by the base `Attention` class may reside in L1 sharded memory (for performance during the softmax computation). The gate tensor is always in DRAM. `ttnn.mul` requires both operands to share a compatible memory layout.

The fix is:

```python
attn_output = ttnn.to_memory_config(attn_output, ttnn.DRAM_MEMORY_CONFIG)
```

This call moves `attn_output` to DRAM before the multiply. Note that `to_memory_config` may return the same tensor aliased if source and destination configs already match. In the current execution path, the source is L1 and the destination is DRAM, so a new buffer is always allocated.

## Why `_gate_input` Must Be Copied Before Delegation

`GatedAttention.forward` must provide the original layer input `x` to the hook. However, the base `Attention.forward` deallocates `x` after the QKV matmul — it has no reason to keep it. If `GatedAttention` simply stored a reference to `x`, the buffer would be freed by the parent before the hook fires.

The solution is to create an explicit copy:

```python
if self.gate_weight is not None:
    self._gate_input = ttnn.add(x, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

`ttnn.add(x, 0)` always allocates a new output tensor (it cannot alias when the source and destination layouts/configs differ), ensuring `_gate_input` is an independent buffer. A simpler `ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)` would not be safe here: if `x` is already in DRAM, that call may return the same underlying buffer that the parent will later free.

The copy is placed in DRAM so it does not compete with L1 resources during the attention computation that follows.

## Explicit Deallocation After Use

After applying the gate, `_apply_gate` explicitly deallocates both the intermediate gate tensor and the stored input copy:

```python
ttnn.deallocate(gate)
ttnn.deallocate(self._gate_input)
self._gate_input = None
```

This is necessary because TT-Metal tensors hold device DRAM allocations. Without explicit deallocation, these tensors remain live until Python garbage collection, which could cause DRAM fragmentation during a long decode session across many attention layers.

---

**Next:** [`forward_flow.md`](./forward_flow.md)
