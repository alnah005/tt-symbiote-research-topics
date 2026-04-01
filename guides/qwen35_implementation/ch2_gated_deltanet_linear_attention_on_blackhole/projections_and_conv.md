# Projections and Conv1d: Device-Side Input Preparation

## Overview

Before the recurrence described in [`recurrence_math.md`](./recurrence_math.md) can execute,
the raw hidden-state vector $\mathbf{x}$ must be projected into Q, K, V, gate ($z$), and
control ($b$, $a$) channels. A causal conv1d filter is then applied to the Q/K/V slice. All
of this happens on device with zero host synchronization — the recurrence itself is the only
step that currently requires a host roundtrip.

---

## Fused Input Projection: `in_proj_all`

A naive implementation would dispatch four separate `ttnn.linear` calls for the four projection
weights: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`. Instead, the constructor fuses
them into a single weight tensor by concatenating along the output dimension:

```python
w_all = torch.cat(
    [
        load_weight("in_proj_qkv"),   # shape: [hidden_size, conv_dim]
        load_weight("in_proj_z"),     # shape: [hidden_size, value_dim]
        load_weight("in_proj_b"),     # shape: [hidden_size, num_v_heads]
        load_weight("in_proj_a"),     # shape: [hidden_size, num_v_heads]
    ],
    dim=-1,  # concatenate along output dimension (after transpose)
)
self._proj_splits = [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads]
self.in_proj_all = ttnn.as_tensor(
    w_all.unsqueeze(0).unsqueeze(0),
    dtype=proj_dtype,
    device=mesh_device,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    cache_file_name=cache_name("in_proj_all_fused"),
)
```

The four outputs are then recovered by slicing the single result:

```
proj shape: [1, 1, B_pad, conv_dim + value_dim + num_v_heads + num_v_heads]
           = [1, 1, 32,   10240 + 6144 + 48 + 48]   (27B example)
```

This reduces four kernel dispatches to one, saving Python overhead and device-side launch
cost. The `_proj_splits` list records where each slice begins so forward can recover the
individual outputs efficiently.

### Split Sizes (Qwen3.5-27B)

| Slice | Width | Content |
|-------|-------|---------|
| `proj[:conv_dim]` | 10240 | QKV for conv1d (Q=2048, K=2048, V=6144) |
| `proj[conv_dim : conv_dim + value_dim]` | 6144 | $z$ gate for RMSNorm |
| `proj[...+value_dim : ...+value_dim+num_v_heads]` | 48 | $b$ (beta logits) |
| `proj[...+num_v_heads : end]` | 48 | $a$ (decay logits) |

### Split Sizes (Qwen3.5-35B-A3B)

| Slice | Width | Content |
|-------|-------|---------|
| `proj[:conv_dim]` | 8192 | QKV (Q=2048, K=2048, V=4096) |
| `proj[...+value_dim]` | 4096 | $z$ gate |
| `proj[...+num_v_heads]` | 32 | $b$ (beta logits) |
| `proj[...+num_v_heads]` | 32 | $a$ (decay logits) |

The conv dimension is $\text{conv dim} = 2 \times \text{key dim} + \text{value dim}$,
where $\text{key dim} = \text{head k dim} \times \text{num k heads}$.

---

## Causal Conv1d: 4-Slot Circular Ring Buffer

Qwen3.5 applies a depthwise causal conv1d (kernel size 4) to the Q, K, V slice before feeding
it to the recurrence. This introduces a short-range local context window — each token sees a
weighted mixture of its own and the 3 preceding token projections. In decode mode (one token at
a time) the standard `F.conv1d` cannot be used directly; a state buffer must be maintained.

### Why a Ring Buffer

The conv1d state is a history of the last `conv_kernel_size - 1 = 3` Q/K/V vectors, padded to
`conv_kernel_size = 4` slots for uniform weight application. A ring buffer of 4 device tensors
is used so that:

1. The in-place `ttnn.copy` pattern preserves tensor addresses across token steps. This is
   essential for Metal Trace compatibility — Trace captures a fixed graph of ops and their
   tensor addresses; if a new tensor were allocated on every step the captured graph would
   be invalidated.
2. No host synchronization is needed: the entire conv state lives in device DRAM.

### Ring Buffer Layout

Four device tensors are allocated in `initialize_states`:

```python
self._conv_rows = []
for _ in range(self.conv_kernel_size):       # 4 slots
    self._conv_rows.append(
        ttnn.from_torch(
            torch.zeros(1, 1, B_pad, self.conv_dim, dtype=torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    )
self._oldest = 0
```

Each slot has shape [1, 1, B_pad, conv_dim] in bfloat16. `B_pad` is `tile_padded_batch_rows`
(= 32, the minimum tile height), which pads a single-sample batch to the tile boundary.

### Ring Buffer Update (Per Token)

In `forward`, the oldest slot is overwritten with the new Q/K/V slice using an in-place copy:

```python
# Slice the QKV portion from the fused projection result
qkv_new = ttnn.slice(proj, [0, 0, 0, 0], [1, 1, B, self.conv_dim])

# In-place copy into oldest slot (preserves tensor address for Trace)
ttnn.copy(qkv_new, self._conv_rows[self._oldest])
ttnn.deallocate(qkv_new)

# Advance the oldest pointer (wraps around modulo kernel_size)
self._oldest = (self._oldest + 1) % self.conv_kernel_size
```

After the copy, `_oldest` points to the slot that is now the "next oldest" — i.e., the
beginning of the oldest valid history window. The weighted sum then reads slots in order
starting from `_oldest`:

```python
acc = ttnn.multiply(self._conv_rows[self._oldest], self._conv_w_devs[0])
for i in range(1, self.conv_kernel_size):
    idx     = (self._oldest + i) % self.conv_kernel_size
    product = ttnn.multiply(self._conv_rows[idx], self._conv_w_devs[i])
    old_acc = acc
    acc     = ttnn.add(acc, product)
    ttnn.deallocate(product)
    ttnn.deallocate(old_acc)
conv_out = ttnn.silu(acc)
ttnn.deallocate(acc)
```

`_conv_w_devs[i]` is the $i$-th column of the depthwise conv weight. The weighted sum is:

```math
\text{conv out} = \text{SiLU}\!\left(\sum_{i=0}^{3} w_i \cdot \text{slot}_{(\text{oldest}+i) \bmod 4}\right)
```

### Conv Weight Layout

The conv1d weight has raw shape [conv_dim, 1, kernel_size] in the checkpoint (standard
PyTorch depthwise format). It is loaded and transposed to [kernel_size, conv_dim] then split
into `conv_kernel_size = 4` separate device tensors, one per kernel position:

```python
conv_weight_raw = state_dict[f"{layer_prefix}.conv1d.weight"].float().squeeze(1)
conv_w_host = conv_weight_raw.T   # shape: [kernel_size, conv_dim]

B = getattr(args, "tile_padded_batch_rows", 32)
self._conv_w_devs = []
for r in range(self.conv_kernel_size):
    w_pad = torch.zeros(1, 1, B, self.conv_dim, dtype=torch.bfloat16)
    w_pad[0, 0, 0, :] = conv_w_host[r].bfloat16()   # one row per kernel position
    self._conv_w_devs.append(
        ttnn.from_torch(
            w_pad, layout=ttnn.TILE_LAYOUT,
            device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    )
```

Each weight tensor has shape [1, 1, B_pad, conv_dim] where only the first row is non-zero;
the padding to B_pad rows allows the element-wise multiply with the ring buffer slots to
broadcast correctly in tile layout without an explicit reshape.

---

## Constant Device Tensors

Three additional parameter tensors are loaded once at construction and kept on device for the
lifetime of the module:

| Name | Shape on Device | dtype | Content |
|------|-----------------|-------|---------|
| `_dt_bias_dev` | [1, num_v_heads, 1, 1] | bfloat16 | `dt_bias` parameter |
| `_neg_A_exp_dev` | [1, num_v_heads, 1, 1] | bfloat16 | $-\exp(A_{\log})$ |
| `_norm_w_dev` | [1, num_v_heads, 1, head_v_dim] | bfloat16 | `norm.weight` expanded per-head |

The `_neg_A_exp_dev` precomputation avoids recomputing the exponential of `A_log` at every
token step (it is a per-layer constant). The [1, num_v_heads, 1, 1] shape allows broadcasting
against per-token per-head tensors without an explicit reshape in the kernel.

`_norm_w_dev` stores the `norm.weight` vector (shape [head_v_dim]) expanded to
[num_v_heads, head_v_dim] so that the fused kernel can apply per-head, per-dimension scaling
without a gather:

```python
norm_w = state_dict[f"{layer_prefix}.norm.weight"].float()
self._norm_w_dev = ttnn.from_torch(
    norm_w.unsqueeze(0).expand(self.num_v_heads, -1).unsqueeze(0).unsqueeze(2).contiguous().bfloat16(),
    layout=ttnn.TILE_LAYOUT, device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

---

## State Initialization

Before the first token is processed, `initialize_states` must be called. It allocates the two
stateful structures that persist across token steps:

```python
def initialize_states(self, batch_size=1, B_pad=32):
    H, D = self.num_v_heads, self.head_v_dim

    # Conv ring buffer: conv_kernel_size zero tensors
    self._conv_rows = []
    for _ in range(self.conv_kernel_size):
        self._conv_rows.append(
            ttnn.from_torch(
                torch.zeros(1, 1, B_pad, self.conv_dim, dtype=torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    self._oldest = 0

    # Recurrent state: float32 on device DRAM
    self._dev_state = ttnn.from_torch(
        torch.zeros(batch_size, H, self.head_k_dim, D),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=self.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
```

Key design decisions:

1. **`_dev_state` is float32.** The recurrent state accumulates outer products across many
   token steps. bfloat16 has only 7 mantissa bits; the accumulated rounding error grows
   visibly after 10–20 tokens and becomes garbage output past 30+ layers. float32 is required.
   See [`host_recurrence.md`](./host_recurrence.md) for the full hardware explanation.

2. **`_conv_rows` are bfloat16.** The conv buffer holds recent projections (not accumulated
   state), so bfloat16 precision is sufficient.

3. **`_oldest = 0` pointer.** The circular buffer starts with all slots zero and `_oldest`
   at 0. On the first token, slot 0 is overwritten with the new Q/K/V and `_oldest` advances
   to 1. After `conv_kernel_size` tokens, all slots contain real data.

4. **Fixed tensor addresses.** Both the conv ring buffer and `_dev_state` are allocated once.
   Subsequent updates use `ttnn.copy` (in-place) rather than reallocation, so the tensor
   addresses observed in a Metal Trace capture remain valid across decode steps.

---

**Next:** [`host_recurrence.md`](./host_recurrence.md)
