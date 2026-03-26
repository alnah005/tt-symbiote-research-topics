# Current Implementation of QK Normalization

## `_apply_qk_norm` — Line-by-Line Walk-Through

`_apply_qk_norm` is defined at lines 2454–2493 of `attention.py`. The call at line 2659 of `_forward_decode_paged` is unconditional; the `use_qk_norm` flag is checked inside the method at line 2456, where an early return is taken if the flag is `False`. For Bailing MoE, `use_qk_norm=True`, so the early return is never taken and the method always proceeds.

### Detecting decode vs prefill mode (line 2462)

```python
is_decode_mode = q_shape[0] == 1 and len(q_shape) == 4
```

The method distinguishes between the decode layout (S B H D: `[1, B, H, D]` with the sequence dimension outermost) and the prefill layout (B H S D: `[B, H, S, D]`). In decode, `q_shape[0] == 1` because there is exactly one new token being decoded. This single flag governs which reshape formulas are used.

### Reshape: `[1, B, H, D]` → `[B*H, D]` (lines 2464–2468)

```python
seq_len, batch_size, num_heads, head_dim = q_shape
_, batch_kv, num_kv_heads, head_dim_k = k_shape
q_reshaped = ttnn.reshape(query_states, (batch_size * num_heads, head_dim))
k_reshaped = ttnn.reshape(key_states, (batch_kv * num_kv_heads, head_dim_k))
```

For Bailing MoE at B=32:
- Q: `[1, 32, 16, 128]` → `[512, 128]`
- K: `[1, 32, 4, 128]` → `[128, 128]`

This reshape is what requires L1 interleaved input. The `ttnn.reshape` op does not support sharded tensors (confirmed by the comment at line 2655: "reshape doesn't work on sharded tensors"). If Q and K were in a sharded layout (such as HEIGHT_SHARDED) rather than L1 interleaved, the reshape would fail.

### Norm application (lines 2474–2475)

```python
q_normed = self.query_layernorm(q_reshaped)
k_normed = self.key_layernorm(k_reshaped)
```

`self.query_layernorm` and `self.key_layernorm` are instances of `TTNNRMSNorm`. Their `forward` method (lines 92–96 of `normalization.py`) calls:

```python
x = ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.torch_layer.variance_epsilon)
```

`ttnn.rms_norm` runs independently on each device with the full norm input and the replicated weight.

The `TTNNRMSNorm.forward` method includes a layout guard at lines 93–94 of `normalization.py`:

```python
if x.layout != ttnn.TILE_LAYOUT:
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

Q and K are in TILE_LAYOUT at this point — their layout is determined by the output of the `q_proj` and `k_proj` matmul ops (which produce TILE_LAYOUT output), not by the `hidden_states` layout conversion at lines 2621–2622. Because the inputs are already in TILE_LAYOUT, this branch is not taken. When the branch is not taken and the input was already in L1 (which it is, from the `ttnn.to_memory_config` at lines 2655–2657), the norm output remains in L1 interleaved layout. No additional layout conversion occurs inside the norm on the normal decode path.

### Typecast guards (lines 2477–2484)

```python
if hasattr(q_normed, "to_ttnn"):
    q_normed = q_normed.to_ttnn
if hasattr(k_normed, "to_ttnn"):
    k_normed = k_normed.to_ttnn
if q_normed.dtype != ttnn.bfloat16:
    q_normed = ttnn.typecast(q_normed, ttnn.bfloat16)
if k_normed.dtype != ttnn.bfloat16:
    k_normed = ttnn.typecast(k_normed, ttnn.bfloat16)
```

The first two `hasattr` checks handle `TorchTTNNTensor` wrapper objects that may surface if the norm falls back to a Torch implementation. For normal TTNN execution these branches are not taken.

The `ttnn.typecast` guards at lines 2481–2484 check whether the norm output is already `bfloat16`. Since `ttnn.rms_norm` preserves the input dtype and the input Q/K tensors are already `bfloat16` (enforced by the explicit typecasts at lines 2648–2653 of `attention.py` before the `to_memory_config` calls), these typecast ops are **not triggered** in the normal decode path. They are defensive guards for cases where the fallback Torch norm path returns a `float32` output.

### Reshape back: `[B*H, D]` → `[1, B, H, D]` (lines 2486–2488)

```python
query_states = ttnn.reshape(q_normed, (seq_len, batch_size, num_heads, head_dim))
key_states = ttnn.reshape(k_normed, (seq_len, batch_kv, num_kv_heads, head_dim_k))
```

Restores the S B H D layout expected by the rest of the decode path.

---

## `TTNNRMSNorm` Internals

`TTNNRMSNorm` is defined at lines 69–96 of `normalization.py`. Key implementation details:

### Weight preprocessing (`preprocess_weights_impl`, lines 80–86)

```python
self.tt_weight = ttnn.from_torch(
    self.torch_layer.weight.unsqueeze(0).expand(32, -1),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT
)
```

The original weight is a 1-D tensor of shape `[D]` = `[128]`. It is unsqueezed to `[1, 128]` and then expanded to `[32, 128]` so it occupies a full tile (TILE=32 rows). This expansion is required because `ttnn.rms_norm` expects the weight tensor to have a row dimension that is a multiple of the tile size. The `expand` does not allocate new memory (it is a view in PyTorch); the materialization into a TTNN tile-layout tensor happens at `ttnn.from_torch`.

### Weight placement (`move_weights_to_device_impl`, lines 88–90)

```python
def move_weights_to_device_impl(self):
    self.tt_weight = ttnn.to_device(self.tt_weight, self.device)
```

No `mesh_mapper` is explicitly specified here. The `TTNNModule` base class provides `self.device`, which for a multi-device mesh is the full `MeshDevice`. Without specifying a mapper, `ttnn.to_device` replicates the weight tensor to all N=8 devices using the default behavior. The result is that each device holds an identical copy of the `[32, 128]` weight.

This means `TTNNRMSNorm` is **non-distributed**: the weight is replicated, and `ttnn.rms_norm` runs the full norm independently on each device. There is no cross-device communication during the forward pass. This is the correct and intended behavior because, at the point `_apply_qk_norm` is called, each device already holds the full Q and K tensors (they were replicated by the all-gather).

---

## The "Head Dim Too Small to Shard Across Devices" Constraint

The comment at line 2380 of `attention.py` reads:

```python
# QK norms use non-distributed version (head_dim too small to shard across devices)
```

This comment refers to the choice to use `TTNNRMSNorm` instead of `TTNNDistributedRMSNorm`. To understand what constraint it describes, it is necessary to examine what `TTNNDistributedRMSNorm` does and what tensor it would receive.

`TTNNDistributedRMSNorm` (lines 100–151 of `normalization.py`) is designed for the case where the input tensor has its **reduction dimension (hidden_size) sharded across devices** (its forward pass is described in detail in `distributed_alternative.md`).

For it to work, the shard width `hidden_size/N` must be compatible with the tile size. The TTNN tile is 32 elements wide: a shard that is narrower than 32 elements cannot form a valid tile and will be rejected by the kernel.

Applied to QK norm: if one attempted to use `TTNNDistributedRMSNorm` on a pre-all-gather Q tensor, each device would hold a shard of Q's last dimension. The Q tensor at the reduce-scatter output stage has shape `[B, 1, H/N * D]` = `[B, 1, 256]` per device (N=8 devices, H=16 Q heads, D=128; see Chapter 2 for the sharding details). If one tried to shard not across heads but across the D=128 dimension of a single head, each device would hold `D/N = 128/8 = 16` elements per head — below the minimum tile width of TILE=32. This is what the comment is referring to: sharding the head_dim itself across 8 devices is infeasible because 16 < 32.

However, this framing is important to evaluate carefully: the constraint is specifically about **sharding a single head's D=128 dimension across N=8 devices**, which is what `TTNNDistributedRMSNorm` would require if naively applied to a per-head norm where D=128 is the reduction axis. It is **not** a constraint on operating on full per-head tensors independently on each device after the all-gather — that is exactly what the current `TTNNRMSNorm` approach does, and it works correctly. The feasibility of an alternative (performing the norm before the all-gather, on the col-sharded Q) is analyzed in `distributed_alternative.md`, and the conclusion there is different: because each device holds 2 complete Q heads in the pre-all-gather col-sharded layout, the per-head norm can run entirely intra-device with no cross-device communication required.

---

## DRAM→L1 Transition Cost

The two `ttnn.to_memory_config` calls at lines 2655–2657 of `attention.py` copy Q and K from DRAM to L1 interleaved layout on every decode step. These calls are the direct cost imposed by the QK norm's reshape requirement:

```python
# Move to L1 for QK norm (reshape doesn't work on sharded tensors)
query_states = ttnn.to_memory_config(query_states, ttnn.L1_MEMORY_CONFIG)
key_states = ttnn.to_memory_config(key_states, ttnn.L1_MEMORY_CONFIG)
```

At B=32, the copies move 131,072 bytes (Q) and 32,768 bytes (K) from DRAM to L1 — 163,840 bytes total per decode step per device (shapes and byte sizes established in `index.md`'s Tensor Shape Reference). At B=1 the volumes are 4,096 bytes (Q) and 1,024 bytes (K), a total of 5,120 bytes. These are small relative to DRAM bandwidth capacity, but the transition still requires the TTNN runtime to dispatch two separate NoC copy operations and stall any subsequent ops on Q and K until the copies complete. The copy also occupies L1 space for the duration of the norm computation.

After `_apply_qk_norm` returns, Q and K are in L1 interleaved layout. The next operations are the HEIGHT_SHARDED `ttnn.to_memory_config` calls for RoPE (Chapter 3, step 4 and 5), so the Q and K tensors transition immediately from L1 interleaved to HEIGHT_SHARDED L1 — two further memory config ops on the same tensors, all within a few lines of the decode path.

---

**Next:** [Distributed Alternative Analysis](distributed_alternative.md)
