# Implementation Plan: Fused QKV + AllReduce for BailingAttention Decode

**Date:** 2026-03-26
**Author:** Architect Agent
**Target file:** `models/experimental/tt_symbiote/modules/attention.py` -- `TTNNBailingMoEAttention`
**Linear file:** `models/experimental/tt_symbiote/modules/linear.py`

---

## Background and Motivation

Phase 1 (nlp_create_qkv_heads_decode) was a REGRESSION. Reducing device-side op count
does not help when **host dispatch overhead dominates**. The real bottleneck is too many
CCL collective ops. Each collective incurs ~0.5ms host dispatch overhead on T3K.

**Current CCL ops per decode (5 total):**
1. `reduce_scatter` inside Q projection (TTNNLinearIColShardedWRowSharded)
2. `all_gather` to get hidden_states for K/V projections (line 2626)
3. `all_gather` on Q output after reduce_scatter (line 2631)
4. `all_gather` on K output (line 2632)
5. `all_gather` on V output (line 2633)

**Target: Reduce 5 CCL ops to 1** (a single `all_reduce`).

**Expected savings:** 2-4ms (4 fewer host dispatches at ~0.5ms each).

---

## Model Dimensions (Ling-mini-2.0)

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| num_heads (Q) | 16 |
| num_kv_heads (K/V) | 4 |
| head_dim | 128 |
| Q output size | 16 * 128 = 2048 |
| K output size | 4 * 128 = 512 |
| V output size | 4 * 128 = 512 |
| QKV total output | 2048 + 512 + 512 = 3072 |
| num_devices (T3K) | 8 |
| hidden_size / 8 | 256 (per-device column shard) |

---

## Approach Chosen: Column-Parallel Fused QKV + all_reduce

### Why all_reduce (not reduce_scatter)

With reduce_scatter, each device gets 1/8 of the output (3072/8 = 384). This means:
- Q: 2048/8 = 256 = 2 heads per device (OK)
- K: 512/8 = 64 = 0.5 heads per device (DOES NOT DIVIDE)
- V: 512/8 = 64 = 0.5 heads per device (DOES NOT DIVIDE)

With all_reduce, each device gets the FULL output (3072). This means every device has
all 16 Q heads + 4 KV heads. This wastes bandwidth (8x replication) but:
1. The tensors are tiny in decode (batch * 3072 = ~96KB for batch=32)
2. We eliminate ALL 5 CCL ops and replace with 1
3. The all_reduce result is already replicated -- no `_to_replicated` needed

### Why NOT all_reduce_create_qkv_heads (the galaxy-only fused op)

The `ttnn.experimental.all_reduce_create_qkv_heads` op exists but:
1. It is designed for TG (8x4 Galaxy mesh), not T3K (1x8 mesh)
2. It requires persistent fabric, sub-devices, and buffer tensors
3. It interleaves QKV in a specific Galaxy-oriented layout
4. Bringing it up on T3K is a separate project

We use the simpler `ttnn.all_reduce` which wraps `all_reduce_async` internally and
handles T3K topology automatically.

### Why NOT the composite reduce_scatter + all_gather approach

The `tt_all_reduce` function in `models/tt_transformers/tt/ccl.py` decomposes all_reduce
into reduce_scatter + all_gather, which is 2 CCL ops. The native `ttnn.all_reduce`
is a single fused CCL operation (1 host dispatch).

---

## Detailed Implementation

### Step 1: Create Fused QKV Weight in `from_torch()`

**File:** `attention.py`, `TTNNBailingMoEAttention.from_torch()`

**Current code (lines 2336-2378):**
```python
# Currently creates 3 separate nn.Linear modules:
q_linear = nn.Linear(hidden_size, q_size)      # [2048, 2048]
k_linear = nn.Linear(hidden_size, kv_size)      # [512, 2048]
v_linear = nn.Linear(hidden_size, kv_size)      # [512, 2048]

new_attn.q_proj = TTNNLinearIColShardedWRowSharded.from_torch(q_linear)
new_attn.k_proj = TTNNLinearIReplicatedWColSharded.from_torch(k_linear)
new_attn.v_proj = TTNNLinearIReplicatedWColSharded.from_torch(v_linear)
```

**New code:**
```python
# Keep separate Q, K, V projections for PREFILL (unchanged)
new_attn.q_proj = TTNNLinearIColShardedWRowSharded.from_torch(q_linear)
new_attn.k_proj = TTNNLinearIReplicatedWColSharded.from_torch(k_linear)
new_attn.v_proj = TTNNLinearIReplicatedWColSharded.from_torch(v_linear)

# Create FUSED QKV weight for DECODE (new)
# Concatenate Q, K, V weights vertically: [3072, 2048]
qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [3072, 2048]

qkv_bias = None
if q_bias is not None:
    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)  # [3072]

qkv_linear = nn.Linear(new_attn.hidden_size, q_size + 2 * kv_size,
                        bias=qkv_bias is not None)
qkv_linear.weight.data = qkv_weight
if qkv_bias is not None:
    qkv_linear.bias.data = qkv_bias

new_attn.qkv_proj = TTNNLinearIColShardedWAllReduced.from_torch(qkv_linear)
```

**Weight layout on each device:**
- `W_qkv` shape: `[3072, 2048]` (out_features x in_features)
- Column-shard on dim=-2 (the input dimension after transpose): each device holds `[3072, 256]`
- After matmul: `[B, 1, 3072]` partial sums on each device
- After all_reduce: `[B, 1, 3072]` full result replicated on all devices

### Step 2: Create New Linear Class `TTNNLinearIColShardedWAllReduced`

**File:** `linear.py`

This class is similar to `TTNNLinearIColShardedWRowSharded` but replaces `reduce_scatter`
with `all_reduce`. The key difference: reduce_scatter gives 1/N of the output,
all_reduce gives the full output replicated on all devices.

```python
class TTNNLinearIColShardedWAllReduced(TTNNLinearInputShardedWeightSharded):
    """Linear layer with column-sharded input, row-sharded weight, and all_reduce output.

    Like TTNNLinearIColShardedWRowSharded, but uses all_reduce instead of
    reduce_scatter. This is needed when the output dimension does not divide
    evenly by num_devices (e.g., KV heads < num_devices in GQA).

    The result is replicated on all devices (not sharded).
    """

    def __init__(self, in_features, out_features) -> None:
        super().__init__(in_features, out_features, input_dim=-1, weight_dim=-2)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass: matmul + all_reduce."""
        # Validate sharding (same as IColShardedWRowSharded)
        if len(input_tensor.tensor_topology().placements()) == 1:
            assert input_tensor.tensor_topology().placements()[0].dim == self.input_dim
        elif len(input_tensor.tensor_topology().placements()) == 2:
            assert input_tensor.tensor_topology().placements()[0].dim == 0
            assert input_tensor.tensor_topology().placements()[1].dim == self.input_dim

        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)

        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        # Matmul: partial sum on each device
        tt_output = ttnn.linear(input_tensor, self.tt_weight,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # All-reduce: sum partial results, replicate on all devices
        # Uses ttnn.all_reduce which auto-selects topology for the mesh
        tt_output = ttnn.all_reduce(
            tt_output,
            cluster_axis=1,          # T3K is 1x8, reduce along axis 1
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=1,
        )

        if self.tt_bias is not None:
            tt_output += self.tt_bias

        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output
```

**Key differences from `TTNNLinearIColShardedWRowSharded`:**

| Aspect | IColShardedWRowSharded | IColShardedWAllReduced |
|--------|----------------------|----------------------|
| CCL op | `reduce_scatter_minimal_async` | `ttnn.all_reduce` |
| Output topology | Sharded (1/N per device) | Replicated (full on all) |
| Semaphores needed | RS + barrier semaphore handles | None (auto-managed) |
| Output shape | `[B, 1, out_features/N]` | `[B, 1, out_features]` |

**IMPORTANT:** The `ttnn.all_reduce` API (in `all_reduce.cpp`) takes `cluster_axis` as
an optional parameter and auto-determines topology. For T3K (1x8 mesh), `cluster_axis=1`
reduces along the 8 devices. It internally calls `all_reduce_async` with `ReduceType::Sum`.

**Fallback for non-T3K:** The `@run_on_devices(DeviceArch.T3K)` decorator handles
fallback to PyTorch for non-T3K architectures. For single-device, the base class
`forward()` (no CCL) runs instead.

### Step 3: New Decode Forward Path

**File:** `attention.py`, `TTNNBailingMoEAttention._forward_decode_paged()`

**Current flow (lines 2610-2799):**
```
hidden_states [B,1,256] (column-sharded across 8 devices)
  |
  +-> q_proj (matmul + reduce_scatter) -> q [B,1,256] (sharded)
  |     +-> all_gather -> q_full [B,1,2048]            # CCL #1 (inside q_proj)
  |                                                      # CCL #2 (line 2631)
  +-> all_gather hidden_states -> hidden_rep [B,1,2048] # CCL #3 (line 2626)
  |     +-> k_proj (matmul) -> k [B,1,512]              # (inside k_proj)
  |     +-> v_proj (matmul) -> v [B,1,512]              # (inside v_proj)
  |     +-> all_gather k -> k_full [B,1,512]            # CCL #4 (line 2632)
  |     +-> all_gather v -> v_full [B,1,512]            # CCL #5 (line 2633)
  |
  ... reshape, QK norm, RoPE, SDPA, concat_heads, dense ...
```

**New flow:**
```
hidden_states [B,1,256] (column-sharded across 8 devices)
  |
  +-> qkv_proj (matmul + all_reduce) -> qkv [B,1,3072]  # 1 matmul + 1 CCL
  |
  +-> slice qkv into Q [B,1,2048], K [B,1,512], V [B,1,512]
  |
  ... reshape, QK norm, RoPE, SDPA, concat_heads, dense ...
```

**Exact new code for `_forward_decode_paged()`:**

```python
def _forward_decode_paged(
    self,
    hidden_states: ttnn.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[ttnn.Tensor],
    past_key_values: "TTNNPagedAttentionKVCache",
    cache_position: Optional[torch.LongTensor],
) -> tuple:
    """Decode path using fused QKV matmul + all_reduce."""
    batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

    if hidden_states.layout != ttnn.TILE_LAYOUT:
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # === FUSED QKV: 1 matmul + 1 all_reduce (replaces 3 matmuls + 5 CCL ops) ===
    qkv_states = self.qkv_proj(hidden_states)
    # qkv_states shape: [B, 1, 3072] replicated on all devices

    if hasattr(qkv_states, "to_ttnn"):
        qkv_states = qkv_states.to_ttnn

    # Split into Q, K, V by slicing last dimension
    q_size = self.num_heads * self.head_dim         # 2048
    kv_size = self.num_kv_heads * self.head_dim     # 512

    query_states = qkv_states[:, :, :q_size]                          # [B, 1, 2048]
    key_states = qkv_states[:, :, q_size:q_size + kv_size]            # [B, 1, 512]
    value_states = qkv_states[:, :, q_size + kv_size:]                # [B, 1, 512]
    ttnn.deallocate(qkv_states)

    # Reshape Q/K/V to decode format [1, batch, heads, head_dim]
    query_states = ttnn.reshape(query_states, (1, batch_size, self.num_heads, self.head_dim))
    key_states = ttnn.reshape(key_states, (1, batch_size, self.num_kv_heads, self.head_dim))
    value_states = ttnn.reshape(value_states, (1, batch_size, self.num_kv_heads, self.head_dim))

    if query_states.dtype != ttnn.bfloat16:
        query_states = ttnn.typecast(query_states, ttnn.bfloat16)
    if key_states.dtype != ttnn.bfloat16:
        key_states = ttnn.typecast(key_states, ttnn.bfloat16)
    if value_states.dtype != ttnn.bfloat16:
        value_states = ttnn.typecast(value_states, ttnn.bfloat16)

    # Move to L1 for QK norm
    query_states = ttnn.to_memory_config(query_states, ttnn.L1_MEMORY_CONFIG)
    key_states = ttnn.to_memory_config(key_states, ttnn.L1_MEMORY_CONFIG)

    query_states, key_states = self._apply_qk_norm(query_states, key_states)

    # --- Remaining path is IDENTICAL to current code from line 2661 onwards ---
    layer_idx = self._fallback_torch_layer.layer_idx

    if cache_position is None:
        cur_pos = past_key_values.get_seq_length(layer_idx)
        cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
    else:
        cp = cache_position
        if isinstance(cp, TorchTTNNTensor):
            cp = cp.to_torch
        if isinstance(cp, ttnn.Tensor):
            mesh_composer = None
            if hasattr(cp, "device") and cp.device() is not None and cp.device().get_num_devices() > 1:
                mesh_composer = ttnn.ConcatMeshToTensor(cp.device(), dim=0)
            cp = ttnn.to_torch(cp, mesh_composer=mesh_composer)
        cache_position_tensor = cp.flatten()[:batch_size].to(torch.int32)

    # ... (RoPE setup, paged cache update, SDPA, concat_heads, dense -- unchanged)
```

**Slicing detail:** TTNN supports slicing via `ttnn.slice` or Python `__getitem__`:
```python
# Option A: ttnn.slice (explicit)
query_states = ttnn.slice(qkv_states, [0, 0, 0], [B, 1, q_size])
key_states = ttnn.slice(qkv_states, [0, 0, q_size], [B, 1, q_size + kv_size])
value_states = ttnn.slice(qkv_states, [0, 0, q_size + kv_size], [B, 1, q_size + 2*kv_size])

# Option B: Python indexing (may internally call ttnn.slice)
query_states = qkv_states[:, :, :q_size]
key_states = qkv_states[:, :, q_size:q_size+kv_size]
value_states = qkv_states[:, :, q_size+kv_size:]
```

Use Option A (explicit `ttnn.slice`) for clarity and to ensure it stays on device.

### Step 4: Prefill Path -- NO CHANGES

The prefill path (`_forward_prefill`) remains **completely unchanged**. It continues to use:
- `self.q_proj` (TTNNLinearIColShardedWRowSharded)
- `self.k_proj` (TTNNLinearIReplicatedWColSharded)
- `self.v_proj` (TTNNLinearIReplicatedWColSharded)

**Rationale:** Prefill processes large sequences (e.g., 1024+ tokens). The CCL overhead
is amortized over the larger computation. The reduce_scatter approach is more
bandwidth-efficient for large tensors. The fused QKV + all_reduce is specifically
optimized for the decode case where tensors are tiny (batch * 3072 bytes).

The `qkv_proj` weight adds ~3072 * 2048 * 2 = 12.6MB to memory. This is acceptable
since the separate Q/K/V weights are the same total size and both must coexist.

**Alternative (memory optimization, deferred):** If memory becomes tight, we could
derive the fused QKV weight from the separate Q/K/V weights at `move_weights_to_device`
time, avoiding double storage. But this adds complexity and is not needed yet.

### Step 5: Dense Projection -- NO CHANGES

The dense (output) projection is `TTNNLinearIReplicatedWColSharded`:
- Input: replicated (from `nlp_concat_heads_decode`)
- Weight: column-sharded
- Output: column-sharded (for the next layer's hidden_states)

This is already correct. After the fused QKV change, the dense projection still receives
a replicated input from `nlp_concat_heads_decode`, so nothing changes.

---

## Op Count Comparison

| Operation | Before | After | Savings |
|-----------|--------|-------|---------|
| CCL collective ops | 5 (1 RS + 4 AG) | 1 (1 AR) | -4 ops |
| Host dispatches (CCL) | 5 | 1 | -4 dispatches |
| Matmul ops | 4 (Q + K + V + Dense) | 2 (QKV + Dense) | -2 ops |
| ttnn.slice | 0 | 3 | +3 ops |
| Total host dispatches | ~30+ | ~26 | -4+ |

**Estimated latency impact:**
- 4 fewer CCL host dispatches: **-2.0 to -2.5ms** (at ~0.5ms each)
- 2 fewer matmul dispatches: **-0.2 to -0.4ms**
- 3 new slice ops: **+0.1 to 0.2ms** (slice is lightweight)
- all_reduce vs reduce_scatter: **~neutral** (both move similar data for these tiny tensors)
- **Net savings: ~2.0 to 2.7ms**

---

## Weight Preparation Detail

### Current weight flow in `from_torch()`:

```
torch_attn.query_key_value.weight  # [(16+2*4)*128, 2048] = [3072, 2048]
    |
    +-> split into q_weight [2048, 2048], k_weight [512, 2048], v_weight [512, 2048]
    +-> _reverse_permute_weight on Q and K (HF -> Meta layout for RoPE)
    +-> Create 3 nn.Linear modules
    +-> q_proj = TTNNLinearIColShardedWRowSharded.from_torch(q_linear)
    +-> k_proj = TTNNLinearIReplicatedWColSharded.from_torch(k_linear)
    +-> v_proj = TTNNLinearIReplicatedWColSharded.from_torch(v_linear)
```

### New weight flow (additional):

```
    +-> qkv_weight = cat([q_weight, k_weight, v_weight], dim=0)  # [3072, 2048]
    +-> qkv_linear = nn.Linear(2048, 3072)
    +-> qkv_proj = TTNNLinearIColShardedWAllReduced.from_torch(qkv_linear)
```

The fused weight is `[3072, 2048]`. After `preprocess_linear_weight` with
`shard_tensor_to_mesh_mapper(device, dim=-2)`, each device holds `[3072, 256]`.

The matmul `[B, 1, 256] x [256, 3072]` produces `[B, 1, 3072]` partial sums.
After all_reduce (sum), we get the correct full QKV output replicated on all devices.

### Bias handling:

If bias exists (it does for Ling):
```python
qkv_bias = cat([q_bias, k_bias, v_bias], dim=0)  # [3072]
```
The bias is added AFTER all_reduce, so it must NOT be sharded across devices.
It should be replicated. The `TTNNLinearIColShardedWAllReduced` class handles this:
bias is added after all_reduce, and since all_reduce output is replicated, the bias
just needs to be replicated too.

**Check:** In `TTNNLinearInputShardedWeightSharded.move_weights_to_device_impl()`,
the bias is sharded with `shard_tensor_to_mesh_mapper(device, dim=self.input_dim)`.
For IColShardedWAllReduced, `input_dim=-1`, so bias gets column-sharded. But after
all_reduce the output is full-sized, so we need the FULL bias, not sharded.

**FIX NEEDED:** Override `move_weights_to_device_impl()` in `TTNNLinearIColShardedWAllReduced`
to replicate the bias instead of sharding it:
```python
def move_weights_to_device_impl(self):
    # Weight is row-sharded (same as parent)
    if isinstance(self.tt_weight_host, torch.Tensor):
        self.tt_weight_host = preprocess_linear_weight(
            self.tt_weight_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
        )
    # Bias is REPLICATED (not sharded) because output is all_reduced (full)
    if isinstance(self.tt_bias_host, torch.Tensor):
        self.tt_bias_host = preprocess_linear_bias(
            self.tt_bias_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
    self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
    self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None
```

---

## Edge Cases

### 1. Single device (no mesh / non-T3K)

The `@run_on_devices(DeviceArch.T3K)` decorator on `TTNNLinearIColShardedWAllReduced.forward()`
causes fallback to the PyTorch linear layer on non-T3K devices. The `qkv_proj` attribute
would still exist but would run through PyTorch fallback.

The `_forward_decode_paged` code should check for single-device mode:
```python
num_devices = self.device.get_num_devices() if hasattr(self.device, 'get_num_devices') else 1
if num_devices > 1 and self.qkv_proj is not None:
    # Fused QKV path
    qkv_states = self.qkv_proj(hidden_states)
    ...
else:
    # Existing separate Q/K/V path (unchanged)
    query_states = self.q_proj(hidden_states)
    hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
    key_states = self.k_proj(hidden_states_replicated)
    value_states = self.v_proj(hidden_states_replicated)
    ...
```

### 2. Different mesh sizes

The `ttnn.all_reduce(cluster_axis=1)` works for any 1xN mesh (T3K is 1x8).
For 2D meshes (Galaxy TG), `cluster_axis` would need to be adjusted. But TT-Symbiote
currently only targets T3K, so this is not a concern.

### 3. QKV slice alignment with TILE_SIZE

TTNN tile size is 32. The slice boundaries must be tile-aligned:
- Q end: 2048 (divisible by 32) -- OK
- K start: 2048, K end: 2560 (divisible by 32) -- OK
- V start: 2560, V end: 3072 (divisible by 32) -- OK

### 4. Memory budget

Additional weight memory: 3072 * 2048 * 2 bytes = 12.6MB per layer.
Ling-mini-2.0 has 32 layers, so 32 * 12.6MB = 403MB additional weight storage.
This is acceptable on T3K with 12GB DRAM per device.

**Future optimization:** Derive fused weight from separate weights to avoid doubling.
Or, once the fused path is validated, remove the separate Q/K/V projections for decode
and only keep them for prefill.

### 5. Tensor topology after all_reduce

After `ttnn.all_reduce`, the tensor has replicated data on all devices. The topology
metadata should reflect this. If downstream ops (reshape, QK norm, RoPE) require
specific topology, we may need `_to_replicated()`. However, since we are NOT using
`nlp_create_qkv_heads_decode` (which requires replicated topology), and reshape/RoPE
work on any topology, this should not be an issue.

**Key validation:** Ensure `ttnn.reshape`, `ttnn.to_memory_config`, and
`ttnn.experimental.rotary_embedding_llama` work correctly on all_reduce output tensors.

---

## Implementation Order

### Phase 2a: Core implementation (this plan)

1. Add `TTNNLinearIColShardedWAllReduced` class to `linear.py`
2. Add `qkv_proj` creation to `TTNNBailingMoEAttention.from_torch()`
3. Modify `_forward_decode_paged()` to use fused QKV path
4. Run existing tests: `tests/experimental/tt_symbiote/test_bailing_attention*.py`
5. Profile with Tracy to measure actual latency improvement

### Phase 2b: Validation

1. Compare decode output numerically (fused vs separate Q/K/V)
2. Run end-to-end text generation to verify correctness
3. Profile to confirm CCL op count reduction (5 -> 1)

### Phase 3: Trace capture (separate plan)

After Phase 2 validates, enable TTNN trace capture for the decode path.
This eliminates remaining host dispatch overhead for non-CCL ops.

---

## Risks and Rollback Plan

### Risk 1: `ttnn.all_reduce` not available on T3K

**Mitigation:** The `ttnn.all_reduce` C++ implementation delegates to `all_reduce_async`
and auto-selects topology. T3K test files exist (`tests/nightly/t3000/ccl/test_all_reduce.py`)
confirming it works on T3K. NOTE: The test shows `(4, 1)` devices pass but
`(8, 1)` is skipped due to "hang in all gather". This is a risk.

**Fallback:** If `all_reduce` hangs on 8 devices, use the composite approach:
```python
# reduce_scatter + all_gather = 2 CCL ops (still better than 5)
tt_output = ttnn.experimental.reduce_scatter_minimal_async(tt_output, dim=3, ...)
tt_output = ttnn.all_gather(tt_output, dim=3, num_links=1)
```
This would still save 3 CCL ops (5 -> 2).

### Risk 2: Tensor topology mismatch after all_reduce

**Mitigation:** Test that downstream ops (reshape, to_memory_config, RoPE,
paged_update_cache, paged_sdpa_decode) work correctly with all_reduce output.
If topology issues arise, add `_to_replicated()` call after all_reduce.

### Risk 3: Numerical accuracy

**Mitigation:** The fused QKV matmul computes exactly the same thing as separate
Q/K/V matmuls (just concatenated). The all_reduce performs the same sum as
reduce_scatter + all_gather. Numerical results should be identical within floating
point tolerance. Run PCC comparison tests.

### Risk 4: Memory increase from duplicate weights

**Mitigation:** 403MB across all layers is ~3.4% of T3K total DRAM (96GB across 8 devices).
Acceptable for now. Remove duplicates once fused path is validated.

### Rollback plan

The fused path is gated behind `self.qkv_proj is not None` and only used during decode.
To rollback:
1. Remove the `qkv_proj` creation from `from_torch()`
2. Remove the `if self.qkv_proj` branch from `_forward_decode_paged()`
3. The separate Q/K/V path remains untouched throughout

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| CCL ops per decode | 5 | 1 |
| Matmul ops per decode | 4 | 2 |
| Host dispatches saved | -- | ~6 |
| Estimated latency savings | -- | 2.0-2.7ms |
| Memory cost | 0 | +12.6MB/layer |
| Prefill changes | -- | None |
| Code files modified | -- | 2 (attention.py, linear.py) |
| Risk level | -- | Medium (all_reduce on 8-device T3K may hang) |
