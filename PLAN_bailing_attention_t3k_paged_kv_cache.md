# Plan: Refactor TTNNBailingMoEAttentionT3K to Use TTNNPagedAttentionKVCache

## Problem Description

The current `TTNNBailingMoEAttentionT3K` in `bailing_attention_t3k.py` manages its own internal KV cache (`k_cache`, `v_cache`, `page_table` as instance attributes) and has a `forward()` API that differs from `TTNNBailingMoEAttention` in `attention.py`. This creates two issues:

1. **Inconsistent Interface**: The T3K version requires explicit `mode` and `user_id` parameters and internal cache initialization via `init_kv_cache()`, while the reference implementation uses external `TTNNPagedAttentionKVCache` with automatic mode detection.

2. **Code Duplication**: Both modules duplicate KV cache management logic rather than sharing the centralized `TTNNPagedAttentionKVCache` implementation.

## Current vs Target API Comparison

**Current T3K forward() Signature:**
```python
def forward(
    self,
    hidden_states: ttnn.Tensor,
    position_embeddings: Tuple[ttnn.Tensor, ttnn.Tensor],
    attention_mask: Optional[ttnn.Tensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.Tensor] = None,
    mode: str = None,                  # T3K-specific
    user_id: int = 0,                  # T3K-specific
    page_table: ttnn.Tensor = None,    # T3K-specific
    **kwargs,
) -> Tuple[ttnn.Tensor, None, None]
```

**Target forward() Signature (matching attention.py):**
```python
def forward(
    self,
    hidden_states: ttnn.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[ttnn.Tensor] = None,
    past_key_values=None,              # TTNNPagedAttentionKVCache
    past_key_value=None,               # legacy compatibility
    cache_position: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple
```

## Step-by-Step Implementation Plan

### Step 1: Remove Internal KV Cache Attributes

**Remove from `__init__`:**
```python
# Remove these lines:
self.k_cache = None
self.v_cache = None
self.page_table = None
self.paged_config = None
self._seq_lengths = [0]
self._seen_tokens = 0
```

**Remove `init_kv_cache()` method entirely** - cache initialization is now external.

### Step 2: Update forward() Signature

**Change from:**
```python
def forward(
    self,
    hidden_states: ttnn.Tensor,
    position_embeddings: Tuple[ttnn.Tensor, ttnn.Tensor],
    attention_mask: Optional[ttnn.Tensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.Tensor] = None,
    mode: str = None,
    user_id: int = 0,
    page_table: ttnn.Tensor = None,
    **kwargs,
) -> Tuple[ttnn.Tensor, None, None]:
```

**Change to:**
```python
def forward(
    self,
    hidden_states: ttnn.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[ttnn.Tensor] = None,
    past_key_values=None,
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
```

### Step 3: Update forward() Mode Detection Logic

**Change from explicit mode parameter:**
```python
if mode is None:
    mode = "decode" if seq_length == 1 else "prefill"
```

**Change to automatic detection based on KV cache type:**
```python
if past_key_value is not None and past_key_values is None:
    past_key_values = past_key_value  # legacy support

use_paged = isinstance(past_key_values, TTNNPagedAttentionKVCache)

if use_paged and seq_length == 1:
    return self._forward_decode_paged(...)
return self._forward_prefill(...)
```

### Step 4: Update forward_prefill() to Use TTNNPagedAttentionKVCache

**Current approach:**
```python
ttnn.experimental.paged_fill_cache(self.k_cache, k_8b, pt, batch_idx=user_id)
ttnn.experimental.paged_fill_cache(self.v_cache, v_8b, pt, batch_idx=user_id)
```

**New approach:**
```python
layer_idx = self.layer_idx
past_key_values.paged_fill_on_device(
    k_8b,
    v_8b,
    layer_idx=layer_idx,
    batch_idx=0,  # Following reference implementation pattern
)
```

### Step 5: Update forward_decode() to Use TTNNPagedAttentionKVCache

**Current approach:**
```python
# Update cache
pt = page_table if page_table is not None else self.page_table
ttnn.experimental.paged_fused_update_cache(
    self.k_cache, k, self.v_cache, v,
    update_idxs_tensor=current_pos,
    page_table=pt,
)

# SDPA decode
attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q, self.k_cache, self.v_cache,
    page_table_tensor=pt,
    cur_pos_tensor=current_pos,
    ...
)
```

**New approach (matching attention.py):**
```python
layer_idx = self.layer_idx

# Update paged KV cache
past_key_values.paged_update_on_device(
    k,
    v,
    layer_idx=layer_idx,
    current_pos=current_pos,
)

# Paged SDPA decode
attn_output = past_key_values.paged_sdpa_decode(
    q,
    layer_idx,
    current_pos=current_pos,
    scale=self.scale,
    program_config=self.sdpa_decode_program_config,
    compute_kernel_config=self.compute_kernel_config,
)
```

### Step 6: Handle T3K-Specific KV Head Sharding

The T3K version has `n_local_kv_heads = 1` per device (4 KV heads across 8 devices). The `TTNNPagedAttentionKVCache` needs to be initialized with the correct per-device KV head count.

**External initialization:**
```python
# Called by the model before inference
kv_cache = TTNNPagedAttentionKVCache(
    num_layers=num_layers,
    num_kv_heads=n_local_kv_heads,  # 1 for T3K (4 heads / 8 devices)
    head_dim=head_dim,
    config=paged_config,
    device=mesh_device,
)
kv_cache.to_device(mesh_device)
```

### Step 7: Update Tensor Permutation for paged_update_cache

The reference `_forward_decode_paged` does explicit permutation:
```python
# Permute B H S D -> S B H D for paged kernels
query_states = ttnn.permute(query_states, (2, 0, 1, 3))
key_states = ttnn.permute(key_states, (2, 0, 1, 3))
value_states = ttnn.permute(value_states, (2, 0, 1, 3))
```

The T3K version uses `nlp_create_qkv_heads_decode` which outputs in decode format. Verify the output format matches what `paged_update_on_device` expects.

### Step 8: Handle Multi-Device Replication

The reference implementation has `_to_replicated()` for converting all-gathered tensors to replicated topology for paged kernels:
```python
if self.device.get_num_devices() > 1:
    query_states = self._to_replicated(query_states)
    key_states = self._to_replicated(key_states)
    value_states = self._to_replicated(value_states)
```

The T3K version already uses `ReplicateTensorToMesh` for KV cache. Ensure consistency.

### Step 9: Update deallocate_weights_impl()

**Remove KV cache deallocation** (now external):
```python
def deallocate_weights_impl(self) -> None:
    if self.wqkv is not None:
        ttnn.deallocate(self.wqkv)
    if self.wo is not None:
        ttnn.deallocate(self.wo)
    # Remove k_cache, v_cache, page_table deallocation
    super().deallocate_weights_impl()
```

### Step 10: Remove get_seq_length() and reset_cache()

These methods managed internal state. With external KVCache, callers use:
```python
past_key_values.get_seq_length(layer_idx)
# Reset by creating new TTNNPagedAttentionKVCache
```

## T3K-Specific Considerations

1. **KV Head Sharding**: With 4 KV heads on 8 devices, each device has 1 local KV head. The `TTNNPagedAttentionKVCache` must be configured accordingly.

2. **CCL Operations**: The T3K version uses async CCL operations (`all_reduce_async`, `all_gather_async`, `reduce_scatter_minimal_async`). These remain unchanged - they operate on hidden states, not KV cache.

3. **Fused Cache Update**: T3K uses `paged_fused_update_cache` for K+V together. The `TTNNPagedAttentionKVCache.paged_update_on_device()` uses separate `paged_update_cache` calls. Consider whether to:
   - A) Use separate calls (matching reference)
   - B) Add `paged_fused_update_on_device()` to `TTNNPagedAttentionKVCache`

## Success Criteria

1. **API Match**: `forward()` signature exactly matches `TTNNBailingMoEAttention.forward()` in `attention.py`

2. **External KV Cache**: All KV cache management delegated to `TTNNPagedAttentionKVCache`

3. **Backward Compatible**: Existing tests pass (with updated test code to provide external KVCache)

4. **T3K Functionality Preserved**: CCL operations, weight sharding, and T3K-specific optimizations remain intact

5. **Numerical Correctness**: PCC > 0.99 against reference PyTorch implementation

## Migration Guide for Callers

**Before:**
```python
attn = TTNNBailingMoEAttentionT3K.from_torch(torch_attn, mesh_device, config)
attn.init_kv_cache(mesh_device, paged_config)
output, _, _ = attn.forward(
    hidden_states, position_embeddings,
    mode="prefill", user_id=0
)
```

**After:**
```python
attn = TTNNBailingMoEAttentionT3K.from_torch(torch_attn, mesh_device, config)
kv_cache = TTNNPagedAttentionKVCache(
    num_layers=1,
    num_kv_heads=config.num_kv_heads // config.num_devices,  # local heads
    head_dim=config.head_dim,
    config=paged_config,
)
kv_cache.to_device(mesh_device)
output, _, kv_cache = attn.forward(
    hidden_states, position_embeddings,
    past_key_values=kv_cache
)
```

## Critical Files

- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/bailing_attention_t3k.py` - Main file to refactor
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` - Reference implementation (lines 76-264, 2198-2958)
