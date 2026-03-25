# PLAN: Bailing Attention T3K nlp_concat_heads_decode Sharding Fix

## Problem Statement

After the RoPE partial rotation fix, a new error appeared:

```
RuntimeError: Input tensor must be sharded in `nlp_concat_heads_decode`
```

The `nlp_concat_heads_decode` kernel requires HEIGHT_SHARDED input, but the attention output from `paged_sdpa_decode` is in DRAM_MEMORY_CONFIG.

## Root Cause Analysis

### nlp_concat_heads_decode Kernel Requirements

From `/home/ttuser/salnahari/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp`:

**Input Requirements:**
- Shape: `[seqlen=1, batch<=32, padded_heads=32, head_dim]` = `[1, B, 32, D]`
- Must be HEIGHT_SHARDED (line 44-48)
- `shard_spec.shape[0] == 32` (padded heads, line 56-59)
- `shard_spec.shape[1] == head_dim` (line 51-54)
- `num_cores == batch` (line 68)

**Output:**
- Shape: `[1, 1, batch, num_heads * head_dim]` = `[1, 1, B, H*D]`

### Current tt_symbiote Flow (Broken)

In `_forward_decode_paged()` at `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`:

1. **Line 2808-2815**: `paged_sdpa_decode()` returns output in DRAM_MEMORY_CONFIG
   ```python
   attn_output = past_key_values.paged_sdpa_decode(
       query_states,
       layer_idx,
       current_pos=cur_pos_tt,
       scale=self.scaling,
       program_config=self.sdpa.decode_program_config,
       compute_kernel_config=self.sdpa.compute_kernel_config,
   )
   ```

2. **Line 2817-2821**: `nlp_concat_heads_decode` called directly on DRAM tensor (FAILS)
   ```python
   attn_output = ttnn.experimental.nlp_concat_heads_decode(
       attn_output,
       num_heads=self.num_heads,
   )
   ```

### tt_transformers Reference Pattern (Working)

In `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/attention.py`:

1. **Line 618-628**: SDPA decode returns to DRAM_MEMORY_CONFIG
2. **Line 644-649**: Convert output to HEIGHT_SHARDED before concat_heads
   ```python
   attn_output_11BH = ttnn.to_memory_config(
       attn_output_1G4D,
       memory_config=self.args.get_attn_sdpa_output_mem_config(
           Mode.DECODE, self.batch_size_per_device_group, self.prefetcher
       ),
   )
   ```
3. **Line 651-655**: Then call nlp_concat_heads_decode on sharded tensor
   ```python
   attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
       attn_output_11BH,
       num_heads=self.n_local_heads,
       sub_core_grids=self.prefetcher.all_worker_cores_range_set if self.prefetcher is not None else None,
   )
   ```

### The Memory Config Pattern from tt_transformers

From `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/model_config.py` line 1649-1664:

```python
def get_attn_sdpa_output_mem_config(
    self, mode: Mode, batch_size_per_device_group: int = 1, prefetcher: Prefetcher = None
):
    """Get the memory config for SDPA output in attention."""
    if mode == Mode.DECODE:
        if prefetcher is not None:
            start_core = ttnn.CoreCoord(1, 0)
            return ttnn.create_sharded_memory_config(
                shape=(math.ceil(self.n_local_heads / 32) * 32, self.head_dim),
                core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    start_core,
                    batch_size_per_device_group,
                    prefetcher.all_worker_cores_range_set,
                    row_wise=True,
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
                ...
```

## Solution

Add a memory config conversion step between `paged_sdpa_decode()` and `nlp_concat_heads_decode()`.

### Exact Fix for TTNNBailingMoEAttention._forward_decode_paged()

Replace lines ~2808-2821 with:

```python
# Paged SDPA decode
# Output shape: [1, batch, num_heads, head_dim] (DRAM_MEMORY_CONFIG)
attn_output = past_key_values.paged_sdpa_decode(
    query_states,
    layer_idx,
    current_pos=cur_pos_tt,
    scale=self.scaling,
    program_config=self.sdpa.decode_program_config,
    compute_kernel_config=self.sdpa.compute_kernel_config,
)

# Convert SDPA output to HEIGHT_SHARDED for nlp_concat_heads_decode
# nlp_concat_heads_decode requires:
#   - Input shape: [1, batch, 32, head_dim] (padded_heads=32)
#   - HEIGHT_SHARDED with shard_shape=(32, head_dim), num_cores=batch
import math
padded_heads = 32  # nlp_concat_heads_decode requires exactly 32 padded heads
sdpa_output_memcfg = ttnn.create_sharded_memory_config(
    shape=(padded_heads, self.head_dim),
    core_grid=ttnn.CoreGrid(y=1, x=batch_size),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
attn_output = ttnn.to_memory_config(attn_output, sdpa_output_memcfg)

# Use nlp_concat_heads_decode for direct output (no permute needed)
# Input: [1, batch, 32, head_dim] (HEIGHT_SHARDED)
# Output: [1, 1, batch, num_heads * head_dim]
attn_output = ttnn.experimental.nlp_concat_heads_decode(
    attn_output,
    num_heads=self.num_heads,
)
```

### Key Points

1. **Shard shape**: `(32, head_dim)` - MUST be exactly 32 for padded_heads (kernel requirement)
2. **Core grid**: `y=1, x=batch_size` - one core per batch entry
3. **Strategy**: HEIGHT sharding - each core holds one batch's attention heads

### Bailing Model Specifics

For Bailing (from the attention module):
- `num_heads` = 16 (attention heads)
- `num_kv_heads` = 4 (KV heads)
- `head_dim` = 128

So the shard config will be:
- `shape=(32, 128)` - 32 padded heads, 128 head_dim
- `core_grid=CoreGrid(y=1, x=batch_size)` - batch_size cores

## Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
   - `_forward_decode_paged()` in `TTNNBailingMoEAttention` class (line ~2808-2821)

## Testing

After the fix, run:
```bash
pytest models/experimental/tt_symbiote/tests/test_bailing_attention_t3k.py -v
```

## Status

- [ ] Implement HEIGHT_SHARDED conversion before nlp_concat_heads_decode
- [ ] Verify SDPA output shape matches expected [1, batch, 32, head_dim]
- [ ] Test end-to-end decode flow

## Related Files

- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` - Bailing attention implementation
- `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py` - RoPE implementation (already fixed)
- `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/attention.py` - Reference implementation
- `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/model_config.py` - Memory config patterns
- `/home/ttuser/salnahari/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp` - Kernel requirements
