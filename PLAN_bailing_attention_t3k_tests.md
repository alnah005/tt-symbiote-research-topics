# Plan: Fix Bailing Attention T3K Decode Tests

## Problem Description

The Bailing Attention T3K tests are failing with the following results:

| Test | Status | Details |
|------|--------|---------|
| test_weight_sharing_verification | PASSED | - |
| test_full_forward_prefill_with_paged_cache | FAILED | PCC: 0.967932 (below 0.99) |
| test_full_forward_decode_with_paged_cache | FAILED | RuntimeError: head_dim % TILE_WIDTH == 0 |
| test_multi_token_decode_sequence | FAILED | RuntimeError: head_dim % TILE_WIDTH == 0 |
| test_integration_with_model_layer | FAILED | PCC: 0.958638 (below 0.99) |

The decode tests (`test_full_forward_decode_with_paged_cache` and `test_multi_token_decode_sequence`) fail with `RuntimeError: head_dim % TILE_WIDTH == 0` which is a misleading error message. The actual root cause is that the `rotary_embedding_llama` kernel in decode mode requires HEIGHT_SHARDED memory layout for all input tensors, but the current implementation passes tensors in DRAM (interleaved) layout.

## Root Cause Analysis

### Kernel Requirements

The `rotary_embedding_llama` kernel has strict memory layout requirements when `is_decode_mode=True`:

From `/home/ttuser/salnahari/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp` (lines 78-96):

```cpp
if (operation_attributes.is_decode_mode) {  // Decode mode validation
    uint32_t seq_len = input_tensor.logical_shape()[0];
    TT_FATAL(
        seq_len == 1,
        "rotary_embedding_llama currently only supports sharded inputs in decode mode, and therefore, seq_len (in "
        "dim 0) must be 1.");

    TT_FATAL(
        (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
        "Sharded inputs for RoPE must be HEIGHT_SHARDED.");
    TT_FATAL(
        (cos.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
        "cos tensor for RoPE must be HEIGHT_SHARDED.");
    TT_FATAL(
        (sin.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
        "sin tensor for RoPE must be HEIGHT_SHARDED.");
    TT_FATAL(
        (trans_mat.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED),
        "transformation matrix for RoPE must be HEIGHT_SHARDED.");
```

**Requirements for decode mode:**
1. Input tensor (Q/K): HEIGHT_SHARDED in L1
2. cos tensor: HEIGHT_SHARDED in L1
3. sin tensor: HEIGHT_SHARDED in L1
4. trans_mat: HEIGHT_SHARDED in L1
5. seq_len must be 1 (single token decode)

### Current Implementation Issue

The current `apply_partial_rope_decode()` function in `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py` passes tensors without HEIGHT_SHARDED memory config:

```python
def apply_partial_rope_decode(
    query_states: ttnn.Tensor,
    key_states: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    trans_mat: ttnn.Tensor,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    # ... slicing logic ...
    q_rot_embedded = ttnn.experimental.rotary_embedding_llama(
        q_rot, cos, sin, trans_mat, is_decode_mode=True  # FAILS: tensors not HEIGHT_SHARDED
    )
```

The tensors coming into this function are in DRAM interleaved layout, which violates the kernel's HEIGHT_SHARDED requirement.

### Reference Implementation (tt_transformers)

The tt_transformers attention module handles this correctly in `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/attention.py`:

1. **QKV Head Creation with HEIGHT_SHARDED output:**
```python
q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, v_heads_1BKD = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=self.n_local_heads,
    num_kv_heads=self.n_local_kv_heads,
    memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE, self.prefetcher),  # L1_HEIGHT_SHARDED
)
```

2. **Memory config for QKV heads (from model_config.py):**
```python
def get_attn_create_head_output_mem_config(self, mode: Mode, prefetcher: Prefetcher = None):
    if mode == Mode.DECODE:
        # ...
        return ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG  # HEIGHT_SHARDED by default
```

3. **Transformation matrix is pre-created with HEIGHT_SHARDED layout**

4. **rot_mats (cos/sin) come from a RotarySetup class that provides HEIGHT_SHARDED tensors for decode**

## Step-by-Step Implementation Plan

### Step 1: Update `BailingRotarySetup` to Support HEIGHT_SHARDED Decode Tensors

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py`

**Changes:**

1. Add a method to create HEIGHT_SHARDED memory config for decode tensors:

```python
def _create_decode_height_sharded_memcfg(self, batch_size: int) -> ttnn.MemoryConfig:
    """Create HEIGHT_SHARDED memory config for decode mode tensors.

    For decode mode, tensors are sharded across cores with one batch element per core.
    Shard shape is [TILE_SIZE, rotary_dim] for cos/sin tensors.
    """
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, self.rotary_dim),
        core_grid=ttnn.CoreGrid(y=1, x=batch_size),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
```

2. Modify `get_cos_sin_for_decode()` to return HEIGHT_SHARDED tensors:

```python
def get_cos_sin_for_decode(
    self,
    position_ids: Union[torch.Tensor, ttnn.Tensor],
    batch_size: int,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Get cos/sin embeddings for decode at specified positions.

    Returns HEIGHT_SHARDED tensors compatible with rotary_embedding_llama decode mode.
    """
    # ... existing embedding lookup logic ...

    # Convert to HEIGHT_SHARDED for decode mode
    decode_memcfg = self._create_decode_height_sharded_memcfg(batch_size)
    cos = ttnn.to_memory_config(cos, decode_memcfg)
    sin = ttnn.to_memory_config(sin, decode_memcfg)

    return cos, sin
```

3. Add a method to get HEIGHT_SHARDED transformation matrix for decode:

```python
def get_trans_mat_height_sharded(self, batch_size: int) -> ttnn.Tensor:
    """Get transformation matrix in HEIGHT_SHARDED format for decode.

    The trans_mat must be HEIGHT_SHARDED with shard shape [TILE_SIZE, TILE_SIZE].
    """
    # Create HEIGHT_SHARDED memory config for trans_mat
    trans_mat_memcfg = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        core_grid=ttnn.CoreGrid(y=1, x=batch_size),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.to_memory_config(self.trans_mat_decode, trans_mat_memcfg)
```

### Step 2: Update `apply_partial_rope_decode()` to Accept Memory Config

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py`

**Changes:**

Modify the function to convert tensors to HEIGHT_SHARDED before calling the kernel:

```python
def apply_partial_rope_decode(
    query_states: ttnn.Tensor,
    key_states: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    trans_mat: ttnn.Tensor,
    decode_memcfg: ttnn.MemoryConfig = None,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Apply partial RoPE for decode mode with [1, B, H, D] format tensors.

    IMPORTANT: rotary_embedding_llama with is_decode_mode=True requires ALL tensors
    to be HEIGHT_SHARDED. If decode_memcfg is provided, tensors will be converted.

    Args:
        query_states: Query tensor [1, batch, num_heads, head_dim]
        key_states: Key tensor [1, batch, num_kv_heads, head_dim]
        cos: Cosine position embeddings [1, batch, 1, rotary_dim] (HEIGHT_SHARDED)
        sin: Sine position embeddings [1, batch, 1, rotary_dim] (HEIGHT_SHARDED)
        trans_mat: Transformation matrix [1, 1, 32, 32] (HEIGHT_SHARDED)
        decode_memcfg: HEIGHT_SHARDED memory config for Q/K tensors

    Returns:
        Tuple of (rotated_query, rotated_key) with same shapes as input
    """
    seq_len, batch, num_heads, head_dim = query_states.shape
    rotary_dim = cos.shape[-1]

    # Verify HEIGHT_SHARDED requirement
    if decode_memcfg is None:
        # Create default HEIGHT_SHARDED config
        decode_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim),
            core_grid=ttnn.CoreGrid(y=1, x=batch),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    if rotary_dim < head_dim:
        # Split Q/K into rotary and pass-through portions
        q_rot = ttnn.slice(query_states, [0, 0, 0, 0], [seq_len, batch, num_heads, rotary_dim])
        q_pass = ttnn.slice(query_states, [0, 0, 0, rotary_dim], [seq_len, batch, num_heads, head_dim])
        k_rot = ttnn.slice(key_states, [0, 0, 0, 0], [seq_len, batch, key_states.shape[2], rotary_dim])
        k_pass = ttnn.slice(key_states, [0, 0, 0, rotary_dim], [seq_len, batch, key_states.shape[2], head_dim])

        # Convert to HEIGHT_SHARDED for rotary_embedding_llama
        q_rot_sharded = ttnn.to_memory_config(q_rot, decode_memcfg)
        k_rot_sharded = ttnn.to_memory_config(k_rot, decode_memcfg)

        # Apply RoPE to rotary portion only using decode mode
        q_rot_embedded = ttnn.experimental.rotary_embedding_llama(
            q_rot_sharded, cos, sin, trans_mat, is_decode_mode=True
        )
        k_rot_embedded = ttnn.experimental.rotary_embedding_llama(
            k_rot_sharded, cos, sin, trans_mat, is_decode_mode=True
        )

        # Convert back to interleaved for concat
        q_rot_embedded = ttnn.to_memory_config(q_rot_embedded, ttnn.DRAM_MEMORY_CONFIG)
        k_rot_embedded = ttnn.to_memory_config(k_rot_embedded, ttnn.DRAM_MEMORY_CONFIG)

        # Concatenate back
        query_rotated = ttnn.concat([q_rot_embedded, q_pass], dim=-1)
        key_rotated = ttnn.concat([k_rot_embedded, k_pass], dim=-1)
    else:
        # Full rotation - convert to HEIGHT_SHARDED
        query_sharded = ttnn.to_memory_config(query_states, decode_memcfg)
        key_sharded = ttnn.to_memory_config(key_states, decode_memcfg)

        query_rotated = ttnn.experimental.rotary_embedding_llama(
            query_sharded, cos, sin, trans_mat, is_decode_mode=True
        )
        key_rotated = ttnn.experimental.rotary_embedding_llama(
            key_sharded, cos, sin, trans_mat, is_decode_mode=True
        )

        # Convert back to interleaved
        query_rotated = ttnn.to_memory_config(query_rotated, ttnn.DRAM_MEMORY_CONFIG)
        key_rotated = ttnn.to_memory_config(key_rotated, ttnn.DRAM_MEMORY_CONFIG)

    return query_rotated, key_rotated
```

### Step 3: Update `TTNNBailingMoEAttention._forward_decode_paged()`

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`

**Changes:**

1. Ensure cos/sin from `BailingRotarySetup.get_cos_sin_for_decode()` are HEIGHT_SHARDED
2. Ensure trans_mat is HEIGHT_SHARDED
3. Pass proper memory config to `apply_partial_rope_decode()`

```python
def _forward_decode_paged(
    self,
    hidden_states: ttnn.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[ttnn.Tensor],
    past_key_values: "TTNNPagedAttentionKVCache",
    cache_position: Optional[torch.LongTensor],
) -> tuple:
    """Decode path using paged attention with on-device KV cache (T3K mode only)."""
    batch_size, seq_length = hidden_states.shape[0], hidden_states.shape[1]

    # ... existing QKV projection and reshape code ...

    # Apply QK normalization
    query_states, key_states = self._apply_qk_norm(query_states, key_states)

    # Apply RoPE - position embeddings should be HEIGHT_SHARDED from BailingRotarySetup
    cos, sin = position_embeddings

    # Ensure HEIGHT_SHARDED for decode mode
    # Create HEIGHT_SHARDED config for Q/K
    decode_memcfg = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, self.head_dim),
        core_grid=ttnn.CoreGrid(y=1, x=batch_size),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Ensure cos/sin are HEIGHT_SHARDED
    if cos.memory_config().memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        cos_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, cos.shape[-1]),
            core_grid=ttnn.CoreGrid(y=1, x=batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cos = ttnn.to_memory_config(cos, cos_memcfg)
        sin = ttnn.to_memory_config(sin, cos_memcfg)

    # Ensure trans_mat is HEIGHT_SHARDED
    if self._decode_trans_mat.memory_config().memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        trans_mat_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=ttnn.CoreGrid(y=1, x=batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        trans_mat = ttnn.to_memory_config(self._decode_trans_mat, trans_mat_memcfg)
    else:
        trans_mat = self._decode_trans_mat

    # Ensure query/key states are BFLOAT16 for RoPE compatibility
    query_states = ensure_ttnn_bfloat16(query_states)
    key_states = ensure_ttnn_bfloat16(key_states)

    # Apply RoPE using decode-specific function with HEIGHT_SHARDED tensors
    query_states, key_states = apply_partial_rope_decode(
        query_states, key_states, cos, sin, trans_mat, decode_memcfg
    )

    # ... rest of the decode path ...
```

### Step 4: Pre-compute HEIGHT_SHARDED Trans Mat at Initialization

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`

**Changes:**

In `TTNNBailingMoEAttention.__init__()` or `move_weights_to_device_impl()`, pre-compute the HEIGHT_SHARDED transformation matrix to avoid repeated conversions:

```python
def move_weights_to_device_impl(self):
    # ... existing weight movement code ...

    # Create decode transformation matrix with HEIGHT_SHARDED for max batch size
    trans_mat_torch = _get_rotation_transformation_mat(ttnn.TILE_SIZE)
    mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None

    self._decode_trans_mat = ttnn.from_torch(
        trans_mat_torch.to(torch.bfloat16),
        device=self.device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Start in DRAM
        mesh_mapper=mesh_mapper,
    )

    # Pre-create HEIGHT_SHARDED version for common batch sizes
    self._decode_trans_mat_sharded = {}
    for batch_size in [1, 2, 4, 8, 16, 32]:
        trans_mat_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=ttnn.CoreGrid(y=1, x=batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._decode_trans_mat_sharded[batch_size] = ttnn.to_memory_config(
            self._decode_trans_mat, trans_mat_memcfg
        )
```

## Files to Modify

1. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py`
   - Update `apply_partial_rope_decode()` to convert tensors to HEIGHT_SHARDED
   - Update `BailingRotarySetup.get_cos_sin_for_decode()` to return HEIGHT_SHARDED tensors
   - Add helper methods for HEIGHT_SHARDED memory config creation

2. `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
   - Update `TTNNBailingMoEAttention._forward_decode_paged()` to ensure all RoPE inputs are HEIGHT_SHARDED
   - Pre-compute HEIGHT_SHARDED transformation matrices at initialization

## Success Criteria

1. **Decode tests pass:** `test_full_forward_decode_with_paged_cache` and `test_multi_token_decode_sequence` should complete without RuntimeError
2. **PCC targets met:**
   - Decode tests: PCC >= 0.99 with PyTorch reference
   - Prefill tests: PCC >= 0.99 with PyTorch reference (these may need separate investigation)
3. **No performance regression:** The decode path should maintain acceptable latency with HEIGHT_SHARDED conversions

## Testing Plan

1. Run the existing test suite:
   ```bash
   pytest models/experimental/tt_symbiote/tests/test_bailing_attention_t3k.py -v
   ```

2. Specifically test decode with different batch sizes:
   ```bash
   pytest models/experimental/tt_symbiote/tests/test_bailing_attention_t3k.py::test_full_forward_decode_with_paged_cache -v
   pytest models/experimental/tt_symbiote/tests/test_bailing_attention_t3k.py::test_multi_token_decode_sequence -v
   ```

3. Verify memory config properties in debug mode to ensure HEIGHT_SHARDED is applied correctly

## Additional Notes

### Why HEIGHT_SHARDED is Required

The `rotary_embedding_llama` kernel in decode mode is optimized for single-token generation where:
- Each batch element is processed independently on a separate core
- HEIGHT_SHARDED layout naturally maps batch elements to cores
- The kernel assumes shard_shape[0] == TILE_HEIGHT for efficient tile processing

### Prefill PCC Issues

The prefill tests (`test_full_forward_prefill_with_paged_cache` and `test_integration_with_model_layer`) have PCC below 0.99 (0.967932 and 0.958638 respectively). These are likely separate issues related to:
- Accumulation precision in attention computation
- KV cache fill operations
- Different from the decode HEIGHT_SHARDED issue

These should be investigated separately after the decode tests are fixed.
