# PLAN: Fix Ling-mini-2.0 Garbled Text Output on T3K

**Date:** 2026-03-26
**Status:** Ready for Implementation
**Target File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
**Target Class:** `TTNNBailingMoEAttention`
**Target Method:** `_forward_decode_paged()` (lines 2607-2759)

---

## 1. Problem Statement

The Ling-mini-2.0 model produces garbled/incorrect text during autoregressive decoding on T3K (8-device Wormhole mesh). The root cause is a fundamentally broken decode path in `_forward_decode_paged()` that diverges from the proven TT-Transformers pattern in multiple compounding ways.

---

## 2. Root Causes (Ordered by Severity)

### RC1: No `nlp_create_qkv_heads_decode` -- Manual Reshape/Permute Instead

**Current code (lines 2629-2658):**
```python
# After _project_qkv_t3k returns [batch, seq, num_heads*head_dim]
query_states = ttnn.reshape(query_states, (batch_size, seq_length, self.num_heads, self.head_dim))
# ... same for K, V
query_states = ttnn.permute(query_states, (0, 2, 1, 3))  # -> [B, H, S=1, D]
# ... QK norm and RoPE in [B, H, S, D] format ...
query_states = ttnn.permute(query_states, (2, 0, 1, 3))  # -> [S=1, B, H, D]
```

**Why this breaks:**
- The kernel `nlp_create_qkv_heads_decode` produces `[1, B, H, D]` with correct HEIGHT_SHARDED L1 memory config that downstream paged kernels expect.
- Manual reshape+permute produces tensors in DRAM INTERLEAVED with no sharding. The double permute also introduces BF16 precision loss (each permute is a data movement op with potential rounding).

**TT-Transformers reference (attention.py line 554):**
```python
q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, v_heads_1BKD = (
    ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=self.n_local_heads,
        num_kv_heads=self.n_local_kv_heads,
        memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE),
    )
)
```

### RC2: Prefill-Style RoPE in Decode Path

**Current code (line 2652):**
```python
query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin)
```

This calls `TTNNRotaryPositionEmbedding.forward()` which uses `ttnn.experimental.rotary_embedding()` -- the **prefill** kernel. This kernel expects `[B, H, S, D]` format tensors and cos/sin with shape `[1, 1, S, rotary_dim]`.

The decode kernel `ttnn.experimental.rotary_embedding_llama()` with `is_decode_mode=True` expects:
- Input: `[1, B, H, D]` (HEIGHT_SHARDED in L1, one shard per batch element)
- cos/sin: `[1, B, 1, head_dim]` (gathered via embedding lookup for each position)
- trans_mat: `[1, 1, TILE_SIZE, TILE_SIZE]` (HEIGHT_SHARDED, replicated per batch core)

Using the wrong RoPE kernel means the rotation is computed differently, producing wrong positional information at every decode step.

### RC3: Partial RoPE Requires Splitting Before `rotary_embedding_llama`

Ling uses `partial_rotary_factor=0.5`, so `rotary_dim=64` while `head_dim=128`. The `rotary_embedding_llama` kernel applies rotation to the **full** last dimension. For partial RoPE:
1. Split Q/K along dim=-1 into `[:64]` (rotate) and `[64:]` (pass-through)
2. Apply `rotary_embedding_llama` only to the rotary portion
3. Concatenate the results

**Critical detail:** The cos/sin from `BailingRotarySetup.get_cos_sin_for_decode()` already have shape `[1, batch, 1, rotary_dim=64]`. These must be used with the rotary portion only (not padded to head_dim).

### RC4: `_to_replicated()` Host Round-Trip

**Current code (lines 2688-2690):**
```python
query_states = self._to_replicated(query_states)
key_states = self._to_replicated(key_states)
value_states = self._to_replicated(value_states)
```

This calls `to_replicated_topology()` which does `ttnn.to_torch()` -> `ttnn.from_torch()` for each tensor. For decode, the tensors are small (`[1, 1, 16, 128]` for Q), so the overhead is "acceptable" per the docstring, but:
- It moves data from device -> host -> device on every decode step
- It can introduce subtle precision changes
- It is fundamentally unnecessary if the QKV projection outputs have correct topology from the start

**Why the topology mismatch exists:** `_project_qkv_t3k()` calls `_maybe_all_gather()` which produces "concatenated" topology metadata. Paged attention kernels require "replicated" topology. In TT-Transformers, the weights are already per-device (tensor parallel), so no all-gather is needed and the topology is naturally correct.

**Fix approach:** Since tt-symbiote replicates all heads on all devices, the data is already identical after all_gather. We can use `ttnn.to_memory_config()` with replicated config, or restructure to avoid all_gather entirely for the decode path (where tensors are tiny).

### RC5: Incorrect Sharding Before `paged_update_cache`

**Current code (lines 2693-2704):**
```python
core_grid = ttnn.CoreGrid(y=1, x=batch_size)  # batch_size=1 -> 1 core
shard_cfg = ttnn.create_sharded_memory_config(
    shape=(shard_h, self.head_dim),  # (32, 128)
    core_grid=core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
)
key_states = ttnn.to_memory_config(key_states, shard_cfg)
```

With batch_size=1, this creates 1 core but the tensor has 4 KV heads (padded to 32 tiles), requiring multiple shards. This is a shard count vs core count mismatch.

**TT-Transformers approach:** The K/V tensors from `nlp_create_qkv_heads_decode` already have the correct HEIGHT_SHARDED L1 memory config. No manual sharding is needed.

---

## 3. Implementation Plan

### Step 1: Create Fused QKV Tensor for `nlp_create_qkv_heads_decode`

The current code projects Q, K, V separately via `_project_qkv_t3k()`, producing three separate tensors. `nlp_create_qkv_heads_decode` expects a single fused tensor of shape `[1, 1, B, (num_heads + 2*num_kv_heads) * head_dim]`.

**Option A (recommended):** Keep separate projections but concatenate for the decode kernel:

**Lines 2629-2635 -- REPLACE with:**
```python
# Unwrap TorchTTNNTensor if needed
if hasattr(query_states, 'to_ttnn'):
    query_states = query_states.to_ttnn
if hasattr(key_states, 'to_ttnn'):
    key_states = key_states.to_ttnn
if hasattr(value_states, 'to_ttnn'):
    value_states = value_states.to_ttnn

# Fuse Q/K/V for nlp_create_qkv_heads_decode: [B, S, Q_dim + K_dim + V_dim]
xqkv_fused = ttnn.concat([query_states, key_states, value_states], dim=-1)
# Reshape to [1, 1, B, fused_dim] as expected by the kernel
fused_dim = xqkv_fused.shape[-1]
xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, batch_size, fused_dim))

# Ensure bfloat16 (required by nlp_create_qkv_heads_decode)
if xqkv_fused.dtype != ttnn.bfloat16:
    xqkv_fused = ttnn.typecast(xqkv_fused, ttnn.bfloat16)

# Split into heads using the decode kernel -> outputs [1, B, H, D]
query_states, key_states, value_states = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=self.num_heads,       # 16
    num_kv_heads=self.num_kv_heads, # 4
    memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
)
ttnn.deallocate(xqkv_fused)
```

**Tensor shapes after this step:**
| Tensor | Shape | Memory | Format |
|--------|-------|--------|--------|
| query_states | `[1, B, 16, 128]` | L1 HEIGHT_SHARDED | TILE |
| key_states | `[1, B, 4, 128]` | L1 HEIGHT_SHARDED | TILE |
| value_states | `[1, B, 4, 128]` | L1 HEIGHT_SHARDED | TILE |

### Step 2: Apply QK Norm in [1, B, H, D] Format

The existing `_apply_qk_norm()` method (lines 2442-2503) already handles decode mode `[1, B, H, D]` format. The detection logic at line 2464 checks `q_shape[0] == 1`. This should work as-is.

**However**, the reshapes inside `_apply_qk_norm` for decode mode may break HEIGHT_SHARDED memory layout. We need to move to interleaved before reshaping.

**Lines 2642-2643 -- REPLACE with:**
```python
# Move to interleaved for QK norm (reshape ops don't work on sharded tensors)
query_states = ttnn.to_memory_config(query_states, ttnn.L1_MEMORY_CONFIG)
key_states = ttnn.to_memory_config(key_states, ttnn.L1_MEMORY_CONFIG)

# Apply QK normalization in [1, B, H, D] format
query_states, key_states = self._apply_qk_norm(query_states, key_states)
```

**Tensor shapes after this step:** Unchanged `[1, B, 16, 128]` and `[1, B, 4, 128]`, now in L1 INTERLEAVED.

### Step 3: Apply Partial RoPE Using `rotary_embedding_llama` in Decode Mode

**DELETE lines 2645-2652** (the current prefill-style RoPE call).

**ADD new method `_apply_partial_rope_decode`:**

```python
def _apply_partial_rope_decode(
    self,
    query_states: ttnn.Tensor,
    key_states: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    trans_mat: ttnn.Tensor,
) -> tuple:
    """Apply partial RoPE for decode mode on [1, B, H, D] format tensors.

    For partial_rotary_factor < 1.0, splits the head dimension into rotary
    and pass-through portions, applies rotary_embedding_llama to the rotary
    portion only, then concatenates.

    Args:
        query_states: [1, B, num_heads, head_dim]
        key_states: [1, B, num_kv_heads, head_dim]
        cos: [1, B, 1, rotary_dim] from BailingRotarySetup.get_cos_sin_for_decode
        sin: [1, B, 1, rotary_dim] from BailingRotarySetup.get_cos_sin_for_decode
        trans_mat: [1, 1, TILE_SIZE, TILE_SIZE] transformation matrix

    Returns:
        Tuple of (query_rotated, key_rotated) in [1, B, H, D] format
    """
    rotary_dim = cos.shape[-1]  # 64 for Ling
    head_dim = query_states.shape[-1]  # 128

    if rotary_dim < head_dim:
        # Split into rotary and pass-through portions along dim=-1
        q_rot = query_states[:, :, :, :rotary_dim]   # [1, B, 16, 64]
        q_pass = query_states[:, :, :, rotary_dim:]   # [1, B, 16, 64]
        k_rot = key_states[:, :, :, :rotary_dim]      # [1, B, 4, 64]
        k_pass = key_states[:, :, :, rotary_dim:]      # [1, B, 4, 64]

        # Apply rotary_embedding_llama to rotary portion
        q_rot = ttnn.experimental.rotary_embedding_llama(
            q_rot, cos, sin, trans_mat, is_decode_mode=True
        )
        k_rot = ttnn.experimental.rotary_embedding_llama(
            k_rot, cos, sin, trans_mat, is_decode_mode=True
        )

        # Concatenate rotated and pass-through
        query_states = ttnn.concat([q_rot, q_pass], dim=-1)  # [1, B, 16, 128]
        key_states = ttnn.concat([k_rot, k_pass], dim=-1)    # [1, B, 4, 128]
    else:
        # Full rotation (partial_rotary_factor == 1.0)
        query_states = ttnn.experimental.rotary_embedding_llama(
            query_states, cos, sin, trans_mat, is_decode_mode=True
        )
        key_states = ttnn.experimental.rotary_embedding_llama(
            key_states, cos, sin, trans_mat, is_decode_mode=True
        )

    return query_states, key_states
```

**In `_forward_decode_paged`, REPLACE lines 2645-2652 with:**
```python
# Ensure bfloat16 for RoPE
query_states = ensure_ttnn_bfloat16(query_states)
key_states = ensure_ttnn_bfloat16(key_states)

# Apply partial RoPE using decode kernel
cos, sin = position_embeddings
query_states, key_states = self._apply_partial_rope_decode(
    query_states, key_states, cos, sin, self._decode_trans_mat
)
```

**Tensor shapes after this step:** Unchanged `[1, B, 16, 128]` and `[1, B, 4, 128]`.

### Step 4: Pre-compute Transformation Matrix During Weight Loading

**In `move_weights_to_device_impl()` (after line 2439), ADD:**

```python
# Create transformation matrix for decode RoPE
# This follows TT-Transformers RotarySetup pattern (rope.py line 469)
from models.common.tensor_utils import get_rot_transformation_mat
trans_mat_torch = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)  # [1, 1, 32, 32]
# For batch_size=1, we need it replicated per batch core
# trans_mat is used by rotary_embedding_llama which expects HEIGHT_SHARDED in L1
mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
self._decode_trans_mat = ttnn.from_torch(
    trans_mat_torch,
    device=self.device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=mesh_mapper,
)
```

**IMPORTANT NOTE on trans_mat sharding:** In TT-Transformers, the trans_mat is HEIGHT_SHARDED across batch cores (one shard of `[32, 32]` per batch element). For batch_size=1, this means 1 core with shard `[32, 32]`. The `rotary_embedding_llama` kernel reads the trans_mat from L1 on each core. For batch_size=1 in DRAM_MEMORY_CONFIG, the kernel should still work (it can read from DRAM), but if sharding is required:

```python
batch_grid = ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(8, 8), row_wise=True)
trans_mat_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
self._decode_trans_mat = ttnn.from_torch(
    trans_mat_torch.repeat(1, 1, batch_size, 1),
    device=self.device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=trans_mat_mem,
    mesh_mapper=mesh_mapper,
)
```

Test with DRAM first; if `rotary_embedding_llama` fails, switch to HEIGHT_SHARDED.

### Step 5: Remove `_to_replicated()` Host Round-Trip

**DELETE lines 2688-2690:**
```python
query_states = self._to_replicated(query_states)
key_states = self._to_replicated(key_states)
value_states = self._to_replicated(value_states)
```

**Why it's safe to remove:** After Step 1, the tensors go through `nlp_create_qkv_heads_decode` which produces tensors with the correct on-device topology. The QKV projections are produced from replicated inputs (all_gather in `_project_qkv_t3k`), and `nlp_create_qkv_heads_decode` preserves the topology metadata.

**If topology is still wrong after testing:** Instead of the host round-trip, use a lightweight device-side approach. Check if `ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)` followed by proper sharding fixes the metadata. Alternatively, restructure the decode QKV projection to use `ReplicateTensorToMesh` from the start (skip all_gather for decode since tensors are tiny).

### Step 6: Remove Manual Sharding Before `paged_update_cache`

**DELETE lines 2693-2704:**
```python
tile_size = 32
shard_h = ((self.num_kv_heads + tile_size - 1) // tile_size) * tile_size
core_grid = ttnn.CoreGrid(y=1, x=batch_size)
shard_cfg = ttnn.create_sharded_memory_config(
    shape=(shard_h, self.head_dim),
    core_grid=core_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
key_states = ttnn.to_memory_config(key_states, shard_cfg)
value_states = ttnn.to_memory_config(value_states, shard_cfg)
```

**Why:** When K/V come from `nlp_create_qkv_heads_decode` with `L1_HEIGHT_SHARDED_MEMORY_CONFIG`, they already have the correct sharding for `paged_update_cache`. TT-Transformers does NOT apply any manual sharding before `paged_update_cache`.

**If `paged_update_cache` still requires sharding:** Use the correct formula from the test suite (`test_paged_update_cache.py`):
```python
num_cores = batch_size  # 1 core per batch element
compute_grid = self.device.compute_with_storage_grid_size()
shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
shard_spec = ttnn.ShardSpec(
    shard_grid,
    [key_states.volume() // key_states.padded_shape[-1] // num_cores, key_states.padded_shape[-1]],
    ttnn.ShardOrientation.ROW_MAJOR,
)
kv_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
```

### Step 7: Keep Existing SDPA Decode and Concat Heads (Lines 2715-2759)

The code after the KV cache update is mostly correct:
- `paged_sdpa_decode` is called correctly with Q shape `[1, B, 16, 128]`
- `nlp_concat_heads_decode` is used for output
- Output slice and reshape are correct

**One change needed (line 2729-2736):** The HEIGHT_SHARDED conversion for SDPA output before `nlp_concat_heads_decode` uses `CoreGrid(y=1, x=batch_size)`. For batch_size=1 this gives 1 core. With `pnh=32` (16 Q heads padded), the shard is `(32, 128)` on 1 core. This should be fine for batch_size=1 since the full output fits in one shard.

**Verify:** Output from `paged_sdpa_decode` has shape `[1, B, pnh=32, 128]`. For batch_size=1, total volume = 1*1*32*128 = 4096 elements. One shard of (32, 128) = 4096 elements on 1 core. This is valid.

---

## 4. Complete Tensor Shape Flow (After Fix)

| Step | Operation | Q Shape | K/V Shape | Memory Config |
|------|-----------|---------|-----------|---------------|
| 1 | `_project_qkv_t3k` | `[1, 1, 2048]` | `[1, 1, 512]` | DRAM INTERLEAVED |
| 2 | `ttnn.concat` + reshape | `[1, 1, 1, 3072]` fused | - | DRAM INTERLEAVED |
| 3 | `nlp_create_qkv_heads_decode` | `[1, 1, 16, 128]` | `[1, 1, 4, 128]` | L1 HEIGHT_SHARDED |
| 4 | to_memory_config (interleaved) | `[1, 1, 16, 128]` | `[1, 1, 4, 128]` | L1 INTERLEAVED |
| 5 | `_apply_qk_norm` | `[1, 1, 16, 128]` | `[1, 1, 4, 128]` | L1 INTERLEAVED |
| 6 | `_apply_partial_rope_decode` | `[1, 1, 16, 128]` | `[1, 1, 4, 128]` | L1/DRAM INTERLEAVED |
| 7 | `paged_update_cache` | - | `[1, 1, 4, 128]` | (auto from step 3 or re-shard) |
| 8 | `paged_sdpa_decode` | `[1, 1, 32*, 128]` out | - | DRAM (from kernel) |
| 9 | `nlp_concat_heads_decode` | `[1, 1, 1, 2048]` | - | L1 HEIGHT_SHARDED |
| 10 | `dense` projection | `[1, 1, 1, 2048]` | - | output |

*\*pnh=32 is the padded head count (ceil(16/32)*32)*

---

## 5. Position Embedding Pipeline (Decode)

The `BailingRotarySetup` class (rope.py line 411) already provides the correct interface:

```
BailingRotarySetup.get_cos_sin_for_decode(position_ids)
  -> cos: [1, batch, 1, rotary_dim=64]  (TILE_LAYOUT, DRAM, replicated)
  -> sin: [1, batch, 1, rotary_dim=64]  (TILE_LAYOUT, DRAM, replicated)
```

These are passed as `position_embeddings = (cos, sin)` to the attention forward.

For `rotary_embedding_llama` with `is_decode_mode=True`:
- cos/sin shape `[1, B, 1, rotary_dim]` is correct
- The kernel broadcasts cos/sin across the head dimension internally
- The trans_mat `[1, 1, 32, 32]` handles the dimension pairing

**No changes needed to BailingRotarySetup or the position embedding pipeline.**

---

## 6. Changes to `_forward_decode_paged` (Summary)

### Lines to DELETE:
- **2633-2635**: Manual reshape to `[B, S, H, D]`
- **2637-2640**: Permute to `[B, H, S, D]`
- **2654-2658**: Permute to `[S, B, H, D]`
- **2688-2690**: `_to_replicated()` calls
- **2693-2704**: Manual sharding before `paged_update_cache`

### Lines to ADD/REPLACE:
- **After 2630**: Concatenate Q/K/V, reshape, call `nlp_create_qkv_heads_decode`
- **2642-2643**: Move to L1 interleaved before QK norm
- **2645-2652**: Call `_apply_partial_rope_decode` instead of `_apply_partial_rope`

### New methods to ADD:
- `_apply_partial_rope_decode()` on the class

### Changes to `move_weights_to_device_impl`:
- Create `self._decode_trans_mat` transformation matrix

---

## 7. Potential Complications and Mitigations

### 7.1 Topology Metadata After nlp_create_qkv_heads_decode

**Risk:** The fused tensor from `ttnn.concat` after all_gather may have "concatenated" topology. `nlp_create_qkv_heads_decode` may not fix this.

**Mitigation:** Test first. If paged kernels fail with topology errors:
1. Apply `_to_replicated()` to the fused QKV tensor BEFORE `nlp_create_qkv_heads_decode` (one round-trip instead of three)
2. Or restructure decode QKV to skip all_gather entirely (project on each device, then replicate the result with `ReplicateTensorToMesh`)

### 7.2 Partial RoPE + rotary_embedding_llama Interaction

**Risk:** `rotary_embedding_llama` may not support tensors where last dimension != TILE_SIZE multiple. With `rotary_dim=64`, the split tensors have `[1, B, H, 64]` which IS a tile multiple (2 tiles). This should be fine.

**Mitigation:** If the kernel fails on the 64-dim tensors, pad to `head_dim=128` before RoPE and slice back afterward. This matches the approach used for prefill.

### 7.3 Memory Config After Slice/Concat in Partial RoPE

**Risk:** `ttnn.concat` may produce DRAM INTERLEAVED output. Subsequent `paged_update_cache` expects HEIGHT_SHARDED.

**Mitigation:** After RoPE, explicitly re-shard K/V to HEIGHT_SHARDED before `paged_update_cache`:
```python
# Re-shard K/V for paged_update_cache if needed
num_cores = batch_size
shard_grid = ttnn.num_cores_to_corerangeset(
    num_cores,
    self.device.compute_with_storage_grid_size(),
    True,
)
kv_vol = key_states.volume() // key_states.padded_shape[-1] // num_cores
kv_shard = ttnn.ShardSpec(
    shard_grid,
    [kv_vol, key_states.padded_shape[-1]],
    ttnn.ShardOrientation.ROW_MAJOR,
)
kv_mem = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    kv_shard,
)
key_states = ttnn.to_memory_config(key_states, kv_mem)
value_states = ttnn.to_memory_config(value_states, kv_mem)
```

### 7.4 GQA Padding in paged_sdpa_decode

**Risk:** Per the guide (`ch3_gqa_tensor_layout/head_axis_conventions.md`), with `nh=16` padded to `pnh=32` and `nkv=4`, the effective group_size in the kernel must be `32/nkv_padded`. If `nkv_padded` is independently set to 32, group_size becomes 1 (MHA instead of GQA).

**Current state:** `nlp_create_qkv_heads_decode` pads `nh` to 32 but should correctly set `nkv_padded = pnh / group_size = 32 / 4 = 8`. The paged KV cache must be allocated with `nkv_padded=8` KV heads.

**Verification:** Check that the `TTNNPagedAttentionKVCache` allocates with the correct number of KV heads. This is a separate concern from the attention code but must be validated.

---

## 8. Success Criteria

1. **Text generation produces coherent output** -- the model generates readable, contextually appropriate text (not garbled/random tokens)
2. **PCC between TTNN decode and PyTorch reference > 0.98** for the attention layer
3. **No runtime errors** -- no sharding mismatches, no shape assertion failures, no topology errors
4. **No host round-trips during decode** -- no `ttnn.to_torch()` calls in the hot path (or at most one for the fused QKV topology fix)
5. **Performance:** decode latency should improve (fewer permutes, fewer data movement ops)

---

## 9. Testing Strategy

### Test 1: Full Model Text Generation
```bash
pytest /home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v
```
Expected: Coherent text output matching reference model quality.

### Test 2: Attention Layer PCC
```bash
pytest /home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/attention/test_ling_attention_equivalence_t3k.py -v
```
Expected: PCC > 0.98 for decode, > 0.99 for prefill.

### Test 3: Component Tests
```bash
# QK norm
pytest /home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/attention/test_ling_qk_norm.py -v
# Partial RoPE
pytest /home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/attention/test_ling_partial_rope.py -v
```

### Test 4: Manual Validation
Run the model interactively and check that:
1. First token is correct
2. Subsequent tokens form coherent sentences
3. Output is deterministic across runs (same input -> same output)

---

## 10. Implementation Order

1. **Step 4** -- Create `_decode_trans_mat` in `move_weights_to_device_impl` (safe, additive change)
2. **Step 3** -- Add `_apply_partial_rope_decode` method (safe, additive change)
3. **Steps 1, 2, 5, 6, 7** -- Rewrite `_forward_decode_paged` body (the main change)
4. Run tests iteratively, fixing topology/sharding issues as they arise

---

## 11. Critical Files

| File | Purpose |
|------|---------|
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` | Main file to modify (`TTNNBailingMoEAttention` class) |
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py` | `BailingRotarySetup` -- provides cos/sin (no changes needed) |
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/tensor_utils.py` | `to_replicated_topology` utility (to be removed from hot path) |
| `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/attention.py` | Reference implementation (lines 554-644) |
| `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/rope.py` | Reference `RotarySetup` class (lines 383-620) |
| `/home/ttuser/salnahari/tt-metal/models/common/tensor_utils.py` | `get_rot_transformation_mat` utility |
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py` | End-to-end test |

---

## 12. Iteration 2 -- RoPE Decode Diagnosis (Root Cause of Garbled "aukee" Pattern)

**Date:** 2026-03-26
**Status:** Diagnosed -- Ready for Fix

### 12.1 Symptom

After implementing all changes from Sections 1-11 (nlp_create_qkv_heads_decode, partial RoPE decode, replicated topology fix, correct sharding), the model runs without crashes but produces repeating garbled tokens ("aukee" pattern). This strongly indicates every decode step receives identical or near-identical positional information, causing the model to get stuck in a loop.

### 12.2 Root Cause: THREE Compounding RoPE Issues

The investigation reveals **three independent bugs** that compound to produce completely wrong RoPE in decode mode:

#### Bug 1: HuggingFace Format vs Meta Format Mismatch (CRITICAL)

**The `rotary_embedding_llama` kernel expects cos/sin in Meta format (interleaved duplicate pairs), but the model receives HuggingFace format (sequential halves).**

The Ling model's `rotary_emb.forward()` (HuggingFace implementation) produces:
```
cos_hf[position=5, :8] = [0.2837, -0.9876, -0.5697, 0.1340, 0.5835, 0.8107, 0.9161, 0.9632]
```

The `rotary_embedding_llama` kernel expects Meta format:
```
cos_meta[position=5, :8] = [0.2837, 0.2837, -0.9876, -0.9876, -0.5697, -0.5697, 0.1340, 0.1340]
```

These are NOT the same. Meta format duplicates each frequency into adjacent pairs `[cos_0, cos_0, cos_1, cos_1, ...]` to match the `rotate_half` operation used by `rotary_embedding_llama`. HuggingFace format uses the standard `[cos_0, cos_1, ..., cos_n/2, cos_0, cos_1, ...]` layout.

**Evidence:** `BailingRotarySetup._compute_cos_sin_cache()` (rope.py lines 362-408) does the Meta permutation:
```python
cos = cos[:, : cos.shape[1] // 2]
cos = torch.stack((cos, cos), dim=-1).flatten(-2)  # Meta interleaved format
```
But `BailingRotarySetup` is **NEVER actually instantiated** in the production code path. The `test_ling_mini_2_0.py` test calls `model.generate()` which uses HuggingFace's internal `rotary_emb.forward()` that does NOT apply Meta permutation.

**Why the plan was wrong:** Section 5 stated "No changes needed to BailingRotarySetup or the position embedding pipeline." This assumed `BailingRotarySetup.get_cos_sin_for_decode()` was being called, but it is not. The position embeddings come from HuggingFace's native flow.

#### Bug 2: Wrong Shape -- 3D vs 4D (CRITICAL)

HuggingFace `rotary_emb.forward()` returns cos/sin with shape `[B, S, rotary_dim]` = `[1, 1, 64]` (3D).

The `rotary_embedding_llama` kernel in decode mode expects `[1, B, 1, rotary_dim]` = `[1, 1, 1, 64]` (4D).

The current code at line 2742 does:
```python
cos, sin = position_embeddings  # shape [1, 1, 64] from HuggingFace
```
Then at line 2574:
```python
rotary_dim = cos.shape[-1]  # 64 -- correct
```
But the 3D tensor is passed directly to `rotary_embedding_llama` which expects 4D. This will either silently misinterpret the dimensions or crash.

#### Bug 3: Missing HEIGHT_SHARDED Memory Config (MODERATE)

The `rotary_embedding_llama` kernel in decode mode requires ALL inputs to be HEIGHT_SHARDED (per the docstring at rope.py line 51-55 and confirmed by TT-Transformers RotarySetup pattern).

**TT-Transformers pattern (rope.py lines 614-630):**
```python
mem_config = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),
    core_grid=self.batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    ...
)
cos = ttnn.interleaved_to_sharded(cos, mem_config)
sin = ttnn.interleaved_to_sharded(sin, mem_config)
```

**Current tt-symbiote:** cos/sin are in DRAM_MEMORY_CONFIG (from HuggingFace -> `ttnn.from_torch`). The `_decode_trans_mat` is also in DRAM_MEMORY_CONFIG (line 2453). Q/K are in L1_MEMORY_CONFIG (L1 interleaved, not sharded).

The `_apply_partial_rope_decode` method passes these directly to `rotary_embedding_llama` without converting to HEIGHT_SHARDED. This may produce silent incorrect results or crash depending on the kernel implementation.

### 12.3 Why "aukee" Repeating Pattern Occurs

With wrong-format cos/sin (Bug 1), the rotation applied to Q and K at each decode step is mathematically incorrect. Since the format error is systematic (not random), each position gets a similarly wrong rotation, making all positions "look the same" to the attention mechanism. This causes:

1. Every decode step produces nearly identical attention outputs
2. The model samples the same high-probability token repeatedly
3. The token "aukee" emerges as the mode of the corrupted output distribution

### 12.4 Required Fix

The fix must address all three bugs. There are two approaches:

#### Approach A: Intercept and Convert HuggingFace cos/sin in the Decode Path (RECOMMENDED)

Add conversion logic at the beginning of `_forward_decode_paged()` to transform HuggingFace-format cos/sin into the format expected by `rotary_embedding_llama`:

```python
def _forward_decode_paged(self, hidden_states, position_embeddings, ...):
    ...
    # === Fix Bug 1 + Bug 2: Convert HF cos/sin to Meta format + correct shape ===
    cos_hf, sin_hf = position_embeddings  # [B, S, rotary_dim] HF format

    # Convert from HF format to Meta format (interleaved pairs)
    # HF: [cos_0, cos_1, cos_2, ...] -> Meta: [cos_0, cos_0, cos_1, cos_1, ...]
    cos_meta = cos_hf[..., :cos_hf.shape[-1] // 2]          # [B, S, rotary_dim/2]
    cos_meta = torch.stack((cos_meta, cos_meta), dim=-1)      # [B, S, rotary_dim/2, 2]
    cos_meta = cos_meta.flatten(-2)                            # [B, S, rotary_dim]

    sin_meta = sin_hf[..., :sin_hf.shape[-1] // 2]
    sin_meta = torch.stack((sin_meta, sin_meta), dim=-1)
    sin_meta = sin_meta.flatten(-2)

    # Reshape from [B, S, D] -> [1, B, 1, D] for decode mode
    # For decode, S=1, so [B, 1, D] -> [1, B, 1, D]
    B = cos_meta.shape[0]
    cos_decode = cos_meta.reshape(1, B, 1, -1)  # [1, B, 1, rotary_dim]
    sin_decode = sin_meta.reshape(1, B, 1, -1)

    # Convert to TTNN with correct format
    mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
    cos_ttnn = ttnn.from_torch(
        cos_decode.to(torch.bfloat16),
        device=self.device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    sin_ttnn = ttnn.from_torch(
        sin_decode.to(torch.bfloat16),
        device=self.device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # === Fix Bug 3: Convert to HEIGHT_SHARDED for rotary_embedding_llama ===
    batch_grid = ttnn.num_cores_to_corerangeset(
        batch_size, self.device.compute_with_storage_grid_size(), True
    )
    rope_shard_mem = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, cos_ttnn.shape[-1]),  # (32, rotary_dim)
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    cos_ttnn = ttnn.to_memory_config(cos_ttnn, rope_shard_mem)
    sin_ttnn = ttnn.to_memory_config(sin_ttnn, rope_shard_mem)

    # Also shard trans_mat if not already sharded
    if not self._decode_trans_mat_sharded:
        trans_mat_mem = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._decode_trans_mat = ttnn.to_memory_config(
            self._decode_trans_mat, trans_mat_mem
        )
        self._decode_trans_mat_sharded = True

    # Now call _apply_partial_rope_decode with correctly formatted inputs
    query_states, key_states = self._apply_partial_rope_decode(
        query_states, key_states, cos_ttnn, sin_ttnn, self._decode_trans_mat
    )
```

**IMPORTANT:** The cos/sin from HuggingFace may arrive as either torch.Tensor or ttnn.Tensor (wrapped via TorchTTNNTensor). The conversion must handle both cases by detecting the input type and converting to torch first if needed.

#### Approach B: Replace HuggingFace rotary_emb with BailingRotarySetup (ALTERNATIVE)

Instantiate `BailingRotarySetup` during `move_weights_to_device_impl()` and use it to generate cos/sin in the correct format during decode. This requires intercepting `position_embeddings` in the forward method and replacing them with `BailingRotarySetup.get_cos_sin_for_decode(position_ids)`.

**Advantage:** The cos/sin would already be on-device in Meta format with replicated topology. No per-step host conversion needed.

**Disadvantage:** Requires extracting position_ids from the HuggingFace flow (not straightforward since `model.generate` passes position_embeddings, not position_ids, to the attention layer).

### 12.5 Why Approach A is Recommended

1. **Minimal change:** Only modifies the beginning of `_forward_decode_paged()`, no changes to the model integration or HuggingFace flow.
2. **Correctness:** Handles the format conversion explicitly and verifiably.
3. **Sharding:** Adds the missing HEIGHT_SHARDED conversion for cos/sin/trans_mat.
4. **Acceptable overhead:** For decode (batch=1, seq=1), the cos/sin tensors are tiny (64 elements). The host conversion adds negligible latency compared to the existing `_to_replicated` host round-trip for the fused QKV tensor.

### 12.6 Updated Changes to `_forward_decode_paged` (Sections 3+12 Combined)

| Line Range | Action | Description |
|------------|--------|-------------|
| 2692-2713 | KEEP | QKV projection and fusing (already correct from Section 3) |
| 2719-2721 | KEEP | `_to_replicated()` on fused tensor (needed for topology) |
| 2723-2728 | KEEP | `nlp_create_qkv_heads_decode` (already correct from Section 3) |
| 2730-2735 | KEEP | L1 interleaved + QK norm (already correct from Section 3) |
| 2737-2745 | **REPLACE** | Convert HF cos/sin to Meta format + 4D shape + HEIGHT_SHARDED, then call `_apply_partial_rope_decode` |
| 2747-2843 | KEEP | Paged cache update, SDPA, concat heads, dense (already correct) |

### 12.7 trans_mat Sharding at Init Time

The `_decode_trans_mat` should be pre-sharded during `move_weights_to_device_impl()` to avoid per-step sharding overhead. Update the init code:

```python
# In move_weights_to_device_impl(), after creating _decode_trans_mat:
# Pre-shard trans_mat for batch_size=1 (will need to be re-done if batch size changes)
batch_grid = ttnn.num_cores_to_corerangeset(1, self.device.compute_with_storage_grid_size(), True)
trans_mat_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
self._decode_trans_mat = ttnn.to_memory_config(self._decode_trans_mat, trans_mat_mem)
```

### 12.8 Verification Checklist

- [ ] cos/sin arrive at `rotary_embedding_llama` in Meta format (interleaved pairs)
- [ ] cos/sin shape is `[1, B, 1, rotary_dim]` (4D)
- [ ] cos/sin are HEIGHT_SHARDED in L1
- [ ] trans_mat is HEIGHT_SHARDED in L1
- [ ] Q/K inputs to `rotary_embedding_llama` are HEIGHT_SHARDED (or convert them in `_apply_partial_rope_decode`)
- [ ] For partial RoPE: Q/K are split to rotary_dim before RoPE, concatenated after
- [ ] Output text is coherent (not repeating tokens)

---

## 13. Iteration 3 - Simplified Approach (Deep Debug After "aukee" Still Repeating)

### 13.1 Context

After implementing all fixes from Sections 3 and 12 (nlp_create_qkv_heads_decode, decode-mode
rotary_embedding_llama, HF-to-Meta cos/sin conversion, HEIGHT_SHARDED memory configs), the model
runs without runtime errors but STILL produces "aukee" repeating tokens. This section provides a
root-cause analysis based on a detailed trace of the decode dataflow.

### 13.2 Root Cause Analysis

#### Bug A: Slicing HEIGHT_SHARDED Tensors (CRITICAL - This Is Why Output Is Garbled)

The `_apply_partial_rope_decode()` method (line 2561) receives Q/K tensors that are already
HEIGHT_SHARDED (sharded at line 2837-2846) and then tries to SLICE them along dim=-1:

```python
# line 2590 - THIS IS ILLEGAL ON HEIGHT_SHARDED TENSORS
q_rot = query_states[:, :, :, :rotary_dim]   # [1, B, 16, 64] from [1, B, 16, 128]
q_pass = query_states[:, :, :, rotary_dim:]   # [1, B, 16, 64]
```

As stated in the rope.py docstring (line 18): _"This avoids slicing HEIGHT_SHARDED tensors which
is not allowed."_ When you slice a HEIGHT_SHARDED tensor along the width dimension, TTNN either
silently produces garbage data or returns incorrect memory views. This single bug is sufficient
to explain the garbled output.

#### Bug B: trans_mat Shard Grid Mismatch (CRITICAL)

The `_decode_trans_mat` is pre-sharded during `move_weights_to_device_impl()` on **1 core**
(line 2458: `num_cores_to_corerangeset(1, ...)`), but at runtime the Q/K/cos/sin tensors are
sharded on `batch_size` cores (line 2816). The `rotary_embedding_llama` kernel requires ALL
inputs to be on the SAME shard grid. Mismatched grids produce silent data corruption.

#### Bug C: Redundant and Fragile cos/sin Conversion (WRONG APPROACH)

The inline HF-to-Meta conversion (lines 2752-2813) is ~60 lines of complex code that:
1. Pulls TTNN tensors back to PyTorch on host (slow)
2. Does the half/stack/flatten conversion (correct math, but unnecessary)
3. Pushes back to device (slow)
4. Shards to HEIGHT_SHARDED (correct, but the shard width is rotary_dim=64, not head_dim=128)

This approach conflicts with how the standard TT-Transformers pattern works. The standard pattern
uses identity-padded cos/sin at FULL head_dim so that NO tensor slicing is ever needed.

### 13.3 What The Standard TT-Transformers Pattern Does

Reference: `models/common/modules/rope/rope_1d.py` and `models/common/modules/rope/rope_setup.py`

The standard approach for partial RoPE in decode mode:
1. Pre-compute cos/sin in Meta format at `rotary_dim` (64)
2. **Identity-pad** cos to `head_dim` (128) with cos=1.0 for dims 64-127
3. **Identity-pad** sin to `head_dim` (128) with sin=0.0 for dims 64-127
4. Pass FULL Q/K (all 128 dims) through `rotary_embedding_llama` -- no slicing needed
5. The identity padding ensures dims 64-127 pass through unchanged:
   `q_out[i] = q[i] * 1.0 + rotate_half(q)[i] * 0.0 = q[i]` for i >= 64

This is exactly what the rope.py docstring at line 17-18 describes.

### 13.4 The Simplest Fix

**Strategy**: Use identity-padded cos/sin at full head_dim. Eliminate ALL tensor slicing in
the decode RoPE path. Use `BailingRotarySetup` (which already computes Meta-format cos/sin)
instead of inline conversion.

#### Step 1: Modify `BailingRotarySetup._compute_cos_sin_cache()` to identity-pad

Currently `_compute_cos_sin_cache()` returns cos/sin at shape `[1, 1, max_seq_len, rotary_dim]`
(rotary_dim=64). Modify it to return `[1, 1, max_seq_len, head_dim]` (head_dim=128) with
identity padding for the non-rotary portion.

In `rope.py`, function `_compute_cos_sin_cache()` (line 362), add after line 406:

```python
# Identity-pad for partial rotary: dims beyond rotary_dim pass through unchanged
# cos=1.0 means no rotation, sin=0.0 means no cross-term
if rotary_dim < head_dim:
    pad_width = head_dim - rotary_dim
    cos_pad = torch.ones(cos.shape[0], cos.shape[1], cos.shape[2], pad_width)
    sin_pad = torch.zeros(sin.shape[0], sin.shape[1], sin.shape[2], pad_width)
    cos = torch.cat([cos, cos_pad], dim=-1)
    sin = torch.cat([sin, sin_pad], dim=-1)
```

Note: `_compute_cos_sin_cache` currently does NOT receive `head_dim` as a separate arg when
`partial_rotary_factor < 1.0`. The function signature needs to accept `head_dim` explicitly
(currently it computes `rotary_dim = int(head_dim * partial_rotary_factor)` which means the
incoming `head_dim` IS the full head_dim). Actually, looking more carefully: the function takes
`head_dim` and `partial_rotary_factor` separately, computes `rotary_dim`, and returns at
`rotary_dim`. So we just need to pad back to `head_dim` at the end.

#### Step 2: Modify `BailingRotarySetup.get_cos_sin_for_decode()` output shape

Currently returns `[1, batch, 1, rotary_dim]`. After Step 1, the cache will be at
`[1, 1, max_seq_len, head_dim]`, so `get_cos_sin_for_decode()` will automatically return
`[1, batch, 1, head_dim]`. No change needed here.

#### Step 3: Replace the decode cos/sin code in `_forward_decode_paged()`

Replace lines 2752-2851 (the entire inline HF-to-Meta conversion + HEIGHT_SHARDED + partial
RoPE call) with:

```python
# Get cos/sin from BailingRotarySetup (already Meta format, identity-padded, on device)
cos_ttnn, sin_ttnn = self._rotary_setup.get_cos_sin_for_decode(cache_position_tensor)

# Shard cos/sin to HEIGHT_SHARDED
batch_grid = ttnn.num_cores_to_corerangeset(
    batch_size, self.device.compute_with_storage_grid_size(), True
)
rope_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),  # FULL head_dim, not rotary_dim
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
cos_ttnn = ttnn.to_memory_config(cos_ttnn, rope_shard_mem)
sin_ttnn = ttnn.to_memory_config(sin_ttnn, rope_shard_mem)

# Shard Q/K to HEIGHT_SHARDED
q_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, self.head_dim),
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
query_states = ttnn.to_memory_config(query_states, q_shard_mem)
k_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, key_states.shape[-1]),
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
key_states = ttnn.to_memory_config(key_states, k_shard_mem)

# Re-shard trans_mat to match batch_grid (was pre-sharded on 1 core at init)
trans_mat = ttnn.to_memory_config(self._decode_trans_mat, ttnn.DRAM_MEMORY_CONFIG)
trans_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
trans_mat = ttnn.to_memory_config(trans_mat, trans_shard_mem)

# Apply FULL RoPE (no slicing!) - identity padding handles partial rotation
query_states = ttnn.experimental.rotary_embedding_llama(
    query_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True
)
key_states = ttnn.experimental.rotary_embedding_llama(
    key_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True
)
```

#### Step 4: Delete `_apply_partial_rope_decode()` method

The entire `_apply_partial_rope_decode()` method (lines 2561-2615) is no longer needed since
the identity-padded cos/sin eliminate the need for slicing.

#### Step 5: Initialize `BailingRotarySetup` in `move_weights_to_device_impl()`

Add to `move_weights_to_device_impl()`:

```python
from models.experimental.tt_symbiote.modules.rope import BailingRotarySetup

config = self._fallback_torch_layer.config
self._rotary_setup = BailingRotarySetup(
    device=self.device,
    head_dim=self.head_dim,
    max_seq_len=config.max_position_embeddings,
    rope_theta=config.rope_theta,
    partial_rotary_factor=self.partial_rotary_factor,
)
```

#### Step 6: Remove the pre-sharded `_decode_trans_mat` from init

Since `BailingRotarySetup` already provides `trans_mat_decode`, use that instead.
Replace lines 2442-2466 with:

```python
# trans_mat is available from self._rotary_setup.trans_mat_decode
# (sharding happens at decode time to match batch_size)
```

Or, keep a DRAM copy from `BailingRotarySetup.trans_mat_decode` and re-shard per call.

### 13.5 Alternative: Even Simpler Fix (Minimal Changes)

If you want the absolute minimum code change to test the hypothesis:

**Just move the slicing BEFORE the HEIGHT_SHARDED conversion.**

In `_forward_decode_paged()`, lines 2829-2851, reorder so that:
1. Q/K are sliced to rotary_dim WHILE STILL IN L1_INTERLEAVED (after QK norm, before sharding)
2. Then shard the sliced tensors to HEIGHT_SHARDED
3. Apply `rotary_embedding_llama` to the rotary portion
4. Un-shard, concatenate with pass-through, re-shard for paged kernels

This is still ugly but would verify the hypothesis quickly:

```python
# After QK norm, Q is [1, B, 16, 128] in L1_INTERLEAVED
# Slice BEFORE sharding (legal on interleaved tensors)
q_rot = query_states[:, :, :, :rotary_dim]   # [1, B, 16, 64]
q_pass = query_states[:, :, :, rotary_dim:]   # [1, B, 16, 64]
k_rot = key_states[:, :, :, :rotary_dim]      # [1, B, 4, 64]
k_pass = key_states[:, :, :, rotary_dim:]      # [1, B, 4, 64]

# Now shard the rotary portions
batch_grid = ttnn.num_cores_to_corerangeset(batch_size, self.device.compute_with_storage_grid_size(), True)
q_rot_shard = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, rotary_dim), core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
q_rot = ttnn.to_memory_config(q_rot, q_rot_shard)
k_rot_shard = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, rotary_dim), core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
k_rot = ttnn.to_memory_config(k_rot, k_rot_shard)

# Shard cos/sin (already at rotary_dim)
rope_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, rotary_dim), core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
cos_ttnn = ttnn.to_memory_config(cos_ttnn, rope_shard_mem)
sin_ttnn = ttnn.to_memory_config(sin_ttnn, rope_shard_mem)

# Re-shard trans_mat to batch_grid
trans_mat = ttnn.to_memory_config(self._decode_trans_mat, ttnn.DRAM_MEMORY_CONFIG)
trans_shard = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE), core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
trans_mat = ttnn.to_memory_config(trans_mat, trans_shard)

# Apply RoPE
q_rot = ttnn.experimental.rotary_embedding_llama(q_rot, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True)
k_rot = ttnn.experimental.rotary_embedding_llama(k_rot, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True)

# Un-shard and concatenate
q_rot = ttnn.to_memory_config(q_rot, ttnn.L1_MEMORY_CONFIG)
k_rot = ttnn.to_memory_config(k_rot, ttnn.L1_MEMORY_CONFIG)
query_states = ttnn.concat([q_rot, q_pass], dim=-1)
key_states = ttnn.concat([k_rot, k_pass], dim=-1)
```

### 13.6 Recommendation

**Use the identity-padding approach (Section 13.4, Steps 1-6).** It is cleaner, follows the
standard TT-Transformers pattern, eliminates ~80 lines of fragile inline cos/sin conversion,
and is less error-prone. The alternative (Section 13.5) is useful only as a quick hypothesis test.

### 13.7 Additional Concern: Prefill Path

The prefill path (line 2649-2655) uses `TTNNRotaryPositionEmbedding` which calls
`ttnn.experimental.rotary_embedding` (NOT `rotary_embedding_llama`). This kernel expects
**HF-format** cos/sin. The prefill path receives HF cos/sin directly from
`BailingMoeV2RotaryEmbedding.forward()` via `position_embeddings`, so the prefill path
should be CORRECT.

However, if `BailingRotarySetup` is used for decode, the prefill path should continue to
receive HF-format cos/sin from the model's own rotary embedding. The two paths use different
cos/sin sources, which is fine and intentional.

### 13.8 Summary of All Bugs in Current Decode Path

| Bug | Severity | Description |
|-----|----------|-------------|
| A. Slicing HEIGHT_SHARDED tensors | CRITICAL | `_apply_partial_rope_decode` slices Q/K along dim=-1 while HEIGHT_SHARDED. Produces garbage. |
| B. trans_mat shard grid mismatch | CRITICAL | trans_mat sharded on 1 core, Q/K/cos/sin on batch_size cores. Kernel requires same grid. |
| C. Inline cos/sin conversion | Fragile | ~60 lines of host round-trip. Correct math, but unnecessary and slow. |

### 13.9 Verification After Fix

1. Run the existing test: `pytest models/experimental/tt_symbiote/tests/test_bailing_attention_accuracy.py -k decode`
2. Run the end-to-end generation test and verify output is NOT "aukee" repeating
3. Print Q/K values before and after RoPE to verify the pass-through dims (64-127) are unchanged
4. Print cos/sin values at runtime to verify identity padding (cos[64:]=1.0, sin[64:]=0.0)
