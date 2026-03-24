# Plan: TTNN vs PyTorch Attention Equivalence Tests for T3K with Sharded Inputs

## Problem Description

### Scope
Create tests that verify `TTNNBailingMoEAttention` matches PyTorch `BailingMoeV2Attention/SdpaAttention` for **T3K mesh device (8 Wormhole chips)** where input is **sharded on the last dimension**.

### Out of Scope
- Single device testing
- Replicated input accuracy testing (already covered in `test_ling_attention_equivalence_t3k.py`)

### Why This Matters
The current test file (`test_ling_attention_equivalence_t3k.py`) uses `ttnn.ReplicateTensorToMesh` which copies the full input to all 8 devices. In production T3K deployments, inputs are typically **sharded** across devices on the last dimension (hidden_size) for memory efficiency and compute parallelism:

- **Replicated**: Each device holds full tensor `[batch, seq, 2048]` - current test coverage
- **Sharded**: Each device holds `[batch, seq, 256]` (2048 / 8 devices) - **needs test coverage**

This is critical because:
1. The distributed linear modules (`TTNNLinearIColShardedWRowSharded`, `TTNNLinearIReplicatedWColSharded`) expect sharded inputs
2. All-gather operations must correctly reconstruct full tensors for operations that need them
3. QK normalization, RoPE, and SDPA must handle the sharded/gathered data correctly

### Ling Model Configuration
| Parameter | Value | T3K Sharding Impact |
|-----------|-------|---------------------|
| hidden_size | 2048 | Sharded to 256 per device |
| num_attention_heads (Q) | 16 | 2 heads per device |
| num_key_value_heads (KV) | 4 | < 8 devices: must replicate |
| head_dim | 128 | Not sharded directly |
| partial_rotary_factor | 0.5 | rotary_dim = 64 |
| use_qk_norm | True | Applied per-head |

---

## Test Architecture

### Input Flow for Sharded T3K

```
Input: torch.Tensor [batch, seq, 2048]
                    |
                    v
        ttnn.ShardTensorToMesh(device, dim=-1)
                    |
                    v
        TTNN Tensor on T3K: each device has [batch, seq, 256]
                    |
        +-----------+-----------+
        |                       |
        v                       v
   Q Projection             K/V Projection
(TTNNLinearIColSharded)   (TTNNLinearIReplicated*)
        |                       |
        v                       |
  all_gather (Q)          (all_gather to replicate input)
        |                       |
        +-----------+-----------+
                    |
                    v
             reshape, permute
             [batch, heads, seq, head_dim]
                    |
                    v
              QK Normalization
                    |
                    v
             Partial RoPE
                    |
                    v
                  SDPA
                    |
                    v
             Dense Projection
```

*Note: K/V projections use `TTNNLinearIReplicatedWColSharded` because num_kv_heads (4) < num_devices (8)

### Key Verification Points

1. **Sharded Input Handling**: Verify input sharding produces correct per-device slices
2. **QKV Projections with Sharded Input**: Ensure separate Q, K, V paths work correctly
3. **All-Gather Reconstruction**: Verify all_gather reconstructs full tensors
4. **RoPE Application**: Verify partial RoPE works with gathered tensors
5. **QK Norm**: Verify normalization produces correct outputs
6. **SDPA**: Verify scaled dot-product attention matches PyTorch
7. **Dense Projection**: Verify output projection with sharded results
8. **End-to-End**: Compare final attention output to PyTorch reference

---

## Test Cases

### Test 1: Sharded Input Conversion Sanity Check
**File**: `test_ling_attention_t3k_sharded.py`
**Function**: `test_sharded_input_conversion_t3k`

**Purpose**: Verify that input sharding and all-gather reconstruction is lossless.

```python
@pytest.mark.parametrize("seq_len", [32, 64, 128])
def test_sharded_input_conversion_t3k(mesh_device, seq_len):
    """Verify sharded input can be converted and reconstructed correctly."""
    batch_size = 1
    hidden_size = 2048

    # Create input
    input_torch = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Shard to T3K (8 devices, last dim)
    sharded_tt = ttnn.from_torch(
        input_torch,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    # Verify shape on each device
    assert sharded_tt.shape == [batch_size, seq_len, 256]  # 2048/8

    # All-gather to reconstruct
    gathered_tt = ttnn.all_gather(sharded_tt, dim=-1, num_links=1)

    # Convert back and compare
    reconstructed = ttnn.to_torch(
        gathered_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[:batch_size]

    pcc = compute_pcc(input_torch.float(), reconstructed.float())
    assert pcc > 0.9999, f"Reconstruction PCC too low: {pcc}"
```

**Success Criteria**: PCC > 0.9999 (essentially perfect reconstruction)

---

### Test 2: QKV Projection with Sharded Input
**Function**: `test_qkv_projection_sharded_t3k`

**Purpose**: Verify Q, K, V projections produce correct outputs when input is sharded.

```python
@pytest.mark.parametrize("seq_len", [32, 64])
def test_qkv_projection_sharded_t3k(mesh_device, reset_seeds, seq_len):
    """Test QKV projection correctness with sharded input."""
    batch_size = 1

    # Load PyTorch reference
    config = get_ling_config()
    model = get_ling_model()
    torch_attn = get_ling_attention_layer(0, model)

    # Create input
    hidden_states = torch.randn(batch_size, seq_len, LING_HIDDEN_SIZE, dtype=torch.float32)

    # === PyTorch Reference ===
    # Manual QKV projection
    qkv_weight = torch_attn.query_key_value.weight
    qkv_bias = torch_attn.query_key_value.bias
    qkv_pt = F.linear(hidden_states, qkv_weight, qkv_bias)

    # Split Q, K, V
    q_size = 16 * 128  # num_heads * head_dim = 2048
    kv_size = 4 * 128  # num_kv_heads * head_dim = 512
    q_pt = qkv_pt[..., :q_size]
    k_pt = qkv_pt[..., q_size:q_size+kv_size]
    v_pt = qkv_pt[..., q_size+kv_size:]

    # === TTNN with Sharded Input ===
    ttnn_attn = TTNNBailingMoEAttention.from_torch(torch_attn, distributed=True)
    set_device(ttnn_attn, mesh_device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    # Shard input to T3K
    hidden_tt = ttnn.from_torch(
        hidden_states.to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    # Run Q projection (accepts sharded input)
    q_tt = ttnn_attn.q_proj(hidden_tt)

    # All-gather for K/V input (they need replicated input)
    hidden_replicated = ttnn.all_gather(hidden_tt, dim=-1, num_links=1)
    k_tt = ttnn_attn.k_proj(hidden_replicated)
    v_tt = ttnn_attn.v_proj(hidden_replicated)

    # Gather outputs and compare
    # ... convert and compute PCC for q, k, v separately ...
```

**Success Criteria**:
- Q projection PCC > 0.99
- K projection PCC > 0.99
- V projection PCC > 0.99

---

### Test 3: QK Normalization with Sharded Flow
**Function**: `test_qk_norm_sharded_t3k`

**Purpose**: Verify QK normalization produces correct outputs in the sharded pipeline.

```python
@pytest.mark.parametrize("seq_len", [32, 64])
def test_qk_norm_sharded_t3k(mesh_device, reset_seeds, seq_len):
    """Test QK normalization in sharded T3K flow."""
    # Create Q and K tensors in the shape after projection and reshape
    # [batch, num_heads, seq, head_dim]
    batch_size = 1
    q_states = torch.randn(batch_size, 16, seq_len, 128, dtype=torch.float32)
    k_states = torch.randn(batch_size, 4, seq_len, 128, dtype=torch.float32)

    # === PyTorch Reference ===
    # Apply RMSNorm per-head
    # ... compute q_normed_pt, k_normed_pt ...

    # === TTNN ===
    # Convert to TTNN and apply normalization via ttnn_attn._apply_qk_norm
    # ... compare outputs ...
```

**Success Criteria**: PCC > 0.99 for both normalized Q and K

---

### Test 4: Partial RoPE with Sharded Flow
**Function**: `test_partial_rope_sharded_t3k`

**Purpose**: Verify partial RoPE (rotary_factor=0.5) works correctly when tensors have been gathered from sharded inputs.

```python
@pytest.mark.parametrize("seq_len", [32, 64])
def test_partial_rope_sharded_t3k(mesh_device, reset_seeds, seq_len):
    """Test partial RoPE application in sharded T3K flow."""
    batch_size = 1

    # Create Q, K tensors and position embeddings
    q_states = torch.randn(batch_size, 16, seq_len, 128, dtype=torch.float32)
    k_states = torch.randn(batch_size, 4, seq_len, 128, dtype=torch.float32)

    config = get_ling_config()
    cos, sin = create_position_embeddings(config, seq_len)

    # === PyTorch Reference ===
    # Apply partial rotary embedding
    # ... q_rope_pt, k_rope_pt ...

    # === TTNN ===
    # The key test is ensuring cos/sin are REPLICATED (not sharded)
    # even when input flow starts with sharded tensors
    # ... compare outputs ...
```

**Success Criteria**: PCC > 0.99 for both rotated Q and K

---

### Test 5: Full Prefill with Sharded Input
**Function**: `test_prefill_sharded_input_t3k`

**Purpose**: End-to-end prefill test where input hidden_states is sharded.

```python
@pytest.mark.parametrize("seq_len", [32, 64, 128])
@pytest.mark.parametrize("batch_size", [1])
def test_prefill_sharded_input_t3k(mesh_device, reset_seeds, seq_len, batch_size):
    """Test full prefill with sharded input on T3K."""
    config = get_ling_config()
    model = get_ling_model()
    torch_attn = get_ling_attention_layer(0, model)

    # Create inputs
    hidden_states = torch.randn(batch_size, seq_len, LING_HIDDEN_SIZE, dtype=torch.float32)
    cos, sin = create_position_embeddings(config, seq_len)

    # === PyTorch Reference ===
    cache_pt = DynamicCache()
    out_pt, _, cache_pt = torch_attn(
        hidden_states,
        position_embeddings=(cos, sin),
        past_key_value=cache_pt,
    )

    # === TTNN with Sharded Input ===
    ttnn_attn = TTNNBailingMoEAttention.from_torch(torch_attn, distributed=True)
    set_device(ttnn_attn, mesh_device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    paged_cache = create_paged_kv_cache(config, mesh_device, batch_size)

    # CRITICAL: Shard input on last dimension
    hidden_tt = ttnn.from_torch(
        hidden_states.to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),  # <-- KEY DIFFERENCE
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    # Position embeddings must be REPLICATED
    cos_tt = ttnn.from_torch(
        cos.to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),  # <-- Must be replicated
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    sin_tt = ttnn.from_torch(
        sin.to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),  # <-- Must be replicated
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    # Run TTNN forward
    out_tt, _, _ = ttnn_attn(
        hidden_tt,
        position_embeddings=(cos_tt, sin_tt),
        past_key_values=paged_cache,
    )

    # Convert output and compare
    out_torch = convert_ttnn_output_to_torch(out_tt, mesh_device, batch_size)

    passing, pcc_msg = comp_pcc(out_pt.float(), out_torch.float(), pcc_threshold=0.99)
    assert passing, f"Prefill with sharded input failed: {pcc_msg}"
```

**Success Criteria**: PCC > 0.99

---

### Test 6: Decode with Sharded Input (Single Token)
**Function**: `test_decode_sharded_input_t3k`

**Purpose**: Verify decode phase (seq_len=1) works correctly with sharded input.

```python
@pytest.mark.parametrize("cur_pos", [10, 32, 63, 64, 65, 127])
def test_decode_sharded_input_t3k(mesh_device, reset_seeds, cur_pos):
    """Test decode with sharded input on T3K."""
    batch_size = 1
    prefill_len = cur_pos

    config = get_ling_config()
    model = get_ling_model()
    torch_attn = get_ling_attention_layer(0, model)

    # Prefill first (to populate KV cache)
    prefill_hidden = torch.randn(batch_size, prefill_len, LING_HIDDEN_SIZE, dtype=torch.float32)
    prefill_cos, prefill_sin = create_position_embeddings(config, prefill_len)

    # Decode token
    decode_hidden = torch.randn(batch_size, 1, LING_HIDDEN_SIZE, dtype=torch.float32)
    decode_cos, decode_sin = create_position_embeddings(config, 1, start_pos=cur_pos)

    # === PyTorch Reference ===
    cache_pt = DynamicCache()
    # Prefill
    _, _, cache_pt = torch_attn(prefill_hidden, position_embeddings=(prefill_cos, prefill_sin), past_key_value=cache_pt)
    # Decode
    out_pt, _, _ = torch_attn(decode_hidden, position_embeddings=(decode_cos, decode_sin),
                              past_key_value=cache_pt, cache_position=torch.tensor([cur_pos]))

    # === TTNN with Sharded Input ===
    ttnn_attn = TTNNBailingMoEAttention.from_torch(torch_attn, distributed=True)
    set_device(ttnn_attn, mesh_device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    paged_cache = create_paged_kv_cache(config, mesh_device, batch_size)

    # Prefill with sharded input
    prefill_tt = to_sharded_ttnn(prefill_hidden, mesh_device)
    prefill_cos_tt = to_replicated_ttnn(prefill_cos, mesh_device)
    prefill_sin_tt = to_replicated_ttnn(prefill_sin, mesh_device)
    _, _, paged_cache = ttnn_attn(prefill_tt, (prefill_cos_tt, prefill_sin_tt), past_key_values=paged_cache)

    # Decode with sharded input
    decode_tt = to_sharded_ttnn(decode_hidden, mesh_device)
    decode_cos_tt = to_replicated_ttnn(decode_cos, mesh_device)
    decode_sin_tt = to_replicated_ttnn(decode_sin, mesh_device)
    cache_pos_tt = to_replicated_ttnn(torch.tensor([cur_pos], dtype=torch.int32), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)

    out_tt, _, _ = ttnn_attn(decode_tt, (decode_cos_tt, decode_sin_tt),
                             past_key_values=paged_cache, cache_position=cache_pos_tt)

    out_torch = convert_ttnn_output_to_torch(out_tt, mesh_device, batch_size)

    passing, pcc_msg = comp_pcc(out_pt.float(), out_torch.float(), pcc_threshold=0.98)
    assert passing, f"Decode with sharded input at pos={cur_pos} failed: {pcc_msg}"
```

**Success Criteria**: PCC > 0.98 (relaxed due to potential Issue #30362)

---

### Test 7: Full Attention Flow (Prefill + Multiple Decodes)
**Function**: `test_full_flow_sharded_t3k`

**Purpose**: Verify accuracy over multiple decode steps with sharded input.

```python
@pytest.mark.parametrize("prefill_len", [32, 64])
@pytest.mark.parametrize("decode_steps", [16, 32])
def test_full_flow_sharded_t3k(mesh_device, reset_seeds, prefill_len, decode_steps):
    """Test full prefill + decode flow with sharded input."""
    # Similar to test_full_attention_equivalence_t3k but with sharded inputs
    # Track PCC at each decode step
    # Success: prefill PCC > 0.99, at least 80% decode steps PCC > 0.98
```

**Success Criteria**:
- Prefill PCC > 0.99
- At least 80% of decode steps pass PCC > 0.98
- No NaN/Inf values in any output

---

## Helper Functions Needed

```python
def to_sharded_ttnn(tensor: torch.Tensor, mesh_device, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    """Convert torch tensor to TTNN with sharding on last dimension."""
    return ttnn.from_torch(
        tensor.to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=layout,
        dtype=ttnn.bfloat16,
    )

def to_replicated_ttnn(tensor: torch.Tensor, mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Convert torch tensor to TTNN with replication across all devices."""
    return ttnn.from_torch(
        tensor.to(torch.bfloat16) if dtype == ttnn.bfloat16 else tensor,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=layout,
        dtype=dtype,
    )

def convert_ttnn_output_to_torch(ttnn_tensor, mesh_device, batch_size) -> torch.Tensor:
    """Convert TTNN output back to torch, handling mesh composition.

    For sharded outputs, use ConcatMeshToTensor on the shard dimension.
    """
    if hasattr(ttnn_tensor, "to_ttnn"):
        ttnn_tensor = ttnn_tensor.to_ttnn

    # Output from TTNNBailingMoEAttention is [batch, seq, hidden_size]
    # If distributed=True, output is sharded on last dim, need to concat
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    torch_tensor = ttnn.to_torch(ttnn_tensor, mesh_composer=mesh_composer)

    # Handle batch dimension replication
    if torch_tensor.shape[0] > batch_size:
        torch_tensor = torch_tensor[:batch_size]

    return torch_tensor
```

---

## Success Criteria Summary

| Test | PCC Threshold | Notes |
|------|---------------|-------|
| Sharded Input Conversion | 0.9999 | Near-perfect reconstruction |
| QKV Projection | 0.99 | Per projection |
| QK Normalization | 0.99 | Per Q and K |
| Partial RoPE | 0.99 | Per Q and K |
| Prefill (sharded) | 0.99 | End-to-end |
| Decode (sharded) | 0.98 | Relaxed for Issue #30362 |
| Full Flow | Prefill: 0.99, Decode: 80% at 0.98 | Multi-step |

---

## Step-by-Step Implementation Plan for Implementer Agent

### Step 1: Create Test File
Create `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/attention/test_ling_attention_t3k_sharded.py`

### Step 2: Add Helper Functions
Add the helper functions defined above at the top of the test file after imports.

### Step 3: Implement Test 1 (Sharded Input Conversion)
- Start with the simplest test to verify infrastructure works
- This validates the sharding/all-gather roundtrip

### Step 4: Implement Test 5 (Full Prefill with Sharded Input)
- This is the most important test
- If this fails, debug using Tests 2-4 to isolate the issue

### Step 5: Implement Test 6 (Decode with Sharded Input)
- Depends on prefill working
- Test multiple cur_pos values including page boundaries

### Step 6: Implement Test 7 (Full Flow)
- End-to-end validation
- Good for catching accuracy drift over multiple steps

### Step 7: (Optional) Implement Tests 2-4
- Only if Tests 5-6 fail and isolation is needed
- These are debugging aids

### Step 8: Run Tests
```bash
# From tt-metal repo root
export MESH_DEVICE=T3K
pytest models/experimental/tt_symbiote/tests/attention/test_ling_attention_t3k_sharded.py -v -s
```

### Step 9: Document Results
- Record PCC values for each test
- Note any failures with cur_pos values for Issue #30362 correlation

---

## Potential Issues to Watch For

1. **Sharded Position Embeddings**: The implementation has logic to detect and re-replicate sharded cos/sin tensors. Verify this works with explicitly sharded inputs.

2. **Output Sharding**: The dense projection output may be sharded. Ensure `convert_ttnn_output_to_torch` handles this correctly.

3. **Page Boundary Effects**: cur_pos values near block_size (64) may have different accuracy. Track these separately.

4. **Memory Management**: Sharded tensors use different memory than replicated. Watch for OOM or allocation failures.

5. **All-Gather Synchronization**: The implementation calls `ttnn.synchronize_device` after all-gather. Verify this is sufficient.

---

## File Locations

| File | Purpose |
|------|---------|
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/attention/test_ling_attention_t3k_sharded.py` | New test file (to create) |
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/attention/test_utils.py` | Existing test utilities |
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/attention/conftest.py` | Pytest fixtures |
| `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` | TTNNBailingMoEAttention implementation |
