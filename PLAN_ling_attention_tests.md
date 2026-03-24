# PLAN: Ling (BailingMoeV2) Attention TTNN vs PyTorch Equivalence Tests

## 1. Problem Description

The Ling model (BailingMoeV2) generates incorrect text during decode. To diagnose the root cause, we need comprehensive tests that verify each component of the TTNN attention implementation matches the PyTorch/HuggingFace reference implementation.

### Model-Specific Characteristics

The Ling attention has several unique features that require targeted testing:

| Feature | Value | Notes |
|---------|-------|-------|
| Hidden size | 2048 | Standard for mini model |
| Number of Q heads (`nh`) | 16 | Full attention heads |
| Number of KV heads (`nkv`) | 4 | GQA with 4:1 ratio |
| Head dimension (`dh`) | 128 | 2048 / 16 |
| Group size | 4 | 16 Q heads / 4 KV heads |
| `partial_rotary_factor` | 0.5 | Only half of head_dim gets RoPE |
| Rotary dimension | 64 | 128 * 0.5 |
| `use_qk_norm` | True | RMSNorm on Q and K before RoPE |
| RMS norm eps | 1e-6 | |
| `use_qkv_bias` | False | No bias in QKV projection |
| `use_bias` | False | No bias in output projection |

### Key Differences Between Implementations

From the comparison document (`ling_attention_implementations.md`):

| Aspect | PyTorch | TTNN |
|--------|---------|------|
| QKV Projection | Fused `query_key_value` | Separate Q/K/V in distributed mode |
| QKV Split | `split([num_heads, num_kv_heads, num_kv_heads], dim=-2)` | `ttnn.slice()` along last dim |
| QK Normalization | Direct call on 4D tensor `[B,H,S,D]` | Reshape to 2D, normalize, reshape back |
| RoPE | `apply_rotary_pos_emb()` with partial support | `TTNNRotaryPositionEmbedding` module |
| GQA Handling | Explicit `repeat_kv()` to expand K/V | SDPA kernel handles internally |
| is_causal (decode) | `is_causal=False` when `q_len==1` | Always `is_causal=True` |
| KV Cache Update | `past_key_value.update()` | `paged_update_on_device()` |
| Attention | `torch.nn.functional.scaled_dot_product_attention` | `ttnn.sdpa()` or `paged_sdpa_decode()` |

### Identified Potential Issues

1. **Double Sequence Length Increment**: After `paged_update_on_device()`, there's manual `_seq_lengths[layer_idx] += seq_length` but the op may already increment internally.
2. **is_causal Mismatch**: PyTorch uses `is_causal=False` for decode, TTNN uses `True`.
3. **Partial RoPE**: The TTNN implementation uses `TTNNRotaryPositionEmbedding` which may not handle partial rotary correctly in distributed mode.

---

## 2. Research Findings

### 2.1 Existing Test Patterns

From analysis of tt-metal test files:

#### Test Structure Pattern (from `test_attention.py`)

```python
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [...], indirect=True)
@pytest.mark.parametrize("paged_attention", [True, False])
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("max_seq_len", (256,))
def test_attention_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    pcc = 0.99

    # Setup model args and load state dict
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    state_dict = model_args.load_state_dict()

    # Create reference and TT models
    reference_model = model_args.reference_attention()
    reference_model.load_state_dict(partial_state_dict)

    tt_model = Attention(mesh_device, ...)

    # Run inference loop
    for i in range(generation_length):
        # Create input
        pt_attention_input = torch.randn(batch_size, seq_len, dim)

        # Run PyTorch reference
        reference_output = reference_model(pt_attention_input, ...)

        # Run TTNN
        tt_out = tt_model(attention_input, ...)
        tt_output_torch = ttnn.to_torch(tt_out, ...)

        # Compare
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)
```

#### Key Utility Functions

From `models/common/utility_functions.py`:

```python
from models.common.utility_functions import comp_allclose, comp_pcc
```

- `comp_pcc(ref, test, threshold)` - Returns `(passing, pcc_message)`
- `comp_allclose(ref, test)` - Returns detailed allclose comparison

### 2.2 PCC Thresholds

From existing tests:

| Component | Typical PCC Threshold | Notes |
|-----------|----------------------|-------|
| Full attention | 0.99 | End-to-end attention block |
| KV cache | 0.99 | After cache update |
| Individual ops | 0.999 | For isolated ops like matmul |

### 2.3 Test File Location Convention

Based on existing structure:

```
models/
  experimental/
    tt_symbiote/
      tests/
        attention/
          test_ling_attention.py          # Full attention tests
          test_ling_qkv_projection.py     # QKV projection tests
          test_ling_qk_norm.py            # QK normalization tests
          test_ling_partial_rope.py       # Partial RoPE tests
          test_ling_sdpa.py               # SDPA tests
```

Alternatively, following `tt_transformers` pattern:

```
models/
  tt_transformers/
    tests/
      ling/
        test_ling_attention.py
        test_ling_attention_prefill.py
```

---

## 3. Step-by-Step Implementation Plan

### Phase 1: Test Infrastructure Setup

#### Step 1.1: Create test directory structure

```bash
mkdir -p models/experimental/tt_symbiote/tests/attention
```

#### Step 1.2: Create test utilities module

**File:** `models/experimental/tt_symbiote/tests/attention/test_utils.py`

```python
"""Test utilities for Ling attention tests."""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from models.common.utility_functions import comp_pcc, comp_allclose

def get_ling_config():
    """Load Ling model config."""
    config = AutoConfig.from_pretrained(
        "inclusionAI/Ling-mini-2.0",
        trust_remote_code=True
    )
    return config

def get_ling_attention_layer(layer_idx=0):
    """Load a single attention layer from Ling model."""
    model = AutoModelForCausalLM.from_pretrained(
        "inclusionAI/Ling-mini-2.0",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    return model.model.layers[layer_idx].self_attn

def create_position_embeddings(config, seq_len, device='cpu'):
    """Create cos/sin position embeddings for Ling."""
    # Ling uses partial rotary: rotary_dim = head_dim * partial_rotary_factor
    head_dim = config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)
    rotary_dim = int(head_dim * partial_rotary_factor)

    # Compute freqs
    inv_freq = 1.0 / (config.rope_theta ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
    ))

    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, rotary_dim/2]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    # Expand to full rotary_dim (interleaved)
    cos = torch.cat([cos, cos], dim=-1)  # [1, 1, seq_len, rotary_dim]
    sin = torch.cat([sin, sin], dim=-1)

    return cos.to(device), sin.to(device)
```

### Phase 2: Component Tests

#### Step 2.1: QKV Projection Test

**File:** `test_ling_qkv_projection.py`

**Purpose:** Verify fused QKV projection matches separate Q/K/V projections used in distributed mode.

**Test Cases:**
1. Fused QKV vs separate Q, K, V (PyTorch comparison)
2. TTNN fused vs TTNN separate (verify split works)
3. Different batch sizes: 1, 8, 32
4. Different sequence lengths: 1 (decode), 32, 128, 256 (prefill)

**Key assertions:**
- Output shapes match: `[B, S, (num_heads + 2*num_kv_heads) * head_dim]`
- After split: Q `[B, S, num_heads * head_dim]`, K/V `[B, S, num_kv_heads * head_dim]`
- PCC > 0.999 for projection outputs

```python
def test_qkv_projection_equivalence():
    """Test QKV projection TTNN vs PyTorch."""
    # Setup
    config = get_ling_config()
    torch_attn = get_ling_attention_layer()

    batch_size, seq_len = 1, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # PyTorch fused QKV
    qkv_pt = torch_attn.query_key_value(hidden_states)

    # Split in PyTorch
    q_size = config.num_attention_heads * (config.hidden_size // config.num_attention_heads)
    kv_size = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)

    q_pt = qkv_pt[..., :q_size]
    k_pt = qkv_pt[..., q_size:q_size+kv_size]
    v_pt = qkv_pt[..., q_size+kv_size:]

    # TTNN implementation
    ttnn_attn = TTNNBailingMoEAttention.from_torch(torch_attn, distributed=False)
    # ... move to device and run

    # Compare
    assert comp_pcc(q_pt, q_ttnn, 0.999)[0], "Q projection mismatch"
    assert comp_pcc(k_pt, k_ttnn, 0.999)[0], "K projection mismatch"
    assert comp_pcc(v_pt, v_ttnn, 0.999)[0], "V projection mismatch"
```

#### Step 2.2: QK Normalization Test

**File:** `test_ling_qk_norm.py`

**Purpose:** Verify QK RMSNorm matches between PyTorch (4D) and TTNN (2D reshape approach).

**Test Cases:**
1. Q normalization: `[B, num_heads, S, head_dim]` input
2. K normalization: `[B, num_kv_heads, S, head_dim]` input
3. Different head counts and sequence lengths
4. Edge case: seq_len=1 (decode)

**Key assertions:**
- Output dtype matches (bfloat16)
- RMSNorm eps = 1e-6 applied correctly
- PCC > 0.99 (lower threshold due to reshape/typecast)

```python
def test_qk_normalization():
    """Test QK normalization TTNN vs PyTorch."""
    config = get_ling_config()
    torch_attn = get_ling_attention_layer()

    batch_size, seq_len = 1, 32
    num_heads = config.num_attention_heads  # 16
    num_kv_heads = config.num_key_value_heads  # 4
    head_dim = config.hidden_size // num_heads  # 128

    # Create Q and K tensors
    q_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    # PyTorch normalization (operates on 4D tensor directly)
    q_normed_pt = torch_attn.query_layernorm(q_states)
    k_normed_pt = torch_attn.key_layernorm(k_states)

    # TTNN approach: reshape to 2D, normalize, reshape back
    # ... TTNN implementation

    assert comp_pcc(q_normed_pt, q_normed_ttnn, 0.99)[0]
    assert comp_pcc(k_normed_pt, k_normed_ttnn, 0.99)[0]
```

#### Step 2.3: Partial RoPE Test

**File:** `test_ling_partial_rope.py`

**Purpose:** Verify partial RoPE (rotary_factor=0.5) applies correctly to only first 64 dims of head_dim=128.

**Test Cases:**
1. Single position (decode)
2. Multiple positions (prefill)
3. Verify non-rotated dims (64-127) are unchanged
4. Verify rotated dims (0-63) match reference

**Key assertions:**
- First 64 dims get rotation applied
- Last 64 dims are identity (pass-through)
- Output shape unchanged
- PCC > 0.999 for rotated portion

```python
def test_partial_rope():
    """Test partial RoPE with rotary_factor=0.5."""
    config = get_ling_config()
    head_dim = 128
    rotary_dim = 64  # partial_rotary_factor = 0.5

    # Create test tensors
    q = torch.randn(1, 16, 32, head_dim)  # [B, H, S, D]
    k = torch.randn(1, 4, 32, head_dim)

    cos, sin = create_position_embeddings(config, seq_len=32)

    # PyTorch partial RoPE
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotation to first half only
    q_embed_pt = torch.cat([apply_rotation(q_rot, cos, sin), q_pass], dim=-1)
    k_embed_pt = torch.cat([apply_rotation(k_rot, cos, sin), k_pass], dim=-1)

    # TTNN partial RoPE
    # ... TTNN implementation

    # Verify pass-through dims unchanged
    assert torch.allclose(q_embed_pt[..., rotary_dim:], q[..., rotary_dim:])

    # Verify rotated portion matches
    assert comp_pcc(q_embed_pt, q_embed_ttnn, 0.999)[0]
```

#### Step 2.4: SDPA Tests (Prefill)

**File:** `test_ling_sdpa_prefill.py`

**Purpose:** Verify SDPA prefill produces correct attention output.

**Test Cases:**
1. Standard prefill: seq_len=128
2. Short prefill: seq_len=32
3. Long prefill: seq_len=1024
4. With/without causal mask

**Key assertions:**
- Output shape: `[B, S, num_heads, head_dim]` before reshape
- Causal masking applied correctly
- PCC > 0.99 against `torch.nn.functional.scaled_dot_product_attention`

```python
def test_sdpa_prefill():
    """Test SDPA prefill TTNN vs PyTorch."""
    batch_size, seq_len = 1, 128
    num_heads, num_kv_heads, head_dim = 16, 4, 128

    # Create inputs after QK norm and RoPE
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    # PyTorch: expand KV for GQA
    k_expanded = repeat_kv(k, num_heads // num_kv_heads)  # [B, 16, S, D]
    v_expanded = repeat_kv(v, num_heads // num_kv_heads)

    # PyTorch SDPA
    attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    out_pt = torch.nn.functional.scaled_dot_product_attention(
        q, k_expanded, v_expanded, attn_mask=attn_mask
    )

    # TTNN SDPA (handles GQA internally)
    # ... TTNN implementation

    assert comp_pcc(out_pt, out_ttnn, 0.99)[0]
```

#### Step 2.5: SDPA Tests (Decode with Paged Attention)

**File:** `test_ling_sdpa_decode.py`

**Purpose:** Verify paged SDPA decode produces correct output for GQA.

**Test Cases:**
1. First decode step (cur_pos=0)
2. Mid-sequence decode (cur_pos=50)
3. Various batch sizes
4. Verify `cur_pos` semantics match

**Key assertions:**
- Output shape: `[1, B, num_heads, head_dim]`
- `cur_pos` is 0-indexed next-write position
- KV cache updated correctly
- PCC > 0.99 against reference

```python
def test_paged_sdpa_decode():
    """Test paged SDPA decode for GQA."""
    batch_size = 1
    num_heads, num_kv_heads, head_dim = 16, 4, 128
    max_seq_len = 256
    block_size = 32

    # Setup paged KV cache
    # ...

    # First decode step
    cur_pos = torch.tensor([10], dtype=torch.int32)  # 10 tokens already cached
    q = torch.randn(1, batch_size, num_heads, head_dim)  # [1, B, H, D]

    # PyTorch reference with full KV cache
    # ...

    # TTNN paged decode
    # ...

    assert comp_pcc(out_pt, out_ttnn, 0.99)[0]
```

### Phase 3: Full Attention Block Tests

#### Step 3.1: Full Forward Pass Test (Prefill)

**File:** `test_ling_attention.py`

**Purpose:** End-to-end attention block test for prefill.

**Test Cases:**
1. Standard prefill with DynamicCache
2. Prefill with paged attention
3. Multiple sequence lengths
4. Verify KV cache contents

```python
@pytest.mark.parametrize("seq_len", [32, 128, 256])
@pytest.mark.parametrize("paged_attention", [True, False])
def test_ling_attention_prefill(seq_len, paged_attention, mesh_device):
    """Full attention prefill test."""
    config = get_ling_config()
    torch_attn = get_ling_attention_layer()

    batch_size = 1
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    cos, sin = create_position_embeddings(config, seq_len)

    # PyTorch reference
    out_pt, _, cache_pt = torch_attn(
        hidden_states,
        position_embeddings=(cos, sin),
        past_key_value=DynamicCache(),
    )

    # TTNN implementation
    ttnn_attn = TTNNBailingMoEAttention.from_torch(torch_attn)
    # ... setup and run

    passing, msg = comp_pcc(out_pt, out_ttnn, 0.99)
    assert passing, f"Attention prefill failed: {msg}"
```

#### Step 3.2: Full Forward Pass Test (Decode)

**File:** `test_ling_attention_decode.py`

**Purpose:** End-to-end attention block test for autoregressive decode.

**Test Cases:**
1. Sequential decode steps (position 0, 1, 2, ...)
2. Batched decode
3. Verify accumulated KV cache

```python
def test_ling_attention_decode(mesh_device):
    """Full attention decode test with KV cache."""
    config = get_ling_config()
    torch_attn = get_ling_attention_layer()

    batch_size = 1
    max_seq_len = 256
    generation_length = 10

    # Initialize caches
    cache_pt = DynamicCache()
    cache_ttnn = TTNNPagedAttentionKVCache(...)

    # Prefill first
    prefill_len = 32
    # ...

    # Decode loop
    for step in range(generation_length):
        cur_pos = prefill_len + step
        hidden_states = torch.randn(batch_size, 1, config.hidden_size)
        cos, sin = create_position_embeddings(config, cur_pos + 1)

        # PyTorch reference
        out_pt, _, cache_pt = torch_attn(
            hidden_states,
            position_embeddings=(cos[:, :, cur_pos:cur_pos+1], sin[:, :, cur_pos:cur_pos+1]),
            past_key_value=cache_pt,
        )

        # TTNN
        # ...

        passing, msg = comp_pcc(out_pt, out_ttnn, 0.99)
        assert passing, f"Decode step {step} failed: {msg}"
```

---

## 4. Test File Structure

```
models/experimental/tt_symbiote/tests/attention/
    __init__.py
    conftest.py                      # Shared fixtures
    test_utils.py                    # Test utilities

    # Component tests
    test_ling_qkv_projection.py      # QKV projection
    test_ling_qk_norm.py             # QK normalization
    test_ling_partial_rope.py        # Partial RoPE
    test_ling_sdpa_prefill.py        # SDPA prefill
    test_ling_sdpa_decode.py         # Paged SDPA decode

    # Integration tests
    test_ling_attention.py           # Full prefill
    test_ling_attention_decode.py    # Full decode
```

### conftest.py

```python
import pytest
import torch
import ttnn

@pytest.fixture
def reset_seeds():
    torch.manual_seed(42)
    yield

@pytest.fixture
def mesh_device():
    """Single device fixture."""
    device = ttnn.open_device(0)
    yield device
    ttnn.close_device(device)

@pytest.fixture
def t3k_mesh_device():
    """T3K mesh device fixture."""
    device_ids = ttnn.get_device_ids()
    if len(device_ids) < 8:
        pytest.skip("T3K requires 8 devices")
    mesh = ttnn.open_mesh_device(device_ids[:8])
    yield mesh
    ttnn.close_mesh_device(mesh)

@pytest.fixture
def ensure_gc():
    yield
    import gc
    gc.collect()
```

---

## 5. Success Criteria

### PCC Thresholds

| Component | Threshold | Rationale |
|-----------|-----------|-----------|
| QKV projection | > 0.999 | Direct matmul, should be very close |
| QK normalization | > 0.99 | Reshape operations may introduce minor variance |
| Partial RoPE | > 0.999 | Math should match exactly |
| SDPA prefill | > 0.99 | Softmax and matmul accumulation |
| SDPA decode (paged) | > 0.98 | Known Issue #30362, some positions may have lower PCC |
| Full attention (prefill) | > 0.99 | End-to-end |
| Full attention (decode) | > 0.98 | Includes paged cache |

### Test Coverage Requirements

1. **Batch sizes:** 1, 8, 32 (where applicable)
2. **Sequence lengths:**
   - Decode: 1
   - Prefill: 32, 128, 256, 1024
3. **Modes:**
   - Non-distributed (single device)
   - Distributed (T3K mesh)
4. **Cache types:**
   - DynamicCache
   - TTNNPagedAttentionKVCache

### Known Issue Handling

Per `guides/paged_sdpa_decode_for_gqa/`:

1. **Issue #30362**: Sporadic PCC failures at certain `cur_pos` values in paged decode. Test at multiple positions to isolate.
2. **Grid-size constraint**: Ensure `num_cores >= batch_size * num_kv_heads`
3. **Padding-collapse bug**: Verify `nh_padded / nkv_padded == original_group_size`

---

## 6. Implementation Priority

### Phase 1 (Immediate - Diagnose Issue)
1. `test_ling_attention.py` - Full prefill test to establish baseline
2. `test_ling_sdpa_decode.py` - Paged SDPA decode test (likely issue source)

### Phase 2 (Component Isolation)
3. `test_ling_qkv_projection.py`
4. `test_ling_qk_norm.py`
5. `test_ling_partial_rope.py`
6. `test_ling_sdpa_prefill.py`

### Phase 3 (Comprehensive)
7. `test_ling_attention_decode.py` - Full decode loop
8. Distributed mode tests
9. Edge case tests (max seq len, boundary positions)

---

## 7. References

### Source Files
- PyTorch reference: `/home/ttuser/.cache/huggingface/modules/transformers_modules/inclusionAI/Ling-mini-2.0/ae2925e082ef9e311fbbb01f2720006611bbdb69/modeling_bailing_moe_v2.py`
- TTNN implementation: `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py`
- Comparison document: `/home/ttuser/salnahari/research-topics/tt-symbiote-research-topics/ling_attention_implementations.md`

### Test Patterns
- `/home/ttuser/tt-metal/models/tt_transformers/tests/test_attention.py`
- `/home/ttuser/tt-metal/models/tt_transformers/tests/test_attention_prefill.py`
- `/home/ttuser/tt-metal/models/demos/t3000/mixtral8x7b/tests/test_mixtral_attention.py`

### Guides
- `/home/ttuser/salnahari/research-topics/tt-symbiote-research-topics/guides/paged_sdpa_decode_for_gqa/`
