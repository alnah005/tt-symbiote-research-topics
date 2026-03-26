# PLAN: Ling-mini-2.0 Correct Text Generation

## 14. Iteration 4 - Baseline Testing and Fundamental Analysis

### 14.1 Was the Original Code Also Producing Garbled Output?

**Answer: YES - the original committed code also FAILS the test.**

The original (committed) code at `HEAD:models/experimental/tt_symbiote/modules/attention.py`
was tested by stashing all local modifications and running:
```
pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v --timeout=0
```

**Result:** `1 failed, 1 warning in 617.75s (0:10:17)` -- the test crashed with C-level
stack traces (segfault or similar). The original code was never producing correct output.

**Key Implication:** All 3 prior iterations of "fixes" were trying to fix something that was
fundamentally broken from the start. The original committed code (`0ccfd9fa911`) already had
the decode path producing crashes/garbled output.

### 14.2 What Does the Test Replacement Mapping Look Like?

From `test_ling_mini_2_0.py` line 85-92:
```python
nn_to_ttnn = {
    # model.model.layers[1].mlp.__class__: TTNNBailingMoE,  # COMMENTED OUT
    model.model.layers[0].attention.__class__: TTNNBailingMoEAttention,
}
nn_to_ttnn2 = {
    # nn.Linear: TTNNLinearIColShardedWRowSharded,  # COMMENTED OUT
    # nn.SiLU: TTNNSilu,  # COMMENTED OUT
}
```

The `register_module_replacement_dict` function is CLASS-based: it replaces ALL instances of
`model.model.layers[0].attention.__class__` (which is `BailingMoeV2SdpaAttention`). Since ALL
20 layers use the same class, **ALL 20 attention layers are replaced**, not just layer 0.

This means:
- ALL 20 attention layers use TTNNBailingMoEAttention
- ALL 20 layers use TTNN paged attention
- If ANY layer's attention is wrong, the ENTIRE model output is garbled
- The MoE/MLP layers remain as PyTorch (the replacement is commented out)
- No linear layers or activations are replaced either

### 14.3 ROOT CAUSE ANALYSIS: Three Bugs Found

#### BUG 1 (CRITICAL - in modified code only): `past_key_value` parameter name mismatch

The HuggingFace BailingMoeV2 decoder layer calls attention with:
```python
hidden_states, self_attn_weights, present_key_value = self.attention(
    hidden_states=hidden_states,
    past_key_value=past_key_value,   # <-- SINGULAR
    position_embeddings=position_embeddings,
    ...
)
```

The **original committed** TTNNBailingMoEAttention.forward() had:
```python
def forward(self, hidden_states, position_embeddings, ..., past_key_values=None,
            past_key_value=None, ...):
    if past_key_value is not None and past_key_values is None:
        past_key_values = past_key_value  # fallback mapping
```

The **modified** TTNNBailingMoEAttention.forward() REMOVED the `past_key_value` parameter:
```python
def forward(self, hidden_states, position_embeddings, ..., past_key_values=None,
            cache_position=None, **kwargs):
    # past_key_value goes into **kwargs and is SILENTLY IGNORED
```

**Impact:** With the modified code, `past_key_values` is ALWAYS `None`:
- Prefill: KV cache is never populated (paged_fill_on_device is skipped)
- Decode: The condition `past_key_values is not None and seq_length == 1` is always False
- Every call goes to `_forward_prefill` with no cache, even for decode tokens
- Model output is essentially random/garbled because there is no KV cache context

#### BUG 2 (in BOTH original and modified code): Double-counting _seq_lengths

The `paged_update_on_device` method at line 206 already increments `_seq_lengths`:
```python
def paged_update_on_device(self, key_states, value_states, layer_idx, current_pos):
    ...
    seq_len = key_states.shape[0]
    self._seq_lengths[layer_idx] += seq_len   # <-- increments by 1
    if layer_idx == 0:
        self._seen_tokens += seq_len          # <-- increments by 1
```

The original committed `_forward_decode_paged` at the call site ALSO increments:
```python
past_key_values._seq_lengths[layer_idx] += seq_length  # <-- DOUBLE COUNT
if layer_idx == 0:
    past_key_values._seen_tokens += seq_length          # <-- DOUBLE COUNT
```

This means every decode step advances `_seq_lengths` by 2 instead of 1. On the next
decode step, `get_seq_length()` returns position N+2 instead of N+1, so:
- `cur_pos_tt` is wrong (position skips by 2)
- KV cache writes to wrong positions (every other slot is skipped)
- paged_sdpa_decode reads KV with wrong position info

The modified code correctly removed this double-counting, but it doesn't matter because
Bug 1 prevents the decode path from ever being reached.

Note: The same double-counting bug exists in `TTNNGlm4MoeLiteAttention._forward_decode_paged`
(the GLM model). However, GLM may still appear to work because the position skip may not
cause catastrophic output for short sequences.

#### BUG 3 (in original committed code): Crash in original decode path

The original committed code crashed with a segfault during the test. This is likely because
the original decode path uses `permute(query_states, (2, 0, 1, 3))` (B H S D -> S B H D)
and then attempts to convert topology with `_to_replicated()` which round-trips through the
host. The combination of topology conversion + memory config issues in the original code may
have been causing the crash.

### 14.4 The Simplest Path to Coherent Text

**Priority 1: Fix the parameter name mismatch (Bug 1)**

In `TTNNBailingMoEAttention.forward()`, add back `past_key_value` parameter support:

```python
def forward(
    self,
    hidden_states: ttnn.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[ttnn.Tensor] = None,
    past_key_values=None,
    past_key_value=None,          # <-- ADD THIS BACK
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
    # Map singular -> plural for HuggingFace compatibility
    if past_key_value is not None and past_key_values is None:
        past_key_values = past_key_value
    ...
```

This single fix should:
- Allow the KV cache to be properly passed to the attention module
- Enable the prefill path to populate the KV cache
- Enable the decode path to be reached (seq_length == 1 with non-None cache)
- Allow decode to read from a properly-populated KV cache

**Priority 2: Verify the double-counting bug is fixed (Bug 2)**

The modified code already removed the double-counting from `_forward_decode_paged`. Verify
that `_forward_prefill` also does not double-count (it calls `paged_fill_on_device` which
already increments, so no additional increment should be present).

Current modified prefill code (line ~2589-2596):
```python
if past_key_values is not None:
    layer_idx = self._fallback_torch_layer.layer_idx
    past_key_values.paged_fill_on_device(
        key_states, value_states, layer_idx=layer_idx, batch_idx=0,
    )
```
This is correct - no additional _seq_lengths increment.

**Priority 3: Verify all other aspects of the decode path**

After fixing Bug 1, the decode path should be exercised. The modified decode path uses:
- `nlp_create_qkv_heads_decode` for head splitting
- `BailingRotarySetup` for decode-mode RoPE
- `rotary_embedding_llama` kernel
- `nlp_concat_heads_decode` for output

These are all proven patterns from TT Transformers, so they should work IF the data
flows correctly (proper shapes, memory configs, topologies).

### 14.5 Comparison with Working GLM Model

GLM Flash test (`test_glm_flash.py`) uses `TTNNGlm4MoeLiteAttention` which:
- Also replaces ALL attention layers (same class-based replacement)
- Also replaces MoE layers (TTNNMoE) AND all linear layers (TTNNLinearIColShardedWRowSharded)
- Uses the same paged attention infrastructure
- Its `forward()` signature does NOT have `past_key_value` (singular) because GLM's
  decoder layer uses `past_key_values` (plural) - this is the standard HuggingFace naming

The Bailing model is different because it uses the older naming convention (`past_key_value`
singular). This is why the GLM model doesn't need the fallback mapping.

### 14.6 Summary of Changes Needed

| Fix | File | Change | Impact |
|-----|------|--------|--------|
| Bug 1 | attention.py | Add `past_key_value=None` param + fallback mapping | **Critical** - without this, no KV cache works |
| Bug 2 | attention.py | Already fixed in modified code (removed double-count) | Position tracking correctness |
| Bug 3 | attention.py | Already addressed by modified decode path using nlp_create_qkv_heads_decode | Crash prevention |

### 14.7 Why 3 Iterations Failed to Find This

The previous iterations focused on:
1. RoPE implementation (partial rotary, identity padding, decode-mode kernels)
2. Memory config / sharding (HEIGHT_SHARDED for nlp_create_qkv_heads_decode)
3. Tensor topology (replicated vs all-gathered)

None of these mattered because the KV cache was never being passed to the module in the
first place due to the `past_key_value` vs `past_key_values` naming mismatch. The entire
decode path was a dead code path - it was never executed. All tokens were going through
the prefill path with no cache, producing random outputs.

This is a classic "plumbing before optimization" issue. The fundamental data flow was broken
at the API boundary between HuggingFace and the TTNN module.

## 15. Iteration 5 - Repetition Loop Diagnosis

### 15.1 Symptom

The test now runs without crashing (Bugs 1 and 2 from Iteration 4 were fixed), but the
output is **degenerate** - stuck in a repetition loop:

```
Ling-mini-2.0 PAGED ATTENTION OUTPUT: It's a programming language?

It's a programming language?
It's a programming language?
... (repeats ~18 times)
```

The prompt was "What is your favorite condiment?" and the response is completely unrelated
and repeating. This indicates that the decode path is now being reached (which is progress
from Iteration 4), but the attention mechanism is producing incorrect results - likely
attending to the wrong positions or producing garbage attention scores.

### 15.2 ROOT CAUSE: RoPE Format Mismatch Between Prefill and Decode

**This is the primary bug causing the repetition loop.**

#### The Two RoPE Formats

There are two different mathematical representations of rotary position embeddings:

**HuggingFace (HF) Format:**
- cos/sin shape: `[batch, seq, rotary_dim]` where `rotary_dim = head_dim * partial_rotary_factor`
- cos = `[c0, c1, c2, ..., c31, c0, c1, ..., c31]` (freqs concatenated with themselves)
- Rotation: `q * cos + rotate_half(q) * sin` where `rotate_half` splits at `dim/2`
- This is what HuggingFace's `apply_rotary_pos_emb` and `ttnn.experimental.rotary_embedding` use

**Meta Format:**
- cos/sin: `[c0, c0, c1, c1, ..., c31, c31]` (each frequency doubled, interleaved)
- Rotation uses a transformation matrix (`trans_mat`) that swaps adjacent pairs: `[x0, x1] -> [-x1, x0]`
- This is what `ttnn.experimental.rotary_embedding_llama` uses

Both formats produce the same mathematical result IF cos/sin are in the matching format.
But if HF-format cos/sin is fed to a Meta-format kernel (or vice versa), the rotation is wrong.

#### Where the Mismatch Occurs

**Prefill path** (`_forward_prefill`, lines 2548-2618):
```python
cos, sin = position_embeddings  # From HuggingFace BailingMoeV2RotaryEmbedding -> HF format
query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin)
# -> calls self.rope() which is TTNNRotaryPositionEmbedding
# -> calls ttnn.experimental.rotary_embedding(q_rot, cos, sin)  # HF-format kernel
```

The prefill stores K in the paged KV cache with HF-format rotation applied.

**Decode path** (`_forward_decode_paged`, lines 2620-2837):
```python
cos_ttnn, sin_ttnn = self._rotary_setup.get_cos_sin_for_decode(cache_position_tensor)
# -> BailingRotarySetup._compute_cos_sin_cache uses Meta format:
#    cos = cos[:, : cos.shape[1] // 2]
#    cos = torch.stack((cos, cos), dim=-1).flatten(-2)
# -> Returns Meta-format cos/sin

query_states = ttnn.experimental.rotary_embedding_llama(
    query_states, cos_ttnn, sin_ttnn, trans_mat, is_decode_mode=True
)  # Meta-format kernel
```

The decode computes Q with Meta-format rotation.

**Result:** During `paged_sdpa_decode`, the Q (Meta-format rotated) attends to K (HF-format
rotated) in the cache. Since the rotations don't match, the attention scores are garbage.
This explains both:
1. **Wrong content** ("programming language?" instead of something about condiments) -
   because attention is not correctly matching positions
2. **Repetition** - because the same garbage attention pattern repeats every decode step,
   since the format mismatch is systematic (not random)

#### Mathematical Proof of Mismatch

For position `p` and frequency index `i`:

**HF format cos at indices [i, i+32]:** `[cos(p * freq_i), cos(p * freq_i)]`
**Meta format cos at indices [2i, 2i+1]:** `[cos(p * freq_i), cos(p * freq_i)]`

These look the same, BUT the rotation operation is different:

**HF rotate_half:** splits at `dim//2`, so `[x0,...,x31, x32,...,x63]` -> `[-x32,...,-x63, x0,...,x31]`
**Meta trans_mat:** swaps adjacent pairs, so `[x0, x1, x2, x3,...]` -> `[-x1, x0, -x3, x2,...]`

With HF cos/sin = `[c0, c1, ..., c31, c0, c1, ..., c31]`:
- HF: `x[i]*c[i] + (-x[i+32])*s[i]` for i in [0..31]
- HF: `x[i+32]*c[i+32] + x[i]*s[i+32]` for i in [0..31]

With Meta cos/sin = `[c0, c0, c1, c1, ..., c31, c31]`:
- Meta: `x[2i]*c[2i] + (-x[2i+1])*s[2i]` = `x[2i]*c[i] - x[2i+1]*s[i]`
- Meta: `x[2i+1]*c[2i+1] + x[2i]*s[2i+1]` = `x[2i+1]*c[i] + x[2i]*s[i]`

These are fundamentally different operations on the data layout.

#### Secondary Issue: `attention_scaling` Not Applied in BailingRotarySetup

The HuggingFace `BailingMoeV2RotaryEmbedding.forward()` applies `attention_scaling`:
```python
cos = emb.cos() * self.attention_scaling
sin = emb.sin() * self.attention_scaling
```

The `BailingRotarySetup._compute_cos_sin_cache()` does NOT apply this scaling. If
`attention_scaling != 1.0` (which can happen with certain rope_type configurations),
this would be another source of numerical error. For the default rope_type, `attention_scaling`
is typically 1.0, so this may not matter for Ling-mini-2.0.

### 15.3 Additional Potential Issues (Lower Priority)

#### Issue A: Dense Layer Receiving Padded Batch

After `nlp_concat_heads_decode`, the output shape is `[1, 1, 32, H*D]` where padded_batch=32
but actual batch=1. The code passes this directly to `self.dense()` (a
`TTNNLinearIReplicatedWColSharded`), which runs the linear on all 32 rows. Then it slices
to batch=1 afterwards (line 2830-2831). While mathematically correct (the padding rows are
zero, so the linear output for those rows is just the bias), this is wasteful.

The GLM decode path (lines 1937-1942) does NOT use `nlp_concat_heads_decode`. Instead it
does:
```python
attn_output = ttnn.permute(attn_output, (1, 0, 2, 3))  # [B, 1, H, D]
attn_output = ttnn.reshape(attn_output, (batch_size, seq_length, num_heads * v_head_dim))
attn_output = self.o_proj(attn_output)
```

This difference is not necessarily a bug, but the padded batch could interact badly with
bias terms or normalization in downstream layers.

#### Issue B: `paged_sdpa_decode` Program Config Differences

The Bailing decode uses `self.sdpa.decode_program_config`:
```python
ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    q_chunk_size=0,
    k_chunk_size=0,
    exp_approx_mode=False,
)
```

The GLM decode uses `self.sdpa.program_config`:
```python
ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    q_chunk_size=256,
    k_chunk_size=256,
    exp_approx_mode=False,
)
```

Using chunk_size=0 vs 256 may affect the SDPA kernel behavior. The decode program config
with q_chunk_size=0 and k_chunk_size=0 is the standard config for decode in TT Transformers
(where sequence length is always 1), so this should be fine.

### 15.4 Fix Plan

#### Fix 1 (CRITICAL): Unify RoPE Format Between Prefill and Decode

**Option A (Recommended): Use BailingRotarySetup for BOTH prefill and decode**

Modify `_forward_prefill` to ignore the HuggingFace `position_embeddings` and instead use
`BailingRotarySetup.get_cos_sin_for_prefill()`. This ensures both paths use the same
Meta-format cos/sin with `rotary_embedding_llama`.

Changes needed in `_forward_prefill()` (around line 2579-2586):
```python
# BEFORE (uses HF format from position_embeddings):
cos, sin = position_embeddings
query_states, key_states = self._apply_partial_rope(query_states, key_states, cos, sin)

# AFTER (uses BailingRotarySetup for Meta format):
seq_len = query_states.shape[2]  # seq dimension in [B, H, S, D]
cos, sin = self._rotary_setup.get_cos_sin_for_prefill(seq_len)
trans_mat = self._rotary_setup.get_trans_mat(is_decode=False)  # [1, 1, head_dim, head_dim]
query_states = ttnn.experimental.rotary_embedding_llama(
    query_states, cos, sin, trans_mat, is_decode_mode=False
)
key_states = ttnn.experimental.rotary_embedding_llama(
    key_states, cos, sin, trans_mat, is_decode_mode=False
)
```

NOTE: The prefill `rotary_embedding_llama` call uses `is_decode_mode=False` and the prefill
trans_mat (head_dim x head_dim), while decode uses `is_decode_mode=True` and the decode
trans_mat (TILE_SIZE x TILE_SIZE). Both use the same Meta-format cos/sin from BailingRotarySetup.

NOTE: The cos/sin from `get_cos_sin_for_prefill` already include identity padding for
partial_rotary_factor < 1.0, so applying RoPE to the full head_dim is correct - the
identity-padded dims will pass through unchanged.

NOTE: This removes the need for `TTNNRotaryPositionEmbedding` and `_apply_partial_rope`
in the Bailing attention entirely. The `self.rope` field is no longer used.

**Option B (Alternative): Convert decode to use HF format**

This would mean changing `BailingRotarySetup._compute_cos_sin_cache` to produce HF-format
cos/sin and using `ttnn.experimental.rotary_embedding` in the decode path. However, this is
more complex because `rotary_embedding` does not support the decode tensor layout `[1, B, H, D]`
and would require reshape operations. Option A is simpler and follows the TT Transformers
pattern.

#### Fix 2 (SAFETY): Verify `attention_scaling`

After fixing the RoPE format, verify that the Ling-mini-2.0 model's `attention_scaling` is
1.0. If not, `BailingRotarySetup._compute_cos_sin_cache` must be updated to multiply
cos/sin by `attention_scaling`.

To check: `model.model.layers[0].attention.rotary_emb.attention_scaling`

If it's not 1.0, add a `scaling` parameter to `BailingRotarySetup.__init__` and apply it:
```python
cos = cos * scaling
sin = sin * scaling
```

#### Fix 3 (CLEANUP): Remove unused TTNNRotaryPositionEmbedding from Bailing

After Fix 1, the `from_torch` method should no longer create `TTNNRotaryPositionEmbedding`
for partial_rotary. The `self.rope` field becomes unused. Clean up:

```python
# In from_torch(), remove:
#   uses_partial_rotary = new_attn.partial_rotary_factor < 1.0
#   if uses_partial_rotary:
#       new_attn.rope = TTNNRotaryPositionEmbedding()
#   else:
#       new_attn.rope = TTNNDistributedRotaryPositionEmbedding()

# And remove: _apply_partial_rope method
```

### 15.5 Implementation Priority

| Priority | Fix | Impact | Effort |
|----------|-----|--------|--------|
| P0 | Fix 1: Unify RoPE format | **Critical** - root cause of repetition loop | Medium |
| P1 | Fix 2: Verify attention_scaling | Safety - could cause subtle errors | Low |
| P2 | Fix 3: Remove unused RoPE code | Code cleanliness | Low |

### 15.6 Expected Outcome After Fixes

After applying Fix 1, both prefill and decode will use the same Meta-format RoPE via
`BailingRotarySetup` and `rotary_embedding_llama`. The K values stored in the KV cache
during prefill will be compatible with Q values computed during decode, producing correct
attention scores. This should eliminate the repetition loop and produce coherent text.

### 15.7 How to Verify the Fix

1. Run `pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v --timeout=0`
2. Check that the output is:
   - Related to the prompt ("What is your favorite condiment?")
   - Not repetitive
   - Grammatically coherent
3. For a numerical sanity check, compare a few positions' cos/sin values from
   `BailingRotarySetup` against HuggingFace's `rotary_emb` to confirm they represent
   the same rotations (just in different formats).

## 16. Iteration 6 - trans_mat Shape Fix

### 16.1 The Error

The decode path in `TTNNBailingMoEAttention._forward_decode_paged` crashes with:
```
TT_FATAL @ rotary_embedding_llama_device_operation.cpp:153:
    trans_mat.logical_shape()[-2] == TILE_HEIGHT
info:
    Transformation matrix must have 3rd dim equal to TILE_HEIGHT
```

### 16.2 Root Cause

The `rotary_embedding_llama` kernel in decode mode (`is_decode_mode=True`) requires ALL
input tensors -- Q, K, cos, sin, AND trans_mat -- to be HEIGHT_SHARDED across `batch_size`
cores. The trans_mat must have one `(TILE_SIZE, TILE_SIZE)` shard on each core.

The current Bailing decode code does this:
```python
trans_mat = self._rotary_setup.get_trans_mat(is_decode=True)
# returns shape [1, 1, 32, 32] from DRAM (INTERLEAVED)

trans_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
    core_grid=batch_grid,               # batch_size cores
    strategy=ttnn.ShardStrategy.HEIGHT,
    ...
)
trans_mat = ttnn.to_memory_config(trans_mat, trans_shard_mem)
```

**The problem:** The trans_mat tensor has logical shape `[1, 1, 32, 32]`, which means its
total height is 32 (one tile). When you create a HEIGHT_SHARDED config with `batch_size`
cores and shard_shape `(32, 32)`, it expects the total tensor height to be
`batch_size * 32`. But the tensor only has height 32. The `ttnn.to_memory_config` call
either fails or produces a malformed sharded tensor whose shard_spec reports the wrong
shape, triggering the kernel's assertion on `trans_mat.shard_spec()->shape[0] == TILE_HEIGHT`.

### 16.3 How tt-transformers Solves This (The Working Reference)

In `models/tt_transformers/tt/rope.py` (lines 469-490), the trans_mat is **repeated
along the height dimension** before being placed into the sharded memory config:

```python
trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
    1, 1, self.batch_size_per_device_group, 1,
)  # shape: [1, 1, batch_size * TILE_SIZE, TILE_SIZE]

trans_mat_mem_config = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
    core_grid=self.batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
self.transformation_mat = ttnn.from_torch(
    trans_mat,
    device=device,
    layout=ttnn.TILE_LAYOUT,
    dtype=datatype,
    memory_config=trans_mat_mem_config,   # placed directly into sharded config
    mesh_mapper=replicate_tensor_to_mesh_mapper(device),
)
```

The same pattern appears in `models/common/modules/attention/attention_1d.py` (lines
1905-1921):
```python
trans_mat = get_rot_transformation_mat().repeat(1, 1, doubled_batch_size, 1)
```

**Key insight:** The `.repeat(1, 1, batch_size, 1)` call replicates the single
`[1, 1, 32, 32]` tile `batch_size` times along dim-2, producing a torch tensor of shape
`[1, 1, batch_size*32, 32]`. Each core then gets one `(32, 32)` shard -- an identical
copy of the transformation matrix.

### 16.4 The Fix

The fix must happen in `BailingRotarySetup.__init__` in `rope.py`, so that the
trans_mat_decode is pre-built with the correct repeated shape AND placed directly into
the HEIGHT_SHARDED memory config at initialization time (not at forward-pass time).

**File:** `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/rope.py`

**Change 1 -- BailingRotarySetup.__init__ (around line 536-544):**

Replace the current trans_mat_decode creation:
```python
# Create transformation matrix for decode (TILE_SIZE x TILE_SIZE)
trans_mat_decode_torch = _get_rotation_transformation_mat(ttnn.TILE_SIZE)
self.trans_mat_decode = ttnn.from_torch(
    trans_mat_decode_torch.to(torch.bfloat16),
    device=device,
    layout=ttnn.TILE_LAYOUT,
    dtype=datatype,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=mesh_mapper,
)
```

With:
```python
# Create transformation matrix for decode (TILE_SIZE x TILE_SIZE)
# Must be repeated batch_size times along dim-2 so it can be HEIGHT_SHARDED
# across batch_size cores (one [TILE_SIZE, TILE_SIZE] shard per core).
# This follows the pattern from tt-transformers' RotarySetup.
trans_mat_decode_torch = _get_rotation_transformation_mat(ttnn.TILE_SIZE)
# Store the torch version for later repetition (batch_size not known at init)
self._trans_mat_decode_torch = trans_mat_decode_torch.to(torch.bfloat16)
self.trans_mat_decode = None  # Will be lazily created on first decode call
```

However, this lazy approach is more complex. A simpler approach that follows the
tt-transformers pattern more closely requires knowing the batch_size at init time.

**Recommended approach -- accept batch_size as an __init__ parameter:**

```python
def __init__(
    self,
    device,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    partial_rotary_factor: float = 1.0,
    datatype: ttnn.DataType = ttnn.bfloat16,
    max_batch_size: int = 8,              # <-- NEW PARAMETER
) -> None:
```

Then create the trans_mat_decode with the repeated shape:
```python
# Create transformation matrix for decode, repeated across batch cores
trans_mat_decode_torch = _get_rotation_transformation_mat(ttnn.TILE_SIZE)
trans_mat_decode_torch = trans_mat_decode_torch.repeat(1, 1, max_batch_size, 1)
# shape: [1, 1, max_batch_size * TILE_SIZE, TILE_SIZE]

batch_grid = ttnn.num_cores_to_corerangeset(
    max_batch_size, device.compute_with_storage_grid_size(), True
)
trans_mat_decode_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
self.trans_mat_decode = ttnn.from_torch(
    trans_mat_decode_torch.to(torch.bfloat16),
    device=device,
    layout=ttnn.TILE_LAYOUT,
    dtype=datatype,
    memory_config=trans_mat_decode_mem,
    mesh_mapper=mesh_mapper,
)
```

**Change 2 -- attention.py `_forward_decode_paged` (around lines 2758-2767):**

Remove the trans_mat sharding code since it's now pre-sharded:
```python
# OLD (delete):
trans_mat = self._rotary_setup.get_trans_mat(is_decode=True)
trans_shard_mem = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
    core_grid=batch_grid,
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
trans_mat = ttnn.to_memory_config(trans_mat, trans_shard_mem)

# NEW (replace with):
trans_mat = self._rotary_setup.get_trans_mat(is_decode=True)
# trans_mat is already HEIGHT_SHARDED from BailingRotarySetup init
```

**Change 3 -- Update the call site where BailingRotarySetup is constructed:**

Find where `BailingRotarySetup(...)` is called and add `max_batch_size=<value>`.
This is likely in the attention module's `move_weights_to_device_impl` or similar init method.

### 16.5 Alternative: Lazy Initialization (No __init__ Change)

If adding a `max_batch_size` parameter to `BailingRotarySetup.__init__` is undesirable,
an alternative is to build the sharded trans_mat lazily on the first decode call:

In `BailingRotarySetup.__init__`, keep the DRAM version as-is and also store the torch tensor:
```python
self._trans_mat_decode_torch = trans_mat_decode_torch.to(torch.bfloat16)
self._trans_mat_decode_sharded_cache = {}  # batch_size -> sharded tensor
```

Add a new method:
```python
def get_trans_mat_decode_sharded(self, batch_size: int) -> ttnn.Tensor:
    """Get trans_mat pre-sharded for the given batch_size."""
    if batch_size not in self._trans_mat_decode_sharded_cache:
        trans_mat_torch = self._trans_mat_decode_torch.repeat(1, 1, batch_size, 1)
        batch_grid = ttnn.num_cores_to_corerangeset(
            batch_size, self.device.compute_with_storage_grid_size(), True
        )
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None
        self._trans_mat_decode_sharded_cache[batch_size] = ttnn.from_torch(
            trans_mat_torch,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.datatype,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )
    return self._trans_mat_decode_sharded_cache[batch_size]
```

Then in attention.py, replace:
```python
trans_mat = self._rotary_setup.get_trans_mat(is_decode=True)
trans_shard_mem = ...
trans_mat = ttnn.to_memory_config(trans_mat, trans_shard_mem)
```
With:
```python
trans_mat = self._rotary_setup.get_trans_mat_decode_sharded(batch_size)
```

### 16.6 Summary

| Item | Detail |
|------|--------|
| **Root cause** | trans_mat `[1,1,32,32]` cannot be HEIGHT_SHARDED across `batch_size` cores; needs `.repeat(1,1,batch_size,1)` first |
| **Reference** | `models/tt_transformers/tt/rope.py:469-490` and `models/common/modules/attention/attention_1d.py:1905-1921` |
| **Files to change** | `rope.py` (BailingRotarySetup) and `attention.py` (_forward_decode_paged) |
| **Recommended fix** | Add `max_batch_size` param to BailingRotarySetup, repeat trans_mat at init, place directly into HEIGHT_SHARDED memory |
| **Alternative fix** | Lazy cache with `get_trans_mat_decode_sharded(batch_size)` method |

### 16.7 Verification

1. Run `pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v --timeout=0`
2. Confirm the `TT_FATAL @ rotary_embedding_llama_device_operation.cpp:153` error is gone
3. Confirm decode output is coherent text (not garbled or repetitive)

## 17. Iteration 7 - Garbled Output After RoPE Unification

### 17.1 Symptom

After unifying RoPE format (both prefill and decode now use `BailingRotarySetup` with
Meta-format cos/sin and `rotary_embedding_llama`), the test passes but produces garbled
output that starts somewhat sensibly then degenerates:

```
"I'm a big fan of the taste of the taste of the best way to make a list of 10000000000000000000000000..."
```

The prompt was "What is your favorite condiment?" - the first few tokens ("I'm a big fan")
are plausible, but the output quickly degenerates into repetition and then numeric garbage.

### 17.2 ROOT CAUSE: Missing Q/K Weight Permutation (HF-to-Meta Layout)

**The `rotary_embedding_llama` kernel expects Q/K values in Meta (interleaved) layout,
but tt-symbiote feeds them in HF (split-half) layout because the Q/K projection weights
are not permuted.**

#### The Two Data Layouts

The HuggingFace RoPE format and the Meta (llama) RoPE format represent the same mathematical
rotation but operate on **different data layouts**:

**HF Layout (split-half):**
For head_dim=128 with rotary_dim=64, the Q/K projection output has pairs at positions
`(i, i+32)` for `i = 0..31`. The `rotate_half` function splits at `dim//2`:
- Input: `[x0, x1, ..., x31, x32, x33, ..., x63]`
- rotate_half: `[-x32, -x33, ..., -x63, x0, x1, ..., x31]`
- Pairs: `(x[i], x[i+32])` are treated as complex numbers

**Meta Layout (interleaved):**
Pairs are at adjacent positions `(2i, 2i+1)` for `i = 0..31`. The trans_mat swaps adjacent
elements:
- Input: `[x0, x1, x2, x3, ..., x62, x63]`
- trans_mat: `[-x1, x0, -x3, x2, ..., -x63, x62]`
- Pairs: `(x[2i], x[2i+1])` are treated as complex numbers

#### How tt-transformers Handles This

In `models/tt_transformers/tt/load_checkpoints.py` (lines 365-374), when loading HuggingFace
checkpoints, tt-transformers **permutes the Q/K weights** to convert from HF layout to Meta
layout:

```python
elif "q_proj.weight" in key or "k_proj.weight" in key:
    n_heads = tensor.shape[0] // head_dim
    converted_weights[key] = reverse_permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
elif "q_proj.bias" in key or "k_proj.bias" in key:
    n_heads = tensor.shape[0] // head_dim
    converted_weights[key] = reverse_permute(tensor.unsqueeze(-1), n_heads, tensor.shape[0], 1).squeeze(-1)
elif "q_norm.weight" in key or "k_norm.weight" in key:
    converted_weights[key] = reverse_permute_1d(tensor)
```

The `reverse_permute` function (line 785-786):
```python
def reverse_permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
```

And `reverse_permute_1d` (lines 793-801):
```python
def reverse_permute_1d(tensor):
    """Convert the last dim from separate real and imaginary parts
    (r1, r2, i1, i2, ...) to interleaved rope format (r1, i1, r2, i2, ...)"""
    shape = tensor.shape
    dim = shape[-1]
    reals = tensor[..., : dim // 2]
    imags = tensor[..., dim // 2 :]
    interleaved = torch.stack((reals, imags), dim=-1).flatten(start_dim=len(shape) - 1)
    return interleaved
```

The `reverse_permute_1d` docstring spells it out explicitly: it converts from
"separate real and imaginary parts (r1, r2, i1, i2)" to "interleaved rope format
(r1, i1, r2, i2)".

#### What tt-symbiote Is Missing

tt-symbiote uses HuggingFace model weights **without any permutation**. The Q/K projection
weights produce outputs in HF layout (split-half pairs at `(i, i+head_dim/2)`), but
`rotary_embedding_llama` applies rotation to Meta layout (adjacent pairs at `(2i, 2i+1)`).
This means:
- **Wrong dimension pairs are being rotated together**
- The rotation is mathematically incoherent
- The first few tokens may appear plausible because position 0 has all-ones cos and all-zeros sin (identity rotation)
- As positions increase, the rotation errors compound, leading to degeneration

Similarly, Q/K norm weights (if `use_qk_norm=True`) also need to be permuted because the
norm is applied per-dimension before rotation.

### 17.3 Additional Confirmation: trans_mat Size Is Correct

An earlier concern was that the prefill trans_mat was changed from `head_dim x head_dim`
(128x128) to `TILE_SIZE x TILE_SIZE` (32x32). Investigation confirms this is **correct**:

1. **tt-transformers `get_rot_transformation_mat`** (in `common.py` line 473-475) always
   forces `dhead = 32` regardless of the input parameter:
   ```python
   def get_rot_transformation_mat(dhead=32):
       # ROPE op uses a single tile
       dhead = 32
       return get_rot_transformation_mat_v2(dhead)
   ```
   Even though callers pass `head_dim` (e.g., 128), the function ignores it and always
   returns a 32x32 matrix.

2. **The C++ kernel** (`rotary_embedding_llama.cpp` line 42) loads exactly one tile of
   trans_mat: `cb_wait_front(trans_mat_cb, onetile)`. On line 72, it does
   `matmul_tiles(in_cb, trans_mat_cb, j, in1_index, j)` where `in1_index` stays 0.
   The kernel applies the **same single 32x32 tile** to each tile column across the
   width dimension (Wt tiles). For head_dim=128, Wt=4, so the 32x32 trans_mat is
   applied 4 times independently.

3. **The C++ validation** (line 152-155) for prefill mode explicitly asserts:
   ```cpp
   trans_mat.logical_shape()[-2] == TILE_HEIGHT  // 32
   trans_mat.logical_shape()[-1] == TILE_WIDTH   // 32
   ```

4. **The TODO comment** in tt-transformers `rope.py` line 492 asks:
   ```python
   # TODO: Colman, should this be TILE_SIZE or head_dim?
   ```
   The answer is: it should be TILE_SIZE (32). The function that was being called
   (`get_rot_transformation_mat`) always returned 32x32 anyway.

### 17.4 cos/sin Format Verification

The cos/sin computation in `_compute_cos_sin_cache` was verified to match the tt-transformers
reference (`gather_cos_sin` and `permute_to_meta_format`):

Both produce Meta format by:
1. Computing `freqs = outer(positions, inv_freq)` (shape `[seq, rotary_dim/2]`)
2. Doubling: `emb = cat(freqs, freqs)` (shape `[seq, rotary_dim]`)
3. Taking cos/sin of `emb`
4. Taking first half: `cos[:, :dim//2]`
5. Interleaving pairs: `stack([cos, cos], dim=-1).flatten(-2)`

The `inv_freq` computation also matches HuggingFace exactly:
- HF: `inv_freq = 1/(base^(arange(0, dim, 2) / dim))` where `dim = head_dim * partial_rotary_factor`
- Ours: `inv_freq = 1/(theta^(arange(0, rotary_dim, 2) / rotary_dim))` -- identical

### 17.5 Why the Output Starts Somewhat Sensibly

At position 0, `cos(0 * freq) = 1.0` and `sin(0 * freq) = 0.0` for all frequencies.
This means position 0 is an identity transformation regardless of data layout, so the
first token's prefill attention is approximately correct. As positions increase, the
rotation angles grow, and the mismatch between HF and Meta layouts causes increasingly
wrong rotations. This explains the pattern: somewhat sensible start, rapid degeneration.

### 17.6 Fix Plan

#### Fix 1 (CRITICAL): Permute Q/K Projection Weights to Meta Layout

When `rotary_embedding_llama` is used (i.e., always in the T3K path), the Q/K projection
weights must be permuted from HF layout to Meta layout. This must happen during weight
loading, before any forward pass.

**Where to apply:** In `TTNNBailingMoEAttention.from_torch()` or in the weight loading
pipeline, add weight permutation for Q and K projections.

**Implementation:**

Add these utility functions (following tt-transformers pattern):

```python
def _reverse_permute_weight(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Permute Q/K projection weight from HF (split-half) to Meta (interleaved) layout.

    HF layout pairs elements at (i, i+head_dim//2).
    Meta layout pairs elements at (2i, 2i+1).

    This permutation rearranges the output dimension of the weight matrix so that
    when the projection is applied, the result is in Meta interleaved layout.
    """
    dim1, dim2 = tensor.shape[0], tensor.shape[1]
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _reverse_permute_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Permute 1D tensor (bias or norm weight) from HF to Meta layout.

    Converts (r1, r2, ..., i1, i2, ...) to (r1, i1, r2, i2, ...).
    """
    dim = tensor.shape[-1]
    reals = tensor[..., : dim // 2]
    imags = tensor[..., dim // 2 :]
    return torch.stack((reals, imags), dim=-1).flatten(start_dim=len(tensor.shape) - 1)
```

**Apply in `from_torch()` or `move_weights_to_device_impl()`:**

After loading the HF model's Q/K weights, before converting to TTNN:

```python
# Get the underlying torch Q/K projection weights
q_weight = self._fallback_torch_layer.query_key_value.weight  # or q_proj.weight
# Determine the Q/K portion dimensions
n_q_heads = self.num_heads
n_kv_heads = self.num_kv_heads
head_dim = self.head_dim

# Permute Q portion of the fused QKV weight
q_dim = n_q_heads * head_dim
k_dim = n_kv_heads * head_dim
# If weights are fused as [Q | K | V]:
fused_weight = self._fallback_torch_layer.query_key_value.weight
q_weight = fused_weight[:q_dim, :]
k_weight = fused_weight[q_dim:q_dim+k_dim, :]
v_weight = fused_weight[q_dim+k_dim:, :]

q_weight_permuted = _reverse_permute_weight(q_weight, n_q_heads)
k_weight_permuted = _reverse_permute_weight(k_weight, n_kv_heads)
fused_weight_permuted = torch.cat([q_weight_permuted, k_weight_permuted, v_weight], dim=0)

# Similarly for biases if present
# Similarly for q_norm/k_norm weights if use_qk_norm=True
```

**IMPORTANT:** The Q/K norm weights (if `use_qk_norm=True`) must ALSO be permuted using
`_reverse_permute_1d`, because the norm is applied in the dimension that is being permuted.

#### Fix 2 (ALTERNATIVE): Use HF-Format RoPE Instead

If permuting weights is too invasive, an alternative is to NOT use `rotary_embedding_llama`
and instead:
- Use `ttnn.experimental.rotary_embedding` (HF format) for prefill
- Implement decode RoPE using pure TTNN ops (element-wise multiply + add) in HF format

However, this is NOT recommended because:
1. `rotary_embedding` does not support the decode tensor layout `[1, B, H, D]`
2. It would diverge from the proven tt-transformers pattern
3. `rotary_embedding_llama` is specifically optimized for this use case

#### Fix 3 (REQUIRED): Verify Q/K Norm Permutation

If `use_qk_norm=True` (which it is for Ling-mini-2.0 based on the `_apply_qk_norm` calls),
the Q/K RMS norm weights must also be permuted. RMS norm operates element-wise:
`norm(x) = x / rms(x) * weight`, so the weight vector must be in the same layout as the
data it multiplies.

Check what `_apply_qk_norm` does and ensure the norm weight is permuted:

```python
# In from_torch() or move_weights_to_device_impl():
if self.use_qk_norm:
    q_norm_weight = self._fallback_torch_layer.q_norm.weight  # shape [head_dim]
    k_norm_weight = self._fallback_torch_layer.k_norm.weight  # shape [head_dim]
    q_norm_weight_permuted = _reverse_permute_1d(q_norm_weight)
    k_norm_weight_permuted = _reverse_permute_1d(k_norm_weight)
    # ... update the norm weights
```

### 17.7 Why This Wasn't Caught Earlier

The RoPE format mismatch was diagnosed in Iteration 5 (Section 15), but the diagnosis
focused on **cos/sin format** (HF vs Meta) between prefill and decode paths. The solution
in Iteration 6 was to unify cos/sin to Meta format and use `rotary_embedding_llama` for
both paths. This was correct and necessary.

However, using `rotary_embedding_llama` with Meta-format cos/sin is only half the story.
The **other half** is that the Q/K data itself must be in Meta (interleaved) layout for
the rotation to be applied to the correct dimension pairs. This requires permuting the
Q/K projection weights, which is a weight-loading concern, not a cos/sin concern.

The tt-transformers codebase handles this in `load_checkpoints.py` using `reverse_permute`,
which is separate from the RoPE code. This separation made it easy to miss.

### 17.8 Summary

| Finding | Detail |
|---------|--------|
| **Root cause** | Q/K projection weights not permuted from HF to Meta layout |
| **Evidence** | tt-transformers `load_checkpoints.py:365-374` permutes Q/K weights with `reverse_permute()` |
| **Why it matters** | `rotary_embedding_llama` rotates adjacent pairs `(x[2i], x[2i+1])`, but HF weights produce split-half pairs `(x[i], x[i+dim/2])` |
| **Why output starts OK** | Position 0 is identity rotation (cos=1, sin=0) regardless of layout |
| **Why output degenerates** | Rotation errors compound as position increases |
| **trans_mat 32x32** | CONFIRMED correct -- kernel always uses single tile, tt-transformers also forces dhead=32 |
| **cos/sin format** | CONFIRMED correct -- matches tt-transformers Meta format exactly |
| **Fix** | Add `reverse_permute` for Q/K weights and `reverse_permute_1d` for Q/K norm weights |
| **Reference code** | `models/tt_transformers/tt/load_checkpoints.py` lines 365-374, 785-801 |

### 17.9 Implementation Order

1. **Add `_reverse_permute_weight` and `_reverse_permute_1d` utility functions** to
   `rope.py` or a new `weight_utils.py`
2. **Permute Q/K projection weights** in the weight loading path (before TTNN conversion)
3. **Permute Q/K norm weights** if `use_qk_norm=True`
4. **Permute Q/K projection biases** if present (Ling uses `bias=True` for attention)
5. **Run test** and verify coherent output
6. **Verify numerically**: Compare a single layer's Q output (after projection + norm +
   RoPE) between HF reference and TTNN, for both prefill and decode at a few positions

### 17.10 Key Files

| File | What to change |
|------|---------------|
| `models/experimental/tt_symbiote/modules/attention.py` | Add weight permutation in `from_torch()` or `move_weights_to_device_impl()` |
| `models/experimental/tt_symbiote/modules/rope.py` | (No changes needed -- cos/sin and trans_mat are correct) |
| `models/tt_transformers/tt/load_checkpoints.py` | Reference only -- `reverse_permute` and `reverse_permute_1d` implementations |

## 18. Iteration 8 - Deep Decode Diagnosis

### 18.1 Symptom Recap

After all previous fixes (RoPE unification, weight permutation, trans_mat, _seq_lengths):
```
"It seems like to me, I'm not sure what you're asking about, but I'll try to help you
with your favorite thing. I'll be to you. I your favorite thing. I'll be to be to you're..."
```

Output starts somewhat sensibly but degenerates rapidly during decode.

### 18.2 CPU Baseline Verification

**Result: CPU baseline produces PERFECT output.**
```
"As an AI, I don't have personal preferences or taste buds, so I don't have a favorite
condiment. However, I can certainly help you explore different condiments and their uses!"
```

This confirms the HuggingFace model weights and tokenizer are correct. The bug is
exclusively in the TTNN/paged-attention integration.

### 18.3 Systematic Analysis of Decode Path

The following components were verified as CORRECT:

| Component | Status | Reasoning |
|-----------|--------|-----------|
| Weight permutation (reverse_permute) | CORRECT | Q/K weights permuted to Meta layout, V/dense NOT permuted (correct) |
| QK norm weight permutation | CORRECT | _reverse_permute_1d produces matching Meta layout for element-wise scaling |
| RoPE cos/sin computation | CORRECT | _compute_cos_sin_cache matches Meta interleaved-pair format |
| attention_scaling factor | N/A | Value is 1.0 for default rope_type (no scaling needed) |
| Prefill RoPE | CORRECT | Uses rotary_embedding_llama with Meta-format cos/sin from BailingRotarySetup |
| Decode RoPE | CORRECT | Same BailingRotarySetup, HEIGHT_SHARDED trans_mat via get_trans_mat_decode_sharded() |
| _project_qkv_t3k | CORRECT | Q col-sharded + reduce_scatter, K/V replicated input + col-sharded weight; all_gather produces replicated output |
| QKV concat + _to_replicated | CORRECT | Data is identical on all devices after all_gather; host round-trip preserves data |
| nlp_create_qkv_heads_decode split | CORRECT | Splits [Q(2048), K(512), V(512)] matching concat order |
| _seq_lengths tracking | CORRECT | Prefill adds seq_len, decode adds 1; cur_pos read before update |
| cache_position handling | CORRECT | HF BailingMoeV2 does not pass cache_position; falls back to get_seq_length() |
| KV cache format compatibility | CORRECT | paged_fill_cache [B,H,S,D] and paged_update_cache [1,B,H,D] both write to [blocks,H,block_size,D] |
| nlp_concat_heads_decode + dense | CORRECT | Output [1,1,32,H*D] -> dense -> slice to batch=1 -> reshape |
| SDPAProgramConfig for decode | CORRECT | q_chunk_size=0, k_chunk_size=0 matches tt-transformers convention |

### 18.4 ROOT CAUSE FOUND: TTNNPagedAttentionKVCache.update() Does Not Accumulate K/V History

**File:** `models/experimental/tt_symbiote/modules/attention.py`, lines 239-260

**The Bug:**

The test replaces only layer 0's attention with TTNN (`TTNNBailingMoEAttention`). Layers 1-23
remain as standard PyTorch `BailingMoeV2Attention`. However, the test passes a single
`TTNNPagedAttentionKVCache` as `past_key_values` to `model.generate()`, which feeds it to ALL
24 layers.

Layer 0 (TTNN) correctly uses `paged_fill_on_device()` and `paged_update_on_device()` to
store K/V in the on-device paged cache. It reads from this cache via `paged_sdpa_decode()`.

Layers 1-23 (PyTorch) call `past_key_value.update(key_states, value_states, layer_idx, ...)`,
which is the standard HuggingFace Cache interface. The HF contract requires `update()` to:
1. **Accumulate** the new K/V with all previously cached K/V
2. **Return** the full accumulated K/V tensors

`DynamicCache.update()` does this correctly:
```python
self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
return self.key_cache[layer_idx], self.value_cache[layer_idx]
```

But `TTNNPagedAttentionKVCache.update()` is BROKEN:
```python
def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
    # ... just converts types and increments _seq_lengths ...
    return key_states, value_states  # <-- Returns ONLY current token's K/V!
```

**Impact:**
- During **prefill**, layers 1-23 call `update()` with the full prompt K/V. Since there's no
  prior cache, the returned K/V IS the full prefill K/V. Prefill works correctly.
- During **decode**, layers 1-23 call `update()` with just the current token's K/V (shape
  `[B, H, 1, D]`). The method returns this single token, NOT the accumulated history.
  Therefore `attn_weights = Q @ K^T` at line 529 of the HF model only computes attention
  over the current token, not over the full context.
- The attention mechanism degenerates because each decode step only attends to itself for
  23 out of 24 layers.

**Why output starts somewhat sensibly:**
- The first decode token benefits from the prefill computation: all 24 layers processed the
  full prompt during prefill, producing a reasonable hidden state. The LM head then picks
  a plausible next token from this hidden state.
- From the 2nd decode token onward, layers 1-23 only attend to the current token. The hidden
  state progressively loses context, causing rapid degeneration.

### 18.5 The Fix

`TTNNPagedAttentionKVCache.update()` must accumulate K/V for PyTorch layers. Two approaches:

**Option A (Recommended): Maintain a CPU-side DynamicCache for PyTorch layers**

Add an internal `DynamicCache` for layers that use the `update()` path (PyTorch layers).
The paged on-device cache continues to be used for TTNN layers via the
`paged_fill_on_device()` / `paged_update_on_device()` path.

```python
class TTNNPagedAttentionKVCache(Cache):
    def __init__(self, ...):
        ...
        # CPU-side cache for PyTorch layers that call update()
        self._cpu_key_cache: list[Optional[torch.Tensor]] = [None] * num_layers
        self._cpu_value_cache: list[Optional[torch.Tensor]] = [None] * num_layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Convert to torch if needed
        if isinstance(key_states, TorchTTNNTensor):
            key_states = key_states.to_torch
        if isinstance(value_states, TorchTTNNTensor):
            value_states = value_states.to_torch
        if isinstance(key_states, ttnn.Tensor):
            key_states = ttnn.to_torch(key_states)
        if isinstance(value_states, ttnn.Tensor):
            value_states = ttnn.to_torch(value_states)

        seq_len = key_states.shape[2]
        self._seq_lengths[layer_idx] += seq_len
        if layer_idx == 0:
            self._seen_tokens += seq_len

        # Accumulate K/V history for PyTorch layers
        if self._cpu_key_cache[layer_idx] is None:
            self._cpu_key_cache[layer_idx] = key_states
            self._cpu_value_cache[layer_idx] = value_states
        else:
            self._cpu_key_cache[layer_idx] = torch.cat(
                [self._cpu_key_cache[layer_idx], key_states], dim=-2
            )
            self._cpu_value_cache[layer_idx] = torch.cat(
                [self._cpu_value_cache[layer_idx], value_states], dim=-2
            )

        return self._cpu_key_cache[layer_idx], self._cpu_value_cache[layer_idx]
```

**Option B (Alternative): Use a DynamicCache for non-TTNN layers + TTNNPagedAttentionKVCache for TTNN layers**

Create a hybrid cache that delegates to `DynamicCache` for PyTorch layers and to the
paged cache for TTNN layers. This requires a way to know which layers are TTNN vs PyTorch.

**Recommendation: Option A** is simpler and more robust. It keeps the API the same and
just fixes the `update()` method to match HF's `DynamicCache` contract.

### 18.6 Secondary Considerations

Once the `update()` bug is fixed, verify:

1. **Numerical accuracy**: Compare TTNN layer 0 output vs PyTorch layer 0 output for the same
   input at various decode positions. PCC > 0.99 expected.
2. **Memory**: The CPU-side cache grows linearly with sequence length for 23 layers. For
   max_seq_len=2048, each layer stores `[B, H, S, D]` = `[1, 4, 2048, 128]` = 1MB for K+V.
   Total for 23 layers: ~23MB. This is negligible.
3. **Performance**: The CPU-side cache uses `torch.cat()` which copies data every step.
   For long sequences, consider pre-allocating the cache tensors.

### 18.7 Why Previous Iterations Missed This

- Previous debugging focused on the TTNN layer 0 internals (RoPE format, weight permutation,
  trans_mat, etc.) because the symptom ("output degenerates") seemed like a RoPE/attention bug.
- The `update()` method was assumed correct because it "just passes data through" and the
  focus was on the more complex TTNN code paths.
- The bug is actually in the SIMPLEST code (3 lines of Python) rather than the complex TTNN
  kernels. This is a classic case of "looked too deep, missed the obvious."

### 18.8 Summary

| Finding | Detail |
|---------|--------|
| **Root cause** | `TTNNPagedAttentionKVCache.update()` does NOT accumulate K/V history |
| **Evidence** | HF `DynamicCache.update()` concatenates `[cached_K, new_K]` and returns full tensor; our `update()` returns only `new_K` |
| **Impact** | Layers 1-23 (PyTorch) only attend to current token during decode (no history) |
| **Why prefill works** | First call to `update()` has no prior cache, so returned K/V = full prefill K/V |
| **Why output starts OK** | First decode token uses prefill hidden states from all 24 layers |
| **Why it degenerates** | From token 2+, 23/24 layers see only 1-token context |
| **Fix** | Add CPU-side K/V accumulation in `update()` to match `DynamicCache` contract |
| **Complexity** | ~15 lines of code change |

### 18.9 Implementation Order

1. **Fix `TTNNPagedAttentionKVCache.update()`** -- add `_cpu_key_cache` / `_cpu_value_cache`
   lists and concatenate K/V in `update()`, returning accumulated tensors
2. **Run test** -- verify coherent multi-sentence output matching CPU baseline quality
3. **Numerical validation** -- compare TTNN layer 0 vs PyTorch layer 0 at decode positions
4. **Clean up** -- remove debug prints, add docstring explaining the dual-cache design

### 18.10 Key Files

| File | What to change |
|------|---------------|
| `models/experimental/tt_symbiote/modules/attention.py` | Fix `TTNNPagedAttentionKVCache.update()` to accumulate K/V (lines 239-260) |
| `models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py` | Add output quality assertion (not just non-empty) |

## 19. Iteration 9 - Corrected Decode Diagnosis (All 20 Layers Are TTNN)

### 19.1 Critical Correction

The Section 18 diagnosis was **WRONG**. It assumed some layers use PyTorch attention and
call `update()`. In reality, **all 20 attention layers** are replaced with
`TTNNBailingMoEAttention`. The `update()` method is never called. The CPU-side
`_cpu_key_cache` / `_cpu_value_cache` fix from Section 18 was correct code but irrelevant
because no code path ever reaches `update()`.

The degenerate output ("It seems like to me, I'm not sure what you're asking about...")
must be caused by something in the TTNN paged decode path itself.

### 19.2 Investigation: Position Tracking for All 20 Layers

**Finding: Position tracking is CORRECT.**

The flow for each layer:

1. **After prefill**: `_seq_lengths[0..19]` = 25 (each layer calls `paged_fill_on_device`
   which does `_seq_lengths[layer_idx] += seq_len`)
2. **First decode, layer N**: `cur_pos = get_seq_length(N) = 25`. RoPE applied at position 25.
   After `paged_update_on_device`, `_seq_lengths[N] = 26`.
3. **Second decode, layer N**: `cur_pos = get_seq_length(N) = 26`. Correct.

Each layer tracks its own `_seq_lengths[layer_idx]` independently. Since all layers
process the same token in sequence within a forward pass, all 20 layers stay perfectly
synchronized (all increment by 1 per decode step).

`_seen_tokens` is only updated for `layer_idx == 0` and is used by HF's
`get_seq_length()` (default layer_idx=0) to compute `past_seen_tokens` for position_ids.
This is also correct.

### 19.3 Investigation: cache_position Is Always None

**Finding: `cache_position` is NEVER passed to TTNNBailingMoEAttention.**

The call chain:
1. HF `generate()` -> `prepare_inputs_for_generation()` adds `cache_position` to model_inputs
2. `BailingMoeV2ForCausalLM.forward(**kwargs)` -> `self.model(**kwargs)` passes it through
3. `BailingMoeV2Model.forward(**kwargs)` captures `cache_position` in its `**kwargs` but
   **never passes it to decoder layers** (line 1292-1301 of modeling_bailing_moe_v2.py)
4. `BailingMoeV2DecoderLayer.forward()` -> `self.attention()` never receives `cache_position`
5. `TTNNBailingMoEAttention.forward(cache_position=None)` always hits the fallback:
   ```python
   cur_pos = past_key_values.get_seq_length(layer_idx)
   ```

This is actually fine because `get_seq_length(layer_idx)` returns the correct value (see 19.2).

### 19.4 Investigation: HF Model Does NOT Call update() Outside Attention

**Finding: The decoder layer does NOT call `past_key_value.update()` externally.**

```python
# BailingMoeV2DecoderLayer.forward() (line 998):
hidden_states, self_attn_weights, present_key_value = self.attention(
    hidden_states=..., past_key_value=past_key_values, ...
)
# No additional cache operations after this
```

The original HF attention (`BailingMoeV2Attention.forward()`, line 524) calls
`past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)` internally.
But since we replaced the attention module entirely, this call is replaced by our
`paged_fill_on_device` / `paged_update_on_device` calls. Correct.

### 19.5 Investigation: RoPE Format Consistency

**Finding: RoPE format is consistent between prefill and decode.**

Both prefill and decode use `BailingRotarySetup` which computes cos/sin in Meta
(interleaved) format:
- `_compute_cos_sin_cache`: `torch.stack((cos, cos), dim=-1).flatten(-2)` produces
  `[f0, f0, f1, f1, ...]` interleaved format
- Q/K weights are permuted from HF to Meta layout via `_reverse_permute_weight()` in
  `from_torch()` (line 2389-2390)
- Both paths use `rotary_embedding_llama` kernel (just different `is_decode_mode` flag)

The HF model's `position_embeddings` (computed by `BailingMoeV2RotaryEmbedding` in HF
split-half format) are **ignored** in decode mode. Instead, our code recomputes cos/sin
from `BailingRotarySetup` using the same Meta format as prefill. This is correct.

`attention_scaling` from HF is 1.0 for the default rope type, so no scaling mismatch.

### 19.6 Investigation: inv_freq Computation Match

**Finding: Frequency computation matches between HF and BailingRotarySetup.**

Both use the same formula:
```python
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
```
where `rotary_dim = head_dim * partial_rotary_factor = 128 * 0.5 = 64`.

The only difference is the output format (HF split-half vs Meta interleaved), which is
intentional and matched by the Q/K weight permutation.

### 19.7 Investigation: Dense Projection in Decode Mode

**Finding: Dense projection flow looks correct but has a topology concern.**

After `nlp_concat_heads_decode` (output: `[1, 1, 32, 2048]`), the dense layer
(`TTNNLinearIReplicatedWColSharded`) applies a col-sharded linear. The output is
col-sharded across 8 devices (each device has 256 dims of the 2048 hidden_size).

After slicing to `[1, 1, 1, 2048]` and reshaping to `[1, 1, 2048]`, this is returned
to the HF decoder layer for the residual connection: `hidden_states = residual + hidden_states`.

**Question**: Is the tensor topology of the dense output compatible with the residual?
The residual comes from the input_layernorm path. If the residual is replicated but the
attention output is col-sharded, the addition would produce garbage.

### 19.8 Likely Root Causes (Prioritized)

#### Hypothesis A: Topology/Sharding Mismatch in Residual Connection

The TTNN attention's decode path returns a tensor that went through:
1. `_to_replicated` (for fused QKV input)
2. `nlp_create_qkv_heads_decode` (HEIGHT_SHARDED output)
3. `rotary_embedding_llama` (HEIGHT_SHARDED)
4. `paged_update_on_device` (writes to KV cache)
5. `paged_sdpa_decode` (outputs DRAM)
6. `nlp_concat_heads_decode` (outputs in some topology)
7. `dense` projection with col-sharded weights

The output topology after step 7 is col-sharded. But the HF decoder layer adds this
to `residual` which may have a different topology (e.g., replicated from the input
layernorm). On a mesh device, if one tensor is col-sharded and the other is replicated,
`ttnn.add` may produce incorrect results silently or the element-wise add may happen
between misaligned data.

**However**, this same issue would exist in prefill too, not just decode. So if prefill
produces reasonable first tokens, this might not be the root cause.

#### Hypothesis B: Cumulative Numerical Error from bfloat16

With all 20 layers running on-device in bfloat16, cumulative numerical errors across
layers could cause drift. The `HiFi2` math fidelity in the dense projection
(line 297 of linear.py) is lower precision than `HiFi4`. Each layer compounds the error.

For a 20-layer model with bfloat16, small errors in attention outputs propagate through
the residual stream and get amplified by the MLP (which may still be running on CPU/PyTorch).

**This is a plausible cause** of degenerate output. The first few tokens would be
reasonable (attention values based on prefill context) but errors would accumulate over
decode steps.

#### Hypothesis C: SDPA Decode Seeing Stale/Zero KV Entries

The `paged_scaled_dot_product_attention_decode` kernel uses `cur_pos_tensor` to determine
how many KV entries to attend over. After prefill stores 25 entries and decode stores 1
more at position 25, the SDPA with `cur_pos=25` should attend over positions 0..25.

But what if the kernel interprets `cur_pos` differently? If it uses `cur_pos` as a
count (exclusive bound) rather than an index (inclusive bound), then on the first decode
step with `cur_pos=25`, SDPA would attend over 0..24 (missing the just-stored decode
token at position 25).

The decode token's Q would still attend to the full prefill context (0..24), so it would
produce a reasonable output. But the Q for the decode token was computed from the just-
generated embedding, so missing its own KV entry shouldn't matter (self-attention on a
single new token doesn't add much information). This is likely NOT the cause.

#### Hypothesis D: MoE Layers Running on CPU with Wrong Input Format

The test code (line 86) has `TTNNBailingMoE` commented out:
```python
# model.model.layers[1].mlp.__class__: TTNNBailingMoE,
```

So all MoE/MLP layers run in PyTorch on CPU. The attention output from TTNN (a mesh
tensor) gets passed to the PyTorch MLP. The `TorchTTNNTensor` wrapper should handle
the conversion, but if the data arrives col-sharded and the conversion produces only
1/8 of the hidden dimensions, the MLP would receive garbage.

**This is the MOST LIKELY root cause.** Let me trace:
1. TTNN attention returns `attn_output` shaped `[1, 1, 2048]` as a mesh tensor
2. The HF decoder layer does `hidden_states = residual + hidden_states`
3. `residual` is a PyTorch tensor (the MLP from the previous layer returned PyTorch)
4. Adding a mesh tensor to a PyTorch tensor goes through TorchTTNNTensor conversion
5. If the mesh tensor is col-sharded, converting to PyTorch might produce `[1, 1, 256]`
   per device, and concatenation might not happen correctly

### 19.9 Recommended Next Steps

1. **Instrument the decode path**: After `_forward_decode_paged` returns, print the shape
   and values of `attn_output` after converting to PyTorch. Check if it's the full 2048
   dims or only 256.

2. **Check topology after dense**: Add a debug print to verify the topology of the dense
   output. Is it replicated or col-sharded?

3. **Test with explicit all-gather**: After `self.dense(attn_output)`, add an all-gather
   and `_to_replicated()` call to ensure the output has replicated topology before
   returning to PyTorch land.

4. **Compare single-layer output**: Run a single TTNN attention layer (layer 0 only) with
   all others in PyTorch. If the output is coherent, the issue is cumulative. If it's
   still degenerate, the issue is in the single-layer decode path.

5. **Check bfloat16 precision**: Compare TTNN decode output vs PyTorch decode output for
   a single layer at multiple decode positions. Measure PCC (Pearson Correlation Coefficient).

### 19.10 Key Insight About Mixed TTNN/PyTorch Execution

The attention layers run on TTNN (T3K mesh, 8 devices) while MLP/MoE layers run on
CPU (PyTorch). Every layer boundary involves:
- TTNN -> PyTorch conversion (after attention, before residual add + MLP)
- PyTorch -> TTNN conversion (after MLP, before next layer's attention)

These conversions must handle mesh topology correctly. If the TTNN attention output
is col-sharded across 8 devices, the PyTorch conversion must all-gather to produce
the full tensor. The `TorchTTNNTensor` wrapper's `to_torch` property handles this,
but it may use `ConcatMeshToTensor` which concatenates device shards. If the shards
are col-sharded (each device has 256 dims), concatenation on dim 0 would be wrong;
it needs to concatenate on dim -1. Concatenation on the wrong dim would produce a
tensor with correct total elements but scrambled data.

### 19.11 Summary

| Investigation Area | Finding |
|---|---|
| Position tracking (`_seq_lengths`) | CORRECT - each layer tracks independently, all stay in sync |
| `cache_position` parameter | Always None (not passed by HF model), but fallback to `get_seq_length` is correct |
| `update()` called outside attention? | NO - decoder layer only calls `self.attention(...)` |
| RoPE format (prefill vs decode) | CONSISTENT - both use Meta format via BailingRotarySetup |
| inv_freq computation | MATCHES between HF and BailingRotarySetup |
| Dense projection topology | CONCERN - output may be col-sharded, incompatible with PyTorch residual |
| **Most likely root cause** | **Mesh tensor topology mismatch at TTNN->PyTorch boundary** - col-sharded attention output gets incorrectly converted when adding to PyTorch residual tensor |

### 19.12 Key Files

| File | Relevance |
|------|-----------|
| `modules/attention.py` lines 2684-2893 | `_forward_decode_paged` - the decode path under investigation |
| `modules/attention.py` lines 2895-2960 | `forward()` - routing between prefill and decode |
| `modules/attention.py` lines 80-284 | `TTNNPagedAttentionKVCache` - position tracking (`_seq_lengths`) |
| `modules/linear.py` lines 304-329 | `TTNNLinearIReplicatedWColSharded` - dense projection (col-sharded output) |
| `modules/rope.py` lines 421-709 | `BailingRotarySetup` - RoPE for decode (Meta format) |
| `modeling_bailing_moe_v2.py` lines 1187-1301 | HF model forward - does NOT pass `cache_position` to layers |
| `modeling_bailing_moe_v2.py` lines 960-1030 | HF decoder layer - does NOT call `update()` outside attention |

## 20. Iteration 10 - Weight Permutation Verification

### 20.1 The Core Question

The model output improved slightly after adding weight permutation (Iteration 5) but remains
degenerate. The question is: is the weight permutation correct, or is it making things worse?

Two hypotheses:
- **H1:** Weight permutation is correct, but another bug (e.g., topology mismatch at TTNN/PyTorch
  boundary) is causing degenerate output
- **H2:** Weight permutation is WRONG, either because it is the wrong transformation or because
  the cos/sin cache format does not match the permuted weight layout

### 20.2 HuggingFace Bailing RoPE Analysis

The HF model code (`modeling_bailing_moe_v2.py`) uses the standard Llama-style RoPE:

**HF Rotary Embedding** (lines 216-227):
```python
freqs = inv_freq_expanded @ position_ids_expanded  # shape: [B, rotary_dim//2, seq]
freqs = freqs.transpose(1, 2)                       # shape: [B, seq, rotary_dim//2]
emb = torch.cat((freqs, freqs), dim=-1)             # shape: [B, seq, rotary_dim]
cos = emb.cos()   # cos[..., i] == cos[..., i + rotary_dim//2] for i < rotary_dim//2
sin = emb.sin()
```

**HF apply_rotary_pos_emb** (lines 239-272):
```python
rotary_dim = cos.shape[-1]   # = 64 for Ling-mini-2.0 (head_dim=128, partial_rotary=0.5)
q_rot = q[..., :rotary_dim]  # first 64 dims
q_pass = q[..., rotary_dim:] # last 64 dims (passed through unchanged)
q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
# where rotate_half splits at dim//2 = 32: (-x2, x1)
```

So HF RoPE pairs element `i` with element `i + rotary_dim//2`:
- `q_new[i] = q[i] * cos[i] - q[i + 32] * sin[i]`  (for i in [0, 32))
- `q_new[i] = q[i] * cos[i] + q[i - 32] * sin[i]`  (for i in [32, 64))

The HF weight layout is "split-half": frequencies are organized so that the first half
of rotary dims and the second half form paired groups.

### 20.3 Meta (rotary_embedding_llama) RoPE Analysis

The TTNN `rotary_embedding_llama` kernel operates on adjacent pairs:
- `q_new[2i] = q[2i] * cos[2i] - q[2i+1] * sin[2i]`
- `q_new[2i+1] = q[2i+1] * cos[2i+1] + q[2i] * sin[2i+1]`

This is the "interleaved" layout where elements (2i, 2i+1) are a rotation pair.

### 20.4 Weight Permutation Correctness Analysis

**`_reverse_permute_weight`** (attention.py line 2220):
```python
tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
```

For Ling-mini-2.0 Q weights (dim1=2048, n_heads=16, head_dim=128):
- View as `[16, 2, 64, 2048]` - each head's output has 2 halves of 64 dims
- Transpose dims 1,2 to get `[16, 64, 2, 2048]` - interleave the halves
- Reshape to `[2048, 2048]`

This transforms each head's output from `[r0, r1, ..., r63, i0, i1, ..., i63]`
to `[r0, i0, r1, i1, ..., r63, i63]` - which is exactly the split-half to
interleaved conversion needed.

**The weight permutation itself is CORRECT** - it properly converts from HF split-half
layout to Meta interleaved layout.

### 20.5 Cos/Sin Cache Format Analysis

**`_compute_cos_sin_cache`** (rope.py lines 362-418) for partial_rotary_factor=0.5:
```python
rotary_dim = 64     # int(128 * 0.5)
inv_freq = ...      # shape: [32]  (rotary_dim // 2 = 32 frequencies)
freqs = outer(t, inv_freq)  # shape: [max_seq_len, 32]
emb = cat((freqs, freqs), dim=-1)  # shape: [max_seq_len, 64]
cos = emb.cos()  # shape: [max_seq_len, 64]
sin = emb.sin()

# Then permute to Meta format:
cos = cos[:, :32]                               # take first half: [seq, 32]
cos = stack((cos, cos), dim=-1).flatten(-2)     # duplicate each: [seq, 64]
# Result: cos[pos, 2i] == cos[pos, 2i+1] == cos_freq_i(pos)
```

After this permutation, `cos[2i] == cos[2i+1]` for each frequency index i.
This is correct for `rotary_embedding_llama` which needs identical cos/sin
values at adjacent pair positions (2i, 2i+1).

**Identity padding** (lines 411-416): Beyond the 64 rotary dims, cos is padded with 1.0
and sin with 0.0, so `q_new[j] = q[j]*1 + q[j']*0 = q[j]` for j >= 64. This correctly
passes non-rotary dims through unchanged.

### 20.6 CRITICAL FINDING: The Cos/Sin Cache Has a Bug

Wait - let me re-examine more carefully.

**HF format:** `emb = cat(freqs, freqs)` where `freqs[i] = position * inv_freq[i]`.
So `cos_hf[i] = cos(position * inv_freq[i])` for i in [0, 32) and
`cos_hf[i] = cos(position * inv_freq[i-32])` for i in [32, 64).

The HF `rotate_half` pairs element `i` with element `i+32`. Both elements in the pair
see the SAME cos value: `cos_hf[i] == cos_hf[i+32]` since both are `cos(pos * inv_freq[i])`.

**Meta format in `_compute_cos_sin_cache`:**
```python
emb = cat((freqs, freqs), dim=-1)    # [seq, 64], same as HF
cos = emb.cos()                       # [seq, 64]
cos = cos[:, :32]                     # first 32 cols: cos(pos * inv_freq[0..31])
cos = stack((cos, cos), dim=-1).flatten(-2)  # [seq, 64]: duplicated pairs
```

Result: `cos_meta[2i] = cos_meta[2i+1] = cos(pos * inv_freq[i])` for i in [0, 32).

After weight permutation, what was HF element `i` (for i in [0, 32)) is now at Meta
position `2i`, and what was HF element `i+32` is now at Meta position `2i+1`.

Check:
- HF: `q_new[i] = q[i] * cos(pos*inv_freq[i]) - q[i+32] * sin(pos*inv_freq[i])`
- Meta: `q_new[2i] = q[2i] * cos_meta[2i] - q[2i+1] * sin_meta[2i]`
  = `q_orig[i] * cos(pos*inv_freq[i]) - q_orig[i+32] * sin(pos*inv_freq[i])`

These are identical. The math checks out.

**CONCLUSION: The weight permutation AND cos/sin cache are both correct.**

### 20.7 So Why is Output Still Degenerate?

Since the weight permutation and RoPE format are mathematically correct, the degenerate
output must come from a different source. The most likely candidates from Iteration 9 are:

1. **Mesh tensor topology mismatch at TTNN/PyTorch boundary** (Iteration 9, Section 19.10):
   The `dense` (output) projection uses `TTNNLinearIReplicatedWColSharded` which produces
   col-sharded output across 8 devices. When this output passes back to PyTorch (for the
   residual add with the non-replaced decoder layer), the `TorchTTNNTensor.to_torch`
   conversion may incorrectly concatenate device shards on the wrong dimension.

2. **QK Norm weight permutation**: The QK norm weights are also permuted with
   `_reverse_permute_1d`. If QK norm expects split-half data layout but receives
   interleaved data (post-permutation), the normalization would be applied to the wrong
   paired dimensions. However, RMSNorm is element-wise (sqrt of mean of squares),
   so permuting the weights should not matter for RMSNorm. The normalization weight
   is just a per-element scale. **This is fine.**

3. **`_to_replicated` round-trip for decode fused QKV tensor**: In the decode path
   (line 2726), `xqkv_fused = self._to_replicated(xqkv_fused)` does a host round-trip
   to fix mesh topology. If `to_replicated_topology` uses `ConcatMeshToTensor` to gather
   and then re-distributes, the concatenation may scramble data from col-sharded tensors.

### 20.8 Recommended Debug Test: Single-Layer Attention Comparison

To definitively isolate the problem, we need a debug test that compares TTNN attention
output against PyTorch CPU attention output for a single layer. Here is the approach:

**Test Design:**
```
For layer 0, at decode step 1 (after prefilling with 2 tokens):
1. Capture the hidden_states input to the attention layer
2. Run it through BOTH:
   a. TTNNBailingMoEAttention (the replaced module)
   b. BailingMoeV2SdpaAttention (the original PyTorch module)
3. Compare outputs using PCC
```

**Implementation approach - hook-based comparison:**

Add a forward hook on the first decoder layer that:
1. Before the TTNN attention call, saves `hidden_states` as a CPU tensor
2. After the TTNN attention call, saves the output as a CPU tensor
3. Also runs the ORIGINAL PyTorch attention with the SAME inputs
4. Computes PCC between TTNN output and PyTorch output

**The hook should be added in the test file** (`test_ling_mini_2_0.py`), not in the
module itself. This keeps the module clean and makes the test self-contained.

```python
# Pseudocode for the debug hook:
class AttentionComparisonHook:
    def __init__(self, original_torch_attn, layer_idx):
        self.original_attn = original_torch_attn
        self.layer_idx = layer_idx
        self.results = []

    def __call__(self, module, args, kwargs, output):
        # Only check on decode steps (seq_len == 1)
        hidden_states = args[0]
        if isinstance(hidden_states, TorchTTNNTensor):
            hs_torch = hidden_states.to_torch.float()
        elif isinstance(hidden_states, ttnn.Tensor):
            hs_torch = ttnn.to_torch(hidden_states).float()
        else:
            hs_torch = hidden_states.float()

        if hs_torch.shape[1] != 1:  # not decode
            return output

        # Get position_embeddings from kwargs
        position_embeddings = kwargs.get('position_embeddings', args[1] if len(args) > 1 else None)

        # Run original PyTorch attention with same inputs
        with torch.no_grad():
            ref_output, _, _ = self.original_attn(
                hidden_states=hs_torch,
                position_embeddings=position_embeddings,  # need CPU cos/sin
                past_key_value=...,  # need PyTorch DynamicCache
            )

        # Compare
        ttnn_out = output[0]
        if isinstance(ttnn_out, TorchTTNNTensor):
            ttnn_out = ttnn_out.to_torch.float()
        elif isinstance(ttnn_out, ttnn.Tensor):
            ttnn_out = ttnn.to_torch(ttnn_out).float()

        pcc = torch.corrcoef(torch.stack([
            ttnn_out.flatten(), ref_output.flatten()
        ]))[0, 1].item()
        print(f"[LAYER {self.layer_idx}] TTNN vs PyTorch attention PCC: {pcc}")
        self.results.append(pcc)
        return output
```

**Challenge:** The PyTorch reference attention needs its own KV cache (DynamicCache), which
must be populated with the same prefill data. This makes the comparison complex.

### 20.9 Simpler Alternative: Compare Pre-RoPE Q/K Values

A simpler diagnostic that does not require managing two KV caches:

1. Before RoPE is applied in `_forward_decode_paged`, capture Q and K values
2. In the HF attention, capture Q and K values at the same point
3. If the weight permutation is correct AND the QKV projection is correct,
   the pre-RoPE Q/K values should match (up to the permutation mapping)

Specifically:
- TTNN Q at position 2i should equal HF Q at position i
- TTNN Q at position 2i+1 should equal HF Q at position i+32
(within each head)

**This is the cleanest test** because it isolates the weight projection from the RoPE
and from the KV cache, narrowing down where the mismatch occurs.

### 20.10 Simplest Possible Test: Offline Weight Permutation Verification

The simplest test requires no hardware at all:

```python
import torch

# Simulate one head with head_dim=8 (4 rotary dims for partial_rotary_factor=0.5)
head_dim = 8
rotary_dim = 4
hidden_dim = 16

# Create a random weight matrix for one head: [head_dim, hidden_dim]
W_hf = torch.randn(head_dim, hidden_dim)

# Create a random input
x = torch.randn(1, 1, hidden_dim)

# --- HF path ---
q_hf = x @ W_hf.T  # [1, 1, head_dim=8]
q_rot_hf = q_hf[..., :rotary_dim]  # first 4 dims
q_pass_hf = q_hf[..., rotary_dim:]  # last 4 dims

# HF RoPE (split-half)
pos = 5
inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
freqs = pos * inv_freq  # [rotary_dim//2 = 2]
cos_hf = torch.cat([freqs.cos(), freqs.cos()])  # [4]
sin_hf = torch.cat([freqs.sin(), freqs.sin()])  # [4]
x1 = q_rot_hf[..., :rotary_dim//2]
x2 = q_rot_hf[..., rotary_dim//2:]
rot_half = torch.cat([-x2, x1], dim=-1)
q_rot_result_hf = q_rot_hf * cos_hf + rot_half * sin_hf
q_result_hf = torch.cat([q_rot_result_hf, q_pass_hf], dim=-1)

# --- Meta path ---
# Permute weight
W_meta = W_hf.view(1, 2, head_dim // 2, hidden_dim).transpose(1, 2).reshape(head_dim, hidden_dim)
# BUT WAIT - this permutation is for the FULL head_dim, not just the rotary part.
# For partial rotary, the permutation still operates on the full weight matrix.
```

**CRITICAL INSIGHT:** The `_reverse_permute_weight` permutes the ENTIRE weight matrix
(all head_dim=128 dimensions per head), not just the rotary dimensions (64).

This means:
- HF element [0..63] are the rotary dims, [64..127] are pass-through dims
- After permutation, the interleaving mixes rotary and pass-through dims!

Specifically, for head_dim=128 viewed as `[2, 64]`:
- First half (indices 0..63) = rotary dimensions
- Second half (indices 64..127) = pass-through dimensions

After `transpose(1, 2).reshape`:
- Position 0 = original position 0 (rotary)
- Position 1 = original position 64 (pass-through!)
- Position 2 = original position 1 (rotary)
- Position 3 = original position 65 (pass-through!)
- ...

**THIS IS THE BUG.** The weight permutation interleaves rotary and non-rotary dimensions.
The `rotary_embedding_llama` kernel then rotates adjacent pairs (0,1), (2,3), etc.
So it rotates pairs like (rotary_dim_0, pass_through_dim_0) instead of
(rotary_dim_0, rotary_dim_1). This is completely wrong for partial rotary.

### 20.11 Root Cause: Weight Permutation Is Wrong for Partial Rotary

When `partial_rotary_factor = 0.5` (as in Ling-mini-2.0):
- `head_dim = 128`
- `rotary_dim = 64` (only first 64 dims are rotated)
- The remaining 64 dims should pass through unchanged

The `_reverse_permute_weight` function views each head as `[2, head_dim//2]` and transposes.
This treats the FULL head dimension as split-half pairs, but for partial rotary models,
only the FIRST `rotary_dim` dimensions form rotation pairs. The last `head_dim - rotary_dim`
dimensions are not part of any rotation pair and should NOT be permuted.

**Correct approach for partial rotary:**
```python
def _reverse_permute_weight_partial(tensor, n_heads, rotary_dim):
    """Permute only the rotary portion of Q/K weights from HF to Meta layout."""
    dim1, dim2 = tensor.shape
    head_dim = dim1 // n_heads
    pass_dim = head_dim - rotary_dim

    # Reshape to per-head view
    t = tensor.view(n_heads, head_dim, dim2)

    # Split into rotary and pass-through portions
    t_rot = t[:, :rotary_dim, :]    # [n_heads, rotary_dim, hidden]
    t_pass = t[:, rotary_dim:, :]   # [n_heads, pass_dim, hidden]

    # Only permute the rotary portion: split-half -> interleaved
    t_rot = t_rot.view(n_heads, 2, rotary_dim // 2, dim2).transpose(1, 2).reshape(n_heads, rotary_dim, dim2)

    # Concatenate back: rotary (interleaved) + pass-through (unchanged)
    result = torch.cat([t_rot, t_pass], dim=1)
    return result.reshape(dim1, dim2)
```

The identity-padding in `_compute_cos_sin_cache` (cos=1, sin=0 for dims >= rotary_dim)
is designed to handle the pass-through dimensions, BUT only when those dimensions are
contiguous at the END of each head. The current weight permutation scatters them into
odd positions throughout the head, breaking this assumption.

### 20.12 Impact Analysis

With the WRONG permutation (current code):
- Elements that should be pass-through (dims 64-127) are scattered into odd positions
  throughout the head
- The identity-padded cos/sin values (cos=1, sin=0) are at positions 64-127, but the
  pass-through elements are now at positions 1, 3, 5, ... (the odd positions)
- Result: pass-through elements get rotated (wrong), and some rotary elements get
  identity-treated (also wrong)
- The attention computation is corrupted at every layer, every position

With the CORRECT permutation (proposed fix):
- Rotary dims 0-63 are interleaved to form adjacent pairs
- Pass-through dims 64-127 remain contiguous at the end
- Identity padding at positions 64-127 correctly leaves pass-through dims alone
- The `rotary_embedding_llama` kernel correctly rotates pairs in dims 0-63

### 20.13 Verification: Does This Explain the Observed Behavior?

**Before weight permutation (Iteration 4):**
- HF weights with Meta-format RoPE: complete mismatch between weight layout and RoPE format
- Output: "I'm a big fan of the taste of the taste..." (degenerate repetition)

**After WRONG weight permutation (Iteration 5-9):**
- Interleaved rotary+pass-through dims with Meta-format RoPE
- The rotary dims that happen to land on even positions (0, 2, 4, ...) are correct
- That is 50% of the rotary dims correct, 50% wrong, plus 50% of pass-through dims wrong
- Output: "It seems like to me, I'm not sure what you're asking about..." (less degenerate
  but still incoherent, which matches partial correctness)

**With CORRECT weight permutation (proposed):**
- Rotary dims correctly interleaved in first 64 positions
- Pass-through dims correctly contiguous in last 64 positions
- Should produce coherent output (if no other bugs remain)

### 20.14 Additional Fix Needed: QK Norm and Bias Permutation

The `_reverse_permute_1d` function (line 2231) has the same issue for partial rotary.
It permutes the ENTIRE vector, interleaving what should be rotary and pass-through
dimensions.

For QK norm weights and bias vectors, the fix is the same principle: only permute
the rotary portion, leave the pass-through portion contiguous.

```python
def _reverse_permute_1d_partial(tensor, rotary_dim):
    """Permute only the rotary portion of a 1D tensor from HF to Meta layout."""
    dim = tensor.shape[-1]
    pass_dim = dim - rotary_dim

    rot = tensor[..., :rotary_dim]
    pas = tensor[..., rotary_dim:]

    # Interleave rotary portion
    reals = rot[..., :rotary_dim // 2]
    imags = rot[..., rotary_dim // 2:]
    rot_interleaved = torch.stack((reals, imags), dim=-1).flatten(start_dim=len(tensor.shape) - 1)

    return torch.cat([rot_interleaved, pas], dim=-1)
```

### 20.15 The Fix

In `attention.py`, the `from_torch` method of `TTNNBailingMoEAttention` (line 2342):

1. Replace `_reverse_permute_weight(q_weight, new_attn.num_heads)` with a
   partial-rotary-aware version that only permutes the first `rotary_dim` dimensions
   per head.

2. Replace `_reverse_permute_weight(k_weight, new_attn.num_kv_heads)` similarly.

3. Replace `_reverse_permute_1d(qkv_bias[:q_size])` with a partial-rotary-aware version.

4. Replace `_reverse_permute_1d(torch_attn.query_layernorm.weight)` and
   `_reverse_permute_1d(torch_attn.key_layernorm.weight)` similarly.

The `rotary_dim` value is `int(head_dim * partial_rotary_factor)` and is already computed
in `from_torch` as `new_attn.partial_rotary_factor * new_attn.head_dim`.

### 20.16 Numerical Verification (CONFIRMED)

Ran a numerical test comparing Q@K^T attention scores between HF and Meta paths for
partial rotary (rotary_dim=4, head_dim=8):

```
WRONG permutation (current code):
  HF attn[0,0] = 18.79, Meta attn[0,0] = 18.79   (diagonal matches by chance)
  HF attn[0,1] = -29.52, Meta attn[0,1] = -26.79  (OFF by 2.7)
  HF attn[1,0] = 5.17, Meta attn[1,0] = 29.76     (OFF by 24.6!)
  Max absolute difference: 24.59

CORRECT permutation (proposed fix):
  Max absolute difference: 0.000000
  All attention scores match EXACTLY.
```

The current code produces attention scores that are COMPLETELY WRONG for every
off-diagonal entry. This corrupts the attention weights (softmax probabilities),
which cascades through all 20 layers to produce garbled output.

**Full rotary case** (tested separately): Both current and proposed permutations
produce identical results, confirming the bug is specific to partial_rotary_factor < 1.0.

### 20.17 Confidence Assessment

**CONFIRMED BUG via numerical verification.**

The wrong permutation interleaves rotary dims with pass-through dims. The identity
padding (cos=1, sin=0) at positions 64-127 does NOT compensate because the pass-through
elements are scattered at odd positions 1, 3, 5, ... throughout the head.

**Remaining risk:**
- Even after this fix, the topology mismatch at the TTNN/PyTorch boundary (Iteration 9,
  Section 19.10) may still cause issues
- The fix should be tested to confirm whether it produces coherent output or if additional
  bugs remain

### 20.17 Key Files

| File | Lines | What to Change |
|------|-------|----------------|
| `modules/attention.py` | 2220-2228 | `_reverse_permute_weight` - needs partial rotary variant |
| `modules/attention.py` | 2231-2241 | `_reverse_permute_1d` - needs partial rotary variant |
| `modules/attention.py` | 2386-2390 | `from_torch` Q/K weight permutation calls |
| `modules/attention.py` | 2396-2398 | `from_torch` Q/K bias permutation calls |
| `modules/attention.py` | 2436-2442 | `from_torch` QK norm weight permutation calls |
| `modules/rope.py` | 362-418 | `_compute_cos_sin_cache` - identity padding assumes contiguous pass-through dims (correct after fix) |
