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
