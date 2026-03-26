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
