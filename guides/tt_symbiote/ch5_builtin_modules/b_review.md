# Agent B Review — Chapter 5: Built-In TTNN Modules — Pass 1

## Issues

### Issue 1 — `TTNNSelfAttention` described as using separate Q/K/V projections (Severity: High)

**Files:** `attention_and_conv.md` line 16; `index.md` line 85

**What guide said:**
- `attention_and_conv.md`: "Self-attention with separate Q, K, V projection layers"
- `index.md`: "Separate Q/K/V projection + SDPA self-attention"

**What source says:**
`TTNNSelfAttention.from_torch` (source `attention.py` line 557) builds `self.query_key_value = TTNNFusedQKVSelfAttention.from_torch(PytorchFusedQKVSelfAttention(...))`. There are no separate Q, K, V linear layers — a single fused QKV projection is used internally.

**Fix:** Both descriptions updated to state that `TTNNSelfAttention` uses an internal fused QKV projection via `TTNNFusedQKVSelfAttention`.

---

### Issue 2 — `TTNNDistributedRMSNorm.from_torch` parameter type implied as `DeepseekV2RMSNorm` (Severity: Medium)

**File:** `normalization_and_activation.md` line 126

**What guide said:**
"Same guard pattern as `TTNNRMSNorm`" — by cross-reference this implies the accepted type is `DeepseekV2RMSNorm` (the type shown for `TTNNRMSNorm.from_torch`).

**What source says:**
`TTNNDistributedRMSNorm.from_torch` (source `normalization.py` line 107) has type annotation `rms_norm: "RMSNorm"`, which is a different forward-reference type, not `DeepseekV2RMSNorm`.

**Fix:** Added a clarifying note to `normalization_and_activation.md` that the actual parameter type annotation in source is `"RMSNorm"`, not `DeepseekV2RMSNorm`.

---

## Change Log — Pass 1 fixes applied

1. **`index.md` line 85** — Changed `TTNNSelfAttention` one-line description from "Separate Q/K/V projection + SDPA self-attention" to "Fused QKV projection (via `TTNNFusedQKVSelfAttention`) + SDPA self-attention".

2. **`attention_and_conv.md` line 16** — Changed `TTNNSelfAttention` one-line description from "Self-attention with separate Q, K, V projection layers" to "Self-attention using an internal fused QKV projection (`TTNNFusedQKVSelfAttention`) + SDPA".

3. **`normalization_and_activation.md` line 126** — Appended note to `TTNNDistributedRMSNorm.from_torch` paragraph clarifying that the source type annotation is `"RMSNorm"`, not `DeepseekV2RMSNorm`.

---

# Agent B Review — Chapter 5: Built-In TTNN Modules — Pass 2

## Pass 1 Fix Verification

Both Pass 1 fixes confirmed present in the current guide files:

1. `TTNNSelfAttention` fused QKV fix — `index.md` line 85 and `attention_and_conv.md` line 15 now correctly describe the internal fused QKV projection via `TTNNFusedQKVSelfAttention`. Verified against `attention.py` lines 557–565.

2. `TTNNDistributedRMSNorm` type annotation fix — `normalization_and_activation.md` line 126 now explicitly states the source type annotation is `"RMSNorm"`, not `DeepseekV2RMSNorm`. Verified against `normalization.py` line 107.

## New issues found

### Issue 1 — `TTNNLinearInputShardedWeightSharded` dimension meanings are inverted (Severity: High)

**File:** `linear_layers.md`, constraint table for `TTNNLinearInputShardedWeightSharded`

**What guide said:**

| Parameter | Constraint | Meaning |
|---|---|---|
| `input_dim` | Must be `-1` | Shard input tensor on second-to-last dimension |
| `weight_dim` | Must be `-2` | Shard weight tensor on last dimension |

**What source says:**

`linear.py` lines 98–101 assert `input_dim == -1` and `weight_dim == -2`. In Python/PyTorch negative indexing, `-1` is the **last** dimension and `-2` is the **second-to-last** dimension. The guide had the dimension labels swapped: it described `-1` as "second-to-last" and `-2` as "last", which is the reverse of the correct meaning.

**Fix:** Corrected both meanings in the constraint table.

## Verdict

Chapter 5 passes Pass 2 review after one fix. All Pass 1 fixes are confirmed in place. No further factual errors found across `index.md`, `linear_layers.md`, `normalization_and_activation.md`, or `attention_and_conv.md`.

## Change Log — Pass 2 fixes applied

1. **`linear_layers.md`, `TTNNLinearInputShardedWeightSharded` constraint table** — Corrected `input_dim=-1` meaning from "second-to-last dimension" to "last dimension"; corrected `weight_dim=-2` meaning from "last dimension" to "second-to-last dimension".

---

# Agent B Review — Chapter 5: Built-In TTNN Modules — Post-Compression Review

## Review scope

Checked each of the 8 locations where compression removed trailing prose sentences (Passes 1–3). For each location, verified: (a) the retained code block matches the source file exactly, (b) any retained prose sentences are self-contained and factually accurate, and (c) no dangling cross-references or incomplete sentences were left behind.

Sources checked:
- `linear.py` (all linear class implementations)
- `normalization.py` (all normalization class implementations)
- Guide files: `linear_layers.md`, `normalization_and_activation.md`, `attention_and_conv.md`, `index.md`

## Issues found

None. All compression edits verified. Chapter 5 is factually accurate.

Detailed per-location findings:

1. **`linear_layers.md` — TTNNLinear `preprocess_weights_impl`**: Code block matches `linear.py` lines 58–63 exactly. No trailing prose remains. Clean.

2. **`linear_layers.md` — TTNNLinearLLama `forward`**: Code block matches `linear.py` lines 190–193 exactly. Preceding prose (lines 76–79) correctly states the two differences from the base class (`bfloat8_b` dtype and `@deallocate_weights_after`). Clean.

3. **`linear_layers.md` — TTNNLinearInputShardedWeightSharded `move_weights_to_device_impl`**: Code block matches `linear.py` lines 107–123 exactly. Immediately follows the preceding prose which correctly describes the deferred-storage pattern. Clean.

4. **`linear_layers.md` — TTNNLinearActivation `forward`**: Code block matches `linear.py` lines 321–324 exactly. No following prose in the section. Clean.

5. **`linear_layers.md` — TTNNLinearLLamaIColShardedWRowSharded section**: Section ends with two prose sentences. Both are self-contained and accurate: the class overrides `move_weights_to_device_impl` to use `bfloat8_b`, and all other logic matches the parent. Verified against `linear.py` lines 197–222. Clean.

6. **`normalization_and_activation.md` — TTNNLayerNorm `preprocess_weights_impl`**: Code block matches `normalization.py` lines 27–32 exactly. No trailing prose remains in the subsection. Clean.

7. **`normalization_and_activation.md` — TTNNRMSNorm `preprocess_weights_impl`**: Code block matches `normalization.py` lines 80–86 exactly. The retained second sentence ("No bias is stored; `DeepseekV2RMSNorm` has no bias parameter.") is accurate — `DeepseekV2RMSNorm` has only `self.weight`, and `preprocess_weights_impl` sets only `self.tt_weight`. Clean.

8. **`normalization_and_activation.md` — TTNNDistributedRMSNorm `move_weights_to_device_impl`**: Code block matches `normalization.py` lines 116–124 exactly (minor formatting difference in chaining is semantically equivalent). Preceding prose ("Does not use `preprocess_weights_impl`. Directly reshapes and places the weight on the device mesh:") is accurate and self-contained. Clean.

## Verdict

Approved

## Change Log

None.
