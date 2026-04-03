# Agent B Review: Chapter 4

## Pass 1

### Issue 1 (Factual Error) --- inv_freq denominator uses rotary dim, not full head_dim

**Files:** `global_proportional_rope.md` (Step 2), `rope_precomputation.md` (Global Table Computation)

The chapter claims the inverse frequency formula for global p-RoPE divides by the full `head_dim=512`:

> `inv_freq_rotated[i] = 1 / (1,000,000^{2i / 512})`
>
> "Note that the denominator uses the **full** `head_dim=512`, not the rotary dimension count."

This contradicts every existing HuggingFace implementation of `partial_rotary_factor`. In `modeling_rope_utils.py` (all scaling types: linear, dynamic, yarn, longrope, llama3) and in model-specific defaults (GLM4, Bamba, etc.), the pattern is:

```python
dim = int(head_dim * partial_rotary_factor)  # = int(512 * 0.25) = 128
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
```

The denominator is `dim=128`, not `head_dim=512`. This changes the frequency distribution substantially: dividing by 128 produces much higher frequencies than dividing by 512 for the same index `i`. The correct formula should be:

```
inv_freq_rotated[i] = 1 / (1,000,000^{2i / 128})   for i = 0, 1, ..., 63
```

This error propagates into `rope_precomputation.md` where the pseudocode uses `/ 512.0` as the denominator in the `inv_freq_rotated` computation.

### Issue 2 (Factual Error) --- Zero-padding of inv_freq is not the HuggingFace pattern

**Files:** `global_proportional_rope.md` (Steps 1--3, "Why Zero-Padding Instead of Narrowing"), `index.md` (table and note), `rope_precomputation.md`

The chapter describes a `_compute_proportional_rope_parameters` function that zero-pads `inv_freq` from 64 elements to 256 elements, producing full-width cos/sin tables of shape `[max_seq_len, 512]`. It presents this as the existing HuggingFace implementation.

No such function exists in the HuggingFace transformers library. The actual pattern for models with `partial_rotary_factor < 1.0` (verified in GLM4, Bamba, and the generic rope utilities) is:

1. `inv_freq` has length `dim/2 = 64` with no zero-padding.
2. The `forward()` method produces cos/sin of width `dim = 128` (after the repeat-twice concatenation), not 512.
3. The `apply_rotary_pos_emb` function handles partial rotation via split-apply-concat internally (see GLM4's implementation at `models/glm4/modeling_glm4.py:186-196`).

The chapter's "Strategy A" (full-width tables with identity values) is a valid TTNN implementation approach, but it should not be attributed to HuggingFace as existing behavior.

### Issue 3 (Factual Error) --- Cos/sin table shapes and inv_freq length in the summary table

**File:** `index.md` (Quick Reference table)

The table states for Global p-RoPE:
- `inv_freq` length: 256
- Cos/sin table shape: `[max_seq_len, 512]`

Based on the standard HuggingFace pattern (see Issue 2), the correct values are:
- `inv_freq` length: 64
- Cos/sin table shape: `[max_seq_len, 128]`

The table's values correspond to the chapter's hypothetical zero-padded representation, not the actual HuggingFace output.

### Issue 4 (Factual Error) --- Wavelength calculation uses wrong denominator

**File:** `global_proportional_rope.md` ("High Theta for Long Context" section)

The chapter computes the wavelength of the highest-frequency rotated pair (i=63) as:

> wavelength = 2*pi * (10^6)^{126/512} ~ 2*pi * 10^1.48 ~ 189 tokens

With the correct denominator of `dim=128` (per Issue 1), the formula becomes:

```
wavelength = 2*pi * (10^6)^{126/128} ~ 2*pi * 10^5.91 ~ 5,100,000 tokens
```

The actual highest-frequency pair (i=63) with `dim=128` has an extremely long wavelength because `2*63/128 = 0.984`, so `(10^6)^0.984` is close to `10^6`. This means ALL 64 rotated dimension pairs have very long wavelengths when theta=10^6 and the denominator is 128, which is actually the intended behavior for long-context extrapolation --- even the "fastest" rotation is slow.

### Issue 5 (Factual Error) --- The `unsqueeze_dim` value in the sliding RoPE reference code

**File:** `sliding_rope.md` (Reference Code section, line 93/98)

The reference code shows `unsqueeze_dim=2`:

```python
query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
```

In the Gemma3 source code (`models/gemma3/modeling_gemma3.py:365`), `apply_rotary_pos_emb` is called WITHOUT an explicit `unsqueeze_dim` argument, which defaults to `1`. The Gemma3 code also transposes Q/K BEFORE applying RoPE (line 357: `.view(hidden_shape).transpose(1, 2)`), not after. The chapter shows RoPE applied before transpose with `unsqueeze_dim=2`, which would be functionally equivalent but does not match the reference implementation's code path.

## Pass 2

All five Pass 1 issues have been fixed:

1. **inv_freq denominator** --- Now correctly uses `dim=128` everywhere (`global_proportional_rope.md` formula, `rope_precomputation.md` pseudocode). Verified.
2. **Zero-padding attribution** --- Strategy A (full-width tables) is now clearly labeled as a "TTNN Optimization" distinct from the HuggingFace reference. Strategy B (narrow tables with split-apply-concat) is correctly presented as the HuggingFace pattern. Verified.
3. **Table shapes in summary table** --- `index.md` Quick Reference now shows `inv_freq` length = 64 and cos/sin table shape = `[max_seq_len, 128]` for global p-RoPE. Verified.
4. **Wavelength calculation** --- Now uses `d = 128` and correctly computes the highest-frequency pair wavelength as ~5.1M tokens. Verified.
5. **unsqueeze_dim** --- Now shows `unsqueeze_dim=1` with RoPE applied after transpose into `[B, H, S, D]` layout. Verified.

### New Issue 1 (Factual Error) --- Tensor layout typo in sliding_rope.md

**File:** `sliding_rope.md` (line 112, "All 256 Dimensions Are Rotated" section)

The text states:

> the full `[B, S, H, 256]` tensor passes through `apply_rotary_pos_emb`

Earlier in the same file (lines 77, 94, 99, 105), the chapter correctly establishes that RoPE is applied **after** the transpose into `[B, H, S, D]` layout. The shape on line 112 should be `[B, H, S, 256]`, not `[B, S, H, 256]`.

### New Issue 2 (Factual Error) --- Shape comments in Strategy B forward pass use pre-transpose layout

**File:** `global_proportional_rope.md` (Strategy B forward pass, lines 232--243)

The shape comments on the split/concat pseudocode use `[B, S, H, D]` ordering:

```python
q_rot = query_states[:, :, :, :rotary_dim]      # [B, S, 32, 128]
q_pass = query_states[:, :, :, rotary_dim:]      # [B, S, 32, 384]
k_rot = key_states[:, :, :, :rotary_dim]         # [B, S, 4, 128]
k_pass = key_states[:, :, :, rotary_dim:]         # [B, S, 4, 384]
```

But the forward pass diagram immediately above (line 141) shows `query_states` as `[B, 32, 1, 512]` and `key_states` as `[B, 4, 1, 512]` --- post-transpose `[B, H, S, D]` layout. The shape comments should be `[B, 32, S, 128]`, `[B, 32, S, 384]`, `[B, 4, S, 128]`, and `[B, 4, S, 384]` respectively. The concat result comments on lines 242--243 have the same issue.

## Pass 3

Both Pass 2 layout fixes verified:

1. **`sliding_rope.md` line 112** --- Now reads `[B, H, S, 256]`. Correct (matches post-transpose layout established earlier in the file).
2. **`global_proportional_rope.md` Strategy B forward pass (lines 232--243)** --- Shape comments now use `[B, H, S, D]` ordering throughout (e.g., `[B, H, S, 128]`, `[B, H, S, 384]`, `[B, H, S, 512]`). Correct (matches the forward pass diagram at line 141).

Checked all four files for remaining factual issues: inv_freq formulas, table shapes, wavelength calculations, memory footprint arithmetic, layer/head counts, Strategy A vs B attribution, position indexing logic, and tensor layouts. All consistent and correct.

**No feedback --- chapter approved.**
