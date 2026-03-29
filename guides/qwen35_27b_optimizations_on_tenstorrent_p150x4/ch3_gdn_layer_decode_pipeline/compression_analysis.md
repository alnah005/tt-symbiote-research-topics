# Compression Analysis: GDN Layer Decode Pipeline — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~350 lines
- Estimated post-compression line count: ~330 lines
- Estimated reduction: ~6%

## CRUCIAL Suggestions

None.

## MINOR Suggestions

### 1. [gdn_decode_flow.md / recurrence_math.md] ~lines 97-110 / 9-88
**Issue:** L2 normalization and head expansion are explained in both `gdn_decode_flow.md` (unfused path, lines 100-102) and `recurrence_math.md` (lines 9-38, 42-54) with code snippets.
**Suggestion:** The decode flow file should reference `recurrence_math.md` for these details rather than inline them. A single sentence like "Q and K are L2-normalized and expanded to value heads (see recurrence_math.md)" would replace roughly 6 lines without information loss.

### 2. [conv1d_shift_register.md] ~lines 62-70
**Issue:** The SiLU definition `SiLU(x) = x * sigmoid(x)` is standard knowledge for the target audience and is already contextually apparent from the `ttnn.silu` call. The 6-line "SiLU Activation" section is more than needed.
**Suggestion:** Reduce to a single line noting that SiLU is applied after the weighted sum.

### 3. [gdn_decode_flow.md / recurrence_math.md] ~lines 89-90 / 42-52
**Issue:** The calculation `num_pairs = B * Nv_TP = 32 * 12 = 384` and `repeat_factor = Nv_TP / Nk_TP = 12 / 4 = 3` appear in both files.
**Suggestion:** Define these once in the decode flow (where they first appear) and reference them thereafter.

### 4. [conv1d_shift_register.md] ~lines 74-80
**Issue:** The trace-compatibility "Why" section restates the two properties (fixed tensor IDs, no dynamic control flow) that were already demonstrated in the preceding sections.
**Suggestion:** Cut to a brief summary sentence, since the mechanism section already makes the point implicitly.

### 5. Shape annotation `[1, 32, 2560]`
**Issue:** Appears six times across the chapter (gdn_decode_flow.md lines 28, 58; conv1d_shift_register.md lines 16, 70; and twice more inline).
**Suggestion:** After the first occurrence with full derivation, subsequent files could use the symbolic form `[1, B, qkv_dim_tp]` without the numeric expansion.

## Load-Bearing Evidence
- `gdn_decode_flow.md` line ~3: "The TtGatedDeltaNet class in gdn.py implements both a fused path and an unfused fallback" — load-bearing because this dual-path design is the structural frame for the chapter and referenced by Ch4
- `conv1d_shift_register.md` line ~34: "ttnn.copy(src, dst) operation copies the data from src into dst in-place, preserving the tensor ID of the destination" — load-bearing because this is the key property enabling trace compatibility, cannot be cut
- `recurrence_math.md` line ~93: "state[t] = exp(g[t]) * state[t-1] + outer(k[t], delta[t])" — load-bearing because this is the canonical recurrence equation referenced by Ch4, Ch5, and Ch7
- `gdn_decode_flow.md` line ~58: "conv_out = ttnn.silu(conv_acc)" — load-bearing as the pipeline stage connecting conv1d to the recurrence

## VERDICT
- Crucial updates: no
