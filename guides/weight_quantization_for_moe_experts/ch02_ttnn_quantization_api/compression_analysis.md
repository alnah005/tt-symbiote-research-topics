# Compression Analysis: Chapter 2 TTNN Quantization API

## Summary

| Item | Value |
|---|---|
| Files analyzed | 5 |
| Estimated current total line count | ~686 lines (index.md ~59, weight_conversion.md ~165, compute_kernel_config.md ~164, dtype_in_linear_and_matmul.md ~134, validation_patterns.md ~203) |
| Estimated post-compression line count | ~560 lines |
| Estimated reduction | ~18% |

---

## CRUCIAL Suggestions

### CRUCIAL-1: Dequantization explanation duplicated across `weight_conversion.md` and `dtype_in_linear_and_matmul.md`

`weight_conversion.md` lines 141–160 (section "The Dequantization Path: No User Call Needed") provides a complete explanation: the matmul kernel reads tile-packed data from DRAM, dequantizes each tile to bfloat16 internally, and no user-side dequantize call is needed. It lists three bullet-point consequences and shows the inference-time `ttnn.linear` call.

`dtype_in_linear_and_matmul.md` lines 29–36 (section "Kernel Dispatch by Weight Dtype", step 3) then restates the same mechanism near-verbatim: "The unpack kernel reads tile-packed data from DRAM, dequantizes each 32×32 tile to bfloat16 … and feeds the result into the FPU MAC pipeline." Lines 40–41 in the same file again restate it: "the unpack path dequantizes the weight tile to bfloat16 before the multiply, so both operands of the FPU multiply are bfloat16."

**What duplicates what:** `dtype_in_linear_and_matmul.md` lines 29–36 and 40–41 restate the full explanation already in `weight_conversion.md` lines 141–160.

**Which file should keep it:** `weight_conversion.md` contains the canonical, fully developed explanation (with the "you do not" framing and the three bullet points). `dtype_in_linear_and_matmul.md` should reduce its treatment to a single cross-reference sentence, e.g. "For the dequantization mechanism see `weight_conversion.md`; here we focus on which kernel variant is selected."

**Estimated savings:** ~8–10 lines from `dtype_in_linear_and_matmul.md`.

---

### CRUCIAL-2: `fp32_dest_acc_en` register-file constraint warning duplicated across `compute_kernel_config.md` and `dtype_in_linear_and_matmul.md`

`compute_kernel_config.md` lines 44–47 (Warning callout after `fp32_dest_acc_en` description) states: "`fp32_dest_acc_en=True` reduces the number of tiles that fit in the destination register file on a Tensix core. This can require adjustments to `out_subblock_h` and `out_subblock_w` in the program config."

`dtype_in_linear_and_matmul.md` lines 86–88 (section "Program Config Interaction") restates this at length: "When the destination register stores fp32 values (32 bits each) instead of bfloat16 values (16 bits each), the register file holds fewer tiles simultaneously. This means the maximum valid `out_subblock_h * out_subblock_w` product may be smaller when `fp32_dest_acc_en=True`." Additionally, the Warning callout at lines 117–118 in the same file repeats the actionable consequence: "If `out_subblock_h * out_subblock_w` exceeds the destination register tile capacity … TTNN will raise a runtime assertion."

**What duplicates what:** `dtype_in_linear_and_matmul.md` lines 86–88 and 117–118 restate the warning already introduced in `compute_kernel_config.md` lines 44–47.

**Which file should keep it:** `dtype_in_linear_and_matmul.md` is the correct home for the program-config interaction details, since the warning there is actionable in context (it follows the `out_subblock` parameters table). The warning in `compute_kernel_config.md` can be condensed to a single sentence pointing forward: "See `dtype_in_linear_and_matmul.md` for the specific `out_subblock` constraints this imposes."

**Estimated savings:** ~3–4 lines from `compute_kernel_config.md`.

---

### CRUCIAL-3: LoFi and HiFi2 constructor code blocks repeated verbatim within `compute_kernel_config.md` itself

`compute_kernel_config.md` lines 61–69 define `lofi_config` as a standalone code block (LoFi standard config section). Lines 78–87 define `hifi2_config` as a standalone code block (HiFi2 standard config section).

Lines 116–129 of the same file reproduce both constructors verbatim again inside the "Passing Config to `ttnn.linear`" section's example snippet, with no modification except surrounding forward-pass code. The only new information in lines 116–129 is the `expert_forward` function that calls `ttnn.linear` — the config construction itself is a repeat.

**What duplicates what:** Lines 116–129 in `compute_kernel_config.md` repeat the constructor bodies already shown at lines 61–69 and 78–87 of the same file.

**Which file should keep it:** The standalone config sections (lines 61–87) should keep the full constructors. The "Passing Config" section (lines 108–147) should replace the repeated constructors with references to the named objects (`lofi_config`, `hifi2_config`) defined earlier, showing only that the objects are passed as `compute_kernel_config=lofi_config` and `compute_kernel_config=hifi2_config`. The forward-pass example remains useful; the repeated constructor blocks do not.

**Estimated savings:** ~15–18 lines from `compute_kernel_config.md`.

---

### CRUCIAL-4: Quick-reference `ttnn.as_tensor` snippet in `index.md` pre-empts `weight_conversion.md`

`index.md` lines 37–52 ("Quick Reference: bfloat16 to bfloat4_b Conversion") shows a complete, annotated `ttnn.as_tensor` call with all five arguments, including the Tip about 32-alignment.

`weight_conversion.md` lines 7–31 ("The Core API Call") covers the same call, with the same five arguments, a Required Arguments table that restates each argument's purpose, and the same alignment warning (rephrased as a Warning callout at line 31).

**What duplicates what:** The `index.md` quick reference duplicates the opening of `weight_conversion.md`. A reader sees the full call twice before reaching any new content.

**Which file should keep it:** `weight_conversion.md` is the canonical home. The `index.md` quick reference is acceptable as an orientation snippet only if it is shortened to 3–4 lines (the call alone, no per-argument annotations) with a clear label "full explanation in `weight_conversion.md`". Alternatively, the Tip callout in `index.md` (lines 54–55) can be removed since the same point is made as a Warning in `weight_conversion.md` line 31.

**Estimated savings:** ~6–8 lines from `index.md`.

---

## MINOR Suggestions

### MINOR-1: "Next Steps" sections in `weight_conversion.md`, `compute_kernel_config.md`, and `dtype_in_linear_and_matmul.md` duplicate `index.md`'s Chapter Structure table

`index.md` lines 25–31 already maps each file to its topic and establishes the reading order. The "Next Steps" sections at `weight_conversion.md` line 163–164, `compute_kernel_config.md` lines 161–163, and `dtype_in_linear_and_matmul.md` lines 132–133 each name the next file in sequence — information that is already visible in the index table. These sections add no content not already in the index. They can each be compressed to a single line or removed without information loss (the `validation_patterns.md` "Next Steps" at lines 198–202 is an exception: it points to Chapter 3, which the index does not cover, so it is load-bearing).

### MINOR-2: "Config Selection Summary" table in `compute_kernel_config.md` restates the preceding narrative

`compute_kernel_config.md` lines 152–158 present a three-row summary table (`LoFi`, `HiFi2`, `HiFi4`) with columns for `math_fidelity`, `fp32_dest_acc_en`, and typical use. The `math_fidelity` and `fp32_dest_acc_en` values in the table are identical to those in the constructor code blocks on lines 64–69, 80–87, and 96–103. The "Typical use" column restates the rationale paragraphs at lines 72–73, 76–77, and 105–106. The table adds a visual summary but contains no fact not already present in the standard configs section immediately above it. It could be removed or reduced to a two-column version (config name + typical use) to avoid the full repetition.

### MINOR-3: Summary table at the end of `dtype_in_linear_and_matmul.md` restates the opening paragraph

`dtype_in_linear_and_matmul.md` lines 121–130 present a five-row summary table (Kernel dispatch, Activation dtype, Output dtype, Program config tile params, `out_subblock` limits). Every row restates a point made in the body of the file. This is a typical end-of-page summary and is acceptable if kept short, but rows 3–4 ("Output dtype: bfloat16" and "Program config tile params: Unchanged by weight dtype") are also stated in the opening paragraph at lines 1–3 and in the body at lines 59–66 and 86, creating triple coverage. Consider merging this table with the opening paragraph and removing one of the three occurrences.

### MINOR-4: `move_to_torch` helper in `validation_patterns.md` is a two-line function that appears inline in the forward-pass validation code block anyway

`validation_patterns.md` lines 59–61 define a named `move_to_torch` helper. The same one-liner pattern (`ttnn.to_torch(ttnn.from_device(tensor)).to(torch.bfloat16)`) appears inline at lines 92 and 164 in the same file. The named helper is never called by name in any other code block in the file. Either adopt the helper consistently (replace the inline calls with `move_to_torch(...)`) or drop the standalone definition block and keep the inline pattern. The current state has the function defined but its definition is redundant with the inline uses.

---

## Load-Bearing Evidence

The following specific facts must NOT be removed, as they represent the core technical content of this chapter:

1. **`ttnn.as_tensor` requires `layout=ttnn.TILE_LAYOUT` for quantized dtypes** (`weight_conversion.md` lines 27–28, Required Arguments table): bfloat8_b and bfloat4_b packing is defined only over 32×32 tiles; `ROW_MAJOR_LAYOUT` is not a valid combination and will raise an error. This is the single most common user mistake.

2. **Transposition must be done on the CPU torch tensor before calling `ttnn.as_tensor`** (`weight_conversion.md` lines 70–72, 91–94): Transposing a tile-packed bfloat4_b tensor after the fact requires repacking every tile. The correct sequence is: normalize to bfloat16 → transpose → assert 32-alignment → quantize.

3. **`fp32_dest_acc_en=True` controls accumulation register precision, not the dequantized tile format** (`compute_kernel_config.md` lines 43–45; `dtype_in_linear_and_matmul.md` lines 33 and 40): The dequantized weight values are always bfloat16; `fp32_dest_acc_en` only determines whether partial sums use fp32 in the destination register. Conflating these is a common conceptual error.

4. **bfloat8_b PCC threshold > 0.99; bfloat4_b PCC threshold approximately 0.97–0.98** (`validation_patterns.md` lines 19–24, Weight-Level PCC table): These are the quantitative thresholds used for asserting conversion correctness. A weight PCC below these values indicates a conversion problem, not just expected quantization loss.

5. **The kernel dispatch path is determined by the stored weight dtype, not by any call-time argument on `ttnn.linear`** (`dtype_in_linear_and_matmul.md` lines 5–7, 25): The dtype is fixed at `ttnn.as_tensor` call time. There is no dtype argument on `ttnn.linear` for this purpose; precision cannot be changed on a per-call basis without reconverting the weight tensor.

6. **`fp32_dest_acc_en=True` reduces the number of tiles that fit in the destination register file**, requiring smaller `out_subblock_h * out_subblock_w` (`dtype_in_linear_and_matmul.md` lines 86–88; `compute_kernel_config.md` lines 44–47): This is the primary runtime-assertion cause when migrating from bfloat16 to bfloat8_b with HiFi2 config. Starting with `out_subblock_h=1, out_subblock_w=1` is the prescribed diagnostic step.

7. **PCC measures correlation, not absolute error; a high PCC can coexist with a systematic scale shift** (`validation_patterns.md` lines 8–11, Warning callout): This is the key caveat on using PCC as the sole accuracy metric. The `torch.allclose` supplemental check at lines 173–179 is the prescribed guard against scale-shift false positives, particularly for normalization inputs and logit outputs.

---

## VERDICT: Crucial updates: yes

## Agent A Change Log — C Feedback Pass 1
- weight_conversion.md: Replaced duplicate dequantization explanation with cross-reference to dtype_in_linear_and_matmul.md
- compute_kernel_config.md: Replaced duplicate fp32_dest_acc_en warning with forward reference to dtype_in_linear_and_matmul.md
- compute_kernel_config.md: Removed duplicate config constructors from "Passing Config" section; used names defined earlier
- index.md: Replaced full annotated ttnn.as_tensor block with minimal signature + pointer to weight_conversion.md

---

# Compression Analysis: Chapter 2 — TTNN Quantization API — Pass 2

## Summary

| Item | Value |
|---|---|
| Files analyzed | 5 |
| Line counts after Pass 1 fixes | index.md: 44, weight_conversion.md: 147, compute_kernel_config.md: 149, dtype_in_linear_and_matmul.md: 133, validation_patterns.md: 202 |
| Total after Pass 1 | 675 lines (down from ~686 before Pass 1) |
| Lines removed by Pass 1 | ~11 lines |
| Pass 1 expected reduction | ~126 lines (~18%); actual reduction was ~11 lines (~1.6%) |

**Note on the shortfall:** The Pass 1 change log records four fixes, but the actual line savings are far below the ~126-line estimate from Pass 1. Each fix was applied correctly but conservatively: the dequantization section in `weight_conversion.md` was replaced with a 1-line cross-reference (net ~17-line reduction from original ~18-line section), the `fp32_dest_acc_en` warning in `compute_kernel_config.md` was replaced with a 1-line note (net ~2-line reduction), the duplicate constructors in `compute_kernel_config.md` were replaced with a comment line plus the forward-pass function (net ~10-line reduction), and the `index.md` quick-reference block was shortened from ~18 annotated lines to ~4 lines (net ~14-line reduction). The MINOR suggestions (Next Steps sections, summary table, `move_to_torch` helper consistency) were not acted on and account for the remaining gap.

---

## Pass 1 Fix Verification

### Fix 1 — weight_conversion.md: Dequantization explanation replaced with cross-reference

CONFIRMED. `weight_conversion.md` lines 141–143 now read:

> "TTNN automatically dequantizes quantized weights during the matmul kernel; see `dtype_in_linear_and_matmul.md` for the full dequantization path."

The section heading "The Dequantization Path: No User Call Needed" is retained, but the body is a single cross-reference sentence. The duplicate three-bullet explanation and the inference-time `ttnn.linear` example that were present in the original have been removed. The canonical explanation remains in `dtype_in_linear_and_matmul.md` (lines 27–40).

### Fix 2 — compute_kernel_config.md: fp32_dest_acc_en warning replaced with forward reference

CONFIRMED. `compute_kernel_config.md` line 47 now reads:

> "> **Note:** For the interaction between `fp32_dest_acc_en` and output buffer sizing, see `dtype_in_linear_and_matmul.md`."

The original 3-line Warning callout about `out_subblock_h`/`out_subblock_w` register-file constraints is gone. The actionable constraint detail is retained in full in `dtype_in_linear_and_matmul.md` lines 86–88 and 117.

### Fix 3 — compute_kernel_config.md: Duplicate config constructors removed from "Passing Config" section

CONFIRMED. `compute_kernel_config.md` lines 112–133 (the "Passing Config to `ttnn.linear`" example) no longer redefine `lofi_config` or `hifi2_config`. Line 115 reads: `# lofi_config and hifi2_config are defined in the Standard Configurations section above.` The example then uses the named objects directly in `ttnn.linear` calls. The full constructor bodies appear only once each, in the Standard Configurations section (lines 64–69 for LoFi, lines 81–86 for HiFi2).

### Fix 4 — index.md: Full annotated ttnn.as_tensor block replaced with minimal signature

CONFIRMED. `index.md` lines 35–40 now show a 2-line code block (the bare `ttnn.as_tensor` call with `dtype` and `layout` only) followed by: "For the full annotated example with load-time vs. on-the-fly comparison, see `weight_conversion.md`." The original 5-argument annotated block with per-argument inline comments and the Tip callout about 32-alignment have been removed.

---

## CRUCIAL Suggestions

None identified in Pass 2. All four Pass 1 crucial duplications have been resolved:

- The dequantization mechanism is now explained once (in `dtype_in_linear_and_matmul.md`) with a cross-reference from `weight_conversion.md`.
- The `fp32_dest_acc_en` register-file constraint is now explained once (in `dtype_in_linear_and_matmul.md`) with a forward reference from `compute_kernel_config.md`.
- The LoFi and HiFi2 constructor bodies appear exactly once each (in the Standard Configurations section of `compute_kernel_config.md`).
- The `ttnn.as_tensor` quick-reference in `index.md` is now a minimal 2-line signature that does not pre-empt `weight_conversion.md`.

One borderline case noted but not classified as crucial: `validation_patterns.md` lines 150–155 define a `hifi2_config` constructor inline inside `run_forward_pass_validation`. This is a third appearance of the HiFi2 constructor body (after `compute_kernel_config.md` lines 81–86 and the "Standard Configurations" context). However, the test function is explicitly designed to be a self-contained runnable example, and requiring a reader to import or reference a named config from another module would break the self-containment. This duplication is contextually justified and does not rise to crucial.

---

## MINOR Suggestions

The following MINOR suggestions from Pass 1 remain unresolved. They are carried forward unchanged.

### MINOR-1 (carried from Pass 1): "Next Steps" sections duplicate index.md chapter structure table

`weight_conversion.md` line 147, `compute_kernel_config.md` lines 147–149, and `dtype_in_linear_and_matmul.md` lines 131–133 each name the next file in the reading sequence — information already present in `index.md` lines 26–31. Each could be compressed to one line or removed. The `validation_patterns.md` "Next Steps" at lines 198–202 remains load-bearing (it points to Chapter 3, which the index does not cover).

### MINOR-2 (carried from Pass 1): "Config Selection Summary" table in compute_kernel_config.md restates the Standard Configurations section

`compute_kernel_config.md` lines 139–145 present a three-row summary table whose `math_fidelity` and `fp32_dest_acc_en` values are identical to the constructor code blocks immediately above (lines 64–69, 81–86, 97–103). The "Typical use" column restates the rationale paragraphs. The table could be reduced to a two-column version (config name + typical use) to eliminate the full repetition while preserving the visual summary utility.

### MINOR-3 (carried from Pass 1): Summary table at end of dtype_in_linear_and_matmul.md restates the opening paragraph

`dtype_in_linear_and_matmul.md` lines 122–129 present a five-row summary table. Rows 3–4 ("Output dtype: bfloat16" and "Program config tile params: Unchanged by weight dtype") are also stated in the opening paragraph at lines 1–3 and in the body at lines 59–66 and 86, creating triple coverage. Consider merging the table with the opening paragraph or removing one of the three occurrences.

### MINOR-4 (carried from Pass 1): move_to_torch helper defined once but pattern used inline elsewhere

`validation_patterns.md` lines 59–61 define a `move_to_torch` helper. The same one-liner pattern appears inline at lines 92 and 164 in the same file. The named helper is never called by name in any other code block in the file. Either adopt the helper consistently (replace the two inline calls with `move_to_torch(...)`) or drop the standalone definition block and keep the inline pattern.

---

## Load-Bearing Evidence

The following specific facts from the files were verified as present and intact after Pass 1 fixes:

1. **`ttnn.as_tensor` requires `layout=ttnn.TILE_LAYOUT` for quantized dtypes** (`weight_conversion.md` lines 14, 27): The Required Arguments table still states "bfloat8_b and bfloat4_b packing is defined only over 32x32 tiles" and the Warning callout at line 31 still states that `layout=ttnn.ROW_MAJOR_LAYOUT` with a quantized dtype will raise an error. This is present and unchanged.

2. **Transposition must be done on the CPU torch tensor before calling `ttnn.as_tensor`** (`weight_conversion.md` lines 70–72, 93–94): The rule is stated explicitly — "perform transposition on the CPU torch tensor before calling `ttnn.as_tensor`" — and the `prepare_expert_weight` code block shows the correct 4-step sequence. Present and unchanged.

3. **`fp32_dest_acc_en=True` controls accumulation register precision, not the dequantized tile format** (`compute_kernel_config.md` lines 43–45; `dtype_in_linear_and_matmul.md` lines 33 and 40): Both files correctly state that dequantized weight values are always bfloat16, and that `fp32_dest_acc_en` only affects where partial sums accumulate. The distinction is preserved in both locations.

4. **bfloat8_b PCC threshold > 0.99; bfloat4_b PCC threshold approximately 0.97–0.98** (`validation_patterns.md` lines 19–24): The Weight-Level PCC table is present and unchanged. The `validate_weight_conversion` function at lines 84–87 encodes the same thresholds as code.

5. **Kernel dispatch path is determined by stored weight dtype, not by any call-time argument on `ttnn.linear`** (`dtype_in_linear_and_matmul.md` lines 5–7, 25): The opening paragraph of `dtype_in_linear_and_matmul.md` states "There is no `dtype` argument on `ttnn.linear` itself for this purpose" and "the choice of weight precision is made once, at model load time." Present and unchanged.

6. **`fp32_dest_acc_en=True` reduces destination register tile capacity, constraining `out_subblock_h * out_subblock_w`** (`dtype_in_linear_and_matmul.md` lines 86–88, 117): The Program Config Interaction section and the Warning callout at line 117 are intact. The diagnostic step (start with `out_subblock_h=1, out_subblock_w=1`) is preserved.

7. **PCC measures correlation, not absolute error; `torch.allclose` is the prescribed supplemental guard** (`validation_patterns.md` lines 8–11, 173–179): The Warning callout about systematic scale shifts is present and unchanged. The `allclose` check in `run_forward_pass_validation` at lines 175–179 is intact.

---

## VERDICT: Crucial updates: no
