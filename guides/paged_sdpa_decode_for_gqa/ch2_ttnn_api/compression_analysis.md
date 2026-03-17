# Compression Analysis: ch2_ttnn_api

**Files analyzed:** `index.md`, `function_signature.md`, `tensor_shape_reference.md`, `sdpa_program_config.md`

---

## CRUCIAL Duplications

### CRUCIAL-1: `page_table_tensor` int32 silent-failure explanation

**Files and lines:**
- `function_signature.md` lines 191–196: warning callout stating the kernel reads raw bytes as int32, producing incorrect physical block addresses, with no error raised.
- `tensor_shape_reference.md` lines 54–59: same warning, extended with the float32 bit-pattern worked example (`2.0 → 0x40000000 = 1073741824`).

**What is duplicated:**
Both files state: "If `page_table_tensor` is int64 or float32, the kernel reads the raw byte buffer as packed int32 values; incorrect physical block addresses result; no dtype error is raised." The `tensor_shape_reference.md` version adds the bit-pattern arithmetic example, which is the more informative version.

**Resolution:**
- `tensor_shape_reference.md` **keeps** the full explanation including the bit-pattern example (lines 54–59). This is the authoritative dtype-constraint section.
- `function_signature.md` lines 191–196 should be **replaced** with a one-sentence cross-reference: `> **[SILENT FAILURE]** See the Dtype Constraints section in tensor_shape_reference.md for the full description of the int32 requirement and the consequences of passing int64 or float32.`

---

### CRUCIAL-2: Old KV cache layout `[nkv x b x s x dh]` silent-failure explanation

**Files and lines:**
- `function_signature.md` lines 71–75: warning callout stating the old layout was `[nkv x b x s x dh]`, the current is `[b x nkv x s x dh]`, and passing an old-layout tensor silently indexes wrong heads.
- `tensor_shape_reference.md` lines 106–114: same warning, extended with the concrete example (`nkv=4, b=8` — kernel sees "8 KV heads and 4 batch elements") and the diagnostic tip (inspect axis 0).

**What is duplicated:**
Both files state the axis-swap between old and new non-paged KV layout and that no shape error is raised. `tensor_shape_reference.md` additionally explains why no error is raised (product of axes is identical) and gives the diagnostic procedure.

**Resolution:**
- `tensor_shape_reference.md` **keeps** the full explanation with the concrete example and diagnostic tip (lines 106–117). The Cache Layout History section is the authoritative location.
- `function_signature.md` lines 71–75 should be **replaced** with a one-sentence cross-reference: `> **[SILENT FAILURE]** The non-paged KV cache layout changed at v0.55. See the Cache Layout History section in tensor_shape_reference.md for the full description, the worked example, and the diagnostic procedure.`

---

### CRUCIAL-3: `num_cores >= b * nkv` constraint explanation

**Files and lines:**
- `tensor_shape_reference.md` lines 77–98: full 22-line section titled "The `num_cores >= b * nkv` Constraint" covering the one-core-per-pair assignment, the violation symptom (wrong/zero output for some pairs), the first-`num_cores`-pairs-correct pattern, and the fix.
- `sdpa_program_config.md` lines 44–53: re-states the same constraint under `compute_with_storage_grid_size`, including the "some pairs are never computed / output left unwritten" failure description, followed by a scenario table (lines 55–60).

**What is duplicated:**
Both files independently derive why under-provisioning the grid leaves output positions unwritten and describe the silent-failure symptom. The scenario table in `sdpa_program_config.md` is unique and should be kept; the prose failure description is the duplicate.

**Resolution:**
- `tensor_shape_reference.md` lines 77–98 **keeps** the full standalone constraint section, as it is the natural place to document a correctness constraint associated with tensor outputs.
- `sdpa_program_config.md` lines 46–49 (the prose failure description, starting "The kernel assigns one..." through "...see the constraint section in tensor_shape_reference.md for the full failure description") should be **condensed**: the existing cross-reference sentence on line 49 (`see the constraint section in tensor_shape_reference.md`) is already correct; the two preceding sentences that re-derive the failure should be removed, leaving only the hard constraint statement and the cross-reference. The scenario table (lines 55–60) is unique and must be kept.

---

## MINOR Items

### MINOR-1: `pnh` formula repeated four times

`pnh = ceil(nh / 32) * 32` is stated in:
- `function_signature.md` line 44 (in the `input_tensor_q` parameter prose)
- `function_signature.md` line 214 (in the Output Tensor section)
- `tensor_shape_reference.md` line 14 (Master Shape Table Notes column)
- `tensor_shape_reference.md` line 136 (GQA Padding Gotcha worked example)

The formula is load-bearing in the output section and the GQA padding section; the two inline mentions in the parameter prose and the table Notes column are redundant. Consider replacing the Notes-column instance with a pointer to the Output Tensor section, and removing the inline derivation on `function_signature.md` line 44 in favor of a parenthetical `(see Output Tensor section below)`.

### MINOR-2: `sdpa_program_config.md` "Default Config Behavior" section partially restates `function_signature.md`

`function_signature.md` lines 153–156 (the `program_config` parameter entry) notes: "When `None` the kernel selects a default grid, which is often suboptimal for small-batch GQA configurations (e.g., b=1, nkv=4). See `sdpa_program_config.md` for field details and a worked example." `sdpa_program_config.md` lines 173–179 then re-states the same suboptimality claim with a concrete number (60 idle cores on an 8x8 device). The `function_signature.md` entry is a pointer and is fine; the `sdpa_program_config.md` section adds quantitative detail that is not duplicated. No change needed beyond noting the intentional layering — but the sentence "Getting this configuration wrong is one of the most common sources of silent performance and correctness regressions" at `sdpa_program_config.md` lines 4–6 partially echoes the index.md description without adding new information. This sentence can be trimmed.

### MINOR-3: Single-sentence "next steps" navigation embedded in parameter descriptions

`function_signature.md` line 155: "See `sdpa_program_config.md` for field details and a worked example." This is a navigation pointer that duplicates what `index.md` already provides in the Chapter Contents table (line 56). The pointer is useful in-context for a reader who jumps directly to the file, so it is low-priority but could be removed without information loss.

### MINOR-4: Master Shape Table restates information visible in the function signature code block

`tensor_shape_reference.md` lines 12–19 (Master Shape Table) lists all tensor shapes, layouts, and dtypes. All of those shapes are already visible in the annotated function signature at `function_signature.md` lines 10–27. The table does add the Layout and Dtype columns explicitly and the Notes column, which are not in the code comment, so the table is partially load-bearing. The pure shape columns (Non-paged shape, Paged shape) are fully redundant with the code comments. A compression option is to drop the shape columns from the table and keep only Layout, Dtype, and Notes, with a header note pointing to the function signature for shapes.

---

## Load-Bearing Evidence

The following facts must NOT be removed from their current location, as they are authoritative technical statements not duplicated elsewhere:

1. **`tensor_shape_reference.md` lines 56–59** — The bit-pattern example `float32(2.0) → 0x40000000 = 1073741824` making "block index 2" become "block index 1073741824". This concrete numerical illustration of the int32 silent failure is unique to this file and is the most actionable diagnostic detail in the chapter.

2. **`tensor_shape_reference.md` lines 121–147** — The GQA Padding Gotcha section, including the three-step padding procedure (`pnh = ceil(nh/32)*32`, then `nkv_padded = pnh / group_size`) and the worked numerical example for `nkv=4, nh=16`. This derivation appears nowhere else in the chapter.

3. **`sdpa_program_config.md` lines 55–60** — The scenario table mapping `(b, nkv)` combinations to required core counts and example grids (Single-batch GQA through Batch-8 MHA). This is the only place in the chapter that gives concrete grid-sizing guidance for multiple configurations.

4. **`sdpa_program_config.md` lines 83–100** — The `k_chunk_size` field description explaining the online softmax accumulation granularity, the numerical-stability trade-off at 32K+ tokens, and the `k_chunk_size % block_size == 0` paged-mode constraint with the illegal `k_chunk_size=48, block_size=32` example. This constraint is not stated anywhere else.

5. **`sdpa_program_config.md` lines 104–129** — The Parallelization Strategy section (steps 1–4 and the group_size implication), specifically the statement that one core computes attention for an entire GQA group of query heads and that per-core work scales with `group_size * s / k_chunk_size` iterations. This is the only place in the chapter that explains why GQA decode can be slower than MHA decode.

6. **`function_signature.md` lines 117–120** — The recompilation trade-off for `cur_pos`: "the kernel is retraced and recompiled for each unique combination of values ... accumulates many cached programs. Prefer `cur_pos_tensor` for production serving." This production-serving recommendation is not restated in `tensor_shape_reference.md`.

7. **`tensor_shape_reference.md` lines 33–48** — The Layout Requirements section, specifically the explanation that TTNN does not support int32 in tile layout, which is why `page_table_tensor` and `cur_pos_tensor` must be row-major. The "why" (int32 tile-layout limitation) appears only here.

---

## VERDICT

Crucial updates: yes

---

## Summary

| Metric | Value |
|--------|-------|
| Total files analyzed | 4 |
| Estimated current line count | 66 (`index.md`) + 228 (`function_signature.md`) + 148 (`tensor_shape_reference.md`) + 180 (`sdpa_program_config.md`) = **622 lines** |
| Estimated post-compression line count | ~530 lines (remove ~22 lines from CRUCIAL-1/2, condense ~8 lines from CRUCIAL-3, trim ~20 lines from MINOR-1/2/4 partial reductions, net ~62 lines saved) |
| Estimated reduction | ~10% |

The bulk of the chapter is non-redundant. Three crucial duplications exist (the `page_table_tensor` int32 failure, the KV layout axis-swap failure, and the `num_cores >= b * nkv` constraint prose), all of which involve a short version in `function_signature.md` or `sdpa_program_config.md` restating a more complete version in `tensor_shape_reference.md`. The fix in each case is to replace the shorter version with a cross-reference rather than deleting information.

## Agent A Change Log — C Feedback Pass 1
- function_signature.md: Replaced duplicate page_table_tensor int32 warning with cross-reference to tensor_shape_reference.md
- function_signature.md: Replaced duplicate old KV axis-swap warning with cross-reference to tensor_shape_reference.md
- sdpa_program_config.md: Removed re-derived num_cores failure prose; retained unique scenario table; kept existing cross-reference sentence

---

# Compression Analysis: Chapter 2 — TTNN paged_sdpa_decode API — Pass 2

**Files analyzed:** `index.md`, `function_signature.md`, `tensor_shape_reference.md`, `sdpa_program_config.md`

---

## Summary

| Metric | Value |
|--------|-------|
| `index.md` | 65 lines |
| `function_signature.md` | 219 lines (was 228) |
| `tensor_shape_reference.md` | 147 lines (was 148) |
| `sdpa_program_config.md` | 176 lines (was 180) |
| **Total** | **607 lines** (was 622; net reduction of 15 lines = ~2.4%) |

---

## Pass 1 Fix Verification

### Fix 1 — `function_signature.md`: `page_table_tensor` int32 warning

**Status: CONFIRMED APPLIED.**

`function_signature.md` line 187 now reads:

> `**[SILENT FAILURE]** See 'tensor_shape_reference.md' for the full page_table_tensor dtype/layout constraint and the silent-failure consequences.`

The original 6-line warning callout (which re-stated the raw-byte-reinterpretation explanation and the consequence of no error being raised) has been replaced with this single-sentence cross-reference. The authoritative full explanation with the float32 bit-pattern example (`2.0 → 0x40000000 = 1073741824`) remains intact at `tensor_shape_reference.md` lines 54–59.

---

### Fix 2 — `function_signature.md`: Old KV layout axis-swap warning

**Status: CONFIRMED APPLIED.**

`function_signature.md` line 71 now reads:

> `**[SILENT FAILURE]** The non-paged KV cache layout changed at v0.55. See 'tensor_shape_reference.md' for the full axis-order constraint and concrete diagnostic example.`

The original 5-line warning callout (which re-stated the `[nkv x b x s x dh]` → `[b x nkv x s x dh]` axis-swap and the silent wrong-head-indexing failure) has been replaced with this single-sentence cross-reference. The authoritative full explanation with the concrete `nkv=4, b=8` worked example and the diagnostic tip (inspect axis 0) remains intact at `tensor_shape_reference.md` lines 106–117.

---

### Fix 3 — `sdpa_program_config.md`: Re-derived `num_cores` failure prose

**Status: CONFIRMED APPLIED.**

`sdpa_program_config.md` lines 44–46 now read:

```
**Hard constraint:** `cols * rows >= b * nkv`.

See the constraint section in `tensor_shape_reference.md` for the full failure description.
```

The two prose sentences that previously re-derived why under-provisioning the grid leaves output positions unwritten ("The kernel assigns one..." through "...some pairs are never computed / output left unwritten") have been removed. The hard constraint statement, the cross-reference sentence, and the scenario table (lines 52–57) are all retained.

---

## CRUCIAL Suggestions

None — all Pass 1 CRUCIAL items resolved.

All three crucial duplications identified in Pass 1 have been correctly addressed. No new crucial duplications were found during Pass 2 review. Specifically checked and confirmed non-duplicated:

- `cur_pos` scalar-vs-list silent failure: appears only in `function_signature.md` lines 107–111.
- `scale` wrong-value silent degradation: appears only in `function_signature.md` lines 139–141.
- K/V dtype mismatch silent failure: appears only in `tensor_shape_reference.md` lines 61–64.
- `k_chunk_size % block_size == 0` constraint with illegal example: appears only in `sdpa_program_config.md` lines 92–97.
- Output slicing worked example (`output[:, :, :nh, :]`): full derivation only in `function_signature.md` lines 204–219; `tensor_shape_reference.md` line 17 carries only a brief table-note pointer, which is not a duplication of the worked example.
- Default config idle-core quantification (60 idle cores on 8×8): full statement only in `sdpa_program_config.md` lines 172–176; `function_signature.md` lines 149–152 is a deliberate pointer, not a re-derivation.

---

## MINOR Suggestions

The following MINOR items from Pass 1 remain unresolved. None are crucial.

### MINOR-1 (carry-forward): `pnh` formula repeated four times

`pnh = ceil(nh / 32) * 32` still appears in:
- `function_signature.md` line 44 (input_tensor_q parameter prose)
- `function_signature.md` line 206 (Output Tensor section)
- `tensor_shape_reference.md` line 14 (Master Shape Table Notes column)
- `tensor_shape_reference.md` line 141 (GQA Padding Gotcha worked example)

The Output Tensor occurrence and the GQA Padding Gotcha occurrence are load-bearing. The inline derivation in the input_tensor_q prose (line 44) and the Notes column in the Master Shape Table (line 14) are redundant with those. Optional compression: replace the line-44 inline formula with `(see Output Tensor section below)` and replace the Notes-column formula with a pointer to the Output Tensor section.

### MINOR-2 (carry-forward): `sdpa_program_config.md` opening sentence partially echoes `index.md`

`sdpa_program_config.md` lines 4–6: "Getting this configuration wrong is one of the most common sources of silent performance and correctness regressions in multi-batch or GQA attention." This restates the chapter-level framing from `index.md` without adding quantitative detail. The sentence can be trimmed without information loss.

### MINOR-3 (carry-forward): Navigation pointer in `function_signature.md` duplicates `index.md` contents table

`function_signature.md` line 151: "See `sdpa_program_config.md` for field details and a worked example." The `index.md` Chapter Contents table already provides this navigation. The pointer is low-value but useful for readers who jump directly to `function_signature.md`, so removal is optional.

### MINOR-4 (carry-forward): Master Shape Table shape columns duplicate function signature code comments

`tensor_shape_reference.md` lines 12–19 (Master Shape Table) lists Non-paged shape and Paged shape columns. All of those shapes are already annotated in the function signature code block at `function_signature.md` lines 11–26. The table's Layout, Dtype, and Notes columns are load-bearing and unique. Optional compression: remove the shape columns from the table and add a header note pointing to the function signature for shapes.

---

## Load-Bearing Evidence

The following facts must NOT be removed from their current location, as they are the sole authoritative statements of the information in the chapter:

1. **`tensor_shape_reference.md` lines 54–59** — The float32 bit-pattern example: `float32(2.0) → 0x40000000 = 1073741824`, turning "block index 2" into "block index 1073741824". This is the only concrete numerical illustration of the int32 silent-failure consequence in the chapter. The cross-reference in `function_signature.md` line 187 now depends on this content being present here.

2. **`tensor_shape_reference.md` lines 106–117** — The Cache Layout History section: the full `[nkv x b x s x dh]` → `[b x nkv x s x dh]` axis-swap explanation, the `nkv=4, b=8` worked example showing why no shape error is raised (product of axes is identical), and the diagnostic procedure (inspect axis 0). The cross-reference in `function_signature.md` line 71 now depends on this content being present here.

3. **`tensor_shape_reference.md` lines 77–98** — The `num_cores >= b * nkv` Constraint section: the one-core-per-pair assignment, the violation symptom (first `num_cores` pairs correct, remainder wrong/zero), and the fix. The cross-reference in `sdpa_program_config.md` line 46 now depends on this content being present here.

4. **`sdpa_program_config.md` lines 52–57** — The scenario table mapping `(b, nkv)` configurations to required core counts and example grids (Single-batch GQA through Batch-8 MHA). This is the only place in the chapter that gives concrete grid-sizing guidance across multiple configurations.

5. **`sdpa_program_config.md` lines 83–97** — The `k_chunk_size` field description: online softmax granularity, the numerical-stability trade-off at 32K+ tokens, and the `k_chunk_size % block_size == 0` paged-mode constraint with the illegal `k_chunk_size=48, block_size=32` example. This constraint and the illegal example appear nowhere else.

6. **`sdpa_program_config.md` lines 104–127** — The Parallelization Strategy section (steps 1–4 and the group_size implication): the only place in the chapter that explains why GQA decode can be slower than MHA decode (`group_size` multiplies per-core work), and that gives the per-core iteration count formula `group_size * s / k_chunk_size`.

7. **`function_signature.md` lines 107–116** — The `cur_pos` scalar-vs-list silent failure and the recompilation trade-off. The scalar failure (silent undefined masking for all but the first batch element) and the production-serving recommendation ("Prefer `cur_pos_tensor` for production serving") appear only here and are not restated in `tensor_shape_reference.md`.

8. **`tensor_shape_reference.md` lines 121–147** — The GQA Padding Gotcha: the three-step padding procedure preserving the GQA ratio and the `nkv=4, nh=16, group_size=4` worked numerical derivation. This section is the only place in the chapter that explains why independent padding of `nkv` and `nh` breaks GQA group semantics.

---

## VERDICT

Crucial updates: no
