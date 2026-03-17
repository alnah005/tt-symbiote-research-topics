# Compression Analysis — Chapter 5: Sparsity Tensor Construction — Pass 1

## Summary
- Files reviewed: `index.md`, `sparsity_tensor_format.md`, `constructing_from_router_output.md`, `sparsity_tensor_placement.md`, `common_pitfalls.md`
- Current line count: ~1107 lines (index.md: 85, sparsity_tensor_format.md: 154, constructing_from_router_output.md: 353, sparsity_tensor_placement.md: 204, common_pitfalls.md: 311)
- Estimated post-compression: ~1040 lines (~6% reduction)

---

## CRUCIAL Suggestions

**C1 — Duplicate mask size calculations (decode 7 KB / prefill 56 KB)**

The tile count and byte size for decode and prefill masks are computed identically in two files:

- `sparsity_tensor_format.md`, Section 4 (lines 70–76): Full tile-count formula with exact arithmetic for both decode and prefill, concluding 7 tiles = 7168 bytes and 56 tiles = 57344 bytes.
- `constructing_from_router_output.md`, Step 5 comment block (lines 104, 216–217): Repeats the same conclusions inline — `7 tiles × 1024 bytes = 7168 bytes` (line 104) and `# Decode size: 7 tiles × 1024 bytes = 7168 bytes in L1 / # Prefill size: 56 tiles × 1024 bytes = 57344 bytes in L1` (lines 216–217).
- `sparsity_tensor_placement.md`, Section 2 table (lines 30–36): Re-derives the same calculation a third time, showing `1 × 7 = 7` tiles and `7168 bytes ≈ 7 KB` and `56 tiles = 57344 bytes ≈ 56 KB`.

All three instances arrive at identical numbers. The full derivation belongs once in `sparsity_tensor_format.md` Section 4 (authoritative format spec). The placement table in `sparsity_tensor_placement.md` Section 2 is load-bearing because it covers a third regime (B=32, S=1 decode) not shown elsewhere, but the arithmetic for the decode and prefill entries can simply cite `sparsity_tensor_format.md` rather than re-deriving. The inline comments in `constructing_from_router_output.md` lines 216–217 are pure duplicates of the `sparsity_tensor_format.md` results and can be removed without information loss.

**Remove:** `constructing_from_router_output.md` lines 216–217 (the two comment lines `# Decode size: 7 tiles × 1024 bytes = 7168 bytes in L1` and `# Prefill size: 56 tiles × 1024 bytes = 57344 bytes in L1` inside the code block). The surrounding code is load-bearing; only these two repetitive comment lines are redundant.

**C2 — L1 MEMORY CONFIG code snippet duplicated verbatim in three files**

The `ttnn.from_torch(...)` call with `dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG` appears four times as a standalone, identically structured snippet:

- `constructing_from_router_output.md`, Step 5 (lines 95–101): Prose-explained standalone snippet.
- `constructing_from_router_output.md`, inside `build_sparsity_tensor` function (lines 208–214): Inside the complete worked example — load-bearing (part of executable code).
- `sparsity_tensor_placement.md`, Section 3 (lines 43–49): Presented as the canonical L1 placement snippet.
- `common_pitfalls.md`, P4 Fix (lines 175–181): Presented as the corrected form after DRAM regression.

The Step 5 standalone snippet in `constructing_from_router_output.md` (lines 95–101) and the P4 Fix snippet in `common_pitfalls.md` (lines 175–181) are exact duplicates of the snippet in `sparsity_tensor_placement.md` Section 3. The canonical location for "how to place the mask in L1" is `sparsity_tensor_placement.md` Section 3. The Step 5 prose context in `constructing_from_router_output.md` is useful, but the 7-line fenced code block there can be reduced to a 1–2 line inline reference, with a pointer to `sparsity_tensor_placement.md` Section 3 for the full snippet. The P4 fix snippet in `common_pitfalls.md` adds a meaningful inline comment (`# not ttnn.DRAM_MEMORY_CONFIG`) that distinguishes it from the others; it should be kept as-is.

**Remove:** `constructing_from_router_output.md` lines 94–102 (the fenced `mask_ttnn = ttnn.from_torch(...)` snippet in Step 5). Replace with a one-sentence inline reference: "Transfer using `ttnn.from_torch` with `dtype=ttnn.uint8`, `layout=ttnn.TILE_LAYOUT`, and `memory_config=ttnn.L1_MEMORY_CONFIG` (see `sparsity_tensor_placement.md`, Section 3 for the full snippet and placement rationale)." The Tip that follows (lines 106–107) is load-bearing and must be kept.

---

## Load-Bearing Evidence

- **`index.md`**: The Chapter Notation table (lines 63–74) with Qwen3.5-35B reference values for all symbols ($M_t$, $K_t$, $E$, $E_d$, $k$, $B$, $S$, $\rho$, $C$) and the sparsity ratio tip ($\rho \approx 0.031$, 97% compute avoided). The Quick-Reference Checklist table (lines 32–39) mapping each check to its pitfall. The Warning on items 2 and 3 being silent correctness bugs (line 41).

- **`sparsity_tensor_format.md`**: Section 1 shape formula $M_t = \lceil C/32 \rceil$, $K_t = \lceil H/32 \rceil$ with the batched multi-expert mask shape $[E_d \times M_t, K_t]$; Section 2 value encoding table (0=skip, 1=compute) and the K-dimension broadcast rule; Section 4 tile size derivation (the authoritative 7 KB / 56 KB result); Section 7 partial-tile rule with the correctness-critical formula for the partial boundary and the Warning for B=1, S=1 decode; Section 8 Option A vs. Option B comparison; Section 9 static shape requirement and the fix strategy. All Warnings in this file are correctness-critical.

- **`constructing_from_router_output.md`**: The complete `build_sparsity_tensor` function (lines 133–219) including the scatter-based assignment loop with capacity overflow handling — this is the only full worked implementation in the chapter. The vectorized `build_sparsity_tensor_vectorized` function (lines 251–315) using `scatter_add_` — functionally distinct from the loop version. Both Shape Trace tables (decode B=1,S=1 and prefill B=4,S=2048, lines 319–343) showing every intermediate tensor shape. The Step 4 Warning about partial tile rows (line 86).

- **`sparsity_tensor_placement.md`**: Section 2 mask size table (the three-regime table including the B=32 decode row not present elsewhere); Section 5 sharding explanation for multi-device expert parallelism; Section 6 stale-mask lifetime rule including the "static prompts" exception paragraph; Section 7 TTNN Trace integration pattern with the correct pre-allocate-then-copy_ pattern and the WRONG anti-pattern (lines 175–179); Section 8 placement recommendations table. The Warning about creating tensors inside a `ttnn.Trace` context (line 181).

- **`common_pitfalls.md`**: All six pitfall entries (P1–P6) in full: each symptom description, root cause, detection code snippet, and fix snippet. The P1 `assert_mask_shape` utility function; P2 `validate_mask_vs_assignment` utility function; P3 step-counter detection loop; P4 `assert_mask_in_l1` utility and the profiler note about 7168 bytes accumulating over steps; P5 dtype error message verbatim; P6 `CANONICAL_CONFIGS` table and `get_canonical_mask_rows` function. The Summary Table (lines 294–301).

---

## MINOR Suggestions

**M1 — Redundant cross-file pointer sentences in every References section**

Each file ends with a References section that lists the other four files (or a subset). These sections are structurally useful as navigation aids, but several individual entries restate content that the file itself already explained inline. For example, `sparsity_tensor_format.md` References (lines 147–153) lists `common_pitfalls.md — P1 (shape), P2 (partial tiles), P4 (DRAM), P5 (dtype), P6 (recompilation)` — the same pitfall tags that already appear inline as cross-references within the file's own sections. No information is lost if the References sections are trimmed to list file paths only (without parenthetical content summaries), since the parenthetical descriptions duplicate what is already said inline.

**M2 — Repeated "see other file" inline callouts that duplicate the References section**

Across all five files there are 14+ inline `See <filename>, SectionN` callouts that also appear in the same file's References section. Examples: `sparsity_tensor_format.md` Section 1 Warning ends with "See `common_pitfalls.md`, P1" (line 26), and P1 also appears in that file's References. `constructing_from_router_output.md` Step 4 Warning ends with "See `sparsity_tensor_format.md`, Section 7, and `common_pitfalls.md`, P2" (line 86), and both appear in its References. The inline callouts are load-bearing for navigation while reading; the References section duplicates them at the end. If the References sections are kept, the duplication is minor but measurable. If References sections are simplified per M1, this redundancy resolves automatically.

**M3 — Verbose preamble sentences in `constructing_from_router_output.md` and `sparsity_tensor_placement.md`**

Both files open with two-sentence orientation paragraphs pointing to other files for the format contract:
- `constructing_from_router_output.md` lines 3–7: "This file walks through... For the format contract... see `sparsity_tensor_format.md`."
- `sparsity_tensor_placement.md` lines 3–5: "This file covers... For the format of the mask itself... For how to construct the mask..."

These are stylistic navigation aids. They are not wrong, but they replicate the Reading Order section in `index.md` (lines 46–55) which already explains what each file covers and its dependencies. A single sentence per file (rather than two) would suffice.

**M4 — "L1 capacity (1.5 MB per Tensix core on Wormhole B0)" stated four times**

This specific figure appears in: `sparsity_tensor_format.md` Section 5 (line 86), `constructing_from_router_output.md` Step 5 (line 104), `sparsity_tensor_placement.md` Section 1 (line 13) and Section 3 (line 54), and `sparsity_tensor_placement.md` References (line 202). The authoritative location is `sparsity_tensor_placement.md` Section 1 (L1 vs. DRAM Background). Occurrences in other files' References sections and as asides in `constructing_from_router_output.md` could be shortened to "within L1 capacity" without the numeric value, pointing to `sparsity_tensor_placement.md` for the exact figure.

---

VERDICT: Crucial updates: yes

---

# Compression Analysis — Chapter 5: Sparsity Tensor Construction — Pass 2

## Summary

After the C1 and C2 fixes from Pass 1, the chapter is in a clean state with no remaining duplicate derivations or verbatim repeated standalone code blocks. C1 removed the two redundant mask-size comment lines inside `build_sparsity_tensor`. C2 replaced the standalone `ttnn.from_torch(...)` snippet in Step 5 of `constructing_from_router_output.md` with a single inline reference sentence pointing to `sparsity_tensor_placement.md`, Section 3. The four Pass 1 MINOR suggestions (M1–M4) remain unaddressed and are carried forward below. No new crucial items were found.

---

## CRUCIAL Suggestions

None.

---

## Load-Bearing Evidence

- **`index.md`**: Chapter Notation table with Qwen3.5-35B reference values for all symbols and the $\rho \approx 0.031$ sparsity ratio tip. Quick-Reference Checklist table mapping each pre-call check to its pitfall. Warning that items 2 and 3 are silent correctness bugs.

- **`sparsity_tensor_format.md`**: Section 1 shape formula and batched multi-expert mask shape $[E_d \times M_t, K_t]$. Section 2 value encoding table and K-dimension broadcast rule. Section 4 tile-size derivation (authoritative 7 KB / 56 KB result). Section 7 partial-tile boundary formula and the B=1, S=1 correctness Warning. Section 8 Option A vs. Option B comparison. Section 9 static shape requirement and canonical-padding fix strategy.

- **`constructing_from_router_output.md`**: Complete `build_sparsity_tensor` function with scatter-based assignment loop and capacity overflow handling. Vectorized `build_sparsity_tensor_vectorized` using `scatter_add_`. Both Shape Trace tables (decode and prefill). Step 4 Warning about partial tile rows.

- **`sparsity_tensor_placement.md`**: Section 2 three-regime mask size table (including the B=32 decode row absent from other files). Section 3 canonical L1 placement snippet. Section 4 DRAM placement snippet (distinct from L1: used as the fallback pattern). Section 5 multi-device expert parallelism sharding explanation. Section 6 stale-mask lifetime rule and the static-prompts exception. Section 7 TTNN Trace integration with the correct pre-allocate-then-`copy_` pattern and the explicit WRONG anti-pattern. Section 8 placement recommendations table.

- **`common_pitfalls.md`**: All six pitfall entries (P1–P6) in full: symptom, root cause, detection snippet, and fix snippet. Utility functions `assert_mask_shape`, `validate_mask_vs_assignment`, `assert_mask_in_l1`. P5 dtype error message verbatim. P6 `CANONICAL_CONFIGS` table and `get_canonical_mask_rows` function. Summary Table.

---

## MINOR Suggestions

**M1 — References section parenthetical summaries duplicate inline cross-references (carry-forward from Pass 1)**

Each file's References section lists sibling files with parenthetical content summaries (e.g., `common_pitfalls.md — P1 (shape), P2 (partial tiles), P4 (DRAM), P5 (dtype), P6 (recompilation)` in `sparsity_tensor_format.md`, lines 147–153). These parenthetical descriptions restate what the inline `See <file>, P<N>` callouts already communicate. Trimming References entries to file path only (without parentheticals) would remove the duplication without losing navigation value.

**M2 — Inline "see other file" callouts and References sections double-list the same pointers (carry-forward from Pass 1)**

Across all five files there are 14+ inline `See <filename>, SectionN` callouts that also appear verbatim in the same file's References section. The inline callouts are load-bearing for in-context navigation; the References section then restates them. If References sections are simplified per M1, this redundancy resolves automatically with no separate action needed.

**M3 — Two-sentence orientation preambles in `constructing_from_router_output.md` and `sparsity_tensor_placement.md` replicate `index.md` Reading Order (carry-forward from Pass 1)**

`constructing_from_router_output.md` lines 3–7 and `sparsity_tensor_placement.md` lines 3–5 each contain a two-sentence paragraph pointing to other files for the format contract. The `index.md` Reading Order section (lines 46–55) already describes each file's scope and dependencies. One sentence per file preamble would suffice.

**M4 — "1.5 MB per Tensix core on Wormhole B0" stated four times across five files (carry-forward from Pass 1)**

This figure appears in `sparsity_tensor_format.md` Section 5 (line 86), `constructing_from_router_output.md` Step 5 (line 94), `sparsity_tensor_placement.md` Sections 1 and 3 (lines 13, 54), and `sparsity_tensor_placement.md` References (line 202). The authoritative location is `sparsity_tensor_placement.md` Section 1. Occurrences in other files could be shortened to "within L1 capacity" with a pointer to `sparsity_tensor_placement.md` for the exact figure, eliminating three restatements of the same hardware spec.

---

VERDICT: Crucial updates: no
