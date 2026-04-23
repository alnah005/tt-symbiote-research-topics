# Compression Analysis: Chapter 2 — TTNN Decomposition — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~324 lines (182 + 145 + 122, minus trailing blank lines)
- Estimated post-compression line count: ~275 lines
- Estimated reduction: ~15%

---

## CRUCIAL Suggestions

### 1. Tile alignment section duplicated across two files
**Location:** `recurrence_math_and_tensor_ops.md` lines 159–168 ("Tile Alignment" section) AND `state_tensor_memory_config.md` lines 93–103 ("Tile Alignment Analysis" section)

**Issue:** Both sections address the same question (are d_k=128 and d_v=128 multiples of 32, and does the state matrix require padding?) and reach the same conclusion (no padding required, 4×4=16 tiles, favorable case). The recurrence file presents this as a prose paragraph; the state file presents it as a table. No new facts appear in the second occurrence. A reader of both files encounters the identical analysis twice.

**Concrete suggestion:** Remove the "Tile Alignment" section from `recurrence_math_and_tensor_ops.md` (lines 159–168) entirely. Replace it with a single sentence such as: "Tile alignment for all six operations is analyzed in [state_tensor_memory_config.md § Tile Alignment Analysis](state_tensor_memory_config.md)." The state file is the appropriate home for this content because it already covers layout, sizing, and alignment for all 12 ops.

---

### 2. Six core recurrence TTNN calls reproduced verbatim in both files
**Location:** `recurrence_math_and_tensor_ops.md` — six separate code blocks in Ops 1–6 (lines 51, 71, 92–95, 114, 132, 152) AND `ttnn_ops_per_step.md` — steps 6–11 in the "Annotated Code" block (lines 65–96)

**Issue:** The exact six TTNN API calls (`ttnn.mul(g_broadcast, S_prev)`, `ttnn.matmul(S_prev, k_tilde, transpose_a=True)`, `ttnn.sub` + `ttnn.mul` for error, `ttnn.matmul` outer product, `ttnn.add`, `ttnn.matmul` for output) appear in both files. The math file presents each call in isolation with a shape table and a correctness rationale. The ops file re-presents all six calls inside a unified 12-op listing that additionally covers ops 1–5 (QKV split, reshape, g_t, beta_t, L2-norm) and op 12 (flatten), plus memory config and availability columns. The six shared calls are character-for-character identical or differ only in variable names (`v_t` vs `v_heads`), which adds confusion rather than information.

**Concrete suggestion:** In `ttnn_ops_per_step.md`, replace the inline code for steps 6–11 in the annotated code block with a comment directing the reader to the derivation file, e.g.:

```python
# Steps 6–11: Recurrence ops (decay, retrieval, error, write, new state, output readout)
# Full derivation, shape tables, and correctness notes:
#   see recurrence_math_and_tensor_ops.md §§ Operations 1–6
# Memory config for each step: see op table above and Notes on Memory Config below.
S_decayed = ttnn.mul(g_broadcast, S_prev)          # Op 6 — DRAM→L1
retrieval  = ttnn.matmul(S_prev, k_tilde, transpose_a=True)  # Op 7
# ... (abbreviated; see derivation file for full forms)
```

Alternatively, keep the code in `ttnn_ops_per_step.md` as the single canonical location and replace the six isolated code blocks in `recurrence_math_and_tensor_ops.md` with pseudocode or math-only notation, retaining only the shape tables and correctness notes (which are load-bearing — see below).

---

## MINOR Suggestions

### 1. Model dimensions restated in prose in ttnn_ops_per_step.md
**Location:** `recurrence_math_and_tensor_ops.md` lines 5–13 (dimensions table) AND `ttnn_ops_per_step.md` lines 4–5 (prose sentence)

**Issue:** The concrete values B=1, nH=32, d_k=128, d_v=128 are defined in a table in the math file and then restated in a prose sentence at the top of the ops file. This is a minor inline restatement, not a full duplicate section.

**Concrete suggestion:** In `ttnn_ops_per_step.md`, replace the prose restatement with a forward reference: "All shapes use the concrete Qwen3.6-35B-A3B dimensions defined in [recurrence_math_and_tensor_ops.md § Model dimensions](recurrence_math_and_tensor_ops.md)." Retain `nH_local` definition (32/8=4) since it is specific to the ops file's T3K sharding context.

### 2. DRAM persistence rationale partially repeated in memory config notes
**Location:** `state_tensor_memory_config.md` lines 66–81 ("Why DRAM, Not L1" section) AND `ttnn_ops_per_step.md` lines 103–113 ("Notes on Memory Config" section)

**Issue:** The ops file's memory config notes explain that S lives in DRAM between steps and is read into L1 for computation, which overlaps with the dedicated "Why DRAM, Not L1" section in the state file. The ops file already ends with a cross-reference link to `state_tensor_memory_config.md`, which partially manages this. The remaining overlap (the DRAM→L1→DRAM round-trip description in the ops file notes) is a minor repetition because the ops file's version is shorter and scoped to the shorthand legend.

**Concrete suggestion:** In `ttnn_ops_per_step.md` Notes on Memory Config, trim the DRAM→L1 and L1→DRAM bullet descriptions to one line each (e.g., "state read from DRAM into L1 for this step — see state_tensor_memory_config.md for persistence rationale") rather than re-explaining why DRAM is required.

### 3. "S_prev not S_decayed" rationale stated twice at different lengths
**Location:** `recurrence_math_and_tensor_ops.md` lines 74 (full Note under Op 2, ~5 sentences) AND `ttnn_ops_per_step.md` line 75 (`# why S_prev not S_decayed:` comment, 1 sentence)

**Issue:** The single-sentence comment in the ops file is a reasonable abbreviation and already points implicitly to the longer explanation. This is borderline minor — it does not meet the crucial bar because the two occurrences are at very different levels of detail and serve different reader contexts (derivation vs. implementation checklist).

**Concrete suggestion:** In the ops file comment, append an explicit pointer: "# why S_prev not S_decayed: delta rule requires pre-decay state — see recurrence_math_and_tensor_ops.md § Operation 2 for full rationale."

---

## Load-Bearing Evidence

1. **`recurrence_math_and_tensor_ops.md` lines 56–74 (Op 2 — Retrieval, full Note):** The five-sentence note explaining why retrieval must use `S_{t-1}` and not `S_decayed`, including the consequence of using `S_decayed` (incorrect error signal) and the independence of ops 1 and 2. This is the only place in the chapter where this correctness constraint is fully argued. Must not be cut.

2. **`recurrence_math_and_tensor_ops.md` lines 141–155 (Op 6 — Output, full Note):** The explanation of why both retrieval and output use `S^T` rather than `S`, including the silent-error risk that arises when `d_k = d_v = 128` (TTNN will not raise a shape error but will produce numerically wrong values). This is the only place this silent-failure risk is identified. Must not be cut.

3. **`recurrence_math_and_tensor_ops.md` lines 172–182 (Summary Table):** The consolidated op-to-primitive mapping table covering all six recurrence ops with output shapes. Provides a single-glance reference that does not appear in the same form anywhere else. Must not be cut.

4. **`state_tensor_memory_config.md` lines 20–43 (Head-Parallel Sharding and DRAM totals):** The per-device sharding calculation (32/8=4 heads, 128 KB per device per layer, 3.75 MB total) and the correction of the "3.84 MB" decimal approximation error. This is the only place these numbers are derived from first principles. Must not be cut.

5. **`state_tensor_memory_config.md` lines 85–91 (L1 Feasibility During Kernel Execution):** The peak working-set calculation (65.5 KB), the identification of the two peak moments (ops 6–7 and ops 9–10), and the proof that S_prev and write are never simultaneously in L1. This analysis is unique to this file and not summarized anywhere else. Must not be cut.

6. **`state_tensor_memory_config.md` lines 105–130 (Conv State section):** Shape, size, and initialization of the conv state tensor, including the `ROW_MAJOR_LAYOUT` requirement (last dim = 4, not tile-aligned). This content appears nowhere else in the chapter. Must not be cut.

7. **`ttnn_ops_per_step.md` lines 9–22 (Op Table):** The 12-op table with availability status, memory config column, and the ops 1–5 and 12 that do not appear in `recurrence_math_and_tensor_ops.md`. This is the only place ops 1–5 (QKV split, reshape, g_t, beta_t, L2-norm) and op 12 (flatten) are specified. Must not be cut.

8. **`ttnn_ops_per_step.md` lines 38–44 (GQA repeat TODO in annotated code):** The note that `ttnn.repeat` for grouped-query attention expansion may not be needed if downstream matmuls handle GQA broadcasting internally, flagged as a TODO for verification. This open question is unique to this file. Must not be cut.

---

## VERDICT
- Crucial updates: yes

---

# Compression Analysis: Chapter 2 — TTNN Decomposition — Pass 1 (Change Log)

Changes applied in response to Pass 1 CRUCIAL suggestions:
1. `recurrence_math_and_tensor_ops.md` Tile Alignment section: removed duplicate analysis; replaced with cross-reference to `state_tensor_memory_config.md § Tile Alignment Analysis`
2. `recurrence_math_and_tensor_ops.md` Ops 1–6 code blocks: removed 6 isolated Python code blocks (duplicates of the annotated code in `ttnn_ops_per_step.md`); replaced each with an inline cross-reference sentence naming the TTNN call and pointing to `ttnn_ops_per_step.md`

---

# Compression Analysis: Chapter 2 — TTNN Decomposition — Pass 2

## CRUCIAL fixes verification

1. **Fix 1 — Tile Alignment section in `recurrence_math_and_tensor_ops.md`:** Applied correctly. The duplicate prose analysis (d_k=128, d_v=128, 4×4=16 tiles) is gone. Lines 138–141 contain exactly one cross-reference sentence: "Tile alignment for all six operations is analyzed in [`state_tensor_memory_config.md` — Tile Alignment Analysis](./state_tensor_memory_config.md)." The canonical analysis remains intact in `state_tensor_memory_config.md` lines 93–103.

2. **Fix 2 — Six Python code blocks in `recurrence_math_and_tensor_ops.md`:** Applied correctly. Ops 1–6 no longer have standalone Python code blocks. Each operation section contains an inline cross-reference sentence naming the TTNN call and pointing to `ttnn_ops_per_step.md` (e.g., Op 1: "TTNN call: `ttnn.mul(g_broadcast, S_prev)` — see `ttnn_ops_per_step.md`…"). All shape tables are present for every op. The Op 2 Note about S_prev (pre-decay state requirement and silent-error consequence) is intact. The Op 6 Note about the silent error risk when d_k = d_v = 128 is intact. The Summary Table covering all six ops with TTNN primitives and output shapes is present at lines 144–154.

## Remaining CRUCIAL issues

None found. The two canonical locations now hold their respective content without cross-file duplication of full concepts:
- `recurrence_math_and_tensor_ops.md` owns the math derivation, shape tables, correctness notes, and summary table; all TTNN code is by reference only.
- `ttnn_ops_per_step.md` owns the full annotated code sequence (all 12 ops) and the op table with memory config and availability columns.
- `state_tensor_memory_config.md` owns tile alignment, DRAM sizing, sharding calculations, L1 feasibility, and conv state details.

## VERDICT
- Crucial updates: no
