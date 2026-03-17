# Compression Analysis — Chapter 6: Debugging Incorrect Decode Output

## Crucial updates: yes

---

### Duplication 1 — `cur_pos` `-1` skip-behavior code block

**Source (canonical):** `ch4_cur_pos_semantics/cur_pos_definition.md`, "The `-1` Sentinel" section (lines 76–100). Contains a full code example with `cur_pos = [64, -1, 32, -1]`, the SDPA call, and explicit comments that indices 1 and 3 are undefined.

**Duplicate:** `ch6_debugging/cur_pos_validation.md`, Section 5 "Testing the `-1` Skip Behavior" (lines 87–99). Contains a near-verbatim code example with `cur_pos = [42, -1, 17]`, the same SDPA call pattern, and the same "do not use undefined slice" annotation.

**Why crucial:** Both blocks are multi-line executable code samples illustrating the same sentinel behavior. The ch6 version adds no new information beyond what ch4 already states; a reader who has followed the guide already encountered this in full.

**Recommended action for Agent A:** Replace the entire Section 5 code block in `cur_pos_validation.md` (lines 87–99) with a one-sentence cross-reference: "For the `-1` skip sentinel and safe downstream handling, see `ch4_cur_pos_semantics/cur_pos_definition.md`." Keep the one-line summary row in the Validation Summary table (line 109) but remove the preceding prose and code.

---

### Duplication 2 — GQA ratio preservation assert block

**Source (canonical):** `ch3_gqa_tensor_layout/gqa_grouping_in_kernel.md`, "Correct Padding Strategy" section (lines 74–87). Contains the full multi-line code block:
```python
original_group_size = nh // nkv
pnh = ((nh + 31) // 32) * 32
nkv_padded = pnh // original_group_size
effective_group_size = pnh // nkv_padded
```
plus the three-row comparison table at lines 89–95.

**Duplicate:** `ch6_debugging/shape_validation_checklist.md`, Section 5 "GQA Ratio Preservation After Padding" (lines 54–73). Contains near-verbatim assert statements and the same `nkv_padded = nh_padded // original_group_size` formula, repeating the mechanism explained fully in ch3.

**Secondary overlap:** `ch2_ttnn_api/tensor_shape_reference.md` "GQA Padding Gotcha" (lines 121–147) also contains the same padding procedure with a worked example for `nkv=4, nh=16`.

**Why crucial:** The multi-line assert block and the formula derivation are already presented in full in both ch2 and ch3. The ch6 copy reproduces the same executable code without adding debugging-specific context that could not be provided by a cross-reference.

**Recommended action for Agent A:** In `shape_validation_checklist.md` Section 5, remove the two `assert` code blocks (lines 58–63 and 65–73) and the `nkv_padded` formula line. Replace with: "Verify both assertions hold; the derivation and worked example are in `ch3_gqa_tensor_layout/gqa_grouping_in_kernel.md`." Keep the summary table rows (lines 112–113) and the surrounding prose warning, as those are useful quick-checks.

---

### Duplication 3 — `cur_pos` scalar-vs-list (batch consistency) code block

**Source (canonical):** `ch4_cur_pos_semantics/per_user_vs_shared.md`, "Common Mistake 1 — Scalar Instead of a Length-`b` List" (lines 22–33). Multi-line code example showing `cur_pos = seq_len` (wrong) vs `cur_pos = [seq_len] * b` (correct), with explanation.

**Duplicate:** `ch6_debugging/cur_pos_validation.md`, Section 3 "Batch Consistency" (lines 50–61). Contains a structurally identical code block showing `cur_pos = 42` (wrong) vs `cur_pos = [42]` (correct), with the same broadcast-risk explanation.

**Why crucial:** Both are multi-line instructional code blocks making the identical point. The ch6 version uses `b=1` as a concrete case but the ch4 version already covers this as a general rule with comparable code.

**Recommended action for Agent A:** Collapse Section 3 in `cur_pos_validation.md` to two sentences referencing ch4: "For `b=1`, `cur_pos` must be `[42]`, not `42`. For `b > 1` with divergent lengths, see `ch4_cur_pos_semantics/per_user_vs_shared.md` for the full scalar-vs-list and shared-value mistake patterns." Remove the code block entirely.

---

### Duplication 4 — `cur_pos_tensor` vs `cur_pos` list code block in root cause isolation

**Source (canonical):** `ch4_cur_pos_semantics/cur_pos_definition.md`, "Two Passing Modes" section (lines 27–73). Contains two full code blocks — Mode 1 (Python list triggering recompilation) and Mode 2 (device tensor avoiding recompilation) — with explanatory prose.

**Duplicate:** `ch6_debugging/root_cause_isolation.md`, Step 3 "cur_pos_tensor vs cur_pos List" (lines 83–101). Contains a near-verbatim pair of code blocks showing list form (`cur_pos=[42, 17]`) and tensor form (`ttnn.Tensor([42, 17], ...)`), plus the `device.clear_compiled_program_cache()` call.

**Why crucial:** The two-form comparison is already a full multi-line code block in ch4. The ch6 copy reproduces it with minor renaming. The only ch6-specific addition is the `clear_compiled_program_cache()` call, which could be mentioned in one sentence.

**Recommended action for Agent A:** In `root_cause_isolation.md` Step 3, remove the two code blocks (lines 88–94 and 96–100). Replace with: "If one form succeeds and the other fails, clear the op program cache with `device.clear_compiled_program_cache()` and rerun. For the full list-vs-tensor passing modes and recompilation trade-offs, see `ch4_cur_pos_semantics/cur_pos_definition.md`."

---

## Summary of edits for Agent A

| File | Lines to remove/replace | Replacement |
|------|------------------------|-------------|
| `ch6_debugging/cur_pos_validation.md` | Section 5 code block (lines 87–99) + preceding prose | One-sentence cross-reference to `ch4_cur_pos_semantics/cur_pos_definition.md` |
| `ch6_debugging/shape_validation_checklist.md` | Section 5 two assert code blocks (lines 58–63, 65–73) + formula line | One-sentence cross-reference to `ch3_gqa_tensor_layout/gqa_grouping_in_kernel.md` |
| `ch6_debugging/cur_pos_validation.md` | Section 3 code block (lines 52–57) | Two-sentence summary with cross-reference to `ch4_cur_pos_semantics/per_user_vs_shared.md` |
| `ch6_debugging/root_cause_isolation.md` | Step 3 two code blocks (lines 88–94, 96–100) | One-sentence mention of `clear_compiled_program_cache()` + cross-reference to `ch4_cur_pos_semantics/cur_pos_definition.md` |

## Agent A Change Log — C Feedback Pass 1
- cur_pos_validation.md: Removed -1 sentinel code block (Section 5); replaced with cross-reference to Chapter 4, cur_pos_definition.md
- shape_validation_checklist.md: Removed GQA ratio assert code blocks (Section 5); replaced with cross-reference to Chapter 3, gqa_grouping_in_kernel.md
- cur_pos_validation.md: Collapsed scalar-vs-list code block (Section 3); replaced with prose + cross-reference to Chapter 4, per_user_vs_shared.md
- root_cause_isolation.md: Removed cur_pos format code blocks (Step 3); kept cache-clear mention; added cross-reference to Chapter 4, cur_pos_definition.md

---

## Pass 2 Verification

**Fix 1 — `cur_pos_validation.md` Section 5 (`-1` sentinel code block):** Applied. The multi-line code block and preceding prose were removed. Section 5 now contains a single prose sentence with a cross-reference to `ch4_cur_pos_semantics/cur_pos_definition.md`. The Validation Summary table row (row 5) was retained. Fix confirmed.

**Fix 2 — `shape_validation_checklist.md` Section 5 (GQA ratio assert blocks):** Applied. The two `assert` code blocks and the `nkv_padded` formula derivation were removed. Section 5 now contains two prose sentences stating the two conditions plus a cross-reference to `ch3_gqa_tensor_layout/gqa_grouping_in_kernel.md`. The summary table rows (rows 5 and 6) were retained. Fix confirmed.

**Fix 3 — `cur_pos_validation.md` Section 3 (scalar-vs-list code block):** Applied. The multi-line code block was removed. Section 3 is now a single prose sentence covering the `b=1` case and directing to `ch4_cur_pos_semantics/per_user_vs_shared.md` for the general pattern. Fix confirmed.

**Fix 4 — `root_cause_isolation.md` Step 3 (cur_pos format code blocks):** Applied. The two code blocks were removed. Step 3 is now a single prose sentence that mentions `device.clear_compiled_program_cache()` inline and adds a cross-reference to `ch4_cur_pos_semantics/cur_pos_definition.md`. Fix confirmed.

## Crucial updates: no

No further duplications found. The remaining executable code blocks in ch6 are all ch6-specific: the off-by-one increment assertion in `cur_pos_validation.md` Section 2, the paged block count check in Section 4, the disable-paging and disable-GQA isolation blocks in `root_cause_isolation.md` Steps 1–2, the binary-search-over-`cur_pos` workflow in `pcc_comparison_workflow.md` Section 4, and the KV-write readback check in Section 5. None of these patterns appear in ch3, ch4, or elsewhere in ch6 with sufficient similarity to constitute a crucial duplication.
