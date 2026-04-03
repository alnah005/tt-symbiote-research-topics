# Compression Analysis: Chapter 5 --- Heterogeneous Attention Module Design

## Summary

Chapter 5 spans five files totaling approximately 820 lines. The content is well-structured and technically detailed, but there is meaningful redundancy in three categories: (1) pseudocode in design_options.md that is repeated nearly verbatim in the forward-pass files, (2) summary tables at the end of the forward-pass files that restate what the step-by-step prose already covered, and (3) the sliding-window strategy discussion that appears in both sliding_attention_forward.md (Steps 4-5) and paged_sdpa_sliding_window.md. No crucial compression is warranted --- the redundancy serves a navigational purpose (each file can stand alone), and removing it would break the self-contained reading experience of individual files.

## File-by-File Analysis

### 1. index.md

**Redundancy found:** The "Key Parameters Quick Reference" table (lines 78-95) repeats parameter values that appear in design_options.md's "Shared vs Divergent Logic Inventory" tables (lines 15-38) and throughout both forward-pass files. However, this table is the canonical quick-reference and is appropriately placed in the index.

**Load-Bearing Evidence:** The table at lines 78-95 is the only location that consolidates all key parameters (head_dim, num_kv_heads, weight shapes, RoPE type, window, K=V sharing, V-norm, K-norm, GQA ratio) into a single comparative view. The forward-pass files reference these values inline but never present them side-by-side. Removing this table would force readers to cross-reference two separate files to compare parameters.

**MINOR suggestion:** The "Central Question" section (lines 27-43) restates the three options already listed in the Overview (lines 17-20). The three numbered options and their class names could be removed from the Central Question section, leaving only the link to design_options.md. This would save approximately 12 lines.

---

### 2. design_options.md

**Redundancy found:** The pseudocode for Options B and C (lines 160-226 and 258-357) contains complete forward-pass implementations for both sliding and global attention subclasses. These implementations are then repeated with only minor differences in sliding_attention_forward.md (lines 319-354) and global_attention_forward.md (lines 399-433). The Option C `TTNNGemma4SlidingAttention._project_kv_and_rope` method (lines 318-326) is nearly identical to the same method in sliding_attention_forward.md (lines 322-338). The Option C `TTNNGemma4GlobalAttention._project_kv_and_rope` (lines 345-351) matches global_attention_forward.md (lines 402-417).

**Load-Bearing Evidence:** The "Comparative Summary" table (lines 391-401) is unique to this file and cannot be found elsewhere. It provides the only side-by-side comparison of all three design options across seven criteria. This table is the analytical core of the design recommendation.

**MINOR suggestion:** The Option B and Option C pseudocode blocks could be shortened to show only the structural skeleton (class names, method signatures, and brief inline comments indicating what each method does) rather than full forward-pass implementations. The reader is directed to the forward-pass files for complete implementations anyway. This could reduce the file by approximately 60-80 lines without losing the design comparison value.

---

### 3. sliding_attention_forward.md

**Redundancy found:** Two forms of internal redundancy:
- The "Complete Tensor Shape Trace" table (lines 300-315) restates every operation already described in Steps 1-6 with their shapes. Every row in that table has a corresponding code block earlier in the file that shows the same input/output shapes.
- The "TTNN Pseudocode (Sliding Subclass)" section (lines 317-355) repeats the Option C sliding subclass from design_options.md (lines 309-333) with only cosmetic differences (e.g., `self.k_proj_weight` vs `TTNNLinear`).

**Load-Bearing Evidence:** The "Window Enforcement at Cache Level" discussion (lines 194-216) presenting Strategy 1 vs Strategy 2 with a concrete recommendation for Gemma 4 31B is unique context that ties the general windowed-attention strategies to the specific sliding-layer requirements. This framing does not appear elsewhere in this form.

**MINOR suggestion:** The "Complete Tensor Shape Trace" table could be removed or converted to a one-line reference ("See the step-by-step sections above for all tensor shapes"). The TTNN pseudocode section could be replaced with a forward reference to design_options.md Option C. Combined savings: approximately 50 lines.

---

### 4. global_attention_forward.md

**Redundancy found:** Three forms:
- The "Complete Tensor Shape Trace" table (lines 339-353) duplicates the step-by-step shape information from Steps 1-7.
- The "TTNN Pseudocode (Global Subclass)" section (lines 397-433) repeats design_options.md's Option C global subclass (lines 336-357).
- The "Fused QKV Optimization When V Shares K" section (lines 355-395) partially restates the "Fused QK Alternative" already covered in Step 1 (lines 80-97). Specifically, the fused weight shape `[5376, 18432]`, the slice boundaries (`:16384` and `16384:`), and the Q/shared_kv recovery code appear in both places. The "Weight Construction for Fused QK" subsection (lines 377-395) adds new information about checkpoint loading, but the fused matmul mechanics are duplicated.

**Load-Bearing Evidence:** The "Key Implementation Considerations" section (lines 437-457) is unique. Point 1 (tensor aliasing verification for K=V sharing) and Point 3 (L1 pressure from head_dim=512) provide actionable implementation guidance not found in any other file.

**MINOR suggestion:** Merge the "Fused QKV Optimization When V Shares K" section into Step 1's "Fused QK Alternative" subsection, keeping only the new content (weight construction from checkpoint). Remove the "Complete Tensor Shape Trace" table and the "TTNN Pseudocode" section, replacing both with cross-references. Combined savings: approximately 60-70 lines.

---

### 5. paged_sdpa_sliding_window.md

**Redundancy found:** Two forms:
- The "Interaction With `cur_pos_tensor`" section (lines 89-109) restates the window boundary formula `[max(0, T - W + 1), T]` already given in "Window Masking Within the Kernel" (lines 39-53, specifically item 2). The formula appears at line 48 and again at lines 95-96.
- The overall investigation of native `sliding_window_size` support and the two fallback strategies overlaps with sliding_attention_forward.md's Step 4 "Window Enforcement at Cache Level" (lines 194-216) and Step 5's "`sliding_window_size` Behavior" (lines 251-264). The sliding file presents the same two strategies (let SDPA handle windowing vs circular-buffer-as-pages) and the same recommendation.

**Load-Bearing Evidence:** The "Page Loading Optimization" section (lines 56-87) with its quantitative table comparing optimistic vs conservative page loading at various sequence lengths (1K through 256K) is unique and provides concrete performance analysis not found in any other file. This is the analytical justification for the phased recommendation.

**MINOR suggestion:** The "Interaction With `cur_pos_tensor`" section (lines 89-109) could be folded into the "Window Masking Within the Kernel" section as a brief note about correctness requirements, eliminating the duplicate formula. Savings: approximately 15 lines.

---

## Cross-File Redundancy Summary

| Redundant Content | Appears In | Suggested Resolution |
|---|---|---|
| Option C sliding subclass pseudocode | design_options.md + sliding_attention_forward.md | Keep in forward file, shorten in design_options |
| Option C global subclass pseudocode | design_options.md + global_attention_forward.md | Keep in forward file, shorten in design_options |
| Tensor shape trace tables | sliding_attention_forward.md + global_attention_forward.md (internal duplication of step-by-step prose) | Remove tables, prose already covers shapes |
| Sliding window strategies (Strategy 1 vs 2) | sliding_attention_forward.md Step 4 + paged_sdpa_sliding_window.md | Keep detailed version in paged_sdpa file, shorten in sliding file to a forward reference |
| Window boundary formula | paged_sdpa_sliding_window.md (appears twice internally) | Merge into single section |
| Fused QK mechanics | global_attention_forward.md (appears twice internally) | Merge into Step 1 |

## Estimated Compression

If all MINOR suggestions were applied: approximately 200-220 lines could be removed from the current ~820 lines, a reduction of roughly 25%. The remaining ~600 lines would preserve all unique analytical content, all actionable recommendations, and the self-contained readability of each file (via cross-references replacing duplicated content).

## VERDICT

**Crucial updates: no.** The redundancy is real but serves a navigational purpose --- each file can currently be read standalone without consulting others. The duplicated pseudocode and shape tables provide convenient reference points within each file. Removing them would improve conciseness at the cost of requiring more cross-file navigation. The MINOR suggestions above identify specific, low-risk compression opportunities that maintain all load-bearing content.
