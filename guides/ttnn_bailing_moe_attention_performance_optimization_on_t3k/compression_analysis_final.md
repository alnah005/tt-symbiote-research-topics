# Compression Analysis — Final Pass 7

## Verdict: Crucial updates: no

## Load-Bearing Evidence

- **index.md — "Research Questions Answered" table:** Primary Q# → Chapter navigation; cross-referenced by every chapter's Scope section. Load-bearing.
- **index.md — "Chapter Overview" table:** Directory names not repeated elsewhere; required for filesystem navigation. Load-bearing.
- **index.md — "How to Use This Guide" section:** Two distinct reader paths (sequential vs. profiling-first). Not duplicated anywhere else. Load-bearing.
- **index.md — Q7 table cell full three-clause question text:** Verbatim research question that scopes Chapter 6. Trimming it would make the table row inconsistent with Chapter 6's Scope section. Load-bearing.
- **Ch2/index.md — Key Symbols table `C` entry:** `C = (Nq + 2·Nkv)·D = 3072` is the canonical definition used without re-derivation in Ch3, Ch4, and Ch5. Load-bearing.
- **Ch3/index.md — "Why the Round-Trip Exists" section:** Contains the mechanistic explanation of the TTNN type-system constraint (no in-place reinterpret-distribution primitive). Not stated in any content file index. Load-bearing.
- **Ch4/index.md — 9-transition catalog table plus non-additivity note:** Symbol definitions (T1a–T4, T_norm_in_Q, T_norm_in_K) are the canonical names referenced in Ch5, Ch6, and Ch7; the note disambiguates the 83–86 µs total from the 64 µs eliminable subset. Both are load-bearing.
- **Ch5/index.md — `WormholeComputeKernelConfig` code block:** Only index-level location where the three config fields appear verbatim. Load-bearing reference anchor.
- **Ch6/index.md — Ling Configuration Recap table "Consequence" column:** Consequence entries (e.g., "forces TTNNRotaryPositionEmbedding (non-distributed)") are not stated in Ch1 and are load-bearing for Ch6 content.
- **Ch6/index.md — T_norm_in/T_norm_out symbol table entries with µs estimates:** Cross-chapter calibration values referenced in Ch4 and Ch7.
- **Ch7/index.md — "Profiling Tools at a Glance" comparison table:** Sole index-level comparison of TTNN op timers vs. Tracy. Load-bearing.
- **Ch7/index.md — non-additivity clarification in QK norm prerequisites bullet:** Sole location warning that Ch4's 83–86 µs and Ch6's 74–92 µs share a ~32 µs component and must not be added. Load-bearing (Pass 6 CU-7 proposed collapsing this to a pointer sentence, but the arithmetic detail is necessary for correct profiling interpretation).
- **All chapter indexes — "Start reading:" footer links:** Load-bearing navigation.

## CRUCIAL Updates

(none)

## MINOR Suggestions

1. **index.md — remove the preamble sentence before the Chapter Overview table (line 41).** "Seven chapters cover the full optimization surface from hardware context through profiling methodology (note: Chapter 5 answers two research questions — Q4 and Q5 — within a single chapter)." The chapter count is visible from the table; the Q4/Q5 dual-answer is already shown in the Research Questions table above it. The entire sentence is removable without any information loss.

2. **Ch1/index.md — "Reading Order" second sentence is redundant with the Scope bullet list.** The Scope section (lines 8–13) enumerates the four categories Ch1 defines. The Reading Order section (lines 16–19) then partially re-lists those same categories as justification for the file order. Only the sentence "The model overview comes first because the hardware topology discussion references GQA head counts and tensor shapes that are defined there" is non-redundant. The surrounding restatement of the four Scope topics can be removed; keep only that one ordering-rationale sentence and the two file links.

3. **Ch3/index.md — same cross-reference cited twice in consecutive Prerequisites bullets (lines 15–16).** Both the "T3K topology and PCIe connectivity" bullet and the "TTNN multi-device tensor types" bullet end with "(see Chapter 1, [`t3k_topology_primer.md`](../ch1_model_and_hardware_context/t3k_topology_primer.md))". The two bullets can share a single parenthetical placed at the end of the second bullet, removing the duplicate citation from the first.

4. **Ch5/index.md — Prerequisites third bullet re-lists the three config field names already shown in the Scope code block.** The phrase "Its three fields (`math_fidelity`, `fp32_dest_acc_en`, `packer_l1_acc`) are covered in this chapter" names fields the reader just saw in the code block six lines above. Drop this parenthetical; retain only the Chapter 1 cross-reference that closes the bullet.

5. **Ch6/index.md — remove the four non-chapter-specific rows from the Ling Configuration Recap table.** The `num_heads`, `num_kv_heads`, `head_dim`, and `hidden_size` rows duplicate values stated identically in Ch1, Ch2, and Ch4. Only `partial_rotary_factor` and `use_qk_norm` (with their Consequence entries) are specific to Ch6's subject matter. The four redundant rows can be removed.

6. **Ch6/index.md — remove the three class-name rows from the Key Symbols table.** The last three rows (`TTNNRMSNorm`, `TTNNRotaryPositionEmbedding`, `TTNNDistributedRotaryPositionEmbedding`) are class/function names, not mathematical symbols. Their descriptions are already given in the Scope section and the Configuration Recap table; they add no lookup value in a symbols table.
