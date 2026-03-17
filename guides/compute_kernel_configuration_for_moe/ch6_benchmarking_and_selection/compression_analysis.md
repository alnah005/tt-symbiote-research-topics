# Compression Analysis — Chapter 6: Performance Benchmarking and Config Selection

## Crucial updates: yes

---

### Duplication 1 — Canonical config Python definitions (config_decision_matrix.md)

**Source (original):**
`ch1_kernel_config_fundamentals/wormhole_compute_kernel_config_api.md`, lines 190–219 ("Canonical Production Configs" section), which gives the authoritative commented definition of both constants. Also reproduced verbatim in `ch5_moe_expert_matmul_config/deepseek_v3_config_analysis.md`, lines 15–31.

**Duplicate (ch6 location):**
`config_decision_matrix.md`, lines 20–36. The file opens a "Canonical Production Configs (Reference)" section, presents the two-row parameter table, and then gives a full Python code block reproducing both `COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2` definitions verbatim. The code block adds no new comments over ch1's or ch5's versions.

**Why it is crucial:**
The code block in `config_decision_matrix.md` is a self-contained repaste of definitions that already exist in ch1 (canonical, with detailed inline comments) and ch5 (with production-context rationale). A reader who has followed the guide in order already has these definitions in two earlier locations. The parameter table immediately above the code block (lines 13–17) carries the same information in a more compact form and is more appropriate here. Removing the code block and replacing it with a one-line cross-reference (`See ch1/wormhole_compute_kernel_config_api.md §Canonical Production Configs for the constructor code`) would meaningfully reduce redundancy without removing any reasoning unique to ch6.

**Recommended action:**
Remove the Python code block at `config_decision_matrix.md` lines 20–36. Replace it with a single cross-reference sentence pointing to `ch1_kernel_config_fundamentals/wormhole_compute_kernel_config_api.md` §Canonical Production Configs (or equivalently to `ch5_moe_expert_matmul_config/deepseek_v3_config_analysis.md` §Config Definitions). Retain the parameter table (lines 13–17), which is the load-bearing summary for this file.

---

### Duplication 2 — Canonical config Python definitions (production_config_checklist.md)

**Source (original):**
`ch1_kernel_config_fundamentals/wormhole_compute_kernel_config_api.md`, lines 190–219; also `ch5_moe_expert_matmul_config/deepseek_v3_config_analysis.md`, lines 15–31.

**Duplicate (ch6 location):**
`production_config_checklist.md`, lines 45–67. Checklist item 4 ("Configs Stored in a Shared config_helpers.py") includes a full Python code block showing both config definitions inside a `config_helpers.py` module template. The code body — field names, values, and constructor calls — is identical to the ch1 and ch5 versions. The only differences are the surrounding comment strings ("Safe for any SwiGLU/SiLU MoE gate/up projection on Wormhole B0" etc.) and the module path docstring.

**Why it is crucial:**
The purpose of checklist item 4 is to instruct the reader to centralize config definitions in a `config_helpers.py` file, not to re-teach what the config values are. The comment strings added in the ch6 block are useful, but they do not require a full code reproduction: they can be expressed as prose requirements ("Add a comment explaining the projection type and hardware constraint each config is intended for") plus a pointer to the ch1/ch5 reference. Repeating the full block inflates the checklist without adding decision-making content.

**Recommended action:**
Replace the Python code block at `production_config_checklist.md` lines 45–67 with a short prose paragraph describing the required comment style for the shared module, followed by a cross-reference: "For the config constructor values, see `ch1_kernel_config_fundamentals/wormhole_compute_kernel_config_api.md` §Canonical Production Configs." Alternatively, retain only the module-path template and the comment strings, stripping the constructor body lines, so the block illustrates the comment pattern without repeating the field values.

---

### Duplication 3 — L1 budget prevention procedure (production_config_checklist.md)

**Source (original):**
`ch3_packer_l1_acc/packer_l1_acc_constraints.md`, lines 119–153. This file contains a full L1 budget check: the accumulation buffer formula, a Python estimator function `estimate_l1_usage(...)`, and the instruction to run a dry dispatch. The four overflow-resolution options are also given in a table at lines 59–65.

**Duplicate (ch6 location):**
`production_config_checklist.md`, lines 109–129 ("Common Mistake: L1 Budget Not Verified" section). The section restates: (a) the failure mode and error message (also in ch3 lines 43–55), (b) a 5-step prevention procedure that re-derives the accumulation buffer arithmetic (`per_core_N * 32 * 32 * 2 bytes` for BF16, `* 4 bytes` for FP32), and (c) the advice to reduce `per_core_N` or `out_subblock_w` rather than disabling `packer_l1_acc`. Steps 1–4 of the procedure are a prose restatement of the Python formula in ch3. The final tip ("the correct fix is to reduce per_core_N") is present nearly verbatim in ch3's resolution options table.

**Why it is crucial:**
The prevention procedure in ch6 is a multi-step numbered block (lines 119–129) that is functionally identical in scope and content to ch3's L1 budget check procedure. A reader working through the guide has already been pointed to ch3 in the prerequisites (ch6 `index.md` line 68), and checklist item 3 (lines 28–35) already links to `production_config_checklist.md` for the verification procedure while capturing the essential facts (1.5 MB limit, 2× buffer size, dry dispatch). The separate "Common Mistake" section duplicates this itemised procedure rather than referring back to it. Removing the numbered steps 1–4 (the buffer-size arithmetic) and replacing them with a reference to `ch3_packer_l1_acc/packer_l1_acc_constraints.md` §L1 Budget Check Procedure would eliminate the duplication while keeping step 5 ("run a dry dispatch at max batch size") and the tip about `per_core_N` as actionable reminders.

**Recommended action:**
Trim `production_config_checklist.md` lines 119–129: remove the arithmetic steps 1–4 of the prevention procedure and replace with a one-sentence cross-reference ("For the full L1 budget formula and `estimate_l1_usage` helper, see `ch3_packer_l1_acc/packer_l1_acc_constraints.md` §L1 Budget Check Procedure"). Retain step 5 (dry dispatch) and the tip paragraph, as these are the actionable deployment reminders appropriate to a checklist context.

---

## Load-Bearing Evidence

The following retained content in ch6 appears superficially similar to prior chapters but is NOT a crucial duplication:

**Bandwidth reduction formula in benchmarking_methodology.md (lines 176–182).**
The formula `(K_t−1)/K_t` appears in abbreviated form with explicit attribution ("See ch3_packer_l1_acc/throughput_impact.md for the full derivation and table — do not re-derive here"). The ch6 use is a contextual anchor for interpreting benchmark speedup results, not a re-derivation. The surrounding prose ("The observed end-to-end speedup is lower than the theoretical bandwidth reduction because...") is original analysis not present in ch3.

**Summary decision table in config_decision_matrix.md (lines 110–118).**
The ch5 index.md (lines 27–32) has a three-row table mapping projections to configs. The ch6 table adds a regime column (decode vs. prefill), a notes column with PCC threshold language, and an explicit row for the strict-PCC scenario. It is a synthesis across multiple chapters, not a copy of any single source table.

**Decision flowchart in index.md (lines 27–58).**
No prior chapter contains a decision flowchart of this structure. It is original to ch6.

**Token-level vs. layer-level PCC discussion in benchmarking_methodology.md (lines 224–237).**
While ch2 discusses PCC measurement in general, the layer-level vs. token-level distinction with the specific warning about M=1 statistical unreliability is introduced in ch6. The PCC threshold of 0.999 appears in ch2 `fidelity_selection_workflow.md` but the layered measurement architecture is ch6-original.

---

## MINOR Suggestions

1. **benchmarking_methodology.md, lines 244–252 (reporting table):** The table template rows ("LoFi | False | False | — (baseline)") are a useful scaffold but the specific row ordering could be reframed as the recommended sweep order (run `packer_l1_acc=True` first, per the tip at line 184) to avoid inadvertently suggesting the False/False baseline is the natural starting point.

2. **config_decision_matrix.md, lines 128–132 (d_ff/d_model table):** The DeepSeek-V3 and Qwen 235B-A22B rows are duplicated across ch5 (deepseek_v3_config_analysis.md §DeepSeek-V3 Dimensions and applying_configs_to_qwen.md). Here they serve a comparison purpose alongside the hypothetical model, which is new. This is acceptable as a synthesis table, but a brief note ("DeepSeek and Qwen values from Chapter 5") would clarify their origin.

3. **production_config_checklist.md, checklist summary card (lines 133–164):** The card is a useful at-a-glance tool but repeats the structure of the numbered checklist items above it. Consider whether a separate summary card is warranted or whether a well-formatted "quick reference" block within the existing checklist items would serve the same purpose with less vertical space.

4. **benchmarking_methodology.md, lines 197–223 (compute_pcc function):** The `torch.corrcoef` based PCC implementation is distinct from ch2's `fidelity_selection_workflow.md` template (which uses an inline `torch.corrcoef` one-liner). The ch6 version wraps it in a named function with a docstring explaining the flatten-then-correlate strategy. This is a minor improvement over the ch2 version, but the two files now present subtly different PCC implementations without cross-reference. Add a note that this function generalizes the one-liner in `ch2_math_fidelity_levels/fidelity_selection_workflow.md`.

## Agent A Change Log — C Feedback Pass 1
- config_decision_matrix.md: Replaced canonical config Python block with Chapter 1 + Chapter 5 cross-reference
- production_config_checklist.md: Removed duplicate config constructor block; added Chapter 1 cross-reference
- production_config_checklist.md: Collapsed L1 budget arithmetic derivation; kept dry-dispatch step; added Chapter 3 cross-reference

## Pass 2 Verification

**Fix 1 — config_decision_matrix.md canonical config Python block:** Confirmed removed. Lines 20–36 (the LOFI/HIFI2 `ttnn.WormholeComputeKernelConfig(...)` block) are replaced by a single cross-reference sentence pointing to Chapter 1 (`wormhole_compute_kernel_config_api.md`) and Chapter 5 (`deepseek_v3_config_analysis.md`). The parameter table (two-row summary above the former code block) is retained.

**Fix 2 — production_config_checklist.md config Python block (checklist item 4):** Confirmed removed. The constructor block (old lines 45–67) is gone. The module path template, comment-style requirements, and Chapter 1 cross-reference are present as a single prose paragraph. No constructor field values remain in this section.

**Fix 3 — production_config_checklist.md L1 budget prevention procedure:** Confirmed collapsed. The 5-step numbered procedure is replaced by one sentence cross-referencing Chapter 3 (`packer_l1_acc_constraints.md`) followed by the retained dry-dispatch step and the closing tip about reducing `per_core_N` rather than disabling `packer_l1_acc`.

### Remaining Crucial Duplications Check

No remaining crucial duplications found.

Scanned content reviewed and cleared:

- `config_decision_matrix.md` lines 77–84 (`COMPUTE_KERNEL_CONFIG_HIFI4_STRICT` block): This defines a third config constant not present in ch1 or ch5 (which cover only LOFI and HIFI2). It is ch6-original content introduced for the strict PCC regime and is not a duplication.
- `production_config_checklist.md` lines 71–81 (routed/shared expert alias block): Uses alias assignment syntax (`COMPUTE_KERNEL_CONFIG_ROUTED_GATE_UP = COMPUTE_KERNEL_CONFIG_LOFI`) to illustrate the naming pattern for heterogeneous expert configs. No constructor field values are redefined; the block is pedagogically distinct from the ch1/ch5 constructor definitions.
- `benchmarking_methodology.md` lines 63–68 (inline `config_under_test` object): A single anonymous benchmark object used as a placeholder in the standalone benchmark template. It is not presented as a named canonical constant and is not verbatim from ch1 or ch5.
- `benchmarking_methodology.md` lines 150–165 (`config_packer_off` / `config_packer_on` objects): Two inline objects defined to isolate the `packer_l1_acc` variable in an A/B comparison. Both are LoFi configs, not the named LOFI/HIFI2 canonical constants, and the packer isolation pattern is ch6-original methodology content.

## Crucial updates: no

## Load-Bearing Evidence

The three fixes applied in Pass 1 resolved all crucial duplications identified in the original compression analysis. The remaining code blocks in ch6 are load-bearing for the following reasons:

- **`COMPUTE_KERNEL_CONFIG_HIFI4_STRICT` block** (`config_decision_matrix.md`): The only definition of the strict PCC config anywhere in the guide. Ch1 and Ch5 do not cover this regime; the block must remain here or the strict PCC section has no concrete specification.
- **Routed/shared expert alias block** (`production_config_checklist.md`): Illustrates a naming convention for heterogeneous expert configs that is not taught elsewhere. The aliasing pattern (using existing constants rather than new constructors) is the load-bearing content, not the field values.
- **Inline benchmark config objects** (`benchmarking_methodology.md`): The benchmark template and packer isolation test are ch6-original methodology. The inline constructors are instantiated with deliberately varied field values (e.g., `packer_l1_acc=False` vs. `True`) to demonstrate the isolation technique; they cannot be replaced with cross-references to named constants without losing the pedagogical structure.

## MINOR Suggestions

1. **config_decision_matrix.md, `COMPUTE_KERNEL_CONFIG_HIFI4_STRICT` block (lines 77–84):** The block currently does not have a comment explaining why `fp32_dest_acc_en=True` is paired with `packer_l1_acc=True` here. A one-line inline comment ("# L1 budget must be verified before deploying — see production_config_checklist.md §3") would reinforce the cross-reference in the warning immediately below.
2. **production_config_checklist.md, checklist item 4 prose (line 43):** The cross-reference sentence currently reads "For the config constructor values, see Chapter 1...". Consider rephrasing to "For the full constructor definitions (field names, values, and types), see Chapter 1..." to make it explicit what the reader will find there, since the comment-style requirements are stated inline in ch6 but the field values are not.
3. **production_config_checklist.md, "Common Mistake" section intro (lines 85–93):** The failure mode error string (`ttnn.exceptions.TTNNException: ... L1 allocation failed ...`) and the "only manifests at max batch size" observation are also present in ch3's `packer_l1_acc_constraints.md` (per the original compression analysis). This is a minor duplication (short error string + one explanatory sentence), not a crucial one, since the surrounding context in ch6 (checklist framing, production deployment angle) is distinct. A brief "(also described in Chapter 3)" attribution would be sufficient if strict de-duplication is desired.
