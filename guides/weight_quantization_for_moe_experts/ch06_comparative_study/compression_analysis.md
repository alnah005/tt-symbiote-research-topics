# Compression Analysis — Chapter 6: Comparative Study

## Crucial updates: yes

---

### Duplication 1 — LoFi kernel config code block (internal ch06 + cross-chapter)

**Source (original definition):**
`ch02_ttnn_quantization_api/compute_kernel_config.md`, lines 62–70 (LoFi Standard Configuration)

**Duplicate A (ch06-internal):**
`ch06_comparative_study/index.md`, lines 87–94

**Duplicate B (ch06-internal):**
`ch06_comparative_study/deepseek_v3_quantization_design.md`, lines 33–40

Both ch06 files reproduce the identical LoFi `WormholeComputeKernelConfig` constructor verbatim. The block was originally defined and explained in ch02.

**Important discrepancy to resolve before acting:** `ch02/compute_kernel_config.md` defines the canonical LoFi config with `math_approx_mode=False` and `packer_l1_acc=False`, but the ch06 files show `math_approx_mode=False` and `packer_l1_acc=True`. This divergence is not flagged anywhere in ch06. If the ch06 values are the production-validated ones (as claimed by the surrounding Warning callout), ch02 should be updated; if ch02 is authoritative, ch06 contains an inaccurate copy.

**Recommended action:** Confirm which field values are correct for the MoE expert production path. Then replace both ch06 code blocks with a single cross-reference sentence such as: "LoFi config construction is defined in `ch02/compute_kernel_config.md` § LoFi Config; the production MoE values are `math_approx_mode=False`, `packer_l1_acc=True`." Retain the Warning callout about `fp32_dest_acc_en=False` in `deepseek_v3_quantization_design.md` because it carries load-bearing context not present in ch02.

---

### Duplication 2 — HiFi2 kernel config code block (internal ch06 + cross-chapter)

**Source (original definition):**
`ch02_ttnn_quantization_api/compute_kernel_config.md`, lines 80–89 (HiFi2 Standard Configuration)

**Duplicate A (ch06-internal):**
`ch06_comparative_study/index.md`, lines 97–104

**Duplicate B (ch06-internal):**
`ch06_comparative_study/deepseek_v3_quantization_design.md`, lines 44–51

**Important discrepancy to resolve before acting:** The canonical ch02 HiFi2 config specifies `fp32_dest_acc_en=True` and `math_approx_mode=False`. Both ch06 copies show `fp32_dest_acc_en=False` and `math_approx_mode=True`. These differ on two fields. The ch06 Warning callout (`deepseek_v3_quantization_design.md`, lines 53–56) explicitly states that `fp32_dest_acc_en=False` is the production value for MoE expert projections on Wormhole B0 — meaning ch02's HiFi2 example is a general-purpose template that does not match the MoE production config. This is a meaningful source of confusion across the guide.

**Recommended action:** Add a note to `ch02/compute_kernel_config.md` clarifying that its HiFi2 example is a general template and that the MoE-specific production config sets `fp32_dest_acc_en=False`. Then remove the duplicated constructor from `ch06/index.md` (lines 97–104), replacing it with a cross-reference to `ch02` plus a short inline note about the `fp32_dest_acc_en=False` override. The full constructor in `deepseek_v3_quantization_design.md` carries the authoritative Warning callout and can be retained as the single in-chapter definition, reducing two copies to one.

---

### Duplication 3 — Per-expert bfloat16 memory calculation

**Source (earlier definition):**
`ch05_per_projection_strategy/mixed_precision_memory_layout.md`, line 41 (reference calculation, inline prose)

**Duplicate A:**
`ch06_comparative_study/deepseek_v3_quantization_design.md`, lines 129–135

```
3 × d_model × d_ff × 2 bytes
= 3 × 7168 × 2048 × 2
= 88,080,384 bytes
= 84.0 MB per expert
```

**Duplicate B:**
`ch06_comparative_study/qwen_bfloat16_baseline.md`, lines 31–35

```
Per-expert memory (bfloat16) = 3 projections × d_model × d_ff × 2 bytes
                              = 3 × 7168 × 2048 × 2
                              = 88,080,384 bytes
                              = 84.0 MB per expert
```

The same arithmetic is expanded in both ch06 files independently, and neither references the prior derivation. At minimum, the two ch06 instances are redundant with each other; they are also redundant with ch05.

**Recommended action:** Keep the calculation in `qwen_bfloat16_baseline.md` (it is the natural home — the section is specifically about the bfloat16 baseline cost). In `deepseek_v3_quantization_design.md`, replace the bfloat16 baseline sub-block with a cross-reference: "bfloat16 baseline: 84.0 MB per expert (derived in `qwen_bfloat16_baseline.md`)." The DeepSeek-V3 mixed-precision breakdown that follows (gate/up/down individual lines) is unique to that file and should be retained.

---

### Duplication 4 — T3K system-level memory table

**Source (earlier definition):**
`ch05_per_projection_strategy/mixed_precision_memory_layout.md`, lines 44–57 (summary table and calculations)

**Duplicate A:**
`ch06_comparative_study/deepseek_v3_quantization_design.md`, lines 153–157

| Metric | bfloat16 | DeepSeek-V3 mixed | Reduction |
|---|---|---|---|
| Per-expert weight | 84.0 MB | 28.0 MB | 3× |
| Per-chip (16 experts) | 1,344 MB | 448 MB | 3× |
| Total system (128 experts) | 10,752 MB | 3,584 MB | 3× |

**Duplicate B:**
`ch06_comparative_study/qwen_bfloat16_baseline.md`, lines 43–46

| Scope | bfloat16 | Mixed bfloat4_b gate/up + bfloat8_b down |
|---|---|---|
| Per expert | 84.0 MB | 28.0 MB |
| Per chip (16 experts) | 1,344 MB | 448 MB |
| Full system (128 experts) | 10,752 MB (~10.5 GB) | 3,584 MB (~3.5 GB) |

Both tables convey the same three-row T3K system breakdown with identical numbers. The ch05 file contains the same numbers in a slightly different summary table. Across two ch06 files, the tables are near-verbatim duplicates of each other.

**Recommended action:** Consolidate into a single authoritative table in `qwen_bfloat16_baseline.md` (the file dedicated to this cost analysis). In `deepseek_v3_quantization_design.md`, replace the system-level table with one sentence referencing the consolidated table: "System-level impact figures for T3K are tabulated in `qwen_bfloat16_baseline.md` § Memory Cost Analysis." The per-expert breakdown specific to the DeepSeek-V3 mixed calculation (gate/up/down individual byte lines) is unique to `deepseek_v3_quantization_design.md` and should be retained.

---

## Load-Bearing Evidence

The following content in ch06 is NOT duplicative and must be preserved:

- **`deepseek_v3_quantization_design.md`, "Training Context" section (lines 64–89):** The QAT mechanism explanation (reduced outlier magnitudes, lower per-layer quantization error, implication for Qwen) does not appear in prior chapters at this level of detail. It is the mechanistic foundation for every recommendation in the chapter.

- **`deepseek_v3_quantization_design.md`, "Key Insight" section (lines 177–191):** The two-stage practitioner recipe ("start with bfloat8_b, validate, then consider bfloat4_b") is synthesized here for the first time and drives the downstream framework.

- **`deepseek_v3_quantization_design.md`, Warning callout at lines 53–56:** Explicitly states that `fp32_dest_acc_en=False` is the authoritative production value for MoE expert projections, overriding the ch02 general template. This is the only place this override is documented. Must be retained and ideally referenced back from ch02.

- **`qwen_bfloat16_baseline.md`, "Why Qwen Uses bfloat16" section (lines 112–138):** The four-point rationale (no QAT, conservative bringup, simpler weight loading, accuracy risk aversion) is original to ch06 and not present in prior chapters.

- **`qwen_bfloat16_baseline.md`, "The 16ms Gap Motivation" section (lines 140–170):** This section names the specific gap (~16ms) and quantifies its bandwidth origin with per-MoE-layer DRAM read volume calculations. The 16ms figure and its bfloat8_b remedy are specific to the Qwen production context and do not appear earlier in the guide.

- **`recommendations_and_decision_framework.md`, Decision Criterion 1 budget tier table (lines 22–27):** The three-row table mapping PCC threshold + PPL budget to dtype strategy is a new synthesis not tabulated in prior chapters.

- **`recommendations_and_decision_framework.md`, Decision Criterion 3 training history table (lines 107–111):** Maps training history category to per-dtype risk level. First occurrence in the guide.

- **`recommendations_and_decision_framework.md`, "Path to the Aggressive Tier" 5-step procedure (lines 163–173) and "When to Fall Back" conditions (lines 175–193):** Concrete procedural content that is new in ch06 and not derivable from prior chapters.

- **`recommendations_and_decision_framework.md`, Summary Decision Tree (lines 197–208):** The ASCII decision tree synthesizes the three criteria into a single actionable flow. Not present elsewhere.

- **`index.md`, Side-by-Side Summary table (lines 55–68):** The three-column comparison table (DeepSeek-V3 production | Qwen current | Recommended Qwen start) is the primary comparative artifact of the chapter. It is unique to ch06 even though individual column values appear in prior chapters.

---

## MINOR Suggestions

1. **`index.md` "Compute Kernel Configs (Authoritative)" section (lines 81–104):** This section reproduces both kernel config code blocks and is the clearest internal duplication within ch06 itself (the same blocks appear in `deepseek_v3_quantization_design.md`). After resolving Duplications 1 and 2, this section in `index.md` can be reduced to a two-line cross-reference plus the `fp32_dest_acc_en=False` note, making the index leaner without losing any technical content.

2. **Decode latency table discrepancy:** `deepseek_v3_quantization_design.md` (lines 167–172) uses `576 GB/s` (n300 peak) as the bandwidth basis, while `qwen_bfloat16_baseline.md` (lines 87–93) uses `230 GB/s` (80% of `288 GB/s` n150 peak), and `recommendations_and_decision_framework.md` (lines 63–70) uses `461 GB/s` (80% of `576 GB/s` n300 peak). The three latency tables for the same gate projection operation produce different numbers because they use different hardware targets and efficiency assumptions without declaring this up front. This is not a duplication problem but a consistency risk — a reader scanning across files will see `~51.0 µs`, `~127.7 µs`, and `~63.7 µs` for bfloat16 gate projection and may conclude the numbers are wrong. A brief hardware-context header on each table (e.g., "n300, peak bandwidth" vs "n150, 80% effective") would resolve the confusion at low cost.

3. **`recommendations_and_decision_framework.md`, Deployment Regime section (lines 63–70):** The decode latency table in this file is a third instance of per-projection latency estimates, using yet another bandwidth baseline (461 GB/s). After addressing suggestion 2 above, consider whether this table can be removed in favour of a reference to `ch04/decode_memory_bandwidth.md`, which is the canonical home for this analysis. The unique content in the Deployment Regime section (decode vs. prefill distinction, T3K routing overhead caveat) should be retained.

## Agent A Change Log — C Feedback Pass 1
- index.md: Removed duplicate LoFi/HiFi2 code blocks; added internal cross-reference to deepseek_v3_quantization_design.md
- qwen_bfloat16_baseline.md: Removed duplicate per-expert BF16 calculation; added cross-reference to deepseek_v3_quantization_design.md
- qwen_bfloat16_baseline.md: Removed duplicate T3K memory table; added cross-reference to deepseek_v3_quantization_design.md

## Pass 2 Verification

- **Fix 1 (index.md kernel config code blocks):** Confirmed. The two `WormholeComputeKernelConfig` constructor code blocks (LoFi and HiFi2) have been removed from `index.md` lines 87–104. The "Compute Kernel Configs (Authoritative)" section now contains only the single cross-reference sentence pointing to `deepseek_v3_quantization_design.md`. The canonical constructors remain intact in `deepseek_v3_quantization_design.md` at lines 33–51, along with the load-bearing Warning callout at lines 53–56.

- **Fix 2 (per-expert BF16 calculation in qwen_bfloat16_baseline.md):** Confirmed. The four-line derivation block (`3 × 7168 × 2048 × 2 = 88,080,384 bytes = 84.0 MB per expert`) has been removed from `qwen_bfloat16_baseline.md`. It is replaced by the single cross-reference sentence: "The per-expert BF16 footprint is 84.0 MB (derived in `deepseek_v3_quantization_design.md` in this chapter)." The canonical derivation remains in `deepseek_v3_quantization_design.md` lines 130–135.

- **Fix 3 & 4 (T3K system-level memory table in qwen_bfloat16_baseline.md):** Confirmed. The three-row per-expert/per-chip/full-system table has been removed from `qwen_bfloat16_baseline.md`. It is replaced by the cross-reference sentence: "For the T3K system-level memory totals, see the table in `deepseek_v3_quantization_design.md` in this chapter." The surrounding prose context (10.5 GB headroom impact, 7 GB freed system-wide) has been preserved. The authoritative table remains in `deepseek_v3_quantization_design.md` lines 152–157.

### Remaining Crucial Duplications Check

One `WormholeComputeKernelConfig` constructor block remains in `recommendations_and_decision_framework.md` (lines ~134–142). This is the HiFi2 config presented as the prescriptive Qwen starting-point recommendation, accompanied by a per-projection table with expected PCC values and a code comment ("apply to gate, up, and down projections"). It serves a different narrative role from the canonical definition in `deepseek_v3_quantization_design.md` (which defines the DeepSeek-V3 production config and carries the `fp32_dest_acc_en` Warning callout). These are not redundant — one is the authoritative definition with rationale, the other is a prescriptive recipe for a different model. This does not rise to a CRUCIAL duplication.

No remaining crucial duplications found.

## Crucial updates: no

## Load-Bearing Evidence

All four targeted duplications have been resolved. The content that remains is load-bearing:

- `deepseek_v3_quantization_design.md` retains the LoFi and HiFi2 constructor definitions, the Warning callout on `fp32_dest_acc_en=False`, the bfloat16 baseline derivation (84.0 MB), the DeepSeek-V3 mixed breakdown (gate/up/down individual byte lines), and the T3K memory table — all of which are now the sole ch06 instances of this information.
- `index.md` retains the Side-by-Side Summary table, the Tip and Warning callouts, and the cross-reference section header — none of which are duplicative.
- `qwen_bfloat16_baseline.md` retains all unique content: the "Why Qwen Uses bfloat16" four-point rationale, the "16ms Gap Motivation" section with per-MoE-layer DRAM read calculations, the arithmetic intensity analysis, and the throughput headroom discussion. The cross-reference sentences are accurate and point to the correct file.

## MINOR Suggestions

1. **`qwen_bfloat16_baseline.md`, Per-Expert Footprint section:** The prose "For a single expert with `d_model=7168` and `d_ff=2048`:" now precedes the cross-reference sentence without a calculation. Consider tightening the prose to "The per-expert BF16 footprint for Qwen's dimensions (`d_model=7168`, `d_ff=2048`) is 84.0 MB (derived in `deepseek_v3_quantization_design.md` in this chapter)." and removing the now-redundant preceding sentence, saving one line without losing information.

2. **The ch02 discrepancy flagged in Duplications 1 and 2 remains unresolved.** The compression_analysis.md notes that `ch02/compute_kernel_config.md` uses `fp32_dest_acc_en=True` for HiFi2 (the wrong value for MoE experts), while `deepseek_v3_quantization_design.md` has the correct `fp32_dest_acc_en=False`. Now that ch06 no longer carries the duplicate constructors in `index.md`, a reader following ch02 before ch06 has no correction path until they reach `deepseek_v3_quantization_design.md`. The Warning callout there is sufficient for in-chapter readers, but a note in ch02 pointing forward to ch06 would close the gap for readers who stop at ch02.
