# Compression Analysis: Chapter 6 — Math Fidelity — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~330 lines
- Estimated post-compression line count: ~295 lines
- Estimated reduction: ~11%

---

## CRUCIAL Suggestions

### CRUCIAL-1: `math_fidelity` level table vs bullet list (~10 lines saved)

`index.md` lines 61–66 presents a three-row table (HiFi4 / HiFi2 / LoFi with mantissa bits and approximate relative speed). `hifi4_vs_hifi2.md` lines 5–9 restates the same three levels as a bullet list with identical content: mantissa bit counts, the "default high accuracy" label, and the "~1.5–2×" speed estimate. One of the two representations should be removed. The table in `index.md` is a reference section; the bullet list in `hifi4_vs_hifi2.md` is introductory framing. Keeping the table and replacing the `hifi4_vs_hifi2.md` opening bullets with a one-line forward reference ("see `index.md` parameter reference for the level breakdown") saves ~8–10 lines and removes the restatement entirely.

### CRUCIAL-2: `fp32_dest_acc_en` mechanism restated (~8–10 lines saved)

`index.md` lines 69–74 describes `fp32_dest_acc_en`: FP32 widens the accumulator, `False` reduces dst tile size to 16-bit BFP, enables the "4 to 8 tile" increase from the `qwen_attention.py` comment, and states the Bailing vs Qwen3 settings. `hifi4_vs_hifi2.md` lines 15–19 covers the identical mechanism ("dst register holds partial accumulations... with `fp32_dest_acc_en=True` each dst tile uses 32-bit floats... doubling tile capacity from 4 to 8 tiles"). The `hifi4_vs_hifi2.md` section adds one new detail (interaction with parallel tile dispatch), but the core parameter explanation is a near-verbatim restatement. The `hifi4_vs_hifi2.md` section should be trimmed to start with the new interaction detail and cross-reference `index.md` for the base definition, saving ~8 lines.

### CRUCIAL-3: Decode chunk size rationale duplicated (~8 lines saved)

`index.md` lines 37 explains `q_chunk_size=0` / `k_chunk_size=0`: the value instructs the kernel to auto-select chunk sizes, decode mode Q has only 1 token so the prefill size of 256 is not meaningful, and the kernel defaults to a single tile. `accuracy_throughput_tradeoff.md` lines 94–100 restates the same rationale nearly verbatim: "The value 0 instructs the kernel to pick chunk sizes automatically based on the input dimensions. In decode mode, Q has only 1 token, so the kernel typically defaults to a single tile per chunk." The `accuracy_throughput_tradeoff.md` section adds one sentence about prefill chunk size L1 capacity concern but otherwise duplicates `index.md`. The duplicate prose should be replaced by a cross-reference; only the new L1-capacity sentence should be retained.

---

## MINOR Suggestions

### MINOR-1: QK norm safety rationale repeated across files (~3 lines saved)

`hifi4_vs_hifi2.md` line 52 states that `_apply_qk_norm` "bounds the magnitude of QK^T entries, reducing the dynamic range that fidelity reduction must handle." `accuracy_throughput_tradeoff.md` line 48 states "Models with QK norm tend to have bounded attention logit magnitudes, which typically keeps absolute errors below 1e-2 at HiFi2 fidelity." `accuracy_throughput_tradeoff.md` lines 108 makes the same point again in the `exp_approx_mode` section. The safety argument appears three times. The `hifi4_vs_hifi2.md` version is the most detailed; the two occurrences in `accuracy_throughput_tradeoff.md` can be shortened to one-clause references.

### MINOR-2: "Both use `exp_approx_mode=False`" stated twice (~2 lines saved)

`index.md` line 89 states "Both Bailing MoE and Qwen3 use `False` for this setting." `accuracy_throughput_tradeoff.md` line 106 opens its `exp_approx_mode` section by restating "Both Bailing MoE and Qwen3 currently use `exp_approx_mode=False`." The `accuracy_throughput_tradeoff.md` statement is redundant given the summary table in `index.md`. Remove the sentence and let the section open directly with the risk analysis.

### MINOR-3: `math_approx_mode` noted as `False` in both files (~2 lines saved)

`index.md` lines 83–85 describes `math_approx_mode` and notes that both models use `False`. The summary table on line 125 then restates `math_approx_mode=False` for both sides. Since this parameter requires no action and is fully settled, the prose description (lines 83–85) can be collapsed to a single sentence inline within the parameter reference table, saving the dedicated subsection.

---

## Load-Bearing Evidence

### LBE-1: HiFi2 error propagation analysis (`hifi4_vs_hifi2.md` lines 36–44)

> "Switching from HiFi4 to HiFi2 introduces additional rounding error in each BFP8 product. The error propagates through the attention computation as follows: [QK^T matmul: 128-product dot product, ±4 ulps per product at HiFi2, accumulated error bounded by sqrt(128) × ulp ≈ 11 ulps] ... [Softmax: error propagates additively through the exponential...] ... [Attention-weighted V sum: errors in the attention weights multiply the V values...]"

This quantitative walkthrough of error propagation is the analytical core of the chapter. It cannot be cut or summarized — it provides the only numerical basis for assessing whether HiFi2 is safe for this specific architecture (`head_dim=128`, 128-product dot products). No other file restates this reasoning.

### LBE-2: HiFi2 feasibility factors specific to Bailing MoE (`hifi4_vs_hifi2.md` lines 50–60)

> "Three factors favor HiFi2 for Bailing MoE attention: [1. QK norm bounds logit magnitude] [2. GQA with 4 KV heads — less peaked attention distribution] [3. Dense projection averages across head dimension, smoothing per-head errors.] One factor that argues for caution: Bailing MoE is a precision-sensitive model class. The MoE routing decisions depend on the final hidden state representation..."

This is the only location where the specific architectural properties of Bailing MoE (QK norm, GQA ratio H/Hkv=16/4=4, downstream dense projection, MoE routing sensitivity) are connected to the HiFi2 feasibility decision. It is non-redundant and must not be cut.

### LBE-3: Accuracy measurement thresholds and generation quality gate (`accuracy_throughput_tradeoff.md` lines 44–52)

> "Thresholds for safe adoption (rough guidance; to be validated with generation quality tests): `max_abs_err < 1e-2` for the attention output in bfloat16 space; `p99_rel_err < 1%` at B=32 decode... If tensor-level error metrics pass, run a generation quality benchmark... A perplexity degradation of more than 0.1 points is typically significant for production deployment."

These specific numerical thresholds and the two-gate validation protocol (tensor error first, then perplexity) are actionable criteria for whoever runs the benchmarking. They appear only in `accuracy_throughput_tradeoff.md` and must be preserved in full.

---

## VERDICT
- Crucial updates: yes

---

# Compression Analysis: Chapter 6 — Math Fidelity — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~215 lines (index.md ~139, hifi4_vs_hifi2.md ~66, accuracy_throughput_tradeoff.md ~111; net of blank lines and headers)
- Estimated post-compression line count: ~195 lines
- Estimated reduction: ~9%

---

## CRUCIAL Suggestions

### CRUCIAL-1 (Pass 1 re-check): `math_fidelity` level restatement — RESOLVED (partially, diminished to MINOR)

Pass 1 described a "bullet list" in `hifi4_vs_hifi2.md` restating the three fidelity levels. The actual file (`hifi4_vs_hifi2.md` line 5) contains a single compressed prose sentence: "In brief: `math_fidelity` controls mantissa bits per BFP8 product (4 / 2 / 1 for HiFi4 / HiFi2 / LoFi); `fp32_dest_acc_en=True` uses a 32-bit FP accumulator (4 dst tiles available) while `False` uses a 16-bit BFP accumulator (8 dst tiles available, higher throughput)." This is not a bullet list and is already tightly compressed — the duplication is ~1 line. Pass 1 overstated the saving opportunity at 8–10 lines; the real saving is 1–2 lines. Downgraded to MINOR. Status: RESOLVED as CRUCIAL.

### CRUCIAL-2 (Pass 1 re-check): `fp32_dest_acc_en` mechanism restated in combined-effects table — RESOLVED

Pass 1 flagged `hifi4_vs_hifi2.md` lines 15–19 (the combined-effects table) as a "near-verbatim restatement." On re-examination the table synthesizes four distinct parameter combinations not expressly assembled in `index.md`, making it additive rather than duplicative. The only actual overlap is the single inline sentence on line 5 already addressed under CRUCIAL-1 above. The combined-effects table must be preserved. Status: RESOLVED.

### CRUCIAL-3 (Pass 1 re-check): Decode chunk size rationale duplicated — OPEN (~4 lines)

`index.md` line 37 fully explains `q_chunk_size=0` / `k_chunk_size=0`: the value instructs the kernel to auto-select chunk sizes, decode Q has only 1 token so 256 is not meaningful, and the kernel defaults to a single tile. `accuracy_throughput_tradeoff.md` lines 94–97 restates the identical rationale verbatim before adding one new sentence: "the profiling comparison should pin the chunk size to isolate the fidelity/accumulator effect from chunk-size variation." The restatement spans ~3–4 lines. Fix: replace the duplicate prose in `accuracy_throughput_tradeoff.md` with a one-clause cross-reference ("As noted in `index.md`, the decode config auto-selects chunk sizes...") and retain only the new pinning sentence. Saves ~3–4 lines. Status: OPEN.

---

## MINOR Suggestions

### MINOR-1 (Pass 1, carried): QK norm safety argument appears three times (~2 lines saved)

The argument that QK norm bounds logit magnitudes and mitigates fidelity risk appears in `hifi4_vs_hifi2.md` line 42 (most detailed), `accuracy_throughput_tradeoff.md` line 48 (threshold section), and `accuracy_throughput_tradeoff.md` line 104 (`exp_approx_mode` section). The two shorter occurrences in `accuracy_throughput_tradeoff.md` can each be reduced to a parenthetical clause referencing `hifi4_vs_hifi2.md`. Combined saving: ~2 lines.

### MINOR-2 (Pass 1, carried): "Both use `exp_approx_mode=False`" stated twice (~1 line saved)

`index.md` line 89 and `accuracy_throughput_tradeoff.md` line 102 both open their respective `exp_approx_mode` paragraphs with the same factual statement. The `accuracy_throughput_tradeoff.md` sentence is redundant given the summary table in `index.md`. Remove it and let the risk analysis open directly.

### MINOR-3 (Pass 1, carried): `math_approx_mode` prose section vs summary table (~2 lines saved)

`index.md` lines 83–85 is a dedicated subsection that states `math_approx_mode` controls transcendental approximations and notes both models use `False`. The summary table on line 125 already captures this with a `False / False` row. Since this parameter is fully settled and requires no action, the subsection prose can be collapsed to a single inline parenthetical ("(`math_approx_mode=False` for both; no action required)") at the start of the following `exp_approx_mode` subsection or dropped entirely. Saves ~2 lines.

### MINOR-4 (new): Redundant "rough estimates" qualification (~1 line saved)

`accuracy_throughput_tradeoff.md` line 82 states: "These are rough order-of-magnitude estimates based on the architectural changes; actual numbers require profiling per the methodology above." The table caption already conveys this intent, and the same caveat appears in the preceding throughput measurement methodology section. Remove the standalone sentence and let the table stand without it. Saves 1 line.

---

## Load-Bearing Evidence

### LBE-1: HiFi2 error propagation with `head_dim=128` quantification (`hifi4_vs_hifi2.md` lines 28–34)

> "At HiFi2, each product's mantissa is rounded to 2 bits, introducing an error of up to ±4 ulps per product. Over 128 products, accumulated error is bounded (with high probability) by approximately ±`sqrt(128)` × ulp ≈ 11 ulps in the HiFi2 case vs ±2 ulps for HiFi4."

This is the only numerical anchor in the chapter. It ties the fidelity level directly to the specific `head_dim=128` of Bailing MoE and gives the only quantitative basis for comparing HiFi2 vs HiFi4 error magnitude. Must not be cut or paraphrased.

### LBE-2: Three feasibility factors for Bailing MoE HiFi2 adoption (`hifi4_vs_hifi2.md` lines 40–50)

> "[1] `_apply_qk_norm`... normalizes Q and K to unit scale... bounds the magnitude of QK^T entries... [2] GQA with 4 KV heads... H/Hkv = 16/4 = 4... less peaked... [3] Dense projection follows attention... projection averages across the head dimension... One factor that argues for caution: Bailing MoE is a precision-sensitive model class. The MoE routing decisions depend on the final hidden state representation, which accumulates errors across all operations in the layer."

These architecture-specific factors and the caution note about MoE routing sensitivity are the decision-relevant core of the feasibility assessment. They appear only here and must be preserved intact.

### LBE-3: Two-gate validation protocol with numerical thresholds (`accuracy_throughput_tradeoff.md` lines 44–52)

> "Thresholds for safe adoption... `max_abs_err < 1e-2` for the attention output in bfloat16 space; `p99_rel_err < 1%` at B=32 decode... If tensor-level error metrics pass, run a generation quality benchmark... A perplexity degradation of more than 0.1 points is typically significant for production deployment."

These are the only concrete numerical pass/fail criteria and the only statement of the two-step validation gate in the entire chapter. They are actionable and non-redundant; any compression must leave them word-for-word intact.

---

## VERDICT
- Crucial updates: yes

---

# Compression Analysis: Chapter 6 — Math Fidelity — Pass 3

## Summary
- Total files analyzed: 3
- Estimated current line count: ~313 lines (index.md: 138, hifi4_vs_hifi2.md: 65, accuracy_throughput_tradeoff.md: 110)
- Estimated post-compression line count: ~305 lines
- Estimated reduction: ~3%

---

## CRUCIAL Suggestions

### CRUCIAL-1 (Pass 1, Pass 2 re-check): `math_fidelity` level restatement — RESOLVED

Pass 1 flagged a "bullet list" in `hifi4_vs_hifi2.md` as a ~8–10 line duplication of the `index.md` table. Pass 2 confirmed the content had already been compressed to a single prose sentence (`hifi4_vs_hifi2.md` line 5: "In brief: `math_fidelity` controls mantissa bits per BFP8 product (4 / 2 / 1 for HiFi4 / HiFi2 / LoFi)...") and downgraded the item to MINOR. The current file confirms this sentence is unchanged. No CRUCIAL action remains. Status: RESOLVED.

### CRUCIAL-2 (Pass 1, Pass 2 re-check): `fp32_dest_acc_en` mechanism restatement — RESOLVED

Pass 1 flagged `hifi4_vs_hifi2.md` lines 15–19 as a near-verbatim restatement. Pass 2 confirmed the combined-effects table synthesizes four distinct parameter combinations not assembled in `index.md` and is therefore additive. The current `hifi4_vs_hifi2.md` lines 12–19 are unchanged — still the four-row combined table. No CRUCIAL action remains. Status: RESOLVED.

### CRUCIAL-3 (Pass 1, Pass 2 re-check): Decode chunk size rationale duplicated — RESOLVED

Pass 2 described `accuracy_throughput_tradeoff.md` as restating the "q_chunk_size=0 auto-select / 1-token decode" rationale verbatim before adding one new pinning sentence. The current `accuracy_throughput_tradeoff.md` lines 94–97 read: "When benchmarking configs B–D against the baseline, pin chunk sizes explicitly (override `q_chunk_size=0` to a fixed value) to isolate the fidelity/accumulator effect from any variation introduced by the kernel's automatic chunk selection. The current chunk sizes are documented in `index.md`'s Current Configuration section." The verbatim restatement described in Pass 2 is no longer present — the section now opens directly with the new pinning instruction and cross-references `index.md` for the base rationale. The recommended fix has already been applied. Status: RESOLVED.

### No new CRUCIAL items found

A full scan of the three files found no new redundancies reaching the ~8-line threshold. The candidates examined and ruled out:

- **Config A/B/C/D table** (`accuracy_throughput_tradeoff.md` lines 6–13) vs `index.md` summary table (lines 122–133): the tradeoff table adds the Label column and groups all four candidate configs not assembled in `index.md`. Additive, not duplicative.
- **`packer_l1_acc` mechanism**: `index.md` lines 76–81 contains the full parameter definition; `hifi4_vs_hifi2.md` references it only in table cells ("Reduced (L1 acc)" / "Direct to output buffer"). No prose restatement.
- **`fp32_dest_acc_en=False` Qwen3 precedent** (`hifi4_vs_hifi2.md` lines 56–61): states the T3K shared-hardware argument for safe adoption. `index.md` lines 115–116 only notes that the two parameters are "the two most impactful divergences." Overlap is ~1 line; well below threshold.
- **`exp_approx_mode` treatment**: `index.md` lines 87–89 (2 lines: defines parameter, notes both use `False`) vs `accuracy_throughput_tradeoff.md` lines 100–106 (risk analysis and evaluation sequencing). Overlap is ~1 line (the shared-false fact); the `accuracy_throughput_tradeoff.md` section is substantively new content.

---

## MINOR Suggestions

### MINOR-1 (Pass 2 carried): QK norm safety argument appears three times (~2 lines saved)

The argument that `_apply_qk_norm` bounds logit magnitudes still appears at `hifi4_vs_hifi2.md` line 42 (detailed), `accuracy_throughput_tradeoff.md` line 48 (threshold section), and `accuracy_throughput_tradeoff.md` line 104 (`exp_approx_mode` section). The two occurrences in `accuracy_throughput_tradeoff.md` remain as full sentences; each can be shortened to a parenthetical reference to `hifi4_vs_hifi2.md`. Combined saving: ~2 lines.

### MINOR-2 (Pass 2 carried): "Both use `exp_approx_mode=False`" stated twice (~1 line saved)

`index.md` line 89 ("Both Bailing MoE and Qwen3 use `False` for this setting.") and `accuracy_throughput_tradeoff.md` line 102 ("Both Bailing MoE and Qwen3 currently use `exp_approx_mode=False`.") remain unchanged from Pass 2. The `accuracy_throughput_tradeoff.md` sentence is redundant given the `index.md` summary table. Removing it allows the risk analysis to open directly. Saves ~1 line.

### MINOR-3 (Pass 2 carried): `math_approx_mode` prose section vs summary table (~2 lines saved)

`index.md` lines 83–85 still carries a dedicated subsection for `math_approx_mode` whose entire content is that it controls transcendental approximations and both models use `False`. The summary table at line 125 already captures `False / False`. The subsection can be collapsed to a single parenthetical or dropped. Saves ~2 lines.

### MINOR-4 (Pass 2 carried): Redundant "rough estimates" qualification (`accuracy_throughput_tradeoff.md` line 82) (~1 line saved)

The sentence "These are rough order-of-magnitude estimates based on the architectural changes; actual numbers require profiling per the methodology above." remains unchanged. The caveat is already implied by the section header ("Expected Throughput Gains (Rough Estimates)") and the preceding methodology section. The standalone sentence can be removed. Saves ~1 line.

---

## Load-Bearing Evidence

### LBE-1: HiFi2 error propagation with `head_dim=128` quantification (`hifi4_vs_hifi2.md` lines 28–34)

> "At HiFi2, each product's mantissa is rounded to 2 bits, introducing an error of up to ±4 ulps per product. Over 128 products, accumulated error is bounded (with high probability) by approximately ±`sqrt(128)` × ulp ≈ 11 ulps in the HiFi2 case vs ±2 ulps for HiFi4."

This is the only numerical anchor in the chapter and the only location tying fidelity level to Bailing MoE's specific `head_dim=128`. It cannot be paraphrased or merged with any other passage without losing the quantitative basis for the safety assessment.

### LBE-2: Three feasibility factors and MoE caution note (`hifi4_vs_hifi2.md` lines 40–50)

> "[1] `_apply_qk_norm`... normalizes Q and K to unit scale... bounds the magnitude of QK^T entries... [2] GQA with 4 KV heads... H/Hkv = 16/4 = 4... less peaked... [3] Dense projection follows attention... averages across the head dimension... One factor that argues for caution: Bailing MoE is a precision-sensitive model class. The MoE routing decisions depend on the final hidden state representation, which accumulates errors across all operations in the layer."

These architecture-specific factors are the decision-relevant core of the feasibility assessment and appear only in `hifi4_vs_hifi2.md`. Any compression pass must leave this passage intact.

---

## VERDICT
- Crucial updates: no
