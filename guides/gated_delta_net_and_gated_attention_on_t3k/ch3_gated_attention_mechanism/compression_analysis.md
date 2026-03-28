# Compression Analysis: Chapter 3 — Gated Attention: Mechanism and Tensor Shapes — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~325 lines
- Estimated post-compression line count: ~265 lines
- Estimated reduction: ~18%

## CRUCIAL Suggestions

### [gated_attention_formulation.md] ~lines 30–46
**Issue:** "Shape arithmetic" prose immediately before and after each code block restates what the arithmetic in the block already shows. Lines 30–31 say "`n_q_h × d_h × 2 = 16 × 256 × 2 = 8192`" and "The factor of 2 interleaves Q and gate channels," then line 46 repeats "Shape check: `n_q_h × d_h = 16 × 256 = 4096`. Correct." The inline arithmetic inside the code blocks (e.g., `= [16 × 256 × 2, 2048] = [8192, 2048]`) already performs these calculations; the prose sentences add nothing.
**Suggestion:** Drop the "Shape arithmetic:" prose sentences (lines 30–31 and 59–60) and the trailing "Shape check: … Correct." lines (lines 46, 75) entirely. The code blocks are self-documenting.

### [gated_attention_formulation.md] ~lines 88–90 and gated_vs_vanilla_attention_shapes.md ~lines 86–93
**Issue:** The geometric interpretation of the sigmoid gate is given in full in `gated_attention_formulation.md` (lines 88–90: "Each element of `gate_sigmoid` lies in (0, 1). Multiplying by Q entry-wise performs input-dependent suppression…") and then substantially restated in `gated_vs_vanilla_attention_shapes.md` (lines 86–93, Difference 1 and Difference 2). The "gate-before-norm" ordering explanation is also duplicated: `gated_attention_formulation.md` lines 108–109 and `gated_vs_vanilla_attention_shapes.md` lines 86–93.
**Suggestion:** In `gated_vs_vanilla_attention_shapes.md` Difference 1, remove the explanatory sentences about what the gate does (keep the weight-shape comparison code block only, which is unique). In Difference 2, keep the code block and single-sentence consequence; remove the restated explanation of gate-before-norm that already appears verbatim in `gated_attention_formulation.md` section 4.

### [gated_vs_vanilla_attention_shapes.md] ~lines 118–131
**Issue:** Section 2, Difference 4 includes a worked KV cache memory calculation (lines 118–131) that occupies 14 lines to show K+V = 2,048 bytes/token per layer, then compare against a vanilla MHA hypothetical. This same kind of per-token byte arithmetic is re-presented more compactly (with a summary table) in Section 3 (lines 145–178). Having the raw per-layer comparison prose in Difference 4 followed immediately by the full Section 3 table creates redundancy.
**Suggestion:** In Difference 4, collapse the 14-line KV cache arithmetic block to a single sentence referencing Section 3 (e.g., "See Section 3 for a per-layer memory breakdown."), and remove the inline vanilla-MHA comparison byte calculation. The summary table in Section 3 already covers all T values including T=1 (decode).

## MINOR Suggestions

### [index.md] ~lines 20–24
**Issue:** Learning objectives 1–4 are detailed paraphrases of content already previewed in the two section links and the symbol table directly below. Objective 1 restates the entire forward-pass sequence; objective 4 restates what `gated_vs_vanilla_attention_shapes.md` is explicitly described as covering in the Sections list.
**Suggestion:** Condense objectives to two tight bullet points (forward-pass + shapes/comparison), or merge objectives 1 and 3 (both concern shape tracing) and objectives 2 and 4 (both concern architectural differences).

### [gated_attention_formulation.md] ~lines 200–201 and 237–238
**Issue:** Two identical shape-check lines: "Shape check: `sqrt(d_h) = sqrt(256) = 16.0`" (line 200) and "Shape check: `n_q_h × d_h = 16 × 256 = 4096`" (line 237) appear as standalone lines after code blocks. Like the earlier shape-check lines, these arithmetic facts are visible inside the code blocks themselves.
**Suggestion:** Remove both trailing "Shape check:" lines; the preceding code blocks already carry the information.

### [gated_vs_vanilla_attention_shapes.md] ~lines 11–16
**Issue:** The abbreviation legend at the top of Section 1 defines `n_h`, `n_q_h`, `n_kv_h`, `d_h`, and `d_k`. These are already defined in `index.md`'s symbol table (which the formulation file explicitly points to on line 3). The `d_k` entry is only used once, in the Delta Net comparison table header in Section 4.
**Suggestion:** Remove the abbreviation legend block; add a one-line cross-reference to `index.md` for symbol definitions. Keep `d_k` defined inline in Section 4 where it first appears.

### [gated_vs_vanilla_attention_shapes.md] ~lines 30 and 45
**Issue:** "Shape check: `n_h × d_h = 16 × 128 = 2048`. Output matches hidden size directly." (line 30) and "Shape check: `n_q_h × d_h = 16 × 128 = 2048`. Expand factor: `n_q_h / n_kv_h = 16 / 4 = 4×`." (line 45) are post-table prose that restate what the Reshaped-output row in each table already shows.
**Suggestion:** Remove both "Shape check:" lines beneath the MHA and GQA tables; the table rows are sufficient.

### [gated_attention_formulation.md] ~lines 132–133
**Issue:** "This is a deliberate design choice that reduces the fraction of head capacity consumed by positional encoding." The sentence is speculative intent-attribution and adds no technical content beyond what "only 64 of 256 dimensions carry positional signal" already conveys.
**Suggestion:** Delete the final sentence of Section 5; the preceding factual sentence is sufficient.

## Load-Bearing Evidence
- `index.md` line ~8: "It is **not** Gated Linear Attention (GLA). GLA is a separate recurrent architecture (related to linear attention); it does not appear in Qwen3.5-35B-A3B." — load-bearing because it is the critical disambiguation that prevents misidentification of the architecture.
- `gated_attention_formulation.md` line ~88: "Each element of `gate_sigmoid` lies in (0, 1). Multiplying by Q entry-wise performs input-dependent suppression: query dimensions that the model deems irrelevant for the current token are scaled toward zero." — load-bearing because this is the primary conceptual explanation of the gate; the comparison file's restatements are the redundant copies, not this one.
- `gated_attention_formulation.md` line ~108: "The sequence is: project → gate_sigmoid ⊙ Q → RMSNorm(Q_gated). Most models that use both gating and normalization apply normalization first. Here the gating precedes the norm…" — load-bearing as the canonical gate-before-norm explanation.
- `gated_vs_vanilla_attention_shapes.md` line ~116: "This reduces KV cache size by 8× relative to MHA (at the same head dim), but places a strong structural constraint on what the KV representations must encode." — load-bearing architectural consequence unique to this section.
- `gated_vs_vanilla_attention_shapes.md` line ~197: "Both architectures produce a 4096-dimensional pre-projection representation that is mapped back to the 2048-dimensional hidden state by a shared output projection structure." — load-bearing because it states the architectural interface equivalence between Gated Delta Net and Gated Attention.

## VERDICT
- Crucial updates: yes

---

## Agent A Change Log — Compression Pass 1

**Date:** 2026-03-27

### Item 1 applied — `gated_attention_formulation.md`
- Removed "Shape arithmetic: `n_q_h × d_h × 2 = 16 × 256 × 2 = 8192`. The factor of 2 interleaves Q and gate channels in the output dimension." prose sentence (was line 30).
- Removed trailing "Shape check: `n_q_h × d_h = 16 × 256 = 4096`. Correct." line (was line 46).
- Removed "Shape arithmetic: `n_kv_h × d_h × 2 = 2 × 256 × 2 = 1024`." prose sentence (was line 59).
- Removed trailing "Shape check: `n_kv_h × d_h = 2 × 256 = 512`. Correct." line (was line 75).
- The code blocks already show all arithmetic inline; no information is lost.

### Item 2 applied — `gated_vs_vanilla_attention_shapes.md`
- In Difference 1: replaced the explanatory sentence about the gate's geometric effect with a cross-reference: "See [`gated_attention_formulation.md`](./gated_attention_formulation.md) Section 3 for the geometric interpretation." The weight-shape comparison code block (unique content) is retained.
- In Difference 2: removed the multi-sentence prose explanation of gate-before-norm ordering and its consequence (fully duplicated from `gated_attention_formulation.md` Section 4). Kept the code block showing the two orderings and replaced the prose with: "See [`gated_attention_formulation.md`](./gated_attention_formulation.md) Section 4 for the gate-before-norm explanation."

### Item 3 applied — `gated_vs_vanilla_attention_shapes.md`
- In Difference 4: collapsed the 14-line inline KV cache byte arithmetic block (K, V, total-per-layer, total-10-layers, and vanilla MHA comparison) into a single appended sentence: "See Section 3 for a per-layer memory breakdown." The load-bearing sentence about 8× KV cache reduction and structural constraint is retained verbatim.

---

# Compression Analysis: Chapter 3 — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~512 lines (index.md ~41, gated_attention_formulation.md ~284, gated_vs_vanilla_attention_shapes.md ~187)
- Estimated post-compression line count: ~490 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions

None — all Pass 1 crucials resolved.

Pass 1 items verified:
- Item 1 (shape-arithmetic prose + "Shape check…Correct." lines around Q+gate and KV projection blocks in `gated_attention_formulation.md`): confirmed removed.
- Item 2 (geometric gate and gate-before-norm explanations duplicated in `gated_vs_vanilla_attention_shapes.md` Differences 1–2): confirmed replaced with cross-references.
- Item 3 (14-line inline KV cache byte arithmetic in Difference 4): confirmed collapsed to one sentence with Section 3 cross-reference.

## MINOR Suggestions

### [gated_attention_formulation.md] lines 196 and 233 — two remaining "Shape check:" lines
The line `Shape check: \`sqrt(d_h) = sqrt(256) = 16.0\`` (after the prefill SDPA block) and `Shape check: \`n_q_h × d_h = 16 × 256 = 4096\`` (after the output projection block) are standalone arithmetic sentences. Both facts are already shown inside the preceding code blocks (`/ sqrt(d_h)` annotated inline; `= [B, T, 16 × 256] = [B, T, 4096]` shown in reshape step). These were flagged as MINOR in Pass 1 and remain unaddressed.
**Suggestion:** Delete both standalone "Shape check:" lines. Save ~2 lines.

### [gated_vs_vanilla_attention_shapes.md] lines 30 and 45 — "Shape check:" lines below MHA and GQA tables
`"Shape check: \`n_h × d_h = 16 × 128 = 2048\`. Output matches hidden size directly."` and `"Shape check: \`n_q_h × d_h = 16 × 128 = 2048\`. Expand factor: \`n_q_h / n_kv_h = 16 / 4 = 4×\`."` follow their respective tables. The Reshaped-output row in each table already shows `[B, T, 2048]`; the expand factor is derivable from the K/V row headers. Also flagged in Pass 1, still present.
**Suggestion:** Remove both post-table "Shape check:" lines. Save ~2 lines.

### [gated_vs_vanilla_attention_shapes.md] lines 11–16 — abbreviation legend duplicates index.md symbol table
The five-item legend (`n_h`, `n_q_h`, `n_kv_h`, `d_h`, `d_k`) restates definitions already in `index.md`'s symbol table (which `gated_attention_formulation.md` line 3 explicitly cross-references). `d_k` appears only once, in the Section 4 Delta Net comparison table.
**Suggestion:** Replace the legend block with a single cross-reference line to `index.md`. Define `d_k` inline in Section 4 where it first appears. Save ~6 lines.

### [gated_attention_formulation.md] line 128 — intent-attribution sentence
`"This is a deliberate design choice that reduces the fraction of head capacity consumed by positional encoding."` adds speculative authorial intent beyond the factual `"only 64 of 256 dimensions carry positional signal"` already stated in the prior sentence. Flagged in Pass 1, still present.
**Suggestion:** Delete the sentence. Save ~1 line.

### [gated_attention_formulation.md] line 167 — redundant post-block summary sentence
`"After this step all three tensors Q_rope, K_exp, V_exp have the same head count (16) and head dimension (256)."` restates what the three code lines immediately above it show explicitly (`K_exp: [B, T, 16, 256]`, `V_exp: [B, T, 16, 256]`, and Q_rope already at `[B, T, 16, 256]`).
**Suggestion:** Remove the sentence. Save ~1 line.

### [gated_vs_vanilla_attention_shapes.md] lines 102–103 — normalization rationale partially echoes formulation file
`"The per-head weight vectors \`q_norm_weight [256]\` and \`k_norm_weight [256]\` are learned and allow the model to independently rescale each dimension after normalization. Normalizing K before SDPA bounds the key vector norms, which stabilizes attention score magnitudes regardless of input scale."` The stabilization rationale echoes `gated_attention_formulation.md` Section 4 ("stabilizing the magnitude of the surviving signal"). The first sentence (about learned weight vectors) is factual and unique; the second sentence restates the stabilization reasoning.
**Suggestion:** Remove the second sentence ("Normalizing K before SDPA…"). Retain the first sentence about learned weight vectors as it adds specificity not present in the formulation file. Save ~1 line.

## Load-Bearing Evidence
- `index.md` line 8: `"It is **not** Gated Linear Attention (GLA). GLA is a separate recurrent architecture (related to linear attention); it does not appear in Qwen3.5-35B-A3B."` — the critical disambiguation; no redundant copy exists elsewhere in the chapter.
- `gated_attention_formulation.md` line 84: `"Each element of \`gate_sigmoid\` lies in (0, 1). Multiplying by Q entry-wise performs input-dependent suppression: query dimensions that the model deems irrelevant for the current token are scaled toward zero."` — canonical gate explanation; the comparison file now cross-references here rather than duplicating it.
- `gated_vs_vanilla_attention_shapes.md` line 114: `"This reduces KV cache size by 8× relative to MHA (at the same head dim), but places a strong structural constraint on what the KV representations must encode."` — load-bearing architectural consequence unique to this section.

## VERDICT
- Crucial updates: no
