# Compression Analysis: Chapter 2 — Gated DeltaNet Deep Dive — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~262 + ~252 + ~223 = ~737 lines
- Estimated post-compression line count: ~590 lines
- Estimated reduction: ~20%

---

## CRUCIAL Suggestions

### C1 — Decay gate re-derivation duplicated across two files
**Files:** `delta_rule_formulation.md` §4 (lines 115–143) and `comparison_to_linear_attention_variants.md` §2.3 Mamba2 (lines 69–78) and §5 "Mamba2 Relationship" (lines 199–207).

The full formula $g_t = \exp(-\exp(A_{\log}) \cdot \text{softplus}(a_t + \text{dt\_bias}))$ is written out three times. The 5-step sign-analysis bullet list ("why $\alpha_t < 0$ is guaranteed") in `delta_rule_formulation.md` §4 (lines 130–136) appears once in full; `comparison_to_linear_attention_variants.md` §2.3 then re-states the same guarantee in prose ("Because $\exp(A_{\log}) > 0$ and softplus is always positive…") for a second full derivation; §5 "Mamba2 Relationship" then repeats the formula a third time as the opening equation.

**Recommended action:** Keep the canonical derivation in `delta_rule_formulation.md` §4. In `comparison_to_linear_attention_variants.md` §2.3, replace the re-derivation with a one-sentence cross-reference: "The decay gate uses the same $\exp(-\exp(A_{\log}) \cdot \text{softplus}(\cdot))$ parameterization as Mamba2 (see `delta_rule_formulation.md` §4)." In §5 "Mamba2 Relationship," remove the repeated formula block — the prose that follows it is sufficient.

**Estimated saving:** ~15 lines.

---

### C2 — Gated RMSNorm formula duplicated verbatim across two files
**Files:** `delta_rule_formulation.md` §6 (lines 181–199) and `head_asymmetry_and_projections.md` §2.2 (lines 98–101).

The formula
$$\text{output}_t = \frac{o_t}{\sqrt{\text{mean}_{d_v}(o_t^2) + \varepsilon}} \cdot w_{\text{norm}} \cdot \text{SiLU}(z_t)$$
is written out identically in both files. `head_asymmetry_and_projections.md` §2.2 follows this with the sentence "The SiLU gate allows the model to dynamically suppress or amplify each output dimension based on the input context," which also repeats the same explanation given in `delta_rule_formulation.md` §6 (lines 196–199).

**Recommended action:** `head_asymmetry_and_projections.md` §2.2 should drop the re-stated formula and the explanatory sentence, replacing both with a cross-reference: "See `delta_rule_formulation.md` §6 for the full gated RMSNorm derivation." The projection shape table and $z_t$ reshape description in §2.2 are distinct and should be kept.

**Estimated saving:** ~6 lines.

---

### C3 — "Scalar gate more natural for delta-rule" argument repeated twice in the same file
**File:** `comparison_to_linear_attention_variants.md` §3 key design choice 1 (lines 115–120) and §5 "Favorable Comparison to GLA" (lines 183–195).

§3 point 1 reads: "With a full matrix gate, different (key-dim, value-dim) entries decay at different rates, making the 'predicted value under the decayed state' a direction-dependent mixture that is harder to correct precisely with a single rank-1 write."

§5 "Favorable Comparison to GLA" repeats this almost word-for-word (lines 188–191): "With a full matrix gate, different key dimensions decay at different rates, making the 'predicted value under the decayed state' a direction-dependent mixture that is harder to correct precisely."

**Recommended action:** Keep the version in §3 (the conceptual justification section). In §5, replace lines 186–195 ("The scalar gate is both simpler…cannot achieve: stale associations…") with one sentence that back-references §3: "As noted in §3, the scalar gate is the more natural pairing for the delta-rule correction — see §3 key design choice 1 for the direction-mixing argument."

**Estimated saving:** ~10 lines.

---

### C4 — "Neither alone is sufficient" motivation restated from `delta_rule_formulation.md`
**Files:** `delta_rule_formulation.md` §1 (lines 18–29) and `comparison_to_linear_attention_variants.md` §3 point 3 (lines 128–134) and §5 "Long-Range Retrieval" (lines 169–177).

`delta_rule_formulation.md` §1 already explains in full why GLA-style gating and the delta rule each solve only half the problem and how Gated DeltaNet combines them. §3 point 3 in the comparison file then re-states "Neither alone is sufficient: Mamba2 can forget but cannot precisely overwrite; DeltaNet can precisely overwrite but cannot globally flush. GLA has richer per-entry decay but still uses a raw outer-product write with no error correction." §5 "Long-Range Retrieval" repeats this a third time for Qwen3.6 specifically.

**Recommended action:** §3 point 3 should be trimmed to: "Combines the strengths of Mamba2 (data-dependent scalar forgetting) and DeltaNet (targeted delta-rule writes) — see `delta_rule_formulation.md` §1 for motivation." §5 "Long-Range Retrieval" can keep its Qwen3.6-specific framing (the context-window saturation argument is specific to this model) but should drop the re-statement of the generic trade-off already covered in §1 and §3.

**Estimated saving:** ~8 lines.

---

### C5 — State size (≈ 2 MB per layer) restated in two files
**Files:** `delta_rule_formulation.md` §7 (lines 220–257) provides the canonical memory analysis. `comparison_to_linear_attention_variants.md` §5 "Hardware Suitability" (lines 216–218) re-states "the $[32, 128, 128]$ state at fp32 is ≈ 2 MB per layer."

**Recommended action:** Replace the hardware suitability bullet with a cross-reference: "State memory: ≈ 2 MB per layer (fp32) — see `delta_rule_formulation.md` §7 for the full breakdown vs KV cache." No other content in the hardware suitability bullet is duplicated.

**Estimated saving:** ~2 lines.

---

## MINOR Suggestions

### M1 — "GQA sharing convention" restated within one section
**File:** `head_asymmetry_and_projections.md` §1 (lines 28–31 and 47–50).

Lines 28–31 state: "This matches the GQA pattern used in the Gated Attention (softmax) layers of the same model." Lines 47–50 then re-state: "This is the same GQA sharing convention used in the Gated Attention layers, applied here to the linear attention recurrence." Both sentences carry the same information. Remove one.

**Estimated saving:** ~2 lines.

---

### M2 — Verbose dimensional consistency walkthrough
**File:** `delta_rule_formulation.md` §2 "Dimensional consistency" (lines 57–63).

The five bullet points each spell out a sub-expression's shape. This is useful but over-explains steps that follow directly from reading the formula: e.g., "scalar times matrix" and "matvec retrieval" are self-evident to a reader who has seen the symbol table immediately above. The two less-obvious steps (the outer product producing a rank-1 matrix, and the scaled query producing a $d_v$ output) can be retained; the three trivial steps can be folded into one line.

**Estimated saving:** ~3 lines.

---

### M3 — Hedging language in `comparison_to_linear_attention_variants.md` §5
**File:** `comparison_to_linear_attention_variants.md` §5 "Long-Range Retrieval" (lines 168–177).

"Qwen3.6 is designed for long-context applications (the model supports up to 32K or 128K tokens depending on configuration)." The parenthetical "depending on configuration" is vague and adds no actionable information — the reader is pointed to no source. Either state the exact configuration or drop the parenthetical.

**Estimated saving:** ~1 line.

---

### M4 — Conv1d decode pseudo-code partially re-narrates the preceding prose
**File:** `head_asymmetry_and_projections.md` §3 Decode (lines 204–211).

The numbered list (steps 1–4, lines 204–207) re-states in prose what the formula on line 211 already expresses mathematically. The two together are redundant; the numbered list is more readable and the formula adds precision. Keep the formula; condense the numbered list to a single sentence ("Each step writes the new projection into the oldest slot, advances the pointer, and computes a weighted sum followed by SiLU.").

**Estimated saving:** ~4 lines.

---

## Load-Bearing Evidence

Not applicable — VERDICT is yes (crucial duplications found).

---

## VERDICT
- Crucial updates: yes

---

## Change Log

### Pass 1 compression applied — 2026-04-21

All 5 CRUCIAL suggestions applied by Agent A.

**C1** — `comparison_to_linear_attention_variants.md` §2.3 Mamba2: replaced 6-line re-derivation
(formula + prose sign guarantee) with a one-sentence cross-reference to `delta_rule_formulation.md`
§4. §5 "Mamba2 Relationship": removed the standalone formula block (`$$g_t = \exp(-\exp(A_{\log})
\cdot \text{softplus}(\cdot))$$`) and replaced with an inline cross-reference; surrounding prose
retained unchanged. Canonical derivation in `delta_rule_formulation.md` §4 untouched.

**C2** — `head_asymmetry_and_projections.md` §2.2: removed the verbatim gated RMSNorm formula and
the "SiLU gate allows the model to dynamically suppress…" explanatory sentence; replaced with a
cross-reference to `delta_rule_formulation.md` §6. The `z_t` reshape description and projection
shape table were preserved.

**C3** — `comparison_to_linear_attention_variants.md` §5 "Favorable Comparison to GLA": collapsed
the ~10-line scalar-gate direction-mixing argument (near-verbatim repeat of §3 key design choice 1)
to a two-sentence back-reference to §3. §3 version kept intact.

**C4** — `comparison_to_linear_attention_variants.md` §5 "Long-Range Retrieval": removed the
restatement of the "neither alone is sufficient" trade-off (GLA/Mamba2 can forget but not overwrite;
DeltaNet can overwrite but not forget). Replaced with a cross-reference to `delta_rule_formulation.md`
§1 and §3. The Qwen3.6-specific framing and the synthesis paragraph were retained. Also resolved
the M3 vagueness noted in MINOR suggestions: "32K or 128K tokens depending on configuration" replaced
with "up to 128K tokens."

**C5** — `comparison_to_linear_attention_variants.md` §5 "Hardware Suitability": replaced bare
"the $[32, 128, 128]$ state at fp32 is ≈ 2 MB per layer" with a cross-reference to
`delta_rule_formulation.md` §7; the surrounding context sentence was kept.

---

# Compression Analysis: Chapter 2 — Gated DeltaNet Deep Dive — Pass 2

## Summary
- Pass 1 CRUCIAL items resolved: 5 / 5
- Remaining crucial redundancy: none found
- Minor open items carried forward from Pass 1: M1, M2, M4 (M3 was resolved in Pass 1)
- New minor items identified: 1

---

## Re-check of Pass 1 CRUCIAL Items

**C1 — RESOLVED.** `comparison_to_linear_attention_variants.md` §2.3 now contains a one-sentence cross-reference to `delta_rule_formulation.md` §4 in place of the re-derivation. §5 "Mamba2 Relationship" contains an inline reference ("the formula is derived in full in `delta_rule_formulation.md` §4") with no standalone formula block. Canonical derivation in `delta_rule_formulation.md` §4 untouched.

**C2 — RESOLVED.** `head_asymmetry_and_projections.md` §2.2 (line 97) now reads: "See `delta_rule_formulation.md` §6 for the full formula and derivation." The verbatim RMSNorm formula and the "SiLU gate allows the model to dynamically suppress…" sentence are absent.

**C3 — RESOLVED.** `comparison_to_linear_attention_variants.md` §5 "Favorable Comparison to GLA" collapses the direction-mixing argument to one sentence that back-references §3 key design choice 1.

**C4 — RESOLVED.** `comparison_to_linear_attention_variants.md` §5 "Long-Range Retrieval" replaces the generic "neither alone is sufficient" restatement with a cross-reference to `delta_rule_formulation.md` §1 and §3. The Qwen3.6-specific 128K framing is preserved (M3 also resolved here).

**C5 — RESOLVED.** `comparison_to_linear_attention_variants.md` §5 "Hardware Suitability" bullet now reads: "State memory: ≈ 2 MB per layer (fp32) — see `delta_rule_formulation.md` §7 for the full breakdown and comparison to KV cache."

---

## VERDICT
- Crucial updates: no

---

## Load-Bearing Evidence

- **`delta_rule_formulation.md` §4, lines 130–136** — The 5-step sign-analysis list ("why $\alpha_t < 0$ is guaranteed") is the canonical, first-occurrence derivation. Every other file now cross-references this section rather than repeating it. Cutting any of these five bullets would remove the only complete logical proof that $g_t \in (0,1)$ unconditionally; no other file carries it after Pass 1 edits.

- **`head_asymmetry_and_projections.md` §1, lines 35–50** — The `repeat_interleave` code block and the head-expansion prose are the only location in Ch. 2 explaining exactly how 16 Q/K heads expand to 32 via `repeat_interleave(gqa_ratio, dim=2)` and which head indices map to which. The description in §4 summary table (line 243) references this expansion but does not re-derive it, so the §1 block cannot be cut.

- **`comparison_to_linear_attention_variants.md` §2, lines 29–101** — The four variant formulations (RetNet, GLA, Mamba2, standard DeltaNet) each appear only once in the entire chapter. The §2 subsections are the only location where the general gated form $S_t = G_t \odot S_{t-1} + k_t v_t^\top$ is unpacked per-variant with the precise $G_t$ structure identified and its limitations stated. The summary table in §4 compresses these to a row each but omits the derivations (e.g. the DeltaNet algebraic equivalence proof at lines 85–88); the §2 subsections are load-bearing.

---

## MINOR Suggestions

### M1 — "GQA sharing convention" repeated within one section (carried from Pass 1, not yet applied)
**File:** `head_asymmetry_and_projections.md` §1.

Line 29–30: "This matches the GQA pattern used in the Gated Attention (softmax) layers of the same model."
Line 48–49: "This is the same GQA sharing convention used in the Gated Attention layers, applied here to the linear attention recurrence."

Both sentences deliver the same cross-layer GQA analogy. Remove the second instance (lines 48–49) — the first occurrence is immediately in context where it motivates the design choice; the repeat in the `repeat_interleave` sub-section adds nothing new.

**Estimated saving:** ~1 line.

---

### M2 — Dimensional consistency bullets partially self-evident (carried from Pass 1, not yet applied)
**File:** `delta_rule_formulation.md` §2 "Dimensional consistency" (lines 59–63).

The five-bullet walkthrough spells out "scalar times matrix," "matvec retrieval," "scalar times vector," "rank-1 outer product," and "matvec retrieval with scaled query." The first, third, and fifth are self-evident from the symbol table that immediately precedes them. The two non-trivial steps (rank-1 outer product producing a $[d_k, d_v]$ matrix, and the scaled query producing a $d_v$ output) should be kept. Fold the three obvious steps into a single sentence: "The decay ($g_t \cdot S_{t-1}$) and scalar scaling ($\beta_t \cdot (\cdots)$) are straightforward; the non-trivial step is the outer product $\tilde{k}_t (\cdots)^\top \in \mathbb{R}^{d_k \times d_v}$ adding a rank-1 matrix to the state, with output read as $S_t^\top (\tilde{q}_t/\sqrt{d_k}) \in \mathbb{R}^{d_v}$."

**Estimated saving:** ~3 lines.

---

### M4 — Conv1d decode numbered list re-narrates its own formula (carried from Pass 1, not yet applied)
**File:** `head_asymmetry_and_projections.md` §3 Decode (lines 197–207).

Steps 1–4 in prose ("The new QKV projection for the current token is written into the oldest slot… The shift register pointer advances… The weighted sum of all 4 slots is computed… SiLU is applied") describe exactly what the formula on line 207 expresses mathematically. Keep the formula; replace the four-step list with one sentence: "Each decode step writes the new projection into the oldest slot of the shift register (in-place, for Metal Trace compatibility), advances the pointer, then computes the weighted sum followed by SiLU."

**Estimated saving:** ~4 lines.

---

### M5 — "Decode/Prefill" section header duplication of labels in conv1d context (new)
**File:** `head_asymmetry_and_projections.md` §3, lines 173–188.

The §3 "Conv1d Local Mixing" section uses three labeled sub-sections: "Purpose," "Prefill (Batch Mode)," and "Decode (Shift Register)." The word "prefill" appears both in the header "Prefill (Batch Mode)" and in the opening sentence of that sub-section ("During prefill, the conv1d operates in standard `F.conv1d` mode over the full sequence"). The label in the header makes the opening sentence's "During prefill" phrase redundant — the sentence could simply begin "The conv1d operates in standard `F.conv1d` mode over the full sequence" and lose nothing. Same pattern in the Decode sub-section: "During autoregressive decode, only one token is processed at a time (T = 1)" — the phrase "During autoregressive decode" echoes the section header "Decode (Shift Register)" immediately above it. Drop the redundant framing phrase from each opening sentence.

**Estimated saving:** ~1 line (half-line savings ×2, rounding up).

---

## Change Log

### Pass 2 re-check completed — 2026-04-21

All 5 CRUCIAL items from Pass 1 confirmed resolved. No new CRUCIAL redundancy found. Four MINOR suggestions remain open (M1, M2, M4 carried forward; M5 newly identified). No edits applied in Pass 2 — verdict is "Crucial updates: no."
