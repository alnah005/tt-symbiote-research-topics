# Compression Analysis: Chapter 3 — partial_rotary_factor Promotion — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~175 lines (post-fix)
- Estimated pre-compression line count: ~199 lines
- Estimated reduction: ~12%

## CRUCIAL Suggestions

### [ttnn_rope_impact.md] ~lines 40–63 (pre-fix)
**Issue:** Section 4 duplicated the `AttributeError` failure description and the full two-line `getattr` guard code block already presented in Section 1 (lines 11–20). The closing paragraph additionally restated the promotion rationale already covered in Section 3. Combined redundancy was ~18 lines.
**Suggestion:** Collapse Section 4 to a short cross-reference back to Section 1 for the guard, retain only the diagnostic characterisation ("loud, immediate, easy to diagnose") and the promotion note that is not present in Section 1. Applied.

### [hf_config_resolution.md] ~line 73 (pre-fix)
**Issue:** The Key Finding blockquote restated the guard pattern in full — `getattr(config, "partial_rotary_factor", config.rope_scaling.get("partial_rotary_factor", 1.0))` — which was already shown as a runnable code block five lines earlier (lines 27–28 table area, and the Section 4 `transformers` discussion). The sentence also restated the "value is 0.25 in both locations" fact covered in the table at lines 33–38.
**Suggestion:** Shorten the Key Finding to a one-sentence reminder pointing back to the guard already shown, dropping the inline re-quote of the full expression. Applied.

## MINOR Suggestions

### [ttnn_rope_impact.md] ~line 38
**Issue:** "identical in every respect to one constructed with a **Qwen3.5** config" is verbose. "in every respect" adds no information beyond "identical."
**Suggestion:** Trim to "...produces an embedding module identical to one constructed with a **Qwen3.5** config..."

### [hf_config_resolution.md] ~line 47
**Issue:** The sentence starting "The `AttributeError` risk exists only in external consumer code..." restates the same restriction explained two sentences earlier in the same paragraph ("no `AttributeError` can occur inside `__init__` itself"). The double-negative framing forces re-reading.
**Suggestion:** Delete the second sentence and fold its content into the preceding sentence: "...so no `AttributeError` can occur inside `__init__` itself; the risk is in external code that bypasses `__init__` and reads `config.partial_rotary_factor` as a raw attribute."

### [index.md] ~line 11
**Issue:** "No TTNN cos/sin table shapes change." and "`TTNNRotaryPositionEmbedding` produces an identical embedding module for both checkpoints." are two separate sentences restating the same conclusion. One is sufficient at the index level.
**Suggestion:** Merge: "No TTNN cos/sin table shapes or `TTNNRotaryPositionEmbedding` outputs change."

### [hf_config_resolution.md] ~line 71 (pre-fix)
**Issue:** "Any consumer code that skips `Qwen3_5MoeConfig.__init__` and reads `config.partial_rotary_factor` via raw attribute access will fail on **Qwen3.5** configs because `PretrainedConfig`'s generic `setattr` path never set that attribute." is a third restatement in Section 4 of the same AttributeError condition already stated in Section 3 line 47 and Section 2 lines 13–17.
**Suggestion:** Remove this sentence from Section 4; Section 3 is the correct location for this warning.

## Load-Bearing Evidence
- `index.md` line ~15: "Any TT-Symbiote code that reads `config.partial_rotary_factor` as a bare top-level attribute will raise `AttributeError` on **Qwen3.5** configs" — load-bearing because this is the canonical statement of the one actionable risk; all other files derive from it.
- `hf_config_resolution.md` line ~41: "`Qwen3_5MoeConfig` — the HuggingFace config class used for both checkpoints — computes `rotary_dim` via..." — load-bearing because it identifies the specific class name and links the top-level attribute access to the rotary_dim computation path; needed to understand why the `AttributeError` matters.
- `ttnn_rope_impact.md` line ~36: "No TTNN op configuration changes, no re-sharding, no dtype changes are needed." — load-bearing because it is the explicit, unambiguous sign-off that the TTNN layer requires zero changes; removing or weakening it would undermine the chapter's conclusion.

## VERDICT
- Crucial updates: yes

## C Compression Application Log — Pass 1
- C1: `ttnn_rope_impact.md` Section 4 — replaced ~18 lines (duplicate `AttributeError` description, duplicate guard code block, restated promotion rationale) with 4-line cross-reference to Section 1. Lines 40–63 pre-fix → lines 40–47 post-fix.
- C2: `hf_config_resolution.md` Key Finding blockquote (~line 73) — removed inline re-quote of full guard expression and redundant restatement of "value is 0.25 in both locations." Sentence shortened from 3 clauses to 2, pointing to the guard shown above rather than repeating it.

---

# Compression Analysis: Chapter 3 — partial_rotary_factor Promotion — Pass 2

## Summary
- Total files analyzed: 3 (index.md, hf_config_resolution.md, ttnn_rope_impact.md)
- Estimated post-Pass-1 line count: ~150 lines
- Estimated post-Pass-2 line count: ~147 lines
- Estimated reduction this pass: ~2%

## CRUCIAL Suggestions

### [index.md] ~lines 15–19 (post-Pass-1)
**Issue:** The "One Actionable Risk" section in `index.md` reproduced the full multi-line guard expression as a fenced code block — a word-for-word copy of the guard already shown as a runnable code block in `ttnn_rope_impact.md` Section 1 (lines 14–20). `index.md` is a chapter overview whose Contents section already points to `ttnn_rope_impact.md` as the canonical location for the guard pattern; duplicating the code block here is pure restatement.
**Suggestion:** Replace the 4-line fenced code block with a one-line cross-reference to `ttnn_rope_impact.md` Section 1. **APPLIED as C3.**

## MINOR Suggestions

### [index.md] ~line 11
**Issue:** "No TTNN cos/sin table shapes change." and "`TTNNRotaryPositionEmbedding` produces an identical embedding module for both checkpoints." are two separate sentences expressing the same conclusion. At the index level, one is sufficient.
**Suggestion:** Merge into a single sentence, e.g. "No TTNN cos/sin table shapes or `TTNNRotaryPositionEmbedding` outputs change." Do NOT apply now.

### [hf_config_resolution.md] ~line 49
**Issue:** Section 3 ends with a second full statement of the `AttributeError` scope ("The `AttributeError` risk exists only in external consumer code…"), which restates the same restriction already stated two sentences earlier in the same paragraph. The double statement forces re-reading without adding information.
**Suggestion:** Delete the second sentence and fold its constraint into the preceding sentence's trailing clause. Do NOT apply now.

## Load-Bearing Evidence
- `index.md` line 8: `` rotary_dim = int(head_dim × partial_rotary_factor) = int(128 × 0.25) = 32 `` — load-bearing at the index level; this is the chapter's primary numeric summary and appears here in a display-formula fenced block, not as a code snippet, so it is not a duplicate of the table row in `ttnn_rope_impact.md`.
- `ttnn_rope_impact.md` lines 14–20: full guard code block with `prf` variable — load-bearing as the single canonical, runnable statement of the guard pattern; all other files now point here rather than repeat it.
- `hf_config_resolution.md` lines 33–39: access-path table (`config.partial_rotary_factor`, `config.rope_scaling["partial_rotary_factor"]`, `config.rope_scaling.get(…)` × both checkpoints) — load-bearing because it is the only location that enumerates all three access paths side-by-side; unique content not duplicated elsewhere.

## VERDICT
- Crucial updates: **yes**

## C Compression Application Log — Pass 2
- C3: `index.md` "The One Actionable Risk" section — removed 4-line fenced code block reproducing the guard expression verbatim; replaced with inline cross-reference to `ttnn_rope_impact.md` Section 1. Net removal: 4 lines (the code fence + blank lines around it) → 0 lines; surrounding prose paragraph condensed by one sentence fragment.
