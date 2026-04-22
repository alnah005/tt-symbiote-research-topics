# Cross-Chapter Compression Analysis — Pass 1

## Summary
- Files reviewed: 6 index files (guide index + ch1–ch5 index files)
- Estimated current line count: ~52 (guide) + ~43 (ch1) + ~117 (ch2) + ~74 (ch3) + ~27 (ch4) + ~36 (ch5) = ~349 lines total

## CRUCIAL Suggestions

### CRUCIAL-1: "Critical Facts for tt_symbiote Integrators" in guide index duplicates ch4 and ch5 content nearly verbatim

The guide-level `index.md` contains a five-bullet "Critical Facts for tt_symbiote Integrators" section (lines 41–52). Three of those bullets restate content that the chapter indexes already own:

- **TP hard-cap at 2 / `gcd(12,2)=2` / 8-device reservation:** The guide index explains this in prose (lines 43–45). Ch4 `index.md` owns exactly this derivation — the Quick Reference table explicitly states `gcd(12, 2) = 2 → TP ≤ 2`, "6 devices are idle at TP=2", and the submesh shape. The guide-level bullet adds no new information; it restates the same numbers in the same terms.
- **PCC >0.99 is a target, not confirmed / run `test_vision_tower_pcc.py` and `test_e2e_pcc.py`:** The guide index (lines 49–50) says this nearly word-for-word. Ch5 `index.md` "Recommendation for tt_symbiote Integrators" (lines 30–31) says the same thing: "Run `test_vision_tower_pcc.py` and `test_e2e_pcc.py` on target hardware before declaring the model production-ready." The phrase "before promoting to production, run `test_vision_tower_pcc.py` and `test_e2e_pcc.py`" is substantively identical across both files.
- **Commit 6 / residual `Qwen*` symbols:** The guide index (lines 51–52) warns about Commit 6 being incomplete and advises grepping for `Qwen*` symbols. Ch5 `index.md` (line 33) says the same: "Verify no residual `Qwen*` names in `tt/` or `reference/`; see the Commit 6 warning in `commit_history_and_stabilization.md`."

**Fix:** The guide-level "Critical Facts" section should be replaced with a concise navigation pointer (e.g., "Key integration constraints are summarized in Chapter 4 and Chapter 5") or reduced to one-line callouts that reference the owning chapters. The full prose explanations belong only in the chapter indexes and their detail files.

---

### CRUCIAL-2: TP cap `gcd(12,2)=2` stated three times across guide index, ch4 index, and ch5 index

- Guide `index.md` Quick Summary table (line 16): `Max TP on T3K | 2 (gcd(12,2)=2)`
- Guide `index.md` Critical Facts bullet (line 43): "The GQA configuration (12 query heads, 2 KV heads) yields `gcd(12,2)=2`, so the maximum tensor-parallel degree is 2."
- Ch4 `index.md` Quick Reference table (lines 19–21): `num_key_value_heads | 2 | TP ∈ {1, 2} only` and `num_attention_heads | 12 | gcd(12, 2) = 2 → TP ≤ 2`

The guide Quick Summary table (one-line entry) is acceptable as a navigation aid. The full prose in the guide's Critical Facts bullet is the duplication — it restates ch4's derivation word-for-word with no additional framing. The fix for CRUCIAL-1 above resolves this simultaneously.

---

### CRUCIAL-3: `image_token_id=151665` difference stated both in guide index and ch1 index comparison table

- Guide `index.md` Quick Summary table (line 19): `image_token_id | 151665 (Qwen2.5-VL uses 151655 — different by 10)`
- Guide `index.md` Critical Facts bullet (lines 47–48): Full prose explanation of the off-by-10 risk.
- Ch1 `index.md` comparison table (line 31): `image_token_id | 151665 | 151655`

The Quick Summary table one-liner is defensible as a navigation signal. The full prose "silently incorrect output" warning in the guide's Critical Facts section is a near-verbatim restatement of what ch1 detail files own. Again, resolved by CRUCIAL-1's fix.

---

### CRUCIAL-4: PCC confirmation statement duplicated verbatim between guide index and ch5 index

- Guide `index.md` Quick Summary table (line 17): `Confirmed PCC | >0.98 text decoder prefill (commit "prefill at 0.98")`
- Guide `index.md` Quick Summary table (line 18): `Target PCC | >0.99 per IMPLEMENTATION_STEPS.md (not confirmed by commit history)`
- Ch5 `index.md` Status Dashboard (line 21): `Text decoder PCC > 0.99 | Target (confirmed >0.98) | commit "prefill at 0.98"`

The Quick Summary table entry is a one-liner and acceptable in the guide. The Critical Facts bullet (lines 49–50) expands on this again in prose ("The only commit-verified figure is >0.98 for text decoder prefill (commit message 'prefill at 0.98')"), which is then restated in ch5 Status Dashboard in essentially identical terms. Resolved by CRUCIAL-1's fix.

---

## MINOR Suggestions

### MINOR-1: Ch3 "Prerequisites" sentence lists hyperparameters already navigable via ch1

Ch3 `index.md` (lines 7–8) states: "Prerequisites: Chapter 1 (`vision_encoder_specs.md`) for the `post_norm=True` flag, `patch_size=14`, `spatial_merge_size=2`, and the 42-layer depth." This is fine as a cross-reference pointer. However, the component flow diagram in ch3 also re-embeds `hidden_size=1536` and the `1536` shape annotations, which are load-bearing for understanding the diagram and should stay.

### MINOR-2: Ch2 and ch3 both explain PatchMergerTT reuse from qwen25_vl

- Ch2 `index.md` (lines 111–113): "tt/patch_merger.py is reused directly from the qwen25_vl port (PatchMergerTT). This is consistent with the architectural lineage described in Chapter 1."
- Ch3 `index.md` Overview (line 5): "the reuse of `PatchMergerTT` from the `qwen25_vl` port" and component flow diagram labels it.

Ch2 establishes the reuse fact; ch3 legitimately references it to explain the full vision stack. The ch3 mention is brief and additive (the Overview sentence sets up the flow diagram). This is minor and context-dependent — removing it from ch3's one-sentence overview would leave the diagram label unexplained.

### MINOR-3: Guide index "Reading Order" section partly restates the Chapter Guide table

The "Reading Order" section (lines 31–39) adds prose guidance beyond the table (jump-to-ch4, jump-to-ch5 callouts), so it is not pure duplication. However, the sentence "Chapter 1 is the prerequisite for all others; it establishes the exact hyperparameters and the relationship to Qwen 2.5 VL" repeats the ch1 index Overview's first sentence ("This chapter establishes the complete model specification…"). This is minor.

## Load-Bearing Evidence

The following content must stay in multiple places because it is navigation-critical or structurally necessary:

- **Guide index Quick Summary table** — One-liner parameter/config values are a legitimate navigation hub. The table format (not the Critical Facts prose) is the right vehicle for this.
- **TP=2 and `gcd(12,2)=2` in the Quick Summary table (guide index)** — The one-liner entry is a navigation signal pointing readers to ch4. It does not duplicate the full derivation; it summarizes it.
- **Ch1 comparison table** — The full `dots.ocr vs Qwen2.5-VL` table must stay in ch1; it is the primary reference for all other chapters that cite specific hyperparameters.
- **Ch4 Quick Reference table** — Owns the topology constraint derivation; must stay.
- **Ch5 Status Dashboard** — Owns the milestone audit; must stay.
- **Cross-reference pointers** (ch3's "Prerequisites: Chapter 1…", ch2's note about ch1 architectural lineage) — These are navigational and should not be removed.

## VERDICT
- Crucial updates: yes

---

# Cross-Chapter Compression Analysis — Pass 2

## Pass 1 Fix Verification

**CRUCIAL-1 (guide index "Critical Facts for tt_symbiote Integrators" duplicating ch4/ch5 prose):** Fixed. The five bullets in the current `index.md` (lines 43–52) have each been reduced to 1–2 sentences with an explicit chapter cross-reference appended. No bullet now re-derives or re-explains content owned by a chapter — each bullet states the essential constraint and delegates to the owning chapter link.

Specific bullets verified:
- TP cap bullet (line 43): States `gcd(12,2)=2` hard-cap and links to Ch4 for derivation and submesh setup. No prose re-derivation.
- 8-device reservation bullet (line 45): States the scheduling implication and links to Ch4 `t3k_submesh_and_env_vars.md`. No prose re-explanation.
- `image_token_id` bullet (line 47): States the off-by-10 difference as a one-liner operational warning. No verbose explanation of failure mechanism.
- PCC bullet (lines 49–50): States the confirmed vs. target distinction in one sentence and links to Ch5. No elaboration of test methodology.
- Commit 6 bullet (lines 51–52): States the risk and directs the reader to Ch5 `commit_history_and_stabilization.md`. No re-description of the commit.

**CRUCIAL-2, CRUCIAL-3, CRUCIAL-4:** All were contingent on CRUCIAL-1's fix. Confirmed resolved — no residual full-prose duplication of the TP derivation, `image_token_id` failure mechanism, or PCC confirmation language exists in the guide index.

## CRUCIAL Suggestions

None identified.

No full prose blocks are duplicated across the three reviewed files with no added value. Cross-chapter references between ch4 and ch5 (e.g., ch5's "Register dots.ocr as an 8-device workload") are single-phrase pointers that delegate to ch4's detail files, not re-explanations.

## MINOR Suggestions

### MINOR-P2-1: Ch5 "Recommendation" bullet slightly overlaps ch4 scope

Ch5 `index.md` line 32 states "Register dots.ocr as an 8-device workload; see `tt_symbiote_integration_gaps.md` integration checklist." This scheduling fact is derived from ch4's topology analysis. The bullet is one line and cross-references the correct detail file, so it does not constitute a CRUCIAL duplication. However, the phrasing could be tightened to "Register as an 8-device workload (derived from TP=2 submesh; see Ch4 and `tt_symbiote_integration_gaps.md`)" to make the dependency chain explicit rather than implied. Low priority.

## VERDICT
- Crucial updates: no
