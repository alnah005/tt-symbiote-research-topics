# Cross-Chapter Agent B Review — Pass 1

## Issues Found: 1

### Issue 1: Text decoder parameter count understated and total range inconsistent with guide index
**File:** `ch1_model_architecture/index.md`
**Claim:** Table row (line 38): `"~2.7–3.0B (~1.7B LLM + ~1.2B vision)"`; body text (line 7): `"approximately 2.7–3.0B total"`
**Correct fact:** The verified text decoder count is ~1.78B (~1,777M), not ~1.7B. The guide-level `index.md` (Quick Summary table) already states the correct figures: `"~3.0B (~1.78B text decoder + ~1.22B vision encoder)"`. The lower bound of `~2.7B` in ch1 is unsupported — with 1.78B text + 1.22B vision the total is ~3.0B, not as low as 2.7B.
**Fix:** In `ch1_model_architecture/index.md`, update the comparison table cell and the body-text sentence to match the guide-level index: replace `"~2.7–3.0B (~1.7B LLM + ~1.2B vision)"` with `"~3.0B (~1.78B text decoder + ~1.22B vision encoder)"`, and replace `"approximately 2.7–3.0B total"` with `"approximately 3.0B total"`.

---

## VERDICT: Needs revision

---

# Cross-Chapter Agent B Review — Pass 2

**Reviewer:** Agent B (Factual Reviewer)
**Date:** 2026-04-22
**Scope:** Re-check of Pass 1 fix; full factual sweep of all three files against VERIFIED FACTS.

## Pass 1 Fix Verification

**ch1_model_architecture/index.md** — fix applied correctly.

- Body text (line 7): now reads "~1.78B" (text decoder) and "approximately 3.0B total". Correct.
- Comparison table (line 38): now reads `~3.0B (~1.78B text decoder + ~1.22B vision)`. Correct.
- Both instances that Pass 1 flagged are resolved. No residual "~1.7B" or "~2.7B" language remains.

## Factual Sweep — All Three Files

### guide index.md

| Claim | Verdict |
|-------|---------|
| Total params "~3.0B (~1.78B text decoder + ~1.22B vision encoder)" (line 13) | Correct |
| Text decoder "28 layers, hidden_size=1536, GQA 12Q/2KV, attention_bias=True" (line 14) | Correct |
| Vision encoder "42-layer ViT, post_norm=True, patch_size=14, spatial_merge_size=2" (line 15) | Correct |
| Max TP on T3K "2 (gcd(12,2)=2)" (line 16) | Correct |
| Confirmed PCC ">0.98 text decoder prefill" (line 17) | Correct |
| Target PCC ">0.99 per IMPLEMENTATION_STEPS.md (not confirmed by commit history)" (line 18) | Correct |
| image_token_id "151665 (Qwen2.5-VL uses 151655 — different by 10)" (line 19) | Correct |

### ch1_model_architecture/index.md

| Claim | Verdict |
|-------|---------|
| Text decoder ~1.78B (line 7) | Correct — fix applied |
| Total ~3.0B (lines 7 and 38) | Correct — fix applied |
| Vision encoder ~1.22B (lines 7 and 38) | Correct |
| image_token_id dots.ocr=151665, Qwen=151655 (line 31) | Correct |
| temporal_patch_size dots.ocr=1, Qwen2.5-VL-7B=2 (line 37) | Correct — documented architectural difference |
| Vision layers dots.ocr=42, Qwen2.5-VL-7B=32 (line 33) | Correct |
| hidden_size dots.ocr=1536, Qwen2.5-VL-7B=3584 (line 25) | Correct |

### ch5_implementation_status_and_deployment/index.md

| Claim | Verdict |
|-------|---------|
| Step 3 status "Target (confirmed >0.98)" (line 21) | Correct — properly distinguishes confirmed from targeted |
| PCC >0.99 not confirmed by commit; >0.98 confirmed by "prefill at 0.98" commit | Correct — consistent with VERIFIED FACTS |
| Step 5 "Demo works with vision_backbone hf" (hybrid, not full TTNN) (line 23) | Correct |
| Step 6 "In progress" / residual Qwen cleanup warning (line 24) | Correct |

## Issues Found: 0

No factual errors detected in any of the three files. All parameter counts, token IDs, topology constraints, and PCC figures are consistent with the VERIFIED FACTS list and internally consistent across chapters.

## VERDICT: Approved
