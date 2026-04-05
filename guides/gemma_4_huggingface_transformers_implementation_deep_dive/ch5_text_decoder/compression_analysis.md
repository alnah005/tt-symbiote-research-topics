# Compression Analysis -- Chapter 5: Text Decoder

**Pass**: 1
**Analyst**: Agent C (Compressor)
**Date**: 2026-04-05
**Scope**: Redundancy, bloat, duplicate explanations, verbose prose. NOT factual corrections.

---

## Crucial updates: no

### Load-Bearing Evidence

- **index.md**: Every technical detail (dual head dims, K=V mode, KV sharing, double-width MLP, per-layer embeddings, RoPE buffers, layer scalar, attention scaling, bidirectional masking) is architecturally specific and referenced downstream by TTNN porting notes. Removing any section would break the chapter's self-contained narrative.
- **moe_details.md**: The router forward steps (norm -> scale -> project -> softmax -> topk -> renormalize -> per_expert_scale), the fused `gate_up_proj` weight layout, and the `index_add_` scatter-accumulate pattern are unique to this file and not covered in index.md. The TTNN porting notes for expert parallelism, token routing, and sparse activation are MoE-specific and not duplicated elsewhere.

---

## Redundancy and Duplication Found

### R1 -- MLP+MoE parallel path explained three times (CROSS-FILE)

The MLP+MoE parallel combination is described in three overlapping locations:

1. **index.md Section 5.5, Stage 2** (lines 296-313): Full pseudocode of the MLP and MoE parallel paths with post-norm and combination.
2. **index.md Section 5.5, "Key detail" paragraph** (line 314): Prose re-explaining the same parallel structure that the pseudocode just showed.
3. **moe_details.md, "MoE Data Path Overview"** (lines 8-43): ASCII diagram and bullet list covering the same MLP+MoE parallel path.
4. **moe_details.md, "How MLP and MoE Outputs Are Combined"** (lines 163-191): A near-verbatim repeat of index.md Stage 2 pseudocode, with the same code block and the same "parallel, not sequential" explanation.

**Suggestion**: In index.md Stage 2, keep the pseudocode but remove the "Key detail" paragraph (line 314) and replace it with a single-line forward reference: "See [MoE Details](moe_details.md) for the full MoE data path." In moe_details.md, remove the "How MLP and MoE Outputs Are Combined" section entirely (lines 163-191) -- the overview diagram at the top already covers this, and the index.md pseudocode is the canonical location. This eliminates ~35 lines of pure duplication.

**Severity**: MINOR -- no information loss, purely redundant prose and code.

### R2 -- TTNN parallel scheduling mentioned twice (CROSS-FILE)

- **index.md TTNN item 7** (line 446): "On TTNN, these could be scheduled concurrently if device resources allow."
- **moe_details.md TTNN item 5** (line 241): "On multi-core TTNN hardware, these could execute concurrently."

Both say the same thing about MLP+MoE concurrent execution.

**Suggestion**: In index.md TTNN item 7, trim to a forward reference: "The MLP and MoE run in parallel from the same residual. See [MoE Details](moe_details.md) TTNN considerations for parallelism strategy." Remove the scheduling sentence from index.md since moe_details.md covers it with more context.

**Severity**: MINOR -- one sentence of duplication.

### R3 -- "five RMSNorms" stated twice in moe_details.md

- Line 43: "This means MoE layers have **five** RMSNorm instances in the feedforward block alone..."
- Lines 226-227 (Additional RMSNorm Overhead section): "Each MoE-enabled layer adds three extra RMSNorm instances..."
- Line 249 (TTNN item 9): "MoE layers have five separate RMSNorm applications in the feedforward block."

The five-norm count appears at the top of the file and again in two later sections.

**Suggestion**: Keep the count in the overview (line 43) and in the TTNN note (line 249, since it's actionable). Remove lines 226-228 ("Additional RMSNorm Overhead" paragraph) or fold it into the overview bullet. Saves ~4 lines.

**Severity**: MINOR.

---

## Verbose Prose

### V1 -- index.md Section 5.3 "Normalization" (lines 135-140)

The three QKV norm lines are individually described but could be a compact table:

Current (6 lines):
> - `q_norm`: `Gemma4RMSNorm(head_dim, with_scale=True)` -- standard RMSNorm with learnable scale
> - `k_norm`: `Gemma4RMSNorm(head_dim, with_scale=True)` -- standard RMSNorm with learnable scale
> - `v_norm`: `Gemma4RMSNorm(head_dim, with_scale=False)` -- RMSNorm **without** learnable scale

The `q_norm` and `k_norm` entries are identical descriptions. A two-row table (q/k share config, v differs) would be more concise.

**Severity**: MINOR -- cosmetic.

### V2 -- moe_details.md line 159

> "The loop iterates only over active experts (those assigned at least one token), making it efficient for sparse routing. The `index_add_` at the end accumulates contributions from multiple experts per token."

The first sentence restates what the code already shows (`for expert_idx in expert_hit`). The second sentence restates the `index_add_` call. Both are readable from the code block directly above.

**Suggestion**: Trim to: "Only active experts (those with assigned tokens) are iterated; `index_add_` accumulates multi-expert contributions per token."

**Severity**: MINOR -- saves ~1 line.

---

## Summary

| ID | Type | Severity | Est. Lines Saved |
|----|------|----------|-----------------|
| R1 | Cross-file duplication (MLP+MoE path x3) | MINOR | ~35 |
| R2 | Cross-file duplication (TTNN parallel note) | MINOR | ~2 |
| R3 | Within-file repetition (five RMSNorms) | MINOR | ~4 |
| V1 | Verbose QKV norm listing | MINOR | ~2 |
| V2 | Redundant prose after code block | MINOR | ~1 |
| **Total** | | | **~44 lines** |

No crucial updates required. All findings are MINOR redundancy/verbosity reductions with zero information loss.
