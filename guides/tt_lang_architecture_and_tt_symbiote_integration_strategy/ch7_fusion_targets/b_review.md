# Agent B Review: Chapter 7 -- Pass 1

## Issue 1: Factual error -- three-pass softmax mislabeled as "Flash Attention style"

**File:** `fused_attention.md`, line 251

The text states: "This three-pass softmax is the standard online-softmax approach (Flash Attention style)."

This is incorrect. The kernel shown performs three separate passes over the KV tiles (pass 1: find row max, pass 2: compute exp and sum, pass 3: normalize and matmul with V). This is a **standard multi-pass softmax**, not the Flash Attention online approach. Flash Attention (Dao et al., 2022) performs a **single-pass** online softmax that incrementally updates the running max, rescales partial sums, and accumulates the output in one sweep over KV blocks -- which is precisely what makes it memory-efficient. The three-pass design shown here also reads `scores_dfb` three times, which undermines the stated goal of avoiding DRAM materialization of the softmax output (scores must be streamed from somewhere three times).

**Fix:** Either redesign the kernel to use true single-pass online softmax (rescale-and-accumulate), or correct the description to say "standard multi-pass softmax" and remove the Flash Attention attribution.

## Issue 2: Missing navigation footers on all three content files

**Files:** `moe_expert_pipeline.md`, `fused_attention.md`, `fused_activations.md`

Each content file has a `**Next:**` link but no `**Previous:**` link. Per the chapter conventions established in earlier chapters, content files (not index.md) should have both Previous and Next navigation footers.

- `moe_expert_pipeline.md` -- missing `**Previous:** [index.md](./index.md)`
- `fused_attention.md` -- missing `**Previous:** [moe_expert_pipeline.md](./moe_expert_pipeline.md)`
- `fused_activations.md` -- missing `**Previous:** [fused_attention.md](./fused_attention.md)`

**Fix:** Add `**Previous:**` links to each content file above the existing `**Next:**` line.

## Summary

2 items flagged (1 factual error, 1 structural gap across 3 files). All class names, line numbers, API calls, and size calculations verified against source. No other issues found.

# Agent B Review: Chapter 7 -- Pass 2

Pass 1 Issue 1 (softmax mislabeling) has been corrected. The text in `fused_attention.md` line 251 now reads "tile-streaming three-pass softmax, not a Flash Attention style online softmax" with an accurate explanation of the difference. Verified -- resolved.

Pass 1 Issue 2 (missing navigation footers) has **not** been addressed. Carrying forward below plus one new issue discovered.

## Issue 1 (carried): Missing `**Previous:**` navigation links on all three content files

**Files:** `moe_expert_pipeline.md`, `fused_attention.md`, `fused_activations.md`

Each content file has a `**Next:**` link but still no `**Previous:**` link. Per chapter conventions:

- `moe_expert_pipeline.md` -- needs `**Previous:** [Chapter 7 Index](./index.md)` above the `**Next:**` line
- `fused_attention.md` -- needs `**Previous:** [MoE Expert Pipeline Fusion](./moe_expert_pipeline.md)` above the `**Next:**` line
- `fused_activations.md` -- needs `**Previous:** [Fused Attention](./fused_attention.md)` above the `**Next:**` line

## Issue 2 (new): Broken forward link in `fused_activations.md`

**File:** `fused_activations.md`, line 296

The `**Next:**` link points to `../ch8_workflow_and_multidevice/index.md`, but the directory `ch8_workflow_and_multidevice/` does not exist in the guide tree. This produces a dead link. Either create the Chapter 8 directory with at least a stub `index.md`, or change the link to point to the guide-level table of contents (or remove it and add a "End of Chapter 7" marker) until Chapter 8 is written.

## Summary

2 issues flagged (1 carried structural gap, 1 new broken link). The Pass 1 factual error on softmax labeling is confirmed fixed. All class names, line numbers, API references, and size calculations re-verified against source -- no new factual errors found.
