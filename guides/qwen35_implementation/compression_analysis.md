# Compression Analysis: Qwen3.5 Implementation Guide — Cross-Chapter Pass 1

## Summary
- Total files analyzed: 27 content files across 8 chapters + root index
- Estimated current line count: ~2,100 lines
- Estimated post-compression line count: ~2,000 lines
- Estimated reduction: ~5%

## CRUCIAL Suggestions

### [ch2/host_recurrence.md, ch7/sync_overhead.md] DeltaNet sync root cause
**Issue:** The Blackhole SrcB TF32 constraint explanation (bf16 element-wise ops on fp32 CBs hang the device) appears in both ch2/host_recurrence.md and ch7/sync_overhead.md Section 1 at paragraph depth.
**Suggestion:** ch7/sync_overhead.md Section 1 should reference ch2/host_recurrence.md as the authoritative explanation and reduce its own root-cause paragraph to one sentence + cross-reference.

### [ch5/dram_budget.md, ch7/latency_breakdown.md] DRAM usage numbers
**Issue:** A3B DRAM breakdown numbers (expert weights ~15.0 GiB, shared ~0.8 GiB, etc.) are stated in full in both ch5/dram_budget.md and ch7/latency_breakdown.md.
**Suggestion:** ch7/latency_breakdown.md should reference ch5/dram_budget.md for the full breakdown and retain only the total (15.7 GiB of 28 GiB) inline.

## MINOR Suggestions

### Cross-chapter: "host recurrence" terminology
**Issue:** The historical implementation is called "host recurrence" in ch2, "host-recurrence path" in ch6, and "host-recurrence era" in ch7. All three are clear but the variation is unnecessary.
**Suggestion:** Standardize to "host-recurrence" (hyphenated) throughout.

### Cross-chapter: efficiency percentage precision
**Issue:** A3B efficiency is "~6.7%" in some early references and "~6.8%" in ch7 (corrected). The earlier chapters predate the correction.
**Suggestion:** Verify ch1 and ch2 do not contain the stale 6.7% figure.

### [ch3/partial_rope.md, ch8/optimization_roadmap.md] Device RoPE fix description
**Issue:** The cos/sin patching fix for partial RoPE is described at moderate depth in both ch3/partial_rope.md and ch8/optimization_roadmap.md.
**Suggestion:** ch8 description can be reduced to one sentence + reference to ch3.

## Load-Bearing Evidence
- `ch2/recurrence_math.md` lines ~20–60: the five-step DeltaNet equations — load-bearing; the mathematical derivation exists only in this file and is the foundation for ch7's bottleneck analysis.
- `ch5/router_and_routing.md` lines ~30–50: 512-byte DMA calculation for MoE router — load-bearing; this specific claim (bf16 × 256 = 512 bytes) is the basis for the "negligible DMA" conclusion in ch7.
- `ch6/moe_key_protection.md` lines ~9–54: two failure mode examples — load-bearing; removing either would orphan the "why" explanation referenced by ch4.
- `ch8/testing_infrastructure.md` lines ~107–124: PCC metric implementation — load-bearing; the only place the PCC formula is given in executable form.

## VERDICT
- Crucial updates: no
