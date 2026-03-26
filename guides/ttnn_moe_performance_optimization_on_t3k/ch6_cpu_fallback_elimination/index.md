# Chapter 6 — CPU Fallback Elimination

## Context

This chapter addresses **Q6**: identifying, quantifying, and eliminating silent CPU execution paths in the GLM-4 MoE stack on T3K.

---

## Motivation

A silent CPU fallback is uniquely dangerous in a hardware-accelerated inference stack: no exception is raised, no log message is emitted, and the output is numerically correct. The only observable symptom is throughput that is 2–3 orders of magnitude below what T3K is capable of. In a production deployment, a single undetected CPU fallback in the MoE expert computation dominates the total decode latency and renders every other optimization effort meaningless.

The current `moe.py` contains exactly one such fallback: the hardcoded `ttnn = False` flag at `moe.py:L569–570`, which ensures that `Glm4MoeNaiveMoeHybrid` always instantiates `Glm4MoeExpertLayersTorch` and never `TTNNGlm4MoeExpertLayers`. When `TTNNGlm4MoeMoE` is used as the outer MoE module, the router and shared experts run on Tensix cores, but all routed expert computation executes as a sequential Python for-loop on the host CPU.

---

## Scope

The single identified silent CPU fallback in the current `moe.py` is:

- **Class:** `Glm4MoeNaiveMoeHybrid` (instantiated inside `TTNNGlm4MoeMoE.from_torch`)
- **Root cause:** `ttnn = False` at `moe.py:L569–570`
- **Effect:** `Glm4MoeExpertLayersTorch` handles all routed expert computation on CPU

No other silent CPU fallbacks have been identified. The grep audit in `fallback_detection_and_testing.md` documents the search methodology and expected findings.

---

## Chapter Structure

**[GLM-4 CPU Path Audit](glm4_cpu_path_audit.md)**

Analyzes the architecture of `TTNNGlm4MoeMoE` and the latency cost of running routed expert computation through `Glm4MoeNaiveMoeHybrid` on CPU. Provides a step-by-step migration checklist for replacing the CPU path with `TTNNGlm4MoeExpertLayers`, including weight format conversion for the fused `gate_up_proj` tensor.

**[Fallback Detection and Testing](fallback_detection_and_testing.md)**

Provides systematic tooling: grep patterns for source-level detection, a runtime hook for host-device transfer monitoring, a module-level introspection function, and a pytest fixture for automated regression testing.

---

## Cross-Reference

Chapter 1's `cpu_fallback_paths.md` establishes the foundational context for this chapter:

- The `ttnn = False` flag and its dead-code implications (`moe.py:L569–570`)
- The CPU expert loop structure in `Glm4MoeNaiveMoeHybrid.forward` (`moe.py:L584–L613`)
- A side-by-side comparison of `TTNNExperts.forward` and `Glm4MoeNaiveMoeHybrid.forward`
- The six-item checklist for verifying the TTNN path is active before collecting any performance data
- Instructions for enabling the TTNN path by setting `ttnn = True`

The files in this chapter do not duplicate that material. They assume familiarity with the Ch1 analysis and go deeper: latency quantification methodology, the weight format impedance between `Glm4MoeExpertLayersTorch` and `TTNNExperts`, and automated detection infrastructure.
