# Chapter 7 — Implementation Roadmap and Risk Assessment

## Learning Objectives

After completing this chapter, you will be able to:

- Describe the four-phase implementation plan for porting the Gemma 4 vision encoder to TTNN
- Estimate the total timeline and identify the critical path through the phases
- Enumerate the six primary risks and their mitigations
- Make go/no-go decisions at each phase gate based on concrete criteria
- Assign sprint tasks using the phase breakdown and dependency ordering from [Chapter 6](../ch06_reuse_strategy/index.md)

## Prerequisites

- Completion of all prior chapters, especially:
  - [Chapter 5 — CPU vs. TTNN Latency Analysis](../ch05_cpu_vs_ttnn_latency/index.md) (justification for the port)
  - [Chapter 6 — Reuse Strategy](../ch06_reuse_strategy/index.md) (module effort estimates and dependency ordering)

## Chapter Contents

| File | Topic |
|------|-------|
| [`phased_plan.md`](./phased_plan.md) | Four-phase implementation plan with milestones, timelines, and deliverables |
| [`risk_register.md`](./risk_register.md) | Six identified risks with severity, likelihood, impact, and mitigation strategies |

## Overview

This chapter synthesizes the findings from all prior chapters into an actionable implementation plan. The Gemma 4 vision encoder port is organized into four sequential phases, each with a clear entry criterion, deliverable, and phase gate that determines whether to proceed.

### High-Level Timeline Summary

| Phase | Description | Duration | Cumulative |
|-------|-------------|----------|------------|
| **Phase 1** | CPU reference and correctness baseline | 1 week | 1 week |
| **Phase 2** | Module-level TTNN port | 2-3 weeks | 3-4 weeks |
| **Phase 3** | End-to-end integration and optimization | 1-2 weeks | 4-6 weeks |
| **Phase 4** | Variable resolution support | 1 week | 5-7 weeks |

**Total estimated duration: 5-7 engineer-weeks.**

This estimate assumes one to two engineers working full-time. The critical path runs through Phase 2, where the attention module with 2D RoPE integration is both the highest-effort and highest-risk component. Phase 1 is a prerequisite for all subsequent work because it establishes the golden reference outputs used for PCC validation.

### Decision Points

The plan includes three explicit go/no-go gates:

1. **After Phase 1:** If CPU profiling shows the vision encoder accounts for less than 5% of total inference latency in the target deployment scenario, the TTNN port may not be justified. See the [decision matrix](../ch05_cpu_vs_ttnn_latency/decision_matrix.md) from Chapter 5.

2. **After Phase 2:** If module-level PCC validation reveals systematic numerical issues (especially in 2D RoPE or adaptive pooling), additional investigation is needed before proceeding to integration.

3. **After Phase 3:** If end-to-end TTNN latency does not meet the target speedup over CPU, evaluate whether optimization effort (Phase 3 extension) or selective CPU fallback is the better path.

## Reading Order

Start with [`phased_plan.md`](./phased_plan.md) for the detailed phase breakdown, then read [`risk_register.md`](./risk_register.md) to understand the threats to the plan and how to mitigate them.
