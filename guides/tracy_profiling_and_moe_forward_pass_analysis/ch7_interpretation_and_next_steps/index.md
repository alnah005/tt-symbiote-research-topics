# Chapter 7: Interpretation and Next Steps

This chapter is the synthesis point of the guide. You have collected Tracy CPU-zone traces
(Chapter 2), annotated the MoE forward pass with named zones (Chapter 3), mapped the expected
op sequence (Chapter 4), attributed gaps to one of four patterns (Chapter 5), and quantified
how each gap scales with sequence length (Chapter 6). Chapter 7 translates those findings
into two concrete deliverables: a written **gap analysis document** and a **prioritized
optimization backlog**.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Map each of the four gap patterns (A–D from Chapter 5) to a specific TTNN optimization
   lever, with conditions and caveats.
2. Use the scaling analysis results from Chapter 6 to rank which gap has the largest total
   latency impact across a realistic inference workload.
3. Write a gap analysis document that includes enough evidence for an optimization engineer
   who was not present during profiling to reproduce and act on your findings.
4. Apply the optimization action reference table to select the right TTNN API for each root
   cause, and validate that a change did not degrade output correctness.

---

## Prerequisites

This chapter assumes you have completed — or are familiar with — all preceding chapters:

| Chapter | Key concept needed here |
|---|---|
| Ch 1: Tracy Profiler Overview | Tracy zone semantics, `.tracy` file format |
| Ch 2: Setting Up Tracy Profiling | How to collect and export Tracy traces |
| Ch 3: TTNN Op-Level Profiling API | `TT_METAL_DEVICE_PROFILER`, `profile_log_device.csv`, zone naming conventions |
| Ch 4: MoE Forward Pass Op Breakdown | Expected op sequence, per-op latency budget for Qwen 235B / DeepSeek-V3 |
| Ch 5: Identifying the 16ms Gap | Gap patterns A–D, attribution methods, `MoE/dispatch` / `MoE/expert_matmul` / `MoE/combine` zone names |
| Ch 6: Sequence Length Scaling Analysis | Standard seq_len sweep `{64, 128, 256, 512, 1024, 2048, 4096}`, scaling exponent, linear decomposition |

> **Warning:** Do not skip the seq_len sweep from Chapter 6 before writing the gap analysis.
> Scaling evidence is what separates a "confirmed" root cause from a "suspected" one. Without
> the sweep, your gap analysis will have lower confidence and lower actionability.

---

## The Two Outputs of This Chapter

### Output 1: Gap Analysis Document

A gap analysis document is a self-contained write-up of your profiling investigation. Its
purpose is to communicate findings to optimization engineers who will implement fixes. The
document follows a seven-section template defined in `writing_a_gap_analysis.md`:

1. Observed latency
2. Profiling methodology
3. Per-op breakdown table
4. Gap attribution findings
5. Scaling behavior
6. Prioritized root causes
7. Recommended actions

The gap analysis is the primary artifact that justifies engineering investment. It should be
written with enough detail that someone who was not present during profiling can reproduce
every measurement described.

### Output 2: Prioritized Optimization Backlog

A backlog of optimization actions, ordered by expected latency impact across a typical
inference workload. Each backlog item maps to a specific TTNN API or configuration change,
with an expected latency reduction and a PCC validation requirement.

The format for a backlog item is:

```
[ID]  [Pattern]  [Gap component]  [Expected reduction]  [API lever]  [Validation]
```

Example row:

```
OPT-1  Pattern B  ~14ms constant term  Eliminate ttnn.synchronize_device
       check correctness of ordering; replace with ttnn.event if needed
       PCC > 0.999 vs. CPU reference
```

`gap_to_action_mapping.md` provides the full pattern-to-action mapping.
`optimization_action_reference.md` provides the detailed reference for each TTNN lever.

---

## Files in This Chapter

| File | Contents |
|---|---|
| `index.md` | This overview, learning objectives, and navigation (you are here) |
| `gap_to_action_mapping.md` | Systematic mapping from Patterns A–D to TTNN optimization actions; prioritization using scaling data |
| `writing_a_gap_analysis.md` | Seven-section template, evidence requirements, confidence levels, audience guidance |
| `optimization_action_reference.md` | Reference table of TTNN optimization levers with expected reductions, conditions, and caveats |

---

## Next Steps

Start with [`gap_to_action_mapping.md`](./gap_to_action_mapping.md) to determine which
optimization action applies to each gap pattern you identified in Chapter 5. Then read
[`writing_a_gap_analysis.md`](./writing_a_gap_analysis.md) to structure your findings into
a shareable document. Finally, consult
[`optimization_action_reference.md`](./optimization_action_reference.md) for the full API
details and validation procedure before beginning any implementation work.
