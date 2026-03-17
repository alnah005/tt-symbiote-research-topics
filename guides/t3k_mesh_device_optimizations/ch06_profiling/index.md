# Chapter 6: Profiling and Performance Analysis on T3K

## Prerequisites

This chapter assumes familiarity with:

- **Chapter 2** — `ttnn` mesh API: `MeshDevice`, `MeshTensor`, distributed tensor sharding, and
  `ttnn.all_to_all` semantics (see `collective_primitives.md`)
- **Chapter 3** — `num_links` parameter, Ethernet bandwidth arithmetic, and link utilization
  trade-offs (see `num_links_parameter.md`)
- **Chapter 4** — L1 vs. DRAM placement decisions and the Wormhole B0 memory hierarchy
  (`decode_memory_strategy.md`, `prefill_memory_strategy.md`)
- **Chapter 5** — Expert placement strategies and how routing topology affects cross-device traffic
  (`expert_placement_strategies.md`)

Readers who have not completed Chapters 2–5 will encounter undefined terms and missing context for
the remediation guidance. Return here after completing those chapters.

---

## Goal and Motivation

Tuning a MoE workload on T3K without profiling first is guesswork. The four bottleneck categories
— compute-bound, memory-bandwidth-bound, communication-bound, and L1-pressure — each require
entirely different remediation. Applying the wrong fix wastes engineering time and can silently
degrade performance (for example, increasing `num_links` when the workload is actually
compute-bound adds link setup overhead without reducing layer latency).

This chapter provides:

1. The tools needed to collect timing and hardware-counter data on T3K.
2. A structured decision procedure for mapping measurements to bottleneck categories.
3. Targeted remediation strategies cross-referenced to earlier chapters where the relevant
   parameters are introduced.

> **Warning:** Do not tune `num_links`, change memory placement, or modify expert distribution
> before establishing a profiled baseline. The bottleneck may not be where intuition suggests.

---

## Key Message

Profile first. The MoE layer on T3K operates in at least two distinct regimes:

- **Decode (B ≤ 32, C ≈ 1–2):** all-to-all latency typically dominates; Tensix cores are often
  idle waiting for Ethernet transfers to complete.
- **Prefill (large B·S):** expert FFN matmul dominates; communication becomes a smaller fraction
  of total layer time.

A tuning decision that is correct for decode can be harmful for prefill, and vice versa. Both
phases must be profiled separately before any parameter is changed.

---

## Chapter Navigation

| File | Contents |
|---|---|
| `ttnn_profiler.md` | Enabling the TTNN profiler; reading op-level timing output; Tracy timeline visualization; comparing prefill vs. decode profiles |
| `device_perf_counters.md` | Ethernet link utilization, DRAM bandwidth, NOC traffic, and Tensix core utilization counters; detecting link saturation and L1 pressure |
| `bottleneck_diagnosis_guide.md` | Decision tree for identifying bottleneck category; per-category remediation procedures; common anti-patterns |

Read the files in the order listed. `ttnn_profiler.md` introduces the vocabulary used in
`device_perf_counters.md`, and both are prerequisites for `bottleneck_diagnosis_guide.md`.

---

## 5-Step Profiling Workflow

The following workflow is the recommended procedure for any T3K MoE performance investigation.
Details for each step are in the sub-files.

### Step 1 — Establish a Correct Baseline

Before profiling performance, verify correctness. Run the MoE layer against a CPU reference and
confirm Pearson Correlation Coefficient (PCC) > 0.99 for all output tensors. A numerically
incorrect implementation can appear fast because it is skipping work.

### Step 2 — Enable the TTNN Profiler and Run a Representative Workload

Set the profiler environment variable [VERIFY environment variable name] and run at least
10 warmup iterations followed by 100 measurement iterations. Warmup is required to allow DRAM
caches and Ethernet link state to reach steady state. Use the batch size and sequence length that
matches the target deployment scenario; decode and prefill must be profiled separately.

See `ttnn_profiler.md` §1–2 for profiler setup.

### Step 3 — Parse Profiler Output and Identify Top-Latency Operations

Read the CSV output produced by the TTNN profiler. Sort by `device_time_ns` (descending) and
identify the top-5 most expensive operations. Compute the fraction of total MoE layer time
attributable to `ttnn.all_to_all` (dispatch + combine) and to the expert FFN `ttnn.matmul`.

See `ttnn_profiler.md` §3 for the Python parsing snippet.

### Step 4 — Identify the Bottleneck Category Using Device Counters

Consult the hardware performance counters to confirm which resource is the binding constraint:

- Ethernet link utilization ≥ 90% during all-to-all → communication-bound
- DRAM read bandwidth high during expert matmul → memory-bandwidth-bound
- Tensix core utilization > 80% during expert matmul, DRAM BW moderate → compute-bound
- Unexpected DRAM BW during operations expected to be L1-resident → L1 spill / L1-pressure

See `device_perf_counters.md` for counter access and interpretation.

### Step 5 — Apply Targeted Remediation and Re-Profile

Apply the remediation for the identified bottleneck category, re-run the profiling workflow from
Step 2, and compare the new profile against the baseline. A single change should be made between
each profiling run so that cause and effect remain clear.

See `bottleneck_diagnosis_guide.md` for per-category remediation procedures.

---

## Summary Notation

The notation below is used consistently across this chapter's sub-files.

| Symbol | Meaning | Typical Value (Qwen3.5-35B / T3K) |
|---|---|---|
| $B$ | Batch size | 1–32 (decode) |
| $H$ | Hidden dimension | 7168 |
| $E$ | Total experts | 256 |
| $k$ | Top-$k$ per token | 8 |
| $N$ | Devices in mesh | 8 |
| $E_d$ | Experts per device ($E/N$) | 32 |
| $C$ | Expert capacity = $\lceil k B / E \rceil$ | 1–2 (decode) |
| $D$ | Expert FFN intermediate dimension | [UNVERIFIED] |

---

## Capacity Formula

For reference, the expert capacity per device used throughout this chapter is:

$$C = \left\lceil \frac{k \cdot B \cdot CF}{E} \right\rceil$$

where $CF = 1.25$ is the capacity factor used to absorb load imbalance in routing. At $B=32$:

$$C = \left\lceil \frac{8 \times 32 \times 1.25}{256} \right\rceil = \left\lceil 1.25 \right\rceil = 2$$

At $B=1$: $C = \lceil 8 \times 1 \times 1.25 / 256 \rceil = 1$. These two operating points — $C=1$
and $C=2$ — represent the primary decode regime for Qwen3.5-35B on T3K and are the reference
points for all latency and bandwidth estimates in this chapter.

---

## References

- TTNN profiler documentation: [VERIFY documentation location]
- Tracy profiler: https://github.com/wolfpld/tracy
- Chapters 2–5 of this guide
- `tt-metal` GitHub repository: performance tooling under `ttnn/tools/profiler/`
- `ch01_t3k_topology/ethernet_link_bandwidth.md` — Ethernet link bandwidth baseline
- `ch02_ttnn_mesh_api/collective_primitives.md` — `ttnn.all_to_all` API and dispatch semantics
