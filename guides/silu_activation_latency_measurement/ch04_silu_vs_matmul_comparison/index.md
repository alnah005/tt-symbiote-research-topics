# Chapter 4: SiLU vs. Matmul Latency Comparison

This chapter uses the measurements collected in Chapter 3 to compare `ttnn.silu` latency against gate_proj and up_proj matmul latency, place both operations on the Wormhole B0 roofline, and determine when SiLU cost is relevant to MoE FFN optimization.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Locate SiLU and batched matmul on the Wormhole B0 roofline model and identify their compute-bound vs. memory-bound regimes.
2. Interpret a latency ratio table to determine what fraction of FFN time is consumed by `ttnn.silu` at different token counts.
3. Explain why SiLU latency is most significant at decode batch sizes (1–8 tokens) and negligible during prefill (128+ tokens).
4. Identify the token count threshold above which fusing SiLU into the preceding matmul provides no meaningful speedup.
5. Use the regime decision table to quickly determine whether a given workload warrants SiLU fusion.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Chapter 3 | Profiling methodology: `TT_METAL_DEVICE_PROFILER=1`, warm-up protocol, CSV column selection (`DEVICE KERNEL DURATION [ns]`), and pitfalls table |
| Chapter 2 | Understanding that `ttnn.silu` executes on the SFPU and is memory-bandwidth-bound at practical batch sizes |
| Measured data | A completed benchmark sweep over `num_tokens ∈ {1, 8, 32, 128}` and `hidden_dim ∈ {2048, 4096, 8192}` using the methodology from Chapter 3 |

---

## Chapter Structure

| File | Contents |
|---|---|
| `index.md` (this file) | Chapter overview, learning objectives, prerequisites, file map |
| [`roofline_analysis.md`](./roofline_analysis.md) | Wormhole B0 roofline model: hardware ceilings, ridge point, arithmetic intensity of SiLU and matmul, ASCII sketch |
| [`latency_ratio_by_shape.md`](./latency_ratio_by_shape.md) | Latency ratio table across shapes; expected vs. measured values; how to read the table and draw conclusions |
| [`compute_vs_memory_bound_regimes.md`](./compute_vs_memory_bound_regimes.md) | Compute-bound / memory-bound transition walkthrough; SiLU linear scaling; fusion threshold; decision table |

---

## How This Chapter Fits in the Guide

- **Chapter 2** established the hardware execution model: SiLU runs on the SFPU and reads/writes activation tiles through the L1-to-DRAM path.
- **Chapter 3** provided the measurement protocol and the profiler CSV column (`DEVICE KERNEL DURATION [ns]`) that yields hardware execution time, free of dispatch overhead.
- **Chapter 4** (this chapter) applies that data to a roofline analysis and produces actionable thresholds for fusion decisions.
- A hypothetical **Chapter 5** would cover kernel fusion implementations that eliminate the separate SiLU kernel for decode workloads.

---

## Key Hardware Constants

These constants appear throughout this chapter. They are fixed for Wormhole B0.

| Constant | Value |
|---|---|
| BF16 FPU peak throughput | 131 TFLOP/s |
| DRAM bandwidth (practical peak) | ~300 GB/s |
| Roofline ridge point | ~437 FLOP/byte |
| BF16 tile size | 2,048 bytes (32×32×2) |
| DRAM controllers | 6 |
| DRAM banks | 12 GDDR6 |

---

## Next Steps

Begin with [`roofline_analysis.md`](roofline_analysis.md) to understand the hardware performance model before examining the latency ratio data.
