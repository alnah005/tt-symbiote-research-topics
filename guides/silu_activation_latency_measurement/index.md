# Guide: SiLU Activation Latency Measurement

This guide characterizes the latency of `ttnn.silu` in Mixture-of-Experts (MoE) FFN inference on
Tenstorrent Wormhole B0. It explains why SiLU is non-trivial on SFPU-based hardware, provides a
complete methodology for measuring its cost in isolation, compares it against the surrounding
matmul operations, and delivers concrete fusion recommendations for decode and prefill workloads.

---

## Audience

**Who should read this guide:**
ML engineers integrating SwiGLU-based MoE models (Llama 3, Mixtral 8x7B, Qwen2-MoE,
DeepSeek-V3) into a TTNN inference stack on Wormhole B0 or T3K. Readers should already be
comfortable with `ttnn.matmul`, basic tensor operations, and conceptual MoE routing. No prior
knowledge of Tensix hardware internals or TTNN kernel scheduling is required.

**What you will learn:**
- Where SiLU sits in the SwiGLU compute graph and why MoE multiplies the number of SiLU calls
- How SiLU executes on the SFPU inside a Tensix core, and why it is always memory-bound
- How to isolate and measure `ttnn.silu` latency using the Tracy profiler CSV
- The quantitative SiLU-to-matmul latency ratio across token counts and hidden dimensions
- Which TTNN fusion APIs eliminate the separate SiLU kernel dispatch and when to apply them

---

## Chapter Navigation

| Directory | Description | Key Files |
|---|---|---|
| `ch01_silu_in_moe_architecture/` | Establishes the computational context: where SiLU appears in the SwiGLU FFN graph, why MoE dispatch makes per-expert SiLU invocations non-trivial, and the cost hypothesis that motivates measurement. | `ffn_compute_graph.md`, `swiglu_variant.md`, `compute_role_and_cost_hypothesis.md` |
| `ch02_silu_on_wormhole_hardware/` | Explains how SiLU maps to SFPU LLK instructions inside a Tensix core, why it cannot overlap with the preceding FPU matmul, and how to place it on the Wormhole B0 roofline. | `tensix_compute_engine.md`, `silu_sfpu_execution.md`, `cycles_vs_matmul.md` |
| `ch03_measuring_silu_latency/` | Provides a hands-on benchmark methodology: enabling the TTNN device profiler, structuring warm-up loops, reading `DEVICE KERNEL DURATION [ns]` from the Tracy CSV, and two strategies for isolating SiLU from surrounding matmuls. | `profiling_setup.md`, `isolating_silu_from_matmul.md`, `measurement_methodology.md` |
| `ch04_silu_vs_matmul_comparison/` | Uses the Chapter 3 methodology to produce a roofline analysis and a latency ratio table across `num_tokens ∈ {1, 8, 32, 128}` and `hidden_dim ∈ {2048, 4096, 8192}`. Derives the `num_tokens = 16` decode/prefill crossover threshold. | `roofline_analysis.md`, `latency_ratio_by_shape.md`, `compute_vs_memory_bound_regimes.md` |
| `ch05_fused_activation_strategies/` | Documents the TTNN APIs for fusing SiLU into the preceding matmul kernel. Defines Pattern A (fused, 3 dispatches) and Pattern B (unfused baseline, 4 dispatches), covers `activation_dtype` precision tradeoffs, and identifies when sharded tensors require the program config path. | `ttnn_fused_activation_api.md`, `swiglu_fusion_pattern.md`, `activation_dtype_and_precision.md` |
| `ch06_performance_impact_and_recommendations/` | Synthesizes all findings into a quick-reference recommendation table covering decode (BF16 and BFP8), prefill, and T3K expert-parallel scenarios. Includes a Tracy profiler verification checklist and open questions for future work. | `when_fusion_helps.md`, `configuration_recommendations.md`, `measurement_summary_and_next_steps.md` |

---

## Quick-Start Paths

Choose the path that matches your immediate need.

**I just want the configuration recommendation.**
Go directly to `ch06_performance_impact_and_recommendations/index.md`. The quick-reference table
there maps `(num_tokens, hidden_dim, dtype)` to the correct TTNN API call. The key findings
summary below gives a one-page version of the same conclusions.

**I am debugging elevated SiLU latency on a real workload.**
Start at `ch03_measuring_silu_latency/profiling_setup.md` to enable the Tracy profiler and confirm
you are reading `DEVICE KERNEL DURATION [ns]` (not `OP TO OP LATENCY [ns]`). Then read
`isolating_silu_from_matmul.md` to separate SiLU cost from matmul cost in your trace.
Cross-reference `ch04_silu_vs_matmul_comparison/latency_ratio_by_shape.md` to compare your
measured ratio against expected values.

**I want to understand the hardware before reading the measurements.**
Read the guide in linear order: Ch1 → Ch2 → Ch3 → Ch4 → Ch5 → Ch6. Chapter 1 requires no
hardware background; Chapter 2 builds the SFPU model needed to interpret everything that follows.

**I need to set up the benchmark from scratch.**
Start at `ch03_measuring_silu_latency/index.md` (prerequisites: Ch1 and Ch2) and follow through
`profiling_setup.md` and `measurement_methodology.md`. The methodology section specifies the
recommended input shapes, warm-up count (minimum 2 iterations to populate program cache), and
statistical protocol (median over 20 timed iterations, report p95 for jitter characterization).

---

## Key Findings Summary

These are the primary quantitative and architectural conclusions of the guide. Each finding is
derived in the chapter indicated.

**SiLU arithmetic intensity is ~0.5 FLOP/byte (always memory-bound).**
SiLU performs 2 FLOPs per element (sigmoid approximation + multiply by x) and transfers 4 bytes
per element (2 bytes read + 2 bytes write), yielding arithmetic intensity ≈ 0.5 FLOP/byte. Practical estimates with SFPU LReg overhead place effective intensity near
0.5 FLOP/byte. This is far below the Wormhole B0 ridge point (~437 FLOP/byte), meaning SiLU is
memory-bandwidth-limited at all practical MoE token counts regardless of hidden dimension.
*Derived in `ch04_silu_vs_matmul_comparison/roofline_analysis.md`.*

**At prefill token counts (128+ tokens), SiLU is 4–8% of gate_proj matmul latency.**
The gate_proj matmul becomes compute-bound as token count grows and its latency scales faster
than SiLU, which remains memory-bound and scales linearly with tensor size. At 128 tokens, SiLU
represents 4–8% of gate_proj matmul time. At 256+ tokens this fraction continues to shrink.
*Derived in `ch04_silu_vs_matmul_comparison/latency_ratio_by_shape.md`.*

**At decode token counts (1–8 tokens), SiLU may be 15–40% of gate_proj matmul latency.**
Both matmul and SiLU are memory-bandwidth-limited at small batch sizes, so the latency ratio
remains high. SiLU absolute latency ranges from approximately 2 µs (`[1, 1, 8, 2048]` BF16) to
approximately 20 µs (`[1, 1, 2048, 8192]` BF16) on Wormhole B0.
*Derived in `ch04_silu_vs_matmul_comparison/latency_ratio_by_shape.md` and `ch06_performance_impact_and_recommendations/index.md`.*

**The decode/prefill crossover is num_tokens ≈ 16.**
Below 16 tokens: both matmul and SiLU are memory-bound; fusion provides a first-order speedup.
At or above 16 tokens: fusion benefit drops below 5% of total FFN latency. At or above 64 tokens:
SiLU latency is negligible and fusion is optional.
*Derived in `ch04_silu_vs_matmul_comparison/compute_vs_memory_bound_regimes.md`.*

**Pattern A (fused SiLU into gate_proj matmul) uses 3 kernel dispatches; Pattern B uses 4.**
Pattern A: fused `gate_proj + SiLU` → `up_proj` → `ttnn.mul` (3 dispatches).
Pattern B: `gate_proj` → `ttnn.silu` → `up_proj` → `ttnn.mul` (4 dispatches).
Pattern A eliminates the separate SFPU kernel launch and the intermediate L1 read/write for the
activation tensor. It is the recommended default for decode workloads.
*Defined in `ch05_fused_activation_strategies/swiglu_fusion_pattern.md`.*

**Use fused activation for decode; standalone ttnn.silu is acceptable for prefill.**
- Decode (1–8 tokens), BF16: `ttnn.matmul(gate_proj, x, activation="silu")` + `ttnn.mul`.
- Decode (1–8 tokens), BFP8 tensors: Pattern A with `fused_activation` in program config and
  `activation_dtype=ttnn.bfloat8_b`; validate PCC > 0.999 against BF16 reference.
- Prefill (128+ tokens): standalone `ttnn.silu` is acceptable; prioritize matmul program config
  tuning over activation fusion. Fusion for sharded large hidden dims may still reduce L1
  pressure even when latency benefit is small.
*Synthesized in `ch06_performance_impact_and_recommendations/configuration_recommendations.md`.*

---

## Cross-Chapter Dependency Note

Chapters are written to be read in order. The dependency structure is:

```
Ch1 (no prerequisites)
 └─> Ch2 (requires Ch1: FFN dataflow context)
      └─> Ch3 (requires Ch1 + Ch2: shapes from Ch1, hardware model from Ch2)
           └─> Ch4 (requires Ch2 + Ch3: roofline model + measured data)
                └─> Ch5 (requires Ch2 + Ch4: SFPU model + latency threshold)
                     └─> Ch6 (requires Ch3 + Ch4 + Ch5: all measurement and fusion results)
```

Forward references are marked in each file with the pattern *See Chapter N, `filename.md`* so
they can be read out of order once the dependency is satisfied. Readers following a quick-start
path who skip earlier chapters should verify they meet the prerequisites listed in the target
chapter's `index.md` before proceeding.
