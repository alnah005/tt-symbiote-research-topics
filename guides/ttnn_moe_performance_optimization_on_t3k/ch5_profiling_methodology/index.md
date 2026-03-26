# Chapter 5: Profiling Methodology

## Why Op-Level Granularity is Necessary

`TTNNMoE.forward` (`moe.py:L1412–L1496`) consists of roughly a dozen distinct operations spanning four categories: CCL collectives (`all_gather_async`, `reduce_scatter_minimal_async`), device-compute kernels (`ttnn.linear`, `sparse_matmul`, `silu`), in-place tensor manipulation (`moe_expert_token_remap`, `ttnn.add`), and Python-level dispatch overhead. A wall-clock measurement of the entire forward pass collapses all of these into a single number that cannot direct optimization work.

Three concrete problems arise from coarse measurement:

1. **Misattribution.** If the total forward pass time is 2 ms and the CCL `all_gather_async` takes 0.8 ms, the matmul stages are at most 1.2 ms. Without per-op timing, an engineer chasing matmul configuration might tune `in0_block_w` exhaustively and achieve a 10% improvement on the matmul — a net reduction of 0.12 ms (6% of total). The same effort applied to the CCL stage would yield far more.

2. **Invisible pipeline bubbles.** TTNN dispatches ops asynchronously to the device command queue; the host returns before the device kernel completes. A coarse wall-clock measurement on the host reflects host-side dispatch overhead, not device execution time. Op timers and Tracy both capture device-side execution, which is the relevant metric for kernel tuning.

3. **Routing decision isolation.** `TTNNMoERouterDecode.forward` (`moe.py:L891–L1024`) runs the 3-pass BF16 centering trick — but only when `n_group <= r.topk_group` (the non-group-selection code path). The group-based path (`n_group > topk_group`) does not use 3-pass centering and proceeds differently. Its contribution to total latency is invisible in a single-pass forward measurement but may be non-trivial at high token throughput. Isolating the router is only possible with op-level instrumentation.

---

## The Three Profiling Tools

### Tool 1 — Tracy (Visual Timeline)

Tracy is a frame profiler and timeline tool that attaches to the running process via a TCP socket. It captures annotated zones (wall-clock spans) from both the host-side Python dispatch loop and from TTNN device-side C++ callbacks. The output is a visual timeline showing zone nesting, allowing critical-path identification at a glance.

Tracy is the right tool when you need to see **overlap and sequencing** — for example, whether the `reduce_scatter_minimal_async` in `TTNNMoE.forward` (moe.py:L1478–L1490) overlaps with `shared_experts` (moe.py:L1493) in practice, or whether there is a stall between them.

**Full setup guide:** [tracy_profiling_setup.md](./tracy_profiling_setup.md)

---

### Tool 2 — TTNN Op Timers (Programmatic / CSV)

TTNN's built-in op timer infrastructure records per-op device-side elapsed cycles and converts them to microseconds using the device clock frequency. It does not require any additional tooling: set two environment variables, run the forward pass, and collect a CSV of op names, op types, and device-execution times.

TTNN op timers are the right tool when you need **numeric per-op latency** that can be aggregated, filtered, sorted, and compared across configuration sweeps — for example, comparing `in0_block_w=2` vs `in0_block_w=4` for the three expert `sparse_matmul` calls.

**Full setup guide:** [ttnn_op_timer_profiling.md](./ttnn_op_timer_profiling.md)

---

### Tool 3 — Device-Side Cycle Counters

Wormhole Tensix cores expose a 64-bit free-running cycle counter accessible from within a kernel via `get_arg_val<uint64_t>()` (host injection) or via the device-side timer API. At the Python level, the `ttnn.device` object exposes `device.get_device_profiler_events()` after enabling `TT_METAL_DEVICE_PROFILER=1`. This produces per-core cycle timestamps that can be correlated with op dispatch order to identify stalls caused by L1 bank conflicts or DRAM saturation.

Cycle counters are the right tool when TTNN op timers show a kernel taking longer than its arithmetic intensity predicts — that is, when the question shifts from "which op is slow" to "why is this kernel slow on this particular core grid".

**Full setup guide:** [router_latency_profiling.md](./router_latency_profiling.md) (the router isolation harness uses device-side cycle counters as the precision measurement substrate)

---

## Research Questions Covered

| File | Research question |
|---|---|
| [`tracy_profiling_setup.md`](./tracy_profiling_setup.md) | Q8 — What is the op-level latency breakdown for `TTNNMoE.forward` on T3K? |
| [`ttnn_op_timer_profiling.md`](./ttnn_op_timer_profiling.md) | Q8 — Same question, programmatic extraction for sweep automation |
| [`router_latency_profiling.md`](./router_latency_profiling.md) | Q7 — What is the latency and precision impact of the 3-pass BF16 centering trick in `TTNNMoERouterDecode.forward`? |

---

## Reading Order

Read in the order listed above. `tracy_profiling_setup.md` gives the conceptual grounding and the zone annotation model; `ttnn_op_timer_profiling.md` builds on that by showing how to extract the same information programmatically; `router_latency_profiling.md` applies both tools to the router isolation problem and adds the precision dimension that neither timing tool covers on its own.
