# Estimating Trace Speedup

Once you have measured values for `total_step_time` and `dispatch_overhead` using the methods in `measuring_dispatch_overhead.md`, the speedup that trace will provide is determined by a single formula. This file defines that formula, explains precisely what trace eliminates from the measured overhead and what it cannot eliminate, works through a concrete example with a 32-op decode step, describes the diminishing-returns regime, and closes with a checklist for deciding whether the predicted speedup justifies the implementation cost.

---

## The Speedup Model

The predicted speedup from applying trace to a decode step is:

```
speedup = total_step_time / (total_step_time - dispatch_overhead)
```

where:

- `total_step_time` — measured wall-clock time from the start of the first op dispatch to the completion of the last device kernel in one decode step, including any synchronization cost.
- `dispatch_overhead` — the sum of all host dispatch times across all ops in the step: `D = sum(d_i)` for all ops `i` in the step. Only the portion of dispatch overhead that trace actually eliminates should be used here (see the next section).

The formula is an idealized model that assumes the entire measured `dispatch_overhead` is on the critical path — that is, every microsecond of encoding delay directly extends the step's wall-clock time. In practice, async op mode (Chapter 2) already hides some encoding latency behind device execution, so the actual speedup will typically be somewhat less than this formula predicts. The formula gives an upper-bound estimate; measured post-trace results will land at or below it.

> **Note:** The formula inverts the expected relationship: a larger `dispatch_overhead` relative to `total_step_time` produces a larger speedup. When dispatch overhead equals 50% of step time, `speedup = T / (T - 0.5T) = T / 0.5T = 2.0x`. When dispatch overhead equals 10%, `speedup = T / 0.9T = 1.11x`. The formula makes diminishing returns explicit and calculable before you write any trace code.

---

## What Trace Eliminates

Trace bypasses phases 1–3 of host dispatch (argument validation, kernel selection, and command encoding) on every replay call. From the per-op dispatch budget of 17–63 us, trace eliminates:

| Phase | Cost range | Eliminated by trace? |
|---|---|---|
| Argument validation | 5–15 us | Yes — skipped entirely on replay |
| Kernel selection (warm) | 1–3 us | Yes — pre-selected at capture time |
| Command encoding | 10–40 us | Yes — pre-encoded in the trace buffer |
| CQ submission | 1–5 us | Partially — replaced by a single execute command |

Phase 4 (CQ submission) is not eliminated outright. Trace replay submits a single `ttnn.execute_trace` command to the CQ in place of N individual op commands. The submission cost for the execute command is still 1–5 us per step, regardless of how many ops the step contains. For a 32-op step, this replaces approximately 32–160 us of individual CQ submissions (32 ops × 1–5 us each) with a single 1–5 us submission — a ~6–160x reduction in per-step submission cost. This is not a complete elimination.

**Net effect per op on replay:** approximately 12–58 us of eliminated overhead. Chapter 3 (`trace_internals.md`) states that replay achieves a 36–288x reduction in per-op overhead cost, which corresponds to reducing per-op host cost from 17–63 us to roughly 0.2–0.5 us (the cost of the execute-command submission amortized over all ops).

---

## What Trace Cannot Eliminate

The speedup model applies only to dispatch overhead. The following components of `total_step_time` are unaffected by trace:

**Kernel execution time.** The time the device spends running the actual compute kernels is identical before and after tracing. Trace records and replays the same kernel binaries with the same arguments. If a matmul takes 200 us to execute on the device, it will take 200 us to execute on replay.

**Data movement latency.** DRAM read and write latency for tensor data is unchanged. The trace replays the same buffer addresses and the same DMA operations. Memory bandwidth is a hardware constant that trace does not affect.

**Host Python overhead.** Any Python code that runs between trace replay calls — preparing the next input, running the KV cache update, calling non-TTNN Python logic — is not part of the trace and continues to run at its pre-trace cost. If your decode loop has 500 us of Python overhead outside the traced region, that 500 us remains after tracing.

**Synchronization barriers.** Calls to `ttnn.synchronize_device()` between steps block the host until the device drains its command queue. These calls exist outside the trace boundary and are not recorded. If you add synchronization calls for debugging and forget to remove them, they will cap your achievable speedup regardless of how much dispatch overhead trace eliminates.

> **Warning:** A common mistake is to measure `total_step_time` with synchronization after every step (which is correct for measurement purposes) and then apply the speedup formula to predict production speedup — without accounting for the fact that production may use asynchronous-only synchronization at the end of a batch. Ensure the `total_step_time` denominator in your formula matches the production execution mode, not the measurement-only mode.

---

## Worked Example: 32-Op Decode Step

Consider a decode step with the following characteristics, measured using the TTNN profiler:

```
Model: small LLM, 1-token decode
Ops per step: 32
  - 8 x ttnn.matmul (attention projections, FFN)
  - 8 x ttnn.add
  - 4 x ttnn.softmax
  - 4 x ttnn.layernorm
  - 8 x ttnn.mul (elementwise scaling)

Measured per-op dispatch times (warm path):
  matmul:     avg 42 us  × 8 ops  =  336 us
  add:        avg  8 us  × 8 ops  =   64 us
  softmax:    avg 22 us  × 4 ops  =   88 us
  layernorm:  avg 18 us  × 4 ops  =   72 us
  mul:        avg  7 us  × 8 ops  =   56 us
                                   ─────────
  Total dispatch overhead (D):      616 us

Measured total step latency (T):  2,400 us  (2.4 ms)

Kernel execution total (K):       1,784 us  (2,400 us - 616 us)
Kernel occupancy:                 74.3%     (1,784 / 2,400)
```

Applying the speedup formula:

```
speedup = T / (T - D)
        = 2400 us / (2400 us - 616 us)
        = 2400 / 1784
        = 1.345x
```

The predicted post-trace step latency is:

```
post_trace_step_time = T / speedup
                     = 2400 us / 1.345
                     = 1,784 us  (1.78 ms)
```

Or equivalently: `T - D = 2400 - 616 = 1784 us`.

**Latency reduction: 616 us (25.7%) per step.**

For a model generating 1,000 tokens per inference request, this represents a saving of:

```
616 us × 1,000 tokens = 616 ms (0.616 s) per request
```

That is a meaningful reduction in time-to-first-token-stream-complete for a latency-sensitive deployment. At 1.345x speedup, trace implementation is almost certainly justified.

> **Example:** If the same 32-op step ran on heavier hardware where each matmul kernel took 600 us instead of 42 us, `T` would grow to roughly 15 ms (15,000 us) while `D` stays at 616 us. The speedup formula then gives `15000 / (15000 - 616) = 15000 / 14384 = 1.043x`. A 4.3% improvement from a trace implementation that requires significant structural refactoring is unlikely to be worth the cost.

---

## Diminishing Returns: When Kernel Time Dominates

The speedup formula makes explicit that speedup approaches 1.0 as `dispatch_overhead` shrinks relative to `total_step_time`. The following table shows predicted speedup at various overhead fractions, holding `total_step_time` constant:

| Dispatch overhead fraction | Example: T = 5 ms | Predicted speedup |
|---|---|---|
| 50% | D = 2,500 us | 2.00x |
| 40% | D = 2,000 us | 1.67x |
| 30% | D = 1,500 us | 1.43x |
| 20% | D = 1,000 us | 1.25x |
| 10% | D = 500 us | 1.11x |
| 5% | D = 250 us | 1.05x |
| 2% | D = 100 us | 1.02x |
| 1% | D = 50 us | 1.01x |

Three factors push a workload into the low-overhead-fraction regime and reduce trace's value:

1. **Long-running kernels.** Large matmuls on a full device grid (e.g., a 70B parameter model shard) can run for 2–10 ms per op. Even at 63 us dispatch overhead per op, a 10-op step running 5 ms per kernel has `D = 630 us`, `T = 50 ms`, and `speedup = 1.013x`. Trace provides no meaningful benefit.

2. **Fewer ops per step.** A step with 4 ops contributes at most 252 us of dispatch overhead (4 × 63 us). Unless the step is very fast overall (<2 ms total), this is likely under 10% of step time.

3. **High batch size.** As batch size grows, each kernel runs longer (more data to process) while dispatch overhead stays fixed per call. The same 32-op decode step at batch size 32 has 32x more device compute per op but the same 616 us of host dispatch overhead.

> **Note:** The diminishing-returns regime is not a reason to avoid measuring. Measuring first and finding that speedup would be <5% is a valuable outcome — it tells you exactly why trace is not the right tool for that workload, and it gives you a documented justification for that decision.

---

## Checklist: Does the Speedup Justify Implementation Cost?

Use the following checklist after computing your predicted speedup to decide whether to proceed with the trace implementation.

**Predicted speedup is a clear yes if:**
- Calculated speedup is ≥1.15x (dispatch overhead ≥13% of step time).
- The decode step runs thousands of times per inference request (long generation sequences).
- Your deployment is latency-sensitive and sub-millisecond improvements have business value.
- The model's decode loop already satisfies trace structural constraints (fixed shapes, no data-dependent dispatch, no host readbacks) with little or no refactoring.

**Predicted speedup is marginal — weigh carefully:**
- Calculated speedup is 1.05x–1.15x (dispatch overhead 5–13% of step time).
- The model requires moderate refactoring to fit trace constraints (e.g., moving one control-flow branch outside the capture boundary).
- Regression testing infrastructure is not yet in place; you will need to build it.
- The model is actively being developed; captured traces will need to be invalidated and re-captured as the architecture changes.

**Predicted speedup is not worth implementation cost if:**
- Calculated speedup is <1.05x (dispatch overhead <5% of step time).
- The model uses dynamic shapes, data-dependent branching inside the decode loop, or other patterns that make tracing structurally incompatible without major refactoring.
- The measured step latency is dominated by data movement (memory bandwidth saturation) rather than dispatch overhead.

> **Note:** Even when the per-step speedup is small, trace can reduce host CPU load in multi-device serving scenarios where the host thread is dispatching to many devices simultaneously. In those configurations, reducing per-op host encoding cost per device allows the host to serve more devices without becoming the bottleneck. This secondary benefit is not captured by the single-device speedup formula.

---

**Next:** [`profiling_workflow.md`](./profiling_workflow.md)
