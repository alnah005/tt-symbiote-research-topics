# Latency-Sensitive Workloads

Trace is a latency optimization. This distinction matters: eliminating host encoding overhead reduces the wall-clock time of each individual step, but if your bottleneck is device compute throughput rather than host dispatch time, trace will not improve your benchmark numbers in any meaningful way. This file walks through the anatomy of a latency-sensitive workload, uses the autoregressive decode loop as the primary concrete example, contrasts it with the prefill pass where trace rarely pays off, and examines how batch size shifts the relative weight of host overhead versus device compute.

---

## Latency vs. Throughput: Why the Distinction Matters

Throughput measures how much work the system completes per unit of time — tokens per second across a sustained stream of requests, or requests processed per hour. Latency measures the elapsed time for a single unit of work to complete — the time from the start of a decode step to the completion of that step.

Trace exclusively reduces latency per step. The mechanism is straightforward: by removing phases 1–3 of dispatch from the host's critical path on each replay, the total time from "start replay" to "device finishes executing the recorded sequence" is shorter. No more device work is done; no kernels are made faster; no memory bandwidth is freed. The device executes the same kernels at the same speed. The reduction comes entirely from eliminating the host's contribution to the step timeline.

This has a direct implication for throughput: if you are batching many requests together and the bottleneck is device compute (the device is executing flat-out and the host is comfortably keeping up with encoding), trace will not improve tokens-per-second. The device was not waiting for the host; removing host overhead does not speed anything up. Trace helps when the host is on the critical path — when the device would be idle if it weren't for trace.

> **Note:** The regime where the host is on the critical path is exactly the regime described in Chapter 1's `host_dispatch_path.md`: when the device finishes each kernel before the host finishes encoding the next command, the device sits idle. Trace collapses that idle gap. If you have profiled your workload and the device is running at close to 100% utilization already, trace will not help.

---

## The Decode Loop as the Canonical Trace Workload

Autoregressive token generation is the workload for which trace was designed. Its properties align precisely with the conditions that make trace beneficial.

### Fixed shapes, every iteration

In the standard single-token decode case, each step generates exactly one token. The input tensor to the decode step has shape `[batch, 1, hidden_dim]` — the sequence length dimension is always 1 regardless of how many tokens have been generated so far. Weight tensors are unchanged. KV-cache buffers have a fixed allocated size (pre-allocated to the maximum context length). Every tensor the decode step touches has a shape determined by the model configuration constants, not by runtime input content. Because buffer addresses are allocated based on tensor size, and sizes do not change, buffer addresses remain constant across all steps — the address fixity requirement that Chapter 3's `trace_internals.md` establishes is trivially satisfied.

### High op count per step, short device kernels

A typical transformer layer involves dozens of individual TTNN op calls: multiple matmuls for the QKV projections and output projection, a softmax, elementwise adds for residual connections, layernorm operations, and FFN matmuls. A 32-layer model might dispatch 960 or more individual ops per decode step. At 17–63 us per op, total dispatch overhead per step is roughly 16–60 ms (16,000–60,000 us) in the absence of any optimization.

Meanwhile, each individual kernel on the device is short: a single-token attention computation over a small query vector has far less arithmetic than a prefill over a long sequence. Device time per step might be 2–8 ms (2,000–8,000 us). The ratio of dispatch overhead to device compute time is high — often 3:1 or worse — which means async op mode alone cannot fully hide the overhead. Even if the dispatch thread encodes op N+1 while the device executes op N, the encoding time (17–63 us per op, ~20–50 us typical) exceeds the execution time for short device kernels, and the device stalls waiting for the next command.

Trace inverts this dynamic entirely. After capture, each `ttnn.execute_trace` call submits a single high-level command to the CQ. The device fetches the pre-encoded buffer and executes the full sequence without any further host involvement. Host overhead for the entire step drops from 16–60 ms to the overhead of one `execute_trace` call plus the CQ submission latency — on the order of 7–15 us total.

### Thousands of repetitions amortize the capture cost

The capture run is slower than a normal dispatch run: it performs all four phases of dispatch for every op (live dispatch), simultaneously writes the encoded commands into the trace buffer, and then finalizes the buffer in device DRAM. This capture overhead is paid once. With thousands of decode steps, the one-time cost of capture is amortized across the entire generation sequence. Even a capture that takes 5–10 ms longer than a normal dispatch run becomes negligible when spread over 1,000 steps.

### Concrete timeline comparison

The following two timelines show a five-op sequence of one decode step, first with async dispatch (Chapter 2) and then with trace. Each unit of work on the dispatch thread takes ~25 us per op; each kernel on the device takes ~10 us. With these numbers, async dispatch alone cannot prevent device stalls between ops.

```
Async dispatch (no trace) — device stalls between short kernels
────────────────────────────────────────────────────────────────────────────────
Time ──────────────────────────────────────────────────────────────────────────►
     0us       25us      50us      75us      100us     125us     150us

Dispatch thread
├── enc A ──┤├── enc B ──┤├── enc C ──┤├── enc D ──┤├── enc E ──┤

Device
           ├─ A ─┤ [idle] ├─ B ─┤ [idle] ├─ C ─┤ [idle] ├─ D ─┤ [idle] ├─ E ─┤
           10us          10us          10us          10us          10us

Total wall-clock time to complete all 5 ops: ~135 us
Device utilization: ~37%  (50us execute / 135us elapsed)
```

```
Trace replay — single execute_trace command, no per-op encoding
────────────────────────────────────────────────────────────────────────────────
Time ──────────────────────────────────────────────────────────────────────────►
     0us    10us   20us   30us   40us   50us   60us

Host thread
├─ execute_trace (1 CQ write, ~5us) ─┤

Device
       ├─A─┤├─B─┤├─C─┤├─D─┤├─E─┤
       10us  10us  10us  10us  10us

Total wall-clock time to complete all 5 ops: ~55 us
Device utilization: ~91%  (50us execute / 55us elapsed)
```

The gap closes from 135 us to ~55 us — a 2.4x improvement in step latency for this simplified example — purely from removing encoding overhead. On a 32-layer model with 30 ops per layer and slightly longer kernels, the proportional improvement is similar.

---

## Contrasting Case: Prefill

The prefill pass processes the input prompt before the first decode token is generated. It is executed once per request, and its input shapes vary with prompt length.

### Why prefill rarely benefits from trace

**Variable sequence length.** The defining characteristic of prefill is that the prompt sequence length S varies per request. Tensors produced by the QKV projections have shape `[batch, heads, S, head_dim]`; the attention score matrix has shape `[batch, heads, S, S]`, which grows quadratically with S. As established in Chapter 3's `trace_constraints.md` (Category 1: Dynamic Shapes), shapes that vary between calls produce tensors at different DRAM addresses, invalidating the buffer bindings encoded in the trace. There is no safe way to replay a prefill trace across requests with different prompt lengths without re-capturing for each distinct length.

**Single execution.** Even if you constrain all prompts to a fixed padded length and capture a trace for that fixed shape, you pay the capture cost once per session and gain the replay savings for... one execution per request. A prefill that takes 50 ms (50,000 us) in device compute time has dispatch overhead of perhaps 1–5 ms. Eliminating that 1–5 ms per request produces at best a 2–10% improvement on a 50 ms operation. The trace machinery — buffer pinning, shape fixity requirements, debugging difficulty (see `when_not_to_trace.md`) — is hard to justify for a 2–10% gain on a once-per-request operation.

**Kernels dominate prefill time.** Prefill processes S tokens in parallel, which means each matmul and attention kernel processes a large input and runs for a correspondingly long time on the device. Kernel execution time dominates; host dispatch overhead is a small fraction. The "dispatch overhead fraction" from Chapter 1's formula — (num_ops x ~40 us) / total_step_time — yields a small percentage for long-sequence prefill because total_step_time is large. With a low overhead fraction, trace produces a proportionally small speedup.

For a full comparison, see the prefill/decode table in [`trace_constraints.md`](../ch3_trace_capture/trace_constraints.md).

> **Note:** Some deployments use fixed-length padding to make prefill shapes static and then capture a trace for the padded shape. This is a valid technique if the padding overhead (wasted compute on pad tokens) is acceptable and if the deployment always uses the same prompt length. Measure actual prefill throughput with padding before assuming the shape-fixity tradeoff is worth it.

---

## Batch Size and the Host Overhead Fraction

Batch size is the other major axis on which the value of trace changes. Host dispatch overhead per step is independent of batch size: dispatching a matmul on a `[1, seq=1, hidden_dim]` tensor costs the same 17–63 us as dispatching the same matmul on a `[32, seq=1, hidden_dim]` tensor, because encoding cost scales with structural complexity (number of cores, number of buffer bindings) rather than data volume.

Device execution time, however, scales with batch size: a batch of 32 sequences processes 32 times more data than a batch of 1, and the device kernel runs proportionally longer. This means:

- **At batch size 1**, device execution per step is shortest and dispatch overhead represents its largest fraction of total step time. Trace has the highest relative impact.
- **At larger batch sizes**, device execution time grows while dispatch overhead stays flat. The fraction of step time attributable to dispatch overhead shrinks. Eventually, at large enough batch sizes, the device is the bottleneck and trace provides diminishing returns.

The crossover point depends on the model size, the device hardware, and the specific op mix. A reasonable empirical approach is to profile your workload at the target batch size and measure the device utilization (or equivalently, the gap between dispatch completion and kernel completion that a timeline profiler shows). If the device is sitting idle waiting for encoded commands, trace will help. If the device is running continuously and the host is the idle party, trace is not the bottleneck solution.

```
Illustrative: dispatch overhead fraction vs. batch size
────────────────────────────────────────────────────────────────────────────────

Assumptions:
  - 64 ops per decode step
  - ~35 us average dispatch overhead per op  -->  ~2.24 ms total dispatch per step
  - Device execution time scales linearly with batch size
  - At batch size 1, device execution = 2 ms per step

Batch  Device exec   Dispatch OH   Total step   Dispatch fraction   Trace benefit
─────  ───────────   ───────────   ──────────   ─────────────────   ─────────────
  1      2.0 ms        2.2 ms       4.2 ms           53%              High
  4      8.0 ms        2.2 ms      10.2 ms            22%              Moderate
 16     32.0 ms        2.2 ms      34.2 ms             6%              Low
 64    128.0 ms        2.2 ms     130.2 ms             2%              Negligible
```

> **Note:** The numbers above are illustrative. Device execution does not scale perfectly linearly with batch size (KV-cache reads and attention patterns create non-linearities), and dispatch overhead is not strictly constant (some ops encode differently for larger batches). Measure your specific workload. The principle holds: as batch size grows, the relative benefit of trace shrinks.

> **Warning:** Even at batch sizes where trace provides only a moderate speedup, the structural constraints trace imposes (buffer pinning, shape fixity) remain in full force. If the expected latency reduction is small, weigh it carefully against the maintenance cost described in `when_not_to_trace.md`.

---

**Next:** [`when_not_to_trace.md`](./when_not_to_trace.md)
