# Chapter 5 — Latency Cost of synchronize_device and Throughput Impact

Chapters 3 and 4 established that `ttnn.synchronize_device()` inside `_maybe_all_gather` is unnecessary for correctness and that it is present in at least two forward-path call sites (`TTNNQwen3FullAttention._maybe_all_gather` and `TTNNQwen3LinearAttention._maybe_all_gather`) that are trace-blocking on the critical path of the hybrid decoder stack. This chapter answers the natural follow-on question: how much does the call actually cost, and what throughput improvement can be expected from removing it?

---

## Context

The cost of `ttnn.synchronize_device()` is not fixed. It varies with how far ahead the host has submitted work to CQ0 relative to where the device is currently executing. At decode batch=1 on T3K, device kernels are short (typically under 50 µs each); the host tends to stay close to the device in the command queue. In this regime the dominant component of `synchronize_device` latency is the PCIe round-trip overhead of flushing the Finish token and receiving the device acknowledgment, not kernel completion time. This makes each `synchronize_device` call a predictable, approximately fixed overhead — and one that compounds multiplicatively with the number of attention layers in the stack.

The figures in this chapter are estimates derived from known PCIe latency characteristics for Wormhole on T3K. They must be validated against measured profiling output before being cited as authoritative. The files in this chapter describe both the model and the measurement procedure needed to obtain confirmed values.

---

## Answer First

At decode batch=1 on T3K, a single `ttnn.synchronize_device()` call costs approximately **0.1–0.5 ms** (100–500 µs), dominated by PCIe round-trip latency. With two `_maybe_all_gather` call sites per hybrid decoder layer (one in `TTNNQwen3FullAttention`, one in `TTNNQwen3LinearAttention`), the total overhead per decode step is approximately **N × 0.2–1.0 ms**, where N is the number of hybrid attention layers in the stack. For a model with 16 such layers this estimates to **3.2–16 ms per step** — a significant fraction of a 30 ms decode budget. Removing these calls is a prerequisite for Metal Trace enablement and delivers a measurable latency improvement even in non-traced mode.

---

## What's Next

Read the following files in order:

1. [`synchronize_device_latency_model.md`](./synchronize_device_latency_model.md) — The latency components that contribute to a `ttnn.synchronize_device()` call at decode batch=1, with per-component estimates for Wormhole on T3K.

2. [`measuring_the_cost.md`](./measuring_the_cost.md) — How to measure the actual cost using Tracy or Python wall-clock instrumentation, with expected order-of-magnitude results and guidance on interpreting profiling output.

3. [`throughput_improvement_estimate.md`](./throughput_improvement_estimate.md) — A decode throughput model that translates the per-call latency into per-step and per-token improvements, including a worked example for a 16-layer hybrid attention stack.
