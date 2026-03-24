# Chapter 2 — How tt-transformers Uses Trace Capture

This chapter walks through the concrete mechanics of how the `Generator` class in `models/tt_transformers/tt/generator.py` uses the TTNN trace capture API to accelerate both decode and prefill phases. By the end of this chapter you will understand the full lifecycle of a trace — from the first-call capture that compiles and records a sequence of operations, through the per-step replay that re-executes that sequence with updated device-side tensors — and you will know which knobs in `model_config.py` and `trace_region_config.py` control when and how tracing is allowed.

---

## State Machine of a Generator Instance

The first call to each trace path runs a compile pass and then records the TTNN op graph; all subsequent calls replay the recorded graph. The files below walk through decode and prefill separately.

---

## What's Next

Read the files in the following order to build understanding progressively:

1. [`decode_trace_flow.md`](./decode_trace_flow.md) — how `decode_forward` routes to the trace path and the full capture-then-replay cycle for decode, including the `sampling_on_device` keying scheme and split-sampling variant.
2. [`prefill_trace_flow.md`](./prefill_trace_flow.md) — how `prefill_forward_text` decides whether a given sequence length is traceable and the capture-then-replay cycle for prefill.
3. [`model_config_trace_settings.md`](./model_config_trace_settings.md) — the per-device and per-model configuration tables that govern which sequence lengths are allowed, how large the trace region must be, and why a single command queue is required.
