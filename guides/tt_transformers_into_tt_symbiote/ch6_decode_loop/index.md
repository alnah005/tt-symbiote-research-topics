# Chapter 6: The Decode Loop — From Generator to Symbiote Inference

## Overview

LLM inference divides into two phases with very different compute profiles: **prefill** and
**decode**. Understanding how TT Transformers manages both phases — and where TT Symbiote
currently diverges — is the prerequisite for closing the performance gap between the two
systems.

## The two phases of LLM inference

**Prefill** processes the full prompt in parallel. The input is a 2-D token matrix
`[batch, seq_len]`; attention is causal but all positions are computed simultaneously.
Prefill is compute-bound and typically runs once per request.

**Decode** generates one token at a time. The input shrinks to `[batch, 1]`, but attention
must read the entire KV cache that was populated during prefill. Decode repeats hundreds or
thousands of times per request. It is memory-bandwidth-bound and its per-step latency
dominates end-to-end generation time.

## Why trace capture matters

Each decode step makes identical TTNN calls with new data but the same graph structure.
Without trace capture, Python must re-dispatch every op on every step: the TTNN command
queue is rebuilt from scratch, tensor handles are looked up, and host-to-device copies are
scheduled from Python. On a fast accelerator this Python overhead can be a significant
fraction of total step latency.

TTNN trace capture records the device command stream during a representative forward pass and
replays it with `ttnn.execute_trace`. After the trace is captured, Python does three things
per decode step: (1) copy new data into the pre-allocated input buffers,
(2) call `ttnn.execute_trace`, and (3) read the output tensor. The saved Python overhead
compounds across thousands of decode steps.

## What this chapter documents vs. what you need to implement

This chapter documents the **existing TT Transformers `Generator`** decode infrastructure
and the **current TT Symbiote inference path** as found in the source code. It does not
prescribe a complete implementation, but Section 4 (`integration_roadmap.md`) gives concrete
milestones and two integration options based on what actually exists in the code.

## Navigation

| File | Contents |
|------|----------|
| [`tt_transformers_generator.md`](tt_transformers_generator.md) | Full walkthrough of `Generator.__init__`, warmup, prefill trace, decode trace, `decode_forward`, paged attention, and the vLLM/SGLang extensions |
| [`symbiote_inference_path.md`](symbiote_inference_path.md) | How `test_llama.py` drives inference today, the `@trace_enabled`/`@trace_disabled` system, `TT_SYMBIOTE_RUN_MODE=TRACED`, KV cache handling, and dispatch overhead |
| [`integration_roadmap.md`](integration_roadmap.md) | Option A (minimal trace wiring inside `TTNNModule`), Option B (Symbiote-native `Generator`), recommended path, and ordered milestones |
