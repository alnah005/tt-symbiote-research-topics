# Chapter 2 — GatedDeltaNet: Linear Attention on Blackhole

## Overview

This chapter is a deep dive into the `GatedDeltaNet` module — the building block that handles
three out of every four layers in Qwen3.5. It covers the mathematical recurrence that defines
linear attention, how weights are fused and structured on device, the hardware constraint that
forces the recurrence onto the host CPU, and the fully-fused Metalium kernel that resolves that
constraint once Metal Trace is available.

## Prerequisites

- Chapter 1: layer type dispatch, DeltaNet vs full-attention hyperparameters, GQA ratio.
- Familiarity with `ttnn.linear`, `ttnn.from_torch`, `ttnn.TILE_LAYOUT`, and device memory configs.

No prior knowledge of linear attention, the DeltaNet algorithm, or SSM (state-space model)
recurrences is assumed. This chapter builds those concepts from scratch.

## Reading Order

| File | Contents |
|------|----------|
| [`recurrence_math.md`](./recurrence_math.md) | The five DeltaNet equations, gating, L2 norm, GQA expansion, gated RMSNorm |
| [`projections_and_conv.md`](./projections_and_conv.md) | Fused `in_proj_all` matmul, circular conv ring buffer, state initialization |
| [`host_recurrence.md`](./host_recurrence.md) | Blackhole fp32 CB constraint, why bf16 hangs, the host float32 recurrence path |
| [`fused_kernel.md`](./fused_kernel.md) | `ttnn.experimental.gated_delta_net` kernel, PCC results, path to production |

## Forward References

- Chapter 6 (weight precision): dtype choices for `in_proj_all` (bfp8_b) and state tensor (float32).
- Chapter 4 (decoder block): how `DeltaNetDecoderBlock` calls `initialize_states` and dispatches
  to `GatedDeltaNet.forward`.
