# Chapter 1 — Tenstorrent Hardware and TTNN Foundations

## Overview

This chapter establishes the mental model that every subsequent chapter in the guide builds on. You will learn how a Tensix core is structured — its matrix engine, scalar unit, and memory system — and how TTNN expresses tensors and operations on top of that hardware. The three content files move from the physical substrate (the chip) upward through the programming abstraction (TTNN tensor layout and memory configs), and finally into the numerical precision system (math fidelity and block floating-point formats). Understanding these foundations is non-negotiable for interpreting the optimization choices made throughout tt-transformers: why weights are laid out in tiles, why sharding activations to L1 is beneficial, and why "LoFi" is not simply lower accuracy but a fundamentally different hardware execution mode.

---

## Prerequisites

Before reading this chapter, you should be comfortable with:

- Transformer architecture fundamentals: attention, MLP, layer norm, KV cache
- PyTorch tensor operations and GPU memory concepts (device memory, memory bandwidth)
- Block floating-point and quantization basics (INT8, FP16, BF16)
- The high-level distinction between prefill (processing a prompt) and decode (generating one token at a time)

You do **not** need prior Tenstorrent or TTNN experience.

---

## What You Will Be Able to Do After This Chapter

- Describe the internal structure of a Tensix core and explain why 32×32 tiles are the native computation unit
- Explain why L1 SRAM is preferred over DRAM for active computation tiles, and quantify the capacity difference
- Interpret and construct `ttnn.MemoryConfig` objects for interleaved and sharded tensor layouts
- Choose the correct data type (BF16, BFP8_B, BFP4_B) for a given weight matrix and know its memory footprint
- Understand the four math fidelity levels and predict their throughput impact on matmul-heavy ops
- Read a `WormholeComputeKernelConfig` in tt-transformers source code and understand each field

---

## Files in This Chapter

| File | Description |
|---|---|
| [tensix_architecture.md](./tensix_architecture.md) | Tensix core internals: matrix FPU, SFPU, Packer/Unpacker, Dst register, NoC, L1 vs DRAM |
| [ttnn_tensor_model.md](./ttnn_tensor_model.md) | TTNN tensor layout (TilizedLayout, RowMajorLayout), BufferType, TensorMemoryLayout, sharding API, data types |
| [math_fidelity_and_data_formats.md](./math_fidelity_and_data_formats.md) | Math fidelity modes (LoFi through HiFi4), WormholeComputeKernelConfig, packer_l1_acc, decision guide |

---

## Reading Order

Read the files in the order listed below. Each file builds on the previous one:

1. [tensix_architecture.md](./tensix_architecture.md) — Start here; all TTNN abstractions are designed around the hardware constraints described here.
2. [ttnn_tensor_model.md](./ttnn_tensor_model.md) — Read second; sharding strategies in later chapters rely on the APIs introduced here.
3. [math_fidelity_and_data_formats.md](./math_fidelity_and_data_formats.md) — Read last; the throughput vs accuracy trade-offs recur throughout Chapters 2–7.
