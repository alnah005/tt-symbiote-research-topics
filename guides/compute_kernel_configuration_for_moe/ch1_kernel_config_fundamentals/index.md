# Chapter 1: Compute Kernel Config Fundamentals

## Prerequisites

- Basic TTNN matmul usage: constructing input tensors, calling `ttnn.matmul`, reading output tensors
- Familiarity with `bfloat16` as a dtype and its 7-bit mantissa / 8-bit exponent layout
- No prior knowledge of Tensix FPU pipeline internals required

---

## Overview

`ttnn.WormholeComputeKernelConfig` is the primary Python handle for controlling per-op compute behavior on Wormhole B0 hardware. Passing it to `ttnn.matmul` (or other compute-intensive ops) via the `compute_kernel_config` argument lets you trade numerical precision for throughput in a principled, per-operation way.

Without an explicit config, TTNN falls back to internal defaults. Those defaults are hardware-conservative: they avoid accumulation precision loss but do not optimize for throughput. For bandwidth-bound MoE expert matmuls — which is the regime for decode-mode inference — the default config leaves measurable performance on the table.

This chapter introduces the four primary fields of `WormholeComputeKernelConfig`, explains what each controls at the hardware level, and establishes the vocabulary used throughout the rest of this guide.

---

## Learning Objectives

After completing this chapter you should be able to:

1. Construct a `ttnn.WormholeComputeKernelConfig` object with explicit field values and pass it to `ttnn.matmul`.
2. Explain in one sentence what each of the four primary fields controls.
3. Distinguish the two canonical production configs (`COMPUTE_KERNEL_CONFIG_LOFI` and `COMPUTE_KERNEL_CONFIG_HIFI2`) and identify which MoE projection each is used for.
4. Describe why the default TTNN behavior (no `compute_kernel_config`) is suboptimal for decode-mode MoE workloads.

---

## Field Summary Table

See `wormhole_compute_kernel_config_api.md` for the complete field reference.

---

## Chapter Structure

| File | Contents |
|---|---|
| `index.md` (this file) | Overview, learning objectives, field summary table |
| `wormhole_compute_kernel_config_api.md` | API construction, field descriptions, Python examples, default behavior |
| `fp32_dest_acc_en.md` | Destination accumulator register deep dive; when fp32 precision matters for MoE |
| `math_fidelity_overview.md` | Math fidelity enum values, mantissa bit intuition, throughput/accuracy trade-off overview |

---

## Next Steps

Read `wormhole_compute_kernel_config_api.md` to see how the config object is constructed in Python and passed to `ttnn.matmul`, including the exact syntax for both canonical configs.
