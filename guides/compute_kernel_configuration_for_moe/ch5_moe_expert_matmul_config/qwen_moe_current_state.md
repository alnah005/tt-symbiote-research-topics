# Qwen MoE Current State

## Overview

This file describes the current state of `compute_kernel_config` usage in the Qwen MoE expert forward pass and explains why the absence of explicit configs leaves a measurable performance gap. The analysis uses Qwen 2.5 MoE dimensions as the concrete reference case.

---

## Current State: No Explicit `compute_kernel_config`

The Qwen MoE expert matmul ops do not pass a `compute_kernel_config` argument. This means every expert matmul — gate, up, and down projections — falls back to the device-level defaults. On Wormhole hardware, the device defaults are:

| Field | Device default |
|---|---|
| `math_fidelity` | `LoFi` |
| `math_approx_mode` | `False` |
| `fp32_dest_acc_en` | `False` |
| `packer_l1_acc` | `False` |

The fidelity default (LoFi) happens to be correct for gate/up projections. However, **`packer_l1_acc=False` is the wrong default for all three projections in decode mode**, and **`math_fidelity=LoFi` is suboptimal for the down projection**, which accumulates into the residual stream.

---

## Qwen 2.5 MoE Dimensions

Reference model: Qwen2-235B-A22B (Qwen MoE 235B-A22B).

| Parameter | Value |
|---|---|
| `d_model` | 7168 |
| `d_ff` per expert | 2048 |
| Number of experts | 128 |
| `top_k` | 8 |
| K tiles for down projection (`K_t = d_ff / 32`) | 64 |
| Decode batch size (`b`) | 1 |

---

## Cost of `packer_l1_acc=False`

For Qwen's down projection at decode, K_t=64 yields the same **98.4% bandwidth reduction** established for DeepSeek-V3; for the full derivation, see Chapter 3, `throughput_impact.md`.

For gate and up projections, K_t = d_model / 32 = 7168 / 32 = 224 tiles. The bandwidth savings from enabling `packer_l1_acc=True` on those projections: 223 / 224 ≈ **99.6%**.

---

## Cost of LoFi for Down Projection

The device default assigns LoFi to the down projection. LoFi uses one accumulation pass per output tile. For the down projection, which adds directly to the residual stream, single-pass accumulation introduces rounding drift that compounds across MoE layers. HiFi2 (two passes) reduces that drift and is the appropriate choice.

The accuracy cost of LoFi on the down projection is not catastrophic — PCC will still be high for a single forward pass — but layer-by-layer residual accumulation means the error is not absorbed by a nonlinearity. Over many layers, LoFi drift on the down projection degrades output quality measurably relative to a HiFi2 baseline.

---

## Summary of Missing Optimizations

| Projection | Current (default) | Recommended | Gap |
|---|---|---|---|
| gate (w1) | LoFi, `packer_l1_acc=False` | LoFi, `packer_l1_acc=True` | Missing 99.6% DRAM bandwidth reduction |
| up (w3) | LoFi, `packer_l1_acc=False` | LoFi, `packer_l1_acc=True` | Missing 99.6% DRAM bandwidth reduction |
| down (w2) | LoFi, `packer_l1_acc=False` | HiFi2, `packer_l1_acc=True` | Missing fidelity upgrade and 98.4% DRAM bandwidth reduction |

The fix requires adding two config objects and passing them to three matmul calls. The code change is minimal; the expected latency improvement for decode-mode expert matmuls is measurable. See `applying_configs_to_qwen.md` for the step-by-step implementation.

---

## Why the Default Exists

Device defaults are conservative: `packer_l1_acc=False` avoids any risk of L1 overflow for unexpected tensor shapes or large `per_core_N` tile counts. For production MoE models where the shapes are known and the L1 budget has been verified, overriding the default with `packer_l1_acc=True` is safe and beneficial. DeepSeek-V3's explicit config definitions are the model to follow.
