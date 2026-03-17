# Chapter 3: `packer_l1_acc` — Throughput Effect

## Overview

This chapter covers the `packer_l1_acc` field of `WormholeComputeKernelConfig` and explains how it controls whether partial sums are accumulated in L1 SRAM (Static Random-Access Memory) or written back to DRAM (Dynamic Random-Access Memory) after each outer-loop iteration of a matmul. For decode-mode Mixture-of-Experts (MoE) workloads, enabling `packer_l1_acc` is one of the highest-leverage single-field changes available.

> **Tip:** Enable `packer_l1_acc=True` for almost all matmul-dominant workloads. The throughput benefit is always non-negative; the only risk is L1 overflow, which is detectable at dispatch time and resolvable by reducing tile counts.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Explain what the Tensix packer stage does and where it sits in the compute pipeline — specifically, how it serializes output tiles from the destination register and writes them to memory.
2. Describe what `packer_l1_acc=False` (the default) causes: after each outer-loop K-step the packer writes partial sums out to DRAM, and the next iteration must re-read them from DRAM to continue accumulation, creating unnecessary DRAM round-trips.
3. Describe what `packer_l1_acc=True` does: partial sums are accumulated into an L1 buffer local to the core; DRAM is written only once when the full K-dimension accumulation is complete, eliminating per-iteration DRAM read-modify-write traffic.
4. Quantify the bandwidth reduction for decode-mode matmuls: disabling `packer_l1_acc` causes redundant DRAM reads per output tile; enabling it reduces that count to zero. For the quantified example (K_t=64, b=1 → 63 extra reads), see [`throughput_impact.md`](./throughput_impact.md).
5. Identify when `packer_l1_acc` can cause L1 overflow and how to detect it: large `per_core_N` tile counts push the accumulation buffer past the 1.5 MB/core L1 capacity; TTNN raises an allocation error at op dispatch time.
6. Explain the interaction with `fp32_dest_acc_en`: when both flags are enabled the accumulation buffer holds fp32 values (4 bytes/element), requiring 2× the L1 space compared to bfloat16 (2 bytes/element); this combination is more likely to trigger L1 overflow.

---

## Quick Reference

| Configuration | `packer_l1_acc` | Accumulation location | DRAM writes per output tile |
|---|---|---|---|
| Default (off) | `False` | DRAM | K_t / `in0_block_w` |
| Recommended (on) | `True` | L1 | 1 |
| With fp32 dest acc | `True` + `fp32_dest_acc_en=True` | L1 (fp32) | 1 (but 2× L1 buffer size) |

---

## Prerequisites

- **Chapter 1** — Tensix compute pipeline overview: FPU (Floating-Point Unit), packer/unpacker stages, destination register, and the role of L1 SRAM in the Tenstorrent Wormhole memory hierarchy.
- Basic understanding of L1 SRAM vs. DRAM hierarchy: L1 is per-core on-chip SRAM (~1.5 MB/core); DRAM is off-chip with much higher access latency and lower bandwidth relative to L1.

---

## Chapter Contents

| File | Topic |
|---|---|
| [`tensix_packer_pipeline.md`](./tensix_packer_pipeline.md) | The Tensix packer stage: default vs. accumulation mode, bandwidth model, K-loop walk-through |
| [`throughput_impact.md`](./throughput_impact.md) | Quantitative impact: bandwidth reduction formula, decode vs. prefill regimes, DeepSeek-V3 example |
| [`packer_l1_acc_constraints.md`](./packer_l1_acc_constraints.md) | L1 budget constraints, overflow detection, interaction with `fp32_dest_acc_en`, safe defaults |

---

## Next Steps

Continue to [`tensix_packer_pipeline.md`](./tensix_packer_pipeline.md) for a detailed walkthrough of where the packer sits in the Tensix pipeline and how the two accumulation modes differ at the hardware level.
