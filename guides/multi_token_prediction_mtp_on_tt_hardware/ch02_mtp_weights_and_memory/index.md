# Chapter 2 — MTP Head Weight Shapes and Memory Footprint

## Prerequisites

- Chapter 1 [`index.md`](../ch01_mtp_foundations/index.md)

Chapter 1 established the MTP head architecture (one transformer decoder block with dense FFN, GQA 64/8 with `head_dim` = 112, `hidden_size` = 7168, `intermediate_size` = 2048), the weight key naming convention (`model.future_prediction.0.*`), and the distinction between the MTP head's dense FFN and the backbone's sparse MoE FFN layers. This chapter builds directly on those findings.

---

## Goal

Produce a concrete memory budget for the MTP head that informs placement decisions (Chapter 5, `memory_placement_for_mtp.md`). The key questions answered here are:

1. What are the exact weight tensor shapes and total parameter count for the MTP head?
2. How large are the MTP head weights in BF16, and how does that compare to the backbone?
3. What is the FLOP cost of one MTP head forward pass at decode, and what fraction of total model compute does it represent?

---

## Summary Finding

> **MTP head (Qwen3.6-35B-A3B, `mtp_num_hidden_layers: 1`)**
>
> | Metric | Value |
> |--------|-------|
> | Total MTP weight parameters | ~159.67M |
> | BF16 weight memory | ~304.6 MiB |
> | Fraction of full model parameters | ~0.45% |
> | FLOPs per decode step (batch=1) | ~319M |
> | MTP FLOPs as fraction of one backbone block | ~34% |

The MTP head adds approximately 160 million parameters and 305 MiB of BF16 weight memory. At decode (batch=1), it is entirely memory-bandwidth-bound with an arithmetic intensity near 1.0 FLOPs/byte. The practical overhead of running the MTP head is dominated by the time to stream its weights from DRAM, not by compute.

---

## Chapter Files

| File | Contents |
|------|----------|
| [`mtp_weight_inventory.md`](./mtp_weight_inventory.md) | All weight tensors, shapes, parameter count |
| [`mtp_memory_footprint.md`](./mtp_memory_footprint.md) | BF16 memory, backbone comparison, L1 feasibility |
| [`mtp_vs_backbone_compute_cost.md`](./mtp_vs_backbone_compute_cost.md) | FLOP count, arithmetic intensity, overhead fraction |
