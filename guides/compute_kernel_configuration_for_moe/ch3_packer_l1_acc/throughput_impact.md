# Quantitative Throughput Impact of `packer_l1_acc`

## Bandwidth Reduction Formula

Enabling `packer_l1_acc=True` eliminates the redundant per-iteration DRAM read-modify-write traffic described in [`tensix_packer_pipeline.md`](./tensix_packer_pipeline.md). The fractional reduction in redundant DRAM reads per output tile is:

```
Redundant read reduction = (K_t/b − 1) / (K_t/b)
                         = 1 − (b / K_t)
```

where K_t = K/32 is the K-dimension tile count and `b` = `in0_block_w` is the number of K tiles consumed per outer-loop iteration.

| K_t | `in0_block_w` (b) | Outer-loop iterations | Redundant reads (off) | Redundant reads (on) | Reduction |
|---|---|---|---|---|---|
| 64 | 1 | 64 | 63 | 0 | 98.4% |
| 64 | 4 | 16 | 15 | 0 | 93.8% |
| 32 | 1 | 32 | 31 | 0 | 96.9% |
| 16 | 1 | 16 | 15 | 0 | 93.8% |
| 8 | 1 | 8 | 7 | 0 | 87.5% |

For the K_t = 64, b = 1 case (common in MoE expert weight matrices): 63 out of 64 DRAM reads per output tile are eliminated — a **98.4% reduction in redundant DRAM read traffic**.

> **Tip:** The formula shows that even for relatively small K_t values (K_t = 8), enabling `packer_l1_acc=True` still eliminates 87.5% of the redundant reads. The benefit is always positive; there is no K_t threshold below which enabling the flag becomes harmful.

---

## Regime Analysis: Where `packer_l1_acc` Matters Most

### Decode-Mode MoE (High Impact)

In decode-mode inference, the batch size M is small — typically 1 to 32 sequences. This makes the matmul shape [M, K] × [K, N] extremely skinny in the M dimension:

- M_t = M/32 is often 1 (for batch size 1–32)
- The matmul is **memory-bandwidth-bound**: arithmetic intensity is low because there are few output tiles relative to the weight matrix that must be read from DRAM

In this regime, DRAM bandwidth is the dominant bottleneck. Each redundant DRAM read caused by `packer_l1_acc=False` directly competes with useful weight-loading traffic for the same bandwidth budget. Eliminating 63 redundant reads per output tile (K_t = 64, b = 1) directly translates into lower latency.

**Expected end-to-end matmul speedup for decode-mode MoE:**

| Workload | Estimated speedup from enabling `packer_l1_acc` |
|---|---|
| K_t = 64, b = 1, bandwidth-bound | 10–40% matmul latency reduction |
| K_t = 32, b = 1, bandwidth-bound | 8–25% matmul latency reduction |
| K_t = 64, b = 4, bandwidth-bound | 5–20% matmul latency reduction |

The wide ranges reflect variation in DRAM utilization, core count, and whether other bottlenecks (e.g., NoC (Network-on-Chip) routing, dispatch overhead) are present.

> **Tip:** For MoE decode workloads, `packer_l1_acc=True` is among the highest-ROI single-field changes in `WormholeComputeKernelConfig`. It should be set before tuning `math_fidelity` or `in0_block_w`.

### Prefill with Large Sequence Length (Lower Impact)

In prefill mode with sequence length ≥ 512, M_t = M/32 ≥ 16. The matmul becomes more **compute-bound**:

- FPU utilization is high
- DRAM bandwidth is still used for weight loading, but the ratio of useful DRAM reads (weight tiles) to computation is more favorable
- Redundant partial-sum DRAM traffic from `packer_l1_acc=False` is still present but represents a smaller fraction of total execution time

For prefill, enabling `packer_l1_acc=True` is still beneficial and never harmful, but the measured speedup is typically smaller — often 2–8% for large M.

---

## DeepSeek-V3 TTNN Reference Configurations

The DeepSeek-V3 TTNN implementation provides concrete evidence that `packer_l1_acc=True` is the correct default across both LoFi (Low Fidelity) and HiFi2 (High Fidelity 2) configurations. The LoFi and HiFi2 configurations used in DeepSeek-V3 are shown in full in [`packer_l1_acc_constraints.md`](./packer_l1_acc_constraints.md)'s Safe Default Configurations section.

Key observations:

1. `packer_l1_acc=True` is set in **both** configs — the improvement applies at every fidelity level.
2. The LoFi config (with `fp32_dest_acc_en=False`) accumulates in bfloat16 (2 bytes/element), using a smaller L1 buffer.
3. The HiFi2 config (with `fp32_dest_acc_en=True`) accumulates in fp32 (4 bytes/element), using a larger L1 buffer. See [`packer_l1_acc_constraints.md`](./packer_l1_acc_constraints.md) for the L1 budget implications.
4. The DeepSeek-V3 team did not make `packer_l1_acc` conditional on workload shape — it is unconditionally `True` in production.

---

## Why It Is Safe to Enable Unconditionally

The throughput argument for `packer_l1_acc=True` is strictly non-negative:

- If the matmul is memory-bandwidth-bound (decode mode): large speedup from eliminating redundant DRAM traffic.
- If the matmul is compute-bound (large-M prefill): negligible impact; no slowdown.
- The FPU computation itself is identical in both modes; only the memory traffic pattern changes.

The only risk is L1 overflow, which is a configuration error detectable at dispatch time — not a correctness risk or a silent performance regression. This risk is fully covered in the next file.

> **Warning:** Do not leave `packer_l1_acc=False` in production decode-mode configurations as a "safe default." The default value is a legacy of conservative hardware bring-up and does not reflect the optimal setting for Wormhole matmul workloads.

---

**Next:** [`packer_l1_acc_constraints.md`](./packer_l1_acc_constraints.md)
