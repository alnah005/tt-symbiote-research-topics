# Architecture Summary: Qwen3.5-35B and T3K Parameters

This file collects the model and hardware constants that drive all expert parallelism decisions.
All values derive from prior chapters; references are given for each parameter.

---

## 1. Qwen3.5-35B MoE Layer Parameters

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| Total experts per MoE layer | $E$ | 256 | Chapter 1, `qwen35b_config.md` |
| Top-k routing | $k$ | 8 | Chapter 1, `qwen35b_config.md` |
| Hidden dimension | $H$ | 7168 | Chapter 1, `qwen35b_config.md` |
| Expert FFN intermediate dim | $D$ | [VERIFY: exact value for Qwen3.5-35B] | Chapter 1, `qwen35b_config.md` |
| Router weight matrix shape | $[H, E]$ | $[7168, 256]$ | Chapter 5, `router_forward_pass.md` |
| Router weight size (BF16) | — | $7168 \times 256 \times 2 = 3{,}670{,}016$ bytes $\approx 3.67$ MB | Chapter 5, `router_forward_pass.md` |
| Router weight size (INT8) | — | $7168 \times 256 \times 1 \approx 1.84$ MB | Chapter 5, `router_kernel_fusion.md` |
| Routing normalization | — | Sigmoid (not softmax); renormalize top-k after selection | Chapter 5, `weight_normalization.md` |
| Capacity factor (default) | $\text{CF}$ | 1.25 | Chapter 7, `capacity_overflow_handling.md` |
| Expert capacity (B=32, decode) | $C$ | $\lceil 8 \times 32 \times 1.25 / 256 \rceil = 2$ | Chapter 7, `capacity_factor_mechanics.md` |
| Expert capacity (B=1, decode) | $C$ | $\lceil 8 \times 1 \times 1.25 / 256 \rceil = 1$ | Chapter 7, `capacity_factor_mechanics.md` |

### Expert Weight Size Per Device

Under uniform EP ($E_d = 32$ experts per device), each device stores three weight matrices per expert
($W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$) in BF16:

$$\text{Expert weight size per device} \approx 3 \times E_d \times H \times D \times 2 \text{ bytes} = 3 \times 32 \times 7168 \times D \times 2$$

With $D$ [UNVERIFIED], this totals several GB per device. Expert weights are always DRAM-resident;
they cannot fit in L1 (120 MB aggregate per Wormhole B0 chip). See Chapter 4, `uniform_partitioning.md`.

### MoE vs. Dense Layers

[VERIFY: confirm which transformer blocks in Qwen3.5-35B use MoE FFN vs. dense FFN. The MoE layer
forward pass described here applies only to MoE blocks.]

---

## 2. T3K Hardware Parameters

| Parameter | Value | Source |
|---|---|---|
| Devices | 8 (Wormhole B0) | Chapter 2, `collective_communication_background.md` |
| Topology | 1×8 linear mesh | Chapter 2, `collective_communication_background.md` |
| Ethernet bandwidth per link | ~12.5 GB/s | Chapter 2, `dispatch_combine_overhead.md` |
| Ethernet links per device | 1 (endpoints) or 2 (interior devices) | Chapter 2 |
| Average hop count (1×8 mesh) | 3.0 | Chapter 2, `collective_communication_background.md` |
| Tensix cores per chip | 80 | Chapter 6, `expert_ffn_tiling.md` |
| L1 SRAM per core | ~1.5 MB | Chapter 6, `expert_ffn_tiling.md` |
| Aggregate L1 per chip | ~120 MB | Chapter 6, `expert_ffn_tiling.md` |
| DRAM bandwidth per chip | ~300 GB/s [UNVERIFIED] | Chapter 6 |

---

## 3. Derived Constraints Under Uniform EP

These values are derived from the constants above for the standard decode regime ($B=32$, $S=1$).

### Expert Assignment

$$E_d = E / N = 256 / 8 = 32 \text{ experts per device}$$

This is exact: 256 divides evenly by 8. No fractional expert assignment is needed.
See Chapter 4, `uniform_partitioning.md`.

### Token Capacity and Dispatch Volume

$$C = \left\lceil \frac{k \times B \times \text{CF}}{E} \right\rceil = \left\lceil \frac{8 \times 32 \times 1.25}{256} \right\rceil = \left\lceil 1.25 \right\rceil = 2$$

$$V_{\text{dispatch}} = (N-1) \times C \times E_d \times H \times 2 \text{ bytes} = 7 \times 2 \times 32 \times 7168 \times 2 = 6{,}422{,}528 \text{ bytes} \approx 6.4 \text{ MB}$$

At 12.5 GB/s per link: $t_{\text{dispatch}} \approx 6.4 / 12{,}500 \approx 0.51$ ms (one direction).
See Chapter 2, `dispatch_combine_overhead.md`.

### Core Utilization at Decode

```
Active cores per chip:  E_d × 2 cores/expert = 32 × 2 = 64  (16 idle)
per_core_M:             ceil(C / 32) = ceil(2 / 32) = 1
Token utilization:      C / 32 = 2/32 = 6.25%
```

At $B=32$, only 6.25% of each core's tile rows contain real token data; the remainder are zero-padded.
`sparse_matmul` is required to skip zero tile rows. See Chapter 6, `expert_ffn_tiling.md`.

### Double-Buffer Size for Fused Pipeline

$$M_{\text{double}} = 2 \text{ stages} \times 4 \text{ buffers/stage} \times C \times E_d \times H \times 2$$

At $C=2$: $8 \times 2 \times 32 \times 7168 \times 2 = 7{,}340{,}032$ bytes $\approx 7.0$ MB.
See Chapter 6, `pipeline_design.md`.

### Bottleneck Regime

| Component | Time (B=32) | Regime indicator |
|---|---|---|
| Dispatch all-to-all | ~0.51 ms | **Communication-bound** |
| Combine all-to-all | ~0.51 ms | **Communication-bound** |
| Expert FFN (32 experts, C=2) | [UNVERIFIED — depends on D] | Likely < A2A at B=32 |
| Router matmul [B, H] × [H, 256] | Sub-ms (~117 MFLOPs) | Negligible |

At $B \leq 32$, the workload is **communication-bound**: Tensix cores idle during A2A transfers.
See Chapter 6, `end_to_end_latency_model.md`.

---

## References

- Chapter 1, `qwen35b_config.md` — Qwen3.5-35B architectural constants
- Chapter 2, `collective_communication_background.md` — T3K topology; bandwidth model
- Chapter 2, `dispatch_combine_overhead.md` — dispatch/combine latency at B=32
- Chapter 4, `uniform_partitioning.md` — 32-expert-per-device baseline
- Chapter 5, `router_forward_pass.md` — W_r dimensions and sizing
- Chapter 6, `expert_ffn_tiling.md` — core utilization and sparse_matmul applicability
- Chapter 6, `pipeline_design.md` — double-buffer formula
- Chapter 7, `capacity_factor_mechanics.md` — C formula; CF=1.25 default

---

**Next:** [recommended_configuration.md](./recommended_configuration.md)
