# fp32_dest_acc_en: Destination Accumulator Precision

## Overview

The `fp32_dest_acc_en` field controls the precision of the FPU destination register — the internal register array where partial sums accumulate during K-loop iteration. It is a boolean that costs register file space in exchange for preventing bfloat16 rounding of partial sums between K tiles.

This document explains:

- What the destination register is and where it sits in the Tensix FPU pipeline
- What happens to partial sums with `fp32_dest_acc_en=False` (default)
- What changes with `fp32_dest_acc_en=True`
- When the difference is measurable for MoE expert matmuls
- How it interacts with `packer_l1_acc`
- The cost: reduced simultaneously live accumulation registers

---

## The Tensix FPU Destination Register

The Tensix FPU pipeline processes matrix multiply-accumulate operations tile by tile. For a matmul `[M, K] x [K, N]`, the K dimension is divided into tiles of width 32 (K_t = K/32 tiles). The FPU processes these K tiles in a loop, adding each tile's contribution into a running partial sum.

That running partial sum lives in the **destination register** (often written as the `dst` register in Tensix documentation). The destination register is a fixed-size on-chip register file — not L1 SRAM — that holds the in-progress output tiles for the current compute block.

The pipeline stages for each K tile are approximately:

1. **Unpack** — read two input tiles (one from A, one from B) from L1 into the source registers
2. **FPU compute** — multiply-accumulate: `dst[i,j] += A_tile * B_tile`
3. **Pack** — once all K tiles are accumulated, serialize `dst` tiles into L1 (and eventually DRAM)

The `fp32_dest_acc_en` field controls the numeric format of `dst` during step 2.

---

## Default Behavior: `fp32_dest_acc_en=False`

With `fp32_dest_acc_en=False`, the destination register stores values in **bfloat16** format (1 sign bit, 8 exponent bits, 7 mantissa bits).

After each K tile's contribution is added to the running sum, the result is rounded to bfloat16 before the next tile is processed:

```
dst_bf16 += A_tile * B_tile    # dst rounded to bfloat16 after each accumulation
```

For a K-loop with K_t tiles, this means K_t - 1 rounding events occur on the running sum. Each rounding discards information below the 7th mantissa bit of the current partial sum.

For small K (e.g., K=256, K_t=8), these rounding events are typically negligible — bfloat16 has enough dynamic range that the accumulated error stays small relative to the final sum magnitude.

---

## With `fp32_dest_acc_en=True`

When `fp32_dest_acc_en=True`, the destination register stores values in **float32** format (1 sign bit, 8 exponent bits, 23 mantissa bits).

The running sum is maintained at full float32 precision throughout all K tile iterations:

```
dst_fp32 += A_tile * B_tile    # dst maintained in float32; no truncation between tiles
```

The partial sum is only converted to the output dtype (typically bfloat16) once, at pack time, after all K tiles have been accumulated. This means only a single rounding event occurs on the final sum, regardless of K depth.

---

## When It Matters: K-Dimension Depth

The practical significance of `fp32_dest_acc_en` scales with the K dimension of the matmul. The key intuition:

- Each bfloat16 rounding of the running sum introduces an absolute error proportional to `ulp(|partial_sum|)` — roughly `|partial_sum| * 2^-7`.
- Over K_t accumulation steps, the worst-case error bound grows as `O(K_t * ulp)`.
- With float32 dest, the single final rounding has error `|final_sum| * 2^-23`, which is orders of magnitude smaller.

### For MoE Expert Down Projections

Consider the down projection in DeepSeek-V3 experts: `[M, d_ff] x [d_ff, d_model]` with `d_ff = 2048`, so `K = 2048`, `K_t = 64`.

At LoFi fidelity, the multiplier itself introduces mantissa error (covered in `math_fidelity_overview.md`). Combined with K_t=64 rounds of bfloat16 truncation in the destination register, the accumulated error on the partial sum is measurable.

At HiFi2 fidelity (used for DeepSeek down projections), the multiplier is more accurate, but 64 rounds of bfloat16 truncation would still degrade the running sum below what HiFi2 is trying to achieve. Hence `COMPUTE_KERNEL_CONFIG_HIFI2` sets `fp32_dest_acc_en=True`: it pairs higher-fidelity multiplication with a higher-precision accumulation register so neither step is the bottleneck.

### For MoE Expert Gate/Up Projections

Gate and up projections in DeepSeek-V3 have shape `[M, d_model] x [d_model, d_ff]` with `d_model = 7168`, `K = 7168`, `K_t = 224`.

At first glance, a deeper K-loop might seem to make `fp32_dest_acc_en` more important. However, `COMPUTE_KERNEL_CONFIG_LOFI` sets `fp32_dest_acc_en=False` for these projections. The reasoning:

- Gate and up projection outputs feed into SiLU and element-wise multiplication, not directly into the residual stream.
- SiLU's near-linear behavior for typical activation magnitudes means accumulated mantissa error does not amplify through the activation.
- The PCC of gate/up outputs at LoFi with bfloat16 dest remains above 0.999, which is within the practical tolerance for MoE routing.
- The register file cost of float32 dest is avoided, preserving more simultaneously live output tiles.

### Summary Table

> **Note:** K_t = K/32 (tile width = 32 elements); e.g., K=7168/32=224 tiles, K=2048/32=64 tiles.

| Projection | K depth | `fp32_dest_acc_en` | Rationale |
|---|---|---|---|
| Gate (w1), d_model=7168 | K_t=224 | `False` | Output feeds SiLU; error does not accumulate into residual stream |
| Up (w3), d_model=7168 | K_t=224 | `False` | Same reasoning as gate |
| Down (w2), d_ff=2048 | K_t=64 | `True` | Output enters residual stream directly; rounding across K_t tiles measurably degrades PCC |

> **Note:** Even though down has a shallower K-loop (K_t=64 vs K_t=224), it uses `fp32_dest_acc_en=True` because its output is more sensitivity-relevant. K depth alone does not determine the setting — sensitivity of the downstream use matters more.

---

## Interaction with `packer_l1_acc`

Both `fp32_dest_acc_en` and `packer_l1_acc` relate to accumulation precision, but they operate at different pipeline stages:

| Field | Pipeline Stage | What It Controls |
|---|---|---|
| `fp32_dest_acc_en` | FPU compute (within a K-block) | Precision of the running sum in the on-chip destination register |
| `packer_l1_acc` | Packer (between K-blocks / outer loop iterations) | Whether partial output tiles are accumulated in L1 or written to DRAM between outer loop steps |

The two fields are orthogonal and can be set independently:

```python
# Case 1: float32 dest + L1 packer accumulation (DeepSeek HIFI2 config)
# Best accuracy and best throughput for bandwidth-bound down projections
ttnn.WormholeComputeKernelConfig(
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
)

# Case 2: bfloat16 dest + L1 packer accumulation (DeepSeek LOFI config)
# Accepts bfloat16 dest rounding; still eliminates DRAM round-trips via packer L1 acc
ttnn.WormholeComputeKernelConfig(
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
)

# Case 3: float32 dest + DRAM packer (never recommended in practice)
# High precision within each K block but pays DRAM round-trip between blocks
ttnn.WormholeComputeKernelConfig(
    fp32_dest_acc_en=True,
    packer_l1_acc=False,   # suboptimal for bandwidth-bound workloads
)
```

When both are `True`, the FPU destination register holds the running sum in float32 precision throughout the K-loop. The packer then converts the accumulated float32 value to the output dtype (typically bfloat16) when writing to L1 or DRAM — the L1 accumulation buffer itself holds bfloat16, not float32. The `fp32_dest_acc_en` field affects only the on-chip FPU destination register (not L1), so there is no "twice the L1 space" overhead from enabling it. The L1 footprint implications of `packer_l1_acc` are analyzed in depth in Chapter 3 (`packer_l1_acc_constraints.md`).

---

## Cost: Reduced Simultaneously Live Accumulation Registers

The destination register file on each Tensix FPU has a fixed total size in bits. When `fp32_dest_acc_en=True`, each output value occupies 32 bits instead of 16 bits, so the register file can hold **half as many simultaneously live output tile values**.

In practice this means:

- The FPU can hold fewer output tiles in-flight at once in the destination register.
- The kernel may need to flush and reload destination tiles more frequently, reducing the opportunity to overlap compute with pack operations.
- For large `out_subblock_h * out_subblock_w` values (the number of output tile positions computed together per inner loop), enabling `fp32_dest_acc_en` may force a reduction in subblock size to fit within the halved register capacity.

For most standard MoE expert shapes, this subblock-size pressure is absorbed by the matmul program config optimizer and does not manifest as a visible throughput regression. The net effect is that the precision benefit of float32 dest outweighs the register pressure cost for the down projection workloads where it is used.

---

## Quick Decision Rule

```
Is this projection's output used directly in the residual stream (down projection)?
├── Yes → fp32_dest_acc_en=True  (paired with HiFi2 or higher fidelity)
└── No  → fp32_dest_acc_en=False (safe for gate/up projections feeding activations)
```

---

**Next:** [`math_fidelity_overview.md`](./math_fidelity_overview.md)
