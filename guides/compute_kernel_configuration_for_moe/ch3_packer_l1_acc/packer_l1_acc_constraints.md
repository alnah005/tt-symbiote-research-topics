# `packer_l1_acc` Constraints and Failure Modes

## L1 Footprint of the Accumulation Buffer

When `packer_l1_acc=True`, the packer requires a dedicated accumulation buffer in the per-core L1 SRAM (Static Random-Access Memory) alongside the input tile buffers, output tile buffers, and program code. This buffer holds the running partial sums for all output tiles assigned to the core.

The accumulation buffer size is determined by two factors:

1. **`per_core_N`** — the number of N_t output tile columns assigned to each core; more columns means a larger accumulation footprint.
2. **`fp32_dest_acc_en`** — controls whether the accumulation buffer holds bfloat16 (2 bytes/element) or fp32 (4 bytes/element) values.

For `per_core_N` output tile columns and M_t = 1 (decode mode), the total accumulation buffer size is:

```
accum_buffer_bytes = per_core_N × tile_bytes
                   = per_core_N × 2048   (bfloat16 accumulation)
                   = per_core_N × 4096   (fp32 accumulation)
```

---

## L1 Overflow Risk

Each Tenstorrent Wormhole core has approximately **1.5 MB of L1 SRAM**. This space must be shared among:

- Input tile double-buffers (in0 and in1 circular buffers)
- Output tile buffers
- The `packer_l1_acc` accumulation buffer (when enabled)
- Program binary and kernel stack overhead

When `per_core_N` is large, the accumulation buffer can push total L1 usage past 1.5 MB. This is more likely when:

- `per_core_N` is large (e.g., N_t / num_cores is a large tile count)
- `fp32_dest_acc_en=True` is also set (doubles the accumulation buffer size)
- Input circular buffers are deep (large `in0_block_w` or `in1_block_w`)

> **Warning:** Enabling both `packer_l1_acc=True` and `fp32_dest_acc_en=True` makes L1 overflow significantly more likely for wide output tile assignments. Always perform a manual L1 budget check (see below) when combining these flags.

---

## How to Detect L1 Overflow

TTNN (TT-NN) validates L1 allocations at **op dispatch time**, not at `ttnn.linear` call time. This means:

- The `ttnn.linear(...)` call may return without error.
- The allocation error is raised when the operation is dispatched to the device — typically when the program queue is flushed or the result tensor is synchronized.

A typical error message looks like:

```
RuntimeError: Out of L1 memory. Requested allocation of XXXXX bytes exceeds
available L1 capacity on core (x, y). Reduce per_core_N or out_subblock_w.
```

> **Warning:** Because the error is deferred to dispatch time, a failing `packer_l1_acc=True` configuration may not surface immediately in test harnesses that do not synchronize outputs promptly. Always use explicit `ttnn.synchronize_device()` calls when validating new kernel configurations.

**Resolution options** when L1 overflow occurs:

| Fix | Effect | Trade-off |
|---|---|---|
| Reduce `per_core_N` | Smaller accumulation buffer | May require more cores or fewer output tiles per core |
| Reduce `out_subblock_w` | Reduces output tile sub-blocking | Minor impact on FPU efficiency |
| Set `fp32_dest_acc_en=False` | Halves accumulation buffer (bfloat16) | Lower accumulation precision |
| Disable `packer_l1_acc` | Eliminates accumulation buffer | Loses throughput benefit; last resort |

---

## Interaction with `fp32_dest_acc_en`

The `fp32_dest_acc_en` flag controls whether the FPU destination register accumulates in fp32 (32-bit floating point) or bfloat16 (16-bit bfloat). When `packer_l1_acc=True` is also enabled, this precision choice flows directly into the L1 accumulation buffer. The LoFi and HiFi2 configurations used in DeepSeek-V3 are shown in full in the Safe Default Configurations section below.

**Summary of the interaction:**

| `packer_l1_acc` | `fp32_dest_acc_en` | Accum buffer dtype | Accum buffer size per tile |
|---|---|---|---|
| `False` | Either | N/A (no accum buffer) | 0 |
| `True` | `False` | bfloat16 | 2,048 bytes |
| `True` | `True` | fp32 | 4,096 bytes |

> **Warning:** When both `packer_l1_acc=True` and `fp32_dest_acc_en=True` are enabled, the accumulation buffer requires **2× the L1 space** compared to bfloat16 accumulation. For HiFi2 configurations on down projections (which tend to have larger N dimensions), always verify the L1 budget explicitly before deployment.

---

## Safe Default Configurations

Based on the DeepSeek-V3 production configuration and the L1 budget analysis above, the following defaults are recommended for MoE workloads:

### LoFi Configuration (Gate and Up Projections)

```python
# Recommended for MoE gate and up projections
# fp32_dest_acc_en=False → bfloat16 accumulation buffer
# Smaller L1 footprint; suitable for larger per_core_N values
COMPUTE_KERNEL_CONFIG_LOFI = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.LoFi,
    math_approx_mode=True,
    fp32_dest_acc_en=False,   # bfloat16 accumulation (2 bytes/element)
    packer_l1_acc=True,       # enabled; L1 overflow risk is lower here
)
```

### HiFi2 Configuration (Down Projections)

```python
# Recommended for MoE down projections requiring higher precision
# fp32_dest_acc_en=False → bfloat16 accumulation buffer; sufficient for bfloat8_b weights
# fp32 would double the L1 accumulation buffer and increase overflow risk
COMPUTE_KERNEL_CONFIG_HIFI2 = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,   # bfloat16 accumulation (2 bytes/element)
    packer_l1_acc=True,       # enabled; verify L1 budget explicitly
)
```

---

## L1 Budget Check Procedure

Use the following formula to estimate whether `packer_l1_acc=True` will fit within the 1.5 MB L1 budget:

```python
# L1 budget estimate (simplified, per core, decode-mode M_t=1)
TILE_BYTES_BF16 = 32 * 32 * 2   # 2,048 bytes
TILE_BYTES_FP32 = 32 * 32 * 4   # 4,096 bytes
L1_CAPACITY     = 1.5 * 1024 * 1024  # 1,572,864 bytes

def estimate_l1_usage(per_core_N, in0_block_w, fp32_dest_acc_en):
    # Accumulation buffer
    accum_tile_bytes = TILE_BYTES_FP32 if fp32_dest_acc_en else TILE_BYTES_BF16
    accum_buffer = per_core_N * accum_tile_bytes

    # Input circular buffers (approximate; actual depends on cb depth)
    in0_buffer = in0_block_w * TILE_BYTES_BF16 * 2   # double-buffered
    in1_buffer = per_core_N * in0_block_w * TILE_BYTES_BF16

    # Output buffer
    out_buffer = per_core_N * accum_tile_bytes

    # Program overhead (approximate)
    program_overhead = 50 * 1024   # ~50 KB conservative estimate

    total = accum_buffer + in0_buffer + in1_buffer + out_buffer + program_overhead
    return total

# Example: per_core_N=8, in0_block_w=4, HiFi2 (fp32 accum)
usage = estimate_l1_usage(per_core_N=8, in0_block_w=4, fp32_dest_acc_en=True)
fits  = usage <= L1_CAPACITY
# Check: usage ≤ 1,572,864 bytes before enabling packer_l1_acc=True + fp32_dest_acc_en=True
```

> **Tip:** This estimate is a lower bound on actual L1 usage. Actual usage depends on additional circular buffers, kernel stack frames, and TTNN internal allocation overhead. If the estimate is within 20% of the 1.5 MB limit, treat it as potentially at risk and test empirically.

---

## Next Steps

This completes Chapter 3. Proceed to **Chapter 4** for coverage of `math_approx_mode` — the second single-bit flag in `WormholeComputeKernelConfig` — including which transcendental operations it affects, the accuracy trade-offs for MoE softmax and gating, and when to enable or disable it for production deployments.
