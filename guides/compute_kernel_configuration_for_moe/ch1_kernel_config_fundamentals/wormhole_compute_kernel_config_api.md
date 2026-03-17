# WormholeComputeKernelConfig API

## Overview

`ttnn.WormholeComputeKernelConfig` is a Python dataclass-style object that encodes compute behavior for a single TTNN op execution on Wormhole B0 hardware. It is passed to compute-intensive ops (primarily `ttnn.matmul`) via the `compute_kernel_config` keyword argument. When omitted, TTNN selects hardware-conservative defaults that are correct but not performance-optimal.

This file covers:

- How to construct the config object
- The four primary fields and their semantics
- Two secondary fields (brief)
- What the omit-case actually gives you
- The two canonical production configs from DeepSeek-V3

---

## Construction and Usage

The config is constructed by calling the class with keyword arguments. All fields have defaults so you can specify only the fields you care about.

```python
import ttnn

# Minimal construction — all fields at their defaults
cfg = ttnn.WormholeComputeKernelConfig()

# Full explicit construction — the pattern used in production
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

Pass the config to `ttnn.matmul` via the `compute_kernel_config` keyword:

```python
import ttnn
import torch

# Assume device is already opened and tensors are on device
# input_tensor: [batch, seq, d_model] on device, bfloat16
# weight: [d_model, d_ff] on device, bfloat16

cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

output = ttnn.matmul(
    input_tensor,
    weight,
    compute_kernel_config=cfg,    # <-- kernel config goes here
    dtype=ttnn.bfloat16,
)
```

The `compute_kernel_config` argument is orthogonal to `program_config` (which controls tile layout and core grid assignment). Both can be specified on the same `ttnn.matmul` call:

```python
output = ttnn.matmul(
    input_tensor,
    weight,
    program_config=my_program_config,       # controls core mapping and tiling
    compute_kernel_config=cfg,              # controls FPU precision and packer behavior
    dtype=ttnn.bfloat16,
)
```

---

## Primary Fields

### `math_fidelity`

**Type:** `ttnn.MathFidelity` enum
**Default:** `ttnn.MathFidelity.LoFi`

Controls how many mantissa bits from each bfloat16 operand are presented to the multiplier hardware during each dot-product accumulation step. Higher fidelity uses more mantissa bits, produces more accurate products, but requires more cycles per tile.

```python
# Available values
ttnn.MathFidelity.LoFi   # Fastest; fewer mantissa bits per multiply
ttnn.MathFidelity.HiFi2  # Medium-high accuracy; standard production choice for sensitive projections
ttnn.MathFidelity.HiFi3  # Intermediate; rarely used directly
ttnn.MathFidelity.HiFi4  # Highest accuracy; closest to PyTorch float32 reference
```

For a detailed breakdown of each level's throughput multiplier and PCC characteristics, see Chapter 2.

---

### `math_approx_mode`

**Type:** `bool`
**Default:** `False`

Enables hardware-approximated implementations of SFPU (Special Function Processing Unit) transcendental functions: `exp`, `reciprocal`, `sqrt`, `sigmoid`, `gelu`, `silu`. These approximations use piecewise polynomial lookups instead of iterative refinement, trading ~0.1–0.3% relative error per evaluation for lower cycle count.

> **Warning:** `math_approx_mode` has no effect on the FPU matrix multiply path. Setting it to `True` on a pure matmul op (no fused activation) does not change outputs and does not improve throughput. Its impact is only visible when the op includes a fused transcendental activation.

For pure expert matmuls (no fused activation), this field is effectively inert. See Chapter 4 for detailed coverage of when this matters.

---

### `fp32_dest_acc_en`

**Type:** `bool`
**Default:** `False`

Controls the precision of the FPU destination register — the register where partial sums land between K-loop iterations.

- `False` (default): partial sums are stored in bfloat16 in the destination register. Each K-loop accumulation step rounds the running sum to 7 mantissa bits before the next tile is added.
- `True`: partial sums are stored in float32. The full 23-bit mantissa of the running sum is preserved throughout the K-loop, preventing rounding error from compounding across K tiles.

The cost is that float32 destination registers occupy twice the register file space, slightly reducing the number of simultaneously live output tiles per core.

For a detailed analysis of when this precision difference is measurable for MoE expert matmuls, see `fp32_dest_acc_en.md`.

---

### `packer_l1_acc`

**Type:** `bool`
**Default:** `False`

Controls where the packer stage writes partial output sums between K-loop iterations.

- `False` (default): after each K-loop iteration, the packer writes the partial sum tile to DRAM. The next iteration reads it back from DRAM to accumulate. This is a DRAM round-trip for every K-loop step except the last.
- `True`: the packer accumulates partial sums into an L1 buffer. DRAM is written only once — after the full K-loop completes. This eliminates `(K_t / in0_block_w - 1)` DRAM reads of the output buffer per core, where `K_t = K/32` is the number of K tiles.

> **Tip:** For bandwidth-bound MoE decode matmuls (small M, large K), `packer_l1_acc=True` is the single most impactful setting. It directly reduces DRAM traffic without any accuracy trade-off.

The L1 budget implications of enabling this flag (especially when combined with `fp32_dest_acc_en=True`) are covered in Chapter 3.

---

## Secondary Fields

These two fields exist in `WormholeComputeKernelConfig` but are not the focus of this guide. Use them only if you have hardware-specific guidance.

### `dst_full_sync_en`

**Type:** `bool`
**Default:** `False`

Controls synchronization behavior between the FPU and the packer pipeline. When `True`, the FPU stalls until the packer has fully consumed each output tile before producing the next. In the default (`False`) state, the FPU and packer operate in a pipelined fashion, allowing some overlap. Enabling `dst_full_sync_en` can help with rare correctness issues on specific op patterns but comes with a throughput cost. Leave at `False` unless instructed otherwise.

### `throttle_level`

**Type:** `int` (typically 0–3)
**Default:** `0`

Controls how aggressively the Tensix core throttles its clock speed (and thus power consumption) during op execution. Higher values reduce clock speed to stay within a power envelope. For inference performance work, leave at `0`.

---

## Default Behavior When `compute_kernel_config` Is Omitted

When `compute_kernel_config` is not passed to `ttnn.matmul`, TTNN selects internal defaults at the device level. As of current tt-metal releases, the effective behavior is approximately:

```python
# Approximate equivalent of what TTNN uses when compute_kernel_config is omitted
# Do NOT use this as a substitute for an explicit config — TTNN defaults may change across releases
ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,   # <-- key: DRAM round-trips are NOT eliminated
)
```

The critical difference from the production configs is `packer_l1_acc=False`. For decode-mode MoE expert matmuls where the output is small (M=1 to 32) and K is large (2048 to 7168), this means every K-loop iteration pays a DRAM round-trip for the partial output tiles — a significant bandwidth overhead that the DeepSeek configs eliminate by explicitly setting `packer_l1_acc=True`.

> **Warning:** Do not rely on TTNN default behavior for production performance work. The defaults are chosen for correctness portability, not performance. Always pass an explicit `compute_kernel_config` to matmuls in performance-critical code paths.

---

## Canonical Production Configs

These two configs are defined in `models/demos/deepseek_v3/utils/config_helpers.py` and used for all MoE expert matmuls in DeepSeek-V3.

### `COMPUTE_KERNEL_CONFIG_LOFI`

Used for gate projections (w1) and up projections (w3) in each MoE expert.

```python
import ttnn

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,   # Fastest fidelity; gate/up outputs feed SiLU, tolerant of mantissa error
    math_approx_mode=False,                  # No transcendental ops in the matmul; this flag is inert
    fp32_dest_acc_en=False,                  # bfloat16 dest register; saves register file space
    packer_l1_acc=True,                      # Eliminate DRAM round-trips during K-loop accumulation
)
```

Rationale: gate and up projection outputs feed into SiLU and element-wise multiplication before the down projection. These operations are tolerant of small mantissa rounding errors introduced by LoFi. The PCC of gate/up outputs at LoFi vs HiFi4 is above 0.999, well above the practical threshold for MoE routing and gating.

### `COMPUTE_KERNEL_CONFIG_HIFI2`

Used for down projections (w2) in each MoE expert.

```python
import ttnn

COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,  # Higher fidelity; down projection accumulates into residual stream
    math_approx_mode=True,                   # Inert for pure matmuls with no fused transcendental activation; carried forward from source model config for traceability only
    fp32_dest_acc_en=False,                  # bfloat16 dest register; sufficient for bfloat8_b weights; fp32 would double L1 accumulation buffer
    packer_l1_acc=True,                      # Eliminate DRAM round-trips; applies regardless of fidelity level
)
```

Rationale: the down projection reduces from d_ff back to d_model, accumulating across all K tiles of the intermediate dimension. For DeepSeek-V3, d_ff = 2048 per expert; accumulated rounding error at LoFi fidelity degrades PCC of the residual stream contributions measurably. HiFi2 with `fp32_dest_acc_en=False` restores PCC above the threshold without the full cycle cost of HiFi4. BF16 dest accumulation is sufficient because input weights are bfloat8_b; enabling fp32 would double the L1 accumulation buffer and increase overflow risk.

---

## Putting It Together: Two-Config Pattern

The pattern of defining LOFI for gate/up and HIFI2 for down is directly transferable to other SwiGLU/SiLU MoE models (e.g., Qwen MoE):

```python
# In the expert forward pass:
gate_output = ttnn.matmul(hidden_states, w1, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
up_output   = ttnn.matmul(hidden_states, w3, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
# ... SiLU and element-wise multiply ...
down_output = ttnn.matmul(activated,     w2, compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2)
```

---

**Next:** [`fp32_dest_acc_en.md`](./fp32_dest_acc_en.md)
