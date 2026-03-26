# Chapter 6 — Math Fidelity, Compute Kernel Settings, and SDPA Configuration

## Overview

Every matrix multiplication and attention computation on Wormhole runs through a configurable compute kernel. The configuration controls accuracy, throughput, and L1 memory pressure via four parameters: `math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`, and `packer_l1_acc`. `TTNNBailingMoEAttention` currently uses the highest-accuracy combination, which is not necessarily the fastest. This chapter documents the current configuration, explains what each parameter does, and evaluates whether switching to the configuration used by `TTNNQwen3FullAttention` would be safe.

---

## Current Configuration

`TTNNBailingMoEAttention.move_weights_to_device_impl` (lines 2413–2440 of `attention.py`) initializes two `SDPAProgramConfig` objects and one shared compute kernel config:

### Prefill program config (lines 2422–2427)

```python
self.sdpa.program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
    q_chunk_size=256,
    k_chunk_size=256,
    exp_approx_mode=False,
)
```

`q_chunk_size=256` and `k_chunk_size=256` set the tile-block size used for the QK^T tiling in the prefill SDPA kernel. Standard TTNN tiles are 32×32, so 256 tokens span 8 tiles (32 tokens per tile). `exp_approx_mode=False` disables the polynomial approximation of the exponential in softmax, using the exact hardware exponential instead.

### Decode program config (lines 2428–2433)

```python
self.sdpa.decode_program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
    q_chunk_size=0,
    k_chunk_size=0,
    exp_approx_mode=False,
)
```

`q_chunk_size=0` and `k_chunk_size=0` instruct the kernel to pick chunk sizes automatically based on the input dimensions. In decode mode the Q has only 1 token (sequence length 1), so the prefill chunk size of 256 does not apply.

### Compute kernel config (lines 2434–2440)

```python
self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
    self.device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

This is a single config shared between the prefill and decode SDPA kernels.

---

## Parameter Reference

### `math_fidelity`

Controls the precision of the BFP8 product units inside the matrix engine. The three levels are:

| Level | Mantissa bits | Approximate relative speed |
|---|---|---|
| `HiFi4` | 4 | 1× (baseline) |
| `HiFi2` | 2 | ~1.5–2× vs HiFi4 |
| `LoFi` | 1 | ~3× vs HiFi4 |

"Mantissa bits" refers to the number of mantissa bits used when multiplying two BFP8 values in the inner product. HiFi4 uses 4 bits, producing results close to BFP8 precision. HiFi2 uses 2 bits, introducing more rounding error per multiply-accumulate but completing the operation faster. The fidelity does not change the accumulator format — that is controlled by `fp32_dest_acc_en`.

### `fp32_dest_acc_en`

When `True`, the destination (accumulator) register is widened to 32-bit floating point. This preserves accumulated sum precision even when individual products are computed at BFP8 fidelity. When `False`, the accumulator uses the smaller `dst` format (16-bit BFP), which frees 2× more destination tiles: the comment in `qwen_attention.py` (lines 338–340) states explicitly that `fp32_dest_acc_en=False` "increases dst_size from 4 to 8", allowing the inner loop to process larger tile batches per kernel dispatch.

Current Bailing MoE setting: `True` (FP32 accumulator, larger per-tile format — 4 dst tiles available).
Qwen3 setting: `False` (BFP16 accumulator, smaller per-tile format — 8 dst tiles available, more throughput).

### `packer_l1_acc`

When `True`, the packer stage accumulates partial results into L1 before writing the final output to DRAM or the output buffer. This reduces DRAM write traffic for accumulation-heavy ops like QK^T but increases peak L1 occupancy during the kernel. When `False`, the packer writes directly to the output buffer without L1 accumulation. The comment in `qwen_attention.py` (line 340) notes that `packer_l1_acc=False` "reduces L1 pressure".

Current Bailing MoE setting: `True` (L1 accumulation active).
Qwen3 setting: `False` (direct packer output).

### `math_approx_mode`

When `True`, certain transcendental functions (including reciprocal and square root) use hardware approximations rather than exact implementations. Both Bailing MoE and Qwen3 use `False` for this setting.

### `exp_approx_mode` (in `SDPAProgramConfig`)

Controls the softmax exponential implementation. When `False`, the exact hardware exponential is used. When `True`, a polynomial approximation replaces it. Both Bailing MoE and Qwen3 use `False`.

---

## Qwen3 Configuration for Comparison

`TTNNQwen3FullAttention.move_weights_to_device_impl` (lines 332–347 of `qwen_attention.py`) sets:

```python
self.sdpa.program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
    q_chunk_size=128,  # Reduced for head_dim=256 (matches DeepSeek V3)
    k_chunk_size=128,  # Reduced for head_dim=256 (matches DeepSeek V3)
    exp_approx_mode=False,
)
# Match DeepSeek V3 settings for head_dim=256 compatibility:
# fp32_dest_acc_en=False increases dst_size from 4 to 8
# packer_l1_acc=False reduces L1 pressure
self.sdpa.compute_kernel_config = ttnn.init_device_compute_kernel_config(
    self.device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)
```

The chunk sizes are 128 rather than 256 because Qwen3 uses `head_dim=256`. Bailing MoE uses `head_dim=128`, so larger chunk sizes are more appropriate. The key difference is `fp32_dest_acc_en=False` and `packer_l1_acc=False`.

---

## Summary: Current vs Qwen3 Config

| Parameter | Bailing MoE (current) | TTNNQwen3FullAttention |
|---|---|---|
| `math_fidelity` | HiFi4 | HiFi4 |
| `math_approx_mode` | False | False |
| `fp32_dest_acc_en` | **True** | **False** |
| `packer_l1_acc` | **True** | **False** |
| `exp_approx_mode` | False | False |
| `q_chunk_size` (prefill) | 256 | 128 |
| `k_chunk_size` (prefill) | 256 | 128 |
| `q_chunk_size` (decode) | 0 (auto) | Not set separately |
| `k_chunk_size` (decode) | 0 (auto) | Not set separately |

The two most impactful divergences are `fp32_dest_acc_en` and `packer_l1_acc`. Both implementations use HiFi4 math fidelity. The tradeoffs of switching are analyzed in `hifi4_vs_hifi2.md` and `accuracy_throughput_tradeoff.md`.

---

**Next:** [HiFi4 vs HiFi2 Analysis](hifi4_vs_hifi2.md)
