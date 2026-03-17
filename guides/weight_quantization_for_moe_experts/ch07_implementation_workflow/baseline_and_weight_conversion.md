# Baseline and Weight Conversion

## Purpose

Before quantizing any expert weights, establish a numerically correct bfloat16 baseline.
The baseline serves two roles: it confirms that the TTNN forward pass matches a CPU
reference (ruling out bugs independent of quantization), and it provides the reference
tensor against which all quantized outputs are compared. Steps 1 and 2 of the workflow
are covered here.

---

## Step 1 — Establish the bfloat16 Baseline

Run a single MoE expert forward pass on TTNN device using bfloat16 weights and bfloat16
activations. Compute PCC (Pearson Cross-Correlation) between the TTNN output and a
PyTorch CPU reference computed in float32. The target is PCC > 0.999.

```python
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

def run_bfloat16_baseline(x_torch, w1_bf16, w3_bf16, w2_bf16, device):
    """Run a single expert forward pass in bfloat16 and validate against CPU reference.

    Args:
        x_torch: Input activations, torch.bfloat16, shape [num_tokens, d_model].
        w1_bf16: Gate weight, torch.bfloat16, shape [d_ff, d_model].
        w3_bf16: Up weight, torch.bfloat16, shape [d_ff, d_model].
        w2_bf16: Down weight, torch.bfloat16, shape [d_model, d_ff].
        device: TTNN device handle.

    Returns:
        (out_tt_torch, pcc_val): TTNN output as torch.Tensor; PCC against CPU reference.
    """
    # -- CPU reference in float32 --
    x_f32   = x_torch.float()
    gate_ref = torch.nn.functional.silu(x_f32 @ w1_bf16.float().T)
    up_ref   = x_f32 @ w3_bf16.float().T
    inter_ref = gate_ref * up_ref
    out_ref   = inter_ref @ w2_bf16.float().T  # shape [num_tokens, d_model]

    # -- TTNN bfloat16 forward pass --
    x_tt  = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
    w1_tt = ttnn.as_tensor(w1_bf16, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w3_tt = ttnn.as_tensor(w3_bf16, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w2_tt = ttnn.as_tensor(w2_bf16, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

    gate_pre_tt = ttnn.linear(x_tt, w1_tt)
    gate_out_tt = ttnn.silu(gate_pre_tt)
    up_out_tt   = ttnn.linear(x_tt, w3_tt)
    inter_tt    = ttnn.mul(gate_out_tt, up_out_tt)
    out_tt      = ttnn.linear(inter_tt, w2_tt)

    out_tt_torch = ttnn.to_torch(out_tt).float()
    pcc_val, _ = comp_pcc(out_ref, out_tt_torch, pcc=0.999)
    print(f"bfloat16 baseline PCC = {pcc_val:.4f} (target > 0.999)")
    assert pcc_val > 0.999, f"Baseline PCC {pcc_val:.4f} failed. Debug forward pass before quantizing."
    return out_tt_torch, pcc_val
```

> **Warning:** Do not proceed to weight conversion if the bfloat16 baseline PCC is below
> 0.999. A failing baseline indicates a layout error, incorrect weight shape, or a missing
> transpose — not a quantization issue. Fix the baseline first.

### Why bfloat16 Baseline PCC Should Exceed 0.999

TTNN executes matmuls on Tensix cores using bfloat16 intermediate arithmetic. The CPU
reference uses float32. The small numerical difference between bfloat16 and float32
accumulation produces a PCC loss of roughly 0.0001–0.0005. A well-formed bfloat16 forward
pass with correct weight shapes and layouts routinely achieves PCC > 0.9995 against a
float32 CPU reference. If PCC falls below 0.999, the most common causes are:

1. Incorrect weight transpose — TTNN `linear` expects `[out_features, in_features]`.
2. Non-tile-aligned shape passed without padding (see tile alignment section below).
3. Wrong memory config causing a layout mismatch on read.

---

## Step 2 — Convert Weights to Target Dtypes

With a passing baseline, convert all expert weights to their target dtypes. Gate (`w1`)
and up (`w3`) projections convert to `bfloat4_b`; down (`w2`) projections convert to
`bfloat8_b`. All converted weights must use `TILE_LAYOUT`.

### Weight Conversion Script

```python
import os
import ttnn
import torch

def convert_expert_weights(w1_bf16, w3_bf16, w2_bf16, device):
    """Convert one expert's weights to mixed precision.

    Args:
        w1_bf16: Gate weight, torch.bfloat16, shape [d_ff, d_model].
        w3_bf16: Up weight, torch.bfloat16, shape [d_ff, d_model].
        w2_bf16: Down weight, torch.bfloat16, shape [d_model, d_ff].
        device: TTNN device handle.

    Returns:
        (w1_tt, w3_tt, w2_tt): TTNN tensors in DRAM at target dtypes.
    """
    w1_tt = ttnn.as_tensor(
        w1_bf16,
        dtype=ttnn.bfloat4_b,        # gate: 4-bit block float, 4× size reduction
        layout=ttnn.TILE_LAYOUT,      # required for all quantized dtypes
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w3_tt = ttnn.as_tensor(
        w3_bf16,
        dtype=ttnn.bfloat4_b,        # up: 4-bit block float
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w2_tt = ttnn.as_tensor(
        w2_bf16,
        dtype=ttnn.bfloat8_b,        # down: 8-bit block float, 2× size reduction
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return w1_tt, w3_tt, w2_tt
```

> **Tip:** For 128 experts across all MoE layers, conversion is a one-time startup cost.
> Cache converted weights to disk using `ttnn.dump_tensor` / `ttnn.load_tensor` to avoid
> re-conversion on every restart (see Chapter 5, `qwen_adaptation_guide.md` for the full
> caching pattern).

### Conversion Correctness Check

After conversion, dequantize the TTNN tensor back to float32 and compute PCC against the
original bfloat16 weight. This isolates the quantization error introduced by the dtype
conversion itself, independent of any matmul compute error.

```python
def check_weight_conversion_pcc(weight_bf16, weight_tt, expected_pcc, label="weight"):
    """Verify that a converted TTNN weight round-trips with acceptable PCC.

    Args:
        weight_bf16: Original torch.bfloat16 tensor.
        weight_tt: Converted TTNN tensor (bfloat4_b or bfloat8_b).
        expected_pcc: Minimum acceptable PCC (0.99 for bfloat8_b, 0.97 for bfloat4_b).
        label: Human-readable label for log output.
    """
    # Dequantize by converting back to bfloat16 via torch
    weight_dequant = ttnn.to_torch(
        ttnn.to_dtype(weight_tt, ttnn.bfloat16)
    ).float()
    weight_ref = weight_bf16.float()

    pcc_val, _ = comp_pcc(weight_ref, weight_dequant, pcc=expected_pcc)
    print(f"{label} conversion PCC = {pcc_val:.4f} (target >= {expected_pcc})")
    assert pcc_val >= expected_pcc, (
        f"{label} conversion PCC {pcc_val:.4f} < {expected_pcc}. "
        "Check for shape padding issues or dtype mismatch."
    )
    return pcc_val
```

Expected conversion PCC values:

| Dtype | Conversion PCC target | Reason for lower bound |
|---|---|---|
| `bfloat8_b` | > 0.99 | 8-bit mantissa; minimal outlier distortion |
| `bfloat4_b` | > 0.97 | 4-bit mantissa; shared exponent compresses non-outlier elements |

> **Warning:** A conversion PCC below 0.97 for `bfloat4_b` indicates large weight
> outliers in the tile. A shared exponent across all 1024 elements in a 32×32 tile is
> forced to accommodate the outlier, compressing representable range for the remaining
> elements. Consider applying per-channel weight clipping before conversion, or fall back
> to `bfloat8_b` for that projection.

---

## Handling Non-Tile-Aligned Weight Shapes

All quantized dtypes (`bfloat8_b`, `bfloat4_b`) require `TILE_LAYOUT`, which enforces
a 32×32 tile boundary. Weight shapes that are not multiples of 32 in both dimensions must
be padded before conversion.

For Qwen 235B-A22B, both `d_model=7168` and `d_ff=2048` are exact multiples of 32
(`7168 = 224 × 32`, `2048 = 64 × 32`), so no padding is required. For other models,
apply the following padding utility:

```python
def pad_to_tile(weight_bf16):
    """Pad a 2D weight tensor to the nearest 32×32 tile boundary.

    Args:
        weight_bf16: torch.Tensor of shape [rows, cols].

    Returns:
        (padded, original_shape): Padded tensor; original shape for unpadding after output.
    """
    rows, cols = weight_bf16.shape
    pad_rows = (32 - rows % 32) % 32
    pad_cols = (32 - cols % 32) % 32
    if pad_rows == 0 and pad_cols == 0:
        return weight_bf16, (rows, cols)  # already aligned

    padded = torch.nn.functional.pad(weight_bf16, (0, pad_cols, 0, pad_rows))
    return padded, (rows, cols)

def unpad_output(output_torch, original_shape):
    """Remove padding from a 2D output tensor.

    Args:
        output_torch: torch.Tensor, possibly padded.
        original_shape: (rows, cols) tuple from pad_to_tile.

    Returns:
        Unpadded tensor sliced to original_shape.
    """
    rows, cols = original_shape
    return output_torch[..., :rows, :cols]
```

> **Tip:** Tile alignment must be verified for the activation dimension as well. If the
> number of tokens routed to an expert is not a multiple of 32, pad the token dimension
> before the matmul and slice the output afterward. This is separate from weight padding.

---

## TTNN Weight Serialization

Weight conversion with `ttnn.as_tensor` for a full 128-expert checkpoint takes tens of
seconds. Serialize converted weights to disk after the first conversion:

```python
CONVERTED_WEIGHT_CACHE_DIR = "/path/to/converted_weights_cache"

def save_converted_weight(weight_tt, param_name, dtype_str, cache_dir=CONVERTED_WEIGHT_CACHE_DIR):
    """Serialize a converted TTNN weight to disk.

    Args:
        weight_tt: TTNN tensor on device.
        param_name: Unique string key (parameter name from state dict).
        dtype_str: 'bfloat4_b' or 'bfloat8_b', appended to filename for clarity.
        cache_dir: Target directory.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = param_name.replace(".", "_").replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{safe_name}_{dtype_str}.bin")
    ttnn.dump_tensor(cache_path, ttnn.from_device(weight_tt))
    return cache_path

def load_converted_weight(param_name, dtype_str, device, cache_dir=CONVERTED_WEIGHT_CACHE_DIR):
    """Load a serialized TTNN weight from disk onto device.

    Args:
        param_name: Same unique string key used during save.
        dtype_str: 'bfloat4_b' or 'bfloat8_b'.
        device: TTNN device handle.
        cache_dir: Source directory.

    Returns:
        weight_tt: TTNN tensor on device in DRAM.
    """
    safe_name = param_name.replace(".", "_").replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{safe_name}_{dtype_str}.bin")
    weight_tt = ttnn.load_tensor(cache_path)
    return ttnn.to_device(weight_tt, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

For the caching and version-pinning pattern, see `../ch05_per_projection_strategy/qwen_adaptation_guide.md`.

---

## Summary

| Sub-step | Action | Pass criterion |
|---|---|---|
| 1a | Run bfloat16 forward pass on device | PCC > 0.999 vs. float32 CPU reference |
| 1b | If PCC < 0.999, debug layout / shape before proceeding | N/A |
| 2a | Convert gate/up to `bfloat4_b`, `TILE_LAYOUT`, DRAM | Conversion PCC > 0.97 |
| 2b | Convert down to `bfloat8_b`, `TILE_LAYOUT`, DRAM | Conversion PCC > 0.99 |
| 2c | Pad non-tile-aligned shapes before conversion | Both dims multiples of 32 |
| 2d | Cache converted weights to disk | Reload produces same TTNN tensor |

---

**Next:** [`per_layer_pcc_validation.md`](./per_layer_pcc_validation.md)
