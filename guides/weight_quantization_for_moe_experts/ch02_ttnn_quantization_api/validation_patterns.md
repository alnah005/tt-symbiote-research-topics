# Validation Patterns

Quantized weight conversion introduces irreversible information loss. Validation confirms that the loss is within acceptable bounds before a quantized model is used in production or benchmarking. This page describes the standard patterns for checking conversion accuracy: weight-level Pearson Correlation Coefficient (PCC) checks, forward-pass PCC checks, and the test scaffolding that ties them together.

## PCC as the Primary Accuracy Metric

PCC (Pearson Correlation Coefficient) measures the linear correlation between two tensors — a reference (typically bfloat16) and the quantized output. A PCC of 1.0 indicates perfect linear correlation; 0.0 indicates no correlation.

PCC is the standard TTNN convention for accuracy assertions in model validation because it is invariant to global scale shifts, which can occur when quantization introduces a systematic bias. This makes PCC more informative than mean absolute error for detecting structural accuracy degradation.

> **Warning:** PCC measures correlation, not absolute error. A high PCC output (e.g., 0.98) can still have a systematic scale shift or DC offset relative to the reference. Always supplement PCC checks with `torch.allclose` using appropriate tolerances (`atol`, `rtol`) for layers where absolute magnitude matters — for example, normalization inputs or logit outputs.

## Expected PCC Thresholds

### Weight-Level PCC

Comparing the dequantized weight tensor against the original bfloat16 weight tensor:

| Weight dtype | Expected weight PCC |
|---|---|
| bfloat8_b | > 0.99 |
| bfloat4_b | approximately 0.97 – 0.98 |

A weight PCC below these thresholds suggests a problem with the conversion (wrong shape, misaligned tile boundaries, or an unexpected dtype coercion).

### Forward-Pass PCC

Comparing a single-layer forward pass output against a `torch.bfloat16` CPU reference:

| Configuration | Expected forward-pass PCC |
|---|---|
| Single MLP with bfloat8_b weights | approximately 0.975 |
| Full MoE layer, mixed bfloat4_b gate/up + bfloat8_b down | approximately 0.97 |

These values are observed in practice on Wormhole hardware with HiFi2 config for bfloat8_b projections. Actual values depend on the model's hidden dimension, the number of active experts, and the input distribution.

## The `comp_pcc` Helper

TTNN includes a `comp_pcc` utility function for asserting PCC within tests. The pattern is:

```python
from ttnn.model_demos.utils import comp_pcc  # or equivalent import path for your TTNN version

pcc_passed, pcc_value = comp_pcc(
    torch_reference_output,  # torch.Tensor on CPU
    ttnn_output_torch,       # ttnn output moved to CPU and converted to torch
    pcc=0.97,                # minimum acceptable PCC
)
assert pcc_passed, f"PCC {pcc_value:.4f} below threshold 0.97"
```

The TTNN output must be moved to CPU and converted to a torch tensor before comparison. The function returns a boolean pass/fail and the actual PCC value.

```python
import torch
import ttnn
from ttnn.model_demos.utils import comp_pcc

def move_to_torch(ttnn_tensor: ttnn.Tensor) -> torch.Tensor:
    """Move a TTNN device tensor to a CPU torch.Tensor in bfloat16."""
    return ttnn.to_torch(ttnn.from_device(ttnn_tensor)).to(torch.bfloat16)
```

## Weight-Level Validation

Check conversion accuracy at load time, before running any forward passes. This catches shape and alignment problems immediately.

```python
import torch
import ttnn
from ttnn.model_demos.utils import comp_pcc

def validate_weight_conversion(
    original_weight: torch.Tensor,    # bfloat16 CPU tensor, shape [d_ff, d_model]
    converted_weight_tt: ttnn.Tensor, # device tensor, bfloat8_b or bfloat4_b
    dtype: ttnn.DataType,
    layer_name: str,
) -> None:
    """
    Assert that the converted weight has acceptable PCC against the original.
    Raises AssertionError if PCC is below threshold.
    """
    # Set threshold based on dtype
    if dtype == ttnn.bfloat8_b:
        threshold = 0.99
    elif dtype == ttnn.bfloat4_b:
        threshold = 0.97
    else:
        raise ValueError(f"Unexpected dtype {dtype}")

    # Dequantize by reading back through TTNN (returns bfloat16)
    dequant_weight = ttnn.to_torch(ttnn.from_device(converted_weight_tt)).to(torch.bfloat16)

    pcc_passed, pcc_value = comp_pcc(original_weight, dequant_weight, pcc=threshold)
    assert pcc_passed, (
        f"Weight PCC for {layer_name}: {pcc_value:.4f} below threshold {threshold} "
        f"(dtype={dtype})"
    )
    print(f"[OK] {layer_name}: weight PCC = {pcc_value:.4f} (threshold {threshold})")
```

## Forward-Pass Validation

The full validation test loads real checkpoint weights, converts them, runs a forward pass on device, and compares against a CPU bfloat16 reference.

```python
import torch
import ttnn
from ttnn.model_demos.utils import comp_pcc

def run_forward_pass_validation(
    checkpoint_path: str,
    device: ttnn.Device,
    pcc_threshold: float = 0.97,
) -> None:
    """
    End-to-end validation: load weights, convert, run forward, compare to bfloat16 CPU reference.
    """
    # Step 1: Load checkpoint and prepare reference weights (bfloat16, CPU)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    w1_ref = state_dict["expert.w1.weight"].to(torch.bfloat16)  # shape [d_model, d_ff]
    w2_ref = state_dict["expert.w2.weight"].to(torch.bfloat16)  # shape [d_ff, d_model]

    # Step 2: Convert to target dtypes on device (load-time, one-time cost)
    w1_tt = ttnn.as_tensor(
        w1_ref.t().contiguous(),    # transpose to [d_ff, d_model]
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w2_tt = ttnn.as_tensor(
        w2_ref.t().contiguous(),    # transpose to [d_model, d_ff]
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Step 3: Prepare a random activation input
    batch, seq_len, d_model = 1, 128, w1_ref.shape[0]
    x_torch = torch.randn(batch, seq_len, d_model, dtype=torch.bfloat16)

    # Step 4: CPU bfloat16 reference forward pass
    hidden_ref = torch.nn.functional.linear(x_torch, w1_ref)  # [batch, seq_len, d_ff]
    out_ref    = torch.nn.functional.linear(hidden_ref, w2_ref)  # [batch, seq_len, d_model]

    # Step 5: TTNN forward pass on device
    from ttnn import WormholeComputeKernelConfig, MathFidelity
    hifi2_config = WormholeComputeKernelConfig(
        math_fidelity=MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

    hidden_tt = ttnn.linear(x_tt, w1_tt, compute_kernel_config=hifi2_config)
    out_tt    = ttnn.linear(hidden_tt, w2_tt, compute_kernel_config=hifi2_config)

    # Step 6: Move TTNN output to CPU for comparison
    out_tt_torch = ttnn.to_torch(ttnn.from_device(out_tt)).to(torch.bfloat16)

    # Step 7: Assert PCC
    pcc_passed, pcc_value = comp_pcc(out_ref, out_tt_torch, pcc=pcc_threshold)
    assert pcc_passed, (
        f"Forward pass PCC {pcc_value:.4f} below threshold {pcc_threshold}"
    )
    print(f"[OK] Forward pass PCC = {pcc_value:.4f} (threshold {pcc_threshold})")

    # Step 8: Supplemental allclose check for scale integrity
    # atol and rtol chosen based on bfloat16 dynamic range; adjust for your model's output scale
    allclose = torch.allclose(out_ref, out_tt_torch, atol=1e-1, rtol=1e-1)
    if not allclose:
        max_diff = (out_ref - out_tt_torch).abs().max().item()
        print(f"[WARN] allclose failed: max absolute difference = {max_diff:.4f}. "
              f"PCC {pcc_value:.4f} passed but scale shift may be present.")
```

## Interpretation and Debugging

If PCC falls below threshold, use these steps to narrow down the cause:

1. **Check weight-level PCC first.** If weight PCC is already low, the issue is in the conversion step (shape misalignment, wrong dtype, unexpected quantization behavior). Fix the conversion before investigating the forward pass.

2. **Check activation dtype.** If the activation entering the matmul is float32 rather than bfloat16, the reference and TTNN forward passes are computing different things.

3. **Isolate the layer.** Run the validation on a single projection (one weight matrix, one matmul) rather than a full MoE layer. Identify which projection's PCC is degraded.

4. **Check `fp32_dest_acc_en`.** For bfloat8_b weights, switching from `fp32_dest_acc_en=False` to `True` often improves PCC by 0.005–0.01. If forward-pass PCC is borderline, verify this setting is enabled.

5. **Check `out_subblock` dimensions.** Incorrectly large subblock values can cause silent numerical errors or raise runtime assertions. Reduce to `out_subblock_h=1, out_subblock_w=1` as a diagnostic step.

> **Tip:** Run weight-level PCC validation for every expert weight at model load time and log the results. This creates a baseline record that can be compared against later if model output quality degrades after a TTNN version update or hardware configuration change.

---

**Next:** [Chapter 3 — Accuracy Analysis for MoE Expert Quantization](../ch03_accuracy_analysis/index.md)
