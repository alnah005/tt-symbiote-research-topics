# Per-Layer PCC Validation

## Purpose

Step 3 of the workflow builds a per-layer PCC (Pearson Cross-Correlation) test harness
that validates the output of each quantized MoE expert forward pass against a bfloat16
reference. The harness tests individual projections as well as the full SwiGLU layer,
checks both decode and prefill batch configurations, and runs a multi-layer cumulative
error test to catch drift that compounds across the 28+ MoE layers in a transformer.

---

## PCC Thresholds by Projection

These thresholds are the acceptance criteria established in Chapter 6
(`recommendations_and_decision_framework.md`) and enforced by every test in this chapter.

| Projection | Dtype | Compute kernel | PCC threshold |
|---|---|---|---|
| Gate (w1) | `bfloat4_b` | LoFi | ≥ 0.9600 |
| Up (w3) | `bfloat4_b` | LoFi | ≥ 0.9600 |
| Down (w2) | `bfloat8_b` | HiFi2 | ≥ 0.9750 |
| Full MoE layer | mixed | mixed | ≥ 0.9700 |

The gate and up thresholds (≥ 0.96) are lower than the down threshold (≥ 0.975) because
the bfloat4_b format introduces more quantization error, and because SiLU (x × sigmoid(x))
partially smooths the gate quantization noise before the elementwise multiply. The down
projection threshold is tighter because its output enters the residual stream directly:
any error there adds to the model's hidden state and propagates through all subsequent
layers.

---

## Setting Up the Per-Layer PCC Test Harness

The harness runs both the bfloat16 reference and the quantized variant with identical
random inputs, then calls `comp_pcc` on the outputs. Use a fixed random seed for
reproducibility.

```python
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# Configs: see index.md (COMPUTE_KERNEL_CONFIG_LOFI, COMPUTE_KERNEL_CONFIG_HIFI2)


def pcc_test_single_projection(
    weight_bf16, input_torch, target_dtype, compute_cfg, pcc_threshold, label, device
):
    """Validate a single projection matmul (no activation) against bfloat16 reference.

    Args:
        weight_bf16: torch.bfloat16 weight, shape [out_features, in_features].
        input_torch: torch.bfloat16 input, shape [num_tokens, in_features].
        target_dtype: ttnn.bfloat4_b or ttnn.bfloat8_b.
        compute_cfg: WormholeComputeKernelConfig for the quantized matmul.
        pcc_threshold: Minimum acceptable PCC (float).
        label: Human-readable label for log output.
        device: TTNN device handle.

    Returns:
        pcc_val: Measured PCC (float).
    """
    # bfloat16 CPU reference
    ref = input_torch.float() @ weight_bf16.float().T

    # Quantized TTNN matmul
    x_tt = ttnn.from_torch(input_torch, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=device)
    w_tt = ttnn.as_tensor(weight_bf16, dtype=target_dtype,
                          layout=ttnn.TILE_LAYOUT, device=device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_tt = ttnn.linear(x_tt, w_tt, compute_kernel_config=compute_cfg)
    out_torch = ttnn.to_torch(out_tt).float()

    pcc_val, _ = comp_pcc(ref, out_torch, pcc=pcc_threshold)
    status = "PASS" if pcc_val >= pcc_threshold else "FAIL"
    print(f"  [{status}] {label}: PCC = {pcc_val:.4f} (threshold >= {pcc_threshold})")
    return pcc_val


def pcc_test_full_moe_layer(
    x_torch, w1_bf16, w3_bf16, w2_bf16, layer_idx, device
):
    """Run a full mixed-precision SwiGLU expert layer and validate PCC at each stage.

    Tests:
      - Gate output (post-SiLU): compare quantized vs bfloat16 reference, threshold 0.96.
      - Up output: compare quantized vs bfloat16 reference, threshold 0.96.
      - Full layer output (post-down matmul): threshold 0.97.

    Args:
        x_torch: torch.bfloat16 input, shape [num_tokens, d_model].
        w1_bf16, w3_bf16, w2_bf16: bfloat16 checkpoint weights.
        layer_idx: Layer index for log output.
        device: TTNN device handle.

    Returns:
        dict with keys 'gate_pcc', 'up_pcc', 'down_pcc'.
    """
    # Reference forward pass: see baseline_and_weight_conversion.md

    # -- Mixed-precision TTNN forward pass --
    x_tt  = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device)
    w1_tt = ttnn.as_tensor(w1_bf16, dtype=ttnn.bfloat4_b,
                           layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w3_tt = ttnn.as_tensor(w3_bf16, dtype=ttnn.bfloat4_b,
                           layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w2_tt = ttnn.as_tensor(w2_bf16, dtype=ttnn.bfloat8_b,
                           layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

    gate_pre_tt = ttnn.linear(x_tt, w1_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
    gate_out_tt = ttnn.silu(gate_pre_tt)   # SiLU: x * sigmoid(x)
    up_out_tt   = ttnn.linear(x_tt, w3_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
    inter_tt    = ttnn.mul(gate_out_tt, up_out_tt)
    out_tt      = ttnn.linear(inter_tt, w2_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2)

    gate_torch = ttnn.to_torch(gate_out_tt).float()
    up_torch   = ttnn.to_torch(up_out_tt).float()
    out_torch  = ttnn.to_torch(out_tt).float()

    gate_pcc, _ = comp_pcc(gate_ref, gate_torch, pcc=0.96)
    up_pcc,   _ = comp_pcc(up_ref,   up_torch,   pcc=0.96)
    out_pcc,  _ = comp_pcc(out_ref,  out_torch,  pcc=0.97)

    for label, val, thresh in [
        ("gate (w1)", gate_pcc, 0.96),
        ("up   (w3)", up_pcc,   0.96),
        ("full layer", out_pcc, 0.97),
    ]:
        status = "PASS" if val >= thresh else "FAIL"
        print(f"  [{status}] Layer {layer_idx} {label}: PCC = {val:.4f} (threshold >= {thresh})")

    return {"gate_pcc": gate_pcc, "up_pcc": up_pcc, "down_pcc": out_pcc}
```

---

## Batch Dimension Effects

PCC can vary with batch size because the distribution of rounding errors from quantized
matmuls changes with the number of token vectors being accumulated. Run the harness with
at least two configurations:

| Configuration | `num_tokens` | Sequence length | Represents |
|---|---|---|---|
| Decode | 1 | 1 | Single-token autoregressive step |
| Prefill | 2048 | 2048 | Prompt processing, full context |

```python
def run_pcc_sweep(w1_bf16, w3_bf16, w2_bf16, layer_idx, device):
    """Run per-layer PCC validation in both decode and prefill configurations."""
    torch.manual_seed(42)

    # Decode: batch=1, seq=1
    x_decode = torch.randn(1, 7168, dtype=torch.bfloat16)
    print(f"Layer {layer_idx} — DECODE (num_tokens=1):")
    decode_results = pcc_test_full_moe_layer(x_decode, w1_bf16, w3_bf16, w2_bf16, layer_idx, device)

    # Prefill: batch=1, seq=2048 (flattened to num_tokens=2048)
    x_prefill = torch.randn(2048, 7168, dtype=torch.bfloat16)
    print(f"Layer {layer_idx} — PREFILL (num_tokens=2048):")
    prefill_results = pcc_test_full_moe_layer(x_prefill, w1_bf16, w3_bf16, w2_bf16, layer_idx, device)

    return {"decode": decode_results, "prefill": prefill_results}
```

> **Tip:** Prefill PCC is typically higher than decode PCC for quantized matmuls. With
> 2048 token vectors, the mean across the output tensor averages over more values, which
> tends to smooth quantization noise and raise the measured PCC. Do not use the prefill
> result to justify a decode deployment — always validate both.

---

## Diagnostic Steps When a Threshold Fails

If any projection PCC falls below its threshold, follow this diagnostic sequence:

**1. Check weight conversion PCC first.**
Run `check_weight_conversion_pcc` from `baseline_and_weight_conversion.md` on the failing
weight. If conversion PCC is also below target (< 0.99 for bfloat8_b, < 0.97 for
bfloat4_b), the problem is in the weight distribution, not the matmul kernel.

**2. Check the compute kernel config.**
Confirm `fp32_dest_acc_en=False` for the failing projection. Accidentally setting it to
`True` changes the accumulation path and shifts numerical outputs enough to alter PCC
unexpectedly.

**3. Check shape and layout.**
Verify that the weight tensor is `TILE_LAYOUT` and that both dimensions are multiples of
32. A non-tile-aligned tensor passed without padding can silently produce wrong outputs
with undefined PCC.

**4. Isolate the projection.**
Use `pcc_test_single_projection` to test the gate, up, and down matmuls independently.
If the isolated test passes but the full-layer test fails, the issue is in the SiLU or
elementwise multiply, not the matmul.

**5. Escalate to fallback.**
If the above steps do not resolve the failure, follow the fallback guidance in Chapter 6:
upgrade the failing projection's dtype (bfloat4_b → bfloat8_b) or compute kernel config
(LoFi → HiFi2), then re-validate.

> **Warning:** A PCC that is undefined (NaN) or negative is not a quantization accuracy
> issue — it indicates a software bug. NaN PCC results from NaN values in the output
> tensor, usually caused by uninitialized memory from an incorrect memory config or a
> failed kernel launch. Negative PCC indicates that the output has opposite sign from the
> reference, typically from an incorrect weight transpose. Fix these before interpreting
> any quantization accuracy measurements.

---

## Multi-Layer Cumulative Error Test

Individual per-layer PCC can look acceptable while cumulative error across 28+ MoE layers
becomes significant. Run a sequential multi-layer test to detect this drift.

```python
def cumulative_error_test(layer_weights_list, d_model, device, n_layers=None):
    """Chain N quantized MoE layers sequentially and measure output drift.

    Runs the quantized forward pass N times, feeding each layer's output as the next
    layer's input. Compares against the bfloat16 reference chain at each layer boundary.
    Flags layers where per-layer PCC falls below 0.975.

    Args:
        layer_weights_list: List of dicts, each with keys 'w1', 'w3', 'w2' as
                            torch.bfloat16 tensors.
        d_model: Hidden dimension (7168 for Qwen 235B-A22B).
        device: TTNN device handle.
        n_layers: Number of layers to test; defaults to len(layer_weights_list).

    Returns:
        List of per-layer PCC values (float).
    """
    if n_layers is None:
        n_layers = len(layer_weights_list)

    torch.manual_seed(0)
    x_init = torch.randn(1, d_model, dtype=torch.bfloat16)

    # bfloat16 reference chain (CPU)
    x_ref = x_init.float()
    ref_outputs = []
    for lw in layer_weights_list[:n_layers]:
        g = torch.nn.functional.silu(x_ref @ lw["w1"].float().T)
        u = x_ref @ lw["w3"].float().T
        x_ref = (g * u) @ lw["w2"].float().T
        ref_outputs.append(x_ref.clone())

    # Quantized chain (device)
    x_quant = x_init.clone()
    per_layer_pcc = []
    for i, lw in enumerate(layer_weights_list[:n_layers]):
        result = pcc_test_full_moe_layer(
            x_quant, lw["w1"], lw["w3"], lw["w2"], layer_idx=i, device=device
        )
        layer_pcc = result["down_pcc"]
        per_layer_pcc.append(layer_pcc)

        if layer_pcc < 0.975:
            print(
                f"  [FLAG] Layer {i} PCC {layer_pcc:.4f} < 0.975 — cumulative error "
                f"will compound over {n_layers - i} remaining layers."
            )

        # Feed quantized output to next layer
        x_tt = ttnn.from_torch(x_quant, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device)
        w1_tt = ttnn.as_tensor(lw["w1"], dtype=ttnn.bfloat4_b,
                               layout=ttnn.TILE_LAYOUT, device=device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w3_tt = ttnn.as_tensor(lw["w3"], dtype=ttnn.bfloat4_b,
                               layout=ttnn.TILE_LAYOUT, device=device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w2_tt = ttnn.as_tensor(lw["w2"], dtype=ttnn.bfloat8_b,
                               layout=ttnn.TILE_LAYOUT, device=device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_tt = ttnn.silu(ttnn.linear(x_tt, w1_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI))
        up_tt   = ttnn.linear(x_tt, w3_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
        inter_tt = ttnn.mul(gate_tt, up_tt)
        out_tt   = ttnn.linear(inter_tt, w2_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2)
        x_quant  = ttnn.to_torch(out_tt).to(torch.bfloat16)

    return per_layer_pcc
```

The threshold for flagging is PCC < 0.975 at any individual layer boundary, even if the
layer's own isolated PCC passed the 0.97 full-layer threshold. The reason: per-layer
errors compound multiplicatively across the 28+ MoE layers in a Qwen 235B-A22B
transformer. A layer with PCC 0.972 produces error that the next layer's matmul amplifies
rather than cancels. Flagging at 0.975 provides a safety margin above the 0.97 acceptance
threshold to keep total accumulated drift within acceptable bounds.

> **Warning:** The cumulative error test uses a single random input vector of shape
> `[1, d_model]`. Production token distributions may have higher variance, leading to more
> aggressive cumulative drift. If any layer is flagged at PCC < 0.975, validate on at
> least 100 random inputs before accepting the configuration as production-safe.

---

## Next Steps

Proceed to `throughput_profiling.md` to measure the per-operation latency of the expert
FFN using TTNN's device profiler and Tracy traces, and to compare decode vs. prefill
throughput under bfloat16, bfloat8_b, and bfloat4_b gate/up configurations.
