# Qwen 235B-A22B Adaptation Guide

## Starting Point

This guide assumes you have a Qwen 235B-A22B checkpoint loaded in bfloat16, either as
HuggingFace `safetensors` files or as a `transformers` model state dict. The goal is to
convert all MoE expert projection weights to the mixed-precision layout described in
this chapter and validate the result before committing to the converted weights.

Qwen 235B-A22B model dimensions:

| Dimension | Value |
|---|---|
| `d_model` | 7168 |
| `d_ff` | 2048 |
| `num_experts` | 128 |
| `top_k` | 8 (tokens are routed to 8 experts per forward pass) |
| Expert projections | gate (`gate_proj`), up (`up_proj`), down (`down_proj`) |

These dimensions are tile-aligned (`7168 = 224 × 32`, `2048 = 64 × 32`), so no padding
is required before conversion.

## Step 1 — Identify All MoE Expert Weight Parameters

Qwen 235B-A22B follows the HuggingFace naming convention. MoE expert weights appear
under parameter names of the form:

```
model.layers.<layer_idx>.mlp.experts.<expert_idx>.gate_proj.weight
model.layers.<layer_idx>.mlp.experts.<expert_idx>.up_proj.weight
model.layers.<layer_idx>.mlp.experts.<expert_idx>.down_proj.weight
```

```python
import re

def get_moe_expert_param_names(state_dict):
    """Return sorted lists of gate, up, and down expert parameter names.

    Args:
        state_dict: dict mapping parameter name -> torch.Tensor (bfloat16).

    Returns:
        (gate_names, up_names, down_names): Each is a list of parameter name strings,
        sorted by (layer_idx, expert_idx).
    """
    pattern = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    )
    gate_names, up_names, down_names = [], [], []

    for name in state_dict:
        m = pattern.match(name)
        if m:
            proj = m.group(3)
            if proj == "gate_proj":
                gate_names.append(name)
            elif proj == "up_proj":
                up_names.append(name)
            elif proj == "down_proj":
                down_names.append(name)

    # Sort by (layer_idx, expert_idx) for deterministic ordering
    key_fn = lambda n: tuple(int(x) for x in re.findall(r"\d+", n)[:2])
    return sorted(gate_names, key=key_fn), sorted(up_names, key=key_fn), sorted(down_names, key=key_fn)
```

> **Tip:** Before converting, verify that you have found the expected number of
> parameters: `num_layers_with_moe × num_experts × 3`. For Qwen 235B-A22B, cross-check
> the total count against the model card.

## Step 2 — Convert Gate and Up Projection Weights to bfloat4_b

Gate (`gate_proj`) and up (`up_proj`) weights are converted to `bfloat4_b` with
`TILE_LAYOUT`. HuggingFace stores weights in shape `[out_features, in_features]`, which
is `[d_ff, d_model]` for gate and up projections — already in the transposed form
expected by TTNN's `ttnn.linear`.

```python
import ttnn
import torch

def convert_gate_up_to_bfloat4b(weight_bf16, device):
    """Convert a single gate or up projection weight to bfloat4_b.

    Args:
        weight_bf16: torch.Tensor of shape [d_ff, d_model] in bfloat16.
                     HuggingFace convention: [out_features, in_features].
        device: TTNN device.

    Returns:
        weight_tt: ttnn.Tensor, dtype=bfloat4_b, layout=TILE_LAYOUT, in DRAM.
    """
    # ttnn.as_tensor quantizes to bfloat4_b at conversion time (one-time cost)
    weight_tt = ttnn.as_tensor(
        weight_bf16,
        dtype=ttnn.bfloat4_b,        # 4-bit block float; 4× size reduction
        layout=ttnn.TILE_LAYOUT,      # required; bfloat4_b is tile-only
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return weight_tt
```

> **Warning:** `bfloat4_b` requires `ttnn.TILE_LAYOUT`. Passing `ttnn.ROW_MAJOR_LAYOUT`
> will raise an error. Always specify `layout=ttnn.TILE_LAYOUT` explicitly.

## Step 3 — Convert Down Projection Weights to bfloat8_b

Down (`down_proj`) weights are stored in HuggingFace as `[d_model, d_ff]`
(`[out_features, in_features]`), already transposed for use in `ttnn.linear` where the
matmul computes `inter @ w2.T`.

```python
def convert_down_to_bfloat8b(weight_bf16, device):
    """Convert a single down projection weight to bfloat8_b.

    Args:
        weight_bf16: torch.Tensor of shape [d_model, d_ff] in bfloat16.
                     HuggingFace convention: [out_features, in_features].
        device: TTNN device.

    Returns:
        weight_tt: ttnn.Tensor, dtype=bfloat8_b, layout=TILE_LAYOUT, in DRAM.
    """
    weight_tt = ttnn.as_tensor(
        weight_bf16,
        dtype=ttnn.bfloat8_b,        # 8-bit block float; 2× size reduction
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return weight_tt
```

## Step 4 — Assign Compute Kernel Configs per Projection in the Forward Pass

In the MoE expert forward pass function, use the correct compute kernel config for each
projection. Both configs use `fp32_dest_acc_en=False`.

```python
import ttnn

# Gate and up: LoFi, fp32_dest_acc_en=False
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Down: HiFi2, fp32_dest_acc_en=False
COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


def qwen_expert_forward(x_tt, w1_tt, w3_tt, w2_tt):
    """Mixed-precision SwiGLU expert forward pass for Qwen 235B-A22B.

    Args:
        x_tt: Routed token activations, shape [num_tokens, d_model], bfloat16.
        w1_tt: Gate weight, bfloat4_b, shape [d_ff, d_model].
        w3_tt: Up weight, bfloat4_b, shape [d_ff, d_model].
        w2_tt: Down weight, bfloat8_b, shape [d_model, d_ff].

    Returns:
        out_tt: Expert output, shape [num_tokens, d_model], bfloat16.
                Caller accumulates this into the residual stream.
    """
    # Gate projection: bfloat4_b + LoFi
    gate_pre = ttnn.linear(x_tt, w1_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
    gate_out = ttnn.silu(gate_pre)   # SiLU compresses quantization noise

    # Up projection: bfloat4_b + LoFi
    up_out = ttnn.linear(x_tt, w3_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)

    # SwiGLU combine
    inter = ttnn.mul(gate_out, up_out)

    # Down projection: bfloat8_b + HiFi2
    out_tt = ttnn.linear(inter, w2_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2)
    return out_tt
```

## Step 5 — Run the Validation Suite

### Per-Layer PCC Validation

For each MoE layer, run both the bfloat16 reference and the quantized version with
identical inputs and compute PCC between their outputs.

```python
def validate_moe_layer_pcc(layer_idx, x_torch, w1_bf16, w3_bf16, w2_bf16, device):
    """Validate per-projection and full-layer PCC for one MoE layer.

    Thresholds:
        gate_out * up_out (inter): PCC >= 0.96
        w2_out (full layer):       PCC >= 0.975

    Args:
        layer_idx: Layer index for logging.
        x_torch: Input activation, torch.bfloat16, shape [num_tokens, d_model].
        w1_bf16, w3_bf16, w2_bf16: Original bfloat16 weights from checkpoint.
        device: TTNN device.

    Returns:
        dict with 'inter_pcc' and 'out_pcc'.
    """
    import torch

    # -- bfloat16 CPU reference --
    gate_ref = torch.nn.functional.silu(x_torch.float() @ w1_bf16.float().T)
    up_ref   = x_torch.float() @ w3_bf16.float().T
    inter_ref = gate_ref * up_ref
    out_ref   = inter_ref @ w2_bf16.float().T

    # -- Mixed-precision TTNN forward pass --
    w1_tt = convert_gate_up_to_bfloat4b(w1_bf16, device)
    w3_tt = convert_gate_up_to_bfloat4b(w3_bf16, device)
    w2_tt = convert_down_to_bfloat8b(w2_bf16, device)
    x_tt  = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)

    gate_pre_tt = ttnn.linear(x_tt, w1_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
    gate_out_tt = ttnn.silu(gate_pre_tt)
    up_out_tt   = ttnn.linear(x_tt, w3_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI)
    inter_tt    = ttnn.mul(gate_out_tt, up_out_tt)
    out_tt      = ttnn.linear(inter_tt, w2_tt, compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2)

    # Bring results to CPU for PCC computation
    inter_tt_cpu = ttnn.to_torch(inter_tt).float()
    out_tt_cpu   = ttnn.to_torch(out_tt).float()

    # comp_pcc is the standard TTNN PCC helper (see Chapter 2, validation_patterns.md)
    inter_pcc = comp_pcc(inter_ref, inter_tt_cpu, pcc=0.96)
    out_pcc   = comp_pcc(out_ref,   out_tt_cpu,   pcc=0.975)

    print(f"Layer {layer_idx}: inter PCC = {inter_pcc:.4f}, out PCC = {out_pcc:.4f}")
    assert inter_pcc >= 0.96,   f"Layer {layer_idx} inter PCC {inter_pcc:.4f} < 0.96"
    assert out_pcc   >= 0.975,  f"Layer {layer_idx} out PCC {out_pcc:.4f} < 0.975"

    return {"inter_pcc": inter_pcc, "out_pcc": out_pcc}
```

### End-to-End Perplexity on a Calibration Set

After per-layer validation, run end-to-end perplexity on a calibration set to confirm
that per-layer PCC thresholds translate to acceptable model quality:

1. Use WikiText-2 or C4, evaluating on 512-token segments.
2. Compare perplexity against the bfloat16 baseline.
3. Accept if perplexity delta ≤ 2.0 PPL for the mixed-precision configuration (bfloat4_b
   gate/up + bfloat8_b down).
4. If the delta exceeds 2.0 PPL, consult the fallback guidance in `down_projection_strategy.md`.

> **Tip:** Running the full calibration perplexity sweep takes time. Do a quick sanity
> check first: generate 10 short completions and visually inspect for degenerate output
> (repetition loops, incoherence). If the quick check passes, proceed to formal
> perplexity measurement.

## Checkpoint Caching

Weight conversion with `ttnn.as_tensor` for 128 experts × 3 projections is a one-time
cost at startup but adds noticeable latency (typically tens of seconds). Cache the
converted TTNN weight tensors to disk to avoid re-conversion on every restart.

```python
import os
import ttnn

CACHE_DIR = "/path/to/converted_weights_cache"

def load_or_convert_weight(param_name, weight_bf16, dtype, device, cache_dir=CACHE_DIR):
    """Load a converted TTNN weight from cache, or convert and cache it.

    Args:
        param_name: Unique string identifying this weight (used as cache filename).
        weight_bf16: Original bfloat16 torch.Tensor.
        dtype: Target TTNN dtype (ttnn.bfloat4_b or ttnn.bfloat8_b).
        device: TTNN device.
        cache_dir: Directory for cached weight files.

    Returns:
        weight_tt: TTNN tensor on device in DRAM.
    """
    # Sanitize parameter name for use as a filename
    safe_name = param_name.replace(".", "_").replace("/", "_")
    cache_path = os.path.join(cache_dir, f"{safe_name}_{dtype}.bin")

    if os.path.exists(cache_path):
        # Load from cache: avoids re-quantization at startup
        weight_tt = ttnn.load_tensor(cache_path)
        weight_tt = ttnn.to_device(weight_tt, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        # Convert and save to cache
        os.makedirs(cache_dir, exist_ok=True)
        weight_tt = ttnn.as_tensor(
            weight_bf16,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.dump_tensor(cache_path, ttnn.from_device(weight_tt))

    return weight_tt
```

> **Warning:** Cached weights are tied to the TTNN version that produced them. After a
> TTNN version update, delete the cache and re-convert to ensure the on-disk format is
> still valid. Stale cached weights can silently produce incorrect outputs.

## Complete Conversion Workflow

```python
def convert_qwen_moe_checkpoint(state_dict, device, cache_dir=CACHE_DIR):
    """Convert all Qwen 235B-A22B MoE expert weights to mixed precision.

    Returns:
        dict mapping parameter name -> TTNN tensor.
    """
    gate_names, up_names, down_names = get_moe_expert_param_names(state_dict)
    converted = {}

    for name in gate_names:
        converted[name] = load_or_convert_weight(
            name, state_dict[name], ttnn.bfloat4_b, device, cache_dir
        )

    for name in up_names:
        converted[name] = load_or_convert_weight(
            name, state_dict[name], ttnn.bfloat4_b, device, cache_dir
        )

    for name in down_names:
        converted[name] = load_or_convert_weight(
            name, state_dict[name], ttnn.bfloat8_b, device, cache_dir
        )

    print(f"Converted {len(converted)} expert weight tensors to mixed precision.")
    return converted
```

## Summary of the Five Steps

| Step | Action | Key detail |
|---|---|---|
| 1 | Identify MoE expert parameters | Match `model.layers.*.mlp.experts.*.{gate,up,down}_proj.weight` |
| 2 | Convert gate and up weights | `dtype=ttnn.bfloat4_b`, `layout=ttnn.TILE_LAYOUT` |
| 3 | Convert down weights | `dtype=ttnn.bfloat8_b`, `layout=ttnn.TILE_LAYOUT` |
| 4 | Assign compute kernel configs | Gate/up: `COMPUTE_KERNEL_CONFIG_LOFI`; down: `COMPUTE_KERNEL_CONFIG_HIFI2` |
| 5 | Validate and cache | PCC ≥ 0.96 (inter), ≥ 0.975 (out); cache converted weights to avoid re-conversion |

---

**Next:** [Chapter 6 — Comparative Study: DeepSeek-V3 vs. Qwen](../ch06_comparative_study/index.md)
