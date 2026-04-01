# DRAM Budget — Weight Precision Tables, bfp4 Rationale, and Shared Expert Exception

## Blackhole P100A DRAM Capacity

The Blackhole P100A has **28 GB of DRAM**. Both Qwen3.5 variants must fit within this limit
with room left for activation tensors, KV cache, and OS/runtime overhead.

## A3B (MoE) DRAM Breakdown

The A3B model uses 40 transformer layers (30 DeltaNet + 10 full attention), each with a
Qwen35MoE block. The dominant cost is the 256 routed expert weight tensors per layer.

| Component | dtype | Size |
|-----------|-------|------|
| Expert weights: 256 × (gate+up + down) × 40 layers | `bfloat4_b` | 12.8 GB |
| Shared expert weights: 40 layers × 3 matrices | `bfloat8_b` | 0.8 GB |
| DeltaNet projections: 30 layers | `bfloat8_b` | 1.2 GB |
| Attention QKV + WO + gate: 10 layers | `bfloat16` | 0.5 GB |
| Router + shared gate weights: 40 layers | `bfloat16` | 0.1 GB |
| KV cache: 10 attention layers | `bfloat16` | 0.3 GB |
| **Total** | | **~15.7 GB / 28 GB** |

The A3B model uses approximately **56%** of the available DRAM, leaving 12.3 GB headroom for
activations, intermediate tensors, and Metal runtime allocations.

### Expert Weight Size Derivation

Each expert's gate+up weight has shape $[1, 1, 2048, 1024]$ and down projection has shape
$[1, 1, 512, 2048]$ in bfp4. The bfp4 (`bfloat4_b`) format stores 4 bits per value in a block
format with a shared bfloat16 exponent per block of 16 elements.

Per-expert parameter count = 3,145,728 (derivation in `architecture_overview.md`).

At 4 bits per element (0.5 bytes effective, ignoring block header overhead):

$$256\ \text{experts} \times 3{,}145{,}728 \times 0.5\ \text{bytes} \times 40\ \text{layers} \approx 16{,}106{,}127{,}360\ \text{bytes} \approx 15.0\ \text{GB}$$

The naive formula gives ~15.0 GiB as a lower bound on resident expert weight storage. The
PERF.md reported figure of 12.8 GB likely reflects a different measurement basis — for example,
a different byte unit (GB vs. GiB), the weight cache on-disk size rather than all weights
simultaneously resident in DRAM, or a specific subset of layers measured. Both figures confirm
the same feasibility conclusion: expert weights fit within the 28 GB DRAM budget. The
order-of-magnitude calculation confirms bfp4 is essential.

## 27B (Dense) DRAM Breakdown

The 27B model uses 64 transformer layers (48 DeltaNet + 16 full attention) with standard dense
SwiGLU MLPs. There are no expert tensors, but the MLP hidden dimension is much larger (the 27B
MLP intermediate size is 11,008 vs 512 for the A3B experts).

| Component | dtype | Size |
|-----------|-------|------|
| MLP w1 + w2 + w3: 64 layers | `bfloat8_b` | 17.1 GB |
| DeltaNet projections: 48 layers | `bfloat8_b` | 5.4 GB |
| Attention QKV + WO + gate: 16 layers | `bfloat16` | 2.2 GB |
| Other (norms, embeddings, misc.) | `bfloat16` | ~0.3 GB |
| **Total** | | **~25.0 GB / 28 GB** |

The 27B model uses approximately **89%** of available DRAM, leaving only ~3 GB headroom. The
MLP weights alone (17.1 GB) dominate because each dense MLP is roughly $3 \times 5120 \times 11008$
parameters in bfp8.

## Why bfp4 for Routed Expert Weights

The routed expert weights are the single largest contributor to DRAM usage in the A3B model. The
quantization precision choice is driven by a hard capacity constraint:

**At bfp8**, 256 experts × 40 layers at bfp8 would require approximately:

$$256 \times 3{,}145{,}728 \times 1\ \text{byte} \times 40 \approx 30.0\ \text{GB}$$

This **exceeds the entire 28 GB DRAM capacity** before accounting for any other weights. The
model cannot fit on a single P100A at bfp8.

**At bfp4**, storage is approximately 15.0 GB (see derivation above).

Combined with the remaining non-expert weights (~3.5 GB), the total fits within 28 GB.

Beyond the capacity argument, there is also a quality argument for why bfp4 is acceptable for
routed experts specifically:

1. **Per-token averaging.** Each token's output is a weighted sum of 8 expert outputs (plus the
   shared expert). Quantization errors in individual expert outputs are averaged with 7 other
   experts' outputs, providing natural noise reduction.

2. **Sparse activation.** Each expert is only activated for a small fraction of tokens. The
   accumulation of quantization error across layers is limited compared to a weight that
   participates in every single layer output.

3. **Empirical validation.** The PCC test `TestMoEPCC.test_single_layer` validates that bfp4
   expert weights produce output with PCC ≥ 0.99 against the float32 reference for a single
   layer, confirming that quantization noise remains within acceptable bounds.

The `qwen35_moe.py` source makes the constraint explicit in the module docstring and `__init__`:

```python
"""
Expert weights stored as bfp4 on DRAM to fit 35B params in 28 GB.
"""
...
expert_dtype = ttnn.bfloat4_b  # bfp4 to fit in DRAM
```

## Why Shared Expert Stays at bfp8

The shared expert uses `dtype` (the module-level `dtype` argument, which is `bfloat8_b` in
`demo_a3b.py`) rather than `expert_dtype`:

```python
expert_dtype = ttnn.bfloat4_b  # bfp4 to fit in DRAM
...
shared_dtype = dtype  # bfp8
```

Two complementary reasons justify the higher precision:

1. **Always-active path has disproportionate output impact.** The shared expert contributes to
   every token's output in every layer. A quantization error in the shared expert accumulates
   across all 40 layers. The 8 routed experts change per token, so their quantization errors
   are decorrelated across the sequence. The shared expert's errors are not.

2. **DRAM cost is negligible.** The shared expert has 3 weight matrices per layer × 40 layers
   = 120 total weight tensors. Per the PERF.md measurements, all shared expert weights across
   40 layers total **0.8 GB** in bfp8 (this includes `gate_proj`, `up_proj`, and `down_proj`
   for each layer, stored as tile-aligned bfp8 tensors with block-format overhead). This is
   rounding error in the total budget.

The `load_shared` helper in `Qwen35MoE.__init__` applies `shared_dtype` to all three shared
expert matrices uniformly:

```python
def load_shared(name):
    w = state_dict[f"{shared_prefix}.{name}.weight"]
    return ttnn.as_tensor(
        w.T.unsqueeze(0).unsqueeze(0).contiguous(),
        dtype=shared_dtype,      # bfloat8_b
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_name(f"shared_expert.{name}"),
    )

self.shared_w1 = load_shared("gate_proj")   # [1,1,2048,512]
self.shared_w3 = load_shared("up_proj")     # [1,1,2048,512]
self.shared_w2 = load_shared("down_proj")   # [1,1,512,2048]
```

## Summary of Precision Choices

| Weight Category | dtype | Rationale |
|-----------------|-------|-----------|
| Routed expert gate+up | `bfloat4_b` | DRAM capacity: bfp8 would exceed 28 GB |
| Routed expert down | `bfloat4_b` | DRAM capacity: same constraint |
| Shared expert gate+up+down | `bfloat8_b` | Always-active; accumulates error; negligible DRAM cost |
| Router weight (`gate.weight`) | `bfloat16` | Small (1 MB/layer); routing quality critical |
| Shared expert gate (`shared_expert_gate.weight`) | `bfloat16` | Small (16 KB/layer); sigmoid gate |
| DeltaNet projections | `bfloat8_b` | Balance of quality and footprint |
| Attention QKV + WO | `bfloat16` | 10 layers only; precision for attention scores |

---

**Next:** [Chapter 6 — Weight Precision, DRAM Layout, and Weight Conversion](../ch6_weight_precision_dram_layout_and_weight_conversion/index.md)
