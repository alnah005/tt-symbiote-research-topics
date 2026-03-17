# Dtype Flow Through `ttnn.linear`

Understanding how dtype propagates through `ttnn.linear` is important for reasoning about kernel dispatch, output precision, and the interaction with program configs. This page traces the dtype of each tensor — weight, activation, and output — through a `ttnn.linear` call.

## Dtype Is Set at Conversion Time, Not at Call Time

The weight tensor's dtype is fixed at the moment `ttnn.as_tensor` is called. When `ttnn.linear` is invoked, TTNN inspects the dtype of the weight tensor already stored on device and uses that to determine which internal matmul kernel to dispatch. There is no `dtype` argument on `ttnn.linear` itself for this purpose.

```python
import ttnn

# Two weights converted to different dtypes from the same checkpoint tensor
weight_8b = ttnn.as_tensor(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
weight_4b = ttnn.as_tensor(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Same ttnn.linear call syntax — different kernels dispatched based on weight dtype
out_8b = ttnn.linear(activation, weight_8b, compute_kernel_config=hifi2_config)
out_4b = ttnn.linear(activation, weight_4b, compute_kernel_config=lofi_config)
# out_8b uses the bfloat8_b unpack kernel path
# out_4b uses the bfloat4_b unpack kernel path
```

This design means the choice of weight precision is made once, at model load time, and affects every subsequent forward pass automatically. You cannot switch precision on a per-call basis without converting the weight tensor again.

## Kernel Dispatch by Weight Dtype

When `ttnn.linear` receives a weight tensor, the dispatch path is roughly:

1. Inspect `weight.dtype`.
2. If `dtype` is `ttnn.bfloat8_b` or `ttnn.bfloat4_b`, select the packed-tile unpack kernel variant.
3. The unpack kernel reads tile-packed data from DRAM, dequantizes each 32x32 tile to bfloat16 (dequantization always yields bfloat16), and feeds the result into the FPU (Floating Point Unit) MAC (multiply-accumulate) pipeline. `fp32_dest_acc_en=True` controls the precision of the accumulation register (where partial sums accumulate), not the dequantized tile format.
4. If `dtype` is `ttnn.bfloat16`, the standard bfloat16 kernel is used with no dequantization step.

From the caller's perspective, the API call is identical. The kernel selection is an implementation detail of TTNN's dispatch layer.

## Activation Dtype

Activation tensors are typically bfloat16 at the input to expert projections. The matmul kernel handles the asymmetry between bfloat16 activations and quantized weights natively — the unpack path dequantizes the weight tile to bfloat16 before the multiply, so both operands of the FPU multiply are bfloat16. `fp32_dest_acc_en=True` controls whether partial sums accumulate in fp32 in the destination register, not the dtype of the dequantized weight operand.

```python
# Typical activation shape entering an expert MLP projection:
# activation: [batch, seq_len, d_model] as bfloat16
# weight_tt:  [d_ff, d_model] as bfloat8_b (after transpose at load time)

output = ttnn.linear(
    activation,                        # bfloat16
    weight_tt,                         # bfloat8_b — dequantized internally per tile
    compute_kernel_config=hifi2_config,
)
# output dtype is bfloat16 by default
```

There is no user-side cast or padding of the activation tensor to match the weight dtype. The kernel handles the format mismatch.

## Output Dtype

The output tensor dtype follows this logic:

1. Accumulation precision is controlled by `fp32_dest_acc_en` in the compute kernel config.
   - `fp32_dest_acc_en=False`: partial sums accumulate in bfloat16 throughout.
   - `fp32_dest_acc_en=True`: partial sums accumulate in fp32 within the Tensix destination register file.
2. In both cases, the packer writes the result back to L1 (level-1 SRAM) or DRAM as **bfloat16**, unless an explicit output dtype is configured in the program config.

The output tensor returned to the caller is bfloat16. The fp32 accumulation happens only inside the Tensix core and is not visible as an fp32 tensor in DRAM.

```
Activation (bfloat16)  ──────────────────────────────────┐
                                                          ▼
Weight (bfloat8_b) → [unpack: tile-packed → bfloat16] → FPU MAC → [fp32 acc if enabled] → packer → Output (bfloat16)
```

## Program Config Interaction

`MatmulMultiCoreReuseMultiCastProgramConfig` controls how the matmul is tiled across Tensix cores. Its key parameters are:

| Parameter | Meaning |
|---|---|
| `in0_block_w` | Number of tiles along the shared dimension processed per inner loop iteration |
| `per_core_M` | Output tiles along M (rows) per core |
| `per_core_N` | Output tiles along N (columns) per core |
| `out_subblock_h` | Output subblock height in tiles (must fit in destination register) |
| `out_subblock_w` | Output subblock width in tiles (must fit in destination register) |

**Weight dtype does not change the tile count logic.** A `[d_ff, d_model]` bfloat8_b weight has the same number of tiles as a `[d_ff, d_model]` bfloat16 weight — tiles are always 32x32 elements. The program config parameters depend on the shape in tiles, not on the element dtype.

What *does* change with `fp32_dest_acc_en=True` is the constraint on `out_subblock_h` and `out_subblock_w`. When the destination register stores fp32 values (32 bits each) instead of bfloat16 values (16 bits each), the register file holds fewer tiles simultaneously. This means the maximum valid `out_subblock_h * out_subblock_w` product may be smaller when `fp32_dest_acc_en=True`.

```python
import ttnn

# Program config for a [d_ff=4096, d_model=2048] weight matrix
# Tile dimensions: [4096/32, 2048/32] = [128 tiles M, 64 tiles N]
# This config is valid for both bfloat16 and bfloat8_b weights of the same shape.
# Only out_subblock_h/out_subblock_w may need reduction when fp32_dest_acc_en=True.

program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 4),  # 8x4 core grid
    in0_block_w=2,          # process 2 tiles along d_model per inner iteration
    per_core_M=4,           # 4 output tiles along M per core
    per_core_N=2,           # 2 output tiles along N per core
    out_subblock_h=2,       # subblock height — may need to reduce for fp32 acc
    out_subblock_w=2,       # subblock width — may need to reduce for fp32 acc
    fuse_batch=True,
    transpose_mcast=False,
)

output = ttnn.linear(
    activation,
    weight_8b,                          # bfloat8_b weight
    program_config=program_config,
    compute_kernel_config=hifi2_config,  # fp32_dest_acc_en=True → check subblock limits
)
```

> **Warning:** If `out_subblock_h * out_subblock_w` exceeds the destination register tile capacity for your chosen `fp32_dest_acc_en` setting, TTNN will raise a runtime assertion. Start with `out_subblock_h=1, out_subblock_w=1` when debugging a new config, then increase to improve performance.

> **Tip:** The tile count parameters (`per_core_M`, `per_core_N`, `in0_block_w`) are determined by the tensor shape in tiles, which is the same regardless of whether the weight is bfloat16, bfloat8_b, or bfloat4_b. You can develop and validate your program config with a bfloat16 weight and then swap in the quantized weight without changing the program config.

## Summary

| Aspect | Behavior |
|---|---|
| Kernel dispatch | Determined by stored weight dtype, not by call-time argument |
| Activation dtype | bfloat16; hardware handles asymmetry with quantized weights |
| Output dtype | bfloat16 (fp32 accumulation is internal, not visible in output tensor) |
| Program config tile params | Unchanged by weight dtype; same shape = same tile count |
| `out_subblock` limits | May need reduction when `fp32_dest_acc_en=True` |

---

**Next:** [`validation_patterns.md`](./validation_patterns.md)
