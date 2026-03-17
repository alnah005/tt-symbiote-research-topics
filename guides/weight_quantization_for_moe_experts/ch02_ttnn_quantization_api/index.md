# Chapter 2: TTNN Quantization API

This chapter covers the TTNN (Tenstorrent Neural Network) API surface for converting weights to reduced-precision formats, loading them onto device memory, and computing with them through matmul kernels. By the end of this chapter you will have a working mental model of how bfloat8_b and bfloat4_b weights flow from checkpoint files to hardware execution.

## Prerequisites

**Chapter 1** must be completed before proceeding. Chapter 1 defines:
- The bfloat8_b and bfloat4_b tile-packed formats and their block floating-point encoding
- The tile layout (TILE_LAYOUT) constraint: both spatial dimensions of the weight tensor must be multiples of 32 before conversion
- Expected accuracy trade-offs relative to bfloat16

If you have not read Chapter 1, the compute kernel configuration choices in this chapter will not make sense.

## Learning Objectives

After completing this chapter you will be able to:

1. Call `ttnn.as_tensor` with the correct `dtype` and `layout` arguments to convert a bfloat16 weight tensor to bfloat8_b or bfloat4_b on device.
2. Construct a `WormholeComputeKernelConfig` configured for LoFi (low fidelity) and HiFi2 (high fidelity 2) math fidelity modes.
3. Pass the compute kernel config as the `compute_kernel_config` argument to `ttnn.linear`.
4. Explain why no explicit user-side dequantization call is needed at inference time — TTNN's matmul kernel handles the dequantization path internally.
5. Apply `shard_dims` when converting expert weight tensors for T3K (Tenstorrent 3000-series multi-chip) distributed inference.

## Chapter Structure

| File | Topic |
|---|---|
| `weight_conversion.md` | Converting weights at checkpoint load time with `ttnn.as_tensor` |
| `compute_kernel_config.md` | `WormholeComputeKernelConfig` fields and how they affect matmul precision |
| `dtype_in_linear_and_matmul.md` | How weight dtype drives kernel dispatch inside `ttnn.linear` |
| `validation_patterns.md` | Verifying conversion accuracy with Pearson Correlation Coefficient checks |

## Quick Reference: bfloat16 to bfloat4_b Conversion

```python
weight_tt = ttnn.as_tensor(weight, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

For the full annotated example with load-time vs. on-the-fly comparison, see `weight_conversion.md`.

## Next Steps

Continue to `weight_conversion.md` for a detailed walkthrough of load-time weight conversion, transposition considerations, and T3K sharding.
