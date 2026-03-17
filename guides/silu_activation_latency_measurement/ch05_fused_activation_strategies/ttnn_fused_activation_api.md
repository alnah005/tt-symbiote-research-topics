# TTNN Fused Activation API

This document describes the `activation` parameter on `ttnn.matmul`, the `fused_activation` field in matmul program configs, and the kernel-level semantics that make fusion effective for SiLU in MoE FFN compute sequences.

---

## The `activation` Parameter on `ttnn.matmul`

`ttnn.matmul` accepts an optional `activation` keyword argument that specifies a post-op activation to apply to the matmul output before the result is written to its destination tensor.

```python
output = ttnn.matmul(
    input_tensor_a,
    input_tensor_b,
    activation="silu",   # post-op fusion: SiLU applied in same kernel as matmul
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

Accepted string values for `activation` include:

| String | Activation |
|---|---|
| `"silu"` | SiLU: `x * sigmoid(x)` |
| `"relu"` | ReLU: `max(0, x)` |
| `"gelu"` | GELU: approximate Gaussian error linear unit |

The same parameter is available on `ttnn.linear`. For `ttnn.linear`, the activation is applied after the bias add.

---

## `fused_activation` in Program Configs

When a custom program config is provided to `ttnn.matmul`, the top-level `activation` parameter is superseded by the program config. The activation must be specified inside the config itself using the `fused_activation` field.

This path is required whenever:

- The output tensor uses a sharded memory config (e.g., `ttnn.MemoryConfig` with `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` or `WIDTH_SHARDED`).
- A specific core grid assignment is needed (large hidden dims at decode batch sizes).
- `activation_dtype` needs to differ from the default output dtype.

```python
import ttnn

program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=4,
    per_core_M=1,
    per_core_N=8,
    transpose_mcast=False,
    fused_activation=(ttnn.UnaryOpType.SILU, True),  # fused SiLU post-op
    activation_dtype=ttnn.bfloat16,
)

gate = ttnn.matmul(
    x,
    w1,
    program_config=program_config,
    memory_config=ttnn.L1_MEMORY_CONFIG,
    dtype=ttnn.bfloat16,
)
```

If a program config is passed and `fused_activation` is omitted from the config, no activation is fused — even if `activation="silu"` is also passed as a top-level keyword argument.

---

## Kernel-Level Fusion Semantics

Without fusion, a SiLU post-matmul sequence requires two separate kernel dispatches:

1. A `MatmulMultiCoreReuse*` kernel that computes FPU matrix multiply-accumulate and writes the result tile to L1.
2. A `ttnn_silu` SFPU kernel that reads that tile from L1, computes `x * sigmoid(x)`, and writes the result back to L1 (or DRAM).

With fusion, step 2 is folded into step 1. The SFPU activation pass is appended to the FPU tile loop inside the same compiled kernel binary. The tile never leaves the destination register file between the FPU accumulate and the SFPU sigmoid-multiply. There is no separate op launch and no extra L1 read/write for the intermediate matmul output.

The practical consequences are:

- **No extra L1 round-trip.** The activation tensor that would have occupied L1 between the two ops is never materialized as a standalone buffer.
- **No second kernel dispatch.** Kernel dispatch carries fixed host-to-device overhead regardless of tensor size; eliminating it matters most at small token counts where the matmul itself is fast.
- **Single Tracy CSV entry.** A fused op appears as one `MatmulMultiCoreReuse*` row in `ops_perf_results_*.csv`. If a separate `silu` row appears, fusion did not activate. See *See Chapter 6, `configuration_recommendations.md` for the verification checklist.*

---

## Fused vs. Unfused: Code Comparison

```python
import ttnn

# --- Fused: SiLU computed in same kernel as matmul ---
# One kernel dispatch. No intermediate activation tensor in L1.
gate = ttnn.matmul(x, w1, activation="silu")

# --- Unfused: separate kernel dispatch for SiLU ---
# Two kernel dispatches. Matmul output tensor materialized in L1 between ops.
gate_unfused = ttnn.matmul(x, w1)
gate = ttnn.silu(gate_unfused)
```

At decode batch sizes (1–16 tokens, hidden_dim=2048), the unfused path adds:

- One full L1 read + write of the activation tensor (size: `num_tokens * hidden_dim * 2` bytes for BF16).
- One additional kernel dispatch overhead on the host dispatch thread.

For `num_tokens=8`, `hidden_dim=2048`, BF16: the activation tensor is `8 * 2048 * 2 = 32,768 bytes` — 32 KB that must be written to L1 by the matmul kernel, then read back and rewritten by the SiLU kernel.

---

## `activation_dtype` in Program Configs

`MatmulMultiCoreReuseMultiCastProgramConfig` and `MatmulMultiCoreReuseMultiCast1DProgramConfig` expose an `activation_dtype` field that controls the precision of the tensor written out of the fused activation pass.

| `activation_dtype` | Tile size | L1 footprint (1024 elements) | Use case |
|---|---|---|---|
| `ttnn.bfloat16` | 2,048 bytes | 2,048 bytes | Default; full BF16 precision for downstream mul |
| `ttnn.bfloat8_b` | 1,024 bytes | 1,024 bytes | Half L1 footprint; small accuracy delta for SiLU output |

Setting `activation_dtype=ttnn.bfloat8_b` halves the L1 footprint of the fused activation output tile. For large hidden dims (d_ff=2048 as in DeepSeek-V3 and Qwen 235B-A22B, or d_ff=8192 in larger dense experts) with sharded layouts, this can allow tighter shard configurations that keep more tiles resident in L1 simultaneously.

The accuracy implications of `bfloat8_b` for the SiLU output are covered in [`activation_dtype_and_precision.md`](activation_dtype_and_precision.md).

---

For Wormhole B0 hardware constants and SiLU arithmetic intensity context, see Chapter 2 (`tensix_compute_engine.md`) and Chapter 4 (`roofline_analysis.md`).

---

## Next Steps

Continue to [`swiglu_fusion_pattern.md`](swiglu_fusion_pattern.md) for the concrete Pattern A implementation that applies fused SiLU to the gate_proj matmul in a SwiGLU FFN block.
