# SwiGLU Fusion Pattern

This document describes the fusion challenge specific to SwiGLU, defines Pattern A (recommended), Pattern B (unfused baseline), and Pattern C (custom kernel), and specifies the memory config requirements for Pattern A to be valid.

---

SwiGLU computes `silu(gate_proj(x)) * up_proj(x)` (see Chapter 1, `swiglu_variant.md` for the full definition); only the `gate_proj` SiLU is fusible in a single `ttnn.matmul` call.

---

## The Fusion Challenge

TTNN's `activation` parameter on `ttnn.matmul` fuses an activation function into a single matmul's output. This covers exactly one operation: the SiLU applied to the gate_proj output.

The up_proj matmul has no activation — it cannot benefit from activation fusion. The element-wise multiply (`ttnn.mul`) that combines gate and up outputs is a separate operation that has no TTNN fusion path at the high-level API layer.

**What is fusible:** SiLU into gate_proj matmul.

**What is not fusible at the TTNN API level:** the element-wise multiply of gate and up outputs; the up_proj matmul itself.

A fully fused single-kernel SwiGLU (gate_proj + silu + up_proj + mul in one kernel) is only achievable through a custom Metalium kernel (Pattern C), which is outside the scope of this guide.

---

## Pattern A: Recommended — Fuse SiLU into gate_proj

Pattern A fuses the SiLU activation into the gate_proj matmul and issues the up_proj matmul and element-wise multiply as separate ops.

```python
import ttnn

# Pattern A: fuse SiLU into gate matmul
# gate_proj matmul + SiLU in one kernel dispatch
gate = ttnn.matmul(x, w1, activation="silu")

# up_proj matmul: no activation, separate kernel dispatch
up = ttnn.matmul(x, w3)

# Element-wise multiply: combines silu(gate_proj(x)) * up_proj(x)
hidden = ttnn.mul(gate, up)
```

**What Pattern A eliminates compared to Pattern B:**

- One separate `ttnn.silu` kernel dispatch.
- One L1 round-trip: the intermediate gate_proj output tensor is never materialized separately before SiLU is applied.

At `num_tokens=8`, `d_ff=2048`, BF16, this means 32 KB of L1 traffic is eliminated: the 8 * 2048 * 2 = 32,768-byte activation tensor does not need to be written by the matmul kernel and read back by the SiLU kernel.

---

## Pattern B: No Fusion — Baseline

Pattern B is the unfused three-op sequence. It is the correct baseline for measuring the latency improvement that Pattern A provides.

```python
import ttnn

# Pattern B: no fusion
gate_proj_out = ttnn.matmul(x, w1)     # matmul only, no activation
gate = ttnn.silu(gate_proj_out)         # separate SFPU kernel dispatch
up = ttnn.matmul(x, w3)
hidden = ttnn.mul(gate, up)
```

Pattern B uses four kernel dispatches where Pattern A uses three. The Tracy profiler CSV will show a distinct `silu` row between the two `MatmulMultiCoreReuse*` rows.

---

## Pattern C: Custom Metalium Kernel (Not Covered Here)

Pattern C is a custom tt-metal kernel that computes the full SwiGLU — gate_proj, SiLU, up_proj, and element-wise multiply — in a single kernel dispatch. This eliminates all intermediate tensor allocations and all but one kernel launch overhead.

Implementing Pattern C requires authoring a custom Metalium compute kernel in C++ using SFPU and FPU LLK primitives. It is not accessible through the TTNN Python API.

Pattern C is noted here for completeness. *See the `moe_optimization_techniques_for_ttnn` guide for discussion of custom kernel authoring when Pattern A overhead is insufficient.*

---

## Practical Recommendation

**Use Pattern A as the default for all SwiGLU FFN blocks in production MoE inference on Wormhole B0.**

Pattern A is implementable entirely through the TTNN Python API with no Metalium kernel authoring. It eliminates the highest-value inefficiency — the extra SiLU kernel dispatch and L1 round-trip — while leaving the up_proj matmul and element-wise multiply unchanged.

Pattern B is acceptable only as a measurement baseline or in prefill workloads (128+ tokens) where SiLU latency is already below 5% of total FFN time. *See Chapter 4, `compute_vs_memory_bound_regimes.md` for the token count thresholds.*

---

## Memory Config Requirements for Pattern A

For Pattern A to be valid, the following memory config constraints must hold:

**Output of the fused gate_proj matmul (the `gate` tensor) must be readable by `ttnn.mul`.**

The output memory config of the fused `ttnn.matmul` call must match the input memory config expected by `ttnn.mul`. In practice this means one of two configurations:

| Configuration | gate tensor memory config | up tensor memory config | Validity |
|---|---|---|---|
| Both in DRAM | `ttnn.DRAM_MEMORY_CONFIG` | `ttnn.DRAM_MEMORY_CONFIG` | Always valid |
| Both height-sharded on the same core grid | `ttnn.MemoryConfig(HEIGHT_SHARDED, ...)` | `ttnn.MemoryConfig(HEIGHT_SHARDED, ...)` | Valid when shard specs match |
| gate in L1 interleaved, up in DRAM | Mixed | — | Requires explicit format conversion before `ttnn.mul`; avoid |

When using sharded outputs, the `fused_activation` field must be set inside the program config rather than using the top-level `activation` parameter. *See [`ttnn_fused_activation_api.md`](ttnn_fused_activation_api.md) for the program config path.*

```python
import ttnn

# Pattern A with sharded output (program config path required)
shard_spec = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
    [num_tokens, d_ff // 8],
    ttnn.ShardOrientation.ROW_MAJOR,
    False,
)
shard_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    shard_spec,
)

program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(8, 1),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=4,
    per_core_M=num_tokens,
    per_core_N=d_ff // 8 // 32,
    fuse_batch=True,
    fused_activation=(ttnn.UnaryOpType.SILU, True),
    mcast_in0=False,
)

gate = ttnn.matmul(
    x,
    w1,
    program_config=program_config,
    memory_config=shard_mem_config,
    dtype=ttnn.bfloat16,
)

# up_proj must use matching shard spec for ttnn.mul to work without reformat
up = ttnn.matmul(
    x,
    w3,
    program_config=program_config_up,   # separate config without fused_activation
    memory_config=shard_mem_config,
    dtype=ttnn.bfloat16,
)

hidden = ttnn.mul(gate, up)
```

**The element-wise multiply requires both input tensors to have compatible memory configs.** If `gate` is sharded and `up` is interleaved, TTNN will either raise an error or silently insert a format conversion. Verify compatibility by checking the Tracy CSV for unexpected `ttnn_interleaved_to_sharded` or `ttnn_sharded_to_interleaved` entries between the matmul and mul ops.

---

## Model Applicability

| Model | gate_proj activation | Pattern A applicable |
|---|---|---|
| Llama 3 (dense FFN) | SiLU | Yes |
| Mixtral 8x7B | SiLU | Yes |
| Qwen2-MoE / Qwen 235B-A22B | SiLU | Yes (d_ff=2048 per expert) |
| DeepSeek-V3 / DeepSeek-MoE | SiLU | Yes (d_ff=2048 per expert) |

For DeepSeek-V3 and Qwen 235B-A22B: `d_model=7168`, `d_ff=2048` per expert. At these dimensions with `num_tokens=8`, Pattern A eliminates a 32 KB L1 round-trip per expert per forward pass.

---

## Next Steps

Continue to [`activation_dtype_and_precision.md`](activation_dtype_and_precision.md) for guidance on choosing between `bfloat16` and `bfloat8_b` as the `activation_dtype` for the fused SiLU output.
