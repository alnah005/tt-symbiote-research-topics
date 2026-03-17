# Code Patterns: DRAM-Sharded Expert Weight Implementation

## Overview

This file provides complete, annotated code patterns for every step of the DRAM-sharded expert weight workflow: loading weights from a checkpoint, constructing shard configurations, resharding into DRAM-sharded layout, and integrating with `ttnn.matmul`. The canonical example throughout is Mixtral 8x7B (`d_model=4096`, `d_ff=14336`, `num_experts=8`, `top_k=2`).

All shard configurations satisfy the five tile-alignment rules from Chapter 5. Key points of compliance are called out inline.

---

## Device and Program Cache Setup

All code in this chapter assumes a device has been initialized and program caching is enabled. Static tensor shapes are required for program caching — changing any shape between inference calls forces recompilation.

```python
import ttnn
import torch

# Initialize device with program cache enabled.
# This must be done once per process lifetime before any tensor operations.
device = ttnn.open_device(device_id=0)
ttnn.enable_program_cache(device)  # Required for stable per-call latency.
```

> **Warning:** Program cache keys include the `MemoryConfig` of all input tensors. If you toggle between `DRAM_MEMORY_CONFIG` and a DRAM-sharded config during the same process, each distinct config will compile a separate kernel. Always commit to one weight layout before inference begins to avoid repeated recompilation overhead.

---

## Step 1: Loading Expert Weights from a Checkpoint

Expert weights are loaded from a checkpoint as PyTorch tensors on CPU, then transferred to device in `TILE_LAYOUT` with the default interleaved DRAM placement. DRAM-sharding is applied after the initial transfer — this keeps the loading logic simple and the resharding step explicit.

```python
# Model hyperparameters for Mixtral 8x7B.
d_model = 4096
d_ff = 14336
num_experts = 8

# --- CPU-side weight loading (from any checkpoint format) ---
cpu_gate = [torch.randn(d_model, d_ff, dtype=torch.bfloat16) for _ in range(num_experts)]
cpu_up   = [torch.randn(d_model, d_ff, dtype=torch.bfloat16) for _ in range(num_experts)]
cpu_down = [torch.randn(d_ff, d_model, dtype=torch.bfloat16) for _ in range(num_experts)]

# --- Transfer to device with interleaved DRAM placement (one projection shown; repeat for all) ---
# TILE_LAYOUT is required before any sharding operation.
# See Chapter 6, `shard_setup_overhead.md` for the full load function (Option A),
# including the warning against inference-time resharding.
gate_weights = [
    ttnn.from_torch(cpu_gate[i], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for i in range(num_experts)
]
up_weights   = [ttnn.from_torch(cpu_up[i],   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for i in range(num_experts)]
down_weights = [ttnn.from_torch(cpu_down[i], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for i in range(num_experts)]
```

> **Tip:** For `bfloat8_b` weights, set `dtype=ttnn.bfloat8_b` in `ttnn.from_torch`. The shard shapes and alignment rules are identical to bfloat16 in element counts; only the byte sizes change (1 byte/element vs 2 bytes/element). Verify that `shard_shape[0] * shard_shape[1] * 1 % 32 == 0` for Rule 5 compliance.

---

## Step 2: Constructing the Shard Configuration

The shard configuration is derived programmatically from the model hyperparameters. This avoids hardcoding magic numbers and ensures alignment rules are checked at config-build time.

The Wormhole B0 DRAM controller count is 6. Targeting a 1×6 shard grid (one row of 6 DRAM controller columns) is the natural choice for 1D sharding of expert weight matrices.

```python
# DRAM controller count on Wormhole B0.
NUM_DRAM_BANKS = 6

def make_gate_up_shard_config(d_model: int, d_ff: int, num_banks: int) -> ttnn.MemoryConfig:
    """
    Construct a WIDTH_SHARDED DRAM MemoryConfig for gate/up projection weights.

    Gate and up projections have shape [d_model, d_ff].
    We shard along the d_ff (width) dimension.

    See Chapter 2, `constructing_dram_sharded_config.md` for the ShardSpec
    construction pattern (Steps 1–2); Chapter 5 Rules 1–5 for all four
    alignment assertions (d_ff % num_banks, shard_h % 32, shard_w % 32,
    shard_bytes % 32).
    """
    shard_h = d_model
    shard_w = d_ff // num_banks

    # Ch07-original: assert alignment at config-build time so violations are
    # caught before the tensor is allocated (not at inference-time).
    assert d_ff % num_banks == 0, (
        f"d_ff={d_ff} must be divisible by num_banks={num_banks} for WIDTH_SHARDED."
    )
    assert shard_h % 32 == 0, f"shard_h={shard_h} must be a multiple of 32."
    assert shard_w % 32 == 0, f"shard_w={shard_w} must be a multiple of 32."
    shard_bytes = shard_h * shard_w * 2
    assert shard_bytes % 32 == 0, f"shard_bytes={shard_bytes} must be a multiple of 32."

    shard_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))
    })
    shard_spec = ttnn.ShardSpec(shard_grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=shard_spec,
    )


def make_down_shard_config(d_ff: int, d_model: int, num_banks: int) -> ttnn.MemoryConfig:
    """
    Construct a HEIGHT_SHARDED DRAM MemoryConfig for down projection weights.

    Down projection has shape [d_ff, d_model].
    We shard along the d_ff (height) dimension to align shard boundaries with
    the K-dimension of the activation matmul.

    See Chapter 2, `constructing_dram_sharded_config.md` for the ShardSpec
    construction pattern; Chapter 5 Rules 1–5 for alignment requirements.
    """
    shard_h = d_ff // num_banks
    shard_w = d_model

    # Ch07-original: assert alignment at config-build time.
    assert d_ff % num_banks == 0, (
        f"d_ff={d_ff} must be divisible by num_banks={num_banks} for HEIGHT_SHARDED."
    )
    assert shard_h % 32 == 0, f"shard_h={shard_h} must be a multiple of 32."
    assert shard_w % 32 == 0, f"shard_w={shard_w} must be a multiple of 32."
    shard_bytes = shard_h * shard_w * 2
    assert shard_bytes % 32 == 0, f"shard_bytes={shard_bytes} must be a multiple of 32."

    shard_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))
    })
    shard_spec = ttnn.ShardSpec(shard_grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=shard_spec,
    )


# Build configs for Mixtral 8x7B.
gate_up_config = make_gate_up_shard_config(d_model=4096, d_ff=14336, num_banks=NUM_DRAM_BANKS)
down_config    = make_down_shard_config(d_ff=14336, d_model=4096, num_banks=NUM_DRAM_BANKS)
```

For Mixtral 8x7B: `d_ff=14336`, `num_banks=6`. Check: `14336 / 6 = 2389.33` — this does **not** divide evenly. The correct approach is to choose `num_banks` that divides `d_ff`. For Mixtral: `14336 = 2^11 × 7`, so `num_banks=8` works (`14336 / 8 = 1792`, which is `56 × 32`, satisfying Rule 2). Adjust to `num_banks=8` for Mixtral:

```python
# Mixtral 8x7B: d_ff=14336 is not divisible by 6.
# Use 8 banks instead (14336 / 8 = 1792 = 56 × 32 — valid).
gate_up_config_mixtral = make_gate_up_shard_config(d_model=4096, d_ff=14336, num_banks=8)
down_config_mixtral    = make_down_shard_config(d_ff=14336, d_model=4096, num_banks=8)

# Qwen 235B-A22B: d_ff=2048, d_model=7168.
# 2048 / 4 = 512 = 16 × 32 — valid with 4 banks.
# 7168 / 4 = 1792 = 56 × 32 — valid.
gate_up_config_qwen = make_gate_up_shard_config(d_model=7168, d_ff=2048, num_banks=4)
down_config_qwen    = make_down_shard_config(d_ff=2048, d_model=7168, num_banks=4)
```

> **Warning:** Always verify that `d_ff % num_banks == 0` and `(d_ff // num_banks) % 32 == 0` before constructing `ShardSpec`. Neither condition is automatically checked by TTNN at `MemoryConfig` construction time; a violation will produce a runtime error or silent misalignment when the shard is used in a matmul. The `assert` statements in the helper functions above catch this at config-build time.

---

## Step 3: Converting to DRAM-Sharded Layout

With valid configs constructed, `ttnn.to_memory_config` converts each interleaved weight tensor to the sharded layout. This is called once at model load time.

```python
def reshard_expert_weights(
    gate_list, up_list, down_list,
    gate_up_cfg, down_cfg,
):
    """
    Convert all expert weight tensors from interleaved DRAM to DRAM-sharded layout.
    Returns new lists; original tensors are deallocated.
    """
    sharded_gate, sharded_up, sharded_down = [], [], []

    for i in range(len(gate_list)):
        # to_memory_config reallocates the tensor with the new memory layout.
        # The original interleaved allocation is freed after the call.
        sharded_gate.append(ttnn.to_memory_config(gate_list[i], gate_up_cfg))
        sharded_up.append(ttnn.to_memory_config(up_list[i], gate_up_cfg))
        sharded_down.append(ttnn.to_memory_config(down_list[i], down_cfg))

    return sharded_gate, sharded_up, sharded_down


# Execute once at load time — not inside the inference loop.
sharded_gate, sharded_up, sharded_down = reshard_expert_weights(
    gate_weights, up_weights, down_weights,
    gate_up_config_mixtral, down_config_mixtral,
)

# Verify the resulting memory config.
cfg = sharded_gate[0].memory_config()
assert cfg.buffer_type == ttnn.BufferType.DRAM
assert cfg.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
print(f"Shard shape: {sharded_gate[0].shard_spec().shape}")  # Expected: [4096, 1792]
```

---

## Step 4: Integrating with `ttnn.matmul`

### Direct DRAM-sharded weight input

For decode-regime token dispatch (small M), the matmul kernel streams weight tiles directly from DRAM-sharded locations via double-buffered DMA. No explicit DRAM→L1 reshard is needed. The activation tensor is kept in L1-interleaved or L1-sharded layout.

```python
def expert_ffn_forward(
    hidden_states: ttnn.Tensor,    # Shape [batch, seq_len, d_model], L1 or DRAM.
    gate_w: ttnn.Tensor,           # Shape [d_model, d_ff], DRAM-sharded WIDTH.
    up_w: ttnn.Tensor,             # Shape [d_model, d_ff], DRAM-sharded WIDTH.
    down_w: ttnn.Tensor,           # Shape [d_ff, d_model], DRAM-sharded HEIGHT.
    compute_kernel_config: ttnn.WormholeComputeKernelConfig,
) -> ttnn.Tensor:
    """
    SwiGLU expert FFN: output = (gate(x) * silu(up(x))) @ down_w
    Uses DRAM-sharded weight tensors throughout.
    """
    # Gate and up projections: [batch, seq_len, d_model] × [d_model, d_ff]
    gate_out = ttnn.matmul(
        hidden_states,
        gate_w,
        memory_config=ttnn.L1_MEMORY_CONFIG,   # Output lands in L1.
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    up_out = ttnn.matmul(
        hidden_states,
        up_w,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )

    # SwiGLU activation: element-wise silu(gate) * up.
    # ttnn.silu applies sigmoid linear unit in-place.
    gate_act = ttnn.silu(gate_out)
    ttnn.deallocate(gate_out)
    hidden_ffn = ttnn.mul(gate_act, up_out)
    ttnn.deallocate(gate_act)
    ttnn.deallocate(up_out)

    # Down projection: [batch, seq_len, d_ff] × [d_ff, d_model]
    output = ttnn.matmul(
        hidden_ffn,
        down_w,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(hidden_ffn)
    return output
```

### LOFI and HIFI2 compute kernel configs

Chapter 6 references two canonical compute kernel configurations. Use `LOFI` for maximum throughput in decode (slight precision reduction), `HIFI2` when higher accuracy is required.

```python
# LOFI: fastest; acceptable for decode regime with bfloat16 weights.
# math_fidelity=LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True.
lofi_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# HIFI2: higher accuracy; use when PCC threshold must exceed 0.9999 strictly.
# math_fidelity=HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True.
hifi2_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

### Fallback: explicit DRAM→L1 reshard before matmul

For prefill regime or when L1 pressure prevents direct DRAM streaming, stage the weight shard in L1 explicitly:

```python
# Build the L1 counterpart of the gate/up shard config.
# Only the active shard (1 expert's weight) is moved to L1 per forward pass.
l1_gate_up_config = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        [4096, 1792],                   # Same shape as DRAM shard.
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)

def expert_ffn_forward_with_l1_stage(hidden_states, gate_w_dram, up_w_dram, down_w_dram, ckc):
    """Variant that explicitly stages weights in L1 before matmul."""
    gate_w_l1 = ttnn.to_memory_config(gate_w_dram, l1_gate_up_config)
    gate_out   = ttnn.matmul(hidden_states, gate_w_l1, memory_config=ttnn.L1_MEMORY_CONFIG,
                             compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    ttnn.deallocate(gate_w_l1)

    up_w_l1  = ttnn.to_memory_config(up_w_dram, l1_gate_up_config)
    up_out   = ttnn.matmul(hidden_states, up_w_l1, memory_config=ttnn.L1_MEMORY_CONFIG,
                           compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    ttnn.deallocate(up_w_l1)

    gate_act  = ttnn.silu(gate_out)
    ttnn.deallocate(gate_out)
    hidden_ffn = ttnn.mul(gate_act, up_out)
    ttnn.deallocate(gate_act)
    ttnn.deallocate(up_out)

    down_w_l1 = ttnn.to_memory_config(down_w_dram, l1_gate_up_config)  # Use height-sharded config for down.
    output    = ttnn.matmul(hidden_ffn, down_w_l1, memory_config=ttnn.L1_MEMORY_CONFIG,
                            compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    ttnn.deallocate(down_w_l1)
    ttnn.deallocate(hidden_ffn)
    return output
```

> **Tip:** Prefer the direct DRAM-sharded path for decode (small M). Use the L1-staging path only if `ttnn.device.EnableMemoryReports()` shows L1 allocation failures during the direct path. The staging path adds a DRAM→L1 copy inside the inference loop, which is proportional to shard size, not full weight size.

---

## Next Steps

Proceed to `correctness_verification.md` to verify that the resharded weights produce numerically correct outputs by comparing PCC between interleaved and DRAM-sharded inference runs. Only after PCC is confirmed should you proceed to `benchmark_methodology.md` to measure performance.
