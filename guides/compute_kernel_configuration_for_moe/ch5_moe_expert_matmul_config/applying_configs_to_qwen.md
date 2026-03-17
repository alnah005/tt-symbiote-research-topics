# Applying Configs to Qwen MoE

## Overview

This file provides the step-by-step process for adding explicit `WormholeComputeKernelConfig` arguments to the Qwen MoE expert forward pass. It covers where to insert the configs in the code, how to validate correctness via PCC comparison, how to measure the latency effect, and two edge cases worth knowing: L1 interaction with the existing program config and prefill-mode behavior.

---

## Step 1: Define the Two Config Objects

Add the two canonical config objects at module level, near the top of the Qwen expert implementation file (mirroring `models/demos/deepseek_v3/tt/experts.py`).

See Chapter 1, `wormhole_compute_kernel_config_api.md` for the canonical `WormholeComputeKernelConfig` constructor reference.

---

## Step 2: Assign Configs to Projections

In the expert forward pass function, pass the appropriate config to each matmul call via the `compute_kernel_config` keyword argument:

```python
# gate projection (w1): uses LOFI
gate = ttnn.linear(
    hidden_states,
    w1,
    compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    ...
)

# up projection (w3): uses LOFI
up = ttnn.linear(
    hidden_states,
    w3,
    compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    ...
)

# SiLU and element-wise multiply (no compute_kernel_config needed)
hidden = ttnn.mul(ttnn.silu(gate), up)

# down projection (w2): uses HIFI2
out = ttnn.linear(
    hidden,
    w2,
    compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
    ...
)
```

The rule is: gate and up → LOFI; down → HIFI2. This mirrors the DeepSeek-V3 assignment exactly.

---

## Step 3: Validate with PCC Comparison

Before merging, run a PCC comparison between the new config and the unoptimized baseline (device defaults).

Expected PCC thresholds:

| Projection | Expected PCC |
|---|---|
| gate (w1) | > 0.999 |
| up (w3) | > 0.999 |
| down (w2) | > 0.999 |

The gate and up projections use the same `math_fidelity` (LoFi) as the device default; the only change is `packer_l1_acc=True`, which affects memory traffic but not the numerical result of the computation. PCC should be 1.0 or indistinguishable from 1.0 for those two projections.

The down projection changes fidelity from LoFi to HiFi2. This improves accuracy relative to the baseline; PCC of the new result against a high-precision reference will be higher, not lower. PCC of new vs. old baseline may be slightly below 1.0 (reflecting the fidelity improvement), which is expected and desirable.

---

## Step 4: Measure Latency

Use `ttnn.device.profiler` to isolate expert matmul time before and after the config change:

```python
import ttnn

ttnn.device.profiler.start()
out = ttnn.linear(hidden, w2, compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2, ...)
ttnn.device.profiler.stop()
report = ttnn.device.profiler.get_report()
```

Run this for each of the three projections separately. Compare pre/post latency for decode (b=1). The gate and up projections should show improvement from `packer_l1_acc=True` alone. The down projection will show improvement from both `packer_l1_acc=True` and the fidelity change (HiFi2 costs more FPU cycles than LoFi, but the DRAM savings dominate at decode).

---

## Interaction with `MatmulMultiCoreReuseMultiCastProgramConfig`

Qwen MoE expert matmuls may already use an explicit `program_config` (e.g., `MatmulMultiCoreReuseMultiCastProgramConfig`) to control grid layout, subblock sizes, and core tiling. The `compute_kernel_config` argument is orthogonal to `program_config`: both can be specified simultaneously on the same `ttnn.linear` or `ttnn.matmul` call.

```python
out = ttnn.linear(
    hidden,
    w2,
    program_config=existing_program_config,    # grid layout, tiling — unchanged
    compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,  # fidelity, packer — new
    ...
)
```

Adding `compute_kernel_config` does not require modifying or retuning the existing `program_config`. The two configs operate at different levels: `program_config` determines how the problem is partitioned across cores; `compute_kernel_config` determines how each core executes its assigned tiles.

The one interaction to check: `packer_l1_acc=True` increases L1 usage for the accumulation buffer. If the existing `program_config` already uses a large `per_core_N` (many output tiles per core), verify that the total L1 usage (weights + activations + accumulation buffer) does not exceed 1.5 MB/core. TTNN will raise an allocation error at dispatch time if it does, making the failure detectable before any numerical issue arises.

---

## Edge Case: Prefill Mode (M >= 512)

At prefill, M (the number of tokens) is large. Two things change:

1. **Latency difference narrows.** The arithmetic intensity is higher at prefill. The FPU is more often the bottleneck rather than memory bandwidth. The `packer_l1_acc` savings (which are a memory-bandwidth improvement) contribute a smaller fraction of total latency. The improvement is still non-negative but may be below measurement noise for very large M.

2. **PCC difference narrows.** At large M, the output tensor has many more elements. Statistical averaging means that LoFi rounding errors partially cancel across the M dimension. The PCC difference between LoFi and HiFi2 for the down projection will be smaller at prefill than at decode.

The same two configs (LOFI for gate/up, HIFI2 for down) are still the correct assignment at prefill. There is no reason to switch configs based on sequence length. The numerical argument for HiFi2 on the down projection applies regardless of M; the accuracy improvement is smaller at prefill but the cost is also smaller (fewer redundant DRAM reads relative to total compute).
