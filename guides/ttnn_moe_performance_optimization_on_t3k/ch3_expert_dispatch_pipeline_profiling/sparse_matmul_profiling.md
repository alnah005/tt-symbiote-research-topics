# Sparse Matmul Profiling

## Context

This file addresses:
- **Q2** — Which stage of the expert dispatch pipeline dominates latency at batch=1 decode?
- **Q3** — How should `sparse_matmul` program config parameters (`in0_block_w`, `per_core_M`, `math_fidelity`) be tuned for GLM-4-MoE and Bailing?

Source ranges: `moe.py:L62–L91` (`_make_sparse_matmul_program_config`), `moe.py:L1138–L1157` (program config construction in `move_weights_to_device_impl`), `moe.py:L1250–L1289` (Stages 4–5 of `TTNNExperts.forward`).

---

## The Three Sparse Matmul Calls

Stages 4 and 5 of `TTNNExperts.forward` contain three `sparse_matmul` calls:

| Call | Lines | Weight matrix | Input | Output | Sparsity type |
|---|---|---|---|---|---|
| w1 (gate proj) | L1250–L1259 | `w1_proj`: `(hidden_size, intermediate_size)` | `x` sparse blocks | `w1_out` | `is_input_b_sparse=True` |
| w3 (up proj) | L1260–L1269 | `w3_proj`: `(hidden_size, intermediate_size)` | `x` sparse blocks | `w3_out` | `is_input_b_sparse=True` |
| w2 (down proj) | L1280–L1289 | `w2_proj`: `(intermediate_size, hidden_size)` | `intermediate` sparse blocks | `expert_output` | `is_input_a_sparse=True` |

Between w1/w3 and w2, the silu-then-multiply activation runs at `moe.py:L1271–L1275`:

```python
# moe.py:L1271–1275
w1_activated = ttnn.silu(w1_out)
intermediate = ttnn.mul(w1_activated, w3_out)
```

This silu+mul is elementwise on `(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, intermediate_size)` tensors and should be fast relative to the matmuls, but it adds to Stage 4 total time.

### Input shapes at batch=1 decode (GLM-4-MoE)

After padding (32 tokens) and dispatch (only tokens assigned to the local device's experts are retained), the sparse layout produced at `moe.py:L1247–L1248` is:

```
(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, hidden_size)
= (1, num_sparse_blocks, 32, 4096)
```

At batch=1 decode, `num_sparse_blocks` per device is at most `ceil(active_tokens / SPARSITY_BLOCK_SIZE) = ceil(1 / 32) = 1`. Most devices have zero active tokens and skip the matmul entirely (the sparse kernel handles empty blocks without error). The device holding the 1–2 active experts processes one sparse block of 32 tokens.

For the w1 matmul on that device:

```
in0: (1, 1, 32, 4096)   [sparse block, bf16, TILE_LAYOUT]
in1: (1, 1, 4096, 1408) [weight, bf16, TILE_LAYOUT]
out: (1, 1, 32, 1408)
```

FLOP count: `32 × 4096 × 1408 × 2 = 369 098 752 FLOPs ≈ 369 MFLOPs`. At Wormhole peak matmul throughput (~150 TFLOPS bf16), this represents ~2.5 µs of pure compute. Actual time will be higher due to weight-loading and L1/DRAM overhead.

---

## _make_sparse_matmul_program_config Deep Dive (moe.py:L62–L91)

```python
def _make_sparse_matmul_program_config(device, out_features, in0_block_w, per_core_M, ...):
    grid = device.compute_with_storage_grid_size()
    core_x, core_y = grid.x, grid.y          # T3K: 8×8 = 64 cores
    n_tiles = (out_features + TILE_SIZE - 1) // TILE_SIZE
    num_cores = core_x * core_y              # 64
    per_core_N = max(1, ceil(n_tiles / num_cores))
    out_subblock_w = min(per_core_N, 4)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        out_block_w=per_core_N,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        fuse_batch=False,
        mcast_in0=True,
    )
```

### Derived values for GLM-4-MoE on T3K (64 cores)

**Gate/up projection (w1, w3): `out_features = intermediate_size = 1408`**

```
n_tiles    = ceil(1408 / 32) = 44
per_core_N = max(1, ceil(44 / 64)) = max(1, 1) = 1
out_subblock_w = min(1, 4) = 1
```

**Down projection (w2): `out_features = hidden_size = 4096`**

```
n_tiles    = ceil(4096 / 32) = 128
per_core_N = max(1, ceil(128 / 64)) = max(1, 2) = 2
out_subblock_w = min(2, 4) = 2
```

### What each parameter controls

**`in0_block_w`** — the number of contiguous tiles along the K-dimension (the shared reduction axis) that each core loads into L1 at once before performing its partial dot product accumulation. Set at `moe.py:L1148` and `L1153`:

```python
# moe.py:L1148–1153
in0_block_w=min(4, hidden_tiles),    # for gate_up: min(4, 128) = 4
in0_block_w=min(4, intermediate_tiles),  # for down: min(4, 44) = 4
```

- Larger `in0_block_w` → more K-tiles fetched per L1 load cycle → higher arithmetic intensity, fewer DRAM round-trips during the reduction sweep.
- The cap of 4 is an L1 capacity constraint: loading more than 4 K-tiles simultaneously would overflow the per-core L1 budget given the activation tile size.
- At `in0_block_w=4`, each core processes 4 × 32 = 128 columns of the activation matrix in one L1 load, then accumulates partial sums across all such blocks in sequence.
- The K-loop therefore executes `K_tiles / in0_block_w = 128 / 4 = 32` outer iterations for the gate/up matmul (GLM-4-MoE). Reducing `in0_block_w` to 2 would double the K-loop iterations and increase DRAM pressure.

**`per_core_M`** — the number of output row-tiles (M-tiles) each core computes. Hardcoded to 1 at `moe.py:L1149` and `L1154`.

- `per_core_M=1` means each core owns exactly one 32-row tile of the output. For the batch=1 decode case with 32 padded tokens, there is exactly 1 M-tile in the activation matrix: the program naturally assigns all cores the same single M-tile.
- If `per_core_M > 1`, each core would need to load multiple M-tiles from the activation into L1 before beginning accumulation. With only 1 physical M-tile available at batch=1, `per_core_M=1` is the correct value; raising it would waste L1 allocation without improving occupancy.

**`per_core_N`** — the number of output column-tiles each core computes. Derived from `out_features` and `num_cores`.

- For gate/up: `per_core_N=1` → each core handles one tile of 32 output columns → 64 cores × 32 = 2048 output columns. But `intermediate_size=1408 = 44 tiles`, so only 44 of the 64 cores are active. The remaining 20 cores are idle. This underutilization is a direct consequence of `intermediate_size=1408` not being a multiple of 64 tiles.
- For down: `per_core_N=2` → each core handles two tiles of 32 output columns → 64 cores × 64 = 4096 output columns, matching `hidden_size=4096` exactly. All 64 cores are active for the down projection.

**`mcast_in0=True`** — the activation matrix (in0) is multicast from a source core to all destination cores rather than each core loading independently. This is the `MultiCast1D` aspect of the config. At `per_core_M=1`, all cores need the same activation tile row; multicast reduces redundant DRAM loads from 64 individual reads to a single broadcast.

**`fuse_batch=False`** — batch and M dimensions are not fused; each sparse block in the layout `(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, hidden_size)` is treated independently.

---

## Sweep Methodology

### Objective

Identify the `in0_block_w` value (and optionally `per_core_M`) that minimizes Stage 4+5 latency for each model configuration. Separately measure the accuracy impact of `HiFi2` vs `HiFi4` math fidelity.

### Candidate values

**`in0_block_w` sweep:**

For GLM-4-MoE gate/up (`K_tiles=128`): valid divisors of 128 that fit in L1 are `{1, 2, 4}`. The current value is 4.

For GLM-4-MoE down (`K_tiles=44`): valid divisors of 44 that fit in L1 are `{1, 2, 4}`. The current value is 4.

> Note: `in0_block_w` must divide `K_tiles` evenly for the K-loop to be clean. Check divisibility before submitting a config. If `K_tiles=44`, then `in0_block_w=4` is valid (44/4=11 iterations), `in0_block_w=2` is valid (44/2=22 iterations), `in0_block_w=1` is valid (44/1=44 iterations).

**`per_core_M` sweep:**

At batch=1 decode (1 M-tile total), only `per_core_M=1` is valid; the matrix has exactly 1 M-tile and cannot be split further. Sweep `per_core_M` only at larger batch sizes.

**`math_fidelity` sweep:**

Values: `{ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4}`. Current setting is `HiFi2` (`moe.py:L1156`).

### Sweep harness (GLM-4-MoE example)

```python
import ttnn
import torch
import time

TILE = 32
hidden_size = 4096
intermediate_size = 1408
padded_tokens = 32  # SPARSITY_BLOCK_SIZE

# Build synthetic sparse-block inputs
x_block = ttnn.from_torch(
    torch.randn(1, 1, padded_tokens, hidden_size, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
w1_weight = ttnn.from_torch(
    torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

results = []
for in0_block_w in [1, 2, 4]:
    for fidelity in [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]:
        grid = device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y  # 64

        n_tiles = (intermediate_size + TILE - 1) // TILE  # 44
        per_core_N = max(1, -(-n_tiles // num_cores))     # ceil div = 1

        prog_cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=1,
            per_core_N=per_core_N,
            out_block_w=per_core_N,
            out_subblock_h=1,
            out_subblock_w=min(per_core_N, 4),
            fuse_batch=False,
            mcast_in0=True,
        )
        compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Warmup
        for _ in range(20):
            out = ttnn.matmul(x_block, w1_weight, program_config=prog_cfg,
                              compute_kernel_config=compute_cfg)
            ttnn.synchronize_device(device)

        # Timed
        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            out = ttnn.matmul(x_block, w1_weight, program_config=prog_cfg,
                              compute_kernel_config=compute_cfg)
            ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        mean_us = (t1 - t0) / N * 1e6

        results.append({
            "in0_block_w": in0_block_w,
            "fidelity": fidelity.name,
            "mean_us": round(mean_us, 1),
        })
        print(f"in0_block_w={in0_block_w}, fidelity={fidelity.name}: {mean_us:.1f} µs")
```

> This harness uses `ttnn.matmul` directly rather than the `sparse_matmul` wrapper to allow isolated parameter sweeps. To measure the actual sparse kernel, substitute the `sparse_matmul` call with the same `program_config` and `compute_kernel_config` arguments.

### Expected results table (fill in after measurement)

**GLM-4-MoE w1/w3 gate-up projection (in0 = 32×4096, out = 32×1408):**

| `in0_block_w` | `math_fidelity` | w1 latency (µs) | w3 latency (µs) | w1+w3 total (µs) |
|---|---|---|---|---|
| 4 | HiFi2 | ___ | ___ | ___ |
| 4 | HiFi4 | ___ | ___ | ___ |
| 2 | HiFi2 | ___ | ___ | ___ |
| 2 | HiFi4 | ___ | ___ | ___ |
| 1 | HiFi2 | ___ | ___ | ___ |

**GLM-4-MoE w2 down projection (in0 = 32×1408, out = 32×4096):**

| `in0_block_w` | `math_fidelity` | w2 latency (µs) |
|---|---|---|
| 4 | HiFi2 | ___ |
| 4 | HiFi4 | ___ |
| 2 | HiFi2 | ___ |
| 2 | HiFi4 | ___ |

**Expected pattern:** Decreasing `in0_block_w` from 4 to 2 is expected to increase latency by 10–25% due to more DRAM round-trips for the K-loop. The HiFi4 vs HiFi2 delta is expected to be 5–15% in compute-bound conditions; at batch=1 (memory-bound), the fidelity delta may be smaller.

---

## HiFi2 vs HiFi4 Accuracy Measurement

The current setting `math_fidelity=ttnn.MathFidelity.HiFi2` (`moe.py:L1156`) trades numerical precision for speed. HiFi2 uses a reduced-mantissa accumulation path; HiFi4 uses the full bf16 mantissa precision.

### When HiFi2 is safe

For expert FFN layers (the w1/w3/w2 matmuls), the activations are already quantized to bf16. The downstream operation is `silu` + multiply + second matmul — none of which require fp32-level precision at inference time. HiFi2 is generally safe for MoE expert layers when:

- The model was trained without explicit fp32 accumulation in these layers.
- End-to-end accuracy benchmarks (e.g., MMLU, GSM8K) show no degradation vs a reference implementation.

### How to measure the accuracy gap

```python
import torch

def run_matmul_with_fidelity(x_torch, w_torch, fidelity, device):
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w_tt = ttnn.from_torch(w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out = ttnn.matmul(x_tt, w_tt, compute_kernel_config=cfg)
    return ttnn.to_torch(out).float()

x = torch.randn(32, 4096)
w = torch.randn(4096, 1408)
ref = (x.float() @ w.float())  # fp32 reference

out_hifi2 = run_matmul_with_fidelity(x, w, ttnn.MathFidelity.HiFi2, device)
out_hifi4 = run_matmul_with_fidelity(x, w, ttnn.MathFidelity.HiFi4, device)

cos_hifi2 = torch.nn.functional.cosine_similarity(
    ref.flatten(), out_hifi2.flatten(), dim=0
).item()
cos_hifi4 = torch.nn.functional.cosine_similarity(
    ref.flatten(), out_hifi4.flatten(), dim=0
).item()

print(f"HiFi2 cosine similarity vs fp32: {cos_hifi2:.6f}")
print(f"HiFi4 cosine similarity vs fp32: {cos_hifi4:.6f}")

max_abs_hifi2 = (ref - out_hifi2).abs().max().item()
max_abs_hifi4 = (ref - out_hifi4).abs().max().item()
print(f"HiFi2 max abs error: {max_abs_hifi2:.4f}")
print(f"HiFi4 max abs error: {max_abs_hifi4:.4f}")
```

**Interpret results as follows:**

| cosine similarity | Interpretation |
|---|---|
| > 0.9999 | Numerically negligible difference; HiFi2 is safe |
| 0.999–0.9999 | Small but measurable; run end-to-end accuracy test before committing |
| < 0.999 | Meaningful precision loss; HiFi4 is required for this weight shape |

If HiFi2 passes the cosine similarity check for both gate-up and down projections, the performance advantage (expected 5–15% speedup) justifies retaining it. If HiFi2 fails and HiFi4 is required, document the latency penalty in the bottleneck summary.

---

## Core Utilization Analysis

### Gate/up projection: partial core utilization

For GLM-4-MoE gate/up matmul: `n_tiles=44`, `num_cores=64`, `per_core_N=1`. Only `44` of `64` cores are active; `20` cores are idle. This represents 31% underutilization.

**Root cause:** `intermediate_size=1408` is not a multiple of `num_cores × TILE_SIZE = 64 × 32 = 2048`. The nearest multiple is 2048 (intermediate_size would need to be 2048 for 100% utilization, which would require model re-training).

**Implication:** The effective throughput of the gate/up matmul is `44/64 = 68.75%` of peak. This is a structural inefficiency inherent to the model's intermediate dimension and cannot be fixed by tuning the program config. It should be noted in the bottleneck summary as a fixed overhead.

### Down projection: full core utilization

For GLM-4-MoE down matmul: `n_tiles=128`, `num_cores=64`, `per_core_N=2`. All 64 cores are active. The down projection achieves full core utilization, and its per-core workload (2 output tile columns each) is slightly higher than the gate/up workload (1 tile column each). Expect the down projection to have higher absolute latency than a single gate or up projection, even though it runs at full efficiency.

---

**Next:** [`weight_application_overhead.md`](./weight_application_overhead.md)
