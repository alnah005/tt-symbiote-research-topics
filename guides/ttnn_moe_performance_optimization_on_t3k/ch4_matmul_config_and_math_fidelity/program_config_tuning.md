# Program Config Tuning

## Context

This file addresses **Q3**: Are `in0_block_w=min(4, hidden_tiles)` and `per_core_M=1` optimal for GLM-4-MoE and Bailing expert matmuls on T3K?

Source ranges: `moe.py:L62–L91` (`_make_sparse_matmul_program_config`), `moe.py:L1138–L1157` (config construction in `move_weights_to_device_impl`).

---

## `MatmulMultiCoreReuseMultiCast1DProgramConfig` Field Reference

The function `_make_sparse_matmul_program_config` (`moe.py:L62–L91`) is the sole factory for all three expert program configs. It returns a `MatmulMultiCoreReuseMultiCast1DProgramConfig` instance:

```python
# moe.py:L62–L91
def _make_sparse_matmul_program_config(device, out_features, in0_block_w, out_subblock_h=1, out_subblock_w=None, per_core_M=1):
    grid = device.compute_with_storage_grid_size()
    core_x = int(getattr(grid, "x"))
    core_y = int(getattr(grid, "y"))
    n_tiles = (int(out_features) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    num_cores = max(1, core_x * core_y)
    per_core_N = max(1, int(math.ceil(n_tiles / num_cores)))
    out_block_w = per_core_N
    if out_subblock_w is None:
        out_subblock_w = min(per_core_N, 4)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=int(in0_block_w),
        out_subblock_h=int(out_subblock_h),
        out_subblock_w=int(out_subblock_w),
        out_block_h=1,
        out_block_w=int(out_block_w),
        per_core_M=int(per_core_M),
        per_core_N=int(per_core_N),
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
```

### What each field controls in the Wormhole kernel

**`in0_block_w`** — Number of K-direction tiles (columns of the activation matrix in0, rows of the weight matrix in1) that each core loads into its L1 buffer in one outer-loop iteration before performing partial dot-product accumulation.

- Physically: each L1 load fetches `in0_block_w × TILE_SIZE = in0_block_w × 32` columns from the activation.
- With `mcast_in0=True`, the source core pushes these tiles to all destination cores simultaneously; the broadcast is counted once against DRAM bandwidth regardless of core count.
- The K-reduction loop executes `K_tiles / in0_block_w` outer iterations. At `in0_block_w=4`, the loop runs `K_tiles/4` times; at `in0_block_w=2`, it runs `K_tiles/2` times — doubling the loop overhead and the number of activation DMA requests.
- **Constraint:** `in0_block_w` must evenly divide `K_tiles`. If it does not, the kernel will either assert or produce incorrect output. See the valid-values derivation below.
- **L1 budget constraint:** each core must hold `in0_block_w` activation tiles plus `per_core_N` weight tiles plus `per_core_M × per_core_N` output tiles simultaneously. At `per_core_M=1`, `per_core_N=1` (gate/up), `in0_block_w=4`, the L1 occupancy is `4 + 1 + 1 = 6` tiles of bf16 = `6 × 32 × 32 × 2 = 12 288` bytes — well within the 1 MB per-core L1 budget. Values up to `in0_block_w=8` are feasible for GLM-4-MoE without overflowing L1, but only divisors of `K_tiles` are valid.

**`per_core_M`** — Number of M-tiles (row-tiles of the output, or row-tiles of the activation) each core computes. Each core must hold `per_core_M` full rows of activation tiles in L1.

- At batch=1 decode, after `SPARSITY_BLOCK_SIZE=32` padding, the sparse block has shape `(1, num_sparse_blocks, 32, hidden_size)`. The M-dimension in tile units is `32 / TILE_SIZE = 1`. There is exactly one M-tile; `per_core_M` cannot exceed 1.
- `per_core_M > 1` is only valid — and potentially beneficial — at larger batch sizes where `M_tiles > 1`. For example, at batch=4 after padding, `M_tiles = 4` and `per_core_M ∈ {1, 2, 4}` are all valid candidates.
- At batch=1, the only valid value is `per_core_M=1`. The current code hardcodes this correctly at `moe.py:L1149` and `moe.py:L1154`.

**`per_core_N`** — Number of N-tiles (output column-tiles, or weight column-tiles) each core owns. Derived by `_make_sparse_matmul_program_config`; not a tunable parameter.

- Formula: `per_core_N = max(1, ceil(N_tiles / num_cores))`.
- For GLM-4-MoE gate/up on T3K: `N_tiles = ceil(1408/32) = 44`, `num_cores = 64`, `per_core_N = max(1, ceil(44/64)) = 1`.
- For GLM-4-MoE down on T3K: `N_tiles = ceil(4096/32) = 128`, `num_cores = 64`, `per_core_N = max(1, ceil(128/64)) = 2`.

**`out_subblock_h` / `out_subblock_w`** — Subdivide each core's output tile block into smaller register-file subblocks for the FMA inner loop. Current defaults: `out_subblock_h=1`, `out_subblock_w=min(per_core_N, 4)`.

- These affect register pressure inside the Tensix FPU, not DRAM access patterns. For `per_core_N ∈ {1, 2}` (the only values seen in GLM-4-MoE), `out_subblock_w ∈ {1, 2}` and the defaults are reasonable.
- Do not adjust these without hardware-level profiling; incorrect subblock sizing causes kernel compile failures.

**`mcast_in0=True`** — The activation (in0) is multicast from a designated source core to all participant cores rather than each core loading independently. This is correct and required for `MultiCast1D` kernels. The alternative (`mcast_in0=False`) would cast in1 (the weight) instead, which is not appropriate for this layout.

**`fuse_batch=False`** — The batch and M dimensions of the sparse block layout `(1, num_sparse_blocks, 32, hidden_size)` are not fused before entering the matmul. Each sparse block is computed independently. Setting this to `True` would require a contiguous M-dimension across all blocks, which conflicts with the sparse dispatch layout.

**`out_block_h=1`** — Fixed at 1 throughout; each core computes one row of output blocks.

---

## Valid `in0_block_w` Ranges for GLM-4-MoE

### Gate/up projection (w1, w3): K = `hidden_size / TILE_SIZE = 4096 / 32 = 128` tiles

`in0_block_w` must be a divisor of 128. The set of all positive-integer divisors of 128 is:

```
divisors(128) = {1, 2, 4, 8, 16, 32, 64, 128}
```

The current code applies `min(4, hidden_tiles) = min(4, 128) = 4` at `moe.py:L1148`. The `min(4, ...)` cap is **not binding** here — the cap would only bite if `hidden_tiles < 4`, which would require `hidden_size < 128`. For GLM-4-MoE with `hidden_size=4096`, the cap is a no-op and the effective value is 4.

Values above 4 are arithmetically valid divisors but require verifying the L1 budget:

| `in0_block_w` | L1 tiles (activation only) | bf16 bytes | Feasible? |
|---|---|---|---|
| 1 | 1 | 2 048 | Yes |
| 2 | 2 | 4 096 | Yes |
| 4 | 4 | 8 192 | Yes (current) |
| 8 | 8 | 16 384 | Yes — within L1 at per_core_M=1, per_core_N=1 |
| 16 | 16 | 32 768 | Yes — L1 not exceeded, but verify with weight tiles |
| 32 | 32 | 65 536 | Marginal — weight + output tiles push total near 0.1 MB |

**Recommended sweep for gate/up:** `in0_block_w ∈ {1, 2, 4, 8}`. Values of 16 and above require explicit L1 budget verification and are unlikely to yield further improvement given Wormhole's memory hierarchy.

### Down projection (w2): K = `intermediate_size / TILE_SIZE = 1408 / 32 = 44` tiles

`in0_block_w` must be a divisor of 44. Factoring 44:

```
44 = 4 × 11
divisors(44) = {1, 2, 4, 11, 22, 44}
```

The current code applies `min(4, intermediate_tiles) = min(4, 44) = 4` at `moe.py:L1153`. Again, the cap is not binding (44 > 4).

| `in0_block_w` | K-loop iterations | Divisor valid? | Notes |
|---|---|---|---|
| 1 | 44 | Yes | Maximum loop overhead |
| 2 | 22 | Yes | |
| 4 | 11 | Yes (current) | |
| 11 | 4 | Yes | Next valid step up from 4 |
| 22 | 2 | Yes | High L1 activation footprint |
| 44 | 1 | Yes | Full K in one block; likely L1 overflow at per_core_M=1, per_core_N=2 |

**Recommended sweep for down:** `in0_block_w ∈ {1, 2, 4, 11}`. The jump from 4 to 11 is notable: there is no valid value between 4 and 11 for this dimension. If `in0_block_w=11` is L1-feasible and faster than 4, it represents a worthwhile improvement specific to the `intermediate_size=1408` geometry.

L1 check for `in0_block_w=11` at `per_core_M=1, per_core_N=2`:

```
activation tiles : 11 × (32×32×2 bytes) = 11 × 2048 = 22 528 bytes
weight tiles     :  2 × (32×32×2 bytes) =  2 × 2048 =  4 096 bytes
output tiles     :  1 × 2 × (32×32×2)  =              4 096 bytes
total            :                                   = 30 720 bytes ≈ 30 KB
```

30 KB is far below the 1 MB per-core L1 budget. `in0_block_w=11` is feasible.

---

## Valid `in0_block_w` Ranges for Bailing (BailingMoeV2)

Bailing uses `BailingMoeV2Config`. The relevant sizes mirror GLM-4-MoE closely:

- `hidden_size = 4096` → `hidden_tiles = 128`
- `moe_intermediate_size ≈ 1408` → `intermediate_tiles = 44`

If Bailing's `moe_intermediate_size` differs from 1408, recompute divisors as follows:

```python
import math

def valid_in0_block_w(dim_size, tile_size=32, max_feasible=16):
    """Return all valid in0_block_w values for a given dimension."""
    k_tiles = dim_size // tile_size
    assert dim_size % tile_size == 0, f"{dim_size} not tile-aligned"
    return sorted(d for d in range(1, k_tiles + 1)
                  if k_tiles % d == 0 and d <= max_feasible)

# GLM-4-MoE gate/up (K=hidden_size):
print(valid_in0_block_w(4096))   # [1, 2, 4, 8, 16]

# GLM-4-MoE/Bailing down (K=intermediate_size):
print(valid_in0_block_w(1408))   # [1, 2, 4, 11]
```

For any Bailing `moe_intermediate_size` not equal to 1408, run `valid_in0_block_w(bailing_intermediate_size)` before submitting sweep configs. Submitting a non-divisor value will cause a kernel compile assertion.

---

## Tuning Grid

Combining the valid ranges above produces the following tuning grids. "Current" marks the existing production value.

### GLM-4-MoE gate/up (w1, w3): K=128, N=44, M=1

| `in0_block_w` | K-loop iters | Status |
|---|---|---|
| 1 | 128 | Baseline (slowest expected) |
| 2 | 64 | |
| **4** | **32** | **Current** |
| 8 | 16 | Candidate improvement |

### GLM-4-MoE down (w2): K=44, N=128, M=1

| `in0_block_w` | K-loop iters | Status |
|---|---|---|
| 1 | 44 | Baseline |
| 2 | 22 | |
| **4** | **11** | **Current** |
| 11 | 4 | Candidate improvement — no valid values between 4 and 11 |

### `per_core_M` grid

| Batch size | M-tiles | Valid `per_core_M` values |
|---|---|---|
| 1 (decode) | 1 | `{1}` only |
| 2 | 2 | `{1, 2}` |
| 4 | 4 | `{1, 2, 4}` |
| 8 | 8 | `{1, 2, 4, 8}` |

At batch=1, this parameter is not a tuning knob. Sweep `per_core_M` only when validating at prefill batch sizes.

---

## Evaluation Harness

The following harness benchmarks a single program config across the full `in0_block_w` sweep for both gate/up and down projections. It uses `ttnn.matmul` directly for isolation; substitute `sparse_matmul` from the production path to measure the actual dispatch overhead.

```python
import math
import time
import ttnn
import torch

TILE = 32
WARMUP = 20
TIMED = 100

hidden_size       = 4096
intermediate_size = 1408
padded_tokens     = 32   # SPARSITY_BLOCK_SIZE

device = ttnn.open_device(device_id=0)


def make_config(device, out_features, in0_block_w, per_core_M=1):
    """Mirrors _make_sparse_matmul_program_config (moe.py:L62-L91)."""
    grid = device.compute_with_storage_grid_size()
    core_x = int(grid.x)
    core_y = int(grid.y)
    n_tiles = (out_features + TILE - 1) // TILE
    num_cores = core_x * core_y
    per_core_N = max(1, math.ceil(n_tiles / num_cores))
    out_subblock_w = min(per_core_N, 4)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def bench(device, in_shape, weight_shape, out_features, in0_block_w, fidelity):
    x_torch = torch.randn(*in_shape, dtype=torch.bfloat16)
    w_torch = torch.randn(*weight_shape, dtype=torch.bfloat16)

    x = ttnn.from_torch(x_torch, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    w = ttnn.from_torch(w_torch, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)

    prog_cfg = make_config(device, out_features, in0_block_w)
    # WormholeComputeKernelConfig(HiFi2) — see math_fidelity_evaluation.md for the canonical definition
    compute_cfg = ttnn.WormholeComputeKernelConfig(math_fidelity=fidelity,
                                                   math_approx_mode=False,
                                                   fp32_dest_acc_en=True,
                                                   packer_l1_acc=True)

    for _ in range(WARMUP):
        out = ttnn.matmul(x, w, program_config=prog_cfg,
                          compute_kernel_config=compute_cfg)
        ttnn.synchronize_device(device)
    ttnn.deallocate(out)

    t0 = time.perf_counter()
    for _ in range(TIMED):
        out = ttnn.matmul(x, w, program_config=prog_cfg,
                          compute_kernel_config=compute_cfg)
        ttnn.synchronize_device(device)
    t1 = time.perf_counter()

    ttnn.deallocate(out)
    ttnn.deallocate(x)
    ttnn.deallocate(w)
    return (t1 - t0) / TIMED * 1e6   # microseconds


# Gate/up sweep (K=hidden_size=4096, N=intermediate_size=1408)
gate_up_candidates = [1, 2, 4, 8]
print("=== Gate/Up (w1, w3): in0=32x4096, out=32x1408 ===")
print(f"{'in0_block_w':>12}  {'fidelity':>8}  {'latency_us':>12}")
for bw in gate_up_candidates:
    lat = bench(device,
                in_shape=(1, 1, padded_tokens, hidden_size),
                weight_shape=(hidden_size, intermediate_size),
                out_features=intermediate_size,
                in0_block_w=bw,
                fidelity=ttnn.MathFidelity.HiFi2)
    print(f"{bw:>12}  {'HiFi2':>8}  {lat:>12.1f}")

# Down sweep (K=intermediate_size=1408, N=hidden_size=4096)
down_candidates = [1, 2, 4, 11]
print("\n=== Down (w2): in0=32x1408, out=32x4096 ===")
print(f"{'in0_block_w':>12}  {'fidelity':>8}  {'latency_us':>12}")
for bw in down_candidates:
    lat = bench(device,
                in_shape=(1, 1, padded_tokens, intermediate_size),
                weight_shape=(intermediate_size, hidden_size),
                out_features=hidden_size,
                in0_block_w=bw,
                fidelity=ttnn.MathFidelity.HiFi2)
    print(f"{bw:>12}  {'HiFi2':>8}  {lat:>12.1f}")

ttnn.close_device(device)
```

---

## Expected Results and Measurement Tables

Fill in the measured values after running the harness. Expected patterns are noted based on hardware first-principles.

**Gate/up projection (w1 or w3 individually): in0=32×4096, out=32×1408**

| `in0_block_w` | K-loop iters | Latency (µs) | vs current (in0_block_w=4) |
|---|---|---|---|
| 1 | 128 | ___ | ___ |
| 2 | 64 | ___ | ___ |
| 4 | 32 | ___ | current |
| 8 | 16 | ___ | ___ |

**Down projection (w2): in0=32×1408, out=32×4096**

| `in0_block_w` | K-loop iters | Latency (µs) | vs current (in0_block_w=4) |
|---|---|---|---|
| 1 | 44 | ___ | ___ |
| 2 | 22 | ___ | ___ |
| 4 | 11 | ___ | current |
| 11 | 4 | ___ | ___ |

**Expected pattern:** At batch=1 the sparse matmul is memory-bound, so `in0_block_w=4` is likely near-optimal for both projections. The `in0_block_w=11` candidate for the down projection is the one exception worth measuring: the 4→11 jump reduces K-loop iterations from 11 to 4 (a 2.75× reduction) and there is no valid value between 4 and 11 for `K=44`. When batch ≥ 8 and the kernel becomes compute-bound, larger `in0_block_w` values and `per_core_M > 1` become relevant tuning knobs.

---

## Pareto Selection Criterion

After collecting measurements, select the `in0_block_w` value that minimizes mean latency across `{w1_latency + w3_latency + w2_latency}` while remaining within the valid-divisor set for both `K=128` (gate/up) and `K=44` (down). Report the aggregate Stages 4+5 time:

```
Stage_4_5_total = w1_latency + w3_latency + silu_mul_latency + w2_latency
```

If `in0_block_w=4` remains optimal for both projections, document that the current default is confirmed optimal. Do not change production values without a confirmed latency improvement of at least 5% on device — within-noise changes should not be committed.

---

**Next:** [`math_fidelity_evaluation.md`](./math_fidelity_evaluation.md)
