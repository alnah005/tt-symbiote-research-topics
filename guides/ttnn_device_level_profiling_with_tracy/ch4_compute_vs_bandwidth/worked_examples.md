# Worked Examples

This file walks through three complete classification examples using the method from [`classification_method.md`](./classification_method.md). Each example follows the same five-step procedure: (1) theoretical AI, (2) `FPU UTIL`, (3) `NOC BW UTIL`, (4) `DEVICE KERNEL DURATION / PM IDEAL`, (5) TRISC breakdown.

---

## Example 1 — Large Matmul (M=1024, K=4096, N=4096, BF16)

### Step 1: Theoretical Arithmetic Intensity

```python
M, K, N = 1024, 4096, 4096
bytes_per_element = 2  # BF16

flops = 2 * M * K * N
# = 2 × 1024 × 4096 × 4096 = 34,359,738,368  (~34.4 GFLOPs)

bytes_A = M * K * bytes_per_element   # = 1024 × 4096 × 2 =  8,388,608
bytes_B = K * N * bytes_per_element   # = 4096 × 4096 × 2 = 33,554,432
bytes_C = M * N * bytes_per_element   # = 1024 × 4096 × 2 =  8,388,608
total_bytes = bytes_A + bytes_B + bytes_C  # = 50,331,648  (~50.3 MB)

AI = flops / total_bytes
# = 34,359,738,368 / 50,331,648 ≈ 682.7 FLOPs/byte
```

Tile counts: `M_t = 32`, `K_t = 128`, `N_t = 128`.

**AI ≈ 683 FLOPs/byte**, which is vastly above the ridge point of 8.0 FLOPs/byte. Initial hypothesis: strongly compute-bound.

### Step 2: `FPU UTIL`

Expected value from the CSV: **~0.82 – 0.90**.

At this scale, the FPU pipeline is deeply pipelined and the tile loop amortizes all setup overhead. `TRISC1` spends most of its time issuing FMA instructions. An `FPU UTIL` in the 0.82–0.90 range is consistent with a well-formed large matmul — it does not reach 1.0 because there is a small amount of loop overhead and tile boundary work that cannot be fully hidden.

### Step 3: `NOC BW UTIL`

Expected value from the CSV: **~0.25 – 0.35**.

The NoC is reading 50 MB of operand data, but the FPU is consuming it at 256 FLOPs/cycle (128 FMA ops/cycle × 2 FLOPs/FMA), meaning the read bandwidth requirement per FLOP is very low. The NoC is lightly loaded compared to its peak, confirming that bandwidth is not the bottleneck.

### Step 4: `DEVICE KERNEL DURATION / PM IDEAL`

Expected ratio: **~1.1 – 1.3**.

For a well-formed large matmul on a properly sized core grid, the measured wall-clock duration should be close to `PM IDEAL`. A ratio near 1.1–1.3 indicates minimal overhead beyond what the model predicts. If the ratio were > 2.0, it would suggest a suboptimal core grid (too many idle cores at the boundary) or synchronization overhead.

### Step 5: TRISC Duration Breakdown

Expected pattern:
- `TRISC1 DURATION`: **longest** — the FPU is the long pole.
- `TRISC0 DURATION`: moderately long, but shorter than TRISC1 — the unpacker can keep up with the FPU because the NoC is not saturated.
- `TRISC2 DURATION`: shorter still — the output matrix C is smaller than B, so writes are less expensive.

### Conclusion

**Compute-bound.** The FPU is the primary bottleneck. Optimization should focus on:
- Ensuring the core grid assignment maximizes tile occupancy per core.
- Checking whether the data format can be upgraded (e.g., FP8 reduces bytes transferred, raising AI further and keeping the kernel firmly in the compute-bound regime).
- Verifying that TRISC1 duration tracks `PM IDEAL` to confirm the FPU is running at its ceiling.

---

## Example 2 — Small Matmul (M=32, K=256, N=256, BF16)

### Step 1: Theoretical Arithmetic Intensity

```python
M, K, N = 32, 256, 256
bytes_per_element = 2  # BF16

flops = 2 * M * K * N
# = 2 × 32 × 256 × 256 = 4,194,304  (~4.2 MFLOPs)

bytes_A = M * K * bytes_per_element   # = 32 × 256 × 2 =  16,384
bytes_B = K * N * bytes_per_element   # = 256 × 256 × 2 = 131,072
bytes_C = M * N * bytes_per_element   # = 32 × 256 × 2 =  16,384
total_bytes = bytes_A + bytes_B + bytes_C  # = 163,840  (~160 KB)

AI = flops / total_bytes
# = 4,194,304 / 163,840 ≈ 25.6 FLOPs/byte
```

Tile counts: `M_t = 1`, `K_t = 8`, `N_t = 8`. Total tile operations: `1 × 8 × 8 = 64` tile multiplications.

**AI ≈ 25.6 FLOPs/byte**, which is above the ridge point of 8.0 FLOPs/byte. Theoretical hypothesis: compute-bound.

> **Warning:** The theoretical AI predicts compute-bound, but the tile count is only 64. With so few tiles, the FPU pipeline cannot be kept full, and setup/teardown overhead (loop preamble, tile load latency, sync) dominates. The theoretical AI is misleading here — this is a case where measured `FPU UTIL` will tell a very different story.

### Step 2: `FPU UTIL`

Expected value from the CSV: **~0.10 – 0.20**.

With only 64 tile operations and a pipeline depth of ~8–16 cycles per tile, the FPU executes useful work for only a fraction of TRISC1's active time. Most of TRISC1's time is spent in loop overhead, waiting for the first tile to arrive, and flushing the pipeline at the end.

This is a critical diagnostic signal: even though the theoretical AI says compute-bound, the low `FPU UTIL` shows the FPU is severely underutilized.

### Step 3: `NOC BW UTIL`

Expected value: **~0.15 – 0.30**.

The total data volume is only 160 KB. The NoC can transfer this quickly, but the real problem is that the kernel finishes (and mostly stalls) before the NoC is ever stressed. Both `FPU UTIL` and `NOC BW UTIL` are low.

### Step 4: `DEVICE KERNEL DURATION / PM IDEAL`

Expected ratio: **> 3.0, possibly 5.0+**.

`PM IDEAL` would be computed assuming ideal FPU throughput and ideal bandwidth, but neither assumption holds at this tile count. The actual duration will be dominated by fixed per-kernel overhead that `PM IDEAL` does not model. A high ratio (> 3×) confirms the kernel is overhead-bound.

### Step 5: TRISC Duration Breakdown

Expected pattern:
- `TRISC0`, `TRISC1`, `TRISC2` durations are all **similar and short** in absolute terms.
- `DEVICE KERNEL DURATION` is much larger than any individual TRISC duration, indicating synchronization or dispatch overhead between stages.

### Conclusion

**Overhead-bound** (masquerading as compute-bound based on theoretical AI alone).

The kernel has too few tiles to amortize the fixed overhead of the Tensix pipeline.

**Recommendation: reduce the core grid.** If this matmul is dispatched across a 4×4 grid of cores, each core receives only a handful of output tiles. Reducing to a 1×1 or 2×2 grid increases the tiles-per-core count, giving TRISC1 more contiguous work and dramatically improving `FPU UTIL`.

```python
# Example: estimating minimum tiles-per-core for healthy FPU utilization
M_t, K_t, N_t = 1, 8, 8   # from shapes above
total_output_tiles = M_t * N_t  # = 8

# On a 4×4 grid: 8 output tiles / 16 cores → most cores get 0 tiles (wasted)
# On a 1×2 grid: 8 output tiles / 2 cores  → 4 tiles per core (better)
# On a 1×1 grid: 8 output tiles / 1 core   → 8 tiles per core (best for this shape)
```

> **Tip:** As a rule of thumb, each active core should have at least 4–8 output tiles to sustain healthy FPU pipeline fill. Below 4 tiles per core, `FPU UTIL` will typically fall below 0.3 regardless of the theoretical AI.

---

## Example 3 — Elementwise Op (`ttnn.silu` on [1024, 4096] tensor, BF16)

### Step 1: Theoretical Arithmetic Intensity

`silu(x) = x × sigmoid(x) = x / (1 + exp(-x))`. The FPU cost per element is approximately 4–6 FLOPs (one negation, one exp, one add, one reciprocal, one multiply — with some implementations fusing steps). Using 5 FLOPs/element as a reasonable estimate:

```python
H, W = 1024, 4096
bytes_per_element = 2  # BF16
flops_per_element = 5  # silu approximation

flops      = flops_per_element * H * W
# = 5 × 1024 × 4096 = 20,971,520  (~21 MFLOPs)

bytes_read  = H * W * bytes_per_element  # = 1024 × 4096 × 2 = 8,388,608
bytes_write = H * W * bytes_per_element  # = 1024 × 4096 × 2 = 8,388,608
total_bytes = bytes_read + bytes_write   # = 16,777,216  (~16 MB)

AI = flops / total_bytes
# = 20,971,520 / 16,777,216 ≈ 1.25 FLOPs/byte
```

Tile counts: `H_t = 32`, `W_t = 128`. Total tiles: 4096.

**AI ≈ 1.25 FLOPs/byte**, which is well below the ridge point of 8.0 FLOPs/byte. Initial hypothesis: strongly bandwidth-bound.

> **Note:** For any unary elementwise op, AI = `flops_per_element / (2 × bytes_per_element)` because each element is read once and written once. For BF16, this is `flops_per_element / 4`. Even a relatively expensive op like `silu` (5 FLOPs) yields AI = 1.25. Simple ops like `relu` (1 FLOP) give AI = 0.25. All elementwise ops on BF16 are bandwidth-bound.

### Step 2: `FPU UTIL`

Expected value from the CSV: **~0.04 – 0.08**.

The FPU executes the silu operation in a handful of cycles per 32-element vector. Most of TRISC1's time is waiting for TRISC0 to deliver the next input tile and for TRISC2 to drain the output tile. The FPU is nearly idle.

### Step 3: `NOC BW UTIL`

Expected value: **~0.75 – 0.90**.

The kernel is doing almost nothing but read and write data. With 16 MB of total data movement and a NoC link bandwidth of 32 bytes/cycle, the NoC is the primary consumer of time. A `NOC BW UTIL` near 0.8 means the kernel is running close to the NoC bandwidth ceiling — this is the expected and healthy behavior for a well-formed elementwise op.

### Step 4: `DEVICE KERNEL DURATION / PM IDEAL`

Expected ratio: **~1.05 – 1.20**.

`PM IDEAL` is dominated by the memory term (`total_bytes / bytes_per_cycle`). Because the op's compute content is negligible, `PM IDEAL ≈ memory_cycles`. The measured duration should be close to this model prediction, giving a ratio near 1.0–1.2 for a well-formed dispatch.

### Step 5: TRISC Duration Breakdown

Expected pattern:
- `NCRISC DURATION`: **longest** — the data reader is continuously pulling input tiles from DRAM or L1 of neighbor cores.
- `TRISC2 DURATION`: comparable to TRISC0 — TRISC2 packs math results into L1 output buffers; it is NCRISC that subsequently pushes those tiles over the NoC.
- `TRISC1 DURATION`: **shortest** — compute completes instantly relative to the data-movement time.

This is the canonical signature of a bandwidth-bound kernel: the compute TRISC is the shortest, and the I/O TRISCs are the long poles.

### Conclusion

**Bandwidth-bound (NoC).** The kernel is limited by the rate at which data can be moved through the NoC.

This classification is expected and not a problem in itself — elementwise ops are always bandwidth-bound by their mathematical nature. The correct optimization goal is not to make the FPU work harder, but to:

1. **Fuse adjacent ops** to reduce total NoC traffic. For example, fusing `silu` into the preceding or following matmul means the activation tensor never has to be written to L1/DRAM and read back.
2. **Verify the tensor is in L1, not DRAM**, so that the effective bandwidth matches the NoC ceiling rather than the much lower DRAM effective bandwidth per core.
3. **Check that the core grid is appropriate** — too many cores on a small tensor means each core handles only a few tiles, reintroducing dispatch overhead.

> **Tip:** When you see `FPU UTIL < 0.1` and `NOC BW UTIL > 0.7`, the classification is unambiguously bandwidth-bound. The focus for these ops should be op fusion and memory placement, not FPU-level optimization.

---

## Summary Comparison

| Example | Shape | Theoretical AI | `FPU UTIL` | `NOC BW UTIL` | `DKDUR/PM IDEAL` | Classification |
|---|---|---|---|---|---|---|
| Large matmul | (1024, 4096, 4096) BF16 | ~683 FLOPs/B | ~0.85 | ~0.30 | ~1.15 | Compute-bound |
| Small matmul | (32, 256, 256) BF16 | ~26 FLOPs/B | ~0.15 | ~0.20 | ~4.5 | Overhead-bound |
| `silu` elementwise | [1024, 4096] BF16 | ~1.25 FLOPs/B | ~0.06 | ~0.82 | ~1.10 | Bandwidth-bound (NoC) |

The small matmul is the most instructive: its theoretical AI suggests compute-bound, but the actual CSV metrics reveal overhead-bound behavior because the tile count is too small to keep the pipeline fed. Always let the measured `FPU UTIL` and `overhead_ratio` override the theoretical prediction.

---

**Next:** [Chapter 5 — Low FPU Utilization: Causes and Remediation](../ch5_low_fpu_util/index.md)
