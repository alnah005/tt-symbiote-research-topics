# Tensix Packer Pipeline and the Role of `packer_l1_acc`

## The Tensix Compute Pipeline

The Tensix compute pipeline on Tenstorrent Wormhole hardware proceeds through three major stages for every matmul tile operation:

```
Unpacker → FPU (math) → Packer
```

1. **Unpacker** — reads input tiles from L1 SRAM into the source registers consumed by the FPU (Floating-Point Unit).
2. **FPU** — performs the tile-level matrix multiply-accumulate (MAC) operations, writing results into the destination register file.
3. **Packer** — serializes output tiles out of the destination register and writes them to a memory destination (either L1 or DRAM depending on configuration).

The packer is the final stage: it owns the decision of *where* completed output tiles land after the FPU finishes. This is exactly the degree of freedom controlled by `packer_l1_acc`.

---

## Matmul Outer-Loop Structure

A matmul of shape [M, K] × [K, N] is tiled and executed as nested loops over tile coordinates. In tile units, using the subscript-T notation for tile counts (K_t = K/32, M_t = M/32, N_t = N/32):

```python
# Conceptual outer-loop structure for a single Tensix core
for m in range(M_t):
    for n in range(N_t):
        partial_sum = zeros(tile_shape)
        for k_block in range(0, K_t, in0_block_w):   # outer K loop
            # FPU accumulates in0_block_w tiles of K per iteration
            partial_sum += fpu_matmul(in0[m, k_block], in1[k_block, n])
        output[m, n] = partial_sum
```

The `in0_block_w` parameter controls how many K_t-direction tiles are consumed per outer-loop iteration. With `in0_block_w = 1`, the K loop runs K_t times. With `in0_block_w = 4`, it runs K_t / 4 times.

---

## `packer_l1_acc=False` (Default): Per-Iteration DRAM Write-Back

When `packer_l1_acc=False`, the packer writes the partial sum out of the destination register to **DRAM** at the end of each outer-loop K iteration. On the next iteration, the unpacker must read that partial sum back from DRAM before the FPU can accumulate the next block of K tiles.

This creates a read-modify-write round-trip to DRAM for every K-block step:

```
Iteration k_block=0:  FPU compute → Packer writes partial_sum → DRAM
Iteration k_block=1:  DRAM → Unpacker reads partial_sum → FPU accumulate → Packer writes → DRAM
Iteration k_block=2:  DRAM → Unpacker reads partial_sum → FPU accumulate → Packer writes → DRAM
...
Iteration k_block=K_t-1: DRAM → Unpacker reads → FPU accumulate → Packer writes final → DRAM
```

For K_t = 64 and `in0_block_w = 1`, this creates significant unnecessary DRAM reads and writes per output tile — traffic that carries no useful new information, only the running partial sum. For the quantified example (K_t=64, b=1 → 63 extra reads), see [`throughput_impact.md`](./throughput_impact.md).

> **Warning:** `packer_l1_acc=False` is the default in `WormholeComputeKernelConfig`. For decode-mode workloads where matmuls are memory-bandwidth-bound, this default causes significant redundant DRAM traffic and should almost always be overridden.

---

## `packer_l1_acc=True`: L1 Accumulation, Single DRAM Write

When `packer_l1_acc=True`, the packer accumulates partial sums into a dedicated **L1 buffer** on the same core. The destination register output is folded into the L1 accumulation buffer at each K iteration without touching DRAM. DRAM is written exactly once, when the full K_t accumulation is complete:

```
Iteration k_block=0:  FPU compute → Packer writes partial_sum → L1 accum buffer
Iteration k_block=1:  L1 accum buffer + FPU result → updated L1 accum buffer
Iteration k_block=2:  L1 accum buffer + FPU result → updated L1 accum buffer
...
Iteration k_block=K_t-1: L1 accum buffer + FPU result → Packer writes final → DRAM (once)
```

This eliminates per-iteration DRAM read-modify-write entirely. The output DRAM write count drops from K_t / `in0_block_w` to **1** per output tile.

---

## Bandwidth Model

See the bandwidth reduction table in [`throughput_impact.md`](./throughput_impact.md) for quantified examples across K_t/b combinations.

---

## Position in `WormholeComputeKernelConfig`

`packer_l1_acc` is the fourth field of `WormholeComputeKernelConfig`:

```python
from ttnn import WormholeComputeKernelConfig

config = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.LoFi,       # field 1: arithmetic precision
    math_approx_mode=True,                  # field 2: approximate math ops
    fp32_dest_acc_en=False,                 # field 3: fp32 destination accumulator
    packer_l1_acc=True,                     # field 4: accumulate in L1 (this chapter)
)
```

`packer_l1_acc` is independent of `math_fidelity` and `math_approx_mode`. Its only meaningful interaction is with `fp32_dest_acc_en`, which affects the *size* of the L1 accumulation buffer — covered in detail in [`packer_l1_acc_constraints.md`](./packer_l1_acc_constraints.md).

---

## Next Steps

Continue to [`throughput_impact.md`](./throughput_impact.md) for quantitative throughput analysis: bandwidth reduction formulas, decode-mode vs. prefill regime comparison, and the DeepSeek-V3 TTNN configuration as a real-world reference point.
