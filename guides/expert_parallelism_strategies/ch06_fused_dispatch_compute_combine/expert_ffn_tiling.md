# Expert FFN Tiling

## Section 1: Local Expert Batch After Dispatch

After the all-to-all dispatch completes, each device holds a received-token buffer
organized by expert. Let $T_{\text{received}}$ denote the total number of tokens
received by a device. Under uniform routing, $T_{\text{received}} \approx B$ (since
each of the $N = 8$ devices is expected to route $B \times k / N$ token-expert pairs
inward, and $k/N = 8/8 = 1$ on average). Non-uniform routing can cause $T_{\text{received}}$ to vary; the capacity factor $CF = 1.25$ ensures buffers are
large enough to absorb fluctuations.

The received buffer is logically sliced per expert. For expert $e \in \{0, \ldots, 31\}$:

$$\text{recv\_buf}[e] \in \mathbb{R}^{\text{received\_count}_e \times H}$$

where $\text{received\_count}_e$ is the number of tokens routed to expert $e$ on
this device, $H = 7168$, and the total satisfies:

$$\sum_{e=0}^{E_d - 1} \text{received\_count}_e = T_{\text{received}}$$

The $E_d = 32$ experts are independent of one another: the FFN of expert $e$ reads
only from $\text{recv\_buf}[e]$ and produces output only for tokens assigned to
expert $e$. This independence is exploited for parallel execution.

## Section 2: Parallel Expert Execution

Wormhole B0 provides 80 Tensix cores per device. With $E_d = 32$ local experts, a
natural assignment is:

$$\text{cores per expert} = \lfloor 80 / 32 \rfloor = 2 \text{ (with } 80 - 32 \times 2 = 16 \text{ cores remaining)}$$

The 16 remaining cores can be allocated to the experts that receive the most tokens
(load-adaptive assignment), or distributed round-robin to bring some experts to 3
cores each. A static assignment of 2–3 cores per expert is sufficient for most
decode batch sizes.

**Execution model:**

All 32 expert FFN kernels are dispatched simultaneously to their assigned core groups.
Each core group independently executes:

1. Load weight tiles for its expert from DRAM into L1 (streaming, see Section 3).
2. Compute $[\text{received\_count}_e, H] \times [H, D]$ for the up and gate projections.
3. Apply the activation function (e.g., SiLU gating).
4. Compute $[\text{received\_count}_e, D] \times [D, H]$ for the down projection.
5. Write expert output $[\text{received\_count}_e, H]$ to the combine send buffer.

All 32 expert FFN kernels complete before Stage 4 (combine all-to-all) can begin,
because the combine sends the concatenation of all expert outputs.

> **Tip:** On Wormhole B0, the NoC fabric allows multiple core groups to stream DRAM
> weight tiles concurrently. Scheduling all 32 expert kernels before any of them
> execute avoids head-of-line blocking on the NoC.

## Section 3: L1 Memory Management for Expert Weights

The per-expert weight footprint is (from Chapter 4, `uniform_partitioning.md`):

$$W_{\text{expert}} = 6 H D \text{ bytes (BF16)}$$

For $H = 7168$ and $D$ [UNVERIFIED exact value for Qwen3.5-35B FFN intermediate
dimension]: if $D \approx 2H \approx 14336$, then:

$$W_{\text{expert}} \approx 6 \times 7168 \times 14336 \times 2 \approx 1.23 \text{ GB per expert (all 32 weights)}$$

This is far larger than the 1.5 MB L1 capacity of a single core. Even if we consider
the weights for one expert alone:

$$W_{\text{single expert}} = 6 \times 7168 \times D \times 2 \text{ bytes}$$

At $D \approx 14336$ [UNVERIFIED]: $W_{\text{single expert}} \approx 1.17$ GB.

Clearly, expert weights cannot be held resident in L1. The strategy is
**tile-by-tile DRAM streaming**:

- The matmul kernel streams weight tiles of shape $[\text{TILE\_SIZE}, \text{TILE\_SIZE}] = [32, 32]$
  (each tile is $32 \times 32 \times 2 = 2048$ bytes) from DRAM into L1 on demand.
- The activation tensor for a single expert is small and fits entirely in L1:

$$\text{activation size} = \text{received\_count}_e \times H \times 2 \text{ bytes}$$

At $C = 2$ (i.e., $\text{received\_count}_e \approx 2$):

$$2 \times 7168 \times 2 = 28{,}672 \text{ bytes} \approx 28 \text{ KB} \ll 1.5 \text{ MB}$$

The activation tensor fits in L1 for all expected decode batch sizes, avoiding DRAM
reads for activations during the matmul inner loop.

**Tile streaming schedule per expert:**

```
for each weight tile [i, j] in [H, D]:
    load weight_tile[i, j] from DRAM into L1   (2 KB)
    multiply with activation rows               (in L1)
    accumulate into partial output tile          (in L1)
    evict weight_tile[i, j] from L1             (replaced by next tile)
```

> **Warning:** At large prefill batch sizes, both the weight tiles and activation tiles
> are large. L1 pressure increases. Monitor L1 utilization with the TTNN profiler to
> ensure no tile eviction thrashing occurs.

## Section 4: Sparsity Exploitation

Under non-uniform routing or at small batch sizes, many experts receive zero tokens.
Skipping the FFN computation for such experts avoids wasteful DRAM weight streaming
and Tensix core occupancy.

**Skip condition:**

```python
for e in range(E_d):  # E_d = 32
    if received_count[e] == 0:
        # Write zeros to combine send buffer for expert e
        combine_send_buf[e] = torch.zeros(0, H)  # empty slice
        continue
    # Otherwise launch FFN kernel for expert e
    launch_expert_ffn_kernel(e, recv_buf[e], combine_send_buf[e])
```

**Expected token counts at decode:**

| Batch size $B$ | Expected tokens per expert ($B / E_d$) | Expected zero-token experts |
|---|---|---|
| 1 | 0.031 | ~24 of 32 (most experts receive nothing) |
| 8 | 0.25 | ~19 of 32 |
| 32 | 1.0 | ~12 of 32 (Poisson; $P(0) = e^{-1} \approx 37\%$) |
| 256 | 8.0 | ~0 of 32 (essentially full utilization) |

> **Note:** The per-expert token count follows approximately a Poisson distribution
> with mean $\lambda = B \times k / E = B / 32$ under uniform routing. At $B = 32$,
> $\lambda = 1$, so about $e^{-1} \approx 37\%$ of experts receive zero tokens per
> device per micro-batch.

Skipping zero-token experts at $B = 32$ saves roughly $37\%$ of expert FFN kernel
launches, which is meaningful given that each launch incurs NoC traffic for weight
streaming.

## Section 5: Matmul Dimensions and Tile Padding

Each expert FFN consists of two matmuls per expert (using SwiGLU or similar gating):

| Matmul | Input | Weight | Output |
|---|---|---|---|
| Up + Gate projection | $[\text{rc}_e, H]$ | $[H, D]$ | $[\text{rc}_e, D]$ |
| Down projection | $[\text{rc}_e, D]$ | $[D, H]$ | $[\text{rc}_e, H]$ |

where $\text{rc}_e = \text{received\_count}_e$, $H = 7168$, $D$ = [UNVERIFIED exact
value for FFN intermediate dimension].

**Tile alignment issue at small batch sizes:**

Wormhole B0 Tensix matmul kernels require dimensions aligned to `TILE_SIZE = 32`.
At $C = 2$ (decode, $B = 32$): $\text{rc}_e \approx 2$, which is far below 32.
The kernel must pad $\text{rc}_e$ up to 32 rows:

$$\text{padding waste} = \frac{32 - \text{rc}_e}{32} \approx \frac{30}{32} \approx 94\%$$

This means roughly 94% of the matmul FLOPs at decode are wasted on padding zeros.
This is a fundamental inefficiency of dense-matmul kernels at very small token counts.

**Mitigation strategies:**

1. **Sparse matmul kernel:** Operate on only the $\text{rc}_e$ real rows; no padding.
   TTNN does not currently expose a general sparse matmul, but batch-1 matmul
   specializations may be available.
2. **Expert batching:** Group multiple experts' received tokens into a single padded
   matmul call across experts, amortizing the padding overhead.
3. **TTNN padding bypass:** Use `ttnn.pad` with `pad_value=0` to explicitly construct
   a $[32, H]$ tensor; the kernel operates on the full tile but results for padding
   rows are discarded. This avoids kernel-launch overhead while still wasting compute.

> **Tip:** At prefill batch sizes ($B \geq 256$), $\text{rc}_e \gg 32$ and padding
> waste becomes negligible. The tile alignment issue is specific to decode regimes.

## References

- Chapter 4 of this guide: `ch04_expert_device_assignment/uniform_partitioning.md`
  ($W_{\text{expert}}$ derivation, $E_d = 32$).
- Chapter 2 of this guide: `ch02_all_to_all_primitives/all_to_all_dispatch.md`
  (dispatch buffer layout, `received_count` semantics).
- TT-Metalium TTNN documentation: Tensix matmul tiling, `TILE_SIZE`, L1 memory layout.
- Wormhole B0 Architecture Guide: NoC bandwidth, L1 per core (1.5 MB), aggregate
  L1 (120 MB).
