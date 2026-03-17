# Bottleneck Diagnosis Guide

This file provides a structured decision procedure for mapping profiler measurements to a
bottleneck category, and per-category remediation procedures. It is the final step in the
5-step profiling workflow described in `index.md` and assumes you have already collected
TTNN profiler output (see `ttnn_profiler.md`) and device counter readings
(see `device_perf_counters.md`).

---

## Section 1: Decision Tree — Identify the Bottleneck Category

Use the following decision tree with the measurements from Steps 3 and 4 of the profiling
workflow. Assign every profiling run to exactly one primary category before applying any
remediation.

```
Measure total MoE layer time (sum of all device_time_ns across one forward pass)
        |
Is all_to_all time (dispatch + combine) > 50% of total MoE layer time?
   |
   YES → Communication-bound  (see Section 2)
   |
   NO  → Is expert matmul DRAM read BW ≥ 80% of theoretical peak?   [UNVERIFIED threshold]
         |
         YES → Memory-bandwidth-bound  (see Section 3)
         |
         NO  → Is Tensix core utilization > 80% during expert matmul?
               |
               YES → Compute-bound  (see Section 4)
               |
               NO  → L1-pressure or scheduling overhead  (see Section 5)
```

### Regime vs. Batch Size

The dominant bottleneck category shifts with batch size. Profile at $B=1$ and $B=32$ before
concluding. At decode with $B=1$ ($C=1$, dispatch volume ≈ 3.2 MB/device) the workload is
almost always communication-bound. At $B=32$ ($C=2$, dispatch volume ≈ 6.4 MB/device) the
workload may remain communication-bound or cross over into compute-bound territory depending
on `num_links`. During prefill with large $B \cdot S$, the expert FFN matmul dominates and the
workload is typically compute-bound or memory-bandwidth-bound.

---

## Section 2: Communication-Bound Remediation

**Indicator:** `ttnn.all_to_all` (dispatch + combine) accounts for more than 50% of total MoE
layer time. Ethernet link utilization counter is high during these ops; Tensix utilization
is low (cores are idle waiting for transfers).

### Primary Lever: Increase num_links

Adding Ethernet links reduces all-to-all transfer time by parallelizing the payload across
multiple links. The T3K 1×8 linear mesh provides a fixed number of Ethernet ports per device;
consult `ch03_all_to_all_num_links/num_links_parameter.md` for the hardware maximum.

Use the following table as a starting point, then verify with profiling:

| Payload per Device | Recommended num_links |
|---|---|
| < 1 MB | 1 |
| 1–10 MB | 1–2 |
| > 10 MB | 2 or hardware maximum |

At decode with $B=32$, $C=2$: dispatch volume ≈ 6.4 MB/device → start with `num_links=1`,
try `num_links=2` and compare. The latency reduction from adding links is approximately
$\text{transfer\_time} / \text{num\_links}$ at full payload; for small payloads, link setup
overhead erodes this gain (see `device_perf_counters.md` §4).

> **Warning:** Increasing `num_links` beyond the point of diminishing returns wastes Ethernet
> port resources that could be used for other collective operations. Always verify with profiling
> that the additional link reduces latency by at least 15% before keeping the change.

### Secondary Lever: Expert Placement Optimization

Reducing the average number of Ethernet hops between token owners and expert devices reduces
all-to-all transfer latency. On the T3K 1×8 linear mesh, experts concentrated at the endpoints
(devices 0 and 7) require up to 7 hops from the opposite endpoint. Redistribute experts to
minimize average hop count for the expected token distribution. See
`ch05_expert_parallelism/expert_placement_strategies.md` for placement strategies.

### Advanced Lever: Pipeline Dispatch + Compute + Combine

If the workload allows micro-batch splitting, it is possible to overlap the all-to-all dispatch
for micro-batch $i+1$ with the expert FFN compute for micro-batch $i$, and the combine for
$i-1$. This requires double-buffering the dispatch and combine tensors. The implementation
complexity is high; apply this only after the primary and secondary levers have been exhausted.

---

## Section 3: Memory-Bandwidth-Bound Remediation

**Indicator:** Expert matmul time is the dominant op (> 50% of layer time); Ethernet utilization
is low; DRAM read bandwidth during expert FFN matmul is at or near the theoretical peak of
~300 GB/s per chip [UNVERIFIED]. Tensix utilization is moderate rather than high, indicating
the cores are stalled waiting for DRAM data rather than actively computing.

### Primary Lever: Promote Activation Tensors to L1

During decode, the expert activation tensor has shape $[C, H] = [2, 7168]$ in BF16, which is
approximately 28 KB — well within the 1.5 MB L1 budget per core. Place this tensor in L1 to
avoid repeated DRAM reads across matmul iterations:

```python
activation_memory_config = ttnn.L1_MEMORY_CONFIG

expert_activations = ttnn.to_memory_config(
    dispatch_output,
    memory_config=activation_memory_config,
)
```

See `ch04_memory_config/decode_memory_strategy.md` for the full L1 budget calculation and
placement API.

> **Tip:** Expert weight tensors at $H=7168$, $D \approx 14{,}336$ [UNVERIFIED] are approximately
> 205 MB per expert — they cannot fit in L1 under any configuration and must be streamed from DRAM.
> Memory-bandwidth-bound remediation therefore focuses on activations and intermediate tensors,
> not on weights.

### Secondary Lever: DRAM Interleaved Layout

Ensure expert weight tensors use DRAM interleaved layout (`ttnn.DRAM_MEMORY_CONFIG` with
`TensorMemoryLayout.INTERLEAVED`) rather than a contiguous layout. Interleaved layout spreads
tiles across all DRAM channels, allowing simultaneous reads from multiple channels and
approaching peak aggregate DRAM bandwidth.

### Tertiary Lever: INT8 Weight Quantization

Quantizing expert weights from BF16 to INT8 halves the bytes read from DRAM per matmul tile.
At memory-bandwidth-bound regime this directly halves expert FFN time. Verify numerical accuracy
(PCC > 0.99 vs. BF16 reference) before enabling in production.

### Checking Expert Weight Size

The expert FFN weight includes two projections: $W_1 \in \mathbb{R}^{H \times D}$ (gate/up) and
$W_2 \in \mathbb{R}^{D \times H}$ (down), where $D$ is the expert intermediate dimension
[VERIFY $D$ for Qwen3.5-35B]. Total weight bytes per expert in BF16:

$$\text{Weight size} = 2 \times H \times D \times 2 \text{ bytes}$$

At $H=7168$, $D=14{,}336$ [UNVERIFIED]: $2 \times 7168 \times 14336 \times 2 \approx 410$ MB
per expert. Each of the 32 experts on a device contributes to streaming demand; the DRAM
bandwidth required is proportional to the number of active experts per matmul step.

---

## Section 4: Compute-Bound Remediation

**Indicator:** Expert FFN `ttnn.matmul` time is dominant (> 50% of layer time); Tensix core
utilization is above 80%; DRAM read BW is consistent with weight streaming (not above it);
Ethernet utilization is low.

### Primary Lever: Data Format — BFP8_b vs. BF16

Switching expert weight tiles from BF16 to BFP8_b (block floating-point 8-bit) reduces the
tile compute cost on Wormhole B0 Tensix cores [UNVERIFIED exact speedup factor; may be up to
~2× for matmul-bound ops]. BFP8_b also halves the DRAM read bandwidth for weight tiles,
providing an additional benefit in near-boundary cases.

```python
# Request BFP8_b data format for the expert weight tensor
expert_weights = ttnn.to_dtype(expert_weights, ttnn.bfp8_b)
```

Verify PCC > 0.99 after format change.

### Secondary Lever: Increase per_core_M / per_core_N Subblock Sizes

Larger subblock sizes improve temporal reuse of weight tiles within the Tensix core's local
registers, reducing the ratio of memory-fetch cycles to compute cycles. Increase `per_core_M`
in the matmul configuration until L1 CB pressure becomes the limiting factor (see Section 5).

```python
matmul_config = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=grid_size,
    in0_block_w=...,
    out_subblock_h=...,
    out_subblock_w=...,
    per_core_M=4,   # increase from default; watch for L1 pressure
    per_core_N=4,
)
```

### Locating the Compute Bottleneck

Not all matmuls in the MoE layer are equally expensive. The router projection uses weight matrix
$W_r$ of shape $[H, E \cdot k / N]$; for Qwen3.5-35B this is approximately 3.67 MB — small
enough to be fast. The expert FFN matmul involves much larger weight tensors and is almost always
the compute bottleneck if the workload is compute-bound. Confirm by comparing profiler
`device_time_ns` for the router matmul vs. the expert FFN matmul.

---

## Section 5: L1-Pressure Remediation

**Indicator:** Either (a) `ttnn.exceptions.MemoryAllocationError` at compile time, or (b)
DRAM read BW during expert FFN matmul significantly exceeds the expected weight-streaming
baseline, indicating that activation tensors expected to be L1-resident are spilling to DRAM.

### Expected L1 CB Footprint Check

Compute the combined size of all circular buffers that must coexist during the expert FFN
matmul on a single Tensix core:

| Tensor | Shape | BF16 Size | Fits in 1.5 MB L1? |
|---|---|---|---|
| Expert activation CB | $[C, H] = [2, 7168]$ | 28 KB | Yes |
| Expert weight CB (one tile column) | Streaming; not fully resident | N/A | N/A (streamed) |
| Output CB | $[C, D]$, streaming | Depends on $D$ [UNVERIFIED] | Depends |

At $C=2$, the expert activation tensor is 28 KB — far below the 1.5 MB limit — so activation
spill at decode is unlikely unless `per_core_M` has been set very large or multiple tensors are
co-resident. If L1-pressure is observed at decode, first check whether unnecessary intermediate
tensors are being kept in L1 across op boundaries.

### Primary Lever: Reduce per_core_M

Reducing `per_core_M` shrinks the activation CB size proportionally. Halving `per_core_M`
halves the CB footprint at the cost of requiring twice as many core-iterations to process the
same problem size. The latency impact depends on whether the core grid is already fully utilized.

### Secondary Lever: Move Intermediate Tensors to DRAM

Tensors that are computed once and read back only once (non-reused intermediates) do not benefit
from L1 placement. Move them to DRAM:

```python
intermediate = ttnn.to_memory_config(
    intermediate,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

See `ch04_memory_config/decode_memory_strategy.md` for guidance on which tensors are candidates
for DRAM placement without significant latency impact.

---

## Section 6: Common Anti-Patterns in MoE Workloads on T3K

The following anti-patterns are frequently encountered when tuning MoE inference on T3K. Each
represents a mistake that appears reasonable without profiling data but is harmful in practice.

**1. Tuning `num_links` without profiling first.**
`num_links` is the most visible tuning parameter and the easiest to change. However, if the
workload is memory-bandwidth-bound or compute-bound, changing `num_links` has no positive
effect on latency and may increase link setup overhead. Profile first; change `num_links` only
if the profiler confirms communication-bound behavior.

**2. Using L1 placement for prefill activations.**
The full hidden-state activation tensor during prefill has shape $[B \cdot S, H]$. At
$B=4$, $S=512$, $H=7168$ in BF16 this is approximately 29 MB — far exceeding the 120 MB
aggregate L1 per chip but also exceeding what can be sharded efficiently. DRAM interleaved
placement is mandatory for prefill activations beyond short sequences.

**3. Not enabling sparse matmul at decode.**
At decode, only $C=1$–$2$ tokens are routed to each expert's device out of $E_d=32$ expert
slots. Running a batched matmul over all 32 expert slots wastes approximately 97% of compute
on zero-token slots. Use `sparse_matmul` or equivalent to skip experts with no assigned tokens.

**4. Running a batched matmul at decode with capacity $C=2$.**
With $k=8$ and $E=256$ at $B=32$, the theoretical capacity is $C = \lceil k \cdot B / E \rceil = 2$.
Of the 32 expert slots per device, at most 2 receive any tokens. Running all 32 in a batched
matmul means 94% of the compute is padding. This is distinct from anti-pattern 3: even if
`sparse_matmul` is used, ensure that the capacity tensor correctly reflects the actual occupied
slots rather than the maximum possible.

**5. Forgetting to update the sparsity tensor between decode steps.**
If the top-$k$ routing changes between decode steps (as it does for autoregressive generation),
the sparsity mask that indicates which expert slots are occupied must be recomputed each step.
Reusing a stale sparsity tensor from step $t-1$ produces silent correctness errors: the wrong
expert outputs are included in the accumulation. This does not raise a runtime error and will
not be caught unless PCC is checked against a reference at every step.

---

## References

- `ttnn_profiler.md` — TTNN profiler output format and parsing (prerequisite)
- `device_perf_counters.md` — Counter access and interpretation (prerequisite)
- `ch03_all_to_all_num_links/num_links_parameter.md` — `num_links` analysis and hardware maximum
- `ch04_memory_config/decode_memory_strategy.md` — L1 budget estimation and tensor placement
- `ch04_memory_config/prefill_memory_strategy.md` — Prefill memory constraints
- `ch05_expert_parallelism/expert_placement_strategies.md` — Expert placement as communication lever
- Wormhole B0 architecture specification (internal) — Tensix BFP8_b throughput [UNVERIFIED]

---

**Next:** [Chapter 7 — End-to-End Integration](../ch07_end_to_end_integration/index.md)
