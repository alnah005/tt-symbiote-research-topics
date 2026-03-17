# Trade-off Matrix

## Four-Regime Comparison

The table below consolidates the recommendations from `bandwidth_gain_analysis.md` and `shard_setup_overhead.md` into a single reference. Regimes are defined by the combination of inference phase (decode vs prefill) and effective batch size.

`effective_M = batch_size × top_k` is the number of token-expert computation pairs per forward pass. It determines arithmetic intensity and therefore whether the kernel is memory-bound or compute-bound.

| Regime | Definition | Recommended weight layout | Primary bottleneck | Indicative latency delta (sharded vs interleaved) |
|---|---|---|---|---|
| (a) Decode, small batch | `effective_M ≤ 16` | **DRAM-sharded** | Memory bandwidth | −30 to −50% |
| (b) Decode, large batch | `16 < effective_M ≤ 64` | **DRAM-sharded** | Memory bandwidth (moderate) | −10 to −25% |
| (c) Prefill, small batch | `64 < effective_M ≤ 256` | DRAM-sharded (marginal benefit) | Mixed (transitioning) | −5 to −10% |
| (d) Prefill, large batch | `effective_M > 256` | **Interleaved** | Compute | 0 to +5% (overhead, no gain) |

A negative latency delta indicates the sharded layout is faster. A positive delta indicates interleaved is preferable.

> **Note:** These are indicative estimates. The exact crossover between regimes (b) and (c) depends on `d_ff`, `d_model`, and the specific Tensix compute throughput achieved by the matmul kernel. See Chapter 4, `bandwidth_estimation.md` for the roofline derivation.

---

## Model-Specific Regime Boundaries

Different MoE models have different `d_ff` values, which shifts the arithmetic intensity crossover. Using the formula from `bandwidth_gain_analysis.md`:

**Mixtral 8x7B** (`d_model=4096`, `d_ff=14336`, `top_k=2`):

| Batch size | `effective_M` | Regime | Recommendation |
|---|---|---|---|
| 1 | 2 | Decode, small | DRAM-sharded (−40 to −50%) |
| 4 | 8 | Decode, small | DRAM-sharded (−35 to −45%) |
| 8 | 16 | Decode, small | DRAM-sharded (−30 to −40%) |
| 32 | 64 | Decode, large | DRAM-sharded (−15 to −25%) |
| 128 | 256 | Prefill, small | Marginal (−5 to −10%) |
| 256+ | 512+ | Prefill, large | Interleaved (sharding offers no benefit) |

**Qwen 235B-A22B** (`d_model=7168`, `d_ff=2048`, `top_k=8`):

| Batch size | `effective_M` | Regime | Recommendation |
|---|---|---|---|
| 1 | 8 | Decode, small | DRAM-sharded (−30 to −45%) |
| 2 | 16 | Decode, small | DRAM-sharded (−25 to −40%) |
| 8 | 64 | Decode, large | DRAM-sharded (−10 to −20%) |
| 32 | 256 | Prefill, small | Marginal (−3 to −8%) |
| 64+ | 512+ | Prefill, large | Interleaved preferred |

> **Tip:** Qwen 235B-A22B has smaller `d_ff=2048` relative to its `d_model=7168`. The weight matrix for each expert's gate/up projection is `[7168, 2048]` — relatively narrow. The arithmetic intensity crossover occurs at a lower effective_M than Mixtral, meaning Qwen reaches compute-bound behavior at smaller batches. This makes DRAM sharding's high-impact window narrower in terms of batch size, but decode serving with small batches still falls firmly within it.

---

## When Sharding Hurts

DRAM sharding introduces overhead with no offsetting bandwidth benefit in the following conditions:

**1. Large-batch prefill with compute saturation**

When `effective_M > 256` and the kernel is compute-bound, the matmul kernel spends the majority of its time in Tensix multiply-accumulate units, not waiting for DRAM. Improving DRAM bandwidth from ~200 GB/s to ~270 GB/s does not change the compute-bound portion of kernel time. The TTNN dispatch overhead of resolving shard-to-bank mappings (tens of microseconds) adds latency without any reduction in DRAM wait time.

**2. Mismatched shard grid and active core grid**

If the sharded `MemoryConfig` specifies a 1×8 shard grid but the downstream `ttnn.matmul` is configured to use a 4×10 compute grid, the cores outside the shard grid must fetch weight tiles via indirect NoC routing rather than direct shard access. In this case the sharded layout provides no locality benefit and may perform worse than interleaved due to uneven NoC traffic patterns.

> **Warning:** Always verify that the `CoreRangeSet` in the weight's `ShardSpec` is compatible with the compute core grid used by the downstream `ttnn.matmul`. Incompatible grids are not always flagged as errors by TTNN; they can silently degrade performance. Chapter 7, `benchmark_methodology.md` describes how to verify core-grid alignment via Tracy profiling.

**3. Shard width narrower than `in0_block_w`**

Chapter 5, `common_pitfalls.md` Pitfall 3 identified that non-power-of-2 shard widths which are multiples of 32 but smaller than the matmul `in0_block_w` cause suboptimal tiling. In this case the matmul kernel cannot fill its inner-loop tile block from a single shard and must issue additional reads from adjacent shards, increasing DMA scheduling overhead. The bandwidth improvement from sharding is partially cancelled.

---

## Interaction with L1 Memory Pressure

The matmul kernel on Wormhole B0 can operate in two modes for weight fetching:

**Mode 1: Direct DRAM-sharded access (preferred when feasible)**

The matmul kernel reads weight tiles directly from DRAM-sharded locations via DMA, double-buffering in L1. Only the current and next weight tile-blocks are resident in L1 simultaneously. L1 working set for weights is:

```
L1_weight_footprint = 2 × in0_block_w × per_core_N_t × tile_size_bytes
```

where `in0_block_w` is the number of K-direction tiles per block and `per_core_N_t` is the number of N-direction tiles per core shard. For Mixtral gate projection with `in0_block_w=8 tiles`, `per_core_N_t=1` tile, BF16 (`tile_size_bytes=2048`):

```
L1_weight_footprint = 2 × 8 tiles × 1 × 2048 bytes = 32 KB per core
```

This fits comfortably within the 1.5 MB L1 per core on Wormhole B0. In practice, `in0_block_w` can be increased (e.g., to 2–4 or more tiles) while still fitting within L1 constraints. Direct DRAM-sharded access is feasible when the double-buffer footprint fits.

**Mode 2: Explicit DRAM-to-L1 reshard before matmul (fallback)**

When the weight matrix is too large to stream directly from DRAM, a separate `ttnn.to_memory_config` call moves the active shard to L1 before the matmul:

```python
# Explicitly stage the weight shard in L1 for the matmul.
weight_l1 = ttnn.to_memory_config(weight_dram_sharded, l1_sharded_config)
output = ttnn.matmul(activation, weight_l1, ...)
ttnn.deallocate(weight_l1)
```

This pattern is used when the DRAM-sharded layout provides bandwidth benefits for the DRAM-to-L1 transfer but the matmul kernel itself requires L1-resident weights. The DRAM→L1 reshard is inside the inference loop but transfers only the active shard (not the full weight tensor), so the cost is proportional to shard size rather than full weight size.

**L1 pressure implications by regime:**

| Regime | Preferred mode | L1 headroom | Notes |
|---|---|---|---|
| Decode, small batch | Direct DRAM-sharded | High (small activation) | Activation is tiny (M=1); L1 mostly available for weight streaming |
| Decode, large batch | Direct DRAM-sharded | Moderate | Monitor L1 with `ttnn.device.EnableMemoryReports()` |
| Prefill, small batch | DRAM→L1 reshard may be needed | Low (large activation in L1) | Activation tiles compete with weight tiles for L1 |
| Prefill, large batch | Interleaved preferred | Very low | L1 pressure high; avoid extra reshard ops |

---

## Multi-Expert Parallelism Interaction (T3K)

On a T3K mesh (8 Wormhole chips connected via Ethernet links), MoE expert parallelism distributes different experts to different chips. For Mixtral 8x7B with 8 experts on 8 chips, each chip holds the weights for one expert.

**Within-chip DRAM sharding compounds with across-chip expert sharding:**

- Across-chip: each chip holds 1 expert's weights (expert parallelism).
- Within-chip: the one expert's weight tensor is DRAM-sharded across 6 controllers on that chip.

This two-level sharding does not conflict; they operate on different dimensions:

| Level | Dimension sharded | Mechanism | Config object |
|---|---|---|---|
| Across chips (T3K) | Expert index | Distributed tensor placement, one expert per chip | Device mesh assignment |
| Within chip (Wormhole B0) | Weight matrix rows/columns | `TensorMemoryLayout.WIDTH_SHARDED` or `HEIGHT_SHARDED` | `ttnn.MemoryConfig` with `ShardSpec` |

The two levels are independent. Enabling within-chip DRAM sharding on T3K has the same bandwidth benefit per chip as on a single Wormhole B0. The expert parallelism routing (sending each token to the chip holding its selected expert) is handled at a higher level and is not affected by the per-chip weight layout.

**Combined configuration guidance for T3K:**

1. Assign experts to chips as you normally would for expert parallelism (one or more experts per chip depending on `num_experts` and chip count).
2. On each chip, apply the within-chip DRAM-sharding configuration from `shard_setup_overhead.md` to the expert weights resident on that chip.
3. Use the same `decode-regime rule` (`batch_size × top_k ≤ 16`, where `batch_size` is the per-chip token count after expert routing) to decide whether to shard on each chip.

> **Tip:** On T3K, each chip receives a subset of the total batch (those tokens routed to that chip's expert). If the global batch is 64 tokens and 8 experts share the load, each chip receives approximately `64 × top_k / num_experts = 64 × 2 / 8 = 16` token-expert pairs on average. This is exactly at the `effective_M ≤ 16` boundary — T3K expert-parallel deployment is nearly always in the DRAM-sharding benefit zone for moderate batch sizes.

**When T3K compounding creates a risk:** If `num_experts` is large (e.g., Qwen 235B-A22B with 128 experts) but only 8 chips are available, each chip holds 16 experts. The per-chip weight memory is 16× larger than the single-expert case. With 16 experts per chip, the within-chip DRAM sharding must now accommodate multiple expert weight tensors. Ensure that total DRAM capacity is not exceeded and that the shard grid for each expert tensor is compatible with the available DRAM banks.

---

## Summary: Layout Selection Flowchart

```
Is effective_M = batch_size × top_k?
│
├── effective_M ≤ 16  →  USE DRAM-SHARDED (high impact, 30–50% gain)
│
├── 16 < effective_M ≤ 64  →  USE DRAM-SHARDED (moderate impact, 10–25% gain)
│
├── 64 < effective_M ≤ 256  →  DRAM-SHARDED with diminishing returns (5–10% gain)
│                                Profile before committing to this regime.
│
└── effective_M > 256  →  USE INTERLEAVED
                            Sharding adds overhead; compute is bottleneck.
                            Exception: if L1 pressure is extreme and DRAM→L1
                            reshard helps manage working set, profile both options.
```

---

## Next Steps

Chapter 6 has established what bandwidth gains to expect from DRAM-sharded expert weights and under which conditions sharding is beneficial. Chapter 7, `index.md` provides the complete implementation: runnable code patterns for constructing the sharded configs, integrating with `ttnn.matmul`, verifying correctness via PCC, and measuring bandwidth with the Tracy profiler to validate the indicative estimates presented in this chapter against your specific hardware and TTNN version.
