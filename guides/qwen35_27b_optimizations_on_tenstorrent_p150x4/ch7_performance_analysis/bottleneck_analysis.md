# Bottleneck Analysis

This section breaks down where time is spent during decode, traces each bottleneck to its root cause, and catalogs the remaining optimization opportunities.

## Profiler Breakdown

A non-traced decode step shows the following timing distribution:

```
GDN layers (48):     469.6 ms  (85%)  avg = 9.78 ms/layer
Attn layers (16):     69.2 ms  (12%)  avg = 4.33 ms/layer
Overhead:             15.7 ms   (3%)
Total:               554.5 ms
```

GDN layers are 2.26x more expensive per layer than full attention layers (9.78 ms vs 4.33 ms). Combined with the 3:1 ratio of GDN to attention layers, GDN accounts for 85% of total decode time.

## Root Cause: DRAM Bandwidth for Recurrence State

The per-layer cost difference between GDN and attention is dominated by recurrence state I/O. As derived in [`performance_summary.md`](./performance_summary.md), each GDN layer reads and writes 12 MB of state per decode step. This state transfer happens through interleaved DRAM, which lacks the bandwidth optimization of the DRAM-sharded layout used for weight matrices.

By contrast, attention layers use paged KV caches that are already stored in DRAM-sharded configurations. The per-head cache update via `paged_update_cache` (Chapter 2) writes only 32x256 per core on an 8x4 grid, which is a fraction of the GDN state size.

Within the fused GDN kernel, the NOC traffic for state is:

- **Read:** 16 tile reads per pair (4x4 state matrix), 384 pairs = 6,144 NOC read transactions per layer
- **Write:** 16 tile writes per pair, 384 pairs = 6,144 NOC write transactions per layer
- **Total:** 12,288 NOC transactions per GDN layer, 48 layers = ~590,000 NOC transactions per decode step just for state

The reader kernel batches all 44 reads per pair before a single `noc_async_read_barrier()` (Chapter 4), which amortizes barrier overhead. But the fundamental bandwidth cost of moving 12 MB through the NOC per layer remains.

## Remaining Optimization Opportunities

### 1. L1 State (Work In Progress)

The highest-impact remaining optimization. Detailed in Chapter 6.

**Approach:** Move GDN recurrence states from DRAM to L1 using a rolling window of 3 layers. When HEIGHT_SHARDED, state tiles are local to compute cores -- the reader and writer kernels use direct L1 memcpy instead of NOC reads/writes, eliminating all 12,288 NOC state transactions per layer.

**Current status:**

| Configuration | Validated Layers | Status |
|---|---|---|
| DRAM state (baseline) | All 48 | Full correctness, 14.6 tok/s/user |
| L1 INTERLEAVED | 4 layers | Correct output; clean allocate/deallocate cycle |
| HEIGHT_SHARDED | 1-2 layers | Correct output; SDPA conflict blocks scaling |

**Remaining blocker:** SDPA circular buffers expand to approximately 1,264 KB per core during the 1-in-4 attention layers, overlapping with HEIGHT_SHARDED GDN state addresses. L1 INTERLEAVED avoids this by fully deallocating L1 tensors before attention layers, but sacrifices the direct-memcpy benefit. See [`sdpa_l1_conflict.md`](../ch6_l1_state_management/sdpa_l1_conflict.md) for the full analysis.

**Expected impact:** Eliminating NOC state I/O would remove the dominant per-layer cost difference between GDN and attention layers. The residual GDN cost would be driven by the Q/K/V/scalar DRAM reads and output writes, which are inherently needed and much smaller than the full state.

### 2. Further Kernel Fusion

The fused GDN kernel (Chapter 4) already combines L2 norm, gate computation, and DeltaNet recurrence into a single dispatch. Two additional fusion opportunities remain:

- **RMS norm + SiLU fusion:** The post-recurrence path currently runs `ttnn.rms_norm` and then SiLU gating with the Z tensor as separate dispatches after the fused kernel returns. Folding these into the compute kernel's output phase would eliminate two additional kernel dispatches per GDN layer (96 dispatches per step across 48 layers).

- **Conv1d fusion:** The 4-tap causal conv1d shift register (Chapter 3) runs as separate `ttnn.multiply` and `ttnn.mac` operations before the fused kernel. Incorporating the conv1d computation into the reader or compute phase would remove another set of dispatches and avoid materializing intermediate conv results in DRAM.

### 3. Conv1d Shift Register Overhead

The conv1d shift register implementation uses 4 `ttnn.copy` operations per layer for the state shift, plus 1 `ttnn.multiply` (first tap) and 3 `ttnn.mac` operations (taps 1-3) for the weighted sum. While each operation is small, the dispatch overhead across 48 GDN layers adds up. The total dispatch count for conv1d alone is `48 * (4 copies + 1 multiply + 3 macs) = 384` dispatches per decode step. Fusing conv1d into the main kernel (item 2 above) would eliminate all of these.

### 4. Prefill GDN Sequential Bottleneck

GDN prefill computes projections in a single batched pass but must execute the recurrence loop sequentially -- each token's state update depends on the previous token's output. This is a fundamental constraint of the DeltaNet recurrence equation:

```
state[t] = exp(g[t]) * state[t-1] + outer(k[t], delta[t])
```

The sequential dependency means prefill GDN cost scales linearly with sequence length. For long sequences, this becomes the dominant TTFT component. Potential approaches to reduce this cost:

- **Chunked parallel recurrence:** Algorithms that break the sequence into chunks and process chunks in parallel using associative scan properties of linear recurrence. This requires reformulating the recurrence as a scan operation, which is possible for the linear case but adds implementation complexity.

- **Reduced precision for prefill states:** Using BFP8 for the B=1 prefill recurrence state would halve state bandwidth during the sequential loop. The final state is replicated to B=32 at full precision after prefill completes, so prefill-only precision reduction would not affect decode quality.

### 5. Precision Trade-offs

The current configuration uses:

- **BFP8 weights with HiFi2** for all projection matmuls (decode and prefill)
- **HiFi4** for the fused recurrence kernel (FP32 dest accumulation enabled)
- **Bfloat8_b typecast** before SDPA for memory efficiency during prefill

The recurrence uses HiFi4 because the iterative state update accumulates numerical error across tokens. Reducing this to HiFi2 would improve compute throughput at the cost of potential output quality degradation over long sequences. Any precision change in the recurrence path would require careful validation of output quality across sequence lengths.

## Summary

| Bottleneck | Share of Decode | Root Cause | Optimization Path |
|---|---|---|---|
| GDN state DRAM I/O | ~85% (dominant within GDN) | 12 MB read+write per layer via NOC | L1 state with HEIGHT_SHARDED (WIP) |
| Post-recurrence dispatches | Part of GDN 85% | Separate RMS norm + SiLU kernel launches | Fuse into compute kernel |
| Conv1d dispatches | Part of GDN 85% | 8 ttnn ops per layer * 48 layers | Fuse into fused GDN kernel |
| Prefill GDN sequential loop | Dominant TTFT component for long sequences | DeltaNet recurrence dependency | Chunked parallel recurrence |
| Attention layers | 12% | Already well-optimized (DRAM-sharded, flash SDPA) | Limited further opportunity |

The single highest-impact optimization is completing the L1 state work from Chapter 6. The DRAM bandwidth bottleneck for GDN recurrence state is the dominant cost in the decode path, and HEIGHT_SHARDED L1 state has already been validated at small scale. Resolving the SDPA circular buffer conflict is the remaining step to enable this optimization across all 48 GDN layers.

---

**Previous:** [`performance_summary.md`](./performance_summary.md)

---

**End of guide.** Return to [Guide Index](../index.md)
