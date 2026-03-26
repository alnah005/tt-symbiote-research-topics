# BailingAttention Standalone Profiling Analysis

**Date:** 2026-03-26
**Data source:** `bailing_attention_standalone_timing_stats.csv` and `bailing_attention_standalone_timing_stats_pivot.csv`
**Code reference:** `models/experimental/tt_symbiote/modules/attention.py`, method `_forward_decode_paged` (lines 2610-2802)

---

## Executive Summary

The BailingAttention decode path takes **~9.7ms per iteration** at steady state (wall clock). Only **24.5%** of that time is spent in sub-module device compute (matmuls, RMSNorm). The dominant cost (**70.1%**) is **inline attention ops** -- all_gather, to_memory_config resharding, RoPE, host round-trips (`_to_replicated`), paged SDPA, and head manipulation ops that are not individually instrumented in the profiling data but collectively account for ~6.8ms per decode step.

---

## 1. Dataset Overview

- **Total profiling rows:** 1,856
- **Total forward iterations:** 33 (1 prefill + 32 decode)
- **Steady-state decode iterations (20-32):** 13 iterations, after warmup effects subside
- **Measurement backends:** TTNN (device ops), Torch (host overhead), TorchModules (module wrappers)

### Time Distribution by Backend (all 33 iterations)

| Backend       | Total Time (s) | Fraction | Call Count |
|---------------|----------------|----------|------------|
| TorchModules  | 0.717          | 49.0%    | 232        |
| TTNN          | 0.633          | 43.3%    | 696        |
| Torch         | 0.112          | 7.7%     | 928        |
| **Grand Total** | **1.463**    | **100%** | **1,856**  |

Note: TorchModules entries are hierarchical wrappers that encompass the TTNN forward calls. The numbers overlap; the true wall-clock is given by the outermost `TTNNBailingMoEAttention` TorchModules entry.

---

## 2. Per-Op Breakdown Table (sorted by total duration, all iterations)

| Rank | Op Name (func_name)                     | Total (s) | Count | Avg/call (ms) | Max (ms) | Backend      |
|------|------------------------------------------|-----------|-------|---------------|----------|--------------|
| 1    | TTNNBailingMoEAttention (wrapper)        | 0.5295    | 33    | 16.04         | 28.75    | TorchModules |
| 2    | TTNNBailingMoEAttention_forward          | 0.4993    | 33    | 15.13         | 27.77    | TTNN         |
| 3    | TTNNLinearIReplicatedWColSharded (wrapper)| 0.0923   | 99    | 0.93          | 1.43     | TorchModules |
| 4    | wrap_to_torch...set_device_wrap          | 0.0688    | 464   | 0.15          | 0.82     | Torch        |
| 5    | TTNNLinearIReplicatedWColSharded_forward  | 0.0653    | 99    | 0.66          | 1.06     | TTNN         |
| 6    | TTNNLinearIColShardedWRowSharded (wrapper)| 0.0523   | 33    | 1.58          | 5.11     | TorchModules |
| 7    | TTNNLinearIColShardedWRowSharded_forward  | 0.0436    | 33    | 1.32          | 4.91     | TTNN         |
| 8    | TTNNRMSNorm (wrapper)                    | 0.0425    | 66    | 0.64          | 0.97     | TorchModules |
| 9    | wrap_to_torch_ttnn_tensor                | 0.0379    | 232   | 0.16          | 0.30     | Torch        |
| 10   | TTNNRMSNorm_forward                      | 0.0239    | 66    | 0.36          | 0.54     | TTNN         |
| 11   | _set_distributed_config                  | 0.0055    | 232   | 0.02          | 0.11     | Torch        |
| 12   | TTNNSDPAAttention (wrapper)              | 0.0006    | 1     | 0.64          | 0.64     | TorchModules |
| 13   | TTNNSDPAAttention_forward                | 0.0003    | 1     | 0.33          | 0.33     | TTNN         |

---

## 3. Decode Path Bottleneck Ranking (Steady-State, iter 20-32)

### Wall-Clock Breakdown

| Component                      | Time/iter (ms) | Fraction |
|--------------------------------|----------------|----------|
| **Total wall clock**           | **9.72**       | **100%** |
| Forward (TTNN)                 | 9.20           | 94.7%    |
|   - Sub-module device compute  | 2.38           | 24.5%    |
|   - Inline attention ops       | 6.82           | 70.1%    |
| Outer module overhead          | 0.52           | 5.3%     |

### Sub-Module Device Compute Breakdown (2.38ms)

| Sub-module                               | Role           | Time/iter (ms) | Per-call (ms) |
|------------------------------------------|----------------|----------------|---------------|
| TTNNLinearIReplicatedWColSharded (3 inst) | K/V/Dense proj | 1.13           | 0.38          |
| TTNNLinearIColShardedWRowSharded (1 inst) | Q proj (fused) | 0.72           | 0.72          |
| TTNNRMSNorm (2 inst)                     | QK norm        | 0.34           | 0.17          |
| Paged SDPA decode                        | (not instrumented separately -- included in inline ops) | -- | -- |

### Inline Attention Ops Breakdown (6.82ms) -- NOT individually instrumented

These ops are called directly in `_forward_decode_paged` and measured only in aggregate. Based on code analysis, they include:

| Op Category               | Count/iter | Estimated Cost | Notes |
|---------------------------|------------|----------------|-------|
| `all_gather`              | 4          | ~2.0-3.0ms     | 1x hidden_states + 3x Q/K/V via `_maybe_all_gather` |
| `to_memory_config`        | 8          | ~1.5-2.5ms     | Resharding for RoPE, KV cache, SDPA output |
| `_to_replicated`          | 1          | ~1.0-2.0ms     | **HOST ROUND-TRIP**: device->host->device |
| `paged_sdpa_decode`       | 1          | ~0.5-1.0ms     | Core attention compute |
| `rotary_embedding_llama`  | 2          | ~0.3-0.5ms     | RoPE for Q and K |
| `nlp_create_qkv_heads_decode` | 1     | ~0.1-0.3ms     | Head splitting |
| `nlp_concat_heads_decode` | 1          | ~0.1-0.2ms     | Head merging |
| `concat`                  | 1          | ~0.1-0.2ms     | Fuse Q/K/V before head split |
| `reshape` + `typecast` + `slice` + `deallocate` | ~8 | ~0.3-0.5ms | Misc data manipulation |

### Host Overhead Per Iteration

| Category                           | Time/iter (ms) | Description |
|------------------------------------|----------------|-------------|
| `wrap_to_torch_ttnn_tensor`        | ~1.0           | Tensor wrapper conversion (7 calls/iter) |
| `wrap_to_torch...set_device_wrap`  | ~0.7           | Device placement wrappers (14 calls/iter) |
| `_set_distributed_config`          | ~0.2           | Distributed config setup (7 calls/iter) |
| Sub-module wrapper overhead        | ~1.5           | TorchModules __call__ overhead (6 sub-modules) |
| **Total host overhead**            | **~3.4**       | **~35% of wall clock** |

---

## 4. Key Findings

### Finding 1: Inline Attention Ops Dominate (70% of forward time)

The 6.82ms spent in inline attention ops is the single largest contributor. These ops -- all_gather, to_memory_config, _to_replicated, RoPE, SDPA -- are called directly in `_forward_decode_paged` and not captured individually by the profiler. A device-level Tracy trace would be needed to break down the exact time per op.

### Finding 2: `_to_replicated` Host Round-Trip is a Known Bottleneck

The `_to_replicated` method (line 2288-2310) does a full **device -> host -> device** round-trip via `ttnn.to_torch()` + `ttnn.from_torch()`. The code comment says "negligible overhead" but for decode tokens this likely costs 1-2ms per iteration. This exists because after `all_gather`, the mesh topology metadata differs from `ReplicateTensorToMesh`, and paged-attention kernels require the replicated topology.

### Finding 3: 4 All-Gather Ops Per Decode Iteration

Each decode iteration performs 4 all-gather operations:
1. `ttnn.all_gather(hidden_states, dim=-1, num_links=1)` -- line 2626
2. `self._maybe_all_gather(query_states)` -- line 2631
3. `self._maybe_all_gather(key_states)` -- line 2632
4. `self._maybe_all_gather(value_states)` -- line 2633

Each all_gather synchronizes across all 8 devices in the T3K mesh. Estimated total cost: 2-3ms per iteration.

### Finding 4: 8 `to_memory_config` Resharding Ops Per Decode Iteration

The decode path reshards tensors between DRAM, L1, and HEIGHT_SHARDED configurations:
1. Q -> L1 (line 2659)
2. K -> L1 (line 2660)
3. cos -> HEIGHT_SHARDED (line 2711)
4. sin -> HEIGHT_SHARDED (line 2712)
5. Q -> HEIGHT_SHARDED for RoPE (line 2721)
6. K -> HEIGHT_SHARDED for RoPE (line 2730)
7. K -> kv_mem HEIGHT_SHARDED (line 2756)
8. V -> kv_mem HEIGHT_SHARDED (line 2757)
9. attn_output -> HEIGHT_SHARDED (line 2786)

Total: **9 to_memory_config calls**, estimated 1.5-2.5ms combined.

### Finding 5: Actual Compute is a Small Fraction

True compute ops (matmul in Linear projections, SDPA attention, RoPE rotary embedding) account for roughly:
- Linear projections: 2.38ms (24.5%)
- SDPA + RoPE: ~1-1.5ms (estimated 10-15%)
- **Total compute: ~35-40% of wall clock**
- **Data movement + overhead: ~60-65% of wall clock**

### Finding 6: Warmup Effect is Significant

| Phase          | Avg Decode Time |
|----------------|----------------|
| Iterations 0-5 | 13.0ms         |
| Iterations 5-15| 14.2ms         |
| Iterations 20-32| 9.2ms (forward) / 9.7ms (wall) |

The ~30% improvement from warmup to steady-state suggests TTNN op caching and JIT compilation are effective. This also means **trace capture** would benefit this workload by avoiding repeated dispatch overhead.

---

## 5. Prefill vs Decode Characteristics

| Characteristic          | Prefill (Iter 0)     | Decode (Steady-State) |
|-------------------------|----------------------|-----------------------|
| Wall time               | ~8.2ms               | ~9.7ms                |
| Forward TTNN time       | ~7.3ms               | ~9.2ms                |
| Sub-module compute      | ~3.6ms (49%)         | ~2.4ms (25%)          |
| Inline ops              | ~3.7ms (51%)         | ~6.8ms (70%)          |
| Q proj                  | 0.82ms               | 0.72ms                |
| K/V proj                | 0.73ms               | 0.71ms                |
| QK RMSNorm              | 0.74ms               | 0.34ms                |
| Output Dense            | 0.93ms               | 0.44ms                |
| SDPA (prefill)          | instrumented (0.33ms) | not instrumented      |
| `_to_replicated`        | not used              | 1x host round-trip    |
| `all_gather`            | 4x                   | 4x                    |
| `to_memory_config`      | fewer (~3-4x)        | 9x (HEIGHT_SHARDED setup) |

Key differences:
- **Prefill is actually faster** for this small sequence length (1 token prefill), but decode has more overhead from paged attention setup, HEIGHT_SHARDED memory configs, and the `_to_replicated` round-trip.
- Prefill uses standard SDPA; decode uses paged SDPA with additional cache management ops.
- Decode requires significantly more to_memory_config resharding operations for the HEIGHT_SHARDED decode-mode ops.

---

## 6. Optimization Recommendations

### Priority 1: Eliminate `_to_replicated` Host Round-Trip
**Expected impact: -1 to 2ms per decode iteration (10-20% improvement)**

The `_to_replicated` method round-trips through the host solely to change mesh topology metadata. Options:
1. **Use `ttnn.replicate()` on-device** if such an API exists, to set replicated topology without host transfer.
2. **Restructure the K/V projection** to produce replicated-topology tensors directly, avoiding the need for post-hoc topology conversion.
3. **Use `all_gather` with `ReplicateTensorToMesh` semantics** so the result already has the correct topology.

### Priority 2: Reduce All-Gather Count from 4 to 1
**Expected impact: -1.5 to 2ms per decode iteration (15-20% improvement)**

Currently, `hidden_states` is all-gathered once, then Q/K/V are each all-gathered separately. Instead:
1. **Fuse Q/K/V projection into a single matmul** before the all-gather, then all-gather the fused result once.
2. The fused QKV projection already exists (`TTNNLinearIColShardedWRowSharded` for q_proj), but K/V use separate replicated projections that require pre-gathered input and then gather their output.
3. Redesign: single fused QKV matmul with column-parallel sharding -> 1 all_gather -> split heads. This eliminates 3 of 4 all-gathers.

### Priority 3: Reduce `to_memory_config` Resharding Ops
**Expected impact: -0.5 to 1ms per decode iteration (5-10% improvement)**

9 resharding operations is excessive. Options:
1. **Pre-compute sharded memory configs** and reuse them (some may already be cached, but creation overhead may still exist).
2. **Keep Q/K in L1 throughout** rather than moving to DRAM and back.
3. **Fuse resharding with compute ops** -- e.g., configure RoPE to accept DRAM-interleaved input directly.
4. **Eliminate Q->L1, K->L1 steps** (lines 2659-2660) if QK norm can operate on HEIGHT_SHARDED tensors.

### Priority 4: Enable Trace Capture for Decode
**Expected impact: -2 to 3ms per decode iteration (20-30% improvement)**

The warmup curve (13ms -> 9.2ms) shows significant dispatch overhead that trace capture could eliminate. With trace:
- All host dispatch overhead (wrap_to_torch, set_distributed_config, module __call__) would be eliminated.
- The ~3.4ms of host overhead per iteration would drop to near-zero.
- This is the highest-ROI optimization but requires the decode path to have deterministic tensor shapes across iterations.

### Priority 5: Reduce Sub-Module Wrapper Overhead
**Expected impact: -1 to 1.5ms per decode iteration (10-15% improvement)**

Each sub-module call incurs ~0.25ms of wrapper overhead (wrap_to_torch, _set_distributed_config, module dispatch). With 6 sub-modules per iteration, this totals ~1.5ms. Options:
1. **Inline the linear/RMSNorm logic** directly in `_forward_decode_paged` to avoid module call overhead.
2. **Use trace capture** (Priority 4) which would subsume this overhead.

### Priority 6: Fuse QKV Projection
**Expected impact: -0.3 to 0.5ms per decode iteration**

Currently Q, K, V projections are 3 separate matmul calls. A single fused QKV matmul would reduce kernel launch overhead and improve memory access patterns. The code already does `ttnn.concat([Q, K, V])` after separate projections -- doing the fusion at the weight level would be more efficient.

---

## 7. Summary of Expected Gains

| Optimization                          | Est. Savings (ms) | Cumulative Decode Time |
|---------------------------------------|-------------------|------------------------|
| Baseline (steady-state)               | --                | 9.7ms                  |
| Eliminate `_to_replicated`            | 1.0-2.0           | 7.7-8.7ms              |
| Reduce all_gather (4 -> 1)           | 1.5-2.0           | 5.7-7.2ms              |
| Trace capture                         | 2.0-3.0           | 2.7-5.2ms              |
| Reduce to_memory_config resharding    | 0.5-1.0           | 1.7-4.7ms              |
| Fuse QKV + reduce wrapper overhead    | 0.5-1.0           | 1.2-3.7ms              |
| **Best-case total savings**           | **~6-8ms**        | **~2-4ms per decode**  |

Note: Some optimizations overlap (e.g., trace capture subsumes wrapper overhead). The realistic achievable target with all optimizations is approximately **3-5ms per decode iteration**, a 50-70% reduction from the current 9.7ms.

---

## 8. Next Steps

1. **Run Tracy device-level profiling** to get individual op timings for the inline attention ops (all_gather, to_memory_config, RoPE, SDPA, _to_replicated). This would replace the estimates above with actual measurements.
2. **Prototype `_to_replicated` elimination** by testing if paged_sdpa_decode can accept all_gather-topology tensors directly, or if a device-side topology conversion exists.
3. **Prototype fused QKV matmul** with single all_gather to validate the estimated savings.
4. **Enable trace capture** for the decode path and measure the dispatch overhead reduction.
