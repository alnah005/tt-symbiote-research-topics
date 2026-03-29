# Performance Summary

This section presents the current measured performance of Qwen3.5-27B on the P150x4 platform, compares against the baseline, and documents the test commands used to reproduce each measurement.

## Current Numbers

| Metric | Baseline | Current | Improvement |
|---|---|---|---|
| **Decode throughput** (batch=32, traced) | 14.6 tok/s/user | 14.6 tok/s/user | -- (decode unchanged) |
| **TTFT per-token** (prefill) | 498 ms/token | 94 ms/token | **5.3x** |
| **TTFT 96-token prompt** | ~47.8 s | 9.1 s | **5.3x** |

Decode throughput remains at the baseline of 14.6 tokens per second per user with batch=32 and tracing enabled. The decode path has not regressed from the optimizations applied to prefill, and the fused GDN kernel was already part of the baseline decode configuration.

The TTFT improvement is the primary result of this optimization effort. Per-token prefill latency dropped from 498 ms to 94 ms -- a 5.3x speedup. For a 96-token prompt, total TTFT dropped from 47.8 seconds to 9.1 seconds.

## Per-Layer State Size Arithmetic

Understanding why GDN layers dominate decode time requires tracing the recurrence state size from model dimensions down to bytes. This arithmetic is the foundation for the bandwidth analysis in the next section.

Each GDN layer maintains a recurrence state of shape `[B * Nv_TP, Dk, Dv]` per device:

| Dimension | Value | Source |
|---|---|---|
| `B` (batch size) | 32 | Decode batch |
| `Nv_TP` (value heads per device) | 12 | `Nv / TP = 48 / 4` |
| `Dk` (key dimension) | 128 | Model config |
| `Dv` (value dimension) | 128 | Model config |

**State size per layer per device:**

```
num_pairs = B * Nv_TP = 32 * 12 = 384
state_per_pair = Dk * Dv * sizeof(bfloat16) = 128 * 128 * 2 = 32,768 bytes = 32 KB
state_per_layer = 384 * 32 KB = 12,288 KB = 12.0 MB
```

In tile terms, each pair's state is `Kt * Vt = 4 * 4 = 16` tiles, where each tile is 2048 bytes (32x32 bfloat16). So `384 pairs * 16 tiles * 2048 bytes = 12,582,912 bytes = 12 MB`. No tile padding is needed since both Dk=128 and Dv=128 are exact multiples of the 32-wide tile size.

Every decode step reads the full state from DRAM, passes it through the fused kernel's recurrence phases, and writes the updated state back. For 48 GDN layers:

```
Total state I/O per step = 48 layers * 12 MB * 2 (read + write) = ~1.15 GB
```

This 1.2 GB of DRAM bandwidth per decode step is the single largest contributor to GDN layer latency.

## Completed Optimizations

### Prefill (TTFT)

The 5.3x TTFT improvement comes from three complementary optimizations detailed in Chapter 5:

1. **Flash attention for prefill** -- `ttnn.transformer.scaled_dot_product_attention(is_causal=True)` processes the full sequence in parallel for the 16 attention layers, replacing per-token attention computation.

2. **Batched GDN prefill** -- QKVZ and AB projections are computed once for the full input sequence via 2D matmul (`MatmulMultiCoreReuseMultiCastProgramConfig`). Only the per-token recurrence loop remains sequential, and it operates with B=1 states to minimize memory traffic.

3. **State replication** -- After prefill completes with B=1 states, `replicate_prefill_state_to_batch` expands GDN recurrence states and KV caches from B=1 to B=32 for decode.

### Decode

The decode path uses the following optimizations, all present in the baseline configuration:

- **Full fused GDN kernel** -- Custom C++ kernel fusing L2 norm, gate computation, and DeltaNet recurrence into a single dispatch with reader/compute/writer architecture (Chapter 4)
- **DRAM-sharded matmuls** -- All decode projections use `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` with WIDTH_SHARDED weights across 8 DRAM cores (Chapter 2)
- **BFP8 weights with HiFi2** -- Projection weights stored in BFP8 with `WormholeComputeKernelConfig(math_fidelity=HiFi2)` for bandwidth efficiency
- **CCL Ring topology** -- Sampling all-gather uses Ring topology on P150x4 for inter-chip communication

## Reproducing Measurements

All measurements are taken on a P150x4 (4-chip Blackhole) platform. Reset devices before running:

```bash
tt-smi -r 0,1,2,3
```

**Decode throughput** (batch=32, traced):

```bash
HF_MODEL=~/models/Qwen3.5-27B-FP8 \
    pytest models/demos/qwen35_27b/tt/tests/test_e2e_generate.py::test_e2e_generate_traced -v -s
```

**TTFT measurement** (batched prefill):

```bash
HF_MODEL=~/models/Qwen3.5-27B-FP8 \
    pytest models/demos/qwen35_27b/tt/tests/test_ttft.py::test_ttft_batched_prefill -v -s
```

**Profiler breakdown** (per-layer timing, non-traced):

```bash
HF_MODEL=~/models/Qwen3.5-27B-FP8 \
    pytest models/demos/qwen35_27b/tt/tests/test_profile_breakdown.py -v -s
```

---

**Previous:** [`index.md`](./index.md) | **Next:** [`bottleneck_analysis.md`](./bottleneck_analysis.md)
