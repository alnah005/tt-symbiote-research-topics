# Optimization Roadmap

This file presents a prioritized set of optimization techniques for Gemma 4 31B
decode on T3K. Each technique is described with its mechanism, expected impact,
implementation complexity, and dependencies.

## Priority 1: Metal Trace Capture

### Mechanism

Metal Trace captures the entire decode step (60 layers of matmuls, norms, RoPE,
SDPA, and all-reduce) into a replayable trace buffer. On subsequent decode
steps, the trace is replayed without Python-side dispatch overhead. Only input
tensors (`input_ids`, `current_pos`, `page_table`) are updated between replays.

### Expected Impact

Metal Trace typically eliminates 30--50% of decode latency by removing
per-op Python dispatch overhead and enabling the runtime to pre-schedule all
device operations. For a 60-layer model with hundreds of ops per decode step,
the dispatch overhead without tracing is substantial.

At the projected ~16 ms decode latency (S=8K), tracing could reduce this to
~9--11 ms, bringing decode throughput from ~62 to ~90--110 tokens/s.

### Requirements

- **Fixed op sequence:** The polymorphic attention design from
  [Chapter 5](../ch5_attention_module_design/index.md) ensures that sliding
  and global layers resolve at construction time, not at runtime. The trace
  sees a deterministic op sequence on every decode step.
- **Fixed tensor shapes:** Decode always processes B=1, S=1 tokens. KV cache
  updates are in-place. All shapes are constant across steps.
- **No Python conditionals in the hot path:** The 60-layer loop can be
  unrolled at trace time, with each layer's attention subclass determining
  the op sequence.
- **Trace buffer memory:** The trace buffer occupies DRAM (typically 100--300
  MB per device). This must be accounted for in the DRAM budget. At BFP8
  weights with S=8K, there is ~7.2 GB headroom, which is ample.

### Complexity

Low. Metal Trace is a well-established pattern in tt-symbiote for autoregressive
decode. The primary work is verifying that no tensor shape changes or
Python-level branches exist in the decode path.

## Priority 2: DRAM-Sharded Weight Storage

### Mechanism

DRAM-sharded weight storage distributes weight matrix tiles across DRAM banks
to maximize read bandwidth for matrix-vector multiplies during decode. Instead
of reading a contiguous weight block, the hardware reads from multiple DRAM
banks in parallel, increasing effective bandwidth.

### Expected Impact

DRAM-sharded weights can improve memory bandwidth utilization from ~60--70% to
~85--95% of peak, reducing matmul latency proportionally. For decode, where all
projections are memory-bound, this directly reduces per-layer latency.

Applied to all 60 layers:
- Sliding layer projections: ~198 us -> ~140--160 us (20--30% reduction)
- Global layer projections: ~255 us -> ~180--210 us (20--30% reduction)

Across the full model, this could save ~2--4 ms on projection latency alone.

### Requirements

- Weights must be preloaded into DRAM-sharded layout during model
  initialization.
- Each unique matmul shape requires a DRAM-sharded program config. Gemma 4 31B
  has more unique shapes than homogeneous models due to the two attention types:
  - Sliding: Q `[5376, 1024]`, K/V `[5376, 512]`, O `[1024, 5376]`
  - Global: Q `[5376, 2048]`, K `[5376, 2048]`, O `[2048, 5376]`
  - FFN (shared): gate+up `[5376, 5376]`, down `[2688, 5376]`
- Each program config must be tuned for optimal tile-to-bank mapping.

### Complexity

Medium. The sharding layout is model-specific and requires per-shape tuning.
The FFN weights (identical across all 60 layers) amortize the tuning cost well.
Attention weights require separate configs for sliding and global layers.

## Priority 3: Fused QKV Projections

### Mechanism

Combine multiple projection weight matrices into a single matmul to reduce
kernel launch overhead and improve memory access patterns.

#### Sliding Layers: Fused QKV

Concatenate Q, K, V weights along the output dimension:

```text
W_QKV = [W_Q | W_K | W_V]
Per device: [5376, 1024 + 512 + 512] = [5376, 2048]
```

A single matmul replaces three, saving ~10--15 us per layer from avoided kernel
launches. The output is sliced into Q, K, V components (negligible cost).

#### Global Layers: Fused Q+K

Since K=V sharing eliminates the V projection, only Q and K can be fused:

```text
W_QK = [W_Q | W_K]
Per device: [5376, 2048 + 2048] = [5376, 4096]
```

Note: The K weight is replicated while Q is column-sharded. Fusing Q+K requires
the Q shard and the full K weight to be concatenated on each device:
- Device 0: Q shard `[5376, 2048]` + K full `[5376, 2048]` = `[5376, 4096]`
- Device 1: same layout

This is valid because the fused matmul produces `[1, 1, 4096]` which is sliced
into Q `[1, 1, 2048]` (local 4 Q heads) and K `[1, 1, 2048]` (all 4 KV heads).

### Expected Impact

- Sliding layers: ~3 matmuls -> 1, saving ~10--15 us per layer, ~500--750 us
  across 50 layers.
- Global layers: ~2 matmuls -> 1, saving ~5--10 us per layer, ~50--100 us
  across 10 layers.

Total savings: ~550--850 us per decode step.

### Complexity

Low to medium. Fused QKV is a standard optimization in tt-symbiote. The main
nuance is handling the replicated K weight in global layers --- the fused
weight must be constructed per-device at init time with the correct combination
of sharded Q and replicated K.

## Priority 4: Multi-CQ Overlap (CCL Pipelining)

### Mechanism

Multi-CQ (multiple command queues) enables overlapping CCL operations
(all-reduce) with compute on the same device. When an all-reduce is submitted
on one command queue, the next matmul can begin on a different queue without
waiting for the all-reduce to complete.

### Expected Impact

The 120 all-reduce operations per decode step contribute an estimated
0.6--1.2 ms of latency. With Multi-CQ, the all-reduce after the O projection
can overlap with the pre-FFN norm and the start of the gate+up projection.
Similarly, the all-reduce after the down projection can overlap with the
pre-attention norm of the next layer.

Achievable overlap: ~50--70% of CCL latency, saving ~0.3--0.8 ms per decode
step.

### Requirements

- Two command queues per device: one for compute, one for CCL.
- Careful dependency management: the compute following an all-reduce must not
  read the all-reduce output until it completes.
- Compatible with Metal Trace: the trace must capture the multi-CQ schedule.

### Complexity

Medium to high. Multi-CQ requires explicit queue management and synchronization
barriers. The benefit is meaningful but requires careful integration with the
Metal Trace system.

## Priority 5: BFP8 KV Cache

### Mechanism

Store the KV cache in BFP8 (`bfloat8_b`, 1 byte/element) instead of BF16
(2 bytes/element). This halves the KV cache memory and the SDPA memory read
bandwidth, directly reducing SDPA latency for global layers and extending the
maximum context length.

### Expected Impact

**Memory:** KV cache size at S=8,192 drops from 740 MB to 370 MB per device.
At S=131,072, the total DRAM footprint drops from 14.5 GB (exceeds budget)
to 9.3 GB (fits within 12 GB).

**Latency:** Global SDPA latency is approximately halved because the KV cache
read is the bottleneck:

| Seq Length | SDPA per Global Layer (BF16 KV) | SDPA per Global Layer (BFP8 KV) |
|------------|--------------------------------|--------------------------------|
| 8,192 | ~213 us | ~107 us |
| 32,768 | ~853 us | ~427 us |

Across 10 global layers at S=32,768, this saves ~4.3 ms per decode step.

**Quality:** BFP8 KV cache introduces quantization noise into attention. The
impact on output quality must be validated empirically. Gemma 4's V-norm
(which normalizes value vectors to unit RMS) may help stabilize BFP8
quantization by bounding the dynamic range of values stored in the cache.

### Complexity

Low. TTNN supports BFP8 KV cache natively. The change is primarily a dtype
configuration at KV cache initialization time. V-norm ensures that the V
tensors entering the cache are already normalized, which is favorable for
low-precision storage.

## Priority 6: Partial RoPE Optimization

### Mechanism

Global layers apply partial RoPE to only the first 128 of 512 head dimensions.
The current implementation splits the tensor, applies RoPE to the first slice,
and concatenates with the passthrough slice. This split-apply-concat pattern
introduces overhead from tensor slicing and concatenation.

Two optimization approaches:

1. **In-place partial RoPE:** Apply RoPE directly to the first 128 elements
   of each head vector without splitting, using a strided or masked operation.
   This avoids the allocation of temporary tensors.

2. **Extended cos/sin tables with zero padding:** Create cos/sin tables of
   shape `[S, 512]` where the first 128 entries contain the rotary values
   and the remaining 384 are `cos=1, sin=0` (identity rotation). Apply the
   full-dimension RoPE in one pass. The identity entries effectively pass
   through the non-rotary dimensions.

### Expected Impact

Small. The per-layer RoPE cost is ~3 us for global layers, so even a 50%
reduction saves only ~1.5 us per global layer, or ~15 us total. However, this
optimization simplifies the code path and may improve Metal Trace compatibility
by eliminating the slice/concat ops.

### Complexity

Low. Either approach is straightforward to implement and test.

## Priority 7: V-Norm Fusion with KV Cache Write

### Mechanism

V-norm (unscaled RMSNorm on value vectors) is applied on every decode step
before the V tensor is written to the KV cache. Rather than executing V-norm
as a separate kernel followed by a KV cache write, the two operations can be
fused:

```text
Current:  V_projected -> V-norm kernel -> V_normed -> KV cache write kernel
Fused:    V_projected -> fused V-norm + KV cache write kernel
```

This eliminates one tensor read/write round-trip through L1 or DRAM.

### Expected Impact

Small. The V tensor at decode is tiny (e.g., `[1, 2, 1, 256]` for sliding),
so the saved memory traffic is negligible. The primary benefit is reducing
kernel count by 1 per layer (60 fewer kernel launches per decode step), which
helps Metal Trace buffer efficiency.

### Complexity

Medium. Requires a custom fused kernel or verification that existing TTNN
fusion passes can combine RMSNorm with a cache write.

## Priority 8: KV Sharing Potential (Future)

### Mechanism

The Gemma 4 architecture supports `num_kv_shared_layers`, which allows multiple
consecutive decoder layers to share the same KV cache entries. In the 31B
config, `num_kv_shared_layers=0` (no sharing), but future variants may enable
it.

If KV sharing were enabled (e.g., `num_kv_shared_layers=2`), pairs of layers
would read from the same KV cache, eliminating KV projection and cache writes
for the sharing layers. This would:

- Reduce KV cache memory by up to 50% (if every other layer shares).
- Eliminate KV projection compute for sharing layers.
- Reduce SDPA reads (the cache is already in L1 from the prior layer).

### Expected Impact

Not applicable to the current 31B config. Noted here for forward compatibility.

### Complexity

Medium. The module design must support conditional KV cache read/write. The
KV cache sharding strategy would need to account for shared entries.

## Optimization Summary

| Priority | Technique | Est. Latency Savings | Memory Impact | Complexity |
|----------|-----------|---------------------|---------------|------------|
| 1 | Metal Trace | 5--7 ms (30--45%) | +100--300 MB trace buffer | Low |
| 2 | DRAM-sharded weights | 2--4 ms (20--30% on projections) | None | Medium |
| 3 | Fused QKV / Q+K | 0.5--0.8 ms | None | Low--Medium |
| 4 | Multi-CQ overlap | 0.3--0.8 ms | None | Medium--High |
| 5 | BFP8 KV cache | 1--4 ms (context-dependent) | Halves KV cache | Low |
| 6 | Partial RoPE optimization | ~0.015 ms | None | Low |
| 7 | V-norm fusion | ~0.01 ms | None | Medium |
| 8 | KV sharing (future) | N/A (31B has it disabled) | Reduces KV cache | Medium |

### Combined Impact Estimate

Applying priorities 1--5 to the baseline ~16 ms decode (S=8K):

```text
Baseline:                         ~16.2 ms
After Metal Trace:                ~9.5 ms   (-41%)
After DRAM-sharded weights:       ~7.0 ms   (-26%)
After fused QKV:                  ~6.4 ms   (-9%)
After Multi-CQ:                   ~5.8 ms   (-9%)
After BFP8 KV cache:             ~5.2 ms   (-10%)
```

Projected optimized decode: **~5.2 ms per token (~190 tokens/s)** at S=8,192,
B=1. This is competitive with similarly-sized models on T3K.

At S=32,768 with all optimizations (including BFP8 KV cache):

```text
Baseline:                         ~22.6 ms
Optimized (estimated):            ~9--11 ms  (~90--110 tokens/s)
```

The 50-layer sliding window design is a significant architectural advantage:
the majority of the model has sequence-length-independent decode cost, keeping
overall latency manageable even at long contexts.

---

**End of guide.** Return to [Guide Index](../index.md)
