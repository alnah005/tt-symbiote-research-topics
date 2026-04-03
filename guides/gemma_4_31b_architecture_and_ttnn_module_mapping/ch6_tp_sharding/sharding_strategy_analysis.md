# Sharding Strategy Analysis for Global KV Heads

## The Problem

With TP=8 on the T3K 1x8 mesh, the head-count-to-device mapping for each
layer type is:

| Component | Total | Per Device (TP=8) | Status |
|-----------|-------|-------------------|--------|
| Q heads (both types) | 32 | 4 | Clean split |
| Sliding KV heads | 16 | 2 | Clean split |
| Global KV heads | 4 | 0.5 | **Cannot split** |

The sliding layers are straightforward: 2 KV heads per device, each serving
2 Q heads (GQA group size = 2). The global layers are the problem. Four KV
heads across 8 devices means each head must somehow serve Q heads on multiple
devices, or the heads must be replicated, or the sharding dimension must
change.

This file evaluates four strategies for resolving this mismatch.

## Option A --- Replicate All 4 KV Heads on Every Device

### Mechanism

Every device holds a full copy of all 4 global KV heads. The K and V
projection weights are replicated (not sharded) across the mesh. Each
device computes all 4 KV head projections locally and maintains a full
KV cache for all 4 heads.

The Q projection is still column-parallel sharded: each device computes 4
of 32 Q heads. During SDPA, each device's 4 Q heads attend over all 4 KV
heads locally. The O projection is row-parallel sharded, followed by an
all-reduce.

### Per-Device Shapes

| Tensor | Shape per Device | Notes |
|--------|-----------------|-------|
| Q (after projection) | `[B, 1, 4, 512]` | 4 of 32 Q heads |
| K (after projection) | `[B, 1, 4, 512]` | All 4 KV heads (replicated) |
| V (after projection) | `[B, 1, 4, 512]` | All 4 KV heads (replicated) |
| KV cache per layer | `[B, 4, S, 512]` | Full 4 heads x full seq len |

### Memory Cost

KV cache per global layer per device at various sequence lengths (BF16,
2 bytes/element; K + V = 2 tensors):

```math
\text{KV cache per layer per device} = 2 \times B \times 4 \times S \times 512 \times 2
```

At B=1:

| Seq Length (S) | KV Cache per Layer per Device |
|----------------|------------------------------|
| 1,024 | 2 x 1 x 4 x 1,024 x 512 x 2 = 8.0 MB |
| 8,192 | 2 x 1 x 4 x 8,192 x 512 x 2 = 64.0 MB |
| 32,768 | 2 x 1 x 4 x 32,768 x 512 x 2 = 256.0 MB |
| 131,072 | 2 x 1 x 4 x 131,072 x 512 x 2 = 1,024.0 MB |
| 262,144 | 2 x 1 x 4 x 262,144 x 512 x 2 = 2,048.0 MB |

Across 10 global layers at S=8,192: 10 x 64.0 MB = **640 MB per device**.
At S=262,144 (full 256K): 10 x 2,048 MB = **20,480 MB per device** ---
exceeds the 12 GB DRAM budget on its own.

### Weight Memory

The K projection weight for global layers is `[5376, 2048]`. With K=V
sharing, there is no separate V weight. Replicated means each device stores
the full `[5376, 2048]` weight.

Per device: 5376 x 2048 x 2 = 22.0 MB per global layer (BF16).
Across 10 layers: 220 MB per device.

Compare with column-parallel sharding (Option B/C/D): if sharded by 4 or 8,
this drops to 55 MB or 27.5 MB per device.

### CCL Cost

- **No cross-device KV communication during SDPA.** Every device has all KV
  heads locally, so attention is fully local.
- The all-reduce after the O projection is the only CCL operation per global
  attention layer, same as sliding layers.
- However, the K projection computation is **redundantly computed** on all 8
  devices (8x the compute for KV projection vs a sharded approach).

### Pros and Cons

| Pros | Cons |
|------|------|
| No cross-device KV communication | 8x redundant KV projection compute |
| Simple implementation: replicate weights, no gather needed | 8x KV cache memory (each device holds all 4 heads) |
| SDPA sees correct GQA grouping (4Q:4KV = 1:1 per device) | KV cache memory explodes at long sequence lengths |
| Identical SDPA call signature to non-TP case | 8x redundant K projection weight storage |

---

## Option B --- Shard 4 KV Heads Across 4 Devices

### Mechanism

Assign 1 KV head to each of devices 0--3. Devices 4--7 hold no KV heads for
global layers. Before SDPA, devices 4--7 must obtain the KV cache entries they
need via an `ttnn.all_gather` on the KV tensors.

Alternatively, after the KV projection, perform an all-gather so that every
device holds all 4 KV heads. This is functionally equivalent to Option A for
the SDPA step, but the KV projection is computed on only 4 devices.

### Per-Device Shapes (Before All-Gather)

| Tensor | Devices 0--3 | Devices 4--7 |
|--------|-------------|-------------|
| Q (after projection) | `[B, 1, 4, 512]` | `[B, 1, 4, 512]` |
| K (after projection) | `[B, 1, 1, 512]` | Empty |
| V (after projection) | `[B, 1, 1, 512]` | Empty |

After all-gather on KV, every device holds `[B, 1, 4, 512]` for both K and V.

### Memory Cost

If the all-gather is performed before KV cache update, every device ends up
storing the full 4-head KV cache (same as Option A). If the KV cache is stored
in sharded form (1 head on devices 0--3, 0 on devices 4--7), then only 4
devices contribute KV cache memory, but an all-gather is needed before every
SDPA call.

Sharded KV cache variant at S=8,192 (B=1, BF16):
- Devices 0--3: 2 x 1 x 1 x 8,192 x 512 x 2 = 16.0 MB per layer
- Devices 4--7: 0 MB per layer
- All-gather transfers 4 x 16.0 MB = 64.0 MB per layer per SDPA call

### CCL Cost

- **All-gather on KV** before every SDPA call (or every decode step): 4 KV
  heads x S tokens x 512 dims x 2 bytes (K) + same for V.
- At S=8,192: all-gather payload = 2 x 4 x 8,192 x 512 x 2 = 64.0 MB per
  layer per decode step.
- At S=262,144: all-gather payload = 2,048 MB per layer --- catastrophic.
- The all-gather latency grows linearly with sequence length, making this
  option untenable for long contexts.

### Pros and Cons

| Pros | Cons |
|------|------|
| KV projection computed on 4 devices only (4x vs 8x) | Devices 4--7 are idle for KV projection |
| Sharded KV cache saves memory on devices 4--7 | All-gather required before every SDPA call |
| | All-gather cost grows linearly with sequence length |
| | Asymmetric device utilization |

---

## Option C --- Shard by head_dim Instead of Head Count

### Mechanism

Each device holds all 4 KV heads, but only a slice of the 512-dim head
dimension: 512 / 8 = 64 dims per device. The K projection weight is sharded
as `[5376, 2048]` with the output dimension split by 8, giving each device
`[5376, 256]`. Each device computes all 4 KV heads at 64 dims each.

Before SDPA, the full 512-dim head must be reconstructed via an all-gather
along the head_dim axis, because SDPA requires the complete head dimension for
the dot product.

### Per-Device Shapes

| Tensor | Shape per Device | Notes |
|--------|-----------------|-------|
| Q (after projection) | `[B, 1, 4, 512]` | 4 Q heads, full 512 dims (column-parallel on head count) |
| K (after projection) | `[B, 1, 4, 64]` | All 4 KV heads, 64/512 dims |
| V (after projection) | `[B, 1, 4, 64]` | All 4 KV heads, 64/512 dims |

After all-gather: K and V become `[B, 1, 4, 512]`.

### Memory Cost

KV cache stored in sharded form (64 dims per device) at S=8,192 (B=1, BF16):
- Per device per layer: 2 x 1 x 4 x 8,192 x 64 x 2 = 8.0 MB
- Across 10 layers: 80 MB per device

This is 8x less than Option A at the same sequence length.

However, the all-gather before SDPA reconstructs the full `[B, 4, S, 512]`
tensor, which temporarily requires the same memory as Option A on every device.
The temporary buffer at S=8,192 is 64 MB, which must be allocated in addition
to the sharded KV cache.

### CCL Cost

- All-gather on KV before every SDPA call: each device contributes its 64-dim
  slice and receives the full 512 dims.
- Payload per device per KV tensor: 4 heads x S x 64 dims x 2 bytes.
- At S=8,192: 4 x 8,192 x 64 x 2 = 4.0 MB per device per tensor. Total
  all-gather payload (all devices, K+V): 2 x 8 x 4.0 MB = 64.0 MB.
- This is the same total data movement as Option B. The all-gather cost is
  equivalent.

### Pros and Cons

| Pros | Cons |
|------|------|
| Even device utilization (all 8 devices compute) | All-gather required before every SDPA call |
| Sharded KV cache is 8x smaller than replicated | Temporary full-dim buffer needed for SDPA |
| K projection weight evenly sharded | Sharding by head_dim is non-standard for GQA |
| | SDPA cannot consume partial-dim heads natively |
| | Incompatible with standard `paged_sdpa_decode` which expects whole heads |

---

## Option D --- Replicate KV Across All Devices (Recommended)

### Mechanism

This is a refinement of Option A with a critical optimization. Replicate all
4 global KV heads on every device, but:

1. **Column-parallel shard the Q projection** as usual: each device computes 4
   of 32 Q heads.
2. **Replicate the K projection weight** on every device: each device computes
   all 4 KV heads. Since K=V sharing is active in global layers, the single
   `[5376, 2048]` weight produces both K and V --- there is no separate V
   weight to worry about.
3. Each device maintains a **full KV cache for all 4 heads**.
4. SDPA runs locally on each device with 4 Q heads attending over 4 KV heads
   (GQA ratio 1:1 per device).
5. **Row-parallel shard the O projection** and follow with `ttnn.all_reduce`.

This is operationally identical to Option A. The reason to prefer it over
Options B and C is that it **eliminates all per-step CCL operations for KV
data**, at the cost of 8x KV cache memory and 8x redundant KV projection
compute. The key insight is that the redundant compute and memory costs are
manageable:

### Why the Costs Are Acceptable

**Compute cost of redundant KV projection:**

The K projection for global layers is `[5376, 2048]` --- one of the smaller
matmuls in the model. At B=1 decode, this is a matrix-vector multiply:

```math
\text{FLOPs} = 2 \times 5376 \times 2048 = 22{,}020{,}096 \approx 22\text{M FLOPs}
```

Even at 8x redundancy, this is 176M FLOPs total across 8 devices (22M per device) --- negligible compared to the Q
projection (2 x 5376 x 16384 = 176M FLOPs total, also 22M per device after TP=8 sharding) or the FFN
projections (2 x 5376 x 21504 = 231M FLOPs total per projection, ~29M per device with TP=8). The
KV projection for global layers is the smallest matmul in the attention block.

**Memory cost of replicated KV cache:**

The 10 global layers represent only 10/60 = 16.7% of all layers. The global
KV cache is the only component that grows with sequence length; all other
weight storage is sequence-length-independent.

At practical decode sequence lengths for T3K (up to ~8K--32K tokens given DRAM
constraints from weights), the replicated global KV cache is bounded:

| Seq Length | Global KV (10 layers, replicated, B=1, BF16) | Sliding KV (50 layers, 2 heads/dev, B=1, BF16) |
|------------|----------------------------------------------|------------------------------------------------|
| 2,048 | 10 x 2 x 4 x 2,048 x 512 x 2 = 160 MB | 50 x 2 x 2 x 1,024 x 256 x 2 = 100 MB |
| 4,096 | 10 x 2 x 4 x 4,096 x 512 x 2 = 320 MB | 100 MB (window capped) |
| 8,192 | 10 x 2 x 4 x 8,192 x 512 x 2 = 640 MB | 100 MB |
| 32,768 | 10 x 2 x 4 x 32,768 x 512 x 2 = 2,560 MB | 100 MB |

The sliding KV cache is capped at 1,024 tokens by the window, so it remains
constant regardless of sequence length. The global KV cache grows linearly but
stays manageable at moderate lengths.

**Weight memory cost of replication:**

The replicated K weight per global layer is 5376 x 2048 x 2 = 22.0 MB (BF16).
Compared to the Q weight per device (5376 x 2048 x 2 = 22.0 MB after TP=8
column shard of the `[5376, 16384]` Q weight), the replicated K weight is the
same size. Across 10 global layers: 220 MB of replicated K weights per device.
At BFP8 quantization (1 byte/element), this drops to 110 MB.

### Per-Device Summary

| Component | Shape per Device | Memory (BF16) |
|-----------|-----------------|---------------|
| Q weight | `[5376, 2048]` | 22.0 MB |
| K weight (replicated) | `[5376, 2048]` | 22.0 MB |
| O weight | `[2048, 5376]` | 22.0 MB |
| Q activation | `[B, 1, 4, 512]` | negligible |
| K activation | `[B, 1, 4, 512]` | negligible |
| V activation | `[B, 1, 4, 512]` | negligible |
| KV cache (per layer) | `[B, 4, S, 512]` x 2 | S-dependent |

### CCL Operations

Per global attention layer:
- `ttnn.all_reduce` after O projection (same as every other layer)
- **No** all-gather or reduce-scatter for KV data

Per sliding attention layer (for comparison):
- `ttnn.all_reduce` after O projection

The CCL pattern is **identical** for sliding and global layers under this
strategy, which simplifies the Metal Trace capture and multi-CQ scheduling.

---

## Comparison Matrix

| Criterion | Option A/D (Replicate) | Option B (4-device shard) | Option C (head_dim shard) |
|-----------|----------------------|--------------------------|--------------------------|
| Per-step CCL for KV | None | All-gather (grows with S) | All-gather (grows with S) |
| KV cache per device | Full (4 heads) | 1 head on 4 devices | 64-dim slice (all devices) |
| Device utilization | All 8 compute KV | 4 compute KV, 4 idle | All 8 compute KV |
| SDPA compatibility | Standard GQA | Needs gather first | Needs gather first |
| Implementation complexity | Low (replicate weight) | Medium (asymmetric) | High (non-standard dim split) |
| Sequence length scalability | Limited by DRAM | Limited by CCL latency | Limited by CCL latency |

## Recommendation

**Use Option D: replicate all 4 global KV heads on every device.**

The rationale:

1. **Simplicity.** The SDPA call on each device sees 4 Q heads and 4 KV heads
   with no gather or scatter. The same `paged_sdpa_decode` call works for both
   sliding (4Q:2KV) and global (4Q:4KV) layers, just with different head
   counts.

2. **No per-step CCL overhead.** Options B and C require an all-gather before
   every SDPA call, with payload proportional to sequence length. At S=32K, the
   all-gather payload for 10 global layers would be 10 x 2 x 4 x 32,768 x 512
   x 2 = 2,560 MB --- larger than the model weights on a single device. This
   latency penalty occurs on every decode step and cannot be overlapped with
   compute.

3. **Negligible compute overhead.** The redundant K projection for global
   layers adds ~176M FLOPs per decode step across all 8 devices --- less than
   1% of total model FLOPs.

4. **Manageable memory.** The replicated global KV cache is the binding
   constraint, but it grows only for 10 layers (not 60). At S=8,192 the total
   global KV cache is 640 MB per device, well within the DRAM budget after
   accounting for weights (see [`kv_cache_sharding.md`](./kv_cache_sharding.md)
   for the full budget).

5. **Metal Trace compatibility.** Uniform CCL patterns across all layers (only
   all-reduce after O and down projections) enable a single traced decode step
   to cover both layer types, simplifying the trace capture and replay loop.

---

**Next:** [`weight_sharding.md`](./weight_sharding.md)
