# Sharding Strategies for Windowed KV Cache on T3K

This file describes the T3K hardware topology, derives two candidate sharding
strategies for the windowed KV cache tensor `[B, H_kv, w, d]`, and provides a
decision matrix that leads to a concrete recommendation.

## T3K Topology Recap

### Physical Configuration

The T3K board contains eight Wormhole N300 chips arranged in a 1×8 linear
mesh. The chips are numbered 0 through 7. Each chip is connected to its
adjacent neighbours by Ethernet links:

```text
T3K mesh topology (1×8):

  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
  │ Dev0 ├─────┤ Dev1 ├─────┤ Dev2 ├─────┤ Dev3 │
  └──────┘     └──────┘     └──────┘     └──────┘
                                               │
  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
  │ Dev7 ├─────┤ Dev6 ├─────┤ Dev5 ├─────┤ Dev4 │
  └──────┘     └──────┘     └──────┘     └──────┘

  ── = bidirectional Ethernet link (~12.5 GB/s per direction)

  Interior chips (1–6): two Ethernet neighbours each.
  Endpoint chips (0, 7): one Ethernet neighbour each.
  Total distinct links: 7 (one per adjacent pair in the chain).
```

The mesh is linear (not a ring). In TTNN's collective communication library
(CCL), an all-gather across all eight devices traverses the chain in at most
`N_devices - 1 = 7` hops. The effective all-gather bandwidth scales with the
number of links that can carry data in parallel; for a 1×8 linear chain the
pipeline depth is seven hops, and the CCL implementation overlaps data
transmission with relay forwarding to approach the single-link bandwidth as
the tensor size grows.

### Per-Device Resources

Each Wormhole chip in the T3K provides the following relevant resources for
the attention layer:

| Resource | Capacity |
|---|---|
| DRAM (per chip) | 12 GB |
| DRAM bandwidth (per chip) | ~288 GB/s aggregate (12 channels) |
| L1 SRAM (per Tensix core) | 1536 KiB |
| Tensix cores | 80 (in an 8×10 grid) |
| Peak BF16 matmul throughput | ~262 TFLOPS (all cores) |
| Ethernet link speed | ~12.5 GB/s per direction per link |

For windowed decode, DRAM bandwidth is the dominant resource limit. The
Ethernet link speed (12.5 GB/s) is approximately 23× slower than the
per-chip DRAM bandwidth (288 GB/s). This asymmetry strongly disfavours
sharding strategies that require large all-gathers on every decode step.

### Tensor Shape Definitions

The per-layer KV cache tensor shape for windowed attention decode is:

```text
K cache : [B, H_kv, w, d]     — key circular buffer
V cache : [B, H_kv, w, d]     — value circular buffer
Q       : [B, H_q,  1, d]     — single query vector per head
```

Notation used in this file:

| Symbol | Meaning |
|---|---|
| `B` | Batch size (number of concurrent sequences) |
| `H_q` | Number of query heads |
| `H_kv` | Number of key/value heads (`H_kv <= H_q`; GQA ratio `g = H_q / H_kv`) |
| `w` | Window size (circular-buffer length in tokens) |
| `d` | Head dimension |
| `N` | Number of devices (8 on T3K) |

Throughout this file all sizes are in bytes and use BF16 (2 bytes/element)
unless noted otherwise.

## Candidate Sharding Strategy A: Head-Parallel Sharding

### Core Idea

Partition the `H_kv` dimension across the eight devices. Each device holds a
non-overlapping subset of `H_kv / N` KV heads and the complete `w`-token window
for those heads:

```text
Head-parallel KV layout across 8 devices (H_kv = 8, N = 8):

  Device 0:  K_shard[B, H_kv/8, w, d]   heads  0..0
  Device 1:  K_shard[B, H_kv/8, w, d]   heads  1..1
  ...
  Device 7:  K_shard[B, H_kv/8, w, d]   heads  7..7

  Each device owns its heads' FULL window (all w positions).
  No cross-device KV data is needed during the QK matmul.
```

The query tensor `[B, H_q, 1, d]` is similarly sharded: device `i` receives
`[B, H_q/N, 1, d]` (query heads `i * H_q/N` through `(i+1) * H_q/N - 1`). In
GQA, each device receives the query heads whose corresponding KV heads are also
local to that device.

### Decode Step Under Head-Parallel Sharding

Each device independently executes the full attention pipeline for its local
head group:

```text
Device i (head-parallel, decode step T):

  Input (local):
    q_shard    : [B, H_q/N,  1, d]
    k_shard    : [B, H_kv/N, w, d]    (full window, local heads only)
    v_shard    : [B, H_kv/N, w, d]    (full window, local heads only)

  Compute (local, no CCL):
    ttnn.scaled_dot_product_attention_decode(q_shard, k_shard, v_shard, ...)
    output_shard : [B, H_q/N, 1, d]

  CCL (post-compute):
    If the output projection W_o requires the full [B, H_q, 1, d] vector
    on each device, an all-gather over the head dimension is required:
      ttnn.all_gather(output_shard, dim=1, num_links=1)
      → output_full : [B, H_q, 1, d]   (replicated on all 8 devices)

    If W_o is itself head-sharded (column-parallel linear), the all-gather
    can be deferred or replaced by a reduce_scatter on the W_o output.
```

The key property is that **no CCL is needed before or during the attention
computation**. Each device can stream its local KV shard from DRAM and complete
the attention independently. This preserves the bandwidth-bound character of
single-device decode: each device runs at full local DRAM bandwidth for its
`H_kv/N` heads.

### Memory Per Device — Head-Parallel

The KV cache per device per layer:

```math
\text{KV memory per device} = 2 \times B \times \frac{H_{kv}}{N} \times w \times d \times 2 \text{ bytes}
```

where the leading 2 is for K and V, and the trailing 2 bytes is for BF16.

For representative values `B = 32`, `H_kv = 8`, `N = 8`, `w = 4096`, `d = 128`:

```math
= 2 \times 32 \times 1 \times 4096 \times 128 \times 2 = 67{,}108{,}864 \text{ bytes} = 64 \text{ MiB per device per layer}
```

With 32 transformer layers, the total KV cache across all layers is
`32 × 64 MiB = 2 GiB per device`, well within the 12 GiB DRAM capacity.

### CCL Volume — Head-Parallel (Post-Attention All-Gather)

The output all-gather volume is approximately 224 KiB for representative
parameters (`B = 32`, `H_q = 32`, `N = 8`, `d = 128`, BF16) — a modest
overhead relative to the memory-bound attention compute time. See
[`ccl_operations.md`](./ccl_operations.md) for the full derivation.

If W_o is column-sharded (the standard tensor-parallel layout), the all-gather
is replaced by a `ttnn.reduce_scatter` on the W_o output, which has the same
data volume and can be pipelined with the next layer's computation.

## Candidate Sharding Strategy B: Sequence-Parallel Sharding

### Core Idea

Partition the `w` dimension (the sequence / window length dimension) of the
KV cache across the eight devices. Each device holds a slice of `w/N` tokens
from the global window, covering all heads:

```text
Sequence-parallel KV layout across 8 devices (w = 4096, N = 8):

  Device 0:  K_shard[B, H_kv, w/8,  d]  window tokens    0 ..  511
  Device 1:  K_shard[B, H_kv, w/8,  d]  window tokens  512 .. 1023
  ...
  Device 7:  K_shard[B, H_kv, w/8,  d]  window tokens 3584 .. 4095

  Each device owns ALL heads for its token slice.
  Cross-device all-gather over the window dimension is required before
  any device can compute full attention scores.
```

The query `[B, H_q, 1, d]` is replicated on all devices (or requires an
initial broadcast), since the single query vector must attend to the entire
window.

### Decode Step Under Sequence-Parallel Sharding

Each decode step requires a cross-device all-gather to reconstruct the full
`w`-length KV window before the attention computation can proceed:

```text
Device i (sequence-parallel, decode step T):

  Locally holds:
    q         : [B, H_q,  1, d]          (replicated)
    k_shard_i : [B, H_kv, w/N, d]        (tokens i*(w/N) .. (i+1)*(w/N) - 1)
    v_shard_i : [B, H_kv, w/N, d]

  Step 1 — All-gather K and V across all devices (REQUIRED, before compute):
    ttnn.all_gather(k_shard_i, dim=2, num_links=1)
    → k_full : [B, H_kv, w, d]   on every device
    ttnn.all_gather(v_shard_i, dim=2, num_links=1)
    → v_full : [B, H_kv, w, d]   on every device

  Step 2 — Compute attention (redundant, replicated on all 8 devices):
    ttnn.scaled_dot_product_attention_decode(q, k_full, v_full, ...)
    output : [B, H_q, 1, d]
```

The fundamental problem with sequence-parallel sharding in the decode setting
is that after the all-gather, every device holds the full `w`-length KV tensor
and performs the complete attention computation. The eight devices do identical
work. There is no computational speedup — only the KV data traffic is
distributed, at the cost of a mandatory all-gather on the critical path.

An alternative formulation is to compute partial attention scores on each
device (over its `w/N` local tokens) and then apply a distributed softmax
normalisation. This avoids materialising the full KV tensor on each device but
requires multiple CCL operations (all-gather of partial max and sum for the
online softmax denominator, then all-gather or reduce of partial AV products).
The CCL volume is comparable to the simple all-gather approach.

### Memory Per Device — Sequence-Parallel

The KV cache per device per layer:

```math
\text{KV memory per device} = 2 \times B \times H_{kv} \times \frac{w}{N} \times d \times 2 \text{ bytes}
```

For `B = 32`, `H_kv = 8`, `N = 8`, `w = 4096`, `d = 128`:

```math
= 2 \times 32 \times 8 \times 512 \times 128 \times 2 = 67{,}108{,}864 \text{ bytes} = 64 \text{ MiB per device per layer}
```

The per-device memory footprint is identical to head-parallel sharding. The
total KV data volume across all eight devices is the same for both strategies:
`8 × 64 MiB = 512 MiB per layer`.

### CCL Volume — Sequence-Parallel (Pre-Compute All-Gather)

The K+V all-gather volume is approximately 448 MiB for representative
parameters (`B = 32`, `H_kv = 8`, `N = 8`, `w = 4096`, `d = 128`, BF16),
corresponding to ≈ 38 ms at 12.5 GB/s — roughly 20× the single-device
attention compute time. See [`ccl_operations.md`](./ccl_operations.md) for
the full derivation. Sequence-parallel sharding is therefore
**communication-bound** for all practical window sizes and batch sizes.

Even with pipeline overlap the CCL volume dominates: the T3K all-gather
pipeline can sustain roughly the single-link bandwidth for large transfers, so
the lower bound on latency is `total_bytes / (12.5 GB/s)`, which remains
prohibitive.

## Decision Matrix

| Criterion | Head-Parallel | Sequence-Parallel |
|---|---|---|
| KV memory per device | `2 × B × (H_kv/N) × w × d × 2` bytes | `2 × B × H_kv × (w/N) × d × 2` bytes |
| Per-device memory footprint | Equal to seq-parallel | Equal to head-parallel |
| CCL operation(s) required | Optional output all-gather (post-compute) | Mandatory KV all-gather (pre-compute, on critical path) |
| CCL volume per decode step | `B × (H_q/N) × d × 2 × (N-1)` ≈ 224 KiB | `2 × B × H_kv × (w/N) × d × 2 × (N-1)` ≈ 448 MiB |
| CCL latency at 12.5 GB/s | ~18 µs | ~38 ms |
| Is CCL on critical path? | No (can be pipelined with W_o) | Yes (blocks QK matmul) |
| Compute replicated across devices? | No — each device computes its heads only | Yes — all devices compute full attention |
| Parallelism efficiency | `N×` speedup on attention | No speedup; CCL overhead added |
| Scales with `w`? | CCL volume independent of `w` | CCL volume grows linearly with `w` |
| Divisibility requirement | `H_kv` divisible by `N` | `w` divisible by `N` |
| Applicable to GQA? | Yes — shard `H_kv`; GQA ratio preserved per device | Yes — all heads local, but full CCL still required |

For representative values `B = 32`, `H_q = 32`, `H_kv = 8`, `w = 4096`,
`d = 128`, `N = 8`, BF16:

```text
Head-parallel CCL:
  Volume   ≈ 224 KiB
  Latency  ≈ 18 µs   (at 12.5 GB/s, lower bound)

Sequence-parallel CCL (K + V all-gather, per-device shard w/N = 512):
  Volume   ≈ 448 MiB
  Latency  ≈ 38 ms   (at 12.5 GB/s, lower bound)

Single-device attention compute time (decode, bandwidth-bound):
  KV bytes = 2 × 32 × 8 × 4096 × 128 × 2 = 512 MiB
  Time     ≈ 512 MiB / 288 GB/s ≈ 1.8 ms

Ratio (seq-parallel CCL / compute): 38 ms / 1.8 ms ≈ 20×
```

Sequence-parallel sharding imposes a CCL overhead that is approximately 20
times the attention compute time for these parameters. The ratio worsens
linearly as `w` increases.

## Recommendation

**Head-parallel sharding is the correct strategy for windowed attention decode
on T3K.**

The rationale is as follows:

1. **CCL is off the critical path.** The optional output all-gather required
   by head-parallel sharding is small (sub-millisecond) and can be pipelined
   with the output projection matmul. Sequence-parallel sharding forces a
   hundreds-of-MiB all-gather (≈ 448 MiB for w=4096) on every decode step
   before any attention work can begin, serialising compute behind communication.

2. **Each device achieves full DRAM bandwidth utilisation.** Under head-
   parallel sharding each device streams its `H_kv/N` heads' KV window from
   its local DRAM at the full 288 GB/s chip bandwidth. The per-device KV
   data volume is reduced by `N×` compared to single-device execution, giving
   an `N×` throughput improvement on the bandwidth-bound attention kernel.

3. **Parallelism is genuine.** Devices compute disjoint subsets of the
   attention outputs. There is no replicated work. The eight-way speedup on
   the attention kernel is real.

4. **CCL volume is independent of `w`.** As the window size grows, the output
   all-gather volume (which depends on `H_q/N`, `B`, `d`) does not change.
   Head-parallel sharding therefore scales to arbitrarily large windows
   without increasing communication cost.

5. **Compatibility with tensor-parallel weight sharding.** The standard
   tensor-parallel layout for transformer attention layers shards query, key,
   and value weight matrices by head dimension and applies column-parallel
   linear for Q/K/V projection and row-parallel linear (with reduce-scatter)
   for W_o. Head-parallel KV cache sharding is exactly compatible with this
   weight sharding: each device projects to its own heads' Q/K/V and reads its
   own heads' KV cache, with no misalignment between weight and activation
   shards.

6. **Divisibility.** Typical model configurations have `H_kv = 8`, `N = 8`
   (one KV head per device), or `H_kv = 16`, `N = 8` (two KV heads per
   device). Both are perfectly divisible. In contrast, window sizes `w` are
   chosen for model quality rather than hardware alignment, and values like
   `w = 4096` or `w = 8192` are both divisible by 8; however, this is not
   guaranteed for all models.

### Limitation of Head-Parallel Sharding

The only scenario in which head-parallel sharding is not applicable is when
`H_kv < N` — for example, a multi-query attention (MQA) model with `H_kv = 1`
cannot distribute one KV head across eight devices. In this case an alternative
strategy is required: either replicate the single KV head on all devices (no
sharding of KV), or fall back to sequence-parallel sharding with a different
parallelism strategy for the output. For models with `H_kv = 1`, windowed
attention decode on T3K is effectively a single-device KV problem, and the
attention op can be computed identically on all eight devices with replicated
KV data, parallelising only the Q projection and W_o projection across devices.

---

**Next:** [`ccl_operations.md`](./ccl_operations.md)
