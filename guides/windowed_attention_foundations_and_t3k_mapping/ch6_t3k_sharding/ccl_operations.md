# CCL Operations for Windowed Attention on T3K

This file identifies the collective communication operations required under
each KV-cache sharding strategy, quantifies the bandwidth demand against the
T3K Ethernet link speed, examines opportunities to overlap CCL with compute,
and compares the CCL cost of windowed attention to that of full attention.

## CCL Primitives in TTNN

The TTNN Collective Communication Library (CCL) provides device-to-device
tensor operations that execute over the Ethernet links of the T3K mesh. The
two primitives relevant to attention are:

### `ttnn.all_gather`

Gathers shards of a tensor from all participating devices and writes the
complete tensor to every device. If device `i` holds shard `X_i` of a tensor
`X` along dimension `dim`, after `ttnn.all_gather(X_i, dim=dim)` every device
holds the concatenated `X = [X_0 || X_1 || ... || X_{N-1}]` along `dim`.

```text
All-gather over head dimension (dim=1), N=4 illustrative example:

  Before:
    Dev0: [B, H/4, S, d]  (heads 0..H/4-1)
    Dev1: [B, H/4, S, d]  (heads H/4..H/2-1)
    Dev2: [B, H/4, S, d]  (heads H/2..3H/4-1)
    Dev3: [B, H/4, S, d]  (heads 3H/4..H-1)

  After all_gather(dim=1):
    All devices: [B, H, S, d]   (all heads, replicated)
```

**Data volume:** For a shard of shape `[B, H/N, S, d]`, the total bytes moved
across all links to complete the all-gather is:

```math
\text{AllGather bytes} = B \times \frac{H}{N} \times S \times d \times \text{dtype\_bytes} \times (N - 1)
```

Each device sends its shard `(N-1)` times (once toward each other device in
the chain). The ring-reduce pattern used internally by the CCL pipelines
these transmissions, so the per-link utilisation for a single all-gather
approaches `1/N` of the total volume, but the elapsed time is bounded by the
slowest link.

### `ttnn.reduce_scatter`

Applies a reduction (typically sum) across device shards and distributes the
result such that each device holds a unique slice of the reduced output. This
is the inverse of all-gather: if the operation to be computed produces a result
that is naturally partitioned (e.g., the output of a row-parallel linear
layer), `reduce_scatter` both reduces the partial results and shards the output
in a single collective.

```text
Reduce-scatter over head dimension (dim=1), N=4 illustrative example:

  Each device holds a full [B, H, d] partial sum (from its column of W_o).
  After reduce_scatter(dim=1):
    Dev0: reduced [B, H/4, d]  (heads 0..H/4-1)
    Dev1: reduced [B, H/4, d]
    Dev2: reduced [B, H/4, d]
    Dev3: reduced [B, H/4, d]
```

`ttnn.reduce_scatter` is used in the output projection (W_o) when the result
of that projection is consumed by a layer-norm and residual add that can accept
head-sharded inputs.

## CCL Operations Under Head-Parallel Sharding

### Attention Phase: No CCL Required

Under head-parallel sharding, device `i` holds KV heads `[i*H_kv/N, (i+1)*H_kv/N)`
and query heads `[i*H_q/N, (i+1)*H_q/N)`. The call to
`ttnn.scaled_dot_product_attention_decode` on each device reads only local
DRAM and produces a local output shard:

```text
Device i — attention (no CCL):

  q_shard    : [B, H_q/N,  1, d]  (local, no transfer needed)
  k_shard    : [B, H_kv/N, w, d]  (local KV cache shard)
  v_shard    : [B, H_kv/N, w, d]  (local KV cache shard)
    ↓
  ttnn.scaled_dot_product_attention_decode(...)
    ↓
  output_shard : [B, H_q/N, 1, d]  (local result)
```

No cross-device data movement is required during this phase. Each device
operates independently at its local DRAM bandwidth limit.

### Post-Attention CCL: Output All-Gather

After the attention computation each device holds a shard of the output
`[B, H_q/N, 1, d]`. The subsequent output projection W_o has two possible
sharding configurations:

**Case 1 — W_o is column-sharded (head-parallel, the preferred case):**
W_o has shape `[H_q * d, model_dim]`; each device holds columns corresponding
to its head group, i.e., `[H_q/N * d, model_dim]`. The local attention output
shard is multiplied by the local W_o shard, producing a partial sum of shape
`[B, 1, model_dim]`. An `ttnn.reduce_scatter` then accumulates the partial
sums across all devices and shards the result along the model dimension:

```text
  output_shard : [B, H_q/N, 1, d]
    ↓
  ttnn.matmul(output_shard.reshape(B, 1, H_q/N * d), W_o_shard)
    → partial_proj : [B, 1, model_dim]   (partial sum, local)
    ↓
  ttnn.reduce_scatter(partial_proj, dim=2)
    → reduced_proj_shard : [B, 1, model_dim/N]   (on each device)
```

The `ttnn.reduce_scatter` here moves `B × model_dim × 2 bytes` total across
all devices, which equals `B × model_dim × 2 × (N-1) / N` bytes per device.
For `B = 32`, `model_dim = 4096`, `N = 8`, BF16: `32 × 4096 × 2 = 256 KiB`
total, ≈ 224 KiB per device transferred over the wire.

**Case 2 — W_o is replicated (no sharding):**
Each device must first reconstruct the full attention output `[B, H_q, 1, d]`
by all-gathering over the head dimension before multiplying by W_o:

```text
  output_shard : [B, H_q/N, 1, d]
    ↓
  ttnn.all_gather(output_shard, dim=1, num_links=1)
    → output_full : [B, H_q, 1, d]   (replicated on all devices)
    ↓
  ttnn.matmul(output_full.reshape(B, 1, H_q * d), W_o)
    → proj : [B, 1, model_dim]
```

The all-gather data volume is:

```math
\text{AllGather(output) volume} = B \times \frac{H_q}{N} \times d \times 2 \times (N - 1)
```

For `B = 32`, `H_q = 32`, `N = 8`, `d = 128`, BF16:
`32 × 4 × 128 × 2 × 7 = 229,376 bytes ≈ 224 KiB`.

In both cases the CCL volume is in the hundreds of KiB range, not the GiB
range. The Case 1 configuration (column-sharded W_o with reduce-scatter) is
preferred because it keeps all intermediate tensors head-sharded and avoids
materialising the full `[B, H_q, 1, d]` output on any single device.

### KV Cache Write: Per-Device, No CCL

The `ttnn.update_cache` call that writes the new `k_T` and `v_T` vectors into
the circular buffer is local to each device. Device `i` projects the input
token through its local Q/K/V weight shards (which are head-sharded along the
same head groups), obtaining:

```text
  k_T_shard : [B, H_kv/N, 1, d]   (new key vector for local heads)
  v_T_shard : [B, H_kv/N, 1, d]   (new value vector for local heads)
    ↓
  ttnn.update_cache(k_shard, k_T_shard, T % w)   (in-place, local DRAM)
  ttnn.update_cache(v_shard, v_T_shard, T % w)
```

No cross-device data movement is required. The write pointer `T % w` is a
scalar that is advanced uniformly across all devices; it can be maintained by
the host decode loop and broadcast to all devices at the start of each step,
or derived deterministically from the step counter.

## CCL Operations Under Sequence-Parallel Sharding

Sequence-parallel sharding requires two large all-gathers per decode step —
one for K and one for V — before the attention computation. These are on the
critical path and cannot be deferred.

### Pre-Attention CCL: KV All-Gather

Each device holds a window slice `[B, H_kv, w/N, d]`. The full KV window must
be assembled on each device before the QK matmul. For the decode-step
pseudocode showing the `ttnn.all_gather(k_shard_i, dim=2)` call and the
subsequent redundant full-attention computation on all devices, see
[`sharding_strategies.md`](./sharding_strategies.md).

The all-gather data volume per collective (K only):

```math
\text{AllGather(K) volume} = B \times H_{kv} \times \frac{w}{N} \times d \times 2 \times (N - 1)
```

Combined K + V:

```math
\text{AllGather(K + V) volume} = 2 \times B \times H_{kv} \times \frac{w}{N} \times d \times 2 \times (N - 1)
```

For `B = 32`, `H_kv = 8`, `N = 8`, `w = 4096`, `d = 128`, BF16 (note: `w/N = 512`):

```math
= 2 \times 32 \times 8 \times \underbrace{512}_{w/N} \times 128 \times 2 \times 7 \approx 448 \text{ MiB}
```

This volume must cross at least one Ethernet link per device pair along the
T3K chain, and each link carries its fair share of the all-gather traffic in
a pipelined ring implementation.

## Bandwidth Analysis: Ethernet vs KV Data Volume

### T3K Link Speed

Each Ethernet link on T3K provides approximately 12.5 GB/s of unidirectional
bandwidth. In a 1×8 linear mesh, a ring-based all-gather uses all 7 links
between adjacent devices in sequence. For a well-implemented ring all-gather
on `N` devices, the theoretical minimum time to transfer a tensor of total
size `S` bytes is:

```math
t_{\text{all-gather}} \approx \frac{S \times (N-1)/N}{\text{link\_bandwidth}} \approx \frac{S}{\text{link\_bandwidth}}
```

for large `N` (where `(N-1)/N ≈ 1`). For the T3K with 8 devices and 12.5 GB/s:

```math
t_{\text{all-gather}} \approx \frac{S}{12.5 \text{ GB/s}}
```

### KV Data Volume Per Decode Step

The total KV data that would need to be transferred per layer per decode step
under each strategy:

| Strategy | Transfer tensor | Shape | Volume (B=32, H_kv=8, w=4096, d=128, BF16) |
|---|---|---|---|
| Head-parallel (Case 2) | Attention output | `[B, H_q/N, 1, d]` × 7 hops | ≈ 224 KiB |
| Head-parallel (Case 1) | Proj partial sum | `[B, 1, model_dim]` × N | ≈ 256 KiB |
| Sequence-parallel (K) | KV window K shard | `[B, H_kv, w/N, d]` × 7 hops | ≈ 224 MiB |
| Sequence-parallel (K+V) | KV window K + V | 2 × above | ≈ 448 MiB |

### Transfer Time Comparison

```text
Transfer time at 12.5 GB/s:

  Head-parallel (output gather/scatter): 224–256 KiB / 12.5 GB/s ≈ 18–20 µs
  Sequence-parallel (KV all-gather K+V, w/N=512): 448 MiB / 12.5 GB/s  ≈ 38 ms

  Single-device attention compute (bandwidth-bound, B=32):
    KV bytes = 2 × 32 × 8 × 4096 × 128 × 2 = 512 MiB
    DRAM BW  = 288 GB/s
    Compute  ≈ 512 MiB / 288 GB/s ≈ 1.8 ms

  With head-parallel (8 devices, each processes 1/8 of heads):
    Per-device KV bytes = 512 / 8 = 64 MiB
    Per-device compute  ≈ 64 MiB / 288 GB/s ≈ 222 µs
    CCL overhead (Case 1) ≈ 18 µs
    Total ≈ 240 µs  (≈ 7.5× speedup vs single-device)

  With sequence-parallel:
    CCL alone ≈ 38 ms  (≈ 20× single-device compute; still prohibitive)
```

The head-parallel approach achieves near-linear throughput scaling (7.5× on
8 devices) with only a small CCL overhead. The sequence-parallel approach is
dominated by CCL latency.

### Impact of Window Size `w` on CCL Volume

For head-parallel sharding, the CCL volume (output all-gather or reduce-scatter)
depends on `H_q`, `B`, `d`, and `model_dim`, none of which depend on `w`. The
CCL cost is **constant with respect to `w`**:

```text
Head-parallel CCL volume vs w (B=32, H_q=32, d=128, N=8, BF16):

  w =  1024:  ≈ 224 KiB   (same as any w)
  w =  4096:  ≈ 224 KiB
  w =  8192:  ≈ 224 KiB
  w = 32768:  ≈ 224 KiB

  CCL cost is invariant under changes to w.
```

For sequence-parallel sharding, the CCL volume grows linearly with `w`:

```math
\text{SeqPar CCL volume} \propto w
```

```text
Sequence-parallel CCL volume vs w (B=32, H_kv=8, d=128, N=8, BF16):
  Formula: 2 × B × H_kv × (w/N) × d × 2 × (N-1)

  w =  1024:  ≈  112 MiB   (at 12.5 GB/s: ≈   9 ms)
  w =  4096:  ≈  448 MiB   (at 12.5 GB/s: ≈  38 ms)
  w =  8192:  ≈  896 MiB   (at 12.5 GB/s: ≈  72 ms)
  w = 32768:  ≈ 3,584 MiB  (at 12.5 GB/s: ≈ 287 ms)
```

This linear growth makes sequence-parallel sharding progressively worse as
model window sizes increase. Models with `w = 8192` (Qwen2) or `w = 32768`
(Mistral-style large-context variants) are entirely impractical under
sequence-parallel sharding on T3K.

## Overlap Opportunities

### Head-Parallel: CCL–Compute Overlap

The small output CCL in head-parallel sharding creates an opportunity for
pipelining. In a multi-layer transformer, the all-gather (or reduce-scatter)
for layer `l` can be initiated while the next layer's Q/K/V projections are
being computed. This form of overlap is standard in tensor-parallel inference
and is supported by TTNN's asynchronous CCL dispatch model:

```text
Pipelining pattern (head-parallel, N layers):

  Layer l compute:    |QKVO_proj|SDPA_decode|W_o_matmul|reduce_scatter|
  Layer l+1 compute:                         |QKVO_proj ....>          |
                                                         ↑
                                        reduce_scatter for layer l
                                        can overlap with Q/K/V proj
                                        for layer l+1 if they use
                                        disjoint resources
```

For this overlap to be effective, the host Python loop must issue the
`reduce_scatter` call and immediately continue issuing the next layer's
Q/K/V matmul commands on the same command queue. TTNN's asynchronous dispatch
model allows the device to schedule both operations and execute them when
resources are available.

Whether full overlap is achieved depends on whether the `reduce_scatter` uses
the Ethernet DMA engine (which is independent of the Tensix compute fabric)
while the Q/K/V projections use the Tensix cores. If the CCL implementation
occupies Tensix cores for any phase of the collective, partial overlap is
still possible but not complete. The TTNN CCL team's recommended practice is
to treat the link bandwidth as an independent resource and assume that Tensix-
level compute can proceed in parallel with Ethernet transfer for large enough
tensors.

### Sequence-Parallel: No Effective Overlap

In sequence-parallel sharding the KV all-gather is on the critical path —
the QK matmul cannot start until the full `[B, H_kv, w, d]` tensor is
assembled. There is no computation that depends only on the local KV shard
that can be performed while waiting for the gather to complete. The query
projection Q = token × W_q could be pipelined with the KV all-gather, but
the Q projection is a small operation (Q shape `[B, H_q, 1, d]`) that
completes in microseconds, providing negligible overlap benefit against a
hundreds-of-milliseconds gather.

### KV Cache Write Overlap (Both Strategies)

The `ttnn.update_cache` call that inserts the new token into the circular
buffer is local to each device and does not involve CCL. It writes one slot
of `[B, H_kv/N, 1, d]` (head-parallel) or `[B, H_kv, 1, d]` (seq-parallel)
into DRAM at position `T % w`. This write is small and completes quickly
relative to the subsequent SDPA kernel. There is no overlap opportunity here
because the write must precede the attention read (write-before-read ordering
on the cache), but the latency is negligible.

## CCL Summary

```text
CCL operations per decode step per layer:

  Strategy           │ Pre-attention CCL        │ Post-attention CCL
  ───────────────────┼──────────────────────────┼────────────────────────────
  Head-parallel      │ None                     │ reduce_scatter(output proj)
                     │                          │ or all_gather(attn output)
                     │                          │ Volume: ~224–256 KiB
                     │                          │ Latency: ~18–20 µs
  ───────────────────┼──────────────────────────┼────────────────────────────
  Sequence-parallel  │ all_gather(K) +          │ None (computation
                     │ all_gather(V)            │ replicated on all devices)
                     │ Volume: ~448 MiB (w=4k) │
                     │ Latency: ~38 ms (w=4k)   │
```

Under head-parallel sharding, the CCL overhead per layer is approximately 18–20 µs
regardless of window size. For a 32-layer transformer, the aggregate CCL
overhead is `32 × 20 µs = 640 µs` per decode step — a small fraction of the
total step latency. The attention compute on eight devices (≈ 222 µs per
device per layer) dominates, and a significant fraction of the CCL cost can
be hidden behind adjacent-layer compute.

---

**Next:** [`per_device_window_application.md`](./per_device_window_application.md)
