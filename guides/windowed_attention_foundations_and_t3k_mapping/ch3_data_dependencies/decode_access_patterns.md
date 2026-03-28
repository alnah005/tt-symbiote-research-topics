# Decode Memory Access Patterns

During decode the model generates one new token at a time. At generation step
T (0-indexed: the T-th new token after the prompt), the query is a single
vector rather than a matrix. Attention is computed against the KV cache entries
that are currently resident in the circular buffer. This document characterises
the exact DRAM reads that this entails, compares those reads to full-attention
decode, derives the bandwidth reduction factor, and describes the data
dependency graph within the window read.

## The Decode Scenario

### Inputs and Outputs

At decode step T the attention layer receives:

- Query: `q_T` with shape `[B, H, 1, d]` — one vector per head per batch item,
  freshly projected from the new token's embedding.
- Key cache: `k_cache` with shape `[B, H, w, d]` — the circular buffer holding
  key vectors for the most recent w positions (or fewer if T < w).
- Value cache: `v_cache` with shape `[B, H, w, d]` — same structure for value
  vectors.
- Position offset scalar: `pos_offset = max(0, T - w + 1)` — the absolute
  position of the oldest entry currently in the buffer.

Before attention computation the new key and value vectors for position T are
written into the cache:

```python
k_cache[:, :, T % w, :] = k_T    # k_T shape: [B, H, 1, d]
v_cache[:, :, T % w, :] = v_T    # v_T shape: [B, H, 1, d]
pos_offset = max(0, T - w + 1)
```

After this write, the cache holds min(T+1, w) valid entries. The attention
computation then reads those entries back to produce the output:

$$o_T = \sum_{s \in \mathcal{A}_{\text{win}}(T)} \alpha_{T,s} \, v_s
\quad\text{where}\quad
\mathcal{A}_{\text{win}}(T) = \{\max(0, T-w+1), \ldots, T\}$$

### Number of Valid Cache Entries

The number of K vectors (and identically V vectors) that must be read is:

$$n_{\text{read}}(T) = \min(T + 1,\; w)$$

During the fill phase (T < w) the buffer is partially populated and n_read = T+1.
In steady state (T >= w-1) n_read = w, and it stays constant for all subsequent
steps regardless of how large T grows.

## DRAM Read Volume: Reading Exactly w K and V Vectors

### Steady-State (T >= w - 1)

Once the sequence has reached steady state, every decode step reads exactly w
key vectors and w value vectors. For a single head and batch item:

| Read          | Tensor slice      | Element count | Bytes (bfloat16) |
|---------------|-------------------|---------------|------------------|
| K vectors     | k_cache[b, h, :, :] | w · d        | w · d · 2        |
| V vectors     | v_cache[b, h, :, :] | w · d        | w · d · 2        |
| Query vector  | q_T[b, h, 0, :]   | d             | d · 2            |
| **Total**     |                   | **2·w·d + d** | **(2w+1)·d·2**   |

The query read (d · 2 bytes) is negligible compared with the K and V reads
when w >> 1. The dominant cost is 2·w·d·2 bytes of KV reads.

For H heads and B batch items in a model layer:

$$\text{bytes per decode step} = B \cdot H \cdot 2 \cdot w \cdot d \cdot 2$$

For a 7B-class model with H = 32 heads, d = 128, B = 1, w = 4096, bfloat16:

$$1 \times 32 \times 2 \times 4096 \times 128 \times 2 = 64 \text{ MiB per layer per step}$$

This is the fixed, T-independent per-step DRAM bandwidth cost of windowed
decode.

### Wrap Boundary: Two-Segment Read

The w slots to be read are not always physically contiguous in the
`[B, H, w, d]` tensor's slot dimension. The write pointer after step T is:

$$wp = (T + 1) \bmod w$$

Slot `wp` will be written next; the current valid entries occupy slots:

$$[\underbrace{wp,\; wp+1,\; \ldots,\; w-1}_{\text{older segment}},\;
 \underbrace{0,\; 1,\; \ldots,\; wp-1}_{\text{newer segment}}]$$

If `wp == 0` (the write pointer is at the start of the array) then all w slots
are contiguous: [0, 1, ..., w-1]. This is the only case with a single
contiguous read. In all other cases the valid entries span two contiguous
segments separated by the wrap boundary at index w.

```text
Physical slot layout (w = 8, T = 11, wp = 4):

  Slot index:   0     1     2     3     4     5     6     7
               ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  K cache:     │ t=8 │ t=9 │t=10 │t=11 │ t=4 │ t=5 │ t=6 │ t=7 │
               └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                ←── newer ──────────────→ ←─── older ────────────→
                  segment [0, wp-1=3]         segment [wp=4, 7]

Read order (oldest→newest): slots [4,5,6,7] then slots [0,1,2,3]
Segment 1 (older): k_cache[b, h, 4:8, :]    (4 vectors, contiguous)
Segment 2 (newer): k_cache[b, h, 0:4, :]    (4 vectors, contiguous)
```

The two-segment structure means the kernel issues two separate DRAM reads for
K and two for V (total four reads per head per step), rather than one. Each
segment is individually contiguous, so each read benefits from DRAM burst
efficiency. The number of DRAM transactions is bounded by 2 regardless of w;
there is no per-token granularity in the read path.

A kernel can eliminate the two-segment complexity by treating the `[B, H, w, d]`
buffer as a flat ring and always reading all w slots starting from index 0,
regardless of logical ordering. In this case the attention scores are computed
over physically ordered slots, and the mask must account for the wrap: slots
[wp, w-1] are older than slots [0, wp-1]. The mask is a permuted variant of
the standard causal mask. This approach avoids gather operations at the cost
of a more complex mask construction.

## Comparison with Full-Attention Decode

In full-attention decode at step T, the query attends to all T+1 prior
tokens. The KV cache is a `[B, H, T+1, d]` tensor (or a pre-allocated
`[B, H, max_T, d]` tensor with T+1 valid entries). The read volume is:

$$\text{full-attention KV reads} = B \cdot H \cdot 2 \cdot (T+1) \cdot d \cdot 2 \text{ bytes}$$

This grows linearly with T. At T = 32 768, H = 32, d = 128, B = 1:

$$1 \times 32 \times 2 \times 32768 \times 128 \times 2 = 512 \text{ MiB per layer per step}$$

versus 64 MiB for windowed attention with w = 4096 — an 8× reduction at
this point in generation. The reduction factor grows as T increases.

### Bandwidth Reduction Factor

At generation step T (steady-state, T >= w-1), the bandwidth reduction factor
of windowed decode relative to full-attention decode is:

$$\frac{\text{windowed KV reads}}{\text{full-attention KV reads}}
= \frac{2 \cdot w \cdot d}{2 \cdot (T+1) \cdot d}
= \frac{w}{T+1}
\approx \frac{w}{T}$$

For T = w (just reached steady state): ratio = 1 — no saving yet.
For T = 2w: ratio = 1/2 — bandwidth halved.
For T = 10w: ratio = 1/10.
For T = 100w: ratio = 1/100 — two orders of magnitude saving.

```text
Bandwidth reduction factor w/T as a function of T (w = 4096):

T =  4,096  (T = w)    : ratio = 1.00   (steady state just reached)
T =  8,192  (T = 2w)   : ratio = 0.50
T = 16,384  (T = 4w)   : ratio = 0.25
T = 32,768  (T = 8w)   : ratio = 0.125
T = 65,536  (T = 16w)  : ratio = 0.0625
T = 131,072 (T = 32w)  : ratio = 0.0313
```

This is the practical motivation for windowed attention in long-generation
scenarios: as T grows, the per-step bandwidth cost of full attention becomes
unsustainable on DRAM-bandwidth-limited hardware, while windowed attention
holds constant.

## Data Dependency Graph Within the Window Read

### Absence of Inter-Token Dependencies

The key structural property of windowed decode is that the w K and V vectors
read from the cache are **independent of each other** from the memory access
perspective. There is no dependency chain: reading K[slot_0] does not require
a result computed from K[slot_1]. The w reads are embarrassingly parallel at
the data level.

This can be expressed as a dependency graph where nodes are memory read
operations and edges are data-flow dependencies:

```text
Dependency graph for decode step T (w = 4, slots [s0, s1, s2, s3]):

  READ k_cache[s0] ──┐
  READ k_cache[s1] ──┤
  READ k_cache[s2] ──┤──→  QK^T dot products  ──→  softmax  ──→  score·V  ──→  o_T
  READ k_cache[s3] ──┘                                             ↑
  READ v_cache[s0] ──────────────────────────────────────────────┘
  READ v_cache[s1] ──────────────────────────────────────────────┘
  READ v_cache[s2] ──────────────────────────────────────────────┘
  READ v_cache[s3] ──────────────────────────────────────────────┘
  READ q_T         ──→  QK^T dot products
```

Each DRAM read is an independent leaf with no predecessor. The first compute
dependency appears only when the QK^T dot product consumes q_T and all w K
vectors. Similarly, the score·V weighted sum requires all w V vectors and the
softmax output, but the V reads have no dependency on the K reads or on each
other.

The implication for pipeline scheduling on Wormhole is that all w K reads (and
all w V reads) can be issued as independent DMA requests. On Wormhole's NoC
(Network-on-Chip) these requests can be pipelined: the NoC can have multiple
in-flight DRAM requests simultaneously, hiding DRAM latency behind bandwidth
utilisation. With two contiguous segments per tensor, the two K-read DMA
requests and two V-read DMA requests can all be in flight concurrently,
achieving close to peak DRAM bandwidth for a single decode step.

### Write Ordering and Hazard Analysis

The output `o_T` of decode step T is used to produce the embedding for the
next token, which in turn produces `q_{T+1}`, `k_{T+1}`, `v_{T+1}`. These
are written into the cache before step T+1's attention computation begins.
The w-1 K and V vectors from steps [T-w+1, T-1] that are re-read at step
T+1 are exactly the same physical memory that was read at step T (minus one
slot, plus one new slot). Each step reads its own snapshot of the cache, and
the cache is only mutated by the single write before each step.

At step T the write slot is `T mod w`, which held position T - w (the oldest
entry being evicted). The slot immediately after it, `(T + 1) mod w`, holds
position T - w + 1 — the oldest entry still within A_win(T) = [T - w + 1, T].
The write slot itself is also read: the write fills slot `T mod w` with the
key/value for position T, and the attention computation then reads all w slots
including slot `T mod w`. This is correct — position T (the current token)
must be included in A_win(T).

```text
Write slot at step T:  T mod w         ← written with k_T / v_T (position T)
Oldest read slot:      (T + 1) mod w   ← adjacent slot, holds position T-w+1
Newest read slot:      T mod w         ← same as write slot; holds position T
```

There is no hazard because the write completes before the read: the cache
update precedes the attention call in the kernel schedule, imposing a
write-before-read ordering on slot `T mod w`. This ordering is enforced by
the decode loop issuing `ttnn.update_cache` before
`ttnn.scaled_dot_product_attention` (or equivalent).

## Arithmetic Intensity for Decode

Unlike prefill, decode involves computing a single row of the score matrix
(one query vector against w key vectors). The compute and bandwidth breakdown
per head per batch item is:

| Operation         | FLOPs            | Notes                                      |
|-------------------|------------------|--------------------------------------------|
| QK^T scores       | 2 · w · d        | One query row dotted with w key rows       |
| Softmax           | ~5 · w           | exp, sum, divide over w entries            |
| score · V         | 2 · w · d        | Scalar-weighted sum of w value vectors     |
| **Total FLOPs**   | **~4 · w · d**   | (ignoring softmax at large d)              |

| Data             | Bytes (bfloat16)          |
|------------------|---------------------------|
| K cache read     | w · d · 2                 |
| V cache read     | w · d · 2                 |
| Query read       | d · 2                     |
| Output write     | d · 2                     |
| **Total bytes**  | **(2w + 2) · d · 2**      |

Arithmetic intensity:

$$\text{AI}_{\text{decode}}
= \frac{4 \cdot w \cdot d}{(2w + 2) \cdot d \cdot 2}
\approx \frac{4w}{4w}
= 1 \text{ FLOP/byte}$$

At AI ≈ 1 FLOP/byte, decode is deeply memory-bandwidth-bound. Wormhole's
roofline crossover is approximately 111 FLOPs/byte, so decode runs at roughly
1/111 of peak compute utilisation. This is not a kernel deficiency — it is an
inherent property of the operation: for batch size 1 there is simply not
enough arithmetic to hide the DRAM latency of loading w K and V vectors.

Increasing the batch size B multiplies both FLOPs and bytes by B, leaving AI
unchanged. The key to improving decode throughput is therefore not AI
optimisation but maximising DRAM bandwidth utilisation:

- Issue large, aligned DMA requests (the two-segment structure ensures both
  segments are aligned and of size at least w/2 · d · 2 bytes).
- Overlap K and V reads with Q projection compute where possible.
- Use bfloat16 rather than float32 to halve the DRAM traffic.

These are the micro-architectural considerations addressed in Chapter 4 when
mapping the decode step onto concrete TTNN primitives.

---

**Next:** [Chapter 4 — TTNN Primitive Operations and Tensor Shapes](../ch4_ttnn_primitives/index.md)
