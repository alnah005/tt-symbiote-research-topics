# Roofline Analysis for Windowed Attention on Wormhole

This document determines whether windowed attention decode is compute-bound or
bandwidth-bound on a single Wormhole chip, quantifies the expected throughput
advantage over full attention, and analyses how batch size shifts the operating
point on the roofline curve.

## Wormhole Hardware Numbers

The roofline model requires two peak hardware limits for a single Wormhole chip:

| Resource                        | Value            | Notes                                             |
|---------------------------------|------------------|---------------------------------------------------|
| Peak FP16 matrix-multiply TFLOPS | ~32 TFLOPS       | Tensor Engine (matrix unit); single chip          |
| Peak DRAM bandwidth             | ~288 GB/s        | Aggregate across all DRAM channels; single chip   |
| L1 SRAM per core                | 1,536 KiB        | Wormhole Tensix core local SRAM                   |
| NoC bisection bandwidth         | ~240 GB/s        | On-chip network; typically not the bottleneck for DRAM-bound ops |
| Number of Tensix cores          | 64 (8×8 grid)    | Available for compute in a standard program config |

The roofline crossover arithmetic intensity — the AI at which peak compute and
peak bandwidth are simultaneously saturated — is:

```math
\text{AI}_{\text{crossover}}
= \frac{\text{Peak TFLOPS}}{\text{Peak bandwidth}}
= \frac{32 \times 10^{12}\ \text{FLOPs/s}}{288 \times 10^{9}\ \text{bytes/s}}
\approx 111\ \text{FLOPs/byte}
```

Any operation with AI below 111 FLOPs/byte is bandwidth-bound. Any operation
above 111 FLOPs/byte is compute-bound. For context, a pure matrix multiply on
large square matrices achieves AI ~ 10,000 FLOPs/byte (deeply compute-bound),
while a simple elementwise operation achieves AI ~ 0.5 FLOPs/byte (deeply
bandwidth-bound).

## Arithmetic Intensity of Windowed Attention Decode

### Setup

The decode path at generation step T processes one new query vector per head
against the `w`-slot circular-buffer KV cache. The notation follows the
guide-wide convention:

- `B` — batch size
- `H_q` — number of query heads
- `H_kv` — number of key/value heads (`H_q / H_kv` is an integer GQA group size)
- `w` — window size (number of KV slots in the circular buffer)
- `d` — head dimension
- dtype bytes: 2 (bfloat16)

### FLOP Count

Each decode step performs two matrix multiplications per head (QK scores and
attention-weighted value aggregation):

```math
\text{FLOPs}_{\text{QK}}   = 2 \times H_q \times 1 \times w \times d
                           = 2 \cdot H_q \cdot w \cdot d
```

```math
\text{FLOPs}_{\text{AV}}   = 2 \times H_q \times 1 \times d \times w
                           = 2 \cdot H_q \cdot w \cdot d
```

```math
\text{FLOPs}_{\text{total}} = 4 \cdot H_q \cdot w \cdot d
\quad\text{(per batch item, ignoring softmax at large } d\text{)}
```

The factor of 2 in each term accounts for the multiply-accumulate structure
(one multiply + one addition per element). Softmax contributes O(w) FLOPs
which is negligible compared to O(w·d) for typical d = 64 or 128.

When GQA is active (H_q > H_kv), the same K and V head serves `H_q / H_kv`
query heads. The FLOPs above use H_q throughout because each of the H_q query
heads still performs a full inner product against w key (or value) vectors,
regardless of whether those vectors are shared.

### Byte Count

The dominant data movement is loading the K and V caches from DRAM:

```math
\text{bytes}_{\text{K cache}} = H_{kv} \times w \times d \times 2
```

```math
\text{bytes}_{\text{V cache}} = H_{kv} \times w \times d \times 2
```

```math
\text{bytes}_{\text{Q}}      = H_q \times 1 \times d \times 2
\quad\text{(negligible for } w \gg 1\text{)}
```

```math
\text{bytes}_{\text{output}} = H_q \times 1 \times d \times 2
\quad\text{(negligible for } w \gg 1\text{)}
```

```math
\text{bytes}_{\text{total}} \approx 2 \cdot H_{kv} \cdot w \cdot d \cdot 2
                             = 4 \cdot H_{kv} \cdot w \cdot d
\quad\text{(KV dominant)}
```

### Arithmetic Intensity Formula

```math
\text{AI}_{\text{decode}}
= \frac{\text{FLOPs}_{\text{total}}}{\text{bytes}_{\text{total}}}
= \frac{4 \cdot H_q \cdot w \cdot d}{4 \cdot H_{kv} \cdot w \cdot d}
= \frac{H_q}{H_{kv}}
```

For **multi-head attention (MHA)** where H_q = H_kv:

```math
\text{AI}_{\text{MHA decode}} = \frac{H_q}{H_{kv}} = 1\ \text{FLOP/byte}
```

For **grouped query attention (GQA)** with group size G = H_q / H_kv:

```math
\text{AI}_{\text{GQA decode}} = G = \frac{H_q}{H_{kv}}
```

Typical GQA configurations (e.g., G = 4 for a 32-head / 8-head GQA model)
yield AI = 4 FLOPs/byte. Even at G = 8 (AI = 8 FLOPs/byte), the operation
remains over an order of magnitude below the 111 FLOPs/byte crossover.

### Key Insight: AI Does Not Depend on w or T

The window size `w` cancels out of the AI formula entirely: changing `w`
scales both FLOPs and bytes proportionally, leaving their ratio unchanged.
**Windowed attention does not change the bandwidth-bound characterisation** of
decode; it merely reduces the absolute bandwidth requirement by a factor of
`w / T` relative to full attention, increasing throughput accordingly.

## Roofline Position: Bandwidth-Bound Regime

With AI ≈ 1–8 FLOPs/byte (for GQA groups 1–8) versus a crossover of
111 FLOPs/byte, windowed attention decode sits at approximately
**1% to 7% of the crossover intensity**. The roofline model predicts that
performance (in FLOPs/s achieved) scales linearly with AI and DRAM bandwidth,
not with peak TFLOPS:

```math
\text{Achieved throughput (FLOP/s)}
= \text{AI} \times \text{Bandwidth (bytes/s)}
= G \times 288 \times 10^9\ \text{FLOP/s}
```

For MHA (G = 1): `1 × 288 GB/s = 288 GFLOP/s` — versus a peak of 32 TFLOPS.
The compute units are idle more than 99% of the time.

```text
Roofline diagram for single Wormhole chip (log-log, batch=1):

  Peak throughput (FLOP/s)
  |
  32T ─────────────────────────────────────────────────/ compute-bound
  |                                                   /
  |                                                  /  roofline ceiling
  |                                                 /
  |  bandwidth-bound                               / crossover
  |  (slope = bandwidth = 288 GB/s)               / AI = 111
  |                                              /
  288G ────────────────────────────────────────/────── peak BW line
  |              ^  ^  ^                      /
  |              |  |  |                     /
  |       MHA   GQA G=4 G=8                 /
  |        AI=1  4   8                     /
  |                                        /
  +──────────────────────────────────────────────────────→ AI (FLOP/byte)
     0.1   1   4   8      32     64    111
```

The caret markers show where MHA and GQA-4/8 windowed decode operate. All are
solidly in the bandwidth-bound region.

## Comparison: Windowed vs Full Attention at Various T and w

The following table shows the DRAM bandwidth requirement per decode step per
attention layer, and the resulting expected decode throughput, for windowed
attention at different window sizes versus full attention at different sequence
lengths.

Parameters: H_kv = 8, H_q = 32 (GQA group size G = 4), d = 128, B = 1, BF16.

```math
\text{BW required (bytes/step/layer)}
= B \cdot 2 \cdot H_{kv} \cdot n_{\text{read}} \cdot d \cdot 2
```

where n_read = w for windowed (steady state) or T for full attention.

```math
\text{Expected throughput (layers/s)}
= \frac{\text{Peak BW (bytes/s)}}{\text{BW required (bytes/step/layer)}}
```

| Mode              | n_read | BW required per layer | Throughput (layers/s) | vs full T=32768 |
|-------------------|--------|-----------------------|-----------------------|-----------------|
| Full attn T=4096  | 4,096  | 16.0 MiB              | 17,166                | 8×              |
| Full attn T=8192  | 8,192  | 32.0 MiB              | 8,583                 | 4×              |
| Full attn T=32768 | 32,768 | 128.0 MiB             | 2,146                 | 1× (baseline)   |
| Windowed w=4096   | 4,096  | 16.0 MiB              | 17,166                | 8×              |
| Windowed w=8192   | 8,192  | 32.0 MiB              | 8,583                 | 4×              |

Notes:
- "Full attn T=4096" and "Windowed w=4096" have identical bandwidth requirements
  and throughput — windowed attention at T = w provides no bandwidth saving over
  full attention at the same T. The saving grows as T grows beyond w.
- All throughput figures are for a single chip, single batch item, single layer,
  assuming 100% DRAM bandwidth utilisation (theoretical maximum).
- Practical throughput is 60–80% of the theoretical maximum due to DRAM row
  activation overhead and alignment constraints.

## Effect of Increasing Batch Size

The analysis above assumes B = 1. Increasing the batch size multiplies both
FLOPs and bytes by B (each batch item uses the same w-slot KV cache size),
leaving AI unchanged. The bandwidth requirement grows as:

```math
\text{BW required} = B \cdot 2 \cdot H_{kv} \cdot w \cdot d \cdot 2
```

At some batch size the required bandwidth exceeds the chip's peak DRAM
bandwidth. At that point the chip can no longer sustain one decode step per
cycle and the throughput saturates. However, the compute resources also scale
with B, and at B_crossover, the total FLOP rate (batch × FLOPs_per_seq)
reaches the 32 TFLOPS compute ceiling — the system transitions from
bandwidth-limited to compute-limited throughput. The arithmetic intensity of a
single sequence is unchanged; it is the aggregate work per memory transfer that
saturates the ALUs:

```math
B_{\text{crossover}}
= \frac{\text{AI}_{\text{crossover}}}{G}
= \frac{111}{G}
```

For GQA group G = 4: B_crossover ≈ 28. For MHA (G = 1): B_crossover ≈ 111.

At batch sizes below B_crossover the chip is bandwidth-bound and throughput
scales linearly with B. At batch sizes above B_crossover the chip is
compute-bound and throughput saturates (additional batch items do not increase
throughput because the Tensor Engine is fully utilised).

```text
Throughput vs batch size (windowed, G=4, single chip, theoretical):

  Throughput
  (tokens/s)
  |
  |                   _______________  ← compute ceiling (32 TFLOPS / AI)
  |                  /
  |                 /
  |                /  bandwidth-bound: throughput ∝ B
  |               /
  |              /
  | B_cross ≈ 28/
  |____________/
  +──────────────────────────────→ Batch size B
  0       16      32      64
```

In production inference serving, typical decode batch sizes are 8–32 for
latency-optimised deployments. At these batch sizes the chip remains
bandwidth-bound (or at the knee), confirming that maximising DRAM bandwidth
utilisation — not compute throughput — is the correct optimisation target for
windowed attention decode.

## Implications for Implementation

The roofline analysis yields four actionable conclusions for the implementation
choices made in Chapters 4–6:

1. **Use bfloat16 for KV cache storage.** Switching to float32 doubles the
   DRAM traffic with no benefit to decode accuracy; it halves throughput.
   Bfloat16 is the correct dtype for KV cache elements.

2. **Window size w directly controls throughput.** Halving w (e.g., from 8192
   to 4096) doubles decode throughput. Model architects must select w based on
   the accuracy/throughput trade-off; from a hardware perspective, smaller is
   strictly better.

3. **Head-parallel T3K sharding is correct.** As established in
   [`../ch6_t3k_sharding/sharding_strategies.md`](../ch6_t3k_sharding/sharding_strategies.md),
   head-parallel sharding distributes the bandwidth load across 8 chips, each
   handling H_kv/8 KV heads. The per-chip bandwidth requirement drops by 8×,
   and the 8 chips collectively saturate 8× more total DRAM bandwidth. The
   throughput scales linearly with device count in the bandwidth-bound regime.

4. **A band-mask-aware Flash Attention kernel for prefill would yield a
   real benefit.** Prefill AI (derived in
   [`../ch3_data_dependencies/prefill_access_patterns.md`](../ch3_data_dependencies/prefill_access_patterns.md))
   is `T·w / (w + T)`, which for T >> w approaches w FLOPs/byte and can
   be compute-bound for large w. Note: this formula applies to the
   Flash-Attention style tiled implementation where KV tiles of size w are
   reused across query tiles; it counts bytes as Q (length T) plus KV
   (length w, the working-set window), giving AI = T·w/(T+w). For the
   naive full-sequence path (full T-length K and V tensors, no tile reuse),
   AI ≈ G·w/d FLOPs/byte (where G = H_q/H_kv and d is the per-head
   dimension; for MHA G=1 this simplifies to w/d), because FLOPs scale as
   4·H_q·T·w while bytes scale as 4·H_kv·T·d (loading full-length K and V
   tensors), giving AI = (H_q/H_kv)·(w/d) = G·w/d. For example, with G=4,
   w=4096, d=128 the naive prefill AI is 4×4096/128 = 128 FLOPs/byte. The
   `T·w/(T+w)` formula is always ≤ w (and thus ≤ G·w/d·d = G·w) and gives a
   slightly more pessimistic (lower) AI; either way, at w=4096 the AI is
   well above the roofline crossover of 111 FLOPs/byte — 128 FLOPs/byte
   exceeds the roofline crossover of 111 FLOPs/byte, confirming the
   compute-bound regime for T >> w.
   Skipping fully-masked tiles in prefill eliminates O(T²) − O(T·w) wasted
   FLOPs; the throughput benefit grows quadratically with T. For decode, no
   such optimisation is available because the operation is bandwidth-bound at
   all practical batch sizes below B_crossover.

---

**Next:** [`existing_kernel_survey.md`](./existing_kernel_survey.md)
