# MTP Head Memory Footprint

## BF16 Weight Memory

All weight tensors on Tenstorrent hardware are stored in BF16 (Brain Float 16), which uses 2 bytes per element. The total BF16 weight memory for the MTP head is therefore:

$$\text{BF16 memory} = \text{params}_{\text{MTP}} \times 2\ \text{bytes} = 159{,}674{,}368 \times 2 = 319{,}348{,}736\ \text{bytes}$$

Converting to mebibytes:

$$\frac{319{,}348{,}736}{1024^2} = \frac{319{,}348{,}736}{1{,}048{,}576} \approx \mathbf{304.6\ \text{MiB}}$$

The breakdown by component group:

| Component group | Parameters | BF16 size |
|----------------|-----------|----------|
| Attention projections (q, k, v, o) | 115,605,504 | 220.6 MiB |
| Dense FFN (gate, up, down) | 44,040,192 | 84.1 MiB |
| Layer norms (4 vectors of 7168) | 28,672 | 0.05 MiB |
| **MTP head total** | **159,674,368** | **~304.6 MiB** |

The attention projections dominate (72% of the total), followed by the FFN projections (28%). The layer norm vectors are negligible.

---

## Comparison to One Backbone Block

The MTP head's attention sub-layer uses the same GQA configuration as the backbone (see `mtp_weight_inventory.md`), so attention weights are identical at 115.6M params / 220.6 MiB. The full weight-count and BF16-size comparison across MTP head, backbone attention sub-layer, and backbone MoE FFN is tabulated in `mtp_weight_inventory.md` § "MTP Head vs. One Backbone Block".

On a T3K with tensor parallelism across 8 devices, backbone MoE weights are sharded across chips; the MTP head weights (at 304.6 MiB total, or ~38 MiB per chip after sharding) are a small additional allocation.

---

## Activation Memory During MTP Forward Pass

The following activation sizes apply at decode (batch=1, seq_len=1, single new token). All shapes use `[B, S, H]` notation with B=1, S=1, H=7168.

| Activation tensor | Shape | Size |
|------------------|-------|------|
| Backbone hidden state input $h_t$ | `[1, 1, 7168]` | $1 \times 1 \times 7168 \times 2 = 14{,}336$ bytes $\approx$ 14 KB |
| Shifted token embedding $\text{embed}(x_{t+1})$ | `[1, 1, 7168]` | 14 KB |
| After `hnorm` and `enorm` | `[1, 1, 7168]` each | 14 KB each |
| Combined input $c_t$ (after element-wise add) | `[1, 1, 7168]` | 14 KB |
| Attention Q intermediate | `[1, 1, 64, 112]` | $1 \times 64 \times 112 \times 2 = 14{,}336$ bytes $\approx$ 14 KB |
| Attention K intermediate | `[1, 1, 8, 112]` | $1 \times 8 \times 112 \times 2 = 1{,}792$ bytes $\approx$ 1.75 KB |
| Attention V intermediate | `[1, 1, 8, 112]` | $\approx$ 1.75 KB |
| FFN intermediate (gate and up) | `[1, 1, 2048]` each | $1 \times 2048 \times 2 = 4{,}096$ bytes $\approx$ 4 KB each |
| FFN output (after down) | `[1, 1, 7168]` | 14 KB |

The total activation memory for one MTP forward pass at batch=1, seq_len=1 is on the order of **~100–120 KB** — well below one Tensix core's L1 capacity of 1.5 MB. Activations are not a concern for L1 residency at this batch size.

At larger batch sizes, activation memory scales linearly with $B$. At batch=32, the largest activation tensor (`[32, 1, 7168]`) reaches 32 × 14 KB = 448 KB, still within single-core L1. The FFN intermediate `[32, 1, 2048]` reaches 32 × 4 KB = 128 KB.

---

## Wormhole Hardware Context

**Wormhole L1 capacity:**
- L1 per Tensix core: 1.5 MiB
- Tensix cores per Wormhole chip: 72 (8×9 grid, excluding Ethernet and DRAM cores)
- Aggregate L1 per chip: $72 \times 1.5\ \text{MiB} = 108\ \text{MiB}$

**Can MTP head weights fit in L1?**

The MTP head weights total ~304.6 MiB. The aggregate L1 on one Wormhole chip is 108 MiB. Even before accounting for the backbone's own L1 allocations (activations, KV cache slices, attention buffers, and the far larger backbone weight tensors), the MTP head weights alone exceed the per-chip L1 capacity by a factor of approximately $304.6 / 108 \approx 2.8\times$.

**Conclusion: full MTP head weight L1 residency is not feasible on a single chip.** The MTP head weights must reside in DRAM and be streamed to compute cores during each decode step.

On a T3K (8 Wormhole chips) with tensor parallelism, the MTP head weights are sharded across 8 chips. Per-chip MTP weight shard: $304.6 / 8 \approx 38.1\ \text{MiB}$. Aggregate T3K L1: $8 \times 108 = 864\ \text{MiB}$. Even with this sharding, the per-chip MTP weight shard (~38.1 MiB) is a substantial fraction of one chip's 108 MiB L1, and the backbone's attention and FFN weight shards will already be competing for L1 in the decode loop. **The final placement decision is deferred to Chapter 5, `memory_placement_for_mtp.md`**, which weighs the full decode-phase L1 budget including backbone weights, KV cache slices, and activation buffers.

---

## DRAM Bandwidth Cost

At decode (batch=1), the MTP head forward pass is weight-bound: each matrix multiplication streams its weight matrix from DRAM once per token. The time to stream MTP head weights determines the DRAM bandwidth cost per decode step.

**Single-chip Wormhole DRAM bandwidth:** ~36 GB/s per chip (Wormhole has 12 DRAM channels × ~3 GB/s per channel in practice).

**T3K total DRAM bandwidth:** 8 chips × 36 GB/s = **288 GB/s** aggregate. With tensor parallelism, each chip streams its $1/8$ shard, so the parallel streaming time is determined by the per-chip shard size and per-chip bandwidth.

### MTP Head DRAM Streaming Cost

| Config | Weight bytes per chip | DRAM BW per chip | Streaming time |
|--------|----------------------|-----------------|---------------|
| Single chip (P150) | 319,348,736 bytes (~304.6 MiB) | 36 GB/s | $\approx 8.9\ \text{ms}$ |
| T3K (8-chip, tensor parallel) | 39,918,592 bytes (~38.1 MiB per chip) | 36 GB/s per chip | $\approx 1.1\ \text{ms}$ |

### Comparison to One Backbone Block

One backbone MoE block has ~5,752.7M parameters = ~10,973 MiB BF16. On T3K with 8-way tensor parallelism, each chip holds a shard of approximately $10{,}973 / 8 \approx 1{,}372\ \text{MiB}$. Streaming this shard at 36 GB/s per chip:

$$\frac{1{,}372 \times 1{,}048{,}576\ \text{bytes}}{36 \times 10^9\ \text{bytes/s}} \approx 39.9\ \text{ms}$$

**MTP head streaming time on T3K vs. one backbone MoE block:**

$$\frac{1.1\ \text{ms}}{39.9\ \text{ms}} \approx 2.8\%$$

The MTP head adds approximately 2.8% of the per-block DRAM bandwidth cost relative to one backbone MoE block. Over the full 94-block backbone, the relative overhead is $\approx 1.1 / (94 \times 39.9)\ \text{ms} \approx 0.03\%$ — entirely negligible from a bandwidth perspective.

These estimates assume that MTP weights are not pipelined with backbone computation. If MTP weights are pre-fetched during the final backbone block's compute phase, the MTP DRAM streaming latency can be fully hidden, reducing the practical per-step overhead to zero.

---

## Key Finding

> **MTP head weight memory is not the binding constraint for L1 residency.**
>
> The MTP head weights (~304.6 MiB total, ~38.1 MiB per chip on T3K) exceed the per-chip L1 capacity available in the decode loop. They must reside in DRAM. However, the DRAM streaming cost is small (~1.1 ms on T3K at batch=1) and represents less than 3% of the cost of streaming one backbone MoE block's weights. The MTP head is not a meaningful contributor to total per-step decode latency from a memory bandwidth perspective. The question of whether MTP weights could be resident in a shared L1 pool across chips, or whether they should be pre-fetched, is addressed in Chapter 5, `memory_placement_for_mtp.md`.

---

## References

- [Wormhole] Tenstorrent, "Wormhole Architecture Overview", internal documentation, 2024.
- [Qwen3] Qwen Team, "Qwen3 Technical Report", Alibaba Cloud, 2025.
- Chapter 1, `qwen36_mtp_config.md` — source of model hyperparameter values.
- Chapter 2, `mtp_weight_inventory.md` — source of the parameter count (159,674,368) used throughout this file.
- Chapter 2, `mtp_vs_backbone_compute_cost.md` — the complementary FLOP analysis for the same MTP head.
- Chapter 5, `memory_placement_for_mtp.md` — final placement decision incorporating the full decode-phase L1 budget.

---

**Next:** [`mtp_vs_backbone_compute_cost.md`](./mtp_vs_backbone_compute_cost.md)
