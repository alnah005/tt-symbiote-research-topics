# MTP Head vs. Backbone Compute Cost

## FLOP Count at Decode (batch=1, seq_len=1)

All FLOP counts below use the convention that a multiply-accumulate (MAC) counts as 2 FLOPs (one multiply plus one add). At decode (batch=1, generating one new token), the input to each matrix multiplication is a row vector of shape `[1, H]`, where `H = 7168`.

### Attention Projection FLOPs

Weight matrices use stored-checkpoint shape `[out_features, in_features]`. For a matmul with input `[1, in_features]` and weight `[out_features, in_features]`, the FLOP count is $2 \times \text{in\_features} \times \text{out\_features}$.

| Operation | Input shape | Weight shape | FLOPs |
|-----------|------------|-------------|-------|
| `q_proj` | `[1, 7168]` | `[7168, 7168]` | $2 \times 7168 \times 7168 = 102{,}760{,}448$ |
| `k_proj` | `[1, 7168]` | `[896, 7168]` | $2 \times 7168 \times 896 = 12{,}845{,}056$ |
| `v_proj` | `[1, 7168]` | `[896, 7168]` | $2 \times 7168 \times 896 = 12{,}845{,}056$ |
| `o_proj` | `[1, 7168]` | `[7168, 7168]` | $2 \times 7168 \times 7168 = 102{,}760{,}448$ |
| **Attention projections total** | | | **231,211,008** |

Expressed symbolically, using $H = 7168$, $N_q = 64$, $N_{kv} = 8$, $d = 112$:

$$\text{FLOPs}_{\text{attn proj}} = 2H(N_q d + N_{kv} d + N_{kv} d + N_q d) = 2H \cdot 2(N_q + N_{kv}) d$$

$$= 2 \times 7168 \times 2 \times (64 + 8) \times 112 = 2 \times 7168 \times 2 \times 72 \times 112 = 231{,}211{,}008$$

### Attention Matmul FLOPs (KV Cache Accesses)

In addition to the projection FLOPs above, the attention mechanism computes QK scores and weighted value aggregation against the KV cache of length $S$ (the current context length):

$$\text{FLOPs}_{\text{QK}} = 2 \times N_q \times 1 \times S \times d = 2 \times 64 \times S \times 112 = 14{,}336 \cdot S$$

$$\text{FLOPs}_{\text{AV}} = 2 \times N_q \times 1 \times S \times d = 14{,}336 \cdot S$$

$$\text{FLOPs}_{\text{attn matmuls}} = 28{,}672 \cdot S$$

At $S = 0$ (first decode step, empty KV cache), attention matmul FLOPs are zero. At $S = 2048$, they add $28{,}672 \times 2048 \approx 58.7\text{M}$ FLOPs — about 25% of the projection FLOPs. At $S = 32768$ (a long context), attention matmul FLOPs reach $\approx 939\text{M}$, exceeding the projection FLOPs. However, the attention matmuls operate on KV cache entries (already resident in memory), not on weight matrices; their memory access pattern differs from the weight-streaming regime.

The analysis below uses $S = 0$ (the weight-bound regime that characterizes early decode steps and is the worst case for weight streaming overhead).

### Dense FFN FLOPs

The MTP head's dense SwiGLU FFN (structure described in `mtp_weight_inventory.md`) contributes:

| Operation | Input shape | Weight shape | FLOPs |
|-----------|------------|-------------|-------|
| `gate_proj` | `[1, 7168]` | `[2048, 7168]` | $2 \times 7168 \times 2048 = 29{,}360{,}128$ |
| `up_proj` | `[1, 7168]` | `[2048, 7168]` | $2 \times 7168 \times 2048 = 29{,}360{,}128$ |
| SwiGLU elementwise (SiLU + multiply) | `[1, 2048]` | — | $\approx 2{,}048$ (negligible) |
| `down_proj` | `[1, 2048]` | `[7168, 2048]` | $2 \times 2048 \times 7168 = 29{,}360{,}128$ |
| **Dense FFN total** | | | **88,080,384** |

The elementwise SiLU and multiply contribute negligibly (~2K FLOPs) relative to the matrix multiplications (~88M FLOPs) and are excluded from the totals below.

### Total MTP Head FLOPs (at S=0)

```math
\text{FLOPs}_{\text{MTP}} = \text{FLOPs}_{\text{attn proj}} + \text{FLOPs}_{\text{FFN}} = 231{,}211{,}008 + 88{,}080{,}384 = \mathbf{319{,}291{,}392} \approx \mathbf{319\text{M FLOPs}}
```

The layer norm forward passes (RMS norm over a `[1, 7168]` vector) contribute approximately $4 \times 7168 \approx 28\text{K}$ multiplies = ~57K FLOPs, which is negligible relative to 319M FLOPs and is excluded.

---

## Comparison to One Backbone Block

### Backbone Attention FLOPs

The backbone uses the same GQA configuration as the MTP head, so backbone attention projection FLOPs are identical:

$$\text{FLOPs}_{\text{backbone attn proj}} = 231{,}211{,}008 \approx 231\text{M FLOPs}$$

### Backbone MoE FFN Active FLOPs

In each backbone MoE layer, 8 experts are activated per token (from 128 total). Each expert is a dense FFN with `intermediate_size = 2048` — identical in structure to the MTP head's dense FFN. The active FFN FLOPs per token:

$$\text{FLOPs}_{\text{backbone MoE FFN, active}} = 8 \times \text{FLOPs}_{\text{one dense FFN}} = 8 \times 88{,}080{,}384 = 704{,}643{,}072 \approx 705\text{M FLOPs}$$

### One Backbone MoE Block Total Active FLOPs

$$\text{FLOPs}_{\text{backbone block}} = \text{FLOPs}_{\text{attn proj}} + \text{FLOPs}_{\text{MoE FFN active}} = 231\text{M} + 705\text{M} = 936\text{M FLOPs}$$

### MTP Head vs. One Backbone Block

| Component | MTP Head | One Backbone MoE Block |
|-----------|----------|----------------------|
| Attention projections | ~231M FLOPs | ~231M FLOPs (identical) |
| FFN (active) | ~88M FLOPs (1 × dense) | ~705M FLOPs (8 × dense experts) |
| **Total (at S=0)** | **~319M FLOPs** | **~936M FLOPs** |
| **Ratio (MTP / backbone block)** | **~34%** | — |

The MTP head's FLOP count is approximately **34% of one backbone MoE block's active FLOP count**. The difference is entirely in the FFN: the MTP head runs one dense FFN sub-network while the backbone activates eight expert sub-networks per token.

---

## Fraction of Full Backbone Compute

The Qwen3.6-35B-A3B backbone contains 94 transformer layers (all of which are MoE layers, with a small number of dense exceptions; the analysis treats all 94 as MoE for a conservative upper-bound estimate of backbone compute).

$$\text{FLOPs}_{\text{backbone total}} = 94 \times 936\text{M} \approx 87{,}984\text{M} \approx 88\text{B FLOPs}$$

$$\frac{\text{FLOPs}_{\text{MTP}}}{\text{FLOPs}_{\text{backbone total}}} = \frac{319\text{M}}{87{,}984\text{M}} \approx 0.36\%$$

**The MTP head adds approximately 0.36% of the full backbone's active FLOP count.** As a rough cross-check: the MTP head is equivalent to $319/936 \approx 0.34$ backbone blocks, and $0.34/94 \approx 0.36\%$.

Since the MTP head uses a single dense FFN while the backbone uses MoE with 8 active experts, the MTP head is actually cheaper per FLOP than a backbone block of the same attention size. If the backbone used dense FFNs throughout (as in the non-MoE Qwen3 dense models), each backbone block would have only 88M FFN FLOPs, and the MTP head's FFN cost would equal exactly one backbone block's FFN cost, yielding $1/94 \approx 1.1\%$ of backbone FFN compute — still a small overhead.

---

## Arithmetic Intensity

Arithmetic intensity quantifies whether a kernel is compute-bound (limited by FLOPs/second throughput) or memory-bandwidth-bound (limited by GB/s bandwidth):

$$\text{Arithmetic intensity} = \frac{\text{FLOPs}}{\text{bytes loaded from memory}}$$

At decode (batch=1, $S = 0$), each matrix multiplication loads its weight matrix once from memory. The bytes loaded equal the weight size in BF16.

$$\text{AI}_{\text{MTP}} = \frac{319{,}291{,}392\ \text{FLOPs}}{319{,}348{,}736\ \text{bytes}} \approx \mathbf{1.0\ \text{FLOPs/byte}}$$

This is essentially 1:1 — one floating-point operation per byte of weight data loaded. This figure is well below the arithmetic intensity ridge point on Wormhole hardware. Wormhole's Tensix cores have a theoretical peak compute of ~32 TFLOPs/s (BF16) per chip and a DRAM bandwidth of ~36 GB/s per chip, giving a ridge point of approximately:

$$\frac{32 \times 10^{12}\ \text{FLOPs/s}}{36 \times 10^9\ \text{bytes/s}} \approx 889\ \text{FLOPs/byte}$$

At $\text{AI} \approx 1.0\ \text{FLOPs/byte}$, the MTP head forward pass at batch=1 is approximately **889× below the ridge point** — firmly in the memory-bandwidth-bound regime. The practical execution time is determined almost entirely by how fast the weight matrices can be streamed from DRAM, not by how fast the cores can execute multiply-accumulate operations.

As batch size $B$ increases, the arithmetic intensity scales as:

$$\text{AI}(B) = \frac{B \times 319\text{M FLOPs}}{319\text{M bytes}} \approx B\ \text{FLOPs/byte}$$

The MTP head becomes compute-bound when $\text{AI}(B) \gtrsim 889$, i.e., at $B \gtrsim 889$. For typical speculative decoding use cases (batch=1 to batch=32), the MTP head remains memory-bandwidth-bound throughout.

**Implication for adding MTP to the decode loop:** Since both the backbone and the MTP head are memory-bandwidth-bound at decode-relevant batch sizes, adding the MTP head is essentially adding more weight bytes to stream per decode step. The latency overhead of the MTP head is:

$$\Delta t_{\text{MTP}} \approx \frac{\text{MTP weight bytes per chip}}{\text{DRAM BW per chip}}$$

On T3K at batch=1, this is approximately 1.1 ms per decode step (derived in `mtp_memory_footprint.md`). If MTP weights can be pre-fetched during the tail of the backbone computation (pipelining weight DMA with the previous layer's compute), this overhead can be reduced further or eliminated entirely.

---

## Key Finding

> **At batch=1 decode (the common speculative-decode use case), the MTP head adds approximately 0.36% of the full backbone's active FLOP count.** Since both the backbone and the MTP head are memory-bandwidth-bound on Wormhole hardware (arithmetic intensity ~1 FLOPs/byte, well below the ~889 FLOPs/byte ridge point), the practical latency overhead is not determined by FLOP count at all. It is determined by the additional weight-streaming time (~1.1 ms on T3K at batch=1), which represents approximately 2.8% of the streaming cost of one backbone MoE block. The MTP head's compute overhead is negligible; the relevant cost is its DRAM bandwidth cost, which is analyzed in `mtp_memory_footprint.md`.

---

## References

- [Wormhole] Tenstorrent, "Wormhole Architecture Overview", internal documentation, 2024.
- [Qwen3] Qwen Team, "Qwen3 Technical Report", Alibaba Cloud, 2025.
- [DeepSeek-V3] DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437, 2024.
- Chapter 1, `qwen36_mtp_config.md` — source of model hyperparameter values (H, heads, head_dim, intermediate_size).
- Chapter 2, `mtp_weight_inventory.md` — weight tensor shapes and parameter count used to derive FLOPs and bytes.
- Chapter 2, `mtp_memory_footprint.md` — DRAM bandwidth cost analysis complementing the FLOP analysis here.
- Chapter 4, `throughput_analysis_on_tt_hardware.md` — uses the FLOP ratios derived here to estimate end-to-end speculative decoding throughput.

---

**Next:** [Chapter 3 — MTP in HuggingFace Transformers](../ch03_mtp_in_huggingface/index.md)
