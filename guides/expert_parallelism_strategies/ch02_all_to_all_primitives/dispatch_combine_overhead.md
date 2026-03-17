# Dispatch and Combine Overhead

## Purpose

This file quantifies the performance cost of the two all-to-all collectives — `all_to_all_dispatch` and `all_to_all_combine` — relative to the expert FFN computation they bracket. It develops a latency decomposition for a full MoE layer, characterizes the arithmetic intensity of each component, provides a roofline sketch for the T3K hardware, and identifies the batch-size crossover threshold at which communication time equals compute time.

**Prerequisite:** `collective_communication_background.md` (bandwidth and latency model); `all_to_all_dispatch.md` and `all_to_all_combine.md` (buffer layouts and communication volumes); Chapter 1 for the Qwen3.5-35B model parameters.

---

## End-to-End MoE Layer Latency Breakdown

A single MoE layer on an $N = 8$-device T3K system consists of six sequential components. The following table lists each component, its primary cost driver, and a rough cost formula. All formulas use the model parameters $H = 7168$, $E = 256$, $k = 8$, $N = 8$, $E_d = E/N = 32$, and $C$ for expert capacity.

> **Note on $D$:** $D$ denotes the per-expert FFN intermediate dimension (the hidden dimension inside each expert's feed-forward network). This value is **unverified** for Qwen3.5-35B — treat all formulas involving $D$ as estimates pending confirmation against the Qwen3 Technical Report. See `[D UNVERIFIED]` markers below.

### Component Table

All values are **per-device** costs for one T3K chip.

| # | Component | Primary cost driver | Rough cost formula |
|---|---|---|---|
| 1 | Router compute | Matrix multiply $[B, H] \times [H, E]$ | $2BHE$ FLOPs |
| 2 | Pre-dispatch packing | Gather/scatter across $B \times k$ token-expert pairs | $O(BkH)$ memory ops |
| 3 | `all_to_all_dispatch` collective | Network bandwidth over $N-1 = 7$ Ethernet links | $(N-1) \cdot C \cdot E_d \cdot H \cdot 2 \text{ bytes} / \text{BW}$ |
| 4 | Expert FFN compute | $C$ tokens × $E_d$ local experts; assuming balanced load ($k=N$): $C = B/E_d$ | $6(Bk/N)HD = 6BHD$ FLOPs per device `[D UNVERIFIED]` |
| 5 | `all_to_all_combine` collective | Same as dispatch (identical buffer sizes) | $(N-1) \cdot C \cdot E_d \cdot H \cdot 2 \text{ bytes} / \text{BW}$ |
| 6 | Combine unpack + accumulation | Weighted sum over $[B, k, H]$ receive buffer | $O(BkH)$ memory ops + $2BkH$ FLOPs |

> **Table footnote — Expert FFN formula:** The per-device FFN cost is $E_d \times C \times 6HD$ where $E_d = 32$ local experts each process $C$ token slots. Under uniform routing $C = Bk/E = B/32$, giving $32 \times (B/32) \times 6HD = 6BHD$ per device. The cluster-total (all $N$ devices combined) is $6BkHD$, but this is **not** the per-device cost shown here.

### Cost Formula Details

**Component 1 — Router compute:**

$$T_\text{router} = \frac{2BHE}{\text{TFLOP}_\text{peak}} = \frac{2 \times B \times 7168 \times 256}{\text{TFLOP}_\text{peak}}$$

For $B = 32$: $2 \times 32 \times 7168 \times 256 = 117{,}440{,}512 \approx 117$ MFLOPs. At 262 TFLOPs/chip (theoretical peak; see Section 4), $T_\text{router} \approx 0.45\,\mu\text{s}$. This is very fast relative to the collective steps.

**Component 2 — Pre-dispatch packing:**

The packing kernel reads each of the $B \times k = 32 \times 8 = 256$ token-expert pairs and writes the corresponding token embedding (length $H$) into the packed send buffer. The cost is dominated by memory bandwidth:

$$T_\text{pack} \approx \frac{Bk \cdot H \cdot 2}{\text{mem\_BW}_\text{device}}$$

For $B = 32$, $H = 7168$ at BF16: $32 \times 8 \times 7168 \times 2 = 3{,}670{,}016$ bytes $\approx 3.5$ MiB (~3.67 MB) read. At a device DRAM bandwidth of ~200 GB/s (Wormhole B0 approximate figure), $T_\text{pack} \approx 3{,}670{,}016 / (200 \times 10^9) \approx 18.4\,\mu\text{s}$. This is non-negligible; efficient packing implementations avoid a full DRAM round-trip by keeping the token batch in L2/L1.

**Component 3 and 5 — Dispatch and combine collectives:**

Each collective transfers $C \times E_d \times H \times 2$ bytes to each of the $N-1 = 7$ remote devices:

$$T_\text{collective} = \frac{(N-1) \cdot C \cdot E_d \cdot H \cdot 2}{\text{BW}_\text{link}}$$

where $\text{BW}_\text{link} \approx 12.5\,\text{GB/s}$ is the Tenstorrent T3K per-link Ethernet bandwidth. For a concrete example with $B = 32$, $k = 8$, $E = 256$: the expected tokens per expert is $Bk/E = 32 \times 8 / 256 = 1$, so $C \geq 1$; taking $C = 1$ (tile-aligned $C$ may round up to 32 — see the tile alignment discussion in `all_to_all_dispatch.md`):

$$T_\text{collective}(C=1) = \frac{7 \times 1 \times 32 \times 7168 \times 2}{12.5 \times 10^9} = \frac{3{,}211{,}264}{12.5 \times 10^9} \approx 257\,\mu\text{s}$$

This is the dominant latency term at small batch sizes. With $C = 32$ (tile-aligned minimum), the volume increases 32× to $\approx 8.2$ ms — motivating care in choosing the minimum valid $C$.

**Component 4 — Expert FFN compute:** `[D UNVERIFIED — verify against Qwen3 Technical Report before using]`

Each active expert performs two matrix multiplies (gate projection $[C, H] \times [H, D]$ and down projection $[C, D] \times [D, H]$) plus a SiLU activation. See `all_to_all_combine.md` for the weighted accumulation formula; the per-expert FFN cost follows from the $6HD$ FLOPs model.

Total FFN FLOPs across all $k = 8$ active experts per token, for $B$ tokens:

$$T_\text{FFN} = \frac{6BkHD}{\text{TFLOP}_\text{peak}} \quad \text{[D UNVERIFIED]}$$

**Component 6 — Combine unpack + accumulation:**

The accumulation step reads the $[B, k, H]$ receive buffer and writes the $[B, H]$ output:

$$T_\text{accum} \approx \frac{Bk \cdot H \cdot 2}{\text{mem\_BW}_\text{device}} + \frac{2BkH}{\text{TFLOP}_\text{peak}}$$

The arithmetic term ($2BkH$ FLOPs for $k$ multiply-adds per token per output element) is negligible compared to the memory term at small batch sizes.

---

## Communication vs. Compute Crossover

### Total Communication Volume Per Device (Dispatch + Combine)

The combined volume for both collectives (dispatch out + combine in, summed over all $N-1 = 7$ remote devices) is:

$$V_\text{comm} = 2 \times (N-1) \times C \times E_d \times H \times 2 \text{ bytes}$$

For $N = 8$, $E_d = 32$, $H = 7168$, BF16:

$$V_\text{comm} = 2 \times 7 \times C \times 32 \times 7168 \times 2 = 6{,}422{,}528 \times C \text{ bytes}$$

With the minimum physically valid capacity $C = 1$ and $B = 32$:

$$V_\text{comm}(C=1) = 6{,}422{,}528 \text{ bytes} \approx 6.1 \text{ MB per device}$$

At T3K Ethernet bandwidth of $\text{BW} \approx 12.5\,\text{GB/s}$ per link, using the ring all-to-all model ($N-1 = 7$ sequential rounds, each sending one slice over one link):

$$T_\text{comm}(B=32, C=1) = \frac{V_\text{comm}/2}{\text{BW}} = \frac{3{,}211{,}264}{12.5 \times 10^9} \approx 257\,\mu\text{s} \text{ per collective}$$

$$T_\text{comm, total}(B=32, C=1) \approx 514\,\mu\text{s}$$

> **Note on ring all-to-all model:** The formula $T = (N-1) \times s / \text{BW}_\text{link}$ reflects the ring algorithm: in each of the $N-1 = 7$ rounds, a device sends one slice of volume $s = C \cdot E_d \cdot H \cdot 2$ bytes over a single link, so the total bytes carried by the bottleneck link equals $(N-1) \times s$. This is the conservative (pessimistic) model and is appropriate for a T3K linear topology where non-adjacent devices have no direct cross-link. On a mesh with shared hops, effective per-hop bandwidth may be lower still. Actual performance should be measured on hardware.

### Total Expert FFN Compute Per Device `[D UNVERIFIED]`

Each device runs $E_d = 32$ local experts. For batch size $B$ tokens, the expected number of tokens dispatched to device $d$ is $B \times k / N = B \times 8 / 8 = B$ (under uniform routing). Each expert receives $B/E_d = B/32$ tokens on average.

Total FFN FLOPs per device:

$$F_\text{FFN} = E_d \times C \times 6HD = 32 \times C \times 6 \times 7168 \times D \quad \text{[D UNVERIFIED]}$$

(Using $C$ as the token count per expert, which under uniform routing equals the actual load when $C = B/E_d$.)

FFN compute time:

$$T_\text{FFN} = \frac{32 \times C \times 6 \times 7168 \times D}{\text{TFLOP}_\text{peak}} = \frac{1{,}376{,}256 \times C \times D}{262 \times 10^{12}} \approx 5.25 \times 10^{-9} \times C \times D \text{ seconds} \quad \text{[D UNVERIFIED]}$$

### Crossover Condition

Communication dominates when $T_\text{comm} > T_\text{FFN}$ and compute dominates when $T_\text{FFN} > T_\text{comm}$.

Setting $T_\text{comm, total} = T_\text{FFN}$ and solving for $C$:

$$2 \times \frac{(N-1) \times C \times E_d \times H \times 2}{\text{BW}} = \frac{32 \times C \times 6 \times H \times D}{\text{TFLOP}_\text{peak}}$$

The $C$ factor cancels (capacity affects both sides equally under uniform load):

$$\frac{2(N-1) \times E_d \times 2}{\text{BW}} = \frac{32 \times 6 \times D}{\text{TFLOP}_\text{peak}}$$

$$D^* = \frac{2(N-1) \times E_d \times 2 \times \text{TFLOP}_\text{peak}}{32 \times 6 \times \text{BW}} \quad \text{[D UNVERIFIED — this gives the D at which comm = compute at any } C\text{]}$$

Substituting numbers ($N-1 = 7$, $E_d = 32$, $\text{BW} = 12.5\,\text{GB/s}$, $\text{TFLOP}_\text{peak} = 262 \times 10^{12}$):

$$D^* = \frac{2 \times 7 \times 32 \times 2 \times 262 \times 10^{12}}{32 \times 6 \times 12.5 \times 10^9} = \frac{234{,}752 \times 10^{12}}{2400 \times 10^9} = \frac{234{,}752}{2400} \approx 97{,}813 \quad \text{[D UNVERIFIED]}$$

This result ($D^* \approx 97{,}813$) is far larger than any plausible expert FFN intermediate dimension, which suggests that **at the model parameters of Qwen3.5-35B (with $k = N = 8$), communication dominates over expert FFN compute for all realistic values of $D$** — unless D is extraordinarily large. `[D UNVERIFIED — this conclusion depends on verifying D against the Qwen3 Technical Report]`

If, for example, $D \approx 2048$ (a plausible intermediate dimension for a 35B-class model): `[D UNVERIFIED]`

$$T_\text{FFN}(C=1) \approx 5.25 \times 10^{-9} \times 1 \times 2048 \approx 10.75\,\mu\text{s}$$

versus $T_\text{comm, total}(C=1) \approx 514\,\mu\text{s}$. Communication is roughly **50× slower** than expert FFN compute at $B = 32$, $C = 1$. `[D UNVERIFIED]`

The crossover batch size $B^*$ (where comm time equals compute time, treating $B$ as continuous) is derived in Section 5.

---

## Arithmetic Intensity of Dispatch and Combine

### Dispatch: Pure Data Movement

`all_to_all_dispatch` is a data movement operation. It reads token embeddings from the send buffer and writes them to the receive buffer at the destination device via Ethernet DMA. The operation performs no arithmetic on the token data; it only reads and writes $H$-dimensional vectors.

- **Data volume per device pair:** $C \times E_d \times H \times 2$ bytes
- **FLOPs per device pair:** 0
- **Arithmetic intensity:** 0 FLOPs/byte (memory-bound / network-bound)

The dispatch collective's throughput is bounded entirely by Ethernet link bandwidth and DMA engine throughput. There is no compute-bound regime for the collective itself.

### Combine: Pure Data Movement

`all_to_all_combine` has the same structure as dispatch — it moves expert output vectors across the network with no arithmetic on the payload. Its arithmetic intensity is also 0 FLOPs/byte.

The weighted accumulation that follows combine involves $2Bk$ multiply-adds per output element (one per expert slot), yielding:

$$\text{AI}_\text{accum} = \frac{2BkH}{BkH \times 2 + BH \times 2} = \frac{2k}{2k + 2} = \frac{k}{k+1} = \frac{8}{9} \approx 0.89 \text{ FLOPs/byte}$$

This is very low arithmetic intensity — the accumulation is read-bandwidth-bound, not compute-bound.

### Expert FFN: Arithmetic Intensity Analysis `[D UNVERIFIED]`

The expert FFN is a two-matrix-multiply operation (SwiGLU structure: three linear projections). `[D UNVERIFIED — all formulas in this section depend on D]`

For a single expert processing $C$ tokens:
- **FLOPs:** $6CHD$ (three matrix multiplies of shape $[C, H] \times [H, D]$, $[C, H] \times [H, D]$, and $[C, D] \times [D, H]$, each contributing $2CHD$ FLOPs)
- **Weight bytes read:** $3 \times H \times D \times 2 = 6HD$ bytes (three weight matrices, read once from DRAM per forward pass if weights don't fit in L1)
- **Activation bytes:** $CHD \times 2$ bytes read + $CHD \times 2$ bytes written per matmul — but these stream through L1 and may be counted separately

For the **weight-stationary** regime (large $C$, weights fit in L1 or L2):

$$\text{AI}_\text{FFN, weight-stationary} = \frac{6CHD}{6HD \times 2} = \frac{C}{2} \text{ FLOPs/byte} \quad \text{[D UNVERIFIED]}$$

At $C = 32$: $\text{AI} = 16\,\text{FLOPs/byte}$, clearly compute-bound on Wormhole B0 (which has a compute-to-bandwidth ratio of roughly $262\,\text{TFLOPs} / 200\,\text{GB/s} = 1310\,\text{FLOPs/byte}$).

For the **weight-streaming** regime (small $C$, weights streamed from DRAM every forward pass):

$$\text{AI}_\text{FFN, weight-streaming} = \frac{6CHD}{6HD \times 2 + 2 \times 3CHD} = \frac{6CHD}{12HD + 6CHD} = \frac{C}{2 + C} \text{ FLOPs/byte} \quad \text{[D UNVERIFIED]}$$

At $C = 1$: $\text{AI} = 1/3\,\text{FLOPs/byte}$ — deeply memory-bound. The expert FFN is in the memory-bound regime for the small decode batch sizes typical in autoregressive inference.

---

## Roofline Sketch for T3K

### Hardware Parameters

| Parameter | Value | Source |
|---|---|---|
| T3K Ethernet bandwidth per link | ~12.5 GB/s | T3K datasheet (consult for authoritative value) |
| T3K Ethernet links per device (outbound) | 7 (one per remote device in 8-device mesh) | T3K topology |
| Wormhole B0 peak BF16 matmul throughput | ~262 TFLOPs/chip | Theoretical: 80 Tensix cores × ~3.27 TFLOPs/core |
| Wormhole B0 peak DRAM bandwidth | ~200 GB/s | Approximate; verify against hardware datasheet |

> **Note on theoretical peak:** The 262 TFLOPs/chip figure is a theoretical peak computed as 80 Tensix cores × approximately 3.27 TFLOPs/core (derived from 2 FMAs/cycle × 32 ops/FMA × 1 GHz × 64-way SIMD, or similar micro-architectural estimates). Real-world sustained throughput on large matrix multiplies is lower. These figures should be validated against benchmark measurements on actual Wormhole B0 hardware.

### Roofline Diagram (Textual)

```text
Arithmetic Intensity (FLOPs/byte)
0.01   0.1    1      10     100    1000   10000
|-------|-------|-------|-------|-------|-------|
                                          ^
                                          | Compute ceiling (262 TFLOPs/chip)
                                          |
         ^ Dispatch/Combine               |
         | AI = 0 (network-bound, not     |
         | on roofline; bandwidth = BW)   |
                                          |
               ^ Accum (AI ≈ 0.89)        |
                                          |
                    ^ FFN small C (AI < 1) |
                                          |
                              ^ FFN large C (AI >> 1)
                                          ^
Memory bandwidth bound (200 GB/s DRAM) --|
                                 Ridge point: 262T / 200G ≈ 1310 FLOPs/byte
```

### Quantitative Crossover for B=32 `[D UNVERIFIED]`

Using $C = 1$ (minimum capacity, $B = 32$ tokens per device):

| Component | Latency (approx.) | Bottleneck |
|---|---|---|
| Router compute | ~0.45 μs | Compute |
| Pre-dispatch packing | ~18.4 μs | DRAM bandwidth |
| `all_to_all_dispatch` | ~257 μs | Ethernet bandwidth |
| Expert FFN (D=2048) | ~10.75 μs | DRAM bandwidth (weight streaming) `[D UNVERIFIED]` |
| `all_to_all_combine` | ~257 μs | Ethernet bandwidth |
| Combine accumulation | ~18.4 μs | DRAM bandwidth |
| **Total** | **~562 μs** | **Ethernet bandwidth (collectives dominate)** |

**Conclusion (UNVERIFIED):** At decode batch sizes ($B \leq 32$), communication from the two all-to-all collectives dominates MoE layer latency. Expert FFN compute is approximately 50× faster than the collectives under plausible $D$ values. `[D UNVERIFIED — conclusion depends on actual D from Qwen3 Technical Report]`

At prefill batch sizes ($B \geq 1024$):
- $C \approx 1024 \times 8 / 256 = 32$ tokens per expert
- FFN AI enters weight-stationary regime: $\text{AI} = C/2 = 16\,\text{FLOPs/byte}$ `[D UNVERIFIED]`
- FFN time: $5.25 \times 10^{-9} \times 32 \times 2048 \approx 344\,\mu\text{s}$ `[D UNVERIFIED]`
- Collective time: $257\,\mu\text{s} \times 32 = 8{,}224\,\mu\text{s}$ (scales linearly with $C$)

At large batch/prefill sizes, both communication and compute scale with $C$ (and therefore with $B$), so the communication-dominance conclusion persists unless collective bandwidth is dramatically increased. `[D UNVERIFIED]`

---

## Why No Batch-Size Crossover Exists

### Setup

Let $B$ be the number of tokens per device per forward pass. Under uniform routing:

$$C(B) = \left\lceil \frac{Bk}{E} \right\rceil = \left\lceil \frac{8B}{256} \right\rceil = \left\lceil \frac{B}{32} \right\rceil$$

In continuous approximation, $C \approx B/32 = B \cdot k / E$.

**Communication time** (both collectives, all $N-1$ links, per device):

$$T_\text{comm}(B) = 2 \times \frac{(N-1) \times C(B) \times E_d \times H \times 2}{\text{BW}} = 2 \times \frac{7 \times (B/32) \times 32 \times 7168 \times 2}{12.5 \times 10^9}$$

$$= \frac{2 \times 7 \times B \times 7168 \times 2}{12.5 \times 10^9} = \frac{200{,}704 B}{12.5 \times 10^9} \approx 1.606 \times 10^{-5} \cdot B \text{ seconds}$$

**Expert FFN compute time** (all $E_d = 32$ local experts, $C$ tokens each): `[D UNVERIFIED]`

$$T_\text{FFN}(B) = \frac{E_d \times C(B) \times 6HD}{\text{TFLOP}_\text{peak}} = \frac{32 \times (B/32) \times 6 \times 7168 \times D}{262 \times 10^{12}} = \frac{6 \times 7168 \times D \times B}{262 \times 10^{12}}$$

$$= \frac{43{,}008 \times D}{262 \times 10^{12}} \times B \approx 1.64 \times 10^{-10} \times D \times B \text{ seconds} \quad \text{[D UNVERIFIED]}$$

### Confirming D* via Linear Scaling `[D UNVERIFIED]`

Both $T_\text{comm}$ and $T_\text{FFN}$ scale linearly with $B$, so the crossover batch size $B^*$ is actually independent of $B$ — the ratio $T_\text{comm}/T_\text{FFN}$ is constant for all $B$:

$$\frac{T_\text{comm}}{T_\text{FFN}} = \frac{1.606 \times 10^{-5}}{1.64 \times 10^{-10} \times D} = \frac{1.606 \times 10^{5}}{1.64 \times D} \approx \frac{9.79 \times 10^{4}}{D} \quad \text{[D UNVERIFIED]}$$

For this ratio to equal 1 (crossover), we would need:

$$D^* \approx 9.79 \times 10^4 \approx 98{,}000 \quad \text{[D UNVERIFIED]}$$

Since this is far larger than any realistic expert FFN intermediate dimension (typically $D \sim H$ to $4H$, i.e., $D \sim 7168$ to $\sim 28{,}672$ for a 35B-class model), **communication from the all-to-all collectives dominates expert FFN compute at all batch sizes on T3K under these parameters**, assuming the hardware parameters above are accurate. This result is consistent with the $D^* \approx 97{,}813$ derived in the "Crossover Condition" section above. `[D UNVERIFIED — this conclusion must be verified against actual D and T3K measured bandwidth]`

### Symbolic Summary `[D UNVERIFIED]`

$$\frac{T_\text{comm}}{T_\text{FFN}} = \frac{2(N-1) \cdot k \cdot E_d \cdot H \cdot 2 \cdot \text{TFLOP}_\text{peak}}{E \cdot 6HD \cdot \text{BW}} = \frac{2(N-1) \cdot k \cdot E_d \cdot 2 \cdot \text{TFLOP}_\text{peak}}{E \cdot 6D \cdot \text{BW}} \quad \text{[D UNVERIFIED]}$$

Substituting $E = N \cdot E_d$ and $k = N$ (the Qwen3.5-35B parameter coincidence where $k = N = 8$):

$$\frac{T_\text{comm}}{T_\text{FFN}} = \frac{2(N-1) \cdot N \cdot E_d \cdot 2 \cdot \text{TFLOP}_\text{peak}}{N \cdot E_d \cdot 6D \cdot \text{BW}} = \frac{4(N-1) \cdot \text{TFLOP}_\text{peak}}{6D \cdot \text{BW}} = \frac{2(N-1)}{3} \cdot \frac{\text{TFLOP}_\text{peak}}{D \cdot \text{BW}} \quad \text{[D UNVERIFIED]}$$

This compact form reveals:
- The ratio is independent of $H$, $E_d$, $B$, and $C$ (all cancel).
- It depends only on $N$, $D$, and the hardware ratio $\text{TFLOP}_\text{peak} / \text{BW}$.
- Communication dominates when $D < \frac{2(N-1)}{3} \cdot \frac{\text{TFLOP}_\text{peak}}{\text{BW}}$. `[D UNVERIFIED]`

For Wormhole B0 at T3K bandwidth:

$$D_\text{crossover} = \frac{2 \times 7}{3} \times \frac{262 \times 10^{12}}{12.5 \times 10^9} \approx 4.67 \times 20{,}960 \approx 97{,}813 \quad \text{[D UNVERIFIED]}$$

Because expert FFN intermediate dimensions are expected to be $\ll 97{,}813$, the all-to-all collectives are the dominant cost at any practical batch size for this hardware configuration. `[D UNVERIFIED]`

---

## Implications for Optimization

### Communication Is the Bottleneck

The analysis above indicates that on T3K, for the Qwen3.5-35B MoE configuration, reducing all-to-all communication time is the highest-priority optimization for MoE layer latency. The main levers are:

1. **Minimize capacity $C$:** Because both collectives scale linearly with $C$, using the minimum capacity that avoids significant token dropping directly reduces collective time. However, decreasing $C$ increases the probability of token dropping under load imbalance. This trade-off is analyzed in Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md`.

2. **Overlap communication and compute:** Chapter 6, `ch06_fused_dispatch_compute_combine/pipeline_design.md` describes pipeline designs that issue `all_to_all_dispatch` for the next layer while the current layer's expert FFN is computing, hiding the collective latency behind compute.

3. **Reduce $H$ in transit:** Some implementations project token embeddings to a lower-dimensional space before dispatch and project back at the destination, reducing the per-token payload at the cost of additional compute. This is not standard for Qwen3.5-35B but is analyzed in Chapter 3 for alternative routing schemes.

4. **Use higher-bandwidth interconnect:** If T3K Ethernet bandwidth is the bottleneck, hardware upgrades to a higher-bandwidth topology (e.g., NVLink-class interconnects) would shift the balance. This is a hardware rather than software optimization.

---

## References

- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Williams2009] Williams, S., Waterman, A., Patterson, D., "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Communications of the ACM, 2009.
- [Rajbhandari2022] Rajbhandari, S. et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale", ICML, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [T3KDatasheet] Tenstorrent, "T3K System Datasheet", Tenstorrent Product Documentation. (Authoritative source for Ethernet bandwidth and Wormhole B0 FLOP rates.)
- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — cross-device communication volume and Qwen3.5-35B model parameters.
- [Ch2Background] Chapter 2, `collective_communication_background.md` — bandwidth/latency model and $\alpha$-$\beta$ framework.
- [Ch2Dispatch] Chapter 2, `all_to_all_dispatch.md` — dispatch buffer layout and tile-alignment constraints on capacity $C$.
- [Ch2Combine] Chapter 2, `all_to_all_combine.md` — combine buffer layout and accumulation step.
- [Ch6Pipeline] Chapter 6, `ch06_fused_dispatch_compute_combine/pipeline_design.md` — overlapping communication and compute.
- [Ch7Capacity] Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md` — formal analysis of capacity factor and token dropping trade-offs.

---

**Next:** [Chapter 3 — Alternative Routing Schemes](../ch03_alternative_routing_schemes/index.md)
