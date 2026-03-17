# Collective Communication Background

## Purpose

This file establishes the vocabulary and quantitative framework for collective communication operations used throughout this chapter and the rest of the guide. Readers who have worked with MPI (Message Passing Interface) or NCCL (NVIDIA Collective Communications Library) will find most of the taxonomy familiar; the emphasis here is on the specific properties of the all-to-all collective that make it the right primitive for MoE expert dispatch and combine.

---

## Taxonomy of Collective Operations

A **collective operation** is a communication pattern in which all $N$ devices in a communicator participate simultaneously, each executing a defined role with respect to a shared data layout. The operations below are ordered from simplest to most general.

### Broadcast

One source device transmits the same data to all other devices. Every receiver ends up with an identical copy of the source data.

```text
Before:  device 0 holds [A]; devices 1..N-1 hold nothing
After:   all N devices hold [A]
Communication volume: (N-1) * |A|
```

### Scatter

One source device sends a distinct slice of its data to each other device. Each receiver gets one slice; no data is replicated.

```text
Before:  device 0 holds [A0, A1, ..., A_{N-1}]
After:   device i holds A_i
Communication volume: (N-1)/N * |total_data|
```

### Gather

The inverse of scatter. Each device sends its local data to one designated root device, which assembles the full concatenated result.

```text
Before:  device i holds A_i
After:   device 0 holds [A0, A1, ..., A_{N-1}]
Communication volume: (N-1)/N * |total_data|
```

### All-Gather

Every device sends its local data to all other devices. After the operation every device holds the full concatenated data from all devices. Unlike gather, there is no single root — all devices end up with the complete picture.

```text
Before:  device i holds A_i,  |A_i| = M bytes each
After:   all devices hold [A0, A1, ..., A_{N-1}],  N*M bytes each
Communication volume per device sent: (N-1) * M
Total volume in network: N * (N-1) * M
```

All-gather is relevant to MoE as an alternative dispatch strategy: instead of routing token embeddings only to the devices that need them, each device broadcasts its entire local token batch to all devices, and each device then locally selects the tokens routed to its experts. This alternative is analyzed in the comparison section below and in Chapter 3, `ch03_alternative_routing_schemes/expert_sharding.md`.

### Reduce-Scatter

The complement of all-gather. Each device holds a partial result for the full data range; the operation reduces (sums, or applies another associative reduction) across devices and scatters the result so that device $i$ receives the reduced result for its assigned slice.

```text
Before:  device i holds [B0_i, B1_i, ..., B_{N-1}_i]  (partial sums)
After:   device i holds sum over j of Bj_i  (the slice assigned to device i)
Communication volume: same asymptotic as all-gather
```

Reduce-scatter appears in tensor parallelism and gradient communication. It is not used directly for MoE dispatch/combine but is relevant context for understanding the collective landscape.

### All-Reduce

Combines all-gather and reduce: every device ends up with the full reduced result across all devices. All-reduce = reduce-scatter + all-gather (ring decomposition). Used for gradient synchronization in data parallelism.

### All-to-All

The most general personalized collective. Each device sends a **distinct** slice of its data to every other device, and receives a **distinct** slice of data from every other device. Unlike all-gather (where all receivers get the same data) and scatter (where only one device sends), all-to-all is a many-to-many exchange in which the content sent from device $i$ to device $j$ depends on both $i$ and $j$.

```text
Before:  device i holds [S_{i,0}, S_{i,1}, ..., S_{i,N-1}]
         where S_{i,j} is the slice device i will send to device j

After:   device i holds [R_{0,i}, R_{1,i}, ..., R_{N-1,i}]
         where R_{j,i} = S_{j,i} is the slice sent from device j to device i
```

In other words: after all-to-all, device $i$'s receive buffer slot $j$ contains whatever device $j$ placed in its send buffer slot $i$ before the collective. The operation is a complete transposition of the $N \times N$ data matrix.

---

## Formal Definition of All-to-All

Let $M_{i,j}$ denote the message that device $i$ sends to device $j$. The all-to-all operation delivers $M_{i,j}$ to device $j$ for all ordered pairs $(i, j)$ with $i \neq j$. On completion:

- Device $j$ has received $M_{i,j}$ from every $i \neq j$.
- No device receives its own self-addressed message (the $i = j$ case corresponds to data that never leaves the device).

The size of $M_{i,j}$ need not be uniform. In the **uniform all-to-all**, all messages have equal size $s$ bytes, and the total volume exchanged by any single device is:

$$V_\text{device} = (N-1) \times s \quad \text{(sent)} + (N-1) \times s \quad \text{(received)}$$

The total data in flight across the network at any instant (summed across all $N$ devices, counting each link once) is:

$$V_\text{network} = N \times (N-1) \times s$$

For a batch of $B$ tokens on $N$ devices, with each token embedding of size $H \times \text{dtype\_bytes}$ bytes, and assuming uniform routing (each device routes $B \times k / N$ token-expert pairs to each other device), the per-device send volume for dispatch is:

$$V_\text{dispatch, per device} = \frac{(N-1)}{N} \times B \times k \times H \times \text{dtype\_bytes}$$

For Qwen3.5-35B with $B = 32$ tokens per device, $k = 8$, $N = 8$, $H = 7168$, BF16 (2 bytes):

$$V_\text{dispatch, per device} = \frac{7}{8} \times 32 \times 8 \times 7168 \times 2 = 3{,}211{,}264 \approx 3.2 \text{ MB}$$

This matches the per-layer communication estimate from Chapter 1, `routing_problem.md`, confirming that the all-to-all model correctly accounts for the dispatch volume. The combine step has the same per-device volume as dispatch because expert outputs have the same shape $\mathbb{R}^H$ as token embeddings.

---

## Bandwidth and Latency Model for All-to-All on a Mesh

### The $\alpha$-$\beta$ Model

The standard linear cost model for point-to-point message transmission on a network is:

$$T_\text{message}(n) = \alpha + \beta \cdot n$$

where:
- $\alpha$ is the **latency** (per-message overhead, seconds), independent of message size
- $\beta$ is the **per-byte transfer time** ($1/\text{BW}$, seconds per byte)
- $n$ is the message size in bytes

For an all-to-all on $N$ devices where each device sends $N-1$ distinct messages of size $s$ bytes each:

$$T_\text{all-to-all} \geq \max\left(\alpha + \beta \cdot (N-1) \cdot s, \; (N-1) \cdot \alpha + \beta \cdot (N-1) \cdot s\right)$$

The first term is a single large serialized message; the second term reflects the overhead of initiating $N-1$ separate message transfers. In practice, the two terms coexist: latency accumulates from the number of distinct messages initiated ($(N-1)\alpha$), and bandwidth cost accumulates from the total bytes transferred ($\beta \cdot (N-1) \cdot s$).

A simplified lower bound under the assumption that all $N-1$ messages can be pipelined on the link:

$$T_\text{all-to-all} \geq (N-1)\alpha + \beta \cdot (N-1) \cdot s = (N-1)(\alpha + \beta \cdot s)$$

### Mesh Topology and Hop Count

On a 2D torus mesh — the topology used by T3K — devices are arranged in a grid with wraparound connections. For an $N$-device system laid out as a $p \times q$ grid (where $p \times q = N$), the diameter (maximum shortest-path hop count between any two devices) is $\lfloor p/2 \rfloor + \lfloor q/2 \rfloor$.

For T3K with $N = 8$ devices, a natural layout is $2 \times 4$, giving a diameter of $1 + 2 = 3$ hops. Each hop on a T3K Ethernet link contributes additional latency. The effective latency for a message traversing $h$ hops is approximately $h \cdot \alpha_\text{link}$, where $\alpha_\text{link}$ is the per-hop latency.

The all-to-all communication time on a mesh is therefore bounded below by:

$$T_\text{all-to-all, mesh} \geq \underbrace{h_\text{max} \cdot \alpha_\text{link}}_{\text{latency term}} + \underbrace{\beta \cdot (N-1) \cdot s}_{\text{bandwidth term}}$$

where $h_\text{max}$ is the diameter of the mesh. For the T3K $2 \times 4$ layout, $h_\text{max} = 3$, so:

$$T_\text{all-to-all, T3K} \geq 3\alpha_\text{link} + \beta \cdot 7s$$

The exact values of $\alpha_\text{link}$ and $\text{BW}$ for T3K Ethernet links are hardware-specific; readers should consult the official T3K datasheet for authoritative figures. The structural observation — that latency scales with mesh diameter while bandwidth cost scales with $(N-1)s$ — guides the roofline analysis in `dispatch_combine_overhead.md`.

### Ring All-to-All

Many collective libraries implement all-to-all using a **ring algorithm**: devices are arranged in a logical ring; in each of $N-1$ steps, each device simultaneously sends to the next device in the ring and receives from the previous device. After $N-1$ steps, all messages have been delivered.

Ring all-to-all time (uniform message size $s$, $N-1$ steps):

$$T_\text{ring} = (N-1)\alpha + \beta \cdot (N-1) \cdot s$$

The ring algorithm achieves the lower bound for the latency term (one initiation per step, $N-1$ steps total) and saturates the link bandwidth with $(N-1) \times s$ bytes in flight per step per link. It does not reduce the bandwidth term below the information-theoretic lower bound.

---

## All-to-All vs. All-Gather + Local Select

### The All-Gather Alternative

An alternative to the all-to-all dispatch pattern is:

1. **All-gather:** every device broadcasts its local token batch to all other devices. After the all-gather, every device holds all $N \times B$ tokens.
2. **Local select:** each device applies its local experts only to the tokens that were routed to those experts.
3. **Reduce-scatter:** partial outputs (zeros where no local expert was applied) are summed across devices and scattered back.

This **all-gather + local select + reduce-scatter** pattern avoids routing-dependent irregular transfers. The all-gather and reduce-scatter can be implemented with well-optimized ring algorithms that achieve near-peak link utilization.

### Communication Volume Comparison

For the all-gather approach, every device sends its full token batch $[B, H]$ to all other devices:

$$V_\text{all-gather, per device} = (N-1) \times B \times H \times \text{dtype\_bytes}$$

For the all-to-all approach, each device sends only the tokens destined for remote experts (under uniform routing, a fraction $(N-1)/N$ of $B \times k$ token slots, each of size $H \times \text{dtype\_bytes}$):

$$V_\text{all-to-all, per device} = \frac{N-1}{N} \times B \times k \times H \times \text{dtype\_bytes}$$

The ratio is:

$$\frac{V_\text{all-to-all}}{V_\text{all-gather}} = \frac{(N-1)/N \times k}{N-1} = \frac{k}{N}$$

For Qwen3.5-35B with $k = N = 8$:

$$\frac{V_\text{all-to-all}}{V_\text{all-gather}} = \frac{8}{8} = 1$$

In this specific model configuration, the two approaches exchange the same volume of token embedding data. The all-to-all has no volume advantage over all-gather for this parameter choice. The key difference is what is being communicated:

- **All-to-all** sends only the embeddings that will actually be consumed by remote experts; no redundant copies of locally-consumed tokens are transmitted.
- **All-gather** transmits every token to every device; each device receives $N$ copies of each token (including $N-1$ copies it will immediately discard for tokens not routed to its experts).

Although the total bytes are equal when $k = N$, the **memory footprint** differs substantially: after all-gather, each device must buffer $N \times B \times H$ bytes (the full collective output), compared to $C \times E_d \times H$ bytes after all-to-all (only the tokens actually dispatched to local experts). For large batches, this buffer size difference is significant.

### Sparsity Exploitation

When routing is sparse — that is, when many experts receive zero tokens in a given batch — the all-to-all can transmit zero bytes for those empty expert slots (if the implementation supports variable-length messages). The all-gather approach has no analogous savings: it always transmits the full batch regardless of routing sparsity.

For Qwen3.5-35B under balanced routing, sparsity is low (each of $E = 256$ experts receives $B \times k / E = B/32$ tokens on average), so the sparsity benefit of all-to-all is modest. Under skewed routing, however, some experts may receive very few tokens and others many — and the all-to-all can be adapted to skip empty experts, whereas all-gather cannot.

### Summary Comparison

| Property | All-to-All Dispatch | All-Gather + Local Select |
|---|---|---|
| Communication volume ($k = N = 8$) | $(N-1)/N \times Bk \times H$ | $(N-1) \times B \times H$ |
| Volume ratio (Qwen3.5-35B) | 1.0 (equal) | 1.0 (equal) |
| Per-device receive buffer | $C \times E_d \times H$ (expert-capacity-bounded) | $N \times B \times H$ (full batch × N) |
| Routing-sparsity exploitation | Yes — empty expert slots transmit nothing | No — full batch always transmitted |
| Implementation complexity | Higher — requires routing-index-aware packing | Lower — simple ring all-gather |
| Applicability | Any routing; efficient for sparse experts | Best when routing is dense and batch is small |

The all-to-all approach is preferred for Qwen3.5-35B primarily for buffer efficiency and sparsity exploitation potential, even though the raw communication volume is identical at $k = N$. Chapter 3, `ch03_alternative_routing_schemes/expert_sharding.md` provides a more detailed quantitative comparison including the cost of the reduce-scatter in the all-gather approach and the batch-size regimes where each approach is dominant.

---

## References

- [Rabenseifner2004] Rabenseifner, R., "Optimization of Collective Reduction Operations", International Conference on Computational Science (ICCS), 2004.
- [Thakur2005] Thakur, R., Rabenseifner, R., Gropp, W., "Optimization of Collective Communication Operations in MPICH", International Journal of High Performance Computing Applications, 2005.
- [Gropp1999] Gropp, W., Lusk, E., Skjellum, A., "Using MPI: Portable Parallel Programming with the Message-Passing Interface", MIT Press, 1999.
- [Kumar1994] Kumar, V. et al., "Introduction to Parallel Computing: Design and Analysis of Algorithms", Benjamin/Cummings, 1994.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Lepikhin2021] Lepikhin, D. et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR, 2021.
- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — cross-device communication volume derivation.
- [Ch2Index] Chapter 2, `index.md` — chapter overview and notation.
- [Ch2Dispatch] Chapter 2, `all_to_all_dispatch.md` — all-to-all dispatch implementation.
- [Ch3ExpertSharding] Chapter 3, `ch03_alternative_routing_schemes/expert_sharding.md` — quantitative comparison of all-gather-based expert sharding.
- [Ch2Overhead] Chapter 2, `dispatch_combine_overhead.md` — roofline analysis and batch-size threshold for T3K.
