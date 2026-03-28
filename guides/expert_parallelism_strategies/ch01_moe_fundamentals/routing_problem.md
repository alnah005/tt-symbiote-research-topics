# The Routing Problem

## From Mathematical Abstraction to Physical Placement

The MoE routing equations described in `moe_architecture.md` are clean and self-contained on paper. In practice, on a system with $N = 8$ devices such as the T3K (Tenstorrent 8-device mesh), no single device can hold all $E = 256$ expert weight matrices in its local memory at full precision. Expert weights must be distributed across devices. This distribution is what transforms routing from a purely algorithmic concern into a systems engineering problem.

When a token arrives at device $d$ during inference, the router on that device computes $k = 8$ expert indices from $\{0, \ldots, 255\}$. With 256 experts spread across 8 devices (32 experts per device under uniform assignment), under uniform routing each of the $k$ selected expert slots has a $\frac{N-1}{N} = \frac{7}{8} = 87.5\%$ probability of being remote — that is, residing on a device other than $d$. Therefore, on average $k(N-1)/N = 8 \times 7/8 = 7$ of the 8 expert slots per token require inter-device sends. The token's embedding therefore must travel over inter-device links to reach those 7 remote experts on average, and the expert outputs must travel back. Neither trip is free.

This section explains why this overhead exists, how it compounds with load imbalance, and how the dispatch/combine communication pattern addresses it.

---

## Why Routing Creates Communication Overhead

### Expert Placement and Cross-Device Traffic

Assume a uniform assignment of experts to devices: device $d$ owns experts $\{32d, 32d+1, \ldots, 32d+31\}$ for $d = 0, \ldots, 7$. For a batch of $B$ tokens, each token selects $k = 8$ experts, producing $B \times k = 8B$ total (token, expert) pairs across the batch.

Each device originates $B/N$ tokens and must forward those tokens to $(B/N) \times k$ expert slots per device. Under uniform routing (where each token selects $k$ experts independently and uniformly at random from all $E$ experts), on average $k/N = 8/8 = 1$ expert per token is expected to reside on the originating device. The remaining expected $k(N-1)/N = 8 \times 7/8 = 7$ require inter-device communication.

The actual count of remote experts per token is random and depends on the routing distribution. The formula below states the expected value under uniform routing (condition: experts are selected independently and uniformly at random across devices):

$$\mathbb{E}[\text{cross-device volume per token}] \approx \mathbb{E}[k_\text{remote}] \times H \times \text{dtype bytes}$$

where $\mathbb{E}[k_\text{remote}] = k(N-1)/N = 7$ under uniform routing (equal to $k-1$ here only because $k = N = 8$). Scaling to a batch of $B$ tokens, the expected total cross-device volume is $B \times k(N-1)/N \times H \times \text{dtype bytes}$.

For $B = 32$ tokens, $k = 8$, $N = 8$, $H = 7168$ (Qwen3.5-35B hidden dimension), and BF16 (2 bytes), this expected total works out to:

$$32 \times k(N-1)/N \times 7168 \times 2 = 32 \times 7 \times 7168 \times 2 = 3{,}211{,}264 \text{ bytes} \approx 3.1 \text{ MB per MoE layer}$$

With 80 MoE layers in Qwen3.5-35B (the remaining 14 layers use dense FFN and have no dispatch traffic), a single full-model forward pass requires roughly 245 MiB of token embedding traffic over inter-device links for this batch configuration — not counting the corresponding combine traffic for expert outputs. The combine step returns expert outputs of the same shape $\mathbb{R}^H$ per routed token (since expert outputs are vectors in the same space as token embeddings, both in $\mathbb{R}^H$, the combine step has the same traffic size as the dispatch step), producing an equal return-traffic volume (~245 MiB); the total round-trip communication per forward pass is therefore approximately 490 MiB for this batch configuration.

This is the fundamental cost of expert parallelism: compute is sparse and cheap per token, but communication is neither.

### The Two Communication Legs

The cross-device traffic in a MoE layer occurs in two distinct phases:

1. **Dispatch.** Before expert computation can begin, token embeddings must be sent from their originating device to the device that owns the target expert. For each of the $k$ selected experts that lives on a remote device, the originating device must transmit the token's embedding vector.

2. **Combine.** After expert computation, the expert output (also a vector in $\mathbb{R}^H$) must be returned to the originating device. The originating device then applies the routing weight $\hat{w}_i$ (where $\hat{w}_i = p_i / \sum_{j \in I} p_j$ is the renormalized router weight for expert $i$, with $I$ the set of selected top-k experts) to that expert's output and accumulates the results across all $k$ experts to produce the MoE layer output $y$.

Together, dispatch and combine form a matched pair of collective communication operations. The collective used for both is the all-to-all, described in detail in Chapter 2, `all_to_all_dispatch.md` and `all_to_all_combine.md`.

---

## Token Distribution Imbalance

### Load Skew

In an ideal world, routing would distribute tokens uniformly across all $E = 256$ experts, and by extension uniformly across all $N = 8$ devices. In practice, neural network routers do not produce perfectly uniform distributions. During training, load-balancing auxiliary losses are used to encourage uniformity, but they cannot fully eliminate skew. During inference, the auxiliary losses are absent entirely (see Chapter 7, `auxiliary_loss_free_inference.md`), and the router may exhibit more concentrated routing behavior on specific types of inputs.

**Load skew** occurs when the router consistently selects a small subset of "popular" experts for a disproportionate fraction of tokens. If expert $e^*$ is selected by a fraction $f_{e^*}$ of tokens in the batch and the uniform expectation is $\frac{k}{E} = \frac{8}{256} = 0.03125$, then the load imbalance factor is:

$$L = \frac{f_{e^*}}{k / E}$$

Here $f_{e^*}$ is the empirical selection frequency of the most popular expert across the batch, distinct from the per-token routing probability $p_e$ defined in `moe_architecture.md`. $p_e$ is a per-token scalar in $[0, 1]$ produced by the router softmax for a single token; $f_{e^*}$ is a batch-level statistic counting what fraction of all tokens in the batch were routed to expert $e^*$.

Note: $f_{e^*} = \text{count}_{e^*} / B$ (fraction of batch tokens), so $L = f_{e^*} / (k/E) = \text{count}_{e^*} / (B \cdot k / E)$, which matches the glossary definition: ratio of the most-loaded expert's token count to the average token count.

An imbalance factor of $L = 2$ means the most popular expert receives twice the average token load. For $L = 4$, it receives four times the average load. The device hosting the most popular experts becomes the bottleneck: all other devices must wait for it to finish before the combine step can proceed.

### Impact on End-to-End Latency

In a synchronous execution model, the MoE layer completes only when the last expert has finished computing. If device $d^*$ hosts all the popular experts and receives $L$ times the average token count, the expert FFN compute time on $d^*$ is $L$ times longer than on other devices. The total MoE layer latency is therefore dominated by $d^*$, not by the average.

This is not a hypothetical concern. Empirical measurements on production MoE models have shown load imbalance factors of 2–5$\times$ for specific input distributions, particularly for inputs in narrow domains (code, mathematics, a specific language) that consistently trigger domain-specialized experts.

### Visualizing Load Skew

The following pseudocode illustrates how to measure per-expert token counts from a batch of routing decisions:

```python
def measure_load_distribution(
    routing_indices: list,   # list of B lists, each of length k
    E: int,                  # total number of experts
    k: int,                  # top-k
) -> list:
    """
    Returns a list of length E with token counts per expert.

    routing_indices: shape [B, k], values in {0, ..., E-1}
    """
    counts = [0] * E
    for token_experts in routing_indices:
        for expert_idx in token_experts:
            counts[expert_idx] += 1
    return counts

def load_imbalance_factor(counts: list, B: int, k: int, E: int) -> float:
    """
    Returns the ratio of max expert load to the expected average.
    """
    expected_avg = B * k / E
    return max(counts) / expected_avg
```

For a batch of $B = 512$ tokens with $k = 8$ and $E = 256$, the expected average is 16 tokens per expert. If the most popular expert receives 80 tokens, the load imbalance factor is $80 / 16 = 5.0$.

---

## Expert Capacity

When routing is skewed, an expert may receive far more tokens than it was designed to handle in a single forward pass. Expert capacity is the concept used to bound this: each expert is allocated a fixed-size buffer that can hold at most $C$ tokens per forward pass.

The capacity $C$ is parameterized by a **capacity factor** $CF \geq 1.0$, which scales the expected average token load per expert; a higher $CF$ provides more buffer for skewed routing at the cost of wasted compute on empty slots and wasted communication bandwidth (padding in dispatch/combine buffers). The precise formula — along with the dispatch buffer sizing implications and the interaction between $C$ and the all-to-all buffer layout — is covered in Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md`.

### What Happens at Overflow

When more tokens are routed to an expert than its capacity $C$ allows, two strategies exist:

- **Token dropping.** Tokens that exceed the capacity are discarded for that expert. The affected tokens receive a zero contribution from that expert slot, which reduces the quality of the model's output for those tokens.
- **Overflow routing.** Overflowed tokens are redirected to a fallback expert (typically a designated overflow expert or the next highest-scoring expert). This avoids zero contributions but complicates the dispatch logic and may create secondary load spikes on the overflow expert.

The formal mechanics of capacity factor, overflow handling, and the interaction between $C$ and dispatch buffer sizing are covered in detail in Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md`. At this point, readers need only understand that expert capacity exists as a bounding mechanism and that exceeding it has consequences for output quality or routing complexity.

The capacity concept also has a direct effect on the memory layout of the dispatch operation: the all-to-all buffer must be sized to $C$ slots per expert, not to the actual number of tokens routed to that expert. This padding is what allows the all-to-all to use a fixed-size, statically allocated buffer.

---

## The Dispatch/Combine Communication Pattern

The solution to the cross-device routing problem is a pair of collective communication operations that shuttle token embeddings and expert outputs between devices in a coordinated way. This pattern is known as **dispatch** and **combine**, and it is the central mechanism studied throughout this guide.

### Dispatch: Moving Token Embeddings to Expert Devices

Before expert computation begins, each device knows (from the router's output) which of its tokens need to be processed by which remote experts. The dispatch operation moves each token embedding from its originating device to the device that owns the target expert. After dispatch, each device holds all the token embeddings destined for its local experts, packed contiguously in a buffer of shape $[C \times E_d, H]$, where $E_d = E/N = 256/8 = 32$ experts per device under uniform assignment. Here $C$ is the per-expert token capacity — the maximum number of tokens each expert can receive in a single forward pass; $C$ will be formally defined in Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md`.

At a conceptual level, dispatch is a **personalized exchange**: each device sends a different subset of its tokens to each other device, and receives a different subset of tokens from each other device. This is the defining property of the all-to-all collective, as explained in Chapter 2, `ch02_all_to_all_primitives/collective_communication_background.md`.

### Combine: Returning Expert Outputs to Originating Devices

After each device runs its local expert FFNs on the dispatched tokens, the expert outputs must travel back to the originating devices. The combine operation is the inverse of dispatch: each device sends expert outputs back to the device that originated each token. The originating device then multiplies each expert output by its routing weight $\hat{w}_i$ and accumulates across all $k$ experts to form the final MoE layer output.

### Conceptual Flow Diagram

The following pseudocode traces the flow of a single token $t$ through one MoE layer on an 8-device system:

```text
Device d_orig (where token t originates):
  1. Compute router logits g = x_t @ W_r            // shape [E=256]
  2. Softmax -> p                                    // shape [256]
  3. Top-k -> indices I = {e_0, ..., e_7}, weights w_hat
  4. For each e_i in I:
       if e_i is on device d_orig: keep locally
       else: mark x_t for dispatch to device owning e_i

DISPATCH collective:
  All devices exchange token embeddings.
  After dispatch, device d holds all tokens routed to its experts.

Device d (any device, including d_orig):
  5. For each local expert e in {32d, ..., 32d+31}:
       o_e = FFN_e(dispatched tokens for e)

COMBINE collective:
  All devices exchange expert outputs.
  After combine, device d_orig holds o_{e_0}, ..., o_{e_7} for token t.

Device d_orig:
  6. y_t = sum over i in {0..7} of w_hat_i * o_{e_i}  // weighted combination
  7. output_t = x_t + y_t                               // residual connection
```

This sequence defines the structure of every MoE layer in an expert-parallel setting. Steps 1–3 are the router forward pass, step 4 is the pre-dispatch token assignment, the two collective steps are the all-to-all operations, and steps 5–7 are local compute. Step 5 runs on each expert device; steps 6–7 run on the originating device after the combine collective returns results. The implementation details of each step are the subject of Chapters 2 through 6.

### Why All-to-All?

The dispatch and combine collectives match the communication pattern of an all-to-all operation because the destinations are token-dependent and fully general: any token on any device may be routed to any expert on any device. Other collectives such as broadcast, all-gather, or scatter do not match this pattern, because they either send the same data to all recipients or send data to a single recipient. The all-to-all collective is the only standard collective that supports the many-to-many, personalized exchange required by MoE routing.

Alternative approaches — such as gathering all token embeddings to all devices before routing, then running local experts — exist and have different trade-offs. These are analyzed in Chapter 3, `ch03_alternative_routing_schemes/`.

---

## References

- [Shazeer2017] Shazeer, N. et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR, 2017.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Zhou2022] Zhou, Y. et al., "Mixture-of-Experts with Expert Choice Routing", NeurIPS, 2022.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Ch1MoEArch] `moe_architecture.md` — MoE layer definition and routing equations.
- [Ch1Config] `qwen35b_config.md` — Qwen3.5-35B architectural constants.
- [Ch2AllToAll] Chapter 2, `collective_communication_background.md` — formal definition of all-to-all.
- [Ch2Dispatch] Chapter 2, `all_to_all_dispatch.md` — dispatch implementation.
- [Ch2Combine] Chapter 2, `all_to_all_combine.md` — combine implementation.
- [Ch3Alt] Chapter 3, `ch03_alternative_routing_schemes/scheme_comparison_matrix.md` — alternative routing schemes including all-gather-based approaches.
- [Ch7Capacity] Chapter 7, `capacity_factor_mechanics.md` — formal capacity factor definition and overflow handling.
- [Ch7AuxFree] Chapter 7, `auxiliary_loss_free_inference.md` — inference-time load skew behavior.

---

**Next:** [qwen35b_config.md](./qwen35b_config.md)
