# Expert Sharding via All-Gather

## Purpose

This file defines static expert sharding, derives the communication volume of the all-gather-before-routing pattern, and compares it carefully with the all-to-all baseline from Chapter 2. It establishes when expert sharding reduces total latency and identifies the trade-off between communication simplicity and compute waste.

**Prerequisite:** Chapter 1, `ch01_moe_fundamentals/routing_problem.md` (dispatch/combine concept); Chapter 2, `ch02_all_to_all_primitives/dispatch_combine_overhead.md` (all-to-all volume formula and baseline latency decomposition); Chapter 3, `index.md` (notation).

---

## Definition: Static Expert Sharding

Static expert sharding assigns each expert permanently to exactly one device at model-load time. For Qwen3.5-35B with $E = 256$ experts and $N = 8$ devices, the uniform assignment places exactly $E_d = E / N = 256 / 8 = 32$ experts on each device. Expert $e$ is assigned to device $\lfloor e / E_d \rfloor$; this assignment does not change during inference.

The key property is that the **expert weights never move** — each device owns its 32 experts, and every forward pass uses the same device-to-expert mapping.

---

## The All-Gather-Before-Routing Pattern

Under static expert sharding, the natural execution pattern for an MoE layer is:

```text
1. All-gather:  every device sends its local token batch to all other devices
                Result: each device holds the FULL token batch from all N devices
2. Local route: each device runs its router to identify which of the full batch
                tokens are assigned to its 32 local experts
3. Local FFN:   each device computes expert FFN for all tokens routed to its local experts
4. Reduce-scatter: each device sends computed expert outputs back to the originating
                   device; each device accumulates weighted expert outputs for its tokens
```

Steps 1 and 4 are collective communication operations; steps 2 and 3 are local. This pattern replaces the two all-to-all operations of the baseline with an all-gather and a reduce-scatter.

Note that all-gather + reduce-scatter together are equivalent to all-reduce in terms of buffer requirements, but not in terms of data layout: the all-gather produces a full replicated tensor on each device (which each device then partially consumes in step 3), while the reduce-scatter returns a partitioned result.

---

## Communication Volume: All-Gather

### Derivation

In step 1, each device starts with $B$ local tokens (each a vector of length $H$ in BF16, occupying $H \times 2$ bytes). The all-gather operation causes every device to receive the tokens from every other device, so after the all-gather each device holds $N \times B$ tokens total.

The volume sent by one device during the all-gather is $B \times H \times 2$ bytes (its local batch, sent to all other $N - 1$ devices). The volume received is $(N-1) \times B \times H \times 2$ bytes.

Counted as the total bytes flowing over each device's outbound links:

$$V_\text{gather} = (N - 1) \times B \times H \times 2 \text{ bytes}$$

For Qwen3.5-35B on T3K ($N = 8$, $H = 7{,}168$, BF16):

$$V_\text{gather} = 7 \times B \times 7{,}168 \times 2 = 100{,}352 \times B \text{ bytes}$$

**Arithmetic check:** $7 \times 7{,}168 = 50{,}176$; $50{,}176 \times 2 = 100{,}352$ bytes per token. For $B = 32$: $100{,}352 \times 32 = 3{,}211{,}264$ bytes $\approx 3.06$ MiB.

Note: The formula $(N-1)/N \times N \times B \times H \times 2$ simplifies to $(N-1) \times B \times H \times 2$; the $(N-1)/N$ factor arises in the all-to-all formulation because only $k/N$ fraction of traffic goes to each remote device. For all-gather, every device receives every other device's full local batch — there is no routing-dependent sparsity.

---

For the all-to-all baseline volume formula and Qwen3.5-35B numerical instantiation, see `index.md` Section [notation/baseline section].

---

## Comparing All-Gather and All-to-All Volumes

### The Key Coincidence: $k = N = 8$

For Qwen3.5-35B, $k = 8$ and $N = 8$, so both volumes evaluate to the same formula:

$$V_\text{gather} = (N-1) \times B \times H \times 2 = V_{a2a}$$

This equality is not a coincidence of specific batch sizes — it holds for all $B$. The algebraic reason is:

- All-gather sends one full copy of the local batch to each of the $N-1$ other devices: volume proportional to $1 \times (N-1)$.
- All-to-all sends $k/N$ of each token's expert slots to each remote device, across $N-1$ remote devices, and $k$ expert slots total per token: volume proportional to $(k/N) \times (N-1) = k(N-1)/N$. When $k = N$, this is $(N-1)$, matching the all-gather.

**Statement:** For Qwen3.5-35B on T3K, the all-gather and all-to-all dispatch operations transfer identical byte volumes. Communication time is the same for both schemes assuming equal link utilization patterns.

### When Do Volumes Differ?

For models where $k < N$, the all-to-all is cheaper:

$$\frac{V_{a2a}}{V_\text{gather}} = \frac{(N-1) \times k/N}{(N-1)} = \frac{k}{N}$$

For $k < N$, this ratio is less than 1, and all-to-all transfers fewer bytes. For example, a hypothetical model with $k = 2$, $N = 8$ would have $V_{a2a}/V_\text{gather} = 2/8 = 0.25$ — all-to-all would use only 25% of the all-gather volume.

For Qwen3.5-35B ($k = N = 8$), the ratio is exactly 1. Communication volume is not a differentiating factor between the two schemes for this model at this device count.

---

## The Decisive Difference: Compute Waste

Since communication volumes are equal for Qwen3.5-35B, the decision between all-gather and all-to-all must be made on other grounds. The critical difference is **compute efficiency**.

### All-to-All: No Compute Waste

With all-to-all dispatch, each device receives exactly the tokens that are routed to its 32 local experts. Every token that arrives at device $d$ is processed by one of device $d$'s experts. No token computation is wasted.

Per device, under uniform routing, the number of tokens processed is:

$$\text{Tokens per device (all-to-all)} = B \times \frac{k}{N} = B \times \frac{8}{8} = B$$

Each of the $E_d = 32$ local experts processes $B / E_d = B / 32$ tokens on average (originating-device perspective: only the local $B$ tokens are dispatched to this device's experts).

### All-Gather: Every Device Runs the Router on the Full Batch

With all-gather, after step 1 each device holds $N \times B$ tokens. In step 2, the device runs its local router on all $N \times B$ tokens to identify which tokens are destined for its 32 experts. Only a fraction of those tokens will actually be routed to local experts.

Under all-gather, each device receives all NB tokens in the batch. Of these, only tokens with at least one selected expert on this device will produce a local expert computation; under uniform routing with k=8 from E=256 with E_d=32 local experts, approximately 66% of tokens have at least one expert on any given device. The all-gather communication transfers all $NB$ tokens to each device, but ~34% of those tokens have no local expert assignment on a given device — they are transferred but produce no local expert FFN computation. This is the communication overhead of the all-gather scheme. The router must still process all $NB$ tokens to identify which tokens have local experts.

More concretely:
- Router compute with all-gather: $2 \times (NB) \times H \times E = 2 \times 8B \times H \times 256$ FLOPs (on the full $NB$-token batch).
- Router compute with all-to-all: $2 \times B \times H \times E = 2 \times B \times H \times 256$ FLOPs (on the local $B$-token batch).

The router itself is $N = 8\times$ more expensive under all-gather for the same local token count.

### Expert FFN Compute Under All-Gather

After local routing (step 2), each device processes only the tokens assigned to its local experts. From the processing-device perspective, each device has received all $NB$ tokens and each of its $E_d = 32$ local experts processes $NB / E_d = 8B / 32 = B/4$ tokens on average — not $B/32$ as in the all-to-all originating-device view. The difference in perspective: under all-to-all, only $B$ tokens (the local batch) are dispatched to this device's experts; under all-gather, all $NB$ tokens are available and ~66% of them have at least one expert on this device, spread across $E_d$ local experts. No expert FFN compute is wasted (every processed token legitimately belongs to a local expert), but the per-expert load is $N\times$ higher due to receiving the full global batch. The router overhead (step 2) also scales with the full $NB$ token batch, not just the local $B$-token batch.

**Summary:** All-gather and all-to-all carry the same communication volume for Qwen3.5-35B. All-gather is more expensive in router compute by a factor of $N$ (applied to the full gathered batch), but expert FFN compute is equivalent. For small batch sizes where router compute is a small fraction of total time, this difference is negligible; it becomes relevant only when the router projection $[NB, H] \times [H, E]$ is a significant portion of layer latency.

---

## Memory Implications

Expert sharding via all-gather simplifies buffer management in two ways:

1. **No routing metadata in the send buffer.** The all-gather sends complete token batches, not selectively packed token subsets. There is no need to maintain per-expert send-count arrays or per-token expert-index metadata in the send buffer. The send buffer is simply the local token batch tensor of shape $[B, H]$.

2. **No capacity reservation.** All-to-all dispatch requires pre-allocating capacity $C$ per expert (Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md`), which creates memory overhead and requires tuning the capacity factor $CF$. All-gather requires no capacity reservation: the full token batch is gathered regardless of routing outcomes.

The trade-off is a larger intermediate tensor: after the all-gather, each device holds $[N \times B, H]$ tokens rather than $[B, H]$, increasing L1/DRAM pressure by a factor of $N = 8$ during steps 2 and 3.

---

## When Expert Sharding (All-Gather) Wins

Given the above analysis, expert sharding via all-gather is preferred over all-to-all in the following circumstances:

1. **$k = N$ (volumes are equal) AND implementation simplicity is the priority.** Because volumes are identical for Qwen3.5-35B, removing the need for dispatch metadata and capacity management can justify all-gather when engineering bandwidth is limited.

2. **Very small batch sizes ($B \times S \ll E$) with high load imbalance.** At extremely small batches, load skew causes some experts in the all-to-all scheme to overflow capacity. All-gather avoids this by sending the full batch to all devices; no token is ever dropped.

3. **Batch sizes where $V_{a2a}$ overhead of capacity-padded buffers exceeds $V_\text{gather}$.** Tile-alignment constraints (Chapter 2, `all_to_all_dispatch.md`) can force the effective dispatch volume significantly above the theoretical $V_{a2a}$ when $C$ rounds up to a tile boundary. All-gather has no such alignment overhead on the send side.

Expert sharding is **not** preferred when $k < N$ (the all-to-all is cheaper in communication volume) or when the $N \times$ router compute overhead is unacceptable.

---

## References

- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Lepikhin2021] Lepikhin, D. et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", ICLR, 2021.
- [Rajbhandari2022] Rajbhandari, S. et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale", ICML, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Ch1Config] Chapter 1, `ch01_moe_fundamentals/qwen35b_config.md` — Qwen3.5-35B architectural constants.
- [Ch2Background] Chapter 2, `ch02_all_to_all_primitives/collective_communication_background.md` — all-gather/all-to-all collective definitions.
- [Ch2Overhead] Chapter 2, `ch02_all_to_all_primitives/dispatch_combine_overhead.md` — all-to-all volume formula and latency decomposition.
- [Ch2Dispatch] Chapter 2, `ch02_all_to_all_primitives/all_to_all_dispatch.md` — tile-alignment constraints on dispatch buffers.
- [Ch3Index] Chapter 3, `ch03_alternative_routing_schemes/index.md` — chapter overview and notation table.
- [Ch4Uniform] Chapter 4, `ch04_expert_device_assignment/uniform_partitioning.md` — 32-experts-per-device assignment.
- [Ch7Capacity] Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md` — capacity factor and token-dropping mechanics.

---

**Next:** [pipeline_expert_parallelism.md](./pipeline_expert_parallelism.md)
