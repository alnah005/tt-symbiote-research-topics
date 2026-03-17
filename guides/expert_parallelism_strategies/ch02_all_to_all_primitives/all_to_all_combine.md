# All-to-All Combine

## Purpose

This file specifies the semantics of `all_to_all_combine` — the TTNN (Tenstorrent tensor library) operation that routes expert output embeddings back from the devices that computed them to the originating devices, and accumulates them with their routing weights to produce the final MoE (Mixture-of-Experts) layer output.

`all_to_all_combine` is the exact inverse of `all_to_all_dispatch`. Where dispatch sends token embeddings *toward* the expert devices, combine sends expert outputs *back* to the originating devices. The two operations share buffer layout conventions and slot metadata structures, so this file focuses on the aspects unique to combine: weighted accumulation, the ordering constraint, and numerical considerations.

**Prerequisite:** `all_to_all_dispatch.md` (this chapter) for buffer layout conventions and slot metadata; Chapter 1, `routing_problem.md` for the weighted accumulation semantics at the end of the MoE forward pass.

---

## Semantics of `all_to_all_combine`

### Conceptual Contract

After all devices have executed their local expert FFNs, device $d$ holds an expert output buffer of shape $[C \times E_d, H]$ — the same shape as its dispatch receive buffer. Each row of this buffer is one expert output vector in $\mathbb{R}^H$ corresponding to a specific (originating device, token, expert) triple recorded in the slot metadata.

`all_to_all_combine` performs the following:

1. **Read expert output buffers.** On each device $d$, for each row of the expert output buffer, read the originating device from the slot metadata. This determines which device should receive this expert output.

2. **Execute all-to-all.** Transmit expert outputs to their originating devices. The send/receive pattern is the transpose of the dispatch operation: if device $d$ received a token from device $d'$ during dispatch (because device $d$ owns the target expert), then during combine device $d$ sends the corresponding expert output back to device $d'$.

3. **Write combine buffers.** Each originating device assembles all received expert outputs into a combine buffer, keyed by original token index and expert slot index.

4. **Weighted accumulation.** For each token $t$ on the originating device, multiply each received expert output by its routing weight $\hat{w}_{t,j}$ (the renormalized weight for expert $j$'s contribution to token $t$) and sum across all $k$ contributions to produce the MoE layer output:

$$y_t = \sum_{j=0}^{k-1} \hat{w}_{t,j} \cdot o_{e_{t,j}}$$

where $o_{e_{t,j}} \in \mathbb{R}^H$ is the output of expert $e_{t,j}$ for token $t$.

5. **Residual add.** The originating device then computes the MoE layer final output as $x_t + y_t$ (residual connection), where $x_t$ is the token embedding that was the original input to the MoE layer. This step is local and not part of the combine collective itself.

### TTNN Signature (Illustrative)

```python
moe_output = ttnn.all_to_all_combine(
    expert_outputs,    # Tensor [C * E_d, H]: local expert outputs on this device
    slot_metadata,     # Tensor [C * E_d, 3]: (orig_device, token_id, routing_weight)
    num_experts,       # int: E = 256
    num_devices,       # int: N = 8
    batch_size,        # int: B, tokens per originating device
    top_k,             # int: k = 8
    dtype,             # ttnn.bfloat16 or equivalent
)
# Returns: Tensor [B, H] — MoE layer output (before residual add) on originating device
```

As with `all_to_all_dispatch`, the exact API is subject to the TTNN version in use; the illustrative signature captures the logical inputs and outputs.

---

## Weighted Accumulation

### Where the Weights Live

The routing weights $\hat{w}_{t,j}$ are computed by the router on the originating device before the dispatch step. They are not transmitted with the token embeddings during dispatch; they remain on the originating device throughout. This is intentional: weights are small scalars (one $\text{float16}$ or $\text{bfloat16}$ value per token-expert pair, a total of $B \times k \times 2$ bytes per device) and never leave the device.

During the combine step, the originating device retrieves the routing weights from its local weight buffer and applies them to the received expert outputs. The slot metadata returned by the combine collective maps each received expert output to its `(token_id, expert_slot_j)` pair, allowing the originating device to look up the correct $\hat{w}_{t,j}$.

A discussion of deferring weight normalization to the combine step (rather than normalizing before dispatch) is in Chapter 5, `ch05_routing_weight_optimization/weight_normalization.md`.

### Accumulation Pseudocode

```python
def weighted_accumulate(
    received_expert_outputs: list,   # list of (token_id, expert_slot, output_vec)
    routing_weights: list,           # shape [B, k]: pre-computed renormalized weights
    B: int,                          # batch size
    H: int,                          # hidden dimension
) -> list:
    """
    Returns y: list of B output vectors in R^H.
    routing_weights[t][j] is the renormalized weight for token t, expert slot j.
    """
    y = [[0.0] * H for _ in range(B)]
    for (token_id, expert_slot_j, output_vec) in received_expert_outputs:
        w = routing_weights[token_id][expert_slot_j]
        for h in range(H):
            y[token_id][h] += w * output_vec[h]
    return y
```

In a production implementation this inner loop is a fused multiply-accumulate over a $[k, H]$ output tile per token, vectorized across the $H$ dimension using the Tenstorrent tile compute primitives.

---

## The Ordering Constraint

### Why Order Matters

The combine operation must correctly associate each received expert output with the token it was computed for and the routing weight that should multiply it. This association is what the slot metadata — transmitted with or ahead of the dispatch — provides.

The ordering constraint is: **the receive buffer layout of the combine collective must be consistent with the send buffer layout of the dispatch collective, using the same slot metadata as the authority for the mapping.**

Concretely, when device $d$ sends expert outputs back to originating device $d'$, the rows of the combine send buffer must correspond to the same slots as the rows of the dispatch send buffer that $d'$ sent to $d$. If the dispatch buffer for device $d$ from device $d'$ used the layout:

```text
Row 0:  embedding for token_id=1, expert=2, slot=0
Row 1:  embedding for token_id=4, expert=2, slot=1
...
```

then the combine send buffer from device $d$ to device $d'$ must use the corresponding layout:

```text
Row 0:  expert output for token_id=1, expert=2, slot=0
Row 1:  expert output for token_id=4, expert=2, slot=1
...
```

The originating device $d'$ uses these row indices, combined with the slot metadata, to correctly route each received expert output to the right accumulation target.

### Implication: Metadata Must Survive Across Both Collectives

The slot metadata (mapping each slot index to a `(token_id, expert_slot_j, routing_weight)` triple) is produced during the pre-dispatch packing step on the originating device and consumed during the post-combine accumulation step on the same device. However, the metadata must also be available on the expert-hosting device to correctly organize the combine send buffer.

In practice, there are two approaches:

1. **Transmit metadata with dispatch.** The slot metadata is transmitted as a separate small tensor during the dispatch all-to-all (alongside the token embeddings). This adds a small overhead of $C \times E_d \times \text{metadata\_bytes}$ per device per dispatch, but ensures that the expert device has all the information it needs to pack the combine send buffer correctly.

2. **Mirror-image convention.** If the combine send buffer is defined to use the same row order as the dispatch receive buffer (which was packed by the originating device and therefore implicitly encodes the metadata order), no explicit metadata transmission is needed for the combine. The receiving device (originating device during combine) already holds the slot metadata from the dispatch step and can apply it directly to the received expert outputs.

The mirror-image convention is the more memory-efficient approach; it requires that the combine operation preserves the row order of the expert FFN output buffer exactly as received from dispatch.

---

## Buffer Layout Symmetry with Dispatch

The combine send-count matrix is the transpose of the dispatch send-count matrix; buffer shapes are symmetric. See `all_to_all_dispatch.md` for the full send-count derivation.

```text
Dispatch:
  Device 0 send buffer for device 1: shape [C * E_d, H]
    -> transmitted from device 0 to device 1 during dispatch
    -> becomes device 1's receive buffer for source device 0

Combine:
  Device 1 combine send buffer for device 0: shape [C * E_d, H]
    -> expert outputs corresponding to the same slots as the dispatch receive buffer
    -> transmitted from device 1 to device 0 during combine
    -> becomes device 0's combine receive buffer for source device 1

Observation: both buffers have the same shape [C * E_d, H].
The row-to-slot mapping is the same in both directions.
```

---

## Numerical Considerations

### Accumulation Order and Floating-Point Associativity

The weighted accumulation from Conceptual Contract step 4 is not associative in floating-point arithmetic: different orderings of the $k = 8$ terms may produce different results due to rounding. The magnitude of the difference is typically small (within a few ULPs, units in the last place) for BF16 (bfloat16) with $k = 8$ terms, but it is not zero.

Three sources of non-associativity are relevant:

1. **Order of expert outputs in the combine receive buffer.** Depending on the all-to-all implementation and mesh routing, different devices' contributions may arrive in different orders across forward passes (e.g., if messages are pipelined via different mesh paths under varying network load). If the accumulation always processes received expert outputs in the order they arrive in the receive buffer, the result may differ slightly between passes for the same input.

2. **Parallel accumulation on Tenstorrent cores.** If multiple Tensix cores each accumulate a subset of the $k$ expert outputs for a given token and then their partial sums are combined, the order in which partial sums are merged introduces another non-associativity.

3. **BF16 precision.** BF16 has only 7 bits of mantissa (compared to 23 for FP32). Accumulated rounding errors from 8 multiply-add operations are larger in BF16 than in FP32 or FP16 (which has 10 bits of mantissa). In the worst case, if two expert outputs of similar magnitude but opposite sign cancel partially, precision loss can affect the 1–2 least significant bits of the result.

### Mitigations

- **Fixed accumulation order.** The TTNN implementation should define a deterministic order for combining the $k$ contributions, independent of message arrival order. The simplest convention is to sort by expert index before accumulation: accumulate $\hat{w}_{t,0} o_{e_{t,0}}, \hat{w}_{t,1} o_{e_{t,1}}, \ldots$ in increasing expert index order. This makes results reproducible.

- **Compensated summation.** Kahan summation or a similar technique can reduce accumulated floating-point error below the naive summation bound. For $k = 8$ terms in BF16, the improvement is approximately $k \times \epsilon_\text{machine} \approx 8 \times 3.9 \times 10^{-3} \approx 0.031$ relative error reduction, which is modest but may matter for numerically sensitive decoding operations.

- **FP32 accumulator.** Casting the $k$ weighted products to FP32 before summation and casting back to BF16 at the end eliminates virtually all accumulation error at the cost of higher intermediate storage. This option is worth considering when $k$ is large and numerical stability is critical.

In practice, for most MoE inference workloads, BF16 accumulation with a fixed order is sufficient — the non-associativity errors are below the model's output sensitivity threshold. The choice of mitigation should be validated empirically for any latency-critical deployment.

---

## References

- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Rajbhandari2022] Rajbhandari, S. et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale", ICML, 2022.
- [Higham2002] Higham, N. J., "Accuracy and Stability of Numerical Algorithms", 2nd ed., SIAM, 2002. (Kahan summation and floating-point accumulation error bounds.)
- [Kahan1965] Kahan, W., "Pracniques: Further Remarks on Reducing Truncation Errors", Communications of the ACM, 1965.
- [TTNNDocs] Tenstorrent, "TTNN API Reference", Tenstorrent Developer Documentation.
- [Ch1Routing] Chapter 1, `ch01_moe_fundamentals/routing_problem.md` — weighted accumulation and the combine step conceptual description.
- [Ch2Dispatch] Chapter 2, `all_to_all_dispatch.md` — dispatch operation, slot metadata, and buffer layout conventions.
- [Ch2Background] Chapter 2, `collective_communication_background.md` — all-to-all formal definition.
- [Ch5WeightNorm] Chapter 5, `ch05_routing_weight_optimization/weight_normalization.md` — deferring weight normalization to the combine step.
- [Ch6CombineAcc] Chapter 6, `ch06_fused_dispatch_compute_combine/combine_accumulation.md` — scatter-add accumulation implementation and parallelization.
- [Ch7Capacity] Chapter 7, `ch07_load_balancing/capacity_factor_mechanics.md` — expert capacity $C$ and its effect on buffer sizing.

---

**Next:** [dispatch_combine_overhead.md](./dispatch_combine_overhead.md)
