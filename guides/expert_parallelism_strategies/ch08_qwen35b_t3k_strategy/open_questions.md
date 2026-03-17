# Open Questions and Future Investigation Areas

This file lists unresolved questions that affect expert parallelism decisions for Qwen3.5-35B on
T3K. Each item describes what is unknown, why it matters for the configuration in
`recommended_configuration.md`, and what evidence would resolve it.

---

## Q1: Expert Co-Activation Structure

**Question:** Which pairs (or groups) of experts are frequently co-activated — selected together
by the same token in the same forward pass — and does this structure persist across inputs?

**Why it matters:** Co-activation patterns could inform locality-aware expert placement (Chapter 4,
`mesh_topology_constraints.md`). If experts $e_i$ and $e_j$ are co-activated by a large fraction
of tokens, placing them on the same device eliminates one leg of the dispatch round-trip for those
tokens. For $k=8$, each token selects 8 experts; there are $\binom{256}{8} \approx 10^{14}$ possible
co-activation tuples, but in practice only a small number of subsets dominate.

**What would resolve it:** Profile routing decisions on a representative Qwen3.5-35B serving
workload (≥50,000 tokens) and compute pairwise co-activation counts $c_{ij}$ for all expert pairs.
Build a co-activation graph with edge weight $c_{ij}$; run a graph partition into 8 balanced groups
minimizing cut weight. Compare per-device dispatch volume before and after locality-aware assignment.

**Current assumption:** Round-robin uniform assignment. This is load-balanced in expectation but
ignores co-activation locality.

---

## Q2: TTNN `all_to_all` Link Utilization on T3K's Physical Topology

**Question:** Does TTNN's `all_to_all` implementation exploit the 1×8 linear mesh topology, or
does it use a generic all-to-all schedule that may not minimize hop count?

**Why it matters:** On a 1×8 linear mesh, the average hop count is 3.0. An optimal all-to-all
schedule would pipeline transfers to avoid link contention and saturate each link's 12.5 GB/s
capacity. A naive schedule (all devices send simultaneously to all destinations) can cause
contention on interior links, reducing effective bandwidth below the single-link rate.

**What would resolve it:** Measure Ethernet link utilization per port (inner vs. outer devices)
during `ttnn.all_to_all` dispatch at $B=32$. Compare to the theoretical maximum of 6.4 MB / 12.5 GB/s
= 0.51 ms. A measured latency significantly above 0.51 ms (e.g., > 2×) would indicate contention.
Inspect the TTNN `all_to_all` implementation for topology-aware routing.

**Current assumption:** $t_{\text{dispatch}} \approx 0.51$ ms at $B=32$, treating the mesh as a
single effective link of 12.5 GB/s. This may be optimistic for interior devices.

---

## Q3: Speculative Decoding Interaction

**Question:** How does speculative decoding change the expert load distribution on T3K?

**Why it matters:** In speculative decoding, a small draft model generates several candidate tokens,
and a larger verifier model (Qwen3.5-35B) evaluates them in a single forward pass. This changes
the effective batch structure: the verifier sees a batch of $B' = B \times \text{spec\_depth}$
positions at once, with some positions being draft tokens that will be rejected. The MoE routing
distribution for rejected tokens is discarded, but their expert capacity slots are consumed.

**Specific sub-questions:**

1. Does the draft model's routing distribution correlate with the verifier's? If they diverge
   significantly, load imbalance worsens because draft tokens consume capacity slots of experts
   they will not actually contribute to.
2. At $B' = 32 \times 4 = 128$ (4× speculation depth), capacity $C = \lceil 8 \times 128 \times 1.25 / 256 \rceil = 5$,
   and the workload may transition toward compute-bound. How does this interact with the
   `sparse_matmul` vs. dense matmul threshold?

**What would resolve it:** Run Qwen3.5-35B in speculative decode mode on T3K; profile per-expert
activation frequencies and the fraction of capacity consumed by rejected draft positions; compare
end-to-end latency against non-speculative decode.

---

## Q4: INT4 Expert Weight Quantization

**Question:** Is INT4 quantization of expert FFN weights ($W_{\text{gate}}$, $W_{\text{up}}$,
$W_{\text{down}}$) feasible for Qwen3.5-35B, and what does it imply for dispatch buffer sizes
and DRAM bandwidth?

**Why it matters (memory):** INT4 reduces expert weight size from BF16's 2 bytes/element to 0.5
bytes/element — a 4× reduction. Expert weights dominate per-device DRAM usage; INT4 could reduce
expert weight footprint from several GB to ~1 GB per device, enabling expert replication that is
currently DRAM-infeasible.

**Why it matters (computation):** DRAM read bandwidth during expert FFN matmul is currently
proportional to weight size. INT4 reduces weight-streaming bandwidth by 4×, potentially shifting
the bottleneck from memory-bandwidth-bound (weight streaming) to compute-bound (INT4 matmul
throughput on Tensix FPUs).

**Why it matters (dispatch):** Dispatch volume is determined by *activation* tensor size (BF16
hidden states), not weight size. INT4 quantization of weights does not reduce dispatch volume.

**What would resolve it:** Run Qwen3.5-35B with INT4 expert weights (e.g., using a W4A16 scheme)
and measure perplexity degradation vs. BF16. Profile the FFN matmul latency with INT4 weights to
confirm the expected 4× reduction in DRAM read bandwidth.

---

## Q5: Expert Parallelism + Tensor Parallelism Interaction

**Question:** Can tensor parallelism (TP) within each expert FFN be combined with the EP scheme
described in this guide, and if so, how do the all-to-all schedules interact?

**Why it matters:** At $B=32$, token utilization per expert is only 6.25% (C=2 out of 32 tile rows).
Even with `sparse_matmul`, the effective compute per expert is small. Tensor parallelism within
each expert — splitting $W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$ across multiple
cores — could improve arithmetic intensity by increasing the per-core workload without changing
the number of tokens.

**Conflict to resolve:** EP all-to-all moves tokens across devices. TP all-reduce moves partial
weight products across cores (or devices). These two collectives would need to be scheduled without
deadlock and without saturating the Ethernet links simultaneously. On a 1×8 linear mesh, EP uses
inter-device Ethernet; TP uses either intra-device NOC (if cores on the same chip) or also
Ethernet (if experts are split across devices). If both require Ethernet simultaneously, link
contention could negate the TP benefit.

**What would resolve it:** Model the combined EP+TP communication schedule as a directed graph;
identify potential link-contention cycles. Alternatively, measure experimentally with TP within
each expert confined to intra-device cores only (no additional Ethernet traffic).

---

## Q6: Qwen3.5-35B Expert FFN Intermediate Dimension

**Question:** What is the exact value of $D$ (expert FFN intermediate dimension) for Qwen3.5-35B?

**Why it matters:** $D$ determines:

- Expert weight size per device: $3 \times 32 \times 7168 \times D \times 2$ bytes
- `per_core_N` in the matmul program config: $\lceil D / 32 \rceil / \text{grid\_cols}$
- Expert FFN FLOP count and arithmetic intensity

Several estimates in prior chapters are marked [UNVERIFIED] because $D$ is not confirmed in
publicly available Qwen3.5-35B documentation.

**What would resolve it:** Inspect the Qwen3.5-35B model checkpoint or architecture config file
for the `moe_intermediate_size` field.

---

## Q7: Which Transformer Layers Use MoE vs. Dense FFN

**Question:** Does Qwen3.5-35B use MoE in all transformer blocks, alternating blocks, or a
different pattern?

**Why it matters:** If some blocks use dense FFN, the per-layer latency model changes: dense blocks
have no dispatch/combine overhead and their FFN matmul has different shape and core assignment.
The fraction of MoE layers determines the total communication budget for the model as a whole.

**What would resolve it:** Inspect the Qwen3.5-35B model configuration (e.g., `config.json`) for
per-layer `ffn_type` or similar field. Cross-reference with any published architecture description.

---

## References

- Chapter 4, `mesh_topology_constraints.md` — co-activation graph partition approach (Q1)
- Chapter 2, `dispatch_combine_overhead.md` — A2A latency model; assumption of 0.51 ms (Q2)
- Chapter 7, `capacity_overflow_handling.md` — Poisson overflow; capacity with speculation (Q3)
- Chapter 6, `expert_ffn_tiling.md` — sparse_matmul threshold; INT4 arithmetic intensity (Q4)
- Chapter 6, `pipeline_design.md` — TP+EP scheduling conflict (Q5)
- Chapter 1, `qwen35b_config.md` — architectural constants including unverified D (Q6, Q7)
