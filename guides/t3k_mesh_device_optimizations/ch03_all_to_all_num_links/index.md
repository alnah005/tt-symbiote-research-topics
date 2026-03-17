# Chapter 3 — All-to-All Operations and `num_links` Tuning on T3K

Chapter 2 introduced the full TTNN collective API and established the `ttnn.all_to_all` signature. This chapter narrows the focus to `ttnn.all_to_all` specifically: why it is the dominant collective in Mixture-of-Experts (MoE) inference, how the data volumes it moves are derived, and how to select the `num_links` parameter to minimize latency or maximize throughput on T3K.

---

## Why `all_to_all` Dominates MoE Inference

In a dense transformer, the expensive communication primitive is all-reduce: the partial products of column-sharded weight matrices must be summed across devices. In an MoE transformer, the dominant primitive is all-to-all.

The reason is the expert-dispatch mechanism. After the router assigns each token to its top-K experts, the token embedding must travel to the device that holds the selected expert. With N=8 devices and k=8 active experts per token, each token is sent to (N-1)/N × k = 7 distinct remote devices on average. Every MoE layer triggers this dispatch all-to-all and a second combine all-to-all that returns expert outputs to the originating device. A model such as Qwen3.5-35B [D UNVERIFIED] has approximately 80 MoE layers, so the two all-to-all calls per layer execute approximately 160 times per forward pass.

By contrast, all-reduce appears mainly in attention layers and in the shared (non-MoE) FFN blocks, which are fewer in number and smaller in communication volume relative to the MoE expert dispatch traffic.

Understanding and tuning `ttnn.all_to_all` therefore has an outsized impact on end-to-end inference throughput, especially in the decode phase where each step is latency-bound.

---

## What `num_links` Controls and Why It Is the Primary Tuning Knob

`num_links` controls how many physical Ethernet links between adjacent device pairs are allocated to a collective operation, directly trading link setup overhead against aggregate bandwidth. For the complete `num_links` definition, bandwidth model, and tuning guidance, see `num_links_parameter.md`.

---

## Chapter Files

| File | What It Covers |
|---|---|
| [`all_to_all_in_moe.md`](all_to_all_in_moe.md) | Token routing, expert compute, and combine phases; data volume derivation with arithmetic; prefill vs. decode regimes; T3K linear-chain implementation details |
| [`num_links_parameter.md`](num_links_parameter.md) | Definition of `num_links`; bandwidth model; latency vs. throughput trade-off formula; link contention; recommended values by regime |
| [`benchmarking_num_links.md`](benchmarking_num_links.md) | Benchmark setup; sweep methodology; interpreting results; controlling for variability; reference results table structure; when to re-benchmark |

---

## Prerequisites

- **Chapter 1, `ethernet_link_bandwidth.md`**: per-link bandwidth of ~12.5 GB/s, multi-hop routing cost, and link saturation behavior are used directly in the bandwidth model in `num_links_parameter.md`.
- **Chapter 1, `topology_implications_for_collectives.md`**: the linear-chain collective implementation and the 7-hop round sequence for T3K are the foundation of the latency model in `all_to_all_in_moe.md`.
- **Chapter 2, `collective_primitives.md`**: the `ttnn.all_to_all` signature, `cluster_axis=1` convention, `ttnn.Topology.Linear` requirement, and the distinction between dispatch and combine phases are assumed known. This chapter is a direct extension of that material.

If any of those concepts are unfamiliar, return to the prerequisite material before proceeding.

---

## New Notation

Symbol definitions: see `all_to_all_in_moe.md` Quick Reference.

---

## References

- Chapter 1, `ethernet_link_bandwidth.md` — per-link bandwidth and latency figures
- Chapter 1, `topology_implications_for_collectives.md` — linear-chain collective algorithm and hop count analysis
- Chapter 2, `collective_primitives.md` — `ttnn.all_to_all` API signature and parameters
- Chapter 2, `mesh_device_setup.md` — `MeshDevice` construction for T3K `(1, 8)` mesh
