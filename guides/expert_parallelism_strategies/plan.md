# Research Guide Plan: Expert Parallelism Strategies

## Topic Context

Qwen3.5-35B employs a Mixture-of-Experts (MoE) architecture with 256 total experts and top-8 routing per token. Efficiently dispatching tokens to experts across a multi-device mesh (e.g., T3K with 8 devices) requires careful design of dispatch/combine communication patterns, expert-to-device assignment, and routing weight computation. This guide addresses those three central questions systematically.

---

## Audience

**Primary audience:** ML systems engineers and compiler/runtime developers working on large-scale MoE inference on Tenstorrent hardware. Readers are assumed to have:

- Familiarity with transformer architecture fundamentals (attention, FFN, residual connections)
- Basic understanding of data parallelism and tensor parallelism in multi-device settings
- Working knowledge of Python and familiarity with tensor operation APIs (TTNN or equivalent)
- Conceptual awareness that MoE replaces a single FFN with a learned routing function plus a bank of expert FFNs

Readers are **not** assumed to have prior experience with all-to-all collective communication, expert sharding, or the specifics of Tenstorrent mesh topology.

---

## Chapter List

### Chapter 1 — MoE Fundamentals and the Routing Problem

**Description:** Establishes the core MoE architecture, the role of the router, and why expert placement across devices creates a non-trivial communication problem.

**Directory:** `ch01_moe_fundamentals/`

**Files:**

- `index.md`
  - Chapter overview and reading guide; explains the dependency on no prior chapters and what readers will be able to explain after completing the chapter.

- `moe_architecture.md`
  - Definition of MoE layer: gating network, expert bank, top-k selection.
  - Mathematical formulation: given input token `x`, router produces logits `g = Wx`; softmax over logits; top-k selection yields indices `I` and weights `w_i`.
  - Distinction between sparse MoE (only top-k experts compute) and dense MoE.
  - Qwen3.5-35B specifics: 256 experts per MoE layer, top-8 routing, fraction of parameters that are active vs. total.

- `routing_problem.md`
  - Why routing creates communication overhead: tokens on one device may need experts on another device.
  - Token distribution imbalance: load skew when the router consistently selects a subset of popular experts.
  - Expert capacity: the capacity factor concept and what happens when an expert's token queue overflows (token dropping vs. overflow routing).
  - Introduction of the dispatch/combine communication pattern at a conceptual level (before implementation details).

- `qwen35b_config.md`
  - Architectural constants for Qwen3.5-35B: number of layers, hidden dimension, intermediate dimension, number of experts (256), top-k (8), expert hidden size.
  - Which layers are MoE layers vs. dense layers (if mixed).
  - Activation function choice and its effect on expert compute cost.
  - Why 256 experts with top-8 routing (rather than smaller expert counts) creates unique sharding pressure.

---

### Chapter 2 — All-to-All Communication Primitives

**Description:** Deep-dives into all-to-all collective communication as the canonical mechanism for expert dispatch and combine, covering both the abstract operation and its TTNN implementation.

**Directory:** `ch02_all_to_all_primitives/`

**Files:**

- `index.md`
  - Chapter overview; prereqs: Chapter 1. Summary of what operations are introduced and where they appear in the MoE forward pass.

- `collective_communication_background.md`
  - Taxonomy of collective operations: broadcast, scatter, gather, all-gather, reduce-scatter, all-to-all.
  - Precise definition of all-to-all: each device sends a distinct data slice to every other device, and receives a distinct data slice from every other device.
  - Bandwidth and latency model for all-to-all on a mesh with N devices: volume = `(N-1)/N * total_data`, latency dominated by number of hops on mesh topology.
  - Comparison to all-gather + local select as an alternative (higher bandwidth, no routing-dependent sparsity exploitation).

- `all_to_all_dispatch.md`
  - Semantics of `all_to_all_dispatch` in TTNN: input is a token tensor with routing indices; output routes token embeddings to the device that owns the target expert.
  - Pre-dispatch steps: computing the send counts per device from the routing index tensor; padding to capacity.
  - Sparse vs. dense packing: how tokens are packed into contiguous buffers before the collective to avoid irregular memory access on receive.
  - Kernel-level considerations: tile alignment requirements, shard layout on Tenstorrent cores.
  - Worked example: 4-device case, 8 tokens, top-2 routing — trace the send/receive buffers.

- `all_to_all_combine.md`
  - Semantics of `all_to_all_combine` in TTNN: inverse of dispatch; routes expert output embeddings back to the originating device for accumulation.
  - Weighted accumulation: routing weights `w_i` multiply expert outputs before summing across the top-k contributions for each token.
  - Ordering constraint: combine must know which expert output corresponds to which token and which routing weight.
  - Buffer layout symmetry with dispatch: why dispatch and combine share the same send-count metadata.
  - Numerical considerations: accumulation order and floating-point associativity.

- `dispatch_combine_overhead.md`
  - End-to-end latency breakdown of a single MoE layer: router compute, dispatch collective, expert FFN compute, combine collective.
  - Arithmetic intensity of dispatch and combine vs. expert FFN: why communication can dominate at small batch sizes.
  - Roofline analysis sketch for dispatch/combine on T3K Ethernet links.
  - Identifying the batch-size threshold below which communication overhead exceeds expert compute.

---

### Chapter 3 — Alternative Expert Routing Schemes

**Description:** Surveys routing architectures beyond all-to-all, including expert sharding, hierarchical routing, pipeline-based expert parallelism, and hybrid approaches, with quantitative trade-off analysis.

**Directory:** `ch03_alternative_routing_schemes/`

**Files:**

- `index.md`
  - Chapter overview; prereqs: Chapters 1 and 2. Framing: all-to-all is the baseline; this chapter characterizes when alternatives win.

- `expert_sharding.md`
  - Static expert sharding: assign each expert permanently to one device; no cross-device routing for expert compute.
  - Trade-off: eliminates dispatch/combine communication but requires all tokens to be replicated (or gathered) on the device owning the expert.
  - All-gather-before-routing pattern: all devices gather the full token batch, each device runs its local expert subset, results are reduce-scattered back.
  - Communication volume comparison: all-gather volume = `(N-1)/N * batch * hidden` regardless of routing sparsity vs. all-to-all volume proportional to actual cross-device traffic.
  - When expert sharding wins: very small batch sizes, high expert utilization imbalance.

- `pipeline_expert_parallelism.md`
  - Pipeline expert parallelism: devices arranged in a pipeline; tokens pass through device 0's experts, then device 1's, etc.
  - Only applicable when each token visits multiple experts sequentially (e.g., top-8 with sequential execution).
  - Bubble analysis: pipeline depth = number of expert stages; efficiency degrades with small microbatches.
  - Comparison to all-to-all: pipeline avoids simultaneous all-to-all but introduces serial latency.
  - Applicability to Qwen3.5-35B top-8 routing: if experts are grouped by device, tokens could traverse a 2-stage pipeline (4 experts per stage, 2 stages on 2 device groups).

- `hierarchical_routing.md`
  - Two-level routing: a coarse router first selects a device group, then a fine router selects an expert within the group.
  - Reduces all-to-all communication volume by restricting cross-group traffic.
  - Load balancing challenge: the coarse router must distribute tokens evenly across groups without knowing fine-router decisions.
  - Auxiliary loss implications: hierarchical load-balancing auxiliary losses during training vs. pure inference implications.
  - Relevance to 256-expert, 8-device setting: group size of 32 experts per device; 8-group hierarchical routing could confine top-8 selection within one device in the best case.

- `scheme_comparison_matrix.md`
  - Summary table: rows = routing schemes (all-to-all, expert sharding via all-gather, pipeline, hierarchical); columns = communication volume, latency at various batch sizes, load-balance sensitivity, implementation complexity, applicability to Qwen3.5-35B.
  - Decision flowchart: given batch size B, load imbalance factor L, and device count N, which scheme is preferred.
  - Recommendation rationale for Qwen3.5-35B on T3K.

---

### Chapter 4 — Expert-to-Device Assignment for 256 Experts on 8 Devices

**Description:** Covers static and dynamic strategies for assigning 256 experts to 8 devices, including uniform partitioning, load-aware rebalancing, and expert replication, with analysis specific to the T3K mesh.

**Directory:** `ch04_expert_device_assignment/`

**Files:**

- `index.md`
  - Chapter overview; prereqs: Chapters 1–3. Goal: determine the assignment that minimizes peak device load while respecting memory constraints.

- `uniform_partitioning.md`
  - Baseline: assign 32 experts per device (256 / 8 = 32), round-robin by expert index.
  - Memory footprint per device: expert weight size × 32, plus activations during forward pass.
  - Expected load balance under uniform token distribution: each device receives `batch × top-8 / 8 = batch` tokens on average.
  - Variance in load under non-uniform routing: if some experts are consistently preferred, device hosting popular experts becomes a bottleneck.
  - TTNN sharding implications: how expert weights are laid out in DRAM shards and L1 for 32-expert local batch.

- `load_aware_assignment.md`
  - Expert popularity profiling: collect routing statistics (per-expert activation frequency) over a calibration dataset.
  - Bin-packing formulation: given expert weights `p_e` (popularity), assign to 8 bins minimizing max-bin weight sum.
  - Greedy decreasing algorithm vs. ILP for small expert counts.
  - Dynamic reassignment at inference time: when to re-profile and re-assign (e.g., per-request batch vs. periodic).
  - Practical constraint: reassignment requires weight migration across devices, which has its own overhead cost.

- `expert_replication.md`
  - Replicating hot experts: placing copies of frequently-selected experts on multiple devices to absorb load spikes.
  - Memory cost of replication: replicating 1 expert across all 8 devices costs 8× expert memory.
  - Consistency: replicated experts are read-only at inference; no synchronization needed beyond weight loading.
  - Optimal replication factor: if expert `e` receives `f_e` fraction of tokens, replicate on `ceil(f_e * N)` devices.
  - Interaction with dispatch: dispatch must be aware of replicated experts; tokens can be sent to any replica (load-balance across replicas).
  - For Qwen3.5-35B: estimating the fraction of tokens routed to the top-32 most popular experts and the memory budget for replication.

- `mesh_topology_constraints.md`
  - T3K mesh topology: 8 devices, 2D torus or ring interconnect, link bandwidth and latency characteristics.
  - Locality-aware assignment: place experts whose token streams are correlated (co-activated) on adjacent devices to shorten all-to-all hops.
  - Expert co-activation analysis: if experts A and B are frequently activated together by the same tokens, placing them on the same device eliminates one round-trip dispatch leg.
  - Torus-aware placement algorithm: formulate as a graph partition problem where edge weight = co-activation frequency, partition into 8 groups minimizing cut weight.
  - Practical limits: co-activation data requires profiling; static assignment still outperforms dynamic when profiling cost amortizes.

---

### Chapter 5 — Routing Weight Computation and Overhead Minimization

**Description:** Examines the router's forward pass in detail — from logit computation through softmax and top-k selection — and identifies optimization opportunities to reduce routing overhead.

**Directory:** `ch05_routing_weight_optimization/`

**Files:**

- `index.md`
  - Chapter overview; prereqs: Chapters 1 and 2. Focus: the router is a small but latency-critical sub-graph preceding every MoE layer.

- `router_forward_pass.md`
  - Step-by-step router computation: linear projection `g = xW_r` (hidden_dim → num_experts), softmax or sigmoid normalization, top-k selection.
  - Dimensions: for Qwen3.5-35B, `W_r` is `[hidden_dim, 256]`; at batch size B, `g` is `[B, 256]`.
  - Softmax vs. sigmoid routing: sigmoid allows independent expert probabilities (not forced to sum to 1); implications for weight normalization before combine.
  - Numerical stability: log-sum-exp trick for softmax; importance of not computing full softmax when only top-k values are needed.
  - Auxiliary load-balancing loss: not computed at pure inference time; can be stripped from the inference graph.

- `topk_selection_efficiency.md`
  - Algorithmic complexity of top-k: full sort is O(E log E); partial sort / selection algorithm is O(E + k log k); for E=256, k=8, partial sort saves ~4× over full sort.
  - Hardware-level top-k on Tenstorrent: availability of hardware sort/reduce primitives; tile granularity constraints.
  - Batched top-k: for batch B, compute top-k on `[B, 256]` tensor; vectorization opportunities across the batch dimension.
  - Fusing router projection with top-k: tiled matrix multiply that writes directly into a top-k selection buffer without materializing the full `[B, 256]` logit tensor.

- `weight_normalization.md`
  - Post-selection weight normalization: after top-k, re-normalize the k selected weights to sum to 1 (or pass raw sigmoid probabilities through).
  - When normalization happens relative to dispatch: can be deferred to the combine step (multiply after expert output, not before dispatch) to avoid carrying weight metadata through the all-to-all.
  - Precision requirements: FP16 vs. BF16 for routing weights; weight underflow when k=8 and some experts receive very small probability mass.
  - Fusing weight normalization with combine: implementing weighted accumulation in the combine kernel to avoid a separate pass.

- `router_kernel_fusion.md`
  - Fusion opportunities: (1) linear projection + top-k + index extraction as a single kernel; (2) weight normalization + scatter metadata preparation fused before dispatch.
  - Data flow: routing indices and weights must be available on the device before dispatch; minimizing the critical path from token arrival to dispatch initiation.
  - Latency hiding: overlapping router compute for the next micro-batch with dispatch/combine of the current micro-batch (requires double-buffering).
  - Quantization of router weights: INT8 quantization of `W_r` to reduce router projection cost; accuracy impact at 256 experts.

---

### Chapter 6 — Fused Dispatch-Compute-Combine Pipeline

**Description:** Describes how dispatch, expert FFN computation, and combine can be pipelined and fused to maximize hardware utilization and minimize end-to-end MoE layer latency.

**Directory:** `ch06_fused_dispatch_compute_combine/`

**Files:**

- `index.md`
  - Chapter overview; prereqs: Chapters 2, 4, and 5. This chapter ties primitives together into an optimized execution pipeline.

- `pipeline_design.md`
  - Execution stages of a single MoE layer: (1) route, (2) pack/dispatch, (3) expert FFN, (4) combine/unpack, (5) residual add.
  - Dependency graph: stage 3 (expert FFN) depends on stage 2 completing; stage 4 depends on stage 3; stages 2 and 4 both involve all-to-all.
  - Micro-batch pipelining: divide the token batch into micro-batches; while micro-batch i is in expert FFN, micro-batch i-1 is in combine and micro-batch i+1 is being packed.
  - Buffer requirements for double-buffering dispatch and combine.

- `expert_ffn_tiling.md`
  - Tiling expert FFN computation for 32 local experts: each expert is an independent FFN; can be computed in parallel on different core groups.
  - L1 memory management: expert weight tiles must fit in L1 during computation; streaming strategy for weight tiles that exceed L1.
  - Matmul dimensions after dispatch: received tokens are packed into a `[local_tokens, hidden]` tensor; expert-batched matmul vs. looping over each expert separately.
  - Sparsity exploitation: if some local experts receive zero tokens (due to routing), skip their compute entirely.

- `combine_accumulation.md`
  - After all-to-all combine, each device holds a sparse tensor of expert outputs with routing weights; accumulate into the output tensor.
  - Scatter-add pattern: for each token position, sum k weighted expert outputs; potential race condition if two experts return output for the same token.
  - Parallelizing accumulation: assign token ranges to core groups; each core group handles its assigned tokens' accumulation.
  - Numerical precision of FP16/BF16 accumulation across 8 additions per token.

- `end_to_end_latency_model.md`
  - Parameterized latency model: function of batch size B, hidden dim H, expert FFN intermediate dim D, number of devices N, link bandwidth BW, link latency L.
  - Identify the dominant cost regime: for small B, communication dominates; for large B, expert FFN compute dominates.
  - Model calibration procedure: how to measure each sub-stage on a real T3K system.
  - Bottleneck identification for Qwen3.5-35B at typical inference batch sizes (1–32 tokens).

---

### Chapter 7 — Load Balancing at Inference Time

**Description:** Addresses the practical challenge of uneven expert utilization during inference, covering token dropping, overflow routing, dynamic capacity adjustment, and auxiliary-loss-free load balancing.

**Directory:** `ch07_load_balancing/`

**Files:**

- `index.md`
  - Chapter overview; prereqs: Chapters 1, 4, and 6. Focuses on production inference considerations rather than training-time load balancing.

- `capacity_factor_mechanics.md`
  - Expert capacity: maximum tokens an expert can process in one forward pass = `capacity_factor × (batch_size × top_k / num_experts)`.
  - Consequences of overflow: token dropping (token contributes zero to output) vs. overflow routing to a fallback expert.
  - Choosing capacity factor: trade-off between memory (larger buffer) and quality (fewer dropped tokens).
  - Capacity factor interaction with dispatch buffer size: determines the padded tensor shape sent in all-to-all.

- `dynamic_load_rebalancing.md`
  - Runtime monitoring: tracking per-expert token counts across recent forward passes using a sliding window counter.
  - Adaptive capacity adjustment: increasing the capacity factor for consistently overloaded experts without restarting.
  - Expert load redistribution signal: using monitored load statistics to trigger expert replication or reassignment (link to Chapter 4).
  - Overhead of monitoring: counter updates must not add significant latency to the forward pass; implementation via asynchronous device-side counters.

- `auxiliary_loss_free_inference.md`
  - During training, auxiliary losses (e.g., load-balancing loss, z-loss) encourage even expert utilization.
  - At inference, auxiliary loss terms are absent; the router may exhibit different load distribution than during training.
  - Inference-time load skew: empirical evidence that routers trained with auxiliary losses still show moderate skew at inference.
  - Mitigation strategies: router logit normalization at inference, expert score bias adjustment (adding a learnable or hand-tuned per-expert bias to logits to re-balance).
  - Expert score bias tuning procedure: calibration on a representative dataset to set per-expert biases that equalize utilization.

---

### Chapter 8 — Putting It All Together: Optimal Strategy for Qwen3.5-35B on T3K

**Description:** Synthesizes all prior chapters into a concrete recommended configuration for running Qwen3.5-35B expert parallelism on a T3K 8-device mesh, with justification for each design choice.

**Directory:** `ch08_qwen35b_t3k_strategy/`

**Files:**

- `index.md`
  - Chapter overview; prereqs: all previous chapters. This chapter is a synthesis and decision record, not a source of new concepts.

- `architecture_summary.md`
  - Recap of Qwen3.5-35B MoE layer parameters relevant to expert parallelism decisions: 256 experts, top-8, hidden dim, expert FFN dim, number of MoE layers.
  - T3K hardware parameters: 8 devices, DRAM capacity per device, Ethernet link bandwidth and latency, on-chip SRAM (L1) per device.
  - Derived constraints: expert weight size per device under uniform assignment, peak token throughput per device.

- `recommended_configuration.md`
  - **Expert-to-device assignment:** uniform 32-experts-per-device as the default; load-aware rebalancing if calibration data shows top-decile experts receiving >4× average traffic; replicate the top-8 most popular experts on all devices if memory budget allows.
  - **Dispatch/combine scheme:** `all_to_all_dispatch` and `all_to_all_combine` as the primary mechanism; switch to all-gather-based expert sharding only if batch size consistently falls below the crossover threshold identified in Chapter 6's latency model.
  - **Routing weight processing:** fuse router projection with top-k selection; defer weight normalization to the combine kernel; use BF16 for routing weights.
  - **Pipeline:** enable micro-batch double-buffering for dispatch and combine when batch size > 4; use capacity factor 1.25 as default.
  - **Load balancing:** enable per-expert score biases calibrated on a 512-sample calibration set; monitor with sliding window of 64 steps.
  - Justification for each choice referencing specific chapters and quantitative estimates.

- `open_questions.md`
  - Unresolved questions and areas for future investigation:
    - Optimal co-activation-based expert placement for Qwen3.5-35B (requires large-scale profiling).
    - Whether TTNN's `all_to_all_dispatch` exploits Ethernet link asymmetries in T3K's actual topology.
    - Impact of speculative decoding on MoE load distribution (draft model may use different experts than verifier).
    - Feasibility of INT4 quantization for expert weights and its effect on all-to-all buffer sizes.
    - Expert parallelism interaction with tensor parallelism within each expert FFN.

---

## Conventions

### Terminology

| Term | Definition |
|---|---|
| **Expert** | A single FFN sub-network within an MoE layer. |
| **Router / Gating network** | The learned linear layer + selection function that assigns tokens to experts. |
| **Top-k routing** | Selecting the k experts with highest router logits for each token. For Qwen3.5-35B, k = 8. |
| **Dispatch** | The collective communication step that routes token embeddings from their originating device to the device(s) owning the selected expert(s). |
| **Combine** | The collective communication step that routes expert outputs back to the originating device for weighted accumulation. |
| **Expert capacity** | Maximum number of tokens an expert can process in one forward pass, set by capacity factor × expected tokens per expert. |
| **Capacity factor** | Multiplier (≥ 1.0) applied to the expected tokens-per-expert to set expert capacity. |
| **Load imbalance** | The ratio of the most-loaded expert's token count to the average token count across experts. |
| **Expert replication** | Placing copies of an expert's weights on multiple devices to distribute its token load. |
| **Co-activation** | Two experts being selected by the same token in the same forward pass. |
| **T3K** | Tenstorrent 8-device mesh system used as the target hardware platform throughout this guide. |
| **TTNN** | Tenstorrent's tensor operation library; `all_to_all_dispatch` and `all_to_all_combine` are TTNN operations. |

### Notation

- `E` — total number of experts (256 for Qwen3.5-35B).
- `k` — top-k selection count (8 for Qwen3.5-35B).
- `N` — number of devices (8 for T3K).
- `B` — batch size (number of tokens in a single forward pass).
- `H` — hidden dimension of the model.
- `D` — expert FFN intermediate dimension.
- `p_e` — routing probability for expert `e` (scalar in [0, 1]).
- `I_e` — set of token indices routed to expert `e`.
- `C` — expert capacity (integer, max tokens per expert per forward pass).
- `CF` — capacity factor (float, CF ≥ 1.0).
- Tensors are described with shape notation `[dim0, dim1, ...]`; dimension names are spelled out on first use per file.
- Device indices are 0-based: devices 0 through N−1.
- Expert indices are 0-based: experts 0 through E−1.

### Formatting Rules

- All code snippets use fenced code blocks with language tag (`python`, `cpp`, or `text` for pseudocode).
- Mathematical expressions use LaTeX inline notation (single `$...$`) or display notation (`$$...$$`); do not mix prose fractions with LaTeX in the same sentence.
- Tables must have a header row and use consistent column alignment.
- Figures (diagrams) are referenced by file name within the chapter directory (e.g., `fig_dispatch_flow.png`) and have an accompanying caption immediately below the reference.
- Each file ends with a `## References` section listing cited papers, documentation pages, or other chapters using the format: `- [Short label] Author(s), "Title", Venue/URL, Year.`
- Cross-chapter references use the form: "see Chapter N, `filename.md`" with the exact chapter number and filename.
- Abbreviations are spelled out on first use in each file, even if defined in another file's glossary.
- The guide uses American English spelling throughout.

---

## Cross-Chapter Dependencies

```
Chapter 1 (MoE Fundamentals)
    └── Chapter 2 (All-to-All Primitives)
            ├── Chapter 3 (Alternative Routing Schemes)
            │       └── Chapter 4 (Expert-to-Device Assignment) [also depends on Ch 1]
            ├── Chapter 4 (Expert-to-Device Assignment) [also depends on Ch 1, Ch 3]
            └── Chapter 5 (Routing Weight Optimization) [also depends on Ch 1]
                    └── Chapter 6 (Fused Dispatch-Compute-Combine) [also depends on Ch 2, Ch 4]
                            └── Chapter 7 (Load Balancing) [also depends on Ch 1, Ch 4]
                                    └── Chapter 8 (Synthesis) [depends on all chapters]
```

**Detailed dependency notes:**

- **Ch 2 → Ch 1:** All-to-all dispatch/combine semantics assume familiarity with the MoE layer structure and routing problem defined in Ch 1.
- **Ch 3 → Ch 1, Ch 2:** Alternative schemes are benchmarked against all-to-all (Ch 2) and require understanding the routing problem (Ch 1).
- **Ch 4 → Ch 1, Ch 2, Ch 3:** Expert assignment strategies reference the communication primitives (Ch 2) and benefit from knowing alternative scheme trade-offs (Ch 3).
- **Ch 5 → Ch 1, Ch 2:** Router optimization requires knowing what inputs the router must produce for dispatch (Ch 2) and what the routing problem is (Ch 1).
- **Ch 6 → Ch 2, Ch 4, Ch 5:** The fused pipeline uses the dispatch/combine operations (Ch 2), assumes a fixed expert assignment (Ch 4), and relies on optimized routing weight processing (Ch 5).
- **Ch 7 → Ch 1, Ch 4, Ch 6:** Load balancing mechanisms reference capacity concepts from Ch 1, assignment strategies from Ch 4, and the runtime pipeline from Ch 6.
- **Ch 8 → all:** The synthesis chapter references every prior chapter when justifying its recommendations and discussing open questions.

**Specific concept forward-references to be aware of:**

- Ch 1 (`routing_problem.md`) introduces "expert capacity" conceptually; the precise definition is given in Ch 7 (`capacity_factor_mechanics.md`). Ch 1 should note that the formal definition is deferred.
- Ch 2 (`all_to_all_dispatch.md`) references "send counts per device" which depends on routing indices; Ch 5 details how those indices are computed. Ch 2 should treat routing indices as a given input.
- Ch 3 (`scheme_comparison_matrix.md`) refers to the crossover batch-size threshold; this is formally derived in Ch 6 (`end_to_end_latency_model.md`). Ch 3 should provide a qualitative estimate and point forward to Ch 6.
- Ch 4 (`expert_replication.md`) discusses dispatch awareness of replicated experts; the dispatch mechanism itself is defined in Ch 2. Readers of Ch 4 must have completed Ch 2.
- Ch 7 (`dynamic_load_rebalancing.md`) references expert reassignment from Ch 4; the two chapters form a feedback loop that Ch 8 integrates.
