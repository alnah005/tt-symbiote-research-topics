# Pipeline Expert Parallelism

## Purpose

This file describes pipeline expert parallelism — a routing scheme in which devices are arranged as a sequence of stages and tokens flow through those stages one at a time. It analyzes bubble overhead, derives the per-stage communication cost, compares the scheme against all-to-all, and evaluates its applicability to Qwen3.5-35B on T3K (1×8 linear chain, `ttnn.Topology.Linear`).

**Prerequisite:** Chapter 2, `ch02_all_to_all_primitives/dispatch_combine_overhead.md` (baseline latency); Chapter 3, `index.md` (notation); Chapter 3, `expert_sharding.md` (all-gather as an alternative).

---

## Concept: Pipeline Expert Parallelism

Pipeline expert parallelism treats the expert execution path as a directed linear pipeline. Devices are numbered $0, 1, \ldots, N-1$. Each device owns a group of experts. A token entering the pipeline visits device 0's experts first, then device 1's experts, then device 2's experts, and so on, until it has visited all required expert stages.

```text
Token batch
    |
    v
[Device 0 — experts 0..31]
    | point-to-point send
    v
[Device 1 — experts 32..63]
    | point-to-point send
    v
[Device 2 — experts 64..95]
    | ...
    v
[Device 7 — experts 224..255]
    |
    v
Final token outputs (weighted combine on originating device)
```

This is distinct from the all-to-all scheme, where all $N$ devices communicate simultaneously. In pipeline execution, communication is point-to-point and sequential: at each stage, the active device sends tokens to the next device in the chain.

---

## Precondition: Sequential Expert Execution

Pipeline expert parallelism requires that a token's $k$ selected experts are executed **sequentially**, one stage at a time. This is in contrast to the all-to-all scheme, where all $k$ expert computations for a token's selected experts happen in **parallel** — device 0 processes its assigned tokens while devices 1 through 7 simultaneously process their respective tokens.

For Qwen3.5-35B, the standard top-$k = 8$ routing executes expert computations in parallel: once `all_to_all_dispatch` completes, all 8 devices simultaneously compute their local expert FFNs. Pipeline expert parallelism is applicable to Qwen3.5-35B only if the 8 selected experts per token are deliberately arranged into a sequential execution schedule, where each stage contributes to a running accumulation rather than all stages executing simultaneously.

**Practical note:** Qwen3.5-35B was trained with parallel top-$k = 8$ execution. Applying pipeline expert parallelism would require either retraining with a sequential objective or post-hoc re-ordering that increases end-to-end latency relative to the parallel baseline. This limits pipeline expert parallelism to cases where sequential execution is architecturally mandated.

One scenario where sequential execution is natural is a model with $k = 1$ and many layers (each token visits exactly one expert per MoE layer, and successive MoE layers are pipelined). Another is a model with a deliberate "chain-of-experts" training objective. Neither applies to Qwen3.5-35B's flat top-8 routing.

---

## Pipeline Structure for Qwen3.5-35B (Hypothetical)

To analyze the pipeline scheme concretely, consider how it might be applied to Qwen3.5-35B with $k = 8$ experts per token and $N = 8$ devices:

One natural mapping is $P = k = 8$ pipeline stages, where each stage executes exactly one of the 8 selected experts. The token visits each device in sequence; device $i$ runs the $i$-th selected expert for each token. In this arrangement, the pipeline depth equals $k = 8$.

An alternative mapping uses $P = N = 8$ stages, with each device executing its $E_d = 32$ local experts for whatever tokens arrive in that stage. With $k = 8$ experts to visit and $N = 8$ devices, one expert is visited per device per pipeline pass.

Both mappings yield $P = 8$ stages for this configuration.

---

## Bubble Analysis

### Pipeline Efficiency

In a pipeline of $P$ stages, the first stage is active during the first micro-batch, but stages 2 through $P$ are idle (waiting for data) — this is the "fill" bubble. Similarly, when the last micro-batch enters stage $P$, stages 1 through $P-1$ drain — the "drain" bubble. For $\mu$ micro-batches of $M$ tokens each:

- Total pipeline stages active: $\mu + (P - 1)$ time slots (fill + $\mu$ active + drain counted together).
- Useful work slots: $\mu \times P$.
- Pipeline efficiency:

$$\eta = \frac{\mu \times P}{(\mu + P - 1) \times P} = \frac{\mu}{\mu + P - 1}$$

For $P = 8$ stages and $\mu = 1$ micro-batch: $\eta = 1 / (1 + 7) = 1/8 = 12.5\%$. Hardware is idle 87.5% of the time. This is extremely poor utilization.

For $P = 8$ stages and $\mu = 8$ micro-batches: $\eta = 8 / (8 + 7) = 8/15 \approx 53.3\%$.

For $P = 8$ stages and $\mu = 64$ micro-batches: $\eta = 64 / (64 + 7) = 64/71 \approx 90.1\%$.

**Arithmetic check ($\mu = 8$, $P = 8$):** numerator $= 8$; denominator $= 8 + 8 - 1 = 15$; $\eta = 8/15 \approx 0.533$. Confirmed.

The bubble overhead is severe unless the number of micro-batches substantially exceeds the pipeline depth. For decode inference ($B \leq 32$, $S = 1$), achieving $\mu \gg 8$ requires extremely fine micro-batch granularity, down to $M = 1$ to $4$ tokens per micro-batch — which may not be achievable without padding overhead.

---

## Per-Stage Communication Cost

### Point-to-Point Send

At each pipeline stage, the active device sends its token batch to the next device in the chain. This is a single point-to-point send (device $i \to$ device $i+1$), not an all-to-all.

Volume sent per pipeline stage (one micro-batch of $M$ tokens, each of length $H$, BF16):

$$V_\text{stage} = M \times H \times 2 \text{ bytes}$$

For $M = B / \mu$ (uniform micro-batch split), with $B = 32$ and $\mu = 8$ ($M = 4$ tokens):

$$V_\text{stage} = 4 \times 7{,}168 \times 2 = 57{,}344 \text{ bytes} \approx 56 \text{ KB}$$

**Arithmetic check:** $4 \times 7{,}168 = 28{,}672$; $28{,}672 \times 2 = 57{,}344$ bytes. Confirmed.

Per-stage send time (at T3K Ethernet bandwidth $\text{BW} = 12.5\,\text{GB/s}$):

$$T_\text{stage, comm} = \frac{V_\text{stage}}{\text{BW}} = \frac{57{,}344}{12.5 \times 10^9} \approx 4.6\,\mu\text{s}$$

This is far lower than the all-to-all collective time of $\sim 257\,\mu\text{s}$ per collective at $B = 32$, $C = 1$ (Chapter 2, `dispatch_combine_overhead.md`). However, the pipeline incurs this send time $P - 1 = 7$ times serially per token, and each stage must wait for the previous stage's computation to complete before sending.

### Total Serial Latency

For a single token traversing $P = 8$ stages, the critical-path latency is:

$$T_\text{pipeline, token} = \sum_{i=0}^{P-1} \left(T_{\text{FFN}, i} + T_{\text{comm}, i}\right)$$

where $T_{\text{FFN}, i}$ is the expert FFN compute time at stage $i$ and $T_{\text{comm}, i}$ is the point-to-point send from stage $i$ to stage $i+1$ ($0$ for the last stage). Under uniform loading, each stage processes the same number of tokens, so all $T_{\text{FFN}, i}$ are equal and the sum simplifies to:

$$T_\text{pipeline, token} = P \times T_\text{FFN, stage} + (P-1) \times T_\text{comm, stage}$$

For $P = 8$, $T_\text{FFN, stage} = T_\text{FFN}/P$ (expert FFN time split across stages), and $T_\text{comm, stage}$ as above.

The pipeline only improves throughput when multiple micro-batches overlap; a single token or single micro-batch always pays the full serial pipeline latency.

---

## Comparison to All-to-All

### Latency at Small Batch Size (Decode: $B \leq 32$, $S = 1$)

| Scheme | Communication | Compute | Total critical path |
|---|---|---|---|
| All-to-all (baseline) | $\sim 257\,\mu\text{s}$ per collective $\times 2$ collectives $= 514\,\mu\text{s}$ | $\sim 10.75\,\mu\text{s}$ expert FFN `[D UNVERIFIED]` | $\sim 525\,\mu\text{s}$ |
| Pipeline ($P=8$, $\mu=1$) | $(P-1) \times T_\text{stage, comm} \approx 7 \times 4.6 = 32\,\mu\text{s}$ (for $M=32$) | $P \times T_\text{FFN}/P = T_\text{FFN} \approx 10.75\,\mu\text{s}$ `[D UNVERIFIED]` | $\sim 43\,\mu\text{s}$, but at 12.5% efficiency |

> Note: these communication figures for pipeline assume $M = B = 32$ with $\mu = 1$; with $M = 32$, $V_\text{stage} = 32 \times 7{,}168 \times 2 = 458{,}752$ bytes, $T_\text{stage, comm} \approx 458{,}752 / (12.5 \times 10^9) \approx 36.7\,\mu\text{s}$ per stage; 7 stages $\approx 257\,\mu\text{s}$ total — matching all-to-all when only one micro-batch is in flight.

The comparison changes when $\mu > 1$: with enough micro-batches, pipeline throughput improves, but per-token latency (time from entering the pipeline to exiting) always equals $P \times (T_\text{FFN, stage} + T_\text{comm, stage})$ regardless of $\mu$.

### Topological Fit on T3K

The T3K 8-device mesh uses a 1×8 linear layout with `ttnn.Topology.Linear`. This is **not** a ring — data flows along a linear chain from device 0 to device 7. Pipeline expert parallelism is a natural fit for this topology: the sequential send from device $i$ to device $i+1$ matches the physical point-to-point links along the linear chain. No routing through intermediate devices is required.

By contrast, the all-to-all collective on a linear chain requires multi-hop routing for non-adjacent device pairs (e.g., device 0 to device 7 traverses 7 hops), which increases effective latency and contends for shared links. Pipeline expert parallelism avoids this contention by using only adjacent-device links at each stage.

---

## When Pipeline Expert Parallelism Wins

Pipeline expert parallelism is preferred when all of the following hold:

1. **Tokens visit experts sequentially** (not in parallel). This requires either $k = 1$ or an explicit sequential execution schedule.

2. **Micro-batch count $\mu \gg P$**. Efficiency $\eta = \mu / (\mu + P - 1)$ approaches 1 only when $\mu$ is large relative to $P$. For T3K with $P = 8$, at least $\mu \approx 32$ to $64$ micro-batches are needed for $\eta > 80\%$.

3. **Batch size is large enough to sustain fine micro-batching** without padding overhead. Prefill batches ($B \times S \geq 512$) are more amenable than decode batches ($B \times S \leq 32$).

4. **The model uses a linear execution topology** (which matches T3K's physical layout). Ring topologies would also work but do not add extra benefit over all-to-all in that case.

As established in the Precondition section, Qwen3.5-35B's parallel top-$k$ routing makes this scheme inapplicable without retraining.

---

## References

- [Huang2019] Huang, Y. et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", NeurIPS, 2019.
- [Narayanan2021] Narayanan, D. et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", SC'21, 2021.
- [Rajbhandari2022] Rajbhandari, S. et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale", ICML, 2022.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Ch2Overhead] Chapter 2, `ch02_all_to_all_primitives/dispatch_combine_overhead.md` — baseline all-to-all latency decomposition.
- [Ch3Index] Chapter 3, `ch03_alternative_routing_schemes/index.md` — chapter overview and notation.
- [Ch3Sharding] Chapter 3, `ch03_alternative_routing_schemes/expert_sharding.md` — all-gather alternative.
- [Ch6Pipeline] Chapter 6, `ch06_fused_dispatch_compute_combine/pipeline_design.md` — micro-batch pipelining of all-to-all collectives (distinct from pipeline expert parallelism).
