# Scheme Comparison Matrix

## Purpose

This file synthesizes the analysis from `expert_sharding.md`, `pipeline_expert_parallelism.md`, and `hierarchical_routing.md` into a structured comparison table, an ASCII decision flowchart, and a concrete recommendation for Qwen3.5-35B on T3K. The formal crossover derivation is deferred to Chapter 6, `ch06_fused_dispatch_compute_combine/end_to_end_latency_model.md`; this file provides a summary with references to that chapter for quantitative thresholds.

**Prerequisite:** Chapter 3, `expert_sharding.md`; Chapter 3, `pipeline_expert_parallelism.md`; Chapter 3, `hierarchical_routing.md`.

---

## Summary Comparison Table

The table below compares all four schemes across six dimensions. "Small batch" means decode workload ($B \leq 32$, $S = 1$, $B \times S \leq 32$). "Large batch" means prefill workload ($B = 32$, $S = 2{,}048$, $B \times S = 65{,}536$).

Notation: $V$ = bytes per device per collective; $B$ = total tokens per device; $H = 7{,}168$; $N = k = 8$; $E_d = 32$.

| Dimension | All-to-All (baseline) | Expert Sharding (all-gather) | Pipeline Expert Parallelism | Hierarchical Routing |
|---|---|---|---|---|
| **Cross-device volume (formula)** | $\frac{N-1}{N} \times B \times k \times H \times 2$ per collective | $(N-1) \times B \times H \times 2$ per all-gather | $M \times H \times 2$ per pipeline stage send (point-to-point) | $\frac{k_c(N-1)}{N} \times k_f \times B \times H \times 2$ (equals all-to-all when $k_c k_f = k$, $k = N$) |
| **Volume for Qwen3.5-35B ($k=N=8$)** | $100{,}352 \times B$ bytes | $100{,}352 \times B$ bytes | $M \times 14{,}336$ bytes (one stage) | Same as all-to-all in uniform case; 0 if local group always selected |
| **Latency: small batch (decode, $B\times S \leq 32$)** | $\sim 257\,\mu\text{s}$ per collective; total $\sim 514\,\mu\text{s}$ `[D UNVERIFIED]` | Same comm; $N\times$ router compute overhead; same total | Serial: $\sim 257\,\mu\text{s}$ comm for $\mu=1$; 12.5% pipeline efficiency at $P=8$ | Not applicable to Qwen3.5-35B (requires retraining) |
| **Latency: large batch (prefill, $B \times S = 65{,}536$)** | Scales linearly with $B\times S$; communication still dominates `[D UNVERIFIED]` | Same comm; $8\times$ router overhead at $NB\times S$ tokens | Improves with $\mu$; efficiency $\eta = \mu/(\mu+7)$; need $\mu \geq 64$ for $\eta > 90\%$ | Not applicable to Qwen3.5-35B (requires retraining) |
| **Load-balance sensitivity** | Moderate: load skew increases effective $C$ and dispatch volume | Low: all tokens gathered regardless of routing; no capacity issue | High: uneven stage loads create bubbles and stall the pipeline | Very high: coarse router must distribute evenly across $G = 8$ groups without fine-router knowledge |
| **Implementation complexity** | High: routing metadata, capacity allocation, two all-to-all collectives | Medium: no routing metadata; simple all-gather/reduce-scatter; $N\times$ larger gathered tensor | High: pipeline orchestration, micro-batch management, bubble padding, sequential execution constraint | Very high: two-level router architecture, hierarchical auxiliary losses, modified dispatch logic; requires retraining |
| **Applicability to Qwen3.5-35B on T3K** | **Yes — recommended default** | Yes (with $k=N$, volumes equal; simplicity benefit) | No (requires sequential expert execution; Qwen3.5-35B uses parallel top-8) | No (requires retraining with hierarchical routing objective) |

### Volume Arithmetic (Qwen3.5-35B, $B = 32$, BF16)

- All-to-All: $\frac{7}{8} \times 32 \times 8 \times 7{,}168 \times 2 = \frac{7}{8} \times 3{,}670{,}016 = 3{,}211{,}264$ bytes $\approx 3.06$ MiB. **Arithmetic check:** $7 \times 4{,}096 = 28{,}672$; no — directly: $7 \times 32 \times 8 \times 7{,}168 \times 2 / 8 = 7 \times 32 \times 7{,}168 \times 2 = 7 \times 458{,}752 = 3{,}211{,}264$ bytes. Confirmed.
- All-gather: $(8-1) \times 32 \times 7{,}168 \times 2 = 7 \times 32 \times 7{,}168 \times 2 = 3{,}211{,}264$ bytes. Equal to all-to-all. Confirmed.

---

## Crossover: All-to-All vs. All-Gather

For Qwen3.5-35B with $k = N = 8$, the communication volumes of all-to-all and all-gather are algebraically identical (derived in `expert_sharding.md`):

$$V_{a2a} = V_\text{gather} = (N-1) \times B \times H \times 2 \text{ bytes}$$

Therefore the crossover is **not determined by communication volume** but purely by the relative compute overhead:

- All-to-all: router runs on $B$ local tokens; no compute wasted.
- All-gather: router runs on $N \times B$ gathered tokens; $N = 8\times$ more router compute.

**Decision rule for Qwen3.5-35B ($k = N = 8$):**

$$\text{Prefer all-gather if: } T_\text{router}(NB) - T_\text{router}(B) < T_\text{metadata overhead of all-to-all dispatch}$$

where the right-hand side captures the engineering cost of routing metadata management (send-count computation, sparse packing, tile alignment, capacity reservation). For small batches where the router is very cheap and implementation simplicity is paramount, all-gather may be preferred.

The formal threshold for this comparison is not derived in this chapter; it depends on TTNN-specific metadata overhead that is benchmarked in Chapter 6, `end_to_end_latency_model.md`.

---

## ASCII Decision Flowchart

The following flowchart guides scheme selection given workload and model characteristics. Inputs are: batch size $B \times S$, load imbalance factor $L$ (ratio of peak to average expert load), whether the model was trained with hierarchical routing, and whether $k = N$.

```text
START
  |
  v
Was the model trained with hierarchical routing?
  |
  +--YES--> [Use hierarchical routing]
  |           Benefit: potential communication reduction
  |           if coarse router achieves p_local > 1/N
  |
  +--NO
      |
      v
   Is sequential expert execution required (k=1 or
   chain-of-experts training objective)?
      |
      +--YES--> Is micro-batch count mu >> pipeline depth P?
      |           |
      |           +--YES (mu >= 4P)--> [Pipeline expert parallelism]
      |           |                    Best for prefill, large mu
      |           |
      |           +--NO (mu < 4P)---> [All-to-all]
      |                               Pipeline efficiency < 50%; all-to-all cheaper
      |
      +--NO (parallel top-k, standard MoE)
          |
          v
       Is k == N?   (For Qwen3.5-35B: k=8, N=8 => YES)
          |
          +--YES--> Is implementation simplicity the priority
          |         AND B*S very small (B*S < E/k = 32)?
          |           |
          |           +--YES--> [Expert sharding (all-gather)]
          |           |         Volumes equal; simpler buffer management;
          |           |         avoids capacity reservation overhead
          |           |
          |           +--NO---> [All-to-all (baseline)]
          |                     Avoids N-fold router compute overhead;
          |                     better compute efficiency
          |
          +--NO (k < N)
              |
              V_{a2a}/V_gather = k/N < 1
              |
              +--All-to-all is cheaper in communication AND compute.
              |
              v
           [All-to-all (baseline)] — clear winner when k < N
```

**Summary rules (numbered for quick reference):**

1. If the model was trained with hierarchical routing: use hierarchical routing.
2. If sequential expert execution is required AND $\mu \geq 4P$: use pipeline expert parallelism.
3. If $k < N$: use all-to-all (lower volume and no compute waste).
4. If $k = N$ AND $B \times S < E/k$ AND simplicity is valued: use all-gather (expert sharding).
5. Otherwise: use all-to-all.

---

## Recommendation for Qwen3.5-35B on T3K

### Recommended Default: All-to-All

All-to-all dispatch and combine (Chapter 2, `all_to_all_dispatch.md` and `all_to_all_combine.md`) is the **recommended default** for Qwen3.5-35B on T3K. The justification:

- Qwen3.5-35B uses flat top-$k = 8$ routing: hierarchical routing requires retraining and is not applicable.
- Qwen3.5-35B executes all 8 experts per token in parallel: pipeline expert parallelism requires sequential execution and is not applicable.
- All-to-all avoids the $N = 8\times$ router compute overhead of all-gather.
- All-to-all integrates directly with TTNN's `all_to_all_dispatch` and `all_to_all_combine` primitives, which are the supported path for expert parallelism on T3K.

### Viable Alternative: Expert Sharding (All-Gather)

Expert sharding via all-gather is a viable secondary option for Qwen3.5-35B on T3K when:

- $B \times S$ is very small (e.g., single-token decode with $B = 1$, $S = 1$), where capacity management overhead of all-to-all may dominate.
- Implementation complexity must be minimized (e.g., prototyping or debugging without dispatch metadata bookkeeping).

Because $k = N = 8$, the communication volumes are identical between all-to-all and all-gather for this model. The trade-off is purely between the $8\times$ router compute overhead (all-gather's cost) and the metadata/capacity overhead (all-to-all's cost). At $B = 1$, both overheads are small in absolute terms and the implementation simplicity of all-gather may justify it.

At larger batch sizes, the $8\times$ router compute overhead of all-gather is non-negligible ($2 \times NB \times H \times E$ vs. $2 \times B \times H \times E$ FLOPs), and all-to-all is preferred.

**Crossover batch size formula:** the all-to-all metadata overhead equals the extra router compute cost when:

$$T_\text{metadata}^\text{a2a} = T_\text{router}(NB) - T_\text{router}(B) = 2(N-1) \times B \times H \times E / \text{TFLOP}_\text{peak}$$

Substituting $N = 8$, $H = 7{,}168$, $E = 256$, $\text{TFLOP}_\text{peak} = 262 \times 10^{12}$:

$$T_\text{extra router}(B) = \frac{2 \times 7 \times B \times 7{,}168 \times 256}{262 \times 10^{12}} = \frac{25{,}690{,}112 \times B}{262 \times 10^{12}} \approx 9.8 \times 10^{-8} \times B \text{ seconds}$$

For $B = 32$: extra router time $\approx 3.1\,\mu\text{s}$. This is small compared to the $\sim 514\,\mu\text{s}$ all-to-all collective time, confirming that at $B = 32$, all-to-all is preferred on compute grounds. The crossover batch size $B^*$ (where extra router cost equals metadata overhead) depends on the TTNN-specific metadata latency, which is measured in Chapter 6, `end_to_end_latency_model.md`.

### Not Applicable: Pipeline Expert Parallelism and Hierarchical Routing

Both pipeline expert parallelism and hierarchical routing require architectural changes incompatible with off-the-shelf Qwen3.5-35B:

- **Pipeline:** requires sequential expert execution (Qwen3.5-35B uses parallel top-8).
- **Hierarchical:** requires retraining with a two-level routing objective and auxiliary losses (Qwen3.5-35B uses flat routing).

These schemes remain relevant for future model designs. Chapter 6, `ch06_fused_dispatch_compute_combine/end_to_end_latency_model.md`, provides the formal latency crossover derivation that would govern selection between schemes if a model supporting multiple schemes were considered.

---

## Key Results Carried Forward to Later Chapters

| Result | File where it is used |
|---|---|
| All-to-all is recommended default for Qwen3.5-35B | Chapter 8, `ch08_qwen35b_t3k_strategy/recommended_configuration.md` |
| All-gather viable at very small $B\times S$ for Qwen3.5-35B | Chapter 8, `ch08_qwen35b_t3k_strategy/recommended_configuration.md` |
| $V_\text{gather} = V_{a2a}$ when $k = N$ | Chapter 4, `ch04_expert_device_assignment/uniform_partitioning.md` |
| Hierarchical routing requires retraining | Chapter 4, `ch04_expert_device_assignment/load_aware_assignment.md` |
| Pipeline expert parallelism not applicable to parallel top-$k$ | Chapter 6, `ch06_fused_dispatch_compute_combine/pipeline_design.md` |

---

## References

- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", JMLR, 2022.
- [Rajbhandari2022] Rajbhandari, S. et al., "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale", ICML, 2022.
- [Hwang2023] Hwang, C. et al., "Tutel: Adaptive Mixture-of-Experts at Scale", MLSys, 2023.
- [Zoph2022] Zoph, B. et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", arXiv:2202.08906, 2022.
- [Huang2019] Huang, Y. et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", NeurIPS, 2019.
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Ch2Overhead] Chapter 2, `ch02_all_to_all_primitives/dispatch_combine_overhead.md` — all-to-all baseline latency and $D^* \approx 97{,}813$ crossover [D UNVERIFIED].
- [Ch3Sharding] Chapter 3, `ch03_alternative_routing_schemes/expert_sharding.md` — all-gather volume formula and compute waste analysis.
- [Ch3Pipeline] Chapter 3, `ch03_alternative_routing_schemes/pipeline_expert_parallelism.md` — bubble analysis and pipeline efficiency.
- [Ch3Hierarchical] Chapter 3, `ch03_alternative_routing_schemes/hierarchical_routing.md` — two-level routing and load-balance challenges.
- [Ch6Latency] Chapter 6, `ch06_fused_dispatch_compute_combine/end_to_end_latency_model.md` — formal crossover derivation for all-to-all vs. all-gather; referenced for $B^*$ threshold.
- [Ch8Config] Chapter 8, `ch08_qwen35b_t3k_strategy/recommended_configuration.md` — final recommended configuration.

---

**Next:** [Chapter 4 — Expert Device Assignment](../ch04_expert_device_assignment/index.md)
