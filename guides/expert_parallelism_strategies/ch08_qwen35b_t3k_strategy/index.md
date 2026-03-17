# Chapter 8 — Putting It All Together: Optimal Strategy for Qwen3.5-35B on T3K

> **Quick Reference — Key Constants**
>
> | Symbol | Value | Source |
> |---|---|---|
> | `E` | 256 experts | Qwen3.5-35B config |
> | `k` | 8 (top-8 routing) | Qwen3.5-35B config |
> | `N` | 8 devices | T3K hardware |
> | `E_d` | 32 experts/device | E / N |
> | `H` | 7168 | Qwen3.5-35B hidden dim |
> | `CF` | 1.25 | recommended default |
> | `BW` | 12.5 GB/s | T3K Ethernet per link |
> | `C` (B=32) | 2 | ceil(8×32×1.25/256) |

This chapter is a synthesis and decision record. It does not introduce new concepts; instead, it assembles the recommendations from Chapters 1–7 into a single, actionable configuration for running Qwen3.5-35B expert parallelism on a T3K 8-device mesh.

**Prerequisite chapters:** All of Chapters 1–7.

---

## Chapter Contents

| File | Description |
|---|---|
| `architecture_summary.md` | Qwen3.5-35B and T3K parameters; derived constraints under uniform EP |
| `recommended_configuration.md` | Concrete recommended config with justification for each decision |
| `open_questions.md` | Unresolved questions and future investigation areas |

---

## Coordinating Insight

The dominant design constraint for Qwen3.5-35B on T3K is the combination of **high expert count** ($E=256$) and **low decode batch size** ($B \leq 32$):

- **High $E$** means each device holds only 32 experts under uniform EP, giving token capacity $C = \lceil k \times B \times \text{CF} / E \rceil = 2$ at $B=32$. This is far below the 32-token tile boundary, so token utilization of each expert's compute is at most 6.25%.
- **Low $B$** means dispatch volume is small (~6.4 MB at $B=32$), and the workload is **communication-bound**: the all-to-all latency (~0.51 ms each direction) dominates expert FFN compute.

All recommendations in this chapter flow from this regime. Readers familiar with the individual chapter analyses will find the synthesis unsurprising; the value here is consolidation and explicit justification.

---

## Reading Order

For a practitioner implementing this system for the first time:

1. `architecture_summary.md` — confirm your hardware and model match the assumptions
2. `recommended_configuration.md` — adopt the configuration; read justifications for any choice you need to tune
3. `open_questions.md` — track which decisions may need revisiting as the system matures

---

## References

- Chapter 1, `moe_architecture.md` — MoE layer structure; top-k routing definition
- Chapter 2, `dispatch_combine_overhead.md` — communication vs. compute regime thresholds
- Chapter 4, `uniform_partitioning.md` — baseline 32-expert-per-device assignment
- Chapter 6, `end_to_end_latency_model.md` — bottleneck identification at small batch sizes
- Chapter 7, `capacity_overflow_handling.md` — CF=1.25 default and overflow rate
