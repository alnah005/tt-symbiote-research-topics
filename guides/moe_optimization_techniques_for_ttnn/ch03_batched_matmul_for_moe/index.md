# Chapter 3: Batched Matmul for MoE — Approach and Performance

## Overview

Chapter 2 established the TTNN programming model and the matmul configuration vocabulary: tile dimensions, output subblock constraints, L1 budgeting, and program config selection. This chapter puts those tools to work on the first concrete MoE optimization strategy: **batched matmul**.

The central idea is to gather tokens by expert assignment, stack them into a dense batched tensor, and run a single batched `ttnn.matmul` call across all experts simultaneously. This avoids a Python-level loop over 256 experts — a loop that would issue 256 separate kernel dispatches and leave most of the Tensix grid idle during each one.

This chapter covers how to formulate that batched tensor, how to select and validate a program config for it, and what performance to expect across different batch size and sequence length regimes.

---

## Learning Objectives

By the end of this chapter you should be able to:

1. Describe the gather-pad-matmul pattern for batched MoE expert computation and explain why padding to expert capacity C is required for static-shape tracing.
2. Derive the expert capacity formula $C = \lceil k \times B \times S / E \rceil$ and compute the resulting FLOP efficiency for a given routing load.
3. Construct the correct activation and weight tensor shapes for a batched expert FFN matmul (`[E, C, H]` × `[E, H, D]` → `[E, C, D]`) and identify the role of the batch dimension in TTNN kernel dispatch.
4. Select `MatmulMultiCoreReuseMultiCastProgramConfig` parameters (`in0_block_w`, `out_subblock_h`, `out_subblock_w`, `per_core_M`, `per_core_N`) for the batched MoE case and validate them against tile divisibility and L1 budget constraints.
5. Explain how latency decomposes into gather/scatter overhead and pure matmul time, and identify when each term dominates.
6. State the conditions under which batched matmul is the preferred strategy and name the primary bottlenecks at low expert capacity (decode regime).

---

## Chapter Notation

The following symbols are used throughout this chapter. Refer back to this table whenever a variable appears without context.

| Symbol | Meaning | Value for Qwen3.5-35B MoE |
|--------|---------|--------------------------|
| $B$ | Batch size (number of sequences) | varies |
| $S$ | Sequence length (tokens per sequence) | varies |
| $H$ | Model hidden dimension (`d_model`) | 7168 |
| $D$ | Per-expert intermediate dimension (`d_ff`) | **[D UNVERIFIED — verify against Qwen3 Technical Report]** |
| $E$ | Number of experts | 256 |
| $k$ | Top-K routing: experts activated per token | 8 |
| $C$ | Expert capacity (max tokens routed to one expert) | $\lceil k \times B \times S / E \rceil$ |
| $M_t$ | Tile count along M dimension | $\lceil C / 32 \rceil$ |
| $K_t$ | Tile count along K dimension | $\lceil H / 32 \rceil = 224$ |
| $N_t$ | Tile count along N dimension | $\lceil D / 32 \rceil$ **[N_t UNVERIFIED]** |

`out_subblock_h` and `out_subblock_w` are defined in Chapter 2 (`matmul_fundamentals_in_ttnn.md`, Section 3) and are used here without re-definition.

---

## Summary: Batched Matmul for MoE

| | |
|---|---|
| **Core idea** | Gather tokens by expert → pad to capacity C → stack into `[E, C, H]` → one batched matmul call |
| **TTNN op** | `ttnn.matmul` with a 3D or 4D batched activation tensor |
| **Program config** | `MatmulMultiCoreReuseMultiCastProgramConfig` (prefill / high-C) or `MatmulMultiCoreProgramConfig` (decode / C=1) |

### Pros, Cons, and Recommended Use Cases

| Dimension | Detail |
|-----------|--------|
| **Pro: single kernel dispatch** | All 256 experts are computed in one `ttnn.matmul` call; no Python loop overhead, no per-expert kernel launch latency. |
| **Pro: static shapes for tracing** | Padding to capacity C gives a fixed tensor shape per forward pass; TTNN program cache hits on every step after the first. |
| **Pro: hardware utilization at high load** | When $C$ is large (prefill with long sequences), most token slots are filled and compute efficiency is high. |
| **Pro: simple code path** | Gather + matmul + scatter is a three-step pattern with no sparsity bookkeeping. |
| **Con: padding waste at low load** | Decode with $B=32$, $E=256$, $k=8$ gives $C=1$; 255 of 256 token slots in any given expert may be empty. FLOP efficiency = filled / $(C \times E)$ can be very low. |
| **Con: gather/scatter DRAM cost** | Constructing the `[E, C, H]` tensor requires gathering non-contiguous token rows from a `[B×S, H]` buffer, which is a strided DRAM read with poor locality. |
| **Con: shape recompilation** | Any change to $B$, $S$, or $C$ invalidates the program cache and triggers recompilation. |
| **Recommended: prefill, large batch** | High $C$ → high FLOP efficiency; gather cost amortized over many tokens. |
| **Recommended: balanced routing** | When routing is uniform, all experts receive similar load and padding waste is minimized. |
| **Not recommended: decode, imbalanced routing** | $C=1$ per expert means extreme padding waste; Chapter 4 (`sparse_matmul`) is likely preferred here. |

---

## Files in This Chapter

| File | Contents |
|------|----------|
| [formulating_batched_matmul.md](formulating_batched_matmul.md) | Gather-pad-matmul pattern, expert capacity derivation, tensor shapes, weight layout, worked decode example |
| [program_configs_batched.md](program_configs_batched.md) | Config parameter selection, tile count derivation, L1 budget formulas, decode and prefill example configs, validation checklist |
| [performance_profile_batched.md](performance_profile_batched.md) | Latency breakdown, FLOP efficiency vs. capacity, arithmetic intensity, known bottlenecks, contrast with Chapter 4 |

Read the files in order. `program_configs_batched.md` assumes you have read `formulating_batched_matmul.md` and understand the tensor shapes involved.

---

## Prerequisites

- **Chapter 1** — MoE Architecture Fundamentals: familiarity with top-K routing, expert capacity, and the token dispatch-combine pattern is assumed throughout. `expert_capacity`, `sparsity ratio`, and `load balancing` are used without re-introduction.
- **Chapter 2** — TTNN and Wormhole Primer: all matmul program config vocabulary (`M_t`, `K_t`, `N_t`, `per_core_M`, `per_core_N`, `out_subblock_h`, `out_subblock_w`, `in0_block_w`), L1 budget formulas, and the distinction between `MatmulMultiCoreReuseMultiCastProgramConfig` and `MatmulMultiCoreProgramConfig` are assumed known.

---

## Next Steps

Begin with [formulating_batched_matmul.md](formulating_batched_matmul.md) to understand the tensor construction pattern, then proceed to [program_configs_batched.md](program_configs_batched.md) for kernel configuration, and finish with [performance_profile_batched.md](performance_profile_batched.md) for performance analysis.

After completing this chapter, Chapter 4 covers `sparse_matmul` — an alternative kernel strategy suited to the low-capacity decode regime where batched matmul incurs high padding waste.
