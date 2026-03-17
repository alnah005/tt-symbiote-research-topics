# Chapter 2: TTNN and Wormhole Hardware Primer

## Overview

Before optimizing Mixture-of-Experts models in TTNN, you need a working mental model of the hardware you are targeting and the programming abstractions sitting on top of it. This chapter builds that foundation.

Chapter 1 established what MoE models are and why they are computationally interesting. This chapter answers the question: *what does the hardware actually do with those computations, and how does TTNN expose control over that hardware?*

The goal is not exhaustive coverage of every Wormhole feature or every TTNN API. The goal is to give you exactly enough background to reason about kernel performance, memory pressure, and configuration choices that appear in Chapters 3 through 7.

---

## Learning Objectives

By the end of this chapter you should be able to:

1. Describe the internal structure of a Tensix core and explain how the RISC-V cores, Math engine, and NoC interact during a matmul.
2. Explain the memory hierarchy on Wormhole B0 — specifically L1 SRAM per core versus DRAM bandwidth — and articulate why that hierarchy shapes tile scheduling strategy.
3. Describe how a logical 2D core grid maps to physical Tensix cores and how a T3K multi-chip mesh extends that grid across 8 chips.
4. Distinguish between TTNN tensor layouts (tile vs. row-major), dtypes (bfloat16 vs. bfloat8_b), and memory placement configs (DRAM vs. L1, interleaved vs. sharded).
5. Trace the path from a Python `ttnn.matmul(...)` call down to a Metalium kernel on device, and explain why static shapes enable program caching.
6. Map M, K, N matmul dimensions onto the Tensix grid, identify the role of the 32×32 tile as the atomic compute unit, and state the output subblock constraints.
7. Choose between `MatmulMultiCoreReuseMultiCastProgramConfig` and `MatmulMultiCoreProgramConfig` for a given workload shape.

---

## Prerequisites

- **Chapter 1 — MoE Architecture Fundamentals**: familiarity with MoE layer structure (router, expert FFNs, token dispatch) is assumed throughout this chapter. References to "expert weight matrices" and "token batches" are used without re-introduction.

---

## Files in This Chapter

| File | Contents |
|------|----------|
| [wormhole_architecture.md](wormhole_architecture.md) | Tensix core internals, L1 vs. DRAM memory hierarchy, core grid layout, T3K multi-chip mesh |
| [ttnn_programming_model.md](ttnn_programming_model.md) | Tensor shapes and dtypes, memory configs, op dispatch from Python to Metalium, tracing and program caching |
| [matmul_fundamentals_in_ttnn.md](matmul_fundamentals_in_ttnn.md) | Grid mapping for M/K/N, tile size as atomic unit, output subblock constraints, program config selection |

Read the files in order. `matmul_fundamentals_in_ttnn.md` builds on concepts from both preceding files.

---

## Official Reference Documentation

This chapter summarizes concepts for practical use. For deeper reference consult:

- [Tenstorrent TTNN Python API documentation](https://docs.tenstorrent.com/ttnn/latest/)
- [TT-Metalium programmer guide](https://docs.tenstorrent.com/tt-metalium/latest/)
- Tenstorrent Wormhole architecture whitepaper (available on the Tenstorrent developer portal)

> **Tip:** The TTNN docs and Metalium docs describe the same hardware from different abstraction levels. When a TTNN config parameter behaves unexpectedly, reading the corresponding Metalium kernel source is often the fastest way to understand why.

---

## Key Terms Introduced in This Chapter

The following key terms are used without re-explanation in later chapters (some defined inline below, others defined in the file indicated):

- **Tensix core** — the programmable compute tile on Wormhole
- **L1 SRAM** — the fast, per-core scratchpad memory
- **NoC** — Network-on-Chip, the on-die interconnect between Tensix cores
- **Tile** — a 32×32 matrix of elements; the atomic unit of TTNN compute
- **`M_t`, `K_t`, `N_t`** — tile counts along the M, K, N matmul dimensions (e.g., `M_t = seq / 32`)
- **`out_subblock_h`, `out_subblock_w`** — output subblock dimensions in tiles; defined fully in `matmul_fundamentals_in_ttnn.md`
- **`MatmulMultiCoreReuseMultiCastProgramConfig`** — program config for large matmuls with weight multicasting
- **`MatmulMultiCoreProgramConfig`** — program config for small or irregular matmuls without multicasting
- **Interleaved placement** — see `ttnn_programming_model.md` Section 2.3
- **Sharded placement** — see `ttnn_programming_model.md` Section 2.3

---

## Next Steps

Start with [wormhole_architecture.md](wormhole_architecture.md) to build the hardware mental model, then proceed to [ttnn_programming_model.md](ttnn_programming_model.md) and [matmul_fundamentals_in_ttnn.md](matmul_fundamentals_in_ttnn.md) in order.

Once you have completed all three files, Chapter 3 (Expert Tensor Parallelism) will build directly on the grid layout and program config vocabulary established here.
