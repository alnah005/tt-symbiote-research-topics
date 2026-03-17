# Chapter 2: SiLU on Wormhole Hardware

This chapter explains how SiLU activation physically executes on a Tenstorrent Wormhole B0
Tensix core. You will learn which execution unit runs SiLU, why it cannot overlap with the
preceding matmul, and how its instruction depth compares to simpler activations like ReLU.

**Prerequisites:** Complete Chapter 1 (`ch01_silu_in_moe_architecture/`) before reading this
chapter. Chapter 1 establishes the FFN dataflow and the role of SiLU in the gated MLP block.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Identify the three execution units in a Tensix core: the RISC-V unpacker/packer, the FPU
   (matrix engine), and the SFPU (vector engine).
2. Explain why FPU and SFPU are sequential pipelines — a post-matmul activation cannot overlap
   with the matmul that produced its input tile.
3. Describe how SiLU decomposes into SFPU LLK instructions and why sigmoid requires a
   polynomial approximation rather than a single hardware instruction.
4. State the 32-pass SFPU constraint for a BF16 tile: the SFPU LReg is 32 elements wide, so a
   32×32 tile requires 32 sequential passes through the LReg.
5. Compare the relative cost ordering of element-wise activations: ReLU < SiLU ≈ GELU in SFPU
   cycles per tile.
6. Place matmul (FPU) and SiLU (SFPU) on a roofline: matmul can be compute-bound at large batch
   sizes, while SiLU is always memory-bound at practical sizes.

---

## Chapter Contents

| File | Topic |
|---|---|
| `tensix_compute_engine.md` | Tensix core architecture: RISC-V, FPU, and SFPU pipelines |
| `silu_sfpu_execution.md` | How SiLU decomposes into SFPU LLK instructions; the 32-pass constraint |
| `cycles_vs_matmul.md` | SFPU cycle cost of SiLU versus FPU matmul cost; roofline placement |

---

## Reading Order

Read the files in the order listed in the table above. Each file builds on the previous one:
`tensix_compute_engine.md` establishes the hardware model, `silu_sfpu_execution.md` applies that
model to the specific instruction sequence for SiLU, and `cycles_vs_matmul.md` uses both to
build quantitative cost estimates and roofline intuition. After this chapter, proceed to
Chapter 3 (`ch03_measurement_methodology/`) to learn how to measure these costs empirically.
