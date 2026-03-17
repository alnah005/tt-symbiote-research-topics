# Chapter 2: Math Fidelity Levels — LoFi vs HiFi2 vs HiFi4

## Prerequisites

- Chapter 1: `ch1_kernel_config_fundamentals/` — in particular `math_fidelity_overview.md`, which introduces the `ttnn.MathFidelity` enum and the concept of mantissa bit truncation

---

## Overview

Chapter 1 established that `math_fidelity` controls how many mantissa bits from each BF16 operand are presented to the Tensix FPU multiplier during a dot-product accumulation, and that the default is `LoFi`. This chapter characterizes the four fidelity levels quantitatively: how much throughput each one costs, how much PCC each one preserves, and which level is appropriate for each MoE projection type.

The central result is that MoE expert matmuls do not all require the same fidelity. Gate and up projections tolerate `LoFi` because their outputs feed into a nonlinearity that absorbs small mantissa errors. Down projections require `HiFi2` because their outputs accumulate directly into the residual stream, where rounding errors compound and persist through subsequent layers.

---

## Learning Objectives

After completing this chapter you should be able to:

1. State what math fidelity controls at the silicon level — specifically, the number of mantissa bits used per multiply-accumulate in the Tensix FPU tile pipeline.
2. Explain why LoFi offers higher throughput (~2× over HiFi4) for matmul-intensive workloads by completing each tile FMA in fewer cycles.
3. Distinguish when HiFi2 vs HiFi4 precision is needed: HiFi2 for down projections (residual stream accumulation), HiFi4 for reference validation or operations feeding layer norm / softmax.
4. Describe the PCC measurement approach for validating fidelity choices: run the same inputs through a PyTorch float32 reference and a TTNN BF16 matmul, then compute Pearson correlation coefficient on the flattened outputs.
5. Apply the fidelity selection workflow: start at HiFi4 to confirm baseline correctness, step down to HiFi2 and then LoFi, accept the lowest fidelity that keeps PCC above the deployment threshold.

---

## Fidelity Selection Decision Table

For the full fidelity comparison with K_t and PCC data, see `fidelity_selection_workflow.md`.

---

## Chapter Contents

| File | Contents |
|---|---|
| `index.md` (this file) | Chapter overview, learning objectives, decision table, reading order |
| `fidelity_precision_model.md` | What math fidelity controls in the FPU; throughput and PCC characterization table |
| `fidelity_and_moe_accuracy.md` | Why gate/up projections tolerate LoFi; why down projections need HiFi2; K-loop depth and rounding accumulation; PCC measurement in Python |
| `fidelity_selection_workflow.md` | Step-by-step workflow for choosing fidelity for a new MoE model; parameterized PCC sweep template; decision thresholds |

---

## Reading Order

1. `fidelity_precision_model.md` — establish the quantitative characterization of each fidelity level
2. `fidelity_and_moe_accuracy.md` — understand why different MoE projections map to different fidelity levels
3. `fidelity_selection_workflow.md` — apply the selection process to a new model

---

## Next Steps

Begin with `fidelity_precision_model.md` for the hardware-level explanation of what the fidelity enum actually changes in the FPU pipeline, along with the throughput and PCC reference table for MoE-representative matrix shapes.
