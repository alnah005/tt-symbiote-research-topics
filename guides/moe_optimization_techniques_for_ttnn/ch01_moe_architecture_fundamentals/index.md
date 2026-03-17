# Chapter 1: MoE Architecture Fundamentals

## Overview

This chapter establishes the vocabulary and structural patterns of Mixture of Experts (MoE) models that all subsequent optimization discussion in this guide depends on. Before you can tune a TTNN kernel or choose between `batched matmul` and `sparse_matmul`, you need a clear mental model of what MoE layers are doing, how routing decisions shape the compute graph, and why MoE workloads are structurally different from dense transformer FFN layers.

This chapter does not assume any prior knowledge of MoE models beyond a basic understanding of transformer architecture. If you already understand top-K routing, expert capacity, and load balancing losses, you can skim the first two files (`moe_overview.md` and `routing_and_sparsity.md`) and focus on `moe_on_hardware.md` before moving to Chapter 2.

---

## Learning Objectives

By the end of this chapter, you will be able to:

1. Describe the dispatch-combine pattern that defines all MoE layers.
2. Explain top-K routing and why it produces sparse activation across the expert pool.
3. Identify the key structural differences between Switch Transformer, Mixtral, and DeepSeek-MoE routing strategies.
4. Define **expert_capacity**, **sparsity ratio**, and **sparsity pattern** as these terms are used throughout this guide.
5. Explain why MoE layers are harder to execute efficiently on accelerators than dense FFN layers.
6. Describe the two TTNN strategies (batched matmul and `sparse_matmul`) that this guide evaluates.

---

## Prerequisites Checklist

Before proceeding, verify that you are comfortable with the following:

- [ ] You can write a standard two-layer FFN forward pass in PyTorch (`linear → activation → linear`).
- [ ] You understand what a softmax over logits produces (a probability distribution).
- [ ] You are familiar with `torch.topk`, `torch.gather`, and basic tensor indexing.
- [ ] You have run at least one `ttnn.matmul` call and understand the role of `ttnn.to_device` and `ttnn.from_device`.

If any item above is unfamiliar, review the TTNN Getting Started documentation before continuing.

> **Note:** TTNN uses 32×32 element tiles as its atomic compute unit. This is formally introduced in Chapter 2. The only tile-relevant detail in this chapter is expert_capacity tile-alignment to 32 (defined in Chapter 2; in this chapter, you need only know that `expert_capacity` must be a multiple of 32); incorrect alignment causes silent shape errors. This is explained inline where it appears.

---

## Contents

| File | What it covers |
|---|---|
| [`moe_overview.md`](./moe_overview.md) | What MoE layers are; the gating network, expert pool, and dispatch-combine pattern; sparse vs. dense activation; common model families; how MoE changes the compute graph vs. dense FFN |
| [`routing_and_sparsity.md`](./routing_and_sparsity.md) | Token-to-expert assignment; expert capacity; load balancing losses; dropped tokens; the relationship between routing decisions and downstream compute shape; sparsity ratio definition |
| [`moe_on_hardware.md`](./moe_on_hardware.md) | Why MoE is challenging on accelerators; naive looping over experts and why it is slow; preview of batched matmul and `sparse_matmul` approaches in TTNN |

Read the files in the order listed above. Each file ends with a "Next Steps" section that links forward.

---

## Key Terms Introduced or Referenced in This Chapter

The following terms are defined precisely in the sub-topic files and are used without re-definition in all later chapters.

| Term | Introduced in |
|---|---|
| **expert** | `moe_overview.md` |
| **gating network** | `moe_overview.md` |
| **expert pool** | `moe_overview.md` |
| **dispatch-combine pattern** | `moe_overview.md` |
| **top-K routing** | `moe_overview.md` |
| **sparsity ratio** | `routing_and_sparsity.md` |
| **sparsity pattern** | `routing_and_sparsity.md` |
| **expert_capacity** | `routing_and_sparsity.md` |
| **dropped tokens** | `routing_and_sparsity.md` |
| **load balancing loss** | `routing_and_sparsity.md` |
| **sparsity tensor** | Conceptually introduced in `routing_and_sparsity.md`; formally defined in Chapter 5 |

`sparsity pattern` (introduced in this chapter) is an operational concept describing which experts are active per token. The `sparsity tensor` is the TTNN data structure encoding this pattern; its format is defined in Chapter 5.

---

## Next Steps

Start with [`moe_overview.md`](./moe_overview.md).
