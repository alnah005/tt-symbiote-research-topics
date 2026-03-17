# MoE on Hardware

## Why MoE Is Challenging on Accelerators

Dense transformer FFN layers map cleanly onto the compute model of modern accelerators: they are large, regular matrix multiplications with static shapes. Accelerators like Tenstorrent Wormhole are built to maximize throughput on exactly this kind of operation — large tiles, reusable weight data, predictable memory access patterns, and no branching.

MoE layers violate every one of these properties. The following three problems arise directly from the routing and sparsity characteristics described in `routing_and_sparsity.md`.

---

### Problem 1: Irregular Memory Access

After the router assigns tokens to experts, the dispatch step must gather tokens from their original positions in the input tensor and place them into per-expert buffers. In terms of memory access, this is an **indexed gather** — not a sequential read, but an indirect one where the address of each element to be read is stored in a separate index tensor.

```python
# Gathering tokens for each expert
# flat_x: [T, d_model] — all tokens in sequence order
# indices: [T, top_k] — for each token, which k experts it goes to

# Conceptual gather for expert e:
# Find all token positions where indices[:, k] == e for any k
token_positions = (indices == e).any(dim=1).nonzero(as_tuple=True)[0]
expert_input = flat_x[token_positions]  # non-sequential read from flat_x
```

Non-sequential memory access is slow on accelerators. DRAM bandwidth is optimized for sequential (burst) reads; random access patterns saturate fewer memory channels, causing the dispatch step to become memory-bound rather than compute-bound. On Tenstorrent Wormhole, the NoC (Network on Chip) routes data between DRAM banks and Tensix cores; random access patterns require more NoC transactions and reduce effective bandwidth. The same problem occurs in reverse at the combine step.

### Problem 2: Expert Imbalance

Even with a well-trained router and a load balancing loss, actual token-to-expert assignments at inference time are not perfectly uniform. Some experts consistently receive more tokens than others for certain input distributions (e.g., certain token types, domains, or languages tend to activate specific experts).

Expert imbalance creates a **tail latency problem**: the total time to complete one MoE forward pass is determined by the slowest expert (the one with the most assigned tokens), not the average expert. If one expert receives 80% of capacity and another receives 20%, computing both takes as long as computing the full-capacity expert — even though the under-loaded expert's compute is largely wasted.

On a hardware mesh like T3K (8 Wormhole chips), expert parallelism distributes experts across chips. Expert imbalance then becomes **inter-chip load imbalance**: some chips spend most of the forward pass time computing while others wait. This is a fundamental efficiency barrier for large-scale MoE inference.

### Problem 3: Synchronization Costs

The dispatch-combine pattern requires synchronization between the routing step and the expert computation step. In a sequential execution model, this means:

1. Complete all router computations across all tokens.
2. Complete all dispatch (gather) operations for all experts.
3. Run all expert FFN computations.
4. Complete all combine (scatter) operations.
5. Continue to the next layer.

Each phase must fully complete before the next can begin, because phase N+1 reads the outputs of phase N. This serializes what would otherwise be independent expert computations: even if experts could theoretically run in parallel (on separate cores), the gather step that precedes them and the scatter step that follows them create synchronization barriers.

On TTNN, ops are dispatched asynchronously to the device, but data dependencies enforce ordering. The gather operation must complete before expert matmuls can begin, and all expert matmuls must complete before the scatter/combine can start. The latency of synchronization is non-trivial at the scales required by models like Mixtral or DeepSeek-MoE.

---

## Why Naive Looping Over Experts Is Slow

The conceptual implementation in `moe_overview.md` loops over `num_experts` and calls the expert FFN for each one independently. This is the intuitive approach, but it is deeply problematic on accelerators.

### Kernel Launch Overhead

Every call to a TTNN op (like `ttnn.matmul`) involves dispatching a program to the device — allocating a Tensix grid, loading the kernel, and configuring data movement. This overhead is on the order of microseconds per dispatch (exact cost varies with program cache state; see below) (kernel launch overhead; memory-access latency is separate). For a model with `num_experts = 8` and `top_k = 2`:

- The naive loop iterates over all `num_experts` regardless of which experts receive non-zero token assignments in the batch — even experts receiving zero tokens are dispatched and produce zero-padded output. As a result, each MoE layer requires up to 24 matmul dispatches in the unfused case: 3 weight matrices per 3-matrix SwiGLU expert × 8 experts. Fused gate+up implementations reduce this to 16 dispatches per MoE layer.
- Mixtral 8x7B's SwiGLU experts use 3 weight matrices per expert (W_gate, W_up, W_down). With 32 transformer layers each containing an MoE sublayer, the per-forward-pass dispatch total is 768 matmul dispatches.
- At even 10 µs per dispatch on a cache miss (a conservative lower-bound estimate; actual cold-start misses can be significantly longer), that is 7.68 ms of dispatch overhead alone, before any compute.

> **Note:** Mixtral 8x22B uses 56 transformer layers rather than 32 (Mistral AI, 2024), giving 56 × 24 = 1,344 matmul dispatches per forward pass under the naive loop approach.

> **Note:** Actual dispatch latency varies with firmware version, op type, and program cache state. The 10 µs figure is illustrative. Performance numbers are indicative; always re-profile on the target firmware and model checkpoint.

Batching expert computations into a single op (or a small number of ops) eliminates this overhead.

### Underutilized Cores

Each individual expert matmul in the naive loop has a small matrix dimension: `[n_tokens_for_expert, d_model]` multiplied by `[d_model, d_ff]`. For `n_tokens_for_expert = 10` and `d_model = 4096`, the activation matrix spans less than one tile in the M dimension (`⌈10 / 32⌉ = 1` tile, but only 10 of 32 rows are populated). The core problem is very low arithmetic intensity: dispatching one token (or a handful of tokens) to an expert means the expert's entire weight matrix (`[d_model, d_ff]`) must be read from memory for just a single activation vector (or a small number of them). The ratio of compute (FLOPs) to memory traffic (bytes of weight data loaded) is tiny, so the matmul is heavily memory-bound rather than compute-bound. A Tensix grid assigned to this matmul will be severely underutilized because the available work cannot saturate the compute units.

Dense accelerator throughput is achieved when matrix dimensions are large enough to tile across the full grid. Individual expert matmuls with a small number of tokens per expert produce minuscule matrices that cannot saturate the hardware.

### No Pipeline Overlap

When expert computations are dispatched one at a time, there is no opportunity to pipeline expert N's computation with expert N+1's data movement. All expert inputs must be gathered before any expert matmul starts (in the per-expert loop model), and the loop processes experts serially. A batched approach can overlap data movement and compute across experts, improving hardware utilization.

---

## Preview: Two TTNN Strategies

This guide covers two approaches that address the above problems, each suited to different operating regimes.

### Strategy 1: Batched Matmul (Chapter 3)

The batched matmul strategy pads all expert inputs to `expert_capacity` and stacks them into a single 3D tensor, then computes all expert FFNs with a single batched matmul call. This eliminates per-expert kernel launch overhead and gives the matmul enough work to fill the Tensix grid. The cost is padding waste: empty token slots (sparsity) are computed anyway. Batched matmul is preferred in the prefill regime, where sequence lengths are long and most expert capacity slots are occupied. Chapter 3 covers this in full.

### Strategy 2: sparse_matmul (Chapter 4)

The `sparse_matmul` strategy uses a sparsity tensor — a tile-level boolean mask derived from the router output — so that the kernel skips zero-padded tiles entirely during computation. This avoids the padding waste of the batched matmul approach. It is most effective in the decode regime, where short sequence lengths mean most experts receive few or no tokens and the sparsity ratio is high. Chapter 4 covers this in full.

### Comparison at a Glance

| Property | Batched Matmul | sparse_matmul |
|---|---|---|
| Handles zero-padded slots | Computes them (waste) | Skips them (efficient at high sparsity) |
| Requires sparsity tensor | No | Yes |
| Best regime | Prefill (long seq, many tokens) | Decode (short seq, high sparsity) |
| Kernel launch count | Low (1–2 per MoE layer; dependent on kernel fusion configuration; see Chapter 3) | Low (1–2 per MoE layer; dependent on kernel fusion configuration; see Chapter 4) |
| Config complexity | Low (standard matmul; static shapes) | Medium (sparsity mask; variable non-zero count) |

The boundary between "batched matmul is better" and "`sparse_matmul` is better" depends on the sparsity ratio, sequence length, `d_model`, and `d_ff`. Chapter 6 provides the full decision framework and crossover analysis.

---

## Roadmap for the Rest of This Guide

With Chapter 1 complete, you have the vocabulary and structural understanding needed to engage with the hardware-level details in the chapters that follow:

- **Chapter 2** introduces the Wormhole hardware architecture and TTNN programming model — the substrate on which both strategies run.
- **Chapter 3** covers batched matmul for MoE in full detail: formulation, program configs, and performance profiles.
- **Chapter 4** covers `sparse_matmul` for MoE: internals, when it wins, and program configs.
- **Chapter 5** is a complete practical guide to sparsity tensor construction from router output.
- **Chapter 6** synthesizes Chapters 3–5 into a decision framework.
- **Chapter 7** extends to T3K multi-chip expert parallelism.
- **Chapter 8** provides an end-to-end optimization workflow and troubleshooting reference.

---

## Next Steps

Proceed to [Chapter 2: TTNN and Wormhole Hardware Primer](../ch02_ttnn_wormhole_primer/index.md) to build the hardware and programming model knowledge required before diving into kernel configuration in Chapter 3.
