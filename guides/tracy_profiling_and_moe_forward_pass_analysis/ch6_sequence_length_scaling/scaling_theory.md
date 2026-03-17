# Scaling Theory

This file explains why sequence length is the primary independent variable for MoE scaling
experiments, derives the expected scaling law for each op class in the MoE forward pass, and
provides worked examples using the Qwen 235B / DeepSeek-V3 model configuration.

---

## Why Sequence Length Is the Primary Variable

The MoE forward pass takes as input a hidden state tensor of shape `[seq_len, d_model]`.
Sequence length (`seq_len`) is the only dimension that changes across different inference
workloads while keeping the model architecture (d_model, d_ff, num_experts, top_k) fixed.
At inference time:

- **Batch size is typically 1** in latency-critical serving scenarios, so batch does not
  contribute additional parallelism or cost.
- **`num_experts` and `top_k` are architecture constants.** The number of active tokens per
  expert (`expert_capacity ≈ seq_len × top_k / num_experts`) scales directly with `seq_len`.
- **`d_model` and `d_ff` are fixed** by the model checkpoint. Varying them would require
  re-initializing the model and is not a valid sweep variable for a production profiling
  investigation.

The consequence is that every operation in the MoE forward pass has its cost determined
primarily by how many tokens flow through it — which is exactly `seq_len` (or a fixed
multiple of it, via `top_k`). This makes `seq_len` the natural independent variable for
understanding performance scaling.

---

## Expected Scaling by Op Class

### Matrix Multiplications

Matrix multiplications dominate MoE compute time. Each expert FFN applies two linear
projections: `gate_proj` and `up_proj` (combined as a fused matmul in practice) followed by
`down_proj`. The matmul shape is `[expert_capacity, d_model] × [d_model, d_ff]`.

Since `expert_capacity ≈ seq_len × top_k / num_experts`, the M-dimension of the matmul
scales linearly with `seq_len`.

**Memory-bound regime** (small `seq_len`): When the M-dimension is small — specifically when
`expert_capacity / 32 < num_tensix_cores`, where `expert_capacity ≈ seq_len × top_k / num_experts` —
the matmul is memory-bound. Each Tensix core processes one or a few tiles of the output, and
the bottleneck is loading weight tiles from DRAM (~300 GB/s on Wormhole B0). In this regime,
latency scales as **O(seq_len)** because processing twice as many token tiles doubles DRAM
reads.

> **Note:** For Qwen 235B / DeepSeek-V3 (top_k=8, num_experts=128), `expert_capacity ≈ seq_len / 16`.
> The memory-bound condition becomes `(seq_len / 16) / 32 < 80`, i.e. `seq_len < 40,960` —
> meaning essentially all practical decode and prefill sequence lengths are in the memory-bound
> regime on T3K.

**Compute-bound regime** (large `seq_len`): When `seq_len` is large enough that the grid
is fully occupied and the roofline is determined by FLOPs rather than bandwidth, latency
becomes flat or grows slowly. The transition point for Wormhole B0 with 80 Tensix cores is
reached when `expert_capacity ≥ 80 × 32 = 2560`, i.e. `seq_len ≥ 2560 × 16 = 40,960` for
the Qwen config. Below this threshold, assume memory-bound and expect linear scaling.

> **Tip:** When `expert_capacity` is below 2,560 tokens (i.e., `seq_len` below ~40,960 for Qwen with top_k=8, num_experts=128) and the matmul appears to scale sub-linearly, the
> likely explanation is a tile-count boundary: not all 80 Tensix cores are utilized at small
> `seq_len`, so the cost is determined by a fixed kernel dispatch overhead plus a small
> variable compute cost.

### Softmax and TopK

The router computes a softmax over `[seq_len, num_experts]` logits and a topk over the
same shape to select the `top_k` active experts per token. Both operations are strictly
**element-wise along the `seq_len` dimension**, so their latency scales as **O(seq_len)**
in all regimes. There is no compute-bound transition — throughput is always limited by
memory reads of the logit tensor.

For the Qwen 235B / DeepSeek-V3 configuration with 128 experts:
- Softmax input shape: `[seq_len, 128]`, BF16, `= seq_len × 256 bytes`
- TopK: same shape, output is `[seq_len, 8]` indices and weights

At `seq_len=1024`: input is `1024 × 256 = 256 KB`, well within L2 cache. TopK is fast.
At `seq_len=4096`: input is `4096 × 256 = 1 MB`; DRAM bandwidth begins to matter.

### CCL All-to-All

On T3K (8-chip mesh), the dispatch phase must redistribute token embeddings from the chip
that owns each token to the chip that holds each active expert shard. This is an all-to-all
collective communication over Wormhole ethernet links.

**Message size formula.** The total volume of data transferred for one all-to-all is:

```
total_bytes = num_active_tokens × d_model × bytes_per_element
            = seq_len × top_k × d_model × 2
```

Per-chip message size (what each chip sends to one other chip, averaged):

```
per_chip_bytes = total_bytes / num_chips
               = (seq_len × top_k × d_model × 2) / num_chips
```

See Chapter 5, `gap_attribution.md`, Method 3 for the Python calculation confirming ~2.1 ms at seq_len=1024.

**Scaling law.** Because `total_bytes` is proportional to `seq_len`, CCL latency scales as
**O(seq_len)** — linear in sequence length. The per-chip message also scales linearly
because `num_chips` is a constant fixed by the hardware configuration.

> **Note on the /num_chips factor.** The `/num_chips` term in the per-chip message formula
> reflects the fact that each chip only needs to send 1/num_chips of the total data to each
> peer. This is a per-chip sizing detail — a constant determined by the fixed hardware
> topology. The CCL *gap* observed in Tracy scales with the total message size:
> `O(num_active_tokens × d_model)`. Since `d_model` and `num_chips` are both constants in a
> fixed-topology experiment, the gap scales linearly with `seq_len`, but the correct
> expression of the scaling law does not include `/num_chips`.

**Contrast with the 16ms gap.** At `seq_len=1024`, the estimated CCL latency is ~2.1ms.
The observed gap is ~16ms. This large discrepancy confirms the hypothesis from Chapter 5
(`gap_attribution.md`) that the 16ms gap cannot be explained by CCL alone, and that a
synchronization barrier or host-side overhead must be the dominant contributor.

The scaling experiment is designed to distinguish these contributions: CCL grows linearly
with `seq_len`, while a synchronization barrier stays constant. If the measured gap at
`seq_len=64` is already ~14ms but at `seq_len=4096` is ~18ms, the constant term (~14ms)
dominates, and the linear CCL contribution (~4ms increment) is secondary.

### Host-Side Python Operations

Host-side Python code that executes between TTNN op dispatches contributes to gap duration
depending on its complexity:

| Host Operation | Complexity | Example |
|---|---|---|
| Token index construction via Python loop | O(seq_len) | Iterating over expert assignments to build gather indices |
| Tensor-ized index construction | O(1) dispatch + device compute | `ttnn.topk` + `ttnn.argsort` without Python loops |
| Shape calculation and TTNN config construction | O(1) | Calling `ttnn.ShardMemoryConfig(...)` once per forward pass |
| Logging or assertion checks | O(seq_len) if iterating tokens | `for i in range(seq_len): assert ...` |

If a Python loop over token assignments is responsible for part of the gap, it will show up
as a linear-scaling component in the `gap vs. seq_len` plot. Replacing the loop with a
tensor-ized TTNN op removes the linear component and leaves only any remaining constant
overhead.

### Device Synchronization Barriers

`ttnn.synchronize_device(device)` blocks the host thread until all previously enqueued
kernels have completed on the device. Its duration is determined by the time required to
drain the device queue, which is a property of the last enqueued kernel — not of the current
`seq_len` in most cases.

**Scaling:** O(1) — constant in `seq_len`.

**Exception:** If the barrier follows a kernel whose duration itself scales with `seq_len`
(e.g., a memory-bound matmul), then the barrier's observed duration will also scale with
`seq_len` because it is waiting for that kernel to finish. In this case the scaling is an
artifact of the kernel being waited on, not the barrier itself. To distinguish: look at
whether the device profiler shows the device actively executing during the "barrier gap"
(kernel running → kernel scaling) or idle (barrier waiting for a fixed-cost operation).

---

## Summary Table

| Op Class | Scaling | Log-Log Slope | Notes |
|---|---|---|---|
| Matmul (memory-bound) | O(seq_len) | ~1.0 | Regime: expert_capacity < 2,560 tokens (seq_len < ~40,960 for Qwen top_k=8, num_experts=128) |
| Matmul (compute-bound) | O(1) | ~0 | Regime: expert_capacity ≥ 2,560 tokens (seq_len ≥ ~40,960 for Qwen); rarely reached in decode |
| Softmax | O(seq_len) | ~1.0 | Always memory/bandwidth limited |
| TopK | O(seq_len) | ~1.0 | Scales with logit tensor size |
| CCL all-to-all | O(seq_len) | ~1.0 | Linear in `num_active_tokens × d_model × 2` (total); `/num_chips` is per-chip sizing, not part of scaling law |
| Host Python loop | O(seq_len) | ~1.0 | If iterating over token assignments |
| Host Python (tensor-ized) | O(1) | ~0 | All index ops are TTNN tensor ops |
| Synchronization barrier | O(1) | ~0 | Independent of seq_len unless following a scaling kernel |
| Program cache miss (Pattern D) | O(1) | ~0 | Fixed compilation cost; only on first call |

---

## Next Steps

Proceed to [`experiment_design.md`](./experiment_design.md) to learn how to design and
automate the controlled `seq_len` sweep that will empirically confirm or refute the scaling
predictions made in this file.
