# Routing and Sparsity

## Token-to-Expert Assignment

The router (gating network) is responsible for mapping each token to a set of experts. Understanding its output format precisely is important because the router's outputs directly determine the shapes of all downstream tensor operations.

### Router Architecture

The router is typically a single linear layer with no bias:

```python
# Router: maps token hidden states to expert logits
# in_features = d_model, out_features = num_experts
router_weight = torch.nn.Linear(d_model, num_experts, bias=False)

# x: [batch, seq, d_model]
# router_logits: [batch, seq, num_experts]
router_logits = router_weight(x)
```

The logits are then converted to a routing distribution. The exact normalization varies by model:

- **Switch Transformer**: softmax over all experts, then take top-1.
- **Mixtral**: softmax over all experts, then take top-2 and re-normalize the selected scores.
- **DeepSeek-V2**: computes per-expert gate scores using a sigmoid function (not softmax). Unlike softmax, sigmoid scores are computed independently per expert — each expert's gate value depends only on the token's dot product with that expert's key vector, with no mutual competition between experts. Post-selection renormalization then forces the top-k selected scores to sum to 1, but the pre-selection gate values do not form a joint distribution over all experts (arXiv:2405.04434).

For the purposes of this guide, we use the Mixtral-style formulation: softmax followed by top-K selection and re-normalization. The mechanics generalize to other variants.

### Router Outputs

After top-K selection, the router produces two tensors per token:

- **Expert indices**: `[batch, seq, top_k]`, dtype `int64`. Entry `[b, s, k]` is the index (in `[0, num_experts)`) of the `k`-th selected expert for token `(b, s)`.
- **Expert scores**: `[batch, seq, top_k]`, dtype `float32` (or `bfloat16`). Entry `[b, s, k]` is the weight applied to the `k`-th expert's output when combining results. These sum to 1.0 only after re-normalization to the selected top-k set (see `compute_routing` below); the raw softmax probabilities over all experts sum to 1.0, but the top-k selection does not. In exact arithmetic the re-normalized scores sum to 1.0; in bfloat16, the sum may differ from 1.0 by a small rounding error.

```python
import torch

def compute_routing(x, router_weight, top_k, num_experts):
    # x: [batch, seq, d_model]
    batch, seq, d_model = x.shape

    # Flatten to [batch * seq, d_model] for router
    flat_x = x.view(-1, d_model)                              # [T, d_model], T = batch * seq

    # Produce logits and softmax
    logits = router_weight(flat_x)                            # [T, num_experts]
    probs = torch.softmax(logits.float(), dim=-1)             # [T, num_experts]

    # Select top-K experts
    scores, indices = torch.topk(probs, top_k, dim=-1)        # [T, top_k], [T, top_k]

    # Re-normalize selected scores
    scores = scores / scores.sum(dim=-1, keepdim=True)        # [T, top_k]

    return scores.view(batch, seq, top_k), indices.view(batch, seq, top_k)
```

The flat token dimension `T = batch * seq` is important. In practice, routing operates on all tokens in the batch simultaneously. This is the dimension over which expert load is measured.

---

## Expert Capacity and Capacity Factor

### The Expert Capacity Problem

If we were to let the router send any number of tokens to each expert, the expert input shapes would be data-dependent and different on every forward pass. This creates two problems:

1. **Static compilation**: Hardware kernels compiled for a specific input shape cannot be reused if the shape changes. TTNN's program cache (see Chapter 2) requires static shapes for efficient execution.
2. **Memory allocation**: GPU/accelerator memory must be allocated ahead of time. Variable-size allocations require either dynamic allocation (slow) or worst-case pre-allocation (wasteful).

The standard solution is to fix an **expert_capacity** — a maximum number of tokens that each expert will process in a single forward pass. This makes all expert input shapes static.

### Capacity Factor Definition

Expert capacity is typically specified via a **capacity factor** `C`:

```
expert_capacity = C * T * top_k / num_experts
```

where `T = batch * seq` is the total number of tokens and `top_k` is the number of experts each token is assigned to.

The term `T × top_k / num_experts` is the **expected number of tokens per expert** under uniform routing. If `top_k = 2` and `num_experts = 8`, each expert is expected to receive `2/8 = 0.25 × T tokens (25% of the total T = batch × seq tokens)`. Multiplying by `C` scales this up: `C = 1.0` gives exactly the expected load, `C = 1.25` adds a 25% buffer above the expected load.

> **Warning:** Setting CF=1.0 (C=⌈T×top_k/num_experts⌉) is mathematically optimal only for perfectly uniform routing. In practice, routing exhibits natural load imbalance across input distributions, making CF=1.0 insufficient even for well-trained routers — dropped tokens will occur. A minimum CF≥1.25 is recommended for training stability; see Chapter 6 for production CF selection.

```python
import math

def compute_expert_capacity(batch, seq, num_experts, top_k, capacity_factor):
    T = batch * seq
    # Raw tokens per expert slot, accounting for top_k over-subscription
    raw_capacity = (T * top_k) / num_experts
    capacity = math.ceil(capacity_factor * raw_capacity)  # ceil to nearest integer; tile alignment applied next
    # Always a multiple of the tile size for hardware alignment
    tile_size = 32
    capacity = ((capacity + tile_size - 1) // tile_size) * tile_size
    return capacity
```

> **Tip:** Always round `expert_capacity` up to the nearest multiple of 32 (one tile row). Non-tile-aligned capacities require padding to tile boundaries in TTNN's tile layout and can cause silent shape errors. See Chapter 5 for the full treatment.

### Impact on Tensor Shapes

With a fixed `expert_capacity`, the expert input and output tensors become:

- **Per-expert input batch**: `[expert_capacity, d_model]` — gathered tokens for one expert, padded to capacity.
- **Batched expert tensor**: `[num_experts, expert_capacity, d_model]` — all expert inputs stacked.
- **Expert weight matrix**: `[d_model, d_ff]` per expert, or `[num_experts, d_model, d_ff]` batched.
- **Expert computation output (after W1 projection)**: `[num_experts, expert_capacity, d_ff]` — this intermediate tensor is the reason large `d_ff` values drive memory cost.
- **Expert output (after W2 projection, ready for combine)**: `[num_experts, expert_capacity, d_model]` — result that is scattered back to token positions in the combine step.

These static shapes are what make efficient batched matmul possible (Chapter 3). The cost is padding: if an expert receives only 10 tokens but `expert_capacity = 64`, 54 token slots are zero-padded and compute wasted on them.

---

## Load Balancing Losses

Without any explicit regularization, a router will often converge to a degenerate solution where it sends most tokens to a small number of popular experts and starves the rest. This is called **expert collapse** or **routing imbalance**. It is bad for two reasons:

1. **Model quality**: Underutilized experts learn little during training; the effective model capacity shrinks.
2. **Hardware efficiency**: If most tokens go to one expert, that expert's compute becomes the bottleneck, while all other experts waste cycles on empty or near-empty batches.

Most MoE models address this with an auxiliary **load balancing loss** added to the training objective.

### Switch Transformer Load Balancing

Switch Transformer uses:

```
load_balance_loss = num_experts * sum_over_experts(f_i * P_i)
```

where:
- `f_i` = fraction of tokens dispatched to expert `i` in the current batch (`f_i = count_i / T`)
- `P_i` = fraction of router probability mass assigned to expert `i` across the batch (`P_i = mean of softmax score for expert i`)

> **Note:** The full expression includes a scalar coefficient `alpha` (typically `1e-2`) that controls the strength of the penalty; it is omitted here for brevity.

When routing is perfectly uniform, `f_i = P_i = 1/num_experts` for all `i`, and the loss equals 1.0 (before alpha scaling; see the note about alpha above). The loss is minimized when both the dispatch fraction and the probability mass are evenly spread.

### Mixtral / Top-K Variants

Mixtral and similar models typically use a variant where the loss penalizes the variance in per-expert load. The specific formulation is model-dependent, but the goal is the same: encourage the router to distribute tokens roughly uniformly so that `expert_capacity` is an efficient bound for all experts.

> **Warning:** Load balancing losses affect training behavior. At inference time, the router weights are fixed, and actual expert load depends on the input data. Even a well-trained model will show load imbalance on specific input distributions (e.g., certain languages, domains, or token types). Plan for this when setting `expert_capacity` and capacity factor for inference deployments.

---

## Dropped Tokens

When more than `expert_capacity` tokens are routed to an expert, the excess tokens are **dropped** for that expert slot. This guide adopts the **zero-contribution convention**, consistent with Switch Transformers and TTNN's zero-padded dispatch buffers: the dropped expert slot contributes a **zero vector** to the weighted combination, so the token's final output is `moe_output(t) = sum_of_non_dropped_expert_outputs(t)`. The token is **not discarded from the forward pass** — its residual stream carries the input `x(t)` unchanged, because the residual addition (`x(t) + moe_output(t)`) is performed by the transformer block, not the MoE layer. The "pass through unchanged" phrasing sometimes seen elsewhere conflates the residual stream (which carries the token unchanged) with the MoE expert contribution (which is zero for the dropped slot); these are distinct. The zero-contribution convention is the only behavior assumed by Chapters 3, 5, and 8; Chapter 8's correctness validation (`correctness_validation.md`) treats deviations from it as a source of PCC degradation.

Token dropping is a correctness-quality trade-off made explicit by the capacity factor:

- **`C = 1.0`**: Expected load exactly; any deviation above the mean causes drops. In practice, natural variance in per-expert load across different input distributions means some experts may be consistently over- or under-utilized.
- **`C = 1.25`**: 25% buffer reduces drop rate significantly but wastes 25% of FLOPs to padding.
- **`C = 2.0`**: Very conservative; drops become rare but FLOP waste is high.

For inference, some frameworks disable capacity limits entirely (Mixtral inference often works this way) and instead use variable-length expert batches. This avoids dropped tokens but requires a different execution strategy. Chapter 3 covers the fixed-capacity (padded) approach, and Chapter 4 covers fixed-capacity tensors with sparsity masks.

```python
def dispatch_with_capacity(flat_x, indices, scores, num_experts, expert_capacity, top_k):
    # flat_x: [T, d_model]
    # indices: [T, top_k]
    # scores: [T, top_k]
    T, d_model = flat_x.shape

    # expert_inputs[e] will hold tokens assigned to expert e, padded to capacity
    expert_inputs = torch.zeros(num_experts, expert_capacity, d_model)
    expert_scores = torch.zeros(num_experts, expert_capacity)
    # Track how many tokens each expert has received
    slot_counts = torch.zeros(num_experts, dtype=torch.long)
    # reverse_mapping[e, s] = original token index for expert e's slot s, or -1 if empty.
    # This is required to implement the combine step: after expert FFNs run, each expert's
    # output slot must be scattered back to the correct token position in the output tensor.
    # Without this mapping, the combine step cannot determine where to write expert e's slot s output.
    reverse_mapping = torch.full((num_experts, expert_capacity), -1, dtype=torch.long)

    for token_idx in range(T):
        for k in range(top_k):
            expert_id = indices[token_idx, k].item()
            slot = slot_counts[expert_id].item()
            if slot < expert_capacity:
                # Token fits within capacity
                expert_inputs[expert_id, slot] = flat_x[token_idx]
                expert_scores[expert_id, slot] = scores[token_idx, k]
                reverse_mapping[expert_id, slot] = token_idx  # record original token index
                slot_counts[expert_id] += 1
            # else: token is dropped (no assignment, no gradient through this expert)

    # Note: the combine step requires the reverse mapping from (expert_id, slot_index) → token_idx.
    # reverse_mapping is constructed above as the inverse of the dispatch_buf assignment;
    # see Chapter 3 for the full combine implementation using this tensor.
    return expert_inputs, expert_scores, slot_counts, reverse_mapping
```

---

## Routing Decisions and Downstream Compute Shape

The routing step is computationally trivial compared to the expert FFNs, but its output has a large structural impact:

| Router output | Shape | Determines |
|---|---|---|
| `expert_indices` | `[T, top_k]` | Which experts receive non-zero input |
| `expert_scores` | `[T, top_k]` | The combine weights for output aggregation |
| `slot_counts` | `[num_experts]` | Actual utilization of each expert's capacity |

The **sparsity pattern** of the MoE layer is determined by `slot_counts`: for experts where `slot_counts[e] == 0`, the entire expert computation produces a zero contribution. For experts where `slot_counts[e] < expert_capacity`, only the first `slot_counts[e]` rows of the expert input matrix are non-zero.

This pattern is not known until runtime (it depends on the input data and the learned router weights), which is the root cause of the hardware efficiency challenges described in `moe_on_hardware.md`.

---

## Sparsity Pattern and Sparsity Ratio

### Sparsity Pattern

The **sparsity pattern** of a forward pass is the set of `(expert_id, token_slot)` pairs that contain non-zero data. Conceptually, the sparsity pattern is a binary mask derived from routing decisions that indicates which portions of the expert activation matrices are populated versus zero-padded.

The **sparsity tensor** is the TTNN data structure that encodes this pattern; its exact format and tile-level encoding details (including granularity and block structure) are defined in Chapter 5, `sparsity_tensor_format.md`.

### Sparsity Ratio

The **sparsity ratio** is the fraction of expert slots (after capacity padding) that receive zero tokens:

```
sparsity_ratio = 1 - (sum(slot_counts) / (num_experts * expert_capacity))
```

Since `sum(slot_counts) = T * top_k` (assuming no dropped tokens — when tokens are dropped, `sum(slot_counts) < T * top_k`; see Note below), under the no-drop condition, this simplifies to:

```
sparsity_ratio = 1 - (T * top_k) / (num_experts * expert_capacity)
```

Substituting `expert_capacity = C * T * top_k / num_experts` (at uniform load; using the pre-rounding formula for the derivation):

```
= 1 - (T * top_k) / (num_experts * C * T * top_k / num_experts)
= 1 - (T * top_k * num_experts) / (num_experts * C * T * top_k)
= 1 - 1/C
```

> **Note:** The identity `sparsity_ratio = 1 - 1/C` holds under the no-drop condition (when tokens are dropped, the actual ratio exceeds this value) and only when `expert_capacity` equals the unrounded formula — after tile-ceiling rounding to the nearest multiple of 32, the true sparsity ratio is slightly greater than `1 - 1/C` (typically sub-percent difference). When tokens are dropped (C is small or routing is highly imbalanced), use the general form with `sum(slot_counts)`.

So with `C = 1.25`, the sparsity ratio under the no-drop condition is `1 - 1/1.25 = 0.20` (20% empty slots under perfectly uniform routing). Two distinct cases arise under routing imbalance:

(a) **Imbalance without drops**: The aggregate sparsity ratio remains approximately `1 - 1/C`; routing variance affects per-expert utilization but not the global fill fraction. The tail-latency consequence still applies: the highest-utilization expert sets the bottleneck for that forward pass.

(b) **Imbalance with drops**: If imbalanced routing causes some experts to exceed capacity, tokens are dropped and `sum(slot_counts) < T * top_k`. The aggregate sparsity ratio increases above `1 - 1/C`, as the effective fill falls below the no-drop baseline.

> **Tip:** The `1 - 1/C` formula assumes sufficient tokens for all experts to receive load (i.e., `T * top_k ≥ num_experts`). In decode mode (batch=1, seq=1), `T = 1` and `top_k = 2 ≪ 8 = num_experts`, so the effective regime is extreme sparsity. Using the minimum tile-aligned `expert_capacity = 32`, the decode sparsity ratio is `1 - (T * top_k) / (num_experts * expert_capacity) = 1 - (1 × 2) / (8 × 32) = 1 - 2/256 ≈ 0.992`. This is the regime where `sparse_matmul` provides its largest gains — nearly 99% of expert slots are empty. See Chapter 6 for regime-specific analysis.

---

## Summary

| Concept | Key value / formula |
|---|---|
| `expert_capacity` | `C * T * top_k / num_experts`, rounded up to the nearest multiple of 32 |
| Dropped tokens | Tokens in excess of `expert_capacity` per expert; the overloaded expert slot contributes a zero vector — token still propagates via the residual connection (zero-contribution convention; see Dropped Tokens section) |
| Load balancing loss | Auxiliary training loss to encourage uniform routing; does not affect inference behavior directly |
| Sparsity ratio | General form: `1 - (sum(slot_counts)) / (num_experts * expert_capacity)` (see full derivation above). Simplifies to `1 - (T * top_k) / (num_experts * expert_capacity)` = `1 - 1/C` only under the no-drop condition (no tokens dropped); when tokens are dropped, use the general form with `sum(slot_counts)`. |
| Sparsity pattern | The token-slot granularity boolean map of which expert input slots are non-zero |

---

## Next Steps

Proceed to [`moe_on_hardware.md`](./moe_on_hardware.md) to understand how the structural properties defined in this section (variable load, data-dependent routing, sparse activation) translate into hardware efficiency challenges on Tenstorrent accelerators, and to get a first preview of the two TTNN strategies this guide evaluates.
