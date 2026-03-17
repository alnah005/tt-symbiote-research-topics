# Router Kernel Fusion

The Mixture-of-Experts (MoE) router sits on the critical path of every MoE layer: no token can be dispatched to its assigned experts until routing is complete. Reducing the latency of the router sub-graph — from the moment a token embedding arrives to the moment the dispatch metadata is ready — directly reduces all-to-all initiation latency and, by extension, total MoE layer latency.

This file catalogs the kernel fusion opportunities available in the Qwen3.5-35B router on Tenstorrent T3K hardware, analyzes the double-buffering strategy for latency hiding, and evaluates INT8 quantization of the router weight matrix $W_r$.

Throughout this file, the following constants apply:

| Symbol | Value |
|---|---|
| $H$ | 7168 (hidden dimension) |
| $E$ | 256 (number of experts) |
| $k$ | 8 (top-k selection) |
| $N$ | 8 (T3K devices) |
| $B$ | token batch size |

---

## Section 1: The Critical Path

The router computation forms an unbroken dependency chain from token embedding input to dispatch metadata output. No operation in this chain can be skipped or reordered.

**Latency sequence:**

```
[receive token embedding]
       ↓
[router projection: g = xW_r]         ← matmul [B,H] × [H,256] → [B,256]
       ↓
[sigmoid activation: s = σ(g)]        ← element-wise, [B,256]
       ↓
[top-k selection]                      ← indices [B,8], scores [B,8]
       ↓
[scatter / pack dispatch metadata]    ← per-device token counts, offsets
       ↓
[dispatch all-to-all can begin]
```

Each step has non-trivial overhead:

| Step | Primary cost |
|---|---|
| Router projection | Matmul FLOPs: $2 \times B \times H \times E = 2 \times B \times 7168 \times 256$ |
| Sigmoid | Element-wise exp over $B \times 256$ values |
| Top-k | $O(B \times (E + k \log k)) = O(B \times 280)$ comparisons |
| Scatter prep | Iterate over $B \times k$ (index, device) pairs; compute per-device counts |

Without fusion, each step requires a separate kernel launch, writing an intermediate result to L1 and reading it back for the next kernel. The kernel launch boundary overhead and L1 round-trips accumulate across four sequential steps.

**Goal:** Reduce the total wall-clock time from token embedding arrival to dispatch metadata ready, by fusing steps and eliminating unnecessary memory traffic.

---

## Section 2: Fusion 1 — Projection + Top-k + Index Extraction

### Baseline (Unfused)

```python
# Step 1: Full matmul — writes [B, 256] to L1
g = ttnn.matmul(x, W_r)                    # [B, 7168] × [7168, 256] → [B, 256]

# Step 2: Sigmoid — reads [B, 256], writes [B, 256] to L1
s = ttnn.sigmoid(g)                         # [B, 256]

# Step 3: Top-k — reads [B, 256], writes [B, 8] indices and [B, 8] scores
indices, scores = ttnn.topk(s, k=8, dim=1)  # [B, 8] each
```

This involves two full read-write cycles of the $[B, 256]$ intermediate tensor.

**Intermediate tensor size:** $B \times 256 \times 2$ bytes. At $B = 32$: $32 \times 256 \times 2 = 16{,}384$ bytes = 16 KB.

### Fused Kernel

The fused kernel computes the matmul tile-by-tile and maintains a running min-heap of size $k$ in local registers:

```python
def fused_project_topk(
    x: "ttnn.Tensor",    # [B, H] = [B, 7168], BF16
    W_r: "ttnn.Tensor",  # [H, E] = [7168, 256], BF16
    k: int = 8,
) -> tuple:
    """
    Fused kernel: linear projection + sigmoid + top-k + index extraction.

    Processes W_r in column tiles of 32 experts. As each tile's output
    logits are computed, sigmoid is applied tile-locally and the tile's
    32 values are compared against the running min-heap of size k.

    After all 8 column tiles are processed, the heap contains the global
    top-k indices and raw sigmoid scores.

    Returns:
        indices: [B, k] = [B, 8], int32 — expert indices for each token
        scores:  [B, k] = [B, 8], BF16 — raw sigmoid scores (unnormalized)
    """
    # Pseudocode for the tile-sequential fused kernel:
    #
    # heap = MinHeap(capacity=k)  # per token row
    # for tile_col in range(E // 32):           # 8 tiles
    #     col_start = tile_col * 32
    #     col_end   = col_start + 32
    #     g_tile = x @ W_r[:, col_start:col_end]  # [B, 32]
    #     s_tile = sigmoid(g_tile)                 # [B, 32], element-wise
    #     for e_local in range(32):
    #         e_global = col_start + e_local
    #         heap.try_insert(e_global, s_tile[:, e_local])
    # return heap.indices(), heap.values()

    # In TTNN: use ttnn.fused_router_topk if available, or compose via
    # the kernel compiler's tile-pipeline API.
    pass
```

**Savings from Fusion 1:**

- Eliminates writing and reading the full $[B, 256]$ logit tensor: saves one L1 read-write cycle of 16 KB (at $B = 32$).
- Eliminates two kernel launch boundaries (between matmul→sigmoid and sigmoid→topk).
- More importantly: reduces the latency from "matmul starts" to "top-k indices ready" because the heap can be updated as each output tile is produced, pipelining matmul computation with top-k bookkeeping.

> **Warning:** Tile-level fusion of matmul and top-k requires the kernel compiler to support partial output accumulation into the top-k buffer as each output tile completes. Verify TTNN kernel composition capabilities before relying on this optimization. If not available, a two-pass approach (full matmul, then fused sigmoid + topk) still eliminates one kernel boundary.

---

## Section 3: Fusion 2 — Weight Normalization + Scatter Metadata Preparation

After top-$k$ selection, two operations must complete before the dispatch all-to-all can begin:

1. **Scatter count:** For each of the $N = 8$ destination devices, count how many tokens are being sent (sum of tokens whose selected experts reside on that device).
2. **Dispatch buffer packing:** Arrange token embeddings in per-device contiguous buffers.

And, as described in `weight_normalization.md`, the raw sigmoid scores must be retained locally for the combine step (deferred normalization approach).

### Baseline (Unfused)

```
top-k output → kernel 1: compute per-device scatter counts
             → kernel 2: pack dispatch buffer (token reordering)
             → kernel 3: normalize weights [B, k] (deferred, kept local)
```

Three separate kernels, each reading from and writing to L1 intermediate buffers.

### Fused Kernel

```python
def fused_scatter_normalize(
    x: "ttnn.Tensor",         # [B, H] = [B, 7168], token embeddings, BF16
    indices: "ttnn.Tensor",   # [B, k] = [B, 8], expert indices, int32
    scores: "ttnn.Tensor",    # [B, k] = [B, 8], raw sigmoid scores, BF16
    expert_to_device: list,   # mapping: expert_id → device_id (0–7)
    N: int = 8,               # number of devices
    k: int = 8,
) -> tuple:
    """
    Fused kernel: iterate over (token, expert) pairs once to:
      1. Compute per-device scatter counts.
      2. Compute scatter offsets (prefix sum over counts).
      3. Pack token embeddings into per-device contiguous dispatch buffer.
      4. Normalize routing weights locally (kept for combine step).

    Single pass over B × k = 32 × 8 = 256 (index, device) pairs.

    Returns:
        dispatch_buffer: Packed token embeddings per device.
        scatter_metadata: Per-device (count, offset) pairs.
        norm_scores: [B, k] normalized combination weights (local, BF16).
    """
    # Phase 1: single pass over indices to compute scatter counts
    device_counts = [0] * N
    for b in range(B):
        for j in range(k):
            dev = expert_to_device[indices[b, j]]
            device_counts[dev] += 1

    # Phase 2: prefix sum for offsets (O(N) = O(8))
    device_offsets = prefix_sum(device_counts)  # O(8)

    # Phase 3: pack dispatch buffer + normalize weights (same pass over [B, k])
    score_sums = scores.sum(dim=1, keepdim=True)    # [B, 1]
    norm_scores = scores / score_sums               # [B, k], normalized

    dispatch_buffer = scatter_pack(x, indices, expert_to_device, device_offsets)

    return dispatch_buffer, (device_counts, device_offsets), norm_scores
```

**Benefits of Fusion 2:**

- Replaces three separate kernel launches with one.
- Eliminates two sets of L1 intermediate buffers (per-device count array and intermediate score tensor).
- The single pass over $B \times k = 32 \times 8 = 256$ index pairs is cache-friendly.

> **Tip:** The scatter pack operation (reordering token embeddings into per-device contiguous buffers) is the most memory-intensive part of this fused kernel. At $B = 32$ and $H = 7168$, the dispatch buffer is at most $32 \times 7168 \times 2 = 458{,}752$ bytes $\approx 448$ KB. This fits comfortably in the aggregate L1 of a few Tensix cores.

---

## Section 4: Double-Buffering for Latency Hiding

Double-buffering overlaps router computation for one micro-batch with the dispatch, expert compute, and combine operations for the preceding micro-batch. This is only beneficial when the router computation time is comparable to or less than the sum of (dispatch + expert compute + combine) time.

### Without Double-Buffering

```
Route m₀ → Dispatch m₀ → Expert compute m₀ → Combine m₀ → Route m₁ → Dispatch m₁ → ...
```

Total time per micro-batch: $T_{\text{route}} + T_{\text{dispatch}} + T_{\text{expert}} + T_{\text{combine}}$

### With Double-Buffering

```
Route m₀ → Dispatch m₀ → Expert m₀ → Combine m₀
                  ↕ overlap
           Route m₁ → Dispatch m₁ → Expert m₁ → Combine m₁
```

When $T_{\text{route}} \leq T_{\text{dispatch}} + T_{\text{expert}} + T_{\text{combine}}$, the routing of $m_1$ is fully hidden behind the execution of $m_0$. The steady-state throughput becomes limited by the longer of the two pipelines.

### Memory Cost

Double-buffering requires two independent sets of router buffers:

| Buffer | Per micro-batch | Two micro-batches |
|---|---|---|
| Input token embedding | $B/2 \times H \times 2$ bytes | $2 \times (B/2 \times 7168 \times 2)$ |
| Logit tensor $[B/2, 256]$ | $(B/2) \times 512$ bytes | $2 \times (B/2 \times 512)$ |
| Indices $[B/2, k]$ | $(B/2) \times k \times 4$ bytes | $2 \times (B/2 \times 8 \times 4)$ |
| Scores $[B/2, k]$ | $(B/2) \times k \times 2$ bytes | $2 \times (B/2 \times 8 \times 2)$ |

At $B = 32$ (so each micro-batch has $B/2 = 16$ tokens):

- Two logit tensors: $2 \times 16 \times 512 = 16{,}384$ bytes = 16 KB
- Total double-buffer overhead: $\approx 16$ KB for the logit tensors plus the negligible index and score tensors.

This is a trivially small L1 cost.

### Condition for Benefit

At small batch sizes, $T_{\text{route}}$ may exceed the sum of downstream steps if expert compute is very fast (e.g., with highly optimized expert kernels). In that case, double-buffering does not help and adds complexity. At larger $B$ where the all-to-all communication latency and expert compute dominate, double-buffering reliably hides router overhead.

```python
# Pseudocode for double-buffered router pipeline
def double_buffered_moe_forward(micro_batches):
    """
    Overlap routing of micro-batch i+1 with expert compute of micro-batch i.
    """
    # Prime the pipeline with the first micro-batch
    indices_0, scores_0, metadata_0 = router_fused(micro_batches[0])

    for i in range(1, len(micro_batches)):
        # Start routing micro-batch i (asynchronous)
        route_future = async_router_fused(micro_batches[i])

        # Dispatch, compute, and combine micro-batch i-1 (synchronous on this path)
        dispatch_payload = scatter_pack(micro_batches[i-1], indices_0, metadata_0)
        all_to_all_dispatch(dispatch_payload)
        expert_outputs = expert_compute(dispatch_payload)
        all_to_all_combine(expert_outputs)
        y = combine_with_renormalization(expert_outputs, scores_0)

        # Collect routing results for micro-batch i
        indices_0, scores_0, metadata_0 = route_future.get()

    # Drain the last micro-batch
    # ... (same dispatch/compute/combine sequence for final micro-batch)
```

---

## Section 5: INT8 Quantization of $W_r$

### Motivation

The router weight matrix $W_r \in \mathbb{R}^{7168 \times 256}$ in BF16 occupies:

$$7168 \times 256 \times 2 = 3{,}670{,}016 \text{ bytes} \approx 3.67 \text{ MB}$$

This exceeds the per-core L1 budget of 1.5 MB on Wormhole B0. During the router projection matmul, $W_r$ must be staged from DRAM (or a shared L1 region) into computing cores. Reducing $W_r$ to INT8 halves its size:

$$7168 \times 256 \times 1 = 1{,}835{,}008 \text{ bytes} \approx 1.84 \text{ MB}$$

At 1.84 MB, $W_r$ can be held in the aggregate L1 of 2 Tensix cores ($2 \times 1.5 = 3$ MB) with room for input and output buffers.

### Quantization Accuracy

The router's output is categorical: it selects the top-$k = 8$ experts from $E = 256$ by ranking. Small perturbations in the logit values $g_e$ rarely change which experts rank in the top 8, provided there is a natural margin between the 8th and 9th highest scores.

**Why the margin is typically adequate:**

After training with the auxiliary load-balancing loss, the routing distribution assigns clear preferences to a subset of experts for most tokens. The margin between the 8th-ranked and 9th-ranked expert score is typically larger than the quantization error introduced by INT8.

**INT8 quantization error:** For a BF16 value in the range $[-4, 4]$, INT8 quantization with scale $s = 8 / 255 \approx 0.031$ introduces an absolute error of at most $\pm s/2 \approx 0.016$. The resulting logit error is $\leq 0.016$, which maps to a sigmoid output perturbation of at most:

$$\Delta \sigma \approx \sigma'(g) \times 0.016 = \sigma(g)(1 - \sigma(g)) \times 0.016 \leq 0.25 \times 0.016 = 0.004$$

A perturbation of $0.004$ in sigmoid output is smaller than the typical selection margin between neighboring expert scores, so top-$k$ selection is robust to INT8 quantization of $W_r$.

### Implementation

```python
def quantize_router_weights(W_r_bf16: "torch.Tensor") -> tuple:
    """
    Quantize router weight matrix W_r from BF16 to INT8 for TTNN inference.

    W_r shape: [7168, 256]
    BF16 size: 3.67 MB → INT8 size: 1.84 MB

    Returns:
        W_r_int8: Quantized weight matrix, shape [7168, 256], dtype INT8.
        scale:    Per-column scale factors, shape [256], dtype FP32.
        zero_pt:  Per-column zero points, shape [256], dtype INT8.
    """
    import torch

    # Per-column (per-expert) symmetric quantization
    col_max = W_r_bf16.abs().max(dim=0).values.float()  # [256]
    scale = col_max / 127.0                              # [256]
    W_r_int8 = (W_r_bf16.float() / scale.unsqueeze(0)).round().clamp(-127, 127).to(torch.int8)
    zero_pt = torch.zeros(256, dtype=torch.int8)

    return W_r_int8, scale, zero_pt


def router_projection_int8(
    x_bf16: "ttnn.Tensor",      # [B, 7168], BF16
    W_r_int8: "ttnn.Tensor",    # [7168, 256], INT8
    scale: "ttnn.Tensor",       # [256], FP32 per-column dequantization scale
) -> "ttnn.Tensor":
    """
    INT8 matrix multiply for router projection; dequantize output to BF16.

    Args:
        x_bf16:   Input activations [B, H].
        W_r_int8: Quantized router weights [H, E].
        scale:    Per-expert dequantization scale [E].

    Returns:
        g: Router logits [B, E], BF16, ready for sigmoid activation.
    """
    import ttnn

    # INT8 matmul (accumulates in INT32 internally)
    g_int32 = ttnn.matmul(x_bf16, W_r_int8, dtype=ttnn.int32)  # [B, 256]

    # Dequantize: multiply by per-column scale and convert to BF16
    g_bf16 = ttnn.mul(g_int32.to(ttnn.bfloat16), scale)         # [B, 256]
    return g_bf16
```

### Calibration Requirement

Before deploying INT8-quantized $W_r$, validate that the expert selection distribution does not shift significantly:

1. Run inference on a representative calibration set (at least 1,000 tokens per domain).
2. Compare per-expert routing frequency between BF16 and INT8 configurations.
3. Accept if per-expert routing frequency shifts by less than ~1% in absolute terms.
4. If shifts exceed this threshold, apply per-row (per-input-dimension) quantization rather than per-column, or revert to BF16.

> **Tip:** TTNN's INT8 matmul operator dequantizes to BF16 output automatically when the dequantization scale is provided. Verify that the scale tensor is applied per-column (per-expert) rather than globally; a global scale reduces quantization accuracy for expert columns with atypically large or small weight magnitudes.

### Summary of INT8 Benefits

| Metric | BF16 | INT8 |
|---|---|---|
| $W_r$ size | 3.67 MB | 1.84 MB |
| L1 fit (cores needed) | 3 cores minimum | 2 cores minimum |
| Accuracy impact on routing | Baseline | $< 0.004$ sigmoid perturbation; negligible rank changes |
| Memory bandwidth for $W_r$ load | $2\times$ INT8 | $1\times$ |

---

## References

- Tenstorrent, *TTNN Developer Guide*, 2024. (Kernel fusion API, INT8 matmul support, tile pipeline.)
- Wormhole B0 Architecture Reference, Tenstorrent, 2024.
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," NeurIPS 2022.
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," ICLR 2021.
- Chapter 2 of this guide: `ch02_all_to_all_primitives/all_to_all_dispatch.md`, `ch02_all_to_all_primitives/all_to_all_combine.md`.
- Chapter 5 of this guide: `router_forward_pass.md`, `topk_selection_efficiency.md`, `weight_normalization.md`.

---

**Next:** [Chapter 6 — Fused Dispatch Compute Combine](../ch06_fused_dispatch_compute_combine/index.md)
