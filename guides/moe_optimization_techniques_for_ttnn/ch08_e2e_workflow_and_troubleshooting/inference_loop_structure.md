# 8.2 Inference Loop Structure

## 8.2.1 Full Decode Step

A single decode step processes one new token per sequence in the batch. The high-level compute
graph for each step is:

```
token embedding
    ↓
[attention layer]
    ↓
[MoE layer] × (number of MoE layers in model)
    ↓
output projection
    ↓
logits / next-token sample
```

Each MoE layer is the dominant cost at decode time because it involves two all-to-all collectives
over the T3K Ethernet mesh. The sections below describe the MoE layer internals in detail.

## 8.2.2 MoE Layer: Step-by-Step

All steps below assume the decode path with batch size $B$ and capacity $C = \lceil B / 25.6 \rceil$.
The input to the MoE layer is the hidden-state tensor $x$ of shape $[B, H]$ where $H = 7168$.

### Step 1: Router

Compute routing logits and select the top-$k$ experts for each token.

$$g = x \cdot W_r \quad \in \mathbb{R}^{B \times E}$$

$$s = \sigma(g) \quad \text{(element-wise sigmoid)}$$

$$(\text{indices}, \text{scores}) = \operatorname{topk}(s,\ k=8) \quad \in \mathbb{Z}^{B \times 8},\ \mathbb{R}^{B \times 8}$$

Key API calls [VERIFY]:

```python
g = ttnn.matmul(x, W_r)          # [B, H] × [H, E] → [B, E]  [VERIFY]
s = ttnn.sigmoid(g)               # [B, E]  [VERIFY]
indices, scores = ttnn.topk(s, k=8)  # [B, 8] each  [VERIFY]
```

$W_r$ is in L1 on all devices (replicated). Each device independently computes the router over
the full expert set and obtains the same indices and scores. This redundant computation is
preferable to broadcasting the router result because it avoids an additional collective.

### Step 2: Pack the Dispatch Send Buffer

Each device constructs its send buffer of shape $[N, C \times E_d, H]$. Slot $[j, t, :]$ in
this buffer holds the hidden state of a token being sent to device $j$ for expert processing,
where up to $C \times E_d$ tokens can be sent per destination device.

The packing step:

1. For each token $b \in [0, B)$ and each of its $k$ selected experts, determine which device
   owns that expert: $\text{device} = \lfloor \text{expert\_index} / E_d \rfloor$.
2. Place the token's hidden state into the appropriate destination slot in the send buffer.
3. Record the dispatch metadata (token index, expert index, routing score) for use at the combine
   step.

The `per_core_M` parameter for the expert FFN matmul is set from this step:

$$\text{per\_core\_M} = \left\lceil \frac{C}{32} \right\rceil = 1 \quad \text{for all decode } B \text{ (since } C < 32\text{)}$$

### Step 3: All-to-All Dispatch

Scatter the send buffer across all devices so each device receives the tokens destined for its
local experts.

```python
recv_buffer = ttnn.all_to_all(
    send_buffer,
    mesh_device,
    num_links=2,    # [VERIFY] use 2 links for volumes > 1 MB
)
# recv_buffer shape: [C × E_d, H] on each device
```

T3K provides 12.5 GB/s per Ethernet link on a $1 \times 8$ linear mesh [UNVERIFIED]. For $B = 32$,
the dispatch volume per device is:

$$\text{Dispatch volume} = 7 \times 2 \times 32 \times 7168 \times 2 = 6{,}422{,}528 \text{ bytes} \approx 6.4 \text{ MB}$$

At one link this would take approximately 0.51 ms. Using two links halves the effective transfer
time for volumes exceeding 1 MB.

### Step 4: Expert FFN on Received Tokens

Each device runs the FFN of each of its 32 local experts on the tokens it received. Only experts
whose tokens were dispatched to this device execute computation; all others are idle (sparsity
ratio $\rho = k/E = 3.1\%$).

```python
# For each local expert e in [0, E_d):
expert_out = ttnn.matmul(
    recv_tokens_for_expert_e,   # [C, H]
    expert_weight[e],           # [H, D] or [D, H] depending on layout  [VERIFY]
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    # ... program_config with per_core_M=1 at decode
)
```

Expert weights are read from DRAM for each matmul. The read volume per expert call is
$6HD$ bytes [D UNVERIFIED], which at DRAM bandwidth [UNVERIFIED] determines the minimum expert
FFN latency per decode step.

### Step 5: All-to-All Combine

Return processed tokens to their originating devices.

```python
combined = ttnn.all_to_all(
    expert_out_buffer,
    mesh_device,
    num_links=2,    # [VERIFY]
)
# combined shape: [B, k, H] or equivalent flattened form on each device
```

The combine all-to-all has the same volume and latency characteristics as the dispatch all-to-all
(Step 3).

### Step 6: Renormalize Weights and Weighted Accumulation

The routing scores from Step 1 must be renormalized before the weighted sum. Normalization divides
each token's $k$ scores by their sum (not their maximum):

$$\hat{s}_i = \frac{s_i}{\sum_{j=1}^{k} s_j} \quad \forall i \in [1, k]$$

Dividing by the maximum is a common implementation error that produces incorrect output
distributions. See Section 8.3 (error 3) for the consequences.

The final output for each token is:

$$y = \sum_{i=1}^{k} \hat{s}_i \cdot \text{expert\_out}_i$$

### Step 7: Residual Add

Add the MoE layer output back to the layer input:

$$\text{output} = y + x$$

This is a standard elementwise add over $[B, H]$.

## 8.2.3 Memory Config Switching Between Decode and Prefill

The decode and prefill paths use different memory configurations for activations.

| Tensor | Decode memory config | Prefill memory config | Reason |
|---|---|---|---|
| Router input $x$ | L1 | DRAM | Prefill activations $[S, H]$ too large for L1 |
| Routing logits $g$ | L1 | DRAM | Same reason |
| Dispatch send buffer | L1 | DRAM | Same reason |
| Recv buffer after dispatch | L1 | DRAM | Same reason |
| Expert FFN output | L1 | DRAM | Same reason |
| Expert weights | DRAM | DRAM | Always DRAM; too large for L1 regardless of path |
| $W_r$ | L1 | L1 | Small enough to remain in L1 on both paths |

At decode time, $[B, H]$ activations with $B \leq 32$ occupy at most $32 \times 7168 \times 2 =
458{,}752$ bytes $\approx 0.46$ MB, well within the 120 MB aggregate L1 budget. At prefill time,
sequence lengths $S$ can be in the thousands, making $[S, H]$ activations tens of megabytes or
more; DRAM is the only viable option.

The memory config must be switched explicitly when transitioning between prefill and decode phases.
Failing to switch results in either an allocation error (DRAM tensor passed to an L1-configured
operation) or degraded throughput (L1 config applied to a large prefill tensor that overflows into
spill).

## 8.2.4 Prefill Path Notes

The prefill path follows the same logical sequence as decode (Steps 1–7 above) with the following
differences:

- All activation tensors use `ttnn.DRAM_MEMORY_CONFIG`.
- The router input has shape $[S, H]$ instead of $[B, H]$.
- Capacity $C$ is computed over the sequence dimension rather than the batch dimension; for long
  sequences, $C$ can be much larger than 2 and `per_core_M` must be recomputed accordingly.
- The all-to-all dispatch and combine volumes scale with $S$, and link saturation is more likely
  for long sequences.

## 8.2.5 Full API Reference for MoE Layer (Key Calls)

All API names below are [VERIFY].

| Operation | TTNN call | Input shape | Output shape |
|---|---|---|---|
| Router matmul | `ttnn.matmul` | $[B, H] \times [H, E]$ | $[B, E]$ |
| Sigmoid activation | `ttnn.sigmoid` | $[B, E]$ | $[B, E]$ |
| Top-k selection | `ttnn.topk` | $[B, E]$, $k=8$ | $[B, 8]$ indices + scores |
| Dispatch collective | `ttnn.all_to_all` | $[N, C \times E_d, H]$ | $[C \times E_d, H]$ per device |
| Expert FFN | `ttnn.matmul` | $[C, H] \times [H, D]$ | $[C, D]$ (and back) |
| Combine collective | `ttnn.all_to_all` | $[C \times E_d, H]$ per device | $[B, k, H]$ |
| Residual add | `ttnn.add` | $[B, H]$ | $[B, H]$ |
