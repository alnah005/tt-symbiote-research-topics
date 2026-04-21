# Qwen3.6 MoE Architecture

## Configuration

The MoE FFN block appears in every one of the 40 decoder layers — after every Gated DeltaNet layer and after every Gated Attention layer. The configuration fields from `config.json` that govern this block are:

| Field | Value |
|---|---|
| `num_experts` | 256 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 512 |
| `hidden_size` | 2048 |
| `num_hidden_layers` | 40 |

There are 256 routed experts and 1 shared expert, giving 257 experts per layer in total.

---

## Expert Architecture

Each expert — routed or shared — is a SwiGLU feed-forward network. SwiGLU uses three weight matrices:

- $W_{\text{gate}} \in \mathbb{R}^{2048 \times 512}$: gate projection
- $W_{\text{up}} \in \mathbb{R}^{2048 \times 512}$: up projection
- $W_{\text{down}} \in \mathbb{R}^{512 \times 2048}$: down projection

For a hidden state $x \in \mathbb{R}^{2048}$, the expert output is:

$$y = W_{\text{down}} \left( \text{SiLU}(x W_{\text{gate}}^T) \odot (x W_{\text{up}}^T) \right)$$

The intermediate activations have shape $[512]$ at the bottleneck. Input and output are both in $\mathbb{R}^{2048}$, matching the model hidden dimension.

### Per-Expert Parameter Count

Each expert has exactly 3 matrices:

$$N_{\text{expert}} = 3 \times 2048 \times 512 = 3{,}145{,}728 \approx 3.1\text{M parameters}$$

### Shared Expert

The shared expert has the same SwiGLU architecture with `shared_expert_intermediate_size=512` — identical dimensions to the routed experts. It is always active: every token passes through the shared expert regardless of which routed experts are selected. The shared expert contributes a fixed, non-routed component to the FFN output.

---

## Router

The router maps the hidden state to a probability distribution over all 256 routed experts using a learned linear projection:

$$\text{logits} = x W_r^T, \quad W_r \in \mathbb{R}^{256 \times 2048}$$

The router weight matrix $W_r$ has shape [256, 2048]. The routing procedure is:

1. Compute logits: `[B, T, 2048]` → `[B, T, 256]` via linear projection (no bias).
2. Apply softmax over the 256 expert dimension.
3. Select the top-8 experts by softmax probability.
4. Normalize the 8 selected weights so they sum to 1.

The router adds approximately $256 \times 2048 = 524{,}288 \approx 0.5\text{M}$ parameters per layer, which is small relative to the expert weight mass.

---

## Effective Computation Per Token

For each token at each MoE layer, the execution path is:

- **8 routed expert forward passes** (top-8 selection)
- **1 shared expert forward pass** (always active)

Total: **9 expert forward passes per token per layer**.

The final MoE output is:

$$\text{MoE}(x) = \sum_{i \in \text{top-8}} \hat{w}_i \cdot E_i(x) + E_{\text{shared}}(x)$$

where $\hat{w}_i$ are the normalized routing weights and $E_i$ denotes the $i$-th routed expert's SwiGLU forward pass.

---

## FLOP Analysis

Each expert forward pass involves three matrix multiplications (SwiGLU):

- Gate projection: 2 × 2048 × 512 FLOPs
- Up projection: 2 × 2048 × 512 FLOPs
- Down projection: 2 × 512 × 2048 FLOPs

Total per expert: 3 × 2 × 2048 × 512 = 6,291,456 ≈ 6.3M FLOPs. For 9 active experts per token per layer:

$$\text{FLOPs}_{\text{MoE}} = 9 \times 3 \times 2 \times 2048 \times 512 = 56{,}623{,}104 \approx 56.6\text{M FLOPs per token per layer}$$

(The model plan uses 37.7M, counting gate and up projections jointly as a single matmul: $9 \times 2 \times 2 \times 2048 \times 512 = 37.7\text{M}$. Both figures are internally consistent; 56.6M counts all three weight matrices independently.)

---

## Total and Active Parameter Counts

### Per-Layer Expert Parameters

All 257 experts (256 routed + 1 shared) have the same architecture:

$$N_{\text{layer}} = 257 \times 3 \times 2048 \times 512 = 257 \times 3{,}145{,}728 \approx 808\text{M parameters per layer}$$

(Approximately 805M when rounding 257 × 3.1M.)

### Total Expert Parameters Across All Layers

$$N_{\text{total expert}} = 257 \times 3 \times 2048 \times 512 \times 40 \approx 32.2\text{B parameters}$$

This is the dominant contribution to the model's 35B total parameter count. The remaining parameters come from attention projections, embedding tables, RMSNorm weights, the vision encoder, and the MTP head.

### Active Parameters Per Token

For a single token, 9 of the 257 experts are active (8 routed + 1 shared):

$$\text{Active expert params} = 9 \times 3 \times 2048 \times 512 = 28{,}311{,}552 \approx 28.3\text{M per layer}$$

Across 40 layers, active expert parameters total approximately $28.3\text{M} \times 40 = 1.13\text{B}$. Adding non-expert parameters (attention, embeddings, norms), the total active parameter count per token is approximately **3B**, consistent with the model name "A3B" (3B active).

The fraction of expert parameters activated per token per layer:

$$\frac{9}{257} \approx 3.5\%$$

This sparse activation is the defining characteristic of MoE models: the 35B total parameter budget is accessed at roughly 3% per token, yielding the compute cost of a ~3B parameter model while retaining the capacity of a 35B model.

---

## Load Balancing

Qwen3.6 uses **auxiliary-loss-free load balancing** for expert routing — the same approach adopted in the Qwen3.5 design, which was inspired by DeepSeek-V3. Rather than adding an auxiliary load-balancing loss term to the training objective (which can conflict with the primary language modeling objective and degrade model quality), the load balance is encouraged via a bias-correction mechanism applied to the router logits during training.

This approach:
- Avoids the accuracy-vs-balance tradeoff introduced by auxiliary loss weighting
- Maintains load balance by dynamically adjusting per-expert bias terms based on observed routing frequency
- Has no inference-time overhead: at inference the router logits are used directly without bias correction

---

## TTNN Deployment Implications

The MoE architecture is identical between Qwen3.5 and Qwen3.6 (see Chapter 3). No changes to the TTNN model code are required to support Qwen3.6 weights. The key deployment considerations for the 256-expert, top-8 configuration on Tenstorrent hardware are:

- **Expert batching**: 9 expert forward passes per token require batching strategy to avoid 9 serial small-matmul dispatches. Existing guides cover this in `guides/moe_optimization_techniques_for_ttnn/`.
- **DRAM bandwidth**: loading 9 × 3 expert weight matrices (each 2048×512 or 512×2048) per token per layer is the primary performance bottleneck, not compute. See `guides/ttnn_moe_performance_optimization_on_t3k/`.
- **Expert parallelism**: with 256 experts across 8 T3K devices, 32 experts reside per device. See `guides/expert_parallelism_strategies/` for sharding details.
- **Quantization**: bfp4 expert weight quantization dramatically reduces DRAM pressure for the 30,720 routed expert weight matrices (256 experts × 3 matrices × 40 layers).

---

**Next:** [`cross_model_moe_comparison.md`](./cross_model_moe_comparison.md)
