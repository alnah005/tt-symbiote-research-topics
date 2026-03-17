# MoE Overview

## What Is a Mixture of Experts Layer?

A Mixture of Experts (MoE) layer is a neural network component that replaces the single feed-forward network (FFN) in a transformer block with a collection of FFN sub-networks called **experts**, plus a lightweight **gating network** (also called a router) that decides, for each input token, which subset of those experts should process it.

The key insight is conditional computation: not every expert runs on every token. A standard dense FFN applies the same computation to every token in the sequence. An MoE layer, by contrast, routes each token to only a small number of experts and runs only those experts for that token. This allows a model to have a very large total parameter count while keeping the per-token FLOPs roughly constant — the model is large in parameter space but sparse in computation space.

The three structural elements of every MoE layer are:

1. **The gating network**: A small learned linear projection (and usually a softmax or similar normalization) that maps each token's hidden representation to a probability distribution or score over all experts.
2. **The expert pool**: A set of `num_experts` independent FFN sub-networks. Each expert has its own weight matrices and is structurally identical to a standard FFN (e.g., two linear layers with a nonlinearity between them, or a gated variant like SwiGLU).
3. **The dispatch-combine pattern**: The two-phase execution structure described in detail below.

---

## The Dispatch-Combine Pattern

Every MoE forward pass follows the same logical sequence:

**Phase 1 — Dispatch (scatter):**
1. The gating network reads the input activation tensor of shape `[batch, seq, d_model]` and produces a score tensor of shape `[batch, seq, num_experts]`.
2. From those scores, the router selects exactly `top_k` experts per token and produces two outputs: expert indices of shape `[batch, seq, top_k]` and expert scores (weights) of shape `[batch, seq, top_k]`.
3. Tokens are gathered and re-organized so that all tokens assigned to expert `i` are placed together, forming a per-expert input batch. This is the **dispatch** step.

**Phase 2 — Expert computation:**
4. Each selected expert runs its FFN forward pass on the tokens assigned to it.

**Phase 3 — Combine (gather):**
5. The outputs of all expert computations are gathered back to their original sequence positions and weighted by the expert scores. If a token was routed to `top_k` experts, its `top_k` output vectors are combined (typically via a weighted sum) to produce the final token output.

The result has the same shape as the input: `[batch, seq, d_model]`. From the outside, an MoE layer has the same input/output contract as a dense FFN; only the internal execution pattern differs.

```python
# Conceptual MoE forward pass — illustrative structure only
def moe_forward_concept(x, router_weights, expert_weights):
    # Routing (steps 1–2) — each token selects top_k experts
    logits = x @ router_weights          # [batch*seq, num_experts]
    probs = logits.softmax(-1)           # softmax first, then top-k (Mixtral style); see routing_and_sparsity.md for sigmoid variant
    scores, indices = torch.topk(probs, k=top_k, dim=-1)  # scores are raw top-k softmax probs; selected scores do not sum to 1.0 without re-normalization
    # scores: [batch*seq, top_k], indices: [batch*seq, top_k]

    # Dispatch (step 3) — send tokens to assigned experts
    # (In practice: gather tokens per expert, batch by expert_capacity)

    # Expert compute (step 4) — each expert processes its assigned tokens
    # (In practice: matmul for each expert; see Chapter 3 for implementation)

    # Combine (step 5) — weighted sum of expert outputs
    # (In practice: scatter-add with routing scores; see Chapter 3)
    ...

# Full implementation details in Chapter 3 (batched matmul) and Chapter 4 (sparse_matmul).
```

> **Warning:** A naive implementation of step 3 above would loop over `num_experts` and call the expert FFN sequentially for each one. This is slow on accelerators; `moe_on_hardware.md` explains why and previews the batched alternatives.

---

## Sparse vs. Dense MoE Activation

In a **dense** model, every FFN parameter is used on every forward pass. The computational cost is proportional to the number of parameters times the number of tokens.

In a **sparse** MoE model, the router selects only `top_k` out of `num_experts` experts for each token. The rest of the experts produce no output for that token and receive no gradient through that token during training. This is **sparse activation**: the set of parameters that participate in any given forward pass is a small fraction of the total.

For a model with `num_experts = 8` and `top_k = 2`, each token activates 2 out of 8 experts, so 75% of the expert pool is inactive for any given token. This notion is made precise in `routing_and_sparsity.md` where the sparsity ratio is formally defined.

The trade-off is straightforward: sparse activation means more parameters per total FLOPs, enabling better model quality per unit of compute, but it introduces routing complexity and hardware efficiency challenges that this guide addresses.

---

## Common MoE Model Families

### Switch Transformer

Switch Transformer (Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", arXiv January 2021; JMLR 2022, arXiv:2101.03961) introduced the simplest possible routing scheme: `top_k = 1`. Each token is sent to exactly one expert (hence "switch" — the router switches between experts). This minimizes routing overhead and simplifies the load balancing problem, but means the weighted-combine step reduces to a single multiplication by the scalar router score.

Switch Transformer uses an **expert capacity** buffer of fixed size per expert. If more tokens are routed to an expert than its capacity allows, the excess tokens are **dropped** — the overloaded expert contributes a zero vector to the combine step for those tokens, while the token continues through the residual connection. This makes the computation shape static and predictable, at the cost of some token loss. See the Dropped Tokens section in `routing_and_sparsity.md` for the precise definition used in this guide.

> **Note:** Switch Transformer uses `CF = 1.25` as its operational default (Fedus et al., arXiv:2101.03961, Table 5). `CF = 1.0` is the theoretical minimum used in the paper to analyze token-drop behavior under uniform routing; it is not the recommended setting, as any load imbalance at `CF = 1.0` causes token drops.

### Mixtral 8x7B

Mixtral (Mistral AI, 2024, arXiv:2401.04088) uses `top_k = 2` with `num_experts = 8`. Each expert holds ~1/8 of the total expert FFN parameter budget; the total model has ~46.7B parameters, with approximately 12.8B active parameters per token (~1.5B from non-MoE components (attention weights, norms, and embedding tables) plus ~11.3B from 2 active expert FFNs × 32 layers, where each expert FFN has ~176M parameters); see arXiv:2401.04088 for the precise component breakdown. Each token is processed by 2 experts, and their outputs are combined using the softmax-normalized router scores as weights. Mixtral does not use hard capacity limits during inference, which avoids token dropping but requires handling variable-length expert inputs.

Mixtral's expert FFNs use the SwiGLU activation function (three weight matrices per expert: `W_gate`, `W_up`, `W_down`), which is standard in modern LLMs. The hidden dimension per expert is approximately 14,336 for Mixtral 8x7B.

### DeepSeek-MoE / DeepSeek-V2

Both DeepSeek-MoE and DeepSeek-V2 share two architectural innovations over earlier MoE designs:

1. **Shared experts**: A subset of experts (typically 2) are not gated — they process every token unconditionally, acting like a shared dense FFN that all tokens pass through. Only the remaining experts are routed sparsely.
2. **Fine-grained experts**: Uses a much larger number of smaller routed experts rather than fewer large ones. This increases routing flexibility and specialization.

**DeepSeek-MoE** (January 2024; arXiv:2401.06066): The original paper introducing these innovations. The 16B model variant uses `num_routed_experts=64` routed experts with `top_k=6` and `num_shared_experts=2`.

**DeepSeek-V2** (May 2024; arXiv:2405.04434): The scaled-up follow-on. Increases the routed expert pool to `num_routed_experts=160` while retaining the same shared-expert and fine-grained routing design. DeepSeek-V2 specifically introduces sigmoid-based gating for the routed experts (rather than softmax), where each expert's gate score is computed independently via sigmoid — this is a DeepSeek-V2 innovation, not shared with DeepSeek-MoE (see Section 3.1 of arXiv:2405.04434 for the sigmoid gating details). See the Router Architecture section of `routing_and_sparsity.md` for an overview of sigmoid gating.

Note that the active-expert fraction differs substantially: DeepSeek-MoE activates 6/64 = 9.375% of routed experts per token, while DeepSeek-V2 activates only 6/160 = 3.75% — a sparser routing regime with different hardware implications (discussed in Chapter 6). Note: both DeepSeek-MoE 16B and DeepSeek-V2 have 8 active experts per token (6 routed + 2 shared); the percentages above count only routed experts.

This design gives DeepSeek-V2 a different sparsity pattern from Switch Transformer or Mixtral, because the shared experts create a dense compute component that coexists with the sparse routed component. When optimizing DeepSeek-V2 on TTNN, the shared experts are best treated separately from the sparse routing path.

### Cross-Model Parameter Comparison

| Model | `num_experts` (routed) | `num_shared_experts` | `top_k` | `d_model` | `d_ff` per expert |
|---|---|---|---|---|---|
| Switch Transformer (Switch-C) | up to 2048 (8–64 typical) | 0 | 1 | varies | varies |
| Mixtral 8x7B | 8 | 0 | 2 | 4096 | 14336 |
| DeepSeek-MoE 16B | 64 | 2 | 6 | 2048 | 1408 |
| DeepSeek-V2 | 160 | 2 | 6 | varies | very small (fine-grained) |

---

## How MoE Changes the Compute Graph vs. Dense FFN

In a standard dense transformer, the FFN sublayer is straightforward:

```
input [batch, seq, d_model]
    → linear_1 [d_model → d_ff]        # weight: [d_model, d_ff]
    → activation (e.g., GELU)
    → linear_2 [d_ff → d_model]        # weight: [d_ff, d_model]
output [batch, seq, d_model]
```

The shapes of all operations are fully determined by `batch`, `seq`, `d_model`, and `d_ff`. There is no data-dependent branching. The compute graph is static and identical for every forward pass.

An MoE layer breaks this regularity in three ways:

**1. Data-dependent routing:** The assignment of tokens to experts is determined by the gating network's output, which depends on the input data. Two different batches of tokens will produce different routing decisions, activating different subsets of experts and sending different numbers of tokens to each.

**2. Variable-length expert inputs:** Even with fixed `batch` and `seq`, the number of tokens assigned to expert `i` varies across forward passes. Expert `i` might receive 0 tokens in one pass and 50 in the next. This means the effective input shape to each expert's FFN is not statically known.

**3. Scatter/gather overhead:** The dispatch-combine pattern requires two additional memory operations — a scatter to assemble per-expert batches and a gather to reassemble the output — that have no equivalent in the dense FFN.

The practical consequence for hardware execution is significant: operations that expect static shapes (like most compiled kernels) must either pad inputs to a fixed capacity, recompile for each new shape, or use a different op (like `sparse_matmul`) designed to handle structural sparsity. Chapter 3 and Chapter 4 address both strategies.

```python
# Dense FFN: static shapes, no routing
import ttnn

def dense_ffn(x, w1, w2):
    # x:  [batch, seq, d_model] — shapes fully static
    # w1: [d_model, d_ff]
    # w2: [d_ff, d_model]
    h = ttnn.matmul(x, w1)             # [batch, seq, d_ff]
    h = ttnn.gelu(h)
    return ttnn.matmul(h, w2)          # [batch, seq, d_model]

# MoE FFN: routing introduces data-dependent shapes
# (See Chapter 3 for the TTNN-optimized version)
def moe_ffn_naive(x, router, expert_w1s, expert_w2s, top_k):
    # x: [batch, seq, d_model]
    # Routing step produces indices of shape [batch, seq, top_k] — static
    # But the per-expert input shape is [n_tokens_i, d_model] — data-dependent
    ...
```

---

---

**Next:** [routing_and_sparsity.md](./routing_and_sparsity.md)
