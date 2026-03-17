# FFN Compute Graph: Dense vs MoE, and the SwiGLU Path

This file covers the structure of the FFN block in both dense and MoE transformers, with a detailed annotated compute graph for the SwiGLU variant. It explains why SiLU appears per-token and per-expert in MoE inference.

---

## Dense FFN Block Structure

In a standard transformer, the FFN block applies two linear projections with a nonlinearity in between. For a token representation `x` of shape `[1, 1, T, d_model]` (T tokens, d_model hidden dimension):

```
x  →  W_up  →  activation(·)  →  W_down  →  output
```

In the classic formulation (e.g., GPT-2, BERT):

```
FFN(x) = activation(x @ W_up + b_up) @ W_down + b_down
```

where `W_up` has shape `[d_model, d_ffn]` and `W_down` has shape `[d_ffn, d_model]`. The activation is typically ReLU or GELU.

The SwiGLU variant (used in Llama 3, Mixtral, Qwen2-MoE, and others) introduces a third projection and replaces the simple activation with a gated activation:

```
FFN_SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
```

This creates a three-branch compute graph. SiLU is applied to the gate branch before element-wise multiplication with the up-projection branch. The down projection then maps the intermediate back to `d_model`.

---

## Annotated SwiGLU Compute Graph

```
Input x: [1, 1, T, d_model]
         │
         ├─────────────────────┬──────────────────────┐
         │                     │                      │
         ▼                     ▼                      │
   ttnn.matmul            ttnn.matmul                 │ (same x, two separate matmuls)
   (x, W_gate)            (x, W_up)                   │
         │                     │
         │ gate_hidden          │ up_hidden
         │ [1,1,T,d_ffn]        │ [1,1,T,d_ffn]
         │                     │
         ▼                     │
   ttnn.silu(gate_hidden)       │
         │                     │
         │ gated: [1,1,T,d_ffn] │
         │                     │
         └──────── ⊙ ──────────┘
                   │
                   │ intermediate: [1,1,T,d_ffn]
                   │
                   ▼
             ttnn.matmul
             (intermediate, W_down)
                   │
                   ▼
             output: [1,1,T,d_model]
```

**Notation:**
- `⊙` = element-wise multiply (`ttnn.mul`)
- `W_gate`, `W_up` both have shape `[d_model, d_ffn]`
- `W_down` has shape `[d_ffn, d_model]`
- Bias terms omitted for clarity (Llama-family models typically use no bias in FFN)

> **Tip:** In code, `gate_hidden` and `up_hidden` can be computed with a single fused matmul against a concatenated weight `[W_gate | W_up]` of shape `[d_model, 2 * d_ffn]`, then split. Whether the SiLU is fused into that matmul depends on the TTNN op configuration. See [`swiglu_variant.md`](swiglu_variant.md) for details on fused vs standalone paths.

---

## MoE FFN Differences

In an MoE model, the FFN weight matrices (`W_gate`, `W_up`, `W_down`) are not single matrices but are stacked across N experts. For a model with N experts and per-expert FFN dimension `d_ffn_expert`:

| Tensor | Dense shape | MoE stacked shape |
|---|---|---|
| W_gate | `[d_model, d_ffn]` | `[N, d_model, d_ffn_expert]` |
| W_up | `[d_model, d_ffn]` | `[N, d_model, d_ffn_expert]` |
| W_down | `[d_ffn, d_model]` | `[N, d_ffn_expert, d_model]` |

For Mixtral 8x7B: N=8, K=2 (top-2 routing), `d_ffn_expert=14336`, `d_model=4096`.

### Expert Weight Slicing

During forward pass, the router assigns each token to K experts. Only the K selected expert weight slices need to be used. In TTNN, this involves indexing or slicing along the N dimension of the stacked weight tensors to extract sub-tensors for each expert.

```
full_W_gate: [N, d_model, d_ffn_expert]
expert_i_W_gate = full_W_gate[i]  # [d_model, d_ffn_expert]
```

After slicing, the SwiGLU compute graph from the previous section applies independently to each expert.

### Per-Expert Dispatch

Token-to-expert dispatch reorganizes the token batch so that each expert processes a contiguous subset of tokens. The general flow:

```
1. Router: logits = x @ W_router          → [T, N]
2. Softmax + top-K selection              → indices [T, K], weights [T, K]
3. Scatter: group tokens by expert index  → T_i tokens per expert i
4. For each active expert i:
     gate_i   = ttnn.matmul(tokens_i, W_gate[i])   → [T_i, d_ffn_expert]
     up_i     = ttnn.matmul(tokens_i, W_up[i])     → [T_i, d_ffn_expert]
     gated_i  = ttnn.silu(gate_i) ⊙ up_i           → [T_i, d_ffn_expert]
     out_i    = ttnn.matmul(gated_i, W_down[i])     → [T_i, d_model]
5. Gather: accumulate weighted expert outputs back into [T, d_model]
```

> **Warning:** Step 4 is typically sequential over active experts in a single-device implementation, though multi-device expert parallelism (T3K) can distribute experts across devices. Either way, `ttnn.silu` is invoked once per active expert per forward pass, not once per layer.

---

## Why SiLU Is Applied Per-Token Per-Expert

In a dense model with T tokens, SiLU is applied once per FFN layer to a tensor of shape `[1, 1, T, d_ffn]`. The total element count is `T * d_ffn`.

In an MoE model with top-K routing, SiLU is applied K times per layer, once per dispatched expert. Each invocation processes `T_i` tokens (where `sum(T_i) = T * K` across all active experts). The total element count across all expert SiLU calls is `T * K * d_ffn_expert`.

For Mixtral 8x7B with K=2:
- Dense-equivalent: `T * 14336` elements per SiLU call (if dense)
- MoE actual: `T * 2 * 14336 = T * 28672` elements total across 2 expert SiLU calls

The MoE model therefore invokes more total SiLU work per forward pass than the density-equivalent dense model would. Furthermore, each invocation operates on a smaller, potentially irregular tensor (because different experts receive different numbers of tokens after routing), which can reduce hardware utilization per call.

---

## Next Steps

Proceed to [`swiglu_variant.md`](swiglu_variant.md) for the mathematical definition of SwiGLU, a comparison of SiLU vs other activations, and a survey of which production models use SwiGLU vs plain activations.
