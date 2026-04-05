# Chapter 5: MoE Details

This page provides a detailed walkthrough of the Mixture-of-Experts (MoE) data path within `Gemma4TextDecoderLayer`, covering the router, fused expert weights, the parallel MLP+MoE combination, and parameter/memory implications. Refer to [Chapter 2](../ch2_configuration_hierarchy/index.md) for `Gemma4TextConfig` MoE parameter defaults and [Chapter 5 index](index.md) for how MoE fits into the overall decoder layer.

---

## MoE Data Path Overview

When `Gemma4TextConfig.enable_moe_block` is `True`, each decoder layer contains a parallel MoE path alongside the standard MLP. The two paths share the same residual input but have independent normalization and computation:

```
residual (pre-feedforward hidden states) [B, S, hidden_size]
    |
    +--[MLP path]----------------------------------------------+
    |   pre_feedforward_layernorm(residual)                    |
    |   -> mlp(...)                                            |
    |   -> post_feedforward_layernorm_1(mlp_output)            |
    |   = hidden_states_1                                      |
    |                                                          |
    +--[MoE path]----------------------------------------------+
    |   residual.reshape(-1, hidden_size)  [B*S, hidden_size]  |
    |   -> router(hidden_states_flat)                          |
    |      returns: top_k_weights [B*S, K], top_k_index [B*S, K]
    |   -> pre_feedforward_layernorm_2(hidden_states_flat)     |
    |   -> experts(normed_input, top_k_index, top_k_weights)   |
    |   -> reshape back to [B, S, hidden_size]                 |
    |   -> post_feedforward_layernorm_2(moe_output)            |
    |   = hidden_states_2                                      |
    +----------------------------------------------------------+
    |
    hidden_states = hidden_states_1 + hidden_states_2   (element-wise sum)
    |
    post_feedforward_layernorm(hidden_states)
    |
    output = residual + hidden_states
```

Key architectural decisions:
- The MoE path receives the **raw residual** (not the pre-feedforward-normed version), then applies its own `pre_feedforward_layernorm_2`.
- The MLP path output goes through `post_feedforward_layernorm_1` before combination.
- The MoE path output goes through `post_feedforward_layernorm_2` before combination.
- After summation, the combined result goes through the standard `post_feedforward_layernorm` before being added back to the residual.
- This means MoE layers have **five** RMSNorm instances in the feedforward block alone: `pre_feedforward_layernorm`, `post_feedforward_layernorm_1`, `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_2`, and `post_feedforward_layernorm`.

---

## Gemma4TextRouter

```python
class Gemma4TextRouter(nn.Module):
```

The router determines which experts process each token. It uses a norm-scale-project-softmax-topk pipeline.

### Sub-Modules

| Component | Type | Shape/Value |
|---|---|---|
| `norm` | `Gemma4RMSNorm(hidden_size, with_scale=False)` | No learnable scale |
| `scale` | `nn.Parameter` | `[hidden_size]`, initialized to ones |
| `proj` | `nn.Linear(hidden_size, num_experts, bias=False)` | Projects to expert logits |
| `per_expert_scale` | `nn.Parameter` | `[num_experts]`, initialized to ones |

### Constants

- `scalar_root_size = hidden_size^(-0.5)` -- a pre-computed normalization constant

### Forward

```python
def forward(self, hidden_states):
    # Step 1: RMSNorm without learnable scale
    hidden_states = self.norm(hidden_states)                        # [B*S, hidden_size]

    # Step 2: Apply learnable scale and root-size normalization
    hidden_states = hidden_states * self.scale * self.scalar_root_size  # [B*S, hidden_size]

    # Step 3: Project to expert scores
    expert_scores = self.proj(hidden_states)                        # [B*S, num_experts]

    # Step 4: Softmax to get routing probabilities
    router_probabilities = softmax(expert_scores, dim=-1)           # [B*S, num_experts]

    # Step 5: Top-K selection
    top_k_weights, top_k_index = topk(router_probabilities, k=top_k_experts)  # both [B*S, K]

    # Step 6: Normalize top-k weights to sum to 1
    top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)       # [B*S, K]

    # Step 7: Apply per-expert scale
    top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]  # [B*S, K]

    return router_probabilities, top_k_weights, top_k_index
```

**Return values:**
- `router_probabilities` `[B*S, num_experts]` -- full softmax output (used for auxiliary loss logging via `OutputRecorder`)
- `top_k_weights` `[B*S, K]` -- normalized and per-expert-scaled routing weights
- `top_k_index` `[B*S, K]` -- indices of selected experts per token

Note that the normalization in Step 6 and the per-expert scale in Step 7 mean the final weights do **not** sum to 1.0 per token. The per-expert scale acts as a learned importance multiplier applied after normalization.

---

## Gemma4TextExperts

```python
class Gemma4TextExperts(MixtralExperts):
```

In `modular_gemma4.py`, this extends `MixtralExperts` but overrides `__init__` to use Gemma 4 config parameter names. In the flattened `modeling_gemma4.py`, it is a standalone class with the `@use_experts_implementation` decorator.

### Parameters

| Parameter | Shape | Description |
|---|---|---|
| `gate_up_proj` | `[num_experts, 2 * moe_intermediate_size, hidden_size]` | Fused gate and up projection for all experts |
| `down_proj` | `[num_experts, hidden_size, moe_intermediate_size]` | Down projection for all experts |

The `gate_up_proj` fuses what would normally be separate `gate_proj` and `up_proj` matrices into a single weight tensor per expert, split during forward via `.chunk(2, dim=-1)`.

### Forward

```python
def forward(self, hidden_states, top_k_index, top_k_weights):
    # hidden_states: [B*S, hidden_size]
    # top_k_index:   [B*S, K]
    # top_k_weights: [B*S, K]

    final_hidden_states = torch.zeros_like(hidden_states)

    # Build expert assignment mask
    expert_mask = one_hot(top_k_index, num_classes=num_experts)  # [B*S, K, E]
    expert_mask = expert_mask.permute(2, 1, 0)                   # [E, K, B*S]
    expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()   # active experts

    for expert_idx in expert_hit:
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

        current_state = hidden_states[token_idx]                 # [T, hidden_size]

        # Fused gate_up projection, then split
        gate, up = F.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
        # gate: [T, moe_intermediate_size], up: [T, moe_intermediate_size]

        current_hidden_states = act_fn(gate) * up                # SwiGLU activation
        current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])
        # [T, hidden_size]

        # Multiply by routing weight
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

        # Scatter-add back to output
        final_hidden_states.index_add_(0, token_idx, current_hidden_states)

    return final_hidden_states  # [B*S, hidden_size]
```

The loop iterates only over active experts (those assigned at least one token), making it efficient for sparse routing. The `index_add_` at the end accumulates contributions from multiple experts per token.

---

## How MLP and MoE Outputs Are Combined

The combination happens inside `Gemma4TextDecoderLayer.forward`:

```python
# MLP output
hidden_states = self.mlp(pre_feedforward_layernorm(residual))

if self.enable_moe_block:
    # Post-norm MLP output
    hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

    # MoE operates on raw residual
    hidden_states_flat = residual.reshape(-1, residual.shape[-1])
    _, top_k_weights, top_k_index = self.router(hidden_states_flat)
    hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
    hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
    hidden_states_2 = hidden_states_2.reshape(residual.shape)
    hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

    # Parallel sum
    hidden_states = hidden_states_1 + hidden_states_2

# Common exit
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

This is a **parallel** design, not sequential. Both the dense MLP and the sparse MoE receive the same residual-stream input (with different normalizations) and their outputs are summed. This differs from architectures where MoE replaces the MLP entirely.

---

## Parameter Counts and Memory Implications

### Per-Expert Parameters

Using the default config values from [Chapter 2](../ch2_configuration_hierarchy/index.md):

| Component | Shape | Parameters per Expert |
|---|---|---|
| `gate_up_proj` (one expert slice) | `[2 * moe_intermediate_size, hidden_size]` | `2 * moe_intermediate_size * hidden_size` |
| `down_proj` (one expert slice) | `[hidden_size, moe_intermediate_size]` | `hidden_size * moe_intermediate_size` |
| **Total per expert** | | `3 * moe_intermediate_size * hidden_size` |

### Router Parameters

| Component | Shape | Parameters |
|---|---|---|
| `proj.weight` | `[num_experts, hidden_size]` | `num_experts * hidden_size` |
| `scale` | `[hidden_size]` | `hidden_size` |
| `per_expert_scale` | `[num_experts]` | `num_experts` |
| **Total router** | | `num_experts * hidden_size + hidden_size + num_experts` |

### MoE vs Dense MLP Comparison

Each MoE-enabled layer contains **both** a dense MLP and a sparse MoE block:

- **Dense MLP** per layer: `3 * intermediate_size * hidden_size` parameters (gate + up + down)
- **MoE block** per layer: `3 * moe_intermediate_size * hidden_size * num_experts + router_params`

The MoE block is activated only for a subset of tokens (top_k out of num_experts), but all expert weights must reside in memory. This makes MoE layers significantly more memory-intensive than pure MLP layers despite their computational sparsity.

### Additional RMSNorm Overhead

Each MoE-enabled layer adds three extra RMSNorm instances (`post_feedforward_layernorm_1`, `pre_feedforward_layernorm_2`, `post_feedforward_layernorm_2`), each with `hidden_size` learnable parameters. This is negligible compared to expert weights but adds to the module count.

---

## TTNN Porting Considerations

1. **Expert parallelism**: The reference implementation loops over active experts sequentially. On TTNN, the fused `gate_up_proj` and `down_proj` 3D tensors enable batch-matmul across all experts simultaneously if the hardware supports it. This is the primary optimization opportunity.

2. **Token-to-expert routing**: The `one_hot` -> `permute` -> `where` pattern for building the expert assignment mask is not hardware-friendly. Consider using a gather/scatter approach or pre-sorted token indices for TTNN.

3. **Fused gate_up_proj**: The `[num_experts, 2 * moe_intermediate_size, hidden_size]` tensor is split via `.chunk(2, dim=-1)` after the matmul. On TTNN, this could be two separate matmuls with `gate_proj` and `up_proj` stored separately, or a single fused matmul with a post-split -- whichever maps better to the hardware.

4. **index_add_ scatter operation**: The `index_add_` used to accumulate expert outputs back into the token dimension requires atomic-like semantics when parallelized. On TTNN, this may need a dedicated scatter-add kernel or a reduction step.

5. **Parallel MLP + MoE scheduling**: The MLP and MoE paths are independent from the same residual. On multi-core TTNN hardware, these could execute concurrently. The MLP path is deterministic (all tokens), while the MoE path is data-dependent (routed tokens only).

6. **Router normalization chain**: The router applies RMSNorm (no scale) -> element-wise multiply by `scale * scalar_root_size` -> linear -> softmax -> topk -> normalize -> per_expert_scale multiply. This is a chain of simple ops that could be fused into a single kernel on TTNN.

7. **Memory layout for expert weights**: The 3D expert weight tensors `[num_experts, ...]` are naturally suited for expert-parallel sharding across cores, where each core holds one or more complete expert weight matrices.

8. **Sparse activation**: With `top_k_experts` out of `num_experts` active per token, only a fraction of expert compute is needed per token. However, different tokens may route to different experts, so all experts must be resident in memory. The TTNN implementation should consider whether to always run all experts (padding inactive ones) for deterministic compute graphs, or use truly sparse dispatch.

9. **Five RMSNorms in feedforward**: MoE layers have five separate RMSNorm applications in the feedforward block. Fusing consecutive norm-then-linear patterns where possible will reduce kernel launch overhead.

---

**Next:** [Chapter 6 -- Top-Level Model Assembly](../ch6_top_level_model_assembly/index.md)
