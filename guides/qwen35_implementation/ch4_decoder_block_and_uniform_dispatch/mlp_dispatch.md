# MLP Dispatch: Dense vs MoE Substitution

## The Factory Pattern

`DeltaNetDecoderBlock` selects its MLP implementation through a single `or` expression. The instantiation (`mlp_cls = mlp_class or MLP` and the full `self.feed_forward = mlp_cls(...)` call) is shown in full in `block_structure.md` §MLP Branch.

`mlp_cls` resolves to `MLP` when `mlp_class=None` (the default) and to the provided
class otherwise. Both `MLP` and `Qwen35MoE` accept the same keyword arguments, so the
instantiation line is identical in both cases. The block never inspects `mlp_cls` after
construction — it stores the result as `self.feed_forward` and calls it as:

```python
hidden_states = self.feed_forward.forward(hidden_states, mode)
```

This is the entirety of the dispatch. Neither the block nor the inference loop needs to
know which MLP class is held.

## Dense MLP: 27B Layers

For the 27B dense model, `mlp_class` is never passed to `DeltaNetDecoderBlock`, so
`mlp_cls` defaults to `MLP`. The standard `MLP` implements SwiGLU:

$$\text{MLP}(x) = \bigl(\text{silu}(x W_1) \odot x W_3\bigr) W_2$$

where $W_1, W_3 \in \mathbb{R}^{d \times d_{\text{ff}}}$ are the gate and up projections
and $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$ is the down projection. All three
weight matrices are loaded in `bfp8_b` from the state dict. The intermediate dimension
for the 27B model is $d_{\text{ff}} = 17408$ (stored as `hidden_dim` in model args).

There is no expert routing, no shared expert, and no topk selection. The forward path is
a pair of matmuls plus an element-wise fused SiLU-gate operation.

## MoE Substitution: 35B-A3B Layers

For the A3B model, `mlp_class=Qwen35MoE` is passed to every layer in the build loop (see `block_structure.md` §A3B Build Loop for the full `DeltaNetDecoderBlock(...)` call).

`Qwen35MoE` implements the full MoE forward — router matmul, host topk/softmax, per-expert
SwiGLU, and shared expert — while presenting the same two-argument interface
`forward(hidden_states, mode)` as the dense `MLP`. From the decoder block's perspective,
nothing changes; only the latency and DRAM access pattern differ.

MoE replaces the SwiGLU above with:

$$\text{MoE}(x) = \sigma(x W_{\text{sg}}) \cdot \text{SharedExpert}(x) + \sum_{e \in \text{TopK}(r(x),\,8)} w_e \cdot \text{Expert}_e(x)$$

where $r(x) = x W_{\text{router}}$ produces 256 logits, $w_e$ are the softmax-normalized
routing weights for the 8 selected experts, each expert is a SwiGLU MLP with
$d_{\text{ff}} = 512$, $W_{\text{sg}}$ is the `shared_expert_gate.weight`, and $\sigma$
is the sigmoid function.

> **Note:** The sigmoid gate $\sigma(x W_{\text{sg}})$ is a learned scalar scaling factor
> applied to the shared expert's output. It allows the model to dynamically suppress or
> amplify the shared expert's contribution on a per-token basis, independent of the routed
> expert weights.

Chapter 5 covers `Qwen35MoE` internals in full.

## State Dict Prefix Isolation

Both `MLP` and `Qwen35MoE` receive `state_dict` and `layer_num` and construct their own
weight key lookups. The convention after weight conversion in `qwen35_utils.py` is:

- Dense MLP keys: `model.layers.{i}.feed_forward.w1.weight`,
  `model.layers.{i}.feed_forward.w2.weight`,
  `model.layers.{i}.feed_forward.w3.weight`

- MoE keys: `model.layers.{i}.feed_forward.experts.gate_up_proj` (gate+up fused, shape [256, 1024, 2048]),
  `model.layers.{i}.feed_forward.experts.down_proj` (down, shape [256, 2048, 512]),
  plus `model.layers.{i}.feed_forward.shared_expert.*` and
  `model.layers.{i}.feed_forward.gate.weight`

The `mlp_weight_cache_path` parameter allows caching MoE expert tensors in a separate
directory from attention weights:

```python
self.feed_forward = mlp_cls(
    ...
    weight_cache_path=mlp_weight_cache_path or weight_cache_path,
    ...
)
```

This separation is useful because 256 experts $\times$ 40 layers $\times$ 2 matrices per
expert at `bfp4` consume approximately 15 GB of cache files, while the attention and
norm caches for the same model total roughly 1.4 GB. Keeping them in separate directories
makes selective cache invalidation straightforward — for example, when re-quantizing
expert weights without re-converting the attention weights.

## Constructor Signature Parity

The shared constructor interface that makes the substitution transparent is:

```python
MLP(
    mesh_device,
    tt_ccl,
    args,
    state_dict,
    weight_cache_path,
    layer_num,
    dtype,
    model_config,
    prefetcher,
)

Qwen35MoE(
    mesh_device,
    tt_ccl,
    args,
    state_dict,
    weight_cache_path,
    layer_num,
    dtype,
    model_config,
    prefetcher,
)
```

Both classes accept exactly these keyword arguments. Adding a new MLP variant (for
example, a sparse activation MLP or a fused expert kernel) requires only matching this
interface and passing the class as `mlp_class` at build time. No changes to
`DeltaNetDecoderBlock` or the inference loop are needed.

---

**Next:** [Chapter 5 — Mixture of Experts](../ch5_mixture_of_experts/index.md)
