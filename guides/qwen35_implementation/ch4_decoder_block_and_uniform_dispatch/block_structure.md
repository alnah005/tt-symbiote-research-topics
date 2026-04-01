# Block Structure: `DeltaNetDecoderBlock` Construction

`DeltaNetDecoderBlock` is declared in
`models/tt_transformers/tt/qwen35_decoder.py` and inherits from `LightweightModule`.
It has the same outward-facing forward signature as the standard `TransformerBlock`
so that the model forward loop can store all layers in a flat Python list and iterate
over them without any per-index type checks.

## Constructor Signature

```python
class DeltaNetDecoderBlock(LightweightModule):
    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        prefetcher=None,
        attention_class=None,   # None → GatedDeltaNet; otherwise e.g. GatedAttention
        mlp_class=None,         # None → MLP; otherwise e.g. Qwen35MoE
        mlp_dtype=None,         # override dtype for the MLP sub-module
        mlp_weight_cache_path=None,  # separate cache path for MLP weights
    ):
```

Every parameter that `attention_class` or `mlp_class` needs is forwarded explicitly;
nothing is looked up globally.

## Attention Branch

The `attention_class` parameter selects between the two attention implementations at
construction time:

```python
if attention_class is not None:
    self.attention = attention_class(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=args,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        layer_num=layer_num,
        dtype=dtype,
        transformation_mats=None,
        configuration=args,
    )
else:
    self.attention = GatedDeltaNet(
        mesh_device=mesh_device,
        args=args,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        layer_num=layer_num,
        dtype=dtype,
    )
    batch = getattr(args, "batch_size", 1)
    self.attention.initialize_states(batch_size=batch)
```

Two things are noteworthy:

1. When `attention_class` is `None`, the block instantiates `GatedDeltaNet` and
   immediately calls `initialize_states(batch_size=batch)`. This allocates the
   recurrent state tensor $S \in \mathbb{R}^{B \times H \times d_k \times d_v}$ and
   the conv ring-buffer rows on device DRAM (see Chapter 2 for details). No equivalent
   call is needed for `GatedAttention` because its state lives in the KV cache, which
   is managed externally.

2. When `attention_class` is provided (e.g. `GatedAttention`), `transformation_mats=None`
   is passed. Qwen3.5's partial-RoPE correction is baked into the cos/sin matrices on the
   `rope_setup` object rather than via a separate transformation matrix argument; passing
   `None` tells `GatedAttention` to use the standard RoPE path with those pre-corrected
   matrices.

### When `hasattr(self.attention, "initialize_states")` is true

The `forward` method uses `hasattr(self.attention, "initialize_states")` as a runtime
sentinel to pick the calling convention:

- **True** → DeltaNet path: call `self.attention.forward(attn_in)` with only the
  input tensor.
- **False** → GatedAttention path: call
  `self.attention.forward(attn_in, current_pos=current_pos, rot_mats=rot_mats_global, mode=mode)`.

Because `GatedDeltaNet` defines `initialize_states` and `GatedAttention` does not, this
`hasattr` check is equivalent to `isinstance(self.attention, GatedDeltaNet)` without
requiring an import of `GatedDeltaNet` in the forward path.

## MLP Branch

The MLP is selected with a simple `or` fallback:

```python
mlp_cls = mlp_class or MLP
self.feed_forward = mlp_cls(
    mesh_device=mesh_device,
    tt_ccl=tt_ccl,
    args=args,
    state_dict=state_dict,
    weight_cache_path=mlp_weight_cache_path or weight_cache_path,
    layer_num=layer_num,
    dtype=mlp_dtype or dtype,
    model_config=args.get_model_config(),
    prefetcher=prefetcher,
)
```

Both `MLP` and `Qwen35MoE` expose the same constructor keyword arguments, making the
substitution transparent. The two optional overrides —

- `mlp_dtype` — allows the MLP to use a different weight precision than the attention
  sub-module (e.g. `bfp4` for MoE expert weights while attention stays in `bfp8`).
- `mlp_weight_cache_path` — allows expert weights to be cached in a separate directory,
  which is useful when expert weight files are large and should be stored independently
  of the attention weight cache.

## Layer Norm Wrappers

Both norms are constructed identically:

```python
self.attention_norm = DistributedNorm(
    RMSNorm(
        device=mesh_device,
        dim=args.dim,
        eps=args.norm_eps,
        state_dict=state_dict,
        state_dict_prefix=args.get_state_dict_prefix("", layer_num),
        weight_cache_path=None if args.dummy_weights else weight_cache_path,
        weight_dtype=ttnn.bfloat16,
        weight_key="attention_norm",
        is_distributed=args.is_distributed_norm,
        add_unit_offset=args.rms_norm_add_unit_offset,
        ccl_topology=args.ccl_topology(),
        tt_ccl=tt_ccl,
    ),
    args,
    tt_ccl=tt_ccl,
    prefetcher=prefetcher,
    TG=args.is_galaxy,
    ag_config_key="ATTN_LN_AG_CONFIG",
)
```

`ff_norm` is built the same way but uses `weight_key="ffn_norm"` and
`ag_config_key="FFN_LN_AG_CONFIG"`.

`DistributedNorm` wraps a standard `RMSNorm` to add the AllGather collective
needed when tensors are sharded across chips. On a single P100A device,
`args.is_distributed_norm` is `False` and the wrapper is a pass-through. The
`add_unit_offset` flag enables the $\gamma = 1 + w$ zero-centered weight
initialization used by Qwen3.5's norms.

## 27B Dense Build Loop (demo.py)

In the 27B demo, each layer is inspected via `args.layer_types[i]`:

```python
for i in tqdm(range(args.n_layers), desc="Layers"):
    if args.layer_types[i] == "linear_attention":
        layers.append(
            DeltaNetDecoderBlock(
                args=args,
                mesh_device=device,
                tt_ccl=model.tt_ccl,
                dtype=ttnn.bfloat8_b,
                state_dict=sd,
                layer_num=i,
                weight_cache_path=wcp,
            )
        )
    else:
        layers.append(
            TransformerBlock(
                args=args,
                mesh_device=device,
                tt_ccl=model.tt_ccl,
                dtype=ttnn.bfloat8_b,
                state_dict=sd,
                layer_num=i,
                weight_cache_path=wcp,
                transformation_mats=model.trans_mats_dict,
                attention_class=GatedAttention,
            )
        )
```

For the 27B model, the 48 `linear_attention` layers use `DeltaNetDecoderBlock` with the
default `attention_class=None` (DeltaNet) and the default `mlp_class=None` (standard
dense `MLP`). The 16 `full_attention` layers use the standard `TransformerBlock` class
directly. This means the 27B model has *two* distinct Python types in `model.layers`.
The uniform forward signature is what makes the iteration `for layer in model.layers`
correct for both.

## 35B-A3B MoE Build Loop (demo_a3b.py)

The A3B demo uses `DeltaNetDecoderBlock` for *all* 40 layers — including the 10
full-attention layers. This unifies the layer list to a single Python type:

```python
for i in tqdm(range(args.n_layers), desc="Layers"):
    if args.layer_types[i] == "linear_attention":
        attention_class = None
    else:
        attention_class = GatedAttention

    layers.append(
        DeltaNetDecoderBlock(
            args=args,
            mesh_device=device,
            tt_ccl=model.tt_ccl,
            dtype=ttnn.bfloat8_b,
            state_dict=sd,
            layer_num=i,
            weight_cache_path=wcp,
            attention_class=attention_class,
            mlp_class=Qwen35MoE,
        )
    )
```

`mlp_class=Qwen35MoE` is passed unconditionally to every layer. Every layer in the A3B
model — whether DeltaNet or full-attention — uses a MoE MLP. The `attention_class`
variable is set to `None` for the 30 DeltaNet layers and to `GatedAttention` for the 10
full-attention layers, but the `DeltaNetDecoderBlock` constructor always receives it as
a keyword argument.

The contrast between the two demos is summarized below:

| Model | Layer types in `model.layers` | `attention_class` varies? | `mlp_class` varies? |
|-------|-------------------------------|---------------------------|----------------------|
| 27B dense | `DeltaNetDecoderBlock` (48) + `TransformerBlock` (16) | By Python class | Always `None` (→ MLP) |
| 35B-A3B MoE | `DeltaNetDecoderBlock` (40) | By kwarg per layer | Always `Qwen35MoE` |

---

**Next:** [`forward_signature.md`](./forward_signature.md)
