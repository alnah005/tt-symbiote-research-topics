# Chapter 4 — MLP Layers and Norms

This chapter covers the feed-forward (MLP) and normalization layers used inside every
`TransformerBlock` in TT Transformers, and describes how the corresponding building
blocks are provided in TT Symbiote.

## Chapter files

| File | Contents |
|------|----------|
| [`tt_transformers_mlp.md`](tt_transformers_mlp.md) | Detailed reference for `MLP` in TT Transformers: weight tensors, sharding, SwiGLU activation, DRAM-sharded memory configs, program configs, and the `RMSNorm` / `DistributedNorm` pair used as `attention_norm`, `ff_norm`, `pre_ff_norm`, and `post_ff_norm` inside `TransformerBlock`. |
| [`symbiote_mlp.md`](symbiote_mlp.md) | TT Symbiote linear layer modules (`TTNNLinear` family) that map to the MLP projection roles; their `preprocess_weights_impl`, `move_weights_to_device_impl`, and `forward` implementations; weight lifecycle hooks. |
| [`integration_gaps.md`](integration_gaps.md) | A structured comparison of what TT Transformers' native MLP provides versus what TT Symbiote currently supplies, including gaps in fused operations, memory layout, precision control, norm support, and CCL (collective communication) integration. |

## Source files referenced in this chapter

| Source file | Role |
|-------------|------|
| `models/tt_transformers/tt/mlp.py` | `MLP` class — the primary MLP implementation |
| `models/tt_transformers/tt/decoder.py` | `TransformerBlock` — shows how `MLP`, norms, and residuals connect |
| `models/tt_transformers/tt/model.py` | `Transformer` — top-level model, output norm |
| `models/tt_transformers/tt/model_config.py` | `ModelArgs`, `ModelOptimizations`, `TensorGroup`, `OpGroup` — dtype/fidelity/memory config logic |
| `models/common/rmsnorm.py` | `RMSNorm` — standalone norm module called by `DistributedNorm` |
| `models/experimental/tt_symbiote/modules/linear.py` | `TTNNLinear` and all subclasses |
| `models/experimental/tt_symbiote/core/module.py` | `TTNNModule` base class — weight lifecycle, `deallocate_weights_after`, `run_on_devices` |

## Navigation

- Previous chapter: [Chapter 3 — Attention](../ch3_attention/)
- Guide root: [TT Transformers into TT Symbiote](../plan.md)
