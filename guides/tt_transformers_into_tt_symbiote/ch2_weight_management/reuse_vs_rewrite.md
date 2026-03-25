# Reuse vs. Rewrite: Weight Management Features

This file classifies every significant weight-pipeline feature from both codebases into one of three categories: **Reuse** (works unchanged), **Adapt** (usable with targeted changes), or **Rewrite** (must be built from scratch in Symbiote).

---

## Reuse

These features can be carried into TT Symbiote without modification.

### `ttnn.model_preprocessing` functions

`preprocess_linear_weight` and `preprocess_linear_bias` are already used in Symbiote's linear classes. TT Transformers does not use these functions directly (it calls `ttnn.as_tensor` instead), but the functions are a correct and supported way to produce host-resident TTNN tensors from raw `torch.Tensor` parameters. Any port that follows Symbiote's two-phase lifecycle can continue to call these functions in `preprocess_weights_impl()`.

### bfloat8_b and bfloat16 dtype choices

The dtype tokens `ttnn.bfloat8_b` and `ttnn.bfloat16` are used identically in both codebases. Symbiote already applies `bfloat8_b` in `TTNNLinearLLama` and `SmartTTNNLinearLLama`. No adaptation is needed to use the same dtype values when porting TT Transformers weight-loading code.

### The host-staging pattern

Symbiote's split between `preprocess_weights_impl()` (CPU conversion, produces `tt_weight_host`) and `move_weights_to_device_impl()` (device transfer via `ttnn.to_device()`) is a sound pattern for any weight that does not require a mesh device at conversion time. TT Transformers non-sharded weights — for example, norm weights loaded directly with `ttnn.as_tensor` onto a single device — can be adapted to this pattern without behavioral change.

### `@deallocate_weights_after` decorator

The `deallocate_weights_after` decorator defined in `module.py` is model-agnostic. Any new linear class ported from TT Transformers can apply it to `forward()` to match the auto-deallocation behavior of `TTNNLinearLLama`. Classes that auto-deallocate must be `@trace_disabled` — see `symbiote_weight_pipeline.md` for the full rationale and the `SmartTTNN*` exception.

### Dummy-weights mode concept

The `args.dummy_weights` flag in TT Transformers is a simple `None`-cache-name idiom. Symbiote can implement the same behavior by passing `cache_file_name=None` to any `ttnn.as_tensor` call, or by substituting zero / random tensors in `preprocess_weights_impl()`. No new infrastructure is required.

---

## Adapt

These features are usable but require targeted changes to fit Symbiote's module lifecycle or configuration model.

### TT Transformers weight-cache file path system

TT Transformers builds cache paths as:

```python
cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"
```

and passes them to `ttnn.as_tensor`'s `cache_file_name` argument. This system requires a `weight_cache_path` (a `pathlib.Path`) and a `state_dict_prefix` string (the layer name). Neither concept exists in TT Symbiote today.

To adopt the cache, one of two approaches is required:

1. **Add `weight_cache_path` to the Symbiote module lifecycle.** `TTNNModule.preprocess_weights()` (or a new `preprocess_weights_cached()` variant) would need to accept a cache path and thread it into `preprocess_weights_impl()`. Each module would construct a per-layer cache key from its `_unique_name`.
2. **Skip the cache entirely.** Pass `cache_file_name=None` to `ttnn.as_tensor` for all Symbiote-managed weights. This is safe but removes the startup performance benefit.

> **Warning:** Sharing a cache directory between TT Transformers and a Symbiote port of the same model would require matching the exact cache key strings, including the `hidden_dim_string` suffix. Mismatched keys will silently produce incorrect weights at runtime.

### `bfloat4_b` dtype

`bfloat4_b` is used in TT Transformers' performance mode for FF1/FF3 MLP weights and for `slice_mat` / `user_selection_matrix` helper tensors in TG mode. It is a valid `ttnn` dtype and can be passed to `preprocess_linear_weight` or `ttnn.as_tensor` in Symbiote. However, no existing Symbiote linear class uses `bfloat4_b`, so a new class (e.g., `TTNNLinearLLamaBFloat4`) would need to be created, following the same pattern as `TTNNLinearLLama`.

### `ShardTensorToMesh` (1-D mesh sharding)

`ttnn.ShardTensorToMesh` is already available in Symbiote's run_config infrastructure (for example, `DistributedConfig.__post_init__` uses `ttnn.ShardTensor2dMesh`). Porting a single-dimension mesh-sharded weight from TT Transformers requires adding the `weights_mesh_mapper` argument to the `preprocess_linear_weight` call inside `move_weights_to_device_impl()`, exactly as done in `TTNNLinearInputReplicatedWeightSharded`. The dimension argument must be provided explicitly; it cannot be inferred from the module alone.

---

## Rewrite

These features have no direct equivalent in Symbiote and require new infrastructure or significant redesign.

### `ShardTensor2dMesh` loading inside `TTNNModule.preprocess_weights_impl()`

The full dimension-selection logic (`w1_dims`, `w2_dims`, and the `Attention` `wqkv` variants) is shown in `transformers_weight_pipeline.md` under `ShardTensor2dMesh`. The key gap: Symbiote's `TTNNModule` does not carry an `args` object; it carries only `self.device` (set via `to_device()`) and `self._model_config` (a plain dict set via `set_model_config()`). Neither field encodes `cluster_shape` or `is_galaxy`.

To support 2-D mesh sharding in Symbiote, the following must be added:

- A `cluster_shape` attribute accessible from within `move_weights_to_device_impl()`. This could be stored in `_model_config`, derived from `self.device.shape`, or passed as a constructor argument.
- Per-class logic to select `dims` based on whether the deployment is galaxy or non-galaxy. This logic currently lives inline in `MLP.__init__` and `Attention.__init__`, driven by `args.is_galaxy` and `args.cluster_shape`. In Symbiote there is no equivalent runtime flag; one must be added.

Until this infrastructure exists, any module using `ShardTensor2dMesh` cannot be ported as a Symbiote `TTNNModule` without a rewrite of its `move_weights_to_device_impl()`.

### Per-layer dtype selection (`ModelOptimizations` / `TensorGroup` / `OpGroup` pipeline)

TT Transformers assigns dtypes at construction time via:

```python
ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num,
    tensor=TensorGroup.FF1_FF3,
    prefetcher=use_prefetcher,
)
```

This pipeline requires:

- A `ModelOptimizations` instance (`decoders_optimizations`) attached to `args`.
- A `layer_num` integer.
- Knowledge of whether a hardware prefetcher is active.

Symbiote has none of these. The `_model_config` dict is a flat key-value store with no concept of per-layer or per-tensor-group dtype selection. A minimal port could hard-code the dtype in the linear class constructor (as `TTNNLinearLLama` does with `bfloat8_b`), but that loses the ability to switch between accuracy and performance modes at deploy time without changing code.

A full port requires building an equivalent of `ModelOptimizations` that integrates with Symbiote's module construction flow — either as a constructor argument to each linear module or as a structured entry in `_model_config`.

### `create_dram_sharded_mem_config` (DRAM-bank weight placement)

The width-sharded DRAM memory config produced by `create_dram_sharded_mem_config` places each weight tile in a specific DRAM bank, enabling hardware-level memory-bandwidth parallelism during matrix multiplication. Symbiote currently uses `ttnn.DRAM_MEMORY_CONFIG` everywhere (unsharded DRAM).

Adopting `create_dram_sharded_mem_config` requires:

- The `dram_grid_size` (number of DRAM cores on the device, which varies: 12 on Wormhole, 8 on P150, 7 on P100).
- The per-chip weight dimensions `(k, n // num_devices)` at construction time.
- Access to `self.device` during preprocessing (currently not available until `move_weights_to_device_impl()`).

Because the DRAM shard spec requires device-specific geometry, this configuration cannot be created on the host before `move_weights_to_device_impl()`. A rewrite of the Symbiote linear base class would be needed to pass the config into `ttnn.to_device()` (or to use `ttnn.as_tensor` in place of the two-phase host-staging approach) and to make device-grid information available at weight-placement time.

---

## Summary table

| Feature | Source | Category | Effort |
|---|---|---|---|
| `preprocess_linear_weight` / `preprocess_linear_bias` | Symbiote (already present) | Reuse | None |
| `bfloat8_b` / `bfloat16` dtypes | Both | Reuse | None |
| Host-staging pattern (`tt_weight_host` + `ttnn.to_device`) | Symbiote | Reuse | None |
| `@deallocate_weights_after` decorator | Symbiote | Reuse | None |
| Dummy-weights mode (`cache_file_name=None`) | TT Transformers concept | Reuse | Trivial |
| `ttnn.as_tensor` + `cache_file_name` weight cache | TT Transformers | Adapt | Medium — requires adding `weight_cache_path` to module lifecycle |
| `bfloat4_b` dtype | TT Transformers | Adapt | Low — new linear subclass following existing pattern |
| `ShardTensorToMesh` (1-D sharding) | TT Transformers | Adapt | Low — add `weights_mesh_mapper` in `move_weights_to_device_impl()` |
| `ShardTensor2dMesh` loading | TT Transformers | Rewrite | High — requires `cluster_shape` and `is_galaxy` in module context |
| `ModelOptimizations` / `TensorGroup` / `OpGroup` dtype pipeline | TT Transformers | Rewrite | High — no equivalent in Symbiote; needs new configuration infrastructure |
| `create_dram_sharded_mem_config` DRAM placement | TT Transformers | Rewrite | High — requires device-grid info at weight placement time |

---

**Next:** [Chapter 3 — Attention — RoPE, KV Cache, and SDPA](../ch3_attention/index.md)
