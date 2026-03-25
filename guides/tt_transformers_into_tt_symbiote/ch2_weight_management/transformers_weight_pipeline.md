# TT Transformers Weight Pipeline

TT Transformers loads weights directly onto the mesh device at module construction time, with on-disk caching, 2-D mesh sharding, per-layer dtype selection, and DRAM-bank-width sharding all applied at that single moment.

## Overview: load at construction, not at first forward

Unlike Symbiote, TT Transformers has no separate `preprocess_weights` / `move_weights_to_device` phases. Every weight tensor is created and placed on the mesh inside `__init__`. The mesh device, `args` (model configuration), `state_dict`, and `weight_cache_path` are all passed as constructor arguments.

```python
class MLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        ...
    ):
```

---

## `ttnn.as_tensor` with `cache_file_name`

The central weight-loading primitive in TT Transformers is `ttnn.as_tensor`. When a `cache_file_name` path is supplied, `ttnn.as_tensor` reads the already-converted tensor from disk on subsequent runs instead of reconverting from the raw `torch.Tensor`. This avoids the cost of `bfloat8_b` / `bfloat4_b` conversion on every process start.

```python
result = ttnn.as_tensor(
    torch_tensor,
    dtype=type,            # e.g. ttnn.bfloat8_b or ttnn.bfloat4_b
    device=self.mesh_device,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        self.mesh_device, dims=dims, mesh_shape=args.cluster_shape
    ),
    layout=ttnn.TILE_LAYOUT,
    memory_config=(
        ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
    ),
    cache_file_name=cache_name(name),   # Path | None
)
```

`cache_name` is a closure built from `weight_cache_path` and a layer-specific prefix string:

```python
cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"
```

When `args.dummy_weights` is `True`, `cache_name` always returns `None`, disabling caching:

```python
if args.dummy_weights:
    cache_name = lambda _: None
```

Dummy-weights mode allows compile-time kernel compilation and graph tracing without a real checkpoint. The tensors are random but have the correct shapes and dtypes. `ttnn.as_tensor` with `cache_file_name=None` behaves normally — it converts the supplied `torch_tensor` without writing or reading a cache file.

> **Note:** The `weight_cache_path` is a `pathlib.Path` object. Cache files are named by concatenating the `state_dict_prefix` (layer name), the weight name, and an optional `hidden_dim_string` suffix that encodes the padded hidden dimension when padding has been applied. There is no equivalent mechanism in TT Symbiote.

---

## `ShardTensor2dMesh` and `ShardTensorToMesh`

TT Transformers shards weights across the mesh at load time rather than at inference time. Two mappers are used:

### `ShardTensor2dMesh`

Shards a single tensor along two dimensions simultaneously, mapping to a 2-D mesh grid. The `dims` argument is a tuple of two tensor dimension indices, and `mesh_shape` is `args.cluster_shape` (e.g., `(4, 8)` for a TG galaxy configuration).

```python
mesh_mapper=ttnn.ShardTensor2dMesh(
    self.mesh_device,
    dims=(-1, -2),          # (column_shard_dim, row_shard_dim)
    mesh_shape=args.cluster_shape
)
```

In `MLP.__init__`, the sharding dimensions depend on whether the model is running on a galaxy (`args.is_galaxy`):

```python
w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)
```

In `Attention.__init__`, the wqkv weight uses:

```python
mesh_mapper=ttnn.ShardTensor2dMesh(
    self.mesh_device,
    dims=(3, 2) if self.TG else (2, 3),
    mesh_shape=configuration.cluster_shape
)
```

The sharding dimension choice is therefore tightly coupled to `args.cluster_shape`, `args.is_galaxy`, and the number of devices. This is not a configuration that can be determined without knowing the full `args` object.

### `ShardTensorToMesh`

Used for 1-D sharding (single dimension). Seen in `Attention.__init__` for bias tensors and the `slice_mat` / `user_selection_matrix` helper tensors in TG mode:

```python
mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1)
mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
```

The `ReplicateTensorToMesh` mapper copies the tensor to every device on the mesh without splitting.

---

## `create_dram_sharded_mem_config`

After choosing where on the mesh a weight lives, TT Transformers further specifies where within each device's memory it is placed. `create_dram_sharded_mem_config` (defined in `model_config.py`) creates a `ttnn.MemoryConfig` that width-shards a weight matrix across the device's DRAM banks:

```python
def create_dram_sharded_mem_config(self, k, n, dram_grid=None):
    dram_cores = self.dram_grid_size.x   # 12 on Wormhole, 8 on P150, 7 on P100
    padded_size = math.ceil(n / (self.tile_size * dram_cores)) * (self.tile_size * dram_cores)
    shard_spec = ttnn.ShardSpec(
        dram_grid,
        (k, padded_size // dram_cores),
        ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec
    )
```

In `MLP.__init__`, two configs are created:

```python
w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
w2_mem_config    = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)
```

These configs are passed directly to `ttnn.as_tensor`'s `memory_config` argument. On galaxy configurations `ttnn.DRAM_MEMORY_CONFIG` (unsharded DRAM) is used instead because DRAM sharding assumes a specific core layout that does not apply to TG.

The benefit: hardware can prefetch weight tiles from multiple DRAM banks in parallel during matrix multiplication, improving memory bandwidth utilization.

---

## Per-layer dtype selection: `ModelOptimizations`, `TensorGroup`, `OpGroup`

TT Transformers assigns dtypes at per-layer granularity through a three-level hierarchy:

### `TensorGroup` (enum)

Groups of weight tensors that receive the same dtype:

| Enum value | Weights covered |
|---|---|
| `TensorGroup.FF1_FF3` | MLP gate and up projections (`w1`, `w3`) |
| `TensorGroup.FF2` | MLP down projection (`w2`) |
| `TensorGroup.WQKV` | Fused query/key/value projection |
| `TensorGroup.WO` | Attention output projection |
| `TensorGroup.KV_CACHE` | Key-value cache tensors |
| `TensorGroup.ACTIVATION` | Activation output dtype (not a weight) |

### `OpGroup` (enum)

Groups of compute operations that receive the same math fidelity kernel config:

| Enum value | Operation |
|---|---|
| `OpGroup.LI_FF1_FF3` | Linear: MLP gate/up projections |
| `OpGroup.LI_FF2` | Linear: MLP down projection |
| `OpGroup.LI_QKV_DECODE` | Linear: QKV projection in decode mode |
| `OpGroup.LI_O_DECODE` | Linear: output projection in decode mode |
| `OpGroup.SDPA_DECODE` | Scaled dot-product attention in decode mode |
| `OpGroup.LI_QKV_PREFILL` | Linear: QKV projection in prefill mode |
| `OpGroup.LI_O_PREFILL` | Linear: output projection in prefill mode |
| `OpGroup.SDPA_PREFILL` | Scaled dot-product attention in prefill mode |

### `ModelOptimizations`

The top-level configuration object. Two factory methods cover the most common deployments:

- `ModelOptimizations.accuracy(model_name)` — For Llama 3, Mistral 7B, Phi3-mini, and phi-4 models, uses `bfloat8_b` for attention/KV weights and `HIFI2_FP16` MLP fidelity. For other smaller models, uses `bfloat16` for attention and `HiFi4` fidelity. For 70B+ models (e.g. Llama-3.1-70B), falls back to `bfloat4_b` FF1/FF3 and `LOFI` fidelity even in accuracy mode.
- `ModelOptimizations.performance(model_name)` — uses `bfloat4_b` for FF1/FF3 and `LOFI` math fidelity for most models; Qwen2.5-7B/VL-7B use `bfloat8_b` MLP and `bfloat16` attention with `HiFi4` fidelity instead.

In `MLP.__init__`, the dtype for a given layer is retrieved as:

```python
ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num,
    tensor=TensorGroup.FF1_FF3,
    prefetcher=use_prefetcher,
)
```

When a hardware prefetcher is active, `get_tensor_dtype` returns a uniform dtype across all layers (to avoid race conditions from different block sizes).

The dtype is then passed directly to `ttnn.as_tensor`:

```python
self.w1 = as_sharded_tensor("w1_sharded", ff1_3_dtype, dims=w1_dims)
```

---

## `load_checkpoints.py` — weight conversion utilities

The file `/localdev/salnahari/testing_dir/tt-metal/models/tt_transformers/tt/load_checkpoints.py` contains utilities for normalizing checkpoint formats before they are handed to `MLP` or `Attention`.

### `standardize_hf_keys(state_dict)`

Moves `model.embed_tokens.weight` to `lm_head.weight` to align HuggingFace checkpoint key names with the Meta key convention used internally:

```python
def standardize_hf_keys(state_dict):
    key_meta = "lm_head.weight"
    key_hf = "model.embed_tokens.weight"
    if key_meta not in state_dict and key_hf in state_dict:
        state_dict[key_meta] = state_dict[key_hf]
        del state_dict[key_hf]
    return state_dict
```

### `convert_hf_to_meta(state_dict, head_dim, n_heads, n_kv_heads)`

Chains three sub-operations:
1. `split_hf_keys` — splits fused `qkv_proj` or `gate_up_proj` tensors into separate Q/K/V and gate/up tensors.
2. `convert_hf_qkv_to_meta_format` — applies `reverse_permute` to Q and K weight/bias tensors.
3. `map_hf_to_meta_keys` — renames HuggingFace layer names (e.g., `self_attn.q_proj`) to Meta names (e.g., `attn.wq`).

### `reverse_permute(tensor, n_heads, dim1, dim2)`

Undoes the HuggingFace RoPE permutation applied to query and key weights. HuggingFace stores Q/K weights in an interleaved head-dimension order that is incompatible with the Meta / TT Transformers RoPE implementation. `reverse_permute` reshapes, transposes, and reshapes again to correct this:

```python
def reverse_permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
```

This function is called inside `convert_hf_qkv_to_meta_format` for every `q_proj.weight` and `k_proj.weight` key.

### `load_hf_state_dict(ckpt_dir)` and `load_hf_state_dict_filtered(ckpt_dir, key_prefixes)`

Two checkpoint loaders for HuggingFace safetensors format. The filtered variant uses `safetensors_safe_open` to load only the tensor keys matching a given prefix list, reducing peak host memory usage for large models.

---

**Next:** [reuse_vs_rewrite.md](reuse_vs_rewrite.md)
