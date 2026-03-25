# TT Transformers Attention: Architecture Deep Dive

The TT Transformers `Attention` class is defined in `models/tt_transformers/tt/attention.py`. It inherits from `LightweightModule` (not `nn.Module`) and owns its own weights, KV cache, RoPE state, and CCL handles. This document covers each major aspect of its design.

---

## 1. `Attention.__init__` Parameters

```python
class Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
    ):
```

| Parameter | Type | Role |
|-----------|------|------|
| `mesh_device` | `ttnn.MeshDevice` | Target device or mesh; used for all tensor allocation |
| `tt_ccl` | `TT_CCL` | Collective communication handle (all-gather, all-reduce, reduce-scatter) |
| `args` | model args object | Provides program configs, memory configs, and layout helpers via methods like `get_attn_qkv_program_config` |
| `state_dict` | `dict[str, torch.Tensor]` | Full model checkpoint; weights are extracted by key prefix |
| `weight_cache_path` | `Path \| None` | Directory for pre-converted TTNN weight files; `None` disables caching |
| `layer_num` | `int` | Layer index; used to look up per-layer dtype overrides and to build weight key prefixes |
| `dtype` | `ttnn.DataType` | Fallback dtype for weights not covered by per-layer overrides |
| `transformation_mats` | `dict` | Pre-computed RoPE transformation matrices; must contain `"decode"` and `"prefill"` keys (provided by `RotarySetup.get_both_trans_mats()`) |
| `configuration` | model configuration object | Provides `n_heads`, `n_kv_heads`, `head_dim`, `max_seq_len`, `layer_types`, `sliding_window`, and `ccl_topology` |
| `paged_attention_config` | `PagedAttentionConfig \| None` | If set, `init_kv_cache` allocates paged cache tensors shaped `(max_num_blocks, n_local_kv_heads, block_size, head_dim)` |
| `use_paged_kv_cache` | `bool` | If `True`, `init_kv_cache` is skipped entirely (vLLM provides its own cache externally) |
| `prefetcher` | `Prefetcher \| None` | If set, weights are registered with the hardware prefetcher and sub-device IDs are passed to all ops |

### Head Distribution

```python
self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
self.n_local_heads    = self.n_heads    // self.num_devices_per_group
self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group
```

On TG (32-device Galaxy), `num_devices_per_group` equals `n_kv_heads`, so each device holds exactly one KV head. On non-TG deployments, all devices share the full set of KV heads split evenly across `num_devices`.

---

## 2. Per-Layer Dtype Selection

TT Transformers selects dtypes per-layer via `decoders_optimizations.get_tensor_dtype`. There are four distinct tensor groups tracked for attention:

```python
self.activation_dtype = decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.ACTIVATION, prefetcher=use_prefetcher
)
self.wqkv_dtype = decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.WQKV, prefetcher=use_prefetcher
)
self.wo_dtype = decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.WO, prefetcher=use_prefetcher
)
self.kv_cache_dtype = decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.KV_CACHE, prefetcher=use_prefetcher
)
```

The mapping from `TensorGroup` and `PrecisionSetting` to `ttnn.DataType`:

| `PrecisionSetting` | `ttnn.DataType` |
|--------------------|-----------------|
| `BFP4` | `ttnn.bfloat4_b` |
| `BFP8` | `ttnn.bfloat8_b` |
| `BF16` | `ttnn.bfloat16` |
| `None` (ACTIVATION default) | original input dtype |

**Default settings** (from `ModelOptimizations._default_settings` in `model_config.py`):

| Tensor | Default precision |
|--------|-------------------|
| `WQKV` | `BFP8` |
| `WO` | `BFP8` |
| `KV_CACHE` | `BFP8` |
| `ACTIVATION` | `None` (pass-through) |

The `performance` and `accuracy` class methods on `ModelOptimizations` override these defaults per model family. For example, `accuracy` mode on Llama 3 uses `BFP8` for WQKV, WO, and KV_CACHE, while `accuracy` mode on smaller models uses `BF16` for all attention tensors.

---

## 3. Compute Kernel Configurations

Six per-op compute kernel configs are fetched at init time:

```python
self.li_qkv_decode_compute_kernel_cfg   # QKV matmul, decode
self.sdpa_decode_compute_kernel_cfg     # SDPA, decode
self.li_o_decode_compute_kernel_cfg     # output projection, decode
self.sdpa_prefill_compute_kernel_cfg    # SDPA, prefill
self.li_qkv_prefill_compute_kernel_cfg  # QKV matmul, prefill
self.li_o_prefill_compute_kernel_cfg    # output projection, prefill
```

These are returned by `decoders_optimizations.get_math_fidelity(decoder_id, op, configuration)`, which looks up the `MathFidelitySetting` enum stored for the given op and returns the corresponding **compute kernel config object** (e.g., `configuration.compute_kernel_config_hifi2`) — not the `MathFidelitySetting` enum itself. Callers receive a fully-constructed `ttnn.WormholeComputeKernelConfig` (or equivalent) that is passed directly to the TTNN op.

Additionally, three base configs are pulled from `configuration` directly:

```python
self.compute_kernel_config_hifi2        # HiFi2 — used for DRAM-sharded matmuls
self.compute_kernel_config_hifi2_fp16   # HiFi2 with FP16 accumulation
self.compute_kernel_config_hifi4        # HiFi4 — used for accuracy-sensitive ops
```

**Default fidelities by op** (from `ModelOptimizations._default_settings`):

| Op | Default `MathFidelitySetting` | Rationale |
|----|-------------------------------|-----------|
| `LI_QKV_DECODE` | `HIFI2` | DRAM-bound; precision loss acceptable |
| `SDPA_DECODE` | `HIFI2` | Decode SDPA is memory-bound |
| `LI_O_DECODE` | `HIFI2` | Output projection is DRAM-bound |
| `LI_QKV_PREFILL` | `HIFI2` | Prefill QKV is compute-bound but memory allows HiFi2 |
| `SDPA_PREFILL` | `HIFI4` | Prefill SDPA is accuracy-sensitive |
| `LI_O_PREFILL` | `HIFI2` | FP32 accumulation important here |

---

## 4. Sliding Window Attention

```python
self.is_sliding = (
    configuration.layer_types[layer_num] == "sliding_attention"
    if configuration.layer_types else False
)
self.sliding_window = configuration.sliding_window if self.is_sliding else None
```

When `is_sliding` is `True`, the `sliding_window` value is passed to both `ttnn.transformer.paged_scaled_dot_product_attention_decode` and `ttnn.transformer.scaled_dot_product_attention_decode` as the `sliding_window_size` argument. `sliding_window_size=None` (the default) disables the constraint and allows full-context attention.

---

## 5. TG (Galaxy) Topology — 32-Device Logic

When `self.TG` is `True` (i.e., `num_devices == 32`), two additional matrices are constructed at init time.

### `slice_mat`

```python
weight = torch.zeros(1, 32, 8, 32)
for i in range(32):
    col = i % 4
    weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

self.slice_mat = ttnn.from_torch(
    weight,
    dtype=ttnn.bfloat4_b,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
)
```

`slice_mat` is a `[1, 32, 8, 32]` selection matrix sharded across the 32 devices along `dim=1`. In `forward_decode`, after the all-reduce, the fused QKV output (which has been gathered from all device groups) is multiplied by `slice_mat` to extract only the 8 users that belong to each device group:

```python
xqkv_fused = ttnn.matmul(self.slice_mat, xqkv_fused, ...)
```

### `user_selection_matrix`

```python
user_selection_matrix = torch.eye(8, 8)
user_selection_matrix = torch.nn.functional.pad(user_selection_matrix, (0, 24), "constant", 0)  # (8, 32)
user_selection_matrix = [user_selection_matrix] * 4
user_selection_matrix = torch.block_diag(*user_selection_matrix)  # (32, 128)

self.user_selection_matrix = ttnn.from_torch(
    user_selection_matrix,
    dtype=ttnn.bfloat4_b,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
```

`user_selection_matrix` is a `[32, 128]` block-diagonal selection matrix replicated to all 32 devices. After the output projection and the second all-gather in `forward_decode`, the TG path multiplies the gathered output by `user_selection_matrix` to re-select each device group's 8 users from the full 128-user concatenation.

---

## 6. Forward Methods: Prefill vs. Decode

TT Transformers uses two separate forward methods rather than a mode flag:

| Method | Input shape | Mode |
|--------|-------------|------|
| `forward_decode` | `(seq_len, 1, batch, dim)` | `Mode.DECODE` |
| `forward_prefill` | `(1, 1, seq_len, dim)` | `Mode.PREFILL` |

`Mode` is an enum defined in `models/tt_transformers/tt/common.py`.

### Mode-Dependent Behavior Summary

| Concern | Decode | Prefill |
|---------|--------|---------|
| Input shape | `(seq_len, 1, batch, dim)` | `(1, 1, seq_len, dim)` |
| Input reshape | — | Reshape to `[1, seq_len // MAX_QKV_MM_SEQ_LEN, MAX_QKV_MM_SEQ_LEN, dim]` if `seq_len > MAX_QKV_MM_SEQ_LEN` |
| QKV dtype | HiFi2 | HiFi2 or HiFi2_FP16 |
| Post-all-reduce step | [TG: `slice_mat` matmul] | — |
| Head reshape op | `nlp_create_qkv_heads_decode` | `nlp_create_qkv_heads` |
| Pre-RoPE typecast | — | `bfloat16` |
| RoPE `is_decode_mode` | `True` | `False` |
| Transformation matrix key | `"decode"` | `"prefill"` |
| Pre-cache typecast | — | K/V → `kv_cache_dtype` |
| Cache write op | `paged_update_cache` / `paged_fused_update_cache` | `paged_fill_cache` / `fill_cache` |
| SDPA op | `paged_scaled_dot_product_attention_decode` or `scaled_dot_product_attention_decode` | `scaled_dot_product_attention` (HiFi4) |
| Head concat op | `nlp_concat_heads_decode` | `nlp_concat_heads` |
| Output collective | [Ring: `all_gather_matmul_async`] or [TG/default: `all_gather_async` + `linear(wo)` + `tt_all_reduce` cluster_axis=0] | `tt_all_reduce` / `all_gather + linear(wo)` |
| Seq len constraint | None (single token per step) | Must be divisible by 128 |

---

## 7. `RotarySetup` (`models/tt_transformers/tt/rope.py`)

`RotarySetup` is a `LightweightModule` that pre-computes all cosine/sine matrices and transformation matrices at model initialization time. The `Attention` class does not instantiate `RotarySetup` itself — it is created by the model-level init (e.g., in `model.py`) and the resulting `transformation_mats` dict is passed in.

### Constructor

```python
class RotarySetup(LightweightModule):
    def __init__(
        self,
        device,
        batch_size,
        head_dim,
        max_seq_len,
        rope_theta,
        rope_scaling=None,
        use_qk_fused=False,
        datatype=ttnn.bfloat16,
        shard_batch_to_mesh_dim=1,
        prefetcher=None,
    ): ...
```

**What is allocated at init:**

| Attribute | Layout | Shape | Notes |
|-----------|--------|-------|-------|
| `cos_matrix` | `ROW_MAJOR_LAYOUT` | `[1, 1, max_seq_len, head_dim]` | Used for decode embedding lookup |
| `sin_matrix` | `ROW_MAJOR_LAYOUT` | `[1, 1, max_seq_len, head_dim]` | Used for decode embedding lookup |
| `cos_matrix_prefill` | `TILE_LAYOUT` | `[1, 1, max_seq_len, head_dim]` | Used for prefill |
| `sin_matrix_prefill` | `TILE_LAYOUT` | `[1, 1, max_seq_len, head_dim]` | Used for prefill |
| `transformation_mat` | `TILE_LAYOUT`, HEIGHT-sharded | `[1, 1, batch_size_per_device_group * TILE_SIZE, TILE_SIZE]` | Decode transformation matrix. `batch_size_per_device_group` is derived from `doubled_batch_size`, where `doubled_batch_size = batch_size * 2` when `use_qk_fused=True`, otherwise `doubled_batch_size = batch_size`. On non-TG, `batch_size_per_device_group == doubled_batch_size`. On TG, `batch_size_per_device_group = doubled_batch_size // devices_in_shard_dim`. |
| `transformation_mat_prefill` | `TILE_LAYOUT`, DRAM | `[1, 1, TILE_SIZE, TILE_SIZE]` (i.e., `[1, 1, 32, 32]`) | Prefill transformation matrix. Despite being called with `dhead=head_dim`, `get_rot_transformation_mat` hardcodes `dhead = 32` internally, so the shape is always `[1, 1, 32, 32]` regardless of `head_dim`. |

The decode matrix is additionally repeated along dimension 2 to tile it across `batch_size_per_device_group` cores.

### `get_both_trans_mats()`

```python
def get_both_trans_mats(self) -> Dict[str, ttnn.Tensor]:
    return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}
```

This is the method called at model init to produce the `transformation_mats` dict that is injected into every `Attention` layer.

### `get_rot_mats(position_idxs, return_rot_idxs=False)`

```python
def get_rot_mats(self, position_idxs, return_rot_idxs=False) -> List[ttnn.Tensor]:
```

Called once per decode step (from the generation loop, not from `Attention.forward`). It:

1. Converts `position_idxs` (a `[batch]` torch tensor) to a `[1, batch]` TTNN uint32 tensor via `get_rot_idxs`.
2. Runs `ttnn.embedding` over `cos_matrix` and `sin_matrix` to extract the rows corresponding to current positions.
3. Returns `[cos, sin]` — a list of two tensors, both in the memory config expected by `rotary_embedding_llama`.

When `use_qk_fused=True`, position indices are doubled (`position_idxs.repeat(2)`) so that the same index tensor can serve both the Q and K rotary embedding in a single fused op.

### Prefetcher Integration

When a `Prefetcher` is provided, `get_rot_mats` uses `ttnn.num_cores_to_corerangeset_in_subcoregrids` to place the embedding output on the prefetcher's worker sub-core grids, avoiding cross-core data movement before the rotary op.

---

## 8. Weight Construction

All attention weights are constructed in `__init__` from the `state_dict` directly (no `from_torch` pattern).

### QKV Weight

```python
for i in range(self.num_devices_per_group):
    wq_selected = torch.chunk(state_dict[f"{wq_str}.weight"], self.num_devices_per_group, dim=0)[i]
    wk_selected = torch.chunk(state_dict[f"{wk_str}.weight"], self.num_devices_per_group, dim=0)[i]
    wv_selected = torch.chunk(state_dict[f"{wv_str}.weight"], self.num_devices_per_group, dim=0)[i]
    qkv = torch.cat([wq.T, wk.T, wv.T], dim=-1)
    qkv_list.append(qkv)

qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

self.wqkv = ttnn.as_tensor(
    qkv_cat,
    dtype=self.wqkv_dtype,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG if self.TG else wqkv_mem_config,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        self.mesh_device,
        dims=(3, 2) if self.TG else (2, 3),
        mesh_shape=configuration.cluster_shape,
    ),
    cache_file_name=cache_name("wqkv_sharded_2d"),
)
```

The combined weight is 2D-sharded: along `dim=3` for TG and along `dim=2` for non-TG. The weights are transposed (`wq.T`) before concatenation so that the on-device layout matches the expected input-dimension ordering for `ttnn.linear`.

### Output Weight

```python
pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
self.wo = ttnn.as_tensor(
    pt_wo,
    dtype=self.wo_dtype,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG if (self.use_fused_all_gather_matmul or self.TG) else wo_mem_config,
    mesh_mapper=ttnn.ShardTensor2dMesh(
        self.mesh_device,
        dims=self.shard_wo_dims,
        mesh_shape=configuration.cluster_shape,
    ),
    ...
)
```

`shard_wo_dims = (2, 3)` when using fused all-gather matmul or TG; `(3, 2)` otherwise.

### KV Cache

When `use_paged_kv_cache=False` and `paged_attention_config` is set, `init_kv_cache` allocates cache tensors shaped `(max_num_blocks, n_local_kv_heads, block_size, head_dim)`. Without `paged_attention_config`, the shape is `(batch_size_per_device_group, n_local_kv_heads, max_seq_len, head_dim)`. Both variants use `kv_cache_dtype` and `ttnn.ReplicateTensorToMesh`.

---

[Previous: Symbiote Attention Overview](symbiote_attention_overview.md) | [Next: Integration Gaps](integration_gaps.md)
