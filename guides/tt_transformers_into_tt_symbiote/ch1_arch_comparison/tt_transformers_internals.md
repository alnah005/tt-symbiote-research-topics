# TT Transformers Internals

---

## `LightweightModule` base class

Source file: `models/common/lightweightmodule.py`

```python
from models.common.lightweightmodule import LightweightModule

class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```

`LightweightModule` is intentionally minimal. Its entire implementation is `__call__ -> self.forward`. It does not inherit from `torch.nn.Module`, which means:

- No PyTorch autograd graph is constructed.
- No `_modules`, `_parameters`, or `_buffers` dicts are maintained.
- Attribute access goes directly to `__dict__` with no interception overhead.

Every component in TT Transformers — `Transformer`, `Attention`, `MLP`, `LMHead`, `Embedding`, `RMSNorm`, `TransformerBlock` — is a `LightweightModule`. This is a deliberate trade-off: the entire stack gives up PyTorch training infrastructure in exchange for lower host dispatch latency during inference.

---

## `Transformer`

Source file: `models/tt_transformers/tt/model.py`

```python
from models.tt_transformers.tt.model import Transformer
```

`Transformer` is the top-level model container. It is constructed once with a fully-loaded `state_dict` and a resolved `weight_cache_path`; all weights are moved to device during `__init__`.

### Key members

```python
class Transformer(LightweightModule):
    def __init__(self, args, dtype, mesh_device, state_dict, weight_cache_path, ...):
        self.args = args
        self.model_config = args.get_model_config()
        self.decoders_optimizations = args.decoders_optimizations
        self.tt_ccl = TT_CCL(self.mesh_device)
        self.embd = Embedding(...)          # or ScaledEmbedding if args.embed_scale is set
        self.rope_setup = RotarySetup(...)  # precomputes cos/sin matrices on device
        self.layers = [TransformerBlock(..., layer_num=i) for i in range(self.n_layers)]
        self.norm = DistributedNorm(RMSNorm(...), ...)
        self.lm_head = LMHead(...)
        self.sampling = SamplingGenerator(...)  # only when _supports_on_device_sampling is True
```

`self.sampling` is only created when `_supports_on_device_sampling` is `True`. That flag requires **both** conditions to hold: (1) `prefetcher is None`, AND (2) `vocab_size // sampling_splits <= 64 * 1024`. When neither or only one condition is met, `self.sampling` is set to `None`. Guard any access to `self.sampling` by checking `self._supports_on_device_sampling` (or testing `self.sampling is not None`) before use.

`sampling_splits` equals `num_devices` on multi-device meshes (where `mesh_device.shape != [1, 1]`), but is hardcoded to `2` on single-device (1×1) meshes. So on a single-chip setup, the effective threshold is `vocab_size // 2 <= 64 * 1024`, i.e., `vocab_size <= 131,072`.

`RotarySetup` (`models/tt_transformers/tt/rope.py`) computes and stores rotary position embedding matrices on device at construction time. Both global and local RoPE setups can be created (`rope_theta` vs `rope_theta_local`) and their transformation matrices are passed down to each `TransformerBlock`.

`TT_CCL` (`models/tt_transformers/tt/ccl.py`) manages collective communication operations (all-gather, reduce-scatter) for multi-device execution. It is constructed once at the `Transformer` level and shared with every `TransformerBlock`, `Attention`, and `MLP` instance via the `tt_ccl` argument.

`decoders_optimizations` is a `ModelOptimizations` instance (see below) that carries per-layer dtype and fidelity overrides. It is read by every `Attention` and `MLP` at their own construction time.

---

## `Attention`

Source file: `models/tt_transformers/tt/attention.py`

```python
from models.tt_transformers.tt.attention import Attention
```

### Per-layer dtype selection via `decoders_optimizations`

Every `Attention` instance resolves its weight and activation dtypes from `args.decoders_optimizations` at `__init__` time, not at forward time:

```python
decoders_optimizations = self.args.decoders_optimizations
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

`TensorGroup` is an `Enum` from `models/tt_transformers/tt/model_config.py` with values `FF1_FF3`, `FF2`, `WQKV`, `WO`, `KV_CACHE`, and `ACTIVATION`. Each entry names a logical group of tensors whose dtype can be overridden independently per decoder layer.

### Compute kernel configs

Six compute kernel configurations are resolved at construction time, one per operation group:

```python
self.li_qkv_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
    decoder_id=layer_num, op=OpGroup.LI_QKV_DECODE, configuration=configuration
)
self.sdpa_decode_compute_kernel_cfg  = decoders_optimizations.get_math_fidelity(
    decoder_id=layer_num, op=OpGroup.SDPA_DECODE, configuration=configuration
)
# … pattern repeats for LI_O_DECODE, LI_QKV_PREFILL, SDPA_PREFILL, LI_O_PREFILL
```

For `MathFidelitySetting` values, see the ModelOptimizations enums table below.

The `configuration` object also carries three pre-built compute kernel configs used as defaults:
- `configuration.compute_kernel_config_hifi2`
- `configuration.compute_kernel_config_hifi2_fp16` — this is the default config for MLP FF1/FF3 and FF2 linear operators.
- `configuration.compute_kernel_config_hifi4`

### Sliding window attention

Layer types are declared in `configuration.layer_types` (indexed by `layer_num`). When the type is `"sliding_attention"`, `Attention` sets:

```python
self.is_sliding = True
self.sliding_window = configuration.sliding_window
```

### TG-specific slice matrices

When `self.num_devices == 32` (i.e., a Galaxy / TG topology), `Attention.__init__` allocates two on-device matrices:

```python
self.slice_mat = ttnn.from_torch(
    weight,
    dtype=ttnn.bfloat4_b,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
)
self.user_selection_matrix = ttnn.from_torch(
    user_selection_matrix,
    dtype=ttnn.bfloat4_b,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
```

These matrices implement a KV-head routing scheme that distributes 32 devices across KV-head groups. They are constructed once and referenced throughout the forward pass.

---

## `MLP`

Source file: `models/tt_transformers/tt/mlp.py`

```python
from models.tt_transformers.tt.mlp import MLP
```

### DRAM-sharded weight loading

`MLP.__init__` loads all three weight tensors (`w1`, `w2`, `w3`) directly into TTNN tensors at construction time using `ttnn.as_tensor` with explicitly computed DRAM-sharded memory configs:

```python
w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
w2_mem_config    = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)
```

The `as_sharded_tensor` helper inside `__init__` transposes the raw weight, optionally pads to `args.hidden_dim`, makes the tensor 4D (`[1, 1, H, W]` — required for the DRAM prefetcher to parse weight shapes correctly), then calls:

```python
result = ttnn.as_tensor(
    torch_tensor,
    dtype=type,
    device=self.mesh_device,
    mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=args.cluster_shape),
    layout=ttnn.TILE_LAYOUT,
    memory_config=w1_w3_mem_config,  # or w2_mem_config or DRAM_MEMORY_CONFIG for TG
    cache_file_name=cache_name(name),
)
```

`ShardTensor2dMesh` shards the tensor across both mesh dimensions. The `dims` tuple and the padding axis differ by topology:

| Topology | `w1`/`w3` dims | `w2` dims | Padding axis |
|---|---|---|---|
| Non-TG | `(-2, -1)` | `(-1, -2)` | `dims[-1]` |
| TG | `(-1, -2)` | `(-2, -1)` | `dims[0]` |

Padding is applied per weight using its own `dims` tuple, not a single assumed axis.

### Per-layer dtype

`FF1_FF3` and `FF2` dtypes are resolved from `decoders_optimizations` independently:

```python
ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
)
ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
)
```

**Default:** Both `FF1_FF3` (`w1`/`w3`) and `FF2` (`w2`) default to `bfloat8_b` in `_default_settings()`. `bfloat4_b` for `FF1_FF3` is used in performance mode only (and in the 70B+ accuracy special case for `w1`/`w3`).

### Hidden-dim padding

When `args.hidden_dim != args.unpadded_hidden_dim`, the cache file name includes a suffix like `.hidden_dim_28672` to prevent loading incorrectly-padded cached weights:

```python
hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""
cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"
```

---

## `Generator`

Source file: `models/tt_transformers/tt/generator.py`

```python
from models.tt_transformers.tt.generator import Generator
```

`Generator` orchestrates the full prefill/decode loop. It is constructed with a list of `Transformer` instances (one per data-parallel replica), the shared `model_args`, and an optional `mesh_device`.

### Trace capture

After the first warm compilation run, `Generator._capture_trace_prefill` captures the prefill forward pass into a replayable TTNN trace:

```python
trace_id = ttnn.begin_trace_capture(self.model_args[model_id].mesh_device, cq_id=0)
transformed_inputs = self.model[model_id].transform_and_embed_prefill_inputs_device(*device_inputs)
tt_out_trace = self.model[model_id].ttnn_prefill_forward(
    x=transformed_inputs[0],
    rot_mats_global=tt_rot_mats_prefill_global,
    rot_mats_local=tt_rot_mats_prefill_local,
    page_table=transformed_inputs[1],
    chunk_page_table=transformed_inputs[2],
    kv_cache=kv_cache,
)
ttnn.end_trace_capture(self.model_args[model_id].mesh_device, trace_id, cq_id=0)
```

Decode traces are stored per `(device_sampling_bool, device_id)` key in `self.trace_ids_decode`. Trace inputs and outputs are stored in `self.trace_inputs_decode` and `self.trace_output_decode` so that subsequent decode steps can write new token IDs into the pre-allocated input buffers and replay the captured trace without re-running host dispatch.

### Warmup sweeps

`warmup_model_prefill` sweeps all supported prefill sequence lengths from `args.get_warmup_prefill_supported_seq_lens()`, and for each length runs a dummy forward pass to compile all TTNN kernels before serving real requests:

```python
for supported_length in sequence_lengths_to_warmup:
    self.prefill_forward_text(
        warmup_tokens, page_table_warmup, kv_cache,
        warmup_prompt_lens, warmup_empty_slots, enable_trace, model_id, param,
    )
```

The warmup also sweeps sampling parameters (`can_sample_on_device`, `non_greedy_decoding_on_device`) to ensure all kernel variants are compiled.

### Paged attention

When `kv_cache` is not `None`, `Generator` constructs a `page_table` tensor of shape `[batch_size, num_blocks]` where `num_blocks = ceil(seq_len / block_size)`. This enables variable-length sequences to share a common KV cache pool without pre-allocating the maximum sequence length for every request.

---

## `ModelOptimizations` and `model_config.py`

Source file: `models/tt_transformers/tt/model_config.py`

### Enums

```python
from models.tt_transformers.tt.model_config import TensorGroup, OpGroup, PrecisionSetting, MathFidelitySetting
```

| Enum | Values |
|---|---|
| `TensorGroup` | `FF1_FF3`, `FF2`, `WQKV`, `WO`, `KV_CACHE`, `ACTIVATION` |
| `OpGroup` | `LI_FF1_FF3`, `LI_FF2`, `LI_QKV_DECODE`, `LI_O_DECODE`, `SDPA_DECODE`, `LI_QKV_PREFILL`, `LI_O_PREFILL`, `SDPA_PREFILL`, `ACCURACY` |
| `PrecisionSetting` | `BFP4`, `BFP8`, `BF16` |
| `MathFidelitySetting` | `LOFI`, `HIFI2`, `HIFI2_NA`, `HIFI2_FP16`, `HIFI4`, `HIFI4_FP32` |

### `ModelOptimizations`

`ModelOptimizations` holds a mapping from `(TensorGroup, layer_id)` and `(OpGroup, layer_id)` to precision/fidelity settings. It exposes two main query methods:

- `get_tensor_dtype(decoder_id, tensor, prefetcher)` — returns the `ttnn.DataType` for a given tensor group and layer index.
- `get_math_fidelity(decoder_id, op, configuration)` — returns a `ttnn.WormholeComputeKernelConfig` (or equivalent) configured for the requested fidelity level.

Two factory class methods are provided:

- `ModelOptimizations.accuracy(model_name)` — builds a configuration optimised for numerical accuracy. Models larger than 70B still use `BFP4` FF1/FF3 weights (`w1`/`w3`) and `BFP8` attention because they are empirically insensitive to precision at that scale. Note: `FF2` (`w2`, the down-projection) is **not** set to BFP4 in this profile — it inherits `BFP8` from `_default_settings()`.
- `ModelOptimizations.performance(model_name)` — builds a configuration optimised for throughput.

Per-decoder-layer overrides can be loaded from JSON files named `performance_decoder_config.json` or `accuracy_decoder_config.json` placed alongside the model weights. The file constants are:

```python
PERFORMANCE_DECODER_CONFIG_FILENAME = "performance_decoder_config.json"
ACCURACY_DECODER_CONFIG_FILENAME    = "accuracy_decoder_config.json"
```

This JSON-driven override mechanism is how production deployments tune individual layers (e.g., raising the last few layers of a model to `BF16` for output stability) without modifying source code.

---

**Next:** [Chapter 2 — Weight Management and Precision](../ch2_weight_management/index.md)
