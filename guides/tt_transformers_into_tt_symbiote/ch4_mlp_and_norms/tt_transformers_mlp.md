# TT Transformers MLP and Norms

Source files: `models/tt_transformers/tt/mlp.py`, `models/tt_transformers/tt/decoder.py`,
`models/tt_transformers/tt/model.py`, `models/tt_transformers/tt/model_config.py`,
`models/common/rmsnorm.py`

---

## 1. The `MLP` class

The primary MLP implementation lives in `mlp.py` as class `MLP(LightweightModule)`.
It is not named `TtLlamaMLP`; the class is simply `MLP`. Inside `TransformerBlock`
(`decoder.py` line 88) it is instantiated as `self.feed_forward = MLP(...)`.

### 1.1 Constructor signature

```python
class MLP(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,            # ModelArgs instance
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        model_config,
        state_dict_prefix=None,
        prefetcher=None,
    ):
```

Source: `mlp.py` lines 14–27.

### 1.2 Weight tensors

The MLP has three projection matrices, following the SwiGLU / LLaMA naming convention:

| Attribute | HuggingFace name | Role in forward pass |
|-----------|-----------------|----------------------|
| `self.w1` | `gate_proj` | Gating branch — passed through the activation function |
| `self.w2` | `down_proj` | Reduction from hidden back to model dim |
| `self.w3` | `up_proj` | Value branch — element-wise multiplied with activated gate |

All three weights are loaded via the local helper `as_sharded_tensor` defined inside
`__init__`. The raw weight is transposed (`torch.transpose(..., -2, -1)`) before being
placed on device, so the stored tensor shape is `[1, 1, in_features, out_features_per_device]`
(i.e., column-major from the linear algebra perspective).

### 1.3 Weight dtypes — `TensorGroup` and `ModelOptimizations`

Dtypes are not hardcoded. They are resolved at construction time via
`decoders_optimizations.get_tensor_dtype(decoder_id, tensor, prefetcher)`.

Relevant `TensorGroup` members (`model_config.py` lines 48–55):

| `TensorGroup` | MLP weights it covers |
|---------------|----------------------|
| `TensorGroup.FF1_FF3` | `w1` (gate_proj) and `w3` (up_proj) |
| `TensorGroup.FF2` | `w2` (down_proj) |

Default precision settings (`model_config.py` lines 253–276):

| Tensor group | Default dtype | Notes |
|--------------|--------------|-------|
| `FF1_FF3` | `bfp8_b` (BFP8) | Can be lowered to BFP4 in `performance` mode for most models |
| `FF2` | `bfp8_b` (BFP8) | Separate dtype so down_proj can differ |

In `performance` mode (`model_config.py` lines 186–188), the default for most models
sets `FF1_FF3` to `BFP4` with `LOFI` math fidelity. In `accuracy` mode the exact
settings depend on the model size (see `ModelOptimizations.accuracy` method,
`model_config.py` lines 91–156).

At `mlp.py` lines 90–95:
```python
ff1_3_dtype = self.decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.FF1_FF3, prefetcher=use_prefetcher
)
ff2_dtype = self.decoders_optimizations.get_tensor_dtype(
    decoder_id=layer_num, tensor=TensorGroup.FF2, prefetcher=use_prefetcher
)
```

On Galaxy (TG, 32-device), the **output activation** of the `w1` and `w3` linear ops is
always `ttnn.bfloat8_b` regardless of the optimization setting (see `mlp.py` line 148:
`dtype=ttnn.bfloat8_b if TG else activation_dtype or ttnn.bfloat16`). The **weight**
dtype for `w1` and `w3` is still resolved through `decoders_optimizations.get_tensor_dtype`
(i.e., `ff1_3_dtype`) on all devices including Galaxy.

### 1.4 Weight memory configs — DRAM sharding

Weight memory configs for `w1`/`w3` and `w2` are constructed at the start of `__init__`
(`mlp.py` lines 51–52):

```python
w1_w3_mem_config = args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices)
w2_mem_config    = args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim)
```

`create_dram_sharded_mem_config` (`model_config.py` lines 2957–2965) returns a
`WIDTH_SHARDED` DRAM memory config. The weight tensor is spread across all DRAM banks
(12 banks on Wormhole, 8 on P150, 7 on P100) along its output-feature axis. This enables
the DRAM-sharded matmul kernel path during decode (each DRAM bank feeds a subset of
compute cores).

For Galaxy the `as_sharded_tensor` helper always uses `ttnn.DRAM_MEMORY_CONFIG` (plain
interleaved DRAM) instead of the width-sharded config:
```python
memory_config=(
    ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
),
```
Source: `mlp.py` line 72.

### 1.5 Sharding dimensions — `w1_dims` / `w2_dims`

Tensor-parallel sharding for multi-device is controlled by `dims` passed to
`ttnn.ShardTensor2dMesh`:

```python
w1_dims = (-1, -2) if args.is_galaxy else (-2, -1)
w2_dims = (-2, -1) if args.is_galaxy else (-1, -2)
```
Source: `mlp.py` lines 79–80.

### 1.6 Weight shape — 4D unsqueeze requirement

`as_sharded_tensor` applies `unsqueeze(0).unsqueeze(0)` to make every weight 4D
(`[1, 1, H, W]`). This is noted as critical for the DRAM prefetcher to correctly
interpret all weights (`mlp.py` lines 55–56).

### 1.7 Weight caching

Cache file names are derived from `weight_cache_path / f"{state_dict_prefix}.{name}"`.
If `args.dummy_weights` is set, `cache_name` returns `None` and no file is written or
read. If `args.hidden_dim != args.unpadded_hidden_dim` (padding was applied), the
padded dimension is appended to the cache name to prevent loading mismatched weights.
Source: `mlp.py` lines 44–49.

### 1.8 Prefetcher registration

When a `Prefetcher` is provided, weights are registered in order `w1, w3, w2` via
callbacks:
```python
def register_weights():
    self.prefetcher.insert_tensor(self.w1)
    self.prefetcher.insert_tensor(self.w3)
    self.prefetcher.insert_tensor(self.w2)
self.prefetcher.register_callback(register_weights)
```
Source: `mlp.py` lines 111–116. The ordering `w1, w3, w2` matches the execution order
in `forward`, allowing the prefetcher to overlap DRAM loads with computation.

---

## 2. `MLP.forward` — SwiGLU activation path

```python
def forward(self, x: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
```
Source: `mlp.py` lines 118–324.

### 2.1 Compute kernel configuration

Op-level math fidelity is resolved per layer per operation group:
```python
li_ff1_3_compute_kernel_cfg = self.decoders_optimizations.get_math_fidelity(
    decoder_id=layer_num, op=OpGroup.LI_FF1_FF3, configuration=self.args
)
```
Default fidelity for MLP linear ops is `HIFI2_FP16` (HiFi2 with `fp32_dest_acc_en=False`
to save L1 memory). In performance mode, FF1/FF3 use `LOFI`. Source: `model_config.py`
lines 266–267.

### 2.2 Prefill reshape

When `mode == Mode.PREFILL` and `seq_len >= args.prefill_len_cutoff` (1024 on Wormhole,
512 on Blackhole), the input is reshaped to chunk it into segments for on-device
parallelism:
```python
x = ttnn.reshape(x, [1, seq_len // self.args.prefill_len_cutoff, self.args.prefill_len_cutoff, -1])
```
Source: `mlp.py` lines 135–137.

### 2.3 Gate and up projections (w1, w3)

Both projections use `ttnn.linear`:

```python
w1_out = ttnn.linear(x, self.w1, dtype=..., compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                     program_config=pc_1, memory_config=..., global_cb=..., sub_device_id=...)
w3_out = ttnn.linear(x, self.w3, dtype=..., compute_kernel_config=li_ff1_3_compute_kernel_cfg,
                     program_config=pc_3, memory_config=..., global_cb=..., sub_device_id=...)
```
Source: `mlp.py` lines 145–170. After both matmuls, the input `x` is deallocated
(`ttnn.deallocate(x)`, line 171).

On Galaxy, a reduce-scatter or all-reduce is applied to both `w1_out` and `w3_out`
across `cluster_axis=1` before the element-wise multiply. This reduces the partial sums
from the column-parallel split. Source: `mlp.py` lines 173–234.

### 2.4 SwiGLU activation — fused element-wise multiply

The SwiGLU non-linearity is computed via a single fused `ttnn.mul` call that applies the
activation function as a fused unary op on the first input:

```python
w2_in = ttnn.mul(
    w1_out,
    w3_out,
    input_tensor_a_activations=[self.activation_type],   # default: ttnn.UnaryOpType.SILU
    dtype=activation_dtype or ttnn.bfloat8_b,
    memory_config=w1_out.memory_config(),
)
```
Source: `mlp.py` lines 236–242.

`self.activation_type` defaults to `ttnn.UnaryOpType.SILU` but can be overridden via
`args.mlp_activation_type` if the attribute exists (`mlp.py` lines 104–106).

Both `w1_out` and `w3_out` are deallocated after the multiply (`mlp.py` lines 248–249), but
only after an optional `ttnn.to_memory_config` call at line 246 and before the optional
Galaxy `all_gather_async` block (lines 251–266) that operates on `w2_in`.

### 2.5 Down projection (w2)

```python
w2_out = ttnn.linear(
    w2_in,
    self.w2,
    compute_kernel_config=li_ff2_compute_kernel_cfg,
    dtype=self.args.ccl_dtype if TG else activation_dtype or ttnn.bfloat16,
    program_config=pc_2,
    memory_config=self.args.get_mlp_ff2_mem_config(mode, self.prefetcher),
    ...
)
```
Source: `mlp.py` lines 275–287.

### 2.6 All-reduce after down projection

The output of `w2` is partially summed across devices (row-parallel output) and must be
all-reduced:
```python
w2_out_reduced = tt_all_reduce(
    w2_out, self.mesh_device, self.tt_ccl,
    cluster_axis=0,
    dim=0 if (TG and self.dim < 8192) else 3,
    sharded=(mode == Mode.DECODE),
    ...
)
```
Source: `mlp.py` lines 290–311.

After the all-reduce, the tensor is reshaped to ensure dims 0 and 1 are 1:
```python
w2_out_reduced = ttnn.reshape(
    w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
)
```

### 2.7 Output memory config

In `Mode.DECODE`, the output is moved to the memory config from
`args.get_mlp_output_mem_config(mode, self.prefetcher)`. In `Mode.PREFILL`, no explicit
move is performed; the tensor is already in DRAM from the all-reduce.

### 2.8 Program configs and memory configs for activations

| Config helper | Decode | Prefill |
|---------------|--------|---------|
| `get_mlp_ff1_3_mem_config` | Prefetcher: L1 width-sharded per ring slice; else `L1_WIDTH_SHARDED` | `DRAM_MEMORY_CONFIG` |
| `get_mlp_ff2_mem_config` | Prefetcher: L1 width-sharded per ring slice; else `L1_WIDTH_SHARDED` | `DRAM_MEMORY_CONFIG` |
| `get_mlp_ff1_3_prg_config` | Galaxy: `matmul_1d_config`; prefetcher: ring config; else DRAM matmul config | `matmul_config` with prefill grid |
| `get_mlp_ff2_prg_config` | Same pattern as ff1_3 | `matmul_config` with prefill grid |

Source: `model_config.py` lines 1151–1290.

---

## 3. Norms inside `TransformerBlock`

Norms in TT Transformers are always `RMSNorm` objects (from `models/common/rmsnorm.py`)
wrapped by `DistributedNorm` (from `models/tt_transformers/tt/distributed_norm.py`). The
`DistributedNorm` wrapper is responsible for optional all-gather operations that
reconstruct a full tensor across devices when distributed norm is active.

### 3.1 Norm instances in `TransformerBlock`

| Attribute | State dict key | Applied when |
|-----------|---------------|--------------|
| `self.attention_norm` | `attention_norm` | Pre-attention |
| `self.ff_norm` | `ffn_norm` | Pre-feedforward (or post-attention residual add) |
| `self.pre_ff_norm` | `pre_feedforward_layernorm` | Additional pre-MLP norm (present only if key exists in state dict) |
| `self.post_ff_norm` | `post_feedforward_layernorm` | Post-MLP norm (present only if key exists in state dict) |

Source: `decoder.py` lines 100–200.

### 3.2 `RMSNorm` class

```python
class RMSNorm(LightweightModule):
    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        is_distributed=None,
        eps: float = 1e-05,
        add_unit_offset=False,
        sharded_program_config=None,
        sharded_output_config=None,
        output_mem_config=None,
        ccl_topology=ttnn.Topology.Ring,
        tt_ccl=None,
        fp32_dest_acc_en=True,
    ):
```
Source: `rmsnorm.py` lines 37–57.

#### Weight layout and dtype

| Property | Value |
|----------|-------|
| Layout | `ttnn.ROW_MAJOR_LAYOUT` |
| Dtype | `ttnn.bfloat16` (passed as `weight_dtype` from `TransformerBlock`) |
| Memory config | `ttnn.DRAM_MEMORY_CONFIG` (default `weight_memory_config`) |
| Mesh mapper | `ttnn.ReplicateTensorToMesh` — weight is replicated to every device |

The weight tensor is reshaped from `[dim]` to `[1, 1, dim // 32, 32]` before being
sent to device, where `32` is `TILE` / `SHARD_HEIGHT` (`rmsnorm.py` line 9).

#### Distributed variant

When `is_distributed(mode)` is true, the norm uses an additional `weight_distributed`
tensor sharded along `dim=2` using `ShardTensor2dMesh(device, dims=(None, 2), ...)`.
The forward pass in distributed mode calls `rms_norm_pre_all_gather`, performs an
async all-gather of the partial statistics, then calls `rms_norm_post_all_gather`.
Source: `rmsnorm.py` lines 94–203.

#### Compute kernel config

All RMSNorm ops use:
```python
ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=fp32_dest_acc_en,  # True by default; False for Qwen2.5-7B
    packer_l1_acc=True,
)
```
Source: `rmsnorm.py` lines 115–120.

### 3.3 `RMSNorm.forward`

```python
def forward(self, x: ttnn.Tensor, mode: Mode | str, in_sharded=False,
            out_sharded=False, norm_config=None) -> ttnn.Tensor:
```

- Non-distributed path: calls `ttnn.rms_norm(x, epsilon=..., weight=..., program_config=..., memory_config=..., compute_kernel_config=...)`.
- Distributed path: calls `_distributed_rmsnorm` which chains `ttnn.rms_norm_pre_all_gather` → async all-gather → `ttnn.rms_norm_post_all_gather`.
- If `in_sharded=True` and `out_sharded=False`, calls `ttnn.sharded_to_interleaved` after the norm.

Source: `rmsnorm.py` lines 122–168.

### 3.4 Norm application order in `TransformerBlock.forward`

The full execution order is:

1. `attn_in = self.attention_norm(x, mode, norm_config=...)` — norm before attention.
2. Attention forward; residual add.
3. `hidden_states = self.ff_norm(hidden_states, mode, norm_config=...)` — norm before MLP (standard) or pre-residual-add (when `pre_ff_norm` is present).
4. If `self.pre_ff_norm` is not None: residual add, then `hidden_states = self.pre_ff_norm(hidden_states, ...)`.
5. `hidden_states = self.feed_forward.forward(hidden_states, mode)` — MLP.
6. If `self.post_ff_norm` is not None: `hidden_states = self.post_ff_norm(hidden_states, ...)`.
7. Residual add.

Source: `decoder.py` lines 230–310.

### 3.5 Final output norm in `Transformer`

The top-level `Transformer` class wraps an `RMSNorm` with key `"norm"` (state dict key
`norm.weight`) inside `DistributedNorm` as `self.norm`. It is called after all decoder
layers and before the LM head:
```python
x = self.norm(x, mode=mode, norm_config=self.args.get_norm_config("lm_head", mode, self.prefetcher))
```
Source: `model.py` lines 108–127 (construction), line 638 (call in `forward`).
Weight dtype is `ttnn.bfloat16` (`model.py` line 117).

---

## Navigation

- [Index](index.md)
- Next: [Symbiote MLP](symbiote_mlp.md)
- [Integration Gaps](integration_gaps.md)
