# Normalization Layer Comparison: TT Transformers vs. TT Symbiote

This document compares every normalization layer used in the two stacks, maps each
TT Symbiote module to its TT Transformers counterpart, and documents the remaining
functional gaps.

---

## 1. TT Transformers normalization stack

### 1.1 `RMSNorm` (`models/common/rmsnorm.py`)

`RMSNorm` is a `LightweightModule` that loads, caches, and applies a single RMS
normalization weight.  Key constructor parameters:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `device` | `MeshDevice` or single device | Determines mesh vs. single-chip paths |
| `dim` | `int` | Hidden dimension; shapes the weight tensor |
| `state_dict` / `weight_key` / `state_dict_prefix` | dict / str / str | Resolves the weight key in the raw state dict |
| `weight_cache_path` | `Path \| None` | If set, tilized weight is cached on disk via `ttnn.as_tensor(cache_file_name=...)` |
| `weight_dtype` | `ttnn.DataType` | Weight dtype; callers pass `ttnn.bfloat16` |
| `add_unit_offset` | `bool` | **Gemma-style offset** — adds `1.0` to the raw weight before caching (see section 1.3) |
| `is_distributed` | callable | Per-mode predicate; controls whether the distributed path is used |
| `eps` | `float` | Epsilon for RMS denominator; default `1e-05` |
| `fp32_dest_acc_en` | `bool` | Passed to `WormholeComputeKernelConfig`; set `False` for Qwen2.5-7B (tracked in `decoder.py` comment, issue #35650) |

**Weight layout.** The raw `torch.Tensor` is reshaped to
`[1, 1, dim // 32, 32]` (ROW_MAJOR_LAYOUT) before being sent to device.  When
`is_distributed` is truthy, a second copy is uploaded with
`ttnn.ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape))`
as the mapper, producing `weight_distributed`.

**`forward` signature.**
```python
def forward(self, x, mode: Mode | str, in_sharded=False, out_sharded=False, norm_config=None)
```
The `norm_config` dict carries optional `sharded_program_config`,
`sharded_output_config`, and `output_mem_config`.  When `in_sharded=True` and
`mode == DECODE`, a sharded `LayerNormShardedMultiCoreProgramConfig` is used.

**Distributed path (inside `RMSNorm._distributed_rmsnorm`).**
```
ttnn.rms_norm_pre_all_gather(inp, ...)
  -> ttnn.experimental.all_gather_async(tt_stats, dim=3, topology=ccl_topology, ...)
  -> ttnn.rms_norm_post_all_gather(inp, tt_stats, ...)
```
The all-gather is performed on the stats tensor (not on the full activation), using
`TT_CCL` semaphore handles cycled from `self.tt_ccl`.

### 1.2 `DistributedNorm` (`models/tt_transformers/tt/distributed_norm.py`)

`DistributedNorm` wraps one `RMSNorm` instance and owns the multi-chip gather logic
that feeds into it.

**Constructor fields of interest.**

| Field | Purpose |
|-------|---------|
| `norm` | The inner `RMSNorm` |
| `enable_all_gather` | If `False`, the post-distributed-norm gather is suppressed; set to `False` on `ff_norm` when `pre_ff_norm` is present so that the output stays sharded |
| `ag_config_key` | Selects tuned CCL parameters from `model_config` (e.g. `"ATTN_LN_AG_CONFIG"`, `"FFN_LN_AG_CONFIG"`) |
| `TG` | Boolean; `True` selects the Galaxy-specific path |

**Non-TG `forward` flow.**

1. If `args.is_multichip` **and** `not args.is_distributed_norm(mode)`:
   all-gather activations across `dim=3` before passing to `RMSNorm`.
2. Call `self.norm(x, mode, in_sharded=..., out_sharded=..., norm_config=norm_config)`.
3. If `args.is_distributed_norm(mode)` **and** `enable_all_gather`:
   all-gather the norm output across `dim=3`.

The two all-gather sites serve different topologies:
- The *pre-norm* gather is used when the norm is **not** distributed (single-pass
  RMSNorm needs the full activation on each chip).
- The *post-norm* gather is used when the norm **is** distributed (distributed
  RMSNorm produces a sharded result; gather reconstructs the full tensor).

**TG path.** For Galaxy (32 devices), `forward` delegates entirely to
`tt_distributed_rmsnorm` (prefill) or `tt_sharded_distributed_rmsnorm` (decode)
from `models.tt_transformers.tt.ccl`.  These functions use pre-computed
`gather_in_mem_cfg`, `ln_prg_cfg`, and `ln_sharded_stats_memcfg` built in
`DistributedNorm.__init__` based on `args.dim // 4` (per-device hidden size for
a 4-column mesh).

### 1.3 `rms_norm_add_unit_offset` flag

Source: `decoder.py` and `model.py`, all `RMSNorm(...)` calls pass
`add_unit_offset=self.args.rms_norm_add_unit_offset`.

When this flag is `True`, `RMSNorm.__init__` executes:
```python
torch_weight = torch_weight + 1.0
```
**before** uploading the weight to device (and before caching it to disk).

**Purpose.** Some model families (most prominently Gemma) define their RMSNorm
weight as an offset from 1, i.e. the stored weight value represents `γ - 1` and
the true scale factor is `weight + 1`. Setting `add_unit_offset=True` folds this
offset at load time so that the standard `ttnn.rms_norm` formula
(`x / rms(x) * γ`) produces numerically correct output without any change to the
norm kernel itself.

**No Symbiote equivalent.** Neither `TTNNRMSNorm` nor `TTNNDistributedRMSNorm` in
`modules/normalization.py` expose this parameter.

### 1.4 Norm positions in `TransformerBlock`

`TransformerBlock` (`decoder.py`) instantiates up to four `DistributedNorm` objects
per layer:

| Attribute | `weight_key` | Used before | Always present |
|-----------|-------------|-------------|----------------|
| `attention_norm` | `attention_norm` | Attention | Yes |
| `ff_norm` | `ffn_norm` | MLP (after attn residual, or after attn output when `pre_ff_norm` is absent) | Yes |
| `pre_ff_norm` | `pre_feedforward_layernorm` | MLP gate input — Gemma 2 style | Only if key in `state_dict` |
| `post_ff_norm` | `post_feedforward_layernorm` | MLP output — Gemma 2 style | Only if key in `state_dict` |

When `pre_ff_norm` is present, `ff_norm.enable_all_gather` is set to `False` so
that `ff_norm`'s output stays sharded and feeds directly into the
`attention_out + ff_norm_out` residual add before `pre_ff_norm`.

The final output norm in `Transformer` (`model.py`) is also a `DistributedNorm`
wrapping an `RMSNorm` with `weight_key="norm"`.

---

## 2. TT Symbiote normalization stack

All three normalization modules live in
`models/experimental/tt_symbiote/modules/normalization.py`.

### 2.1 `TTNNLayerNorm`

Wraps `torch.nn.LayerNorm`.

| Aspect | Detail |
|--------|--------|
| Base class | `TTNNModule` |
| `from_torch` | Accepts `nn.LayerNorm`; guards against `None` weight and falls back to the PyTorch layer if absent |
| `preprocess_weights_impl` | Converts `weight` and `bias` to `ttnn.bfloat16` TILE_LAYOUT on host |
| `move_weights_to_device_impl` | `ttnn.to_device(self.tt_weight, self.device)` and same for `tt_bias` |
| `forward` | Ensures TILE_LAYOUT, calls `ttnn.layer_norm(input, weight=self.tt_weight, bias=self.tt_bias)` |
| eps | Not stored; `ttnn.layer_norm` uses its own default |
| Distributed | No |
| `add_unit_offset` | No |

### 2.2 `TTNNRMSNorm`

Wraps `DeepseekV2RMSNorm`, which is a standard `(x / rms(x)) * weight` formula
defined inline in the same file.

| Aspect | Detail |
|--------|--------|
| Base class | `TTNNModule` |
| `from_torch` | Accepts any object with a `.weight` attribute; type hint is `DeepseekV2RMSNorm` but the guard only checks `weight is None` |
| `preprocess_weights_impl` | Calls `weight.unsqueeze(0).expand(32, -1)` — expands to shape `[32, dim]` — then converts to `ttnn.bfloat16` TILE_LAYOUT on host |
| `move_weights_to_device_impl` | `ttnn.to_device(self.tt_weight, self.device)` |
| `forward` | Ensures TILE_LAYOUT, calls `ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.torch_layer.variance_epsilon)` |
| Compute kernel config | None passed; `ttnn.rms_norm` uses default |
| Distributed | No |
| `add_unit_offset` | No |

**Weight shape difference.** `TTNNRMSNorm` expands to `[32, dim]` (TILE_LAYOUT), while
`RMSNorm` in TT Transformers reshapes to `[1, 1, dim // 32, 32]` (ROW_MAJOR_LAYOUT).
Both represent the same vector; they differ only in layout and how the host tensor is
arranged before upload.

### 2.3 `TTNNDistributedRMSNorm`

Performs a two-phase RMS norm with an all-gather between phases.

| Aspect | Detail |
|--------|--------|
| Base class | `TTNNModule` |
| Decorator | `@trace_enabled` (class-level) |
| `from_torch` | Same guard pattern as `TTNNRMSNorm` |
| `preprocess_weights_impl` | Not overridden; inherits the no-op from `TTNNModule` |
| `move_weights_to_device_impl` | Builds `weight_distributed` via `ttnn.as_tensor` with `ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape))` |
| `forward` decorator | `@run_on_devices(DeviceArch.T3K)` — raises `RuntimeError` on any other architecture |
| `forward` flow | `rms_norm_pre_all_gather` → `all_gather_async(dim=-1, topology=ttnn.Topology.Linear)` → `rms_norm_post_all_gather` |

---

## 3. Cross-stack mapping

| TT Transformers component | TT Symbiote equivalent | Notes |
|--------------------------|------------------------|-------|
| `RMSNorm` (non-distributed) | `TTNNRMSNorm` | Maps correctly for standard LLaMA-style models; no `HiFi2` compute config in Symbiote |
| `RMSNorm` (distributed path via `_distributed_rmsnorm`) | `TTNNDistributedRMSNorm` | Both split pre/post all-gather; see gap table in section 4 |
| `DistributedNorm` (wrapper) | No direct equivalent | Symbiote has no wrapper that owns the pre-norm or post-norm gather decision; user must wire this manually |
| `nn.LayerNorm` (not used in LLaMA/Llama-family decoders) | `TTNNLayerNorm` | Used in encoder-style models; not relevant to decoder block assembly |
| `RMSNorm` with `add_unit_offset=True` | Not implemented | Gap; see section 4.1 |
| `DistributedNorm` TG path | Not implemented | Gap; see section 4.3 |

---

## 4. Gaps

### 4.1 `rms_norm_add_unit_offset` (Gemma-style)

**TT Symbiote:** Neither `TTNNRMSNorm` nor `TTNNDistributedRMSNorm` accepts this
parameter.  A user running a Gemma checkpoint through Symbiote normalization modules
will obtain incorrect norm outputs because the weight will be used as stored (i.e.
`γ - 1`) rather than as the intended scale (`γ`).

**What to add:** An `add_unit_offset: bool = False` parameter in
`preprocess_weights_impl` (or in `from_torch` / `__init__`) that applies
`weight = weight + 1.0` before the TTNN conversion.

### 4.2 `all_gather_async` parameter differences

Both stacks call `ttnn.experimental.all_gather_async` on the stats tensor.  The
calling conventions differ in several ways that affect performance and correctness on
T3K:

| Parameter | TT Transformers `RMSNorm._distributed_rmsnorm` | `TTNNDistributedRMSNorm.forward` |
|-----------|-----------------------------------------------|----------------------------------|
| `dim` | `3` | `-1` (equivalent on 4-D tensors, same physical axis) |
| `topology` | `self.ccl_topology` (configurable; `Ring` or `Linear` per model) | `ttnn.Topology.Linear` (hardcoded) |
| `num_links` | `1` | `1` |
| `chunks_per_sync` | `10` | not passed (default) |
| `num_workers_per_link` | `2` | not passed (default) |
| `num_buffers_per_channel` | `2` | not passed (default) |
| `memory_config` | `ttnn.DRAM_MEMORY_CONFIG` | not passed (default) |
| Semaphore source | `self.tt_ccl.get_and_cycle_ag_semaphore_handles()` and `...barrier...` from `TT_CCL` | `self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1)` — cluster axis `1` hardcoded |
| `persistent_output_buffer` | `None` (explicit) | not passed |

The hardcoded `topology=ttnn.Topology.Linear` in `TTNNDistributedRMSNorm` is
adequate for T3K but would need to be made configurable for Ring or other topologies.
The cluster-axis-`1` hardcoding in the semaphore call also diverges from TT
Transformers, where the axis is implicit (the `TT_CCL` object handles this
internally without an axis argument).

### 4.3 No TG (Galaxy) path in `TTNNDistributedRMSNorm`

`TTNNDistributedRMSNorm.forward` is decorated with `@run_on_devices(DeviceArch.T3K)`
and will raise `RuntimeError` on Galaxy.  TT Transformers' `DistributedNorm`
executes `tt_sharded_distributed_rmsnorm` / `tt_distributed_rmsnorm` (from
`tt_transformers.tt.ccl`) on the TG path, using a pre-built `ln_prg_cfg` and
sharded memory configs.  No Symbiote equivalent exists for TG.

### 4.4 No `DistributedNorm` wrapper in Symbiote

TT Transformers' `DistributedNorm` owns the decision of *when* to all-gather: before
the norm (for non-distributed single-pass norm on multi-chip) or after (for
distributed two-phase norm).  It also controls `enable_all_gather` per site to keep
output sharded when a subsequent `pre_ff_norm` is present.

Symbiote has no equivalent wrapper.  The two existing classes (`TTNNRMSNorm` for
non-distributed, `TTNNDistributedRMSNorm` for distributed) never all-gather
activations before entering the norm.  A user who wants multi-chip-correct norms
must manually insert `ttnn.experimental.all_gather_async` calls at the right points
in the block's forward pass.

### 4.5 No `HiFi2` compute kernel config in `TTNNRMSNorm`

`RMSNorm` in TT Transformers always passes a `WormholeComputeKernelConfig` with
`MathFidelity.HiFi2` to `ttnn.rms_norm`.  `TTNNRMSNorm.forward` calls
`ttnn.rms_norm` without a `compute_kernel_config`, so TTNN selects its default
fidelity (LoFi on Wormhole).  This may produce lower-accuracy outputs in sensitive
layers.

### 4.6 `pre_feedforward_layernorm` / `post_feedforward_layernorm` positions

The two extra norm sites (Gemma 2 style) are not modelled in Symbiote at all; there
is no Symbiote counterpart for them, and assembling a Gemma 2 decoder with Symbiote
modules requires implementing and wiring both manually.

---

## 5. Summary table

| Feature | TT Transformers | TT Symbiote | Gap |
|---------|-----------------|-------------|-----|
| Standard RMSNorm | `RMSNorm` | `TTNNRMSNorm` | Missing `HiFi2` compute config; weight expand strategy differs |
| Distributed two-phase RMSNorm | `RMSNorm._distributed_rmsnorm` | `TTNNDistributedRMSNorm` | T3K only; CCL params differ; no TG path |
| Gather-or-norm wrapper | `DistributedNorm` | None | Full gap — user must wire manually |
| LayerNorm | `nn.LayerNorm` (PyTorch) | `TTNNLayerNorm` | No bias guard on device; no eps storage |
| `add_unit_offset` (Gemma) | Supported | Not supported | Full gap |
| TG distributed norm | `tt_sharded_distributed_rmsnorm` | Not supported | Full gap |
| `pre_ff_norm` / `post_ff_norm` | Supported | Not supported | Full gap |
| Weight caching | Via `cache_file_name` | Not supported | Full gap |

---

## Navigation

- [Chapter 4 index](index.md)
- Previous: [Integration Gaps](integration_gaps.md)
- Next: [Decoder Block Assembly](decoder_block_assembly.md)
- Guide root: [TT Transformers into TT Symbiote](../plan.md)
