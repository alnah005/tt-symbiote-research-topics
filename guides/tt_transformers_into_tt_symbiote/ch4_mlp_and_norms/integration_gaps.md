# Integration Gaps: TT Transformers MLP vs. TT Symbiote

This document compares TT Transformers' native `MLP` class with the linear layer modules
provided by TT Symbiote, and identifies what a Symbiote user must implement themselves
to reach equivalent functionality.

---

## 1. High-level comparison

| Capability | TT Transformers `MLP` | TT Symbiote (`TTNNLinear*` family) |
|------------|-----------------------|-------------------------------------|
| Three-weight SwiGLU MLP as a single module | Yes — `MLP` encapsulates `w1`, `w2`, `w3` | No — each projection is a separate module; user must compose |
| Fused SwiGLU activation (`silu(w1_out) * w3_out` in one op) | Yes — `ttnn.mul(..., input_tensor_a_activations=[SILU])` | No — `TTNNLinearSilu` is a sequential linear+silu, not a gated multiply |
| Per-layer per-tensor dtype control (`TensorGroup` / `DecodersPrecision`) | Yes — fully controlled; BFP4/BFP8/BF16 per decoder, per tensor group | No — dtype is fixed per class (`bfloat16` or `bfloat8_b`) |
| Per-layer per-op math fidelity control (`OpGroup`) | Yes — LOFI/HiFi2/HiFi4 configurable per decoder and op | No |
| DRAM width-sharded weight memory configs for decode | Yes — `create_dram_sharded_mem_config` used for `w1`, `w2`, `w3` | No — all classes use `ttnn.DRAM_MEMORY_CONFIG` (interleaved) |
| L1 width-sharded activation memory configs for decode | Yes — `get_mlp_ff1_3_mem_config`, `get_mlp_ff2_mem_config` | No — `DRAM_MEMORY_CONFIG` hardcoded in all linear `forward` methods |
| DRAM-sharded matmul program configs | Yes — `get_mlp_ff1_3_prg_config`, `get_mlp_ff2_prg_config` | No — no program config is passed to `ttnn.linear` |
| Prefetcher integration (DRAM prefetch ring) | Yes — ring config, global CB, sub-device ID | No |
| Multi-device all-reduce after down projection (`w2`) | Yes — `tt_all_reduce` on `cluster_axis=0` | No — not present in any Symbiote linear class |
| Reduce-scatter after gate/up projections (`w1`, `w3`) | Yes — for Galaxy topology | Partial — `TTNNLinearIColShardedWRowSharded` does a reduce-scatter, but after its single matmul, not after a gate+up pair |
| Support for Galaxy (32-device TG) topology | Yes — explicit `is_galaxy` branches throughout | No — `TTNNLinearIColShardedWRowSharded` and `TTNNLinearLLamaIColShardedWRowSharded` are restricted to `DeviceArch.T3K` only |
| Weight padding for hidden dim alignment | Yes — `pad_to_size` / `pad_hidden_dim` applied before `as_tensor` | No |
| Weight caching to disk | Yes — `cache_file_name` parameter to `ttnn.as_tensor` | No |
| RMSNorm module (pre/post MLP) | Yes — `RMSNorm` / `DistributedNorm` included in `TransformerBlock` | Partial — `TTNNRMSNorm` exists in `modules/normalization.py` but uses `TILE_LAYOUT` weight (not `ROW_MAJOR_LAYOUT`) and omits `HiFi2` compute kernel config; no `DistributedNorm` wrapper |
| Distributed RMSNorm (pre-all-gather / post-all-gather split) | Yes — `_distributed_rmsnorm` in `rmsnorm.py` | Partial — `TTNNDistributedRMSNorm` exists in `modules/normalization.py`; T3K only; CCL params differ (topology hardcoded to `Linear`; no chunks_per_sync/num_workers/num_buffers_per_channel) |
| Configurable `add_unit_offset` on norm weights | Yes — `args.rms_norm_add_unit_offset` | No |
| `pre_feedforward_layernorm` / `post_feedforward_layernorm` support | Yes — conditional instantiation in `TransformerBlock` | No |
| `ttnn.TILE_LAYOUT` enforcement before matmul | Yes — implicit via weight preprocessing and program config | Partial — `TTNNLinear.forward` converts input to TILE layout on-the-fly; no program config to enforce efficient tiling |
| TTNN tracing compatibility | Selective (`@trace_enabled` / `@trace_disabled` decorators) | Same mechanism; `TTNNLinear` is `@trace_enabled`; `LLama*` variants are `@trace_disabled` |

---

## 2. Detailed gap analysis

### 2.1 No composed SwiGLU MLP class

**TT Symbiote:** No equivalent composite class exists. A user must create three
separate linear modules, manually call them in the correct order, manually implement
the element-wise gated multiply, and manage intermediate tensor lifetimes including
`ttnn.deallocate` calls.

**What to implement:** A `SwiGLUMLP` module that composes `gate_proj`, `up_proj`, and
`down_proj` instances and implements the fused gate-and-multiply in `forward`.

### 2.2 No fused gated activation

**TT Symbiote:** `TTNNLinearSilu` calls `ttnn.linear` then `ttnn.silu` sequentially.
There is no class or helper that produces the `silu(gate) * value` fused pattern.

**What to implement:** A helper function or `forward` step that calls
`ttnn.mul(..., input_tensor_a_activations=[...])` after separately computing the gate
and value branch outputs.

### 2.3 No DRAM width-sharded weight placement

**TT Transformers:** `as_sharded_tensor` places `w1`, `w2`, `w3` into DRAM with a
`WIDTH_SHARDED` memory config. Each weight is spread across all DRAM banks, enabling
the DRAM-sharded matmul kernel to parallelize reads. This is the primary mechanism for
efficient decode-mode throughput.

**TT Symbiote:** All classes call `ttnn.to_device(self.tt_weight_host, self.device)`,
which places the tensor in interleaved DRAM. No `MemoryConfig` with
`TensorMemoryLayout.WIDTH_SHARDED` and a `ShardSpec` is ever constructed.

**What to implement:** Override `move_weights_to_device_impl` to call
`create_dram_sharded_mem_config` (or construct the equivalent `MemoryConfig` manually)
and pass it to `ttnn.as_tensor` or `ttnn.to_device`. The shard dimensions must match
the matmul kernel's expectations for decode.

### 2.4 No program configs for matmuls

**TT Transformers:** `ttnn.linear` is always called with an explicit `program_config`
from `get_mlp_ff1_3_prg_config` or `get_mlp_ff2_prg_config`. These configs specify
the core grid, tiling, subblock shapes, and whether a DRAM-sharded or 1D ring path
is used.

**TT Symbiote:** `ttnn.linear(input_tensor, self.tt_weight, ..., memory_config=ttnn.DRAM_MEMORY_CONFIG)`
— no `program_config` is passed. TTNN will select a default program, which may not
match the layout assumptions implied by DRAM-sharded weights.

**What to implement:** Accept a `program_config` argument (or a config factory) in
each linear class's `forward` method, and pass it to `ttnn.linear`. For decode-mode
performance this is non-trivial and requires tuning `MatmulMultiCoreReuseProgramConfig`
or the 1D ring variant to match the weight's DRAM sharding.

### 2.5 No L1-sharded activation memory configs

**TT Transformers:** For decode mode, activations (outputs of `w1`, `w3`, `w2`) are
placed in L1 width-sharded memory (`L1_WIDTH_SHARDED_MEMORY_CONFIG` or a per-core
sharded config). This keeps intermediate tensors on-chip between operations.

**TT Symbiote:** All activations use `ttnn.DRAM_MEMORY_CONFIG`. Every intermediate
tensor is written to DRAM and re-read, reducing throughput.

**What to implement:** Expose `memory_config` parameters for outputs in each linear
`forward` method, or build a mode-aware wrapper that selects DRAM for prefill and
L1-sharded for decode.

### 2.6 No multi-device all-reduce after down projection

**TT Transformers:** After `w2_out = ttnn.linear(w2_in, self.w2, ...)`, an all-reduce
is performed across `cluster_axis=0` to sum the row-parallel partial results:
```python
w2_out_reduced = tt_all_reduce(w2_out, ...)
```

**TT Symbiote:** `TTNNLinearIColShardedWRowSharded` does a reduce-scatter after its
single matmul (appropriate for column-parallel output), but there is no class that
performs an all-reduce on `cluster_axis=0` as needed for the down projection. The user
must call `tt_all_reduce` (from `models.tt_transformers.tt.ccl`) manually after calling
the down-projection linear.

### 2.7 No Galaxy topology support

**TT Transformers:** The `MLP` class has explicit branches for `args.is_galaxy` (32
devices), including different sharding dimensions (`w1_dims`, `w2_dims`), TG-specific
memory configs (`FF1_OUT_REDUCE_SCATTER_MEMCFG`, `FF1_OUT_GATHERED_MEMCFG`), and
`all_gather_async` calls on `w2_in` (after the `ttnn.mul` SwiGLU step, before the `w2`
down-projection matmul) — not between w1/w3 outputs and the element-wise multiply.

**TT Symbiote:** `TTNNLinearIColShardedWRowSharded` and
`TTNNLinearLLamaIColShardedWRowSharded` are decorated with
`@run_on_devices(DeviceArch.T3K)` and will raise `RuntimeError` on any other device
architecture, including Galaxy. No Symbiote linear class supports the Galaxy topology.

**What to implement:** Galaxy-aware MLP linear classes (or topology-dispatch logic)
that select correct sharding dimensions and collective operations for `DeviceArch.TG`.

### 2.8 Partial RMSNorm coverage — functional gaps remain

**TT Transformers:** `RMSNorm` and `DistributedNorm` are standalone modules in
`models/common/rmsnorm.py` and `models/tt_transformers/tt/distributed_norm.py`. They
handle weight loading, caching, distributed pre/post all-gather, sharded and
non-sharded execution paths, and `HiFi2` compute kernel config.

**TT Symbiote:** `TTNNRMSNorm`, `TTNNDistributedRMSNorm`, and `TTNNLayerNorm` exist in
`models/experimental/tt_symbiote/modules/normalization.py`. However, they have
functional gaps relative to TT Transformers (see `normalization_comparison.md` section 4
for the full gap inventory):

- `TTNNRMSNorm` uses `TILE_LAYOUT` for the weight (TT Transformers uses `ROW_MAJOR_LAYOUT`)
  and passes no `compute_kernel_config` to `ttnn.rms_norm` (TT Transformers always passes
  `HiFi2`).
- `TTNNDistributedRMSNorm` is restricted to `DeviceArch.T3K` via `@run_on_devices` and
  hardcodes `topology=ttnn.Topology.Linear`; TT Transformers' topology is configurable
  per-model.
- Neither class supports `add_unit_offset` (needed for Gemma-family checkpoints).
- No `DistributedNorm` wrapper owns the gather-before-norm / gather-after-norm decision.

**What to add:** See `normalization_comparison.md` sections 4.1–4.6 for a complete list
of gaps and remediation steps.

### 2.9 No per-decoder precision / fidelity control

**TT Transformers:** `ModelOptimizations` / `DecodersPrecision` provide layer-by-layer
control of weight dtypes (`BFP4`, `BFP8`, `BF16`) and math fidelity (`LOFI`, `HIFI2`,
`HIFI4`) for every operator group in every decoder layer. This is configurable via JSON
(`performance_decoder_config.json`, `accuracy_decoder_config.json`) or programmatically.

**TT Symbiote:** Dtype is fixed at the class level (`bfloat16` for base classes,
`bfloat8_b` for `LLama*` variants). There is no per-layer override mechanism.

**What to implement:** A configuration injection mechanism — either accept a
`dtype` parameter in constructors and `forward`, or build a `DecoderConfig` object
analogous to `ModelOptimizations` that can be attached to a Symbiote module tree.

### 2.10 No weight caching

**TT Transformers:** The `cache_file_name` parameter to `ttnn.as_tensor` enables
on-disk caching of tilized/sharded weights. Subsequent runs skip the preprocessing
and load directly from the cache file.

**TT Symbiote:** No caching is performed. `ttnn.to_device(self.tt_weight_host, ...)` is
called without a cache path each time.

**What to implement:** Accept an optional `cache_path` in `move_weights_to_device_impl`
and use `ttnn.as_tensor(..., cache_file_name=cache_path)` instead of `ttnn.to_device`.

---

## 3. Summary: minimum work to match TT Transformers MLP performance

The following are the highest-impact items, in descending order of performance impact:

1. **DRAM width-sharded weights + matching program configs** — without these, decode
   throughput will not use the DRAM-sharded matmul kernel path and will be significantly
   slower.

2. **Fused SwiGLU activation** — replacing sequential `linear + silu + mul` with a
   single `ttnn.mul(..., input_tensor_a_activations=[SILU])` saves one kernel launch
   and one round-trip through DRAM for the intermediate.

3. **L1 activation memory configs for decode** — keeping intermediates in L1 between
   matmuls avoids unnecessary DRAM traffic.

4. **All-reduce after down projection** — required for correctness on multi-device
   deployments; without it the output is a partial sum only.

5. **Per-layer dtype and fidelity control** — needed to reproduce performance/accuracy
   trade-offs selectable in TT Transformers.

6. **RMSNorm module** — required for any complete transformer layer; partial
   implementations exist (`TTNNRMSNorm`, `TTNNDistributedRMSNorm` in
   `modules/normalization.py`) but have functional gaps (weight layout, missing
   `HiFi2` config, T3K restriction, no `add_unit_offset`). See section 2.8 for details.

7. **Galaxy support** — required for 32-device deployments; involves different sharding
   dimensions, an `all_gather_async` on `w2_in` (after the SwiGLU multiply, before the
   `w2` matmul), and TG-specific memory configs.

---

**Next:** [Chapter 5 — Multi-Chip Parallelism and Collective Communication](../ch5_multi_chip/index.md)
