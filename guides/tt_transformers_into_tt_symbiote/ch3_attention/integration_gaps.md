# Attention Integration Gaps

This document identifies the four concrete gaps between TT Symbiote's attention subsystem and TT Transformers' `Attention` class. Each gap must be closed to run TT Transformers attention from within a TT Symbiote model graph.

---

## Gap 1: Weight Construction — `state_dict` and `weight_cache_path` vs. `from_torch`

### The Problem

TT Transformers' `Attention.__init__` requires two arguments that TT Symbiote has no concept of:

- `state_dict: dict[str, torch.Tensor]` — the raw model checkpoint from which weights are extracted by string key.
- `weight_cache_path: Path | None` — a filesystem path to a directory of pre-serialized TTNN tensors.

Inside `__init__`, every weight (QKV, output projection, optional Q/K norms, KV cache) is constructed by indexing directly into `state_dict`:

```python
wq_str = f"{layer_name}.wq"
wq_selected = torch.chunk(state_dict[f"{wq_str}.weight"], self.num_devices_per_group, dim=0)[i]
```

The weight cache is used to skip the chunk/transpose/concat pipeline on subsequent runs:

```python
cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")
self.wqkv = ttnn.as_tensor(..., cache_file_name=cache_name("wqkv_sharded_2d"))
```

TT Symbiote's `LlamaAttention.from_torch` takes a fully instantiated HuggingFace `LlamaAttention` module and reads weights from its attributes (`llama_attn.q_proj.weight`, etc.). It performs no device-side sharding, no weight transposition into the combined QKV layout, and no caching.

### What Needs to Happen

To use TT Transformers `Attention` from TT Symbiote, the caller must:

1. Have access to a complete `state_dict` (or a remapped version thereof) with the exact key prefixes that TT Transformers' `configuration.get_state_dict_prefix` produces.
2. Either provide a valid `weight_cache_path` or pass `None` and accept the latency of re-converting weights on every run.
3. Ensure that HuggingFace-format checkpoint keys are remapped to meta-format keys (or whichever format `configuration` expects) before passing to `Attention.__init__`.

TT Transformers provides `convert_hf_to_meta` in `load_checkpoints.py` for this purpose, but TT Symbiote has no equivalent integration point.

### Impact

Without resolving this gap, TT Transformers `Attention` cannot be instantiated at all from a TT Symbiote model graph. This is a blocking prerequisite, not an optimization.

---

## Gap 2: Prefill vs. Decode Mode Switching

### The Problem

TT Transformers exposes two separate forward methods on `Attention`:

| Method | Input shape | RoPE mode | Head op | SDPA op |
|--------|-------------|-----------|---------|---------|
| `forward_decode` | `(seq_len, 1, batch, dim)` | `is_decode_mode=True`, `transformation_mats["decode"]` | `nlp_create_qkv_heads_decode` | `paged_scaled_dot_product_attention_decode` |
| `forward_prefill` | `(1, 1, seq_len, dim)` | `is_decode_mode=False`, `transformation_mats["prefill"]` | `nlp_create_qkv_heads` | `scaled_dot_product_attention` |

These two paths differ not only in which ops are called but in the memory layouts of intermediate tensors, the sharding strategies, and the dtype casts applied before the KV cache write:

- Prefill requires `seq_len % 128 == 0`.
- Prefill reshapes the input to `[1, seq_len // MAX_QKV_MM_SEQ_LEN, MAX_QKV_MM_SEQ_LEN, dim]` if the sequence exceeds `MAX_QKV_MM_SEQ_LEN`.
- Decode calls `nlp_concat_heads_decode` (which returns a `[1, 1, batch, heads * head_dim]` tensor), while prefill calls `nlp_concat_heads` (returning `[1, 1, seq_len, dim]`).

TT Symbiote's `LlamaAttention.forward` is a single unified path. It does not distinguish prefill from decode, does not use `nlp_create_qkv_heads_decode` or `nlp_concat_heads_decode`, and always calls `ttnn.transformer.scaled_dot_product_attention` (the prefill variant). There is no `Mode` enum in TT Symbiote.

### What Needs to Happen

A mode-switching mechanism must be introduced at the TT Symbiote call site. The options are:

1. **Explicit mode parameter.** Add a `mode: Mode` argument to `LlamaAttention.forward` (or whatever wraps TT Transformers `Attention`) and dispatch to `forward_prefill` or `forward_decode` based on its value.
2. **Sequence-length detection.** Infer mode from `hidden_states.shape[-2] == 1` at runtime. This is fragile because it breaks batched prefill of length-1 sequences and chunked prefill.
3. **Two separate module handles.** Expose `forward_prefill` and `forward_decode` as separate callable entry points from the Symbiote model graph, selected by the generation loop.

Option 3 most closely matches how TT Transformers itself is called (from `Generator`, which explicitly calls `forward_decode` or `forward_prefill`).

### Impact

Without mode switching, using TT Transformers `Attention` in a Symbiote context will result in incorrect outputs during decode (wrong ops, wrong tensor layouts) or will fail with shape assertion errors (`Seqlen must be divisible by 128` in `forward_prefill` when called for decode).

---

## Gap 3: Collective Communication — `TT_CCL` vs. Direct `reduce_scatter_minimal_async`

### The Problem

TT Transformers passes a `tt_ccl` object into every `Attention` layer. All collective communication — all-gather and all-reduce across the mesh — goes through this object:

```python
# Decode: all-reduce after QKV matmul
xqkv_fused = tt_all_reduce(
    xqkv_fused_sharded,
    self.mesh_device,
    self.tt_ccl,
    cluster_axis=1,
    ...
)

# Decode: all-gather before output projection (non-TG ring path)
all_gather_output = ttnn.experimental.all_gather_async(
    attn_output_cat,
    ...
    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
    ...
)
```

`TT_CCL` manages semaphore handles that must be allocated once and cycled across calls. `tt_all_reduce` and `tt_all_gather` are wrappers in `models/tt_transformers/tt/ccl.py` that dispatch to the appropriate low-level async collectives.

TT Symbiote's sharded linear layers (`TTNNLinearIColShardedWRowSharded`, `TTNNLinearIReplicatedWColSharded`) call `ttnn.experimental.reduce_scatter_minimal_async` directly. There is no `TT_CCL` object in TT Symbiote and no semaphore cycling infrastructure.

### What Needs to Happen

To use TT Transformers `Attention`, a `TT_CCL` instance must be constructed and passed in. `TT_CCL` is defined in `models/tt_transformers/tt/ccl.py` and requires:

- A `mesh_device`
- A topology (ring or linear, from `configuration.ccl_topology()`)
- An optional number of semaphores per direction

The TT Symbiote model-level code would need to construct this object once (likely at the same level where `mesh_device` is constructed) and thread it through to every attention layer that uses it.

The existing Symbiote sharded linear layers cannot be straightforwardly replaced by TT Transformers CCL calls because they use different collective strategies (reduce-scatter vs. all-reduce). The architectural difference is:

| Symbiote sharded linear | TT Transformers attention |
|------------------------|--------------------------|
| Column-parallel: input replicated, weight column-sharded; output reduced via `reduce_scatter` | Weight row-sharded; output gathered via `all_gather` then matmul, or matmul then `all_reduce` |

These are equivalent in mathematical output but require different tensor layouts entering and exiting the linear op.

### Impact

Without `TT_CCL`, TT Transformers `Attention` will fail at the first collective op inside `forward_decode`. The fix requires instantiation of `TT_CCL` and a decision about whether TT Symbiote's existing linear CCL strategy should be unified with TT Transformers'.

---

## Gap 4: Paged KV Cache — `TTNNPagedAttentionKVCache` Not Wired into a Generation Loop

### The Problem

TT Symbiote defines `TTNNPagedAttentionKVCache` (a `Cache` subclass in `modules/attention.py`) with all the TTNN infrastructure needed for paged attention:

```python
def paged_fill_on_device(self, key_states, value_states, layer_idx, batch_idx=0): ...
def paged_update_on_device(self, key_states, value_states, layer_idx, current_pos): ...
def paged_sdpa_decode(self, query, layer_idx, current_pos, scale, ...): ...
```

However, these methods are never called by TT Symbiote's standard path. The `update` method — the one that HuggingFace's `generate()` pipeline calls — converts tensors back to host and returns them:

```python
def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
    # Converts ttnn/TorchTTNNTensor inputs to torch.Tensor
    return key_states, value_states  # host tensors, no paged op
```

This means passing `TTNNPagedAttentionKVCache` to `LlamaAttention.forward` via `past_key_values` exercises none of the TTNN paged cache logic.

TT Transformers, by contrast, drives paged cache operations explicitly from a `Generator` class (or equivalent model-level loop) that:

1. Calls `forward_prefill` with `page_table` and `paged_attention_config` for the initial fill.
2. Calls `forward_decode` with `page_table` and `current_pos` for each decode step.
3. Passes these arguments through to `ttnn.experimental.paged_fill_cache` and `ttnn.experimental.paged_update_cache` inside `Attention`.

### What Needs to Happen

To make `TTNNPagedAttentionKVCache` actually functional, a new generation loop is needed that:

1. Constructs a `TTNNPagedAttentionKVCache` and calls `to_device` before the first forward pass.
2. During prefill, calls `paged_fill_on_device` for each layer (or integrates the fill into the model forward via `page_table` argument, as TT Transformers does).
3. During decode, calls `paged_update_on_device` for each layer and passes `current_pos` as a `ttnn.Tensor`.
4. Uses `paged_sdpa_decode` as the attention kernel rather than `TTNNSDPAAttention`.

This requires a `Generator`-like class analogous to `models/tt_transformers/tt/generator.py` that drives the attention layer with explicit page tables and position tensors, rather than relying on HuggingFace's abstract `Cache.update` interface.

> **Note:** The `_PagedCacheLayer` helper class in `modules/attention.py` is a minimal stub that satisfies HuggingFace's `CacheLayerMixin` interface with no-ops. Its presence confirms that full HuggingFace `Cache` integration was intentionally deferred.

### Impact

Without a dedicated generation loop, all TTNN paged attention infrastructure in TT Symbiote is unreachable dead code. The paged cache objects can be constructed and moved to device, but no token generation path calls the paged ops.

---

## Gap Summary

| Gap | Blocking? | Complexity | Primary Change Required |
|-----|-----------|------------|------------------------|
| 1 — Weight construction (`state_dict` / `weight_cache_path`) | Yes | Medium | Map HF checkpoint keys to TT Transformers format; thread `state_dict` and cache path through Symbiote model init |
| 2 — Prefill / decode mode switching | Yes | Medium | Introduce `Mode` enum or mode parameter at the Symbiote model level; dispatch to `forward_prefill` or `forward_decode` |
| 3 — `TT_CCL` collective communication | Yes | High | Instantiate `TT_CCL` once per mesh; unify CCL strategy between Symbiote sharded linears and TT Transformers attention collectives |
| 4 — Paged KV cache wiring | No (for correctness), Yes (for performance) | High | Implement a `Generator`-like generation loop that drives prefill fill and decode update via explicit paged ops, bypassing HuggingFace `Cache.update` |

Gap 4 does not prevent functional correctness (the HuggingFace `Cache.update` fallback returns correct tensors from host) but eliminates the paged TTNN acceleration entirely.

---

[Previous: TT Transformers Attention Overview](transformers_attention_overview.md) | [Chapter Index](index.md)
