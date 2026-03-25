# Chapter 3: Attention — RoPE, KV Cache, and SDPA

Attention is the hardest subsystem to integrate when porting TT Transformers into TT Symbiote because it is not a single operation — it is a pipeline of five interlocking concerns that must all be handled correctly and in concert.

## Why Attention Is Different

Every other subsystem in a transformer — embedding lookup, layer norm, MLP — maps cleanly to one or two ops. Attention does not. A single attention layer in TT Transformers orchestrates:

1. **Fused QKV weight sharding.** The Q, K, and V projections are merged into a single weight tensor (`wqkv`) that is 2D-sharded across the mesh before the model even runs. The sharding dimension differs between standard multi-device (`(2, 3)`) and TG Galaxy (`(3, 2)`).

2. **Rotary Position Embedding (RoPE).** TT Transformers pre-computes cosine/sine matrices at init time via `RotarySetup`, slices them with a position-index embedding lookup at runtime, and applies `ttnn.experimental.rotary_embedding_llama` with a pre-computed transformation matrix. The transformation matrix itself is sharded per-core. TT Symbiote's `TTNNRotaryPositionEmbedding` and `TTNNDistributedRotaryPositionEmbedding` use a different API surface and carry no `RotarySetup` equivalent.

3. **Paged KV cache.** Both codebases have paged KV cache objects, but they are wired up differently. TT Transformers' `Attention` allocates `layer_past` tensors directly in `init_kv_cache` and calls `ttnn.experimental.paged_update_cache` / `ttnn.experimental.paged_fill_cache` inline. TT Symbiote's `TTNNPagedAttentionKVCache` is a `transformers.Cache` subclass that provides `paged_fill_on_device`, `paged_update_on_device`, and `paged_sdpa_decode` methods, but it is not connected to a generation loop that drives those methods.

4. **Prefill / decode mode switching.** TT Transformers exposes two separate methods — `forward_prefill` and `forward_decode` — that use different program configs, different memory layouts, different `is_decode_mode` flags for RoPE, and different dtype coercions. TT Symbiote has no mode concept; `LlamaAttention.forward` runs a single path regardless of sequence length.

5. **Collective communication (CCL).** After the output projection (`wo`), TT Transformers calls `tt_all_gather` or `ttnn.experimental.all_gather_async` followed by a reduce-scatter through a `TT_CCL` wrapper object. TT Symbiote's sharded linear layers call `ttnn.experimental.reduce_scatter_minimal_async` directly; there is no unified CCL abstraction.

6. **TG (Galaxy) topology.** On 32-device meshes, TT Transformers applies additional `slice_mat` and `user_selection_matrix` projections to reshape batch dimensions across device groups. TT Symbiote has no equivalent.

## Chapter Contents

| File | Contents |
|------|----------|
| `symbiote_attention_overview.md` | Inventory of every attention class in `modules/attention.py` and `modules/rope.py`, including `TTNNPagedAttentionKVCache` and `PagedAttentionConfig` |
| `transformers_attention_overview.md` | Deep dive into `Attention.__init__`, per-layer dtype selection, compute kernel configs, sliding window, TG topology, and `RotarySetup` |
| `integration_gaps.md` | Four concrete gaps that must be closed to use TT Transformers attention from TT Symbiote |

## Prerequisites

- Chapter 1: Architecture Comparison (`../ch1_arch_comparison/`)
- Chapter 2: Weight Management (`../ch2_weight_management/`)
- Familiarity with `ttnn.transformer.scaled_dot_product_attention`, `ttnn.experimental.paged_update_cache`, and `ttnn.experimental.rotary_embedding_llama`

---

[Next: Symbiote Attention Overview](symbiote_attention_overview.md)
