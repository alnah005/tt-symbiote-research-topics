# Comparison with TT-Symbiote Attention

## Prerequisites

- [Chapter 4 -- Joint Attention](./joint_attention.md): TT-DiT's `Attention` class with joint SDPA, per-head norm, and RoPE.
- [Chapter 4 -- Transformer Block](./transformer_block.md): TT-DiT's `TransformerBlock` with adaptive layer normalization and gated residuals.
- [Chapter 1 -- Comparison with TTNNModule](../ch1_architecture_overview/comparison_with_ttnnmodule.md): fundamental differences between TT-DiT's `Module` and TT-Symbiote's `TTNNModule`.

---

## Overview

TT-Symbiote's attention module hierarchy in `modules/attention.py` provides several attention implementations designed for different model architectures: `TTNNSelfAttention` for ViT-style models, `LlamaAttention` for decoder-only LLMs, `TTNNWhisperAttention` for encoder-decoder models, and `TTNNGR00TSelfAttention` for robotics foundation models. All of these share a common underlying `TTNNSDPAAttention` kernel wrapper.

This file compares TT-Symbiote's attention with TT-DiT's `Attention` class across every architectural dimension, identifying the gaps that a porting effort must address.

---

## Attention Class Hierarchy

### TT-DiT

```
Module
  └── Attention (blocks/attention.py)
        ├── to_qkv: ColParallelLinear (fused spatial Q/K/V)
        ├── add_qkv_proj: ColParallelLinear (fused prompt Q/K/V)
        ├── norm_q, norm_k: RMSNorm (per-head, on spatial)
        ├── norm_added_q, norm_added_k: RMSNorm (per-head, on prompt)
        ├── to_out: ColParallelLinear (spatial output projection)
        ├── to_add_out: ColParallelLinear (prompt output projection)
        └── context_head_factors: Parameter (optional per-head scaling)
```

Single class, handles all DiT models via constructor parameters.

### TT-Symbiote

```
TTNNModule
  ├── TTNNSDPAAttention             # Core SDPA kernel wrapper
  ├── TTNNFusedQKVSelfAttention     # Fused Q/K/V linear for ViT
  ├── TTNNSelfAttention             # ViT self-attention
  │     └── TTNNViTSelfAttention    # ViT-specific subclass
  ├── LlamaAttention                # LLM decoder attention with RoPE
  ├── TTNNWhisperAttention          # Encoder-decoder with cross-attention
  └── TTNNGR00TSelfAttention        # Robotics model with optional Q/K norm, RoPE, GQA
```

Multiple specialized classes, each hard-coded for a specific model family. The `from_torch` factory pattern adapts from PyTorch layer instances.

---

## Feature Comparison Matrix

| Feature | TT-DiT `Attention` | TT-Symbiote Attention Classes |
|---|---|---|
| **Joint attention (spatial + prompt)** | Native: `joint_scaled_dot_product_attention` with separate Q/K/V per stream | Not supported. Closest is `TTNNWhisperAttention` cross-attention, but that uses separate Q from one source and K/V from another -- not concatenated K/V from both |
| **Per-head QKV norm** | `RMSNorm` applied to Q and K per head after QKV split | `TTNNGR00TSelfAttention` optionally applies Q/K RMSNorm per head after reshaping to 4D (same ordering as TT-DiT, different code path). Other classes have no per-head norm |
| **Fused QKV projection** | Single `ColParallelLinear` with interleaved weight layout for TP-compatible fracturing | `TTNNFusedQKVSelfAttention` concatenates Q/K/V weights and uses `ttnn.experimental.nlp_create_qkv_heads` for splitting. No TP-aware interleaving |
| **Tensor parallelism** | Built-in via `ColParallelLinear` and `CCLManager` all-gather | Not supported. Single-device only |
| **Sequence parallelism** | `ring_joint_scaled_dot_product_attention` with persistent ping-pong buffers | Not supported |
| **RoPE** | `_apply_rope` using `ttnn.alt_complex_rotate90` on spatial and prompt separately | `TTNNRotaryPositionEmbedding` in `LlamaAttention` and `TTNNGR00TSelfAttention`. Uses `ttnn.experimental.rotary_embedding_llama` or manual cos/sin multiplication |
| **KV cache** | None (full recomputation each step) | `TTNNPagedAttentionKVCache` with paged fill, update, and decode SDPA |
| **SDPA kernel** | `ttnn.transformer.joint_scaled_dot_product_attention` | `ttnn.transformer.scaled_dot_product_attention` (non-joint) |
| **SDPA fallback** | None -- must succeed | `TTNNSDPAAttention` falls back to manual matmul attention if SDPA kernel fails |
| **Attention mask** | No explicit mask (joint SDPA is bidirectional) | `TTNNGR00TSelfAttention` builds additive masks for encoder padding and causal masking |
| **Output projection** | Separate `to_out` (spatial) and `to_add_out` (prompt), both `ColParallelLinear` | Single output projection (e.g., `o_proj` in `LlamaAttention`, `out_proj` in `TTNNWhisperAttention`) |
| **Head padding** | `PaddingConfig` pads heads to TP/tile-aligned count | No head padding infrastructure |
| **Weight sharing** | `UnregisteredModule` for spatial/prompt weight reuse | Not applicable (no dual-stream architecture) |
| **Compute config** | `HiFi2`, `fp32_dest_acc=False` | `HiFi4`, `fp32_dest_acc=True` (higher precision, lower throughput) |
| **Chunk sizes** | `q_chunk_size=128`, `k_chunk_size=512` (asymmetric) | `q_chunk_size=256`, `k_chunk_size=256` (symmetric) |

---

## Detailed Comparison: Key Areas

### 1. Joint Attention vs. Standard SDPA

TT-DiT's joint attention is the most fundamental difference. In TT-DiT:

```python
spatial, prompt = ttnn.transformer.joint_scaled_dot_product_attention(
    q, k, v,                    # spatial Q/K/V
    add_q, add_k, add_v,        # prompt Q/K/V
    joint_strategy="rear",
)
```

This single kernel call internally concatenates `[K_spatial; K_prompt]` and `[V_spatial; V_prompt]`, computes attention for both spatial Q and prompt Q against the combined K/V, and returns separate outputs. Both streams attend to the full combined sequence.

In TT-Symbiote, the closest equivalent is `TTNNWhisperAttention`, which performs **cross-attention**:

```python
# Cross-attention: Q from decoder, K/V from encoder
query = self.q_proj_ttnn(hidden_states)   # Q from one source
key = self.k_proj_cross(key_value_states)  # K from another
value = self.v_proj_cross(key_value_states) # V from another

attn_out = self.sdpa(self, query, key, value, ...)
```

This is unidirectional (decoder attends to encoder) and does not concatenate sequences. Implementing DiT-style joint attention in TT-Symbiote would require:

- A new attention class or mode that handles two input sequences.
- Use of the `joint_scaled_dot_product_attention` TTNN kernel (which is a stable API).
- Separate QKV projections, norms, and RoPE for each stream.
- Return of two output tensors.

### 2. Per-Head QKV Normalization

TT-DiT normalizes Q and K independently per head after the QKV split:

```python
q = self.norm_q(q)  # RMSNorm on [batch, n_heads, seq_len, head_dim]
k = self.norm_k(k)  # same norm structure, different weights
```

TT-Symbiote's `TTNNGR00TSelfAttention` has optional Q/K normalization:

```python
if self.tt_q_norm is not None:
    q = self._rms_norm_on_device(q, self.tt_q_norm, device)
if self.tt_k_norm is not None:
    k = self._rms_norm_on_device(k, self.tt_k_norm, device)
```

These norms are `TTNNRMSNorm` instances mapped from the source model's `q_norm` and `k_norm` modules. In `TTNNGR00TSelfAttention`, Q and K are first reshaped to 4D `[batch, n_heads, seq_len, head_dim]` via `prepare_heads_on_device`, then the norms are applied per-head on these 4D tensors — the same ordering as TT-DiT.

The key difference is not the ordering but the implementation path: TT-DiT uses inline `RMSNorm` modules defined within `Attention.__init__`, while TT-Symbiote maps them from the source model's existing norm modules via `from_torch`. For a DiT port, the per-head norm pattern from `TTNNGR00TSelfAttention` could be reused directly.

### 3. KV Cache: Present vs. Absent

TT-Symbiote includes a sophisticated paged KV cache (`TTNNPagedAttentionKVCache`) designed for autoregressive LLM decoding:

```python
class TTNNPagedAttentionKVCache(Cache):
    # Pre-allocated blocks on device
    cache_shape = (max_num_blocks, num_kv_heads, block_size, head_dim)

    # Paged fill (prefill)
    def paged_fill_on_device(self, key_states, value_states, layer_idx, batch_idx):
        ttnn.experimental.paged_fill_cache(k_cache, key_states, page_table, ...)

    # Paged update (decode, single token)
    def paged_update_on_device(self, key_states, value_states, layer_idx, current_pos):
        ttnn.experimental.paged_update_cache(k_cache, key_states, update_idxs_tensor=current_pos, ...)

    # Paged SDPA decode
    def paged_sdpa_decode(self, query, layer_idx, current_pos, scale, ...):
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(query, k_cache, v_cache, ...)
```

TT-DiT has **no KV cache at all**. During each denoising step, the full spatial and prompt sequences are processed from scratch. This is because:

- DiT models are not autoregressive -- every denoising step recomputes attention over the entire sequence.
- The noise prediction at timestep $t$ depends on the noisy input $x_t$, which changes at every step, so previous KV states are invalid.

This is a simplification for porting: any DiT implementation in TT-Symbiote would not need the KV cache infrastructure. However, TT-Symbiote's attention modules assume KV cache integration, so a port would need to either bypass the cache or create a new attention path that omits it.

### 4. SDPA Compute Configuration

The two frameworks make different precision/performance tradeoffs:

| Parameter | TT-DiT | TT-Symbiote |
|---|---|---|
| `math_fidelity` | `HiFi2` | `HiFi4` |
| `math_approx_mode` | `False` | `False` |
| `fp32_dest_acc_en` | `False` | `True` |
| `packer_l1_acc` | N/A | `True` |
| `q_chunk_size` | 128 | 256 |
| `k_chunk_size` | 512 | 256 |
| `exp_approx_mode` | `False` | `False` |

TT-DiT uses lower fidelity (`HiFi2`) and disables FP32 destination accumulation. This is acceptable because:

- DiT inference is iterative (many denoising steps), so small per-step errors tend to average out.
- Per-head RMSNorm on Q/K keeps the attention logits bounded, reducing the need for high-precision accumulation.
- The throughput gain from lower fidelity is critical for real-time image/video generation.

TT-Symbiote uses `HiFi4` and `fp32_dest_acc=True` because LLM inference is autoregressive and cumulative errors in KV cache values persist across all subsequent tokens.

The asymmetric chunk sizes in TT-DiT (`q=128`, `k=512`) are optimized for the access pattern where Q stays in L1 while K/V are streamed from DRAM. TT-Symbiote uses symmetric `256/256`, which may not be optimal for all sequence lengths.

### 5. Matmul Fallback

TT-Symbiote's `TTNNSDPAAttention` includes a fallback path:

```python
if self._sdpa_available:
    try:
        attn_output = ttnn.transformer.scaled_dot_product_attention(...)
    except RuntimeError as e:
        self._sdpa_available = False

return self._matmul_attention(query, key, value, ...)
```

The `_matmul_attention` method implements attention using explicit matmul, softmax, and matmul operations. This handles edge cases where the SDPA kernel fails (unusual shapes, unsupported configurations).

TT-DiT has no fallback -- the SDPA kernel must succeed. This is acceptable because TT-DiT controls the input shapes precisely (via padding) and does not need to handle arbitrary model configurations.

### 6. Adaptive Layer Normalization

TT-DiT's `TransformerBlock` uses adaptive LayerNorm (adaLN) with time-conditioned modulation:

```python
spatial_normed = self.norm1_norm(
    spatial,
    dynamic_weight=(1 + spatial_scale_attn),
    dynamic_bias=spatial_shift_attn,
)
```

TT-Symbiote has no equivalent concept. Its attention modules (`LlamaAttention`, `TTNNGR00TSelfAttention`) assume the normalization is handled by the surrounding transformer layer (which in HuggingFace's architecture is a separate module). The normalization is always static (learned weight and bias), never time-conditioned.

Porting the `TransformerBlock` to TT-Symbiote would require:

- A new block-level module that handles the time embedding projection and chunking.
- Integration with `DistributedLayerNorm` or a compatible distributed norm that supports `dynamic_weight` and `dynamic_bias`.
- Gated residual connections (simple element-wise multiply and add).

---

## Porting Strategy: Gap Analysis

### Gaps That Require New Code

| Gap | Complexity | Notes |
|---|---|---|
| Joint SDPA kernel integration | **Medium** | `ttnn.transformer.joint_scaled_dot_product_attention` is a stable TTNN API. TT-Symbiote would need a new attention class that calls it with the correct tensor shapes |
| Dual-stream QKV projections | **Low** | Two separate fused QKV linear layers -- straightforward to implement |
| Per-head RMSNorm after QKV split | **Low** | Apply `ttnn.rms_norm` to `[B, H, S, D]` tensors. Requires calling the norm on a 4D tensor in head-major layout |
| Adaptive LayerNorm | **Medium** | Requires `dynamic_weight` / `dynamic_bias` support in the norm layer, or a custom wrapper. `DistributedLayerNorm` already supports this, but TT-Symbiote's norms do not |
| Time-conditioned modulation | **Medium** | Linear projection of time embedding, 6-way chunk, and integration into the block forward pass. No equivalent abstraction exists in TT-Symbiote |
| Gated residual connections | **Low** | Element-wise multiply and add -- trivial in TTNN |
| Head padding infrastructure | **Low** | `PaddingConfig` logic can be replicated or the TT-DiT utility can be imported directly |

### Gaps That May Not Be Needed

| Feature | Reason |
|---|---|
| KV cache | DiT models do not use KV cache |
| Causal masking | DiT attention is bidirectional |
| GQA (Grouped Query Attention) | DiT models use standard MHA (all heads have Q, K, V) |
| Sliding window attention | Not used in DiT models |

### Existing TT-Symbiote Features That Can Be Reused

| Feature | Source |
|---|---|
| `TTNNLinear` for projections | `modules/linear.py` -- though TP-parallel variants would need to be added |
| `TTNNRotaryPositionEmbedding` | `modules/rope.py` -- usable for prompt RoPE; spatial RoPE may need a 2D/3D variant |
| `TTNNRMSNorm` | `modules/normalization.py` -- usable for per-head norm (with shape adjustments) |
| `TTNNSDPAAttention` compute config | `modules/attention.py` -- SDPA program config pattern is transferable |

---

## Summary: What Makes DiT Attention Fundamentally Different

The gap between TT-Symbiote's LLM-oriented attention and TT-DiT's diffusion attention is not just a matter of missing features -- it reflects fundamentally different computational models:

1. **Autoregressive vs. full-sequence**: LLMs generate one token at a time with a growing KV cache. DiTs process the entire spatial+prompt sequence at every denoising step. This eliminates KV cache needs but requires efficient full-sequence attention.

2. **Single-stream vs. dual-stream**: LLMs have one hidden state stream. DiTs maintain two parallel streams (spatial and prompt) that interact only through joint attention. This doubles the number of projection layers and norms.

3. **Static normalization vs. dynamic modulation**: LLM layers are time-invariant. DiT layers are conditioned on the diffusion timestep via learned shift/scale/gate parameters, making every layer's behavior dependent on the denoising schedule.

4. **Causal vs. bidirectional**: LLM attention is masked to prevent future token leakage. DiT attention is fully bidirectional -- every spatial token attends to every prompt token and vice versa.

A successful port would likely involve creating a new `TTNNDiTAttention` class and a `TTNNDiTTransformerBlock` class rather than trying to extend the existing LLM-oriented classes.

---

## Key Takeaways

1. **Joint SDPA is the critical missing primitive**: TT-Symbiote has no equivalent of `joint_scaled_dot_product_attention`. This single kernel is the cornerstone of DiT attention and would need to be integrated as a new attention class, not grafted onto existing ones.

2. **The entire TransformerBlock architecture is novel for TT-Symbiote**: adaptive LayerNorm, time-conditioned modulation, dual-stream processing, and gated residuals are all absent from TT-Symbiote's module library. A port requires building a new block-level abstraction.

3. **Precision tradeoffs differ by workload**: TT-DiT uses `HiFi2` without FP32 accumulation (iterative denoising tolerates per-step noise), while TT-Symbiote uses `HiFi4` with FP32 accumulation (autoregressive decoding requires precision). A port should preserve TT-DiT's precision choices for correctness.

4. **KV cache, causal masking, and GQA are unnecessary**: these features represent significant complexity in TT-Symbiote's attention code that would not be needed for DiT models. A new DiT attention class can be simpler than the existing LLM attention classes.

5. **Tensor parallelism is the largest infrastructure gap**: TT-DiT's attention is deeply integrated with TP and SP via `ColParallelLinear`, `CCLManager`, and ring attention. TT-Symbiote currently has no multi-device parallelism in its attention modules. Porting without parallelism is straightforward; porting with parallelism requires the full TP/SP infrastructure from [Chapter 2](../ch2_parallelism_and_ccl/index.md).

---

**Next:** [Chapter 5 -- End-to-End Pipelines and Model Registration](../ch5_pipelines_and_serving/index.md)
