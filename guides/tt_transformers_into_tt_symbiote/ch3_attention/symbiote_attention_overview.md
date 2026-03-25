# TT Symbiote Attention Module Inventory

All attention-related classes in TT Symbiote live in two files:

- `models/experimental/tt_symbiote/modules/attention.py` — attention classes, paged KV cache, and helper functions
- `models/experimental/tt_symbiote/modules/rope.py` — rotary position embedding classes

---

## 1. Attention Classes

### 1.1 `TTNNViTSelfAttention`

```python
class TTNNViTSelfAttention(TTNNSelfAttention):
    @classmethod
    def from_torch(cls, self_attention: "ViTSelfAttention"): ...
```

`TTNNViTSelfAttention` is a subclass of `TTNNSelfAttention` that overrides `from_torch` to accept a HuggingFace `ViTSelfAttention` object (which stores Q, K, V as separate `Linear` layers rather than through the `fused_qkv` attribute used by `SelfAttention`). The body of `from_torch` is otherwise identical to `TTNNSelfAttention.from_torch`: it wraps the three projections into a `TTNNFusedQKVSelfAttention` and attaches a `TTNNSDPAAttention`.

> **Note:** The class docstring says "deprecated" only implicitly — the class comment reads "TTNN-accelerated ViT Self-Attention layer", which is identical to `TTNNSelfAttention`. In practice, `TTNNViTSelfAttention` is the entry point for ViT models because HuggingFace ViT exposes separate Q/K/V attributes, whereas `TTNNSelfAttention.from_torch` expects a `SelfAttention` object whose `fused_qkv` attribute is a `PytorchFusedQKVSelfAttention`.

### 1.2 `TTNNSDPAAttention`

```python
class TTNNSDPAAttention(TTNNModule):
    def __init__(self):
        self._fallback_torch_layer = TorchSDPAAttention()
        self.program_config = None
        self.compute_kernel_config = None
        self.memory_config = None
        self._sdpa_available = True
```

`TTNNSDPAAttention` is the innermost attention kernel. It is not constructed with weights — it is a stateless operator wrapper. Its `forward` method:

1. Ensures Q, K, V are in `TILE_LAYOUT` and `DRAM_MEMORY_CONFIG`.
2. Infers `is_causal` from the `module` argument's `is_causal` attribute if not provided explicitly. Then unconditionally overrides it to `False` when `query.shape[2] <= 1` (single-token query) or when `attention_mask is not None` — even if `is_causal=True` was passed explicitly.
3. Calls `ttnn.transformer.scaled_dot_product_attention` with the stored `program_config` and `compute_kernel_config`.
4. On `RuntimeError`, sets `self._sdpa_available = False` and falls back to `_matmul_attention`, which implements attention as explicit `ttnn.matmul` + `ttnn.softmax` ops.

The `program_config` and `compute_kernel_config` are set lazily by the containing module's `move_weights_to_device_impl`. All callers (`TTNNSelfAttention`, `TTNNViTSelfAttention`, `TTNNWhisperAttention`, `LlamaAttention`) configure them with `HiFi4` fidelity and `fp32_dest_acc_en=True`.

### 1.3 `TTNNFusedQKVSelfAttention`

```python
class TTNNFusedQKVSelfAttention(TTNNModule):
    @classmethod
    def from_torch(cls, fused_qkv: "PytorchFusedQKVSelfAttention"): ...

    def forward(self, hidden_states): ...
```

`TTNNFusedQKVSelfAttention` fuses the Q, K, V weight matrices into a single `TTNNLinear` and then calls `ttnn.experimental.nlp_create_qkv_heads` to split the output back into separate head tensors. The construction path is:

1. Concatenate `query.weight`, `key.weight`, `value.weight` along `dim=0` to form a `[3*hidden_size, hidden_size]` weight.
2. Concatenate the biases similarly (zero-padding any absent bias to zeros of the same shape).
3. Wrap the fused `nn.Linear` in `TTNNLinear.from_torch`.

In `forward`, the module:

1. Unsqueezes a 3-D hidden state to 4-D if needed.
2. Runs the fused linear.
3. Moves the result to L1.
4. Calls `ttnn.experimental.nlp_create_qkv_heads` with `transpose_k_heads=False`.
5. Deallocates the intermediate QKV tensor.

### 1.4 `TTNNSelfAttention`

```python
class TTNNSelfAttention(TTNNModule):
    def __init__(self, attention_config: SelfAttentionConfig) -> None: ...
    @classmethod
    def from_torch(cls, self_attention: "SelfAttention"): ...
    def move_weights_to_device_impl(self): ...
    def forward(self, hidden_states, head_mask=None, output_attentions=False): ...
```

`TTNNSelfAttention` is the base class for encoder-style self-attention (e.g., ViT). It composes `TTNNFusedQKVSelfAttention` (for Q/K/V) with `TTNNSDPAAttention` (for the attention computation). Key properties:

- `core_grid = ttnn.CoreGrid(y=8, x=8)` — fixed 8x8 grid; `program_config` is set in `move_weights_to_device_impl` with `q_chunk_size=256, k_chunk_size=256`.
- `is_causal = False` — encoder self-attention is bidirectional.
- `should_reallocate_in_attention = False` — when set to `True` by a caller, the value tensor is reallocated before SDPA to improve memory layout.

`forward` calls `ttnn.experimental.nlp_concat_heads` (not `nlp_concat_heads_decode`) on the SDPA output, typecasts back to the original dtype, and squeezes the batch dimension.

### 1.5 `TTNNWhisperAttention`

```python
class TTNNWhisperAttention(TTNNModule):
    def __init__(self, embed_dim, num_heads, dropout=0.0, is_causal=False, layer_idx=None): ...
    @classmethod
    def from_torch(cls, whisper_attn: "WhisperAttention"): ...
    def forward(self, hidden_states, key_value_states=None, past_key_value=None, attention_mask=None, **kwargs): ...
```

`TTNNWhisperAttention` handles both self-attention and cross-attention for Whisper. In `from_torch`, **all four projection attributes are always constructed**, regardless of whether the module will be used for self-attention or cross-attention:

- `qkv_proj` — fused Q/K/V `TTNNLinear` (used in self-attention forward path); K bias is zero-padded because Whisper's K projection has no bias.
- `q_proj_ttnn` — separate Q `TTNNLinear` (used in cross-attention forward path).
- `k_proj_cross`, `v_proj_cross` — separate K and V `TTNNLinear` instances (used in cross-attention forward path).
- `out_proj` — output projection `TTNNLinear` (used in both self-attention and cross-attention forward paths; the final step of `forward` calls `self.out_proj(attn_out)`).

The `is_cross` runtime flag (derived from `key_value_states is not None` in `forward`) determines which attributes are used; it is not set at construction time and does not affect which projections are stored.

The `forward` method uses `past_key_value.is_updated` to avoid recomputing cross-attention K/V after the first decode step. Cache updates are performed through the HuggingFace `Cache.update` interface (which, for `TTNNPagedAttentionKVCache`, moves tensors back to host).

`program_config` and `compute_kernel_config` are set identically to `TTNNSelfAttention`: HiFi4, `q_chunk_size=256`, `k_chunk_size=256`.

### 1.6 `LlamaAttention`

```python
class LlamaAttention(TTNNModule):
    def __init__(self):
        self.sdpa = TTNNSDPAAttention()
        self.rope = TTNNRotaryPositionEmbedding()
        self.core_grid = ttnn.CoreGrid(y=8, x=8)
        self.qkv_same_shape = True
```

`LlamaAttention` is TT Symbiote's wrapper for decoder-style GQA/MHA (LLaMA family). It is the only attention class in `modules/attention.py` that applies RoPE.

**Construction via `from_torch`:**

```python
@classmethod
def from_torch(cls, llama_attn: "LlamaAttention"):
    new_attn = cls()
    new_attn._fallback_torch_layer = llama_attn
    new_attn.num_key_value_groups = getattr(llama_attn, "num_key_value_groups", 1)
    new_attn.qkv_same_shape = (
        llama_attn.q_proj.weight.shape == llama_attn.k_proj.weight.shape
        and llama_attn.q_proj.weight.shape == llama_attn.v_proj.weight.shape
    )
    if new_attn.qkv_same_shape:
        new_attn.init_fused_parameters(...)   # uses TTNNFusedQKVSelfAttention
    else:
        new_attn.init_parameters()            # stores q_proj, k_proj, v_proj, and o_proj separately
```

When Q, K, V have the same shape (MHA or equal-size GQA), the weights are fused via `TTNNFusedQKVSelfAttention`. When they differ (GQA with fewer KV heads), they remain separate `TTNNLinear` instances.

**Forward:**

The `forward` method:

1. Projects hidden states to Q, K, V (fused or separate path).
2. Extracts `cos, sin` from `position_embeddings` if provided, otherwise falls back to calling `self.torch_layer.rotary_emb`.
3. Calls `self.rope(query_states, key_states, cos, sin)` — i.e., `TTNNRotaryPositionEmbedding.forward`.
4. Updates `past_key_values` via `past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)`.
5. **Prepends** zero padding to query along the sequence dimension to reach `kv_len` if `is_causal` and `q_len < kv_len` (i.e., `ttnn.concat([zero_pad, query_states], dim=2)`). Appending rather than prepending would produce wrong outputs because the causal mask aligns the real query tokens to the trailing positions.
6. Calls `self.sdpa` with `scaling=self.torch_layer.scaling`, `is_causal=self.torch_layer.is_causal`, and **`transpose_output=False`**. Because `transpose_output=False`, the SDPA output is returned in `[batch, heads, seq, head_dim]` order (no permute applied inside `TTNNSDPAAttention`). Passing `transpose_output=True` (the default) would permute the output to `[batch, seq, heads, head_dim]`, causing `nlp_concat_heads` in the next step to receive the wrong layout and produce incorrect results.
7. Calls `ttnn.experimental.nlp_concat_heads` on the `[batch, heads, seq, head_dim]` output (which expects heads at axis 1) and squeezes the batch dimension to yield a `[batch, seq, dim]` tensor. **If query was padded in step 5**, slices the last `original_q_len` rows: `attn_out[:, -original_q_len:, :]`. This slice must occur after `squeeze` because the tensor is 3D at that point; applying it to the 4D SDPA output (`[batch, heads, seq, head_dim]`) would incorrectly index the heads dimension instead of the sequence dimension.
8. Passes the result through `self.o_proj` (the output projection) and returns `(attn_out, None)`.

> **Warning:** `LlamaAttention.forward` prints a warning if `attention_mask is not None` and proceeds without applying it. The mask is explicitly passed as `None` to `self.sdpa`. This is safe for standard causal generation but will silently produce wrong outputs if a non-trivial mask is required.

---

## 2. RoPE Classes (`modules/rope.py`)

### 2.1 `TTNNRotaryPositionEmbedding`

```python
class TTNNRotaryPositionEmbedding(TTNNModule):
    def forward(self, q, k, cos, sin) -> Tuple[ttnn.Tensor, ttnn.Tensor]: ...
```

`TTNNRotaryPositionEmbedding` is a stateless module. It receives pre-computed `cos` and `sin` tensors and applies them to Q and K using `ttnn.experimental.rotary_embedding`.

The implementation handles two cases:

| Case | Condition | Behavior |
|------|-----------|----------|
| Partial rotary | `cos.shape[-1] < q.shape[-1]` (i.e., `rotary_dim < head_dim`) | Splits Q/K into rotary and pass-through portions, applies rotation only to the rotary portion, concatenates back |
| Full rotary | `rotary_dim >= head_dim` | Pads `cos`/`sin` to `head_dim` if needed and applies `rotary_embedding` directly |

In both cases, tile-boundary padding (multiples of 32) is applied and then trimmed off afterward. Inputs are coerced to `TILE_LAYOUT` before the op.

### 2.2 `TTNNDistributedRotaryPositionEmbedding`

```python
class TTNNDistributedRotaryPositionEmbedding(TTNNModule):
    def move_weights_to_device_impl(self): ...
    def forward(self, q, k, cos, sin) -> Tuple[ttnn.Tensor, ttnn.Tensor]: ...
```

`TTNNDistributedRotaryPositionEmbedding` uses `ttnn.experimental.rotary_embedding_llama`, which is the same op used by TT Transformers. The key difference from `TTNNRotaryPositionEmbedding` is that it pre-computes a **transformation matrix** and caches it.

**Transformation matrix construction** (in `move_weights_to_device_impl`):

```python
dhead = ttnn.TILE_SIZE  # 32
trans_mat = torch.zeros(1, 1, dhead, dhead)
trans_mat[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
trans_mat[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
```

This is the standard "swap-and-negate" rotation matrix used by `rotary_embedding_llama`. The `move_weights_to_device_impl` loop iterates over `[True, False]` and builds one matrix per flag value, storing both in `self._trans_mat_cache`. However, `forward` always retrieves the entry for `is_decode_mode = False` (the line `is_decode_mode = False  # (seq_len == 1)` is hardcoded), so the `True` entry is built but never used in the current implementation.

> **Note:** The current `TTNNDistributedRotaryPositionEmbedding.forward` always passes `is_decode_mode=False` regardless of input sequence length. The commented-out line `# (seq_len == 1)` shows intent to detect decode mode, but that logic is not active. This means the distributed class always runs in prefill mode, which may be incorrect for single-token decode steps.

Both RoPE classes require all inputs to be `ttnn.bfloat16`; `TTNNDistributedRotaryPositionEmbedding` explicitly typecasts Q, K, cos, sin to `bfloat16` before calling the op.

---

## 3. KV Cache (`modules/attention.py`)

### 3.1 `PagedAttentionConfig`

```python
@dataclass
class PagedAttentionConfig:
    block_size: int = 64
    max_num_blocks: int = 2048
    batch_size: int = 1

    @property
    def max_seq_length(self) -> int:
        return self.max_num_blocks * self.block_size

    @property
    def blocks_per_sequence(self) -> int:
        return self.max_num_blocks // self.batch_size
```

| Field | Default | Derived from |
|-------|---------|--------------|
| `block_size` | 64 | tokens per page |
| `max_num_blocks` | 2048 | total pages in the pool |
| `batch_size` | 1 | number of concurrent sequences |
| `max_seq_length` | 131072 | `max_num_blocks * block_size` |
| `blocks_per_sequence` | 2048 | `max_num_blocks // batch_size` |

### 3.2 `TTNNPagedAttentionKVCache`

```python
class TTNNPagedAttentionKVCache(Cache):
    def __init__(self, num_layers, num_kv_heads, head_dim, config, device=None, dtype=torch.bfloat16): ...
```

`TTNNPagedAttentionKVCache` is a HuggingFace `Cache` subclass. It allocates and manages per-layer key and value caches on TTNN devices.

**Host-side state:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `page_table` | `torch.Tensor` `[batch_size, blocks_per_sequence]` | Linear page mapping; initialized as `arange(max_num_blocks).reshape(batch_size, blocks_per_sequence)` |
| `_tt_key_cache` | `list[Optional[ttnn.Tensor]]` length `num_layers` | Per-layer key cache; `None` until `to_device` is called |
| `_tt_value_cache` | `list[Optional[ttnn.Tensor]]` length `num_layers` | Per-layer value cache; `None` until `to_device` is called |
| `_tt_page_table` | `Optional[ttnn.Tensor]` | Device copy of `page_table` in `ttnn.int32` / `ROW_MAJOR_LAYOUT` |

**On-device cache tensor shape** (set in `to_device`):

```
(max_num_blocks, num_kv_heads, block_size, head_dim)
```

All cache tensors are allocated as `ttnn.bfloat16`, `TILE_LAYOUT`, `DRAM_MEMORY_CONFIG` via `ttnn.zeros` with no mesh mapper. On multi-device meshes, `ttnn.ReplicateTensorToMesh` is used only for the `_tt_page_table`, not for the K/V cache tensors themselves.

**Key methods:**

| Method | Description |
|--------|-------------|
| `to_device(device)` | Allocates `_tt_key_cache`, `_tt_value_cache`, `_tt_page_table` on `device`; idempotent if already on the same device |
| `paged_fill_on_device(key_states, value_states, layer_idx, batch_idx=0)` | Calls `ttnn.experimental.paged_fill_cache` for prefill; truncates to `max_len` if the sequence is too long |
| `paged_update_on_device(key_states, value_states, layer_idx, current_pos)` | Calls `ttnn.experimental.paged_update_cache` for decode; `current_pos` is a `ttnn.Tensor` of current token positions |
| `paged_sdpa_decode(query, layer_idx, current_pos, scale, ...)` | Runs `ttnn.transformer.paged_scaled_dot_product_attention_decode` against the stored cache |
| `update(key_states, value_states, layer_idx, cache_kwargs)` | HuggingFace `Cache` interface; moves tensors to host and returns them — **does not use paged ops** |
| `get_seq_length(layer_idx=0)` | Returns `_seq_lengths[layer_idx]` |
| `get_max_cache_shape()` | Returns `config.max_seq_length` |

> **Warning:** The paged hardware acceleration is only available through the three `paged_*` methods, which must be called explicitly. `TTNNPagedAttentionKVCache` used naively through the HuggingFace `generate()` pipeline will not exercise the TTNN paged cache at all.

---

## Summary

| Class | File | Purpose | Weights? | RoPE? | Paged cache? |
|-------|------|---------|----------|-------|--------------|
| `TTNNViTSelfAttention` | attention.py | ViT encoder attention | via `from_torch` | No | No |
| `TTNNSDPAAttention` | attention.py | Stateless SDPA kernel wrapper | No | No | No |
| `TTNNFusedQKVSelfAttention` | attention.py | Fused QKV linear + head split | via `from_torch` | No | No |
| `TTNNSelfAttention` | attention.py | Encoder self-attention | via `from_torch` | No | No |
| `TTNNWhisperAttention` | attention.py | Whisper self + cross attention | via `from_torch` | No | Optional (HF Cache) |
| `LlamaAttention` | attention.py | LLaMA decoder attention | via `from_torch` | Yes (`TTNNRotaryPositionEmbedding`) | Optional (HF Cache) |
| `TTNNRotaryPositionEmbedding` | rope.py | Stateless RoPE via `rotary_embedding` | No | — | — |
| `TTNNDistributedRotaryPositionEmbedding` | rope.py | RoPE via `rotary_embedding_llama` + transform mat | Pre-computed transform mat | — | — |
| `TTNNPagedAttentionKVCache` | attention.py | Paged KV cache (HF `Cache` subclass) | KV tensors | No | Yes |
| `PagedAttentionConfig` | attention.py | Dataclass for paged cache geometry | — | — | — |

---

[Previous: Chapter Index](index.md) | [Next: TT Transformers Attention Overview](transformers_attention_overview.md)
