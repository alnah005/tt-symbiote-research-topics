# Design Options for Heterogeneous Attention Modules

This file analyzes three approaches for structuring the TTNNModule classes that
implement the two attention types in Gemma 4 31B, and concludes with a
recommendation.

## Shared vs Divergent Logic Inventory

Before evaluating class designs, it is useful to catalogue which operations
are shared between sliding and global attention and which diverge. This
inventory drives the design decision.

### Shared Operations

| Operation | Sliding | Global | Notes |
|-----------|---------|--------|-------|
| Q projection | `ttnn.linear(x, W_Q)` | `ttnn.linear(x, W_Q)` | Same op, different weight shapes |
| Q-norm | Scaled RMSNorm, `head_dim` | Scaled RMSNorm, `head_dim` | Same op, different dim (256 vs 512) |
| V-norm | Unscaled RMSNorm, `head_dim` | Unscaled RMSNorm, `head_dim` | Same op, different dim |
| K-norm | Scaled RMSNorm, `head_dim` | Scaled RMSNorm, `head_dim` | Same op, different dim |
| O projection | `ttnn.linear(attn_out, W_O)` | `ttnn.linear(attn_out, W_O)` | Same op, different weight shapes |
| Output dtype/shape | `[B, 1, 5376]` | `[B, 1, 5376]` | Identical output interface |

### Divergent Operations

| Operation | Sliding | Global |
|-----------|---------|--------|
| K/V projection | Separate `k_proj` + `v_proj` | Single `k_proj`, V reuses K |
| Fused QKV | Q+K+V `[5376, 16384]` | Q+K `[5376, 18432]` |
| Post-projection split | 3-way slice (Q, K, V) | 2-way slice (Q, shared_KV) |
| Tensor duplication | None | Clone/functional split for K and V paths |
| RoPE type | Full, theta=10K, 256 dims | Partial, theta=1M, 128/512 dims |
| RoPE module | `TTNNDistributedRotaryPositionEmbedding` | `TTNNRotaryPositionEmbedding` (non-distributed) |
| Cos/sin table shape | `[max_seq_len, 256]` | `[max_seq_len, 128]` |
| KV cache geometry | 16 heads x 256 dim, window=1024 | 4 heads x 512 dim, full causal |
| SDPA call | `paged_sdpa_decode` with `sliding_window_size=1024` | `paged_sdpa_decode` with full causal |
| GQA ratio per device (TP=8) | 4Q : 2KV = 2:1 | 4Q : varies (see Ch6) |
| Program config | Tuned for w=1024, head_dim=256 | Tuned for full context, head_dim=512 |

### Assessment

The shared operations are limited to the Q and O projections and the norm
layers --- and even these differ in dimensionality. The KV projection path,
RoPE application, KV cache update, and SDPA call are all structurally
different. This means the `forward` method has more divergent code than shared
code.

## Option A --- Single Unified Class

### Design

```python
class TTNNGemma4Attention(TTNNModule):
    def __init__(self, layer_idx: int, config: Gemma4Config):
        self.is_sliding = layer_idx not in config.global_layer_indices
        self.head_dim = 256 if self.is_sliding else 512
        self.num_kv_heads = 16 if self.is_sliding else 4

        # Q and O projections (always present)
        self.q_proj = TTNNLinear(...)
        self.o_proj = TTNNLinear(...)

        # K projection (always present)
        self.k_proj = TTNNLinear(...)

        # V projection (sliding only)
        self.v_proj = TTNNLinear(...) if self.is_sliding else None

        # Norms
        self.q_norm = TTNNDistributedRMSNorm(self.head_dim, with_scale=True)
        self.k_norm = TTNNDistributedRMSNorm(self.head_dim, with_scale=True)
        self.v_norm = TTNNDistributedRMSNorm(self.head_dim, with_scale=False)  # all-ones weight

        # RoPE
        if self.is_sliding:
            self.rope = TTNNDistributedRotaryPositionEmbedding(...)
        else:
            self.rope = TTNNRotaryPositionEmbedding(...)  # non-distributed for partial

    def forward(self, hidden_states, cos, sin, kv_cache, current_pos, page_table):
        # Q projection (shared)
        query_states = self.q_proj(hidden_states)
        query_states = reshape_to_heads(query_states, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        # KV projection (divergent)
        if self.is_sliding:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            key_states = reshape_to_heads(key_states, self.num_kv_heads, self.head_dim)
            value_states = reshape_to_heads(value_states, self.num_kv_heads, self.head_dim)
        else:
            shared_kv = self.k_proj(hidden_states)
            shared_kv = reshape_to_heads(shared_kv, self.num_kv_heads, self.head_dim)
            key_states = shared_kv  # will diverge through norms
            value_states = shared_kv

        # K-norm and V-norm (shared op, different dims)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        # RoPE (divergent)
        query_states = self.rope(query_states, cos, sin)
        key_states = self.rope(key_states, cos, sin)

        # KV cache update (divergent geometry)
        kv_cache.paged_update_on_device(key_states, value_states, self.layer_idx, current_pos)

        # SDPA (divergent window constraint)
        if self.is_sliding:
            attn_output = kv_cache.paged_sdpa_decode(
                query_states, self.layer_idx, current_pos,
                scale=self.scale, sliding_window_size=1024,
                page_table=page_table
            )
        else:
            attn_output = kv_cache.paged_sdpa_decode(
                query_states, self.layer_idx, current_pos,
                scale=self.scale, page_table=page_table
            )

        # O projection (shared)
        attn_output = reshape_from_heads(attn_output)
        return self.o_proj(attn_output)
```

### Pros

1. **Single code path to maintain.** Bug fixes and improvements apply to both
   layer types simultaneously.
2. **Mirrors HuggingFace.** The reference `Gemma4TextAttention` uses a single
   class with `self.is_sliding` branching, making numerical validation easier.
3. **Consistent interface.** The decoder layer module calls
   `self.attention.forward(...)` regardless of layer type.
4. **Simpler module replacement.** tt-symbiote's `from_torch` needs only one
   mapping rule: `Gemma4TextAttention` -> `TTNNGemma4Attention`.

### Cons

1. **Conditional branches in forward.** The `if self.is_sliding` checks in
   the hot path add code complexity and make it harder to reason about the
   execution flow for either type in isolation.
2. **Different program configs.** The Q, K, O projections have different
   shapes across layer types, requiring per-type program configs stored as
   attributes. The constructor must set up two different sets of matmul
   configs.
3. **Different KV cache geometries.** The two types need different
   `PagedAttentionConfig` instances (different `num_kv_heads`, `head_dim`,
   and window constraints). The unified class must manage this asymmetry.
4. **Optimization friction.** Profiling and optimizing one attention type
   requires mentally filtering out the other type's code paths. Metal Trace
   capture must account for both branches.
5. **Different RoPE modules.** Sliding uses the distributed variant; global
   uses the non-distributed variant. Storing both (only one used) or
   conditionally constructing them adds constructor complexity.

## Option B --- Two Separate Classes

### Design

```python
class TTNNGemma4SlidingAttention(TTNNModule):
    """Sliding-window attention: 16 KV heads, head_dim=256, window=1024."""

    def __init__(self, layer_idx: int, config: Gemma4Config):
        self.q_proj = TTNNLinear(...)   # [5376, 8192]
        self.k_proj = TTNNLinear(...)   # [5376, 4096]
        self.v_proj = TTNNLinear(...)   # [5376, 4096]
        self.o_proj = TTNNLinear(...)   # [8192, 5376]
        self.q_norm = TTNNDistributedRMSNorm(256, with_scale=True)
        self.k_norm = TTNNDistributedRMSNorm(256, with_scale=True)
        self.v_norm = TTNNDistributedRMSNorm(256, with_scale=False)
        self.rope = TTNNDistributedRotaryPositionEmbedding(...)

    def forward(self, hidden_states, cos, sin, kv_cache, current_pos, page_table):
        # Straight-line code: no conditionals
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = reshape_to_heads(q, 32, 256)
        k = reshape_to_heads(k, 16, 256)
        v = reshape_to_heads(v, 16, 256)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        q, k = self.rope(q, k, cos, sin)
        kv_cache.paged_update_on_device(k, v, self.layer_idx, current_pos)
        attn_out = kv_cache.paged_sdpa_decode(
            q, self.layer_idx, current_pos,
            scale=self.scale, sliding_window_size=1024,
            page_table=page_table
        )
        attn_out = reshape_from_heads(attn_out)
        return self.o_proj(attn_out)


class TTNNGemma4GlobalAttention(TTNNModule):
    """Global attention: 4 KV heads, head_dim=512, K=V sharing, p-RoPE."""

    def __init__(self, layer_idx: int, config: Gemma4Config):
        self.q_proj = TTNNLinear(...)   # [5376, 16384]
        self.k_proj = TTNNLinear(...)   # [5376, 2048]
        # No v_proj --- K=V sharing
        self.o_proj = TTNNLinear(...)   # [16384, 5376]
        self.q_norm = TTNNDistributedRMSNorm(512, with_scale=True)
        self.k_norm = TTNNDistributedRMSNorm(512, with_scale=True)
        self.v_norm = TTNNDistributedRMSNorm(512, with_scale=False)
        self.rope = TTNNRotaryPositionEmbedding(...)  # non-distributed, partial

    def forward(self, hidden_states, cos, sin, kv_cache, current_pos, page_table):
        q = self.q_proj(hidden_states)
        shared_kv = self.k_proj(hidden_states)
        q = reshape_to_heads(q, 32, 512)
        shared_kv = reshape_to_heads(shared_kv, 4, 512)
        q = self.q_norm(q)
        k = self.k_norm(shared_kv)       # produces new tensor
        v = self.v_norm(shared_kv)        # consumes same input, produces new tensor
        q, k = self.rope(q, k, cos, sin)  # partial RoPE: 128/512 dims
        kv_cache.paged_update_on_device(k, v, self.layer_idx, current_pos)
        attn_out = kv_cache.paged_sdpa_decode(
            q, self.layer_idx, current_pos,
            scale=self.scale, page_table=page_table
        )
        attn_out = reshape_from_heads(attn_out)
        return self.o_proj(attn_out)
```

### Pros

1. **No runtime branching.** Each class has a clean, linear forward pass. The
   code for each attention type is self-contained and easy to follow.
2. **Independent optimization.** Program configs, Metal Trace profiles, and
   performance tuning are entirely separate. Changes to the sliding attention
   path cannot accidentally affect global attention.
3. **Independent profiling.** Tracy or op-trace profiling naturally separates
   the two classes, making it easy to attribute latency to the correct
   attention type.
4. **Cleaner `from_torch`.** Two distinct mapping rules:
   sliding layers -> `TTNNGemma4SlidingAttention`,
   global layers -> `TTNNGemma4GlobalAttention`.

### Cons

1. **Code duplication.** The Q projection, Q-norm, O projection, V-norm, and
   the reshape/transpose logic are repeated in both classes. Any fix to these
   shared operations must be applied twice.
2. **Two sets of everything.** Two program config setups, two
   `move_weights_to_device_impl` methods, two weight loading paths. The
   maintenance surface doubles.
3. **Diverges from reference.** The HuggingFace implementation uses a single
   class, so numerical debugging requires mentally mapping two TTNN classes
   to one reference class.

## Option C --- Base Class With Specialized Subclasses

### Design

```python
class TTNNGemma4AttentionBase(TTNNModule):
    """Shared logic for both attention types."""

    def __init__(self, layer_idx: int, config: Gemma4Config,
                 num_kv_heads: int, head_dim: int):
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads  # 32
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)

        # Shared projections
        self.q_proj = TTNNLinear(config.hidden_size, self.num_heads * head_dim)
        self.o_proj = TTNNLinear(self.num_heads * head_dim, config.hidden_size)

        # Shared norms
        self.q_norm = TTNNDistributedRMSNorm(head_dim, with_scale=True)
        self.k_norm = TTNNDistributedRMSNorm(head_dim, with_scale=True)
        self.v_norm = TTNNDistributedRMSNorm(head_dim, with_scale=False)

    def forward(self, hidden_states, cos, sin, kv_cache, current_pos, page_table):
        # Step 1: Q projection (shared)
        query_states = self.q_proj(hidden_states)
        query_states = reshape_to_heads(query_states, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        # Step 2: KV projection + norms + RoPE (subclass-specific)
        query_states, key_states, value_states = self._project_kv_and_rope(
            hidden_states, query_states, cos, sin
        )

        # Step 3: KV cache update (subclass-specific geometry)
        kv_cache.paged_update_on_device(
            key_states, value_states, self.layer_idx, current_pos
        )

        # Step 4: SDPA (subclass-specific window constraint)
        attn_output = self._sdpa(query_states, kv_cache, current_pos, page_table)

        # Step 5: O projection (shared)
        attn_output = reshape_from_heads(attn_output)
        return self.o_proj(attn_output)

    def _project_kv_and_rope(self, hidden_states, query_states, cos, sin):
        raise NotImplementedError

    def _sdpa(self, query_states, kv_cache, current_pos, page_table):
        raise NotImplementedError


class TTNNGemma4SlidingAttention(TTNNGemma4AttentionBase):
    """Sliding-window attention subclass."""

    def __init__(self, layer_idx: int, config: Gemma4Config):
        super().__init__(layer_idx, config, num_kv_heads=16, head_dim=256)
        self.k_proj = TTNNLinear(5376, 4096)
        self.v_proj = TTNNLinear(5376, 4096)
        self.rope = TTNNDistributedRotaryPositionEmbedding(...)

    def _project_kv_and_rope(self, hidden_states, query_states, cos, sin):
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        k = reshape_to_heads(k, 16, 256)
        v = reshape_to_heads(v, 16, 256)
        k = self.k_norm(k)
        v = self.v_norm(v)
        query_states, k = self.rope(query_states, k, cos, sin)
        return query_states, k, v

    def _sdpa(self, query_states, kv_cache, current_pos, page_table):
        return kv_cache.paged_sdpa_decode(
            query_states, self.layer_idx, current_pos,
            scale=self.scale, sliding_window_size=1024,
            page_table=page_table
        )


class TTNNGemma4GlobalAttention(TTNNGemma4AttentionBase):
    """Global attention subclass with K=V sharing and partial RoPE."""

    def __init__(self, layer_idx: int, config: Gemma4Config):
        super().__init__(layer_idx, config, num_kv_heads=4, head_dim=512)
        self.k_proj = TTNNLinear(5376, 2048)
        # No v_proj --- K=V sharing
        self.rope = TTNNRotaryPositionEmbedding(...)  # non-distributed, partial

    def _project_kv_and_rope(self, hidden_states, query_states, cos, sin):
        shared_kv = self.k_proj(hidden_states)
        shared_kv = reshape_to_heads(shared_kv, 4, 512)
        k = self.k_norm(shared_kv)       # new tensor from shared input
        v = self.v_norm(shared_kv)        # new tensor from shared input
        query_states, k = self.rope(query_states, k, cos, sin)
        return query_states, k, v

    def _sdpa(self, query_states, kv_cache, current_pos, page_table):
        return kv_cache.paged_sdpa_decode(
            query_states, self.layer_idx, current_pos,
            scale=self.scale, page_table=page_table
        )
```

### Pros

1. **Shared code lives once.** The Q projection, Q-norm, O projection, and
   the overall forward flow are defined in the base class. Bug fixes apply
   automatically to both types.
2. **Clean separation of divergent logic.** The KV projection, RoPE, and
   SDPA call are isolated in subclass methods. Each subclass is a short,
   focused implementation.
3. **No runtime branching.** The base class calls abstract methods; the
   correct implementation is resolved at construction time via polymorphism.
4. **Extensible.** If future Gemma variants introduce a third attention type
   (e.g., a linear attention variant), it can be added as another subclass
   without modifying existing code.
5. **Natural `from_torch` mapping.** The decoder layer constructor inspects
   `layer_idx` and instantiates either `TTNNGemma4SlidingAttention` or
   `TTNNGemma4GlobalAttention`. The base class type hint
   (`TTNNGemma4AttentionBase`) provides a uniform interface.

### Cons

1. **Slightly more complex class hierarchy.** Three classes instead of one or
   two. Developers must understand the base/subclass relationship.
2. **Method call overhead.** The `_project_kv_and_rope` and `_sdpa` virtual
   method calls add a trivial amount of Python overhead per forward pass.
   This is negligible compared to device op latency.
3. **Program config management.** Each subclass needs its own program configs
   for the type-specific projections. This is the same as Option B but now
   split across base and subclass `move_weights_to_device_impl`.

## Comparative Summary

| Criterion | Option A (Unified) | Option B (Separate) | Option C (Base+Sub) |
|-----------|-------------------|--------------------|--------------------|
| Code duplication | None | High (Q, O proj, norms, reshape) | None (shared in base) |
| Runtime branching | Yes (`if is_sliding`) | None | None (polymorphism) |
| Forward readability | Moderate (interleaved branches) | High (linear code) | High (linear + inherited) |
| Per-type optimization | Harder (branches) | Easiest (isolated) | Easy (isolated subclasses) |
| `from_torch` complexity | Simple (one mapping) | Simple (two mappings) | Simple (two mappings) |
| HuggingFace alignment | Closest (mirrors single class) | Divergent | Moderate (base mirrors structure) |
| Extensibility | Low (add more branches) | Moderate (add new class) | High (add new subclass) |
| Class count | 1 | 2 | 3 |
| Maintenance surface | Low (one class) | High (two duplicated) | Low (one base + two thin subs) |

## Recommendation

**Option C (base class with specialized subclasses) is the recommended
approach.** The rationale:

1. **The divergent logic dominates.** The KV projection path, RoPE
   application, and SDPA call are structurally different between the two
   types. These are the most complex and performance-critical parts of the
   forward pass. Isolating them in subclasses makes each type's hot path
   self-contained and independently optimizable.

2. **The shared logic is non-trivial.** The Q projection, Q-norm, O
   projection, and the overall forward orchestration (project -> norm ->
   RoPE -> cache update -> SDPA -> output) are identical in structure. Putting
   this in a base class avoids the duplication penalty of Option B without
   the branching penalty of Option A.

3. **Program configs are naturally scoped.** The base class owns Q and O
   projection program configs (which differ in shape but follow the same
   pattern). Each subclass owns its KV projection configs and SDPA
   `SDPADecodeProgramConfig`. This scoping matches the logical ownership.

4. **Metal Trace compatibility.** Each subclass has a deterministic forward
   pass with no branches, making it compatible with Metal Trace capture. The
   decoder layer instantiates a concrete subclass, so the trace sees a fixed
   op sequence per layer.

5. **`from_torch` integration.** The decoder layer's `from_torch` inspects
   `layer_idx` to determine the layer type and constructs the appropriate
   subclass:

   ```python
   # In TTNNGemma4DecoderLayer.from_torch
   if layer_idx in config.global_layer_indices:
       self.attention = TTNNGemma4GlobalAttention.from_torch(hf_attn, layer_idx, config)
   else:
       self.attention = TTNNGemma4SlidingAttention.from_torch(hf_attn, layer_idx, config)
   ```

   The decoder layer's `forward` calls `self.attention.forward(...)` without
   knowing or caring which subclass is active.

### When to Reconsider

Option A (unified class) may be preferable if:

- The implementation is a quick prototype where minimizing class count matters
  more than long-term maintainability.
- The two attention types converge in a future Gemma revision (e.g., all layers
  become global), making the subclass hierarchy unnecessary.

Option B (separate classes) may be preferable if:

- The shared code turns out to be minimal in practice (e.g., if fused QKV
  optimization means Q projection is not a separate step).
- The two types require fundamentally different `move_weights_to_device_impl`
  flows that do not compose well in a base class.

---

**Next:** [`sliding_attention_forward.md`](./sliding_attention_forward.md)
