# K=V Sharing Mechanism

This file describes the K=V weight sharing mechanism used in the 10 global
attention layers of Gemma 4 31B. K=V sharing eliminates the V projection
entirely, reusing the K projection output for both keys and values with
divergent post-processing paths.

## Activation Condition

K=V sharing is controlled by two conditions evaluated in the
`Gemma4TextAttention.__init__` constructor:

```python
self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
```

The flag is `True` only when **both** of the following hold:

1. `attention_k_eq_v=true` in the model config (always true for 31B).
2. The layer is **not** a sliding-window layer (`self.is_sliding == False`).

This means K=V sharing is active in exactly the 10 global layers (indices 5,
11, 17, 23, 29, 35, 41, 47, 53, 59) and inactive in the 50 sliding layers.

## Dataflow: Single Projection, Two Paths

### Step 1 --- Shared Linear Projection

A single `k_proj` weight matrix of shape `[5376, 2048]` projects the input
hidden states into a shared tensor:

```math
\text{shared} = x \cdot W_K^T \quad \text{where } W_K \in \mathbb{R}^{5376 \times 2048}
```

This shared tensor has shape `[B, S, 2048]` (or `[1, 1, 2048]` during
single-token decode).

### Step 2 --- Assignment to K and V

The shared projection output is assigned to both `key_states` and
`value_states` before any normalization:

```python
key_states = self.k_proj(hidden_states).view(hidden_shape)
value_states = key_states  # V reuses K's raw output
```

At this point, `key_states` and `value_states` reference the **same underlying
tensor**. The critical insight is that all subsequent operations on `key_states`
are **not in-place** --- they produce new tensors via functional operations
(RMSNorm, RoPE, transpose). This means the original shared tensor remains
intact for the V path.

After reshape, both tensors have shape `[B, 4, S, 512]` --- 4 KV heads, each
with `head_dim=512`.

### Step 3 --- Divergent Post-Processing

The K and V paths diverge through different normalization and positional
encoding operations:

**K path:**

1. **K-norm** --- `Gemma4RMSNorm(dim=512, eps=1e-6, with_scale=True)`:
   RMSNorm with a learned per-element scale $\gamma_K \in \mathbb{R}^{512}$.
   This normalizes the magnitude and applies a learned rescaling.

2. **Partial RoPE** --- rotary position embedding applied to the first 128 of
   512 dimensions (`partial_rotary_factor=0.25`), with
   $\theta = 1{,}000{,}000$. The remaining 384 dimensions pass through
   unchanged.

3. **Transpose** --- from `[B, num_kv_heads, S, head_dim]` layout ready for
   SDPA.

**V path:**

1. **V-norm** --- `Gemma4RMSNorm(dim=512, eps=1e-6, with_scale=False)`:
   RMSNorm **without** a learned scale parameter. This performs pure magnitude
   normalization only.

2. **No RoPE** --- value vectors are not position-encoded. This is
   intentional: V vectors carry semantic content that should remain
   position-invariant.

3. **Transpose** --- from `[B, num_kv_heads, S, head_dim]` layout ready for
   SDPA.

### Complete Dataflow Diagram

```text
hidden_states  [B, 1, 5376]
      |
      v
  k_proj linear  W_K: [5376, 2048]
      |
      v
  shared_output  [B, 1, 2048]
      |
      +--- reshape ---> key_states  [B, 4, 1, 512]
      |                      |
      |                      v
      |                 k_norm (RMSNorm, with_scale=True, gamma in R^512)
      |                      |
      |                      v
      |                 partial RoPE (128/512 dims, theta=1M)
      |                      |
      |                      v
      |                 key_states  [B, 4, 1, 512]   --> KV cache (K slot)
      |
      +--- reshape ---> value_states  [B, 4, 1, 512]
                             |
                             v
                        v_norm (RMSNorm, with_scale=False, no gamma)
                             |
                             v
                        value_states  [B, 4, 1, 512]  --> KV cache (V slot)
```

### Reference Code (HuggingFace)

The forward pass in `Gemma4TextAttention` implements K=V sharing as follows:

```python
key_states = self.k_proj(hidden_states).view(hidden_shape)
value_states = (
    self.v_proj(hidden_states).view(hidden_shape)
    if self.v_proj is not None
    else key_states
)

key_states = self.k_norm(key_states)
key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
key_states = key_states.transpose(1, 2)

value_states = self.v_norm(value_states)
value_states = value_states.transpose(1, 2)
```

When `self.v_proj is None` (global layers with K=V sharing), `value_states`
is assigned to `key_states` directly. The subsequent `self.k_norm(key_states)`
creates a **new tensor** (RMSNorm is not in-place), so `value_states` still
points to the original pre-norm projection output.

## Why K=V Sharing Works

### Intuition

In standard attention, K and V are conceptually different: K participates in
the dot-product similarity computation (with Q) to produce attention weights,
while V provides the content that is aggregated using those weights. However,
both K and V are derived from the same input hidden states via independent
linear projections. K=V sharing posits that the **raw linear projection** can
be shared because the downstream operations (normalization, RoPE, or lack
thereof) sufficiently differentiate the two roles:

- **K receives RoPE**, making it position-dependent. The attention weights
  $\text{softmax}(QK^T / \sqrt{d})$ thus encode both semantic similarity and
  positional relationships.
- **V receives no RoPE**, keeping it position-invariant. The aggregated output
  carries semantic content without positional artifacts.

The divergent normalization (scaled vs unscaled RMSNorm) further separates the
two representations: K-norm's learned $\gamma$ can rescale individual
dimensions to optimize the dot-product similarity space, while V-norm's
scale-free normalization simply stabilizes magnitudes.

### Parameter Savings

Each global layer saves one V projection weight matrix:

```math
\text{savings per layer} = 5376 \times 2048 = 11{,}010{,}048 \text{ parameters}
```

```math
\text{total savings (10 layers)} = 10 \times 11{,}010{,}048 = 110{,}100{,}480 \text{ parameters}
```

At BF16 (2 bytes per parameter):

```math
\text{memory savings} = 110{,}100{,}480 \times 2 = 220{,}200{,}960 \text{ bytes} \approx 220 \text{ MB}
```

On a T3K mesh with TP=8, this translates to ~27.5 MB saved per device ---
modest but meaningful when the total DRAM budget per chip is 12 GB.

## TTNN Mapping

### Single Shared Projection

The K/V projection maps to a single `ttnn.linear` call:

```python
shared_kv = ttnn.linear(hidden_states, k_proj_weight)  # [B, 1, 2048]
```

Under tensor parallelism, `k_proj_weight` is column-parallel sharded across
8 devices. With 4 KV heads and `head_dim=512`, the output dimension per device
depends on the sharding strategy for global KV heads (see
[Chapter 6](../ch6_tp_sharding/index.md)).

### Tensor Duplication for Divergent Paths

After the shared projection, the TTNN implementation must produce two
independent tensors for the K and V paths. There are two options:

**Option A --- `ttnn.clone`:**

```python
key_states = shared_kv
value_states = ttnn.clone(shared_kv)
```

This creates a physical copy of the tensor. The K path can then modify
`key_states` through K-norm and RoPE without affecting `value_states`.

**Option B --- Rely on functional semantics:**

If all downstream operations (K-norm, RoPE, V-norm) produce new output tensors
rather than modifying in-place, then no explicit clone is needed:

```python
key_states = k_norm(shared_kv)          # new tensor
key_states = apply_rope(key_states, ...) # new tensor

value_states = v_norm(shared_kv)        # new tensor
```

In this case, `shared_kv` is consumed as a read-only input by both paths.

**Recommendation:** Option B is preferred when the norm and RoPE operations are
guaranteed to allocate new output tensors. This avoids the memory and latency
cost of an explicit clone. However, if any operation is configured to run
in-place (e.g., via output tensor reuse), Option A is required for correctness.

### Divergent Norm and RoPE Paths

After duplication (implicit or explicit), the two paths proceed independently:

```python
# K path
key_states = ttnn_k_norm(shared_kv)                    # scaled RMSNorm
key_states = ttnn_partial_rope(key_states, cos, sin)   # 128/512 dims
key_states = ttnn.transpose(key_states, 1, 2)

# V path
value_states = ttnn_v_norm(shared_kv)                  # unscaled RMSNorm
value_states = ttnn.transpose(value_states, 1, 2)
```

See [`vnorm_implementation.md`](./vnorm_implementation.md) for the three TTNN
strategies for implementing the unscaled V-norm, and
[Chapter 4](../ch4_dual_rope/index.md) for the partial RoPE implementation.

### Impact on Fused QKV

Standard fused QKV packs three projection weights into a single matrix and
performs one large matmul. With K=V sharing in global layers, the fused weight
packs **only Q and K**:

```math
W_{QK}^{\text{global}} = [W_Q \mid W_K] \quad \text{shape: } [5376, 18432]
```

where $18432 = 16384 + 2048$ (Q output dim + K output dim).

After the fused matmul, the output `[B, 1, 18432]` is sliced:

- Slice `[:, :, :16384]` --- Q activation, reshaped to `[B, 32, 1, 512]`.
- Slice `[:, :, 16384:]` --- shared K/V activation `[B, 1, 2048]`, reshaped to
  `[B, 4, 1, 512]` and used as input to both K and V post-processing paths.

For **sliding layers**, fused QKV remains the standard three-way pack:

```math
W_{QKV}^{\text{sliding}} = [W_Q \mid W_K \mid W_V] \quad \text{shape: } [5376, 16384]
```

where $16384 = 8192 + 4096 + 4096$.

This means the TTNN implementation needs two different fused weight layouts ---
one for sliding layers (Q+K+V) and one for global layers (Q+K only). The slice
offsets after the fused matmul differ accordingly.

### Contrast With Sliding Layers

In sliding layers, K=V sharing is **not** active. Each layer has separate
`k_proj` and `v_proj` weights, both of shape `[5376, 4096]`. The V projection
produces an independent `value_states` tensor that goes through V-norm (but no
RoPE), while K goes through K-norm and full RoPE (all 256 dims rotated,
$\theta = 10{,}000$).

| Aspect | Sliding Layers | Global Layers |
|--------|---------------|---------------|
| K=V sharing | No | Yes |
| V projection weight | `[5376, 4096]` (separate) | None (reuses K) |
| K norm | Scaled RMSNorm ($\gamma \in \mathbb{R}^{256}$) | Scaled RMSNorm ($\gamma \in \mathbb{R}^{512}$) |
| V norm | Unscaled RMSNorm (no $\gamma$) | Unscaled RMSNorm (no $\gamma$) |
| K RoPE | Full (256/256 dims, $\theta$=10K) | Partial (128/512 dims, $\theta$=1M) |
| V RoPE | None | None |
| Fused weight | Q+K+V `[5376, 16384]` | Q+K `[5376, 18432]` |

## Key Implementation Considerations

1. **Tensor aliasing safety.** In PyTorch, `value_states = key_states` creates
   an alias. The subsequent `self.k_norm(key_states)` returns a new tensor,
   leaving the original intact for V-norm. In TTNN, ensure that the K-norm and
   RoPE operations do not write to the input buffer if the same buffer is also
   used as V-norm input.

2. **KV cache writes.** After normalization and RoPE, both K and V tensors are
   written to the paged KV cache. Despite sharing the same projection source,
   K and V are **different tensors** at cache-write time due to the divergent
   post-processing. The KV cache must store both independently.

3. **Weight loading.** When loading HuggingFace checkpoints for global layers,
   there is no `v_proj.weight` key in the state dict. The TTNN weight loader
   must handle the absence of V projection weights gracefully for global layers
   while still expecting them for sliding layers.

4. **Fused QK weight construction.** If using fused projections, the weight
   packing code must concatenate `q_proj.weight` and `k_proj.weight` (not
   `v_proj.weight`) along the output dimension for global layers. This is a
   structural difference from the standard Q+K+V packing used elsewhere.

---

**Next:** [`vnorm_implementation.md`](./vnorm_implementation.md)
