# V-Norm Implementation

This file covers the V-norm operation --- an RMSNorm variant with no learned
scale parameter --- that is applied to value vectors in every attention layer
of Gemma 4 31B. It defines the mathematical operation, traces through the
HuggingFace reference implementation, and analyzes three TTNN implementation
strategies with their trade-offs.

## Definition

V-norm is defined as:

```math
\text{v-norm}(v) = \frac{v}{\sqrt{\text{mean}(v^2) + \epsilon}}
```

where $\epsilon = 10^{-6}$ and the mean is computed over the last dimension
(`head_dim`).

Expanding the mean:

```math
\text{v-norm}(v) = \frac{v}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} v_i^2 + \epsilon}}
```

where $d$ is the head dimension (256 for sliding layers, 512 for global
layers).

### Contrast With Standard RMSNorm

Standard RMSNorm (used for `q_norm`, `k_norm`, `input_layernorm`, etc.)
includes a learned per-element scale:

```math
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma
```

where $\gamma \in \mathbb{R}^d$ is a trainable parameter initialized to ones.

V-norm omits the $\gamma$ multiplication entirely. The `Gemma4RMSNorm` class
implements this via the `with_scale` flag:

```python
class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)
```

When `with_scale=False`:

- No `self.weight` parameter is registered.
- The forward pass skips the $\gamma$ multiplication.
- The module has **zero trainable parameters**.

## Presence in All 60 Layers

V-norm is instantiated in **every** `Gemma4TextAttention` module, regardless
of layer type:

```python
self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
```

This means:

| Layer Type | Count | `head_dim` | V-norm dimension |
|------------|-------|------------|------------------|
| Sliding | 50 | 256 | 256 |
| Global | 10 | 512 | 512 |

The V-norm dimension matches the per-head dimension because the normalization
is applied **after** the V projection output has been reshaped into per-head
format `[B, num_kv_heads, S, head_dim]`. The mean and normalization operate
over the last axis (`head_dim`), independently for each head.

### Interaction With K=V Sharing

In global layers where K=V sharing is active, V-norm is applied to the
**raw K projection output** (before K-norm or RoPE has been applied to it).
The sequence is:

1. `shared = k_proj(hidden_states)` --- shape `[B, S, 2048]`
2. `shared = shared.view(B, S, 4, 512)` --- reshape into per-head format
3. K path: `k_norm(shared)` then RoPE --- produces K for cache
4. V path: `v_norm(shared)` --- produces V for cache (no RoPE)

V-norm and K-norm both receive the same input tensor but produce different
outputs because K-norm includes a learned scale and V-norm does not.

In sliding layers, V-norm is applied to the output of the separate `v_proj`:

1. `value_states = v_proj(hidden_states)` --- shape `[B, S, 4096]`
2. `value_states = value_states.view(B, S, 16, 256)` --- per-head format
3. `value_states = v_norm(value_states)` --- pure magnitude normalization

## Numerical Properties

### Output Magnitude

V-norm guarantees that the RMS magnitude of each value head vector is
approximately 1.0:

```math
\text{RMS}(\text{v-norm}(v)) = \sqrt{\text{mean}(\text{v-norm}(v)^2)} \approx 1.0
```

The approximation is exact up to the $\epsilon$ term. This stabilization
prevents value vectors with large magnitudes from dominating the attention
output.

### Gradient Flow

Because V-norm has no learned parameters, it acts as a fixed (non-trainable)
normalization during training. The gradients flow through the normalization
operation itself (via the chain rule on the division) but there is no $\gamma$
gradient to accumulate. This reduces the optimizer state by one vector per
layer compared to a standard RMSNorm.

### Float32 Upcast

The HuggingFace implementation performs the normalization in float32
(`hidden_states.float()`) and casts back to the input dtype afterward
(`.type_as(hidden_states)`). This is standard practice for norm layers to
prevent numerical instability in BF16, where the squared values and mean
computation can overflow or lose precision.

In TTNN, the equivalent behavior depends on the RMSNorm kernel
implementation. If the TTNN kernel operates internally in higher precision,
no explicit upcast is needed. Otherwise, the TTNN implementation should
ensure sufficient precision for the `mean(v^2)` computation.

## TTNN Implementation Options

### Option A --- `TTNNDistributedRMSNorm` With All-Ones Weight

The most straightforward approach reuses the existing `TTNNDistributedRMSNorm`
module by supplying a dummy weight tensor of all ones:

```python
# During module init
v_norm_weight = ttnn.ones([head_dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
# Store on device DRAM

# During forward
value_states = ttnn_distributed_rms_norm(value_states, v_norm_weight, eps=1e-6)
```

**Pros:**

- Uses an existing, tested, and optimized TTNN module.
- No custom kernel or op sequence needed.
- Correct behavior: multiplying by ones is a no-op, so the result matches
  pure RMSNorm.
- Works with the distributed (multi-device) RMSNorm variant out of the box.

**Cons:**

- Wastes a small amount of DRAM for the dummy weight tensor. Per layer:
  - Sliding: 256 elements x 2 bytes = 512 bytes
  - Global: 512 elements x 2 bytes = 1024 bytes
  - Total across 60 layers: ~36 KB (negligible).
- Wastes compute on the element-wise multiply by ones. In a tiled kernel this
  is typically fused with the normalization and adds minimal overhead, but it
  is not zero.
- Semantically misleading: the "weight" has no corresponding entry in the
  model checkpoint.

**Verdict:** Recommended as the **default implementation**. The memory overhead
is negligible (36 KB across all layers), the compute overhead of multiplying
by ones is marginal within a fused RMSNorm kernel, and this approach avoids
any custom op development.

### Option B --- `TTNNDistributedRMSNorm` With `with_scale=False` Path

If the TTNN RMSNorm kernel supports a `with_scale=False` or `weight=None`
mode, V-norm can be expressed directly without a dummy weight:

```python
value_states = ttnn_distributed_rms_norm(value_states, weight=None, eps=1e-6)
```

**Current status:** As of the tt-symbiote and tt-metal codebases at the time
of writing, `TTNNDistributedRMSNorm` and the underlying `ttnn.rms_norm` op
**expect a weight tensor**. The API does not expose a `with_scale=False`
parameter or accept `weight=None`.

**Required changes if pursuing this option:**

1. Modify the `ttnn.rms_norm` op to accept an optional weight parameter
   (defaulting to the current behavior when provided).
2. When weight is `None`, skip the element-wise multiply in the kernel.
3. Update `TTNNDistributedRMSNorm` to pass through `weight=None`.
4. Ensure the distributed variant handles the all-gather / reduce-scatter
   correctly when no weight is present (the norm computation itself does not
   depend on the weight; the weight multiply is a local, per-element
   operation that can simply be omitted).

**Pros:**

- Cleanest semantic mapping to the HuggingFace implementation.
- Zero wasted memory or compute.
- Future-proof: other models may adopt scale-free norms (V-norm is a
  design pattern, not specific to Gemma 4).

**Cons:**

- Requires kernel-level changes to TTNN.
- Must be tested for correctness and performance parity.
- The benefit over Option A is marginal for this model (36 KB memory,
  sub-microsecond compute).

**Verdict:** Worth implementing as a **long-term improvement** to the TTNN
RMSNorm API, but not a blocker for Gemma 4 31B bringup. Option A is
sufficient in the interim.

### Option C --- Manual TTNN Op Sequence

V-norm can be implemented manually using primitive TTNN operations:

```python
# v has shape [B, num_kv_heads, S, head_dim]
v_squared = ttnn.square(v)                                   # element-wise v^2
mean_sq = ttnn.mean(v_squared, dim=-1, keepdim=True)         # mean over head_dim
mean_sq_eps = ttnn.add(mean_sq, eps)                         # add epsilon
inv_rms = ttnn.rsqrt(mean_sq_eps)                            # 1 / sqrt(mean + eps)
v_normed = ttnn.mul(v, inv_rms)                              # normalize
```

**Pros:**

- No dependency on the RMSNorm kernel at all.
- Complete control over the computation.
- No dummy weight needed.

**Cons:**

- **Multiple kernel launches** instead of one fused RMSNorm call. Each of the
  5 operations (`square`, `mean`, `add`, `rsqrt`, `mul`) is a separate kernel
  dispatch on Wormhole. The dispatch overhead can dominate for small tensors
  during decode (single-token, few heads).
- **No fusion.** The fused RMSNorm kernel in TTNN performs all of these
  operations in a single pass over the data. The manual sequence reads and
  writes intermediate tensors to L1 or DRAM between each op.
- **Intermediate tensor memory.** The `v_squared` and `mean_sq` tensors
  require additional buffer space.
- **Harder to maintain.** Any future optimizations to the TTNN RMSNorm kernel
  (e.g., mixed-precision accumulation, tiled reduction) would not
  automatically benefit the manual implementation.

**Verdict:** Not recommended for production. Use only as a **debugging
reference** to validate the fused kernel output against a known-correct
manual computation.

## Performance Comparison

The following table estimates the relative cost of each option for a single
V-norm invocation during decode (`B=1, S=1`). Absolute latencies are
hardware-dependent but the relative ordering is stable.

| Metric | Option A (all-ones weight) | Option B (with\_scale=False) | Option C (manual ops) |
|--------|---------------------------|-----------------------------|-----------------------|
| Kernel launches | 1 | 1 | 5 |
| DRAM reads (weight) | 1 (dummy, ~0.5--1 KB) | 0 | 0 |
| Intermediate tensors | 0 | 0 | 2--3 |
| Fused computation | Yes | Yes | No |
| Implementation effort | None | Kernel modification | None (but fragile) |
| Memory overhead | ~36 KB total | 0 | Transient buffers |
| Dispatch overhead | Baseline | Baseline | ~5x baseline |

For decode workloads where the V-norm tensor is small (`[1, num_kv_heads, 1,
head_dim]`), the dominant cost is kernel dispatch latency, not compute. A
single fused kernel (Options A or B) is strictly better than five separate
dispatches (Option C).

## Recommended Implementation Strategy

1. **For initial bringup:** Use **Option A** (`TTNNDistributedRMSNorm` with
   all-ones weight). This is correct, requires no TTNN modifications, and the
   overhead is negligible.

2. **For optimization:** If profiling reveals that the all-ones multiply adds
   measurable overhead (unlikely but possible in tight decode loops with Metal
   Trace), transition to **Option B** by adding `with_scale=False` support to
   the TTNN RMSNorm kernel.

3. **For debugging:** Keep **Option C** (manual ops) available as a reference
   implementation to validate correctness of the fused kernel output.

## V-Norm Fusion Opportunities

### Fusion With KV Cache Write

During decode, the V-norm output is immediately written to the paged KV cache.
A fused "norm-and-write" kernel could combine the RMSNorm computation with the
cache page update, eliminating one read-write cycle:

```text
v [B, Hkv, 1, D] --> v_norm --> v_normed [B, Hkv, 1, D] --> page_table write
                                           ^
                            (potential fusion point)
```

This is a speculative optimization that depends on the KV cache write kernel
supporting inline normalization. See
[Chapter 8](../ch8_performance/index.md) for the full optimization roadmap.

### Fusion With V Projection (Sliding Layers Only)

In sliding layers where V has a separate projection, V-norm could potentially
be fused with the `v_proj` matmul epilogue. The matmul output would be
RMS-normalized in-place as part of the output writeback. This is a more
aggressive fusion that requires matmul kernel support for custom epilogues.

## Weight Loading Considerations

V-norm has **no weights to load**. When processing a HuggingFace checkpoint:

- The state dict contains `model.layers.{i}.self_attn.v_norm` as a key
  prefix, but there is **no `.weight` subkey** under it (since
  `with_scale=False` means no `nn.Parameter` is registered).
- The TTNN weight loader must not expect or search for a `v_norm.weight`
  entry.
- If using Option A (all-ones dummy weight), the weight is constructed
  programmatically during module initialization, not loaded from the
  checkpoint.

This contrasts with `q_norm` and `k_norm`, which both have `.weight` entries
in the state dict (learned $\gamma$ vectors of dimension `head_dim`).

| Norm | State Dict Key | Shape | Loaded? |
|------|---------------|-------|---------|
| `q_norm` | `layers.{i}.self_attn.q_norm.weight` | [head_dim] | Yes |
| `k_norm` | `layers.{i}.self_attn.k_norm.weight` | [head_dim] | Yes |
| `v_norm` | (none) | --- | No (no weight exists) |

---

**Next:** [Chapter 4 --- Dual RoPE and Partial Rotary Embedding](../ch4_dual_rope/index.md)
