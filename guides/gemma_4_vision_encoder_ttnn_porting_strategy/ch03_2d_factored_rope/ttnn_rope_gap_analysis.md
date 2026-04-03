# TTNN RoPE Gap Analysis

This file identifies the gaps between current TTNN RoPE capabilities and the requirements of Gemma 4's 2D factored vision RoPE, then evaluates three implementation strategies ranked by effort.

## Current TTNN RoPE Capabilities

The existing TTNN RoPE infrastructure was built for **language model decoding** and is optimized for that use case:

| Capability | Description |
|-----------|-------------|
| `ttnn.experimental.rotary_embedding_llama` | Applies 1D RoPE to Q and K tensors; supports decode mode with precomputed rotation matrices on device |
| Fused RoPE kernel | Applies RoPE to Q and K in parallel, each height-sharded across a distinct set of cores (see [tt-metal #14540](https://github.com/tenstorrent/tt-metal/issues/14540)) |
| cos/sin cache | Rotation matrices are precomputed and cached on device, eliminating per-step host-device transfers |
| Position indexing | Accepts a scalar position index per sequence element (1D position) |
| Sharding | Height-sharded across cores for decode; optimized for batch-of-1 autoregressive generation |

### Assumptions Baked Into Current Kernels

1. **1D position**: Each token has a single integer position $m$. The kernel computes or looks up $\cos(m \cdot \omega_i)$ and $\sin(m \cdot \omega_i)$ for each frequency $\omega_i$.

2. **Full head dimension**: The rotation is applied to the entire head dimension $d$ in one pass. There is no concept of splitting $d$ into sub-ranges for different position coordinates.

3. **Standard theta**: While `rope_theta` is configurable, the kernels are tested and optimized for values like 10,000 and 500,000. The value 100.0 is unusual but should work --- it only changes the `inv_freq` values.

4. **Decode-optimized**: The fused kernel is designed for single-token decode steps (sequence length = 1 per batch element). Prefill mode (arbitrary sequence length) may use a different code path.

## Gap Analysis

### Gap 1: 2D Coordinate Pair Needed (Severity: High)

**Requirement:** Gemma 4 vision RoPE takes position_ids of shape `[batch, num_patches, 2]`, where the last dimension contains `(x, y)` grid coordinates. Each spatial dimension is processed independently to produce separate cos/sin values that are concatenated.

**Current state:** TTNN RoPE kernels accept a 1D position index per token. There is no mechanism to pass a 2D coordinate pair and route each coordinate to different frequency computations.

**Impact:** This is the fundamental mismatch. The kernel cannot be used as-is for 2D positions. Either the cos/sin tables must be precomputed externally and the kernel bypassed, or the kernel must be extended.

### Gap 2: Head Dimension Split Not Natively Supported (Severity: High)

**Requirement:** The head dimension (72) is split into two halves of 36. The first half is rotated using x-position frequencies, the second half using y-position frequencies. The `apply_multidimensional_rope` function performs `torch.split` on Q/K along the last dimension, applies RoPE independently to each chunk, and concatenates.

**Current state:** The TTNN RoPE kernel rotates the entire head dimension using a single set of cos/sin values. There is no built-in split-apply-concat pattern.

**Impact:** Even if we solve Gap 1 by precomputing the concatenated cos/sin table `[batch, seq, 72]`, the `rotate_half` operation inside the kernel would operate on the full 72 dimensions. This is **incorrect** for 2D factored RoPE: `rotate_half` must operate independently on each 36-element chunk. The split boundary must be respected.

> **Warning:** This is subtle. If you pass a concatenated `[72]`-dimensional cos/sin vector to a standard RoPE kernel that does `rotate_half` on all 72 dimensions, the first 36 dimensions (x-component) would be mixed with the last 36 (y-component) during the half-swap. This produces numerically wrong results. The rotation must be applied per-chunk.

### Gap 3: rope_theta=100.0 Non-Standard (Severity: Low)

**Requirement:** `rope_theta=100.0`, producing much higher-frequency rotations than typical text model values.

**Current state:** The theta parameter is typically configurable in the frequency computation. The TTNN kernel itself does not hardcode theta --- it receives precomputed cos/sin values or an inv_freq table.

**Impact:** No functional gap. The value 100.0 produces valid floating-point results. The only concern is that higher-frequency rotations mean cos/sin values change more rapidly across adjacent positions, which could amplify BF16 quantization effects. However, the rotation angles are well within the normal range (not near floating-point edge cases), so this should not cause numerical issues.

> **Tip:** Validate BF16 PCC against a float32 reference specifically for the high-frequency entries ($i = 0, 1$) where the rotation angle per position step is largest ($\Delta\phi = 1.0$ for $i=0$). If PCC drops below 0.999, consider computing the frequency table in float32 and casting to BF16 only for the final cos/sin values.

## Gap Summary Table

| Gap | Description | Severity | Workaround Available? |
|-----|-------------|----------|----------------------|
| 1 | 2D position coordinates | High | Yes: precompute cos/sin on CPU |
| 2 | Head dimension split for rotate_half | High | Yes: split/concat with TTNN ops or precompute correctly shaped tables |
| 3 | rope_theta=100.0 | Low | No workaround needed; works as-is |

## Implementation Strategies

### Strategy 1: Precompute on CPU, Apply on Device (Lowest Effort)

**Approach:** Compute the full `[batch, num_patches, 72]` cos and sin tables on CPU using the HuggingFace reference code, transfer them to the device, and apply the rotation using element-wise TTNN operations.

**Implementation:**

```python
import torch
import ttnn

def precompute_2d_rope_cos_sin(position_ids, inv_freq, device=None):
    """
    Precompute cos/sin tables on CPU exactly matching HuggingFace reference.

    Args:
        position_ids: [batch, num_patches, 2] — (x, y) grid coordinates
        inv_freq: [18] — inverse frequency table
    Returns:
        cos: [batch, 1, num_patches, 72] — ready for broadcasting over heads
        sin: [batch, 1, num_patches, 72]
    """
    all_cos, all_sin = [], []
    inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)

    for dim_idx in range(2):
        dim_pos = position_ids[:, :, dim_idx].float()        # [batch, num_patches]
        dim_pos_expanded = dim_pos[:, None, :]                # [batch, 1, num_patches]
        freqs = (inv_freq_expanded @ dim_pos_expanded).transpose(1, 2)
                                                              # [batch, num_patches, 18]
        emb = torch.cat((freqs, freqs), dim=-1)               # [batch, num_patches, 36]
        all_cos.append(emb.cos())
        all_sin.append(emb.sin())

    cos = torch.cat(all_cos, dim=-1).unsqueeze(1)             # [batch, 1, num_patches, 72]
    sin = torch.cat(all_sin, dim=-1).unsqueeze(1)             # [batch, 1, num_patches, 72]
    return cos, sin


def apply_2d_rope_ttnn(x_tt, cos_tt, sin_tt):
    """
    Apply 2D factored RoPE on device using element-wise TTNN ops.

    Args:
        x_tt: [batch, num_heads, num_patches, 72] on device
        cos_tt: [batch, 1, num_patches, 72] on device (broadcasts over heads)
        sin_tt: [batch, 1, num_patches, 72] on device
    Returns:
        Rotated tensor, same shape as x_tt
    """
    # Split along head_dim into two halves of 36
    x_first = x_tt[:, :, :, :36]    # x-axis component
    x_second = x_tt[:, :, :, 36:]   # y-axis component
    cos_first = cos_tt[:, :, :, :36]
    cos_second = cos_tt[:, :, :, 36:]
    sin_first = sin_tt[:, :, :, :36]
    sin_second = sin_tt[:, :, :, 36:]

    # Apply rotate_half to each half independently
    def rotate_half_ttnn(t):
        t1 = t[:, :, :, :18]
        t2 = t[:, :, :, 18:]
        return ttnn.concat([ttnn.neg(t2), t1], dim=-1)

    y_first = ttnn.add(
        ttnn.mul(x_first, cos_first),
        ttnn.mul(rotate_half_ttnn(x_first), sin_first),
    )
    y_second = ttnn.add(
        ttnn.mul(x_second, cos_second),
        ttnn.mul(rotate_half_ttnn(x_second), sin_second),
    )

    return ttnn.concat([y_first, y_second], dim=-1)
```

**Pros:**
- Fastest to implement (1-2 days)
- Numerically identical to HuggingFace reference (CPU computation is float32)
- No kernel development required
- Unblocks attention layer validation immediately

**Cons:**
- Host-to-device transfer of cos/sin tables every time the image resolution changes
- Multiple element-wise ops (split, mul, neg, concat) instead of a single fused kernel
- Cannot be traced as a single op; the sequence of element-wise ops may not fuse optimally

**Transfer overhead estimate:** For `batch=1, num_patches=840`: the cos and sin tables are $840 \times 72 \times 2 \times 2 = 242$ KB in BF16. At PCIe Gen4 bandwidth (~12 GB/s), transfer time is $\sim 20$ microseconds. This is negligible compared to the attention matmul cost ($\sim 100+$ microseconds).

> **Tip:** Cache the cos/sin tables per resolution. The five standard token budgets (70, 140, 280, 560, 1120) correspond to a finite set of grid dimensions. Precompute and cache cos/sin for each, then select at runtime based on the image. This eliminates recomputation and transfer after the first image at each resolution.

### Strategy 2: Compose from Existing TTNN Ops (Medium Effort)

**Approach:** Compute `inv_freq` on device, use TTNN matmul to compute the outer product with position IDs, apply cos/sin via TTNN element-wise ops, and perform the rotation entirely on device.

**Implementation outline:**

```python
def compute_and_apply_2d_rope_ttnn(
    x_tt,               # [batch, num_heads, num_patches, 72] on device
    position_ids_tt,    # [batch, num_patches, 2] on device
    inv_freq_tt,        # [18] on device (stored once at init)
):
    """
    Full on-device 2D RoPE: compute cos/sin tables and apply.
    """
    all_cos, all_sin = [], []

    for dim_idx in range(2):
        # Extract positions for this dimension
        dim_pos = position_ids_tt[:, :, dim_idx:dim_idx+1]  # [batch, num_patches, 1]

        # Outer product: positions x frequencies
        # [batch, num_patches, 1] @ [1, 1, 18] -> [batch, num_patches, 18]
        inv_freq_row = ttnn.reshape(inv_freq_tt, [1, 1, 18])
        freqs = ttnn.matmul(dim_pos, inv_freq_row)           # [batch, num_patches, 18]

        # Double for rotate_half
        emb = ttnn.concat([freqs, freqs], dim=-1)            # [batch, num_patches, 36]

        cos_dim = ttnn.cos(emb)
        sin_dim = ttnn.sin(emb)
        all_cos.append(cos_dim)
        all_sin.append(sin_dim)

    cos_tt = ttnn.concat(all_cos, dim=-1)                    # [batch, num_patches, 72]
    sin_tt = ttnn.concat(all_sin, dim=-1)

    # Unsqueeze for head broadcasting, then apply as in Strategy 1
    # ... (same apply logic as Strategy 1)
```

**Pros:**
- Fully on-device; no host-device transfers for cos/sin tables
- Uses only standard TTNN ops (matmul, concat, cos, sin, mul, neg)
- Position IDs can be transferred once (small: $840 \times 2 \times 4 = 6.7$ KB for int32)

**Cons:**
- Requires `ttnn.cos` and `ttnn.sin` element-wise ops to be available and efficient
- More complex than Strategy 1; more ops to compose and validate
- The loop over 2 dimensions adds sequential dependency (though each iteration is independent and could be parallelized)
- Still not a single fused kernel; many small ops that may underutilize the device

**Estimated effort:** 3-5 days including validation.

> **Warning:** Verify that `ttnn.cos` and `ttnn.sin` are available and numerically accurate in BF16. Transcendental functions in reduced precision can introduce errors, especially at the higher rotation angles produced by the low-index frequencies (where $\phi$ can be several radians). If accuracy is insufficient, compute cos/sin in float32 and cast to BF16 for the multiply.

### Strategy 3: Custom TTNN Kernel (Highest Effort)

**Approach:** Implement a dedicated fused 2D RoPE kernel that:
1. Takes Q/K tensors and 2D position IDs as inputs
2. Internally computes the frequency table, applies rotation to both spatial halves, and outputs the rotated tensors
3. Processes Q and K in parallel (similar to the existing fused 1D RoPE kernel)

**Design sketch:**

```
Inputs:
  - Q: [batch, num_heads_q, seq, head_dim]  (height-sharded across Q-cores)
  - K: [batch, num_heads_k, seq, head_dim]  (height-sharded across K-cores)
  - position_ids: [batch, seq, 2]           (broadcast or replicated)
  - inv_freq: [head_dim/4]                  (compile-time constant)

Kernel logic (per core, per head, per patch):
  1. Load Q[h, s, :] and K[h, s, :] from L1
  2. Load position_ids[s, :] = (x, y)
  3. For each frequency index i = 0..17:
       angle_x = x * inv_freq[i]
       angle_y = y * inv_freq[i]
       // Rotate Q[h, s, 2i:2i+2] by angle_x (first half)
       // Rotate Q[h, s, 36+2i:36+2i+2] by angle_y (second half)
       // Same for K
  4. Write rotated Q, K back to L1

Outputs:
  - Q_rotated: same shape as Q
  - K_rotated: same shape as K
```

**Pros:**
- Maximum performance: single kernel call, no intermediate tensors, no host-device transfers
- Fuses frequency computation, cos/sin evaluation, and rotation into one pass
- Can be height-sharded across cores like the existing fused 1D RoPE kernel
- Eliminates all the split/concat overhead

**Cons:**
- Significant development effort: new kernel code, new compute kernel, new reader/writer
- Requires kernel testing infrastructure (unit tests, PCC validation, edge cases)
- Must handle variable sequence lengths (different image resolutions)
- Maintenance burden: another custom kernel to support across hardware generations

**Estimated effort:** 2-3 weeks for initial implementation, plus 1 week for testing and edge cases.

## Strategy Comparison

| Criterion | Strategy 1: CPU Precompute | Strategy 2: TTNN Compose | Strategy 3: Custom Kernel |
|-----------|---------------------------|-------------------------|--------------------------|
| **Implementation effort** | 1-2 days | 3-5 days | 2-4 weeks |
| **Numerical accuracy** | Excellent (float32 CPU) | Good (depends on ttnn.cos/sin) | Good (needs validation) |
| **Host-device transfer** | ~20 us per resolution (cacheable) | None (position IDs only: ~7 KB) | None |
| **On-device op count** | ~10 element-wise ops per Q/K | ~15 ops per Q/K (including cos/sin) | 1 fused op |
| **Traceable** | Yes (fixed cos/sin tensors) | Yes (if ops are traceable) | Yes |
| **Latency overhead vs. ideal** | Low (~5-10% of attention) | Medium (~10-15% of attention) | Minimal (~1-2% of attention) |
| **Dependencies** | None | ttnn.cos, ttnn.sin availability | Kernel development infrastructure |

## Recommendation

### For Initial Bringup: Strategy 1 (CPU Precompute)

Use Strategy 1 to unblock the attention layer port immediately. The 20-microsecond transfer overhead per resolution is negligible, and caching eliminates it for repeated resolutions. This approach:

- Delivers numerically correct results on day one
- Allows validating the full attention layer (Q/K projection, QK-norm, RoPE, SDPA, output projection) without waiting for any kernel work
- Provides the baseline PCC numbers against which optimized implementations are measured

### For Optimized Deployment: Strategy 1 with Precomputed Cached Tables

For production deployment, Strategy 1 with aggressive caching is likely sufficient. The RoPE overhead is a small fraction of total attention latency, and the five standard token budgets mean at most five cos/sin table pairs to cache.

Strategy 3 (custom kernel) is justified **only** if profiling reveals that the element-wise RoPE application is a significant bottleneck (unlikely, given that the attention matmuls dominate) or if the model is deployed in a latency-critical scenario where every microsecond matters.

Strategy 2 is a reasonable middle ground if the team wants to avoid any host-device transfers, but the additional complexity over Strategy 1 is hard to justify given the small transfer size.

### Decision Tree

```
Is this initial bringup?
├── Yes → Strategy 1 (CPU precompute)
└── No → Is RoPE overhead > 5% of attention latency?
    ├── No → Strategy 1 with caching (sufficient)
    └── Yes → Is kernel dev capacity available?
        ├── Yes → Strategy 3 (custom kernel)
        └── No → Strategy 2 (TTNN compose)
```

> **Tip:** Do not start with Strategy 3. The vision encoder has 27 layers, but the cos/sin tables are computed once and reused. The per-layer application cost is just element-wise multiplies and adds on tensors of size `[batch, 16, num_patches, 72]`. For `num_patches=840`, this is $\sim 967$K elements per Q or K --- well within the range where element-wise TTNN ops are fast. Profile first, optimize only if the data demands it.

## Bringup Checklist

For the recommended Strategy 1 approach, the implementation steps are:

- [ ] Implement `precompute_2d_rope_cos_sin()` on CPU using the HuggingFace reference
- [ ] Validate cos/sin tables against HuggingFace output (exact match in float32)
- [ ] Transfer cos/sin to device as BF16 tensors with shape `[batch, 1, num_patches, 72]`
- [ ] Implement `apply_2d_rope_ttnn()` using element-wise TTNN ops (split, mul, neg, concat)
- [ ] Validate rotated Q/K against HuggingFace reference (PCC > 0.999 in BF16)
- [ ] Cache cos/sin tables for the five standard token budgets
- [ ] Integrate into `Gemma4VisionAttention` forward pass between QK-norm and transpose
- [ ] Profile and measure RoPE overhead as a fraction of total attention latency

---

**Next:** [Chapter 4 — Patch Embedding and Adaptive Pooling](../ch04_patch_embedding_and_pooling/index.md) — Porting the two vision-specific operations that differ most from Gemma 3.
