# Global-Layer Proportional RoPE (p-RoPE)

This file covers the proportional RoPE configuration used by all 10 global
(full-causal) attention layers in Gemma 4 31B. Global p-RoPE uses a high base
frequency ($\theta = 1{,}000{,}000$) and a partial rotary factor of 0.25,
meaning only the first 128 of 512 head dimensions are rotated. The remaining
384 dimensions pass through unchanged. This combination is designed to support
robust long-context extrapolation to 256K tokens.

## Mathematical Formulation

### Inverse Frequencies

The inverse frequency vector for global p-RoPE is computed over the rotary
dimensions only. The effective rotary dimension is:

```math
\texttt{dim} = \left\lfloor \texttt{head\_dim} \times \texttt{partial\_rotary\_factor} \right\rfloor = \left\lfloor 512 \times 0.25 \right\rfloor = 128
```

The inverse frequency vector has 64 elements (one per pair of rotary
dimensions):

```math
\texttt{inv\_freq}[i] = \frac{1}{\theta^{2i / \texttt{dim}}} = \frac{1}{1{,}000{,}000^{2i / 128}} \quad \text{for } i = 0, 1, \ldots, 63
```

Note that the denominator uses `dim=128` (the rotary dimension count), **not**
the full `head_dim=512`. This means the frequency progression is scaled to the
partial rotary subspace.

### Cos/Sin Table Shape (Reference Behavior)

The HuggingFace reference implementation produces **narrow** cos/sin tables of
shape `[max_seq_len, 128]`, covering only the rotary dimensions:

```math
\texttt{cos\_table}[p, :] = \cos\bigl([p \cdot \texttt{inv\_freq}[0], \ldots, p \cdot \texttt{inv\_freq}[63]]\bigr) \quad \text{repeated twice}
```

The "repeated twice" step doubles the 64-element vector to 128 elements by
concatenation, matching the `rotate_half` convention.

### How Partial Rotation Works (Split-Apply-Concat)

The HuggingFace `apply_rotary_pos_emb` handles partial rotation via a
split-apply-concat pattern. When the cos/sin width (128) is narrower than
the head dimension (512), the function:

1. Splits the head tensor at the rotary boundary: `x_rot = x[..., :128]`,
   `x_pass = x[..., 128:]`.
2. Applies the standard RoPE rotation to `x_rot` using the narrow cos/sin
   tables.
3. Concatenates the result: `output = concat(rotated_x_rot, x_pass)`.

```math
\text{RoPE}(x, p) = [\text{rotate}(x_{0..127}, p), \; x_{128..511}]
```

The non-rotated dimensions (128--511) pass through unchanged without any
multiply-by-one overhead. This is the reference behavior and produces
numerically identical results to any correct implementation.

## Why p-RoPE for Global Layers

### High Theta for Long Context

Standard RoPE with $\theta = 10{,}000$ assigns relatively high frequencies to
most dimension pairs. At very long sequence lengths (100K+ tokens), the
highest-frequency pairs undergo many full rotations, which can degrade the
model's ability to distinguish positions far apart in the sequence. Raising
$\theta$ to $1{,}000{,}000$ lowers all frequencies by a factor of 100x,
stretching the wavelengths:

```math
\text{wavelength}_i = \frac{2\pi}{\texttt{inv\_freq}[i]} = 2\pi \cdot \theta^{2i/d}
```

For $\theta = 10^6$ and $d = 128$ (the correct rotary dimension):

- Lowest frequency pair ($i = 0$): wavelength $= 2\pi \approx 6.3$ tokens.
- Highest frequency pair ($i = 63$): wavelength $= 2\pi \cdot (10^6)^{126/128} \approx 2\pi \cdot 10^{5.91} \approx 5.1\text{M}$ tokens.

With the correct denominator of 128, **all** rotary pairs have very long
wavelengths. Even the lowest-indexed pair (i=0) has a wavelength of ~6.3
tokens, and the progression reaches ~5.1 million tokens for the
highest-indexed pair. This is the intended behavior: the combination of
$\theta = 10^6$ and $d = 128$ ensures that the rotary encoding space is
extremely spread out, providing robust long-context extrapolation well beyond
the 256K training context window.

### Partial Rotation as Semantic Channels

By rotating only 128 of 512 dimensions, the remaining 384 dimensions serve as
**pure semantic channels** that carry content information without any
positional modulation. This is analogous to the design philosophy behind NoPE
(No Position Encoding) models, where some capacity is reserved for
position-independent representations.

The 75/25 split (75% semantic, 25% positional) reflects a deliberate design
choice: global layers attend over the full sequence and need less positional
granularity per head dimension (since they have more positions to attend to),
while the high-dimensional semantic channels support richer content-based
retrieval across long distances.

### Contrast With Sliding Layers

Sliding layers use $\theta = 10{,}000$ with full rotation because their
1024-token window means they never need to distinguish positions more than
1024 tokens apart. The higher frequencies from $\theta = 10{,}000$ provide
finer-grained position discrimination within this short window.

| Aspect | Sliding | Global |
|--------|---------|--------|
| Context range | 1024 tokens (window) | 256K tokens (full) |
| Position discrimination need | Fine-grained within window | Coarse over long range |
| $\theta$ | 10,000 (higher frequencies) | 1,000,000 (lower frequencies) |
| Rotary dims | 256/256 (100%) | 128/512 (25%) |
| Semantic dims | 0 | 384 (75%) |

## Application in Global Layers

### Forward Pass Position

Like sliding layers, p-RoPE is applied to Q and K after projection and
normalization:

```text
hidden_states  [B, 1, 5376]
     |
     v
  q_proj  [5376, 16384]            k_proj  [5376, 2048]
     |                                  |
     v                                  +--- key_states [B, 1, 4, 512]
  query_states [B, 1, 32, 512]         +--- value_states [B, 1, 4, 512]  (K=V sharing)
     |                                  |
     v                                  v
  q_norm (scaled RMSNorm)          k_norm (scaled RMSNorm)
     |                                  |
     v                                  v
  transpose --> [B, 32, 1, 512]    transpose --> [B, 4, 1, 512]
     |                                  |
     v                                  v
  apply_rotary_pos_emb             apply_rotary_pos_emb
  (p-RoPE: 128/512 rotated)       (p-RoPE: 128/512 rotated)
     |                                  |
     v                                  v
  SDPA                             KV cache write
```

The V path receives no RoPE (see
[Chapter 3](../ch3_kv_sharing_and_vnorm/k_eq_v_mechanism.md)).

### Effect on Individual Dimensions

After p-RoPE application at position $p$, the 512-dimensional head vector is
partitioned as follows:

- **Dimensions 0--127:** Rotated by position-dependent angles. These carry
  both semantic and positional information.
- **Dimensions 128--511:** Unchanged (identity operation via cos=1, sin=0).
  These carry pure semantic information.

The attention dot product $Q \cdot K^T$ therefore has two additive components:

```math
Q \cdot K^T = \underbrace{\sum_{j=0}^{127} q_j k_j}_{\text{position-modulated}} + \underbrace{\sum_{j=128}^{511} q_j k_j}_{\text{position-invariant}}
```

The position-invariant term acts as a content-based similarity score that is
independent of where the tokens appear in the sequence.

## TTNN Mapping

### Implementation Strategy A --- Full-Width Tables (TTNN Optimization)

This approach constructs full-width cos/sin tables of shape
`[max_seq_len, 512]` with identity values (cos=1, sin=0) in the non-rotated
columns, allowing `TTNNRotaryPositionEmbedding` to be applied to the entire
head tensor without splitting. **Note:** This does NOT mirror the HuggingFace
reference implementation (which uses narrow tables with split-apply-concat).
It is a valid TTNN optimization that produces numerically identical results
by encoding the identity operation directly into the tables.

```python
# Precomputation (host-side, at model init)
inv_freq_rotated = 1.0 / (1_000_000.0 ** (torch.arange(0, 128, 2).float() / 128.0))  # [64]
inv_freq_nope = torch.zeros(192)                                                        # [192]
inv_freq = torch.cat([inv_freq_rotated, inv_freq_nope])                                 # [256]

positions = torch.arange(0, max_seq_len).float()
freqs = torch.outer(positions, inv_freq)        # [max_seq_len, 256]
emb = torch.cat([freqs, freqs], dim=-1)          # [max_seq_len, 512]
cos_table = emb.cos()                            # [max_seq_len, 512]
sin_table = emb.sin()                            # [max_seq_len, 512]
```

**Pros:**

- Uses the standard `TTNNRotaryPositionEmbedding` kernel without modification.
- No tensor splitting or concatenation in the forward pass.

**Cons:**

- Wastes compute: 384 of 512 dimensions are multiplied by cos=1 and sin=0.
- Wastes memory: the cos/sin tables are 4x larger than necessary (512 vs 128
  effective columns).
- Does not match the HuggingFace reference code path (may complicate
  numerical debugging).

### Implementation Strategy B --- Narrow Tables With Split-Apply-Concat (HuggingFace Reference)

This approach matches the HuggingFace reference implementation: narrow
cos/sin tables of shape `[max_seq_len, 128]` with explicit splitting of the
head tensor at the rotary boundary:

```python
# Precomputation (host-side)
inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, 128, 2).float() / 128.0))  # [64]
positions = torch.arange(0, max_seq_len).float()
freqs = torch.outer(positions, inv_freq)         # [max_seq_len, 64]
emb = torch.cat([freqs, freqs], dim=-1)           # [max_seq_len, 128]
cos_table = emb.cos()                             # [max_seq_len, 128]
sin_table = emb.sin()                             # [max_seq_len, 128]
```

```python
# Forward pass (device-side, per decode step)
rotary_dim = 128

# Split head tensor into rotary and pass-through slices
q_rot = query_states[:, :, :, :rotary_dim]      # [B, H, S, 128]
q_pass = query_states[:, :, :, rotary_dim:]      # [B, H, S, 384]

k_rot = key_states[:, :, :, :rotary_dim]         # [B, H, S, 128]
k_pass = key_states[:, :, :, rotary_dim:]         # [B, H, S, 384]

# Apply RoPE only to the rotary slice
q_rot, k_rot = ttnn_rope(q_rot, k_rot, cos, sin)

# Concatenate back
query_states = ttnn.concat([q_rot, q_pass], dim=-1)   # [B, H, S, 512]
key_states = ttnn.concat([k_rot, k_pass], dim=-1)     # [B, H, S, 512]
```

**Pros:**

- Saves compute: RoPE kernel only processes 128 dimensions instead of 512.
- Saves memory: cos/sin tables are 4x smaller.
- More explicit about which dimensions carry positional information.

**Cons:**

- Requires two `ttnn.slice` and two `ttnn.concat` operations per Q and K
  tensor per layer (4 slices + 4 concats total per layer).
- The slice/concat overhead may exceed the compute savings for small decode
  tensors (`[1, 1, H, D]` during single-token decode), where the RoPE kernel
  itself is dispatch-latency-bound rather than compute-bound.
- Incompatible with `TTNNDistributedRotaryPositionEmbedding` if that module
  assumes the full head dimension.

### Compatibility With `TTNNDistributedRotaryPositionEmbedding`

The distributed RoPE module in tt-symbiote applies rotary embedding across
multiple devices in the TP mesh. Based on analysis from the TTNNBailingMoEAttention
guide, `partial_rotary_factor < 1.0` forces the use of the non-distributed
`TTNNRotaryPositionEmbedding` variant in the current tt-symbiote codebase.

The reason: `TTNNDistributedRotaryPositionEmbedding` assumes the cos/sin
tables span the full head dimension and that all dimensions participate in
the rotation. When `partial_rotary_factor < 1.0`, the module would need to
either:

1. Accept full-width tables with zero-padded identity values (Strategy A), or
2. Internally handle the split-apply-concat pattern (Strategy B).

Neither is currently supported. Therefore, **global-layer RoPE must use the
non-distributed path**.

### Performance Impact of Non-Distributed RoPE

Using non-distributed RoPE for global layers means Q and K tensors may need
to be gathered to a single device, have RoPE applied, and then scattered back
--- or, more practically, each device applies RoPE independently to its local
Q/K slices using a non-distributed module instance.

Since RoPE is a per-element operation (each dimension is independently
rotated), it does not require cross-device communication. Each device can
apply `TTNNRotaryPositionEmbedding` to its local Q and K heads using the
same cos/sin tables. The "non-distributed" constraint means only that the
TTNN module wrapper is the single-device variant, not that the computation
is serialized to one device.

**Practical impact:** Minimal. The per-device RoPE application is identical
whether it uses `TTNNDistributedRotaryPositionEmbedding` or per-device
`TTNNRotaryPositionEmbedding` instances, because the underlying kernel is
the same. The distributed wrapper primarily adds multi-device tensor handling
(device mesh awareness), not algorithmic changes.

### Workaround: Enabling Distributed p-RoPE

To use `TTNNDistributedRotaryPositionEmbedding` with partial rotation, the
following changes would be needed:

1. **Modify the module to accept `partial_rotary_factor`** as a constructor
   parameter.
2. **Strategy A integration:** Accept full-width cos/sin tables and rely on
   the identity values to handle non-rotated dimensions. This requires no
   kernel changes --- only the module's validation logic needs to stop
   rejecting `partial_rotary_factor < 1.0`.
3. **Strategy B integration:** Add split-apply-concat logic inside the
   distributed module. This is more invasive but produces a cleaner
   computation graph.

Until these changes are made, the recommended approach is to use per-device
`TTNNRotaryPositionEmbedding` instances for global layers.

### Recommendation

For initial bringup, use **Strategy B (narrow tables with
split-apply-concat)** with per-device `TTNNRotaryPositionEmbedding`. This
matches the HuggingFace reference implementation, simplifying numerical
validation and debugging.

For optimization, evaluate **Strategy A (full-width tables)** if profiling
shows that the split/concat overhead exceeds the compute savings from
skipping 384 identity multiplications. Strategy A eliminates the
slice/concat operations at the cost of 4x larger table reads. The break-even
point depends on whether the RoPE kernel is memory-bandwidth-bound (favoring
narrow tables) or dispatch-latency-bound (favoring fewer ops).

---

**Next:** [`rope_precomputation.md`](./rope_precomputation.md)
