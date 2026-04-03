# Sliding-Layer RoPE: Standard Full Rotation

This file covers the standard RoPE configuration used by all 50 sliding-window
attention layers in Gemma 4 31B. Sliding RoPE uses the original RoPE
formulation with $\theta = 10{,}000$ and full rotation across every dimension
of the 256-dimensional attention heads.

## Mathematical Formulation

### Inverse Frequencies

RoPE encodes position by rotating pairs of dimensions at different
frequencies. The inverse frequency vector for sliding layers is:

```math
\texttt{inv\_freq}[i] = \frac{1}{\theta^{2i / d}} = \frac{1}{10000^{2i / 256}} \quad \text{for } i = 0, 1, \ldots, 127
```

where $d = 256$ is the head dimension and the vector has 128 elements (one per
pair of dimensions).

### Cos/Sin Tables

Given a position index $p$ and the inverse frequency vector, the cos and sin
values are:

```math
\cos(p \cdot \texttt{inv\_freq}[i]) \quad \text{and} \quad \sin(p \cdot \texttt{inv\_freq}[i]) \quad \text{for } i = 0, \ldots, 127
```

The full cos/sin tables are constructed for all positions up to
`max_position_embeddings`:

```math
\texttt{cos\_table}[p, :] = \cos\bigl([p \cdot \texttt{inv\_freq}[0], \; p \cdot \texttt{inv\_freq}[1], \; \ldots, \; p \cdot \texttt{inv\_freq}[127]]\bigr) \quad \text{repeated twice}
```

The "repeated twice" step doubles the 128-element vector to 256 elements by
concatenation: `[cos_0, cos_1, ..., cos_127, cos_0, cos_1, ..., cos_127]`.
This matches the `rotate_half` convention used in HuggingFace, where the
rotation operates on the first and second halves of the head dimension.

The resulting table shape is `[max_seq_len, 256]`, matching `head_dim`.

### Rotation Operation

For a query or key vector $x \in \mathbb{R}^{256}$, the rotated vector is:

```math
\text{RoPE}(x, p) = x \odot \cos(p \cdot \texttt{inv\_freq\_expanded}) + \text{rotate\_half}(x) \odot \sin(p \cdot \texttt{inv\_freq\_expanded})
```

where $\odot$ denotes element-wise multiplication and `rotate_half` splits the
vector into two halves and negates the first:

```math
\text{rotate\_half}([x_0, x_1, \ldots, x_{127}, x_{128}, \ldots, x_{255}]) = [-x_{128}, -x_{129}, \ldots, -x_{255}, x_0, x_1, \ldots, x_{127}]
```

## Application in Sliding Layers

### Forward Pass Position

RoPE is applied to Q and K **after** projection, normalization, and the
transpose into SDPA layout:

```text
hidden_states  [B, 1, 5376]
     |
     v
  q_proj / k_proj  (linear)
     |
     v
  q_norm / k_norm  (scaled RMSNorm)
     |
     v
  transpose(1, 2)  -->  [B, S, H, D] to [B, H, S, D]
     |
     v
  apply_rotary_pos_emb(states, cos, sin)   <-- RoPE here (after transpose)
     |
     v
  SDPA
```

### Reference Code (HuggingFace)

The sliding-layer forward pass applies RoPE identically to Q and K, after
transposing into `[B, H, S, D]` layout:

```python
query_states = self.q_proj(hidden_states).view(hidden_shape)  # [B, S, 32, 256]
query_states = self.q_norm(query_states)
query_states = query_states.transpose(1, 2)                   # [B, 32, S, 256]
query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=1)

key_states = self.k_proj(hidden_states).view(hidden_shape)    # [B, S, 16, 256]
key_states = self.k_norm(key_states)
key_states = key_states.transpose(1, 2)                       # [B, 16, S, 256]
key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=1)
```

The `unsqueeze_dim=1` parameter broadcasts the cos/sin vectors (which lack the
head-count dimension) across all query or key heads. Note that RoPE is applied
**after** the transpose, so the tensor layout is `[B, H, S, D]` at the point
of application.

### All 256 Dimensions Are Rotated

Because `partial_rotary_factor=1.0` for sliding layers, every dimension of the
head vector participates in the rotation. There is no split-apply-concat step
--- the full `[B, H, S, 256]` tensor passes through `apply_rotary_pos_emb` as
a single operation. This is the simplest and most efficient RoPE path.

## TTNN Mapping

### Module Choice

Sliding-layer RoPE maps directly to the existing TTNN rotary embedding modules:

- **Single-device:** `TTNNRotaryPositionEmbedding`
- **Multi-device (TP):** `TTNNDistributedRotaryPositionEmbedding`

Both modules accept precomputed cos/sin tables and apply the rotation in a
single fused kernel. Because all 256 dimensions are rotated
(`partial_rotary_factor=1.0`), the standard code path applies without
modification.

### Cos/Sin Table Precomputation

The cos and sin tables are computed once at model initialization:

```python
# Pseudocode for TTNN cos/sin table construction
inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 256, 2).float() / 256.0))  # [128]
positions = torch.arange(0, max_seq_len).float()                          # [max_seq_len]
freqs = torch.outer(positions, inv_freq)                                   # [max_seq_len, 128]
emb = torch.cat([freqs, freqs], dim=-1)                                    # [max_seq_len, 256]
cos_table = emb.cos()                                                      # [max_seq_len, 256]
sin_table = emb.sin()                                                      # [max_seq_len, 256]
```

These tables are converted to `ttnn.Tensor` with `bfloat16` dtype and
`TILE_LAYOUT`, then stored in device DRAM.

### Device Placement

Under TP=8 on T3K, each device holds Q heads with `head_dim=256` and K heads
with `head_dim=256`. The cos/sin tables are the same on all devices (they
depend only on position and dimension index, not on head identity). Two
placement strategies:

**Option 1 --- Replicate on all devices:**

Each device stores a full copy of the cos/sin tables in DRAM. During decode,
each device reads the row corresponding to the current position index.

- Memory per device: 2 tables x `max_seq_len` x 256 x 2 bytes.
- At 256K context: 2 x 262144 x 256 x 2 = 256 MB per device.
- Advantage: no cross-device communication for RoPE.

**Option 2 --- Store once, broadcast per step:**

A single device holds the tables; per-step slices are broadcast via CCL. This
saves DRAM but adds communication latency per decode step.

- Memory: 256 MB on one device, 0 on the other 7.
- Disadvantage: broadcast latency on every decode step.

**Recommendation:** Option 1 (replicate). The 256 MB cost is acceptable given
12 GB DRAM per chip, and it avoids per-step CCL overhead. For shorter context
deployments (e.g., 8K or 32K max sequence length), the cost drops to 8 MB or
32 MB respectively.

### Per-Step Slicing

During decode at position $p$, the TTNN forward pass slices a single row from
each table:

```python
cos = cos_table[p:p+1, :]   # [1, 256]
sin = sin_table[p:p+1, :]   # [1, 256]
```

This slice is then broadcast across batch and head dimensions during the
rotary kernel execution. For prefill with sequence length $S$, the slice spans
rows $0$ through $S-1$.

### Distributed RoPE Compatibility

`TTNNDistributedRotaryPositionEmbedding` applies the rotation across multiple
devices in the TP mesh. For sliding layers this works without issues because:

1. All 256 dimensions are rotated (no partial factor).
2. The cos/sin tables are identical on all devices.
3. Each device applies the rotation to its local Q and K head slices
   independently --- RoPE is a per-element operation that does not require
   cross-head or cross-device coordination.

The distributed variant is preferred for sliding layers because it avoids
gathering Q/K tensors to a single device before RoPE and scattering them back
afterward.

## Sliding Window and RoPE Interaction

The sliding window of 1024 tokens affects which positions are visible during
attention but does **not** change the RoPE encoding. Each token receives a
RoPE encoding based on its absolute position in the sequence, not its position
within the window. This means:

- Token at absolute position 5000 receives `cos_table[5000]` and
  `sin_table[5000]`, regardless of the 1024-token window.
- The attention mask restricts the dot-product computation to the nearest 1024
  tokens, but the positional encoding itself is global.

This design preserves the relative position information within the window: two
tokens separated by $k$ positions always have the same relative RoPE phase
difference $\Delta\phi = k \cdot \texttt{inv\_freq}$, which is the
foundational property that makes RoPE effective.

---

**Next:** [`global_proportional_rope.md`](./global_proportional_rope.md)
