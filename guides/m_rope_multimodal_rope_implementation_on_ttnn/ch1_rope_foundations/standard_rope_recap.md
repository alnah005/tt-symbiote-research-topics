# Standard RoPE and Partial RoPE: Mathematical Recap

This file recaps standard 1D RoPE from first principles, derives the rotate-half
operation algebraically, introduces partial RoPE, and applies all of these to
the concrete Qwen3.6 numbers that recur throughout this guide.

## Frequency Table Construction

### Inverse Frequencies

RoPE encodes position by rotating consecutive pairs of dimensions at distinct
angular frequencies. The frequency for pair index `i` is the inverse frequency:

```math
\theta_i = \frac{1}{\texttt{rope\_theta}^{\,2i \,/\, \texttt{rotary\_dim}}}
\quad \text{for } i = 0, 1, \ldots, \frac{\texttt{rotary\_dim}}{2} - 1
```

where `rope_theta` is the base (commonly 10,000 for original RoPE, 1,000,000
for Qwen3.6) and `rotary_dim` is the number of dimensions that will be rotated.
The exponent `2i / rotary_dim` increases from 0 to `(rotary_dim - 2) / rotary_dim`
as `i` runs over `[0, rotary_dim/2)`, so `θ_i` decreases monotonically from
`1.0` (at `i=0`) toward zero (at `i = rotary_dim/2 - 1`). Low-frequency pairs
carry coarse positional structure; high-frequency pairs encode fine-grained
position differences.

The resulting `inv_freq` vector has length `rotary_dim / 2` — one scalar per
rotation pair.

### Concrete Example: Qwen3.6 Text Layer

For Qwen3.6-35B-A3B text attention layers:

```
rope_theta          = 1,000,000
head_dim            = 128
partial_rotary_factor = 0.5
rotary_dim          = floor(128 * 0.5) = 64
rotary_dim / 2      = 32 pairs  (indices i = 0..31)
```

```python
import torch

rope_theta   = 1_000_000.0
rotary_dim   = 64
i            = torch.arange(0, rotary_dim // 2, dtype=torch.float32)  # [0..31]
inv_freq     = 1.0 / (rope_theta ** (2 * i / rotary_dim))             # shape [32]
# inv_freq[0]  = 1.0 rad/token  (highest frequency; period ≈ 2π ≈ 6.28 tokens)
# inv_freq[31] = 1.0 / (1e6 ** (62/64)) ≈ 1.54e-6  (very slow rotation)
```

### Precomputing Cos and Sin Tables

Given `inv_freq` of shape `[rotary_dim/2]` and a set of sequence positions
`t ∈ {0, 1, …, max_seq_len - 1}`, the outer product produces all
`(position, pair)` angular values:

```math
\texttt{freqs}[t, i] = t \cdot \theta_i
\quad \text{for } t \in [0, \texttt{max\_seq\_len}), \; i \in [0, \tfrac{\texttt{rotary\_dim}}{2})
```

The cos and sin tables are then constructed by concatenating each row with
itself (the "double" step that matches the rotate-half convention):

```math
\texttt{emb}[t, :] = [\,t\theta_0,\; t\theta_1,\; \ldots,\; t\theta_{\frac{r}{2}-1},\;\;
                       t\theta_0,\; t\theta_1,\; \ldots,\; t\theta_{\frac{r}{2}-1}\,]
```

where `r = rotary_dim`. The full-length tables are:

```math
\texttt{cos\_table}[t, :] = \cos(\texttt{emb}[t, :]) \in \mathbb{R}^{\texttt{rotary\_dim}}
```
```math
\texttt{sin\_table}[t, :] = \sin(\texttt{emb}[t, :]) \in \mathbb{R}^{\texttt{rotary\_dim}}
```

Both tables have shape `[max_seq_len, rotary_dim]`.

```python
positions  = torch.arange(0, max_seq_len, dtype=torch.float32)           # [T]
freqs      = torch.outer(positions, inv_freq)                             # [T, 32]
emb        = torch.cat([freqs, freqs], dim=-1)                            # [T, 64]
cos_table  = emb.cos()                                                    # [T, 64]
sin_table  = emb.sin()                                                    # [T, 64]
```

The concatenation is the crucial step: the first 32 columns of `cos_table[t]`
are `cos(t·θ_0)…cos(t·θ_31)` and the second 32 columns are the exact same
values. This double layout makes the rotate-half formula a simple elementwise
multiply and add (see below), at the cost of doubling table width from
`rotary_dim/2` to `rotary_dim`.

## The Rotate-Half Operation

### Algebraic Basis

Standard 2D rotation of a pair `(x, y)` by angle `φ` yields:

```math
\begin{pmatrix} x' \\ y' \end{pmatrix}
= \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}
  \begin{pmatrix} x \\ y \end{pmatrix}
= \begin{pmatrix} x\cos\phi - y\sin\phi \\ x\sin\phi + y\cos\phi \end{pmatrix}
```

RoPE applies this independently to each dimension pair `(x_i, x_{i + rotary_dim/2})` of
a query or key vector, using angle `φ = t · θ_i` for token at position `t`. (Under the rotate-half convention, dimension `i` is paired with dimension `i + rotary_dim/2`, not with `i+1`.)

### The rotate_half Convention

In practice, rather than looping over pairs, the rotate-half approach reshapes
the head dimension to apply all pairs in parallel. Given a vector
`x ∈ ℝ^{rotary_dim}`, define:

```math
\text{rotate\_half}(x) = [-x_{\frac{r}{2}},\; -x_{\frac{r}{2}+1},\; \ldots,\; -x_{r-1},\;\;
                           x_0,\; x_1,\; \ldots,\; x_{\frac{r}{2}-1}]
```

where `r = rotary_dim`. This operation moves the second half of the vector to
the front with negation, and the first half to the back unchanged. Then:

```math
\text{RoPE}(x,\, t) = x \odot \texttt{cos\_table}[t] + \text{rotate\_half}(x) \odot \texttt{sin\_table}[t]
```

where `⊙` is elementwise multiplication. This is algebraically equivalent to
applying the 2×2 rotation matrix to each pair `(x_i, x_{i + rotary_dim/2})`:

- Position `i` in the output: `x_i · cos(t·θ_i) − x_{i + rotary_dim/2} · sin(t·θ_i)`
- Position `i + rotary_dim/2` in the output: `x_i · sin(t·θ_i) + x_{i + rotary_dim/2} · cos(t·θ_i)`

Rotate-half pairs `x_i` with `x_{i + rotary_dim/2}`,
**not** `x_{2i}` with `x_{2i+1}`. This is the HuggingFace convention (used by
Qwen, LLaMA, and Mistral). Concretely, for a vector of length 64:

- Pair `i=0`: dimensions 0 and 32 rotate together at frequency `θ_0`
- Pair `i=1`: dimensions 1 and 33 rotate together at frequency `θ_1`
- Pair `i=31`: dimensions 31 and 63 rotate together at frequency `θ_{31}`

> **[SILENT FAILURE]** The alternative "adjacent-pair" convention pairs
> `(x_0, x_1)`, `(x_2, x_3)`, etc. at frequencies `θ_0, θ_1, …`. If a
> TTNN implementation uses adjacent-pair convention while the HuggingFace
> reference uses rotate-half convention (or vice versa), the output Q/K
> tensors will be numerically different but the error will not raise any
> exception — it will silently produce wrong attention scores. Always
> confirm which convention the reference implementation uses before
> writing the TTNN kernel.

### Python Reference Implementation

```python
def rotate_half(x):
    """Split x along last dim into two halves; negate-swap them."""
    x1 = x[..., : x.shape[-1] // 2]   # first half:  dims [0, rotary_dim/2)
    x2 = x[..., x.shape[-1] // 2 :]   # second half: dims [rotary_dim/2, rotary_dim)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    q, k:    [batch, num_heads, seq_len, head_dim]  (or rotary_dim for partial RoPE)
    cos, sin: [1,    1,         seq_len, rotary_dim] after unsqueeze
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # broadcast over heads
    sin = sin.unsqueeze(unsqueeze_dim)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

## Partial RoPE

### Definition

When `partial_rotary_factor < 1.0`, only the first `rotary_dim` dimensions of
each head are rotated. The remaining `head_dim - rotary_dim` dimensions are
concatenated unchanged:

```math
\texttt{rotary\_dim} = \left\lfloor \texttt{head\_dim} \times \texttt{partial\_rotary\_factor} \right\rfloor
```

The apply step splits the head vector, rotates the prefix, and reassembles:

```math
\text{PartialRoPE}(x, t) =
  \bigl[\,\text{RoPE}(x_{0:\texttt{rotary\_dim}},\, t) \;\|\; x_{\texttt{rotary\_dim}:\texttt{head\_dim}}\,\bigr]
```

where `‖` denotes concatenation. The cos/sin table has width `rotary_dim`, not
`head_dim`; only the prefix is touched.

### Why Partial RoPE

Partial RoPE is used in long-context models where large `head_dim` (e.g., 128)
is desirable for expressiveness, but rotary encoding all dimensions would
introduce frequency components too high or too low for the target context
length. Reserving a suffix of dimensions as "position-free" semantic channels
gives the model headroom to store content information without positional phase
contamination, particularly relevant at very large `rope_theta` values.

### Concrete Example: Qwen3.6 Text Layers

```
head_dim              = 128
partial_rotary_factor = 0.5
rotary_dim            = floor(128 * 0.5) = 64
non-rotary dimensions = 128 - 64 = 64
```

For a query vector `q ∈ ℝ^{128}` at position `t`:

```
q[0:64]   → rotated by cos_table[t] and sin_table[t] (shape [64])
q[64:128] → concatenated unchanged
```

The cos/sin table has shape `[max_seq_len, 64]`, not `[max_seq_len, 128]`. The
output vector has the same shape `[128]` as the input: `[rotated_64 || unchanged_64]`.

```python
def apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_dim):
    """
    Apply RoPE to first rotary_dim dimensions; pass the rest through unchanged.
    q, k:    [batch, num_heads, seq_len, head_dim]
    cos, sin: [1,    1,         seq_len, rotary_dim]
    """
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot, k_rot  = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    q_out = torch.cat([q_rot, q_pass], dim=-1)  # restore full head_dim
    k_out = torch.cat([k_rot, k_pass], dim=-1)
    return q_out, k_out
```

> **[SILENT FAILURE]** If the cos/sin table is constructed with
> `rotary_dim=head_dim` instead of the correct `rotary_dim=64`, the table
> will have the wrong frequency spectrum — the high-index pairs will have
> much higher frequencies than intended. The rotation will still execute
> without error, but the model will produce wrong Q/K values. Always
> derive `rotary_dim` from `floor(head_dim * partial_rotary_factor)`, not
> from `head_dim` directly.

## Summary: Key Relationships

| Quantity | Formula | Qwen3.6 Value |
|---|---|---|
| `rotary_dim` | `floor(head_dim * partial_rotary_factor)` | 64 |
| Number of rotation pairs | `rotary_dim / 2` | 32 |
| `inv_freq` vector length | `rotary_dim / 2` | 32 |
| `cos_table` / `sin_table` shape | `[max_seq_len, rotary_dim]` | `[T, 64]` |
| Non-rotated suffix | `head_dim - rotary_dim` dimensions | 64 dimensions |
| rotate-half pairing | dimension `i` with dimension `i + rotary_dim/2` | `i` with `i+32` |

---

**Next:** [`mrope_motivation_and_design.md`](./mrope_motivation_and_design.md)
