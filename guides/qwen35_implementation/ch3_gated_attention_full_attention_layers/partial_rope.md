# Partial RoPE in Qwen3.5

## What Partial RoPE Means

Standard RoPE rotates every dimension of every attention head. Qwen3.5 restricts rotation to only the first `rotary_dim` dimensions of each head. The remaining dimensions are left unchanged (multiplied by $\cos = 1$, $\sin = 0$).

The relevant hyperparameters for both the 27B and 35B-A3B variants are:

```
head_dim              = 256
partial_rotary_factor = 0.25
rotary_dim            = int(head_dim * partial_rotary_factor) = 64
rope_theta            = 1_000_000
```

For a query or key vector of shape `[..., 256]`, only dimensions `[0, 63]` participate in rotation. Dimensions `[64, 255]` pass through unchanged.

## The Rotation Formula

Rotation pairs are formed by splitting the first `rotary_dim=64` dimensions into two halves of `half_dim = rotary_dim // 2 = 32`:

- First half: dimensions `[0, 31]`
- Second half: dimensions `[32, 63]`

For token at position $p$, frequency index $i \in [0, 31]$, the inverse frequency is:

$$\theta_i^{-1} = \frac{1}{\text{rope theta}^{2i / \text{rotary dim}}}$$

Note the denominator is `rotary_dim=64`, **not** `head_dim=256`. This is the central correctness requirement.

The frequency for position $p$ and index $i$ is $\phi_i = p \cdot \theta_i^{-1}$. The rotation applied to a pair $(x_1, x_2) = (x[..., i], x[..., i+32])$ is:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos\phi_i & -\sin\phi_i \\ \sin\phi_i & \cos\phi_i \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

In full:

$$x_1' = x_1 \cos\phi_i - x_2 \sin\phi_i$$
$$x_2' = x_2 \cos\phi_i + x_1 \sin\phi_i$$

Dimensions $[\text{rotary dim}, \text{head dim}) = [64, 255]$ are untouched: $x_j' = x_j$.

From `test_attention_pcc.py`, the reference implementation:

```python
half_dim = rotary_dim // 2   # 32
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
freqs = pos * inv_freq        # shape [32]
cos_val = freqs.cos()
sin_val = freqs.sin()

def apply_partial_rope(x):
    x1 = x[..., :half_dim]           # dims [0, 31]
    x2 = x[..., half_dim:rotary_dim] # dims [32, 63]
    x_rot = torch.cat([x1 * cos_val - x2 * sin_val,
                       x2 * cos_val + x1 * sin_val], dim=-1)
    return torch.cat([x_rot, x[..., rotary_dim:]], dim=-1)  # pass-through [64, 255]
```

## The Three Failure Modes of Standard `rotary_embedding_llama`

The standard `rotary_embedding_llama` op (used for Llama-family models) cannot be applied unmodified to Qwen3.5. The `gated_attention.py` module docstring identifies exactly three reasons:

### Failure 1 — Wrong Frequency Denominator

`rotary_embedding_llama` computes inverse frequencies using:

$$\theta_i^{-1} = \frac{1}{\text{rope theta}^{2i / \text{head dim}}}$$

For Qwen3.5, the denominator must be `rotary_dim=64`, not `head_dim=256`. Since $64 \ll 256$, the exponent $2i/64$ grows 4x faster than $2i/256$. Higher-frequency basis vectors — those responsible for encoding fine positional distinctions — would be wrongly assigned slow, low-frequency oscillations. The result is that the model sees a compressed, incorrect frequency spectrum and produces positional embeddings that do not match the training distribution.

### Failure 2 — Interleaved vs Non-Interleaved Pairing

`rotary_embedding_llama` uses **interleaved** (Meta-style) pairing: dimension $2i$ is paired with dimension $2i+1$ (adjacent odd/even indices) for $i \in [0, \text{head dim}/2)$. Qwen3.5 follows Hugging Face non-interleaved convention: the device kernel `ttnn.experimental.rotary_embedding` pairs dimension $j$ with dimension $j + \text{head dim}/2 = j + 128$ (split-half pairing across the full head dimension).

These two conventions conflict even when only the rotary dimensions are considered. Under Meta-style interleaved pairing, dim 0 rotates with dim 1, dim 2 with dim 3, and so on. Under HF-style non-interleaved pairing used by `ttnn.experimental.rotary_embedding`, dim 0 pairs with dim 128, dim 1 with dim 129, etc. — the split is at `head_dim/2 = 128`, not at `rotary_dim/2 = 32`. Applying the interleaved convention to HF-format Q/K vectors produces rotations between the wrong dimension pairs, corrupting the output regardless of whether the frequencies are correct.

The `transformation_mat` stored in `RotarySetup` encodes this interleaved pairing pattern. For Llama-style models it is constructed assuming Meta-style layout; Qwen3.5 uses `HfRotarySetup` so that `ttnn.experimental.rotary_embedding` handles the non-interleaved pairing at distance `head_dim/2 = 128` directly. `HfRotarySetup` does not use `transformation_mat` — it is set to `None`.

### Failure 3 — cos/sin Format Mismatch

The `gated_attention.py` docstring identifies the third failure as: "cos/sin are in Meta interleaved format but Q/K are in HF format." The standard `rotary_embedding_llama` kernel expects cos/sin lookup tables laid out in Meta interleaved format — alternating values that align with the adjacent odd/even dimension pairing of interleaved Q/K storage. Qwen3.5 Q/K vectors are stored in HF non-interleaved format. Even if the frequency values were correct (Failure 1 resolved) and the pairing distance were somehow fixed (Failure 2 resolved), feeding a Meta-format cos/sin table to HF-format Q/K tensors would still produce incorrect rotations because the index mapping assumed by the lookup table does not match the actual tensor layout. `HfRotarySetup` resolves this by using `ttnn.experimental.rotary_embedding`, which operates directly on HF-format tensors — no format conversion is needed.

## The Corrected Cos/Sin Matrix Patch (Production Solution)

The production solution avoids any host roundtrip by patching the cos/sin matrices stored inside the rope setup object. This is done once at model build time in `build_model()` inside `demo_a3b.py`.

**Step 1: Construct `HfRotarySetup` with full `head_dim` frequencies.**

```python
model.rope_setup = HfRotarySetup(
    device=device,
    batch_size=args.max_batch_size,
    head_dim=args.head_dim,          # 256
    max_seq_len=args.max_seq_len,
    rope_theta=args.rope_theta,      # 1_000_000
    rope_scaling=args.rope_scaling,
)
```

At this point, `cos_matrix` and `sin_matrix` contain frequencies computed with denominator `head_dim=256` — incorrect for the partial-rotation positions.

**Step 2: Compute corrected frequencies.**

```python
rotary_dim = int(args.head_dim * partial)   # 64
half_rotary = rotary_dim // 2               # 32
half_head   = args.head_dim // 2            # 128

inv_freq_correct = 1.0 / (
    args.rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
)  # shape [32], denominator = rotary_dim = 64

positions = torch.arange(args.max_seq_len).float()
freqs = torch.outer(positions, inv_freq_correct)  # [max_seq_len, 32]
```

**Step 3: Pull matrices to host, overwrite the rotary and pass-through slices, push back.**

```python
cos_h = ttnn.to_torch(model.rope_setup.cos_matrix)
sin_h = ttnn.to_torch(model.rope_setup.sin_matrix)

# Overwrite rotary slice with corrected frequencies
cos_h[:, :, :, :half_rotary]                         = freqs.cos().unsqueeze(0).unsqueeze(0)
cos_h[:, :, :, half_head : half_head + half_rotary]  = freqs.cos().unsqueeze(0).unsqueeze(0)

sin_h[:, :, :, :half_rotary]                         = freqs.sin().unsqueeze(0).unsqueeze(0)
sin_h[:, :, :, half_head : half_head + half_rotary]  = freqs.sin().unsqueeze(0).unsqueeze(0)

# Set pass-through dims to cos=1, sin=0 (identity rotation)
cos_h[:, :, :, half_rotary:half_head]            = 1.0
cos_h[:, :, :, half_head + half_rotary :]        = 1.0
sin_h[:, :, :, half_rotary:half_head]            = 0.0
sin_h[:, :, :, half_head + half_rotary :]        = 0.0

ttnn.deallocate(model.rope_setup.cos_matrix)
ttnn.deallocate(model.rope_setup.sin_matrix)
model.rope_setup.cos_matrix = ttnn.from_torch(
    cos_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
model.rope_setup.sin_matrix = ttnn.from_torch(
    sin_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
)
```

The HF-style `rotary_embedding` op then applies these corrected matrices directly on device. No host roundtrip occurs during inference.

### Why the Patch Addresses All Three Failure Modes

- **Failure 1 (wrong denominator):** Corrected — `inv_freq_correct` uses `/ rotary_dim` (64), overwriting the wrong values in positions `[0:half_rotary]` and `[half_head : half_head+half_rotary]`.
- **Failure 2 (wrong pairing distance):** Resolved implicitly — `HfRotarySetup` does not use `transformation_mat` and delegates pairing entirely to `ttnn.experimental.rotary_embedding`, which applies the HF non-interleaved convention. The patched cos/sin values are written into `[:half_rotary]` and `[half_head:half_head+half_rotary]` to supply the correct values for the HF pairing convention (dim $j$ with dim $j+128$).
- **Failure 3 (format mismatch):** Resolved by using `HfRotarySetup` (HF format) instead of `RotarySetup` (Meta interleaved format).

## 27B vs 35B-A3B: `RotarySetup` vs `HfRotarySetup`

The 27B dense demo (`demo.py`) and the reference PCC test (`test_attention_pcc.py`) use `RotarySetup` (Meta-style). The 35B-A3B demo (`demo_a3b.py`) uses `HfRotarySetup` (HF-style).

The difference matters for the patch because the cos/sin matrix layout differs between the two classes:

| Property | `RotarySetup` | `HfRotarySetup` |
|----------|---------------|-----------------|
| RoPE op | `rotary_embedding_llama` | `ttnn.experimental.rotary_embedding` |
| Dimension pairing | interleaved (dim $2i$ with $2i+1$) | non-interleaved (dim $j$ with $j + \text{head dim}/2 = j + 128$) |
| Transformation matrix | stored in `transformation_mat` | `None` (not needed) |
| `get_rot_mats()` return | cos/sin sliced by position, sharded | full cos/sin cache (unsliced) |

When patching `RotarySetup` (the Meta-style setup used for the 27B reference test), the slice offsets must use `half_head = head_dim // 2 = 128` to address the interleaved layout positions. For `HfRotarySetup` (A3B), the same `half_head` offset applies but the pairing convention is different, which is why the patch was written for and validated against `HfRotarySetup`.

## Historical Host-Based `custom_rope_fn` (Superseded)

An earlier approach implemented in `_setup_partial_rope()` in `gated_attention.py` ran the rotation on the host CPU. This bypassed `rotary_embedding_llama` entirely via the `custom_rope_fn` hook on the base `Attention` class. The roundtrip transfers Q and K for one token step: at `head_dim=256` with 24 Q heads and 4 K heads, that is $(24 + 4) \times 256 \times 2$ bytes = ~14 KB. Latency is negligible.

The method was superseded by the corrected-matrix approach because the device-side patch eliminates all 5 host-device syncs per attention layer, keeping the inference graph fully resident on device and compatible with Metal Trace.

---

**Next:** [`output_gate.md`](./output_gate.md)
