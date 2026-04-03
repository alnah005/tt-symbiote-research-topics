# Multidimensional RoPE Theory

This file derives 2D factored Rotary Position Embeddings from first principles, starting with a recap of standard 1D RoPE and building to the multidimensional extension used in Gemma 4's vision encoder.

## 1D RoPE Recap

### The Core Idea

Rotary Position Embedding (RoPE) encodes position information by **rotating** query and key vectors in the complex plane. The rotation angle for each dimension pair depends on the token's position in the sequence, producing position-dependent phase shifts that cause the dot product $q \cdot k$ to be a function of **relative** position.

### Frequency Computation

Given a base frequency parameter $\theta$ and a head dimension $d$, RoPE defines a set of inverse frequencies:

$$
\omega_i = \frac{1}{\theta^{2i/d}}, \quad i = 0, 1, \ldots, \frac{d}{2} - 1
$$

For a token at position $m$, the rotation angle for dimension pair $i$ is:

$$
\phi_i(m) = m \cdot \omega_i = \frac{m}{\theta^{2i/d}}
$$

This produces a spectrum of wavelengths:

$$
\lambda_i = \frac{2\pi}{\omega_i} = 2\pi \cdot \theta^{2i/d}
$$

- The first dimension pair ($i = 0$) has the shortest wavelength: $\lambda_0 = 2\pi$
- The last dimension pair ($i = d/2 - 1$) has the longest: $\lambda_{d/2-1} = 2\pi \cdot \theta^{(d-2)/d}$

The parameter $\theta$ controls the range of this spectrum. A larger $\theta$ stretches the wavelengths, allowing the model to distinguish positions over a longer range.

### Rotation of Q/K Pairs

For a query vector $\mathbf{q} \in \mathbb{R}^d$ at position $m$, RoPE applies pairwise rotations:

$$
\text{RoPE}(\mathbf{q}, m) = \begin{pmatrix} q_0 \cos\phi_0(m) - q_1 \sin\phi_0(m) \\ q_0 \sin\phi_0(m) + q_1 \cos\phi_0(m) \\ q_2 \cos\phi_1(m) - q_3 \sin\phi_1(m) \\ q_2 \sin\phi_1(m) + q_3 \cos\phi_1(m) \\ \vdots \end{pmatrix}
$$

In the standard implementation, this is computed efficiently using the `rotate_half` formulation:

$$
\text{RoPE}(\mathbf{q}, m) = \mathbf{q} \odot \cos(\boldsymbol{\phi}(m)) + \text{rotate\_half}(\mathbf{q}) \odot \sin(\boldsymbol{\phi}(m))
$$

where $\odot$ denotes element-wise multiplication and `rotate_half` splits the vector in two, negates the second half, and swaps the halves:

$$
\text{rotate\_half}([q_0, q_1, \ldots, q_{d/2-1}, q_{d/2}, \ldots, q_{d-1}]) = [-q_{d/2}, \ldots, -q_{d-1}, q_0, \ldots, q_{d/2-1}]
$$

### The Relative Position Property

The key property of RoPE is that the dot product of two rotated vectors depends only on their **relative** position:

$$
\text{RoPE}(\mathbf{q}, m)^T \cdot \text{RoPE}(\mathbf{k}, n) = \mathbf{q}^T R^T(m) R(n) \mathbf{k} = \mathbf{q}^T R(n - m) \mathbf{k}
$$

where $R(p)$ is the block-diagonal rotation matrix at position $p$. This means attention scores naturally encode relative distance between tokens.

## Extension to 2D: Factored Spatial RoPE

### The Problem

In a language model, tokens have a single position index $m \in \{0, 1, \ldots, L-1\}$. In a vision encoder, patches live on a 2D grid with coordinates $(x, y)$ where $x \in \{0, \ldots, W_p - 1\}$ and $y \in \{0, \ldots, H_p - 1\}$, with $W_p$ and $H_p$ being the patch grid width and height.

Flattening this grid to 1D (raster scan order) and applying standard RoPE would lose the 2D spatial structure. Patch $(0, 1)$ (top-left, second row) would be at position $W_p$ in the flattened sequence, while patch $(1, 0)$ (second column, first row) would be at position 1. The RoPE frequencies would not reflect that these two patches are equidistant from $(0, 0)$.

### The Solution: Dimension Factoring

2D factored RoPE solves this by splitting the head dimension into two halves and assigning each half to one spatial axis:

$$
d_{\text{spatial}} = \frac{d}{2}
$$

For a patch at grid position $(x, y)$:

- **First half** (dimensions $0$ to $d_{\text{spatial}} - 1$): rotated using position $x$ with frequencies $\omega_i^{(x)}$
- **Second half** (dimensions $d_{\text{spatial}}$ to $d - 1$): rotated using position $y$ with frequencies $\omega_i^{(y)}$

Both halves use the **same** frequency schedule, computed over $d_{\text{spatial}}$ dimensions rather than $d$:

$$
\omega_i = \frac{1}{\theta^{2i/d_{\text{spatial}}}}, \quad i = 0, 1, \ldots, \frac{d_{\text{spatial}}}{2} - 1
$$

> **Warning:** Note the denominator is $d_{\text{spatial}}$ (not $d$). This is a critical detail. Using $d$ instead of $d_{\text{spatial}}$ would halve the effective frequency range, compressing the rotation spectrum and reducing the model's ability to distinguish nearby positions.

### The Full 2D RoPE Formula

For a query vector $\mathbf{q} \in \mathbb{R}^d$ at grid position $(x, y)$:

$$
\text{RoPE}_{2D}(\mathbf{q}, x, y) = \text{concat}\Big(\text{RoPE}_{1D}(\mathbf{q}_{[0:d/2]}, x),\;\; \text{RoPE}_{1D}(\mathbf{q}_{[d/2:d]}, y)\Big)
$$

where $\text{RoPE}_{1D}$ is the standard rotation applied to a $d/2$-dimensional sub-vector.

Expanding the cos/sin formulation:

$$
\text{RoPE}_{2D}(\mathbf{q}, x, y) = \begin{pmatrix} \mathbf{q}_{[0:d/2]} \odot \cos(\boldsymbol{\phi}(x)) + \text{rotate\_half}(\mathbf{q}_{[0:d/2]}) \odot \sin(\boldsymbol{\phi}(x)) \\ \mathbf{q}_{[d/2:d]} \odot \cos(\boldsymbol{\phi}(y)) + \text{rotate\_half}(\mathbf{q}_{[d/2:d]}) \odot \sin(\boldsymbol{\phi}(y)) \end{pmatrix}
$$

where:

$$
\boldsymbol{\phi}(p) = \Big[\frac{p}{\theta^{0/d_s}},\; \frac{p}{\theta^{2/d_s}},\; \ldots,\; \frac{p}{\theta^{(d_s-2)/d_s}},\; \frac{p}{\theta^{0/d_s}},\; \frac{p}{\theta^{2/d_s}},\; \ldots,\; \frac{p}{\theta^{(d_s-2)/d_s}}\Big] \in \mathbb{R}^{d_s}
$$

with $d_s = d_{\text{spatial}} = d/2$, and the frequency vector is formed by concatenating the base frequencies with themselves via `torch.cat((freqs, freqs), dim=-1)`, yielding the layout $[f_0, f_1, \ldots, f_{d_s/2-1}, f_0, f_1, \ldots, f_{d_s/2-1}]$ that pairs with the `rotate_half` scheme.

### The 2D Relative Position Property

The factored approach preserves the relative position property independently along each axis:

$$
\text{RoPE}_{2D}(\mathbf{q}, x_q, y_q)^T \cdot \text{RoPE}_{2D}(\mathbf{k}, x_k, y_k) = \sum_{i=0}^{d/4-1} f_i(x_q - x_k) + \sum_{i=0}^{d/4-1} g_i(y_q - y_k)
$$

The attention score decomposes into a sum of terms that depend on the relative x-distance and relative y-distance independently. This means the model can learn spatial attention patterns like "attend strongly to patches in the same row" or "attend to patches directly above" without coupling the two axes.

## Concrete Numbers for Gemma 4 Vision

For the Gemma 4 31B vision encoder:

| Parameter | Value |
|-----------|-------|
| `head_dim` ($d$) | 72 |
| `rope_theta` ($\theta$) | 100.0 |
| Spatial dim ($d_{\text{spatial}} = d/2$) | 36 |
| Number of frequency pairs per axis ($d_{\text{spatial}}/2$) | 18 |
| `inv_freq` length | 18 |

### Inverse Frequency Table

The 18 inverse frequencies are:

$$
\omega_i = \frac{1}{100^{2i/36}}, \quad i = 0, 1, \ldots, 17
$$

| Index $i$ | $\omega_i$ | Wavelength $\lambda_i$ | Interpretation |
|-----------|-----------|----------------------|----------------|
| 0 | 1.000 | $2\pi \approx 6.3$ positions | Distinguishes adjacent patches |
| 4 | 0.359 | $\approx 17.5$ positions | Medium-range spatial structure |
| 8 | 0.129 | $\approx 48.7$ positions | Large-scale spatial structure |
| 12 | 0.046 | $\approx 135.4$ positions | Exceeds typical grid extent |
| 17 | 0.013 | $\approx 486.5$ positions | Far exceeds any grid size |

### Typical Position Ranges

| Token Budget | Approx. Patches | Example Grid | Max x or y Coordinate |
|-------------|----------------|--------------|----------------------|
| 70 | ~210 | 15 x 14 | ~15 |
| 280 | ~840 | 30 x 28 | ~30 |
| 1120 | ~3360 | 60 x 56 | ~60 |

The highest-frequency rotation ($i = 0$) completes a full cycle every $\approx 6.3$ positions. For a grid of 30 patches wide (280-token budget), this means $\sim 4.7$ full rotations across the horizontal extent --- more than enough to give each column a distinct phase signature.

The lowest-frequency rotation ($i = 17$) has wavelength $\approx 486.5$ positions. For any realistic grid size (max $\sim 100$ patches per axis), this rotation barely turns, providing a near-constant baseline. This is by design: the low-frequency components serve as a smooth positional bias, while the high-frequency components provide fine-grained discrimination.

## Why rope_theta = 100.0?

The choice of $\theta = 100.0$ (versus 10,000 or higher for text models) is driven by the position range:

| Domain | $\theta$ | Max Position | $\lambda_{\text{min}}$ | $\lambda_{\text{max}}$ |
|--------|----------|-------------|----------------------|----------------------|
| Text (Llama 3) | 500,000 | 128K tokens | $2\pi$ | $\sim 3.14 \times 10^6$ |
| Text (Gemma 2) | 10,000 | 8K tokens | $2\pi$ | $\sim 6.1 \times 10^4$ |
| Vision (Gemma 4) | 100 | $\sim 100$ patches/axis | $2\pi$ | $\sim 487$ |

The design principle is that $\lambda_{\text{max}}$ should comfortably exceed the maximum position range. For the vision encoder, positions never exceed $\sim 100$, so $\lambda_{\text{max}} \approx 487$ provides ample headroom.

If we mistakenly used $\theta = 10000$ for the vision encoder:

$$
\lambda_{\text{max}} = 2\pi \cdot 10000^{34/36} \approx 37{,}667
$$

This would make all wavelengths far longer than the grid, causing the rotation angles to be vanishingly small. The cos values would all be near 1.0 and the sin values near 0.0, effectively disabling the positional encoding.

> **Warning:** Always use the vision-specific `rope_theta=100.0` from `Gemma4VisionConfig.rope_parameters`. Do not accidentally inherit the text model's theta. The HuggingFace code reads theta from `config.rope_parameters["rope_theta"]`, which is set correctly for the vision config.

## Concatenation of cos/sin from Both Dimensions

The final cos and sin tensors passed to each attention layer have the full head dimension $d = 72$:

$$
\cos_{\text{2D}} = \text{concat}(\cos_x, \cos_y) \in \mathbb{R}^{B \times L \times 72}
$$
$$
\sin_{\text{2D}} = \text{concat}(\sin_x, \sin_y) \in \mathbb{R}^{B \times L \times 72}
$$

where $\cos_x, \sin_x \in \mathbb{R}^{B \times L \times 36}$ are computed from x-coordinates and $\cos_y, \sin_y \in \mathbb{R}^{B \times L \times 36}$ from y-coordinates. Both use the same 18-element `inv_freq` table but different position values.

The `apply_multidimensional_rope` function then splits Q and K along the last dimension into two chunks of 36, applies standard `rotate_half` RoPE to each chunk with the corresponding cos/sin slice, and concatenates the results. This split-apply-concat pattern is the "factored" aspect of the approach.

## Summary

| Aspect | 1D RoPE (text) | 2D Factored RoPE (vision) |
|--------|----------------|---------------------------|
| Position type | Scalar $m$ | 2D coordinate $(x, y)$ |
| Head dimension split | None (all $d$ dims encode $m$) | Two halves: first $d/2$ for $x$, second $d/2$ for $y$ |
| Frequency table | $d/2$ entries over $d$ | $d/4$ entries over $d/2$ (same formula, different dim) |
| Frequency denominator | $\theta^{2i/d}$ | $\theta^{2i/(d/2)}$ |
| Position range | Thousands to millions | Tens to low hundreds |
| Typical $\theta$ | 10,000 -- 1,000,000 | 100.0 |
| Relative position property | $f(m_q - m_k)$ | $f(x_q - x_k) + g(y_q - y_k)$ |

---

**Next:** [`reference_implementation.md`](./reference_implementation.md) — Line-by-line walkthrough of the HuggingFace code that implements 2D factored RoPE.
