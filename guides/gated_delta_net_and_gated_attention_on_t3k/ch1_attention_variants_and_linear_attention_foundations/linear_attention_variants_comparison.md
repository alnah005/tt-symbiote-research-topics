# Linear Attention Variants: Comparison

## General Gated State Update

Once a forgetting gate is introduced, the general state update for linear recurrent attention takes the form:

$$S_t = G_t \odot S_{t-1} + k_t v_t^\top$$

where **$G_t \in \mathbb{R}^{d_k \times d_v}$** is the forgetting gate matrix and $\odot$ denotes elementwise multiplication. $G_t$ controls how much of each entry of the previous state $S_{t-1}$ is retained before the new write is added. The variants below are distinguished by how $G_t$ is constructed — whether it is data-dependent, and whether it has full matrix structure or is rank-constrained.

## RetNet

RetNet (Sun et al. 2023) uses a fixed scalar decay applied uniformly to all state entries:

$$G_t = \gamma \cdot \mathbf{1} \qquad (\mathbf{1} \in \mathbb{R}^{d_k \times d_v} \text{ is the all-ones matrix; equivalently, scalar } \gamma \text{ broadcast to all entries})$$

$$S_t = \gamma \cdot S_{t-1} + k_t v_t^\top$$

- $\gamma \in (0, 1)$ is a fixed hyperparameter (e.g., $1 - 1/T_\text{target}$), not input-dependent.
- All entries of S decay at the same rate, regardless of what has been written to S or what the current input is.
- No data-dependence: $G_t$ is constant across all positions and all sequences; it is determined at architecture design time.
- Effect: exponential recency bias — older writes receive exponentially lower weight, with the rate controlled by $\gamma$. This recovers a form of relative position weighting without explicit positional encodings.
- Limitation: the decay rate is fixed globally. The model cannot decide to hold a piece of information for longer because the current input is important, nor can it selectively forget only part of the state.

## GLA (Gated Linear Attention)

GLA (Yang et al., ICML 2024) introduces input-dependent, row-wise gating:

$$G_t = \alpha_t \mathbf{1}^\top \qquad \text{(outer product)}$$

$$S_t = (\alpha_t \mathbf{1}^\top) \odot S_{t-1} + k_t v_t^\top$$

- $\alpha_t \in \mathbb{R}^{d_k}$ is computed from the current input (e.g., via a small MLP or linear projection followed by sigmoid).
- The outer product $\alpha_t \mathbf{1}^\top \in \mathbb{R}^{d_k \times d_v}$ produces a gate matrix where each row i has the same value $\alpha_t[i]$ applied uniformly across all $d_v$ columns.
- This provides **row-wise data-dependent decay**: each key dimension can independently control how much of its row in S is retained, but all value dimensions within a key row decay at the same rate.
- The rank-1 constraint on $G_t$ (it is the outer product of two vectors) is what makes GLA amenable to efficient parallel hardware scans: the structured form enables factored recurrence similar to associative scan.
- Limitation: row-wise gating cannot selectively retain specific (key-dim, value-dim) entries of S; the gate is not fully data-dependent across both axes.

## Mamba2

Mamba2 (Dao and Gu, 2024) uses a scalar per-step gate broadcast uniformly:

$$G_t = \gamma_t \cdot \mathbf{1}\mathbf{1}^\top \qquad \text{(scalar } \gamma_t \text{ broadcast to all entries)}$$

$$S_t = \gamma_t \cdot S_{t-1} + k_t v_t^\top$$

- $\gamma_t \in (0, 1]$ is computed from the input at each step as $\gamma_t = \exp(-\text{softplus}(a_t))$, where $a_t$ is a learned linear projection of the input. Because softplus outputs values $\geq 0$, the negated exponent maps to $(0, 1]$, ensuring $\gamma_t$ is always a valid decay (never an amplifier). This matches the Mamba2 paper's parameterization of A in log-space as a negative quantity, subsequently exponentiated.
- Unlike RetNet, the decay rate can vary per step — the model can choose to retain state strongly ($\gamma_t \approx 1$) or flush it rapidly ($\gamma_t \approx 0$) depending on the current token.
- All entries of S decay at the same rate at any given step (no per-entry control).
- Mamba2's specific contribution is showing that this scalar structured SSM form enables fast parallel prefix scans via the SSD (Structured State Space Duality) framework, achieving near-linear time prefill while maintaining O(1) decode.
- Limitation: a single scalar $\gamma_t$ cannot selectively retain some key-value associations while forgetting others within the same step; the entire state is scaled uniformly.

## DeltaNet (Standard)

DeltaNet (Schlag et al. 2021; revisited in Yang et al. 2024) takes a fundamentally different approach: rather than adding a forgetting gate, it reformulates the write as an **error-correcting update**:

$$S_t = S_{t-1} - \beta_t \tilde{k}_t \left(\tilde{k}_t^\top S_{t-1} - v_t^\top\right)$$

where $\tilde{k}_t = k_t / \|k_t\|$ is the L2-normalized key and $\beta_t \in (0, 1]$ is a per-step scalar learning rate (analogous to a learning rate in Hebbian updates), computed from the input.

This is the **delta rule**: S is updated to reduce the squared error between its current prediction $S_{t-1}^\top \tilde{k}_t$ and the target value $v_t$. Expanding:

$$S_t = (I - \beta_t \tilde{k}_t \tilde{k}_t^\top) S_{t-1} + \beta_t \tilde{k}_t v_t^\top$$

- The term $(I - \beta_t \tilde{k}_t \tilde{k}_t^\top) S_{t-1}$ left-multiplies $S_{t-1}$ by a rank-1 projection, zeroing the component of each column of $S_{t-1}$ that lies along $\tilde{k}_t$; this selectively erases the old association at key $\tilde{k}_t$.
- The term $\beta_t \tilde{k}_t v_t^\top$ writes the new target value.
- This is a **targeted, key-localized write**: only the portion of S that responds to $\tilde{k}_t$ is modified. Other keys whose directions are orthogonal to $\tilde{k}_t$ are unaffected.
- There is no coarse forgetting: the overall magnitude of S is not globally reduced. DeltaNet can overwrite stale specific memories but cannot globally forget irrelevant context.
- $\beta_t = 1$ produces a hard overwrite (the old association at $\tilde{k}_t$ is fully replaced); $\beta_t < 1$ blends old and new.

The retrieval quality advantage of DeltaNet over vanilla linear attention is that repeated writes to the same key direction converge rather than accumulate — the state does not grow unboundedly dense in any one direction.

## Summary Table

| Variant | Gate G_t form | Data-dependent? | Coarse forgetting? | Write mechanism |
|---------|--------------|-----------------|-------------------|-----------------|
| Vanilla linear attn | I (identity; no gate) | No | No | Additive outer product |
| RetNet | γ · 𝟙 (fixed scalar, all entries) | No | Yes (uniform) | Additive outer product |
| GLA | α_t 1^T (rank-1, row-wise) | Yes (per key-dim) | Yes (row-wise) | Additive outer product |
| Mamba2 | γ_t · 1 1^T (scalar broadcast) | Yes (scalar per step) | Yes (uniform) | Additive outer product |
| DeltaNet | Implicit via delta rule | Yes (via β_t, k̃_t) | No (key-localized only) | Error-correcting (delta rule) |

## Forward Reference to Chapter 2

The analysis above reveals a gap in the design space: no variant combines both a full forgetting gate (to handle coarse context expiration) and the delta rule write (to handle targeted overwrite of stale associations). Applying a forgetting gate alone loses specific old memories but cannot correct them; applying the delta rule alone corrects specific memories but cannot flush irrelevant context globally.

**Gated Delta Net** closes this gap by composing both mechanisms in a single recurrence:

$$S_t = g_t \cdot S_{t-1} + \beta_t \tilde{k}_t v_t^\top - \beta_t \tilde{k}_t \left(\tilde{k}_t^\top S_{t-1}\right)$$

where $g_t \in (0, 1]$ is a **scalar** decay gate (a single number, not a matrix) and $\beta_t$ is the delta rule learning rate. Chapter 2 derives this recurrence in full, analyzes the mathematical relationship between the two mechanisms, and shows how both can be computed efficiently in the chunked-recurrence formulation that maps to T3K hardware.

---

**Next:** [Chapter 2 — Gated Delta Net: Mathematical Formulation and Recurrence Structure](../ch2_gated_delta_net_math_and_recurrence/index.md)
