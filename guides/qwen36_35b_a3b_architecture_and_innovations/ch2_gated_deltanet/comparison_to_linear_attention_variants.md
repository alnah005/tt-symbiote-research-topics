# Comparison to Linear Attention Variants

This file places Gated DeltaNet in the broader landscape of recurrent linear attention mechanisms
by analyzing four prior variants — RetNet, GLA, Mamba2, and standard DeltaNet — against a common
state-update framework. It then shows how Gated DeltaNet synthesizes their key ideas, summarizes
the comparison in a table, and explains why this mechanism was chosen for Qwen3.6.

---

## 1. General Gated Linear Attention Form

All linear recurrent attention variants can be expressed as a special case of:

$$S_t = G_t \odot S_{t-1} + k_t v_t^\top$$

where:

- $S_t \in \mathbb{R}^{d_k \times d_v}$ is the fixed-size recurrent state.
- $G_t \in \mathbb{R}^{d_k \times d_v}$ is the **forgetting gate matrix**, applied elementwise.
- $k_t v_t^\top \in \mathbb{R}^{d_k \times d_v}$ is the rank-1 write (outer product of key and value).
- $\odot$ denotes elementwise multiplication.

The variants below differ in:
1. How $G_t$ is structured (scalar, rank-1, identity, learned function of input).
2. Whether the write $k_t v_t^\top$ is a raw outer product or a targeted error correction.

---

## 2. Variant-by-Variant Formulations

### 2.1 RetNet

$$G_t = \gamma \cdot \mathbf{1} \qquad \text{(scalar broadcast to all entries)}$$

$$S_t = \gamma \cdot S_{t-1} + k_t v_t^\top$$

- $\gamma \in (0, 1)$ is a **fixed hyperparameter** — not input-dependent.
- All entries of $S$ decay at the same rate at every step, regardless of the current input.
- Provides exponential recency bias (older writes receive geometrically lower weight) without
  any positional encodings.
- Limitation: the decay rate is globally fixed. The model cannot choose to hold important
  information for longer, nor can it selectively forget only part of the state. There is no
  data-dependent gating at all.

### 2.2 GLA (Gated Linear Attention)

$$G_t \in \mathbb{R}^{d_k \times d_v} \qquad \text{(full elementwise gate matrix)}$$

$$S_t = G_t \odot S_{t-1} + k_t v_t^\top$$

- $G_t$ is a full $d_k \times d_v$ elementwise gate matrix computed from the current input,
  giving each $(i, j)$ entry of the state its own independent data-dependent decay value.
- This is **full elementwise data-dependent decay**: every (key-dim, value-dim) pair in $S$
  can be retained or forgotten independently at each step.
- The factored low-rank structure used in practice (e.g. $G_t \approx \alpha_t \beta_t^\top$)
  enables efficient parallel prefix scans while approximating the full matrix gate.
- Limitation: the write is still a direct outer product $k_t v_t^\top$ — there is no error
  correction. GLA can selectively gate any entry of the state but cannot perform targeted
  delta-rule overwrites of specific associations.

### 2.3 Mamba2

$$G_t = \gamma_t \cdot \mathbf{1}\mathbf{1}^\top \qquad \text{(scalar per step, broadcast)}$$

$$S_t = \gamma_t \cdot S_{t-1} + k_t v_t^\top$$

- $\gamma_t \in (0, 1)$ is computed from the current input using the same
  $\exp(-\exp(A_{\log}) \cdot \text{softplus}(\cdot))$ parameterization adopted by Gated
  DeltaNet (see `delta_rule_formulation.md` §4 for the full derivation and sign-analysis).
- Unlike RetNet, the decay rate can vary per step — the model can choose to retain state
  ($\gamma_t \approx 1$) or flush it ($\gamma_t \approx 0$) based on the current token.
- All entries of $S$ decay at the same rate within any given step (no per-dimension control).
- The scalar structured form enables the SSD (Structured State Space Duality) parallel scan
  framework: near-linear prefill time with O(1) decode.
- Limitation: a single $\gamma_t$ cannot retain some key-value associations while selectively
  forgetting others within the same step. The write is still a raw outer product.

### 2.4 Standard DeltaNet

Standard DeltaNet takes a fundamentally different approach: rather than adding a forgetting
gate, it replaces the write with an **error-correcting update** derived from the delta rule:

$$S_t = S_{t-1} + \tilde{k}_t \Bigl(\beta_t \cdot \bigl(v_t - S_{t-1}^\top \tilde{k}_t\bigr)\Bigr)^\top$$

which is equivalent to:

$$S_t = \bigl(I - \beta_t \tilde{k}_t \tilde{k}_t^\top\bigr) S_{t-1} + \beta_t \tilde{k}_t v_t^\top$$

- The term $(I - \beta_t \tilde{k}_t \tilde{k}_t^\top) S_{t-1}$ selectively erases the
  component of $S_{t-1}$ that lies along the direction $\tilde{k}_t$.
- The term $\beta_t \tilde{k}_t v_t^\top$ writes the new target association.
- This is a **targeted, key-localized write**: only the portion of $S$ that responds to
  $\tilde{k}_t$ is modified. Orthogonal associations are untouched.
- There is no coarse decay. The standard DeltaNet cannot globally flush irrelevant context.
- $\beta_t = 1$ gives a hard overwrite; $\beta_t < 1$ blends old and new.

Note: by setting $G_t = I$ (no forgetting) and replacing the write with the delta-rule
correction, standard DeltaNet fits into the general gated form only with an implicit gate
through the rank-1 projection $(I - \beta_t \tilde{k}_t \tilde{k}_t^\top)$. It is not a direct
special case of the $G_t \odot S_{t-1}$ form.

---

## 3. Gated DeltaNet: Combining Both Mechanisms

Gated DeltaNet adds a scalar decay gate $g_t \in (0, 1)$ to the standard DeltaNet recurrence:

$$S_t = g_t \cdot S_{t-1} + \tilde{k}_t \Bigl(\beta_t \cdot \bigl(v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t\bigr)\Bigr)^\top$$

The key design choices:

1. **Scalar decay (not full matrix)**: $g_t$ is a single scalar per head per step — the same
   design as Mamba2, not GLA's full $G_t \in \mathbb{R}^{d_k \times d_v}$ elementwise gate
   matrix. This keeps the gate simple and ensures the "predicted value under decayed state"
   in the delta correction is consistent: the retrieval $(g_t S_{t-1})^\top \tilde{k}_t$
   uses exactly the same decayed state that the carry-forward term $g_t S_{t-1}$ uses.
   With a full matrix gate, different (key-dim, value-dim) entries decay at different rates,
   making the "predicted value under the decayed state" a direction-dependent mixture that
   is harder to correct precisely with a single rank-1 write.

2. **Delta correction over the decayed prediction**: the error term uses
   $g_t S_{t-1}^\top \tilde{k}_t$ (not $S_{t-1}^\top \tilde{k}_t$). This means the correction
   accounts for the decay: if the state has been scaled down by $g_t$, the predicted value is
   proportionally smaller and the correction toward $v_t$ is proportionally larger.

3. **Combines the strengths of both parents**:
   - From Mamba2: data-dependent scalar gating for coarse selective forgetting.
   - From DeltaNet: targeted error-correcting writes that converge rather than accumulate.
   - Neither alone is sufficient: Mamba2 can forget but cannot precisely overwrite;
     DeltaNet can precisely overwrite but cannot globally flush. GLA has richer per-entry
     decay but still uses a raw outer-product write with no error correction.

---

## 4. Summary Comparison Table

| Variant | Gate $G_t$ form | Data-dependent? | Coarse forgetting? | Write mechanism | State size |
|---------|----------------|-----------------|-------------------|-----------------|------------|
| Vanilla linear attn | $I$ (identity) | No | No | Additive outer product | $[d_k, d_v]$ |
| RetNet | $\gamma \cdot \mathbf{1}$ (fixed scalar) | No | Yes (uniform) | Additive outer product | $[d_k, d_v]$ |
| GLA | $G_t \in \mathbb{R}^{d_k \times d_v}$ (full elementwise matrix) | Yes (per entry) | Yes (full matrix) | Additive outer product | $[d_k, d_v]$ |
| Mamba2 | $\gamma_t \cdot \mathbf{1}\mathbf{1}^\top$ (scalar) | Yes (scalar/step) | Yes (uniform) | Additive outer product | $[d_k, d_v]$ |
| DeltaNet | Implicit (rank-1 projection) | Yes (via $\beta_t, \tilde{k}_t$) | No (key-localized) | Error-correcting delta rule | $[d_k, d_v]$ |
| **Gated DeltaNet** | $g_t \cdot \mathbf{1}$ (scalar, data-dep.) | Yes (scalar/step) | Yes (uniform) | Error-correcting delta rule | $[d_k, d_v]$ |

All variants maintain the same $[d_k, d_v]$ state shape; the differences are entirely in the
update rule, not the state size.

---

## 5. Why Gated DeltaNet Was Chosen for Qwen3.6

### O(1) Decode Cost

All variants in the table above support $O(1)$ per-step decode cost: each new token requires a
fixed number of operations on the fixed-size state, regardless of sequence length. This is the
fundamental motivation for using any of these over softmax attention for the majority of layers.

In Qwen3.6-35B-A3B, 30 of 40 layers are Gated DeltaNet. The 10 Gated Attention (softmax)
layers provide the expressiveness needed for tasks that require precise token-specific retrieval,
while the Gated DeltaNet layers handle the bulk of the forward pass at constant memory cost.

### Long-Range Retrieval with Selective Forgetting

Qwen3.6 is designed for long-context applications (the model supports up to 128K tokens).
For why neither pure forgetting (GLA/Mamba2) nor pure error-correction (DeltaNet) alone is
sufficient, see `delta_rule_formulation.md` §1 and §3 above.

Gated DeltaNet's combination of uniform decay (for global context expiration) and targeted
error-correcting writes (for precise, convergent association learning) achieves a better
balance: the model can maintain long-range retrieval for important associations while still
clearing stale context globally.

### Favorable Comparison to GLA

GLA uses a full $d_k \times d_v$ elementwise gate matrix $G_t$ per head per step, giving each
state entry its own independent decay value. Gated DeltaNet instead uses a single scalar $g_t$
per head per step. The real trade-off is therefore: **GLA has richer decay** (full matrix,
any entry can be controlled independently) while **GDN has simpler decay but adds targeted
content updates** via the delta rule.

As noted in §3 key design choice 1, the scalar gate is the more natural pairing for the
delta-rule correction — the direction-mixing argument explains why a full matrix gate makes
the predicted value harder to correct precisely with a single rank-1 write.

### Mamba2 Relationship

The decay gate computation in Gated DeltaNet matches Mamba2's parameterization almost exactly
(the formula is derived in full in `delta_rule_formulation.md` §4). Gated DeltaNet can
be thought of as Mamba2 with the write replaced by the delta rule — it inherits Mamba2's
principled scalar gating while upgrading the associative memory update from a raw outer product
to an error-correcting write.

### Hardware Suitability

The scalar gate and rank-1 write structure map directly to efficient hardware implementations:

- **Prefill**: the chunkwise WY-decomposition (see
  [`guides/gated_delta_net_and_gated_attention_on_t3k/`](../../../gated_delta_net_and_gated_attention_on_t3k/ch2_gated_delta_net_math_and_recurrence/index.md))
  can parallelize the recurrence over chunks of the sequence via associative scan.
- **Decode**: a single step is 3 matrix-vector operations and 2 outer products — all highly
  regular, fixed-shape operations that fit well in TTNN tile-layout kernels.
- **State size**: ≈ 2 MB per layer (fp32) — see `delta_rule_formulation.md` §7 for the full
  breakdown and comparison to KV cache. Small enough to reside in device DRAM and be loaded
  into L1 for the recurrence without paging.

---

**Next:** [Chapter 3 — Qwen3.5 vs Qwen3.6: Exact Differences](../ch3_qwen35_vs_qwen36_differences/index.md)
