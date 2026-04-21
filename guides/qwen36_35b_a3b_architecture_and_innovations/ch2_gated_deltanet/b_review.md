# Chapter 2 — Critic Review (Agent B, Pass 1)

---

## Issue 1 — `g_t` range is inconsistent between Section 2 and Section 4 of `delta_rule_formulation.md`

**File:** `delta_rule_formulation.md`

**Where:**
- Section 2 symbols table: `g_t | scalar ∈ (0, 1] | Decay gate`
- Section 4: "Therefore g_t = exp(α_t) ∈ (0, 1) unconditionally."
- `index.md` notation table: `g_t | scalar ∈ (0, 1)`

**Problem:** Section 2 claims the upper bound is closed (`(0, 1]`, meaning g_t can equal 1), while Section 4 proves the bound is open (`(0, 1)`, meaning g_t is always strictly less than 1) because `softplus > 0` strictly. The index.md notation table agrees with Section 4. The Section 2 symbols table is wrong.

**Effect on reader:** An implementer writing a clamp or a range assertion would use the wrong bound if they read only Section 2. A reader reconciling Section 2 and Section 4 will not know which to trust.

**Fix:** Change the Section 2 symbols table entry to `g_t ∈ (0, 1)` (open interval), consistent with Section 4 and the index.md notation table.

---

## Issue 2 — `index.md` Key Takeaways states "approximately 1 MB" for the fp32 state, but the body calculation gives 2 MB

**File:** `index.md` (Key Takeaways, fourth bullet) and `delta_rule_formulation.md` (Section 7)

**Where:**
- `index.md` line: "The state S ∈ R^{128×128} per head is kept in fp32 for numerical stability; the full per-layer state at B=1 is approximately **1 MB**, independent of sequence length T."
- `delta_rule_formulation.md` Section 7: `524,288 × 4 = 2,097,152 bytes ≈ **2 MB**` (fp32); `524,288 × 2 = 1,048,576 bytes ≈ **1 MB**` (BF16).

**Problem:** The index.md preview says 1 MB, but 1 MB is the BF16 figure. The fp32 figure (which matches the stated `mamba_ssm_dtype: "float32"` configuration) is 2 MB. The KV cache comparison in Section 7 also uses the 2 MB figure. The 1 MB claim in the index is directly contradicted by the chapter's own arithmetic.

**Effect on reader:** A reader planning memory budgets from the Key Takeaways will underestimate the state memory by 2×.

**Fix:** Change "approximately 1 MB" in `index.md` to "approximately 2 MB (fp32)" to match the Section 7 calculation. Optionally note that the BF16 equivalent would be 1 MB, but the deployed configuration uses fp32.

---

## Issue 3 — Mamba2 formula in Section 2.3 omits A_log, making Section 5's "matches almost exactly" claim misleading

**File:** `comparison_to_linear_attention_variants.md`

**Where:**
- Section 2.3: `γ_t = exp(-softplus(a_t))`
- Section 5 (Mamba2 Relationship): `g_t = exp(-exp(A_log) · softplus(a_t + dt_bias))` with the claim "The decay gate computation in Gated DeltaNet matches Mamba2's parameterization almost exactly."

**Problem:** The Mamba2 formula shown in Section 2.3 has neither `exp(A_log)` scaling nor a `dt_bias` additive offset. The GDN formula in Section 5 has both. Section 5 then asserts GDN "matches Mamba2's parameterization almost exactly," which directly contradicts the two formulas being visibly different. A reader who reads Section 2.3 to understand Mamba2, then reads Section 5 trying to understand what GDN inherits from Mamba2, will correctly notice the formulas differ and be confused about what "almost exactly" means.

In Mamba2's actual parameterization, the A matrix (a learnable per-head log-scale parameter) and a dt_bias are indeed part of the SSM, so the Section 2.3 formula is an incomplete simplification that omits the parameters that GDN is said to borrow.

**Effect on reader:** A reader implementing either Mamba2 or GDN from this chapter may omit `exp(A_log)` from the decay computation, which would produce incorrect (non-learnable baseline) decay rates.

**Fix:** Update the Section 2.3 Mamba2 formula to include the A-matrix factor: `γ_t = exp(-exp(A_log) · softplus(a_t + dt_bias))` (with a note that A_log and dt_bias are the Mamba2 A-matrix and dt_bias parameters). Then Section 5's "almost exactly" claim becomes accurate, and the structural parallel between Mamba2 and GDN is clear.

---

# Chapter 2 — Critic Review (Agent B, Pass 2)

---

## Issue 1 — Pass 1 Issue 1 fix was incomplete: two occurrences of `g_t ∈ (0, 1]` remain in `delta_rule_formulation.md`

**File:** `delta_rule_formulation.md`

**Where:**
- **Line 37 (Section 2 display-math formula block):** `$$g_t = \exp(\alpha_t), \qquad \alpha_t < 0 \;\Rightarrow\; g_t \in (0, 1]$$`
- **Line 71 (Section 3.1 prose):** `The scalar $g_t \in (0, 1]$ multiplies every entry of the $d_k \times d_v$ state matrix uniformly.`

**Problem:** The pass 1 fix updated the Section 2 *symbols table* (now correctly `(0, 1)`) but left the closed upper bound `1]` in both the display-math condition on line 37 and the Section 3.1 description on line 71. A reader reading the formula block or Section 3.1 — the most natural starting points — will see the wrong interval. The symbols table fix is buried after both occurrences and contradicts them.

**Effect on reader:** A reader writing a validation assert (e.g., `assert 0 < g_t <= 1`) or trying to understand the edge-case behavior (`g_t = 1` means no forgetting at all) will get the wrong answer. Section 4 proves `g_t` is strictly less than 1 for any finite input; the residual `(0, 1]` claims an unreachable value is possible.

**Fix:**
- Line 37: change `g_t \in (0, 1]` to `g_t \in (0, 1)`.
- Line 71: change `g_t \in (0, 1]` to `g_t \in (0, 1)`.

---

# Chapter 2 — Critic Review (Agent B, Pass 3)

---

## Issue 1 — GLA gate is misrepresented as row-wise-only; stated limitation is factually wrong

**File:** `comparison_to_linear_attention_variants.md`

**Where:** Section 2.2 (GLA), and the rationale carried into Section 5 ("Favorable Comparison to GLA")

**Problem:** The document defines GLA's gate as $G_t = \alpha_t \mathbf{1}^\top$ where
$\alpha_t \in \mathbb{R}^{d_k}$ — a rank-1 outer product that is constant across all value
dimensions in each key row. It then states: "row-wise gating cannot selectively retain specific
(key-dim, value-dim) pairs; all value dimensions in the same key row decay at the same rate."

This is wrong. The GLA paper (Yang et al., 2023) uses a full $\mathbb{R}^{d_k \times d_v}$
element-wise gate — each entry $(i, j)$ of the state matrix has its own data-dependent decay
scalar. The actual GLA gate is strictly more expressive than the row-wise simplification
presented here.

The document's architectural argument in Section 5 that GDN's scalar gate is "simpler and more
natural" than GLA's row-wise gate relies on this mischaracterization. The real trade-off is
between a full $[d_k, d_v]$ gate (true GLA) and a single scalar (GDN), not between a
$d_k$-vector gate and a scalar.

**Effect on reader:**

1. A reader implementing GLA from this description would produce a row-wise gate instead of a
   full elementwise gate — the wrong architecture.
2. A reader accepting the Section 5 argument at face value would believe GDN was chosen over a
   less expressive alternative. The actual trade-off (full-matrix gate vs. scalar gate) is the
   opposite of "simpler and more natural vs. row-wise."

**Fix:** Section 2.2 should define the GLA gate as $G_t \in \mathbb{R}^{d_k \times d_v}$ (a
full learned elementwise gate, computed as $\sigma(\text{linear}(x_t))$ reshaped to
$[d_k, d_v]$), with the rank-1 structure noted as a special structured variant used for
efficient parallel scans. Section 5 should then accurately state the trade-off: Gated DeltaNet
uses a scalar gate (less parameter overhead, consistent with the delta correction term) in
contrast to GLA's full elementwise gate (more expressive forgetting, higher projection cost,
requires a different parallel scan strategy).

---

## Issue 2 — RMSNorm `mean` axis is unspecified; global vs. per-head computation gives different results

**File:** `delta_rule_formulation.md`

**Where:** Section 6, the gated RMSNorm formula

**Problem:** The formula is:

$$\text{normed}_t = o_t \cdot \bigl(\text{mean}(o_t^2) + \epsilon\bigr)^{-1/2} \cdot w_{\text{norm}}$$

The output $o_t$ has shape $[H_v, d_v] = [32, 128]$ across all heads. The formula does not
specify the axis of `mean`. There are two distinct interpretations:

- **Per-head (correct):** `mean` over the 128 elements of each head independently, giving 32
  separate normalization constants. Each head is normalized by its own RMS.
- **Global (wrong):** `mean` over all $32 \times 128 = 4096$ elements, giving a single scalar.
  This would conflate the scale of all 32 heads into one normalization and produce incorrect
  output.

The per-head interpretation is consistent with `w_norm` having shape $[d_v] = [128]$ (a single
shared scale applied per-dimension after per-head normalization). However, an implementer reading
the formula without this context could write `o_t.mean()` (global) instead of
`o_t.mean(dim=-1, keepdim=True)` (per-head), producing wrong results.

**Fix:** Add the axis to the formula explicitly:

$$\text{normed}_t[h] = o_t[h] \cdot \Bigl(\frac{1}{d_v}\sum_{j=1}^{d_v} o_t[h,j]^2 + \epsilon\Bigr)^{-1/2} \cdot w_{\text{norm}}$$

or equivalently add a note: "`mean` is computed over the $d_v = 128$ elements within each head
$h$ independently (axis = $d_v$ dimension)."

---

# Chapter 2 — Critic Review (Agent B, Pass 4)

---

## Issue 1 — Section 3 of `comparison_to_linear_attention_variants.md` still describes GLA as row-wise scalar, contradicting the pass 3 fix in Section 2.2 and Section 5

**File:** `comparison_to_linear_attention_variants.md`

**Where:**

- **Section 3, design choice 1 (lines 115–116):** "the same design as Mamba2, not GLA's row-wise
  $\alpha_t \in \mathbb{R}^{d_k}$"
- **Section 3, design choice 3 (line 127):** "From GLA/Mamba2: data-dependent **scalar** gating
  for coarse selective forgetting."

**Problem:** Pass 3 corrected Section 2.2 to define GLA's gate as $G_t \in \mathbb{R}^{d_k \times
d_v}$ (a full elementwise matrix, not a row-wise $d_k$-vector). Pass 3 also corrected Section 5
("Favorable Comparison to GLA") to state the real trade-off: "GLA has richer decay (full matrix,
any entry can be controlled independently) while GDN has simpler decay but adds targeted content
updates." Section 3 was not updated and still carries the pre-fix characterization in two places:

1. Design choice 1 contrasts GDN's scalar $g_t$ against "GLA's row-wise $\alpha_t \in
   \mathbb{R}^{d_k}$". After the fix, GLA is a full $d_k \times d_v$ matrix gate, not a $d_k$
   row vector. The contrast as written is wrong.

2. Design choice 3 groups "GLA/Mamba2" together as providing "data-dependent **scalar** gating".
   Mamba2 uses a scalar gate; GLA (per the corrected Section 2.2 and Section 5) uses a full
   $[d_k, d_v]$ elementwise matrix gate. Calling GLA's gate "scalar" is factually wrong.

**Effect on reader:** A reader learning what makes GDN's design distinctive will be told GDN
differs from GLA by being "scalar rather than row-wise" (line 116) and will attribute "scalar
gating" to GLA (line 127). Both claims conflict with what Section 2.2 and Section 5 now say. An
implementer using Section 3's framing to understand GLA would implement a $d_k$-vector row gate
instead of the full $[d_k, d_v]$ matrix gate — which is the same implementation error that pass
3 was intended to prevent in Section 2.2.

**Fix:**

- Design choice 1: replace "not GLA's row-wise $\alpha_t \in \mathbb{R}^{d_k}$" with "in contrast
  to GLA's full $G_t \in \mathbb{R}^{d_k \times d_v}$ elementwise gate" (or similar).
- Design choice 3: replace "From GLA/Mamba2: data-dependent **scalar** gating" with "From
  Mamba2/GLA: data-dependent gating for coarse selective forgetting" and note that GLA uses a
  full matrix gate while Mamba2 uses a scalar gate — GDN specifically adopts the scalar variant.

---

# Chapter 2 — Critic Review (Agent B, Pass 5)

---

## Issue 1 — `head_asymmetry_and_projections.md` Section 2.2 double-applies `w_norm` via ambiguous `RMSNorm` notation

**File:** `head_asymmetry_and_projections.md`

**Where:** Section 2.2 ("Output Gate Projection: `in_proj_z`"), the gated RMSNorm formula:

$$\text{output}_t = \text{RMSNorm}(o_t) \cdot w_{\text{norm}} \cdot \text{SiLU}(z_t)$$

**Problem:** Standard RMSNorm is defined as `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * w_norm`,
i.e., the learned scale `w_norm` is **part of the operation**. Under that standard definition the
formula above expands to:

$$\text{output}_t = \frac{o_t}{\sqrt{\text{mean}(o_t^2)+\epsilon}} \cdot w_{\text{norm}} \cdot w_{\text{norm}} \cdot \text{SiLU}(z_t)$$

meaning `w_norm` is applied twice — a squared scale that is not what is implemented.

`delta_rule_formulation.md` Section 6 writes the same operation without ambiguity:

$$\text{normed}_t = o_t \cdot \bigl(\text{mean}_{d_v}(o_t^2) + \epsilon\bigr)^{-1/2} \cdot w_{\text{norm}}$$
$$\text{output}_t = \text{normed}_t \cdot \text{SiLU}(z_t)$$

where the normalization step and the learned scale are shown explicitly and `RMSNorm(·)` is never
invoked. The `head_asymmetry` formula is inconsistent with this authoritative expansion and would
produce a different (wrong) result if taken literally.

**Effect on reader:** An implementer reading Section 2.2 and using a standard `nn.RMSNorm` module
(which internally multiplies by `w_norm`) would then multiply by `w_norm` a second time,
producing outputs scaled by `w_norm^2` instead of `w_norm`. At inference this inflates all
dimensions where `w_norm > 1` and deflates where `w_norm < 1`, silently corrupting output logits.

**Fix:** Replace the formula in Section 2.2 with the explicit form from `delta_rule_formulation.md`
Section 6 (normalization without the `RMSNorm` shorthand), or add a parenthetical clarifying that
`RMSNorm(o_t)` here denotes only the normalization step — `o_t / sqrt(mean(o_t^2) + eps)` — and
that `w_norm` is applied once, explicitly, in the outer expression.

---

# Chapter 2 — Critic Review (Agent B, Pass 6)

---

## Issue 1 — `delta_rule_formulation.md` Section 5 gives a wrong justification for the `1/sqrt(d_k)` query scaling

**File:** `delta_rule_formulation.md`

**Where:** Section 5, final paragraph (lines 159–164):

> "After L2 normalization, the query is also scaled by $1/\sqrt{d_k}$: ... This is the same
> temperature scaling used in standard dot-product attention and **prevents the dot product
> $\tilde{q}_t^\top \tilde{k}_t$ from saturating for large $d_k$**."

**Problem:** The stated justification is wrong. In standard (un-normalized) dot-product attention,
the $1/\sqrt{d_k}$ factor is needed because the inner product of two random $d_k$-dimensional
vectors has standard deviation $\sqrt{d_k}$; without the scaling, large $d_k$ drives the logit
into the saturating tails of softmax. That reasoning does not apply here: both $\tilde{q}_t$ and
$\tilde{k}_t$ have already been L2-normalized, so $|\tilde{q}_t^\top \tilde{k}_t| \leq 1$ by
Cauchy-Schwarz regardless of $d_k$. The dot product cannot saturate.

The actual purpose of the scaling here is to control the magnitude of the recurrence output. The
query is used to *read from the state*, not to form a softmax logit: $o_t = S_t^\top \tilde{q}_t$.
After many accumulated rank-1 writes, $S_t$ can have entries $O(T)$ in magnitude; scaling the
query by $1/\sqrt{d_k}$ keeps the output magnitude in a predictable range relative to the
dimension size. The justification given in the text is borrowed from the wrong setting.

**Effect on reader:** An implementer who reads this explanation and correctly notes "but Q and K are
already normalized, so there is no saturation issue" may reason that the $1/\sqrt{d_k}$ factor is
unnecessary and remove it. The actual $\tilde{q}_t$ used in the output retrieval $S_t^\top
\tilde{q}_t$ would then be $\sqrt{d_k} = \sqrt{128} \approx 11.3\times$ larger, producing outputs
that are over an order of magnitude too large and causing downstream logit and loss instability.

**Fix:** Replace the justification. The correct explanation is that the output $o_t = S_t^\top
\tilde{q}_{\text{scaled}}$ scales linearly with the query norm, and after many token steps $S_t$
accumulates $O(T)$ outer-product mass; dividing the query by $\sqrt{d_k}$ normalizes the expected
output magnitude to be comparable to the un-accumulated case, keeping activations in range across
sequence lengths.

---

# Chapter 2 — Critic Review (Agent B, Pass 7)

---

## Issue 1 — Core recurrence in Section 2 uses unscaled $\tilde{q}_t$ in the output formula; the $1/\sqrt{d_k}$ scaling introduced in Section 5 is never reflected in the canonical formula

**File:** `delta_rule_formulation.md`

**Where:**

- **Section 2, output formula (line 43):** $o_t = S_t^\top \tilde{q}_t$
- **Section 2, symbols table (line 51):** `$\tilde{q}_t$ | $[d_k] = [128]$ | L2-normalized query vector`
- **Section 5 (lines 159–161):** introduces $\tilde{q}_{\text{scaled}} = \tilde{q}_t / \sqrt{d_k}$
  as a new variable and states this is what is used for the output read, fixing only the
  *justification* for the scaling (pass 6) but not updating the canonical formula or symbol table.
- **`index.md` notation table:** defines $\tilde{q}_t$ as "L2-normalized query vector" with no
  mention of the $1/\sqrt{d_k}$ factor.

**Problem:** The definitive output step shown in Section 2 is $o_t = S_t^\top \tilde{q}_t$, where
$\tilde{q}_t$ is defined everywhere as only the L2-normalized query. The actual computation (per
Section 5) is $o_t = S_t^\top (\tilde{q}_t / \sqrt{d_k})$. The two are different by a factor of
$\sqrt{128} \approx 11.3$. Section 5 quietly introduces a renamed variable $\tilde{q}_{\text{scaled}}$
but the canonical formula in Section 2 is never updated to use it, and the symbol table never
acknowledges the scaling.

An implementer who copies the Section 2 recurrence verbatim — the natural starting point — will
omit the $1/\sqrt{d_k}$ factor, producing outputs approximately 11× too large. This error is not
prevented by the pass 6 fix, which only corrected the prose *reason* for the scaling, not the
formula that shows *whether* the scaling is present.

**Effect on reader:** The core recurrence as stated in Section 2 is wrong as a self-contained
specification. Any implementation derived from it alone produces numerically incorrect output.

**Fix:** Update the Section 2 output formula to use the scaled query directly:

$$o_t = S_t^\top \left(\frac{\tilde{q}_t}{\sqrt{d_k}}\right)$$

and update the symbols table entry for $\tilde{q}_t$ to note that it enters the output step scaled
by $1/\sqrt{d_k}$, or rename it to $\tilde{q}_{\text{scaled}}$ throughout and drop the separate
Section 5 introduction. Also update the `index.md` notation table accordingly.

---

# Chapter 2 — Critic Review (Agent B, Pass 8)

No feedback — chapter approved.

---

# Chapter 2 — Critic Review (Agent B, Pass 9)

No feedback — chapter approved.
