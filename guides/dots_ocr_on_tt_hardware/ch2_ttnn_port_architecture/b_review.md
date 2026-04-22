## Agent B Review — Pass 1

**3 factual correctness issues found.**

---

### Issue 1 — `model_args_and_transformer.md`, lines 15–25: Wrong mechanism for `trust_remote_code_hf` and `dummy_weights`

The section heading reads "Fixed Class-Level Overrides — Three fields are set as class-level attributes and are not configurable at instantiation." This is incorrect for two of the three fields:

- `trust_remote_code_hf = True` is set in `__post_init__`, not as a class-level attribute.
- `dummy_weights = False` is forced via kwargs at instantiation, not as a class-level attribute.

Only `use_hf_rope = True` appears to be a class-level attribute in the base sense. An implementer following this description would place all three as bare class-level assignments, which would not reproduce the actual initialization behavior — particularly for `trust_remote_code_hf`, which must be set after `__init__` has run.

**Correction:** Distinguish the three mechanisms. `use_hf_rope = True` is a class-level default. `trust_remote_code_hf = True` is set in `__post_init__`. `dummy_weights = False` is enforced by passing it as a kwarg during instantiation.

---

### Issue 2 — `model_args_and_transformer.md`, lines 57–59: LM head op count formula is TP=1 only, but is presented without qualification

The formula and result are numerically correct for TP=1:

$$\left\lceil \frac{151936}{2048} \right\rceil = 75 \text{ ops per device}$$

However, the text presents this as the general case with no mention of tensor parallelism degree. At TP=2, each device handles only half the vocabulary columns (151936 / 2 = 75968 unsharded), and the per-device op count changes to ceil(75968 / 2048) = 38, not 75. A reader deploying on a multi-device T3K mesh (the standard dots.ocr target) would compute the wrong op count from this formula.

**Correction:** Add the qualification that the formula applies at TP=1. For TP=2 (the T3K default), each device holds 75968 columns, yielding 38 ops per device.

---

### Issue 3 — `pcc_validation_framework.md`, line 114: `test_e2e_pcc.py` PCC target presented as a confirmed measurement

The end-to-end test description states: "A PCC > 0.99 here means the full multimodal forward pass on TT hardware agrees with the HF baseline within acceptable numerical tolerance." This implies the > 0.99 threshold has been measured and confirmed for the end-to-end pipeline.

The ground truth for this number is the `IMPLEMENTATION_STEPS.md` claim "PCC > 0.99: Framework implemented across all components," which is a stated target, not a confirmed measurement. The only confirmed PCC figure from commit history is > 0.98 for text prefill. Presenting the > 0.99 E2E figure as a measured result would mislead a reader validating their own deployment against this number.

**Correction:** Mark the > 0.99 E2E figure as a target from `IMPLEMENTATION_STEPS.md`, not a confirmed measurement. The confirmed figure is > 0.98 for text prefill, sourced from the commit message.

---

## Agent A Change Log — Pass 1

- Fix 1 applied: Corrected DotsModelArgs initialization: dummy_weights is forced via kwargs before super().__init__(); trust_remote_code_hf is set post-init as self.trust_remote_code_hf = True (not a class-level attribute).
- Fix 2 applied: Qualified LM head op-count: 75 ops/device at TP=1; 38 ops/device at TP=2 (T3K deployment with 75968 columns per device).
- Fix 3 applied: Distinguished confirmed PCC (>0.98 text prefill from commit history) from target (>0.99 per IMPLEMENTATION_STEPS.md, not independently confirmed).

---

## Agent B Review — Pass 2

**1 factual correctness issue found.**

---

### Issue 1 — `pcc_validation_framework.md`, line 15: PCC range stated as $(-1, 1]$ instead of $[-1, 1]$

The text states: "The result is a scalar in $(-1, 1]$". This is mathematically incorrect. Pearson correlation coefficient has the closed range $[-1, 1]$; both endpoints are achievable. A value of $-1$ occurs exactly when one vector is a perfect negative linear transformation of the other ($\mathbf{b} = -c\mathbf{a} + d$ for $c > 0$). Writing $(-1, 1]$ incorrectly excludes $-1$ as a possible output.

While perfect negative correlation is rare in practice for neural network activations, the stated range is wrong and a reader implementing a PCC bounds-check in code (e.g., `assert -1 < pcc <= 1`) would have a subtly incorrect assertion.

**Correction:** Change $(-1, 1]$ to $[-1, 1]$.

## Agent A Change Log — Pass 2

- Fix applied: Corrected PCC range from $(-1, 1]$ to $[-1, 1]$ in pcc_validation_framework.md (Pearson correlation can equal -1 for perfect negative correlation).

---

## Agent B Review — Pass 3

No feedback — chapter approved.
