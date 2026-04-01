# Agent B Review — Chapter 2: GatedDeltaNet — Pass 1

1. **`recurrence_math.md`, ~line 206 — GatedRMSNorm formula is wrong.**
   The guide writes $\text{output\_normed} = \mathbf{o}_t / (\text{RMS}(\mathbf{o}_t) + \epsilon)$, which places epsilon *outside* the square root (i.e., $x / (\sqrt{\text{mean}(x^2)} + \epsilon)$). The actual implementation in `Qwen3_5RMSNormGated.forward` (and the reference test) computes `hidden_states * torch.rsqrt(variance + self.variance_epsilon)`, which is $x / \sqrt{\text{mean}(x^2) + \epsilon}$ — epsilon is *inside* the sqrt, added to the variance. These are numerically distinct operations. A reader implementing from the formula would produce a different (incorrect) normalizer.
   **Fix:** Change the formula to $\text{output\_normed} = \mathbf{o}_t \cdot \left(\text{mean}(\mathbf{o}_t^2) + \epsilon\right)^{-1/2} \cdot \mathbf{w}_{\text{norm}}$, matching the code.

2. **`host_recurrence.md`, ~lines 99–102 — decomposition components do not sum to the stated 54 ms.**
   The text says "The 54 ms for DeltaNet decomposes as approximately: ~35 ms total sync overhead + ~26 ms Python dispatch + ~20 ms device compute." 35 + 26 + 20 = 81 ms, not 54 ms. The three terms are roughly the breakdown of the full 86 ms token time (as confirmed by PERF.md), not of the 54 ms DeltaNet row alone. A reader doing arithmetic on these numbers to understand where the 54 ms comes from will get a wrong answer.
   **Fix:** Either attribute the breakdown to the full 86 ms total (matching how PERF.md presents it), or correct the per-component numbers so they sum to 54 ms.

---

# Agent B Review — Chapter 2: GatedDeltaNet — Pass 2

1. [`host_recurrence.md`, ~line 27, compressed recurrence equation conceals that δ depends on the post-decay state] The equation `S ← S·exp(g) + k⊗δ` presents δ as an independent operand. In fact δ = (v − (S·exp(g))ᵀ k)·β — it is computed from the *already-decayed* state (Step 2 reads the decayed S to get m, Step 3 uses m to get δ). An implementer who reads only this line and computes δ from the original S before applying the decay will produce a numerically wrong update. Fix: add a note after the equation — e.g., "where δ is itself a function of the decayed state S·exp(g) (see the five-step recurrence)" — or replace the compressed form with the explicit two-line version: `S ← S·exp(g); δ = (v − Sᵀk)·β; S ← S + k⊗δ`.

2. [`recurrence_math.md`, ~lines 64–65 and 105, code snippet claimed as "exact" omits a branch present in the real source] The guide states "The following is the exact Python reference used in the test suite" but the snippet (line 105) unconditionally returns `last_recurrent_state`. The actual function in `reference/test_deltanet_pcc.py` (lines 63–66) contains `if not output_final_state: last_recurrent_state = None` before the return. Any reader implementing the function from the guide's listing would produce a version that always returns the state regardless of `output_final_state`, violating the actual interface contract. Fix: either include the missing branch in the snippet, or remove the word "exact" and note that the snippet is a simplified excerpt.

3. [`fused_kernel.md`, ~lines 74 and 81–83, permute guard condition misattributed to "batch > 1"] The comment in the code block (`# batch > 1: permute needed`) and the prose explanation ("The conditional permute handles the batch > 1 case") both say the permute fires when batch count exceeds 1. But the actual condition is `output_tt.shape[2] > 1`, which checks dim 2 (the padded batch-row dimension B_pad, always 32 in production). The condition is therefore always true; it has nothing to do with the batch-count dimension (dim 0). An engineer trying to understand or extend this logic for a genuinely batch-size-1 vs batch-size-N distinction would be misled. Fix: correct the comment and prose to say the permute is applied whenever B_pad > 1 (i.e., always), and explain that `shape[2]` is the tile-padded row dimension, not the outer batch dimension.

---

# Agent B Review — Chapter 2: GatedDeltaNet — Pass 3

1. **`host_recurrence.md`, line 108 — efficiency percentage is arithmetically wrong.**
   The text states: "The theoretical compute limit is ~5.8 ms/token (172 tok/s). Current efficiency is 6.3%."
   At 86 ms/token, measured throughput is 1000/86 ≈ 11.6 tok/s. The theoretical limit is 1000/5.8 ≈ 172 tok/s. The ratio is 11.6/172 ≈ 6.7%, not 6.3%. (Using the A3B measured speed of 11.7 tok/s from PERF.md gives 11.7/172 = 6.8%.) No arithmetic path from the stated inputs yields 6.3%. A reader who uses this figure to cross-check system efficiency measurements will get the wrong answer.
   **Fix:** Replace "6.3%" with "~6.7%" (or derive it explicitly from 11.6/172).

---

# Agent B Review — Chapter 2: GatedDeltaNet — Pass 4

1. **`recurrence_math.md`, lines 156–157 — L2 norm denominator formula is wrong.**
   The guide writes:
   $$\hat{\mathbf{q}} = \frac{\mathbf{q}}{\|\mathbf{q}\|_2 + \epsilon}$$
   The actual implementation (source `l2norm`, line 163 of `test_deltanet_pcc.py`, and kernel call path) computes `x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)`, which is:
   $$\hat{\mathbf{q}} = \frac{\mathbf{q}}{\sqrt{\|\mathbf{q}\|_2^2 + \epsilon}}$$
   The guide adds $\epsilon$ outside the square root (to the norm), but the code adds $\epsilon$ inside the square root (to the squared norm before taking the root). These are numerically different: for a near-zero vector the guide's formula would divide by approximately $\epsilon$, whereas the code divides by $\sqrt{\epsilon}$. An implementer who codes the guide's formula will produce a normalizer that is off by a factor of $\sqrt{\epsilon}/\epsilon = 1/\sqrt{\epsilon} \approx 1000$ for near-zero inputs, and subtly wrong for all inputs.
   **Fix:** Change the denominator to $\sqrt{\|\mathbf{q}\|_2^2 + \epsilon}$ in both the Q and K formulas.

2. **`fused_kernel.md`, line 219 — efficiency percentage still states 6.3%.**
   The "Current Profiling Context" table and accompanying text in `fused_kernel.md` says "Current efficiency: 6.3%." Pass 3 identified that the correct figure is ~6.7% (1000/86 ≈ 11.6 tok/s ÷ 172 tok/s theoretical = 6.76%), and flagged this error in `host_recurrence.md`. However, `fused_kernel.md` repeats the same number uncorrected. A reader comparing the two files will see 6.3% and 6.7% for identical inputs and conclude one of them is wrong — and the 6.3% figure is the one that cannot be derived from the stated inputs.
   **Fix:** Replace "6.3%" in `fused_kernel.md` with "~6.7%" to match `host_recurrence.md` once that file is corrected, and to match the arithmetic.

---

# Agent B Review — Chapter 2: GatedDeltaNet — Pass 5

1. **`fused_kernel.md`, line 174 vs line 145 — multi-step test step count is internally inconsistent.**
   The prose at line 145 states: "The `test_deltanet_multi.py` reference test runs **20** sequential decode steps through a single DeltaNet layer." The PCC threshold table at line 174 then gives the rationale for the 0.95 floor as "Conservative floor for **10** accumulated decode steps." Both claims refer to the same test and the same threshold; they cannot both be correct. The body text ("20 steps") is the more detailed and specific claim; the table footnote ("10 steps") contradicts it with a different number that has no other support in the chapter.
   **Fix:** Change the table rationale at line 174 to "Conservative floor for 20 accumulated decode steps" to match the body text.

---

# Agent B Review — Chapter 2: GatedDeltaNet — Pass 6

No feedback — chapter approved.

---

# Agent B Review — Chapter 2: GatedDeltaNet — Pass 7

1. **`host_recurrence.md`, line 99 — Attention sync count contradicts the stated total.**
   The profiling table lists DeltaNet 30 syncs + Attention 50 syncs + norm/LM head 1 sync = 81 syncs, but the Total row reads "~70". The column does not add up. Additionally, 50 syncs across 10 attention layers implies 5 syncs per attention layer, which is 5x the per-layer sync cost of DeltaNet (1 sync/layer for the state roundtrip) and is never explained. Either the Attention sync count of 50 is wrong (a plausible value would be 5, giving 30+5+1=36, or some figure consistent with ~70 total), or the Total of ~70 is wrong.
   **Fix:** Correct the Attention sync count and/or the Total so the column sums match and the per-layer figure is explained or substantiated.

2. **`host_recurrence.md`, line 109 — efficiency derivation formula is arithmetically incoherent as written.**
   The text states: "measured ~11.6 tok/s from 86 ms/token divided by theoretical 172 tok/s = ~6.7%". Reading this literally, "86 ms/token divided by 172 tok/s" = 86/172 = 0.5, which is 50%, not 6.7%. The actual derivation requires first converting 86 ms/token to a throughput (1000/86 ≈ 11.6 tok/s), then dividing by 172 tok/s to get ~6.7%. The intermediate conversion step is absent from the parenthetical, making the written arithmetic wrong even though the final answer (6.7%) is correct. An implementer cross-checking the arithmetic from the text as written cannot reproduce 6.7% from the stated operation.
   **Fix:** Rewrite the parenthetical to read: "measured ~11.6 tok/s (= 1000 ms ÷ 86 ms/token) divided by theoretical 172 tok/s = ~6.7%".

---

# Agent B Review — Chapter 2: GatedDeltaNet — Pass 8

No feedback — chapter approved.

---

# Agent B Review — Chapter 2: GatedDeltaNet — Pass 9

No feedback — chapter approved.
