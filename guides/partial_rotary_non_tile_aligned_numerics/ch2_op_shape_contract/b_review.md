# B Review — Pass 1

1. [`index.md`, ~line 25, Data-flow diagram: Python caller annotation states wrong shape requirement]
   The data-flow diagram annotates the "Python caller" boundary with `cos_cache.shape[-1] == rotary_dim (or padded variant)`. This is the incorrect shape. The op requires `cos_cache.shape[-1] == head_dim`; providing `rotary_dim` or `nearest_32(rotary_dim)` is exactly the failure mode the chapter is documenting. The annotation reads as a shape requirement at that stage, not as a description of a buggy caller. Fix: change the annotation to `cos_cache.shape[-1] == head_dim  (required; rotary_dim and nearest_32(rotary_dim) both rejected)` to match the gate stated two lines later in the same diagram.

2. [`index.md` / `kernel_rotate_half_pairing.md`, Convention reconciliation missing: `rotary_dim/2` vs `head_dim/2` split not explicitly named as distinct conventions]
   Ch1 `index.md` glossary defines "rotate-half pairing" as pairing element `i` in `[0, rotary_dim/2)` with element `i + rotary_dim/2` — the Python partial-RoPE convention. The kernel analysis in `kernel_rotate_half_pairing.md` documents that the kernel pairs on `head_dim/2`. These are two different split points (24 vs 64 for the Qwen3 example), and neither file states explicitly that they are different conventions operating at different layers. A reader who holds the Ch1 glossary definition while reading Ch2 will conflate them. The `kernel_rotate_half_pairing.md` section 4 mentions the correct math uses `[0,24)` paired with `[24,48)` and the kernel uses `[0,64)` paired with `[64,128)`, which is close — but it frames this as "wrong pairing" rather than "different convention". Fix: add one paragraph in `kernel_rotate_half_pairing.md` (or in `index.md`'s recap section) explicitly stating: "The Ch1 glossary defines rotate-half pairing relative to `rotary_dim/2` — this is the Python partial-RoPE convention used in the reference formula. The TTNN kernel always pairs on `head_dim/2` because it derives `half_Wt` from the full padded `head_dim`. These are different conventions at different layers of the stack and must not be conflated."

3. [`shape_validation_in_invoke.md`, ~lines 38–53, `TT_FATAL` code block adds `[0]==1` and `[1]==1` checks not supported by the stated domain facts]
   The code block shown for `RotaryEmbeddingOperation::invoke` includes checks `cos_cache.padded_shape()[0] == 1 && cos_cache.padded_shape()[1] == 1` alongside the `[-1] == X` check. The domain facts provided for this review confirm only the `cos_cache.shape[-1] == head_dim` gate. The batch/head-dim-1 equality checks are an embellishment not mentioned in the domain facts and are presented as actual source code. If these checks are not in the real source, the code block is fabricated. Fix: either verify these checks exist in the actual source and cite the exact function signature, or remove `[0]==1 && [1]==1` from the code block and update the error message string to match only the confirmed constraint.

4. [`what_the_golden_function_reveals.md`, ~lines 56–57, Hedged claim about `rotary_dim` usage in golden body is vague and internally inconsistent]
   Section 2 states: "The `rotary_dim` parameter is accepted by the golden function signature but is not used in the body shown above. In the actual implementation it may gate a slice: `input[..., :rotary_dim]` for the rotated portion..." This hedge introduces an alternative code path (slicing on `input`) that contradicts the chapter's own conclusion that the golden applies rotation to the full last dimension. If the actual implementation slices `input[..., :rotary_dim]`, then `cos.shape[-1]` need only be `rotary_dim`, not `head_dim` — which contradicts the Key Finding in section 3. Fix: either show the actual golden body (including the `rotary_dim` slice if present) or remove the speculative alternative. The Key Finding must be consistent with the code that is actually shown.

5. [`kernel_rotate_half_pairing.md`, ~lines 88–92, Implication section overstates incorrectness without acknowledging Strategy C]
   Section 4 states the kernel's pairing `([0,64)` vs `[64,128))` is "entirely wrong for this model." This is correct when cos/sin carry only `rotary_dim`-wide values — but the same kernel pairing becomes correct under Strategy C (identity-filled cos/sin with `cos=1, sin=0` in `[48,128)`), which is the fix introduced in Chapter 4. The unqualified claim "entirely wrong" is accurate only for the naive (zero-padded or rotary_dim-wide) cos/sin case. The chapter does mention Strategy C at the end of the section but the "entirely wrong" label is not scoped to the naive case. Fix: scope the claim: "With naive cos/sin construction (zero-padded to `head_dim` or only `rotary_dim`-wide), this pairing is entirely wrong for this model. Strategy C — analyzed in Chapter 4 — corrects this by filling `[rotary_dim, head_dim)` with identity values so the same pairing produces correct output."

---

# B Review — Pass 2 (Change Log)

Changes applied in response to Pass 1:
1. `index.md` diagram: changed `rotary_dim (or padded variant)` annotation to `head_dim`
2. `kernel_rotate_half_pairing.md`: added explicit reconciliation sentence for `rotary_dim/2` vs `head_dim/2` convention difference
3. `shape_validation_in_invoke.md`: removed unconfirmed batch/num-heads dimension checks; added Note about verification scope
4. `what_the_golden_function_reveals.md`: removed hedged speculation about `rotary_dim` slicing; replaced with direct statement about full-width slice
5. `kernel_rotate_half_pairing.md`: scoped "entirely wrong" to naive padding case; added explicit forward reference to Strategy C

---

# B Review — Pass 2

1. [`kernel_rotate_half_pairing.md`, line 96, wrong chapter number for Strategy C forward reference]
   The line reads: "Under Strategy C (identity values at `[rotary_dim:]` — see Chapter 3), the same kernel produces correct output." Strategy C is introduced and analyzed in Chapter 4 (root cause analysis is Chapter 3). The same file correctly names Chapter 4 nine lines later (line 105): "This is **Strategy C (identity-filled cos/sin)**, which is analyzed in Chapter 4." Fix: change "see Chapter 3" to "see Chapter 4" on line 96, making it consistent with the correct reference at line 105 and with `index.md`'s description of each chapter.

2. [`what_the_golden_function_reveals.md`, lines 9–17 and 47, golden `rotate_half` splits on `head_dim/2`, not `rotary_dim/2` — contradicts the stated domain fact]
   The `rotate_half` helper is shown as splitting at `x.shape[-1] // 2`. When `x` is the full input with `head_dim=128`, this is 64. The domain fact states: "Python golden pairs on `rotary_dim/2` (= 24 for rotary_dim=48)." `kernel_rotate_half_pairing.md` line 52 correctly cites this distinction: "The Python golden function pairs on `rotary_dim/2`; the TTNN kernel always pairs on `head_dim/2`." But the golden body shown in `what_the_golden_function_reveals.md` applies `rotate_half(input)` to the full-width input with no prior slice to `rotary_dim`, producing a split at `head_dim/2 = 64`, not `rotary_dim/2 = 24`. This is internally inconsistent and contradicts the authoritative domain fact. The real golden likely slices `input[..., :rotary_dim]` before calling `rotate_half`, so that `x.shape[-1] = rotary_dim` and the split lands at `rotary_dim/2`. Fix: either (a) show the actual golden body that slices input to `rotary_dim` before calling `rotate_half`, or (b) if the simplified golden truly operates on the full input width, remove the claim in `kernel_rotate_half_pairing.md` that "The Python golden function pairs on `rotary_dim/2`" and update the domain-fact-corroborating sentence in the Key Finding note at line 52 of that file. The two files must be consistent with each other and with the domain fact.

---

# B Review — Pass 3 (Change Log)

Changes applied in response to Pass 2:
1. `kernel_rotate_half_pairing.md` ~line 96: changed "see Chapter 3" to "see Chapter 4" for Strategy C reference
2. `what_the_golden_function_reveals.md`: fixed rotate_half pairing explanation — the golden pre-slices to rotary_dim before calling rotate_half, so the split is at rotary_dim/2=24, not head_dim/2=64

---

# B Review — Pass 3

1. [`index.md`, line 29, data-flow diagram: annotation states golden applies cos/sin to full `input[-1]` — contradicts the corrected golden body]
   The diagram entry for the `ttnn.experimental.rotary_embedding(...)` step is annotated: `(transformer.py) — golden function applies cos/sin to full input[-1]`. After Pass 2, `what_the_golden_function_reveals.md` was corrected to show the golden pre-slices the input to `[..., :rotary_dim]` before calling `rotate_half`, then concatenates the passthrough tail `x_pass = input[..., rotary_dim:]` unchanged. The golden therefore does NOT apply cos/sin to the full `input[-1]`; it applies them only to `input[..., :rotary_dim]`. This diagram annotation is now inconsistent with the corrected golden body in `what_the_golden_function_reveals.md` and with domain fact #5 ("Python golden pre-slices input to `[..., :rotary_dim]` before `rotate_half`"). Fix: change the annotation from `(transformer.py) — golden function applies cos/sin to full input[-1]` to `(transformer.py) — golden slices input to [:rotary_dim], applies cos/sin, concatenates passthrough unchanged`.

---

# B Review — Pass 4 (Change Log)

Changes applied in response to Pass 3:
1. `index.md` diagram ~line 29: updated golden function annotation — it pre-slices input to [:rotary_dim] before applying cos/sin, then concatenates passthrough; does not apply to full input

# B Review — Pass 4

No feedback — chapter approved.
