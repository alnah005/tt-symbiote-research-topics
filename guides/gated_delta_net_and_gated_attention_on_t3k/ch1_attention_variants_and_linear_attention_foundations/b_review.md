# B Review — Chapter 1 — Pass 2

1. [ALL FILES, N/A] None of the four files specified for review exist on disk. The entire guide directory `guides/gated_delta_net_and_gated_attention_on_t3k/` is absent. Files missing: `index.md`, `softmax_attention_review.md`, `linear_attention_landscape.md`, `deltanet_baseline.md`. There is nothing to review — chapter content must be authored before a correctness review is possible.

# B Review — Chapter 1 — Pass 3

1. **`linear_attention_rnn_equivalence.md`, lines 45 and 65 — Readout formula has wrong dimension.**
   The state is defined as `S_t ∈ R^{d_k × d_v}` (via the outer product `φ(k_j) v_j^T` where `φ(k_j) ∈ R^{d_k}` and `v_j ∈ R^{d_v}`). The readout formulas are written as `o_t = S_t φ(q_t) / z_t^T φ(q_t)` (line 45) and `o_t = S_t q_t` (line 65). With `S_t ∈ R^{d_k × d_v}` and `φ(q_t) ∈ R^{d_k}`, the product `S_t φ(q_t)` is not dimensionally valid. The correct readout is `o_t = S_t^T φ(q_t)` (producing `R^{d_v}`). A reader implementing this directly would get a shape error or, if they silently transposed, an incorrect formula. Fix: replace `S_t φ(q_t)` with `S_t^T φ(q_t)` on line 45, and `S_t q_t` with `S_t^T q_t` on line 65.

2. **`linear_attention_variants_comparison.md`, lines 39–40 — GLA gate direction mislabelled as "column-wise" when it is row-wise.**
   The gate is `G_t = α_t 1^T ∈ R^{d_k × d_v}`, which means entry `(i, j) = α_t[i]`. Row `i` of `S_{t-1}` is uniformly scaled by `α_t[i]` across all `d_v` columns — this is **row-wise** decay indexed by key-dimension `i`. The file calls it "column-wise data-dependent decay" and later says "each key dimension can independently control how much of its row in S is retained," which correctly describes row-wise behavior but directly contradicts the "column-wise" label. A reader building a mental model or implementing GLA would assign the gate to the wrong axis. Fix: replace "column-wise data-dependent decay" with "row-wise data-dependent decay" (i.e., per key-dimension decay applied uniformly across value dimensions).

3. **`linear_attention_rnn_equivalence.md`, lines 27–33 — `φ(k_t) v_t^T` and `v_t k_t^T` presented as equivalent without qualification.**
   Line 27 states the recurrence as `S_t = S_{t-1} + φ(k_t) v_t^T`, then line 33 says "or equivalently... `S_t = S_{t-1} + v_t k_t^T`." These two forms are only equivalent when `φ` is the identity function. For any non-identity kernel (e.g., `φ(x) = elu(x) + 1`), they differ. Beyond the unqualified "equivalent", note that `φ(k_t) v_t^T` and `v_t k_t^T` also have opposite shapes (`R^{d_k × d_v}` vs `R^{d_v × d_k}`), so at most one can be correct given the declared state shape `S_t ∈ R^{d_k × d_v}`. The second form `v_t k_t^T` is dimensionally inconsistent with the stated `S_t ∈ R^{d_k × d_v}`. Fix: remove the "or equivalently" clause on lines 32–34; retain only `S_t = S_{t-1} + φ(k_t) v_t^T` as the definitive recurrence.

## Agent A Change Log — Pass 3

Applied all three feedback items from Agent B's Pass 3 review:

1. **Item 1 — Readout formula dimension fix (`linear_attention_rnn_equivalence.md`):** Replaced `o_t = S_t φ(q_t)` with `o_t = S_t^T φ(q_t)` (line ~45) and `o_t = S_t q_t` with `o_t = S_t^T q_t` (line ~65). Also updated the prose reference `S_T q` to `S_T^T q` for consistency.

2. **Item 2 — GLA gate direction label fix (`linear_attention_variants_comparison.md`):** Changed "column-wise data-dependent decay" to "row-wise data-dependent decay" in the GLA section prose, the section introduction line, the limitation bullet, and the summary table. The outer product `α_t 1^T` scales each row of S by the corresponding element of `α_t`, making it row-wise (per key-dimension) decay.

3. **Item 3 — State write shape fix (`linear_attention_rnn_equivalence.md` and `linear_attention_variants_comparison.md`):** Removed the "or equivalently, writing v_t as a column vector..." block that incorrectly equated `φ(k_t) v_t^T` with `v_t k_t^T`. Replaced all `v_t k_t^T` state-write occurrences with `k_t v_t^T` (the correct `R^{d_k × d_v}` outer product) across both files, including the Forgetting Problem section and all gated-update formulas in the variants comparison.

# B Review — Chapter 1 — Pass 4

1. **`linear_attention_variants_comparison.md`, line 63 — DeltaNet compact form is dimensionally inconsistent.**
   With `S_{t-1} ∈ R^{d_k × d_v}` and `k̃_t ∈ R^{d_k}`: the product `S_{t-1} k̃_t ∈ R^{d_v}`, so `(S_{t-1} k̃_t - v_t) ∈ R^{d_v}`, and the outer product `(S_{t-1} k̃_t - v_t) k̃_t^T ∈ R^{d_v × d_k}`. Subtracting this from `S_{t-1} ∈ R^{d_k × d_v}` is a shape mismatch. A reader implementing this directly gets a runtime shape error. Fix: rewrite as `S_t = S_{t-1} - β_t k̃_t (k̃_t^T S_{t-1} - v_t^T)`, where `k̃_t^T S_{t-1} ∈ R^{d_v}^T` and `k̃_t (k̃_t^T S_{t-1} - v_t^T) ∈ R^{d_k × d_v}`, which is consistent with `S ∈ R^{d_k × d_v}`.

2. **`linear_attention_variants_comparison.md`, line 71 — DeltaNet expanded form has wrong multiplication order and wrong write term.**
   The file shows `S_t = S_{t-1} (I - β_t k̃_t k̃_t^T) + β_t v_t k̃_t^T`. With `S_{t-1} ∈ R^{d_k × d_v}`, right-multiplying by `(I - β_t k̃_t k̃_t^T) ∈ R^{d_k × d_k}` requires the inner dimensions to match: `(d_k × d_v) · (d_k × d_k)` is invalid. Additionally, `v_t k̃_t^T ∈ R^{d_v × d_k}` but S lives in `R^{d_k × d_v}`. Fix: replace with `S_t = (I - β_t k̃_t k̃_t^T) S_{t-1} + β_t k̃_t v_t^T`, where left-multiplication `(d_k × d_k) · (d_k × d_v)` is valid and the write term `k̃_t v_t^T ∈ R^{d_k × d_v}` is correct.

3. **`linear_attention_variants_comparison.md`, line 74 — projection description inverts the correct multiplication side.**
   Prose states "`(I - β_t k̃_t k̃_t^T)` subtracts the component of S_{t-1} along k̃_t" in a way that implicitly treats the projection as right-multiplying S. With `S ∈ R^{d_k × d_v}`, the rank-1 projection `(I - β_t k̃_t k̃_t^T)` must **left**-multiply S to zero out the rows of S aligned with k̃_t. A reader implementing from the prose alongside the incorrect formula on line 71 would transpose the projection application. Fix: clarify that the projection left-multiplies S: "`(I - β_t k̃_t k̃_t^T) S_{t-1}` zeroes the component of each column of S_{t-1} that lies along k̃_t."

4. **`linear_attention_variants_comparison.md`, line 99 — Gated Delta Net teaser write term has wrong outer product order.**
   The teaser formula ends with `+ β_t v_t k̃_t^T`. With `v_t ∈ R^{d_v}` and `k̃_t ∈ R^{d_k}`, the outer product `v_t k̃_t^T ∈ R^{d_v × d_k}`, which is inconsistent with `S ∈ R^{d_k × d_v}`. This propagates the same error as item 2 into the forward reference. A reader previewing Gated Delta Net from this section would build an incorrect mental model of the write term shape. Fix: replace `β_t v_t k̃_t^T` with `β_t k̃_t v_t^T` so the write term is `∈ R^{d_k × d_v}`.

## Agent A Change Log — Pass 4

Applied all 4 feedback items from Agent B's Pass 4 review to `linear_attention_variants_comparison.md`:

1. **Item 1 — DeltaNet compact form dimensional fix (~line 63):** Replaced `S_t = S_{t-1} - β_t (S_{t-1} k̃_t - v_t) k̃_t^T` with the correct form `S_t = S_{t-1} - β_t k̃_t (k̃_t^T S_{t-1} - v_t^T)`. In the corrected form, `k̃_t^T S_{t-1} ∈ R^{d_v}` (row vector), `k̃_t^T S_{t-1} - v_t^T ∈ R^{d_v}`, and the outer product `k̃_t (k̃_t^T S_{t-1} - v_t^T) ∈ R^{d_k × d_v}`, which is consistent with `S ∈ R^{d_k × d_v}`.

2. **Item 2 — DeltaNet expanded form ordering and write term fix (~line 71):** Replaced `S_t = S_{t-1} (I - β_t k̃_t k̃_t^T) + β_t v_t k̃_t^T` with `S_t = (I - β_t k̃_t k̃_t^T) S_{t-1} + β_t k̃_t v_t^T`. The projection now correctly left-multiplies `S_{t-1}` as `(d_k × d_k) · (d_k × d_v) → R^{d_k × d_v}`, and the write term `k̃_t v_t^T ∈ R^{d_k × d_v}` is correctly shaped.

3. **Item 3 — Prose fix for projection description (~line 74):** Updated the explanation bullets to describe left-multiplication: "`(I - β_t k̃_t k̃_t^T) S_{t-1}` left-multiplies S_{t-1} by a rank-1 projection, zeroing the component of each column of S_{t-1} that lies along k̃_t." Also updated the write term reference from `β_t v_t k̃_t^T` to `β_t k̃_t v_t^T` in the prose.

4. **Item 4 — Gated Delta Net teaser write term fix (~line 99):** Replaced `β_t v_t k̃_t^T` with `β_t k̃_t v_t^T` in the forward-reference formula, making the write term `∈ R^{d_k × d_v}` and consistent with `S ∈ R^{d_k × d_v}`.

# B Review — Chapter 1 — Pass 5

1. **`index.md`, line 30 — GLA gate direction still labelled "column-wise" after Pass 3 fix was never applied to this file.**
   The Key Takeaway paragraph reads: "whether the forgetting gate G_t is data-independent (RetNet), **column-wise** data-dependent (GLA), scalar data-dependent (Mamba2)..." Pass 3 corrected `linear_attention_variants_comparison.md` to say "row-wise" but left `index.md` unchanged. The gate `G_t = α_t 1^T ∈ R^{d_k × d_v}` decays row `i` of S by `α_t[i]` uniformly across all d_v columns — it is row-wise (per key-dimension) decay, not column-wise. A reader who reads only the index summary, or who uses the index as a reference, will implement or describe the gate on the wrong axis. Fix: replace "column-wise data-dependent (GLA)" with "row-wise data-dependent (GLA)" in `index.md` line 30.

2. **`linear_attention_variants_comparison.md`, line 68 — delta rule objective stated with a dimensionally invalid matrix product.**
   The prose reads: "S is updated to reduce the squared error between its current prediction **S_{t-1} k̃_t** and the target value v_t." With `S_{t-1} ∈ R^{d_k × d_v}` and `k̃_t ∈ R^{d_k}`, the product `S_{t-1} k̃_t` requires multiplying a (d_k × d_v) matrix by a (d_k × 1) vector — the inner dimensions d_v and d_k do not match unless d_k = d_v, and even then it returns a d_k-dimensional vector rather than a d_v-dimensional value prediction. The correct prediction at key k̃_t is `S_{t-1}^T k̃_t ∈ R^{d_v}` (consistent with the readout `o_t = S_t^T q_t`), or equivalently expressed as the row vector `k̃_t^T S_{t-1} ∈ R^{d_v}` as used in the compact form on line 63. A reader deriving or implementing the DeltaNet objective from this prose description will write an invalid or wrong matrix multiply. Fix: replace `S_{t-1} k̃_t` with `S_{t-1}^T k̃_t` (or equivalently `k̃_t^T S_{t-1}` expressed as a row vector) to match the actual compact form and the established readout convention.

3. **`linear_attention_rnn_equivalence.md`, line 64 — "H_v=32 value heads" is an unsupported number inconsistent with any stated configuration.**
   The sentence reads: "With H_k=16 key heads and H_v=32 value heads (in the interleaved Gated Delta Net layers)..." The only head-count figures established in this chapter are from the Gated Attention GQA configuration: n_q_h=16, n_kv_h=2. No configuration with 32 value heads has been defined anywhere in Chapter 1, and 32 ≠ 16 ≠ 2. A reader computing state memory for the Gated Delta Net layers would use a wrong head count, producing an incorrect memory footprint figure and a wrong understanding of the model architecture. Fix: replace "H_v=32 value heads" with the correct value for the Gated Delta Net configuration, or remove the parenthetical until Chapter 2 establishes the precise interleaved layer head counts.

## Agent A Change Log — Pass 5

Applied all 3 feedback items from Agent B's Pass 5 review:

1. **Item 1 — "column-wise" corrected to "row-wise" in `index.md` (~line 30):** The Key Takeaway paragraph previously read "column-wise data-dependent (GLA)". Changed to "row-wise data-dependent (GLA)". The gate `G_t = α_t 1^T` scales row `i` of S by `α_t[i]` across all d_v columns — it is row-wise (per key-dimension) decay. This mirrors the fix that was applied to `linear_attention_variants_comparison.md` in Pass 3 but was not propagated to `index.md` at that time.

2. **Item 2 — `S_{t-1} k̃_t` corrected to `S_{t-1}^T k̃_t` in `linear_attention_variants_comparison.md` (~line 68):** The DeltaNet objective prose previously stated "S is updated to reduce the squared error between its current prediction S_{t-1} k̃_t and the target value v_t." With `S_{t-1} ∈ R^{d_k × d_v}` and `k̃_t ∈ R^{d_k}`, the product `S_{t-1} k̃_t` is dimensionally invalid. Changed to `S_{t-1}^T k̃_t ∈ R^{d_v}`, which is consistent with the established readout convention `o_t = S_t^T q_t` and with the compact form `k̃_t^T S_{t-1}` used elsewhere in the section.

3. **Item 3 — "H_v=32 value heads" removed from `linear_attention_rnn_equivalence.md` (~line 64):** The sentence "With H_k=16 key heads and H_v=32 value heads (in the interleaved Gated Delta Net layers), state memory is bounded regardless of T." referenced a head count (H_v=32) that is never defined in Chapter 1 and belongs to Chapter 2. Replaced the entire sentence with a per-head memory footprint example (d_k × d_v = 128 × 128 = 16,384 entries × 2 bytes = 32 KB per head) and a forward reference directing the reader to Chapter 2 for specific model head counts.

# B Review — Chapter 1 — Pass 6

1. **`linear_attention_variants_comparison.md`, line 99 — Gated Delta Net teaser gate term has a shape incompatible with elementwise multiplication against S.**
   The formula is `S_t = (g_t · exp(-β_t k̃_t k̃_t^T)) ⊙ S_{t-1} + β_t k̃_t v_t^T`. With `k̃_t ∈ R^{d_k}`, the outer product `k̃_t k̃_t^T ∈ R^{d_k × d_k}`. Whether `exp` is read as the matrix exponential or as elementwise exponentiation of each scalar entry, the result is `∈ R^{d_k × d_k}`. Multiplying by scalar `g_t` leaves it `R^{d_k × d_k}`. But `S_{t-1} ∈ R^{d_k × d_v}`, so `⊙` requires the gate to be `R^{d_k × d_v}`. The shapes are incompatible unless `d_k = d_v`, and even then the semantics are wrong: the gate acts only on the key-axis projection `k̃_t k̃_t^T` but leaves the value axis unaddressed. A reader implementing this forward-reference formula directly will get a shape error (or silent wrong result if d_k=d_v=128 and they interpret elementwise exp naively). The correct gate shape for a GLA-style row-wise decay would be `R^{d_k × d_v}`, typically expressed as an outer product over both axes.

2. **`standard_softmax_attention.md`, line 57 — per-head dimension d_h=256 conflicts with the canonical d_k=d_v=128 used throughout the rest of Chapter 1.**
   The Gated Attention GQA section states "Per-head dimension is d_h=256" and derives KV cache size as `2 × T × 256 × 2 = 1024 × T entries`. The linear attention sections (`linear_attention_rnn_equivalence.md` line 50, key correctness facts) establish `d_k = d_v = 128` with a per-head state of 128 × 128 = 16,384 entries. The chapter frames linear attention as replacing the KV cache with a fixed-size state, directly comparing the two. A reader computing this comparison using d_h=256 for the KV cache baseline against d_k=128 for the linear state is comparing different head dimensions and will get an incorrect relative memory figure. If the two layer types genuinely have different per-head dimensions (256 for Gated Attention, 128 for Gated Delta Net), this must be stated explicitly; as written, the discrepancy is unacknowledged and misleads any reader doing a cross-section numerical comparison.

3. **`linear_attention_rnn_equivalence.md`, lines 58-59 — RNN Recurrence section silently drops the kernel function φ from both the state update and the readout.**
   The State Matrix Formulation section (lines 21-39) carefully defines the state as `S_t = sum φ(k_j) v_j^T` and the readout as `o_t = S_t^T φ(q_t) / z_t^T φ(q_t)`. The RNN Recurrence section then presents the definitive implementation table as `S_t = S_{t-1} + k_t v_t^T` and `o_t = S_t^T q_t`, with no note that φ has been dropped or that the identity kernel is being assumed. A reader implementing from the RNN Recurrence table — which is the action-oriented summary — would omit the kernel transform on keys and queries entirely, implementing vanilla (unkernel-ized) outer-product accumulation. For non-identity kernels (e.g., φ(x) = elu(x)+1), this produces incorrect outputs. The table should either include φ explicitly (`S_t = S_{t-1} + φ(k_t) v_t^T`, `o_t = S_t^T φ(q_t)`) or note that the identity kernel φ(x)=x is assumed in this simplified presentation.

## Agent A Change Log — Pass 6

Applied all 3 feedback items from Agent B's Pass 6 review:

1. **Item 1 — Gated Delta Net teaser formula gate fixed to scalar (`linear_attention_variants_comparison.md`, ~line 99):** Replaced the shape-incompatible formula `S_t = (g_t · exp(-β_t k̃_t k̃_t^T)) ⊙ S_{t-1} + β_t k̃_t v_t^T` with the correct form `S_t = g_t · S_{t-1} + β_t k̃_t v_t^T − β_t k̃_t (k̃_t^T S_{t-1})`. The gate is now clearly `g_t ∈ (0,1]` — a single scalar — which can legitimately multiply `S_{t-1} ∈ R^{d_k × d_v}`. Added inline shape annotation explaining why each term lives in `R^{d_k × d_v}`, so a reader implementing the forward-reference formula will not encounter a shape error.

2. **Item 2 — d_h=256 vs d_k=128 dimension discrepancy made explicit (`standard_softmax_attention.md`, ~line 57):** Added a blockquote note immediately after the KV cache calculation that calls out d_h=256 (Gated Attention) versus d_k=128 (DeltaNet / vanilla linear attention). The note warns readers not to divide the KV cache figure by the linear attention state size to derive a relative memory comparison, because the head dimensions differ. Cross-variant numerical comparisons are explicitly deferred to Chapter 2 where both architectures are fully specified.

3. **Item 3 — Identity-kernel assumption noted in RNN Recurrence section (`linear_attention_rnn_equivalence.md`, ~lines 58-59):** Added a blockquote note immediately after the recurrence code block (`S_t = S_{t-1} + k_t v_t^T`, `o_t = S_t^T q_t`) stating that these bare forms assume the identity kernel (φ = identity) or L2-normalization, and that many modern linear attention variants — including DeltaNet — use exactly this assumption, making the forms valid for those variants. Readers implementing a general linear attention with a nontrivial φ are thus alerted to re-introduce the kernel transform.

# B Review — Chapter 1 — Pass 7

1. **`linear_attention_rnn_equivalence.md`, RNN Recurrence section (~lines 55–62) — normalization denominator z_t omitted from the implementation table.**
   The State Matrix Formulation section (lines 32–39) defines the normalization accumulator `z_t = z_{t-1} + φ(k_t) ∈ R^{d_k}` and the normalized output `o_t = S_t^T φ(q_t) / z_t^T φ(q_t)`. The RNN Recurrence section then presents the definitive two-line summary `S_t = S_{t-1} + k_t v_t^T` and `o_t = S_t^T q_t` with no mention of z_t and no division in the readout. A reader implementing directly from the RNN table will produce unnormalized outputs whose magnitude grows with sequence length T (since S accumulates T outer products). This is a functional correctness error: the denominator normalization is not optional in vanilla linear attention and its omission produces a quantitatively wrong result for any sequence longer than a single token. Fix: either add z_t to the symbol table and include `o_t = S_t^T q_t / (z_t^T q_t)` as the readout, or add an explicit note directly in the recurrence block stating that the denominator normalization is omitted here and referring back to the full form in the State Matrix Formulation section above.

## Agent A Change Log — Pass 7

Applied feedback Item 1 from Agent B's Pass 7 review to `linear_attention_rnn_equivalence.md`:

1. **Item 1 — Normalization denominator z_t added to RNN Recurrence section:** Added `z_t` (normalization accumulator) to the symbol table in the RNN Recurrence section. Expanded the two-line recurrence block from `S_t = S_{t-1} + k_t v_t^T` / `o_t = S_t^T q_t` to a three-line block including `z_t = z_{t-1} + k_t` and the normalized readout `o_t = S_t^T q_t / (z_t^T q_t)`. Added a second blockquote note ("Denominator note") explaining that the denominator prevents output magnitude from growing with T, and that DeltaNet omits the denominator in practice because L2-normalized keys and queries make `z_t^T q_t` approximately uniform — but that the general linear attention form requires it.

# B Review — Chapter 1 — Pass 8

No feedback — chapter approved.

# B Review — Chapter 1 — Pass 9

1. **`standard_softmax_attention.md`, Complexity table — prefill memory column omits the O(T²) cost of materializing the attention matrix.**
   The table lists prefill memory as "O(T · d_k) KV cache". During prefill the full T×T logit/attention matrix must be materialized in SRAM or HBM before the softmax and the weighted sum with V can be computed. That matrix costs O(T²) memory — the dominant term for large T — which exceeds the O(T · d_k) KV cache when T > d_k (e.g., T=8192, d_k=128: the attention matrix is 64× larger than the KV cache). The prose below the table says only "The O(T²) prefill cost comes from materializing the full T×T attention matrix" but attributes it to FLOPs, not memory. A reader planning memory budgets for prefill will underestimate peak memory by up to O(T/d_k) and will not understand why flash-attention-style tiling is necessary to reduce the memory footprint. Fix: add a second row or footnote to the memory column noting peak activation memory during prefill is O(T²), distinct from the O(T · d_k) KV cache that is stored between steps.

2. **`linear_attention_rnn_equivalence.md`, lines 32–33 — the normalization accumulator in the State Matrix Formulation uses φ(k_t) but the RNN Recurrence table silently drops it to bare k_t with an inconsistent result.**
   Line 33 defines `z_t = z_{t-1} + φ(k_t) ∈ R^{d_k}` as the canonical accumulator. The RNN Recurrence block (line 60) then writes `z_t = z_{t-1} + k_t` (bare k_t, no φ). The kernel note covers the state update and readout but does not explicitly say it also applies to the normalization accumulator. A reader implementing the normalization accumulator from the State Matrix section (the only place the denominator is derived) uses φ(k_t); one implementing from the RNN table uses bare k_t. For a non-identity kernel these produce different denominators and hence different output magnitudes. Even under the identity kernel the inconsistency forces the reader to reconcile two formulas that look different without explanation. Fix: add φ(·) to the z_t line in the RNN recurrence block (matching line 33), or extend the existing kernel note to explicitly state it applies to z_t as well.

## Agent A Change Log — Pass 9

Applied both feedback items from Agent B's Pass 9 review:

1. **Item 1 — Prefill peak memory O(T²) note added (`standard_softmax_attention.md`, Complexity table):** Updated the prefill Memory cell from "O(T · d_k) KV cache" to "O(T · d_k) KV cache + O(T²) peak activation (see note)". Added a blockquote note below the table explaining that the O(T · d_k) figure is only the persistent KV cache; during prefill the full T×T logit matrix must also be materialized, costing O(T²) peak activation memory. The note quantifies the gap (T=8192, d_k=128 → attention matrix is 64× larger than the KV cache) and explains that this is precisely why FlashAttention-style tiling is needed — it avoids ever allocating the full O(T²) buffer in HBM by processing the matrix in SRAM-resident blocks.

2. **Item 2 — Kernel note in RNN Recurrence section extended to cover z_t (`linear_attention_rnn_equivalence.md`, kernel note):** Extended the existing kernel note to explicitly include `z_t = z_{t-1} + k_t` in the list of bare forms that assume the identity kernel (φ = identity). The note now states that this bare form is the identity-kernel specialization of the canonical `z_t = z_{t-1} + φ(k_t)` from the State Matrix Formulation section, and instructs readers using a non-identity kernel to substitute φ(k_t) for k_t in the normalization accumulator as well as in the state update and readout. Both sections are now consistent: the State Matrix Formulation uses φ(k_t) as the canonical form, and the RNN Recurrence section uses bare k_t with an explicit note that this assumes φ = identity throughout — including z_t.

# B Review — Chapter 1 — Pass 10

1. **`linear_attention_variants_comparison.md`, Mamba2 section (~line 52) — "softplus" does not produce a value in (0, 1); an implementer following this description will produce a gate γ_t > 1 that causes state explosion.**
   The file states: "γ_t ∈ (0, 1) is computed from the input at each step (e.g., a learned linear projection followed by softplus)." Softplus(x) = log(1 + exp(x)) outputs values in (0, ∞), so a raw softplus output is not constrained to (0, 1). An implementer reading this and writing `gamma_t = softplus(linear(x))` will produce γ_t > 1 for many inputs, causing `S_t = γ_t · S_{t-1} + ...` to amplify rather than decay the state — the exact opposite of the intended behavior. The actual Mamba2 parameterization uses something like `γ_t = exp(-softplus(dt))` (equivalently, sigmoid of a projection in some formulations), which maps to (0, 1). Fix: replace "followed by softplus" with the correct activation that maps to (0, 1), e.g., "followed by sigmoid" or "as `exp(-softplus(·))`", and add a brief note that the constraint γ_t ∈ (0, 1) requires the output to be in the unit interval.

## Agent A Change Log — Pass 10

Applied feedback Item 1 from Agent B's Pass 10 review to `linear_attention_variants_comparison.md`:

1. **Item 1 — Mamba2 γ_t computation corrected from "softplus" to `exp(-softplus(a_t))` (~line 52):** Replaced the incorrect description "a learned linear projection followed by softplus" with the correct Mamba2 parameterization `γ_t = exp(-softplus(a_t))`, where `a_t` is a learned linear projection of the input. The range claim was also corrected from `γ_t ∈ (0, 1)` to `γ_t ∈ (0, 1]` — since softplus outputs values ≥ 0, negating and exponentiating maps to (0, 1], ensuring γ_t is always a valid decay factor and never an amplifier. Added prose explanation that this matches the Mamba2 paper's parameterization of A in log-space as a negative quantity, subsequently exponentiated.

# B Review — Chapter 1 — Pass 11

1. **`linear_attention_rnn_equivalence.md`, State Matrix Formulation section (~lines 21–27) — the state recurrence silently drops φ relative to the sum definition directly above it.**
   Line 21 defines the state as `S_t = sum_{j<=t} φ(k_j) v_j^T`, explicitly using the kernel function φ on the keys. Line 27 then presents the recurrence as `S_t = S_{t-1} + k_t v_t^T` — bare `k_t`, no φ — with no local explanation of the switch. The kernel note that explains the identity-kernel assumption appears only later in the RNN Recurrence section (~line 64), not here. A reader implementing directly from the State Matrix Formulation section — the natural place to derive the recurrence from first principles — sees the sum use `φ(k_j)` and the recurrence drop it with no indication of why. They would either (a) implement `S_t = S_{t-1} + φ(k_t) v_t^T` (following the sum definition) or (b) implement `S_t = S_{t-1} + k_t v_t^T` (following the recurrence), producing different behavior for any non-identity kernel. Fix: add a brief inline note on line 27 that the recurrence is written under the identity kernel (φ(x) = x), and refer the reader to the kernel note in the RNN Recurrence section for the general φ substitution rule.

## Agent A Change Log — Pass 11

Applied feedback Item 1 from Agent B's Pass 11 review to `linear_attention_rnn_equivalence.md`:

1. **Item 1 — Inline identity-kernel note added to State Matrix Formulation section (~line 27):** Added a single parenthetical sentence immediately after the recurrence `S_t = S_{t-1} + k_t v_t^T` stating that this form uses the identity kernel φ(x) = x, and pointing the reader to the Kernel note in the RNN Recurrence section for the general φ substitution rule. This resolves the internal contradiction in the State Matrix Formulation section where the sum definition uses φ(k_j) but the recurrence silently drops it, without duplicating the full kernel explanation that already exists later in the file.

# B Review — Chapter 1 — Pass 12

1. **`linear_attention_variants_comparison.md`, RetNet section (~lines 18–19) — `G_t = γ · I` uses the identity matrix, which produces wrong behavior when applied elementwise to S.**
   The general gated update is defined as `S_t = G_t ⊙ S_{t-1} + k_t v_t^T` with `G_t ∈ R^{d_k × d_v}`. The RetNet section writes `G_t = γ · I`, where `I` is the conventional identity matrix `∈ R^{d_k × d_k}`. Even in the square case d_k = d_v = 128, `γ · I` has γ on the diagonal and 0 elsewhere. Applying `(γ · I) ⊙ S_{t-1}` elementwise zeroes every off-diagonal entry of S at every step and retains only the diagonal entries scaled by γ — destroying the vast majority of stored associations each step. The stated intent, confirmed by the parenthetical "equivalently, scalar γ broadcast to all entries," is that every entry of S is scaled by γ, which requires the gate to be an all-ones matrix scaled by γ (i.e., `γ · 1_{d_k × d_v}`), not the identity matrix. A reader implementing `(γ · I) ⊙ S_{t-1}` directly will produce a model whose state collapses to its diagonal on the first step. Fix: replace `G_t = γ · I` with `G_t = γ · 1` (where `1 ∈ R^{d_k × d_v}` denotes the all-ones matrix, or equivalently use the scalar multiplication form `S_t = γ · S_{t-1} + k_t v_t^T` directly, which already appears on the next line and is unambiguous).

## Agent A Change Log — Pass 12

Applied feedback Item 1 from Agent B's Pass 12 review to `linear_attention_variants_comparison.md`:

1. **Item 1 — `G_t = γ · I` corrected to `G_t = γ · 𝟏` in the RetNet section and summary table:** Replaced the identity matrix `I` (which has γ on the diagonal and 0 elsewhere, zeroing all off-diagonal entries of S via elementwise multiply) with `𝟏 ∈ R^{d_k × d_v}` (the all-ones matrix), so that `G_t ⊙ S_{t-1} = (γ · 𝟏) ⊙ S_{t-1} = γ · S_{t-1}` correctly scales every entry of S by scalar γ. The code block parenthetical was updated to "where 𝟏 ∈ R^{d_k × d_v} is the all-ones matrix; equivalently, scalar γ broadcast to all entries". The summary table row for RetNet was updated from `γ · I (fixed scalar)` to `γ · 𝟏 (fixed scalar, all entries)` for consistency.

# B Review — Chapter 1 — Pass 13

No feedback — chapter approved.
