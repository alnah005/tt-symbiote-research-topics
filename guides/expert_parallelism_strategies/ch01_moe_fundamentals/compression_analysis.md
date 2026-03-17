# Change Log — Agent B Feedback Applied

Date: 2026-03-16
Applied by: Agent A

All five mandatory items from Agent B's review were applied verbatim. Details follow.

---

## Item 1 — `qwen35b_config.md`: Attention head count and head_dim removed from constants table

**Action:** Removed the "Number of attention heads" (64) and "Head dimension" (128) rows from the Model-Level Parameters table. These values are inconsistent with H = 7,168 (64 × 128 = 8,192 ≠ 7,168) and cannot be reconciled by rounding. The note at the end of the table was rewritten to explain that both rows were removed pending verification and to direct readers to the official Qwen3 Technical Report for correct values. The prose sentence below the table that referenced "64 query heads" was also updated to remove the unverified head count.

---

## Item 2 — `qwen35b_config.md`: D = 2048 marked unverified; all derived figures marked as unverified placeholders

**Action:** The MoE Layer Parameters table row for D was annotated with "UNVERIFIED; do not use for implementation decisions". The prose sentence in the Dense FFN section that stated "the MoE experts use a narrower intermediate dimension D = 2048" was updated to flag D as unverified. The Per-Expert Parameter Count and Per-Layer Expert Weight Memory subsections were prefixed with a prominent warning block explaining that all numeric results in those sections are unverified placeholders because they depend on D = 2048, and that D = 2048 yields a per-layer expert count of ~11.3B × 80 layers ≈ 902B, which exceeds the entire 35B model by 25×. The formula display lines were updated to show D_unverified with the D = 2048 numeric result bracketed as a placeholder. The FLOPs per expert and total expert FLOPs per token in the Activation Function section were similarly prefixed with an unverified-placeholder warning and reformulated to separate the correct formula from the unverified numeric substitution. The active parameter first-principles bullet using 44M per expert was updated to show P_expert with the ~28.2B result marked as an unverified placeholder.

---

## Item 3 — `qwen35b_config.md`: Pure EP per-device memory bound re-derived from authoritative 35B total only

**Action:** The "Note" block explaining the pure EP per-device memory calculation was rewritten. The revised version: (a) explicitly states the assumption that non-expert parameters ≈ 19.3B is derived by subtraction from the authoritative 35B total, not from per-layer arithmetic; (b) labels the ~42.5 GB figure as contingent on that estimate and defers the exact value to after architectural constants are verified; (c) makes clear that the derivation path is 35B total → subtract non-expert estimates → remainder is expert weights → divide by N and add back non-expert for per-device footprint. The summary paragraph later in the Expert Weight Sizes section that restated the 22.5 GB per-layer figure was also updated to acknowledge that figure depends on unverified D.

---

## Item 4 — `routing_problem.md`: (N-1)/N cross-device fraction clarified to apply per expert slot, with k(N-1)/N total remote sends stated explicitly

**Action:** The opening paragraph of the "From Mathematical Abstraction to Physical Placement" section was edited to explicitly state: "each of the k selected expert slots has a (N-1)/N probability of being remote" and "on average k(N-1)/N = 8 × 7/8 = 7 of the 8 expert slots per token require inter-device sends." This prevents readers from misreading the 87.5% figure as a fraction of tokens rather than a per-slot probability, and makes the absolute count of remote sends per token (7 of 8) explicit.

---

## Item 5 — `moe_architecture.md`: Renormalization sentence added at line 39 for sigmoid routing

**Action:** A sentence was appended to the softmax normalization paragraph at the location where sigmoid routing is introduced. The added sentence states that the renormalization step w-hat_i = p_i / sum_{j in I} p_j is applied regardless of whether raw probabilities come from softmax or sigmoid, ensuring the weighted combination is always a convex combination of expert outputs. This closes the gap that would allow an implementor to skip renormalization for sigmoid-routed variants.

---

## Pass 2 Change Log — Agent B Feedback Applied

Date: 2026-03-16
Applied by: Agent A

### Pass 2 Item 1 — `qwen35b_config.md`: D_dense = 18,944 marked UNVERIFIED with propagation note

**Action:** The Dense FFN Layer Parameters table entry for the intermediate dimension was annotated with "UNVERIFIED; do not use for implementation decisions". A new warning block was added immediately after the table explaining that D_dense = 18,944 has not been confirmed against the Qwen3 Technical Report, and that all figures derived from it — the dense FFN parameter count (~5.7B), the 19.3B non-expert parameter sum, and the ~42.5 GB per-device EP memory figure — are unverified placeholders that must not be used for implementation decisions until sourced from the Qwen3 Technical Report. This brings D_dense into symmetric treatment with D (the per-expert intermediate dimension), which was already marked UNVERIFIED in Pass 1.

---

### Pass 2 Item 2 — `qwen35b_config.md`: "every device" all-to-all density claim corrected to N-1=7 remote devices

**Action:** The sentence "This means every token participates in cross-device communication to every device" was replaced. The revised text states that under uniform routing one expert slot per token is expected to land on the local device (requiring no cross-device send), while the remaining k-1 = N-1 = 7 slots require sends to 7 remote devices. The expected fan-out is therefore N-1 = 7 remote devices, not N = 8. The surrounding context was updated to preserve the conclusion that the traffic pattern is nearly maximally dense, while correcting the arithmetic.

---

### Pass 2 Item 3 — `routing_problem.md`: "~245 MB" corrected to "~245 MiB"

**Action:** Both occurrences of "~245 MB" on the line describing per-forward-pass dispatch and combine traffic were changed to "~245 MiB", and "approximately 490 MB" was changed to "approximately 490 MiB". The raw byte count (3,211,264 bytes per layer × 80 layers = 256,901,120 bytes) equals 245 MiB in binary units, not 245 MB in SI units (which would be ~257 MB). Using MiB is consistent with hardware capacity figures in TTNN documentation, which typically use binary units, and avoids a ~5% understatement of the SI figure relevant to bandwidth and buffer-sizing calculations against hardware datasheets.

---

## Pass 3 Change Log — Agent B Feedback Applied

Date: 2026-03-16
Applied by: Agent A

### Pass 3 Item 1 — `qwen35b_config.md`: ~12.4B attention weight estimate added as unverified contributor to the ~42.5 GB per-device EP memory warning block

**Action:** The existing warning block for the UNVERIFIED PLACEHOLDER D_dense = 18,944 (which already flagged the ~42.5 GB per-device EP memory figure as contingent on D_dense being unverified) was extended to also name the ~12.4B attention weight estimate as an additional unverified contributor to that figure. The extension explains that the ~12.4B estimate depends on (query head count × head_dim) = H = 7,168, but that the query head count and head dimension were removed from the constants table as unverified because the candidate values (64 × 128 = 8,192 ≠ 7,168) are internally inconsistent and cannot be reconciled. The note warns that a reader who independently verifies D_dense and then trusts the remainder of the ~42.5 GB sum will silently inherit the unverified ~12.4B attention weight input. The correct query head count and head dimension must be sourced from the Qwen3 Technical Report before either the attention weight estimate or the ~42.5 GB figure can be treated as verified.

---

# Compression Analysis: MoE Fundamentals and the Routing Problem — Pass 1

## Summary
- Total files analyzed: 4 (`index.md`, `moe_architecture.md`, `routing_problem.md`, `qwen35b_config.md`)
- Estimated current line count: ~750 lines
- Estimated post-compression line count: ~610 lines
- Estimated reduction: ~18%

---

## CRUCIAL Suggestions

### C1 — `moe_architecture.md` lines 51 and 115: Renormalization convexity stated twice in the same section

The property "ensures that $\sum_{i \in I} \hat{w}_i = 1$, so the weighted combination is a convex combination of expert outputs" appears at line 51 (end of the Top-$k$ Selection subsection) and is restated almost identically in the prose of the Softmax Normalization subsection at line 39: "the weighted combination $y = \sum_{i \in I} \hat{w}_i \cdot o_i$ is always a convex combination of expert outputs." One of these two statements is redundant. The line-39 sentence is the more precise placement (it explains the invariant across both softmax and sigmoid routing); the line-51 sentence can be cut to a single short note or deleted, saving ~2 lines.

### C2 — `routing_problem.md` lines 20–27: $\mathbb{E}[k_\text{remote}]$ formula and note restated three times in quick succession

The expected cross-device remote sends per token is established once in the opening section (line 7: "on average $k(N-1)/N = 8 \times 7/8 = 7$ of the 8 expert slots"), then stated again in the "Expert Placement and Cross-Device Traffic" subsection at line 19 ("The remaining expected $k(N-1)/N = 8 \times 7/8 = 7$ require inter-device communication"), and then a boxed note at line 22–23 re-explains the same formula ("$\mathbb{E}[k_\text{remote}] = k(N-1)/N$ is the general formula"), followed by the inline formula at line 27 which again writes $k(N-1)/N = 7$. The opening paragraph and the detailed subsection are the right places; the Note block at lines 22–23 adds nothing that is not already present and can be deleted (~4 lines).

### C3 — `routing_problem.md` lines 107–119 and `index.md` line 57: Expert capacity / capacity factor explained in both files

`routing_problem.md` lines 107–108 define expert capacity and the capacity factor CF in prose ("each expert is allocated a fixed-size buffer … parameterized by a capacity factor $CF \geq 1.0$, which scales the expected average token load … a higher $CF$ provides more buffer … at the cost of wasted compute on empty slots and wasted communication bandwidth"). `index.md` line 57 in the notation table carries a near-identical explanation inside the $CF$ row: "controls the buffer size allocated per expert by scaling the expected average token load. A higher $CF$ tolerates more load imbalance but increases memory usage (larger dispatch/combine buffers) and communication overhead (padding in those buffers)." The notation table entry for $CF$ should be trimmed to one sentence; the full prose explanation belongs only in `routing_problem.md`. This saves approximately 2–3 lines in `index.md`.

### C4 — `qwen35b_config.md` lines 133 and 139: The same EP memory infeasibility conclusion stated twice in immediate succession

Lines 133 (end of the large Note block) and 139 (the summary paragraph that immediately follows) both conclude, in nearly identical language, that pure EP alone is insufficient for Qwen3.5-35B on T3K and that combined EP + TP or quantization is required. Line 133 ends: "This strengthens the conclusion that combined EP + TP or quantization is required." Line 139 then says: "Under pure expert parallelism, the ~42.5 GB per-device BF16 footprint … significantly exceeds typical T3K per-device DRAM capacity, confirming that pure EP alone is insufficient — combined EP + TP or quantization is required for Qwen3.5-35B on T3K." The second statement is a full sentence re-run of the first. One of the two should be cut; the summary paragraph at line 139 is the better placement because it also states the guide's BF16 default convention. The tail of the Note block at line 133 can end after "The target configuration for this guide assumes some form of combined EP + TP or quantization; see Chapter 8." (~3 lines saved).

### C5 — `qwen35b_config.md` "Active vs. Total Parameters" section: The 46.3B impossibility is explained four separate times

The fact that the first-principles active-parameter sum (~46.3B) exceeds the total model parameter count (~35B) is stated and re-explained at:
1. Lines 61–62 in the VERIFICATION WARNING block ("The first-principles active parameter sum (~46.3B … exceeds the total model parameter count (~35B). Active parameters for a token cannot exceed the model's total parameters. This is arithmetically impossible.")
2. Line 175 ("**Note:** The following first-principles derivation … does not reconcile with the manufacturer's 22B figure … the 46.3B figure … exceeds the total model parameter count of ~35B, which is impossible.")
3. Lines 177 prose ("This first-principles sum yields approximately $12.4 + 5.7 + 28.2 \approx 46.3\text{B}$. 46.3B exceeds the total model size of ~35B, which is impossible — active parameters for a token cannot exceed the model's total parameters.")

The warning block (item 1) is the authoritative location. Items 2 and 3 are redundant re-explanations of the same impossibility in the same section. The Note at line 175 can be compressed to one sentence pointing to the warning block; the inline prose repetition at line 177 ("46.3B exceeds the total model size of ~35B, which is impossible — active parameters for a token cannot exceed the model's total parameters.") can be cut from the prose paragraph (the computation already shows the number; calling it impossible adds nothing after the warning block has already done so). Estimated savings: ~5 lines.

---

## MINOR Suggestions

### M1 — `index.md` line 5: Opening paragraph restates the chapter title

"This chapter establishes the conceptual and mathematical foundation for the rest of the guide." The chapter title already says "MoE Fundamentals and the Routing Problem." The phrase "conceptual and mathematical foundation" can be trimmed to simply "foundation," or the sentence can start directly with the stronger clause ("It introduces the Mixture-of-Experts (MoE) architecture …").

### M2 — `moe_architecture.md` line 7: "the dominant approach" is hedging amplification

"has become the dominant approach for scaling language models to very large parameter counts without proportionally increasing the computation per token." The clause "without proportionally increasing the computation per token" is already captured by the surrounding sentences explaining FLOP sparsity. It can be cut from this sentence to keep the sentence tight.

### M3 — `moe_architecture.md` lines 115 and 119: Dense MoE section contains a self-evident parenthetical

Line 115: "Dense MoE (where all experts run on every token) is rarely used at scale, as it eliminates the compute-efficiency benefit of sparse routing." The parenthetical "(where all experts run on every token)" merely restates the section heading "Dense MoE (All Experts Active)" and the formula on line 111. It can be cut.

### M4 — `routing_problem.md` line 9: "This section explains why … and how … and how" is a three-clause preview immediately after a two-sentence introduction

Lines 5–9 form an introduction paragraph whose last sentence ("This section explains why this overhead exists, how it compounds with load imbalance, and how the dispatch/combine communication pattern addresses it.") previews the rest of the file. This is reasonable for a long document, but the three clauses can be tightened: "why this overhead exists" is already stated in the first two sentences; the sentence can be cut to "This section explains how load imbalance compounds the overhead and how dispatch/combine addresses it."

### M5 — `qwen35b_config.md` line 5–7: "Purpose of This File" introductory sentences

"Abstract discussions of MoE systems become actionable only when grounded in specific numbers." This is a general truism that does not add information about this file specifically. It can be cut without loss.

### M6 — `qwen35b_config.md` lines 204 footnote: Two-sentence note on negligible non-linear ops is verbose

The parenthetical "element-wise SiLU and gating operations ≈ D per token, roughly 43,000× fewer FLOPs than the 6HD matmul cost per token — negligible in practice" is a correct aside, but the ratio 43,000× is a derived figure that will need updating once D is verified anyway. The note can be trimmed to "(element-wise SiLU and gating are negligible compared to the three matrix multiplies)".

---

## Load-Bearing Evidence

Not required — VERDICT is "Crucial updates: yes."

---

## VERDICT
- Crucial updates: yes

---

## Agent C Compression Pass — Change Log

Date: 2026-03-17
Applied by: Agent A

All five CRUCIAL items from Agent C were applied. Details follow.

### C1 — `moe_architecture.md` line 51: Renormalization convexity restatement removed

The sentence "This renormalization ensures that $\sum_{i \in I} \hat{w}_i = 1$, so the weighted combination is a convex combination of expert outputs." at line 51 (Top-k Selection subsection) was replaced with a short pointer: "This renormalization ensures that $\sum_{i \in I} \hat{w}_i = 1$ (see convexity note at line 39)." The authoritative statement covering both softmax and sigmoid routing remains at line 39 of the Softmax Normalization subsection.

### C2 — `routing_problem.md` lines 22–23: Boxed Note block for $\mathbb{E}[k_\text{remote}]$ deleted

The blockquote Note "**Note:** $\mathbb{E}[k_\text{remote}] = k(N-1)/N$ is the general formula under uniform routing. It simplifies to $k - 1$ only because $k = N = 8$ for Qwen3.5-35B; in general, $k(N-1)/N \neq k-1$." was deleted. The formula is already established in the opening paragraph (line 7) and the Expert Placement subsection (line 19); the Note added no new information.

### C3 — `index.md` line 57: CF notation table entry trimmed to one sentence

The CF row in the notation table was reduced from a multi-sentence explanation to: "Capacity factor; scales the per-expert buffer size relative to the average expected token load. Formally defined in Chapter 7, `capacity_factor_mechanics.md`." The full prose explanation of CF mechanics remains in `routing_problem.md`.

### C4 — `qwen35b_config.md` line 133: Duplicate EP infeasibility conclusion removed from Note block

The sentence "This strengthens the conclusion that combined EP + TP or quantization is required." was deleted from the end of the Note block. The Note block now ends with the pointer to Chapter 8 for the target configuration. The full infeasibility conclusion is stated once in the summary paragraph at line 139.

### C5 — `qwen35b_config.md` lines 175 and 177: 46.3B impossibility repetition compressed

The Note at line 175 was compressed to one sentence pointing to the VERIFICATION WARNING block as the authoritative explanation. The inline prose at line 177 ("46.3B exceeds the total model size of ~35B, which is impossible — active parameters for a token cannot exceed the model's total parameters.") was deleted from the prose paragraph; the computation already displays the 46.3B figure and the impossibility is covered authoritatively in the warning block.

---

# Compression Analysis: MoE Fundamentals and the Routing Problem — Pass 2

## Summary

- Files re-examined: 4 (`index.md`, `moe_architecture.md`, `routing_problem.md`, `qwen35b_config.md`)
- Prior CRUCIAL items re-checked: 5
- Items confirmed resolved: 2 (C1, C3)
- Items still unresolved: 3 (C2, C4, C5)

---

## CRUCIAL Suggestions

### C2 — `routing_problem.md` lines 21–25: $\mathbb{E}[k_\text{remote}] = k(N-1)/N = 7$ still stated three times — NOT RESOLVED

The change log (Agent C pass) states the blockquote Note block was deleted. That deletion did occur: there is no blockquote Note in lines 20–27. However, the three inline restatements of the same formula remain, and the redundancy identified in Pass 1 is still present.

The formula $k(N-1)/N = 7$ appears in all three of the following locations within a 5-line span:

- Line 21: "The formula below states the expected value under uniform routing … where $\mathbb{E}[k_\text{remote}] = k(N-1)/N = 7$"
- Line 23 (display formula): "$\mathbb{E}[\text{cross-device volume per token}] \approx \mathbb{E}[k_\text{remote}] \times H \times \text{dtype\_bytes}$" followed immediately by the inline restatement "where $\mathbb{E}[k_\text{remote}] = k(N-1)/N = 7$ under uniform routing"
- Line 25: "Scaling to a batch of $B$ tokens, the expected total cross-device volume is $B \times k(N-1)/N \times H \times \text{dtype\_bytes}$ (which evaluates to $B \times 7 \times H \times \text{dtype\_bytes}$ for Qwen3.5-35B)"

The formula display block (line 23) is the correct load-bearing placement — it introduces the volume formula and names the quantity $\mathbb{E}[k_\text{remote}]$. The restatement at line 21 ("where $\mathbb{E}[k_\text{remote}] = k(N-1)/N = 7$" appended to the condition clause) and the repeated numeric substitution at line 25 ("which evaluates to $B \times 7 \times \ldots$") add nothing after the formula block has already done both. Fix: remove the "$= k(N-1)/N = 7$" tail from the condition clause at line 21, and remove the parenthetical "which evaluates to $B \times 7 \times H \times \text{dtype\_bytes}$ for Qwen3.5-35B" from line 25 (readers can substitute directly from the formula on line 23). Estimated savings: ~2–3 lines or ~30 words of inline prose.

### C4 — `qwen35b_config.md` lines 133 and 139: EP memory infeasibility conclusion still stated twice — NOT RESOLVED

The change log states the sentence "This strengthens the conclusion that combined EP + TP or quantization is required." was deleted from the end of the Note block. That specific sentence no longer appears. However, the infeasibility conclusion was not eliminated — it was replaced by different language that makes the same point. The Note block at line 133 now ends: "It substantially exceeds typical T3K per-device DRAM capacity and makes the memory-infeasibility concern evident. The target configuration for this guide assumes some form of combined EP + TP or quantization; see Chapter 8, `ch08_qwen35b_t3k_strategy/recommended_configuration.md` for the reference configuration."

Line 139 then states: "Under pure expert parallelism, the ~42.5 GB per-device BF16 footprint (derived from the authoritative 35B total — see note above; exact figure deferred until non-expert parameter estimates are verified) significantly exceeds typical T3K per-device DRAM capacity, confirming that pure EP alone is insufficient — combined EP + TP or quantization is required for Qwen3.5-35B on T3K."

Both sentences assert (a) the ~42.5 GB figure substantially/significantly exceeds T3K DRAM, and (b) combined EP + TP or quantization is required. The duplication is complete. Fix: delete "It substantially exceeds typical T3K per-device DRAM capacity and makes the memory-infeasibility concern evident." from the end of the Note block at line 133. The Note block should end after the Chapter 8 pointer, leaving the full infeasibility statement to line 139 where it belongs (the summary paragraph also carries the BF16/INT8 default convention, making it the better permanent location).

### C5 — `qwen35b_config.md` line 177: Candidate-explanation prose still substantially duplicates the warning block — PARTIALLY RESOLVED, residual redundancy remains

The change log states the specific phrase "46.3B exceeds the total model size of ~35B, which is impossible — active parameters for a token cannot exceed the model's total parameters." was deleted from line 177. That phrase is gone. The Note at line 175 is now a single pointer sentence. This partial fix resolves the most egregious restatement.

However, line 177 remains a very long sentence (approximately 120 words) that contains material substantially overlapping the warning block. Specifically:

- The warning block at line 61 already states: "No valid combination of these figures can simultaneously satisfy all three: the 35B total, the per-expert dimension of $D = 2048$, and the 80-layer, 256-expert configuration."
- Line 177 then adds: "One candidate explanation is that the 22B designation counts only parameters accessed per inference token in a weight-streaming model … under a different convention for shared embedding and layer-norm parameters. The discrepancy may reflect a different convention … but the more likely root cause is that the per-expert intermediate dimension $D$ used in this calculation is incorrect (see verification warning)."

The warning block already identifies $D$ as the primary suspect (line 58: "The primary suspect is $D$"). The candidate explanations in line 177 (weight-streaming model, shared embedding convention, normalization across expert pool) are speculative prose that go beyond the scope of a content verification file and are not referenced anywhere else. The only load-bearing information in line 177 is the numeric result ($12.4 + 5.7 + 28.2 \approx 46.3\text{B}$), the pointer to the 22B authoritative figure, and the gap (~24.3B). Fix: trim line 177 to three sentences: the arithmetic result, the gap to 22B, and a pointer to the Qwen3 Technical Report for the authoritative explanation. Delete the speculative candidate-explanation list. Estimated savings: ~50–60 words.

---

## MINOR Suggestions

### M1 — `routing_problem.md` line 25: Redundant parenthetical in the batch-volume sentence

The parenthetical "(which evaluates to $B \times 7 \times H \times \text{dtype\_bytes}$ for Qwen3.5-35B)" at the end of line 25 restates a substitution already shown in the numeric example on line 27. Remove the parenthetical; readers can substitute $k(N-1)/N = 7$ themselves from the formula on line 23 or from the computed example three lines later.

### M2 — `moe_architecture.md` line 149: "This means the model's effective computation footprint per token is similar to a 22B dense model" is a restatement

The sentence "This means the model's effective computation footprint per token is similar to a 22B dense model, while its total parameter capacity is roughly 35B" is a paraphrase of the numbers just given (active ~22B, total ~35B). The numbers themselves make this point; the interpretive sentence can be cut or collapsed into the preceding sentence.

### M3 — `qwen35b_config.md` line 256: Summary table note is redundant with line 7 of `routing_problem.md`

The note beneath the summary table at line 256 — "Note: The cross-device fraction $(N-1)/N$ holds for any value of $k$ under uniform routing — it is $k$-independent. The expected **count** of remote experts per token, $k(N-1)/N$, does depend on $k$. See `routing_problem.md` for the full derivation." — is a clarification already visible from the opening of `routing_problem.md`. This note adds value for readers who skip `routing_problem.md` but is otherwise a minor duplication. Consider cutting the last two sentences and keeping only "Note: The cross-device fraction $(N-1)/N$ is $k$-independent; the expected remote-expert count $k(N-1)/N$ depends on $k$."

---

## Load-Bearing Evidence

- **`moe_architecture.md` line 39:** The sentence "Regardless of whether raw probabilities come from softmax or sigmoid, the renormalization step $\hat{w}_i = p_i / \sum_{j \in I} p_j$ (applied over the selected top-$k$ index set $I$) is always performed, so the weighted combination $y = \sum_{i \in I} \hat{w}_i \cdot o_i$ is always a convex combination of expert outputs." This is the sole authoritative statement covering both routing variants. It cannot be cut; it is the load-bearing anchor that makes the pointer at line 51 valid.

- **`routing_problem.md` lines 104–106:** "Expert capacity is the concept used to bound this: each expert is allocated a fixed-size buffer that can hold at most $C$ tokens per forward pass. The capacity $C$ is parameterized by a capacity factor $CF \geq 1.0$, which scales the expected average token load per expert…" This is the first definition of $C$ and $CF$ in the chapter and cannot be cut; it is the antecedent for every downstream reference to capacity in Chapters 2–7.

- **`qwen35b_config.md` lines 53–67 (VERIFICATION WARNING block):** The four-point contradiction summary is load-bearing infrastructure: it is the single source of truth explaining why all derived numerics are unreliable. Every unverified-placeholder annotation elsewhere in the file points back to this block. It cannot be compressed without destroying that audit trail.

- **`index.md` lines 43–62 (notation table):** The table is the sole chapter-wide definition of symbols ($E$, $k$, $N$, $B$, $H$, $D$, $p_e$, $\hat{w}_i$, $I$, $E_d$, $I_e$, $C$, $CF$, $L$). Every file in the chapter and in subsequent chapters references these symbols. The table cannot be reduced further beyond the CF entry trim applied in C3.

---

## VERDICT

- Crucial updates: yes (items C2, C4, and C5 remain unresolved and require the specific edits described above)

---

## Pass 3 Compression Change Log — Residual CRUCIAL Items Applied

Date: 2026-03-17
Applied by: Agent A

### C2 (residual) — `routing_problem.md`: k(N-1)/N = 7 restatement in condition clause removed

The condition clause at line 21 previously ended with ", where $\mathbb{E}[k_\text{remote}] = k(N-1)/N = 7$", restating the formula a second time immediately before the display formula that carries the load-bearing statement. That tail was deleted; the condition clause now ends after "(condition: experts are selected independently and uniformly at random across devices)". The parenthetical "(which evaluates to $B \times 7 \times H \times \text{dtype\_bytes}$ for Qwen3.5-35B)" at the end of the batch-volume sentence (line 25) was also deleted. The formula $k(N-1)/N = 7$ now appears once in this passage: in the "where" clause beneath the display formula, which is the load-bearing placement.

### C4 (residual) — `qwen35b_config.md`: Infeasibility restatement deleted from Note block tail

The sentence "It substantially exceeds typical T3K per-device DRAM capacity and makes the memory-infeasibility concern evident." was deleted from the end of the Note block at line 133. The Note block now ends with the Chapter 8 pointer sentence. The full infeasibility conclusion remains solely in the summary paragraph at line 139, which is the correct permanent location.

### C5 (residual) — `qwen35b_config.md`: Speculative candidate explanations trimmed from line 177

Approximately 90 words of speculative candidate explanations (weight-streaming model, shared-embedding convention, normalization across the expert pool) were deleted from the prose paragraph at line 177. The paragraph now contains only: the arithmetic result (~46.3B sum), the ~24.3B gap (46.3B − 22B), and a single pointer to the Qwen3 Technical Report for authoritative architectural constants. The warning block at line 58 already names D as the primary suspect; the speculative list was redundant with that block and added no load-bearing information.

---

# Compression Analysis: MoE Fundamentals and the Routing Problem — Pass 3

## Summary

- Files re-examined: 4 (`index.md`, `moe_architecture.md`, `routing_problem.md`, `qwen35b_config.md`)
- Prior CRUCIAL items re-checked: 3 (C2 residual, C4 residual, C5 residual)
- Items confirmed resolved: 3 (all)
- New CRUCIAL items: 0
- VERDICT: Crucial updates: no

---

## Prior CRUCIAL Items — Resolution Status

### C2 residual — `routing_problem.md`: k(N-1)/N = 7 restatements — RESOLVED

The change log states the condition-clause tail `= k(N-1)/N = 7` was deleted from line 21 and the parenthetical `(which evaluates to B × 7 × H × dtype_bytes for Qwen3.5-35B)` was deleted from line 25. Both deletions are confirmed in the current file. The formula `k(N-1)/N = 7` now appears exactly twice in the file: once in the opening paragraph (line 7, first establishment) and once in the "where" clause of the display formula (line 25, load-bearing placement). The subsection at line 19 states `k(N-1)/N = 8 × 7/8 = 7` as the expected remote count — this is the second distinct context (a per-device token-load calculation rather than a per-token probability statement) and is not a redundant restatement. No further action required on C2.

### C4 residual — `qwen35b_config.md`: EP memory infeasibility conclusion duplicated — RESOLVED

The sentence "It substantially exceeds typical T3K per-device DRAM capacity and makes the memory-infeasibility concern evident." is no longer present in the Note block at line 133. The Note block ends with the Chapter 8 pointer. The full infeasibility conclusion appears exactly once, in the summary paragraph at line 139. No further action required on C4.

### C5 residual — `qwen35b_config.md`: Speculative candidate explanations at line 177 — RESOLVED

The ~90 words of speculative candidate explanations (weight-streaming model, shared-embedding convention, normalization across expert pool) are gone. Line 177 now contains only: arithmetic result (~46.3B), gap to 22B (~24.3B), pointer to the Qwen3 Technical Report, and adoption of the 22B authoritative figure. No further action required on C5.

---

## Load-Bearing Evidence

- **`moe_architecture.md` line 39:** "Regardless of whether raw probabilities come from softmax or sigmoid, the renormalization step $\hat{w}_i = p_i / \sum_{j \in I} p_j$ (applied over the selected top-$k$ index set $I$) is always performed, so the weighted combination $y = \sum_{i \in I} \hat{w}_i \cdot o_i$ is always a convex combination of expert outputs." This is the sole statement in the chapter that covers the renormalization invariant for both routing variants. It anchors the pointer at line 51 and the corresponding explanation in `routing_problem.md` line 41. Cannot be cut.

- **`routing_problem.md` lines 104–106:** The first definition of expert capacity $C$ and capacity factor $CF$ in the chapter ("Expert capacity is the concept used to bound this … parameterized by a capacity factor $CF \geq 1.0$, which scales the expected average token load per expert; a higher $CF$ provides more buffer for skewed routing at the cost of wasted compute on empty slots and wasted communication bandwidth"). Every Chapter 7 reference and every dispatch-buffer sizing discussion in Chapters 2–6 traces back to this paragraph. Cannot be cut.

- **`qwen35b_config.md` lines 53–67 (VERIFICATION WARNING block):** The four-point contradiction summary is the single source-of-truth audit trail explaining why all derived numerics in the file are unreliable. All unverified-placeholder annotations elsewhere in the file point to this block. Removing or shortening it would destroy the audit trail and leave individual placeholder warnings without an authoritative anchor.

- **`index.md` lines 43–62 (notation table):** The table is the sole chapter-wide definition point for all shared symbols ($E$, $k$, $N$, $B$, $H$, $D$, $p_e$, $\hat{w}_i$, $I$, $E_d$, $I_e$, $C$, $CF$, $L$). Subsequent chapters reference these symbols without re-defining them. The table cannot be reduced further after the CF entry trim applied in Pass 1 C3.

- **`qwen35b_config.md` lines 122–133 (per-device memory table and EP note):** The table quantifies the BF16 vs INT8 per-device memory gap between full TP sharding (~8.75 GB) and pure EP (~42.5 GB). This is the only place in Chapter 1 where the memory infeasibility of pure EP is given a concrete number. The note deriving the ~42.5 GB figure from 35B − 19.3B cannot be cut; it is the arithmetic basis for the conclusion at line 139 and for Chapter 8's recommended configuration.

---

## MINOR Suggestions

### M1 — `moe_architecture.md` line 149: Interpretive sentence restates the numbers just given

"This means the model's effective computation footprint per token is similar to a 22B dense model, while its total parameter capacity is roughly 35B." This sentence paraphrases the numbers stated two sentences earlier (active ~22B, total ~35B). The numbers themselves already make the comparison; the interpretive gloss adds approximately 25 words without new content. The sentence can be cut or merged into the preceding sentence as a short clause: "…while its total parameter capacity is ~35B (the 'A22B' designation)."

### M2 — `qwen35b_config.md` line 202: "However, this is offset by…" clause restates information from a prior sentence

Line 202: "However, this is offset by the fact that the intermediate dimension $D = 2048$ is smaller than typical dense FFN intermediate dimensions, so the total FLOPs are comparable." The claim that $D$ is smaller than typical dense FFN dimensions is already established at line 49 (Dense FFN section: "while the MoE experts … use a narrower intermediate dimension $D$"). The "offset" observation is a qualitative aside that adds approximately 30 words and is marked as depending on the unverified $D$ anyway. It can be deleted without losing any load-bearing content; the formula on line 206 ($6HD$) stands on its own.

### M3 — `routing_problem.md` line 115: Capacity section ending duplicates its own first sentence

The final sentence of the "What Happens at Overflow" subsection (line 115) reads: "At this point, readers need only understand that expert capacity exists as a bounding mechanism and that exceeding it has consequences for output quality or routing complexity." This almost exactly restates the first sentence of the "Expert Capacity" section (line 104): "Expert capacity is the concept used to bound this: each expert is allocated a fixed-size buffer …" The closing sentence at line 115 is a gentle reminder, but given that the section is only ~12 lines long, the reminder adds no navigational value. It can be cut, saving approximately 1 line.

---

## VERDICT

- Crucial updates: no
