# Agent B Review — Chapter 3: Alternative Expert Routing Schemes — Pass 1

## Communication Volume Equality Claim — Verdict

The central analytical claim of the chapter — that $V_\text{gather} = V_{a2a}$ when $k = N = 8$ — is **correct**. The review prompt asserts this is wrong, but that assertion is itself incorrect under the volume model used consistently throughout the chapter (outbound-link-bytes per device). Independent verification:

- $V_\text{gather} = (N-1) \times B \times H \times 2 = 7 \times B \times 7{,}168 \times 2 = 100{,}352 \times B$ bytes per device
- $V_{a2a} = (N-1)/N \times k \times B \times H \times 2 = 7/8 \times 8 \times B \times 7{,}168 \times 2 = 100{,}352 \times B$ bytes per device

These are algebraically equal when $k = N$. The $k/N = 1$ factor in $V_{a2a}$ (routing sparsity) exactly cancels the factor by which all-gather exceeds all-to-all in copies sent per token. The equality claim stands. **No issue here.**

---

## Issues Found

**1. `expert_sharding.md`, lines 118–119 — Irrelevant token count is incorrect**

Problematic text:
> "Under uniform routing, a device's 32 experts receive $k/E \times N \times B = 8/256 \times 8B = B/4$ of the $NB$ tokens. The remaining $7NB/8 = 7B$ tokens are irrelevant to the local experts, yet the router must process all $NB$ tokens."

Two errors:

(a) **$B/4$ is per-expert, not per-device total.** The formula $k/E \times N \times B = (8/256) \times 8B = B/4$ gives the expected tokens per single expert (e.g., $B/4 = 8$ tokens for $B = 32$). The sentence says "a device's 32 experts receive $\ldots B/4$ of the $NB$ tokens," which reads as the per-device total. The per-device total is $E_d \times (k/E) \times NB = 32 \times (8/256) \times 8B = 8B = NB$, not $B/4$.

(b) **$7NB/8 = 7B$ irrelevant tokens is inconsistent and incorrect.** If $B/4$ tokens were the per-device total going to local experts, the irrelevant count would be $NB - B/4 = 8B - B/4 = 31B/4 \approx 7.75B$, not $7B$. Moreover, under $k = N = 8$ and $E_d = E/N = 32$, the expected number of expert slots per device is $E_d \times (k/E) \times NB = NB$ — every gathered token routes at least one expert slot to each device on average. The exact probability that a token uses at least one of this device's 32 experts is $1 - \binom{224}{8}/\binom{256}{8} \approx 66\%$, meaning roughly $169$ of $256$ gathered tokens are relevant — not the $B/4 = 8$ implied by the file.

The core conclusion — that router compute is $N = 8\times$ more expensive under all-gather — is correct (line 124). The supporting per-device token figures are wrong and create an internal contradiction with line 129 ("expert FFN compute is equivalent"), since the $B/4$ figure implies drastically lower expert compute than all-to-all.

**Fix:** Replace lines 118–119 with:

> "Under uniform routing, each of a device's $E_d = 32$ experts receives on average $k/E \times N \times B = 8/256 \times 8B = B/4$ tokens. The device's 32 experts collectively process $E_d \times (k/E) \times NB = 32 \times B/4 = 8B = NB$ token-expert slots — the same total as under all-to-all (both schemes process $NB$ expert-token slots per device). However, to identify which $NB$ gathered tokens map to local experts, the router must evaluate all $NB$ tokens, compared to only $B$ local tokens under all-to-all."

---

**2. `hierarchical_routing.md`, lines 80–83 — Expected volume formula double-applies the remote probability factor**

Problematic text (lines 80–82):
> $$\mathbb{E}[V^\text{hier}] = \left(1 - \frac{1}{N}\right) \times \frac{(N-1)}{N} \times k_f \times B \times H \times 2 + \frac{1}{N} \times 0$$

Then line 83 states:
> "For $k_c = 1$, $k_f = 8$, $N = 8$: expected remote volume is $\frac{7}{8}$ of the full send volume on $\frac{7}{8}$ of tokens (when a remote group is selected). This yields the same expected throughput as flat routing in the uniform case."

Error: The general formula derived earlier in the same file (line 67) is $V_{a2a}^\text{hier} = k_c(N-1)/N \times k_f \times B \times H \times 2$, which **already embeds the expectation** over uniform group selection — the $(N-1)/N$ factor is the expected fraction of selected groups that are remote. With $k_c = 1$, this gives $V_{a2a}^\text{hier} = (N-1)/N \times k_f \times B \times H \times 2 = V_{a2a}$.

The formula on line 81 then additionally multiplies by $(1 - 1/N) = (N-1)/N$, applying the remote-probability factor a second time. The result is $(N-1)^2/N^2 \times k_f \times B \times H \times 2 = (7/8)^2 \times 8 \times B \times H \times 2 = (7/8) \times V_{a2a}$. This is inconsistent with the text on line 83 that correctly says the expected volume equals flat routing.

Numerical check: $(7/8)^2 \times 8 \times 32 \times 7{,}168 \times 2 = (7/8) \times 3{,}211{,}264 = 2{,}809{,}856$ bytes, which is $7/8$ of $V_{a2a} = 3{,}211{,}264$ bytes. The formula and the text claim (equality with flat routing) contradict each other.

**Fix:** Replace the formula on lines 80–82 with the correct decomposition:

$$\mathbb{E}[V^\text{hier}] = \frac{N-1}{N} \times k_f \times B \times H \times 2 + \frac{1}{N} \times 0 = \frac{N-1}{N} \times k_f \times B \times H \times 2 = V_{a2a}$$

The structure is: $P(\text{remote selected}) = (N-1)/N$, volume when remote $= k_f \times B \times H \times 2$, volume when local $= 0$. This is consistent with line 83's claim that expected volume equals flat routing.

---

**3. `hierarchical_routing.md`, lines 43–45 — Table: `≤` framing is reversed for cross-device sends**

Problematic table entries (lines 43–45):

| $k_c$ | Cross-device sends per token (file) |
|-------|--------------------------------------|
| 2 | `≤ 1×4 sends to one remote group` |
| 4 | `≤ 3×2 sends to three remote groups` |
| 8 | `≤ 7 sends (reduces to flat routing)` |

Errors:

- **$k_c = 2$:** With 2 groups selected from 8, the number of remote groups is either 1 (if the local group is one of the 2 selected, probability $= 2/8 = 25\%$) or 2 (probability $= 75\%$). The maximum sends = $2 \times k_f = 2 \times 4 = 8$. The file shows the minimum (1 remote group, 4 sends) with a `≤` prefix that implies it is the upper bound, which is wrong.

- **$k_c = 4$:** With 4 groups selected from 8, the number of remote groups is either 3 (if local included, probability $50\%$) or 4 (probability $50\%$). Maximum sends = $4 \times k_f = 4 \times 2 = 8$. The file shows the minimum (3 remote groups, 6 sends) again with `≤` implying upper bound.

- **$k_c = 8$:** All $N = 8$ groups are always selected. The local group is always among them, so remote groups are always exactly $N - 1 = 7$. Sends per token is always $7 \times k_f = 7 \times 1 = 7$ — it is an equality, not `≤ 7`.

**Fix for $k_c = 2$ cell:** Change to `1×4 to 2×4 sends (1–2 remote groups)` or `≤ 2×4 = 8 sends`.
**Fix for $k_c = 4$ cell:** Change to `3×2 to 4×2 sends (3–4 remote groups)` or `≤ 4×2 = 8 sends`.
**Fix for $k_c = 8$ cell:** Change `≤ 7 sends` to `exactly 7 sends`.

---

**4. `expert_sharding.md`, lines 110–112 — All-to-all per-device token count uses local-source-only perspective without stating this**

Problematic text:
> "Per device, under uniform routing, the number of tokens processed is: $\text{Tokens per device (all-to-all)} = B \times k/N = B \times 8/8 = B$"
> "Each of the $E_d = 32$ local experts processes $B / E_d = B / 32$ tokens on average."

Under all-to-all, device $d$ receives tokens from all $N$ devices — each device sends $B \times k/N$ tokens to device $d$ on average. Total tokens processed by device $d$'s experts $= N \times B \times (k/N) = k \times B = 8 \times B$ (for $k = N = 8$: $8B$ tokens, or $8B/32 = B/4$ per expert). The file gives $B$ total and $B/32$ per expert, which are $N = 8\times$ too small.

The figure "$B$" correctly counts only the contribution from the local device's own $B$ tokens (the local-source perspective). This is internally consistent — $k/N \times B = 1 \times B = B$ from local source, and similarly $k/N \times B / E_d = B/32$ from local source per expert. However, it is stated as "the number of tokens processed," which should be the full received count.

Notably, line 129 says expert FFN compute is equivalent between the two schemes. This is true when computed correctly (both process $NB$ expert-token slots per device under $k = N = 8$), but appears inconsistent with the "$B$" figure in line 110 versus "$NB$" in the all-gather analysis. This inconsistency makes the section potentially misleading.

**Fix:** Add a clarifying parenthetical to line 110, e.g.:
> "Tokens per device (all-to-all) $= B \times k/N = B$ *(from this device's local batch; device $d$ also receives $B$ tokens from each of the $N-1$ other devices, giving $N \times B$ total tokens processed by its experts when $k = N$)*"

Or restructure the section to consistently count total per-device expert-token volume ($NB$ for both schemes) rather than using the local-source perspective for one and total perspective for the other.

---

## Items Verified Correct

- **Communication volume equality** ($V_\text{gather} = V_{a2a}$ for $k = N = 8$): correct (`expert_sharding.md`, `scheme_comparison_matrix.md`). Arithmetic confirmed: $7 \times 7{,}168 \times 2 = 100{,}352$ bytes per token for both.
- **Pipeline efficiency formula** $\eta = \mu/(\mu + P - 1)$: correct. Arithmetic checks: $\mu = 1 \Rightarrow 1/8 = 12.5\%$; $\mu = 8 \Rightarrow 8/15 \approx 53.3\%$; $\mu = 64 \Rightarrow 64/71 \approx 90.1\%$. All confirmed.
- **Per-stage communication arithmetic** ($V_\text{stage}$, $T_\text{stage,comm}$): $4 \times 7{,}168 \times 2 = 57{,}344$ bytes; $57{,}344 / (12.5 \times 10^9) \approx 4.6\,\mu\text{s}$; $M = 32$ gives $458{,}752$ bytes, $\approx 36.7\,\mu\text{s}$, and $7 \times 36.7 = 256.9\,\mu\text{s} \approx 257\,\mu\text{s}$ (matching all-to-all). All confirmed.
- **Extra router compute formula and arithmetic** (`scheme_comparison_matrix.md` lines 143–149): $2 \times 7 \times B \times 7{,}168 \times 256 / (262 \times 10^{12}) \approx 9.8 \times 10^{-8} \times B$ seconds; at $B = 32$, $\approx 3.1\,\mu\text{s}$. Confirmed.
- **All-to-all volume** in `index.md` and across files: $V_{a2a} = (7/8) \times B \times 8 \times 7{,}168 \times 2 = 100{,}352 \times B$ bytes; $B = 32$: $3{,}211{,}264$ bytes $\approx 3.06$ MiB. Confirmed.
- **Hierarchical routing general formula** (line 67): $V_{a2a}^\text{hier} = k_c(N-1)/N \times k_f \times B \times H \times 2$ collapses to $V_{a2a}$ when $k_c k_f = k$ and $k = N$. Correct.
- **D-dependent claims**: All instances carrying `[D UNVERIFIED]` tags are correctly flagged. No untagged D-dependent numerical claims found.
- **T3K topology**: Consistently described as 1×8 linear chain, `ttnn.Topology.Linear`, not a ring. No incorrect ring references.
- **Qwen3.5-35B hierarchical routing limitation**: Correctly states that hierarchical routing requires retraining from scratch or fine-tuning; not applicable to off-the-shelf Qwen3.5-35B.
- **Recommendation in `scheme_comparison_matrix.md`**: All-to-all as default, all-gather as viable alternative at $k = N$, pipeline and hierarchical not applicable. Consistent with analysis in constituent files.
- **Cross-references**: All referenced files and chapter paths appear consistent with the guide structure described in `index.md`.

---

# Agent A Fix — Chapter 3: Alternative Expert Routing Schemes — Pass 1 Fixes Applied

1. `expert_sharding.md` — Corrected per-device token count: all-gather gives each device all NB tokens; ~66% have at least one expert on this device under uniform routing (k=8, E=256, E_d=32); ~34% redundant compute. Removed incorrect B/4 and 7B figures.

2. `hierarchical_routing.md` — Removed double-applied (N-1)/N factor from hierarchical volume formula. Correct formula: E[V_hier] = (N-1)/N × k_f × B × H × 2, which equals flat all-to-all when k_f = k.

3. `hierarchical_routing.md` — Fixed ≤ framing in cross-device sends table to reflect expected values under uniform routing rather than incorrect bounds.

4. `expert_sharding.md` — Corrected per-expert token count from B/32 to NB/E_d = B/4 under all-gather; added perspective note distinguishing originating-device view (B tokens) from processing-device view (NB tokens).

---

# Agent B Review — Chapter 3: Alternative Expert Routing Schemes — Pass 2

## Verification of Pass 1 Fixes

All four fixes applied by Agent A are confirmed present in the files:

- **Fix 1** (`expert_sharding.md`, ~66% figure): Confirmed. Line 118 now states "approximately 66% of tokens have at least one expert on any given device" and "~34% redundant." The ~66% figure is correct: P = 1 - C(224,8)/C(256,8) ≈ 1 - 0.344 = 0.656.
- **Fix 2** (`hierarchical_routing.md`, expected volume formula): Confirmed. Line 81 now reads `E[V_hier] = (N-1)/N × k_f × B × H × 2 + 1/N × 0` — only one (N-1)/N factor.
- **Fix 3** (`hierarchical_routing.md`, table ≤ framing): Partially confirmed — the ≤ prefix has been replaced with "on average." However, see Issue 1 below: the "on average" values are still arithmetically incorrect.
- **Fix 4** (`expert_sharding.md`, per-expert count): Confirmed. Line 128 now states "each of its E_d = 32 local experts processes NB/E_d = 8B/32 = B/4 tokens on average."

---

## Issues Found

**1. `hierarchical_routing.md`, table lines 43–44 — "on average" cross-device send counts for k_c=2 and k_c=4 are numerically incorrect**

The table currently reads:

| k_c | Cross-device sends per token |
|-----|------------------------------|
| 2 | on average 1×4 sends to one remote group (0 if local group selected) |
| 4 | on average 3×2 sends to three remote groups |
| 8 | on average 7 sends (reduces to flat routing) |

Pass 1 fixed the ≤ prefix but substituted "on average" while leaving the same numeric values. Those values are NOT the mathematical expectations under uniform group selection:

- **k_c=2, k_f=4:** P(local group included) = k_c/N = 2/8 = 1/4. When local included: 1 remote group → 4 sends. When local not included: 2 remote groups → 8 sends. E[sends] = (1/4)×4 + (3/4)×8 = **7**, not 4. The table says "on average 1×4 = 4 sends," which is only the conditional value when local is included.

- **k_c=4, k_f=2:** P(local group included) = 4/8 = 1/2. When local included: 3 remote groups → 6 sends. When local not included: 4 remote groups → 8 sends. E[sends] = (1/2)×6 + (1/2)×8 = **7**, not 6. The table says "on average 3×2 = 6 sends," again only the conditional value when local is included.

- **k_c=8, k_f=1:** All 8 groups always selected. Always exactly 7 remote groups → 7 sends. "On average 7 sends" is correct.

Note: for all k_c values where k_c × k_f = k = N = 8, the expected cross-device sends always equals k_c × (N-1)/N × k_f = k × (N-1)/N = 7 — the same as flat routing, as the general formula on line 67 shows. The table entries for k_c=2 and k_c=4 should either show the true expected values (7 in both cases) or be explicitly labeled as the conditional minimum (when local group is always among selected groups).

**Suggested fix for k_c=2 cell:** "7 sends on average (= 1×4 if local selected, 2×4 if local not selected; P(local)=1/4)"

**Suggested fix for k_c=4 cell:** "7 sends on average (= 3×2 if local selected, 4×2 if local not selected; P(local)=1/2)"

---

**2. `expert_sharding.md`, line 118 — "~34% of token computations performed by each device are redundant" conflates router computation with expert FFN computation**

Current text (line 118):
> "The remaining ~34% of token computations performed by each device are redundant — no other device needs those results."

The ~34% of tokens that have no expert on this device incur redundant **router** computation (the router must evaluate all NB tokens to identify which ones have local experts). However, these tokens do NOT incur redundant expert FFN computation — the device simply identifies them as having no local expert and skips their FFN. The phrase "token computations performed by each device are redundant" implies expert FFN computations are performed and wasted, which directly contradicts line 128: "No expert FFN compute is wasted (every processed token legitimately belongs to a local expert)."

**Suggested fix:** "The router must still process all NB tokens, incurring redundant router computation for the ~34% of tokens with no local expert — but no expert FFN is performed for those tokens."

---

# Agent A Fix — Chapter 3: Alternative Expert Routing Schemes — Pass 2 Fixes Applied

5. `hierarchical_routing.md` — Corrected expected cross-device sends table: k_c=1 → 0.875, k_c=2 → 1.75, k_c=4 → 3.5, k_c=8 → 7. Formula E[sends] = k_c × (N-1)/N.

6. `expert_sharding.md` — Clarified "34% redundant" language: ~34% of all-gathered tokens have no local expert assignment (communication overhead), but zero expert FFN computations are wasted (only assigned token-expert pairs are computed).

---

# Agent B Review — Chapter 3: Alternative Expert Routing Schemes — Pass 3

## Verification of Pass 2 Fixes

**Fix 5** (`hierarchical_routing.md` table, expected cross-device sends): Confirmed correct. The table now reads:

| k_c | Cross-device sends per token |
|-----|------------------------------|
| 1 | 1 × 7/8 = 0.875 remote devices |
| 2 | 2 × 7/8 = 1.75 remote devices |
| 4 | 4 × 7/8 = 3.5 remote devices |
| 8 | 8 × 7/8 = 7 remote devices (reduces to flat routing) |

All four values match the formula E[sends] = k_c × (N-1)/N with N=8. Arithmetic confirmed: 1×7/8=0.875, 2×7/8=1.75, 4×7/8=3.5, 8×7/8=7.

**Fix 6** (`expert_sharding.md` line 118, "34% redundant" language): Confirmed correct. The current text states: "~34% of transferred tokens produce no local FFN work (communication overhead)" and separately (line 128): "No expert FFN compute is wasted (every processed token legitimately belongs to a local expert)." These are now consistent — the ~34% overhead is correctly characterized as communication overhead (tokens transferred but producing no local FFN), not as wasted FFN compute.

## Key Check Results

**Check 1 — All-gather volume formula and relationship to all-to-all:**

The chapter correctly and consistently uses:

- V_gather = (N-1) × B × H × 2 = 100,352 × B bytes per device (all-gather sends full local batch to each of N-1 other devices)
- V_a2a = (N-1)/N × k × B × H × 2 = 100,352 × B bytes per device (all-to-all dispatch with k/N routing sparsity per remote device)

For Qwen3.5-35B with k=N=8, these are algebraically equal: (N-1) = (N-1)/N × k when k=N. The chapter's claim V_gather = V_a2a for k=N=8 is mathematically correct under the standard per-device outbound-byte counting model. The prompt context's suggested "8× cheaper" figure for all-gather uses a non-standard formula ((N-1)/N × H × 2 = 12,544 bytes) that does not correspond to the actual all-gather volume for this scheme (which sends the full local batch, not 1/N of it). No issue with the chapter; the equality claim is correct.

**Check 2 — Hierarchical table values:**

All four expected cross-device send values are correct: 0.875, 1.75, 3.5, 7. Verified by formula k_c × (N-1)/N = k_c × 7/8 for N=8.

**Check 3 — scheme_comparison_matrix.md crossover analysis:**

The crossover section correctly states: (a) V_gather = V_a2a for k=N=8 (communication volume is not the differentiating factor); (b) the decision turns on router compute overhead — all-gather incurs N=8× router compute over NB tokens vs. B tokens under all-to-all; (c) at B=32, the extra router cost is ~3.1 μs vs. ~514 μs all-to-all collective time, confirming all-to-all is preferred at non-trivial batch sizes; (d) the formal crossover batch size B* is deferred to Chapter 6. The recommendation (all-to-all as default, all-gather viable at very small B×S) correctly accounts for both communication parity and the compute asymmetry. No issue.

**Check 4 — V_a2a = V_gather equality (not 8× difference):**

The chapter correctly states V_a2a = V_gather for k=N=8. This is the algebraically correct result: the k/N routing sparsity in the all-to-all formula (k/N = 8/8 = 1 when k=N) exactly offsets the factor by which all-gather might otherwise exceed all-to-all. No correction is needed; the equality has been the stated result since Pass 1 and remains correct.

## Fresh Correctness Re-check

All five files were re-read in full. No new issues found:

- All arithmetic in `index.md`, `expert_sharding.md`, `pipeline_expert_parallelism.md`, `hierarchical_routing.md`, and `scheme_comparison_matrix.md` checks out.
- Pipeline efficiency formula η = μ/(μ+P-1) and all numerical examples (12.5%, 53.3%, 90.1%) remain correct.
- Per-stage communication figures (57,344 bytes, 4.6 μs, 458,752 bytes at M=32, 257 μs for 7 stages) remain correct.
- The ~66% token relevance probability (P = 1 - C(224,8)/C(256,8) ≈ 0.656) is correctly cited; the ~34% communication overhead figure is consistent.
- All D-dependent claims carry [D UNVERIFIED] tags; no untagged D-dependent numeric claims found.
- T3K topology consistently described as 1×8 linear, ttnn.Topology.Linear (not ring), throughout all five files.
- Hierarchical routing correctly marked not applicable to Qwen3.5-35B (requires retraining); pipeline expert parallelism correctly marked not applicable (requires sequential expert execution).
- scheme_comparison_matrix.md recommendation (all-to-all default, all-gather viable alternative at k=N, pipeline and hierarchical not applicable) is consistent with all constituent file analyses.
- Cross-chapter references are internally consistent.

**No feedback — chapter approved.**

---

# Agent B Review — Chapter 3: Alternative Expert Routing Schemes — Pass 4

## Verification of Compression Changes C1, C2, C3

**C1 — `expert_sharding.md`, "Baseline Recap" section removed; cross-reference to `index.md` inserted (line 61).**

Confirmed applied. The replacement text reads: "For the all-to-all baseline volume formula and Qwen3.5-35B numerical instantiation, see `index.md` Section [notation/baseline section]." The actual `index.md` section header is "Relationship to Chapter 2 Baseline" (lines 73–84), not "notation/baseline section" — so the section label in the cross-reference is imprecise, but the target content exists and is correct. All analytical content from `expert_sharding.md` (the algebraic k=N coincidence derivation, the V_gather < V_a2a formula for k < N, compute waste analysis) is intact; only the verbatim repetition of the baseline formula was removed. No correctness issue introduced.

**C2 — `pipeline_expert_parallelism.md`, duplicate not-applicable conclusion replaced with back-reference to Precondition section (line 157).**

Confirmed applied. The "When Pipeline Expert Parallelism Wins" section now ends with: "As established in the Precondition section, Qwen3.5-35B's parallel top-$k$ routing makes this scheme inapplicable without retraining." The Precondition section (lines 39–47) does establish this — the back-reference is accurate. No information lost; no logical gap created.

**C3 — `hierarchical_routing.md`, premature retraining verdict at ~lines 105–107 removed.**

Confirmed applied. The "Inference-Time Load Skew" subsection now ends without a premature disqualification statement; the authoritative verdict is correctly placed in the "Training Dependency" section (lines 110–118) and reiterated in "When Hierarchical Routing Wins" (line 148). Logical flow is now clean: load-skew problem identified → Training Dependency explains why retraining is required → conditions for applicability stated in "When Hierarchical Routing Wins." No correctness issue introduced.

---

## Full Re-check

All key correctness items from Passes 1–3 verified unchanged after C1–C3:

- **V_gather = V_a2a for k=N=8:** Both equal (N-1)×B×H×2 = 100,352×B bytes per device. Algebraic equality confirmed; C1 did not touch this derivation. Correct.
- **Hierarchical table expected sends (0.875, 1.75, 3.5, 7):** Present and correct at `hierarchical_routing.md` lines 42–45. Formula k_c × (N-1)/N = k_c × 7/8 checks out for all rows. C3 removal was confined to ~lines 105–107 and did not affect the table.
- **Expected hierarchical volume formula:** E[V_hier] = (N-1)/N × k_f × B × H × 2 + 1/N × 0, still at `hierarchical_routing.md` lines 83–85. Single (N-1)/N factor confirmed; no double-application of remote-probability factor (the Pass 2 error). Correct.
- **Pipeline efficiency formula and arithmetic:** η = μ/(μ+P-1); μ=1→12.5%, μ=8→8/15≈53.3%, μ=64→64/71≈90.1%. All present and correct in `pipeline_expert_parallelism.md`. C2 did not alter these sections.
- **~66% token relevance / ~34% communication overhead:** Correctly stated in `expert_sharding.md` lines 110–111 as communication overhead (not wasted FFN compute). C1 did not touch this content.
- **D-dependent claims:** All instances of D-dependent numerical claims (communication-to-compute ratio, expert FFN latency) carry [D UNVERIFIED] tags across all five files. C1–C3 introduced no untagged D-dependent claims.
- **Not-applicable verdicts:** Pipeline not applicable (parallel top-k) correctly stated at Precondition section and line 157 of `pipeline_expert_parallelism.md`, and in `scheme_comparison_matrix.md`. Hierarchical not applicable (requires retraining) correctly stated in "Training Dependency" and "When Hierarchical Routing Wins" in `hierarchical_routing.md`, and in `scheme_comparison_matrix.md`. Both verdicts present exactly where they should be.
- **T3K topology:** Consistently described as 1×8 linear chain, `ttnn.Topology.Linear`, not a ring, throughout all five files. No change from C1–C3.
- **`scheme_comparison_matrix.md`:** Untouched by C1–C3. Recommendation (all-to-all as default, all-gather viable at k=N, pipeline and hierarchical not applicable) remains consistent with constituent file analyses. All arithmetic in the volume comparison subsection confirmed correct.
- **Cross-references:** All file paths referenced within files are consistent with the guide structure; no dangling or incorrect cross-references introduced by C1–C3.

No new issues found.

**No feedback — chapter approved.**
