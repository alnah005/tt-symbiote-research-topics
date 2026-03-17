# Agent B Review — Chapter 4: Expert-to-Device Assignment for 256 Experts on 8 Devices — Pass 1

1. **`uniform_partitioning.md` Section 4 — Wrong formula for per-device load $T_d$.**

   The file writes:

   > $T_d = k \cdot B \cdot \sum_{e \in \mathcal{E}_d} f_e$

   and then derives the mean as:

   > $\bar{T} = k \cdot B \cdot \sum_e f_e / N = k \cdot B \cdot k / N = k^2 B / N = 64B/8 = 8B$

   This is wrong. With the chapter's normalization ($\sum_e f_e = k$, meaning $f_e$ is the fraction of *tokens* routed to expert $e$ so that summing over all $k$ selections per token gives $\sum_e f_e = k$), the expected number of token-dispatch events landing on device $d$ is:

   $$T_d = B \cdot \sum_{e \in \mathcal{E}_d} f_e$$

   Under uniform routing, $\sum_{e \in \mathcal{E}_d} f_e = 32 \times (k/E) = 32 \times (8/256) = 1$, giving $T_d = B$. This matches the correct answer stated in Section 3. The formula as written introduces a spurious factor of $k$, producing $T_d = kB = 8B$ and mean $\bar{T} = 8B$ — both $8\times$ too large. The document partially acknowledges the error with an in-text "Wait" comment but never corrects the displayed formula. The formula itself must be fixed to $T_d = B \cdot \sum_{e \in \mathcal{E}_d} f_e$.

   Note: `load_aware_assignment.md` Section 2 correctly writes $T_d(\sigma) = k \cdot B \cdot \sum_{e:\,\sigma(e)=d} f_e$, which is consistent with the same erroneous factor-of-$k$ convention as `uniform_partitioning.md`. Either (a) the formula is correct and $f_e$ should be re-defined to normalize to 1 (not $k$), making $f_{\text{avg}} = 1/E$ instead of $k/E$, or (b) the formula should drop the leading $k$ and keep the current normalization. The two files are internally consistent with each other, but their formula is inconsistent with the numerical example in Section 3 of `uniform_partitioning.md` (which correctly gives $T_d = B$, not $kB$). The discrepancy must be resolved uniformly across both files.

2. **`expert_replication.md` Section 2 "Alternate derivation" — False equivalence claim between two replication-factor formulas.**

   The file presents two formulas for minimum replication factor:

   - Primary: $r_e = \max(1,\, \lceil f_e \cdot E \rceil)$
   - Alternate: $r_e = \max(1,\, \lceil f_e \cdot N \rceil)$

   It then claims: "With $N = 8$ and $f_{\text{avg}} = k/E = 1.0/N$, both formulas agree: $\lceil f_e \cdot N \rceil = \lceil f_e \cdot E/k \rceil$."

   This is false. The claim requires $N = E/k$, i.e., $8 = 256/8 = 32$, which is not true. The two formulas produce drastically different values:

   | $f_e$ | Primary: $\lceil f_e \cdot E \rceil$ | Alternate: $\lceil f_e \cdot N \rceil$ |
   |---|---|---|
   | $f_{\text{avg}} = 0.03125$ | $\lceil 8 \rceil = 8$ | $\lceil 0.25 \rceil = 1$ |
   | $2 \times f_{\text{avg}} = 0.0625$ | $\lceil 16 \rceil = 16$ (cap at 8) | $\lceil 0.5 \rceil = 1$ |
   | $4 \times f_{\text{avg}} = 0.125$ | $\lceil 32 \rceil = 32$ (cap at 8) | $\lceil 1.0 \rceil = 1$ |

   The alternate formula $\lceil f_e \cdot N \rceil$ is wrong: it would classify even a $4\times$-overloaded expert as needing no replication ($r_e = 1$). The alternate derivation section should be removed or corrected. The primary formula $r_e = \max(1,\, \lceil f_e \cdot E \rceil)$ is the correct one (as derived from the capacity constraint in the lines immediately above it).

---

# Agent A Fix — Chapter 4: Expert-to-Device Assignment — Pass 1 Fixes Applied

1. `uniform_partitioning.md` and `load_aware_assignment.md` — Fixed T_d formula: removed spurious factor of k. Correct formula is T_d = B × sum_{e in E_d} f_e (not k × B × sum). Rewrote Section 4 of uniform_partitioning.md to define f_e clearly (expected tokens / B, sum = k), restate the correct mean T_bar = B, and remove the "Wait —" inconsistency. Updated worked example: Delta T_d = (0.125 - 0.03125) × B = 0.09375B → T_d^hot ≈ 1.09B, L ≈ 1.09. Fixed load_aware_assignment.md line 99 formula accordingly.

2. `expert_replication.md` — Removed "Alternate derivation using N" block. The alternate formula r_e = max(1, ceil(f_e × N)) is incorrect: N=8 ≠ E/k=32, so for f_e = 4×f_avg the two formulas give r_e=8 vs r_e=1 respectively. Retained the Warning blockquote (with reference to the correct formula r_e = ceil(f_e × E)).

---

# Agent B Review — Chapter 4: Expert-to-Device Assignment for 256 Experts on 8 Devices — Pass 2

**Verification of Pass 1 fixes:**

Fix 1 confirmed: `uniform_partitioning.md` Section 4 now correctly states $T_d = B \cdot \sum_{e \in \mathcal{E}_d} f_e$ (no spurious $k$), mean $\bar{T} = B$, and the worked example gives $\Delta T_d = 0.09375B$, $T_d^{\text{hot}} \approx 1.09B$, $L \approx 1.09$. `load_aware_assignment.md` line 99 likewise reads $T_d(\sigma) = B \cdot \sum_{e:\,\sigma(e)=d} f_e$. Both correct.

Fix 2 confirmed: the "Alternate derivation using N" block has been removed from `expert_replication.md`. The file proceeds directly from the primary formula to the worked table, with only the Warning blockquote remaining. Correct.

**Remaining issues:**

1. **`expert_replication.md` Sections 1–2 — overflow condition and replication formula are inconsistent with the now-fixed chapter convention for $f_e$.**

   After the Pass 1 fix, the chapter's uniform convention is: $f_e$ = expected tokens routed to expert $e$ per forward pass, divided by $B$; therefore tokens to expert $e$ per forward pass = $f_e \cdot B$, and $\sum_e f_e = k$.

   However, `expert_replication.md` Sections 1 and 2 still use tokens-to-expert-$e$ = $f_e \cdot k \cdot B$ (the old, pre-fix convention). This produces two downstream errors:

   a. **Overflow condition (Section 1):** The file writes the overflow condition as $f_e \cdot k \cdot B > C$, giving $f_e > 1/E = 1/256 \approx 0.0039$. Under the fixed convention, tokens to expert $e$ = $f_e \cdot B$, so the correct overflow condition is $f_e \cdot B > C \approx kB/E$, giving $f_e > k/E = f_{\text{avg}} = 0.03125$. The document's threshold is $k = 8$ times too small. As a direct consequence, the claim that follows — "any expert with above-average frequency overflows when $CF = 1.0$" — is true under the fixed convention but stated as if it is derived from the formula $f_e > 1/E$; with CF = 1.25 the document gives threshold $\approx 0.0049$, but the correct threshold under the fixed convention with $CF = 1.25$ is $CF \cdot k/E = 1.25 \times 0.03125 = 0.0391$.

   b. **Replication factor formula (Section 2):** The file writes $T_{\text{replica}}(e) = f_e \cdot k \cdot B / r_e$, sets this $\leq kB/E$, and cancels $k \cdot B$ to obtain $r_e \geq f_e \cdot E$, hence $r_e = \max(1, \lceil f_e \cdot E \rceil)$. Under the fixed convention, tokens per replica = $f_e \cdot B / r_e$; setting $\leq kB/E$ gives $r_e \geq f_e \cdot E / k$, hence $r_e = \max(1, \lceil f_e \cdot E / k \rceil)$.

   The magnitude of the error: under the fixed convention and the current formula $r_e = \max(1, \lceil f_e \cdot E \rceil)$, an expert at the mean frequency $f_{\text{avg}} = k/E = 0.03125$ would require $r_e = \lceil 0.03125 \times 256 \rceil = 8$ replicas — all 8 devices. But a mean-load expert is by definition exactly at capacity $C$ and needs only 1 copy. The worked table in Section 2 reflects this error: it shows $f_{\text{avg}}$ needing $r_e = 8$ ("every device holds a copy"), which is incorrect; an average-load expert requires $r_e = 1$.

   The correct table under the fixed convention, using $r_e = \max(1, \lceil f_e \cdot E / k \rceil)$:

   | $f_e$ | $f_e \cdot E / k$ | Correct min $r_e$ |
   |---|---|---|
   | $f_{\text{avg}} = 0.03125$ | 1 | 1 |
   | $2 \times f_{\text{avg}} = 0.0625$ | 2 | 2 |
   | $4 \times f_{\text{avg}} = 0.125$ | 4 | 4 |
   | $0.5 \times f_{\text{avg}} = 0.01563$ | 0.5 | 1 (no replication needed) |
   | $0.25 \times f_{\text{avg}} = 0.00781$ | 0.25 | 1 (no replication needed) |

   Note also the sentence at the end of Section 2: "experts with $f_e \cdot E > 8$ (i.e., $f_e > 1/32 = f_{\text{avg}}$) must be replicated on all 8 devices" — under the fixed formula this would instead be experts with $f_e \cdot E / k > 8$, i.e., $f_e > 8k/E = 8 \times 8/256 = 0.25$, not $f_e > f_{\text{avg}}$.

   **Required fix:** Update Sections 1 and 2 of `expert_replication.md` to use the fixed convention throughout. Specifically:
   - Replace $f_e \cdot k \cdot B$ with $f_e \cdot B$ as the tokens-to-expert-$e$ expression.
   - Replace overflow threshold $f_e > 1/E$ with $f_e > k/E = f_{\text{avg}}$.
   - Replace $r_e = \max(1, \lceil f_e \cdot E \rceil)$ with $r_e = \max(1, \lceil f_e \cdot E / k \rceil)$.
   - Update the worked table accordingly.

2. **`index.md` Goal Statement — two residual inconsistencies with the fixed $f_e$ convention.**

   a. **Line 28 objective formula** still writes the per-device load as $\sum_{e:\,\sigma(e)=d} f_e \cdot k \cdot B$. Under the fixed convention (tokens to device $d$ = $B \cdot \sum_{e:\sigma(e)=d} f_e$), the factor of $k$ is spurious. The formula should read $\sum_{e:\,\sigma(e)=d} f_e \cdot B$ (or equivalently $B \cdot \sum_{e:\,\sigma(e)=d} f_e$).

   b. **Line 32 uniform-routing parenthetical** states "Under uniform routing ($f_e = 1/E$ for all $e$)". But the chapter's notation (index.md lines 60–61) defines $f_e$ satisfying $\sum_e f_e = k$, so under uniform routing $f_e = k/E = 8/256 = 1/32$, not $1/E = 1/256$. The parenthetical should read $f_e = k/E$ for all $e$.

   **Required fix:** In `index.md` line 28, change $f_e \cdot k \cdot B$ to $f_e \cdot B$. In line 32, change $f_e = 1/E$ to $f_e = k/E$.

---

# Agent A Fix — Chapter 4: Expert-to-Device Assignment — Pass 2 Fixes Applied

1. `expert_replication.md` Section 1 — Fixed overflow condition to use fixed convention (tokens per expert = f_e × B, not f_e × k × B): changed `f_e · k · B > C` to `f_e · B > C`, fixed threshold from `f_e > 1/E = 0.0039` to `f_e > k/E = f_avg = 0.03125`, fixed CF=1.25 threshold from `CF/E ≈ 0.0049` to `CF × k/E = 0.0391`.

2. `expert_replication.md` Section 2 — Fixed replication factor formula: changed `T_replica = f_e·k·B/r_e` to `T_replica = f_e·B/r_e`; derivation now gives `r_e ≥ f_e·E/k`; formula updated to `r_e = max(1, ceil(f_e·E/k))`; for E=256, k=8 simplifies to `max(1, ceil(32·f_e))`. Updated worked table: f_avg now needs r_e=1 (not 8); 2×f_avg→r_e=2; 4×f_avg→r_e=4; sub-average frequencies remain r_e=1. Updated cap threshold: experts with f_e·E/k > 8 (i.e., f_e > 0.25) need all 8 devices. Fixed Warning formula. Fixed Section 6: T_expert = f_top × B = 0.125 × 32 = 4 tokens (not 32); correct r_e = ceil(0.125×256/8) = 4; each replica receives 1 token.

3. `index.md` line 28 — Changed `f_e · k · B` to `f_e · B` in the objective maximization formula.

4. `index.md` line 32 — Changed `f_e = 1/E` to `f_e = k/E` in uniform routing parenthetical.

---

# Agent B Review — Chapter 4: Expert-to-Device Assignment for 256 Experts on 8 Devices — Pass 3

**Verification of Pass 2 fixes:**

Fix 1 confirmed: `expert_replication.md` Section 1 now correctly states the overflow condition as `f_e · B > C`, gives threshold `f_e > k/E = f_avg = 0.03125`, and CF=1.25 threshold as `CF × k/E = 0.0391`. Correct.

Fix 2 confirmed: `expert_replication.md` Section 2 now uses `T_replica(e) = f_e · B / r_e`, derives `r_e ≥ f_e · E / k`, and states `r_e = max(1, ceil(f_e · E / k)) = max(1, ceil(32 · f_e))`. The worked table correctly shows f_avg → r_e=1, 2×f_avg → r_e=2, 4×f_avg → r_e=4, sub-average → r_e=1. Cap threshold correctly stated as f_e > 0.25 (i.e., f_e · E/k > 8). Correct. Section 6 lower portion also correctly shows T_expert = 0.125 × 32 = 4 tokens, r_e = ceil(0.125 × 256/8) = 4, each replica receives 1 token. Correct.

Fix 3 confirmed: `index.md` line 28 objective formula now reads `sum_{e: σ(e)=d} f_e · B` (factor of k removed). Correct.

Fix 4 confirmed: `index.md` line 32 now reads `f_e = k/E` for all e under uniform routing. Correct.

**Remaining issues:**

1. **`expert_replication.md` Section 6 — Zipf top-expert replication factor still uses the old (pre-fix) formula.**

   File: `expert_replication.md`, Section 6 "Practical Recommendation for Qwen3.5-35B", second paragraph (line 178).

   The text states:

   > This implies $r_{(1)} = \lceil 0.17 \times 256 \rceil = \lceil 43.5 \rceil = 44$, which exceeds $N = 8$ — so the top expert should be replicated on all 8 devices.

   This uses the old formula `r_e = ceil(f_e · E)` instead of the corrected formula `r_e = max(1, ceil(f_e · E / k))`. With `f_(1) ≈ 0.17`, `E = 256`, `k = 8`, the correct calculation is:

   $$r_{(1)} = \left\lceil \frac{0.17 \times 256}{8} \right\rceil = \lceil 0.17 \times 32 \rceil = \lceil 5.44 \rceil = 6$$

   The intermediate value 43.5 is wrong (correct: 5.44) and the result 44 is wrong (correct: 6). The follow-on conclusion "which exceeds $N = 8$" is also wrong: 6 does not exceed N = 8. The correct conclusion is that the top expert needs 6 replicas (6 devices), not all 8 devices.

   Required fix: Replace `r_{(1)} = \lceil 0.17 \times 256 \rceil = \lceil 43.5 \rceil = 44, which exceeds N = 8 — so the top expert should be replicated on all 8 devices` with `r_{(1)} = \lceil 0.17 \times 256 / 8 \rceil = \lceil 5.44 \rceil = 6`, and update the conclusion to state the top expert needs 6 replicas (which is less than N = 8, so full replication on all 8 devices is not required, though it is acceptable as a conservative choice).

2. **`load_aware_assignment.md` Section 2 — prose names wrong constant when reducing the objective.**

   File: `load_aware_assignment.md`, Section 2 "Load Imbalance Cost Function", line 107.

   The text states:

   > Since $k \cdot B$ is a constant across all devices, this reduces to: $\min_{\sigma} \max_{d} \sum_{e:\,\sigma(e)=d} f_e$

   But the formula being reduced (line 99) is `T_d(σ) = B · sum_{e: σ(e)=d} f_e`. The constant factor being divided out is `B`, not `k · B`. The phrase "Since $k \cdot B$ is a constant" is a residual from the old (pre-Pass-1-fix) formula where `T_d = k · B · sum f_e`. Under the corrected formula, the text should read "Since $B$ is a constant across all devices."

   Required fix: In `load_aware_assignment.md` line 107, replace "Since $k \cdot B$ is a constant across all devices" with "Since $B$ is a constant across all devices".

---

# Agent A Fix — Chapter 4: Expert-to-Device Assignment — Pass 3 Fixes Applied

5. `expert_replication.md` Section 6 — Fixed Zipf replication factor calculation: replaced `r_{(1)} = ceil(0.17 × 256) = ceil(43.5) = 44, which exceeds N=8 — all 8 devices` with correct formula `r_{(1)} = ceil(0.17 × 256 / 8) = ceil(5.44) = 6`. Updated conclusion: top expert needs 6 replicas (does not exceed N=8; full replication is acceptable but not required).

6. `load_aware_assignment.md` Section 2 — Fixed explanatory prose: changed "Since k·B is a constant across all devices" to "Since B is a constant across all devices" (residual from old pre-fix formula where T_d used k·B·sum; corrected formula only has B·sum).

---

# Agent A Fix — Chapter 4: Expert-to-Device Assignment — Compression Pass 1 Fixes Applied

C1. `load_aware_assignment.md` Section 1 — Replaced redundant re-derivation of f_avg ("The normalization satisfies ... average frequency is f_avg = k/E = 8/256 = 1/32 ≈ 0.03125") with a single back-reference: "The normalization satisfies sum f_e = k (f_avg = k/E = 1/32; see `uniform_partitioning.md`, Section 4)."

C2. `load_aware_assignment.md` Section 6.3 — Replaced re-statement of W_expert formula ("where W_expert = 3 × H × D × 2 bytes (BF16, three weight matrices)") with pointer: "where W_expert is the per-expert weight size (= 6HD bytes BF16; see `uniform_partitioning.md`, Section 2)".
    `expert_replication.md` Section 3 — Replaced expanded formula block (lines 73-77 with M × 6 × H × D and the separate per-Qwen equation) with condensed form: `= M × W_expert` followed by pointer to `uniform_partitioning.md` Section 2.

C3. `mesh_topology_constraints.md` Section 9 — Replaced the 4-step re-enumeration of re-assignment steps ("1. Re-running the profiling... 2. Re-solving... 3. Migrating... 4. Updating...") and the "Steps 3 and 4 introduce latency..." paragraph with a single pointer to `load_aware_assignment.md` Section 6. Retained the unique deployment policy guidance: "maintenance windows or when L sustained above 1.5 for more than 10,000 forward passes".

---

# Agent B Review — Chapter 4: Expert-to-Device Assignment for 256 Experts on 8 Devices — Pass 4

**Verification of Pass 3 fixes:**

Fix 5 confirmed: `expert_replication.md` Section 6 now reads `r_{(1)} = \lceil 0.17 \times 256 / 8 \rceil = \lceil 5.44 \rceil = 6`, and the conclusion correctly states the top expert needs 6 replicas, which does not exceed N=8, so full replication on all 8 devices is acceptable but not required. Correct.

Fix 6 confirmed: `load_aware_assignment.md` Section 2 line 107 now reads "Since $B$ is a constant across all devices". Correct.

**Full review of all five files — no remaining technical errors found.**

All formulas, constants, and numerical examples in all five files are now consistent with the established ground truth:

- `index.md`: objective formula uses `f_e · B` (not `f_e · k · B`); uniform routing parenthetical correctly states `f_e = k/E`; notation table correctly defines `E_d = 32`, `f_avg = k/E = 0.03125`.
- `uniform_partitioning.md`: `T_d = B · sum_{e in E_d} f_e`; `T_bar = B`; `f_avg = k/E = 8/256 = 1/32`; uniform routing sum check `32 × (1/32) = 1` correct; load imbalance worked example (`L = 0.15/0.125 = 1.2`) correct; hot-expert delta calculation (`Delta T_d = 0.09375B`, `T_d^hot ≈ 1.09B`, `L ≈ 1.09`) correct.
- `load_aware_assignment.md`: `T_d(σ) = B · sum f_e`; "Since B is a constant" reduction correct; GDF bound `(4/3 - 1/(3N)) × OPT`; for N=8: `(4/3 - 1/24) = 31/24 ≈ 1.292` correct.
- `expert_replication.md`: overflow condition `f_e · B > C`, threshold `f_e > k/E = 0.03125`, CF=1.25 threshold `f_e > 0.0391` correct; replication formula `r_e = max(1, ceil(f_e · E / k)) = max(1, ceil(32 · f_e))` correct; worked table (`f_avg → r_e=1`, `2×f_avg → r_e=2`, `4×f_avg → r_e=4`) correct; cap threshold `f_e > 0.25` correct; Zipf calculation `r_{(1)} = ceil(5.44) = 6` correct; `T_expert = 0.125 × 32 = 4 tokens`, each replica 1 token, `r_e = 4` correct.
- `mesh_topology_constraints.md`: average hop count derivation (`sum = 84`, `h_bar = 168/56 = 3.0`) correct; uniform co-activation `w_{ij} = 8×7/(256×255) = 56/65280 ≈ 0.000858` correct.

No feedback — chapter approved

---

# Agent B Review — Chapter 4: Expert-to-Device Assignment — Pass 5

**Verification of Compression Pass 1 fixes (C1, C2, C3):**

C1 confirmed: `load_aware_assignment.md` Section 1 no longer contains the full re-derivation of $f_{\text{avg}}$. Line 21 now reads "The normalization satisfies $\sum_{e=0}^{E-1} f_e = k$ ($f_{\text{avg}} = k/E = 1/32$; see `uniform_partitioning.md`, Section 4)." — a single back-reference with no redundant prose. Correct.

C2 confirmed: `load_aware_assignment.md` Section 6.3 (line 389) now reads "where $W_{\text{expert}}$ is the per-expert weight size ($W_{\text{expert}} = 6HD$ bytes BF16; see `uniform_partitioning.md`, Section 2)" — the full three-matrix re-derivation is gone; only a pointer plus the final scalar formula remain. `expert_replication.md` Section 3 (line 75) now reads "(see `uniform_partitioning.md`, Section 2 for the full derivation: $W_{\text{expert}} = 6HD$ bytes BF16)" — same pattern; expanded formula block removed. Both correct.

C3 confirmed: `mesh_topology_constraints.md` Section 9 no longer contains the four-step enumeration (re-profiling, re-solving, migrating, updating dispatch metadata) or the follow-on migration-latency paragraph. The section now reads "For re-assignment mechanics (re-profiling, re-solving, weight migration cost, and dispatch metadata updates), see `load_aware_assignment.md`, Section 6." The deployment threshold unique to Section 9 — "L sustained above 1.5 for more than 10,000 forward passes" and the maintenance-window recommendation — is retained. Correct.

**Full ground-truth consistency check (all five files, Pass 5):**

All previously confirmed facts remain correct and no regressions introduced by compression edits:

- `index.md`: objective formula $\sum_{e:\sigma(e)=d} f_e \cdot B$; uniform routing $f_e = k/E$; notation table $E_d = 32$, $f_{\text{avg}} = k/E = 0.03125$. Correct.
- `uniform_partitioning.md`: $T_d = B \cdot \sum_{e \in \mathcal{E}_d} f_e$; $\bar{T} = B$; $f_{\text{avg}} = 1/32$; Section 2 defines $W_{\text{expert}} = 6HD$ bytes BF16 (canonical definition that C2 back-references). Correct.
- `load_aware_assignment.md`: $T_d(\sigma) = B \cdot \sum f_e$ (Section 2); "Since $B$ is a constant" reduction (Section 2); back-reference to `uniform_partitioning.md` Section 4 for $f_{\text{avg}}$ (C1, Section 1); back-reference to `uniform_partitioning.md` Section 2 for $W_{\text{expert}}$ with inline value $6HD$ bytes BF16 (C2, Section 6.3). Correct.
- `expert_replication.md`: overflow threshold $f_e > k/E = 0.03125$; CF=1.25 threshold $0.0391$; $r_e = \max(1, \lceil f_e \cdot E / k \rceil) = \max(1, \lceil 32 f_e \rceil)$; worked table ($f_{\text{avg}} \to r_e = 1$, $2 \times f_{\text{avg}} \to r_e = 2$, $4 \times f_{\text{avg}} \to r_e = 4$); cap at $f_e > 0.25$; Zipf top $r_{(1)} = \lceil 5.44 \rceil = 6$; back-reference to `uniform_partitioning.md` Section 2 for $W_{\text{expert}}$ (C2, Section 3). Correct.
- `mesh_topology_constraints.md`: average hop count $\bar{h} = 3.0$; uniform co-activation $w_{ij}^{\text{uniform}} \approx 0.000858$; condensed Section 9 with deployment threshold retained (C3). Correct.

No remaining technical errors found.

No feedback — chapter approved
