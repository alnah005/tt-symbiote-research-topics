# Agent B Review — Chapter 1: MoE Fundamentals and the Routing Problem — Pass 1

1. **`qwen35b_config.md`, line 26 — Attention head count and head dimension are inconsistent with $H$.**
   The constants table states 64 query heads with 128 head_dim, giving $64 \times 128 = 8{,}192$. The hidden dimension is stated as $H = 7{,}168$. In a standard transformer the product of head count and head dimension equals the hidden dimension; these three values cannot all be correct simultaneously. The file flags this in an inline note but leaves all three values in the table as if they are usable facts. Any reader who computes attention weight sizes using 64 heads × 128 head_dim × 4 matrices × 94 layers will get a parameter count inconsistent with the rest of the chapter's memory budget. Fix: remove head count and head_dim from the constants table until they are verified against `config.json`, or replace them with the correct values so the product equals $H = 7{,}168$.

2. **`qwen35b_config.md`, lines 58–62 and 169–173 — Per-expert intermediate dimension $D = 2048$ is demonstrably wrong and silently corrupts every numerical result that depends on it.**
   The file itself proves $D = 2048$ is impossible: $256 \times 3 \times 7{,}168 \times 2{,}048 \approx 11.3\text{B}$ parameters per MoE layer × 80 layers $\approx 902\text{B}$ total expert parameters, which exceeds the entire 35B model by more than 25×. Despite this, every downstream calculation continues to use $D = 2048$ without correction: per-expert FLOPs ($6HD \approx 88\text{M}$), total expert FLOPs per token ($\approx 56.4\text{G}$), per-expert memory ($\approx 44\text{M}$ parameters, 88 MB in BF16), and the active-parameter estimate of 46.3B. A reader who sizes dispatch buffers, computes hardware roofline numbers, or implements expert FFN kernels using these figures will build a system with wrong resource estimates. Fix: determine the correct $D$ from the Qwen3 Technical Report before publishing; replace all derived figures, or mark every figure derived from $D$ as an unverified placeholder that must not be used for implementation decisions.

3. **`qwen35b_config.md`, lines 120–123 — The "pure EP per-device memory" figure of ~42.5 GB is derived from the unverified 35B total and the inconsistent non-expert parameter sum, making the memory feasibility conclusion unreliable.**
   The calculation $(15.7\text{B} / 8 + 19.3\text{B}) \times 2 \approx 42.5\text{ GB}$ is presented as a concrete engineering bound used to motivate EP + TP or quantization. However, 15.7B is computed as $35\text{B} - 19.3\text{B}$, and 19.3B is itself a sum that includes the 5.7B dense FFN estimate, which uses $D_\text{dense} = 18{,}944$ (unverified) and $L_\text{dense} = 14$. Because the 35B total is treated as authoritative while the per-layer derivations are acknowledged to be inconsistent, the subtraction $35\text{B} - 19.3\text{B}$ is mixing an authoritative figure with an unverified one. The 42.5 GB conclusion — which drives the guide's central claim that pure EP is insufficient for T3K — could be wrong in either direction. Fix: derive the per-device EP memory bound from the authoritative 35B total only (not from the per-layer constants), clearly state the assumption, and defer the exact figure to after the architectural constants are verified.

4. **`routing_problem.md`, line 7 — The $(N-1)/N$ cross-device fraction is stated as a fraction of expert slots per token, but the prose preceding it could be read as applying to the fraction of tokens, causing dispatch-volume estimates to be wrong by up to $k\times$.**
   The sentence "an expected $\frac{N-1}{N} = \frac{7}{8} = 87.5\%$ of the selected experts live on a device other than $d$" is correct: it is the per-expert-slot remote probability. However, the paragraph opens by saying "when a token arrives at device $d$" and then gives $(N-1)/N$ without immediately clarifying the unit (per slot, not per token). A reader building a dispatch volume estimator could interpret this as "87.5% of tokens require a cross-device send" rather than "87.5% of the $k$ expert slots per token are remote," underestimating dispatch volume by the factor $k = 8$ if they then multiply by $B$ instead of $B \times k$. The note on line 23 partially clarifies this but comes after the calculation. Fix: at line 7, change the framing to explicitly say "each of the $k$ selected expert slots has a $(N-1)/N$ probability of being remote, so on average $k(N-1)/N = 7$ of the 8 expert slots per token require inter-device sends."

5. **`moe_architecture.md`, line 39 — The claim that sigmoid routing "does not enforce the sum-to-one constraint" is left unresolved against the renormalization formula immediately following, and could cause implementors to skip renormalization for sigmoid variants.**
   The chapter defines renormalized routing weights as $\hat{w}_i = p_i / \sum_{j \in I} p_j$ (lines 47–51) and states that this ensures $\sum_{i \in I} \hat{w}_i = 1$. Three paragraphs earlier it states sigmoid routing "does not enforce this sum-to-one constraint." A reader implementing a sigmoid router who takes the "does not enforce sum-to-one" language to mean renormalization is inapplicable or optional will produce a weighted combination $y = \sum_{i \in I} \hat{w}_i \cdot o_i$ where the $\hat{w}_i$ are raw sigmoid outputs that can sum to any positive value, yielding an arbitrarily scaled MoE output that does not match the mathematical formulation. Fix: add one sentence at line 39 stating that the renormalization step $\hat{w}_i = p_i / \sum_{j \in I} p_j$ is applied regardless of whether the raw probabilities come from softmax or sigmoid, so the weighted combination is always a convex combination.

---

# Agent B Review — Chapter 1: MoE Fundamentals and the Routing Problem — Pass 2

All five issues from Pass 1 have been addressed. The prior fixes are confirmed applied. Three residual issues remain, none as severe as the originals.

1. **`qwen35b_config.md`, dense FFN intermediate dimension $D_\text{dense} = 18{,}944$ carries no UNVERIFIED caveat, but it propagates directly into the 42.5 GB per-device EP memory figure.**
   The chapter carefully marks the per-expert intermediate dimension $D = 2048$ as UNVERIFIED throughout, yet the dense FFN intermediate dimension $D_\text{dense} = 18{,}944$ is stated as fact with no corresponding warning. This value feeds into the dense FFN parameter estimate ($14 \times 3 \times 7{,}168 \times 18{,}944 \approx 5.7\text{B}$), which is one of the addends in the 19.3B non-expert parameter total, which is then subtracted from the 35B model total to yield the 15.7B expert-weight figure used in the 42.5 GB per-device EP memory calculation. If $D_\text{dense}$ is wrong, the 42.5 GB figure — the guide's central argument for why pure EP is insufficient on T3K — is wrong by a proportional amount. The inconsistency is not that $D_\text{dense} = 18{,}944$ is certainly incorrect; it is that it is treated as authoritative while $D$ is treated as suspect, without any stated basis for the asymmetry. Fix: add a verification caveat to $D_\text{dense} = 18{,}944$ matching the treatment of $D$, and propagate that caveat to the 19.3B non-expert sum and the 42.5 GB figure.

2. **`qwen35b_config.md`, line 232 — "every token participates in cross-device communication to every device" overstates the all-to-all density under uniform routing.**
   The text argues that because $k = N = 8$, each token on average sends to one expert on each of the $N = 8$ devices. This is used to conclude that "every token participates in cross-device communication to every device." Under uniform routing, the expected number of local expert slots per token is $k/N = 1$, so on average a token is dispatched to $k - 1 = 7$ remote devices, not all 8. The one local slot requires no cross-device send. A reader who implements a dispatch scheduler based on this claim and assumes all $N$ devices are always active recipients for every token will over-provision dispatch buffers and mis-characterize the all-to-all as a full $N \times N$ exchange rather than an expected $(N-1)$-way fan-out per token. Fix: replace "every device" with "on average $N-1 = 7$ other devices" to match the expected-value analysis already given earlier in the same file and in `routing_problem.md`.

3. **`routing_problem.md`, line 33 — the stated "~245 MB" total dispatch volume mixes binary and decimal unit conventions and understates the SI-megabyte result by approximately 5%.**
   The per-layer volume is computed as $32 \times 7 \times 7{,}168 \times 2 = 3{,}211{,}264$ bytes (correct arithmetic). Multiplied by 80 MoE layers: $3{,}211{,}264 \times 80 = 256{,}901{,}120$ bytes. In SI megabytes ($1\text{ MB} = 10^6$ bytes) this is $\approx 257\text{ MB}$; in binary mebibytes ($1\text{ MiB} = 2^{20}$ bytes) it is $\approx 245\text{ MiB}$. The text calls the result "~245 MB," which is the mebibyte value written with the megabyte label. A reader who uses this figure in a bandwidth or buffer-sizing calculation will be working with a number that is ~5% too low if they apply SI conventions (as most hardware datasheets do). The round-trip total of "~490 MB" is similarly understated. Fix: either use "MiB" throughout this calculation, or restate as "~257 MB" using consistent SI units.

No further correctness issues were found. The mathematical formulations, routing equations, and conceptual descriptions of dispatch/combine are accurate. The chapter's handling of unverified constants (with explicit placeholder warnings throughout) prevents most downstream misuse.

---

# Agent B Review — Chapter 1: MoE Fundamentals and the Routing Problem — Pass 3

All three issues from Pass 2 are confirmed fixed. One residual correctness issue remains.

1. **`qwen35b_config.md`, attention weight estimate of ~12.4B carries no UNVERIFIED caveat yet is a direct input to the 42.5 GB per-device EP memory figure.**
   The non-expert parameter total of 19.3B is computed as the sum of four components: embedding ($\approx 1.1$B), attention weights ($\approx 12.4$B), dense FFN weights ($\approx 5.7$B), and layer norms ($< 0.1$B). The dense FFN and its downstream consequences are now correctly flagged UNVERIFIED. However, the $\approx 12.4$B attention weight estimate is stated without any caveat, despite depending on query head count and head dimension — both of which are explicitly flagged as unverified (removed from the constants table with a note that the candidate values are internally inconsistent with $H = 7{,}168$). If the attention weight estimate is wrong, the 19.3B non-expert total is wrong, and the 42.5 GB per-device EP memory figure — the chapter's central argument for why pure EP is insufficient on T3K — is wrong for a second independent reason beyond the already-flagged $D_\text{dense}$ dependence. The existing warning block does state "This figure is contingent on the 19.3B non-expert estimate being correct," but does not identify the attention sub-estimate as an unverified contributor. A reader who notes that $D_\text{dense}$ is flagged, confirms it, and then trusts the remainder of the 19.3B sum will still be using an unverified $12.4$B figure without knowing it. Fix: in the UNVERIFIED warning block, add the $\approx 12.4$B attention weight estimate as an additional unverified contributor to the 19.3B non-expert total, alongside the $D_\text{dense}$-derived dense FFN term. The attention estimate requires verification of query head count and head dimension (head\_count × head\_dim must equal $H = 7{,}168$) before it can be used in the 42.5 GB calculation.

**No feedback on items 2–5 — no further correctness issues found.** The mathematical formulations (SwiGLU parameter count $3HD$, FLOPs formula $6HD$ under MAC convention, load imbalance ratio $L = f_{e^*}/(k/E)$, per-layer dispatch volume arithmetic, MiB unit fix) are all verified correct. The dispatch/combine conceptual flow and the $(N-1)/N$ per-slot remote probability (confirmed $k$-independent, as stated) are accurate. The chapter as a whole correctly identifies its own inconsistencies and prevents downstream misuse through explicit placeholder warnings.

---

# Agent B Review — Chapter 1: MoE Fundamentals and the Routing Problem — Pass 4

The Pass 3 fix is confirmed applied: the ~12.4B attention weight estimate is now explicitly named in the UNVERIFIED warning block in `qwen35b_config.md` as an unverified contributor to the ~42.5 GB figure, alongside the $D_\text{dense}$-derived term.

All prior fixes are verified in place. All formulas have been re-checked:
- SwiGLU parameter count $P_\text{expert} = 3HD$ (and its numeric check at $D = 2048$: $3 \times 7{,}168 \times 2{,}048 \approx 44\text{M}$) — correct.
- FLOPs formula $6HD$ under MAC convention (gate $2HD$ + up $2HD$ + down $2HD = 6HD$) — correct.
- Per-layer dispatch volume ($32 \times 7 \times 7{,}168 \times 2 = 3{,}211{,}264$ bytes; $\times 80 = 256{,}901{,}120$ bytes $\approx 245\text{ MiB}$) — correct.
- Load imbalance ratio $L = f_{e^*}/(k/E)$ and the code example ($B=512$, expected avg $= 512 \times 8 / 256 = 16$, $L = 80/16 = 5.0$) — correct.
- Layer count consistency: $L_\text{MoE} = 80$ + $L_\text{dense} = 14$ = $L_\text{total} = 94$ — correct.
- Router weight matrix shape $[H, E] = [7{,}168, 256]$ — consistent with $H$ and $E$ as stated.
- $(N-1)/N$ per-slot remote probability: $k$-independent, as stated and as noted in the summary table — correct.

**No feedback — chapter approved.**

---

# Agent B Review — Chapter 1: MoE Fundamentals and the Routing Problem — Pass 5

All five compression changes (C1–C5) have been reviewed. All prior fixes from Passes 1–4 remain in place. The compression changes are verified not to introduce any new correctness gaps:

- C1: The line-51 pointer in `moe_architecture.md` ("see convexity note at line 39") correctly references the authoritative statement at line 39, which covers both softmax and sigmoid cases. No information loss.
- C2: The deleted blockquote in `routing_problem.md` was a restatement of $\mathbb{E}[k_\text{remote}]$; the derivation remains present in the main text at lines 21–25. No information loss.
- C3: The CF notation table entry in `index.md` is accurate and complete as compressed; the formal definition pointer to Chapter 7 is retained.
- C4: The "This strengthens the conclusion…" sentence removal from `qwen35b_config.md` does not affect any formula or numerical result; the conclusion stands in the summary paragraph.
- C5: The compressed Note at line 175 of `qwen35b_config.md` still states that the 46.3B result is arithmetically impossible and directs the reader to the warning block, where the impossibility is stated explicitly (lines 60–61). No correctness loss.

All formulas re-verified: $P_\text{expert} = 3HD$, FLOPs $= 6HD$ (MAC convention), dispatch volume ($32 \times 7 \times 7168 \times 2 \times 80 \approx 245$ MiB), load imbalance ratio $L = f_{e^*}/(k/E)$, layer counts $80 + 14 = 94$, router matrix shape $[7168, 256]$, and $(N-1)/N$ per-slot remote probability — all correct and unchanged.

**No feedback — chapter approved.**

---

# Agent B Review — Chapter 1: MoE Fundamentals and the Routing Problem — Pass 6

All three Pass 5 compression residuals have been reviewed. All prior fixes from Passes 1–5 remain in place. The compression changes introduce no new correctness issues:

- C2 residual: The two removed restatements of $k(N-1)/N = 7$ in `routing_problem.md` were genuine redundancies. The formula is derived once in the introduction (line 7) and once in the formal expected-value section (lines 19–25). Both remaining occurrences serve distinct pedagogical roles; no information has been lost.
- C4 residual: Deletion of "It substantially exceeds…" from `qwen35b_config.md` does not affect any formula or numerical result. The substantive conclusion (pure EP insufficient on T3K) is retained in the surrounding paragraph.
- C5 residual: The compressed Note at `qwen35b_config.md` line 177 correctly states the gap as $46.3\text{B} - 22\text{B} = 24.3\text{B}$ (arithmetic verified) and retains the pointer to the Technical Report.

All formulas re-verified: $P_\text{expert} = 3HD$, FLOPs $= 6HD$ (MAC convention), dispatch volume ($32 \times 7 \times 7168 \times 2 \times 80 = 256{,}901{,}120$ bytes $\approx 245$ MiB), load imbalance ratio $L = f_{e^*}/(k/E)$, layer counts $80 + 14 = 94$, router matrix shape $[7168, 256]$, $(N-1)/N$ per-slot remote probability ($k$-independent), and active-parameter gap $24.3\text{B}$ — all correct and unchanged.

**No feedback — chapter approved.**
