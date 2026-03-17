# Compression Analysis — T3K Mesh Chapter 5: Expert Parallelism — Pass 1

## Summary
- Files reviewed: `index.md`, `expert_placement_strategies.md`, `token_routing_and_dispatch.md`, `combine_and_accumulation.md`
- Current line count: 206 (index.md) + 283 (expert_placement_strategies.md) + 287 (token_routing_and_dispatch.md) + 274 (combine_and_accumulation.md) = ~1050 lines
- Estimated post-compression: ~870 lines (~17% reduction)

---

## CRUCIAL Suggestions

### 1. Exact duplicate: dispatch volume formula + B=32 calculation
- **Location A:** `index.md` lines 173–183 — section "All-to-All Volume Scaling" derives $V_{\text{send}} = (N-1) \times C \times E_d \times H \times 2$ and computes the $B=32$ result as $\approx 6.4$ MB, plus a decode note.
- **Location B:** `expert_placement_strategies.md` lines 62–66 — section "All-to-All Payload" states the same formula $V_{\text{dispatch}} = (N-1) \times C \times E_d \times H \times 2$ and evaluates it to $6{,}422{,}528$ bytes $\approx 6.4$ MB at $B=32$.
- **Location C:** `token_routing_and_dispatch.md` lines 178–192 — section "All-to-All Dispatch Communication" restates the same formula and computes both $B=1$ ($\approx 3.2$ MB) and $B=32$ ($6{,}422{,}528$ bytes $\approx 6.4$ MB) with identical arithmetic.
- **Location D:** `combine_and_accumulation.md` lines 51–55 — "Volume Symmetry" asserts $V_{\text{combine}} = V_{\text{dispatch}}$ and re-evaluates to the same $6{,}422{,}528$ bytes $\approx 6.4$ MB at $B=32$ and $\approx 3.2$ MB at $B=1$.
- **What to remove:** Remove the full formula derivation and worked numeric examples from `index.md` lines 173–183 (the index should only cite the result via reference) and from `expert_placement_strategies.md` lines 62–66 (placement strategies file need not rederive the volume; a cross-reference to `token_routing_and_dispatch.md` §4 suffices). The authoritative derivation belongs in `token_routing_and_dispatch.md` §4. `combine_and_accumulation.md` may retain the symmetry assertion $V_{\text{combine}} = V_{\text{dispatch}}$ and cite the numeric result by reference rather than re-deriving it.
- **Information lost:** None.

### 2. Exact duplicate: naive uniform placement summary
- **Location A:** `index.md` lines 129–136 — "Summary of Expert Placement Strategies §1: Naive Uniform Placement" lists the $T_d = B$ load result, $L = 2$–$4$ imbalance warning, and `device_id = expert_index // 32` rule verbatim.
- **Location B:** `expert_placement_strategies.md` lines 13–73 — Section 1 "Baseline — Naive Uniform Placement" contains all of the same facts (assignment table, `device_id = expert_index // 32`, $T_d = B \times 32 \times (1/32) = B$, Zipf $L = 2$–$4$ warning) with full derivation.
- **What to remove:** Remove the four-strategy prose summary from `index.md` lines 127–164 (the entire "Summary of Expert Placement Strategies" section). This section duplicates `expert_placement_strategies.md` in condensed form; the index's navigation table (lines 65–78) already points readers to the authoritative file.
- **Information lost:** None. The index navigation table and "When to Read Which File" table already serve the orientation purpose without restating the content.

### 3. Exact duplicate: expert replication threshold and worked example
- **Location A:** `index.md` lines 158–163 — "Expert Replication §4" gives $f_e > 1.25/32 \approx 0.0391$ threshold, the $f_e = 0.17 \Rightarrow r_e = 6$ worked example, the per-replica load $0.17/6 \approx 0.028$, and the memory cost $6 \times W_{\text{expert}}$.
- **Location B:** `expert_placement_strategies.md` lines 194–214 — Section 4 "Expert Replication" derives the same overflow threshold (same formula, same numbers), the same Zipf worked example ($f_e = 0.17$, $r_e = 6$), the same per-replica load, and the same memory cost with full analysis plus a feasibility note.
- **What to remove:** The `index.md` copy (lines 156–163) is subsumed by the detail file. Remove these lines from `index.md` as part of removing the entire "Summary of Expert Placement Strategies" section (see CRUCIAL §2 above — same removal action covers both).
- **Information lost:** None.

---

## Load-Bearing Evidence

The following unique, technically precise content must be preserved (one copy each):

1. **`expert_placement_strategies.md` lines 131–133** — Full hop-count derivation: $\bar{h} = 84/28 = 3.0$ with the explicit sum broken out. This is the only location deriving the average-hop formula from first principles.
2. **`expert_placement_strategies.md` lines 148–168** — `build_coactivation_matrix` Python function; unique to this file and not replicated elsewhere.
3. **`expert_placement_strategies.md` lines 222–233** — `replica_device` hash-based token-to-replica assignment function; unique to this file.
4. **`token_routing_and_dispatch.md` lines 94–136** — `build_send_buffer` Python function; unique to this file.
5. **`token_routing_and_dispatch.md` lines 226–253** — `fused_router_dispatch` pseudocode; unique to this file.
6. **`token_routing_and_dispatch.md` lines 207–209** — Prefill send buffer size derivation: $C = \lceil 8 \times 32 \times 2048 \times 1.25 / 256 \rceil = 2{,}560$; total $\approx 9.4$ GB. This exact calculation appears only here.
7. **`combine_and_accumulation.md` lines 89–113** — `weighted_combine` Python function; unique to this file.
8. **`combine_and_accumulation.md` lines 188–198** — Numerical precision bound: $\epsilon_{\text{accum}} \lesssim k \times \epsilon_{\text{BF16}} = 8 \times 0.004 = 3.2\%$; unique to this file.
9. **`combine_and_accumulation.md` lines 133–136** — L1 accumulation DRAM-savings calculation: $\Delta t_{\text{DRAM}} \approx 12.2\ \mu\text{s}$; unique to this file.
10. **`combine_and_accumulation.md` lines 233–237** — Prefill combine latency: $t_{\text{combine}} \approx 117.4\ \text{MB} / 12.5\ \text{GB/s} \approx 9.4\ \text{ms}$ at $B=4, S=2048$; unique to this file.
11. **`expert_placement_strategies.md` lines 260–264** — Locality-aware latency savings formula: $\Delta t \approx 6.4\ \text{MB} / 12.5\ \text{GB/s} \approx 0.51$ ms per saved hop; unique to this file.
12. **`index.md` lines 98–123** — MoE forward-pass data-flow ASCII diagram; unique orientation artifact not reproduced elsewhere.

---

## MINOR Suggestions

1. **Constant restatement in file headers:** Both `token_routing_and_dispatch.md` (line 9) and `combine_and_accumulation.md` (line 11) open with "All values use Qwen3.5-35B constants: $E=256$, $k=8$, $N=8$, $E_d=32$, $H=7168$." These five numbers are already canonically defined in `index.md`'s "Model and Hardware Constants" table (lines 52–62). The per-file restatements can be shortened to a single-line cross-reference: "Constants follow `index.md` §Model and Hardware Constants."

2. **Hop-count average stated twice:** `index.md` line 151 states "average 3.0 hops across all 28 device pairs in the 1×8 mesh." `expert_placement_strategies.md` line 133 derives it fully. The `index.md` mention is fine as a bare fact (not a re-derivation), but the parenthetical "derived as…" in the locality-aware summary bullet could be trimmed since the derivation is in the placement file.

3. **$B=1$ and $B=32$ volume summary table in `index.md`:** `index.md` lines 177–181 reproduces a three-row prefill volume table ($B=1,4,32$ at $S=2048$: 29.4 / 117.4 / 939.5 MB) attributed to Chapter 3. This table is not otherwise duplicated within Chapter 5 files, but it re-presents Chapter 3 data without adding Chapter 5-specific insight. It could be replaced with a single sentence and a cross-reference to `ch03_all_to_all_num_links/all_to_all_in_moe.md`.

4. **Redundant `num_links` guidance:** `expert_placement_strategies.md` lines 266–268 and `token_routing_and_dispatch.md` lines 202–203 both give the same threshold advice ("use `num_links=1` for small decode payloads $\lesssim 1$ MB; `num_links=2` for larger payloads $\gtrsim 10$ MB"). One copy should defer to the other via cross-reference.

---

VERDICT: Crucial updates: yes

---

# Compression Analysis — T3K Mesh Chapter 5: Expert Parallelism — Pass 2

## Summary
C1, C2 applied correctly: **yes**

- **C1 confirmed:** The "Summary of Expert Placement Strategies" section (all four strategy summaries) no longer appears in `index.md`. The section has been replaced by two sentences in the "Expert Placement and Communication Summary" block (lines 127–133 of current `index.md`): one pointing to `expert_placement_strategies.md` and one pointing to `token_routing_and_dispatch.md` Section 4. No strategy prose, no $T_d = B$ derivation, no $r_e = 6$ worked example, no Zipf warning — all removed.
- **C2 confirmed:** The standalone "Key Quantitative Summary / All-to-All Volume Scaling" section with the volume formula derivation and per-$B$ byte table no longer appears in `index.md`. The only volume figures remaining in `index.md` are a brief parenthetical in the data-flow section (line 123: "approximately 3.2–6.4 MB per device at $B=1$–$32$ decode, and up to 939.5 MB per device at $B=32, S=2048$ prefill"), which is a summary reference, not a re-derivation.

## CRUCIAL Suggestions

C3. **Residual formula re-derivation in `combine_and_accumulation.md` — Volume Symmetry block (lines 51–53)**

- **Location:** `combine_and_accumulation.md` lines 51–53 — the "Volume Symmetry" subsection states the full formula $V_{\text{combine}} = V_{\text{dispatch}} = (N-1) \times C \times E_d \times H \times 2\ \text{bytes}$ and then re-evaluates it numerically at $B=32$ ($6{,}422{,}528$ bytes $\approx 6.4$ MB) and $B=1$ ($\approx 3.2$ MB).
- **Authoritative location:** `token_routing_and_dispatch.md` Section 4 "All-to-All Dispatch Communication" (lines 180–192) contains the canonical derivation and the identical numeric evaluation.
- **What to do:** Replace `combine_and_accumulation.md` lines 51–53 with: the symmetry assertion $V_{\text{combine}} = V_{\text{dispatch}}$ (one sentence) plus a cross-reference to `token_routing_and_dispatch.md` §4 for the formula and worked values. The numeric results ($\approx 6.4$ MB at $B=32$, $\approx 3.2$ MB at $B=1$) can be cited inline without re-deriving.
- **Information lost:** None. The symmetry fact is preserved; the derivation remains authoritative in `token_routing_and_dispatch.md`.

## Load-Bearing Evidence

All key unique content identified in Pass 1 remains intact and has not been disturbed by the C1/C2 removals:

1. `index.md` lines 98–123 — MoE forward-pass data-flow ASCII diagram is present and unchanged.
2. `index.md` lines 49–63 — "Model and Hardware Constants" table ($E$, $k$, $N$, $E_d$, $H$, $f_{\text{avg}}$, $C$, CF) is present and unchanged.
3. `expert_placement_strategies.md` lines 131–133 — Full hop-count derivation $\bar{h} = 84/28 = 3.0$ is present and unchanged.
4. `expert_placement_strategies.md` lines 148–168 — `build_coactivation_matrix` function is present and unchanged.
5. `expert_placement_strategies.md` lines 222–233 — `replica_device` hash-based function is present and unchanged.
6. `token_routing_and_dispatch.md` lines 94–136 — `build_send_buffer` function is present and unchanged.
7. `token_routing_and_dispatch.md` lines 226–253 — `fused_router_dispatch` pseudocode is present and unchanged.
8. `token_routing_and_dispatch.md` lines 207–209 — Prefill send buffer size derivation ($C = 2{,}560$; total $\approx 9.4$ GB) is present and unchanged.
9. `combine_and_accumulation.md` lines 89–113 — `weighted_combine` function is present and unchanged.
10. `combine_and_accumulation.md` lines 188–198 — Numerical precision bound ($\epsilon_{\text{accum}} \lesssim 3.2\%$) is present and unchanged.
11. `combine_and_accumulation.md` lines 133–136 — L1 accumulation DRAM-savings ($\Delta t_{\text{DRAM}} \approx 12.2\ \mu\text{s}$) is present and unchanged.
12. `combine_and_accumulation.md` lines 233–237 — Prefill combine latency ($t_{\text{combine}} \approx 9.4\ \text{ms}$) is present and unchanged.
13. `expert_placement_strategies.md` lines 260–264 — Locality-aware latency savings ($\Delta t \approx 0.51$ ms per saved hop) is present and unchanged.

## MINOR Suggestions

The four minor items from Pass 1 remain unaddressed and are carried forward:

1. **Constant restatement in file headers:** `token_routing_and_dispatch.md` (line 9) and `combine_and_accumulation.md` (line 11) both restate all five Qwen3.5-35B constants already canonical in `index.md`'s table. Shorten to a single cross-reference line per file.
2. **Hop-count average bare fact in `index.md`:** Line 151 (now renumbered after C1/C2 removals) states 3.0 hops without re-deriving; this is acceptable as a bare fact but the phrase structure could note it is derived in `expert_placement_strategies.md`.
3. **Prefill volume table in `index.md`:** The passing reference to 939.5 MB at $B=32, S=2048$ in the data-flow section (line 123) is now a single inline figure rather than a full table — already improved by C2 removal. No further action needed unless the inline figure is also considered redundant with Chapter 3.
4. **Redundant `num_links` threshold guidance:** `expert_placement_strategies.md` lines 266–268 and `token_routing_and_dispatch.md` lines 202–203 both give the `num_links=1` vs. `num_links=2` threshold advice verbatim. One should defer to the other by cross-reference.

VERDICT: Crucial updates: yes

---

# Compression Analysis — T3K Mesh Chapter 5: Expert Parallelism — Pass 3

## Summary
C3 applied correctly: **yes**

- **C3 confirmed:** `combine_and_accumulation.md` lines 51-53 no longer contain the re-derived formula $V_{\text{combine}} = (N-1) \times C \times E_d \times H \times 2\ \text{bytes}$ or the numeric evaluations at $B=32$ ($6{,}422{,}528$ bytes $\approx 6.4$ MB) and $B=1$ ($\approx 3.2$ MB). The Volume Symmetry block now contains only the symmetry assertion $V_{\text{combine}} = V_{\text{dispatch}}$ and a pointer to `token_routing_and_dispatch.md` Section 4 for the formula and worked examples. The authoritative derivation in `token_routing_and_dispatch.md` Section 4 is unchanged.

## CRUCIAL Suggestions

None. C3 resolved the only remaining CRUCIAL duplicate. No new formula re-derivations or exact duplicate blocks were found across the four chapter files.

## Load-Bearing Evidence

All key unique content identified in Pass 2 remains intact and has not been disturbed by the C3 removal:

1. `index.md` — MoE forward-pass data-flow ASCII diagram is present and unchanged.
2. `index.md` — "Model and Hardware Constants" table ($E$, $k$, $N$, $E_d$, $H$, $f_{\text{avg}}$, $C$, CF) is present and unchanged.
3. `expert_placement_strategies.md` — Full hop-count derivation $\bar{h} = 84/28 = 3.0$ is present and unchanged.
4. `expert_placement_strategies.md` — `build_coactivation_matrix` function is present and unchanged.
5. `expert_placement_strategies.md` — `replica_device` hash-based function is present and unchanged.
6. `token_routing_and_dispatch.md` — `build_send_buffer` function is present and unchanged.
7. `token_routing_and_dispatch.md` — `fused_router_dispatch` pseudocode is present and unchanged.
8. `token_routing_and_dispatch.md` — Prefill send buffer size derivation ($C = 2{,}560$; total $\approx 9.4$ GB) is present and unchanged.
9. `token_routing_and_dispatch.md` Section 4 — Dispatch volume formula, B=1 and B=32 worked examples, and summary table are present and unchanged (authoritative location for the volume numbers).
10. `combine_and_accumulation.md` — `weighted_combine` function is present and unchanged.
11. `combine_and_accumulation.md` — Numerical precision bound ($\epsilon_{\text{accum}} \lesssim 3.2\%$) is present and unchanged.
12. `combine_and_accumulation.md` — L1 accumulation DRAM-savings ($\Delta t_{\text{DRAM}} \approx 12.2\ \mu\text{s}$) is present and unchanged.
13. `combine_and_accumulation.md` — Prefill combine latency ($t_{\text{combine}} \approx 9.4\ \text{ms}$) is present and unchanged.
14. `expert_placement_strategies.md` — Locality-aware latency savings ($\Delta t \approx 0.51$ ms per saved hop) is present and unchanged.

## MINOR Suggestions

The four minor items from Passes 1 and 2 remain unaddressed and are carried forward:

1. **Constant restatement in file headers:** `token_routing_and_dispatch.md` (line 9) and `combine_and_accumulation.md` (line 11) both restate all five Qwen3.5-35B constants already canonical in `index.md`'s table. Shorten to a single cross-reference line per file.
2. **Hop-count average bare fact in `index.md`:** States 3.0 hops without re-deriving; acceptable as a bare fact but the phrase structure could note that the derivation is in `expert_placement_strategies.md`.
3. **Prefill volume inline figure in `index.md`:** The inline reference to 939.5 MB at $B=32, S=2048$ in the data-flow section is a single figure rather than a full table — already improved by C2. No further action required unless this figure is also considered redundant with Chapter 3.
4. **Redundant `num_links` threshold guidance:** `expert_placement_strategies.md` and `token_routing_and_dispatch.md` both give the `num_links=1` vs. `num_links=2` threshold advice verbatim. One should defer to the other by cross-reference.

VERDICT: Crucial updates: no
