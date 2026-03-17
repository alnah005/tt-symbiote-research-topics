# Compression Analysis — Chapter 4: sparse_matmul for MoE — Pass 1

## Summary
- Files reviewed: `index.md`, `sparse_matmul_internals.md`, `when_sparse_matmul_wins.md`, `program_configs_sparse.md`
- Current line count: ~849 lines (breakdown: `index.md` 96, `sparse_matmul_internals.md` 231, `when_sparse_matmul_wins.md` 206, `program_configs_sparse.md` 316)
- Estimated post-compression line count: ~790 lines (~7% reduction)

---

## CRUCIAL Suggestions

**C1 — Duplicate B=1 sparsity derivation across `sparse_matmul_internals.md` §7 and `when_sparse_matmul_wins.md` §2.1 and §3.**

`sparse_matmul_internals.md` lines 193–206 derive $\rho \approx 0.031$ for the B=1 decode case from first principles (total assignments = 8, per device = 1, $C=1$, $M_t=1$, $\rho = 1/32$), then explicitly state "A detailed regime analysis is in `when_sparse_matmul_wins.md`." `when_sparse_matmul_wins.md` §2.1 lines 48–52 re-derives the exact same value with the same arithmetic ("$\rho = 1/32 \approx 0.031$ — Only 1 of 32 local expert slots is active"). §3 lines 108–112 then performs a third independent verification of the same result ("$\rho = \min(32, 1) / 32 \approx 0.031$... This matches $\rho \approx 0.031$ computed in §2.1").

The B=1 case in `sparse_matmul_internals.md` §7 (lines 193–204) is redundant with `when_sparse_matmul_wins.md` §2.1. Since `sparse_matmul_internals.md` §7 explicitly defers the full analysis to `when_sparse_matmul_wins.md`, the B=1 and B=32 sub-examples in `sparse_matmul_internals.md` §7 (lines 193–206) can be reduced to a single cross-reference sentence pointing to `when_sparse_matmul_wins.md` §2.1. The second verification in `when_sparse_matmul_wins.md` §3 (lines 108–112) is structurally justified as a consistency check for the general formula, so it should be retained.

Estimated savings: ~12 lines from `sparse_matmul_internals.md`.

**C2 — End-of-file Summary table in `when_sparse_matmul_wins.md` duplicates the §3 regime table.**

`when_sparse_matmul_wins.md` §3 (lines 116–127) contains a 9-row summary table covering all regimes (B=1, B=8, B=16, B=22, B=32 decode; short/medium/long/large prefill) with columns ($B$, $S$, $C$, $M_t$, $\rho$, Use). The end-of-file Summary (lines 184–195) is a 6-row table with columns ($\rho$, Verdict) covering 6 of the same regimes (B=1, B=8, B=16, B=22, B=32, long prefill). Every row in the Summary table is a strict subset of a row in the §3 table: the regime labels and $\rho$ values are identical, and the Verdict column adds only the words "strongly preferred / preferred / preferred / Profile both / Preferred / Preferred" — all of which are already expressed as "sparse_matmul / Either / batched matmul" in the §3 Use column.

The end-of-file Summary table (lines 184–195) can be removed and replaced with a single sentence pointing to the §3 table. The following paragraph (lines 195) describing the hybrid strategy should be retained as it adds the Chapter 6 forward reference.

Estimated savings: ~10 lines from `when_sparse_matmul_wins.md`.

---

## Load-Bearing Evidence

- **`index.md`**: The notation table (lines 28–36) defining $\rho$, $\rho^*$, $E_d$, $M_t$, $T_\text{act}$, $\alpha$ is load-bearing — these symbols are used without re-definition in all three sub-files. The Summary Comparison table (lines 41–55) comparing batched matmul vs. sparse_matmul across 10 dimensions is load-bearing as a navigational artifact unique to the index; it synthesizes information spread across three files. The Learning Objectives (lines 13–21) and Prerequisites (lines 73–79) are load-bearing scaffolding. The per-file descriptions (lines 63–65) are load-bearing navigation.

- **`sparse_matmul_internals.md`**: The `build_sparsity_mask` Python function (lines 55–87) is load-bearing executable code. The pseudocode tile-skip loop (lines 97–108) is load-bearing pedagogical code. The BFP8 byte arithmetic ($1088/2048 \approx 0.531$, multiplicative factor at $\rho=0.03$, lines 159–161) is load-bearing numerical calculation. The FLOP cost model formulas (lines 167–179) and the output-write caveat (lines 181) are load-bearing. The static-shape constraint rationale (lines 122–132) is load-bearing. The BFP8 weight tile size derivation ($32 \times 32 \times 1 + 64 = 1088$ bytes, line 153) is load-bearing.

- **`when_sparse_matmul_wins.md`**: The crossover formula $\rho^* \approx 1 - t_{\text{mask}}/t_{\text{tile}}$ (lines 136–138) is load-bearing. The 9-row Qwen3.5-35B regime table (lines 116–127) is load-bearing — it is the primary quantitative deliverable of the file. The $M_t$ cancellation derivation in §3 (lines 92–96) is load-bearing. The sequence length transition formula $B \times S > 1024$ (line 164) is load-bearing. The "When NOT to Use" table (lines 174–180) is load-bearing — it contains unique per-condition alternatives not repeated elsewhere. The B=8 and B=32 decode worked examples in §2.1 (lines 54–66) are load-bearing — only the B=1 case is duplicated in `sparse_matmul_internals.md`.

- **`program_configs_sparse.md`**: All three Python code blocks (§2.1 lines 85–115, §2.2 lines 133–153, §2.3 lines 165–185) are load-bearing — they are the primary executable artifacts of the chapter. The L1 footprint estimate in §2.1 (lines 117–123) is load-bearing numerical verification. The `validate_sparse_config` Python function (lines 247–283) is load-bearing executable code with sparse-specific checks not present elsewhere. The comparison table in §3 (lines 193–207) is load-bearing — it consolidates the Ch. 3 vs. Ch. 4 parameter differences in a single view. The summary table in §5 (lines 290–305) is load-bearing — it covers three regimes with per_core_N and mask shape together in one place not consolidated elsewhere. The SparsityConfig code snippet (lines 15–22) is load-bearing as the canonical API signature.

---

## MINOR Suggestions

**M1 — Redundant "Read X before this file" preamble sentences.**

`when_sparse_matmul_wins.md` line 5 ("Read `sparse_matmul_internals.md` before this file; the tile-skip mechanism and sparsity ratio definition are assumed known.") is a near-duplicate of the reading-order instruction already given in `index.md` lines 67–68 ("Read the files in order. `when_sparse_matmul_wins.md` assumes familiarity with the tile-skip mechanism from `sparse_matmul_internals.md`."). The instruction in `index.md` is the canonical location; the in-file repeat is stylistic bloat. The same applies to `program_configs_sparse.md` line 5–6 ("All config vocabulary... is defined in Chapter 2... The Chapter 3 prefill config... is used as the contrast baseline."), which duplicates the Prerequisites section of `index.md` lines 74–79. These opening-paragraph cross-reference sentences across both files could be condensed.

**M2 — Duplicate SparsityConfig construction across §2.1 and §2.2 of `program_configs_sparse.md`.**

`program_configs_sparse.md` §2.1 lines 99–102 define `sparsity_cfg_b1 = ttnn.SparsityConfig(mask_layout=ROW_MAJOR_LAYOUT, mask_dtype=uint8)`. §2.2 lines 141–144 define `sparsity_cfg_b8` with identical body. The comment in §2.2 ("same per-expert shape as B=1") already acknowledges the identity. Since the `SparsityConfig` is shown once in §1.1 and once in §2.1, the second instantiation in §2.2 is redundant. The §2.2 code block could simply reference `sparsity_cfg_b1` or remove the `SparsityConfig` object entirely and note it is unchanged. This is minor: the repetition illustrates the static-config point, so it is borderline pedagogical — flag only for consideration.

**M3 — Verbose "placeholder / UNVERIFIED" warnings repeated identically in §2.1, §2.2, §2.3, §3 comparison table, and §5 summary table of `program_configs_sparse.md`.**

The caveat "placeholder; replace with N_t/4 once D known [UNVERIFIED]" (or equivalent) appears at least 8 times across `program_configs_sparse.md` (inline comments on lines 94, 97, 108, 110, 120, 121, 125, 139, 140, 150, 152, 180, 182, 200, 298, 299, 305). The first occurrence in §2.1 with the explicit Warning callout (lines 125–126) is load-bearing; subsequent occurrences in §2.2, §2.3, §3, and §5 are redundant reminders. A single consolidated UNVERIFIED notice at the top of §2 (or a footnote) would reduce character count while preserving the warning semantics. This is stylistic bloat that adds visual noise without new information after the first occurrence.

**M4 — Repeated constraint block carried verbatim across chapters.**

`program_configs_sparse.md` §1.2 (lines 30–37) reproduces the six divisibility constraints (`out_subblock_h × out_subblock_w ≤ 8`, etc.) with the note "carry over unchanged" from Chapter 2. Since the file explicitly defers to Chapter 2 for derivation, listing all six constraints here is redundant with the source. A cross-reference sentence ("All Chapter 2 divisibility constraints from `matmul_fundamentals_in_ttnn.md` §3 apply unchanged") would suffice. However, the constraints are short and aid recall — this is a minor stylistic call.

---

VERDICT: Crucial updates: yes

---

## Change Log — Pass 1 Fixes Applied

- C1 applied: Removed the B=1 and B=32 decode worked sub-examples from `sparse_matmul_internals.md` §7 (formerly lines 193–206: the four-bullet B=1 breakdown, the three-bullet B=32 breakdown, and the concluding two-sentence paragraph restating ρ=0.03 and ρ=1). Replaced with a single sentence retaining the M_t cancellation note and adding a cross-reference to `when_sparse_matmul_wins.md` §2 for the full regime analysis including all worked examples. The ρ formula definition (`ρ = active expert slots / E_d`) at line 191 was preserved unchanged.
- C2 applied: Removed the 6-row duplicate Summary table from `when_sparse_matmul_wins.md` (formerly the table under the `## Summary` heading covering regimes B=1 through long prefill with ρ and Verdict columns). Replaced with a single sentence pointing to the §3 regime table as the authoritative consolidated reference. The hybrid strategy sentence ("The practical hybrid strategy: use `sparse_matmul` for decode...") was preserved unchanged.

---

# Compression Analysis — Chapter 4: sparse_matmul for MoE — Pass 2

## Summary
- Files reviewed: `index.md`, `sparse_matmul_internals.md`, `when_sparse_matmul_wins.md`, `program_configs_sparse.md`
- Current line count: ~825 lines (breakdown: `index.md` 95, `sparse_matmul_internals.md` 217, `when_sparse_matmul_wins.md` 198, `program_configs_sparse.md` 315)
- Estimated post-compression: ~800 lines (~3% reduction from current; ~6% from original Pass 1 baseline)

---

## Pass 1 Item Verification

**C1 — CONFIRMED CORRECT.** `sparse_matmul_internals.md` §7 (lines 185–193) no longer contains the B=1 or B=32 decode sub-examples. The section now ends with a single sentence noting the $M_t$ cancellation and cross-referencing `when_sparse_matmul_wins.md` §2 for the full worked-example analysis. The `ρ = active expert slots / E_d` formula definition at line 191 is preserved. No residual duplication; no load-bearing content removed.

**C2 — CONFIRMED CORRECT.** `when_sparse_matmul_wins.md` Summary section (lines 184–188) no longer contains the 6-row regime/verdict table. The section now consists of one pointer sentence to the §3 table and the hybrid strategy sentence with the Chapter 6 forward reference. No residual duplication; no load-bearing content removed.

---

## CRUCIAL Suggestions

None.

---

## Load-Bearing Evidence

- **`index.md`**: The notation table (lines 28–36) defining $\rho$, $\rho^*$, $E_d$, $M_t$, $T_\text{act}$, $\alpha$ is load-bearing — these symbols are used without re-definition in all three sub-files. The Summary Comparison table (lines 41–55) synthesizing batched vs. sparse across 10 dimensions is load-bearing as the chapter's unique navigational overview. Learning Objectives and Prerequisites are load-bearing scaffolding.

- **`sparse_matmul_internals.md`**: The `build_sparsity_mask` Python function (lines 55–87) is load-bearing executable code. The pseudocode tile-skip loop (lines 97–108), the BFP8 byte arithmetic (lines 153, 159–161), the FLOP cost model formulas (lines 167–179), and the output-write caveat (line 181) are all load-bearing. The static-shape constraint rationale (§4.1–4.2, lines 118–132) is load-bearing conceptual explanation.

- **`when_sparse_matmul_wins.md`**: The 9-row Qwen3.5-35B regime table (§3, lines 116–127) is the primary quantitative deliverable and is load-bearing. The crossover formula $\rho^* \approx 1 - t_\text{mask}/t_\text{tile}$ (§4, lines 136–138), the $M_t$ cancellation derivation (§3, lines 92–96), the sequence-length transition formula $B \times S > 1024$ (§5, line 164), and the "When NOT to Use" condition table (§6, lines 174–180) are all load-bearing.

- **`program_configs_sparse.md`**: All three Python config code blocks (§2.1–2.3), the `SparsityConfig` API snippet (§1.1), the L1 footprint estimate (§2.1 lines 117–123), the `validate_sparse_config` function (§5 lines 247–283), the Ch.3-vs-Ch.4 comparison table (§3 lines 193–207), and the §5 summary table (lines 290–305) are all load-bearing executable or consolidating artifacts.

---

## MINOR Suggestions

**M1** (carried from Pass 1 — unaddressed): Redundant "Read X before this file" preamble sentences. `when_sparse_matmul_wins.md` line 5 repeats the reading-order instruction already given in `index.md` lines 67–68. `program_configs_sparse.md` lines 5–6 repeat the prerequisite cross-references already given in `index.md` lines 74–79. Condensing these to a single short line per file would save 2–4 lines with no information loss.

**M2** (carried from Pass 1 — unaddressed): Duplicate `SparsityConfig` construction in `program_configs_sparse.md` §2.1 (lines 99–102) and §2.2 (lines 141–144). The two objects are body-identical; §2.2 could reference the §2.1 object or omit the re-declaration, saving 4 lines. Borderline pedagogical — flag for consideration only.

**M3** (carried from Pass 1 — unaddressed): The `[UNVERIFIED]` / "placeholder; replace with N_t/4 once D known" warning is repeated in inline comments and table cells at least 8 times across `program_configs_sparse.md` (§2.1 comments, §2.1 Warning callout, §2.2 comments, §2.3 comments, §3 table rows, §5 summary table rows). The §2.1 Warning callout (lines 125–126) is the canonical first occurrence; all subsequent inline repetitions are redundant reminders. A single consolidated note at the top of §2 referencing the Warning callout would reduce visual noise.

**M4** (carried from Pass 1 — unaddressed): `program_configs_sparse.md` §1.2 (lines 29–37) reproduces the six Chapter 2 divisibility constraints verbatim with the note "carry over unchanged." Since the file explicitly defers derivation to Chapter 2, a single cross-reference sentence would suffice. The constraints are short and aid recall, so this is a low-priority stylistic call.

**M5** (new): `sparse_matmul_internals.md` §4.3 "Implication for MoE Deployment" (lines 134–144) and `program_configs_sparse.md` §4 (lines 218–238) both state the same $C_\text{fixed}$ formula and evaluate it for the same example ($B_\text{max}=32$, $k=8$, $E=256$, $C=1$). The `program_configs_sparse.md` §4 version is strictly more detailed — it adds the per-$B$ mask bit-count breakdown and multi-$C$ config-cache guidance not present in `sparse_matmul_internals.md`. The `sparse_matmul_internals.md` §4.3 block (11 lines) could be condensed to 2–3 lines (state the constraint, give the formula, cross-reference `program_configs_sparse.md` §4 for the worked deployment example), saving ~8 lines. No load-bearing content would be lost because the canonical numeric example lives in `program_configs_sparse.md` §4.

---

VERDICT: Crucial updates: no
