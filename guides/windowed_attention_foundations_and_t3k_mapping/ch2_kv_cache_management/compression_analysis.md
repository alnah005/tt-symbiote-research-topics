# Compression Analysis: Chapter 2 — KV Cache Management During Decode — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~569 lines (index.md ~49, kv_cache_lifecycle.md ~230, circular_buffer_layout.md ~290)
- Estimated post-compression line count: ~490 lines
- Estimated reduction: ~14%

## CRUCIAL Suggestions

### [index.md] ~lines 28–41
**Issue:** The "Connection to Chapter 1" section quotes both size formulae in full and then restates in prose what those formulae mean — identical content is derived in detail in `kv_cache_lifecycle.md` lines 128–146 and introduced as vocabulary in Chapter 1. The section adds no new information beyond what the reading-order bullets already tell the reader.
**Suggestion:** Replace the entire "Connection to Chapter 1" section (lines 28–41) with a single sentence, e.g. "Chapter 2 derives the size formulae stated in Chapter 1's decode table and explains the data structure that keeps the windowed cache size constant." This cuts ~10 lines without any information loss.

### [kv_cache_lifecycle.md] ~lines 195–209
**Issue:** The "Eviction Correctness: Why No Entry Is Evicted Too Early" section formally proves `t − w < t' − w + 1` for all `t' > t`. This conclusion was already stated as a plain-English fact at lines 91–93 ("Position `t − w` ... will never again be attended to by any future query, because future queries have t' > t and their windows shift further right"). The algebraic proof adds length but not new insight for readers who accepted the prose argument; it belongs in a footnote or can be omitted entirely.
**Suggestion:** Delete the section heading and the formal proof body (lines 194–209). Optionally append a parenthetical to the existing prose at line 93: "(formally: the evicted position `t − w` is strictly below every future window's left boundary `t' − w + 1` since `t' > t`)."

### [kv_cache_lifecycle.md] ~lines 48–54
**Issue:** After stating the full-attention cache size formula (line 45), the file immediately restates it in the guise of a separate "DRAM read bandwidth" formula (lines 51–54): `BW_reads_full(g) = 2 · B · H · (P+g) · d · dtype_bytes`. This is the same expression — bandwidth in bytes equals bytes in the cache — and the only new claim is the O(T²) / O(1) characterization, which was already made in lines 12–16.
**Suggestion:** Delete the redundant bandwidth formula block (lines 51–54) and fold the bandwidth observation into the preceding sentence: "...so DRAM read bandwidth per decode step grows at the same rate as the cache itself."

### [circular_buffer_layout.md] ~lines 144–159
**Issue:** Lines 148–159 introduce a second formula for recovering the absolute token position from a slot index — `absolute_token(s) = pos_offset + ((s − write_ptr) mod w)` — and then explicitly state it is "consistent with the canonical formula shown above." This is a restatement of the same arithmetic with `write_ptr` substituted for `pos_offset` as the reference point; the derivation is circular and adds no new formula or operational step.
**Suggestion:** Delete lines 148–159 entirely. The paragraph at lines 133–142 already gives the operational definition (`pos_offset = t − w + 1`) and the formula for `position(s)`. A reader who needs `write_ptr`-based indexing can derive it in one line from what is already given.

## MINOR Suggestions

### [kv_cache_lifecycle.md] ~lines 128–146
**Issue:** The "Steady-State Size Formulae" section (lines 128–146) re-derives both size formulae under a heading that implies they are new results, but both formulae were already given inline: windowed at line 104 and full-attention at line 45. The section's only new content is the shared-factor observation at lines 145–146.
**Suggestion:** Remove the duplicate math blocks (lines 133–143). Retain the shared-factor observation and reframe the section as a summary cross-reference: "The two steady-state formulae derived above share the factor `2·B·H·d·dtype_bytes`; they differ only in the length dimension: constant `w` vs. growing `T`."

### [kv_cache_lifecycle.md] ~lines 211–225
**Issue:** The "Summary of the Lifecycle" table (lines 213–219) and the following paragraph (lines 221–225) restate the phase definitions and the `g_fill` formula that were already introduced and explained in sections above.
**Suggestion:** Keep the summary table as a navigation aid (it is genuinely useful at the end of the file) but cut the paragraph at lines 221–225, which simply repeats `g_fill = max(0, w − P)` and the "cache size never changes" observation — both already made in the body text.

### [circular_buffer_layout.md] ~lines 281–285
**Issue:** The final paragraph of "Relation to the TTNN KV Cache Update Primitive" (lines 281–285) describes what `ttnn.matmul` does and says "these steps are detailed in Chapter 4." This partially duplicates the Chapter 3 forward-reference already made at lines 112–113 and introduces a Chapter 4 reference that is not mentioned anywhere else in the chapter index.
**Suggestion:** Cut lines 281–285. The single forward-reference at line 112 already tells the reader where the read-side arithmetic is elaborated; adding a second one from a different section creates inconsistency rather than clarity.

### [index.md] ~lines 3–10
**Issue:** The opening paragraph (lines 3–10) restates vocabulary from Chapter 1 and then previews the content of the chapter — the same content that the "Reading Order" bullets immediately below cover in more precise terms.
**Suggestion:** Shorten to two sentences: one establishing the link to Chapter 1's vocabulary, one stating what Chapter 2 adds (concrete memory implications). The reading-order bullets carry the detail; the intro need not duplicate them.

## Load-Bearing Evidence

- `index.md` line ~14–24: The two reading-order bullets with their per-file descriptions — load-bearing because they are the navigational entry point for readers who want to read only one sub-file; removing them would orphan the files structurally.
- `kv_cache_lifecycle.md` line ~68–76: The `g_fill = max(0, w − P)` definition and the case where `P ≥ w` skips the fill phase — load-bearing because this edge case is not stated elsewhere and affects correctness of any implementation that starts generation with a prompt longer than the window.
- `kv_cache_lifecycle.md` line ~150–162: The numeric example table (Windowed vs. Full at T=32768 and T=131072) — load-bearing because it provides the only concrete absolute-byte figures in the chapter and grounds the T3K memory headroom argument; it is not duplicated anywhere else in the chapter.
- `kv_cache_lifecycle.md` line ~178–192: The R(T, w) = T/w derivation and the six-row table of representative saving factors — load-bearing because this is the quantitative payoff of the chapter; no other section gives these ratios.
- `circular_buffer_layout.md` line ~62–75: The worked example table for w = 4 tracing slots 0 through 8 — load-bearing because it is the only place in Chapter 2 where the wrap-around behavior is shown concretely step by step; the surrounding prose is insufficient without it.
- `circular_buffer_layout.md` line ~204–239: The two DRAM layout diagrams (before and after t=12) — load-bearing because they make the physical-vs-logical slot distinction visually concrete in a way the formulas do not; cutting either diagram would require substantially more explanatory prose.
- `circular_buffer_layout.md` line ~244–253: The comparison table (windowed circular buffer vs. full-attention grow-in-place) — load-bearing because it is the only place in Chapter 2 that puts both strategies side-by-side across all relevant properties; it consolidates information that would otherwise require cross-file reading.

## VERDICT
- Crucial updates: yes

# Compression Analysis: Chapter 2 — KV Cache Management During Decode — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~521 lines (index.md ~36, kv_cache_lifecycle.md ~206, circular_buffer_layout.md ~279)
- Estimated post-compression line count: ~497 lines
- Estimated reduction: ~5%

## CRUCIAL Suggestions
None — all Pass 1 CRUCIAL items are confirmed resolved.

## MINOR Suggestions

### [kv_cache_lifecycle.md] lines 128–138 — "Steady-State Size Formulae" duplicate math blocks
The section re-states both the windowed formula (already given at line 96) and the full-attention formula (already given at line 45) under a new heading. The only original content is the shared-factor observation at lines 137–138. The two math blocks (lines 128–136) can be dropped and the section rewritten as: "The two steady-state formulae derived above share the factor `2·B·H·d·dtype_bytes`; they differ only in the length dimension: constant `w` vs. growing `T`." Saves ~6 lines with no information loss.

## Load-Bearing Evidence

- `index.md` lines 14–24: The two reading-order bullets with per-file descriptions are the navigational entry point for readers who want to read only one sub-file; removing them would orphan the files structurally.
- `kv_cache_lifecycle.md` lines 63–68: The `g_fill = max(0, w − P)` definition and the `P ≥ w` edge-case note — the only place in the chapter where the skip-fill-phase condition is stated; it affects correctness of any implementation starting with a prompt longer than the window.
- `circular_buffer_layout.md` lines 62–85: The worked example table tracing w = 4 through slots 0–8 — the only place in Chapter 2 where wrap-around behavior is shown step by step; the surrounding prose alone is insufficient.

## VERDICT
- Crucial updates: no

## Change Log (Pass 1)

Applied 2026-03-27. All 4 CRUCIAL compression suggestions were applied verbatim. No factual content, diagrams, worked examples, or navigation footers were altered.

1. **[index.md] "Connection to Chapter 1" section replaced** — Lines 28–41 (the multi-bullet block quoting both size formulae and the two per-file forward-references) were replaced with the single sentence: "Chapter 2 derives the size formulae stated in Chapter 1's decode table and explains the data structure that keeps the windowed cache size constant."

2. **[kv_cache_lifecycle.md] Formal proof section deleted; parenthetical appended** — The "Eviction Correctness: Why No Entry Is Evicted Too Early" section heading and its formal proof body (original lines 194–209) were deleted. A parenthetical "(formally: the evicted position `t − w` is strictly below every future window's left boundary `t' − w + 1` since `t' > t`)" was appended inline to the existing prose sentence about position `t − w` never being attended to again.

3. **[kv_cache_lifecycle.md] Redundant DRAM bandwidth formula deleted** — The `BW_reads_full(g)` math block (original lines 51–54) and its introductory clause ("so DRAM read bandwidth per decode step also grows without bound:") were removed. The surrounding prose retains the plain-English observation about unbounded growth; the O() characterization is already present earlier in the file.

4. **[circular_buffer_layout.md] Second `absolute_token(s)` derivation removed** — The paragraph beginning "RoPE implementations consume this `pos_offset`... To recover the absolute token position for physical slot `s`..." along with its `absolute_token(s) = pos_offset + ((s - write_ptr) mod w)` math block and the consistency note (original lines 148–160) were deleted. The canonical formula `position(s) = pos_offset + ((s - pos_offset) mod w)` on the preceding lines is retained unchanged.
