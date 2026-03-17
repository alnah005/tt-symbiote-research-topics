# Compression Analysis — Chapter 6: Performance Analysis and Trade-offs

## Crucial updates: yes

---

### Duplication 1: Bandwidth efficiency tables (interleaved and sharded)

**Source (original):** `ch04_prefetch_patterns_and_bandwidth/bandwidth_estimation.md`, lines 49–66

```
| Workload | Theoretical Peak | Measured (Interleaved) | Efficiency |
|---|---|---|---|
| Full-grid matmul, large K | 300 GB/s | 180–240 GB/s | 60–80% |
| Decode (M=1), small batch | 300 GB/s | 160–200 GB/s | 53–67% |
...
| Layout | Effective Bandwidth | vs. Interleaved |
|---|---|---|
| Interleaved | ~180–200 GB/s (decode) | baseline |
| WIDTH_SHARDED, DRAM | ~260–290 GB/s (decode) | +30–50% |
```

**Duplicate:** `ch06_performance_analysis_and_tradeoffs/bandwidth_gain_analysis.md`, lines 19–23 and 39–43

The ch06 tables are nearly identical in structure and figures. The only differences are cosmetic: column header wording ("Indicative Peak" vs "Theoretical Peak", "Indicative Effective (Interleaved)" vs "Measured (Interleaved)") and a minor numeric tightening (~255–285 vs ~260–290 GB/s). These tables were already fully established in ch04 and are reproduced in ch06 without adding new analytical content.

**Recommended action:** Replace the ch06 tables with a single cross-reference sentence pointing to `ch04/bandwidth_estimation.md`. If a brief recap is needed for flow, retain one summary row (e.g., "~180–200 GB/s interleaved → ~255–285 GB/s sharded, decode regime") inline in prose, not as a full duplicated table.

---

### Duplication 2: GDDR6 residual-gap explanation (three-bullet list)

**Source (original):** `ch04_prefetch_patterns_and_bandwidth/bandwidth_estimation.md`, lines 67–70

```
The residual gap from 300 GB/s is due to:
- GDDR6 bank activation overhead (row precharge cycles between non-contiguous accesses).
- L1 write latency after the NoC packet arrives.
- DMA engine scheduling overhead between shard prefetch requests.
```

**Duplicate:** `ch06_performance_analysis_and_tradeoffs/bandwidth_gain_analysis.md`, lines 46–48

The ch06 version reproduces the same three bullet-point causes in the same order with near-identical wording (only "row precharge cycles between non-contiguous accesses within a bank" slightly expanded from ch04's shorter form). The explanation belongs in ch04 where the DRAM model is built; ch06 can cite it.

**Recommended action:** Remove the three-bullet explanation from ch06 `bandwidth_gain_analysis.md` lines 46–48. Replace with: "The remaining gap from peak has three hardware-level causes explained in `ch04/bandwidth_estimation.md` (GDDR6 bank activation, L1 write latency, DMA scheduling overhead)."

---

### Duplication 3: Arithmetic intensity derivation and prefill calculation

**Source (original):** `ch04_prefetch_patterns_and_bandwidth/bandwidth_estimation.md`, lines 74–114

Full derivation of the arithmetic intensity formula `M * N / (N + M)`, decode regime result (~1 FLOP/byte), and prefill regime calculation (M=512 → ~494 FLOP/byte), all derived from the roofline ridge point of ~437 FLOP/byte.

**Duplicate:** `ch06_performance_analysis_and_tradeoffs/bandwidth_gain_analysis.md`, lines 56–98

Ch06 re-derives the same formula step-by-step (`arithmetic_intensity ≈ effective_M × N / (N + effective_M)`), recomputes the decode case (~1 FLOP/byte), and recomputes the prefill case (M=512 → ~494 FLOP/byte). The only addition in ch06 is substituting `effective_M = batch_size × top_k` as the variable name — a renaming, not new content.

The ridge-point crossover calculation for `effective_M_crossover` (lines 128–134 of `bandwidth_gain_analysis.md`) is the only genuinely new analytical step in this section; it was not in ch04 and is ch06's original contribution.

**Recommended action:** Trim `bandwidth_gain_analysis.md` lines 56–98 to a brief recap paragraph that states the key results from ch04 (decode: ~1 FLOP/byte; prefill M=512: ~494 FLOP/byte; ridge point: ~437 FLOP/byte) with a cite to `ch04/bandwidth_estimation.md`. Retain the crossover derivation (lines 128–134) in full — it is load-bearing and new to ch06.

---

### Duplication 4: Key Constants (Wormhole B0) table in `index.md`

**Source (original):** `ch04_prefetch_patterns_and_bandwidth/index.md`, lines 79–89

```
| Parameter | Value |
|---|---|
| DRAM controllers | 6 |
| GDDR6 banks | 12 (2 per controller) |
| Peak DRAM bandwidth | ~300 GB/s |
| Tensix cores | 80 (8×10 grid) |
| L1 per core | 1.5 MB |
| BF16 tile size | 2,048 bytes (32×32×2) |
| Peak compute | ~131 TFLOP/s (BF16) |
| Ridge point | ~437 FLOP/byte |
```

**Duplicate:** `ch06_performance_analysis_and_tradeoffs/index.md`, lines 63–73

The ch06 table reproduces seven of the eight rows verbatim (BF16 tile size is omitted; "Aggregate peak DRAM bandwidth" and "Tensix core grid" use slightly different label wording). The values are identical. This table was introduced in ch04 as the definitive constant reference; ch01 also establishes several of these constants. Repeating it in ch06 creates a maintenance risk (values that must be updated in multiple places).

**Recommended action:** Replace the ch06 Key Constants table with a one-line note: "For Wormhole B0 constants (DRAM controllers, bandwidth, core grid, ridge point), see `ch04/index.md`." If the ridge point value alone is needed for context in the ch06 overview, cite it inline ("the ~437 FLOP/byte ridge point from Chapter 4").

---

## Load-Bearing Evidence

Not applicable (crucial duplications were found above). The following ch06 content is genuinely new and must be retained:

- `index.md` Decision Table (lines 29–38): regime-to-layout mapping with latency deltas — not present in prior chapters.
- `bandwidth_gain_analysis.md` crossover derivation (`effective_M_crossover` formula, lines 128–134) and model-specific threshold table (lines 140–145).
- `bandwidth_gain_analysis.md` Summary table by regime (lines 151–158): consolidates decode/prefill regimes with effective_M ranges — new synthesized view.
- `shard_setup_overhead.md` in its entirety: load-time vs inference-time resharding analysis, reshard latency estimate, program cache stability rules, and dual-config warm-up pattern. None of this material appears in prior chapters.
- `tradeoff_matrix.md` model-specific regime boundary tables for Mixtral and Qwen (lines 26–47).
- `tradeoff_matrix.md` "When Sharding Hurts" section (lines 51–68): conditions 1–3 synthesize prior content into actionable negative-case guidance not stated elsewhere.
- `tradeoff_matrix.md` L1 memory pressure interaction table and Mode 1/Mode 2 analysis (lines 71–112).
- `tradeoff_matrix.md` T3K multi-chip compounding section (lines 115–141): per-chip token-count analysis is new to ch06.
- `tradeoff_matrix.md` Layout Selection Flowchart (lines 146–161).

---

## MINOR Suggestions

1. **`bandwidth_gain_analysis.md`, line 62:** The decode arithmetic intensity result is stated as "~1 FLOP/byte" inline, but the derivation in ch04 already shows this. A forward pointer ("as derived in `ch04/bandwidth_estimation.md`") would suffice without reproduced working.

2. **`bandwidth_gain_analysis.md`, lines 64–79 (Mixtral weight-bytes calculation):** The specific weight_bytes arithmetic for Mixtral is also present implicitly in ch04 `bandwidth_estimation.md` lines 17–18 and `shard_setup_overhead.md` lines 147–148. These three locations now carry the same `4096 × 14336 × 2 = 117,440,512 bytes` calculation. Consider a shared "Mixtral Weight Footprint" note in ch03 and citing it everywhere else.

3. **`tradeoff_matrix.md`, "When Sharding Hurts" condition 3 (lines 65–68):** References `ch05/common_pitfalls.md` Pitfall 3 accurately and does not duplicate it — this cross-reference pattern is the correct approach and should be followed for the other duplications identified above.

4. **`shard_setup_overhead.md`, lines 169–172 (total_reshard_time calculation):** States "24 weight tensors per layer (8 experts × 3 projections each), totaling 768 tensors across 32 layers" but the arithmetic is `8 × 3 = 24` per layer × 32 layers = 768; the preceding sentence also says "32 × 8 × 3 = 768 resharding calls per decode step" which is consistent but the label "per decode step" vs "across 32 layers" could be clarified — this is not a duplication but a minor consistency note.

5. **`index.md` note box (line 11) and `bandwidth_gain_analysis.md` note box (lines 9):** Both carry nearly identical disclaimer text about bandwidth figures being indicative and pointing to Chapter 7. Consider keeping only the one in `bandwidth_gain_analysis.md` (where the figures appear) and removing it from `index.md`.

## Agent A Change Log — C Feedback Pass 1
- bandwidth_gain_analysis.md: Replaced bandwidth efficiency tables with Chapter 4 cross-reference
- bandwidth_gain_analysis.md: Replaced GDDR6 3-bullet explanation with 1-sentence summary + Chapter 4 cross-reference
- bandwidth_gain_analysis.md: Collapsed arithmetic intensity derivation; kept crossover formula; added Chapter 4 derivation cross-reference
- index.md: Replaced Key Constants table with Chapter 4 cross-reference

## Pass 2 Verification

**Fix 1 — Bandwidth efficiency tables:** CONFIRMED. The interleaved bandwidth table (Regime/Indicative Peak/Indicative Effective/Efficiency rows) and the sharded bandwidth table (Layout/Indicative Effective Bandwidth/vs. Interleaved rows) have both been removed from `bandwidth_gain_analysis.md`. Replaced with the single sentence: "See Chapter 4, `bandwidth_estimation.md` for the interleaved and sharded bandwidth tables." at line 17. The surrounding prose (interleaved round-robin contention explanation, DRAM-sharded fan-out elimination explanation) is retained as it provides ch06-specific framing.

**Fix 2 — GDDR6 residual-gap 3-bullet explanation:** CONFIRMED. The three-bullet list (GDDR6 bank activation overhead, L1 write latency, DMA engine scheduling) has been removed. Replaced with the single sentence at line 32: "The residual gap arises from bank activation overhead, L1 write latency, and DMA scheduling overhead; see Chapter 4, `bandwidth_estimation.md` for the full explanation." The Tip block (shard height guidance) is retained as it is ch06-original actionable guidance.

**Fix 3 — Arithmetic intensity derivation:** CONFIRMED. The foundational derivation steps have been removed:
- The `arithmetic_intensity (decode, M=1) ≈ 1 FLOP/byte` code block has been replaced by the one-sentence summary at line 40: "Arithmetic intensity = M×N/(M+N) (derived in Chapter 4); at decode, intensity ≈ 1 FLOP/byte; at large prefill, intensity approaches N FLOP/byte."
- The `arithmetic_intensity (prefill, M=512) ≈ 494 FLOP/byte` code block has been removed; the prefill section now cites ch04 inline at line 67.
- The arithmetic intensity formula re-statement and two example substitutions (effective_M=16 → 15.98 FLOP/byte; effective_M=512 → 494 FLOP/byte) in the Rule of Thumb section have been removed.
- The ch06-original crossover formula (`effective_M_crossover ≈ 437 × N / (N − 437)`) and the Mixtral/Qwen threshold table are fully retained at lines 87–99.

**Fix 4 — Key Constants table in index.md:** CONFIRMED. The 7-row hardware constants table has been removed from `index.md`. Replaced with the single line at line 64: "Hardware constants (DRAM bandwidth, ridge point, tile sizes): see Chapter 4, `index.md`." The subsequent Chapter 4 derivation cross-reference that was already present has been absorbed into this replacement.

### Remaining Crucial Duplications Check

One borderline item was examined: the Mixtral weight_bytes calculation (`4096 × 14336 × 2 = 117,440,512 bytes`) at lines 46–56 of `bandwidth_gain_analysis.md` also appears in ch04 `bandwidth_estimation.md` line 18. However, this is not a CRUCIAL duplication warranting removal because the ch06 version extends the calculation to `total_weight_bytes` (2 experts × 3 projections × 112 MB = 672 MB) and derives ch06-original latency estimates (3.4 ms vs 2.5 ms at interleaved vs sharded bandwidth). The raw weight size figure is a shared constant; the per-layer, per-step latency impact analysis is new to ch06. This was already flagged as MINOR suggestion 2 in the original analysis.

No remaining crucial duplications found.

## Crucial updates: no

## Load-Bearing Evidence

All 4 fixes were straightforward removals of content that was already fully established in Chapter 4. The retained ch06-original content is load-bearing:

- `bandwidth_gain_analysis.md` lines 21–24: Two-point inefficiency analysis (cross-controller fan-out + NoC hotspot formation) — synthesizes ch04 NoC model into a direct explanation of the interleaved-layout penalty at M=1; not a verbatim repeat of ch04.
- `bandwidth_gain_analysis.md` lines 44–64 (decode section, weight bytes + latency calculation): The per-layer latency impact (3.4 ms vs 2.5 ms) is ch06-original and not present in ch04.
- `bandwidth_gain_analysis.md` lines 84–101 (crossover formula + model-specific threshold table): Entirely ch06-original.
- `bandwidth_gain_analysis.md` lines 105–113 (Summary table): Synthesized regime-by-regime sharding gain table — not present in ch04.
- `index.md` lines 29–38 (Decision Table): Regime-to-layout mapping with latency deltas — not in prior chapters.

## MINOR Suggestions

1. **`bandwidth_gain_analysis.md` line 40 (arithmetic intensity summary):** The prescribed one-liner states "at large prefill, intensity approaches N FLOP/byte." For Mixtral N=14336, this is numerically correct but may be confusing since FLOP/byte is a dimensionless ratio and saying it "approaches N" conflates the formula variable N (weight dimension) with the unit. Consider rewording to: "at large prefill (M >> 1), intensity approaches N FLOP/byte where N is the weight output dimension."

2. **`bandwidth_gain_analysis.md` lines 46–56 (Mixtral weight_bytes):** The `4096 × 14336 × 2 = 117,440,512 bytes` calculation appears in both ch04 `bandwidth_estimation.md` line 18 and ch06. As noted in the original Minor Suggestion 2, this is a candidate for a shared "Mixtral Weight Footprint" reference in ch03. Not critical to address now, but worth tracking.

3. **`bandwidth_gain_analysis.md` lines 1–7 (intro paragraph):** The statement "Wormhole B0 has 6 DRAM controllers, each driving 2 GDDR6 banks" restates a hardware constant that now cross-references ch04 `index.md` via the Fix 4 change to `index.md`. This sentence could itself be replaced with a ch04 cross-reference, but the duplication is minor (one sentence of context framing, not a table or derivation).
