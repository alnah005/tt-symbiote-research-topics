# Compression Analysis — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 1

## Summary
- Files reviewed:
  - `index.md`
  - `wormhole_memory_hierarchy.md`
  - `memory_config_api.md`
  - `decode_memory_strategy.md`
  - `prefill_memory_strategy.md`
- Current line count (approximate):
  - `index.md`: 105 lines
  - `wormhole_memory_hierarchy.md`: 224 lines
  - `memory_config_api.md`: 368 lines
  - `decode_memory_strategy.md`: 316 lines
  - `prefill_memory_strategy.md`: 352 lines
  - **Total: ~1,365 lines**
- Estimated post-compression: ~1,290 lines (~75 lines saved, ~5%)

---

## CRUCIAL Suggestions

**1. Duplicate all-to-all buffer sizing table in `prefill_memory_strategy.md`**

Location A: `prefill_memory_strategy.md` lines 127–135 — the "All-to-All Buffer Sizing Table" inside the "All-to-All Dispatch and Combine Buffers" section, showing B=1/4/32, S=2048, C values, and per-device buffer sizes.

Location B: `prefill_memory_strategy.md` lines 330–340 — the Summary section repeats the same formula (`V_prefill`, `C = ceil(k·B·S/E)`) and the identical three-row table (B=1, C=64, 29.4 MB; B=4, C=256, 117.4 MB; B=32, C=2048, 939.5 MB).

What to remove: Delete the duplicate formula block and table in the Summary section (lines 328–340 in `prefill_memory_strategy.md`). Replace with a one-line forward reference: "All-to-all buffer sizing formula and table: see the All-to-All Dispatch and Combine Buffers section above." The Summary table row for the all-to-all dispatch and combine buffers can retain just the size values (29.4 MB at B=1, S=2048) without re-deriving them.

Information lost: None. The derivation and all numerical values are preserved in the earlier section.

**2. Duplicate `ttnn.DRAM_MEMORY_CONFIG` equivalence code block**

Location A: `wormhole_memory_hierarchy.md` lines 106–115 — a Python snippet constructing `ttnn.MemoryConfig(INTERLEAVED, DRAM)` and asserting it equals `ttnn.DRAM_MEMORY_CONFIG`, introduced as part of the DRAM Interleaving section.

Location B: `memory_config_api.md` lines 43–47 — the `INTERLEAVED` subsection shows the identical `dram_interleaved = ttnn.MemoryConfig(...)` construction. Location C: `memory_config_api.md` lines 136–141 — the Predefined Configurations section shows `ttnn.DRAM_MEMORY_CONFIG` with a comment block giving the equivalent constructor, verbatim.

What to remove: In `wormhole_memory_hierarchy.md`, remove the full Python code block (lines 106–115). Replace with a prose sentence: "TTNN exposes this as `ttnn.DRAM_MEMORY_CONFIG`; the constructor equivalent is shown in `memory_config_api.md`." The canonical demonstration of the equivalence belongs in the API file, not the hardware overview file.

Information lost: None. The code block and equivalence are fully covered in `memory_config_api.md`.

---

## Load-Bearing Evidence

The following content is unique, technically precise, and must be kept in full:

- `wormhole_memory_hierarchy.md` — Warning block at lines 24–24 (L1 is private per-core; the 120 MB aggregate is only meaningful with explicit sharding; per-core 1.5 MB is the binding constraint). This is the single clearest statement of why the aggregate L1 figure is misleading.
- `wormhole_memory_hierarchy.md` — Warning block at lines 141–141 (CB allocation failure is a hard compile-time error with no DRAM spill fallback). Critical failure-mode explanation.
- `wormhole_memory_hierarchy.md` — The three shard layout derivations (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED) with per-core shard formulas (lines 149–172). These are the foundational geometry calculations referenced by both strategy files.
- `wormhole_memory_hierarchy.md` — The `per_core_bytes_height_sharded` utility function with the example output (lines 178–204). Unique worked example not duplicated elsewhere.
- `memory_config_api.md` — The `decode_step` canonical pattern (lines 295–337): the "one migration in, one migration out" principle with a full working code example. This is the central practical pattern of the chapter.
- `memory_config_api.md` — The `MemoryAllocationError` diagnosis steps and the try/except fallback pattern (lines 256–286). Unique operational guidance.
- `decode_memory_strategy.md` — The KV cache size derivation at lines 24–30 (full formula with $n_{\text{layers}} \times 2 \times B \times S_{\text{max}} \times n_{KV} \times H_{KV} \times 2$ = 34.4 GB example). Confirms DRAM is the only option with quantitative proof.
- `decode_memory_strategy.md` — The worked per-core L1 budget example (lines 182–258): full Python budget calculation yielding 42.5 KB / 2.8% utilization at B=32. This is the primary quantitative justification for all decode L1 recommendations.
- `decode_memory_strategy.md` — Warning at lines 44–44 (never attempt KV cache in L1; allocation that fits at step 1 will fail at step 512). The time-varying nature of the KV cache is unique reasoning not stated elsewhere.
- `decode_memory_strategy.md` — Incremental promotion strategy (lines 282–291): ordered seven-step workflow from DRAM baseline to L1 promotion. Unique actionable process.
- `prefill_memory_strategy.md` — The activation tensor size table (lines 22–29: B=1/4/32, S=512/2048/8192 with MB totals and % of chip L1). The 97.9% and 782.9% figures are the quantitative proof that DRAM is mandatory for prefill activations.
- `prefill_memory_strategy.md` — The Q/K/V crossover derivation (lines 52–67): the algebra solving for the $B \cdot S \leq 2{,}880$ threshold where Q+K+V simultaneously fit in per-core L1. This is unique decision logic.
- `prefill_memory_strategy.md` — The `attention_memory_config` function (lines 73–113): the conditional L1/DRAM selector with the worked $B=1,S=2048$ and $B=4,S=2048$ examples.
- `prefill_memory_strategy.md` — The chunked prefill section (lines 200–278): the chunk-size formula, `chunked_prefill` implementation, and the trade-off comparison table. Unique strategy content not covered in any other file.
- `prefill_memory_strategy.md` — The prefill-to-decode transition section (lines 282–312): the `get_memory_config` lookup table showing which phase uses which config for each tensor type. Unique cross-phase decision reference.
- `index.md` — The "When to Read This Chapter vs. Accepting TTNN Defaults" section (lines 25–40). Unique entry-point guidance that saves readers from reading the full chapter unnecessarily.
- `index.md` — The Chapter Notation table (lines 66–82). Canonical symbol definitions referenced throughout the chapter.

---

## MINOR Suggestions

**1. Repeated 14.0 KB/core derivation for `[32, 7168]` HEIGHT_SHARDED**

The calculation `ceil(32/80) = 1 row/core × 7168 × 2 = 14,336 bytes = 14.0 KB/core` appears three times:
- `index.md` line 52 (summary table, "Key Reason" column)
- `decode_memory_strategy.md` lines 56–58 (hidden state section)
- `decode_memory_strategy.md` lines 140–143 (all-to-all dispatch section)

The two appearances in `decode_memory_strategy.md` are each in their own tensor-analysis subsection and serve as local justifications, so they are appropriate context. The value in `index.md` is a pre-computed summary and also appropriate. No removal is needed, but the two inline derivations in `decode_memory_strategy.md` could each be shortened from four lines to one sentence ("HEIGHT_SHARDED across 80 cores yields 1 row/core = 14.0 KB/core") since the arithmetic is trivial and the full derivation is worked in the budget example that follows.

**2. Cross-file References sections partially overlap with `index.md` Reading Order**

Each sub-file ends with a References section that lists the other chapter files (e.g., `wormhole_memory_hierarchy.md` line 223 lists `ch04_memory_config/index.md`; `memory_config_api.md` lines 366–367 list both strategy files). These cross-references are useful for navigation but duplicate the Reading Order section in `index.md`. They could be reduced to a single line ("See chapter index for reading order") rather than enumerating all sibling files by name, saving 2–4 lines per file.

**3. `alltoall_buffer_bytes` helper function in `prefill_memory_strategy.md` (lines 144–147)**

This four-line function encodes the formula $C \times E_d \times H \times 2$, which is already stated in prose immediately above it and again in the summary. It is a trivial wrapper that adds lines without adding insight. It could be removed and the loop beneath it rewritten to inline the formula, or the function retained but its re-derivation in the summary deleted (covered by CRUCIAL item 1 above).

---

VERDICT: Crucial updates: yes

---

# Compression Analysis — Chapter 4: Memory Configuration: L1 vs. DRAM for Decode and Prefill — Pass 2

## Summary

C1 and C2 are both correctly applied. `prefill_memory_strategy.md` Summary section no longer duplicates the all-to-all buffer formula and three-row table; a one-sentence reference to the authoritative section replaces them. `wormhole_memory_hierarchy.md` DRAM Interleaving section no longer contains the 10-line Python equivalence block; a prose pointer to `memory_config_api.md` replaces it. All previously confirmed numerical fixes (Passes 1–6) remain intact. No regressions introduced. Estimated total line count: ~1,290 lines (~75 lines saved from Pass 1 baseline of ~1,365 lines).

---

## CRUCIAL Suggestions

None.

---

## Load-Bearing Evidence

The following unique, load-bearing content is confirmed present and intact after C1+C2:

- `wormhole_memory_hierarchy.md` — Warning block: L1 is private per-core; 120 MB aggregate is only meaningful with explicit sharding; 1.5 MB per-core is the binding constraint.
- `wormhole_memory_hierarchy.md` — Warning block: CB allocation failure is a hard compile-time error with no DRAM spill fallback.
- `wormhole_memory_hierarchy.md` — Three shard layout derivations (HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED) with ceiling operators in all formulas.
- `wormhole_memory_hierarchy.md` — `per_core_bytes_height_sharded` utility function with correct 448.0 KB annotation.
- `wormhole_memory_hierarchy.md` — DRAM Interleaving prose with round-robin tile distribution formula; pointer to `memory_config_api.md` for constructor equivalence.
- `memory_config_api.md` — `decode_step` canonical pattern ("one migration in, one migration out") with full working code example.
- `memory_config_api.md` — `MemoryAllocationError` diagnosis steps and try/except fallback pattern.
- `decode_memory_strategy.md` — KV cache size derivation: 64 × 2 × 32 × 4096 × 8 × 128 × 2 = 34.4 GB.
- `decode_memory_strategy.md` — Worked per-core L1 budget example: 42.5 KB / 2.8% utilization at B=32.
- `decode_memory_strategy.md` — Warning: never attempt KV cache in L1; allocation that fits at step 1 will fail at step 512.
- `decode_memory_strategy.md` — Incremental promotion strategy: seven-step workflow from DRAM baseline to L1 promotion.
- `prefill_memory_strategy.md` — Activation tensor size table: B=1/4/32, S=512/2048/8192, with 97.9% and 782.9% of chip L1 figures.
- `prefill_memory_strategy.md` — Q/K/V crossover derivation: algebra solving B·S ≤ 2,880 threshold for simultaneous Q+K+V L1 fit.
- `prefill_memory_strategy.md` — `attention_memory_config` function with `3 * shard_bytes <= L1_PER_CORE` condition.
- `prefill_memory_strategy.md` — All-to-all buffer sizing table (authoritative copy in "All-to-All Dispatch and Combine Buffers" section): B=1→29.4 MB, B=4→117.4 MB, B=32→939.5 MB with decimal divisor 1_000_000.
- `prefill_memory_strategy.md` — Chunked prefill section: chunk-size formula, implementation, and trade-off table.
- `prefill_memory_strategy.md` — Prefill-to-decode transition section: `get_memory_config` lookup table.
- `index.md` — "When to Read This Chapter vs. Accepting TTNN Defaults" entry-point guidance.
- `index.md` — Chapter Notation table with canonical symbol definitions.

---

## MINOR Suggestions

1. The `sharded_per_core` function in `decode_memory_strategy.md` (defined at lines 196–200) is dead code — it was used to compute the now-removed `hidden_state_per_core` variable (Pass 3 fix) and is never called anywhere else in the block. It can be deleted without any information loss. Removing it would save ~5 lines and eliminate a source of reader confusion.

2. The `alltoall_buffer_bytes` helper function in `prefill_memory_strategy.md` (lines 144–147) is a trivial four-line wrapper for the formula already stated in prose. It could be removed and the loop inlined, saving ~4 lines without any information loss.

3. Repeated 14.0 KB/core derivation (`ceil(32/80) = 1 row × 7168 × 2 = 14,336 bytes`) appears in `decode_memory_strategy.md` twice (hidden state section and dispatch buffer section). Both are contextually appropriate justifications, but each could be shortened to a single sentence since the full arithmetic is worked in the budget example that follows in the same file.

---

VERDICT: Crucial updates: no
