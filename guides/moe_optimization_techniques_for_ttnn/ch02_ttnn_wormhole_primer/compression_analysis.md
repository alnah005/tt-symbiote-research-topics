# Compression Analysis — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 1

## Summary
- Files reviewed: index.md, wormhole_architecture.md, ttnn_programming_model.md, matmul_fundamentals_in_ttnn.md
- Current line count: ~926 lines (total across all files)
- Estimated post-compression line count: ~820 lines

---

## CRUCIAL Suggestions

**C1 — Duplicate tile-size rows across wormhole_architecture.md and ttnn_programming_model.md summary tables**

`wormhole_architecture.md` Summary table (lines 179–180):
```
| bfloat16 tile size | 2 KB (32×32 × 2 bytes) |
| bfloat8_b tile size | 1088 bytes (1024 value bytes + 64 exponent bytes) |
```
`ttnn_programming_model.md` Summary table (lines 243–244):
```
| BF16 tile size | 2 KB (32×32 × 2 bytes) |
| BFP8 tile size | 1088 bytes (1024 value bytes + 64 exponent bytes) |
```
These two rows are identical facts in adjacent-file summary tables. Dtypes are formally defined in `ttnn_programming_model.md` Section 1.2, so that file is the correct home for these rows. Remove both rows from `wormhole_architecture.md`'s summary table and add a single cross-reference line: "Tile sizes: see `ttnn_programming_model.md` Section 1.2."

**C2 — BFP8 tile byte breakdown re-derived in matmul_fundamentals_in_ttnn.md after full definition in ttnn_programming_model.md**

`ttnn_programming_model.md` Section 1.2 gives the full authoritative BFP8 structure (1024 value bytes + 64 exponent bytes = 1088 bytes, one shared exponent per 16-value block). The inline comment in `matmul_fundamentals_in_ttnn.md` lines 99–104 re-explains the same breakdown in a long parenthetical:

```
(substitute tile_size = 2048 bytes for BF16 tensors and 1088 bytes for BFP8 tensors
 per buffer. For a mixed-dtype config with BFP8 weights and BF16 activations/output — the canonical
 MoE production pattern — only the B buffer uses 1088 bytes/tile; A and C buffers remain at
 2048 bytes/tile. For a pure BFP8 config where all three tensors are BFP8, all three buffers
 use 1088 bytes/tile.)
```

The load-bearing content is which buffer uses which byte count in mixed vs. pure BFP8 configs. That nuance is worth keeping but can be expressed in one line: "Use 2048 bytes/tile for BF16 and 1088 bytes/tile for BFP8; in the canonical mixed config (BFP8 weights, BF16 activations/output), only the B buffer uses 1088 bytes/tile." The current 6-line parenthetical can be replaced with that single line, saving ~4 lines.

**C3 — Interleaved/sharded concepts pre-defined in index.md Key Terms then re-defined in ttnn_programming_model.md Section 2.3**

`index.md` lines 69–70 define:
```
- **Interleaved placement** — tensor pages striped across multiple banks
- **Sharded placement** — tensor slices pinned to specific cores' L1
```
`ttnn_programming_model.md` Section 2.3 lines 110–114 re-state and expand these same definitions with identical wording. A reader proceeding in order encounters the full Section 2.3 definition before they would return to the Key Terms list. The `index.md` Key Terms section is a glossary stub, not the authoritative definition point; having it duplicate content from the file that defines those terms adds pure redundancy. Collapse each of the two Key Terms bullets to a pointer: "**Interleaved placement** — see `ttnn_programming_model.md` Section 2.3" and "**Sharded placement** — see `ttnn_programming_model.md` Section 2.3."

---

## Load-Bearing Evidence

- **index.md** — Lines 17–23 (Learning Objectives 1–7): Each objective names a specific concept and the file where it is established, giving the reader their entry contract and serving as the assumed-knowledge list for Chapter 3. These cannot be cut.

- **wormhole_architecture.md** — Lines 154–169 (Section 4.3, ring all-reduce formula and 16 MB transfer-volume calculation): These are quantitative, non-obvious results that Chapters 6 and 7 use for communication-overlap analysis. They do not appear elsewhere in the chapter and cannot be removed.

- **ttnn_programming_model.md** — Lines 194–217 (tracing pattern with `ttnn.begin_trace_capture` / `ttnn.execute_trace` and the MoE-specific constraint list): The code block and accompanying constraints (static shapes, routing outside the trace, fixed-size token padding) appear nowhere else and constitute the sole explanation of this execution mode in the chapter. Cannot be cut.

- **matmul_fundamentals_in_ttnn.md** — Lines 344–374 (Section 5 worked example, final `MatmulMultiCoreReuseMultiCastProgramConfig` with `WormholeComputeKernelConfig`): The only end-to-end configuration instantiation in the chapter, explicitly forward-referenced by Chapter 3. It synthesizes all parameters from Sections 1–4 and cannot be cut.

---

## MINOR Suggestions

**M1 — "Next Steps" sections restate index.md navigation**

Each of the three content files ends with a "Next Steps" block that names the next file and previews its contents. `wormhole_architecture.md` lines 190–192, `ttnn_programming_model.md` lines 253–255, and `matmul_fundamentals_in_ttnn.md` lines 399–401 all cover navigation already captured in the `index.md` Files table. Each could be collapsed to a single sentence ("Proceed to [next file]."), saving 2–3 lines per file (6–9 lines total).

**M2 — Code comment in matmul_fundamentals_in_ttnn.md line 193 restates what the code already shows**

```python
# Example: per_core_M=2, per_core_N=4 → out_subblock_h=2, out_subblock_w=4 (product=8, max efficiency)
```
The parameter values are immediately visible from the named arguments in lines 195–203. The comment can be trimmed to: `# per_core_M=2, per_core_N=4: max valid subblock (product=8)` — or dropped, since the surrounding prose in Section 3.4 already explains the selection reasoning.

**M3 — index.md "Next Steps" section duplicates the Files table reading-order note**

`index.md` line 41 already states "Read the files in order. matmul_fundamentals_in_ttnn.md builds on concepts from both preceding files." The two-paragraph "Next Steps" section at lines 74–78 restates this order and adds a Chapter 3 forward reference. The Chapter 3 forward reference is worth keeping, but it could be appended to the existing line 41 sentence rather than occupying a separate heading and two-paragraph block, saving ~4 lines.

**M4 — Hedging parenthetical in wormhole_architecture.md tip at line 150 adds no precision**

The tip states expert parallelism "generally outperforms intra-expert tensor parallelism for standard MoE configurations (8–64 experts, d_model up to 8192)." The parenthetical range is not a constraint — the tip's own next sentence names the actual exception condition (very large d_ff / DRAM capacity). The parenthetical can be dropped without narrowing the claim.

---

VERDICT: Crucial updates: yes

---

## Change Log — Pass 1 Fixes Applied

- C1 applied: Removed duplicate tile-size rows (BF16 2 KB, BFP8 1088 bytes) from `wormhole_architecture.md` summary table; replaced with cross-reference to `ttnn_programming_model.md` Section 1.2.
- C2 applied: Compressed 6-line BFP8 parenthetical in `matmul_fundamentals_in_ttnn.md` to single line preserving both mixed-dtype and pure-BFP8 cases.
- C3 applied: Replaced inline "Interleaved placement" and "Sharded placement" definitions in `index.md` Key Terms with pointers to `ttnn_programming_model.md` Section 2.3.

---

# Compression Analysis — Chapter 2: TTNN and Wormhole Hardware Primer — Pass 2

## Summary
- Files reviewed: index.md, wormhole_architecture.md, ttnn_programming_model.md, matmul_fundamentals_in_ttnn.md
- Current line count: ~921 lines (total across all files)
- Estimated post-compression line count: ~910 lines

## Pass 1 Item Verification

**C1 — ADDRESSED.** `wormhole_architecture.md` summary table line 179 now reads `| Tile sizes | See ttnn_programming_model.md Section 1.2 |`. The two duplicate tile-size rows (BF16 2 KB, BFP8 1088 bytes) have been removed and replaced with the single cross-reference line. `ttnn_programming_model.md` summary table retains the authoritative BF16 (2 KB) and BFP8 (1088 bytes) rows at lines 243–244. No further action needed.

**C2 — ADDRESSED.** `matmul_fundamentals_in_ttnn.md` lines 99–100 now contain a single compressed inline comment: `(Use 2048 bytes/tile for BF16 and 1088 bytes/tile for BFP8; in the canonical mixed config — BFP8 weights, BF16 activations/output — only the B buffer uses 1088 bytes/tile; all three use 1088 bytes/tile for pure BFP8.)` The original 6-line parenthetical is gone. No further action needed.

**C3 — ADDRESSED.** `index.md` lines 69–70 now read:
```
- **Interleaved placement** — see `ttnn_programming_model.md` Section 2.3
- **Sharded placement** — see `ttnn_programming_model.md` Section 2.3
```
The inline definitions have been replaced with pointers. No further action needed.

## CRUCIAL Suggestions

None

## Load-Bearing Evidence

- **index.md** — Lines 17–23 (Learning Objectives 1–7): Seven numbered objectives each name a precise concept and its defining file. This is the reader's entry contract for the chapter and the assumed-knowledge checklist for Chapter 3. Removing or collapsing any objective would silently drop a prerequisite claim. Cannot be cut.

- **wormhole_architecture.md** — Lines 154–169 (Section 4.3): The ring all-reduce bandwidth formula (`8 / (2×7) × 12.5 GB/s ≈ 7.1 GB/s`) and the 16 MB per-forward-pass transfer-volume derivation are quantitative anchor points cited in Chapters 6 and 7 for communication-overlap budgeting. They do not appear elsewhere in the chapter. Cannot be removed.

- **ttnn_programming_model.md** — Lines 194–217 (Section 3.2 tracing block): The `ttnn.begin_trace_capture` / `ttnn.execute_trace` code example and the accompanying three-bullet MoE constraint list (static shapes, routing outside trace, fixed token padding) are the sole explanation of trace-mode execution in the chapter. No other file repeats this content. Cannot be cut.

- **matmul_fundamentals_in_ttnn.md** — Lines 286–295 (Section 4.3 decision flowchart): The two-level ASCII decision tree for choosing between `MatmulMultiCoreReuseMultiCastProgramConfig` and `MatmulMultiCoreProgramConfig` is the only explicit decision aid in the chapter. Section 4.1 and 4.2 prose describes each config but does not give the switching logic in a scannable form. The flowchart is load-bearing for fast reference. Cannot be cut.

## MINOR Suggestions

**M1 (carry-forward, not yet applied) — "Next Steps" sections in the three content files restate index.md navigation.** `wormhole_architecture.md` lines 190–192, `ttnn_programming_model.md` lines 253–255, and `matmul_fundamentals_in_ttnn.md` lines 393–397 each name the next file and preview its contents, duplicating what the `index.md` Files table already captures. Each could be collapsed to a single sentence, saving 6–9 lines total.

**M2 (carry-forward, not yet applied) — Redundant code comment at matmul_fundamentals_in_ttnn.md line 189.** The comment `# Example: per_core_M=2, per_core_N=4 → out_subblock_h=2, out_subblock_w=4 (product=8, max efficiency)` immediately precedes a code block where all four values are visible as named arguments. The comment can be shortened to `# per_core_M=2, per_core_N=4: max valid subblock (product=8)` or dropped entirely.

**M5 (new) — Section 2.2 in matmul_fundamentals_in_ttnn.md opens with a cross-reference that could be tightened.** Line 90 reads: "Refer to `wormhole_architecture.md` for L1 capacity (1.5 MB per core). The practical L1 budget for a matmul kernel's buffers is roughly 1 MB per core after reserving space for kernel code and metadata." The first sentence is a redirect that the reader already encountered; restating "1.5 MB per core" in the redirect makes it partially self-defeating (the reader doesn't need to go look it up if it is already quoted here). The sentence could be collapsed to: "L1 capacity is 1.5 MB per core (`wormhole_architecture.md` Section 2.1); the practical budget for data buffers is ~1 MB after kernel metadata." This saves one line and removes the redundancy of simultaneously redirecting and answering the redirect.

VERDICT: Crucial updates: no
