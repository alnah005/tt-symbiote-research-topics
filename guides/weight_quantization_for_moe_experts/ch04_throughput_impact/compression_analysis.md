# Compression Analysis — Chapter 4: Throughput and Memory Bandwidth Impact

## Crucial updates: yes

---

### Duplication 1 — MathFidelity pass-count table (within Chapter 4 itself)

**Source:** `ch04_throughput_impact/tile_compute_efficiency.md`, section "MathFidelity and Throughput"

```
| MathFidelity | Passes | Relative throughput | Use case |
|--------------|--------|--------------------|---------  |
| LoFi | 1 | 1× (fastest) | Gate/up projections; decode throughput |
| HiFi2 | 2 | ~0.77× | Down projection; residual stream |
| HiFi4 | 4 | ~0.50× | Dense MLP; highest numerical fidelity |
```

**Duplicate:** `ch04_throughput_impact/prefill_compute_throughput.md`, section "MathFidelity Overhead"

```
| MathFidelity | FPU passes | Latency vs LoFi | Notes |
|--------------|-----------|-----------------|-------|
| LoFi | 1 | 1× (baseline) | BF16 accumulation |
| HiFi2 | 2 | ~1.2–1.3× | FP32 intermediate accumulation |
| HiFi4 | 4 | ~2× | Full FP32 accumulation throughout |
```

Both tables convey the same LoFi/HiFi2/HiFi4 pass count and relative overhead information within Chapter 4. The surrounding prose in both files repeats the 20–30% latency penalty for HiFi2 vs LoFi.

**Recommended action:** Keep the table in `tile_compute_efficiency.md` (the tile-level reference file). In `prefill_compute_throughput.md`, replace the full table with a one-sentence statement of the overhead and a cross-reference: "See `tile_compute_efficiency.md` § MathFidelity and Throughput for pass counts and relative throughput."

---

### Duplication 2 — MathFidelity pass-count table (Chapter 4 vs. Chapter 1 and Chapter 2)

**Source:** `ch01_quantization_formats/hardware_dtype_support.md`, section "MathFidelity Levels"

```
| MathFidelity Constant | Accumulation precision | Recommended for |
|---|---|---|
| `ttnn.MathFidelity.LoFi` | Low — fast accumulation, fewer passes | `bfloat4_b` weights |
| `ttnn.MathFidelity.HiFi2` | Medium — 2 accumulation passes | `bfloat8_b` weights |
| `ttnn.MathFidelity.HiFi4` | High — 4 accumulation passes | `bfloat16` weights, full precision |
```

And `ch02_ttnn_quantization_api/compute_kernel_config.md`, section "math_fidelity"

```
| MathFidelity | Description | Accumulation Passes | Relative Speed |
|---|---|---|---|
| `LoFi` | Single-pass accumulation … | 1 | Fastest |
| `HiFi2` | Two-pass accumulation … | 2 | Medium |
| `HiFi3` | Three-pass accumulation … | 3 | Slower |
| `HiFi4` | Four-pass accumulation … | 4 | Slowest |
```

**Duplicate:** Both MathFidelity tables in `ch04_throughput_impact/prefill_compute_throughput.md` and `ch04_throughput_impact/tile_compute_efficiency.md` (see Duplication 1 above).

Chapters 1 and 2 already define MathFidelity levels and pass counts authoritatively. Chapter 4 re-derives the same pass-count and overhead data without adding new information.

**Recommended action:** In both `ch04/prefill_compute_throughput.md` and `ch04/tile_compute_efficiency.md`, remove the inline MathFidelity definition tables and replace with a cross-reference to `ch02_ttnn_quantization_api/compute_kernel_config.md` § MathFidelity Levels (or `ch01_quantization_formats/hardware_dtype_support.md` § MathFidelity Levels). Chapter 4 may retain one sentence stating the overhead impact in the throughput context (e.g., "HiFi2 adds ~20–30% latency vs LoFi on compute-bound prefill; see Chapter 2 for the full fidelity reference.").

---

### Duplication 3 — Tile byte-size table (Chapter 4 vs. Chapter 1, and within Chapter 4 itself)

**Source:** `ch04_throughput_impact/tile_compute_efficiency.md`, section "Fixed Tile Geometry"

```
| Dtype | Bits/element | Bytes per 32×32 tile | Tiles per 1 MB L1 |
|-------|-------------|----------------------|-------------------|
| bfloat16 | 16 | 2,048 | 512 |
| bfloat8_b | 8 | 1,024 | 1,024 |
| bfloat4_b | 4 | 512 | 2,048 |
```

**Duplicate 1:** `ch04_throughput_impact/index.md`, section "Key Hardware Constants (Reference)"

```
| BF16 tile bytes | 2,048 |
| bfloat8_b tile bytes | 1,024 |
| bfloat4_b tile bytes | 512 |
```

**Duplicate 2:** `ch01_quantization_formats/bfloat8_b_format.md`, section "Packing Behavior in Tiles"

```
32 × 32 × 1 byte = 1,024 bytes per tile
…
32 × 32 × 2 bytes = 2,048 bytes per tile
```

And `ch01_quantization_formats/bfloat4_b_format.md` (implicitly, 4-bit → 512 bytes per tile).

Chapter 1 establishes tile byte sizes as format properties. Chapter 4 restates the same numbers in two separate places (index.md and tile_compute_efficiency.md).

**Recommended action:** Remove the tile byte rows from `ch04/index.md` and replace with a cross-reference to `ch04/tile_compute_efficiency.md`. In `ch04/tile_compute_efficiency.md`, add a cross-reference to Chapter 1 for the format derivation ("tile byte sizes derived from format encoding; see Chapter 1") and keep only the throughput-relevant "Tiles per 1 MB L1" column as new material.

---

### Duplication 4 — Pareto-optimal projection configuration table (Chapter 4 vs. Chapter 1 and Chapter 3)

**Source:** `ch04_throughput_impact/bandwidth_vs_accuracy_tradeoff.md`, section "Pareto-Optimal Configurations"

```
| Projection | Recommended dtype | MathFidelity | Rationale |
|------------|------------------|--------------|-----------|
| Gate | bfloat4_b | LoFi | SwiGLU path filters error; max decode throughput |
| Up | bfloat4_b | LoFi | Same rationale as gate |
| Down | bfloat8_b | HiFi2 | Residual stream sensitivity; PCC stays above 0.977 |
| Dense MLP | bfloat8_b | HiFi2 | Conservative; no expert-specific noise path |
```

**Duplicate 1:** `ch01_quantization_formats/hardware_dtype_support.md`, section "DeepSeek-V3 Expert Quantization Summary"

```
| Projection | Weight dtype | MathFidelity | Rationale |
|---|---|---|---|
| Gate (w1) | `ttnn.bfloat4_b` | `LoFi` | Pre-activation; errors suppressed by SiLU gate |
| Up (w3) | `ttnn.bfloat4_b` | `LoFi` | Pre-activation; errors suppressed by SiLU gate |
| Down (w2) | `ttnn.bfloat8_b` | `HiFi2` | Post-activation; feeds residual stream directly |
```

**Duplicate 2:** `ch03_accuracy_analysis/projection_sensitivity.md`, section "Recommended Fidelity Settings"

```
| Projection | Dtype | MathFidelity | Rationale |
|---|---|---|---|
| down (w2) | bfloat8_b | HiFi2 | Highest sensitivity; 2 accumulation passes |
| gate (w1) | bfloat4_b or bfloat8_b | LoFi | SiLU absorbs error; 1 accumulation pass |
| up (w3) | bfloat4_b or bfloat8_b | LoFi | Gate product dilutes error; 1 accumulation pass |
```

All three tables carry the same gate/up/down dtype and MathFidelity recommendations with the same SiLU/SwiGLU rationale. Chapter 4's version is a near-verbatim restatement of tables already established in Chapters 1 and 3.

**Recommended action:** In `ch04/bandwidth_vs_accuracy_tradeoff.md`, replace the "Pareto-Optimal Configurations" table with a cross-reference: "The per-projection dtype and MathFidelity recommendations are established in Chapter 3 (`projection_sensitivity.md` § Recommended Fidelity Settings). Chapter 4 confirms those selections sit on the bandwidth-accuracy Pareto frontier." Keep the Pareto diagram and the Efficiency Frontier section as they add new visual and analytical content not present in earlier chapters.

---

### Duplication 5 — SwiGLU/SiLU noise-filtering rationale (Chapter 4 vs. Chapter 3)

**Source:** `ch03_accuracy_analysis/projection_sensitivity.md`, sections "Why Gate/Up Projections Tolerate Lower Fidelity" and "SiLU as Error Filter"

Multi-paragraph explanation that SiLU saturation compresses large-magnitude gate errors, and that the element-wise product with `up_out` further dilutes errors from either gate or up.

**Duplicate:** `ch04_throughput_impact/tile_compute_efficiency.md`, section "Why bfloat4_b + LoFi for Gate/Up Projections"

```
The element-wise product and sigmoid in SwiGLU absorb small quantization errors in
gate and up outputs without accumulating them into the residual stream directly.
```

And `ch04_throughput_impact/bandwidth_vs_accuracy_tradeoff.md`, section "Gate Projection":

```
Gate is tolerant of bfloat4_b + LoFi: the SwiGLU nonlinearity attenuates quantization
noise before it reaches the residual stream.
```

Chapter 3 already provides the full mechanistic explanation (SiLU saturation, element-wise product dilution, error propagation path). Chapter 4 repeats the core argument without adding new content.

**Recommended action:** In `ch04/tile_compute_efficiency.md` and `ch04/bandwidth_vs_accuracy_tradeoff.md`, condense the SwiGLU/SiLU rationale to one sentence and add a cross-reference: "The SiLU noise-filtering mechanism is analyzed in detail in `ch03/projection_sensitivity.md` § Why Gate/Up Projections Tolerate Lower Fidelity."

---

### Duplication 6 — Qwen MoE expert weight bytes per projection (Chapter 4 vs. Chapter 1)

**Source:** `ch01_quantization_formats/bfloat16_format.md`, section "Expert weight example"

Full derivation of gate/up/down weight bytes for a three-projection expert, including a per-dtype breakdown for BF16/8b/4b and the 256-expert total.

**Duplicate:** `ch04_throughput_impact/prefill_compute_throughput.md`, section "Expert FFN Dimensions: Qwen MoE"

```
| Dtype | Bytes per projection | Bytes (gate+up+down) |
|-------|---------------------|----------------------|
| bfloat16 | 16,777,216 | 50,331,648 |
| bfloat8_b | 8,388,608 | 25,165,824 |
| bfloat4_b | 4,194,304 | 12,582,912 |
```

Chapter 1 already establishes per-expert weight byte totals as part of the format memory footprint analysis. Chapter 4 re-derives and re-presents the same numbers (for the specific Qwen MoE dimensions used in that chapter) as if introducing new content.

**Recommended action:** In `ch04/prefill_compute_throughput.md`, replace the weight bytes table with a reference to the relevant Chapter 1 memory footprint section ("Weight bytes per projection follow directly from the format footprints in Chapter 1; at d_model=4096, d_ff=2048: BF16=16 MB, bfloat8_b=8 MB, bfloat4_b=4 MB per projection."). Retain the arithmetic intensity formula and the practical bottleneck note, which are new material.

## Agent A Change Log — C Feedback Pass 1
- prefill_compute_throughput.md: Replaced duplicate MathFidelity table with cross-reference to Chapter 1 and tile_compute_efficiency.md
- index.md: Replaced duplicate tile byte-size table with one-line summary and Chapter 1 cross-reference
- bandwidth_vs_accuracy_tradeoff.md: Replaced per-projection Pareto table with cross-reference to Chapter 3, projection_sensitivity.md
- tile_compute_efficiency.md and/or bandwidth_vs_accuracy_tradeoff.md: Collapsed SwiGLU/SiLU noise-filtering rationale to one sentence + Chapter 3 cross-reference
- prefill_compute_throughput.md: Replaced per-dtype weight bytes table with Chapter 1 cross-reference

---

## Pass 2 Verification

**Fix 1 — prefill_compute_throughput.md: MathFidelity table replaced**
CONFIRMED. The full duplicate MathFidelity table (LoFi/HiFi2/HiFi4 with FPU passes and latency ratios) is gone. Lines 33–37 now read: "For MathFidelity level definitions and throughput overhead, see Chapter 1 and `tile_compute_efficiency.md` in this chapter." The one-sentence impact statement ("HiFi2 with `fp32_dest_acc_en=True` adds approximately 20–30% latency relative to LoFi") is retained as permitted.

**Fix 2 — index.md: Tile byte-size rows removed from hardware constants table**
CONFIRMED. The three tile byte-size rows (`BF16 tile bytes`, `bfloat8_b tile bytes`, `bfloat4_b tile bytes`) are no longer in the Key Hardware Constants table. Line 75 now contains a single prose sentence: "Tile memory footprints: BF16 = 2,048 bytes, bfloat8_b = 1,024 bytes, bfloat4_b = 512 bytes per 32×32 tile. See Chapter 1 for derivation." The sentence format is an acceptable condensation; however, it still lists all three concrete byte values inline. This is a minor residual — the values now appear in a sentence rather than a table row, which reduces visual weight but does not eliminate the numbers. Acceptable as a partial fix; no further action required unless strict zero-repetition is mandated.

**Fix 3 — bandwidth_vs_accuracy_tradeoff.md: Pareto-optimal projections table replaced**
CONFIRMED. The four-row table (Gate/Up/Down/Dense MLP with Recommended dtype, MathFidelity, Rationale) has been removed from the "Pareto-Optimal Configurations" section. Lines 91–93 now read: "For the recommended dtype and MathFidelity per projection type, see Chapter 3, `projection_sensitivity.md`." The Efficiency Frontier diagram and the "Why Full bfloat4_b Is Not on the Pareto Frontier" section are retained as new analytical content.

**Fix 4a — tile_compute_efficiency.md: SwiGLU noise-filtering rationale condensed**
CONFIRMED. The "Why bfloat4_b + LoFi for Gate/Up Projections" section (lines 72–89) previously contained multi-sentence elaboration of the SwiGLU mechanism. It now contains one sentence: "The SwiGLU nonlinearity absorbs gate/up quantization errors before they reach the residual stream. See Chapter 3, `projection_sensitivity.md` for the full mechanistic analysis." The three-bullet rationale list below it is retained and is acceptable (it covers tile density and FPU throughput, not the SwiGLU mechanism itself).

**Fix 4b — bandwidth_vs_accuracy_tradeoff.md: SwiGLU noise-filtering rationale condensed**
CONFIRMED. The Gate Projection section (line 22) now reads: "Gate is tolerant of bfloat4_b + LoFi due to SwiGLU noise filtering; see Chapter 3, `projection_sensitivity.md` for the full mechanistic analysis." This is one sentence plus a cross-reference, matching the recommended action exactly.

**Fix 5 — prefill_compute_throughput.md: Per-dtype weight bytes table replaced**
CONFIRMED. The three-row dtype/bytes table (bfloat16 / bfloat8_b / bfloat4_b with bytes per projection and total) has been removed. Lines 52–53 now read: "For per-projection memory footprints by dtype, see Chapter 1 format files." The element-count derivation block and the arithmetic intensity note are retained as new content.

## Crucial updates: yes

**Remaining duplication — MathFidelity pass-count table in tile_compute_efficiency.md (Duplication 2, partially unresolved)**

Pass 1 fixed the duplicate in `prefill_compute_throughput.md` by pointing to `tile_compute_efficiency.md`. However, `tile_compute_efficiency.md` lines 36–42 still contain the full LoFi/HiFi2/HiFi4 pass-count and relative-throughput table, which is a near-verbatim restatement of the authoritative tables already present in:

- `ch01_quantization_formats/hardware_dtype_support.md` § MathFidelity Levels (LoFi/HiFi2/HiFi4 with accumulation precision and recommended dtype)
- `ch02_ttnn_quantization_api/compute_kernel_config.md` § math_fidelity (LoFi/HiFi2/HiFi3/HiFi4 with pass counts and relative speed)

`tile_compute_efficiency.md` is now the cross-reference target for `prefill_compute_throughput.md`, which is valid only if `tile_compute_efficiency.md` itself adds genuinely new content. The "Use case" column (Gate/up projections; decode throughput / Down projection; residual stream / Dense MLP; highest numerical fidelity) is throughput-context-specific and does represent new material. The "Passes" and "Relative throughput" columns are duplicates of Chapter 1 and Chapter 2.

Recommended action: In `tile_compute_efficiency.md`, keep only the "Use case" column as new content and replace the "Passes" and "Relative throughput" columns with a one-line cross-reference note beneath the table: "Pass counts and relative throughput figures are defined in Chapter 2, `compute_kernel_config.md` § math_fidelity." Alternatively, drop the full table and replace with: "LoFi (1 pass) is optimal for gate/up projections; HiFi2 (2 passes, ~1.2–1.3× slower) is required for down projection. See Chapter 2 `compute_kernel_config.md` for the full fidelity reference."

**Remaining duplication — DRAM read volume bytes in decode_memory_bandwidth.md vs. tile_compute_efficiency.md**

`decode_memory_bandwidth.md` lines 30–34 contain a three-row DRAM read volume table for the gate projection (bfloat16: 16,777,216 bytes; bfloat8_b: 8,388,608 bytes; bfloat4_b: 4,194,304 bytes). These same byte values appear in `prefill_compute_throughput.md` (which now cross-references Chapter 1 for footprints) and are directly derivable from the tile byte sizes in `tile_compute_efficiency.md`. The concrete per-projection byte counts for d_model=4096, d_ff=2048 are also present in the removed table in `prefill_compute_throughput.md` (Fix 5 above). This is a lower-priority duplication because `decode_memory_bandwidth.md` provides the latency formula derivation (Latency ≈ DRAM_bytes / bandwidth) as new content. The byte values serve the formula and the regime is different (decode vs. prefill). No action strictly required; if strict deduplication is needed, the three byte values could be replaced with a formula reference and the note that d_model=4096, d_ff=2048 gives BF16=16 MB, bfloat8_b=8 MB, bfloat4_b=4 MB (matching Chapter 1 footprint derivation).

## Agent A Change Log — C Feedback Pass 2
- tile_compute_efficiency.md: Removed duplicate LoFi/HiFi2/HiFi4 pass-count table; retained Use case context; added Chapter 2 cross-reference for pass counts and throughput figures

---

## Pass 3 Verification

**Pass 2 fix — tile_compute_efficiency.md: MathFidelity pass-count table removed**
CONFIRMED. The full three-column table (MathFidelity / Passes / Relative throughput / Use case) is gone. Lines 38–43 now present only a bullet list of Use case assignments (LoFi → gate/up; HiFi2 → down projection/residual; HiFi4 → dense MLP/highest fidelity) followed by a prose cross-reference: "Pass counts and relative throughput figures are defined in Chapter 2, `compute_kernel_config.md` § math_fidelity." The Passes and Relative throughput columns no longer exist as standalone columns. The fix matches the recommended action exactly.

**New finding — index.md: MathFidelity pass counts still present as table rows**
`index.md` lines 71–73 retain three rows in the Key Hardware Constants table:
- `MathFidelity LoFi passes | 1`
- `MathFidelity HiFi2 passes | 2`
- `MathFidelity HiFi4 passes | 4`

These are a verbatim restatement of the Accumulation Passes column in `ch02_ttnn_quantization_api/compute_kernel_config.md` § math_fidelity (which is the authoritative source per the Pass 2 cross-reference strategy). The pass-count integers `1`, `2`, `4` add no new chapter-4-specific information: they are hardware constants defined in Chapter 2, not throughput observations. The fix applied in `tile_compute_efficiency.md` deliberately deferred these numbers to Chapter 2 — yet `index.md` continues to state them inline without a cross-reference. This is a residual duplication of the same kind that Pass 2 resolved in `tile_compute_efficiency.md`, present one file over.

## Crucial updates: yes

The three MathFidelity pass-count rows in `index.md` (lines 71–73) duplicate the authoritative Chapter 2 table that Pass 2 explicitly established as the canonical source. Removing or cross-referencing these rows is the direct parallel of the confirmed Pass 2 fix.

Recommended action: Replace the three `MathFidelity * passes` rows in the Key Hardware Constants table in `index.md` with a single note row or prose sentence: "MathFidelity pass counts (LoFi=1, HiFi2=2, HiFi4=4): see Chapter 2, `compute_kernel_config.md` § math_fidelity."

## Agent A Change Log — C Feedback Pass 3
- index.md: Removed 3 MathFidelity pass-count rows from Key Hardware Constants table; added cross-reference to Chapter 2, compute_kernel_config.md § math_fidelity

---

## Pass 4 Verification

**Pass 3 fix — index.md: MathFidelity pass-count rows removed**
CONFIRMED. The three rows (`MathFidelity LoFi passes | 1`, `MathFidelity HiFi2 passes | 2`, `MathFidelity HiFi4 passes | 4`) are no longer present in the Key Hardware Constants table. Line 72 now reads: "MathFidelity pass counts (LoFi=1, HiFi2=2, HiFi4=4) are defined in Chapter 2, `compute_kernel_config.md` § math_fidelity." The fix matches the Pass 3 recommended action exactly: the integer values appear only in a parenthetical within a cross-reference sentence, not as standalone table rows claiming to define the constants.

**Remaining crucial duplications: none found.** All six duplications identified across Passes 1–3 have been addressed:

- Duplication 1 (MathFidelity table in prefill_compute_throughput.md): resolved in Pass 1.
- Duplication 2 (MathFidelity table in tile_compute_efficiency.md vs. Chapters 1 and 2): resolved in Pass 2.
- Duplication 3 (tile byte rows in index.md): resolved in Pass 1 (condensed to prose sentence with Chapter 1 cross-reference).
- Duplication 3b (MathFidelity pass-count rows in index.md): resolved in Pass 3.
- Duplication 4 (Pareto-optimal projection table in bandwidth_vs_accuracy_tradeoff.md): resolved in Pass 1.
- Duplication 5 (SwiGLU/SiLU rationale in tile_compute_efficiency.md and bandwidth_vs_accuracy_tradeoff.md): resolved in Pass 1.
- Duplication 6 (per-dtype weight bytes table in prefill_compute_throughput.md): resolved in Pass 1.

The DRAM read volume table in `decode_memory_bandwidth.md` (lines 30–34) was flagged as a lower-priority overlap in Pass 2 and explicitly deemed not requiring action. Re-examined in this pass: the table directly drives the latency formula derivation (Latency ≈ DRAM_bytes / bandwidth) that follows it on lines 38–43, and its byte values for the gate projection are used numerically in that formula. Removing or cross-referencing these values would break the formula's self-contained derivation. This table does not meet the threshold for a crucial duplication requiring a fix.

## Crucial updates: no

## Load-Bearing Evidence

- `prefill_compute_throughput.md` line 14: "For S=2048, d_model=4096, d_ff=2048 (Qwen MoE expert): AI ≈ 2×2048 = 4096 FLOP/byte, which far exceeds Wormhole's ridge point of ~437 FLOP/byte." — This is the only location in the chapter that derives the concrete arithmetic intensity value for the prefill regime and establishes compute-bound status by comparing it to the ridge point. Cutting it would leave the prefill compute-bound claim unsupported by any calculation.

- `decode_memory_bandwidth.md` lines 38–43: The latency formula block (Latency ≈ DRAM_bytes / bandwidth, with three computed µs values for BF16/bfloat8_b/bfloat4_b) — This is the only place in the chapter where quantization's decode latency benefit is expressed as a concrete time value. The µs figures are new analytical output not present in any earlier chapter.

- `decode_memory_bandwidth.md` lines 76–88: The batch_crossover derivation and table (bfloat16: ~437, bfloat8_b: ~219, bfloat4_b: ~109) — This formula and the resulting crossover batch sizes are unique to this file and this chapter. They explain why all practical decode batch sizes remain memory-bound and directly justify the claim that bandwidth reduction is the primary decode speedup mechanism.

- `tile_compute_efficiency.md` lines 8–12: The tile geometry table with the "Tiles per 1 MB L1" column — The Bits/element and Bytes per tile columns are also present in Chapter 1, but the "Tiles per 1 MB L1" column is unique to this file and is used in the Grid Utilization section below it. The table must remain as the local reference for L1 capacity reasoning.

- `tile_compute_efficiency.md` lines 20–31: The hardware unpack pipeline description (4-step block-FP unpack sequence and the note on negligible latency vs. a software loop) — This mechanistic description of the silicon unpack path is not replicated in any other chapter file. It is the chapter's only explanation of why no software dequantization loop is needed.

- `bandwidth_vs_accuracy_tradeoff.md` lines 14–65: The four per-projection PCC vs. bandwidth tables (Gate, Up, Down, Dense MLP with five dtype/fidelity rows each) — These PCC measurements are the quantitative basis for the Pareto analysis. They are not present in any earlier chapter and cannot be replaced with cross-references without eliminating the chapter's core empirical content.

- `bandwidth_vs_accuracy_tradeoff.md` lines 68–89: The Efficiency Frontier ASCII diagram — This visual representation of the Pareto frontier (PCC on Y-axis, bandwidth reduction on X-axis) is unique to this file. It synthesizes the per-projection tables into a single actionable plot.

- `index.md` lines 32–48: The roofline sketch with annotated ridge point, decode/prefill regime labels, and bandwidth/compute ceiling constants — This ASCII roofline diagram contextualizes the entire chapter's analysis within the compute/memory-bound duality. It is not replicated in the per-section files.

## MINOR Suggestions

- `tile_compute_efficiency.md` line 94: "bfloat4_b + LoFi is optimal for gate/up; bfloat8_b + HiFi2 for down projection." — This summary bullet restates the recommendation already in lines 79–86 of the same file and also in `bandwidth_vs_accuracy_tradeoff.md` summary (lines 112–113). The two summary sections across different files are not a crucial duplication, but the final bullet of `tile_compute_efficiency.md` § Summary could be trimmed to "See `bandwidth_vs_accuracy_tradeoff.md` for the per-projection Pareto analysis." to avoid repeating a conclusion that belongs to the bandwidth file.

- `index.md` line 24: Learning objective 3 states "State the MathFidelity pass counts (LoFi / HiFi2 / HiFi4) and their throughput cost." Given that pass counts are now delegated entirely to Chapter 2 and all Chapter 4 files cross-reference rather than define them, this learning objective is no longer achievable from Chapter 4 alone. Consider rewording to "Explain how MathFidelity selection affects prefill and decode latency in the context of quantized expert FFNs." to align with what Chapter 4 actually teaches.
