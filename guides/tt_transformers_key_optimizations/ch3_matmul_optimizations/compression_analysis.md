# Compression Analysis — Chapter 3: MatMul Optimizations

**Date**: 2026-03-24
**Files reviewed**:
- `index.md` (32 lines)
- `matmul_program_configs.md` (112 lines)
- `weight_layout_and_quantization.md` (112 lines)
- `l1_sharding_for_matmul.md` (113 lines)

---

## Summary

4 files reviewed, 369 total lines. Estimated reduction: ~12 lines removed from `matmul_program_configs.md` (the DRAM-sharded mechanism paragraph, shape constraint paragraph, and throughput sentence are duplicated verbatim or near-verbatim in `l1_sharding_for_matmul.md`).

---

## CRUCIAL Suggestions

### C1 — Near-verbatim shape constraint paragraph duplicated across two files

**Location A**: `matmul_program_configs.md` lines 87–89 ("Shape Constraint" subsection under DRAM-Sharded config)
**Location B**: `l1_sharding_for_matmul.md` lines 91–93 ("Shape Constraint" subsection under DRAM-Sharded Matmul section)

The second sentence is word-for-word identical in both:
> "standard transformer hidden dimensions (4096, 8192, etc.) satisfy this constraint in practice."

The first sentence differs only in phrasing ("The weight shape must allow clean sharding across DRAM banks. N must be divisible by…" vs "For DRAM-sharded to apply, the weight N dimension must be divisible by…"), conveying the same information.

**Natural home**: `l1_sharding_for_matmul.md` — the dedicated sharding file is the authoritative source for shape constraints on sharding patterns.

**Action**: Replace the "Shape Constraint" subsection in `matmul_program_configs.md` with a one-line cross-reference to `l1_sharding_for_matmul.md`.

---

### C2 — Near-verbatim DRAM-sharded mechanism description duplicated across two files

**Location A**: `matmul_program_configs.md` lines 77–85 (the "Why This Matters for Decode" subsection plus surrounding text in the DRAM-Sharded config section)
**Location B**: `l1_sharding_for_matmul.md` lines 60–68 ("How It Works" subsection under DRAM-Sharded Matmul section)

Both describe the same mechanism: weight columns distributed across DRAM banks, each core reads exclusively from its own bank, all banks run in parallel for maximum aggregate bandwidth. Key overlapping sentences:

- `matmul_program_configs.md` line 79: "Each core is assigned a contiguous slice of weight columns that resides in its local DRAM bank partition."
- `l1_sharding_for_matmul.md` line 64: "weight columns are sharded contiguously: all column tiles belonging to core k's assigned output range are placed in DRAM bank k."

- `matmul_program_configs.md` line 79: "A core assigned to DRAM bank k fetches only the weight column tiles in that bank — it does not read weight tiles from any other bank."
- `l1_sharding_for_matmul.md` line 64: "When core k begins its matmul, it reads exclusively from DRAM bank k — no other core reads from that bank simultaneously."

**Natural home**: `l1_sharding_for_matmul.md` — the sharding file is the correct location for the per-bank mechanism explanation.

**Action**: In `matmul_program_configs.md`, replace the detailed mechanism description in the DRAM-Sharded section with a brief use-case summary and a cross-reference to `l1_sharding_for_matmul.md` for the mechanism. Retain the "Why to use for decode" framing (small M, bandwidth-bound) since that is the config-selection context not present in the sharding file.

---

### C3 — Verbatim throughput figure duplicated across two files

**Location A**: `matmul_program_configs.md` line 93: "DRAM-sharded typically delivers 2–4× the decode throughput of `MatmulMultiCoreReuseProgramConfig`…"
**Location B**: `l1_sharding_for_matmul.md` lines 83–87: throughput comparison table with row "DRAM-sharded | ~2–4x baseline"

The 2–4× figure is duplicated. The sharding file owns the authoritative throughput comparison table.

**Action**: Remove the standalone throughput sentence from `matmul_program_configs.md` and add a cross-reference to the throughput table in `l1_sharding_for_matmul.md`.

---

## MINOR Suggestions

### M1 — Thematic overlap: decode regime description (M=1–32, bandwidth-bound)

`matmul_program_configs.md` lines 6–8 and `l1_sharding_for_matmul.md` lines 10–12 both describe the decode regime as small-M and bandwidth-bound. Each does so in its own framing context (config selection vs sharding rationale), so these are not verbatim duplicates. No edit required.

### M2 — Thematic overlap: activation DRAM elimination via L1 sharding

`matmul_program_configs.md` lines 57–61 (1D sharded config section) and `l1_sharding_for_matmul.md` lines 3–14 both note that L1 sharding eliminates the activation DRAM read. The program configs file frames this as a property of the 1D config; the sharding file frames it as the motivation for sharding generally. Different angles, no edit required.

---

## Load-Bearing Evidence

1. **`matmul_program_configs.md` lines 77–80** (DRAM-Sharded use-case framing): The decode-specific rationale — "In decode, M is small (1–32 tokens), so the FPU finishes each output tile quickly. The bottleneck is the time spent reading the weight matrix from DRAM" — is the config-selection motivation. This framing does not appear in `l1_sharding_for_matmul.md` and must be retained in `matmul_program_configs.md` after the mechanism detail is removed.

2. **`matmul_program_configs.md` lines 85–86** (concrete size example): "a 4096×16384 BFP8 weight matrix is approximately 68 MB … 32 tokens × 4096 hidden = 512 KB in BF16" — concrete sizing examples that ground the bandwidth-bound claim. These do not appear in `l1_sharding_for_matmul.md`. Must be retained or relocated, not simply dropped.

3. **`l1_sharding_for_matmul.md` lines 64–66** (DRAM-sharded mechanism detail): "weight columns are sharded contiguously … core k reads exclusively from DRAM bank k … All DRAM banks are utilized simultaneously with no bank contention." This is the authoritative mechanism description and must remain intact.

4. **`l1_sharding_for_matmul.md` lines 80–88** (throughput table with caveats): The table is accompanied by the caveat "The following numbers are illustrative for Llama 3.1 8B decode on N150. Actual results vary by layer size, batch size, and data type." This load-bearing hedging must not be lost when the `matmul_program_configs.md` throughput sentence is removed.

5. **`weight_layout_and_quantization.md` lines 26–30** (BFP4 vs BFP8 measured throughput numbers): These specific measurements (28 t/s/u vs 23 t/s/u on N150) are unique to this file and load-bearing for readers making dtype decisions. Not duplicated elsewhere.

---

## VERDICT

- Crucial updates: yes

---

## Agent A Compression Change Log — Pass 1

**Date**: 2026-03-24

### Change 1 — `matmul_program_configs.md`: Replace near-verbatim Shape Constraint subsection with cross-reference

Removed the "Shape Constraint" subsection (lines 87–89) from `matmul_program_configs.md` and replaced with a one-line cross-reference to `l1_sharding_for_matmul.md`, where the authoritative constraint description lives.

**Before** (lines 87–89):
```
### Shape Constraint

The weight shape must allow clean sharding across DRAM banks. N must be divisible by the product of DRAM bank count and core grid column count; standard transformer hidden dimensions (4096, 8192, etc.) satisfy this constraint in practice.
```

**After**:
```
### Shape Constraint

See [L1 Sharding for MatMul — DRAM-Sharded Shape Constraint](l1_sharding_for_matmul.md#dram-sharded-matmul-decode) for the N-divisibility requirement.
```

---

### Change 2 — `matmul_program_configs.md`: Compress near-verbatim DRAM-sharded mechanism into brief summary with cross-reference

Replaced the duplicated per-bank mechanism description (paragraphs in lines 78–85 overlapping with `l1_sharding_for_matmul.md` lines 60–66) with a brief summary. Retained the unique decode-rationale framing and the concrete size example (load-bearing per LBE items 1 and 2 above). Removed the standalone throughput sentence (C3).

**Lines affected**: 77–93 in `matmul_program_configs.md` (the body of the DRAM-Sharded section, excluding the heading and trailing separator).

---

# Compression Analysis — Chapter 3: MatMul Optimizations — Pass 2

## Summary

4 files reviewed. Current approximate line counts after Pass 1 edits: `index.md` ~32 lines, `matmul_program_configs.md` ~101 lines (reduced from 112 by Pass 1), `weight_layout_and_quantization.md` ~112 lines, `l1_sharding_for_matmul.md` ~113 lines. Total: ~358 lines. Estimated reduction from Pass 2 edits: 0 lines (no CRUCIAL duplications found).

---

## CRUCIAL Suggestions

None found.

---

## MINOR Suggestions

### M3 — Thematic overlap: activation DRAM elimination (1D config vs general sharding motivation)

`matmul_program_configs.md` lines 57–61 (1D sharded config section) and `l1_sharding_for_matmul.md` lines 3–7 both state that L1 sharding eliminates the activation DRAM read. The config file frames this as a property specific to the 1D config; the sharding file frames it as the general motivation for any L1 sharding. Different angles, no edit required. (Previously noted as M2 in Pass 1; confirmed still MINOR in Pass 2.)

### M4 — Thematic overlap: op fusion mention

`matmul_program_configs.md` lines 59–61 (1D config use case: FF1 output height-sharded, fed to FF2 without DRAM round-trip) and `l1_sharding_for_matmul.md` lines 13–14 (general statement: L1 sharding enables op fusion by keeping op A output in L1 as op B input). One is a concrete use-case example; the other is the general mechanism statement. No edit required.

---

## Load-Bearing Evidence

No new load-bearing items beyond those identified in Pass 1. All items from Pass 1 LBE list remain valid and intact in the current file state.

---

## VERDICT

- Crucial updates: no
