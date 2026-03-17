# B Review — Chapter 6: Sequence Length Scaling Analysis — Pass 1

## Verdict
5 error(s) found.

### Error 1
- **File:** `scaling_theory.md`
- **Line:** 123
- **Stated:** "The gap therefore scales as `O(seq_len × d_model / num_chips)`"
- **Correct:** The CCL latency scaling law is linear with the total message size: `num_active_tokens × d_model` (equivalently `seq_len × top_k × d_model`). The `/num_chips` factor is a per-chip sizing detail that explains how large each chip's send buffer is, but it is not part of the scaling law. Since `d_model` and `num_chips` are both constants in a fixed-topology experiment, the scaling law is simply O(seq_len). Expressing it as `O(seq_len × d_model / num_chips)` incorrectly incorporates a constant factor into the scaling exponent notation and misrepresents which quantity drives the law.

### Error 2
- **File:** `scaling_theory.md`
- **Line:** 181
- **Stated:** Notes column for CCL all-to-all reads "Linear in `num_active_tokens × d_model × 2 / num_chips`"
- **Correct:** The scaling law is linear in total bytes: `num_active_tokens × d_model × 2` (i.e. `seq_len × top_k × d_model × 2`). Dividing by `num_chips` gives the per-chip message size, which is a hardware sizing constant, not a component of the scaling relationship. The Notes entry should read "Linear in `num_active_tokens × d_model × 2` (total); `/num_chips` is per-chip sizing, not part of scaling law."

### Error 3
- **File:** `index.md`
- **Line:** 139
- **Stated:** Pattern B predicted scaling: "Sublinear to constant — depends on whether a CCL or barrier precedes the combine", expected log-log slope: "0–0.5"
- **Correct:** Pattern B is defined as the sync barrier gap between the last expert matmul and the first combine op. A synchronization barrier has O(1) / constant scaling (log-log slope ≈ 0), not sublinear-to-constant with slope up to 0.5. The "depends on whether a CCL or barrier precedes the combine" language conflates Pattern B with Pattern C. Per the authoritative pattern definitions, Pattern B is specifically the sync barrier; its predicted scaling should be O(1) with expected slope ~0.

### Error 4
- **File:** `experiment_design.md`
- **Line:** 109
- **Stated:** "compute the gap between the end of the last zone in the dispatch phase (`MoE/dispatch`) and the start of the first zone in the expert matmul phase (`MoE/expert_matmul`). This is the gap duration for Pattern B/C analysis."
- **Correct:** The gap between `MoE/dispatch` and `MoE/expert_matmul` is the CCL all-to-all latency gap — this is Pattern C, not Pattern B. Pattern B is the gap between the last expert matmul and the first combine op, i.e. between `MoE/expert_matmul` (end) and `MoE/combine` (start). Labeling the dispatch→expert_matmul gap as "Pattern B/C analysis" is wrong; it is Pattern C only.

### Error 5
- **File:** `scaling_theory.md`
- **Line:** 43
- **Stated:** "specifically when `seq_len / 32 < num_tensix_cores`"
- **Correct:** The matmul M-dimension is `expert_capacity`, not `seq_len`. As stated two sentences earlier (lines 38–39), `expert_capacity ≈ seq_len × top_k / num_experts`. The correct memory-bound condition is `expert_capacity / 32 < num_tensix_cores`, i.e. `(seq_len × top_k / num_experts) / 32 < num_tensix_cores`. For the Qwen 235B / DeepSeek-V3 config (top_k=8, num_experts=128), this gives `seq_len × 8 / (128 × 32) < 80`, or `seq_len < 40,960` — a much higher threshold than the `seq_len < 2560` implied by the stated (incorrect) condition. Using `seq_len / 32` instead of `expert_capacity / 32` misidentifies the transition point by a factor of 16 (= num_experts / top_k).

## Agent A Change Log — B Feedback Pass 1
- scaling_theory.md: Removed /num_chips from CCL scaling law; scaling is O(num_active_tokens × d_model)
- scaling_theory.md: Removed /num_chips from summary table CCL row Notes
- index.md: Fixed Pattern B predicted slope from "Sublinear to constant / 0-0.5" to "Constant (slope ≈ 0)"
- experiment_design.md: Corrected pattern label for dispatch→expert_matmul gap from "B/C" to "C"
- scaling_theory.md: Fixed memory-bound condition from seq_len/32 < num_tensix_cores to expert_capacity/32 < num_tensix_cores

# B Review — Chapter 6: Sequence Length Scaling Analysis — Pass 2

## Pass 1 Fix Verification

**Fix 1 — CCL scaling law /num_chips removed (scaling_theory.md ~line 125):**
VERIFIED. The note now reads: "The CCL *gap* observed in Tracy scales with the total message
size: `O(num_active_tokens × d_model)`." The `/num_chips` factor is correctly described as a
per-chip sizing detail, not part of the scaling law.

**Fix 2 — Summary table CCL Notes /num_chips removed (scaling_theory.md ~line 186):**
VERIFIED. The Notes column for CCL all-to-all now reads: "Linear in
`num_active_tokens × d_model × 2` (total); `/num_chips` is per-chip sizing, not part of
scaling law." This matches the authoritative statement exactly.

**Fix 3 — Pattern B slope corrected (index.md line 139):**
VERIFIED. Pattern B entry now reads: Predicted Scaling = "Constant (O(1)) — sync barrier
whose duration is independent of token count"; Expected Log-Log Slope = "Constant (slope ≈ 0)".
This correctly reflects that Pattern B is a sync barrier with O(1) scaling.

**Fix 4 — Pattern label dispatch→expert_matmul corrected (experiment_design.md ~line 109):**
VERIFIED. The gap between `MoE/dispatch` and `MoE/expert_matmul` is now labeled "Pattern C
analysis (CCL all-to-all latency between dispatch and expert compute)". The text also adds a
clarifying note that Pattern B is the separate gap between `MoE/expert_matmul` and
`MoE/combine`. Both labels are now correct.

**Fix 5 — Memory-bound condition uses expert_capacity (scaling_theory.md ~line 43):**
VERIFIED. The condition is now `expert_capacity / 32 < num_tensix_cores`, with
`expert_capacity ≈ seq_len × top_k / num_experts` stated inline. A Note block is added
explaining that for Qwen (top_k=8, num_experts=128), `expert_capacity ≈ seq_len / 16` and
the true seq_len threshold is ~40,960. The compute-bound regime transition in the following
paragraph is also correctly updated to reference `seq_len ≥ 40,960`.

## Remaining Issue Found in Pass 2

**File:** `scaling_theory.md`
**Location:** Summary Table, Matmul rows (~line 182–183)
**Issue:** After Fix 5 corrected the body text, the summary table still states the
memory-bound regime as "seq_len < 2560 on Wormhole B0" and the compute-bound regime as
"seq_len ≥ 2560". These values are now inconsistent with the corrected body text, which
correctly states the transition occurs at seq_len ≈ 40,960 for the Qwen config
(expert_capacity = 80 × 32 = 2560 tiles maps to seq_len = 2560 × 16 = 40,960). The figure
2560 is the expert_capacity tile threshold, not the seq_len threshold; using it as a seq_len
value in the summary table understates the true threshold by a factor of 16.

**Required correction:** Update the Matmul (memory-bound) Notes to read
"Regime: expert_capacity < 2560 tiles, i.e. seq_len < 40,960 for Qwen (top_k=8, experts=128)"
and update the Matmul (compute-bound) Notes to read
"Regime: expert_capacity ≥ 2560 tiles, i.e. seq_len ≥ 40,960 for Qwen; not reached in decode".

## Verdict
1 error remains. The five Pass 1 fixes are all correctly applied. One new error was found:
the summary table Matmul regime thresholds still use the old `seq_len < 2560` figure, which
is inconsistent with the corrected body text after Fix 5. That figure is the expert_capacity
tile threshold, not a seq_len threshold; the correct seq_len threshold for the Qwen config
is ~40,960. The summary table Notes for the two Matmul rows must be updated accordingly.

## Agent A Change Log — B Feedback Pass 2
- scaling_theory.md: Fixed summary table Matmul threshold from "seq_len < 2560 / ≥ 2560" to "expert_capacity < 2560 (seq_len < ~40960 for Qwen) / ≥ 2560"

# B Review — Chapter 6: Sequence Length Scaling Analysis — Pass 3

## Pass 2 Fix Verification

**Fix — Summary table Matmul thresholds updated (scaling_theory.md ~line 182–183):**
VERIFIED. The Matmul (memory-bound) Notes column now reads: "Regime: expert_capacity < 2,560 tokens (seq_len < ~40,960 for Qwen top_k=8, num_experts=128)". The Matmul (compute-bound) Notes column now reads: "Regime: expert_capacity ≥ 2,560 tokens (seq_len ≥ ~40,960 for Qwen); rarely reached in decode". Both rows are now consistent with the corrected body text, which states the transition at expert_capacity ≥ 80 × 32 = 2560, i.e. seq_len ≥ 40,960 for the Qwen config.

## Remaining Issue Found in Pass 3

**File:** `scaling_theory.md`
**Location:** Tip block, ~line 60
**Issue:** The Tip block following the compute-bound regime paragraph reads: "When `seq_len` is below 2560 and the matmul appears to scale sub-linearly..." The value 2560 is the expert_capacity tile threshold, not a seq_len threshold. The body text immediately above (line 57–58) correctly states the seq_len threshold for the Qwen config is 40,960. Saying "seq_len is below 2560" in the Tip is internally inconsistent: it restates the uncorrected, pre-Fix-5 language, which confuses expert_capacity tiles with seq_len tokens. For Qwen (top_k=8, num_experts=128), expert_capacity = seq_len / 16, so seq_len = 2560 corresponds to expert_capacity = 160 — far below the memory-bound boundary.

**Required correction:** Update the Tip to read: "When `expert_capacity` is below 2560 tiles (i.e. `seq_len` is below ~40,960 for Qwen) and the matmul appears to scale sub-linearly..."

## Verdict
1 error remains. The Pass 2 fix is correctly applied and the summary table is now internally consistent. One residual error was found: the Tip block at ~line 60 of `scaling_theory.md` still uses the pre-Fix-5 phrasing "seq_len is below 2560", which conflates the expert_capacity tile threshold with a seq_len value. This is inconsistent with the corrected body text on lines 57–58. The Tip must be updated to reference expert_capacity < 2560 tiles (seq_len < ~40,960 for Qwen).

## Agent A Change Log — B Feedback Pass 3
- scaling_theory.md: Fixed Tip block from "When seq_len is below 2560" to "When expert_capacity is below 2,560 tokens (seq_len below ~40,960 for Qwen)"

# B Review — Chapter 6: Sequence Length Scaling Analysis — Pass 4

## Pass 3 Fix Verification

**Fix — Tip block updated (scaling_theory.md line 60):**
VERIFIED. The Tip block now reads: "When `expert_capacity` is below 2,560 tokens (i.e., `seq_len` below ~40,960 for Qwen with top_k=8, num_experts=128) and the matmul appears to scale sub-linearly, the likely explanation is a tile-count boundary: not all 80 Tensix cores are utilized at small `seq_len`, so the cost is determined by a fixed kernel dispatch overhead plus a small variable compute cost." This correctly references `expert_capacity` as the threshold quantity rather than `seq_len`, and is now internally consistent with the surrounding body text (lines 57–58) and the summary table (lines 182–183).

## Verdict
No feedback — chapter approved. All five Pass 1 fixes, the Pass 2 summary-table fix, and the Pass 3 Tip block fix are correctly applied. No remaining errors found across all four files (`index.md`, `scaling_theory.md`, `experiment_design.md`; `interpreting_scaling_results.md` is referenced but out of scope for this chapter's review). The following facts were verified against the authoritative reference:

- CCL scaling law: O(num_active_tokens × d_model) total; `/num_chips` correctly described as per-chip sizing detail, not part of the scaling law (scaling_theory.md lines 128–131 and summary table line 186).
- Pattern B: sync barrier, O(1), slope ≈ 0 (index.md line 139). Pattern C: CCL, slope ≈ 1 (index.md line 140). Pattern D: cache miss, O(1) (index.md line 141).
- Pattern A: gap after ttnn.topk (index.md line 138). Pattern B: gap between expert matmul and combine (index.md line 139, experiment_design.md lines 110–111). Pattern C: gap between dispatch and expert matmul (experiment_design.md lines 108–109). Pattern D: gap at very start of MoE layer (index.md line 141).
- Worked example: 1024 × 8 × 7168 × 2 = 117,440,512 bytes ≈ 112 MB total; per chip ≈ 14 MB; at 7 GB/s ≈ 2.10 ms (scaling_theory.md lines 109–118). All values correct.
- Standard sweep: {64, 128, 256, 512, 1024, 2048, 4096} (experiment_design.md line 15). Correct.
- 20 timed iterations, median + p95 (experiment_design.md line 117). Correct.
- expert_capacity ≈ seq_len × top_k / num_experts = seq_len/16 for Qwen (scaling_theory.md line 49). Correct.
- Memory-bound condition: expert_capacity/32 < num_tensix_cores (80 for Wormhole B0); expert_capacity < 2560; seq_len < 40,960 (scaling_theory.md lines 43–52). Correct.
