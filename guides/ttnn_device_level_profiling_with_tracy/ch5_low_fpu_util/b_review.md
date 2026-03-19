## Pass 1

Three correctness issues found.

---

**Issue 1 — csv_signatures.md, line 55: Cause 1 tile-starvation threshold is wrong by a factor of 2**

File: `csv_signatures.md`, ~line 55

Error: The CSV signature for Cause 1 reads:

```
CORE COUNT > 4 × (output_tiles / 32)
```

This simplifies to `core_count > output_tiles / 8`. But the guideline stated everywhere else (causes_of_low_fpu_util.md line 35, remediation_levers.md line 31) is `M_t × N_t / core_count ≥ 4`, i.e., tile-starved when `core_count > M_t × N_t / 4`. The `/32` in the CSV signature introduces a spurious extra factor of 8, making the detection threshold 8× looser than the guideline. A reader using this formula to detect Cause 1 would miss tile-starvation in the majority of real cases.

Fix: Remove the `/32`. The correct signature is:

```
CORE COUNT > (M_t × N_t) / 4
```

which is the direct complement of the `M_t × N_t / core_count ≥ 4` guideline used throughout the chapter.

---

**Issue 2 — causes_of_low_fpu_util.md, ~line 187; csv_signatures.md, ~line 39: Cause 6 claims FPU UTIL is low because DEVICE KERNEL DURATION is inflated — this is wrong**

Files: `causes_of_low_fpu_util.md` line ~187, `csv_signatures.md` line ~39

Error: Both files state that `FPU UTIL` is low on the first call because compile time inflates the duration. However, the formula defined at the top of the chapter is:

```
FPU_UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles
```

`TRISC1 KERNEL DURATION` is measured on the device, from when the kernel actually starts executing math on the core to when it finishes. Host-side compilation happens before the kernel is dispatched; once dispatched, TRISC1 runs at its normal rate. Host compile time inflates `DEVICE KERNEL DURATION` (wall-clock from first core start to last core end, which in practice includes host-dispatch latency), but it does not inflate `TRISC1 KERNEL DURATION`. Therefore `FPU UTIL` is not low on the first call — only `DEVICE KERNEL DURATION` is anomalously large.

A reader implementing a Cause 6 check by looking for low `FPU UTIL` on the first call will not find it, while the real signal — a large first-call `DEVICE KERNEL DURATION` relative to steady-state — is already correctly stated in the same section.

Fix: Remove the claim that `FPU UTIL` is low on the first call. The observable effect for Cause 6 is solely the elevated `DEVICE KERNEL DURATION`, not a depressed `FPU UTIL`. Both affected lines should be updated to state: "Cause 6 is identified by a first-call `DEVICE KERNEL DURATION` more than 10× the steady-state value; `FPU UTIL` (which uses `TRISC1 KERNEL DURATION`) is not necessarily affected."

---

**Issue 3 — csv_signatures.md, ~line 174: Cause 7 M=16 example self-contradicts the mechanism**

File: `csv_signatures.md`, ~line 174

Error: The quantitative signal section for Cause 7 states: "shapes like M = 16 (→ M_t = 1, but padded loop count may be 2) can show FPU UTIL around 0.5." However, M=16 gives `M_t = ⌈16/32⌉ = 1`. Padding M=16 to the next tile boundary of 32 still produces exactly 1 tile — there is no second tile to generate a loop-count mismatch. The same paragraph immediately above correctly notes that M=33 → `M_t=2` and padded `M_t_padded=2` with no discrepancy either. The example therefore fails to illustrate a loop-count mismatch and will mislead a reader trying to construct or reproduce a Cause 7 scenario.

Fix: Replace the M=16 example with one that actually demonstrates a mismatch, for example: M=33 → `M_t = ⌈33/32⌉ = 2` logical tiles, but if the kernel's loop is generated from the padded shape (M_padded=64 → `M_t_padded=2`), there is no discrepancy at M=33 either. A genuine mismatch example is a tensor whose logical tile count differs from the kernel-compiled loop bound, e.g., a shape where the program config was generated for M=64 (`M_t=2`) but is reused unchanged for M=96 (`M_t=3`), causing the kernel to iterate only 2 tiles while 3 are present. Alternatively, remove the quantitative example entirely and leave only the qualitative description, since a correct concrete example requires kernel-implementation details not established in this chapter.

---

## Pass 2

Pass 1 Issues 1 and 3 target text that does not appear in the current source files: `csv_signatures.md` already shows the correct `CORE COUNT > (M_t × N_t) / 4` threshold (no `/32` factor), and the Cause 7 section contains no M=16 example. Those two issues are already resolved. Pass 1 Issue 2 (Cause 6 implying `FPU UTIL` is affected) remains open — see below. One new issue identified.

---

**Issue 1 (Pass 1 Issue 2, still open) — causes_of_low_fpu_util.md, line 188: Cause 6 Observable Effect implies FPU UTIL is abnormal on first call**

File: `causes_of_low_fpu_util.md`, ~line 188

Error: The Observable Effect for Cause 6 lists two bullets. The second reads: "Subsequent calls with identical parameters show normal `DEVICE KERNEL DURATION` and `FPU UTIL`." By contrast, this implies the first call shows abnormal `FPU UTIL`. But per the formula `FPU_UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles`, host-side compilation time does not inflate `TRISC1 KERNEL DURATION` — only `DEVICE KERNEL DURATION` is affected. A reader using the Observable Effect to triage a first-call anomaly would look for a depressed `FPU UTIL` and not find it, potentially ruling out Cause 6 incorrectly.

Per the strict scope requirement, the Cause 6 signal is `DEVICE KERNEL DURATION` only.

Fix: Change the second bullet from "show normal `DEVICE KERNEL DURATION` and `FPU UTIL`" to "show normal `DEVICE KERNEL DURATION`; `FPU UTIL` is unaffected by the cache miss in either call."

---

**Issue 2 — causes_of_low_fpu_util.md, line 219: Cause 7 mechanism description inverts the direction of the loop-count error**

File: `causes_of_low_fpu_util.md`, ~line 219

Error: The mechanism paragraph states "the math engine performs fewer useful FMA iterations than the op's logical shape would require." This says the kernel under-executes relative to the logical shape. The actual bug is the opposite: the kernel loop is generated from the padded tile count (which is >= the logical tile count), so the kernel iterates the same or more times than the logical count, spending excess cycles on padding zeros. The following sentence ("the FPU is physically running … but a fraction of those cycles operate on padding zeros") correctly describes over-iteration, directly contradicting the preceding sentence.

A reader diagnosing this cause would search for a loop that exits early (under-execution) when the real defect is a loop that runs too long (over-execution on padding). This leads to looking in the wrong place in generated kernel source.

Fix: Replace "performs fewer useful FMA iterations than the op's logical shape would require" with "performs more total FMA iterations than the op's logical shape requires, with the excess iterations operating on padding zeros." The remainder of the paragraph is consistent with this corrected framing.

---

## Pass 3

Pass 2 issues are resolved in the current source: `causes_of_low_fpu_util.md` no longer implies `FPU UTIL` is abnormal on the first call for Cause 6, and the Cause 7 mechanism correctly describes over-iteration rather than under-iteration. One new correctness issue found.

---

**Issue 1 — remediation_levers.md, ~line 40: Worked example labels a 2-tiles/core grid as "borderline" when it clearly violates the chapter's own guideline**

File: `remediation_levers.md`, ~line 40

Error: The worked example for M=128, N=128 (M_t=4, N_t=4, output_tiles=16) evaluates three candidate grids:

- Grid (2, 2) = 4 cores → 16/4 = 4 tiles/core → "meets guideline."
- Grid (2, 4) = 8 cores → 16/8 = 2 tiles/core → "borderline; test both."

The chapter's guideline, stated in `causes_of_low_fpu_util.md` line 35 and `remediation_levers.md` line 31, is `M_t × N_t / core_count ≥ 4`. A ratio of 2.0 is exactly half the minimum threshold — it does not meet the guideline by any reading. Labeling it "borderline" directly contradicts the stated criterion and will cause a reader to accept a grid that the chapter's own rule classifies as tile-starved (Cause 1 active).

Fix: Replace "borderline; test both" with "below guideline (Cause 1 active); avoid unless total parallelism requires it." The only grid in the example that meets the guideline is (2, 2) = 4 cores.

## Pass 4

Pass 3's issue is resolved in the current source (`remediation_levers.md` line 40 already reads "below guideline (2.0 < 4) — Cause 1 tile starvation active; reduce to 4 cores or fewer"). One new correctness issue found.

---

**Issue 1 — causes_of_low_fpu_util.md, ~line 50: Wrong hardware mechanism stated for FP32 throughput loss**

File: `causes_of_low_fpu_util.md`, ~line 50

Error: The Cause 2 mechanism section explains the ~0.5× FP32 throughput penalty with: "the 512-bit SIMD can fit 32 BF16 elements but only 16 FP32 elements." This is architecturally incorrect for Wormhole B0. The Tensix FPU is a tile-based dataflow engine that operates on 32×32-element tiles, not a classical SIMD unit with a fixed-width vector register. There is no 512-bit SIMD lane in the Tensix compute pipeline; throughput differences between BF16 and FP32 arise from internal tile-processing cost (wider accumulation paths, more internal passes per tile in the math engine), not from fitting fewer elements into a SIMD register.

A reader who internalises the "512-bit SIMD packing" explanation will construct a wrong mental model of the Tensix FPU — specifically, they will believe the bottleneck is element packing in a vector register rather than tile-processing cost in a dataflow pipeline. This would lead them to draw incorrect conclusions about other format-throughput tradeoffs (e.g., expecting FP8 to give a further 2× benefit over BF16 purely from packing, which is not how Tensix delivers FP8 throughput gains).

The 0.5× throughput ratio conclusion is stated consistently throughout the chapter and may be correct; only the stated mechanism is wrong.

Fix: Remove the "512-bit SIMD" sentence. Replace it with a mechanistically accurate description, e.g.: "When operands are in FP32, the Tensix math engine requires more internal processing passes per tile — the tile computation involves wider intermediate values and more accumulation work per cycle — resulting in roughly half the FMA throughput compared to BF16 for the same tile dimensions."

## Pass 5

Pass 4's issue is resolved: the current `causes_of_low_fpu_util.md` Cause 2 section contains no "512-bit SIMD" language and correctly describes the throughput penalty in terms of tile-processing cost. One new correctness issue found.

---

**Issue 1 — remediation_levers.md, ~line 190: L1 cost formula introduces an undefined variable, producing wrong numerical results**

File: `remediation_levers.md`, ~line 190

Error: The formula for estimating the L1 cost of increasing `in0_block_w` is:

```
extra_L1_bytes = in0_block_w × K_t_per_block × 32 × 32 × bytes_per_element
```

The variable `K_t_per_block` is not defined anywhere in the section or chapter. `in0_block_w` is described in this same section as the K-tile block width — that is, it already represents the number of K-dimension tiles per inner loop block. Multiplying by a second, undefined K-dimension factor (`K_t_per_block`) double-counts the K dimension. A reader computing L1 consumption to decide whether to increase `in0_block_w` would substitute a guess for `K_t_per_block` (most likely `K_t`, the total K tile count) and arrive at an estimate that is `K_t_per_block` times larger than the actual buffer cost. This would cause them to reject safe increases to `in0_block_w` as L1-unsafe when they are in fact feasible, or to misidentify the source of an actual L1 overflow.

The correct double-buffer cost for one extra copy of the input-A block held in L1 is:

```
extra_L1_bytes = per_core_M × in0_block_w × 32 × 32 × bytes_per_element
```

where `per_core_M` is the number of output rows per core (already defined in the `MatmulMultiCoreReuseProgramConfig` example above this formula).

Fix: Replace `in0_block_w × K_t_per_block` with `per_core_M × in0_block_w` and remove `K_t_per_block` entirely. The corrected formula is:

```
extra_L1_bytes = per_core_M × in0_block_w × 32 × 32 × bytes_per_element
```

## Pass 6

Pass 5's issue is resolved: `remediation_levers.md` line 190 already shows the corrected formula `per_core_M × in0_block_w × 32 × 32 × bytes_per_element`. One new correctness issue found.

---

**Issue 1 — remediation_levers.md, ~line 81: BF8 FPU throughput attributed to a bandwidth property, producing a wrong causal claim**

File: `remediation_levers.md`, ~line 81

Error: The format comparison table has a column labelled "FPU throughput (relative)". For `ttnn.bfloat8_b` the entry reads `≥1.0× BF16 (fewer bytes to read)`. The parenthetical explanation "(fewer bytes to read)" is a NoC/bandwidth property — it describes why TRISC0 finishes faster, not why the TRISC1 math engine issues more FMA operations per cycle. In this chapter's own framework, `FPU UTIL = PM_IDEAL / TRISC1_DURATION`: a reduction in bytes to read affects TRISC0 time (and can reduce stalls from Cause 4), but it does not change the TRISC1 FMA rate, which is what "FPU throughput" means throughout this chapter.

A reader using this table to reason about Cause 2 (sub-optimal data format) will conclude that switching to BF8 raises FPU throughput for the same reason that BF16 beats FP32. That is wrong: BF8 reduces TRISC0 load (a bandwidth effect), whereas the BF16-vs-FP32 gap is a TRISC1 internal processing cost. A practitioner diagnosing low `FPU UTIL` with BF8 operands who expects a throughput uplift in `TRISC1 KERNEL DURATION` will be confused when none appears, and will incorrectly rule out Cause 2.

Fix: Change the `ttnn.bfloat8_b` FPU throughput cell from `≥1.0× BF16 (fewer bytes to read)` to `≈1.0× BF16 (TRISC1 rate similar to BF16; throughput gain comes from reduced TRISC0 load, not increased FMA rate)`. Alternatively, split the table so that TRISC0-side and TRISC1-side effects are not conflated under a single "FPU throughput" column.

## Pass 7

Pass 6's issue is resolved: `remediation_levers.md` line 81 now reads `1.0× BF16 (same TRISC1 FMA rate as BF16; gain is reduced TRISC0/NCRISC load — unpacker reads half the bytes, cutting bandwidth stalls)`, correctly separating the TRISC1 rate from the TRISC0 bandwidth effect. One new correctness issue found.

---

**Issue 1 — csv_signatures.md, ~line 149: Cause 5 warning threshold (0.7) is inconsistent with the detection threshold (0.8), creating a diagnostic gap**

File: `csv_signatures.md`, ~line 149

Error: The Cause 5 detection pattern requires `NOC BW UTIL > 0.8`. The warning at the bottom of the same section reads: "An elementwise op with `FPU UTIL < 0.3` and `NOC BW UTIL > 0.7` is working correctly — it is legitimately bandwidth-bound." These two thresholds are inconsistent and straddle the range 0.7–0.8. A matmul-class op (AI > 8.0 FLOPs/byte) with `NOC BW UTIL = 0.75` and `FPU UTIL < 0.3` would: (a) not trigger Cause 5 detection (0.75 < 0.8), and (b) be covered by the warning's framing that 0.7 is the boundary for "legitimately bandwidth-bound." The reader would conclude the op is working correctly for an elementwise case, even though the op is a matmul that should not be bandwidth-limited at all. The result: a real NoC contention problem in the 0.7–0.8 `NOC BW UTIL` band goes undiagnosed and falls through incorrectly to Cause 7 or is dismissed entirely.

Fix: Align both thresholds to the same value. Since the detection condition is stated as `> 0.8`, change the warning's boundary from `> 0.7` to `> 0.8`: "An elementwise op with `FPU UTIL < 0.3` and `NOC BW UTIL > 0.8` is working correctly — it is legitimately bandwidth-bound." This closes the diagnostic gap.

## Pass 8

Pass 7's issue is resolved: `csv_signatures.md` line 149 already uses `NOC BW UTIL > 0.8` in the warning, consistent with the detection threshold. One new correctness issue found.

---

**Issue 1 — csv_signatures.md, line 166: Cause 7 exclusion condition for Cause 3 only tests for `"HiFi4"`, missing the `"HiFi2"` case**

File: `csv_signatures.md`, ~line 166

Error: The Cause 7 residual checklist requires `MATH FIDELITY is not "HiFi4"` to rule out Cause 3. However, `csv_signatures.md` line 103 (in the Cause 3 section) explicitly states that `"HiFi2"` produces approximately half the `FPU UTIL` of `LoFi` (~0.5× factor). This is a real, significant throughput penalty from math fidelity — the same category of problem as Cause 3. A reader profiling an op with `MATH FIDELITY == "HiFi2"` would pass the Cause 7 checklist (because `"HiFi2" ≠ "HiFi4"`), conclude Cause 3 is ruled out, and potentially escalate to a kernel-level investigation (Cause 7) when the actual fix is simply changing fidelity. The chapter itself documents the HiFi2 throughput impact in the Cause 3 section but fails to include it in the ruling-out criterion, creating a gap that leads a reader to the wrong diagnosis.

Fix: Change line 166 from `MATH FIDELITY is not "HiFi4"` to `MATH FIDELITY is "LoFi"` (ruling out Cause 3 only when fidelity is at baseline). Alternatively, expand the condition to: `MATH FIDELITY is not "HiFi4"` **and** `MATH FIDELITY is not "HiFi2"`. Either form ensures that any non-LoFi fidelity triggers a Cause 3 investigation before proceeding to Cause 7.

## Pass 9

Pass 8's issue is resolved: `csv_signatures.md` line 166 now reads `MATH FIDELITY is not "HiFi2"` or `"HiFi4"` (ruling out Cause 3; only proceed if `MATH FIDELITY == "LoFi"`)`, closing the HiFi2 diagnostic gap.

**No feedback — chapter approved.**

All seven causes, their CSV signatures, remediation levers, worked examples, and the diagnostic checklist are internally consistent and numerically correct across all four files. No issues found that meet the strict scope criteria.

## Pass 10

One correctness issue found.

---

**Issue 1 — causes_of_low_fpu_util.md, line 73: Fidelity iteration-count comparison is stated backwards**

File: `causes_of_low_fpu_util.md`, line 73

Error: The sentence reads: "`LoFi` runs ~4× more FMA iterations per tile than `HiFi4` and ~2× more than `HiFi2`."

This is the exact opposite of the correct relationship. LoFi is the lowest-fidelity, fastest setting — it runs the fewest FMA iterations per tile (1×, by definition; it is the baseline). HiFi4 runs 4× more iterations than LoFi, and HiFi2 runs 2× more than LoFi. This is confirmed within the same paragraph on line 75 ("When `HiFi4` is the active fidelity, the math engine performs four times as many passes per tile as `LoFi`") and by the authoritative table in `remediation_levers.md` (LoFi = 1× FMA iterations, HiFi2 = 2×, HiFi4 = 4×).

A reader who reads only the first sentence of the mechanism description will believe LoFi is the most expensive fidelity and HiFi4 the cheapest — the opposite of the truth. They would then set `math_fidelity=MathFidelity.HiFi4` expecting better throughput and instead get the worst throughput, or avoid `LoFi` under the mistaken belief it incurs the most iterations.

Fix: Replace the sentence with the correct relationship:

> `HiFi4` runs ~4× more FMA iterations per tile than `LoFi`, and `HiFi2` runs ~2× more than `LoFi`.

## Pass 11

Pass 10's issue is resolved: `causes_of_low_fpu_util.md` line 73 already reads "`HiFi4` runs ~4× more FMA iterations per tile than `LoFi`, and `HiFi2` runs ~2× more than `LoFi`" — the direction is correct.

**No feedback — chapter approved.**

All seven causes, their CSV signatures, remediation levers, worked examples, and the diagnostic checklist are internally consistent and numerically correct across all four files. The fidelity iteration counts match the authoritative table in `remediation_levers.md` (LoFi = 1×, HiFi2 = 2×, HiFi4 = 4×) and are applied consistently in every threshold, ratio, and worked example throughout the chapter. No issues meeting the strict scope criteria were found.
