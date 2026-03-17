# Compression Analysis — Chapter 7: Interpretation and Next Steps

## Crucial updates: yes

---

### Duplication 1: Pattern A–D root-cause prose re-stated in full

**Source (original):**
`ch5_identifying_gap/common_gap_patterns.md`, entire file — each pattern has a "Root Cause"
section with a complete explanation and, for Pattern A, the exact `ttnn.to_torch` / Python
loop / `ttnn.from_torch` code block (lines 43–54), and for Pattern B, the
`synchronize_device` search snippet (lines 119–129).

**Duplicate (ch07):**
`gap_to_action_mapping.md`, Pattern A "Root Cause" prose (lines 22–26) and the
"Before (Pattern A)" code block (lines 37–45), and Pattern B "Root Cause" prose
(lines 93–99) plus the `synchronize_device` search snippet (lines 108–113).

**Recommended action:**
In `gap_to_action_mapping.md`, collapse each "Root Cause" subsection to a single sentence
and a back-reference such as "Root cause: see `ch5_identifying_gap/common_gap_patterns.md`
— Pattern A." Remove the re-stated code blocks entirely. The "Optimization Action" material
(what to do) is genuinely new to ch07 and must be kept.

---

### Duplication 2: `ttnn.record_event` / `ttnn.wait_for_event` code block appears twice within ch07

**Source (original within ch07):**
`gap_to_action_mapping.md`, Pattern B Step 2 code block (lines 134–139) — the
`matmul_done_event = ttnn.record_event(...)` / `ttnn.wait_for_event(...)` pattern.

**Duplicate (ch07):**
`optimization_action_reference.md`, Lever 2 code block (lines 77–84) — identical
`record_event` / `wait_for_event` call sequence with the same inline comments.

**Recommended action:**
Keep the code block in `optimization_action_reference.md` (Lever 2) as the single
authoritative version — that file is explicitly the reference for API details.
In `gap_to_action_mapping.md` Pattern B Step 2, replace the duplicated code block with
a pointer: "For the full API pattern see `optimization_action_reference.md`, Lever 2."

---

### Duplication 3: Worked-example decomposition numbers reproduced verbatim from ch06

**Source (original):**
`ch6_sequence_length_scaling/interpreting_scaling_results.md`, "Worked Example: Attributing
the 16ms Gap" section (lines 208–242): constant term 13.8ms, linear slope 1.68µs/token,
CCL slope 2.05µs/token, ratio 0.82, R²=0.997.

**Duplicate (ch07):**
`gap_to_action_mapping.md`, "Step 1: Estimate Total Latency Impact" (lines 332–336) —
reproduces the identical constant=13.8ms and slope ≈ 2.05µs/token values inline as if they
are newly derived, with no cross-reference to ch06.

**Recommended action:**
Replace the inline reproduction with an explicit reference: "Using the ch06 worked-example
values (constant=13.8ms, linear slope 2.05µs/token — see
`ch6_sequence_length_scaling/interpreting_scaling_results.md`):" and then cite only the
final arithmetic result. This retains the prioritization calculation without duplicating the
derivation.

---

### Duplication 4: `ttnn.enable_program_cache` warm-up loop duplicated within ch07

**Source (original within ch07):**
`gap_to_action_mapping.md`, Pattern D Step 2 code block (lines 277–285): for-loop over
`SUPPORTED_SEQ_LENS = [64, 128, 256, 512, 1024, 2048, 4096]` calling `moe_forward` three
times per seq_len.

**Duplicate (ch07):**
`optimization_action_reference.md`, Lever 4 code block (lines 159–165): for-loop over the
same seven seq_len values, same three-iteration warm-up, same `moe_forward` call. The only
difference is that the reference file uses `ttnn.zeros` to construct the dummy input while
the mapping file uses `torch.zeros` + `ttnn.from_torch`.

**Recommended action:**
Keep the code block in `optimization_action_reference.md` (Lever 4) as the canonical
version — it is the reference file. In `gap_to_action_mapping.md` Pattern D Step 2, remove
the for-loop and add a pointer: "Pre-warm the cache for each supported seq_len — see
`optimization_action_reference.md`, Lever 4 for the full warm-up script."

---

## Load-Bearing Evidence

The following content in ch07 is load-bearing and must not be removed or reduced:

- `gap_to_action_mapping.md`: the full "Optimization Action" subsection for each pattern
  (tensor-native index construction for A; remove-vs-event choice for B; ep_degree analysis
  and async CCL overlap for C; `enable_program_cache` call placement for D). This is the
  only location in the guide that specifies what to do rather than what to observe.
- `gap_to_action_mapping.md`: the "Prioritized Backlog" table (lines 345–351) and the
  "Step 1: Estimate Total Latency Impact" formula (lines 326–329). The formula itself is
  new; only the worked-example numbers are duplicated.
- `writing_a_gap_analysis.md`: the seven-section template, confidence vocabulary
  (confirmed/suspected), reproducibility checklist, and audience-guidance section. None of
  this appears in prior chapters.
- `optimization_action_reference.md`: the Quick-Reference Table, all seven Lever sections
  (expected reductions, conditions, correctness caveats), and the full PCC validation
  procedure including `compute_pcc`. These are not present anywhere in ch01–ch06.
- `index.md`: the two-output description (gap analysis document + prioritized backlog) and
  the prerequisites table. These are structural to ch07 and are not duplicated elsewhere.

---

## MINOR Suggestions

- `gap_to_action_mapping.md` lines 71–69: the `ttnn.begin_trace_capture` Pattern A example
  overlaps topically with `optimization_action_reference.md` Lever 3. The ch07 version is a
  shorter illustration, so it adds context, but consider adding a pointer to Lever 3 for the
  full decode-mode conditions checklist.
- `writing_a_gap_analysis.md` Section 2 example build block (lines 84–99) repeats build
  commands introduced in ch02. This is intentional (the gap analysis must be self-contained
  for reproduction), but a parenthetical "(build commands from Chapter 2)" would clarify the
  origin.
- `optimization_action_reference.md` Lever 5 memory calculation (84 MB per expert) matches
  the calculation in `gap_to_action_mapping.md` Pattern C (line 191). Trivial, but one could
  consolidate to a single inline note.

---

## Pass 2 Verification

### Fix Verification

**Fix 1 — Pattern A–D root-cause prose:** Confirmed removed. Each of the four Pattern
sections in `gap_to_action_mapping.md` now opens its "Root Cause" subsection with a single
back-reference sentence ("Pattern X is defined in `ch5_identifying_gap/common_gap_patterns.md`.
When you observe this pattern [...], the recommended actions are below."). The multi-line
explanations and re-stated code blocks that duplicated ch5 are gone.

**Fix 2 — `ttnn.record_event`/`ttnn.wait_for_event` code block:** Confirmed removed from
Pattern B Step 2. The duplicate code block (`matmul_done_event = ttnn.record_event(...)` /
`ttnn.wait_for_event(...)`) is no longer in `gap_to_action_mapping.md`. In its place is the
pointer: "For the implementation pattern, see `optimization_action_reference.md` Lever 2."
The canonical code block remains intact in Lever 2 of `optimization_action_reference.md`.

**Fix 3 — Worked-example decomposition numbers (13.8ms, 2.05µs/token, R²=0.997):** Confirmed
removed. The "Step 1: Estimate Total Latency Impact" section no longer reproduces the inline
arithmetic using those specific ch6 numbers. The section now reads: "See the worked example in
`ch6_sequence_length_scaling/interpreting_scaling_results.md` for the derivation of the
constant and linear terms." The prioritization formula itself (which is new to ch07) is
preserved.

**Fix 4 — `ttnn.enable_program_cache` warm-up loop:** Confirmed removed from Pattern D
Step 2. The `SUPPORTED_SEQ_LENS = [64, 128, 256, 512, 1024, 2048, 4096]` for-loop with
`torch.zeros`/`ttnn.from_torch` is no longer in `gap_to_action_mapping.md`. In its place is
the pointer: "see `optimization_action_reference.md` Lever 4 for the full warm-up script."
The canonical `ttnn.zeros`-based loop remains intact in Lever 4 of
`optimization_action_reference.md`.

---

## Crucial updates: no

## Load-Bearing Evidence

The prioritization formula in `gap_to_action_mapping.md` (`total_gap_ms = constant_term_ms ×
(N+1) + linear_slope_ms_per_token × S + ccl_ms_per_token × 1 × N`) is the only location in
the guide that derives the workload-level cost model; it must not be cut. Pattern A's
"Optimization Action" section contains the only complete before/after illustration of
tensor-izing the index construction (the `ttnn.argsort`-based replacement) and the only
warning about trace capture being inapplicable to variable-shape prefill. Pattern B Step 1's
source-search snippet (`pathlib.Path("models/").rglob(...)`) is the only code in the guide
that shows how to locate synchronization calls, and the correctness verification prose
("Trace through the code to determine whether the combine phase ops... read from the output
tensors") is the only place that explains why removal is safe. The Prioritized Backlog table
is the single authoritative ranking of the four patterns by total latency impact and is not
duplicated anywhere in ch01–ch06.

## MINOR Suggestions

None.
