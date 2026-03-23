# Cross-Chapter Compression Analysis — Pass 1

## Summary
- Total files analyzed: 10 (guide index + 6 chapter indexes + 3 spot-check content files)
- Estimated cross-chapter redundancy: ~35–45 lines
- Estimated reduction: ~5–7%

---

## CRUCIAL Suggestions

### Cross-chapter: Ch3 (`trace_internals.md`) → Ch5 (`estimating_trace_speedup.md`): dispatch-phase cost table reproduced verbatim

**Issue:** The four-row dispatch-phase cost table (Argument validation 5–15 us / Kernel selection 1–3 us / Command encoding 10–40 us / CQ submission 1–5 us) appears in full in three locations:

1. `ch1_dispatch_fundamentals/host_dispatch_path.md` lines 119–123 — the authoritative definition.
2. `ch3_trace_capture/trace_internals.md` lines 48–51 — verbatim repeat, same four rows, same cost column.
3. `ch5_estimating_improvement/estimating_trace_speedup.md` lines 32–35 — near-verbatim repeat, same four rows, third column ("Eliminated by trace?") added.

The Ch3 instance reproduces all four rows of the Ch1 table with no additional columns and no new facts. It exists only to remind the reader of costs before explaining what replay skips. Ch5's instance adds the "Eliminated by trace?" column, which is the chapter's actual contribution.

**Suggestion:** In `trace_internals.md`, replace the four-row table with a prose sentence citing `host_dispatch_path.md` ("Live dispatch for a single op proceeds through four phases — validation (5–15 us), kernel selection (1–3 us), encoding (10–40 us), and CQ submission (1–5 us), as detailed in `host_dispatch_path.md`") and immediately proceed to the replay flow. The table belongs in Ch1 (definition) and Ch5 (with its "eliminated?" column). Reproducing it in Ch3 without the new column adds ~8 lines with zero new information.

---

### Cross-chapter: Ch4 (`latency_sensitive_workloads.md`) → Ch6 (`index.md`): "Why the Decode Loop Is Canonical" written out in full twice

**Issue:** The argument that the autoregressive decode loop is the canonical trace workload — covering fixed shapes, repeated execution, high dispatch-overhead fraction, and absence of disqualifying conditions — is stated in full in two places:

1. `ch4_when_to_use_trace/latency_sensitive_workloads.md` lines 19–77: full prose section "The Decode Loop as the Canonical Trace Workload" with subsections (Fixed shapes, High op count, Thousands of repetitions, Concrete timeline) and two ASCII timelines. This is the authoritative treatment.
2. `ch6_reference_implementation/index.md` lines 45–53: a section titled "Why the Decode Loop Is the Canonical Case" that restates all four of the same criteria — fixed shapes (with inline arithmetic `head_dim = hidden_dim / num_heads = 2048 / 16 = 128`), repeated execution (512 tokens × 500 us = 256 ms), dispatch overhead fraction (17–63 us across 32–64 ops = 544 us to 4.0 ms), and no disqualifying conditions — as a numbered list.

The Ch6 section is self-described as "restating" Ch4 ("it is worth restating why..."). This is explicit acknowledgement of redundancy. The four criteria are each developed with worked arithmetic that already appeared in Ch4 and/or Ch1. No criterion adds a fact that is new to Ch6's context.

**Suggestion:** Replace the Ch6 "Why the Decode Loop Is the Canonical Case" section (lines 45–53 in `ch6_reference_implementation/index.md`) with a two-sentence forward reference: "The decode loop satisfies all four Chapter 4 criteria that make trace beneficial — fixed shapes, thousands of repetitions, dispatch overhead as a measurable fraction of step time, and no disqualifying conditions. See `latency_sensitive_workloads.md` for the full argument; the reference implementation assumes those criteria are met." This eliminates ~12 lines of re-argued content and ~4 lines of inline arithmetic that duplicates Ch1 numbers.

---

## MINOR Suggestions

### Cross-chapter: Ch2 (`pipelining_host_and_device.md`) → Ch4 (`latency_sensitive_workloads.md`): "Decode loop as primary beneficiary" framing duplicated

**Issue:** `ch2_async_ops/pipelining_host_and_device.md` has a section "The Decode Loop as the Primary Beneficiary" (line 103–) that covers fixed shapes, high op count with short kernels, and async mode's inability to fully hide overhead when device kernels are faster than encoding. `ch4_when_to_use_trace/latency_sensitive_workloads.md` covers the same three points — fixed shapes (lines 23–25), high op count with short kernels (lines 27–31), and the async-can't-hide-it argument (lines 31–33) — in its own "The Decode Loop as the Canonical Trace Workload" section. The two sections are not verbatim copies (Ch2 frames this as an async benefit; Ch4 frames it as a trace benefit), so this is not a CRUCIAL cut, but the "fixed shapes" and "short kernels" sub-arguments are restated in essentially the same terms across both chapters.

**Suggestion:** In `pipelining_host_and_device.md`'s decode loop section, condense the "Fixed shapes" and "High op count / short kernels" reasoning to one or two sentences each and add a forward pointer: "Chapter 4 (`latency_sensitive_workloads.md`) develops this argument fully in the context of trace." This trims ~8 lines of parallel prose while preserving the Ch2-specific point that async mode is the relevant optimization in that chapter.

### Cross-chapter: Ch3 (`trace_internals.md`) → Ch5 (`measuring_dispatch_overhead.md`): per-op overhead summary numbers repeated

**Issue:** `trace_internals.md` (lines 63–76) includes a before/after overhead calculation for a 32-op step (live: 544–2,016 us; replay: 7–15 us; claimed 36–288x reduction). `measuring_dispatch_overhead.md` (lines 204–210) includes a reference table that independently states the per-op warm-path ranges and derives "Total dispatch for 32-op decode step: 544 us–2,016 us". The 32-op example and the same summed range appear in both chapters. The Ch3 version is for illustrating replay speedup; Ch5's version is a measurement reference anchor. Neither is wrong to exist, but the identical "32 ops × 17–63 us = 544–2,016 us" arithmetic is trivially cross-referenced rather than needing to be derived in two places.

**Suggestion:** In `trace_internals.md`'s before/after overhead block, replace the explicit lower-bound arithmetic (`~16–58 us × 32 ops = ~512–1,856 us`) with a cross-reference: "For a 32-op step, Chapter 5's reference table (`measuring_dispatch_overhead.md`) establishes the warm-path total at 544–2,016 us." This saves ~4 lines of arithmetic without losing the speedup illustration.

---

## Load-Bearing Evidence

- `ch1_dispatch_fundamentals/host_dispatch_path.md` lines 117–125: the four-phase dispatch table with the **~17–63 us per op** total row — load-bearing in Ch1 because this is the authoritative definition of dispatch overhead costs that every other chapter cites. Removing or condensing it here would break the entire citation chain.

- `ch2_async_ops/index.md` line 21: "In synchronous mode, the Python thread is blocked for ~17–63 us per op during the four dispatch phases. If encoding the next op takes longer than executing the current one, the device sits idle" — load-bearing in Ch2 because it establishes the baseline synchronous-mode cost before the async execution model is introduced; the contrast between synchronous blocking and async return-immediately is the organizing concept of the chapter. Without this sentence, the baseline is undefined.

- `ch3_trace_capture/trace_internals.md` lines 62–76: the before/after overhead block showing live dispatch (544–2,016 us) vs replay (7–15 us) for a 32-op step — load-bearing in Ch3 because this is the first quantitative statement of how much faster replay is than live dispatch, which is the central claim of the chapter. The arithmetic makes the claim concrete and checkable. (The MINOR suggestion targets the derivation line inside this block, not the block itself.)

- `ch4_when_to_use_trace/latency_sensitive_workloads.md` lines 42–77: the two ASCII timelines (async-dispatch vs trace-replay for a 5-op sequence with 25 us encoding and 10 us kernels, showing 135 us → ~55 us) — load-bearing in Ch4 because this is the only visual demonstration in the guide that shows the device-idle gap closing. No other chapter contains this specific timeline. The 2.4x improvement claim rests on it.

- `ch5_estimating_improvement/estimating_trace_speedup.md` lines 30–35: the four-phase table augmented with the "Eliminated by trace?" column — load-bearing in Ch5 because the third column is the chapter's specific analytical contribution: it distinguishes which phases trace removes (phases 1–3) from which it only partially reduces (phase 4). This differs from the Ch1 and Ch3 instances that have no such column. The table here is not purely redundant.

- `ch6_reference_implementation/traced_decode_loop.md` line 22: "the untraced decode step dispatches approximately 46 ops per step... total dispatch overhead per step is 782–2,898 us" — load-bearing in Ch6 because it ties the abstract overhead range to the specific reference model configuration (hidden_dim=2048, num_layers=2) used throughout the implementation. The number 46 and the resulting range are specific to this chapter's model and do not appear elsewhere.

---

## VERDICT
- Crucial updates: yes
