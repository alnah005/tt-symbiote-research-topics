# B Review — Chapters 5, 6, and 7: Latency Analysis, Implementation, and Validation

## Pass 1

### Issues found: 2

---

**Issue 1 — `synchronize_device_latency_model.md`, Section "Total Estimate at Decode Batch=1 on T3K", Latency model internal inconsistency**

**Error:** The table in this section sums the three components to a maximum of 60 µs:

> | PCIe round-trip (dominant) | 10–30 µs |
> | Kernel completion overlap  | 0–20 µs (often zero at batch=1) |
> | Host scheduling jitter     | 0–10 µs  |
> | **Total per call**         | **10–60 µs** (low end: 10 µs; expected: 20–40 µs; high end: 60 µs) |

The note immediately below attempts to reconcile this with the 0.1–0.5 ms (100–500 µs) figure used in the chapter overview (`index.md`) and in `measuring_the_cost.md`:

> "The range 0.1–0.5 ms stated in the chapter overview (and in the plan specification) represents the observed wall-clock overhead when Python call overhead and OS scheduling noise are included in the measurement."

This reconciliation fails on its own numbers. The model already includes OS scheduling noise as Component 3 at 0–10 µs. And `measuring_the_cost.md` (Method 1 note) explicitly states that Python `time.perf_counter()` call overhead is "typically 0.1–0.5 µs, which is negligible." There is no source of additional overhead identified in the model that could bridge the 60 µs ceiling to the 100–500 µs claimed range — a gap of 40–440 µs. Component 3 as written (0–10 µs) is inconsistent with the note's assertion that OS scheduling noise explains the difference. The two quantitative claims — the structural component table (ceiling 60 µs) and the practical per-call range (100–500 µs) — contradict each other without a mechanism that accounts for the 40–440 µs residual.

**Correction:** One of the two figures must be made consistent with the other. The most defensible fix is to revise Component 3 (Host Scheduling Jitter) to reflect the realistic OS scheduling preemption range on Linux under normal conditions: a spike of 100–500 µs is the documented tail latency for OS scheduler preemption on a non-real-time kernel, not 0–10 µs. Revising Component 3 to "0–10 µs typical; up to 400–500 µs on OS preemption" would bring the total range to 10 µs (best case) through ~550 µs (worst case), consistent with the 100–500 µs practical envelope. Alternatively, if the 0–10 µs jitter estimate for Component 3 is correct for a well-tuned system, the chapter overview should lower its practical estimate to 10–60 µs per call and note that the 0.1–0.5 ms figure in the plan specification is a conservative bound rather than a structural prediction.

---

**Issue 2 — `multi_replay_stability.md`, Section "Trace Replay Consistency Test", Capture procedure step 5 and step 6 duplicate `begin_trace_capture`**

**Error:** The capture procedure in the N-replay consistency test specifies:

> 5. Execute the pre-capture checklist (steps 1–5 in the Chapter 6 wrapper checklist).
> 6. Call `ttnn.begin_trace_capture(...)`.

In the Chapter 6 condensed checklist (`trace_capture_wrapper_changes.md`), item [5] is explicitly:

> [5] ttnn.begin_trace_capture(...)

Step 5 of the multi_replay procedure therefore instructs the engineer to execute the pre-capture checklist including calling `ttnn.begin_trace_capture(...)`. Step 6 then instructs the engineer to call `ttnn.begin_trace_capture(...)` a second time. Calling `begin_trace_capture` twice on the same device without an intervening `end_trace_capture` is an error — it will either raise a runtime assertion or begin a nested trace capture that corrupts the trace buffer.

**Correction:** The boundary between the pre-capture checklist and the explicit call in the multi_replay procedure must be made consistent. The cleanest fix is to redefine the "pre-capture checklist" reference in step 5 to cover only the index snapshot and semaphore reset steps (condensed checklist items [1]–[4]), and let step 6 remain as the explicit `begin_trace_capture` call. Update step 5 to read: "Execute the pre-capture checklist (steps 1–4 in the Chapter 6 wrapper checklist)." This matches the prose pre-capture section in `trace_capture_wrapper_changes.md`, where steps 1–4 are the snapshot-and-reset operations and step 4 ends with "Call `ttnn.begin_trace_capture`" — meaning the condensed checklist's [5] should either be removed from the condensed list or the multi_replay reference should exclude it. Either way, `begin_trace_capture` must appear exactly once in the multi_replay capture procedure.

---

## Pass 1 Change Log

Changes applied in response to Pass 1 issues:

1. **Issue 1** — `synchronize_device_latency_model.md`, Component 3 estimate and reconciliation note: Revise Component 3 (Host Scheduling Jitter) upper bound from 0–10 µs to a value consistent with actual OS preemption tail latency (up to ~400–500 µs), and update the table's total range accordingly. Alternatively, revise the chapter overview's 0.1–0.5 ms estimate downward to match the structural model's 10–60 µs ceiling, and add a separate note that the 0.1–0.5 ms figure is a conservative operational bound rather than a structural prediction.

2. **Issue 2** — `multi_replay_stability.md`, capture procedure step 5: Change the checklist reference in step 5 from "steps 1–5" to "steps 1–4" (excluding the `begin_trace_capture` item from the pre-capture checklist citation), so that `begin_trace_capture` appears exactly once — in step 6 of the multi_replay procedure.

---

## Pass 2

### Verification of Pass 1 fixes

**Fix 1 (synchronize_device_latency_model.md — Component 3 and reconciliation note):** APPLIED

Current Component 3 estimate: "0–10 µs under typical steady-state conditions with CPU affinity set; can spike to hundreds of microseconds (up to ~500 µs) if the OS preempts the Python thread at an inopportune moment"

Current table — Host scheduling jitter row: "0–500 µs (typical < 10 µs; can spike to hundreds of µs under OS preemption)"

Current table — Total per call row: "**10–550 µs** (typical 20–60 µs; rare spikes to ~500 µs under OS preemption)"

**Fix 2 (multi_replay_stability.md — steps 1–5 → steps 1–4):** APPLIED

Current step 5 text: "Execute the pre-capture checklist (steps 1–4 in the Chapter 6 wrapper checklist)."

### Issues found: 1

---

**Issue 3 — `measuring_the_cost.md`, Section "Expected Results", table note attributes ~80–400 µs to "Python overhead" — a mislabeled residual left over from the pre-Fix-1 model**

**Error:** The Expected Results table reads:

> | Tracy host zone duration | 10–60 µs (structural) | Python overhead in wall-clock measurement adds ~80–400 µs on top |

The column note claims that the gap between the Tracy zone duration (10–60 µs) and the Python wall-clock median (100–500 µs) is explained by "Python overhead." This is contradicted by the same file two paragraphs earlier, in the Method 1 note:

> "This measurement includes Python function call overhead for `time.perf_counter()` itself (typically 0.1–0.5 µs), which is negligible relative to the expected 100–500 µs synchronize cost. Do not apply any correction for it."

Python function call overhead is explicitly quantified as 0.1–0.5 µs — it cannot account for 80–400 µs. The actual source of the gap is OS scheduler preemption, which Fix 1 correctly identified and incorporated into `synchronize_device_latency_model.md` as Component 3 (up to ~500 µs). The Tracy zone duration would itself reflect OS preemption that occurs between the two PCIe transactions constituting the round-trip, so the Tracy zone is not bounded at 60 µs when preemption occurs. The table note's 10–60 µs "structural" label for the Tracy zone implicitly assumes a preemption-free capture, which is not guaranteed.

The practical inconsistency: the note tells the engineer to expect Tracy zone readings of 10–60 µs, but then explains the 100–500 µs wall-clock readings as "Python overhead." A reader who sees Tracy zone readings above 60 µs (due to OS preemption captured within the Tracy zone) will be confused about which number is wrong. A reader who sees Tracy zones consistently at 10–30 µs will compute that "Python overhead" = 70–470 µs, contradicting the 0.1–0.5 µs figure two paragraphs above.

**Correction:** Revise the table note for the Tracy host zone duration row to name OS preemption — not Python overhead — as the mechanism that accounts for the gap. The revised note should make clear that (1) the 10–60 µs structural floor applies to preemption-free captures, (2) OS preemption events can inflate the Tracy zone itself beyond 60 µs (since the OS can preempt between PCIe transactions, which Tracy captures as host-side wall time), and (3) the wall-clock median of 100–500 µs reflects this practical distribution rather than Python call overhead. A suitable replacement for the note column text: "Preemption-free floor; OS preemption (up to ~500 µs per Fix 1) can inflate the Tracy zone itself and accounts for the gap to the 100–500 µs wall-clock range — not Python call overhead, which is 0.1–0.5 µs."

---

## Pass 2 Change Log

Changes applied in response to Pass 2 issues:
1. `measuring_the_cost.md` "Interpreting the Tracy Gap" bullet: replaced "plus Python overhead, total 100–500 µs in practice" with accurate attribution — OS thread preemption (not Python call overhead) inflates the PCIe round-trip to hundreds of µs; Python call overhead is 0.1–0.5 µs (negligible)
2. `measuring_the_cost.md` Expected Results table, Tracy host zone duration row note: replaced "Python overhead in wall-clock measurement adds ~80–400 µs on top" with "Upper tail (to ~500 µs) reflects OS preemption; preemption-free runs stay near 10–60 µs"

---

## Pass 3

### Verification of Pass 2 fix

**Fix (measuring_the_cost.md — "Python overhead" replaced with OS preemption attribution):** APPLIED

Current "Interpreting the Tracy Gap" bullet text (line 68):

> "If the `all_gather` kernel completes before the host reaches the `synchronize_device` Python call: the Tracy zone duration equals approximately 1× PCIe round-trip latency (expected: 10–30 µs) under preemption-free conditions; OS thread preemption can inflate this to hundreds of µs (up to ~500 µs), yielding the practical 100–500 µs wall-clock range observed in production."

Current Expected Results table Tracy host zone duration row note (line 94):

> "Upper tail (to ~500 µs) reflects OS preemption; preemption-free runs stay near 10–60 µs"

Both changes are present and correctly worded. The "Interpreting the Tracy Gap" bullet no longer attributes the 100–500 µs range to Python overhead; it correctly names OS thread preemption as the mechanism. The Expected Results table note no longer claims Python overhead adds 80–400 µs; it correctly identifies the upper tail as an OS preemption artifact and preserves the 10–60 µs structural floor as the preemption-free baseline.

### Issues found: 0

---

No issues found. Chapters 5, 6, and 7 approved.

---
