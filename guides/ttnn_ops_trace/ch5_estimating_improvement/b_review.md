# B Review — Chapter 5: Pass 1

1. **File:** `measuring_dispatch_overhead.md`, ~line 210
   **Issue:** The reference table states "Total dispatch for 32-op decode step: 540 us–2.0 ms". The lower bound is wrong: 32 ops × 17 us/op = 544 us, not 540 us. A reader checking the arithmetic against the per-op values (17–63 us, stated two rows above) will find the numbers do not reconcile.
   **Fix:** Change "540 us" to "544 us" (or "~544 us" if rounding is intentional, but then the upper bound should also be stated as "~2,016 us" for consistency).

2. **File:** `estimating_trace_speedup.md`, ~line 34
   **Issue:** The "What Trace Eliminates" table lists command encoding cost as "10–40 us". The reference numbers in `measuring_dispatch_overhead.md` (the authoritative table) give 6–12 us for small ops and 20–50 us for large ops, implying a combined range of 6–50 us. A reader using the 10–40 us range from this table to compute a per-op overhead budget will get a materially different answer than a reader using the reference table.
   **Fix:** Change "10–40 us" to "6–50 us" to match the reference table in `measuring_dispatch_overhead.md`.

3. **File:** `estimating_trace_speedup.md`, ~line 37
   **Issue:** The text states that replacing 32 individual CQ submissions with one execute command yields "a 6–30x reduction in submission cost." The arithmetic does not support this. Using the stated ranges (individual submissions: 1–5 us each; execute command: 1–5 us total), the reduction range is: lower bound = (32 × 1 us) / (5 us) = 6.4x; upper bound = (32 × 5 us) / (1 us) = 160x. The upper bound of 30x is roughly 5× too low and would cause a reader to significantly underestimate the submission-cost benefit of trace on steps with many slow CQ submissions.
   **Fix:** Change "6–30x reduction" to "6–160x reduction".

4. **File:** `profiling_workflow.md`, ~line 9
   **Issue:** The introductory sentence states "The five stages are sequential" but the workflow diagram immediately below lists six labeled stages (Stage 1 through Stage 6). A reader following this workflow will reach Stage 6 and be uncertain whether Stage 6 is optional, a later addition not reflected in the count, or an error — and may skip regression testing as a result.
   **Fix:** Change "The five stages are sequential" to "The six stages are sequential".

## Change Log — Pass 1 Fixes

- `measuring_dispatch_overhead.md`: Corrected 32-op lower-bound total from 540 us to 544 us.
- `estimating_trace_speedup.md`: Corrected command encoding range to 6–50 us. Corrected submission cost reduction to 1–5x per op (live 1–5 us → trace ~1 us).
- `profiling_workflow.md`: Changed "five stages" to "six stages" to match diagram.

---

# B Review — Chapter 5: Pass 2

1. **File:** `estimating_trace_speedup.md`, ~line 37
   **Issue:** The CQ submission reduction multiplier is still wrong after Pass 1. The text states "a 1–5x reduction in submission cost per step." The arithmetic is: 32 individual submissions at 1–5 us each = 32–160 us total, replaced by a single execute command at 1–5 us = a **6.4–160x** reduction (lower bound: 32 us / 5 us = 6.4x; upper bound: 160 us / 1 us = 160x). The stated "1–5x" would imply the execute command costs nearly as much as all 32 individual submissions combined, which contradicts the text immediately preceding it. Pass 1 flagged this correctly as requiring "6–160x" but the fix applied "1–5x" instead, which is a new incorrect value.
   **Fix:** Change "a 1–5x reduction in submission cost per step" to "a ~6–160x reduction in submission cost per step".

## Change Log — Pass 2 Fixes

- `estimating_trace_speedup.md`: Corrected CQ submission reduction to ~6–160x per step (32 individual 1–5 us submissions → one 1–5 us execute command).

---

# B Review — Chapter 5: Pass 3

No feedback — chapter approved.

---

# B Review — Chapter 5: Pass 4

No feedback — chapter approved.
