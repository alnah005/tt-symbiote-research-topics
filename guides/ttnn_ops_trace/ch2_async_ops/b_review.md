# B Review — Chapter 2: Pass 1

1. **File:** `async_execution_model.md`, ~line 19
   **Issue:** The prose states async op mode is "controlled at device-open time via the `num_hw_cqs` parameter and an environment variable." Neither claim is accurate: `num_hw_cqs=1` selects the number of hardware command queues and does not enable async dispatch. No environment variable is named or shown. The code block then comments "# Open a device with async op mode enabled" on the `open_device` line, reinforcing the false attribution. The actual enabling mechanism — `device.enable_async(True)` — appears only afterwards with no connection to the prose claim. A reader could conclude that passing `num_hw_cqs=1` is sufficient to enable async mode and omit the `enable_async(True)` call.
   **Fix:** Correct the prose to state that async mode is enabled post-open via `device.enable_async(True)`. Remove or correct the reference to `num_hw_cqs` as a control for async mode (it is not). Fix or remove the code comment "# Open a device with async op mode enabled" on the `open_device` line.

2. **File:** `pipelining_host_and_device.md`, ~line 109
   **Issue:** The final sentence reads: "But it can occur if the device is very fast relative to the host's dispatch rate, which is exactly the regime where async dispatch provides the most benefit." Both clauses are inverted. CQ backpressure — the CQ filling up — happens when the dispatch thread submits commands faster than the device consumes them, meaning the device is *slow* relative to dispatch throughput. A fast device drains the CQ and prevents backpressure. Additionally, "exactly the regime where async dispatch provides the most benefit" is also wrong: async dispatch benefits most when device kernels are short relative to per-op dispatch overhead (fast device), but that regime is precisely where CQ backpressure is least likely.
   **Fix:** Correct the causal direction: CQ backpressure occurs when the device is slow relative to the dispatch thread's submission rate (long kernels, many in-flight commands). Fast devices drain the CQ and do not cause this stall.

3. **File:** `async_execution_model.md`, ~lines 151–154
   **Issue:** The event API example shows `ttnn.record_event(device, cq_id=0)` and `ttnn.wait_for_event(device, event)`. The actual TTNN Python API for event recording does not take a bare device handle plus an integer `cq_id` keyword argument in this form; the API operates on a command queue object. A reader implementing event-based synchronization from this snippet would use the wrong call signature and get a runtime error or unexpected behavior.
   **Fix:** Verify and correct the API signatures against the TTNN source. The correct form uses the command queue object (e.g., obtained from the device) rather than passing `device` with a separate `cq_id` integer.

4. **File:** `pipelining_host_and_device.md`, ~lines 16 and 121
   **Issue:** Two different dispatch latency ranges are given for the same concept within the same file. Line 16 states "~20–50 us per op"; line 121 states "20–40 us" per op. A reader computing total dispatch overhead for a 960-op decode step will get different answers (19.2–48 ms vs. 19.2–38.4 ms) depending on which range they use. The discrepancy is large enough (50 us vs. 40 us upper bound) to matter when estimating whether dispatch is the bottleneck.
   **Fix:** Reconcile the two ranges to a single consistent value. If one location is citing a different condition (e.g., warm path only vs. cold path), state that distinction explicitly.

## Change Log — Pass 1 Fixes

- `async_execution_model.md`: Corrected async-enabling mechanism — removed incorrect num_hw_cqs/env-var attribution; made device.enable_async(True) the primary mechanism; fixed open_device comment. Replaced incorrect ttnn.record_event/wait_for_event signatures with CQ-object-based pattern plus version note.
- `pipelining_host_and_device.md`: Corrected CQ backpressure causation (fast dispatch thread relative to device, not fast device). Unified dispatch latency to ~20–50 us per op throughout the file.

---

# B Review — Chapter 2: Pass 2

1. **File:** `pipelining_host_and_device.md`, ~line 121
   **Issue:** The text claims that for a 32-layer model with 960 ops per step and per-op dispatch overhead of ~20–50 us, "the dispatch overhead is several times the kernel execution time" relative to a 5–10 ms compute time. At the low end of both ranges — 19.2 ms dispatch vs. 10 ms compute — the ratio is 1.9×, which is not "several times." The claim is only accurate at the favorable end of the ranges (e.g., 48 ms dispatch vs. 5 ms compute ≈ 9.6×). Using "several times" as a blanket statement is numerically inaccurate for a significant portion of the stated range.
   **Fix:** Replace "several times the kernel execution time" with a range-qualified statement, e.g., "2–10× the kernel execution time depending on op count and model size," or tighten the example to a single set of numbers that supports the claim.

2. **File:** `index.md`, line 23 vs. `pipelining_host_and_device.md`, lines 16 and 121
   **Issue:** `index.md` states dispatch phases take "roughly 17–63 us per op on the warm path" (attributing this to Chapter 1). Both detail files in this chapter consistently use "~20–50 us per op." These ranges are materially different at both ends: 17 us vs. 20 us at the lower bound, and 63 us vs. 50 us at the upper bound. A reader cross-referencing `index.md` against the detail files will encounter inconsistent numbers for the same quantity. The upper-bound discrepancy (63 us vs. 50 us) is large enough to produce a noticeably different estimate for total step dispatch overhead on a 960-op model (~60 ms vs. ~48 ms).
   **Fix:** Reconcile the range stated in `index.md` with the range used in the detail files, or explicitly note that `index.md` is citing the Chapter 1 range while the detail files narrow it to the common warm-path case.

## Change Log — Pass 2 Fixes

- `pipelining_host_and_device.md`: Replaced unqualified "several times" with range-qualified multiplier statement.
- `index.md`: Annotated dispatch latency range to reconcile Chapter 1 range (17–63 us) with typical range (~20–50 us) used in detail files.

---

# B Review — Chapter 2: Pass 3

1. **File:** `pipelining_host_and_device.md`, line 121
   **Issue:** The text states "at the lower end, overhead and compute are roughly comparable." At the lower end of the stated ranges — 19.2 ms dispatch (960 ops × 20 us) vs. 10 ms compute — the ratio is approximately 1.9:1. A nearly 2:1 ratio is not "roughly comparable"; the host dispatch overhead is still nearly twice the device compute time. "Roughly comparable" implies an approximately 1:1 relationship, which does not hold anywhere in the stated ranges (the closest case, 19.2 ms vs. 10 ms, is still ~2:1). The claim understates how dominant dispatch overhead is across the full range.
   **Fix:** Replace "roughly comparable" with a numerically accurate characterization, e.g., "dispatch overhead is still roughly 2× the kernel execution time."

## Change Log — Pass 3 Fixes

- `pipelining_host_and_device.md`: Replaced "roughly comparable" with accurate ~2× ratio statement for lower-end dispatch-to-compute comparison.

---

# B Review — Chapter 2: Pass 4

No feedback — chapter approved.

---

# B Review — Chapter 2: Pass 5

No feedback — chapter approved.

---

# B Review — Chapter 2: Pass 6

No feedback — chapter approved.
