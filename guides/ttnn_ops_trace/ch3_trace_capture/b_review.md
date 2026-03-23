# B Review — Chapter 3: Pass 1

1. **File:** `trace_api.md`, ~line 188–193
   **Issue:** The `decode_step` example in the collapsible "what model.decode_step must look like" block is presented as a working decode step template, but it is logically incorrect as attention code. Line 189 reassigns `x` to `ttnn.matmul(input_tensor, self.wk)`, discarding the `wq` result from line 188. The subsequent `ttnn.softmax(x, dim=-1)` is then applied to the raw k-projection, not to a `q @ k^T` attention score. A reader implementing attention from this template will produce a wrong model. Either the example should be fixed to show a correct (even simplified) attention data flow, or it should be explicitly labeled as a structural placeholder that intentionally omits correctness for brevity.
   **Fix:** Correct the data flow so the matmul, softmax, and second matmul express a recognizable attention pattern (e.g., `attn = softmax(q @ k.T); out = attn @ v`), or add a prominent comment stating the code is intentionally pseudocode and must not be used as an attention implementation.

2. **File:** `trace_internals.md`, ~line 75
   **Issue:** The stated reduction ratio ("roughly 40–270× less host overhead") is inconsistent with the numbers in the same table. Using the upper-bound figures from that table: phase 1–3 total is 58 us × 32 = 1,856 us, phase 4 total is 5 us × 32 = 160 us, total live = 2,016 us; trace replay total = 7 us (lower bound). That ratio is 2,016 / 7 ≈ 288×, which exceeds the stated upper bound of 270×. A reader checking the arithmetic will get a different number than the one in the text.
   **Fix:** Change "roughly 40–270×" to "roughly 40–290×" (or recompute consistently from the table values).

3. **File:** `trace_constraints.md`, ~line 227–235
   **Issue:** The Step 4 validation pattern calls `ttnn.release_trace(device, trace_id)` and then immediately re-runs `model.decode_core(input_tensor, kv_cache)` as the "live dispatch" reference to compare against. However, the KV-cache was already mutated by the capture run. Re-running `decode_core` with the same `kv_cache` object processes a *different* (already-updated) cache state than what the capture processed. The comparison is between the capture output (KV-cache at step 0) and the live-dispatch output (KV-cache at step 1), so they will differ even for a correct trace. A reader following this pattern will see a spurious assertion failure and incorrectly conclude their trace is broken.
   **Fix:** Either (a) save a copy of the KV-cache state before capture and restore it before the live-dispatch run, or (b) restructure the validation to replay the trace once and compare against the capture output directly (without a separate live-dispatch re-run after release), documenting that the capture run itself serves as the live-dispatch baseline.

## Change Log — Pass 1 Fixes

- `trace_api.md`: Corrected decode_step attention data flow — q and k now separate, softmax applied to q@k^T scores.
- `trace_internals.md`: Corrected speedup ratio to 40–288× (was 40–270×) to match table upper-bound calculation.
- `trace_constraints.md`: Fixed Step 4 validation pattern — added KV-cache backup/restore so capture and live runs use identical state before comparison.

---

# B Review — Chapter 3: Pass 2

1. **File:** `trace_constraints.md`, Step 4 (lines ~223–225)
   **Issue:** The validation code snippet calls `ttnn.end_trace_capture` with a `trace_id` positional argument that the function does not accept. The API defined in `trace_api.md` is `ttnn.end_trace_capture(device, cq_id)` — it takes two arguments and *returns* the `trace_id`. The snippet instead writes:

   ```python
   trace_id = ttnn.begin_trace_capture(device, cq_id=0)   # begin returns None
   out_trace = model.decode_core(input_tensor, kv_cache)
   ttnn.end_trace_capture(device, trace_id, cq_id=0)       # wrong: trace_id passed as arg
   ```

   `ttnn.begin_trace_capture` returns `None`, so `trace_id` is `None` at that point. Then `ttnn.end_trace_capture` is called with three arguments where the API takes two, and the return value (the actual `trace_id`) is discarded. This code will raise a runtime error and contradicts the API specification given two files earlier in the same chapter. A reader following this pattern to validate their trace will have broken, non-running code.

   **Fix:** Change to match the API as documented in `trace_api.md`:
   ```python
   ttnn.begin_trace_capture(device, cq_id=0)
   out_trace = model.decode_core(input_tensor, kv_cache)
   trace_id = ttnn.end_trace_capture(device, cq_id=0)
   ```

## Change Log — Pass 2 Fixes

- `trace_constraints.md`: Fixed trace API call pattern in Step 4 — begin_trace_capture takes no assigned return value; trace_id assigned from end_trace_capture return value.

---

# B Review — Chapter 3: Pass 3

1. **File:** `trace_internals.md`, lines 65–67 (overhead calculation block)
   **Issue:** The per-op phase cost minimum is internally inconsistent. The table immediately above (lines 48–51) gives phase 1 as 5–15 us, phase 2 as 1–3 us, and phase 3 as 10–40 us. The minimum sum is 5 + 1 + 10 = **16 us**, but the calculation block uses **17 us** as the per-op minimum, yielding "~540–1,860 us" instead of the correct ~512–1,856 us. This carries through to the stated reduction lower bound: at the corrected minimum, live overhead = 512 (phases 1–3) + 32 (phase 4) = 544 us; trace replay upper = 15 us; ratio = 544 / 15 ≈ 36×. The stated lower bound of 40× is therefore too high. The upper bound of 288× remains correct. A reader auditing the arithmetic against the table will find the numbers do not reconcile.
   **Fix:** Change "~17–58 us" to "~16–58 us" and update the corresponding totals to "~512–1,856 us". Recompute the reduction lower bound from corrected figures (approximately 36× at the most conservative end) or state the bounds as approximate.

## Change Log — Pass 3 Fixes

- `trace_internals.md`: Corrected per-op lower-bound to 16 us (sum of phase costs 5+1+10), 32-op total to 512 us, reduction ratio to ~36–288×.

---

# B Review — Chapter 3: Pass 4

No feedback — chapter approved.

---

# B Review — Chapter 3: Pass 5

1. **File:** `trace_api.md`, end of file (line 207)
   **Issue:** The required navigation footer is malformed. The spec requires the file to end with `---` on one line followed by `**Next:** [\`trace_internals.md\`](./trace_internals.md)`. The current file ends with a `</details>` block, two blank lines, and then the `**Next:**` line — with no `---` separator before it. The other two files (`trace_internals.md` and `trace_constraints.md`) both have the `---` separator correctly in place; only `trace_api.md` is missing it.
   **Fix:** Insert `---` on its own line between the closing `</details>` block and the `**Next:**` line.

## Change Log — Pass 5 Fixes

- `trace_api.md`: Added missing `---` separator before navigation footer.

---

# B Review — Chapter 3: Pass 6

No feedback — chapter approved.
