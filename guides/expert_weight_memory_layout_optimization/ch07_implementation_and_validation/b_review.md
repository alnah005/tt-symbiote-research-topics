# B Review — Chapter 7: Implementation and Validation — Pass 1

## Verdict
3 error(s) found.

### Error 1
- **File:** `index.md`
- **Line:** 60
- **Stated:** "PCC computed: threshold `> 0.9999` for bfloat16, `> 0.999` for bfloat8_b."
- **Correct:** The authoritative PCC threshold for bfloat16 output is `> 0.999`, not `> 0.9999`. The same wrong value is repeated in the Key Constants table at line 109 (`PCC threshold (BF16) | > 0.9999`). The `> 0.9999` figure is stricter than the specified threshold and is inconsistent with the authoritative fact.

### Error 2
- **File:** `correctness_verification.md`
- **Line:** 26
- **Stated:** "`bfloat16` | > 0.9999 | Near-bit-identical; layout change only"
- **Correct:** The authoritative PCC threshold for bfloat16 output is `> 0.999`. The incorrect value `> 0.9999` propagates further into this file: line 116 (`assert pcc > 0.9999`), line 126 (`PASS_THRESHOLD_BF16 = 0.9999`), line 149 (`assert min_pcc > PASS_THRESHOLD_BF16`), and line 174 (`if pcc_val > 0.9999`). All should use `0.999`.

### Error 3
- **File:** `benchmark_methodology.md`
- **Line:** 41
- **Stated:** `NUM_TIMED   = 20   # Iterations to time; report mean and p95.`
- **Correct:** The authoritative benchmark methodology specifies 20 timed iterations with **median** and p95 reported — not mean. The error is structural: `statistics.mean(latencies_ms)` is computed at line 83, the result is stored as `mean_ms` throughout, and the print at line 86 outputs `mean=...`. The checklist in `index.md` line 67 repeats the same error ("Latency recorded: mean and p95"). All occurrences should use `statistics.median` and label the result `median_ms`.

## Agent A Change Log — B Feedback Pass 1
- index.md: Fixed PCC threshold from 0.9999 to 0.999
- correctness_verification.md: Fixed all PCC threshold occurrences from 0.9999 to 0.999 (threshold constant, assert statements, prose)
- benchmark_methodology.md: Changed mean→median throughout (statistics.mean→statistics.median, variable names, print statements, prose)
- index.md checklist: Updated latency metric from mean to median

# B Review — Chapter 7: Implementation and Validation — Pass 2

## Pass 1 Fix Verification

**Fix 1 — PCC threshold 0.9999 → 0.999:**
- `index.md` line 60 (checklist Step 4): Fixed. Now reads `> 0.999` for bfloat16.
- `index.md` line 109 (Key Constants table): Fixed. Now reads `PCC threshold (BF16) | > 0.999`.
- `correctness_verification.md` line 27 (Acceptable thresholds table): Fixed. Now reads `> 0.999`.
- `correctness_verification.md` line 116 (`assert pcc > ...`): Fixed. Now `assert pcc > 0.999` with updated error message.
- `correctness_verification.md` line 126 (`PASS_THRESHOLD_BF16`): Fixed. Now `0.999`.
- `correctness_verification.md` line 174 (`verify_single_projection` status check): Fixed. Now `pcc_val > 0.999`.
- `correctness_verification.md` line 192 (Failure pattern 1 heading): Fixed. Now reads "PCC between 0.99 and 0.999".
- `correctness_verification.md` lines 227/230 (Expected PCC Ranges table): Fixed. Both BF16 rows now show `> 0.999`.

**Fix 2 — mean → median in benchmark_methodology.md and index.md:**
- `benchmark_methodology.md` line 40 comment: Fixed. Now "report median and p95".
- `benchmark_methodology.md` line 63 docstring: Fixed. Now "Returns dict with median_ms, p95_ms, label."
- `benchmark_methodology.md` line 83: Fixed. `statistics.mean` → `statistics.median`, variable `median_ms`.
- `benchmark_methodology.md` line 86 print: Fixed. Now `median={median_ms:.3f} ms`.
- `benchmark_methodology.md` line 87 return dict: Fixed. Key is `median_ms`.
- `benchmark_methodology.md` `compute_bandwidth_stats`: Fixed. Uses `result["median_ms"]`, prints `median=`.
- `benchmark_methodology.md` delta computation: Fixed. Uses `results_interleaved["median_ms"]` and `results_sharded["median_ms"]`.
- `benchmark_methodology.md` tip prose: Fixed. Now "median latency".
- `index.md` line 67 (checklist Step 5): Fixed. Now "Latency recorded: median and p95 in milliseconds."

## Verdict

No feedback — chapter approved.
