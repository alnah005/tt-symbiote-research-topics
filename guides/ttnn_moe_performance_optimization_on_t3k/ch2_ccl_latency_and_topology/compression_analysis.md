## Change Log — B Feedback Pass 1

The following changes were applied to Chapter 2 files based on Agent B Pass 1 feedback.

---

### Item 1 — `all_gather_linear_topology.md`: `cluster_axis` verification

**Action:** Verified against `moe.py:L1429–1436`. The `all_gather_async` call in the source does **not** include `cluster_axis`. The existing snippet was already correct and was not modified. An annotation block was added immediately after the code snippet to document this finding: `cluster_axis` is absent from the source; the op uses its default axis behavior (axis 0). Adding `cluster_axis=1` to the snippet would have been incorrect.

---

### Item 2 — `ccl_sensitivity_analysis.md` ~line 28: "Opposite neighbor" corrected

**Action:** Replaced all occurrences of "opposite neighbor" with "adjacent neighbor." Ring topology passes data to and from adjacent neighbors, not diametrically opposite ones. The replacement was applied globally across the file (one occurrence found and corrected at line 28).

---

### Item 3 — `all_gather_linear_topology.md` ~lines 46, 64–65: Last-hop message size corrected

**Action:** The final forwarding hop (hop 7) on a linear chain carries 7 shards, not 8 — the 8th shard (the destination device's own shard) is assembled locally without a network transfer. Two corrections were made:

- Bullet point at ~line 46: clarified that hop 7 carries 7 shards because the final shard is assembled locally.
- Formula block at ~lines 58–65: updated `M_full` definition from "full gather payload ≈ 14 KB" to "7 shards × 1 792 B = 12 544 B ≈ 12.3 KB"; updated the numerical estimate from `T_xfer(14 KB) ≈ 1.2 µs` to `T_xfer(12.5 KB) ≈ 1.04 µs`; updated the total from `≈ 29 µs` to `≈ 28 µs`.

---

### Item 4 — `reduce_scatter_ring_topology.md` ~line 52: Ring sum vs. mean corrected

**Action:** The original text stated the ring sum "produces a mean rather than a sum," which was incorrect. The ring CCL kernel performs a sum. The `1/8` mean normalization comes from the `ttnn.mul(routed_out, 1.0/float(n_rs))` pre-scaling applied at `moe.py:L1407–L1409`, before the ring op. The description was corrected to make this distinction explicit: the ring performs a sum; the pre-scaling produces the final normalized output.

---

### Item 5 — `ccl_sensitivity_analysis.md` ~lines 109, 132: Overlap feasibility anchored to source

**Action:** The data-independence argument for overlapping `shared_experts` with the reduce-scatter was not pinned to a source line. Two additions were made:

- ~line 109 (key observation paragraph): added citation `moe.py:L1426` for the `residual = x` assignment, confirming it is set before `all_gather_async` and therefore holds the pre-gather shard.
- ~line 132 (feasibility assessment, data dependency check): added the same citation `moe.py:L1426` with explanatory context confirming the assignment precedes the all-gather call.

---

## Change Log — B Feedback Pass 2

The following changes were applied to Chapter 2 files based on Agent B Pass 2 feedback.

---

### Item 1 — `index.md` ~line 18 and ~line 41: False claim that both CCL ops use `cluster_axis=1` corrected

**Action:** The `all_gather_async` call in the source does not include a `cluster_axis` parameter. Two sentences in `index.md` incorrectly stated that both CCL operations communicate along `cluster_axis=1`.

- ~line 18 (T3K Physical Topology section): updated the `shape[1]` bullet to state that `cluster_axis=1` applies to `reduce_scatter_minimal_async` only, and added an explicit note that `all_gather_async` does not accept a `cluster_axis` parameter.
- ~line 41 (CCL Operations section): replaced "Both operations communicate along `cluster_axis=1`" with a corrected statement that only `reduce_scatter_minimal_async` uses `cluster_axis=1`; `all_gather_async` does not take that parameter.

---

### Item 2 — `ccl_sensitivity_analysis.md` ~lines 34 and 74: Latency values updated to match `all_gather_linear_topology.md`

**Action:** `ccl_sensitivity_analysis.md` used the old values of 1.2 µs per-transfer and ~29 µs total for the all-gather first-principles estimate, which were superseded by the corrected values in `all_gather_linear_topology.md` (Pass 1, Item 3). Two locations were updated:

- First-principles latency estimate block (~line 34): updated from `1.2 µs transfer` and `≈ 29 µs lower bound` to `1.04 µs transfer` and `≈ 28 µs lower bound`.
- `num_links=1` assessment paragraph (~line 74): updated the ratio from `1.2 µs / 29 µs` to `1.04 µs / 28 µs`; updated the derived savings from `0.6 µs × 7 = 4.2 µs` to `0.52 µs × 7 = 3.6 µs`; updated the improvement estimate from `5–14%` to `5–13%`.

---

## Change Log — B Feedback Pass 3

The following changes were applied to Chapter 2 files based on Agent B Pass 3 feedback.

---

### Item 1 — `ccl_sensitivity_analysis.md`: Wrong line number annotation in inline code comment

**Action:** The code block in the "Current Code Structure" section had `# moe.py:L1426` annotating the `output = ttnn.add(...)` line. In the actual source, `moe.py:L1426` is `residual = x` (the assignment that stores the pre-gather shard for later use by shared experts), not `ttnn.add`. The fix adds `residual = x  # moe.py:L1426` as the first line of the code snippet (before the Step 4 comment, reflecting that the assignment occurs before the all-gather at L1429), and removes the incorrect `# moe.py:L1426` annotation from the `ttnn.add` line. The prose references to `moe.py:L1426` for `residual = x` were already correct and were not modified.

---

### Item 2 — `all_gather_linear_topology.md` ~line 25: Wrong source-line range in verification note

**Action:** The verification note for the `cluster_axis`-absent finding cited `moe.py:L1429–1436`. The actual `all_gather_async` call block spans `moe.py:L1429–1437` (L1429 opens the call, L1430–1435 are arguments, L1436 is the closing parenthesis, and L1437 is the blank line that closes the block). The citation was updated from `moe.py:L1429–1436` to `moe.py:L1429–1437`.

---

## Change Log — B Feedback Pass 4

The following changes were applied to Chapter 2 files to correct all `moe.py` line number references to match verified exact line numbers from `TTNNMoE.forward`.

---

### Item 1 — `all_gather_linear_topology.md` L8: Source range header corrected

**Old:** `moe.py:L1363–L1373`
**New:** `moe.py:L1429–L1436`

**Reason:** The `all_gather_async` call block in `TTNNMoE.forward` is at L1429–L1436, not L1363–L1373. The old range was stale from an earlier version of the source file.

---

### Item 2 — `all_gather_linear_topology.md` L25: Verification note citation corrected

**Old:** `moe.py:L1429–1437`
**New:** `moe.py:L1429–L1436`

**Reason:** Pass 3 had updated this citation to `L1429–1437`, but the verified exact range for the `all_gather_async` block is `L1429–L1436`. Corrected to match the authoritative line numbers. Also normalized the separator to use an en-dash with `L` prefix on the right side for consistency.

---

### Item 3 — `all_gather_linear_topology.md` L179 (Tracy method annotation): Source range corrected

**Old:** `moe.py:L1363–L1373`
**New:** `moe.py:L1429–L1436`

**Reason:** The Tracy zone marker instruction referred to the old stale range for `all_gather_async`. Corrected to the verified range.

---

### Item 4 — `index.md` L37: `all_gather_async` citation corrected

**Old:** `moe.py:L1363–L1373`
**New:** `moe.py:L1429–L1436`

**Reason:** Same stale range as Item 1 above, appearing in the CCL Operations summary table in the index.

---

### Item 5 — `index.md` L39: `reduce_scatter_minimal_async` citation corrected

**Old:** `moe.py:L1410–L1423`
**New:** `moe.py:L1478–L1490`

**Reason:** The `reduce_scatter_minimal_async` call block is at L1478–L1490 in the current source. The old range was stale.

---

### Item 6 — `ccl_sensitivity_analysis.md` L10: Source range header corrected

**Old:** `moe.py:L1363–L1423`
**New:** `moe.py:L1429–L1490`

**Reason:** The file's header source range spanned from the old `all_gather_async` start through the old `reduce_scatter_minimal_async` end. Updated to reflect the verified ranges for both ops.

---

### Item 7 — `ccl_sensitivity_analysis.md`: "Current Code Structure" section header corrected

**Old:** `moe.py:L1404–L1430`
**New:** `moe.py:L1426–L1494`

**Reason:** The section covers the code block from `residual = x` (L1426) through `output = ttnn.add(...)` (L1494). Updated both the section header and the matching prose reference later in the same section.

---

### Item 8 — `ccl_sensitivity_analysis.md`: Inline code comment for `ttnn.mul` pre-normalization corrected

**Old:** `# moe.py:L1407–L1409`
**New:** `# moe.py:L1477`

**Reason:** The `ttnn.mul(routed_out, 1.0/float(n_rs))` pre-normalization is a single line at L1477, not a three-line block at L1407–L1409.

---

### Item 9 — `ccl_sensitivity_analysis.md`: Inline code comment for `reduce_scatter_minimal_async` corrected

**Old:** `# moe.py:L1410–L1423`
**New:** `# moe.py:L1478–L1490`

**Reason:** The `reduce_scatter_minimal_async` call block is at L1478–L1490.

---

### Item 10 — `ccl_sensitivity_analysis.md`: Inline code comment for `shared_experts` corrected

**Old:** `# moe.py:L1425`
**New:** `# moe.py:L1493`

**Reason:** `shared_output = self.shared_experts(residual)` is at L1493.

---

### Item 11 — `reduce_scatter_ring_topology.md` L8: Source range header corrected

**Old:** `moe.py:L1410–L1423`
**New:** `moe.py:L1478–L1490`

**Reason:** The `reduce_scatter_minimal_async` call block is at L1478–L1490.

---

### Item 12 — `reduce_scatter_ring_topology.md` L30: Post-call-site prose reference corrected

**Old:** `moe.py:L1407–L1409`
**New:** `moe.py:L1477`

**Reason:** The pre-normalization multiply is a single line at L1477.

---

### Item 13 — `reduce_scatter_ring_topology.md`: Section header "Pre-Scatter Normalization" corrected

**Old:** `(moe.py:L1407–L1409)`
**New:** `(moe.py:L1477)`

**Reason:** Single-line operation; corrected to match verified line number.

---

### Item 14 — `reduce_scatter_ring_topology.md` L52: Body prose reference for pre-normalization corrected

**Old:** `at \`moe.py:L1407–L1409\``
**New:** `at \`moe.py:L1477\``

**Reason:** Same single-line operation; all occurrences of the old three-line range corrected to L1477.

---

### Item 15 — `reduce_scatter_ring_topology.md` L247: Isolation notes reference corrected

**Old:** `at \`moe.py:L1408\``
**New:** `at \`moe.py:L1477\``

**Reason:** The `ttnn.mul` pre-normalization is at L1477, not L1408.

---

# Compression Analysis: Ch2 CCL Latency and Topology — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~704 lines
- Estimated post-compression line count: ~560 lines
- Estimated reduction: ~20%

## CRUCIAL Suggestions

### [all_gather_linear_topology.md] ~lines 74–88
**Issue:** The Ring topology section is framed as a comparison but contains a complete, self-contained latency derivation (formula, worked numbers, and a conclusion) that largely restates material the reader can infer from the already-presented Linear model. The explanation of why Linear is the current choice (~lines 86–88) duplicates reasoning given in `index.md` line 27 ("The code chooses differently for each op ... and Chapter 2 examines whether those choices are optimal") and again in `ccl_sensitivity_analysis.md` ~line 69 ("Linear is the correct topology for a 1×8 chain without a confirmed wrap-around link"). The 15 lines of Ring derivation collapse to 2–3 sentences of comparative commentary.
**Suggestion:** Cut the Ring formula block and worked arithmetic entirely. Keep one sentence noting that Ring would require the wrap-around link and one sentence with the ~22 µs estimate. The conclusion ("Linear is the current choice") is already in both the index and the sensitivity analysis — remove the duplicate paragraph here.

### [reduce_scatter_ring_topology.md] ~lines 98–118 and ccl_sensitivity_analysis.md ~lines 83–85
**Issue:** The `num_workers_per_link` section in `reduce_scatter_ring_topology.md` presents a worked ΔT formula with a numerical result (≈2.3 µs). `ccl_sensitivity_analysis.md` then restates the same conclusion in different wording: "the 2-worker configuration saves approximately 2.3 µs versus 1 worker at batch=1. This is a genuine improvement that should be kept." The second file adds nothing not already in the first.
**Suggestion:** In `ccl_sensitivity_analysis.md`, replace the restatement with a single cross-reference: "As analysed in `reduce_scatter_ring_topology.md`, `num_workers_per_link=2` saves ~2.3 µs at batch=1 and should be retained." Delete the explanatory sentence that follows it.

### [all_gather_linear_topology.md] ~lines 139–175 (Method 2 microbenchmark code block)
**Issue:** Method 2 is an 36-line code block whose structure nearly duplicates the reduce_scatter measurement harness in `reduce_scatter_ring_topology.md` ~lines 136–175. The timing scaffold (warmup loop, `time.perf_counter`, divide by N, print) is identical in both files. The `Important:` paragraph at line 175 (explaining `ttnn.synchronize_device`) repeats the same instruction already given at line 137 under Method 1 in the same section ("insert `ttnn.synchronize_device` after each call to force the async op to complete").
**Suggestion:** Merge the duplicate synchronize-device explanation into a single note at the top of the Isolation section, then remove the trailing `Important:` block. Consider extracting the common timing scaffold into a shared code snippet referenced by both files, or simply note in Method 2 "use the same timing harness as in `reduce_scatter_ring_topology.md`" and show only the op-specific lines.

### [ccl_sensitivity_analysis.md] ~lines 97–145 (Overlap feasibility section)
**Issue:** The data-independence argument for overlapping `shared_experts` with the reduce-scatter is made three times in this section: (1) in the key-observation paragraph at ~line 111, (2) in the "Data dependency check" bullet at ~line 134, and (3) in the "Buffer conflict check" bullet at ~lines 136–137. All three are saying the same thing: `residual` is set before the all-gather, so `shared_experts` does not read any CCL output. The buffer-conflict check adds one genuinely new point (no aliasing between `residual` and `routed_out`), but the first two paragraphs fully overlap each other.
**Suggestion:** Merge the key-observation paragraph and the data dependency check into a single statement. Remove one of the two citations of `moe.py:L1426`. Preserve the buffer-conflict bullet as its one unique fact.

## MINOR Suggestions

### [index.md] ~lines 3–9
**Issue:** The first two paragraphs (~lines 5–7) are a run-on block making the same core point ("batch=1 decode makes CCL relatively expensive") three different ways: the prefill vs. decode contrast, the message-size-independence observation, and the 40–60% estimate. The material is valid but the three phrasings can be tightened.
**Suggestion:** Cut the middle sentence of line 7 ("The matmul compute time, on the other hand, collapses with the token count") — it is already implied by the preceding sentence about CCL not shrinking. Saves ~1 line of prose.

### [index.md] ~lines 55–61 (Research Questions Addressed section)
**Issue:** Q1 is quoted verbatim here but was already stated in full at line 9 of the same file. The section adds only two forwarding sentences about Chapter 7 and Chapter 3.
**Suggestion:** Replace the Q1 repetition with a back-reference ("This chapter addresses Q1 (stated above).") and keep only the forwarding sentences. Saves 2 lines.

### [reduce_scatter_ring_topology.md] ~lines 228–239 (Synchronization Granularity vs. Latency Tradeoff section)
**Issue:** The prose paragraph after the table (~lines 233–239) explains the "null result" scenario (chunks_per_sync values of 5–20 producing identical latency). This is informative, but it repeats — in prose — what the table immediately above it already conveys via the "pipeline efficiency" and "sync overhead" rows, and what the expected-pattern bullets at ~lines 186–188 already stated.
**Suggestion:** Cut or heavily trim the null-result paragraph. The table captures the tradeoff; the expected-pattern bullets already state the informative null outcome. Saves ~5 lines.

### [ccl_sensitivity_analysis.md] ~lines 68–79 (all_gather_async assessment)
**Issue:** The `num_links=1` assessment block reruns the same arithmetic (T_transfer/T_total ≈ 4%, saving ≈ 3.6 µs, 5–13% improvement) that was already fully worked in `all_gather_linear_topology.md` ~lines 95–108. The only new element here is the word "Verdict."
**Suggestion:** Keep the Verdict sentence and one supporting number. Replace the re-derived arithmetic with a cross-reference to `all_gather_linear_topology.md`. Saves ~5 lines.

### [all_gather_linear_topology.md] ~lines 116–137 (Method 1 code block)
**Issue:** The profiler snippet (Method 1) uses a placeholder `tt_lib.profiler` import with a comment "or equivalent TTNN timing API" — signalling it is not an actual callable API. This makes the snippet lower value than Method 2, yet it receives the same amount of explanation. The `Key discipline` paragraph restates what is already covered at the end of Method 2 and in the measurement notes in `reduce_scatter_ring_topology.md`.
**Suggestion:** Shorten Method 1 to a 2–3 sentence description ("Enable the TTNN device profiler; the op log will contain `all_gather_async` durations. See ch5 for exact API.") and remove the non-runnable code block. This removes 10 lines of placeholder code that cannot be used as-is.

## Load-Bearing Evidence
- `index.md` line ~27: "The code chooses differently for each op (Linear for all-gather, Ring for reduce-scatter), and Chapter 2 examines whether those choices are optimal." — load-bearing because this is the structural premise of the entire chapter; it cannot be cut without losing the reason the chapter exists.
- `all_gather_linear_topology.md` line ~25: "Do not add `cluster_axis=1` to this snippet; the source does not include it." — load-bearing because it is a verified factual annotation that prevents future editors from introducing an incorrect parameter, as documented in the B Pass 1 change log.
- `reduce_scatter_ring_topology.md` line ~52: "The `1/8` pre-scaling applied here (at `moe.py:L1477`) is what produces the final normalized (mean) output... The pattern matters for profiling: `ttnn.mul` executes as a separate op immediately before `reduce_scatter_minimal_async`. Its cost should be measured independently to avoid conflating element-wise multiply overhead with CCL overhead." — load-bearing because the distinction between the pre-scale and the CCL op is a measurement discipline point that was explicitly corrected in B Pass 1 and must be retained.
- `ccl_sensitivity_analysis.md` line ~130: "The barrier semaphore on `reduce_scatter_minimal_async` may already enforce this ordering correctly if `ttnn.add` checks it. Verify by profiling: if Tracy shows `shared_experts` running in the gap while the scatter is in flight, overlap is already occurring." — load-bearing because this is the actionable, non-obvious guidance distinguishing "overlap may already exist" from "overlap requires restructuring," which is the crux of the highest-priority optimization.

## VERDICT
- Crucial updates: yes

---

## Change Log — C Compression Pass 1

The following changes were applied to Chapter 2 files as specified in the four CRUCIAL compression suggestions.

---

### Item 1 — `all_gather_linear_topology.md` ~lines 74–88: Ring topology comparison collapsed

**Action:** Removed the full Ring topology latency derivation (formula, worked arithmetic, and "Why Linear is the current choice" conclusion paragraph). Replaced with 3 comparative sentences noting the ~22 µs vs ~28 µs estimate, the wrap-around link requirement, and a cross-reference to `ccl_sensitivity_analysis.md` for the topology comparison in context. The duplicated derivation and duplicated conclusion paragraph were deleted entirely.

---

### Item 2 — `ccl_sensitivity_analysis.md` ~lines 83–85: ΔT ≈ 2.3 µs restatement replaced with cross-reference

**Action:** Replaced the verbatim restatement "the 2-worker configuration saves approximately 2.3 µs versus 1 worker at batch=1. This is a genuine improvement that should be kept." with a single cross-reference sentence pointing to the derivation in `reduce_scatter_ring_topology.md`. The sentence about `num_workers_per_link=4` diminishing returns was retained as it adds forward-looking guidance not present in the source file.

---

### Item 3 — `all_gather_linear_topology.md` ~line 175: Duplicate closing `Important:` block removed; cross-reference added

**Action:** Removed the trailing `Important:` block ("ttnn.synchronize_device is a blocking host-side call...") that restated synchronize-device discipline already stated at line 137 under Method 1 in the same section. Replaced it with a brief cross-reference note that the warmup-loop / `perf_counter` / divide-by-N harness is the same measurement approach used for reduce-scatter in `reduce_scatter_ring_topology.md`. The all-gather-specific scaffold code was kept in full for completeness.

---

### Item 4 — `ccl_sensitivity_analysis.md` ~lines 97–145: Data-independence argument merged from 3 items to 2

**Action:** Merged the key-observation paragraph ("The key observation: `shared_experts(residual)` operates on `residual`...") and the "Data dependency check" bullet into a single combined paragraph labeled "Data independence." The duplicate citation of `moe.py:L1426` was consolidated into the merged paragraph. The "Buffer conflict check" item was retained as a separate paragraph since it adds the distinct fact that `residual`, `routed_out`, and `shared_output` are non-aliased DRAM buffers. Result: 2 feasibility assessment items instead of 3.

---

# Compression Analysis: Ch2 CCL Latency and Topology — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~695 lines (index.md ~62, all_gather_linear_topology.md ~203, reduce_scatter_ring_topology.md ~258, ccl_sensitivity_analysis.md ~172)
- Estimated post-compression line count: ~668 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions
None remaining.

All four Pass 1 CRUCIAL items are verified addressed:
1. `all_gather_linear_topology.md` ~74–77: Ring comparison is now 3 sentences with cross-reference. Confirmed.
2. `ccl_sensitivity_analysis.md` ~85: ΔT 2.3 µs derivation replaced with single cross-reference to source file. Confirmed.
3. `all_gather_linear_topology.md` ~164: Duplicate `Important:` synchronize-device block removed; cross-reference note added. Confirmed.
4. `ccl_sensitivity_analysis.md` ~111–136: Data-independence argument merged from 3 repetitions to 2; duplicate `moe.py:L1426` citation consolidated. Confirmed.

## MINOR Suggestions

### [index.md] ~lines 55–61
**Issue:** Q1 ("What are the actual latency costs of each CCL op, and are the current topology/link/buffer settings optimal for T3K's 1×8 mesh?") is quoted in full at line 9 and then quoted again verbatim at line 59 in the "Research Questions Addressed" section.
**Suggestion:** Replace the second Q1 quotation with a back-reference: "This chapter addresses Q1 (stated above)." Retain the two forwarding sentences about Chapter 7 and Chapter 3. Saves ~2 lines.

### [all_gather_linear_topology.md] ~lines 109–126 (Method 1 code block)
**Issue:** The Method 1 code block uses `tt_lib.profiler` with an inline comment "or equivalent TTNN timing API," signalling this is not a callable API. The `Key discipline` note immediately after the block restates the `ttnn.synchronize_device` requirement already stated under Method 2 at ~line 162 and in the Isolation notes in `reduce_scatter_ring_topology.md`. The non-runnable snippet provides less value than its length justifies.
**Suggestion:** Replace the code block with 2–3 descriptive sentences ("Enable the TTNN device profiler; the op log will contain an `all_gather_async` entry with device timestamps. See ch5 for exact API.") and remove the placeholder import. Delete or collapse the `Key discipline` note since it duplicates Method 2's closing note. Saves ~10 lines.

### [reduce_scatter_ring_topology.md] ~lines 233–239 (null-result prose paragraph)
**Issue:** The paragraph following the Synchronization Granularity table restates the null-result scenario in prose ("For batch=1 decode with 14 KB messages... `chunks_per_sync` values of 5, 10, and 20 will produce identical latency"). This is already communicated by the expected-pattern bullets at ~lines 186–188 and is inferable from the table rows directly above.
**Suggestion:** Cut the null-result paragraph entirely or reduce it to one sentence. The table and the expected-pattern bullets already carry this information. Saves ~5 lines.

### [ccl_sensitivity_analysis.md] ~lines 71–77 (num_links=1 arithmetic re-derivation)
**Issue:** The `num_links=1` assessment block re-derives `T_transfer/T_total ≈ 4%`, the `0.52 µs × 7 = 3.6 µs` saving, and the `5–13%` improvement estimate. All three numbers were already fully worked in `all_gather_linear_topology.md` ~lines 86–97. The only unique content here is the Verdict sentence.
**Suggestion:** Keep the Verdict sentence and one supporting number (e.g., "~3.6 µs saving, ~5–13% improvement at batch=1 — see `all_gather_linear_topology.md` for derivation"). Remove the re-derived formula block. Saves ~5 lines.

### [ccl_sensitivity_analysis.md] ~lines 165–167 (closing prose after Priority Matrix)
**Issue:** Line 167 ("The highest-leverage action is investigating whether the reduce-scatter and shared-expert compute already overlap and, if not, explicitly enabling that overlap. This does not require topology or link changes and has the largest expected impact at batch=1 decode.") restates the Priority column of the table row immediately above it ("High") and the savings column ("20–50 µs"). The table already carries this verdict.
**Suggestion:** Cut the closing prose paragraph entirely. The Priority Matrix table is self-explanatory. Saves ~2 lines.

## Load-Bearing Evidence
- `all_gather_linear_topology.md` line ~25: "Do not add `cluster_axis=1` to this snippet; the source does not include it." — load-bearing because it is a verified factual annotation preventing incorrect parameter insertion, as documented in the B Pass 1 change log.
- `reduce_scatter_ring_topology.md` line ~52: "The `1/8` pre-scaling applied here (at `moe.py:L1477`) is what produces the final normalized (mean) output... The pattern matters for profiling: `ttnn.mul` executes as a separate op immediately before `reduce_scatter_minimal_async`. Its cost should be measured independently." — load-bearing because the distinction between pre-scale and CCL op is a measurement discipline point explicitly corrected in B Pass 1.
- `reduce_scatter_ring_topology.md` lines ~113–116: the ΔT ≈ 2.3 µs derivation formula — load-bearing because `ccl_sensitivity_analysis.md` now cross-references this as the source derivation; removing it would break the reference.
- `ccl_sensitivity_analysis.md` line ~130: "The barrier semaphore on `reduce_scatter_minimal_async` may already enforce this ordering correctly if `ttnn.add` checks it. Verify by profiling: if Tracy shows `shared_experts` running in the gap while the scatter is in flight, overlap is already occurring." — load-bearing because this is the actionable non-obvious guidance distinguishing whether overlap is already in effect from whether restructuring is required.

## VERDICT
- Crucial updates: no
