# B Review — Chapter 6: Pass 1

1. **File:** `traced_decode_loop.md`, ~line 22
   **Issue:** The introductory paragraph states "approximately 38 ops per step (2 layers × 19 ops per layer)" and uses 38 as the basis for the dispatch overhead range ("646 us to 2.4 ms (2394 us)"). The inline comment at the end of `decode_core` (line 350) correctly counts 39 ops: "2 layers × 19 ops + 1 lm_head = 39 ops." The rest of the document (lines 467, 571, 584, 588) consistently uses 39. The opening description is wrong by one op (the lm_head is omitted), and as a result the overhead range of 646–2394 us is internally consistent only with 38 ops, not 39.
   **Fix:** Change "approximately 38 ops per step (2 layers × 19 ops per layer)" to "approximately 39 ops per step (2 layers × 19 ops per layer + 1 lm_head op)" and update the overhead range to 663–2457 us (17–63 us × 39 ops).

2. **File:** `operational_concerns.md`, ~line 414–444 (`decode_with_recovery`)
   **Issue:** `decode_with_recovery` catches `TraceReplayError` from calls to `decoder.decode()`. However, `TracedDecoder.decode()` calls `ttnn.execute_trace` directly (line 270), not via `safe_execute_trace`. `TraceReplayError` is only raised inside `safe_execute_trace`. Therefore `decoder.decode()` will raise a raw `RuntimeError` on replay failure, which is not caught by the `except TraceReplayError` clause. The recovery loop silently passes raw `RuntimeError` through to the outer `raise` at the bottom of the `except` branch on the final attempt, but on non-final attempts it is not caught at all and propagates immediately. The recovery mechanism as written never actually fires.
   **Fix:** Either (a) have `TracedDecoder.decode()` call `safe_execute_trace` instead of `ttnn.execute_trace` directly, so that typed exceptions are raised, or (b) have `decode_with_recovery` catch `RuntimeError` directly and apply the same re-capture logic.

## Change Log — Pass 1 Fixes

- `traced_decode_loop.md`: Corrected op count to 39 (added lm_head); updated overhead range to 663–2,457 us.
- `operational_concerns.md`: Fixed decode_with_recovery — TracedDecoder.decode() now calls safe_execute_trace so TraceReplayError is properly raised and caught.

---

# B Review — Chapter 6: Pass 2

1. **File:** `traced_decode_loop.md`, line 350 (comment at end of `decode_core`) and line 22
   **Issue:** The claimed op count of 19 ops per layer does not match the actual code. Counting the device ops dispatched inside the per-layer loop: rms_norm(1) + matmul×3(3) + reshape+permute for q, k, v(6) + update_cache×2(2) + scaled_dot_product_attention(1) + permute+reshape on attn_out(2) + matmul+add for output projection and residual(2) = 17 attention ops; rms_norm(1) + matmul+silu+matmul+add(4) = 5 FFN ops; total 22 ops per layer. With 2 layers and 1 lm_head: 2×22 + 1 = 45, not 39. The stated total (39) and the stated per-layer count (19) are both incorrect relative to the code as written.
   **Fix:** Either correct the comment to "2 layers × 22 ops + 1 lm_head = 45 ops" and update every downstream reference that uses 39, or reconcile the code so it dispatches 19 ops per layer (by collapsing some ops).

2. **File:** `traced_decode_loop.md`, line 588
   **Issue:** The estimated dispatch overhead eliminated per step is stated as "646–2394 us (17–63 us × 39 ops)." However, 17×39 = 663, not 646, and 63×39 = 2457, not 2394. The values 646 and 2394 equal 17×38 and 63×38 respectively — they are left over from the pre-fix era when the op count was 38. This is a direct numerical error: the parenthetical arithmetic contradicts the stated multiplier.
   **Fix:** Change to "663–2457 us (17–63 us × 39 ops)" to match the stated op count. (If the op count is corrected to 45 per item 1 above, the range becomes 765–2835 us.)

## Change Log — Pass 2 Fixes

- `traced_decode_loop.md`: Corrected per-layer op count to 22 (17 attention + 5 FFN), total 45 ops (2×22+1 lm_head). Updated overhead range to 765–2,835 us throughout; updated profiler example if needed.

---

# B Review — Chapter 6: Pass 3

1. **File:** `index.md`, line 49
   **Issue:** The KV-cache shape is described as `[max_seq_len, hidden_dim]`. This is wrong on two counts: the batch and num_heads dimensions are omitted, and `hidden_dim` is not the correct innermost dimension. The actual allocation in `traced_decode_loop.md` (preallocate_tensors, line 165) is `[batch, num_heads, max_seq_len, head_dim]` — a 4-D tensor where the last dimension is `head_dim` (128), not `hidden_dim` (2048).
   **Fix:** Replace `[max_seq_len, hidden_dim]` with `[batch, num_heads, max_seq_len, head_dim]`.

2. **File:** `traced_decode_loop.md`, `decode_core` docstring (lines 244–246) and `capture_trace` / `run_traced_decode` interaction
   **Issue:** The `decode_core` docstring states "The returned tensor is the same object as output_tensor (allocated in preallocate_tensors) — not a newly allocated tensor." This is incorrect. `decode_core` ends with `logits = ttnn.matmul(hidden, weights["lm_head"]); return logits` — `ttnn.matmul` allocates and returns a new tensor. `output_tensor` is never passed into `decode_core` and is never written to by it. Additionally, `capture_trace` accepts `output_tensor` as a parameter but never uses it. As a result, the trace command buffer has no write targeting `output_tensor`'s address. The production loop in `run_traced_decode` reads `output_tensor` after `execute_trace` and would read stale zeros, not the computed logits. The docstring claim and the implementation are mutually contradictory, and the loop logic is broken as written.
   **Fix:** Either (a) pass `output_tensor` into `decode_core` and use an in-place write (e.g., `ttnn.matmul(..., output_tensor=output_tensor)` if the API supports it, or copy logits into `output_tensor` before returning), so the trace encodes a write to that address; or (b) remove the docstring claim that the return is the same object as `output_tensor`, and in `run_traced_decode` read from the tensor returned by the capture step rather than from `output_tensor` — but note that option (b) conflicts with the trace address-fixity requirement and is not a valid production pattern.

## Change Log — Pass 3 Fixes

- `index.md`: Corrected KV-cache shape to [batch, num_heads, max_seq_len, head_dim].
- `traced_decode_loop.md`: Fixed output_tensor write pattern — decode_core now ends with ttnn.copy_(output_tensor, result) so output_tensor is populated in-place during every replay.

---

# B Review — Chapter 6: Pass 4

1. **File:** `traced_decode_loop.md`, lines 22, 597–601, 710, 736, and 749–751
   **Issue:** The Pass 3 fix added `ttnn.copy_(output_tensor, result)` as the final op in `decode_core`, correctly reflected in the inline comment at line 360: "2 layers × 22 ops + 1 lm_head + 1 copy_ = 46 ops." However, every other op-count reference in the file was not updated and still states 45:
   - Line 22: "approximately 45 ops per step (2 layers × 22 ops per layer, plus 1 final lm_head projection)" — omits copy_.
   - Lines 597–601: "Phases 1–3 of dispatch do not run for any of the 45 traced ops" and "Estimated overhead eliminated per step: 765–2,835 us (17–63 us × 45 ops)."
   - Profiler section headers (lines 710 and 736): "45 ops/step" and "45-op command buffer."
   The overhead range 765–2,835 us equals 17×45 and 63×45. With 46 ops the correct range is 782–2,898 us (17×46 and 63×46).
   **Fix:** Change all "45" op-count references to "46" and update the overhead range to 782–2,898 us throughout.

No further correctness issues found. Navigation footers are correct on both files.

## Change Log — Pass 4 Fixes

- `traced_decode_loop.md`: Updated all op count references from 45 to 46 (ttnn.copy_ added in Pass 3 counts as one more traced op); updated overhead range to 782–2,898 us.

---

# B Review — Chapter 6: Pass 5

No feedback — chapter approved.

---

# B Review — Chapter 6: Pass 6

No feedback — chapter approved.

---

# B Review — Final Pass: Cross-Chapter Consistency

1. **Command encoding phase cost range — Ch1 vs Ch5 numerical inconsistency.**
   `ch1_dispatch_fundamentals/host_dispatch_path.md` (Phase 3 row in the dispatch cost table) states the command encoding cost as **10–40 us**. `ch5_estimating_improvement/estimating_trace_speedup.md` (the "What Trace Eliminates" table, Command encoding row) states the same phase as **6–50 us**. The lower bound differs by 4 us and the upper bound by 10 us. These two files are the only places in the guide where a per-phase breakdown is given; a reader who reads both will see conflicting authoritative figures for the same phase.
   **Fix:** Align the encoding row in `estimating_trace_speedup.md` to match the range established in `host_dispatch_path.md`: change "6–50 us" to "10–40 us".

2. **"Encoding overhead" vs "dispatch overhead" — terminology mislabeling in Ch4 back-reference to Ch1.**
   `ch4_when_to_use_trace/index.md` (opening paragraph) describes the 17–63 us figure as "**host-side encoding overhead** across four phases." Ch1 defines "command encoding" as phase 3 only (cost 10–40 us) and reserves "dispatch overhead" for the all-four-phase total (17–63 us). Using "encoding overhead" to label the four-phase total contradicts the glossary Ch1 establishes and implies only the encoding phase is being measured.
   **Fix:** Change "host-side encoding overhead across four phases" to "host-side **dispatch** overhead across four phases" to match Ch1's own terminology.

3. **Guide-level `index.md` Chapter Index table — all six chapters have clickable markdown links; all "How to Use" deep links point to files that exist.** No issue found.

4. **Guide-level `index.md` Quick Reference API table** — all seven linked files (`async_execution_model.md`, `pipelining_host_and_device.md`, `trace_api.md`, `trace_internals.md`, `traced_decode_loop.md`, `operational_concerns.md`) exist in their referenced directories. No broken links found.

5. **Ch5 cross-attribution to Ch3 reduction factor — verified consistent.** `ch5_estimating_improvement/estimating_trace_speedup.md` line 39 claims "Chapter 3 (`trace_internals.md`) states that replay achieves a 36–288x reduction in per-op overhead cost." `ch3_trace_capture/trace_internals.md` line 75 states exactly "roughly 36–288× less host overhead per step." The citation and the source are consistent. No issue.

## Change Log — Final Pass Fixes

- `estimating_trace_speedup.md`: Aligned command encoding range to 10–40 us (Ch1 authoritative value).
- `ch4_when_to_use_trace/index.md`: Replaced "host-side encoding overhead" with "dispatch overhead" for four-phase total, consistent with Ch1 glossary.

---

# B Review — Final Pass: Cross-Chapter Check 2

1. **Trace replay overhead range — Ch3 vs Ch4 numerical inconsistency.**
   `ch3_trace_capture/trace_internals.md` ("Why Replay Bypasses Host Dispatch Overhead") gives a detailed breakdown: "Single 'execute trace' CQ write: ~5–10 us; Runtime dispatch bookkeeping: ~2–5 us; Total host time: ~7–15 us." `ch4_when_to_use_trace/latency_sensitive_workloads.md` (Decode Loop section, third bullet) states the same quantity as "on the order of 5–20 us total." The upper bound diverges by 5 us (15 us vs 20 us). Ch3 is the authoritative source — it provides the breakdown that derives the total; Ch4 cites the total without showing the arithmetic.
   **Fix:** Align `latency_sensitive_workloads.md` to "on the order of 7–15 us total" to match the Ch3 authoritative breakdown.

2. **"Four disqualifying conditions" — count stated in Ch4/index.md contradicts the numbered structure in Ch4/when_not_to_trace.md.**
   `ch4_when_to_use_trace/index.md` (Learning Objectives) states "State the **four** primary disqualifying conditions for trace." `ch4_when_to_use_trace/when_not_to_trace.md` explicitly numbers only three: "Disqualifying Condition 1: Dynamic Shapes Per Call," "Disqualifying Condition 2: Ops That Read Back to Host Mid-Loop," "Disqualifying Condition 3: Ops That Self-Configure Based on Prior Results." Data-dependent dispatch (Ch3 Category 2 — Python branches on device tensor values) is discussed in depth within Condition 2's body but is never given its own numbered condition. The parallel source, `ch3_trace_capture/trace_constraints.md`, lists exactly four numbered categories. A reader following the index's promise of four will not find a fourth numbered condition in the file.
   **Fix:** Either (a) add "Disqualifying Condition 4: Data-Dependent Dispatch" as a distinct numbered section in `when_not_to_trace.md`, extracting the data-dependent-branching content already present in Condition 2's body; or (b) change the index's Learning Objective and the Ch4 opening paragraph to say "three" and align them with Ch3 by noting that Conditions 2 and 3 together cover Ch3's Categories 2, 3, and 4.

3. **Ch4/index.md opening paragraph — op-count example uses a step size not established anywhere in the guide.**
   The opening "Core Question" paragraph states "a decode step takes 5 ms (5,000 us) end-to-end and contains **64 ops**." The reference model in `ch6_reference_implementation/traced_decode_loop.md` is canonically defined as 46 ops (2 layers × 22 ops + 1 lm_head + 1 copy_), and `ch6_reference_implementation/index.md` cites "32–64 ops in a decode step" as the range. The 64-op figure is at the high end of that stated range and is consistent with it, but it is never explained where 64 ops come from — the guide's only concrete model has 46 ops. A reader who has read Ch6 will find the example inconsistent with the only fully specified model in the guide.
   **Fix:** Either use 46 ops (the Ch6 reference model) as the concrete example in Ch4/index.md, or add a parenthetical noting that 64 ops represents a larger model configuration than the Ch6 reference.

4. **Guide-level index.md Chapter Index — Ch2 description uses "async op mode" while Ch2 title is "Async Op Execution" but the chapter's own index.md heading is "Asynchronous Op Execution."**
   The guide-level `index.md` Chapter Index table (row 2) titles Ch2 as "Ch 2 — Async Op Execution." The Ch2 `index.md` H1 heading is "Chapter 2 — Asynchronous Op Execution." The guide-level "How to Use" table links it as "Ch 2 — Async Op Execution." The Ch2 content consistently uses "async op mode" in prose. The mismatch is between the full-word heading ("Asynchronous") and the abbreviated title ("Async") used in the guide index — a minor inconsistency but visible to any reader who clicks through.
   **Fix:** Align the guide-level Chapter Index and "How to Use" table to "Async Op Execution" (consistent with the existing link text) and update Ch2's own H1 heading from "Asynchronous Op Execution" to "Async Op Execution" — or reverse the direction and update the guide index table entry to "Asynchronous Op Execution."

5. **Ch5/estimating_trace_speedup.md — the "What Trace Eliminates" table net-effect claim does not match the phase cost ranges after the previous pass's fix.**
   After the Cross-Chapter Consistency fix, the Command encoding row in the "What Trace Eliminates" table was aligned to 10–40 us. The table's four rows now sum to: validation 5–15 us + selection 1–3 us + encoding 10–40 us + CQ submission 1–5 us = 17–63 us per op. The text immediately following the table states "Net effect per op on replay: approximately **16–58 us** of eliminated overhead." But 17–63 us total minus 1–5 us (the CQ submission cost that is not fully eliminated, only reduced) gives 12–58 us, not 16–58 us — the lower bound is wrong. Subtracting the maximum CQ cost (5 us) from the minimum total (17 us) gives 12 us, not 16 us.
   **Fix:** Correct the net-effect sentence to "approximately **12–58 us** of eliminated overhead per op" (17 − 5 = 12 us minimum, 63 − 5 = 58 us maximum).

---

## Change Log — Final Pass Cross-Chapter Fixes

- `latency_sensitive_workloads.md` (Ch4): Aligned trace replay range to 7–15 us (Ch3 authoritative).
- `ch4/index.md`: Corrected disqualifying condition count to 3 numbered + 1 addendum; updated 64-op example to 46 ops.
- `index.md` (guide): Aligned Ch2 title to "Asynchronous Op Execution".
- `estimating_trace_speedup.md` (Ch5): Corrected net-effect lower bound to 12–58 us.
- `trace_internals.md` (Ch3): Replaced duplicate phase cost table with prose cross-reference to Ch1.
- `ch6/index.md`: Replaced duplicate "Why decode is canonical" section with 2-sentence forward reference to Ch4.
