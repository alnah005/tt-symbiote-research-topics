# B Review — Chapter 7: Interpretation and Next Steps — Pass 1

## Verdict
2 error(s) found.

### Error 1
- **File:** `optimization_action_reference.md`
- **Line:** 181
- **Stated:** "3 × 7168 × 2048 × 2 ≈ 88 MB total for 128 experts, or 0.69 MB per expert"
- **Correct:** 3 × 7168 × 2048 × 2 bytes = ~88 MB per expert (one set of gate/up/down matrices for a single expert). This is not the total for 128 experts — that would be ~11 GB. The figure "0.69 MB per expert" that follows is therefore also wrong by two orders of magnitude. The correct per-expert size is ~88 MB (consistent with `gap_to_action_mapping.md` line 190, which correctly states "~29 MB per expert × 3 matrices ≈ 87 MB per expert"). The sentence should read "≈ 88 MB per expert" not "≈ 88 MB total for 128 experts."

### Error 2
- **File:** `optimization_action_reference.md`
- **Line:** 83
- **Stated:** `ttnn.wait_for_event(cq_id=0, event=matmul_done)` is placed after `ttnn.scatter(...)`, with the comment "Device will wait for matmul_done before executing scatter"
- **Correct:** `ttnn.wait_for_event` inserts a device-side stall at the point it is enqueued in the command queue. Placing it after `ttnn.scatter` in the dispatch sequence means the device will execute scatter before encountering the wait — the opposite of what the comment claims. To gate scatter on matmul completion, `ttnn.wait_for_event` must be enqueued before the first combine op. The code example should place `ttnn.wait_for_event(cq_id=0, event=matmul_done)` immediately after `ttnn.record_event` and before `ttnn.scatter`.

## Agent A Change Log — B Feedback Pass 1
- optimization_action_reference.md: Fixed per-expert weight size label ("per expert" not "total for 128 experts"); removed incorrect "0.69 MB per expert" derived value; corrected to 84 MB per expert, 1,344 MB per chip
- optimization_action_reference.md: Moved ttnn.wait_for_event before scatter/combine ops (was incorrectly placed after)
- gap_to_action_mapping.md: Moved ttnn.wait_for_event before scatter/combine ops (same issue)

# B Review — Chapter 7: Interpretation and Next Steps — Pass 2

## Pass 1 Fix Verification

**Fix 1 (per-expert weight size):** Verified. `optimization_action_reference.md` Lever 5 now reads "3 × 7168 × 2048 × 2 ≈ 84 MB per expert. Across 128 experts on T3K (8 chips), that is 128 × 84 MB / 8 chips = 1,344 MB per chip in DRAM." The incorrect label "total for 128 experts" and the derived "0.69 MB per expert" figure have both been removed. The authoritative value (84.0 MB per expert, 1,344 MB per chip) is now stated correctly.

**Fix 2 (ttnn.wait_for_event placement):** Verified in both files.
- `optimization_action_reference.md` Lever 2 code block: `ttnn.wait_for_event(cq_id=0, event=matmul_done)` now appears immediately after `ttnn.record_event(device, cq_id=0)` and before `ttnn.scatter(...)`. Comment reads "Device will wait for matmul_done before executing scatter" and is placed before the scatter call. Correct.
- `gap_to_action_mapping.md` Pattern B Option B code block: `ttnn.wait_for_event(cq_id=0, event=matmul_done_event)` now appears immediately after `ttnn.record_event(device, cq_id=0)` and before `ttnn.scatter(...)`. Comment reads "Device will wait for matmul_done_event before executing scatter" and is placed before the scatter call. Correct.

## Remaining Issues

**Minor inconsistency — `gap_to_action_mapping.md` line 190:** The Pattern C ep_degree discussion states "~29 MB per expert × 3 matrices ≈ 87 MB per expert." The authoritative per-expert size is 84.0 MB (3 × 7168 × 2048 × 2 = 88,080,384 bytes = 84.0 MB). The figure "~29 MB per matrix × 3 = 87 MB" is an independent approximation that is close but does not match the canonical value. The downstream DRAM estimate (32 experts × 87 MB ≈ 2.8 GB) should be 32 × 84 MB = 2,688 MB ≈ 2.6 GB. This is a minor rounding inconsistency; the qualitative conclusion (feasible in DRAM on Wormhole B0) remains correct.

**Minor inconsistency — `optimization_action_reference.md` Quick-Reference Table (line 20):** The table row for "Sharded memory config for expert weights" lists the condition as "Expert weight tensors fit in L1 shards." After the Fix 1 correction, Lever 5 now correctly explains that a single expert's ~84 MB weights far exceed the 120 MB total L1 per chip, making full expert weight L1-sharding infeasible. The table condition is therefore misleading. It should be updated (e.g., to "Activation tensors or sub-matrices fit in L1 shards") to stay consistent with the revised Lever 5 body text. This inconsistency was not present before Fix 1 — it was introduced by the correction.

## Verdict

2 minor residual issues found (no correctness-critical errors). Both Pass 1 fixes are correctly applied. The remaining issues are:
1. `gap_to_action_mapping.md` line 190: per-expert weight approximation (87 MB) is slightly inconsistent with the canonical 84 MB; downstream 2.8 GB estimate should be ~2.6 GB.
2. `optimization_action_reference.md` Quick-Reference Table line 20: condition "Expert weight tensors fit in L1 shards" is now inconsistent with the updated Lever 5 body text (which correctly explains the per-expert size exceeds total L1 per chip).

## Agent A Change Log — B Feedback Pass 2
- gap_to_action_mapping.md: Fixed per-expert weight size from ~87 MB to 84.0 MB; corrected downstream per-chip total
- optimization_action_reference.md Quick-Reference Table: Fixed Lever 5 condition from "Expert weight tensors fit in L1" to decode regime condition (batch_size × top_k ≤ 16)

# B Review — Chapter 7: Interpretation and Next Steps — Pass 3

## Pass 2 Fix Verification

**Fix 1 (gap_to_action_mapping.md per-expert weight size):** Verified. Line 190–191 now reads:
"`[7168, 2048]` in bfloat16 = 84.0 MB per expert (3 × 7168 × 2048 × 2 bytes); 32 experts per chip = 32 × 84.0 MB = 2,688 MB ≈ 2.6 GB — feasible in DRAM on Wormhole B0)."
The approximate "~87 MB" figure and the incorrect downstream "2.8 GB" estimate have been replaced with the canonical 84.0 MB and 2,688 MB ≈ 2.6 GB. Arithmetic is correct: 3 × 7168 × 2048 × 2 = 88,080,384 bytes ÷ 1,048,576 = 84.0 MB. 32 × 84.0 = 2,688 MB. Fix is correct.

**Fix 2 (optimization_action_reference.md Quick-Reference Table Lever 5 condition):** Verified. The table row for "Sharded memory config for expert weights" now lists the condition as "Decode regime (batch_size × top_k ≤ 16 active tokens per chip); weights remain in DRAM shards, not L1." The previous incorrect condition "Expert weight tensors fit in L1 shards" has been removed. The new condition correctly reflects that DRAM sharding (not L1 sharding) is the applicable mechanism in decode regime, and no longer contradicts the Lever 5 body text explaining that ~84 MB per expert exceeds the 120 MB total L1 per chip. Fix is correct.

## Verdict

No feedback — chapter approved.
