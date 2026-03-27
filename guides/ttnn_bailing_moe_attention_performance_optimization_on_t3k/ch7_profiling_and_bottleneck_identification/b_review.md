# B Review — Ch7 Pass 5

## Verdict: APPROVED

## Issues (if any)

None. All prior flagged issues have been resolved:

- Thread count (tracy_profiling.md line 57) now correctly states "up to 9 CPU lanes" (1 main + 8 dispatch threads).
- Reshape count in Branch E (bottleneck_decision_tree.md lines 24, 103, 111, 113) now reads ×8 total (4 per norm call × 2 for Q and K), with validation instructions updated to match.
- `ttnn_op_timers.md` op-name mapping table (line 94) now annotates `ttnn::reshape (×8, four per norm call for Q and K each)`, consistent with the Branch E body.
- `ttnn::from_torch` ranges are now coherent: without-sync 2–10 µs, with-sync 5–15 µs (ttnn_op_timers.md lines 167 and 208). The with-sync upper bound (15 µs) is strictly above the without-sync upper bound (10 µs), which is physically correct.

## Notes

- The decision tree trunk order (A → B → C → D → E → F → H → G) places Branch G (catch-all success condition) after Branch H (Paged Fill Cache). This is deliberate: G is the fall-through for all unmatched cases. It is not a structural gap — a reader who matches H exits before reaching G, and a reader who matches nothing reaches G correctly. No action needed.
- The Branch E symptom header (line 103) reads "These ten ops (8 reshapes + 2 RMSNorm calls)" and the decision tree trigger at line 24 reads `ttnn::reshape × 8`. These are now consistent with the confirmed reshape count from prior fixes. No issues remain in this area.
- All `[ESTIMATE]` labels in the chapter are appropriately applied to figures that depend on firmware revision, hardware state, and model configuration. The chapter does not overclaim measured values as estimates or vice versa.
