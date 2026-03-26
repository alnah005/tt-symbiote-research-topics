# B Review — Chapter 5 — Pass 1

1. **`index.md` L13 — 3-pass centering stated as unconditional when it is branch-conditional.**
   The sentence reads: "`TTNNMoERouterDecode.forward` … runs the 3-pass BF16 centering trick before `TTNNExperts` sees any tokens." Ground truth confirms the 3-pass centering only runs inside the `n_group <= r.topk_group` branch (`moe.py:L925–L957`). The group-based path (`n_group > topk_group`) does not use it. A reader who takes this statement at face value will incorrectly believe the router always pays the 3-pass overhead and will misattribute latency when profiling the group-based routing path.

2. **`router_latency_profiling.md` L15 — wrong numerical value cited for smallest positive BF16 subnormal.**
   The document states "the smallest positive BF16 subnormal is ~1.2e-38". BF16 has 8 exponent bits and 7 mantissa bits. The smallest positive **normal** BF16 value is 2^(-126) ≈ 1.18e-38; the smallest positive **subnormal** BF16 value is 2^(-126) × 2^(-7) = 2^(-133) ≈ 9.2e-41. The document has conflated the two. A reader computing minimum representable differences from this figure will be off by a factor of ~128.

# B Review — Chapter 5 — Pass 2

1. **`tracy_profiling_setup.md` Step 5 and `ttnn_op_timer_profiling.md` Step 3 — `pre_norm` stage at `moe.py:L1477` is absent from all annotation guides and mapping tables.**
   Ground truth: `TTNNMoE.forward` contains a `pre_norm` step at `L1477`, between `L1471` (TTNNExperts.forward) and `L1478–L1490` (reduce_scatter_minimal_async). Neither the Tracy zone annotation list in Step 5 of `tracy_profiling_setup.md` nor the CSV-to-source mapping table in Step 3 of `ttnn_op_timer_profiling.md` includes this stage. A reader following these guides will have a silent gap in the annotated timeline and in the mapping table; the pre_norm kernel's device cycles will be unaccounted for or silently merged into the reduce_scatter attribution, producing an inflated reduce_scatter latency figure and an incorrect critical-path analysis.

2. **`router_latency_profiling.md` L283–L285 — internal numerical inconsistency between per-pass overhead range and the example 3-pass total.**
   L283 states each additional pass adds ~10–30 µs. The 3-pass path has 2 additional passes relative to single-pass, giving an overhead range of 20–60 µs. L285 then states: "if a single-pass topk takes ~15 µs, the 3-pass variant takes ~35–45 µs." The upper bound of 45 µs is inconsistent with the stated per-pass range: 15 + 60 = 75 µs at the high end. A reader using the "~35–45 µs" figure to budget router latency could underestimate 3-pass cost by up to ~30 µs in a high-overhead scenario.

# B Review — Chapter 5 — Pass 3

1. **`router_latency_profiling.md` L285 — "2×–3×" multiplier lower bound is inconsistent with the stated 35–45 µs example.**
   L285 states: "the production 3-pass path thus costs approximately 2×–3× the latency of a single-pass topk. At batch=1, if a single-pass topk takes ~15 µs, the 3-pass variant takes ~35–45 µs." The "2×" lower bound applied to 15 µs yields 30 µs, not 35 µs. The 35 µs lower bound of the concrete example corresponds to 2.33× of 15 µs. A reader who uses the "2×–3×" ratio directly with a measured single-pass baseline of 15 µs will compute a range of 30–45 µs, underestimating the lower bound by 5 µs and arriving at an incorrect budget for the 3-pass path. The correct multiplier consistent with the 35–45 µs example and the ~10–15 µs per-pass overhead (ground truth) is approximately 2.3×–3×.

# B Review — Chapter 5 — Pass 4

No feedback — chapter approved.

# B Review — Chapter 5 — Pass 5

No feedback — chapter approved.
