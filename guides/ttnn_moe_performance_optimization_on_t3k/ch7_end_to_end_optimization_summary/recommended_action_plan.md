# Recommended Action Plan

## Context

This document addresses all research questions **Q1–Q8**. It sequences the findings from the Priority Matrix into a phase-by-phase action plan with explicit code locations, measurement gates, and model-specific notes. Execute phases in order. Do not skip Phase 1.

---

## Phase 1 — Establish Baseline (before any optimization)

1. **Confirm the active expert implementation.** Run the Ch6 module audit (`audit_moe_modules`) to determine whether `Glm4MoeNaiveMoeHybrid` is active. If it is, the baseline timing data will be dominated by CPU execution and will not reflect TTNN performance. Apply Phase 2 before collecting any baseline numbers.

2. **Collect per-op timing baseline.** Run the `ttnn_op_timer_profiling` harness (Ch5) with 100-pass warmup followed by 1000-pass measurement. Export the results to CSV. Record the per-op median latency for every stage in `TTNNMoE.forward` (`moe.py:L1346–L1496`) and `TTNNExperts.forward` (`moe.py:L1027–L1343`). This CSV is the baseline against which all subsequent phases are measured.

3. **Compute the CCL fraction.** Calculate:

   ```
   ccl_fraction = (all_gather_latency + reduce_scatter_latency) / total_forward_latency
   ```

   If `ccl_fraction > 0.50`, CCL is the primary bottleneck and Phase 3 should be executed before Phase 4. If `ccl_fraction < 0.50`, expert compute is unexpectedly dominant — re-examine the baseline for measurement artifacts before proceeding.

---

## Phase 2 — CPU Fallback Elimination (applies only if `Glm4MoeNaiveMoeHybrid` is active)

4. **Apply the one-line fix.** Set `ttnn = True` at `moe.py:L569`. This switches `Glm4MoeNaiveMoeHybrid` from instantiating `Glm4MoeExpertLayersTorch` (CPU sequential loop) to `TTNNGlm4MoeExpertLayers` (Tensix kernels).
   - Code location: `moe.py:L569`
   - Applies to: GLM-4 path (`TTNNGlm4MoeMoE`) only; `TTNNMoE` and `TTNNBailingMoE` are unaffected

5. **Verify TTNN path activation.** Run the split `gate_up_proj` weight compatibility check from Ch6 Step 2. Then run the Ch6 detection harness to confirm that no host-device transfer activity is observed during expert computation. Run a numeric validation comparing output tensors before and after the change; expect small floating-point differences but identical routing decisions.

6. **Handle weight format incompatibilities if `TTNNGlm4MoeExpertLayers.from_parameters` fails.** The fused `gate_up_proj` tensor in `Glm4MoeExpertLayersTorch` must be split before loading into `TTNNGlm4MoeExpertLayers`:

   ```python
   w1 = gate_up_proj[e, :intermediate_dim, :]   # gate projection
   w3 = gate_up_proj[e, intermediate_dim:, :]   # up projection
   ```

   See Ch6 `glm4_cpu_path_audit.md` for the full weight format migration checklist.

7. **Re-collect baseline after Phase 2.** The expert compute latency should drop by orders of magnitude. Record the new per-op median latencies as the updated baseline. All subsequent phases compare against this post-Phase-2 baseline.

---

## Phase 3 — CCL Parameter Tuning

8. **Sweep `reduce_scatter_minimal_async` parameters.** The current configuration at `moe.py:L1478–L1490` uses `chunks_per_sync=10` and `num_workers_per_link=2`. Run the TTNN op timer harness for each combination in the sweep grid:

   ```
   chunks_per_sync      ∈ {1, 5, 10, 20}
   num_workers_per_link ∈ {1, 2}
   ```

   Record median reduce_scatter latency for each combination. Select the configuration with the lowest median. This is a Ring topology operation — do not change the topology.

9. **Current defaults for reference:**
   - `reduce_scatter_minimal_async`: `moe.py:L1478–L1490`, Ring topology, `chunks_per_sync=10`, `num_workers_per_link=2`
   - `all_gather_async`: `moe.py:L1429–L1436`, Linear topology, `num_links=1`

10. **Evaluate `all_gather_async` topology.** The current Linear topology is appropriate for a 1×8 mesh. Optionally, profile a Ring topology configuration for `all_gather_async` to determine whether latency improves. Because `all_gather_async` contributes ~1 µs at baseline (Ch2), the expected gain is small; do not change the topology unless profiling confirms a meaningful reduction.

11. **Measurement gate.** The sweep is successful if: `reduce_scatter_latency` decreases by ≥5% from the Phase 2 baseline, and total `TTNNMoE.forward` latency decreases proportionally. If no improvement is found, retain the original `chunks_per_sync=10, num_workers_per_link=2` defaults.

---

## Phase 4 — Expert Compute Tuning

12. **Tune `_make_sparse_matmul_program_config`.** The current implementation at `moe.py:L62–L91` uses `in0_block_w=min(4, hidden_tiles)` and `per_core_M=1`. For Bailing's hidden/intermediate sizes, `hidden_tiles` differs from GLM-4-MoE, making `in0_block_w=4` the effective cap for GLM-4 but potentially suboptimal for Bailing. Run the Ch4 benchmark harness sweeping:

    ```
    in0_block_w ∈ {1, 2, 4, 8}
    per_core_M  ∈ {1, 2, 4}
    ```

    Sweep separately for GLM-4-MoE and Bailing hidden/intermediate size combinations. Select the configuration with the lowest sparse_matmul latency that does not exceed available L1 capacity.

13. **Evaluate LoFi math fidelity for expert matmuls.** Run the Ch4 accuracy harness comparing expert matmul outputs against an FP32 reference. The acceptance criteria are:

    ```
    cosine_similarity  >= 0.999
    max_absolute_error <= 0.5% of output norm
    ```

    If both criteria are met, change the expert matmul math fidelity from `ttnn.MathFidelity.HiFi2` to `ttnn.MathFidelity.LoFi` in `_make_sparse_matmul_program_config`. The gate linear uses `HiFi4` and must not be changed — `HiFi4` is required for routing accuracy.

14. **Measurement gate.** Expert matmul latency (the three `sparse_matmul` calls in `TTNNExperts.forward`) decreases by ≥10% from the Phase 3 baseline.

---

## Phase 5 — Secondary Optimizations

15. **Profile weight application overhead.** Locate the repeat+permute block after `all_to_all_combine` in `TTNNExperts.forward`. Using the Phase 1 baseline CSV, check whether this block exceeds 5% of total `TTNNExperts.forward` time. If it does, implement the elementwise-after-reshape alternative from Ch3 and benchmark both implementations. Adopt the alternative only if it reduces weight application latency without introducing correctness regressions.

16. **Profile `TTNNMoERouterDecode` 3-pass centering.** The 3-pass BF16 centering logic at `moe.py:L891–L1024` is only executed when `n_group <= r.topk_group`. Use the Ch5 harness to measure total centering latency in your deployment. If total centering latency exceeds 30 µs and the `n_group <= r.topk_group` condition is typically true, implement the single-pass topk baseline from Ch5 and measure the routing agreement rate against the 3-pass result. Only replace the 3-pass centering with the single-pass version if agreement rate is ≥99.9%. Routing errors compound across decode steps and are not recoverable at inference time.

17. **Measurement gate.** Total `TTNNMoE.forward` latency is within 10% of the CCL-only theoretical minimum — that is, all non-CCL ops (expert compute, routing, weight application) complete in less time than the dominant CCL op (`reduce_scatter_minimal_async` at ~28 µs). When this condition holds, the MoE forward pass is CCL-bound and no further software optimization is possible without addressing CCL latency at the hardware or firmware level.

---

## Model-Specific Notes

**GLM-4-MoE (`TTNNGlm4MoeMoE`):**
- Phase 2 (CPU fallback elimination) is relevant and should be executed first if `Glm4MoeNaiveMoeHybrid` is active.
- In `_make_sparse_matmul_program_config`, `in0_block_w=min(4, hidden_tiles)` evaluates to `in0_block_w=4` at GLM-4-MoE's hidden size. The program config sweep in Phase 4 Step 12 may confirm this as optimal, but it should still be measured explicitly rather than assumed.
- The split `gate_up_proj` weight format requires the Ch6 migration path (Step 6) when enabling the TTNN expert path.

**Bailing (`TTNNBailingMoE`):**
- `TTNNBailingMoE` is a subclass of `TTNNMoE` and inherits `TTNNMoE.forward` unchanged. Phase 2 does not apply.
- Phases 3–5 apply identically to Bailing.
- Bailing uses different hidden and intermediate sizes from GLM-4-MoE. The program config sweep in Phase 4 Step 12 will yield different optimal values. Do not reuse GLM-4-MoE sweep results for Bailing.

---

**Previous:** [Optimization Priority Matrix](optimization_priority_matrix.md)
