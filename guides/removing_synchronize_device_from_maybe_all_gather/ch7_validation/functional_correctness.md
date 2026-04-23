# Functional Correctness

This file describes the numerical correctness test that confirms the modified `_maybe_all_gather` — whether Type A (synchronize_device deleted, synchronous all_gather kept) or Type B2 (all_gather_async with cycling semaphores) — produces per-token output tensors that are numerically indistinguishable from the original implementation. By the end of this file you will understand how to run the PCC comparison, how to interpret a failure, and how to extend the test to cover the full hybrid decoder stack.

---

## Test Setup

Run both the reference implementation and the modified implementation in **non-traced mode** (no `begin_trace_capture` / `end_trace_capture` bracket). Non-traced mode isolates the correctness of the all_gather operation itself from any trace-replay-specific issues; those are addressed in [`multi_replay_stability.md`](./multi_replay_stability.md).

**Reference:** The original `_maybe_all_gather` with synchronous `ttnn.all_gather` and `ttnn.synchronize_device()`.

**Modified:** The implementation from [Chapter 6](../ch6_implementation/structural_changes.md): either `ttnn.all_gather` without `synchronize_device` (Type A) or `ttnn.experimental.all_gather_async` with cycling semaphores and without `synchronize_device` (Type B2).

Both runs must use:
- Identical model weights
- Identical input token ids and positions (same prefill KV cache state, if applicable)
- Identical mesh device configuration (same T3K layout, same cluster_axis values)
- Identical dtype and memory config settings

---

## PCC Comparison Procedure

For each attention module under test, compute the Pearson Correlation Coefficient (PCC) between the reference output tensor and the modified output tensor across the full hidden dimension:

```python
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# Bring both output tensors to CPU for comparison
ref_output_cpu  = ttnn.to_torch(ref_output)    # [batch, seq_len, hidden_dim]
mod_output_cpu  = ttnn.to_torch(mod_output)    # same shape

# Compute PCC across the hidden dimension, averaged over batch and sequence
pcc_value, pcc_str = comp_pcc(ref_output_cpu, mod_output_cpu, pcc=0.999)

assert pcc_value, (
    f"PCC check failed: {pcc_str}. "
    f"Modified _maybe_all_gather output does not match reference."
)
```

**Acceptance criterion:** PCC > 0.999 across all sequence positions and all batch elements.

> **Note:** PCC > 0.999 is the minimum threshold for attention output. For hidden states that feed directly into softmax or layer normalization (where small numerical deviations compound across layers), prefer PCC > 0.9999 if the model's stacking depth is large.

---

## Test Coverage

Run the PCC comparison for both affected modules individually, then for the combined stack:

### Test 1A — TTNNQwen3FullAttention in isolation

Run a single `TTNNQwen3FullAttention.forward` decode step with a random input tensor of shape `[batch=1, seq_len=1, hidden_dim]`. Compare output against reference. Confirm PCC > 0.999.

### Test 1B — TTNNQwen3LinearAttention in isolation

Run a single `TTNNQwen3LinearAttention.forward` decode step with the same approach. Confirm PCC > 0.999.

> **Note:** If `_maybe_all_gather` is a shared base-class method, Tests 1A and 1B both exercise the same method implementation via different callers. A failure in one but not the other would indicate a call-site-specific issue (different `cluster_axis`, different memory config, or different tensor shape at the call site) rather than a bug in the shared method body.

### Test 1C — Full hybrid decoder layer (combined stack)

Run one `TTNNQwen3LinearAttention` followed by one `TTNNQwen3FullAttention` in a single forward pass, simulating one hybrid decoder layer. Feed the output of the linear attention as input to the full attention (or use the same input to both if they operate on separate tensor streams). Compare both outputs against their respective reference runs. Confirm PCC > 0.999 for each output.

This test confirms that the two modules' `_maybe_all_gather` calls do not interfere with each other when executed in sequence — for example, that the cycling semaphore state is consistent across both calls.

---

## Interpreting a PCC Failure

### Failure Pattern 1 — Low PCC with spatially correlated error

Error is concentrated in specific device shards or specific token positions that correspond to rank boundaries in the all_gather output:

- **Most likely cause:** Semaphore initialization bug (Type B2 only). The `GlobalSemaphore` was not reset to zero before the `all_gather_async` call, causing the device-side kernel to read a stale completion signal and return before the all_gather is complete. The output buffer contains partial data from only some ranks.
- **Diagnostic step:** Add a `ttnn.reset_global_semaphore_value(handle, 0)` call immediately before `all_gather_async` in `_maybe_all_gather` (not just in the trace wrapper) and re-run. If PCC recovers, the semaphore was not being reset correctly in steady-state.

### Failure Pattern 2 — Low PCC with random error distribution

Error is spread uniformly across the hidden dimension with no spatial correlation to rank boundaries:

- **Most likely cause:** Race condition — the downstream op is reading the all_gather output buffer before the all_gather has delivered all data to that buffer. This can occur if:
  - (Type B2) The `multi_device_global_semaphore` mechanism is misconfigured and the device-side barrier is never triggered, allowing the downstream op to start immediately.
  - (Type A) There is a genuine multi-CQ dependency that CQ0 ordering does not cover — though this should not occur in the single-CQ trace-compatible deployment (see [Chapter 3](../ch3_root_cause_analysis/command_queue_ordering_guarantee.md)).
- **Diagnostic step:** For Type B2, verify that the semaphore handles returned by `get_and_cycle_ag_semaphore_handles` and `get_and_cycle_barrier_semaphore_handle` are valid (non-None) objects with correctly allocated L1 addresses. For Type A, verify that no multi-CQ dispatch path is active.

### Failure Pattern 3 — PCC near 1.0 but below threshold (e.g., 0.998)

Small but consistent numerical deviation across all positions:

- **Most likely cause:** Not a race condition or semaphore bug. More likely a dtype rounding difference introduced by a memory config change, or a difference in how the all_gather accumulates values when the input is in a different layout. Compare the memory configs of the reference and modified implementations.
- **Diagnostic step:** Run with `ttnn.bfloat16` vs. `ttnn.float32` accumulation if configurable, and confirm that the memory config in `all_gather_async` matches the original `all_gather`'s memory config exactly.

---

## Hybrid Stack Test: Sequence Interaction Check

After Tests 1A–1C pass individually, run a 4-layer alternating stack:

```
TTNNQwen3LinearAttention [layer 0] → TTNNQwen3FullAttention [layer 0]
→ TTNNQwen3LinearAttention [layer 1] → TTNNQwen3FullAttention [layer 1]
```

Compare the final output of the 4-layer stack against the reference. PCC > 0.999 here confirms that:
- Cycling semaphore counters advance and reset correctly across layers.
- No L1 aliasing occurs between the linear and full attention modules' semaphore pools.
- The all_gather output buffers (program-cached) do not interfere across module instances.

A failure in this test that was absent in Tests 1A–1C indicates an inter-layer interaction, most likely a semaphore alias between the two modules' `TT_CCL` semaphore slots.
