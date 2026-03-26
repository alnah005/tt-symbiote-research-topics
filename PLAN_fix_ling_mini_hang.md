# Plan: Fix Ling-mini-2.0 T3K Hang

## Problem Description

The test `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py` **hangs indefinitely** when run on T3K (1x8 mesh device with 8 Wormhole chips).

The test completes model loading and weight preprocessing (44 TTNN modules), then hangs during the first `model.generate()` call (warmup at line 125). No output appears after the weight preprocessing progress bar reaches 100%.

## Root Cause Analysis

### Direct Cause: `ttnn.all_reduce` Hangs with 8 Devices on T3K

The hang is caused by `ttnn.all_reduce()` being called across all 8 devices on the T3K mesh. This is a **known, documented issue** in the TTNN CCL test suite.

**Evidence:**
- File: `tests/nightly/t3000/ccl/test_all_reduce.py`, line 22-23:
  ```python
  # (8, 1), # skipped as 8 devices result in hang in all gather
  ```
- The 8-device `all_reduce` test case is explicitly commented out/skipped because it causes a hang in the all-gather phase of the composite all_reduce operation.
- Only 4-device `all_reduce` and 2-device `all_reduce` tests are enabled for T3K.

### Code Path to the Hang

1. Test calls `model.generate(**inputs, max_new_tokens=2, ...)` (line 125)
2. HuggingFace generate runs the prefill forward pass
3. Forward pass reaches the first attention layer (`TTNNBailingMoEAttention`)
4. Attention calls `self.qkv_proj(hidden_states)` (line 2543 for prefill, line 2632 for decode)
5. `qkv_proj` is an instance of `TTNNLinearIColShardedWAllReduced`
6. `TTNNLinearIColShardedWAllReduced.forward()` calls:
   ```python
   tt_output = ttnn.all_reduce(
       tt_output,
       num_links=1,
       topology=ttnn.Topology.Ring,
       cluster_axis=1,        # axis 1 has 8 devices on T3K
       memory_config=ttnn.DRAM_MEMORY_CONFIG,
   )
   ```
7. This `all_reduce` across 8 devices **hangs**.

### Why This Was Not Caught Earlier

The `TTNNLinearIColShardedWAllReduced` class was created as part of the "Fused QKV + AllReduce" optimization plan. The prior research (see "Fused QKV + AllReduce Detailed Implementation Plan" in research_topics.md) explicitly identified this risk:
> "RISK: T3K test skips 8-device all_reduce due to 'hang in all gather'; fallback is composite RS+AG (2 ops, still saves 3)"

However, the fallback was not implemented -- the code went directly to using `ttnn.all_reduce` without the composite fallback.

### Why Other Models (gpt_oss/DeepSeek V3) Don't Hang

The `gpt_oss` (DeepSeek V3) model also uses `ttnn.all_reduce` on T3K with 8 devices and Ring topology. However, it runs on a different code path with different initialization sequences and the CCL manager may handle semaphore coordination differently. The key difference is that gpt_oss is a native tt-metal demo model with its own CCL manager (`TT_CCL` class) that initializes and manages semaphores explicitly, while tt_symbiote's `TTNNLinearIColShardedWAllReduced` calls `ttnn.all_reduce` directly without semaphore management.

The MoE module (`TTNNMoE.forward()`) in the SAME model works correctly because it uses the async CCL variants:
- `ttnn.experimental.all_gather_async()` with `multi_device_global_semaphore` and `barrier_semaphore`
- `ttnn.experimental.reduce_scatter_minimal_async()` with `multi_device_global_semaphore` and `barrier_semaphore`

These async variants use proper semaphore coordination via `device_state.ccl_manager`, avoiding the deadlock.

## Affected Files

| File | What's Affected |
|------|----------------|
| `models/experimental/tt_symbiote/modules/linear.py` (line 218) | `TTNNLinearIColShardedWAllReduced.forward()` -- the direct `ttnn.all_reduce` call |
| `models/experimental/tt_symbiote/modules/attention.py` (line 2376) | `TTNNBailingMoEAttention.from_torch()` -- creates `qkv_proj` as `TTNNLinearIColShardedWAllReduced` |
| `models/experimental/tt_symbiote/modules/attention.py` (line 2543, 2632) | Prefill and decode paths that call `self.qkv_proj()` |

## Fix Options

### Option A: Replace `all_reduce` with Composite `reduce_scatter` + `all_gather` (Recommended)

Replace the single `ttnn.all_reduce` call with two separate operations that are known to work on T3K:

```python
# BEFORE (hangs on T3K with 8 devices):
tt_output = ttnn.all_reduce(
    tt_output,
    num_links=1,
    topology=ttnn.Topology.Ring,
    cluster_axis=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# AFTER (works on T3K):
# Step 1: reduce_scatter -- each device gets 1/8 of the reduced output
tt_output = ttnn.experimental.reduce_scatter_minimal_async(
    tt_output,
    persistent_output_buffers=None,
    dim=3,  # reduce along last dimension
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    cluster_axis=1,
    topology=ttnn.Topology.Ring,
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
# Step 2: all_gather -- replicate the reduced result to all devices
tt_output = ttnn.experimental.all_gather_async(
    tt_output,
    dim=-1,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    topology=ttnn.Topology.Linear,
)
```

**Pros:**
- Uses the exact same CCL pattern proven to work in `TTNNMoE.forward()` on the same model
- Async variants with semaphore management are well-tested on T3K
- No external dependency changes

**Cons:**
- 2 CCL ops instead of 1 (slightly higher latency)
- Need access to `device_state.ccl_manager` -- the linear class doesn't currently have this

### Option B: Revert to Separate Q/K/V Projections

Revert the fused QKV optimization and go back to separate Q, K, V projections with individual all-gather operations:

```python
# Use 3 separate TTNNLinearIColShardedWRowSharded for Q, K, V
# Each does: matmul + reduce_scatter (works on T3K)
q = self.q_proj(hidden_states)  # reduce_scatter output
k = self.k_proj(hidden_states)  # reduce_scatter output
v = self.v_proj(hidden_states)  # reduce_scatter output
# Then all_gather Q/K/V individually if needed for attention
```

**Pros:**
- Well-tested pattern that works on T3K
- Simpler code

**Cons:**
- Loses the performance benefit of fused QKV (3 matmuls + 3 CCL ops vs 1 matmul + 2 CCL ops)
- Was the original code before the optimization

### Option C: Use Synchronous `all_gather` + Manual Reduction (Fallback)

```python
# all_gather the partial sums, then reduce locally
gathered = ttnn.all_gather(tt_output, dim=0, num_links=1)  # Gather all partials
tt_output = ttnn.sum(gathered, dim=0)  # Local reduction
```

**Pros:**
- Simple to implement

**Cons:**
- Very inefficient -- transfers 8x data then reduces locally
- Not a real solution, just a workaround

## Recommended Implementation Plan

### Step 1: Modify `TTNNLinearIColShardedWAllReduced` to Use Composite RS+AG

**File:** `models/experimental/tt_symbiote/modules/linear.py`

1. Add `device_state` access to the forward method (the class inherits from `TTNNModule` which has `self.device_state`)
2. Replace `ttnn.all_reduce()` with `reduce_scatter_minimal_async()` + `all_gather_async()`
3. Handle the dimension carefully: reduce_scatter reduces along dim=3 (last dim), then all_gather restores it along dim=-1

The key challenge is that after `reduce_scatter`, the output on each device has `output_size / num_devices` elements along the reduced dimension. The subsequent `all_gather` then concatenates these back to the full size. This produces the same result as `all_reduce` (sum of all partial products, replicated to all devices).

### Step 2: Handle the Bias Addition Correctly

After `all_reduce`, the bias is added. With the composite RS+AG approach:
- After `reduce_scatter`, each device has 1/8 of the output -- bias cannot be added here
- After `all_gather`, each device has the full output -- bias can be added here
- Current code adds bias after `all_reduce` (line 225-226) -- this should remain unchanged

### Step 3: Test on T3K

```bash
cd /home/ttuser/salnahari && source tt_bashrc && cd tt-metal
unset TT_VISIBLE_DEVICES
tt-smi -r
timeout 600 python -m pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v --timeout=0 -s 2>&1 | tail -50
```

### Step 4: Verify Correctness

1. Compare output text with reference (should be coherent text, not garbage)
2. Compare attention output numerics with PyTorch reference (PCC > 0.99)
3. Ensure the test generates 128 tokens without hanging or crashing

## Success Criteria

1. **No hang:** `test_ling_mini_2_0.py` completes within 10 minutes on T3K
2. **Correct output:** Generated text is coherent (non-empty, not garbage)
3. **No regression on other tests:** Existing tests that use `TTNNLinearIColShardedWAllReduced` (if any) still pass
4. **Performance:** Decode latency per token is reasonable (< 2 seconds, given current unoptimized state)

## Additional Notes

- The `_to_replicated()` method in `TTNNBailingMoEAttention` (line 2292-2317) does a host round-trip to convert all-gathered tensors to replicated topology for paged attention. With the composite RS+AG approach, the output should already be in the correct topology since `all_gather_async` produces the same result as replicated data.
- The MoE module's CCL pattern is the gold standard for T3K multi-device communication in tt-symbiote. Following it exactly ensures compatibility.
- Long-term fix: File a bug with the TTNN CCL team to fix `ttnn.all_reduce` with 8 devices on T3K, so the simpler single-op approach can be used in the future.
