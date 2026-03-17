# Gap to Action Mapping

This file provides a systematic mapping from each gap pattern identified in Chapter 5 to a
recommended TTNN optimization action. For each pattern the mapping covers: the root cause
summary, the primary optimization lever, conditions under which it applies, and how the
Chapter 6 scaling results should inform prioritization.

The four patterns are:

| Pattern | Gap location in trace | Scaling behavior |
|---|---|---|
| **A** | Between `MoE/dispatch/topk` and `MoE/dispatch/gather` — CPU index construction | Roughly linear with `seq_len` (matches or exceeds CCL estimate) |
| **B** | Between `MoE/expert_matmul` end and `MoE/combine` start — sync barrier | Constant (k ≈ 0), present every iteration |
| **C** | Between `MoE/dispatch/gather` and `MoE/expert_matmul` — CCL all-to-all | Linear with `seq_len` (k ≈ 1.0, magnitude matches CCL bandwidth estimate) |
| **D** | At the very start of `MoE/forward`, before any child zones — program cache miss | Constant (k ≈ 0), first call only or after shape change |

---

## Pattern A: CPU Index Construction After `ttnn.topk`

### Root Cause

Pattern A is defined in `ch5_identifying_gap/common_gap_patterns.md`. When you observe this
pattern (host round-trip between `MoE/dispatch/topk` and `MoE/dispatch/gather`), the
recommended actions are below.

### Optimization Action

**Tensor-ize the index construction and keep it on device.**

Replace the Python loop with a sequence of TTNN ops that computes the gather permutation
without leaving the device:

```python
# Before (Pattern A — causes host round-trip)
expert_indices_cpu = ttnn.to_torch(expert_indices)   # D→H copy
token_assignments = []
for token_idx in range(seq_len):
    for k in range(top_k):
        expert_id = expert_indices_cpu[token_idx, k].item()
        token_assignments.append((token_idx, expert_id))
token_assignments.sort(key=lambda x: x[1])
gather_indices = torch.tensor([ta[0] for ta in token_assignments])
gather_indices_device = ttnn.from_torch(gather_indices, device=device)  # H→D copy

# After (tensor-native — no host round-trip)
# expert_indices: [seq_len, top_k] on device, dtype=int32
flat_indices = ttnn.reshape(expert_indices, [seq_len * top_k])
sorted_order = ttnn.argsort(flat_indices)  # stable sort ascending by expert id
# sorted_order is the gather permutation, entirely on device
```

**Move any remaining host-side bookkeeping inside `ttnn.begin_trace_capture`.**

If the model is running in decode mode (fixed `seq_len = 1`), a trace region can eliminate
per-op host dispatch overhead for the entire dispatch-and-gather sequence:

```python
# Capture the dispatch sub-graph into a reusable trace
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
flat_indices = ttnn.reshape(expert_indices, [seq_len * top_k])
sorted_order = ttnn.argsort(flat_indices)
gathered_tokens = ttnn.gather(hidden_states, sorted_order)
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# On every decode step, execute the trace instead of re-dispatching ops
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
```

> **Warning:** `ttnn.begin_trace_capture` records ops with fixed tensor addresses. Tensors
> used inside a trace must not be reallocated or reshaped between `begin_trace_capture` and
> `execute_trace`. For prefill (variable `seq_len`) this approach is not applicable — use
> the tensor-native rewrite instead.

### Conditions

- Applies when the model codebase contains `ttnn.to_torch` / Python loop / `ttnn.from_torch`
  between `topk` and `gather` in the dispatch phase.
- Trace capture applies only in decode mode where input shapes are fixed across steps.

### Expected Reduction

Eliminates the full gap attributed to Pattern A. For a seq_len=1024 run where the gap
scales at ~2µs/token above the CCL estimate, the reduction at seq_len=1024 is approximately
`(measured_slope - ccl_slope) × 1024` ms.

---

## Pattern B: Synchronization Barrier Between Expert Matmul and Combine

### Root Cause

Pattern B is defined in `ch5_identifying_gap/common_gap_patterns.md`. When you observe this
pattern (constant-overhead gap between `MoE/expert_matmul` and `MoE/combine`), the
recommended actions are below.

### Optimization Action

**Step 1: Verify the barrier is not required for correctness.**

Search the MoE forward pass source for any synchronization calls:

```python
import pathlib

for sf in pathlib.Path("models/").rglob("moe*.py"):
    for lineno, line in enumerate(sf.read_text().splitlines(), start=1):
        if "synchronize_device" in line or "wait_for_event" in line:
            print(f"{sf}:{lineno}: {line.strip()}")
```

Trace through the code to determine whether the combine phase ops (`ttnn.scatter`,
`ttnn.matmul` for the weighted sum) read from the output tensors of the expert matmul phase.
If so, the device command queue already serializes them — the explicit synchronization is
redundant.

**Step 2: Remove the barrier or replace it with a fine-grained event.**

If the barrier is confirmed redundant, remove it. If you are uncertain about correctness,
replace it with `ttnn.record_event` / `ttnn.wait_for_event`, which enforces device-side
ordering without blocking the host thread. For the implementation pattern, see
`optimization_action_reference.md` Lever 2.

> **Tip:** Use Option A (remove entirely) first. If the output PCC check passes, the barrier
> was redundant and the simpler solution is correct. Use Option B (record/wait event) only if
> correctness requires inter-phase synchronization that the single command queue does not
> naturally provide (e.g., ops dispatched across two command queues).

### Conditions

- Applies when `ttnn.synchronize_device` or `ttnn.wait_for_event` is present between the
  expert matmul phase and the combine phase.
- Also applies if an unannotated Python `time.sleep` or a blocking tensor read (e.g.,
  `ttnn.to_torch`) is present at this location.

### Expected Reduction

Eliminates the constant term from the linear regression (approximately **13–14ms** based on
the Chapter 6 worked example). This is the highest-priority action because it affects every
single inference step regardless of sequence length.

---

## Pattern C: CCL All-to-All Latency Between Dispatch and Expert Matmul

### Root Cause

Pattern C is defined in `ch5_identifying_gap/common_gap_patterns.md`. When you observe this
pattern (linear-scaling gap between `MoE/dispatch/gather` and `MoE/expert_matmul`), the
recommended actions are below.

### Optimization Action

**Primary lever: Evaluate expert parallelism degree (`ep_degree`).**

If the model currently distributes all 128 experts across all 8 chips (ep_degree=8), each
chip holds 16 experts. Reducing ep_degree to 4 means each chip holds 32 experts; the
all-to-all only crosses 4 chips and the message size per link is reduced:

```
# At ep_degree=8: all-to-all crosses 8 chips
bytes_total = seq_len * top_k * d_model * 2  # all dispatched tokens
# At ep_degree=4: all-to-all crosses 4 chips, each receiving 2x more tokens
# but traversal latency is lower because fewer hops are required
```

Before reducing `ep_degree`, verify that the remaining chips have sufficient L1 and DRAM
to hold 32 experts per chip (each expert has gate/up/down weight matrices of shape
`[d_model, d_ff]` = `[7168, 2048]` in bfloat16 = 84.0 MB per expert (3 × 7168 × 2048 × 2 bytes);
32 experts per chip = 32 × 84.0 MB = 2,688 MB ≈ 2.6 GB — feasible in DRAM on Wormhole B0).

**Secondary lever: Async CCL overlap.**

Use `ttnn.experimental.ccl.all_to_all_async` to pipeline the all-to-all for the current
layer with the expert matmul computation of the previous layer:

```python
# Layer N-1 expert matmul completes on device
# Layer N dispatch gather produces dispatched_tokens_n

# Launch CCL for layer N asynchronously — does not block host
ccl_handle = ttnn.experimental.ccl.all_to_all_async(
    dispatched_tokens_n,
    num_links=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Meanwhile, host dispatches layer N-1 combine ops (they can overlap with the CCL)
combined_n_minus_1 = ttnn.matmul(
    expert_outputs_n_minus_1, combine_weights, ...
)

# Wait for the CCL to complete before dispatching layer N expert matmul
redistributed_tokens_n = ccl_handle.get()
expert_outputs_n = ttnn.matmul(redistributed_tokens_n, expert_weights, ...)
```

> **Warning:** Async CCL overlap requires that the combine ops from layer N-1 do not read
> from the same buffers being written by the layer N all-to-all. Verify buffer aliasing before
> enabling this pattern. Incorrect overlap will produce silent data corruption that may not
> be caught without a PCC check.

> **Tip:** Annotate the CCL call with a Tracy zone (`MoE/dispatch/all_to_all`) before
> attempting async overlap. This lets you measure the baseline CCL duration and set a
> realistic target for how much of it can be hidden.

### Conditions

- Applies only on multi-chip configurations (T3K or larger). Single-chip runs do not execute
  CCL ops.
- Async overlap requires that two consecutive MoE layers both use the same CCL call pattern
  with non-aliased intermediate buffers.

### Expected Reduction

Async overlap can hide up to the full CCL latency (~2ms at seq_len=1024) if the combine
phase computation is at least as long as the CCL duration. Reducing `ep_degree` reduces the
CCL message size proportionally.

---

## Pattern D: Program Cache Miss at Start of `MoE/forward`

### Root Cause

Pattern D is defined in `ch5_identifying_gap/common_gap_patterns.md`. When you observe this
pattern (large one-time gap at the very start of `MoE/forward` before any child zones),
the recommended actions are below.

### Optimization Action

**Step 1: Ensure `ttnn.enable_program_cache()` is called before the first inference.**

```python
ttnn.enable_program_cache()  # call once at startup, before any inference

# Warm up: run 3 forward passes to pre-compile all programs
for _ in range(3):
    _ = moe_forward(warm_up_input, router_weights)
```

> **Warning:** `ttnn.enable_program_cache()` must be called before the very first TTNN op
> that uses the device, not just before the first MoE forward call. If any other model
> component runs before MoE and populates the cache without this call, those entries will
> not be reused.

**Step 2: Pad or canonicalize input shapes to a fixed set.**

If the model is called with varying `seq_len` values (e.g., prefill at many lengths, then
decode), pre-compile the program cache for each supported length. Pre-warm the cache for each
supported `seq_len` — see `optimization_action_reference.md` Lever 4 for the full warm-up
script.

**Step 3: Pad `expert_capacity` to a multiple of 32 to prevent tile-count discontinuities.**

When `seq_len` changes at runtime, `expert_capacity = seq_len × top_k / num_experts` changes
too. If the new capacity crosses a 32-token tile boundary, the expert matmul kernel must be
recompiled. Padding capacity to a fixed multiple of 32 (e.g., always round up to the nearest
64) keeps the compiled kernel valid:

```python
# Compute expert_capacity and pad to next multiple of 32
raw_capacity = (seq_len * top_k + num_experts - 1) // num_experts  # ceil division
expert_capacity = ((raw_capacity + 31) // 32) * 32  # round up to tile boundary
```

### Conditions

- Cold-cache recompilation is unavoidable on first run. It only becomes a performance problem
  if it recurs on hot-path calls.
- Tile-count discontinuities can cause recompilation mid-sweep; padding prevents this.

### Expected Reduction

Eliminates the 200–500ms first-call overhead and any per-step recompilation that occurs
due to shape changes. On steady-state decode (fixed `seq_len=1`), the steady-state cost
of Pattern D after warm-up is effectively zero.

---

## Prioritization Using Scaling Analysis

After running the Chapter 6 sweep and decomposing the gap into constant and linear terms,
rank optimization actions by total latency impact across a typical inference workload.

### Step 1: Estimate Total Latency Impact

For a workload consisting of one prefill at `seq_len=S` followed by `N` decode steps at
`seq_len=1`:

```
total_gap_ms = (constant_term_ms × (N + 1))
             + (linear_slope_ms_per_token × S)
             + (ccl_ms_per_token × 1 × N)    # decode CCL is trivial at seq_len=1
```

See the worked example in `ch6_sequence_length_scaling/interpreting_scaling_results.md` for
the derivation of the constant and linear terms. Substituting those values into the formula
above confirms that Pattern B (synchronization barrier) dominates by two orders of magnitude
for any workload with more than a handful of decode steps. Pattern C matters most at large
prefill `seq_len`.

### Step 2: Prioritized Backlog

| Priority | Pattern | Gap Component | Expected Reduction | Action |
|---|---|---|---|---|
| 1 (highest) | B | ~14ms constant per step | ~14ms/step × N steps | Remove `ttnn.synchronize_device`; validate PCC > 0.999 |
| 2 | C | ~2ms linear at seq_len=1024 | Up to 2ms at long prefill | `ttnn.experimental.ccl.all_to_all_async` overlap |
| 3 | A | ~0ms if not detected; varies | Proportional to excess linear slope | Tensor-ize index construction |
| 4 (lowest) | D | 200–500ms first call only | Eliminate first-call stall | `ttnn.enable_program_cache()` + warm-up; shape canonicalization |

> **Tip:** Always address Pattern B first. Even if the CCL and host-Python gaps are fully
> eliminated, a 14ms synchronization barrier repeated across 256 decode steps costs 3.6
> seconds of pure stall time. This is the highest-leverage single change in the MoE forward
> pass optimization plan.

---

## Next Steps

Proceed to [`writing_a_gap_analysis.md`](./writing_a_gap_analysis.md) to structure the
findings from Chapters 5, 6, and this mapping into a shareable document with the seven-section
template. Include the prioritized backlog table above as the final section of your gap
analysis.
