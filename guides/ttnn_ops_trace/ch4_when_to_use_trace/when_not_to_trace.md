# When Not to Trace

Not every loop benefits from trace, and some loops cannot be traced at all. This file covers the disqualifying conditions that prevent a loop from being traced correctly, the ongoing maintenance burden that trace imposes on code that does qualify, the debugging difficulty that trace introduces, and the standard architectural pattern for handling mixed workloads — a traced inner core surrounded by an untraced outer wrapper that handles the parts trace cannot accommodate.

---

## Disqualifying Condition 1: Dynamic Shapes Per Call

The most common reason a loop cannot be traced is that tensor shapes vary between iterations.

Trace encodes the specific DRAM and L1 buffer addresses of every tensor involved in the captured sequence. Those addresses are determined by the allocator at the time the buffers are created, based on tensor size. A tensor with shape `[1, S, 512]` occupies a different number of bytes for different values of S, so the allocator places it at a different address for each distinct S. When the trace replays, it uses the addresses from the capture run. If the current iteration's tensors were allocated at different addresses — because S is different — the kernels will read from and write to the wrong locations. The result is incorrect output, and the runtime may not detect the mismatch.

This is not a limitation that can be worked around at the call site. The constraint lives in the trace buffer itself; the replay mechanism has no way to adjust addresses at replay time. If shapes vary, the trace is invalid.

**The prefill pass is the canonical example.** As detailed in `latency_sensitive_workloads.md`, prompt sequence length varies per request, making the prefill path unfit for trace in the general case. For the decode case: any model that varies the number of tokens processed per step — speculative decoding that accepts a variable number of draft tokens, variable-batch-size decode, or dynamic sequence packing — has the same disqualification.

**Diagnosis.** Walk through every tensor in the candidate loop body and ask: could this tensor's shape be different on the next call? If yes, the tensor's buffer address could differ, and the trace is invalid for that case. Shapes derived purely from model configuration constants (hidden dimension, number of heads, number of layers) are safe. Shapes derived from any runtime input — sequence length, batch size, number of accepted draft tokens — are potentially disqualifying.

---

## Disqualifying Condition 2: Ops That Read Back to Host Mid-Loop

Any operation that causes a device value to be transferred to host memory during the dispatch sequence is a synchronization point (as defined in Chapter 2's `async_execution_model.md`). When a host readback happens mid-loop, it does more than pause dispatch: it gates what ops come next on the outcome of the device computation that preceded the readback. The trace cannot capture this decision — it can only capture the specific command sequence that was dispatched given the capture run's specific device outputs.

The three patterns that introduce host readbacks are:

**Explicit `ttnn.to_torch()` inside the capture region.** A `ttnn.to_torch()` call inside `begin_trace_capture` / `end_trace_capture` executes during the capture run and returns a value. The trace records whatever ops were dispatched as a result of that value. On replay, those ops execute unconditionally regardless of what `ttnn.to_torch()` would return for the current inputs. If the readback was used to make a decision — "if this value exceeds a threshold, take branch A; otherwise branch B" — the trace will always replay branch A (or branch B, whichever was taken at capture time), even when the current step's value would select the other branch.

> **Warning:** The runtime does not raise an error when `ttnn.to_torch()` is called inside a capture region. The capture completes successfully, and the trace appears valid. The failure mode is silent numerical incorrectness: the replay always takes the capture-time branch, which may produce wrong results for inputs that would have taken the other branch.

**Python control flow whose condition depends on a device tensor value.** This is the same problem at the Python level rather than the TTNN API level. Any `if`, `while`, `for`, or comparison that reads `.item()`, `.tolist()`, or indexes into a tensor that is on the device forces a readback. After the readback, the dispatch sequence diverges based on the value. The trace encodes only the capture-time branch.

```python
# DISQUALIFIED: Python branch inside the capture region that reads device values.
ttnn.begin_trace_capture(device, cq_id=0)
logits = ttnn.matmul(hidden, self.lm_head)
logits_host = ttnn.to_torch(logits)          # readback during capture
if logits_host.max() > threshold:            # decision baked into trace
    output = ttnn.softmax(logits, dim=-1)    # only this branch is recorded
else:
    output = ttnn.relu(logits)               # this branch never replayed
trace_id = ttnn.end_trace_capture(device, cq_id=0)
# Replay always executes softmax, regardless of current logits.max().
```

**Device-to-host transfers used as loop bounds.** A `while not ttnn.to_torch(done_flag).item()` pattern inside a capture region captures one iteration (the one that ran during capture) and produces a trace that replays that fixed iteration regardless of the actual termination condition.

**A related pattern: Python control flow on device values.** Not all Python branches are disqualifying — only those whose condition depends on a runtime device tensor value. An `if config.use_gated_ffn` branch whose condition is a Python configuration flag resolved at import time is *structure-invariant*: the trace captures whichever branch ran, and that branch will always be taken because the config flag does not change. A `if ttnn.to_torch(gate_value).item() > 0.5` branch is *structure-variant*: the trace captures whichever branch ran during capture, but a different branch might be correct for the current input, producing silent numerical incorrectness on those inputs.

---

## Disqualifying Condition 3: Ops That Self-Configure Based on Prior Results

Some ops inspect their own output — or the output of a preceding op — to determine how they will execute on the next call. Adaptive softmax implementations that adjust their computation based on the distribution of values seen in the previous step are a canonical example; ops that examine their own output to set the next call's arguments (tile size, sparsity mask, core grid assignment) fall into the same category. The dispatch structure for such an op is determined by a runtime device value, not by the static model configuration. This is a specific form of data-dependent dispatch (Condition 2), but it differs in that the data dependency reaches across call boundaries rather than being contained within a single step's Python control flow. The trace captures the self-configuration state at capture time and replays it unconditionally, even when subsequent steps have produced values that would have driven a different configuration. The result is that the traced op silently uses the capture-time configuration for all replays, producing incorrect output whenever the correct configuration diverges from it. There is no workaround that allows such an op to remain inside the trace boundary; it must be isolated in the untraced outer wrapper.

---

## The Maintenance Cost of Trace

Even when a loop satisfies all traceability conditions, the trace imposes ongoing maintenance costs that are absent from a straightforward dispatch loop.

### Captured buffers are pinned

Every tensor buffer that was allocated at the time of capture — inputs, outputs, KV-cache buffers, all intermediate tensors that the captured ops read or write — must remain allocated and at the same DRAM address for the lifetime of the trace. The trace encodes those addresses. If any buffer is deallocated and the allocator reuses that address for a different tensor, a subsequent replay will silently corrupt the new tensor by overwriting it with the captured op's output.

In practice, this means:

- You cannot free the input or output tensors between replays. They must persist as long as the trace is in use.
- The device memory region occupied by all traced buffers is effectively locked until you call `ttnn.release_trace(device, trace_id)`. This reduces the available allocatable memory on the device for other allocations made after capture.
- If you need to resize any of the traced tensors — for example, because you want to support a larger batch size — you must release the existing trace, reallocate the tensors at the new size, and re-capture.

### Shape changes require re-capture

Any model update, configuration change, or deployment parameter change that affects the shape of any tensor in the traced region requires a full re-capture. This includes:

- Changing the batch size.
- Changing the maximum context length (which affects KV-cache buffer dimensions).
- Changing the model precision (which changes tensor element sizes and thus buffer sizes and addresses).
- Adding or removing a model component that contributes ops to the traced region.
- Upgrading the model to a new version with different hidden dimensions or layer counts.

Re-capture is not free: it requires running the full captured sequence once (live dispatch, which is slower), finalizing the buffer, and verifying that the new trace produces correct outputs. For models that are updated frequently or served in multi-tenant environments where configuration varies across requests, this re-capture overhead may be significant.

### Trace interacts with memory allocators in non-obvious ways

Because captured buffers are pinned at fixed addresses, the allocator's behavior for subsequent allocations changes. Allocations made before capture are pinned; allocations made after capture for non-traced purposes must avoid those addresses. On devices with limited DRAM, this can cause fragmentation or allocation failures for new tensors that need large contiguous regions. The trace lifetime (from `begin_trace_capture` to `release_trace`) is the window during which this pinning is in effect.

> **Note:** `ttnn.release_trace(device, trace_id)` releases the trace buffer from device DRAM and unpins the captured buffer addresses. After this call, the allocator can reuse those regions. If you call `ttnn.execute_trace` after `release_trace`, the behavior is undefined.

---

## Debugging Difficulty

Tracing introduces a layer of indirection between what your Python code says and what the device executes, which significantly complicates debugging.

### Replay errors are silent

The most dangerous failure mode is silent numerical incorrectness. If the trace was captured with an untraceable pattern (a mid-capture readback, a data-dependent branch, or a shape that changed) the runtime may not raise an error. The trace replays successfully from the device's perspective — it executes the recorded command buffer exactly as stored — but the commands are wrong for the current inputs. Output tensors contain incorrect values, and there is no error message pointing to the trace as the cause.

The recommended safeguard, shown in Chapter 3's `trace_constraints.md` (Step 4), is to run both the trace replay and a live dispatch run on identical inputs and compare outputs with `ttnn.allclose` before committing to the trace loop. This validation step catches silent errors during development but requires you to maintain a live dispatch code path in parallel with the trace code path during testing.

### Per-op debugging is not possible during replay

In live dispatch mode, you can insert `ttnn.to_torch()` calls after any op to inspect intermediate values. In trace replay mode, the trace executes as an atomic command sequence from the device's perspective — there is no mechanism to pause replay mid-sequence and read an intermediate tensor. Debugging incorrect trace outputs requires re-running the sequence in live dispatch mode (without trace) and bisecting to find which op produces the divergence.

This means maintaining two execution paths: one traced (for production performance) and one untraced (for debugging). The operational cost of this dual-path maintenance grows with model complexity.

### Trace captures silent assumptions about device state

The trace captures the command sequence but assumes that device state at replay time matches device state at capture time in all relevant ways — the same weights at the same addresses, the same KV-cache memory layout, the same core firmware configuration. If any of these change between capture and replay in a way that does not surface as a shape change (for example, weight values are modified in-place while the buffer address is unchanged), the trace will silently replay the old command structure against the new state. The result may be numerically wrong without any indication from the trace machinery.

---

## The Traced Inner Loop / Untraced Outer Wrapper Pattern

The standard architectural response to a model that contains a mix of traceable and untraceable operations is to restructure the code so that the traceable operations form an isolated inner function and the untraceable operations remain in an outer wrapper that is never captured.

This pattern is introduced in detail in Chapter 3's `trace_constraints.md` (Steps 1–3). The summary for decision-making purposes is:

**Identify the boundary.** Walk through the candidate decode loop. Mark every construct that falls into one of the four disqualifying conditions above. The traced region is everything between the first and last traceable op, subject to the constraint that no disqualifying construct appears within that region.

**Move disqualifying constructs outside the boundary.** The outer wrapper handles: token sampling (which typically requires a host readback on the argmax output), EOS detection (host-side check on the sampled token), KV-cache position tracking (if position is a Python variable that changes each step), attention mask updates (if generated dynamically in Python), and any logging or profiling instrumentation that reads device values.

**Pass per-step variables through pre-allocated device tensors.** Any value that changes each step but has a fixed size — the current token embedding, the step position index, a scaling factor — should be represented as a pre-allocated device tensor whose content is updated in-place before each replay. The buffer address stays constant (satisfying the trace's address fixity requirement); only the value changes. This is the mechanism described in Chapter 3's `trace_constraints.md` (Step 3).

The result is a structure like this:

```python
# Outer wrapper: not traced. Handles untraceable operations.
def decode_step(self, token_id: int, step: int, kv_cache, trace_id: int) -> int:
    # Write the current token embedding into the pre-allocated device tensor.
    # In-place update: same buffer address as capture time.
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            self.embedding_table[token_id].unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        ),
        self.input_tensor,   # pre-allocated at capture time
    )

    # Replay the traced inner loop. Non-blocking: returns before device finishes.
    # Output appears in self.output_tensor at the same address used during capture.
    ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=False)

    # Synchronize to read the output. This is the one mandatory sync per step.
    # If streaming output is required, use a ttnn event barrier instead of a
    # full synchronize_device() to avoid draining the entire CQ.
    ttnn.synchronize_device(self.device)

    # Token sampling: host-side argmax on the output logits.
    # This is a host readback, which is why it lives in the outer wrapper.
    logits_host = ttnn.to_torch(self.output_tensor)
    next_token = int(logits_host[0, -1, :].argmax(-1).item())

    # EOS detection: also host-side, also in the outer wrapper.
    if next_token == self.eos_token_id:
        return -1   # signal caller to stop

    return next_token
```

```python
# Inner loop: captured as a trace. Contains only fixed-shape, traceable ops.
# This function is called once during capture; execute_trace replays it thereafter.
def decode_core(self, input_tensor: ttnn.Tensor, kv_cache) -> ttnn.Tensor:
    hidden = input_tensor
    for layer in self.layers:
        # Fixed-shape attention with pre-allocated KV-cache buffers.
        residual = hidden
        hidden = layer.self_attn(hidden, kv_cache[layer.idx])
        hidden = residual + hidden
        # Fixed-shape FFN.
        residual = hidden
        hidden = layer.ffn(hidden)
        # Fixed-shape layernorm and residual.
        hidden = layer.norm(residual + hidden)
    logits = ttnn.matmul(hidden, self.lm_head_weight)
    return logits
    # No ttnn.to_torch() here. No Python branches on device values.
    # No ops whose dispatch structure depends on runtime device outputs.
```

For the complete capture-and-replay setup pattern (pre-allocation, `begin_trace_capture`, decode loop, `release_trace`), see [`trace_constraints.md`](../ch3_trace_capture/trace_constraints.md) Steps 1–4.

---

## Summary: When Not to Trace

| Condition | Guidance |
|---|---|
| Loop executes fewer than ~10 times | Capture cost not amortized; use live dispatch |
| Tensor shapes vary between iterations | Cannot trace; use async dispatch or restructure |
| `ttnn.to_torch()` appears inside the loop body | Move readback to outer wrapper; re-evaluate |
| Python `if` / `while` on device tensor value inside loop | Move to outer wrapper; re-evaluate |
| Op self-configures based on a prior result (adaptive compute) | Cannot trace that op; try to isolate it outside trace boundary |
| Batch size is large and device utilization is already high | Trace provides marginal improvement; weigh against maintenance cost |
| Workload is prefill (variable sequence length) | Almost never worth tracing; use async dispatch |
| Model is under active development (frequent shape changes) | Re-capture cost is high; defer tracing until model is stable |
| Need to debug intermediate tensor values frequently | Trace prevents per-op inspection; maintain a live dispatch mode for debugging |

---

**Next:** [Chapter 5 — Estimating Improvement from Trace](../ch5_estimating_improvement/index.md)
