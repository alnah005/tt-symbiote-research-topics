# Verifying Correctness of the Trace Implementation

After implementing the changes described in `code_changes_required.md`, you need to confirm that the trace is behaving correctly on both the host and the device. This is not trivial: the two primary failure modes — deadlock and silent numerical corruption — present very differently, and the silent corruption mode produces wrong answers without any error signal. This file describes a test strategy that catches both failure modes, explains what to look for at each stage, and provides a checklist of mistakes that are easy to make and hard to notice.

---

## Numerical Comparison Test

The most reliable end-to-end test is to run the decode loop twice under identical conditions — once with tracing disabled and once with tracing enabled — and compare the output logits or token ids at every step.

### Setup

```python
# Reference run: tracing off, same model weights, same input tokens.
reference_outputs = []
for step in range(N_STEPS):
    out = model.decode_step(tokens, use_trace=False)
    reference_outputs.append(out.cpu())

# Traced run: tracing on, same model weights, same input tokens.
traced_outputs = []
for step in range(N_STEPS):
    out = model.decode_step(tokens, use_trace=True)
    traced_outputs.append(out.cpu())

# Compare.
for step, (ref, traced) in enumerate(zip(reference_outputs, traced_outputs)):
    if not torch.allclose(ref, traced, atol=1e-4):
        print(f"Step {step}: MISMATCH — max delta = {(ref - traced).abs().max()}")
```

### What to look for

- If outputs match for step 0 but diverge at step 1 or later, the most likely cause is a missing device semaphore reset between replays. Step 0 starts with freshly allocated (zero) semaphores, so it passes; step 1 sees residual non-zero values from step 0's kernels.
- If outputs diverge at every step including step 0, the capture itself is using wrong handles or the device was not reset before capture.
- If outputs match across all steps, the numerical correctness condition is satisfied. This does not by itself prove that host index state is being managed correctly (a host-index bug may be masked if no non-traced CCL calls run between replays), so also run the host-counter verification below.

> **Note:** Run at least 5–10 decode steps before declaring success. Semaphore state bugs often depend on the timing of kernel completion relative to the next dispatch cycle, and some bugs are intermittent under light load but consistent under sustained load.

---

## Detecting the Deadlock Case

A missing reset of a barrier semaphore or an AG semaphore that a kernel is waiting to reach a specific value will cause `execute_trace` with `blocking=True` to stall indefinitely. The process will appear hung.

### Reproducing it intentionally

To confirm that your reset code is actually doing something, temporarily comment out one of the `reset_global_semaphore_value` calls and run a two-step decode loop with a timeout:

```python
import signal

class TraceTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TraceTimeout("execute_trace timed out — likely deadlock")

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(30)  # 30-second timeout
try:
    ttnn.execute_trace(mesh_device, cq_id=0, tid=trace_id, blocking=True)
    signal.alarm(0)
except TraceTimeout:
    print("Deadlock confirmed: semaphore was not reset")
```

### What to look for

- A hang on step 2 (not step 1) when you remove the reset is the expected behavior for a barrier semaphore whose first replay leaves it at 1. The kernel's reader waits for the value to return to 0; since no host reset was enqueued, it waits forever.
- A hang on step 1 indicates the device was not properly reset before capture either, and the capture-time run itself left a non-zero value.
- A hang that resolves after a few seconds and then proceeds normally is not a deadlock — it is normal kernel execution latency. True deadlocks do not resolve without a process kill.

> **Warning:** Do not use `blocking=False` to work around a suspected deadlock. It will cause `execute_trace` to return immediately, but the stalled kernel remains on device and will interfere with all subsequent operations, potentially including the next `blocking=True` call, which will then hang at the synchronization point.

---

## Detecting Silent Corruption

Silent corruption (the "skip-through" failure) occurs when a semaphore that a kernel is waiting on to reach zero is instead already at zero at the start of replay because the expected self-reset from the *previous* replay was not yet written when the host enqueued the next replay's resets. The kernel treats the stale zero as a signal to proceed immediately, skipping a synchronization step and producing wrong output without any error.

### Why it is hard to detect

There is no exception, no NaN, and no obviously wrong value in isolation. The output tensors have valid floating-point contents. The corruption is only visible when you compare against a known-good reference.

### Pattern recognition

Silent corruption from skip-through tends to exhibit these characteristics:

- The corrupted tensor is not all-zeros and not all-NaN. It has plausible-looking values that are numerically close to but not equal to the reference.
- The corruption magnitude does not grow monotonically across steps. Because the skip-through depends on kernel timing, some steps may be correct and others incorrect under light load, but most or all steps are incorrect under sustained load.
- The corruption is reproducible when the system is under consistent load (e.g., running a long decode loop without interruption) and may be intermittent when the system is lightly loaded (because idle-system kernel completion is faster, and the self-reset may complete before the next replay begins).

### Confirming skip-through vs. other corruption

If you see numerical mismatches and want to distinguish skip-through from, say, a wrong handle being reset or a wrong `semaphore_index`:

- Add a `ttnn.synchronize_device(mesh_device)` call after `execute_trace` and before the next replay's reset. This forces all kernels from the current replay to complete before the host enqueues the next reset. If the mismatches disappear after adding the synchronize call, the bug is skip-through due to `blocking=False` timing.
- If mismatches persist after adding the synchronize call, the bug is in the reset logic itself (wrong handle, wrong index, wrong `si`).

---

## Verifying Host-Counter Reset

Even when the numerical outputs look correct, you should verify that the host-side index fields in `TT_CCL` are being managed as intended. A host-index bug may be latent — invisible when no non-traced CCL calls run between replays but activated when the model is extended or refactored.

### Logging the index state

Add print statements (or log calls) at three points:

```python
def _log_ccl_indices(tt_ccl, label):
    print(f"[{label}] ag_idx={list(tt_ccl.ag_semaphores_idx)}"
          f"  rs_idx={list(tt_ccl.rs_semaphores_idx)}"
          f"  barrier_idx={list(tt_ccl.barrier_semaphore_idx)}")

# Point 1: immediately after snapshotting, before begin_trace_capture.
_log_ccl_indices(tt_ccl, "pre-capture snapshot")

# Point 2: immediately before the restore call in the replay loop.
_log_ccl_indices(tt_ccl, f"step {step} before restore")

# Point 3: immediately after the restore call, before execute_trace.
_log_ccl_indices(tt_ccl, f"step {step} after restore")
```

### Expected output

```
[pre-capture snapshot]   ag_idx=[0, 0, 0]  rs_idx=[0, 0, 0]  barrier_idx=[0, 0, 0]
[step 0 before restore]  ag_idx=[0, 0, 1]  rs_idx=[0, 0, 1]  barrier_idx=[0, 0, 1]
[step 0 after restore]   ag_idx=[0, 0, 0]  rs_idx=[0, 0, 0]  barrier_idx=[0, 0, 0]
[step 1 before restore]  ag_idx=[0, 0, 1]  rs_idx=[0, 0, 1]  barrier_idx=[0, 0, 1]
[step 1 after restore]   ag_idx=[0, 0, 0]  rs_idx=[0, 0, 0]  barrier_idx=[0, 0, 0]
```

The "before restore" line shows the indices as they were left by the previous `execute_trace` call (advanced past the capture-time slot because the trace's `get_and_cycle_*` calls ran again at capture time; Python does not re-run during replay, so the values only advance at capture time and then stay there until the next restore). The "after restore" line must always match the "pre-capture snapshot" line exactly.

> **Note:** The exact values in the snapshot depend on how many cycles the compile pass performed. If the compile pass advanced `si=2` from 0 to 1, the snapshot will show `ag_idx=[0, 0, 1]` and the "after restore" must also show `[0, 0, 1]`. The important invariant is that "after restore" equals "pre-capture snapshot" for every step.

---

## Verifying Device-Side Reset

Confirming that the device-side semaphore counter is actually 0 at the L1 address before `execute_trace` begins is more involved than the host-side check, but it can be approximated.

### Approach 1: Read-back before execute_trace

`ttnn.get_global_semaphore_address(handle)` returns the L1 address of the semaphore. After enqueuing the reset and before calling `execute_trace`, you can enqueue a read-back using a small on-device program that reads from that address and writes to a host-readable DRAM buffer. If the value returned is non-zero, the reset either did not reach the device or reset the wrong address.

This approach requires writing a small TT kernel and is mainly useful for debugging a specific suspected bug. It is not recommended as a routine production check.

### Approach 2: ttnn watcher

The ttnn watcher (`TTNN_WATCHER=1` or the equivalent environment flag) logs all kernel activity and semaphore operations to a trace file. After a failing run, inspect the watcher log for the semaphore address in question:

- Look for the `reset_global_semaphore_value` write event and confirm its address matches the handle's reported address.
- Look for the first kernel access to that address after the reset and check that it reads 0.

### Approach 3: Use `blocking=True` as a proxy

If you use `blocking=True` in `execute_trace` and the call returns without hanging, the kernels that were waiting on their semaphores did eventually proceed — which means the semaphores reached their expected values at some point during the replay. This does not confirm that the value was 0 at the *start* of the replay, but it rules out a full deadlock. Combine this with the numerical comparison test to distinguish "semaphore reached correct value eventually" from "semaphore was already correct at the start."

---

## Common Mistake Checklist

The following mistakes are easy to make and, in most cases, produce no hard error — only wrong outputs or intermittent hangs.

- **Forgetting to reset RS handles when `use_composite=True`.** The `tt_all_reduce` composite path consumes RS semaphore handles in addition to AG and barrier handles. Resetting only AG and barrier leaves the RS semaphore slot with a non-zero value from the previous replay. The RS op's reader kernel sees that value and either stalls or skips through.

- **Forgetting the second barrier slot for `use_composite=True`.** The composite path calls `get_and_cycle_barrier_semaphore_handles` twice — once for the RS op and once for the AG op. Both barrier slots are baked into the trace. If you only reset `captured_barrier_idx[si]` and not `(captured_barrier_idx[si] + 1) % 2`, the AG op's barrier will not be reset.

- **Resetting handles at the wrong `semaphore_index` due to `cluster_axis=0` mapping to 2 in older code.** In `models/tt_transformers/tt/ccl.py`, the formula `2 if not cluster_axis else cluster_axis` maps both `None` and `0` to index 2. If you assume `cluster_axis=0` means `si=0` and reset `tt_ccl.ag_semaphore_handles[0][...]`, you reset the wrong slot. The actual trace uses handles at index 2, which remain non-zero.

- **Restoring host indices only once (after capture) instead of before every replay.** The restore must happen before every `execute_trace` call. If you restore once after capture and then call `execute_trace` N times without restoring each time, the first replay is correct, but any Python code between replays that calls `get_and_cycle_*` will advance the indices, and subsequent replays will see the advanced (incorrect) values if you rely on those values for anything.

- **Resetting device semaphores but not restoring host indices, or vice versa.** Both operations are required. Resetting device semaphores without restoring host indices means the trace runs correctly (correct semaphores), but any non-traced CCL call between replays cycles the wrong handle. Restoring host indices without resetting device semaphores means the Python state is correct but the on-device counters may be non-zero, causing deadlock or skip-through.

- **Enqueueing the device semaphore reset after `execute_trace` instead of before.** The reset must be in the CQ FIFO before the trace commands. If you enqueue the reset after `execute_trace`, it will execute after the trace kernels, which means the kernels see the pre-reset (stale) value. The CQ FIFO ordering guarantee only applies within a single enqueue sequence: commands enqueued before `execute_trace` will complete before the trace begins; commands enqueued after will execute after the trace completes.

- **Using a shared list reference instead of a copy for the snapshot.** Assigning `captured_ag_idx = tt_ccl.ag_semaphores_idx` (without `list(...)`) stores a reference to the same list object. Later mutations to `tt_ccl.ag_semaphores_idx` (whether from `get_and_cycle_*` or from the restore step itself) will also modify `captured_ag_idx`, silently corrupting the snapshot. Always use `list(...)` or `.copy()`.

- **Running the numerical comparison test with too few steps.** Skip-through corruption from `blocking=False` timing is sometimes absent on step 1 if the kernel completes fast enough. Run at least 5 steps, and prefer 10 or more under a realistic workload (e.g., with the model actually processing tokens rather than dummy inputs) to ensure you are testing under realistic kernel timing.
