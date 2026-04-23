# Audit Methodology: Finding synchronize_device in Forward-Path Code

The goal of this audit is to find every call to `ttnn.synchronize_device` or `ttnn.synchronize_devices` (plural form) that can be reached during a forward pass under `TracedRun`. Any such call inside the trace bracket will raise a Metal Trace incompatibility error at capture time, while calls just outside but immediately adjacent to the bracket can prevent the bracket from being widened as the trace-enablement project progresses. Both categories must be identified and catalogued before remediation begins.

## Section 1: Search Command

Run the following command from the tt-symbiote repository root to locate every occurrence in model source files:

```bash
# From the tt-symbiote repository root:
grep -rn "synchronize_device\|synchronize_devices" \
    models/ \
    --include="*.py" \
    | grep -v "test" | grep -v "benchmark" | grep -v "profil"
```

This command excludes test files and benchmark scripts. Those files may contain `synchronize_device` calls that are intentional correctness checkpoints or throughput measurements; they are irrelevant to trace compatibility and would only add noise to the results.

> **Note:** If the `models/` directory is named differently in your checkout (e.g., `tt_symbiote/` at the repo root), adjust the path argument accordingly. Run `ls` at the repo root to confirm the top-level layout before executing the search.

Save the full output to a file for analysis:

```bash
grep -rn "synchronize_device\|synchronize_devices" \
    models/ \
    --include="*.py" \
    | grep -v "test" | grep -v "benchmark" | grep -v "profil" \
    > /tmp/sync_device_hits.txt

wc -l /tmp/sync_device_hits.txt   # quick count of hits
cat /tmp/sync_device_hits.txt
```

## Section 2: Distinguishing Forward-Path from Non-Forward-Path Calls

Not every call found by the search command is trace-blocking. A call is a **forward-path call** if it appears in a method that is reachable from `TTNNModule.forward` or `TracedRun.__call__` during normal inference.

**Non-forward-path calls — safe to ignore for this analysis:**

- Inside `__init__` or `__post_init__`: these run once at construction time, not during inference.
- Inside `move_weights_to_device_impl` or any method called only during model initialization: these run before trace capture begins.
- Inside `warmup`, `_warmup`, or `warm_up` methods: the warm-up compile run precedes `begin_trace_capture` and is outside the capture bracket.
- Inside `compile_run` if it is a separate code path from `forward`: same reasoning as warm-up.
- Inside standalone test functions or `if __name__ == "__main__"` blocks: outside the module forward path entirely.

**Forward-path calls that must be investigated:**

- Inside `forward`, `_forward`, or `__call__` methods of any `TTNNModule` subclass.
- Inside helper methods called from `forward` (e.g., `_maybe_all_gather`, `_apply_rotary_embedding`, `_qkv_projection`).
- Inside `TracedRun._capture_trace` or `TracedRun.execute` if those methods invoke module forward methods.

For each grep hit, open the file, locate the enclosing method, and trace the call chain upward to determine whether it is reachable from `forward` or `TracedRun.__call__`. A static call-graph tool (e.g., `pyan3` or manual inspection) can assist, but in practice the method names listed above are sufficient to classify the majority of hits.

## Section 3: Classifying Each Call

For each forward-path `synchronize_device` call found, work through the following four questions and record the answers in the summary table in `index.md`.

### Question 1: Is the enclosing module `@trace_enabled` or `@trace_disabled`?

- **`@trace_enabled` modules** are those intended to run inside the trace bracket. Their `synchronize_device` calls block trace capture directly — the capture will fail when the trace compiler encounters the call.
- **`@trace_disabled` modules** run outside the trace bracket. Their calls do not block the current trace but may block future trace expansion if the module is later promoted to `@trace_enabled`.

To check the decorator, open the class definition file and look for `@trace_enabled` or `@trace_disabled` on the class or its parent. If no decorator is present, check whether the class is instantiated inside a `TracedRun` context.

### Question 2: What precedes the call?

Identify the TTNN op immediately before `synchronize_device`. This context tells you what ordering concern (if any) the call was intended to address:

- A preceding `ttnn.all_gather` suggests the call was added to ensure all_gather completion before the next op reads the result.
- A preceding `ttnn.experimental.all_gather_async` (or similar async CCL variant) suggests the same intent but via an async dispatch path.
- A preceding compute op (e.g., `ttnn.matmul`, `ttnn.softmax`) with no CCL in the vicinity suggests the call may be a debugging artifact or an overly conservative ordering guard.

### Question 3: Is the call inside a multi-device conditional?

If the call is guarded by `if self.num_devices > 1:` or a similar condition, it only fires on multi-device deployments. Note this in the audit table, because:

- Single-device CI may not exercise the call at all, making it easy to miss.
- The fix must still be applied — the trace will be used on multi-device hardware.

### Question 4: What is the remedy type?

Based on the preceding op and the call's apparent purpose, classify the remedy using the following scheme:

- **Type A (delete only):** The preceding op is the synchronous variant of `ttnn.all_gather` AND the intended dispatch model is synchronous. CQ0 ordering suffices to guarantee the gather result is ready before the next consumer; the `synchronize_device` call can be deleted without replacement. If the preceding op is the synchronous `ttnn.all_gather` but the intended dispatch model is async, do NOT apply Type A — apply Type B2 instead (see below). If the preceding op is a compute op (e.g., `ttnn.matmul`, `ttnn.softmax`) with no CCL dependency, the purpose of the synchronize call is unclear — classify as Type C and investigate before deleting.
- **Type B (delete + async CCL):** The preceding op is or should be `ttnn.experimental.all_gather_async` or another async CCL dispatch. Deleting `synchronize_device` alone would break ordering. Two sub-cases apply:
  - **B1 — already async:** The preceding op is already `all_gather_async`. The all_gather call itself does not change; the fix is to delete `synchronize_device` and replace it with TT_CCL cycling semaphore management.
  - **B2 — should be async:** The preceding op is the synchronous `ttnn.all_gather` but the intent is async dispatch. The fix requires two steps: first replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, then delete `synchronize_device` and add TT_CCL cycling semaphore management. Deleting `synchronize_device` alone is insufficient in this sub-case — leaving the synchronous `ttnn.all_gather` in place alongside new semaphore management produces a broken async dispatch pattern.
  
  See Chapter 3 for the full semaphore pattern for both sub-cases.
- **Type C (investigate further):** The purpose is unclear from local context. The call requires source code archaeology — checking git blame, related issues, and any associated unit tests — before a remedy can be determined.

Record the remedy type for each call in the summary table in `index.md` and in the detailed results in `audit_results.md`.
