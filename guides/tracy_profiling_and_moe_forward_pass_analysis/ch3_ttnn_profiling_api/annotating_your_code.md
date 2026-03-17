# Annotating Your Code with Custom Tracy Zone Markers

## Overview

The device profiler CSV tells you how long each TTNN op ran on the hardware. Tracy zones tell you how long each *section of your Python code* ran on the host CPU — including the parts that are not TTNN ops, such as routing index construction, tensor slicing, and Python control flow. Together, these two data sources let you account for every millisecond in your forward pass.

This section covers how to import the Tracy Python bindings, create named zone markers with a context manager, use slash-separated naming for a readable timeline hierarchy, insert free-form text annotations, and understand how zones interact with TTNN's trace capture/replay mechanism.

---

## Tracy Python Bindings

Tracy is built into tt-metal as an optional compile-time feature, controlled by the `ENABLE_TRACY=ON` CMake flag (covered in Chapter 2). When built with Tracy support, the Python bindings are available either as part of the tt-metal Python package or as a standalone `tracy-client` pip package, depending on your installation.

### Importing the bindings

```python
# Standard import — works if tt-metal was built with ENABLE_TRACY=ON
# and the tt-metal Python package is on your PYTHONPATH
import tracy
```

If the import fails with `ModuleNotFoundError`, check that:
1. tt-metal was built with `-DENABLE_TRACY=ON`.
2. The tt-metal Python package directory is on `PYTHONPATH` or you have activated the correct virtual environment.

> **Warning:** Custom Tracy zones require `ENABLE_TRACY=ON` at build time. If tt-metal was built without Tracy support, the `tracy` module either will not be importable or will expose stub implementations where all zone context managers are no-ops. Zones created against a no-op stub produce no data and no error — they silently do nothing. Always verify that your Tracy capture is receiving data before concluding that a zone has zero overhead.

### Verifying Tracy is active

```python
import tracy

# A quick sanity check: if tracy.is_connected() returns True,
# a Tracy server is connected and zones will be recorded
if hasattr(tracy, "is_connected") and tracy.is_connected():
    print("Tracy server connected — zones will be captured")
else:
    print("No Tracy server connected — zones will be discarded")
```

---

## Custom Zone Context Manager

The primary API for annotating Python code is `tracy.zone()`, a context manager that creates a named CPU-side zone. The zone's start timestamp is recorded when the `with` block is entered and the end timestamp when it exits.

```python
import tracy
import ttnn

# Wrap an entire MoE forward pass section
with tracy.zone("MoE/forward"):
    # Routing: compute expert assignment scores
    with tracy.zone("MoE/dispatch"):
        router_logits = ttnn.linear(hidden_states, router_weight)
        routing_weights = ttnn.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = ttnn.topk(routing_weights, k=2)

    # Expert matmul: dispatch tokens to selected experts
    with tracy.zone("MoE/expert_matmul"):
        expert_output = ttnn.matmul(dispatched_tokens, expert_weight)

    # Combine: weighted sum of expert outputs
    with tracy.zone("MoE/combine"):
        combined = ttnn.matmul(top_k_weights, expert_output)
```

Each named zone appears in the Tracy GUI timeline as a colored bar on the CPU thread lane, nested according to the `with` block nesting in your source code.

---

## Zone Naming Conventions for Readability

Tracy supports arbitrary zone names, but following a slash-separated hierarchical convention produces a collapsible tree view in the Tracy GUI that maps naturally to MoE layer structure.

### Recommended hierarchy for MoE analysis

```
MoE/dispatch          → routing logic before any TTNN ops
MoE/expert_matmul     → per-expert weight matrix multiplications
MoE/combine           → weighted aggregation of expert outputs
MoE/all_to_all        → cross-device token dispatch (for T3K multi-chip)
MoE/all_reduce        → cross-device output aggregation
```

### Example with layer index

```python
for layer_idx in range(num_layers):
    layer_name = f"Layer{layer_idx}"

    with tracy.zone(f"{layer_name}/MoE/dispatch"):
        # routing for this layer
        pass

    with tracy.zone(f"{layer_name}/MoE/expert_matmul"):
        # expert computation for this layer
        pass
```

> **Tip:** Keep zone names short enough to read in the Tracy GUI timeline bars. Names longer than roughly 30 characters get truncated in the default zoom level. Use the message API (described below) for longer contextual information.

---

## Tracy Message Annotations

`tracy.message()` inserts a free-form text annotation at the current timestamp. Unlike zones, messages have no duration — they appear as vertical marker lines in the Tracy GUI. They are useful for tagging key parameters in the trace without creating a zone.

```python
import tracy

def run_moe_forward(hidden_states, seq_len, num_experts):
    # Tag this forward pass with its key parameters
    tracy.message(f"seq_len={seq_len} num_experts={num_experts}")

    with tracy.zone("MoE/dispatch"):
        # routing logic
        pass
```

Common uses for message annotations in MoE analysis:

```python
# Tag a cache miss (first call, compilation overhead expected)
tracy.message("program_cache_miss: first call for shape")

# Tag the start of measurement iterations after warm-up
tracy.message("warm_up_complete: measurement iterations begin")

# Tag a device synchronization point
tracy.message("ttnn.synchronize_device called")
ttnn.synchronize_device(device)
```

---

## Interaction with `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`

TTNN's trace capture mechanism records a sequence of op dispatches that can be replayed without re-running the Python interpreter for each iteration. This is distinct from Tracy's zone recording.

The key rule is: **annotate the `ttnn.execute_cached_trace()` call site, not the capture site**.

### Why this matters

Tracy zones recorded during the capture phase (`ttnn.begin_trace_capture` to `ttnn.end_trace_capture`) *will* appear in the Tracy timeline during capture. However, when the trace is replayed with `ttnn.execute_cached_trace()`, Python code inside the original capture block does not re-execute — only the recorded TTNN op dispatches are replayed. This means Tracy zones inside the capture block will appear once (during capture) but will not appear on subsequent replays.

```python
import tracy
import ttnn

# --- CAPTURE PHASE ---
# Zones inside capture run once during capture only.
# They will NOT appear on replay iterations.
ttnn.begin_trace_capture(device, trace_id=0)

with tracy.zone("MoE/capture_dispatch"):  # appears once, during capture
    router_logits = ttnn.linear(hidden_states, router_weight)
    top_k_weights, top_k_indices = ttnn.topk(routing_weights, k=2)

ttnn.end_trace_capture(device, trace_id=0)

# --- REPLAY PHASE ---
# Annotate the execute_cached_trace call site to measure per-replay timing.
# This zone appears on every replay iteration.
for i in range(num_iterations):
    tracy.message(f"trace_replay iteration={i}")
    with tracy.zone("MoE/trace_replay"):  # appears on every replay
        ttnn.execute_cached_trace(device, trace_id=0)
```

> **Tip:** The Tracy zone around `ttnn.execute_cached_trace()` includes host-to-device dispatch latency for the entire replayed trace. To isolate the device-side execution time of the replayed trace, cross-reference this zone with the device profiler CSV rows produced during the same replay iteration.

---

## Summary of Tracy Python API

| API | What it does |
|---|---|
| `import tracy` | Import the Tracy Python bindings |
| `tracy.zone("Name")` | Context manager — creates a named CPU zone for the duration of the `with` block |
| `tracy.message("text")` | Inserts a timestamped text annotation at the current moment (no duration) |
| `tracy.is_connected()` | Returns `True` if a Tracy server is connected and recording |

---

## Next Steps

Continue to [`reading_op_timing_output.md`](./reading_op_timing_output.md) to learn how to read and interpret the profiler CSV output, identify MoE ops in the timeline, sum kernel durations to reconstruct total forward pass time, and diagnose the gaps between hardware execution time and wallclock measurement.
