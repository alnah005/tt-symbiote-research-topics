# Q3 — Requirements and Known Limitations for Using `ttnn.all_reduce` Inside a Traced Region

This file answers the third research question: what concrete requirements must be met
and what limitations exist when `ttnn.all_reduce` is called from inside a
`ttnn.begin_trace_capture` / `ttnn.execute_trace` region, specifically as invoked by
`TTNNLinearIColShardedWAllReduced`. The requirements below are derived from the call
chain analysis in Chapter 2 and the path selection analysis in Q1. They are stated as
actionable preconditions that the `TracedRun` integration must satisfy before a trace
can be safely captured and replayed.

---

## Concrete requirements

### Requirement 1 — Output buffer stability

The tensor returned by `ttnn.all_reduce` must be pre-allocated before
`ttnn.begin_trace_capture` is called and must be reused, at the same device address,
across all subsequent replays.

The trace capture records the address of every buffer that is written during the
captured forward pass. If the output tensor is freshly allocated on each call (the
default behaviour when no persistent output buffer is provided), its address will
differ between the compile run and the trace capture run, or between successive
replays, causing the trace to write results to the wrong memory location.

**Action:** Allocate the output tensor once, before capture, at `ttnn.DRAM_MEMORY_CONFIG`
(matching the `memory_config` argument already used by `TTNNLinearIColShardedWAllReduced`),
and confirm that `TracedRun._capture_trace` stores the resulting `trace_output` in the
`TraceEntry` so that downstream ops consume the stable address.

### Requirement 2 — Intermediate buffer stability (`scattered_tensor`)

On the non-composite path, `ttnn::reduce_scatter` allocates a `scattered_tensor` whose
shape is $\bigl[\ldots,\ \text{out features} / \text{ring size}\bigr]$ in DRAM. This
tensor is created anew on every call to `ttnn.all_reduce` and is passed to
`ttnn::all_gather`. Its address is recorded into the trace.

To meet the address-stability requirement, `scattered_tensor` must be pre-allocated as
a persistent buffer and passed via the `output_tensor` argument to
`ttnn::reduce_scatter` before trace capture begins.

> **Warning:** The current public Python API of `ttnn.all_reduce` does not expose an
> `output_tensor` parameter for the internal `scattered_tensor`. Satisfying
> this requirement may require either:
> (a) switching to `ttnn.experimental.all_reduce_async` with explicit semaphore and
>     buffer management (see Requirement 2a below), or
> (b) calling `ttnn.reduce_scatter` and `ttnn.all_gather` directly in
>     `TTNNLinearIColShardedWAllReduced.forward`, passing pre-allocated persistent
>     buffers to each, or
> (c) verifying empirically that the DRAM allocator returns the same address for
>     identically-shaped tensors within a fixed trace region, which is not guaranteed
>     and should not be relied upon.

**Requirement 2a (alternative implementation path):** Replace the single
`ttnn.all_reduce` call in `TTNNLinearIColShardedWAllReduced.forward` with explicit
`ttnn.reduce_scatter` and `ttnn.all_gather` calls, passing a pre-allocated
`scattered_tensor` via the `output_tensor` keyword argument to `ttnn.reduce_scatter`.
This gives the caller direct control over the intermediate buffer address, satisfying
the trace stability requirement. See
[`integration_checklist.md`](../ch4_integration/integration_checklist.md) for the
complete annotated implementation pattern.

> **Critical constraint — this is a required code change, not a configuration option.**
> Satisfying Requirement 2 is impossible while `TTNNLinearIColShardedWAllReduced.forward`
> calls `ttnn.all_reduce`. The source of `all_reduce_async.cpp` (cluster\_axis overload,
> lines 344–358) shows that `ttnn::reduce_scatter` is called with `std::nullopt`
> hardcoded for `optional_output_tensor`; the `ttnn.all_reduce` Python API provides no
> parameter to override this. There is no configuration flag, keyword argument, or
> wrapper that can inject a pre-allocated buffer through the `ttnn.all_reduce` entry
> point. The `ttnn.all_reduce` call in the forward method **must be replaced** with
> direct `ttnn.reduce_scatter` + `ttnn.all_gather` calls before the non-composite path
> can be made trace-safe.

> **Note:** `ttnn.experimental.all_reduce_async(input, buffer_tensor, ...)` maps to a
> different overload (`all_reduce_async.cpp` line 426) that routes directly to
> `ttnn::prim::all_reduce_async` — a single-step fused primitive that does **not**
> use the two-step reduce-scatter + all-gather path taken by `ttnn.all_reduce`. The
> `buffer_tensor` in that overload is the intermediate for the fused op, not the
> `scattered_tensor` of the non-composite path. Using the fused overload as a
> drop-in replacement would invoke a fundamentally different execution path with
> different latency, bandwidth, and semaphore requirements.

### Requirement 3 — Fabric configuration

`ttnn.all_reduce` on T3K requires `FABRIC_1D_RING` fabric configuration. Setting
`FABRIC_2D` causes `composite_for_2d_mesh` to evaluate to `true` and forces the
composite path (`composite_all_gather` using `all_broadcast` + `concat`, followed by a
local reduce), which is trace-incompatible.

> **Warning:** `ttnn.FabricConfig.FABRIC_1D` and `ttnn.FabricConfig.FABRIC_1D_RING`
> are distinct enum values. `FABRIC_1D` (value 1) instantiates 1D routing without
> deadlock avoidance; `FABRIC_1D_RING` (value 2) adds deadlock avoidance using
> datelines and is the correct constant for T3K ring topology. Using `FABRIC_1D`
> instead of `FABRIC_1D_RING` will route the ring collective incorrectly.

**Action:** Confirm that `device_params` in the test fixture or the inference harness
sets `"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING` for T3K runs that use
`TTNNLinearIColShardedWAllReduced` under `TracedRun`.

### Requirement 4 — Even ring size

The ring-based collective requires:

$$\text{ring size} \bmod 2 = 0$$

T3K cluster axis 1 has 8 devices, so $8 \bmod 2 = 0$. This requirement is satisfied
for all standard T3K deployments. It becomes relevant only if the mesh view is changed
(e.g., a submesh with an odd number of devices along the collective axis).

### Requirement 5 — Non-composite path confirmed for the production shape

The composite path must not be taken during trace capture or replay. This is
guaranteed when all three of the following hold simultaneously:

1. `memory_config=ttnn.DRAM_MEMORY_CONFIG` is passed to `ttnn.all_reduce` (ensures
   `change_mem_config = false`, skipping the `sharded_to_interleaved` conversion; this
   does **not** affect the `composite_for_2d_mesh` guard, which depends solely on
   fabric config and mesh shape — see item 2).
2. `FABRIC_1D_RING` fabric is active (prevents `composite_for_2d_mesh`, since that guard
   checks `GetFabricConfig() == FABRIC_2D && is_true_2d_mesh(input_tensor)` — fabric
   and mesh shape are the only inputs; memory config has no bearing on this guard).
3. `out_features / ring_size` is a multiple of the tile width (32 by default):
   $$\frac{\text{out features}}{8} \bmod 32 = 0$$
   which is equivalent to requiring $\text{out features}$ to be a multiple of 256.

Standard model hidden dimensions used on T3K (4096, 8192, 14336, 28672, etc.) satisfy
condition 3. If a model with a non-standard hidden dimension is introduced, the
composite path may be activated and the trace will be invalid.

**Action:** Add an assertion in `TTNNLinearIColShardedWAllReduced.__init__` or in the
`TracedRun` path that validates:

```python
assert self.out_features % (8 * 32) == 0, (
    f"out_features={self.out_features} is not divisible by 256; "
    "ttnn.all_reduce will take the composite path, which is trace-incompatible."
)
```

---

## `@trace_enabled` status of `TTNNLinearIColShardedWAllReduced`

`TTNNLinearIColShardedWAllReduced` inherits from `TTNNLinearIColShardedWRowSharded`,
which inherits from `TTNNLinear`. `TTNNLinear` is decorated with `@trace_enabled`
(line 15 of `linear.py`). Neither `TTNNLinearIColShardedWRowSharded` nor
`TTNNLinearIColShardedWAllReduced` carry an explicit `@trace_disabled` decorator.

The `is_trace_enabled` predicate in `run_config.py` is:

```python
def is_trace_enabled(module) -> bool:
    return (isinstance(module, tuple(_TRACE_ENABLED_CLASSES)) and
            not isinstance(module, tuple(_TRACE_DISABLED_CLASSES)))
```

Because `TTNNLinear` is in `_TRACE_ENABLED_CLASSES` and `TTNNLinearIColShardedWAllReduced`
is not in `_TRACE_DISABLED_CLASSES`, instances of `TTNNLinearIColShardedWAllReduced`
satisfy `is_trace_enabled`. `TracedRun.module_run` will therefore attempt to trace
them, making Requirements 1–5 above load-bearing for correct execution.

> **Key finding:** `TTNNLinearIColShardedWAllReduced` is in the `@trace_enabled` set
> by inheritance from `TTNNLinear`. No explicit annotation change is needed to enable
> tracing, but all five requirements above must be satisfied before a trace is
> attempted.

---

## Known limitations

### Limitation 0 — `ttnn.all_reduce` cannot be used as the trace-safe entry point

Using `ttnn.all_reduce` as the entry point for the non-composite path prevents the
`scattered_tensor` pre-allocation required for trace compatibility. The internal call
to `ttnn::reduce_scatter` inside `all_reduce_async.cpp` hardcodes `std::nullopt` for
`optional_output_tensor` (cluster\_axis overload, lines 344–358); the `ttnn.all_reduce`
public API exposes no mechanism to override this. The forward method of
`TTNNLinearIColShardedWAllReduced` must therefore be refactored to call
`ttnn.reduce_scatter` + `ttnn.all_gather` directly, bypassing `ttnn.all_reduce`, before
the trace-safety precondition in Requirement 2 can be satisfied.

### Limitation 1 — Composite path is unconditionally trace-incompatible

If any of the conditions in Requirement 5 is violated and the composite path is taken
during trace capture, the trace will be structurally incorrect. The composite path
calls `composite_common::composite_all_gather` (which uses `all_broadcast` + `concat`)
followed by a local reduce: `local_sum` (using `ttnn::moreh_sum`) for non-float32
dtypes, or `local_sum_float32` (using `ttnn::transpose` + `ttnn::sum`) for float32.
This allocates two transient buffers for `TTNNLinearIColShardedWAllReduced`
(`reshaped_tensor` and `gather_tensor`). A third buffer (`interleaved_tensor` from
`sharded_to_interleaved`) is only present when the input is sharded
(`change_mem_config = true`); since `TTNNLinearIColShardedWAllReduced` always produces
DRAM interleaved input, this third buffer does not appear in this caller's composite
path. None of these buffers are currently surfaced through the `ttnn.all_reduce` public
API for pre-allocation. Detecting this situation requires either an assertion on the
input shape or a runtime check of which branch was taken.

### Limitation 2 — `scattered_tensor` is not currently managed by `TracedRun`

`TracedRun._capture_trace` in `run_config.py` pre-allocates persistent buffers only
for the function's positional tensor arguments (the forward pass inputs). It does not
pre-allocate or track the internal intermediates of the ops called inside `forward`.
The `scattered_tensor` created by `ttnn::reduce_scatter` inside `ttnn.all_reduce` is
therefore not under `TracedRun`'s control. Until the implementation is extended to
manage this buffer — or the forward method is rewritten to call `ttnn.reduce_scatter`
directly with a pre-allocated output — the intermediate buffer instability risk from
Requirement 2 is not mitigated by `TracedRun` alone.

### Limitation 3 — Bias addition after `ttnn.all_reduce`

`TTNNLinearIColShardedWAllReduced.forward` adds the bias with `tt_output += self.tt_bias`
after `ttnn.all_reduce`. The in-place add allocates a new output tensor. This is a
separate trace buffer stability concern: `self.tt_bias` is a persistent weight on
device (stable address), but the output of `+=` may or may not reuse the same buffer
as `tt_output`. This must be verified or explicitly handled before trace capture.

---

**Next:** [Chapter 4 — Integration Checklist and Test Strategy](../ch4_integration/index.md)
