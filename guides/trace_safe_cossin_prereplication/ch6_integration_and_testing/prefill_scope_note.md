# Prefill Scope Note

This guide covers the decode trace path only. The pre-replication change — replacing `_ensure_replicated` with `ttnn.copy` into pre-allocated `_cos_replicated` and `_sin_replicated` buffers — is designed for decode, where the sequence length is always 1 and the cos/sin tensor shape is fixed at `[1, 1, 1, rotary_dim]`. Prefill has a fundamentally different cos/sin lifecycle that makes the same approach inapplicable without additional design work.

---

## Why Prefill Is Different

At prefill time, the sequence length varies: the model processes an input of length `seq_len`, where `seq_len` can be anywhere from 1 to the maximum supported context length. The cos/sin tensors at prefill time have shape `[1, 1, seq_len, rotary_dim]`. Because `seq_len` varies per request, a single pre-allocated buffer with a fixed shape cannot serve all prefill inputs without reallocation or padding.

Prefill traces (when used) are typically compiled once per supported sequence length. This means a correct prefill pre-allocation strategy requires one `_cos_replicated_prefill_{seq_len}` buffer per supported sequence length — a potentially large set of buffers. The allocation and lookup logic is more complex, and the memory footprint scales linearly with the number of supported lengths.

---

## Interim Recommendation

Retain `_ensure_replicated` on the prefill code path in the short term. `_ensure_replicated` calls `ttnn.from_torch` inside the forward pass, which is trace-unsafe. However, prefill is typically run in eager mode (without trace capture) or under a separate trace regime, and the trace-safety constraint is less urgent for prefill than for decode. Using `_ensure_replicated` for prefill allows the decode trace path to be validated and merged independently.

The relevant code pattern in `TTNNQwen3FullAttention.forward` is:

```python
# Separate decode (traced) and prefill (eager or separately traced) paths:
# why: decode uses a fixed-shape pre-allocated buffer (trace-safe);
#      prefill uses dynamic replication because seq_len varies per call.
if is_decode:
    ttnn.copy(cos, self._cos_replicated)
    ttnn.copy(sin, self._sin_replicated)
    cos_for_rotary = self._cos_replicated
    sin_for_rotary = self._sin_replicated
else:
    # Prefill path: dynamic replication; not in a trace bracket.
    cos_for_rotary = self._ensure_replicated(cos)
    sin_for_rotary = self._ensure_replicated(sin)
```

This branching approach keeps `_ensure_replicated` available for prefill without polluting the decode trace path.

---

## Future Prefill Pre-Allocation Strategy

When prefill trace support becomes a requirement, the recommended approach is:

1. Define the set of supported prefill sequence lengths (e.g., powers of two: 128, 256, 512, 1024, 2048).
2. In `move_weights_to_device_impl`, pre-allocate one pair of `(_cos_prefill_{L}, _sin_prefill_{L})` buffers for each supported length `L`, using `TILE_LAYOUT`, `DRAM_MEMORY_CONFIG`, `ReplicateTensorToMesh`, and shape `[1, 1, L, rotary_dim]`.
3. In `forward`, at prefill time, look up the correct buffer pair by `seq_len` and use `ttnn.copy` to update it before the trace bracket executes.

This mirrors the decode pattern exactly, but applied across a dictionary of buffers keyed by sequence length. The lookup itself (a Python dict access) is not inside the trace bracket and is therefore trace-safe.

Defer this work until the decode trace path has been verified in production. Introducing both decode and prefill pre-allocation in the same change makes it harder to isolate failures.

> **Note:** The separate research topics `full_stack_trace_prerequisites/` cover the broader set of requirements for full-stack trace capture, including prefill. This guide's scope is intentionally limited to the decode path to keep the change reviewable and testable independently.
