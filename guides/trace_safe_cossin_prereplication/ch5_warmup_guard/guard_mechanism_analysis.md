# Guard Mechanism Analysis

`TTNNRotaryPositionEmbedding.forward` contains a guard that detects mis-sharded cos/sin tensors: when `rotary_dim % 64 != 0` the method raises an error immediately, before the rotary embedding kernel is dispatched. Understanding when this guard fires, what it detects, and why it continues to work after the pre-replication change requires tracing back to why a sharded cos/sin tensor triggers it in the first place.

---

## Why a Sharded Cos/Sin Triggers the Guard

On a T3K 8-device mesh, if cos/sin are produced by sharding a `[1, 1, 1, rotary_dim]` tensor across all 8 devices along the last axis, each device's local shard has shape `[1, 1, 1, rotary_dim / 8]`. For Qwen3 with `rotary_dim = 64`, that is `[1, 1, 1, 8]` columns per device.

The guard checks the column count visible on each device. The value `8` is not a multiple of 64, so the guard fires. This is the correct behavior: a sharded cos/sin cannot be used directly by `ttnn.experimental.rotary_embedding` because the kernel on each device expects the full `rotary_dim` columns, not a shard.

The guard is phrased as `rotary_dim % 64 != 0` rather than checking for replication explicitly because the anomaly — a column count smaller than a tile — is a reliable proxy for "this tensor is sharded and should have been replicated before being passed here."

---

## When the Guard Fires

`TTNNRotaryPositionEmbedding.forward` is called on every forward pass, including:

- The warm-up compile run that happens before `begin_trace_capture`. This is the call that matters for most diagnostic purposes, because it is the first full execution of the model in Python eager mode.
- Any non-traced inference call (e.g., prefill, or evaluation runs without tracing enabled).
- Decode steps executed inside the trace bracket — but note that the trace bracket contains one Python-executed capture pass (during which device commands are recorded) and then zero or more replay calls (which re-issue those commands without executing Python). The guard check is Python code and therefore runs during the capture pass but not on any replay iteration.

The key point is that the guard runs in Python eager execution. It fires during warm-up before trace capture, and it fires again during the one Python-executed capture pass. It does not fire on replay iterations (which re-issue recorded device commands only).

---

## Why the Guard Still Works After Pre-Replication

Before the change, `_ensure_replicated(cos)` was called inside `forward`, and its output was passed to `TTNNRotaryPositionEmbedding.forward`. The guard checked the output of `_ensure_replicated`, which was a replicated tensor with full `rotary_dim` columns on each device — and the guard passed without raising.

After the change, `ttnn.copy(cos, self._cos_replicated)` is called instead, and `self._cos_replicated` — the pre-allocated replicated buffer — is passed downstream. The guard now checks `self._cos_replicated`. Because `_cos_replicated` was pre-allocated with `ReplicateTensorToMesh(mesh_device)` and shape `[1, 1, 1, rotary_dim]`, every device holds all `rotary_dim` columns. The column count is `rotary_dim`, which is 64 for Qwen3, and `64 % 64 == 0` — the guard passes without raising.

The only way the guard would fire after the change is if `_cos_replicated` were accidentally pre-allocated with a sharded mapper instead of a replicated mapper. That is exactly the misconfiguration the guard is meant to catch, so the behavior is correct and intentional.

---

## Edge Case: `forward` Called Before `move_weights_to_device_impl`

If `forward` is called before `move_weights_to_device_impl` has run, `self._cos_replicated` and `self._sin_replicated` do not exist yet. Accessing them in `forward` would raise an `AttributeError`, which would surface as an obscure error unrelated to the guard.

To prevent this, use one of two patterns:

**Pattern 1 — `__init__` default with `None`:**

```python
# In TTNNQwen3FullAttention.__init__:
# why: establish the attribute at init time so hasattr/None checks work
#      even if move_weights_to_device_impl has not been called yet.
self._cos_replicated = None
self._sin_replicated = None
```

**Pattern 2 — `hasattr` guard in `forward`:**

```python
# In TTNNQwen3FullAttention.forward, before the ttnn.copy calls:
# why: raises a clear, actionable error if the pre-allocation hook
#      was not called before entering the traced forward path.
if not hasattr(self, "_cos_replicated") or self._cos_replicated is None:
    raise RuntimeError(
        "TTNNQwen3FullAttention: _cos_replicated is not initialized. "
        "Call move_weights_to_device_impl before entering the trace path."
    )
```

Pattern 1 is preferred because it makes the object's state explicit from construction. Pattern 2 provides a better error message but requires the `hasattr` check to execute on every forward pass, including inside the trace capture pass. Both patterns are trace-safe because they are Python-level checks that run only in eager mode.

> **Warning:** Do not place a `None` sentinel check inside the trace bracket — even though Python code in the capture pass runs eagerly, any tensor operation that depends on the `None` check branching differently at replay time can cause silent correctness errors. Keep the guard check before any device operations.
