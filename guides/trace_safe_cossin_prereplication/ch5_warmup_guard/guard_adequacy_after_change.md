# Guard Adequacy After the Pre-Replication Change

The `rotary_dim % 64 != 0` guard in `TTNNRotaryPositionEmbedding.forward` was designed to catch sharded cos/sin tensors that have not been replicated. This document evaluates whether the guard remains adequate after the pre-replication change — and recommends a complementary post-allocation assertion to close the gap between what the guard checks and what the pre-allocation code guarantees.

---

## Before the Change: What the Guard Caught

Before the change, `forward` called `_ensure_replicated(cos)` to convert a potentially sharded input tensor into a replicated tensor. The guard in `TTNNRotaryPositionEmbedding.forward` acted as a post-condition check on that conversion: if `_ensure_replicated` returned something still sharded (e.g., due to a bug in the replication logic or an incorrect mapper), the guard would fire.

The guard's effective scope was:

- Detect that the tensor reaching the rotary embedding kernel has fewer than `rotary_dim` columns per device.
- Raise immediately before the kernel is dispatched, producing a diagnostic rather than a silent numerical error.

---

## After the Change: What the Guard Catches Now

After the change, `self._cos_replicated` is the tensor that reaches `TTNNRotaryPositionEmbedding.forward`. Its replication is established at pre-allocation time in `move_weights_to_device_impl`, not at forward-pass time. The guard now checks a buffer whose replication is a static property, not a dynamic property computed on each call.

This has one important consequence: the guard can no longer catch a replication failure that happens dynamically during `forward`. However, because `ttnn.copy(cos, self._cos_replicated)` does not change the tensor's memory layout or mapper — it only writes data — a correctly pre-allocated `_cos_replicated` will always present the correct column count to the guard. The only misconfiguration the guard can now catch is a wrong mapper used during pre-allocation.

The guard remains necessary and should not be removed. A misconfigured pre-allocation (e.g., `ShardTensorToMesh` used instead of `ReplicateTensorToMesh`) would still be caught at warm-up time, before any trace capture.

---

## Recommended Post-Allocation Assertion in `move_weights_to_device_impl`

Because the guard in `TTNNRotaryPositionEmbedding.forward` now checks a pre-allocated buffer rather than a dynamically produced tensor, it is best practice to add an explicit assertion immediately after pre-allocation in `move_weights_to_device_impl`. This assertion runs once at device-load time, before any forward pass, and catches misconfigured mappers at the earliest possible point.

```python
# In move_weights_to_device_impl, immediately after pre-allocating _cos_replicated:
# why: verify that every per-device tensor has the full rotary_dim columns,
#      confirming that ReplicateTensorToMesh was applied correctly.
for i, per_device_tensor in enumerate(ttnn.get_device_tensors(self._cos_replicated)):
    actual_cols = per_device_tensor.shape[-1]
    assert actual_cols == self.rotary_dim, (
        f"_cos_replicated device {i} has {actual_cols} columns; "
        f"expected {self.rotary_dim}. "
        "Check that ReplicateTensorToMesh was used, not ShardTensorToMesh."
    )
```

Apply the same assertion for `_sin_replicated`. This assertion executes in Python eager mode and is not inside any trace bracket, so it has no trace-safety implications.

---

## Recommended Warm-Up-Only Debug Assertion

For diagnosing issues that only manifest on the first warm-up pass, a lightweight debug assertion can be added at the top of `TTNNRotaryPositionEmbedding.forward`, guarded by an environment variable so it does not run in production:

```python
import os

# In TTNNRotaryPositionEmbedding.forward, before the guard check:
# why: when TTNN_DEBUG=1, log the per-device column count of the incoming
#      cos tensor so that sharding misconfigurations are visible in logs
#      without requiring a debugger.
if os.environ.get("TTNN_DEBUG") == "1":
    for i, t in enumerate(ttnn.get_device_tensors(cos)):
        print(
            f"[TTNN_DEBUG] rotary forward device {i}: "
            f"cos.shape={t.shape}, cols={t.shape[-1]}"
        )
```

This block runs only in Python eager mode (during warm-up and the capture pass). It does not execute on trace replay iterations and therefore cannot affect trace correctness. Set `TTNN_DEBUG=1` in the shell environment to enable it; leave it unset in production.

> **Note:** The `ttnn.get_device_tensors` call itself is pure Python inspection — it does not issue any device commands and is safe to call both inside and outside the trace bracket. However, the print statements make it unsuitable for use inside a performance-critical path.

---

## Summary of Guard Coverage Before and After

| Failure mode | Guard catches before change | Guard catches after change | Post-allocation assertion catches |
|---|---|---|---|
| Sharded mapper at pre-allocation | N/A | Yes, at warm-up | Yes, at device-load time |
| `_ensure_replicated` bug | Yes, at warm-up | N/A (removed from traced path) | N/A |
| `ttnn.copy` changing layout | N/A | No (copy does not change layout) | N/A (assertion runs once before copy) |
| `None` pre-allocated tensor | AttributeError | AttributeError | Would fail assertion first |

The combined coverage of the warm-up guard plus the post-allocation assertion is at least as strong as the original guard alone.
