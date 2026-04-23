# The Replicated Mesh Mapping for cos/sin

A replicated tensor on a T3K mesh means every device holds a complete copy of the tensor data, as opposed to a sharded tensor where each device holds a distinct slice. By the end of this file you will understand why cos/sin must be replicated rather than sharded, how to specify the correct mapper in the pre-allocation call, and how to verify at runtime that replication was applied correctly.

---

## Section 1: What "Replicated" Means on a T3K Mesh

A replicated tensor means every device in the 8-device T3K mesh (1×8 logical ring) holds a full copy of the tensor data. When `ttnn.from_torch` is called with `mesh_mapper=ReplicateTensorToMesh(mesh_device)`, TTNN transfers the tensor from host and DMA-copies it to all 8 devices; each device's shard is identical to the host tensor.

Contrast with sharded: a sharded tensor (e.g., from `ShardTensorToMesh`) places different slices on different devices. For a `[1, 1, 1, 64]` cos/sin table, a row-shard across 8 devices would give each device a `[1, 1, 1, 8]` slice — not the full frequency table.

> **Note:** The terms "shard" and "replicated tensor" can be confusing because `ttnn.get_device_tensors` returns one per-device tensor in both cases. The distinction is the content: for a replicated tensor every per-device tensor is an identical full copy; for a sharded tensor every per-device tensor is a distinct slice.

---

## Section 2: Why cos/sin Must Be Replicated

`TTNNQwen3FullAttention` uses tensor parallelism: attention heads are sharded across devices so that each device holds `n_heads / num_devices` heads. Each device applies rotary embedding to its own head shard using its local copy of the cos/sin table. Because each device needs the full `[1, 1, 1, rotary_dim]` cos/sin table to rotate all its heads, the table must be replicated — a partial slice is incorrect.

The original crash occurred because cos/sin arrived sharded: each device received only `rotary_dim / num_devices = 8` columns, which is insufficient for the 64-column table expected by `ttnn.experimental.rotary_embedding`. This is the bug `_ensure_replicated` was added to fix — and the reason `ReplicateTensorToMesh` is required for the pre-allocated buffer.

> **Trace Invariant:** The mesh mapper specified when the pre-allocated buffer is created in `move_weights_to_device_impl` is permanent for the lifetime of the buffer. `ttnn.copy` into a replicated destination does not change the buffer's mesh distribution — the destination remains replicated. The source tensor passed to `ttnn.copy` must therefore also be replicated (or at minimum broadcastable to replicated); copying a sharded source into a replicated destination will produce device-specific values that are not identical across devices, which may be incorrect depending on the upstream sharding axis.

---

## Section 3: The Correct Mapper

The correct mapper is `ReplicateTensorToMesh(self.mesh_device)`. See the annotated implementation in [`move_weights_impl_changes.md` Section 1](./move_weights_impl_changes.md) for the complete call with all attributes explained.

> **Note:** `ReplicateTensorToMesh` is the same mapper used by `_decode_cur_pos` in the existing pre-allocation pattern, as established in Chapter 2.

---

## Section 4: Verifying Replication at Runtime

Add a debug assertion in `move_weights_to_device_impl` (guarded by a debug flag or warm-up phase) to confirm that the buffer is correctly replicated before the trace capture begins:

```python
# Debug assertion: verify replication
device_tensors = ttnn.get_device_tensors(self._cos_replicated)
assert len(device_tensors) == self.mesh_device.get_num_devices(), \
    "Expected one tensor per device"
for t in device_tensors:
    assert t.shape[-1] == rotary_dim, \
        f"Expected full rotary_dim={rotary_dim} on each device, got {t.shape[-1]}"
```

This assertion is O(1) in device ops (the tensors are already on device; this just checks metadata) and can be run during the warm-up compile pass. The warm-up guard analysis in [`../ch5_warmup_guard/guard_adequacy_after_change.md`](../ch5_warmup_guard/guard_adequacy_after_change.md) discusses this assertion further and recommends it as an explicit replacement for the shape-heuristic guard.

---

## Section 5: Memory Cost

**Decode (seq_len=1):**

Per device: `rotary_dim * 2 bytes` = `64 * 2 = 128 bytes` payload per device per decode step. Across 8 T3K devices: `8 * 128 = 1,024 bytes` total. This is negligible.

With TILE_LAYOUT padding (seq_len dimension padded from 1 to 32): effective per-device footprint is `1 * 1 * 32 * 64 * 2 = 4,096 bytes`. Still negligible.

**Prefill (seq_len > 1):**

The buffer size grows to `seq_len * rotary_dim * 2 bytes` per device. For a 2,048-token prefill: `2048 * 64 * 2 = 262,144 bytes ≈ 256 KB` per device. Still DRAM-appropriate. However, the pre-allocated buffer shape `[1, 1, 1, 64]` cannot serve prefill without reallocation because the seq_len dimension is fixed at 1. The prefill path is out of scope for this guide's decode-trace fix; see [`../ch6_integration_and_testing/prefill_scope_note.md`](../ch6_integration_and_testing/prefill_scope_note.md).
