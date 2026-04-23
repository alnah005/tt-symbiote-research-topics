# What `ttnn.copy` Records in the Trace Command Buffer

When `ttnn.copy(source, destination)` is called inside a Metal Trace capture bracket, the Metal runtime enqueues a single DMA transfer command into the trace command buffer. That command encodes the source device buffer address and the destination device buffer address — and nothing else. No new device buffer is created. The destination tensor's address was fixed when it was pre-allocated before capture, and that same address is what the trace records. On every replay, the trace engine re-issues that DMA command verbatim, writing updated data from source into the pre-existing destination buffer at the same stable address.

---

## Comparison: `ttnn.copy`, `ttnn.clone`, and Python Assignment

These three constructs look superficially similar but have completely different effects on the trace command buffer:

```python
# ttnn.copy(src, dst)
# why: enqueues a DMA transfer command; source address -> destination address;
#      no new buffer is allocated; destination address is stable (pre-allocated);
#      this command is recorded in the trace and replayed verbatim each step.
ttnn.copy(cos, self._cos_replicated)   # trace-safe ✓

# ttnn.clone(tensor)
# why: allocates a NEW destination buffer internally; the new buffer's address
#      was not known at capture time; recording this command would embed an
#      address that becomes invalid on the next replay; trace-UNSAFE.
replicated_cos = ttnn.clone(cos)       # trace-UNSAFE ✗

# dst = src  (Python assignment)
# why: this is a Python variable rebind; it does NOT issue any device command;
#      the trace command buffer is unaffected; no DMA transfer occurs;
#      the destination variable now points to the same underlying device buffer
#      as the source, but no data movement has happened.
self._cos_replicated = cos             # no device command at all — NOT a copy
```

The table below summarizes the three cases:

| Construct | Device command recorded | Allocates new buffer | Trace-safe |
|---|---|---|---|
| `ttnn.copy(src, dst)` | DMA transfer, stable addresses | No | Yes ✓ |
| `ttnn.clone(tensor)` | Allocates + DMA transfer | Yes | No ✗ |
| `dst = src` (Python) | None | No | N/A (no-op on device) |

---

## Replicated Destination on a Multi-Device Mesh

On a T3K 8-device mesh, `_cos_replicated` is a replicated tensor — each device holds a local shard at its own device-local buffer address. When `ttnn.copy` is called inside the trace bracket, the runtime issues one DMA command per device. Each device copies from its local source tensor shard to its local `_cos_replicated` shard. Both addresses are stable because both tensors were pre-allocated across all devices before `begin_trace_capture` was called. The trace command buffer records all eight per-device DMA commands. On every replay, all eight DMA commands are re-issued with the same eight pairs of source and destination addresses.

```python
# Conceptual view of what the trace records on an 8-device mesh:
# [device 0] DMA: src_addr_dev0 -> cos_replicated_addr_dev0
# [device 1] DMA: src_addr_dev1 -> cos_replicated_addr_dev1
# ...
# [device 7] DMA: src_addr_dev7 -> cos_replicated_addr_dev7
# why: all 8 source shards and all 8 destination shards were pre-existing
#      at capture time — every address is stable.
```

> **Trace Invariant:** Both the source and destination tensors passed to `ttnn.copy` inside the trace bracket must be pre-existing device tensors at addresses that were valid at capture time. Creating either tensor after `begin_trace_capture` (e.g., via `ttnn.from_torch`) invalidates the trace.

---

## Source Address Stability

Source address stability — and what would break it — is analyzed in [`source_tensor_stability.md`](./source_tensor_stability.md).
