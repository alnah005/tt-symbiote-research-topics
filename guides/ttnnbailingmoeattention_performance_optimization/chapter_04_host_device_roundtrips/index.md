# Chapter 4 — Host-Device Round-Trips and On-Device Alternatives

## Why Host-Device Round-Trips Are Latency Killers in Decode

Every synchronous data transfer between the host CPU and a Tenstorrent device passes through the PCIe bus and stalls the Python dispatch thread. The mechanism is:

1. Before a `ttnn.to_torch` can read back a result, the host must issue a device synchronization to drain the command queue — all preceding ops that have been dispatched but not yet executed must complete before the read is safe. On T3K this synchronization flush costs several hundred microseconds in practice, even when the tensor being read back is only a few bytes.
2. After the read, the host performs some Python-level computation on the result (e.g., slicing, casting), then calls `ttnn.from_torch` to upload the modified tensor back to the device. This upload incurs PCIe write latency plus topology metadata setup on each of the N=8 devices in the mesh, even for tiny 1-D tensors.
3. The next decode step cannot begin dispatching ops that depend on the uploaded tensor until the upload completes, so the stall is directly on the critical path of each autoregressive token step.

For comparison, a fully on-device operation dispatches asynchronously; the host enqueues it and immediately proceeds to dispatch the next op. The only serialization is within the device's own command queue, which runs entirely in hardware without CPU involvement. Replacing a synchronous host round-trip with an on-device equivalent therefore removes a guaranteed per-step stall and replaces it with hardware-pipelined execution.

### Quantitative Reference

On T3K hardware (8 × Wormhole N300 connected by PCIe), a single `ttnn.to_torch` + `ttnn.from_torch` sequence on a small tensor costs roughly:

| Component | Approximate Cost |
|---|---|
| Device command queue drain (sync flush) | 200–500 µs (rough estimate; varies with queue depth) |
| PCIe read: device→host, N=8 devices | ~1–5 µs for a 32-byte tensor (per device read, PCIe 4×16 lanes) |
| Host-side Python processing | <10 µs for simple slice/cast |
| PCIe write: host→device, N=8 devices | ~5–20 µs total across 8 devices with `ReplicateTensorToMesh` |

The queue drain dominates. At 500 µs per round-trip and a 2 ms target decode step budget for the full transformer block, a single round-trip consumes 25% of the budget before any attention computation begins.

---

## Inventory of Host-Device Touches in `_forward_decode_paged`

The following table enumerates every host-device data transfer in `TTNNBailingMoEAttention._forward_decode_paged` (lines 2610–2799 of `attention.py`).

| # | Operation | Line(s) | Direction | Tensor | Size (typical B=1) | Per-step? |
|---|---|---|---|---|---|---|
| 1 | `ttnn.to_torch(cp, mesh_composer=...)` | 2674 | device→host | `cache_position` (int32) | 4 bytes × B | Yes, only when `cache_position` is a `ttnn.Tensor` |
| 2 | `ttnn.from_torch(cache_position_tensor, ...)` | 2678–2685 | host→device | `cur_pos_tt` (int32, 1-D, length B) | 4 bytes × B replicated to N=8 devices | Yes, every decode step |
| 3 | `get_cos_sin_for_decode`: `ttnn.from_torch(pos_indices, ...)` | rope.py 444–451 | host→device | position index (uint32, shape `[1, B]`) | 4 bytes × B replicated to N=8 devices | Yes, every decode step |

Notes:

- Row 1 (`ttnn.to_torch` at line 2674) is conditional: it fires only when `cache_position` is supplied as a `ttnn.Tensor`. When `cache_position` is `None` (lines 2663–2665), the host constructs a fresh `torch.tensor` directly and skips the device→host read. In practice, callers that use persistent `cache_position` tracking will hit the round-trip path.
- Row 2 (`ttnn.from_torch` at lines 2678–2685) fires unconditionally on every decode step regardless of which branch produced `cache_position_tensor`. This is the unavoidable per-step upload.
- Row 3 occurs inside `BailingRotarySetup.get_cos_sin_for_decode` (`rope.py`, lines 444–451). The `cache_position_tensor` is already a host `torch.Tensor` at the call site (line 2696 of `attention.py`), so no device→host read is needed; however, a host→device upload of the position index still happens each step before the `ttnn.embedding` call.
- The `_to_replicated` method (lines 2288–2313 of `attention.py`) implements a full device→host→device round-trip but is **not called** anywhere in `TTNNBailingMoEAttention._forward_decode_paged`. It is retained in the class for potential use but has been eliminated from the hot path. In contrast, `TTNNQwen3FullAttention` calls `_to_replicated` on Q, K, and V at lines 766–768 of `qwen_attention.py`, and `TTNNGlm4MoeLiteAttention` calls it on Q, K, and V at lines 1895–1897 of `attention.py`.

---

## Files in This Chapter

- [**`cur_pos_roundtrip.md`**](cur_pos_roundtrip.md) — Deep-dive on the `cur_pos_tt` creation pattern (lines 2663–2685), its two execution paths, its cost on T3K, and the on-device persistent-tensor alternative.
- [**`to_replicated_analysis.md`**](to_replicated_analysis.md) — Analysis of the `_to_replicated` method: why it exists, why Bailing eliminated it from `_forward_decode_paged`, and why Qwen3 and GLM4 still pay its cost three times per decode step.
- [**`get_cos_sin_host_ops.md`**](get_cos_sin_host_ops.md) — Examination of `BailingRotarySetup.get_cos_sin_for_decode` (`rope.py` lines 420–472): what host ops remain, what the data flow looks like, and how to eliminate the residual per-step upload.
