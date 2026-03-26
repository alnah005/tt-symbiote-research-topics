# `cur_pos_tt` Creation: Per-Step Host-Device Transfer

## The Pattern in `_forward_decode_paged`

Every decode step in `TTNNBailingMoEAttention._forward_decode_paged` must supply the current sequence position to the paged KV-cache kernels (`paged_update_on_device` at line 2756 and `paged_sdpa_decode` at line 2766). Both kernels accept this as a device-resident 1-D int32 tensor of length B called `cur_pos_tt`. The block that creates it spans lines 2663–2685 of `attention.py` and has two paths depending on whether `cache_position` is provided by the caller.

### Path 1: `cache_position` is `None` (lines 2663–2665)

```python
cur_pos = past_key_values.get_seq_length(layer_idx)          # host: int lookup
cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)  # host: alloc
```

When no `cache_position` argument is supplied, the current position is read from the KV cache's own sequence-length counter (`get_seq_length` is a host-side Python call into the cache metadata; it does not touch device memory). A 1-element `torch.Tensor` is constructed on host. This path avoids a device→host read but still requires the `ttnn.from_torch` upload that follows.

### Path 2: `cache_position` is a `ttnn.Tensor` (lines 2667–2675)

```python
cp = cache_position
if isinstance(cp, TorchTTNNTensor):
    cp = cp.to_torch
if isinstance(cp, ttnn.Tensor):
    mesh_composer = None
    if hasattr(cp, "device") and cp.device() is not None and cp.device().get_num_devices() > 1:
        mesh_composer = ttnn.ConcatMeshToTensor(cp.device(), dim=0)
    cp = ttnn.to_torch(cp, mesh_composer=mesh_composer)       # device→host sync stall
cache_position_tensor = cp.flatten()[:batch_size].to(torch.int32)
```

When `cache_position` arrives as a device-resident `ttnn.Tensor` — which is the case when callers maintain a persistent position tracker on device — the code calls `ttnn.to_torch` with a `ConcatMeshToTensor` mesh composer. This is a **synchronous device→host read**:

1. The host issues a device synchronization that forces all in-flight ops in the command queue to complete.
2. Data is read back from all N=8 devices over PCIe and concatenated on the host.
3. The host slices to `[:batch_size]` and casts to `int32`.

The queue drain in step 1 is the expensive part. For a T3K system that has issued a full sequence of attention ops in the previous decode step, the queue may be several hundred ops deep; draining it before the read is the serialization barrier that blocks the next decode step.

### Unconditional Upload (lines 2677–2685)

Both paths converge at:

```python
mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
cur_pos_tt = ttnn.from_torch(
    cache_position_tensor,
    device=self.device,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=mesh_mapper,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

`ttnn.from_torch` with `ReplicateTensorToMesh` uploads `cache_position_tensor` to all N=8 devices. For a batch-1 tensor of 4 bytes, the data volume is negligible, but the upload requires:

- Allocating a DRAM buffer on each of the 8 devices via the device driver.
- Writing 4 bytes × 8 devices over PCIe (8 separate PCIe write transactions, one per device).
- Recording the topology metadata (`ReplicateTensorToMesh`) in the multi-device tensor wrapper.

The cumulative host-side overhead of buffer allocation + 8 PCIe writes + metadata bookkeeping is measurably larger than the 32 bytes of actual data being transferred. This cost is paid unconditionally on every decode step, and it serializes the creation of `cur_pos_tt` that both `paged_update_on_device` and `paged_sdpa_decode` depend on.

---

## The Same Pattern in Qwen3 and GLM4

The three implementations differ in single-device handling. Bailing (line 2677) sets `mesh_mapper` unconditionally:

```python
mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
```

Qwen3 (line 745 of `qwen_attention.py`) and GLM4 (line 1871 of `attention.py`) both apply a conditional guard:

```python
mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
```

On a single-device configuration, Bailing always applies `ReplicateTensorToMesh` while Qwen3 and GLM4 pass `None`. All three check for a `ttnn.Tensor` `cache_position`, call `ttnn.to_torch` if found, and then call `ttnn.from_torch` with their respective `mesh_mapper`. This is a codebase-wide pattern, not specific to the Bailing attention implementation, but any fix must account for the single-device guard difference when applying changes consistently across all three classes.

---

## Cost Estimation

For a 1-D int32 tensor of length B on a T3K 1×8 mesh, Path 1 (no prior device tensor) costs roughly 20–60 µs per step, while Path 2 (with device→host sync stall) costs roughly 220–560 µs per step. See [`index.md`](index.md) for a breakdown of individual cost components. At batch=1 on T3K, Path 2 can therefore contribute 5–25% of the 2–5 ms per-step attention budget before any tensor computation begins.

---

## On-Device Alternative: Persistent Device-Resident Position Tensor

The fundamental problem is that `cur_pos_tt` is reconstructed from scratch each decode step when it could instead be maintained as a persistent device tensor that is incremented in-place.

**Proposed approach:**

At model initialization, create a device-resident position tensor once:

```python
# One-time setup: shape [B] to support variable batch size
self._cur_pos_device = ttnn.from_torch(
    torch.zeros(batch_size, dtype=torch.int32),
    device=self.device,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
_ones = ttnn.from_torch(
    torch.ones(batch_size, dtype=torch.int32),
    device=self.device,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

At the start of each decode step, instead of calling `ttnn.from_torch`:

```python
# On-device increment: dispatches asynchronously, no host stall
self._cur_pos_device = ttnn.add(self._cur_pos_device, _ones)
cur_pos_tt = self._cur_pos_device
```

The `ttnn.add` dispatches to the device command queue without a host synchronization. The host immediately proceeds to enqueue the subsequent ops (`paged_update_on_device`, `paged_sdpa_decode`) that depend on `cur_pos_tt`, and the device executes the chain in order without CPU involvement.

**Requirements for this approach:**

1. The persistent tensor must be reset to the correct starting position at the beginning of each sequence (i.e., at prefill time or when a KV slot is reused). This requires one `ttnn.from_torch` call per sequence start — a one-time cost, not per-step.
2. For batch processing where different batch slots may be at different positions (e.g., continuous batching with heterogeneous prompts), the increment would need to be position-specific. A per-slot position vector can still be updated on-device using `ttnn.add` with a sparse or masked increment tensor.
3. Callers that currently pass `cache_position` as a `ttnn.Tensor` should instead keep their position tracking as the persistent device tensor and pass it directly, bypassing the `ttnn.to_torch` + `ttnn.from_torch` round-trip entirely.

This optimization eliminates the `ttnn.from_torch` call at lines 2678–2685 from the hot path and, when `cache_position` arrives as a device tensor, also eliminates the `ttnn.to_torch` call at line 2674. The net savings is one full PCIe round-trip per decode step (Path 2) or one upload per step (Path 1), replaced by a single on-device `ttnn.add` that is pipelined with other compute.

---

**Next:** [Chapter 4 — `_to_replicated` Round-Trip Analysis](to_replicated_analysis.md)
