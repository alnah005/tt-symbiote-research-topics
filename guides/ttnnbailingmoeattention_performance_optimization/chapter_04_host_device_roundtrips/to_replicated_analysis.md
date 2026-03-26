# `_to_replicated`: Host Round-Trip Analysis and Bailing's Elimination

## What `_to_replicated` Does

`TTNNBailingMoEAttention._to_replicated` (lines 2288–2313 of `attention.py`) converts a multi-device tensor whose topology metadata was set by an all-gather operation into a tensor whose topology metadata is `ReplicateTensorToMesh`. The implementation performs a full device→host→device round-trip:

```python
def _to_replicated(self, tensor: ttnn.Tensor) -> ttnn.Tensor:
    # ...
    orig_shape = list(t.shape)
    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
    t_torch = ttnn.to_torch(t, mesh_composer=mesh_composer)   # device→host, sync stall
    t_torch = t_torch[: orig_shape[0]]                        # host: slice first logical slice
    return ttnn.from_torch(                                   # host→device, 8 uploads
        t_torch,
        device=self.device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        dtype=t.dtype,
        layout=t.layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
```

The docstring (lines 2289–2295) explains the motivation: after a `ttnn.all_gather`, the data on each device is identical (all devices hold the complete gathered tensor), but the TTNN topology metadata still records the original AllGather mesh mapping. Paged attention kernels inspect this metadata to determine which device's slice corresponds to the canonical tensor; an AllGather topology label on what is logically a replicated tensor causes incorrect device-slice mapping.

---

## Where `_to_replicated` Is Called — and Where It Is Not

### `TTNNBailingMoEAttention._forward_decode_paged`: NOT called

Searching `attention.py` lines 2610–2799 finds no call to `_to_replicated`. The method exists in the class (lines 2288–2313) but has been explicitly removed from the decode path. The comment at line 2642–2643 records why:

```python
# Reshape Q/K/V to decode format [1, batch, heads, head_dim] directly
# This avoids the concat + _to_replicated + nlp_create_qkv_heads_decode round-trip
```

Bailing eliminated the round-trip by constructing Q, K, and V directly in the `[1, B, H, D]` layout using `ttnn.reshape` (lines 2644–2646) rather than going through `nlp_create_qkv_heads_decode` + `_to_replicated`.

### `TTNNQwen3FullAttention._forward_decode_paged`: called for Q, KV\_key, KV\_value

Lines 765–768 of `qwen_attention.py`:

```python
if self.device.get_num_devices() > 1:
    query_states_paged = self._to_replicated(query_states_paged)
    kv_key = self._to_replicated(kv_key)
    kv_value = self._to_replicated(kv_value)
```

Three invocations per decode step. Each incurs a full device→host sync stall plus 8 PCIe reads and 8 PCIe writes.

### `TTNNGlm4MoeLiteAttention._forward_decode_paged`: called for Q, K, V

Lines 1894–1897 of `attention.py`:

```python
if self.device.get_num_devices() > 1:
    query_states = self._to_replicated(query_states)
    key_states = self._to_replicated(key_states)
    value_states = self._to_replicated(value_states)
```

Same pattern: three invocations, three sync stalls per decode step. The `_to_replicated` method in `TTNNGlm4MoeLiteAttention` (lines 1802–1826 of `attention.py`) is identical in implementation to the Bailing and Qwen3 versions.

---

## Why Paged Kernels Require `ReplicateTensorToMesh` Topology

The `paged_update_on_device` and `paged_sdpa_decode` kernels within `TTNNPagedAttentionKVCache` inspect the topology metadata of their input tensors. On a multi-device mesh, a tensor can carry one of several topology labels:

- **`ReplicateTensorToMesh`**: every device holds an identical full copy of the logical tensor. Each device independently reads or writes the full tensor; there is no device-specific slice.
- **AllGather topology** (the topology assigned by `ttnn.all_gather`): every device holds what was the full gathered result, but the metadata may record a shard-per-device structure from before the gather. The kernel's topology inspection may interpret this as "device 0 owns slice 0, device 1 owns slice 1, …" even though all slices are identical.

For `paged_update_on_device`, each device must write the new KV vectors into the correct slots of its local KV cache copy. If the kernel reads AllGather topology and interprets it as a shard assignment, it may write device 0's data from only the first `H/N` heads rather than all H heads, corrupting the cache layout.

The safe, explicitly-labeled topology is `ReplicateTensorToMesh`, which tells the kernel: this tensor is fully replicated; use all of it on every device.

---

## How Bailing Avoids `_to_replicated`

After `_maybe_all_gather` returns Q, K, and V (lines 2631–2633 of `attention.py`), each tensor has all data present on every device but carries AllGather topology. Bailing bypasses the topology conversion by calling `ttnn.reshape` directly:

```python
query_states = ttnn.reshape(query_states, (1, batch_size, self.num_heads, self.head_dim))
key_states   = ttnn.reshape(key_states,   (1, batch_size, self.num_kv_heads, self.head_dim))
value_states = ttnn.reshape(value_states, (1, batch_size, self.num_kv_heads, self.head_dim))
```

`ttnn.reshape` is a metadata-only op: it changes the logical shape but does not copy data or modify topology metadata. The resulting tensors still carry AllGather topology. These are then passed directly to `paged_update_on_device` (line 2756) and `paged_sdpa_decode` (line 2766).

**Verification note**: Whether `paged_update_on_device` and `paged_sdpa_decode` accept AllGather topology without correctness issues should be verified by comparing the KV cache contents and SDPA output against a reference implementation. The current Bailing decode path works in practice (tests pass), which suggests the paged kernels either ignore topology metadata when the data is provably identical on each device, or that `ttnn.reshape` performs an implicit topology normalization. This is worth confirming against the kernel implementation before relying on it in other models.

---

## Cost Savings from Eliminating `_to_replicated`

On the `TTNNQwen3FullAttention` and `TTNNGlm4MoeLiteAttention` paths, the three `_to_replicated` calls per decode step collectively cost (rough estimate):

| Item | Per Call | × 3 calls |
|---|---|---|
| Device sync flush | 200–500 µs | 600–1 500 µs |
| PCIe reads (8 devices × tensor size) | 5–30 µs | 15–90 µs |
| Host slice | <1 µs | <3 µs |
| PCIe writes (8 devices × tensor size) | 5–30 µs | 15–90 µs |
| Buffer alloc + metadata setup | ~10 µs | ~30 µs |
| **Total** | **~220–570 µs** | **~660–1 710 µs** |

This is a rough estimate. Actual figures require profiling. However, the scale suggests that the three `_to_replicated` calls in Qwen3 and GLM4 may consume 30–80% of the per-step attention budget at batch=1. Bailing's elimination of this pattern — by using `ttnn.reshape` directly from the all-gathered result — is the single most impactful optimization already in place relative to the other implementations.

---

## Adopting Bailing's Approach in Qwen3 and GLM4

The pattern to adopt:

1. After `_maybe_all_gather` (or equivalent), call `ttnn.reshape` to put Q, K, V into `[1, B, H, D]` format directly.
2. Remove the `if self.device.get_num_devices() > 1: _to_replicated(...)` block.
3. Pass the reshaped (AllGather-topology) tensors directly to `paged_update_on_device` and `paged_sdpa_decode`.
4. Verify correctness against reference outputs before removing the fallback.

---

**Next:** [Chapter 4 — `get_cos_sin_for_decode` Host Operations](get_cos_sin_host_ops.md)
