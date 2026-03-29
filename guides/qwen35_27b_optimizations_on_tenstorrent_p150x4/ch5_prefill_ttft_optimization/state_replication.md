# State Replication: B=1 to B=32 Post-Prefill

After prefill completes, the model holds B=1 states representing a single user's processed prompt. Decode, however, operates with B=32 -- all 32 batch slots process tokens simultaneously using traced execution. Before the first decode step can run, the prefilled user's states must be replicated to all 32 batch slots. This affects both GDN recurrence states and attention KV caches.

The replication is a one-time cost that bridges the prefill and decode phases. It runs on the host (CPU-side via PyTorch), moving data through `ttnn.to_torch` and `ttnn.from_torch`. While this introduces a host-device round-trip, it happens only once per prefill and is not on the critical path for per-token latency.

## GDN State Replication

`TtGatedDeltaNet.replicate_prefill_state_to_batch()` (`gdn.py`, lines 522-576) handles three categories of GDN state:

### Conv States

The conv1d shift register has 4 slots, each holding `[1, 1, qkv_dim_tp]` per device after prefill. These must expand to `[1, B, qkv_dim_tp]` for B=32 decode:

```python
for i in range(self.conv_kernel_size):         # 4 slots
    per_dev = ttnn.get_device_tensors(self._prefill_conv_states[i])
    batched_parts = []
    for dev_t in per_dev:
        cpu = ttnn.to_torch(dev_t)              # [1, 1, qkv_dim_tp]
        batched_parts.append(cpu.expand(1, B, -1).contiguous())  # [1, 32, qkv_dim_tp]
    combined = torch.cat(batched_parts, dim=0)  # [num_devices, 32, qkv_dim_tp]
    new_state = ttnn.from_torch(
        combined, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    ttnn.copy(new_state, self.conv_states[i])   # Write into decode state buffers
    ttnn.deallocate(new_state)
```

The pattern is: read each device's B=1 tensor to CPU, expand along the batch dimension using `torch.expand` + `contiguous()`, concatenate all devices along dim=0, then shard back to the mesh with `ShardTensorToMesh(mesh, dim=0)`. The final `ttnn.copy` writes into the pre-existing decode conv state buffers (allocated during `reset_state()`).

### Recurrence States

The recurrence state is the primary GDN state tensor. After prefill it holds `[Nv_TP, Dk, Dv] = [12, 128, 128]` per device (approximately 393 KB per device in bfloat16). For decode, this must become `[B*Nv_TP, Dk, Dv] = [384, 128, 128]` per device (12 MB):

```python
per_dev_rec = ttnn.get_device_tensors(self._prefill_rec_states)
batched_rec_parts = []
for dev_t in per_dev_rec:
    cpu = ttnn.to_torch(dev_t)                # [Nv_TP, Dk, Dv] = [12, 128, 128]
    batched_rec_parts.append(cpu.repeat(B, 1, 1))  # [B*Nv_TP, Dk, Dv] = [384, 128, 128]
combined_rec = torch.cat(batched_rec_parts, dim=0)  # [num_devices*B*Nv_TP, Dk, Dv]
new_rec = ttnn.from_torch(
    combined_rec, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    device=mesh, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
)
ttnn.copy(new_rec, self.rec_states)
ttnn.deallocate(new_rec)
```

Note the use of `torch.repeat(B, 1, 1)` rather than `expand` -- `repeat` is required here because the first dimension is `Nv_TP` (not 1), and the replication pattern is to repeat each head's state B times, producing `[B*Nv_TP, Dk, Dv]` where batch slot `b`'s head `h` is at index `b * Nv_TP + h`.

### Cleanup

After replication, all B=1 prefill state tensors are deallocated:

```python
for s in self._prefill_conv_states:
    ttnn.deallocate(s)
ttnn.deallocate(self._prefill_rec_states)
ttnn.deallocate(self._prefill_fused_output)
self._prefill_conv_states = None
self._prefill_rec_states = None
self._prefill_fused_output = None
```

This frees approximately 393 KB per device of recurrence state, plus the conv state and output buffers. These are no longer needed once decode begins.

## KV Cache Replication

`Qwen35Attention.replicate_kv_cache_to_batch()` (`attention.py`, lines 250-293) performs the analogous operation for full attention layers. Each attention layer maintains per-head K and V caches of shape `[B, 1, max_seq_len, HD]` per device. After prefill, only user 0 (batch slot 0) contains valid data.

```python
for h in range(self.n_local_kv_heads):         # NKV heads per device
    k_per_device = ttnn.get_device_tensors(self.k_caches[h])
    v_per_device = ttnn.get_device_tensors(self.v_caches[h])

    k_torch_per_dev = []
    v_torch_per_dev = []
    for k_dev, v_dev in zip(k_per_device, v_per_device):
        k_cpu = ttnn.to_torch(k_dev)           # [B, 1, max_seq_len, HD]
        v_cpu = ttnn.to_torch(v_dev)
        k_user0 = k_cpu[0:1]                   # [1, 1, max_seq_len, HD]
        v_user0 = v_cpu[0:1]
        k_torch_per_dev.append(k_user0.expand(B, -1, -1, -1).contiguous())
        v_torch_per_dev.append(v_user0.expand(B, -1, -1, -1).contiguous())

    self.k_caches[h] = ttnn.from_torch(
        torch.cat(k_torch_per_dev, dim=0),     # [num_devices*B, 1, max_seq_len, HD]
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    self.v_caches[h] = ttnn.from_torch(
        torch.cat(v_torch_per_dev, dim=0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
```

Unlike the GDN replication (which uses `ttnn.copy` into pre-existing buffers), the KV cache replication replaces `self.k_caches[h]` and `self.v_caches[h]` entirely with new tensors via `ttnn.from_torch`. This is because the KV cache tensors are freshly created with the full `max_seq_len` dimension and the old tensors are implicitly freed when the references are overwritten.

The KV cache replication loops over `n_local_kv_heads` (the number of KV heads on this device after TP sharding). With `NKV=4` heads and `TP=4`, each device holds `NKV_TP=1` head, so this loop executes once per device per attention layer.

## Replication Order and Decode Readiness

The full post-prefill sequence, called from the model driver, is:

1. For each of the 48 GDN layers: `gdn_layer.replicate_prefill_state_to_batch()`
2. For each of the 16 attention layers: `attn_layer.replicate_kv_cache_to_batch()`
3. Decode begins with `B=32` and traced execution

After step 2, all 64 layers hold valid B=32 state, and the model can immediately enter the traced decode loop where every batch slot generates tokens independently. The replication cost is amortized over the entire generation sequence -- for a 512-token generation, the one-time replication is negligible compared to the 512 decode steps that follow.

---

**Previous:** [`gdn_prefill_strategy.md`](./gdn_prefill_strategy.md) | **Next:** [Chapter 6 — L1 State Management and Rolling Window](../ch6_l1_state_management/index.md)
