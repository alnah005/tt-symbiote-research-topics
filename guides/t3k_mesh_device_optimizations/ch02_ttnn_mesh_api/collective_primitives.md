# All-to-All, All-Reduce, and Other Collective Operations

> **Quick Reference — TTNN API Symbols Introduced in This File**
>
> | Symbol | Module | Description |
> |---|---|---|
> | `ttnn.all_to_all` | `ttnn` | Routes distinct data from each device to every other device; the primary communication primitive for MoE expert dispatch and combine. |
> | `ttnn.all_reduce` | `ttnn` | Reduces (sums or otherwise combines) a tensor across all devices and writes the result to every device; used for weight gradient synchronization and shared-activation reduction. |
> | `ttnn.reduce_scatter` | `ttnn` | Reduces across all devices and distributes distinct output slices to each device; used when downstream operations can consume a sharded result directly. |
> | `ttnn.all_gather` | `ttnn` | Concatenates per-device tensor slices into a single full tensor replicated on every device; used to reconstruct a sharded tensor before an operation that requires the full value. |

Collective operations are the primitives that move data between devices across the T3K Ethernet links. Each of the four operations documented here has a distinct communication pattern and a distinct set of use cases. Understanding when to use each is as important as knowing their signatures.

This file documents the API signatures, key parameters, use cases, synchronization semantics, and error handling for each primitive. The topological analysis that motivates the linear-chain implementation of these primitives is in Chapter 1; the quantitative `num_links` tuning analysis is in Chapter 3.

---

## `ttnn.all_to_all`

### Purpose

`ttnn.all_to_all` implements the all-to-all collective: each device sends a distinct data slice to each other device and receives a distinct data slice from each other device. After the operation completes, device N holds N slices total — one from each device — including the slices sent to it by the other N-1 devices over Ethernet links and its own self-addressed slice retained locally. The output buffer size per device is therefore N × slice_size (7 cross-device + 1 self-addressed).

This is the dominant communication primitive for Mixture-of-Experts (MoE) inference. During the dispatch phase of each MoE layer, activated-token embeddings are routed to the devices holding the selected experts; each device sends tokens to N-1=7 other devices over Ethernet links and retains its own slice locally, for a total of eight output slots (seven cross-device transfers + one local). It likewise receives tokens from seven other devices plus its own retained slice. The combine phase is a second all-to-all that routes expert outputs back to their originating devices.

### Signature

```python
output = ttnn.all_to_all(
    input_tensor,
    cluster_axis,
    mesh_device,
    num_links=1,
    memory_config=None,
    num_workers=None,
    num_buffers_per_channel=None,
    topology=ttnn.Topology.Linear,
)
```

### Key Parameters

**`input_tensor`**

The input tensor must be a sharded mesh tensor resident on the `mesh_device`. Its logical shape along `cluster_axis` must be consistent with the number of devices on that axis. For a `(1, 8)` mesh with `cluster_axis=1`, the tensor must have eight slices along the column axis of the mesh — one slice per device.

For MoE dispatch, the input tensor has shape `(num_devices, tokens_per_device, hidden_dim)` where the leading `num_devices` dimension identifies the destination device for each group of tokens. This pre-grouped layout is required because the all-to-all implementation on T3K does not perform routing by content; it performs routing by position — the slice at position N in the send buffer goes to device N. You must sort your tokens into the correct position (by their target device) before calling `ttnn.all_to_all`.

**`cluster_axis`**

An integer selecting which axis of the logical mesh the collective traverses. For a `(1, 8)` T3K mesh, `cluster_axis=1` traverses the column axis — the axis along which the eight devices are arranged. This is the correct value for all single-board T3K all-to-all calls.

`cluster_axis=0` would traverse the row axis; for a `(1, 8)` mesh that axis has only one device, so a `cluster_axis=0` all-to-all on a single T3K board does nothing useful. `cluster_axis=0` becomes meaningful in multi-board deployments with a `(2, 8)` or `(4, 8)` mesh.

**`mesh_device`**

The `MeshDevice` instance created by `ttnn.open_mesh_device`. This argument is required and must match the mesh on which `input_tensor` is resident.

**`num_links`**

An integer in the range 1–4 controlling how many of the available Ethernet links between adjacent device pairs are allocated to this collective operation. Higher values increase bandwidth proportionally for large tensors; for small tensors the coordination overhead of additional links may outweigh the bandwidth benefit.

The choice of `num_links` is the primary tuning knob for all-to-all performance on T3K. A full analysis of how to select `num_links` for prefill vs. decode workloads is in Chapter 3. The brief guidance here is:

| Phase | Tensor size | Recommended starting value |
|---|---|---|
| prefill | Large (many tokens × hidden_dim) | `num_links=4` (maximum) |
| decode | Small (1–32 tokens × hidden_dim) | `num_links=1` or `num_links=2`; benchmark to confirm |

**`memory_config`**

A `ttnn.MemoryConfig` specifying where the output tensor is placed on each device. Passing `None` uses the same memory configuration as the input tensor. For performance-sensitive paths, pass an explicit `memory_config` to control whether the output lands in L1 or DRAM. Chapter 4 provides detailed guidance on this choice; the general rule is:

- Decode phase: use L1 if the output fits (the subsequent expert matmul can consume directly from L1, eliminating a DRAM round-trip).
- Prefill phase: use DRAM (output tensors are too large for L1 at typical sequence lengths).

**`topology`**

Controls the network traversal algorithm. `ttnn.Topology.Linear` uses a linear-chain (open-path) algorithm appropriate for T3K's linear mesh: devices are connected in a path from device 0 through device 7, with no wrap-around edge between device 7 and device 0. `ttnn.Topology.Ring` assumes a closed ring topology with a wrap-around edge between device 7 and device 0, and should not be used on a single T3K board where that edge does not exist. Use `ttnn.Topology.Linear` for all single-board T3K calls.

### Example: MoE Dispatch All-to-All

```python
# Pre-condition: tokens have been sorted into groups by target device.
# dispatch_buffer has shape (num_devices=8, tokens_per_device, hidden_dim=4096).
# dispatch_buffer is a sharded mesh tensor on mesh_device.

dispatched = ttnn.all_to_all(
    dispatch_buffer,
    cluster_axis=1,
    mesh_device=mesh_device,
    num_links=2,                          # decode phase; benchmark to confirm
    memory_config=ttnn.L1_MEMORY_CONFIG,  # output directly into L1 for expert compute
    topology=ttnn.Topology.Linear,
)
# After this call, device N holds the tokens that were destined for it from all 8 sources.
```

---

## `ttnn.all_reduce`

### Purpose

`ttnn.all_reduce` reduces a tensor across all devices — summing (or applying another reduction operator) the corresponding elements from each device's copy — and writes the identical full result to every device. After the operation, every device holds the same tensor whose values are the element-wise sum of the input tensors from all devices.

All-reduce is distinct from all-to-all: all-reduce produces a single result replicated everywhere (all devices end up with the same data), while all-to-all produces device-specific results (each device ends up with different data).

### Signature

```python
output = ttnn.all_reduce(
    input_tensor,
    cluster_axis,
    mesh_device,
    num_links=1,
    memory_config=None,
    reduce_op=ttnn.ReduceType.Sum,
    topology=ttnn.Topology.Linear,
)
```

### Use Cases in MoE Context

**Partial-product all-reduce for row-wise sharded weights.** When an expert weight is sharded row-wise across devices (each device holds a different input-dimension slice), each device's matmul produces a partial output that must be summed across all devices to produce the correct full output. `ttnn.all_reduce` performs this summation and broadcasts the result to all devices.

```python
# Each device computed a partial matmul output: shape (batch, seq_len, output_dim).
# partial_output holds partial matmul results (different on each device): every device holds
# a partial sum of shape (batch, seq_len, output_dim), but the values differ — each is a
# 1/N contribution to the full output that must be summed across all devices via all-reduce.

full_output = ttnn.all_reduce(
    partial_output,
    cluster_axis=1,
    mesh_device=mesh_device,
    num_links=4,                            # prefill phase, large tensor
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    reduce_op=ttnn.ReduceType.Sum,
    topology=ttnn.Topology.Linear,
)
# full_output is identical on all 8 devices.
```

**Attention output aggregation.** In tensor-parallel attention, each device computes a partial attention output (its head slice attends to its value slice). The partial outputs must be summed across devices before feeding into the MLP or the next layer. This is an all-reduce over the head-parallel dimension.

**What all-reduce is NOT used for in MoE.** Token routing — sending different tokens to different expert devices — is not an all-reduce operation because the result is not the same on all devices. That is the all-to-all pattern. Using all-reduce for token routing would mean every device receives and processes all tokens from all devices, which eliminates the sparsity benefit of MoE. Use `ttnn.all_to_all` for expert dispatch and combine; use `ttnn.all_reduce` only for operations where every device genuinely needs the same reduced value.

### Comparison with All-to-All

| Property | `ttnn.all_reduce` | `ttnn.all_to_all` |
|---|---|---|
| Result on each device | Identical (same reduced tensor) | Distinct (device-specific data) |
| Primary use in MoE | Partial-product sums, gradient sync | Expert dispatch and combine |
| Communication volume (N devices) | 2×(N−1)/N × input_size (~1.75× for N=8) | (N−1) × slice_size per device |
| Output size relative to input | Same size (reduction, not expansion) | Same total size (permutation) |
| Communication step count | 8 steps total (4 reduce-scatter + 4 all-gather for N=8 bidirectional) | ⌈(N−1)/2⌉ = 4 rounds (bidirectional) |

---

## `ttnn.reduce_scatter`

### Purpose

`ttnn.reduce_scatter` combines a reduce and a scatter into a single collective: it reduces the input tensors across all devices (as all-reduce does) but instead of replicating the full result everywhere, it distributes distinct non-overlapping slices of the result to each device. Device N receives only the N-th slice of the reduced result.

`ttnn.reduce_scatter` is equivalent to `ttnn.all_reduce` followed by sharding the output, but it avoids materializing the full reduced tensor on any single device and reduces the inter-device communication volume compared to a full all-reduce (each device receives 1/N of the full result instead of the full result).

### Signature

```python
output = ttnn.reduce_scatter(
    input_tensor,
    scatter_dim,
    cluster_axis,
    mesh_device,
    num_links=1,
    memory_config=None,
    reduce_op=ttnn.ReduceType.Sum,
    topology=ttnn.Topology.Linear,
)
```

The additional parameter `scatter_dim` specifies which dimension of the reduced tensor is split across devices in the output.

### When to Use in MoE Context

`ttnn.reduce_scatter` is appropriate when:

1. You need the reduced result but the downstream operation can consume a sharded tensor (it does not need the full result on every device).
2. Communication bandwidth is the bottleneck and you want to minimize the data each device receives.

In MoE inference, `ttnn.reduce_scatter` is most commonly used when expert down-projection weights are sharded row-wise. Each device's matmul produces a partial output that must be summed; if the next layer (for example, the next all-to-all dispatch or a residual add) can operate on a sharded activation, reduce-scatter avoids materializing the full summed activation on every device.

The tradeoff: `ttnn.reduce_scatter` produces a sharded output that may require an eventual `ttnn.all_gather` to reconstruct the full tensor before operations that require the complete data. Whether the bandwidth saving from reduce-scatter is larger than the bandwidth cost of the subsequent all-gather depends on how many operations can proceed on the sharded intermediate before needing the full tensor.

---

## `ttnn.all_gather`

### Purpose

`ttnn.all_gather` is the inverse of sharding: it takes per-device tensor slices and concatenates them into a single full tensor replicated on every device. After the operation, every device holds a complete tensor assembled from all devices' contributions.

### Signature

```python
output = ttnn.all_gather(
    input_tensor,
    dim,
    cluster_axis,
    mesh_device,
    num_links=1,
    memory_config=None,
    topology=ttnn.Topology.Linear,
)
```

The `dim` parameter specifies which dimension of the output tensor the gather assembles. For column-wise sharded inputs (`ShardTensorToMesh` with `dim=1`), use `dim=1` to gather back along the column axis.

### When to Use in MoE Context

`ttnn.all_gather` is used:

- **After reduce-scatter**, when the full reduced tensor is needed for a downstream operation that does not support sharded input.
- **Before operations requiring full context**: for example, if the router needs to see the full activation `(batch, seq_len, hidden_dim)` before computing expert assignments, and the activation is currently sharded, an all-gather reconstructs the full activation on every device.
- **For tensor-parallel output reconstruction**: after column-wise sharded matmul, if the combined output is needed for a layer-norm or residual add that does not support sharded input.

`ttnn.all_gather` should be avoided when the downstream operation supports sharded input natively, since the gather incurs unnecessary communication bandwidth and output-side DRAM capacity.

### Relationship to Reduce-Scatter

`ttnn.reduce_scatter` followed by `ttnn.all_gather` is equivalent to `ttnn.all_reduce` in terms of final output (every device ends up with the full reduced result), and the final memory footprint is identical to all-reduce — every device ends up holding a tensor of the same size either way. The two-phase approach is only beneficial when the all-gather can be **deferred or eliminated entirely**: if one or more downstream operations can consume the sharded reduce-scatter output directly (without needing the full tensor on every device), the all-gather is skipped and you save the `(N-1)/N × data` bandwidth of the all-gather phase, cutting total communication roughly in half compared to all-reduce.

If the all-gather is always required immediately after reduce-scatter, the two-phase approach has the same total communication cost as a direct all-reduce and provides no memory advantage. In that case, `ttnn.all_reduce` is simpler and should be preferred. The split into two phases is useful only when (a) downstream ops accept sharded input so the all-gather can be skipped, or (b) you want to overlap the all-gather phase with independent computation on another queue.

---

## Synchronization Semantics

### Blocking vs. Async Dispatch

By default, all TTNN collective operations are dispatched asynchronously to the device command queues. Calling `ttnn.all_to_all` does not block the Python thread until the collective completes; it enqueues the operation and returns a tensor handle. The operation executes on the device command queue in the order it was enqueued, after all previously enqueued operations on that device complete.

From the Python caller's perspective, this means:

- You can enqueue multiple operations in sequence without waiting for each to finish.
- The device executes them in order (for a single-queue device), so dependencies are respected automatically as long as you pass the output tensor of one operation as the input of the next.
- You do not need explicit synchronization for simple sequential pipelines.

**When synchronization is needed.** You must explicitly synchronize in two situations:

1. **Reading results back to host.** Calling `ttnn.from_device` on a mesh tensor implicitly waits for all enqueued operations that produce that tensor to complete before transferring data to the host. You do not need to call `ttnn.synchronize_device` before `ttnn.from_device`.

2. **Two-queue overlapped execution.** If you use `num_command_queues=2` (set in `ttnn.open_mesh_device`) to overlap compute on queue 0 with a collective on queue 1, you must use device-side events to express the dependency between the queues:

```python
# Enqueue the expert matmul on queue 0.
expert_output = ttnn.matmul(
    expert_input,
    expert_weight,
    device=mesh_device,
    queue_id=0,
)

# Enqueue the next-layer all-to-all dispatch on queue 1, overlapping with queue 0.
next_dispatch = ttnn.all_to_all(
    next_dispatch_buffer,
    cluster_axis=1,
    mesh_device=mesh_device,
    num_links=2,
    topology=ttnn.Topology.Linear,
    queue_id=1,  # Dispatch to queue 1 so all-to-all runs in parallel with queue 0 matmul
)

# Record a device-side event on queue 1 when the all-to-all finishes.
# This does NOT block the host thread.
event = ttnn.record_event(mesh_device, queue_id=1)

# Make queue 0 wait for that event before it proceeds past this point.
# This is a device-side dependency: queue 0 stalls on the device, not the host.
ttnn.wait_for_event(mesh_device, event, queue_id=0)

# Both queues are now synchronized at this point on the device side.
# The host can continue dispatching further operations to either queue.
```

Note: the exact API for selecting queue IDs per operation may differ between TTNN versions. Consult the current TTNN documentation for the `queue_id` parameter status.

### Event-Based Synchronization

For fine-grained synchronization between operations on different devices or different queues, TTNN provides device events. An event is a lightweight synchronization token that can be recorded after an operation completes on one queue and waited on before an operation starts on another.

```python
# Record an event after the collective completes on device 0's queue.
event = ttnn.record_event(mesh_device, queue_id=0)

# Wait for that event on queue 1 before dispatching the dependent operation.
ttnn.wait_for_event(mesh_device, event, queue_id=1)
```

Event-based synchronization is used in advanced pipelining scenarios (covered in Chapter 5) where the combine all-to-all of one MoE layer needs to complete before the dispatch all-to-all of the next layer can use the link bandwidth without contention.

### Default Behavior Without Explicit Synchronization

For typical sequential inference pipelines (no two-queue overlap, no cross-device dependency management), you do not need to call `ttnn.synchronize_device` manually. The implicit dependency tracking through tensor handles ensures correct ordering. The only time you must explicitly synchronize is when you need the CPU to observe a result (triggering `ttnn.from_device` or printing a tensor value), or when you are orchestrating concurrent multi-queue execution.

---

## Error Handling

### Shape Mismatches

The most common error at collective dispatch time is a shape mismatch between the input tensor's layout and the collective operation's expectations.

**Symptom:** An error at the point of calling `ttnn.all_to_all` reporting that the input tensor's shard count does not match the number of devices on `cluster_axis`.

**Cause:** The input tensor was created with a sharding that does not align with the mesh axis the collective traverses. For example, creating a tensor sharded across 4 devices when the mesh has 8 devices on `cluster_axis=1`, then passing that tensor to an all-to-all with `cluster_axis=1`, produces a mismatch.

**Fix:** Verify that the number of shards in the input tensor matches `mesh_device.shape[cluster_axis]`. For T3K with `cluster_axis=1`, the input must have exactly 8 shards.

```python
# NOTE: This assertion assumes the input tensor is shaped [num_devices, ...] with
# devices in dimension 0. This check is layout-specific: it is only valid when
# the tensor has been pre-grouped so that dim 0 equals num_devices. For other
# sharded tensor layouts, use a more general check — for example, verify that the
# total element count is divisible by the number of devices:
#
#   num_devices = mesh_device.shape[1]
#   assert input_tensor.volume() % num_devices == 0
#
# Do NOT apply the dim-0 check below to arbitrarily sharded tensors; it will
# produce false positives and block correct code.
assert input_tensor.shape[0] == mesh_device.shape[1], (
    f"Expected {mesh_device.shape[1]} device-axis slices; "
    f"got {input_tensor.shape[0]}"
)
```

### Tile Alignment Errors

**Symptom:** An error during the all-to-all operation (not at construction time) reporting that the payload per link is not tile-aligned.

**Cause:** The tensor's hidden dimension or batch dimension is not a multiple of the TTNN tile size (32 elements) after sharding. For example, if `hidden_dim=4097` (not divisible by 32), the per-device shard after all-to-all has a fractional tile that the Ethernet DMA engine cannot handle.

**Fix:** Pad your tensor to the next multiple of 32 before the all-to-all, then unpad the output.

### Device Unreachable

**Symptom:** An error during collective dispatch reporting that one or more devices in the mesh did not respond, or that the Ethernet link is down.

**Cause:** A hardware fault (link failure, device reset, PCIe connection issue) has made a device in the mesh unreachable. Collective operations require all N devices to participate; if any device is unreachable, the collective cannot complete.

**Fix:** Perform a hardware check using the system management interface:

```bash
tt-smi --status
```

This reports the health of each device and the Ethernet link status between device pairs. If a device is in error state, a driver reset followed by `MeshDevice` re-initialization is required. There is no mechanism to continue inference with fewer than the configured number of devices; the `MeshDevice` must be closed and re-opened.

### Link Errors During Collective

**Symptom:** An error during a collective reporting a CRC failure or packet drop on an Ethernet link, or a collective that hangs without completing.

**Cause:** A transient or persistent error on one of the physical Ethernet links used by the collective. Transient errors (single-bit errors corrected by the link's error correction layer) are handled transparently by the link hardware. Persistent errors (repeated failures) escalate to the TTNN runtime and are reported as exceptions.

**Fix for transient errors:** Retry the operation. The TTNN runtime may retry automatically; if not, a single retry at the Python level is appropriate.

**Fix for persistent errors:** The link hardware may need to be re-initialized. A driver reset (via `tt-smi --reset`) re-trains all Ethernet links from scratch and resolves most persistent link errors that are not caused by hardware damage. If errors persist after reset, inspect the physical board for damage or cable issues.

### `num_links` Out of Range

**Symptom:** A ValueError at `ttnn.all_to_all` call time reporting that `num_links` is out of the valid range.

**Cause:** A `num_links` value less than 1 or greater than 4 was passed. The valid range on T3K is 1–4 inclusive.

**Fix:** Clamp `num_links` to the valid range before passing it to the collective:

```python
num_links = max(1, min(num_links, 4))
```

If you are unsure of the maximum `num_links` for your hardware, query it at runtime:

```python
max_links = mesh_device.get_num_ethernet_links_per_pair()  # returns 4 on T3K
```

---

## Summary: Choosing the Right Collective

| Question | Collective to use |
|---|---|
| Each device sends distinct data to a different destination device (token routing) | `ttnn.all_to_all` |
| All devices have partial sums that need to be totalled and all devices need the total | `ttnn.all_reduce` |
| All devices have partial sums; each device only needs its own slice of the total | `ttnn.reduce_scatter` |
| Tensor is sharded; the next operation needs the full tensor on every device | `ttnn.all_gather` |
| Reconstruct full tensor from reduce-scatter output for a subsequent non-sharded op | `ttnn.all_gather` (after `ttnn.reduce_scatter`) |

For MoE inference, the dominant operations are the two `ttnn.all_to_all` calls per MoE layer (dispatch and combine). The `ttnn.all_reduce` and `ttnn.reduce_scatter` / `ttnn.all_gather` primitives appear in tensor-parallel attention heads and in any weight-sharded MLP layers outside the MoE structure. Chapter 3 provides a detailed analysis of all-to-all performance and `num_links` tuning; Chapter 5 shows how all four primitives compose in a full MoE forward pass.
