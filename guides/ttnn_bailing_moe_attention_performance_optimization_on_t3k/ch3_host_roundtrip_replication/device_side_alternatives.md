# Device-Side Alternatives to the Host Round-Trip

## Purpose

This file surveys TTNN primitives and patterns that could, in principle, convert the post-all-reduce QKV tensor from its sharded distribution type to a `ReplicateTensorToMesh` distribution without involving the host CPU or PCIe bus. For each candidate, the file analyses whether it is immediately actionable given the constraints of the `paged_sdpa_decode` kernel or whether it requires changes to the kernel, the TTNN runtime, or both. The file concludes with a concrete recommendation.

## Framing the Problem

The core constraint is:

1. After the fused QKV all-reduce, the tensor has shape `(1, 1, 1, 3072)` BF16, with numerically identical values on all 8 chips, but is tagged as a non-replicated multi-device tensor type.
2. `paged_sdpa_decode` requires its QKV input to have `ReplicateTensorToMesh` distribution — a hard constraint enforced at dispatch time.
3. TTNN has no "in-place reinterpret distribution" primitive that promotes a tensor's metadata tag from sharded to replicated without a data movement.
4. The existing workaround (`_to_replicated`) moves data through host DRAM, adding 7–25 µs of PCIe overhead [ESTIMATE].

A device-side solution must either:

- **Produce a tensor already in `ReplicateTensorToMesh` distribution** without going through the host, or
- **Modify `paged_sdpa_decode` to accept a sharded distribution** with a guarantee that shard values are identical.

## Alternative 1: `ttnn.all_gather` with `ReplicateTensorToMesh` Output Configuration

### Concept

Instead of performing the all-reduce with `ttnn.all_reduce` (which produces a non-replicated output), replace it with a two-step sequence:

1. Keep the partial WIDTH_SHARDED output from the fused matmul (each chip holds a `(1, 1, 1, 384)` shard).
2. Run `ttnn.all_gather` on this sharded tensor, configured to produce a `ReplicateTensorToMesh`-distributed output.

`ttnn.all_gather` is a CCL primitive that collects all per-chip shards and concatenates them. If configured to produce a `(1, 1, 1, 3072)` full tensor on every chip, and if the TTNN runtime tags the output as `ReplicateTensorToMesh`, the host round-trip would be eliminated.

### Why This Approach Is Numerically Incorrect as a Drop-In Replacement

**The current path uses `all_reduce` (sum-reduce of partial matmul outputs).** This is numerically correct because each chip computed a partial projection over a disjoint subset of weight columns (`X @ W_shard[i]`), and summing the partial results across chips yields the full projection. Replacing `all_reduce` with `all_gather` would instead concatenate the partial outputs — producing a `(1, 1, 1, 3072)` tensor where each 384-element block is the partial projection from one chip, not the full projection. **This is the wrong numerical result.**

`all_gather` is not a drop-in replacement for `all_reduce` in a column-parallel matmul. The two operations are only interchangeable if the matmul sharding strategy is also changed: from column-sharded weights (which require all-reduce) to row-sharded inputs with full weights (which require all-gather on the inputs). That is a more invasive restructuring.

**Verdict:** Alternative 1 as described is **not directly applicable** without restructuring the weight sharding strategy. It is presented here only for completeness; it is rejected as a solution.

### Implementation Sketch (Shown for Reference Only — Produces Incorrect Results)

```python
# ⚠️ WARNING: THIS SKETCH PRODUCES NUMERICALLY INCORRECT RESULTS.
# all_gather concatenates partial matmul outputs instead of summing them,
# yielding a wrong QKV projection. Do NOT use as a drop-in replacement
# for ttnn.all_reduce in the column-parallel matmul pattern.
#
# Replace ttnn.all_reduce + _to_replicated with:
qkv_replicated = ttnn.all_gather(
    qkv_partial,                          # (1, 1, 1, 384) WIDTH_SHARDED per chip
    dim=3,                                # gather along channel dimension
    num_links=num_links,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    # output_tensor_sharding_spec=ReplicateTensorToMesh(...) — see feasibility
)
```

If the sketch were numerically correct, every chip would hold `(1, 1, 1, 3072)` after the gather. The feasibility question of whether TTNN tags this output as `ReplicateTensorToMesh` is therefore moot given the correctness failure.

**Verdict:** Not immediately actionable in isolation. Requires weight sharding strategy change.

## Alternative 2: Device-Side Distribution Reinterpretation (Runtime Metadata Update)

### Concept

Add a TTNN runtime primitive — call it `ttnn.reinterpret_distribution` or `ttnn.mark_replicated` — that asserts, without moving data, that an existing multi-device tensor's per-chip sub-tensors are identical and should be treated as `ReplicateTensorToMesh`. The operation updates only the tensor's metadata descriptor; no PCIe traffic occurs.

This is the most direct solution to the root cause: the round-trip exists only because there is no way to relabel distribution metadata.

### Feasibility Analysis

**Runtime correctness guarantees:** TTNN's distribution type system exists to ensure correctness of downstream operations. An `all_reduce` output is tagged as non-replicated for safety: the runtime cannot verify at dispatch time that all chips hold identical values without expensive cross-chip comparison. A metadata-only relabeling operation would be safe only if the caller guarantees value identity — which is true here (the all-reduce guarantees it) but cannot be checked by the runtime automatically.

This primitive would need to:
1. Accept a multi-device tensor and a target distribution type (`ReplicateTensorToMesh`).
2. Assert (in debug builds) or trust (in release builds) that per-chip values are identical.
3. Return a new `ttnn.Tensor` with the same device-side storage but updated distribution metadata.

**Implementation scope:** The change is localised to the TTNN tensor descriptor and distribution API:

```
ttnn/cpp/ttnn/distributed/api.cpp           # Add reinterpret_distribution()
ttnn/python/ttnn/multi_device.py            # Expose Python binding
ttnn/cpp/ttnn/tensor/tensor.cpp             # Metadata update without data copy
```

No kernel changes are required. The function would be a pure C++ object manipulation with no device DRAM reads or writes.

**Verdict:** High feasibility, minimal implementation scope, and zero runtime data movement cost. This is the recommended approach if TTNN contributor access is available. The function signature would be approximately:

```python
qkv_replicated = ttnn.reinterpret_as_replicated(
    qkv_all_reduced,    # ttnn.Tensor with any distribution type
    mesh_device,        # MeshDevice to replicate across
)
```

The caller's contract is: all per-chip sub-tensors of `qkv_all_reduced` must be numerically identical. After an all-reduce, this is guaranteed by the CCL semantics.

## Alternative 3: Route All-Reduce Output Directly as Replicated

### Concept

Modify `TTNNLinearIColShardedWAllReduced` to configure its all-reduce output memory config such that TTNN's CCL layer natively produces a `ReplicateTensorToMesh`-typed tensor as the all-reduce result.

Some CCL implementations expose an `output_tensor_sharding_spec` or equivalent parameter that controls the output tensor's distribution type. If `ttnn.all_reduce` can be asked to produce a `ReplicateTensorToMesh` output directly — because after sum-reduce every chip holds an identical result — then no post-hoc conversion or host round-trip is needed.

### Feasibility Analysis

**CCL all-reduce output type:** The current `ttnn.all_reduce` implementation produces output tensors whose distribution type is tied to the reduce-scatter + all-gather decomposition. The all-gather phase of the all-reduce produces an identical value on all chips; if the TTNN CCL layer tagged its output as `ReplicateTensorToMesh` at this point, the problem would be solved at the CCL layer without any changes to the attention code.

Checking the relevant source:
```
ttnn/cpp/ttnn/operations/ccl/all_reduce/device/all_reduce_op.cpp
```
reveals whether the output tensor's `TensorMemoryLayout` and multi-device distribution tag are configurable or hard-coded. If hard-coded to a non-replicated type, adding an optional `output_distribution=ReplicateTensorToMesh` parameter to `ttnn.all_reduce` is a contained change.

**Required change scope:**
- `ttnn.all_reduce` Python interface: add `output_distribution` parameter.
- CCL op kernel: after the all-gather phase completes, set the output tensor's distribution tag to `ReplicateTensorToMesh` when the parameter is set.
- `TTNNLinearIColShardedWAllReduced`: pass `output_distribution=ReplicateTensorToMesh` to the all-reduce call.
- `TTNNBailingMoEAttention`: remove the `_to_replicated` call since the all-reduce already produces the correct distribution.

**Verdict:** Moderately actionable. Requires CCL op interface change (not just attention code), but the scope is well-defined and the change is semantically clean. This is likely the most architecturally correct fix because it eliminates the impedance mismatch at the source — the all-reduce is the last operation that changes data values, so it is the right place to declare the final distribution type.

## Alternative 4: Custom CCL Kernel that Combines All-Reduce and Replication Tagging

### Concept

Write a custom CCL op that combines the all-reduce sum reduction with a device-side "distribute to all chips as replicated" step, using Ethernet fabric bandwidth rather than PCIe. This is essentially Alternative 3 implemented as a new kernel rather than a parameter addition to an existing one.

### Feasibility Analysis

This is the highest implementation effort of the four alternatives and provides no benefit over Alternative 3 in terms of the final result. Custom CCL kernels on Wormhole require:

- Tensix compute kernel development in C++ targeting the Wormhole NOC and Ethernet subsystem
- Integration with TTNN's dispatch and memory management layer
- Extensive correctness testing for all mesh sizes and tensor shapes

Given that Alternative 3 achieves the same outcome (device-side replication, no host PCIe round-trip) through a parameter addition to an existing, well-tested CCL primitive, Alternative 4 is not recommended as a primary path.

**Verdict:** Not recommended. Alternative 3 is strictly preferable.

## Feasibility Summary

Table: Summary of device-side alternatives to the host round-trip

| Alternative | Data movement eliminated? | Implementation scope | Immediately actionable? |
|---|---|---|---|
| 1: `all_gather` replacing `all_reduce` | Yes | Large (weight sharding restructure) | No |
| 2: `ttnn.reinterpret_as_replicated` | Yes (zero-copy) | Small (metadata only, new API) | Yes with TTNN contribution |
| 3: `all_reduce` with `ReplicateTensorToMesh` output | Yes (device-side) | Medium (CCL interface + attention code) | Yes with TTNN contribution |
| 4: Custom CCL kernel | Yes | Very large | No |

## Recommended Path Forward

The recommended approach is **Alternative 3** as the primary target, with **Alternative 2** as a simpler interim mitigation if Alternative 3 requires longer review cycles.

### Alternative 3 as Primary Target

Adding `output_distribution=ReplicateTensorToMesh` to `ttnn.all_reduce` is the architecturally correct fix.

**Implementation checklist:**

```
[ ] Confirm that ttnn.all_reduce's output tensor descriptor is writeable after
    the all-gather phase completes (check all_reduce_op.cpp)
[ ] Add optional `output_mesh_distribution` parameter to ttnn.all_reduce
[ ] In TTNNLinearIColShardedWAllReduced, pass output_mesh_distribution=
    ReplicateTensorToMesh(mesh_device) to the all_reduce call
[ ] Remove _to_replicated call from TTNNBailingMoEAttention.forward()
[ ] Add a unit test: verify paged_sdpa_decode accepts the all_reduce output
    directly and produces numerically identical results to the round-trip path
[ ] Measure decode step latency before/after (see Chapter 7, tracy_profiling.md)
```

### Alternative 2 as Interim Mitigation

If CCL interface changes require a longer TTNN review cycle, `ttnn.reinterpret_as_replicated` (Alternative 2) can be implemented and merged independently. It is a smaller change, is purely metadata, and unblocks the attention code immediately. It can be replaced by Alternative 3 once that lands.

### What Requires Kernel Changes

Neither Alternative 2 nor Alternative 3 requires changes to the `paged_sdpa_decode` kernel itself. Both produce a tensor that already satisfies the kernel's `ReplicateTensorToMesh` input constraint. Kernel changes would only be needed if the goal were to modify `paged_sdpa_decode` to accept sharded inputs (an invasive change that is not recommended given the correctness risks).

---

**Next:** [Chapter 4 — Memory-Config Transitions](../ch4_memory_config_transitions/index.md)
