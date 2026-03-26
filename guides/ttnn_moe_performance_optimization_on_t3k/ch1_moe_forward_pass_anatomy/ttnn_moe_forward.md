# TTNNMoE.forward — Annotated Walkthrough

## Context

This file addresses:
- **Q1** — What is the per-op breakdown of a single MoE forward pass?
- **Q2** — Which CCL operations dominate end-to-end latency, and what parameters govern them?

Source range: `moe.py:L1346–L1496`

---

## Class Overview

`TTNNMoE` is the baseline MoE module for DeepSeek V3. It is a `TTNNModule` subclass decorated with `@run_on_devices(DeviceArch.T3K)`, which means its `forward` method is only dispatched on T3K meshes. The class owns:

- The gate weight (`_gate_weight_tt`) used for top-k routing.
- A reference to a `TTNNExperts` instance (`self.experts`) for the dispatch-compute-combine cycle.
- A reference to `self.shared_experts` for the residual path.
- `self.device_state.ccl_manager` — the object that allocates semaphores for CCL operations.

`TTNNBailingMoE` is a direct subclass that overrides configuration handling for the Bailing architecture but reuses `TTNNMoE.forward` unchanged. The inheritance means every statement below applies to both.

---

## Step 0: Per-Call Device Geometry (moe.py:L1358–L1361)

```python
self.num_devices = self.device.get_num_devices()
self.num_dispatch_devices = self.device.shape[0]
self.num_experts_per_device = even_int_div(self.config.n_routed_experts, self.num_devices)
residual = x
```

These three attributes are recomputed on every forward call rather than cached at init. `num_dispatch_devices` reads the first mesh dimension (`shape[0]`); the second dimension (`shape[1]`) is the reduce-scatter axis and is read later as `n_rs`. `residual` is a reference to the input tensor; it bypasses the MoE routing and feeds the shared experts branch at the end.

---

## Step 1: All-Gather to Revert Tensor Parallelism (moe.py:L1363–L1370)

```python
x = ttnn.experimental.all_gather_async(
    x,
    dim=-1,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    topology=ttnn.Topology.Linear,
)
```

The input tensor `x` arrives sharded along the hidden dimension because the preceding attention layer uses tensor parallelism. Before the gate linear can compute router logits over the full vocabulary of experts, every device must see the full hidden state. `all_gather_async` reassembles the hidden dimension in-place across all 8 devices.

Key parameters:

| Parameter | Value | Meaning |
|---|---|---|
| `dim` | `-1` | Gather along the last (hidden) dimension |
| `topology` | `ttnn.Topology.Linear` | Chain gather; each device forwards to the next along a linear chain (no wrap-around) |
| `num_links` | `1` | Single Ethernet link per hop |
| `multi_device_global_semaphore` | cycled from `ccl_manager` | Prevents semaphore reuse before previous op completes; `get_and_cycle_ag_semaphore_handles(1)` advances the manager's pointer each call to avoid handle starvation under pipelining; "async" means the op issues CCL work without blocking the host Python thread |
| `barrier_semaphore` | cycled from `ccl_manager` | Synchronizes all devices before the next op may begin; enforces completion before any subsequent op reads the output buffer |

---

## Step 2: Gate Linear for Router Logits (moe.py:L1372–L1393)

```python
if x.layout != ttnn.TILE_LAYOUT:
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
if x.dtype != ttnn.float32:
    x_f32 = ttnn.typecast(x, ttnn.float32)
else:
    x_f32 = x
router_logits_f32 = ttnn.linear(
    x_f32,
    self._gate_weight_tt,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    ),
)
if x_f32 is not x:
    ttnn.deallocate(x_f32)
router_logits = ttnn.typecast(router_logits_f32, ttnn.bfloat16)
ttnn.deallocate(router_logits_f32)
```

The gate linear produces router logits: a `[T, n_routed_experts]` tensor that maps each token to a score for every expert.

**Precision decisions:**

- The input is unconditionally upcast to `float32` via `ttnn.typecast` before the linear. Working activations in the model are typically `bfloat16`; the typecast adds a memory round-trip but is required for the `fp32_dest_acc_en=True` accumulator path.
- `math_fidelity=ttnn.MathFidelity.HiFi4` selects the highest-accuracy multiply-accumulate mode on Wormhole. HiFi4 uses full mantissa bits in the FP16 MAC, which matters for routing decisions where score differences can be small.
- `math_approx_mode=False` disables the fast-approximate transcendental path; not applicable to a linear but set defensively.
- After the linear, the logits are immediately downcast back to `bfloat16` and the `float32` intermediate is deallocated. This keeps DRAM pressure low while preserving routing accuracy.

**Reshape before routing (moe.py:L1395–L1396):**

```python
T = router_logits.shape[-2]
router_logits = ttnn.reshape(router_logits, ttnn.Shape((T, self.n_routed_experts)))
```

The linear output may carry a batch leading dimension; this reshape collapses it to a flat `[T, n_routed_experts]` view expected by `route_tokens_to_experts`.

**Top-k routing (moe.py:L1398–L1399):**

```python
topk_experts_indices, topk_experts_weights = self.route_tokens_to_experts(router_logits)
```

`route_tokens_to_experts` (not shown in the plan excerpt) returns two tensors: integer expert indices and floating-point weights for each of the `num_experts_per_tok` selected experts per token. These tensors flow directly into `TTNNExperts.forward`.

---

## Step 3: Expert Dispatch, Compute, Combine, Weight (moe.py:L1401–L1403)

```python
x = ttnn.unsqueeze(x, 1)
routed_output = self.experts(x, topk_experts_indices, topk_experts_weights)
```

`ttnn.unsqueeze(x, 1)` inserts a dimension at position 1, reshaping `x` from `[B, T, H]` to `[B, 1, T, H]`. This 4-D layout is required by `TTNNExperts.forward`, which expects `x.shape[0]` to be `batch_size_per_device` and `x.shape[2]` to be `seq_len`.

The entire dispatch-compute-combine-weight cycle is encapsulated in `self.experts`. See [`ttnn_experts_forward.md`](./ttnn_experts_forward.md) for the full walkthrough.

`routed_output` is a `TTNNWrapper` object (a Python container); its `.to_ttnn` property materializes the underlying `ttnn.Tensor` when needed.

---

## Step 4: Reduce-Scatter Final Output (moe.py:L1406–L1422)

```python
n_rs = self.device.shape[1]
routed_out = routed_output.to_ttnn
if n_rs > 1:
    routed_out = ttnn.mul(routed_out, 1.0 / float(n_rs))
routed_output = ttnn.experimental.reduce_scatter_minimal_async(
    routed_out,
    persistent_output_buffers=None,
    dim=3,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    cluster_axis=1,
    topology=ttnn.Topology.Ring,
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
```

After experts produce the full-width output, `reduce_scatter_minimal_async` both sums across the second mesh axis and re-shards the result along `dim=3` (the hidden dimension), restoring tensor-parallel layout for the next layer.

Key parameters and their significance:

| Parameter | Value | Meaning |
|---|---|---|
| `dim` | `3` | Scatter (output shard) dimension — hidden |
| `cluster_axis` | `1` | Reduce across the second mesh dimension |
| `topology` | `ttnn.Topology.Ring` | All-reduce ring; each device sends to its neighbor |
| `num_links` | `1` | Single Ethernet link per hop |
| `chunks_per_sync` | `10` | Number of payload chunks between synchronization points within the op |
| `num_workers_per_link` | `2` | Parallel Ethernet DMA workers per link |
| `num_buffers_per_channel` | `2` | Double-buffering for overlap of compute and communication |

**The pre-scatter normalization:** When `n_rs > 1`, the tensor is pre-multiplied by `1/n_rs` before the scatter. This is a mean-reduce pattern: the ring sum produces a sum, and the pre-multiplication converts it to a mean. This is mathematically equivalent to `reduce_mean` but allows the scalar multiply to run as a cheap fused op before the expensive CCL.

**Ring vs. Linear topology:** The all-gather at Step 1 uses `Linear` topology; the reduce-scatter uses `Ring`. Linear topology is optimal for gather operations on a chain of devices because latency grows as O(N) hops but bandwidth is fully utilized. Ring topology is standard for reduce-scatter because it achieves O(1) latency-per-chunk in pipeline-steady-state.

**`reduce_scatter_minimal_async`:** The "minimal" variant uses a stripped-down kernel that minimizes L1 usage compared to the standard reduce-scatter. `chunks_per_sync=10` controls the pipeline depth: higher values reduce synchronization overhead at the cost of larger in-flight buffer requirements.

---

## Step 5: Add Shared Experts Output (moe.py:L1424–L1428)

```python
shared_output = self.shared_experts(residual)
output = ttnn.add(routed_output, shared_output.to_ttnn)
output = ttnn.squeeze(output, 1)
return output
```

The shared experts run on the original unrouted residual (`x` before the all-gather modified it). Their output is added element-wise to the reduce-scattered routed output. `ttnn.squeeze(output, 1)` removes the dimension inserted by `unsqueeze` before the experts call, returning a tensor with the same rank as the input.

The shared experts are not subject to the routing decision and process every token with the same weights. They are a capacity mechanism that provides a minimum baseline transformation regardless of which routed experts were selected.

---

## Summary: Op Sequence in TTNNMoE.forward

```
all_gather_async          (Linear, num_links=1)
to_layout + typecast      (bf16 → f32)
ttnn.linear               (HiFi4, fp32 acc, DRAM output)
typecast + deallocate      (f32 → bf16)
reshape
route_tokens_to_experts
unsqueeze
TTNNExperts.forward       (see ttnn_experts_forward.md)
[optional] ttnn.mul       (1/n_rs normalization)
reduce_scatter_minimal_async  (Ring, chunks_per_sync=10, num_workers_per_link=2, num_links=1)
shared_experts(residual)
ttnn.add
ttnn.squeeze
```

---

**Next:** [`ttnn_experts_forward.md`](./ttnn_experts_forward.md)
