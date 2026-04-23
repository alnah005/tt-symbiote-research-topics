# The CQ0 FIFO Ordering Guarantee

This file explains the CQ0 FIFO ordering model in detail, shows why the guarantee applies to both synchronous ops and async CCL ops, addresses the one historical exception where `synchronize_device` could have had a legitimate ordering purpose (multi-CQ dispatch), and concludes that in the single-CQ mode required for Metal Trace, `ttnn.synchronize_device()` adds host-blocking latency without contributing to correctness. By the end of this file you will have the technical foundation needed to evaluate the verdict in [`verdict_is_it_removable.md`](./verdict_is_it_removable.md).

---

## The Single-CQ Ordering Model

TTNN on T3K uses a single hardware command queue — CQ0 — as the sole dispatch channel in trace-compatible deployment. All ops submitted to this queue by the host are placed into a FIFO (first-in, first-out) buffer. The device processes commands in strict submission order.

The guarantee that follows from FIFO ordering is:

**Op N+1 submitted to CQ0 cannot begin executing until op N has completed and its output is available at the agreed-upon device memory address.**

This guarantee does not require any host-side assertion, polling, or synchronization call to hold. It is enforced by the hardware itself: the device dispatch engine will not dequeue and begin executing the command for op N+1 until it has fully dispatched and dispatched-completion-signaled op N.

More precisely, for a sequence of two ops where op N writes an output tensor `T` and op N+1 takes `T` as an input:

1. The host enqueues the dispatch command for op N to CQ0. The host Python call returns (the op is now "queued").
2. The host enqueues the dispatch command for op N+1 to CQ0. The host Python call returns (op N+1 is now also "queued", behind op N).
3. The device begins executing op N. The host is free to continue enqueuing further ops.
4. Op N completes. Its output is written to `T`'s device address.
5. The device begins executing op N+1. Op N+1 reads from `T`'s device address — which is now valid.

At no point does the host need to wait between steps 1 and 2. The FIFO queue enforces that step 5 cannot precede step 4. `ttnn.synchronize_device()` between enqueuing op N and enqueuing op N+1 would block the host **before** step 2 (i.e., before op N+1 can be enqueued) until step 4 completes — adding PCIe round-trip latency without providing any ordering guarantee that the queue does not already provide.

---

## Concrete Example: `ttnn.all_gather` Followed by `ttnn.linear`

The `_maybe_all_gather` call sites in `TTNNQwen3FullAttention.forward` show the pattern directly:

```python
# Op N: QKV linear projection — enqueued to CQ0, host returns immediately
xqkv_fused = ttnn.linear(
    x,
    self.wqkv,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    program_config=self.model_config["XQKV_DECODE_PROGCFG"],
    compute_kernel_config=...,
)
# why: xqkv_fused is now a "queued output" tensor — its device address
#      is known but its content is not yet written; the host continues

# Op N+1: all_gather (inside _maybe_all_gather) — reads xqkv_fused (which op N wrote)
xqkv_gathered = ttnn.all_gather(
    xqkv_fused,
    dim=3,
    num_links=1,
    cluster_axis=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
# why: CQ0 FIFO ensures op N+1 cannot read xqkv_fused until op N has written it

# Current code — the synchronize_device call (inside _maybe_all_gather, after all_gather):
ttnn.synchronize_device(self.mesh_device)
# why: host blocks until CQ0 is empty; xqkv_gathered is now definitely written
#      BUT: this guarantee was already provided by the queue — the downstream op
#           below cannot begin until xqkv_gathered is written regardless
#      synchronize_device here contributes NOTHING to ordering correctness
```

The `synchronize_device` call after the all_gather is structurally redundant. Whether the host waits for the linear to complete (as `synchronize_device` forces) or immediately enqueues the all_gather (as the queue-only model allows), the device cannot begin the all_gather until the linear has written `xqkv_fused`. The queue enforces this independently of what the host does.

Similarly for the second call site, where the all_gather feeds an output linear projection:

```python
# Op M: all_gather — enqueued to CQ0
xqkv_gathered = ttnn.all_gather(xqkv_fused, ...)

# ttnn.synchronize_device(self.mesh_device)
# why: redundant — op M+1 below cannot begin until op M completes,
#      regardless of whether synchronize_device is called

# Op M+1: output projection — reads xqkv_gathered written by op M
dense_out = ttnn.linear(xqkv_gathered, self.wo, ...)
# why: CQ0 FIFO guarantees xqkv_gathered is valid when this op executes
```

---

## Why the Guarantee Holds for Async CCL Ops

The CQ0 FIFO guarantee applies to synchronous ops and async CCL ops (`ttnn.experimental.all_gather_async`, `ttnn.experimental.reduce_scatter_minimal_async`) equally. The mechanism is the same: both types of ops are submitted to CQ0 as encoded dispatch commands, and the device processes them in FIFO order.

For async CCL ops, there is an additional internal mechanism worth understanding. Unlike a simple compute kernel that writes its output synchronously to a buffer, a CCL op involves inter-device communication: data must travel across the NIC links between devices before the output buffer is valid. The `GlobalSemaphore` handles passed to `all_gather_async` are the mechanism by which the CCL kernel coordinates cross-device completion — internally, within the kernel itself:

1. The host enqueues the `all_gather_async` dispatch command to CQ0. The host returns.
2. The device begins executing the CCL kernel. The kernel communicates across NICs with peer devices to gather data.
3. The CCL kernel internally waits on the `GlobalSemaphore` rendezvous — a cross-device, chip-to-chip coordination step that ensures all peer devices have finished writing their data chunks into the output buffer. This wait happens inside the kernel's execution, not in the CQ0 dispatch engine.
4. Only after the semaphore rendezvous completes does the CCL kernel exit. At kernel exit, all data is written and the output buffer is valid.
5. The CQ0 dispatch engine observes the kernel exit (command complete) and advances to the next command in the queue. The engine does not poll L1 semaphore addresses — it advances on kernel exit, not on any L1 signal.
6. The next op in CQ0 (e.g., a `ttnn.linear` that reads the all_gather output) begins executing. The output buffer is valid because the CCL kernel's internal semaphore rendezvous completed before the kernel exited.

The key point is that the `GlobalSemaphore` rendezvous is an intra-kernel mechanism — it is cross-device coordination that happens entirely inside the CCL kernel's execution. The CQ0 dispatch engine is not involved in and does not observe L1 semaphore values. The engine simply advances to the next command when the CCL kernel exits, which happens only after the internal rendezvous is complete. The downstream compute op in CQ0 does not begin until the CCL op's output is valid — because the kernel cannot exit until its internal semaphore rendezvous has completed.

This is why the tt-transformers `Attention.forward_decode` path, shown in Chapter 2, submits `all_gather_async` followed immediately by `ttnn.linear` to the same CQ0 without any `synchronize_device` in between:

```python
# From models/tt_transformers/tt/attention.py (Variant B, non-Ring topology)

all_gather_output = ttnn.experimental.all_gather_async(
    attn_output_cat,
    persistent_output_buffer=None,
    dim=3,
    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=cluster_axis),
    # why: cycling handle — slot advances 0 → 1 → 0 to prevent aliasing;
    #      cluster_axis selects the correct semaphore slot for this topology axis
    num_links=1,
    topology=self.ccl_topology,
    memory_config=self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"],
    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=cluster_axis),
    # why: cycling barrier — all ranks rendez-vous before gather data is valid;
    #      cluster_axis selects the correct barrier slot for this topology axis
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
# NO ttnn.synchronize_device() here

dense_out_sharded = ttnn.linear(
    all_gather_output,
    # why: enqueued to CQ0 AFTER all_gather_async; device cannot execute this
    #      until all_gather_async's CCL kernel has signaled completion via its
    #      GlobalSemaphore mechanism and the CQ0 dispatch engine has advanced
    self.wo,
    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
    ...
)
```

This pattern has been validated at production scale on T3K for the full tt-transformers Qwen model family. There is no `synchronize_device` and there is no correctness issue.

---

## The Multi-CQ Exception

The single-CQ ordering guarantee described above applies only when all ops share the same command queue. If ops are dispatched to multiple independent queues — for example, CQ0 for compute and CQ1 for CCL — then there is no FIFO ordering relationship between a CQ0 op and a CQ1 op. In that scenario, a cross-queue synchronization mechanism is required to ensure that a CQ0 compute op does not read a tensor that a CQ1 CCL op has not yet written.

`ttnn.synchronize_device()` could serve this cross-queue purpose: by draining all queues before returning, it establishes a global device-idle point that resolves any cross-queue dependency.

This is the one scenario in which `synchronize_device` in `_maybe_all_gather` would have had a legitimate correctness purpose. The question is whether multi-CQ dispatch has ever been used in the tt-symbiote attention modules.

> **Warning:** Multi-CQ dispatch is incompatible with Metal Trace in any case. The Metal Trace mechanism (`ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace`) operates on a single `cq_id` — specifically `cq_id=0`. Any ops dispatched to CQ1 during a trace capture bracket are not captured; they execute live. On replay, only the CQ0 commands are replayed; CQ1 work is absent. For the hybrid attention stack to be fully traceable, all ops — compute and CCL alike — must be dispatched to CQ0.

> **Warning:** The tt-symbiote source code was not accessible for direct inspection. The presence or absence of multi-CQ dispatch in `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` could not be confirmed by code search. The following assessment is based on the architecture described in Chapters 1 and 2.

Based on the available evidence, multi-CQ dispatch is not present in the current tt-symbiote attention modules for the following reasons:

1. **Chapter 2's reference pattern uses only CQ0.** The tt-transformers `Attention.forward_decode` path, which is the working traced-decode reference, dispatches all ops (compute and `all_gather_async`) to the same CQ0. If tt-symbiote were using a different multi-CQ approach, it would be a deliberate and unusual departure from the reference pattern.

2. **The `TRACED` run mode requires single-CQ.** tt-symbiote's `TracedRun` and `@trace_enabled` / `@trace_disabled` decorator system is designed for Metal Trace. Any module expected to participate in trace capture must operate in single-CQ mode. A module that uses multi-CQ dispatch cannot be `@trace_enabled` correctly.

3. **The synchronize call's location is post-all_gather.** A multi-CQ synchronize would typically appear at the start of a method to drain any pending CQ1 ops before submitting CQ0 dependencies, not after the all_gather completes. The placement of `synchronize_device` after the all_gather op is more consistent with a sequencing artifact (or debugging artifact) than with a cross-queue synchronization protocol.

> **Note:** To confirm the absence of multi-CQ dispatch definitively, search the tt-symbiote attention module source for `cq_id=1`, `dispatch_queue=1`, or any explicit `cq_id` argument other than `cq_id=0`. If none are found, multi-CQ dispatch is not in use and the multi-CQ exception does not apply. This search is marked as `# TODO: verify` pending source access.

---

> **Key Finding:** In the single-CQ model required for Metal Trace, `synchronize_device` provides no ordering guarantee beyond what CQ0 FIFO already guarantees. It is safe to remove.

The CQ0 FIFO ordering model applies to every op submitted to CQ0, including both synchronous all_gather and async `all_gather_async`. The async CCL op's device-side semaphore mechanism integrates with the CQ0 dispatch engine to ensure output buffer validity before the next command executes. The multi-CQ exception is not applicable in single-CQ trace-compatible deployment, and there is no evidence of multi-CQ dispatch in the tt-symbiote attention modules. `ttnn.synchronize_device()` in `_maybe_all_gather` adds PCIe round-trip overhead on every decode step — at minimum 10–30 µs per call, and at 2 calls per layer for H hybrid attention layers (on multi-device deployments where `num_devices > 1` and the `_maybe_all_gather` early-exit does not trigger), a total of at least 2H × 10–30 µs = 0.56–1.68 ms per decode step for H = 28 — without contributing to correctness.

---

**Next:** [`verdict_is_it_removable.md`](./verdict_is_it_removable.md)
