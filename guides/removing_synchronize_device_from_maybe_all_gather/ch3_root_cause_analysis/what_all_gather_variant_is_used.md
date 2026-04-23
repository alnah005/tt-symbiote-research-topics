# What all_gather Variant Does _maybe_all_gather Use?

This file determines which all_gather variant is called inside `_maybe_all_gather` in `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention`. The answer has direct bearing on the root cause analysis: the necessity — and the correct remedy — for `ttnn.synchronize_device()` depends entirely on whether the underlying gather is synchronous or async. By the end of this file you will know the exact call signature used, understand what `synchronize_device` was likely intended to guarantee in each case, and be able to assess whether that guarantee is actually necessary given CQ0 ordering semantics.

---

## Source Code Investigation

> **Warning:** The tt-symbiote repository was not accessible on the machine where this guide was authored (`/Users/salnahari/dev/tt-symbiote` does not exist). The source code of `_maybe_all_gather` could not be read directly. The analysis below is based on the architecture described in Chapters 1 and 2 and on the pattern established by the tt-transformers reference implementation. All code shown with `# TODO: verify` annotations must be confirmed against the actual source before acting on the implementation plan in Chapter 6.

> **Note:** To locate the source: glob for `**/*qwen3*attention*` or `**/qwen3_full_attention*` in the tt-symbiote repository, find `_maybe_all_gather` in `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention`, and check whether the method body calls `ttnn.all_gather` or `ttnn.experimental.all_gather_async`.

---

## The Two Possible Variants

### Variant 1 — Synchronous `ttnn.all_gather`

Based on the architecture established in Chapter 1 and the historical context of the tt-symbiote codebase, the most likely form of `_maybe_all_gather` is:

```python
def _maybe_all_gather(self, tensor, cluster_axis):
    # TODO: verify — this reconstruction is based on Chapter 1 analysis,
    #                not directly from source
    if self.num_devices == 1:
        return tensor                                    # why: no-op on single device; synchronize_device never called

    gathered = ttnn.all_gather(
        tensor,
        dim=3,                                           # why: gather along the hidden-dim axis (axis 3 for [1,1,B,D])
        num_links=1,                                     # why: single NIC link on T3K ring topology
        cluster_axis=cluster_axis,                       # why: T3K uses cluster_axis=1 for the 1×8 ring
        memory_config=ttnn.DRAM_MEMORY_CONFIG,           # why: TODO: verify memory config
    )

    ttnn.synchronize_device(self.mesh_device)            # why: host-blocking wait — the subject of this chapter

    return gathered
```

> **Warning:** The exact arguments — `dim`, `num_links`, `cluster_axis`, `memory_config` — require verification against the actual source. The reconstruction above reflects the pattern described in Chapter 1 ([`call_sites_and_control_flow.md`](../ch1_maybe_all_gather_anatomy/call_sites_and_control_flow.md)) and is consistent with the T3K deployment configuration (1×8 ring, `cluster_axis=1`).

### Variant 2 — Async `ttnn.experimental.all_gather_async`

A less likely but possible variant would already be using the async form without cycling semaphores:

```python
def _maybe_all_gather(self, tensor, cluster_axis):
    # TODO: verify — alternative reconstruction
    if self.num_devices == 1:
        return tensor

    gathered = ttnn.experimental.all_gather_async(
        tensor,
        persistent_output_buffer=None,
        dim=3,
        num_links=1,
        topology=ttnn.Topology.Ring,
        cluster_axis=cluster_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # NOTE: no multi_device_global_semaphore or barrier_semaphore arguments
        #       — this is what makes the synchronize_device call suspicious here;
        #       all_gather_async without cycling semaphores is functionally incorrect
    )

    ttnn.synchronize_device(self.mesh_device)            # why: blunt host-wait substituting for missing semaphore management

    return gathered
```

---

## Analysis: What Does `synchronize_device` Mean in Each Case?

### If the variant is synchronous `ttnn.all_gather`

As established in [`command_queue_ordering_guarantee.md`](./command_queue_ordering_guarantee.md), `ttnn.all_gather` enqueues the operation to CQ0 and returns immediately — the downstream op cannot execute until the all_gather's output is valid because CQ0 processes commands in FIFO order. `ttnn.synchronize_device` adds no ordering guarantee beyond what CQ0 already provides.

**Conclusion for Variant 1:** `ttnn.synchronize_device` after synchronous `ttnn.all_gather` provides no ordering guarantee beyond what CQ0 FIFO already provides. It is one of two things:

- **(a) A debugging artifact left in production code.** The synchronize call was inserted during development to force serial execution and confirm correctness. Once correctness was confirmed, the call was not removed. The model behaves correctly with or without it; removing it is safe.

- **(b) A conservative insertion to ensure stability before the CQ0 ordering guarantee was well understood.** Early in the development of the tt-symbiote tensor-parallel stack, the CQ0 ordering guarantee may not have been trusted or fully documented; `synchronize_device` was inserted as a belt-and-suspenders measure. Now that the guarantee is understood (as established in Chapter 1 and confirmed by the tt-transformers reference implementation in Chapter 2), the conservative measure can be removed.

In neither sub-case does `synchronize_device` provide a correctness guarantee that CQ0 cannot provide. It is strictly unnecessary.

### If the variant is async `ttnn.experimental.all_gather_async`

If `_maybe_all_gather` already calls `ttnn.experimental.all_gather_async` but without cycling semaphore handles, the situation is different. The async form requires the caller to manage the semaphore lifecycle: the CCL kernel uses `GlobalSemaphore` handles to signal completion and synchronize across devices. If no `multi_device_global_semaphore` or `barrier_semaphore` arguments are provided, or if the same handle is reused without cycling, the `synchronize_device` call may have been added as a blunt instrument to force device drain before the next call can re-use the semaphore.

But this is not the correct fix. The fundamental problem is that `synchronize_device` is a host-side local CQ0 drain — it guarantees the local device's CQ0 queue is empty, but it provides no cross-device ordering guarantee whatsoever. Without a `multi_device_global_semaphore` argument, the CCL kernel has no mechanism to signal or wait for cross-device completion: remote peer devices may not have finished writing their data chunks into the local output buffer when the host unblocks from `synchronize_device`. The host seeing an empty local CQ0 says nothing about what remote peers have done. As a result, the output buffer can be read before all peers have finished writing — a silent data corruption. An additional complication is that `synchronize_device` does not reset the `GlobalSemaphore` L1 values to their initial state, so stale completion signals can persist and corrupt subsequent calls. But this stale-L1 issue is secondary: the primary failure is the absence of any cross-device completion guarantee. The blunt host drain is the wrong mechanism for cross-device CCL completion, not merely an imperfect one.

**Conclusion for Variant 2:** `synchronize_device` after `all_gather_async` without cycling semaphores indicates a semaphore management deficiency. The correct replacement is not to keep `synchronize_device` but to adopt proper cycling semaphores from `TT_CCL`. Removing `synchronize_device` alone, without fixing semaphore management, would expose the underlying lifecycle bug. The correct fix is to wire `TT_CCL` and remove `synchronize_device` simultaneously.

---

## The Most Likely Variant

Given the Chapter 1 analysis (which describes `_maybe_all_gather` as calling an all_gather op followed by `ttnn.synchronize_device`), and given that Chapter 2 establishes `all_gather_async` as the pattern adopted in tt-transformers for traced decode, the most likely scenario is that `_maybe_all_gather` uses **synchronous `ttnn.all_gather`** — the older, simpler form — and that `synchronize_device` is a debugging or stability artifact from when the synchronous form was the only available option.

This assessment is supported by:

1. Chapter 1's description of `_maybe_all_gather` as calling "an all_gather op" with no mention of semaphore handles (which would be required for `all_gather_async`).
2. The fact that `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` do not hold a `TT_CCL` instance (as noted in [`../ch2_async_ccl_pattern/cycling_semaphore_mechanics.md`](../ch2_async_ccl_pattern/cycling_semaphore_mechanics.md)), which would be required for `all_gather_async`.
3. The pattern in the broader tt-symbiote codebase, where synchronous all_gather is the default and `all_gather_async` is an explicit upgrade path.

> **Warning:** This assessment is circumstantial and must be verified against the actual source code before proceeding with the implementation in Chapter 6. Locate `_maybe_all_gather` in the tt-symbiote repository and confirm whether it calls `ttnn.all_gather` or `ttnn.experimental.all_gather_async`. The implementation plan differs in structural detail depending on which variant is found.

---

## Summary

| Variant found | What `synchronize_device` was intended to do | Is it necessary? | Correct remedy |
|---|---|---|---|
| Synchronous `ttnn.all_gather` | Ensure completion before downstream op reads output | No — CQ0 FIFO already guarantees this | Delete `synchronize_device`; optionally upgrade to `all_gather_async` for latency; trace compatibility of synchronous `ttnn.all_gather` is unverified — see TODO in `verdict_is_it_removable.md` |
| Async `ttnn.experimental.all_gather_async` without cycling semaphores | Work around semaphore lifecycle deficiency by draining device queue | No — but removing it alone exposes the underlying bug | Wire `TT_CCL` cycling semaphores AND remove `synchronize_device` simultaneously |

In both cases, the `synchronize_device` call is removable. The difference is what must accompany the removal.

---

**Next:** [`command_queue_ordering_guarantee.md`](./command_queue_ordering_guarantee.md)
