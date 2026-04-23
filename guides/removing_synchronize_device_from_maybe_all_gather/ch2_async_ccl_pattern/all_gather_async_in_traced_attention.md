# all_gather_async in the Traced Attention Path

This file walks the `Attention.forward_decode` path in `models/tt_transformers/tt/attention.py`, showing the exact `ttnn.experimental.all_gather_async` call (and the fused `all_gather_matmul_async` variant) as they appear in the source. By the end you will know the complete argument list for each call, have a confirmed answer to whether `ttnn.synchronize_device()` appears anywhere in this decode path, and understand why CQ0 FIFO ordering makes a host-side barrier unnecessary.

Source file: `models/tt_transformers/tt/attention.py` in the `tt-metal` repository.

---

## The Decode Path — Two Variants

`Attention.forward_decode` has two CCL code paths controlled by `self.use_fused_all_gather_matmul` and `self.ccl_topology`. Both are shown below.

### Variant A — Fused `all_gather_matmul_async` (Ring topology only)

When `self.use_fused_all_gather_matmul` is `True` and `self.ccl_topology == ttnn.Topology.Ring`, the attention output projection is performed as a single fused kernel that gathers across devices and multiplies by `wo` in one dispatch:

```python
# From Attention.forward_decode, lines ~551-568
# self.ccl_topology == ttnn.Topology.Ring and self.use_fused_all_gather_matmul == True

attn_output_cat = ttnn.to_memory_config(
    attn_output_cat,
    self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"],  # why: fused kernel needs specific sharded layout
)

_, dense_out_sharded = ttnn.experimental.all_gather_matmul_async(
    attn_output_cat,
    self.wo,                                                      # why: output projection weight; fused into the gather
    persistent_output_buffer=None,                               # why: program cache allocates on first call, reuses on replay
    dim=3,                                                        # why: gather along the hidden-dim axis
    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                                                                  # why: cycling double-buffered GlobalSemaphore for
                                                                  #      all_gather completion signaling; slot cycles
                                                                  #      0 → 1 → 0 on consecutive calls to avoid aliasing
    all_gather_core_grid_offset=(0, 4),                          # why: core grid offset for the all_gather sub-op
    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                                                                  # why: cycling barrier GlobalSemaphore that all ranks
                                                                  #      must reach before the gather proceeds
    num_links=1,                                                  # why: single NIC link for ring topology
    memory_config_ag=self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"],
                                                                  # why: memory config for the gathered intermediate
    memory_config_mm=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                                                                  # why: memory config for the matmul output
    program_config=self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"],
    compute_kernel_config=self.compute_kernel_config_hifi2,
    chunks_per_sync=10,                                           # why: number of data chunks transferred per semaphore sync
    num_workers_per_link=2,                                       # why: worker count per NIC link
    num_buffers_per_channel=2,                                    # why: double-buffered data channel
)
# NO ttnn.synchronize_device() here or anywhere after this call
```

### Variant B — Standalone `all_gather_async` + separate `ttnn.linear` (non-Ring topology)

When `self.use_fused_all_gather_matmul` is `True` but `self.ccl_topology != ttnn.Topology.Ring`, the all_gather and matmul are issued as two separate ops.

> **Note:** Variant B and Variant A both require `use_fused_all_gather_matmul is True` (outer gate). Variant B is selected when the inner gate `ccl_topology == Ring` is NOT met.

```python
# From Attention.forward_decode, lines ~570-592
# self.use_fused_all_gather_matmul == True, ccl_topology != Ring

all_gather_output = ttnn.experimental.all_gather_async(
    attn_output_cat,
    persistent_output_buffer=None,                               # why: program cache provides address stability for trace
    dim=3,                                                        # why: gather along the hidden-dim axis (all heads → full dim)
    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                                                                  # why: cycling double-buffered GlobalSemaphore for
                                                                  #      completion signaling; returns slot 0 or slot 1
                                                                  #      alternately to prevent aliasing across consecutive
                                                                  #      decode steps in a traced loop
    num_links=1,                                                  # why: single NIC link
    topology=self.ccl_topology,                                   # why: Linear or Ring topology passed through from config
    memory_config=self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"],
                                                                  # why: sharded output layout expected by the subsequent
                                                                  #      ttnn.linear call
    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                                                                  # why: cycling barrier GlobalSemaphore; all ranks rendez-vous
                                                                  #      before the gather data is considered complete
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
# NO ttnn.synchronize_device() between all_gather_async and the linear below

dense_out_sharded = ttnn.linear(
    all_gather_output,                                            # why: this is enqueued to CQ0 AFTER all_gather_async;
                                                                  #      CQ0 FIFO ordering guarantees it will not execute
                                                                  #      until all_gather_async has delivered its output
    self.wo,
    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
    program_config=self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"],
    compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
)

ttnn.deallocate(all_gather_output)
```

### Variant C — `tt_all_gather` helper (non-fused path)

When `self.use_fused_all_gather_matmul` is `False`, the path calls `tt_all_gather` (defined in `models/tt_transformers/tt/ccl.py`), which in turn calls `ttnn.experimental.all_gather_async` with the same cycling semaphore pattern (see `ccl.py` lines ~239-252 for the `cluster_axis is not None` branch). No `ttnn.synchronize_device()` appears in that path either. Unlike Variants A and B which use `ttnn.linear` after the all_gather, Variant C then calls `ttnn.matmul` followed by `tt_all_reduce` before returning the attention output.

---

## Is There a `synchronize_device` Anywhere in the Decode Path?

> **Key finding:** There is no `ttnn.synchronize_device()` call anywhere in `Attention.forward_decode` in `models/tt_transformers/tt/attention.py`. A search of the entire file confirms the same: the word `synchronize_device` does not appear anywhere in `attention.py` or in `ccl.py`.

This was verified by searching both files. The absence is intentional: the entire decode path — QKV projection, all-reduce, head splitting, RoPE, KV cache update, SDPA, all_gather, output projection — is issued to CQ0 as a purely asynchronous command stream, with no host-side barrier at any point.

---

## Why This Works Without `synchronize_device`

CQ0 FIFO ordering guarantees that the downstream op (e.g., `ttnn.linear`) cannot execute before the `all_gather_async` completes and its output is present at the agreed device address. No host-side barrier is needed to enforce this ordering. For the full argument, see [`synchronize_device_semantics.md`](../ch1_maybe_all_gather_anatomy/synchronize_device_semantics.md).

---

## `persistent_output_buffer=None` — What It Means

Both Variant A and Variant B pass `persistent_output_buffer=None`, delegating output buffer management to the op's program cache. The program cache allocates a buffer address on the compile run and reuses that same address on every subsequent capture and replay, making the call trace-safe without requiring the caller to pre-allocate a buffer explicitly. For the full three-phase lifecycle and what happens when this guarantee is violated, see [`persistent_output_buffer_contract.md`](./persistent_output_buffer_contract.md).

---

## Forward Reference

For the cycling semaphore handles passed as `multi_device_global_semaphore` and `barrier_semaphore` in each call above, see [`cycling_semaphore_mechanics.md`](./cycling_semaphore_mechanics.md).
