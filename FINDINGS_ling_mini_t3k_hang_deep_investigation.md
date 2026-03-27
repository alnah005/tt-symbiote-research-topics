# Ling-mini-2.0 T3K Hang: Deep Investigation Findings

**Date:** 2026-03-27
**Status:** Investigation complete, revised fix plan below

## Executive Summary

After exhaustive parameter-by-parameter comparison of ALL CCL operations in the
Ling-mini-2.0 execution path, the `reduce_scatter_minimal_async` and
`all_gather_async` calls in `TTNNLinearIColShardedWAllReduced.forward()` now have
correct parameters matching the working MoE gold standard. The hang is NOT caused
by missing CCL parameters. The investigation reveals the real root cause is the
**sync `ttnn.all_gather` at attention.py:2287** which is dead code NOW but was
NOT dead code in a previous iteration -- OR one of several secondary issues
identified below.

---

## Investigation Results

### 1. `all_gather_async` comparison (linear.py vs MoE gold standard)

| Parameter | linear.py:234 | moe.py:1429 | normalization.py:134 | Match? |
|-----------|--------------|-------------|---------------------|--------|
| dim | -1 | -1 | -1 | YES |
| multi_device_global_semaphore | cycle_ag(1) | cycle_ag(1) | cycle_ag(1) | YES |
| barrier_semaphore | cycle_barrier(1) | cycle_barrier(1) | cycle_barrier(1) | YES |
| num_links | 1 | 1 | 1 | YES |
| topology | Linear | Linear | Linear | YES |
| cluster_axis | NOT SET | NOT SET | NOT SET | YES |
| memory_config | NOT SET | NOT SET | NOT SET | YES |
| chunks_per_sync | NOT SET | NOT SET | NOT SET | YES |
| num_workers_per_link | NOT SET | NOT SET | NOT SET | YES |
| num_buffers_per_channel | NOT SET | NOT SET | NOT SET | YES |

**Conclusion:** ALL `all_gather_async` calls in tt_symbiote are consistent. The
MoE all_gather works on T3K, so the linear.py all_gather should too.

NOTE: The tt-transformers gold standard `tt_all_gather()` in ccl.py:365 includes
`cluster_axis`, `memory_config`, `chunks_per_sync`, `num_workers_per_link`, and
`num_buffers_per_channel`. These are NOT present in any tt_symbiote
`all_gather_async` call, but the MoE works without them. This is safe for now but
may become a correctness issue in the future.

### 2. `reduce_scatter_minimal_async` comparison

| Parameter | linear.py:219 | moe.py:1478 | qwen_moe.py:922 | Match? |
|-----------|--------------|-------------|-----------------|--------|
| dim | 3 | 3 | 3 | YES |
| persistent_output_buffers | None | None | None | YES |
| num_links | 1 | 1 | 1 | YES |
| cluster_axis | 1 | 1 | 1 | YES |
| topology | Ring | Ring | Ring | YES |
| chunks_per_sync | 10 | 10 | 10 | YES |
| num_workers_per_link | 2 | 2 | 2 | YES |
| num_buffers_per_channel | 2 | 2 | 2 | YES |
| memory_config | DRAM | NOT SET | NOT SET | DIFF |
| intermediate_memory_config | DRAM | NOT SET | NOT SET | DIFF |

**Conclusion:** The linear.py reduce_scatter has 2 EXTRA parameters
(`memory_config`, `intermediate_memory_config`) that the MoE does not have. These
should be harmless as they explicitly specify DRAM which is likely the default.
However, they could be removed to match the MoE exactly.

### 3. `TTNNDistributedRMSNorm` -- NOT IN EXECUTION PATH

The test does NOT replace `BailingMoeV2RMSNorm` with `TTNNDistributedRMSNorm`.
The RMSNorm layers remain as PyTorch modules running on CPU. Therefore,
`TTNNDistributedRMSNorm.forward()` and its `all_gather_async` are never called.

### 4. `TTNNBailingMoEAttention._maybe_all_gather` sync call

Located at attention.py:2287:
```python
gathered = ttnn.all_gather(t, dim=-1, num_links=1)
```

This is **DEAD CODE** -- confirmed by exhaustive search. Neither `_forward_prefill`
nor `_forward_decode_paged` call `_maybe_all_gather`. Both paths use
`self.qkv_proj(hidden_states)` which goes through
`TTNNLinearIColShardedWAllReduced`.

**Recommendation:** Remove this dead code to prevent confusion.

### 5. `TTNNLinearIReplicatedWColSharded` (dense projection)

No CCL operations. Just matmul + optional bias. Safe.

### 6. `move_weights_to_device()` CCL operations

No CCL operations in any weight movement code. Weight preprocessing uses
`ttnn.to_device()` which is a point-to-point operation, not a collective.

### 7. `fabric_config=ttnn.FabricConfig.FABRIC_1D_RING` compatibility

All other tests (test_glm_flash, test_qwen3_5_35b_a3b, test_moe, etc.) use the
same `FABRIC_1D_RING` config. Compatible with both `Topology.Ring` (reduce_scatter)
and `Topology.Linear` (all_gather).

---

## Secondary Findings: Potential Hang Causes

### Finding A: `TTNNQwen3NextGatedAttention` reduce_scatter MISSING PARAMETERS

At attention.py:2188, there is a `reduce_scatter_minimal_async` call that is
**MISSING critical parameters**:

```python
out = ttnn.experimental.reduce_scatter_minimal_async(
    out,
    persistent_output_buffers=None,
    dim=3,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    # MISSING: num_links, cluster_axis, topology, chunks_per_sync,
    #          num_workers_per_link, num_buffers_per_channel
)
```

However, this is in `TTNNQwen3NextGatedAttention`, NOT `TTNNBailingMoEAttention`.
It is NOT in the Ling-mini-2.0 execution path. But if any Qwen3 model uses this
on T3K, it would hang.

### Finding B: Dense MLP layer 0 data flow

Layer 0 uses `BailingMoeV2MLP` (dense, not MoE). Its `gate_proj`, `up_proj`,
`down_proj` are replaced with `TTNNLinearIColShardedWRowSharded`. The data flow
is mathematically correct:

1. gate_proj: [seq, hidden/8] x [hidden/8, intermediate] -> reduce_scatter -> [seq, intermediate/8]
2. up_proj: [seq, hidden/8] x [hidden/8, intermediate] -> reduce_scatter -> [seq, intermediate/8]
3. gate_output * up_output -> [seq, intermediate/8]
4. down_proj: [seq, intermediate/8] x [intermediate/8, hidden] -> reduce_scatter -> [seq, hidden/8]

This is correct. But the PyTorch activation `act_fn` (SiLU) between gate_proj
and the multiplication runs through `__torch_dispatch__`. If the dispatch fails
to route SiLU to TTNN, it would fall back to PyTorch, requiring a device-to-host
roundtrip for each device. This is slow but should not hang.

### Finding C: `_to_replicated()` host round-trip (attention.py:2292-2317)

The `_to_replicated()` method in `TTNNBailingMoEAttention` does:
```python
mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
t_torch = ttnn.to_torch(t, mesh_composer=mesh_composer)
```

This brings the tensor back to host and re-uploads with `ReplicateTensorToMesh`.
However, `_to_replicated()` is NOT called in either forward path (prefill or
decode). It exists for potential future use. Not a hang risk.

### Finding D: KV cache tensor topology mismatch

The KV cache is created with `ttnn.zeros(..., device=mesh_device)` without a
`mesh_mapper`. On multi-device, `ttnn.zeros` creates a tensor that is identical
on all devices. The `paged_fill_cache` and `paged_update_cache` operations
receive Q/K/V states that went through `all_gather` (replicated) and then various
reshapes. The paged attention kernels require specific tensor topologies
(replicated). If the tensors from the QKV projection path have incorrect topology
metadata after the all_gather, the paged attention kernel could hang.

This is a PLAUSIBLE hang cause.

### Finding E: HuggingFace generation pipeline interaction

During `model.generate()`, HuggingFace's `GenerationMixin` runs:
1. Prefill forward pass
2. Sample/select next token (requires logits on CPU)
3. Decode forward passes (one per token)

Step 2 requires the `lm_head` output (logits) to be brought to CPU for
`torch.argmax` or sampling. The `lm_head` is `TTNNLinearIColShardedWRowSharded`
which produces a reduce_scattered output of shape `[1, 1, 1, vocab/8]` per
device. HuggingFace's generation code would try to operate on this as a
`TorchTTNNTensor`. The `to_torch` conversion would use the distributed config's
`mesh_composer` (`ConcatMesh2dToTensor(dims=(0,-1))`) which would concatenate
the sharded pieces back. This SHOULD work.

But if the `lm_head` output shape is unexpected for the generation pipeline
(e.g., 4D instead of 2D/3D), it could cause issues. The
`TTNNLinearIColShardedWRowSharded.forward()` reshapes back at line 175:
`ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])`. The -1 here would be
`vocab_size/8` (the reduce_scattered dim). The `logical_shape_fn` in the
distributed config should adjust this to the full vocab_size. If it doesn't, the
generation pipeline would see wrong logits shape and potentially hang or crash.

### Finding F: TTNNSilu replacement for act_fn

The test replaces `nn.SiLU -> TTNNSilu`. But the dense MLP's `act_fn` in
`BailingMoeV2MLP` is `ACT2FN[config.hidden_act]` which is likely `nn.SiLU()`.
After the `nn.SiLU` instance is replaced with `TTNNSilu`, the dense MLP forward:
```python
self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```
calls `self.act_fn(self.gate_proj(x))`. The `gate_proj(x)` returns a
`TorchTTNNTensor`. Then `self.act_fn` (now `TTNNSilu`) is called on it. The
`TTNNSilu.__call__` goes through `module_run` which converts input to TTNN and
runs `ttnn.silu`. This should work.

---

## Revised Root Cause Analysis

Given that all CCL parameters are now correct, the most likely remaining hang
causes are, in order of probability:

### MOST LIKELY: Semaphore Exhaustion / Cycling Bug

The `TT_CCL` class (ccl.py) allocates **only 2** semaphore handle sets per type
(double-buffered). In one transformer layer, the CCL operations are:

**Layer 0 (dense MLP):**
1. Attention qkv_proj: RS semaphore [1][0], barrier [1][0], AG semaphore [1][1], barrier [1][1]
2. Dense MLP gate_proj: RS semaphore [1][0], barrier [1][0]
3. Dense MLP up_proj: RS semaphore [1][1], barrier [1][1]
4. Dense MLP down_proj: RS semaphore [1][0], barrier [1][0]

**Layer 1+ (MoE):**
1. Attention qkv_proj: RS semaphore [1][1], barrier [1][1], AG semaphore [1][0], barrier [1][0]
2. MoE all_gather: AG semaphore [1][1], barrier [1][1]
3. MoE reduce_scatter: RS semaphore [1][0], barrier [1][0]

With double-buffering and cycling, the semaphore indices alternate 0->1->0->1...
Each layer uses multiple CCL ops that cycle through the same 2 slots. If an
async op from the previous layer hasn't completed before the same semaphore is
reused, there could be a deadlock.

This is especially risky during **prefill** where the sequence length is long and
operations take more time. The `num_command_queues: 1` means all ops go through
one queue and are serialized. However, the "async" nature of the CCL ops means
they may return to the host before the device finishes. If the host reuses a
semaphore before the device is done with it, deadlock.

**Mitigation:** Add `ttnn.synchronize_device(self.device)` before each CCL op
as a diagnostic step. If this fixes the hang, the issue is semaphore
reuse/cycling.

### SECOND MOST LIKELY: Prefill tensor shape incompatibility with CCL

During prefill, the QKV projection input is `[1, seq_len, hidden_size]` (3D).
The `TTNNLinearIColShardedWAllReduced.forward()` pads to 4D:
`[1, 1, seq_len, hidden_size/8]`. After matmul: `[1, 1, seq_len, qkv_dim]`.
The reduce_scatter on dim=3 then splits qkv_dim by 8: `[1, 1, seq_len, qkv_dim/8]`.

For Ling-mini-2.0: hidden_size=2048, qkv_dim = num_heads*head_dim + 2*num_kv_heads*head_dim = 16*128 + 2*4*128 = 3072.

After reduce_scatter: `[1, 1, seq_len, 384]`. After all_gather: `[1, 1, seq_len, 3072]`.

The reduce_scatter requires `dim_size % num_devices == 0` (3072 % 8 = 0). OK.

But for the dense MLP's `gate_proj`: out_features = intermediate_size. Need to
check if `intermediate_size % 8 == 0`. From config: `intermediate_size = 5632`.
5632 / 8 = 704. OK.

And `down_proj` reduce_scatter output: `hidden_size / 8 = 2048 / 8 = 256`. OK.

So tensor shapes are all divisible. This is not the issue.

### THIRD: `_forward_prefill` using `_forward_decode_paged` routing

During the first `model.generate()` call:
- Prefill: seq_length > 1, routes to `_forward_prefill`
- Decode: seq_length == 1, routes to `_forward_decode_paged`

If HuggingFace somehow calls with seq_length == 1 first (e.g., for the first
token), it would go to `_forward_decode_paged` which uses paged attention. The
paged attention requires the KV cache to have been filled first (via prefill).
If decode is called before prefill, the KV cache is empty and the decode might
hang.

But HuggingFace's `generate()` always does prefill first, so this should not
happen.

---

## Revised Fix Plan

### Step 1: Diagnostic -- Add synchronize_device before CCL ops

Add `ttnn.synchronize_device(self.device)` before each `reduce_scatter` and
`all_gather` call in `TTNNLinearIColShardedWAllReduced.forward()` to rule out
semaphore cycling race conditions:

```python
# In TTNNLinearIColShardedWAllReduced.forward() (linear.py:219)
ttnn.synchronize_device(self.device)  # DIAGNOSTIC
tt_output = ttnn.experimental.reduce_scatter_minimal_async(...)
ttnn.synchronize_device(self.device)  # DIAGNOSTIC
tt_output = ttnn.experimental.all_gather_async(...)
```

Also add to `TTNNLinearIColShardedWRowSharded.forward()` (linear.py:158).

### Step 2: Match MoE reduce_scatter exactly

Remove `memory_config` and `intermediate_memory_config` from both
`TTNNLinearIColShardedWRowSharded.forward()` (line 158) and
`TTNNLinearIColShardedWAllReduced.forward()` (line 219) to match the MoE gold
standard exactly:

```python
# Remove these two lines from reduce_scatter calls:
#   memory_config=ttnn.DRAM_MEMORY_CONFIG,
#   intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
```

### Step 3: Add debug prints to identify exact hang location

Add print statements before/after each CCL op and before/after each module
forward to identify exactly which operation hangs:

```python
print(f"[DEBUG] {self.module_name}: Before reduce_scatter, shape={tt_output.shape}")
tt_output = ttnn.experimental.reduce_scatter_minimal_async(...)
print(f"[DEBUG] {self.module_name}: After reduce_scatter, shape={tt_output.shape}")
tt_output = ttnn.experimental.all_gather_async(...)
print(f"[DEBUG] {self.module_name}: After all_gather, shape={tt_output.shape}")
```

### Step 4: Clean up dead code

Remove `TTNNBailingMoEAttention._maybe_all_gather` (the sync all_gather at
attention.py:2284-2290) and `_to_replicated` (attention.py:2292-2317) which are
dead code and could confuse future debugging.

### Step 5: Fix TTNNQwen3NextGatedAttention reduce_scatter

Add the missing parameters to the reduce_scatter at attention.py:2188 (not in
Ling execution path but will hang for Qwen3 models on T3K):

```python
out = ttnn.experimental.reduce_scatter_minimal_async(
    out,
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

---

## Files Investigated

| File | Path | Relevant Lines |
|------|------|---------------|
| linear.py | `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/linear.py` | 158-172, 219-241 |
| attention.py | `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/attention.py` | 2188-2194, 2252-2843 |
| moe.py | `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/moe.py` | 1429-1490 |
| normalization.py | `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/modules/normalization.py` | 99-151 |
| ccl.py | `/home/ttuser/salnahari/tt-metal/models/tt_transformers/tt/ccl.py` | 59-133, 136-297 |
| run_config.py | `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/core/run_config.py` | 41-80, 534-590 |
| module.py | `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/core/module.py` | 24-33, 166-172 |
| test_ling_mini_2_0.py | `/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py` | full file |
| modeling_bailing_moe_v2.py | HF cache | 271-284, 941-1013, 1146-1275 |
| configuration_bailing_moe_v2.py | HF cache | 38 (first_k_dense_replace=1) |
