# All-Gather Topology and Detailed Breakdown

This file walks through each collective operation in `TTNNBailingMoEAttention._forward_decode_paged` (lines 2610–2799 of `attention.py`) in execution order, explaining what each op does, why it is required, and what the data layout looks like on entry and exit.

---

## Collective 1 — Reduce-Scatter Inside Q Projection (line 2624, executed inside `q_proj`)

```python
query_states = self.q_proj(hidden_states)
```

`self.q_proj` is a `TTNNLinearIColShardedWRowSharded` instance (set at line 2374 of `attention.py`). Its `forward` method (lines 133–176 of `linear.py`) does the following:

1. Verify input is col-sharded on dim=-1.
2. Reshape to 4D: `[B, 1, 1, d_model/N]` per device.
3. `ttnn.linear(input_tensor, self.tt_weight)` — local matmul. Each device holds a row-shard of the Q weight: `W_Q_shard_i` has shape `[H*D/N, d_model]` (= `[256, 2048]` for Bailing). The local matmul produces `[B, 1, 1, H*D]` on every device — the full Q output, but computed only from the local input shard. These partial results must be summed across devices to get the correct Q.
4. `ttnn.experimental.reduce_scatter_minimal_async(tt_output, dim=3, cluster_axis=1, topology=ttnn.Topology.Ring, …)` — reduces (sums) the partial Q outputs across devices and scatters the result so device `i` ends up with columns `i*(H*D/N)` through `(i+1)*(H*D/N)−1` of the full Q tensor.

The output of `q_proj` is therefore col-sharded Q: each device holds `[B, 1, H*D/N]` = `[B, 1, 256]` columns.

**Why `IColShardedWRowSharded` was chosen for Q.** The comment at line 2274–2275 reads: "Q is sharded (num_heads >= num_devices), K/V must be replicated." With H=16 query heads and N=8 devices, each device can hold H/N = 2 complete query heads. By sharding Q on the output dimension and distributing the weight rows correspondingly, the matmul compute is parallelized. However, this requires a reduce-scatter to aggregate the partial sums.

**Data volume of the reduce-scatter:** Each device contributes B × 1 × H*D × 2 = B × 4096 bytes to the ring; each device receives back B × (H*D/N) × 2 = B × 512 bytes. For B=32: sent 128 KB, received 16 KB.

---

## Collective 2 — Hidden States All-Gather (line 2626)

```python
hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
key_states = self.k_proj(hidden_states_replicated)
value_states = self.v_proj(hidden_states_replicated)
ttnn.deallocate(hidden_states_replicated)
```

**Why this is needed.** `TTNNLinearIReplicatedWColSharded` (used for `k_proj` and `v_proj`) requires a fully-replicated input: each device must hold the complete `[B, 1, d_model]` hidden state so it can perform a local matmul against its own column-shard of the weight matrix. The K weight shard on device `i` holds columns `i*(Hkv*D/N)` through `(i+1)*(Hkv*D/N)−1` of the `[d_model, Hkv*D]` matrix; the matmul `hidden_states × W_K_shard_i` produces the K output shard for device `i`.

The hidden states arriving at this layer from the MoE FFN layer are col-sharded: each device holds `d_model/N = 256` columns, not the full 2048. This is because the MoE output linear projection (`TTNNLinearIColShardedWRowSharded`) ends with a reduce-scatter, producing col-sharded output. The all-gather at line 2626 reconstitutes the full `[B, 1, 2048]` tensor on every device.

**Layout summary:**
- Input to op: `[B, 1, d_model/N]` col-sharded across 8 devices, DRAM
- Output of op: `[B, 1, d_model]` replicated on every device, DRAM

**Data volume:** B × d\_model bytes total on each device after the gather. For B=32: 32 × 2048 × 2 bytes (bfloat16) = 131 072 bytes ≈ 128 KB per device.

---

## Collective 3 — Q All-Gather (line 2631)

```python
query_states = self._maybe_all_gather(query_states)
```

`_maybe_all_gather` is defined at lines 2280–2286 of `attention.py`:

```python
def _maybe_all_gather(self, tensor):
    t = tensor.to_ttnn if hasattr(tensor, "to_ttnn") else tensor
    gathered = ttnn.all_gather(t, dim=-1, num_links=1)
    if gathered.dtype != ttnn.bfloat16:
        gathered = ttnn.typecast(gathered, ttnn.bfloat16)
    return gathered
```

This call reconstitutes the full Q tensor from the reduce-scatter output. After this op, every device holds `[B, 1, H*D]` = `[B, 1, 2048]` for Q.

**Why this is needed.** The downstream operations — QK norm, RoPE, and paged SDPA — require each device to have the full Q tensor. In particular, `paged_sdpa_decode` computes attention for all H query heads and all Hkv key-value heads; it cannot work on a partial column shard.

**Layout summary:**
- Input to op: `[B, 1, d_model/N]` = `[B, 1, 256]` col-sharded, DRAM
- Output of op: `[B, 1, d_model]` = `[B, 1, 2048]` replicated, DRAM

**Data volume:** identical to Collective 2 (same tensor shape). For B=32: ≈112 KB per device received (= 7/8 × 32 × 2048 × 2 bytes; each device already holds its own 1/8 shard and receives only the remaining 7/8 from peers).

**This op is the redundant one.** The same data was just all-gathered in Collective 2 to feed K/V, processed through the Q matmul, reduced, and is now being all-gathered again. See [sharding_alternatives.md](sharding_alternatives.md) for how to eliminate it.

---

## Collectives 4 & 5 — K and V All-Gathers (lines 2632–2633)

```python
key_states = self._maybe_all_gather(key_states)
value_states = self._maybe_all_gather(value_states)
```

Both `k_proj` and `v_proj` use `TTNNLinearIReplicatedWColSharded` (lines 2375–2376 of `attention.py`). Each projection's `forward` (lines 263–276 of `linear.py`) performs a local matmul on the replicated input against the col-sharded weight, producing col-sharded output without any inter-device communication: device `i` computes `hidden_states × W_K_shard_i` (or `W_V_shard_i`), giving it columns `i*(Hkv*D/N)` through `(i+1)*(Hkv*D/N)−1`. With Hkv=4, D=128, and N=8, each device holds 64 elements per token per batch item.

The all-gather makes K (and identically V) fully replicated so that each device has the complete `[B, 1, Hkv*D]` = `[B, 1, 512]` tensor before the reshape to decode layout. V is mechanically identical to K.

**Layout summary (both K and V):**
- Input to op: `[B, 1, Hkv*D/N]` = `[B, 1, 64]` col-sharded, DRAM
- Output of op: `[B, 1, Hkv*D]` = `[B, 1, 512]` replicated, DRAM

**Data volume (each):** B × Hkv × D bytes. For B=32: 32 × 512 × 2 = 32 768 bytes total per device, but bytes actually received over the link = 7/8 × 32 768 = 28 672 bytes ≈ 28 KB per device received (each device already holds its own 1/8 shard).

---

## Synchronous `ttnn.all_gather` vs Async `ttnn.experimental.all_gather_async`

`TTNNBailingMoEAttention` uses the **synchronous** `ttnn.all_gather` for all four of its all-gathers (Collectives 2, 3, 4, 5). This means each call blocks until every device has finished sending and receiving data before the Python runtime moves to the next op dispatch.

`TTNNQwen3FullAttention` (in `qwen_attention.py`) uses the **async** variant:

```python
# qwen_attention.py lines 398–408
gathered = ttnn.experimental.all_gather_async(
    t,
    dim=-1,
    multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1),
    barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    topology=ttnn.Topology.Linear,
)
ttnn.synchronize_device(self.device)
```

The async API launches the collective without blocking the host dispatch thread, but the explicit `ttnn.synchronize_device` call immediately after reintroduces a hard barrier. In the current Qwen3 implementation this means the async variant does not achieve overlap with subsequent compute — it only differs from the synchronous variant in that the CCL engine can schedule the transfer at the device level while the host prepares the next command. The practical latency difference depends on how much host-side kernel dispatch overhead exists between the gather and the next dependent op.

A meaningful overlap would require placing independent compute between the `all_gather_async` launch and the `synchronize_device`. Qwen3 does not currently do this; `TTNNBailingMoEAttention` does not use the async path at all. An opportunity exists for `TTNNBailingMoEAttention` to adopt `all_gather_async` with an explicit overlap window.

`TTNNGlm4MoeLiteAttention` also uses `ttnn.experimental.all_gather_async` (line 1604 of `attention.py`) via its own `_maybe_all_gather` implementation (lines 1600–1611), but unlike Qwen3 it does **not** call `ttnn.synchronize_device` afterward — it returns the async gather result directly. GLM4's `_maybe_all_gather` is therefore genuinely async: the collective is launched and the function returns without waiting for completion, leaving downstream ops to consume the result when it arrives. This contrasts with Qwen3, which issues `all_gather_async` and then immediately synchronizes, effectively making it synchronous.

---

## Ring Topology vs Linear Topology

The synchronous `ttnn.all_gather` in `TTNNBailingMoEAttention` does not specify a topology argument; it receives the library default. The `TTNNLinearIColShardedWRowSharded` reduce-scatter explicitly sets `topology=ttnn.Topology.Ring` (line 168 of `linear.py`).

`TTNNQwen3FullAttention._maybe_all_gather` specifies `topology=ttnn.Topology.Linear` for its async all-gathers.

The difference:

- **Ring topology:** data travels in both directions around the 8-device ring simultaneously. Effective per-step volume per link is halved because the ring uses N/2 hops in each direction for a balanced ring. This is more bandwidth-efficient for large payloads.
- **Linear topology:** data travels in one direction only (a chain). This is simpler to schedule and can have lower latency for small payloads where setup overhead dominates, or for topologies where the physical interconnect is not a closed ring. For 8 devices in a chain, each message traverses up to 7 hops.

For the small tensors in decode (≤128 KB per all-gather at B=32), the topology choice has a second-order effect. For prefill with long sequences the ring topology's balanced routing becomes more significant.

---

## All-Gather Count Comparison

| Model | All-gathers per decode step | Reduce-scatters per decode step | Total collectives | Sync API |
|-------|----------------------------|---------------------------------|-------------------|----------|
| `TTNNBailingMoEAttention` | 4 (1 input + 3 post-proj) | 1 (inside Q proj) | 5 | Synchronous `ttnn.all_gather` for all-gathers; async reduce-scatter inside `q_proj` |
| `TTNNQwen3FullAttention` | 6 (1 input + 3 post-proj + 2 cos/sin) | 0 (no `IColShardedWRowSharded`) | 6 | Async `ttnn.experimental.all_gather_async` + synchronize |
| `TTNNGlm4MoeLiteAttention` | 4+ (depends on MLA projection path; `q_states`, `compressed_kv`, `kv_full`, cos, sin each go through `_maybe_all_gather`) | 1 (inside `q_a_proj` or `q_b_proj` if using `TTNNLinearIColShardedWRowSharded`) | varies | Async `all_gather_async` (no synchronize) |

---

## Navigation

**Next:** [Sharding Alternatives](sharding_alternatives.md)
