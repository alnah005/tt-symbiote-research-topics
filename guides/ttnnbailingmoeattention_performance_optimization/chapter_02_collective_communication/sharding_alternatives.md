# Sharding Alternatives

This file analyzes the current Q projection sharding strategy, explains why the design produces a redundant pair of collectives in the decode path, and evaluates three alternatives that could reduce the per-step collective count.

---

## Current Q Sharding Scheme: `TTNNLinearIColShardedWRowSharded`

`self.q_proj` is instantiated as `TTNNLinearIColShardedWRowSharded` at line 2374 of `attention.py`:

```python
new_attn.q_proj = TTNNLinearIColShardedWRowSharded.from_torch(q_linear)
```

The comment on lines 2274–2275 explains the design choice:
```
# Q is sharded (num_heads >= num_devices), K/V must be replicated
```

`TTNNLinearIColShardedWRowSharded` performs a column-parallel matmul with a reduce-scatter output:

1. **Input requirement:** col-sharded on dim=-1; each device holds `d_model/N = 256` input features.
2. **Weight layout:** row-sharded (`weight_dim=-2`); device `i` holds rows `i*(d_model/N)` to `(i+1)*(d_model/N)−1` of the `[H*D, d_model]` Q weight matrix.
3. **Local matmul:** `input_shard_i × W_Q_shard_i` → partial Q result of shape `[B, 1, H*D]` on every device.
4. **Reduce-scatter:** `ttnn.experimental.reduce_scatter_minimal_async(dim=3, cluster_axis=1, topology=Ring)` — sums the partial Q results across devices and gives device `i` only columns `i*(H*D/N)` through `(i+1)*(H*D/N)−1` of the full Q.

The output is col-sharded Q: `[B, 1, H*D/N]` = `[B, 1, 256]` per device.

The subsequent `_maybe_all_gather(query_states)` at line 2631 then reconstitutes the full Q, giving every device `[B, 1, H*D]` = `[B, 1, 2048]`.

**Net result for Q:** the matmul work is parallelized (each device processes `d_model/N` input features), but the pair reduce-scatter + all-gather together consumes one full all-gather worth of link bandwidth in addition to the reduce-scatter bandwidth, achieving no net saving over simply not sharding Q at all.

---

## Why the Hidden All-Gather and the Q Reduce-Scatter Are Structurally Redundant

The hidden states arrive at `_forward_decode_paged` as col-sharded `[B, 1, d_model/N]`. Two operations must transform them:

- For **K/V**: need replicated `[B, 1, d_model]` → perform all-gather (Collective 2, line 2626).
- For **Q**: need col-sharded `[B, 1, d_model/N]` to serve as input to `TTNNLinearIColShardedWRowSharded` → no transformation needed, the input is already in the right form.

So the K/V all-gather and the Q col-sharded input are handled by opposite transformations of the same source tensor. The Q reduce-scatter then contracts the full `[B, 1, H*D]` matmul output back down to `[B, 1, H*D/N]`, which is immediately reversed by the Q all-gather. From the perspective of the caller:

```
hidden (col-sharded) → [K/V path] → all_gather → replicated hidden → k_proj → k (col-sharded) → all_gather → replicated K
hidden (col-sharded) → [Q path]  → q_proj (matmul + reduce-scatter) → Q (col-sharded) → all_gather → replicated Q
```

Both paths end with an all-gather on col-sharded output. The Q path additionally pays a reduce-scatter inside `q_proj`. If Q were projected the same way as K/V (replicated input, col-sharded output, no reduce-scatter), the Q all-gather would be pulling from a cheaper starting point and no reduce-scatter would have occurred.

---

## Alternative 1 — Use `TTNNLinearIReplicatedWColSharded` for Q

Replace line 2374:

```python
# Current
new_attn.q_proj = TTNNLinearIColShardedWRowSharded.from_torch(q_linear)

# Proposed
new_attn.q_proj = TTNNLinearIReplicatedWColSharded.from_torch(q_linear)
```

With `TTNNLinearIReplicatedWColSharded`, Q projection works identically to K and V:
1. Requires replicated input `[B, 1, d_model]`.
2. Weight is col-sharded (`weight_dim=-1`); device `i` holds columns `i*(H*D/N)` through `(i+1)*(H*D/N)−1` of the Q weight.
3. Local matmul produces col-sharded Q: `[B, 1, H*D/N]` per device. No reduce-scatter.
4. The single initial `ttnn.all_gather(hidden_states)` at line 2626 covers all three projections — Q, K, and V all use the same replicated `hidden_states_replicated`.

**Collective savings:**
- Eliminate the reduce-scatter inside `q_proj` (Collective 1).
- The Q all-gather (Collective 3) still occurs and still moves the same data volume as before.
- Net saving: **one reduce-scatter** per decode step.

**Feasibility analysis.** This requires `q_proj` to receive replicated `hidden_states_replicated` rather than the original col-sharded `hidden_states`. The code change is:

```python
# Current (lines 2624–2631 of attention.py)
query_states = self.q_proj(hidden_states)                          # uses col-sharded hidden_states
hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
key_states = self.k_proj(hidden_states_replicated)
value_states = self.v_proj(hidden_states_replicated)
ttnn.deallocate(hidden_states_replicated)
query_states = self._maybe_all_gather(query_states)

# Proposed (reorder: all_gather first, then all three projections)
hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
query_states = self.q_proj(hidden_states_replicated)               # replicated input
key_states = self.k_proj(hidden_states_replicated)
value_states = self.v_proj(hidden_states_replicated)
ttnn.deallocate(hidden_states_replicated)
query_states = self._maybe_all_gather(query_states)
```

The same change applies to `_forward_prefill` (lines 2539–2549).

**Compute cost:** `TTNNLinearIReplicatedWColSharded` performs the matmul on replicated input, so each device performs the full `[B, d_model] × [d_model, H*D/N]` matmul rather than the sharded `[B, d_model/N] × [d_model/N, H*D]` matmul. The arithmetic workload is identical (same number of multiply-adds). The difference is in how the input is partitioned, not the total FLOPs.

**Weight change:** Q weight must be transposed and sharded differently:
- Current (`IColShardedWRowSharded`): weight layout is `[H*D, d_model]`, sharded on dim=-2 (rows), so each device holds `[H*D/N, d_model]` (= `[256, 2048]` for Bailing).
- Proposed (`IReplicatedWColSharded`): weight layout is `[H*D, d_model]`, sharded on dim=-1 (columns), so each device holds `[H*D, d_model/N]` (= `[2048, 256]` for Bailing).

This requires a change in `from_torch` (lines 2357–2376 of `attention.py`): replace `TTNNLinearIColShardedWRowSharded.from_torch(q_linear)` with `TTNNLinearIReplicatedWColSharded.from_torch(q_linear)`. No weight permutation is needed beyond what `from_torch` already handles (the HF-to-Meta layout permute of lines 2345–2348).

**Correctness:** `TTNNQwen3FullAttention` uses exactly this configuration (`LinearClsIn = TTNNLinearIReplicatedWColSharded` at line 250 of `qwen_attention.py`) for all three projections including Q, and it is validated on T3K.

---

## Alternative 2 — Fused QKV Projection

Project Q, K, and V in a single matmul call on replicated hidden states:

```
hidden_states_replicated [B, 1, d_model] × W_QKV [d_model, (H+2*Hkv)*D] → QKV [B, 1, (H+2*Hkv)*D]
```

Then slice to extract Q, K, V before the all-gathers.

This mirrors the original `BailingMoeV2Attention` in PyTorch, which uses a fused `query_key_value` projection (as documented at line 2337 of `attention.py` where the weight is split out):

```python
qkv_weight = torch_attn.query_key_value.weight  # [(num_heads + 2*num_kv_heads) * head_dim, hidden_size]
```

The fused weight is `[(16 + 2×4) × 128, 2048]` = `[3072, 2048]`.

**Collective savings:** With a fused projection using `TTNNLinearIReplicatedWColSharded`:
- One all-gather for hidden states (same as current Collective 2).
- One local matmul producing col-sharded QKV.
- Three slices on col-sharded output (no collective).
- Three all-gathers for Q, K, V (same as current Collectives 3, 4, 5).

The reduce-scatter (Collective 1) is eliminated, same saving as Alternative 1. The overhead is the same as Alternative 1.

**Additional benefit:** Fusing the three matmuls into one reduces kernel launch overhead (three `ttnn.linear` calls → one) and can improve arithmetic intensity if the combined weight matrix fits more favorably in L1 cache.

**Complication:** The fused output is `(H+2*Hkv)*D = (16+8)*128 = 3072` columns, and slicing a col-sharded tensor requires each device to receive the correct sub-range. With `IReplicatedWColSharded`, the fused projection produces col-sharded output of shape `[B, 1, 3072/N]` = `[B, 1, 384]` per device. Slicing at the shard boundary requires that the Q and K/V boundaries align with shard boundaries. With N=8 and Q occupying 2048 columns and K/V occupying 512 columns each:
- Q: columns 0–2047, sharded: per device 256 columns → Q is `[B, 1, 2048/8]` = `[B, 1, 256]` per device (shard-aligned).
- K: columns 2048–2559, sharded: 512/8 = 64 per device (shard-aligned).
- V: columns 2560–3071, sharded: 512/8 = 64 per device (shard-aligned).

All boundaries divide evenly, so slicing the fused col-sharded output by shard-local ranges is well-defined. Each device slices its local 384 columns into Q (0–255), K (256–319), V (320–383) without any inter-device communication.

---

## Alternative 3 — Permanently Sharded Q Heads (No Q All-Gather)

The most aggressive reduction: keep Q permanently sharded at `H/N = 2` heads per device and redesign the SDPA kernel to operate on head-sharded rather than replicated Q.

**Feasibility analysis for Bailing (H=16, Hkv=4, N=8):**

- Q: 16 heads / 8 devices = **2 Q heads per device** — divides evenly.
- Kv: 4 KV heads / 8 devices = **0.5 KV heads per device** — does not divide evenly.

The non-integer KV head assignment is the fundamental obstacle. Paged SDPA requires each device to attend its local Q heads against KV heads. With 4 KV heads and 8 devices, there is no valid head assignment: at least 4 devices would receive 0 complete KV heads and 4 devices would receive 1 KV head, and the attention pattern would be incorrect unless KV heads are replicated on all devices (which loses the bandwidth saving).

One partial mitigation: replicate KV heads across all 8 devices (since they are small: Hkv=4, D=128 → 512 elements per token) and keep Q sharded at 2 heads per device. This avoids the Q all-gather (Collective 3) entirely:

- Collective 2 (hidden all-gather for K/V): unchanged, still required.
- Collective 1 (Q reduce-scatter): eliminated (if Q proj uses `IReplicatedWColSharded`).
- Collective 3 (Q all-gather): eliminated — each device runs SDPA on its 2 local Q heads against the replicated 4 KV heads.
- Collective 4, 5 (K/V all-gathers): still required to replicate KV for paged attention.

**Combined saving:** eliminates Collective 1 (reduce-scatter) and Collective 3 (Q all-gather). Both `index.md` and the data volumes show that Collectives 2 and 3 each move ≈112 KB per device received (at B=32); eliminating Collective 3 alone saves ≈112 KB of link traffic per step.

**Implementation obstacles:**
1. The `paged_sdpa_decode` kernel (called at line 2766–2773 of `attention.py`) is written for fully-replicated Q input (the paged cache has replicated KV on all devices). It would need a `cluster_axis`-aware variant that processes only the local Q head shard.
2. `ttnn.experimental.nlp_concat_heads_decode` (line 2785–2788) expects all H heads on the device; this would need a cross-device concat or a gather of the attention output.
3. The SDPA output `[B, 1, 2 heads, D]` per device must be reassembled into `[B, 1, H*D]` before the dense projection, requiring an additional all-gather of the attention output. This might cost more than the Q all-gather it replaces, depending on the SDPA output volume.

This alternative is technically non-trivial and depends on custom kernel changes. It is worth investigating if the attention computation itself (SDPA) is a bottleneck, but for the collective communication analysis the simpler Alternative 1 is the correct first step.

---

## Expected Latency Impact of Removing One All-Gather per Decode Step

Alternative 1 removes only the reduce-scatter (Collective 1), not any all-gather. The reduce-scatter volume is B × d\_model elements total / N × 2 bytes/element = B × 512 bytes per device received. For B=32: 16 384 bytes ≈ 16 KB.

If Collectives 3 (Q all-gather) and 1 (reduce-scatter) are combined as one logical cost (since they exist as a matched pair only because of the `IColShardedWRowSharded` choice), the effective saving of Alternative 1 is one reduce-scatter: roughly B × d\_model / N × 2 bytes of link traffic eliminated per decode step.

The latency reduction depends on the link bandwidth and scheduling pipeline. At peak application-level inter-device bandwidth of ~50 GB/s (rough estimate for ring all-gather on T3K), 16 KB takes approximately **0.32 µs** to transfer. However, the end-to-end latency of a collective operation includes CCL setup overhead, semaphore synchronization, and kernel dispatch, which together can be several microseconds even for tiny payloads. The actual saving from removing one reduce-scatter is therefore likely in the **1–5 µs** range per decode step (rough estimate; to be validated with profiling per chapter 7 methodology).

For context: at 100 decode steps/second, 5 µs per step = 0.5 ms saved per second, or roughly 0.05% of step budget. The Q all-gather (Collective 3) itself, at ~112 KB per device received at B=32 and ~5–15 µs latency (rough estimate), is a more valuable target; but it is not removable without also removing Collective 1 (they are a matched pair). Removing both together (by switching to Alternative 1 + eliminating the Q all-gather through head-sharded SDPA, i.e., Alternative 3) would give a more meaningful saving.

All estimates above are **rough order-of-magnitude** guidance; actual numbers require profiling with `ttnn.synchronize_device` wrapping individual ops as described in chapter 7.

---

## Navigation

---

**Next:** [Chapter 3 — Memory Layout Transitions and L1 Pressure](../chapter_03_memory_layout_transitions/index.md)
