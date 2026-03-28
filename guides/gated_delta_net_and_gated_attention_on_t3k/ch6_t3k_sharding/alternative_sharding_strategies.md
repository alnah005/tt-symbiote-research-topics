# Alternative Sharding Strategies

This section analyzes two alternatives to head-parallel sharding: replicated state and sequence-parallel sharding. Neither is recommended for the reasons described below. A comparison table closes the section.

---

## 1. Replicated State

**Strategy:** All 8 devices hold the complete state tensor `[B, H_v, d_k, d_v] = [B, 32, 128, 128]`. Each device independently computes all 32 heads of the recurrent step. Input projections are still column-sharded (each device computes its portion of Q, K, V and aggregates via all-gather before the recurrent step). Output projections are similarly sharded.

**Memory per device:**

```
Full state per layer (B=1):  1,048,576 bytes = 1 MiB
Total across 30 layers:      30 MiB per device
```

30 MiB is well within the 3.25 GB budget.

**CCL requirement:** If input projections are column-sharded and each device only has a partial QKV output, the device must all-gather the full `[B, T, 8192]` input before the recurrent step can run on all 32 heads. This all-gather is large:

```
All-gather input (per layer, B=1, T=1):
  8,192 elements × 2 bytes = 16,384 bytes = 16 KiB
  Time at 25 GB/s: 16,384 / 25 × 10^9 = 655 ns ≈ 0.66 µs per layer
```

Alternatively, if the input projections are replicated (each device holds the full weight matrix), the all-gather can be eliminated, but the weight memory doubles.

**Consistency concern:** If each device computes all 32 heads from a local copy of the state, the state replicas must be synchronized on every update. During decode this is automatic (all devices run the same recurrence on the same input), but for batch sizes B > 1 with different sequences per device, or in any configuration where inputs diverge, the replicas will differ. Head-parallel sharding avoids this by design: each device owns and updates only its 4 heads.

**Verdict:** Viable at small B with replicated weights, but the synchronization requirement for B > 1 is fragile. Head-parallel sharding is cleaner and uses less per-device memory in the weight-replicated variant.

---

## 2. Sequence-Parallel State

**Strategy:** Partition the sequence dimension T across devices. Device i processes tokens in range `[i × T/N, (i+1) × T/N)`.

**Why this does not apply to Gated Delta Net:** The recurrent state S_t is updated sequentially — each step depends on S_{t-1}. Partitioning the sequence across devices requires passing S between devices at every chunk boundary. For a 1×8 linear mesh with N=8 devices and T=8192:

```
Chunk boundaries: 8 boundaries per prefill sequence
State transfer per boundary: [B, H_v, d_k, d_v] in BF16
  = 1 × 32 × 128 × 128 × 2 = 1,048,576 bytes = 1 MiB
Time per transfer at 25 GB/s: 1,048,576 / 25 × 10^9 ≈ 41.9 µs
Total boundary cost: 8 × 41.9 µs ≈ 335 µs per layer per prefill
```

For comparison, the compute cost of one prefill pass at T=8192 is:

```
Prefill time (state + KQV I/O, 32 heads): 536,870,912 / 288 × 10^9 ≈ 1,864 µs per layer
```

State boundary transfers would add 335 µs / 1,864 µs ≈ 18% overhead per layer — a significant penalty for no memory saving (each device still holds a full 1 MiB state and must reconstruct the complete state at each boundary). Sequence parallelism has no useful application to recurrent state computation.

---

## 3. Head-Blocked vs. Head-Interleaved Sharding

Within head-parallel sharding, heads can be assigned to devices in two ways:

- **Blocked:** device 0 gets heads 0–3, device 1 gets heads 4–7, ..., device 7 gets heads 28–31.
- **Interleaved:** device 0 gets heads 0, 8, 16, 24; device 1 gets heads 1, 9, 17, 25; ...

For state matrix access (reading and writing `[d_k, d_v]` per head), the two layouts are equivalent in FLOPs and bytes. However, blocked sharding produces **contiguous memory access patterns** in the state tensor: each device's 4-head state occupies a single contiguous slice of the weight matrix, which is more cache-friendly for DRAM streaming and avoids strided access patterns that can reduce GDDR6 throughput on Wormhole.

**Recommendation:** Use blocked (contiguous) head assignment. This matches the existing `TTNNLinearIReplicatedWColSharded` strategy in `qwen_attention.py`, where column sharding over the output dimension is naturally blocked.

---

## 4. Comparison Table

| Strategy | State per device | CCL per decode layer | State sync needed | Recommended |
|---|---|---|---|---|
| **Head-parallel (blocked)** | 128 KiB × 30 ≈ 3.75 MiB | **4 KiB (output all-gather)** | No | **Yes** |
| Replicated (weight-replicated) | 1 MiB × 30 = 30 MiB | None (no input all-gather) | At B > 1 | No |
| Replicated (col-sharded input) | 1 MiB × 30 = 30 MiB | 16 KiB (input all-gather) | At B > 1 | No |
| Sequence-parallel | 1 MiB × 30 = 30 MiB | 2 MiB × 8 boundaries | No | No |

The head-parallel strategy minimizes CCL cost, avoids state synchronization concerns, and matches the existing TP column-sharding implementation pattern. It is the recommended choice.

---

**Previous:** [`head_parallel_state_sharding.md`](./head_parallel_state_sharding.md) | **Next:** [`kv_cache_sharding_for_gated_attention.md`](./kv_cache_sharding_for_gated_attention.md)
