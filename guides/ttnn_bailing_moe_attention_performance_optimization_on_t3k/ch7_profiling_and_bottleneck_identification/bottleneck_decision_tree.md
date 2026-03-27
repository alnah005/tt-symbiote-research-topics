# Bottleneck Decision Tree

After running TTNN op timers (see [`ttnn_op_timers.md`](./ttnn_op_timers.md)) or a Tracy capture (see [`tracy_profiling.md`](./tracy_profiling.md)) on a warm decode step, this file tells you which optimization to apply first. The decision tree maps every plausible dominant contributor to the chapter that addresses it, provides concrete example scenarios, and closes with the recommended iteration loop for systematic performance improvement.

## 1. The Decision Tree

Use the profiling output to answer each question in order. Stop at the first matching branch.

```
Start: sort op timer CSV by host_duration_us descending (post-warm-up, PROFILER_SYNC=1)
│
├─ Is ttnn::paged_sdpa_decode the #1 entry AND it accounts for >40% of T_total?
│    └─ YES → Go to Branch A: SDPA Kernel Bottleneck
│
├─ Is ttnn::all_reduce the #1 or #2 entry AND CCL time + QKV matmul together >35% of T_total?
│    └─ YES → Go to Branch B: CCL All-Reduce / QKV Projection Bottleneck
│
├─ Is (ConcatMeshToTensor + ttnn::from_torch) together >15% of T_total?
│    └─ YES → Go to Branch C: Host Round-Trip Bottleneck
│
├─ Is the sum of all ttnn::to_memory_config rows >20% of T_total?
│    └─ YES → Go to Branch D: Memory-Config Transition Bottleneck
│
├─ Is (ttnn::rms_norm × 2 + ttnn::reshape × 8) together >15% of T_total?
│    └─ YES → Go to Branch E: QK Norm Bottleneck
│
├─ Is ttnn::rotary_embedding >10% of T_total?
│    └─ YES → Go to Branch F: Non-Distributed RoPE Bottleneck
│
├─ Is ttnn::paged_fill_cache in the top 3 ops by duration?
│    └─ YES → Go to Branch H: Paged Fill Cache Bottleneck
│
└─ No single op cluster dominates (all <10% of T_total individually)
     └─ Go to Branch G: Distributed / Long-Tail Bottleneck
```

### Branch A — SDPA Kernel Bottleneck

**Symptom:** `ttnn::paged_sdpa_decode` appears at rank 1 and accounts for ≥40% of total attention decode time. In the Tracy timeline, the device lane bars for `paged_sdpa_decode` are long relative to all other kernel bars, and the host is idle while the device executes.

**Root cause:** The attention computation itself is the dominant cost. This is the expected steady-state condition at long sequence lengths (context ≥ 2048 tokens [ESTIMATE]), where KV cache reads dominate.

**Optimization path:** Chapter 5.

- First, confirm whether `q_chunk_size=0` and `k_chunk_size=0` are using the kernel's default tiling. Experiment with explicit chunk sizes to determine whether a different tiling strategy reduces register pressure or improves DRAM bandwidth utilization (see Chapter 5, [`paged_sdpa_chunk_sizes.md`](../ch5_sdpa_and_compute_config/paged_sdpa_chunk_sizes.md)).
- Second, evaluate whether `HiFi4` fidelity can be reduced to `HiFi2` for the QK dot-product accumulation. This is the single highest-impact configuration change available within the SDPA kernel (see Chapter 5, [`math_fidelity_tradeoff.md`](../ch5_sdpa_and_compute_config/math_fidelity_tradeoff.md)).
- Expected improvement from `HiFi4` → `HiFi2`: 15–40% reduction in `paged_sdpa_decode` kernel time [ESTIMATE].

**Validation:** After the change, re-run op timers and confirm `ttnn::paged_sdpa_decode` `host_duration_us` decreased. Also run a numeric accuracy test (perplexity on a validation set or cosine similarity between `HiFi4` and `HiFi2` attention outputs) before committing to `HiFi2`.

---

### Branch B — CCL All-Reduce / QKV Projection Bottleneck

**Symptom:** `ttnn::all_reduce` is at rank 1 or 2, and together `ttnn::all_reduce` + `ttnn::linear` (QKV) exceed 35% of total decode time. In Tracy, the all-reduce spans on all 8 device lanes are long (>80 µs [ESTIMATE]) and synchronized.

**Root cause:** The fused QKV projection's all-reduce step is communication-bound. At `num_links=1`, only one Ethernet path carries the all-reduce traffic. Increasing `num_links` exposes additional bandwidth.

**Optimization path:** Chapter 2.

- Check current `num_links` setting in the `TTNNLinearIColShardedWAllReduced` constructor. If it is 1, try `num_links=2` and measure whether all-reduce time drops proportionally (see Chapter 2, [`num_links_tuning.md`](../ch2_fused_qkv_projection/num_links_tuning.md)).
- Verify that the QKV weights are stored as a single fused weight matrix (column-sharded across 8 chips) rather than three separate projections. If not, the matmul itself is contributing avoidable overhead (see Chapter 2, [`fusion_mechanics.md`](../ch2_fused_qkv_projection/fusion_mechanics.md)).
- Expected improvement from `num_links=1` → `num_links=2` on 4096 hidden size all-reduce: 25–50% reduction in `ttnn::all_reduce` time [ESTIMATE], depending on whether the all-reduce is latency-bound or bandwidth-bound at this tensor size.

**Validation:** Re-run op timers. Confirm `ttnn::all_reduce` time decreased and `ttnn::linear` (QKV) time is unchanged (fusion was not affected).

---

### Branch C — Host Round-Trip Bottleneck

**Symptom:** The combined time of `ConcatMeshToTensor` + `ttnn::from_torch` (the `_to_replicated` host round-trip) exceeds 15% of total decode time. In the Tracy timeline, all 8 device lanes are idle simultaneously for a window of 7–25 µs [ESTIMATE], with the CPU main thread active.

**Root cause:** After the fused QKV all-reduce, the tensor must be replicated to all chips for the paged-attention kernel. The current implementation goes through host DRAM. At batch=1 and short sequence lengths, this PCIe round-trip is a material fraction of the decode budget.

**Optimization path:** Chapter 3.

- Evaluate whether the `_to_replicated` call can be replaced by a device-side `ttnn::all_gather` with `ReplicateTensorToMesh` output config, which performs the replication entirely on-device via Ethernet CCL without host involvement (see Chapter 3, [`device_side_alternatives.md`](../ch3_host_roundtrip_replication/device_side_alternatives.md)).
- If the paged-attention kernel accepts a replicated TTNN tensor without the host round-trip, this optimization eliminates the host idle gap entirely. Feasibility depends on the kernel's input memory config constraints; verify against the kernel source before implementing.
- Expected improvement: elimination of the 6.8–25.2 µs [ESTIMATE] round-trip, replaced by a device-side all-gather of comparable or lower latency depending on `num_links` and tensor size.

**Validation:** In the Tracy timeline after the change, confirm that the host dead zone (all 8 device lanes idle) between QKV projection and the next device op is absent or substantially shorter.

---

### Branch D — Memory-Config Transition Bottleneck

**Symptom:** Summing all `ttnn::to_memory_config` rows in the op timer CSV gives a value exceeding 20% of total decode time. The individual transitions may each appear small, but their aggregate is substantial. Alternatively, a single `ttnn::to_memory_config` row appears in the top 5 individually.

**Root cause:** The decode path accumulates six or more memory-config transitions (see Chapter 4, [`decode_tensor_lifecycle.md`](../ch4_memory_config_transitions/decode_tensor_lifecycle.md)). Each transition invokes a data-movement kernel that traverses L1 → DRAM → L1 or equivalent. The aggregate cost is 150–171 µs [ESTIMATE] for the full decode path.

**Optimization path:** Chapter 4.

- Identify which transition dominates using the `metadata` column in the op timer CSV (source and destination memory config are encoded there). The most expensive individual transition is typically the Q tensor transition from DRAM INTERLEAVED to HEIGHT_SHARDED for SDPA, costing ~64 µs [ESTIMATE] (Priority 1 in Chapter 4, [`optimization_opportunities.md`](../ch4_memory_config_transitions/optimization_opportunities.md)).
- Check whether the upstream op that produces the Q tensor can be configured to output directly in HEIGHT_SHARDED format, eliminating the transition entirely. If `TTNNRMSNorm` or the post-RoPE eviction can target HEIGHT_SHARDED SDPA layout directly, the Priority 1 transition is removed.
- The combined savings from the Priority 1 transition and the K-entry transition is 83–86 µs [ESTIMATE] per decode step (see Chapter 4, [`optimization_opportunities.md`](../ch4_memory_config_transitions/optimization_opportunities.md)).

**Validation:** After eliminating a transition, confirm in the op timer CSV that the corresponding `ttnn::to_memory_config` row is absent (not merely faster), and verify correctness of the downstream op's output.

---

### Branch E — QK Norm Bottleneck

**Symptom:** The combined time of `ttnn::rms_norm` (×2) + `ttnn::reshape` (×8) exceeds 15% of total decode time. These ten ops (8 reshapes + 2 RMSNorm calls) all appear in the top half of the sorted report.

**Root cause:** The `use_qk_norm=True` path applies per-token `TTNNRMSNorm` to both Q and K tensors after RoPE. Each norm call requires four reshape ops: (1,1,16,128)→(1,16,128)→(16,128) before TTNNRMSNorm and (16,128)→(1,16,128)→(1,1,16,128) after, for a total of 4 reshape ops per QK norm pair and potentially an L1 move, for a combined cost of 74–92 µs [ESTIMATE] (see Chapter 6, [`qk_norm_latency.md`](../ch6_rope_and_qk_norm/qk_norm_latency.md)).

**Optimization path:** Chapter 6.

- Determine whether the reshape and L1 move are mandated by `TTNNRMSNorm`'s input constraints or are precautionary. If the norm kernel can accept a 3D HEIGHT_SHARDED tensor directly, the 4 reshape ops are eliminable.
- If `TTNNRMSNorm` cannot be fused with reshape, consider whether an in-place norm variant avoids the L1 move, reducing the transition cost from ~21 µs (Q) + ~11 µs (K) [ESTIMATE] per direction.

**Validation:** Re-run op timers. Confirm all eight `ttnn::reshape` rows are absent (4 from Q norm + 4 from K norm) from the report (if reshape was eliminated) and that `ttnn::rms_norm` duration is unchanged or reduced.

---

### Branch F — Non-Distributed RoPE Bottleneck

**Symptom:** `ttnn::rotary_embedding` appears in the top 10 and accounts for >10% of total decode time. In the Tracy timeline, only one or a small subset of device lanes are active during the RoPE kernel — not all 8 simultaneously.

**Root cause:** `partial_rotary_factor=0.5` forces `TTNNRotaryPositionEmbedding` (non-distributed) instead of `TTNNDistributedRotaryPositionEmbedding`. The non-distributed kernel runs on a subset of the mesh without exploiting all 8 chips in parallel (see Chapter 6, [`partial_rotary_rope.md`](../ch6_rope_and_qk_norm/partial_rotary_rope.md)).

**Optimization path:** Chapter 6.

- Evaluate whether the cos/sin tables can be padded to full `head_dim=128` (i.e., effectively using `partial_rotary_factor=1.0` for the embedding tables while zeroing out the upper 64 elements post-embedding or masking them in the attention score). If so, `TTNNDistributedRotaryPositionEmbedding` becomes applicable and the kernel can exploit all 8 chips.
- Alternatively, apply the distributed RoPE to all 128 `head_dim` elements and then slice away the unrotated upper half, if the downstream SDPA kernel accepts the result. This trades a slice op for the non-distributed execution penalty.
- At batch=1 with `head_dim=128` and `rotary_dim=64`, the non-distributed RoPE cost is small relative to SDPA [ESTIMATE], so this optimization is lower priority than Branches A–D unless the measurement clearly shows it in the top 3.

**Validation:** Verify that the `ttnn::distributed_rotary_embedding` op replaces `ttnn::rotary_embedding` in the op timer CSV, and that the op now appears on all 8 device lanes in the Tracy timeline simultaneously.

---

### Branch G — Distributed / Long-Tail Bottleneck

**Symptom:** No single op or op cluster accounts for more than 10% of T_total. The decode time is spread across many small contributions.

**This is a success condition.** It means the attention layer is well-balanced across the ops identified in Chapters 2–6. At this point, further optimization requires either:

1. **Cross-layer fusion:** examining whether ops at the boundaries of the attention module (e.g., the residual add and layer norm in the decoder stack) can be fused with the first or last attention op to reduce kernel launch overhead.
2. **Kernel-level optimization:** profiling with Tracy's hardware counter view to look for cache miss rate, occupancy, or memory bandwidth saturation inside specific kernels — a task that typically requires kernel engineering rather than model-level changes.
3. **Batch size scaling:** re-running the profile at batch=4 or batch=8, where the balance of compute-bound vs. communication-bound ops shifts, to identify whether the current configuration scales efficiently.

---

### Branch H — Paged Fill Cache Bottleneck

**Symptom:** `ttnn::paged_fill_cache` appears in the top 3 ops by duration in the sorted op timer CSV.

**Root cause:** `ttnn::paged_fill_cache` writes K and V tokens into the paged KV-cache on device. At decode batch=1, this is typically a single slot write and should be fast. If it dominates, one or more of the following may be true:

- **(a) KV cache pages are in DRAM and the write is bandwidth-limited.** If the KV cache was allocated in DRAM rather than L1, each slot write incurs a DRAM access. At large `block_size` values, the effective write volume per call is amplified.
- **(b) Cache page size / `block_size` parameter is suboptimal.** A very large `block_size` causes each `paged_fill_cache` call to write more data than necessary for a single decode token, wasting bandwidth.
- **(c) There is unexpected sequence-length growth at decode time.** If `seq_len` at the call site is larger than 1 (e.g., due to a bug in position tracking or a re-encode path being triggered), `paged_fill_cache` writes multiple tokens per step, multiplying its cost.

**Optimization path:**

- Check the KV cache memory config: verify that pages are allocated in L1 (or the intended tier) rather than falling back to DRAM unexpectedly. Inspect the `metadata` column for `paged_fill_cache` rows to confirm the memory config.
- Verify `block_size` in the paged KV-cache configuration. A value in the range 16–64 is typical for decode; a value of 512 or larger is a signal of a misconfiguration.
- Confirm that `seq_len` at the `paged_fill_cache` call site is 1 during decode (not the full context length). Add a debug assert or log the tensor shape from the `metadata` column.

**Validation:** After correcting the memory config or `block_size`, re-run op timers and confirm that `ttnn::paged_fill_cache` is no longer in the top 3. Also verify that KV cache output correctness is unchanged by comparing attention outputs against a known-good baseline.

---

## 2. Example Scenarios

### Scenario 1: Short-Context Decode Dominated by Host Round-Trip (Branch C)

**Setup:** Decode step at batch=1, context length = 128 tokens.

**Observed op timer output:**

Example scenario — short-context decode, context=128 tokens [ESTIMATE]

| Rank | `op_name` | `host_duration_us` [ESTIMATE] | % of total |
|---|---|---|---|
| 1 | `ConcatMeshToTensor` | 18 µs | 22% |
| 2 | `ttnn::all_reduce` | 15 µs | 18% |
| 3 | `ttnn::from_torch` | 10 µs | 12% |
| 4 | `ttnn::paged_sdpa_decode` | 12 µs | 14% |
| 5 | `ttnn::linear` (QKV) | 8 µs | 10% |
| ... | ... | ... | ... |
| — | **Total** | **83 µs** | 100% |

**Decision tree path:** `ConcatMeshToTensor` + `ttnn::from_torch` = 28 µs = 34% of total → Branch C.

**Action:** Investigate device-side `ttnn::all_gather` with `ReplicateTensorToMesh` as a replacement for `_to_replicated`. At short context lengths, the device-side all-gather is expected to be faster than the PCIe round-trip because the tensor is small. For the Ling model (hidden_size=4096, 16 Q heads + 4 K heads + 4 V heads, head_dim=128), the fused QKV output per token is:

- Q: 16 heads × 128 = 2,048 elements
- K: 4 heads × 128 = 512 elements
- V: 4 heads × 128 = 512 elements
- Total: 3,072 elements per token → shape (1, 1, 1, 3072)
- At BF16: 3,072 × 2 = 6,144 bytes = 6 KB per chip (after all-reduce each chip holds the full fused output)

At 6 KB per chip, Ethernet latency is lower than PCIe round-trip latency for small payloads [ESTIMATE].

---

### Scenario 2: Long-Context Decode Dominated by SDPA (Branch A)

**Setup:** Decode step at batch=1, context length = 4096 tokens.

**Observed op timer output:**

Example scenario — long-context decode, context=4096 tokens [ESTIMATE]

| Rank | `op_name` | `host_duration_us` [ESTIMATE] | % of total |
|---|---|---|---|
| 1 | `ttnn::paged_sdpa_decode` | 280 µs | 47% |
| 2 | `ttnn::all_reduce` | 65 µs | 11% |
| 3 | `ttnn::linear` (QKV) | 40 µs | 7% |
| 4 | `ttnn::rms_norm` (Q) | 38 µs | 6% |
| 5 | `ttnn::rms_norm` (K) | 22 µs | 4% |
| ... | ... | ... | ... |
| — | **Total** | **595 µs** | 100% |

**Decision tree path:** `paged_sdpa_decode` = 280 µs = 47% of total → Branch A.

**Action:** Switch from `HiFi4` to `HiFi2` fidelity in the SDPA `ComputeKernelConfig`. Run accuracy validation (cosine similarity of attention output between `HiFi4` and `HiFi2` across 100 decode steps on the Ling validation set) before deploying. Expected outcome: `paged_sdpa_decode` time drops to 170–240 µs [ESTIMATE], shifting total from ~595 µs to ~480 µs, a 19% end-to-end improvement.

---

### Scenario 3: Memory Transitions Dominating After CCL Fix (Branch D)

**Setup:** After applying Branch B fix (`num_links=2`), re-profiling reveals a new top contributor.

**Observed op timer output (post-CCL fix):**

Example scenario — post-CCL fix, memory transitions now visible [ESTIMATE]

| Rank | `op_name` | `host_duration_us` [ESTIMATE] | % of total |
|---|---|---|---|
| 1 | `ttnn::paged_sdpa_decode` | 280 µs | 51% |
| 2 | `ttnn::all_reduce` | 38 µs | 7% | ← improved from 65 µs
| 3 | `ttnn::to_memory_config` (Q→SDPA) | 62 µs | 11% |
| 4 | `ttnn::linear` (QKV) | 40 µs | 7% |
| 5 | `ttnn::to_memory_config` (K→SDPA) | 35 µs | 6% |
| ... | ... | ... | ... |
| — | Total `to_memory_config` rows | 115 µs | 21% |
| — | **Total** | **550 µs** | 100% |

**Decision tree path:** Sum of all `to_memory_config` rows = 115 µs = 21% of total → Branch D.

**Action:** Examine the Priority 1 transition (Q tensor DRAM INTERLEAVED → HEIGHT_SHARDED for SDPA, 62 µs). Determine whether `TTNNRMSNorm` can be configured to output directly in HEIGHT_SHARDED SDPA format, eliminating the downstream transition. If successful, save ~62 µs per step, reducing total from 550 µs to ~488 µs.

---

## 3. Recommended Iteration Loop

The systematic process for driving `TTNNBailingMoEAttention` decode latency to its minimum on T3K is:

```
┌─────────────────────────────────────────────────────────────────┐
│  PROFILE  →  IDENTIFY  →  CHANGE  →  VALIDATE  →  REPEAT       │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Profile (warm run)

Run TTNN op timers with `TT_METAL_PROFILER_SYNC=1` after ≥5 warm-up decode steps. Save the CSV with a descriptive name including the configuration being tested:

```bash
cp /tmp/attn_profile/op_perf_results.csv \
   ./profiles/baseline_hifi4_num_links1_ctx4096.csv
```

### Step 2: Identify the top bottleneck

Sort the CSV by `host_duration_us` descending. Apply the decision tree in Section 1. Note the bottleneck op name, its duration, and its percentage of total. Write this down before making any change.

### Step 3: Change exactly one variable

Apply the single optimization recommended by the decision tree branch. Do not combine multiple changes in one iteration — it makes it impossible to attribute the improvement (or regression) to the correct change.

Examples of single-variable changes:
- Change `math_fidelity` in `ComputeKernelConfig` from `MathFidelity.HiFi4` to `MathFidelity.HiFi2`
- Change `num_links=1` to `num_links=2` in the `TTNNLinearIColShardedWAllReduced` constructor
- Replace `_to_replicated` with a device-side `ttnn::all_gather` call
- Change the output `MemoryConfig` of `TTNNRMSNorm` from DRAM INTERLEAVED to HEIGHT_SHARDED

### Step 4: Re-profile (warm run)

Repeat Step 1 with the new configuration. Save the CSV with the change encoded in the filename:

```bash
cp /tmp/attn_profile/op_perf_results.csv \
   ./profiles/hifi2_num_links1_ctx4096.csv
```

### Step 5: Compare and validate

Compute the delta:

```python
import pandas as pd

baseline = pd.read_csv("./profiles/baseline_hifi4_num_links1_ctx4096.csv")
new_run  = pd.read_csv("./profiles/hifi2_num_links1_ctx4096.csv")

baseline_total = baseline["host_duration_us"].sum()
new_total      = new_run["host_duration_us"].sum()

print(f"Baseline total: {baseline_total:.1f} µs")
print(f"New total:      {new_total:.1f} µs")
print(f"Change:         {new_total - baseline_total:+.1f} µs  "
      f"({100*(new_total/baseline_total - 1):+.1f}%)")
```

If the target op improved but another op regressed unexpectedly, investigate before proceeding. Regressions in unrelated ops indicate an indirect effect (e.g., a memory config change causing a downstream op to receive a suboptimal layout).

Run correctness validation (model output comparison against a CPU reference or a known-good T3K baseline) whenever changing math fidelity, memory configs, or replacing a host-side op with a device-side equivalent.

### Step 6: Repeat

Return to Step 2 with the updated profile. Reapply the decision tree. The top bottleneck may have shifted to a different branch — this is expected and desirable. Continue until the decode step meets the latency target or until the optimization is in Branch G (no single contributor >10%), at which point further improvement requires kernel-level work.

---

**End of guide.** Return to [Guide Index](../index.md)
