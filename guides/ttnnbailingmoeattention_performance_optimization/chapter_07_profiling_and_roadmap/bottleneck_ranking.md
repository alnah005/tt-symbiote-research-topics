# Bottleneck Ranking and Optimization Roadmap

The following bottlenecks are ranked by estimated latency impact at B=1 decode on T3K. Estimates are rough order-of-magnitude guidance; actual numbers require profiling per `profiling_methodology.md`.

---

## Rank 1 — Redundant Q All-Gather (Chapter 2)

**Root cause:** `TTNNLinearIColShardedWRowSharded` (line 2374 of `attention.py`) produces Q via a reduce-scatter, which is then reconstituted by `_maybe_all_gather(query_states)` (line 2631). The reduce-scatter step is unnecessary because Q does not need to remain sharded — the subsequent all-gather undoes it. `TTNNQwen3FullAttention` avoids the reduce-scatter by using `TTNNLinearIReplicatedWColSharded` for Q (line 255 of `qwen_attention.py`), which takes replicated input and produces col-sharded output without a reduce-scatter.

**Proposed fix:** Switch Q projection to `TTNNLinearIReplicatedWColSharded` (same as K/V at line 2375 and Qwen3 Q at line 255 of `qwen_attention.py`). This eliminates the reduce-scatter in Q projection; the `_maybe_all_gather(query_states)` call at line 2631 remains necessary (Q output is still col-sharded).

**Impact:** Removes the reduce-scatter — 1 fewer collective per decode step (5 → 4 projection-related collectives). The Q all-gather at line 2631 remains.

**Risk:** Low. `TTNNQwen3FullAttention` uses this configuration for Q and is validated on T3K. The model has H=16 Q heads and N=8 devices (2 heads/device), so per-device Q is a complete 2-head slice — compatible with all downstream kernels.

**Reference:** `chapter_02_collective_communication/sharding_alternatives.md` Alternative 1.

---

## Rank 2 — Memory Layout Transitions (Chapter 3)

**Root cause:** Nine `ttnn.to_memory_config` calls per decode step, totalling approximately 528 KB of NoC data movement. Six of the nine are classified as avoidable (approximately 480 KB, 91% of the total). The most costly avoidable transition is the DRAM→L1 copy before QK norm (steps 8a/8b, 163,840 bytes at B=32).

**Proposed fix (highest-value):** Move Q norm before the all-gather (Chapter 5 analysis). This eliminates step 8a (131,072 bytes at B=32). Step 8b (K DRAM→L1, 32,768 bytes) cannot be eliminated with the same approach (Hkv/N = 0.5, K heads split across devices pre-all-gather).

**Proposed fix (second-highest):** Investigate whether step 20 (`attn_output` DRAM→L1 for `nlp_concat_heads_decode`, 131,072 bytes) can be eliminated by adjusting the SDPA output memory config to produce HEIGHT_SHARDED output directly.

**Impact:** Pre-all-gather Q norm alone saves 131,072 bytes/step/device. Step 20 elimination adds another 131,072 bytes. Combined: approximately 262 KB of the 480 KB avoidable total.

**Risk:** Moderate. Pre-all-gather Q norm requires restructuring the call sequence in `_forward_decode_paged` and verifying that `ttnn.rms_norm` operates correctly on the pre-gather shard layout. Step 20 elimination depends on SDPA kernel output format flexibility.

**Reference:** `chapter_03_memory_layout_transitions/avoidable_transitions.md`, `chapter_05_qk_normalization/distributed_alternative.md`.

---

## Rank 3 — `cur_pos_tt` Host Round-Trip (Chapter 4)

**Root cause:** Lines 2674–2685 of `attention.py` construct a host-side `torch.tensor` and transfer it to device via `ttnn.from_torch` on every decode step. This is a blocking PCIe transfer that introduces a scheduling stall.

**Proposed fix:** Maintain a persistent device-resident `cur_pos_tt` tensor updated in-place each step. If `cache_position` arrives as a host integer, increment the on-device position tensor with `ttnn.add` or `ttnn.assign` instead of creating a new `from_torch` tensor each step.

**Impact:** Eliminates one `from_torch` PCIe transfer per step. The tensor is small (1-D int32 of length B=32 = 128 bytes) but the transfer stalls the device pipeline while the host constructs the Python object and initiates the DMA.

**Risk:** Low. The pattern already exists in the codebase: Qwen3 and GLM4 both face the same pattern; `TTNNBailingMoEAttention` already eliminated `_to_replicated` (a similar host round-trip), demonstrating that on-device topology management is feasible.

**Reference:** `chapter_04_host_device_roundtrips/cur_pos_roundtrip.md`.

---

## Rank 4 — Math Fidelity / Accumulator Config (Chapter 6)

**Root cause:** `TTNNBailingMoEAttention` uses `fp32_dest_acc_en=True` and `packer_l1_acc=True`, limiting the SDPA kernel to 4 dst tiles. `TTNNQwen3FullAttention` uses `fp32_dest_acc_en=False` and `packer_l1_acc=False`, enabling 8 dst tiles and potentially higher throughput at HiFi4 fidelity.

**Proposed fix (low risk):** Switch to `fp32_dest_acc_en=False`, `packer_l1_acc=False` (Config B in Chapter 6 terminology). This matches the Qwen3 configuration and has been validated on T3K at HiFi4 fidelity.

**Proposed fix (higher risk):** Additionally switch to HiFi2 fidelity (Config D). Requires accuracy benchmarking per the methodology in `chapter_06_math_fidelity/accuracy_throughput_tradeoff.md`.

**Impact:** Config B alone: estimated 10–20% SDPA speedup, 3–7% full-step speedup. Config D: estimated 50–80% SDPA speedup if accuracy permits.

**Risk:** Config B is low risk (same fidelity, Qwen3-validated). Config D requires accuracy measurement.

**Reference:** `chapter_06_math_fidelity/hifi4_vs_hifi2.md`, `chapter_06_math_fidelity/accuracy_throughput_tradeoff.md`.

---

## Rank 5 — QK Norm Dispatch Overhead (Chapter 5)

**Root cause:** `_apply_qk_norm` dispatches through `TTNNRMSNorm` module objects (`self.query_layernorm`, `self.key_layernorm`) with Python-level `hasattr` checks and conditional typecast guards. `TTNNQwen3FullAttention` uses direct `ttnn.rms_norm(tensor, weight=self._weight, epsilon=...)` calls instead (lines 556, 565 of `qwen_attention.py`), eliminating module dispatch overhead.

**Proposed fix:** Replace the module dispatch in `_apply_qk_norm` with direct `ttnn.rms_norm` calls using pre-stored device weight tensors. The weight tensors (`query_layernorm.tt_weight` and `key_layernorm.tt_weight`) are already on device; expose them at the parent class level and call `ttnn.rms_norm` directly.

**Impact:** Python-level overhead only. For a model running thousands of decode steps, removing two module dispatch calls per step is measurable but smaller than the collective/transition savings above.

**Risk:** Very low. This is a Python-level refactor with no change to device computation.

**Reference:** `chapter_05_qk_normalization/distributed_alternative.md` (Fusion Opportunity section).

---

## Rank 6 — `get_cos_sin_for_decode` Partial Host Op (Chapter 4)

**Root cause:** If `position_ids` arrives as a `ttnn.Tensor` at `BailingRotarySetup.get_cos_sin_for_decode` (line 444 of `rope.py`), an unconditional `ttnn.from_torch` call uploads a position tensor to device. When called from `_forward_decode_paged`, `cache_position_tensor` is already a host `torch.Tensor`, so the device→host copy path is not taken — but the from_torch call still occurs.

**Proposed fix:** Always pass `cache_position_tensor` (already a host tensor) to `get_cos_sin_for_decode` to skip the conditional device→host path. The `from_torch` call in `rope.py` line 444 is then a straightforward host→device transfer that can be pipelined.

**Impact:** Minor. Eliminates a conditional code path; in practice the `from_torch` at line 444 is already executed on every decode step anyway. The optimization value is in removing the conditional logic and the device→host extract path when the input is a ttnn.Tensor.

**Risk:** Very low. Simple refactor.

**Reference:** `chapter_04_host_device_roundtrips/get_cos_sin_host_ops.md`.

---

## Optimization Priority Summary

| Rank | Optimization | Risk | Estimated full-step speedup |
|---|---|---|---|
| 1 | Switch Q projection to `TTNNLinearIReplicatedWColSharded` | Low | 5–15% (1 collective eliminated) |
| 2 | Pre-all-gather Q norm + step 20 transition elimination | Moderate | 3–8% (NoC bandwidth) |
| 3 | Persistent `cur_pos_tt` device tensor | Low | 1–3% (PCIe stall) |
| 4a | `fp32_dest_acc_en=False`, `packer_l1_acc=False` (Config B) | Low | 3–7% (SDPA throughput) |
| 4b | HiFi2 fidelity (Config D, accuracy required) | Medium | 15–25% (SDPA throughput) |
| 5 | Inline K norm dispatch (direct `ttnn.rms_norm`) | Very low | <1% (Python overhead) |
| 6 | `get_cos_sin_for_decode` refactor | Very low | <1% (conditional path) |

Estimates are rough order-of-magnitude guidance; all require profiling validation. Optimizations 1, 3, and 4a can be implemented and measured independently. Optimization 2 has a dependency on 1 (the Q projection refactor changes the pre-gather Q layout). Optimization 4b depends on completing the accuracy measurement methodology from Chapter 6.

---

**Next:** [Cross-Model Comparison](comparison_to_other_implementations.md)
