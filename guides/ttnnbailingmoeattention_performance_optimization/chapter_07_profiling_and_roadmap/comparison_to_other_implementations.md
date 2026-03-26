# Cross-Model Comparison: Attention Implementation Strategies on T3K

## Overview

Three attention implementations co-exist in the tt-symbiote codebase targeting T3K. Comparing them reveals which performance techniques have already been validated and which remain model-specific.

---

## Side-by-Side Comparison

| Dimension | `TTNNBailingMoEAttention` | `TTNNQwen3FullAttention` | `TTNNGlm4MoeLiteAttention` |
|---|---|---|---|
| **Q projection type** | `TTNNLinearIColShardedWRowSharded` (line 2374 of `attention.py`) | `TTNNLinearIReplicatedWColSharded` (line 255 of `qwen_attention.py`) | `TTNNLinearIColShardedWRowSharded` |
| **K/V projection type** | `TTNNLinearIReplicatedWColSharded` (line 2375) | `TTNNLinearIReplicatedWColSharded` | `TTNNLinearIReplicatedWColSharded` |
| **Collective ops per decode step** | 5 (1 reduce-scatter in Q proj + 4 all-gathers) | 6 (1 hidden all-gather + 3 post-proj all-gathers including Q + 2 cos/sin all-gathers; no reduce-scatter) | varies by path |
| **all_gather API** | `ttnn.all_gather` (synchronous, line 2626) | `ttnn.experimental.all_gather_async` + `synchronize_device` | `ttnn.all_gather` (synchronous) |
| **QK norm dispatch** | `TTNNRMSNorm` module (lines 2474–2475) | Direct `ttnn.rms_norm` with pre-loaded weight | `TTNNRMSNorm` or `TTNNDistributedRMSNorm` (configurable via `distributed` flag) |
| **`_to_replicated` host round-trip** | **Not called in decode path** (bypassed; lines 2642–2646 reshape approach) | Called for Q/K/V (lines 766–768 of `qwen_attention.py`) | Called for Q/K/V (lines 1895–1897 of `attention.py`) |
| **`cur_pos_tt` host round-trip** | `ttnn.from_torch` each step (lines 2678–2685) | Same pattern | Same pattern |
| **SDPA math fidelity** | HiFi4 | HiFi4 | HiFi4 |
| **`fp32_dest_acc_en`** | `True` (lines 2434–2440) | `False` (lines 341–346 of `qwen_attention.py`) | `True` |
| **`packer_l1_acc`** | `True` (line 2439) | `False` (line 346 of `qwen_attention.py`) | `True` |
| **Async collective for distributed norm** | No | No | Yes (`TTNNDistributedRMSNorm` uses `all_gather_async` at line 1604 of `attention.py`) |
| **Separate decode program config** | Yes (`decode_program_config` with `q_chunk_size=0`) | No (single config) | No (single config) |

---

## Key Technique: `_to_replicated` Elimination

`TTNNBailingMoEAttention` bypasses `_to_replicated` (lines 2288–2310 of `attention.py`) by using `ttnn.reshape` directly at lines 2642–2646 to produce the `[1, batch, heads, head_dim]` decode format, avoiding the AllGatherMesh→ReplicateTensorToMesh topology conversion. `TTNNQwen3FullAttention` (lines 766–768 of `qwen_attention.py`) and `TTNNGlm4MoeLiteAttention` (lines 1895–1897 of `attention.py`) still call `_to_replicated` and should adopt the reshape-based approach.

---

## Key Technique: Q Projection Strategy

See `bottleneck_ranking.md` Rank 1 for the full analysis.

---

## Key Technique: Async All-Gather

`TTNNQwen3FullAttention` uses `ttnn.experimental.all_gather_async` (with `synchronize_device` called afterwards to wait for completion). This allows the runtime to overlap communication with subsequent compute dispatches. `TTNNBailingMoEAttention` uses synchronous `ttnn.all_gather`, which blocks until the gather completes before the next op is dispatched.

`TTNNGlm4MoeLiteAttention` also uses `ttnn.experimental.all_gather_async` for `TTNNDistributedRMSNorm` (line 1604 of `attention.py`) but retains synchronous `ttnn.all_gather` for projection outputs.

Adopting async all-gather in `TTNNBailingMoEAttention` would require wiring up the `ccl_manager` semaphore infrastructure already present in `TTNNQwen3FullAttention`.

---

## Key Technique: SDPA Compute Config

See `bottleneck_ranking.md` Rank 4 and `chapter_06_math_fidelity/hifi4_vs_hifi2.md` for the full analysis.

---

## Techniques Unique to Each Implementation

| Technique | Present in |
|---|---|
| `_to_replicated` eliminated (reshape-based Q/K/V) | Bailing MoE only |
| `TTNNLinearIReplicatedWColSharded` for Q (no reduce-scatter in Q proj) | Qwen3 only |
| Async all-gather for projections | Qwen3 only |
| `fp32_dest_acc_en=False` + `packer_l1_acc=False` | Qwen3 only |
| Distributed RMSNorm (`TTNNDistributedRMSNorm`) | GLM4 only |
| Separate `decode_program_config` (auto chunk sizes) | Bailing MoE only |

---

## Recommended Adoption Path

For `TTNNBailingMoEAttention`:

1. **Adopt Qwen3's Q projection strategy** (use `TTNNLinearIReplicatedWColSharded` for Q → eliminates the reduce-scatter, removing 1 collective)
2. **Adopt Qwen3's SDPA config** (`fp32_dest_acc_en=False`, `packer_l1_acc=False`)
3. **Propagate Bailing MoE's `_to_replicated` elimination** to Qwen3 and GLM4

For `TTNNQwen3FullAttention` and `TTNNGlm4MoeLiteAttention`:

1. **Adopt the reshape-based Q/K/V construction** from `TTNNBailingMoEAttention` to eliminate `_to_replicated`
