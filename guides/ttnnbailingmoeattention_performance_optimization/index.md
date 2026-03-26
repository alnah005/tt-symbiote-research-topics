# TTNNBailingMoEAttention Performance Optimization on T3K

## Scope

This guide analyzes the performance-critical decode path of `TTNNBailingMoEAttention._forward_decode_paged` running on a T3K 1×8 Wormhole mesh (N=8 devices). The model is BailingMoeV2 with grouped-query attention: H=16 query heads, Hkv=4 KV heads, head dim D=128, d_model=2048. Every chapter traces a specific bottleneck category back to exact source file locations and proposes concrete fixes.

The guide is written for engineers who have working T3K inference and want to reduce decode-step latency. Profiling guidance and a prioritized roadmap are in Chapter 7.

---

## Chapter Map

| Chapter | Topic | Key bottleneck identified |
|---|---|---|
| [1 — Decode Path Architecture](chapter_01_decode_path_architecture/index.md) | Complete op sequence of `_forward_decode_paged` | Baseline map: 5 collectives, 9 `to_memory_config` calls, 3 host-device transfers |
| [2 — Collective Communication](chapter_02_collective_communication/index.md) | Reduce-scatter + all-gather pair in Q projection | Q projection uses `TTNNLinearIColShardedWRowSharded` (line 2374), producing an unnecessary reduce-scatter; switch to `TTNNLinearIReplicatedWColSharded` to eliminate it |
| [3 — Memory Layout Transitions](chapter_03_memory_layout_transitions/index.md) | 9 `ttnn.to_memory_config` calls, ~528 KB NoC data movement per step | 6 of 9 transitions are avoidable; DRAM→L1 copies before QK norm (lines 2656–2657) are the highest-value targets |
| [4 — Host-Device Round-Trips](chapter_04_host_device_roundtrips/index.md) | `cur_pos_tt` constructed via `ttnn.from_torch` each step | Lines 2678–2685: blocking PCIe transfer every step; persistent on-device tensor eliminates it |
| [5 — QK Normalization](chapter_05_qk_normalization/index.md) | DRAM→L1 transition forced by `_apply_qk_norm` layout constraint | Pre-all-gather Q norm eliminates the 131 KB Q DRAM→L1 copy; K norm cannot be pre-gathered (Hkv/N = 0.5) |
| [6 — Math Fidelity and SDPA Config](chapter_06_math_fidelity/index.md) | `fp32_dest_acc_en=True` limits SDPA to 4 dst tiles | Switching to `fp32_dest_acc_en=False`, `packer_l1_acc=False` (Qwen3 config) enables 8 dst tiles at the same HiFi4 fidelity |
| [7 — Profiling Methodology and Roadmap](chapter_07_profiling_and_roadmap/index.md) | — | Prioritized bottleneck list with impact estimates; `ttnn.synchronize_device` timing methodology; cross-model comparison |

---

## Reading Order

Read Chapter 1 first — it establishes the op sequence and symbolic conventions (H, Hkv, D, N, B) used throughout. Chapters 2–6 can be read independently after Chapter 1. Chapter 7 synthesizes all findings and should be read last.

---

## Quick Reference: Top Optimizations

| Rank | Change | Risk | Estimated full-step speedup |
|---|---|---|---|
| 1 | Switch Q projection to `TTNNLinearIReplicatedWColSharded` | Low | 5–15% |
| 2 | Pre-all-gather Q norm + step 20 transition elimination | Moderate | 3–8% |
| 3 | Persistent `cur_pos_tt` device tensor | Low | 1–3% |
| 4a | `fp32_dest_acc_en=False`, `packer_l1_acc=False` | Low | 3–7% |
| 4b | HiFi2 fidelity (requires accuracy validation) | Medium | 15–25% |

All estimates require profiling validation per Chapter 7. Optimizations 1, 3, and 4a can be implemented and measured independently.

---

## Source File Reference

All line citations refer to the tt-symbiote codebase at the revision current when this guide was written:

- `attention.py`: `TTNNBailingMoEAttention` and `TTNNGlm4MoeLiteAttention` implementations
- `qwen_attention.py`: `TTNNQwen3FullAttention` implementation
- `rope.py`: `BailingRotarySetup.get_cos_sin_for_decode`
- `linear.py`: `TTNNLinearIColShardedWRowSharded` and `TTNNLinearIReplicatedWColSharded` internals
