# TTNNBailingMoEAttention Performance Optimization on T3K

This guide covers the key decode-path performance bottlenecks in the `TTNNBailingMoEAttention` module of the Ling (BailingMoeV2) model running on a T3K 8-chip Wormhole mesh, and provides concrete measurement steps and code-level optimization strategies for each. It is written for ML systems engineers and kernel developers who are already familiar with T3K hardware topology, TTNN sharding, and transformer attention mechanics. The eight decode-path research questions are evaluated specifically for the Ling / BailingMoeV2 model on T3K.

---

## Prerequisites

Readers are assumed to be comfortable with the following before working through this guide:

- **T3K mesh topology:** Physical layout of 8 Wormhole n300 chips on a 1×8 logical mesh, Ethernet CCL bandwidth and latency characteristics, and how `num_links` maps to physical Ethernet connections.
- **TTNN programming model:** Tensor memory configs (`TensorMemoryLayout` variants: `HEIGHT_SHARDED`, `WIDTH_SHARDED`, `BLOCK_SHARDED`, `INTERLEAVED`), `ShardSpec`, `MemoryConfig`, and multi-device tensor types (`ReplicateTensorToMesh`, `ShardTensorToMesh`).
- **Basic transformer attention mechanics:** Multi-head attention, grouped-query attention (GQA), rotary position embedding (RoPE), and paged KV-cache management.
- **Python-level model code in `tt-transformers` / `tt-metal` stacks:** Reading TTNN op invocations, following tensor data flow through a decoder layer, and interpreting profiler traces.

Prior exposure to the Ling (BailingMoeV2) model is not required; Chapter 1 provides all necessary model-specific context.

---

## Research Questions Answered

The guide is organized around eight concrete research questions. The table below maps each question to the chapter that answers it.

Research questions addressed by this guide and their corresponding chapters.

| Q# | Research Question | Chapter |
|----|-------------------|---------|
| Q1 | Fused QKV projection latency savings and `num_links` optimality | Chapter 2 |
| Q2 | Host round-trip overhead of `_to_replicated` and device-side alternatives | Chapter 3 |
| Q3 | Memory-config transitions per decode step and dominant overhead | Chapter 4 |
| Q4 | `paged_sdpa_decode` chunk sizes (`q_chunk_size=0`, `k_chunk_size=0`) for Ling's GQA configuration | Chapter 5 |
| Q5 | `HiFi4` vs. `HiFi2` math fidelity trade-off for attention correctness | Chapter 5 |
| Q6 | QK norm latency and L1 move avoidability | Chapter 6 |
| Q7 | Why does `partial_rotary_factor < 1.0` disable `TTNNDistributedRotaryPositionEmbedding` in favor of the non-distributed `TTNNRotaryPositionEmbedding`, what is the performance cost of running non-distributed RoPE on T3K, and are there avoidance strategies? | Chapter 6 |
| Q8 | Best profiling approach for identifying the single biggest decode bottleneck | Chapter 7 |

---

## Chapter Overview

Seven chapters cover the full optimization surface from hardware context through profiling methodology (note: Chapter 5 answers two research questions — Q4 and Q5 — within a single chapter).

| Chapter | Directory | Summary |
|---------|-----------|---------|
| Chapter 1 | `ch1_model_and_hardware_context` | Ling model attention structure and T3K hardware topology |
| Chapter 2 | `ch2_fused_qkv_projection` | Fused QKV projection mechanics, latency savings, and `num_links` tuning |
| Chapter 3 | `ch3_host_roundtrip_replication` | `_to_replicated` host round-trip overhead and device-side alternatives |
| Chapter 4 | `ch4_memory_config_transitions` | Memory-config transition catalog, cost model, and elimination opportunities |
| Chapter 5 | `ch5_sdpa_and_compute_config` | Paged SDPA chunk sizes and `HiFi4` vs. `HiFi2` math fidelity |
| Chapter 6 | `ch6_rope_and_qk_norm` | QK norm latency and partial-rotary RoPE cost on T3K |
| Chapter 7 | `ch7_profiling_and_bottleneck_identification` | Profiling methodology and bottleneck decision tree |

---

## How to Use This Guide

**First-time readers** should read the chapters in order. Chapter 1 establishes the model and hardware context that all subsequent chapters depend on. Chapters 2 through 6 each address a specific optimization area in the decode path, building on concepts introduced in prior chapters. Chapter 7 ties everything together with profiling methodology and a decision tree that maps measurement outcomes to the appropriate optimization chapter.

**Experienced users** who already have a profiling hypothesis or want to identify the dominant bottleneck before studying individual optimizations should jump directly to Chapter 7 (`ch7_profiling_and_bottleneck_identification`). The bottleneck decision tree in that chapter will direct you to the most relevant optimization chapter (2 through 6) based on your profiling results. Return to Chapter 1 only if you need a refresher on the Ling model structure or T3K topology.

---

**Start here:** [Chapter 1 — Model and Hardware Context](ch1_model_and_hardware_context/index.md)
