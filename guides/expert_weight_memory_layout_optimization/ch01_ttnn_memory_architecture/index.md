## Prerequisites

Before reading this chapter you should be comfortable with:

- Creating and running TTNN ops: `ttnn.matmul`, `ttnn.to_device`, `ttnn.from_device`
- Using `ttnn.DRAM_MEMORY_CONFIG` and `ttnn.L1_MEMORY_CONFIG` to control where a tensor lives
- The basic structure of a Mixture-of-Experts (MoE) FFN layer: routing, expert selection, per-expert matrix multiply
- PyTorch tensor shapes and basic dtype handling

You do **not** need to know how TTNN's sharding subsystem works internally, how `ShardSpec` maps to physical DRAM banks, or how the NoC routes requests between cores. This chapter builds that foundation from scratch.

---

## Chapter 1: TTNN Memory Architecture

### Overview

This chapter establishes the physical and software-level memory model you need to reason about weight placement on Wormhole B0 hardware. It answers three questions that every subsequent chapter in this guide depends on:

1. **Where does memory live?** The Wormhole B0 die has a hierarchy of DRAM (off-chip, high capacity) and L1 SRAM (on-chip, per-core). Understanding the physical topology — 6 DRAM controllers, 12 banks, 12 GB total, 1.5 MB L1 per Tensix core — explains why a weight tensor's placement has a measurable effect on throughput.

2. **How do you express placement to TTNN?** The `ttnn.MemoryConfig` class is the single control surface for specifying where a tensor is allocated, how its pages are distributed, and whether it is interleaved or sharded. This chapter covers every field of that API.

3. **What is the difference between interleaved and sharded layout?** These two allocation strategies have different NoC access patterns, different contention profiles, and different interactions with the matmul kernel. This chapter gives you the mental model; later chapters show how to exploit it.

### Learning Objectives

By the end of this chapter you will be able to:

- Describe the Wormhole B0 DRAM and L1 memory topology in terms of controllers, banks, bandwidth, and latency
- Explain how the NoC connects Tensix cores to DRAM and what "NoC hop" means for latency
- Construct a `ttnn.MemoryConfig` from scratch, specifying `buffer_type`, `memory_layout`, and an optional `shard_spec`
- Distinguish interleaved from sharded allocation and articulate which scenarios favor each
- Read `tensor.memory_config()` output and identify whether a tensor is DRAM interleaved, L1 interleaved, or sharded

---

## Navigation

| File | Contents |
|---|---|
| `wormhole_memory_hierarchy.md` | Physical DRAM topology, L1 per core, bandwidth figures, NoC model |
| `memory_config_api.md` | `ttnn.MemoryConfig`, `BufferType`, `TensorMemoryLayout`, predefined configs, code patterns |
| `interleaved_vs_sharded.md` | Page distribution mechanics, NoC contention, the reshard pattern |

Read the files in the order listed above. Each file assumes the concepts introduced in the files before it.

---

## Summary Table: Memory Config Options at a Glance

| Config | Typical use |
|---|---|
| `ttnn.DRAM_MEMORY_CONFIG` | Interleaved DRAM (default for weights) |
| `ttnn.L1_MEMORY_CONFIG` | Single-bank L1 (small tensors only) |

---

## Next Steps

Proceed to `wormhole_memory_hierarchy.md` to understand the physical substrate that these configs operate on. If you are already familiar with Wormhole B0's DRAM topology, you may skip directly to `memory_config_api.md`, but note that the bandwidth and latency figures in `wormhole_memory_hierarchy.md` are referenced in later discussions about shard strategy trade-offs.
