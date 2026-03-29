# Chapter 1: Qwen3.5-27B Architecture and Hardware Mapping

This chapter introduces the Qwen3.5-27B model architecture and explains how it maps onto the Tenstorrent P150x4 platform with TP=4 tensor parallelism across four Blackhole chips.

## Learning Objectives

After reading this chapter you will understand:

- The hybrid 64-layer structure: 48 Gated DeltaNet (GDN) layers and 16 full attention layers arranged in a repeating 3+1 pattern
- The key model dimensions and how GDN layers replace KV caches with fixed-size recurrence states
- How weights and activations are sharded across 4 devices, including column-parallel and row-parallel projection strategies
- The weight preparation helpers that reorder tensors for clean TP slicing

## Files

| File | Description |
|------|-------------|
| [`hybrid_architecture.md`](./hybrid_architecture.md) | The 48 GDN + 16 full attention layer structure, model dimensions, and the `Transformer` class construction flow |
| [`tp_sharding_strategy.md`](./tp_sharding_strategy.md) | TP=4 dimension splits, column-parallel vs row-parallel projections, weight preparation helpers, and CCL topology |

See [hybrid_architecture.md](./hybrid_architecture.md) for complete model dimensions.

---

**Next:** [`hybrid_architecture.md`](./hybrid_architecture.md)
