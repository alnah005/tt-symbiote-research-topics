# Chapter 1: Qwen3.5-27B Architecture and Hardware Mapping

This chapter introduces the Qwen3.5-27B model architecture and explains how it maps onto the Tenstorrent P150x4 platform with TP=4 tensor parallelism across four Blackhole chips.

## Files

| File | Description |
|------|-------------|
| [`hybrid_architecture.md`](./hybrid_architecture.md) | The 48 GDN + 16 full attention layer structure, model dimensions, and the `Transformer` class construction flow |
| [`tp_sharding_strategy.md`](./tp_sharding_strategy.md) | TP=4 dimension splits, column-parallel vs row-parallel projections, weight preparation helpers, and CCL topology |

## Process Files

The following files are internal pipeline artifacts and are not part of the reader guide:
- `b_review.md` — Agent B review notes
- `compression_analysis.md` — Agent C compression analysis

---

**Next:** [`hybrid_architecture.md`](./hybrid_architecture.md)
