# Chapter 2: DRAM-Sharded Memory Layout

This chapter provides a complete treatment of DRAM-sharded memory configuration in TTNN. You will learn how `ttnn.ShardSpec` describes the geometry of a sharded buffer, how to choose among the three sharding strategies for expert weight tensors, and how to construct a valid `MemoryConfig` targeting DRAM banks instead of L1 cores.

**Prerequisites:** Chapter 1 — [TTNN Memory Architecture](../ch01_ttnn_memory_architecture/)

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Describe the three fields of `ttnn.ShardSpec`: `grid` (CoreRangeSet), `shape` (shard dimensions in elements), and `orientation` (ROW_MAJOR or COL_MAJOR).
2. Construct a `CoreRange` and `CoreRangeSet` targeting a specific rectangular core grid.
3. Distinguish HEIGHT_SHARDED, WIDTH_SHARDED, and BLOCK_SHARDED strategies and select the right one for a given expert weight tensor shape.
4. Explain why DRAM-sharded buffers use DRAM banks (not L1 cores) as the shard target, and how that reduces NoC contention.
5. Build a complete `MemoryConfig` for DRAM-sharded placement of an expert weight tensor using the `ttnn.MemoryConfig` constructor directly.
6. Verify a tensor's memory configuration using `tensor.memory_config()` and `tensor.shard_spec()`.

---

## ShardSpec Quick Reference

For the full ShardSpec field reference, see `shard_spec_deep_dive.md`.

---

## Chapter Contents

| File | Topic |
|---|---|
| `shard_spec_deep_dive.md` | `ttnn.ShardSpec`, `CoreCoord`, `CoreRange`, `CoreRangeSet`, shard shape arithmetic |
| `sharding_strategies.md` | HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED — mechanics, use cases, comparison |
| `constructing_dram_sharded_config.md` | Step-by-step construction of a DRAM-sharded `MemoryConfig`, `to_memory_config`, verification, common mistakes, end-to-end example |

**Recommended reading order:** `shard_spec_deep_dive.md` → `sharding_strategies.md` → `constructing_dram_sharded_config.md`
