# Common Pitfalls

This file documents the five failure modes most frequently encountered when sharding expert weight tensors on Wormhole B0. Each pitfall is described with its root cause, the symptom it produces, and the corrective action.

---

## Pitfall 1: Using ROW_MAJOR_LAYOUT for Weights That Will Be Sharded

**Description.** A tensor in `ttnn.ROW_MAJOR_LAYOUT` stores elements densely in row order with no tile structure. When you attempt to shard such a tensor using a TILE_LAYOUT-oriented `MemoryConfig`, the tile-alignment rules in `shard_shape_alignment_rules.md` do not apply in the way you expect because the internal buffer layout is different.

**Symptom.** The `ShardSpec` may appear to accept the configuration, but downstream `ttnn.matmul` calls will raise an error because the matmul kernel requires `TILE_LAYOUT` input. If the layout mismatch is caught earlier, you see:

```
RuntimeError: Input tensor must be in TILE_LAYOUT for matmul
```

**Root cause.** `ttnn.matmul` on Wormhole B0 operates exclusively on TILE_LAYOUT inputs. ROW_MAJOR weights bypass the tile packing that the matrix engine requires.

**Fix.** Always convert to TILE_LAYOUT before constructing the sharded `MemoryConfig`:

```python
# Wrong: sharding a ROW_MAJOR tensor
weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
# weight.layout is ttnn.ROW_MAJOR_LAYOUT here
sharded = ttnn.to_memory_config(weight, sharded_memory_config)  # may not error here...
ttnn.matmul(activation, sharded)  # ...but errors or miscomputes here

# Correct: convert layout first
weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)  # convert before sharding
sharded = ttnn.to_memory_config(weight, sharded_memory_config)
ttnn.matmul(activation, sharded)  # OK
```

> **Warning:** `ttnn.from_torch` defaults to `ROW_MAJOR_LAYOUT`. If you do not explicitly call `ttnn.to_layout`, the resulting tensor is not tile-packed and cannot be used directly with sharded matmul kernels.

---

## Pitfall 2: Specifying ShardSpec.shape in Tiles Rather Than Elements

**Description.** `ShardSpec.shape` takes element counts as `[shard_H_elements, shard_W_elements]`. A common mistake is to divide by 32 before passing the value, treating the field as a tile count.

**Symptom.** The shard shape is 32× smaller than intended. For a `[4096, 14336]` tensor with 8-core WIDTH_SHARDED:

```python
# Intended: shard_W = 1792 elements
# Wrong: passing tile count
shard_spec = ttnn.ShardSpec(
    core_range_set,
    [128, 56],   # 128 = 4096/32, 56 = 1792/32 — these are tile counts, not element counts
    ttnn.ShardOrientation.ROW_MAJOR,
)
# Shard shape is [128, 56] elements = 4 rows x 1.75 tile-columns
# Rule 2 violation: 56 % 32 != 0
```

This config will raise a tile-alignment error at `ttnn.to_memory_config`, but only because 56 is not a multiple of 32. If you accidentally pick tile counts that happen to be multiples of 32 (e.g., `[128, 64]`), there is no error, but the shard covers only a 128×64 element region, which does not match the intended 4096×1792 element shard. The resulting matmul will use the wrong weight regions silently.

**Fix.** Always pass element counts. Convert tile counts to elements by multiplying by 32 if you are reading from a program config:

```python
# If you know the shard shape in tiles:
shard_H_tiles, shard_W_tiles = 128, 56
shard_spec = ttnn.ShardSpec(
    core_range_set,
    [shard_H_tiles * 32, shard_W_tiles * 32],  # [4096, 1792] elements
    ttnn.ShardOrientation.ROW_MAJOR,
)
```

---

## Pitfall 3: Shard Width That Is a Multiple of 32 But Not a Multiple of `in0_block_w`

**Description.** The matmul kernel tiles the inner (K) dimension in blocks of `in0_block_w` tile-columns. When the shard width is not a multiple of `in0_block_w * 32` elements, the kernel must handle a partial final block differently from full blocks, preventing full loop unrolling and reducing throughput.

**Symptom.** No error is raised. The matmul produces correct numerical output. But kernel profiling shows a throughput reduction of 5–20% compared to an `in0_block_w`-aligned shard width.

**Example.** For Mixtral gate projection, suppose `in0_block_w = 8` tiles = 256 elements:

```python
# These shard widths are valid (Rule 2) but have different efficiency:
shard_W_aligned   = 1792  # 1792 / 256 = 7.0 — exactly 7 blocks of in0_block_w  OPTIMAL
shard_W_misaligned = 1760  # 1760 / 256 = 6.875 — 6 full blocks + 1 partial block  SUBOPTIMAL
# Both satisfy 1792 % 32 == 0 and 1760 % 32 == 0
```

**Fix.** Select shard widths that are multiples of both 32 and `in0_block_w * 32`. If `in0_block_w` is not known, run a quick sweep and profile, or use powers-of-2 multiples of 32 as a heuristic. For Mixtral with `in0_block_w = 8`:

```python
# Target: shard_W is a multiple of in0_block_w * 32 = 256
# 1792 = 7 * 256  OK
# 1536 = 6 * 256  OK (if tensor width allows this grid size)
```

> **Tip:** `in0_block_w` is set in the matmul program config. Retrieve it with `ttnn.get_program_config` after a first matmul call, or set it explicitly via the `program_config` argument to `ttnn.matmul`.

---

## Pitfall 4: Shard Grid Exceeding the Number of Available DRAM Banks

**Description.** Wormhole B0 has 12 DRAM banks (6 controllers × 2 banks each). If you specify a shard grid with more than 12 shards for a DRAM-sharded layout, multiple shards are assigned to the same physical bank. The bandwidth benefit of sharding — each shard accessing a dedicated bank without contention — disappears when two shards share a bank.

**Symptom.** No error. The config is accepted and the tensor is written correctly. Profiling shows DRAM bandwidth no better than the interleaved baseline because contention is re-introduced at the bank level.

**Example.** A 16-core WIDTH_SHARDED layout on a `[4096, 14336]` tensor:

```python
# 16 shards across 12 banks => at least 4 banks have 2 shards each
core_range_set = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(15, 0))
])
shard_W = 14336 // 16  # 896 elements — satisfies Rule 2 (896 % 32 == 0)
# Config is valid but provides no additional bandwidth over 12-core grid
```

**Fix.** Cap the shard count at 12 (one shard per DRAM bank) for bandwidth-optimal configurations. Use 6 shards if the weight tensor width divides more cleanly by 6 (one shard per controller pair). For shapes that do not divide evenly by 6 or 12, choose the largest factor of the tensor dimension that does not exceed 12.

```python
# For tensor_W = 14336, valid options that are <= 12 and divide evenly:
# 14336 / 1 = 14336 (1 shard, no parallelism)
# 14336 / 2 = 7168  (2 shards)
# 14336 / 4 = 3584  (4 shards)
# 14336 / 8 = 1792  (8 shards — fits within 12 banks)
# 14336 is not divisible by 12, so 8 is the largest bank-limited option
```

See Chapter 3, `tensor_to_shard_grid_mapping.md`, for the grid selection heuristic that accounts for DRAM controller count.

---

## Pitfall 5: Forgetting That the `num_experts` Batch Dimension Is Not Sharded by Height/Width Strategy

**Description.** Expert weight tensors are often stored as stacked 3D tensors of shape `[num_experts, d_model, d_ff]`. `TensorMemoryLayout.HEIGHT_SHARDED` and `WIDTH_SHARDED` partition the height and width of the 2D weight matrix `[d_model, d_ff]`. They do not partition across the `num_experts` batch dimension. If you apply a WIDTH_SHARDED config to the stacked `[num_experts, d_model, d_ff]` tensor, TTNN either flattens the batch dimension into the height (treating the effective shape as `[num_experts * d_model, d_ff]`) or raises a shape mismatch error, depending on how the tensor was constructed.

**Symptom.** Either an error at `ttnn.to_memory_config`, or a config that accepts the tensor but maps all expert weights into a single shard grid in a way that was not intended, causing the matmul to treat all experts as one large matrix rather than `num_experts` independent matrices.

**Example.** For Mixtral 8x7B, `num_experts = 8`, gate projection shape `[8, 4096, 14336]`:

```python
# Wrong: applying WIDTH_SHARDED directly to the 3D tensor
stacked_weight = ttnn.from_torch(
    torch.randn(8, 4096, 14336, dtype=torch.bfloat16), dtype=ttnn.bfloat16
)
stacked_weight = ttnn.to_layout(stacked_weight, ttnn.TILE_LAYOUT)
# Effective shape seen by sharding: [8, 4096, 14336]
# WIDTH_SHARDED partitions the last dimension (14336) — but the height
# is now [8, 4096] which gets flattened to 32768, NOT 4096.
# shard_H must be 32768 / num_cores, not 4096 / num_cores.
```

**Fix.** Use one of two approaches:

1. Store each expert's weights as a separate 2D tensor `[d_model, d_ff]` and shard each independently. This is the most straightforward path and matches how `ttnn.matmul` dispatches per-expert computation.

```python
# Store as a Python list of per-expert tensors
expert_weights = []
for expert_idx in range(num_experts):
    w = ttnn.from_torch(
        torch_weights[expert_idx],  # shape [d_model, d_ff]
        dtype=ttnn.bfloat16,
    )
    w = ttnn.to_layout(w, ttnn.TILE_LAYOUT)
    w = ttnn.to_memory_config(w, sharded_memory_config)  # shard_H = d_model, shard_W = d_ff // num_banks
    expert_weights.append(w)
```

2. Treat the stacked tensor with the correct 3D-aware shard arithmetic. In this case `shard_H` must account for the flattened `[num_experts * d_model]` height. See Chapter 2, `sharding_strategies.md`, for when this is appropriate.

> **Warning:** DeepSeek-V3 and Qwen 235B-A22B have `num_experts = 128`. Applying HEIGHT_SHARDED across the flattened `[128 * 7168, 2048] = [917504, 2048]` dimension without explicit 3D awareness produces shard heights of `917504 / 8 = 114688` elements — valid by Rule 1 but not what is needed for per-expert dispatching. Always verify that the shard arithmetic matches the intended per-expert dispatch granularity.

---

## Summary Table

| Pitfall | Detectable at config time? | Impact |
|---|---|---|
| 1: ROW_MAJOR_LAYOUT | At matmul call (not always at config) | Hard error or incorrect compute |
| 2: Tile count vs element count | Only if accidental mis-multiple of 32 | Silent wrong shard coverage |
| 3: Not multiple of `in0_block_w` | Never (not an error) | 5–20% throughput degradation |
| 4: Grid exceeds DRAM bank count | Never (not an error) | Bandwidth matches interleaved baseline |
| 5: Num-experts batch dimension | Depends on TTNN version | Shape error or silent per-expert grouping failure |

---

**Next:** [Chapter 6 — Performance Analysis and Trade-offs](../ch06_performance_analysis_and_tradeoffs/index.md)
