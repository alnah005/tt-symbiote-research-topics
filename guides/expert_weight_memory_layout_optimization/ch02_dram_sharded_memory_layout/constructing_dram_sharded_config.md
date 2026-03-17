# Constructing a DRAM-Sharded MemoryConfig

This file walks through every step of building and validating a DRAM-sharded `MemoryConfig`, then shows a complete end-to-end example loading an expert weight checkpoint and preparing it for matmul.

---

## Step-by-Step Construction

The three-step pattern is: build a `ShardSpec`, pass it to `MemoryConfig`, then move the tensor with `to_memory_config`.

### Step 1: Define the ShardSpec

```python
import ttnn

# Goal: WIDTH_SHARDED DRAM layout for gate_proj weight [4096, 14336]
# across 8 DRAM banks (1 row of 8 cores)

shard_spec = ttnn.ShardSpec(
    grid=ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))
    }),
    shape=[4096, 1792],  # shard_height=4096, shard_width=14336/8=1792
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
```

### Step 2: Construct the MemoryConfig

```python
dram_sharded_config = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.DRAM,
    shard_spec=shard_spec,
)
```

Note: for DRAM-sharded layouts, always use the `ttnn.MemoryConfig` constructor directly with `buffer_type=BufferType.DRAM` (see Common Mistake #4 below for details on why `ttnn.create_sharded_memory_config` must not be used here).

### Step 3: Move the Tensor

```python
# weight is initially DRAM-interleaved (from ttnn.from_torch or checkpoint load)
weight_sharded = ttnn.to_memory_config(weight, dram_sharded_config)
```

`ttnn.to_memory_config` issues the necessary data movement on-device. After this call, `weight` can be deallocated if no longer needed in interleaved form.

---

## Verifying the Configuration

After calling `to_memory_config`, confirm that the tensor landed in the correct layout:

```python
mc = weight_sharded.memory_config()
print(mc)               # shows TensorMemoryLayout.WIDTH_SHARDED, BufferType.DRAM

ss = weight_sharded.shard_spec()
print(ss.shape)         # [4096, 1792]
print(ss.orientation)   # ShardOrientation.ROW_MAJOR
```

> **Tip:** Always verify `memory_config()` and `shard_spec()` after a layout transition, especially during initial integration. Silent mismatches (e.g., landing in INTERLEAVED because a helper defaulted `buffer_type`) can cause hard-to-diagnose performance regressions rather than immediate errors.

---

## Common Mistakes

**1. Missing `shard_spec` for a sharded MemoryConfig**

```python
# Wrong: shard_spec omitted for a sharded layout
bad_cfg = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.DRAM,
    # shard_spec=...  <-- missing
)
# TTNN raises an error at construction or allocation time.
```

**2. Supplying `shard_spec` for an INTERLEAVED config**

```python
# Wrong: shard_spec must be None for INTERLEAVED
bad_cfg = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.DRAM,
    shard_spec=shard_spec,   # <-- must be omitted or None
)
# TTNN raises an error.
```

**3. Non-integer or non-tile-aligned shard dimension**

```python
# Suppose you choose 6 shards for a width of 14336:
shard_width = 14336 / 6  # = 2389.33... (not an integer) → allocation error

# Or 6 divides cleanly into 14334, but 14334 / 6 = 2389 and 2389 % 32 != 0 → tile alignment error
```

Always choose `num_shards` such that `tensor_dim / num_shards` is an integer AND divisible by 32. For width 14336: valid divisors of 14336 that also divide 32 include 1, 2, 4, 8, 14, 16, ... — with 8 being the canonical choice. Note: Wormhole B0 has 6 DRAM controllers (see `sharding_strategies.md` for hardware context), not 8; the choice of 8 shards is justified by alignment with the 8-column Tensix grid width and the resulting tile-aligned shard size (14336/8=1792 elements).

> **Warning:** The error from a non-tile-aligned shard shape surfaces at buffer allocation, not at `ShardSpec` construction. Arithmetic errors may not be caught until `ttnn.to_memory_config` is called. Validate dimensions before constructing the spec.

**4. Using `create_sharded_memory_config` for DRAM placement**

```python
# Wrong: this helper defaults to L1 — do not use for DRAM-sharded weights
bad_cfg = ttnn.create_sharded_memory_config(
    shape=[4096, 1792],
    core_grid=grid,
    strategy=ttnn.ShardStrategy.WIDTH,
)
# buffer_type will be L1, not DRAM. Use ttnn.MemoryConfig directly instead.
```

---

## End-to-End Example

The following snippet loads an expert gate projection weight from a CPU checkpoint, converts it to DRAM-sharded layout, and feeds it to a matmul. In production MoE code, steps 1 and 2 happen once at model load time; the DRAM-sharded tensor persists across all decode steps.

```python
import ttnn
import torch

device = ttnn.open_device(device_id=0)

# Load weight from checkpoint (CPU, float32)
weight_cpu = torch.randn(4096, 14336)   # gate_proj for one expert

# Step 1: convert to bfloat16 tile layout, interleaved DRAM
weight_tt = ttnn.from_torch(
    weight_cpu,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Step 2: move to DRAM-sharded layout
shard_spec = ttnn.ShardSpec(
    grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
    shape=[4096, 1792],
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)
dram_sharded_cfg = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.DRAM,
    shard_spec,
)
weight_sharded = ttnn.to_memory_config(weight_tt, dram_sharded_cfg)
ttnn.deallocate(weight_tt)   # free the interleaved copy

# Step 3: use in matmul — TTNN dispatches data movement from DRAM-sharded banks
# activation_l1 is a batch of token activations already in L1-sharded format
output = ttnn.matmul(activation_l1, weight_sharded, ...)

# Verify (during development / debugging)
mc = weight_sharded.memory_config()
ss = weight_sharded.shard_spec()
assert mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
assert mc.buffer_type == ttnn.BufferType.DRAM
assert ss.shape == [4096, 1792]
```

> **Tip:** In production MoE code, weights are loaded once and reused across decode steps. Keep the DRAM-sharded weight tensor alive for the model's lifetime; only deallocate the intermediate interleaved copy immediately after conversion.

---

**Next:** [Chapter 3 — Expert Weight Tensor Structure](../ch03_expert_weight_tensor_structure/index.md)
