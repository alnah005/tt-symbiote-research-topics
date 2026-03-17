# TTNN Memory Configuration API

## Overview

TTNN exposes memory placement decisions through a small set of dataclasses and helper functions. This file covers the complete API surface needed to specify, inspect, and migrate tensor memory placement. It assumes you have read `wormhole_memory_hierarchy.md` and understand the underlying hardware concepts.

---

## `ttnn.MemoryConfig`

`MemoryConfig` is the primary dataclass for specifying where a tensor lives and how it is laid out within that memory region.

```python
import ttnn

# Full constructor signature
config = ttnn.MemoryConfig(
    memory_layout: ttnn.TensorMemoryLayout,
    buffer_type: ttnn.BufferType,
    shard_spec: ttnn.ShardSpec | None = None,
)
```

### Fields

| Field | Type | Description |
|---|---|---|
| `memory_layout` | `ttnn.TensorMemoryLayout` | How tiles are distributed: `INTERLEAVED`, `HEIGHT_SHARDED`, `WIDTH_SHARDED`, or `BLOCK_SHARDED` |
| `buffer_type` | `ttnn.BufferType` | Physical memory target: `DRAM` or `L1` |
| `shard_spec` | `ttnn.ShardSpec` or `None` | Required for sharded layouts; describes the core grid, per-core shard shape, and orientation. `None` for `INTERLEAVED`. |

---

## `ttnn.TensorMemoryLayout` Variants

### `INTERLEAVED`

Tile stripes of the tensor are distributed across multiple banks (DRAM) or cores (L1) in a round-robin fashion. No explicit sharding geometry is needed; the layout is determined by the number of banks/cores and the number of tiles.

This is the **default layout** for all TTNN ops when no `memory_config` is specified. It works for any tensor size and is the safe choice when you are uncertain about L1 budget.

```python
# DRAM interleaved — tiles spread across DRAM banks
dram_interleaved = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.DRAM,
)

# L1 interleaved — tiles spread across L1 of multiple cores (less common)
l1_interleaved = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.L1,
)
```

> **Warning:** L1 interleaved (`INTERLEAVED` + `BufferType.L1`) distributes tiles across cores but does not guarantee that a single core's share is small. For large tensors, L1 interleaved can still exceed per-core L1 budget. Use sharded layouts to explicitly control per-core footprint.

### `HEIGHT_SHARDED`

The tensor's row (height) dimension is split across cores in a 1D core grid. Each core holds a contiguous block of rows — its shard — in its local L1.

Required fields in `ShardSpec`:
- `grid`: the set of cores to use (1D or 2D grid)
- `shape`: `[per_core_rows_in_tiles * 32, full_width_in_elements]` — the shape of one shard
- `orientation`: `ShardOrientation.ROW_MAJOR` (most common)

```python
import ttnn

# Example: HEIGHT_SHARDED for a [512, 7168] tensor across 16 cores
# Each core holds 512/16 = 32 rows
shard_spec = ttnn.ShardSpec(
    grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
    shape=[32, 7168],       # per-core shard shape in elements
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)

height_sharded_l1 = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=shard_spec,
)
```

### `WIDTH_SHARDED`

The tensor's column (width) dimension is split across cores. Each core holds all rows but only its slice of the columns.

```python
# Example: WIDTH_SHARDED for a [32, 7168] tensor across 8 cores
# Each core holds 7168/8 = 896 columns
shard_spec = ttnn.ShardSpec(
    grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
    shape=[32, 896],        # per-core shard shape in elements
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)

width_sharded_l1 = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=shard_spec,
)
```

### `BLOCK_SHARDED`

Both row and column dimensions are split across a 2D core grid. This minimizes per-core shard size for large square-ish tensors.

```python
# Example: BLOCK_SHARDED for a [512, 7168] tensor on an 8x10 grid
# Each core holds (512/8) x (7168/10) = 64 x 717 elements
# (In practice tile-align to nearest 32: 64 x 736 after padding)
shard_spec = ttnn.ShardSpec(
    grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))}),
    shape=[64, 736],         # per-core shard shape (tile-aligned)
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
)

block_sharded_l1 = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    buffer_type=ttnn.BufferType.L1,
    shard_spec=shard_spec,
)
```

---

## Predefined Configurations

TTNN provides two constants that cover the most common cases. Use these by preference over constructing `MemoryConfig` manually when the intent matches.

```python
import ttnn

# DRAM interleaved — the global default for output tensors
ttnn.DRAM_MEMORY_CONFIG
# Equivalent to:
# ttnn.MemoryConfig(
#     memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
#     buffer_type=ttnn.BufferType.DRAM,
# )

# L1 interleaved — use for small tensors where L1 placement helps latency
ttnn.L1_MEMORY_CONFIG
# Equivalent to:
# ttnn.MemoryConfig(
#     memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
#     buffer_type=ttnn.BufferType.L1,
# )
```

---

## Passing `memory_config` to Operations

All major TTNN operations accept an optional `memory_config` keyword argument that controls where the output tensor is placed. If the argument is omitted, the output goes to `ttnn.DRAM_MEMORY_CONFIG`.

```python
import ttnn

# Matmul: output to L1 (for small decode tensors)
output = ttnn.matmul(
    input_tensor_a,
    input_tensor_b,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

# All-to-all: output to DRAM (for large prefill buffers)
dispatched = ttnn.all_to_all(
    expert_tokens,
    cluster_axis=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Softmax: output to L1 (routing scores are small)
routing_scores = ttnn.softmax(
    router_logits,
    dim=-1,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

# Layernorm: output placement depends on phase
hidden_states = ttnn.layer_norm(
    hidden_states,
    weight=ln_weight,
    bias=ln_bias,
    memory_config=ttnn.L1_MEMORY_CONFIG,   # decode: L1
    # memory_config=ttnn.DRAM_MEMORY_CONFIG,  # prefill: DRAM
)
```

> **Tip:** Setting `memory_config` on each op call is the most explicit and recommended approach. It makes memory placement auditable by code review and avoids surprises from TTNN's default inference rules.

---

## Querying Current Memory Placement

To inspect where a tensor currently lives, call `.memory_config()` on the tensor object:

```python
import ttnn

tensor = ttnn.from_torch(
    torch_tensor,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

mc = tensor.memory_config()
print(mc.buffer_type)       # ttnn.BufferType.DRAM
print(mc.memory_layout)     # ttnn.TensorMemoryLayout.INTERLEAVED
print(mc.shard_spec)        # None (for INTERLEAVED)
```

This is useful for debugging unexpected placements or verifying that an op correctly respected a `memory_config` argument.

---

## Explicit Tensor Migration: `ttnn.to_memory_config`

To move a tensor between L1 and DRAM explicitly, use `ttnn.to_memory_config`. This issues a device-side copy operation and has non-trivial latency.

```python
import ttnn

# Move a DRAM tensor to L1 before a latency-sensitive op
tensor_l1 = ttnn.to_memory_config(tensor_dram, ttnn.L1_MEMORY_CONFIG)

# Move back to DRAM for persistence
tensor_dram_out = ttnn.to_memory_config(tensor_l1, ttnn.DRAM_MEMORY_CONFIG)
```

> **Warning:** Each call to `ttnn.to_memory_config` is a memory copy operation that runs on the device. Minimize round-trips: if you intend an op's output to live in L1, pass `memory_config=ttnn.L1_MEMORY_CONFIG` directly to that op rather than producing a DRAM output and then migrating it. Unnecessary migrations waste device time and memory bandwidth.

### When Migration Is Necessary

Migration is appropriate when:
- The op that produces a tensor does not support a direct `memory_config` argument.
- You are integrating with a pre-built module that always outputs to DRAM, and you need the tensor in L1 for the next op.
- You are explicitly checkpointing: writing L1 tensors to DRAM before op boundaries where L1 is not guaranteed to persist.

---

## Memory Pressure Monitoring

TTNN performs L1 allocation checks at program compilation time (before device execution). If the sum of all CB allocations on any core would exceed 1.5 MB, TTNN raises:

```
ttnn.exceptions.MemoryAllocationError: Failed to allocate circular buffer ...
```

There is **no transparent fallback** to DRAM when L1 is exhausted. The kernel will not silently use DRAM instead; the entire compilation fails.

### Diagnosis Steps

When you encounter a `MemoryAllocationError`:

1. Identify which op triggered the failure (the traceback will point to a kernel compilation call).
2. Calculate the CB footprint for that op at the current tensor shapes and core count.
3. Either:
   - Increase the number of cores assigned to the op (reduces per-core shard size).
   - Move input or output tensors to DRAM to reduce CB pressure.
   - Reduce CB double-buffering depth if the op accepts a configuration parameter for it.

```python
# Pattern: catch allocation errors during development to identify pressure points
try:
    output = ttnn.matmul(
        q_proj,
        k_proj,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
except Exception as e:
    if "MemoryAllocation" in str(type(e).__name__) or "allocate" in str(e).lower():
        print(f"L1 allocation failed for matmul: {e}")
        print("Falling back to DRAM placement")
        output = ttnn.matmul(
            q_proj,
            k_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        raise
```

> **Tip:** Use the TT-Metal profiler (`tracy` or the built-in TTNN profiler) to observe per-core CB allocation sizes during development. This surfaces L1 pressure before it manifests as hard errors in production.

---

## Common Pattern: Decode Memory Management

The canonical pattern for keeping decode activations in L1 throughout a step is:

```python
import ttnn

def decode_step(
    hidden_states: ttnn.Tensor,  # assumed to start in DRAM after prefill
    kv_cache: ttnn.Tensor,       # stays in DRAM throughout
    attn_module,
    ffn_module,
) -> ttnn.Tensor:
    """
    Run one decode step, keeping activations in L1 for latency.
    KV cache stays in DRAM (persistent, too large for L1).
    """
    # Bring activations into L1 at the start of the step.
    # Shape [B, 1, H] = [B, H] for decode; small enough for L1.
    hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

    # Attention: Q projection stays in L1
    q = ttnn.matmul(
        hidden_states,
        attn_module.wq,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # KV cache update writes to DRAM — persistent state
    # (kv_cache read happens in DRAM; write path to DRAM)
    attn_output = attn_module.attend(
        q=q,
        kv_cache=kv_cache,          # DRAM tensor; read inside attend()
        output_memory_config=ttnn.L1_MEMORY_CONFIG,  # output stays in L1
    )

    # FFN intermediate: L1 for decode (small at B=32)
    ffn_output = ffn_module(
        attn_output,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Write final hidden state back to DRAM for persistence across steps
    output = ttnn.to_memory_config(ffn_output, ttnn.DRAM_MEMORY_CONFIG)
    return output
```

The key principle: **one migration in at the start, one migration out at the end**. All intermediate ops use `memory_config=ttnn.L1_MEMORY_CONFIG` to avoid redundant DRAM writes between ops.

---

## API Quick Reference

| API Element | Purpose |
|---|---|
| `ttnn.MemoryConfig(layout, buffer_type, shard_spec)` | Construct a memory placement specification |
| `ttnn.DRAM_MEMORY_CONFIG` | Predefined DRAM interleaved config |
| `ttnn.L1_MEMORY_CONFIG` | Predefined L1 interleaved config |
| `ttnn.TensorMemoryLayout.INTERLEAVED` | Tiles distributed across banks/cores |
| `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` | Rows split across cores |
| `ttnn.TensorMemoryLayout.WIDTH_SHARDED` | Columns split across cores |
| `ttnn.TensorMemoryLayout.BLOCK_SHARDED` | Rows and columns split across a 2D core grid |
| `ttnn.BufferType.DRAM` | Place in DRAM |
| `ttnn.BufferType.L1` | Place in L1 |
| `ttnn.ShardSpec(grid, shape, orientation)` | Describe sharding geometry for sharded layouts |
| `tensor.memory_config()` | Query current placement of a tensor |
| `ttnn.to_memory_config(tensor, config)` | Migrate tensor to a new memory config (device copy) |

---

## References

- `tt-metal` source: `ttnn/cpp/ttnn/tensor/types.hpp` — `MemoryConfig`, `TensorMemoryLayout`, `BufferType`, `ShardSpec`
- `tt-metal` source: `ttnn/cpp/ttnn/operations/core/core.cpp` — `to_memory_config` implementation
- TT-NN Python API documentation: `ttnn.MemoryConfig`, `ttnn.to_memory_config`
- Chapter 4 — `wormhole_memory_hierarchy.md`: L1 and DRAM hardware background
- Chapter 4 — `decode_memory_strategy.md`: decode-phase placement decisions using this API
- Chapter 4 — `prefill_memory_strategy.md`: prefill-phase placement decisions using this API
