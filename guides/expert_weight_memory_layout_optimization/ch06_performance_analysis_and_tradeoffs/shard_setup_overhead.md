# Shard Setup Overhead

## What Shard Setup Overhead Means

"Shard setup overhead" is the one-time cost of calling `ttnn.to_memory_config` to convert a weight tensor from its initial interleaved DRAM layout into a DRAM-sharded layout. It is not a per-inference cost — once the tensor is resharded, it stays in the sharded layout for the lifetime of the model unless explicitly changed.

This overhead is distinct from inference-time costs. It covers:

1. **DRAM-to-DRAM data movement:** The weight bytes must be read from their interleaved locations and written to new contiguous shard locations in DRAM. On Wormhole B0 this is a DMA operation mediated by Tensix cores.
2. **`ShardSpec` validation:** TTNN verifies that the requested `shard_shape` and `CoreRangeSet` are tile-valid and consistent with the tensor shape before issuing any DMA work.
3. **Metadata table construction:** The runtime builds a shard-to-bank mapping table used by the matmul kernel during inference to locate each shard without runtime address arithmetic.

The DRAM-to-DRAM data movement dominates. Steps 2 and 3 are microseconds; the DMA transfer is milliseconds.

---

## When This Cost Is Paid

There are two points in the model lifecycle at which `ttnn.to_memory_config` can be called for expert weights:

**Option A: At model load time (recommended)**

```python
# Called once during model initialization, before any inference requests are served.
# weight_interleaved was transferred from CPU with ttnn.DRAM_MEMORY_CONFIG.
weight_sharded = ttnn.to_memory_config(weight_interleaved, sharded_dram_config)
# weight_interleaved can now be freed; weight_sharded is kept for the model's lifetime.
```

The resharding cost is paid once, during the model loading phase that already includes checkpoint loading, weight dtype conversion, and device transfer. Adding a few milliseconds per expert weight tensor to this phase has no effect on request latency.

**Option B: At inference time (not recommended for deployment)**

```python
# Called inside the forward pass, once per token or per batch.
weight_sharded = ttnn.to_memory_config(weight_interleaved, sharded_dram_config)
output = ttnn.matmul(activation, weight_sharded, ...)
# weight_sharded is freed after use; weight_interleaved is kept.
```

This pattern pays the resharding cost on every forward pass. For Mixtral with 32 layers and 3 weight projections per expert, that is 32 × 8 × 3 = 768 resharding calls per decode step. Even at a few milliseconds each, this adds hundreds of milliseconds of resharding overhead per token — completely defeating the bandwidth gain.

> **Warning:** Never call `ttnn.to_memory_config` inside the inference forward pass for weights that are static across inference calls. Always reshard at load time.

---

## The Recommended Pattern: Load-Time Resharding

The canonical pattern for deploying DRAM-sharded expert weights is:

```python
import ttnn

def load_expert_weights(
    checkpoint_path: str,
    device: ttnn.Device,
    model_config: dict,
) -> dict[str, ttnn.Tensor]:
    """
    Load all expert weight tensors onto device in DRAM-sharded layout.
    This function is called once during model initialization.
    """
    # Build the DRAM-sharded config for gate/up projections [d_model, d_ff].
    # WIDTH_SHARDED along d_ff; 6 shards across 6 DRAM controller columns.
    d_model = model_config["d_model"]  # e.g., 4096 for Mixtral
    d_ff    = model_config["d_ff"]     # e.g., 14336 for Mixtral
    num_dram_banks = 6                 # Wormhole B0 DRAM controller count

    # Shard width: d_ff / num_dram_banks must be tile-aligned (multiple of 32).
    shard_width = d_ff // num_dram_banks  # e.g., 14336 // 6 = 2389 — not aligned!
    # Align to the nearest multiple of 32 that divides d_ff evenly.
    # For Mixtral: 14336 / 448 = 32 shards; use 4 banks × 2 shards = 8-shard grid.
    # See Chapter 5 shard_shape_alignment_rules.md for derivation.
    # This example uses 8 shards (1 × 8 core grid) for illustration.
    shard_width = d_ff // 8             # 14336 // 8 = 1792; 1792 % 32 == 0 ✓

    gate_up_shard_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(
            # 1×8 core grid spanning 8 DRAM bank columns.
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            [d_model, shard_width],      # shard shape in elements
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Down projection [d_ff, d_model] uses HEIGHT_SHARDED along d_ff.
    shard_height = d_ff // 8            # 14336 // 8 = 1792 ✓
    down_shard_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.DRAM,
        shard_spec=ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            [shard_height, d_model],     # shard shape in elements
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    expert_weights = {}
    for expert_id in range(model_config["num_experts"]):
        # Load from checkpoint to CPU torch tensors (not shown: checkpoint parsing).
        w_gate_cpu = load_from_checkpoint(checkpoint_path, expert_id, "gate")
        w_up_cpu   = load_from_checkpoint(checkpoint_path, expert_id, "up")
        w_down_cpu = load_from_checkpoint(checkpoint_path, expert_id, "down")

        # Transfer to device with default interleaved layout.
        w_gate_interleaved = ttnn.from_torch(w_gate_cpu, dtype=ttnn.bfloat16,
                                             layout=ttnn.TILE_LAYOUT, device=device,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w_up_interleaved   = ttnn.from_torch(w_up_cpu,   dtype=ttnn.bfloat16,
                                             layout=ttnn.TILE_LAYOUT, device=device,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w_down_interleaved = ttnn.from_torch(w_down_cpu, dtype=ttnn.bfloat16,
                                             layout=ttnn.TILE_LAYOUT, device=device,
                                             memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshard to DRAM-sharded. This is the one-time overhead cost.
        # For [4096, 14336] BF16: takes single-digit milliseconds per tensor.
        expert_weights[f"expert_{expert_id}_gate"] = ttnn.to_memory_config(
            w_gate_interleaved, gate_up_shard_config
        )
        expert_weights[f"expert_{expert_id}_up"] = ttnn.to_memory_config(
            w_up_interleaved, gate_up_shard_config
        )
        expert_weights[f"expert_{expert_id}_down"] = ttnn.to_memory_config(
            w_down_interleaved, down_shard_config
        )

        # Free the interleaved copies to recover DRAM.
        ttnn.deallocate(w_gate_interleaved)
        ttnn.deallocate(w_up_interleaved)
        ttnn.deallocate(w_down_interleaved)

    return expert_weights
    # expert_weights is held in model state and reused for all inference calls.
```

After `load_expert_weights` returns, all expert weight tensors are in DRAM-sharded layout. The inference forward pass uses them directly with no further resharding.

---

## Reshard Latency Estimate

For a weight tensor of shape `[4096, 14336]` in `bfloat16`:

```
weight_bytes = 4096 × 14336 × 2 = 117,440,512 bytes ≈ 112 MB
```

The `ttnn.to_memory_config` call must read all 112 MB from DRAM (interleaved) and write 112 MB back to DRAM (sharded). The effective DRAM-to-DRAM copy bandwidth on Wormhole B0 is bounded by the lower of read and write bandwidth; in practice this is in the range of 50–100 GB/s for DMA copies (not the full matmul-style ~300 GB/s, since DMA copies use fewer cores and have higher per-byte overhead).

At 100 GB/s effective copy bandwidth:

```
reshard_time ≈ 2 × 112 MB / 100 GB/s
             = 224 MB / 100 GB/s
             = 2.24 ms
```

At 50 GB/s:

```
reshard_time ≈ 4.5 ms
```

**The reshard cost for a single `[4096, 14336]` BF16 expert weight tensor is on the order of 2–5 ms** — single-digit milliseconds, negligible in the context of model loading (which typically takes seconds to tens of seconds for large MoE models).

For Mixtral 8x7B with 24 weight tensors per layer (8 experts × 3 projections each), totaling 768 tensors across 32 layers:

```
total_reshard_time ≈ 24 × 32 × 3 ms ≈ 2.3 seconds
```

This adds approximately 2–5 seconds to model load time. For a model that serves inference for hours, this is an entirely acceptable one-time cost.

> **Tip:** If model loading time is tightly constrained, the resharding loop can be parallelized across experts using `ttnn`'s async dispatch mode. Consult the TTNN async dispatch documentation for details.

---

## Program Cache Interaction

TTNN's program cache keys each compiled kernel on a combination of:

- Operation type (e.g., `ttnn.matmul`)
- Input tensor shapes
- Input tensor `MemoryConfig` objects (including `ShardSpec` contents)
- Op configuration (e.g., `compute_kernel_config`, `core_grid`)

If a weight tensor's `MemoryConfig` changes between two calls to `ttnn.matmul`, the program cache sees a new key and triggers recompilation. Recompilation on Wormhole B0 takes on the order of seconds per kernel, which dominates inference latency completely.

**Stability rule:** The `ShardSpec` of every weight tensor passed to `ttnn.matmul` must be identical across all inference calls that share a program cache slot. Specifically:

- `shard_spec.grid` must be the same `CoreRangeSet` every call.
- `shard_spec.shape` must be the same `[H, W]` pair every call.
- `shard_spec.orientation` must be the same every call.

Because the load-time resharding pattern reshards weights once and holds them fixed, program cache stability is guaranteed automatically: the weight tensors' `MemoryConfig` never changes during inference.

> **Warning:** Do not attempt to dynamically adjust `shard_shape` based on runtime batch size. Even if both shapes are tile-valid, changing the `ShardSpec` forces a cache miss and recompilation. If you need different shard configurations for different batch sizes (e.g., a dedicated decode config and a dedicated prefill config), compile and warm up both program cache entries during model initialization, then switch between the pre-warmed configs without triggering recompilation.

The warm-up pattern for dual-config deployment:

```python
# During model initialization (not inference):
# Warm up decode program cache entry.
_ = ttnn.matmul(decode_activation_dummy, weight_sharded_decode_config, ...)

# Warm up prefill program cache entry.
_ = ttnn.matmul(prefill_activation_dummy, weight_sharded_prefill_config, ...)

# Both entries are now in cache. Inference calls will hit cache for both regimes.
```

---

**Next:** [`tradeoff_matrix.md`](./tradeoff_matrix.md)
