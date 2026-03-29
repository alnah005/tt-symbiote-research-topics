# DRAM-Sharded Decode: Bandwidth-Optimized Matmul for M=1

During decode, each token generation step processes a single position per user. The projection matmuls have M=1 (one tile row of activations), making them **bandwidth-bound** rather than compute-bound. The DRAM-sharded matmul strategy addresses this by distributing weight tiles across all 8 DRAM cores and streaming them through the compute grid in a single pass.

This section covers the configuration builders in `model_config.py` and the `_shard_linear` helper pattern used throughout `Qwen35Attention.forward_decode()`.

## DRAM-Sharded Memory Configuration

The `create_dram_sharded_mem_config(k, n)` function creates a WIDTH_SHARDED memory config that distributes weight matrices across the 8 DRAM cores of a Blackhole chip:

```python
DRAM_CORES = 8
DRAM_GRID = ttnn.CoreRangeSet(
    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))}
)

def create_dram_sharded_mem_config(k, n):
    padded_n = _roundup(n, TILE_SIZE * DRAM_CORES)   # pad N to multiple of 256 (32 * 8)
    shard_spec = ttnn.ShardSpec(
        DRAM_GRID,
        (k, padded_n // DRAM_CORES),                  # each core holds (K, N/8) slice
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )
```

The N dimension is padded to a multiple of `TILE_SIZE * DRAM_CORES = 256` to ensure each DRAM core holds an equal number of tile columns. The weights are stored as WIDTH_SHARDED across the 8 DRAM cores, meaning each core holds the full K (input) dimension but only `N/8` columns of the output dimension.

## DRAM-Sharded Matmul Program Configuration

The `create_dram_sharded_matmul_program_config(m, k, n)` function creates the corresponding matmul dispatch configuration:

```python
def create_dram_sharded_matmul_program_config(m, k, n, num_cores=None):
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE

    if num_cores is None:
        rows, cols = _find_grid(k_tiles)
        num_cores = rows * cols

    k_tiles_per_core = k_tiles // num_cores
    in0_block_w = _find_largest_divisor(k_tiles_per_core)
    per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=m_tiles,           # = 1 for decode (single token)
        per_core_N=per_core_N,
        fused_activation=None,
    )
```

Key aspects of this configuration:

- **`per_core_M = 1`** for decode: each core processes a single tile row of activations (32 rows, but only 1 contains real data for B users packed into tiles)
- **`in0_block_w`**: controls how many K-tiles are loaded at once from the activation shard; `_find_largest_divisor` finds the largest divisor up to 8 for optimal L1 reuse
- **`_find_grid(k_tiles)`**: searches for a grid layout (up to 8x8) whose total core count divides `k_tiles` evenly, preferring core counts near 32
- The `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` type tells tt-metal to stream weight tiles directly from DRAM shards through the compute cores, avoiding an explicit weight-to-L1 copy step

## The _shard_linear Pattern

All decode projections in `Qwen35Attention` use the `_shard_linear` helper:

```python
def _shard_linear(x_tt, weight, act_shard_cfg, prog_cfg, compute_cfg):
    x_sharded = ttnn.to_memory_config(x_tt, act_shard_cfg)
    return ttnn.linear(
        x_sharded, weight,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=prog_cfg,
        compute_kernel_config=compute_cfg,
    )
```

The data flow for each matmul is:

1. **Input sharding**: `ttnn.to_memory_config(x_tt, act_shard_cfg)` moves the activation from DRAM interleaved to L1 WIDTH_SHARDED. The `act_shard_hidden` config distributes the 5120-dim input across compute cores.
2. **DRAM-sharded matmul**: The weight matrix stays in DRAM (WIDTH_SHARDED across 8 cores). The matmul streams weight tiles from DRAM while the activation is already in L1.
3. **Output in L1**: The result is produced in `L1_WIDTH_SHARDED_MEMORY_CONFIG`.
4. **Unshard to DRAM**: The `_unshard()` helper moves the result back to DRAM interleaved via `ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)` for subsequent reshape/slice operations.

This pattern is applied three times in `forward_decode()` for Q+gate, K, and V projections, and once more for the output projection `wo` (which uses `act_shard_attn_out` instead of `act_shard_hidden` since its input dimension is `NH*HD = 1536` per device, not 5120).

## Compute Kernel Configuration

All decode matmuls use the `COMPUTE_HIFI2` kernel configuration defined in `model_config.py`:

```python
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

- **HiFi2 math fidelity**: provides a good balance between accuracy and throughput for BFP8 weight matmuls
- **FP32 destination accumulation**: partial products are accumulated in FP32 before conversion to bfloat16, reducing numerical error from the BFP8 weights
- **Packer L1 accumulation**: enables the packer to accumulate results in L1 before writing, reducing NOC traffic

This configuration is referenced as `self.compute_cfg = args.compute_kernel_config_hifi2` in the `Qwen35Attention.__init__()` constructor.

## Per-Head KV Cache Update

After Q, K, and V are projected and normalized, the K and V tensors must be inserted into the KV cache at the current position. The implementation supports two paths:

### Paged KV Cache (vLLM integration)

When `kv_cache` is provided externally:

```python
ttnn.experimental.paged_update_cache(keys, k, update_idxs_tensor=cur_pos_tt, page_table=page_table)
ttnn.experimental.paged_update_cache(values, v, update_idxs_tensor=cur_pos_tt, page_table=page_table)
```

### Internal Per-Head KV Cache (standalone mode)

When using the internal cache (`self.k_caches` / `self.v_caches`), each of the `NKV = 1` KV heads is updated separately:

```python
for h in range(NKV):
    k_h = ttnn.slice(k, (0, 0, h, 0), (1, B, h + 1, HD))
    v_h = ttnn.slice(v, (0, 0, h, 0), (1, B, h + 1, HD))
    k_h_padded = ttnn.pad(k_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
    v_h_padded = ttnn.pad(v_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
    k_sh = ttnn.to_memory_config(k_h_padded, self._kv_update_shard_cfg)
    v_sh = ttnn.to_memory_config(v_h_padded, self._kv_update_shard_cfg)
    ttnn.experimental.paged_update_cache(self.k_caches[h], k_sh, update_idxs_tensor=cur_pos_tt)
    ttnn.experimental.paged_update_cache(self.v_caches[h], v_sh, update_idxs_tensor=cur_pos_tt)
```

The `_kv_update_shard_cfg` is a HEIGHT_SHARDED config (from Chapter 1) with shard shape `(32, 256)` on an 8x4 grid:

```python
self._kv_update_shard_cfg = ttnn.create_sharded_memory_config(
    shape=(TILE_SIZE, self.head_dim),  # (32, 256)
    core_grid=ttnn.CoreGrid(x=8, y=4),
    strategy=ttnn.ShardStrategy.HEIGHT,
)
```

Each of the 32 cores (8x4 grid) holds one user's KV entry -- a single tile of shape `[32, 256]`. The padding to `[1, B, 32, HD]` ensures the head dimension is tile-aligned before sharding.

After cache updates, the full KV tensors are assembled for SDPA:

```python
attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
    q, k_full, v_full,
    cur_pos_tensor=cur_pos_tt,
    scale=self.scale,                  # head_dim ** -0.5
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

## Memory Lifecycle

The decode forward pass is careful about memory management, deallocating intermediate tensors as soon as they are no longer needed:

```python
qg_r = ttnn.reshape(qg_tt, (1, B, NH, HD * 2))
ttnn.deallocate(qg_tt)       # original projection output no longer needed
q = ttnn.slice(qg_r, ...)
gate = ttnn.slice(qg_r, ...)
ttnn.deallocate(qg_r)        # reshaped tensor no longer needed after slicing
```

This discipline is critical because DRAM is shared across all 64 layers. Each intermediate tensor consumes precious bandwidth capacity, and timely deallocation ensures buffers can be reused by subsequent operations within the same decode step.

---

**Previous:** [`attention_architecture.md`](./attention_architecture.md) | **Next:** [`flash_attention_prefill.md`](./flash_attention_prefill.md)
