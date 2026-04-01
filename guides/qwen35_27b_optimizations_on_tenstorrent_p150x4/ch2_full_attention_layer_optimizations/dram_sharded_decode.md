# DRAM-Sharded Decode: Bandwidth-Optimized Matmul for M=1

During decode, each token generation step processes a single position per user. The projection matmuls have M=1 (one tile row of activations), making them **bandwidth-bound** rather than compute-bound. The DRAM-sharded matmul strategy addresses this by distributing weight tiles across all 8 DRAM cores and streaming them through the compute grid in a single pass.

This section covers the configuration builders in `model_config.py` and the `_shard_linear` helper pattern used throughout `Qwen35Attention.forward_decode()`.

## DRAM-Sharded Memory Configuration

The `create_dram_sharded_mem_config(k, n)` function (model_config.py:80–92) creates a WIDTH_SHARDED memory config that distributes weight matrices across the 8 DRAM cores defined by `DRAM_GRID` (model_config.py:47–49):

```python
TILE_SIZE  = 32
DRAM_CORES = 8
DRAM_GRID  = ttnn.CoreRangeSet(
    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))}
)

def create_dram_sharded_mem_config(k, n):
    padded_n   = _roundup(n, TILE_SIZE * DRAM_CORES)   # pad N to multiple of 256 (32 * 8)
    shard_spec = ttnn.ShardSpec(
        DRAM_GRID,
        (k, padded_n // DRAM_CORES),                   # each core holds (K, N/8) slice
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )
```

The N dimension is padded to a multiple of `TILE_SIZE * DRAM_CORES = 256` to ensure each DRAM core holds an equal number of tile columns. Weights are stored WIDTH_SHARDED: each core holds the full K (input) dimension but only `N/8` columns of the output dimension.

## DRAM-Sharded Matmul Program Configuration

The `create_dram_sharded_matmul_program_config(m, k, n)` function (model_config.py:95–118) creates the corresponding matmul dispatch configuration:

```python
def create_dram_sharded_matmul_program_config(m, k, n, num_cores=None):
    m_tiles          = math.ceil(m / TILE_SIZE)
    k_tiles          = math.ceil(k / TILE_SIZE)
    n_padded         = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles          = n_padded // TILE_SIZE

    if num_cores is None:
        rows, cols   = _find_grid(k_tiles)   # finds grid near 32 cores dividing k_tiles
        num_cores    = rows * cols

    k_tiles_per_core = k_tiles // num_cores
    in0_block_w      = _find_largest_divisor(k_tiles_per_core)   # largest divisor <= 8
    per_core_N       = n_tiles // num_cores if n_tiles >= num_cores else 1

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=m_tiles,       # = 1 for decode (single token)
        per_core_N=per_core_N,
        fused_activation=None,
    )
```

This kernel streams weight tiles from DRAM one row-shard at a time, matching the physical DRAM-sharded layout.

All four attention decode program configs are instantiated at model init time with `M = 1` (model_config.py:290–296):

```python
M = 1
self.attn_qg_progcfg = create_dram_sharded_matmul_program_config(M, self.dim, self.n_local_heads * self.head_dim * 2)
self.attn_k_progcfg  = create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
self.attn_v_progcfg  = create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
self.attn_wo_progcfg = create_dram_sharded_matmul_program_config(M, self.attn_out_dim_tp, self.dim)
```

## The _shard_linear Pattern

All decode projections in `Qwen35Attention` use the `_shard_linear` helper (attention.py:28–35):

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

1. **Input sharding**: `ttnn.to_memory_config(x_tt, act_shard_cfg)` moves the activation from DRAM interleaved to L1 WIDTH_SHARDED. The `act_shard_hidden` config (model_config.py:308) distributes the 5120-dim hidden state across compute cores.
2. **Unshard to DRAM**: The `_unshard()` helper (attention.py:38–41) moves the result back to DRAM interleaved via `ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)` for subsequent reshape and slice operations.

This pattern is applied three times in `forward_decode()` for Q+gate, K, and V projections (attention.py:151–153), and once more for the output projection `wo` (attention.py:244), which uses `act_shard_attn_out` since its input dimension is `NH*HD = 1536` per device rather than `dim = 5120`.

## Compute Kernel Configuration

All decode matmuls use the HiFi2 compute kernel configuration defined in `model_config.py:177–182` and referenced as `self.compute_cfg = args.compute_kernel_config_hifi2` in `Qwen35Attention.__init__()` (attention.py:82):

```python
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

- **HiFi2 math fidelity**: provides a balance between accuracy and throughput suitable for BFP8 weight matmuls
- **FP32 destination accumulation** (`fp32_dest_acc_en=True`): partial products accumulate in FP32 before conversion to bfloat16, reducing numerical error from BFP8 weights
- **Packer L1 accumulation** (`packer_l1_acc=True`): enables the packer to accumulate results in L1 before writing, reducing NOC traffic

## Per-Head KV Cache Update

After Q, K, and V are projected and normalized, the K and V tensors must be inserted into the KV cache at the current decode position. The implementation supports two paths.

### Paged KV Cache (vLLM integration)

When `kv_cache` is provided externally (attention.py:177–189):

```python
ttnn.experimental.paged_update_cache(keys,   k, update_idxs_tensor=cur_pos_tt, page_table=page_table)
ttnn.experimental.paged_update_cache(values, v, update_idxs_tensor=cur_pos_tt, page_table=page_table)
```

### Internal Per-Head KV Cache (standalone mode)

When using the internal cache (`self.k_caches` / `self.v_caches`), each KV head is updated separately (attention.py:200–214):

```python
for h in range(NKV):
    k_h        = ttnn.slice(k, (0, 0, h, 0), (1, B, h + 1, HD))
    v_h        = ttnn.slice(v, (0, 0, h, 0), (1, B, h + 1, HD))
    k_h_padded = ttnn.pad(k_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)   # pad to tile height
    v_h_padded = ttnn.pad(v_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
    k_sh       = ttnn.to_memory_config(k_h_padded, self._kv_update_shard_cfg)
    v_sh       = ttnn.to_memory_config(v_h_padded, self._kv_update_shard_cfg)
    ttnn.experimental.paged_update_cache(self.k_caches[h], k_sh, update_idxs_tensor=cur_pos_tt)
    ttnn.experimental.paged_update_cache(self.v_caches[h], v_sh, update_idxs_tensor=cur_pos_tt)
```

The `_kv_update_shard_cfg` is a HEIGHT_SHARDED config defined in `model_config.py:313–319` with shard shape `(TILE_SIZE, head_dim)` = `(32, 256)` on an 8x4 grid:

```python
self.kv_update_shard_cfg = ttnn.create_sharded_memory_config(
    shape=(TILE_SIZE, self.head_dim),   # (32, 256)
    core_grid=ttnn.CoreGrid(x=8, y=4),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
```

Each of the 32 cores (8x4 grid) holds one user's KV entry — a single tile of shape `[32, 256]`. The padding to `[1, B, 32, HD]` ensures the head-count dimension is tile-aligned before sharding.

After cache updates, the full KV tensors are assembled for SDPA. In the internal-cache path with `NKV = 1`, the single cache tensor is used directly (attention.py:218–229):

```python
if NKV == 1:
    k_full, v_full = self.k_caches[0], self.v_caches[0]
else:
    k_full = ttnn.concat([self.k_caches[h] for h in range(NKV)], dim=1)
    v_full = ttnn.concat([self.v_caches[h] for h in range(NKV)], dim=1)

attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
    q, k_full, v_full,
    cur_pos_tensor=cur_pos_tt,
    scale=self.scale,                   # head_dim ** -0.5
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

---

**Next:** [`flash_attention_prefill.md`](./flash_attention_prefill.md)
