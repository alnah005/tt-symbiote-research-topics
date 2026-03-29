# Flash Attention Prefill: 2D Matmuls and SDPA on 8x8 Grid

During prefill, the attention layer processes an entire input sequence at once. Unlike decode (M=1, bandwidth-bound), prefill matmuls have M=seq_len and are **compute-bound**. This requires a fundamentally different matmul strategy: 2D multicast matmuls on the full 8x8 compute grid, with DRAM-interleaved inputs and outputs.

This section covers the `create_prefill_matmul_program_config` builder, the flash SDPA configuration, and the complete `forward_prefill()` flow in `Qwen35Attention`.

## 2D Matmul Program Configuration

The `create_prefill_matmul_program_config(m, k, n, grid_size=(8, 8))` function in `model_config.py` creates a `MatmulMultiCoreReuseMultiCastProgramConfig` that distributes work across a 2D grid:

```python
def create_prefill_matmul_program_config(m, k, n, grid_size=(8, 8)):
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))   # M parallelized over grid_y
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))   # N parallelized over grid_x

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)

    k_tiles = math.ceil(k / TILE_SIZE)
    in0_block_w = min(4, max(1, k_tiles // grid_size[0]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
```

Key design decisions:

- **M over grid_y, N over grid_x**: The M (sequence length) dimension is split across the 8 rows of the grid, while the N (output) dimension is split across the 8 columns. For a 96-token prefill with `per_core_M = ceil(96/32/8) = 1` and the Q+gate projection `per_core_N = ceil(3072/32/8) = 12`.
- **`out_subblock_w` respects FP32 DST limit**: The `_get_out_subblock_w` helper finds the largest subblock width such that `out_subblock_h * out_subblock_w <= 4`. This constraint comes from the Wormhole/Blackhole destination register file: with FP32 accumulation enabled, only 4 tiles fit in the DST register simultaneously.
- **`in0_block_w`**: Controls how many K-dimension tiles are loaded per iteration, capped at 4 to balance L1 usage against reuse.

### Contrast with Decode DRAM-Sharded

| Property | Decode (DRAM-Sharded) | Prefill (2D Multicast) |
|----------|----------------------|----------------------|
| Config type | `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` | `MatmulMultiCoreReuseMultiCastProgramConfig` |
| Activation location | L1 WIDTH_SHARDED | DRAM interleaved |
| Weight location | DRAM WIDTH_SHARDED across 8 cores | DRAM interleaved |
| Output location | L1 WIDTH_SHARDED | DRAM interleaved |
| Bottleneck | DRAM bandwidth (M=1) | Compute (M=seq_len) |
| Grid | Variable (from `_find_grid`) | Fixed 8x8 = 64 cores |

## Prefill Forward Pass

The `forward_prefill()` method in `Qwen35Attention` follows this sequence:

### 1. QKV Projections via 2D Matmul

The input `x` is first ensured to be 4D `[1, 1, seq_len, dim]` and moved to DRAM:

```python
x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

qg_progcfg = create_prefill_matmul_program_config(seq_len, dim, NH * HD * 2)
qg_tt = ttnn.linear(x_dram, tw["wqkv"], memory_config=ttnn.DRAM_MEMORY_CONFIG,
                     program_config=qg_progcfg, compute_kernel_config=self.compute_cfg)
```

Three separate `ttnn.linear` calls dispatch the Q+gate, K, and V projections. K and V share the same program config since they have the same output dimension (`NKV * HD`):

```python
k_progcfg = create_prefill_matmul_program_config(seq_len, dim, NKV * HD)
kp_tt = ttnn.linear(x_dram, tw["wk"], ...)
vp_tt = ttnn.linear(x_dram, tw["wv"], ...)
```

### 2. Reshape to Head Format

The projections are reshaped into head-major format for SDPA. The K/V reshape path depends on the number of local KV heads (`NKV`). On P150x4 with TP=4 and 4 KV heads, `NKV = 1`, so each device has a single KV head and the projection output `[1, 1, seq_len, HD]` is already in the correct shape. In this case, the code skips the reshape entirely and uses `ttnn.clone` to create an independent DRAM buffer:

```python
if NKV == 1:
    k = ttnn.clone(kp_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.clone(vp_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
else:
    k = ttnn.to_memory_config(ttnn.reshape(kp_tt, (1, NKV, seq_len, HD)), ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.to_memory_config(ttnn.reshape(vp_tt, (1, NKV, seq_len, HD)), ttnn.DRAM_MEMORY_CONFIG)
```

Q+gate is always reshaped to `[1, NH, seq_len, HD*2]` and then sliced:

```python
qg_r = ttnn.to_memory_config(ttnn.reshape(qg_tt, (1, NH, seq_len, HD * 2)), ttnn.DRAM_MEMORY_CONFIG)
q = ttnn.to_memory_config(ttnn.slice(qg_r, (0, 0, 0, 0), (1, NH, seq_len, HD)), ttnn.DRAM_MEMORY_CONFIG)
gate = ttnn.to_memory_config(ttnn.slice(qg_r, (0, 0, 0, HD), (1, NH, seq_len, HD * 2)), ttnn.DRAM_MEMORY_CONFIG)
```

Note the explicit `ttnn.to_memory_config(..., ttnn.DRAM_MEMORY_CONFIG)` and `ttnn.clone` calls throughout. The code comments explain this as forcing "independent DRAM buffers at each step to avoid buffer-sharing" -- a necessary precaution because reshape, slice, and the original projection output may share the same underlying buffer, which can cause data corruption if one is modified while the other is still in use.

### 3. QK Normalization and Partial RoPE

L2 normalization and partial RoPE are applied identically to decode, but now operating on the prefill tensor layouts:

```python
q = ttnn.multiply(_rms_norm_dev(q), tw["q_norm"])
k = ttnn.multiply(_rms_norm_dev(k), tw["k_norm"])

cos_tt, sin_tt = get_prefill_rot_mats(self.args._rope_setup_ref, seq_len, self.mesh_device)
q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH)    # [1, NH, seq_len, HD]
k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV)   # [1, NKV, seq_len, HD]
```

If the rope setup reference is unavailable, the method falls back to computing cos/sin tables from scratch using the same `inv_freq` formula as `Qwen35PartialRopeSetup`.

### 4. KV Cache Fill

For prefill, the entire sequence is written into the KV cache at once using `ttnn.fill_cache` (not `paged_update_cache` which is per-position):

```python
for h in range(NKV):
    k_h = ttnn.slice(k, (0, h, 0, 0), (1, h + 1, seq_len, HD))
    v_h = ttnn.slice(v, (0, h, 0, 0), (1, h + 1, seq_len, HD))
    ttnn.fill_cache(self.k_caches[h], k_h, 0)    # fill starting at position 0
    ttnn.fill_cache(self.v_caches[h], v_h, 0)
```

### 5. Flash SDPA

The Q, K, V tensors are typecast to `bfloat8_b` before SDPA for memory efficiency:

```python
q_8b = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
k_8b = ttnn.typecast(k, dtype=ttnn.bfloat8_b)
v_8b = ttnn.typecast(v, dtype=ttnn.bfloat8_b)
```

The SDPA program config uses the full 8x8 compute grid with dynamic chunk sizes:

```python
padded_seq = max(32, ((seq_len + 31) // 32) * 32)
q_chunk = min(256 if seq_len >= 2048 else 64, padded_seq)
k_chunk = min(256 if seq_len >= 2048 else 64, padded_seq)

sdpa_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    exp_approx_mode=False,
    q_chunk_size=q_chunk,
    k_chunk_size=k_chunk,
)
```

The chunk size selection is a throughput/memory tradeoff:

- **seq_len >= 2048**: Uses `q_chunk = k_chunk = 256` (sequence positions) for maximum throughput. Larger chunks amortize the overhead of loading Q/K/V blocks and reduce the number of softmax passes.
- **seq_len < 2048**: Uses `q_chunk = k_chunk = 64` (sequence positions) to avoid over-allocating circular buffer space for short sequences.
- **Both capped at `padded_seq`**: Ensures chunk size never exceeds the actual (padded) sequence length.

The `exp_approx_mode=False` setting uses exact exponential computation rather than a hardware approximation, preserving numerical accuracy for the softmax operation.

The SDPA call itself:

```python
attn_out = ttnn.transformer.scaled_dot_product_attention(
    q_8b, k_8b, v_8b,
    is_causal=True,
    scale=self.scale,                      # head_dim ** -0.5
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    program_config=sdpa_config,
)
```

The `is_causal=True` flag applies a causal attention mask, ensuring each position can only attend to itself and earlier positions. This is the standard autoregressive constraint.

### 6. Gating and Output Projection

After SDPA, the sigmoid gating is applied (identical to decode):

```python
gate_val = ttnn.sigmoid(gate)
gated = ttnn.multiply(attn_out, gate_val)
```

The gated output is reshaped from `[1, NH, seq_len, HD]` to `[1, 1, seq_len, NH*HD]` and passed through the output projection using another 2D matmul:

```python
gated_flat = ttnn.reshape(gated, (1, 1, seq_len, NH * HD))   # [1, 1, seq_len, 1536]
wo_progcfg = create_prefill_matmul_program_config(seq_len, NH * HD, dim)
wo_out = ttnn.linear(gated_flat, tw["wo"], memory_config=ttnn.DRAM_MEMORY_CONFIG,
                     program_config=wo_progcfg, compute_kernel_config=self.compute_cfg)
```

The result is then all-reduced across the 4 TP devices to produce the final output with the full `dim=5120` hidden dimension.

## Summary: Prefill vs Decode Data Flow

| Step | Decode | Prefill |
|------|--------|---------|
| Input shape | `[1, 1, B, dim]` = `[1, 1, 32, 5120]` | `[1, 1, seq_len, dim]` = `[1, 1, S, 5120]` |
| Matmul type | DRAM-sharded (bandwidth-bound) | 2D multicast (compute-bound) |
| Activation memory | L1 WIDTH_SHARDED | DRAM interleaved |
| Q+gate output | `[1, B, NH, HD*2]` = `[1, 32, 6, 512]` | `[1, NH, seq_len, HD*2]` = `[1, 6, S, 512]` |
| SDPA variant | `scaled_dot_product_attention_decode` | `scaled_dot_product_attention` (flash, causal) |
| SDPA precision | bfloat16 | bfloat8_b (typecast for memory) |
| KV cache op | `paged_update_cache` (per-position) | `fill_cache` (full sequence) |
| Output shape | `[1, 1, B, dim]` | `[1, 1, seq_len, dim]` |

---

**Previous:** [`dram_sharded_decode.md`](./dram_sharded_decode.md) | **Next:** [Chapter 3 — GDN Layer Decode Pipeline](../ch3_gdn_layer_decode_pipeline/index.md)
