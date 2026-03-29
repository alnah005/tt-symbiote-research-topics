# Kernel Dispatch: Python-Side Program Construction and Launch

The fused GDN kernel is dispatched from Python through the `gdn_kernel_op.py` module, which constructs a `ProgramDescriptor` (or `MeshProgramDescriptor` for multi-device) containing kernel descriptors for the reader, compute, and writer, along with all circular buffer definitions and per-core runtime arguments. The entire program is launched via a single `ttnn.generic_op` call.

This section covers the public API, program construction, circular buffer layout, argument conventions, and the multi-device dispatch path.

## Public API: `gdn_full_fused_inplace`

The entry point is `gdn_full_fused_inplace()`, which accepts 11 tensors plus configuration parameters:

```python
def gdn_full_fused_inplace(
    conv_out,       # [1, B, qkv_dim_tp]  -- post-conv1d output (Q/K/V packed)
    a_fused,        # [1, B, Nv_TP]       -- gate input a (batched)
    b_fused,        # [1, B, Nv_TP]       -- gate input b (batched)
    neg_exp_A,      # [1, 1, Nv_TP]       -- precomputed -exp(A) constant
    dt_bias,        # [1, 1, Nv_TP]       -- dt_bias constant
    norm_w,         # [1, 1, Dv]          -- RMS norm weight
    scale_tt,       # [1, 1, 1]           -- Q scale = Dk^(-0.5)
    rms_scale_tt,   # [1, 1, 1]           -- sqrt(Dv)
    rms_eps_tt,     # [1, 1, 1]           -- Dv * eps
    state,          # [num_pairs, Dk, Dv] -- recurrence state (in-place update)
    output,         # [num_pairs, 1, Dv]  -- pre-allocated output buffer
    num_pairs,      # B * Nv_TP = 384
    num_cores=40,
    Nv_TP=12, Nk_TP=4, repeat_factor=3, key_dim_tp=512,
):
```

The function detects whether the full fused kernel is available (controlled by the `GDN_DISABLE_FULL_FUSED` environment variable). If the kernel fails at runtime, `_full_fused_available` is set to `False` globally and subsequent calls raise `RuntimeError`, signaling the caller to use the unfused path.

Note that `z` (the SiLU gate projection) is **not** passed to the kernel. The SiLU gate is applied by the caller in Python after the kernel returns, via `ttnn.silu`. This design decision keeps the kernel focused on the recurrence-critical path and avoids adding another tensor input.

## Program Construction: `_build_full_fused_device_program`

The core of the dispatch is `_build_full_fused_device_program()`, which builds a `ProgramDescriptor` for a single device. It receives the per-device tensor handles (already extracted from the mesh via `ttnn.get_device_tensors`) and constructs:

1. Core coordinate assignment
2. Per-core runtime arguments (buffer addresses + pair assignments)
3. Circular buffer descriptors
4. Kernel descriptors for reader, compute, and writer

### Core Assignment

Pairs are distributed across cores using a simple division with remainder:

```python
num_cores = min(num_cores, num_pairs_total, max_cores)
pairs_per_core = num_pairs_total // num_cores
remainder = num_pairs_total % num_cores
```

Cores are assigned in column-major order across the compute grid: `CoreCoord(i % grid.x, i // grid.x)`. The first `remainder` cores each get one extra pair. For the typical case of `num_pairs=384` and `num_cores=40`, this gives 9 pairs per core with 24 cores getting an extra pair (10 pairs each).

Cores with different pair counts are grouped separately for compile-time specialization -- the `num_pairs` compile-time argument differs between groups, so each group gets its own set of `KernelDescriptor` objects.

### Runtime Arguments

**Reader runtime args** (12 values per core):

| Index | Value | Description |
|-------|-------|-------------|
| 0 | `conv_out_dev.buffer_address()` | DRAM address of packed `[1, B, qkv_dim_tp]` tensor |
| 1 | `a_dev.buffer_address()` | DRAM address of `[1, B, Nv_TP]` gate input a |
| 2 | `b_dev.buffer_address()` | DRAM address of `[1, B, Nv_TP]` gate input b |
| 3 | `neg_exp_A_dev.buffer_address()` | DRAM address of `[1, 1, Nv_TP]` constant |
| 4 | `dt_bias_dev.buffer_address()` | DRAM address of `[1, 1, Nv_TP]` constant |
| 5 | `norm_w_dev.buffer_address()` | DRAM address of `[1, 1, Dv]` RMS norm weight |
| 6 | `scale_dev.buffer_address()` | DRAM address of Q scale tile |
| 7 | `rms_scale_dev.buffer_address()` | DRAM address of RMS scale tile |
| 8 | `state_dev.buffer_address()` | DRAM or L1 address of recurrence state |
| 9 | `rms_eps_dev.buffer_address()` | DRAM address of RMS epsilon tile |
| 10 | `pair_offset` | First pair index assigned to this core |
| 11 | `n` | Number of pairs on this core |

**Writer runtime args** (4 values per core):

| Index | Value | Description |
|-------|-------|-------------|
| 0 | `output_dev.buffer_address()` | DRAM address of output tensor |
| 1 | `state_dev.buffer_address()` | DRAM or L1 address for state writeback |
| 2 | `pair_offset` | First pair index assigned to this core |
| 3 | `n` | Number of pairs on this core |

Runtime args are passed via `ttnn.RuntimeArgs`, keyed by `[core_x][core_y]` coordinates.

### Compile-Time Arguments

**Reader compile-time args** (11 values):

| Index | Value | Description |
|-------|-------|-------------|
| 0 | `Kt = 4` | Key dimension in tiles (`Dk / 32`) |
| 1 | `Vt = 4` | Value dimension in tiles (`Dv / 32`) |
| 2 | `BF16_TILE_BYTES = 2048` | Tile size in bytes |
| 3 | `state_l1_flag` | 1 if state is in L1, 0 if DRAM |
| 4 | `packed_reduce_scaler = 0x3F803F80` | Packed bfloat16 1.0 for reduce operations |
| 5 | `Nv_TP = 12` | Value heads per device |
| 6 | `Nk_TP = 4` | Key heads per device |
| 7 | `repeat_factor = 3` | Value heads per key head (`Nv_TP / Nk_TP`) |
| 8 | `key_tile_off = 16` | Tile offset to K region in `conv_out` (`key_dim_tp / 32`) |
| 9 | `v_tile_off = 32` | Tile offset to V region in `conv_out` (`2 * key_dim_tp / 32`) |
| 10 | `sharded_flag` | 1 if state is HEIGHT_SHARDED in L1, 0 otherwise |

**Writer compile-time args** (7 values):

| Index | Value | Description |
|-------|-------|-------------|
| 0 | `Kt = 4` | Key dimension in tiles |
| 1 | `Vt = 4` | Value dimension in tiles |
| 2 | `BF16_TILE_BYTES = 2048` | Tile size in bytes |
| 3 | `state_l1_flag` | 1 if state is in L1 |
| 4 | 0 | Unused (Nv_TP, kept for compatibility) |
| 5 | 0 | Unused (out_tiles_per_batch, kept for compatibility) |
| 6 | `sharded_flag` | 1 if state is HEIGHT_SHARDED |

**Compute compile-time args** (3 values):

| Index | Value | Description |
|-------|-------|-------------|
| 0 | `Kt = 4` | Key dimension in tiles |
| 1 | `Vt = 4` | Value dimension in tiles |
| 2 | `n_pairs` | Number of pairs assigned to this core group |

The compute kernel also receives a `ComputeConfigDescriptor` specifying `MathFidelity.HiFi4`, `fp32_dest_acc_en=True`, and `math_approx_mode=False` for maximum numerical precision during the recurrence.

## Circular Buffer Descriptors

The program defines 28 circular buffers. Each is created via the `_make_cb` helper with a CB index, tile count, and the core range set. All use bfloat16 format with 2048-byte pages.

| CB Index | Name | Tiles | Role |
|----------|------|-------|------|
| `c_0` | `cb_q_raw` | Kt=4 | Raw Q head from conv_out (reader fills) |
| `c_1` | `cb_k_raw` | Kt=4 | Raw K head from conv_out (reader fills) |
| `c_2` | `cb_k_col` | Kt=4 | K transposed (compute fills for outer product) |
| `c_3` | `cb_v` | Vt=4 | V head from conv_out (reader fills) |
| `c_4` | `cb_g` | 1 | Computed decay gate (compute fills) |
| `c_5` | `cb_beta` | 1 | sigmoid(b) gate (compute fills) |
| `c_6` | `cb_state_in` | 16 | Recurrence state input (reader fills) |
| `c_7` | `cb_state_b` | 16 | Decayed state = state * exp(g) (compute intermediate) |
| `c_8` | `cb_state_out` | 16 | Updated state after recurrence (compute fills, writer reads) |
| `c_9` | `cb_a` | 1 | Gate input a scalar (reader fills) |
| `c_10` | `cb_b` | 1 | Gate input b scalar (reader fills) |
| `c_12` | `cb_neg_exp_A` | 1 | -exp(A) constant (reader fills per pair) |
| `c_13` | `cb_dt_bias` | 1 | dt_bias constant (reader fills per pair) |
| `c_14` | `cb_norm_w` | Vt=4 | RMS norm weight (reader fills once, persistent) |
| `c_15` | `cb_scale` | 1 | Q scale Dk^(-0.5) (reader fills once, persistent) |
| `c_16` | `cb_out` | Vt=4 | Final output tiles (compute fills, writer reads) |
| `c_17` | `cb_q` | Kt=4 | L2-normed and scaled Q (compute fills) |
| `c_18` | `cb_k_row` | Kt=4 | L2-normed K row (compute fills) |
| `c_19` | `cb_reduce_scaler` | 1 | All-ones tile for reduce operations (persistent) |
| `c_20` | `cb_rms_eps` | 1 | Dv * eps for RMS norm stability (persistent) |
| `c_21` | `cb_scratch` | 1 | Reader scratch buffer for sub-tile extraction |
| `c_24` | `cb_exp_g` | 1 | exp(g) for state decay (compute intermediate) |
| `c_25` | `cb_kv_mem` | Vt=4 | k_row @ state result (compute intermediate) |
| `c_26` | `cb_delta` | Vt=4 | v - kv_mem (compute intermediate) |
| `c_27` | `cb_delta_s` | Vt=4 | beta * delta (compute intermediate) |
| `c_28` | `cb_sq_acc` | Kt=4 | Squared norm accumulator (compute intermediate) |
| `c_29` | `cb_tmp` | 1 | General scratch for norm computation (compute intermediate) |
| `c_31` | `cb_rms_scale` | 1 | sqrt(Dv) for RMS norm (persistent) |

Total L1 per core: `(4+4+4+4+1+1+16+16+16+1+1+1+1+4+1+4+4+4+1+1+1+1+4+4+4+4+1+1) * 2048 = 109 tiles * 2 KB = 218 KB`. This is well within the ~1.2 MB L1 budget per core on Blackhole.

Note: `c_11` (originally `cb_z`) and `c_30` (originally `cb_rec_out`) were removed during development. The SiLU gate is handled in Python, and the recurrence output is written directly to `cb_out` rather than through an intermediate buffer.

## Multi-Device Dispatch

For the P150x4 with TP=4, each device processes its own shard of the data. The dispatch path is:

1. Call `ttnn.get_device_tensors()` on each of the 11 input tensors to get per-device views
2. Iterate over the `mesh_shape` (1x4 for P150x4)
3. For each device, extract that device's tensor handles and call `_build_full_fused_device_program` with the device-local buffer addresses
4. Wrap all per-device programs in a `MeshProgramDescriptor`, keyed by `MeshCoordinateRange`
5. Call `ttnn.generic_op(all_tensors, mesh_program)` to dispatch to all devices simultaneously

The critical detail is that each device's program gets its own buffer addresses via `devs[i].buffer_address()`. Even though the program structure (kernel sources, CB layout, compile-time args) is identical across devices, the runtime args differ because each device's tensors reside at different physical addresses.

## Program Cache and Kernel Hashing

The module computes an MD5 hash of all kernel source files at import time via `_compute_kernel_hash()`:

```python
def _compute_kernel_hash():
    h = hashlib.md5()
    for path in [READER_PATH, WRITER_PATH, ..., COMPUTE_FUSED_PATH]:
        full = os.path.join(tt_home, path)
        with open(full, "rb") as f:
            h.update(f.read())
    return int(h.hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF
```

This hash is used for program cache invalidation: when kernel source files change during development, the hash changes, forcing recompilation. The hash is computed once at module load time and stored in `_KERNEL_CONTENT_HASH`.

## Fallback and Error Handling

The dispatch follows a two-tier fallback strategy:

1. **Full fused kernel** (`gdn_full_fused_inplace`): Tries `_gdn_full_fused` first. If it raises an exception, sets `_full_fused_available = False` and raises `RuntimeError` to signal the caller.

2. **Caller fallback**: The `TtGatedDeltaNet._forward_decode_fused` method catches the error and falls back to `_forward_decode_unfused`, which uses individual `ttnn` operations for each step of the pipeline.

The `GDN_DISABLE_FULL_FUSED` environment variable can be set to skip the fused kernel entirely, which is useful for debugging and validation.

---

**Previous:** [`index.md`](./index.md) | **Next:** [`reader_kernel.md`](./reader_kernel.md)
