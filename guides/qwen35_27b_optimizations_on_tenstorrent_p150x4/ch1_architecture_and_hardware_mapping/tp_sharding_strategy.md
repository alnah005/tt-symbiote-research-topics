# TP Sharding Strategy: Mapping Qwen3.5-27B to 4 Devices

The P150x4 provides 4 Blackhole chips connected in a ring topology. Qwen3.5-27B uses TP=4 tensor parallelism, distributing weight matrices and activations across all 4 devices. This section details the exact dimension splits, the column-parallel and row-parallel sharding patterns, and the weight preparation helpers that make clean TP slicing possible.

## TP-Derived Dimensions

`Qwen35ModelArgs.__init__()` computes all TP-local dimensions from the global constants and `tp = self.num_devices = 4` (see `model_config.py:263-270`):

| Dimension | Formula | Value | Description |
|-----------|---------|-------|-------------|
| `gdn_nk_tp` | `GDN_Nk / tp` | 4 | GDN key heads per device |
| `gdn_nv_tp` | `GDN_Nv / tp` | 12 | GDN value heads per device |
| `gdn_qkv_dim_tp` | `GDN_QKV_DIM / tp` | 2560 | GDN Q+K+V output dim per device |
| `gdn_z_dim_tp` | `GDN_Z_DIM / tp` | 1536 | GDN Z (gate) output dim per device |
| `gdn_qkvz_dim_tp` | `(GDN_QKV_DIM + GDN_Z_DIM) / tp` | 4096 | Fused QKVZ output dim per device |
| `gdn_value_dim_tp` | `GDN_VALUE_DIM / tp` | 1536 | GDN value dim per device |
| `gdn_key_dim_tp` | `GDN_KEY_DIM / tp` | 512 | GDN key dim per device |
| `attn_out_dim_tp` | `(n_heads * head_dim) / tp` | 1536 | Attention output dim per device |
| `n_local_heads` | `n_heads / tp` | 6 | Attention Q heads per device |
| `n_local_kv_heads` | `max(1, n_kv_heads / tp)` | 1 | Attention KV heads per device |

The `gdn_qkvz_dim_tp = 4096` value is particularly important: it is the output dimension of the single fused QKVZ matmul that drives each GDN layer, computed as $(\text{GDN QKV DIM} + \text{GDN Z DIM}) / 4 = 4096$ — for the derivation of those aggregate constants see `hybrid_architecture.md` (see `model_config.py:267`).

## Column-Parallel vs Row-Parallel Projections

Every linear projection in the model follows one of two sharding patterns:

### Column-Parallel (Shard Output Dimension)

The weight matrix is split along the output dimension so each device computes a slice of the output independently. No communication is needed after the matmul.

**Pattern:** $[\text{hidden}, \text{full out}] \rightarrow \text{device } d \text{ gets } [\text{hidden}, \text{out tp}]$

In tt-metal, the weight is stored transposed as $[\text{in}, \text{out}]$, so column-parallel corresponds to `ShardTensorToMesh(dim=-1)` on the transposed weight. All column-parallel weight configs are created with `create_dram_sharded_mem_config(self.dim, out_dim_tp)` (see `model_config.py:277-282`):

| Projection | Input Dim | Full Output Dim | Per-Device Output | Config Attribute |
|-----------|-----------|-----------------|-------------------|-----------------|
| GDN QKVZ | 5120 | 16384 | 4096 | `gdn_qkvz_weight_memcfg` |
| GDN A+B | 5120 | 96 ($2 \times N_v$) | 24 ($2 \times N_{v,\text{TP}}$) | (DRAM\_MEMORY\_CONFIG) |
| Attention Q+gate | 5120 | 12288 ($24 \times 256 \times 2$) | 3072 | `attn_qg_weight_memcfg` |
| Attention K | 5120 | 1024 ($4 \times 256$) | 256 | `attn_k_weight_memcfg` |
| Attention V | 5120 | 1024 ($4 \times 256$) | 256 | `attn_v_weight_memcfg` |
| MLP w1 (gate) | 5120 | 17408 | 4352 | `mlp_w1_weight_memcfg` |
| MLP w3 (up) | 5120 | 17408 | 4352 | `mlp_w3_weight_memcfg` |

### Row-Parallel (Shard Input Dimension)

The weight matrix is split along the input dimension. Each device computes a partial sum, and an **all-reduce** across all 4 devices produces the final output.

**Pattern:** $[\text{full in}, \text{hidden}] \rightarrow \text{device } d \text{ gets } [\text{in tp}, \text{hidden}]$

In tt-metal, this corresponds to `ShardTensorToMesh(dim=0)` on the transposed weight (`dim=0` is the input dimension after transposition). Row-parallel weight configs are created with `create_dram_sharded_mem_config(in_dim_tp, self.dim)` (see `model_config.py:285-287`):

| Projection | Per-Device Input | Output Dim | Config Attribute |
|-----------|-----------------|------------|-----------------|
| GDN output | 1536 ($N_{v,\text{TP}} \times D_v$) | 5120 | `gdn_out_weight_memcfg` |
| Attention wo | 1536 ($6 \times 256$) | 5120 | `attn_wo_weight_memcfg` |
| MLP w2 (down) | 4352 ($17408 / 4$) | 5120 | `mlp_w2_weight_memcfg` |

The all-reduce uses the CCL ring topology on the P150x4's 4-chip interconnect. The `tt_ccl` object is created by the framework `TTTransformer` and passed to both `Qwen35Attention` and `TtGatedDeltaNet` constructors (see `model.py:80-81` and `gdn.py:93`).

## DRAM-Sharded Weight Storage

All decode projection weights are stored in DRAM using a WIDTH\_SHARDED layout across 8 DRAM cores per device. The `create_dram_sharded_mem_config(k, n)` function (see `model_config.py:80-92`) builds the memory config:

1. Pad $n$ up to the nearest multiple of $\text{TILE SIZE} \times \text{DRAM CORES} = 32 \times 8 = 256$
2. Create a `ShardSpec` with the 8-core DRAM grid (`CoreRange(CoreCoord(0,0), CoreCoord(7,0))`) and shard shape $(k, \text{padded n} / 8)$
3. Wrap in a `WIDTH_SHARDED` DRAM `MemoryConfig`

The corresponding matmul program config is built by `create_dram_sharded_matmul_program_config(m, k, n)` (see `model_config.py:95-118`), which produces a `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`. For decode, $M=1$ (single token), making these matmuls bandwidth-bound. For example (see `model_config.py:291`):

```python
self.gdn_qkvz_weight_memcfg = create_dram_sharded_mem_config(self.dim, self.gdn_qkvz_dim_tp)
self.gdn_qkvz_progcfg = create_dram_sharded_matmul_program_config(M=1, self.dim, self.gdn_qkvz_dim_tp)
```

For prefill, 2D matmul configs are created dynamically via `self.prefill_progcfg(seq_len, k, n)`, which calls `create_prefill_matmul_program_config` with an 8x8 grid (see `model_config.py:300-305`). This is covered in Chapter 5.

## Compute Kernel Configs

Two compute kernel configurations are defined globally in `model_config.py` (lines 177-189):

- **`COMPUTE_HIFI2`**: `MathFidelity.HiFi2`, `math_approx_mode=True`, `fp32_dest_acc_en=True`. Used for BFP8 weight projections (matmuls) where approximate math is acceptable.
- **`COMPUTE_HIFI4`**: `MathFidelity.HiFi4`, `math_approx_mode=False`, `fp32_dest_acc_en=True`. Used for the GDN recurrence computation where numerical precision is critical.

Both enable FP32 destination accumulation and L1 packer accumulation (`packer_l1_acc=True`). `TtGatedDeltaNet` uses `COMPUTE_HIFI2` for weight projections (see `gdn.py:116`: `self.compute_cfg = args.compute_kernel_config_hifi2`).

## Weight Preparation Helpers

The raw HuggingFace weights are not laid out for clean TP sharding. Three helper functions in `model_config.py` reorder weights before distributing them to devices.

### `prepare_gdn_qkv(sd, prefix, tp)`

The HF weight `in_proj_qkv.weight` stores Q, K, V blocks contiguously (see `model_config.py:346-362`):

```
Original: [Q_all (2048) | K_all (2048) | V_all (6144)]
```

This function splits each block into per-device shards and interleaves them so that `ShardTensorToMesh(dim=-1)` (via `_shard_w(..., dim=-1)`) on the transposed weight produces the correct per-device grouping:

```python
for s in range(tp):
    q_s = q_part[s * q_per * GDN_Dk : (s+1) * q_per * GDN_Dk, :]  # 4 key heads * 128 = 512
    k_s = k_part[s * q_per * GDN_Dk : (s+1) * q_per * GDN_Dk, :]  # 4 key heads * 128 = 512
    v_s = v_part[s * v_per * GDN_Dv : (s+1) * v_per * GDN_Dv, :]  # 12 value heads * 128 = 1536
    shards.append(torch.cat([q_s, k_s, v_s], dim=0))
```

Each device shard contains $[\text{Q tp} (512) \mid \text{K tp} (512) \mid \text{V tp} (1536)]$ = 2560 elements along the output dimension, matching `gdn_qkv_dim_tp = 2560`.

The fused QKVZ weight is then constructed in `_load_and_wire_attention_weights()` (see `model.py:149-164`) by concatenating the Z projection onto each device's QKV shard:

```python
for d in range(tp):
    fused_parts.append(torch.cat([
        qkv_reordered[d * qkv_per : (d+1) * qkv_per, :],  # 2560 rows (QKV)
        z_weight[d * z_per : (d+1) * z_per, :],            # 1536 rows (Z)
    ], dim=0))
```

Final per-device QKVZ dimension: $2560 + 1536 = 4096 =$ `gdn_qkvz_dim_tp`.

### `prepare_attn_qg(sd, prefix, n_heads, head_dim, tp)`

For full attention layers, the HF weight `attention.wqkv.weight` already stores Q and gate in per-head interleaved format (see `model_config.py:365-372`):

```
[Q_h0(256), gate_h0(256), Q_h1(256), gate_h1(256), ...]
```

`prepare_attn_qg()` returns this weight unchanged because `ShardTensorToMesh(dim=-1)` on the transposed weight naturally groups contiguous heads per device, preserving the per-head $[\text{Q}, \text{gate}]$ layout expected by the forward pass. Each device gets 6 heads' worth of Q+gate = $6 \times 256 \times 2 = 3072$ elements.

### `prepare_conv_taps(sd, prefix, tp)`

The 4-tap causal conv1d weight has shape `[qkv_dim, 1, 4]`. Each of the 4 taps is extracted, split into Q/K/V blocks, and interleaved per-device in the same pattern as the QKV weight (see `model_config.py:375-393`):

```python
for j in range(GDN_CONV_KERNEL_SIZE):  # 4 taps
    tap = cw[:, 0, j]
    q_tap = tap[:GDN_KEY_DIM]               # 2048
    k_tap = tap[GDN_KEY_DIM:2*GDN_KEY_DIM]  # 2048
    v_tap = tap[2*GDN_KEY_DIM:]             # 6144
    for s in range(tp):
        q_s = q_tap[s * q_per * GDN_Dk : (s+1) * q_per * GDN_Dk]
        k_s = k_tap[s * q_per * GDN_Dk : (s+1) * q_per * GDN_Dk]
        v_s = v_tap[s * v_per * GDN_Dv : (s+1) * v_per * GDN_Dv]
        shards.append(torch.cat([q_s, k_s, v_s]))
```

This ensures each device's conv tap weights align with its QKV shard ordering, so the conv output fed to the QKVZ matmul is already in the correct per-device layout.

## KV Head Replication

With TP=4 and only 4 KV heads, each device gets exactly 1 KV head: `n_local_kv_heads = max(1, 4 // 4) = 1`. No replication is needed in this configuration — `kv_replication = False` because `tp == n_kv_heads` (see `model_config.py:253`: `self.kv_replication = tp > self.n_kv_heads`).

However, the code supports TP=8 where replication would be required. The `replicate_kv_weight()` function handles this (see `model_config.py:398-414`):

```python
def replicate_kv_weight(weight, n_kv_heads, tp, head_dim):
    chunks = weight.reshape(n_kv_heads, head_dim, -1)
    for d in range(tp):
        kv_idx = (d * n_kv_heads) // tp
        parts.append(chunks[kv_idx])
    return torch.cat(parts, dim=0).reshape(tp * head_dim, -1)
```

Each device $d$ gets KV head $(d \times n_{kv} ) // \text{tp}$. For TP=8 with 4 KV heads, devices 0–1 share head 0, devices 2–3 share head 1, and so on.

The `Qwen35ModelArgs._set_params_from_dict()` method temporarily bumps `n_kv_heads` to `num_devices` during parent construction to satisfy the parent `ModelArgs` assertion that `n_kv_heads % num_devices == 0`, then restores the real value after init (see `model_config.py:223-242`).

## Mesh Tensor Helpers

Three utility functions in `model_config.py` convert PyTorch weights to on-device mesh tensors (lines 417-462):

| Function | Purpose | Dtype | Mesh Mapper |
|----------|---------|-------|-------------|
| `_shard_w(tensor, mesh, dim, memcfg, path)` | Large weight matrices | `bfloat8_b` | `ShardTensorToMesh(dim=dim)` |
| `_replicate(tensor, mesh, path)` | Norms, biases | `bfloat16` | `ReplicateTensorToMesh` |
| `_shard_small(tensor, mesh, path)` | Per-head params (`A_log`, `dt_bias`) | `bfloat16` | `ShardTensorToMesh(dim=-1)` |

`_shard_w()` transposes the weight from HF layout `[out_features, in_features]` to tt-metal layout `[in_features, out_features]` before converting (see `model_config.py:419`: `w = torch_tensor.to(torch.bfloat16).T.contiguous()`), and stores in the DRAM-sharded memory config. All mesh tensors are cached to disk at `cache_path` for fast reloading on subsequent runs.

## CCL Topology

`Qwen35ModelArgs.__init__()` configures the sampling all-gather (see `model_config.py:323-327`):

```python
self.model_config["SAMPLING_AG_CONFIG"] = {
    "allow_force_argmax": True,
    "num_links": 1,
    "topology": self.ccl_topology() or ttnn.Topology.Linear,
}
```

## Activation Shard Configs

Activations flowing through the decode matmul pipeline use WIDTH\_SHARDED L1 memory configs, created by `create_activation_shard_config(k)` (see `model_config.py:121-133`):

| Config | Dimension | Usage |
|--------|-----------|-------|
| `act_shard_hidden` | 5120 | Input to column-parallel projections |
| `act_shard_gdn_value` | 1536 | GDN output projection input |
| `act_shard_attn_out` | 1536 | Attention output projection input |

The function finds an appropriate core grid (up to 8×8) such that `k / num_cores` divides evenly, then creates a WIDTH\_SHARDED config with shard shape `(32, width_per_core)` (see `model_config.py:308-310`).

For KV cache updates, a separate HEIGHT\_SHARDED config is used (see `model_config.py:313-319`):

```python
self.kv_update_shard_cfg = ttnn.create_sharded_memory_config(
    shape=(TILE_SIZE, self.head_dim),  # (32, 256)
    core_grid=ttnn.CoreGrid(x=8, y=4),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
```

This places per-head KV cache entries on a 32-core (8×4) grid for the `paged_update_cache` operation.

---

**Next:** [Chapter 2 — Full Attention Layer Optimizations](../ch2_full_attention_layer_optimizations/index.md)
