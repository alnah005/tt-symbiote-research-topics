# Head-Parallel State Sharding (Recommended)

This section specifies the recommended sharding strategy for Gated Delta Net on T3K: **shard the value head dimension across 8 devices**. Each device holds a contiguous block of 4 value heads and computes the full recurrent step for those heads independently.

Symbols used throughout: `H_v = 32` (total value heads), `H_k = 16` (total key heads), `d_k = d_v = 128`, `N = 8` (number of T3K devices).

---

## 1. State Matrix Sharding

The full recurrent state per layer is:

```
Full state per layer (all heads, B=1):
  [B, H_v, d_k, d_v] in BF16
= [1, 32, 128, 128] × 2 bytes
= 1,048,576 bytes  = 1 MiB per layer
```

Under head-parallel sharding, each device holds:

```
Per-device state per layer (B=1):
  [B, H_v/N, d_k, d_v] = [1, 4, 128, 128] in BF16
= 1 × 4 × 128 × 128 × 2 bytes
= 131,072 bytes  = 128 KiB per layer
```

Across all 30 Gated Delta Net layers:

```
Total recurrent state per device (B=1, 30 layers):
= 30 × 131,072 bytes
= 3,932,160 bytes
≈ 3.75 MiB
```

This is negligible relative to the 3.25 GB available per chip.

### Conv State

The causal conv1d runs on the mixed QKV input of dimension 8192 = H_k×d_k + H_k×d_k + H_v×d_v = 2048 + 2048 + 4096. Under head-parallel sharding, the conv state is also sharded over the output dimension:

```
Full conv state per layer:  [B, 8192, 4]  (4 = conv kernel size)
Per-device conv state:      [B, 1024, 4]  (8192 / 8 devices)

Per-device bytes (B=1):     1 × 1024 × 4 × 2 = 8,192 bytes = 8 KiB per layer
Total across 30 layers:     30 × 8,192 = 245,760 bytes ≈ 240 KiB
```

Also negligible.

---

## 2. Projection Sharding

### Input Projections

`in_proj_qkv` maps `[B, T, H] → [B, T, 8192]`. Under column sharding over the output dimension:

- Each device computes `[B, T, 8192/8] = [B, T, 1024]` using a local weight shard.
- No CCL is required before the recurrent step: each device's 1024-element output is exactly the Q̃, K̃, V for its 4 value heads (after reshape and repeat-interleave of K heads).
- Weight shard size: `[2048, 1024]` in BF16 = 2048 × 1024 × 2 = 4 MiB.

`in_proj_z` maps `[B, T, H] → [B, T, 4096]` (the gate for gated RMSNorm). Under column sharding:

- Each device computes `[B, T, 512]` (its 4-head share of the 4096-dim gate output).
- No CCL required.

`in_proj_a` and `in_proj_b` map `[B, T, H] → [B, T, H_v]` = `[B, T, 32]`. The output dimension is small (32 scalars). These use replicated weights across devices; each device computes the full `[B, T, 32]` output locally and uses only its 4-head slice (`[B, T, 4]`).

### Output Projection

`out_proj` maps `[B, T, 4096] → [B, T, H]` = `[B, T, 2048]`. Under row-parallel sharding:

- Each device holds `[B, T, 512]` of the 4096-dim input (its 4 heads × 128-dim output from gated RMSNorm).
- Each device multiplies by a weight shard `[512, 256]` (= `[4096/8, 2048/8]`) to produce `[B, T, 256]`.
- All-gather over 8 devices assembles `[B, T, 2048]`.
- Weight shard size: `[512, 256]` in BF16 = 512 × 256 × 2 = 256 KiB.

---

## 3. Communication Pattern Per Decode Step (B=1, T=1)

The full decode communication flow per Gated Delta Net layer:

| Step | Operation | CCL? | Description |
|---|---|---|---|
| 1 | `in_proj_qkv` | No | Col-sharded matmul; output stays local |
| 2 | `in_proj_z` | No | Col-sharded matmul; output stays local |
| 3 | `in_proj_a`, `in_proj_b` | No | Replicated weights; each device selects its 4 scalars |
| 4 | Causal conv1d update | No | Local op on sharded conv state |
| 5 | Recurrent delta rule (4 heads) | **No** | Entirely local; no cross-device dependency |
| 6 | Gated RMSNorm | No | Local op |
| 7 | `out_proj` (row-parallel) | No | Local partial matmul |
| 8 | All-gather output | **Yes** | Assemble `[1, 1, 2048]` from 8 × `[1, 1, 256]` |

**Only one CCL per layer: the final all-gather of the output projection.**

### CCL Cost Analysis

```
All-gather payload (per layer, B=1, T=1):
  8 devices × [1, 1, 256] × 2 bytes = 4,096 bytes = 4 KiB

Time at 25 GB/s:
  4,096 / 25 × 10^9 = 164 ns ≈ 0.16 µs per layer
```

For comparison, the dominant cost is state matrix DRAM I/O (from Chapter 5):

```
State I/O time per layer (B=1): 7.36 µs

CCL overhead as fraction of state I/O: 0.16 / 7.36 ≈ 2.2%
```

The all-gather is entirely dominated by DRAM bandwidth. T3K scaling introduces negligible overhead for Gated Delta Net decode.

---

## 4. Per-Device Arithmetic Intensity After Sharding

With 4 heads per device (instead of 32), the per-device FLOPs and bytes scale together:

```
Per-device FLOPs (decode, per layer):
  32 heads → 4 heads: 3,678,208 / 8 = 459,776 FLOP

Per-device bytes (decode, per layer):
  2,121,728 / 8 = 265,216 bytes

Arithmetic intensity: 459,776 / 265,216 = 1.73 FLOP/byte
```

Sharding does not change the arithmetic intensity — it scales FLOPs and bytes by the same factor. Each device remains as deeply memory-bandwidth-bound (262× below ridge) as the single-device case, but with 4× smaller state per device, L1-resident state is equally feasible: 4 heads × 32 KiB = 128 KiB, well within one Tensix core's 1.5 MiB L1.

---

**Previous:** [`t3k_mesh_topology.md`](./t3k_mesh_topology.md) | **Next:** [`alternative_sharding_strategies.md`](./alternative_sharding_strategies.md)
