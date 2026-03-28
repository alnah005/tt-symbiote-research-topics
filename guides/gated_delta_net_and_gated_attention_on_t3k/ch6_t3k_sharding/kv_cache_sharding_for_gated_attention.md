# KV Cache Sharding for Gated Attention

Gated Attention uses a standard paged KV cache managed by `TTNNQwenPagedAttentionKVCache`. This section specifies how the cache should be distributed across 8 T3K devices, and quantifies the memory budget implications at long context.

Symbols: `n_q_h = 16` (Q heads), `n_kv_h = 2` (KV heads), `d_h = 256` (head dimension), `N = 8` (devices).

---

## 1. The n_kv_h = 2 Constraint

Gated Attention uses extreme GQA: only 2 KV heads serve all 16 Q heads (8× expansion). With 8 T3K devices, head-parallel sharding of KV heads would require 2 / 8 = 0.25 KV heads per device — non-integer. Two integer-valued options are:

**Option A — Replicate KV, shard Q:**

All 8 devices hold the complete K and V cache (`n_kv_h = 2` heads). Q heads are sharded: each device holds `n_q_h / N = 16 / 8 = 2` Q heads and computes attention for those 2 Q heads against the full 2 KV heads locally. No CCL is required during SDPA.

**Option B — All-reduce output, shard differently:**

Not useful here; with only 2 KV heads there is no beneficial sharding of K/V.

**Recommendation: Option A** — replicate KV, shard Q. This matches the existing `TTNNQwen3FullAttention` implementation, which uses column sharding for the Q projection (2 Q heads per device) and replicates the KV cache across devices. TTNN's `scaled_dot_product_attention_decode` operates on the local KV replica.

---

## 2. Per-Device KV Cache Memory

**KV cache bytes per layer** (combined K and V):

```
K bytes per layer = n_kv_h × T × d_h × 2 bytes
                  = 2 × T × 256 × 2
                  = 1,024 × T bytes

V bytes per layer = same = 1,024 × T bytes

Combined KV per layer = 2,048 × T bytes
```

This is replicated on all 8 devices (each device holds the full 2-head KV cache):

```
At T = 8,192:
  Per layer = 2,048 × 8,192 = 16,777,216 bytes = 16 MiB
  10 attention layers = 167,772,160 bytes ≈ 160 MiB per device

At T = 262,144:
  Per layer = 2,048 × 262,144 = 536,870,912 bytes = 512 MiB
  10 attention layers = 5,368,709,120 bytes = 5,120 MiB ≈ 5.12 GiB per device
```

---

## 3. Memory Budget at Full Context

The available DRAM per device after model weights is approximately 3.25 GB (see `t3k_mesh_topology.md`). Comparing to the KV cache requirement:

| Context length T | KV cache (10 layers) | Budget (3.25 GB) | Headroom |
|---|---|---|---|
| 8,192 | 160 MiB ≈ 0.16 GB | 3.25 GB | 3.09 GB |
| 65,536 | 1,280 MiB ≈ 1.25 GB | 3.25 GB | 2.00 GB |
| 131,072 | 2,560 MiB ≈ 2.50 GB | 3.25 GB | 0.75 GB |
| 262,144 | 5,120 MiB ≈ 5.12 GB | 3.25 GB | **−1.87 GB** |

The KV cache for Gated Attention exceeds the per-device DRAM budget at T = 262,144. This is the primary memory bottleneck of the hybrid model at full context, not the Gated Delta Net recurrent state (which is 3.75 MiB total across all layers — negligible).

At T = 262,144, batch size must be reduced below B = 1 in the KV cache — effectively, the model cannot run a single full-context sequence on a T3K without offloading or model-level changes (e.g., reducing `n_kv_h`, reducing `d_h`, or reducing the number of Gated Attention layers).

---

## 4. Sharding the Q Projection (Gated Attention)

The Q+gate projection maps `[B, T, H] → [B, T, n_q_h × d_h × 2]` = `[B, T, 8192]` (Q and gate interleaved). Under column sharding over 8 devices, each device computes `[B, T, 1024]` — 2 Q heads + 2 gate vectors of size 256 each.

The KV projection maps `[B, T, H] → [B, T, n_kv_h × d_h × 2]` = `[B, T, 1024]`. With only 1,024 output elements total, sharding KV projection across 8 devices is impractical (128 elements per device). KV projection uses replicated weights instead.

### Communication per Gated Attention Decode Layer

| Step | CCL? | Notes |
|---|---|---|
| Q+gate projection (col-sharded) | No | Local matmul |
| KV projection (replicated weights) | No | Local matmul; each device produces full KV |
| KV cache write (paged) | No | Local write to replicated cache |
| GQA expand (2→16 Q heads) | No | Local repeat-interleave; 2 KV heads repeat 8× |
| SDPA (2 Q heads per device) | No | Local attention against full 2 KV heads |
| Output projection (row-parallel) | No | Local partial matmul |
| All-gather output | **Yes** | Assemble `[B, T, H]` from 8 × `[B, T, 256]` |

```
All-gather payload (per layer, B=1, T=1):
  8 × [1, 1, 256] × 2 bytes = 4,096 bytes = 4 KiB
  Time at 25 GB/s: 4,096 / 25 × 10^9 ≈ 0.16 µs per layer
```

CCL cost is identical to the Gated Delta Net layer: 4 KiB all-gather, sub-microsecond.

---

## 5. Summary

| Tensor | Sharding | Per-device size (B=1, T=8192) | CCL per layer |
|---|---|---|---|
| GDN recurrent state (30 layers) | Head-parallel (4/8 heads) | 3.75 MiB total | 4 KiB all-gather |
| GDN conv state (30 layers) | Same as state | 0.24 MiB total | — |
| GA KV cache (10 layers) | Replicated (n_kv_h=2) | 160 MiB | 4 KiB all-gather |
| GA KV cache (10 layers) | Replicated (T=262,144) | **5,120 MiB** | 4 KiB all-gather |

The recurrent state is not a memory bottleneck. The Gated Attention KV cache is.

---

**Previous:** [`alternative_sharding_strategies.md`](./alternative_sharding_strategies.md)

**Next:** [Chapter 7 — Kernel Gaps and Development Roadmap](../ch7_kernel_gaps_and_roadmap/index.md)
