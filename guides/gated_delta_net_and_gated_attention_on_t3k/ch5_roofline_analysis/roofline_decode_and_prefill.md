# Roofline Analysis — Decode and Prefill on a Single Wormhole Chip

This section applies the roofline model to Gated Delta Net for two operating modes: autoregressive **decode** (B=1, T=1) and **prefill** (B=1, T=8192). All arithmetic reflects the recurrence derived in Chapter 4. Analysis is per single Wormhole chip (455 FLOP/byte ridge point).

Symbols used throughout:
- `d_k = 128`, `d_v = 128`, `d_h = 256`
- `H_v = 32` (value/output heads per layer), `H_k = 16` (key heads)
- `C = 64` (chunk size for chunkwise parallelism)
- `n_kv_h = 2` (Gated Attention KV heads, for comparison)

---

## 1. Decode Step (B=1, T=1)

At decode time, each new token triggers one step of the recurrence for every head in every layer. The analysis below is for **one head** and then scaled up.

### 1.1 FLOPs per Head

The six operations from the Chapter 4 recurrence are:

| Operation | Expression | FLOPs |
|---|---|---|
| `g · S` — scalar-matrix decay | `g` × `[d_k × d_v]` matrix | d_k × d_v = 128 × 128 = **16,384** |
| `S^T k̃` — state retrieval | matrix-vector: 2 × d_k × d_v | 2 × 128 × 128 = **32,768** |
| `β · (v − retrieval)` — write error | scale + subtract over d_v | ≈ 2 × d_v = 2 × 128 = **256** |
| `k̃ ⊗ error^T` — outer product write | rank-1 outer product: d_k × d_v | 128 × 128 = **16,384** |
| `S_new = S_decayed + write` — state update | elementwise add: d_k × d_v | 128 × 128 = **16,384** |
| `o = S_new^T q̃` — output retrieval | matrix-vector: 2 × d_k × d_v | 2 × 128 × 128 = **32,768** |

```
Total per head = 16,384 + 32,768 + 256 + 16,384 + 16,384 + 32,768
              = 114,944 FLOP
              ≈ 115K FLOP
```

Scaling up:

```
Per layer (H_v = 32 heads):  32 × 114,944  =  3,678,208 FLOP  ≈  3.68 MFLOP
Per token (30 DeltaNet layers): 30 × 3.68M =  110,346,240 FLOP ≈  110.4 MFLOP
```

### 1.2 Bytes per Head

The state matrix `S` shaped `[d_k, d_v]` in BF16 (2 bytes/element) must be read and written every step. The input vectors `k̃`, `q̃`, `v` are negligible by comparison.

| Data | Expression | Bytes |
|---|---|---|
| S read | d_k × d_v × 2 bytes | 128 × 128 × 2 = 32,768 |
| S write | d_k × d_v × 2 bytes | 128 × 128 × 2 = 32,768 |
| k̃, q̃, v (combined) | (d_k + d_k + d_v) × 2 bytes | (128 + 128 + 128) × 2 = 768 |

```
Total per head = 32,768 + 32,768 + 768
              = 66,304 bytes
              ≈ 66 KB
```

Scaling up:

```
Per layer (32 heads):      32 × 66,304  =  2,121,728 bytes  ≈  2.02 MB
Per token (30 layers):     30 × 2.02 MB =  60.6 MB
```

### 1.3 Arithmetic Intensity and Roofline Placement

```
Arithmetic intensity (per layer) = FLOPs / bytes
                                 = 3,678,208 / 2,121,728
                                 = 3.68 × 10^6 / 2.12 × 10^6
                                 ≈ 1.74 FLOP/byte
```

**Ridge point: 455 FLOP/byte. Measured intensity: 1.74 FLOP/byte.**

The decode step sits more than **262× below the ridge point**. It is **heavily memory-bandwidth-bound**: the chip's FPUs are almost entirely idle while waiting for DRAM to supply and absorb the state matrix.

### 1.4 Decode Time Estimate (Single Chip)

```
State bandwidth per layer   = 2,121,728 bytes
DRAM bandwidth              = 288 GB/s

Time per layer (state only) = 2,121,728 / 288 × 10^9
                            = 7.36 µs

Total for 30 DeltaNet layers = 30 × 7.36 µs = 220.8 µs ≈ 221 µs per decode step
```

This is a lower-bound estimate (state I/O only); kernel overhead, TTNN dispatch, and vector operations add modestly on top.

---

## 2. Comparison: Gated Attention Decode at T=262,144

For the Gated Attention layers, the decode bottleneck is reading the full KV cache. At a context length of T=262,144:

```
KV bytes per layer = n_kv_h × T × d_h × 2 (K and V) × 2 bytes/element
                   = 2 × 262,144 × 256 × 2 × 2
                   = 536,870,912 bytes
                   = 512 MB

Time per layer at 288 GB/s = 536,870,912 / 288 × 10^9
                           ≈ 1.86 ms  = 1,864 µs
```

Comparison:

```
DeltaNet decode per layer   ≈  7.36 µs
Gated Attention per layer   ≈ 1,864 µs  (at T = 262,144)

Speedup = 1,864 / 7.36 ≈ 253×
```

**DeltaNet is approximately 253× faster per layer than Gated Attention at T=262,144**, purely because its memory footprint per step is fixed at `[d_k, d_v]` (32 KB) regardless of context length, while the KV cache grows linearly with T.

---

## 3. Prefill Pass (B=1, T=8192)

During prefill, Gated Delta Net uses chunkwise parallelism with chunk size C=64. The analysis below is per head, per chunk, then accumulated.

### 3.1 FLOPs per Head per Chunk

| Operation | Expression | FLOPs |
|---|---|---|
| Inner-chunk QK matmul | 2 × C × C × d_k | 2 × 64 × 64 × 128 = **1,048,576** |
| Inner-chunk AV matmul | 2 × C × C × d_v | 2 × 64 × 64 × 128 = **1,048,576** |
| State-update matmuls (U, W) | 2 × C × d_k × d_v | 2 × 64 × 128 × 128 = **2,097,152** |

```
Total per head per chunk = 1,048,576 + 1,048,576 + 2,097,152
                        = 4,194,304 FLOP
                        ≈ 4.2 MFLOP
```

Scaling up:

```
Number of chunks (T=8192, C=64) = 8192 / 64 = 128 chunks

Per head (128 chunks):        128 × 4,194,304  = 536,870,912 FLOP  ≈ 536.9 MFLOP
Per layer (H_v = 32 heads):  32 × 536,870,912 = 17,179,869,184 FLOP ≈ 17.2 GFLOP
```

### 3.2 Bytes per Head per Chunk

| Data | Expression | Bytes |
|---|---|---|
| K loaded per chunk | C × d_k × 2 bytes | 64 × 128 × 2 = 16,384 |
| V loaded per chunk | C × d_v × 2 bytes | 64 × 128 × 2 = 16,384 |
| Q loaded per chunk | C × d_k × 2 bytes | 64 × 128 × 2 = 16,384 |
| Output O written per chunk | C × d_v × 2 bytes | 64 × 128 × 2 = 16,384 |
| State S read per chunk | d_k × d_v × 2 bytes | 128 × 128 × 2 = 32,768 |
| State S write per chunk | d_k × d_v × 2 bytes | 128 × 128 × 2 = 32,768 |

```
Total per head per chunk = 16,384 + 16,384 + 16,384 + 16,384 + 32,768 + 32,768
                        = 131,072 bytes
```

Scaling up:

```
Per head (128 chunks):       128 × 131,072  = 16,777,216 bytes  ≈ 16 MB
Per layer (32 heads):         32 × 16 MB    = 512 MB
```

### 3.3 Arithmetic Intensity and Roofline Placement

```
Arithmetic intensity (per layer, T=8192) = FLOPs / bytes
                                         = 17,179,869,184 / 536,870,912
                                         = 32.0 FLOP/byte (exact)
```

**Ridge point: 455 FLOP/byte. Measured intensity: 32.0 FLOP/byte.**

The prefill pass is still **memory-bandwidth-bound**, but considerably less severely than decode (32.0 vs 1.74 FLOP/byte). The inner-chunk matmuls (`[C×C×d_k]` and `[C×C×d_v]`) provide meaningful reuse that lifts intensity well above the decode floor — but the state read/write every chunk keeps intensity far from the ridge point.

---

## 4. Summary and Design Implications

| Mode | Arithmetic Intensity | Regime | Bottleneck |
|---|---|---|---|
| Decode (B=1, T=1) | 1.74 FLOP/byte | Memory-bound (262× below ridge) | State matrix DRAM I/O |
| Prefill (B=1, T=8192, C=64) | 32.0 FLOP/byte | Memory-bound (14.2× below ridge) | State matrix + KQV DRAM I/O |
| Ridge point | 455 FLOP/byte | Compute-bound threshold | — |

**Both operating modes are DRAM-bandwidth-bound on a single Wormhole chip.** The dominant cost in both cases is reading and writing the state matrix `S ∈ [d_k, d_v]`, not the floating-point arithmetic itself.

Implications for kernel design:

1. **Fused kernels are essential.** A kernel that materializes intermediate tensors (`S_decayed`, `retrieval`, `error`) to DRAM between steps pays a bandwidth penalty with no compute benefit. The state update and output retrieval should be fused into a single kernel that keeps intermediates in registers or L1.

2. **L1-resident state eliminates the dominant cost.** If `S` can remain in L1 SRAM across an entire sequence chunk, the DRAM traffic drops to loading `k̃`, `q̃`, `v` only — reducing per-chunk bytes from ~131 KB to ~65 KB and roughly doubling effective arithmetic intensity.

3. **Increasing chunk size C helps prefill.** Larger C amortizes the per-chunk state I/O over more inner-chunk FLOP, raising arithmetic intensity toward (but not past) the ridge point. The practical ceiling is L1 capacity.

### 4.1 L1 Feasibility

Can the state matrix for one or more heads fit in a single Tensix core's 1.5 MB L1?

```
Per-head state size = d_k × d_v × 2 bytes (BF16)
                   = 128 × 128 × 2
                   = 32,768 bytes
                   = 32 KB

Per-layer state (32 heads) = 32 × 32 KB = 1,024 KB = 1 MB

L1 per core = 1,536 KB (1.5 MB)

Heads fitting in one core's L1 = floor(1,536 KB / 32 KB) = 48 heads

Full 32-head layer state (1 MB) fits in one core's 1.5 MB L1
with 512 KB (0.5 MB) spare for activations, stack, and kernel code.
```

The full per-layer state for all 32 heads fits comfortably in a single Tensix core's L1, with room to spare. This means a well-designed fused kernel can keep the state resident in L1 across an entire chunk, eliminating the dominant DRAM traffic and making the decode step approach its theoretical bandwidth ceiling rather than wasting cycles on redundant round-trips.

---

**Next:** [Chapter 6 — T3K Sharding Strategy for Gated Delta Net State](../ch6_t3k_sharding/index.md)
