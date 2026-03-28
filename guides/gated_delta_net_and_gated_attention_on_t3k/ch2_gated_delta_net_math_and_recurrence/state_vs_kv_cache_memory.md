# State vs. KV Cache Memory: Quantitative Comparison

This section computes exact memory footprints for the Gated Delta Net recurrent state and the Gated Attention KV cache, compares them across sequence lengths, and evaluates the T3K DRAM and L1 constraints.

All calculations use BF16 (2 bytes per element) and the Qwen3.5-35B-A3B configuration.

---

## 1. Gated Delta Net Recurrent State

### 1.1 Recurrent state (S matrix)

```
Shape:    [B, H_v, d_k, d_v]  =  [B, 32, 128, 128]

Elements per batch element:
  H_v × d_k × d_v  =  32 × 128 × 128  =  524,288

BF16 bytes per batch element:
  524,288 × 2  =  1,048,576 bytes  =  1,024 KB  =  1.000 MB

Total:  B × 1,048,576 bytes  ≈  B × 1 MB per Gated Delta Net layer
```

This memory is **independent of sequence length T**. Regardless of whether the model has processed 1 token or 256K tokens, the recurrent state remains at exactly `[B, 32, 128, 128]`.

### 1.2 Conv1d state

The causal conv1d maintains a sliding buffer of the last `(kernel_size − 1) = 3` input frames (plus the current frame = 4 total) over all channels:

```
Shape:    [B, key_dim×2 + value_dim, conv_kernel_size]  =  [B, 8192, 4]

Elements per batch element:
  8192 × 4  =  32,768

BF16 bytes per batch element:
  32,768 × 2  =  65,536 bytes  =  64 KB  per Gated Delta Net layer
```

### 1.3 Total per Gated Delta Net layer

```
Recurrent state:   B × 1,048,576 bytes  ≈  B × 1,024 KB
Conv state:        B ×    65,536 bytes  ≈  B ×    64 KB
                   ─────────────────────────────────────
Total:             B × 1,114,112 bytes  ≈  B × 1,088 KB  ≈  B × 1.0625 MB
```

**B × 1.0625 MB per Gated Delta Net layer.**

---

## 2. Gated Attention KV Cache

The 10 Gated Attention layers in Qwen3.5-35B-A3B use standard GQA with paged KV cache. Per layer:

```
K cache shape:  [B, n_kv_h, T, d_h]  =  [B, 2, T, 256]
V cache shape:  [B, n_kv_h, T, d_h]  =  [B, 2, T, 256]

Elements per cache (K or V), per batch element:
  n_kv_h × T × d_h  =  2 × T × 256  =  512 × T

BF16 bytes per cache:
  512 × T × 2  =  1,024 × T bytes

Combined K+V bytes per batch element:
  2 × 1,024 × T  =  2,048 × T bytes  ≈  2 KB × T
```

**B × 2,048 × T bytes per Gated Attention layer** (K+V combined). For memory comparison we use this combined KV figure.

---

## 3. Crossover Analysis

At what sequence length T does the (fixed) DeltaNet state cost equal the (growing) Gated Attention KV cache?

```
DeltaNet state per layer:       B × 1,114,112 bytes
Gated Attention KV per layer:   B × 2,048 × T bytes

Crossover:
  1,114,112  =  2,048 × T
  T  =  1,114,112 / 2,048  =  544 tokens
```

Using the plan's rounded DeltaNet figure (B × 1,048,576 for state only, ignoring conv):

```
  T  =  1,048,576 / 1,024  =  1,024 tokens   (K-only crossover)
  T  =  1,048,576 / 2,048  =  512 tokens      (combined KV crossover)
```

Including conv state at ~64 KB and using combined KV:

```
  T  =  1,114,112 / 2,048  ≈  544 tokens
```

Summary: **for any sequence longer than ~512–544 tokens, the Gated Attention KV cache is more expensive per layer than the DeltaNet recurrent state.** The exact crossover depends on whether you count K-only or K+V and whether you include the conv state (full combined KV crossover: ~512 tokens; including conv state: ~544 tokens). In all cases the crossover is well under 1,500 tokens.

---

## 4. Memory at T = 262,144 (2^18, "256K" Max Context)

### 4.1 Per Gated Attention layer

Using T = 262,144 (2^18, the standard binary meaning of "256K" in hardware/ML contexts):

```
KV cache per layer (step-by-step):
  n_kv_h = 2,  T = 262,144,  d_h = 256,  BF16 = 2 bytes

  Combined K+V bytes per batch element:  2 (K and V) × n_kv_h × T × d_h × 2
    =  2 × 2 × 262,144 × 256 × 2
    =  536,870,912 bytes
    =  512 MB per Gated Attention layer
```

### 4.2 Per Gated Delta Net layer (state)

```
  B × 1,114,112 bytes  ≈  B × 1.0625 MB
```

### 4.3 Ratio

```
KV cache / DeltaNet state  =  512 MB / 1.0625 MB  ≈  482×
```

At T = 262,144, the DeltaNet recurrent state is approximately **482 times smaller** per layer than the KV cache of one Gated Attention layer.

---

## 5. Aggregate Model-Level Memory

### 5.1 All 30 DeltaNet layers

```
Per layer:  B × 1,114,112 bytes  ≈  B × 1.063 MB
30 layers:  B × 30 × 1,114,112  =  B × 33,423,360 bytes
           ≈  B × 31.875 MB  ≈  B × 31.9 MB
```

At B = 1: ~31.9 MB — negligible relative to the 70 GB model weights.
At B = 32: ~1.02 GB — still manageable within T3K DRAM.

### 5.2 All 10 Gated Attention layers at T = 262,144

```
Per layer:  B × 512 MB
10 layers:  B × 5,120 MB  ≈  B × 5 GB

At B = 1:  5 GB
At B = 4:  20 GB  (significant fraction of T3K DRAM)
At B = 8:  40 GB  (exceeds the ~26 GB remaining after weights)
```

### 5.3 T3K DRAM budget

```
T3K total DRAM:  8 devices × 12 GB  =  96 GB
BF16 model weights (Qwen3.5-35B-A3B):  ~70 GB
DRAM remaining for activations + state:  ~26 GB

DeltaNet state at B=1:   ~32 MB   —  fits with large margin
DeltaNet state at B=32:  ~1.02 GB —  fits
Gated Attention KV at B=1, T=262,144:  ~5 GB   —  fits (within ~26 GB headroom)
Gated Attention KV at B=4, T=262,144:  ~20 GB  —  tight; leaves ~6 GB for activations
Gated Attention KV at B=8, T=262,144:  ~40 GB  —  exceeds headroom; not feasible
```

The DeltaNet recurrent state poses **no DRAM pressure** at any practical batch size. The Gated Attention KV cache is the dominant memory consumer at long contexts. At T = 262,144, the combined 10-layer KV footprint is ~5 GB per batch element, limiting effective batch size to approximately B = 4–5 before saturating the ~26 GB DRAM headroom on T3K.

---

## 6. L1 SRAM Constraints

Each Tensix core on Wormhole has **1.5 MB of L1 SRAM**.

### 6.1 Single-layer DeltaNet state at B = 1

```
Full per-layer state:  1 × 32 × 128 × 128 × 2  =  1,048,576 bytes  =  1,024 KB  ≈  1.0 MB
```

1.0 MB fits within a single core's 1.5 MB L1 — but barely, leaving only ~512 KB for activations, weights, and other buffers. In practice:

- **Distributed across cores**: the 32 heads can be split across multiple cores (e.g., 4 heads × 32 KB = 128 KB per core if 8 cores share the layer), keeping each core's state fragment well under 1.5 MB.
- **Streaming from DRAM**: for a decode step, the state-matrix-vector multiply reads the full `[d_k, d_v]` state and streams it through the compute units; only one head's state (~32 KB) needs to be live in L1 at a time if heads are processed sequentially.

At B = 2 or higher, the per-layer state exceeds 2 MB and cannot fit in any single core's L1 regardless of layout. State must be distributed across cores or streamed from DRAM.

### 6.2 Implications

The L1 constraint confirms that:

1. The decode step is **streaming-from-DRAM** dominated, consistent with the arithmetic intensity analysis in Section 3 of `parallelism_and_scan.md` (~1.25 FLOPs/byte).
2. Efficient decode requires kernel designs that maximize DRAM bandwidth utilization during the `S^T k̃` and `k̃ ⊗ error` operations, rather than trying to keep the full state resident in L1.
3. Head-parallel sharding (one T3K device per subset of heads) is a natural strategy: each device holds a shard of the state in its DRAM and processes the corresponding heads locally. This is analyzed in Chapter 6.

---

## 7. Summary Table

| Memory component | Shape | BF16 size (B=1) | Grows with T? |
|-----------------|-------|-----------------|---------------|
| DeltaNet recurrent state (per layer) | `[1, 32, 128, 128]` | 1.0 MB | No |
| DeltaNet conv state (per layer) | `[1, 8192, 4]` | 64 KB | No |
| DeltaNet total per layer | — | ~1.063 MB | No |
| DeltaNet total (30 layers) | — | ~31.9 MB | No |
| Gated Attention KV (per layer, T=262,144) | `[1, 2, 262144, 256] × 2 (K+V)` | ~512 MB | Yes, linear |
| Gated Attention KV (10 layers, T=262,144) | — | ~5 GB | Yes, linear |

For a forward reference on how the recurrent state is sharded across the 8 T3K devices and how the CCL cost is amortized, see **Chapter 6 — Tensor Parallelism and State Sharding on T3K**.

---

**Next:** [Chapter 3 — Gated Attention: Mechanism and Tensor Shapes](../ch3_gated_attention_mechanism/index.md)
