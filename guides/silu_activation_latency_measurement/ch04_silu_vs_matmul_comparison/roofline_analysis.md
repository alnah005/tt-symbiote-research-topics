# Roofline Analysis: SiLU and Matmul on Wormhole B0

The roofline model expresses achievable performance as a function of arithmetic intensity (FLOP/byte). An operation is memory-bound when its arithmetic intensity is below the ridge point; it is compute-bound above the ridge point.

---

## 1. Wormhole B0 Hardware Ceilings

| Parameter | Value | Source |
|---|---|---|
| BF16 FPU peak throughput | 131 TFLOP/s | Hardware specification |
| DRAM bandwidth (practical peak) | ~300 GB/s | 6 controllers × 12 GDDR6 banks |
| Ridge point | 131e12 / 300e9 ≈ **437 FLOP/byte** | Ratio of compute ceiling to memory ceiling |

See Chapter 2, `cycles_vs_matmul.md` for the derivation of these values.

---

## 2. Arithmetic Intensity of SiLU

`ttnn.silu` computes `x * sigmoid(x)` element-wise on BF16 tensors in `TILE_LAYOUT`.

SiLU arithmetic intensity is 0.5 FLOP/byte (2 FLOP per element / 4 bytes per BF16 read-write pair), placing it deeply in the memory-bound regime at all token counts.

For the full derivation, see Chapter 2, `cycles_vs_matmul.md`.

> Note: In-place execution or L1-sharded tensors can substitute L1 bandwidth (~several TB/s) for DRAM bandwidth, but the arithmetic intensity remains the same — the operation is still bandwidth-limited, just against a higher ceiling.

---

## 3. Arithmetic Intensity of Batched Matmul

For the gate_proj matmul in a typical MoE expert layer, the matrix dimensions are:

- Activation: `[num_tokens, hidden_dim]`  (e.g., `[32, 4096]`)
- Weight: `[hidden_dim, ffn_dim]`  (e.g., `[4096, 8192]` for a down-projection after gating)

Arithmetic intensity for a matmul of shape `[M, K] × [K, N]`:

```
FLOPs = 2 * M * K * N
Bytes = 2 * (M*K + K*N + M*N)   # BF16: 2 bytes per element, read A, read B, write C
AI   = FLOPs / Bytes
     = 2*M*K*N / (2*(M*K + K*N + M*N))
     = M*K*N / (M*K + K*N + M*N)
```

Representative values for decode vs. prefill:

| Shape (M×K×N) | Regime | Arithmetic Intensity |
|---|---|---|
| 1 × 4096 × 8192 | Decode, 1 token | ≈ 1 FLOP/byte — memory-bound |
| 8 × 4096 × 8192 | Decode, 8 tokens | ≈ 8 FLOP/byte — memory-bound |
| 32 × 4096 × 8192 | Decode, 32 tokens | ≈ 31 FLOP/byte — memory-bound |
| 128 × 4096 × 8192 | Prefill, 128 tokens | ≈ 121 FLOP/byte — approaching ridge |
| 512 × 4096 × 8192 | Prefill, 512 tokens | ≈ 431 FLOP/byte — near ridge point (slightly memory-bound) |
| 1024 × 4096 × 8192 | Prefill, 1024 tokens | ≈ 745 FLOP/byte — compute-bound |

Weight-only bytes dominate at small M: when `M << K`, the `K×N` weight term dominates and AI ≈ M. This is the standard roofline result for GEMM with a small batch.

---

## 4. Expected Performance Regime Summary

| Operation | Arithmetic Intensity | Regime at all practical token counts |
|---|---|---|
| `ttnn.silu` | ~0.5 FLOP/byte | Always memory-bound |
| gate_proj matmul, decode (M ≤ 32) | 1–31 FLOP/byte | Memory-bound |
| gate_proj matmul, prefill (M = 128) | ~121 FLOP/byte | Memory-bound, approaching ridge |
| gate_proj matmul, prefill (M ≥ 512) | 431–745+ FLOP/byte | Near or above ridge point — memory-bound (M=512, just below ridge); compute-bound (M=1024, well above ridge) |

The critical insight: **both SiLU and decode-batch matmul are memory-bound**. At decode, neither operation saturates the FPU; both are bottlenecked by DRAM bandwidth. At prefill, matmul crosses into compute-bound territory while SiLU remains memory-bound.

---

## 5. ASCII Roofline Sketch

```
 Performance
 (TFLOP/s)
   131 |------------------------------------+===========  FPU ceiling (131 TFLOP/s)
       |                                  /
       |                                 /
       |                                /
       |                               /
       |        Memory-bound          / Compute-bound
       |       (slope = BW)          /
       |                            /  * matmul, M=1024 (compute-bound)
       |                           /
       |                          * matmul, M=128
       |                         /
       |               * matmul, M=32
       |              /
       | * SiLU      * matmul, M=8
       |/
     --+----+------+-------+----------+---+-----------> Arithmetic
       0.5  1      8      31         121  437         Intensity (FLOP/byte)
              ^                           ^
              SiLU                    Ridge point
```

SiLU sits at the far left: 0.5 FLOP/byte, well below the ridge. Decode-batch matmul (M ≤ 32) also sits far left. Prefill matmul (M ≥ 512) crosses or approaches the ridge point.

---

## Next Steps

Proceed to [`latency_ratio_by_shape.md`](latency_ratio_by_shape.md) to see how these roofline positions translate into measured latency ratios across the shape sweep.
