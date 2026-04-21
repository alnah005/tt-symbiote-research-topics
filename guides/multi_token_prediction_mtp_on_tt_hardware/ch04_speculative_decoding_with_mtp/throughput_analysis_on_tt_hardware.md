# Throughput Analysis on TT Hardware

## Bandwidth-Bound Cost Model

At batch size 1, decode throughput on Wormhole-based hardware (P150 single-chip, T3K 4-chip) is dominated by DRAM bandwidth. The key property:

> Loading all model weights from DRAM costs the same whether you process 1 token or a short sequence (2–4 tokens). Arithmetic intensity is too low to saturate the compute units.

Define the basic cost unit:

```
C_decode = time to execute one single-token decode step
         ≈ model_size_bytes / DRAM_bandwidth
```

For Qwen3.6-35B-A3B at BF16:
- ~35B parameters × 2 bytes ≈ 70 GiB of weights loaded per decode step (approximate; exact depends on activated experts / architecture routing)
- P150 DRAM bandwidth: ~288 GB/s → C_decode ≈ 243 ms per step (order-of-magnitude estimate, excluding KV cache and activation traffic)

### Cost of Key Passes

| Pass | Tokens | Cost (BW-bound) |
|---|---|---|
| Primary decode (step 1 in algorithm) | 1 | `C_decode` |
| MTP head (step 3) | 1 | `≈ 0` (tiny head, ~160M params << 35B) |
| Verification pass (step 5) | 2 | `≈ C_decode` (same weight loading) |

The MTP head contributes negligible bandwidth: 304.6 MiB versus ~70 GiB for the backbone. The verification pass, however, traverses the full 94-layer backbone over a 2-token sequence — indistinguishable from C_decode in the bandwidth-bound regime.

## Speedup Derivation for N=1

One speculative decode cycle consists of:

1. Primary backbone pass: cost `C_decode`, produces 1 confirmed token (x_{t+1}) and 1 draft token (x̂_{t+2}).
2. Verification pass: cost `≈ C_decode`, confirms or rejects x̂_{t+2}.

Total cycle cost: `2 × C_decode`

Expected tokens per cycle: `1 + α` (from speculative decoding formula with K=1)

```
Speedup = E[tokens / cycle] / cycle_cost
        = (1 + α) / (2 × C_decode)
        vs. baseline 1 token / C_decode

Relative speedup = (1 + α) / 2
```

Since α < 1 always (no draft model is perfect), `(1 + α) / 2 < 1` for all achievable acceptance rates.

**MTP speculative decoding with N=1 is slower than standard decode at batch size 1 on bandwidth-bound TT hardware.**

## Breakeven Table (Batch Size 1)

| α | E[tokens/cycle] | Cost (passes) | Speedup (BW-bound) |
|---|---|---|---|
| 0.5 | 1.5 | 2 | 0.75 |
| 0.7 | 1.7 | 2 | 0.85 |
| 0.8 | 1.8 | 2 | 0.90 |
| 0.9 | 1.9 | 2 | 0.95 |
| 1.0 | 2.0 | 2 | 1.00 (theoretical limit) |

Even at α = 0.9 (very high acceptance), the overhead of the verification pass reduces throughput by 5%. Breakeven requires α = 1.0 — a perfect drafter.

## Breakeven Condition (General)

For speedup > 1, the verification pass must be cheaper than the primary decode:

```
(1 + α) / (1 + C_verify / C_decode) > 1
→ C_verify / C_decode < α
→ C_verify < α × C_decode
```

At batch size 1 on bandwidth-bound hardware, `C_verify ≈ C_decode`, so the condition requires `1 < α`, which is impossible.

## Where MTP Spec Decoding Is Beneficial

### Larger Batch Sizes (B > 1)

At batch B, the verification pass processes B × 2 tokens but still loads model weights once per layer. The effective cost per token decreases because compute utilization improves:

```
Speedup(B) = (1 + α) / (1 + C_verify(B) / C_decode(B))
```

As B increases, the ratio `C_verify(B) / C_decode(B)` approaches a constant less than 1 once the compute units are saturated (compute-bound regime). At large B:

```
Speedup → (1 + α) / (1 + r)   where r < 1
```

For α = 0.8 and r = 0.5: speedup ≈ 1.8 / 1.5 = 1.2 — a 20% improvement.

### Higher N (Future Models)

If a future checkpoint has `mtp_num_hidden_layers = 3`, then K=3 draft tokens are available at near-zero cost. With α = 0.8:

```
E[tokens / cycle] = (1 - 0.8^4) / (1 - 0.8) = (1 - 0.4096) / 0.2 ≈ 2.95
Speedup (BW-bound, 2 passes) = 2.95 / 2 ≈ 1.48
```

K=3 with α = 0.8 would yield ~48% speedup even at batch size 1, because the verification pass cost is now amortized over nearly 3 accepted tokens on average.

## Summary Table

| Scenario | K | α | Batch | Speedup |
|---|---|---|---|---|
| Qwen3.6, batch 1 | 1 | 0.8 | 1 | 0.90 (slower) |
| Qwen3.6, batch 32 | 1 | 0.8 | 32 | >1.0 (depends on compute regime) |
| Hypothetical N=3, batch 1 | 3 | 0.8 | 1 | ~1.48 |

## Key Finding

> At batch size 1 on memory-bandwidth-bound TT hardware, MTP speculative decoding with N=1 reduces throughput by (1-α)/2 relative to standard decode. The verification pass costs a full backbone traversal and the single draft token does not amortize it. The algorithm is architecturally correct and worth implementing — the throughput benefit materializes at larger batch sizes or with higher N configurations.

---
**Next:** [acceptance_rate_estimation.md](acceptance_rate_estimation.md)
