# Latency Ratio by Shape: SiLU vs. Matmul

This file presents the latency ratio table for `ttnn.silu`, gate_proj matmul, up_proj matmul, and element-wise multiply across the shape sweep defined in Chapter 3. All latencies are median `DEVICE KERNEL DURATION [ns]` values converted to microseconds.

---

## 1. Table Layout

Each row in the table below corresponds to one `(num_tokens, hidden_dim)` configuration. Columns:

| Column | Meaning |
|---|---|
| `gate_proj` | Matmul `[num_tokens, hidden_dim] × [hidden_dim, ffn_dim]` |
| `ttnn.silu` | Element-wise SiLU on gate_proj output `[num_tokens, ffn_dim]` |
| `up_proj` | Matmul `[num_tokens, hidden_dim] × [hidden_dim, ffn_dim]` |
| `elem_mul` | Element-wise multiply of silu output and up_proj output |
| `SiLU / gate_proj` | Ratio: SiLU latency as a fraction of gate_proj matmul latency |

`ffn_dim` is taken as `2 × hidden_dim` (standard MoE FFN expansion). For `hidden_dim=4096`, `ffn_dim=8192`.

---

## 2. Expected Latency Table (Wormhole B0, BF16, DRAM-backed)

All values in microseconds (µs). Values marked `[expected]` are model-derived; values marked `[measured]` should be filled in from your Chapter 3 profiler CSV.

### hidden_dim = 4096, ffn_dim = 8192

| num_tokens | gate_proj [expected] | ttnn.silu [expected] | up_proj [expected] | elem_mul [expected] | SiLU / gate_proj |
|---|---|---|---|---|---|
| 1 | 40–80 µs | 10–20 µs | 40–80 µs | 8–15 µs | **20–35%** |
| 4 | 50–90 µs | 12–22 µs | 50–90 µs | 9–16 µs | **20–30%** |
| 8 | 60–110 µs | 14–28 µs | 60–110 µs | 10–18 µs | **18–28%** |
| 32 | 80–160 µs | 18–38 µs | 80–160 µs | 12–22 µs | **15–25%** |
| 64 | 130–280 µs | 22–45 µs | 130–280 µs | 15–28 µs | **10–18%** |
| 128 | 400–900 µs | 28–55 µs | 400–900 µs | 18–35 µs | **4–8%** |
| 256 | 1200–2400 µs | 40–80 µs | 1200–2400 µs | 25–50 µs | **2–4%** |

### hidden_dim = 2048, ffn_dim = 4096

| num_tokens | gate_proj [expected] | ttnn.silu [expected] | up_proj [expected] | SiLU / gate_proj |
|---|---|---|---|---|
| 1 | 20–45 µs | 6–12 µs | 20–45 µs | **20–35%** |
| 8 | 30–65 µs | 8–16 µs | 30–65 µs | **18–28%** |
| 32 | 45–95 µs | 10–22 µs | 45–95 µs | **15–25%** |
| 128 | 200–500 µs | 14–30 µs | 200–500 µs | **4–8%** |

### hidden_dim = 8192, ffn_dim = 16384

| num_tokens | gate_proj [expected] | ttnn.silu [expected] | up_proj [expected] | SiLU / gate_proj |
|---|---|---|---|---|
| 1 | 80–160 µs | 20–40 µs | 80–160 µs | **20–30%** |
| 8 | 110–200 µs | 25–50 µs | 110–200 µs | **18–28%** |
| 32 | 150–280 µs | 30–65 µs | 150–280 µs | **15–25%** |
| 128 | 700–1600 µs | 45–90 µs | 700–1600 µs | **4–7%** |

---

## 3. Key Finding: Decode Regime (num_tokens ≤ 8)

At 1–8 tokens, gate_proj matmul arithmetic intensity is 1–8 FLOP/byte — far below the ridge point of 437 FLOP/byte. The matmul is memory-bound; it reads the full weight matrix from DRAM and discards most of it after one use.

SiLU is also memory-bound at 0.5 FLOP/byte. Both operations compete for the same DRAM bandwidth. The SiLU activation tensor (`[num_tokens, ffn_dim]`) is small but the access is still DRAM-round-trip if tensors do not fit in L1.

**Expected SiLU / gate_proj ratio: 15–40%.**

This ratio is relatively stable across decode batch sizes because both operations scale with activation map volume (`num_tokens × ffn_dim`), which grows identically for both ops as `num_tokens` increases.

> SiLU latency matters in the decode regime. At 1 token and `hidden_dim=4096`, SiLU may consume 20–35 µs against a gate_proj of 40–80 µs. That is not negligible when the target decode step budget is 100–200 µs.

---

## 4. Key Finding: Prefill Regime (num_tokens >= 128)

At 128+ tokens, gate_proj matmul arithmetic intensity reaches 100–450 FLOP/byte. The matmul begins to saturate FPU throughput: it reuses each weight element across many token rows, amortizing the DRAM load cost.

SiLU does not share this benefit. It has no weight reuse: every output element requires a distinct read-compute-write pass over the activation tensor. SiLU latency grows linearly with `num_tokens × ffn_dim`; matmul latency grows sub-linearly once compute-bound.

**Expected SiLU / gate_proj ratio: typically 4–8% at num_tokens = 128; below 2% at num_tokens = 256.**

At these ratios, eliminating or fusing the SiLU kernel cannot produce more than a 2–5% end-to-end FFN speedup. The optimization effort is not justified by latency reduction alone.

---

## 5. How to Read the Table and Draw Conclusions

1. **Find your operating point** on the `num_tokens` axis. If running autoregressive decode, you are in the 1–32 row range. If running prompt prefill, you are in the 64–256+ range.
2. **Read the SiLU / gate_proj ratio.** If the ratio exceeds 10%, SiLU is a meaningful contributor to FFN latency.
3. **Check elem_mul latency.** Element-wise multiply of SiLU output and up_proj output adds a second memory-bound kernel with similar arithmetic intensity to SiLU. Combined, SiLU + elem_mul may represent 25–50% of decode FFN time.
4. **Consider fusion scope.** A kernel that fuses gate_proj matmul + SiLU + element-wise multiply eliminates two separate DRAM-bandwidth-bound kernels in the decode regime. See `compute_vs_memory_bound_regimes.md` for the threshold analysis.

---

## Next Steps

Proceed to [`compute_vs_memory_bound_regimes.md`](compute_vs_memory_bound_regimes.md) for a detailed regime walkthrough and the fusion decision table.
