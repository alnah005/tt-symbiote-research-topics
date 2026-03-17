# Compression Analysis — Chapter 4: SiLU vs. Matmul Comparison

## Crucial updates: yes

---

### Duplication 1: SiLU Arithmetic Intensity Derivation (0.5 FLOP/byte)

**Source (original):**
`ch02_silu_on_wormhole_hardware/cycles_vs_matmul.md` — Section "SiLU Arithmetic Intensity" (lines 103–114)

```
FLOPs  ≈ 2 per element (one multiply, plus the sigmoid approximation treated as ~1 FLOP
          for the roofline; note the actual instruction count is higher, but FLOPs ≈ 2)
Bytes  = 4 per element (2 bytes read BF16 + 2 bytes write BF16)

Arithmetic intensity ≈ 2 / 4 = 0.5 FLOP/byte

SiLU is **deeply memory-bound at all practical tensor sizes** (0.5 << 437).
```

**Duplicate:**
`ch04_silu_vs_matmul_comparison/roofline_analysis.md` — Section 2 "Arithmetic Intensity of SiLU" (lines 22–27)

```
- **FLOPs per element:** 2 (one multiply, one sigmoid approximation via `exp(x) / (1 + exp(x))`)
- **Bytes per element (BF16 read + write):** 4 (2 bytes read input tile, 2 bytes write output tile)
- **Arithmetic intensity:** 2 FLOP / 4 bytes = **0.5 FLOP/byte**

This is 874× below the ridge point. SiLU is deeply memory-bound at every practical token count
and hidden dimension.
```

**Assessment:** The three-step derivation (FLOPs = 2, Bytes = 4, AI = 0.5 FLOP/byte) and the "deeply memory-bound" conclusion are near-verbatim across both files. Chapter 4 adds only the "874× below the ridge point" phrasing; the derivation itself is fully redundant.

**Recommended action for Agent A:** In `ch04_silu_vs_matmul_comparison/roofline_analysis.md` Section 2, replace the repeated derivation block with a one-line statement of the result and a cross-reference:

> `ttnn.silu` has an arithmetic intensity of **0.5 FLOP/byte** (derived in `ch02_silu_on_wormhole_hardware/cycles_vs_matmul.md` §SiLU Arithmetic Intensity). This is 874× below the Wormhole B0 ridge point of 437 FLOP/byte, placing SiLU firmly in the memory-bound regime at every practical token count and hidden dimension.

---

### Duplication 2: Wormhole B0 Hardware Ceilings Table

**Source (original):**
`ch02_silu_on_wormhole_hardware/cycles_vs_matmul.md` — Section "Wormhole B0 Reference Parameters" (lines 70–76)

| Resource | Value |
|---|---|
| Peak FPU throughput (BF16) | ~131 TFLOP/s aggregate |
| DRAM bandwidth (aggregate) | ~300 GB/s |
| L1 bandwidth (est. aggregate) | ~900+ GB/s |
| Ridge point (DRAM) | ~131e12 / 300e9 ≈ **437 FLOP/byte** |

**Duplicate:**
`ch04_silu_vs_matmul_comparison/roofline_analysis.md` — Section 1 "Wormhole B0 Hardware Ceilings" (lines 9–14)

| Parameter | Value | Source |
|---|---|---|
| BF16 FPU peak throughput | 131 TFLOP/s | Hardware specification |
| DRAM bandwidth (practical peak) | ~300 GB/s | 6 controllers × 12 GDDR6 banks |
| Ridge point | 131e12 / 300e9 ≈ **437 FLOP/byte** | Ratio of compute ceiling to memory ceiling |

**Assessment:** All three core rows (131 TFLOP/s, ~300 GB/s, ~437 FLOP/byte) are near-verbatim. The Chapter 4 table adds a "Source" column with annotations but otherwise restates the same hardware constants already established in Chapter 2. A full re-derivation of the ridge point in Chapter 4 (which Chapter 2 already derived) is the specific redundancy.

**Recommended action for Agent A:** In `ch04_silu_vs_matmul_comparison/roofline_analysis.md` Section 1, retain the three-row table for reader convenience (it is a quick reference used throughout the chapter), but add a note immediately after the table directing readers to the Chapter 2 source rather than re-deriving the ridge point:

> These constants were established in `ch02_silu_on_wormhole_hardware/cycles_vs_matmul.md` §Wormhole B0 Reference Parameters. They are reproduced here as a local reference; see Chapter 2 for derivation details.

Remove the two-sentence derivation paragraph that follows the table (lines 15–16 of `roofline_analysis.md`) since it duplicates the explanation already present in Chapter 2.

---

### Notes on Non-Crucial Repetition (Not Flagged)

The following repeated content was reviewed and judged **not crucial** — it is either a single-line factual statement or contextually appropriate cross-chapter summarization that does not warrant a cross-reference replacement:

- The "15–40% of gate_proj matmul latency" estimate appears in Ch02 `cycles_vs_matmul.md`, Ch03 `measurement_methodology.md`, and Ch04 `compute_vs_memory_bound_regimes.md`. Each instance is a single summary sentence serving a different framing purpose (prior estimate, validation check, and regime analysis respectively). Not crucial.
- The qualitative decode vs. prefill regime narrative (memory-bound at small M, compute-bound at large M) recurs across Ch02, Ch03, and Ch04. This is appropriate progressive reinforcement and does not involve verbatim multi-line blocks.
- The SwiGLU compute graph (`gate_proj → silu → elem_mul → down_proj`) appears in shorthand in Ch04 but is not duplicated as the full annotated ASCII graph from Ch01 `ffn_compute_graph.md`. Not crucial.

## Agent A Change Log — C Feedback Pass 1
- roofline_analysis.md §2: Replaced SiLU AI derivation block with one-line result (0.5 FLOP/byte) + cross-reference to Chapter 2, cycles_vs_matmul.md
- roofline_analysis.md §1: Removed ridge-point re-derivation paragraph; kept 3-row hardware table; added cross-reference to Chapter 2, cycles_vs_matmul.md

---

## Pass 2 Verification

**Fix 1 — §2 SiLU AI derivation block (roofline_analysis.md lines 19–27):** Confirmed applied. The three-bullet derivation (FLOPs = 2, Bytes = 4, AI = 0.5 FLOP/byte) has been replaced with a single-sentence result ("SiLU arithmetic intensity is 0.5 FLOP/byte (2 FLOP per element / 4 bytes per BF16 read-write pair)...") followed by a cross-reference to Chapter 2, `cycles_vs_matmul.md`. Fix is correct and complete.

**Fix 2 — §1 ridge-point re-derivation paragraph (roofline_analysis.md lines 15–16):** Confirmed applied. The two-sentence derivation paragraph has been removed. The three-row hardware table is retained. A single cross-reference line ("See Chapter 2, `cycles_vs_matmul.md` for the derivation of these values.") immediately follows the table. Fix is correct and complete.

## Crucial updates: no

No further duplications found.
