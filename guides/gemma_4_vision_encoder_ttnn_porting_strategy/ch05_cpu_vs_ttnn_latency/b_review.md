# Chapter 5 — Agent B (Critic) Review

## Verdict: Revisions Required

### Issue 1 — MLP FLOP Count Omits the Gate Projection (High Severity)

**Location:** `cpu_baseline_profiling.md`, "FLOP Estimation by Module" section and all downstream tables.

Chapter 1 (`module_hierarchy.md`, lines 200-210) establishes that `Gemma4VisionMLP` has **three** weight matrices: `gate_proj` [1152, 4304], `up_proj` [1152, 4304], and `down_proj` [4304, 1152]. The forward pass is `down_proj(gelu(gate_proj(x)) * up_proj(x))`.

Chapter 5 counts only two MLP projections ("MLP up-projection" and "MLP down-projection"), each at ~8.35 GFLOPs. The missing `gate_proj` has the same shape as `up_proj` and adds another ~8.35 GFLOPs per layer. This error propagates as follows:

| Quantity | Chapter 5 Value | Corrected Value |
|----------|----------------|-----------------|
| MLP FLOPs per layer | ~16.7 GFLOPs | ~25.1 GFLOPs |
| Total FLOPs per layer | ~28.9 GFLOPs | ~37.2 GFLOPs |
| 27 encoder layers total | ~780 GFLOPs | ~1,005 GFLOPs |
| Total vision encoder | ~785 GFLOPs | ~1,010 GFLOPs |
| MLP share of total | 57.4% | 66.5% |

All CPU and TTNN latency estimates, break-even tables, and speedup ratios derived from these FLOP counts need to be recomputed. The qualitative conclusions (TTNN wins at higher token budgets and batch sizes) likely still hold, but the absolute numbers shift upward by roughly 29%.

### Issue 2 — Parameter Count Inconsistency (Low Severity)

**Location:** `index.md` line 29, `cpu_baseline_profiling.md` lines 3, 64.

Chapter 5 states the vision encoder has "approximately 550M parameters." Chapter 1 (`config_parameters.md`, lines 68-72) computes the vision encoder total as ~569M (or ~575M including the multimodal projection). The ~19M discrepancy (~3.4%) likely comes from omitting the position embedding table (~24M parameters). While parameter count does not directly affect the FLOP analysis (the position embeddings contribute negligible FLOPs), the stated figure should be consistent with Chapter 1.

**Fix:** Change "approximately 550M" to "approximately 569M" (or "approximately 575M" if the multimodal projection is included in scope).

### Issue 3 — Weight Size Calculation Underestimates Due to Missing Gate Projection (Medium Severity)

**Location:** `ttnn_latency_projection.md`, "Memory-Bandwidth Estimate" section, lines 66-73.

The per-layer weight traffic calculation lists QKV, output projection, MLP up, and MLP down — but omits the MLP gate projection weights. The missing `gate_proj` adds `1152 * 4304 * 2 = 9.92 MB` per layer:

| Quantity | Chapter 5 Value | Corrected Value |
|----------|----------------|-----------------|
| Total weights per layer | ~30.5 MB | ~40.4 MB |
| 27 layers total | ~823 MB | ~1,091 MB |
| Bandwidth time (288 GB/s) | 2.9 ms | 3.8 ms |

The workload remains compute-bound at batch=1 (since compute time still exceeds bandwidth time), so the qualitative conclusion holds. However, the stated numbers are wrong by ~33%.

---

Items reviewed with no issues found:
- Wormhole B0 hardware specs (compute grid, BF16 peak, DRAM bandwidth, PCIe bandwidth)
- RoPE described as 2D factored, consistent with Chapter 1
- Decision matrix qualitative recommendations are sound given the analysis framework

## Pass 2

All three Pass 1 issues have been addressed:

1. **MLP FLOPs gate projection (was High):** `cpu_baseline_profiling.md` now lists three MLP projections (gate, up, down) at ~8.35 GFLOPs each. Per-layer total is ~37.3 GFLOPs; 27-layer total is ~1,007 GFLOPs; overall vision encoder ~1,010 GFLOPs. Breakdown table shows MLP at 67.1% and attention at 32.6%. All consistent.

2. **Parameter count (was Low):** `index.md` and `cpu_baseline_profiling.md` now state "570M parameters," consistent with Chapter 1's ~569M figure.

3. **Weight size missing gate (was Medium):** `ttnn_latency_projection.md` now includes "MLP gate" at 9.92 MB per layer. Per-layer weight total is ~40.4 MB; 27 layers is ~1,091 MB; bandwidth time is 3.8 ms. Compute-bound conclusion at batch=1 still holds (compute time > bandwidth time).

Spot-checked downstream consistency:
- FLOP scaling across token budgets (70, 140, 280, 560, 1120) verified by independent calculation — table values match.
- Weight transfer size "~1.1 GB (570M params in BF16)" is correct (570M * 2 = 1.14 GB).
- Break-even table speedup ratios are consistent with CPU and TTNN mid-estimates.
- Wormhole B0 hardware specs (262 TOPS BF16, 288 GB/s DRAM, 12.8 GB/s PCIe Gen4 x16) are correct.

**No feedback — chapter approved.**
