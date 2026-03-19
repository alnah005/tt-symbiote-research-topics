# CSV Signatures for Low FPU Utilization

This file provides the exact CSV column patterns that identify each cause of low `FPU UTIL`. For the mechanism behind each cause, see [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md). For the remediation parameters, see [`remediation_levers.md`](./remediation_levers.md).

(See Chapter 4 for FPU UTIL thresholds and ridge point derivation.)

---

## Diagnostic Checklist

Rule out causes in this order. Each step takes less than a minute to check. Stop at the first match.

1. **Cause 6 — Program cache miss** (rule out first; takes seconds to verify)
2. **Cause 1 — Insufficient tile count** (check CORE COUNT vs. output tile grid)
3. **Cause 3 — Math fidelity mismatch** (check MATH FIDELITY column)
4. **Cause 2 — Sub-optimal data format** (check DATA FORMAT column)
5. **Cause 4 — TRISC0/TRISC2 pipeline stalls** (compare TRISC0 vs. TRISC1 durations)
6. **Cause 5 — NoC contention** (check NOC BW UTIL alongside FPU UTIL)
7. **Cause 7 — Incorrect loop count in kernel** (reached only after all above are ruled out)

This order is chosen to progress from causes that require only a single column check (Causes 6, 3, 2) to causes that require computed ratios (Cause 4), to causes that require combined conditions (Cause 5), to causes that require kernel-level investigation (Cause 7).

---

## Cause 6 — Program Cache Miss

**Rule this out first.** It is the easiest to identify and the easiest to eliminate.

**CSV pattern:**

```
DEVICE KERNEL DURATION [ns] (call 1)  >>  10 × DEVICE KERNEL DURATION [ns] (call 2+)
```

- The first call in a session with a given op shape has a `DEVICE KERNEL DURATION` more than 10× larger than all subsequent calls with identical parameters.
- Steady-state calls (call 2, 3, …) exhibit consistent `DEVICE KERNEL DURATION` and `FPU UTIL`.

**How to check:** Sort the CSV by `OP NAME` and then by call index. Compare the first-call duration against the median duration. A ratio > 10 confirms a cache miss on the first call.

> **Note:** If you see inflated first-call durations and the program cache is already enabled, verify that the shapes and config parameters are truly identical across calls. A single differing parameter (e.g., batch size) counts as a different cache key.

---

## Cause 1 — Insufficient Tile Count

**CSV pattern:**

```
FPU UTIL < 0.2
CORE COUNT > (M_t × N_t) / 4
```

Where `output_tiles = M_t × N_t = ⌈M/32⌉ × ⌈N/32⌉`.

The second condition is not a direct CSV column — it requires computing the expected tile count from the op's shape and comparing it against `CORE COUNT`. Shape data is stored in decomposed dimension columns: `INPUT_0_Y` and `INPUT_0_X` for the first input, `INPUT_1_Y` and `INPUT_1_X` for the second input. For a matmul, compute `M_t = INPUT_0_Y / 32` and `N_t = INPUT_1_X / 32`.

**Practical threshold:** if `M_t × N_t / CORE COUNT < 4`, the op is tile-starved and Cause 1 applies. `TRISC1 KERNEL DURATION` is short and `NOC BW UTIL` is low. See [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md) for the full mechanism.

---

## Cause 2 — Sub-Optimal Data Format

**CSV pattern:**

```
DATA FORMAT == "FLOAT32"
FPU UTIL ≈ 0.5 × expected_BF16_util
```

Where `expected_BF16_util` is the `FPU UTIL` you would expect if the same (M, K, N) shape ran in BF16. If you have a BF16 baseline for comparison, a FP32 run should show approximately half the `FPU UTIL`.

**How to check:** Inspect the `DATA FORMAT` column. `"FLOAT32"` immediately flags this cause; no ratio computation is needed. Both Cause 2 (`DATA FORMAT == "FLOAT32"`) and Cause 3 (`MATH FIDELITY == "HiFi4"`) can be present simultaneously. See [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md) for the full mechanism.

---

## Cause 3 — Math Fidelity Mismatch

**CSV pattern:**

```
MATH FIDELITY == "HiFi4"
TRISC1 KERNEL DURATION [ns] (HiFi4 run) ≈ 4 × TRISC1 KERNEL DURATION [ns] (LoFi baseline)
```

The diagnostic signal is the absolute `TRISC1 KERNEL DURATION`: a `HiFi4` run requires ~4× more compute iterations per tile than `LoFi`, so `TRISC1 KERNEL DURATION` will be ~4× longer. `FPU UTIL` stays roughly constant (PM IDEAL scales proportionally), so **do not expect low FPU UTIL as the signature for this cause.** See [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md) for the full mechanism.

**How to check:** Inspect `MATH FIDELITY`. If `"HiFi4"` is present and `TRISC1 KERNEL DURATION` is ~4× a `LoFi` baseline, Cause 3 applies. Causes 2 and 3 are checked before Cause 4 because they require only a single column lookup.

---

## Cause 4 — TRISC0/TRISC2 Pipeline Stalls

**CSV pattern:**

```
TRISC0 KERNEL DURATION [ns] > 1.2 × TRISC1 KERNEL DURATION [ns]
```

The unpacker (TRISC0) is slower than the math engine (TRISC1), causing TRISC1 to stall waiting for tiles. Secondary signals: `FPU UTIL < 0.5` for a compute-bound op; `NOC BW UTIL` moderate (0.4–0.7); `TRISC2 KERNEL DURATION` may also be elevated. Cause 4 differs from Cause 5 in that `NOC BW UTIL < 0.8` (not fully saturated). See [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md) for the full mechanism.

**How to compute the ratio:** Both `TRISC0 KERNEL DURATION [ns]` and `TRISC1 KERNEL DURATION [ns]` are direct CSV columns. Compute `TRISC0 / TRISC1` for each row; values above 1.2 indicate an unpacker bottleneck.

---

## Cause 5 — NoC Contention

**CSV pattern:**

```
NOC BW UTIL > 0.8
FPU UTIL < 0.3
```

Both conditions must hold simultaneously in a context where AI > 8.0 FLOPs/byte (compute-bound region). High `NOC BW UTIL` alone is not a problem for bandwidth-bound ops with AI < ridge point. See [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md) for the full mechanism.

**How to confirm:** Reduce `CORE COUNT` (via `compute_with_storage_grid_size`) and re-profile. If `NOC BW UTIL` drops and `FPU UTIL` rises, NoC contention was the cause.

---

## Cause 7 — Incorrect Loop Count in Kernel

**CSV pattern:**

```
FPU UTIL  stable across calls, consistently < 0.4
No other cause (1–6) identified
```

This is a residual diagnosis. All of the following must be true before concluding Cause 7:

- `FPU UTIL` is stable (not anomalous on first call — ruling out Cause 6).
- `M_t × N_t / CORE COUNT ≥ 4` — sufficient tiles (ruling out Cause 1).
- `MATH FIDELITY` is not `"HiFi2"` or `"HiFi4"` (ruling out Cause 3; only proceed if `MATH FIDELITY == "LoFi"`).
- `DATA FORMAT` is not `"FLOAT32"` (ruling out Cause 2).
- `TRISC0 KERNEL DURATION ≤ 1.2 × TRISC1 KERNEL DURATION` (ruling out Cause 4).
- `NOC BW UTIL ≤ 0.8` or AI < ridge point (ruling out Cause 5).

If all six prior causes are ruled out and `FPU UTIL` remains consistently below 0.4 for a compute-bound shape, the most likely explanation is a loop count discrepancy in the generated kernel (padded tile counts causing FPU cycles wasted on zero-padded tiles). See [`causes_of_low_fpu_util.md`](./causes_of_low_fpu_util.md) for investigation steps and the padding-based workaround.

---

**Next:** [`remediation_levers.md`](./remediation_levers.md)
