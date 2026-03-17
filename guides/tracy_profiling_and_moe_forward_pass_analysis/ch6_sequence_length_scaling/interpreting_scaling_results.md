# Interpreting Scaling Results

This file explains how to plot the `seq_len` sweep output, read the scaling exponent from a
log-log fit, apply the decision table to identify the most likely root cause, decompose a
mixed gap into its constant and linear components, and diagnose non-monotonic results caused
by tile-count discontinuities.

---

## How to Plot the Results

Load the CSV produced by the sweep script and plot gap duration (y-axis, ms) against
`seq_len` (x-axis) with both axes on a log scale. Log-log axes are essential: a power-law
relationship `gap = a × seq_len^k` appears as a straight line with slope `k` in log-log
space, making the scaling exponent visually obvious.

```python
import csv
import math
import pathlib
import statistics

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
results_path = pathlib.Path("results/moe_gap_scaling.csv")
seq_lens = []
medians = []
p95s = []

with open(results_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        seq_lens.append(int(row["seq_len"]))
        medians.append(float(row["median_ms"]))
        p95s.append(float(row["p95_ms"]))

seq_lens = np.array(seq_lens)
medians = np.array(medians)
p95s = np.array(p95s)

# ---------------------------------------------------------------------------
# Fit a line in log-log space to estimate the scaling exponent
# ---------------------------------------------------------------------------
log_seq = np.log10(seq_lens)
log_med = np.log10(medians)

slope, intercept = np.polyfit(log_seq, log_med, 1)
fit_line = 10 ** (intercept + slope * log_seq)

print(f"Scaling exponent (log-log slope): {slope:.3f}")
print(f"Interpretation: gap ∝ seq_len^{slope:.2f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(seq_lens, medians, "o-", label="Median gap (ms)", color="steelblue")
ax.loglog(seq_lens, p95s, "s--", label="p95 gap (ms)", color="tomato", alpha=0.7)
ax.loglog(seq_lens, fit_line, "k:", linewidth=1.5,
          label=f"Best-fit slope = {slope:.2f}")

ax.set_xlabel("seq_len (log scale)")
ax.set_ylabel("Gap duration (ms, log scale)")
ax.set_title("MoE Forward Pass Gap vs. Sequence Length")
ax.legend()
ax.grid(True, which="both", alpha=0.3)

# Annotate the slope on the plot
mid_idx = len(seq_lens) // 2
ax.annotate(
    f"slope ≈ {slope:.2f}",
    xy=(seq_lens[mid_idx], fit_line[mid_idx]),
    xytext=(seq_lens[mid_idx] * 1.5, fit_line[mid_idx] * 1.3),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=10,
)

plt.tight_layout()
plt.savefig("results/moe_gap_scaling.png", dpi=150)
print("Plot saved to results/moe_gap_scaling.png")
```

> **Tip:** Always inspect the raw data points before drawing conclusions from the fit line.
> If one or two points are clear outliers — significantly above or below the trend — those
> points may indicate tile-count discontinuities (see the non-monotonic section below).
> Remove confirmed outliers only after identifying their cause; do not remove them silently.

---

## Reading the Scaling Exponent

The slope of the best-fit line in log-log space is the scaling exponent `k` where
`gap ≈ C × seq_len^k`. Three canonical values are expected:

| Log-Log Slope | Interpretation |
|---|---|
| k ≈ 0 (±0.1) | Constant gap — independent of seq_len |
| k ≈ 1.0 (±0.15) | Linear gap — proportional to seq_len |
| 0.1 < k < 0.9 | Sublinear — mixed constant + linear, or matmul transitioning regimes |

Slopes outside `[0, 1.2]` are unusual and indicate measurement problems. A slope above 1.2
suggests a quadratic component (unexpected in MoE) or a confound such as DRAM fragmentation
at large `seq_len`. A negative slope would suggest the gap is an artifact of the measurement
method (e.g., the warm-up protocol is inadvertently longer at small `seq_len`).

---

## Decision Table: Scaling Exponent to Root Cause

Use the following table to map the measured scaling exponent to the most likely gap source
and recommended next step.

| Measured Slope | Most Likely Gap Source | Chapter 5 Pattern | Investigation Path |
|---|---|---|---|
| k ≈ 0 | Synchronization barrier (`ttnn.synchronize_device` or `ttnn.wait_for_event`) | **Pattern D** (if first-call only) or **Pattern B** (if every call) | Apply Method 2 from `gap_attribution.md`: search for synchronization calls in the MoE forward pass source and confirm with Tracy zones |
| k ≈ 0, first call only | Program cache miss — kernel recompilation cost | **Pattern D** | Confirm with `ttnn.enable_program_cache()` and verify a subsequent call shows no gap; profile with `TT_METAL_DEVICE_PROFILER` to see dispatch overhead in the CSV |
| k ≈ 1.0, gap magnitude matches CCL estimate | CCL all-to-all or reduce-scatter latency | **Pattern C** | Annotate CCL call with `MoE/dispatch/all_to_all` Tracy zone; verify gap disappears into the zone; estimate expected latency using `scaling_theory.md` formula |
| k ≈ 1.0, gap much larger than CCL estimate | Host-side Python loop over token assignments | **Pattern A** | Add Tracy zones around index construction loops; identify which loop scales with seq_len; tensor-ize the offending operation |
| 0.2 < k < 0.8 | Mixed: constant synchronization barrier + linear CCL or host Python | Mix of **Pattern B/C** | Fit linear regression in linear space (see next section); decompose into constant and linear terms; attribute each to the appropriate root cause |
| Non-monotonic | Tile-count discontinuity — program cache miss at a specific seq_len | **Pattern D** at specific boundary | Identify tile boundary; run fine-grained sweep; check if `seq_len % 32 == 0` behavior changes at the discontinuity |

---

## Decomposing a Mixed Gap with Linear Regression

When the scaling exponent is between 0 and 1, the gap is a superposition of a constant
component (synchronization barrier) and a linear component (CCL or host Python). Fitting
a linear regression model in linear (not log-log) space separates the two contributions.

```python
import numpy as np
from scipy import stats

# Load the sweep results (seq_lens and medians arrays from the plotting script above)

# Fit: gap_ms = constant + slope * seq_len
# scipy.stats.linregress fits y = slope * x + intercept
result = stats.linregress(seq_lens, medians)

constant_ms = result.intercept    # the O(1) component
slope_ms_per_token = result.slope  # the O(seq_len) component (ms per token)
r_squared = result.rvalue ** 2

print(f"Linear regression fit:")
print(f"  Constant term:  {constant_ms:.2f} ms  (synchronization barrier or fixed overhead)")
print(f"  Linear term:    {slope_ms_per_token * 1000:.4f} µs/token")
print(f"  R²:             {r_squared:.4f}")

# Reconstruct the fit line for plotting
fit_linear = constant_ms + slope_ms_per_token * seq_lens

# Attribute the linear component to CCL
d_model = 7168
top_k = 8
bytes_per_element = 2
num_chips = 8
effective_bw = 7e9  # 7 GB/s

# Per-token CCL cost: (top_k * d_model * 2 / num_chips) bytes per token
bytes_per_token = (top_k * d_model * bytes_per_element) / num_chips
ccl_ms_per_token = (bytes_per_token / effective_bw) * 1000  # ms

print(f"\nCCL attribution:")
print(f"  Theoretical CCL slope: {ccl_ms_per_token * 1000:.4f} µs/token")
print(f"  Measured linear slope: {slope_ms_per_token * 1000:.4f} µs/token")

ratio = slope_ms_per_token / ccl_ms_per_token if ccl_ms_per_token > 0 else float("inf")
if 0.8 <= ratio <= 1.2:
    print("  MATCH: measured slope is consistent with CCL all-to-all (±20%).")
    print("  Constant term likely attributable to synchronization barrier.")
elif ratio > 1.2:
    print(f"  MISMATCH: measured slope is {ratio:.1f}x the CCL estimate.")
    print("  Additional linear overhead present — suspect host Python loop.")
else:
    print(f"  MISMATCH: measured slope is only {ratio:.1f}x the CCL estimate.")
    print("  CCL may be partially overlapped or bandwidth overestimated.")
```

### Interpreting the Decomposition

Once you have the constant term and the linear term, each can be attributed independently:

**Constant term (`constant_ms`):** This is the gap at `seq_len → 0` — the overhead that
exists regardless of how many tokens flow through the layer. The primary candidates are:

- A `ttnn.synchronize_device` call that blocks for a fixed duration.
- A fixed-cost host operation such as creating a new tensor descriptor or looking up a
  memory config.
- Program cache miss (Pattern D) — but this should have been eliminated by the warm-up
  protocol. If the constant term is large (~20ms) and disappears after the first call,
  Pattern D is confirmed.

**Linear term (`slope_ms_per_token × seq_len`):** This is the portion of the gap that grows
with token count. The primary candidates are:

- CCL all-to-all latency — compare the measured slope against the theoretical
  `(top_k × d_model × 2 / num_chips) / effective_bw` value.
- Host Python loop over token assignments (Pattern A) — if the slope is much larger than
  the CCL estimate, Python overhead is present.

---

## Worked Example: Attributing the 16ms Gap

Using the hypothetical sweep data from `experiment_design.md`:

```
seq_len,median_ms
64,14.21
128,14.35
256,14.58
512,14.94
1024,15.82
2048,17.45
4096,20.71
```

Running the linear regression gives approximately:

```
Constant term:  13.8 ms
Linear term:    0.00168 ms/token  (1.68 µs/token)
R² = 0.997
```

Theoretical CCL slope at top_k=8, d_model=7168, num_chips=8, bw=7 GB/s:

```
bytes_per_token = 8 * 7168 * 2 / 8 = 14336 bytes
ccl_ms_per_token = 14336 / 7e9 * 1000 = 0.00205 ms/token  (2.05 µs/token)
ratio = 1.68 / 2.05 = 0.82
```

The ratio is 0.82 — within the ±20% tolerance, so the linear term is consistent with CCL
all-to-all. The 13.8ms constant term is not explained by CCL and points to a synchronization
barrier or fixed-cost host operation. This is consistent with the Chapter 5 hypothesis that
the 16ms gap at `seq_len=1024` combines a ~14ms synchronization barrier with ~2ms of CCL.

---

## Non-Monotonic Results and Tile-Count Discontinuities

If the gap vs. `seq_len` plot is not monotonically increasing but instead has a sharp jump
or a flat plateau at one sweep point, the MoE kernel is hitting a tile-count boundary.

### What Causes a Tile-Count Discontinuity

Wormhole B0 processes data in 32×32 tiles. The number of M-tiles for the expert matmul is
`M_t = expert_capacity / 32`. When `seq_len` changes such that `expert_capacity` crosses a
multiple of 32 that changes the tile grid layout, the compiled kernel changes. If the new
kernel is not in the program cache, a program cache miss (Pattern D) occurs — even though
the cache was warmed at the original `seq_len`.

For the standard sweep `{64, 128, 256, 512, 1024, 2048, 4096}`, all points are powers of
two and divide evenly by 32, so tile-count discontinuities are unlikely. Discontinuities
appear more commonly in supplementary fine-grained sweeps.

### Diagnosing a Discontinuity

```python
# Detect non-monotonic points in the sweep output
for i in range(1, len(seq_lens)):
    delta = medians[i] - medians[i - 1]
    expected_increase = slope_ms_per_token * (seq_lens[i] - seq_lens[i - 1])
    if delta > expected_increase * 3.0:
        print(
            f"Possible discontinuity between seq_len={seq_lens[i-1]} "
            f"and seq_len={seq_lens[i]}: "
            f"observed Δ={delta:.2f}ms, expected Δ≈{expected_increase:.2f}ms"
        )
        print(
            f"  Check if expert_capacity changes tile alignment at this boundary."
        )
        # expert_capacity ≈ seq_len * top_k / num_experts
        cap_prev = seq_lens[i - 1] * TOP_K // NUM_EXPERTS
        cap_curr = seq_lens[i] * TOP_K // NUM_EXPERTS
        print(f"  expert_capacity: {cap_prev} → {cap_curr} tokens")
        print(f"  M_t: {cap_prev // 32} → {cap_curr // 32} tiles")
```

### What to Do

1. Run a fine-grained sweep around the discontinuity (e.g., sweep every 32 tokens from
   `seq_len - 64` to `seq_len + 64`).
2. Identify the exact `seq_len` at which the jump occurs. If it coincides with a 32-token
   tile boundary in `expert_capacity`, this confirms a tile-count discontinuity.
3. The remediation is to pad `expert_capacity` to a fixed multiple of 32 that is constant
   across all `seq_len` values in the sweep. This ensures the program cache entry remains
   valid across sweep points.
4. Document the boundary in the gap analysis report as a conditional factor:
   *"The gap increases by an additional ~Xms at seq_len=Y due to a tile-count change in the
   expert matmul kernel. This is a program cache effect and not part of the baseline
   scaling trend."*

---

## Full Attribution Summary

After completing the sweep, plotting, and regression, fill in this summary table and include
it in the gap analysis document (Chapter 7 provides the full template):

| Component | Estimated Duration (ms) at seq_len=1024 | Scaling Law | Pattern | Recommended Action |
|---|---|---|---|---|
| Synchronization barrier | ~14 ms (constant term from regression) | O(1) | Pattern B | Identify the `ttnn.synchronize_device` call; evaluate whether it can be replaced with a fine-grained `ttnn.event` |
| CCL all-to-all | ~2 ms (linear term at seq_len=1024) | O(seq_len) | Pattern C | Annotate the CCL call with a Tracy zone; investigate async overlap with expert matmul |
| Host Python index construction | ~0 ms (linear term consistent with CCL; no additional overhead detected) | O(seq_len) if present | Pattern A | No action required if not detected; re-check if linear slope exceeds CCL estimate |
| Program cache miss | ~0 ms (eliminated by warm-up; seen only on cold first call) | O(1), first call only | Pattern D | `ttnn.enable_program_cache()` is already in effect; ensure shapes are canonicalized |

---

## Next Steps

Proceed to [Chapter 7: Interpretation and Next Steps](../ch7_interpretation_and_next_steps/index.md)
to translate the scaling findings into prioritized optimization actions and document the full
gap analysis for the optimization team.
