# Classification Method

This file gives the step-by-step procedure for classifying a kernel as compute-bound, bandwidth-bound, or overhead-bound using only the CSV columns produced by the device profiler (`ops_perf_results.csv`). No additional instrumentation is needed.

---

## Step-by-Step Classification Procedure

### Step 1 — Compute the Theoretical Arithmetic Intensity

Before opening the CSV, derive the kernel's theoretical `AI` from its input/output shapes and data format byte width. This gives you a prior expectation that the CSV metrics should confirm or contradict.

**For a matmul (M, K, N, BF16):**

```python
bytes_per_element = 2  # BF16
flops  = 2 * M * K * N
bytes  = (M * K + K * N + M * N) * bytes_per_element
AI     = flops / bytes
```

**For an elementwise unary op on tensor shape [H, W], BF16:**

```python
bytes_per_element = 2
flops_per_element = <count from op definition>   # e.g., ~5 for silu
flops  = flops_per_element * H * W
bytes  = 2 * H * W * bytes_per_element           # read + write
AI     = flops / bytes
```

Compare `AI` to the ridge point of **8.0 FLOPs/byte** (derived in [`roofline_model_primer.md`](./roofline_model_primer.md)):

- `AI > 8.0` → initial hypothesis: **compute-bound**.
- `AI < 8.0` → initial hypothesis: **bandwidth-bound**.

> **Note:** The ridge point of 8.0 FLOPs/byte assumes the NoC read link (32 bytes/cycle) is the bandwidth ceiling. If the working set spills to DRAM, the effective ridge point is higher (DRAM bandwidth per core is lower), making it easier for an op to be bandwidth-bound even at moderate AI values.

---

### Step 2 — Check `FPU UTIL`

`FPU UTIL` is defined as:

```
FPU_UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles
```

It measures the fraction of the TRISC1 (compute engine) active time that the FPU was actually computing, as predicted by the ideal performance model.

| `FPU UTIL` value | Interpretation |
|---|---|
| > 0.7 | FPU is heavily used; likely **compute-bound** |
| 0.3 – 0.7 | FPU moderately used; could be balanced or overhead-driven |
| < 0.3 | FPU is mostly idle; the kernel is **not** compute-bound |

> **Warning:** High `FPU UTIL` does not automatically mean the kernel is fast. It means the FPU is doing useful work for most of TRISC1's active time. If `DEVICE KERNEL DURATION` is still much larger than `PM IDEAL`, there may be stalls outside TRISC1 (e.g., data-movement waiting in TRISC0 or TRISC2).

---

### Step 3 — Check `NOC BW UTIL`

`NOC BW UTIL` measures the utilization of the NoC link(s) relevant to the kernel's data movement, typically expressed as a fraction of peak link bandwidth.

| `NOC BW UTIL` value | Interpretation |
|---|---|
| > 0.7 | NoC is near saturation; likely **bandwidth-bound (NoC)** |
| 0.4 – 0.7 | NoC moderately loaded; check whether DRAM latency is a factor |
| < 0.4 | NoC is lightly loaded; bandwidth is not the bottleneck |

Cross-reference `FPU UTIL` and `NOC BW UTIL` together:

- `FPU UTIL` high + `NOC BW UTIL` low (< 0.4) → compute-bound.
- `FPU UTIL` low + `NOC BW UTIL` high → bandwidth-bound (NoC).
- `FPU UTIL` low + `NOC BW UTIL` low (< 0.4) → overhead-bound (see Step 5).

---

### Step 4 — Cross-Check `DEVICE KERNEL DURATION` vs. `PM IDEAL`

`DEVICE KERNEL DURATION` is the wall-clock span from the first core's start to the last core's end. `PM IDEAL` is the theoretical minimum duration predicted by the performance model:

```
PM IDEAL = max(compute_cycles, memory_cycles)
```

Compute the ratio:

```
overhead_ratio = DEVICE KERNEL DURATION / PM IDEAL
```

| Ratio | Interpretation |
|---|---|
| ~1.0 (1.0 – 1.3) | Kernel is running close to the modeled bound; healthy |
| 1.3 – 2.0 | Moderate unexplained overhead; worth investigating |
| > 2.0 | Significant unexplained overhead; kernel is far from ideal |

> **Tip:** When `overhead_ratio > 2.0`, the primary bottleneck is unlikely to be either raw FPU throughput or NoC bandwidth. Instead, look for dispatch latency, pipeline bubbles, synchronization stalls, or tile-loop inefficiency. These are overhead-bound ops.

> **Note:** `PM IDEAL` accounts for compute cycles (`compute_cycles = tile_count × cycles_per_tile`) and memory cycles (`memory_cycles = total_bytes / bytes_per_cycle`). It does not model inter-core synchronization or dispatch overhead. A large ratio always signals work that the performance model does not account for.

---

### Step 5 — Examine the TRISC Duration Breakdown

The four per-core RISC processors profiled in the CSV have distinct roles (BRISC is excluded — it does not have per-cycle counters in the ops_perf output):

| TRISC | Role |
|---|---|
| `NCRISC` | NoC DMA reader/writer: moves input tiles from DRAM/NoC into L1 and writes output tiles from L1 to DRAM/NoC |
| `TRISC0` | Math unpacker: moves tile data from L1 into FPU registers |
| `TRISC1` | Compute: runs the FPU kernel on tiles in L1 |
| `TRISC2` | Math packer: packs computed results from FPU output registers back into L1 |

Interpret the relative durations as follows:

| Pattern | Interpretation |
|---|---|
| `TRISC1 DURATION >> TRISC0 DURATION` and `TRISC1 >> TRISC2` | Compute-bound: FPU is the long pole |
| `NCRISC DURATION >> TRISC1 DURATION` | Read-bandwidth-bound: data reader is stalling the pipeline |
| `TRISC2 DURATION >> TRISC1 DURATION` | Output-buffer-stall: TRISC2 (math packer) is stalling because its L1 output buffer is full, typically because NCRISC or a downstream stage is not consuming tiles fast enough |
| All four durations similarly long | Memory stall across the pipeline; check for DRAM pressure |
| All four durations short relative to `DEVICE KERNEL DURATION` | Dispatch overhead or synchronization stall is dominating |

> **Tip:** For a well-tuned compute-bound matmul, you expect `TRISC1 DURATION` to be the longest by a significant margin. If `TRISC0` and `TRISC1` are similar length, the kernel is hovering near the roofline ridge point.

---

## Decision Flowchart

```
START: Kernel to classify
         │
         ▼
[1] Compute theoretical AI
         │
    ┌────┴─────────────┐
   AI < 8.0         AI >= 8.0
    │                   │
    ▼                   ▼
Hypothesis:         Hypothesis:
Bandwidth-bound     Compute-bound
    │                   │
    └─────────┬──────────┘
              ▼
[2] Read FPU UTIL from CSV
              │
    ┌─────────┴──────────────┐
  > 0.7                   < 0.3
    │                        │
    ▼                        ▼
[5] Check TRISCs:       [3] Read NOC BW UTIL
TRISC1 >> others?            │
    │                ┌───────┴──────────┐
   Yes               > 0.7           < 0.3
    │                  │                │
    ▼                  ▼                ▼
COMPUTE-BOUND    BANDWIDTH-BOUND   [4] Check ratio
                   (NoC)            DKDUR / PM IDEAL
                                        │
                                   > 2.0?
                                   ┌────┴────┐
                                  Yes        No
                                   │         │
                                   ▼         ▼
                             OVERHEAD-    BALANCED /
                               BOUND      INVESTIGATE
```

---

## Classification Result Table

See the Quick-Reference Decision Table in [`index.md`](./index.md) for the summary table of threshold values and classifications.

> **Note:** The thresholds 0.7 and 0.3 are empirical guidelines, not hard boundaries. An op with `FPU UTIL = 0.65` and `NOC BW UTIL = 0.15` is almost certainly compute-bound in practice; use engineering judgment and always corroborate with the TRISC breakdown and `overhead_ratio`.

---

## What to Do With the Classification

| Classification | Recommended Next Steps |
|---|---|
| Compute-bound | Check whether the core grid is optimal; investigate FPU pipeline stalls (Chapter 5) |
| Bandwidth-bound (NoC) | Check tile reuse and whether operands are resident in L1; reduce unnecessary data movement |
| Bandwidth-bound (DRAM) | Reduce tensor sizes, enable activation recomputation, or fuse ops to keep data in L1 |
| Overhead-bound | Profile dispatch latency; check for sub-tile padding waste; consider op fusion |
| Balanced | The op is near the ridge point; both compute and bandwidth optimizations may have diminishing returns |

---

**Next:** [`worked_examples.md`](./worked_examples.md)
