# PM IDEAL and FPU UTIL

## What PM IDEAL Represents

`PM IDEAL` is the **performance model's theoretical minimum kernel duration**. It answers the question: *if every memory access were perfectly pipelined and the math engine ran at full throughput with no stalls, how long would this op take?*

The answer comes from the **roofline model** reduced to a single number:

```
PM_IDEAL_cycles = max(compute_cycles, memory_cycles)
```

- If `compute_cycles > memory_cycles`, the op is **compute-bound**: the math engine is the bottleneck even with perfect memory.
- If `memory_cycles > compute_cycles`, the op is **memory-bound**: data movement limits throughput even if the math engine were infinitely fast.

`PM IDEAL [ns]` and `PM IDEAL [cycle]` report this value in the respective units. Wormhole B0 TRISC cores nominally run at approximately 1 GHz, but the exact clock frequency varies by component and silicon revision — do not assume a fixed cycle-to-nanosecond ratio. To obtain the actual clock frequency for your device, check `device_params.json` produced by `process_ops_logs.py`. In practice, because `ops_perf_results.csv` already provides both `[ns]` and `[cycle]` columns, you can use the `[ns]` columns directly for time-based comparisons without needing to know the clock frequency.

> **Note:** PM IDEAL is a *lower bound*, not an achievable target. Real kernels will always take longer due to pipeline stalls, synchronization overhead, and micro-architectural effects not captured by the roofline model. The purpose of PM IDEAL is to give a principled baseline against which actual performance can be compared.

---

## How PM IDEAL Is Computed for a Matmul

For a matrix multiplication of an (M×K) matrix by a (K×N) matrix (tiles: `M_t = M / 32`, `K_t = K / 32`, `N_t = N / 32`), producing 2 × M × K × N total floating-point operations (or in tile units `2 × M_t × K_t × N_t × 32768`):

### Compute cycles

```
compute_cycles = (M_t × K_t × N_t × 2 × 32768) / FPU_peak_ops_per_cycle
```

Breaking down the numerator:

- `M_t × K_t × N_t` — number of tile-triplets (one tile-triplet = one outer-product accumulation step).
- `× 2` — multiply-accumulate counts as two floating-point operations.
- `× 32768` — each tile-triplet performs 32³ = 32,768 multiply-accumulate operations (the output tile is 32×32 and each of its 1,024 elements accumulates 32 partial products from the 32 K-dimension elements of the paired input tile along the inner dimension per tile step, giving 32 × 32 × 32 = 32,768 MACs per tile-triplet).

`FPU_peak_ops_per_cycle` is the number of FP operations the math engine can issue per cycle at peak throughput. On Wormhole B0 this depends on `DATA FORMAT` and `MATH FIDELITY`:

| Data Format | Math Fidelity | Effective FLOPs/cycle for PM IDEAL |
|---|---|---|
| `BFLOAT16` | `HiFi4` | 256 |
| `BFLOAT16` | `HiFi2` | 512 |
| `BFLOAT8_B` | `HiFi2` | 512 |
| `BFLOAT8_B` | `LoFi` | 1024 |

> **Note:** Values above 256 reflect reduced loop iterations in lower-fidelity modes, not a higher hardware FPU ceiling. See Ch4 for the Wormhole B0 hardware ceilings (256 FLOPs/cycle, 8.0 FLOPs/byte ridge point). These values are indicative for Wormhole B0; always cross-reference with the current `tt_metal` architecture specification for the exact parameters of the device and fidelity mode you are profiling.

### Memory cycles

```
memory_cycles = (input_bytes + weight_bytes + output_bytes) / (NoC_BW_bytes_per_cycle × num_active_cores)
```

- `input_bytes`, `weight_bytes`, `output_bytes` — total bytes transferred for each tensor, derived from the shape columns (`INPUT_0_*`, `OUTPUT_0_*`) and `DATA FORMAT`.
- `NoC_BW_bytes_per_cycle` — peak NoC bandwidth per core per cycle. On Wormhole B0 this is typically 32 bytes/cycle per NoC link.
- `num_active_cores` — equals `CORE COUNT` in the CSV.

### Final PM IDEAL

```python
PM_IDEAL_cycles = max(compute_cycles, memory_cycles)
PM_IDEAL_ns     = PM_IDEAL_cycles / core_clock_Hz * 1e9
```

> **Tip:** To manually verify a PM IDEAL value from the CSV, extract `CORE COUNT`, `INPUT_0_*`, `OUTPUT_0_*`, `DATA FORMAT`, and `MATH FIDELITY` for the row and plug them into the formulas above. A large discrepancy between your manual calculation and the CSV value usually means the op performed a different number of tiles than the shape columns imply (e.g., padding or sub-tile alignment).

---

## How FPU UTIL Is Derived

```
FPU_UTIL = PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles
```

`TRISC1` is the math engine RISC processor. Its kernel duration measures how long the FPU was active (including stalls while waiting for unpacked tiles from TRISC0).

Interpretation:

| `FPU UTIL` range | Meaning |
|---|---|
| Close to 1.0 | Math engine ran near theoretical peak; op is well-optimized for compute |
| 0.5 – 0.9 | Moderate FPU inefficiency; possible tile-pipeline stalls or unpacker bottleneck |
| Below 0.5 | Significant stalling; math engine was frequently waiting for data from TRISC0 or L1 |
| Close to 0.0 | Math engine nearly idle; op is dominated by data movement (NCRISC) or host overhead |

> **Tip:** When `FPU UTIL` is low but `NCRISC KERNEL DURATION [ns]` is high, the op is memory-bound: NCRISC (data movement) is too slow to keep TRISC1 fed — TRISC1 stalls waiting for tile data to arrive in L1 rather than being the bottleneck itself. Improving data layout (e.g., switching to a sharded memory config) is the right lever.

---

## Why FPU UTIL Can Exceed 1.0

PM IDEAL is a theoretical minimum, not a hard floor enforced by hardware counters. As a result, `FPU UTIL` values slightly above 1.0 are expected and do not indicate a problem. The two most common causes are:

1. **PM IDEAL rounding:** The performance model rounds tile counts and bandwidth figures to integers. For small ops (low tile counts), this quantization error can make PM IDEAL slightly pessimistic relative to the measured TRISC1 duration.

2. **Measurement noise:** The cycle-counter timestamps have finite resolution. For very short kernels (a few hundred cycles), the start/end event overhead is non-negligible.

Values in the range 1.0–1.05 are therefore **expected and valid**. Only values substantially above 1.05 are cause for concern.

> **Warning:** An `FPU UTIL` substantially greater than 1.05 (e.g., above 1.1) is a signal of a model or tooling bug — either in the performance model parameters (wrong FPU peak, wrong clock frequency) or in the log aggregation logic in `process_ops_logs.py`. Treat such rows as suspect and cross-check the raw per-core logs.

---

## The Relationship Between PM IDEAL and Actual Duration

By construction:

```
DEVICE KERNEL DURATION [cycle] ≥ PM_IDEAL [cycle]
```

This always holds. The **roofline efficiency** of an op is:

```
roofline_efficiency = PM_IDEAL [cycle] / DEVICE KERNEL DURATION [cycle]
```

A value of 1.0 means the op ran at the theoretical roofline limit — this is exceptional and rarely seen outside of microbenchmarks. Typical production workloads fall in the 0.3–0.7 range.

The gap between actual duration and PM IDEAL breaks down into two parts:

1. **FPU stalls** — `stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles`.
2. **Load imbalance and synchronization overhead** — the difference between `TRISC1_KERNEL_DURATION_cycles` on the slowest core and `DEVICE KERNEL DURATION [cycle]`.

The gap breaks down differently depending on the bottleneck. For **compute-bound ops** (TRISC1 dominant):

```
DEVICE KERNEL DURATION [cycle]
  = TRISC1_KERNEL_DURATION_cycles          (slowest core, math-bound path)
  + load_imbalance_cycles
  + synchronization_overhead_cycles
```

For **memory-bound ops** (NCRISC dominant), substitute NCRISC for TRISC1 in the formula above. The stall cycles attributable to FPU inefficiency are:

```
stall_cycles = TRISC1_KERNEL_DURATION_cycles − PM_IDEAL_cycles
```

Improving `FPU UTIL` addresses compute stalls. For memory-bound ops, the equivalent lever is `NOC BW UTIL`. Improving core grid mapping and tensor sharding reduces load imbalance and synchronization overhead in both cases.

---

**Next:** [Chapter 4 — Compute-Bound vs. Bandwidth-Bound Analysis](../ch4_compute_vs_bandwidth/index.md)
