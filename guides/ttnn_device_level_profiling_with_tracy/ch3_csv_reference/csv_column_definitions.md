# CSV Column Definitions — `ops_perf_results.csv`

## How the CSV Is Produced

When `TT_METAL_DEVICE_PROFILER=1` is set, every Tensix core running a kernel writes timestamped cycle-counter entries to its own log buffer. After the program finishes, those per-core logs land under:

```
tt_metal/tools/profiler/logs/
```

The directory contains one file per core, each holding raw `(event_id, cycle_count)` pairs. These files are not directly human-readable in aggregate form — they need post-processing.

`process_ops_logs.py` reads all per-core log files, joins them by op dispatch order, computes derived statistics (durations, utilizations, shapes), and writes the result to a single summarized CSV: `ops_perf_results.csv`.

## Running the Post-Processing Script

```bash
python tt_metal/tools/profiler/process_ops_logs.py \
    --output-dir <path-to-output-dir>
```

Key options:

| Option | Required | Description |
|---|---|---|
| `--output-dir` | Yes | Directory where `ops_perf_results.csv` will be written |
| `--device-id` | No | Filter to a single device (zero-indexed); defaults to all devices |

**Output location:** `<output-dir>/ops_perf_results.csv`

> **Warning:** If you skip this step and try to read the raw per-core logs directly, you will see only `(event_id, cycle)` pairs with no op-level context. The CSV is the authoritative artifact for all downstream analysis; always run `process_ops_logs.py` before inspecting results.

> **Tip:** Re-running `process_ops_logs.py` on the same log directory is idempotent — it will overwrite `ops_perf_results.csv` but will not corrupt the source logs.

---

## Complete Column Reference

### Identity Columns

| Column | Units | Description |
|---|---|---|
| `OP TYPE` | string | TTNN op class name, e.g., `ttnn::operations::matmul::Matmul`. This is the high-level Python-visible operation. |
| `OP CODE` | string | Internal kernel identifier string, often more specific than `OP TYPE`. Two ops of the same `OP TYPE` may use different `OP CODE` values depending on tile layout or data format. |
| `DEVICE ID` | int | Zero-indexed index of the Tenstorrent device on which this op ran. In single-device runs this is always `0`. |
| `CORE COUNT` | int | Number of Tensix cores that participated in executing this op. Larger tensors typically map to more cores. |

### Duration Columns

| Column | Units | Description |
|---|---|---|
| `DEVICE KERNEL DURATION [ns]` | nanoseconds | Wall-clock span from the earliest start timestamp across all participating cores to the latest end timestamp. This is the true elapsed time for the op as seen from the device. |
| `DEVICE KERNEL DURATION [cycle]` | cycles | The same wall-clock span expressed in raw Tensix core clock cycles. Use this for cycle-accurate comparisons independent of clock frequency. |
| `BRISC KERNEL DURATION [ns]` | nanoseconds | **Not present in `ops_perf_results.csv`.** BRISC does not have per-cycle counters and is not profiled in the ops_perf output. This column does not appear in the CSV; BRISC timing is not available from the device profiler. |
| `NCRISC KERNEL DURATION [ns]` | nanoseconds | Time the NoC RISC (NCRISC) processor spent in kernel code. NCRISC drives NoC DMA transfers — large values here indicate data-movement bottlenecks. |
| `TRISC0 KERNEL DURATION [ns]` | nanoseconds | Time the first Tensix RISC (TRISC0, the math unpacker) spent in kernel code. TRISC0 unpacks tiles from L1 memory into the math engine's source registers. |
| `TRISC1 KERNEL DURATION [ns]` | nanoseconds | Time the second Tensix RISC (TRISC1, the math engine) spent in kernel code. TRISC1 issues FPU instructions. This is the column to watch for compute-bound ops. |
| `TRISC2 KERNEL DURATION [ns]` | nanoseconds | Time the third Tensix RISC (TRISC2, the math packer) spent in kernel code. TRISC2 packs computed results from output registers back into L1. |

### Performance Model Columns

| Column | Units | Description |
|---|---|---|
| `PM IDEAL [ns]` | nanoseconds | Performance model theoretical minimum kernel duration, assuming perfect memory and compute with no stalls. See [`pm_ideal_and_fpu_util.md`](./pm_ideal_and_fpu_util.md) for the derivation. |
| `PM IDEAL [cycle]` | cycles | Same theoretical minimum expressed in Tensix clock cycles. |
| `FPU UTIL` | fraction (0.0–1.0) | Ratio of actual FPU throughput to peak FPU throughput: `PM_IDEAL_cycles / TRISC1_KERNEL_DURATION_cycles`. A value near 1.0 indicates the math engine ran close to peak; a value near 0 indicates heavy stalling. |
| `NOC BW UTIL` | fraction (0.0–1.0) | Ratio of actual NoC bandwidth used to peak available NoC bandwidth. High values (near 1.0) indicate data-movement saturation. |

### Tensor Shape Columns

| Column | Units | Description |
|---|---|---|
| `INPUT_0_W` | int | Outermost (batch) dimension of the first input tensor. |
| `INPUT_0_Z` | int | Second dimension of the first input tensor. |
| `INPUT_0_Y` | int | Third dimension (rows) of the first input tensor, in elements. |
| `INPUT_0_X` | int | Innermost dimension (columns) of the first input tensor, in elements. |
| `OUTPUT_0_W` | int | Outermost (batch) dimension of the first output tensor. |
| `OUTPUT_0_Z` | int | Second dimension of the first output tensor. |
| `OUTPUT_0_Y` | int | Third dimension (rows) of the first output tensor, in elements. |
| `OUTPUT_0_X` | int | Innermost dimension (columns) of the first output tensor, in elements. |

> **Note:** Additional inputs and outputs use the same pattern with a different numeric suffix: `INPUT_1_W`, `INPUT_1_Z`, etc.

### Precision Columns

| Column | Units | Description |
|---|---|---|
| `MATH FIDELITY` | string | Precision mode used by the math engine, e.g., `HiFi4`, `HiFi2`, `LoFi`. Lower fidelity modes trade numerical accuracy for higher FPU throughput. |
| `DATA FORMAT` | string | Data type of operands, e.g., `BFLOAT16`, `BFLOAT8_B`, `FLOAT32`. Affects both memory bandwidth requirements and the FPU peak throughput used in PM IDEAL. |

---

## Cycle-to-Nanosecond Conversion

All `[cycle]` columns can be converted to nanoseconds using:

```
duration_ns = cycles / core_clock_Hz × 1e9
```

For **Wormhole B0** at the nominal **1 GHz** clock, this simplifies to `duration_ns = cycles` (one cycle = one nanosecond), so `DEVICE KERNEL DURATION [ns]` and `DEVICE KERNEL DURATION [cycle]` are numerically identical under default operating conditions.

> **Note:** The actual clock frequency can vary with power/thermal state. To confirm the clock frequency used in a specific run, use one of these approaches:
>
> 1. **Preferred — read `device_params.json`:** `process_ops_logs.py` writes a `device_params.json` file alongside the CSV under `generated_profile_log_{test_name}/`. Open it and look for the `aiclk` or equivalent clock frequency field. This file is authoritative for the profiling run.
>
> 2. **Arch-based default:** If `device_params.json` is unavailable, check `device.arch()` to confirm the architecture and apply its well-known nominal clock. Avoid calling `device.sfpu_clock_rate()` directly — this method may not exist in all tt-metal versions and will raise an `AttributeError` if absent.

> **Warning:** If you are analyzing results from a device other than Wormhole B0 (e.g., Grayskull or Blackhole), verify the clock frequency before applying the formula above. The frequency is architecture-specific.

---

## `DEVICE KERNEL DURATION` vs. Individual RISC Durations

`DEVICE KERNEL DURATION` is a **wall-clock measurement** across all participating cores:

```
DEVICE KERNEL DURATION = max(all core end timestamps) − min(all core start timestamps)
```

Within that span, each core runs five independent RISC processors (BRISC, NCRISC, TRISC0, TRISC1, TRISC2) in parallel. The individual RISC duration columns report how long each processor was actively executing kernel code on the **slowest** (representative) core.

Key implications:

- `DEVICE KERNEL DURATION` will always be greater than or equal to any individual RISC duration, because it includes load imbalance across cores.
- A large gap between `DEVICE KERNEL DURATION` and the maximum RISC duration on one core indicates that some cores finished earlier than others — a load-balancing opportunity.
- The five RISC processors on a single core run concurrently. Their durations do not add; instead, the longest one among them determines that core's contribution to the device kernel duration.

> **Tip:** When diagnosing a slow op, compare `TRISC1 KERNEL DURATION [ns]` (math engine) against `NCRISC KERNEL DURATION [ns]` (data movement). Whichever is longer reveals whether the bottleneck is compute or memory.

---

**Next:** [`pm_ideal_and_fpu_util.md`](./pm_ideal_and_fpu_util.md)
