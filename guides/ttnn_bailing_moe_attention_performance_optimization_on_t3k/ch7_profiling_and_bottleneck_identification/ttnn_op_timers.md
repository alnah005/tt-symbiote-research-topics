# TTNN Op Timers

TTNN op timers give the fastest path to a ranked list of which ops consume the most wall-clock time in a decode step, without requiring a special build or a separate profiling server. The timer captures the host-side elapsed time from Python call entry to op completion, which includes dispatch latency, device execution, and any blocking synchronization. For decode steps where the device is fast relative to the host dispatch loop, this is the correct metric to optimize.

## 1. Enabling Op-Level Timing

There are three complementary mechanisms. Use them in the order shown: environment variable first for a quick snapshot, the context manager for selective measurement, and compile-time flags only if you need accurate device-execution times stripped of dispatch overhead.

### 1.1 Environment Variable

Set `TT_METAL_ENABLE_PROFILER=1` before launching your Python process. This activates the global op profiler; every TTNN op executed in the process emits a timing record.

```bash
export TT_METAL_ENABLE_PROFILER=1
export TT_METAL_PROFILER_SYNC=1   # optional: force device sync after every op for accurate per-op times
python run_decode_step.py
```

`TT_METAL_PROFILER_SYNC=1` inserts a `ttnn.synchronize_devices(mesh)` call after each op before recording the end timestamp. Without it, the timer captures host-dispatch latency only, which is correct for identifying dispatch-bound ops but underestimates execution time for long-running kernels such as `paged_sdpa_decode`. For a first pass, enabling sync is recommended because it gives unambiguous per-op attribution.

The profiler writes its output to a CSV file whose path is printed at process exit:

```
[TT_METAL] Op profiler output written to: /tmp/tt_metal_profiler_<pid>/op_perf_results.csv
```

### 1.2 `ttnn.tracer` Context Manager

For selective profiling of just the attention forward pass — avoiding noise from model setup, weight loading, and the surrounding decoder layers — wrap the call in a `ttnn.tracer` block:

```python
import ttnn

# One warm-up step to trigger JIT compilation before profiling.
model.attention(hidden_states, ...)   # warm-up: not timed

with ttnn.tracer.profile(output_dir="/tmp/attn_profile", device=mesh):
    model.attention(hidden_states, ...)  # decode step being profiled
```

The `ttnn.tracer.profile` context manager starts the profiler on entry, runs the enclosed code, synchronizes all devices on exit, and writes the CSV. Parameters:

- `output_dir` (str): directory path where the profiler report is written.
- `device`: the `MeshDevice` (or single device) to synchronize on exit.
- `enabled` (bool, default True): when set to False the context manager is a no-op; useful for disabling profiling during warm-up iterations without changing code structure.

Using this form is strongly preferred over the environment variable when the surrounding model code performs many ops unrelated to attention, because it limits the report to the operations you care about.

### 1.3 Compile-Time Flag (Device-Side Counters)

For the highest-accuracy device-side kernel execution times, rebuild `tt-metal` with the profiler enabled at compile time:

```bash
cmake -B build -DENABLE_PROFILER=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tt_metal -j$(nproc)
```

With this build, the profiler captures hardware performance counter timestamps from inside the kernel rather than from the host. The resulting CSV has an additional column `device_kernel_duration_ns` that is independent of PCIe round-trip latency and host dispatch jitter. This is the definitive source for kernel execution times referenced in Chapters 2–6 as `[ESTIMATE]` targets.

## 2. Reading the Op Timer Report

The output CSV has one row per TTNN op invocation. The columns relevant for decode analysis are:

Op timer CSV columns

| Column | Meaning |
|---|---|
| `op_name` | TTNN Python-level op name (e.g., `ttnn::linear`, `ttnn::paged_sdpa_decode`) |
| `op_hash` | Unique compile-time hash identifying the kernel variant and config |
| `host_start_ns` | Host timestamp (nanoseconds) at Python call entry |
| `host_end_ns` | Host timestamp at completion (sync'd if `PROFILER_SYNC=1`) |
| `host_duration_us` | `(host_end_ns - host_start_ns) / 1000`, the primary latency column |
| `device_kernel_duration_ns` | Device-side kernel time **(compile-time build only** — present only when built with `-DENABLE_PROFILER=ON`; absent in environment-variable-only runs**)** |
| `call_site` | Python file and line number of the TTNN call (Python stack frames preserved) |
| `metadata` | Freeform string with tensor shapes, memory configs, and math fidelity |

The most useful workflow is to sort by `host_duration_us` descending and examine the top 10 rows. For a Ling decode step at batch=1 on T3K, you should expect the report to contain approximately 30–50 op rows for the attention layer alone (counting every `ttnn.to_memory_config`, `ttnn.reshape`, `TTNNRMSNorm`, `TTNNRotaryPositionEmbedding`, `paged_sdpa_decode`, and the fused QKV matmul).

### 2.1 Mapping Op Names to `TTNNBailingMoEAttention` Call Sites

The `op_name` values produced by TTNN op timers map to call sites in `TTNNBailingMoEAttention` as follows:

Op name to source-code mapping

| `op_name` in CSV | Source call site in `TTNNBailingMoEAttention` | Chapter |
|---|---|---|
| `ttnn::linear` (first occurrence) | Fused QKV projection via `TTNNLinearIColShardedWAllReduced` | Ch2 |
| `ttnn::all_reduce` | CCL all-reduce following QKV matmul | Ch2 |
| `ConcatMeshToTensor` | Host-side tensor gather step of `_to_replicated` | Ch3 |
| `ttnn::from_torch` | Host-side `ReplicateTensorToMesh` push step of `_to_replicated` | Ch3 |
| `ttnn::to_memory_config` (Q, K pre-RoPE) | HEIGHT_SHARDED transition for RoPE input | Ch4 |
| `ttnn::rotary_embedding` | `TTNNRotaryPositionEmbedding` (non-distributed; `partial_rotary_factor=0.5`) | Ch6 |
| `ttnn::to_memory_config` (Q, K post-RoPE) | DRAM eviction after RoPE output | Ch4 |
| `ttnn::reshape` (×8, four per norm call for Q and K each) | 4D→3D→2D (and 2D→3D→4D) reshapes around `TTNNRMSNorm` for Q and K: (1,1,16,128) → (1,16,128) → (16,128) before TTNNRMSNorm, and (16,128) → (1,16,128) → (1,1,16,128) after | Ch6 |
| `ttnn::rms_norm` (×2) | `TTNNRMSNorm` applied to Q and then K (when `use_qk_norm=True`) | Ch6 |
| `ttnn::to_memory_config` (Q, K post-norm) | HEIGHT_SHARDED→DRAM transitions after norm output | Ch4 |
| `ttnn::paged_fill_cache` | `paged_update_on_device` for K and V cache update | Ch4, Ch5 |
| `ttnn::paged_sdpa_decode` | Scaled dot-product attention kernel | Ch5 |
| `ttnn::linear` (last occurrence) | Output projection after SDPA | Ch2 |

The `call_site` column in the CSV gives the Python file and line number directly, but the table above provides the architectural context needed to interpret what each op is doing and which optimization chapter addresses it.

### 2.2 Using the `metadata` Column

The `metadata` column contains the tensor shapes and memory configs in flight when the op was dispatched. For example, a `ttnn::to_memory_config` row might have:

```
metadata: input=(1,1,16,128) BF16 HEIGHT_SHARDED->DRAM_INTERLEAVED output=(1,1,16,128) BF16
```

This metadata is essential for distinguishing between the many `ttnn::to_memory_config` calls: there are six or more in a single decode step (see Chapter 4, [`decode_tensor_lifecycle.md`](../ch4_memory_config_transitions/decode_tensor_lifecycle.md)), and they differ only in source and destination memory config. The metadata string encodes both.

## 3. Example Invocation for Decode Batch=1

The following script runs one warm decode step, then one profiled step, and writes the sorted report to stdout.

```python
import ttnn
import torch
import pandas as pd
import os

os.environ["TT_METAL_ENABLE_PROFILER"] = "1"
os.environ["TT_METAL_PROFILER_SYNC"] = "1"

# --- model setup (abbreviated) ---
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
model = TTNNBailingMoEAttention(config, mesh)
hidden_states = make_decode_input(batch=1, seq_len=1, hidden_size=4096, mesh=mesh)
position_ids = torch.zeros(1, 1, dtype=torch.long)
kv_cache = init_paged_kv_cache(mesh)

# --- warm-up: run once to trigger JIT compilation, no timing ---
with ttnn.tracer.profile(output_dir=None, device=mesh, enabled=False):
    _ = model(hidden_states, position_ids, kv_cache)
ttnn.synchronize_devices(mesh)

# --- profiled decode step ---
profile_dir = "/tmp/bailing_decode_profile"
with ttnn.tracer.profile(output_dir=profile_dir, device=mesh, enabled=True):
    output = model(hidden_states, position_ids, kv_cache)
ttnn.synchronize_devices(mesh)

# --- print top-10 ops by host_duration_us ---
csv_path = os.path.join(profile_dir, "op_perf_results.csv")
df = pd.read_csv(csv_path)
df_sorted = df.sort_values("host_duration_us", ascending=False)
print(df_sorted[["op_name", "host_duration_us", "call_site", "metadata"]].head(10).to_string())
ttnn.close_mesh_device(mesh)
```

### Expected Output Structure

On a T3K decode step (batch=1, seq_len=1) with `TT_METAL_PROFILER_SYNC=1`, the top-10 output should resemble the following structure. Absolute numbers are estimates pending empirical measurement on a specific firmware and software revision.

Expected top-10 op timer output — Ling decode step batch=1, T3K [ESTIMATE]

| Rank | `op_name` | `host_duration_us` [ESTIMATE] | Notes |
|---|---|---|---|
| 1 | `ttnn::paged_sdpa_decode` | 180–320 µs | Dominant in long-context decode; see Ch5 |
| 2 | `ttnn::all_reduce` | 60–120 µs | CCL over Ethernet; see Ch2 |
| 3 | `ttnn::linear` (QKV) | 30–80 µs | Fused matmul; see Ch2 |
| 4 | `ttnn::rms_norm` (Q) | 35–50 µs | QK norm Q branch; see Ch6 |
| 5 | `ttnn::rms_norm` (K) | 18–28 µs | QK norm K branch; see Ch6 |
| 6 | `ttnn::to_memory_config` (Q→SDPA) | 20–28 µs | Priority 1 transition; see Ch4 |
| 7 | `ConcatMeshToTensor` | 7–25 µs | Host gather for `_to_replicated`; see Ch3 |
| 8 | `ttnn::from_torch` | 5–15 µs | Host push for `_to_replicated`; see Ch3 |
| 9 | `ttnn::rotary_embedding` | 8–18 µs | Non-distributed RoPE; see Ch6 |
| 10 | `ttnn::to_memory_config` (K→RoPE) | 10–16 µs | Pre-RoPE re-shard; see Ch4 |

The ordering will shift with sequence length (longer sequences favor SDPA's dominance), batch size, and hardware firmware version. The structure — a few large contributors and a long tail of small transitions — is stable across configurations.

## 4. Interpreting Results: Finding the >80% Contributors

### Step 1: Sum the total decode step time

```python
T_total = df["host_duration_us"].sum()
print(f"Total attention decode time: {T_total:.1f} µs")
```

### Step 2: Compute each op's fraction

```python
df["fraction_pct"] = 100.0 * df["host_duration_us"] / T_total
df_sorted = df.sort_values("fraction_pct", ascending=False)
df_sorted["cumulative_pct"] = df_sorted["fraction_pct"].cumsum()
print(df_sorted[["op_name", "host_duration_us", "fraction_pct", "cumulative_pct"]].head(15).to_string())
```

### Step 3: Identify the cutoff for 80%

The row where `cumulative_pct` first exceeds 80 defines the minimal set of ops that account for >80% of decode latency. In practice, for Ling on T3K, this cutoff is typically reached within the top 3–5 ops [ESTIMATE].

### Distinguishing Dispatch-Bound from Execution-Bound Ops

Without the compile-time profiler, `host_duration_us` conflates dispatch overhead with device execution. To separate them, run the same measurement with and without `TT_METAL_PROFILER_SYNC=1`:

- **Dispatch-bound:** the op appears LARGE when measured without sync, and the with-sync time is only marginally larger (sync adds little overhead because device execution completes quickly relative to dispatch). The CPU spends most of its time dispatching commands, not waiting for the device. Because with-sync time = dispatch time + device execution time, with-sync can never be smaller than without-sync; a large without-sync time combined with a small additional increment when sync is added signals dispatch dominance.
- **Execution-bound:** the op appears SMALL when measured without sync (dispatch completes quickly, typically 2–15 µs), and the with-sync time is LARGE (the device takes significant time to execute). The CPU is waiting on the device for most of the measured interval.

For decode batch=1 on T3K, the `paged_sdpa_decode` kernel is overwhelmingly execution-bound: it will appear large with sync and small without. The `ttnn::to_memory_config` calls tend to be dispatch-bound plus short device operations: their sync and no-sync times differ by only a few microseconds [ESTIMATE]. The `ConcatMeshToTensor` op is a blocking host-side read and appears large in both modes. `ttnn::from_torch` is execution-bound for large tensors: dispatch only without sync (2–10 µs [ESTIMATE]); includes DMA wait with sync (5–15 µs [ESTIMATE]).

---

**Next:** [`tracy_profiling.md`](./tracy_profiling.md)
