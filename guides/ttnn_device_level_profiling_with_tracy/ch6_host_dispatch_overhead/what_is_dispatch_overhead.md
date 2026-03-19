# What Is Dispatch Overhead?

## The TTNN Dispatch Pipeline

Every call to a TTNN op travels through a sequence of host and device stages before the kernel begins executing. The pipeline has a host-side half and a device-side half; both contribute to the gap between "Python call entered" and "kernel first instruction executing."

### Host-Side Stages

**1. Python call and argument validation**

The Python-level `ttnn.matmul(a, b, ...)` call dispatches through the TTNN Python bindings into the C++ op dispatch function. Before any device-related work begins, the dispatch layer validates op arguments: tensor shapes are checked for compatibility, data formats are verified, and memory configuration objects (which device, which DRAM bank, which tile layout) are resolved from the supplied parameters or inferred from defaults. This validation runs entirely on the host CPU and involves no MMIO.

**2. Tile layout computation and memory config resolution**

TTNN ops operate on tensors stored in tiled format (32×32 tiles for most compute ops). If the input tensors are in row-major format, the dispatch layer must compute the tiled layout descriptor and — if an implicit conversion is needed — schedule a tilize op. Memory configuration resolution determines the DRAM sharding strategy, L1 buffer allocation, and the core grid assignment for the op.

**3. Program object creation or cache lookup**

A TTNN program object encapsulates the compiled kernel binary, runtime argument descriptors, and core grid configuration for a specific (op, shapes, data format, math fidelity) combination. The dispatch layer checks the program cache for an existing entry matching the current parameters.

- **Cache hit:** the program object is retrieved in a few µs. This is the fast path.
- **Cache miss (first call or shape change):** the dispatch layer must compile the kernel from source (or locate the precompiled binary), instantiate the program object, and populate all runtime argument buffers. This can take 100–500 µs and is entirely host-side work.

> **Note:** `DEVICE KERNEL DURATION [ns]` in `ops_perf_results.csv` is unaffected by whether the call was a cache hit or miss. The CSV clock only starts when the kernel begins executing on Tensix. A 500 µs cache-miss compilation is invisible in the CSV but dominates the Tracy zone for that op. Always compare first-call and steady-state Tracy zone durations when investigating unexpectedly slow op invocations.

**4. Host-to-device command buffer write**

Once the program object is ready (from cache or freshly compiled), the dispatch layer serializes it into the device command buffer and issues an MMIO write over PCIe to transfer the command to the device. This write is the last host-side action before control passes to device firmware. The MMIO write latency depends on PCIe generation and command buffer size; typical values are 1–5 µs for a single op's command.

### Device-Side Pre-Kernel Stages

After the host MMIO write completes, the device firmware must do additional work before the kernel begins executing. This device-side overhead is not captured by `DEVICE KERNEL DURATION [ns]` — cycle counters on Tensix cores only start when kernel code begins running on those cores.

**5. Firmware command decode**

The dispatch firmware (running on dedicated dispatch Tensix cores) reads the incoming command from the command queue, decodes the op type, and looks up the associated kernel binary and runtime arguments. This decode step is typically a few hundred nanoseconds to ~1 µs.

**6. Core grid configuration**

The firmware programs the participating Tensix cores with their runtime arguments: L1 buffer addresses, tile dimensions, iteration counts, and NoC destination addresses. Each core's five RISC processors are configured before the kernel is allowed to execute. With `TT_METAL_DEVICE_PROFILER_DISPATCH_CORES=1`, this stage becomes visible in the CSV via dispatch core timing entries.

**7. NoC route setup and DRAM descriptor writes**

For ops that read from DRAM (most matmuls and large elementwise ops), the NCRISC processors on each participating core must have their NoC DMA descriptors written before data movement can begin. The firmware writes these descriptors as part of core grid configuration, consuming additional device-side time between enqueue completion and the first kernel instruction.

**8. Kernel launch**

After all cores are configured and NoC descriptors are written, the firmware signals each core's BRISC processor to begin executing the compute kernel. At this point, `DEVICE KERNEL DURATION [ns]` begins counting.

---

## Typical Overhead Magnitudes

The total `host_dispatch_time` for a TTNN op call depends primarily on whether the program cache has a valid entry.

### Cache-hit call (~5–50 µs)

For a cache-hit call on a steady-state workload (same shapes, same data format, same device config as a previous call in the same session):

| Stage | Typical cost |
|---|---|
| Argument validation and layout resolution | ~1–5 µs |
| Program cache lookup (hit) | ~1–3 µs |
| Command buffer serialization | ~1–3 µs |
| MMIO write to device | ~1–5 µs |
| **Total host dispatch (cache hit)** | **~5–50 µs** |

The spread within the cache-hit range comes from op complexity (argument validation for a multi-input op like `ttnn.matmul` is heavier than for `ttnn.add`), memory configuration resolution overhead, and PCIe MMIO latency variability.

### Cache-miss call (~100–500 µs)

For a first call with a new shape or data format combination:

| Stage | Typical cost |
|---|---|
| Argument validation and layout resolution | ~1–5 µs |
| Program cache lookup (miss) | <1 µs |
| Kernel binary lookup / compilation | ~80–450 µs |
| Program object instantiation and runtime arg population | ~5–20 µs |
| Command buffer serialization | ~1–3 µs |
| MMIO write to device | ~1–5 µs |
| **Total host dispatch (cache miss)** | **~100–500 µs** |

> **Tip:** `ttnn.enable_program_cache()` must be called before the first op invocation in a session to activate caching. Without it, every call is a cache miss. This is a common source of unexpectedly high Tracy zone durations in development and test workloads.

---

## Why This Matters for Small Ops

For large compute ops, `device_kernel_time` is hundreds of microseconds or longer, so a 10–20 µs `host_dispatch_time` is negligible. For small ops — decode-regime operations on single-sequence or small-batch tensors — the ratio inverts dramatically.

Consider a `ttnn.add` on a [32, 4096] tensor (a common attention bias addition in a transformer decode step):

- **Device kernel time:** ~0.3 µs. A 32×4096 tensor in 32×32 tile layout contains `(32/32) × (4096/32) = 1 × 128 = 128 tiles`, but `ttnn.add` is an elementwise op with very low compute per tile (a single addition per element), so the device kernel completes in a fraction of a microsecond despite the 128-tile input.
- **Host dispatch time (cache hit):** ~6 µs. Argument validation, cache lookup, command buffer write, and MMIO.
- **Dispatch-to-kernel ratio:** ~20×.

The op's total latency is dominated entirely by the host, not the device. Optimizing the kernel itself — reducing FPU stalls, adjusting math fidelity, increasing core count — would have no measurable effect on end-to-end throughput. The only effective remediation is to reduce the number of dispatch calls (by fusing ops) or to eliminate dispatch overhead entirely by using trace capture.

The full latency equation makes this explicit:

```
total_op_latency = host_dispatch_time + device_kernel_time + sync_overhead
                 ≈       6 µs          +       0.3 µs        +   ~0.5 µs
                 ≈       6.8 µs
```

More than 88% of the op's observed latency is `host_dispatch_time`.

> **Warning:** It is a common diagnostic mistake to look only at `DEVICE KERNEL DURATION [ns]` when a workload is slower than expected, see small kernel durations, and conclude the kernels are fast and therefore something else must be wrong. In a dispatch-bound workload, those small kernel durations are exactly the problem — the device is barely running relative to how much time the host is spending on dispatch. Always check Tracy data alongside the CSV.

---

**Next:** [`measuring_dispatch_vs_kernel.md`](./measuring_dispatch_vs_kernel.md)
