# Host Dispatch Path

This file walks through the complete sequence of work the TTNN runtime performs on the host CPU between the moment your Python code calls an op — for example, `ttnn.matmul(a, b)` — and the moment the resulting kernel begins executing on the Tenstorrent device. By the end you will understand what each phase costs, why dispatch overhead is a measurable and non-trivial fraction of total op latency, and how to think about that overhead as a target for optimization.

---

## Overview of the Four Phases

Every TTNN op call passes through four sequential phases on the host before the device does any work:

1. **Argument validation** — verify that the inputs are legal for this op
2. **Kernel selection** — choose the correct compiled kernel binary
3. **Command encoding** — serialize the kernel invocation into the CQ binary format
4. **CQ submission** — write the encoded command into the command queue

These phases run synchronously on the calling host thread. The call to `ttnn.matmul(a, b)` does not return until all four phases are complete and the command has been placed in the CQ. (In async op mode, covered in Chapter 2, all four phases execute on a background dispatch thread — the Python caller returns before any of the phases begin.)

---

## Phase 1: Argument Validation

When `ttnn.matmul(a, b)` is called, the runtime immediately checks that the provided tensors are valid inputs for a matrix multiplication:

- Both tensors must be `ttnn.Tensor` objects allocated on the device (or explicitly moved to it).
- The inner dimensions must be compatible: the number of columns in `a` must equal the number of rows in `b`.
- The data types must be supported by a matmul kernel — for example, `ttnn.bfloat16` tiles are accepted; a raw Python float list is not.
- The memory layout (row-major vs. tiled) and tile dimensions must match what the selected CQ and program configuration expect.

Validation failures raise a Python exception immediately, before any device interaction occurs. For correct inputs, validation typically takes 5–15 us on current hardware generations, depending on the number of checks performed for the specific op.

> **Note:** Validation time scales with the number of argument checks, not with tensor size. A `ttnn.matmul` on a 4096x4096 tensor pays roughly the same validation cost as one on a 64x64 tensor, because the checks are structural, not data-dependent.

---

## Phase 2: Kernel Selection

After validation, the runtime selects which compiled kernel binary to dispatch. TTNN ships with a set of pre-compiled kernels for each op, parameterized along several axes:

- **Data type** — e.g., `bfloat16`, `float32`, `uint16`
- **Memory location** — L1 SRAM vs. DRAM (different bandwidth characteristics require different kernel strategies)
- **Tile shape** — the device operates on 32x32 tiles; non-standard blocked layouts may require a different kernel variant
- **Program config** — op-specific configuration objects (e.g., `ttnn.MatmulMultiCoreReuseProgramConfig`) that control core grid utilization and loop unroll factors

Kernel selection is effectively a lookup into a registry of compiled programs indexed by these parameters. On a cache hit — where the same op has been called with the same parameter combination before in this process — selection is a hash map lookup taking roughly 1–3 us. On a first call (cold path), the runtime may also initialize device-side program structures, which can add 50–200 us.

> **Note:** The cold-path cost is paid once per unique (op, config) combination per device session. After the first call, subsequent calls with identical configurations land on the warm path and pay only the hash map lookup cost.

<details>
<summary>Example: specifying a program config to influence kernel selection</summary>

```python
import ttnn

# A MatmulMultiCoreReuseProgramConfig directs TTNN to use a specific
# tiled matmul kernel that tiles across a given core grid.
program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=(8, 8),
    in0_block_w=2,
    out_subblock_h=1,
    out_subblock_w=4,
    per_core_M=4,
    per_core_N=4,
)

output = ttnn.matmul(
    a,
    b,
    program_config=program_config,
    dtype=ttnn.bfloat16,
    math_fidelity=ttnn.MathFidelity.HiFi2,
)
```

Providing an explicit `program_config` removes ambiguity from kernel selection, which keeps selection time consistently on the warm path for all but the very first call.
</details>

---

## Phase 3: Command Encoding

Command encoding is the most compute-intensive phase of dispatch. The runtime takes the selected kernel and all its parameters and serializes them into the binary wire format that the device firmware reads from the command queue.

A single encoded command includes:

- The kernel binary identifier (an index into the device's program cache)
- The device DRAM or L1 buffer addresses for each input and output tensor
- The tile dimensions, strides, and counts for each buffer
- The runtime arguments that the kernel reads from L1 (loop bounds, scale factors, etc.)
- The compute core grid assignment and semaphore configuration

For a `ttnn.matmul` call on a moderately sized tensor, encoding typically produces a command that is several kilobytes in size and takes 10–40 us to construct on the host CPU. More complex ops with larger core grids or more runtime arguments take proportionally longer.

> **Warning:** The encoding cost does not scale with tensor data size — it scales with the structural complexity of the kernel invocation (number of cores, number of runtime arguments, number of buffer bindings). A distributed matmul across a 64-core grid produces a larger command than a single-core elementwise add, regardless of tensor dimensions.

This is the phase that trace eliminates on replay. When a trace is replayed, the device already has the pre-encoded command buffer from the capture run; phases 1–3 are entirely skipped. Chapter 3 explains this in full.

---

## Phase 4: CQ Submission

After encoding, the runtime writes the command into the command queue (CQ). The CQ is a circular buffer in host-mapped memory that the device firmware polls continuously. Submission consists of:

1. Writing the encoded command bytes into the next available CQ slot.
2. Advancing the host-side write pointer.
3. Optionally issuing a memory fence to ensure the write is visible to device firmware before the pointer update.

Submission is typically the fastest phase — 1–5 us for a single command — because it is a memory write with no computation. The latency is dominated by memory bandwidth and cache coherence overhead between the host CPU and the device's view of that memory region.

Once the write pointer is advanced, the device firmware can begin consuming the command on the next polling cycle. The host call returns at this point (in synchronous mode) or before this point (in async mode, as covered in Chapter 2). The host does not wait for the device to actually execute the kernel.

---

## Putting the Phases Together: Where Time Goes

The table below gives representative costs for a warm-path `ttnn.matmul` dispatch on current Tenstorrent hardware generations. These are host-side times only — kernel execution on the device happens concurrently and is not included.

| Phase | Typical cost (warm path) |
|---|---|
| Argument validation | 5–15 us |
| Kernel selection (warm) | 1–3 us |
| Command encoding | 10–40 us |
| CQ submission | 1–5 us |
| **Total dispatch overhead** | **~17–63 us per op** |

For a decode loop with 32 ops per step, this sums to roughly 544 us–2 ms (2,016 us) of host dispatch overhead per step, even before considering Python interpreter overhead. On hardware where a single decode step takes 5 ms (5,000 us) end-to-end, dispatch overhead can represent 10–40% of total latency — a significant fraction worth optimizing.

> **Note:** The figures above are illustrative. Actual costs depend on the specific op, the core grid size, the number of runtime arguments, and the host CPU model. Always measure your own workload with TTNN's profiler or Tracy (covered in Chapter 5) before estimating potential speedup.

---

## Dispatch Overhead as a Measurable Cost

Dispatch overhead shows up distinctly in profiler output. In a Tracy trace or a TTNN profiler CSV, you will see host-side spans labeled with the op name and phase. The device execution span for the same op starts some time after the dispatch span ends — the gap between them is the CQ transit time (how long the command waited in the queue before the device picked it up).

When the device is faster than the host at dispatching commands — that is, the device finishes each kernel before the host finishes encoding the next command — the device sits idle waiting for work. This is the regime where trace provides the most benefit: by eliminating encoding on the host, trace collapses the dispatch time and eliminates that idle gap.

When the host is faster than the device, the CQ builds up a backlog of commands and the host idle-spins waiting for queue space. In this regime, dispatch overhead is already hidden by the device's slower execution, and trace provides less benefit. Chapter 4 covers how to identify which regime your workload is in.

---

**Next:** [`command_queues.md`](./command_queues.md)
