# What Metal Trace Records

Metal Trace is the TT-Metal subsystem that allows a sequence of device operations to be captured once and replayed many times at reduced host overhead. Understanding exactly what ends up in the trace command buffer — and what does not — is the foundation for reasoning about trace compatibility of any given operation.

---

## What the Capture Run Records

When `ttnn.begin_trace_capture` is called on a `MeshCommandQueue`, the runtime begins intercepting every command enqueued to the device during that capture run. The command buffer accumulates three categories of entries:

**Kernel dispatches.** Each TTNN op that runs on-device compiles to one or more kernels. During capture, the dispatch descriptor for each kernel — including the concrete device memory addresses of its input and output buffers — is written verbatim into the command buffer.

**DMA transfer descriptors.** Operations that move data between host and device, or between device buffers, produce DMA descriptors. Each descriptor encodes the source address, destination address, transfer size, and the command queue channel to use.

**Semaphore operations.** Synchronisation primitives (acquire, release, wait) used to coordinate producer/consumer ordering between kernels or between the host and device are recorded as semaphore commands with their concrete semaphore addresses.

All addresses encoded in the command buffer are physical device DRAM addresses as they exist at the moment of capture. The command buffer is an opaque binary artifact; once sealed by `ttnn.end_trace_capture`, it cannot be patched.

---

## What Replay Does

`ttnn.execute_trace` re-issues the command buffer to the device verbatim. From the device's perspective, replay is indistinguishable from a fresh execution: the same kernel binaries run, the same DMA transfers occur, the same semaphores are signalled. From the host's perspective, replay is dramatically cheaper because:

- No Python forward pass re-executes.
- No tensor shape inference occurs.
- No host-side buffer allocation occurs.
- No kernel compilation or dispatch argument re-computation occurs.

The host simply submits the pre-built command buffer and waits for a completion signal.

---

## What Trace Does NOT Record

The following categories of activity are invisible to the trace recording mechanism:

| Category | Example | Why invisible |
|---|---|---|
| Python control flow | `if x is None: ...` | Executes on host CPU, never touches the device command queue |
| Tensor shape recomputation | `tensor.shape`, `torch.broadcast_shapes` | Host-side Python/C++ computation |
| Host-side buffer allocation | `torch.zeros(...)`, `ttnn.from_torch(...)` | Allocates host or device DRAM via host CPU; does not enqueue a device command |
| Any op that runs on the host CPU | `ttnn.from_torch`, `tensor.numpy()` | Produces a side effect (a new device buffer) that the command buffer does not know about |

> **Key Finding:** The trace does not record *what created* the buffers it references. It records only the addresses of those buffers. This means an op can look like a TTNN op from the Python level and still be entirely invisible to the trace if its primary side effect is a host-side allocation.

---

## The Buffer Address Stability Invariant

Let $B_c$ denote the set of device buffer base addresses referenced by the command buffer at capture time, and let $B_r^{(i)}$ denote the set of device buffer base addresses that actually exist when the $i$-th replay begins. The invariant required for correct replay is:

$$B_c \subseteq B_r^{(i)} \quad \text{for all } i \geq 1$$

That is, every address baked into the command buffer must refer to a live, correctly-populated device buffer on every replay.

A violation occurs when a new device buffer is allocated *inside* the capture bracket. Such a buffer:

1. Did not exist before the capture run, so its address was not recorded by any prior command.
2. Was allocated by a host operation, so no "allocate buffer at address $a$" command was recorded in the command buffer.
3. Will not be re-allocated during replay, because replay re-issues only recorded device commands.

When replay issues a kernel dispatch or DMA transfer that references the address of a buffer allocated inside the capture bracket, one of two failure modes results:

- **Silent data corruption.** The address happens to be occupied by a different, unrelated buffer that was allocated before the capture. The kernel reads or writes that buffer's contents silently.
- **Device crash.** The address is unmapped or no longer valid, causing the device to fault.

> **Trace Invariant:** Every device buffer whose address appears in the command buffer must be allocated *before* `ttnn.begin_trace_capture` is called and must remain alive and at the same address for the lifetime of the trace. No new device buffer allocations may occur inside the capture bracket.

---

**Next:** [`from_torch_is_a_host_operation.md`](./from_torch_is_a_host_operation.md)
