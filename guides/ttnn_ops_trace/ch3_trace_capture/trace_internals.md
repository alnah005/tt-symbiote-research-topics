# Trace Internals

This file explains the internal mechanics that make trace replay faster than live dispatch: how the captured command buffer is stored on the device, exactly which parts of the four-phase dispatch path are eliminated on replay, how buffer aliasing works, and the hard address-fixity constraint that buffer aliasing imposes on input and output tensors. By the end you will understand why replay is fast, why it is correct, and what you must not do to the tensors that participated in a capture if you want replay to remain valid.

---

## How the Captured Trace Is Stored

When you call `ttnn.begin_trace_capture` and then dispatch ops, the TTNN runtime writes the binary-encoded commands into a dedicated trace recording buffer in device DRAM. This buffer is separate from the normal CQ (the circular command buffer described in `command_queues.md`). It is a linear, append-only buffer: each new encoded command is appended to the end of the current recording in sequence.

When `ttnn.end_trace_capture` is called, the runtime:

1. Finalizes the recording buffer — no further appends are possible.
2. Pins the buffer at its current DRAM address for the lifetime of the trace. The buffer's address will not change, and the allocator will not reclaim that DRAM region until `ttnn.release_trace` is called.
3. Assigns the buffer a `trace_id` integer handle, which the runtime uses as a key into a per-device trace registry.

The stored buffer is structurally identical to what would be written into the CQ during live dispatch — it is the output of phases 1–3 (argument validation, kernel selection, and command encoding) already serialized into the device's wire format. Nothing is further interpreted or re-encoded at replay time. The device firmware can read the buffer bytes and begin executing without any host involvement in re-encoding.

```
Device DRAM layout (schematic — not to scale)
──────────────────────────────────────────────────────────────────────
│  Weight buffers (model parameters)        │ pinned for model life  │
├──────────────────────────────────────────────────────────────────────
│  KV-cache buffers (attention state)       │ read/write each step   │
├──────────────────────────────────────────────────────────────────────
│  Input tensor buffer  (e.g. addr 0x1A000) │ written before replay  │
├──────────────────────────────────────────────────────────────────────
│  Output tensor buffer (e.g. addr 0x2F000) │ written by replay      │
├──────────────────────────────────────────────────────────────────────
│  Trace recording buffer (trace_id=3)      │ pinned since end_trace │
│    cmd[0]: matmul  in=0x1A000 out=0x3C000 │                        │
│    cmd[1]: softmax in=0x3C000 out=0x4D000 │                        │
│    ...                                    │                        │
│    cmd[N]: add     in=0x5E000 out=0x2F000 │                        │
└──────────────────────────────────────────────────────────────────────
```

The address values baked into each command entry (`0x1A000`, `0x3C000`, etc.) are the exact DRAM addresses that were live during capture. They are not symbolic names or offsets — they are physical hardware addresses. This is the source of the address-fixity constraint.

---

## Why Replay Bypasses Host Dispatch Overhead

Live dispatch for a single op proceeds through four phases on the host. As established in Chapter 1 ([`host_dispatch_path.md`](../ch1_dispatch_fundamentals/host_dispatch_path.md)), each live op dispatch costs 17–63 us across four phases (5–15 us validation, 1–3 us kernel selection, 10–40 us encoding, 1–5 us CQ submission).

During replay, none of phases 1–3 occur. The replay flow is:

1. `ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)` is called on the Python thread (or dispatch thread, in async mode).
2. The runtime submits a single "execute trace" command to the CQ. This command is a few bytes: it contains only the `trace_id` and the byte length of the recording.
3. The device firmware reads the "execute trace" command from the CQ, looks up the trace buffer's DRAM address in the device's trace registry, and begins streaming command bytes from that buffer directly into the execution pipeline.
4. Each recorded command is executed exactly as if it had arrived through the CQ from a live dispatch — the same kernel binary is invoked, the same DRAM addresses are accessed, the same computation occurs.

The host does not re-encode anything. It does not re-validate arguments. It does not re-select kernels. Its total contribution to replay overhead is the cost of writing the "execute trace" command into the CQ — a few bytes, comparable to CQ submission phase (1–5 us) — plus any async dispatch bookkeeping in the runtime (typically under 5 us).

```
Live dispatch overhead per step (32 ops example)
────────────────────────────────────────────────
  Phase 1–3 per op:  ~16–58 us × 32 ops = ~512–1,856 us
  Phase 4 per op:    ~1–5 us  × 32 ops = ~32–160 us
  Total host time:   ~544 us–2 ms (2,016 us)

Trace replay overhead per step (same 32 ops)
────────────────────────────────────────────────
  Single "execute trace" CQ write:  ~5–10 us
  Runtime dispatch bookkeeping:     ~2–5 us
  Total host time:                  ~7–15 us

Reduction: roughly 36–288× less host overhead per step.
```

The device execution time (the kernels themselves) is identical between live dispatch and replay — the same kernels run with the same data. Trace does not speed up compute; it only removes host encoding from the critical path.

> **Note:** The per-step speedup from trace depends on how much of your step's latency is currently attributable to host dispatch overhead. If your decode step already takes 50 ms (50,000 us) end-to-end and dispatch overhead is 1 ms (1,000 us), trace removes that 1 ms and the total drops to ~49 ms — a 2% improvement. If your step takes 2 ms (2,000 us) and dispatch overhead is 1.5 ms (1,500 us), the improvement is ~75%. Chapter 5 provides the methodology for measuring your specific workload before committing to trace.

---

## Buffer Aliasing

Buffer aliasing is the mechanism by which replay reuses the same device memory regions that were live during capture. It is not a new allocation or a copy — it is a guarantee that the same physical DRAM pages back both the capture-time tensors and the replay-time tensors.

During capture, when the runtime encodes a command for op N, it writes the literal device DRAM address of op N's input and output buffers into the recorded command entry. For example:

```
cmd[5] (matmul):
  input_a_addr  = 0x1A000  (address of input_tensor buffer at capture time)
  input_b_addr  = 0x4C000  (address of weight_tensor buffer at capture time)
  output_addr   = 0x3C000  (address of intermediate buffer at capture time)
  grid_config   = (8, 8)
  per_core_M    = 4
  per_core_N    = 4
  ...
```

At replay time, when the device executes `cmd[5]`, it issues reads and writes to exactly those addresses: `0x1A000`, `0x4C000`, `0x3C000`. The device has no way to substitute different addresses — the command bytes are fixed. For replay to produce correct results, the data at those addresses at replay time must be the data you intend the op to process.

For weight tensors — the model parameters — this is automatically satisfied: weights are loaded once, never moved, and their DRAM addresses never change. The replay will read the same weight bytes at the same addresses, which is correct.

For input tensors — the data you feed into the model each step — this is your responsibility. The input tensor object you use during capture must be the same tensor object you write new input data into before each replay call. Specifically:

- The same `ttnn.Tensor` Python object (which holds the DRAM address of its buffer) must be used for both capture and replay.
- New input data must be written into that tensor's existing buffer, not by replacing the tensor object with a newly allocated one.
- The write must complete (be visible on the device) before `ttnn.execute_trace` is submitted for that replay step.

For output tensors — the tensor that receives the step's result — the same logic applies in reverse. The output tensor object allocated during capture (and returned by the captured model's forward method) is the tensor whose buffer address is encoded in the last command of the trace. After replay, you read from that exact same tensor object.

---

## Implication: Input and Output Tensors Must Live at the Same Addresses Used During Capture

This is the most operationally important consequence of buffer aliasing, so it is worth stating precisely:

> The physical device DRAM addresses of input tensors, output tensors, and all intermediate buffers that appear in the trace were fixed at the moment `ttnn.end_trace_capture` was called. From that moment until `ttnn.release_trace` is called, those buffers must remain at those addresses.

What "remain at those addresses" means in practice:

**Do not deallocate and reallocate traced tensors.** If you call `del input_tensor` or allow the Python object to go out of scope and be garbage-collected, the TTNN runtime may release the underlying DRAM buffer. A subsequent allocation may or may not land at the same address. If it does not, the replay will read from whatever data happens to be at the old address — which is incorrect. If it does, correctness is coincidental and fragile.

**Do not pass a different tensor object to the replay.** `ttnn.execute_trace` takes no input tensors as arguments — it always operates on the buffers that were bound at capture time. There is no mechanism to "redirect" a traced buffer to a different address at replay time. If you want to feed new data, you write it into the existing buffer; you cannot swap in a different buffer.

**Do not resize or reformat traced tensors.** Calling any op that would change a tensor's shape, layout, or dtype after capture will, if it causes a new buffer allocation (most shape changes do), produce a new buffer at a different DRAM address. The trace still points to the old address.

**Intermediate buffers are also pinned.** Not just inputs and outputs — every temporary buffer allocated by any intermediate op during capture is also encoded by address in the trace. The runtime manages these intermediate buffers as part of trace memory and keeps them pinned. You do not interact with them directly, but you should not attempt to manually free TTNN allocations that were live during capture.

> **Warning:** Deallocating a tensor that participated in a capture while a trace referencing it is still live is a use-after-free error at the device level. The trace buffer contains the freed buffer's DRAM address, and replay will issue device reads and writes to that address regardless of what the allocator has put there. This will silently produce incorrect outputs unless the device happens to fault on the access.

---

## Intermediate Buffers and the Trace's Memory Footprint

Every intermediate tensor allocated during the capture run — the outputs of ops that are inputs to subsequent ops but are not the final output — is also pinned in device DRAM for the lifetime of the trace. This is because those intermediate buffer addresses are encoded in the trace commands.

For a decode step with N ops and I intermediate tensors, the trace's memory footprint is:

- The command buffer itself: proportional to N (typically a few kilobytes to a few hundred kilobytes).
- The intermediate tensor buffers: the sum of sizes of all tensors allocated between `begin_trace_capture` and `end_trace_capture` that were not already allocated before capture. For a typical transformer layer decode step, this can range from tens of megabytes to several hundred megabytes depending on model width and intermediate activation sizes.

The input and output tensors that you allocated before `begin_trace_capture` are not part of this footprint — they were already allocated and remain under your management. Only newly allocated buffers within the capture window are pinned by the trace.

> **Note:** If device DRAM is tight, trace can be a significant contributor to memory pressure. Each distinct captured trace (for example, one for a standard decode step and one for a speculative decoding verification step) pins its own intermediate buffer set. Measure total trace memory footprint with TTNN's memory profiler before deploying traces in memory-constrained configurations.

---

**Next:** [`trace_constraints.md`](./trace_constraints.md)
