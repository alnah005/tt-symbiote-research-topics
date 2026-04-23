# synchronize_device Latency Model

This file decomposes the latency of a single `ttnn.synchronize_device(mesh_device)` call at decode batch=1 on T3K into its constituent components and provides per-component estimates. By the end of this file you will be able to reason about what portion of the call's cost is fundamental PCIe overhead (unavoidable regardless of device utilization) and what portion depends on how much device-side work is still in flight when the call reaches the front of the host's execution.

---

## What synchronize_device Does, Precisely

When Python calls `ttnn.synchronize_device(mesh_device)`, the TTNN runtime:

1. Enqueues a Finish token to CQ0. This is a lightweight device-side command that the device's dispatch engine processes after all previously enqueued kernels complete.
2. Spins or sleeps on the host side, polling the device's completion register or an L1-mapped flag, until the Finish token has been processed by the device.
3. Returns control to Python only after the device acknowledgment arrives via PCIe.

Step 1 is fast: enqueueing a single Finish token costs a few microseconds of host-side DMA or MMIO write. Steps 2 and 3 together constitute the observable latency of the call: the host does no useful work while waiting.

---

## Latency Components

### Component 1 — PCIe Round-Trip (host to device, device to host)

The Finish token written to the device's dispatch queue must cross the PCIe bus. The device must process the token, update a completion register or flag, and that update must be read back (or DMA'd back) to the host. On Wormhole hardware with a PCIe Gen4 ×16 link on T3K, the raw round-trip latency for a small command is approximately:

- **Host-to-device write latency:** ~2–5 µs (MMIO or DMA, depending on path)
- **Device processing of Finish token:** ~1–3 µs (single kernel dispatch cycle overhead)
- **Device-to-host notification:** ~5–20 µs (polling interval + PCIe read-back)
- **Total PCIe round-trip estimate:** ~10–30 µs

> **Note:** These values are estimates derived from published Wormhole PCIe latency characteristics. The implementing engineer must measure the actual round-trip latency on their specific T3K unit using the procedure in [`measuring_the_cost.md`](./measuring_the_cost.md) and replace these estimates with confirmed values.

### Component 2 — Kernel Completion Time (if in-flight work remains)

If the all_gather operation submitted immediately before `synchronize_device` is still executing on the device when the Finish token is processed, the device will not acknowledge the Finish until the all_gather completes. In this case:

- **Device-side all_gather execution time at batch=1 on T3K:** typically 20–80 µs for a single-layer gather across 8 devices on the ring, depending on tensor size and link bandwidth
- **Overlap case:** if the host submits `synchronize_device` long after the all_gather completes (due to Python scheduling jitter or intervening host-side logic), the kernel completion component is zero; only PCIe round-trip remains

For `_maybe_all_gather` as written, the pattern is:

```python
output = ttnn.all_gather(...)      # synchronous — enqueued to CQ0
ttnn.synchronize_device(mesh_device)  # enqueued Finish, then host blocks
```

Because `ttnn.all_gather` (synchronous form) is internally a blocking enqueue — it submits the all_gather dispatch commands and returns after enqueueing them, but before device execution completes — the device may or may not have finished the all_gather by the time the Finish token is received by the device. At batch=1, the all_gather is typically short enough that the device completes it before or shortly after the host submits the Finish, leaving only PCIe round-trip as the observable latency.

> **Key finding:** At decode batch=1, the all_gather execution time (Component 2) is typically smaller than the PCIe round-trip overhead (Component 1) for short hidden-dimension slices. The PCIe round-trip dominates the observable `synchronize_device` cost.

### Component 3 — Host Scheduling Jitter

Python's GIL, OS scheduler preemption, and the overhead of Python function dispatch add non-deterministic latency around each TTNN API call:

- **Estimate:** 0–10 µs under typical steady-state conditions with CPU affinity set; can spike to hundreds of microseconds (up to ~500 µs) if the OS preempts the Python thread at an inopportune moment
- **Mitigation:** Pin the Python process to a dedicated CPU core (CPU affinity) and minimize background OS activity during profiling

---

## Total Estimate at Decode Batch=1 on T3K

Summing the components under the common case (all_gather completes before or coincident with Finish acknowledgment):

| Component | Estimate |
|---|---|
| PCIe round-trip (dominant) | 10–30 µs |
| Kernel completion overlap | 0–20 µs (often zero at batch=1) |
| Host scheduling jitter | 0–500 µs (typical < 10 µs; can spike to hundreds of µs under OS preemption) |
| **Total per call** | **10–550 µs** (typical 20–60 µs; rare spikes to ~500 µs under OS preemption) |

> **Note:** The typical range (20–60 µs) reflects steady-state inference with a CPU-pinned Python thread. The upper tail (up to ~500 µs) reflects OS thread preemption, which can be observed in production even without explicit `sleep` calls. The 0.1–0.5 ms range cited in the chapter overview corresponds to this practical distribution: the lower bound represents the uncontested case (PCIe round-trip only), and the upper bound represents a moderate preemption event. The implementing engineer should use the measurement procedure in [`measuring_the_cost.md`](./measuring_the_cost.md) to obtain actual distribution values on their target system.

---

## How Context Changes the Cost

### Case A — All_gather Already Complete

If `ttnn.all_gather` has been on the device queue long enough to complete before the host reaches the `synchronize_device` Python call, the Finish token encounters no in-flight work. The device processes the Finish immediately, and only the PCIe round-trip is observable. This is the best-case scenario and the most common at batch=1.

### Case B — All_gather Still Running

If the host submits the Finish token very quickly after the all_gather dispatch (which can happen if the Python overhead between them is negligible), the device receives the Finish while the all_gather is still executing. The device queues the Finish token and processes it only after the all_gather completes. In this case, `synchronize_device` latency = remaining all_gather execution time + PCIe round-trip.

### Case C — Multi-Device All_gather with Ring Latency

On T3K, a ring all_gather across 8 devices takes longer than a single-device operation because the collective must traverse 7 hops around the ring. At batch=1 with small hidden-dimension slices, the ring traversal time can be 20–80 µs. If `synchronize_device` is called before the ring completes, the wait includes the remaining ring traversal time.

---

## Implication for the Removal Decision

The latency model confirms that `ttnn.synchronize_device()` adds between 10 µs (best case) and several hundred microseconds (worst case with long all_gather + scheduling jitter) per call, with no benefit to correctness or ordering (as established in [Chapter 3](../ch3_root_cause_analysis/verdict_is_it_removable.md)). This makes it a pure overhead cost. The next file, [`measuring_the_cost.md`](./measuring_the_cost.md), describes how to measure the actual distribution on a target system.
