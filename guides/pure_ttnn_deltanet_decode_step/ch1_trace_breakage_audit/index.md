# Chapter 1 — Why the Current Implementation Breaks Trace

This chapter gives the reader a precise, operation-level map of `TTNNQwen3LinearAttention.forward` as it executes at decode time (B=1, T=1). Before any implementation changes are proposed, the reader needs to know exactly which operations are on-device TTNN (and therefore Metal Trace-compatible) and which operations are host-side (and therefore break the static command stream that Metal Trace requires). This chapter delivers that map in three files: a step-by-step walkthrough of the forward pass, a consolidated summary table of every host-crossing call, and an analysis of how the DeltaNet recurrent state and conv state are currently stored between decode steps.

---

## Chapter Goal

The goal of this chapter is to characterize the trace incompatibility problem with precision, so that subsequent chapters can address each incompatibility individually and in the right order. The deliverable is a table of every host-crossing call in `TTNNQwen3LinearAttention.forward`, specifying:

- The operation name and the source file and line where it is called
- The tensors read from the Wormhole device (triggering a device-to-host transfer or synchronization)
- The tensors written back to the device (triggering a host-to-device transfer or dynamic buffer allocation)
- The trace-break mechanism: one of `HOST_KERNEL_LAUNCH`, `TO_TORCH`, `FROM_TORCH`, or `PYTHON_BRANCH`

This classification directly determines the implementation strategy in Chapters 2, 3, and 4.

---

## Learning Objectives

After completing this chapter, the reader will be able to:

1. **Trace the execution path** of `TTNNQwen3LinearAttention.forward` at decode time and identify which of the six logical steps is on-device vs. host-side.

2. **Classify every host crossing** using the four trace-break mechanism tags (`HOST_KERNEL_LAUNCH`, `TO_TORCH`, `FROM_TORCH`, `PYTHON_BRANCH`) and explain why each one is incompatible with `ttnn.begin_trace_capture` / `ttnn.execute_trace`.

3. **State the root cause** of the host crossing at step 4 (the recurrent gated delta rule step): the DeltaNet state matrix S and the conv state are stored as plain PyTorch tensors between decode calls, forcing a device-to-host transfer on every decode step.

4. **Identify the two operations that are already trace-compatible** (input projections and output projection) and require no changes.

5. **Prioritize the four host-crossing operations** by decode latency impact and implementation complexity, so that the most impactful fix is attempted first.

---

## Files in This Chapter

| File | Topic |
|---|---|
| [`forward_pass_walkthrough.md`](./forward_pass_walkthrough.md) | Step-by-step walkthrough of `TTNNQwen3LinearAttention.forward` at decode time, annotating each step as on-device or host-side |
| [`host_crossing_summary_table.md`](./host_crossing_summary_table.md) | Consolidated table of all host-crossing calls with trace-break mechanism classification and fix priority ranking |
| [`device_state_persistence.md`](./device_state_persistence.md) | How the DeltaNet recurrent state S and conv state are currently managed between decode steps, and what must change |

---

## Reading Order

Read the three files in the following order:

1. **[`forward_pass_walkthrough.md`](./forward_pass_walkthrough.md)** — Establishes the full picture of the forward pass. This is the prerequisite for the summary table and the state persistence analysis, both of which reference specific steps from the walkthrough.

2. **[`host_crossing_summary_table.md`](./host_crossing_summary_table.md)** — Consolidates the walkthrough's findings into a single reference table. Read this after the walkthrough so that every row in the table has a corresponding narrative explanation.

3. **[`device_state_persistence.md`](./device_state_persistence.md)** — Zooms in on the root cause of the step 4 host crossing: state storage. This file explains what must change in the cache object before any of the kernel changes in Chapters 2–4 can take effect.

---

## Relationship to Later Chapters

- Chapter 2 (`ch2_ttnn_decomposition/`) takes the host-crossing classification from this chapter as its starting point and derives a complete TTNN-native decomposition of the recurrent delta rule step (step 4). The state tensor shapes documented in `device_state_persistence.md` drive the memory config specification in Chapter 2.
- Chapter 3 (`ch3_auxiliary_ops/`) addresses the two auxiliary host crossings identified here: the causal conv1d update (step 2) and the gated RMSNorm (step 5).
- Chapter 6 (`ch6_latency_and_accuracy/`) uses the tensor sizes from `host_crossing_summary_table.md` to estimate PCIe transfer latencies for the current host fallback.
- Chapter 7 (`ch7_implementation_roadmap/`) uses the priority ranking from `host_crossing_summary_table.md` to sequence the implementation tasks.

---

**Next:** [`forward_pass_walkthrough.md`](./forward_pass_walkthrough.md)
