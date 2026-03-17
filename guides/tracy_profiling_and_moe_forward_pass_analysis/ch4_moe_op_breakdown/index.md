# Chapter 4 — MoE Forward Pass Op Breakdown

## Overview

This chapter documents the complete sequence of TTNN ops that execute during one MoE layer
forward pass. It covers the three phases — dispatch, expert matmul, combine — and provides
expected latency budgets for both prefill and decode regimes on Wormhole B0 and T3K hardware.

The chapter serves two purposes: (1) building a mental model of what the hardware is doing
during each microsecond of the MoE forward pass, and (2) providing a reference table you can
hold next to a Tracy trace or device profiler CSV to verify that every expected op is present
and within its expected latency range.

---

## Prerequisites

This chapter requires:

- **Chapter 1** — Tracy two-process model; distinction between Tracy CPU zones and device profiler
  cycle counts. The terminology "Tracy zone name" used throughout this chapter is defined there.
- **Chapter 2** — Build flags and capture workflow; how to produce `ops_perf_results_<timestamp>.csv`.
- **Chapter 3** — TTNN op naming in the device profiler CSV; `DEVICE KERNEL DURATION [ns]` column;
  how to run `process_ops_logs.py` and read the output.

If you only need the op sequence reference table (`full_op_sequence_reference.md`) without running
any profiling, Chapters 1 and 2 are not strictly required.

---

## Learning Objectives

After completing this chapter you will be able to:

1. **Recite the op sequence** for a MoE forward pass in order from router input to MoE output,
   naming the TTNN op for each step.
2. **State the expected latency** for each phase (dispatch, expert matmul, combine) at seq_len=1024
   for Qwen 235B MoE dimensions on a single Wormhole B0 device.
3. **Identify the T3K-specific ops** — the all-to-all CCL collective in the dispatch phase and the
   reduce-scatter or all-reduce in the combine phase — by their zone names in a Tracy trace.
4. **Explain why expert matmul dominates** total MoE latency in prefill and why this relationship
   inverts in decode.
5. **Use `full_op_sequence_reference.md`** as a ground-truth checklist when reading a device profiler
   CSV or Tracy trace for any of three major MoE architectures.

---

## MoE Forward Pass: Linear Timeline

The following sequence represents one complete MoE layer forward pass from the perspective of
op dispatch on a single chip (T3K CCL ops noted where they appear):

```
Router input: [seq_len, d_model]
      │
      ▼
[1]  ttnn.linear          — router projection → [seq_len, num_experts]
[2]  ttnn.softmax         — router probabilities → [seq_len, num_experts]
[3]  ttnn.topk            — top-K expert indices and scores → [seq_len, top_k]
[4]  index tensor construction  — expert assignment map (host-side or tensor op)
[5]  ttnn.gather          — token reorder by expert → [total_tokens, d_model]
 ── T3K only: all_gather / all_to_all CCL ──────────────────────────────────
[6]  ttnn.matmul          — gate projection (per expert) → [tokens, d_ff]
[7]  ttnn.matmul          — up projection (per expert) → [tokens, d_ff]
[8]  ttnn.silu            — gated activation → [tokens, d_ff]
[9]  element-wise multiply — gate × up → [tokens, d_ff]
[10] ttnn.matmul          — down projection (per expert) → [tokens, d_model]
 ── T3K only: reduce_scatter / all_reduce CCL ──────────────────────────────
[11] ttnn.scatter         — inverse gather back to token order → [seq_len, d_model]
[12] element-wise multiply — apply router scores
[13] ttnn.sum             — accumulate top-K expert outputs → [seq_len, d_model]
      │
      ▼
MoE output: [seq_len, d_model]
```

Steps [1]–[5] are the **dispatch phase** (`dispatch_phase.md`).
Steps [6]–[10] are the **expert matmul phase** (`expert_matmul_phase.md`).
Steps [11]–[13] are the **combine phase** (`combine_phase.md`).

---

## Chapter Structure

| File | Contents |
|---|---|
| [`dispatch_phase.md`](./dispatch_phase.md) | Router ops, gather, T3K all-to-all, latency budget |
| [`expert_matmul_phase.md`](./expert_matmul_phase.md) | Batched matmul structure, Qwen 235B latency, prefill vs. decode |
| [`combine_phase.md`](./combine_phase.md) | Scatter, weighted sum, T3K reduction, load imbalance |
| [`full_op_sequence_reference.md`](./full_op_sequence_reference.md) | Consolidated op table: shapes, durations, Tracy zone names |

---

## Key Model Configuration (Used Throughout This Chapter)

All latency estimates in this chapter use Qwen 235B MoE dimensions unless otherwise noted:

| Parameter | Value |
|---|---|
| `d_model` | 7168 |
| `d_ff` (per expert) | 2048 |
| `num_experts` | 128 |
| `top_k` | 8 |
| Hardware | Wormhole B0; 80 Tensix cores, ~131 TFLOP/s BF16, ~300 GB/s DRAM |
| Dtype | BF16 |

DeepSeek-V3 and Mixtral variations are documented in `full_op_sequence_reference.md`.

---

## Next Steps

Proceed to [`dispatch_phase.md`](./dispatch_phase.md) to walk through the router ops and token
redistribution mechanics in detail.
