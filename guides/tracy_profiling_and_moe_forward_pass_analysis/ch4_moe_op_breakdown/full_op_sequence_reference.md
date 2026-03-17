# Full Op Sequence Reference

This file is the ground-truth checklist for reading a Tracy trace or device profiler CSV from
a MoE forward pass. Use it to verify that every expected op is present, has the correct input
and output shapes, and falls within its expected duration range.

Latency values use Qwen 235B MoE dimensions (d_model=7168, d_ff=2048, num_experts=128, top_k=8)
on Wormhole B0 BF16 unless a model-specific column is noted. Prefill: seq_len=1024. Decode: seq_len=1.

---

## Complete Op Table

| # | Op | Tracy Zone Name | CSV Op Name | Input Shape(s) | Output Shape | Prefill µs | Decode µs | Single/T3K |
|---|---|---|---|---|---|---|---|---|
| 1 | Router linear | `MoE/dispatch/router_linear` | `matmul` | `[1024, 7168]` × `[7168, 128]` | `[1024, 128]` | 80–200 | 5–20 | both |
| 2 | Router softmax | `MoE/dispatch/router_softmax` | `softmax` | `[1024, 128]` | `[1024, 128]` | 10–30 | 1–5 | both |
| 3 | TopK selection | `MoE/dispatch/topk` | `topk` | `[1024, 128]` | `[1024, 8]` (vals+idx) | 20–60 | 2–8 | both |
| 4 | Index construction | `MoE/dispatch/index_construction` | `argsort` / none | routing indices | gather index tensor | 5–50 | 2–10 | both |
| 5 | Token gather | `MoE/dispatch/gather` | `gather` | `[1024, 7168]` + index | `[8192, 7168]` | 30–80 | 3–10 | both |
| 6 | All-to-all CCL | `MoE/dispatch/all_to_all` | `all_gather` | `[8192/ep, 7168]` per chip | `[tokens_local, 7168]` | 300–800 | 300–800 | T3K only |
| 7 | Gate proj matmul | `MoE/expert_matmul/gate_proj` | `matmul` | `[128, 64, 7168]` × `[128, 7168, 2048]` | `[128, 64, 2048]` | 1800–4000 | 80–200 | both |
| 8 | Up proj matmul | `MoE/expert_matmul/up_proj` | `matmul` | `[128, 64, 7168]` × `[128, 7168, 2048]` | `[128, 64, 2048]` | 1800–4000 | 80–200 | both |
| 9 | SiLU activation | `MoE/expert_matmul/silu` | `silu` | `[128, 64, 2048]` | `[128, 64, 2048]` | 200–500 | 10–30 | both |
| 10 | Gate × up multiply | `MoE/expert_matmul/gate_multiply` | `mul` | 2× `[128, 64, 2048]` | `[128, 64, 2048]` | 100–300 | 5–20 | both |
| 11 | Down proj matmul | `MoE/expert_matmul/down_proj` | `matmul` | `[128, 64, 2048]` × `[128, 2048, 7168]` | `[128, 64, 7168]` | 1800–4000 | 80–200 | both |
| 12 | Reduce-scatter CCL | `MoE/combine/reduce_scatter` | `all_gather` | `[8192, 7168]` partial | `[8192/ep, 7168]` | 2000–8000 | 300–800 | T3K only |
| 13 | Inverse scatter | `MoE/combine/scatter` | `scatter` | `[128, 64, 7168]` + index | `[1024, 8, 7168]` | 50–150 | 5–20 | both |
| 14 | Router score multiply | `MoE/combine/score_multiply` | `mul` | `[1024, 8, 7168]` × `[1024, 8]` | `[1024, 8, 7168]` | 20–60 | 2–8 | both |
| 15 | Top-K sum | `MoE/combine/topk_sum` | `sum` | `[1024, 8, 7168]` | `[1024, 7168]` | 15–40 | 2–5 | both |

**Note on decode shapes:** at seq_len=1, `[1024, ...]` → `[1, ...]`; `expert_capacity` shrinks to
`top_k / num_active_experts`; only the 8 selected experts have active tokens. The batched matmul
degenerates toward `[8, 1, 7168]` × `[8, 7168, 2048]`.

---

## How to Use This Table as a Ground-Truth Checklist

1. **Export the device profiler CSV:** run your MoE forward pass with `TT_METAL_DEVICE_PROFILER=1`
   and post-process with `process_ops_logs.py`. The `DEVICE KERNEL DURATION [ns]` column is the
   hardware measurement.

2. **Filter by op name:** use the "CSV Op Name" column to find candidate rows. The same op name
   (e.g., `matmul`) will appear multiple times; disambiguate by cross-referencing input shapes
   from the "Input Shape(s)" column. The device profiler CSV includes per-op shape metadata.

3. **Check for missing ops:** if a row is absent from the CSV, either the op was fused into a
   neighboring op, it ran on the host CPU (not the device), or it was skipped due to shape
   conditions. Index construction (op 4) is the most common missing op.

4. **Flag out-of-range durations:** if a duration falls outside the "Prefill µs" or "Decode µs"
   range by more than 2×, investigate: program cache miss (first-run compilation), unexpected
   shape change, memory config mismatch, or a load-imbalance event.

5. **T3K trace:** ops 6 and 12 are present only in multi-chip traces. If they are absent in a T3K
   trace, expert parallelism is not active (single-chip fallback) or the CCL was not annotated.
   Check the Tracy zone gap between op 5 and op 7 for an unannotated all-to-all.

6. **Summing phases:**
   - Dispatch total = ops 1–5 (+ op 6 on T3K)
   - Expert matmul total = ops 7–11
   - Combine total = ops 12–15 (op 12 T3K only)
   - Any remaining wallclock time is unaccounted gap (see Chapter 5).

---

## Known Variations Across MoE Model Families

### DeepSeek-V3 — Shared Expert Path

DeepSeek-V3 has a "shared expert" that processes all tokens regardless of routing, in addition to
the top-K routed experts. This adds two extra matmul ops (shared gate proj and shared down proj)
that appear as unconditional `matmul` entries in the CSV before the routing dispatch.

| Additional op | Tracy zone | Shape | Notes |
|---|---|---|---|
| Shared gate + up proj | `MoE/shared_expert/gate_up` | `[seq_len, 7168]` × `[7168, 4096]` | All tokens; larger d_ff than routed experts |
| Shared down proj | `MoE/shared_expert/down` | `[seq_len, 4096]` × `[4096, 7168]` | — |
| Add shared output | `MoE/combine/add_shared` | `mul` / `add` | Summed with routed MoE output |

DeepSeek-V3 also has `num_experts=128` (vs. Qwen's 128), the same expert count. The router
softmax and topk operate over a 128-wide expert dimension; router linear is `[seq_len, 7168]` ×
`[7168, 128]`.

### Mixtral — Two-Expert Routing

Mixtral 8×7B uses `num_experts=8`, `top_k=2`. The op sequence is identical but dimensions are
different: d_model=4096, d_ff=14336, num_experts=8.

Key differences in the trace:
- Router linear: `[seq_len, 4096]` × `[4096, 8]` — very fast (~5–15 µs).
- Topk over 8 experts (vs. 128) — near-zero latency.
- Expert matmul phase dominates even more heavily because d_ff=14336 (7× larger than Qwen per-expert d_ff).
- No CCL all-to-all unless ep_degree > 1; at ep_degree=8, all experts fit on one T3K chip each.

### Qwen 235B MoE — 128-Expert High-Sparsity Configuration

This is the primary configuration used throughout this chapter. Key characteristics:
- `num_experts=128`, `top_k=8` → sparsity = 1 - 8/128 = 93.75%; only 6.25% of experts activated per token.
- `expert_capacity` at seq_len=1024: `1024 × 8 / 128 = 64` tokens/expert expected value.
- All 128 expert weight tensors must be resident in DRAM; on a single Wormhole B0 with 12 GDDR6 banks
  at 12 GB total: `128 × 2 × (7168 × 2048 + 2048 × 7168) × 2B ≈ 14.7 GB` for gate+down weights alone.
  This exceeds single-chip DRAM; T3K expert parallelism is required for full Qwen 235B.

---

## Phase Latency Summary

| Phase | Single chip prefill (µs) | T3K prefill (µs) | Single chip decode (µs) | T3K decode (µs) |
|---|---|---|---|---|
| Dispatch | 150–400 | 450–1200 | 15–55 | 320–870 |
| Expert matmul | 6000–13000 | 6000–13000 | 300–650 | 300–650 |
| Combine | 100–250 | 2100–8300 | 15–55 | 315–855 |
| **Total MoE layer** | **~6300–13700** | **~8600–22500** | **~330–760** | **~935–2375** |

---

## Next Steps

This chapter feeds directly into Chapter 5 (`ch5_identifying_gap/`), which uses this op sequence
and latency table as the expected baseline when searching for the 16ms gap in a real Tracy trace.
