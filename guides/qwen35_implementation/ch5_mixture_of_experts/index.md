# Chapter 5 — Mixture of Experts (35B-A3B)

This chapter documents the `Qwen35MoE` implementation: the 256+1 expert structure, device-side
router matmul with host-side top-k selection, fused SwiGLU gate+up projection, bfp4 routed
expert weights, and the DRAM budget implications for the A3B and 27B models.

## Prerequisites

- **Chapter 4** — Decoder Block and Uniform Dispatch: explains how `DeltaNetDecoderBlock` selects
  `Qwen35MoE` via the `mlp_class` parameter and calls `self.feed_forward.forward(hidden, mode)`.
- **Chapter 6** — Weight Precision, DRAM Layout, and Weight Conversion: explains the full bfp4
  rationale and the HF-to-meta conversion that protects MoE keys from renaming transforms.

## Reading Order

| File | Contents |
|------|----------|
| [`architecture_overview.md`](./architecture_overview.md) | 256+1 expert structure, shared expert gating, batched routing assumption, forward-pass overlap design |
| [`router_and_routing.md`](./router_and_routing.md) | Router weight layout, device router matmul, the one mandatory sync, host top-k and softmax |
| [`expert_computation.md`](./expert_computation.md) | Fused gate+up matmul, SwiGLU fusion, bfp4 weight indexing, L1 accumulation loop |
| [`dram_budget.md`](./dram_budget.md) | Full A3B and 27B DRAM tables, bfp4 rationale, why shared expert stays at bfp8 |

## Source File

All content in this chapter is derived from:

- `models/tt_transformers/tt/qwen35_moe.py` — `Qwen35MoE` class (init + forward)
- `models/demos/qwen35/demo/demo_a3b.py` — model build loop, batch handling
- `models/demos/qwen35/tests/test_a3b_pcc.py` — `ref_moe_forward` reference and weight key names
- `models/demos/qwen35/README.md` — architecture summary and profiling table
- `models/demos/qwen35/PERF.md` — DRAM breakdown tables

## Key Facts at a Glance

- **256 routed experts + 1 shared expert** per layer across all 40 layers of the A3B model.
- **Top-8 routing**: 8 experts are selected per token per layer.
- **One host-device sync per MoE layer** per token (router logits readback: 256 floats, ~1 KB).
- **bfp4** for all routed expert weights; **bfp8** for the shared expert; **bf16** for the router.
- All intermediate tensors during the 8-expert loop use `L1_MEMORY_CONFIG` to avoid DRAM roundtrips.
- Shared expert computation is queued on device **before** the routing sync, overlapping compute
  with the CPU top-k/softmax.
