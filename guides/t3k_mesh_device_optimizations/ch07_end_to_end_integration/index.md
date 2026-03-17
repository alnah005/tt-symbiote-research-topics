# Chapter 7: End-to-End Integration — Complete MoE Layer on T3K

> **Quick Reference — No New TTNN API Symbols Introduced in This Index**
>
> This index introduces no new `ttnn` API symbols. All API symbols used in this chapter
> (`ttnn.all_to_all`, `ttnn.Topology.Linear`, `ttnn.L1_MEMORY_CONFIG`,
> `ttnn.DRAM_MEMORY_CONFIG`, `MeshDevice`) are defined in earlier chapters.
> See `ch02_ttnn_mesh_api/collective_primitives.md` and `ch04_memory_config/memory_config_api.md`.

## Prerequisites

This chapter requires familiarity with all six preceding chapters of this guide:

- **Chapter 1** (`ch01_t3k_topology/`) — T3K physical layout: the 1×8 linear mesh, device IDs 0–7,
  `ttnn.Topology.Linear`, per-link Ethernet bandwidth (~12.5 GB/s), and the average hop count of
  3.0 across all device pairs.
- **Chapter 2** (`ch02_ttnn_mesh_api/`) — The `ttnn` mesh API: `MeshDevice`, `MeshTensor`,
  distributed sharding primitives, and `ttnn.all_to_all` as described in `tensor_distribution.md`
  and `collective_primitives.md`.
- **Chapter 3** (`ch03_all_to_all_num_links/`) — All-to-all mechanics for MoE dispatch and
  combine, the `num_links` parameter, and how payload size determines optimal link count;
  see `all_to_all_in_moe.md` and `num_links_parameter.md`.
- **Chapter 4** (`ch04_memory_config/`) — Wormhole B0 memory hierarchy, L1 vs. DRAM tensor
  placement rules, and buffer size recommendations for decode and prefill;
  see `decode_memory_strategy.md`, `prefill_memory_strategy.md`, and `wormhole_memory_hierarchy.md`.
- **Chapter 5** (`ch05_expert_parallelism/`) — Expert assignment across 8 devices, token routing
  and dispatch flow, capacity padding, and the combine and weighted accumulation;
  see `expert_placement_strategies.md`, `token_routing_and_dispatch.md`, and
  `combine_and_accumulation.md`.
- **Chapter 6** (`ch06_profiling/`) — TTNN profiler setup, hardware performance counters,
  bottleneck categorization, and targeted remediation procedures; see `ttnn_profiler.md`,
  `device_perf_counters.md`, and `bottleneck_diagnosis_guide.md`.

Readers who have not completed Chapters 1–6 will encounter undefined terms and forward references.
Return here after completing them.

---

## Goal

This chapter integrates all prior material into a single, production-ready reference implementation
of the Qwen3.5-35B MoE layer on T3K. The goal is not to introduce new concepts but to show
**how every parameter choice from Chapters 1–6 fits together** and what the correct sequencing of
those choices is at initialization time and at every forward pass.

Specific goals:

1. Provide a single `moe_layer_t3k` pseudocode function that applies topology, mesh API,
   `num_links`, memory config, expert parallelism, and profiling hooks in the correct order.
2. Show the complete autoregressive decode loop with MoE layer calls, memory lifecycle management,
   and batch-padding conventions.
3. Identify the key differences in prefill configuration — memory placement, `num_links`, capacity
   $C$, and compute-vs-communication regime — and map each difference to the chapter that
   introduced it.

---

## Chapter Overview

### How the Prior Chapters Connect Here

| Chapter | Key Decision | Where It Appears in This Chapter |
|---|---|---|
| Ch01: T3K Topology | `ttnn.Topology.Linear`, `cluster_axis=1` | `complete_moe_layer_impl.md` §2 — both A2A calls |
| Ch02: Mesh API | `MeshDevice` construction, `ttnn.all_to_all` | `complete_moe_layer_impl.md` §1 and §2 |
| Ch03: `num_links` | `num_links=1` for decode, `num_links=2` for prefill | `complete_moe_layer_impl.md` §3 — selection table |
| Ch04: Memory Config | L1 for decode, DRAM for prefill; set once at phase init | `decode_loop_integration.md` §2, `prefill_considerations.md` §2 |
| Ch05: Expert Parallelism | Send buffer shape, capacity $C$, weighted combine | `complete_moe_layer_impl.md` §1–§4 |
| Ch06: Profiling | First-step profiling, then disable for production | `decode_loop_integration.md` §5 |

No single prior chapter synthesizes all six concerns. This chapter is the integration point.

### Model and Hardware Constants

All quantitative analysis in this chapter uses Qwen3.5-35B parameters on an 8-device T3K mesh.
These constants are unchanged from Chapters 5 and 6.

| Symbol | Meaning | Value |
|---|---|---|
| $E$ | Total experts | 256 |
| $k$ | Top-$k$ experts per token | 8 |
| $N$ | T3K devices | 8 |
| $E_d$ | Experts per device ($E/N$) | 32 |
| $H$ | Hidden dimension | 7168 |
| CF | Capacity factor | 1.25 |
| $C$ (decode) | $\lceil k \times B \times \text{CF} / E \rceil = \lceil B/25.6 \rceil$ | 1 at $B=1$; 2 at $B=32$ |
| $C$ (prefill, $S=2048$) | $\lceil k \times B \times S \times \text{CF} / E \rceil = \lceil 80 \times B \rceil$ | 80 at $B=1$; 320 at $B=4$ |

---

## Navigation

| File | Description |
|---|---|
| [`complete_moe_layer_impl.md`](./complete_moe_layer_impl.md) | Full `moe_layer_t3k` pseudocode: router, pack, dispatch A2A, expert FFN, combine A2A, weighted accumulation, residual add; `num_links` selection table; program config choices; error guards |
| [`decode_loop_integration.md`](./decode_loop_integration.md) | Autoregressive decode loop with MoE layer calls; memory lifecycle; KV cache integration; batch padding to $B=32$; performance targets; profiling discipline |
| [`prefill_considerations.md`](./prefill_considerations.md) | Prefill vs. decode differences: DRAM memory config, `num_links=2`, large capacity $C$; dispatch volume table; compute-vs-communication regime; overlap with next-layer Q/K/V projection |

### Recommended Reading Order

1. Read [`complete_moe_layer_impl.md`](./complete_moe_layer_impl.md) first to understand the canonical forward-pass structure. All
   parameters chosen in decode and prefill contexts are explained by reference to this implementation.
2. Read [`decode_loop_integration.md`](./decode_loop_integration.md) to understand how the MoE layer is called at each generation
   step, how memory configs are managed across steps, and how to handle variable batch sizes.
3. Read [`prefill_considerations.md`](./prefill_considerations.md) last, as it describes the parameter changes relative to
   [`complete_moe_layer_impl.md`](./complete_moe_layer_impl.md) that are required for the prefill phase.

---

## Full Decode Data Flow

The diagram below shows the complete MoE layer forward pass for the decode phase ($B \leq 32$,
$S=1$). Each node references the file and section that covers it.

```
All 8 devices hold identical hidden states [B, H]
        │
        ▼
  Router matmul + top-k                ← complete_moe_layer_impl.md §1
  [B,H] × [H,E] → logits [B,E]
  top-k → indices [B,k], scores [B,k]
        │
        ▼
  Send buffer packing                  ← complete_moe_layer_impl.md §2
  output: [N, C×E_d, H] in L1
  C=1 (B=1) or C=2 (B=32)
        │
        ▼
  ttnn.all_to_all (dispatch)           ← complete_moe_layer_impl.md §2
  num_links=1, topology=Linear
  memory_config=L1_MEMORY_CONFIG
  per-device payload: 3.2 MB (B=1) or 6.4 MB (B=32)
  latency: ~0.26 ms (B=1) or ~0.51 ms (B=32)
        │
        ▼
  Local expert FFN compute             ← complete_moe_layer_impl.md §3
  32 experts per device
  input/output: [C×E_d, H] in L1
        │
        ▼
  ttnn.all_to_all (combine)            ← complete_moe_layer_impl.md §4
  same volume as dispatch
  memory_config=L1_MEMORY_CONFIG
        │
        ▼
  Weighted accumulation [B, H]         ← complete_moe_layer_impl.md §5
  k=8 outputs per token, scores normalized
  output in L1
        │
        ▼
  Residual add + layer norm            ← (next transformer layer)
  residual + output → updated hidden states [B, H]
```

For the prefill phase, replace: L1 → DRAM, num_links=1 → 2, C grows by factor $S$.
See `prefill_considerations.md` for the full prefill flow diagram.

---

## Key Integration Decisions

The following three decisions must be made at model initialization and kept consistent
throughout inference. Changing any of them mid-inference will cause correctness or performance
regressions.

### Decision 1: Expert Placement Strategy

Choose once at startup (see `ch05_expert_parallelism/expert_placement_strategies.md`):

- **Naive uniform** (default): device $d$ holds experts $[32d, 32d+31]$. Simple; balanced under
  uniform routing.
- **Load-balanced**: offline profiled assignment; reduces hot-expert traffic at the cost of a
  lookup table per forward pass.

The placement strategy determines the dispatch routing table and cannot be changed without
re-initializing expert weights.

### Decision 2: Memory Config Phase

Activate the correct memory config set before the first step of each phase:

- **Decode:** `ttnn.L1_MEMORY_CONFIG` for all activations and A2A buffers.
- **Prefill:** `ttnn.DRAM_MEMORY_CONFIG` for all A2A buffers and large activations.

Do not mix L1 and DRAM configs within a single phase; the mismatch causes unnecessary copies.
See `decode_loop_integration.md` §2 and `prefill_considerations.md` §2 for initialization code.

### Decision 3: `num_links` per Phase

Set at the `ttnn.all_to_all` call site, not globally:

- **Decode:** `num_links=1` (payload $\leq 6.4$ MB; adding a second link adds setup overhead
  that exceeds the marginal latency reduction).
- **Prefill:** `num_links=2` (payload 29.4 MB–939.5 MB; second link provides near-linear
  throughput improvement).

Refer to `ch03_all_to_all_num_links/num_links_parameter.md` for the full bandwidth model.

---

## When to Read Which File

| Scenario | Recommended File |
|---|---|
| First implementation of `moe_layer_t3k` | [`complete_moe_layer_impl.md`](./complete_moe_layer_impl.md) |
| Integrating MoE layer into a decode loop | [`decode_loop_integration.md`](./decode_loop_integration.md) |
| Adapting decode implementation for prefill | [`prefill_considerations.md`](./prefill_considerations.md) |
| Debugging incorrect output after integration | [`complete_moe_layer_impl.md`](./complete_moe_layer_impl.md) §6 (error guards) |
| Diagnosing decode latency regression | [`decode_loop_integration.md`](./decode_loop_integration.md) §5, then [`ch06_profiling/`](../ch06_profiling/index.md) |
| Understanding prefill memory OOM errors | [`prefill_considerations.md`](./prefill_considerations.md) §2 |
| Verifying `num_links` choice for a given batch size | [`complete_moe_layer_impl.md`](./complete_moe_layer_impl.md) §3 table |
| Understanding KV cache and MoE buffer interaction | [`decode_loop_integration.md`](./decode_loop_integration.md) §3 |

---

## References

- `ch01_t3k_topology/t3k_physical_layout.md` — T3K 1×8 mesh, device IDs 0–7, hop-count derivation
- `ch01_t3k_topology/ethernet_link_bandwidth.md` — Per-link bandwidth; basis for latency estimates
- `ch02_ttnn_mesh_api/collective_primitives.md` — `ttnn.all_to_all` API reference
- `ch02_ttnn_mesh_api/mesh_device_setup.md` — `MeshDevice` construction and configuration
- `ch03_all_to_all_num_links/num_links_parameter.md` — `num_links` bandwidth model and selection
- `ch03_all_to_all_num_links/all_to_all_in_moe.md` — Dispatch volume derivations and buffer sizes
- `ch04_memory_config/decode_memory_strategy.md` — L1 placement for decode-phase buffers
- `ch04_memory_config/prefill_memory_strategy.md` — DRAM placement for prefill-phase buffers
- `ch04_memory_config/wormhole_memory_hierarchy.md` — Wormhole B0 memory capacities and bandwidths
- `ch05_expert_parallelism/token_routing_and_dispatch.md` — Send buffer construction, capacity formula
- `ch05_expert_parallelism/combine_and_accumulation.md` — Combine A2A structure, weighted accumulation
- `ch05_expert_parallelism/expert_placement_strategies.md` — Placement strategy selection
- `ch06_profiling/ttnn_profiler.md` — Profiler setup and output parsing
- `ch06_profiling/bottleneck_diagnosis_guide.md` — Bottleneck categorization and remediation
