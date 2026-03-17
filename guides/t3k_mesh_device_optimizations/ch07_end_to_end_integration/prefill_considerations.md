# Prefill Considerations

> **Quick Reference — TTNN API Symbols Used**
>
> | Symbol | Description |
> |---|---|
> | `ttnn.DRAM_MEMORY_CONFIG` | DRAM placement for all prefill-phase A2A buffers and activations; see `ch04_memory_config/memory_config_api.md` |
> | `ttnn.L1_MEMORY_CONFIG` | L1 placement for small intermediates only; see `ch04_memory_config/memory_config_api.md` |
> | `ttnn.all_to_all` | Dispatch and combine collectives; see `ch02_ttnn_mesh_api/collective_primitives.md` |

This file identifies the parameter changes required when calling `moe_layer_t3k` (defined in
`complete_moe_layer_impl.md`) in the prefill phase ($S \gg 1$). It maps each change to the
chapter that introduced the underlying concept and explains the performance implications.

All values use Qwen3.5-35B constants: $E=256$, $k=8$, $N=8$, $E_d=32$, $H=7168$, CF$=1.25$.

---

## 1. Key Differences: Prefill vs. Decode

The table below summarises every parameter that changes between phases. Sections 2–6 explain each.

| Parameter | Decode ($S=1$) | Prefill ($S=2048$) | Chapter |
|---|---|---|---|
| `memory_config` | `L1_MEMORY_CONFIG` | `DRAM_MEMORY_CONFIG` | Ch04 |
| `num_links` | 1 | 2 | Ch03 |
| $C$ at $B=1$ | 1 | 80 | Ch05 |
| $C$ at $B=4$ | 1 | 320 | Ch05 |
| Dispatch volume per device ($B=1$) | ~3.2 MB | ~245 MB | Ch03 / Ch05 |
| Dispatch volume per device ($B=4$) | ~3.2 MB | ~981 MB | Ch03 / Ch05 |
| Total A2A buffer size (all-to-all input payload, all devices) ($B=1, S=2048$) | — | ~29.4 MB | Ch03 |
| Total A2A buffer size ($B=4, S=2048$) | — | ~117.4 MB | Ch03 |
| Total A2A buffer size ($B=32, S=2048$) | — | ~939.5 MB | Ch03 |
| Bottleneck regime | Communication-bound | Compute-bound | Ch06 |
| `per_core_M` for expert matmul | 1 | $\lceil C / E_d \rceil$ | Ch07 `complete_moe_layer_impl.md` §3 |

---

## 2. Memory Placement: DRAM Throughout

### Rule

All A2A send/receive buffers, activation tensors, and accumulation buffers for prefill must use
`ttnn.DRAM_MEMORY_CONFIG`. L1 is reserved only for small, short-lived intermediates (e.g.,
individual expert tile computations where the tile fits in a single Tensix core's CB).

### Why DRAM Is Required

At $B=4$, $S=2048$, the send buffer shape is $[N, C \times E_d, H] = [8, 10240, 7168]$:

$$\text{Total send buffer} = 8 \times 10{,}240 \times 7{,}168 \times 2 \text{ bytes}
= 8 \times 146{,}800{,}640 = 1{,}174{,}405{,}120 \text{ bytes} \approx 1.09 \text{ GB}$$

This is many times larger than the 120 MB aggregate L1 capacity of a single Wormhole B0 chip
(see `ch04_memory_config/wormhole_memory_hierarchy.md`). DRAM placement is not a performance
choice — it is mandatory.

```python
def init_prefill_memory_configs() -> ttnn.MemoryConfig:
    """
    Returns the memory config for all prefill-phase tensors.
    Call once before the prefill forward pass; do NOT switch to L1 mid-prefill.

    Contrast with decode: decode uses L1_MEMORY_CONFIG for all activations.
    This function makes the phase transition explicit at call sites.
    """
    return ttnn.DRAM_MEMORY_CONFIG
```

> **Warning:** Do not attempt to place prefill A2A buffers in L1. Even at $B=1$, $S=2048$,
> the send buffer is $8 \times 80 \times 32 \times 7168 \times 2 \approx 294$ MB — far beyond
> L1 capacity. An attempt to allocate in L1 will raise `MemoryAllocationError` at kernel
> compilation time. See `ch04_memory_config/prefill_memory_strategy.md` for the full analysis.

---

## 3. Capacity $C$ at Prefill Scale

### Formula

The expert capacity $C$ includes all $S$ positions across the batch:

$$C = \left\lceil \frac{k \times B \times S \times \text{CF}}{E} \right\rceil
    = \left\lceil \frac{8 \times B \times S \times 1.25}{256} \right\rceil
    = \left\lceil \frac{B \times S}{25.6} \right\rceil$$

### Worked Examples

| $B$ | $S$ | $C = \lceil 8 \times B \times S \times 1.25 / 256 \rceil$ | Send buffer $[C \times E_d, H]$ | Buffer size per device |
|---|---|---|---|---|
| 1 | 2048 | $\lceil 80 \rceil = 80$ | $[2560, 7168]$ | $2560 \times 7168 \times 2 \approx 36.7$ MB |
| 4 | 2048 | $\lceil 320 \rceil = 320$ | $[10240, 7168]$ | $10240 \times 7168 \times 2 \approx 146.8$ MB |
| 32 | 2048 | $\lceil 2560 \rceil = 2560$ | $[81920, 7168]$ | $81920 \times 7168 \times 2 \approx 1.17$ GB |

Compare with decode at $B=32$, $S=1$: $C=2$, send buffer $[64, 7168] = 917$ KB per device.
Prefill capacity is 1280× larger at $B=32$, $S=2048$.

### `per_core_M` for Expert Matmul Program Config

In the prefill phase, each Tensix core must handle more output rows in the expert FFN matmul.
The `per_core_M` parameter (rows per core in the matmul program config) scales as:

$$\text{per\_core\_M} = \left\lceil \frac{C}{E_d} \right\rceil$$

At $B=4$, $S=2048$, $C=320$, $E_d=32$: `per_core_M = ceil(320/32) = 10`.

At decode $B=32$, $S=1$, $C=2$, $E_d=32$: `per_core_M = ceil(2/32) = 1`.

Setting `per_core_M` incorrectly for the phase results in suboptimal Tensix core utilization.
[VERIFY: confirm `per_core_M` is the correct TTNN MatmulProgramConfig field name for Wormhole B0]

---

## 4. Dispatch Volume and `num_links=2`

### Dispatch Volume at Prefill

The per-device dispatch volume formula is the same as for decode:

$$V_{\text{dispatch}} = (N-1) \times C \times E_d \times H \times 2 \text{ bytes}$$

At prefill scales this becomes very large:

| $B$ | $S$ | $C$ | $V_{\text{dispatch}} = 7 \times C \times 32 \times 7168 \times 2$ |
|---|---|---|---|
| 1 | 2048 | 80 | $7 \times 80 \times 32 \times 7168 \times 2 = 256{,}901{,}120 \approx 245$ MB |
| 4 | 2048 | 320 | $7 \times 320 \times 32 \times 7168 \times 2 = 1{,}027{,}604{,}480 \approx 981$ MB |

**Note on total all-to-all buffer sizes:** The dispatch volume formula above gives the raw Ethernet
transfer volume per device. The all-to-all input tensor sizes (as reported in
`ch03_all_to_all_num_links/all_to_all_in_moe.md`) are defined differently — they represent the
input activation payload entering the all-to-all collective:

| $B$ | $S$ | Total A2A payload (Ch03 definition) |
|---|---|---|
| 1 | 2048 | ~29.4 MB |
| 4 | 2048 | ~117.4 MB |
| 32 | 2048 | ~939.5 MB |

These Ch03 values scale as $B \times S \times H \times 2$ (total embedding tensor size) and are
the reference figures used throughout this guide for prefill buffer sizing. The Ethernet transfer
volume per device ($V_{\text{dispatch}}$ above) is larger because it includes capacity padding.
Use the Ch03 figures when comparing against bandwidth measurements; use $V_{\text{dispatch}}$ when
sizing DRAM allocations.

### `num_links=2` Selection

At prefill scale, `num_links=2` is always used. The second Ethernet link provides near-linear
throughput improvement when the payload exceeds the threshold (~20 MB [UNVERIFIED]) identified in
`ch03_all_to_all_num_links/num_links_parameter.md`.

With `num_links=2` and 12.5 GB/s per link:

$$t_{\text{A2A}} \approx \frac{V_{\text{Ch03}}}{2 \times 12.5 \text{ GB/s}}$$

At $B=4$, $S=2048$ (Ch03 value 117.4 MB):

$$t_{\text{A2A}} \approx \frac{117.4 \text{ MB}}{25 \text{ GB/s}} \approx 4.7 \text{ ms per A2A} \quad \text{[UNVERIFIED]}$$

This is an estimate using the Ch03 payload definition. Actual latency depends on T3K routing
table, hop count distribution, and link contention. Verify against profiler output using the
procedure in `ch06_profiling/ttnn_profiler.md`.

---

## 5. Prefill vs. Decode: Compute-vs-Communication Regime

### Regime Identification

The decode phase is **communication-bound**: A2A latency (~0.51 ms per direction at $B=32$)
dominates step time, and Tensix cores are idle during Ethernet transfers.

The prefill phase is **compute-bound** for the expert FFN matmuls: with $C=320$ at $B=4$,
$S=2048$, each expert processes 320 tokens. The FFN matmul shape is $[C, H] \times [H, D_{\text{ffn}}]
= [320, 7168] \times [7168, D_{\text{ffn}}]$ — large enough to saturate Tensix arithmetic units.

| Metric | Decode ($B=32$, $S=1$) | Prefill ($B=4$, $S=2048$) |
|---|---|---|
| A2A time (dispatch, 1 direction) | ~0.51 ms | ~4.7 ms [UNVERIFIED] |
| Expert FFN time (32 experts) | Sub-ms (2 token slots per expert) | Milliseconds (320 token slots per expert) [UNVERIFIED] |
| Dominant bottleneck | Communication (A2A) | Compute (expert FFN matmul) |
| Tensix utilization during A2A | Low (idle waiting) | Moderate (can be overlapped; see §6) |
| `num_links` benefit | Marginal (payload too small) | Significant (~50% A2A latency reduction) |

This regime difference means that tuning strategies for decode (reducing A2A latency, increasing
`num_links`) are largely ineffective for prefill. Prefill optimization focuses on:
- Maximizing expert matmul throughput (batching, `per_core_M`, tiling).
- Overlapping A2A communication with independent computation (see §6).

Confirm the regime for your specific workload using the profiling procedure in
`ch06_profiling/bottleneck_diagnosis_guide.md` before applying any optimization.

---

## 6. Overlap: Combine A2A with Next-Layer Q/K/V Projection

### Overlap Opportunity

During prefill, the combine A2A for MoE layer $l$ and the Q/K/V weight loading for layer $l+1$
are independent: the Q/K/V projection for layer $l+1$ depends on the post-MoE hidden states of
layer $l$ (i.e., the residual add output), not directly on the combine A2A output before
accumulation.

The data dependency chain is:

```
Layer l:
  combine A2A completes → weighted accumulate → residual add → [hidden states l+1]
                                                                        │
Layer l+1:                                                              ▼
  [Q/K/V projection starts] ←────────────────── depends on [hidden states l+1]
```

The Q/K/V projection cannot begin until the residual add of layer $l$ completes, so there is no
direct overlap with layer $l$'s combine. However, **weight tile prefetching** for layer $l+1$'s
Q/K/V projection can proceed while layer $l$'s combine A2A is in flight:

```
Timeline (prefill, single layer):
  t=0:   [Expert FFN compute for layer l]
  t=a:   [Combine A2A starts — layer l expert outputs cross Ethernet]
  t=a:   [Prefetch Q/K/V weights for layer l+1 into Tensix CBs]  ← overlap
  t=a+t₁: [Combine A2A completes]
  t=a+t₁: [Weighted accumulate + residual add — very fast]
  t=a+t₂: [Q/K/V projection for layer l+1 — weight tiles already in CB]
```

The potential latency savings equal approximately the weight prefetch time, which partially
overlaps the ~4.7 ms combine A2A duration. The actual savings depend on the Wormhole B0 NOC
bandwidth available for weight loads during the A2A, which competes with Ethernet DMA transfers.

> **Tip:** Enable async dispatch for `ttnn.all_to_all` in your TTNN program configuration to
> allow the runtime to pipeline combine and the next-layer weight prefetch automatically.
> [VERIFY: async dispatch API for ttnn.all_to_all on Wormhole B0]
> After enabling, verify with the TTNN profiler that combine and prefetch are overlapping and not
> serialized by unexpected data dependencies. See `ch05_expert_parallelism/combine_and_accumulation.md`
> §5 for the general overlap discussion.

---

## 7. Complete Prefill Data Flow

```
All 8 devices hold hidden states [B, S, H]
(processed position-by-position or in a single batched forward pass)
        │
        ▼
  Router matmul + top-k                ← complete_moe_layer_impl.md §1
  [B*S, H] × [H, 256] → logits [B*S, 256]
  top-k → indices [B*S, k], scores [B*S, k]
        │
        ▼
  Send buffer packing                  ← complete_moe_layer_impl.md §2
  C = ceil(k*B*S*CF/E) = 320 at B=4,S=2048
  output: [8, C*32, 7168] in DRAM
  send buffer per device: ~146.8 MB at B=4, S=2048
        │
        ▼
  ttnn.all_to_all (dispatch)           ← complete_moe_layer_impl.md §2
  num_links=2, topology=Linear
  memory_config=DRAM_MEMORY_CONFIG
  A2A payload ~117.4 MB (Ch03 definition)
  estimated latency ~4.7 ms [UNVERIFIED]
        │
        ▼
  Local expert FFN compute             ← complete_moe_layer_impl.md §3
  32 experts × 320 token slots each (B=4, S=2048)
  input/output: [C*E_d, H] = [10240, 7168] in DRAM
  per_core_M = ceil(320/32) = 10
  compute-bound at this scale
        │
        ├──── prefetch Q/K/V weights for next layer into Tensix CBs (overlap)
        │
        ▼
  ttnn.all_to_all (combine)            ← complete_moe_layer_impl.md §4
  same volume as dispatch, num_links=2
  memory_config=DRAM_MEMORY_CONFIG
        │
        ▼
  Weighted accumulation [B*S, H]       ← complete_moe_layer_impl.md §5
  k=8 outputs per token, scores normalized
  output in DRAM
        │
        ▼
  Residual add + layer norm            ← apply_residual in complete_moe_layer_impl.md §5
  → hidden states [B*S, H] for next layer
```

---

## 8. Calling `moe_layer_t3k` for Prefill

```python
def prefill_moe_call(
    hidden_states: ttnn.Tensor,    # [B, S, H] or [B*S, H] — prefill activations (DRAM)
    layer_params: dict,
    mesh_device: ttnn.MeshDevice,
    B: int,
    S: int,
) -> ttnn.Tensor:
    """
    Calls moe_layer_t3k for the prefill phase.
    All parameter choices differ from the decode call; see §1 summary table.

    Hidden states may be passed as [B*S, H] (flattened) if the model processes
    all positions in a single matmul. Unflatten to [B, S, H] after the MoE layer
    if required by the subsequent attention computation.
    [VERIFY: whether Qwen3.5-35B processes prefill as [B*S, H] or [B, S, H]]
    """
    BS = B * S  # total tokens in this prefill batch

    # Flatten to [B*S, H] for routing and dispatch
    # (routing is per-token; position within sequence does not affect expert selection)
    x_flat = ttnn.reshape(hidden_states, [BS, hidden_states.shape[-1]])

    moe_out_flat = moe_layer_t3k(
        x              = x_flat,
        W_r            = layer_params["router_weight"],
        expert_weights = layer_params["expert_weights"],
        mesh_device    = mesh_device,
        B              = BS,          # treat all B*S tokens as a flat batch for routing
        phase          = "prefill",
        S              = 1,           # C already accounts for S via B=B*S
        # Alternatively, pass B=B and S=S directly if moe_layer_t3k's formula
        # C = ceil(k*B*S*CF/E) should account for both; use whichever matches
        # the implementation. [VERIFY]
    )
    # moe_out_flat: [B*S, H]

    # Reshape back to [B, S, H] if needed
    moe_out = ttnn.reshape(moe_out_flat, [B, S, hidden_states.shape[-1]])

    # Residual add
    return apply_residual(hidden_states, moe_out,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

---

## References

- `complete_moe_layer_impl.md` — `moe_layer_t3k` function; `per_core_M` note in §3
- `decode_loop_integration.md` (this chapter) — Decode parameters for comparison
- `ch03_all_to_all_num_links/num_links_parameter.md` — `num_links=2` threshold derivation
- `ch03_all_to_all_num_links/all_to_all_in_moe.md` — Prefill A2A buffer sizes: 29.4 MB ($B=1$), 117.4 MB ($B=4$), 939.5 MB ($B=32$) at $S=2048$
- `ch04_memory_config/prefill_memory_strategy.md` — DRAM placement analysis for prefill A2A buffers
- `ch04_memory_config/wormhole_memory_hierarchy.md` — 120 MB aggregate L1 capacity; mandatory DRAM for large tensors
- `ch05_expert_parallelism/token_routing_and_dispatch.md` — Capacity $C$ formula; prefill send buffer size
- `ch05_expert_parallelism/combine_and_accumulation.md` — Combine A2A volume; overlap with next-layer prefetch §5
- `ch06_profiling/bottleneck_diagnosis_guide.md` — Regime identification: compute-bound vs. communication-bound for prefill
- `ch06_profiling/ttnn_profiler.md` — Profiler setup for prefill phase benchmarking
