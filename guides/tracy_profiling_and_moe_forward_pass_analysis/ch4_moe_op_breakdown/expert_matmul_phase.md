# Expert Matmul Phase

The expert matmul phase executes the feed-forward network (FFN) for each active expert. It is
the most compute-intensive phase and dominates total MoE layer latency in the prefill regime.
In decode, the matmuls shrink to effectively batch size 1 per expert, shifting the bottleneck.

---

## Ops in Order

### 1. `ttnn.matmul` — Gate Projection

```
Input:   [num_experts, expert_capacity, d_model]   e.g., [128, 64, 7168]
Weight:  [num_experts, d_model, d_ff]              e.g., [128, 7168, 2048]
Output:  [num_experts, expert_capacity, d_ff]      e.g., [128, 64, 2048]
Tracy zone: MoE/expert_matmul/gate_proj
CSV op name: matmul
```

The gate projection maps each token's hidden state to the intermediate FFN dimension for the
gating branch. This is the first of two parallel projections in a gated FFN (SwiGLU / SiLU-gated).

### 2. `ttnn.matmul` — Up Projection

```
Input:   [num_experts, expert_capacity, d_model]
Weight:  [num_experts, d_model, d_ff]
Output:  [num_experts, expert_capacity, d_ff]
Tracy zone: MoE/expert_matmul/up_proj
CSV op name: matmul
```

The up projection is the second parallel branch of the gated FFN. In many implementations the gate
and up projections are fused into a single matmul with output shape `[num_experts, expert_capacity,
2 × d_ff]`, then split. In TTNN this fused form also appears as a single `matmul` entry in the CSV.

### 3. `ttnn.silu` — Gated Activation

```
Input:   [num_experts, expert_capacity, d_ff]
Output:  [num_experts, expert_capacity, d_ff]
Tracy zone: MoE/expert_matmul/silu
CSV op name: silu
```

SiLU (Sigmoid Linear Unit): `silu(x) = x × sigmoid(x)`. Applied element-wise to the gate
projection output. Memory-bandwidth bound at all practical shapes because the ratio of FLOPs to
bytes is 2:1 (two FLOPs per element read/write).

### 4. Element-Wise Multiply — Gate × Up

```
Input A: silu(gate_proj)  [num_experts, expert_capacity, d_ff]
Input B: up_proj          [num_experts, expert_capacity, d_ff]
Output:  [num_experts, expert_capacity, d_ff]
Tracy zone: MoE/expert_matmul/gate_multiply
CSV op name: mul (or fused with silu in some kernels)
```

Produces the gated intermediate representation. In hardware this can be fused with the silu op
into a single kernel; check the CSV for either one `silu` entry (fused) or separate `silu` and
`mul` entries.

### 5. `ttnn.matmul` — Down Projection

```
Input:   [num_experts, expert_capacity, d_ff]      e.g., [128, 64, 2048]
Weight:  [num_experts, d_ff, d_model]              e.g., [128, 2048, 7168]
Output:  [num_experts, expert_capacity, d_model]   e.g., [128, 64, 7168]
Tracy zone: MoE/expert_matmul/down_proj
CSV op name: matmul
```

Projects the gated intermediate back to `d_model`. The output feeds directly into the combine
phase scatter op.

---

## Batched Expert Matmul Structure

TTNN executes all expert matmuls as a single batched operation over the `num_experts` dimension
rather than dispatching 128 separate matmul ops. The canonical tensor layout is:

```
Activations: [num_experts, expert_capacity, d_model]
Weights:     [num_experts, d_model, d_ff]           (gate or up)
             [num_experts, d_ff, d_model]            (down)
```

This structure is critical for performance: a single dispatch with `[128, 64, 7168]` × `[128,
7168, 2048]` keeps all 80 Tensix cores busy across the expert and token dimensions simultaneously.
Dispatching 128 separate `[64, 7168]` × `[7168, 2048]` matmuls would incur 128× the host-side
dispatch overhead and lose inter-expert parallelism.

`expert_capacity` is typically `capacity_factor × seq_len × top_k / num_experts`. At seq_len=1024,
num_experts=128, top_k=8, capacity_factor=1.0: `expert_capacity = 1.0 × 1024 × 8 / 128 = 64`. This
means each expert receives on average 64 tokens per forward pass — which is the fundamental reason
prefill is more efficient than decode for MoE: larger seq_len means more tokens per expert and better
hardware utilization.

---

## Expected Latency: Qwen 235B MoE Expert Dims

Configuration: d_model=7168, d_ff=2048, num_experts=128, top_k=8, BF16, single Wormhole B0
(80 Tensix cores, ~131 TFLOP/s peak BF16, ~300 GB/s DRAM bandwidth).

### Theoretical FLOPs per Matmul (gate or up proj, prefill seq_len=1024)

```
total_tokens = seq_len × top_k = 1024 × 8 = 8192
FLOPs = 2 × total_tokens × d_model × d_ff
      = 2 × 8192 × 7168 × 2048
      ≈ 240 GFLOPs per matmul (gate or up)
      ≈ 240 GFLOPs (down proj; transposing dimensions does not change FLOP count)
```

At 131 TFLOP/s peak: theoretical minimum ~1.8 ms per gate/up matmul. Observed values are
typically 1.5–3× above the roofline due to tile-alignment overhead and memory-bound transitions
at the boundary of the `expert_capacity` dimension.

### Observed Latency Estimates (indicative, BF16, seq_len=1024)

| Op | Regime | Expected range (µs) |
|---|---|---|
| Gate proj matmul | Prefill (seq_len=1024) | 1800–4000 |
| Up proj matmul | Prefill (seq_len=1024) | 1800–4000 |
| SiLU + gate multiply | Prefill | 200–500 |
| Down proj matmul | Prefill (seq_len=1024) | 1800–4000 |
| **Expert matmul phase total** | **Prefill** | **~6000–13000 µs** |
| Gate proj matmul | Decode (seq_len=1) | 80–200 |
| Up proj matmul | Decode (seq_len=1) | 80–200 |
| Down proj matmul | Decode (seq_len=1) | 80–200 |
| **Expert matmul phase total** | **Decode** | **~300–650 µs** |

> **Note:** Re-profile on the target system. Numbers above are computed from hardware specs and
> typical TTNN matmul efficiency; actual values depend on weight sharding, L1 reuse, and
> program cache state.

---

## Why Expert Matmul Dominates in Prefill

In prefill with seq_len=1024, `total_tokens = 8192` and each expert receives ~64 tokens
(expert_capacity=64 with capacity_factor=1.0). The batched matmul `[128, 64, 7168] × [128, 7168,
2048]` has a K-dimension of 7168 — deeply compute-bound. The dispatch and combine phases together
are typically under 2 ms; the three matmuls (gate, up, down) are 6–13 ms. Expert matmul is
70–85% of total MoE layer time in prefill.

## Why Expert Matmul Is Less Dominant in Decode

In decode, seq_len=1 (one new token per step). `total_tokens = top_k = 8`. Each expert receives
on average `8 / 128 = 0.0625` tokens. In practice only 8 of 128 experts are active per decode
step; the batched matmul degenerates to `[8, 1, 7168] × [8, 7168, 2048]` — deeply memory-bound
(M=1, K=7168, N=2048). At ~300 GB/s DRAM bandwidth, weight loading dominates; the matmuls are
now ~80–200 µs each while CCL latency for token redistribution on T3K (300–800 µs) can exceed
the compute time. Expert matmul may be only 30–50% of total MoE time in decode on T3K.

---

---

**Next:** [`combine_phase.md`](./combine_phase.md)
