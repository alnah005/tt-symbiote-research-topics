# Kernel Gap Audit — Chapter 4 Summary

This section consolidates all TTNN coverage findings from Chapters 4.1–4.3 into a single audit table and distills three prioritized findings for kernel development planning.

---

## Audit Table

| Operation | Current Implementation | TTNN Status | Gap | Recommended Path |
|---|---|---|---|---|
| `in_proj_qkv` (GDN + GA) | `ttnn.linear` col-sharded | `[AVAILABLE]` | None | — |
| `in_proj_z` (GDN) | `ttnn.linear` col-sharded | `[AVAILABLE]` | None | — |
| `in_proj_a` (GDN) | PyTorch `nn.Linear` | `[GAP — requires wiring]` | Output `[4]` too small to shard; never wired to TTNN graph | Replace with replicated `ttnn.linear`; no sharding needed |
| `in_proj_b` (GDN) | PyTorch `nn.Linear` | `[GAP — requires wiring]` | Same as `in_proj_a` | Replace with replicated `ttnn.linear` |
| `causal_conv1d_fn` (GDN prefill) | PyTorch C extension (`causal-conv1d`) | `[GAP — requires custom kernel]` | No TTNN primitive for stateful causal 1D conv over full sequence | Custom Metalium depthwise-conv1d kernel; or sliding-window `ttnn.matmul` loop as interim |
| `causal_conv1d_update` (GDN decode) | PyTorch C extension (`causal-conv1d`) | `[GAP — requires custom kernel]` | No TTNN primitive for single-step stateful conv update | Custom Metalium kernel; single-step dot-product via `ttnn.matmul` is viable but not fused |
| `chunk_gated_delta_rule` (GDN prefill) | Python/Triton (`flash-linear-attention`) | `[GAP — requires custom kernel]` | Multi-step chunkwise algorithm with sequential inter-chunk state transfer; no single TTNN op covers it | Python loop over C=64 chunks using `ttnn.matmul`, `ttnn.mul`, `ttnn.add`, `ttnn.sub`, `ttnn.where` as first port; dedicated fused kernel for production |
| `recurrent_gated_delta_rule` (GDN decode) | PyTorch | `[GAP — requires custom kernel]` | 6 core ops individually available in TTNN but not fused; separate dispatch incurs launch overhead on small state tensors | Fused Metalium kernel for the full 6-op recurrent step (highest decode-latency priority) |
| `FusedRMSNormSwishGate` (GDN) | PyTorch CUDA kernel (`fla` library) | `[GAP — requires custom kernel]` | Composable from `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul`, but 3 passes vs. 1 fused pass | Composable TTNN decomposition sufficient for correctness; fused kernel preferred for efficiency |
| `out_proj` (GDN + GA) | `ttnn.linear` row-sharded | `[AVAILABLE]` | None | — |
| SDPA prefill (GA) | `ttnn.transformer.scaled_dot_product_attention` | `[AVAILABLE]` | None | — |
| SDPA decode (GA) | `ttnn.transformer.scaled_dot_product_attention_decode` | `[AVAILABLE]` | None | — |
| Q/K RMSNorm (GA) | `ttnn.rms_norm` | `[AVAILABLE]` | None | — |
| Gate sigmoid (GA) | `ttnn.sigmoid` | `[AVAILABLE]` | None | — |
| Q gating mul (GA) | `ttnn.mul` | `[AVAILABLE]` | None | — |
| RoPE (GA) | `ttnn.experimental.rotary_embedding` | `[AVAILABLE]` | None | — |
| GQA KV repeat (GA) | `ttnn.repeat_interleave` | `[AVAILABLE]` | None | — |
| KV cache write (GA) | `TTNNQwenPagedAttentionKVCache` (paged TTNN ops) | `[AVAILABLE]` | None | — |
| All-gather (GDN + GA) | `ttnn.experimental.all_gather_async` | `[AVAILABLE]` | None | — |

---

## Key Finding 1 — Fused Decode Kernel is the Highest-Priority Gap

The GDN recurrent decode step (`recurrent_gated_delta_rule`) can be decomposed into exactly 6 TTNN primitives:

1. `ttnn.mul` — state decay: `S_decayed = g · S_prev`
2. `ttnn.matmul` — retrieval: `S_decayed^T @ k̃`
3. `ttnn.sub` + `ttnn.mul` — error: `β · (v − retrieval)`
4. `ttnn.matmul` — write: outer product `k̃ ⊗ error^T`
5. `ttnn.add` — state update: `S_new = S_decayed + write`
6. `ttnn.matmul` — output: `S_new^T @ q̃`

All 6 ops are available. However, dispatching them separately means 6 independent kernel launches over small tensors (`[1, 32, 128, 128]` state per layer). Each launch incurs fixed overhead (command queue serialization, device-side dispatch latency) that dominates compute time when the tensor footprint is small — which is precisely the decode regime where T=1.

**Recommendation:** Implement a single fused Metalium kernel that executes the full 6-op recurrent step in one dispatch. The state tensor `[1, 32, 128, 128]` = 32 × 128 × 128 × 2 bytes (bfloat16) = 1 MB fits comfortably in L1 SRAM on Wormhole, making a register-resident fused kernel the natural design. This is the highest-priority kernel gap for decode latency.

---

## Key Finding 2 — Chunkwise Delta Rule Can Be Bootstrapped with a Python Loop

`chunk_gated_delta_rule` for prefill is a complex multi-step algorithm involving WY decomposition, intra-chunk attention, and sequential inter-chunk state transfer (see Chapter 2). Porting it as a monolithic kernel is a significant effort.

A pragmatic first step is a **Python loop over C=64 chunks**, where each iteration uses existing TTNN primitives:
- `ttnn.matmul` for intra-chunk QK and AV products.
- `ttnn.mul`, `ttnn.sub`, `ttnn.add` for the WY update and gating.
- `ttnn.where` with a precomputed lower-triangular mask for causal masking within the chunk.

For a T=2048 sequence, this is T/C = 32 loop iterations. The inter-chunk state transfer is inherently sequential, so the Python loop does not sacrifice parallelism that a fused kernel could exploit — the bottleneck is inter-chunk data dependency, not kernel overhead. This makes the chunked Python loop viable as a correctness-first implementation.

A fused Metalium kernel (or a Triton kernel if the TTNN Triton backend matures) is still recommended for production throughput, primarily to eliminate Python dispatch overhead and to fuse the intra-chunk matmuls with the WY accumulation.

---

## Key Finding 3 — Gated Attention Requires No New Kernel Development; DeltaNet Has 4 Distinct Kernel Gaps

Gated Attention is fully covered by existing TTNN primitives. The Q gating, normalization, RoPE, paged KV cache, GQA repeat, Flash Attention (prefill), and decode SDPA paths all have direct TTNN equivalents. Integrating Gated Attention into a T3K inference stack requires wiring work (tensor layout, model-parallel sharding configuration, KV cache page-table setup) but zero kernel development.

Gated Delta Net has four distinct kernel gaps requiring new Metalium kernel work:

| Priority | Kernel | Affects | Notes |
|---|---|---|---|
| 1 (highest) | Fused recurrent decode step | Decode latency per layer | 6 ops, fits in L1, high launch-overhead penalty |
| 2 | `causal_conv1d_update` | Decode latency per layer | Small; dot-product interim viable |
| 3 | `chunk_gated_delta_rule` | Prefill throughput | Python loop interim viable |
| 4 | `causal_conv1d_fn` | Prefill throughput | Sliding-window `ttnn.matmul` interim viable |

`FusedRMSNormSwishGate` is composable from existing TTNN ops and is a lower-priority optimization (3 passes rather than 1, but correctness is fully achievable today).

The two wiring gaps (`in_proj_a`, `in_proj_b`) are minor: swapping the PyTorch `nn.Linear` for a replicated `ttnn.linear` is a one-line change per projection with no kernel work required.

---

**Next:** [Chapter 5 — Compute and Memory Roofline Analysis on Wormhole](../ch5_roofline_analysis/index.md)
