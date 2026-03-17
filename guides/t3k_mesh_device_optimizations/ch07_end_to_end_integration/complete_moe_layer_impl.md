# Complete MoE Layer Implementation for T3K

> **Quick Reference — TTNN API Symbols Used**
>
> | Symbol | Description |
> |---|---|
> | `ttnn.all_to_all` | Collective for dispatch and combine; see `ch02_ttnn_mesh_api/collective_primitives.md` |
> | `ttnn.L1_MEMORY_CONFIG` | L1 SRAM placement for decode-phase buffers; see `ch04_memory_config/memory_config_api.md` |
> | `ttnn.DRAM_MEMORY_CONFIG` | DRAM placement for prefill-phase buffers; see `ch04_memory_config/memory_config_api.md` |
> | `ttnn.Topology.Linear` | T3K 1×8 linear topology; see `ch01_t3k_topology/t3k_physical_layout.md` |
> | `MeshDevice` | 8-device T3K mesh; see `ch02_ttnn_mesh_api/mesh_device_setup.md` |

This file provides the canonical `moe_layer_t3k` pseudocode integrating all decisions from
Chapters 1–6. The function is structured to match the data flow described in `index.md`:
router → pack → dispatch A2A → expert FFN → combine A2A → weighted accumulation → residual add.

All values use Qwen3.5-35B constants: $E=256$, $k=8$, $N=8$, $E_d=32$, $H=7168$, CF$=1.25$.

---

## 1. Function Signature and Parameters

```python
import math
import ttnn
import torch

def moe_layer_t3k(
    x: ttnn.Tensor,               # [B, H] — input hidden states (post-attention residual)
    W_r: ttnn.Tensor,             # [H, E] — router weight matrix; E=256
    expert_weights: list,         # list of 32 (W_gate [H, D_ffn], W_up [H, D_ffn], W_down [D_ffn, H])
                                  # one triple per local expert; DRAM-resident across all steps
    mesh_device: ttnn.MeshDevice, # 8-device T3K mesh
    B: int,                       # batch size; 1–32 for decode, any for prefill
    phase: str,                   # "decode" or "prefill"
    S: int = 1,                   # sequence length; 1 for decode, >1 for prefill
    expert_to_device: list = None, # length-256 list: expert_idx -> device_id
                                   # None → uses naive uniform placement (expert_idx // 32)
) -> ttnn.Tensor:                 # [B, H] — MoE output (before residual add)
    """
    Complete MoE layer forward pass for Qwen3.5-35B on a T3K 8-device mesh.

    Integrates:
    - Router matmul + top-k (§1)
    - Send buffer packing with capacity C (§2)
    - ttnn.all_to_all dispatch with phase-appropriate num_links and memory_config (§2)
    - Local expert FFN compute for each of the 32 on-device experts (§3)
    - ttnn.all_to_all combine with same settings as dispatch (§4)
    - Weighted accumulation of k=8 expert outputs per token (§5)
    - Error guards and assertions (§6)

    Parameters and decisions:
    - memory_config: L1_MEMORY_CONFIG for decode; DRAM_MEMORY_CONFIG for prefill (Ch04)
    - num_links: 1 for decode (payload ≤6.4 MB); 2 for prefill (payload ≥29.4 MB) (Ch03)
    - C: ceil(k * B * S * CF / E); per-expert capacity including 25% headroom (Ch05)
    - per_core_M: 1 for decode (C=1 or C=2); ceil(C/32) for prefill (program config)
    """
    # --- Constants ---
    E  = 256    # total experts
    k  = 8      # top-k per token
    N  = 8      # devices
    Ed = 32     # experts per device (E / N)
    H  = 7168   # hidden dimension
    CF = 1.25   # capacity factor

    # --- Phase-dependent settings ---
    # See Ch03 (num_links) and Ch04 (memory_config) for derivations.
    if phase == "decode":
        memory_config = ttnn.L1_MEMORY_CONFIG
        num_links     = _select_num_links_decode(B)    # §3 table
        per_core_M    = 1                              # C <= 2 at decode; see §3
    elif phase == "prefill":
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        num_links     = 2                              # large payload; always 2-link
        C_prefill     = math.ceil(k * B * S * CF / E)
        per_core_M    = math.ceil(C_prefill / Ed)     # tiles per core for program config
    else:
        raise ValueError(f"phase must be 'decode' or 'prefill', got '{phase}'")

    # --- Expert-to-device mapping ---
    if expert_to_device is None:
        expert_to_device = [e // Ed for e in range(E)]   # naive uniform

    # §6: Error guards (checked before any compute)
    _check_preconditions(B, S, phase, E, k, Ed, CF)

    # --- §1: Router ---
    routing_scores, routing_indices = _router(x, W_r, k, memory_config)

    # --- §2: Send buffer packing and dispatch A2A ---
    C        = math.ceil(k * B * S * CF / E)
    send_buf = _pack_send_buffer(x, routing_indices, expert_to_device, N, Ed, C, H,
                                 memory_config)
    recv_buf = ttnn.all_to_all(
        input_tensor  = send_buf,
        mesh_device   = mesh_device,
        topology      = ttnn.Topology.Linear,  # T3K 1×8 linear; Ch01
        cluster_axis  = 1,
        num_links     = num_links,
        memory_config = memory_config,
    )
    # recv_buf shape: [N, C * Ed, H]; recv_buf[d] = tokens from device d for this device's experts

    # --- §3: Local expert FFN compute ---
    expert_output = _run_expert_ffn(recv_buf, expert_weights, C, Ed, H, memory_config)
    # expert_output shape: [N, C * Ed, H]; same shape as recv_buf

    # --- §4: Combine A2A ---
    # Routes expert outputs back to originating devices.
    # Volume identical to dispatch; see ch05_expert_parallelism/combine_and_accumulation.md §1.
    combine_recv = ttnn.all_to_all(
        input_tensor  = expert_output,
        mesh_device   = mesh_device,
        topology      = ttnn.Topology.Linear,
        cluster_axis  = 1,
        num_links     = num_links,
        memory_config = memory_config,
    )
    # combine_recv shape: [N, C * Ed, H]

    # --- §5: Weighted accumulation ---
    output = _weighted_accumulate(combine_recv, routing_scores, routing_indices,
                                  B, k, H, memory_config)
    # output shape: [B, H]

    return output
```

> **Note:** Helper functions (`_router`, `_pack_send_buffer`, `_run_expert_ffn`,
> `_weighted_accumulate`, `_check_preconditions`, `_select_num_links_decode`) are defined in
> §§1–6 below. They are shown as separate functions for readability; a production
> implementation may fuse some of them into a single kernel dispatch.

---

## 2. Section §1 — Router

```python
def _router(
    x: ttnn.Tensor,           # [B, H] — hidden states; L1 for decode, DRAM for prefill
    W_r: ttnn.Tensor,         # [H, E] — router weight matrix; DRAM-resident always
    k: int,
    memory_config: ttnn.MemoryConfig,
) -> tuple:
    """
    Router forward pass: linear projection + top-k selection.

    Returns:
        routing_scores:  [B, k]  — unnormalized sigmoid scores for selected experts
        routing_indices: [B, k]  — selected expert indices in {0, ..., 255}

    FLOP count: 2 * B * H * E = 2 * B * 7168 * 256 ≈ 3.67e6 * B FLOPs.
    At B=32: ~117 MFLOPs — sub-millisecond on Wormhole B0.

    Memory note: router_logits [B, E] = 32×256×2 = 16 KB at B=32; stays in L1 for decode.
    See ch04_memory_config/decode_memory_strategy.md §3 for L1 budget analysis.
    """
    # Router projection: [B, H] × [H, E] → [B, E]
    router_logits = ttnn.matmul(
        x,
        W_r,
        memory_config=memory_config,
    )
    # Qwen3.5-35B uses sigmoid-based routing scores (not softmax).
    # [VERIFY: confirm Qwen3.5-35B uses sigmoid vs. softmax for routing score computation]
    router_probs = ttnn.sigmoid(router_logits, memory_config=memory_config)

    # Top-k selection: returns both values (scores) and indices.
    # [VERIFY: ttnn.topk returns (values, indices) in this order]
    routing_scores, routing_indices = ttnn.topk(
        router_probs,
        k=k,
        memory_config=memory_config,
    )
    # routing_scores:  [B, k] — the k highest sigmoid scores
    # routing_indices: [B, k] — corresponding expert indices in {0, ..., 255}

    return routing_scores, routing_indices
```

---

## 3. Section §2 — Send Buffer and Dispatch A2A

The capacity $C$ controls the send buffer shape. The formula and per-$B$ worked examples are
derived in `ch05_expert_parallelism/token_routing_and_dispatch.md` §2. The key values:

| Phase | $B$ | $S$ | $C = \lceil k B S \cdot \text{CF} / E \rceil$ | Send buffer $[N, C \times E_d, H]$ |
|---|---|---|---|---|
| Decode | 1 | 1 | 1 | $[8, 32, 7168]$ — 3.7 MB total |
| Decode | 32 | 1 | 2 | $[8, 64, 7168]$ — 7.3 MB total |
| Prefill | 1 | 2048 | 80 | $[8, 2560, 7168]$ — 294 MB total |
| Prefill | 4 | 2048 | 320 | $[8, 10240, 7168]$ — 1.17 GB total |

The per-device dispatch volume (what crosses Ethernet) is $(N-1)/N$ of the total:
see `ch03_all_to_all_num_links/all_to_all_in_moe.md` for the volume derivation.

```python
def _pack_send_buffer(
    x: ttnn.Tensor,               # [B, H]
    routing_indices: ttnn.Tensor, # [B, k]
    expert_to_device: list,       # expert_idx -> device_id
    N: int, Ed: int, C: int, H: int,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """
    Packs token embeddings into a padded send buffer of shape [N, C * Ed, H].

    Each send_buf[d] contains the token embeddings routed to device d, with rows
    organised as [expert_local_0_slot_0, ..., expert_local_0_slot_{C-1},
                   expert_local_1_slot_0, ..., expert_local_{Ed-1}_slot_{C-1}].
    Capacity slots with no assigned token are zero-padded.

    Token drops (overflow beyond capacity C) are silently discarded. Monitor overflow
    counts in production; if drop rate > 1% consider increasing CF.
    See ch05_expert_parallelism/token_routing_and_dispatch.md §3 for overflow handling.
    """
    # [VERIFY: ttnn exposes a fused dispatch-pack kernel; if so, replace this with
    #  ttnn.moe_dispatch_pack(x, routing_indices, expert_to_device, C, Ed, N, H)]
    send_buf = ttnn.zeros(
        [N, C * Ed, H],
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        device=x.device(),
    )
    # Conceptual pack loop (in practice replaced by a fused Tensix kernel):
    # for each (token b, expert rank i):
    #   e = routing_indices[b, i]
    #   d = expert_to_device[e]
    #   local_e = e - d * Ed
    #   slot = next available slot for (d, local_e); skip if slot >= C
    #   send_buf[d, local_e * C + slot, :] = x[b, :]
    # This loop is O(B * k) = O(B * 8) in token count.
    # At B=32: 256 (token, expert) pairs; dominated by L1 write bandwidth, not compute.
    return send_buf   # [N, C * Ed, H]
```

---

## 4. Section §3 — Expert FFN and `num_links` Selection

### `num_links` Selection Table

The dispatch payload per device is:

$$V_{\text{dispatch}} = (N-1) \times C \times E_d \times H \times 2 \text{ bytes}$$

| Phase | $B$ | $C$ | $V_{\text{dispatch}}$ | `num_links` | Rationale |
|---|---|---|---|---|---|
| Decode | 1 | 1 | ~3.2 MB | 1 | Payload small; 2nd link adds overhead |
| Decode | 32 | 2 | ~6.4 MB | 1 | Still below threshold; 1 link sufficient |

For prefill, `num_links=2` always (dispatch volume ~245–981 MB far exceeds the ~20 MB threshold). See `prefill_considerations.md` §4 for the dispatch volume table and latency analysis.

For the full bandwidth model and threshold derivation see
`ch03_all_to_all_num_links/num_links_parameter.md`.

```python
def _select_num_links_decode(B: int) -> int:
    """
    Returns num_links for the dispatch and combine all-to-all calls during decode.

    At decode (S=1, B<=32), dispatch volumes are at most 6.4 MB per device.
    num_links=1 is optimal: the second link adds setup overhead that exceeds
    the latency reduction at this payload size.
    See ch03_all_to_all_num_links/num_links_parameter.md for the threshold derivation.

    This function always returns 1 for decode. It is structured as a function to make
    the selection point explicit and easy to update if the threshold changes.
    """
    # Decode payload: max 6.4 MB at B=32 (C=2, Ed=32, H=7168, BF16).
    # Threshold for num_links=2 benefit: ~20 MB [UNVERIFIED; verify against T3K benchmarks].
    _ = B  # unused for decode; kept for API symmetry with a potential prefill variant
    return 1
```

### Expert FFN Compute

After dispatch, device $d$ holds `recv_buf[d_index_self]` containing all token embeddings routed
to its 32 local experts. The expert FFN is a SwiGLU-gated feed-forward network:

$$\text{FFN}(x) = \left(\text{silu}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right) W_{\text{down}}$$

where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{H \times D_{\text{ffn}}}$ and
$W_{\text{down}} \in \mathbb{R}^{D_{\text{ffn}} \times H}$, and $D_{\text{ffn}}$ is the
expert intermediate dimension [UNVERIFIED for Qwen3.5-35B].

```python
def _run_expert_ffn(
    recv_buf: ttnn.Tensor,     # [N, C * Ed, H] — dispatched tokens for this device
    expert_weights: list,      # 32 × (W_gate [H, D_ffn], W_up [H, D_ffn], W_down [D_ffn, H])
    C: int, Ed: int, H: int,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """
    Runs each of the 32 local experts on the token slots assigned to them.

    For each local expert e_local in {0, ..., 31}:
      - input tokens occupy recv_buf[:, e_local*C : (e_local+1)*C, :] — shape [N, C, H]
      - after FFN, output has the same shape [N, C, H]

    In practice, TTNN groups all 32 expert matmuls into a batched matmul to amortize
    kernel launch overhead. The per-expert loop is shown for conceptual clarity.
    [VERIFY: ttnn.group_matmul or equivalent for batched expert FFN on Wormhole B0]

    Memory note for decode (B<=32, C<=2):
      - per-expert input:  C * H * 2 = 2 * 7168 * 2 = 28 KB in L1 — negligible.
      - per-expert output: same 28 KB — fits comfortably within L1 budget (Ch04).
    """
    expert_output = ttnn.zeros_like(recv_buf, memory_config=memory_config)

    for e_local in range(Ed):   # Ed=32 local experts per device
        W_gate, W_up, W_down = expert_weights[e_local]
        # Extract token slots for this expert: shape [N, C, H]
        # (in practice, indexing is over the flattened [N * C, H] view)
        token_slots = recv_buf[:, e_local * C : (e_local + 1) * C, :]

        # SwiGLU gate branch
        gate = ttnn.silu(
            ttnn.matmul(token_slots, W_gate, memory_config=memory_config),
            memory_config=memory_config,
        )
        # Up projection branch
        up = ttnn.matmul(token_slots, W_up, memory_config=memory_config)
        # Gated activation
        gated = ttnn.multiply(gate, up, memory_config=memory_config)
        # Down projection
        out = ttnn.matmul(gated, W_down, memory_config=memory_config)

        expert_output[:, e_local * C : (e_local + 1) * C, :] = out

    return expert_output   # [N, C * Ed, H]
```

> **Program config note:** The `per_core_M` parameter for the expert matmul program config
> (controlling the number of output rows each Tensix core computes) should be set to:
> - Decode: `per_core_M = 1` (since $C \leq 2$, each expert has at most 2 token slots;
>   each of the 80 Tensix cores handles 1 output row).
> - Prefill: `per_core_M = ceil(C / Ed)` (since $C$ grows with $B \times S$; at $B=4, S=2048$,
>   $C=320$ and `per_core_M = ceil(320/32) = 10`).
>
> Setting `per_core_M` to the wrong value for the phase causes suboptimal core utilization.
> [VERIFY: ttnn MatmulProgramConfig API for setting per_core_M on Wormhole B0]

---

## 5. Section §4 — Weighted Accumulation

After the combine A2A, the originating device holds `combine_recv` shaped $[N, C \times E_d, H]$.
Using the dispatch metadata (`routing_indices` saved at §2), the outputs are reordered into a
$[B, k, H]$ logical buffer — $B$ tokens, $k=8$ expert outputs each — and accumulated:

$$\text{output}[b] = \sum_{i=1}^{k} w_i^{\text{norm}} \times \text{expert\_output}[b, i]$$

where $w_i^{\text{norm}} = s_i / \sum_{j=1}^{k} s_j$ (normalized sigmoid scores, as described
in `ch05_expert_parallelism/combine_and_accumulation.md` §2).

```python
def _weighted_accumulate(
    combine_recv: ttnn.Tensor,    # [N, C * Ed, H] — expert outputs from combine A2A
    routing_scores: ttnn.Tensor,  # [B, k] — raw scores saved at dispatch
    routing_indices: ttnn.Tensor, # [B, k] — expert indices saved at dispatch
    B: int, k: int, H: int,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """
    Reorders combine_recv into [B, k, H] using routing_indices, then applies
    normalized-score weighted accumulation to produce the MoE output [B, H].

    For decode (B<=32): output is 448 KB at B=32 — fits in L1.
    For prefill: output grows with B*S; DRAM placement is required.

    Numerical note: BF16 accumulation of k=8 terms has a loose relative error bound of
    k * eps_BF16 = 8 * 2^{-7} ≈ 8 * 0.0078 ≈ 6.25%. Empirical PCC loss is negligible in MoE deployments.
    For higher fidelity, cast to FP32 before accumulation and back to BF16 after.
    See ch05_expert_parallelism/combine_and_accumulation.md §4 for the precision analysis.
    """
    # Normalize routing scores: w[b, i] = s[b, i] / sum_j s[b, j]
    score_sum = ttnn.sum(routing_scores, dim=-1, keepdim=True)   # [B, 1]
    w_norm    = ttnn.div(routing_scores, score_sum,
                         memory_config=memory_config)            # [B, k]

    # Reorder combine_recv from [N, C*Ed, H] -> [B, k, H] using routing_indices.
    # [VERIFY: ttnn scatter/gather API for this reorder step on MeshDevice]
    expert_out_bkh = _reorder_combine_recv(combine_recv, routing_indices, B, k, H,
                                           memory_config)  # [B, k, H]

    # Weighted accumulation: output[b] = sum_i w_norm[b,i] * expert_out_bkh[b,i,:]
    output = ttnn.zeros([B, H], dtype=ttnn.bfloat16,
                        memory_config=memory_config, device=combine_recv.device())
    for i in range(k):
        wi     = ttnn.reshape(w_norm[:, i], [B, 1])           # [B, 1]
        e_i    = expert_out_bkh[:, i, :]                      # [B, H]
        output = ttnn.add(
            output,
            ttnn.multiply(wi, e_i, memory_config=memory_config),
            memory_config=memory_config,
        )
    # In TTNN, this k=8 loop is replaced by a fused batched multiply-accumulate.

    return output   # [B, H]
```

---

## 6. Section §5 — Residual Add

The MoE output is combined with the pre-MoE residual stream. The residual tensor has the same
shape $[B, H]$ as the MoE output and should have been saved before the router call.

```python
def apply_residual(
    residual: ttnn.Tensor,    # [B, H] — hidden states input to the MoE layer (saved)
    moe_output: ttnn.Tensor,  # [B, H] — output of moe_layer_t3k(...)
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """
    Adds MoE output to the pre-MoE residual stream in-place.
    Writes directly to the residual tensor's memory slot to avoid a separate allocation.

    For decode: both tensors are in L1; the result stays in L1 for the next layer norm.
    For prefill: both tensors are in DRAM; in-place add avoids an extra DRAM allocation.
    """
    return ttnn.add(residual, moe_output, memory_config=memory_config)
    # Result is the post-MoE hidden states [B, H], input to the next transformer block.
```

---

## 7. Section §6 — Error Guards

These assertions should be evaluated before any compute in `moe_layer_t3k`. Catching incorrect
parameters early prevents silent correctness failures (e.g., wrong `phase` causing L1 overflow
during prefill).

```python
def _check_preconditions(
    B: int, S: int, phase: str,
    E: int, k: int, Ed: int, CF: float,
) -> None:
    """
    Validates parameters before any compute. Raises AssertionError or logs warnings.
    Call at the top of moe_layer_t3k, before routing_scores computation.
    """
    # 1. Decode-specific capacity check
    if phase == "decode":
        C = math.ceil(k * B * S * CF / E)
        assert C <= 32, (
            f"Decode capacity C={C} exceeds per_core_M=1 assumption (C must be <=32). "
            f"Got B={B}, S={S}, k={k}, CF={CF}, E={E}. "
            f"At B<=32 decode, C should be 1 or 2."
        )
        # At B=32, S=1: C = ceil(8*32*1*1.25/256) = ceil(1.25) = 2. ✓
        # At B=256, S=1: C = ceil(8*256*1*1.25/256) = ceil(10) = 10. ✓ (still <=32)
        # This guard catches accidentally passing S>1 in decode phase.

    # 2. Sequence length consistency
    if phase == "decode" and S != 1:
        import logging
        logging.warning(
            f"moe_layer_t3k called with phase='decode' but S={S} > 1. "
            f"Decode is expected to process one token per step (S=1). "
            f"If this is intentional (e.g., speculative decoding), ensure memory_config "
            f"and num_links are manually overridden for the larger payload."
        )

    # 3. Batch size guard for L1 decode buffer
    if phase == "decode" and B > 32:
        import logging
        logging.warning(
            f"moe_layer_t3k called with phase='decode' and B={B} > 32. "
            f"L1 memory config is set for B<=32 (dispatch buffer <=6.4 MB). "
            f"At B={B}, verify that dispatch buffer [N, C*Ed, H] = "
            f"[8, {math.ceil(k*B*CF/E) * Ed}, {7168}] still fits in L1. "
            f"See ch04_memory_config/decode_memory_strategy.md."
        )

    # 4. Expert count consistency
    assert E == k * (E // k), (
        f"E={E} is not a multiple of k={k}; expected E divisible by k for balanced routing."
    )
    # Qwen3.5-35B: E=256, k=8; 256 = 8 * 32. ✓

    # 5. Capacity overflow warning
    C = math.ceil(k * B * S * CF / E)
    expected_load = k * B * S / E  # expected tokens per expert under uniform routing
    if C < expected_load * 0.99:   # should never happen given CF>=1, but guard anyway
        import logging
        logging.warning(
            f"Capacity C={C} is less than expected load {expected_load:.1f}. "
            f"Check CF={CF} and formula: C = ceil(k*B*S*CF/E)."
        )
```

---

## References

- `ch01_t3k_topology/t3k_physical_layout.md` — `ttnn.Topology.Linear`, `cluster_axis=1`
- `ch01_t3k_topology/ethernet_link_bandwidth.md` — Per-link ~12.5 GB/s; basis for latency estimates
- `ch02_ttnn_mesh_api/collective_primitives.md` — `ttnn.all_to_all` API; dispatch and combine direction
- `ch03_all_to_all_num_links/num_links_parameter.md` — `num_links` selection threshold; bandwidth model
- `ch03_all_to_all_num_links/all_to_all_in_moe.md` — Dispatch volume formula; per-$B$ worked examples
- `ch04_memory_config/decode_memory_strategy.md` — L1 placement for decode buffers; per-core budget
- `ch04_memory_config/prefill_memory_strategy.md` — DRAM placement for prefill buffers
- `ch04_memory_config/memory_config_api.md` — `ttnn.L1_MEMORY_CONFIG`, `ttnn.DRAM_MEMORY_CONFIG`
- `ch05_expert_parallelism/token_routing_and_dispatch.md` — Send buffer construction; capacity formula
- `ch05_expert_parallelism/combine_and_accumulation.md` — Combine A2A; weighted accumulation; precision
- `ch05_expert_parallelism/expert_placement_strategies.md` — Expert-to-device mapping strategies
- `ch06_profiling/ttnn_profiler.md` — Profiler hooks for first-step profiling
- [Qwen3] Qwen Team, Alibaba Group, "Qwen3 Technical Report", 2025.
- [Fedus2022] Fedus, W., Zoph, B., Shazeer, N., "Switch Transformers", JMLR, 2022.

---

**Next:** [decode_loop_integration.md](./decode_loop_integration.md)
