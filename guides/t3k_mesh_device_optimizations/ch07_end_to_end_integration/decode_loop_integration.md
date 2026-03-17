# Decode Loop Integration

> **Quick Reference — TTNN API Symbols Used**
>
> | Symbol | Description |
> |---|---|
> | `ttnn.L1_MEMORY_CONFIG` | L1 SRAM placement for decode-phase buffers; see `ch04_memory_config/memory_config_api.md` |
> | `ttnn.DRAM_MEMORY_CONFIG` | DRAM placement for KV cache and expert weights; always DRAM-resident |
> | `ttnn.all_to_all` | Dispatch and combine collectives; see `ch02_ttnn_mesh_api/collective_primitives.md` |

This file shows how to integrate `moe_layer_t3k` (defined in `complete_moe_layer_impl.md`) into
the standard autoregressive decode loop for Qwen3.5-35B on T3K. Sections cover the full decode
loop structure, memory lifecycle management, KV cache interaction, batch padding conventions,
performance targets, and profiling discipline.

All values use Qwen3.5-35B constants: $E=256$, $k=8$, $N=8$, $E_d=32$, $H=7168$, CF$=1.25$.

---

## 1. Standard Autoregressive Decode Loop

The decode loop generates one new token per step for each of the $B$ sequences in the batch.
At each step, the model runs a full forward pass over the $B$ current-position token embeddings,
produces logits $[B, V]$, and samples the next token for each sequence.

The MoE layer is called once per transformer block that uses MoE (as opposed to dense FFN blocks).
Qwen3.5-35B uses MoE in alternating or all-FFN blocks [VERIFY: confirm which layers are MoE vs.
dense in Qwen3.5-35B architecture].

```python
import ttnn
import torch
import math

def autoregressive_decode(
    model_params: dict,       # expert_weights, router_weights, attn_weights, etc.
    mesh_device: ttnn.MeshDevice,
    input_ids: torch.Tensor,  # [B, S_prompt] — the prefilled prompt token IDs
    max_new_tokens: int,
    B: int,                   # batch size; padded to 32 if necessary (see §4)
) -> torch.Tensor:            # [B, max_new_tokens] — generated token IDs
    """
    Autoregressive decode loop for Qwen3.5-35B on T3K.

    Memory lifecycle:
    - Decode memory configs (L1) are activated once before the loop (§2).
    - Expert weights remain DRAM-resident across all decode steps (§2).
    - KV cache grows each step; always DRAM (§3).
    - MoE activation tensors (hidden states, A2A buffers) are reallocated each step
      in L1 by moe_layer_t3k; L1 is freed after each step by TTNN's program cache.

    Profile the first 10 steps, then disable profiling for production (§5).
    """
    # --- Initialization ---
    hidden_states = _embed_tokens(input_ids[:, -1], model_params)  # [B, H] — last prompt token
    kv_cache      = _init_kv_cache(B, model_params, mesh_device)   # DRAM; see §3
    generated     = []
    profile_steps = 10  # profile first N steps; see §5

    for step in range(max_new_tokens):
        # --- Optional profiling ---
        if step < profile_steps:
            ttnn.enable_program_cache_profiling()  # [VERIFY: correct TTNN profiler API]
        elif step == profile_steps:
            ttnn.disable_program_cache_profiling()  # disable after warmup; see §5

        # --- Transformer layers ---
        for layer_idx, layer_params in enumerate(model_params["layers"]):
            # Attention (self-attention with KV cache)
            hidden_states = _attention_layer(
                hidden_states, kv_cache[layer_idx], layer_params["attn"], step,
                mesh_device,
            )

            # MoE FFN (or dense FFN for non-MoE layers [VERIFY])
            if layer_params.get("is_moe", True):
                moe_out = moe_layer_t3k(
                    x             = hidden_states,
                    W_r           = layer_params["router_weight"],
                    expert_weights= layer_params["expert_weights"],
                    mesh_device   = mesh_device,
                    B             = B,
                    phase         = "decode",
                    S             = 1,
                )
                hidden_states = apply_residual(hidden_states, moe_out,
                                               memory_config=ttnn.L1_MEMORY_CONFIG)

            # Layer norm
            hidden_states = _layer_norm(hidden_states, layer_params["norm"],
                                        memory_config=ttnn.L1_MEMORY_CONFIG)

        # --- Sample next token ---
        logits = _lm_head(hidden_states, model_params["lm_head"])  # [B, V]
        next_tokens = _sample(logits)                               # [B]
        generated.append(next_tokens)

        # --- Update for next step ---
        hidden_states = _embed_tokens(next_tokens, model_params)   # [B, H] for next step

    return torch.stack(generated, dim=1)   # [B, max_new_tokens]
```

---

## 2. Memory Lifecycle Management

### Principle: Activate Decode Memory Configs Once

All activations and A2A buffers for the decode phase use `ttnn.L1_MEMORY_CONFIG`. These configs
are set once at the start of the decode loop, not re-set at each step. The TTNN program cache
compiles operations with the correct `memory_config` on first execution and reuses compiled
programs for subsequent steps without recompilation overhead.

```python
def init_decode_memory_configs(mesh_device: ttnn.MeshDevice) -> None:
    """
    Activates L1 memory configurations for all decode-phase tensors.
    Call once before the first decode step.

    Do NOT call this function between decode steps — it is a one-time initialization.
    Calling it repeatedly may flush the program cache and trigger recompilation.
    [VERIFY: whether repeated ttnn.L1_MEMORY_CONFIG usage triggers recompilation]
    """
    # L1_MEMORY_CONFIG is a global constant in TTNN; no per-device activation needed.
    # This function is a documentation anchor, not a runtime call.
    # Its purpose is to make the lifecycle decision explicit in code review.
    pass
```

### Expert Weights: DRAM-Resident Across All Steps

Expert weights ($W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$ for each of the 32 local
experts) are loaded once at model initialization and remain DRAM-resident throughout all decode
steps. They are large and persistent:

$$\text{Expert weight size per device} \approx 3 \times E_d \times H \times D_{\text{ffn}} \times 2 \text{ bytes}$$

At $E_d=32$, $H=7168$, and with $D_{\text{ffn}}$ [UNVERIFIED for Qwen3.5-35B], this is several
GB per device — far too large for L1. Loading expert weights from DRAM at each decode step is
expected and unavoidable; this is the memory-bandwidth bottleneck of the expert FFN at small
batch sizes.

```python
def load_expert_weights(
    model_path: str,
    mesh_device: ttnn.MeshDevice,
) -> list:
    """
    Loads the 32 local expert weights for this device from checkpoint.
    Places all weight tensors in DRAM_MEMORY_CONFIG — they are too large for L1.

    Returns a list of 32 (W_gate, W_up, W_down) triples.
    This function is called ONCE at model initialization; the returned list is passed
    to moe_layer_t3k at each decode step.

    Expert weights do not change during inference. Do not reload between steps.
    """
    expert_weights = []
    for e_local in range(32):
        W_gate = ttnn.from_torch(
            _load_tensor(model_path, f"expert_{e_local}.gate"),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
        )
        W_up = ttnn.from_torch(
            _load_tensor(model_path, f"expert_{e_local}.up"),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
        )
        W_down = ttnn.from_torch(
            _load_tensor(model_path, f"expert_{e_local}.down"),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
        )
        expert_weights.append((W_gate, W_up, W_down))
    return expert_weights
```

---

## 3. KV Cache Integration

The attention layer reads and writes the KV cache at each decode step. The KV cache is separate
from the MoE layer and is always DRAM-resident (see `ch04_memory_config/decode_memory_strategy.md`
§2 for the size derivation). The MoE layer receives the **post-attention residual stream** as its
input `x`.

The interaction between attention and MoE in the decode loop:

```
Step t:
  hidden_states [B, H]                         (L1)
        │
        ▼
  Attention + KV cache update                  (KV cache: DRAM; Q,K,V: L1)
        │
        ▼
  hidden_states [B, H]  (post-attention)       (L1)
        │
        ▼
  moe_layer_t3k(x=hidden_states, phase="decode")
  ┌─────────────────────────────────────────┐
  │ Router [B,H]×[H,256] → scores, indices  │  (L1)
  │ Pack send buffer [8, C*32, 7168]        │  (L1)
  │ ttnn.all_to_all (dispatch, num_links=1) │  (Ethernet)
  │ Expert FFN × 32 experts                 │  (L1 activations; DRAM weights)
  │ ttnn.all_to_all (combine, num_links=1)  │  (Ethernet)
  │ Weighted accumulate → [B, H]            │  (L1)
  └─────────────────────────────────────────┘
        │
        ▼
  apply_residual(hidden_states, moe_out)       (L1)
        │
        ▼
  layer_norm → hidden_states [B, H]            (L1)
        │
        ▼
  (next transformer block)
```

The key invariant: `hidden_states` is always in L1 throughout the decode step. The KV cache
in DRAM is accessed only during the attention computation and does not interact with MoE buffers.
There is no memory config conflict between the two.

---

## 4. Batch Padding to B=32

### Rationale

`moe_layer_t3k` uses TTNN programs (compiled kernels) that have shapes baked in at compilation.
Changing $B$ between steps would require recompiling the program. To avoid recompilation overhead
when the actual number of live sequences $B_{\text{live}} < 32$, pad the batch to a fixed size
$B_{\text{padded}} = 32$ throughout the decode loop.

```python
def pad_batch_to_32(
    hidden_states: ttnn.Tensor,      # [B_live, H]
    live_mask: torch.Tensor,         # [B_live] — True for active sequences
    B_padded: int = 32,
) -> tuple:
    """
    Pads the hidden state batch to B_padded=32 with zero rows for inactive sequences.

    Returns:
        padded_hidden:  [B_padded, H] — zero-padded
        padded_mask:    [B_padded] — True for live rows, False for pad rows

    Pad tokens do not contribute to routing or expert outputs because:
    - Their hidden states are zero-padded.
    - Zero hidden states produce near-zero router logits → near-zero routing scores.
    - Even if a pad token is "routed" to an expert, its contribution to the accumulation
      is multiplied by a near-zero weight → negligible impact on live token outputs.

    For hard guarantees, mask out pad positions before accumulation in _weighted_accumulate.
    [VERIFY: whether TTNN top-k on zero inputs produces deterministic expert assignments]
    """
    B_live = hidden_states.shape[0]
    if B_live == B_padded:
        return hidden_states, live_mask

    pad_rows = B_padded - B_live
    padding  = ttnn.zeros([pad_rows, hidden_states.shape[1]],
                          dtype=ttnn.bfloat16,
                          memory_config=ttnn.L1_MEMORY_CONFIG,
                          device=hidden_states.device())
    padded_hidden = ttnn.concat([hidden_states, padding], dim=0,
                                memory_config=ttnn.L1_MEMORY_CONFIG)
    padded_mask   = torch.cat([live_mask, torch.zeros(pad_rows, dtype=torch.bool)])

    return padded_hidden, padded_mask
```

> **Warning:** Pad rows must not contribute to any sampled output. Apply `padded_mask` before
> the `_lm_head` and `_sample` steps to restrict output generation to live rows only. Routing
> pad tokens to experts is harmless for live-token output quality but wastes expert capacity
> slots. At $B=32$ with zero padding, $C=2$ accommodates this; no capacity overflow is expected.

---

## 5. Performance Targets and Profiling Discipline

### Decode Performance Targets

For Qwen3.5-35B on T3K at $B=32$:

| Component | Estimated Time | Notes |
|---|---|---|
| Dispatch A2A | ~0.51 ms | $V=6.4$ MB at 12.5 GB/s, `num_links=1`; see `ch05_expert_parallelism/token_routing_and_dispatch.md` §4 |
| Combine A2A | ~0.51 ms | Same volume as dispatch |
| Expert FFN (32 experts × 1–2 token slots) | [UNVERIFIED] | Depends on $D_{\text{ffn}}$ and Wormhole B0 matmul throughput |
| Router matmul | Sub-ms | ~117 MFLOPs at $B=32$ |
| Total MoE layer | ~1.0 ms + expert FFN | Communication-bound at $B \leq 32$ |

The decode regime is **communication-bound**: Tensix cores are often idle waiting for A2A
transfers to complete. Increasing `num_links` to 2 at $B=32$ is not beneficial because the
payload (6.4 MB) is below the threshold where a second link provides a net latency reduction
(estimated threshold ~20 MB [UNVERIFIED]). See `ch06_profiling/bottleneck_diagnosis_guide.md` for
the decision procedure to confirm whether a given workload is indeed communication-bound.

### Profiling Discipline

Profile the first 10 decode steps. Do not disable earlier: the first step incurs program
compilation overhead that inflates latency readings and is not representative of steady-state
performance.

```python
# Profiling schedule for decode loop
PROFILE_WARMUP_STEPS = 10  # compile + warmup
# Steps 0–9:  profiling enabled — capture timing for all ops
# Step 10+:   profiling disabled — production mode

def should_profile(step: int) -> bool:
    return step < PROFILE_WARMUP_STEPS

# In the decode loop:
# if should_profile(step):
#     # parse ttnn profiler CSV after the run; see ch06_profiling/ttnn_profiler.md §3
#     pass
```

After profiling, parse the TTNN profiler CSV output to confirm:

1. The top-latency operations are `ttnn.all_to_all` (dispatch and combine) — expected for decode.
2. The expert FFN matmuls account for a small fraction of total step time.
3. Ethernet link utilization is high during A2A operations (confirming communication-bound regime).

If the profile shows that expert FFN matmuls dominate (not A2A), the workload may have transitioned
to a compute-bound regime — investigate with `ch06_profiling/device_perf_counters.md`.

---

## 6. Common Integration Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Calling `moe_layer_t3k` with `phase="prefill"` inside the decode loop | L1 OOM error or DRAM placement mismatch for small tensors | Ensure `phase="decode"` is hardcoded in the decode loop |
| Loading expert weights inside the decode loop instead of at init | Decode step latency 10–100× too high; DRAM bandwidth saturated | Load all expert weights once at `load_expert_weights` call site; pass the list to each step |
| Forgetting to pad `B_live < 32` to `B_padded = 32` | TTNN program cache miss on every step with different $B$; recompilation at each step | Use `pad_batch_to_32` before every `moe_layer_t3k` call |
| Not saving `hidden_states` as residual before `moe_layer_t3k` | MoE output added to itself instead of the pre-MoE residual | Save `residual = hidden_states` before the `moe_layer_t3k` call; use as input to `apply_residual` |
| Profiling every step in production | ~5–10% throughput overhead from profiler instrumentation | Use `should_profile(step)` guard; disable after warmup |

---

## References

- `complete_moe_layer_impl.md` — `moe_layer_t3k` function; `_check_preconditions`; `apply_residual`
- `ch02_ttnn_mesh_api/collective_primitives.md` — `ttnn.all_to_all` API
- `ch04_memory_config/decode_memory_strategy.md` — L1 placement for decode activations and A2A buffers
- `ch04_memory_config/wormhole_memory_hierarchy.md` — L1 and DRAM capacities; DRAM bandwidth ~300 GB/s [UNVERIFIED]
- `ch05_expert_parallelism/token_routing_and_dispatch.md` — Dispatch volume 6.4 MB at $B=32$; latency ~0.51 ms
- `ch05_expert_parallelism/combine_and_accumulation.md` — Combine volume = dispatch volume; overlap with next layer
- `ch06_profiling/ttnn_profiler.md` — Profiler setup, CSV parsing, Tracy visualization
- `ch06_profiling/bottleneck_diagnosis_guide.md` — Decode bottleneck categorization; remediation for communication-bound regime
- `prefill_considerations.md` (this chapter) — Differences for prefill phase

---

**Next:** [prefill_considerations.md](./prefill_considerations.md)
