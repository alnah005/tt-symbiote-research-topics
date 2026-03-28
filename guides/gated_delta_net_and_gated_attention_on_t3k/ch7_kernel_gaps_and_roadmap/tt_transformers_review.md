# tt-transformers Review

This section audits the existing tt-transformers and tt-symbiote model code for patterns relevant to the hybrid Gated Delta Net + Gated Attention forward pass. All file paths are relative to the tt-metal repository root.

---

## 1. DeltaNet Implementation — `TTNNQwen3LinearAttention`

**File:** `models/experimental/tt_symbiote/modules/qwen_attention.py`

`TTNNQwen3LinearAttention` implements one Gated Delta Net layer. Its current on-device / off-device split is:

| Step | On-device (TTNN) | Off-device (PyTorch/CUDA) |
|---|---|---|
| `in_proj_qkv` | `ttnn.linear`, col-sharded | — |
| `in_proj_z` | `ttnn.linear`, col-sharded | — |
| `in_proj_a`, `in_proj_b` | — | `nn.Linear` on host |
| All-gather (QKV + Z) | `ttnn.experimental.all_gather_async` | — |
| Causal conv1d (prefill) | — | `causal_conv1d_fn` (CUDA C extension) |
| Causal conv1d (decode) | — | `causal_conv1d_update` (CUDA C extension) |
| β, α, g computation | — | PyTorch elementwise on host |
| K/Q head repeat-interleave | — | PyTorch on host |
| Recurrent delta rule (decode) | — | `recurrent_gated_delta_rule` (Triton/CUDA) |
| Chunkwise delta rule (prefill) | — | `chunk_gated_delta_rule` (Triton/CUDA) |
| Gated RMSNorm + swish gate | — | `FusedRMSNormSwishGate` (PyTorch custom) |
| `out_proj` | `ttnn.linear`, row-parallel + all-gather | — |

**Source libraries:** The DeltaNet kernels (`recurrent_gated_delta_rule`, `chunk_gated_delta_rule`) come from `fla-org/flash-linear-attention`, a Triton-optimized library for linear attention variants. The causal conv1d comes from the `causal-conv1d` package. Both require a CUDA GPU for execution.

**State management:** `recurrent_states[layer_idx]` is a PyTorch tensor (not a TTNN tensor) stored on the host-accessible device. Moving it to a TTNN on-device tensor is part of the porting work for Priority 1 (see `development_roadmap.md`).

---

## 2. Gated Attention Implementation — `TTNNQwen3FullAttention`

**File:** `models/experimental/tt_symbiote/modules/qwen_attention.py`

`TTNNQwen3FullAttention` implements one Gated Attention layer. It is substantially complete in TTNN:

| Step | On-device (TTNN) | Off-device |
|---|---|---|
| Q+gate projection | `ttnn.linear`, col-sharded | — |
| KV projection | `ttnn.linear`, replicated weights | — |
| gate_sigmoid = σ(gate) | `ttnn.sigmoid` | — |
| Q_gated = Q ⊙ gate_sigmoid | `ttnn.mul` | — |
| Q_norm = RMSNorm(Q_gated) | `ttnn.rms_norm` | — |
| K_norm = RMSNorm(K) | `ttnn.rms_norm` | — |
| RoPE | `ttnn.experimental.rotary_embedding` | — |
| KV cache write | `TTNNQwenPagedAttentionKVCache` | — |
| GQA KV expand (2→16 heads) | `ttnn.repeat_interleave` | — |
| SDPA prefill | `ttnn.transformer.scaled_dot_product_attention` | — |
| SDPA decode | `ttnn.transformer.scaled_dot_product_attention_decode` | — |
| `out_proj` + all-gather | `ttnn.linear` + `ttnn.experimental.all_gather_async` | — |

No off-device fallback exists for Gated Attention. This is the reference implementation for what a fully on-device layer looks like.

---

## 3. Cache Management — `TTNNQwenPagedAttentionKVCache`

**File:** `models/experimental/tt_symbiote/modules/qwen_attention.py`

`TTNNQwenPagedAttentionKVCache` manages two parallel cache structures:

- **Paged KV cache:** Used by Gated Attention layers. Implements paged virtual memory for K and V tensors, supporting arbitrary-length context without pre-allocating the full maximum sequence length. Operates entirely in TTNN on-device.

- **Recurrent state dict (`recurrent_states`):** A Python dictionary mapping `layer_idx → torch.Tensor`. Stores the Gated Delta Net state `S ∈ [B, H_v, d_k, d_v]` as a PyTorch tensor on host. This is the primary state transfer bottleneck: each decode step must round-trip the state tensor through the host for the PyTorch `recurrent_gated_delta_rule` kernel.

- **Conv state dict (`conv_states`):** A Python dictionary mapping `layer_idx → torch.Tensor`. Stores the causal conv1d state `[B, 8192, 4]` on host. Also incurs a host round-trip per decode step.

The abstraction structure is correct: `TTNNQwenPagedAttentionKVCache` correctly tracks which layers use which state type. Migrating `recurrent_states` and `conv_states` from PyTorch tensors to on-device TTNN tensors is a necessary step before the new Metalium kernels can read and write state directly from DRAM without involving the host.

---

## 4. Flash-Linear-Attention Library Dependency

**Repository:** `fla-org/flash-linear-attention`

This library provides Triton-optimized CUDA kernels for a family of linear attention variants including Gated Delta Net. The specific functions used:

| Function | Purpose |
|---|---|
| `recurrent_gated_delta_rule` | One-step recurrent update of the state matrix S for all heads; decode path |
| `chunk_gated_delta_rule` | Chunkwise WY-decomposition prefill; processes T tokens in chunks of size C=64 |

Both functions are GPU-only (require CUDA). They are invoked after data has been transferred from TTNN device tensors to PyTorch CPU/GPU tensors and after computation results have been transferred back. This round-trip adds latency and prevents the state from remaining in on-chip SRAM.

No Metalium equivalent of either function currently exists in the tt-metal repository. The recommended porting path is described in `development_roadmap.md` Priority 1 and Priority 5.

---

## 5. tt-transformers Coverage

The tt-transformers package provides TTNN-accelerated implementations of Llama-family and other standard transformer models. As of the time of writing:

- No `delta_net`, `gated_delta`, `linear_attention`, or `gla` entries exist in tt-transformers model modules.
- The hybrid architecture (mixing recurrent and softmax attention layers) is handled entirely within tt-symbiote (`models/experimental/tt_symbiote/`), not in tt-transformers.
- The `TTNNQwen3FullAttention` and `TTNNQwen3LinearAttention` classes in tt-symbiote are the most relevant existing implementations and are not duplicated in tt-transformers.

---

**Previous:** [`existing_ttnn_primitives_survey.md`](./existing_ttnn_primitives_survey.md) | **Next:** [`development_roadmap.md`](./development_roadmap.md)
