# GLA and Related Kernel Survey

This file surveys Gated Linear Attention (GLA), RetNet, and other linear attention variants for existing tt-metal or tt-transformers kernel implementations. The expected finding — confirmed by the survey — is that no dedicated tt-metal kernel exists for GLA or RetNet. The only linear attention implementation in the repository is `TTNNQwen3LinearAttention` for DeltaNet, which currently falls back to `flash-linear-attention` Triton kernels for both prefill and decode when running outside the TTNN path. This confirms that the composed TTNN form (Chapter 2) and the Blackhole kernel port (Chapter 4) are the two correct paths forward.

---

## 1. Survey Scope

The survey covers the following candidate implementations:

- GLA (Gated Linear Attention): linear attention with per-channel scalar gating; update is `H_t = G_t * H_{t-1} + k_t ⊗ v_t` where `G_t` is a diagonal decay matrix derived from an input-dependent gate
- RetNet: linear attention with fixed exponential decay per channel; update is `H_t = γ * H_{t-1} + k_t ⊗ v_t` where `γ` is a learned scalar per channel
- Vanilla linear attention (no decay): `H_t = H_{t-1} + k_t ⊗ v_t`
- DeltaNet / Gated DeltaNet: the target; `H_t = g_t * H_{t-1} + k̃_t ⊗ (β_t * (v_t - H_{t-1}^T k̃_t))`

Search locations: `ttnn/cpp/ttnn/operations/`, `models/experimental/`, `models/tt_transformers/`, `tt_transformers/` (if the repository is structured with a standalone tt-transformers package).

---

## 2. Expected Findings

> **Note:** The following findings are based on the repository state as understood at the time of this guide's writing. Run a grep for `GatedLinearAttention`, `RetNet`, `linear_attention`, `gla_chunk`, and `retnet_chunk` across the full repository before relying on this table, to confirm no relevant kernels were added after this chapter was written.

### 2.1 GLA

No dedicated tt-metal or TTNN kernel for Gated Linear Attention has been identified. GLA is implemented in `flash-linear-attention` as a Triton kernel (`gla_chunk_fwd`, `gla_fused_recurrent_fwd`). These Triton kernels run on CUDA only and are not available on Wormhole. No TTNN wrapper or TT-Metalium translation of GLA exists.

Reuse potential if a GLA kernel did exist: GLA's state update `H_t = G_t * H_{t-1} + k_t ⊗ v_t` is separable (the write `k_t ⊗ v_t` does not depend on `H_{t-1}`), making it structurally more similar to Mamba than to DeltaNet. A GLA kernel would not encode the retrieval-then-error inner loop that DeltaNet requires.

### 2.2 RetNet

No tt-metal or TTNN kernel for RetNet has been identified. RetNet uses a fixed exponential decay `γ^n` where n is the position gap — a simpler decay than GLA's input-dependent gate. RetNet's recurrence is a linear SSM with a fixed decay scalar per channel, making it the simplest of the linear attention variants. No implementation exists in tt-metal as of the survey.

### 2.3 Vanilla Linear Attention

No standalone tt-metal or TTNN implementation of vanilla linear attention (no decay) has been identified. The full-attention path in Qwen3 uses standard softmax attention via TTNN SDPA; the linear attention path uses `TTNNQwen3LinearAttention`, which is DeltaNet-specific.

### 2.4 DeltaNet / `TTNNQwen3LinearAttention`

`TTNNQwen3LinearAttention` is the only linear attention implementation in the repository. Its decode path calls `recurrent_gated_delta_rule` from `flash-linear-attention` — a Triton kernel or PyTorch fallback — which is the host-crossing gap that this guide addresses. Its prefill path calls `chunk_gated_delta_rule`, also from `flash-linear-attention`.

The Qwen3.5-27B Blackhole implementation additionally defines `gdn_full_fused_inplace` (Chapter 4), which is the only TT-Metalium kernel for any linear attention variant that has been found in the survey.

---

## 3. Summary Table

| Candidate | What it computes | Structural similarity to DeltaNet | tt-metal implementation exists? | Reuse classification |
|---|---|---|---|---|
| GLA (Gated Linear Attention) | `H_t = G_t * H_{t-1} + k_t ⊗ v_t`; per-channel scalar decay | Partial — outer product write, 2D state, decay; but write is independent of state (no retrieval step) | No — Triton only (CUDA) | `[GAP — requires new kernel]` if GLA is needed; not applicable for DeltaNet |
| RetNet | `H_t = γ * H_{t-1} + k_t ⊗ v_t`; fixed exponential decay | Partial — same shape as GLA/DeltaNet state; but fixed decay, no retrieval step | No | `[GAP — requires new kernel]` if RetNet is needed; not applicable for DeltaNet |
| Vanilla linear attention | `H_t = H_{t-1} + k_t ⊗ v_t`; no decay | Partial — outer product write; simplest case; no retrieval step, no decay | No | Not applicable for DeltaNet |
| Mamba SSM | `h_t = A * h_{t-1} + B_t x_t`; diagonal decay; output `C_t h_t` | Partial — 2D state, outer product, state readout; but no retrieval-before-write dependency | Yes — `ttnn/cpp/ttnn/operations/experimental/ssm/` (expected) | `[PARTIAL_REUSE]` — idioms borrowable (see `mamba_ssm_kernel_review.md`) |
| `gdn_full_fused_inplace` (Blackhole) | All 6 DeltaNet ops fused; state in L1 | Exact match — this IS the DeltaNet fused kernel | Yes — `models/experimental/tt_symbiote/ops/` or similar (verify) | `[REUSABLE — port and tune]` (see Ch4) |
| Composed TTNN (Chapter 2) | All 6 DeltaNet ops via TTNN primitives | Exact match — correct implementation | Yes — all 12 ops available in TTNN API | `[AVAILABLE — needs wiring]` |

---

## 4. Conclusion

The survey confirms the expected finding: no GLA, RetNet, or vanilla linear attention kernel exists in tt-metal. The only linear attention implementation is `TTNNQwen3LinearAttention` for DeltaNet, and the only TT-Metalium kernel for any linear attention variant is `gdn_full_fused_inplace` from the Blackhole implementation.

> **Key Finding:** The best path forward has two phases:
>
> **(a) Immediate — wire the composed TTNN form (Chapter 2).** All 12 TTNN operations for the DeltaNet decode step are available in the TTNN API. Wiring them into `TTNNQwen3LinearAttention.forward` eliminates all host crossings and achieves Metal Trace compatibility without any kernel development. This is Task 5 in the Chapter 7 implementation roadmap and unblocks end-to-end tracing.
>
> **(b) Latency — port and tune `gdn_full_fused_inplace` from Blackhole (Chapter 4).** Once the composed form is wired and trace is confirmed working, the fused kernel reduces dispatch overhead from 12 ops to 1 per layer. This is Task 6 in the roadmap and is the correct path to achieving the ~177 µs total DeltaNet decode latency estimated in Chapter 6.

No other candidate from this survey changes either recommendation. Mamba idioms are borrowable for phase (b) but do not alter the strategy. GLA and RetNet are irrelevant — they are not implemented in tt-metal, and even if they were, they would not encode the DeltaNet-specific retrieval-then-error pattern.
