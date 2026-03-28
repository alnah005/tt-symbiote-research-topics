# Gated Delta Net — Prefill Pass (B=1, T=full sequence)

This section traces the Gated Delta Net forward pass during prefill. Prefill processes all T tokens of a prompt in a single forward call. The primary differences from decode are: (1) all projections and conv1d operate over the full time dimension; (2) the chunkwise delta rule (`chunk_gated_delta_rule`) replaces the single-step recurrence; (3) the pass saves a `final_state` and `conv_state` that become the initial recurrent state for subsequent decode steps.

Symbols: H=2048, d_k=128, d_v=128, H_k=16, H_v=32, B=1, T=sequence length, C=64 (chunk size).

---

## Step 1 — Input Projections

`[AVAILABLE]` — col-sharded across 8 devices (for `in_proj_qkv` and `in_proj_z`)

The input tensor `x` has shape `[1, T, 2048]`. All four projections are identical to the decode case with T>1 substituted for T=1.

**`in_proj_qkv`:**
- Weight: `[8192, 2048]`.
- Operation: `ttnn.linear(x, weight_qkv, bias=None, core_grid=...)`.
- Output: `[1, T, 8192]`, col-sharded (each device: `[1, T, 1024]`).

**`in_proj_z`:**
- Weight: `[4096, 2048]`.
- Operation: `ttnn.linear(x, weight_z, bias=None, core_grid=...)`.
- Output: `[1, T, 4096]`, col-sharded (each device: `[1, T, 512]`).

**`in_proj_a`:**
- Weight: `[4, 2048]`.
- `[GAP — requires wiring]`: PyTorch `nn.Linear`. Output: `[1, T, 4]`.

**`in_proj_b`:**
- Weight: `[4, 2048]`.
- `[GAP — requires wiring]`: PyTorch `nn.Linear`. Output: `[1, T, 4]`.

**All-gather** (same as decode, applied after `in_proj_qkv` and `in_proj_z`):
- `ttnn.experimental.all_gather_async` on `dim=-1`.
- Result: `[1, T, 8192]` and `[1, T, 4096]` replicated. `[AVAILABLE]`

---

## Step 2 — Causal Conv1d (Prefill)

`[GAP — requires custom kernel]`

During prefill the full-sequence causal conv1d is applied rather than the single-step update used at decode time.

**Inputs:**
- `mixed_qkv` reshaped to `[1, 8192, T]` (batch, channels, time — standard PyTorch conv layout).
- Conv kernel: `[8192, 1, 4]` (depthwise, kernel_size=4).
- Conv bias: `[8192]`.
- Initial conv state: `[1, 8192, 4]` (zeros on the first sequence, or loaded from a cache for multi-turn).

**Operation:** `causal_conv1d_fn(mixed_qkv, weight, bias, initial_states, final_states_out)` — PyTorch C extension from the `causal-conv1d` package.

**Outputs:**
- `mixed_qkv` convolved: `[1, 8192, T]` (causally filtered, no future leakage).
- `conv_state` saved: `[1, 8192, 4]` (the last 4 time steps, written to `conv_states[layer_idx]` for use in the first decode step).

The prefill conv1d applies a size-4 causal kernel at every position in parallel using a custom CUDA kernel that maintains causality without materializing the full T×4 overlap. On T3K there is no equivalent TTNN primitive. A viable first port is to express the conv as a sequence of `ttnn.matmul` calls over the 4-element sliding window, but this does not amortize memory bandwidth efficiently for large T. A dedicated Metalium kernel is the recommended long-term solution.

---

## Step 3 — Chunkwise Gated Delta Rule

`[GAP — requires custom kernel]`

This is the most computationally intensive operation in the Gated Delta Net prefill. It computes the full output over T tokens while maintaining exact recurrent semantics by processing the sequence in chunks of size C=64 (see Chapter 2 for the WY decomposition and inter-chunk state transfer).

**Inputs:**
- Q̃: `[1, T, 32, 128]` (after K/Q repeat_interleave ×2 from H_k=16 to H_v=32).
- K̃: `[1, T, 32, 128]`.
- V: `[1, T, 32, 128]`.
- g: `[1, T, 4]` (per-group log-decay, computed as `exp(α)` where α = −exp(A_log) · softplus(a + dt_bias)).
- β: `[1, T, 4]` (write-gate strength, computed as `sigmoid(b)`).

**Chunking:** T/C chunks (T must be a multiple of C=64; padding applied otherwise). Each chunk processes C=64 tokens.

**Per-chunk computation (WY decomposition — Chapter 2):**
1. Inner-chunk QK product: `ttnn.matmul(Q̃_chunk, K̃_chunk^T)` — `[1, 32, C, C]`.
2. Lower-triangular masking to enforce causality within the chunk.
3. Decay weighting of the attention matrix using g values over the chunk.
4. Inner-chunk AV product: `ttnn.matmul(attn_chunk, V_chunk)` — `[1, 32, C, d_v]`.
5. Cross-chunk contribution: `O_cross = D · ttnn.matmul(Q̃_chunk, S_in)` — state carried from the previous chunk, shape `[1, 32, C, d_v]`. Here Q̃_chunk is `[C, d_k]` and S_in is `[d_k, d_v]`, so the product `Q̃_chunk @ S_in` yields `[C, d_v]` correctly (no transpose). D ∈ R^{C×C} is the diagonal cumulative-decay matrix with `D[τ,τ] = Γ_τ` (cumulative decay within the chunk, as defined in Chapter 2).

**Inter-chunk state transfer (sequential):**
- After each chunk, the recurrent state is updated: S_out = cumulative WY update applied to S_in.
- This update is sequential across chunks (cannot be parallelized along the T dimension) and is currently implemented in PyTorch.
- Shape of state tensor throughout: `[1, 32, 128, 128]` (d_k × d_v per head).

**Current implementation:** `chunk_gated_delta_rule(Q̃, K̃, V, g, β, initial_state)` — Python/Triton kernel from the `flash-linear-attention` library.

**Dominant TTNN ops if ported:**
- `ttnn.matmul`: inner-chunk QK and AV products (`[AVAILABLE]`).
- `ttnn.mul`: decay weighting, β gating (`[AVAILABLE]`).
- `ttnn.sub`, `ttnn.add`: error and state accumulation (`[AVAILABLE]`).
- Lower-triangular mask application: `ttnn.where` with a precomputed mask tensor (`[AVAILABLE]`).
- Inter-chunk state transfer loop: a Python loop over T/C=T/64 iterations, each calling the above TTNN ops.

**Outputs:**
- core_attn_out: `[1, T, 32, 128]`.
- final_state: `[1, 32, 128, 128]` — saved to `recurrent_states[layer_idx]` for decode.

---

## Step 4 — Gated RMSNorm + Output Projection

These steps are identical to their decode counterparts (Steps 6 and 7 in `gated_delta_net_decode_step.md`), scaled to the full sequence length T.

**Gated RMSNorm** — `[GAP — requires custom kernel]`:
- core_attn_out: `[1, T, 32, 128]`.
- z (from `in_proj_z`): `[1, T, 32, 128]` (reshaped).
- Composable TTNN equivalent: `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul`.
- The gating function is SiLU (`z * sigmoid(z)`), not plain sigmoid — see decode Step 6 for the numerical justification.
- Output: gated_out `[1, T, 4096]`.

**Output projection** — `[AVAILABLE]`:
- `ttnn.linear(gated_out, weight_out)`: `[1, T, 4096]` → `[1, T, 2048]`, row-sharded.
- `ttnn.experimental.all_gather_async` on `dim=-1` → `[1, T, 2048]` replicated.

---

## Key Difference: Prefill vs. Decode

| Aspect | Prefill | Decode |
|---|---|---|
| Time dimension | T (full prompt) | 1 |
| Conv1d op | `causal_conv1d_fn` over full `[1, 8192, T]` | `causal_conv1d_update` single step |
| Recurrence op | `chunk_gated_delta_rule` (T/C chunks) | `recurrent_gated_delta_rule` (1 step) |
| State on entry | Zeros (or loaded from prior turn) | Loaded from `recurrent_states[layer_idx]` |
| State saved | `final_state` → `recurrent_states[layer_idx]`; `conv_state` → `conv_states[layer_idx]` | `S_new` → `recurrent_states[layer_idx]`; updated `conv_state` → `conv_states[layer_idx]` |
| Dominant bottleneck | Chunkwise matmuls (compute-bound for large T) | Recurrent state read/write (memory-bandwidth-bound) |

During prefill, the savings from TTNN acceleration of `in_proj_qkv` and `out_proj` are significant because these are large matmuls over T tokens. The three PyTorch kernels (`causal_conv1d_fn`, `chunk_gated_delta_rule`, `FusedRMSNormSwishGate`) still dominate wall time for large T and represent the primary porting targets.

---

## TTNN Gaps for Prefill

| Operation | Status |
|---|---|
| `causal_conv1d_fn` (full sequence) | `[GAP — requires custom kernel]` |
| `chunk_gated_delta_rule` | `[GAP — requires custom kernel]` |
| `FusedRMSNormSwishGate` | `[GAP — requires custom kernel]` (composable from `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul`) |
| `in_proj_a`, `in_proj_b` | `[GAP — requires wiring]` |
| All other ops | `[AVAILABLE]` |

Porting the three custom kernel gaps is the primary kernel development needed to achieve full TTNN acceleration of Gated Delta Net prefill on T3K.

---

**Next:** [`gated_attention_ttnn_ops.md`](./gated_attention_ttnn_ops.md)
