# PCC Accuracy Thresholds for the DeltaNet State Update

This file establishes the numerical accuracy requirements for the on-device DeltaNet decode implementation. It defines the PCC (Pearson Correlation Coefficient) thresholds that both the composed TTNN form and the fused kernel form must meet, explains why state errors do not accumulate unboundedly in DeltaNet recurrence, and specifies the measurement methodology a test suite must use to validate correctness across a multi-step decode sequence.

> **Key Finding:** The PCC threshold for the DeltaNet state update is 0.999 per decode step, measured for both `S_new` (the updated state matrix) and `o_t` (the output token vector) against a PyTorch FP32 reference. State errors do not accumulate exponentially because the DeltaNet decay gate `g_t < 1` contracts errors from prior steps by the factor `g_t^{T-t}`. For `g_t ≈ 0.9` and T=100 steps, a step-0 state error is attenuated by a factor of `(0.9)^{100} ≈ 2.7 × 10^{-5}` by step 100 — effectively erased.

## 1. Standard PCC Thresholds in tt-transformers

The tt-transformers codebase applies a hierarchy of PCC thresholds depending on the nature of the operation. These serve as reference points for interpreting the DeltaNet target:

| PCC threshold | Applicable ops |
|---|---|
| 0.9999 | Lossless or near-lossless ops: `ttnn.reshape`, `ttnn.permute`, elementwise `ttnn.add` on small tensors |
| 0.999 | Matmul-derived ops; expected output of BF16 matrix multiplications against FP32 reference |
| 0.99 | Ops with significant BF16 rounding in long accumulation chains |
| 0.98 | Acceptable degradation for some fused ops with mixed precision intermediate results |

The DeltaNet 6-op recurrence is dominated by two matrix multiplications (the `S K^T` retrieval and the `V K^T` outer product write). This places it squarely in the 0.999 tier.

## 2. Recommended Threshold for DeltaNet State Update

**Both `S_new` and `o_t` must achieve PCC > 0.999 against the PyTorch FP32 reference at every decode step.**

This threshold is achievable in BF16 because:

1. The 6-op recurrence has no long accumulation chains. Each op (decay, retrieve, error, outer product, add, project) is a single matmul or elementwise op — not a softmax over a long sequence.
2. The intermediate tensors `[1, H, d_k, d_v]` = `[1, 4, 128, 128]` are small enough that tile-level BF16 rounding does not compound significantly across ops.
3. The decay gate multiplication (`S ← g_t · S`) is an elementwise scale — exact in BF16 for gate values representable in the BF16 mantissa.

If the composed TTNN form achieves 0.999 PCC, the fused kernel form should achieve the same or better (the same mathematical operations, fewer intermediate host-side tensor transitions to introduce rounding differences).

## 3. Error Decay Argument

Standard recurrent networks accumulate errors across time steps: a step-`t` error `ε_t` propagates forward as `ε_{t+n}` and can grow if the recurrence has eigenvalues near or above 1. DeltaNet is different because the decay gate `g_t ∈ (0, 1)` is applied at every step:

```
S_t = g_t · S_{t-1} + β_t · (v_t - S_{t-1} k̃_t) ⊗ k̃_t
```

Suppose there is an error `δS_0` in the state at step 0 (e.g., a BF16 rounding error in the initial TTNN implementation of the state update). The propagation of this error to step T, in the absence of corrective signal, is bounded by:

```
||δS_T|| ≤ g_max^T · ||δS_0||
```

where `g_max = max_t g_t`. In practice `g_t ≈ 0.9` for a typical DeltaNet gate at inference time. Therefore:

```
T = 10:   (0.9)^{10}  ≈ 0.349   (error reduced to 35%)
T = 50:   (0.9)^{50}  ≈ 0.005   (error reduced to 0.5%)
T = 100:  (0.9)^{100} ≈ 2.7×10⁻⁵ (error effectively zero)
```

**Implication:** A small per-step numerical error in the BF16 state update will decay away within approximately 50–100 steps. The DeltaNet recurrence is *less* numerically sensitive than softmax attention (where attention over a long context has no analogous decay mechanism). This is why 0.999 PCC per step is a sufficient — not merely necessary — threshold.

Note that `g_t = 0` would be perfect memory erasure (pure input-driven state), and `g_t = 1` would be no decay (errors could accumulate). The model learns `g_t` values that balance memory retention and decay; trained values typically stay well below 1 for stability.

## 4. Measurement Methodology

The test suite for the on-device DeltaNet decode should run the following procedure:

1. Initialize a shared random state `S_0` (shape `[1, H, d_k, d_v]` = `[1, 4, 128, 128]`, BF16, values in [-0.1, 0.1]).
2. Generate 200 random decode inputs: `q_tilde`, `k_tilde`, `v`, `g_t`, `beta_t` (shapes from Chapter 2).
3. Run the PyTorch FP32 reference implementation of `recurrent_gated_delta_rule` for all 200 steps, recording `S_ref[t]` and `o_ref[t]` at each step.
4. Run the TTNN implementation (composed or fused) for all 200 steps from the same initial state, recording `S_tt[t]` and `o_tt[t]` at each step.
5. After each step t, compute:
   - `pcc_S[t]   = pearson_correlation(S_tt[t].flatten(), S_ref[t].flatten())`
   - `pcc_o[t]   = pearson_correlation(o_tt[t].flatten(), o_ref[t].flatten())`
   - `l2_S[t]    = ||S_tt[t] - S_ref[t]||_2`
6. Assert `pcc_S[t] > 0.999` and `pcc_o[t] > 0.999` at every step t = 0 … 199.
7. Assert that `l2_S[t]` does not grow monotonically with t (cumulative state drift should be bounded, consistent with the decay argument above).
8. Report step-by-step PCC curves and L2 norm curves in the test output for manual inspection.

Running this test against the PyTorch reference before any hardware testing (using TTNN in simulated/CPU mode or on a single-device Wormhole) is recommended as an early sanity check.

## 5. Acceptable Model-Level Degradation

If the per-step PCC consistently exceeds 0.999 for both `S_new` and `o_t`, the expected model-level degradation is less than 0.1 perplexity point compared to a full FP32 reference run. This is an estimate based on the standard BF16 quantization error budget for transformer inference; it should be verified with an end-to-end perplexity evaluation on a standard benchmark (e.g., WikiText-103 with a 1K token stride) after the implementation is complete.

The 0.1 perplexity degradation estimate is well within the acceptable margin for BF16 versus FP32 quantization error (BF16 inference of large transformers typically shows 0.05–0.3 perplexity degradation on language modeling tasks). DeltaNet's decay gate provides additional robustness that keeps the degradation at the low end of this range.
