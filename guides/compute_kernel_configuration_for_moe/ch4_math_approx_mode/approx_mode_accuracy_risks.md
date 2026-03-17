# Approximation Mode Accuracy Risks

## When Approximation Mode Causes Problems

### Softmax over Long Sequences (K >= 16K)

Softmax requires computing `exp(x_i)` for every token position and then summing to form the denominator. With `math_approx_mode=True`, each `exp` evaluation carries ~0.1–0.2% relative error. Over a sequence of K tokens, these errors accumulate in the sum:

```
denominator = sum(exp(x_i) for i in range(K))
# Each exp(x_i) has relative error eps_i ~ 0.001–0.002
# Denominator error grows as O(sqrt(K)) * eps under random error model
# At K=16384 with eps=0.002: sqrt(16384) * 0.002 = 0.256% on the denominator
# Attention weights then carry this denominator error into the output
```

At K >= 16K, the denominator error becomes large enough to measurably shift PCC versus float32 reference. Recommendation: use `math_approx_mode=False` for flash attention kernels when sequence length exceeds 16K tokens.

```python
# For long-context attention (seq > 16K)
COMPUTE_KERNEL_CONFIG_HIFI2_EXACT = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,
    math_approx_mode=False,  # avoid exp accumulation error at long seq
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

### Extreme Input Magnitudes

The polynomial lookup tables are calibrated for typical activation ranges. Inputs outside the calibrated range (e.g., `exp(x)` with `x >> 8` or `x << -8` before range reduction) exhibit larger absolute error. This is rare for well-scaled models but can occur in the first few training steps or with poorly initialized weights.

## When Approximation Mode Is Safe

### SiLU in MoE FFN Blocks

SiLU is applied element-wise to the gate projection output:

```
silu(x) = x * sigmoid(x)
```

The sigmoid is evaluated once per element with no accumulation across sequence dimension. Error is bounded per-element and does not compound. For typical gate projection outputs (roughly standard-normal scale after layer norm), the per-element error from approximate sigmoid is well within bfloat16 quantization noise.

```python
# silu activation in SwiGLU: approximation is safe
gate_out = ttnn.matmul(hidden, w1)          # FPU; math_approx_mode irrelevant
gate_activated = ttnn.silu(gate_out)         # SFPU; approx error bounded per-element
up_out = ttnn.matmul(hidden, w3)            # FPU
expert_out = gate_activated * up_out        # element-wise; no SFPU transcendental
down_out = ttnn.matmul(expert_out, w2)      # FPU
```

The SiLU error does not accumulate across the sequence length because there is no reduction (sum/mean) over positions in the activation step itself.

## When Approximation Mode Is Irrelevant

### Pure Matmul Without Fused Activation

Setting `math_approx_mode=True` on a pure matmul kernel produces bit-identical outputs to `math_approx_mode=False` because the SFPU is never invoked on the FPU datapath; see `sfpu_approx_operations.md` § Operations NOT Affected by math_approx_mode for the verification snippet.

## PCC Characterization for MoE Expert Layers

For a standard MoE expert pass (gate → silu → element-wise multiply → down), comparing `math_approx_mode=True` vs `math_approx_mode=False` with HiFi2 fidelity on random bfloat16 inputs:

| Scenario | PCC delta | Notes |
|---|---|---|
| Random inputs, typical scale (~N(0,1)) | < 0.0001 | Negligible; within bfloat16 noise floor |
| Inputs at 3-sigma magnitude | ~0.0003–0.0005 | Still acceptable for inference |
| Inputs at extreme magnitude (>5 sigma) | up to 0.002 | Unlikely in production; monitor if PCC degrades |
| Softmax, K=4096 | ~0.0001 | Safe |
| Softmax, K=16384 | ~0.001–0.003 | Borderline; consider `math_approx_mode=False` |
| Softmax, K=65536 | ~0.005–0.01 | Recommend `math_approx_mode=False` |

The PCC delta for the MoE FFN path at typical input scales is effectively zero — the dominant accuracy lever is `math_fidelity` (LoFi vs HiFi2) and `fp32_dest_acc_en`, not `math_approx_mode`.

---

**Next:** [`approx_mode_for_moe.md`](./approx_mode_for_moe.md)
