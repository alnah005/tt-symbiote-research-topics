# math_approx_mode Recommended Settings for MoE

## MoE FFN Structure Reference

Both DeepSeek-V3 and Qwen MoE use SwiGLU (SiLU-gated linear unit):

```
expert_output = down_proj( silu(gate_proj(x)) * up_proj(x) )
# w1 = gate_proj, w3 = up_proj, w2 = down_proj
```

This is the same structure as DeepSeek's naming convention. The kernel config recommendations below apply to both architectures.

## Recommended Settings Per Projection

### Gate Projection (w1) — math_approx_mode=False

Use `COMPUTE_KERNEL_CONFIG_LOFI` (see `index.md` § Key Config Reference).

Rationale: the gate matmul is a pure FPU operation. `math_approx_mode` has no effect. Setting it to `False` is conservative and explicit — this kernel does not use SFPU approximations. LoFi fidelity is appropriate for bfloat16 weights in MoE because the expert layers are narrow (hidden_dim → intermediate_dim per expert) and throughput is the binding constraint.

### Up Projection (w3) — math_approx_mode=False

Use `COMPUTE_KERNEL_CONFIG_LOFI` (see `index.md` § Key Config Reference).

Rationale: same as gate. The SiLU activation is applied to `gate_out` in a separate element-wise op, not fused into the up projection matmul. `math_approx_mode` is therefore irrelevant here. For well-scaled inputs, the SiLU approximation error is bounded and does not require exact SFPU evaluation.

### Down Projection (w2) — math_approx_mode=True (with HiFi2)

Use `COMPUTE_KERNEL_CONFIG_HIFI2` (see `index.md` § Key Config Reference).

Rationale: the down projection uses HiFi2 because it accumulates across the full intermediate dimension, where precision is more important than for gate/up. If any activation is fused into the down projection kernel, `math_approx_mode=True` is acceptable — the HiFi2 config is shared with attention softmax in the broader model where this setting is intentional. For pure matmul (no fused activation), the flag is irrelevant but consistent with the shared HiFi2 config.

## Summary Table

For the complete projection-to-config assignment, see Chapter 5, `index.md`.

## Qwen MoE Note

Qwen MoE FFN uses the same SwiGLU structure as DeepSeek:

```python
# Qwen MoE expert forward (structurally identical to DeepSeek)
gate = self.gate_proj(x)    # w1
up   = self.up_proj(x)      # w3
act  = F.silu(gate) * up    # SwiGLU
out  = self.down_proj(act)  # w2
```

Apply the same kernel config recommendations: LoFi + `math_approx_mode=False` for gate and up; HiFi2 + `math_approx_mode=True` for down.

## Practical Guidance

**When in doubt, set `math_approx_mode=False`.**

For MoE expert matmuls, the throughput difference between `math_approx_mode=True` and `False` is negligible because the SFPU is not the bottleneck — the FPU matrix multiply dominates cycle count. The accuracy levers that actually matter for MoE FFN quality are:

1. `math_fidelity` (LoFi vs HiFi2): controls FPU accumulation passes; meaningful PCC impact
2. `packer_l1_acc`: controls whether partial sums accumulate in L1; meaningful throughput and accuracy impact
3. `fp32_dest_acc_en`: enables fp32 accumulation register; meaningful for sensitive reductions

`math_approx_mode` ranks last in impact for pure MoE FFN expert kernels. Do not tune it as a primary accuracy knob.

```python
# Conservative safe defaults for any MoE expert projection
COMPUTE_KERNEL_CONFIG_SAFE = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,
    math_approx_mode=False,  # safe default; no measurable throughput cost for matmul
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
```

---

**Next:** [Chapter 5 — MoE Expert Matmul Configuration](../ch5_moe_expert_matmul_config/index.md)
