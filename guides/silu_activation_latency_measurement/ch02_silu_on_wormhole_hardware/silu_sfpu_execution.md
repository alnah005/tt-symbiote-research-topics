# SiLU Execution in the Wormhole SFPU

`ttnn.silu` does not map to a single hardware instruction. This file traces the full path from
the mathematical definition of SiLU down to the sequence of SFPU LLK instructions that execute
on Wormhole B0, and explains why the sigmoid sub-expression is the dominant cost.

---

## SiLU: The Math

SiLU (Sigmoid Linear Unit) is defined as:

```
SiLU(x) = x × sigmoid(x)
```

where:

```
sigmoid(x) = 1 / (1 + exp(-x))
```

This is straightforward mathematically, but `exp` is not a primitive instruction on the
Wormhole SFPU. The hardware has no single-cycle exponential. Instead, `exp(-x)` is computed
via an iterative polynomial approximation (Taylor series or Chebyshev polynomial fit), which
requires several multiply-add iterations inside the SFPU pipeline.

---

## The SFPU LLK Instruction Sequence

When the TT-Metalium compiler lowers `ttnn.silu` for Wormhole, the SFPU kernel executes
approximately the following logical steps per 32-element LReg pass:

| Step | Operation | Notes |
|---|---|---|
| 1 | Negate: compute `-x` | One SFPU instruction; cheap |
| 2 | Exp approx: compute `exp(-x)` | Polynomial series — multiple multiply-add iterations; this is the expensive step |
| 3 | Add: compute `1 + exp(-x)` | One SFPU instruction |
| 4 | Reciprocal: compute `1 / (1 + exp(-x))` | One or two SFPU instructions (reciprocal approximation) |
| 5 | Multiply: compute `x × sigmoid(x)` | One SFPU instruction |

Steps 1, 3, 4, and 5 together account for roughly 3–5 SFPU instructions. Step 2 (exp
approximation) accounts for the remaining 2–4 instructions depending on the
`math_approx_mode` setting. Total: approximately **5–8 SFPU instructions per 32-element pass**.

The exact instruction count varies with the precision mode. When `math_approx_mode=True` is
passed to the op, the compiler uses a lower-order polynomial for exp, reducing instruction
count at the cost of some numerical accuracy. When `math_approx_mode=False` (default), a
higher-order approximation is used.

```python
# Example: invoking silu with approximate math mode
import ttnn
output = ttnn.silu(input_tensor, math_approx_mode=True)
```

---

## The 32-Pass Constraint

The SFPU LReg is 32 elements wide. A standard BF16 tile on Wormhole is 32×32 = 1024 elements.
Therefore:

```
passes per tile = 1024 elements / 32 elements per pass = 32 passes
```

Each pass runs the full 5–8 instruction sequence on the 32 loaded elements. The total
instruction count per tile is:

```
~32 passes × (5–8 instructions/pass) = 160–256 SFPU instructions per tile
```

These 32 passes are strictly sequential — there is no parallelism within a single SFPU across
passes. The SFPU is a 32-wide SIMD unit, so within one pass the 32 elements are processed in
parallel, but across passes they are serialized.

This is why SiLU tile latency is not just "instruction depth" but instruction depth **times 32**.

---

## Comparison with Other Activations

| Activation | SFPU Steps per Pass | Est. Instructions per Pass | Relative Cost |
|---|---|---|---|
| ReLU | 1 (conditional max) | ~1 | 1× (baseline) |
| SiLU | negate + exp approx + add + recip + mul | ~5–8 | ~5–8× |
| GELU | erf approx + scale + mul | ~5–8 | ~5–8× |

**ReLU** is the cheapest element-wise activation available. It compiles to a single
`max(x, 0)` instruction in the SFPU, one instruction per pass, 32 passes per tile. Total:
~32 SFPU instructions per tile.

**SiLU** and **GELU** have comparable instruction depth because both require polynomial
approximations of transcendental functions (exp for SiLU's sigmoid; erf for GELU). In practice
their cycle counts are within ~20% of each other for the same approximation order.

The relative ordering is:

```
ReLU  <<  SiLU ≈ GELU
```

If a model's accuracy requirements allow switching from GELU to ReLU, the activation cost
drops by approximately 5–8×. However, for Qwen3-style MoE models that define SiLU as part of
their architecture, the polynomial cost is fixed.

---

## Impact of `math_approx_mode`

The `math_approx_mode` flag controls the polynomial order used for `exp` and `reciprocal`
approximations:

| Mode | Polynomial order | Instructions saved | Accuracy trade-off |
|---|---|---|---|
| `False` (default) | Higher order | 0 (baseline) | Full BF16-class accuracy |
| `True` | Lower order | ~2 instructions/pass → ~64 instructions/tile | Small numerical error in sigmoid tails |

For most inference workloads the accuracy difference is negligible. For latency-critical
deployments, enabling approximate mode is a reasonable tuning lever. Chapter 3 includes
measurements with both modes to quantify the wall-clock difference.

---

**Next Steps:** Read `cycles_vs_matmul.md` to compare the SFPU cycle budget for SiLU against
the FPU cycle budget for the gate_proj matmul, and to place both operations on a roofline.
