# Math Fidelity: Precision Model and Throughput Characterization

## What Happens in the Wormhole FPU During a BF16 Multiply-Accumulate

A Tensix FPU processes data in 32×32 tiles. During a BF16 matmul, each tile operation takes two BF16 input tiles (from the A and B matrices) and accumulates their dot products into a destination tile. At the element level, the FPU takes pairs of BF16 values — one sign bit, eight exponent bits, seven mantissa bits each — and multiplies them together, then adds the product to a running accumulator.

The `math_fidelity` field controls how many mantissa bits from each operand are actually presented to the multiplier hardware in this step. The exponent bits are always fully used. Only the mantissa input to the multiplier is affected.

This distinction matters: mantissa truncation before the multiply is not the same as post-multiply rounding. The truncation happens before the product is formed, which means low-fidelity configurations introduce a systematic, per-element rounding error into each partial product. These errors accumulate across the K loop.

> **Tip:** Think of `math_fidelity` as a precision dial on the multiplier, not on the accumulator. `fp32_dest_acc_en` controls accumulator precision; `math_fidelity` controls per-multiply input precision. The two mechanisms are independent and additive in their effect on output accuracy.

---

## Fidelity Levels

### LoFi

The FPU uses a reduced mantissa representation for each operand. The exact internal bit reduction is hardware-microcode-specific and is not exposed in the TTNN Python API. The observable effects are:

- Faster multiply-accumulate: the hardware completes tile FMAs in approximately 50% of the cycles required by HiFi4. For compute-bound workloads this translates directly to ~2× higher tile throughput.
- Small but measurable per-element rounding error compared to a PyTorch float32 reference. The error is larger than HiFi4 but for many MoE projections remains within an acceptable PCC range.

For decode-mode inference (M=1 to 32, bandwidth-bound), the compute savings may be partially masked by DRAM access latency. Even so, reducing compute time per tile frees the pipeline sooner, contributing to lower stall time and higher throughput.

### HiFi2

Uses more mantissa bits than LoFi, closer to full BF16 precision. Bridges the throughput vs. accuracy trade-off. HiFi2 is the standard choice when PCC > 0.999 is required and the operation is not a reference-validation run.

Approximate throughput relative to LoFi: ~0.85× — a meaningful reduction from LoFi, but significantly faster than HiFi4 (which runs at ~0.5×, making HiFi2 approximately 1.7× faster than HiFi4). HiFi2 is the recommended choice when LoFi precision is insufficient.

### HiFi3

An intermediate level between HiFi2 and HiFi4. In practice HiFi3 is rarely the optimal choice: HiFi2 usually provides sufficient accuracy at better throughput, and HiFi4 is preferred when PCC must be maximized. HiFi3 is documented here for completeness and may be useful in edge cases where HiFi2 falls just short of a PCC threshold and HiFi4 is too costly.

Approximate throughput relative to LoFi: ~0.7×.

### HiFi4

Full BF16 mantissa precision — all seven mantissa bits of each operand are presented to the multiplier. This is the configuration that most closely matches a PyTorch BF16 reference and produces the highest PCC vs. a float32 reference.

HiFi4 is recommended when:
- Validating correctness: establishing a PCC baseline before stepping down to lower fidelity levels
- The operation feeds into softmax or layer norm where small accumulated errors can amplify through the exponential or normalization
- End-to-end model quality requires the tightest possible numerical match to a reference

Approximate throughput relative to LoFi: ~0.5× (roughly 2× slower than LoFi for compute-bound shapes).

---

## Throughput and PCC Reference Table

The table below characterizes the four fidelity levels for a representative MoE matmul shape: 4096×2048 BF16, measured against a float32 PyTorch reference.

| Fidelity | Relative throughput | Typical PCC (vs float32 ref) |
|---|---|---|
| LoFi | 1.0× (baseline) | ~0.99–0.999 |
| HiFi2 | ~0.85× | ~0.999–0.9999 |
| HiFi3 | ~0.7× | ~0.9999 |
| HiFi4 | ~0.5× | ~0.9999–1.0 |

> **Warning:** These values are relative guides, not hard specifications. Exact throughput and PCC depend on matrix shape, data distribution, dtype, and the specific input values. Always run your own PCC sweep on representative activations and weights before finalizing a fidelity choice. See `fidelity_selection_workflow.md` for the sweep procedure.

---

## What Math Fidelity Does NOT Affect

`math_fidelity` ONLY affects the FPU multiply-accumulate path — the part of the pipeline that processes input tile pairs during the K loop. It does NOT affect:

- **SFPU operations**: activations (SiLU, GeLU, ReLU), transcendentals (exp, log, reciprocal), and other element-wise ops that go through the SFPU pipeline are controlled by `math_approx_mode`, not `math_fidelity`.
- **Accumulator precision**: how the partial sums are accumulated in the destination register is controlled by `fp32_dest_acc_en`. `math_fidelity` and `fp32_dest_acc_en` are independent fields.
- **Packer path**: L1 vs. DRAM accumulation of partial sums is controlled by `packer_l1_acc`.

For MoE expert matmuls that are pure matmul (no fused activation kernel), `math_fidelity` is the primary accuracy lever available to the developer. Fusing the SiLU activation into the same kernel would introduce SFPU sensitivity, but in most TTNN MoE implementations the matmul and activation are separate ops.

---

## Next Steps

Read `fidelity_and_moe_accuracy.md` to understand why the three MoE projection types (gate, up, down) map to different entries in the fidelity table above — specifically, how K-loop depth and the presence or absence of a downstream nonlinearity determine whether LoFi mantissa errors remain bounded or compound into residual stream drift.
