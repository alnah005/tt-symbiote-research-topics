# Weight Layout and Quantization

## Default Weight Layout

Transformer weights are large. For a Llama 3.1 8B model, the total parameter count is roughly 8 billion values. Loading them naively at BF16 would require 16 GB — more than the on-device DRAM on most Tenstorrent single-chip configurations. Weight layout decisions at load time determine both whether the model fits on device and how fast each matmul runs.

At load time, the tt-transformers `load_checkpoints.py` routines perform three transformations:

1. **Dtype conversion**: Weights are converted from the checkpoint's native dtype (typically BF16 or FP32) to the target dtype for that layer (BFP4, BFP8, or BF16) on the host CPU before the tensor is sent to device. This conversion is done once per model load, not per inference step.

2. **Tile layout**: Weights are rearranged from row-major layout into TILE layout — a 32×32 tile-major format that matches the hardware's matrix FPU primitive size. The tile layout is mandatory for matmul: the hardware's FPU operates on 16×16 sub-tiles within 32×32 tiles, and it cannot accept row-major input.

3. **DRAM interleaved placement**: Tiles are distributed (interleaved) across all available DRAM banks. This spreads DRAM read load across all banks when a core fetches tiles sequentially, maximizing aggregate DRAM bandwidth for interleaved access patterns.

Weights remain resident on device DRAM across all inference steps. There is no PCIe transfer per request; the weight tensor is loaded once and remains allocated on device until the model is unloaded.

---

## BFP4 vs BFP8 for MLP Weights

MLP layers — FF1, FF2, FF3 (gate projection in gated MLP architectures like Llama) — are the largest weight consumers in transformer models. Hidden dimension expansion ratios of 4× are common, meaning FF1 and FF2 weights are each 4× the size of attention projection weights. Quantizing these layers aggressively has an outsized impact on both model size and decode throughput.

### Throughput Numbers (Llama 3.1 8B on N150)

From measured profiling results:

- **Performance mode** (BFP4 FF1/FF3 weights + BFP4 FF2): approximately 28 tokens/s/user decode throughput
- **Accuracy mode** (BFP8 MLP weights): approximately 23 tokens/s/user decode throughput

BFP4 MLP weights deliver approximately 22% higher decode throughput (+5 t/s/u) relative to BFP8.

### Why BFP4 Helps

BFP4 stores each weight element in approximately 4.5 bits (3-bit mantissa + 1 sign bit + 1/16 share of an 8-bit block exponent), compared to approximately 8.5 bits for BFP8. This is approximately a 1.9× reduction in weight storage and bandwidth (8.5 / 4.5 ≈ 1.89×).

Decode is bandwidth-bound: with M=1–32, the FPU finishes computing each output row far faster than DRAM can deliver the next weight tile. Reducing the weight bandwidth requirement by approximately 1.9× roughly improves the effective matmul throughput by approximately 1.9× for that layer, assuming the bottleneck was the weight DRAM read and not something else (e.g., activation memory or compute).

### When to Avoid BFP4

Not all models tolerate BFP4 MLP weights without accuracy degradation. Qwen-2.5-7B, for example, degrades noticeably on N300 in performance mode. For models with confirmed sensitivity, BFP8 MLP is the default.

Model-specific dtype overrides are specified in `model_params/<model-name>/` configuration files within tt-transformers. These files set layer-level dtype without requiring changes to the model implementation. When porting a new model, benchmarking accuracy at BFP4 vs BFP8 for MLP layers is a recommended first tuning step.

---

## BFP8 for Attention Weights

QKV projection and output projection weights are kept at BFP8 in most configurations. Attention weight quantization is more numerically sensitive than MLP weight quantization for two reasons:

1. **Score distribution sensitivity**: The query and key projections produce the vectors that are multiplied to form attention scores. Quantization noise in Q or K propagates into the softmax input, distorting the score distribution. Small absolute errors in Q/K values can produce large relative errors in softmax outputs, particularly for high-magnitude scores.

2. **Residual vs score path**: MLP output is added to the residual stream, where quantization noise is attenuated by the residual connection over many layers. Attention score computation is not residual — errors compound directly within the attention head.

The Llama 3 series is empirically insensitive to attention weight precision at BFP8, making BFP8 the practical choice: it retains sufficient mantissa precision (7 bits) while reducing memory to approximately 53% of BF16 (approximately 1.88× lower, since BFP8_B at ~8.5 bits/element vs BF16 at 16 bits/element gives 16 / 8.5 ≈ 1.88×).

---

## Math Fidelity Pairing with Weight Dtype

Math fidelity controls how many accumulation passes the matrix FPU performs per output tile. Lower fidelity is faster but accumulates less precisely. The correct pairing matches fidelity to the actual precision of the operands: there is no benefit to running HiFi4 when one operand has only 3–4 bits of effective mantissa.

| Weight dtype | Activation dtype | Recommended fidelity |
|---|---|---|
| BFP4 | BF16 | LoFi |
| BFP8 | BF16 | HiFi2 |
| BF16 | BF16 | HiFi4 |

### Why These Pairings

- **BFP4 + LoFi**: BFP4 weights have a 3-bit mantissa plus sign, giving approximately 4 bits of effective mantissa. LoFi's single-pass 5×7-bit multiplier produces a 12-bit intermediate product, which is more than sufficient to cover the 4-bit effective weight precision. Running HiFi2 or higher adds passes that contribute no additional accuracy but cost throughput.

- **BFP8 + HiFi2**: BFP8's 7-bit mantissa requires a 2-pass accumulation to cover the full mantissa width. HiFi2 provides exactly this. Running HiFi4 adds two more passes without meaningful accuracy gain for BFP8 operands.

- **BF16 + HiFi4**: BF16 has a 7-bit explicit mantissa, but full IEEE-style accumulation across many tiles requires HiFi4's 4-pass accumulation to maintain end-to-end precision for sensitive layers.

LoFi uses 1 accumulation pass versus HiFi4's 4 passes — one quarter the passes of HiFi4. The fidelity choice compounds with the dtype choice: BFP4+LoFi is the fastest possible matmul configuration, approximately 3.56× lower bandwidth (BFP4_B at ~4.5 bits/element versus BF16 at 16 bits/element) and 4× fewer FPU passes than BF16+HiFi4.

---

## Transposed Weight Layout

TTNN matmul follows the convention `output = activation @ weight`, where activation has shape `[M, K]` and weight has shape `[K, N]`. This is the standard convention for linear layers when weights are in `[in_features, out_features]` layout (K=in_features, N=out_features).

PyTorch checkpoints store weight matrices as `[out_features, in_features]` — the transpose of what TTNN expects. For a weight matrix W of shape `[out_features, in_features]` (as stored in a checkpoint), there are two options:

- **Pre-transpose on host**: Transpose W on the host to produce W^T of shape `[in_features, out_features]` and call `ttnn.matmul(activation, W_T)` where `W_T` has shape `[K, N]` with K=in_features and N=out_features. No runtime transpose is needed.
- **Runtime transpose**: Pass `transpose_b=True` in the matmul call. TTNN transposes the weight tile-by-tile at inference time. This adds a small but non-negligible per-inference overhead.

Pre-transposing on host is preferred for production models. The `load_checkpoints.py` routines in tt-transformers apply weight transposition as part of the load step, so the device-resident weight tensor is already in the layout expected by `ttnn.matmul` without a `transpose_b` flag. This eliminates the runtime overhead entirely.

The transposition is also compatible with BFP4/BFP8 layout: the block exponents are stored per 16-value block, and transposition of the tile rearranges which values fall in each block. The `load_checkpoints.py` routines handle this correctly by performing quantization after transposition, not before.

---

## Key Takeaways

- Weight dtype (BFP4 vs BFP8 vs BF16) and math fidelity (LoFi vs HiFi2 vs HiFi4) must be selected together: mismatching them wastes throughput or loses precision.
- BFP4 MLP weights are the primary throughput lever for decode on bandwidth-bound configurations; expect approximately 20–25% improvement over BFP8 MLP when the model tolerates it.
- Attention weights should default to BFP8 rather than BFP4 due to score distribution sensitivity; BFP8 provides sufficient precision at approximately 1.88× lower bandwidth than BF16.
- Pre-transposing weight matrices at load time in `load_checkpoints.py` eliminates per-inference transpose overhead; do not rely on `transpose_b=True` in production matmul calls.

## Further Reading

- BFP4/BFP8 format specification: Chapter 1 of this guide, "Hardware and TTNN Foundations"
- Math fidelity constants: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` (search `MathFidelity`)
- Model-specific dtype configs: `models/demos/llama3/model_params/` in tt-transformers
- `load_checkpoints.py` weight loading routines: `models/demos/llama3/tt/load_checkpoints.py`
- PERF.md throughput benchmarks: `models/demos/llama3/PERF.md` in tt-transformers

---

[Back to Chapter 3 Index](index.md)
