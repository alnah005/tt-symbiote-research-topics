# FFN Projections

This file covers the weight and activation tensor shapes for the feed-forward
network (FFN) in every Gemma 4 31B decoder layer. Unlike the attention
projections, which vary between sliding and global layers, the FFN projections
are **identical across all 60 layers**.

## GeGLU Structure

The FFN in each decoder layer uses a Gated Linear Unit with GELU activation
(GeGLU). The computation is:

```math
\text{FFN}(x) = W_{\text{down}} \cdot \left(\text{GELU}_{\tanh}(W_{\text{gate}} \cdot x) \odot W_{\text{up}} \cdot x\right)
```

where $\odot$ denotes element-wise multiplication.

The data flows through three stages:

1. **Parallel projections**: The gate and up projections operate on the same
   input in parallel, expanding from `hidden_size` to `intermediate_size`.
2. **Gated activation**: The gate output passes through
   `gelu_pytorch_tanh`, then is multiplied element-wise with the up output.
3. **Down projection**: The result is projected back to `hidden_size`.

```text
                    input x  [B, S, 5376]
                       |
            +----------+----------+
            |                     |
            v                     v
     gate_proj linear       up_proj linear
     [5376, 21504]          [5376, 21504]
            |                     |
            v                     |
     GELU_tanh                    |
            |                     |
            +--------->*<---------+
                  (element-wise)
                       |
                       v
               down_proj linear
               [21504, 5376]
                       |
                       v
                output  [B, S, 5376]
```

## Weight Shapes

| Projection | Weight Name | Shape [in, out] | Parameters | Bytes (BF16) |
|------------|-------------|-----------------|-----------|--------------|
| Gate | `mlp.gate_proj.weight` | [5376, 21504] | 115,605,504 | 231,211,008 |
| Up | `mlp.up_proj.weight` | [5376, 21504] | 115,605,504 | 231,211,008 |
| Down | `mlp.down_proj.weight` | [21504, 5376] | 115,605,504 | 231,211,008 |
| **Total per layer** | | | **346,816,512** | **~694 MB** |

None of these projections use bias (`bias=False`).

### Shape Derivation

The shapes come directly from two config parameters:

- `hidden_size = 5376` --- the model dimension
- `intermediate_size = 21504` --- the FFN intermediate dimension

The expansion ratio is:

```math
\frac{\text{intermediate size}}{\text{hidden size}} = \frac{21504}{5376} = 4.0
```

This 4x expansion ratio is standard for GeGLU architectures. Note that the
effective expansion is 8x relative to the hidden size when considering both
the gate and up projections together (two parallel [5376, 21504] matmuls),
but the gated multiplication reduces the intermediate tensor back to 21504
before the down projection.

## Activation Shapes During Decode

For batch=1 single-token decode (`B=1, S=1`):

| Stage | Shape | Notes |
|-------|-------|-------|
| Input (after pre-FFN norm) | [1, 1, 5376] | |
| After gate proj | [1, 1, 21504] | Before activation |
| After GELU_tanh | [1, 1, 21504] | Activated gate |
| After up proj | [1, 1, 21504] | |
| After element-wise multiply | [1, 1, 21504] | Gate * Up |
| After down proj | [1, 1, 5376] | Back to hidden_size |

The peak intermediate activation size per layer is **two** [1, 1, 21504]
tensors held simultaneously (the activated gate output and the up output) during
the element-wise multiply. At BF16, this is 2 x 21504 x 2 = 86,016 bytes per
token --- negligible for decode.

During prefill with sequence length $S$, the intermediate activations scale
linearly: each tensor is [1, S, 21504]. At $S = 8192$ prefill chunks and BF16,
the two intermediate tensors together occupy $2 \times 8192 \times 21504 \times 2
\approx 671$ MB.

## Activation Function: gelu_pytorch_tanh

The gate activation uses the tanh-approximated GELU:

```math
\text{GELU}_{\tanh}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)
```

This maps to `ttnn.gelu` with the tanh approximation mode. The approximation is
critical for numerical fidelity --- using the exact GELU (based on the error
function) would produce slightly different results than what the model was
trained with.

## Uniformity Across Layers

Unlike the attention projections, which differ between sliding and global layers,
the FFN is structurally identical for all 60 layers. This means:

- A single set of `ttnn.linear` program configs covers the FFN matmuls for
  every layer.
- The same tensor-parallel sharding strategy applies uniformly.
- Weight loading code does not need per-layer-type branching for FFN weights.

The uniformity simplifies the TTNN implementation: `TTNNGemma4FFN` can be a
single class with no layer-type-dependent configuration.

## TTNN Implementation Notes

### Fused Gate + Up Projection

The gate and up projections share the same input and have the same shape. They
can be fused into a single matmul by concatenating the weight matrices along the
output dimension:

```math
W_{\text{gate+up}} = [W_{\text{gate}} \mid W_{\text{up}}] \quad \text{shape: } [5376, 43008]
```

The fused output `[B, S, 43008]` is then split into two `[B, S, 21504]`
tensors: one for the gate path (which receives GELU_tanh) and one for the up
path (which passes through unchanged).

This trades a larger single matmul for two smaller ones. On Wormhole, the fused
approach typically wins because:

1. A single `[5376, 43008]` matmul has better compute utilization than two
   `[5376, 21504]` matmuls.
2. It eliminates the overhead of launching a second matmul kernel.
3. The input tensor is read from memory once instead of twice.

The split after the fused matmul is a lightweight `ttnn.slice` operation with
negligible cost.

### Column-Parallel and Row-Parallel Sharding

Under TP=8 on T3K:

- **Gate and up projections** (column-parallel): the output dimension 21504 is
  split across 8 devices, giving each device a `[5376, 2688]` weight shard.
  For the fused gate+up, each device holds `[5376, 5376]`.
- **Down projection** (row-parallel): the input dimension 21504 is split across
  8 devices, giving each device a `[2688, 5376]` weight shard. An
  `ttnn.all_reduce` on the hidden_size dimension combines partial results.

Note the coincidence: with the fused gate+up under TP=8, each device's shard
is `[5376, 5376]` --- a square matrix. This can simplify tiling and program
config selection.

### Matmul Program Config Guidance

The FFN matmuls are the most compute-intensive operations per layer (by total
FLOPs). For single-token decode:

| Matmul | Per-Device Shape (TP=8) | FLOPs | Character |
|--------|------------------------|-------|-----------|
| Gate+Up (fused) | `[1, 5376] x [5376, 5376]` | 57.8M | Memory-bound |
| Down | `[1, 2688] x [2688, 5376]` | 28.9M | Memory-bound |

At decode batch=1, these are strongly memory-bound (reading large weight
matrices for a single-token activation vector). The key optimization is
DRAM-sharded weight storage to maximize memory bandwidth utilization. See
[Chapter 8](../ch8_performance/index.md) for the performance analysis.

---

**Next:** [`ple_shapes.md`](./ple_shapes.md)
