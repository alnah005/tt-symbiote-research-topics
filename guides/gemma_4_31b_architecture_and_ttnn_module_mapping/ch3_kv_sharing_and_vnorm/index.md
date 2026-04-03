# Chapter 3 --- K=V Sharing and V-Norm Implementation

## Overview

This chapter deep-dives into the two most novel attention features in
Gemma 4 31B --- K=V weight sharing in global layers and V-norm without a learned
scale parameter --- and analyzes how each maps to TTNN operations on the T3K
mesh.

**K=V sharing** eliminates the V projection weight in all 10 global attention
layers. A single `k_proj` linear produces a shared tensor that is assigned to
both the K and V paths *before* any normalization or RoPE. The paths then
diverge: the K copy receives scaled RMSNorm followed by partial RoPE (128 of
512 dims), while the V copy receives unscaled RMSNorm (no learned scale) and
no RoPE. This saves one `[5376, 2048]` weight matrix per global layer ---
approximately 220 MB total at BF16 across 10 layers.

**V-norm** is an RMSNorm variant with `with_scale=False` applied to value
vectors in *every* attention layer (all 60 --- both sliding and global). Unlike
standard RMSNorm which multiplies by a learned $\gamma$ after normalization,
V-norm performs pure magnitude normalization with no trainable parameters. This
has direct implications for the TTNN implementation because the standard
`TTNNDistributedRMSNorm` module expects a weight tensor.

### Why These Features Matter for TTNN

1. **K=V sharing changes the projection count.** Global layers have two
   projections (Q and K) instead of three (Q, K, V). Fused QKV optimizations
   must adapt: the fused weight packs Q and K only, with V derived from the K
   slice. Any `TTNNModule` that assumes separate K and V weights will produce
   incorrect results for global layers.

2. **V-norm requires a scale-free RMSNorm variant.** The `with_scale=False`
   semantics mean there is no weight parameter to load, shard, or multiply.
   Depending on `TTNNDistributedRMSNorm` support for this mode, the
   implementation may need a workaround --- either an all-ones dummy weight or
   a manual TTNN op sequence.

3. **The K and V paths must diverge after the shared projection.** In TTNN, the
   shared tensor must be duplicated (or referenced twice) before feeding into
   separate norm and RoPE pipelines. This requires careful tensor lifecycle
   management to avoid overwriting the shared buffer.

## Reading Order

1. [`k_eq_v_mechanism.md`](./k_eq_v_mechanism.md) --- Detailed dataflow of K=V
   sharing in global layers: the single projection, the divergent K and V
   post-processing paths, parameter savings, and TTNN mapping.
2. [`vnorm_implementation.md`](./vnorm_implementation.md) --- V-norm definition,
   mathematical expression, presence in all 60 layers, and three TTNN
   implementation strategies with performance analysis.

## Prerequisites

This chapter builds on:

- [Chapter 1 --- Architecture Overview](../ch1_architecture_overview/index.md):
  attention configuration details, the `attention_k_eq_v` flag, and the
  sliding/global layer distinction.
- [Chapter 2 --- Projection Weights and Tensor Shapes](../ch2_projection_shapes/index.md):
  the exact shapes of Q, K, and V projection weights and the decode activation
  shapes at each stage.

---

**Next:** [Chapter 4 --- Dual RoPE and Partial Rotary Embedding](../ch4_dual_rope/index.md)
