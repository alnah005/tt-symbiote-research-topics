# Gated RMSNorm in TTNN

This file derives a TTNN-native implementation of the `FusedRMSNormSwishGate` operation used in `TTNNQwen3LinearAttention`. The fused module applies RMSNorm to the attention output, applies SiLU (Swish) activation to a learned gate, and multiplies the two. If run as a PyTorch module on CPU/CUDA, it requires host readback. All three component ops are available in the TTNN API; the gap is wiring, not kernel development.

---

## 1. What `FusedRMSNormSwishGate` Computes

Given:
- `x` = attention output, shape `[B, 1, value_dim]`
- `z` = gate projection output, shape `[B, 1, value_dim]`
- `w_norm` = learned RMSNorm weight, shape `[value_dim]`

The operation is:

```
x_normed  = RMSNorm(x, w_norm, eps=1e-6)
          = x / sqrt(mean(x^2) + eps) * w_norm    [element-wise]

gate_act  = SiLU(z)
          = z * sigmoid(z)                         [element-wise, Swish activation]

output    = x_normed * gate_act                    [element-wise product]
```

The result `output` has shape `[B, 1, value_dim]`.

**Key values for Qwen3.6-35B-A3B:**
- `B = 1`
- `value_dim = num_v_heads × d_v` (total, before TP sharding)

> **Note on tensor parallelism:** Under head-parallel sharding on T3K, each device holds `num_v_heads / 8` heads. `x` and `z` on each device have shape `[1, 1, value_dim / 8]`. The RMSNorm weight `w_norm` must be sharded correspondingly to `[value_dim / 8]` per device. The gated RMSNorm is applied per-device with no inter-device communication; an all-reduce or all-gather for the output projection follows downstream.

---

## 2. TTNN Decomposition

```python
def gated_rmsnorm_ttnn(
    x: ttnn.Tensor,           # [B, 1, value_dim_local]  — attention output shard
    z: ttnn.Tensor,           # [B, 1, value_dim_local]  — gate projection shard
    w_norm: ttnn.Tensor,      # [value_dim_local]  — RMSNorm weight shard on device
    eps: float = 1e-6,
) -> ttnn.Tensor:             # [B, 1, value_dim_local]
    # Step 1: RMSNorm on the attention output
    x_normed = ttnn.rms_norm(x, weight=w_norm, epsilon=eps)
    # x_normed.shape: [B, 1, value_dim_local]

    # Step 2: SiLU (Swish) activation on the gate
    gate_act = ttnn.silu(z)
    # gate_act.shape: [B, 1, value_dim_local]

    # Step 3: Element-wise product
    output = ttnn.mul(x_normed, gate_act)
    # output.shape: [B, 1, value_dim_local]

    return output
```

**Availability tags:**
- `ttnn.rms_norm`: `[AVAILABLE]`
- `ttnn.silu`: `[AVAILABLE]`
- `ttnn.mul`: `[AVAILABLE]`
- Full composition: `[AVAILABLE — needs wiring]`

No new kernel development is required. All three primitives exist and are used elsewhere in TTNN models.

---

## 3. Memory Config

All three tensors (`x`, `z`, `output`) are ephemeral within the decode step — they do not persist between steps. They are generated and consumed within a single dispatch sequence and can be held in L1.

**L1 footprint per tensor:**

| Tensor | Shape (local) | Bytes |
|---|---|---|
| `x` | `[1, 1, value_dim_local]` | `value_dim_local × 2` |
| `z` | `[1, 1, value_dim_local]` | `value_dim_local × 2` |
| `x_normed` | `[1, 1, value_dim_local]` | `value_dim_local × 2` |
| `gate_act` | `[1, 1, value_dim_local]` | `value_dim_local × 2` |
| `output` | `[1, 1, value_dim_local]` | `value_dim_local × 2` |
| `w_norm` | `[value_dim_local]` | `value_dim_local × 2` |

For Qwen3.6-35B-A3B with `value_dim = num_v_heads × d_v`, sharded over 8 devices, `value_dim_local = value_dim / 8`. All tensors are small — well within the 1.5 MB L1 per Tensix core. `w_norm` is a persistent weight shard held in L1 throughout the forward pass. Peak simultaneous occupancy at Step 2 entry (after `ttnn.rms_norm` has produced `x_normed`, before `ttnn.silu` starts): `x_normed`, `z`, and `w_norm` are all live — three tensors; `x` may or may not have been freed by the runtime, giving a three-to-four tensor upper bound.

**Recommended memory config:** `(L1, TILE_LAYOUT)` for all three tensors and the weight. `value_dim_local` must be a multiple of 32 for TILE layout; this holds when `value_dim / 8` is tile-aligned (which is the case for standard Qwen3 configurations where `value_dim = 4096`, giving `value_dim_local = 512 = 16 × 32`).

---

## 4. Numerical Equivalence with the PyTorch Reference

The three TTNN ops implement the standard mathematical definitions without modification:

- `ttnn.rms_norm` computes `x / sqrt(mean(x^2) + eps) * w_norm`, identical to PyTorch's `F.rms_norm`.
- `ttnn.silu` computes `z * sigmoid(z)`, identical to PyTorch's `F.silu`.
- `ttnn.mul` is element-wise multiplication, identical to PyTorch's `*` operator.

The composed form is numerically equivalent to `FusedRMSNormSwishGate` up to BF16 rounding. Expected PCC against the PyTorch FP32 reference: **> 0.999**. No long accumulation chains are involved; BF16 rounding errors do not compound across these three ops.

> **Note on fusion opportunity:** `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul` can potentially be fused into a single TT-Metalium kernel to reduce memory round-trips (three intermediate buffers collapse to one). This is a latency optimization rather than a correctness requirement. The composed form is fully correct and trace-compatible; fusion can be added later without changing the interface. Chapter 5 discusses fusion candidates in the broader context of the decode step.

---

## 5. Trace Compatibility

The three-op composition is trace-compatible without any changes:

- No `ttnn.from_torch` is called inside the forward pass.
- `ttnn.rms_norm`, `ttnn.silu`, and `ttnn.mul` each allocate output buffers (`x_normed`, `gate_act`, and `output` respectively), but these allocations are trace-safe: the tensor sizes are fixed for a given model configuration, so the Metal Trace program cache reuses the same buffer addresses on each replay. No explicit pre-allocation is required for these intermediates, as long as the program cache is active and sizes do not vary between trace capture and replay.
- All inputs (`x`, `z`, `w_norm`) are on-device TTNN tensors before the trace bracket; no host-device crossing occurs.

The only prerequisite for trace compatibility is that `x` and `z` arrive from on-device computations (projection + all-gather steps) without a `ttnn.to_torch` / `ttnn.from_torch` round-trip. This is already satisfied when the recurrent delta rule step (Task 5 in Chapter 7) and the projection steps remain on-device.

---

## 6. Summary

| Property | Value |
|---|---|
| Input shapes (local per device) | `[1, 1, value_dim_local]` for `x`, `z`, and `output` |
| RMSNorm weight shape (local) | `[value_dim_local]` |
| Memory layout | `(L1, TILE_LAYOUT)` for all tensors |
| TTNN ops required | `ttnn.rms_norm`, `ttnn.silu`, `ttnn.mul` |
| New kernel required? | No |
| Availability | `[AVAILABLE — needs wiring]` |
| Expected PCC vs. PyTorch reference | > 0.999 |
| Trace compatible? | Yes |
