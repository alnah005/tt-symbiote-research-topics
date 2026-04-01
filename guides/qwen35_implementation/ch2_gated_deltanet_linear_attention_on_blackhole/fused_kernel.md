# Fused Kernel: `ttnn.experimental.gated_delta_net`

## Overview

The fused kernel is a custom Metalium operation that executes all five recurrence steps from
[`recurrence_math.md`](./recurrence_math.md), plus the gated RMSNorm post-processing, in a
single device kernel launch. It uses fp32 arithmetic throughout the state update, working
around the Blackhole SrcB register constraint documented in [`host_recurrence.md`](./host_recurrence.md).

The kernel achieves PCC 0.999997 vs the host float32 reference — essentially lossless compared
to the float32 baseline.

---

## Kernel Location

```
ttnn/cpp/ttnn/operations/experimental/ssm/gated_delta_net/device/
  kernels/gated_delta_net_compute.cpp     # Metalium compute kernel
  gated_delta_net_program_factory.cpp     # Circular buffer layout and dispatch
```

The operation is exposed to Python as `ttnn.experimental.gated_delta_net`.

---

## Kernel Inputs and Outputs

### Inputs

| Argument | Shape | dtype | Content |
|----------|-------|-------|---------|
| `conv_out` | [1, 1, B, conv_dim] | bfloat16 | Q/K/V after conv1d + SiLU |
| `z_flat` | [1, 1, B, value_dim] | bfloat16 | $z$ gate for RMSNorm |
| `ba_flat` | [1, 1, B, 2*num_v_heads] | bfloat16 | $b$ logits \|\| $a$ logits |
| `dt_bias` | [1, num_v_heads, 1, 1] | bfloat16 | Learned time-step bias |
| `neg_A_exp` | [1, num_v_heads, 1, 1] | bfloat16 | $-\exp(\texttt{A\textunderscore{}log})$ |
| `state` | [batch, num_v_heads, head_k_dim, head_v_dim] | float32 | Recurrent state $S$ |
| `norm_w` | [1, num_v_heads, 1, head_v_dim] | bfloat16 | RMSNorm scale weights |

### Scalar Parameters

| Parameter | Type | Content |
|-----------|------|---------|
| `scale` | float | $1/\sqrt{\text{head k dim}}$ |
| `norm_eps` | float | RMSNorm epsilon (1e-6) |
| `key_dim` | int | $\text{head k dim} \times \text{num k heads}$ |
| `gqa_ratio` | int | `num_v_heads / num_k_heads` |

### Outputs

| Return Index | Shape | dtype | Content |
|-------------|-------|-------|---------|
| `result[0]` | [1, num_v_heads, B, head_v_dim] | bfloat16 | Post-norm gated output |
| `result[1]` | [batch, num_v_heads, head_k_dim, head_v_dim] | float32 | Updated state $S$ |

The output is shaped [1, H, B, D] rather than [1, 1, B, H*D] because the kernel operates
per-head internally. The Python wrapper handles the reshape before the output projection:

```python
# Kernel call and state update
result   = ttnn.experimental.gated_delta_net(
    conv_out, z_flat, ba_flat,
    self._dt_bias_dev, self._neg_A_exp_dev,
    self._dev_state, self._norm_w_dev,
    scale=self.scale, norm_eps=self.args.norm_eps,
    key_dim=self.key_dim, gqa_ratio=self.gqa_ratio,
)
output_tt = result[0]
ttnn.copy(result[1], self._dev_state)   # preserve tensor address for Trace
ttnn.deallocate(result[1])

# Reshape for out_proj: [1, H, B, D] -> [1, 1, B, H*D]
if output_tt.shape[2] > 1:             # dim 2 (tile-padded rows, B_pad) > 1: permute needed
    output_tt = ttnn.permute(output_tt, [0, 2, 1, 3])
output_tt = ttnn.reshape(output_tt, [1, 1, -1, self.value_dim])
output    = ttnn.linear(output_tt, self.out_proj,
                        compute_kernel_config=self.proj_compute_config)
```

The conditional permute fires whenever `shape[2]` — the tile-padded row dimension
(`B_pad`, always 32 in production) — is greater than 1, which is effectively always.
Note that `shape[2]` is not the outer batch dimension (`shape[0]`); it is the
tile-padded batch rows that result from padding the input to the nearest tile boundary.
`ttnn.reshape` cannot flatten dim 1 (heads) into dim 3 (features) in tile layout when
`B_pad > 1`, so a `permute([0, 2, 1, 3])` is inserted to bring heads adjacent to
features first before the reshape.

---

## What the Kernel Computes

Inside the kernel the following sequence executes for each head in the batch, using the fp32
workarounds described in [`host_recurrence.md`](./host_recurrence.md):

1. **Unpack Q, K, V** from `conv_out` at positions `[0:key_dim]`, `[key_dim:2*key_dim]`,
   `[2*key_dim:conv_dim]` respectively. Apply L2 normalization to Q and K, and scale Q by
   `scale`.

2. **GQA expansion**: repeat Q and K `gqa_ratio` times to align with the V head count.

3. **Compute gate** $g = \texttt{neg\textunderscore{}A\textunderscore{}exp} \cdot \text{softplus}(a + \texttt{dt\textunderscore{}bias})$
   and **beta** $\beta = \sigma(b)$.

4. **Five-step recurrence** (fp32 throughout via SFPU binary path):
   - $S \leftarrow S \cdot \exp(g)$ — fp32 state scaled by fp32 scalar
   - $\mathbf{m} = S^\top \mathbf{k}$ — fp32 matmul (`matmul_tiles` handles fp32 CBs)
   - $\boldsymbol{\delta} = (\mathbf{v} - \mathbf{m}) \cdot \beta$
   - $S \leftarrow S + \mathbf{k} \otimes \boldsymbol{\delta}$ — rank-1 update
   - $\mathbf{o} = S^\top \mathbf{q}$

5. **Gated RMSNorm**: compute variance of $\mathbf{o}$, normalize, multiply by
   `norm_w`, multiply by $\text{SiLU}(z)$.

6. **Write outputs**: `result[0]` (bfloat16 gated output), `result[1]` (fp32 updated state).

For the full explanation of the fp32 workarounds (`init_sfpu`+`copy_tile`, SFPU binary path, `binary_dest_reuse_tiles`) and why `matmul_tiles` handles fp32 CBs differently, see [`host_recurrence.md`](./host_recurrence.md).

---

## PCC Results

### Single-Step Fused Kernel (test_a3b_pcc.py: `TestFusedKernelPCC`)

The `TestFusedKernelPCC.test_single_step` test in `tests/test_a3b_pcc.py` validates the
kernel against a PyTorch float32 reference using random inputs:

```python
assert pcc_out   >= 0.998, f"Fused kernel output PCC {pcc_out:.4f} < 0.998"
assert pcc_state >= 0.999, f"Fused kernel state PCC {pcc_state:.4f} < 0.999"
```

Observed values:
- **Output PCC: 0.999997** (vs float32 reference)
- **State PCC: 0.999999** (vs float32 reference)

These are near-lossless results. The small deviation from 1.0 is attributable to the bfloat16
precision of the Q/K/V/z/b/a inputs (which are bf16 from the projection step) and the bf16
output, not from precision loss in the fp32 state arithmetic itself.

### Multi-Step Sequential Stability (test_deltanet_multi.py)

The `test_deltanet_multi.py` reference test runs 20 sequential decode steps through a single
DeltaNet layer and prints PCC at each step. With the fused kernel, PCC remains stable above
0.999 throughout all 20 steps. The multi-step test in `test_a3b_pcc.py` (`TestDeltaNetPCC.test_multi_step`)
enforces a weaker threshold:

```python
assert min_pcc >= 0.95, f"DeltaNet multi-step min PCC {min_pcc:.4f} < 0.95"
```

This 0.95 floor is deliberately conservative to allow for the cumulative bfloat16 rounding
in the input projections while still catching catastrophic precision failures.

### Full Single-Layer PCC Tests

Both `test_pcc.py` (27B) and `test_a3b_pcc.py` (35B-A3B) run single-layer decode tests
against their respective model's float32 reference with a 0.99 PCC threshold:

```python
PCC_THRESHOLD = 0.99
assert pcc >= PCC_THRESHOLD, f"DeltaNet PCC {pcc:.6f} < {PCC_THRESHOLD}"
```

The thresholds and their rationale:

| Test | Threshold | Rationale |
|------|-----------|-----------|
| Single-step output | 0.99 | Full layer including bfp8 projection weights |
| Fused kernel output | 0.998 | Kernel-only with bf16 inputs, no weight quantization |
| Fused kernel state | 0.999 | State must preserve fp32 fidelity exactly |
| Multi-step min | 0.95 | Conservative floor for 20 accumulated decode steps |

---

## Why the Fused Kernel Is Not in Production Today

The README states:

> PCC 0.999997 vs host float32. Not used in production (from_torch overhead).
> Becomes viable with Metal Trace.

The specific issue is that without Metal Trace, each call to `ttnn.from_torch` — used to
upload tensors to device DRAM — has a fixed overhead of approximately 1–2 ms. Without Trace,
the Python runtime calls `from_torch` for several tensors on every token step as part of
the general dispatch path. For 30 DeltaNet layers this overhead accumulates to more than
the savings from eliminating the host-recurrence sync.

With Metal Trace:
- The graph of ops for a single decode step is captured once and replayed without Python
  dispatch overhead on subsequent steps.
- `from_torch` calls for weight tensors are eliminated (the trace replays addresses, not
  data uploads).
- The fused kernel's single-launch recurrence replaces the 30 individual host roundtrips.
- Projected improvement: from 86 ms/token to approximately 20–30 ms/token (the device
  compute time alone).

### Current Profiling Context

See the full latency breakdown in [`host_recurrence.md`](./host_recurrence.md) for the per-component profiling table (DeltaNet 54 ms / Attention 18 ms / norm+LM head 14 ms / Total 86 ms).

The fused kernel eliminates the 35 ms sync component. Eliminating the 26 ms Python dispatch
requires Metal Trace. Together, both optimizations are needed to approach the theoretical
compute-bound limit.

---

## Path to Production

The sequence of steps to bring the fused kernel into production is:

1. **Integrate Metal Trace** — capture the full decode forward pass as a Trace. This
   eliminates Python dispatch overhead and removes the need for `from_torch` on device
   tensors that are already resident.

2. **Replace host recurrence** — with Trace active, switch `GatedDeltaNet.forward` to
   call `ttnn.experimental.gated_delta_net` exclusively instead of falling back to the
   host path.

3. **Verify multi-step stability** — run the 20-step sequential test with Trace enabled
   to confirm that the in-place `ttnn.copy` pattern for state update works correctly
   under replay. The fixed tensor addresses required by Trace are already enforced by the
   `initialize_states` design.

4. **Optional: Multi-CQ overlap** — use a second command queue (CQ1) to prefetch the
   state tensor upload during CQ0 compute, further hiding any remaining transfer latency.

The fused kernel is production-ready from a correctness standpoint. The remaining work is
purely in the Trace integration layer.

---

**Next:** [Chapter 3 — GatedAttention: Full-Attention Layers](../ch3_gated_attention_full_attention_layers/index.md)
