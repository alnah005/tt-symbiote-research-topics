# GDN Decode Flow: End-to-End Dataflow for a Single Layer

Each GDN decode step takes a hidden-state input `x` of shape `[1, 1, B, dim]` (where B=32 and dim=5120), updates the layer's mutable conv and recurrence states, and produces an output of the same shape after an all-reduce across the 4 TP devices. The `TtGatedDeltaNet.forward_decode()` method in `gdn.py` dispatches to one of two paths: the fused path (`_forward_decode_fused`) that offloads L2 normalization, gating, and recurrence to a single custom kernel dispatch, or the unfused path (`_forward_decode_unfused`) that decomposes the same computation into individual `ttnn` operations. The fused path is used by default; the unfused path serves as a fallback activated by the `GDN_DISABLE_FULL_FUSED` environment variable or when the fused kernel raises an exception.

Both paths share the same four-stage structure:

1. **Projections** -- Linear projections produce QKVZ and gate inputs
2. **Conv1d** -- A 4-tap causal convolution via shift register
3. **Recurrence** -- The DeltaNet state update and readout
4. **Post-processing** -- RMS norm, SiLU gating, output projection, all-reduce

This section traces each stage, highlighting the differences between fused and unfused paths.

## Stage 1: Projections

The input `x` arrives as `[1, 1, B, dim]` from the framework and is first reshaped to `[1, B, dim]` (3D) for the matmul operations.

### QKVZ Projection

A single DRAM-sharded matmul produces the fused QKVZ output:

```python
qkvz_tt = _unshard(_shard_linear(x, tw["qkvz"], act_shard, self.args.gdn_qkvz_progcfg, self.compute_cfg))
```

The `_shard_linear` helper first shards the activation to L1 WIDTH_SHARDED, then dispatches the matmul with the DRAM-sharded weight, producing an L1 WIDTH_SHARDED result. The `_unshard` helper moves the result to DRAM interleaved memory for subsequent slicing.

The output shape is `[1, B, qkvz_dim_tp]` = `[1, 32, 4096]`, where `qkvz_dim_tp = (GDN_QKV_DIM + GDN_Z_DIM) / TP = (10240 + 6144) / 4 = 4096`. This is split into two tensors via `ttnn.slice`:

- `qkv_tt`: `[1, B, qkv_dim_tp]` = `[1, 32, 2560]` -- the concatenated Q, K, V projections
- `z_tt`: `[1, B, z_dim_tp]` = `[1, 32, 1536]` -- the SiLU gate projection

The dimension breakdown of `qkv_dim_tp = 2560`:
- Q: `Nk_TP * Dk = 4 * 128 = 512` (key heads, key dim)
- K: `Nk_TP * Dk = 4 * 128 = 512` (key heads, key dim)
- V: `Nv_TP * Dv = 12 * 128 = 1536` (value heads, value dim)
- Total: `512 + 512 + 1536 = 2560`

The `qkvz_tt` tensor is immediately deallocated after slicing to free DRAM.

### AB Projection

A separate matmul produces the gate inputs for the recurrence:

```python
ab_tt = ttnn.linear(x, tw["ab"])
```

This produces `[1, B, Nv_TP * 2]` = `[1, 32, 24]`, split into:

- `a_tt`: `[1, B, Nv_TP]` = `[1, 32, 12]` -- input to the exponential decay gate
- `b_tt`: `[1, B, Nv_TP]` = `[1, 32, 12]` -- input to the sigmoid beta gate

The AB projection is small (only 24 output dimensions per device) and uses a simple `ttnn.linear` without DRAM sharding.

## Stage 2: Conv1d (Shift Register)

The `qkv_tt` tensor passes through a 4-tap causal conv1d implemented as a shift register (see [`conv1d_shift_register.md`](./conv1d_shift_register.md) for the detailed mechanism). The conv1d output `conv_out` has the same shape `[1, B, qkv_dim_tp]` = `[1, 32, 2560]` and contains the convolved Q, K, and V values concatenated along the last dimension, followed by a SiLU activation.

## Stage 3: Recurrence

This is where the fused and unfused paths diverge significantly.

### Fused Path (`_forward_decode_fused`)

The fused path passes `conv_out` directly to the custom kernel `gdn_full_fused_inplace`, which handles everything from Q/K/V extraction through L2 normalization, gating, and the DeltaNet recurrence in a single dispatch:

```python
gdn_full_fused_inplace(
    conv_out, a_tt, b_tt,
    self.neg_exp_A, tw["dt_bias"], tw["norm_w"], self.scale_tt, self.rms_scale_tt, self.rms_eps_tt,
    self.rec_states, self.fused_output,
    num_pairs=num_pairs,
    num_cores=min(96, num_pairs),
    Nv_TP=Nv_TP, Nk_TP=Nk_TP, repeat_factor=repeat_factor,
    key_dim_tp=key_dim_tp,
)
```

Key arguments:
- `conv_out`: `[1, B, qkv_dim_tp]` -- the kernel's reader extracts Q/K/V per-pair via sub-tile row reads (detailed in Chapter 4)
- `a_tt`, `b_tt`: gate scalar inputs, unshard'd to DRAM before the kernel call
- `self.neg_exp_A`: precomputed `-exp(A_log)`, a per-head decay rate constant
- `tw["dt_bias"]`: learned bias added to the `a` gate input before softplus
- `tw["norm_w"]`: RMS norm weight for the recurrence readout (used inside the kernel for L2 norm, and again post-kernel for output RMS norm)
- `self.scale_tt`, `self.rms_scale_tt`, `self.rms_eps_tt`: precomputed scalar tiles for Q scale (`Dk^{-0.5}`), RMS scale (`sqrt(Dv)`), and RMS epsilon (`Dv * 1e-6`)
- `self.rec_states`: the mutable recurrence state `[B*Nv_TP, Dk, Dv]` = `[384, 128, 128]`, updated in-place
- `self.fused_output`: pre-allocated output buffer `[num_pairs, 1, Dv]` = `[384, 1, 128]`
- `num_pairs = B * Nv_TP = 32 * 12 = 384`: the total (batch, value_head) pairs to process
- `repeat_factor = Nv_TP / Nk_TP = 12 / 4 = 3`: how many value heads share each key head

The output in `self.fused_output` contains the raw recurrence readout -- one `[1, Dv]` vector per pair.

### Unfused Path (`_forward_decode_unfused`)

The unfused path decomposes the recurrence into explicit `ttnn` operations. After the conv1d, it:

1. **Splits Q/K/V** from `conv_out` via `ttnn.slice` at dimension boundaries `[0, key_dim_tp)`, `[key_dim_tp, 2*key_dim_tp)`, `[2*key_dim_tp, qkv_dim_tp)` and reshapes to head format: Q as `[B, Nk_TP, Dk]`, K as `[B, Nk_TP, Dk]`, V as `[B, Nv_TP, Dv]`

2. **L2 normalizes** Q and K via the `_l2_norm_dev` helper (element-wise square, sum, rsqrt, multiply), operating on the `Nk_TP=4` key heads before expansion

3. **Expands key heads** to value heads via `ttnn.repeat_interleave(q_normed, repeat_factor=3, dim=1)`, producing `[B, Nv_TP, Dk]` = `[32, 12, 128]` for both Q and K. Q is also multiplied by the scale factor `Dk^{-0.5} = 128^{-0.5}`

4. **Computes gates**: `beta = sigmoid(b)` and `g = neg_exp_A * softplus(a + dt_bias)` (see [`recurrence_math.md`](./recurrence_math.md))

5. **Reshapes to pair format** `[num_pairs, ...]` using `_retile` for correct tile layout, then calls `gdn_recurrence_fused_inplace` which handles only the recurrence itself (not L2 norm or gating)

6. **Reads out** via `q @ state` inside the kernel

The unfused path performs L2 normalization on the smaller `Nk_TP=4` tensor before expanding to `Nv_TP=12`, saving 3x compute on the normalization. The `_retile` helper (round-trip through ROW_MAJOR and back to TILE_LAYOUT) is necessary because `ttnn.reshape` changes the logical shape without re-tiling the underlying data.

## Stage 4: Post-Processing

Post-processing is identical for both paths:

### RMS Norm

The recurrence output is reshaped from pair format to head format `[B, Nv_TP, Dv]` and normalized:

```python
out_r = ttnn.reshape(self.fused_output, (B, Nv_TP, Dv))
out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6)
```

The `norm_w` weight is a learned per-element scale applied after RMS normalization.

### SiLU Gating

The Z projection from Stage 1 is activated with SiLU and element-wise multiplied with the normalized output:

```python
z_act = ttnn.silu(z_tt)
gated = ttnn.multiply(out_f, z_act)
```

The gated tensor has shape `[1, B, value_dim_tp]` = `[1, 32, 1536]`.

### Output Projection and All-Reduce

The gated output passes through the row-parallel output projection:

```python
out_partial = _unshard(_shard_linear(gated, tw["out"], act_shard_out, self.args.gdn_out_progcfg, self.compute_cfg))
```

This DRAM-sharded matmul maps from `value_dim_tp=1536` (the TP-local value dimension) to `dim=5120` (the full hidden dimension). Because this is a row-parallel projection (input sharded along the input dimension), each device produces a partial sum that must be combined via all-reduce:

```python
out_partial = ttnn.reshape(out_partial, (1, 1, B, out_partial.shape[-1]))
return self._all_reduce(out_partial)
```

The `_all_reduce` method calls `tt_all_reduce` with the CCL ring topology, summing partial results across all 4 devices to produce the final `[1, 1, B, dim]` output.

## Memory Management

Both paths follow strict `ttnn.deallocate` discipline. Every intermediate tensor is explicitly deallocated as soon as it is no longer needed. This is critical because GDN layers process `num_pairs = 384` pairs worth of data, and without deallocations, intermediate tensors would accumulate and exhaust device memory.

Key deallocation points:
- `qkvz_tt` is deallocated immediately after slicing into `qkv_tt` and `z_tt`
- `ab_tt` is deallocated after slicing into `a_tt` and `b_tt`
- `conv_out` is deallocated after being consumed by the fused kernel (or after Q/K/V slicing in unfused)
- `a_tt` and `b_tt` are deallocated after the fused kernel returns (or after gate computation in unfused)
- In the unfused path, each reshape/expand intermediate is deallocated before the next operation

The recurrence state `self.rec_states` and output buffer `self.fused_output` (or `self.rec_output` in unfused) are persistent -- they survive across decode steps and are only reset via `reset_state()` or `reset_state_inplace()`.

## Fused vs Unfused: Summary

| Aspect | Fused (`_forward_decode_fused`) | Unfused (`_forward_decode_unfused`) |
|--------|--------------------------------|-------------------------------------|
| Kernel dispatches for recurrence | 1 (`gdn_full_fused_inplace`) | 1 (`gdn_recurrence_fused_inplace`) + ~15 `ttnn` ops |
| Q/K/V extraction | Reader kernel sub-tile reads | `ttnn.slice` + `ttnn.reshape` |
| L2 normalization | Inside fused kernel | `_l2_norm_dev` helper (4 `ttnn` ops each) |
| Gate computation | Inside fused kernel | `ttnn.sigmoid`, `ttnn.softplus`, `ttnn.multiply` |
| Head expansion | Inside fused kernel (pair-to-head mapping) | `ttnn.repeat_interleave` |
| Re-tiling needed | No | Yes (`_retile` for correct tile layout after reshape) |
| Fallback trigger | Default path | `GDN_DISABLE_FULL_FUSED` env var or fused kernel exception |

The fused path eliminates the overhead of multiple kernel dispatches, intermediate tensor allocations, and the `_retile` round-trip through ROW_MAJOR layout. Chapter 4 details the internal structure of the `gdn_full_fused_inplace` kernel.

---

**Previous:** [`index.md`](./index.md) | **Next:** [`conv1d_shift_register.md`](./conv1d_shift_register.md)
