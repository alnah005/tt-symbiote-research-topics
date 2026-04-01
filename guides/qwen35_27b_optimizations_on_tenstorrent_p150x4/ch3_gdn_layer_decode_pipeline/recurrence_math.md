# Recurrence Math: DeltaNet Equations and Tensor Operation Mapping

The DeltaNet recurrence is the core computation of every GDN layer. It maintains a state matrix $S$ of shape `[Dk, Dv]` per (batch, value head) pair and updates it with each new token. The recurrence can be understood as a linear attention mechanism where the state matrix plays the role of a compressed KV cache: it accumulates key-value outer products scaled by a learned gating mechanism, and the query reads out from it via matrix-vector multiplication.

This section presents the mathematical equations, their gate computations, the head expansion strategy, and the mapping to `ttnn` operations in the unfused path. The fused kernel (Chapter 4) implements the same math but in a single dispatch.

## Preprocessing: L2 Normalization and Scaling

Before entering the recurrence, Q and K are L2-normalized along their last dimension and Q is scaled (`gdn.py` lines 413–426):

$$q_{\text{normed}} = \frac{q}{\sqrt{\|q\|^2 + \varepsilon}}$$

$$k_{\text{normed}} = \frac{k}{\sqrt{\|k\|^2 + \varepsilon}}$$

$$q_{\text{scaled}} = q_{\text{normed}} \cdot Dk^{-0.5}$$

where $\varepsilon = 10^{-6}$ and $Dk = 128$. The scale factor $Dk^{-0.5} = 128^{-0.5} \approx 0.0884$ prevents the dot products in the recurrence from growing with the key dimension.

In the unfused path, L2 normalization is performed by the `_l2_norm_dev` helper (`gdn.py` lines 58–67):

```python
def _l2_norm_dev(x):
    x_sq = ttnn.multiply(x, x)                   # element-wise square
    ssq = ttnn.sum(x_sq, dim=-1, keepdim=True)   # sum of squares along last dim
    ttnn.deallocate(x_sq)
    inv = ttnn.rsqrt(ttnn.add(ssq, 1e-6))        # 1 / sqrt(sum_sq + eps)
    ttnn.deallocate(ssq)
    normed = ttnn.multiply(x, inv)                # normalize
    ttnn.deallocate(inv)
    return normed
```

A key optimization: normalization is applied to Q and K while they still have `Nk_TP=4` heads, before the `repeat_interleave` expansion to `Nv_TP=12` heads. This saves 3x compute on the normalization operations.

After normalization:

```python
q_exp = ttnn.repeat_interleave(q_normed, repeat_factor, dim=1)  # [B, 4, Dk] -> [B, 12, Dk]
q_ns = ttnn.multiply(q_exp, self.scale)                          # apply Dk^(-0.5) scale
k_exp = ttnn.repeat_interleave(k_normed, repeat_factor, dim=1)  # [B, 4, Dk] -> [B, 12, Dk]
```

## Head Expansion: Key Heads to Value Heads

The GDN architecture uses fewer key heads ($Nk=16$) than value heads ($Nv=48$). After TP=4 sharding, each device has `Nk_TP=4` key heads and `Nv_TP=12` value heads, giving a `repeat_factor = Nv_TP / Nk_TP = 3`.

The `repeat_interleave` operation duplicates each key head 3 times to match the value heads:

```
Key heads:    [k0, k1, k2, k3]
After expand: [k0, k0, k0, k1, k1, k1, k2, k2, k2, k3, k3, k3]
Value heads:  [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]
```

Value heads `v0, v1, v2` all share key head `k0`; `v3, v4, v5` share `k1`; and so on. Each (batch, value head) combination forms a "pair" — the unit of work for the recurrence. The total number of pairs is `num_pairs = B * Nv_TP = 32 * 12 = 384` per device.

In the fused kernel, the head expansion is handled implicitly through the pair-to-head mapping: `k_head = v_head / repeat_factor` (integer division).

## Gate Computation

Two scalar gates control the recurrence dynamics for each pair.

### Beta Gate (Update Strength)

$$\beta = \sigma(b)$$

where $b$ is the per-pair scalar from the AB projection. Beta ranges in $(0, 1)$ and controls how strongly the new key-value observation updates the state.

### Decay Gate (Exponential Forgetting)

$$g = -\exp(A\_\text{log}) \cdot \text{softplus}(a + \text{dt bias})$$

Breaking this down:
- `A_log`: a learned per-head log-space decay rate, precomputed as `neg_exp_A = -exp(A_log)` during `_precompute_constants()` (lines 145–148)
- `a`: the per-pair scalar from the AB projection
- `dt_bias`: a learned per-head bias (`tw["dt_bias"]`)
- $\text{softplus}(x) = \log(1 + e^x)$: ensures the inner term is always positive
- The negation from `neg_exp_A` makes $g$ always negative, so $\exp(g)$ is a decay factor in $(0, 1)$

In the unfused path (lines 429–434):

```python
beta_tt = ttnn.sigmoid(b_tt)
sp = ttnn.softplus(ttnn.add(a_tt, tw["dt_bias"]))
g_pre = ttnn.multiply(self.neg_exp_A, sp)
```

The decay factor applied to the state is $\exp(g)$, which is computed inside the recurrence kernel.

## The DeltaNet Recurrence

For each pair (one batch element, one value head), the recurrence updates a state matrix $S$ of shape `[Dk, Dv]` = `[128, 128]`. Given the preprocessed inputs for one token:

- $q$: row vector `[1, Dk]` — L2-normalized, scaled query
- $k_{\text{row}}$: row vector `[1, Dk]` — L2-normalized key
- $k_{\text{col}}$: column vector `[Dk, 1]` — transpose of $k_{\text{row}}$
- $v$: row vector `[1, Dv]` — value
- $g$: scalar — decay gate (negative)
- $\beta$: scalar — update gate

The recurrence performs 5 steps:

### Step 1: Exponential Decay

$$S \leftarrow S \cdot \exp(g)$$

The state is element-wise multiplied by the decay factor $\exp(g)$. $g$ is always negative per the gate definition above. The decay rate is learned per-head via `A_log` and modulated per-token via $a + \text{dt bias}$.

### Step 2: Key-State Product (Memory Readout for Key)

$$kv\_\text{mem} = k_{\text{row}} \cdot S \quad \rightarrow \quad [1, Dk] \times [Dk, Dv] = [1, Dv]$$

This reads out from the state matrix using the key as a query, producing a vector that represents what the state currently "remembers" about this key direction. This is used in the next step to compute the innovation — how much new information the current value provides beyond what is already stored.

### Step 3: Scaled Innovation (Delta)

$$\delta = \beta \cdot (v - kv\_\text{mem}) \quad \rightarrow \quad \text{scalar} \times [1, Dv] = [1, Dv]$$

The innovation $v - kv\_\text{mem}$ measures the difference between the new value $v$ and what the state already predicts for this key. Scaling by $\beta$ controls the update magnitude. When $\beta$ is small, the state changes slowly; when $\beta$ is large, it aggressively incorporates new information.

This is the "delta" in DeltaNet — the update is proportional to the prediction error, similar to the delta rule in classical neural network learning.

### Step 4: Rank-1 State Update

$$S \leftarrow S + k_{\text{col}} \cdot \delta \quad \rightarrow \quad [Dk, Dv] + [Dk, 1] \times [1, Dv] = [Dk, Dv]$$

The outer product $k_{\text{col}} \cdot \delta$ produces a `[Dk, Dv]` matrix that is added to the state. This is a rank-1 update: it modifies the state along the key direction by the scaled innovation amount. The state matrix accumulates these rank-1 updates across all tokens, building a compressed representation of the key-value history.

### Step 5: Query Readout

$$\text{output} = q \cdot S \quad \rightarrow \quad [1, Dk] \times [Dk, Dv] = [1, Dv]$$

The query reads out from the updated state, producing a `[1, Dv]` = `[1, 128]` output vector. This is analogous to the attention output in standard transformers: the query selects relevant information from the stored key-value associations.

## Recurrence State Shape and Memory

The recurrence state for all pairs on one device is stored as a single contiguous tensor (`gdn.py` lines 179–181):

```
rec_states: [B * Nv_TP, Dk, Dv] = [32 * 12, 128, 128] = [384, 128, 128]
```

In bfloat16, each pair's `[128, 128]` state occupies `128 * 128 * 2 = 32,768 bytes = 32 KB`. With tile padding (4x4 tiles of 32x32), this is `16 tiles * 2048 bytes/tile = 32,768 bytes` — no padding overhead since both dimensions are exact multiples of the tile size.

Per-device totals:
- Per pair: `32 KB`
- Per layer: `384 pairs * 32 KB = 12,288 KB ≈ 12 MB` (no tile padding — both dimensions are exact multiples of 32)
- All 48 GDN layers: `48 * 12 MB ≈ 576 MB` per device

This state is read and written every decode step for every GDN layer, making DRAM bandwidth the primary bottleneck. Chapter 6 discusses the L1 state optimization that addresses this.

## Tensor Operation Mapping Summary

The following table maps each mathematical operation to its `ttnn` implementation in the unfused path and its kernel phase in the fused path:

| Math | Unfused (`ttnn` ops) | Fused kernel phase |
|------|---------------------|-------------------|
| $q / \|q\|$ | `_l2_norm_dev(q_h)` (4 ops: mul, sum, rsqrt, mul) | Phase 1: transpose, matmul dot, rsqrt, mul, scale |
| $k / \|k\|$ | `_l2_norm_dev(k_h)` (4 ops) | Phase 2: transpose, matmul dot, rsqrt, mul |
| $k^T$ | `ttnn.transpose(k_row, -2, -1)` | Phase 3: `transpose_wh_tile` |
| $\beta = \sigma(b)$ | `ttnn.sigmoid(b_tt)` | Phase 4: `sigmoid` |
| $g = \text{neg A} \cdot \text{sp}(a + \text{bias})$ | `ttnn.softplus(ttnn.add(a, bias))`, `ttnn.multiply(neg_A, sp)` | Phase 4: `exp`, `log1p`, mul by `neg_exp_A` |
| $S \mathrel{*}= \exp(g)$ | Inside recurrence kernel | Phase 5: `exp(g)` broadcast multiply |
| $kv\_\text{mem} = k \cdot S$ | Inside recurrence kernel | Phase 5: matmul |
| $\delta = \beta \cdot (v - kv\_\text{mem})$ | Inside recurrence kernel | Phase 5: sub, mul |
| $S \mathrel{+}= k^T \cdot \delta$ | Inside recurrence kernel | Phase 5: copy + matmul accumulate |
| $\text{out} = q \cdot S$ | Inside recurrence kernel | Phase 5: matmul |

## Numerical Precision

`TtGatedDeltaNet` uses two distinct compute configs depending on the execution path:

**Unfused path (projection matmuls):** `self.compute_cfg = args.compute_kernel_config_hifi2` (`gdn.py:116`) uses HiFi2 math for the unfused projection matmuls (QKVZ, AB, output). HiFi2 truncates one mantissa operand for higher throughput.

**Fused kernel (recurrence core):** The custom `gdn_full_fused_inplace` kernel uses a separate HiFi4 config (`gdn_kernel_op.py:485–490`):

```python
config=ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    dst_full_sync_en=False,
)
```

The fused kernel uses HiFi4 with `math_approx_mode=False` for full-precision recurrence. The FP32 accumulator (`fp32_dest_acc_en=True`) is shared across both paths and is critical for the state update ($S \mathrel{+}= k^T \cdot \delta$) where small updates to a large state matrix could otherwise be lost to bfloat16 rounding.

---

**Next:** [Chapter 4 — Custom Fused GDN Kernel](../ch4_custom_fused_gdn_kernel/index.md)
