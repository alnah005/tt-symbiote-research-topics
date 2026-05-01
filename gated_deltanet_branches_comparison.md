# Gated DeltaNet TTNN Implementation - Branch Comparison

## Overview

Two branches in `tt-metal` implement Gated Attention + Gated DeltaNet for Qwen3-Next architecture:
- `ign/chunked_dn` — focused on **chunked (prefill)** optimization
- `ign/delta_recurrent_gate` — focused on **recurrent (decode)** optimization

**Target Hardware**: Single Wormhole chip (N150/N300), NOT T3K
- Uses `ttnn.open_device(device_id=0)` — single device, no mesh
- Uses `ttnn.WormholeComputeKernelConfig`
- README: "Results from Wormhole N300 (DP=1)"
- Environment: `ARCH_NAME=wormhole_b0`

## Location

```
tt-metal/models/experimental/gated_attention_gated_deltanet/
├── torch_functional/           # Pure PyTorch references
│   ├── gated_attention.py      # Gated Attention (SDPA + sigmoid gate)
│   ├── gated_deltanet.py       # Gated DeltaNet full layer
│   └── delta_rule_ops.py       # Core delta-rule algorithms
├── tt/                         # TTNN implementations
│   ├── ttnn_gated_attention.py
│   ├── ttnn_gated_deltanet.py
│   └── ttnn_delta_rule_ops.py  # Core algorithms (main differences here)
└── tests/
    └── test_ttnn_validation.py # PCC validation + benchmarks
```

## Branch Comparison

| Aspect | `ign/chunked_dn` | `ign/delta_recurrent_gate` |
|--------|------------------|---------------------------|
| **Focus** | Chunked (parallel prefill) optimization | Recurrent (token-by-token decode) optimization |
| **Matrix creation** | On-device via TTNN ops (`_create_triu_ones_ttnn`, etc.) | Host-side torch then `ttnn.from_torch()` |
| **Matmul config** | Generic `_get_matmul_program_config()` | Specialized `_recurrent_outer_product_program_config()` and `_recurrent_read_query_program_config()` |
| **State update** | Standard multiply/add | `fused_decay_and_write_ttnn()` — logically fused |
| **Memory config** | Explicit `memory_config=ttnn.L1_MEMORY_CONFIG` everywhere | Some ops lack explicit L1 config |

## Key Functions in `ttnn_delta_rule_ops.py`

### `ign/chunked_dn` branch

```python
# Helper functions for on-device matrix creation (avoids host→device transfer)
_create_eye_matrix_ttnn(size, device, dtype, memory_config)
_create_triu_ones_ttnn(size, device, dtype, memory_config)
_create_tril_ones_ttnn(size, device, dtype, memory_config)
_create_strict_lower_tril_ttnn(size, device, dtype, memory_config)

# Generic matmul program config
_get_matmul_program_config(m, k, n, grid_size, in0_block_w)

# Core algorithms
l2_norm_ttnn(x, dim, eps)
recurrent_delta_rule_step_ttnn(q_t, k_t, v_t, beta_t, g_t, h)
recurrent_gated_delta_rule_ttnn(...)
chunk_gated_delta_rule_ttnn(...)  # Uses on-device triu/tril/eye
```

### `ign/delta_recurrent_gate` branch

```python
# Specialized matmul configs for recurrent step shapes
_recurrent_outer_product_program_config(device, K, V)  # k_col @ d_row → outer product
_recurrent_read_query_program_config(device, K, V)     # row @ h → read from state

# Fused state update
fused_decay_and_write_ttnn(h, k_t, delta, decay_t, beta_t, device)
# Implements: h = decay * h + beta_t * (k_t ⊗ delta)

# Core algorithms
l2_norm_ttnn(x, dim, eps)
recurrent_delta_rule_step_ttnn(q_t, k_t, v_t, beta_t, decay_t, h, seq_len, device)
recurrent_gated_delta_rule_ttnn(...)
chunk_gated_delta_rule_ttnn(...)  # Uses torch.triu/tril/eye → ttnn.from_torch()
```

## Key Commits

### `ign/chunked_dn`
- `6da4935ab3d` removed some comments
- `b0811428e72` ttnn tril triu and eye (on-device creation)
- `09e57e37e1c` wip- binary ops core program config
- `34ef041b53d` changes in conv2d L1_small to DRAM and single multiply by premultiply
- `70d05b2c059` L1 memory config, pre-computed program_configs, matmul time 587μs
- `8a102701dc2` L1 implemented for all, device time 4200μs → 3060μs

### `ign/delta_recurrent_gate`
- `4a2a431708e` optimise matmul tiling
- `c39f6cefe34` update fused operations: precompute
- `0368ab5a871` update config for sdpa matmul
- `7fef0b06835` use BF16 recurrent by default, enable matmul program config
- `4d1d5a9d580` fused recurrent delta rule step

## Algorithm Overview

**Gated DeltaNet** is a linear-attention layer using the delta rule with gated exponential decay:
- Maintains fixed-size recurrent state `[B, H, K, V]` instead of growing KV cache
- O(1) memory per token at decode time

**Two modes**:
1. **Recurrent** (decode): Token-by-token, sequential state updates
2. **Chunked** (prefill): Parallel within chunks, sequential across chunks
   - Uses Neumann series (repeated squaring) for O(log(chunk_size)) intra-chunk resolution

**State update equation**:
```
h_new = exp(g_t) * h + beta_t * (k_t ⊗ (v_t - k_t @ h))
```

Where:
- `g_t`: log-space decay
- `beta_t`: write strength
- `k_t ⊗ delta`: outer product for state write

## Potential Merge Strategy

To combine best of both branches:
1. Use `chunked_dn`'s on-device matrix creation for chunked algorithm
2. Use `delta_recurrent_gate`'s fused ops and specialized matmul configs for recurrent
3. Ensure consistent L1 memory config across all ops
4. Keep both specialized matmul configs (generic for chunked, specialized for recurrent)

---

## Woodbury Identity Resolution Analysis

**Critical finding:** Single-chunk tests fail with PCC=-0.005, indicating the core algorithm produces fundamentally wrong results.

### Mathematical Analysis

#### PyTorch Implementation (lines 196-201 of delta_rule_ops.py)
```python
# Exact iterative row-by-row Woodbury resolution
attn = -((k_beta_c @ k_c.transpose(-1, -2)) * L_mask).masked_fill(mask_upper, 0)
for i in range(1, chunk_size):
    attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
attn = attn + torch.eye(chunk_size)
```

This computes `(I - M)^{-1}` where `M` is strictly lower triangular, using row-by-row forward substitution. The algorithm is **exact** for strictly lower triangular M because:
- For strictly lower triangular `M` of size `n×n`: `M^n = 0` (nilpotent)
- `(I - M)^{-1} = I + M + M^2 + ... + M^{n-1}` (finite, exact)
- The iterative row-by-row approach computes each row using previously computed rows

#### TTNN Implementation (lines 600-616 of ttnn_delta_rule_ops.py)
```python
# Neumann series with repeated squaring
R = ttnn.add(M, eye)           # R = I + M
P = ttnn.matmul(M, M)          # P = M^2
num_steps = max(int(math.ceil(math.log2(max(chunk_size, 2)))) - 1, 0)
for _ in range(num_steps):
    R = ttnn.add(R, ttnn.matmul(R, P))  # R += R @ P
    P = ttnn.matmul(P, P)               # P = P @ P
attn = R
```

This also computes the Neumann series `I + M + M^2 + ...` using repeated squaring.

### Verification of Repeated Squaring Formula

Tracing `R = R + R @ P` with `P = P @ P`:

| Iteration | R terms | P |
|-----------|---------|---|
| Start | I + M | M^2 |
| 1 | I + M + M^2 + M^3 | M^4 |
| 2 | I + M + M^2 + ... + M^7 | M^8 |
| k | I + M + ... + M^{2^{k+1}-1} | M^{2^{k+1}} |

For chunk_size=64: `num_steps = ceil(log2(64)) - 1 = 5`
- After 5 iterations: terms up to M^63 ✓
- For 64×64 strictly lower triangular M: M^64 = 0, so series terminates at M^63 ✓

**The repeated squaring formula is mathematically correct.**

### Convergence Analysis

For strictly lower triangular M:
- All eigenvalues = 0 (nilpotent matrix)
- Spectral radius = 0 < 1
- Neumann series converges in exactly n-1 terms for n×n matrix

**Convergence is guaranteed and exact.**

### Possible Sources of Error

Since the math is correct, the PCC=-0.005 must come from:

1. **Matrix M construction differences** - Subtle shape/broadcasting issues
2. **L_mask computation** - Different cumsum or exp implementations
3. **Strict lower triangular masking** - Off-by-one in diagonal handling
4. **Floating point accumulation** - 11 matmuls vs iterative approach
5. **Broadcasting/reshape bugs** - Batch dimension handling

### Recommended Investigation Steps

1. **Unit test the M matrix construction**: Compare `M` tensor between PyTorch and TTNN before the Woodbury step
2. **Unit test L_mask**: Compare the decay mask computation
3. **Test with identity inputs**: Use k_beta = k = identity to simplify debugging
4. **Add diagnostic outputs**: Print intermediate tensors at each step

### Recommendation

**Option B: Replace with iterative implementation** is safer because:
1. The iterative approach is exactly O(chunk_size) sequential operations
2. For typical chunk_size (32-64), this is acceptable
3. It matches PyTorch exactly, guaranteeing correctness
4. The repeated squaring saves only log2(n) steps but accumulates more numerical error

However, first verify the M matrix and L_mask are correct - the bug may be upstream of the Woodbury computation.

### Implementation Plan for Iterative Approach

Replace lines 600-616 with:
```python
# Iterative Woodbury resolution (exact for strictly lower triangular M)
attn = M.clone()  # [batch, chunk_size, chunk_size]
for i in range(1, chunk_size):
    # attn[..., i, :i] += (attn[..., i, :i, None] * attn[..., :i, :i]).sum(-2)
    row_i = ttnn.slice(attn, [..., i, :i])  # [batch, i]
    submat = ttnn.slice(attn, [..., :i, :i])  # [batch, i, i]
    update = ttnn.matmul(row_i.unsqueeze(-2), submat).squeeze(-2)  # [batch, i]
    # In-place update row i (may need scatter or reconstruction)
attn = ttnn.add(attn, eye)
```

**Note**: TTNN may not support efficient in-place row updates. Alternative: reconstruct the matrix row-by-row, which is O(n^2) memory but O(n^2) compute (same as current).

### Alternative: Fix the Neumann Series

If the Neumann series implementation has a bug (rather than numerical issues), fixing it preserves the O(log n) matmul advantage. Debug by:
1. Comparing R and P at each iteration
2. Checking M and eye shapes/values
3. Verifying strict_lower mask correctness
