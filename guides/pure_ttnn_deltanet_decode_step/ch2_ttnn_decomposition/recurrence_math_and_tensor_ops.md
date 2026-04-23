# DeltaNet Recurrence: Math and TTNN Tensor Operations

This file derives the six operations of the gated DeltaNet recurrent decode step, specifying the mathematical formula, tensor shapes for the single-head case, batched multi-head shapes for Qwen3.6-35B-A3B, and the TTNN primitive for each operation.

## Model dimensions (Qwen3.6-35B-A3B, B=1)

| Symbol    | Value | Meaning                              |
|-----------|-------|--------------------------------------|
| B         | 1     | Batch size                           |
| nH        | 32    | Number of value heads (num_v_heads)  |
| d_k       | 128   | Key/query head dimension             |
| d_v       | 128   | Value head dimension                 |

## Gated DeltaNet Recurrence (per head h, decode step t)

At each decode step, given the previous state `S_{t-1}` of shape `[d_k, d_v]`:

```
S_decayed  = g_t  *  S_{t-1}              [d_k, d_v]
retrieval  = S_{t-1}^T  @  k̃_t           [d_v]
error      = β_t * (v_t − retrieval)      [d_v]
write      = k̃_t ⊗ error                  [d_k, d_v]
S_t        = S_decayed + write            [d_k, d_v]
o_t        = S_t^T  @  q̃_t               [d_v]
```

The inputs at step t are:
- `g_t ∈ ℝ` — scalar decay gate (one per head)
- `β_t ∈ ℝ` — scalar update rate (one per head)
- `k̃_t ∈ ℝ^{d_k}` — L2-normalized key
- `q̃_t ∈ ℝ^{d_k}` — L2-normalized query
- `v_t ∈ ℝ^{d_v}` — value

---

## Operation 1: Decay

**Math:** `S_decayed = g_t * S_{t-1}`

The scalar gate `g_t` is broadcast over the full state matrix, suppressing older memory.

| Representation | Shape             | Concrete shape       |
|----------------|-------------------|----------------------|
| Single head    | `[d_k, d_v]`      | `[128, 128]`         |
| Batched nH     | `[B, nH, d_k, d_v]` | `[1, 32, 128, 128]` |
| g broadcast    | `[B, nH, 1, 1]`   | `[1, 32, 1, 1]`      |
| S_decayed      | `[B, nH, d_k, d_v]` | `[1, 32, 128, 128]` |

TTNN call: `ttnn.mul(g_broadcast, S_prev)` — see [`ttnn_ops_per_step.md`](./ttnn_ops_per_step.md) for the full annotated sequence.

---

## Operation 2: Retrieval

**Math:** `retrieval = S_{t-1}^T @ k̃_t`

`k̃_t` lives in `d_k`-space. The state `S: [d_k, d_v]` maps d_v-space to d_k-space. Its transpose `S^T: [d_v, d_k]` maps d_k-space to d_v-space, which is what retrieval and output readout require: given a d_k-space input (k̃ or q̃), `S^T @ input` produces a d_v-space result.

| Representation | Shape             | Concrete shape       |
|----------------|-------------------|----------------------|
| S_prev         | `[B, nH, d_k, d_v]` | `[1, 32, 128, 128]` |
| k_tilde        | `[B, nH, d_k, 1]` | `[1, 32, 128, 1]`   |
| retrieval      | `[B, nH, d_v, 1]` | `[1, 32, 128, 1]`   |

TTNN call: `ttnn.matmul(S_prev, k_tilde, transpose_a=True)` — see [`ttnn_ops_per_step.md`](./ttnn_ops_per_step.md).

> **Note:** `retrieval` uses `S_{t-1}` (the pre-decay state), NOT `S_decayed`. The DeltaNet delta rule update formula requires the pre-decay state for the retrieval computation: the error is the difference between what the un-decayed memory predicts for key `k̃_t` and the actual value `v_t`. Using `S_decayed` here would under-estimate the prediction and produce an incorrect error signal. Operations (1) and (2) are independent reads of `S_{t-1}` and may be issued in either order, but retrieval must never read `S_decayed`.

---

## Operation 3: Error

**Math:** `error = β_t * (v_t − retrieval)`

The scalar `β_t ∈ (0, 1)` (post-sigmoid) controls how aggressively the memory is updated toward the new association.

| Representation | Shape             | Concrete shape       |
|----------------|-------------------|----------------------|
| v_t            | `[B, nH, d_v, 1]` | `[1, 32, 128, 1]`   |
| retrieval      | `[B, nH, d_v, 1]` | `[1, 32, 128, 1]`   |
| beta_broadcast | `[B, nH, 1, 1]`   | `[1, 32, 1, 1]`     |
| error          | `[B, nH, d_v, 1]` | `[1, 32, 128, 1]`   |

TTNN call: `ttnn.sub(v_t, retrieval)` then `ttnn.mul(beta_broadcast, error_raw)` — see [`ttnn_ops_per_step.md`](./ttnn_ops_per_step.md).

---

## Operation 4: Write (outer product)

**Math:** `write = k̃_t ⊗ error` (outer product)

The outer product of a `[d_k]` column vector with a `[d_v]` row vector produces a `[d_k, d_v]` rank-1 update matrix. In batched matrix form: `[d_k, 1] × [1, d_v] → [d_k, d_v]`.

| Representation | Shape             | Concrete shape       |
|----------------|-------------------|----------------------|
| k_tilde        | `[B, nH, d_k, 1]` | `[1, 32, 128, 1]`   |
| error^T        | `[B, nH, 1, d_v]` | `[1, 32, 1, 128]`   |
| write          | `[B, nH, d_k, d_v]` | `[1, 32, 128, 128]` |

TTNN call: `ttnn.matmul(k_tilde, ttnn.transpose(error, -2, -1))` — see [`ttnn_ops_per_step.md`](./ttnn_ops_per_step.md).

---

## Operation 5: New State

**Math:** `S_t = S_decayed + write`

Elementwise addition of two matrices of identical shape.

| Representation | Shape             | Concrete shape       |
|----------------|-------------------|----------------------|
| S_decayed      | `[B, nH, d_k, d_v]` | `[1, 32, 128, 128]` |
| write          | `[B, nH, d_k, d_v]` | `[1, 32, 128, 128]` |
| S_new          | `[B, nH, d_k, d_v]` | `[1, 32, 128, 128]` |

TTNN call: `ttnn.add(S_decayed, write)` — see [`ttnn_ops_per_step.md`](./ttnn_ops_per_step.md).

---

## Operation 6: Output

**Math:** `o_t = S_t^T @ q̃_t`

`q̃_t` lives in `d_k`-space, just like `k̃_t`. The new state `S_t^T: [d_v, d_k]` maps the query from `d_k`-space to `d_v`-space to produce the output.

| Representation | Shape             | Concrete shape       |
|----------------|-------------------|----------------------|
| S_new          | `[B, nH, d_k, d_v]` | `[1, 32, 128, 128]` |
| q_tilde        | `[B, nH, d_k, 1]` | `[1, 32, 128, 1]`   |
| o_t            | `[B, nH, d_v, 1]` | `[1, 32, 128, 1]`   |

TTNN call: `ttnn.matmul(S_new, q_tilde, transpose_a=True)` — see [`ttnn_ops_per_step.md`](./ttnn_ops_per_step.md).

> **Note:** Both `retrieval` (op 2) and `o_t` (op 6) use `S^T`, not `S`. This is because the query and key vectors (`q̃_t`, `k̃_t`) both live in `d_k`-space. The state matrix `S: [d_k, d_v]` encodes associations from value-space to key-space (it left-multiplies d_v-length vectors to produce d_k-length vectors). To retrieve a d_v-space output given a d_k-space input, the transpose `S^T: [d_v, d_k]` is required. To retrieve from it given a key, or to read out from it given a query, one must apply the transpose: `S^T: [d_v, d_k]` left-multiplied by a `[d_k, 1]` vector produces a `[d_v, 1]` output in value-space. Applying `S` directly (without transpose) would produce a `[d_k, 1]` result in key-space. In this model `d_k = d_v = 128`, so there is no dimensional inconsistency — the shape `[d_k, 1] = [128, 1]` is identical to `[d_v, 1] = [128, 1]` and TTNN would not raise a shape error. The error would be **silent**: numerically wrong values with no runtime signal.

---

## Tile Alignment

Tile alignment for all six operations is analyzed in [`state_tensor_memory_config.md` — Tile Alignment Analysis](./state_tensor_memory_config.md).

---

## Summary Table

| Op | Math                         | TTNN Primitive                                               | Output shape         |
|----|------------------------------|--------------------------------------------------------------|----------------------|
| 1  | g_t * S_{t-1}               | `ttnn.mul(g_broadcast, S_prev)`                              | [B, nH, d_k, d_v]   |
| 2  | S_{t-1}^T @ k̃_t            | `ttnn.matmul(S_prev, k_tilde, transpose_a=True)`             | [B, nH, d_v, 1]     |
| 3a | v_t − retrieval             | `ttnn.sub(v_t, retrieval)`                                   | [B, nH, d_v, 1]     |
| 3b | β_t * (v_t − retrieval)     | `ttnn.mul(beta_broadcast, error_raw)`                        | [B, nH, d_v, 1]     |
| 4  | k̃_t ⊗ error                 | `ttnn.matmul(k_tilde, ttnn.transpose(error, -2, -1))`        | [B, nH, d_k, d_v]   |
| 5  | S_decayed + write           | `ttnn.add(S_decayed, write)`                                 | [B, nH, d_k, d_v]   |
| 6  | S_t^T @ q̃_t                | `ttnn.matmul(S_new, q_tilde, transpose_a=True)`              | [B, nH, d_v, 1]     |
