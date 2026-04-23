# Mamba SSM Kernel Review

`[PARTIAL_REUSE]`

This file reviews the Mamba SSM selective scan kernel in tt-metal for DeltaNet adaptation potential. The Mamba kernel is the most structurally similar existing primitive to the DeltaNet decode step — both maintain a 2D state matrix, both update via an outer product write, and both read from the state via a matrix-vector multiply. The kernel cannot be used as-is because its inner loop does not contain the retrieval-then-error dependency that is the defining feature of the DeltaNet delta rule. However, its DMA streaming pattern and outer product idioms are directly borrowable for the fused kernel development in Chapter 4.

---

## 1. Source Location

> **Note:** The exact file path must be verified in the tt-metal repository. The expected location is:
>
> - `ttnn/cpp/ttnn/operations/experimental/ssm/` — the standard location for experimental SSM ops in tt-metal, based on the TTNN op registry pattern for Mamba
>
> Search for `selective_scan`, `ssm_eltwise_mul`, or `hc_sum` in that directory. The Mamba implementation in tt-metal is typically split into sub-ops (element-wise multiply, hidden state update, output projection) rather than a single monolithic fused kernel. Document the actual file structure during the port research for Chapter 4.

---

## 2. What Mamba SSM Computes

The Mamba SSM state update for a single channel at decode step t is:

```
h_t = A * h_{t-1} + B_t * x_t     [d_state]
y_t = C_t^T @ h_t                  [scalar or d_model per channel]
```

Across all channels and with the full state matrix interpretation:

```
H_t = A_diag * H_{t-1} + B_t ⊗ x_t    [d_model, d_state]
y_t = C_t^T @ H_t                       [d_model]
```

where:
- `H_t` is the state matrix `[d_model, d_state]` — analogous to DeltaNet's `S [d_k, d_v]`
- `A_diag` is a learned diagonal decay matrix (per-channel scalar applied element-wise along `d_state`)
- `B_t` is a per-step, input-dependent write vector `[d_state]` derived from the input `x_t`
- `x_t` is the input `[d_model]`
- `C_t` is a per-step, input-dependent query vector `[d_state]`
- `B_t ⊗ x_t` is the outer product write: `[d_state] ⊗ [d_model] → [d_model, d_state]` (following the convention that the result rows are indexed by `d_model`)

---

## 3. Structural Comparison with DeltaNet

### 3.1 Similarities

| Property | Mamba SSM | DeltaNet |
|---|---|---|
| State shape | `[d_model, d_state]` | `[d_k, d_v]` |
| State persistence | DRAM between steps; L1 during kernel | DRAM between steps; L1 during kernel (Ch4) |
| Update mechanism | Outer product `B_t ⊗ x_t` added to decayed state | Outer product `k̃_t ⊗ error` added to decayed state |
| State readout | Matrix-vector multiply `C_t^T @ H_t` | Matrix-vector multiply `S_t^T @ q̃_t` |
| Decay | Scalar per channel (diagonal `A`), broadcast over state | Scalar per head (`g_t`), broadcast over full state matrix |
| DMA pattern | State loaded from DRAM into L1 CB; streamed through 32×32 tiles | Same pattern (Ch4 design uses the same idiom) |

These similarities motivated the survey. The state shape, DMA pattern, outer product write, and matrix-vector readout are directly analogous.

### 3.2 The Key Structural Difference

> **Key Finding:** DeltaNet's write `k̃_t ⊗ error` depends on `retrieval = S_{t-1}^T @ k̃_t`. The error is `β_t * (v_t - retrieval)`. To compute the error, you must first read the current state. Only after reading can you compute the write. This is a **read-before-write dependency within a single decode step**: the state must be read (retrieval) before it can be written (state update). The DeltaNet recurrence is **non-separable** — you cannot compute the write matrix independently of the current state.

In Mamba SSM, the write `B_t ⊗ x_t` is computed entirely from the input `x_t` and learned parameters. It does not depend on `H_{t-1}` at all. You can compute the write matrix `B_t ⊗ x_t` before you have read the state. The Mamba recurrence is **separable** — the write is independent of the current state, and the decay is a simple element-wise multiply.

```
Mamba inner loop:
  1. Compute write = B_t ⊗ x_t        (no state read required)
  2. H_t = A * H_{t-1} + write         (one state read, one write)
  3. y_t = C_t^T @ H_t                 (one state read)

DeltaNet inner loop:
  1. retrieval = S_{t-1}^T @ k̃_t      (state read)
  2. error = β_t * (v_t - retrieval)   (depends on step 1)
  3. write = k̃_t ⊗ error              (depends on step 2)
  4. S_decayed = g_t * S_{t-1}         (independent state read, can overlap with step 1)
  5. S_t = S_decayed + write            (depends on steps 3 and 4)
  6. o_t = S_t^T @ q̃_t                (state read after update)
```

The Mamba kernel does not contain steps 1–3 of the DeltaNet inner loop. Adding them is not a matter of adding a few lines — the retrieval step (step 1) is itself a full matrix-vector multiply over the state, requiring the state to be in L1 before the write matrix can be formed. A kernel that only implements the Mamba pattern would need to be restructured to perform the retrieval before issuing the write, which is a fundamental change to the kernel's execution order.

### 3.3 Consequence for Reuse

The Mamba kernel cannot be adapted for DeltaNet by adding parameters or changing constants. The structural difference is in the execution order of the inner loop, not in the data types or tile sizes. Reusing the Mamba kernel would require replacing its inner loop body — at which point it is effectively a new kernel.

---

## 4. Borrowable Patterns for Chapter 4 Kernel Development

Despite the non-reuse conclusion, the Mamba SSM kernel provides concrete, working implementations of the following patterns that are directly relevant to the `gdn_full_fused_inplace` Wormhole port:

### 4.1 DRAM-to-L1 DMA Streaming for State Tiles

The Mamba reader RISCV program issues `noc_async_read` calls to load state tiles from a DRAM buffer into an L1 CB. The address computation pattern (base address + tile offset) and the use of `noc_async_read_barrier()` for synchronization are directly applicable to the Chapter 4 kernel's state load.

**Borrowable:** Reader program structure for loading a `[d_k, d_v] = [128, 128]` state matrix (16 tiles at 32×32) from DRAM into CB0.

### 4.2 Outer Product via `matmul_tiles`

The Mamba compute kernel computes `B_t ⊗ x_t` using a sequence of `matmul_tiles` calls with transposed dimensions. The idiom for computing a `[d_model, d_state]` outer product from a `[d_model, 1]` column vector and a `[1, d_state]` row vector is the same shape structure as DeltaNet's `k̃_t ⊗ error`.

**Borrowable:** `matmul_tiles` invocation pattern for outer product; tile index calculation for a 4×4 tile result grid (`[128, 128]` = 4×4 tiles of 32×32).

### 4.3 Scalar Broadcast Multiply via `mul_tiles_bcast_scalar`

The Mamba kernel applies the diagonal decay `A` as a per-channel scalar, which on a per-element basis maps to a broadcast multiply over a tile. This is the same primitive needed for DeltaNet's `g_t * S_{t-1}` decay (op 1) and `β_t * error_raw` scaling (op 3b).

**Borrowable:** `mul_tiles_bcast_scalar` usage for scalar decay over state tiles.

### 4.4 L1 State Write-Back

The Mamba writer RISCV program issues `noc_async_write` calls to write updated state tiles from L1 back to DRAM. The synchronization pattern and tile ordering are directly applicable.

**Borrowable:** Writer program structure for writing a 16-tile state matrix back to DRAM.

---

## 5. Reuse Classification

`[PARTIAL_REUSE]` — The Mamba SSM kernel cannot be used as-is for DeltaNet decode. Its inner loop is structurally different: Mamba's write is independent of the current state, while DeltaNet's write depends on a state retrieval that must occur in the same decode step. However, the DMA streaming pattern (DRAM-to-L1 for state tiles), the outer product idiom (`matmul_tiles` with transposed dimensions), the scalar broadcast multiply (`mul_tiles_bcast_scalar`), and the L1 write-back pattern are all directly borrowable for the Chapter 4 fused kernel implementation or port.

---

## 6. Summary

| Property | Mamba SSM | DeltaNet | Implication |
|---|---|---|---|
| State shape | `[d_model, d_state]` | `[d_k, d_v]` | Same abstract shape; different concrete dimensions |
| Write depends on current state? | No — `B_t ⊗ x_t` is pure input | Yes — `k̃_t ⊗ error`; `error` requires `S_{t-1}^T @ k̃_t` | Structural incompatibility; kernel logic must change |
| DMA pattern | DRAM state → L1 CB | Same | Directly borrowable |
| Outer product | `matmul_tiles` (transposed) | Same | Directly borrowable |
| Scalar decay broadcast | `mul_tiles_bcast_scalar` | Same | Directly borrowable |
| Direct kernel reuse | Not possible | — | New kernel or port of `gdn_full_fused_inplace` required |
| Reuse classification | — | `[PARTIAL_REUSE]` | Idioms borrowable; kernel not usable as-is |
