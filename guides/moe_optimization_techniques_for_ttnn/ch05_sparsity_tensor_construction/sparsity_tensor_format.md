# Sparsity Tensor Format

This file defines the exact format contract for the sparsity tensor accepted by `ttnn.sparse_matmul`. Every property described here is a hard requirement — violating any of them produces either a runtime error or silent wrong output.

See Chapter 4, `sparse_matmul_internals.md` for the kernel-side view of how these properties are consumed.

---

## 1. Shape

The sparsity tensor is a 2D tensor of shape $[M_t, K_t]$, where:

$$M_t = \left\lceil \frac{C}{32} \right\rceil, \quad K_t = \left\lceil \frac{H}{32} \right\rceil$$

- $M_t$ is the number of tile rows in the activation tensor $A$ (shape $[C, H]$).
- $K_t$ is the number of tile columns in $A$.

For Qwen3.5-35B, $H = 7168$, so $K_t = 7168 / 32 = 224$ exactly.

For the batched multi-expert case where all $E_d$ local experts share one flattened activation tensor of shape $[E_d \times C, H]$, the mask shape is:

$$[E_d \times M_t,\ K_t]$$

Each contiguous block of $M_t$ rows in the mask corresponds to one expert. The block for expert $e$ occupies mask rows $[e \times M_t,\ (e+1) \times M_t)$.

> **Warning:** The mask shape must exactly match the tile dimensions of the activation tensor. If the activation tensor is padded to a larger shape (e.g., $C$ padded to the next multiple of 32), $M_t$ must be derived from the **padded** $C$, not the logical token count. See `common_pitfalls.md`, P1.

---

## 2. Value Encoding

Each element of the mask is a scalar tile-level flag:

| Value | Meaning |
|-------|---------|
| `0` | Skip this tile row — `sparse_matmul` does not read or compute this tile row |
| `1` | Compute this tile row — `sparse_matmul` performs the full matmul for this tile row |

TTNN operates at tile granularity (32×32 elements). There is no mechanism to skip individual rows within a tile. If tile row $m$ is marked `1`, all 32 rows within that tile are computed, including any padding rows that may not contain real tokens.

### K-dimension entries

The activation tensor $A$ has shape $[E_d \times C, H]$. The sparsity structure is exclusively in the $M$-dimension (which tokens are routed to which expert). The $K$-dimension (the hidden dimension $H$) is always fully active for any computed tile row.

Therefore, for a given tile row $m$:
- If tile row $m$ is **active**: all $K_t$ entries in mask row $m$ are `1`.
- If tile row $m$ is **inactive**: all $K_t$ entries in mask row $m$ are `0`.

The mask is effectively a column vector of $E_d \times M_t$ binary flags, broadcast across $K_t$ columns. In practice, you construct it this way (see `constructing_from_router_output.md`, Step 4).

---

## 3. Required dtype

The sparsity tensor must have dtype `ttnn.uint8` (8-bit unsigned integer). BF16 and FP32 masks are not accepted by `ttnn.sparse_matmul` and will raise a dtype validation error at call time.

> **Warning:** PyTorch's default floating-point dtype (`torch.float32`) produces a tensor that is incompatible with the mask dtype requirement. When converting from a PyTorch boolean or integer mask, always specify `dtype=ttnn.uint8` in the `ttnn.from_torch` call. See `common_pitfalls.md`, P5.

---

## 4. Required Layout

The sparsity tensor must use `ttnn.TILE_LAYOUT`. `ttnn.ROW_MAJOR_LAYOUT` is not accepted.

This means the tensor's in-memory representation is organized in 32×32 tiles. For a uint8 mask with tile layout:

- One tile occupies $32 \times 32 \times 1\ \text{byte} = 1024\ \text{bytes}$.
- The total number of tiles in the mask is $\lceil (E_d \times M_t) / 32 \rceil \times \lceil K_t / 32 \rceil$.

For Qwen3.5-35B decode ($M_t = 1$, $E_d = 32$, $K_t = 224$):

$$\text{tiles} = \left\lceil \frac{32}{32} \right\rceil \times \left\lceil \frac{224}{32} \right\rceil = 1 \times 7 = 7\ \text{tiles} = 7168\ \text{bytes} \approx 7\ \text{KB}$$

For prefill ($M_t = 8$, $E_d = 32$, $K_t = 224$):

$$\text{tiles} = \left\lceil \frac{256}{32} \right\rceil \times \left\lceil \frac{224}{32} \right\rceil = 8 \times 7 = 56\ \text{tiles} = 57344\ \text{bytes} \approx 56\ \text{KB}$$

The layout conversion from a row-major PyTorch tensor to TTNN tile layout is handled automatically by `ttnn.from_torch` when `layout=ttnn.TILE_LAYOUT` is specified.

---

## 5. Required Memory Placement

The mask can be placed in either L1 or DRAM, but the choice has performance consequences.

**L1 (recommended for decode):** The mask is accessed once per tile row during `sparse_matmul` execution. With L1 placement, each access hits the fast on-chip SRAM (1.5 MB per Tensix core on Wormhole B0). For decode-sized masks (7 KB), the entire mask fits comfortably in L1 with no eviction pressure.

**DRAM (acceptable under L1 pressure):** If L1 is heavily utilized by activation or weight tensors, the mask can be placed in DRAM. This adds a DRAM bandwidth cost on every mask read during kernel execution. For small masks this cost is modest; at larger prefill sizes it becomes more significant.

See `sparsity_tensor_placement.md` for a detailed analysis and concrete placement recommendations.

---

## 6. Dimension Correspondence

The two dimensions of the mask correspond to the two dimensions of the activation tensor $A$:

| Mask dimension | Mask size | Activation tensor dimension | Activation tensor size |
|---|---|---|---|
| dim 0 (rows) | $E_d \times M_t$ | dim 0 (M-dimension) | $E_d \times C$ |
| dim 1 (cols) | $K_t$ | dim 1 (K-dimension) | $H$ |

For mask row $m$ (in the flattened $[E_d \times M_t, K_t]$ view), setting all $K_t$ entries to 0 instructs `sparse_matmul` to skip activation rows $[m \times 32,\ (m+1) \times 32)$ entirely.

---

## 7. Partial Tiles at Boundary

When $C$ is not a multiple of 32, the last tile row of each expert's block covers fewer than 32 real token slots. Specifically, the last tile row for expert $e$ covers rows:

$$[(e \times M_t + M_t - 1) \times 32,\ e \times C + C)$$

where only $C \bmod 32$ of the 32 slots are real (the remaining $32 - (C \bmod 32)$ are padding zeros).

**The mask value for this partial tile row must be `1` if the tile row contains any real token, and `0` only if no token was routed to any of the $C \bmod 32$ real slots in that tile.**

Setting the partial tile row to `0` when it contains real tokens is a silent correctness bug — those tokens are silently dropped from the expert computation. See `common_pitfalls.md`, P2.

> **Warning:** For decode with $B=1$, $S=1$: $C=1$, $M_t=1$. The single tile row contains exactly 1 real token and 31 padding zeros. If routing assigned a token to this expert, the mask entry **must be `1`**.

---

## 8. Per-Expert Mask vs. Per-Layer Mask

There are two ways to organize the mask:

**Option A — One mask per expert (shape $[M_t, K_t]$):** Each expert gets an independent mask. This requires $E_d$ separate mask tensors and $E_d$ separate `sparse_matmul` calls.

**Option B — One mask for all local experts (shape $[E_d \times M_t, K_t]$):** All $E_d$ local experts are stacked into a single batched activation tensor of shape $[E_d \times C, H]$, and the corresponding mask has shape $[E_d \times M_t, K_t]$. A contiguous block of $M_t$ mask rows being all-zero means the corresponding expert is entirely inactive for this step.

Option B is the standard configuration for Qwen3.5-35B on TTNN. The mask structure follows the batched activation tensor layout described in Chapter 3.

---

## 9. Static Mask Shape Requirement

TTNN's program caching compiles one kernel binary per unique combination of (activation shape, mask shape, weight shape, dtype, memory config). A change in any of these attributes invalidates the cached program and triggers JIT recompilation.

Because the mask shape $[E_d \times M_t, K_t]$ depends on $M_t = \lceil C / 32 \rceil$, and $C$ depends on $B$ and $S$, a model that is called with variable batch sizes or sequence lengths will produce different mask shapes across calls. This forces recompilation on every shape change.

**Fix:** Maintain a small set of canonical $(B, S, C)$ shapes. Pad the activation tensor and mask to the nearest canonical shape. The mask padding rows should be all-zero (inactive), which is correct behavior — padded rows are not real tokens and should be skipped.

See `common_pitfalls.md`, P6 for symptoms and detection.

---

## References

- Chapter 4, `sparse_matmul_internals.md` — kernel-side sparsity tensor consumption, tile-skip implementation
- Chapter 2, TTNN Wormhole primer — tile layout, memory config API, L1 capacity (1.5 MB per core on Wormhole B0)
- `constructing_from_router_output.md` — how to build a mask of this format from router top-k indices
- `sparsity_tensor_placement.md` — L1 vs. DRAM placement analysis
- `common_pitfalls.md` — P1 (shape), P2 (partial tiles), P4 (DRAM), P5 (dtype), P6 (recompilation)

---

**Next:** [constructing_from_router_output.md](./constructing_from_router_output.md)
