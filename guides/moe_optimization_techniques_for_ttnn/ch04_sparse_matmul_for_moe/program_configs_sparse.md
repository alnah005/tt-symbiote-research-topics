# Program Configs for Sparse MoE Matmul

This file explains how to select and validate a program config for `ttnn.sparse_matmul` in the MoE decode regime. It introduces the `SparsityConfig` parameter, covers grid sizing differences from the dense case, presents three worked decode-regime configurations, and contrasts them with the prefill configs from Chapter 3.

All config vocabulary (`per_core_M`, `per_core_N`, `out_subblock_h`, `out_subblock_w`, `in0_block_w`, L1 budget formulas) is defined in Chapter 2 (`matmul_fundamentals_in_ttnn.md`). The Chapter 3 prefill config (`program_configs_batched.md` §4.2) is used as the contrast baseline.

---

## 1. How sparse_matmul Program Configs Differ from Dense Matmul

### 1.1 The SparsityConfig Parameter

`ttnn.sparse_matmul` takes an additional `sparsity_config` argument that is absent from `ttnn.matmul`. This parameter is a `ttnn.SparsityConfig` object specifying how the sparsity mask is structured and stored:

```python
import ttnn

sparsity_cfg = ttnn.SparsityConfig(
    mask_layout=ttnn.ROW_MAJOR_LAYOUT,   # mask tensor layout; ROW_MAJOR for per-tile boolean
    mask_dtype=ttnn.uint8,               # one byte per tile entry; 0 = inactive, 1 = active
)
```

The mask tensor passed alongside the activation tensor at call time must match the shape $[\lceil C/32 \rceil, \lceil H/32 \rceil]$ established when the program was compiled. Passing a mask with a different shape is a runtime error.

### 1.2 Constraints Shared with Dense Matmul

The following constraints from Chapter 2 (`matmul_fundamentals_in_ttnn.md` §3) carry over unchanged:

```
out_subblock_h × out_subblock_w ≤ 8
per_core_M % out_subblock_h == 0
per_core_N % out_subblock_w == 0
M_t % grid_y == 0
N_t % grid_x == 0
K_t % in0_block_w == 0
```

These constraints apply to the per-expert tile counts $M_t$, $K_t$, $N_t$ — not the full batched $[E, C, H]$ tensor. This is the same convention as Chapter 3.

### 1.3 per_core_M and Actual Active Tiles

`per_core_M` is set based on the padded capacity $C$ (the static tensor shape), not on the actual number of active token slots. For $C=1$ (tile-padded to 32), $M_t = 1$ and `per_core_M = 1` regardless of how many experts received tokens.

The relationship between `per_core_M` and real work at decode:

$$\text{per core M} = \frac{M_t}{\text{grid y}} = \frac{\lceil C/32 \rceil}{\text{grid y}}$$

For $\rho = 0.03$ (1 active expert out of 32, $M_t = 1$), effective computed tile rows per core = $\rho \times \text{per core M} \approx 0.03 \times 1 = 0.03$. The core completes its mask checks and skips 97% of K-steps. This low per-core utilization is expected and correct: sparse_matmul is efficient precisely because the skipped tiles cost almost nothing.

### 1.4 Grid Sizing for Sparse Decode

In the dense decode config (Chapter 3, `program_configs_batched.md` §4.1), a grid of `(8, 1)` (8 cores, single row) was used. The same grid is valid for sparse decode. However, for very sparse cases ($\rho \ll 0.1$), launching an `8 × 8` grid would create 64 cores that each check their tile against a mask and skip nearly everything — wasting launch overhead.

Recommended grid sizes for sparse decode:

| $\rho$ range | Recommended grid | Rationale |
|-------------|-----------------|-----------|
| $\rho < 0.1$ | `(2, 1)` or `(4, 1)` | Minimal launch overhead; few real tiles to compute |
| $0.1 \leq \rho < 0.4$ | `(4, 1)` or `(4, 2)` | Moderate active tiles; a small 2D grid is warranted |
| $0.4 \leq \rho < 0.7$ | `(8, 1)` | Approaching dense config; single-row grid as in Ch. 3 decode |
| $\rho \geq 0.7$ | Use batched matmul | At or above $\rho^*$; switch to Ch. 3 prefill config |

For the Qwen3.5-35B MoE decode regime ($B=1$ to $B=16$, $\rho \in [0.03, 0.5]$), a `(4, 1)` grid is a good default.

---

## 2. Three Example Decode Configurations

The following configurations are for the gate or up projection matmul: activation $A$ of shape `[E_d, C_pad, H] = [32, 32, 7168]` (padded), weight $W$ of shape `[E_d, H, D]` [D UNVERIFIED — verify against Qwen3 Technical Report].

Tile counts (gate/up projection, fixed):
```
M_t = ceil(C_pad / 32) = ceil(32 / 32) = 1     (C=1, padded to 32)
K_t = ceil(7168 / 32) = 224
N_t = ceil(D / 32)                               [N_t UNVERIFIED — depends on D]
```

### 2.1 Single-Expert Decode: B=1, k=8, E=256, C=1

**Setup:** 1 input token, 8 experts activated, $N=8$ devices. Average active experts per device = 1. $C = 1$, $C_{\text{pad}} = 32$, $M_t = 1$.

Mask shape: $[1, 224]$. On average, 1 of the 32 local experts is active.

```python
import ttnn

# Sparse decode config — single-token input
# Per-expert activation: [32, 7168] (C=1, padded to 32; M_t=1, K_t=224)
# Per-expert weight:     [7168, D]  [D UNVERIFIED]
#
# Grid: (4, 1) — 4 cores in a single row (small grid for low ρ ≈ 0.03)
# per_core_M = M_t / grid_y = 1 / 1 = 1
# per_core_N = N_t / grid_x = N_t / 4   [per_core_N UNVERIFIED — depends on D]
# in0_block_w = 8  (224 % 8 == 0 ✓)
# out_subblock_h = 1  (forced by per_core_M=1)
# out_subblock_w = min(per_core_N, 8)   [UNVERIFIED — verify per_core_N first]

sparsity_cfg_b1 = ttnn.SparsityConfig(
    mask_layout=ttnn.ROW_MAJOR_LAYOUT,
    mask_dtype=ttnn.uint8,
)

sparse_decode_b1 = ttnn.MatmulMultiCoreProgramConfig(
    compute_with_storage_grid_size=(4, 1),   # (grid_x=4, grid_y=1): 4 cores
    in0_block_w=8,                           # 8 K-tiles/step; 224 % 8 == 0 ✓
    out_subblock_h=1,                        # forced by per_core_M=1
    out_subblock_w=4,                        # placeholder; set to min(N_t/4, 8) once D known [UNVERIFIED]
    per_core_M=1,                            # M_t=1, grid_y=1 → per_core_M=1
    per_core_N=4,                            # placeholder; replace with N_t/4 once D known [UNVERIFIED]
)

# Call: ttnn.sparse_matmul(activation, weight, sparsity_cfg_b1,
#           program_config=sparse_decode_b1, ...)
```

**L1 footprint estimate (placeholder per_core_N=4):**
```
A_buf = 2 × 1 × 8 × 2048 bytes =  32 768 bytes ( 32 KB)
B_buf = 2 × 8 × 4 × 1088 bytes =  69 632 bytes ( 68 KB)   [UNVERIFIED — depends on N_t/4]
C_buf = 1 × 4 × 2048 bytes      =   8 192 bytes (  8 KB)   [UNVERIFIED]
Total                            = 110 592 bytes (108 KB) ← well within 1 MB ✓
```

> **Warning:** `per_core_N=4` and `out_subblock_w=4` are placeholders. Replace with `N_t / grid_x = N_t / 4` once $D$ is confirmed. Verify `N_t % 4 == 0` and `out_subblock_w` divides `per_core_N` before use.

### 2.2 Small-Batch Decode: B=8, S=1, C=1

**Setup:** 8 input tokens, $\rho \approx 0.25$ (8 of 32 local experts active on average). Same $C = 1$, $C_{\text{pad}} = 32$, $M_t = 1$.

The per-expert matmul shape is identical to §2.1; only the sparsity pattern differs (more active experts). Because $\rho = 0.25 < \rho^* = 0.7$, `sparse_matmul` is still preferred. A slightly larger grid is appropriate:

```python
# Sparse decode config — small-batch (B=8)
# ρ ≈ 0.25: 8/32 experts active; same per-expert shape as B=1
#
# Grid: (4, 1) — same as B=1 (per_core_M=1 forced by M_t=1)
# in0_block_w = 8  (unchanged)
# out_subblock_h = 1, out_subblock_w = 4 (placeholder, same caveat as §2.1)

sparsity_cfg_b8 = ttnn.SparsityConfig(
    mask_layout=ttnn.ROW_MAJOR_LAYOUT,
    mask_dtype=ttnn.uint8,
)

sparse_decode_b8 = ttnn.MatmulMultiCoreProgramConfig(
    compute_with_storage_grid_size=(4, 1),   # (grid_x=4, grid_y=1)
    in0_block_w=8,                           # 224 % 8 == 0 ✓
    out_subblock_h=1,
    out_subblock_w=4,                        # placeholder [UNVERIFIED]
    per_core_M=1,
    per_core_N=4,                            # placeholder [UNVERIFIED]
)
```

The config object is identical to `sparse_decode_b1`. The mask tensor differs: 8 of the 32 expert-slot rows have mask bits set to 1 instead of 1 of 32. The kernel automatically skips 75% of K-steps instead of 97%. No recompilation is required — the mask **shape** $[1, 224]$ is unchanged; only the values differ.

### 2.3 Medium-Batch Decode: B=32, S=1, C=1

**Setup:** 32 input tokens, $\rho \approx 1.0$ on average (all 32 local experts active). Same $C = 1$, $C_{\text{pad}} = 32$, $M_t = 1$.

At $\rho \approx 1.0$, `sparse_matmul` provides no benefit and its overhead is a net negative. However, to illustrate the static-shape property and the config's form, the config is shown below. In production, switch to the Chapter 3 batched matmul config at this point.

```python
# Medium-batch decode config — for illustration only at B=32 (ρ ≈ 1.0)
# At this point, the Chapter 3 dense decode config is preferred.
# Shown here to confirm: the program config object is identical; only
# the decision to use sparse_matmul vs. batmul changes at the call site.

# The mask for B=32, ρ=1.0: all mask bits = 1 (every expert slot active)
# sparse_matmul will check all 224 mask entries and find all active —
# no tiles are skipped, and mask overhead adds latency relative to dense.

# For reference, same config as §2.2:
sparse_decode_b32_config = ttnn.MatmulMultiCoreProgramConfig(
    compute_with_storage_grid_size=(4, 1),
    in0_block_w=8,
    out_subblock_h=1,
    out_subblock_w=4,                        # placeholder [UNVERIFIED]
    per_core_M=1,
    per_core_N=4,                            # placeholder [UNVERIFIED]
)
# Decision rule: if ρ >= ρ* ≈ 0.7, call ttnn.matmul with decode_gate_config
# from Chapter 3 (program_configs_batched.md §4.1) instead.
```

---

## 3. Contrast with the Prefill Config from Chapter 3

The Chapter 3 prefill config (`program_configs_batched.md` §4.2) uses `MatmulMultiCoreReuseMultiCastProgramConfig` on a `(8, 2)` grid with $M_t = 2$ and `per_core_M = 1`. The sparse decode configs above use `MatmulMultiCoreProgramConfig` on a `(4, 1)` grid with $M_t = 1$ and `per_core_M = 1`.

| Parameter | Ch. 3 Prefill Config | Ch. 4 Sparse Decode Config |
|-----------|---------------------|---------------------------|
| Config type | `MatmulMultiCoreReuseMultiCastProgramConfig` | `MatmulMultiCoreProgramConfig` |
| Grid | `(8, 2)` — 16 cores | `(4, 1)` — 4 cores |
| $C$ | 64 | 1 (padded to 32) |
| $M_t$ | 2 | 1 |
| `per_core_M` | 1 | 1 |
| `per_core_N` | $N_t / 8$ [UNVERIFIED] | $N_t / 4$ [UNVERIFIED] |
| `in0_block_w` | 8 | 8 |
| `out_subblock_h` | 1 | 1 |
| `out_subblock_w` | $\leq 8$, divides `per_core_N` | $\leq 8$, divides `per_core_N` |
| Sparsity mask | None | `[1, 224]` uint8 |
| `SparsityConfig` | None | `mask_layout=ROW_MAJOR, mask_dtype=uint8` |
| $\rho$ | ~1.0 | 0.03–0.5 |
| Preferred regime | Prefill ($S \geq 512$) | Decode ($B \leq 16$, $S = 1$) |

Key differences:

1. **Grid size:** The sparse decode config uses 4 cores vs. 16 cores for prefill. Launching fewer cores reduces NoC setup overhead at high sparsity, where most tiles are skipped and the real computation is minimal.
2. **Config type:** `MatmulMultiCoreProgramConfig` (not `Reuse`) is used because $M_t = 1$ cannot fill a 2D multicast grid. This is the same reason the Chapter 3 decode config also uses `MatmulMultiCoreProgramConfig`.
3. **SparsityConfig presence:** The only structural addition in the sparse config is the `SparsityConfig` object and the mask tensor argument. The core program config parameters follow the same rules as Chapter 3.
4. **`per_core_N` denominator:** The sparse config divides $N_t$ by `grid_x=4` (vs. 8 for the prefill config), so `per_core_N` is doubled. This increases L1 B_buf usage proportionally; verify L1 budget after confirming $D$.

---

## 4. Static-Shape Requirement for Program Caching

The sparsity mask shape $[M_t, K_t] = [\lceil C/32 \rceil, 224]$ is part of the compiled kernel's loop structure and must not change between calls that share a program cache entry.

In MoE serving with T3K, the padded capacity $C$ must be fixed at deployment time:

$$C_{\text{fixed}} = \left\lceil \frac{k \times B_{\max} \times S_{\max}}{E} \right\rceil$$

For a deployment handling variable batch sizes $B \in \{1, 8, 16, 32\}$ with $S=1$:

$$C_{\text{fixed}} = \left\lceil \frac{8 \times 32 \times 1}{256} \right\rceil = 1 \implies C_{\text{pad}} = 32, \quad M_t = 1$$

All four batch sizes use $C = 1$, so the mask shape $[1, 224]$ is the same for all. A single compiled program handles all four, with different mask **values** on each call:

- $B=1$: 1 active expert slot × $K_t = 224$ K-tile columns → 224 mask bits set (out of $32 \times 224 = 7{,}168$ total).
- $B=8$: $\min(32, 8) = 8$ active expert slots × 224 → 1,792 mask bits set.
- $B=32$: $\min(32, 32) = 32$ active expert slots × 224 → 7,168 mask bits set (the full mask is set; all experts are active and sparse_matmul provides no tile-skip benefit — use batched matmul instead).

The static-shape guarantee is maintained because $C_{\text{pad}} = 32$ regardless of $B$.

If the deployment must also handle prefill ($S > 1$), a separate compiled program per $(C, M_t)$ combination is required. Maintain a config cache indexed by $(C, M_t)$ — the same pattern as recommended in Chapter 3 (`performance_profile_batched.md` §4.3) for dynamic batching.

---

## 5. Validation Checklist for Sparse Configs

The Chapter 3 validation function (`program_configs_batched.md` §5.1) applies without modification. Additionally check:

```python
def validate_sparse_config(C, H, grid_x, grid_y, in0_block_w, per_core_M, per_core_N,
                            out_subblock_h, out_subblock_w):
    """
    Sparse-specific additions to the Chapter 3 validation checklist.
    """
    M_t = -(-C // 32)     # ceil(C / 32)
    K_t = -(-H // 32)     # ceil(7168 / 32) = 224

    errors = []

    # Same checks as Chapter 3 (tile divisibility, per_core values, subblock limits)
    # ... (see program_configs_batched.md §5.1 for full list)

    # Sparse-specific check 1: mask shape consistency
    mask_M = M_t
    mask_K = K_t
    if mask_M != per_core_M * grid_y:
        errors.append(
            f"Mask M dim ({mask_M}) != per_core_M ({per_core_M}) × grid_y ({grid_y})"
        )

    # Sparse-specific check 2: MatmulMultiCoreProgramConfig for M_t=1
    if M_t == 1 and grid_y > 1:
        errors.append(
            f"M_t=1 requires grid_y=1; got grid_y={grid_y}. "
            f"Use MatmulMultiCoreProgramConfig, not Reuse variant."
        )

    # Sparse-specific check 3: grid_x small enough for sparse launch overhead
    # (advisory, not a hard error)
    if grid_x > 4:
        errors.append(
            f"Advisory: grid_x={grid_x} may add launch overhead at high sparsity. "
            f"Consider grid_x=4 for ρ < 0.1."
        )

    return errors
```

---

## Summary: Config Parameters by Regime

| Parameter | B=1 Sparse Decode | B=8 Sparse Decode | Ch. 3 Prefill ($C=64$) |
|-----------|------------------|--------------------|------------------------|
| Config type | `MatmulMultiCoreProgramConfig` | `MatmulMultiCoreProgramConfig` | `MatmulMultiCoreReuseMultiCastProgramConfig` |
| Grid | `(4, 1)` | `(4, 1)` | `(8, 2)` |
| $C$ (actual) | 1 | 1 | 64 |
| $C_{\text{pad}}$ | 32 | 32 | 64 |
| $M_t$ | 1 | 1 | 2 |
| `per_core_M` | 1 | 1 | 1 |
| `per_core_N` | $N_t/4$ [UNVERIFIED] | $N_t/4$ [UNVERIFIED] | $N_t/8$ [UNVERIFIED] |
| `in0_block_w` | 8 | 8 | 8 |
| `out_subblock_h` | 1 | 1 | 1 |
| `out_subblock_w` | $\leq 8$, divides `per_core_N` | same | same |
| Mask shape | `[1, 224]` | `[1, 224]` | N/A |
| $\rho$ | ~0.03 | ~0.25 | ~1.0 |

All `per_core_N` and `out_subblock_w` entries marked [UNVERIFIED] depend on confirmed $D$. Verify against the Qwen3 Technical Report before deploying.

---

## References

- Chapter 4, `sparse_matmul_internals.md` — Tile-skip mechanism, SparsityConfig, mask shape definition.
- Chapter 4, `when_sparse_matmul_wins.md` — $\rho$ values by regime; crossover threshold $\rho^*$.
- Chapter 3, `program_configs_batched.md` §4.1–4.2 — Baseline decode and prefill configs compared throughout this file.
- Chapter 2, `matmul_fundamentals_in_ttnn.md` — Config vocabulary, L1 budget formulas, subblock constraints.
- Chapter 3, `performance_profile_batched.md` §4.3 — Program cache management for dynamic batch sizes.

---

**Next:** [Chapter 5 — Sparsity Tensor Construction](../ch05_sparsity_tensor_construction/index.md)
