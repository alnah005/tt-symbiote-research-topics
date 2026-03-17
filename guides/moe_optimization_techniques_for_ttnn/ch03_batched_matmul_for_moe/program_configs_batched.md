# Program Configs for Batched MoE Matmul

This file explains how to select and validate a `MatmulMultiCoreReuseMultiCastProgramConfig` for the batched expert FFN matmul formulated in `formulating_batched_matmul.md`. It covers parameter derivation, L1 budget estimation, and a complete validation checklist.

All config vocabulary (`out_subblock_h`, `out_subblock_w`, `per_core_M`, `per_core_N`, `in0_block_w`, L1 budget formulas) is defined in Chapter 2 (`matmul_fundamentals_in_ttnn.md`) and is used here without re-derivation.

---

## 1. Config Selection for the Batched Case

### 1.1 Which Config to Use

The choice between `MatmulMultiCoreReuseMultiCastProgramConfig` and `MatmulMultiCoreProgramConfig` depends on the per-expert matmul shape — that is, the `[C, H]` × `[H, D]` problem for each expert slice. The batch dimension $E=256$ is unrolled inside the kernel and does not affect this choice.

Apply the Chapter 2 config selection rule (`matmul_fundamentals_in_ttnn.md` Section 4.3): use `MatmulMultiCoreProgramConfig` when C=1 (decode), and `MatmulMultiCoreReuseMultiCastProgramConfig` when C≥4 (prefill/high-C regime).

For the batched MoE case:

- $K_t = \lceil H / 32 \rceil = \lceil 7168 / 32 \rceil = 224$ — this is always $\geq 32$, so multicast amortization is favorable whenever $M_t \times N_t$ fills at least a 2×2 grid.
- $M_t = \lceil C / 32 \rceil$ — depends on capacity $C$ and therefore on the serving regime (decode vs. prefill).
- $N_t = \lceil D / 32 \rceil$ — [N_t UNVERIFIED — verify against Qwen3 Technical Report]

In the decode regime ($C=1$, tile-padded to $C_\text{pad}=32$, so $M_t=1$), a single-row M dimension cannot fill a 2D grid; use `MatmulMultiCoreProgramConfig`.

In the prefill regime ($C \geq 64$, $M_t \geq 2$), the full 2D grid can be used with `MatmulMultiCoreReuseMultiCastProgramConfig`.

### 1.2 Tile Count Derivation

For the gate or up projection matmul (`[C, H]` × `[H, D]`):

$$M_t = \left\lceil \frac{C}{32} \right\rceil$$

$$K_t = \left\lceil \frac{H}{32} \right\rceil = \left\lceil \frac{7168}{32} \right\rceil = 224$$

$$N_t = \left\lceil \frac{D}{32} \right\rceil \quad \text{[N\_t UNVERIFIED — verify against Qwen3 Technical Report]}$$

For the down projection (`[C, D]` × `[D, H]`):

$$M_t = \left\lceil \frac{C}{32} \right\rceil \quad \text{(same as above)}$$

$$K_t = \left\lceil \frac{D}{32} \right\rceil \quad \text{[K\_t for down proj UNVERIFIED — verify against Qwen3 Technical Report]}$$

$$N_t = \left\lceil \frac{H}{32} \right\rceil = \left\lceil \frac{7168}{32} \right\rceil = 224$$

---

## 2. Key Config Parameters

### 2.1 `per_core_M` and How It Scales with Capacity C

`per_core_M` is the number of output tile rows each core computes. Given a grid with `grid_y` rows:

$$\text{per\_core\_M} = \frac{M_t}{\text{grid\_y}} = \frac{\lceil C/32 \rceil}{\text{grid\_y}}$$

For this to be a positive integer, $M_t$ must be divisible by `grid_y`. As $C$ grows (higher batch or longer sequences), $M_t$ grows proportionally and the core grid can accommodate more tile rows per core.

| Capacity $C$ | $M_t = \lceil C/32 \rceil$ | grid_y suggestion | `per_core_M` |
|-------------|--------------------------|-------------------|--------------|
| 1 (decode, tile-padded to 32) | 1 | 1 | 1 |
| 32 | 1 | 1 | 1 |
| 64 | 2 | 2 | 1 |
| 128 | 4 | 4 | 1 |
| 256 | 8 | 4 | 2 |
| 512 | 16 | 8 | 2 |
| 2048 (prefill S=2048) | 64 | 8 | 8 |

> **Tip:** Keeping `per_core_M = 1` (by choosing `grid_y = M_t`) is often optimal for MoE expert matmuls where $M_t$ is small relative to $N_t$. With `per_core_M = 1`, the subblock constraint forces `out_subblock_h = 1`, which is always valid, and the N dimension drives register utilization.

### 2.2 `per_core_N` and How It Scales with D

`per_core_N` is the number of output tile columns each core computes:

$$\text{per\_core\_N} = \frac{N_t}{\text{grid\_x}} = \frac{\lceil D/32 \rceil}{\text{grid\_x}} \quad \text{[D UNVERIFIED — verify against Qwen3 Technical Report]}$$

For the down projection, $N_t = 224$ (the H dimension), so:

$$\text{per\_core\_N} = \frac{224}{\text{grid\_x}}$$

For grid_x=8: `per_core_N = 28`. For grid_x=4: `per_core_N = 56`.

### 2.3 `in0_block_w`

`in0_block_w` controls how many K tiles are loaded and processed per inner loop iteration. It must divide $K_t$ evenly.

For the gate/up projection: $K_t = 224$. Valid `in0_block_w` values are divisors of 224: 1, 2, 4, 7, 8, 14, 16, 28, ...

For the down projection: $K_t = \lceil D/32 \rceil$ [UNVERIFIED — verify against Qwen3 Technical Report]. Valid values depend on this unverified quantity.

Larger `in0_block_w` increases L1 usage (more A and B tiles buffered per loop step) but reduces the number of loop iterations, which can improve compute-to-DRAM-latency overlap. The practical limit is the L1 budget; see Section 3.

### 2.4 `out_subblock_h` and `out_subblock_w`

These are defined in Chapter 2 (`matmul_fundamentals_in_ttnn.md`, Section 3). The constraints are:

```
per_core_M % out_subblock_h == 0
per_core_N % out_subblock_w == 0
out_subblock_h × out_subblock_w ≤ 8
```

Selection strategy for the MoE batched case:

1. If `per_core_M = 1`: set `out_subblock_h = 1`, then `out_subblock_w = min(per_core_N, 8)`.
2. If `per_core_M = 2`: set `out_subblock_h = 2`, then `out_subblock_w = min(per_core_N, 4)` (product ≤ 8).
3. If `per_core_M ≥ 4`: prefer `out_subblock_h = 4`, `out_subblock_w = 2` (product = 8, maximum efficiency) when `per_core_N` is divisible by 2; or `out_subblock_h = 2`, `out_subblock_w = 4` when `per_core_N` is divisible by 4.

---

## 3. L1 Budget Formula and Estimation

The L1 budget formulas are from Chapter 2. For one core in the batched expert matmul, with mixed dtype (BFP8 weights, BF16 activations and output):

```
A_buf = 2 × per_core_M × in0_block_w × tile_bytes_A
B_buf = 2 × in0_block_w × per_core_N × tile_bytes_B
C_buf = per_core_M × per_core_N × tile_bytes_C
Total ≤ ~1 MB (practical budget after kernel code overhead)
```

Where:
- `tile_bytes_A = 2048` (BF16 activation tiles, 32×32 × 2 bytes)
- `tile_bytes_B = 1088` (BFP8 weight tiles, 32×32 × 1 byte + 64 bytes exponent overhead)
- `tile_bytes_C = 2048` (BF16 output tiles)

The factor of 2 in `A_buf` and `B_buf` accounts for double-buffering: one buffer is being filled from DRAM via DMA while the other is being consumed by the FPU.

---

## 4. Example Configurations

### 4.1 Decode Regime: C=1, S=1

**Context:** $B=32$, $S=1$, $C=1$ (tile-padded to 32), $H=7168$, gate/up projection.

**Tile counts:**
```
M_t = ceil(32 / 32) = 1    (C padded to 32 for tile alignment)
K_t = ceil(7168 / 32) = 224
N_t = ceil(D / 32)          [N_t UNVERIFIED — verify against Qwen3 Technical Report]
```

**Config selection:** $M_t = 1$ → only a single tile row → cannot fill a 2×2 grid on the M axis. Use `MatmulMultiCoreProgramConfig`.

```python
import ttnn

# Decode-regime expert FFN gate/up projection
# Per-expert matmul: [32, 7168] × [7168, D] → [32, D]
# M_t=1, K_t=224, N_t=ceil(D/32) [N_t UNVERIFIED]
#
# Grid: (grid_x=8, grid_y=1) — 8 cores in a single row
# per_core_M = M_t / grid_y = 1 / 1 = 1
# per_core_N = N_t / grid_x = N_t / 8  [per_core_N UNVERIFIED — depends on D]
# in0_block_w = 8  (divisor of K_t=224; keep A_buf small for decode)
# out_subblock_h = 1  (forced by per_core_M=1)
# out_subblock_w = min(per_core_N, 8)  [UNVERIFIED — verify per_core_N first]

decode_gate_config = ttnn.MatmulMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, 1),   # (grid_x, grid_y): single row
    in0_block_w=8,                           # 8 K-tiles per loop step; K_t=224, 224%8==0 ✓
    out_subblock_h=1,                        # forced by per_core_M=1
    out_subblock_w=4,                        # placeholder; set to min(per_core_N, 8) once D is known
    per_core_M=1,                            # 1 tile row per core (M_t=1)
    per_core_N=4,                            # placeholder; replace with N_t/8 once D is known [UNVERIFIED]
)
```

> **Warning:** The `per_core_N` and `out_subblock_w` values above are placeholders because $D$ is unverified. Replace them once $D$ is confirmed from the Qwen3 Technical Report. Verify `N_t % per_core_N == 0` before using.

**L1 footprint estimate (with placeholder per_core_N=4):**
```
A_buf = 2 × 1 × 8 × 2048 bytes = 32 KB
B_buf = 2 × 8 × 4 × 1088 bytes ≈ 68 KB   [depends on actual per_core_N — UNVERIFIED]
C_buf = 1 × 4 × 2048 bytes = 8 KB         [depends on actual per_core_N — UNVERIFIED]
Total ≈ 108 KB  ← well within 1 MB budget (placeholder estimate only)
```

### 4.2 Prefill Regime: C=64, S=2048 (single sequence)

**Context:** $B=1$, $S=2048$, $C = \lceil 8 \times 1 \times 2048 / 256 \rceil = 64$, $H=7168$, gate/up projection.

**Tile counts:**
```
M_t = ceil(64 / 32) = 2
K_t = ceil(7168 / 32) = 224
N_t = ceil(D / 32)      [N_t UNVERIFIED — verify against Qwen3 Technical Report]
```

**Config selection:** $M_t = 2 \geq 2$, $N_t \geq 2$ (assuming $D \geq 64$), and $K_t = 224 \geq 32$ → use `MatmulMultiCoreReuseMultiCastProgramConfig`.

**Grid choice:** Use a `2 × 8` grid (grid_y=2, grid_x=8) to distribute across 16 cores.

```
per_core_M = M_t / grid_y = 2 / 2 = 1
per_core_N = N_t / grid_x = N_t / 8   [per_core_N UNVERIFIED — depends on D]
```

```python
import ttnn

# Prefill-regime expert FFN gate/up projection
# Per-expert matmul: [64, 7168] × [7168, D] → [64, D]
# M_t=2, K_t=224, N_t=ceil(D/32) [N_t UNVERIFIED]
#
# Grid: (8, 2) → 16 cores
# per_core_M = 2/2 = 1
# per_core_N = N_t/8  [UNVERIFIED — depends on D]
# in0_block_w = 8  (224 % 8 == 0 ✓)
# out_subblock_h = 1  (forced by per_core_M=1)
# out_subblock_w = min(per_core_N, 8)  [UNVERIFIED — verify per_core_N first]

prefill_gate_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 2),   # (grid_x=8, grid_y=2) → 16 cores
    in0_block_w=8,                           # 8 K-tiles/step; 224 % 8 == 0 ✓
    out_subblock_h=1,                        # forced by per_core_M=1
    out_subblock_w=4,                        # placeholder; set to min(per_core_N, 8) once D known [UNVERIFIED]
    per_core_M=1,                            # M_t=2, grid_y=2 → per_core_M=1
    per_core_N=4,                            # placeholder; replace with N_t/8 once D is known [UNVERIFIED]
    transpose_mcast=False,
    fused_activation=None,                   # SiLU/SwiGLU activation can be fused here
)
```

**L1 footprint estimate (placeholder per_core_N=4):**
```
A_buf = 2 × 1 × 8 × 2048 bytes = 32 KB
B_buf = 2 × 8 × 4 × 1088 bytes ≈ 68 KB   [UNVERIFIED — depends on actual per_core_N]
C_buf = 1 × 4 × 2048 bytes = 8 KB         [UNVERIFIED — depends on actual per_core_N]
Total ≈ 108 KB  ← well within budget (placeholder estimate only)
```

### 4.3 Down Projection Config

The down projection swaps the K and N dimensions relative to the gate/up projection. For the gate/up projection, $K_t = 224$ (H dimension). For the down projection, $K_t = \lceil D/32 \rceil$ [UNVERIFIED] and $N_t = 224$ (H dimension).

```python
# Down projection: [C, D] × [D, 7168] → [C, 7168]
# K_t = ceil(D/32)  [UNVERIFIED — verify against Qwen3 Technical Report]
# N_t = ceil(7168/32) = 224
#
# Grid: (8, grid_y) — keep per_core_N = 224/8 = 28
# in0_block_w: must divide K_t = ceil(D/32)  [UNVERIFIED]
# out_subblock_h: depends on per_core_M
# out_subblock_w: min(28, 8) → out_subblock_w=4 (28 % 4 == 0 ✓), out_subblock_h × 4 ≤ 8

# Placeholder for decode (M_t=1, per_core_M=1):
decode_down_config = ttnn.MatmulMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, 1),
    in0_block_w=4,             # divisor of K_t=ceil(D/32); placeholder [UNVERIFIED]
    out_subblock_h=1,          # forced by per_core_M=1
    out_subblock_w=4,          # per_core_N=28, 28%4==0 ✓, 1×4=4 ≤ 8 ✓
    per_core_M=1,
    per_core_N=28,             # N_t=224, grid_x=8 → per_core_N=28
)

# Placeholder for prefill (M_t=2, per_core_M=1):
prefill_down_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(8, 2),
    in0_block_w=4,             # divisor of K_t=ceil(D/32); placeholder [UNVERIFIED]
    out_subblock_h=1,          # forced by per_core_M=1
    out_subblock_w=4,          # per_core_N=28, 28%4==0 ✓, 1×4=4 ≤ 8 ✓
    per_core_M=1,              # M_t=2, grid_y=2
    per_core_N=28,             # N_t=224, grid_x=8
    transpose_mcast=False,
    fused_activation=None,
)
```

---

## 5. Validation Checklist

Use this checklist before running a program config. Each check can be performed without launching a kernel.

### 5.1 Tile Divisibility Checks

```python
def validate_config(C, H, D, grid_x, grid_y, in0_block_w, per_core_M, per_core_N,
                    out_subblock_h, out_subblock_w, projection="gate"):
    """
    Validate a MatmulMultiCoreReuseMultiCastProgramConfig before use.
    D is unverified — substitute the confirmed value once known.
    """
    # 1. Tile counts
    M_t = -(-C // 32)           # ceil(C / 32)
    K_t_gate_up = -(-H // 32)   # ceil(7168/32) = 224
    N_t_gate_up = -(-D // 32)   # ceil(D/32) [D UNVERIFIED — verify against Qwen3 Technical Report]
    K_t_down    = N_t_gate_up   # down proj: K is D
    N_t_down    = K_t_gate_up   # down proj: N is H

    if projection in ("gate", "up"):
        M_t_eff, K_t_eff, N_t_eff = M_t, K_t_gate_up, N_t_gate_up
    else:  # down
        M_t_eff, K_t_eff, N_t_eff = M_t, K_t_down, N_t_down

    errors = []

    # 2. Grid divisibility
    if M_t_eff % grid_y != 0:
        errors.append(f"M_t ({M_t_eff}) % grid_y ({grid_y}) != 0")
    if N_t_eff % grid_x != 0:
        errors.append(f"N_t ({N_t_eff}) % grid_x ({grid_x}) != 0")

    # 3. per_core values match
    if per_core_M != M_t_eff // grid_y:
        errors.append(f"per_core_M mismatch: expected {M_t_eff // grid_y}, got {per_core_M}")
    if per_core_N != N_t_eff // grid_x:
        errors.append(f"per_core_N mismatch: expected {N_t_eff // grid_x}, got {per_core_N}")

    # 4. K loop divisibility
    if K_t_eff % in0_block_w != 0:
        errors.append(f"K_t ({K_t_eff}) % in0_block_w ({in0_block_w}) != 0")

    # 5. Subblock divisibility
    if per_core_M % out_subblock_h != 0:
        errors.append(f"per_core_M ({per_core_M}) % out_subblock_h ({out_subblock_h}) != 0")
    if per_core_N % out_subblock_w != 0:
        errors.append(f"per_core_N ({per_core_N}) % out_subblock_w ({out_subblock_w}) != 0")

    # 6. Register file limit
    if out_subblock_h * out_subblock_w > 8:
        errors.append(f"out_subblock_h × out_subblock_w = {out_subblock_h * out_subblock_w} > 8")

    return errors  # empty list means config is valid
```

### 5.2 L1 Footprint Estimation

```python
def estimate_l1_bytes(per_core_M, per_core_N, in0_block_w,
                       tile_bytes_A=2048, tile_bytes_B=1088, tile_bytes_C=2048):
    """
    Estimate per-core L1 usage in bytes.
    tile_bytes_A: 2048 for BF16 activations
    tile_bytes_B: 1088 for BFP8 weights
    tile_bytes_C: 2048 for BF16 output
    Returns total bytes; must stay below ~1 MB (1_048_576 bytes).
    """
    A_buf = 2 * per_core_M * in0_block_w * tile_bytes_A     # double-buffered
    B_buf = 2 * in0_block_w * per_core_N * tile_bytes_B     # double-buffered
    C_buf = per_core_M * per_core_N * tile_bytes_C
    total = A_buf + B_buf + C_buf
    return {
        "A_buf_bytes": A_buf,
        "B_buf_bytes": B_buf,
        "C_buf_bytes": C_buf,
        "total_bytes": total,
        "fits_1MB": total <= 1_048_576,
    }

# Example: decode down projection (per_core_M=1, per_core_N=28, in0_block_w=4)
# Note: K_t for down proj = ceil(D/32); in0_block_w=4 is a placeholder [D UNVERIFIED]
result = estimate_l1_bytes(per_core_M=1, per_core_N=28, in0_block_w=4)
# A_buf:  2 × 1 × 4 × 2048 =  16 384 bytes ( 16 KB)
# B_buf:  2 × 4 × 28 × 1088 = 243 712 bytes (238 KB)  [UNVERIFIED — depends on in0_block_w once D known]
# C_buf:  1 × 28 × 2048 =      57 344 bytes ( 56 KB)
# Total:                       317 440 bytes (310 KB) ← fits 1 MB ✓
```

> **Warning:** The B_buf estimate for the down projection uses `in0_block_w=4` as a placeholder. Once $D$ is confirmed, verify that the chosen `in0_block_w` (which must divide $K_t = \lceil D/32 \rceil$) keeps the total within 1 MB. Increasing `in0_block_w` increases B_buf linearly and can cause L1 overflow at large `per_core_N`.

### 5.3 Minimum Grid Check

For `MatmulMultiCoreReuseMultiCastProgramConfig`, both grid_x and grid_y must be at least 2:

```python
def check_multicast_grid(grid_x, grid_y):
    if grid_x < 2 or grid_y < 2:
        raise ValueError(
            f"MatmulMultiCoreReuseMultiCastProgramConfig requires grid_x≥2 and grid_y≥2. "
            f"Got ({grid_x}, {grid_y}). Use MatmulMultiCoreProgramConfig for 1D grids."
        )
```

This check is particularly important for the decode regime ($M_t=1$, grid_y=1) — the single-row grid is invalid for the multicast config.

---

## 6. Summary: Config Parameters by Regime

| Parameter | Decode ($C=1$, $M_t=1$) | Prefill ($C=64$, $M_t=2$) | Notes |
|-----------|------------------------|--------------------------|-------|
| Config type | `MatmulMultiCoreProgramConfig` | `MatmulMultiCoreReuseMultiCastProgramConfig` | $M_t=1$ forces single-row grid |
| `grid_y` | 1 | 2 | Must divide $M_t$ |
| `grid_x` | 8 | 8 | Must divide $N_t$ |
| `per_core_M` | 1 | 1 | $M_t / \text{grid\_y}$ |
| `per_core_N` | $N_t / 8$ [UNVERIFIED] | $N_t / 8$ [UNVERIFIED] | Depends on confirmed $D$ |
| `K_t` (gate/up) | 224 | 224 | $\lceil 7168/32 \rceil$ — confirmed |
| `in0_block_w` (gate/up) | 8 | 8 | Divides $K_t=224$; $224 \% 8 = 0$ ✓ |
| `out_subblock_h` | 1 | 1 | Forced by `per_core_M=1` |
| `out_subblock_w` | $\leq 8$, divides `per_core_N` | $\leq 8$, divides `per_core_N` | Max product constraint applies |

All entries marked [UNVERIFIED] depend on the confirmed value of $D$. [D UNVERIFIED — verify against Qwen3 Technical Report]

---

## Next Steps

Proceed to [performance_profile_batched.md](performance_profile_batched.md) to understand how the configs derived here translate to measured latency, and to learn when batched matmul is the preferred strategy versus the sparse approach covered in Chapter 4.
