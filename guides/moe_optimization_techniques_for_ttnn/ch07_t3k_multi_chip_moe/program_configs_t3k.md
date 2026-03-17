# Program Configs for T3K MoE

## Section 1: How Per-Chip Program Configs Change Under EP

On a single chip, the batched matmul program config is built for the full expert tensor $[E, C, H]$. Under EP with $N = 8$ devices, each chip's program config covers only its local expert subset $[E_d, C, H] = [32, C, H]$.

The config parameters that change are those that depend on the activation tensor height — specifically `per_core_M`.

### `per_core_M` Derivation

`per_core_M` specifies the number of tile rows each core processes. The total tile rows across all local experts are:

$$\text{total\_tiles\_M} = E_d \times \left\lceil \frac{C}{32} \right\rceil = E_d \times M_t$$

With 80 Tensix cores per chip and 32 local experts, a natural core grouping is:

$$\text{core groups} = \frac{\text{cores}}{E_d} = \frac{80}{32} = 2.5 \quad \Rightarrow \quad \text{use 2 core groups (64 cores active)}$$

Assigning 2 cores per expert (a core group handles one expert's matmul):

$$\text{per\_core\_M} = \frac{E_d \times M_t}{\text{num\_core\_groups}} = \frac{32 \times M_t}{32} = M_t$$

So `per_core_M` equals $M_t = \lceil C / 32 \rceil$.

| Batch $B$ | Capacity $C$ | $M_t = \lceil C/32 \rceil$ | `per_core_M` |
|---|---|---|---|
| $B = 1$ | $C = 1$ | $1$ | $1$ |
| $B = 32$ | $C = 2$ | $1$ | $1$ |
| $B = 256$ | $C = 10$ | $1$ | $1$ |
| $B = 512$ | $C = 20$ | $1$ | $1$ |
| $B = 1024$ | $C = 40$ | $2$ | $2$ |

For the Qwen3.5-35B decode regime ($B \leq 256$), `per_core_M = 1` universally. This is a direct consequence of $C < 32$ (one tile row covers all capacity slots per expert).

### `per_core_N` Derivation

`per_core_N` depends on the output dimension $D$ [UNVERIFIED] and the grid column count:

$$\text{per\_core\_N} = \left\lceil \frac{D / 32}{\text{grid\_cols}} \right\rceil = \left\lceil \frac{N_t}{\text{grid\_cols}} \right\rceil$$

where $N_t = \lceil D / 32 \rceil$ is the total tile columns. With `grid_cols = 2` (2 cores per expert group arranged in a 1×2 or similar sub-grid):

$$\text{per\_core\_N} = \left\lceil \frac{N_t}{2} \right\rceil \quad \text{[UNVERIFIED: depends on actual D]}$$

### `out_subblock_h` and `out_subblock_w`

These must divide `per_core_M` and `per_core_N` respectively (see Chapter 3, [program_configs_batched.md](../ch03_batched_matmul_for_moe/program_configs_batched.md)):

- `out_subblock_h`: must divide `per_core_M`; at `per_core_M = 1`, only `out_subblock_h = 1` is valid
- `out_subblock_w`: must divide `per_core_N`; choose the largest divisor that fits within L1 subblock budget

```python
import ttnn

def make_t3k_moe_program_config(
    E_d: int,
    C: int,
    H: int,
    D: int,        # FFN intermediate dimension [UNVERIFIED]
    num_cores: int = 80,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """
    Build per-chip program config for T3K MoE expert matmul.
    Covers local expert batch [E_d, C, H] x [E_d, H, D] -> [E_d, C, D].
    """
    tile_size = 32
    M_t = (C + tile_size - 1) // tile_size         # ceil(C / 32)
    K_t = (H + tile_size - 1) // tile_size         # ceil(H / 32) = 224 for H=7168
    N_t = (D + tile_size - 1) // tile_size         # ceil(D / 32) [UNVERIFIED]

    # 2 cores per expert: 32 experts × 2 cores = 64 active cores; 16 idle
    cores_per_expert = 2
    num_expert_groups = E_d                        # one group per expert
    grid_rows = num_expert_groups                  # 32 rows
    grid_cols = cores_per_expert                   # 2 cols

    per_core_M = M_t                               # = 1 at B <= 256
    per_core_N = (N_t + grid_cols - 1) // grid_cols

    # out_subblock constraints: must divide per_core_M and per_core_N
    out_subblock_h = 1                             # per_core_M = 1 at decode
    # Choose largest out_subblock_w that divides per_core_N and fits L1
    out_subblock_w = 1
    for w in range(per_core_N, 0, -1):
        if per_core_N % w == 0 and out_subblock_h * w <= 8:  # L1 subblock limit
            out_subblock_w = w
            break

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_cols, grid_rows),
        in0_block_w=K_t,          # full K in one block (weights fit in L1 for small K_t)
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )
```

> **Warning:** `in0_block_w=K_t` (loading all 224 K-tiles in one block) requires significant L1. At $H = 7168$, $K_t = 224$ tiles. Verify L1 budget before setting `in0_block_w` to the full K extent; you may need to split into multiple blocks.

---

## Section 2: Grid Utilization Per Chip

With 32 local experts and 80 Tensix cores, the natural assignment leaves 16 cores idle:

$$\text{active cores} = E_d \times \text{cores\_per\_expert} = 32 \times 2 = 64$$
$$\text{idle cores} = 80 - 64 = 16 \quad (20\% \text{ wasted})$$

The utilization within each active core is further reduced by the small token count per expert:

$$\text{token utilization} = \frac{C}{\text{tile\_size}} = \frac{2}{32} = 6.25\% \quad (B = 32)$$

Only 2 of the 32 token slots in each tile row contain real data; the remaining 30 are zero-padded. A dense matmul wastes cycles on those zeros. This is why `sparse_matmul` is critical on T3K:

- **Dense matmul**: processes all 32 × 224 = 7,168 tile multiply-accumulate operations per expert, including 30/32 = 93.75% zero rows
- **`sparse_matmul`**: skips zero tile rows, reducing to $C \times K_t = 2 \times 224 = 448$ effective MAC operations per expert (for non-zero experts)

The savings from `sparse_matmul` at decode are:

$$\text{skip fraction} = 1 - \frac{C}{32} = 1 - \frac{2}{32} = 93.75\%$$

Combined with the expert-level sparsity (at $B = 1$, ~31 of 32 local experts are idle), the total skip fraction approaches $(1 - 1/32) \times (30/32) \approx 90\%$ at $B = 1$.

### Grid Utilization Summary

| $B$ | $C$ | Token utilization | Expert utilization | Combined utilization |
|---|---|---|---|---|
| $1$ | $1$ | $1/32 = 3.1\%$ | $1/32 = 3.1\%$ | $\approx 0.1\%$ |
| $32$ | $2$ | $2/32 = 6.25\%$ | $\sim 100\%$ (all experts get tokens) | $6.25\%$ |
| $256$ | $10$ | $10/32 = 31.25\%$ | $\sim 100\%$ | $31.25\%$ |
| $1024$ | $40$ | $40/64 = 62.5\%$ | $\sim 100\%$ | $62.5\%$ |

> **Tip:** At $B = 256$, $C = 10$, the token utilization reaches ~31% — still sub-optimal but substantially better than decode. Prefill workloads benefit more from dense batched matmul (Chapter 3) once $C$ grows past 16 (50% utilization threshold).

---

## Section 3: Worked Example — Qwen3.5-35B at $B = 32$, $S = 1$ (Decode)

**Given:**
- $E = 256$, $k = 8$, $E_d = 32$, $H = 7168$, $N = 8$
- $B = 32$ tokens per decode step
- Capacity factor $\text{CF} = 1.25$

**Step 1: Compute expert capacity $C$**

$$C = \left\lceil \frac{k \times B \times \text{CF}}{E} \right\rceil = \left\lceil \frac{8 \times 32 \times 1.25}{256} \right\rceil = \left\lceil \frac{320}{256} \right\rceil = \left\lceil 1.25 \right\rceil = 2$$

**Step 2: Local activation shape**

$$[E_d \times C, H] = [32 \times 2, 7168] = [64, 7168]$$

(Before reshaping to $[32, 2, 7168]$ for the batched matmul.)

**Step 3: Tile counts**

The per-expert tile count $M_t = \lceil C / 32 \rceil = \lceil 2 / 32 \rceil = 1$ (one tile row covers all $C = 2$ capacity slots per expert). The flattened activation $[E_d \times C, H] = [64, H]$ has $\lceil 64/32 \rceil = 2$ tile rows total across all experts, but after the per-expert reshape to $[32, 2, 7168]$, each expert contributes $M_t = 1$ tile row to its assigned core group.

$$M_t = \left\lceil \frac{C}{32} \right\rceil = \left\lceil \frac{2}{32} \right\rceil = 1 \quad \text{(per-expert tile rows)}$$
$$K_t = \left\lceil \frac{7168}{32} \right\rceil = 224 \quad \text{(input tile columns)}$$
$$N_t = \left\lceil \frac{D}{32} \right\rceil \quad \text{[UNVERIFIED]}$$

**Step 4: Per-core tile assignment**

With 32 core groups (one per expert) and 2 cores per group (64 total active cores):

$$\text{per\_core\_M} = \frac{E_d \times M_t}{\text{num\_core\_groups}} = \frac{32 \times 1}{32} = 1$$

**Step 5: Grid configuration**

$$\text{grid} = (\text{grid\_cols}, \text{grid\_rows}) = (2, 32) \quad \Rightarrow \quad 64 \text{ active cores, 16 idle}$$

**Step 6: Output subblock**

$$\text{out\_subblock\_h} = 1 \quad (\text{only valid divisor of } \text{per\_core\_M} = 1)$$
$$\text{out\_subblock\_w} = \text{largest divisor of } \text{per\_core\_N} \leq 8 \quad \text{[UNVERIFIED: depends on D]}$$

**Step 7: Core utilization**

- Token utilization per core: $C / 32 = 2 / 32 = 6.25\%$
- 16 of 80 cores are idle (no expert assigned)
- `sparse_matmul` skips 30/32 zero tile rows → effective compute reduced by $\sim 94\%$ vs. dense

**Summary table:**

| Parameter | Value |
|---|---|
| $C$ | 2 |
| Local activation | $[64, 7168]$ → reshape $[32, 2, 7168]$ |
| $M_t$ (per expert) | 1 |
| $K_t$ | 224 |
| `per_core_M` | 1 |
| Grid | $(2, 32)$ = 64 active cores |
| `out_subblock_h` | 1 |
| Token utilization | 6.25% |
| Recommended op | `sparse_matmul` |

---

## Section 4: Worked Example — Mixtral 8x7B on T3K

This example illustrates a degenerate EP case and motivates why Qwen3.5-35B has better T3K utilization than Mixtral 8x7B under pure EP.

**Given:**
- $E = 8$, $k = 2$, $H = 4096$, $D = 14336$, $N = 8$ devices
- EP degree = 8 → $E_d = 8 / 8 = 1$ expert per device

**Per-device local batch at $B = 32$:**

$$C = \left\lceil \frac{k \times B \times \text{CF}}{E} \right\rceil = \left\lceil \frac{2 \times 32 \times 1.25}{8} \right\rceil = \left\lceil 10 \right\rceil = 10$$

Each device holds 1 expert and processes up to 10 tokens per decode step.

$$\text{Local activation shape} = [E_d \times C, H] = [1 \times 10, 4096] = [10, 4096]$$

$$M_t = \left\lceil \frac{10}{32} \right\rceil = 1, \quad K_t = \left\lceil \frac{4096}{32} \right\rceil = 128, \quad N_t = \left\lceil \frac{14336}{32} \right\rceil = 448$$

With only 1 expert per device, the entire 80-core grid handles a single $[10, 4096] \times [4096, 14336]$ matmul. `per_core_M = 1` and token utilization is $10/32 = 31.25\%$ — better than Qwen3.5-35B decode, but the workload is trivially small.

**Why pure EP is suboptimal for Mixtral:**

A single $[10, 4096] \times [4096, 14336]$ matmul is a small kernel that cannot saturate 80 cores. The effective FLOPs are:

$$2 \times 10 \times 4096 \times 14336 = 1.17 \times 10^9 \text{ FLOPs}$$

Spread across 80 cores this is $\sim 14.6$ MFLOPs per core — far below peak Tensix throughput. Mixtral on T3K requires **tensor parallelism within each expert** (splitting the $D = 14336$ FFN dimension across cores) to achieve reasonable arithmetic intensity.

**Contrast with Qwen3.5-35B:**

Qwen3.5-35B has 32 experts per device. Even with $C = 2$ (low token utilization), the matmul kernel covers $32 \times 2 = 64$ activation rows, giving the scheduler more tile rows to pipeline across cores. The 64 active cores each handle 1 tile row, and `sparse_matmul` efficiently skips zero experts. The model's higher $E_d$ is what makes T3K EP tractable for Qwen3.5-35B.

| Comparison | Qwen3.5-35B ($B=32$) | Mixtral 8x7B ($B=32$) |
|---|---|---|
| $E_d$ | 32 | 1 |
| $C$ | 2 | 10 |
| Local activation | $[64, 7168]$ | $[10, 4096]$ |
| Active cores | 64 | 80 (but tiny workload) |
| Token utilization | 6.25% | 31.25% |
| Bottleneck | Token utilization | Kernel size; needs TP within expert |
| Recommended strategy | EP + `sparse_matmul` | EP + TP within expert |

> **Warning:** Do not use pure EP on T3K for Mixtral 8x7B. With $E_d = 1$ and $C = 10$, each device is computing one tiny matmul per MoE layer. Tensor parallelism within each expert is required to utilize the 80-core Tensix array effectively.

---

## References

- Chapter 3: [program_configs_batched.md](../ch03_batched_matmul_for_moe/program_configs_batched.md) — `per_core_M`, `out_subblock_h` constraints
- Chapter 4: [when_sparse_matmul_wins.md](../ch04_sparse_matmul_for_moe/when_sparse_matmul_wins.md) — utilization threshold for `sparse_matmul`
- Chapter 6: [decision_guide.md](../ch06_comparative_analysis/decision_guide.md) — hybrid strategy selection
- [expert_parallelism_on_t3k.md](./expert_parallelism_on_t3k.md) — EP partitioning and all-to-all communication
- [sharding_strategies.md](./sharding_strategies.md) — activation and weight tensor placement

---

**Next:** [Chapter 8 — E2E Workflow and Troubleshooting](../ch08_e2e_workflow_and_troubleshooting/index.md)
