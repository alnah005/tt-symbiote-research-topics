# When sparse_matmul Wins

This file characterizes the conditions under which `sparse_matmul` outperforms batched matmul for MoE expert FFN computation. It derives the sparsity ratio $\rho$ for the decode and prefill regimes of Qwen3.5-35B on T3K, identifies the empirical crossover threshold $\rho^*$, and explains when sparse_matmul should not be used.

Read `sparse_matmul_internals.md` before this file; the tile-skip mechanism and sparsity ratio definition are assumed known.

---

## 1. The Crossover: When Does Skipping Tiles Pay Off?

Every tile that sparse_matmul skips saves:

- $32 \times 32 \times 32 = 32{,}768$ FMAs (per output tile per K step skipped).
- 2048 bytes of BF16 activation DRAM read (per activation tile skipped).
- 1088 bytes of BFP8 weight DRAM read (per corresponding weight tile skipped, once per N-column).

Every tile that sparse_matmul checks costs:

- One mask read (one byte from the mask tensor) plus a conditional branch — a small but non-zero overhead per tile, paid even for active tiles.
- Additional kernel launch setup relative to dense matmul (the mask tensor must be passed as an extra argument and its DRAM address registered in the kernel descriptor).

The crossover occurs when the savings from skipped tiles equal the overhead of the mask check on active tiles. Empirically, this break-even point is approximately:

$$\rho^* \approx 0.7$$

When $\rho < 0.7$ (more than 30% of tiles are inactive), `sparse_matmul` is faster. When $\rho > 0.7$, the overhead of checking nearly all tiles dominates the savings from the few skipped tiles, and batched matmul is faster.

> **Caveat:** $\rho^* \approx 0.7$ is an empirical approximation on Wormhole B0. The exact value depends on the matmul shape (H, D [D UNVERIFIED]), the DRAM bandwidth state (shared with other ops on the same device), and kernel implementation details. Treat this as an operational heuristic rather than a precisely measured constant. Profile both kernels at $\rho$ values near 0.7 when making deployment decisions.

---

## 2. Two-Regime Analysis

### 2.1 Regime 1 — Decode ($B \times S$ Small, $C = 1$)

In the decode regime, each forward pass processes one new token per sequence. With $B$ sequences each generating one token:

$$C = \left\lceil \frac{k \times B \times S}{E} \right\rceil = \left\lceil \frac{8 \times B \times 1}{256} \right\rceil = \left\lceil \frac{B}{32} \right\rceil$$

For $B \leq 32$: $C = 1$.

On a T3K device with $E_d = 32$ local experts, the number of active expert slots (experts that received at least one token) after all-to-all dispatch is at most $k \times B$ for $B \leq 32$ — bounded by the total number of (token, expert) assignments. The sparsity ratio is:

$$\rho = \frac{\text{active expert slots}}{E_d} = \frac{k \times B / N}{E_d} = \frac{8 \times B / 8}{32} = \frac{B}{32}$$

(Here $N = 8$ devices; each device receives $k \times B / N$ assignments on average under uniform routing.)

**Worked example — $B = 1$ (single-token decode):**

$$\rho = \frac{1}{32} \approx 0.031$$

Only 1 of 32 local expert slots is active on average. `sparse_matmul` skips 97% of tile rows. This is the maximum-benefit regime.

**Worked example — $B = 8$ (8-sequence decode):**

$$\rho = \frac{8}{32} = 0.25$$

8 of 32 expert slots active. `sparse_matmul` skips 75% of tiles. Well below $\rho^*$; sparse_matmul strongly preferred.

**Worked example — $B = 32$ (32-sequence decode):**

$$\rho = \frac{32}{32} = 1.0$$

All 32 expert slots active on average. $C = 1$ still, so $M_t = 1$, but every expert has a real token. $\rho = 1.0$: no sparsity. Sparse_matmul provides no benefit; use batched matmul or the `MatmulMultiCoreProgramConfig` decode config from Chapter 3.

> **Note:** The $B=32$ case reaching $\rho = 1$ is specific to the Qwen3.5-35B routing configuration ($k=8$, $E=256$, $N=8$). With $k \times B / N = 8 \times 32 / 8 = 32 = E_d$, every local expert slot receives exactly one token on average. Under non-uniform routing, some experts receive 0 tokens and others receive 2+, so actual $\rho$ may be below 1.0 even at $B=32$.

### 2.2 Regime 2 — Prefill ($B \times S$ Large, $C$ Large)

In the prefill regime, each sequence contributes many tokens. For $B=1$, $S=2048$:

$$C = \left\lceil \frac{8 \times 1 \times 2048}{256} \right\rceil = 64, \qquad M_t = \left\lceil \frac{64}{32} \right\rceil = 2$$

With $C = 64$ and $E_d = 32$ local experts, each local expert receives on average $k \times B \times S / E = 8 \times 1 \times 2048 / 256 = 64$ tokens — exactly filling its capacity. All 32 expert slots are active and all capacity slots are filled:

$$\rho \approx 1.0$$

Sparse_matmul overhead dominates; use batched matmul with `MatmulMultiCoreReuseMultiCastProgramConfig` (see Chapter 3, `program_configs_batched.md` §4.2).

For $B=4$, $S=2048$:

$$C = \left\lceil \frac{8 \times 4 \times 2048}{256} \right\rceil = 256, \qquad M_t = 8, \qquad \rho \approx 1.0$$

The fill rate remains near 1.0 for all prefill scenarios with $B \times S \geq E_d \times N / k = 32 \times 8 / 8 = 32$.

---

## 3. Sparsity Ratio Formula

The general sparsity ratio for expert-parallel MoE on T3K is:

$$\rho = \frac{\text{active expert slots} \times M_t}{E_d \times M_t} = \frac{\text{active expert slots}}{E_d}$$

where $M_t = \lceil C / 32 \rceil$. The $M_t$ terms cancel: each active expert occupies $M_t$ tile rows in the numerator, and the denominator $E_d \times M_t$ is the total tile rows. The formula therefore simplifies to active experts divided by total experts, independent of $M_t$.

> **Note:** This cancellation holds when each active expert fills all of its $M_t$ tile rows (i.e., capacity is fully used). In the decode regime ($M_t = 1$, $C \leq 32$), there is only one tile row per expert and the formula is trivially correct. In the prefill regime ($M_t > 1$, high fill rate), every active expert fills all $M_t$ rows, so the $M_t$ still cancels and the formula $\rho = \text{active\_experts} / E_d$ remains correct. This is why the §2.1 and §2.2 results agree with the formula below regardless of $M_t$.

Under uniform routing:

$$\text{active experts} \approx \min\!\left(E_d,\ \frac{k \times B \times S}{N}\right)$$

So:

$$\rho \approx \frac{\min\!\left(E_d,\ \frac{k \times B \times S}{N}\right)}{E_d} = \frac{\min\!\left(32,\ B \times S\right)}{32}$$

(Substituting $k=8$, $N=8$, $E_d=32$; note $k/N = 8/8 = 1$.)

**Verification for decode ($B=1$, $S=1$):**

$$\rho = \frac{\min(32,\ 1)}{32} = \frac{1}{32} \approx 0.031$$

This matches $\rho \approx 0.031$ computed in §2.1: on average 1 of the 32 local expert slots is active. Sparse_matmul is overwhelmingly preferred at $B=1$.

**Summary table — Qwen3.5-35B on T3K:**

| Regime | $B$ | $S$ | $C$ | $M_t$ | $\rho$ (approx.) | Use |
|--------|-----|-----|-----|--------|-------------------|-----|
| Single decode | 1 | 1 | 1 | 1 | ~0.031 | sparse_matmul |
| Small-batch decode | 8 | 1 | 1 | 1 | ~0.25 | sparse_matmul |
| Medium decode | 16 | 1 | 1 | 1 | ~0.5 | sparse_matmul |
| Decode at crossover | 22 | 1 | 1 | 1 | ~0.7 = $\rho^*$ | Either |
| Full decode | 32 | 1 | 1 | 1 | ~1.0 | batched matmul |
| Short prefill | 1 | 128 | 4 | 1 | ~1.0 | batched matmul |
| Medium prefill | 1 | 512 | 16 | 1 | ~1.0 | batched matmul |
| Long prefill | 1 | 2048 | 64 | 2 | ~1.0 | batched matmul |
| Large prefill | 4 | 2048 | 256 | 8 | ~1.0 | batched matmul |

---

## 4. Effect of H and D on the Crossover

[D UNVERIFIED — all claims in this section that depend on $D$ are unverified. Verify $D$ against the Qwen3 Technical Report before relying on crossover estimates.]

The theoretical crossover $\rho^*$ satisfies:

$$\rho^* \approx 1 - \frac{t_{\text{mask}}}{t_{\text{tile}}}$$

where $t_{\text{mask}}$ is the time cost of checking one mask entry (mask read + branch, approximately constant in hardware cycles), and $t_{\text{tile}}$ is the time to process one active tile (FMA block + two DRAM loads). When $t_{\text{tile}}$ is large relative to $t_{\text{mask}}$, the overhead fraction is small and $\rho^*$ is pushed toward 1 — sparse_matmul wins over a wider range of sparsity values.

$t_{\text{tile}}$ scales with the FMA work per tile:

$$t_{\text{tile}} \propto 32 \times 32 \times 32 = 32{,}768 \text{ FMAs per tile pair} + \text{DRAM time for 2048 + 1088 bytes}$$

This is independent of $H$ and $D$ at the individual tile level — a single tile pair is always $32 \times 32 \times 32$. However, the total work per K reduction (summed over $K_t = \lceil H/32 \rceil = 224$ K steps) is:

$$\text{total FMAs per output row} = M_t \times N_t \times K_t \times 32{,}768$$

Larger $H$ (more K-tiles) means more tiles to skip per inactive row, increasing the absolute savings from sparsity. Larger $D$ [D UNVERIFIED] means more N-tile columns per row, also increasing per-row savings.

**Qualitative conclusion:** Larger $H$ and larger $D$ [D UNVERIFIED] shift $\rho^*$ slightly toward 1 (sparse_matmul competitive up to higher fill rates). For Qwen3.5-35B with $H=7168$ ($K_t=224$), the 0.7 empirical threshold is reasonable. For a model with smaller $H$ (e.g., $H=2048$, $K_t=64$), the overhead fraction increases and $\rho^*$ moves toward 0.5–0.6. Mark these estimates [D UNVERIFIED] until $D$ is confirmed.

---

## 5. Sequence Length Effect

Longer sequences increase $C$, which increases $M_t$, which increases the total number of activation tile rows. With more tile rows, each of the $E_d = 32$ expert slots occupies more rows. The sparsity ratio evolves as:

- At short $S$ (decode, $C=1$, $M_t=1$): one tile row per expert slot. An inactive expert wastes one tile row.
- At moderate $S$ ($C=32$, $M_t=1$): still one tile row — same waste as $C=1$ in tile terms.
- At long $S$ ($C=64$, $M_t=2$): two tile rows per expert slot. As $S$ grows, each active expert contributes more rows, pushing $\rho$ toward 1.

The key transition occurs when $C$ crosses 32 (one full tile):

$$M_t > 1 \iff C > 32 \iff \frac{k \times B \times S}{E} > 32 \iff B \times S > \frac{32 \times E}{k} = \frac{32 \times 256}{8} = 1024$$

For $B \times S > 1024$ tokens in a batch, $M_t \geq 2$ and every local expert slot is likely filling up (approaching $\rho \to 1$). Below this threshold ($B \times S \leq 1024$, $M_t = 1$), there is structural sparsity to exploit. The maximum sparsity benefit is at the smallest $B \times S$.

---

## 6. When NOT to Use sparse_matmul

Avoid `sparse_matmul` in the following conditions:

| Condition | Reason | Alternative |
|-----------|--------|-------------|
| Prefill at full capacity ($\rho > 0.7$, $C \geq 32$) | Mask overhead costs more than skipping the few inactive tiles saves. | Batched matmul (`MatmulMultiCoreReuseMultiCastProgramConfig`). |
| $B = 32$, $S = 1$ ($\rho \approx 1.0$) | All local experts active; no tiles to skip. | Batched matmul decode config (Ch. 3, `program_configs_batched.md` §4.1). |
| Irregular or unpredictable sparsity patterns | Random expert activation patterns make mask construction overhead significant relative to the tile skip savings. | Batched matmul or per-expert matmul with shape dispatch. |
| Dynamic $C$ (varying $B$ or $S$) without a fixed canonical shape | Mask shape changes with $C$, forcing recompilation on each new shape. | Maintain a small set of pre-compiled $(B, S, C)$ canonical shapes and pad to the nearest one. |
| Single expert, single token ($E_d = 1$) | The entire computation is one tile pair; mask overhead is 100% of cost. | Direct `ttnn.matmul` call on the single active token. |

---

## Summary

The per-regime $\rho$ values and kernel recommendations are consolidated in the §3 table above.

The practical hybrid strategy: use `sparse_matmul` for decode ($B \leq 16$) and batched matmul for prefill ($S \geq 512$ or $B \geq 32$). This split is formalized as a deployment pattern in Chapter 6 (`decision_guide.md`).

---

## References

- Chapter 4, `sparse_matmul_internals.md` — Tile-skip mechanism, $\rho$ definition, FLOP cost model.
- Chapter 4, `program_configs_sparse.md` — Config selection for the decode regimes identified here.
- Chapter 3, `performance_profile_batched.md` §2.2 — Tile-level FLOP efficiency at decode; motivation for this analysis.
- Chapter 3, `formulating_batched_matmul.md` §2.2 — Expert capacity formula and FLOP efficiency.
- Chapter 1, `routing_and_sparsity.md` — Top-K routing and expert activation patterns.

---

**Next:** [program_configs_sparse.md](./program_configs_sparse.md)
