# Recommended Configuration for Qwen3.5-35B on T3K

This file gives the concrete recommended configuration for running Qwen3.5-35B expert parallelism
on a T3K 8-device mesh. Each decision is stated as a rule, followed by quantitative justification
and the chapter that establishes the basis.

---

## Decision 1: Expert-to-Device Assignment

**Default:** Uniform 32-experts-per-device, assigned round-robin by expert index.

**When to apply load-aware rebalancing:** If calibration profiling (512+ samples) shows any expert
receiving more than 4× the average token rate ($f_e > 4 \times k/E = 4/32 = 12.5\%$), apply the
greedy decreasing bin-packing algorithm from Chapter 4.

**When to replicate:** Replicate the top-$r$ most popular experts onto additional devices if:

$$r_e = \max\!\left(1, \left\lceil f_e \times N \right\rceil\right) > 1 \quad \text{i.e., } f_e > 1/N = 12.5\%$$

and DRAM headroom of $\geq r_e \times \text{expert\_weight\_size}$ is available on the target device.
At $f_e = 0.17$ (Zipf example from Chapter 7), $r_e = \max(1, \lceil 0.17 \times 8 \rceil) = 2$.

**Why uniform is the default:**

- $E/N = 32$ is exact, so uniform assignment is load-balanced in expectation under the training
  distribution (Chapter 4, `uniform_partitioning.md`).
- Load imbalance at $B=32$ produces overflow rates of approximately 8.0% per expert under Poisson
  approximation ($\lambda = k \times B / E = 1.0$, see Chapter 7, `capacity_overflow_handling.md`).
  This is already accommodated by $\text{CF}=1.25$.
- The cost of load-aware rebalancing (profiling + weight migration) is only justified when imbalance
  exceeds the CF buffer. Monitor using the coefficient of variation (CV) of per-expert token rates;
  trigger rebalancing when CV > 0.5 (Chapter 7, `load_imbalance_detection.md`).

---

## Decision 2: Dispatch/Combine Scheme

**Default:** `ttnn.all_to_all` (dispatch and combine) with `num_links=1`.

**When to switch to all-gather-based sharding:** Only if $B$ is consistently below ~4 tokens per
forward pass, where dispatch volume ($\approx 3.2$ MB at $B=1$) falls below the per-link setup
overhead threshold. See Chapter 3, `scheme_comparison_matrix.md` for the crossover derivation;
confirm with Chapter 6, `end_to_end_latency_model.md`.

**When to use `num_links=2`:** At $B \geq 256$ (dispatch volume $\geq$ ~26 MB), where the second
link provides a net reduction in latency. At $B=32$ (6.4 MB), `num_links=1` is preferred:
link-setup overhead from a second link exceeds its throughput benefit at this payload size.
See Chapter 2, `dispatch_combine_overhead.md`.

**Capacity factor:** $\text{CF} = 1.25$ (default). This yields $C=2$ at $B=32$ and accommodates
approximately 8% per-expert overflow under Poisson load without token dropping in the median case.
Raise to $\text{CF} = 1.5$ if per-expert CV exceeds 1.0; lower to $\text{CF} = 1.0$ only at
$B \geq 256$ where $C=10$ already provides substantial buffer capacity.

**Drop policy:** Hard drop with renormalization. Tokens that exceed expert capacity are zeroed out
and do not contribute to the output. The routing weight of the surviving top-$k'$ experts is
renormalized to sum to the original weight sum. Expected total dropped tokens at $B=32$, CF=1.25:
approximately 27 tokens per step ($\approx 10.6\%$). See Chapter 7, `capacity_overflow_handling.md`.

---

## Decision 3: Routing Weight Processing

**Router projection:** BF16 for `W_r` (size: $7168 \times 256 \times 2 = 3.67$ MB). Do not
quantize to INT8 by default — the router projection at $B=32$ is already sub-millisecond (~117 MFLOPs)
and is not a bottleneck. Use INT8 only if DRAM headroom is critically constrained, accepting a
potential accuracy degradation on low-probability expert selections.

**Top-k selection:** Fused with router projection where the TTNN kernel supports it; otherwise,
run as a separate `ttnn.topk` on the $[B, 256]$ logit tensor. Partial sort (O($E + k \log k$)) is
preferred over full sort (O($E \log E$)); verify TTNN kernel selection.
See Chapter 5, `topk_selection_efficiency.md`.

**Weight normalization:** Defer to the combine step. Do not renormalize the $k$ selected routing
weights before dispatch. Instead, carry the raw top-k weights as metadata through the all-to-all
and apply renormalization fused with weighted accumulation in the combine kernel.

- Justification: avoids an extra pass over the $[B, k]$ weight tensor between dispatch and FFN;
  the combine kernel already reads each weight exactly once during accumulation.
- Precision: BF16 accumulation of $k=8$ expert outputs introduces a bound of $k \times 2^{-7}
  \approx 6.25\%$ relative error. This is inherent to BF16 and not worsened by deferring normalization.

See Chapter 5, `weight_normalization.md` and Chapter 6, `combine_accumulation.md`.

---

## Decision 4: Expert FFN Computation

**Kernel:** `sparse_matmul` (zero-tile-row skipping) for all decode batch sizes ($B \leq 32$).
Token utilization is at most $C/32 = 2/32 = 6.25\%$ at $B=32$; dense matmul wastes 93.75% of
MAC operations on zero-padded rows. See Chapter 6, `expert_ffn_tiling.md`.

**Core assignment:** 2 Tensix cores per local expert, yielding 64 active cores (80 total, 16 idle).
`per_core_M = ceil(C/32) = 1` for all decode batch sizes. See Chapter 6, `expert_ffn_tiling.md`
and `program_configs_t3k.md` (T3K MoE optimization guide, ch07).

**Weight placement:** All expert weights in `DRAM_MEMORY_CONFIG`. Expert weights stream from DRAM
during each FFN matmul. This is expected and not a spill indicator; the DRAM read bandwidth during
FFN matmul reflects intentional weight streaming, not L1 pressure.

**Prefill:** For prefill ($S \geq 512$), switch to dense batched matmul and `num_links=2`.
See Chapter 6, `expert_ffn_tiling.md` for the utilization crossover threshold.

---

## Decision 5: Fused Dispatch-Compute-Combine Pipeline

**Double-buffering:** Enable when $B \geq 4$. The double-buffer memory footprint at $B=32$ is
approximately 7.0 MB ($8 \times C \times E_d \times H \times 2 = 8 \times 2 \times 32 \times 7168 \times 2$).
This fits comfortably within the 120 MB aggregate L1 budget.

**Buffer layout:** 4 buffers per pipeline stage × 2 stages:

1. `dispatch_send` — tokens packed for transmission to remote devices
2. `dispatch_recv` — received tokens from remote devices before expert FFN
3. `combine_send` — expert outputs packed for return transmission
4. `combine_recv` — received expert outputs before weighted accumulation

See Chapter 6, `pipeline_design.md`.

**When to disable:** Do not double-buffer at $B=1$ ($C=1$, buffer footprint ~3.5 MB total — still
small, but the pipeline depth provides no latency benefit when the payload is a single capacity
slot per expert). The A2A at $B=1$ (~3.2 MB, ~0.26 ms) is already short enough that pipeline
overlap adds more orchestration overhead than it saves.

---

## Decision 6: Load Balancing

**Per-expert score biases:** Enable for production inference. Calibrate on 512 representative input
samples; compute the per-expert mean activation frequency $\hat{f}_e$; set bias $b_e$ such that
adjusted logits $g_e' = g_e + b_e$ equalize utilization. Update biases every ~10,000 inference
steps or when CV of measured $f_e$ exceeds 0.5.

**Sliding window monitoring:** Track per-expert token counts with a window of 64 decode steps.
Compute CV across experts at the end of each window. Overhead: O($k \times B = 256$) counter
updates per step — negligible. See Chapter 7, `load_imbalance_detection.md`.

**Expert replication trigger:** If any expert's $f_e > 12.5\%$ persists for $> 1{,}000$ consecutive
steps and DRAM headroom allows, replicate that expert on one additional device and update the
dispatch routing table accordingly. See Chapter 4, `expert_replication.md`.

**Do not use training-time auxiliary loss at inference.** The auxiliary load-balancing loss (e.g.,
$L_{\text{aux}}$) is stripped from the inference graph. Inference-time load balancing is handled
entirely through score biases and optional replication, not through loss gradients.
See Chapter 7, `dynamic_routing_strategies.md`.

---

## Configuration Summary Table

| Decision | Default Value | Trigger for Change |
|---|---|---|
| Experts per device | 32 (uniform) | CV > 0.5 → rebalance |
| Expert replication | None | $f_e > 12.5\%$ sustained |
| Dispatch scheme | `ttnn.all_to_all` | $B < 4$ → consider all-gather |
| `num_links` | 1 | $B \geq 256$ → use 2 |
| Capacity factor CF | 1.25 | CV > 1.0 → raise to 1.5 |
| Drop policy | Hard drop + renorm | — |
| Router weights | BF16 (3.67 MB) | DRAM-constrained → INT8 (1.84 MB) |
| Top-k | Partial sort; fused if available | — |
| Weight normalization | Deferred to combine | — |
| Expert FFN kernel | `sparse_matmul` | $B \geq 256$ → dense matmul |
| Core assignment | 64 active (2/expert) | — |
| Double-buffering | Enabled ($B \geq 4$) | $B = 1$ → disable |
| Score biases | Calibrated on 512 samples | Recalibrate every ~10k steps |
| Monitoring window | 64 steps | — |

---

## References

- Chapter 2, `dispatch_combine_overhead.md` — num_links selection; communication-bound regime
- Chapter 3, `scheme_comparison_matrix.md` — all-to-all vs. all-gather crossover
- Chapter 4, `uniform_partitioning.md` — baseline EP assignment
- Chapter 4, `expert_replication.md` — replication factor formula; dispatch integration
- Chapter 5, `router_forward_pass.md` — W_r sizing; router FLOP count
- Chapter 5, `topk_selection_efficiency.md` — partial sort complexity
- Chapter 5, `weight_normalization.md` — deferred normalization rationale
- Chapter 6, `pipeline_design.md` — double-buffer layout; 4 buffers per stage
- Chapter 6, `expert_ffn_tiling.md` — sparse_matmul applicability; core assignment
- Chapter 6, `combine_accumulation.md` — BF16 accumulation bound
- Chapter 7, `capacity_overflow_handling.md` — Poisson overflow model; drop rate at B=32
- Chapter 7, `load_imbalance_detection.md` — CV threshold; monitoring overhead
- Chapter 7, `dynamic_routing_strategies.md` — score biases; inference-time balancing
