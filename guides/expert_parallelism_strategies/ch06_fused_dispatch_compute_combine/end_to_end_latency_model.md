# End-to-End Latency Model

## Section 1: Parameterized Latency Model

The total time to execute a single MoE layer on one device of the T3K cluster is:

$$T_{\text{total}} = T_{\text{route}} + T_{\text{dispatch}} + T_{\text{expert ffn}} + T_{\text{combine}} + T_{\text{accum}}$$

Each term is defined below using Qwen3.5-35B constants:
$E = 256$, $k = 8$, $N = 8$, $E_d = 32$, $H = 7168$,
link bandwidth $\text{BW} = 12.5$ GB/s per Ethernet link,
$CF = 1.25$, $C(B) = \lceil B \times 1.25 / 32 \rceil$.

### $T_{\text{route}}$: Router Projection and Top-$k$

The router computes a linear projection $[B, H] \times [H, E]$ followed by sigmoid
and top-$k$ selection:

$$T_{\text{route}} = \frac{2 \times B \times H \times E}{\text{FLOP rate}} + T_{\text{topk}}$$

$$= \frac{2 \times B \times 7168 \times 256}{\text{FLOP rate}} + T_{\text{topk}}$$

At $B = 32$: $2 \times 32 \times 7168 \times 256 = 117{,}440{,}512 \approx 117$ MFLOPs.
The top-$k$ selection over $E = 256$ logits is $O(B \times E)$ and typically negligible
relative to the matmul.

### $T_{\text{dispatch}}$: All-to-All Dispatch

The dispatch volume per device is the total bytes sent to all $N-1$ peer devices:

$$V_{\text{dispatch}} = (N-1) \times C \times E_d \times H \times 2 \text{ bytes}$$

$$T_{\text{dispatch}} = \frac{V_{\text{dispatch}}}{\text{BW}}$$

At $B = 32$, $C = 2$, $N = 8$:

$$V_{\text{dispatch}} = 7 \times 2 \times 32 \times 7168 \times 2 = 6{,}422{,}528 \text{ bytes} \approx 6.4 \text{ MB}$$

$$T_{\text{dispatch}} = \frac{6.4 \text{ MB}}{12.5 \text{ GB/s}} \approx 0.51 \text{ ms}$$

At $B = 1$, $C = 1$:

$$V_{\text{dispatch}} = 7 \times 1 \times 32 \times 7168 \times 2 = 3{,}211{,}264 \text{ bytes} \approx 3.2 \text{ MB}$$

$$T_{\text{dispatch}} = \frac{3.2 \text{ MB}}{12.5 \text{ GB/s}} \approx 0.26 \text{ ms}$$

> **Note:** On the T3K 1×8 linear mesh with `ttnn.Topology.Linear`, `cluster_axis=1`,
> tokens traverse multiple hops to reach non-adjacent devices. The formula above uses
> a single-link model; actual latency may be slightly higher due to hop contention.
> Calibrate against measured T_dispatch from the TTNN profiler (see Section 4).

### $T_{\text{expert ffn}}$: Expert FFN Compute

Each device processes approximately $T_d \approx B$ tokens across its $E_d = 32$
experts. Each token requires two matmuls:
up/gate projection $[T_d, H] \times [H, D]$ and down projection $[T_d, D] \times [D, H]$.

$$T_{\text{expert ffn}} = \frac{2 \times T_d \times H \times D \times 2}{\text{FLOP rate}}$$

The factor of 2 in the exponent counts the two matmul pairs (up+gate and down);
the factor of 2 in the numerator counts multiply-add as 2 FLOPs.

$$= \frac{4 \times B \times H \times D}{\text{FLOP rate}}$$

With $D$ [UNVERIFIED exact value for Qwen3.5-35B FFN intermediate dimension]:
if $D \approx 2H \approx 14336$ [UNVERIFIED]:

$$T_{\text{expert ffn}} \approx \frac{4 \times B \times 7168 \times 14336}{\text{FLOP rate}} = \frac{B \times 410{,}894{,}336}{\text{FLOP rate}}$$

### $T_{\text{combine}}$: All-to-All Combine

The combine all-to-all is symmetric to dispatch in volume:

$$T_{\text{combine}} \approx T_{\text{dispatch}}$$

### $T_{\text{accum}}$: Weighted Scatter-Add

The accumulation performs $B \times k$ multiply-add operations each of length $H$:

$$T_{\text{accum}} = \frac{B \times k \times H \times 2}{\text{FLOP rate}} = \frac{B \times 8 \times 7168 \times 2}{\text{FLOP rate}} = \frac{B \times 114{,}688}{\text{FLOP rate}}$$

This is much smaller than $T_{\text{expert ffn}}$ for any realistic FLOP rate.

### Summary Table

| Term | Formula | Value at $B=1$ | Value at $B=32$ |
|---|---|---|---|
| $T_{\text{route}}$ | $2BHE / \text{FR}$ | $3.7 \text{ MFLOPs} / \text{FR}$ | $117 \text{ MFLOPs} / \text{FR}$ |
| $T_{\text{dispatch}}$ | $V_d / \text{BW}$ | $\approx 0.26$ ms | $\approx 0.51$ ms |
| $T_{\text{expert ffn}}$ | $4BHD / \text{FR}$ | $\ll T_{\text{dispatch}}$ | depends on $D$ [UNVERIFIED] |
| $T_{\text{combine}}$ | $\approx T_{\text{dispatch}}$ | $\approx 0.26$ ms | $\approx 0.51$ ms |
| $T_{\text{accum}}$ | $2BkH / \text{FR}$ | $0.11 \text{ MFLOPs} / \text{FR}$ | $3.7 \text{ MFLOPs} / \text{FR}$ |

## Section 2: Dominant Cost Regime Identification

The execution regime is determined by which term dominates $T_{\text{total}}$.

### Communication-Bound (Decode)

When $B$ is small, $T_{\text{dispatch}} + T_{\text{combine}} \gg T_{\text{expert ffn}}$
because:
- $V_{\text{dispatch}}$ has a minimum value at $C = 1$ (floor of capacity formula),
  independent of $B$ below $B = 32$.
- $T_{\text{expert ffn}} \propto B$ but starts near zero.

At $B = 1$: $T_{\text{dispatch}} + T_{\text{combine}} \approx 0.52$ ms; expert FFN
compute is negligible (1–2 tokens per expert). Communication dominates.

### Compute-Bound (Prefill)

When $B$ is large, $T_{\text{expert ffn}} \propto B$ grows without bound while
$T_{\text{dispatch}} + T_{\text{combine}}$ grows only via the ceiling $C(B)$.
At sufficiently large $B$, expert FFN dominates.

### Crossover Batch Size

The crossover $B_{\text{cross}}$ satisfies $T_{\text{dispatch}} \approx T_{\text{expert ffn}}$:

$$\frac{V_{\text{dispatch}}}{\text{BW}} \approx \frac{4 \times B_{\text{cross}} \times H \times D}{\text{FLOP rate}}$$

$$B_{\text{cross}} \approx \frac{V_{\text{dispatch}} \times \text{FLOP rate}}{4 \times \text{BW} \times H \times D}$$

At $D \approx H \approx 7168$ [UNVERIFIED], $V_{\text{dispatch}} \approx 3.2$ MB (at $C=1$),
$\text{BW} = 12.5 \times 10^9$ B/s:

$$B_{\text{cross}} \approx \frac{3.2 \times 10^6 \times \text{FLOP rate}}{4 \times 12.5 \times 10^9 \times 7168^2}$$

For a Wormhole B0 FLOP rate in the range of tens to hundreds of TFLOPs/s [UNVERIFIED],
$B_{\text{cross}}$ is estimated to be in the range of **4–16 tokens**.

> **Warning:** $B_{\text{cross}}$ depends on both $D$ and the achieved FLOP rate, neither
> of which is verified here. This is an estimate. Calibrate $B_{\text{cross}}$ by
> measuring $T_{\text{dispatch}}$ and $T_{\text{expert ffn}}$ directly on hardware
> (see Section 4).

## Section 3: Effect of Double-Buffering on Observed Latency

### Without Double-Buffering

All stages execute serially for a single batch of $B$ tokens:

$$T_{\text{serial}} = T_{\text{route}} + T_{\text{dispatch}} + T_{\text{expert ffn}} + T_{\text{combine}} + T_{\text{accum}}$$

### With Double-Buffering (2 Micro-Batches of $B/2$)

The pipeline overlap hides $T_{\text{route}}$ for micro-batch $i+1$ behind the slow
stages of micro-batch $i$. The effective per-micro-batch time becomes:

$$T_{\text{per mb}} = \max(T_{\text{route}}, T_{\text{dispatch}} + T_{\text{expert ffn}} + T_{\text{combine}}) + T_{\text{accum}}$$

The total time for both micro-batches (after pipeline fill):

$$T_{\text{double buf}} \approx T_{\text{per mb}} + T_{\text{per mb}} - T_{\text{route}}$$

In steady state, the pipeline delivers one micro-batch every $T_{\text{per mb}}$
cycles, saving approximately:

$$\Delta T = T_{\text{serial}} - 2 \times T_{\text{per mb}} \approx T_{\text{route}}$$

whenever $T_{\text{route}} < T_{\text{dispatch}} + T_{\text{expert ffn}} + T_{\text{combine}}$.

### Regime-Specific Benefit

| Regime | $T_{\text{route}}$ vs. pipeline stages | Double-buffer benefit |
|---|---|---|
| Decode ($B \leq 32$) | $T_{\text{route}} \ll T_{\text{dispatch}} + T_{\text{combine}}$ | Saves approximately $T_{\text{route}}$ per layer |
| Prefill ($B \geq 256$) | $T_{\text{route}} \ll T_{\text{expert ffn}}$ | Saves approximately $T_{\text{route}}$ per layer |

At decode, where every microsecond counts for latency-sensitive inference, hiding
$T_{\text{route}}$ provides a consistent, predictable speedup.

```python
def effective_latency_double_buffer(
    T_route: float,
    T_dispatch: float,
    T_expert: float,
    T_combine: float,
    T_accum: float,
) -> float:
    """
    Compute effective per-batch latency with double-buffering.
    All times in milliseconds.
    Returns the time to process the full batch (both micro-batches).
    """
    pipeline_stages = T_dispatch + T_expert + T_combine
    T_per_mb = max(T_route, pipeline_stages) + T_accum
    # Two micro-batches, with second's route overlapping first's pipeline stages
    T_total = T_per_mb + max(0.0, T_accum + T_per_mb - T_route)
    # Simplified: approximate as 2 * T_per_mb - overlap
    overlap = min(T_route, pipeline_stages)
    return 2.0 * T_per_mb - overlap


def effective_latency_serial(
    T_route: float,
    T_dispatch: float,
    T_expert: float,
    T_combine: float,
    T_accum: float,
) -> float:
    """Serial latency (no pipelining)."""
    return T_route + T_dispatch + T_expert + T_combine + T_accum
```

## Section 4: Model Calibration Procedure

Because $D$ [UNVERIFIED] and the hardware FLOP rate are not confirmed, the latency
model must be calibrated against measured data. The following five-step procedure is
recommended.

**Step 1: Measure $T_{\text{dispatch}}$ and $T_{\text{combine}}$**

Run the all-to-all dispatch and combine kernels in isolation using the TTNN profiler
at target batch sizes ($B \in \{1, 4, 8, 16, 32, 64, 128\}$). Record latency for
each.

```python
import ttnn

# Benchmark dispatch latency at B=32
B = 32
dispatch_buf = ttnn.from_torch(
    torch.zeros(capacity(B) * E_d, H, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT,
    device=device,
)
# Use TTNN profiler context to measure kernel time
with ttnn.profiler.ProfileScope("dispatch_b32"):
    ttnn.all_to_all(dispatch_buf, cluster_axis=1,
                    topology=ttnn.Topology.Linear)
```

**Step 2: Measure $T_{\text{expert ffn}}$**

Isolate a single expert FFN kernel (two matmuls) with synthetic received-token buffers
of size $[\text{received count}, H]$ for varying `received_count` values. This
directly measures the compute contribution without communication noise.

**Step 3: Measure $T_{\text{route}}$**

Benchmark the router projection ($[B, H] \times [H, E]$ matmul + sigmoid + top-$k$)
in isolation using the TTNN profiler.

**Step 4: Compute Predicted $T_{\text{total}}$ and Compare**

Sum the measured components:

$$T_{\text{predicted}} = T_{\text{route}} + T_{\text{dispatch}} + T_{\text{expert ffn}} + T_{\text{combine}} + T_{\text{accum}}$$

Run the full end-to-end MoE layer and compare to $T_{\text{predicted}}$. Discrepancy
indicates either kernel launch overhead, L1 eviction pressure, or NoC contention not
captured by the linear model.

**Step 5: Identify Bottleneck and Apply Targeted Optimization**

| Bottleneck | Symptom | Optimization |
|---|---|---|
| $T_{\text{dispatch}}$ or $T_{\text{combine}}$ | $T_{\text{dispatch}} > 50\%$ of $T_{\text{total}}$ | Increase `num_links`; use GDF routing to reduce volume |
| $T_{\text{expert ffn}}$ | $T_{\text{expert ffn}} > 50\%$ of $T_{\text{total}}$ | Expert batching; increase micro-batch size |
| $T_{\text{route}}$ | $T_{\text{route}} > T_{\text{dispatch}}$ | Fuse router into prior layer; use router kernel fusion (ch05) |
| $T_{\text{accum}}$ | Residual after optimization | Fuse with residual add (see `combine_accumulation.md` Section 4) |

## Section 5: Bottleneck Identification for Qwen3.5-35B

### At $B = 1$ (Single-Token Decode)

| Stage | Estimated Latency | Dominant? |
|---|---|---|
| $T_{\text{route}}$ | Very small ($\ll 0.1$ ms) | No |
| $T_{\text{dispatch}}$ | $\approx 0.26$ ms | Yes |
| $T_{\text{expert ffn}}$ | Near-zero (1–2 tokens per expert) | No |
| $T_{\text{combine}}$ | $\approx 0.26$ ms | Yes |
| $T_{\text{accum}}$ | Near-zero | No |
| **Total (approx.)** | **$\approx 0.52$ ms** | Communication-bound |

At $B = 1$, the communication cost for dispatch and combine is fixed at approximately
$2 \times 0.26 = 0.52$ ms, and expert FFN compute is negligible. The design is purely
communication-bound. Double-buffering provides a small but useful reduction by hiding
$T_{\text{route}}$.

### At $B = 32$ (Decode Batch)

| Stage | Estimated Latency | Dominant? |
|---|---|---|
| $T_{\text{route}}$ | Small | No |
| $T_{\text{dispatch}}$ | $\approx 0.51$ ms | Significant |
| $T_{\text{expert ffn}}$ | Depends on $D$ [UNVERIFIED] and FLOP rate | Possibly significant |
| $T_{\text{combine}}$ | $\approx 0.51$ ms | Significant |
| $T_{\text{accum}}$ | Small | No |
| **Total (approx.)** | **$\approx 1.02$ ms + $T_{\text{expert ffn}}$** | Mixed regime |

At $B = 32$, communication ($\approx 1.02$ ms) and expert FFN compute are both
potentially significant. Whether compute has surpassed communication depends on the
confirmed value of $D$ and the achieved Wormhole B0 FLOP rate. Profile to determine
which term dominates at this operating point.

> **Tip:** If $T_{\text{expert ffn}} \approx T_{\text{dispatch}}$ at $B = 32$,
> then $B = 32$ is near the crossover $B_{\text{cross}}$. In this regime, both
> compute and communication optimizations are worth pursuing simultaneously.

## References

- Chapter 2 of this guide: `ch02_all_to_all_primitives/all_to_all_dispatch.md`,
  `ch02_all_to_all_primitives/all_to_all_combine.md` (bandwidth and hop model).
- Chapter 3 of this guide: GDF routing communication cost ($\approx 1.292 \times
  \text{OPT}$ for $N = 8$).
- Chapter 4 of this guide: `ch04_expert_device_assignment/uniform_partitioning.md`
  ($W_{\text{expert}}$, $E_d = 32$, capacity $C$).
- TT-Metalium TTNN profiler documentation.
- Wormhole B0 datasheet: Ethernet link bandwidth (12.5 GB/s), Tensix FLOP rate
  [UNVERIFIED].

---

**Next:** [Chapter 7 — Load Balancing](../ch07_load_balancing/index.md)
