# The `num_links` Parameter: Bandwidth Model and Tuning Guidelines

> **Quick Reference — Parameter Introduced in This File**
>
> | Parameter | Type | Valid values (T3K) | Default |
> |---|---|---|---|
> | `num_links` | `int` | 1, 2, 4 (exact maximum requires verification against T3K documentation) | 1 |

`num_links` is the primary software tuning knob for `ttnn.all_to_all` performance on T3K. This file defines what it controls, derives the bandwidth model that governs its effect on latency, explains the latency vs. throughput trade-off for different payload sizes, describes link contention effects under concurrent operations, and provides qualitative recommendations by inference regime. For empirical measurement methodology, see `benchmarking_num_links.md`.

---

## Definition

`num_links` sets how many of the available Wormhole Ethernet links between each adjacent device pair are allocated to a given all-to-all (or other collective) operation. Each Wormhole device on T3K has multiple Ethernet links available for chip-to-chip communication. `num_links` selects how many of those links the collective runtime uses to carry traffic for this specific operation.

**What changes when `num_links` increases:**
- More physical link lanes carry the data in parallel, increasing aggregate throughput between each device pair.
- More links must be set up (coordinated, credited, and torn down) at the start and end of each message, increasing per-message setup overhead.

**What does NOT change with `num_links`:**
- The number of rounds in the T3K linear all-to-all sequence. The chain still requires $N - 1 = 7$ hops regardless of `num_links`.
- The total data volume being transferred.
- The `ttnn.Topology.Linear` traversal algorithm.

**Valid values.** On T3K, valid `num_links` values are 1, 2, and 4. The exact maximum number of Ethernet links per device pair available on a given T3K hardware revision should be verified against current T3K hardware documentation, as the maximum may differ across board revisions or firmware versions. Do not assume that values above the verified maximum will succeed; a `ValueError` is raised at collective dispatch time if `num_links` exceeds the available link count.

---

## Bandwidth Model

### Peak Bandwidth Per Device Pair

The theoretical peak unidirectional bandwidth between an adjacent device pair when `num_links = $n_l$` is:

$$\text{BW}_{\text{peak}}(n_l) = n_l \times \text{BW}_{\text{link}}$$

where $\text{BW}_{\text{link}} \approx 12.5\ \text{GB/s}$ is the per-link unidirectional bandwidth on Wormhole.

For specific values:
- $n_l = 1$: $\text{BW}_{\text{peak}} = 12.5\ \text{GB/s}$
- $n_l = 2$: $\text{BW}_{\text{peak}} = 25.0\ \text{GB/s}$
- $n_l = 4$ (if available): $\text{BW}_{\text{peak}} = 50.0\ \text{GB/s}$

### Effective Bandwidth for Large Payloads

For large payloads (prefill, or any transfer well above the amortization threshold), the link setup overhead is negligible and the effective bandwidth approaches the peak:

$$\text{BW}_{\text{eff}}(n_l) \approx n_l \times \text{BW}_{\text{link}} \quad \text{(large payload regime)}$$

Scaling is near-linear in this regime: doubling `num_links` approximately doubles the effective bandwidth and halves the transfer time. This holds as long as the workload can saturate all $n_l$ links simultaneously — that is, as long as the per-link payload $V / n_l$ is large enough that each link remains busy for most of the transfer duration.

### Effective Bandwidth for Small Payloads

For small payloads (decode with small batch, or any transfer where $V / n_l$ is small), each link carries a short burst of data. The total operation time includes a per-link setup overhead $\tau_{\text{setup}}$ that must be paid for each active link regardless of payload size:

$$T(n_l, V) = n_l \times \tau_{\text{setup}} + \frac{V}{n_l \times \text{BW}_{\text{link}}}$$

where:
- $V$ is the total payload volume for one hop of the all-to-all (bytes)
- $n_l$ is `num_links`
- $\tau_{\text{setup}}$ is the per-link setup overhead (hardware-dependent; must be measured empirically)
- $\text{BW}_{\text{link}} \approx 12.5\ \text{GB/s}$

This formula captures the fundamental trade-off:
- The first term, $n_l \times \tau_{\text{setup}}$, grows linearly with `num_links`. Using more links increases this cost.
- The second term, $V / (n_l \times \text{BW}_{\text{link}})$, decreases with more links. Using more links reduces the transfer time.

The two terms cross at a payload volume $V^*$ where the marginal benefit of adding a link equals the marginal overhead cost. For $V \gg V^*$, more links are always beneficial. For $V \ll V^*$, fewer links reduce total latency.

**Finding $V^*$:** The crossover point between `num_links = $n_l$` and `num_links = $n_l + 1$` can be found by setting the two expressions equal:

$$n_l \times \tau_{\text{setup}} + \frac{V}{n_l \times \text{BW}_{\text{link}}} = (n_l + 1) \times \tau_{\text{setup}} + \frac{V}{(n_l + 1) \times \text{BW}_{\text{link}}}$$

Solving for $V$:

$$V^*(n_l \to n_l+1) = n_l \times (n_l + 1) \times \tau_{\text{setup}} \times \text{BW}_{\text{link}}$$

Because $\tau_{\text{setup}}$ is hardware-dependent and must be measured empirically, this formula is most useful as a structural guide: $V^*$ scales quadratically in $n_l$ and linearly in $\tau_{\text{setup}}$. Empirical measurement of $\tau_{\text{setup}}$ via the benchmarking procedure in `benchmarking_num_links.md` is necessary to evaluate this formula numerically.

---

## Latency vs. Throughput Trade-Off by Regime

### Large Payload (Prefill): Throughput-Bound

During prefill with $B = 32$ sequences of length $S = 2048$, the per-layer dispatch all-to-all volume per hop is approximately:

$$V_{\text{hop, prefill}} = C \times H \times 2 = \lceil k \times B \times S / E \rceil \times H \times 2$$

For Qwen3.5-35B [D UNVERIFIED]:

$$C = \lceil 8 \times 32 \times 2048 / 256 \rceil = \lceil 2048 \rceil = 2048\ \text{tokens}$$

$$V_{\text{hop, prefill}} = 2048 \times 7168 \times 2 = 29{,}360{,}128\ \text{bytes} \approx 28\ \text{MiB}$$

At 28 MiB per hop and $\text{BW}_{\text{link}} = 12.5\ \text{GB/s}$, a single-link transfer takes approximately:

$$T_1 = 28 \times 10^6 / (12.5 \times 10^9) \approx 2.24\ \text{ms per hop}$$

With `num_links=4`:

$$T_4 \approx 2.24\ \text{ms} / 4 = 0.56\ \text{ms per hop}$$

At these payload sizes, the setup overhead term $n_l \times \tau_{\text{setup}}$ is negligible relative to the transfer time. Use the maximum available `num_links` in prefill.

### Small Payload (Decode): Latency-Bound

During decode with $B = 1$, the per-hop volume is:

$$V_{\text{hop, decode}} = C \times H \times 2 = 1 \times 7168 \times 2 = 14{,}336\ \text{bytes} \approx 14\ \text{KiB}$$

At 14 KiB per hop, the transfer completes in approximately:

$$T_1 = 14{,}336 / (12.5 \times 10^9) \approx 1.1\ \mu\text{s per hop}$$

If $\tau_{\text{setup}}$ is on the order of microseconds (which is plausible for high-speed SerDes initialization), then $n_l \times \tau_{\text{setup}}$ for $n_l = 4$ could easily equal or exceed the 1.1 µs transfer time. In this case, `num_links=4` would increase total latency relative to `num_links=1`.

This illustrates why decode latency must be measured empirically. The crossover volume $V^*$ depends on $\tau_{\text{setup}}$, which is not publicly documented and may change between firmware versions.

### Practical Guidance (Qualitative)

| Inference regime | Tensor size per hop | Recommended starting `num_links` | Rationale |
|---|---|---|---|
| Prefill ($B \geq 8$, $S \geq 512$) | Large (MiB range) | Maximum available | Throughput-bound; setup overhead negligible |
| Mixed-batch decode ($B = 16$–$32$) | Medium (hundreds of KiB) | 2 | Benchmark to confirm; setup overhead partially amortized |
| Single-sequence decode ($B = 1$–$4$) | Small (KiB range) | 1 | Latency-bound; setup overhead may dominate |
| Pipelined multi-op (concurrent collectives) | Any | $\leq$ half max | Reserve links to avoid contention |

These are starting points for benchmarking, not guaranteed optima. Use `benchmarking_num_links.md` to measure and confirm the best value for your specific workload.

---

## Link Contention

### Multiple Concurrent Collectives

When two collective operations are in flight concurrently — for example, when a combine all-to-all from layer $L$ overlaps with a dispatch all-to-all from layer $L+1$ via pipelining — both operations compete for the same physical Ethernet links between adjacent device pairs.

If both operations request `num_links=4`, and only 4 links are available per pair, they must share. The TTNN runtime serializes link allocation if there is contention, which means one collective waits for the other to release links before it can begin. The net effect is that the two collectives run sequentially rather than in parallel, despite the intent to overlap them.

To allow true overlap, limit each concurrent collective to a `num_links` value small enough that both fit simultaneously:

$$n_{l,1} + n_{l,2} \leq n_{l,\text{max}}$$

For example, if the maximum is 4 links, use `num_links=2` for each of two concurrent collectives. This halves the per-collective bandwidth but allows genuine overlap, which can yield lower total latency if the compute phase between the two collectives is short.

### Interaction with `ttnn.Topology.Linear`

On T3K, the linear-chain all-to-all uses unidirectional traversal: in each of the 7 rounds, data moves in one direction along the chain. Each round uses the same $n_l$ links for all 7 hops in sequence. The links are occupied for the full duration of all 7 rounds.

This means that the links are not available for other collectives during the entire execution of one `ttnn.all_to_all` call (which spans all 7 rounds). Contention analysis must account for the full collective duration, not just one round.

### Detecting Contention

Contention manifests as unexpectedly high latency that does not improve (or actually worsens) when `num_links` is increased. If a latency sweep shows non-monotonic behavior (e.g., `num_links=2` is slower than `num_links=1`), contention from a concurrent operation is a likely cause. Chapter 6, `device_perf_counters.md`, describes how to use Ethernet link utilization counters to confirm link contention.

---

## Interaction with `cluster_axis` and Mesh Shape

For T3K with a `(1, 8)` mesh and `cluster_axis=1`, the all-to-all traverses the 8 devices on the column axis. The `num_links` parameter controls the links on this axis. If the mesh is extended to a `(2, 8)` multi-board configuration and `cluster_axis=0` is used for a cross-board collective, `num_links` then controls the inter-board links, which may have different physical characteristics and different optimal values.

The tuning recommendations in this file apply specifically to single-board T3K (`(1, 8)` mesh, `cluster_axis=1`). Multi-board configurations require separate benchmarking.

---

## Summary: Choosing `num_links`

```
if regime == "prefill":
    num_links = max_available           # throughput-bound; use maximum
elif regime == "decode" and batch_size <= 4:
    num_links = 1                       # latency-bound; minimize setup overhead
elif regime == "decode" and batch_size <= 32:
    num_links = 2                       # benchmark to confirm
elif concurrent_collectives:
    num_links = max_available // 2      # reserve bandwidth for each concurrent op
else:
    num_links = 2                       # conservative default; benchmark to verify
```

The above is a heuristic starting point. The optimal value depends on the actual setup overhead $\tau_{\text{setup}}$ on your specific T3K hardware and firmware version. Always confirm with the benchmarking procedure in `benchmarking_num_links.md`.

---

## References

- Chapter 1, `ethernet_link_bandwidth.md` — per-link bandwidth ($\text{BW}_{\text{link}} \approx 12.5\ \text{GB/s}$) and saturation behavior
- Chapter 1, `topology_implications_for_collectives.md` — linear-chain round structure and hop count
- Chapter 2, `collective_primitives.md` — `ttnn.all_to_all` API and the `num_links` parameter definition
- Chapter 3, `all_to_all_in_moe.md` — data volume derivation for prefill and decode regimes
- Chapter 3, `benchmarking_num_links.md` — empirical measurement procedure for $\tau_{\text{setup}}$ and optimal `num_links`
- Chapter 6, `device_perf_counters.md` — Ethernet link utilization counters for contention detection
