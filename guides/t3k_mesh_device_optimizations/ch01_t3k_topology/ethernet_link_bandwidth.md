# Ethernet Link Bandwidth on T3K Wormhole Devices

> **Quick Reference — TTNN API Symbols Introduced in This File**
>
> | Symbol | Module | Description |
> |---|---|---|
> | `num_links` | `ttnn` (parameter) | Integer controlling how many Ethernet links are used per adjacent device pair for a collective operation. **Preview only — formally defined in Chapter 3, `num_links_parameter.md`.** |

> **⚠ Unverified Hardware Figures:** One figure in this file is assumed but not yet confirmed against the Wormhole B0 hardware specification: the physical link count per adjacent device pair on T3K (assumed to be 4). The 12.5 GB/s per-link bandwidth is derived arithmetically from the 100 Gb/s physical-layer specification and is not conditional on that figure. All bandwidth tables that depend on the 4-links-per-pair assumption are labeled `[placeholder — to be verified]`. Note: the total Ethernet port count per Wormhole device is also unverified, but it is an informational figure only — no downstream chapter requires it — so it does not affect any bandwidth derivations or tuning guidance in this guide.

The Ethernet links between Wormhole chips are the sole channel through which data moves between devices on the T3K board. Every all-to-all, all-reduce, all-gather, and reduce-scatter operation is ultimately serialized onto these links. Understanding their bandwidth, latency, and contention behavior is therefore not optional background material — it is the quantitative basis for every tuning decision made in Chapters 3 through 7.

---

## Ethernet Ports on the Wormhole ASIC

Each Wormhole ASIC includes a set of dedicated Ethernet ports that operate independently of the Tensix compute fabric and the NOC. These are high-speed serial links implementing a custom protocol layered on Ethernet-compatible physical signaling. Each individual Ethernet link on Wormhole provides approximately **12.5 GB/s of unidirectional bandwidth** (per Wormhole hardware specification, each Ethernet port provides 100 Gb/s physical-layer bandwidth; protocol and encoding overhead reduce effective data bandwidth to the ~12.5 GB/s range). All subsequent bandwidth figures in this file are in GB/s (gigabytes per second, base-10) per the guide conventions; the 100 Gb/s figure is a physical-layer specification.

A single Wormhole device has `[placeholder — to be verified]` Ethernet ports in total. `[placeholder — to be verified]` On the T3K board, the wiring of these ports between adjacent chips determines how many simultaneous physical links exist between each device pair.

---

## Link Count per Device Pair on T3K

On the current T3K hardware configuration, each adjacent device pair is connected by 4 (assumed; see warning above) Ethernet links. Not all 4 links are necessarily used by a given collective operation — that is controlled by the `num_links` parameter — but all 4 are physically present and available.

The valid range for `num_links` is therefore 1–4 (integers only). Setting `num_links` to any value outside this range, or to a non-integer, is an error. `num_links=1` uses a single link pair; `num_links=4` uses all available link pairs at maximum bandwidth.

The remaining Ethernet ports on each device (beyond those used for intra-board neighbor connections) are either routed to external board-edge connectors for inter-board cabling, or left unpopulated. For single-board T3K deployments, only the intra-board links are active.

---

## Unidirectional and Bidirectional Bandwidth

### Per-Link Bandwidth

Each physical Ethernet link on Wormhole is a full-duplex connection: it can simultaneously carry traffic in both directions. The unidirectional bandwidth per link is approximately **12.5 GB/s** in each direction. This figure represents the peak data throughput achievable for large, well-aligned transfers on a single link with no contention from other traffic.

### Per-Device-Pair Aggregate Bandwidth

With exactly 4 physical links between adjacent device pairs on T3K (Wormhole B0 board revision), the aggregate bandwidth between two neighboring devices scales with the number of active links (assuming exactly 4 wired links per pair):

> **Note:** The following table assumes exactly 4 physical Ethernet links per device pair (4 assumed; see warning above). All bandwidth figures are conditional on this count.

| Active Links | Unidirectional Bandwidth | Bidirectional Bandwidth |
|---|---|---|
| 1 | ~12.5 GB/s | ~25 GB/s |
| 2 | ~25 GB/s | ~50 GB/s |
| 3 | ~37.5 GB/s | ~75 GB/s |
| 4 (assumed; see warning above) | ~50 GB/s (assuming 4 wired links) | ~100 GB/s (assuming 4 wired links) |

These are theoretical peak figures assuming independent, non-contending transfers on each link. Achieved bandwidth in practice is somewhat lower due to protocol overhead, buffer management latency, and NOC arbitration on the receiving device. For large tensor transfers (several MiB per transfer), practical achieved bandwidth typically reaches 85–95% of the theoretical peak per link pair. For small transfers (tens of KiB or less), per-transfer overhead dominates and achieved bandwidth drops significantly; this is the regime relevant to single-token decode steps (see the latency section below).

### Aggregate Bandwidth Across All Seven Link Groups

Eight devices on a linear chain produce seven adjacent device pairs and therefore seven link groups.

For a ring collective that uses all eight devices simultaneously and routes traffic in both directions along the ring, all seven adjacent link groups (0↔1, 1↔2, 2↔3, 3↔4, 4↔5, 5↔6, 6↔7) carry traffic at the same time. At `num_links=4` (assuming 4 wired links per pair), the operative aggregate bandwidth is **350 GB/s in each direction simultaneously** across all seven full-duplex link groups (conditional on the 4-link assumption; see warning box above). Each full-duplex link can in principle use both directions simultaneously, subject to the saturation limits described in the Saturation Thresholds section below. In practice, ring all-reduce and all-to-all operations cannot fully saturate all links simultaneously because the algorithm has phases where some links are idle while computation or buffering occurs; achievable aggregate throughput is lower, and Chapter 3 provides empirical guidance.

---

## Link Latency

Link latency on Wormhole intra-board Ethernet has two main components:

1. **Serialization latency**: the time to serialize a packet of a given size onto the wire at the link's signaling rate. For a minimal-size packet at 100 Gb/s, serialization is sub-microsecond.
2. **Pipeline latency**: the fixed delay through the Ethernet MAC/PHY pipeline on both the transmit and receive sides, plus NOC traversal latency on the destination device to deliver the data to its target cores or DRAM bank.

For intra-board T3K links, the guide's working conservative estimate for the pipeline latency floor is ~1.7 µs (MAC/PHY pipeline plus NOC delivery on both chips, typical firmware stack). This figure accounts for the short PCB trace propagation time (negligible at centimeter scale) and the MAC/PHY pipeline on both chips. The serialization component for an 8 KiB token embedding (Qwen3MoE `hidden_dim=4096`, bfloat16) at 12.5 GB/s is ~0.65 µs (8192 bytes ÷ 12.5×10⁹ bytes/s ≈ 0.00000065 s ≈ 0.65 µs); end-to-end single-hop latency for this payload is therefore ~1.7 + 0.65 ≈ 2.35 µs (pipeline floor plus serialization term), which rounds to approximately ~2.4 µs or the range ~2–3 µs.

For larger messages, the effective per-byte latency is dominated by serialization throughput (bandwidth), not pipeline latency. The pipeline latency term becomes relevant primarily for small messages in decode-phase all-to-all operations, where the payload per device pair may be as small as a single token embedding (for example, 8 KiB for a 4096-dimensional hidden state in bfloat16 — Qwen3MoE uses `hidden_dim=4096`, so a single token is 4096 × 2 bytes = 8 KiB), and where the overhead of initiating the link transfer is comparable in magnitude to the data transfer time itself.

### Latency as a Function of Transfer Size

The table below covers single-hop transfers (between adjacent device pairs, i.e., devices N and N+1); multi-hop latency for non-adjacent pairs is addressed in the next section.

The following table provides estimated end-to-end latency for single-hop transfers at `num_links=1` on T3K Wormhole.

| Transfer Size | Estimated Latency (`num_links=1`) | Status |
|---|---|---|
| 8 KiB (one token, hidden_dim=4096, bfloat16 — Qwen3MoE target) | ~2–3 µs (~2.4 µs; derived as ~1.7 µs pipeline floor + ~0.65 µs serialization) | [placeholder — to be filled] |
| 64 KiB | ~7–12 µs | [placeholder — to be filled] |
| 1 MiB | ~90–110 µs | [placeholder — to be filled] |
| 16 MiB | ~1.3–1.5 ms | [placeholder — to be filled] |

These estimates are based on the theoretical per-link bandwidth and the ~1.7 µs pipeline latency floor established above. Empirical benchmarking using the TTNN profiler (covered in Chapter 6) is required to obtain production-accurate numbers for a given firmware version and driver stack.

---

## Direct-Neighbor vs. Multi-Hop Routing

When a collective operation needs to send data between two non-adjacent devices, TTNN routes the message through intermediate "relay" devices. Each intermediate hop adds approximately one link traversal's worth of latency and consumes link bandwidth on two additional link groups (the incoming link on the relay device and the outgoing link to the next hop).

The hop count for a direct-path traversal on the linear chain 0–1–2–3–4–5–6–7 is:

| Source Device | Destination Device | Hop Count (shorter path) |
|---|---|---|
| 0 | 1 | 1 |
| 0 | 2 | 2 |
| 0 | 3 | 3 |
| 0 | 4 | 4 |
| 0 | 7 | 7 |
| 3 | 4 | 1 |
| 3 | 7 | 4 |

For an open linear chain (no ring), the worst-case hop count is 7 (device 0 to device 7 or vice versa). For ring-based collectives where both directions are available simultaneously, the maximum hop count per data chunk drops to ⌈(N−1)/2⌉ = 4 hops (each chunk is directed toward the nearer half of the chain in its assigned direction; the two halves of data never travel more than 4 hops from their source). This 4-hop maximum results from the ring algorithm's chunk-assignment rule — each chunk is directed toward the shorter-distance half of the chain — and holds on the open chain despite the absence of a physical wrap-around edge.

> **Open chain vs. closed ring:** On a closed ring (with a physical wrap-around edge between devices 0 and 7), the worst-case routing distance is 4 hops in both directions — the wrap-around provides the short path for the "far half" of the chain. On the T3K open chain, there is no wrap-around edge. A ring all-to-all running in only one direction on the open chain must route the longest pairs (e.g., device 0 to device 7) the full 7 hops — there is no short alternative direction. This is why **bidirectional operation (routing simultaneously in both directions along the chain) is essential for the open-chain topology**: bidirectionality allows each chunk to be assigned to whichever direction is shorter, achieving the same 4-hop maximum that a closed ring achieves via wrap-around. The 4-hop figure advertised for ring collectives on T3K applies only when bidirectional routing is active; unidirectional ring operation on this open chain has a worst-case of 7 hops.

**Effective bandwidth degrades with hop count** because the data traversing a multi-hop path consumes link bandwidth on every intermediate link group. A transfer from device 0 to device 3 at `num_links=2` uses the 0↔1 link group, then the 1↔2 link group, then the 2↔3 link group — all at the same bandwidth. However, because each relay device is also simultaneously involved in its own portions of the collective (receiving from one neighbor and forwarding to another), relay devices see higher link utilization than edge devices, and the link groups adjacent to the center of the chain are the first to saturate in asymmetric traffic patterns.

For well-designed ring collectives, each device sends and receives exactly one "chunk" of data to and from each of its two neighbors per communication round, which distributes link load uniformly and avoids the center-chain saturation problem.

---

## How Bandwidth Scales with `num_links`

The `num_links` parameter (formally defined in Chapter 3, `num_links_parameter.md`) controls how many of the up to 4 physical Ethernet links between adjacent device pairs a given collective operation is permitted to use. It is set per-call on `ttnn.all_to_all` and related operations.

Bandwidth scales near-linearly with `num_links` up to the point where either the NOC on the sending or receiving device becomes the bottleneck, or where the link groups become saturated by concurrent operations:

- From `num_links=1` to `num_links=2`: throughput nearly doubles for large payloads, since two independent links carry data simultaneously with no interference.
- From `num_links=2` to `num_links=4`: throughput continues to scale near-linearly, subject to NOC injection rate limits.
- Beyond the saturation point: adding more `num_links` provides no further improvement and may increase scheduling overhead.

The near-linear scaling assumption holds when the collective is the only traffic on the links. When multiple concurrent collectives are in-flight — for example, when a pipelined MoE layer overlaps the all-to-all dispatch of layer N+1 with the expert compute of layer N — the available link bandwidth is divided among the concurrent operations. In this regime, the individual `num_links` setting for each collective may need to be reduced to avoid contention (see Chapter 3 for detailed guidance).

### Bandwidth Scaling Summary

| `num_links` | Theoretical Peak (per device pair) | Practical Large-Tensor Achieved | Status |
|---|---|---|---|
| 1 | 12.5 GB/s | ~11–12 GB/s | [placeholder — to be filled] |
| 2 | 25 GB/s | ~22–24 GB/s | [placeholder — to be filled] |
| 3 | 37.5 GB/s | ~33–36 GB/s | [placeholder — to be filled] |
| 4 | 50 GB/s | ~43–47 GB/s | [placeholder — to be filled] |

---

## Saturation Thresholds and Contention Effects

Ethernet link saturation occurs when the total data rate requested of a link group exceeds its physical capacity. As a practical rule of thumb, observable saturation effects begin when two or more concurrent all-to-all collectives share the same physical links — for example, running two simultaneous all-to-all operations on a single T3K board. Quantitative saturation benchmarks are marked `[placeholder — to be filled]` pending empirical measurement.

Exact quantitative saturation thresholds are hardware- and firmware-version-dependent. They will be measured and populated during the benchmarking phase described in Chapter 3, `benchmarking_num_links.md`.

On T3K, the most common causes of saturation are:

**Multiple concurrent collectives sharing link groups.** If two all-to-all operations are dispatched simultaneously (for example, one for the dispatch phase of the current MoE layer and one for the combine phase of the previous layer), and both route traffic over the same link groups, the effective bandwidth available to each is halved. TTNN's dispatch queue serializes operations by default, but explicit pipelining techniques (covered in Chapter 3 and Chapter 5) can create genuine concurrency.

**Asymmetric traffic.** If one device pair (e.g., 3↔4, the center of the chain) carries disproportionately more traffic than others — for instance, because an expert placement strategy concentrates popular experts at one end of the device range — the center link group saturates before the edge link groups, creating a bandwidth bottleneck for the entire collective. Symmetric expert placement (experts evenly distributed so that each device receives approximately equal token volume) avoids this.

**Small-message flooding.** In decode mode, if many small messages (a few KiB each) are sent in rapid succession on the same link group, the MAC/PHY pipeline on the Wormhole device can become the throughput limiter rather than the raw serialization rate. This manifests as higher-than-expected latency per operation even at low data volumes. Using `num_links=1` or `num_links=2` for decode-phase all-to-all (where payload is small) is often more efficient than using the maximum link count, because the overhead of coordinating across multiple links exceeds the bandwidth benefit for small messages.

**Contention with NOC traffic.** Wormhole's on-chip NOC and Ethernet subsystem share some routing resources in the on-chip interconnect fabric. Heavy concurrent DRAM access or large matmul operations that saturate the NOC can indirectly reduce Ethernet throughput by competing for NOC bandwidth on the delivery side (moving received data from the Ethernet receiver to target L1 or DRAM banks). This is an advanced interaction; it is not typically the primary bottleneck but can be observed in fully-pipelined decode workloads.

---

## Summary of Key Figures

| Parameter | Value | Notes |
|---|---|---|
| Ethernet ports per Wormhole device | `[placeholder — to be verified against Wormhole B0 spec]` | Not all used for intra-board on T3K (informational only — not required by subsequent chapters; resolve against Wormhole B0 spec) |
| Physical links per adjacent device pair (T3K intra-board) | 4 (assumed; see warning box) | All available; controlled by `num_links` at runtime |
| Unidirectional bandwidth per link | ~12.5 GB/s | 100 Gb/s physical; ~85–95% efficiency at large transfer sizes |
| Maximum unidirectional bandwidth per device pair | ~50 GB/s | At `num_links=4` (assumed 4 links; see warning box) |
| Maximum bidirectional bandwidth per device pair | ~100 GB/s | At `num_links=4`, full-duplex (assumed 4 links; see warning box) |
| Single-hop pipeline latency | ~1–3 µs | Intra-board PCB links; firmware/driver dependent |
| Maximum hop count (open chain, unidirectional) | 7 hops | Device 0 ↔ Device 7 (worst-case single-direction pair) |
| Maximum hop count (bidirectional traversal) | 4 hops | Effective max when simultaneous both-direction routing is used on the open chain; assumes concurrent forward and backward chain passes, not a physical ring closure |

These figures establish the bandwidth ceiling and latency floor against which all collective operation designs in later chapters are measured.

---

**Next:** [topology_implications_for_collectives.md](./topology_implications_for_collectives.md)
