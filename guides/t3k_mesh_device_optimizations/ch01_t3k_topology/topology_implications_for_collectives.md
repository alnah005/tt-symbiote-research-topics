# Topology Implications for Collective Operations

> **Quick Reference — TTNN API Symbols Introduced in This File**
>
> | Symbol | Module | Description |
> |---|---|---|
> | `ttnn.all_to_all` | `ttnn` | All-to-all collective: routes distinct data from each device to every other device. Primary communication primitive for MoE expert dispatch and combine. **Preview only — formally defined in Chapter 2, `collective_primitives.md`.** |
> | `ttnn.all_gather` | `ttnn` | Collective that replicates each device's tensor slice to all devices. Used in special-case expert parallelism scenarios. **Preview only — formally defined in Chapter 2, `collective_primitives.md`.** |
> | `ttnn.reduce_scatter` | `ttnn` | Collective combining reduce and scatter: reduces all devices' data and distributes distinct slices to each. **Preview only — formally defined in Chapter 2, `collective_primitives.md`.** |

The previous two files established what the T3K hardware looks like (a linear 1x8 Wormhole mesh) and how fast its links are (up to ~50 GB/s unidirectional per adjacent device pair at `num_links=4` (conditional on the 4-link-per-pair assumption; see `ethernet_link_bandwidth.md`)). This file derives the consequences of that geometry for collective operation design — specifically for the all-to-all and all-reduce collectives that dominate the communication cost of Mixture-of-Experts (MoE) inference.

---

## The Fundamental Constraint: A Linear Topology

The T3K 1x8 mesh is a linear graph, not a hypercube, torus, or fat tree. This simple structure has a profound effect on which collective algorithms are efficient and which are not. In a linear graph of N nodes, the topological properties that matter most for collectives are:

- **Bisection bandwidth**: the total bandwidth across the cut that divides the graph into two equal halves (in this case, the cut between devices 3 and 4). For the T3K linear chain at `num_links=4`, bisection bandwidth is ~50 GB/s unidirectional (~100 GB/s bidirectional at the single link group spanning device 3↔device 4) (conditional on the 4-link-per-pair assumption; see `ethernet_link_bandwidth.md` for the verification status).
- **Diameter**: 7 hops (open linear chain). Note: under bidirectional ring traversal, the maximum hop count any data chunk must travel per communication round is ⌈(N−1)/2⌉ = 4 hops — this is an algorithm property, not the graph diameter.
- **Degree**: each interior node has degree 2 (two neighbors), and the two endpoints (devices 0 and 7) have degree 1 in the intra-board neighbor graph (endpoint devices have additional Ethernet ports routed to inter-board connectors on multi-board configurations). This low degree limits the maximum amount of data a single device can simultaneously send or receive on a single T3K board.

These properties together mean that algorithms that push large amounts of data through the center of the chain — particularly tree-based collectives that funnel all traffic through a root device — perform poorly on T3K, while algorithms that distribute traffic uniformly across all link groups perform well.

---

## Why Ring-Based Collectives Fit the 1x8 Mesh

A ring all-to-all is one in which each device holds N-1 distinct outgoing messages — one per destination device. In each round, each device forwards its current outgoing messages one hop toward their respective destinations. Unlike ring all-gather, where each device circulates a single chunk to all N devices, ring all-to-all handles per-pair data independently. Bidirectional ring variants send data both left and right simultaneously, halving the number of rounds needed.

The ring collective algorithm fits the T3K linear mesh for several reasons:

**Traffic is distributed uniformly.** In a ring all-reduce, every device sends and receives exactly one chunk per round. In a ring all-to-all, in each round each device forwards all its in-transit chunks one hop toward their respective destinations — one hop on the same link toward the single neighbor in the chosen ring direction. These N-1 chunks are distributed evenly across N-1 rounds so that each link carries exactly one chunk per round under optimal scheduling (achieved by the standard ring shift schedule where each device routes one chunk per link per round toward each of its neighbors, with staggered departure times ensuring each link carries at most one chunk in each direction per round) — the uniform-load conclusion holds for this different reason. In both cases, every link group (0↔1, 1↔2, ... 6↔7) carries exactly the same amount of traffic. No single link group is a bottleneck; all links are equally utilized. This is the optimal traffic pattern for a network where all links have equal capacity, which is precisely the T3K intra-board topology.

**The linear chain is structurally a ring minus one edge.** A ring and a linear chain differ only in whether there is an edge connecting the two endpoints (devices 0 and 7). On T3K, the two endpoints are not directly wired on the PCB. However, the ring all-to-all algorithm can still be operated on an open chain — it simply has a slightly higher communication volume than a true ring because the "wrap-around" transfers that would use the missing edge must instead be absorbed into the forward/backward passes along the chain. For most practical tensor sizes, this overhead is negligible, and TTNN's all-to-all implementation handles the open-chain case transparently.

**Scalability with `num_links`.** Because each ring step uses only the single adjacent link group between a device and its neighbor, increasing `num_links` proportionally increases the per-step bandwidth. A ring all-to-all at `num_links=4` has four times the throughput of the same operation at `num_links=1`, with no change in hop count or algorithm structure. This clean scaling relationship makes `num_links` a reliable tuning knob (Chapter 3).

---

## Hop Count Analysis: Ring vs. Tree vs. Direct

To understand why ring is preferred over alternatives, compare the hop count and bandwidth characteristics of three collective patterns applied to a linear 8-device chain:

### Ring All-to-All

As described above, each device holds N-1 distinct outgoing messages, forwarding them one hop per round toward their respective destinations. For an open chain of 8 devices:

| Direction | Rounds | Max hop count | Avg routing distance |
|---|---|---|---|
| Unidirectional | 7 | 7 | 4.0 hops (denominator = 7 one-directional distances: avg of 1,2,3,4,5,6,7 = 28/7 = 4.0; each of the 7 chunks travels the forced-direction distance to its destination) |
| Bidirectional | 4 | 4 | 3.0 hops (denominator = 28 unordered pairs, always taking shorter path: Σd(8−d)/28 = 84/28 = 3.0; see `t3k_physical_layout.md`) |

> **Note on denominators:** The two "Avg routing distance" figures above use different denominators and should not be directly compared as if computed on the same basis. The unidirectional 4.0 figure is the average over the 7 possible forced-direction distances (1 through 7); the bidirectional 3.0 figure is the average over all 28 unordered source–destination pairs taking the shorter of the two ring directions. The bidirectional figure is lower partly because shorter paths are chosen, and partly because the denominator counts all pairs rather than just the 7 forced distances.

> **Note:** For a bidirectional ring on a linear chain, optimal routing always uses the shorter of the two ring directions for each chunk. As a result, the average routing distance equals the average shortest-path distance by construction — this equality is not coincidental but holds because no chunk is ever sent the long way around when the short direction is available.

Note: in both cases the number of rounds equals the maximum hop count by construction (each chunk advances one hop per round). The figures that differ between the two directions are the 7-round maximum (unidirectional) vs. the 4-round maximum (bidirectional).

- Link utilization: uniform across all 7 link groups
- Bisection bandwidth usage: exactly 1× bisection bandwidth per direction (each round uses one direction of the 3↔4 link group; the algorithm never needs more than one simultaneous transfer across the center cut per direction)

### Tree All-Reduce (Binary Tree)

A rooted binary tree mapped to the linear chain (root=device 3, left subtree 0–2, right subtree 4–7 — unbalanced: 3 vs. 4 nodes per subtree) all-reduce on 8 devices requires a root device. Using device 3 as the root, with the left subtree comprising devices 0–2 and the right subtree comprising devices 4–7, the gather phase paths are:

- Device 0 → 1 → 2 → 3 (3 hops)
- Device 1 → 2 → 3 (2 hops)
- Device 2 → 3 (1 hop)
- Device 4 → 3 (1 hop)
- Device 5 → 4 → 3 (2 hops)
- Device 6 → 5 → 4 → 3 (3 hops)
- Device 7 → 6 → 5 → 4 → 3 (4 hops)

Note: Device 1 simultaneously relays Device 0's data and sends its own — intermediate nodes carry asymmetric load in a tree collective.

> **Note:** 'Hops' in the tree analysis above count physical PCB link traversals (the number of on-board Ethernet links crossed), not abstract logical tree depth. For example, the path device 7→6→5→4→3 traverses 4 physical links on the T3K chain, and device 0→1→2→3 traverses 3 physical links. Physical hop counts equal or exceed logical tree depth because the tree is mapped onto a linear topology.

In the scatter phase, the root broadcasts the reduced result back to all leaves, following the same paths in reverse. This creates a highly asymmetric traffic pattern: the center link groups 2↔3 and 3↔4 carry traffic from multiple branches simultaneously, while the edge link groups 0↔1 and 6↔7 carry traffic only for the leaf branches. The center link groups become a bandwidth bottleneck.

> **Footnote:** Tree construction choices affect individual routing paths; the center-link saturation effect holds for any tree construction on a linear chain.

Moreover, each interior relay node (including the root, device 3) must receive data from two branches simultaneously and send a reduced result to two branches simultaneously — requiring it to simultaneously use both its available link connections in each phase — 2× the per-phase link utilization of a leaf device, handling twice the data volume per round. At `num_links=4`, interior relay nodes are already saturating their available link bandwidth.

> **Note:** Ring all-to-all and binary tree all-reduce serve different semantics — all-to-all routes distinct per-pair data while all-reduce aggregates across all devices. The comparison below is purely topological, not an endorsement of all-reduce for MoE combine.

| Metric | Ring All-to-All | Binary Tree All-Reduce |
|---|---|---|
| Max hop count (physical PCB links, single message) | 7 (open chain, unidirectional); 4 (bidirectional) | 4 physical hops (device 7 to root device 3) |
| Communication rounds | N-1=7 (unidirectional); ceil((N-1)/2)=4 (bidirectional) | 4 (longest path: device 7→6→5→4→3 requires 4 causally sequential relay steps on this linear-chain mapping) |
| Link utilization | Uniform | Skewed toward center |
| Center link saturation | No | Yes, under heavy load |
| Scales with `num_links` | Linearly | Sub-linearly (center is bottleneck) |
| Latency (large tensor) | Low (pipeline efficiency) | Higher (center queuing) |
| Latency (small tensor) | Higher (N-1 rounds) | Equal round count vs. bidirectional ring (4 rounds each); fewer than unidirectional ring (4 vs. 7 rounds) (tree not available as a TTNN collective on T3K; ring is the TTNN implementation) |

The binary tree has no round-count advantage over bidirectional ring on this topology (same round count as bidirectional ring; worse in throughput due to center saturation). However, the center link group saturation makes the tree worse in throughput terms for large tensors. For small tensors where latency rather than throughput is the constraint, the tree could in principle be faster — but if the ring algorithm is used for all collective operations on T3K (as documented in the TTNN API), tree-based collectives are not available as an alternative; see `[verify against current TTNN implementation]` for confirmation of this implementation status.

### Direct (All-Pairs) Transfer

A naive "direct" all-to-all where device i sends its data simultaneously to all other devices would require 7 simultaneous transfers per device, each routed through a different path. For device 0, this means simultaneously routing data to devices 1 (1 hop), 2 (2 hops), 3 (3 hops), 4 (4 hops), 5 (5 hops), 6 (6 hops), and 7 (7 hops). All of these transfers must traverse the 0↔1 link group simultaneously, requiring device 0 to push 7 streams of data through that single link group — immediately saturating it at any `num_links` setting. Every intermediate device faces an even worse situation, as it must relay multiple in-transit streams while also injecting its own.

Direct all-pairs transfer is therefore not viable on a linear mesh and is not used by TTNN on T3K.

### Summary Comparison

| Pattern | Topology Fit | Throughput (large tensor) | Latency (small tensor) | Used by TTNN on T3K |
|---|---|---|---|---|
| Ring all-to-all | Excellent | High | Moderate | Yes |
| Ring all-reduce | Excellent | High | Moderate | Yes |
| Binary tree all-reduce | Poor | Moderate (center saturation) | Equal to bidirectional ring (4 rounds); fewer than unidirectional ring (4 vs. 7) | No |
| Direct all-pairs | Very poor | Very low (center saturation) | N/A | No |
| All-gather + reduce-scatter | Excellent | High (equivalent to ring) | Moderate | Yes (special cases) |

---

## Expert Parallelism Implications

In Mixture-of-Experts inference, each of the eight T3K devices holds a distinct subset of the model's expert networks. During each MoE layer, every input token must be routed to the device holding its selected expert, processed there, and the output returned to the originating device. This produces two all-to-all communication events per MoE layer: the dispatch (tokens go to expert devices) and the combine (expert outputs return to token-originating devices).

The T3K ring topology shapes this communication in specific ways.

### Which Devices Talk to Which

With 8 devices and N_experts experts distributed evenly (N_experts/8 per device), each device receives tokens from all other 7 devices during dispatch and sends expert outputs to all other 7 devices during combine. The communication pattern is a full all-to-all: every device is a source and a destination simultaneously. This matches exactly the ring all-to-all algorithm's design, confirming that ring all-to-all is the right primitive for MoE expert dispatch on T3K.

### Hop Count and Expert Placement

While ring all-to-all distributes traffic uniformly regardless of which tokens go to which experts, the hop count for individual token transmissions does depend on expert placement. A token on device 0 whose selected expert is on device 7 must travel 7 hops on the open chain (or 1 hop only if a wrap-around edge between devices 0 and 7 existed, which it does not on a single T3K board). In the ring algorithm this is handled automatically — the token piggybacks on the ring's normal circulation and reaches device 7 after at most ⌈(N−1)/2⌉ = 4 rounds under bidirectional ring traversal (7 rounds under unidirectional ring traversal) — but the effective latency for that specific token is higher than for a token whose expert is on device 1.

For the all-to-all as a whole, this does not change total throughput (all ring steps proceed in parallel), but it does affect the latency of the slowest token in a batch, which matters for decode-phase latency optimization. Expert placement strategies (Chapter 5) can exploit this: if tokens from device 0 disproportionately select experts near device 0 (e.g., devices 1–3 rather than devices 5–7), the effective average hop count decreases and per-token latency improves. However, this requires knowledge of the model's routing distribution, which is workload-dependent.

### Balanced vs. Unbalanced Expert Load

The ring all-to-all algorithm assumes that each device sends and receives approximately equal amounts of data in each round. In Mixture-of-Experts models with top-K routing, the actual number of tokens routed to each expert device in a given batch varies depending on the router's output distribution. If some experts are more popular than others (a common phenomenon in trained MoE models), some devices receive more tokens than others, creating a load imbalance.

Load imbalance affects the all-to-all in two ways. First, the devices with more tokens to send inject more data into the ring, potentially saturating their outgoing links before other devices finish. Second, the devices with more expert compute to perform take longer to complete the expert matmul step, delaying the combine all-to-all. Both effects slow down the end-to-end MoE layer for the entire batch. Chapter 5 covers expert placement strategies for load balancing in detail.

---

## Introducing `num_links` as a Tunable Parameter

The `num_links` parameter was introduced in `ethernet_link_bandwidth.md`; here we consider it from the perspective of collective algorithm design. The `num_links` parameter appears in the signature of `ttnn.all_to_all` and related collectives as an integer that controls how many of the available physical Ethernet links between adjacent device pairs are allocated to the collective operation. Its effect on bandwidth was established in `ethernet_link_bandwidth.md`.

For the ring all-to-all on a 1x8 T3K mesh, increasing `num_links` increases the bandwidth available for each ring step proportionally (up to the saturation threshold). This directly reduces the time spent in each communication round, improving both throughput (for large tensors) and latency (to a lesser extent, since pipeline latency is not reduced by additional links).

The tradeoff is that each active link consumes Ethernet port resources on both sides of each device pair. When `num_links=4` and a collective is in-flight, all 4 link pairs between adjacent devices are occupied by that collective's traffic. If a second concurrent collective also attempts to use those links, it must either wait (if TTNN serializes them) or accept reduced effective bandwidth (if TTNN allows concurrent link sharing).

For this reason, the optimal `num_links` setting depends on:

1. **Tensor size**: large tensors (prefill phase) benefit from high `num_links`; small tensors (decode phase) may not.
2. **Concurrency**: whether other collectives are in-flight simultaneously.
3. **Compute-communication overlap**: whether the links need to be partially idle during expert compute for a future collective to start without contention.

These considerations are analyzed quantitatively in Chapter 3 (`num_links_parameter.md`), with empirical benchmarking guidance in `benchmarking_num_links.md`. The central recommendation — use maximum `num_links` for prefill, benchmark 1–2 links for decode — follows directly from the topology and bandwidth analysis established in this chapter.

---

## Summary: Key Topology Conclusions

The following conclusions follow from the T3K physical layout and Ethernet link characteristics established in this chapter, and are assumed as background in all subsequent chapters:

1. **Ring all-to-all is the canonical collective for T3K MoE inference.** The linear 1x8 mesh, uniform link bandwidth, and MoE all-to-all communication pattern combine to make ring all-to-all the optimal algorithm — high throughput, uniform link utilization, linear scaling with `num_links`.

2. **`num_links` is the primary bandwidth tuning knob.** There are no skip-hop links or alternative routing paths on a single T3K board. The only way to increase the bandwidth available to a collective is to increase `num_links`. Chapter 3 treats this exhaustively.

3. **Expert placement affects hop count but not ring throughput.** Ring all-to-all is throughput-optimal regardless of which tokens go to which devices, but per-token latency is lower when experts are placed near the tokens that most frequently select them. Chapter 5 addresses placement strategy.

4. **Center link group saturation is the primary contention risk.** In asymmetric traffic patterns or concurrent collective operations, the link groups near the center of the chain (3↔4) saturate first. Symmetric expert placement and serialized collective dispatch are the primary mitigations.

5. **Prefill and decode have different optimal `num_links` settings.** Large prefill tensors are throughput-bound and benefit from high `num_links`; small decode tensors are latency-bound and often perform better with low `num_links` due to reduced coordination overhead.
