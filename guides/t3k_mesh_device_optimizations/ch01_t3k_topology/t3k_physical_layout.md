# T3K Physical Layout: The 1x8 Wormhole Device Mesh

The T3K system is a single accelerator board that integrates eight Wormhole ASICs onto one PCB, connected by a fixed pattern of on-board Ethernet traces. Understanding the physical arrangement of those chips — and how TTNN maps them to a logical coordinate grid — is the foundation for every multi-device API call, collective configuration, and expert placement decision in this guide.

> **Quick Reference — TTNN API Symbols Introduced in This File**
>
> | Symbol | Module | Description |
> |---|---|---|
> | `MeshDevice` | `ttnn` | Python class representing a multi-chip T3K device cluster. **Introduced here at a conceptual level; formally defined in Chapter 2, `mesh_device_setup.md`.** |
> | `cluster_axis` | `ttnn` collective ops | Integer (0 or 1) selecting which mesh axis a collective operation traverses. For a single T3K board (1×8 mesh), `cluster_axis=1` selects the column axis. **Preview only — formally defined in Chapter 2, `collective_primitives.md`.** |

---

## The Wormhole ASIC

Each device on the T3K board is a Wormhole ASIC, Tenstorrent's second-generation AI accelerator chip. Wormhole integrates a large grid of Tensix compute cores (each with its own L1 SRAM), a two-dimensional Network-on-Chip (NOC) for intra-chip data movement, multiple DRAM channels for off-chip storage, and a set of dedicated Ethernet MAC/PHY ports for chip-to-chip connectivity. The Ethernet ports on Wormhole are not general-purpose network interfaces; they are hardwired high-speed serial links designed specifically for low-latency, high-bandwidth chip-to-chip communication within and between boards.

Within a single Wormhole device the NOC fabric moves data between Tensix cores and between cores and DRAM at bandwidths that far exceed what the Ethernet links can provide. The Ethernet links are therefore not a replacement for the NOC; they are the interconnect between ASICs. When a collective operation such as all-to-all sends data from device 0 to device 3, that data leaves device 0 via Ethernet, crosses the PCB traces, and arrives at device 3's Ethernet receiver — at which point the NOC on device 3 distributes it to the appropriate Tensix cores or DRAM bank.

---

## Physical Arrangement on the Board

The eight Wormhole chips are laid out in a single row on the T3K PCB. From the perspective of the board's physical edge connectors, they are numbered sequentially from left to right. While the exact mechanical placement of each chip relative to the PCB edge connectors varies by board revision, the logical identity assigned to each chip — its device ID — is fixed in firmware and consistent across boards of the same hardware generation.

The board exposes the eight devices to the host system via PCIe. Typically one or two devices serve as PCIe root-port endpoints; the host driver enumerates them and derives the topology of the remaining chips through firmware-level discovery. TTNN abstracts this enumeration: when you call the `MeshDevice` constructor (Chapter 2), TTNN maps the discovered physical device list onto the logical coordinate space you specify.

---

## Device IDs and the Logical Coordinate System

TTNN identifies each Wormhole chip by a `device_id` in the range 0–7. These IDs correspond to the enumeration order established by the host driver and firmware and are stable within a single board instance across reboots.

For multi-device operations, TTNN introduces a two-dimensional logical mesh coordinate system. Each device is addressed as a `(row, col)` pair. For a T3K system with a single board, the mesh shape is `(1, 8)`: one row, eight columns. The mapping from device ID to `(row, col)` coordinates is:

| Device ID | Logical Row | Logical Col |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 0 | 3 |
| 4 | 0 | 4 |
| 5 | 0 | 5 |
| 6 | 0 | 6 |
| 7 | 0 | 7 |

This mapping is the default for a single T3K board. When constructing a `MeshDevice` with shape `(1, 8)`, TTNN assigns `(row=0, col=0)` to the first device in the provided device ID list, `(row=0, col=1)` to the second, and so on. The ordering of device IDs in the list you pass to the `MeshDevice` constructor therefore controls which physical chip occupies which logical coordinate — a detail that has direct consequences for which direction collective operations traverse the ring. Chapter 2 covers the constructor parameters and their effects in detail; for now, the key takeaway is that device IDs and logical coordinates are distinct concepts that are coupled only by the ordering you specify at initialization time.

The `cluster_axis` parameter used in several TTNN collective operations (introduced in Chapter 2) references the mesh's logical axes. For a `(1, 8)` mesh, `cluster_axis=1` selects the column axis — the axis along which the eight devices are arranged — and is the axis used by all ring collectives on a single T3K board.

---

## Neighbor Adjacency and On-Board Ethernet Wiring

The Wormhole Ethernet ports are wired on the T3K PCB to connect adjacent devices in a linear chain. The adjacency structure is:

```
Device 0 <--> Device 1 <--> Device 2 <--> Device 3 <--> Device 4 <--> Device 5 <--> Device 6 <--> Device 7
```

Each `<-->` represents one or more physical Ethernet links between those two chips. The exact number of link pairs between each adjacent device pair is discussed in `ethernet_link_bandwidth.md`; for topology purposes it is sufficient to note that only neighboring devices (devices whose IDs differ by 1) are directly connected by on-board traces. There are no diagonal or skip-hop PCB connections between non-adjacent chips (e.g., device 0 and device 2 are not directly wired).

This neighbor-only direct connectivity has a critical consequence: any data transfer between non-adjacent devices must be routed through intermediate chips. A message from device 0 to device 3, for example, must traverse either the path 0→1→2→3 (three hops to the right) or, if the linear chain is closed into a ring by inter-board connectors or by software-level loopback, the path 0→7→6→5→4→3 (five hops to the left). The practical meaning of "hops" in terms of latency and bandwidth is analyzed in `ethernet_link_bandwidth.md` and `topology_implications_for_collectives.md`.

### Diagram: Device Neighbor Adjacency

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Device  │   │ Device  │   │ Device  │   │ Device  │   │ Device  │   │ Device  │   │ Device  │   │ Device  │
│    0    │───│    1    │───│    2    │───│    3    │───│    4    │───│    5    │───│    6    │───│    7    │
│(row=0,  │   │(row=0,  │   │(row=0,  │   │(row=0,  │   │(row=0,  │   │(row=0,  │   │(row=0,  │   │(row=0,  │
│ col=0)  │   │ col=1)  │   │ col=2)  │   │ col=3)  │   │ col=4)  │   │ col=5)  │   │ col=6)  │   │ col=7)  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
                                            T3K PCB (single board)
```

Edges in this diagram represent direct on-board Ethernet link groups. There are no edges between non-adjacent devices; all other connectivity is multi-hop.

---

## Intra-Board vs. Inter-Board Links

The T3K board is designed for stand-alone single-board deployment but also supports connection to other T3K boards to form a larger multi-board cluster. This distinction matters because the link characteristics and topology assumptions differ between the two contexts.

### Intra-Board Links

The Ethernet links between the eight chips on a single T3K board are implemented as short, controlled-impedance PCB traces. Because the trace lengths are short (on the order of centimeters), signal integrity is excellent, and the links operate reliably at their rated speeds. Latency on intra-board links is determined primarily by serialization delay and MAC/PHY pipeline latency, not by propagation delay. In practice, intra-board Ethernet links on T3K Wormhole hardware achieve latencies in the low-microsecond range (see `ethernet_link_bandwidth.md` for figures).

The on-board links are always present and fully wired from the factory. You do not need to configure or negotiate them; TTNN's mesh initialization sequence detects and validates them automatically.

### Inter-Board Links

Each Wormhole chip on the T3K board has Ethernet ports beyond those used for intra-board connectivity. Some of these ports are routed to external connectors on the T3K board's edge, enabling chip-to-chip Ethernet connections between multiple T3K boards. When two or more T3K boards are connected via these external cables, TTNN can represent the combined set of devices as a single larger `MeshDevice` (for example, shape `(2, 8)` for two boards stacked in a `(row, col)` grid, or `(1, 16)` for two boards daisy-chained into a longer row). For a (1,16) daisy-chain, device IDs 0–7 occupy board 1 (columns 0–7) and devices 8–15 occupy board 2 (columns 8–15); device IDs 0–7 and 8–15 are assigned by the driver in enumeration order and represent a typical assignment — actual IDs depend on enumeration order and MeshDevice constructor argument ordering. Inter-board cabling connects device 7 on board 1 to device 8 on board 2. Coordinate assignment follows the same linear convention.

Inter-board links traverse a physical cable rather than a PCB trace, which introduces additional latency compared to intra-board links. The inter-board latency is still low by network standards, but it is measurably higher than intra-board latency. Inter-board latency is cable-length and connector-dependent. Exact figures require empirical measurement on the target setup.

| Metric | Estimated Value | Status |
|---|---|---|
| Intra-board single-hop latency | ~1.7–3 µs | `[placeholder — to be filled]` (figure based on estimates from `ethernet_link_bandwidth.md` pending empirical verification) |
| Inter-board latency (short cable) | 5–20 µs | `[placeholder — to be filled]` |

For this guide — which focuses on single-board T3K optimization — inter-board links are not used, and all bandwidth and latency figures refer to intra-board Ethernet links unless explicitly stated otherwise.

The distinction is important to keep in mind because some TTNN operations have `cluster_axis` semantics that become meaningful only in multi-board deployments (axis 0 for the inter-board row dimension vs. axis 1 for the intra-board column dimension). Single-board deployments use `cluster_axis=1` exclusively, which corresponds to the column axis along which devices 0–7 are arranged.

---

## Practical Implications

Three properties of this physical layout dominate all subsequent analysis in this guide:

**Linear topology with no shortcuts.** Because only adjacent devices are directly connected, the maximum shortest path between any two devices in a linear chain of eight is seven hops (device 0 to device 7). The average shortest-path length is 3.0 hops; ring algorithms require ⌈(N−1)/2⌉ = 4 communication rounds under bidirectional traversal. See `topology_implications_for_collectives.md` for the derivation and collective algorithm selection discussion.

**Symmetric bandwidth.** The Ethernet links between adjacent device pairs on T3K are symmetric: the same number of link pairs connect devices 0↔1 as connect devices 3↔4 or any other adjacent pair. This means there is no "bottleneck edge" in the mesh for uniform traffic patterns, and ring collectives that distribute load evenly across all links are bandwidth-optimal.

**Logical coordinate space matches physical adjacency.** Because device IDs increase monotonically from left to right and the `(1, 8)` logical mesh maps col=0 to col=7 in the same order, the logical coordinate distance between two devices equals their physical hop distance. A tensor shard on `(row=0, col=2)` is one hop away from `(row=0, col=3)` both logically and physically. This alignment simplifies reasoning about data locality and expert placement.

---

**Next:** [ethernet_link_bandwidth.md](./ethernet_link_bandwidth.md)
