# NoC and DRAM Access Model

## NoC Packet Model

Every DRAM read issued by a Tensix core is a NoC packet. The packet header contains:

- **Destination X/Y coordinate** — the NoC grid column and row of the target DRAM controller node.
- **Address** — byte offset within the DRAM bank attached to that controller.
- **Size** — number of bytes to read (must be aligned to the NoC flit boundary; in practice a full BF16 tile is 2,048 bytes).

The NoC routes packets hop-by-hop through the mesh. Each hop traverses one NoC link between adjacent nodes. Latency scales linearly with hop count under low-contention conditions. Under contention, queuing at intermediate nodes adds variable latency on top of the base hop cost.

A Tensix core does not have a cache. Every tile read is a DMA request that must complete before the compute engine can consume the data, unless double-buffering is used to pipeline the transfer with a prior computation.

---

## DRAM Controller Topology on Wormhole B0

Wormhole B0 has **6 DRAM controllers**, not 8. Each controller occupies a fixed column in the 10×12 NoC grid. Each controller is attached to **2 GDDR6 banks**, giving 12 total banks across the chip.

```
NoC column index:  0    1    2    3    4    5    6    7    8    9
DRAM controller:   D0   --   D1   --   D2   --   D3   --   D4   D5
```

The exact column positions are fixed by the Wormhole B0 physical floorplan. Cores in the 8×10 Tensix array occupy a contiguous sub-region of the NoC grid. The minimum hop count from any Tensix core to each DRAM column therefore varies by 1–6 hops depending on the core's X-position and the controller column.

---

## Interleaved Access Pattern

Under `TensorMemoryLayout.INTERLEAVED`, TTNN distributes a tensor's tiles across all available DRAM banks in round-robin order. Tile 0 goes to bank 0, tile 1 to bank 1, ..., tile 11 to bank 11, tile 12 back to bank 0, and so on.

When core (0,0) fetches a weight tile at position `(row, col)` in a large weight matrix, the tile's bank assignment is determined by its linear tile index. For a weight matrix with many tiles, consecutive tiles in the K-dimension land on consecutive banks in round-robin order. A single core fetching a row of weight tiles will issue requests to D0, D1, D2, D3, D4, D5, D0, D1, ... in sequence.

Each of those requests traverses a different NoC path:
- Request to D0 (nearest column): 1 hop.
- Request to D5 (farthest column): up to 6 hops.

Variable latency per tile means the compute engine stalls unevenly between tiles.

---

## NoC Hotspots Under Full-Grid Matmul

During a full-grid matmul (all 80 Tensix cores active), every core simultaneously fetches weight tiles from interleaved DRAM. Because tile assignment is purely by index, many cores target the same DRAM column at the same time.

Specifically, all cores that happen to be fetching tiles with index `i % 6 == k` will simultaneously send requests to controller Dk. The 6 DRAM columns receive non-uniform request rates because the 80-core grid does not divide evenly by 6. Controllers that serve 14 cores simultaneously instead of 13 see 8% higher load; under bursty access this creates link saturation on the 1–2 NoC links leading to those controllers.

The result is:
- NoC link utilization near DRAM columns peaks above the sustainable rate.
- Packet queuing at intermediate NoC routers adds 10–50 cycle latency per request.
- Some cores stall longer than others, creating load imbalance in the systolic pipeline.

---

## Measured Effect: Bandwidth Reduction

Empirical measurements on Wormhole B0 show that interleaved DRAM access under full-grid matmul reduces effective DRAM bandwidth by **20–40%** relative to the theoretical peak of ~300 GB/s.

At the lower end (20% reduction), effective bandwidth is ~240 GB/s — achievable when the weight matrix is large enough that DRAM latency is mostly hidden by the depth of the tile pipeline. At the upper end (40% reduction), effective bandwidth falls to ~180 GB/s — observed when the K-dimension is small (decode regime, M=1) and each core issues few enough requests that NoC stalls dominate.

The interleaved layout provides no locality guarantee: a core cannot predict which NoC path its next tile request will take, so prefetching is speculative. DRAM-sharded layout eliminates this uncertainty entirely.

---

## Tile Size and Request Granularity

A BF16 tile in TILE_LAYOUT is 32×32 elements × 2 bytes = **2,048 bytes**. This is the atomic unit of every DRAM read request from a Tensix core. The NoC flit size on Wormhole B0 is 32 bytes; a 2,048-byte tile read requires 64 flits. All 64 flits must traverse the same NoC path; there is no per-flit routing.

A core processing a weight shard of shape `[32, 1792]` (56 tiles wide) will issue 56 read requests per row of activations, each 2,048 bytes, each on whatever NoC path the tile's bank assignment dictates. Under interleaved layout, those 56 requests are distributed across all 6 DRAM columns: 9 requests to each of 4 controllers and 10 requests to each of the remaining 2. Under sharded layout, all 56 requests go to the same DRAM controller: 56 sequential requests to one bank, which the GDDR6 controller services as a streaming burst with minimal row-activation overhead.

---

## Summary

| Property | Interleaved | DRAM-Sharded |
|---|---|---|
| Tile-to-bank assignment | Round-robin by tile index | Contiguous range per shard |
| NoC path per core | Variable (1–6 hops) | Fixed (1–2 hops typical) |
| Hotspot risk | High under full-grid matmul | None |
| Bandwidth efficiency | 60–80% of peak | 87–97% of peak |
| Prefetch predictability | Low | High |
