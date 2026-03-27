# Agent B Review — Chapter 3 — Pass 1

## Item 1 — Factual error: PCIe Gen4 per-lane transfer rate (host_transfer_overhead.md, line 16)

The bandwidth derivation reads:

```
Peak bandwidth (one direction) = 16 lanes × 2 GT/s × 128b/130b encoding
                               ≈ 31.5 GB/s (theoretical)
```

The per-lane rate for PCIe Gen4 is **16 GT/s**, not 2 GT/s (that is PCIe Gen1). The formula as written evaluates to ≈ 3.15 GB/s, not 31.5 GB/s — a factor-of-8 error. The stated result of 31.5 GB/s is correct for PCIe Gen4 x16, but it is inconsistent with the intermediate value shown. A reader working through the arithmetic will reach the wrong number and lose confidence in the model.

Fix: change `2 GT/s` to `16 GT/s`.

---

## Item 2 — Factual inconsistency: device→host volume tagged [ESTIMATE] (roundtrip_mechanics.md, lines 64–67)

The table entry for device→host total bytes is marked `[ESTIMATE]`:

```
Total bytes device→host: 8 × 6,144 = 49,152 bytes = 48.0 KB  [ESTIMATE]
```

This is deterministic arithmetic, not an estimate. The per-chip shape `(1, 1, 1, 3072)` × 2 bytes = 6,144 B is exact; multiplying by 8 chips is exact. Labelling it `[ESTIMATE]` is misleading because `[ESTIMATE]` is used throughout the chapter to flag values that depend on hardware behaviour or timing models. Readers relying on the `[ESTIMATE]` tag to distinguish measured from inferred facts will misclassify this figure.

The same applies to the host→device row (line 109) and the round-trip total row (line 136), though the round-trip total row is more defensible since tile-padding assumptions are involved.

Fix: remove `[ESTIMATE]` from the device→host and host→device byte-count rows. Retain it only on the round-trip total if tile-padding uncertainty is intended to be flagged.

---

## Item 3 — Critical coherence gap: ConcatMeshToTensor transfers 48 KB but uses only 6 KB (roundtrip_mechanics.md, lines 51–56)

The file correctly notes that after the all-reduce each chip holds the full `(1, 1, 1, 3072)` tensor, so `ConcatMeshToTensor(dim=3)` concatenates 8 identical copies into a `(1, 1, 1, 24576)` host tensor and a slice `[..., :3072]` recovers the intended result. The Data Volume Summary then counts 48 KB device→host (8 × 6 KB).

This creates an unresolved coherence problem: the chapter's stated motivation is that the payload is only 6 KB, yet the actual `ConcatMeshToTensor` call reads **48 KB from device to host** to produce 6 KB of useful data — a 8× amplification. Neither this file nor `host_transfer_overhead.md` calls this out as a separate inefficiency, and `host_transfer_overhead.md` uses 6 KB per chip (not 6 KB total) as the per-direction figure in the latency model without explaining the discrepancy. A reader trying to reconcile "6 KB payload" (index.md, line 7) with the 48 KB device→host figure in the summary table will be confused.

Fix: add a sentence in the Data Volume Summary explicitly acknowledging that the physical PCIe read is 48 KB (not 6 KB) due to pulling all 8 chip copies, and that this amplification is a secondary inefficiency of `ConcatMeshToTensor` on a post-all-reduce tensor.

---

## Item 4 — Factual error: Alternative 1 description presents a non-functional substitution without upfront correction (device_side_alternatives.md, lines 25–45)

The Alternative 1 section opens with a concept and implementation sketch for replacing `all_reduce` with `all_gather` on the WIDTH_SHARDED partial matmul output, framed as a viable approach. The correction that this is numerically wrong (all_gather concatenates; all_reduce sums) does not appear until lines 60–62, after a full code snippet and three paragraphs of analysis.

A reader scanning the implementation sketch (lines 34–45) and skipping the feasibility analysis will take away incorrect guidance: that `ttnn.all_gather` on the partial WIDTH_SHARDED outputs is a drop-in replacement for `ttnn.all_reduce`. This is a factual error that produces wrong code if acted on.

Fix: add a one-line warning at the start of the Alternative 1 concept section (before the implementation sketch) stating that all_gather concatenates rather than sums and is therefore not a substitute for all_reduce in the column-parallel pattern. The detailed explanation can remain in the feasibility section.

---

## Item 5 — Missing nav footer on index.md

`index.md` has no footer link to the first file in the reading order (`roundtrip_mechanics.md`). Every other file in the chapter has a `**Next:**` footer. The index is the entry point for a reader following the chapter sequentially; without a footer, the reading order table and the clickable links in the body are the only navigation aids, and a reader who reaches the bottom of the index has no prompt to continue.

This is a minor structural gap but is consistent with the footer convention established by the other three files.

Fix: add a footer to `index.md`:

```
---

**Start reading:** [Round-Trip Mechanics](./roundtrip_mechanics.md)
```

# Agent B Review — Chapter 3 — Pass 2

## Item 1 — Factual error: bandwidth formula produces wrong theoretical peak (host_transfer_overhead.md, line 18)

The formula reads:

```
Peak bandwidth (one direction) = 16 lanes × 16 GT/s per lane × (1 byte / transfer) × 128b/130b encoding
                               = 16 × 16 × 1 × (128/130)
                               ≈ 252 GB/s (theoretical raw)
```

PCIe GT/s is a serial symbol rate (Giga-Transfers per second on a 1-bit-wide lane). To convert to bytes per second, the bit rate must be divided by 8. The correct derivation is:

```
Per-lane data rate = 16 GT/s × (128/130) / 8 ≈ 1.97 GB/s
x16 theoretical peak = 16 × 1.97 ≈ 31.5 GB/s
```

The formula as written evaluates to 252 GB/s — roughly 8× too high — because it treats GT/s directly as GB/s (omitting the ÷8 bit-to-byte conversion). The stated practical figure of 20–25 GB/s is consistent with a ~31.5 GB/s theoretical peak, not with a 252 GB/s peak. A reader working through the formula reaches the wrong theoretical value and will distrust the practical estimate that follows.

Fix: correct the formula to include the ÷8 conversion (or express the per-lane rate as Gbps and convert explicitly), and update the stated theoretical result from "≈ 252 GB/s" to "≈ 31.5 GB/s".

---

## Item 2 — Factual arithmetic error: median percentage calculation is wrong (host_transfer_overhead.md, line 119)

The text states:

> "At the median of both ranges (~28 µs for QKV, ~16 µs for round-trip), the round-trip accounts for approximately **36%** of QKV projection time"

The medians are computed correctly (median of 13–43 µs = 28 µs; median of 6.8–25.2 µs = 16 µs). However, 16 / 28 = 0.571 = **57%**, not 36%. The 36% figure is arithmetically inconsistent with the two numbers that precede it in the same sentence. A reader who checks the arithmetic will find the stated conclusion unsupported by the figures given.

Fix: replace "approximately 36%" with "approximately 57%".

# Agent B Review — Chapter 3 — Pass 3

## Item 1 — Factual error: tile-padding claim is wrong for a 1-row tensor (roundtrip_mechanics.md, lines 117–118)

The Step 3 analysis states:

> "For a `(1, 1, 1, 3072)` tensor with `head_dim=128`, the tile-aligned size is at most 1 tile row × ceil(3072/32) = 96 tile columns, which is already tile-aligned. The padded byte count is the same as the unpadded count in this case."

This is incorrect. TTNN TILE_LAYOUT uses 32×32-element tiles. The tensor has **1 row** along dim 2 (the row dimension), which must be padded to 32 rows to fill a tile. The tile-padded shape is 32 rows × 3072 columns × 2 bytes (BF16) = **192,000 bytes ≈ 188 KB per chip**, not 6,144 bytes. The column dimension (3072) is tile-aligned, but the row dimension (1) is not. The claim that "the padded byte count is the same as the unpadded count" is factually wrong by a factor of 32, and directly contradicts the Data Volume Summary table in the same file, which states 6.0 KB per chip for the host→device direction.

The error affects Step 3's transfer volume claim and, if the tile layout padding is real, the 48 KB aggregate host→device figure in the Data Volume Summary table would also be wrong (it should be ~1.5 MB). Either the tile-padding note should be removed (if `from_torch` pads lazily or the tensor layout differs in practice), or the byte counts throughout the chapter must be revised to reflect the padded size.

Fix: remove or correct the tile-padding sentence. If TTNN's `from_torch` with `TILE_LAYOUT` does pad the row dimension to 32, the per-chip host→device transfer size is ~188 KB, not 6 KB, and the Data Volume Summary table must be updated accordingly. If padding is deferred until the first kernel access (not during the PCIe transfer), state that explicitly rather than asserting the padded count equals the unpadded count.

# Agent B Review — Chapter 3 — Pass 4

## Item 1 — Critical coherence gap: `host_transfer_overhead.md` latency model uses 6 KB without stating the ROW_MAJOR layout assumption, while code in the same file uses `TILE_LAYOUT`

`host_transfer_overhead.md` builds its entire quantitative latency model — the per-component cost table (lines 56–65), the directional estimates, and the total round-trip range of 6.8–25.2 µs — on a per-chip transfer size of 6 KB (ROW_MAJOR_LAYOUT, no tile-padding). This assumption is never stated in `host_transfer_overhead.md`. It appears only in `roundtrip_mechanics.md` (line 129).

At the same time, the measurement code provided in `host_transfer_overhead.md` (lines 143–152 and lines 196–206) explicitly passes `layout=ttnn.TILE_LAYOUT` to `ttnn.from_torch`. Under TILE_LAYOUT the per-chip host→device transfer size is ~192 KB (as established in `roundtrip_mechanics.md`), not 6 KB — a factor-of-32 difference.

A reader working through `host_transfer_overhead.md` in isolation will observe:

1. The latency model states 6 KB per chip and derives 0.3 µs payload transfer time at 20 GB/s.
2. The measurement code they are asked to run uses `TILE_LAYOUT`, which transfers ~192 KB per chip and takes ~9.6 µs at the same bandwidth.

These two figures are inconsistent without the cross-file layout assumption, and the reader has no way to reconcile them from within `host_transfer_overhead.md` alone. The latency model is unreliable without knowing which layout assumption applies.

Fix: add a single explicit sentence near the top of the quantitative model section in `host_transfer_overhead.md` stating: "All transfer-size and latency figures in this section assume **ROW_MAJOR_LAYOUT** at transfer time (6 KB per chip for the `(1, 1, 1, 3072)` BF16 tensor). If the tensor is in TILE_LAYOUT, substitute ~192 KB per chip; see the layout assumption note in `roundtrip_mechanics.md`, Step 3." This makes the file self-contained for a reader who does not read both files in sequence.

# Agent B Review — Chapter 3 — Pass 5

## Item 1 — Factual error: host DRAM bandwidth characterised as "dual-channel DDR5" is wrong for a T3K host platform (`host_transfer_overhead.md`, line 48)

The text states:

> "At 20–25 GB/s practical throughput per chip, the aggregate host DRAM write rate during device→host would be up to 200 GB/s — exceeding typical x86 host DRAM bandwidth (typically 50–100 GB/s for dual-channel DDR5 [ESTIMATE])."

T3K is a server-class system. The host machines that house T3K deployments use multi-channel DDR5 configurations (quad-channel at minimum; workstation and server platforms commonly use 8–12 channels). Quad-channel DDR5-4800 delivers approximately 150 GB/s; 8-channel configurations exceed 300 GB/s. The "dual-channel DDR5 (50–100 GB/s)" figure significantly understates the actual host DRAM bandwidth and causes the text's conclusion — that 200 GB/s aggregate PCIe traffic would exceed host DRAM capacity — to be misstated. On a quad-channel or higher host, 200 GB/s is at or below DRAM bandwidth, materially changing the analysis of whether DRAM contention is a bottleneck.

Fix: replace "dual-channel DDR5" with a more accurate description of T3K host memory configurations (e.g., "quad-channel to 8-channel DDR5, typically 150–400 GB/s depending on platform"), and revise the DRAM bandwidth estimate accordingly. The 50–100 GB/s range is accurate only for consumer dual-channel configurations that would not normally host a T3K.

---

## Item 2 — Critical coherence gap: batch-scaling table carries no layout assumption caveat (`host_transfer_overhead.md`, lines 230–238)

The batch-scaling table presents per-chip byte sizes and latency estimates for batch sizes 1 through 512:

```
| Batch size | Per-chip bytes | Estimated t_RT | Bandwidth-limited? |
| 1          | 6.1 KB         | ...            |                    |
...
```

All per-chip byte sizes in the table are derived from the ROW_MAJOR_LAYOUT assumption (B × 6,144 bytes). If TILE_LAYOUT is in use, the per-chip host→device transfer size is ~192 KB at batch=1 — 32× larger — and the entire table is wrong by that factor for the host→device direction. The ROW_MAJOR layout assumption note appears at line 56 (top of the quantitative model section) but is not repeated at or near the table. A reader navigating directly to the table, or re-reading it after the 57% median comparison, has no in-context signal that the table values are conditional on layout. Given that the measurement code in the same file explicitly passes `layout=ttnn.TILE_LAYOUT`, a reader is likely to measure TILE_LAYOUT behaviour and find the table figures off by 32× at all batch sizes.

Fix: add a table caption or header note immediately above the batch-scaling table stating that all per-chip byte sizes assume ROW_MAJOR_LAYOUT; if TILE_LAYOUT is used, multiply per-chip bytes by 32 for the host→device direction and recalculate the latency estimates accordingly.

# Agent B Review — Chapter 3 — Pass 6

## Item 1 — Critical coherence error: per-chip sub-tensor size is contradicted within `roundtrip_mechanics.md` and corrupts the device→host volume calculation (lines 24, 32, 53, 55)

Line 24 characterises the post-all-reduce distribution as a **"column-sharded output distribution"** and line 32 states explicitly that a sharded tensor gives `(1, 1, 1, 384)` per chip. These two statements together imply each chip holds a 384-element shard after the all-reduce.

Yet line 53 states the opposite:

> "In practice, the TTNN output memory config after the all-reduce leaves each chip's sub-tensor at `(1, 1, 1, 3072)` (the full logical size)"

The data volume calculation immediately below (lines 60–67) is built entirely on the 3072-per-chip interpretation, yielding 8 × 6,144 B = 48 KB device→host. Under the 384-per-chip (column-sharded) interpretation the total would be 8 × 768 B = 6 KB device→host — an 8× difference.

Line 55 hedges with "depends on the `output_memory_config` of the all-reduce, which determines whether each chip's local sub-tensor is `(1, 1, 1, 3072)` or `(1, 1, 1, 384)`" but does not resolve which case applies in the Ling production path. The data volume section then commits definitively to 48 KB without stating it is conditional on the 3072-per-chip interpretation.

This is a factual coherence error: the chapter simultaneously asserts the distribution is column-sharded (384 per chip) and that the per-chip sub-tensor is 3072. Only one can be true. The device→host byte count — a figure cited in both the Data Volume Summary table and the latency model in `host_transfer_overhead.md` — is wrong under one of the two interpretations.

Fix: determine which interpretation reflects the actual Ling all-reduce output memory config. If each chip holds the full `(1, 1, 1, 3072)` (e.g., due to an all-reduce that does not column-shard its output), remove the "column-sharded output distribution" characterisation at line 24 and the `(1, 1, 1, 384)` per-chip shard claim at line 32. If each chip holds `(1, 1, 1, 384)`, correct the `ConcatMeshToTensor` analysis and the device→host volume to 6 KB total (not 48 KB), and update the latency model accordingly.

---

## Item 2 — Unsubstantiated claim: "tile-padding does not apply to the device-side read path" in the Data Volume Summary table (roundtrip_mechanics.md, line 153)

The Data Volume Summary table note for the device→host direction states:

> "tile-padding does not apply to the device-side read path here"

No justification is given. The tensor stored on-device after the all-reduce is in whatever layout the all-reduce output memory config specifies. If the on-device layout is TILE_LAYOUT (the standard for tensors that will be consumed by Tensix kernels), the per-chip device-side read would be ~192 KB per chip (the same tile-padded size derived in Step 3 for the host→device direction), not 6 KB. The asymmetry between the two directions (192 KB host→device vs. 6 KB device→host) is not self-evidently correct, and stating it as fact without citing the all-reduce output layout specification is unsubstantiated.

If the all-reduce output is stored in ROW_MAJOR_LAYOUT on-device (which is unusual for post-compute tensors in TTNN but would justify the 6 KB figure), that should be stated explicitly. If it is stored in TILE_LAYOUT, the device→host direction also transfers ~192 KB per chip and the 6 KB claim is wrong.

Fix: either cite the specific all-reduce output memory config (and its layout type) that produces a ROW_MAJOR on-device tensor, or revise the device→host per-chip figure to ~192 KB and update the table and `host_transfer_overhead.md` latency model accordingly. The current note is a bare assertion that contradicts the tile-padding analysis applied to the host→device direction.

# Agent B Review — Chapter 3 — Pass 7

No feedback — chapter approved.

# Agent B Review — Chapter 3 — Pass 8

No feedback — chapter approved.
