# Agent B Review — Chapter 2 — Pass 2

## Item 1 — Factual error: "~200 GB/s" aggregate Ethernet bandwidth (`num_links_tuning.md`, line 23)

The file states: "The T3K aggregate Ethernet CCL bandwidth across all 8 chips is ~200 GB/s."

The actual aggregate raw Ethernet bandwidth is 8 chips × 16 ports × 12.5 GB/s/port = **1,600 GB/s**. The 200 GB/s figure only holds under a very narrow interpretation: 8 chips × 1 link × 12.5 GB/s × 2 directions = 200 GB/s, which describes the bandwidth consumed by a single `num_links=1` CCL operation, not the hardware's aggregate capacity. Presenting this as "aggregate Ethernet CCL bandwidth" is misleading and numerically wrong by 8×. A reader comparing T3K network specs against this number will get the wrong impression of the chip's connectivity. The sentence should either state the true aggregate (1,600 GB/s raw) or be reframed as "a single `num_links=1` CCL operation on T3K uses ~200 GB/s of the available bidirectional Ethernet capacity."

## Item 2 — Arithmetic inconsistency: stated speedup range vs. table values (`latency_savings_analysis.md`, line 124)

The table immediately above line 124 shows:

- Unfused total: 29–78 µs
- Fused total: 13–43 µs

The inline speedup claim is "~1.6–1.8×." Dividing the table bounds:

- Lower-bound speedup: 29 / 13 ≈ **2.2×**
- Upper-bound speedup: 78 / 43 ≈ **1.8×**
- Mid-point speedup: 53.5 / 28 ≈ **1.9×**

The stated range "~1.6–1.8×" is inconsistent with the table's own numbers; the range derived from those bounds is approximately **1.8–2.2×**. A reader who checks the arithmetic will find the summary contradicts the data in the same table. The speedup figure should be recalculated from the table bounds, or the table bounds should be adjusted to match the stated speedup.

---

All other key arithmetic verified correct: fused weight shape (4096, 3072), per-chip shard (4096, 384), tile counts (1024+256+256=1536 unfused = 1536 fused), arithmetic intensity 1.0 FLOPs/byte, crossover batch ≈ 49, and payload sizes are all internally consistent.

# Agent B Review — Chapter 2 — Pass 3

## Item 1 — Factual error: "1/16 of each chip's available Ethernet port budget" (`num_links_tuning.md`, line 23)

The sentence reads: "A single CCL operation thus uses only 1/16 of each chip's available Ethernet port budget."

This fraction is wrong. With `num_links=1`, the all-reduce uses **one Ethernet port per direction** on each chip (one for clockwise, one for counterclockwise), totalling **2 ports per chip** out of 16 available. The correct fraction is **2/16 = 1/8**, not 1/16. A reader who divides 16 by the stated fraction to infer absolute port usage will conclude only 1 port is consumed rather than 2, which misrepresents the physical cost and can cause incorrect reasoning when computing concurrent CCL port budgets in the "pipelined with other CCL operations" scenario described later in the same file.

---

All other arithmetic verified: pass-2-corrected bandwidth split (1,600 GB/s total / ~200 GB/s per-CCL-op), pass-2-corrected speedup range (1.8–2.2×), reduce-scatter payload (768 B), all-gather payload (672 B), total per-chip bidirectional traffic (~1.44 KB), crossover batch ≈ 49, H=8192 payload (≈3 KB), and per-link bandwidth figures are all internally consistent.

# Agent B Review — Chapter 2 — Pass 4

No feedback — chapter approved.

# Agent B Review — Chapter 2 — Pass 5

No feedback — chapter approved.
