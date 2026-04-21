## C Analysis — Pass 1

One fix applied from B pass 1:
- `prefill_vs_decode_comparison.md` image/video prefill formula: "2 × S × rotary_dim/2 × 2 bytes = 128 KiB" corrected to "S × (2*s_h + 2*s_w) × 2 tables × 2 bytes = 1024 × 42 × 2 × 2 bytes ≈ 168 KiB"; timing updated from 0.85 µs to 1.1 µs at 150 GB/s. Formula is now consistent with the per-section accounting in `operation_cost_breakdown.md`.

Verified all key claims across all five files after fix:
- Total additional dispatches: 5 (3 embedding + 2 concat); total ~7 dispatches vs ~2 for standard RoPE — correct
- Dispatch overhead estimate: 25–50 µs per decode step — correct
- Decode step latency P150: ~250 ms; overhead fraction < 0.02% — correct
- Standard RoPE data: D × 2 × 2 bytes = 256 bytes per position (cos+sin, D=64) — correct
- M-RoPE total data: same 256 bytes per position (sections sum to rotary_dim = 64 × 2 tables × 2 bytes = 256) — correct
- cos/sin table size: 32768 × 64 × 2 bytes × 2 tables = 8 MiB — correct
- mrope_section = [11, 11, 10] sections not tile-aligned (22, 22, 20 real dims < 32 tile size) → fusion deferred — correct
- T3K consideration: TP reduces per-chip multiply work but dispatch overhead is constant — correct
- Image prefill worst-case overhead ~1.1 µs at 150 GB/s — correct after fix; conclusion (< 0.1% of prefill step) unaffected
- Comparison table last row: ~1–5 µs range consistent with 1.1 µs estimate — correct

No remaining crucial inaccuracies found.

**Crucial updates: yes** (formula fix above applied)

---

## C Analysis — Pass 2

All crucial inaccuracies from Pass 1 resolved. Verified:
- h+w random-access formula: 168 KiB at S=1024 — correct and consistent with operation_cost_breakdown.md section accounting
- Timing 1.1 µs < 2 µs conclusion unaffected — correct
- All `[placeholder]` entries for hardware measurements present — correct
- No further crucial inaccuracies found across all five files

**Crucial updates: no**
