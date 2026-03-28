# B Review — Chapter 5 — Pass 1

---

## Item 1 — Wrong formula for `k̃, q̃, v` byte count (implementation error if d_k ≠ d_v)

**File:** `roofline_decode_and_prefill.md`, Section 1.2, bytes table.

The expression given is `3 × d_k × 2 bytes` for the combined `k̃, q̃, v` traffic. This is only correct because `d_k = d_v = 128` in this specific model. The vector `v` has dimension `d_v`, not `d_k`. The correct general expression is `(2 × d_k + d_v) × 2 bytes`. Any reader who applies this formula to a model with `d_k ≠ d_v` will compute the wrong byte count, and consequently the wrong arithmetic intensity.

---

## Item 2 — Decode time uses 2.0 MB instead of the derived 2.02 MB, producing an inconsistent result

**File:** `roofline_decode_and_prefill.md`, Section 1.4.

The bandwidth figure derived in Section 1.2 is 2,121,728 bytes per layer ≈ 2.02 MB, displayed as "2.02 MB" in the scaling block. However, the time calculation in Section 1.4 produces 6.9 µs, which corresponds to exactly 2.0 MB / 288 GB/s = 6.944 µs. Using the stated 2.02 MB gives 2.02 × 10^6 / 288 × 10^9 = 7.01 µs, and using the exact byte count (2,121,728 bytes) gives 7.37 µs. The 6.9 µs figure is inconsistent with the 2.02 MB figure shown one paragraph earlier. The downstream speedup comparison (254×) inherits this error: with the correct per-layer state bandwidth, the speedup is approximately 1,778 µs / 7.37 µs ≈ 241×, not 254×.

---

## Item 3 — Per-head prefill FLOP scaling contains an arithmetic error

**File:** `roofline_decode_and_prefill.md`, Section 3.1.

The per-head per-chunk FLOP total is correctly derived as 4,194,304. Scaling to 128 chunks should give:

```
128 × 4,194,304 = 536,870,912 FLOP
```

The file states 537,001,984 FLOP — an overcount of 131,072 FLOP with no source. The per-layer figure (17,203,863,552) is derived from this inflated per-head number and is likewise wrong. The correctly rounded values are ≈ 536.9 MFLOP per head and ≈ 17.18 GFLOP per layer. The arithmetic intensity calculation (33.6 FLOP/byte) is not materially affected because the per-layer FLOP and byte figures are both approximated, but a reader who reproduces the per-head calculation from scratch will not be able to reconcile their result with the stated 537,001,984.

---

## Item 4 — "260× below the ridge point" in prose contradicts "262×" in the summary table

**File:** `roofline_decode_and_prefill.md`, Sections 1.3 and 4 (summary table).

Section 1.3 states the decode step sits "more than **260×** below the ridge point." The summary table in Section 4 states "Memory-bound (**262×** below ridge)." The correct calculation is 455 / 1.74 = 261.5, which rounds to 262. The prose figure of "260×" is therefore understated and inconsistent with the table. A reader using the prose figure to estimate FPU utilization will get a slightly wrong answer (≈ 0.38% utilization vs the correct ≈ 0.38% — negligible in practice, but the textual inconsistency could cause confusion when cross-referencing).

---

## Item 5 — `index.md` is missing a navigation footer; `wormhole_hardware_specs.md` has no "Previous" link

**Files:** `index.md`, `wormhole_hardware_specs.md`.

`wormhole_hardware_specs.md` links forward to `roofline_decode_and_prefill.md` but has no "Previous" link back to `index.md`. A reader who lands directly on `wormhole_hardware_specs.md` has no navigation path back to the chapter index. `index.md` itself carries no navigation footer at all (no "Previous chapter" link). While this is a navigation issue rather than a numerical one, it is materially misleading in the sense that a reader following the documented navigation structure cannot traverse backward through the chapter, and may believe they are reading a self-contained document rather than part of a multi-chapter guide.

---

## Agent A Change Log — Pass 1

**File edited:** `roofline_decode_and_prefill.md`

**Item 1 applied — Section 1.2 bytes table, k̃/q̃/v formula:**
- Old: `3 × d_k × 2 bytes` → `3 × 128 × 2 = 768`
- New: `(d_k + d_k + d_v) × 2 bytes` → `(128 + 128 + 128) × 2 = 768`
- Numeric result unchanged (d_k = d_v = 128 for this model), but the formula is now correct for the general case where d_v may differ from d_k.

**Item 2 applied — Sections 1.4 and 2, decode time and speedup ratio:**
- Section 1.4 state bandwidth: `2.02 MB` → `2,121,728 bytes` (exact figure already derived in Section 1.2)
- Section 1.4 per-layer time: `2.02 × 10^6 / 288 × 10^9 = 6.9 µs` → `2,121,728 / 288 × 10^9 = 7.36 µs`
- Section 1.4 total for 30 layers: `30 × 6.9 µs = 207 µs` → `30 × 7.36 µs = 220.8 µs ≈ 221 µs`
- Section 2 comparison block: DeltaNet per-layer value `≈ 7 µs` → `≈ 7.36 µs`
- Section 2 speedup: `1,778 / 7 ≈ 254×` → `1,778 / 7.36 ≈ 242×`
- Section 2 prose: "approximately 254×" → "approximately 242×"

**Item 3 applied — Section 3.1 prefill per-head FLOP scaling:**
- Per-head FLOP total: `537,001,984` → `536,870,912` (correct: 128 × 4,194,304 = 536,870,912)
- Per-head rounded label: `≈ 537.6 MFLOP` → `≈ 536.9 MFLOP`
- Per-layer exact figure: `17,203,863,552` → `17,179,869,184` (correct: 32 × 536,870,912)
- Per-layer rounded label `≈ 17.2 GFLOP` unchanged (still rounds correctly)

**Item 4 applied — Section 1.3 prose "below ridge point" multiplier:**
- Old: "more than **260× below the ridge point**"
- New: "more than **262× below the ridge point**"
- Arithmetic: 455 / 1.74 = 261.5 → rounds to 262; now consistent with the summary table in Section 4.

---

# B Review — Chapter 5 — Pass 2

Two numerical errors remain in the current text of `roofline_decode_and_prefill.md`. Both stem from the same root cause: the 536,870,912-byte figure produced by the KV-cache and prefill byte-count formulas is labeled "512 MB" and then treated as 512 × 10^6 bytes in subsequent calculations. 512 MiB = 536,870,912 bytes ≠ 512 MB = 512,000,000 bytes. Using the wrong denominator produces wrong arithmetic intensities and wrong latency figures.

---

## Item 1 — Prefill arithmetic intensity is stated as 33.6 FLOP/byte; the correct value is 32.0 FLOP/byte

**File:** `roofline_decode_and_prefill.md`, Section 3.3.

The exact per-layer byte count for the prefill pass, derived correctly in Section 3.2, is:

```
32 heads × 16,777,216 bytes/head = 536,870,912 bytes
```

However, the file labels this as "512 MB" and then computes the arithmetic intensity as:

```
17,200 / 512 ≈ 33.6 FLOP/byte
```

where the denominator 512 is in units of "MB" treated as 10^6 bytes. The correct division is:

```
17,179,869,184 FLOP / 536,870,912 bytes = 32.0 FLOP/byte  (exact)
```

A downstream reader who reproduces this division from the byte table in Section 3.2 will obtain 32.0, not 33.6 — a 5% error. The summary table in Section 4 inherits this wrong value ("33.6 FLOP/byte") and consequently states the prefill intensity is "13.5× below ridge" when the correct ratio is 455 / 32.0 = 14.2×. Any kernel designer using 33.6 as the baseline for evaluating optimization headroom will get a materially incorrect picture of how far the prefill pass sits from the compute-bound threshold.

---

## Item 2 — Gated Attention per-layer latency is stated as 1,778 µs; the correct value is ~1,864 µs, making the speedup ratio wrong

**File:** `roofline_decode_and_prefill.md`, Section 2.

The KV-cache byte count is derived correctly:

```
2 × 262,144 × 256 × 2 × 2 = 536,870,912 bytes
```

The file immediately labels the result "= 512 MB" and uses this in the latency calculation:

```
512 × 10^6 / 288 × 10^9 ≈ 1.78 ms = 1,778 µs
```

Because 512 MB (SI) ≠ 536,870,912 bytes, the latency is wrong. The correct calculation is:

```
536,870,912 / 288 × 10^9 = 1,864 µs
```

This propagates directly into the speedup comparison. The per-layer DeltaNet decode time is correctly stated as 7.36 µs (after the Agent A Pass 1 fix), so the correct speedup is:

```
1,864 / 7.36 ≈ 253×
```

The file states 242×. A reader implementing a latency model or validating the speedup claim against their own measurements will compute 253× from first principles and will not be able to reconcile that with the 242× written in the text.

---

## Agent A Change Log — Pass 2

**File edited:** `roofline_decode_and_prefill.md`

**Item 1 applied — Section 3.3 prefill arithmetic intensity:**
- Old: `17.2 × 10^9 / 512 × 10^6 = 17,200 / 512 ≈ 33.6 FLOP/byte`
- New: `17,179,869,184 / 536,870,912 = 32.0 FLOP/byte (exact)`
- Bold header updated: `Measured intensity: 33.6 FLOP/byte` → `Measured intensity: 32.0 FLOP/byte`
- Prose comparison updated: `33.6 vs 1.74 FLOP/byte` → `32.0 vs 1.74 FLOP/byte`

**Item 2 applied — Section 2 Gated Attention latency and speedup:**
- Old: `512 × 10^6 / 288 × 10^9 ≈ 1.78 ms = 1,778 µs`
- New: `536,870,912 / 288 × 10^9 ≈ 1.86 ms = 1,864 µs`
- Comparison block: `1,778 µs` → `1,864 µs`, speedup `242×` → `253×`
- Prose sentence: `approximately 242×` → `approximately 253×`

**Summary table (Section 4) updated:**
- Prefill row: `33.6 FLOP/byte` → `32.0 FLOP/byte`, `13.5× below ridge` → `14.2× below ridge`

---

# B Review — Chapter 5 — Pass 3

Pass 2 items verified:
- Item 1: Section 3.3 now divides 17,179,869,184 / 536,870,912 = 32.0 FLOP/byte (exact). The bold header and prose comparison are updated. The summary table reads 32.0 FLOP/byte and 14.2× below ridge. Both correct.
- Item 2: Section 2 time calculation now divides 536,870,912 / 288×10^9 = 1,864 µs. Comparison block shows 1,864 µs and speedup 253×. Prose sentence reads "approximately 253×". All correct.

No new issues found. All numbers in `roofline_decode_and_prefill.md` are internally consistent and reproducible from first principles.

**No feedback — chapter approved.**
