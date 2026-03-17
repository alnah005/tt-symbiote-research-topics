# Writing a Gap Analysis

A gap analysis document is the primary artifact that translates profiling work into
engineering action. Its audience is optimization engineers who will implement fixes — people
who may not have been present during profiling and who need enough evidence to reproduce your
measurements, understand the confidence level of each finding, and prioritize implementation
work.

This file provides the seven-section template, describes what evidence to include in each
section, and explains how to express uncertainty using a two-tier confidence vocabulary.

---

## When to Write the Gap Analysis

Write the gap analysis after you have:

1. Collected at least one annotated Tracy trace with named `MoE/dispatch`,
   `MoE/expert_matmul`, and `MoE/combine` zones (Chapter 2 and 3).
2. Run the standard seq_len sweep `{64, 128, 256, 512, 1024, 2048, 4096}` with 20 timed
   iterations per point (Chapter 6).
3. Produced the linear regression decomposition separating the constant and linear gap
   components (Chapter 6, `interpreting_scaling_results.md`).
4. Identified at least one gap pattern from the Chapter 5 taxonomy (Patterns A–D) and
   mapped it to an optimization action (`gap_to_action_mapping.md`).

> **Warning:** Do not write the gap analysis based on a single profiling run. Iteration-to-
> iteration variance from OS scheduling can make a single measurement misleading. Always
> report median and p95 values from at least 10 iterations, and note the measurement
> conditions (hardware, firmware version, system load).

---

## Seven-Section Template

The sections below define what to include and how to phrase findings. A suggested word count
range is given for each section; stay within the lower bound for internal team documents and
the upper bound for cross-team handoffs.

---

### Section 1: Observed Latency

**Purpose:** State the measured end-to-end latency and the gap that motivated the
investigation. Quantify the gap as a fraction of the total.

**What to include:**

- Model configuration: architecture name, `num_experts`, `top_k`, `d_model`, `d_ff`,
  hardware (T3K or single chip), firmware version.
- Observed MoE forward pass duration: median and p95 at the reference seq_len.
- The gap duration: how it was measured (Tracy CSV export, device profiler CSV, or
  wallclock), which zones bound it, median and p95 at the reference seq_len.
- Gap as a percentage of total MoE forward time.

**Example:**

> On T3K (8× Wormhole B0, firmware vX.Y), the Qwen 235B-A22B MoE forward pass
> (num_experts=128, top_k=8, d_model=7168, d_ff=2048) at seq_len=1024 runs in a median
> 34.1ms (p95: 35.4ms). A gap of 15.8ms (median) is present between the end of the
> `MoE/expert_matmul` zone and the start of the `MoE/combine` zone. This gap accounts for
> 46% of the total forward pass latency.

**Suggested length:** 100–200 words.

---

### Section 2: Profiling Methodology

**Purpose:** Give the reader enough information to reproduce every measurement.

**What to include:**

- Build flags: `ENABLE_PROFILER=ON`, `TRACY_ENABLE` define, any other relevant CMake options.
- Environment variables: `TT_METAL_DEVICE_PROFILER=1`, `TRACY_NO_EXIT=1` if used.
- How Tracy traces were collected: `tracy-capture` invocation, `.tracy` file path, export
  command (`tracy-csvexport`).
- Warm-up protocol: number of warm-up iterations before timing, rationale.
- Timing protocol: number of timed iterations (minimum 20), aggregation method (median, p95).
- Reference to the sweep script or test harness used (file path or commit hash).

**Example code block to include verbatim:**

```bash
# Build
cmake -B build -DENABLE_PROFILER=ON -DTRACY_ENABLE=1
cmake --build build -j$(nproc)

# Collect trace (run in separate terminal)
tracy-capture -o moe_seq1024.tracy &

# Profile (20 timed iterations, seq_len=1024)
TT_METAL_DEVICE_PROFILER=1 python scripts/profile_moe.py \
    --seq_len 1024 --n_warmup 3 --n_timed 20 \
    --output_csv results/moe_seq1024.csv

# Export Tracy CSV
tracy-csvexport moe_seq1024.tracy > results/tracy_moe_seq1024.csv
```

**Suggested length:** 150–300 words plus the code block.

---

### Section 3: Per-Op Breakdown Table

**Purpose:** Show where time is spent across the MoE forward pass so the reader can
verify that the gap is real and not an artifact of a mis-named zone or a missing op.

**What to include:**

- A table with columns: `Phase`, `Op / Zone name`, `Median duration (ms)`, `% of total`.
- Rows for each major zone: `MoE/dispatch/router`, `MoE/dispatch/topk`,
  `MoE/dispatch/index_construction` (if annotated), `MoE/dispatch/gather`,
  `MoE/dispatch/all_to_all` (if annotated), `MoE/expert_matmul`, `MoE/combine`,
  and one row labeled `[unattributed gap]` for each gap.
- Source: note whether durations come from Tracy CPU zones, device profiler CSV
  (`profile_log_device.csv`), or wallclock.

**Example table:**

| Phase | Zone / Op | Median (ms) | % of total |
|---|---|---|---|
| Dispatch | `MoE/dispatch/router` | 0.8 | 2.3% |
| Dispatch | `MoE/dispatch/topk` | 0.4 | 1.2% |
| Dispatch | `MoE/dispatch/gather` | 1.1 | 3.2% |
| **[Gap A/C]** | `[unattributed — between gather and expert_matmul]` | **2.1** | **6.2%** |
| Expert compute | `MoE/expert_matmul` | 14.1 | 41.3% |
| **[Gap B]** | `[unattributed — between expert_matmul and combine]` | **15.8** | **46.3%** |
| Combine | `MoE/combine` | 0.8 | 2.3% |
| **Total** | `MoE/forward` | **34.1** | **100%** |

**Suggested length:** Table plus 50–100 words of explanatory prose.

---

### Section 4: Gap Attribution Findings

**Purpose:** State which Chapter 5 pattern(s) were identified, what evidence supports the
attribution, and the confidence level.

**What to include:**

- For each gap: the pattern label (A, B, C, or D), the evidence used (Tracy zone added,
  code search result, device profiler CSV inspection), and the confidence level (see below).
- Annotated Tracy screenshots showing the gap before and after adding a zone (embed as
  figures or reference as attached files).
- Excerpts from `profile_log_device.csv` showing the `OP TO OP LATENCY [ns]` value between
  the relevant op pair (if available).

**Confidence vocabulary:**

Use exactly two tiers to label each finding:

- **Confirmed:** The gap is consistent across all 20 timed iterations (CV < 10%), is
  attributable to a specific named op or host call (Tracy zone covers the full gap), and
  the same gap is visible in the device profiler CSV as an inter-op latency.
- **Suspected:** The gap is present in most iterations but not all, or the Tracy zone
  covers only part of the gap, or the attribution is inferred from indirect evidence
  (e.g., code search found a `synchronize_device` call but no Tracy zone was added to
  confirm it is the culprit).

> **Warning:** Never label a finding "confirmed" if it rests on a single profiling run or
> on code inspection alone. Confirmation requires that a Tracy zone added around the
> suspected code covers the full gap duration in the trace.

**Example attribution paragraph:**

> **Gap between `MoE/expert_matmul` and `MoE/combine` — 15.8ms median (CONFIRMED, Pattern B)**
>
> A `ttnn.synchronize_device(device)` call was found at line 247 of
> `models/qwen_moe/moe_layer.py`. Adding a Tracy zone around this call
> (`ttnn.synchronize_device — gap B probe`) produced a zone of 15.6ms median duration,
> accounting for 98.7% of the observed gap. The remaining 0.2ms is within measurement noise.
> This is classified as **confirmed**.

**Suggested length:** 200–400 words plus figures.

---

### Section 5: Scaling Behavior

**Purpose:** Quantify how each gap component scales with seq_len using the Chapter 6 sweep.

**What to include:**

- The sweep table: `seq_len`, `median_ms`, `p95_ms` for the gap at each of the 7 standard
  seq_len values `{64, 128, 256, 512, 1024, 2048, 4096}`.
- The log-log plot image (`results/moe_gap_scaling.png`).
- The scaling exponent (log-log slope), R² of the fit, and interpretation.
- Linear regression decomposition: constant term (ms) and linear slope (µs/token), with
  attribution of each to a Chapter 5 pattern.
- Any tile-count discontinuities observed in the sweep, with the seq_len at which they occur
  and the suspected tile boundary.

**Example decomposition summary:**

> Linear regression (in linear space) over the 7 sweep points yields:
>
> - Constant term: **13.8ms** — attributed to the `ttnn.synchronize_device` call (Pattern B,
>   confirmed).
> - Linear slope: **1.68µs/token** — compared against the theoretical CCL slope of
>   2.05µs/token (ratio 0.82, within ±20% tolerance) — attributed to CCL all-to-all
>   (Pattern C, suspected; CCL zone annotation pending).
> - R² = 0.997, indicating the two-component model explains 99.7% of the variance.

**Suggested length:** 150–300 words plus the table and figure.

---

### Section 6: Prioritized Root Causes

**Purpose:** Rank the confirmed and suspected root causes by total latency impact across a
representative inference workload.

**What to include:**

- The workload definition used for prioritization: prefill `seq_len`, number of decode steps.
- The total latency impact estimate for each root cause (from `gap_to_action_mapping.md`).
- A ranked list with the most impactful root cause first.

**Example ranked list:**

> Workload: 1× prefill at seq_len=1024 followed by 256 decode steps at seq_len=1.
>
> 1. **Pattern B (sync barrier) — ~3,547ms total impact** (13.8ms × 257 steps).
>    Highest priority. Affects every inference step.
> 2. **Pattern C (CCL all-to-all) — ~2ms total impact** (linear term at seq_len=1024,
>    negligible at decode seq_len=1).
>    Lower priority for decode-heavy workloads; re-evaluate if prefill dominates.
> 3. **Pattern A (host Python) — not detected** (linear slope consistent with CCL estimate;
>    no additional overhead).
> 4. **Pattern D (program cache) — ~400ms first-call cost only** (eliminated by warm-up;
>    not present in hot-path).

**Suggested length:** 100–200 words plus the ranked list.

---

### Section 7: Recommended Actions

**Purpose:** Give the optimization engineer a concrete, ordered to-do list.

**What to include:**

- One action item per root cause, in priority order.
- For each action: the specific API change, the file and line where the change is needed,
  the expected latency reduction, and the PCC validation requirement.
- A note on the validation protocol: PCC threshold, reference implementation, and how to run
  the check.

**Example action item:**

> **OPT-1 (Priority 1): Remove `ttnn.synchronize_device` in `moe_layer.py:247`**
>
> Replace the call with a `ttnn.record_event` / `ttnn.wait_for_event` pair, or remove it
> entirely if the device command queue provides sufficient ordering (see
> `gap_to_action_mapping.md`, Pattern B). Expected reduction: ~14ms per forward pass step.
>
> Validation: Run `test_moe_pcc.py` with `--seq_len 1024 --n_iters 20`. Confirm PCC > 0.999
> between the optimized output and the CPU reference for all 20 iterations.

**Suggested length:** 150–300 words plus the action item list.

---

## Audience Considerations

Write the gap analysis for two audiences simultaneously:

**Primary audience — optimization engineers:** They need precise file paths, line numbers,
API names, and expected latency numbers. They will run the validation check. Give them the
information they need without making them re-read the profiling chapters.

**Secondary audience — technical leads and reviewers:** They need to understand the
confidence level of each finding, the methodology used, and the prioritization rationale.
Express uncertainty clearly using the confirmed/suspected vocabulary. Do not bury caveats
in footnotes.

> **Tip:** Include the annotated Tracy screenshot for every "suspected" finding, not just
> "confirmed" ones. A screenshot showing an unannotated gap is itself evidence — it
> demonstrates that you looked and did not find a zone, which narrows the attribution even
> if it does not confirm it.

---

## Reproducibility Checklist

Before circulating the gap analysis, verify:

- [ ] All measurements report both median and p95 (not mean).
- [ ] Warm-up protocol is documented (number of warm-up iterations).
- [ ] Hardware configuration is specified (T3K vs. single chip, firmware version).
- [ ] Every "confirmed" finding has a Tracy zone screenshot and a device profiler CSV excerpt.
- [ ] Every "suspected" finding is clearly labeled as suspected.
- [ ] The seq_len sweep table and plot are included.
- [ ] Linear regression decomposition values are present.
- [ ] Recommended actions include file paths, line numbers, and PCC thresholds.
- [ ] The gap analysis document references the commit hash or script paths used for profiling.

---

---

**Next:** [`optimization_action_reference.md`](./optimization_action_reference.md)
