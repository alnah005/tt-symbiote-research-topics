# Compression Analysis: Chapter 3 Measuring SiLU Latency

## Summary

- Files analyzed: `index.md`, `profiling_setup.md`, `measurement_methodology.md`, `isolating_silu_from_matmul.md`
- Estimated current line count: 64 + 144 + 133 + 168 = 509 lines
- Estimated post-compression line count: ~430 lines
- Estimated reduction: ~15%

---

## CRUCIAL Suggestions

### CRUCIAL-1: CSV column table duplicated in `profiling_setup.md` and `measurement_methodology.md`

**Files and approximate lines:**
- `profiling_setup.md` lines 34–41: Full table defining `DEVICE KERNEL DURATION [ns]` and `OP TO OP LATENCY [ns]`, with column descriptions and a Warning block stating "3–10× higher."
- `measurement_methodology.md` lines 61–75 (Section 3): A second table covering the same two columns with near-identical descriptions, plus a concrete numeric example (12 µs vs 80–150 µs).

**Duplication detail:** The column definitions and the core message ("do not use OP TO OP LATENCY for hardware comparison") are restated in full in both files. The Warning in `profiling_setup.md` line 41 and the prose in `measurement_methodology.md` lines 72–75 are near-verbatim.

**Canonical copy:** Keep the full table in `profiling_setup.md` (its natural home — the section that introduces the CSV output format). In `measurement_methodology.md` Section 3, keep only the concrete numeric example (12 µs vs 80–150 µs) as load-bearing evidence and replace the duplicate table with a one-line cross-reference to `profiling_setup.md`.

**Files to reduce:** `measurement_methodology.md` Section 3 (~12 lines → ~4 lines).

**Estimated line savings:** ~8 lines.

---

### CRUCIAL-2: Warm-up code block duplicated in `profiling_setup.md` and `isolating_silu_from_matmul.md`

**Files and approximate lines:**
- `profiling_setup.md` lines 53–59: Standalone warm-up code snippet (`WARMUP_ITERS = 3`, `ttnn.silu`, `ttnn.synchronize_device`).
- `profiling_setup.md` lines 116–120: Same warm-up block embedded in the "Complete Setup Code Example."
- `isolating_silu_from_matmul.md` lines 64–68 (Strategy 1 code): Identical warm-up block (`WARMUP_ITERS = 3`, `ttnn.silu`, `ttnn.synchronize_device`).
- `isolating_silu_from_matmul.md` lines 118–123 (Strategy 2 code): Third instance of the same warm-up pattern.

**Duplication detail:** The warm-up snippet is copy-pasted verbatim (same variable name, same call sequence) across three locations. The canonical explanation of *why* warm-up is needed (kernel compilation overhead, first-run cache-miss inflation, "200ms when true time is 20µs") lives in `profiling_setup.md` Section 3.

**Canonical copy:** Keep the explanation and standalone snippet in `profiling_setup.md` Section 3. The two code examples in `isolating_silu_from_matmul.md` may retain the warm-up calls inline (they are part of complete runnable examples) but should replace the explanatory comment block (`# Warm-up: ensures the SiLU program is compiled and cached.`) with a brief inline comment plus a cross-reference: `# Warm-up — see profiling_setup.md §3 for rationale.`

**Files to reduce:** `isolating_silu_from_matmul.md` warm-up comment prose (minor, ~2 lines of comments per code block).

**Estimated line savings:** ~4 lines.

---

### CRUCIAL-3: `TT_METAL_DEVICE_PROFILER=1` environment variable warning duplicated in `index.md` and `profiling_setup.md`

**Files and approximate lines:**
- `index.md` lines 44–50 ("Required Environment" section): Lists the env var requirement as a bullet, then adds a Warning block (lines 48–49) and a Tip block (lines 50–51) that are near-verbatim copies of the Warning and Tip in `profiling_setup.md`.
- `profiling_setup.md` lines 16–18: Warning block ("must be set in the shell before the Python process starts … tt-metal reads this variable during its C++ initialization") and Tip block.

**Duplication detail:** The Warning text in `index.md` line 48 ("must be present in the shell environment before `python` is invoked. Setting it inside the Python script with `os.environ` after the fact will not enable the profiler.") is a paraphrase of `profiling_setup.md` line 16 ("must be set in the shell **before** the Python process starts. Setting it inside the script via `os.environ[\"TT_METAL_DEVICE_PROFILER\"] = \"1\"` after the interpreter has launched will not activate the profiler"). The Tip in `index.md` line 50 ("check that an `ops_perf_results_<timestamp>.csv` file appears") is not in `profiling_setup.md` and is load-bearing — keep it.

**Canonical copy:** Keep the full Warning and Tip in `profiling_setup.md` (the dedicated setup file). In `index.md` "Required Environment," keep the bullet-point list of requirements but remove the Warning and Tip blocks; replace with: `> See [profiling_setup.md](profiling_setup.md) §1 for the full explanation and the one-liner to verify profiling is active.`

**Files to reduce:** `index.md` lines 48–50 (~3 lines → ~1 line cross-reference).

**Estimated line savings:** ~2 lines (minor on its own, but adds up with the others and removes a maintenance hazard where the two warnings could drift out of sync).

---

### CRUCIAL-4: Complete benchmark setup code nearly duplicated in `profiling_setup.md` and `isolating_silu_from_matmul.md`

**Files and approximate lines:**
- `profiling_setup.md` lines 93–137 (Section 6 "Complete Setup Code Example"): Full benchmark script — device open, tensor allocation (`num_tokens=32, hidden_dim=4096, dtype=bfloat16, TILE_LAYOUT`), warm-up loop (3 iters), timed loop (20 iters), `ReadDeviceProfiler`, `close_device`.
- `isolating_silu_from_matmul.md` lines 43–81 (Strategy 1 code): Identical structure — same shape, same variable names (`WARMUP_ITERS=3`, `TIMED_ITERS=20`), same call sequence, with only minor comment differences.

**Duplication detail:** The two code blocks share the same device-open → tensor-allocate → warm-up → timed-loop → ReadDeviceProfiler → close-device skeleton verbatim. The only meaningful difference is that the `isolating_silu_from_matmul.md` version includes the comment `# standalone — no matmul before or after`, which is load-bearing context for the isolation strategy.

**Canonical copy:** Keep the full code in `isolating_silu_from_matmul.md` (Strategy 1 is precisely where a complete, runnable standalone benchmark belongs). In `profiling_setup.md` Section 6, shorten the code to show only the infrastructure calls (device open, ReadDeviceProfiler, close_device) and replace the body with a comment like `# ... warm-up and timed loop — see isolating_silu_from_matmul.md Strategy 1 for a complete example`. Retain the full tensor-allocation block in `profiling_setup.md` only if it adds shape/dtype guidance not already in `isolating_silu_from_matmul.md` (it does not — both use the same shape and dtype).

**Files to reduce:** `profiling_setup.md` Section 6 (~45 lines → ~20 lines).

**Estimated line savings:** ~25 lines.

## Agent A Change Log — C Feedback Pass 1
- measurement_methodology.md: Reduced CSV column table to cross-reference + kept unique 12µs/80-150µs example
- isolating_silu_from_matmul.md: Replaced warm-up re-explanation with cross-reference to profiling_setup.md
- index.md: Removed Warning/Tip for TT_METAL_DEVICE_PROFILER; added one-line cross-reference
- profiling_setup.md: Shortened §6 complete script to skeleton + cross-reference to isolating_silu_from_matmul.md

---

## MINOR Suggestions

### MINOR-1: `index.md` "How This Chapter Fits in the Guide" restates content from "Prerequisites" and "Next Steps"

Lines 54–58 repeat that Chapter 2 established SiLU/SFPU/memory-bandwidth-bound facts (already in the Prerequisites table at lines 22–23) and that Chapter 4 will compare SiLU against matmul (already in line 14, objective 4). This is typical overview prose and acceptable, but could be trimmed to one sentence each.

### MINOR-2: "Next Steps" footers in each file duplicate the chapter navigation already in `index.md`

Each file ends with a "Next Steps" section pointing to the next file. This is standard chapter-navigation boilerplate and is not harmful, but it means chapter navigation is maintained in five places instead of one. No action required unless the chapter structure changes.

### MINOR-3: `measurement_methodology.md` Section 2 warm-up rule (lines 38–40) repeats `profiling_setup.md` Section 3

The "3 warm-up iterations (minimum 2)" rule is stated in both places. The `measurement_methodology.md` instance is embedded in the statistical protocol list and serves a different purpose (reminding the reader *when* to apply the rule during analysis), so it is borderline acceptable. However, the phrasing is redundant enough to flag. Consider reducing `measurement_methodology.md` line 38 to: `1. Run **3 warm-up iterations** before the timed loop (see `profiling_setup.md` §3 for rationale).`

---

## Load-Bearing Evidence

The following specific facts, formulas, or code must NOT be removed from the chapter regardless of compression decisions:

1. **`DEVICE KERNEL DURATION [ns]` is the correct column; `OP TO OP LATENCY [ns]` must not be used for hardware comparison.** This distinction must remain prominently stated in at least one file — canonical location: `profiling_setup.md` §2.

2. **Concrete numeric example showing the column gap:** "`num_tokens=1, hidden_dim=4096`: `DEVICE KERNEL DURATION` ≈ 12 µs; `OP TO OP LATENCY` ≈ 80–150 µs." (`measurement_methodology.md` lines 70–74.) This must be retained exactly once.

3. **Warm-up minimum: 2 iterations; recommended: 3–5.** Must remain in `profiling_setup.md` §3.

4. **"A single cold-cache run can show SiLU taking 200ms when the true hardware time is 20µs."** (`profiling_setup.md` line 61.) Must stay exactly once.

5. **Decode expected ranges:** "`num_tokens=1, hidden_dim=4096`: SiLU ≈ 10–25 µs; gate_proj ≈ 30–80 µs" and "`num_tokens=32, hidden_dim=4096`: SiLU ≈ 15–35 µs; gate_proj ≈ 60–150 µs." (`measurement_methodology.md` lines 113–114.) Must not be removed.

6. **Prefill ratio threshold:** "SiLU falls to below 5% of total FFN time" at prefill batch sizes. (`measurement_methodology.md` line 120.) Must not be removed.

7. **`TT_METAL_DEVICE_PROFILER=1` must be set before process launch; `os.environ` after launch does not work.** Must remain in `profiling_setup.md` §1.

8. **Verification tip:** "Check that `ops_perf_results_<timestamp>.csv` appears after a test run; if no CSV appears, profiling is not enabled." (`index.md` line 50.) Must remain somewhere — either `index.md` or `profiling_setup.md`.

9. **Pitfalls table** (`measurement_methodology.md` lines 94–101): All six rows (no warm-up, ROW_MAJOR_LAYOUT, DRAM-backed tensor, wrong CSV column, dtype mismatch, ReadDeviceProfiler limit) must be retained in full in `measurement_methodology.md`.

10. **Strategy 1 tensor requirements table** (`isolating_silu_from_matmul.md` lines 35–39): The three-row table listing correct vs. incorrect layout, dtype, and shape must be retained in full.

11. **`process_ops_logs.py` command:** "`python tt-metal/tools/profiler/process_ops_logs.py`" — the exact post-processing command. (`profiling_setup.md` lines 27–28.) Must remain.

12. **`hidden_dim` must be a multiple of 32 for `TILE_LAYOUT`.** (`measurement_methodology.md` line 28.) Must remain.

13. **Median and p95 statistical formula code block** (`measurement_methodology.md` lines 45–56): The `statistics.median` and `sorted(...)[int(0.95 * len(...))]` snippet must stay exactly once.

---

## VERDICT: Crucial updates: yes

---

## Agent A Change Log — C Feedback Pass 1

The following changes should be applied to eliminate the CRUCIAL duplications identified above.

### Fix 1 — `measurement_methodology.md` Section 3: replace duplicate CSV column table with cross-reference

**File:** `measurement_methodology.md`
**Action:** Remove the duplicate `DEVICE KERNEL DURATION` / `OP TO OP LATENCY` table (lines 65–68). Replace it with a single cross-reference sentence. Retain the concrete numeric example (12 µs vs 80–150 µs) as it does not appear in `profiling_setup.md`.

Replace the current Section 3 body (from the table through "The OP TO OP LATENCY is dominated by host dispatch overhead") with:

```
> The profiler CSV contains two latency columns. **Always use `DEVICE KERNEL DURATION [ns]`** for hardware comparisons; see [`profiling_setup.md` §2](profiling_setup.md) for full column definitions and the rationale for excluding `OP TO OP LATENCY [ns]`.

For a SiLU kernel at `num_tokens=1, hidden_dim=4096`, a typical split illustrates why this matters:

- `DEVICE KERNEL DURATION` ≈ 12 µs
- `OP TO OP LATENCY` ≈ 80–150 µs

The OP TO OP LATENCY is dominated by host dispatch overhead and is not a useful measure of device capability.
```

**Estimated line savings:** ~8 lines.

---

### Fix 2 — `isolating_silu_from_matmul.md`: trim warm-up explanatory comments to cross-references

**File:** `isolating_silu_from_matmul.md`
**Action:** In Strategy 1 code (around line 64) and Strategy 2 code (around line 118), change the warm-up comment from:
```python
# Warm-up: ensures the SiLU program is compiled and cached.
```
to:
```python
# Warm-up: populates the program cache — see profiling_setup.md §3 for rationale.
```
(This is cosmetic but prevents the explanatory text from diverging from `profiling_setup.md` §3.)

**Estimated line savings:** 0 lines (comment swap), but eliminates a maintenance hazard.

---

### Fix 3 — `index.md` "Required Environment": remove near-verbatim Warning/Tip blocks, add cross-reference

**File:** `index.md`
**Action:** Remove lines 48–50 (the Warning and Tip blocks in "Required Environment"). Replace with a single cross-reference line:

```
> For the full setup explanation, the verification command, and a note on why `os.environ` does not work at runtime, see [`profiling_setup.md` §1](profiling_setup.md).
```

**Estimated line savings:** ~2 lines.

---

### Fix 4 — `profiling_setup.md` Section 6: replace near-duplicate full benchmark code with a skeleton + cross-reference

**File:** `profiling_setup.md`
**Action:** Section 6 currently contains a ~45-line complete benchmark that is nearly identical to Strategy 1 in `isolating_silu_from_matmul.md`. Shorten Section 6 to show only the infrastructure skeleton (device init, profiler flush, close) and replace the warm-up and timed-loop body with a cross-reference comment. Retain the tensor-allocation block because it shows the `[1, 1, num_tokens, hidden_dim]` shape and `bfloat16 / TILE_LAYOUT` requirements in context.

Suggested replacement for the timed loop and warm-up body in Section 6:

```python
# ── Warm-up and timed measurement loop ──────────────────────────────────────
# See isolating_silu_from_matmul.md §2 (Strategy 1) for a complete,
# runnable benchmark including warm-up (WARMUP_ITERS=3) and timed
# loop (TIMED_ITERS=20) with correct synchronization.

# ── Flush profiler data to disk ──────────────────────────────────────────────
ttnn.ReadDeviceProfiler(device)

ttnn.close_device(device)
```

**Estimated line savings:** ~25 lines.

---

# Compression Analysis: Chapter 3 Measuring SiLU Latency — Pass 2

## Summary
- Pass 1 fixes: all 4 verified as correctly applied
- New crucial duplications: none

## CRUCIAL Suggestions

None

## MINOR Suggestions

- **MINOR-A (carry-forward from Pass 1 MINOR-1):** `index.md` lines 54–55 still restate Chapter 2 SFPU/memory-bandwidth facts and Chapter 4 comparison intent already listed in the Prerequisites table and Learning Objectives. Acceptable overlap, no action required.
- **MINOR-B (carry-forward from Pass 1 MINOR-2):** "Next Steps" footers in all four files duplicate chapter navigation already in `index.md`. Boilerplate; no action required unless chapter structure changes.
- **MINOR-C (carry-forward from Pass 1 MINOR-3):** `measurement_methodology.md` §2 line 38 still restates the "3 warm-up iterations (minimum 2)" rule that also appears in `profiling_setup.md` §3. The instance in §2 serves a different purpose (protocol checklist) and is acceptable, but could link to `profiling_setup.md §3` for rationale consistency.
- **MINOR-D (new):** `index.md` line 48 cross-reference reads "the verification command … see `profiling_setup.md` §1", but `profiling_setup.md` §1 does not contain an explicit verification tip about checking for `ops_perf_results_<timestamp>.csv`. The CSV filename and path appear in §2 (lines 30–31). The cross-reference wording is slightly inaccurate; consider either moving the verification note into §1 of `profiling_setup.md` or adjusting the cross-reference to say "§1 and §2".

## Load-Bearing Evidence

All 13 LBE items confirmed present:

1. `DEVICE KERNEL DURATION` correct / `OP TO OP LATENCY` must-not-use distinction: present in `profiling_setup.md` §2 table.
2. 12 µs / 80–150 µs concrete numeric example: present in `measurement_methodology.md` §3 (lines 65–68). Exactly once.
3. Warm-up minimum 2, recommended 3–5: present in `profiling_setup.md` §3 (lines 49–51).
4. "200ms when true time is 20µs" cold-cache warning: present in `profiling_setup.md` §3 (line 61).
5. Decode expected ranges (`num_tokens=1,32`, `hidden_dim=4096`): present in `measurement_methodology.md` §6 (lines 108–109).
6. Prefill <5% threshold: present in `measurement_methodology.md` §6 (line 115).
7. `TT_METAL_DEVICE_PROFILER=1` must be set before process launch: present in `profiling_setup.md` §1 (line 16).
8. CSV verification tip (`ops_perf_results_<timestamp>.csv`): the Tip block was removed from `index.md` per Fix 3; `index.md` line 48 cross-references `profiling_setup.md §1` for "the verification command." The CSV filename is mentioned in `profiling_setup.md` §2 (lines 30–31) but no explicit "if no CSV appears, profiling is not enabled" sentence remains anywhere. This is a minor accuracy gap in the cross-reference (see MINOR-D above) — the information is adjacent but not literally present in §1. No crucial information is fully missing, but the pointer is imprecise.
9. Pitfalls table (all 6 rows): present in `measurement_methodology.md` §5 (lines 89–97).
10. Strategy 1 tensor requirements table: present in `isolating_silu_from_matmul.md` §2 (lines 35–39).
11. `process_ops_logs.py` command: present in `profiling_setup.md` §2 (lines 27–28).
12. `hidden_dim` must be a multiple of 32: present in `measurement_methodology.md` §1 (line 28).
13. Median and p95 statistical formula code block: present in `measurement_methodology.md` §2 (lines 52–53).

## VERDICT: Crucial updates: no
