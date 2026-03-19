## Pass 1

**Reviewer:** Agent B (Critic)
**Files reviewed:** `index.md`, `csv_column_definitions.md`, `pm_ideal_and_fpu_util.md`

---

### Issue 1 — Wrong matmul dimension notation

**File:** `pm_ideal_and_fpu_util.md`, line 24
**Error:** The matmul is described as having dimensions "M × K × K × N". This looks like a 4-D tensor shape rather than two matrices. A standard matmul multiplies an (M × K) matrix by a (K × N) matrix. Writing "M × K × K × N" implies K is listed twice as a separate dimension, which will confuse any reader trying to implement the formula — they may miscount the number of tile-triplets.
**Fix:** Change to "M × K and K × N (tiles: `M_t = M/32`, `K_t = K/32`, `N_t = N/32`)" or equivalent notation that clearly identifies two operand matrices.

---

### Issue 2 — Self-contradictory claim about PM IDEAL as a strict lower bound

**File:** `pm_ideal_and_fpu_util.md`, lines 93–101
**Error:** Line 93 states that FPU UTIL > 1.0 "would contradict the guarantee that PM IDEAL is a lower bound," implying such values are impossible. Lines 95–99 then explain that values of 1.01–1.05 do appear in practice due to rounding and measurement noise. The text cannot have it both ways: PM IDEAL is described as a guaranteed lower bound in one sentence and then violated in the next. A reader implementing a validator that flags FPU UTIL > 1.0 as always-a-bug (based on line 93) would incorrectly reject valid data in the 1.01–1.05 range.
**Fix:** Remove the claim that exceeding 1.0 contradicts a "guarantee." Restate that PM IDEAL is a model approximation (lower bound in expectation, not by mathematical guarantee), so small exceedances are expected from quantization and measurement noise; only large exceedances indicate bugs.

---

### Issue 3 — Duration decomposition equation is stated as general but only holds for compute-bound ops

**File:** `pm_ideal_and_fpu_util.md`, lines 127–131
**Error:** The breakdown equation

```
DEVICE KERNEL DURATION [cycle]
  = TRISC1_KERNEL_DURATION_cycles   (slowest core, math-bound path)
  + load_imbalance_cycles
  + synchronization_overhead_cycles
```

is presented without qualification, but it is only correct when TRISC1 is the longest-running RISC processor (i.e., compute-bound ops). For memory-bound ops, NCRISC duration exceeds TRISC1 duration and should appear in the dominant term. A reader applying this decomposition to a memory-bound op would compute a negative "load_imbalance_cycles" or misattribute the gap. The note at line 135 partially corrects this but comes after the equation and does not fix the formula itself.
**Fix:** Add a qualifier to the equation, e.g., "(compute-bound case only; for memory-bound ops substitute `NCRISC_KERNEL_DURATION_cycles`)", or restructure to present two cases explicitly.

---

### Issue 4 — Unverified Python API method name

**File:** `csv_column_definitions.md`, lines 114–116
**Error:** The code snippet calls `device.sfpu_clock_rate()` as the way to confirm clock frequency at runtime. If this method does not exist or has a different name in the public `ttnn` API (it is not a widely documented method), any reader who copies this snippet will get an `AttributeError` and conclude the instruction is wrong, which may cause them to distrust the entire clock-frequency section.
**Fix:** Verify that `device.sfpu_clock_rate()` is a real, stable public API method. If uncertain, replace with the `device_params.json` approach (already mentioned two lines later) as the primary recommended method, and demote or remove the Python snippet.

---

**Summary:** 4 correctness issues found. No missing planned files. Navigation footers are present on all three content files. All markdown links in `index.md` are correctly formed.

## Pass 2

**Reviewer:** Agent B (Critic)
**Files reviewed:** `index.md`, `csv_column_definitions.md`, `pm_ideal_and_fpu_util.md`
**Status of Pass 1 issues:** All 4 issues from Pass 1 are resolved in the current files.

---

### Issue 1 — Memory-bound tip inverts the causal direction

**File:** `pm_ideal_and_fpu_util.md`, line 87
**Error:** The tip reads: "the NoC is transferring tiles to L1 *faster* than TRISC1 can consume them, but L1 latency or bandwidth still exceeds the compute time." This description contradicts itself and the definition of memory-bound. A memory-bound op means the data-movement subsystem is *slower* than the math engine — NCRISC cannot deliver tiles quickly enough for TRISC1 to stay busy. If the NoC were transferring tiles *faster* than TRISC1 consumes them, TRISC1 would never stall for data, which is the definition of a compute-bound op. A reader following this tip would misclassify the bottleneck and pull the wrong optimization lever.
**Fix:** Change "the NoC is transferring tiles to L1 faster than TRISC1 can consume them" to "the NoC cannot deliver tiles to L1 fast enough for TRISC1 to stay busy." The corrected tip should read: "When `FPU UTIL` is low but `NCRISC KERNEL DURATION [ns]` is high, the op is memory-bound: the NoC cannot deliver tiles to L1 fast enough for TRISC1 to stay busy. Improving data layout (e.g., switching to a sharded memory config) is the right lever."

---

### Issue 2 — RISC duration comparison incorrectly refers to "sum" of concurrent processors

**File:** `csv_column_definitions.md`, line 134
**Error:** The text says "A large gap between `DEVICE KERNEL DURATION` and the sum of RISC durations on one core indicates load-balancing opportunity." The five RISC processors per core run *concurrently*, so their durations do not sum to the core's elapsed time — they overlap. A reader comparing `DEVICE KERNEL DURATION` against `BRISC + NCRISC + TRISC0 + TRISC1 + TRISC2` would compute a large artificial gap even on a perfectly balanced op, and incorrectly conclude there is a load-balancing problem. The correct comparison is against the *maximum* RISC duration on the core (the slowest RISC sets the core's wall-clock contribution).
**Fix:** Change "the sum of RISC durations on one core" to "the longest individual RISC duration on one core (i.e., `max(BRISC, NCRISC, TRISC0, TRISC1, TRISC2)` for that core)."

---

**Summary:** 2 correctness issues found in Pass 2. All 4 Pass 1 issues are resolved. No broken links. Navigation footers present on all files.

## Pass 3

**Reviewer:** Agent B (Critic)
**Files reviewed:** `index.md`, `csv_column_definitions.md`, `pm_ideal_and_fpu_util.md`
**Status of Pass 2 issues:** Both Pass 2 issues are resolved in the current files.

---

### Issue 1 — Tile-unit FLOPs expression is wrong by a factor of 32

**File:** `pm_ideal_and_fpu_util.md`, line 24 and lines 29–30
**Error:** The text claims total FLOPs = `2 × M × K × N` and states the tile-unit equivalent is `2 × M_t × K_t × N_t × 1024`. These two expressions are not equal. Substituting `M_t = M/32`, `K_t = K/32`, `N_t = N/32`:

```
2 × M_t × K_t × N_t × 1024
  = 2 × (M/32)(K/32)(N/32) × 1024
  = 2MKN × 1024/32768
  = 2MKN / 32
```

The correct tile-unit expansion is `2 × M_t × K_t × N_t × 32768` (each tile-triplet step computes a 32×32×32 outer product = 32³ MACs = 2 × 32³ = 65536 FLOPs; `2 × 32768 = 65536`). Using `× 1024` causes the `compute_cycles` formula to undercount FLOPs by 32×, producing a PM IDEAL value that is 32× too small. Any reader who manually verifies a PM IDEAL from the CSV using this formula will get a number that is off by an order of magnitude.

**Fix:** Replace `× 2 × 1024` with `× 2 × 32768` in the tile-unit expression and in the `compute_cycles` formula numerator. Update the surrounding prose accordingly.

---

### Issue 2 — "Difference between PM_IDEAL_cycles and TRISC1_KERNEL_DURATION_cycles" is conflated with the dimensionless quantity `1 − FPU_UTIL`

**File:** `pm_ideal_and_fpu_util.md`, line 123
**Error:** The text reads: "FPU stalls — captured by the difference between `PM_IDEAL_cycles` and `TRISC1_KERNEL_DURATION_cycles` (i.e., `1 − FPU_UTIL`)."

The "difference" in cycles is `TRISC1_cycles − PM_IDEAL_cycles` (units: cycles). The quantity `1 − FPU_UTIL` equals `(TRISC1_cycles − PM_IDEAL_cycles) / TRISC1_cycles` (dimensionless fraction). The parenthetical asserts these two quantities are the same thing, but one is in cycles and the other is a fraction. A reader computing "stall cycles" by plugging in `1 − FPU_UTIL` (e.g., 0.3) will get 0.3, not the actual stall cycle count (e.g., 30,000 cycles).

**Fix:** Change the parenthetical to correctly express the relationship:
`(i.e., stall cycles = TRISC1_KERNEL_DURATION_cycles × (1 − FPU_UTIL))`

---

**Summary:** 2 correctness issues found in Pass 3. All Pass 1 and Pass 2 issues remain resolved. No broken links or missing navigation footers.

## Pass 4

**Reviewer:** Agent B (Critic)
**Files reviewed:** `index.md`, `csv_column_definitions.md`, `pm_ideal_and_fpu_util.md`
**Status of Pass 3 issues:** Both Pass 3 issues are resolved in the current files.

---

No feedback — chapter approved.

## Pass 5

**Reviewer:** Agent B (Critic)
**Files reviewed:** `index.md`, `csv_column_definitions.md`, `pm_ideal_and_fpu_util.md`
**Status of Pass 4:** Chapter was approved with no issues.

---

No feedback — chapter approved.
