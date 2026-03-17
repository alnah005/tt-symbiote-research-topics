# B Review — Chapter 7: Implementation Workflow — Pass 1

## Issues Found

**Issue 1 — [throughput_profiling.md, lines 59 and 86]: Warm-up iteration count is below the required minimum**

Line 59: "Run at least 2 un-timed iterations before starting the 20-iteration timed benchmark."
Line 86: `NUM_WARMUP = 2`

The authoritative benchmark methodology requires a warm-up of ≥ 3 iterations excluded before the timed loop begins. Both the prose instruction and the constant in the code specify only 2 warm-up iterations, which falls short of the minimum.

Correct value: warm-up must be ≥ 3 iterations (e.g., `NUM_WARMUP = 3`), and the prose must read "Run at least 3 un-timed iterations."

---

## Verdict
1 factual error found — see above.

## Agent A Change Log — B Feedback Pass 1
- throughput_profiling.md: Fixed warm-up count from 2 to 3 iterations (both prose and NUM_WARMUP constant)

---

# B Review — Chapter 7: Implementation Workflow — Pass 2

## Pass 1 Fix Verification

Fix correctly applied. `throughput_profiling.md` line 59 now reads "Run at least 3 un-timed iterations before starting the 20-iteration timed benchmark." and line 86 shows `NUM_WARMUP = 3`. Both locations updated consistently.

## Issues Found

**Issue 1 — [throughput_profiling.md, line 67]: Operation count is wrong**

Line 67: "The SwiGLU expert FFN consists of four operations."

The table immediately below (lines 70–76) lists five distinct operations: gate matmul, SiLU activation, up matmul, elementwise mul, and down matmul. The prose says "four" but the table enumerates five. Correct value: "five operations."

## Verdict

1 factual error found.

## Agent A Change Log — B Feedback Pass 2
- throughput_profiling.md: Fixed "four operations" to "five operations" for SwiGLU expert FFN

---

# B Review — Chapter 7: Implementation Workflow — Pass 3

## Pass 2 Fix Verification
Fix correctly applied. `throughput_profiling.md` line 67 now reads "The SwiGLU expert FFN consists of five operations." The count matches the five-row table immediately below (gate matmul, SiLU activation, up matmul, elementwise mul, down matmul).

## Verdict
No feedback — chapter approved.
