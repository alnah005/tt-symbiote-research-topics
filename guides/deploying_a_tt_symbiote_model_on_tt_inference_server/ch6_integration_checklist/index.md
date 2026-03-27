# Chapter 6 — Integration Checklist and Worked Example

## Overview

This chapter consolidates everything covered in Chapters 1–5 into two actionable artifacts:

1. **A phased integration checklist** (`checklist.md`) — a numbered, phase-by-phase list of every concrete action required to integrate a TT Symbiote model into `tt-inference-server`. Each item links back to the chapter where the underlying concept is explained in depth.

2. **A complete worked example** (`worked_example.md`) — an end-to-end walkthrough using a hypothetical `TTMySymbioteModel` that wraps a 7B decoder-only model. The example includes real directory layouts, full `ModelSpec` entries, method skeletons with correct signatures, and a curated list of common pitfalls.

Together these two documents serve as the primary reference you should keep open while performing an actual integration. The checklist confirms you have not skipped any required step; the worked example shows exactly what correct code looks like at each step.

## Reading Order

Work through the files in this order:

1. [`checklist.md`](./checklist.md) — complete the checklist phase by phase before writing any model code
2. [`worked_example.md`](./worked_example.md) — follow the worked example as a template for your own model class

## What's Next

Chapter 7 covers the practical side of what happens after a model is successfully registered and starts running: how to interpret common error messages, what hard constraints exist on batch size and sequence length, and how to measure and improve inference throughput.

**Chapter 7:** [Debugging, Constraints, and Performance Tuning](../ch7_debugging_and_tuning/index.md)

---

**Next:** [`checklist.md`](./checklist.md)
