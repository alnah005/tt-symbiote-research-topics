# Chapter 3: Measuring SiLU Latency in TTNN

This chapter provides a complete, hands-on guide to measuring the latency of `ttnn.silu` on Tenstorrent hardware using the TTNN (Tenstorrent Neural Network) device profiler. By the end of this chapter you will have a working benchmark script and a clear methodology for isolating SiLU cost from surrounding matrix multiplication operations.

---

## Learning Objectives

By the end of this chapter you will be able to:

1. Enable the TTNN device profiler and collect per-operation (per-op) timing data in CSV format.
2. Structure a benchmark script with correct warm-up and timed loop to avoid first-run cache-miss inflation.
3. Locate and read the `DEVICE KERNEL DURATION [ns]` column from the profiler CSV output — the correct column for hardware execution time — and understand why `OP TO OP LATENCY [ns]` must not be used for hardware comparison.
4. Isolate `ttnn.silu` latency from surrounding matmul operations using either a standalone benchmark or a difference-measurement approach.
5. Recognize and avoid the most common measurement pitfalls: first-run cache misses, data-type (dtype) mismatches, and incorrect CSV column selection.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Chapter 2 | Understanding of the SiLU SFPU (Special Function Processing Unit) execution model and why SiLU is a memory-bandwidth-bound operation at typical decode batch sizes |
| TTNN familiarity | General familiarity with TTNN device initialization, tensor creation, and operation dispatch |
| tt-metal installed | A working tt-metal installation with TTNN Python bindings available |
| `TT_METAL_DEVICE_PROFILER=1` | Environment variable support confirmed for your tt-metal build (profiling is disabled in some release builds; see `profiling_setup.md`) |

---

## Chapter Structure

| File | Contents |
|---|---|
| `index.md` (this file) | Chapter overview, learning objectives, prerequisites |
| `profiling_setup.md` | How to enable the TTNN profiler, what the CSV output contains, and warm-up requirements |
| `isolating_silu_from_matmul.md` | Two strategies for measuring `ttnn.silu` in isolation from gate_proj matmul |
| `measurement_methodology.md` | Recommended input shapes, statistical protocol, pitfalls table, and expected result ranges |

---

## Required Environment

- **tt-metal** installed and on `PYTHONPATH`.
- The `TT_METAL_DEVICE_PROFILER=1` environment variable must be set at process launch (not imported at runtime).
- Python 3.8+ with `ttnn` importable.
- A Tenstorrent device (Grayskull, Wormhole, or later) visible to the tt-metal runtime.

> For the full setup explanation, the verification command, and a note on why `os.environ` does not work at runtime, see [`profiling_setup.md` §1](profiling_setup.md).

---

## How This Chapter Fits in the Guide

- **Chapter 2** established that `ttnn.silu` executes on the SFPU inside each Tensix core and is memory-bandwidth-bound at small batch sizes; this chapter quantifies that cost precisely.
- **Chapter 4** will use the measurements collected here to compare SiLU latency against gate_proj matmul latency and interpret the ratio in the context of MoE (Mixture of Experts) forward-pass optimization.

---

## Next Steps

Proceed to [`profiling_setup.md`](profiling_setup.md) to configure the TTNN device profiler and understand the CSV output format before writing any benchmark code.
