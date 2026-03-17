# Chapter 7: Implementation and Validation

## Overview

Chapters 1–6 built the conceptual foundation: Wormhole's memory hierarchy, the `ShardSpec` API, expert weight tensor shapes, prefetch patterns, tile alignment rules, and performance trade-offs. This chapter closes the loop with runnable code.

The four files in this chapter walk through every step from weight loading to a validated, benchmarked DRAM-sharded expert weight deployment:

1. **`code_patterns.md`** — Constructing valid `MemoryConfig` objects, converting weights to DRAM-sharded layout, and integrating with `ttnn.matmul`.
2. **`correctness_verification.md`** — Verifying that the resharded weights produce numerically correct outputs using Pearson Cross-Correlation (PCC).
3. **`benchmark_methodology.md`** — Measuring the latency and effective DRAM bandwidth improvement with a reproducible benchmark harness.

All code assumes Wormhole B0 hardware, `import ttnn` in scope, and that device initialization has been completed. Shapes and constants match the canonical examples from Chapter 3 (Mixtral 8x7B) and are extended to Qwen 235B-A22B where noted.

> **Tip:** Work through this chapter in order. `code_patterns.md` produces the sharded weight tensors; `correctness_verification.md` checks them before you trust the results; `benchmark_methodology.md` measures what you gained.

---

## Learning Objectives

1. **Construct a DRAM-sharded `MemoryConfig`** programmatically from model hyperparameters, satisfying all five tile-alignment rules from Chapter 5.
2. **Convert expert weight tensors to DRAM-sharded layout** using `ttnn.to_memory_config` and handle the three projection types (gate, up, down) with correct shard orientations.
3. **Integrate DRAM-sharded weights with `ttnn.matmul`**, understanding when a DRAM→L1 reshard is required and when the kernel can stream directly from DRAM.
4. **Verify numerical correctness** using PCC and diagnose shard misalignment from PCC failure patterns.
5. **Run a reproducible bandwidth benchmark** comparing interleaved vs DRAM-sharded weight layouts and report latency, bandwidth, and efficiency.

---

## End-to-End Implementation Checklist

Use this checklist as a completion gate before deploying DRAM-sharded expert weights to a production inference server.

### Step 1: Weight Loading

- [ ] Expert weights loaded from checkpoint onto CPU as `torch.Tensor` objects.
- [ ] Weights transferred to Wormhole device using `ttnn.from_torch(..., memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)`.
- [ ] All three projections (gate, up, down) present in DRAM with correct shapes.
- [ ] Dtype confirmed: `ttnn.bfloat16` or `ttnn.bfloat8_b` as target.

### Step 2: Shard Configuration

- [ ] `shard_shape` derived from `(tensor_height // num_shards, tensor_width)` for WIDTH_SHARDED gate/up, or `(tensor_height, tensor_width // num_shards)` for HEIGHT_SHARDED down.
- [ ] `shard_shape[0] % 32 == 0` verified (Rule 1).
- [ ] `shard_shape[1] % 32 == 0` verified (Rule 2).
- [ ] `num_shards` divides tensor dimension evenly (Rule 3).
- [ ] Shard size in bytes is a multiple of 32 (Rule 5): `shard_shape[0] * shard_shape[1] * bytes_per_element % 32 == 0`.
- [ ] `CoreRangeSet` constructed for the chosen DRAM bank grid (e.g., `ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 5))})`).

### Step 3: Resharding

- [ ] `ttnn.to_memory_config(weight_tensor, sharded_config)` executed for all expert projections.
- [ ] `tensor.memory_config()` inspected post-reshard to confirm `buffer_type=DRAM` and `memory_layout=WIDTH_SHARDED` (or `HEIGHT_SHARDED`).
- [ ] Reshard performed once at model load time; pointer stored in model weight dict.

### Step 4: Correctness Verification

- [ ] Reference output captured using interleaved weights (before resharding) via `ttnn.matmul`.
- [ ] Output captured using DRAM-sharded weights via `ttnn.matmul`.
- [ ] Both outputs converted to CPU with `ttnn.to_torch`.
- [ ] PCC computed: threshold `> 0.999` for bfloat16, `> 0.999` for bfloat8_b.
- [ ] PCC passes across multiple random activation inputs (at least 5 trials).

### Step 5: Benchmark

- [ ] Warmup iterations run (at least 5) to populate program cache.
- [ ] Timed iterations run (at least 20) for both interleaved and sharded configs.
- [ ] Latency recorded: median and p95 in milliseconds.
- [ ] DRAM read bytes recorded from device profiler.
- [ ] Effective bandwidth computed: `bytes_read / latency_s`.
- [ ] Bandwidth efficiency computed: `effective_bandwidth / 300 GB/s × 100%`.
- [ ] Results tabulated showing improvement delta.

---

## Prerequisites

| Chapter | Topics Required |
|---|---|
| Chapter 1 | `MemoryConfig` API; `BufferType`; `TensorMemoryLayout`; TILE_LAYOUT |
| Chapter 2 | `ShardSpec` field definitions; `CoreRangeSet` construction; `WIDTH_SHARDED` vs `HEIGHT_SHARDED` |
| Chapter 3 | Expert weight shapes; Mixtral 8x7B and Qwen 235B-A22B parameters |
| Chapter 4 | Roofline model; decode vs prefill arithmetic intensity |
| Chapter 5 | All five tile-alignment rules; `ShardSpec.shape` in elements |
| Chapter 6 | Expected bandwidth ranges; `batch_size × top_k ≤ 16` rule; program cache stability requirement |

---

## Chapter Structure

| File | Contents |
|---|---|
| `index.md` | This file: learning objectives, checklist, prerequisites |
| [`code_patterns.md`](./code_patterns.md) | Runnable code for weight loading, shard config construction, resharding, and `ttnn.matmul` integration |
| [`correctness_verification.md`](./correctness_verification.md) | PCC computation; step-by-step verification workflow; diagnosing shard misalignment failures |
| [`benchmark_methodology.md`](./benchmark_methodology.md) | Benchmark harness setup; what to measure; Tracy profiler usage; two-config comparison loop; result reporting |

---

## Key Constants Referenced in This Chapter

| Constant | Value | Source |
|---|---|---|
| Tensix cores (Wormhole B0) | 80 (8×10 grid) | Hardware spec |
| L1 per core | 1.5 MB | Hardware spec |
| Peak DRAM bandwidth | ~300 GB/s | Ch4 `bandwidth_estimation.md` |
| Ridge point | ~437 FLOP/byte | Ch4 `bandwidth_estimation.md` |
| BF16 tile size | 2048 bytes | Ch5 `tile_fundamentals.md` |
| bfloat8_b tile size | 1024 bytes | Ch5 `tile_fundamentals.md` |
| PCC threshold (BF16) | > 0.999 | TTNN convention |
| PCC threshold (bfloat8_b) | > 0.999 | TTNN convention |
| Decode regime boundary | `batch_size × top_k ≤ 16` | Ch6 `bandwidth_gain_analysis.md` |
| Mixtral 8x7B | `d_model=4096`, `d_ff=14336`, `num_experts=8`, `top_k=2` | Ch3 `projection_shapes.md` |
| Qwen 235B-A22B | `d_model=7168`, `d_ff=2048`, `num_experts=128`, `top_k=8` | Ch3 `projection_shapes.md` |

---

## Next Steps

Read `code_patterns.md` for complete, annotated code covering weight loading, shard config construction, resharding, and `ttnn.matmul` integration. Then proceed to `correctness_verification.md` to validate that the resharding did not alter numerical outputs, and finally `benchmark_methodology.md` to measure the latency and bandwidth improvement.
