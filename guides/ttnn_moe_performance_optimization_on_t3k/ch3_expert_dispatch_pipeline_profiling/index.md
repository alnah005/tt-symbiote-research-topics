# Chapter 3: Expert Dispatch Pipeline Profiling

## The Central Question

Inside `TTNNExperts.forward`, there are seven distinct pipeline stages that execute between `all_to_all_dispatch` and `all_to_all_combine`. At batch=1 decode, each stage operates on a trivially small number of tokens — yet the aggregate cost of these stages can rival or exceed the CCL operations analyzed in Chapter 2. This chapter asks:

**Which of the seven pipeline stages dominates latency at batch=1 decode, and what is the per-stage breakdown for GLM-4-MoE and Bailing on a T3K 1×8 mesh?**

The seven stages, corresponding to `moe.py:L1191–L1341`, are:

| Stage | Lines | Operation |
|---|---|---|
| 1 | L1191–1212 | Token padding to `SPARSITY_BLOCK_SIZE=32` |
| 2 | L1214–1230 | Layout conversion + `ttnn.all_to_all_dispatch` |
| 3 | L1232–1248 | Reshape + `ttnn.moe_expert_token_remap` |
| 4 | L1250–1275 | w1 + w3 sparse matmul (gate-up projections) + silu |
| 5 | L1280–1289 | w2 sparse matmul (down projection) |
| 6 | L1293–1312 | Reshape + `ttnn.all_to_all_combine` |
| 7 | L1315–1341 | Weight application + padding removal |

The single biggest bottleneck determines the optimization target. If Stage 2 or Stage 6 (the all-to-all CCL pairs) dominate, that points to dispatch topology tuning. If Stages 4 and 5 dominate, the target is sparse matmul configuration. If Stage 7 dominates, the weight application code path deserves scrutiny.

---

## Experimental Setup

**Hardware:** T3K, 1×8 Wormhole mesh. Each device is a Wormhole B0 chip with an 8×8 compute grid (64 cores).

**Sequence configuration:** Batch=1 decode. A single token is processed per forward pass. After padding (Stage 1), the effective batch seen by the sparse matmuls is `SPARSITY_BLOCK_SIZE=32`.

**Model configurations profiled:**

| Config | hidden\_size | intermediate\_size | n\_routed\_experts | in0\_block\_w (gate/up) | in0\_block\_w (down) |
|---|---|---|---|---|---|
| GLM-4-MoE | 4096 | 1408 | 128 | 4 | 4 |
| Bailing | 4096 | ~1408 | 128 | 4 | 4 |

> The GLM-4-MoE `hidden_size` is 4096 in the `Glm4MoeConfig` defaults used here. `hidden_tiles = 4096 / 32 = 128`; `intermediate_tiles = 1408 / 32 = 44`. In both cases `min(4, tiles) = 4`, so the `in0_block_w` cap is not binding for either model.

**Profiling tool:** TTNN device-side op timers (primary) and Tracy wall-clock annotations (secondary). See Chapter 5 for Tracy setup details.

**Warmup protocol:** 20 un-timed forward passes to warm DRAM caches, L1 state, and semaphore handles. Then 100 timed iterations; report the mean and p95.

---

## Files in This Chapter

| File | Contents |
|---|---|
| [`token_padding_and_dispatch.md`](./token_padding_and_dispatch.md) | Stages 1–2: token padding to `SPARSITY_BLOCK_SIZE=32`, layout conversion, and `all_to_all_dispatch`. Why 32 was chosen. How to measure dispatch latency in isolation. |
| [`sparse_matmul_profiling.md`](./sparse_matmul_profiling.md) | Stages 4–5: the three `sparse_matmul` calls (w1, w3, w2). Deep-dive of `_make_sparse_matmul_program_config` (`moe.py:L62–L91`). `in0_block_w` and `per_core_M` sweep guidance. HiFi2 vs HiFi4 accuracy trade-off. |
| [`weight_application_overhead.md`](./weight_application_overhead.md) | Stage 7: post-combine weight application via `ttnn.repeat` + `permute` + `mul` + `sum`. Profile cost. Code sketch of an alternative elementwise approach. |
| [`bottleneck_summary.md`](./bottleneck_summary.md) | Aggregated per-stage latency table. Identifies the single biggest bottleneck. Maps findings back to `moe.py` source lines. |

---

## Research Questions Addressed

This chapter directly addresses:

- **Q2** — Which stage of the expert dispatch pipeline dominates latency at batch=1 decode?
- **Q3** — How should `sparse_matmul` program config parameters (`in0_block_w`, `per_core_M`, `math_fidelity`) be tuned for GLM-4-MoE and Bailing?
- **Q5** — What is the cost of the post-combine weight application step, and is there a lower-overhead alternative?

Findings feed into Chapter 7's optimization priority matrix: the stage identified as the bottleneck in [`bottleneck_summary.md`](./bottleneck_summary.md) determines which subsequent chapter (Chapter 4 for matmul config, Chapter 5 for dispatch tuning) deserves immediate attention.
