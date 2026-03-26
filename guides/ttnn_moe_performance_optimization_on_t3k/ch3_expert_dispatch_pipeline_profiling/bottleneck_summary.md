# Bottleneck Summary

## Context

This file addresses:
- **Q2** — Which stage of the expert dispatch pipeline dominates latency at batch=1 decode?
- **Q3** — How do sparse matmul config parameters affect that dominant stage?
- **Q5** — What is the weight application overhead relative to other stages?

This file synthesizes findings from [`token_padding_and_dispatch.md`](./token_padding_and_dispatch.md), [`sparse_matmul_profiling.md`](./sparse_matmul_profiling.md), and [`weight_application_overhead.md`](./weight_application_overhead.md), and maps each bottleneck back to its `moe.py` source lines.

---

## Per-Stage Latency Table

Run the isolation benchmarks described in the preceding files and record results here. Measurements are for batch=1 decode on T3K (1×8, Wormhole B0). Both models share the same stage structure and `moe.py` line references.

| Stage | Description | Lines | GLM-4-MoE mean (µs) | Bailing mean (µs) |
|---|---|---|---|---|
| 1 | Token padding (3× `ttnn.pad`) | L1191–L1212 | ___ | ___ |
| 2a | Layout conversion (2× `ttnn.to_layout`) | L1214–L1223 | ___ | ___ |
| 2b | `ttnn.all_to_all_dispatch` | L1225–L1230 | ___ | ___ |
| 3 | Reshape + `ttnn.moe_expert_token_remap` | L1232–L1248 | ___ | ___ |
| 4a | w1 `sparse_matmul` (gate projection) | L1250–L1259 | ___ | ___ |
| 4b | w3 `sparse_matmul` (up projection) | L1260–L1269 | ___ | ___ |
| 4c | `ttnn.silu` + `ttnn.mul` | L1271–L1275 | ___ | ___ |
| 5 | w2 `sparse_matmul` (down projection) | L1280–L1289 | ___ | ___ |
| 6a | Reshape + permute (pre-combine) | L1293–L1304 | ___ | ___ |
| 6b | `ttnn.all_to_all_combine` | L1307–L1312 | ___ | ___ |
| 7a | Weight application (`repeat`+`permute`+`mul`+`sum`) | L1321–L1335 | ___ | ___ |
| 7b | Padding removal (`ttnn.slice`) | L1338–L1341 | ___ | ___ |
| **Total** | `TTNNExperts.forward` | L1159–L1343 | ___ | ___ |

---

## Pre-Measurement Predictions

Before running the benchmarks, the following predictions can be derived from first principles. They should be replaced by measured values as soon as data is available.

### Why the all-to-all ops are likely the largest individual stages

At batch=1 decode, with topk=2 and 128 experts across 8 devices:

- `all_to_all_dispatch` routes 32 padded tokens (after padding) with a per-device message size of ~64 KB (computed in `token_padding_and_dispatch.md`). This is 8× larger than the `all_gather_async` message in Chapter 2.
- `all_to_all_combine` performs the inverse routing with a comparable message size.
- Both ops traverse the 8-device mesh with the same per-hop startup latency as the CCL ops analyzed in Chapter 2.

**Expected range for each all-to-all op: 50–150 µs**, depending on expert load distribution, routing implementation, and message fragmentation. Together, Stages 2b and 6b may account for 40–60% of `TTNNExperts.forward` time.

### Why the sparse matmuls are unlikely to dominate at batch=1

As computed in `sparse_matmul_profiling.md`:

- w1 matmul: `(32, 4096) × (4096, 1408)` = ~369 MFLOPs. At 150 TFLOPS peak → ~2.5 µs compute. Even with 5–10× memory-bandwidth overhead, this is 12–25 µs.
- w2 matmul: `(32, 1408) × (1408, 4096)` = ~369 MFLOPs. Same order.
- At batch=1, only 1–2 devices have active experts; the other 6 devices complete their sparse matmuls in near-zero time (empty sparse blocks).

**Expected range for each sparse matmul: 10–30 µs** on the active device. The two all-to-all ops are expected to cost more.

### Weight application: a non-trivial fraction

As analyzed in `weight_application_overhead.md`, Stage 7a involves 5+ separate op dispatches (`repeat`, `permute`, `to_layout`, `mul`, `sum`) on small tensors. Each op incurs kernel launch overhead independent of data size.

**Expected range for Stage 7a: 15–40 µs**. If this fraction exceeds 20% of `TTNNExperts.forward`, the alternative implementation (reshape + broadcast) is worth deploying.

### First-principles prediction table

| Stage | Predicted range (µs) | Primary cost driver |
|---|---|---|
| Stage 1 (padding) | 3–9 | 3 small `ttnn.pad` kernel launches |
| Stage 2a (layout cast) | 2–6 | 2 small `ttnn.to_layout` kernel launches |
| Stage 2b (dispatch) | 50–150 | Cross-device routing, message size ~64 KB |
| Stage 3 (remap) | 2–8 | Metadata-heavy reshape + remap kernel |
| Stage 4a (w1) | 10–30 | Matmul on 1 sparse block, 44/64 core utilization |
| Stage 4b (w3) | 10–30 | Same as w1 (parallel or sequential w/ w1) |
| Stage 4c (silu+mul) | 2–5 | Small elementwise ops |
| Stage 5 (w2) | 10–30 | Matmul on 1 sparse block, full 64-core utilization |
| Stage 6a (reshape) | 1–4 | Metadata-only reshapes |
| Stage 6b (combine) | 50–150 | Cross-device gather, message size ~64 KB |
| Stage 7a (weight app) | 15–40 | Repeat+permute overhead on small tensors |
| Stage 7b (slice) | 1–3 | Single `ttnn.slice` on small output |
| **Total** | **~160–470 µs** | Dominated by dispatch + combine |

---

## Identifying the Single Biggest Bottleneck

Based on the first-principles analysis, the predicted ranking is:

1. **Stage 2b (`ttnn.all_to_all_dispatch`) + Stage 6b (`ttnn.all_to_all_combine`)** — together, the two all-to-all ops are expected to dominate, each potentially costing 50–150 µs. If they are roughly equal, they together account for 60–70% of `TTNNExperts.forward` time.

2. **Stage 4 (w1 + w3 sparse matmuls) + Stage 5 (w2 sparse matmul)** — three matmuls, each 10–30 µs, for a combined 30–90 µs. Second-largest contributor.

3. **Stage 7a (weight application)** — 15–40 µs. Third-largest contributor; structurally reducible by adopting the reshape-based alternative.

4. **Stages 1, 2a, 3, 4c, 6a, 7b** — collectively < 30 µs. Not worth optimizing until the top three are addressed.

**If measurements confirm prediction:** The optimization priority order is:
1. Reduce `all_to_all_dispatch` + `all_to_all_combine` latency (topology, message compression, or overlapping with other work).
2. Tune sparse matmul config (`in0_block_w` sweep, core utilization for gate/up projections).
3. Replace weight application with the reshape-based alternative.

**If measurements contradict prediction:** Update this table and revise the priority order. Specific scenarios that could shift the ranking:

- If sparse matmuls take > 100 µs (possible if the DRAM bandwidth bottleneck is worse than estimated): matmul tuning becomes the top priority, and Chapter 4 should be consulted before any dispatch tuning.
- If Stage 7a takes > 50 µs (possible if `ttnn.repeat` has higher overhead than estimated): the weight application alternative should be implemented immediately as a quick win.
- If the all-to-all ops take < 30 µs (possible if the implementation uses on-chip buffers for small messages): the bottleneck shifts to the matmuls and weight application.

---

## Source Line Map for Optimization Targets

This table maps each identified bottleneck stage to the exact `moe.py` lines and the code structures responsible for the overhead.

| Bottleneck | Primary lines | Responsible construct | Optimization lever |
|---|---|---|---|
| `all_to_all_dispatch` | L1225–L1230 | `ttnn.all_to_all_dispatch` call | Message routing, cluster_axis topology |
| `all_to_all_combine` | L1307–L1312 | `ttnn.all_to_all_combine` call | Same as dispatch; symmetric optimization |
| w1 sparse matmul | L1250–L1259 | `_gate_up_program_config` (L1144–L1149) | `in0_block_w` sweep; intermediate_size alignment |
| w3 sparse matmul | L1260–L1269 | `_gate_up_program_config` (L1144–L1149) | Same config as w1 |
| w2 sparse matmul | L1280–L1289 | `_down_program_config` (L1150–L1154) | `in0_block_w` sweep; full core utilization already |
| `math_fidelity` (all matmuls) | L1155–L1157 | `_expert_compute_cfg` | `HiFi2` vs `HiFi4` accuracy/speed trade-off |
| Weight application | L1321–L1335 | `ttnn.repeat` + `ttnn.permute` + `ttnn.to_layout` + `ttnn.mul` + `ttnn.sum` | Reshape-based broadcast alternative |

---

**Next:** [Chapter 4 — Matmul Config and Math Fidelity](../ch4_matmul_config_and_math_fidelity/index.md)
