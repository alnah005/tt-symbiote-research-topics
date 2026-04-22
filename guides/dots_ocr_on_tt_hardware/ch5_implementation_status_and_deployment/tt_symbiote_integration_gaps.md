# TT-Symbiote Integration Gaps

## Overview

"Production-ready" for tt_symbiote integration requires more than passing `IMPLEMENTATION_STEPS.md` checkboxes: it requires confirmed PCC on real T3K hardware, verified absence of residual class-name artifacts from the renaming cleanup, and validated resource accounting for an 8-device workload. This file enumerates three open questions that the commit history leaves unresolved, a concrete integration checklist, and a clear separation between what is definitively working and what still requires verification before production deployment.

### Open Questions

Three questions remain unresolved after analysis of the six-commit branch history:

1. **Is the Step 6 "removing qwen reference" renaming complete?** The commit message reads "Intermediate changes removing qwen reference." The word "Intermediate" explicitly flags that the sweep is not finished as of this commit. See the Commit 6 warning in `commit_history_and_stabilization.md` for the HuggingFace class resolution failure mechanism. A manual `grep` audit of `tt/` and `reference/` for any remaining `Qwen*` symbols must be performed before tt_symbiote integration.

2. **Has the full TTNN vision stack (`use_full_ttnn=True`) been validated at PCC > 0.99 end-to-end?** As noted in `commit_history_and_stabilization.md` ("What the Commit History Does Not Tell Us"), no commit message records a PCC threshold crossing for the 42-layer `VisionTransformerTT`. Until `test_vision_tower_pcc.py` and `test_e2e_pcc.py` are run on real T3K hardware and produce PCC > 0.99, Step 4 should be treated as claimed but unconfirmed.

3. **Has the demo been validated on real T3K hardware, or only in simulation or single-device mode?** Commit 5 confirms the hybrid demo works (HF vision on CPU + TTNN PatchMerger + TTNN text decoder), but it does not specify whether this was validated on a physical T3K system. Full TTNN end-to-end operation on T3K — with `DOTS_T3K_OPEN_FULL_MESH=1` and the 1x2 submesh active — has not been separately confirmed by any commit. Simulation or single-device validation does not cover the mesh lifecycle paths exercised by `open_dots_mesh_device()` and `close_dots_mesh_device()`.

### Integration Checklist for tt_symbiote

| Item | Action | Verified? |
|------|--------|-----------|
| Model registry | Set `trust_remote_code_hf=True` in the tt_symbiote model registry entry for `dots_ocr` | Required |
| Env vars | Set `DOTS_T3K_OPEN_FULL_MESH=1`, `DOTS_T3K_TP=2` before device initialization | Required |
| LM head budget | Set `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE=2048` | Required |
| Scheduling | Register dots.ocr as an 8-device workload in tt_symbiote resource accounting (all 8 T3K devices claimed when `DOTS_T3K_OPEN_FULL_MESH=1`) | Required |
| Renaming audit | `grep` for any residual `Qwen*` imports and class references in `tt/` and `reference/` | Recommended before production |
| PCC validation | Run all 14 tests on target T3K hardware in the order specified in `pcc_results_and_benchmarks.md` | Required before production |
| E2E demo | Run `demo/demo.py --backend ttnn` with a real OCR document on T3K hardware | Recommended |
| Throughput | Run `perf/benchmark.py` on T3K and record TTFT baseline; tune `DOTS_MAX_SEQ_LEN_WH_LB` | Recommended |

> **Warning:** The "Required" items in the checklist above are not optional hardening steps — they are correctness requirements. Omitting `trust_remote_code_hf=True` will cause the model to fail to load. Omitting `DOTS_T3K_OPEN_FULL_MESH=1` will prevent the submesh from being carved correctly. Failing to register dots.ocr as an 8-device workload will cause resource contention with other models scheduled on the same T3K system.

### What Is Definitively Working

The following capabilities are established by the commit history and are stable:

- **Text decoder prefill and decode on TTNN** (PCC > 0.98 confirmed by commit 3). The 28-layer GQA decoder with `hidden_size=1536`, `attention_bias=True`, and `rope_theta=1e6` runs on the TTNN path with measured prefill PCC above 0.98 against the HF reference.
- **`PatchMergerTT` on TTNN** (structurally reused from the `qwen25_vl` demo and independently tested via `test_patch_merger_pcc.py`). The spatial merge layer with `spatial_merge_size=2` operates correctly on the TTNN device.
- **T3K submesh lifecycle** (open full mesh, carve 1x2 submesh, forward pass, teardown; covered by `test_mesh_topology.py`). The `open_dots_mesh_device()` and `close_dots_mesh_device()` functions implement correct teardown order and are validated by commit 2.
- **Weight loading with `attention_bias=True`** (Q/K/V/O bias tensors handled in `tt/load.py`). The bias tensors are loaded from the dots.ocr checkpoint and applied correctly in the TTNN attention forward pass.
- **Chunked prefill loop** (`Generator.prefill_forward_text()` with `DOTS_MAX_SEQ_LEN_WH_LB`). The prefill-chunked decode path is implemented and the env var tuning interface is available.
- **Hybrid demo end-to-end** (HF vision on CPU + `PatchMergerTT` on TTNN + TTNN text decoder). `demo/demo.py --backend ttnn` works in hybrid mode; `demo/demo.py --backend hf` works as a CPU reference path.

### What Requires Verification Before Production

The following items are claimed or structurally present in the codebase but have not been confirmed by commit history on real T3K hardware:

- **Full TTNN vision tower PCC at or above 0.99** (`test_vision_tower_pcc.py` on real T3K hardware). Until this test passes at PCC > 0.99, Step 4 of `IMPLEMENTATION_STEPS.md` cannot be treated as confirmed. The 42-layer `VisionTransformerTT` with `post_norm=True` and `rms_norm_eps=1e-5` has not been measured against a PCC threshold in any commit message.
- **End-to-end PCC with full TTNN vision stack** (`test_e2e_pcc.py`). A full forward pass from raw image input through the 42-layer ViT, `PatchMergerTT`, and 28-layer text decoder, compared against the HF reference at PCC > 0.99, has not been recorded in the commit history.
- **Complete absence of `Qwen*` class names in the TTNN path.** Commit 6 began but did not complete the renaming sweep. The audit must be run manually: any surviving `Qwen*` symbol in `tt/` or `reference/` is a production blocker.
- **Throughput benchmarks vs. HF baseline on real T3K hardware.** TTFT (ms), decode throughput (tokens/sec), and per-token decode latency have not been recorded in the commit history. `perf/benchmark.py` exists and measures these metrics, but no baseline has been established.
- **Sustained load behavior** (no device handle leaks, clean teardown under repeated requests). The mesh lifecycle has been tested for single-session teardown, but no commit records a repeated-request or multi-session stress test that would surface handle leaks or non-deterministic teardown failures.
