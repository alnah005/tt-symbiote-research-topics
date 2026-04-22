# Commit History and Stabilization

## Overview

The `ign/dots_ocr` branch carries six commits, progressing from an initial skeleton port to an in-progress renaming cleanup. Each commit stabilized a discrete subsystem: device mesh lifecycle, text decoder PCC, chunked prefill, hybrid demo, and class-name cleanup. The analysis below maps what each commit introduced and, equally importantly, what each commit left open or deferred to a subsequent commit.

### Commit 1: "Basic code for dots ocr"

**What it introduced:**
- Initial skeleton of the TTNN port: directory structure under `tt/` and `reference/`, placeholder model files, and a first pass at weight-loading scaffolding.
- Class names at this stage were inherited from the Qwen2.5-VL codebase (`QwenTransformer`, `QwenModelArgs`, and similar); no dots.ocr-specific naming had been applied.
- `DotsModelArgs` fields were present in skeleton form but the `trust_remote_code_hf` post-init assignment and `dummy_weights=False` override were not yet finalized.

**What it left open:**
- No PCC validation of any kind; tests existed as empty stubs or were not yet written.
- Mesh device routing was present as placeholders; `open_dots_mesh_device()` and submesh carving logic had not been implemented.
- Weight loading for `attention_bias=True` (Q/K/V/O bias tensors) was not handled.

### Commit 2: "Partial mesh support"

**What it introduced:**
- `open_dots_mesh_device()` and `close_dots_mesh_device()` functions with the correct teardown order: submesh is released before the full mesh is closed, preventing device handle leaks.
- `DOTS_T3K_OPEN_FULL_MESH` environment variable: when set to `1`, all 8 T3K devices are claimed by the parent mesh before the 1x2 submesh is carved from it.
- T3K submesh carving: the 1x2 active submesh is positioned within the 1x8 parent mesh; `DOTS_T3K_TP` controls the tensor-parallel width (maximum 2, constrained by GQA with 12 query heads and 2 KV heads giving `gcd(12,2)=2`).
- `test_mesh_topology.py` became meaningful as a functional test.

**What it left open:**
- Text decoder PCC was untested at this point; the mesh was wired up but the forward pass had not been validated for numerical correctness.
- Vision stack was not yet connected to the mesh.

### Commit 3: "prefill at 0.98"

**What it introduced:**
- Text decoder prefill PCC crossing 0.98 against the HF reference. This is the only PCC milestone that is directly confirmed by a commit message in the branch history.
- The changes that most likely produced the PCC improvement (inferred from the scope of a 0.98 crossing):
  - **RoPE alignment:** `use_hf_rope=True` flag correctly aligned the rotary embedding implementation in `tt/` with HuggingFace's convention; a mismatch here produces divergent attention scores across all layers.
  - **Attention bias weight loading:** `attention_bias=True` requires that Q, K, V, and O projection bias tensors are loaded from the checkpoint and applied in the forward pass; these were added to `tt/load.py` at this commit.
  - **`DotsModelArgs` post-init timing:** `trust_remote_code_hf=True` must be assigned after `__init__` completes (post-init pattern) to avoid being overwritten by a dataclass default; the timing was corrected here.
- `test_text_prefill_pcc.py` became a passing test.

**What it left open:**
- Per-component vision PCC and end-to-end PCC were not part of this commit.
- Chunked prefill had not been introduced; prefill was single-shot.

### Commit 4: "Changes to support PC decode"

**What it introduced:**
- Chunked prefill loop in `Generator.prefill_forward_text()`: the prefill sequence is processed in windows of `max_prefill_chunk_size` rather than as a single forward pass. "PC decode" stands for prefill-chunked decode — prefill runs in chunks, followed by standard autoregressive decode.
- `DOTS_MAX_SEQ_LEN_WH_LB` environment variable: controls the lower bound on the sequence length for which the chunked window height/width is computed. This affects L1 SRAM pressure during prefill; a higher value reduces the number of chunks but increases per-chunk memory usage.
- `DOTS_MAX_SEQ_LEN` for the maximum total sequence length the model will handle.

**What it left open:**
- Vision stack integration was still absent from the chunked path; the chunked prefill applied to the text decoder only.
- The hybrid demo (HF vision + TTNN decoder) had not been validated end-to-end at this point.

### Commit 5: "Demo works with vision_backbone hf"

**What it introduced:**
- Hybrid mode end-to-end demo confirmed working: the HF `vision_tower` runs on CPU (host PyTorch), its output patch embeddings are transferred to the TTNN device, `PatchMergerTT` merges them, and the TTNN text decoder processes the merged sequence.
- `demo/demo.py --backend hf` path validated: CPU reference demo runs correctly.
- `demo/demo.py --backend ttnn` path working in hybrid configuration: TTNN path operational with the HF vision backbone as the image encoder.

> **Note:** At this commit, `use_full_ttnn=True` (all 42 ViT layers on device) had not been confirmed stable. The working path was hybrid mode: HF vision on CPU, `PatchMergerTT` and text decoder on TTNN.

**What it left open:**
- Full TTNN vision tower (`VisionTransformerTT`, all 42 layers on device) not confirmed working at PCC > 0.99 end-to-end.
- `test_vision_tower_pcc.py` and `test_e2e_pcc.py` status against real T3K hardware is not established by this commit.

### Commit 6: "Intermediate changes removing qwen reference"

**What it introduced:**
- Class renaming sweep across `tt/` and `reference/`: replacing residual `Qwen*` class names with `DotsTransformer`, `DotsModelArgs`, and other dots.ocr-specific names.
- This commit began the cleanup started in IMPLEMENTATION_STEPS.md Step 6 (cleanup and renaming).

> **Warning:** The word "Intermediate" in the commit message is an explicit signal that this renaming sweep is not complete as of this commit. Residual `Qwen*` imports or class references may remain in `tt/` or `reference/`. Any surviving `Qwen*` name in the TTNN path will cause a `trust_remote_code` failure in HuggingFace's dynamic class resolution, because the registry entry for `dots_ocr` does not register Qwen class names.

**What it left open:**
- Whether the full TTNN vision stack (Step 4) is embedded in this commit or deferred to a follow-on commit is not determinable from the message alone.
- The renaming audit (`grep` for `Qwen*` across `tt/` and `reference/`) must be performed manually before production integration.

### What the Commit History Does Not Tell Us

Three questions that the six-commit history leaves unanswered:

1. **No explicit "full TTNN vision complete" commit.** Step 4 of `IMPLEMENTATION_STEPS.md` claims the full TTNN vision stack is complete, but no commit message records a PCC threshold crossing or a "vision tower on device confirmed" milestone. The Step 4 work may be embedded within commit 6 ("Intermediate changes"), or it may be pending a seventh commit that has not yet landed.

2. **No explicit "PCC > 0.99 confirmed" commit.** The target stated in `IMPLEMENTATION_STEPS.md` is PCC > 0.99 across all components. The commit history confirms only PCC > 0.98 for the text decoder prefill (commit 3). Whether any component has reached 0.99 is not attributable to a specific commit.

3. **No benchmark result commit.** Metrics such as TTFT (ms), decode throughput (tokens/sec), and per-token decode latency are measurable via `perf/benchmark.py`, but no commit records a baseline benchmark run or attaches numbers to a particular hardware configuration. Any TTFT or throughput figures cited outside of a benchmark run on real T3K hardware should be treated as unverified estimates.

**Next:** [PCC Results and Benchmarks](pcc_results_and_benchmarks.md)
