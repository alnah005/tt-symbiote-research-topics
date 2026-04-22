# Chapter 2 — TTNN Port Architecture

## Overview

This chapter maps the full directory layout of `models/demos/dots_ocr/` and explains the engineering decisions that shaped it. The port is organized around a two-stack design: a pure-PyTorch `reference/` stack that serves as a correctness oracle, and a `tt/` stack that implements the same computations on Tenstorrent hardware using TTNN ops. A shared PCC validation framework ties them together across 14 test files.

All file paths in this chapter are relative to `models/demos/dots_ocr/` unless explicitly prefixed otherwise.

## Reading Order

| File | Contents |
|------|----------|
| [`model_args_and_transformer.md`](./model_args_and_transformer.md) | `DotsModelArgs`, `DotsTransformer`, `Generator`, weight loading, special env vars |
| [`pcc_validation_framework.md`](./pcc_validation_framework.md) | All 14 test files and their roles, PCC calculation, targets achieved |

Read in order. This index establishes the directory layout and two-stack philosophy before the detail files.

## Annotated Directory Tree

```
models/demos/dots_ocr/
├── ARCHITECTURE.md                  # High-level architecture narrative
├── FULL_TTNN_VISION_PLAN.md         # Detailed vision encoder porting roadmap
├── IMPLEMENTATION_STEPS.md          # Step-by-step porting log with PCC milestones
├── README.md                        # Entry-point docs: setup, quickstart, env vars
│
├── demo/
│   ├── demo.py                      # Main demo: --backend ttnn or --backend hf
│   ├── pyth.py                      # Sandbox / prototype script
│   ├── reference_demo.py            # Pure HF PyTorch demo (no TTNN)
│   └── sample_prompts/
│       ├── README.md                # Prompt format documentation
│       ├── demo.json                # Multi-modal sample prompts (image + text)
│       └── text_only.json           # Text-only sample prompts
│
├── perf/
│   └── benchmark.py                 # TTFT, FPS, latency metrics; --backend both
│
├── reference/                       # Pure-PyTorch oracle stack (CPU-only, no TTNN)
│   ├── embeddings.py                # HF-compatible embedding layer (correctness oracle)
│   ├── fusion.py                    # Scatter fusion: vision tokens → text embedding positions
│   ├── hf_utils.py                  # HF model loading (HFLoadSpec, load_processor_and_model)
│   ├── model.py                     # DotsOCRReference wrapper: modular entry points, PCC oracle
│   ├── patch_merger.py              # Reference PatchMerger (PyTorch)
│   ├── pcc.py                       # PCC calculation utility (Pearson correlation coefficient)
│   ├── rope.py                      # Qwen2RopeHelper (HF-compatible RoPE)
│   └── vision.py                    # vision_tower_forward() helper
│
├── tt/                              # TTNN implementation stack
│   ├── _ttnn_import.py              # Lazy ttnn import: get_ttnn() returns None if not installed
│   ├── common.py                    # Shared utilities: get_block_size, get_max_prefill_chunk_size, num_blocks_in_seq
│   ├── fusion.py                    # TTNN scatter fusion (vision tokens → text embeddings on device)
│   ├── generator.py                 # Generator: wraps TTTGenerator, adds dots.ocr prefill/decode entry points
│   ├── load.py                      # load_dots_text_state_dict(), load_dots_vision_state_dict()
│   ├── mesh.py                      # open_dots_mesh_device(), close_dots_mesh_device()
│   ├── model.py                     # DotsTransformer (extends TTTransformer), DotsOCRModel
│   ├── model_config.py              # DotsModelArgs (extends ModelArgs)
│   ├── patch_merger.py              # PatchMergerTT (reused from qwen25_vl)
│   ├── vision.py                    # VisionEncoder (full TTNN by default)
│   ├── vision_attention.py          # TTNN vision self-attention
│   ├── vision_block.py              # VisionBlockTT (post-norm)
│   ├── vision_config_dataclass.py   # Vision config dataclass definitions
│   ├── vision_mlp.py                # VisionMLPTT (SwiGLU: gate/up/down projections)
│   ├── vision_model_config.py       # DotsVisionModelArgs
│   ├── vision_patch_embed.py        # PatchEmbedTT
│   └── vision_rmsnorm.py            # TTNN RMSNorm for vision blocks
│
└── tests/
    ├── conftest.py                          # Shared pytest fixtures and device setup
    ├── test_decoder_smoke.py                # Fast smoke: decoder forward pass (single token)
    ├── test_demo_hf_torch_only.py           # CPU-only demo path (no TTNN device)
    ├── test_e2e_pcc.py                      # End-to-end PCC: vision + text, single image prompt
    ├── test_environment.py                  # Hardware/software prerequisites check
    ├── test_fusion.py                       # Vision-to-text token fusion correctness
    ├── test_mesh_topology.py                # T3K submesh creation and teardown
    ├── test_patch_merger_pcc.py             # TTNN PatchMerger PCC vs reference
    ├── test_pcc_reference.py                # Full reference model PCC vs HF baseline
    ├── test_reference_embeddings.py         # reference/ embedding layer PCC vs HF
    ├── test_text_prefill_pcc.py             # TT text decoder PCC on prefill pass
    ├── test_vision_components.py            # Per-component vision PCC
    ├── test_vision_pcc.py                   # Vision ViT block-level PCC
    ├── test_vision_tower_pcc.py             # End-to-end vision tower PCC (all 42 blocks)
    └── test_weight_loading.py               # Checkpoint load (shape validation)
```

## Two-Stack Design Philosophy

The port separates concerns across two independent stacks that share only a data format boundary.

### The `reference/` Stack

The `reference/` stack is a pure-PyTorch reimplementation of the dots.ocr forward pass. It imports from HuggingFace `transformers` and runs entirely on CPU — no TTNN dependency, no device required. Its role is to serve as the ground truth for every PCC comparison:

- `reference/model.py` (`DotsOCRReference`) exposes modular entry points so individual subgraphs — the vision tower, the patch merger, the text embedding layer, the scatter fusion — can be exercised in isolation.
- `reference/pcc.py` provides the shared PCC calculation function consumed by all test files.
- `reference/hf_utils.py` handles HF model loading, abstracting `HFLoadSpec` and `load_processor_and_model` so the rest of the stack does not touch HuggingFace APIs directly.

Because the `reference/` stack has no TTNN dependency, it can be used to validate logic on any machine, including CI runners without TT hardware.

### The `tt/` Stack

The `tt/` stack maps each `reference/` module to a TTNN equivalent. The text decoder reuses the `tt_transformers` base classes (`ModelArgs`, `TTTransformer`, `TTTGenerator`) with thin subclassing. The vision encoder is implemented from scratch across `vision.py`, `vision_block.py`, `vision_attention.py`, `vision_mlp.py`, `vision_patch_embed.py`, and `vision_rmsnorm.py`.

The `tt/` stack introduces one important cross-cutting mechanism: the lazy TTNN import in `_ttnn_import.py`. The function `get_ttnn()` attempts to import TTNN at call time and returns `None` if the package is not installed. Every TTNN class guards its device-specific code paths behind a `get_ttnn()` check, which allows `reference/` modules and CPU-only test files to import from `tt/` without triggering import failures on machines where TTNN is not available.

### Data Boundary Between Stacks

The two stacks exchange data at the level of fused embedding tensors. See [`model_args_and_transformer.md`](./model_args_and_transformer.md) for the full `[B, S, D]` embedding handoff design.

### Reuse from `qwen25_vl`

`tt/patch_merger.py` is reused directly from the `qwen25_vl` port (`PatchMergerTT`). This is consistent with the architectural lineage described in Chapter 1: dots.ocr's PatchMerger is structurally identical to Qwen 2.5 VL's, so the TTNN implementation transfers without modification.

---

**Next:** [`model_args_and_transformer.md`](./model_args_and_transformer.md)
