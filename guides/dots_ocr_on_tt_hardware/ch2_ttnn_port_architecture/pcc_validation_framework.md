# PCC Validation Framework

This file covers the test suite in `tests/`, explaining what each of the 14 test files validates and how they use the PCC (Pearson Correlation Coefficient) metric to establish correctness of the TTNN port against the `reference/` oracle.

---

## PCC Calculation (`reference/pcc.py`)

All PCC comparisons in the test suite are computed by the shared utility in `reference/pcc.py`. The calculation operates on two tensors $A$ and $B$ of the same shape, flattened to 1-D vectors $\mathbf{a}$ and $\mathbf{b}$:

$$\text{PCC}(\mathbf{a}, \mathbf{b}) = \frac{\sum_i (a_i - \bar{a})(b_i - \bar{b})}{\sqrt{\sum_i (a_i - \bar{a})^2} \cdot \sqrt{\sum_i (b_i - \bar{b})^2}}$$

The result is a scalar in $[-1, 1]$, where $1.0$ indicates perfect linear agreement. In practice, numerical divergence due to BF16 accumulation, tile padding, and distributed sharding keeps PCC below $1.0$ even for correct implementations. The targets below reflect empirically validated thresholds rather than theoretical bounds.

### PCC Targets

| Scope | Target | Source / Notes |
|-------|--------|----------------|
| Text prefill (per-layer logits) | > 0.98 | Commit history ("prefill at 0.98"); logits at the final LM head accumulate more error than intermediate activations |
| All components (general claim) | > 0.99 | `IMPLEMENTATION_STEPS.md` — stated target, not independently confirmed by commit history |

---

## Test File Reference

The 14 test files span four categories: environment/infrastructure, reference-stack validation, component-level TTNN PCC, and end-to-end TTNN PCC.

### Shared Infrastructure

#### `tests/conftest.py`

Shared pytest fixtures and device setup. Provides the `mesh_device` fixture (calls `open_dots_mesh_device()` from `tt/mesh.py` and tears it down after the test session), the `model_args` fixture (instantiates `DotsModelArgs`), and the `hf_model` fixture (calls `load_processor_and_model()` from `reference/hf_utils.py`). Tests that require a TT device receive these fixtures; CPU-only tests skip device fixtures.

#### `tests/test_environment.py`

Checks hardware and software prerequisites before any TTNN test runs. Validates that the expected TT device is present and reachable, that the TTNN package version is compatible, and that the dots.ocr checkpoint path is set and readable. Intended to be the first test run in a new deployment so that infrastructure failures are caught before spending time on model tests.

#### `tests/test_mesh_topology.py`

Tests T3K submesh creation and teardown using `open_dots_mesh_device()` and `close_dots_mesh_device()` from `tt/mesh.py`. Validates that the correct number of devices is enumerated, that the submesh shape matches the expected T3K configuration, and that teardown completes cleanly without device handle leaks.

---

### Reference Stack Validation

These tests validate the `reference/` stack itself against the HF baseline. They run CPU-only and do not require a TT device.

#### `tests/test_pcc_reference.py`

Runs the full `DotsOCRReference` wrapper from `reference/model.py` on a sample input and compares its output logits to the HuggingFace `AutoModelForCausalLM` forward pass. This confirms that the reference stack is a faithful reimplementation of the HF model before any TTNN comparison is attempted. A failing PCC here indicates a bug in `reference/`, not in `tt/`.

#### `tests/test_reference_embeddings.py`

Compares the `reference/embeddings.py` embedding layer output to the HF embedding layer for the same input token IDs; isolated from the rest of the model so that embedding-layer divergence is detectable independently of attention and MLP layers. TTNN path: additionally validates that when the text decoder embedding table is loaded via `load_dots_text_state_dict()` and evaluated through the TTNN path, the output PCC against the HF embedding layer remains above threshold.

#### `tests/test_demo_hf_torch_only.py`

Runs the CPU-only demo path through `demo/demo.py` with `--backend hf`, verifying that the full HF demo pipeline (image loading, processor, model forward, detokenization) executes correctly without a TT device. This is a regression guard for the demo entry point itself.

---

### Component-Level TTNN PCC

These tests validate individual subgraphs of the TTNN port in isolation. Each test loads the relevant TTNN module, runs a forward pass on a controlled input, runs the equivalent `reference/` module on the same input, and asserts PCC above threshold.

#### `tests/test_weight_loading.py`

Calls `load_dots_text_state_dict()` and `load_dots_vision_state_dict()` from `tt/load.py` and validates that every loaded tensor has the expected shape. Does not run a forward pass. A failing shape check indicates a key-remapping error in the loader or a checkpoint format change.

#### `tests/test_fusion.py`

Validates the TTNN scatter fusion in `tt/fusion.py` against the reference scatter fusion in `reference/fusion.py`. The fusion operation scatters vision patch embeddings into the text embedding positions indicated by the image token IDs, producing the `[B, S, D]` fused embedding tensor consumed by `DotsTransformer.prepare_inputs_prefill()`. PCC is computed on the full fused tensor.

#### `tests/test_patch_merger_pcc.py`

Compares `PatchMergerTT` from `tt/patch_merger.py` against the reference `PatchMerger` from `reference/patch_merger.py`. The PatchMerger spatially merges adjacent vision tokens (merge factor 2×2) before they are handed to the text decoder. Because `PatchMergerTT` is reused from `qwen25_vl`, this test confirms that the reused implementation is numerically consistent with the dots.ocr reference path.

#### `tests/test_decoder_smoke.py`

A fast smoke test: runs a single-token decoder forward pass through `DotsTransformer` and asserts that the output is a valid tensor of the expected shape. Does not assert PCC. Intended as a CI gate that catches device-level failures (buffer allocation errors, shape mismatches, kernel compilation failures) without the overhead of a full PCC run.

#### `tests/test_text_prefill_pcc.py`

Runs the full text prefill pass through `DotsTransformer` on a token sequence of meaningful length, then compares the output logits to the reference model. This is the primary correctness test for the text decoder. The PCC target is > 0.98 (from the commit-message milestone). The test exercises `DotsModelArgs`, `load_dots_text_state_dict()`, `DotsTransformer.prepare_inputs_prefill()`, and the chunked prefill loop in `Generator.prefill_forward_text()`.

#### `tests/test_vision_components.py`

Runs each vision submodule — `VisionMLPTT`, `VisionAttention`, `vision_rmsnorm`, `PatchEmbedTT` — individually against the corresponding reference path. PCC is asserted per component. This level of granularity allows failures to be localized to a specific op without running the full vision tower.

#### `tests/test_vision_pcc.py`

Validates a single `VisionBlockTT` from `tt/vision_block.py` against the reference vision block. A VisionBlockTT applies pre-norm attention followed by post-norm MLP (post-norm is a dots.ocr divergence from standard ViT; see Chapter 1). PCC is computed on the block output.

---

### End-to-End TTNN PCC

These tests run the largest subgraph or the full model pipeline end-to-end.

#### `tests/test_vision_tower_pcc.py`

Runs the full vision tower through all 42 `VisionBlockTT` layers using `VisionEncoder` from `tt/vision.py`, then compares the final vision encoder output to `reference/vision.py`'s `vision_tower_forward()` output. This is the primary end-to-end correctness test for the vision encoder. The stated target is PCC > 0.99 (per `IMPLEMENTATION_STEPS.md`); this figure has not been independently confirmed by commit history. Running all 42 blocks on a T3K mesh exercises the full vision model compilation and tensor-parallel sharding strategy.

#### `tests/test_e2e_pcc.py`

End-to-end test: takes a single image prompt, runs vision encoding → patch merging → scatter fusion → text prefill through the complete TTNN pipeline, and compares the final logits to the HF model's output on the same input. This is the highest-level correctness assertion. Same unconfirmed > 0.99 target as `test_vision_tower_pcc.py`; the only confirmed PCC figure from commit history is > 0.98 for text prefill. Reaching the > 0.99 target end-to-end would mean the full multimodal forward pass on TT hardware agrees with the HF baseline within that tolerance.

---

## Test Dependency Graph

The tests form an implicit dependency chain from infrastructure upward to end-to-end:

```
test_environment.py
    └── test_mesh_topology.py
            └── test_weight_loading.py
                    ├── test_decoder_smoke.py
                    │       └── test_text_prefill_pcc.py
                    │               └── test_e2e_pcc.py
                    ├── test_fusion.py
                    │       └── test_e2e_pcc.py
                    └── test_vision_components.py
                            └── test_vision_pcc.py
                                    └── test_vision_tower_pcc.py
                                            └── test_e2e_pcc.py

test_pcc_reference.py (CPU-only, no device dependency)
test_reference_embeddings.py (CPU-only, no device dependency)
test_demo_hf_torch_only.py (CPU-only, no device dependency)
test_patch_merger_pcc.py (device required)
```

When a failure is observed in `test_e2e_pcc.py`, the graph shows which lower-level tests to run first. A failure in `test_weight_loading.py` invalidates all tests above it; a failure in `test_vision_tower_pcc.py` localizes to the vision encoder and does not implicate the text decoder.

---

**Next:** [Chapter 3 — Full TTNN Vision Stack](../ch3_full_ttnn_vision_stack/index.md)
