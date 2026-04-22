# Research Guide Plan: dots.ocr on TT Hardware

**Topic:** dots.ocr on TT Hardware: Architecture, TTNN Port, and Relationship to Qwen 2.5 VL
**Branch under study:** `tenstorrent/tt-metal` @ `ign/dots_ocr`
**Model:** `rednote-hilab/dots.ocr` (1.7B multimodal document parser)
**Date planned:** 2026-04-22

---

## 1. Audience

**Primary audience:** ML engineers and systems integrators working on tt_symbiote who need to evaluate, adapt, or deploy the dots.ocr TTNN port in a production setting.

**What they already know:**
- Tenstorrent Wormhole hardware fundamentals (L1 memory, NoC, device mesh)
- TTNN tensor operations and the tt_transformers model infrastructure
- Transformer architecture (attention, MLP, RoPE, GQA)
- Basic Vision Transformer (ViT) concepts (patch embedding, positional encoding, class tokens)
- Qwen 2.5 VL at a high level (they may have read the companion guide on that model)

**What this guide teaches:**
- How dots.ocr extends and diverges from the Qwen 2.5 VL family
- The internal structure of the `ign/dots_ocr` TTNN port, component by component
- The topology constraint imposed by GQA and how the T3K submesh resolves it
- The current maturity level of the port and what work remains for tt_symbiote integration

---

## 2. Chapter List

---

### Chapter 1 — dots.ocr Model Architecture

**Description:** Establishes the full model specification of `DotsOCRForCausalLM`, covering both the text decoder and vision encoder, and precisely characterizes its relationship to and differences from Qwen 2.5 VL.

**Files:**

- `index.md`
  - Chapter overview and reading order
  - Summary table: dots.ocr vs Qwen 2.5 VL 7B side-by-side (hidden_size, layers, heads, GQA ratio, vision layers, patch size, temporal_patch_size, vocab size)
  - Key takeaway: dots.ocr is a compact (1.7B), document-specialized model sharing Qwen vocabulary and token IDs but with a distinctly smaller and differently-proportioned architecture

- `text_decoder_hyperparameters.md`
  - Full walkthrough of `config.json` text decoder fields: `hidden_size=1536`, `intermediate_size=8960`, `num_hidden_layers=28`, `num_attention_heads=12`, `num_key_value_heads=2`, `max_position_embeddings=131072`, `rope_theta=1000000`, `vocab_size=151936`, `attention_bias=True`
  - Explanation of the GQA ratio (12Q / 2KV = 6:1), what it implies for memory bandwidth vs Qwen2.5-VL-7B's 28Q/4KV
  - SwiGLU activation (`hidden_act="silu"` with gated projection), sliding window disabled (`use_sliding_window=False`)
  - Shared Qwen vocabulary: why `vocab_size=151936` and the same `image_token_id=151665` / `video_token_id=151656` are used, what that means for tokenizer compatibility
  - Comparison with `Qwen2ForCausalLM` base at the config level: which fields are identical, which are re-parameterized

- `vision_encoder_specs.md`
  - Full walkthrough of `vision_config`: `embed_dim=1536`, `hidden_size=1536`, `intermediate_size=4224`, `num_hidden_layers=42`, `num_attention_heads=12`, `patch_size=14`, `spatial_merge_size=2`, `temporal_patch_size=1`, `post_norm=True`, `rms_norm_eps=1e-5`
  - Why 42 ViT layers makes the vision encoder (~1.2B params) larger than the text decoder (~0.5B params) — unusual inversion compared to typical VLMs
  - `post_norm=True`: RMSNorm applied after attention and MLP (not pre-norm), implications for numerical stability and porting
  - `spatial_merge_size=2`: how the 2×2 spatial patch merge reduces the token count by 4× before the text decoder, formula for output token count given input image dimensions
  - `temporal_patch_size=1` vs Qwen3.6's `temporal_patch_size=2`: what this means for video handling (dots.ocr is image-only in practice)
  - Parameter count breakdown: patch embed, 42 ViT blocks (attention + MLP per block), post-norm, patch merger

- `relationship_to_qwen25vl.md`
  - Architectural lineage: which components are directly Qwen2-style (text decoder), which are shared-design (vision encoder structure), and which are novel or modified
  - Shared identifiers: vocabulary, image/video token IDs, `qwen_vl_utils` preprocessing pipeline
  - `use_hf_rope=True` in `DotsModelArgs`: why this matters (HF-compatible RoPE scaling vs a custom implementation), how it aligns with Qwen2's rotary position embedding
  - Key divergences: `hidden_size=1536` (Qwen2.5-VL-7B uses 3584), 28 decoder layers (same count as 7B variant but much narrower), `num_key_value_heads=2` (vs 4 in 7B), vision encoder depth (42 vs 27–32 in Qwen variants)
  - Why dots.ocr is not a fine-tune of Qwen 2.5 VL: different hidden dimensions mean weights cannot be loaded directly; it is architecturally derived, not a checkpoint fork

---

### Chapter 2 — TTNN Port Architecture

**Description:** Maps the directory layout of `models/demos/dots_ocr/` and explains how each stack (reference/, tt/) is structured, how DotsModelArgs and DotsTransformer extend the tt_transformers base classes, and how the 9-test-file PCC validation framework is organized.

**Files:**

- `index.md`
  - Chapter overview and reading order
  - Directory tree of `models/demos/dots_ocr/` with one-line annotations for every file
  - Two-stack design philosophy: `reference/` as a correctness oracle (pure PyTorch, matches HF outputs), `tt/` as the TTNN implementation; tests validate PCC between the two

- `model_args_and_transformer.md`
  - `DotsModelArgs` class: which fields it inherits from `ModelArgs` (tt_transformers), which it overrides or adds
  - Special env vars baked into `DotsModelArgs`: `DOTS_MAX_SEQ_LEN`, `DOTS_MAX_SEQ_LEN_WH_LB`, `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE` (default 2048), `trust_remote_code_hf=True`
  - Why `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE=2048` exists: vocab=151936 means the unsharded LM head would require ~300MB of L1 circular buffer per device; column capping prevents overflow on Wormhole
  - `DotsTransformer` extending `TTTransformer`: what is inherited unchanged (layer stack, KV cache management, attention via standard `Attention` class), what is overridden (weight loading hooks, mesh device routing)
  - `Generator` class in `tt/generator.py`: wraps `TTTGenerator` from tt_transformers, adds dots.ocr-specific prefill and decode entry points
  - `tt/load.py`: weight loading pipeline — how HF checkpoint weights are mapped to TTNN tensors, handling of `attention_bias=True` (Qwen2 adds bias to Q/K/V/O projections, tt_transformers typically does not)

- `pcc_validation_framework.md`
  - Overview of the 9 test files and what each validates:
    - `test_environment.py`: hardware/software prerequisites check
    - `test_weight_loading.py`: verifies checkpoint can be loaded into both reference and TT models without shape errors
    - `test_reference_embeddings.py`: reference/ embedding layer numerical correctness vs HF
    - `test_pcc_reference.py`: full reference model PCC vs HF baseline
    - `test_text_prefill_pcc.py`: TT text decoder PCC on prefill pass
    - `test_decoder_smoke.py`: fast smoke test for decoder forward pass (single token)
    - `test_patch_merger_pcc.py`: PCC of TTNN PatchMerger against reference
    - `test_vision_pcc.py`: per-component vision PCC (patch embed, individual blocks)
    - `test_vision_tower_pcc.py`: end-to-end vision tower PCC (all 42 blocks)
    - `test_fusion.py`: vision-to-text token fusion correctness
    - `test_e2e_pcc.py`: full model end-to-end PCC (vision + text, single image prompt)
    - `test_mesh_topology.py`: T3K submesh creation and teardown correctness
    - `test_demo_hf_torch_only.py`: CPU-only demo path (no TTNN device required)
  - PCC target achieved: prefill PCC > 0.98 (from commit history); framework targets > 0.99 per IMPLEMENTATION_STEPS.md
  - `reference/pcc.py`: PCC calculation utility (Pearson correlation on flattened tensors)

---

### Chapter 3 — Full TTNN Vision Stack

**Description:** Traces the evolution from the hybrid approach (HF vision_tower on host + TTNN PatchMerger) to the current full TTNN vision stack (all 42 ViT layers on device), covering each TTNN component, the PatchMerger reuse from qwen25_vl, and the fusion mechanism.

**Files:**

- `index.md`
  - Chapter overview and reading order
  - Component diagram: image pixels → PatchEmbedTT → 42× VisionBlockTT → post-norm → PatchMergerTT → scatter fusion → text embedding table
  - Why full TTNN was chosen: PCIe bandwidth savings (keeping 42 ViT layers on host would transfer large intermediate tensors per image), latency, and the availability of a complete TTNN plan (`FULL_TTNN_VISION_PLAN.md` in the branch)
  - Hybrid mode (`use_full_ttnn=False`): retained for CPU-only environments; when `mesh_device is not None` it is unconditionally overridden to `True`

- `vision_components_ttnn.md`
  - `tt/vision_patch_embed.py` — `PatchEmbedTT`: how the 14×14 patch convolution is expressed as a TTNN matmul, weight layout (TILE_LAYOUT, BFLOAT16), handling of `temporal_patch_size=1`
  - `tt/vision_block.py` — `VisionBlockTT`: the post-norm variant (RMSNorm after attention and after MLP, not before); why post-norm changes the forward pass ordering relative to standard pre-norm ViT
  - `tt/vision_attention.py` — attention within each ViT block: multi-head self-attention (12 heads, no GQA in vision), TTNN scaled dot product attention, positional encoding (no RoPE in vision — 2D learned or relative position bias)
  - `tt/vision_mlp.py` — MLP within each ViT block: `intermediate_size=4224`, SiLU activation, two-projection structure
  - `tt/vision_rmsnorm.py` — thin wrapper for TTNN RMSNorm used in `post_norm` context
  - `tt/vision_model_config.py` and `tt/vision_config_dataclass.py` — configuration plumbing for the TTNN vision stack (weights, mesh device assignment, dtype)
  - `tt/vision.py` — `VisionTransformerTT`: orchestrates all 42 blocks, applies post-norm after the final block, exposes `forward(pixel_values)` returning a flat sequence of visual feature tokens

- `patch_merger_and_fusion.md`
  - `tt/patch_merger.py` — `PatchMergerTT`: reuse from `models/demos/qwen25_vl/` patch merger; what "reuse" means concretely (same TTNN ops, same weight layout, same spatial_merge_size=2 logic)
  - How the 2×2 spatial merge works: 4 adjacent patch tokens → 1 merged token via learned linear projection, reducing the vision token count by 4× before the text decoder
  - `tt/fusion.py` — scatter fusion: how vision feature tokens are inserted into the text embedding sequence at positions marked by `image_token_id=151665`; the index-scatter operation on TTNN
  - `reference/fusion.py` — reference implementation of the same scatter; used as PCC oracle for `test_fusion.py`
  - Comparison with Qwen 2.5 VL fusion approach: same conceptual pattern (image token ID placeholders replaced by vision features), confirming that tt_symbiote integration can reuse the same dispatch logic

---

### Chapter 4 — T3K Topology and GQA Constraint

**Description:** Explains why `num_key_value_heads=2` mathematically caps tensor parallelism at TP≤2 on an 8-device T3K mesh, how the submesh approach resolves this, and what the key env vars and LM head memory budget imply for deployment.

**Files:**

- `index.md`
  - Chapter overview and reading order
  - Summary: the single biggest deployment constraint for dots.ocr on T3K is not compute or memory but GQA head count; understanding this prevents wasted debugging time
  - Quick-reference table: model config → TP constraint → submesh shape → effective device count for dots.ocr

- `gqa_tp_constraint.md`
  - Derivation of the TP bound: for correct GQA sharding, TP must divide both `num_attention_heads` (12) and `num_key_value_heads` (2); `gcd(12, 2) = 2`, so TP ∈ {1, 2}
  - What happens if TP > 2 is attempted: KV head sharding produces fewer than one KV head per device, causing either silent numerical errors or shape assertion failures
  - Comparison with Qwen 2.5 VL 7B (`num_key_value_heads=4`): gcd(28,4)=4, TP ≤ 4 — dots.ocr is more constrained despite being a smaller model
  - Why this is unusual: most models are designed so that `num_key_value_heads` scales with intended TP degree; dots.ocr's 2 KV heads appear to be an aggressive compression choice for parameter efficiency, not a topology consideration

- `t3k_submesh_and_env_vars.md`
  - T3K physical topology: 1×8 mesh (8 Wormhole N300 devices, Galaxy interconnect)
  - `tt/mesh.py`: `open_dots_mesh_device()` — opens the full 1×8 parent mesh unconditionally when `DOTS_T3K_OPEN_FULL_MESH=1` (default), then calls `create_submesh` to carve out a logical 1×2 or 1×1 sub-device group
  - `DOTS_T3K_TP` env var: controls submesh width (1 or 2); maps directly to TP degree
  - `close_dots_mesh_device()`: tears down both the submesh and the parent mesh in the correct order (submesh first, then parent)
  - Why opening the full mesh first: Galaxy interconnect requires the host to claim all 8 devices before sub-allocating; partial-open would leave the remaining 6 devices in an undefined state for other processes
  - Practical deployment note: on a shared T3K server, other models (e.g., Llama, Mistral) that need TP=8 cannot co-run with dots.ocr holding the full mesh open; scheduling implications for tt_symbiote
  - `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE=2048` revisited in topology context: with TP=2 and vocab=151936, each device would handle ~75968 columns unsharded; capping at 2048 means the LM head is computed in 37 column-sliced TTNN ops per device per forward pass — trade-off between kernel launches and L1 overflow

- `chunked_prefill.md`
  - Why long sequences require chunked prefill: `max_position_embeddings=131072` means a document image can produce thousands of vision tokens; a single prefill pass for a long sequence may exceed on-device SRAM for activations
  - "PC decode" commit: what prefill-chunked (PC) decode means — splitting the prefill phase into fixed-length chunks that fit in L1, then switching to autoregressive decode
  - `DOTS_MAX_SEQ_LEN` and `DOTS_MAX_SEQ_LEN_WH_LB` env vars: how the first caps the absolute sequence length accepted, and how the second (lower bound) sets the minimum chunk size for prefill
  - Interaction with T3K submesh: chunked prefill must be consistent across both devices in the TP=2 submesh (same chunk boundaries, synchronized via Galaxy link)
  - Expected latency impact: TTFT (time-to-first-token) scales with number of prefill chunks; benchmark.py measures this as part of the TTFT metric

---

### Chapter 5 — Implementation Status and Deployment

**Description:** Audits the current state of the port against the six implementation steps, maps each stabilizing commit to what it fixed, presents PCC results and benchmark methodology, explains demo usage, and identifies the remaining gaps for tt_symbiote production integration.

**Files:**

- `index.md`
  - Chapter overview and reading order
  - Status dashboard: all 6 IMPLEMENTATION_STEPS complete as of latest commit; summary of what "complete" means per step and what production-ready requires beyond completion
  - Recommendation summary for tt_symbiote integrators: what to verify, what to configure, what to test

- `commit_history_and_stabilization.md`
  - Walkthrough of the 6 commits from oldest to newest, explaining what each stabilized:
    1. "Basic code for dots ocr": initial skeleton port, no PCC validation, Qwen class names still present
    2. "Partial mesh support": T3K submesh creation, `DOTS_T3K_OPEN_FULL_MESH` logic, `close_dots_mesh_device` teardown
    3. "prefill at 0.98": text decoder prefill PCC crossed 0.98 threshold; what was likely fixed (RoPE alignment, weight mapping for `attention_bias=True`)
    4. "Changes to support PC decode": chunked prefill added to `tt/generator.py`, `DOTS_MAX_SEQ_LEN_WH_LB` env var introduced
    5. "Demo works with vision_backbone hf": interim hybrid demo confirmed working end-to-end; full TTNN vision not yet complete at this point
    6. "Intermediate changes removing qwen reference": class renaming (DotsTransformer, DotsModelArgs replacing any residual Qwen* names), confirming the refactor is in-progress but not necessarily complete
  - What the commit history does NOT tell us: no explicit commit for "full TTNN vision complete" — this may be embedded in the renaming commit or may still be in-progress despite IMPLEMENTATION_STEPS.md claiming Step 4 complete
  - Implication: integrators should run `test_vision_tower_pcc.py` and `test_e2e_pcc.py` to independently confirm the full TTNN vision stack is numerically stable before relying on IMPLEMENTATION_STEPS.md status

- `pcc_results_and_benchmarks.md`
  - PCC results summary:
    - Prefill PCC: > 0.98 (confirmed by commit), framework targets > 0.99
    - Per-component (patch merger, vision blocks, decoder): > 0.99 per IMPLEMENTATION_STEPS.md
    - End-to-end (test_e2e_pcc.py): target > 0.99, actual result to be confirmed by running tests
  - `perf/benchmark.py` methodology: what metrics it measures (TTFT in ms, FPS tokens/sec, per-token decode latency in ms), how to run it, expected invocation flags
  - `demo/demo.py` usage: `--backend ttnn` flag selects the TTNN path; `--backend hf` selects the CPU reference path; required environment variables before running
  - `demo/reference_demo.py`: pure HF PyTorch demo for correctness comparison without any TTNN dependency
  - `demo/pyth.py`: likely a sandbox or prototype script (name suggests Python-only experimentation); its role relative to the main demo
  - Sample prompts in `demo/sample_prompts/`: what document types are covered, how to add custom prompts for tt_symbiote use cases

- `tt_symbiote_integration_gaps.md`
  - What "production-ready" means beyond IMPLEMENTATION_STEPS.md: sustained throughput under load, graceful error handling, tt_symbiote API surface compatibility
  - Open questions from the commit history:
    - Is the "removing qwen reference" refactor complete, or are there residual `Qwen*` imports that could cause `trust_remote_code` issues?
    - Has the full TTNN vision stack been validated at > 0.99 PCC end-to-end, or is that still a target?
    - Is the demo tested with a real T3K board, or only in simulation/single-device mode?
  - Integration checklist for tt_symbiote:
    - Confirm `trust_remote_code_hf=True` is set in the tt_symbiote model registry entry
    - Set `DOTS_T3K_OPEN_FULL_MESH=1` and `DOTS_T3K_TP=2` in the deployment environment
    - Set `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE=2048` to prevent L1 overflow
    - Run the full test suite (all 9+ test files) on target hardware before declaring production-ready
    - Decide on mesh scheduling policy (can dots.ocr's full-mesh-open coexist with other models on the same T3K server)
  - What is definitively working: text decoder prefill and decode on TTNN, PatchMerger on TTNN, T3K submesh lifecycle, weight loading, chunked prefill
  - What requires verification: full TTNN vision tower PCC at production quality, demo end-to-end on real T3K hardware, throughput benchmarks against HF baseline

---

## 3. Conventions

**Terminology:**
- "dots.ocr" (lowercase with period): the model name as used in `rednote-hilab/dots.ocr` — never "DotsOCR" when referring to the model as a whole; reserve `DotsOCRForCausalLM` for the Python class name
- "ViT" or "vision encoder": the 42-layer transformer that processes image patches; avoid "vision backbone" (that term appears in the interim commit but is not the primary naming in the final port)
- "text decoder": the 28-layer Qwen2-style autoregressive transformer; distinguish clearly from "decoder" in the encoder-decoder sense (dots.ocr is decoder-only)
- "TTNN": the Tenstorrent Neural Network library; always all-caps; "TTNN op" for a single tensor operation, "TTNN stack" for the full implementation
- "tt_transformers": the shared infrastructure library at `models/tt_transformers/`; never "tt-transformers" (hyphenated) in code contexts
- "T3K": Tenstorrent 3000 series, 8-chip Wormhole cluster; always capitalized as T3K, never "t3k" or "T3k"
- "TP": tensor parallelism degree; always written as "TP=N" (e.g., TP=2)
- "GQA": grouped-query attention; write as "GQA 12Q/2KV" when giving the exact configuration
- "PCC": Pearson correlation coefficient; used as the numerical accuracy metric throughout the port; always written PCC ∈ (0, 1) with higher being better
- "hybrid mode": the configuration where the HF vision_tower runs on CPU host and only PatchMerger runs on TTNN device; contrast with "full TTNN mode"

**Notation:**
- Config fields: written in backtick code style matching the JSON key name, e.g., `num_key_value_heads`
- File paths: always relative to `models/demos/dots_ocr/` unless explicitly prefixed with `models/` for cross-demo references
- Environment variables: written in `SCREAMING_SNAKE_CASE` with backticks, e.g., `DOTS_T3K_TP`
- Equations: write tensor parallelism constraint derivation as `TP | gcd(num_attention_heads, num_key_value_heads)`
- Class names: always in code font, e.g., `DotsModelArgs`, `VisionTransformerTT`, `PatchMergerTT`

**Formatting rules:**
- Each content file starts with a `## Overview` section (2–4 sentences) before diving into details
- Use `### Subheading` for major topics within a file; use `#### Detail heading` sparingly, only for technical deep-dives
- Code blocks: use `python` syntax highlighting for Python, `json` for config files, `bash` for shell commands
- Tables: use Markdown tables for comparisons (dots.ocr vs Qwen 2.5 VL, step-by-step status, env var reference)
- Admonitions: use `> **Note:**` for important caveats, `> **Warning:**` for things that will cause runtime errors if missed
- No emojis in any guide content file
- Cross-references: written as `[Chapter N — Title](../chapterN/index.md)` with relative paths

---

## 4. Cross-Chapter Dependencies

- **Chapter 2 depends on Chapter 1**: `DotsModelArgs` and `DotsTransformer` can only be understood after the reader knows the model's hyperparameters (hidden_size, layer counts, GQA config). Chapter 2 files should reference back to Chapter 1 for config field definitions rather than re-explaining them.

- **Chapter 3 depends on Chapter 1**: The vision component files (PatchEmbedTT, VisionBlockTT, etc.) map directly to the vision_config fields explained in Chapter 1. Specifically, `post_norm=True`, `patch_size=14`, and `spatial_merge_size=2` are prerequisites for understanding Chapter 3's TTNN implementation choices.

- **Chapter 4 depends on Chapter 1 and Chapter 2**: The GQA constraint derivation requires knowing `num_attention_heads=12` and `num_key_value_heads=2` (Chapter 1). The env var discussion references `DotsModelArgs` fields like `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE` (Chapter 2).

- **Chapter 5 depends on Chapters 1–4**: The integration gap analysis requires knowing the full architecture (Ch. 1), the port structure (Ch. 2), the vision stack status (Ch. 3), and the topology constraints (Ch. 4). Chapter 5 files should cite specific test files from Chapter 2's PCC framework.

- **No backward dependencies**: Chapters 1 and 3 have no dependencies on later chapters. Chapter 4 does not depend on Chapter 3 (topology is independent of the vision implementation). Chapter 5 is the only chapter that requires all prior chapters.

- **Qwen 2.5 VL companion guide**: Where the text refers to Qwen 2.5 VL architecture details (e.g., vision encoder comparison, patch merger reuse), authors should cross-reference the Qwen 2.5 VL guide rather than re-explaining Qwen internals. The relationship is described locally in Chapter 1 (`relationship_to_qwen25vl.md`) and Chapter 3 (`patch_merger_and_fusion.md`).
