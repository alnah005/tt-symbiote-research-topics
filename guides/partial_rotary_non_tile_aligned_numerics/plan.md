# Plan: Partial Rotary Embedding Numerical Correctness for Non-Tile-Aligned rotary_dim in TTNN

---

## 1. Audience

**Primary audience:** ML framework engineers working on the tt-symbiote / tt-transformers stack who are porting or debugging models that use partial RoPE (`partial_rotary_factor < 1.0`) when `rotary_dim` does not divide evenly by 32 (the TTNN tile width). They have encountered PCC ~0.71 against a PyTorch reference for a configuration like `rotary_dim=48, head_dim=128` even without Metal Trace, and need to understand the root cause and correct fix.

**What they already know:**

- Standard and partial Rotary Position Embedding (RoPE): how cos/sin tables are precomputed, how the rotate-half operation applies them to query/key vectors, and what `rotary_dim` vs. `head_dim` mean
- The TTNN tile layout and the 32-element tile width (`TILE_WIDTH = 32`); that on-device tensors in TILE layout must have dimensions aligned to tile boundaries
- The `TTNNRotaryPositionEmbedding` class in `rope.py` (tt-symbiote): that it pads cos/sin with zeros when `rotary_dim % 32 != 0` before calling `ttnn.experimental.rotary_embedding`, and that `ttnn.pad` is used to perform this padding at runtime
- The signature of `ttnn.experimental.rotary_embedding(input, cos_cache, sin_cache, token_idx, ...)` at a surface level: that it takes a pre-tiled cos/sin cache and applies RoPE to the input tensor

**What they do NOT yet know:**

- The exact shape constraint that `ttnn.experimental.rotary_embedding` enforces on `cos_cache` and `sin_cache` relative to the input tensor: specifically, whether the cos/sin last dimension must equal `rotary_dim` or `head_dim`
- How the kernel's rotate-half pairing is implemented at the tile level: whether it splits on `rotary_dim` or on the full `head_dim`, and what happens when zeros occupy tile slots in the second half of the cos/sin tensor
- Whether zero-padding cos/sin from `rotary_dim=48` to the next tile boundary (64) is sufficient, or whether cos/sin must extend all the way to `head_dim=128`
- Whether there is a trace-safe alternative to `ttnn.pad` (which writes new device buffers and cannot run inside a trace bracket) that also avoids the numerical bug
- Whether `TTNNRotaryPositionEmbedding` should enforce `rotary_dim % 32 == 0` as a hard precondition rather than silently attempting to pad
- Which models in the current tt-symbiote production configuration actually exercise the non-tile-aligned `rotary_dim` path

---

## 2. Chapter List

---

### Chapter 1: Partial RoPE Fundamentals and Tile Alignment Requirements

**Description:** Establishes the mathematical foundation of partial RoPE and the TTNN tile-alignment constraints that determine when special handling of `rotary_dim` is required.

**Directory:** `ch1_rope_fundamentals/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Prerequisite checklist: standard RoPE math, `head_dim` vs. `rotary_dim`, TTNN TILE layout basics
  - Glossary of terms introduced in this chapter: `rotary_dim`, `partial_rotary_factor`, tile-aligned, `nearest_32`, rotate-half pairing
  - "What's next" section listing files in reading order
  - Forward reference: Chapter 2 analyzes how `ttnn.experimental.rotary_embedding` actually enforces these constraints

- `partial_rope_math.md`
  - Standard rotate-half formulation: for a head vector `x` of length `head_dim`, the rotated vector is `(x * cos) + (rotate_half(x) * sin)` where `rotate_half(x) = cat([-x[head_dim/2:], x[:head_dim/2]], dim=-1)`
  - Partial RoPE extension: when `rotary_dim < head_dim`, only the first `rotary_dim` elements of each head are rotated; elements `[rotary_dim:]` pass through unchanged
  - The correct partial RoPE operation requires a cos/sin tensor of shape `[..., rotary_dim]` applied to the first `rotary_dim` elements, with rotate-half splitting that `rotary_dim` slice into two halves `[0, rotary_dim/2)` and `[rotary_dim/2, rotary_dim)`
  - Concrete example with `rotary_dim=48, head_dim=128`: the first 48 elements rotate; the last 80 pass through; rotate-half within the rotated region uses pairs `(x[0], x[24])`, `(x[1], x[25])`, ..., `(x[23], x[47])`
  - Key invariant: the cos/sin tensor must cover exactly `rotary_dim` elements and the rotate-half split must operate within those `rotary_dim` elements, not across the full `head_dim`

- `tile_alignment_in_ttnn.md`
  - TTNN TILE layout requirement: the last two dimensions of any TILE-layout tensor must both be multiples of 32 (`TILE_HEIGHT = TILE_WIDTH = 32`)
  - When `rotary_dim % 32 != 0` (e.g., `rotary_dim=48`), the cos/sin tensor of shape `[1, 1, seq_len, 48]` is not tile-alignable in the last dimension
  - The `nearest_32` utility in tt-symbiote: `nearest_32(n) = ceil(n / 32) * 32`; for `rotary_dim=48`, `nearest_32(48) = 64`
  - `TTNNRotaryPositionEmbedding` padding behavior: when `rotary_dim % 32 != 0`, it calls `ttnn.pad` to extend the cos/sin from shape `[..., rotary_dim]` to `[..., nearest_32(rotary_dim)]`, filling the new positions with zeros
  - The intended semantics: the zeros in positions `[rotary_dim, nearest_32(rotary_dim))` represent "no rotation" for those positions — but this only works if the downstream op reads exactly `rotary_dim` elements, not the full padded width
  - Forward reference: whether `ttnn.experimental.rotary_embedding` actually reads only `rotary_dim` elements or the full padded width is the subject of Chapter 2

---

### Chapter 2: How `ttnn.experimental.rotary_embedding` Processes cos/sin Shapes

**Description:** Traces the C++ implementation of `ttnn.experimental.rotary_embedding` to determine the exact shape contract it enforces on cos/sin and how the rotate-half pairing operates at the tile level.

**Directory:** `ch2_op_shape_contract/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Diagram: data flow from Python call through `RotaryEmbeddingOperation::invoke` to the compute kernel
  - Recap of Chapter 1 prerequisites (partial RoPE math, tile alignment)
  - "What's next" section listing files in reading order

- `shape_validation_in_invoke.md`
  - Walkthrough of `RotaryEmbeddingOperation::invoke` in `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.cpp`:
    - Line: `TT_FATAL(input_tensor.padded_shape()[-1] % (tt::constants::TILE_WIDTH * 2) == 0, ...)` — the input's last dimension must be divisible by 64 (two tiles wide), not 32
    - Line: `uint32_t X = input_tensor.padded_shape()[-1]` — `X` is set to the full `head_dim` (the input's last dimension after any auto-padding)
    - Line: `TT_FATAL(cos_cache.padded_shape()[0] == 1 && cos_cache.padded_shape()[1] == 1 && cos_cache.padded_shape()[-1] == X, ...)` — the cos/sin last dimension must equal `X = head_dim`, not `rotary_dim`
  - Walkthrough of `RotaryEmbedding::validate` in `rotary_embedding_device_operation.cpp`:
    - Same constraint repeated: `cos.padded_shape()[-1] == X` enforced via `TT_FATAL`
  - **Key finding:** `ttnn.experimental.rotary_embedding` requires `cos_cache.shape[-1] == input.shape[-1] == head_dim`. Padding cos/sin from `rotary_dim=48` to `nearest_32(48)=64` is incorrect when `head_dim=128`; the op will either fail (if shapes are checked before autoformat) or proceed with a shape mismatch leading to incorrect computation
  - The autoformat path (`run_with_autoformat`): before invoking the device operation, `AutoFormat::pad_to_tile_shape` is applied to cos/sin, which pads to tile boundaries — but this does not change the `== X` constraint that the device op validates; if the user supplies `cos.shape[-1] = 64` but `X = 128`, the validation `TT_FATAL` fires

- `kernel_rotate_half_pairing.md`
  - Walkthrough of the compute kernel in `device/kernels/compute/rotary_embedding.cpp`:
    - `uint32_t Wt = input.padded_shape()[-1] / TILE_WIDTH` — tile count along the width dimension, computed from `head_dim`
    - `uint32_t half_Wt = Wt / 2` — the rotate-half split point: the first `half_Wt` tiles are "left half" and the second `half_Wt` tiles are "right half"
    - The kernel processes tiles `j = 0` to `j < Wt`; for `j < half_Wt`, it multiplies the "rotated_input" (from the right half of the input) by `-1` then by sin; for `j >= half_Wt`, it multiplies the rotated input (from the left half) by sin directly
  - Walkthrough of the reader kernel in `device/kernels/dataflow/reader_rotary_embedding_interleaved_start_id.cpp`:
    - `uint32_t rotated_input_curr_id = start_id + half_Wt` — the reader fetches the "right half" of the input starting `half_Wt` tiles into the input buffer
    - This confirms that the rotate-half split is computed over `Wt = head_dim / TILE_WIDTH` tiles, not over `rotary_dim / TILE_WIDTH` tiles
  - **Key finding:** the kernel's rotate-half pairing always splits the full `head_dim` into two equal halves of `head_dim / 2` elements each. There is no mechanism to limit rotation to a `rotary_dim` subset within the kernel
  - **Implication for partial RoPE:** `ttnn.experimental.rotary_embedding` does NOT implement partial RoPE natively. To apply RoPE to only the first `rotary_dim` elements of a `head_dim`-wide input, the caller must either: (a) slice the input to `[..., rotary_dim]`, apply the op, then concatenate the passthrough; or (b) set cos/sin to zero for positions `[rotary_dim, head_dim)` with the correct zero pattern that preserves the passthrough elements — but the zero-pattern required is non-trivial

- `what_the_golden_function_reveals.md`
  - Walkthrough of the Python golden function attached to `ttnn.experimental.rotary_embedding` in `ttnn/ttnn/operations/transformer.py`:
    - `rotate_half(x)` uses `x.shape[-1] // 2` as the split point — always the full last dimension, not `rotary_dim`
    - `x_embed = (x * cos) + (rotate_half(x) * sin)` — cos and sin are applied to the full `x` of shape `[..., head_dim]`
  - This confirms that the op is designed for full-head rotation; partial RoPE is not an intended use case of this op without additional structure in the cos/sin tensor
  - The golden function uses `cos_cached[:, :, token_idx : token_idx + 1, ...]` — it slices along the sequence dimension but takes the full last dimension of the cos/sin cache
  - Summary of shape contract: the op requires `cos.shape[-1] == sin.shape[-1] == input.shape[-1] == head_dim`; this is a hard requirement enforced by `TT_FATAL` in C++

---

### Chapter 3: Root Cause Analysis of the PCC ~0.71 Bug

**Description:** Traces the exact sequence of operations in `TTNNRotaryPositionEmbedding` for the `rotary_dim=48, head_dim=128` case, identifies where numerical corruption is introduced, and derives the correct reference output.

**Directory:** `ch3_bug_root_cause/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Answer-first summary: the PCC ~0.71 bug is caused by two compounded errors — incorrect cos/sin padding target (to 64 instead of 128) and incorrect rotate-half pairing (the kernel pairs elements 0–63 with 64–127 in the padded cos/sin, interleaving real rotary values with zeros at the wrong positions relative to the input's actual element layout)
  - Recap of Chapters 1–2 prerequisites
  - "What's next" section listing files in reading order

- `step_by_step_failure_trace.md`
  - Walk through the sequence of operations for `rotary_dim=48, head_dim=128`:
    1. `TTNNRotaryPositionEmbedding` precomputes cos/sin of shape `[1, 1, max_seq_len, 48]` (correct, matching `rotary_dim`)
    2. `ttnn.pad` extends to `[1, 1, max_seq_len, 64]` with zeros in positions 48–63 (incorrect target: should be 128 to satisfy the op's shape constraint)
    3. `ttnn.experimental.rotary_embedding` is called with `input.shape[-1] = 128` — the `TT_FATAL` `cos.shape[-1] == X` check fires unless autoformat pads cos/sin further to `[1, 1, max_seq_len, 128]` (which pad_to_tile_shape would do since 64 is already tile-aligned, so no further padding — the `TT_FATAL` fires at the device op level)
  - Clarify the two possible outcomes depending on whether autoformat intervenes:
    - **Path A (shape mismatch TT_FATAL):** If the padded cos/sin `[..., 64]` reaches the device op with `X=128`, `TT_FATAL` fires with "Cos dims must match input dims" — a crash, not silent corruption
    - **Path B (autoformat pads cos/sin to [..., 128]):** If autoformat extends cos/sin from 64 to 128 by zero-padding, the op receives `cos.shape[-1] = 128 = X` and proceeds; zeros fill positions 64–127 of cos/sin; the kernel then pairs element 0 (real cos value) with element 64 (zero), element 1 with element 65 (zero), ..., element 47 with element 111 (zero) — the rotate-half pairing operates over `head_dim=128`, not `rotary_dim=48`
  - In Path B, the compute is: `output[i] = input[i] * cos[i] + input[i + 64] * (-sin[i])` for `i < 64` (but `cos[48:64]` and `sin[48:64]` are the zeros from the first padding step, not from the autoformat step). The values `input[64:128]` that should pass through unchanged instead get multiplied by the zero sin values and added — giving `output[64:128] = input[64:128] * 0 + ...` which is zero, corrupting the passthrough region
  - Derive a numerical estimate for PCC: if 80 out of 128 elements (positions 48–127) are zeroed or corrupted, the Pearson correlation with the correct output depends on the distribution of values; the observed PCC ~0.71 is consistent with roughly 60% of elements being correct

- `correct_partial_rope_reference.md`
  - Define the correct reference output for `rotary_dim=48, head_dim=128`:
    - `output[0:48] = rotate_half_within_48(input[0:48]) * cos_true[0:48] + ...` where rotate-half uses `input[0:24]` and `input[24:48]`
    - `output[48:128] = input[48:128]` (passthrough, unchanged)
  - Show that PyTorch reference achieves this by calling `torch.cat([rotated_first_half, passthrough], dim=-1)` where `rotated_first_half` is computed using standard rotate-half on the slice `input[..., :rotary_dim]`
  - Explain why this correct behavior cannot be achieved by passing a full-`head_dim` cos/sin tensor with zeros at positions `[rotary_dim, head_dim)`: the kernel's rotate-half split is always at `head_dim/2 = 64`, not at `rotary_dim/2 = 24`; zeros at positions 24–63 of cos/sin would corrupt elements 0–23 and 64–127 differently than the intended passthrough behavior
  - Conclude: there is no cos/sin zero-padding scheme that makes `ttnn.experimental.rotary_embedding` produce correct partial RoPE output when `rotary_dim < head_dim`; the op is mathematically incompatible with partial RoPE without slicing the input

---

### Chapter 4: Correct Implementation Strategies for Non-Tile-Aligned Partial RoPE

**Description:** Presents three concrete implementation strategies for correct partial RoPE in TTNN when `rotary_dim % 32 != 0`, analyzes each for correctness and trace compatibility, and recommends the preferred approach.

**Directory:** `ch4_implementation_strategies/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Decision table: correctness vs. trace compatibility vs. implementation complexity for each strategy
  - Recap of Chapters 1–3 prerequisites (why the current zero-padding approach is wrong)
  - "What's next" section listing files in reading order

- `strategy_a_slice_apply_concat.md`
  - Strategy A: slice the input to `[..., rotary_dim]`, apply `ttnn.experimental.rotary_embedding` with a correctly shaped cos/sin of `[1, 1, seq_len, nearest_32(rotary_dim)]` (padded only to the next tile boundary), then concatenate the passthrough `input[..., rotary_dim:]`
  - Shape analysis: `ttnn.experimental.rotary_embedding` requires `cos.shape[-1] == input.shape[-1]`; after slicing input to `[..., rotary_dim]`, the op requires `cos.shape[-1] == rotary_dim`; since `rotary_dim=48` is not divisible by 64 (`TILE_WIDTH * 2`), the `TT_FATAL` on `input.shape[-1] % (TILE_WIDTH * 2) == 0` will fire for `rotary_dim=48`
  - Conclusion: Strategy A requires `rotary_dim % 64 == 0` (not just `% 32 == 0`) because of the two-tile-wide constraint; for `rotary_dim=48`, even with slicing, the op rejects the input
  - Alternative under Strategy A: pad the input slice from `[..., 48]` to `[..., 64]` before the op call, ensuring the op's tile constraint is met; then the cos/sin must also be `[..., 64]` with zeros at positions 48–63; the op applies rotation uniformly across all 64 positions; elements 48–63 of the padded input receive the zero cos/sin (output = 0 * padded_zeros + ...), and after unpadding the slice back to `[..., 48]`, the result is correct for positions 0–47; the passthrough `input[..., 48:]` is concatenated unchanged
  - Trace compatibility of Strategy A: the slice, pad-to-64, rotary_embedding, unpad-to-48, and concat sequence involves tensor shape operations that may be trace-compatible if shapes are fixed; no host-side allocation (like `ttnn.from_torch`) is needed; the pad operation uses `ttnn.pad` which (like the current implementation) writes new device buffers, making it incompatible with Metal Trace unless the buffer is pre-allocated

- `strategy_b_enforce_tile_alignment.md`
  - Strategy B: enforce `rotary_dim % 32 == 0` as a hard precondition in `TTNNRotaryPositionEmbedding`; raise a `ValueError` at construction time if `rotary_dim % 32 != 0`; document that the only safe configurations are those with tile-aligned `rotary_dim`
  - This does not fix the bug for existing non-tile-aligned configurations; it prevents silent corruption by making the failure explicit
  - Analysis of which model configs are actually affected: enumerate the model configurations in the current tt-symbiote codebase that reach `TTNNRotaryPositionEmbedding` with non-tile-aligned `rotary_dim` (see Chapter 5 for the full audit); if no production-supported model exercises this path, Strategy B (fail-fast enforcement) is the lowest-risk option
  - Recommended constraint to enforce: `rotary_dim % 64 == 0` (not just `% 32 == 0`), because of the two-tile-wide input constraint that `ttnn.experimental.rotary_embedding` imposes; `rotary_dim % 32 == 0` alone is insufficient if `rotary_dim` is an odd multiple of 32 (e.g., `rotary_dim=32` satisfies `% 32 == 0` but not `% 64 == 0`, and the slice-and-rotate approach would fail the two-tile constraint)
  - Implementation: a one-line guard in `TTNNRotaryPositionEmbedding.__init__` with a clear error message citing the tile constraint and listing alternative approaches

- `strategy_c_precomputed_full_head_cos_sin.md`
  - Strategy C: precompute a cos/sin table of shape `[max_seq_len, head_dim]` where positions `[rotary_dim, head_dim/2)` have cos=1/sin=0 (identity rotation) and positions `[head_dim/2 + rotary_dim/2, head_dim)` have cos=1/sin=0 (identity for the rotate-half second half)
  - Mathematical derivation: for `ttnn.experimental.rotary_embedding` to apply partial RoPE correctly, the cos/sin values at positions that should be "passthrough" must encode identity rotation; the rotate-half kernel computes `output[i] = input[i] * cos[i] + input[i + head_dim/2] * (-sin[i])` for `i < head_dim/2`; for element `i < rotary_dim/2`, this should be a real rotation; for `rotary_dim/2 <= i < head_dim/2`, the element should pass through unchanged, requiring `cos[i] = 1, sin[i] = 0`
  - Full cos/sin construction for `rotary_dim=48, head_dim=128`:
    - Positions 0–23 of cos/sin: real cosine/sine values for rotation pairs 0–23
    - Positions 24–63 of cos: set to 1.0; positions 24–63 of sin: set to 0.0 (identity rotation for the "passthrough" pairs in the first half)
    - Positions 64–87 of cos/sin: same real cosine/sine values (the second half of the rotate-half split mirrors the first half)
    - Positions 88–127 of cos: set to 1.0; positions 88–127 of sin: set to 0.0
  - This construction is tile-aligned (`head_dim=128` is divisible by 64) and satisfies the op's shape constraint; the rotate-half pairing in the kernel correctly applies rotation to the first `rotary_dim/2` pairs and identity to the rest
  - Trace compatibility of Strategy C: the cos/sin table is precomputed once during initialization and stored as a fixed device tensor; no runtime padding or allocation is required inside the forward pass; this approach is fully trace-compatible
  - The `ttnn.pad` call is eliminated entirely; the cos/sin table is constructed on CPU using `torch.ones` / `torch.zeros` fills and transferred to device once

- `trace_safe_alternatives_to_ttnn_pad.md`
  - Problem statement: the current `TTNNRotaryPositionEmbedding` calls `ttnn.pad` (or equivalent) at runtime to extend cos/sin; `ttnn.pad` allocates a new device buffer and cannot run inside a Metal Trace bracket
  - Strategy C (precomputed identity-filled cos/sin, see above) eliminates the runtime pad entirely — it is the preferred trace-safe solution for the numerical bug
  - For completeness: alternative trace-safe approaches that do not require precomputation:
    - Pre-allocated zeros buffer: allocate a fixed `[1, 1, max_seq_len, head_dim - rotary_dim]` zeros tensor on device during initialization; at forward time, use `ttnn.concat` along the last dimension to assemble the full cos/sin; `ttnn.concat` is trace-compatible when the output buffer is pre-allocated, but this still requires slice and concat which may have layout restrictions
    - `ttnn.copy` into a pre-allocated full-head buffer: pre-allocate a `[1, 1, max_seq_len, head_dim]` buffer initialized to cos=1/sin=0; at forward time, copy the real cos/sin values into the first `rotary_dim` positions; this requires a strided copy or slice-write which may not be supported in all TTNN versions
  - Recommendation: Strategy C (identity-filled precomputed table) is the cleanest solution — it fixes both the numerical correctness bug and the trace compatibility issue simultaneously, with no runtime overhead

---

### Chapter 5: Model Configurations Using Non-Tile-Aligned rotary_dim in tt-symbiote

**Description:** Audits all models in the current tt-symbiote codebase that pass through `TTNNRotaryPositionEmbedding` to determine which actually exercise the non-tile-aligned `rotary_dim` path and whether the bug affects any production-supported configuration.

**Directory:** `ch5_model_config_audit/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Summary table: model name, `partial_rotary_factor`, `head_dim`, derived `rotary_dim`, tile-aligned status, and whether the bug path is reached
  - "What's next" section listing files in reading order
  - Forward reference: Chapter 6 consolidates findings into a recommended fix and precondition policy

- `which_models_use_ttnn_rope.md`
  - Description of the two RoPE classes in tt-symbiote: `TTNNRotaryPositionEmbedding` (non-distributed, used when `partial_rotary_factor < 1.0`) and `TTNNDistributedRotaryPositionEmbedding` (used when `partial_rotary_factor == 1.0` or when the distributed path is available)
  - Enumerate models that use `TTNNRotaryPositionEmbedding` (non-distributed path) due to `partial_rotary_factor < 1.0`:
    - Qwen3.5-35B-A3B and variants: `partial_rotary_factor=0.5`, `head_dim=128`, derived `rotary_dim = floor(128 * 0.5) = 64`; tile-aligned (`64 % 64 == 0`); bug path NOT reached
    - Qwen3.6-35B-A3B: `partial_rotary_factor=0.5` (text layers), `head_dim=128`, derived `rotary_dim=64`; tile-aligned; bug path NOT reached
    - Gated delta net layers in Qwen3.6 or similar architectures: if `partial_rotary_factor=0.25` with `head_dim=128`, derived `rotary_dim=32`; tile-aligned (`32 % 32 == 0`, satisfies `% 64 == 0` only if `head_dim >= 64`; needs verification for the two-tile constraint)
    - Any hypothetical model with `head_dim=96, partial_rotary_factor=0.5`: `rotary_dim=48`; NOT tile-aligned; bug path reached; this is the `rotary_dim=48` scenario described in the research topic
  - Document the investigation method: grep `rope.py` and model config files for `partial_rotary_factor` values; derive `rotary_dim` and check `rotary_dim % 32` and `rotary_dim % 64`

- `is_this_dead_code.md`
  - Based on the model audit, determine whether the non-tile-aligned `rotary_dim` code path is exercised by any currently supported tt-symbiote model
  - If no production model reaches this path (i.e., all supported models have `rotary_dim % 64 == 0`), the current bug in the zero-padding logic is latent — it would only surface when a new model with non-tile-aligned `rotary_dim` is brought up
  - The `rotary_dim=48` test case described in the research topic was likely a synthetic test or an exploratory configuration, not a current production model
  - Recommendation for latent bugs: even if the path is currently dead code, it should be fixed before it is accidentally exercised by a new model; Strategy B (enforce tile alignment via a precondition) is appropriate if the path is dead; Strategy C (identity-filled precomputed cos/sin) is appropriate if non-tile-aligned configs are anticipated in future models

- `the_rotary_dim_48_test_case.md`
  - Reconstruct the test setup implied by the research topic: `rotary_dim=48, head_dim=128`
  - Trace which ops are called and in which order by `TTNNRotaryPositionEmbedding.forward` for this configuration: (1) cos/sin slice at `cur_pos`, (2) `ttnn.pad` from 48 to 64 with fill_value=0, (3) `ttnn.experimental.rotary_embedding` called with the padded cos/sin
  - Identify the precise point of failure: depending on the TTNN version and whether autoformat is active, either a `TT_FATAL` fires ("Cos dims must match input dims" for shape `[...,64]` vs. required `[...,128]`) or autoformat further pads to `[...,128]` and the kernel runs with incorrect zero-padding producing PCC ~0.71
  - Note that the PCC of ~0.71 in warm-up (no trace) confirms the bug is in the forward pass computation, not in trace capture or replay — the root cause analysis in Chapter 3 explains why

---

### Chapter 6: Recommendations and Implementation Guide

**Description:** Consolidates findings from all prior chapters into a concrete recommendation for fixing `TTNNRotaryPositionEmbedding`, provides an implementation checklist, and specifies the precondition policy for non-tile-aligned `rotary_dim` configurations.

**Directory:** `ch6_recommendations/`

**Files:**

- `index.md`
  - Chapter overview
  - Answer-first summary table: one row per research question from the topic, with a concise answer and a reference to the chapter/file where it is derived
  - Recap of all prior chapter prerequisites
  - "What's next" section pointing to the two implementation files in reading order

- `recommended_fix.md`
  - Primary recommendation: **Strategy C — precomputed identity-filled cos/sin table**
  - Rationale:
    - Fixes the numerical correctness bug for all `rotary_dim < head_dim` configurations regardless of tile alignment
    - Is fully trace-compatible (no runtime pad, no dynamic buffer allocation inside forward)
    - Handles non-tile-aligned `rotary_dim` correctly by extending cos/sin to the full `head_dim` with identity values
    - Adds no runtime overhead over the current (buggy) implementation
  - Step-by-step construction of the identity-filled cos/sin table in `TTNNRotaryPositionEmbedding.__init__`:
    1. Compute the real cos/sin table of shape `[max_seq_len, rotary_dim]` as before
    2. Assert `rotary_dim % 2 == 0` (required for rotate-half pairing)
    3. Assert `head_dim % 2 == 0` and `head_dim >= rotary_dim`
    4. Build the identity extension for the first half: create a `[max_seq_len, head_dim/2 - rotary_dim/2]` cos tensor filled with 1.0 and a corresponding sin tensor filled with 0.0
    5. Concatenate along the last dimension: `cos_full_first_half = cat([cos_real[:, :rotary_dim//2], cos_identity], dim=-1)` (shape `[max_seq_len, head_dim/2]`)
    6. The second half of the rotate-half split mirrors the first half's real values and identity values: `cos_full = cat([cos_full_first_half, cos_full_first_half], dim=-1)` (shape `[max_seq_len, head_dim]`)
    7. Same construction for sin
    8. Transfer to device as a fixed TILE-layout tensor; no further runtime padding needed
  - Remove the `ttnn.pad` call from `TTNNRotaryPositionEmbedding.forward`; the cos/sin tensor is already head_dim-wide
  - Precondition to add: assert `head_dim % 64 == 0` (required by `ttnn.experimental.rotary_embedding`'s two-tile constraint on the input); this is a constraint on `head_dim`, not on `rotary_dim`
  - Optional secondary precondition: warn (but do not error) when `rotary_dim % 32 != 0` to surface unexpected configurations early; do not error because Strategy C handles non-tile-aligned `rotary_dim` correctly

- `precondition_policy.md`
  - Answer to the research question "Should `TTNNRotaryPositionEmbedding` enforce `rotary_dim % 32 == 0`?":
    - **No** — with Strategy C implemented, non-tile-aligned `rotary_dim` is handled correctly; a hard enforcement would unnecessarily restrict valid configurations
    - The constraint that *must* be enforced is `head_dim % 64 == 0` (the two-tile-wide requirement from `ttnn.experimental.rotary_embedding`)
    - `rotary_dim % 2 == 0` must also be enforced (rotate-half requires an even number of elements to pair)
    - `rotary_dim <= head_dim` must be enforced (partial rotary cannot exceed the full head dimension)
  - Answer to the research question "Is Strategy B (enforcing `rotary_dim % 32 == 0`) appropriate?":
    - Strategy B is appropriate as a short-term mitigation if Strategy C is not yet implemented; it converts silent numerical corruption into an explicit error, preventing incorrect results from shipping
    - In the long term, Strategy C supersedes Strategy B entirely
  - Migration path: if Strategy B is deployed first, any new model that would have exercised the non-tile-aligned path will receive a clear error message directing the engineer to round `rotary_dim` up to the next multiple of 32 or implement Strategy C

- `verification_checklist.md`
  - Test case 1 — tile-aligned partial RoPE baseline: `rotary_dim=64, head_dim=128`; verify PCC > 0.9999 against PyTorch reference; this should already pass with the current (buggy) implementation since `rotary_dim` is tile-aligned
  - Test case 2 — non-tile-aligned partial RoPE: `rotary_dim=48, head_dim=128`; verify PCC > 0.9999 against PyTorch reference after applying Strategy C; before the fix, PCC should be ~0.71
  - Test case 3 — full-head RoPE (no partial): `rotary_dim=128, head_dim=128`; verify PCC > 0.9999; this path does not use `TTNNRotaryPositionEmbedding` (it uses the distributed class) but serves as a sanity check
  - Test case 4 — trace compatibility: with Strategy C in place, run a decode step inside `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` and verify that no `TT_FATAL` fires during replay; compare numerical output between traced and non-traced runs; PCC should be > 0.9999
  - Test case 5 — edge case: `rotary_dim=32, head_dim=128`; verify correct rotation of only the first 32 elements; `rotary_dim=32` satisfies `% 32 == 0` and `% 64 != 0`; with Strategy C, the identity-fill extends correctly
  - How to compute the PyTorch reference: use `torch.cat([apply_rotary(x[..., :rotary_dim], cos_real, sin_real), x[..., rotary_dim:]], dim=-1)` where `apply_rotary` applies the standard rotate-half on the `rotary_dim`-wide slice

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| `rotary_dim` | The number of head vector dimensions that receive rotary encoding: `rotary_dim = floor(head_dim * partial_rotary_factor)`; must be even |
| `head_dim` | The full head vector dimension; `head_dim = hidden_size / num_heads`; must satisfy `head_dim % 64 == 0` for `ttnn.experimental.rotary_embedding` |
| `partial_rotary_factor` | The fraction of `head_dim` covered by rotation; `partial_rotary_factor = 1.0` means full-head rotation, `partial_rotary_factor < 1.0` means partial rotation |
| tile-aligned | A dimension value that is a multiple of 32 (`TILE_WIDTH`); when a dimension is not tile-aligned, TTNN TILE layout cannot represent it without padding |
| `nearest_32(n)` | `ceil(n / 32) * 32`; the smallest multiple of 32 that is `>= n` |
| two-tile constraint | The requirement in `ttnn.experimental.rotary_embedding` that the input's last dimension be divisible by `TILE_WIDTH * 2 = 64`; applies to `head_dim`, not to `rotary_dim` |
| rotate-half pairing | The operation that pairs element `i` with element `i + head_dim/2` (or `i + rotary_dim/2` for partial RoPE); the kernel always uses `head_dim/2` as the pairing offset |
| identity rotation | A cos/sin pair where `cos = 1.0` and `sin = 0.0`; applying this to element `i` produces `output[i] = input[i] * 1 + input[i + head_dim/2] * 0 = input[i]` — a passthrough |
| `TTNNRotaryPositionEmbedding` | The non-distributed RoPE class in tt-symbiote's `rope.py`; used when `partial_rotary_factor < 1.0`; takes a precomputed cos/sin table and applies `ttnn.experimental.rotary_embedding` at forward time |
| `TTNNDistributedRotaryPositionEmbedding` | The distributed (tensor-parallel) RoPE class; used when `partial_rotary_factor == 1.0` or when the distributed path is available |
| Strategy A | Fix via slice-apply-concat: slice input to `[..., rotary_dim]`, apply rotary embedding, concat passthrough; requires additional padding to meet the two-tile constraint |
| Strategy B | Fix via enforced precondition: raise `ValueError` at construction time if `rotary_dim % 32 != 0`; converts silent corruption to explicit error |
| Strategy C | Fix via identity-filled precomputed cos/sin table of shape `[max_seq_len, head_dim]`; eliminates runtime padding; correct for any `rotary_dim <= head_dim` with `rotary_dim % 2 == 0` |
| PCC | Pearson Correlation Coefficient; used to measure numerical closeness to a PyTorch reference; target is > 0.9999 for BF16 vs. float32 |
| BF16 | BFloat16; the standard compute dtype for activations in TTNN |
| DRAM | Device DRAM on each Tenstorrent chip; where precomputed cos/sin tables are stored |
| L1 | On-chip SRAM on each Tenstorrent core; where activation tiles are staged for the compute kernel |
| `TT_FATAL` | A runtime assertion in tt-metal C++ that halts execution with an error message when a condition is not met; used to enforce shape contracts in `ttnn.experimental.rotary_embedding` |

### Notation

- Tensor shapes use square brackets with comma separation: `[1, 1, seq_len, 48]`.
- Python code is formatted in fenced code blocks with the `python` language tag.
- C++ source references include the file path relative to the tt-metal repository root and the function or line containing the cited constraint: e.g., `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.cpp`, `RotaryEmbeddingOperation::invoke`, line validating `cos_cache.padded_shape()[-1] == X`.
- Half-open index ranges use `[a, b)` notation: positions 0–23 are written as `[0, 24)`.
- The rotate-half split point is always written explicitly as `head_dim / 2` or `rotary_dim / 2` — never abbreviated — to avoid ambiguity in the context of partial RoPE.
- Identity rotation values are written as `cos=1.0, sin=0.0` (not `cos=1, sin=0`) to make the floating-point nature explicit.
- Key findings that directly answer the original research questions are formatted as `> **Key Finding:** ...` blockquotes.
- Warnings about silent failure modes (no crash, wrong numerical output) are formatted as `> **[SILENT FAILURE]** ...` blockquotes.
- File paths to source code are always given as `inline code` with the repository root prefix, e.g., `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.cpp`.

### Formatting Rules

- Every chapter directory contains an `index.md` with a chapter overview, learning objectives, prerequisite recap, and a "What's next" navigation list.
- Each `.md` file begins with an H1 title and a one-paragraph orientation stating what the reader will learn from this file.
- Section headings use `##` (H2) and `###` (H3) within files; no deeper nesting.
- Numbered lists are used for sequential steps (e.g., the Strategy C construction procedure); unordered bullet lists are used for non-sequential enumerations.
- Comparison tables use GitHub-flavored pipe Markdown; rows represent configurations (e.g., `rotary_dim=48, head_dim=128`), columns represent properties (correctness, trace-safe, implementation complexity).
- No emoji in any file.
- Code examples include inline `# comments` explaining non-obvious steps.

---

## 4. Cross-Chapter Dependencies

```
Chapter 1 (Partial RoPE Fundamentals and Tile Alignment Requirements)
  - Introduces: rotary_dim, head_dim, partial_rotary_factor, nearest_32, rotate-half pairing
                concept, tile-alignment requirement, TTNNRotaryPositionEmbedding padding behavior
  - Required by: all subsequent chapters

Chapter 2 (How ttnn.experimental.rotary_embedding Processes cos/sin Shapes)
  - Depends on: Chapter 1 (tile alignment, rotate-half pairing, TTNNRotaryPositionEmbedding)
  - Introduces: the two-tile constraint (head_dim % 64 == 0), the TT_FATAL cos.shape[-1] == X
                requirement, the kernel's rotate-half pairing over head_dim (not rotary_dim),
                the golden function confirming full-head rotation semantics,
                the mismatch between the op's design intent and partial RoPE usage
  - Required by: Chapter 3 (root cause is derived from the shape contract),
                 Chapter 4 (Strategy A and Strategy C correctness arguments reference the
                 kernel behavior), Chapter 6 (precondition policy derived from the op constraints)

Chapter 3 (Root Cause Analysis of the PCC ~0.71 Bug)
  - Depends on: Chapter 1 (partial RoPE math, TTNNRotaryPositionEmbedding padding behavior),
                Chapter 2 (shape contract, rotate-half pairing over head_dim)
  - Introduces: the two failure paths (TT_FATAL vs. autoformat further padding),
                the numerical derivation of PCC ~0.71, the correct reference output for
                rotary_dim=48/head_dim=128, the conclusion that no zero-padding scheme produces
                correct partial RoPE from ttnn.experimental.rotary_embedding
  - Required by: Chapter 4 (strategies must address the root cause identified here),
                 Chapter 6 (recommended fix is derived from this analysis)

Chapter 4 (Correct Implementation Strategies for Non-Tile-Aligned Partial RoPE)
  - Depends on: Chapter 1 (tile alignment, ttnn.pad incompatibility with trace),
                Chapter 2 (op shape contract, two-tile constraint),
                Chapter 3 (no zero-padding scheme works; identity-filled table is the correct
                alternative)
  - Introduces: Strategy A (slice-apply-concat), Strategy B (enforced precondition),
                Strategy C (identity-filled precomputed cos/sin table), trace-safe alternatives
                to ttnn.pad (pre-allocated zeros buffer, ttnn.copy into pre-allocated buffer)
  - Required by: Chapter 6 (recommended fix is Strategy C from this chapter; precondition
                 policy is Strategy B from this chapter)

Chapter 5 (Model Configurations Using Non-Tile-Aligned rotary_dim in tt-symbiote)
  - Depends on: Chapter 1 (which models use TTNNRotaryPositionEmbedding vs. the distributed class,
                and how partial_rotary_factor maps to rotary_dim)
  - Introduces: the model-by-model audit of rotary_dim tile alignment, the determination of
                whether the bug path is reached in any production-supported model,
                the distinction between dead code (latent bug) and live code (active bug)
  - Required by: Chapter 6 (precondition policy decision depends on whether any current model
                 is affected; the urgency of the fix depends on whether this is dead or live code)

Chapter 6 (Recommendations and Implementation Guide)
  - Depends on: all prior chapters
  - Synthesizes: correct partial RoPE math (Ch1), op shape contract and kernel behavior (Ch2),
                 root cause derivation (Ch3), Strategy C construction steps (Ch4),
                 production impact assessment (Ch5)
  - Introduces no new technical concepts; provides the consolidated recommendation, implementation
    checklist, precondition policy, and verification test suite
```

**Specific forward references to flag:**

- **Ch1 → Ch2:** `tile_alignment_in_ttnn.md` describes the padding behavior of `TTNNRotaryPositionEmbedding` and notes that whether zeros in the padded positions cause corruption depends on how the op processes them — flag readers to see Ch2 for the answer.
- **Ch2 → Ch3:** `kernel_rotate_half_pairing.md` establishes that the kernel splits over `head_dim`, not `rotary_dim` — flag readers to see Ch3 for the numerical impact of this on the `rotary_dim=48, head_dim=128` test case.
- **Ch3 → Ch4:** `correct_partial_rope_reference.md` concludes that no zero-padding scheme can produce correct partial RoPE output — flag readers to see Ch4 for the strategies that do work.
- **Ch4 → Ch6:** `strategy_c_precomputed_full_head_cos_sin.md` describes the identity-filled table approach — flag readers to see Ch6 for the step-by-step construction procedure and the precondition policy.
- **Ch5 → Ch6:** `is_this_dead_code.md` determines whether the bug is currently active in production — flag readers to see Ch6 for how this finding influences the urgency and type of recommended fix.
