# Plan: M-RoPE (Multimodal RoPE) Implementation on TTNN

---

## 1. Audience

**Primary audience:** ML engineers working on multimodal model bring-up in the tt-symbiote / TTNN stack, specifically engineers preparing Qwen3.6-35B-A3B or similar Qwen-VL models for vision/video inference beyond the current text-only path.

**What they already know:**

- Standard 1D Rotary Position Embedding (RoPE): the sinusoidal frequency table, how cos/sin are applied to query/key vectors, how `rope_theta` and `rotary_dim` interact
- Partial RoPE (`partial_rotary_factor < 1.0`): only the first `rotary_dim` dimensions of each head are rotated; the remainder pass through unchanged
- The existing `TTNNRotaryPositionEmbedding` (non-distributed) and `TTNNDistributedRotaryPositionEmbedding` classes in tt-symbiote: their call signatures, how they precompute cos/sin tables, and why non-distributed is forced when `partial_rotary_factor < 1.0`
- TTNN tensor operations at a working level: `ttnn.linear`, `ttnn.matmul`, memory configs, shard configs, `ttnn.to_device`
- The Qwen3.6-35B-A3B model architecture at a high level: it uses the same `Qwen3_5MoeForConditionalGeneration` class as Qwen3.5-35B-A3B, with partial RoPE (`rotary_dim=64`) and GQA

**What they do NOT need to know in advance:**

- How M-RoPE extends standard RoPE to assign independent position coordinates per modality (text, image, video)
- What the `mrope_section` dimensions `[11, 11, 10]` mean and how they partition `rotary_dim=64` into temporal/height/width sub-groups
- Whether M-RoPE collapses to standard RoPE for text-only batches or whether the section structure is always active
- Whether `TTNNRotaryPositionEmbedding` can be extended for M-RoPE or whether a new implementation is required
- What the performance cost of per-section position indexing is on TTNN versus a single unified cos/sin table

This guide answers all four of those gaps, starting from the mathematical foundations of M-RoPE and ending with a concrete TTNN implementation strategy and performance cost analysis.

---

## 2. Chapter List

---

### Chapter 1: Standard RoPE and M-RoPE — Conceptual Foundations

**Description:** Establishes the mathematical progression from standard 1D RoPE through partial RoPE to M-RoPE, making the interleaved section design fully legible before any TTNN-specific material.

**Directory:** `ch1_rope_foundations/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Prerequisite checklist: standard RoPE, partial RoPE, head_dim vs. rotary_dim
  - Forward references: Ch2 for the Qwen3.6 M-RoPE configuration, Ch3 for text-only reduction behavior

- `standard_rope_recap.md`
  - Frequency table construction: for dimension index `i` in `[0, rotary_dim/2)`, frequency `θ_i = 1 / (rope_theta^(2i / rotary_dim))`
  - How cos and sin are precomputed for every sequence position `t`: `cos(t * θ_i)`, `sin(t * θ_i)`
  - Application to a query or key vector: the "rotate-half" operation — each pair of adjacent dimensions `(x_{2i}, x_{2i+1})` is rotated by angle `t * θ_i`
  - Partial RoPE: if `partial_rotary_factor < 1.0`, only the first `rotary_dim = floor(head_dim * partial_rotary_factor)` dimensions are rotated; dimensions `[rotary_dim:]` are concatenated unchanged
  - Concrete example with Qwen3.6 numbers: `head_dim=128`, `partial_rotary_factor=0.5` (text layers), `rotary_dim=64` — the first 64 dimensions rotate and the last 64 pass through

- `mrope_motivation_and_design.md`
  - The core problem standard RoPE cannot solve for multimodal sequences: a flat 1D position index cannot distinguish whether two tokens are at the same temporal frame, the same spatial row, or the same spatial column
  - M-RoPE's solution: assign each token a *triplet* of position coordinates `(t, h, w)` — temporal position, spatial height, spatial width — and encode each coordinate in a dedicated sub-group of rotary dimensions
  - The section partition: `mrope_section = [s_t, s_h, s_w]` where `s_t + s_h + s_w = rotary_dim / 2` (each section covers half-dimension pairs before the rotate-half operation); for Qwen3.6, `[11, 11, 10]` sums to 32, consistent with `rotary_dim=64` (32 pairs)
  - How position IDs are structured: instead of a 1D `[batch, seq_len]` tensor, M-RoPE uses a 3D `[3, batch, seq_len]` tensor where axis 0 indexes the coordinate (t, h, w)
  - For text tokens: all three coordinates are set to the same value (the sequential text position); this is what makes M-RoPE degenerate to standard RoPE for text — covered fully in Ch3

- `section_dimension_assignment.md`
  - Deriving which frequency pairs belong to which section from `mrope_section = [s_t, s_h, s_w]`:
    - Temporal section: dimension pairs `[0, s_t)` of the cos/sin table → uses coordinate `t` from position ID axis 0
    - Height section: dimension pairs `[s_t, s_t + s_h)` → uses coordinate `h` from position ID axis 1
    - Width section: dimension pairs `[s_t + s_h, rotary_dim/2)` → uses coordinate `w` from position ID axis 2
  - Concrete example for Qwen3.6: temporal covers pairs 0–10 (dimensions 0–21 in the full vector), height covers pairs 11–21 (dimensions 22–43), width covers pairs 22–31 (dimensions 44–63)
  - Why the sections are interleaved in the cos/sin lookup but not interleaved in the weight tensor: each section indexes a contiguous slice of the frequency table using a different position coordinate
  - The shape of the effective cos/sin tensor for one batch, one head: `[seq_len, rotary_dim]` where the first `2*s_t` entries use temporal cos/sin, the next `2*s_h` entries use height cos/sin, and the last `2*s_w` entries use width cos/sin

---

### Chapter 2: M-RoPE in Qwen3.6-35B-A3B — Configuration and Reference Implementation

**Description:** Translates the abstract M-RoPE design into the exact configuration used by Qwen3.6-35B-A3B and traces the HuggingFace reference implementation step by step.

**Directory:** `ch2_qwen36_mrope_config/`

**Files:**

- `index.md`
  - Chapter overview
  - Quick-reference table: all M-RoPE-relevant config fields from `config.json`

- `qwen36_rope_config.md`
  - The complete set of relevant fields from the Qwen3.6-35B-A3B config:
    - `rope_theta: 1000000.0` — base theta used for all three sections (temporal, height, width)
    - `partial_rotary_factor: 0.25` — forces `rotary_dim = floor(128 * 0.25) = 32` (note: the model card also quotes 64; document both and resolve the discrepancy by tracing HF code)
    - `mrope_section: [11, 11, 10]` — section split; the three numbers must sum to `rotary_dim / 2`
    - `rope_scaling.type: "mrope"` and `rope_scaling.mrope_section: [11, 11, 10]` — the canonical location of the section config in the `rope_scaling` nested dict
  - How HuggingFace resolves `rotary_dim` when both `partial_rotary_factor` and `rope_scaling.mrope_section` are present: the sum of `mrope_section` values determines the number of rotation pairs; `partial_rotary_factor` is a secondary hint that must be consistent
  - Relationship between `rotary_dim` in the TTNN sense and the M-RoPE section sum: `rotary_dim == 2 * sum(mrope_section)` must hold; document what happens in HF code if this is violated

- `hf_reference_implementation.md`
  - Walkthrough of `Qwen2_5_VLRotaryEmbedding` (or the equivalent class in the Qwen3.6 HF implementation): how it initializes the frequency table using `rope_theta` and `rotary_dim`
  - Walkthrough of `apply_multimodal_rotary_pos_emb()`: how it takes `(q, k, cos, sin, mrope_section, unsqueeze_dim)` and applies independent rotations per section using tensor slicing on the cos/sin dimension
  - The exact tensor operation: for each section `s` with dimensions `[d_start, d_end)`, the code slices `cos[:, :, d_start:d_end]` using the corresponding row of the position ID tensor (axis 0 for temporal, axis 1 for height, axis 2 for width)
  - How position IDs are constructed for each modality:
    - **Text tokens**: all three rows of the position ID tensor are set to `range(seq_len)` — identical across axes
    - **Image tokens**: temporal row is a constant (the frame index), height and width rows are a 2D grid rasterized to 1D
    - **Video tokens**: temporal row encodes the frame index for each spatial token; height and width rows encode the 2D spatial grid per frame
  - The shape of the 3D position ID tensor: `[3, batch_size, seq_len]`, dtype `int32` or `int64`

- `position_id_construction.md`
  - Step-by-step derivation of position IDs for a multimodal sequence containing text + image tokens:
    - Assign sequential text positions to pre-image text tokens
    - For each image patch at grid coordinates `(i_h, i_w)` in a grid of shape `(num_patches_h, num_patches_w)`: temporal = image index (constant across patches), height = `i_h + text_offset`, width = `i_w + text_offset` (or offset strategies used by Qwen VL)
    - Continue sequential positions for post-image text tokens, starting from `max(h, w, t) + 1`
  - The text-only degenerate case: when the input contains no vision tokens, all three rows of the position ID tensor are identical — this is the gateway to Ch3's analysis of text-only reduction

---

### Chapter 3: Text-Only Behavior — Does M-RoPE Reduce to Standard RoPE?

**Description:** Answers the first research question directly: whether M-RoPE with equal position IDs across all three axes produces numerically identical output to standard 1D RoPE with the same theta and rotary_dim.

**Directory:** `ch3_text_only_reduction/`

**Files:**

- `index.md`
  - Chapter overview
  - Answer-first summary: M-RoPE *does* reduce to standard RoPE for text-only inputs, but the mechanism requires careful verification — this chapter provides the proof and the caveats

- `mathematical_equivalence_proof.md`
  - Formal statement: if `position_ids[0, :, :] == position_ids[1, :, :] == position_ids[2, :, :]` (all three coordinate axes use the same integer sequence), then for every token position `t` and every section `s`, the applied cos/sin values are `cos(t * θ_i)` and `sin(t * θ_i)` — identical to standard RoPE with the same frequency table
  - Why this holds: each section slice `cos[:, :, d_start:d_end]` selects the same rows of the frequency table regardless of which position axis is used, because all three axes have the same value
  - The one caveat: the frequency table is constructed over `rotary_dim` total dimensions; in M-RoPE, all `rotary_dim` dimensions are covered by the union of sections, so there is no gap or overlap — the coverage is identical to standard partial RoPE
  - Second caveat: the section partition affects which frequency entries are indexed by which coordinate; if a non-text model accidentally passes mismatched position IDs (e.g., height row = 0 for all text tokens while temporal = sequential), the result diverges from standard RoPE silently

- `practical_implications_for_text_inference.md`
  - For Qwen3.6-35B-A3B used in text-only mode: the HF `generate()` loop constructs 3D position IDs with all three rows identical (standard sequential positions), so M-RoPE is active in name but produces standard RoPE output
  - The existing `TTNNRotaryPositionEmbedding` in tt-symbiote already precomputes a 1D cos/sin table and applies partial RoPE correctly for text-only inputs — this means the current text-only path does NOT need M-RoPE support
  - Key finding: no changes are needed to the current TTNN text inference path for Qwen3.6-35B-A3B; M-RoPE support is only required when vision/video tokens are present
  - Risk of over-engineering: a premature M-RoPE implementation that always constructs 3D position IDs even for text-only batches would add overhead with no correctness benefit; the implementation should gate on whether vision tokens are present

- `mrope_section_always_active.md`
  - Clarifying a potential misconception: the `mrope_section` config field is always active in the model's RoPE module in HuggingFace (the `Qwen2_5_VLRotaryEmbedding` class always splits by section), but its *effect* on output values depends entirely on whether the three position ID rows are identical or divergent
  - The section structure does NOT add overhead per se if the three rows of position IDs are the same tensor — the cos/sin lookup produces the same values whether or not you conceptually split into sections
  - Implication for TTNN: if the three position ID rows are always identical for text-only batches, a standard 1D cos/sin table lookup is sufficient and the section split can be skipped at the TTNN level

---

### Chapter 4: M-RoPE TTNN Implementation Strategy

**Description:** Analyzes whether `TTNNRotaryPositionEmbedding` can be extended for M-RoPE and provides a concrete multi-option implementation plan covering both the extension approach and a new-class approach.

**Directory:** `ch4_ttnn_implementation/`

**Files:**

- `index.md`
  - Chapter overview
  - Decision framework: when to extend the existing class vs. when to implement a new class

- `existing_ttnn_rope_gap_analysis.md`
  - Current capabilities of `TTNNRotaryPositionEmbedding`:
    - Precomputes a single cos/sin table `[max_seq_len, rotary_dim]` using a scalar `rope_theta` and scalar `rotary_dim`
    - At forward time, slices the table at the current position and applies standard rotate-half
    - Non-distributed: all devices hold the full table; used when `partial_rotary_factor < 1.0` (which is the case for Qwen3.6)
  - Current capabilities of `TTNNDistributedRotaryPositionEmbedding`:
    - Shards the cos/sin table across devices along the head dimension; each device computes rotation for its shard
    - Forced off when `partial_rotary_factor < 1.0` because the partial cos/sin table shape does not align with head sharding
  - Gap 1: both classes accept a 1D position index (a scalar or 1D tensor) — M-RoPE requires a 3D position ID tensor `[3, batch, seq_len]`
  - Gap 2: neither class partitions the cos/sin table into sections; the full table is always indexed by a single position coordinate
  - Gap 3: neither class maintains a separate cos/sin table per section (temporal, height, width) or a lookup mechanism that uses different coordinates for different dimension ranges
  - What would NOT need to change: the rotate-half operation itself, the underlying TTNN elementwise multiply with cos/sin, the DRAM placement of the table — these are modality-agnostic

- `extension_approach.md`
  - Option A: extend `TTNNRotaryPositionEmbedding` with M-RoPE support via a `use_mrope` flag
  - Constructor changes: accept `mrope_section: list[int]` parameter; precompute a cos/sin table of shape `[max_seq_len, rotary_dim]` with **the same frequencies** as standard partial RoPE (no change to the table values — the sections only govern how it is *indexed*)
  - Forward signature change: accept `position_ids: ttnn.Tensor` of shape `[3, batch, seq_len]` in addition to the current scalar/1D position argument
  - Forward logic change for M-RoPE:
    1. Slice position IDs into temporal `position_ids[0]`, height `position_ids[1]`, width `position_ids[2]`
    2. For the temporal section `[0, s_t)`: index the cos/sin table using `position_ids[0]` → gather `cos/sin[:, 0 : 2*s_t]`
    3. For the height section `[s_t, s_t+s_h)`: index using `position_ids[1]` → gather `cos/sin[:, 2*s_t : 2*(s_t+s_h)]`
    4. For the width section `[s_t+s_h, rotary_dim/2)`: index using `position_ids[2]` → gather `cos/sin[:, 2*(s_t+s_h) : rotary_dim]`
    5. Concatenate the three slices along the dimension axis to form the full `[batch, seq_len, rotary_dim]` cos/sin tensor
    6. Apply the rotate-half operation as usual
  - Backward compatibility: when `use_mrope=False` (the current text-only path), the class behaves exactly as before with no overhead
  - Key insight: the frequency table is *unchanged* — M-RoPE reuses the same frequencies but indexes different portions of the table with different coordinates

- `new_class_approach.md`
  - Option B: implement a new `TTNNMRoPERotaryPositionEmbedding` class separate from the existing class
  - Rationale for a new class: keeps the existing text-only code path clean, avoids conditional branches in hot decode path, makes the M-RoPE class independently testable
  - Interface: `forward(q, k, position_ids_3d)` where `position_ids_3d` is `[3, batch, seq_len]`
  - Implementation mirrors the extension approach but without the backward-compatibility branching
  - Trade-off: code duplication of the rotate-half logic and cos/sin precomputation; can be mitigated by extracting shared utilities into a base class or helper functions
  - Recommendation: for initial bring-up, Option A (extension) is lower risk because it does not require changing module registration or symbiote module replacement logic; Option B is preferable for a production implementation where the M-RoPE path is always exercised

- `pre_computed_cos_sin_strategy.md`
  - Whether M-RoPE requires per-modality cos/sin tables (one table per section) or a single shared table indexed differently
  - Answer: a single cos/sin table of shape `[max_seq_len, rotary_dim]` is sufficient — M-RoPE does not require separate tables per section because all sections use the same frequency definitions; only the *position coordinate* used to index each section differs
  - Memory implication: no additional cos/sin table storage required versus standard partial RoPE — the M-RoPE overhead is entirely in the indexing step (gathering from three different rows of position_ids), not in table storage
  - Exception for video: if video inputs can have temporal positions significantly larger than `max_seq_len` (e.g., temporal index = frame number for a long video), the table may need to be extended beyond the standard text context length; document the maximum expected temporal, height, and width position values for Qwen3.6 VL inputs

- `gather_operation_on_ttnn.md`
  - The key new TTNN operation required: indexed lookup of the cos/sin table using a 2D position ID tensor `[batch, seq_len]` where each entry is an arbitrary integer (not necessarily monotonic, not necessarily starting from 0)
  - How this differs from the current implementation: current code slices a contiguous range from the table (e.g., `table[cur_pos : cur_pos+seq_len]`); M-RoPE requires random-access gathering with per-token position IDs
  - TTNN op options:
    1. `ttnn.embedding`: treats the position ID tensor as indices into the table; shape `[batch*seq_len]` indices → `[batch*seq_len, rotary_dim]` outputs; then reshape to `[batch, seq_len, rotary_dim]`
    2. `ttnn.gather`: if available, provides direct N-dimensional indexing
    3. Host-side gather: compute the cos/sin for each token on CPU and transfer to device; acceptable for prefill but not for decode where latency matters
  - Analysis of `ttnn.embedding` for this purpose: embedding lookup is designed exactly for integer-index gathering; the cos/sin table is the embedding weight; the position ID is the index tensor; this is the recommended approach
  - Decode vs. prefill considerations: at decode time, `seq_len=1` per token; the embedding lookup degenerates to a single row lookup — negligible overhead; at prefill time, the lookup covers the full sequence with arbitrary position IDs

---

### Chapter 5: Performance Cost Analysis — M-RoPE vs. Standard RoPE on TTNN

**Description:** Quantifies the additional overhead introduced by M-RoPE's per-section indexing compared to standard partial RoPE, broken down by operation type and decode vs. prefill phase.

**Directory:** `ch5_performance_analysis/`

**Files:**

- `index.md`
  - Chapter overview
  - Answer-first summary: the dominant M-RoPE overhead is the three separate gather operations (one per coordinate axis) versus a single contiguous slice; this is bandwidth-bound and expected to be small relative to attention matmul cost

- `operation_cost_breakdown.md`
  - Standard partial RoPE on TTNN at decode time (seq_len=1):
    - Single slice of cos/sin table at current position: `O(rotary_dim)` data movement, negligible compute
    - Elementwise multiply of Q and K with cos/sin: `O(batch * num_heads * rotary_dim)` operations
  - M-RoPE at decode time (seq_len=1):
    - Three `ttnn.embedding` lookups: each lookup retrieves a `[1, section_dim]` row from the table; total data read is `O(rotary_dim)` same as standard RoPE but split across three separate kernel launches
    - Three concatenations to assemble the full cos/sin tensor: `O(rotary_dim)` write
    - Elementwise multiply: unchanged — `O(batch * num_heads * rotary_dim)`
    - Additional overhead: 3 kernel launches instead of 1 for the table lookup; concatenation is an extra kernel; estimate 2–4 additional TTNN op dispatches per decode step
  - M-RoPE at prefill time (seq_len=S):
    - Three `ttnn.embedding` lookups over arbitrary position IDs: each lookup is `O(S * section_dim)` and uses random-access DRAM reads; bandwidth usage is `3x` compared to a contiguous slice
    - For contiguous text-only inputs (position IDs are sequential), the random-access pattern degenerates to sequential reads — hardware prefetcher effectiveness applies
    - For image/video inputs with non-sequential (grid-based) position IDs, DRAM access pattern is non-contiguous; expect reduced effective bandwidth versus sequential reads
  - Summary table: standard RoPE vs. M-RoPE operation counts and kernel launches for decode and prefill

- `memory_access_analysis.md`
  - Contiguous slice (standard RoPE decode): `table[pos]` reads `rotary_dim` consecutive bytes — maximally cache-friendly
  - Gather-based lookup (M-RoPE decode with seq_len=1): reads three small slices from different rows of the table; each read is within the same DRAM bank; overhead negligible for `seq_len=1`
  - Gather-based lookup (M-RoPE prefill with image tokens): `S` arbitrary row reads from the table; worst case is a stride-`max_seq_len` access pattern across the temporal dimension for image patches; analysis of DRAM page hit rate
  - The cos/sin table size for Qwen3.6 partial RoPE: `max_seq_len * rotary_dim * 2 (cos + sin) * 2 bytes (BF16)` = e.g., at `max_seq_len=32768` and `rotary_dim=64`: `32768 * 64 * 4 = 8 MB` — fits comfortably in DRAM, does not require L1 staging for the decode case
  - Whether additional cos/sin tables per section would change the memory footprint: as established in Ch4, no additional tables are needed; the same table is used for all sections

- `kernel_launch_overhead.md`
  - Baseline: standard partial RoPE requires 2 kernel dispatches per decode step (1 for Q rotation, 1 for K rotation, or fused into 1)
  - M-RoPE: adds 3 embedding lookups + 2 concatenations = 5 additional dispatches; total becomes ~7 dispatches vs. ~2
  - On TTNN at decode time, kernel dispatch overhead per op is on the order of microseconds; 5 additional ops adds ~10–50 µs depending on host-dispatch latency
  - Comparison to total decode latency for Qwen3.6 on T3K: the dominant costs are MoE expert matmuls and CCL; RoPE is a small fraction of total latency; the M-RoPE overhead is expected to be < 1% of decode latency
  - Opportunity to reduce dispatch count: if section boundaries align with tile size (32), the three gather + concatenation operations can potentially be fused into a single custom TTNN op; document this as a future optimization

- `prefill_vs_decode_comparison.md`
  - Decode (seq_len=1 per step): M-RoPE overhead is dominated by kernel dispatch count, not compute or bandwidth; negligible relative to attention and MoE latency
  - Prefill (full sequence, including vision tokens): the random-access embedding lookups for image/video position IDs can degrade bandwidth efficiency; quantify the expected slowdown relative to contiguous RoPE for a 1024-token image sequence
  - Text-only prefill with M-RoPE: if position IDs are sequential (which they always are for text), the gather degenerates to sequential reads; overhead is the same 5 additional dispatches, potentially fused if the compiler detects the sequential pattern
  - Recommendation: for the initial bring-up, implement the naive gather-based approach; profile with Tracy to measure actual overhead before investing in a fused M-RoPE kernel

---

### Chapter 6: Integration Plan and Testing Strategy

**Description:** Provides a concrete, step-by-step plan for integrating M-RoPE into the existing tt-symbiote Qwen3.6 implementation and validating correctness against the HuggingFace reference.

**Directory:** `ch6_integration_and_testing/`

**Files:**

- `index.md`
  - Chapter overview
  - Prerequisite: text-only Qwen3.6-35B-A3B inference is already working on TTNN (M-RoPE for text-only does not require changes, as established in Ch3)
  - Scope: this chapter covers the additional work needed to support mixed text+image batches

- `integration_steps.md`
  - Step 1: add `mrope_section` extraction to the Qwen3.6 config loader in tt-symbiote; read `config.rope_scaling.mrope_section` or fall back to `[rotary_dim//4, rotary_dim//4, rotary_dim//2 - rotary_dim//4]` if absent
  - Step 2: modify the `TTNNRotaryPositionEmbedding` constructor to accept `mrope_section` (Option A from Ch4), or create a new `TTNNMRoPERotaryPositionEmbedding` class (Option B from Ch4)
  - Step 3: modify the attention module's forward to pass a `[3, batch, seq_len]` position ID tensor when vision tokens are present, and a standard 1D position tensor for text-only batches
  - Step 4: implement the position ID construction logic for Qwen VL inputs (text + image tokens interleaved): follow the reference construction described in Ch2 (`position_id_construction.md`)
  - Step 5: validate on a CPU reference before moving to device; use `torch.testing.assert_close` to confirm M-RoPE output matches HuggingFace output for a mixed text+image batch
  - Step 6: move to TTNN device; validate using PCC > 0.999 against the CPU reference

- `correctness_validation.md`
  - Test case 1 — text-only batch: confirm that M-RoPE with identical position IDs across all three axes produces the same output as the existing standard partial RoPE path (numerical identity, not just PCC)
  - Test case 2 — single image token: construct a minimal batch with one text prefix token + one image patch + one text suffix token; verify each token's cos/sin application is correct
  - Test case 3 — full image grid: construct a batch with a 16x16 image grid (256 image patches) embedded in a text sequence; verify the temporal, height, and width rotations are applied to the correct dimension ranges
  - Test case 4 — video input: construct a batch with multi-frame video tokens; verify the temporal position index increments correctly across frames while height/width positions repeat per-frame
  - PCC thresholds: > 0.9999 for BF16 against float32 reference (tight threshold because rotation errors accumulate across heads and layers)

- `tracing_and_program_cache_considerations.md`
  - Impact of M-RoPE on Metal Trace capture: if position IDs are variable across decode steps (which they are — each new token appends to the sequence), they must be passed as device tensors (not baked into the trace)
  - The three `ttnn.embedding` lookups use the position ID tensor as a runtime input; this is trace-compatible if the table (embedding weight) is a fixed device tensor and only the index tensor varies
  - Program cache: the embedding lookup kernel is keyed on the table shape and index tensor shape; for decode with `seq_len=1` and fixed `batch`, the kernel shape is constant and the program cache hit rate is 100%
  - Prefill: the sequence length varies per input; program cache misses will occur for new sequence lengths; this is acceptable for prefill but should be noted in performance documentation
  - Recommendation: implement position IDs as `cur_pos_tensor`-style device int32 tensors to maintain trace compatibility; document the shape contract clearly

---

## 3. Conventions

### Terminology

| Term | Definition used in this guide |
|---|---|
| `rotary_dim` | The number of head dimensions that receive rotary encoding: `rotary_dim = floor(head_dim * partial_rotary_factor)`; for Qwen3.6 text layers, `head_dim=128`, `partial_rotary_factor=0.5` gives `rotary_dim=64` |
| `partial_rotary_factor` | The fraction of `head_dim` that receives RoPE rotation; dimensions `[rotary_dim:]` pass through unchanged |
| `mrope_section` | A list of three integers `[s_t, s_h, s_w]` that partition `rotary_dim/2` rotation pairs into temporal, height, and width sub-groups; `s_t + s_h + s_w == rotary_dim / 2` |
| Temporal section | The first `s_t` rotation pairs of the cos/sin table; indexed by the temporal coordinate (frame index or text position) |
| Height section | The next `s_h` rotation pairs; indexed by the height coordinate (patch row or text position for text tokens) |
| Width section | The final `s_w` rotation pairs; indexed by the width coordinate (patch column or text position for text tokens) |
| Position ID triplet | A tuple `(t, h, w)` specifying the three coordinate values for one token; for text tokens, `t == h == w` |
| 3D position IDs | The `[3, batch, seq_len]` integer tensor providing one position coordinate per axis per token; axis 0 = temporal, axis 1 = height, axis 2 = width |
| Degenerate M-RoPE | M-RoPE with all three axes having identical position values; mathematically equivalent to standard 1D RoPE |
| Standard RoPE | 1D RoPE: each token has a single scalar position index; cos/sin table is indexed uniformly across all rotary dimensions |
| Partial RoPE | Standard RoPE applied to the first `rotary_dim` of `head_dim` dimensions; the remaining dimensions are passed through unchanged |
| `TTNNRotaryPositionEmbedding` | The existing non-distributed RoPE class in tt-symbiote; used when `partial_rotary_factor < 1.0` |
| `TTNNDistributedRotaryPositionEmbedding` | The sharded RoPE class in tt-symbiote; used when `partial_rotary_factor == 1.0` (full rotation) |
| rotate-half | The standard RoPE application operation: for a pair `(x_{2i}, x_{2i+1})`, computes `(x_{2i} * cos - x_{2i+1} * sin, x_{2i} * sin + x_{2i+1} * cos)` |
| BF16 | BFloat16; the standard compute dtype for activations in TTNN |
| PCC | Pearson Correlation Coefficient; used in TTNN tests to measure output correctness vs. a reference implementation |
| T3K | Tenstorrent Galaxy board with 8 Wormhole chips in a 1x8 mesh |
| DRAM | Device DRAM on each Wormhole chip; where the cos/sin table and model weights are stored |
| L1 | On-chip SRAM on each Wormhole core; where activations are staged for compute |
| `ttnn.embedding` | TTNN op that performs integer-indexed row lookup from a 2D weight table; recommended for M-RoPE cos/sin gather |

### Notation

- Tensor shapes use square brackets with `x` as separator: `[3 x batch x seq_len]`.
- Config parameter names use their exact HuggingFace attribute names in `code font`: `rope_theta`, `mrope_section`, `partial_rotary_factor`.
- Section index ranges use half-open interval notation: pairs `[d_start, d_end)`.
- TTNN op names use the `ttnn.` prefix.
- Dimensions in the frequency table and cos/sin table use "pairs" to refer to one complex rotation (2 real dimensions); `rotary_dim` is the count of real dimensions (2 per pair), so there are `rotary_dim / 2` pairs.
- Layer indices are 0-based.
- Forward references to other chapters use relative markdown links: `../ch3_text_only_reduction/mathematical_equivalence_proof.md`.

### Formatting Rules

- Every chapter directory has an `index.md` with a chapter overview, learning objectives, and navigation to sub-topics.
- Warnings about silent failure modes (no error raised, numerically wrong output) are formatted as `> **[SILENT FAILURE]** ...` blockquotes.
- Key findings that directly answer the original research questions are formatted as `> **Key Finding:** ...` blockquotes.
- Performance numbers are estimates unless labeled as "measured"; always specify the hardware and firmware version.
- Equations use LaTeX fences: ` ```math ` / ` ``` `.
- Code examples use Python syntax with `# comment` annotations on non-obvious lines.
- Comparison tables use GitHub-flavored pipe Markdown with `Standard RoPE | M-RoPE` columns where applicable.
- Section headings use `##` (H2) and `###` (H3); no deeper nesting within files.

---

## 4. Cross-Chapter Dependencies

```
Ch1 (Rope Foundations)
  ├── Ch2 (Qwen3.6 Config & HF Reference)    — uses M-RoPE math from Ch1; instantiates with concrete parameters
  │     │
  │     └── Ch3 (Text-Only Reduction)         — uses the degenerate-case math from Ch1 and HF position ID construction from Ch2
  │
  └── Ch4 (TTNN Implementation)               — uses Ch1 math to identify what changes; uses Ch2 HF walkthrough to derive required TTNN ops
        │
        └── Ch5 (Performance Analysis)         — uses Ch4 operation list to count dispatches and data movement
              │
              └── Ch6 (Integration & Testing)  — synthesizes Ch2 (reference), Ch3 (text-only behavior), Ch4 (implementation options), Ch5 (performance guidance)
```

**Explicit dependencies by chapter:**

- **Chapter 2** requires: the section-partition math (Ch1 `section_dimension_assignment.md`) to correctly interpret `mrope_section = [11, 11, 10]` and to explain the HuggingFace `apply_multimodal_rotary_pos_emb` function.
- **Chapter 3** requires: Ch1's mathematical formulation of M-RoPE (what happens when all three position ID rows are equal) and Ch2's description of how HuggingFace constructs text-only position IDs.
- **Chapter 4** requires: Ch1's section-dimension assignment (which frequency pairs map to which coordinate) to identify the new TTNN indexing logic; Ch2's `hf_reference_implementation.md` to derive the exact gather pattern needed.
- **Chapter 5** requires: Ch4's operation list (`gather_operation_on_ttnn.md`, `pre_computed_cos_sin_strategy.md`) to count kernel launches and memory accesses; Ch3's finding that text-only M-RoPE is equivalent to standard RoPE (determines whether the performance analysis applies to text-only batches).
- **Chapter 6** requires: Ch3's finding (no changes needed for text-only inference) to scope the integration work correctly; Ch4's implementation options (extension vs. new class) to define the integration steps; Ch5's recommendation to start with naive gather and profile before optimizing; Ch2's position ID construction logic to implement the Qwen VL position ID builder.

**Specific forward references to flag:**

- **Ch1 → Ch3:** `section_dimension_assignment.md` derives the section partition; Ch3 `mathematical_equivalence_proof.md` uses it to show that equal positions across all sections produces standard RoPE output — flag readers in Ch1 to see Ch3 for the text-only implication.
- **Ch2 → Ch4:** `hf_reference_implementation.md` describes the three-way gather pattern; Ch4 `gather_operation_on_ttnn.md` maps this to `ttnn.embedding` — flag in Ch2 that the TTNN mapping is deferred to Ch4.
- **Ch3 → Ch6:** `practical_implications_for_text_inference.md` establishes that no code changes are needed for text-only; Ch6 `integration_steps.md` uses this to scope the integration — flag in Ch3 that the integration work is described in Ch6.
- **Ch4 → Ch5:** `kernel_launch_overhead.md` in Ch5 references the dispatch count derived in Ch4 `extension_approach.md` — flag in Ch4 that the performance cost of the 5-additional-dispatch design is quantified in Ch5.
- **Ch5 → Ch6:** `prefill_vs_decode_comparison.md` recommends profiling before optimizing; Ch6 `tracing_and_program_cache_considerations.md` gives the concrete tracing guidance — flag in Ch5 that tracing compatibility is addressed in Ch6.
