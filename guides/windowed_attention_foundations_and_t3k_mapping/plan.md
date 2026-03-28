# Plan: Windowed Attention — Foundations and T3K Mapping

## Audience

This guide targets ML systems engineers and kernel developers working on the
TT-NN / tt-transformers stack who need to integrate windowed (sliding window)
attention into production inference on the T3K 1×8 Wormhole mesh.

**Assumed knowledge:**
- Familiarity with standard scaled dot-product attention (SDPA) and causal masking
- Working knowledge of TT-NN tensor operations and program configs
- Basic understanding of the T3K device topology (8 Wormhole chips connected in
  a 1×8 mesh over Ethernet)
- Exposure to KV cache concepts for autoregressive decode

**Not assumed:**
- Prior experience with windowed / sliding-window attention
- Knowledge of paged KV cache internals in tt-transformers
- Roofline analysis methodology for Wormhole cores

---

## Chapter List

### Chapter 1 — Mathematical Foundations of Windowed Attention

**Description:** Derives the sliding-window attention formulation from first
principles and contrasts it precisely with full causal attention.

**Directory:** `ch1_math_foundations/`

**Files:**

- `index.md`
  - Overview of the chapter and reading order
  - Motivation: why window-bounding the receptive field reduces both compute
    and memory without sacrificing local context

- `full_vs_windowed_attention.md`
  - Formal definition of full causal attention: query attends to all previous
    tokens `[0, t]`
  - Formal definition of windowed attention: query at position `t` attends only
    to tokens in `[t - w + 1, t]` where `w` is the window size
  - Side-by-side attention mask diagrams for both variants (ASCII art)
  - Mathematical expression of the masked softmax with the window constraint
  - Complexity table: full attention O(T²) vs windowed O(T·w) for prefill;
    O(T) vs O(w) KV reads per decode step

- `window_size_parameter.md`
  - Definition and typical values of `w` (e.g., 4096, 8192 for Qwen/Mistral)
  - Relationship between `w`, model context length, and effective receptive field
  - Discussion of "sink tokens" (global attention on token 0) used by some models
    (StreamingLLM / Mistral style) and how they extend the formulation
  - How `w` is stored in model config and surfaced in tt-transformers

---

### Chapter 2 — KV Cache Management During Decode

**Description:** Explains how windowed attention changes KV cache lifecycle —
insertion, eviction, and steady-state size — compared with full attention.

**Directory:** `ch2_kv_cache_management/`

**Files:**

- `index.md`
  - Overview and connection to Chapter 1 definitions

- `kv_cache_lifecycle.md`
  - How a standard KV cache grows token-by-token during decode
  - Windowed eviction: once the cache holds `w` entries the oldest entry is
    evicted on each new token
  - Steady-state KV cache size: `w * num_heads * head_dim * 2 * dtype_bytes`
    vs full-attention cache size at position T
  - Memory saving factor as a function of generation length T and window w

- `circular_buffer_layout.md`
  - Implementation of the eviction policy as a circular (ring) buffer in device
    DRAM
  - Index arithmetic: write pointer, read pointer, wrap-around logic
  - How the circular buffer maps to a fixed-shape TTNN tensor
    `[batch, num_heads, w, head_dim]` with an associated position offset scalar
  - Contrast with the grow-in-place tensor used by full-attention KV caches

---

### Chapter 3 — Data Dependencies and Memory Access Patterns

**Description:** Characterises the memory access patterns for windowed attention
during prefill and decode and determines whether a specialised kernel is needed.

**Directory:** `ch3_data_dependencies/`

**Files:**

- `index.md`
  - Chapter overview and pointers to kernel implications in later chapters

- `prefill_access_patterns.md`
  - Prefill scenario: full input sequence of length T processed in one pass
  - How the window constraint becomes a band-diagonal mask on the `[T, T]`
    attention score matrix
  - Memory access pattern: each query row reads a contiguous stripe of K/V rows
  - Whether this can be expressed as a masked full-attention kernel (yes, with
    a band mask) vs requiring a tiled streaming kernel
  - Arithmetic intensity analysis for prefill: FLOPs per byte of KV loaded,
    dependency on `w` and `T`

- `decode_access_patterns.md`
  - Decode scenario: single query vector attends to the `w`-token KV window
  - Memory access: read exactly `w` K vectors and `w` V vectors from DRAM
  - Comparison with full-attention decode which reads `T` K/V pairs
  - Bandwidth reduction factor: `w / T` at generation step T
  - Data dependency graph: no inter-token dependencies within the window read

---

### Chapter 4 — TTNN Primitive Operations and Tensor Shapes

**Description:** Maps the windowed attention algorithm onto concrete TTNN ops,
program configs, and tensor shapes for both decode and prefill paths.

**Directory:** `ch4_ttnn_primitives/`

**Files:**

- `index.md`
  - Chapter overview and mapping to Q3 and Q4 from the research questions

- `decode_primitives.md`
  - TTNN ops for the single-token decode path: `ttnn.matmul` (QK), softmax,
    `ttnn.matmul` (AV)
  - Input tensor shapes with the circular-buffer KV layout:
    Q `[batch, num_heads, 1, head_dim]`,
    K/V `[batch, num_heads, w, head_dim]`
  - How the position offset scalar is passed to the kernel to select the correct
    slice of K/V in the circular buffer
  - Relevant program config knobs: `compute_with_storage_grid_size`,
    `in0_block_w`, per-core tile counts

- `prefill_primitives.md`
  - TTNN ops for the full-sequence prefill path
  - Band-mask generation: how to construct the `[T, T]` band-diagonal mask as
    a TTNN tensor
  - Whether `ttnn.scaled_dot_product_attention` with a mask argument covers the
    band mask case, or whether a custom Flash-Attention style tiled kernel is
    needed
  - Tensor shape implications: KV output written to the circular-buffer tensor
    as a side-effect of prefill
  - Trade-offs: full-size masked kernel vs chunked windowed kernel

- `kernel_or_op_gap_analysis.md`
  - Survey of existing TTNN / tt-transformers ops that accept attention masks
  - Determination of what is available vs what gaps remain for windowed patterns
  - Placeholder for Q8 findings (cross-references Chapter 7)

---

### Chapter 5 — Paged KV Cache Interaction

**Description:** Analyses how windowed attention interacts with the paged KV
cache in tt-transformers and whether `paged_sdpa_decode` can enforce a window
constraint.

**Directory:** `ch5_paged_kv_cache/`

**Files:**

- `index.md`
  - Chapter overview; requires understanding of circular buffer layout from
    Chapter 2

- `paged_sdpa_and_windowing.md`
  - Brief recap of the paged KV cache model: page tables, block size, virtual
    vs physical page mapping
  - How `paged_sdpa_decode` currently selects which pages to load for a given
    sequence position
  - Two strategies for combining paging with windowing:
    (a) Page-aware windowing — page table entries for tokens outside the window
        are simply not loaded; requires page table to encode recency
    (b) Circular-buffer-as-pages — allocate exactly `ceil(w / block_size)` pages
        per sequence slot and overwrite them in round-robin order
  - Analysis of which strategy is more compatible with the existing
    `paged_sdpa_decode` interface
  - Required interface changes or new program config fields

- `eviction_and_page_reuse.md`
  - How page eviction maps to the window eviction policy
  - Risk of stale page table entries when a sequence grows beyond `w` tokens
  - Correctness invariants that the paging layer must maintain
  - Memory fragmentation implications of fixed-size windowed page pools

---

### Chapter 6 — T3K Mesh Sharding and CCL Implications

**Description:** Defines the sharding strategy for windowed KV caches across
the 8-device T3K mesh and identifies the collective communication operations
required.

**Directory:** `ch6_t3k_sharding/`

**Files:**

- `index.md`
  - Chapter overview; requires Chapters 1–4 for tensor shape definitions

- `sharding_strategies.md`
  - Recap of T3K topology: 1×8 mesh, 8 Wormhole chips, Ethernet links
  - Two candidate sharding strategies for the windowed KV cache:
    (a) Head-parallel sharding — each device holds a disjoint subset of
        attention heads; full window `w` replicated per device for its heads
    (b) Sequence-parallel sharding — each device holds a `w/8` slice of the
        window; requires all-gather before attention
  - Decision matrix: memory per device, CCL volume, latency impact for each
  - Recommendation and rationale

- `ccl_operations.md`
  - Which CCL primitives (`ttnn.all_gather`, `ttnn.reduce_scatter`) are used
    under each sharding strategy
  - Bandwidth requirements: Ethernet link speed on T3K vs KV data volume per
    decode step
  - Overlap opportunities: can CCL be pipelined with compute?
  - Impact of window size `w` on CCL volume relative to full attention

- `per_device_window_application.md`
  - Whether the window boundary is applied globally before sharding or
    per-device after sharding
  - Correctness analysis: ensuring that the union of per-device windows equals
    the intended global window
  - Edge cases at window boundaries when `w` is not divisible by 8

---

### Chapter 7 — Roofline Analysis and Existing Kernel Survey

**Description:** Determines whether windowed attention decode is compute-bound
or bandwidth-bound on Wormhole for typical window sizes and inventories existing
TTNN support.

**Directory:** `ch7_roofline_and_kernels/`

**Files:**

- `index.md`
  - Chapter overview; connects to hardware performance questions (Q7, Q8)

- `roofline_analysis.md`
  - Wormhole hardware numbers: peak FP16 TFLOPS per chip, DRAM bandwidth (GB/s)
  - Arithmetic intensity of windowed attention decode:
    FLOPs = `2 * batch * num_heads * w * head_dim` for QK + AV matmuls;
    bytes = KV load = `2 * batch * num_heads * w * head_dim * dtype_bytes`
  - Arithmetic intensity formula and roofline intersection point
  - Conclusion: bandwidth-bound at batch=1 for all practical window sizes
  - Comparison table: windowed (w=4096, 8192) vs full attention at T=4096,
    T=8192, T=32768 — bandwidth requirement and expected throughput
  - Implication: performance scales linearly with `w`, not with `T`

- `existing_kernel_survey.md`
  - Survey of `ttnn.scaled_dot_product_attention` and
    `ttnn.scaled_dot_product_attention_decode`
  - Survey of `paged_sdpa` / `paged_sdpa_decode` in tt-transformers
  - Survey of Flash-Attention style chunked kernels in TTNN
  - For each primitive: does it accept an attention mask? Can the mask encode a
    band/window constraint? What are the tensor shape restrictions?
  - Gap summary: list of what is missing for full windowed attention support
  - Recommended path: extend existing op vs write new program config vs new
    kernel

---

## Conventions

### Terminology

| Term | Definition |
|------|------------|
| `w` | Window size — the number of past tokens (including the current token) that each query attends to |
| `T` | Total sequence length at a given decode step |
| `d` | Head dimension (`head_dim`) |
| `H` | Number of attention heads |
| `B` | Batch size |
| Circular buffer | Fixed-size `[B, H, w, d]` KV tensor written in round-robin fashion |
| Full attention | Standard causal attention where query at position `t` attends to all tokens `[0, t]` |
| Windowed attention | Attention where query at position `t` attends only to tokens `[t-w+1, t]` |
| Prefill | Processing the full prompt sequence in a single forward pass |
| Decode | Autoregressive single-token generation steps |
| T3K | Tenstorrent 3000-series board with 8 Wormhole chips in a 1×8 mesh |
| CCL | Collective Communication Library (TTNN's multi-device communication layer) |
| SDPA | Scaled Dot-Product Attention |
| `paged_sdpa` | Paged variant of SDPA in tt-transformers supporting virtual page tables |

### Notation

- Tensor shapes are written in square brackets with named dimensions:
  `[B, H, S, d]` where S is sequence length.
- Complexity expressions use big-O with explicit variables, e.g., O(B · H · w · d).
- All memory sizes are given in bytes unless otherwise noted, with dtype
  specified (e.g., BF16 = 2 bytes/element).
- TTNN op names are written in code font: `ttnn.matmul`, `ttnn.all_gather`.
- File paths and symbol names use inline code font: `paged_sdpa_decode`.

### Formatting Rules

- Every file begins with a `# Title` H1 header matching the filename topic.
- Section headers use `##` (H2) and `###` (H3); no deeper nesting.
- Diagrams are ASCII art inline in Markdown code blocks labelled with
  `text` language tag.
- Tables use standard GitHub-flavoured Markdown pipe syntax.
- All equations are written in LaTeX math fences (` ```math ``` `) for
  compatibility with the repository's Markdown renderer.
- Cross-chapter references use relative paths:
  `../ch1_math_foundations/full_vs_windowed_attention.md`.
- No external links to papers or URLs — findings must be self-contained.

---

## Cross-Chapter Dependencies

```
Ch1 (Math Foundations)
  └── Ch2 (KV Cache Management)          — uses window size w, token indexing
        └── Ch3 (Data Dependencies)      — uses circular buffer layout, KV tensor shapes
              └── Ch4 (TTNN Primitives)  — uses tensor shapes from Ch2, access patterns from Ch3
                    └── Ch5 (Paged KV)  — uses circular buffer from Ch2, TTNN shapes from Ch4
                    └── Ch6 (T3K Mesh)  — uses tensor shapes from Ch4
Ch7 (Roofline & Kernels)
  ├── references hardware numbers (self-contained)
  ├── references tensor shapes from Ch4
  └── cross-references gap analysis in Ch4 kernel_or_op_gap_analysis.md
```

**Explicit dependencies by chapter:**

- **Chapter 2** requires: definition of `w` and the windowed attention mask
  from Chapter 1.
- **Chapter 3** requires: circular buffer layout introduced in Chapter 2;
  windowed KV tensor shape `[B, H, w, d]`.
- **Chapter 4** requires: tensor shapes from Chapter 2; memory access patterns
  from Chapter 3 to motivate op choices.
- **Chapter 5** requires: circular buffer concept from Chapter 2; TTNN tensor
  shapes and op interface from Chapter 4.
- **Chapter 6** requires: final tensor shapes from Chapter 4 to compute shard
  sizes and CCL volumes; prefill vs decode distinction from Chapter 3.
- **Chapter 7** requires: FLOP and byte counts derived from tensor shapes in
  Chapter 4; op list from Chapter 4 for the kernel survey.
