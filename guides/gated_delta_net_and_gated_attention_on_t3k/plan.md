# Plan: Gated Delta Net and Gated Attention on T3K

## 1. Audience

**Primary reader:** An ML systems engineer or hardware-aware model developer who:
- Understands transformer attention at the level of `Q @ K^T @ V`, can read TTNN Python APIs, and is comfortable with tensor shape notation
- Has passing familiarity with the T3K 1×8 Wormhole mesh (knows what a device mesh is, understands tensor sharding and CCL all-gather/reduce-scatter at a conceptual level)
- Has not previously studied linear recurrent attention variants (DeltaNet, GLA, Mamba, RetNet) and does not know how the Gated Delta Net recurrence structure differs from standard softmax attention or how to map it onto hardware

**What the reader already knows:**
- Standard causal multi-head / grouped-query attention (full KV cache, O(n²) attention per layer, SDPA kernels)
- TTNN tensor operations and program configs at an introductory level (can read `ttnn.matmul`, `ttnn.mul`, `ttnn.rms_norm` signatures)
- Basic roofline model intuition (arithmetic intensity, memory-bound vs. compute-bound regimes, ridge point)
- How a KV cache works for standard softmax attention (key and value tensors growing with sequence length)
- The Qwen3.5-35B-A3B / Qwen3-Coder-Next hybrid layer structure at a high level (knows it mixes two attention types)

**What the reader will learn from this guide:**
- The precise mathematical formulation of the Gated Delta Net recurrence (state matrix S, delta rule update, scalar decay gate g, beta update rate) and how it differs from standard softmax attention, RetNet, GLA, and Mamba2
- The Gated Attention mechanism specific to these models (Q gating with sigmoid, Q/K RMS normalization, GQA head configuration, RoPE)
- The parallelism structure of Gated Delta Net: when the state update is strictly sequential and when chunkwise parallel training (WY-decomposition) or associative scan can be used
- The TTNN primitive operations needed for decode and prefill, with complete tensor shapes at each step
- A quantitative memory comparison between the Gated Delta Net recurrent state and the KV cache of the co-located Gated Attention layer on T3K L1 and DRAM
- Whether the decode step of Gated Delta Net is compute-bound or bandwidth-bound on Wormhole, and what the compute-to-bandwidth ratio implies for kernel design
- A concrete tensor parallelism sharding strategy for the Gated Delta Net state matrix across the 8 T3K devices, with CCL cost analysis
- An audit of existing TTNN and tt-transformers primitives against the Gated Delta Net and Gated Attention forward passes, identifying which operations are available, which require new fused kernels, and what the recommended development path is

---

## 2. Chapter List

---

### Chapter 1 — Attention Variants and Linear Attention Foundations

**Description:** Establishes the mathematical landscape of attention mechanisms — from standard softmax attention through RetNet, GLA, and Mamba2 — so the reader has a precise reference frame before encountering Gated Delta Net.

**Directory:** `ch1_attention_variants_and_linear_attention_foundations/`

**Files and content:**

- **`index.md`** — Chapter overview, learning objectives, and navigation links to each section file.
  - Explains why the chapter exists: the reader needs a clear taxonomy of linear recurrent attention variants before Gated Delta Net can be meaningfully positioned
  - Lists the four section files and what each covers
  - Defines the chapter's key take-away: linear attention replaces the O(n²) attention matrix with a fixed-size state matrix S ∈ R^{d_k × d_v}, enabling O(1) decode at the cost of limited expressiveness; different variants differ in how they update S

- **`standard_softmax_attention.md`** — Recap of standard causal multi-head and GQA attention as the baseline.
  - Full SDPA formula: `Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V` where M is the causal mask
  - Interpretation as memory retrieval: the attention pattern selects which past token values to mix; the KV cache grows as O(n × d_k) per layer per batch element
  - Complexity: O(n² · d_k) FLOPs prefill, O(n · d_k) FLOPs per decode step (attending to all n past tokens), O(n · d_k) KV cache bytes (growing with n)
  - GQA: n_kv_heads < n_q_heads; key and value heads are shared across query head groups; this is the configuration used by Gated Attention layers in Qwen3.5

- **`linear_attention_rnn_equivalence.md`** — Derivation of linear attention as a fixed-size RNN.
  - Replace softmax with a kernel function φ: `Attn(q_t, K, V) = φ(q_t)^T (sum_j φ(k_j) v_j^T) / φ(q_t)^T (sum_j φ(k_j))`
  - Associativity: the numerator sum can be written as a state matrix S = sum_{j≤t} φ(k_j) v_j^T ∈ R^{d_k × d_v}, updated as S_t = S_{t-1} + v_t k_t^T
  - RNN recurrence: S_t = S_{t-1} + v_t k_t^T; o_t = S_t q_t; no sequence-length dependency at decode time
  - Limitation of vanilla linear attention: S accumulates all past tokens equally; no forgetting mechanism; retrieval quality degrades with sequence length as S becomes a superposition of all past writes

- **`linear_attention_variants_comparison.md`** — Side-by-side comparison of RetNet, GLA, Mamba2, and DeltaNet state update equations.
  - General form: `S_t = G_t ⊙ S_{t-1} + v_t k_t^T` where G_t is the forgetting gate matrix
  - RetNet: G_t = γ · I (scalar decay, position-independent); scalar γ ∈ (0,1) shrinks all state entries uniformly; no data-dependence
  - GLA (Gated Linear Attention, Yang et al. ICML 2024): G_t = 1 α_t^T (outer product gate); α_t ∈ R^{d_k} is input-dependent; provides column-wise data-dependent decay; state update includes a new v_t k_t^T write
  - Mamba2: G_t = γ_t · 1 1^T (scalar per step, broadcast); γ_t is input-dependent but applies uniformly to all state entries; enables fast parallel scan via structured state-space models
  - DeltaNet (standard): no explicit forgetting gate; instead applies the delta rule — S_t = S_{t-1} - β_t (S_{t-1} k_t - v_t) k_t^T; minimizes squared error between S k_t and v_t; provides targeted error-correcting writes but no coarse forgetting
  - Summary table: variant | gating type | data-dependent | forgetting | write mechanism
  - Forward reference to Chapter 2 where Gated Delta Net combines gating and delta rule

---

### Chapter 2 — Gated Delta Net: Mathematical Formulation and Recurrence Structure

**Description:** Derives the complete Gated Delta Net state update from first principles, defines all tensors and their shapes for the Qwen3.5-35B-A3B configuration, and analyzes whether and how the recurrence can be parallelized across the sequence dimension.

**Directory:** `ch2_gated_delta_net_math_and_recurrence/`

**Files and content:**

- **`index.md`** — Chapter overview and learning objectives.
  - States that this chapter answers Q1 and Q3 from the guide spec
  - Clarifies that the chapter covers Gated Delta Net only; Gated Attention is deferred to Chapter 3
  - Lists the three section files and the key formulas each contains

- **`gated_delta_rule_formulation.md`** — Complete mathematical derivation of the Gated Delta Net recurrence.
  - Motivation: combine GLA's coarse data-dependent forgetting with DeltaNet's precise error-correcting writes; each mechanism alone is insufficient for long-range retrieval and selective forgetting simultaneously
  - Core recurrence for a single head at step t:
    ```
    g_t  = exp(α_t)                       α_t ∈ R  (scalar decay, < 0, so g_t ∈ (0,1])
    β_t  = σ(b_t)                         β_t ∈ (0,1)  (update rate)
    S_t  = g_t · S_{t-1} + k̃_t ⊗ [β_t · (v_t − g_t · S_{t-1}^T k̃_t)]
    o_t  = S_t q̃_t
    ```
    where k̃_t and q̃_t are L2-normalized key and query vectors, ⊗ denotes an outer product (rank-1 write), and S_t ∈ R^{d_k × d_v}
  - Interpretation of each term:
    - `g_t · S_{t-1}`: decays all state entries by the same scalar; coarse forgetting (GLA-style)
    - `g_t · S_{t-1}^T k̃_t`: retrieves what the decayed state predicts for key k̃_t; this is the "predicted value"
    - `β_t · (v_t − g_t · S_{t-1}^T k̃_t)`: the prediction error scaled by β_t; this is the delta rule correction
    - outer product write: the correction is written back with key k̃_t as the addressing vector
    - Result: the state is first decayed (forgetting), then corrected toward the new value (learning)
  - Comparison to standard DeltaNet (no gating): standard DeltaNet sets g_t = 1 (no decay) and applies only the delta correction; Gated Delta Net adds the g_t scalar decay before the correction
  - Comparison to GLA: GLA applies a column-wise outer-product gate and adds v_t k_t^T directly; no error-correction term; Gated Delta Net replaces the direct write with the delta error term
  - State matrix dimensions for Qwen3.5-35B-A3B per head: S ∈ R^{128 × 128} = R^{d_k × d_v}; full multi-head state per layer: [B, num_v_heads, d_k, d_v] = [B, 32, 128, 128]; size in BF16: B × 32 × 128 × 128 × 2 bytes = B × 1,048,576 bytes ≈ B × 1 MB per layer
  - Decay gate derivation: α_t = −exp(A_log) · softplus(a_t + dt_bias); A_log is a learned per-head scalar parameter (log-space); a_t is an input-dependent scalar from in_proj_a projection; dt_bias is a learned per-head bias; the overall α_t is strictly negative, ensuring g_t ∈ (0,1]
  - Full projection inventory for one Gated Delta Net layer (Qwen3.5-35B-A3B):
    - in_proj_qkv: [B, T, 2048] → [B, T, key_dim×2 + value_dim] = [B, T, 2×2048 + 4096] = [B, T, 8192]
    - Split: Q [B, T, 2048], K [B, T, 2048], V [B, T, 4096]
    - Reshape: Q [B, T, 16, 128], K [B, T, 16, 128], V [B, T, 32, 128]
    - K/Q repeated × 2 to match V heads: Q̃ [B, T, 32, 128], K̃ [B, T, 32, 128]
    - in_proj_z: [B, T, 2048] → [B, T, 4096] (output gate Z for gated RMSNorm)
    - in_proj_a: [B, T, 2048] → [B, T, 4] (log-decay input a; num_v_heads=4 here for 35B, 32 for 9B config — use Qwen3.5-35B-A3B actual value from model config)
    - in_proj_b: [B, T, 2048] → [B, T, num_v_heads] (beta logit)
    - conv1d: causal 1D convolution over [B, key_dim×2 + value_dim, T] with kernel size 4
    - out_proj: [B, T, 4096] → [B, T, 2048]

- **`parallelism_and_scan.md`** — Analysis of the recurrence structure and strategies for parallelizing across the sequence dimension.
  - The recurrence S_t = g_t · S_{t-1} + k̃_t ⊗ [β_t · (v_t − g_t · S_{t-1}^T k̃_t)] is **not associative** in its raw form because the correction term depends on S_{t-1}^T k̃_t — the state itself appears inside the write vector
  - Consequence: naive parallel scan cannot be applied directly; each step has a true sequential dependency through S_{t-1}
  - Chunkwise parallel training (used in practice): divide the sequence of length T into chunks of size C (e.g., C=64); within each chunk, use the WY-decomposition to express the multi-step update as a low-rank correction to the chunk-initial state; across chunks, the inter-chunk recurrence is a scalar-gated state transfer that can be computed with a parallel prefix scan
  - WY-decomposition: within a chunk, the combined update S_{t+C} = (product of g scalars) · S_t + U W^T where U and W are [d_k, C] and [d_v, C] matrices; these can be computed in parallel using triangular masking
  - Complexity: O(T · C · d_k · d_v) FLOPs with O(B × num_heads × d_k × d_v) state memory; the C-sized chunk enables tensor-core utilization
  - Associative scan feasibility: the inter-chunk recurrence S_{t+C} = γ_C · S_t + Δ_C (scalar-times-matrix plus matrix) is associative and can be parallelized over chunks; however, each operand is a full d_k × d_v matrix, so the associative scan over T/C chunks requires O(T/C × d_k × d_v) memory — for T=256K, C=64, d_k=d_v=128: ~4M state matrices in DRAM, which is prohibitive
  - Practical parallelism on hardware: the chunk-level parallelism (C=64 chunk, vectorized over heads) maps well to tensor cores; the scan over chunks is done sequentially; this is what `chunk_gated_delta_rule` (used during prefill) implements
  - Decode: strictly sequential; one step of `recurrent_gated_delta_rule` per new token; O(d_k × d_v) = O(128×128) state matrix-vector multiply per head per step
  - Summary: Gated Delta Net is parallelizable across chunks during prefill (similar to FlashAttention's tiling) but sequential across chunks; fully sequential per-token at decode

- **`state_vs_kv_cache_memory.md`** — Quantitative memory comparison: Gated Delta Net state vs. standard attention KV cache.
  - Gated Delta Net state per layer: S ∈ R^{B × num_v_heads × d_k × d_v} = [B, 32, 128, 128]; bytes = B × 32 × 128 × 128 × 2 = B × 1,048,576 ≈ B × 1 MB per layer (independent of sequence length T)
  - Conv state per layer: [B, key_dim×2 + value_dim, conv_kernel_size] = [B, 8192, 4] = B × 65,536 elements × 2 bytes = B × 128 KB per layer
  - Total linear attention state per DeltaNet layer: ≈ B × 1.125 MB (recurrent + conv)
  - Gated Attention KV cache per layer (for 10 full-attention layers in Qwen3.5-35B-A3B): [B, n_kv_heads, T, head_dim] = [B, 2, T, 256]; bytes = B × 2 × T × 256 × 2 = B × 1024 × T bytes
  - Crossover point: Gated Delta Net state exceeds Gated Attention KV cache when B × 1.125 MB > B × 1024 × T, i.e., when T < ~1098 tokens; for any sequence longer than ~1K tokens, the standard KV cache is more expensive than the DeltaNet state
  - At T = 256K (max context for Qwen3-Coder-Next): Gated Attention KV cache = B × 1024 × 256K × 2 = B × 512 MB per layer; Gated Delta Net state = B × 1.125 MB — the DeltaNet state is ~455× smaller
  - 30 DeltaNet layers × B × 1.125 MB = B × 33.75 MB total DeltaNet state
  - 10 Gated Attention layers at T = 256K: 10 × B × 512 MB = B × 5120 MB — dwarfs the DeltaNet state
  - T3K total DRAM: 8 × 12 GB = 96 GB; BF16 model weights alone are ~70 GB; DeltaNet state at B=1 is ~33.75 MB — fits comfortably in DRAM; at B=32 it is ~1.08 GB, still manageable
  - L1 SRAM per Tensix core: 1.5 MB; full DeltaNet state per layer (B=1) = 1.125 MB — too large for a single core's L1; must be distributed across cores or streamed from DRAM during the state-matrix multiply

---

### Chapter 3 — Gated Attention: Mechanism and Tensor Shapes

**Description:** Defines the Gated Attention mechanism used in Qwen3.5 / Qwen3-Coder-Next — specifically the sigmoid Q-gating, Q/K RMSNorm, GQA head configuration, and RoPE — and compares its tensor shapes to vanilla multi-head attention.

**Directory:** `ch3_gated_attention_mechanism/`

**Files and content:**

- **`index.md`** — Chapter overview and learning objectives.
  - Clarifies that "Gated Attention" in this guide refers specifically to the `Qwen3_5MoeGatedAttention` / `TTNNQwen3FullAttention` implementation, not GLA
  - States that this chapter answers Q2: what gating does, what tensor shapes it introduces, and how it differs from vanilla MHA/GQA
  - Lists the two section files

- **`gated_attention_formulation.md`** — Step-by-step derivation of the Gated Attention forward pass.
  - Input: hidden_states [B, T, H] where H = 2048 for Qwen3.5-35B-A3B
  - Combined QKV projection via a single fused linear (or separate Q and KV projections): output split into Q [B, T, n_q_heads, d_head], K [B, T, n_kv_heads, d_head], V [B, T, n_kv_heads, d_head], gate [B, T, n_q_heads, d_head]
    - Qwen3.5-35B-A3B: n_q_heads = 16, n_kv_heads = 2, d_head = 256 (note: larger d_head than DeltaNet's 128)
  - Q gating: gate_sigmoid = σ(gate); Q_gated = Q ⊙ gate_sigmoid — element-wise product of Q with a learned sigmoid gate; this is the defining "gating" feature of this attention variant; it enables input-dependent suppression of query vectors
  - Q normalization: Q_normed = RMSNorm(Q_gated) applied per head (weight vector shape [d_head])
  - K normalization: K_normed = RMSNorm(K) applied per head (weight vector shape [d_head])
  - RoPE application: rotary position embeddings applied to Q_normed and K_normed using the rotary_dim = 64 sub-dimension
  - KV cache update: K_normed and V are written into a paged KV cache indexed by layer_idx; only the 10 full-attention layers use KV cache
  - GQA expansion: K and V are repeated n_q_heads / n_kv_heads = 8× to match Q heads before SDPA
  - SDPA: `ttnn.transformer.scaled_dot_product_attention` or `scaled_dot_product_attention_decode` depending on phase
  - Output: reshape attn_output [B, T, n_q_heads, d_head] → [B, T, n_q_heads × d_head]; output projection o_proj: [B, T, n_q_heads × d_head] → [B, T, H]
  - The layer is in every 4th position (layers 3, 7, 11, ..., 39 in Qwen3.5-35B-A3B)

- **`gated_vs_vanilla_attention_shapes.md`** — Tensor shape comparison between vanilla MHA, standard GQA, and Gated Attention.
  - Shape table: for each variant, list Q shape, K/V shape, gate shape, post-norm shape, output shape at both prefill (T > 1) and decode (T = 1)
  - Key differences from vanilla MHA: (1) the gate tensor doubles the QKV projection output size; (2) Q and K are normalized per-head (RMSNorm weight [d_head]) before SDPA; (3) Q is gated element-wise with sigmoid(gate) before normalization; (4) n_kv_heads = 2 is very small (extreme GQA), so 8× repeat of KV is needed
  - Memory implication of the gate: the gate projection adds [B, T, n_q_heads, d_head] = [B, T, 16, 256] = B × T × 4096 elements × 2 bytes per layer; for T = 1 (decode) this is B × 8 KB — negligible; for T = 8192 (chunked prefill) it is B × 64 MB
  - Comparison to Gated Delta Net: Gated Delta Net uses head_dim = 128 with 32 V-heads and 16 K-heads; Gated Attention uses head_dim = 256 with 16 Q-heads and 2 KV-heads; both feed into the same hidden size H = 2048
  - Forward reference to Chapter 4 for the TTNN implementation of each step

---

### Chapter 4 — TTNN Primitive Mapping: Decode and Prefill Forward Passes

**Description:** Catalogs the specific TTNN operations required to implement one Gated Delta Net decode step, one Gated Delta Net prefill pass, and one Gated Attention forward pass, with complete tensor shapes and notes on which operations exist, which are currently falling back to PyTorch, and which need new kernels.

**Directory:** `ch4_ttnn_primitive_mapping/`

**Files and content:**

- **`index.md`** — Chapter overview and learning objectives.
  - States that this chapter answers Q4 and Q8 from the guide spec
  - Summarizes the current implementation state: TTNN handles linear projections (in_proj_qkv, in_proj_z, out_proj) and the full Gated Attention forward pass; the DeltaNet kernel (conv1d, chunk/recurrent gated delta rule, gated RMSNorm) currently falls back to PyTorch via `flash-linear-attention` / `causal-conv1d` libraries
  - Lists the four section files

- **`gated_delta_net_decode_step.md`** — Tensor shapes and TTNN operations for one Gated Delta Net decode step (B=1, T=1).
  - Input: hidden_states [1, 1, 2048]
  - Step 1 — Input projections (TTNN, col-sharded across 8 devices):
    - `in_proj_qkv`: [1, 1, 2048] → [1, 1, 8192] via `ttnn.linear` (replicated input, col-sharded weight)
    - `in_proj_z`: [1, 1, 2048] → [1, 1, 4096] via `ttnn.linear`
    - `in_proj_a`: [1, 1, 2048] → [1, 1, num_v_heads] via PyTorch nn.Linear (too small to shard)
    - `in_proj_b`: [1, 1, 2048] → [1, 1, num_v_heads] via PyTorch nn.Linear
  - Step 2 — All-gather (CCL): gather sharded in_proj_qkv and in_proj_z outputs across 8 devices; `ttnn.experimental.all_gather_async` on dim=-1
  - Step 3 — Causal conv1d update (PyTorch, `causal_conv1d_update`):
    - Input: mixed_qkv [1, 8192, 1], conv_state [1, 8192, 4]
    - Output: mixed_qkv [1, 8192, 1]; updated conv_state [1, 8192, 4]
  - Step 4 — Split and reshape:
    - Q [1, 1, 16, 128], K [1, 1, 16, 128], V [1, 1, 32, 128]
    - β = σ(b): elementwise sigmoid over [1, 1, num_v_heads] — would map to `ttnn.sigmoid` if on-device
    - α = −exp(A_log) · softplus(a + dt_bias): per-head scalar — would map to `ttnn.exp`, `ttnn.softplus`, `ttnn.mul` if on-device
    - g = exp(α): scalar decay per head
    - K/Q repeat_interleave × 2: K̃ [1, 1, 32, 128], Q̃ [1, 1, 32, 128]
  - Step 5 — Recurrent delta rule step (PyTorch, `recurrent_gated_delta_rule`):
    - Inputs: Q̃ [1, 1, 32, 128], K̃ [1, 1, 32, 128], V [1, 1, 32, 128], g [1, 1, 32], β [1, 1, 32], S_prev [1, 32, 128, 128]
    - Core operations:
      - `S_decayed = g · S_prev`: scalar-times-matrix per head, [1, 32, 128, 128] — maps to `ttnn.mul` (broadcast)
      - `retrieval = S_decayed^T k̃`: matrix-vector multiply [128, 128] × [128] per head — maps to `ttnn.matmul`
      - `error = β · (v − retrieval)`: elementwise scale of vector [128] — maps to `ttnn.mul`, `ttnn.sub`
      - `write = k̃ ⊗ error`: outer product [128] × [128] → [128, 128] per head — maps to `ttnn.matmul` with reshape
      - `S_new = S_decayed + write`: matrix add [128, 128] — maps to `ttnn.add`
      - `o = S_new q̃`: matrix-vector [128, 128] × [128] — maps to `ttnn.matmul`
    - Output: core_attn_out [1, 1, 32, 128], S_new [1, 32, 128, 128]
  - Step 6 — Gated RMSNorm (PyTorch, `FusedRMSNormSwishGate`):
    - Inputs: core_attn_out [1, 32, 128], z [1, 32, 128]
    - Output: normed_out [1, 32, 128] — would map to `ttnn.rms_norm` + `ttnn.mul(ttnn.sigmoid(z))` if on-device
  - Step 7 — Output projection (TTNN, col-sharded):
    - `out_proj`: [1, 1, 4096] → [1, 1, 2048] via `ttnn.linear`; all-gather to restore replicated output

- **`gated_delta_net_prefill_pass.md`** — Tensor shapes and TTNN operations for Gated Delta Net prefill (B=1, T=full sequence).
  - Input: hidden_states [1, T, 2048]
  - Step 1 — Input projections (TTNN): same as decode but with T > 1
    - `in_proj_qkv`: [1, T, 2048] → [1, T, 8192]
    - `in_proj_z`: [1, T, 2048] → [1, T, 4096]
    - `in_proj_a`: [1, T, 2048] → [1, T, num_v_heads]
    - `in_proj_b`: [1, T, 2048] → [1, T, num_v_heads]
  - Step 2 — Causal conv1d (prefill, `causal_conv1d_fn`):
    - Input: mixed_qkv transposed [1, 8192, T]
    - Output: mixed_qkv [1, 8192, T]; saves conv_state [1, 8192, 4] at end of sequence
  - Step 3 — Chunkwise delta rule (PyTorch, `chunk_gated_delta_rule`):
    - Inputs: Q̃ [1, T, 32, 128], K̃ [1, T, 32, 128], V [1, T, 32, 128], g [1, T, 32], β [1, T, 32]
    - Chunks of size C = 64: T/C chunks, each processed with WY-decomposition
    - Inter-chunk: sequential state transfer S_{c+1} = g_C · S_c + U_C W_C^T
    - Output: core_attn_out [1, T, 32, 128], final_state [1, 32, 128, 128]
    - Dominant TTNN ops within each chunk if ported: `ttnn.matmul` (QK, AV inner-chunk attention), `ttnn.mul` (g scaling), `ttnn.sub`, `ttnn.add`
  - Step 4 — Gated RMSNorm and output projection: same as decode, scaled to [1, T, ...]
  - Prefill vs. decode distinction: prefill uses `chunk_gated_delta_rule` (parallelizable within chunks), saves `final_state` to `recurrent_states[layer_idx]` and `conv_state` to `conv_states[layer_idx]`; decode uses `recurrent_gated_delta_rule` with loaded state
  - TTNN implementation gaps: `chunk_gated_delta_rule`, `recurrent_gated_delta_rule`, `FusedRMSNormSwishGate`, and `causal_conv1d_fn/update` currently run on PyTorch CPU/GPU; porting these is the primary kernel development needed for full on-device execution

- **`gated_attention_ttnn_ops.md`** — TTNN operations for the Gated Attention forward pass (both prefill and decode).
  - Input projections (TTNN, col-sharded): Q+gate [1, T, n_q_heads × d_head × 2], KV [1, T, n_kv_heads × d_head × 2]
  - Gate sigmoid (TTNN): `ttnn.sigmoid(gate)` → gate_sigmoid [1, T, n_q_heads, d_head]
  - Q gating (TTNN): `ttnn.mul(Q, gate_sigmoid)` → Q_gated [1, T, n_q_heads, d_head]
  - Q normalization (TTNN): `ttnn.rms_norm(Q_gated, weight=tt_q_norm_weight)` per head
  - K normalization (TTNN): `ttnn.rms_norm(K, weight=tt_k_norm_weight)` per head
  - RoPE (TTNN): `ttnn.experimental.rotary_embedding` or equivalent; uses rotary_dim = 64
  - KV cache write (TTNN paged): `TTNNQwenPagedAttentionKVCache` writes K, V into paged cache slots
  - GQA KV repeat (TTNN): `ttnn.repeat_interleave(K, 8, dim=1)` to expand 2 KV heads to 16 Q heads
  - SDPA prefill (TTNN): `ttnn.transformer.scaled_dot_product_attention` with `SDPAProgramConfig`; tensor shapes Q [1, n_q_heads, T, d_head], K/V [1, n_q_heads, T, d_head]
  - SDPA decode (TTNN): `ttnn.transformer.scaled_dot_product_attention_decode` with `cur_pos`; Q [1, 1, n_q_heads, d_head]
  - Output reshape + projection (TTNN): reshape then `ttnn.linear` out_proj
  - All-gather post-projection (TTNN CCL): `ttnn.experimental.all_gather_async` on dim=-1 to restore replicated hidden state
  - Status: Gated Attention is substantially TTNN-accelerated (no PyTorch fallback for the core attention path); DeltaNet is partially TTNN (projections only)

- **`kernel_gap_summary.md`** — Consolidated audit of available vs. missing TTNN kernels for the full hybrid forward pass.
  - Table: operation | current implementation | TTNN status | gap description | recommended path
  - Rows: in_proj_qkv (TTNN linear, col-sharded — available), in_proj_z (TTNN — available), in_proj_a/b (PyTorch nn.Linear — gap: small output dims prevent sharding; could use replicated TTNN linear), causal_conv1d_fn (PyTorch / `causal-conv1d` C extension — gap: no TTNN conv1d for sequence dim), causal_conv1d_update (PyTorch — gap), chunk_gated_delta_rule (PyTorch / `flash-linear-attention` — gap: entire chunkwise kernel needs TTNN port or custom Metalium kernel), recurrent_gated_delta_rule (PyTorch — gap: the 6 core matrix ops are individually expressible in TTNN but are not fused), FusedRMSNormSwishGate (PyTorch — gap: composite of rms_norm + swish + mul; TTNN `rms_norm` exists but the swish-gated form is not fused), out_proj (TTNN — available), SDPA prefill and decode (TTNN — available for Gated Attention), all_gather_async (TTNN CCL — available)
  - Key finding: the recurrent decode step can be decomposed into 6 TTNN primitives (mul, matmul, sub, matmul for outer product, add, matmul for retrieval) but they are not fused; each dispatched separately will have launch overhead; a custom fused Metalium kernel for the recurrent step is the recommended path for decode latency
  - Key finding: `chunk_gated_delta_rule` for prefill is a complex multi-step chunkwise algorithm; porting it directly as a custom kernel or using TT-Metalium's tiled memory hierarchy with explicit chunk loop is the most viable approach

---

### Chapter 5 — Compute and Memory Roofline Analysis on Wormhole

**Description:** Applies the roofline model to the Gated Delta Net decode step and prefill pass on a single Wormhole chip, determines whether each is compute-bound or bandwidth-bound, and compares the state matrix read/write bottleneck to Wormhole's compute-to-bandwidth ratio.

**Directory:** `ch5_roofline_analysis/`

**Files and content:**

- **`index.md`** — Chapter overview and learning objectives.
  - States that this chapter answers Q6 from the guide spec
  - Notes that T3K head-parallel sharding (Chapter 6) changes per-device arithmetic intensity; this chapter covers single-device analysis first
  - Lists the two section files

- **`wormhole_hardware_specs.md`** — Wormhole hardware specifications relevant to the roofline model.
  - Per-chip specs (n300 half-chip, as used in T3K 1×8 mesh — each n300 card has 2 chips; T3K has 4×n300 = 8 chips total):
    - Tensix cores: 64 per chip (half of n300's 128)
    - L1 SRAM: 96 MB per chip (64 cores × 1.5 MB/core)
    - DRAM: 12 GB GDDR6 per chip (6 controllers × 2 GB)
    - DRAM bandwidth: 288 GB/s per chip (half of n300's 576 GB/s)
    - BF16 compute: 131 TFLOPS per chip (half of n300's 262 TFLOPS)
    - Ridge point: 131e12 / 288e9 ≈ 455 FLOP/byte
  - Inter-chip Ethernet: each pair of chips on the same n300 card connected by 200 Gbps die-to-die link; inter-card Ethernet via 200 Gbps QSFP-DD links; T3K mesh configured as 1×8 linear; aggregate CCL bandwidth ~25 GB/s effective for all-gather over 8 devices (empirical estimate)
  - Memory hierarchy summary table: register (Tensix FPU), L1 SRAM (1.5 MB/core, ~10 TB/s aggregate), DRAM (288 GB/s), Ethernet CCL (~25 GB/s)

- **`roofline_decode_and_prefill.md`** — Arithmetic intensity analysis and roofline conclusions for decode and prefill.
  - **Decode step (B=1, T=1, single head)**:
    - FLOPs: 6 operations dominate (from `gated_delta_net_decode_step.md`):
      - `g · S`: elementwise scalar-matrix = d_k × d_v = 128 × 128 = 16,384 FLOP
      - `S^T k̃`: matrix-vector = 2 × d_k × d_v = 32,768 FLOP
      - `β · (v − retrieval)`: vector ops = 2 × d_v ≈ 256 FLOP (negligible)
      - `k̃ ⊗ error`: outer product = d_k × d_v = 16,384 FLOP
      - `S_new = S_decayed + write`: matrix add = d_k × d_v = 16,384 FLOP
      - `o = S_new q̃`: matrix-vector = 2 × d_k × d_v = 32,768 FLOP
      - Total per head: ≈ 115,000 FLOP; × 32 heads = 3.7 MFLOP per layer
    - Bytes: state S must be read and written = 2 × d_k × d_v × 2 bytes = 2 × 128 × 128 × 2 = 65,536 bytes per head; × 32 heads = 2,097,152 bytes ≈ 2 MB per layer
    - Arithmetic intensity: 3.7e6 / 2.1e6 ≈ 1.8 FLOP/byte — far below the ridge point of 455 FLOP/byte → **decode is heavily memory-bandwidth-bound**
    - Time estimate at 288 GB/s: 2 MB / 288 GB/s ≈ 7 µs per layer; 30 DeltaNet layers → ~210 µs
    - Comparison to standard attention KV bandwidth at T = 256K: 2 × 2 × 256K × 256 × 2 = 512 MB per Gated Attention layer at 288 GB/s → 1.78 ms per attention layer; DeltaNet decode is 254× faster per layer than the full-attention decode at this sequence length
  - **Prefill pass (B=1, T=8192, single head, chunk C=64)**:
    - Dominant FLOPs: chunkwise QK and AV matmuls; within each chunk of size C=64: QK = 2 × C × C × d_k = 2 × 64 × 64 × 128 = 1,048,576 FLOP; AV = 2 × C × C × d_v = same; state-update matmuls per chunk ≈ 2 × C × d_k × d_v = 2,097,152 FLOP; total per head per chunk ≈ 4.2 MFLOP; × T/C = 128 chunks × 32 heads ≈ 17.2 GFLOP per layer
    - Bytes: K and V loaded once per chunk: 2 × C × d_k + 2 × C × d_v = 2 × 64 × 128 × 2 × 2 = 65,536 bytes per chunk × 128 chunks × 32 heads ≈ 268 MB; state read/write per chunk: 2 × d_k × d_v × 2 bytes = 65,536 bytes per chunk ≈ 8 MB total
    - Arithmetic intensity: 17.2e9 / (268e6 + 8e6) ≈ 62 FLOP/byte — below ridge point → **prefill is also memory-bandwidth-bound** (but less severely than decode; moderate compute utilization possible)
    - Note: for very large chunks (C → T), the inner chunk matmul becomes the dominant operation and intensity approaches that of dense attention; chunkwise approach with C=64 keeps intensity low
  - Conclusion: Gated Delta Net on Wormhole is bandwidth-bound in both decode and prefill regimes; the bottleneck is the state matrix read/write, not the matrix multiply FLOP count; kernel design should prioritize minimizing DRAM traffic (e.g., keeping the state in L1 if possible, streaming updates)
  - L1 feasibility for state: per-head state = 128 × 128 × 2 = 32 KB; 32 heads per layer = 1 MB; L1 per chip = 96 MB across 64 cores → if spread across cores, each core holds 1 MB / 64 = 16 KB of state, which is below the 1.5 MB L1 per core; the full per-layer state can fit in the chip's aggregate L1 if distributed across cores

---

### Chapter 6 — T3K Sharding Strategy for Gated Delta Net State

**Description:** Specifies how the Gated Delta Net state matrix, input projections, and output projections should be sharded across the 8 Wormhole devices of a T3K, analyzes CCL overhead for each parallelism option, and recommends a strategy that minimizes all-reduce cost while keeping per-device memory within budget.

**Directory:** `ch6_t3k_sharding/`

**Files and content:**

- **`index.md`** — Chapter overview and learning objectives.
  - States that this chapter answers Q7 from the guide spec
  - Clarifies that the T3K 1×8 mesh topology and per-chip specs were established in Chapter 5; here they are used to derive per-device memory and bandwidth figures
  - Lists the four section files and previews the recommendation: head-parallel sharding for the state matrix, with tensor-parallel column sharding for all input/output projections

- **`t3k_mesh_topology.md`** — T3K 1×8 mesh configuration and CCL bandwidth budget.
  - T3K = 4 × Wormhole n300 cards; each n300 = 2 Wormhole chips; 8 chips total in a 1×8 logical mesh
  - Inter-chip connectivity: pairs of chips on the same n300 card share a 200 Gbps die-to-die link; inter-card links are 200 Gbps QSFP-DD Ethernet; T3K mesh configured as a linear chain for CCL operations
  - CCL operations available: `ttnn.experimental.all_gather_async`, `ttnn.reduce_scatter`, `ttnn.all_reduce`
  - Effective all-gather bandwidth (ring topology, 8 devices, 200 Gbps links): each step transfers (N-1)/N of the tensor; for 8 devices, 7 steps at 200 Gbps → effective bandwidth per chip ≈ 25 GB/s (empirical; dominated by slowest link in the chain)
  - Memory budget per chip: 12 GB DRAM; model weights in BF16 for Qwen3.5-35B-A3B total ~70 GB → across 8 chips = 8.75 GB/chip; leaves ~3.25 GB/chip for activations, KV cache, and recurrent states

- **`head_parallel_state_sharding.md`** — Head-parallel sharding of the Gated Delta Net state matrix (recommended).
  - Strategy: shard num_v_heads across 8 devices; each device holds num_v_heads / 8 = 32 / 8 = 4 heads
  - Per-device state tensor: [B, 4, 128, 128] in BF16 = B × 4 × 128 × 128 × 2 = B × 131,072 bytes ≈ B × 128 KB per layer
  - Total across 30 DeltaNet layers: B × 128 KB × 30 ≈ B × 3.75 MB per device — negligible vs. 3.25 GB budget
  - Conv state per device: [B, (key_dim×2 + value_dim) / 8, 4] ≈ B × 8 KB per layer → trivial
  - Projection sharding: in_proj_qkv, in_proj_z use column sharding over output dim (each device computes its shard of Q, K, V, Z for its 4 heads); in_proj_a, in_proj_b are small (output dim = num_v_heads = 32 or 4) — can use replicated weights or all-gather output; out_proj uses row sharding (each device holds its 4-head shard of value_dim and accumulates via reduce_scatter or all-reduce)
  - Communication pattern per decode step:
    - Input hidden_states: replicated across all 8 devices (standard TP assumption for small batches)
    - in_proj_qkv: column-sharded matmul — no CCL needed
    - All-gather for in_proj_z output (to ensure z is available for gated RMSNorm): optional if z is also sharded per head
    - Recurrent delta rule step: local per-device (each device computes its 4 heads independently — no CCL)
    - After gated RMSNorm: each device has output [B, 1, 4 × 128] = [B, 1, 512] → value_dim / 8
    - out_proj (row-sharded): local matmul on [B, 1, 512] with weight [512, 2048/8] → [B, 1, 256]
    - All-gather: gather 8 × [B, 1, 256] → [B, 1, 2048] replicated output
    - Total CCL bytes per layer (decode B=1): all-gather of [B, 1, 2048] BF16 = 1 × 1 × 2048 × 2 = 4 KB — negligible
  - Advantage: no cross-device communication is needed during the state matrix operations (decode recurrence is per-head independent); CCL is limited to the final all-gather of the output projection

- **`alternative_sharding_strategies.md`** — Analysis of state-replication and model-parallel alternatives.
  - **Replicated state**: all 8 devices hold the full state [B, 32, 128, 128]; each device independently computes all 32 heads; no CCL on the state; in_proj/out_proj still use TP; replicated state = B × 1 MB per layer × 30 layers = B × 30 MB per device — acceptable for small B; disadvantage: state write must be synchronized or each device maintains its own replica (diverges if the recurrence is not replicated in lockstep)
  - **Sequence-parallel state**: each device holds a shard of the KV sequence (not applicable to DeltaNet — the state is not a sequence but a fixed matrix; sequence parallelism has no natural analog for recurrent state)
  - **Head-interleaved vs. head-blocked**: head-blocked sharding (device 0 gets heads 0-3, device 1 gets heads 4-7, ...) is preferred over interleaved for contiguous memory access; recommended for DRAM streaming patterns
  - Recommendation: head-parallel (blocked) sharding for state matrix and output dimension of all projections; replicated input hidden_states; single all-gather after out_proj; this matches the existing `TTNNLinearIReplicatedWColSharded` strategy already implemented in `qwen_attention.py`
  - CCL cost comparison table: head-parallel all-gather (4 KB/layer decode) vs. full hidden state all-gather (4 KB — same) vs. value-dim all-gather (8 KB/layer for num_v_heads=32 shard reconstruction) — all negligible; the bandwidth bottleneck is DRAM state read/write, not CCL

- **`kv_cache_sharding_for_gated_attention.md`** — Sharding the Gated Attention KV cache across 8 T3K devices.
  - Gated Attention has n_kv_heads = 2 for Qwen3.5-35B-A3B; with 8 devices, head-parallel sharding would require 0.25 KV heads per device — non-integer; two options:
    - Option A: replicate both KV heads on all 8 devices; each device computes attention for its n_q_heads / 8 = 2 Q heads against the full 2 KV heads replicated; KV cache size = [B, 2, T, 256] per device = B × T × 1 KB per layer
    - Option B: shard the 16 Q heads across devices (2 Q heads per device) and keep both KV heads replicated; standard GQA TP for small n_kv_heads
  - Recommendation: Option B (replicated KV, sharded Q) — standard approach; matches `TTNNQwen3FullAttention` structure; KV cache DRAM per device at T = 8K: B × 2 × 8192 × 256 × 2 = B × 8 MB per layer × 10 layers = B × 80 MB — fits within the 3.25 GB budget
  - Impact on T3K: KV cache per device for 10 attention layers at T = 256K: B × 2 × 256K × 256 × 2 = B × 256 MB per layer × 10 = B × 2.5 GB — this is the dominant memory consumer; limits batch size to ~1 at full context

---

### Chapter 7 — Existing Implementations, Kernel Gaps, and Development Roadmap

**Description:** Surveys existing TTNN and tt-transformers primitives against the full hybrid forward pass, consolidates the kernel gap analysis from Chapter 4, and provides a prioritized development roadmap for moving from the current partially-TTNN implementation toward a fully on-device forward pass.

**Directory:** `ch7_kernel_gaps_and_roadmap/`

**Files and content:**

- **`index.md`** — Chapter overview and learning objectives.
  - States that this chapter answers Q8 fully and synthesizes findings from all prior chapters
  - Distinguishes between gaps that require only configuration (already expressible in TTNN but not yet connected) and gaps that require new custom kernel development in TT-Metalium
  - Lists the three section files

- **`existing_ttnn_primitives_survey.md`** — Comprehensive survey of available TTNN and tt-transformers primitives relevant to Gated Delta Net and Gated Attention.
  - **Available, already used:**
    - `ttnn.matmul` / `ttnn.linear`: weight projection matmuls with program config support (core grid, transpose options); used for in_proj_qkv, in_proj_z, out_proj
    - `ttnn.mul` (`ttnn.multiply`): elementwise multiply; used for Q gating; needed for g · S scalar broadcast
    - `ttnn.sigmoid`: elementwise sigmoid; used for gate_sigmoid in Gated Attention; needed for β = σ(b)
    - `ttnn.rms_norm`: per-element RMS normalization with weight; used for Q/K norm in Gated Attention; needed for output norm in DeltaNet
    - `ttnn.reshape`, `ttnn.permute`, `ttnn.repeat_interleave`: tensor shape manipulation; used extensively in both attention types
    - `ttnn.experimental.all_gather_async`: CCL all-gather; used post-projection; topology=Linear
    - `ttnn.transformer.scaled_dot_product_attention`: full SDPA with FlashAttention-2; used for Gated Attention prefill
    - `ttnn.transformer.scaled_dot_product_attention_decode`: Flash-Decode for single-token; used for Gated Attention decode
    - `ttnn.transformer.paged_scaled_dot_product_attention_decode`: paged variant for Gated Attention with paged KV cache
  - **Available but not yet connected for DeltaNet:**
    - `ttnn.exp`: needed for g = exp(α); available
    - `ttnn.softplus`: needed for softplus(a + dt_bias); available
    - `ttnn.sub`: needed for error = v − retrieval; available
    - `ttnn.add`: needed for S_new = S_decayed + write; available
    - `ttnn.to_layout`, `ttnn.from_torch`, `ttnn.to_torch`: data movement; available
  - **Not available / not yet implemented for DeltaNet:**
    - Fused recurrent delta rule step: the 6-operation sequence (decay, retrieve, error, outer-product write, add, output query) is not a single TTNN op; must be composed from individual primitives or implemented as a custom Metalium kernel
    - Chunkwise delta rule (prefill): the WY-decomposition chunk algorithm has no TTNN equivalent; requires a custom kernel or a Python loop over chunks calling TTNN primitives
    - Causal conv1d (prefill/decode): sequential 1D convolution over the sequence dimension; `ttnn.conv` targets spatial 2D convolution; the 1D causal variant needs a dedicated implementation (e.g., a 1×K depthwise conv or a custom kernel)
    - FusedRMSNormSwishGate: composite of RMSNorm + SiLU/Swish gate; `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul` can compose it but the fused form is not a single op; fusion improves memory efficiency

- **`tt_transformers_review.md`** — Review of existing tt-transformers model code for relevant patterns.
  - tt-transformers provides model-level wrappers for Llama-family models; Gated Delta Net / GLA are not represented in the existing codebase (confirmed by code search: no `delta_net`, `gated_delta`, or `linear_attention` entries in tt-transformers modules)
  - The `TTNNQwen3LinearAttention` class in `tt_symbiote/modules/qwen_attention.py` is the most relevant existing implementation; it uses TTNN for input/output projections and falls back to `flash-linear-attention` PyTorch kernels for the DeltaNet core
  - The `TTNNQwen3FullAttention` class provides the Gated Attention implementation with TTNN SDPA and paged KV cache; this is complete and on-device
  - The `TTNNQwenPagedAttentionKVCache` class manages the hybrid cache: paged KV for full attention layers, `conv_states` and `recurrent_states` dicts for DeltaNet layers; this structure is the right abstraction but the DeltaNet states are stored as PyTorch tensors, not TTNN tensors on-device
  - Flash-linear-attention library (from `fla-org/flash-linear-attention`): provides Triton-optimized CUDA kernels for `chunk_gated_delta_rule` and `recurrent_gated_delta_rule`; these are the current CPU/GPU fallback; no Metalium equivalent exists

- **`development_roadmap.md`** — Prioritized roadmap for full on-device Gated Delta Net execution on T3K.
  - **Priority 1 (highest impact on decode latency):** Port `recurrent_gated_delta_rule` to a fused TT-Metalium kernel.
    - The 6 core operations (decay, state-vector multiply, error, outer product write, state update, output query) total ~3.7 MFLOP per layer and are bandwidth-limited by the 1 MB state read/write; a fused kernel that keeps the state in L1 across steps (streaming decode) would approach the DRAM bandwidth limit of 288 GB/s rather than paying multiple round-trips per op
    - Tensor shapes: state [4, 128, 128] per device (4 heads after sharding); can fit in a single Tensix core's 1.5 MB L1
    - Estimated kernel complexity: medium; similar to a fused matmul+update; no WY-decomposition needed
  - **Priority 2:** Port `causal_conv1d_update` (decode) to TTNN.
    - This is a sliding-window 1D convolution update with kernel size 4; for decode, it updates one slot of the conv state; it is a simple operation but currently runs on CPU/GPU; porting it as a custom Metalium kernel or as a TTNN elementwise + shift operation would remove the host round-trip
    - Tensor shapes: conv_state [1, 8192/8, 4] per device (sharded); new input [1, 8192/8, 1]
  - **Priority 3:** Port `gated_rms_norm_swish_gate` (FusedRMSNormSwishGate) to TTNN.
    - Compose from `ttnn.rms_norm`, `ttnn.silu`, `ttnn.mul`; verify numerics match the fused PyTorch version; fusion reduces memory traffic by avoiding intermediate tensor allocation
  - **Priority 4:** Port `in_proj_a`, `in_proj_b` (small gate projections) to TTNN with replicated weights.
    - These have output dim = num_v_heads (32 for 9B, 4 for 35B-A3B); use `ttnn.linear` with replicated weights (not col-sharded); minor contribution to latency but removes the last host-side linear projection
  - **Priority 5 (prefill throughput):** Port chunkwise delta rule (prefill) to a TT-Metalium kernel or Python chunk loop with TTNN primitives.
    - A Python loop over C=64 chunks calling `ttnn.matmul` for the QK and AV within-chunk products and `ttnn.matmul` for the cross-chunk state update is feasible as a first step; a fully fused single-kernel implementation is the long-term goal
    - This affects prefill throughput but not decode latency
  - **No new development needed:** SDPA for Gated Attention (both prefill and decode), all-gather CCL, input/output projections for both attention types, Q/K RMSNorm and Q-gating for Gated Attention

---

## 3. Conventions

### Notation

The following symbols are used consistently across all chapters:

| Symbol | Meaning |
|---|---|
| H | Model hidden dimension (2048 for Qwen3.5-35B-A3B) |
| T | Sequence length (number of tokens) |
| B | Batch size |
| d_k | Key/query head dimension for DeltaNet (128) |
| d_v | Value head dimension for DeltaNet (128) |
| d_h | Head dimension for Gated Attention (256) |
| n_q_h | Number of query heads in Gated Attention (16) |
| n_kv_h | Number of KV heads in Gated Attention (2) |
| H_v | Number of value heads in Gated Delta Net (32) |
| H_k | Number of key/query heads in Gated Delta Net (16) |
| S | Gated Delta Net recurrent state matrix, S ∈ R^{d_k × d_v} per head |
| g_t | Scalar decay gate at step t, g_t = exp(α_t) ∈ (0,1] |
| β_t | Update rate at step t, β_t = σ(b_t) ∈ (0,1) |
| k̃_t, q̃_t | L2-normalized key and query vectors |
| C | Chunk size for chunkwise parallel training (C = 64) |
| SDPA | Scaled dot product attention |
| GQA | Grouped query attention |
| TP | Tensor parallelism |
| CCL | Collective communication library |
| L1 | On-chip SRAM per Tensix core (1.5 MB) |
| DRAM | Off-chip GDDR6 memory (12 GB per chip) |

### Formatting Rules

- **All TTNN API names** are written in `monospace code` using their fully-qualified form (e.g., `ttnn.transformer.scaled_dot_product_attention`, `ttnn.experimental.all_gather_async`).
- **All tensor shapes** are written in bracket notation with named dimensions, e.g., `[B, H_v, d_k, d_v]`. Never use unnamed `[A, B, C, D]` without labeling axes in the preceding sentence.
- **All equations** use the notation table above; any newly introduced symbol is defined immediately before or after the equation in which it first appears.
- **Numeric examples** use the Qwen3.5-35B-A3B configuration (H=2048, d_k=d_v=128, H_v=32, H_k=16, n_q_h=16, n_kv_h=2, d_h=256, 40 layers, 30 DeltaNet + 10 Gated Attention) as the primary reference model; the Qwen3-Coder-Next / Qwen3-Next-80B-A3B configuration (H=2048, 48 layers, 36 DeltaNet + 12 Gated Attention) is noted where it differs.
- **FLOPs and bytes calculations** are always shown step-by-step with intermediate quantities labeled; no unexplained final numbers.
- **Hardware specs** always cite the chip variant and configuration (e.g., "Wormhole chip in T3K — half of n300, 64 Tensix cores, 288 GB/s DRAM bandwidth").
- **Implementation status tags**: use `[AVAILABLE]`, `[PARTIAL]`, `[GAP — requires custom kernel]`, or `[GAP — requires wiring]` to label each operation in audit tables.
- **Source code references** use file-relative paths from the tt-metal root, e.g., `models/experimental/tt_symbiote/modules/qwen_attention.py:TTNNQwen3LinearAttention`.

### Terminology

- "Gated Delta Net" and "Gated DeltaNet" are synonymous; this guide uses "Gated Delta Net" (two words, capitalized) as the primary term; "GDN" is acceptable in tables.
- "Gated Attention" refers specifically to the `Qwen3_5MoeGatedAttention` / `TTNNQwen3FullAttention` layer type — standard SDPA with Q-gating and Q/K RMSNorm — not to Gated Linear Attention (GLA).
- "GLA" always refers to Gated Linear Attention (Yang et al., ICML 2024), a distinct model from Gated Delta Net.
- "State matrix" refers to the recurrent hidden state S ∈ R^{d_k × d_v} of a single Gated Delta Net head.
- "KV cache" refers to the standard key-value cache used by the Gated Attention layers only; Gated Delta Net layers use a "recurrent state" and "conv state", not a KV cache.
- "T3K" refers to the Tenstorrent T3000 system with 4 × Wormhole n300 cards (8 Wormhole chips total) in a 1×8 logical mesh.
- "Per-chip" always means one Wormhole ASIC (one half of an n300 card), not one n300 card.
- "Prefill" = processing the full input prompt (T > 1) using `chunk_gated_delta_rule`; "decode" = generating one new token (T = 1) using `recurrent_gated_delta_rule`.
- "Decode step" refers to one token generation step across all layers, not one layer computation.
- "TTNN" refers to the TTNN Python operator library; "TT-Metalium" refers to the lower-level C++/RISC-V kernel programming model.

---

## 4. Cross-Chapter Dependencies

| Chapter | Depends on | Concepts carried forward |
|---|---|---|
| Ch 1 — Attention Variants and Linear Attention Foundations | (none) | General state-matrix formulation S_t = G_t ⊙ S_{t-1} + v_t k_t^T; terminology for RetNet/GLA/Mamba/DeltaNet; complexity baseline for standard attention |
| Ch 2 — Gated Delta Net Math and Recurrence | Ch 1 | S_t formulation from Ch 1 is extended with the decay gate g_t and delta rule correction; chunk parallelism analysis requires the sequential recurrence form derived in Ch 2; state matrix dimensions first defined here and used in Ch 5 and Ch 6 |
| Ch 3 — Gated Attention Mechanism | Ch 1 | Standard GQA SDPA from Ch 1 is the baseline; Gated Attention adds Q-gating and Q/K RMSNorm; tensor shapes defined here are used in Ch 4 and Ch 6 |
| Ch 4 — TTNN Primitive Mapping | Ch 2, Ch 3 | Tensor shapes from Ch 2 (DeltaNet) and Ch 3 (Gated Attention) drive the step-by-step primitive mapping; kernel gap audit references the full operation sequence from Ch 2; Gated Attention TTNN ops reference SDPA from Ch 1 |
| Ch 5 — Roofline Analysis | Ch 2, Ch 4 | FLOPs per decode step use the operation count from Ch 2's mathematical formulation; bytes loaded use the state dimensions from Ch 2; per-chip hardware specs needed before sharding analysis in Ch 6; roofline conclusion (bandwidth-bound) motivates the kernel fusion priorities in Ch 7 |
| Ch 6 — T3K Sharding Strategy | Ch 2, Ch 3, Ch 5 | State matrix dimensions from Ch 2 determine per-device memory after sharding; Gated Attention KV cache sizing from Ch 3 sets the memory budget constraint; T3K per-chip specs from Ch 5 set the bandwidth and DRAM budget; head-parallel strategy references the head counts from Ch 2 and Ch 3 |
| Ch 7 — Kernel Gaps and Roadmap | Ch 2, Ch 4, Ch 5, Ch 6 | Kernel gap audit extends the preliminary table from Ch 4; development priorities are ordered by decode latency impact using the roofline from Ch 5; tensor shapes in priority descriptions reference Ch 2 and Ch 6 sharded dimensions; all-gather analysis references the CCL bandwidth from Ch 6 |

**Specific forward references to flag during writing:**

- Ch 2 (`gated_delta_rule_formulation.md`) references Ch 5 for the per-chip DRAM bandwidth figure when discussing state-read bottleneck.
- Ch 2 (`state_vs_kv_cache_memory.md`) references Ch 6 for how the state is distributed across devices.
- Ch 3 (`gated_attention_formulation.md`) references Ch 4 for the TTNN implementation of Q-gating and Q/K RMSNorm.
- Ch 4 (`kernel_gap_summary.md`) references Ch 7 for the full development roadmap.
- Ch 5 (`roofline_decode_and_prefill.md`) references Ch 6 for the note that head-parallel sharding reduces per-device bytes by 8×.
- Ch 6 (`head_parallel_state_sharding.md`) references Ch 4 (`gated_delta_net_decode_step.md`) for the CCL communication pattern derivation.
- Ch 7 (`development_roadmap.md`) references Ch 5 for the L1 feasibility argument supporting the fused recurrent kernel priority.
