# Qwen3.6-35B-A3B Architecture and Innovations -- Plan

## Audience

This guide is for ML systems engineers and hardware-aware model developers who are working on or evaluating the Qwen3.6-35B-A3B model for deployment on Tenstorrent hardware. Readers are expected to be familiar with:

- Transformer architectures at the level of attention, MLP, RMSNorm, and RoPE
- The Qwen3.5-35B-A3B model at a high level (hybrid DeltaNet + Gated Attention layout, MoE routing, and the general TTNN implementation approach)
- Basic TTNN concepts (tensor operations, device placement, memory configs)
- Mixture of Experts routing and expert computation concepts

Readers do NOT need prior knowledge of Multi-Token Prediction (MTP), M-RoPE (multimodal rotary position embedding), Thinking Preservation, or the specific post-training techniques used in Qwen3.6. The guide builds that knowledge chapter by chapter.

The central question this guide answers: **Given that Qwen3.6-35B-A3B shares the same `Qwen3_5MoeForConditionalGeneration` architecture class as Qwen3.5-35B-A3B, what exactly changed, what stayed the same, and what are the implications for the existing TTNN implementation?**

---

## Chapter List

---

### Chapter 1 -- Complete Architecture Overview

**Description:** Establishes the full architecture of Qwen3.6-35B-A3B from first principles -- the hybrid layer layout, all key hyperparameters, the forward pass data flow, and how the Gated DeltaNet, Gated Attention, and MoE components compose into a single decoder block.

**Directory:** `ch1_architecture_overview/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation links to the two section files
  - Key takeaway: Qwen3.6-35B-A3B is architecturally identical to Qwen3.5-35B-A3B at the config/weight level; all innovations are post-training

- `architecture_and_hyperparams.md`
  - Full hyperparameter table extracted from config.json: 40 layers, hidden_size=2048, vocab_size=248320, max_position_embeddings=262144
  - The hybrid layer layout: 10 repetitions of (3 x Gated DeltaNet + 1 x Gated Attention), governed by `layer_types` list and `full_attention_interval=4`
  - Per-layer structure: each layer consists of (Attention Variant) followed by (MoE FFN), with pre-norm RMSNorm before each
  - Gated DeltaNet config: linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4
  - Gated Attention config: num_attention_heads=16, num_key_value_heads=2, head_dim=256, partial_rotary_factor=0.25
  - MoE config: num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512, shared_expert_intermediate_size=512 (1 shared expert always active)
  - Vision encoder summary: 27-layer ViT, hidden_size=1152, patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_heads=16
  - MTP config: mtp_num_hidden_layers=1
  - Rope config: mrope_interleaved=true, mrope_section=[11,11,10], rope_theta=10000000
  - Total parameters: 35B total, 3B activated per token
  - Context length: 262K native, extensible to ~1M

- `forward_pass_dataflow.md`
  - End-to-end forward pass for a single text token through the decoder:
    1. Token embedding lookup (vocab_size=248320)
    2. For layers 0,1,2 (Gated DeltaNet): RMSNorm -> GDN (conv1d + delta rule recurrence) -> residual -> RMSNorm -> MoE (router -> top-8 experts + shared expert) -> residual
    3. For layer 3 (Gated Attention): RMSNorm -> Gated Attention (Q-gating, Q/K RMSNorm, partial RoPE, SDPA, KV cache) -> residual -> RMSNorm -> MoE -> residual
    4. Repeat pattern for layers 4-39
    5. Final RMSNorm -> LM head
  - Multimodal forward pass extension: vision encoder processes image/video patches -> spatial merge -> projection into text embedding space -> interleaved with text tokens before decoder
  - State management: Gated DeltaNet layers maintain recurrent state S in R^{d_k x d_v} per head + conv state; Gated Attention layers maintain paged KV cache
  - How MoE routing works in the forward pass: router linear -> top-8 selection -> expert dispatch -> weighted accumulation + shared expert contribution

---

### Chapter 2 -- Gated DeltaNet Deep Dive

**Description:** Provides the complete mathematical formulation of the Gated DeltaNet mechanism as used in Qwen3.6, covering the delta rule state update, gating, conv1d local mixing, QK/V head asymmetries, and comparison to other linear attention variants.

**Directory:** `ch2_gated_deltanet/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Cross-references to the existing guide `guides/gated_delta_net_and_gated_attention_on_t3k/` for T3K-specific implementation details and TTNN primitive mapping
  - Cross-references to `guides/qwen35_implementation/` for the Blackhole P100A fused kernel implementation
  - This chapter focuses on the mechanism itself, not the hardware mapping

- `delta_rule_formulation.md`
  - Core recurrence for a single head at step t:
    ```
    g_t  = exp(alpha_t)                    alpha_t < 0, so g_t in (0,1]
    beta_t = sigma(b_t)                    beta_t in (0,1) (update rate)
    S_t  = g_t * S_{t-1} + k_tilde_t outer [beta_t * (v_t - g_t * S_{t-1}^T k_tilde_t)]
    o_t  = S_t q_tilde_t
    ```
  - Term-by-term interpretation: decay (g_t * S_{t-1}), retrieval (S_{t-1}^T k_tilde_t), delta correction (beta_t * error), rank-1 write (outer product), output query
  - Decay gate derivation: alpha_t = -exp(A_log) * softplus(a_t + dt_bias); A_log is learned per-head; a_t from in_proj_a
  - State matrix dimensions for Qwen3.6: S in R^{128 x 128} per head; [B, 32, 128, 128] per layer; ~1 MB per layer at B=1
  - L2 normalization of Q and K before recurrence (prevents state explosion)
  - Gated RMSNorm output: output_normed * silu(z), where z comes from in_proj_z

- `head_asymmetry_and_projections.md`
  - QK/V head asymmetry: 16 QK heads vs 32 V heads, all with dim=128
  - GQA expansion: K and Q are projected with 16 heads then repeat_interleave by 2 to match 32 V heads
  - Full projection inventory:
    - in_proj_qkv: [B, T, 2048] -> [B, T, 8192] (Q: 2048 + K: 2048 + V: 4096)
    - in_proj_z: [B, T, 2048] -> [B, T, 4096] (output gate)
    - in_proj_a: [B, T, 2048] -> [B, T, num_v_heads] (decay input)
    - in_proj_b: [B, T, 2048] -> [B, T, num_v_heads] (beta logit)
    - conv1d: causal 1D convolution with kernel_size=4 over concatenated QKV
    - out_proj: [B, T, 4096] -> [B, T, 2048]
  - Conv1d local mixing: applied to concatenated QKV before split; provides local context for 4 adjacent tokens; implemented as a shift register during decode

- `comparison_to_linear_attention_variants.md`
  - General linear attention state update form: S_t = G_t * S_{t-1} + v_t k_t^T
  - RetNet: G_t = gamma * I (scalar decay, position-independent, no data dependence)
  - GLA (Gated Linear Attention): G_t = 1 alpha_t^T (outer product gate, column-wise data-dependent decay, direct write)
  - Mamba2: G_t = gamma_t * 1 1^T (scalar per step, data-dependent but uniform across state, enables parallel scan via SSM structure)
  - DeltaNet (standard, no gating): no explicit forgetting; applies delta rule correction S_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T; targeted error correction but no coarse forgetting
  - Gated DeltaNet: combines GLA-style scalar decay with DeltaNet delta-rule correction; both forgetting and precise learning in a single update
  - Summary table: variant | gating type | data-dependent | forgetting mechanism | write mechanism | state size
  - Why Gated DeltaNet was chosen for Qwen3.5/3.6: best balance of long-range retrieval and selective forgetting; O(1) decode cost; state is fixed-size regardless of sequence length

---

### Chapter 3 -- Qwen3.5 vs Qwen3.6: Exact Differences

**Description:** Provides a precise, exhaustive comparison of Qwen3.6-35B-A3B against Qwen3.5-35B-A3B at the config, weight, and behavior levels, establishing definitively what changed and what remained identical.

**Directory:** `ch3_qwen35_vs_qwen36_differences/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Key finding preview: the architecture is identical; all differences are post-training and config metadata

- `config_diff.md`
  - Side-by-side config.json comparison (field by field):
    - Identical: architectures (Qwen3_5MoeForConditionalGeneration), model_type (qwen3_5_moe), 40 layers, hidden_size=2048, all layer_types, full_attention_interval=4, all DeltaNet head configs, all MoE configs, all attention configs, all vision encoder configs, rope_theta, max_position_embeddings, vocab_size
    - Added in 3.6: explicit bos_token_id field, output_router_logits field, pad_token_id field, partial_rotary_factor at top level (was only nested before)
    - Removed in 3.6: mlp_only_layers (was an empty list in 3.5, removed entirely)
    - Changed: only metadata fields (model name strings, version identifiers)
  - Conclusion: zero architectural differences; the HuggingFace model class is literally the same; weight tensor shapes are identical
  - Implication for TTNN: the existing Qwen3.5 TTNN implementation can load and run Qwen3.6 weights with zero code changes to the model architecture

- `post_training_differences.md`
  - What post-training means: same pre-trained base model, different RLHF/RL alignment, different data mixtures, different inference-time prompting strategies
  - Agentic coding improvements: RL training on coding agent tasks (SWE-bench environments, terminal interaction, repository-level code generation); improved tool use, multi-step planning, and error recovery
  - Thinking Preservation: a post-training and inference-time technique, not an architectural change; involves retaining reasoning chains from prior turns in the conversation context; mechanical implications for KV cache management (longer effective context, more cache usage) but no changes to the model forward pass
  - Data and training methodology: expanded agentic coding datasets, refined RL reward models for coding tasks, scaffolding improvements for multi-turn agent interactions
  - Weight-level differences: the safetensors files contain different weight values (different post-training) but identical shapes and dtypes; any weight loading code that works for Qwen3.5 works for Qwen3.6

- `benchmark_comparison.md`
  - Agentic coding benchmarks (Qwen3.5 -> Qwen3.6):
    - SWE-bench Verified: 70.0 -> 73.4 (+3.4)
    - Terminal-Bench: 40.5 -> 51.5 (+11.0)
    - SkillsBench: 4.4 -> 28.7 (+24.3)
    - NL2Repo: 20.5 -> 29.4 (+8.9)
    - QwenWebBench: 978 -> 1397 (+419)
  - General benchmarks: competitive or improved across reasoning, math, and coding tasks
  - Vision benchmarks: competitive with or better than Claude-Sonnet-4.5 and Gemma4-31B
  - Analysis: the largest gains are in agentic/multi-step coding tasks, consistent with the focus on post-training for agent capabilities
  - Implication: no hardware optimization changes needed; performance gains come from better weights

---

### Chapter 4 -- Partial Rotary Embedding and M-RoPE

**Description:** Covers the two rotary position embedding schemes used in Qwen3.6 -- partial RoPE for text-only Gated Attention layers and M-RoPE (multimodal rotary position embedding) for mixed-modality inference -- including the mathematical formulation, implementation details, and interaction between the two schemes.

**Directory:** `ch4_rope_and_mrope/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Clarifies that partial RoPE applies only to Gated Attention layers (every 4th layer); Gated DeltaNet layers use L2-normalized Q/K without any positional encoding
  - Lists the two section files

- `partial_rotary_embedding.md`
  - Qwen3.6 Gated Attention uses head_dim=256 with partial_rotary_factor=0.25, so rotary_dim = 0.25 * 256 = 64
  - Only the first 64 dimensions of each head receive RoPE; the remaining 192 dimensions are position-agnostic
  - Mathematical formulation: for a head vector h = [h_rot, h_pass] where h_rot in R^64 and h_pass in R^192:
    - h_rot is rotated using standard RoPE with theta=10,000,000 (10M)
    - h_pass is left unchanged
    - The result is concatenated: [RoPE(h_rot), h_pass]
  - Motivation for partial rotary (25% of dims):
    - Full RoPE on 256-dim heads would create very high-frequency rotation components at the upper dimensions, which can degrade attention quality at long contexts
    - Partial RoPE concentrates positional information in a dedicated subspace while allowing the remaining 75% of dimensions to learn position-independent features
    - The high rope_theta (10M) combined with only 64 rotary dims provides very gradual frequency decay, supporting the 262K context window
  - Frequency spectrum: with rotary_dim=64 and theta=10M, the frequencies range from theta^(-0/64) = 1.0 to theta^(-62/64) ≈ 1.58e-7, covering a wide range of position scales
  - Implementation note: standard RoPE implementations that assume full head_dim rotation need modification; cos/sin matrices must be computed using rotary_dim=64 (not head_dim=256); the existing Qwen3.5 TTNN implementation already handles this via corrected cos/sin matrices (documented in `guides/qwen35_implementation/`)

- `mrope_multimodal_positions.md`
  - M-RoPE (Multimodal RoPE) configuration: mrope_section=[11, 11, 10], mrope_interleaved=true
  - The 32 rotary dimension pairs (64 rotary dims / 2 = 32 pairs) are divided into three sections:
    - Section 0 (temporal): 11 pairs (dims 0-21) -- encodes temporal/sequential position
    - Section 1 (height/spatial-y): 11 pairs (dims 22-43) -- encodes vertical spatial position
    - Section 2 (width/spatial-x): 10 pairs (dims 44-63) -- encodes horizontal spatial position
  - For text-only tokens: all three sections receive the same sequential position ID; M-RoPE degenerates to standard RoPE
  - For vision tokens: each section receives a different position value:
    - Temporal section: frame index (for video) or 0 (for single images)
    - Height section: vertical patch position within the image grid
    - Width section: horizontal patch position within the image grid
  - For video tokens: temporal section encodes the frame number; spatial sections encode the 2D position within each frame
  - mrope_interleaved=true means the three sections' dimensions are interleaved rather than concatenated; dimension i belongs to section (i % 3); this improves numerical stability compared to block-contiguous assignment
  - Interaction with text-only RoPE: during pure text inference, M-RoPE with identical position IDs across all sections is mathematically equivalent to standard RoPE; no special handling needed for text-only TTNN deployment
  - Interaction with partial rotary: M-RoPE operates within the 64 rotary dimensions; the remaining 192 non-rotary dimensions are unaffected regardless of modality
  - Vision encoder position encoding: the vision encoder uses its own spatial position encoding (2D patch positions) independent of M-RoPE; M-RoPE is applied only in the text decoder to the cross-attended vision features after projection into the decoder embedding space
  - Implication for TTNN: text-only inference requires no M-RoPE-specific changes; multimodal inference would require constructing per-section position ID tensors and applying RoPE with section-aware frequencies

---

### Chapter 5 -- Multi-Token Prediction (MTP)

**Description:** Covers the Multi-Token Prediction training objective used in Qwen3.6, including the architectural addition (mtp_num_hidden_layers=1), how it relates to speculative decoding at inference time, and the accuracy/throughput tradeoffs.

**Directory:** `ch5_multi_token_prediction/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Clarifies that MTP is a training-time auxiliary objective with optional inference-time application for speculative decoding
  - Cross-references DeepSeek-V3's MTP implementation as a closely related approach

- `mtp_architecture_and_training.md`
  - MTP configuration: mtp_num_hidden_layers=1 in config.json
  - MTP concept: in addition to the standard next-token prediction head, the model includes an additional prediction head that predicts tokens 2 steps ahead
  - Architecture of the MTP module:
    - Takes the final hidden states from the main decoder
    - Passes through mtp_num_hidden_layers=1 additional transformer layer(s) that refine the representation for the next-next-token prediction
    - Applies a separate LM head (or shared LM head) to predict the token at position t+2
  - Training objective: the MTP loss is an auxiliary loss added to the standard next-token cross-entropy loss; it encourages the model to develop richer internal representations that capture multi-step dependencies
  - Comparison to DeepSeek-V3 MTP: DeepSeek-V3 also uses MTP with similar motivation; both use a small number of additional layers (1 in Qwen3.6) to predict future tokens; the training benefit comes from better representation learning, not from the MTP head itself being useful at inference
  - Weight overhead: 1 additional transformer layer adds a small parameter overhead relative to the 40-layer main decoder (~2.5% increase)

- `speculative_decoding_inference.md`
  - MTP at inference time: the additional prediction head can serve as a draft model for speculative decoding
  - Speculative decoding with MTP:
    1. Main model generates hidden states for token t
    2. MTP head predicts a draft token for position t+1 (using the extra layer)
    3. In the next forward pass, both the true token at t and the draft token at t+1 are processed together
    4. If the draft token matches the model's prediction, both tokens are accepted in a single step, doubling throughput
    5. If the draft does not match, only the first token is accepted and the draft is discarded
  - Acceptance rate: depends on the MTP head's accuracy; higher accuracy -> higher throughput gain; typical acceptance rates for well-trained MTP heads range from 50-80%
  - Throughput tradeoff: speculative decoding adds the overhead of running the MTP head (1 extra transformer layer per step) but can save full decoder passes when drafts are accepted; net benefit depends on acceptance rate and the relative cost of the MTP head vs a full forward pass
  - Accuracy tradeoff: MTP training can slightly improve the main model's quality (better representations) but the speculative decoding path introduces no accuracy change (rejected drafts are discarded, and accepted drafts are verified against the full model)
  - TTNN implication: supporting MTP-based speculative decoding requires:
    1. Loading the additional MTP layer weights
    2. Running the MTP layer forward pass after the main decoder
    3. Implementing the verify-and-accept logic (can be done on host)
    4. Handling variable-length token sequences per step (1 or 2 tokens accepted)
  - Without speculative decoding: the MTP head can be ignored entirely at inference time; the main decoder produces correct output without it; this is the simplest deployment path and requires zero changes to the existing TTNN implementation

---

### Chapter 6 -- Thinking Preservation

**Description:** Explains the Thinking Preservation feature introduced in Qwen3.6, how it works mechanically, its implications for inference, and whether it requires any architectural or implementation changes.

**Directory:** `ch6_thinking_preservation/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Key finding preview: Thinking Preservation is purely a prompting/inference-time technique, not an architectural change

- `thinking_preservation_mechanism.md`
  - What Thinking Preservation is: retaining the model's reasoning/thinking traces from previous turns in a multi-turn conversation, rather than discarding them
  - How it works mechanically:
    - In standard multi-turn inference, the model's chain-of-thought reasoning from prior turns may be truncated or omitted from the context to save tokens
    - With Thinking Preservation, the reasoning context from historical messages is explicitly retained in the conversation history
    - This allows the model to reference its prior reasoning when answering follow-up questions, improving consistency and depth
  - Implementation: purely a prompting/context management technique:
    - The conversation template includes reasoning traces from prior turns
    - No changes to the model architecture, forward pass, or weight structure
    - The model processes the preserved thinking tokens through the same decoder as any other text tokens
  - KV cache implications:
    - Retaining reasoning traces increases the effective context length per conversation turn
    - For Gated DeltaNet layers: no impact, since the recurrent state is fixed-size regardless of sequence length
    - For Gated Attention layers: the KV cache grows proportionally to the total token count including preserved reasoning; this is the primary resource cost
    - At long conversations with preserved thinking, the KV cache for the 10 Gated Attention layers can become the memory bottleneck
  - Interaction with the 262K context window: with thinking preservation, the context fills faster (reasoning traces can be verbose); effective conversation turns before context exhaustion is reduced; context management strategies (summarization, selective preservation) may be needed for very long conversations
  - TTNN implementation implication: zero model code changes required; Thinking Preservation is handled entirely at the application/serving layer (conversation template construction, context window management); the TTNN decoder processes the tokens identically regardless of whether they represent preserved reasoning or new user input

---

### Chapter 7 -- MoE Architecture and Cross-Model Comparison

**Description:** Deep dive into the MoE configuration used in Qwen3.6 and comparison with other recent MoE models (DeepSeek-V3, Gemma4-26B-A4B), focusing on the implications of many-small-experts vs fewer-large-experts for hardware utilization.

**Directory:** `ch7_moe_comparison/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Cross-references to the existing guides on MoE optimization: `guides/moe_optimization_techniques_for_ttnn/`, `guides/expert_parallelism_strategies/`, `guides/ttnn_moe_performance_optimization_on_t3k/`

- `qwen36_moe_architecture.md`
  - Qwen3.6 MoE configuration: 256 routed experts + 1 shared expert, top-8 routing, moe_intermediate_size=512
  - Expert architecture: each expert is a SwiGLU FFN with input=2048, intermediate=512, output=2048
  - Shared expert: same architecture (intermediate=512), always active, gated by a learned scalar
  - Router: linear projection [2048 -> 256], softmax -> top-8 selection -> weight normalization
  - Effective computation per token: 8 routed experts + 1 shared = 9 expert forward passes; each expert has ~2M parameters (2 * 2048 * 512 + 512 * 2048); total MoE FLOPs per token = 9 * 2 * (2 * 2048 * 512 + 512 * 2048) ≈ 56M FLOPs
  - Per-layer total expert parameters: 256 * (2 * 2048 * 512 + 512 * 2048) = ~805M; across 40 layers: ~32.2B total expert parameters
  - Active parameters: per token, only 9/256 = 3.5% of expert parameters are used, yielding the 3B active parameter count

- `cross_model_moe_comparison.md`
  - DeepSeek-V3 comparison:
    - 256 routed experts + 1 shared, top-8 routing (same count as Qwen3.6)
    - Expert intermediate size: 2048 (4x larger than Qwen3.6's 512)
    - Hidden size: 7168 (3.5x larger than Qwen3.6's 2048)
    - Total parameters: 685B vs Qwen3.6's 35B
    - Key difference: DeepSeek-V3 uses the same many-small-expert approach but at much larger scale
    - Both use auxiliary-loss-free load balancing
  - Gemma4-26B-A4B comparison:
    - Fewer, larger experts: exact config from Gemma4 (num_experts, top_k_experts, moe_intermediate_size)
    - Different routing strategy and expert size tradeoff
  - Many-small-experts vs fewer-large-experts analysis:
    - Advantages of many small experts (Qwen3.6 approach with 256 x 512):
      - Finer-grained specialization: each expert can specialize in a narrower domain
      - More routing flexibility: 256-choose-8 provides more diverse expert combinations
      - Lower per-expert memory: each expert is small (~2M params), enabling efficient batched computation
    - Disadvantages of many small experts:
      - Higher routing overhead: 256-way softmax and top-8 selection is more expensive than fewer-way
      - More expert weights to store: 256 * 40 layers = 10,240 expert weight tensors in DRAM
      - Poorer compute utilization on accelerators: small matmuls (2048 x 512) may not saturate compute units
      - Higher all-to-all communication volume in expert-parallel deployments
    - Hardware utilization implications for Tenstorrent:
      - Each expert's matmul (2048 x 512 intermediate) is relatively small; may need batching across experts for compute efficiency
      - DRAM bandwidth is the bottleneck for loading 8 expert weight sets per token per layer
      - Expert parallelism across 8 T3K devices: 256/8 = 32 experts per device; reasonable balance
      - bfp4 quantization of expert weights is critical for fitting all 10,240 weight tensors in aggregate DRAM

---

### Chapter 8 -- Vision Encoder and Multimodal Integration

**Description:** Covers the vision encoder specifications, how vision tokens are processed and integrated with text tokens, and comparison with the Qwen3.5 vision encoder and other recent multimodal models.

**Directory:** `ch8_vision_encoder/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Notes that the vision encoder is identical between Qwen3.5 and Qwen3.6 (same config, same architecture)

- `vision_encoder_specs.md`
  - Architecture: 27-layer Vision Transformer (ViT)
  - Hidden size: 1152
  - Patch size: 16 (each 16x16 pixel patch becomes one token)
  - Spatial merge size: 2 (2x2 spatial pooling reduces vision token count by 4x after encoding)
  - Temporal patch size: 2 (for video, 2 consecutive frames are merged into one temporal token)
  - Number of attention heads: 16 (head_dim = 1152/16 = 72)
  - Image processing pipeline: image -> resize/pad -> split into 16x16 patches -> linear projection -> add position embeddings -> 27 ViT layers -> spatial 2x2 merge -> project to decoder hidden size (2048)
  - Video processing: frames sampled -> each frame processed as image -> temporal_patch_size=2 merges consecutive frame tokens -> projected to decoder embedding space
  - Token count calculation: for an image of H x W pixels: ceil(H/16) * ceil(W/16) / 4 (after spatial merge) vision tokens injected into the text sequence

- `vision_encoder_comparison.md`
  - Qwen3.5 vs Qwen3.6 vision encoder: identical (same depth=27, same hidden_size=1152, same patch_size=16, same spatial_merge_size=2, same temporal_patch_size=2)
  - Comparison with Gemma4 vision encoder:
    - Gemma4: 16 layers, hidden_size=768, patch_size=16, pooling_kernel_size=3
    - Qwen3.6: 27 layers (69% deeper), hidden_size=1152 (50% wider), spatial_merge_size=2 (vs Gemma4's 3x3 pooling)
    - Qwen3.6's vision encoder is significantly larger and deeper
  - Comparison with LLaVA-style vision encoders: LLaVA typically uses a pre-trained CLIP ViT (e.g., ViT-L/14 with 24 layers, hidden_size=1024); Qwen3.6's encoder is custom and slightly larger
  - TTNN deployment considerations:
    - The 27-layer ViT processes only during prefill (when images/videos are first ingested), not during decode
    - At prefill time, vision encoding is a one-time cost amortized over the entire conversation
    - For text-only deployment, the vision encoder can be entirely omitted, saving ~300M parameters of DRAM
    - The spatial merge and projection layers are simple operations (average pooling + linear) that map straightforwardly to TTNN ops

---

## Conventions

### Terminology

- "Qwen3.6" always refers to Qwen3.6-35B-A3B unless explicitly stated otherwise
- "Qwen3.5" always refers to Qwen3.5-35B-A3B unless explicitly stated otherwise
- "Gated DeltaNet" and "GDN" are interchangeable; "Gated DeltaNet" is preferred in running text, "GDN" in tables
- "Gated Attention" refers to the full-attention layer type with Q-gating, Q/K RMSNorm, and partial RoPE (not Gated Linear Attention / GLA)
- "MoE" = Mixture of Experts; "MTP" = Multi-Token Prediction; "M-RoPE" = Multimodal Rotary Position Embedding
- "Post-training" refers to RLHF, RL, and alignment training performed after the base model pre-training
- "Thinking Preservation" refers to the inference-time technique of retaining reasoning context across conversation turns

### Notation

| Symbol | Meaning |
|---|---|
| H | Model hidden dimension (2048) |
| T | Sequence length (number of tokens) |
| B | Batch size |
| d_k | DeltaNet key/query head dimension (128) |
| d_v | DeltaNet value head dimension (128) |
| d_h | Gated Attention head dimension (256) |
| n_q | Gated Attention query heads (16) |
| n_kv | Gated Attention KV heads (2) |
| H_v | DeltaNet value heads (32) |
| H_k | DeltaNet key/query heads (16) |
| S | DeltaNet recurrent state matrix, S in R^{d_k x d_v} per head |
| g_t | Scalar decay gate at step t, g_t = exp(alpha_t) in (0,1] |
| beta_t | Update rate at step t, beta_t = sigma(b_t) in (0,1) |

### Formatting Rules

- **Tensor shapes**: written as `[dim1, dim2, ...]` with named dimensions, e.g., `[B, H_v, d_k, d_v]`
- **Config references**: use `config_field_name` format with the value on first mention, e.g., `partial_rotary_factor=0.25`
- **TTNN API names**: fully-qualified in monospace, e.g., `ttnn.transformer.scaled_dot_product_attention`
- **Cross-references to existing guides**: use relative path from the guides root, e.g., `guides/qwen35_implementation/`
- **Benchmark numbers**: always show both Qwen3.5 and Qwen3.6 values with the delta, e.g., "70.0 -> 73.4 (+3.4)"
- **Code snippets**: include only for non-obvious computations (e.g., M-RoPE section assignment, MTP verification logic); do not reproduce entire forward methods

### TTNN Porting Notes Convention

Each chapter that discusses model components should end with a section titled "TTNN Deployment Implications" summarizing whether any TTNN code changes are needed for Qwen3.6 support. The default finding for most chapters is "no changes needed" given the architectural identity with Qwen3.5.

---

## Cross-Chapter Dependencies

| Chapter | Depends on | Concepts carried forward |
|---|---|---|
| Ch 1 -- Architecture Overview | (none) | Full hyperparameter table, layer layout pattern, forward pass data flow; referenced by all subsequent chapters |
| Ch 2 -- Gated DeltaNet Deep Dive | Ch 1 (layer layout, DeltaNet hyperparameters) | Delta rule formulation, state matrix dimensions, projection inventory; referenced by Ch 3 (comparison), Ch 7 (MoE interaction) |
| Ch 3 -- Qwen3.5 vs Qwen3.6 Differences | Ch 1 (architecture overview), Ch 2 (DeltaNet details) | Definitive finding that architecture is identical; benchmark deltas; referenced by all subsequent chapters as the foundational comparison |
| Ch 4 -- Partial RoPE and M-RoPE | Ch 1 (attention hyperparameters, rotary config) | Partial rotary formulation, M-RoPE section assignment, text-only equivalence; referenced by Ch 8 (vision token positions) |
| Ch 5 -- Multi-Token Prediction | Ch 1 (decoder architecture) | MTP layer architecture, speculative decoding mechanics; standalone topic with minimal dependencies |
| Ch 6 -- Thinking Preservation | Ch 1 (context length, layer types) | KV cache growth implications, context management considerations; references Ch 2 for DeltaNet state independence from sequence length |
| Ch 7 -- MoE Comparison | Ch 1 (MoE config), Ch 2 (DeltaNet interaction with MoE in forward pass) | Cross-model comparison, hardware utilization analysis; references existing MoE optimization guides |
| Ch 8 -- Vision Encoder | Ch 1 (vision encoder config), Ch 4 (M-RoPE for multimodal positions) | Vision token pipeline, spatial merge mechanics, comparison table; references Ch 3 for identity with Qwen3.5 vision encoder |

**Key concept flows spanning multiple chapters:**

- The architectural identity finding (Ch 3) is the thesis that motivates the entire guide: all chapters build on the fact that Qwen3.6 is a post-training evolution, not an architectural change
- The partial RoPE formulation (Ch 4) applies only to Gated Attention layers (Ch 1 layout), and the M-RoPE extension is relevant only for multimodal inference (Ch 8)
- The MoE architecture (Ch 7) appears in every decoder layer after both DeltaNet (Ch 2) and Gated Attention layers, as established in the forward pass data flow (Ch 1)
- The KV cache implications of Thinking Preservation (Ch 6) are bounded by the fixed-size DeltaNet state (Ch 2) for 30/40 layers but grow linearly for the 10 Gated Attention layers (Ch 1)
- MTP (Ch 5) operates after the full decoder forward pass (Ch 1) and is orthogonal to the attention mechanism choice (Ch 2) and MoE routing (Ch 7)
