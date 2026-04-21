# Architecture and Hyperparameters

## Model Identity

Qwen3.6-35B-A3B uses the architecture class `Qwen3_5MoeForConditionalGeneration` with model type `qwen3_5_moe`. This is the same architecture class used by Qwen3.5-35B-A3B. The model is a hybrid linear-attention/full-attention decoder with Mixture-of-Experts feed-forward networks at every layer, paired with a Vision Transformer encoder for multimodal inputs.

---

## 1. Top-Level Hyperparameters

| Parameter | Config Key | Value |
|---|---|---|
| Architecture class | `architectures` | `Qwen3_5MoeForConditionalGeneration` |
| Model type | `model_type` | `qwen3_5_moe` |
| Number of decoder layers | `num_hidden_layers` | 40 |
| Hidden dimension | `hidden_size` | 2048 |
| Vocabulary size | `vocab_size` | 248,320 |
| Max position embeddings | `max_position_embeddings` | 262,144 (256K native context) |
| RMS norm epsilon | `rms_norm_eps` | 1e-6 |
| Tie word embeddings | `tie_word_embeddings` | true |
| Attention output gate | `attn_output_gate` | true |
| DeltaNet state precision | `mamba_ssm_dtype` | `float32` |
| Total parameters | -- | ~35B |
| Active parameters per token | -- | ~3B |

The 262K context window is the native training length. With appropriate RoPE scaling or NTK-aware interpolation, the model can be extended to approximately 1M tokens at inference time, though quality may degrade beyond the native length.

---

## 2. Hybrid Layer Layout

The 40 decoder layers follow a strict repeating pattern governed by the `layer_types` list in `config.json` and the `full_attention_interval=4` parameter:

```
Pattern: [linear_attention, linear_attention, linear_attention, full_attention] x 10
```

This 75%/25% split is the defining structural choice: 30 Gated DeltaNet layers use O(1)-per-token linear attention for efficiency, while 10 Gated Attention layers use O(T)-per-token softmax attention for precise long-range retrieval.

### Layer Layout Table (All 40 Layers)

| Layer | Type | Attention Mechanism | FFN Type |
|---|---|---|---|
| 0--2 | `linear_attention` | Gated DeltaNet | MoE |
| 3 | `full_attention` | Gated Attention | MoE |
| 4--6 | `linear_attention` | Gated DeltaNet | MoE |
| 7 | `full_attention` | Gated Attention | MoE |
| 8--10 | `linear_attention` | Gated DeltaNet | MoE |
| 11 | `full_attention` | Gated Attention | MoE |
| ... | ... | ... | ... |
| 36--38 | `linear_attention` | Gated DeltaNet | MoE |
| 39 | `full_attention` | Gated Attention | MoE |

Every layer, regardless of attention type, uses the same MoE FFN block.

---

## 3. Per-Layer Structure

Each of the 40 decoder layers has the same high-level structure:

```
Input x (residual stream)
  |
  +---> RMSNorm (attention_norm)
  |       |
  |       +---> [Gated DeltaNet OR Gated Attention]
  |       |
  +<---- residual add
  |
  +---> RMSNorm (ffn_norm)
  |       |
  |       +---> MoE FFN (router -> top-8 experts + shared expert)
  |       |
  +<---- residual add
  |
Output x (passed to next layer)
```

Both RMSNorm layers use the zero-centered formulation with `add_unit_offset=True`:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot (1 + w)$$

where $w$ is the learned per-dimension weight initialized to zero, ensuring the initial effective scale is 1.0.

---

## 4. Gated DeltaNet Configuration

The 30 Gated DeltaNet layers use a linear attention mechanism based on the delta rule, maintaining a fixed-size recurrent state matrix per head.

| Parameter | Config Key | Value |
|---|---|---|
| Key/Query heads | `linear_num_key_heads` | 16 |
| Value heads | `linear_num_value_heads` | 32 |
| Key head dimension | `linear_key_head_dim` | 128 |
| Value head dimension | `linear_value_head_dim` | 128 |
| Conv1d kernel size | `linear_conv_kernel_dim` | 4 |
| GQA ratio ($H_v / H_k$) | derived | 2 |
| Key dimension ($H_k \times d_k$) | derived | 2048 |
| Value dimension ($H_v \times d_v$) | derived | 4096 |
| Conv dimension ($2 \times \text{key dim} + \text{value dim}$) | derived | 8192 |

### Projection Inventory

Each Gated DeltaNet layer has the following weight matrices (typically fused into a single `in_proj_all` for efficiency):

| Projection | Input Shape | Output Shape | Purpose |
|---|---|---|---|
| `in_proj_qkv` | `[B, T, 2048]` | `[B, T, 8192]` | Q (2048) + K (2048) + V (4096), fed to conv1d |
| `in_proj_z` | `[B, T, 2048]` | `[B, T, 4096]` | Output gate for gated RMSNorm |
| `in_proj_a` | `[B, T, 2048]` | `[B, T, 32]` | Decay gate input (per value head) |
| `in_proj_b` | `[B, T, 2048]` | `[B, T, 32]` | Beta gate logit (per value head) |
| `out_proj` | `[B, T, 4096]` | `[B, T, 2048]` | Projects concatenated head outputs back to $H$ |

Additional per-layer parameters:
- `A_log` -- shape `[32]`, learned log-scale for exponential decay
- `dt_bias` -- shape `[32]`, learned bias for the decay gate time step
- `conv1d.weight` -- shape `[8192, 1, 4]`, depthwise causal convolution kernel
- `norm.weight` -- shape `[128]`, per-dimension RMSNorm weight for post-recurrence normalization

### Recurrent State Size

Each Gated DeltaNet layer maintains a state matrix $S \in \mathbb{R}^{d_k \times d_v} = \mathbb{R}^{128 \times 128}$ per value head. With 32 value heads per layer:

- Per layer: `[B, 32, 128, 128]` in float32 = $B \times 32 \times 128 \times 128 \times 4$ bytes = $B \times 2$ MB
- All 30 GDN layers at $B = 1$: 30 layers $\times$ 2 MB = 60 MB total recurrent state

The state is computed and stored in float32 (`mamba_ssm_dtype: "float32"`) because the accumulation of rank-1 outer products across many token steps requires higher precision than bfloat16 can provide.

Additionally, each GDN layer maintains a conv1d ring buffer of 4 slots, each of shape `[B, 8192]` in bfloat16 = $B \times 64$ KB per layer.

### GQA in DeltaNet

The 16 key/query heads are expanded to 32 via `repeat_interleave` with a GQA ratio of 2 before the recurrence, so each pair of value heads shares one key/query head. This reduces projection cost while keeping the state at full `[32, 128, 128]` resolution.

---

## 5. Gated Attention Configuration

The 10 Gated Attention layers use standard softmax attention with several Qwen3.5-specific extensions: per-head Q/K RMSNorm, partial rotary positional encoding, GQA with a high expansion ratio, and a learned output gate.

| Parameter | Config Key | Value |
|---|---|---|
| Query heads | `num_attention_heads` | 16 |
| KV heads | `num_key_value_heads` | 2 |
| Head dimension | `head_dim` | 256 |
| GQA ratio ($n_q / n_{kv}$) | derived | 8 |
| Partial rotary factor | `partial_rotary_factor` | 0.25 |
| Rotary dimension | derived ($0.25 \times 256$) | 64 |
| Output gate | `attn_output_gate` | true |

### Key Extensions

**Partial RoPE:** Only the first 64 of 256 head dimensions receive rotary encoding. See [Section 7](#7-rope-configuration) for the full RoPE configuration including frequency spectrum and M-RoPE details.

**Q/K Per-Head RMSNorm:** Both Q and K are normalized per head after projection, using the zero-centered formulation $(1 + w)$. Separate `q_norm.weight` and `k_norm.weight` tensors are loaded per attention layer.

**Output Gate:** The attention output is element-wise multiplied by $\sigma(x W_\text{gate})$ before the output projection, where $x$ is the original layer input and $W_\text{gate}$ is derived from the second half of the HF checkpoint's `q_proj.weight` (which has shape `[n_q \times d_h \times 2, H]`, with the first half for query and the second half for the gate, interleaved per head).

**GQA:** The high GQA ratio (see table above) drastically reduces KV cache memory (only 2 heads cached per layer) while maintaining 16-head query capacity.

### KV Cache Size

Each Gated Attention layer caches K and V tensors with 2 heads of dimension 256:

- Per layer, per token: $2 \times 2 \times 256 \times 2$ bytes (K + V, 2 KV heads, 256 dim, bfloat16) = 2,048 bytes
- All 10 Gated Attention layers at 262K context: $10 \times 262{,}144 \times 2{,}048$ bytes $\approx$ 5.1 GB

This is significantly smaller than what a 40-layer full-attention model would require, because only 10 of 40 layers need KV caches and each layer has only 2 KV heads.

---

## 6. MoE Configuration

Every one of the 40 decoder layers uses a Mixture-of-Experts feed-forward network in place of a standard dense MLP.

| Parameter | Config Key | Value |
|---|---|---|
| Total routed experts | `num_experts` | 256 |
| Active experts per token (top-k) | `num_experts_per_tok` | 8 |
| Expert intermediate dimension | `moe_intermediate_size` | 512 |
| Shared expert intermediate dimension | `shared_expert_intermediate_size` | 512 |
| Number of shared experts | -- | 1 (always active) |

### Expert Architecture

Each expert (routed and shared) is a SwiGLU FFN:

$$\text{hidden} = \text{SiLU}(x W_\text{gate}) \odot (x W_\text{up})$$
$$\text{output} = \text{hidden} \, W_\text{down}$$

where $W_\text{gate}, W_\text{up} \in \mathbb{R}^{H \times m}$ and $W_\text{down} \in \mathbb{R}^{m \times H}$, with $H = 2048$ and $m = 512$.

Parameters per expert:

$$P_\text{expert} = 2 \times H \times m + m \times H = 3 \times 2048 \times 512 = 3{,}145{,}728 \approx 3.1\text{M}$$

For the routed experts, the gate and up projections are fused into a single `[2048, 1024]` weight matrix (stored as `mlp.experts.gate_up_proj` with shape `[256, 1024, 2048]` for all 256 experts). The down projection is stored as `mlp.experts.down_proj` with shape `[256, 2048, 512]`.

### Shared Expert

The shared expert has the same SwiGLU architecture with `intermediate_size=512` but is always active for every token regardless of routing. Its output is additionally gated by a learned scalar:

$$g = \sigma(x \, W_\text{shared\_gate})$$

where $W_\text{shared\_gate} \in \mathbb{R}^{H \times 1}$. The final MoE output combines routed and shared contributions:

$$\text{MoE out} = g \cdot \text{SharedExpert}(x) + \sum_{i \in \text{top-}k} r_i \cdot \text{Expert}_i(x)$$

where $r_i$ are the softmax-normalized routing weights.

### Effective Active Computation

Per token, per MoE layer: 8 routed experts + 1 shared expert = 9 expert forward passes. Each expert forward pass costs $6Hm = 6 \times 2048 \times 512 \approx 6.3\text{M FLOPs}$ (using the $2 \times \text{MACs}$ convention). Total MoE FLOPs per token per layer: $9 \times 6.3\text{M} \approx 56.7\text{M FLOPs}$.

### Total Expert Parameter Count

Per layer: $256 \times 3.1\text{M} + 3.1\text{M (shared)} \approx 800\text{M}$ parameters.
Across 40 layers: $40 \times 800\text{M} \approx 32\text{B}$ expert parameters.

This accounts for the vast majority of the model's 35B total parameter count, with the remaining approximately 3B in attention projections, embeddings, norms, and the vision encoder.

---

## 7. RoPE Configuration

| Parameter | Config Key | Value |
|---|---|---|
| RoPE theta | `rope_theta` | 10,000,000 (10M) |
| M-RoPE interleaved | `mrope_interleaved` | true |
| M-RoPE sections | `mrope_section` | [11, 11, 10] |
| Partial rotary factor | `partial_rotary_factor` | 0.25 |

RoPE is applied only in the 10 Gated Attention layers. Gated DeltaNet layers use L2-normalized Q and K without any positional encoding; position information enters the DeltaNet layers implicitly through the sequential recurrence and the causal conv1d.

### Partial RoPE

With `head_dim=256` and `partial_rotary_factor=0.25`, only the first $0.25 \times 256 = 64$ dimensions of each head receive rotary encoding. The frequency spectrum uses:

$$\theta_i = \frac{1}{10{,}000{,}000^{2i/64}} \quad \text{for } i = 0, 1, \ldots, 31$$

The 32 frequency pairs (64 dimensions) span from $\theta_0 = 1.0$ down to $\theta_{31} \approx 1.58 \times 10^{-7}$, providing gradual frequency decay suitable for very long contexts.

### M-RoPE (Multimodal RoPE)

The 64 rotary dimensions (32 pairs) are divided into three sections for multimodal position encoding:

| Section | Name | Dimension Pairs | Rotary Dims |
|---|---|---|---|
| 0 | Temporal | 11 pairs | dims 0--21 |
| 1 | Height (spatial-y) | 11 pairs | dims 22--43 |
| 2 | Width (spatial-x) | 10 pairs | dims 44--63 |

The section assignment follows the contiguous layout shown in the table above (dims 0--21 for temporal, 22--43 for height, 44--63 for width). The `mrope_interleaved=true` flag does not change the section-to-dimension mapping; instead, it controls the layout of cos/sin rotary pairs within each section. When interleaved is true, the real and imaginary parts of each rotary pair are stored in alternating positions `[r0, i0, r1, i1, ...]` rather than the contiguous layout `[r0, r1, ..., i0, i1, ...]`. This affects the kernel implementation but not the mathematical semantics of which dimensions belong to which positional section. For text-only inference, all three sections receive the same sequential position ID, making M-RoPE mathematically equivalent to standard RoPE.

---

## 8. Vision Encoder Configuration

The vision encoder processes images and video frames into token embeddings that are interleaved with text tokens in the decoder input.

| Parameter | Config Key | Value |
|---|---|---|
| Architecture | -- | Vision Transformer (ViT) |
| Depth (layers) | `depth` | 27 |
| Hidden size | `hidden_size` | 1,152 |
| Intermediate size | `intermediate_size` | 4,304 |
| Number of attention heads | `num_heads` | 16 |
| Head dimension | derived ($1152 / 16$) | 72 |
| Patch size | `patch_size` | 16 |
| Spatial merge size | `spatial_merge_size` | 2 |
| Temporal patch size | `temporal_patch_size` | 2 |
| Input channels | `in_channels` | 3 |
| Output hidden size | `out_hidden_size` | 2,048 |
| Hidden activation | `hidden_act` | `gelu_pytorch_tanh` |
| Number of position embeddings | `num_position_embeddings` | 2,304 |

### Special Tokens

| Token | Token ID |
|---|---|
| Image token | 248,056 |
| Video token | 248,057 |
| Vision start | 248,053 |
| Vision end | 248,054 |

### Token Count Calculation

For an image of $H_\text{px} \times W_\text{px}$ pixels:

1. Divide into $16 \times 16$ patches: $\lceil H_\text{px}/16 \rceil \times \lceil W_\text{px}/16 \rceil$ patch tokens
2. Spatial 2x2 merge reduces count by 4x: $\lceil H_\text{px}/16 \rceil \times \lceil W_\text{px}/16 \rceil / 4$ vision tokens
3. The spatial 2x2 merge concatenates 4 adjacent patches into a single token of $4 \times 1{,}152 = 4{,}608$ dimensions, which is then linearly projected from 4,608 dimensions to 2,048 dimensions (`out_hidden_size`) before being interleaved with text tokens

For video, `temporal_patch_size=2` merges consecutive frame pairs, further halving the temporal token count.

The vision encoder is identical between Qwen3.5-35B-A3B and Qwen3.6-35B-A3B.

---

## 9. Multi-Token Prediction (MTP) Configuration

| Parameter | Config Key | Value |
|---|---|---|
| MTP hidden layers | `mtp_num_hidden_layers` | 1 |

The MTP module adds 1 additional transformer layer after the main decoder for predicting the next-next token (position $t+2$). This is primarily a training-time auxiliary objective that improves representation quality. At inference time, the MTP head can optionally serve as a lightweight draft model for speculative decoding. For standard inference without speculative decoding, the MTP head is ignored entirely.

---

## 10. Parameter Count Analysis

### Total Parameters

| Component | Approximate Parameter Count |
|---|---|
| Token embedding (tied with LM head) | $248{,}320 \times 2{,}048 \approx 0.51$B |
| Gated DeltaNet projections (30 layers) | 30 layers $\times$ ~25M/layer $\approx 0.75$B |
| Gated Attention projections (10 layers) | 10 layers $\times$ ~25M/layer $\approx 0.25$B |
| MoE expert weights (40 layers, 256+1 experts) | 40 $\times$ ~800M $\approx 32$B |
| Router weights (40 layers) | 40 $\times$ 2048 $\times$ 256 $\approx 0.02$B |
| Layer norms, gate weights, biases | $< 0.1$B |
| Vision encoder | $\approx 0.3$B |
| MTP layer | $\approx 0.1$B |
| **Total** | **$\approx 35$B** |

### Active Parameters Per Token

For a single text token, the active parameters are:

| Component | Active Parameters |
|---|---|
| Attention projections (all 40 layers) | $\approx 1.0$B |
| MoE: 8 routed + 1 shared expert per layer (40 layers) | $40 \times 9 \times 3.1\text{M} \approx 1.1$B |
| Router weights, norms, embedding lookup | $\approx 0.6$B |
| **Total active** | **$\approx 3$B** |

The "A3B" designation reflects this approximately 3B active parameter count per token. Only 9 of 257 experts (256 routed + 1 shared) are active per layer, meaning only $9/257 \approx 3.5\%$ of expert parameters are used per token.

Context length and KV cache sizing are covered in [Section 1](#1-top-level-hyperparameters) and [Section 5](#5-gated-attention-configuration).

---

**Next:** [`forward_pass_dataflow.md`](./forward_pass_dataflow.md)
