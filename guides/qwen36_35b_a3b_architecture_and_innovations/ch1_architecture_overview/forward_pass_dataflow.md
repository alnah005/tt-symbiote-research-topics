# Forward Pass Dataflow

This file traces the complete forward pass of Qwen3.6-35B-A3B for both text-only and multimodal inputs. All tensor shapes and operations reference the hyperparameters established in [`architecture_and_hyperparams.md`](./architecture_and_hyperparams.md).

---

## 1. Text-Only Forward Pass: End-to-End

The forward pass for a single text token through the decoder proceeds through five stages: embedding, 40 decoder layers (attention + MoE each), final normalization, and the language model head.

### Stage 1: Token Embedding

```
Input:  token_id (integer in [0, 248319])
Output: x of shape [B, 1, H] = [B, 1, 2048]
```

The embedding table has shape `[248320, 2048]`. In the TTNN implementation, this table is kept on the host CPU as float32. Per-token lookup is a single indexed read of a 2048-element vector, which is then transferred to the device as bfloat16.

### Stage 2: Decoder Layers (Layers 0--39)

Each of the 40 layers applies the same two-sublayer structure (attention variant + MoE FFN), but the attention mechanism alternates between Gated DeltaNet and Gated Attention according to the layer type.

#### Common Decoder Layer Structure

All 40 layers share the same two-sublayer skeleton. Sublayer 1 dispatches to `GatedDeltaNet(x_norm)` for linear attention layers or `GatedAttention(x_norm, position, rot_mats, kv_cache)` for full attention layers:

```
x_in  = x                                          [B, 1, 2048]

# Sublayer 1: Attention variant
x_norm = RMSNorm(x_in, attention_norm.weight)       [B, 1, 2048]
attn_out = AttentionVariant(x_norm, ...)             [B, 1, 2048]
x = x_in + attn_out                                 [B, 1, 2048]   # residual

# Sublayer 2: MoE FFN (identical across all layers)
x_norm = RMSNorm(x, ffn_norm.weight)                [B, 1, 2048]
moe_out = MoE(x_norm)                               [B, 1, 2048]
x = x + moe_out                                     [B, 1, 2048]   # residual
```

#### Gated DeltaNet Layers (Layers 0, 1, 2, 4, 5, 6, ...)

The `GatedDeltaNet(x_norm)` call internally performs:

1. **Fused input projection:** `x_norm` is projected to produce QKV (8192 dims), output gate Z (4096 dims), beta logits (32 dims), and decay logits (32 dims) -- typically via a single fused matmul.

2. **Conv1d local mixing:** The 8192-dim QKV slice passes through a causal depthwise conv1d with kernel size 4, implemented as a circular ring buffer during decode. The ring buffer stores the last 3 QKV vectors; the current QKV overwrites the oldest slot. A weighted sum of all 4 slots followed by SiLU activation produces the filtered QKV. This provides local context over 4 adjacent token positions.

3. **QKV split and reshape:** The conv output is split into Q `[B, 16, 128]`, K `[B, 16, 128]`, and V `[B, 32, 128]`.

4. **L2 normalization:** Q and K are L2-normalized per head to prevent state explosion:

$$\hat{q} = \frac{q}{\sqrt{\|q\|_2^2 + \epsilon}}, \quad \hat{k} = \frac{k}{\sqrt{\|k\|_2^2 + \epsilon}}$$

5. **GQA expansion:** Q and K are expanded from 16 heads to 32 heads via `repeat_interleave(2, dim=head)` to align with the 32 value heads.

6. **Gate computation:**

```math
g_t = -\exp(\texttt{A\_log}) \cdot \text{softplus}(a_t + \texttt{dt\_bias})
```

This produces a negative scalar per value head, so $\exp(g_t) \in (0, 1]$ serves as the decay factor.

7. **Delta rule recurrence (per value head):**

$$S_t \leftarrow \exp(g_t) \cdot S_{t-1}$$
$$m_t = S_t^\top k_t \quad \text{(retrieve current estimate)}$$
$$\delta_t = \sigma(b_t) \cdot (v_t - m_t) \quad \text{(delta correction)}$$
$$S_t \leftarrow S_t + k_t \otimes \delta_t \quad \text{(rank-1 write)}$$
$$o_t = S_t^\top q_t \quad \text{(read output)}$$

The state $S_t \in \mathbb{R}^{128 \times 128}$ per head is maintained in float32 across token steps.

8. **Gated RMSNorm + output gate:**

$$\text{output} = \text{RMSNorm}(o_t) \cdot \text{SiLU}(z_t)$$

where $z_t$ is the output gate from `in_proj_z`.

9. **Output projection:** `[B, 32, 128]` is reshaped to `[B, 4096]` and projected back to `[B, 2048]` via `out_proj`.

#### Gated Attention Layers (Layers 3, 7, 11, ...)

The `GatedAttention(x_norm, ...)` call internally performs:

1. **Save input copy:** A copy of `x_norm` (or the pre-norm `x_in`, depending on the hook design) is saved for the output gate computation. This must be an explicit copy because the base attention class deallocates its input after the QKV projection.

2. **QKV projections:**
   - Q projection: `[B, 1, 2048]` -> `[B, 1, 4096]` (16 heads x 256 dim)
   - K projection: `[B, 1, 2048]` -> `[B, 1, 512]` (2 heads x 256 dim)
   - V projection: `[B, 1, 2048]` -> `[B, 1, 512]` (2 heads x 256 dim)

   Note: The raw HF checkpoint `q_proj.weight` has shape `[n_q * d_h * 2, H]` with interleaved query and gate weights. During weight conversion, this is split into a separate `q_proj.weight` and `q_proj_gate.weight`.

3. **Per-head Q/K RMSNorm:** Both Q and K are normalized per head using zero-centered RMSNorm with separate learned weights (`q_norm.weight`, `k_norm.weight` of shape `[256]`).

4. **Partial RoPE:** Rotary positional encoding is applied to only the first 64 of 256 dimensions per head. The cos/sin matrices are precomputed with `rotary_dim=64` and `rope_theta=10,000,000`. The remaining 192 dimensions are left unchanged (cos=1, sin=0 in those positions).

5. **KV cache update:** The current K and V vectors are written into the paged KV cache for this layer.

6. **GQA expansion:** K and V are expanded from 2 heads to 16 heads via `repeat_interleave(8, dim=head)`.

7. **Scaled dot-product attention:**

```math
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
```

where $d_h = 256$ and the scale factor is $1/\sqrt{256} = 0.0625$. The attention is computed over all cached K/V positions, giving the full O(T) per-token cost.

8. **Output gate:** The attention output is element-wise multiplied by $\sigma(x_\text{saved} \, W_\text{gate})$, where $W_\text{gate}$ has shape `[2048, 4096]`. This gates every dimension of the attention output with a learned value in $(0, 1)$ before the output projection.

9. **Output projection:** `[B, 1, 4096]` -> `[B, 1, 2048]` via `W_O`.

### Stage 3: Final Normalization

```
x = RMSNorm(x, final_norm.weight)                   [B, 1, 2048]
```

After all 40 layers, a final RMSNorm is applied to the residual stream.

### Stage 4: Language Model Head

```
logits = x @ W_lm_head^T                            [B, 1, 248320]
next_token = argmax(logits[0, 0, :vocab_size])
```

The LM head projects from hidden dimension 2048 to vocabulary size 248,320. With `tie_word_embeddings=true`, $W_\text{lm\_head}$ shares parameters with the token embedding table.

### Complete Single-Token Decode Summary

```
token_id
  |
  v
Embedding lookup [248320, 2048]    ------>  x: [B, 1, 2048]
  |
  v
Layer 0 (GDN):   RMSNorm -> GatedDeltaNet -> +residual -> RMSNorm -> MoE -> +residual
Layer 1 (GDN):   RMSNorm -> GatedDeltaNet -> +residual -> RMSNorm -> MoE -> +residual
Layer 2 (GDN):   RMSNorm -> GatedDeltaNet -> +residual -> RMSNorm -> MoE -> +residual
Layer 3 (Attn):  RMSNorm -> GatedAttention -> +residual -> RMSNorm -> MoE -> +residual
Layer 4 (GDN):   ... (same pattern repeats)
  ...
Layer 39 (Attn): RMSNorm -> GatedAttention -> +residual -> RMSNorm -> MoE -> +residual
  |
  v
Final RMSNorm                                ------>  x: [B, 1, 2048]
  |
  v
LM Head [2048, 248320]                      ------>  logits: [B, 1, 248320]
  |
  v
argmax                                       ------>  next_token_id
```

---

## 2. Multimodal Forward Pass Extension

When the input contains images or video, the forward pass is extended with a vision encoder pipeline that runs before the decoder.

### Vision Encoding Pipeline

```
Image (H_px x W_px x 3)
  |
  v
1. Resize/pad to patch-aligned dimensions
  |
  v
2. Split into 16x16 patches           -------> [N_patches, 3, 16, 16]
   where N_patches = ceil(H_px/16) * ceil(W_px/16)
  |
  v
3. Linear patch projection             -------> [N_patches, 1152]
  |
  v
4. Add 2D position embeddings          -------> [N_patches, 1152]
  |
  v
5. 27-layer ViT encoder               -------> [N_patches, 1152]
   (self-attention + MLP per layer,
    gelu_pytorch_tanh activation,
    16 heads with head_dim=72)
  |
  v
6. Spatial 2x2 merge                  -------> [N_patches/4, 1152*4]
   (concatenate adjacent 2x2 patches,
    reducing token count by 4x)
  |
  v
7. Project to decoder dim             -------> [N_patches/4, 2048]
   (linear: 1152*4 -> 2048, or
    equivalent spatial merge + projection)
```

For video, `temporal_patch_size=2` merges consecutive frame pairs before the ViT, further halving the temporal token count.

### Token Interleaving

The vision tokens are interleaved with text tokens in the decoder input sequence:

```
[text_tokens...] [vision_start] [vision_tokens...] [vision_end] [text_tokens...]
```

The special token IDs mark the boundaries:
- `vision_start_token_id = 248053`
- `image_token_id = 248056` (placeholder in the text, replaced by vision tokens)
- `video_token_id = 248057`
- `vision_end_token_id = 248054`

Once interleaved, all tokens (text and vision) pass through the same 40-layer decoder. The decoder does not distinguish between text and vision tokens -- they are processed identically through both Gated DeltaNet and Gated Attention layers.

### M-RoPE for Vision Tokens

In the Gated Attention layers, the three M-RoPE sections encode different position dimensions for vision tokens:

- **Temporal section (11 pairs):** frame index (for video) or 0 (for single images)
- **Height section (11 pairs):** vertical patch position within the image grid
- **Width section (10 pairs):** horizontal patch position within the image grid

For text-only tokens, all three sections receive the same sequential position, making M-RoPE equivalent to standard RoPE.

---

## 3. State Management

The hybrid architecture requires two distinct state management regimes operating simultaneously. State sizes and per-layer memory derivations are detailed in [architecture_and_hyperparams.md, Sections 4--5](./architecture_and_hyperparams.md#4-gated-deltanet-configuration).

### State Comparison

| Property | Gated DeltaNet (30 layers) | Gated Attention (10 layers) |
|---|---|---|
| State type | Recurrent matrix $S$ | KV cache |
| Memory scaling | O(1) per layer, fixed at ~2 MB | O(T) per layer |
| Total at T=4K, B=1 | 63 MB (fixed) | 80 MB |
| Total at T=262K, B=1 | 63 MB (fixed) | 5.1 GB |
| Information retention | Lossy compression (decay + delta updates) | Exact (all K/V stored) |
| Long-range precision | Approximate (decayed memories) | Exact (any position retrievable) |

The hybrid design exploits this tradeoff: 75% of layers use constant-memory recurrence for efficiency, while 25% use exact attention for precise retrieval at long range. The 10 Gated Attention layers provide "anchor points" throughout the network where the model can perform exact lookups over the full context.

---

## 4. MoE Routing in the Forward Pass

The MoE block appears identically in all 40 layers, after the attention sublayer and its residual connection. The routing and computation follow this sequence:

### Step 1: Router Computation

```
router_logits = x_norm @ W_router^T                 [B, 1, 256]
```

$W_\text{router} \in \mathbb{R}^{256 \times 2048}$ projects the normalized hidden state to 256 expert logits. This is a device-side matmul.

### Step 2: Top-k Selection (Host-Side)

The 256-element logit vector is transferred to the host CPU (512 bytes in bfloat16). On the host:

```python
logits_cpu = ttnn.to_torch(router_logits).float()[0, 0, 0, :256]
topk_values, topk_indices = torch.topk(logits_cpu, k=8)
routing_weights = F.softmax(topk_values, dim=-1)
```

This sync to host is mandatory: the host must know which expert indices to dispatch. By running top-k and softmax on CPU, no custom device kernel is needed, and the data volume (512 bytes) makes the DMA transfer negligible.

### Step 3: Shared Expert Computation (Overlapped)

While the router logits are being transferred and processed on the host, the device queue has already begun computing the shared expert forward pass:

```
shared_gate_proj = SiLU(x_norm @ W_shared_gate_proj^T)   [B, 1, 512]
shared_up_proj   = x_norm @ W_shared_up_proj^T            [B, 1, 512]
shared_hidden    = shared_gate_proj * shared_up_proj       [B, 1, 512]
shared_down      = shared_hidden @ W_shared_down_proj^T    [B, 1, 2048]
shared_gate      = sigmoid(x_norm @ W_shared_gate^T)       [B, 1, 1]
shared_out       = shared_down * shared_gate                [B, 1, 2048]
```

This overlap hides the CPU top-k/softmax latency behind device shared-expert computation.

### Step 4: Routed Expert Computation (Sequential)

After top-k completes on the host, the 8 selected experts are computed sequentially:

```
accumulated = 0
for i in range(8):
    expert_idx = topk_indices[i]
    weight     = routing_weights[i]

    gate_up = x_norm @ W_experts_gate_up[expert_idx]^T    [B, 1, 1024]
    gate_proj, up_proj = split(gate_up, [512, 512])
    hidden = SiLU(gate_proj) * up_proj                     [B, 1, 512]
    expert_out = hidden @ W_experts_down[expert_idx]^T     [B, 1, 2048]

    accumulated += weight * expert_out
```

Each expert involves two matmuls (one for the fused gate+up projection, one for the down projection), a SiLU activation, an element-wise multiply, and a weighted addition.

### Step 5: Combine

```
moe_output = shared_out + accumulated                     [B, 1, 2048]
```

The shared expert output (gated by the learned scalar) is added to the weighted sum of the 8 routed expert outputs.

### Routing Characteristics

- **Expert count:** 256 routed + 1 shared = 257 total per layer, 10,280 total across 40 layers
- **Activation ratio:** 9/257 $\approx$ 3.5% of expert parameters active per token per layer
- **Router data volume per sync:** 256 logits $\times$ 2 bytes = 512 bytes (negligible)
- **Expert matmul sizes:** Each routed expert: `[1, 2048] x [2048, 1024]` (gate+up) and `[1, 512] x [512, 2048]` (down). These are small matmuls that may not fully saturate accelerator compute units, making DRAM bandwidth the likely bottleneck for loading expert weights.

---

## 5. Complete Forward Pass Timing Structure

The following diagram shows the logical ordering of operations for a single decode step, highlighting the three distinct execution domains (device queue, host CPU, and device-host sync points):

```
Device Queue:
  Layer i:
    [RMSNorm] -> [Attention variant] -> [residual add]
    -> [RMSNorm] -> [Router matmul] [Shared expert matmuls]
                          |                    |
                    Host sync (512 bytes)       | (device continues shared expert)
                          |                    |
                    [CPU top-k + softmax]      |
                          |                    v
                    [8x Expert matmul loop] [shared expert done]
                          |                    |
                    [Weighted sum + shared add]
                    -> [residual add]
  Layer i+1:
    ...
```

The host sync for router logits is the only mandatory CPU-device synchronization per layer. The shared expert computation overlaps with this sync, so the effective cost of the routing decision is partially hidden.

---

**Next:** [Chapter 2 -- Gated DeltaNet Deep Dive](../ch2_gated_deltanet/index.md)
