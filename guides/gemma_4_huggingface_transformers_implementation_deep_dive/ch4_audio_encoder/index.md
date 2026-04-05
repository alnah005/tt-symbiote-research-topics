# Chapter 4: Audio Encoder

This chapter covers the conformer-based audio encoder in Gemma 4, from mel spectrogram input through the final output projection. The audio encoder converts raw audio features into soft tokens for injection into the text decoder. All audio classes live in [`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py). Refer to [Chapter 2](../ch2_configuration_hierarchy/index.md) for `Gemma4AudioConfig` parameter defaults.

## Module Tree

```
Gemma4AudioModel (Gemma4PreTrainedModel)
  |
  +-- subsample_conv_projection: Gemma4AudioSubSampleConvProjection
  |     +-- layer0: Gemma4AudioSubSampleConvProjectionLayer
  |     |     +-- conv: nn.Conv2d(1, 128, kernel=3, stride=2, padding=1, bias=False)
  |     |     +-- norm: nn.LayerNorm(128, eps=1e-6, bias=False)
  |     |     +-- act: nn.ReLU()
  |     +-- layer1: Gemma4AudioSubSampleConvProjectionLayer
  |     |     +-- conv: nn.Conv2d(128, 32, kernel=3, stride=2, padding=1, bias=False)
  |     |     +-- norm: nn.LayerNorm(32, eps=1e-6, bias=False)
  |     |     +-- act: nn.ReLU()
  |     +-- input_proj_linear: nn.Linear(1024, 1024, bias=False)
  |
  +-- rel_pos_enc: Gemma4AudioRelPositionalEncoding
  |     +-- inv_timescales: Buffer [1, 1, 512]
  |
  +-- layers: ModuleList of Gemma4AudioLayer x 12
  |     +-- feed_forward1: Gemma4AudioFeedForward
  |     |     +-- pre_layer_norm: Gemma4RMSNorm(1024)
  |     |     +-- ffw_layer_1: Gemma4ClippableLinear(1024, 4096)
  |     |     +-- ffw_layer_2: Gemma4ClippableLinear(4096, 1024)
  |     |     +-- post_layer_norm: Gemma4RMSNorm(1024)
  |     +-- self_attn: Gemma4AudioAttention
  |     |     +-- q_proj: Gemma4ClippableLinear(1024, 1024)
  |     |     +-- k_proj: Gemma4ClippableLinear(1024, 1024)
  |     |     +-- v_proj: Gemma4ClippableLinear(1024, 1024)
  |     |     +-- post: Gemma4ClippableLinear(1024, 1024)
  |     |     +-- relative_k_proj: nn.Linear(1024, 1024, bias=False)
  |     |     +-- per_dim_scale: Parameter [128]
  |     |     +-- softcap: Buffer (scalar, 50.0)
  |     +-- norm_pre_attn: Gemma4RMSNorm(1024)
  |     +-- norm_post_attn: Gemma4RMSNorm(1024)
  |     +-- lconv1d: Gemma4AudioLightConv1d
  |     |     +-- pre_layer_norm: Gemma4RMSNorm(1024)
  |     |     +-- linear_start: Gemma4ClippableLinear(1024, 2048)
  |     |     +-- depthwise_conv1d: Gemma4AudioCausalConv1d(1024, 1024, kernel=5, groups=1024, bias=False)
  |     |     +-- conv_norm: Gemma4RMSNorm(1024)
  |     |     +-- linear_end: Gemma4ClippableLinear(1024, 1024)
  |     +-- feed_forward2: Gemma4AudioFeedForward (same structure as feed_forward1)
  |     +-- norm_out: Gemma4RMSNorm(1024)
  |
  +-- output_proj: nn.Linear(1024, 1536, bias=True)
```

---

## 4.1 Gemma4AudioSubSampleConvProjectionLayer

A single stage of subsampling: Conv2d followed by LayerNorm and ReLU.

```python
class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_eps):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(2,2), padding=1, bias=False)
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps, elementwise_affine=True, bias=False)
        self.act = nn.ReLU()
```

**Forward pass:**

1. If a mask is provided, zero out masked positions: `hidden_states = hidden_states * mask[:, None, :, None]`
2. Apply `Conv2d` (cast to weight dtype first)
3. Permute to `[B, T, F, C]`, apply `LayerNorm` over channels, permute back to `[B, C, T, F]`
4. Apply `ReLU`
5. Downsample mask by stride 2: `mask = mask[:, ::2]`

Each layer halves the time dimension due to `stride=2`.

---

## 4.2 Gemma4AudioSubSampleConvProjection

The two-stage subsampling front-end that converts mel spectrograms into the encoder's hidden dimension.

```python
class Gemma4AudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(1, 128, config.rms_norm_eps)
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(128, 32, config.rms_norm_eps)
        proj_input_dim = (128 // 4) * 32  # = 1024
        self.input_proj_linear = nn.Linear(proj_input_dim, config.hidden_size, bias=False)
```

The channel counts come from `Gemma4AudioConfig.subsampling_conv_channels` which defaults to `[128, 32]`.

**Forward pass:**

1. Unsqueeze input to add a channel dim: `[B, T, F]` -> `[B, 1, T, F]`
2. `layer0`: Conv2d(1->128), halves T and F -> `[B, 128, T/2, F/2]`
3. `layer1`: Conv2d(128->32), halves again -> `[B, 32, T/4, F/4]`
4. Permute and reshape to `[B, T/4, (F/4)*32]` -- flattens the frequency and channel dims
5. `input_proj_linear`: project from flattened dim to `Gemma4AudioConfig.hidden_size` (1024)

**Projection dimension derivation:** The formula `proj_input_dim = (subsampling_conv_channels[0] // 4) * subsampling_conv_channels[1]` computes `(128 // 4) * 32 = 1024`. Here the `128` comes from `Gemma4AudioConfig.subsampling_conv_channels[0]` (the first conv's output channel count), **not** from the mel frequency bin count `F`. They happen to share the same value (128) by coincidence, but conceptually `subsampling_conv_channels[0]` is the channel dimension after the first conv layer, and dividing by 4 accounts for the two stride-2 convolutions halving the frequency axis twice (`F/4`). The result equals `hidden_size`, making `input_proj_linear` a 1024->1024 square projection.

**Output:** `(hidden_states: [B, T/4, 1024], mask: [B, T/4])`

---

## 4.3 Gemma4AudioRelPositionalEncoding

Sinusoidal relative position encoding for the chunked local attention.

```python
class Gemma4AudioRelPositionalEncoding(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        self.hidden_size = config.hidden_size  # 1024
        self.context_size = (
            config.attention_chunk_size       # 12
            + config.attention_context_left - 1  # 12
            + config.attention_context_right  # 0
        )  # = 24
```

The `context_size` is computed as `chunk_size + (context_left - 1) + context_right = 12 + 12 + 0 = 24`.

**Timescale computation:** Standard sinusoidal encoding with `min_timescale=1.0`, `max_timescale=10000.0`, and `num_timescales = hidden_size // 2 = 512` frequencies. The `inv_timescales` buffer has shape `[1, 1, 512]`.

**Forward pass:**

1. Create descending position IDs: `torch.arange(12, -1, -1)` -> `[12, 11, 10, ..., 0]` (13 positions)
2. Multiply by `inv_timescales` -> `[13, 512]`
3. Concatenate `[sin, cos]` along last dim -> **output shape `[1, 13, 1024]`** (the leading batch dimension comes from broadcasting with `inv_timescales`)

The 13 positions correspond to `context_size - chunk_size + 1 = 24 - 12 + 1 = 13` relative offsets used by the `_rel_shift` mechanism in the attention layer.

---

## 4.4 Gemma4AudioAttention

Chunked local attention with relative position bias, following the Transformer-XL relative attention pattern.

### 4.4.1 Initialization

```python
class Gemma4AudioAttention(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        self.head_dim = config.hidden_size // config.num_attention_heads  # 1024 // 8 = 128
        self.num_heads = config.num_attention_heads  # 8
        self.q_scale = (self.head_dim ** -0.5) / math.log(2)   # softmax base-2 scaling
        self.k_scale = math.log(1 + math.e) / math.log(2)      # softplus base-2 normalization
        self.chunk_size = config.attention_chunk_size            # 12
        self.max_past_horizon = config.attention_context_left - 1  # 12
        self.max_future_horizon = config.attention_context_right   # 0
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon  # 24
```

The projections are:
- `q_proj`, `k_proj`, `v_proj`: `Gemma4ClippableLinear(1024, 1024)` -- multi-head Q/K/V
- `post`: `Gemma4ClippableLinear(1024, 1024)` -- output projection
- `relative_k_proj`: `nn.Linear(1024, 1024, bias=False)` -- relative position key projection (standard linear, not clippable)
- `per_dim_scale`: learnable parameter of shape `[head_dim]` = `[128]`
- `softcap`: buffer holding `Gemma4AudioConfig.attention_logit_cap` (default 50.0)

### 4.4.2 `_convert_to_block`

Splits a `[B, seq_len, num_heads, head_dim]` tensor into non-overlapping blocks of `chunk_size` along the sequence dimension.

```python
def _convert_to_block(self, hidden_states):
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
    pad = num_blocks * self.chunk_size - seq_len
    hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad))
    return hidden_states.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim)
```

**Output shape:** `[B, num_blocks, chunk_size, num_heads, head_dim]`

### 4.4.3 `_extract_block_context`

Extracts overlapping context windows of `context_size` for every block, strided by `chunk_size`. Used for keys and values so each query block can attend to its local neighborhood.

```python
def _extract_block_context(self, hidden_states):
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1))
    hidden_states = hidden_states.unfold(1, self.context_size, self.chunk_size)
    hidden_states = torch.movedim(hidden_states, -1, 2)
    return hidden_states  # [B, num_blocks, context_size, num_heads, head_dim]
```

The padding ensures that the first block has `max_past_horizon=12` zero-padded past positions, and the last block has up to `max_future_horizon + chunk_size - 1 = 11` zero-padded future positions. The `unfold` operation with `size=context_size=24` and `step=chunk_size=12` creates overlapping windows.

### 4.4.4 `_rel_shift`

Implements the Transformer-XL style relative position shift (Appendix B of [Dai et al., 2019](https://huggingface.co/papers/1901.02860)). This aligns the relative position logits so that position `i` in the query correctly indexes position `i-j` in the relative encoding.

```python
def _rel_shift(self, x):
    # x: [B, num_heads, num_blocks, chunk_size, position_length]
    batch_size, num_heads, num_blocks, block_size, position_length = x.shape
    context_size = self.context_size  # 24
    x = F.pad(x, (0, context_size + 1 - position_length))
    x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
    x = x[..., :block_size * context_size]
    return x.view(batch_size, num_heads, num_blocks, block_size, context_size)
```

**Output shape:** `[B, num_heads, num_blocks, chunk_size, context_size]` -- aligned with the content attention scores.

### 4.4.5 Forward Pass

```python
def forward(self, hidden_states, position_embeddings, attention_mask=None):
    batch_size, seq_length, _ = hidden_states.shape
    hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
```

**Step 1 -- Project and scale:**

```python
query_states = self.q_proj(hidden_states).float().view(hidden_shape)
key_states   = self.k_proj(hidden_states).float().view(hidden_shape)
value_states = self.v_proj(hidden_states).float().view(hidden_shape)

query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
key_states   = key_states * self.k_scale
```

Queries are scaled by `(head_dim^{-0.5} / ln2) * softplus(per_dim_scale)` -- a learned per-dimension scaling. Keys are scaled by `ln(1+e) / ln2`. All projections are cast to float32 for attention computation.

**Step 2 -- Block and extract context:**

```python
query_states = self._convert_to_block(query_states)   # [B, num_blocks, 12, 8, 128]
key_states   = self._extract_block_context(key_states) # [B, num_blocks, 24, 8, 128]
value_states = self._extract_block_context(value_states)
```

**Step 3 -- Content attention (matrix AC):**

```python
queries = query_states.permute(0, 3, 1, 2, 4)          # [B, 8, num_blocks, 12, 128]
matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2) # [B, 8, num_blocks, 12, 24]
```

**Step 4 -- Relative position attention (matrix BD):**

```python
relative_key_states = self.relative_k_proj(position_embeddings)  # [1, 13, 1024] -> [1, 13, 8, 128]
relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)

queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
relative_key_states = relative_key_states.to(dtype=query_states.dtype)
matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)  # [..., 13]
matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
matrix_bd = self._rel_shift(matrix_bd)  # -> [B, 8, num_blocks, 12, 24]
```

**Step 5 -- Combine, softcap, and attend:**

```python
attn_weights = matrix_ac + matrix_bd
attn_weights = attn_weights / self.softcap          # / 50.0
attn_weights = torch.tanh(attn_weights)             # squash to [-1, 1]
attn_weights = attn_weights * self.softcap          # rescale to [-50, 50]
```

The softcap mechanism bounds attention logits to `[-attention_logit_cap, +attention_logit_cap]` using `tanh` saturation, preventing extreme values.

```python
if attention_mask is not None:
    attn_weights = attn_weights.masked_fill(
        attention_mask.logical_not(), config.attention_invalid_logits_value  # -1e9
    )

attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
attn_weights = attn_weights.to(value_states.dtype)  # cast back from float32 to value dtype before matmul
attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)  # [B, 8, num_blocks, 12, 128]
```

**Step 6 -- Reshape and project output:**

```python
attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(B, num_blocks * chunk_size, -1)
attn_output = attn_output[:, :seq_length]  # trim padding
attn_output = self.post(attn_output.to(dtype=self.post.linear.weight.dtype))
```

**Returns:** `(attn_output: [B, seq_len, 1024], attn_weights)`

---

## 4.5 Gemma4AudioFeedForward

A Macaron-style feedforward block with pre-norm, post-norm, and a residual scaled by 0.5.

```python
class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        self.ffw_layer_1 = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size * 4)   # 1024 -> 4096
        self.ffw_layer_2 = Gemma4ClippableLinear(config, config.hidden_size * 4, config.hidden_size)   # 4096 -> 1024
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]   # SiLU
        self.gradient_clipping = config.gradient_clipping  # 1e10
        self.post_layer_scale = config.residual_weight     # 0.5
```

**Forward pass:**

```
residual = hidden_states
hidden_states = clamp(hidden_states, -gradient_clipping, gradient_clipping)
hidden_states = pre_layer_norm(hidden_states)
hidden_states = ffw_layer_1(hidden_states)     # [B, T, 4096]
hidden_states = SiLU(hidden_states)
hidden_states = ffw_layer_2(hidden_states)     # [B, T, 1024]
hidden_states = clamp(hidden_states, -gradient_clipping, gradient_clipping)
hidden_states = post_layer_norm(hidden_states)
hidden_states *= 0.5
hidden_states += residual
```

The gradient clipping is bounded by `min(gradient_clipping, finfo(dtype).max)` to avoid overflow issues with mixed precision. The `post_layer_scale` of 0.5 is characteristic of Macaron-Net conformer blocks where two feedforward modules sandwich the attention -- each contributes half to the residual.

---

## 4.6 Gemma4AudioCausalConv1d

Extends `nn.Conv1d` with left-padding for causal (non-look-ahead) convolution behavior.

```python
class Gemma4AudioCausalConv1d(nn.Conv1d):
    @cached_property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]
```

With the default `kernel_size=5`, `dilation=1`, `stride=1`: `left_pad = (5-1)*1 + 1 - 1 = 4`.

**Forward pass:** Pads `(left_pad, 0)` on the left only, then calls `nn.Conv1d.forward`. This ensures the convolution output at time `t` depends only on inputs at times `<= t`, maintaining causality.

---

## 4.7 Gemma4AudioLightConv1d

A lightweight convolution module combining a gated linear unit with a depthwise causal convolution.

```python
class Gemma4AudioLightConv1d(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        self.linear_start = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size * 2)  # 1024 -> 2048
        self.linear_end   = Gemma4ClippableLinear(config, config.hidden_size, config.hidden_size)      # 1024 -> 1024
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,     # 1024
            out_channels=config.hidden_size,    # 1024
            kernel_size=config.conv_kernel_size, # 5
            groups=config.hidden_size,          # 1024 (depthwise)
            bias=False,
        )
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        self.conv_norm      = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, with_scale=True)
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU
        self.gradient_clipping = config.gradient_clipping  # 1e10
```

**Forward pass:**

```
residual = hidden_states                              # [B, T, 1024]
hidden_states = pre_layer_norm(hidden_states)
hidden_states = linear_start(hidden_states)           # [B, T, 2048]
hidden_states = GLU(hidden_states, dim=-1)            # [B, T, 1024]  (splits and gates)
hidden_states = depthwise_conv1d(hidden_states.T).T   # transpose for Conv1d: [B, 1024, T] -> [B, 1024, T] -> [B, T, 1024]
hidden_states = clamp(hidden_states, -gradient_clipping, gradient_clipping)
hidden_states = conv_norm(hidden_states)
hidden_states = SiLU(hidden_states)
hidden_states = linear_end(hidden_states)             # [B, T, 1024]
hidden_states += residual
```

The GLU gate (`nn.functional.glu`) splits the 2048-dim tensor in half along the last dimension, applying sigmoid to one half and multiplying element-wise with the other. The depthwise convolution (`groups=hidden_size`) applies a separate kernel-5 filter per channel.

---

## 4.8 Gemma4AudioLayer

The full conformer block combining all the above sub-modules. Each of the 12 layers follows this structure:

```python
class Gemma4AudioLayer(nn.Module):
    def __init__(self, config, layer_idx):
        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn     = Gemma4AudioAttention(config, layer_idx)
        self.lconv1d       = Gemma4AudioLightConv1d(config)
        self.norm_pre_attn  = Gemma4RMSNorm(config.hidden_size)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_out       = Gemma4RMSNorm(config.hidden_size)
        self.gradient_clipping = config.gradient_clipping
```

**Forward pass (Macaron conformer order):**

```
# 1. First half-step feedforward
hidden_states = feed_forward1(hidden_states)          # pre-norm, FFW, post-norm, *0.5, +residual

# 2. Self-attention with pre/post norms
residual = hidden_states
hidden_states = clamp(hidden_states, ...)
hidden_states = norm_pre_attn(hidden_states)
hidden_states, _ = self_attn(hidden_states, position_embeddings, attention_mask)
hidden_states = clamp(hidden_states, ...)
hidden_states = norm_post_attn(hidden_states)
hidden_states += residual

# 3. Lightweight convolution
hidden_states = lconv1d(hidden_states)                # pre-norm, GLU, depthwise conv, +residual

# 4. Second half-step feedforward
hidden_states = feed_forward2(hidden_states)          # pre-norm, FFW, post-norm, *0.5, +residual

# 5. Final layer norm
hidden_states = clamp(hidden_states, ...)
hidden_states = norm_out(hidden_states)
```

This is the standard conformer architecture: **FFW -> MHSA -> Conv -> FFW -> Norm**, where both feedforward modules apply a 0.5 residual scale (Macaron-Net style).

---

## 4.9 Gemma4AudioModel

The top-level audio encoder module.

### 4.9.1 Initialization

```python
class Gemma4AudioModel(Gemma4PreTrainedModel):
    config: Gemma4AudioConfig
    main_input_name = "input_features"
    base_model_prefix = "model.audio_tower"

    def __init__(self, config: Gemma4AudioConfig):
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [Gemma4AudioLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.output_proj = nn.Linear(config.hidden_size, config.output_proj_dims, bias=True)
```

Note that `output_proj` is a plain `nn.Linear` with **bias=True** (unlike most other projections in the audio encoder), projecting from `hidden_size=1024` to `output_proj_dims=1536`. This output feeds into `Gemma4MultimodalEmbedder`, which further projects from 1536 to the text decoder's hidden dimension of 2304.

### 4.9.2 `_convert_4d_mask_to_blocked_5d`

Converts a standard 4D attention mask `[B, 1, seq_len, seq_len]` to the 5D blocked format `[B, 1, num_blocks, chunk_size, context_size]` expected by the chunked local attention.

```python
def _convert_4d_mask_to_blocked_5d(self, mask_4d):
    batch_size, _, seq_len, _ = mask_4d.shape
    chunk_size = self.config.attention_chunk_size           # 12
    max_past_horizon = self.config.attention_context_left - 1  # 12
    max_future_horizon = self.config.attention_context_right   # 0

    num_blocks = (seq_len + chunk_size - 1) // chunk_size
    padded_seq_len = num_blocks * chunk_size

    # Pad mask to multiple of chunk_size
    mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
    # Reshape into blocks
    mask_5d = mask_4d.reshape(B, 1, num_blocks, chunk_size, padded_seq_len)
    # Pad for context window
    mask_5d = F.pad(mask_5d, (max_past_horizon, max_future_horizon), value=False)
    # Gather context windows using strided indexing
    block_starts = torch.arange(num_blocks) * chunk_size
    offsets = torch.arange(context_size)
    kv_indices = block_starts[:, None] + offsets[None, :]
    return mask_5d.gather(-1, kv_indices)
```

**Output shape:** `[B, 1, num_blocks, chunk_size, context_size]` -- one boolean mask per query-key pair in the blocked attention.

### 4.9.3 Forward Pass

```python
def forward(self, input_features, attention_mask=None, **kwargs):
```

**Step 1 -- Subsample convolution:**

```python
hidden_states, output_mask = self.subsample_conv_projection(input_features, attention_mask)
# input_features: [B, T, 128]  (mel spectrogram)
# hidden_states:  [B, T/4, 1024]
# output_mask:    [B, T/4]
```

**Step 2 -- Relative position encoding:**

```python
position_embeddings = self.rel_pos_enc(hidden_states)  # [1, 13, 1024]
```

**Step 3 -- Build blocked attention mask:**

```python
attention_mask = create_bidirectional_mask(
    config=self.config,
    inputs_embeds=hidden_states,
    attention_mask=output_mask,
    and_mask_function=sliding_window_mask_function(
        (self.config.attention_context_left - 1, self.config.attention_context_right)  # (12, 0)
    ),
)
attention_mask = self._convert_4d_mask_to_blocked_5d(attention_mask)
# [B, 1, num_blocks, chunk_size, context_size]
```

The mask combines bidirectional attention with a sliding window constraint of `(12, 0)` -- each position can attend to 12 past positions and 0 future positions (causal within the local window).

**Step 4 -- Encoder layers:**

```python
for encoder_layer in self.layers[:self.config.num_hidden_layers]:
    hidden_states = encoder_layer(hidden_states, attention_mask, position_embeddings, **kwargs)
```

**Step 5 -- Output projection:**

```python
hidden_states = self.output_proj(hidden_states)  # [B, T/4, 1024] -> [B, T/4, 1536]
return Gemma4AudioModelOutput(last_hidden_state=hidden_states, attention_mask=output_mask)
```

The output has shape `[B, T/4, 1536]`, which feeds into `Gemma4MultimodalEmbedder` for projection to the text decoder's hidden dimension of 2304 before injection alongside text and vision tokens.

---

## 4.10 End-to-End Data Flow

```
Input mel spectrogram
  [B, T, 128]
       |
       v
Gemma4AudioSubSampleConvProjection
  layer0: Conv2d(1->128, k=3, s=2) + LayerNorm + ReLU
    [B, 128, T/2, 64]
  layer1: Conv2d(128->32, k=3, s=2) + LayerNorm + ReLU
    [B, 32, T/4, 32]
  reshape + input_proj_linear(1024 -> 1024)
    [B, T/4, 1024]
       |
       +-----> Gemma4AudioRelPositionalEncoding -> [1, 13, 1024]
       |                                              |
       v                                              |
  create_bidirectional_mask + sliding_window(12, 0)   |
  _convert_4d_mask_to_blocked_5d                      |
    [B, 1, num_blocks, 12, 24]                        |
       |                                              |
       v                                              v
  +-- Gemma4AudioLayer x12 (conformer) ----------------------+
  |                                                          |
  |  FFW1: RMSNorm -> Linear(1024->4096) -> SiLU             |
  |        -> Linear(4096->1024) -> RMSNorm -> *0.5 + res    |
  |                      |                                   |
  |  Attention: RMSNorm -> Q/K/V proj -> block/context split |
  |        -> content scores + relative pos scores           |
  |        -> softcap(tanh, 50.0) -> masked softmax          |
  |        -> attend values -> post proj -> RMSNorm + res    |
  |                      |                                   |
  |  LightConv: RMSNorm -> Linear(1024->2048) -> GLU         |
  |        -> depthwise CausalConv1d(k=5) -> RMSNorm         |
  |        -> SiLU -> Linear(1024->1024) + res               |
  |                      |                                   |
  |  FFW2: (same as FFW1)                                    |
  |                      |                                   |
  |  norm_out: RMSNorm                                       |
  +----------------------------------------------------------+
       |
       v
  output_proj: nn.Linear(1024 -> 1536, bias=True)
    [B, T/4, 1536]
       |
       v
  Gemma4AudioModelOutput
    .last_hidden_state: [B, T/4, 1536]
    .attention_mask:    [B, T/4]
       |
       v
  Gemma4MultimodalEmbedder
  (projects 1536 -> 2304 for text decoder)
```

---

## 4.11 TTNN Porting Considerations

**Subsampling convolutions.** The two `Conv2d` layers with `kernel=3, stride=2, padding=1` are standard operations with small channel counts (1->128->32). These can map to `ttnn.conv2d`. The interleaved `LayerNorm -> ReLU` sequence requires permuting between `[B, C, T, F]` and `[B, T, F, C]` formats; TTNN layout management will need to handle these transposes efficiently.

**Chunked local attention.** The `_convert_to_block` / `_extract_block_context` / `_rel_shift` trio implements custom blocked attention that does not map to standard `ttnn.transformer.attention`. A TTNN port must either: (a) implement the blocking logic as custom ops, or (b) pre-compute block indices and use gather/scatter. The fixed chunk_size=12 and context_size=24 are small enough that the attention matrices fit comfortably in L1.

**Relative position encoding.** The sinusoidal encoding is computed once and reused across all layers. The `relative_k_proj` linear followed by `_rel_shift` is a per-layer operation. Since position encodings are static for a given input length, they can be precomputed and stored as constants.

**Depthwise causal convolution.** `Gemma4AudioCausalConv1d` uses `groups=hidden_size=1024` (fully depthwise) with `kernel_size=5`. TTNN supports depthwise convolution via `ttnn.conv1d` with `groups` parameter. The left-padding of 4 elements must be handled explicitly.

**Gradient clipping as inference clamp.** The `torch.clamp` calls with `gradient_clipping=1e10` are effectively no-ops at inference with normal activations, but the `Gemma4ClippableLinear` input/output clamps with checkpoint-loaded bounds are meaningful and must be preserved. These map to `ttnn.clip`.

**Softcap attention.** The `tanh`-based softcap (`logits / cap -> tanh -> * cap`) is a pointwise operation that maps to `ttnn.tanh` and `ttnn.multiply`. This is the same softcap pattern used in Gemma 2 text attention.

**Output projection.** The final `nn.Linear(1024, 1536, bias=True)` is one of the few layers with bias in the audio encoder. The output dimension 1536 feeds into `Gemma4MultimodalEmbedder`, which projects from 1536 to the text decoder's hidden dimension of 2304.

**GLU gating.** The `nn.functional.glu` in `Gemma4AudioLightConv1d` splits a tensor and applies sigmoid gating. This can be decomposed into `ttnn.split` + `ttnn.sigmoid` + `ttnn.multiply`, or implemented as a fused op if available.

**Macaron-style double FFW.** Each conformer layer has two feedforward modules (FFW1, FFW2) with identical structure. These double the parameter and compute cost relative to a single-FFW transformer layer. The 0.5 residual scaling is a simple multiply.

---

**Next:** [Chapter 5 — Text Decoder](../ch5_text_decoder/index.md)
