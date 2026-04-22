# Text Decoder Hyperparameters

This file walks through every field in the top-level `config.json` for dots.ocr that governs the text decoder. All values are taken directly from the published `config.json` of `rednote-hilab/dots.ocr`.

## Full config listing (text decoder fields)

```json
{
    "architectures": ["DotsOCRForCausalLM"],
    "model_type": "dots_ocr",
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "max_position_embeddings": 131072,
    "sliding_window": 131072,
    "use_sliding_window": false,
    "rope_theta": 1000000,
    "vocab_size": 151936,
    "image_token_id": 151665,
    "video_token_id": 151656,
    "hidden_act": "silu",
    "attention_bias": true,
    "attention_dropout": 0.0,
    "torch_dtype": "bfloat16",
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": false
}
```

## Architecture class and model type

`architectures: ["DotsOCRForCausalLM"]` names the class registered in HuggingFace's AutoModel registry. `DotsOCRForCausalLM` inherits from the `Qwen2ForCausalLM` base pattern. `model_type: "dots_ocr"` is the string key used by `AutoConfig` to dispatch to the correct config class.

These two fields together establish that dots.ocr is a derived Qwen2-family decoder with a distinct model type registration — not a fork that reuses the `qwen2` type string.

## Width and depth

`hidden_size: 1536` is the model dimension $d_{model}$. Every embedding vector, attention output, and residual stream operates at this width.

`num_hidden_layers: 28` is the total number of transformer decoder blocks. Each block contains one self-attention sublayer and one feed-forward MLP sublayer, each wrapped with RMSNorm and a residual connection.

`intermediate_size: 8960` is the inner dimension of each MLP block. The ratio of intermediate to hidden size is:

$$\frac{8960}{1536} \approx 5.83$$

The conventional SwiGLU target is approximately $\frac{8}{3} d_{model}$, which for $d_{model} = 1536$ would be $4096$. The chosen 8960 is substantially wider than that target, increasing MLP capacity relative to attention capacity.

## Grouped Query Attention (GQA 12Q/2KV)

`num_attention_heads: 12` sets the number of query heads. `num_key_value_heads: 2` sets the number of key and value heads. This is a 6:1 ratio — one of the more aggressive GQA configurations in published models. Each KV head is shared by 6 query heads.

The per-head dimension is:

$$d_{head} = \frac{d_{model}}{n_q} = \frac{1536}{12} = 128$$

The four projection matrices have the following shapes and parameter counts:

| Projection | Shape | Parameters |
|---|---|---|
| Q | $1536 \times (12 \times 128) = 1536 \times 1536$ | 2,359,296 |
| K | $1536 \times (2 \times 128) = 1536 \times 256$ | 393,216 |
| V | $1536 \times (2 \times 128) = 1536 \times 256$ | 393,216 |
| O | $(12 \times 128) \times 1536 = 1536 \times 1536$ | 2,359,296 |
| **Attention weights total** | | **5,505,024** |

`attention_bias: true` adds a bias vector to each of the four projection matrices. The bias dimensions are:

$$1536 + 256 + 256 + 1536 = 3{,}584 \text{ parameters per layer}$$

This is negligible relative to the weight matrices but must be accounted for during weight loading and kernel dispatch.

`attention_dropout: 0.0` disables attention dropout. The field is present for schema compatibility; it has no effect at inference time.

## SwiGLU activation

`hidden_act: "silu"` specifies the SiLU activation, $\text{SiLU}(x) = x \cdot \sigma(x)$. In a Qwen2-style MLP, SiLU is used as the gating nonlinearity in the SwiGLU formulation:

$$\text{MLP}(x) = \left(\text{SiLU}(W_{\text{gate}} \cdot x) \odot (W_{\text{up}} \cdot x)\right) \cdot W_{\text{down}}^T$$

where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{8960 \times 1536}$ and $W_{\text{down}} \in \mathbb{R}^{1536 \times 8960}$.

The three weight matrices contribute the following parameters per block:

$$3 \times 1536 \times 8960 = 41{,}287{,}680 \approx 41.3\text{M}$$

## RoPE configuration

`rope_theta: 1000000` sets the base frequency for Rotary Position Embedding to $\theta_{\text{base}} = 10^6$. For reference, the original LLaMA base is $10^4$ and Qwen2 uses $10^6$ in its larger variants. A higher base frequency extends the effective usable range of RoPE, consistent with the very long context window (131072 tokens).

The `DotsModelArgs` additionally sets `use_hf_rope: True`, selecting the HuggingFace-compatible RoPE implementation rather than a custom kernel variant. This matches the Qwen2 family convention.

## Context length

`max_position_embeddings: 131072` sets the maximum sequence length to 131,072 tokens. This is four times the context length of Qwen2.5-VL-7B (32,768 tokens). The extended context is necessary to handle long documents: a high-resolution multi-page document can produce thousands of vision tokens after the spatial merge step, which must fit within the same sequence as text tokens.

`sliding_window: 131072` and `use_sliding_window: false` — windowed attention is disabled; both fields exist for Qwen2 schema compatibility only.

## Vocabulary and special tokens

`vocab_size: 151936` is the total vocabulary size, shared with the Qwen2 and Qwen2.5 family. The tokenizer is BPE-based (tiktoken). Token IDs 0 through 151,642 cover standard text tokens; the range above 151,643 is reserved for special tokens.

`image_token_id: 151665` is the sentinel inserted into the input token sequence at each position to be filled by a vision feature vector. The image token ID falls in the special token range.

`video_token_id: 151656` is present but unused at runtime — dots.ocr is static-image-only (see [`vision_encoder_specs.md`](./vision_encoder_specs.md)).

`tie_word_embeddings: false` means the input embedding matrix and the output lm head projection are independent. Both are present in the checkpoint and counted separately.

## Normalization

`rms_norm_eps: 1e-06` is the epsilon in the RMSNorm denominator:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

Each decoder block contains two RMSNorm instances (one before attention, one before the MLP — a standard pre-norm arrangement). A final RMSNorm is applied to the output of the last block before the lm head. Each RMSNorm instance has a learned scale vector $\gamma \in \mathbb{R}^{1536}$.

Note: the text decoder uses $\epsilon = 10^{-6}$ while the vision encoder uses $\epsilon = 10^{-5}$. These must not be mixed during weight loading or kernel configuration.

## Precision

`torch_dtype: "bfloat16"` specifies the default weight storage and compute dtype. All weights are stored in BF16 in the published checkpoint. This is the natural dtype for Tenstorrent hardware targets.

## Per-layer and total parameter count

Combining the above, the parameter count per transformer block is:

| Component | Parameters |
|---|---|
| Attention weights (Q, K, V, O) | 5,505,024 |
| Attention biases (Q, K, V, O) | 3,584 |
| MLP (gate + up + down) | 41,287,680 |
| 2x RMSNorm scale vectors ($2 \times 1536$) | 3,072 |
| **Block total** | **46,799,360** |

Across 28 layers:

$$28 \times 46{,}799{,}360 = 1{,}310{,}382{,}080$$

Non-layer components:

| Component | Shape | Parameters |
|---|---|---|
| Input embedding | $151936 \times 1536$ | 233,373,696 |
| lm head (untied) | $1536 \times 151936$ | 233,373,696 |
| Final RMSNorm | $1536$ | 1,536 |
| **Non-layer total** | | **466,748,928** |

Full text decoder parameter count:

$$1{,}310{,}382{,}080 + 466{,}748{,}928 = 1{,}777{,}131{,}008 \approx 1{,}777\text{M}$$

This is the count including both embedding tables. The derived total of 1,777,131,008 ≈ 1.78B is what the model card rounds to "1.7B LLM" — it refers to the text decoder alone; see [`vision_encoder_specs.md`](./vision_encoder_specs.md) for the full-model total including the vision encoder. The embedding tables are the largest single contributor to the text decoder count outside the transformer blocks.

---

**Next:** [`vision_encoder_specs.md`](./vision_encoder_specs.md)
