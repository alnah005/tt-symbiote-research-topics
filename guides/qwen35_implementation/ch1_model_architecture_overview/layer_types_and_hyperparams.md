# Layer Types and Hyperparameters

This file documents the exact hyperparameter values for every layer type in the two
implemented Qwen3.5 variants. All values are sourced from the test constants in
`models/demos/qwen35/tests/test_pcc.py` (27B) and `tests/test_a3b_pcc.py` (35B-A3B),
cross-referenced against the architecture table in `models/demos/qwen35/README.md` and
the `model_config.py` parsing logic.

---

## DeltaNet Layer Hyperparameters

DeltaNet layers occupy 3/4 of all layers in each model variant. Their hyperparameters
are read from the HF `config.json` by `model_config.py` under the `linear_*` keys and
stored on the `ModelArgs` object.

### 27B Dense — DeltaNet Hyperparameters

| Parameter | Config key | Value |
|-----------|-----------|-------|
| Number of K-heads | `linear_num_key_heads` | 16 |
| Number of V-heads | `linear_num_value_heads` | 48 |
| Key head dimension | `linear_key_head_dim` | 128 |
| Value head dimension | `linear_value_head_dim` | 128 |
| Conv1d kernel size | `linear_conv_kernel_dim` | 4 |
| GQA ratio ($N_V / N_K$) | derived | 3 |
| Key dimension ($N_K \times d_K$) | derived | 2048 |
| Value dimension ($N_V \times d_V$) | derived | 6144 |
| Conv dimension ($2 \times \text{key\_dim} + \text{value\_dim}$) | derived | 10240 |

### 35B-A3B (MoE) — DeltaNet Hyperparameters

| Parameter | Config key | Value |
|-----------|-----------|-------|
| Number of K-heads | `linear_num_key_heads` | 16 |
| Number of V-heads | `linear_num_value_heads` | 32 |
| Key head dimension | `linear_key_head_dim` | 128 |
| Value head dimension | `linear_value_head_dim` | 128 |
| Conv1d kernel size | `linear_conv_kernel_dim` | 4 |
| GQA ratio ($N_V / N_K$) | derived | 2 |
| Key dimension ($N_K \times d_K$) | derived | 2048 |
| Value dimension ($N_V \times d_V$) | derived | 4096 |
| Conv dimension ($2 \times \text{key\_dim} + \text{value\_dim}$) | derived | 8192 |

### DeltaNet Derived Dimensions

The `GatedDeltaNet` constructor computes several dimensions from the above hyperparameters:

```python
self.key_dim   = self.head_k_dim * self.num_k_heads
self.value_dim = self.head_v_dim * self.num_v_heads
self.conv_dim  = self.key_dim * 2 + self.value_dim
self.gqa_ratio = self.num_v_heads // self.num_k_heads
self.scale     = 1.0 / math.sqrt(self.head_k_dim)
```

The `conv_dim` determines the size of the fused `in_proj_qkv` projection output that
feeds the conv1d ring buffer. The ring buffer has `conv_kernel_size = 4` slots, each
of size [B, conv_dim]. Chapter 2 covers the conv1d mechanism in detail.

The `_proj_splits` list used for splitting the fused `in_proj_all` output is:

```python
self._proj_splits = [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads]
# 27B:  [10240, 6144, 48, 48]
# A3B:  [8192,  4096, 32, 32]
```

These four slices correspond to the qkv (fed to conv1d), z (gated output), b (beta
gate), and a (decay gate) projections respectively.

### Detection in model_config.py

The presence of `linear_num_key_heads` in `config.json` is the sole signal used to
identify a Qwen3.5 model:

```python
self.is_qwen35      = self.linear_num_key_heads is not None
self.is_qwen35_moe  = self.is_qwen35 and self.num_experts > 0
```

If `is_qwen35` is true, the config also sets `use_hf_rope = True` and
`rms_norm_add_unit_offset = True` (zero-centered RMSNorm: $\text{output} \times (1 + w)$).

---

## Full-Attention Layer Hyperparameters

Full-attention layers occupy 1/4 of all layers. They use `GatedAttention`, which extends
the standard `Attention` base class with an output gate and Qwen3.5-specific partial RoPE.

### 27B Dense — Full-Attention Hyperparameters

| Parameter | Config key | Value |
|-----------|-----------|-------|
| Q heads | `num_attention_heads` | 24 |
| KV heads | `num_key_value_heads` | 4 |
| Head dimension | `head_dim` | 256 |
| GQA ratio ($N_Q / N_{KV}$) | derived | 6 |
| Partial rotary factor | `rope_parameters.partial_rotary_factor` | 0.25 |
| Rotary dimension ($\text{head\_dim} \times \text{factor}$) | derived | 64 |
| RoPE theta | `rope_parameters.rope_theta` | 1,000,000.0 |
| RMSNorm epsilon | `rms_norm_eps` | 1e-6 |
| Number of KV cache layers | derived (16 full-attention layers) | 16 |
| RoPE setup class | (determined by `use_hf_rope`) | `RotarySetup` |

### 35B-A3B (MoE) — Full-Attention Hyperparameters

| Parameter | Config key | Value |
|-----------|-----------|-------|
| Q heads | `num_attention_heads` | 16 |
| KV heads | `num_key_value_heads` | 2 |
| Head dimension | `head_dim` | 256 |
| GQA ratio ($N_Q / N_{KV}$) | derived | 8 |
| Partial rotary factor | `rope_parameters.partial_rotary_factor` | 0.25 |
| Rotary dimension ($\text{head\_dim} \times \text{factor}$) | derived | 64 |
| RoPE theta | `rope_parameters.rope_theta` | 1,000,000.0 |
| RMSNorm epsilon | `rms_norm_eps` | 1e-6 |
| Number of KV cache layers | derived (10 full-attention layers) | 10 |
| RoPE setup class | (determined by `use_hf_rope`) | `HfRotarySetup` |

### Partial RoPE Note

Qwen3.5 applies RoPE to only the first 64 dimensions of each $d = 256$ attention head
($\text{partial\_rotary\_factor} = 0.25$, $\text{rotary\_dim} = 64$). The remaining
192 dimensions are left unrotated. This requires corrected frequency computation using
$\text{rotary\_dim} = 64$ rather than $\text{head\_dim} = 256$ in the standard formula:

$$\theta_i = \frac{1}{\text{rope\_theta}^{2i / \text{rotary\_dim}}} \quad \text{for} \quad i = 0, 1, \ldots, \frac{\text{rotary\_dim}}{2} - 1$$

The 27B and 35B-A3B demos implement this correction differently because they use
different RoPE setup classes (`RotarySetup` vs `HfRotarySetup`), which address cos/sin
matrix positions differently. Chapter 3 covers partial RoPE in detail.

### Q/K RMSNorm

Both variants apply per-head RMSNorm to Q and K before the attention computation,
using zero-centered weights (`add_unit_offset=True`):

$$\text{output} = x \cdot \frac{1}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot (1 + w)$$

Separate `q_norm.weight` and `k_norm.weight` tensors are loaded per attention layer.
The epsilon used is the same `norm_eps = 1e-6` as the layer norms.

---

## MoE-Specific Hyperparameters (35B-A3B Only)

The 35B-A3B model replaces the standard SwiGLU MLP with a Mixture-of-Experts MLP
at every one of its 40 layers. The MoE hyperparameters are:

| Parameter | Config key | Value |
|-----------|-----------|-------|
| Total experts | `num_experts` | 256 |
| Active experts per token (top-k) | `num_experts_per_tok` | 8 |
| Expert hidden dimension | `moe_intermediate_size` | 512 |
| Shared expert hidden dimension | `shared_expert_intermediate_size` | 512 |

The `hidden_dim` used internally by `model_config.py` for MoE models is set to
`moe_intermediate_size` (512), not `intermediate_size` (which is absent from the A3B
config):

```python
elif self.moe_intermediate_size is not None:
    # MoE models (e.g. Qwen3.5-35B-A3B) have no intermediate_size,
    # only moe_intermediate_size for expert MLPs.
    self.hidden_dim = self.moe_intermediate_size
```

**Expert weight tensors** are stored in a fused layout in the safetensors checkpoint:

- `mlp.experts.gate_up_proj` — shape [256, 1024, 2048] (gate+up fused, hidden=512 so $2 \times 512 = 1024$ output width, input=2048)
- `mlp.experts.down_proj` — shape [256, 2048, 512]

**Shared expert** is a standard SwiGLU MLP with `hidden_dim = 512` that is always
active regardless of routing. A scalar gate (`shared_expert_gate.weight`) controls how
much of the shared expert output is added to the routed expert sum:

$$\text{out} = \text{shared\_out} \cdot \sigma(\text{token} \cdot w_{\text{gate}}) + \sum_{i \in \text{top-k}} r_i \cdot \text{expert}_i(\text{token})$$

where $r_i$ are the softmax-normalized routing weights for the selected top-8 experts.

**Routing** is computed on device (router matmul), with topk and softmax performed on
the host (the 256-logit vector is synced to CPU per layer per token). Chapter 5 covers
the full MoE implementation.

---

## Vocabulary Size and Host Embedding

Both models share the same vocabulary:

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 248,320 |
| Padded vocab size (tile-aligned) | 248,320 (already tile-aligned) |
| Embedding dimension | same as `hidden_size` (5120 for 27B, 2048 for A3B) |

The token embedding table is kept on host CPU as a `float32` tensor and never
transferred to the device:

```python
emb_weight_cpu = sd[args.get_state_dict_prefix("", None) + "tok_embeddings.weight"].float()

# Per-token embedding lookup (in the generate loop):
emb_vec = emb_weight_cpu[tok].unsqueeze(0)   # shape [1, hidden_size]
x_pad = torch.zeros(1, 1, B, args.dim)
x_pad[0, 0, 0, :] = emb_vec
x = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
```

The decision to keep the embedding on host avoids transferring the full
248,320 × hidden_size weight table to device DRAM. For single-token decode, the
embedding lookup is a single indexed read of a 5,120 or 2,048-element vector — negligible
cost relative to the 86 ms per-token latency.

The LM head (the linear projection from hidden states to logits) is kept on device in
`bfloat8_b` precision. Its shape is transposed to [1, 1, hidden_size, 248320] for
efficient `ttnn.linear`. The final logit selection (argmax over vocab) happens on host
after a single `to_torch` call:

```python
logits_cpu = ttnn.to_torch(logits_tt).float()[0, 0, 0, : args.vocab_size]
next_token = logits_cpu.argmax().item()
```

---

**Next:** [Chapter 2 — GatedDeltaNet: Linear Attention on Blackhole](../ch2_gated_deltanet_linear_attention_on_blackhole/index.md)
