# MTP Head Weight Inventory

## Introduction

Enumerating the exact weight tensors in the MTP head serves three concrete purposes:

1. **Memory placement (Chapter 5, `memory_placement_for_mtp.md`):** Knowing the total weight size determines whether the head's weights can reside in L1 SRAM or must be streamed from DRAM during each decode step. The per-tensor breakdown also allows finer-grained placement decisions — for example, placing the small layer norm weights in L1 while leaving the large projection matrices in DRAM.

2. **KV cache sizing:** The MTP head contains one self-attention sub-layer. If that layer maintains its own KV cache (resolved in Chapter 3, `huggingface_mtp_forward_pass.md`), the KV cache footprint is proportional to `num_key_value_heads × head_dim` per token per layer — a quantity derived directly from the weight shapes enumerated here.

3. **Weight-loading filter design:** The `tt-transformers` weight loader must distinguish MTP head weights (`model.future_prediction.0.*`) from backbone weights. The inventory here defines the complete set of keys to either load into TTNN tensors or discard, depending on whether MTP is enabled (see Chapter 3, `mtp_weight_loading_behavior.md`).

---

## MTP Head Configuration

The following hyperparameters from Qwen3.6-35B-A3B's `config.json` govern the MTP head's weight shapes. All values are taken directly from the configuration established in Chapter 1, `qwen36_mtp_config.md`.

```
mtp_num_hidden_layers = 1       # one MTP transformer block (draft depth N = 1)
hidden_size           = 7168    # H; shared with backbone
num_attention_heads   = 64      # query heads; matches backbone GQA config
num_key_value_heads   = 8       # KV heads; matches backbone GQA config
head_dim              = 112     # H / num_attention_heads = 7168 / 64 = 112
# MTP head uses a dense FFN (not MoE):
intermediate_size     = 2048    # dense FFN intermediate width
```

Note that `head_dim = 112`, not 128. This follows directly from `7168 / 64 = 112`. This distinction matters for deriving the correct attention projection shapes.

The MTP head does **not** use a Mixture-of-Experts FFN. While the backbone's transformer layers (with the exception of a small number of dense layers at fixed positions) use sparse MoE FFNs with 128 experts, the MTP head block uses a standard dense FFN with intermediate width `intermediate_size = 2048`. This makes the MTP head's FFN substantially lighter than a typical backbone MoE layer.

---

## Complete Weight Tensor Table

All weight tensors for `model.future_prediction.0` (the single MTP head block). Layer norm weights are one-dimensional vectors scaled elementwise in RMS normalization; all other weights are two-dimensional matrices.

### Attention Projections

| Key (under `model.future_prediction.0`) | Shape (symbolic) | Shape (concrete) | Parameters |
|------------------------------------------|------------------|------------------|-----------|
| `self_attn.q_proj.weight` | `[num_attention_heads × head_dim, hidden_size]` | `[7168, 7168]` | 51,380,224 |
| `self_attn.k_proj.weight` | `[num_key_value_heads × head_dim, hidden_size]` | `[896, 7168]` | 6,422,528 |
| `self_attn.v_proj.weight` | `[num_key_value_heads × head_dim, hidden_size]` | `[896, 7168]` | 6,422,528 |
| `self_attn.o_proj.weight` | `[hidden_size, num_attention_heads × head_dim]` | `[7168, 7168]` | 51,380,224 |

Attention projection subtotal: **115,605,504 parameters**.

Note that the output dimension of `q_proj` equals `64 × 112 = 7168 = hidden_size`, so `q_proj` and `o_proj` are square matrices. The `k_proj` and `v_proj` output dimension is `8 × 112 = 896`, reflecting the 8:1 GQA compression ratio.

### Input Combination Layer Norms

These two layer norm weights are applied to the backbone's final hidden state and the shifted token embedding respectively, before their element-wise addition to form the MTP head's combined input. They are distinct from both the backbone's layer norms and the MTP head block's internal layer norms.

| Key (under `model.future_prediction.0`) | Shape (symbolic) | Shape (concrete) | Parameters |
|------------------------------------------|------------------|------------------|-----------|
| `hnorm.weight` | `[hidden_size]` | `[7168]` | 7,168 |
| `enorm.weight` | `[hidden_size]` | `[7168]` | 7,168 |

`hnorm` normalizes the backbone hidden state $h_t$. `enorm` normalizes the shifted token embedding $\text{embed}(x_{t+1})$. After normalization, the two tensors are added element-wise to produce the combined input $c_t$ (see Chapter 1, `mtp_head_architecture.md` for the full input construction equation).

### Transformer Block Layer Norms

| Key (under `model.future_prediction.0`) | Shape (symbolic) | Shape (concrete) | Parameters |
|------------------------------------------|------------------|------------------|-----------|
| `input_layernorm.weight` | `[hidden_size]` | `[7168]` | 7,168 |
| `post_attention_layernorm.weight` | `[hidden_size]` | `[7168]` | 7,168 |

`input_layernorm` is the pre-attention RMS norm; `post_attention_layernorm` is the pre-FFN RMS norm. Both follow the standard pre-norm transformer convention used throughout the backbone.

All four layer norm subtotal: **28,672 parameters**.

### Dense FFN Projections

| Key (under `model.future_prediction.0`) | Shape (symbolic) | Shape (concrete) | Parameters |
|------------------------------------------|------------------|------------------|-----------|
| `mlp.gate_proj.weight` | `[intermediate_size, hidden_size]` | `[2048, 7168]` | 14,680,064 |
| `mlp.up_proj.weight` | `[intermediate_size, hidden_size]` | `[2048, 7168]` | 14,680,064 |
| `mlp.down_proj.weight` | `[hidden_size, intermediate_size]` | `[7168, 2048]` | 14,680,064 |

The dense FFN uses a SwiGLU activation: `gate_proj` and `up_proj` produce parallel intermediate projections; their element-wise product (after applying SiLU to the gate stream) is passed through `down_proj`. This is the same FFN structure as the backbone's dense layers and the backbone's individual MoE expert sub-networks.

Dense FFN subtotal: **44,040,192 parameters**.

### Not Included (Shared with Backbone)

| Key | Shape (concrete) | Parameters | Reason excluded |
|-----|-----------------|-----------|----------------|
| `lm_head.weight` | `[151936, 7168]` | 1,089,470,464 | Shared with backbone; not MTP-specific weight |

The `lm_head.weight` tensor is referenced by both the backbone's primary output projection and the MTP head's auxiliary output projection. It is stored once under the key `lm_head.weight` (not under `model.future_prediction.0.*`) and is not counted as part of the MTP head's dedicated parameter budget. See Chapter 1, `mtp_head_architecture.md` for the shared parameter discussion.

---

## Parameter Count Derivation

Let the following shorthand denote each group's parameter count:

- $P_q = 7168 \times 7168 = 51{,}380{,}224$
- $P_k = 896 \times 7168 = 6{,}422{,}528$
- $P_v = 896 \times 7168 = 6{,}422{,}528$
- $P_o = 7168 \times 7168 = 51{,}380{,}224$
- $P_{\text{hnorm}} = P_{\text{enorm}} = P_{\text{attn\_norm}} = P_{\text{ffn\_norm}} = 7{,}168$
- $P_{\text{gate}} = P_{\text{up}} = P_{\text{down}} = 2048 \times 7168 = 14{,}680{,}064$

The total parameter count is:

```math
\text{params}_{\text{MTP}} = \underbrace{(P_q + P_k + P_v + P_o)}_{\text{attention projections}} + \underbrace{(P_{\text{hnorm}} + P_{\text{enorm}} + P_{\text{attn\_norm}} + P_{\text{ffn\_norm}})}_{\text{4 layer norms}} + \underbrace{(P_{\text{gate}} + P_{\text{up}} + P_{\text{down}})}_{\text{dense FFN}}
```

Instantiated with concrete values:

**Attention projections:**

$$P_{\text{attn}} = 51{,}380{,}224 + 6{,}422{,}528 + 6{,}422{,}528 + 51{,}380{,}224 = 115{,}605{,}504$$

**Layer norms:**

$$P_{\text{norms}} = 4 \times 7{,}168 = 28{,}672$$

**Dense FFN:**

$$P_{\text{FFN}} = 3 \times 14{,}680{,}064 = 44{,}040{,}192$$

**Total:**

$$\text{params}_{\text{MTP}} = 115{,}605{,}504 + 28{,}672 + 44{,}040{,}192 = \mathbf{159{,}674{,}368} \approx 159.67\text{M}$$

---

## MTP Head vs. One Backbone Block

The table below compares the MTP head to a single backbone transformer block in terms of weight tensor sizes for the major components. The backbone block figures show one of the 94 MoE backbone layers; a small number of backbone layers use dense FFNs and would match the MTP head's FFN shape exactly.

| Component | MTP Head | One Backbone MoE Block |
|-----------|----------|----------------------|
| Attention: `q_proj` | `[7168, 7168]` | `[7168, 7168]` (identical) |
| Attention: `k_proj`, `v_proj` | `[896, 7168]` each | `[896, 7168]` each (identical) |
| Attention: `o_proj` | `[7168, 7168]` | `[7168, 7168]` (identical) |
| Attention subtotal | 115.6M params | 115.6M params (identical) |
| FFN type | Dense | Sparse MoE, 128 experts |
| FFN weight count (all experts) | 44.0M params | 128 × 44.0M = 5,637.1M params |
| FFN active params per token | 44.0M | 8 × 44.0M = 352M active (8 × 44,040,192 = 352,321,536) |
| Layer norms | 28,672 params | ~28,672 params (same structure) |
| **Block total** | **~159.7M params** | **~5,752.7M params** |

The MTP head's attention sub-layer is **identical** in shape and parameter count to a backbone attention sub-layer — both use 64 query heads, 8 KV heads, and head dimension 112. The difference lies entirely in the FFN: the MTP head uses a single dense FFN while each backbone MoE layer contains 128 expert sub-networks. The MTP head is therefore approximately $159.7 / 5{,}752.7 \approx 2.8\%$ the size of one full backbone MoE block, and roughly $159.7 / (115.6 + 44.0) \approx 1.0\times$ the size of a backbone block if only the attention weights and one dense expert are counted.

---

## Key Finding

> **MTP head parameter fraction:**
>
> The Qwen3.6-35B-A3B backbone contains approximately 35 billion parameters (the model name's "35B" refers to total parameters, with ~3B active per token due to MoE sparsity). The MTP head adds approximately 159.67M dedicated parameters.
>
> $$\frac{159{,}674{,}368}{35{,}000{,}000{,}000 + 159{,}674{,}368} \approx 0.45\%$$
>
> The MTP head represents less than half a percent of total model parameters. Its weight memory footprint (~305 MiB in BF16, derived in `mtp_memory_footprint.md`) is similarly small relative to the full model, but non-trivial relative to available on-chip L1 memory on Wormhole hardware.

---

## References

- [Qwen3] Qwen Team, "Qwen3 Technical Report", Alibaba Cloud, 2025.
- [DeepSeek-V3] DeepSeek-AI, "DeepSeek-V3 Technical Report", arXiv:2412.19437, 2024.
- Chapter 1, `qwen36_mtp_config.md` — source of all hyperparameter values used in this file.
- Chapter 1, `mtp_head_architecture.md` — source of the shared/unshared parameter taxonomy and the input construction equation.
- Chapter 2, `mtp_memory_footprint.md` — BF16 memory derivation from the parameter count established here.
- Chapter 3, `mtp_weight_loading_behavior.md` — weight key verification procedure and loading strategy.
- Chapter 5, `memory_placement_for_mtp.md` — placement decisions that depend on the weight sizes established here.

---

**Next:** [`mtp_memory_footprint.md`](./mtp_memory_footprint.md)
