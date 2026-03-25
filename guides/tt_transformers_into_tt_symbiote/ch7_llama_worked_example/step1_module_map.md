# Step 1: Module Map — HuggingFace to Symbiote to TT Transformers

## 1. HuggingFace `LlamaForCausalLM` Module Hierarchy

Loading `meta-llama/Llama-3.2-1B-Instruct` with `AutoModelForCausalLM.from_pretrained` produces the following relevant subtree (repeated across 16 decoder layers):

```
LlamaForCausalLM
  model: LlamaModel
    embed_tokens: Embedding                   # vocabulary embedding
    layers: ModuleList[16 x LlamaDecoderLayer]
      input_layernorm: LlamaRMSNorm
      self_attn: LlamaSdpaAttention           # HF SDPA-wrapped attention
        q_proj: Linear
        k_proj: Linear
        v_proj: Linear
        o_proj: Linear
        rotary_emb: LlamaRotaryEmbedding
      mlp: LlamaMLP                           # HF MLP with non-_modules attributes
        gate_proj: Linear
        up_proj: Linear
        down_proj: Linear
        act_fn: SiLUActivation                # stored as a callable, not a submodule
      post_feedforward_layernorm: LlamaRMSNorm
    norm: LlamaRMSNorm                        # final model norm
  lm_head: Linear                             # unembedding projection
```

Key observations:
- `LlamaSdpaAttention` is a subclass of `LlamaAttention`. The exact class is obtained at runtime via `model.model.layers[0].self_attn.__class__`.
- `LlamaRMSNorm` is a subclass of `torch.nn.Module` that stores `weight` as `nn.Parameter` and `variance_epsilon`. It is **not** a `torch.nn.LayerNorm`.
- `LlamaMLP.act_fn` is set as a direct attribute, not registered as a child `nn.Module` in `_modules`. This is the reason the wrapper described in section 2 is required.

## 2. The `LlamaMLP` Wrapper in `test_llama.py`

### Why It Is Needed

`register_module_replacement_dict` discovers replaceable submodules by iterating `model._modules` and matching `module.__class__` against the provided dictionaries. The HuggingFace `LlamaMLP.act_fn` is assigned as a plain attribute (not an `nn.Module` child), so it does not appear in `_modules` and is never visited by the replacement walker.

Additionally, the walker needs `nn.SiLU` instances to be present as registered child modules so they can be replaced by `TTNNSilu`. Wrapping the whole `LlamaMLP` in a plain `nn.Module` that re-registers `act_fn` as `nn.SiLU()` exposes it to the walker.

### How It Works

```python
class LlamaMLP(nn.Module):
    def __init__(self, old_layer):
        super().__init__()
        self.config = old_layer.config
        self.hidden_size = old_layer.hidden_size
        self.intermediate_size = old_layer.intermediate_size
        self.gate_proj = old_layer.gate_proj
        self.up_proj = old_layer.up_proj
        self.down_proj = old_layer.down_proj
        assert old_layer.config.hidden_act == "silu", "Only SiLU activation is supported in this test."
        self.act_fn = nn.SiLU()

    @classmethod
    def from_torch(cls, old_layer):
        return cls(old_layer)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

### How the Two-Pass Replacement Works

`test_llama.py` calls `register_module_replacement_dict` twice:

```python
# Pass 1: replace HF LlamaMLP with the wrapper
nn_to_nn = {
    model.model.layers[0].mlp.__class__: LlamaMLP,
}
modules = register_module_replacement_dict(model, nn_to_nn, model_config=None)

# Pass 2: replace nn.Linear, nn.SiLU, norms, and attention with TTNN equivalents
nn_to_ttnn = {
    nn.Linear: TTNNLinear,
    nn.SiLU: TTNNSilu,
    nn.LayerNorm: TTNNLayerNorm,
    model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
    model.model.layers[0].self_attn.__class__: LlamaAttention,
}
modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
```

## 3. Complete Mapping Table

The table below covers every class replaced in `test_llama.py` and its nearest TT Transformers equivalent. "Nearest equivalent" means the TT Transformers class that performs the same logical function at the same position in the forward pass.

| HuggingFace class | Symbiote replacement (`test_llama`) | Symbiote replacement (`test_llama_intelligent`) | Nearest TT Transformers equivalent |
|---|---|---|---|
| `LlamaMLP` (HF) | `LlamaMLP` (wrapper, `nn_to_nn` pass) | `LlamaMLP` (wrapper, same) | N/A — structural adapter |
| `nn.Linear` (inside `LlamaMLP`) | `TTNNLinear` | `SmartTTNNLinear` | `MLP.w1` / `MLP.w2` / `MLP.w3` (sharded `ttnn.linear` calls) |
| `nn.SiLU` (inside `LlamaMLP`) | `TTNNSilu` | `TTNNSilu` | Fused `ttnn.mul` with `input_tensor_a_activations=[ttnn.UnaryOpType.SILU]` in `MLP.forward` |
| `LlamaRMSNorm` (`input_layernorm`, `post_feedforward_layernorm`, final `norm`) | `TTNNRMSNorm` | `TTNNRMSNorm` | `RMSNorm` from `models/common/rmsnorm.py` (used inside `Attention.__init__` for Q/K norms) |
| `LlamaSdpaAttention` | `LlamaAttention` | `LlamaAttention` | `Attention` from `models/tt_transformers/tt/attention.py` |
| `nn.Linear` (inside `LlamaAttention`: `q_proj`, `k_proj`, `v_proj`, `o_proj`) | Four separate `TTNNLinear` instances (created inside `LlamaAttention.from_torch` via `init_parameters`) | Same — `LlamaAttention` creates its own linears regardless of outer `nn_to_ttnn` | `Attention.wqkv` (fused) / `Attention.wo` (separate sharded tensors) |
| `nn.LayerNorm` | `TTNNLayerNorm` | `TTNNLayerNorm` | Not present in LLaMA 3.x (LLaMA uses RMSNorm; this entry is defensive) |

Notes on the attention row: `LlamaAttention.from_torch` branches based on whether the Q, K, and V projection weights all share the same shape. When they do (equal-head-count models), it calls `init_fused_parameters`, which creates a single `TTNNFusedQKVSelfAttention` (wrapping a concatenated Q+K+V `TTNNLinear`) and a separate `TTNNLinear` for `o_proj`. LLaMA 3.2-1B uses Grouped-Query Attention (`num_attention_heads=32`, `num_key_value_heads=8`), so `q_proj` is `(2048, 2048)` while `k_proj` and `v_proj` are `(512, 2048)` — shapes differ. The `qkv_same_shape` check is `False`, so `init_parameters()` is called instead, producing four separate `TTNNLinear` instances for `q_proj`, `k_proj`, `v_proj`, and `o_proj`. All of these linears are constructed directly inside `from_torch` and are not subject to the outer `nn_to_ttnn` dictionary. When `SmartTTNNLinear` is used as the replacement for `nn.Linear`, it affects only the `gate_proj`, `up_proj`, and `down_proj` inside the `LlamaMLP` wrapper; the attention projections remain `TTNNLinear`.

## 4. `exclude_replacement` — What Is Excluded and Why

`test_llama` (the basic variant) does **not** pass `exclude_replacement`, so `lm_head` is replaced:

```python
modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
```

`test_llama_intelligent` explicitly excludes `lm_head`:

```python
modules = register_module_replacement_dict(
    model, nn_to_ttnn, model_config=None, exclude_replacement={"lm_head"}
)
```

`exclude_replacement` is a `Set[str]` of module names. The walker checks whether a module's name is in this set before replacing it. `lm_head` is the final `nn.Linear` that projects hidden states to vocabulary logits (shape `[hidden_size, vocab_size]`).

Reasons for excluding `lm_head` in the intelligent variant:

1. `SmartTTNNLinear._get_prefill_pc` has an explicit early-exit for modules named `lm_head` (`self._prefill_pc_cache[seq_len] = None; return None`), meaning no multi-cast reuse program config is computed for it. Replacing it with `SmartTTNNLinear` but not excluding it would still work, but the prefill optimization does not apply.
2. The vocabulary projection is typically the largest single `Linear` in the model. During decode, it is invoked once per generated token. Keeping it on Torch CPU avoids device memory pressure for a layer whose latency is dominated by the large output dimension, not compute throughput.
3. Precision-sensitive: a quantized `lm_head` can shift the argmax of logits even when PCC is high, producing different token selections. Excluding it provides a stable reference point for correctness comparisons.

## 5. Modules Not Currently Replaced in Symbiote That TT Transformers Handles Natively

The following modules are present in `LlamaForCausalLM` but are not touched by either test variant:

| Module | HuggingFace class | TT Transformers handling |
|---|---|---|
| Token embedding | `nn.Embedding` | Loaded as a sharded `ttnn.Tensor` directly; not a `LightweightModule` subclass |
| LM head (when excluded) | `nn.Linear` | Treated as a vocabulary decode step in the TT Transformers model loop; handled by a dedicated function outside `Attention` / `MLP` |
| `rotary_emb` inside `LlamaSdpaAttention` | `LlamaRotaryEmbedding` | TT Transformers pre-computes `transformation_mats` once and passes them as constructor arguments to `Attention`; rotation is applied inside `Attention.forward` via `ttnn` RoPE ops |
| KV cache | HF `DynamicCache` | TT Transformers uses its own paged KV cache; Symbiote's `LlamaAttention` delegates to whatever `past_key_values` object HuggingFace passes (including `DynamicCache`) via `past_key_values.update(...)` |

The absence of an `Embedding` replacement is significant: all embedding lookups during `model.generate` run on CPU in PyTorch. In TT Transformers the embedding table is loaded as a device tensor and the lookup is a TTNN gather. Bridging this gap is outside the scope of the current Symbiote test but would be required for full device-resident inference.
