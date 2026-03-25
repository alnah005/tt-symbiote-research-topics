# Decoder Block Assembly Recipe

This document explains how TT Transformers wires a complete decoder block, describes
the current Symbiote approach for LLaMA, and provides a step-by-step recipe for
assembling a Symbiote decoder block that uses TT-Transformers-grade operations.  It
closes with the performance gap that remains around residual additions.

---

## 1. How TT Transformers wires a `TransformerBlock`

Source: `models/tt_transformers/tt/decoder.py` — `TransformerBlock.__init__` and
`TransformerBlock.forward`.

### 1.1 Module graph constructed in `__init__`

```
TransformerBlock
├── attention_norm  : DistributedNorm(RMSNorm(weight_key="attention_norm"))
├── attention       : Attention  (or custom attention_class)
├── ff_norm         : DistributedNorm(RMSNorm(weight_key="ffn_norm"))
├── pre_ff_norm     : DistributedNorm(RMSNorm(weight_key="pre_feedforward_layernorm"))  -- optional
├── post_ff_norm    : DistributedNorm(RMSNorm(weight_key="post_feedforward_layernorm")) -- optional
└── feed_forward    : MLP  (or TtMoeLayer for MoE models)
```

Every `DistributedNorm` wraps a `RMSNorm` that carries both a replicated weight
(`self.norm.weight`) and, when `args.is_distributed_norm(mode)` is truthy, a
sharded copy (`self.norm.weight_distributed`).

### 1.2 `forward` data flow

The forward pass executes as follows (non-TG, standard LLaMA topology):

```
x (fractured across devices, in skip_mem_cfg)
│
├─ residual = x
│
├─ attn_in = attention_norm(x, mode, norm_config=attn_norm_config)
│    DistributedNorm: [all_gather if multichip non-distributed] → RMSNorm → [all_gather if distributed]
│
├─ attn_out = attention.forward(attn_in, ...)
│    Output is fractured (column-parallel output across devices)
│
├─ attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
│
├─ [if pre_ff_norm is None]
│    hidden = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
│    residual = hidden
│    x.deallocate(True)  [prefill only]
│  [else]
│    hidden = attn_out   (no residual add yet; deferred until after ff_norm)
│
├─ hidden = ff_norm(hidden, mode, norm_config=ff_norm_config)
│
├─ [if pre_ff_norm is not None]
│    [if multichip and not distributed] ttnn.mesh_partition(hidden, dim=3, cluster_axis=1)
│    hidden = ttnn.add(residual, hidden, memory_config=skip_mem_cfg)
│    residual = hidden
│    hidden = pre_ff_norm(hidden, mode, norm_config=pre_ff_norm_config)
│
├─ [TG + decode] ttnn.to_memory_config(hidden, mlp_act_mem_config)
│
├─ hidden = feed_forward.forward(hidden, mode)
│    Output is fractured
│
├─ [if post_ff_norm is not None]
│    hidden = post_ff_norm(hidden, mode, norm_config=post_ff_norm_config)
│    [if multichip and not distributed] ttnn.mesh_partition(hidden, dim=3, cluster_axis=1)
│
└─ out = ttnn.add(residual, hidden,
                  memory_config=skip_mem_cfg,
                  dtype=ccl_dtype if TG and not distributed else activation_dtype or bfloat16)
```

**Key points:**
- `residual` is always in `skip_mem_cfg` — a memory config returned by
  `args.get_residual_mem_config(mode, prefetcher)`.  For decode on non-TG hardware
  this is typically an L1-sharded config.  For prefill it is `DRAM_MEMORY_CONFIG`.
- Every `ttnn.add` call explicitly passes `memory_config=skip_mem_cfg`.  The dtype
  handling differs by add site: intermediate residual adds pass `dtype=ttnn.bfloat16`
  for TG and `dtype=None` for non-TG (`decoder.py` lines 250–252, 272–274); only the
  final residual add uses `activation_dtype or ttnn.bfloat16` for non-TG (`decoder.py`
  lines 302–309).
- `attn_out.deallocate()` is called explicitly after `ff_norm` (and after the
  conditional `pre_ff_norm` residual-add block), at `decoder.py` line 279 — not
  immediately after the first residual add.
- The norm → residual add ordering is conditional on whether `pre_ff_norm` is
  present: standard LLaMA uses `norm → residual → MLP`, while Gemma 2 uses
  `residual → norm → additional norm → MLP`.

---

## 2. Current Symbiote approach for LLaMA

Source: `models/experimental/tt_symbiote/tests/test_llama.py` and
`models/experimental/tt_symbiote/utils/module_replacement.py`.

### 2.1 The `register_module_replacement_dict` pattern

Symbiote does not define a Symbiote-native `TransformerBlock` class.  Instead it
walks the existing HuggingFace model graph and **replaces leaf modules in-place**:

```python
nn_to_nn = {
    LlamaMLP: LlamaMLP,   # re-wraps the HF MLP to use plain SiLU
}
nn_to_ttnn = {
    nn.Linear:          TTNNLinear,
    nn.SiLU:            TTNNSilu,
    nn.LayerNorm:       TTNNLayerNorm,
    LlamaRMSNorm:       TTNNRMSNorm,   # matched by exact class from the loaded model
    LlamaSdpaAttention: LlamaAttention,
}
modules = register_module_replacement_dict(model, nn_to_nn, model_config=None)
modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
```

`register_module_replacement_dict` (`utils/module_replacement.py`) recurses through
`model._modules` and replaces any module whose class appears as a key in the dict by
calling `NewClass.from_torch(old_module)`.  The returned `modules` dict maps
`module_name` strings to the new `TTNNModule` instances for weight lifecycle
management.

After replacement the HF `LlamaDecoderLayer.forward` method still runs **unchanged**
as PyTorch Python code — it calls `self.input_layernorm(x)`, `self.self_attn(x)`,
`self.post_attention_layernorm(x)`, `self.mlp(x)`, and performs residual additions
with the `+` operator.  The Symbiote-accelerated modules handle individual ops inside
each replaced leaf, but the *block-level control flow* — norm invocation order,
residual add sites, memory placement between ops — is fully determined by the HF
`LlamaDecoderLayer.forward` source.

### 2.2 What the replacement covers

| HF module replaced | Symbiote replacement | Forward path |
|--------------------|----------------------|-------------|
| `LlamaRMSNorm` (input_layernorm, post_attention_layernorm) | `TTNNRMSNorm` | `ttnn.rms_norm(x, weight=self.tt_weight, epsilon=...)` |
| `nn.Linear` (q/k/v/o projections, mlp gate/up/down) | `TTNNLinear` | `ttnn.linear(x, self.tt_weight)` with DRAM interleaved output |
| `LlamaSdpaAttention` | `LlamaAttention` | TTNN-native attention |
| `LlamaMLP` (after the `nn_to_nn` pass) | user-defined `LlamaMLP` wrapper | decomposes into `gate_proj`, `act_fn`, `up_proj`, `down_proj` as separate calls |

### 2.3 What is not covered

- Residual additions remain PyTorch (`tensor + tensor`); they go through the
  `TorchTTNNTensor` dispatcher.  No TTNN memory config is specified; the output
  lands wherever TTNN defaults to (typically interleaved DRAM).
- No `DistributedNorm` wrapper; `TTNNRMSNorm` receives activations directly from
  HF's forward, which are single-device tensors on the registered `device`.
- No `pre_ff_norm` / `post_ff_norm` sites are modelled.

---

## 3. Step-by-step recipe: `TTNNLlamaDecoder`

The following recipe describes how to assemble a Symbiote decoder block that
replicates TT Transformers' `TransformerBlock` structure.  Each step references the
exact source behavior it replicates.

> **Scope:** This recipe targets standard (non-MoE, non-Gemma) LLaMA-family
> architectures on T3K (8 devices, non-TG).  Gemma and MoE variants require
> additional norms (see [normalization_comparison.md](normalization_comparison.md))
> or a different feed-forward class.

### Step 1 — Define a `TTNNRMSNormDistributed` wrapper

Before building the decoder, you need a Symbiote norm that handles the
gather-or-norm decision that `DistributedNorm` owns in TT Transformers.

The wrapper should:
1. Accept `mesh_device`, `is_multichip: bool`, `is_distributed: bool`, and a
   `TT_CCL` handle.
2. In `forward`:
   - If `is_multichip` and not `is_distributed`: call
     `ttnn.experimental.all_gather_async(x, dim=3, ...)` using `TT_CCL` semaphore
     handles before invoking the inner norm.
   - Call `TTNNRMSNorm.forward(x)` (or `TTNNDistributedRMSNorm.forward(x)`).
   - If `is_distributed`: call `ttnn.experimental.all_gather_async` on the output.

This matches `DistributedNorm.forward` in `distributed_norm.py`.

### Step 2 — Define a `SwiGLUMLP` module

Implement a `TTNNModule` subclass that:
1. Holds three `TTNNLinear` instances (`gate_proj`, `up_proj`, `down_proj`).
2. In `forward`:
   ```python
   gate = self.gate_proj(x)
   up   = self.up_proj(x)
   hidden = ttnn.mul(gate, up,
                     input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                     memory_config=<L1_sharded_or_DRAM>)
   gate.deallocate(True)
   up.deallocate(True)
   return self.down_proj(hidden)
   ```
   The fused `ttnn.mul` with `input_tensor_a_activations` replicates the single
   kernel used by `MLP.forward` in `models/tt_transformers/tt/mlp.py`.

For multi-chip, add an `tt_all_reduce(out, ...)` call from
`models.tt_transformers.tt.ccl` after `down_proj`, matching the all-reduce that
`MLP.forward` performs on `cluster_axis=0` after the `w2` matmul.

### Step 3 — Define `TTNNLlamaDecoder`

```python
class TTNNLlamaDecoder(TTNNModule):
    def __init__(self, hidden_size, num_heads, intermediate_size,
                 mesh_device, tt_ccl, norm_eps, is_multichip, is_distributed_norm):
        super().__init__()
        self.input_layernorm = TTNNRMSNormDistributed(  # from Step 1
            mesh_device=mesh_device, tt_ccl=tt_ccl,
            is_multichip=is_multichip, is_distributed=is_distributed_norm,
        )
        self.self_attn = LlamaAttention(...)   # existing Symbiote module
        self.post_attention_layernorm = TTNNRMSNormDistributed(
            mesh_device=mesh_device, tt_ccl=tt_ccl,
            is_multichip=is_multichip, is_distributed=is_distributed_norm,
        )
        self.mlp = SwiGLUMLP(...)              # from Step 2

    @classmethod
    def from_torch(cls, hf_layer, mesh_device, tt_ccl, is_multichip, is_distributed_norm):
        obj = cls(...)
        obj.input_layernorm = TTNNRMSNormDistributed.from_torch(
            hf_layer.input_layernorm, mesh_device=mesh_device, tt_ccl=tt_ccl, ...)
        obj.post_attention_layernorm = TTNNRMSNormDistributed.from_torch(
            hf_layer.post_attention_layernorm, ...)
        obj.self_attn = LlamaAttention.from_torch(hf_layer.self_attn, ...)
        obj.mlp = SwiGLUMLP.from_torch(hf_layer.mlp, ...)
        return obj
```

### Step 4 — Wire the forward pass with explicit residuals

```python
def forward(self, x, residual_mem_cfg, mode="decode", ...):
    residual = x

    # Norm → attention (mirrors TransformerBlock lines 232–244)
    attn_in  = self.input_layernorm(x)
    attn_out = self.self_attn(attn_in, ...)

    # Move attention output into residual memory config
    # (mirrors TransformerBlock line 247)
    attn_out = ttnn.to_memory_config(attn_out, residual_mem_cfg)

    # Residual add #1 (mirrors TransformerBlock lines 250–253)
    hidden = ttnn.add(residual, attn_out,
                      memory_config=residual_mem_cfg,
                      dtype=ttnn.bfloat16)
    residual = hidden
    attn_out.deallocate(True)

    # Norm → MLP (mirrors TransformerBlock lines 259–285)
    mlp_in  = self.post_attention_layernorm(hidden)
    mlp_out = self.mlp(mlp_in)

    # Residual add #2 (mirrors TransformerBlock lines 302–309)
    out = ttnn.add(residual, mlp_out,
                   memory_config=residual_mem_cfg,
                   dtype=ttnn.bfloat16)
    return out
```

The `residual_mem_cfg` should be the output of
`args.get_residual_mem_config(mode, prefetcher)` in TT Transformers, which is an L1
width-sharded config for decode and `DRAM_MEMORY_CONFIG` for prefill.

### Step 5 — Weight lifecycle

After constructing the decoder:

```python
decoder.preprocess_weights()       # calls preprocess_weights_impl on all children
decoder.to_device(mesh_device)
decoder.set_device_state(DistributedConfig(mesh_device))
decoder.move_weights_to_device()   # uploads tt_weight / weight_distributed
```

This follows the `TTNNModule` lifecycle defined in `core/module.py`.

---

## 4. Performance gap: residual additions

### 4.1 TT Transformers behavior

In `TransformerBlock.forward` every `ttnn.add` call passes two explicit keyword
arguments:

```python
ttnn.add(residual, attn_out,
         memory_config=skip_mem_cfg,   # explicit L1-sharded or DRAM config
         dtype=ttnn.bfloat16)          # explicit output dtype
```

`skip_mem_cfg` is the value of `args.get_residual_mem_config(mode, prefetcher)`.
For decode mode on a non-TG multi-chip system this typically evaluates to an
L1-sharded memory config.  The result of the add therefore lands **in L1** and is
immediately consumable by the next norm's all-gather without a DRAM round-trip.

For the final residual add the dtype is either `ccl_dtype` (for TG) or
`activation_dtype` from `DecodersPrecision` for the current layer (non-TG).  This
allows per-layer precision control over the accumulator.

### 4.2 Current Symbiote behavior

In the HF-graph-walk approach, the LLaMA `LlamaDecoderLayer.forward` performs
residual additions using Python's `+` operator on `TorchTTNNTensor` objects:
```python
hidden_states = residual + hidden_states   # HF decoder source
```
This is dispatched through the `TorchTTNNTensor.__add__` path, which eventually
calls `ttnn.add` but **without** a `memory_config` or `dtype` argument.  TTNN
selects interleaved DRAM as the output memory location.

Even in a hand-written `TTNNLlamaDecoder`, if the developer writes:
```python
hidden = residual + attn_out
```
the same fate applies: Python operator dispatch will not inject a `memory_config`.
The developer must call `ttnn.add` explicitly (as shown in Step 4 above).

### 4.3 Impact

| Scenario | Residual add output location | Round-trips to DRAM between add and norm |
|----------|-----------------------------|-----------------------------------------|
| TT Transformers, decode | L1 width-sharded (`skip_mem_cfg`) | 0 — norm's all-gather reads directly from L1 |
| Symbiote HF-walk, decode | DRAM interleaved (default) | 1 per residual add — activation must be read back from DRAM before norm |
| Symbiote `TTNNLlamaDecoder` with explicit `ttnn.add` | Matches TT Transformers if `memory_config` is passed | 0 |

For a 8-device T3K decode step with a 4096-dim model, a 32-token batch, and two
residual adds per layer, each DRAM round-trip is approximately `32 * 4096 * 2 bytes`
(bfloat16) = 256 KB per chip.  Over 32 decoder layers, this is ~8 MB per chip per
forward pass of unnecessary DRAM traffic relative to the TT Transformers baseline.

### 4.4 Fix

The fix is mechanical: any Symbiote decoder forward implementation must replace
Python-operator residual adds with explicit `ttnn.add` calls that pass:
- `memory_config`: the residual skip memory config (L1-sharded for decode, DRAM for
  prefill).
- `dtype`: the target activation dtype (at minimum `ttnn.bfloat16`; ideally the
  per-layer value from a `DecodersPrecision`-equivalent config).

No new TTNN op is required; the gap is entirely in how the Python-level forward
method is written.

---

**Next:** [`integration_gaps.md`](./integration_gaps.md)
