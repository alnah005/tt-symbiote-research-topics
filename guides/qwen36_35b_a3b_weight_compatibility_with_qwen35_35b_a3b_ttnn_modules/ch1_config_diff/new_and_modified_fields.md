# New and Modified Fields in Qwen3.6 Config

## Purpose of This File

This file documents every `config.json` field that was **added or changed** in
`Qwen/Qwen3.6-35B-A3B` relative to `Qwen/Qwen3.5-35B-A3B`. For each change it
covers: what the field means, how HuggingFace code consumes it, and what the
downstream impact is on TT-Symbiote's TTNN module suite. Each section ends with
a pointer to the chapter that performs the full TTNN impact analysis.

## `partial_rotary_factor`: Promoted to Top Level

### What the Field Means

`partial_rotary_factor` is a floating-point scalar in the range `(0, 1]` that
determines what fraction of a head's dimensions participate in rotary position
encoding. When its value is `p`, the first $\lfloor d_{\text{head}} \cdot p \rfloor$
dimensions of each attention head are rotated by the cosine/sine schedule, and
the remaining dimensions are passed through unchanged.

For both Qwen3.5 and Qwen3.6 the value is `0.25`, giving:

$$
d_{\text{rot}} = \lfloor 128 \cdot 0.25 \rfloor = 32
$$

Only 32 of each head's 128 dimensions are rotated. This is a deliberate design
choice to reduce the cost of position encoding in a model that already has long
context via YaRN scaling.

### Where the Value Lives in Each Config

In **Qwen3.5-35B-A3B**, `partial_rotary_factor` exists only inside the nested
`rope_parameters` sub-object:

```json
{
  "rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    ...
  },
  "rope_parameters": {
    "partial_rotary_factor": 0.25,
    "rope_type": "yarn"
  }
}
```

There is **no top-level key** named `partial_rotary_factor` in the Qwen3.5
root object.

In **Qwen3.6-35B-A3B**, the same value also appears at the root of
`config.json`:

```json
{
  "partial_rotary_factor": 0.25,
  "rope_scaling": { ... },
  "rope_parameters": {
    "partial_rotary_factor": 0.25,
    "rope_type": "yarn"
  }
}
```

Both locations carry `0.25`. The root-level entry is a promotion, not a
replacement.

### How `AutoConfig` Populates the Python Attribute

`AutoConfig.from_pretrained` delegates to `Qwen3_5MoeConfig.__init__`, which
is a subclass of `PretrainedConfig`. `PretrainedConfig.__init__` assigns every
key in `config.json` that is not explicitly consumed by `__init__` parameters
as a direct attribute on the config object via `setattr(self, key, value)`.

The practical outcome:

- For a **Qwen3.5 config**: `config.partial_rotary_factor` raises `AttributeError`
  because no top-level key exists. Code must read
  `config.rope_parameters["partial_rotary_factor"]` or
  `config.rope_scaling.get("partial_rotary_factor")`.
- For a **Qwen3.6 config**: `config.partial_rotary_factor` resolves to `0.25`
  because the root-level key is assigned as a top-level attribute. The nested
  value in `rope_parameters` also remains accessible.

There is no override conflict: the nested `rope_parameters` dict is stored as
`config.rope_parameters` (a dict attribute), and the top-level
`partial_rotary_factor` is stored as `config.partial_rotary_factor` (a float
attribute). They are separate attributes that happen to agree in value.

### TTNN Impact

The numeric value of `partial_rotary_factor` does not change. The promotion is
a defensive addition that allows consumer code to read `config.partial_rotary_factor`
directly without parsing the nested dict. The risk is that any TT-Symbiote code
path that reads `config.partial_rotary_factor` (top-level attribute form) would
silently succeed on Qwen3.6 but raise `AttributeError` on Qwen3.5. See
[Chapter 3](../ch3_partial_rotary_factor/index.md) for the full analysis of how
`TTNNRotaryPositionEmbedding` resolves this attribute and whether a compatibility
fallback is needed.

**Risk level: low** (value is unchanged; concern is attribute-lookup path only).

## `bos_token_id`: Added With Value 248044

### What the Field Means

`bos_token_id` is an integer token ID that identifies the beginning-of-sequence
(BOS) token in the model's vocabulary. When present in `config.json`, it is
read by:

1. `AutoConfig.from_pretrained`, which stores it as `config.bos_token_id`.
2. `AutoTokenizer.from_pretrained`, which uses it to set `tokenizer.bos_token_id`
   if the tokenizer's own vocabulary does not already declare a BOS token.
3. `GenerationMixin.generate`, which may prepend a BOS token to `input_ids`
   if the prompt does not already begin with it.

In **Qwen3.5-35B-A3B**, this field is **absent** from `config.json`. The
HuggingFace default when `bos_token_id` is missing from the config is `None`.
`GenerationMixin.generate` does not prepend a BOS token when
`config.bos_token_id is None` unless the user's `GenerationConfig` explicitly
sets one.

In **Qwen3.6-35B-A3B**, `bos_token_id` is set to `248044`.

### The Token ID Value

`248044` is significant because it lies **outside** the Qwen3.5/3.6 shared
vocabulary. The `vocab_size` field is `151936` for both checkpoints, meaning
valid token IDs are in the range `[0, 151935]`. Token ID `248044` is
out-of-range for the embedding table.

This is not an error in the config. Qwen3.6's tokenizer vocabulary is larger
than `151936` in the full tiktoken vocabulary file (which includes special tokens
beyond the base vocab), and `248044` falls in that extended range. To confirm the
exact token string this ID maps to, inspect the Qwen3.6 tiktoken vocabulary file
directly; `248044` does not correspond to any standard Qwen2/Qwen3 special token
at the known IDs (for example, `<|im_start|>` is token ID `151644` in standard
Qwen2/Qwen3 tiktoken vocabularies, which differs from `248044` by approximately
96,400). The embedding table indexed by `vocab_size = 151936` in the weight file
covers base tokens; special tokens beyond this range are typically handled by an
extended embedding or are injected at the tokenizer boundary rather than looked
up in the weight table.

The practical consequence is: if code naively calls
`model.embed_tokens(torch.tensor([248044]))` against the `[151936, 7168]`
embedding table, it will raise an index-out-of-bounds error. Correctly written
generation code that uses the tokenizer's special-token handling will never
reach this code path.

### How `GenerationMixin.generate` Consumes `bos_token_id`

The relevant code path in `transformers.GenerationMixin.generate` is:

```python
# Simplified from transformers/generation/utils.py
if input_ids is None:
    # No input provided: synthesise a BOS-only sequence
    input_ids = torch.full(
        (batch_size, 1),
        self.generation_config.bos_token_id,
        dtype=torch.long,
        device=device,
    )
```

`GenerationConfig.bos_token_id` is populated from `model.config.bos_token_id`
when no explicit `GenerationConfig` is passed. This code path is only triggered
when the caller passes `input_ids=None` to `generate`. In all normal usage
patterns — where the caller tokenizes a prompt and passes the resulting
`input_ids` — this branch is never taken.

When `input_ids` is already provided, `generate` does **not** automatically
prepend a BOS token. The `bos_token_id` field is then used only for
`forced_bos_token_id` logic (if set) and for stopping-condition checks.
Adding `bos_token_id: 248044` to Qwen3.6 enables the `input_ids=None`
code path without crashing and documents the canonical BOS token for this
checkpoint's tokenizer.

### TTNN Impact

`bos_token_id` does not enter any TTNN tensor unless the generation loop calls
`generate(input_ids=None)`, which TT-Symbiote does not do. TT-Symbiote's
generation loop always constructs `input_ids` from a tokenizer call before
invoking the model. The value `248044` never reaches
`model.embed_tokens` as an index in normal inference.

See [Chapter 4](../ch4_bos_token_id/index.md) for the full analysis of the
generation loop initialization and the out-of-range token ID edge case.

**Risk level: low** (affects tokenizer initialization and `input_ids=None`
generation only; no TTNN tensor impact in the standard path).

## `mtp_num_hidden_layers`: Added With Value 1

### What Multi-Token Prediction Is

Multi-Token Prediction (MTP) is an auxiliary training objective where the model
is trained to predict not just the next token at position $t+1$ but also tokens
at positions $t+2, t+3, \ldots$ simultaneously. This is achieved by attaching
one or more lightweight prediction heads to the backbone. Each MTP head takes
the final hidden states from the main decoder and applies an additional
transformer layer (or set of layers) to project toward a further-future token.

The field `mtp_num_hidden_layers: 1` declares that **one** such auxiliary MTP
transformer layer is present in the Qwen3.6 checkpoint. The value `1` means the
MTP head contains a single hidden layer (one self-attention block plus one FFN
block) rather than being a pure linear projection.

### How the Field Is Declared in `config.json`

In Qwen3.6, the root of `config.json` contains:

```json
{
  "mtp_num_hidden_layers": 1
}
```

This is a top-level integer field. It is not nested inside any sub-object. There
is no corresponding `use_mtp` or `mtp_enabled` boolean flag; the presence of a
non-zero integer value implies the MTP head exists.

### Whether `Qwen3_5MoeForConditionalGeneration` Instantiates the MTP Head

As of the `transformers` versions used for both Qwen3.5 and Qwen3.6 release
(4.48--4.51 range), the class `Qwen3_5MoeForConditionalGeneration` does
**not** contain MTP head instantiation logic in its `__init__` method. The
`mtp_num_hidden_layers` field is read from the config by `AutoConfig` and stored
as `config.mtp_num_hidden_layers`, but the main model class does not branch on
this value to create an additional `nn.Module`.

The MTP head weights — stored in the Qwen3.6 safetensors under a key prefix
such as `model.future_hidden_states_norm.*` or `model.mtp_*` — are present in
the checkpoint file but have no corresponding `nn.Module` parameter in the
standard `Qwen3_5MoeForConditionalGeneration` instance. When
`AutoModelForCausalLM.from_pretrained` loads the checkpoint, these keys will
appear in the "unexpected keys" list returned by `model.load_state_dict`.

By default, `load_state_dict` with `strict=False` (which `from_pretrained` uses)
will silently skip unexpected keys and log them at the `WARNING` level. The MTP
weights are therefore **not loaded into any `nn.Parameter`** in the standard
model instance and do not participate in the forward pass.

### Weight Keys Introduced by the MTP Head

Based on the architecture declaration (`mtp_num_hidden_layers: 1`) and the
Qwen3.5 backbone hyperparameters, the MTP head introduces weight keys under a
dedicated prefix. The exact prefix depends on the implementation, but the
expected pattern is:

```
model.future_hidden_states_norm.weight        # [7168]  RMSNorm over hidden_size
model.mtp_head.0.attn.q_proj.weight           # [8192, 7168]  same shape as backbone
model.mtp_head.0.attn.k_proj.weight           # [512, 7168]
model.mtp_head.0.attn.v_proj.weight           # [512, 7168]
model.mtp_head.0.attn.o_proj.weight           # [7168, 8192]
model.mtp_head.0.mlp.gate_proj.weight         # [14336, 7168]  (intermediate_size; MTP head uses dense FFN)
model.mtp_head.0.mlp.up_proj.weight           # [14336, 7168]
model.mtp_head.0.mlp.down_proj.weight         # [7168, 14336]
model.mtp_head.0.input_layernorm.weight       # [7168]
model.mtp_head.0.post_attention_layernorm.weight # [7168]
```

All shapes are derived from the same hyperparameters as the backbone
(`hidden_size = 7168`, `num_attention_heads = 64`, `head_dim = 128`,
`num_key_value_heads = 4`). The MTP layer is architecturally a copy of one
backbone decoder layer. Crucially, the MTP head uses a **dense** single-expert
FFN with `intermediate_size = 14336`, not a MoE FFN; the `moe_intermediate_size
= 2048` value applies only to the routed experts in the backbone MoE layers and
does **not** apply here.

The total additional parameter count is approximately:

$$
\underbrace{2 \cdot 8192 \cdot 7168}_{\text{q, o\_proj}} +
\underbrace{2 \cdot 512 \cdot 7168}_{\text{k, v\_proj}} +
\underbrace{3 \cdot 14336 \cdot 7168}_{\text{gate, up, down}} +
\underbrace{4 \cdot 7168}_{\text{norms}}
\approx 433\text{M parameters}
$$

These parameters are present in the safetensors but, as discussed above, are
not loaded into the standard model instance and do not reach any TTNN device
tensor through the existing weight preprocessing pipeline.

### TTNN Impact

The MTP weight keys use a distinct prefix that does not match any weight key
consumed by `TTNNQwen3FullAttention`, `TTNNQwen3LinearAttention`, or
`TTNNQwen3MoE`. TT-Symbiote's weight preprocessing hooks are keyed on weight
names; MTP-prefixed keys will be encountered as unknown keys and either silently
ignored or surfaced as warnings depending on the error policy of the loading
function.

See [Chapter 5](../ch5_mtp_head/index.md) for the complete weight key inventory,
the `from_pretrained` loading scenarios, and the recommended safe-loading recipe
for ensuring MTP keys do not reach TTNN device buffers.

**Risk level: low** (MTP head is inference-inactive in standard decode; weight
keys do not conflict with backbone names; loading path concern is manageable
with a key filter).

## Other Fields: No Additional Changes

Beyond the three fields documented above, no other fields in `config.json` are
known to differ between Qwen3.5-35B-A3B and Qwen3.6-35B-A3B. The following
fields that sometimes differ across model generations have been confirmed to be
identical:

| Field | Value (both) | Notes |
|---|---|---|
| `tie_word_embeddings` | `false` | LM head is a separate weight tensor in both versions |
| `attention_bias` | `false` | No bias in any projection in either version |
| `temperature` | _(absent or `1.0`)_ | Not a `config.json` field; belongs in `generation_config.json` |
| `top_p` | _(absent or `1.0`)_ | Not a `config.json` field; belongs in `generation_config.json` |
| `use_cache` | `true` | KV cache enabled in both |
| `output_router_logits` | `false` | MoE router logits not returned in standard forward |

Generation defaults such as `temperature` and `top_p` are stored in a separate
`generation_config.json` file alongside the checkpoint, not in `config.json`
itself. Changes to those files affect sampling behaviour but have no TTNN module
impact.

---

**Next:** [Chapter 2 — Weight Tensor Shape Analysis](../ch2_weight_shapes/index.md)
