# Plan: Qwen3.6-35B-A3B Weight Compatibility with Qwen3.5-35B-A3B TTNN Modules

## Audience

This guide targets ML systems engineers working on TT-Symbiote who need to
validate or extend the existing Qwen3.5-35B-A3B TTNN module suite
(`TTNNQwen3FullAttention`, `TTNNQwen3LinearAttention`, `TTNNQwen3MoE`, and
related submodules) to run correctly with Qwen3.6-35B-A3B checkpoints.

**Assumed knowledge:**
- Familiarity with the Qwen3.5-35B-A3B model architecture (40-layer hybrid
  MoE, 30 linear-attention layers + 10 full-attention layers, 256 routed
  experts + 1 shared expert, `partial_rotary_factor=0.25`)
- Working knowledge of TT-Symbiote's `TTNNModule` authoring pattern
  (module replacement, `forward` signatures, weight preprocessing hooks)
- Understanding of HuggingFace `AutoModelForCausalLM.from_pretrained` loading
  mechanics and the `config.json` / `model.safetensors.index.json` file layout
- Familiarity with TTNN tensor dtypes (`bfloat16`, `bfloat8_b`, `bfloat4_b`,
  `float32`), memory configs (DRAM interleaved, L1, sharded), and the
  `TTNNRotaryPositionEmbedding` / `TTNNDistributedRotaryPositionEmbedding`
  interface
- Basic understanding of the HuggingFace `rope_scaling` / `rope_parameters`
  configuration pattern and how `AutoConfig` resolves rotary dimensions

**Not assumed:**
- Prior exposure to Qwen3.6 config specifics (the `partial_rotary_factor`
  promotion, `bos_token_id: 248044`, or `mtp_num_hidden_layers: 1` field)
- Knowledge of the Multi-Token Prediction (MTP) architecture or whether it is
  active during standard autoregressive decoding
- Understanding of how `AutoModelForCausalLM.from_pretrained` responds to
  unexpected weight keys when `ignore_mismatched_sizes` and
  `_keys_to_ignore_on_load_missing` are or are not set

---

## Chapter List

---

### Chapter 1 --- Config Diff: Qwen3.6 vs Qwen3.5

**Description:** Performs a field-by-field comparison of the two models'
`config.json` files, identifies every changed, added, and removed field, and
explains what each change means before any TTNN impact is analysed.

**Directory:** `ch1_config_diff/`

**Files:**

- `index.md`
  - Chapter overview and reading order
  - Quick-reference diff table: one row per changed/added/removed field,
    columns for Qwen3.5 value, Qwen3.6 value, and which chapter analyses
    the TTNN impact

- `structural_fields.md`
  - Fields that are identical between the two versions: architecture class
    (`Qwen3_5MoeForConditionalGeneration`), model type (`qwen3_5_moe`),
    `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
    `num_key_value_heads`, `intermediate_size`, `moe_intermediate_size`,
    `num_experts_per_tok`, `num_experts`, `shared_expert_intermediate_size`,
    `head_dim`, `vocab_size`, `rope_theta`, and all `rope_scaling` /
    `rope_parameters` sub-fields
  - Why identical structural fields mean all existing weight tensor shapes
    are preserved and no TTNN matmul program configs need to change
  - Fields that changed numerically but do not affect tensor shapes:
    `hidden_act`, `initializer_range`, `rms_norm_eps` (if any)

- `new_and_modified_fields.md`
  - `partial_rotary_factor: 0.25` promoted to the top level of `config.json`
    (in Qwen3.5 this value exists only inside `rope_parameters`): what the
    field means, which Python attribute `AutoConfig` populates it into, and
    whether top-level presence overrides or duplicates the nested value
  - `bos_token_id: 248044` added (absent in Qwen3.5): the numerical value,
    how `AutoModelForCausalLM.generate` and `AutoTokenizer` consume this
    field, and what happens if it is absent (Qwen3.5 behaviour)
  - `mtp_num_hidden_layers: 1` added: what the Multi-Token Prediction (MTP)
    head is at the architecture level, how it is declared in the config, and
    whether HuggingFace's `Qwen3_5MoeForConditionalGeneration` class
    instantiates it or ignores it during standard decoding
  - Any other fields added or changed (e.g., `tie_word_embeddings`,
    `attention_bias`, or generation defaults like `temperature`,
    `top_p`)

---

### Chapter 2 --- Weight Tensor Shape Analysis

**Description:** Derives the full weight key inventory and tensor shape table
for both checkpoints, confirms that every shape consumed by the existing TTNN
modules is identical, and documents any extra keys present in the Qwen3.6
checkpoint.

**Directory:** `ch2_weight_shapes/`

**Files:**

- `index.md`
  - Chapter overview
  - Summary finding: shape-compatible or shape-incompatible, with a one-line
    rationale
  - Pointer to the authoritative shape tables in the two files below

- `shared_weight_shapes.md`
  - Complete enumeration of all weight keys consumed by the existing TTNN
    modules, grouped by module:
    - `TTNNQwen3LinearAttention` (linear attention layers 0--29): all
      in_proj, out_proj, conv weight, norm, gate, A_log, dt_bias keys with
      their shapes expressed in terms of model hyperparameters and their
      concrete values for the 35B-A3B config
    - `TTNNQwen3FullAttention` (full attention layers 30--39): `q_proj`,
      `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm` weights and their
      shapes; gate weight extracted during conversion
    - `TTNNQwen3MoE` (all 40 layers): router weight, 256 x gate_up_proj,
      256 x down_proj, shared expert gate_up, shared expert down, shared
      expert gate scalar
    - Norms: attention_norm, ffn_norm per layer; final norm
    - Embeddings: token embedding table, LM head (tied or untied)
  - Confirmation that all shapes are numerically identical between the two
    checkpoints given that hyperparameters governing them are unchanged
  - How to verify: `safetensors.torch.load_file` + shape comparison script
    pseudocode

- `extra_weight_keys.md`
  - Identification of weight keys present in the Qwen3.6 safetensors but
    absent in Qwen3.5 — specifically MTP head weights under a prefix such
    as `model.mtp_head.*` or `model.future_prediction_head.*`
  - Exact shapes of each MTP key (norm weight, projection weight, embedding
    tie status) derived from `mtp_num_hidden_layers: 1` and the MTP
    architecture specification
  - How `AutoModelForCausalLM.from_pretrained` handles unexpected keys by
    default: whether it raises an error, emits a warning, or silently skips
    them; the `ignore_mismatched_sizes` and `_keys_to_ignore_on_load_*`
    class attributes in `Qwen3_5MoeForConditionalGeneration`
  - Impact on TT-Symbiote's weight loading path: whether the extra MTP keys
    reach the TTNN module weight preprocessing hooks or are discarded before
    that point; what a safe loading recipe looks like

---

### Chapter 3 --- `partial_rotary_factor` Promotion and RoPE Resolution

**Description:** Analyses the single most structurally ambiguous config
change -- `partial_rotary_factor: 0.25` moving from nested
`rope_parameters` to the top-level config -- and traces how HuggingFace
resolves the rotary dimension at every layer of the stack, then maps the
outcome to `TTNNRotaryPositionEmbedding`.

**Directory:** `ch3_partial_rotary_factor/`

**Files:**

- `index.md`
  - Chapter overview
  - Finding summary: does the promotion change the resolved `rotary_dim`
    value seen by any code path, or is it a no-op redundancy?

- `hf_config_resolution.md`
  - How `AutoConfig` populates `partial_rotary_factor` from `config.json`:
    the `PretrainedConfig.__init__` attribute assignment path, the
    `Qwen3_5MoeConfig.__init__` init order, and which value wins when the
    same logical field exists at both the top level and inside a nested
    `rope_parameters` dict
  - The `rope_scaling` / `rope_parameters` sub-object: its fields (`type`,
    `rope_type`, `partial_rotary_factor`, `mrope_section`), which fields
    `AutoConfig` promotes to top-level attributes, and precedence rules
  - The exact Python expression used by `Qwen3_5MoeForConditionalGeneration`
    (and its base classes) to compute `rotary_dim` or `rope_dim`: whether
    it reads `config.partial_rotary_factor`, `config.rope_scaling.partial_rotary_factor`,
    or a derived field like `config.rope_dim`
  - Whether `transformers >= 4.51` (the version bundled with Qwen3.6) changed
    this resolution logic relative to the version used for Qwen3.5

- `ttnn_rope_impact.md`
  - How `TTNNRotaryPositionEmbedding` reads `partial_rotary_factor` from the
    model config to compute `rotary_dim = head_dim * partial_rotary_factor`:
    the attribute name it consults (`model_config.rope_dim`,
    `model_args.partial_rotary_factor`, or a constructor argument)
  - Whether the promotion of `partial_rotary_factor` to the top level
    changes the value of `rotary_dim` seen by `TTNNRotaryPositionEmbedding`
    when the module is constructed with a Qwen3.6 config vs a Qwen3.5 config
  - The concrete computed value in both cases: `rotary_dim = int(256 * 0.25)
    = 64`; confirming it is identical means cos/sin table shapes are
    unchanged and no RoPE code changes are needed
  - Edge case: what happens if a consumer code path reads
    `config.partial_rotary_factor` (top-level) and the Qwen3.5 config does
    not have this top-level field -- does it fall back gracefully or raise
    `AttributeError`; whether the Qwen3.6 promotion is thus a defensive fix
    for code paths that read the top-level attribute directly

---

### Chapter 4 --- `bos_token_id` and Generation Loop Initialization

**Description:** Traces the effect of the new `bos_token_id: 248044` field
through HuggingFace generation utilities and TT-Symbiote's generation loop,
establishing whether it changes any tensor content or control flow.

**Directory:** `ch4_bos_token_id/`

**Files:**

- `index.md`
  - Chapter overview
  - Finding summary: does `bos_token_id` affect any input tensor to the
    TTNN forward pass?

- `hf_generation_usage.md`
  - How `GenerationMixin.generate` consumes `bos_token_id`: the code path
    that prepends BOS when `input_ids` does not already start with it,
    the `GenerationConfig` field and its precedence over `model.config.bos_token_id`
  - Tokenizer-side: how `AutoTokenizer` for `Qwen/Qwen3.6-35B-A3B` sets
    `tokenizer.bos_token` and `tokenizer.bos_token_id`; whether Qwen3.5's
    tokenizer already had an implicit BOS that users were manually prepending
  - Concrete scenario analysis: if a user passes a prompt already tokenized
    without BOS, does Qwen3.6's `bos_token_id` cause the model to receive a
    different `input_ids` tensor than Qwen3.5 would have?

- `tt_symbiote_generation_loop.md`
  - Where TT-Symbiote's generation loop initializes the first input token
    tensor: whether it respects `model.config.bos_token_id` or reads
    BOS from the tokenizer directly
  - Whether the KV cache initialization, paged KV cache page table, or
    position ID tensor for step 0 is affected by `bos_token_id`
  - Safe recipe: how to ensure the Qwen3.6 generation loop initializes
    identically to Qwen3.5 (e.g., always pass pre-tokenized `input_ids`
    rather than relying on auto-prepend)
  - Whether the token ID value `248044` itself lands in any TTNN tensor
    at inference time and whether its embedding lookup is in-range for the
    existing embedding table (vocab_size = 151,936 vs 248,044 -- flag if
    this is out of range)

---

### Chapter 5 --- MTP Head: Weight Loading and Inference Impact

**Description:** Establishes whether the Multi-Token Prediction head
(`mtp_num_hidden_layers: 1`) is a training-only artefact or an inference-
active module, enumerates its weight keys, and provides a definitive safe-
loading recipe for TT-Symbiote.

**Directory:** `ch5_mtp_head/`

**Files:**

- `index.md`
  - Chapter overview
  - Finding summary: is MTP active during `AutoModelForCausalLM` standard
    decode, and does it require any changes to TT-Symbiote's loading path?

- `mtp_architecture.md`
  - What Multi-Token Prediction is: a secondary prediction head trained to
    predict tokens at positions `t+2`, `t+3`, ... using shared backbone
    representations; how `mtp_num_hidden_layers: 1` parametrises it
  - The MTP head module structure in `Qwen3_5MoeForConditionalGeneration`:
    whether it exists as a separate `nn.Module` submodule in the class or
    is defined separately and attached conditionally
  - Whether `forward` is called during `generate` in autoregressive mode:
    the standard decode call path through `Qwen3_5MoeForConditionalGeneration.forward`
    and whether the MTP head's forward is invoked or bypassed
  - Weight keys introduced by the MTP head: prefix pattern, shapes of each
    tensor (norm weight, projection weight, embedding reference), and
    total parameter count contribution
  - Training-only vs inference status: whether the MTP head is guarded by
    a `use_mtp` flag or `training` mode check, and what the default
    behaviour is when `model.eval()` is set

- `loading_recipe.md`
  - The three loading scenarios and their outcomes:
    1. `AutoModelForCausalLM.from_pretrained` with default settings: does it
       error on unexpected MTP keys, or load them into the model silently?
    2. `AutoModelForCausalLM.from_pretrained` with `ignore_mismatched_sizes=True`:
       behaviour change if any
    3. Direct safetensors load + manual state dict filtering: how to strip
       MTP keys before passing the state dict to TT-Symbiote's weight
       preprocessing pipeline
  - TT-Symbiote weight preprocessing impact: whether the existing TTNN module
    weight hooks consume any MTP-prefixed keys and whether unexpected keys
    cause errors or are silently passed through
  - Recommended safe loading recipe: the minimal change needed (if any) to
    load a Qwen3.6 checkpoint into the existing Qwen3.5 TTNN module suite
    without errors or silent weight corruption
  - Validation step: how to confirm no MTP weights leaked into TTNN device
    tensors by inspecting the loaded parameter inventory

---

### Chapter 6 --- End-to-End Compatibility Verdict and Migration Guide

**Description:** Synthesises findings from all prior chapters into a definitive
compatibility verdict, documents any required code changes, and provides a
step-by-step migration guide for switching from a Qwen3.5 to a Qwen3.6
checkpoint in TT-Symbiote.

**Directory:** `ch6_migration_guide/`

**Files:**

- `index.md`
  - Chapter overview
  - Master compatibility table: one row per research question, columns for
    finding, risk level (none / low / medium / high), and required action
  - Reading order for this chapter

- `compatibility_verdict.md`
  - Weight tensor shapes: verdict and evidence (all shapes identical because
    governing hyperparameters are unchanged)
  - `partial_rotary_factor` promotion: verdict and evidence (no-op or
    requires defensive attribute fallback)
  - `bos_token_id` addition: verdict and evidence (tokenizer/generation-side
    only, no TTNN tensor impact if `input_ids` are pre-formed)
  - MTP head weight keys: verdict and evidence (loading-path concern,
    inference-inactive; filtered before TTNN modules see the state dict)
  - Overall verdict: can the existing TTNN module suite run Qwen3.6 weights
    without modification, with minor config changes, or with code changes?

- `migration_steps.md`
  - Step 1 --- Update the checkpoint path: point TT-Symbiote's model loader
    at `Qwen/Qwen3.6-35B-A3B` instead of `Qwen/Qwen3.5-35B-A3B`
  - Step 2 --- Handle `partial_rotary_factor` attribute lookup: if any
    TT-Symbiote code reads `config.partial_rotary_factor` directly (top-level),
    verify it now resolves correctly for both checkpoints; add a
    `getattr(config, "partial_rotary_factor", config.rope_parameters.get(...))`
    fallback if needed for backward compatibility with Qwen3.5
  - Step 3 --- Strip or ignore MTP weight keys: add a key filter in the
    weight preprocessing pipeline to exclude keys matching the MTP prefix;
    document the exact filter predicate and where to insert it
  - Step 4 --- Tokenizer and generation config: update `bos_token_id` in
    TT-Symbiote's generation loop config if it reads from `model.config`;
    verify the first-token embedding index is within the embedding table bounds
  - Step 5 --- Numerical validation: run the existing PCC test suite
    (`test_pcc.py` / `test_a3b_pcc.py`) against the Qwen3.6 checkpoint;
    expected PCC thresholds are identical to Qwen3.5 since architecture is
    unchanged
  - Step 6 --- End-to-end generation smoke test: run `demo_a3b.py` with a
    known prompt and compare output token sequence and throughput numbers
    (86 ms/token target) against the Qwen3.5 baseline

---

## Conventions

### Terminology

| Term | Definition |
|------|------------|
| Qwen3.5 | `Qwen/Qwen3.5-35B-A3B` checkpoint; the baseline against which Qwen3.6 is compared |
| Qwen3.6 | `Qwen/Qwen3.6-35B-A3B` checkpoint; the new separately-trained weights |
| Architecture class | `Qwen3_5MoeForConditionalGeneration` -- the HuggingFace PyTorch module class shared by both versions |
| Model type | `qwen3_5_moe` -- the `config.json` `model_type` field shared by both versions |
| `rotary_dim` | The number of head dimensions that receive RoPE encoding: `int(head_dim * partial_rotary_factor)` = 64 for both versions |
| MTP head | Multi-Token Prediction head; the auxiliary prediction module declared by `mtp_num_hidden_layers: 1` in Qwen3.6 |
| TTNN module suite | The collection of TT-Symbiote modules targeting Qwen3.5-35B-A3B: `TTNNQwen3FullAttention`, `TTNNQwen3LinearAttention`, `TTNNQwen3MoE`, and related helpers |
| Weight preprocessing pipeline | TT-Symbiote's weight loading path that converts HuggingFace safetensors into TTNN device tensors, including dtype casting, sharding, and key renaming |
| `from_pretrained` | `AutoModelForCausalLM.from_pretrained` or `AutoConfig.from_pretrained` unless otherwise qualified |
| PCC | Pearson Correlation Coefficient; the numerical agreement metric used in the existing Qwen3.5 test suite (threshold ≥ 0.99) |
| BOS | Beginning-of-sequence token; `bos_token_id = 248044` in Qwen3.6 |

### Notation

- Tensor shapes use square bracket notation with named dimensions:
  `[B, H, S, D]` for activations; `[out_features, in_features]` for weight
  matrices (HuggingFace convention).
- Config field names are written in `code font` matching the exact key in
  `config.json`, e.g., `partial_rotary_factor`, `mtp_num_hidden_layers`.
- Weight key names use the HuggingFace checkpoint prefix convention:
  `model.layers.{i}.self_attn.q_proj.weight`.
- TTNN dtype abbreviations: BF16 = `ttnn.bfloat16`; BFP8 = `ttnn.bfloat8_b`;
  BFP4 = `ttnn.bfloat4_b`; FP32 = `ttnn.float32`.
- "top-level config field" means a key at the root of `config.json`, as
  opposed to a key nested inside `rope_scaling`, `rope_parameters`, or another
  sub-object.

### Formatting Rules

- Every file begins with a `# Title` H1 header matching the file's topic.
- Section headers use `##` (H2) and `###` (H3); no deeper nesting.
- All findings tables use GitHub-flavored Markdown pipe syntax.
- Code samples (config snippets, pseudocode, shell commands) use fenced code
  blocks with an appropriate language tag (`json`, `python`, `bash`).
- Cross-chapter references use relative paths:
  `../ch1_config_diff/new_and_modified_fields.md`.
- No external URLs in the body text; all findings are self-contained.
- Risk levels in compatibility tables are one of: **none**, **low**,
  **medium**, or **high**.

---

## Cross-Chapter Dependencies

```
Ch1 (Config Diff)
  ├── Ch2 (Weight Shapes)            — uses structural field list from Ch1 to
  │                                    confirm which hyperparameters govern shapes
  ├── Ch3 (partial_rotary_factor)    — uses the promotion finding from Ch1's
  │                                    new_and_modified_fields.md as its entry point
  ├── Ch4 (bos_token_id)             — uses the bos_token_id addition from Ch1
  │                                    as its entry point
  └── Ch5 (MTP Head)                 — uses mtp_num_hidden_layers finding from Ch1
                                       as its entry point
Ch2 ─────────────────────────────────┐
Ch3 ─────────────────────────────────┤
Ch4 ─────────────────────────────────┤
Ch5 ─────────────────────────────────┘
                                      ▼
                             Ch6 (Migration Guide) — synthesises verdicts from
                                                     Chapters 2--5 into the
                                                     compatibility table and
                                                     step-by-step migration
```

**Explicit dependencies by chapter:**

- **Chapter 2** requires: the list of unchanged structural hyperparameters from
  Chapter 1 (`structural_fields.md`) to establish which weight shapes are
  governed by identical values.
- **Chapter 3** requires: the `partial_rotary_factor` promotion finding from
  Chapter 1 (`new_and_modified_fields.md`); no dependency on Chapter 2.
- **Chapter 4** requires: the `bos_token_id` addition from Chapter 1
  (`new_and_modified_fields.md`); no dependency on Chapters 2 or 3.
- **Chapter 5** requires: the `mtp_num_hidden_layers` addition from Chapter 1
  (`new_and_modified_fields.md`); the extra-key analysis in Chapter 2
  (`extra_weight_keys.md`) for the weight key inventory.
- **Chapter 6** requires: the verdicts from all of Chapters 2, 3, 4, and 5;
  readers must complete those chapters before the migration guide is meaningful.

**Research questions addressed by chapter:**

| Research Question | Primary Chapter | Supporting Chapter |
|-------------------|-----------------|--------------------|
| Weight tensor shape differences causing loading failures | Ch 2 | Ch 1 |
| `partial_rotary_factor` promotion affecting `TTNNRotaryPositionEmbedding` | Ch 3 | Ch 1 |
| `bos_token_id: 248044` affecting tokenizer / generation loop | Ch 4 | Ch 1 |
| `mtp_num_hidden_layers: 1` adding keys that interfere with loading | Ch 5 | Ch 2 |
