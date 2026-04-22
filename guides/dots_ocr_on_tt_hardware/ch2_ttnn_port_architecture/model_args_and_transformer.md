# Model Args, Transformer, Generator, and Weight Loading

This file covers the four components in `tt/` that connect the dots.ocr checkpoint to the `tt_transformers` base infrastructure: `DotsModelArgs`, `DotsTransformer`, `Generator`, and the weight loading functions in `load.py`.

---

## `DotsModelArgs` (`tt/model_config.py`)

`DotsModelArgs` extends `ModelArgs` from `models.tt_transformers.tt.model_config`. Its purpose is to configure the `tt_transformers` base stack for the specific numerical and hardware characteristics of the dots.ocr checkpoint.

### Fixed Overrides — Three Different Initialization Mechanisms

Three fields are fixed for dots.ocr, but each is set through a distinct mechanism during initialization, not as bare class-level attributes:

```python
use_hf_rope = True          # class-level attribute (default)
dummy_weights = False       # forced via kwargs before super().__init__()
trust_remote_code_hf = True # set post-init after super().__init__() returns
```

- `use_hf_rope = True` — set as a class-level attribute. Selects Qwen2-style HF-compatible RoPE computation, matching the RoPE implementation in `reference/rope.py` (`Qwen2RopeHelper`). Setting this to `False` would engage `tt_transformers`'s default RoPE path, which diverges numerically.
- `dummy_weights = False` — forced by injecting `kwargs["dummy_weights"] = False` before calling `super().__init__()`. The base `ModelArgs` supports a dummy-weight mode for fast structural tests; dots.ocr disables this unconditionally so the checkpoint is always loaded from disk. Because `dummy_weights` is controlled through `kwargs` passed to the parent constructor, it cannot be overridden by a caller that omits it.
- `trust_remote_code_hf = True` — set as `self.trust_remote_code_hf = True` after `super().__init__()` returns, because the parent `ModelArgs.__init__` does not accept this field as a constructor kwarg in this version of the repo. dots.ocr registers custom model code in its HuggingFace repository; the base `ModelArgs` passes this flag to `AutoConfig.from_pretrained()` and related HF loading calls. Without it the config load fails with an untrusted-code error.

### `LOCAL_HF_PARAMS` Instance-Level Injection

`tt_transformers`'s `ModelArgs.load_state_dict()` resolves the checkpoint path by looking up a class-level dictionary `LOCAL_HF_PARAMS` keyed on model name. The dots.ocr checkpoint lives at a path unknown to the base class at import time.

`DotsModelArgs.__init__()` resolves this without modifying `tt_transformers` source code by injecting an entry into the class-level dict at instantiation:

```python
ModelArgs.LOCAL_HF_PARAMS[self.model_name] = self.model_path
```

This seeds the lookup before any call to `load_state_dict()`, ensuring the base class can find the dots.ocr config and weights using its standard resolution logic.

### Environment Variables Parsed in `__init__`

Three environment variables are consumed during `DotsModelArgs.__init__()`. None are required; all have defaults inferred from the checkpoint config.

#### `DOTS_MAX_SEQ_LEN`

Caps `max_seq_len`. This is the primary tuning knob for memory-constrained deployments. A lower value reduces KV cache allocation at the cost of context length.

#### `DOTS_MAX_SEQ_LEN_WH_LB`

Legacy alias for `DOTS_MAX_SEQ_LEN`. Honored for backward compatibility with earlier invocations that used the `_WH_LB` suffix. When both are set, `DOTS_MAX_SEQ_LEN` takes precedence.

#### `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE`

Default: `2048`.

This variable caps the column-chunk width used when sharding the LM head across TTNN ops. The motivation is hardware-specific: dots.ocr has a vocabulary of 151,936 tokens. An unsharded LM head matmul on a single Wormhole device would require roughly 300 MB of L1 circular-buffer space per device, exceeding the available budget and causing an L1 overflow error at compile time.

The fix is to split the LM head into multiple TTNN ops, each operating on a column slice of width at most `DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE`. At the default of 2048, the op count depends on the tensor-parallelism (TP) degree:

- **TP=1** — each device holds all 151,936 columns:

$$\text{ops} = \left\lceil \frac{151936}{2048} \right\rceil = 75 \text{ ops per device}$$

- **TP=2 (standard T3K deployment)** — the vocabulary columns are split evenly, so each device holds 151,936 / 2 = 75,968 columns:

$$\text{ops} = \left\lceil \frac{75968}{2048} \right\rceil = 38 \text{ ops per device}$$

Each op fits comfortably in L1. Results are concatenated before the softmax. Raising this value reduces op count at the cost of higher L1 pressure; lowering it is safe but slower.

---

## `DotsTransformer` (`tt/model.py`)

`DotsTransformer` extends `TTTransformer` from `models.tt_transformers.tt.model`. It inherits the full layer stack, paged KV cache management, and all standard TTNN attention and MLP ops without modification. The subclass adds only two targeted overrides.

### `prepare_inputs_prefill()`

The base `TTTransformer.prepare_inputs_prefill()` accepts `[B, S]` integer token ID tensors, looks up embeddings, and returns a `[B, S, D]` activation tensor to pass through the decoder layers.

`DotsTransformer` overrides this method to accept either format:

- `[B, S]` token IDs — standard path; behavior is identical to the base class.
- `[B, S, D]` pre-fused embeddings — dots.ocr-specific path for multimodal inputs. When the input has three dimensions, the method skips the embedding lookup and passes the tensor directly into the decoder stack.

The pre-fused path is used by the full multimodal pipeline. After the vision encoder produces patch embeddings and `reference/fusion.py` (or `tt/fusion.py`) scatters them into the text embedding positions, the result is a `[B, S, D]` tensor on the host. Transferring this tensor directly, rather than re-encoding as token IDs, preserves the continuous vision representations that cannot be expressed as discrete tokens.

### `_prepare_cos_sin()`

The base class's RoPE setup produces cos/sin matrices on the host. `DotsTransformer._prepare_cos_sin()` overrides this to:

1. Expand the host cos/sin matrices to the batch dimension.
2. Replicate the expanded tensors onto every device in the mesh using the mesh device's broadcast strategy.

This is required because Qwen2-style HF-compatible RoPE (`use_hf_rope = True`) produces cos/sin in a format that differs from the default `tt_transformers` RoPE layout, and the mesh replication must be applied after the format conversion.

### `rope_setup_class = None`

`DotsTransformer` sets `rope_setup_class = None`, deferring RoPE configuration to `DotsModelArgs`. This prevents the base class from instantiating its own RoPE helper, which would conflict with the `use_hf_rope = True` path.

### Attention Bias

The dots.ocr text decoder uses attention bias on all four projections (Q, K, V, O). This is atypical: `tt_transformers`'s standard `Attention` class is designed for bias-free projections. `DotsTransformer` uses the standard `Attention` class from `tt_transformers` but relies on `load.py` to map the bias tensors correctly — see the weight loading section below.

---

## `Generator` (`tt/generator.py`)

`Generator` wraps `TTTGenerator` from `models.tt_transformers.tt.generator`. It adds dots.ocr-specific entry points on top of the base generator's token-loop and sampling infrastructure.

### `prefill_forward_text()`

The primary addition is `prefill_forward_text()`, which implements a chunked prefill loop over users. The loop:

1. Iterates over the user batch.
2. For each user, calls `DotsTransformer.prepare_inputs_prefill()` with either token IDs or pre-fused embeddings.
3. Runs the prefill forward pass in chunks of at most `get_max_prefill_chunk_size()` tokens (from `tt/common.py`).
4. Accumulates KV cache entries per user.

Chunking is necessary because very long multimodal sequences (document images can produce thousands of vision tokens) may exceed the TTNN op tile limit if processed as a single prefill call.

### Exposed Properties

`Generator` exposes `.model`, `.model_args`, `.mesh_device`, `.tokenizer`, and `.processor` as properties. The `.processor` property is specific to dots.ocr and not present in the base `TTTGenerator`; it holds the HuggingFace processor used for image preprocessing and token encoding.

---

## Weight Loading (`tt/load.py`)

`tt/load.py` provides two functions that filter and remap the HuggingFace checkpoint into the formats expected by `DotsTransformer` and `VisionEncoder`.

### `load_dots_text_state_dict()`

Loads the text decoder keys from the HF checkpoint. The main tasks are:

1. **Key filtering** — the HF checkpoint stores text and vision parameters under a shared namespace. This function selects only the keys corresponding to the text decoder layers, the embedding table, and the LM head.
2. **Key remapping** — HF parameter names use a different naming convention than `tt_transformers`. The function renames keys to match the names expected by `TTTransformer`'s weight loading logic.
3. **Attention bias handling** — the dots.ocr text decoder has `attention_bias=True`: every Q, K, V, and O projection carries a bias tensor in addition to a weight tensor. The standard `tt_transformers` loader assumes no attention bias. `load_dots_text_state_dict()` maps these bias tensors to the correct TTNN tensor slots so they are included in the loaded state dict without modifying `tt_transformers` source.

### `load_dots_vision_state_dict()`

Loads the vision encoder keys from the HF checkpoint. Tasks are analogous: key filtering to select vision parameters, and key remapping to match the names expected by `VisionEncoder` and its sub-modules (`VisionBlockTT`, `VisionMLPTT`, `VisionAttention`, `PatchEmbedTT`, `vision_rmsnorm`).

---

**Next:** [`pcc_validation_framework.md`](./pcc_validation_framework.md)
