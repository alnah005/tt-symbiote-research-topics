# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 1

## Issues

**1. [symbiote_attention_overview.md], line ~255 — K/V cache tensors incorrectly attributed ReplicateTensorToMesh**

The guide stated "All cache tensors are allocated as `ttnn.bfloat16`, `TILE_LAYOUT`, `DRAM_MEMORY_CONFIG`. On multi-device meshes, `ttnn.ReplicateTensorToMesh` is used."

The source (`attention.py`, `to_device`) allocates K/V cache tensors via `ttnn.zeros` with no `mesh_mapper` argument at all. `ReplicateTensorToMesh` is only applied to `_tt_page_table` (via `ttnn.from_torch`). A reader implementing a multi-device cache from the guide would incorrectly apply a mesh mapper to every K/V tensor.

Fix applied: Clarified that `ttnn.zeros` is used for K/V tensors (no mesh mapper), and `ReplicateTensorToMesh` is only used for the page table.

---

**2. [symbiote_attention_overview.md], line ~195 — TTNNDistributedRoPE transformation matrix cache description contradicts source**

The guide stated the matrix "is built once per `is_decode` flag (always `False` in the current implementation — `is_decode_mode = False` is hardcoded)."

The source (`rope.py`, `move_weights_to_device_impl`) iterates `for is_decode in [True, False]` and populates `self._trans_mat_cache` with two entries. The parenthetical "always `False`" describes the `forward` method's behavior, not `move_weights_to_device_impl`. Conflating the two misrepresents what is built vs. what is used.

Fix applied: Separated the two concerns — the cache holds both `True` and `False` entries; `forward` always retrieves `False` because `is_decode_mode = False` is hardcoded there.

---

**3. [symbiote_attention_overview.md], line ~150 — LlamaAttention.forward missing output projection step**

The guide's numbered forward steps ended at step 7: "Calls `ttnn.experimental.nlp_concat_heads` and squeezes." The source (`attention.py`, line 1007) returns `self.o_proj(attn_out), None` — the output projection is the final computation step. A reader implementing `LlamaAttention.forward` from the guide would omit the output projection entirely, producing wrong outputs.

Fix applied: Added step 8 describing the `self.o_proj` call and the `(attn_out, None)` return value.

---

## Change Log — Pass 1 fixes applied

- Fix 1: `symbiote_attention_overview.md` — Corrected `to_device` description to state that K/V cache tensors use `ttnn.zeros` with no mesh mapper; `ReplicateTensorToMesh` applies only to `_tt_page_table`.
- Fix 2: `symbiote_attention_overview.md` — Corrected `TTNNDistributedRotaryPositionEmbedding` transformation matrix description to distinguish that `move_weights_to_device_impl` builds both `True` and `False` entries, while `forward` always uses `False`.
- Fix 3: `symbiote_attention_overview.md` — Added missing step 8 to `LlamaAttention.forward` describing the `self.o_proj` output projection before the return.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 2

## Re-check of Pass 1 fixes

All 3 Pass 1 fixes are correctly applied in the current text of `symbiote_attention_overview.md`:

- Fix 1 (line ~256): The K/V cache tensor description now correctly states `ttnn.zeros` with no mesh mapper; `ReplicateTensorToMesh` is attributed only to `_tt_page_table`.
- Fix 2 (lines ~196–198): The transformation matrix description now correctly separates `move_weights_to_device_impl` (builds both `True` and `False` entries) from `forward` (always retrieves `False`).
- Fix 3 (line ~151): Step 8 describing `self.o_proj` and the `(attn_out, None)` return is present.

## New issues found

**Issue 1. [`symbiote_attention_overview.md`], TTNNWhisperAttention — all projections always constructed, not conditional on attention type**

The guide stated "Self-attention path: Q, K, V are fused into `qkv_proj`" and "Cross-attention path: Q is stored separately as `q_proj_ttnn`; K and V are stored as `k_proj_cross` and `v_proj_cross`" — framing these as two mutually exclusive construction branches.

Source (`attention.py`, `TTNNWhisperAttention.from_torch`, lines 699–711): `qkv_proj`, `q_proj_ttnn`, `k_proj_cross`, and `v_proj_cross` are **all** created unconditionally for every `WhisperAttention` instance. The `is_cross` flag is a runtime check in `forward` (`key_value_states is not None`), not a construction-time condition. A reader implementing `from_torch` from the guide would create only one set of projections per instance, omitting the other set and causing `AttributeError` at runtime when the other path is taken.

Correct statement: All four attributes are always stored at construction time. `forward` selects between them at runtime based on whether `key_value_states` is provided.

Fix applied: Replaced the two-path description with a single-construction description listing all four always-stored attributes and clarifying the runtime `is_cross` dispatch.

---

**Issue 2. [`symbiote_attention_overview.md`], TTNNSDPAAttention — undocumented unconditional `is_causal` override**

The guide stated: "Infers `is_causal` from the `module` argument's `is_causal` attribute if not provided explicitly."

Source (`attention.py`, `TTNNSDPAAttention.forward`, lines 356–357):
```python
is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
```
The second line unconditionally overrides `is_causal` to `False` when `query.shape[2] <= 1` (single-token query) **or** when `attention_mask is not None` — even if `is_causal=True` was passed as an explicit argument. A reader relying on the guide's description would expect explicit `is_causal=True` to be honored, but it is silently ignored for single-token queries and masked inputs, producing wrong numerical outputs (causal masking disabled when expected).

Correct statement: After the `module` fallback lookup, `is_causal` is further overridden to `False` whenever `query.shape[2] <= 1` or `attention_mask is not None`.

Fix applied: Added the second override condition to the description of step 2 in `TTNNSDPAAttention.forward`.

---

**Issue 3. [`symbiote_attention_overview.md`], LlamaAttention.forward step 5 — padding direction omitted**

The guide stated: "Pads query to `kv_len` if `is_causal` and `q_len < kv_len`."

Source (`attention.py`, `LlamaAttention.forward`, lines 987, 1003–1005):
```python
query_states = ttnn.concat([zero_pad, query_states.to_ttnn], dim=2)  # prepend
...
attn_out = attn_out[:, -original_q_len:, :]  # take last q_len rows
```
The padding is **prepended** (zeros placed before the real query tokens). After SDPA, the **last** `original_q_len` rows are sliced. A reader who appended the zeros instead of prepending them would apply the causal mask to the wrong token positions — real query tokens would be in the leading positions, so the causal mask would attend to future padded positions rather than the cache context, producing wrong attention outputs.

Correct statement: Zero padding is prepended at `dim=2`; the post-SDPA slice takes `attn_out[:, -original_q_len:, :]`.

Fix applied: Updated step 5 to specify that padding is prepended and that the post-SDPA slice takes the last `original_q_len` rows.

---

## Verdict

Chapter approved after 3 new fixes. All Pass 1 fixes held. No remaining correctness issues found.

## Change Log — Pass 2 fixes applied

- Fix 4: `symbiote_attention_overview.md` — Replaced the two-path "self-attention / cross-attention" framing of `TTNNWhisperAttention.from_torch` with an accurate description stating all four projection attributes are always constructed; `is_cross` is a runtime dispatch in `forward`.
- Fix 5: `symbiote_attention_overview.md` — Added the unconditional `is_causal` override (`query.shape[2] > 1 and attention_mask is None and is_causal`) to the `TTNNSDPAAttention.forward` step 2 description.
- Fix 6: `symbiote_attention_overview.md` — Specified in `LlamaAttention.forward` step 5 that zero padding is prepended (`ttnn.concat([zero_pad, query_states], dim=2)`) and that the post-SDPA slice takes the last `original_q_len` rows (`attn_out[:, -original_q_len:, :]`).

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 3

## Re-check of Pass 2 fixes

All 3 Pass 2 fixes are correctly applied in the current text of `symbiote_attention_overview.md`:

- Fix 4 (lines ~98–104): `TTNNWhisperAttention.from_torch` now correctly states all four projection attributes (`qkv_proj`, `q_proj_ttnn`, `k_proj_cross`, `v_proj_cross`) are always constructed; `is_cross` is identified as a runtime dispatch in `forward`.
- Fix 5 (lines ~38–39): `TTNNSDPAAttention.forward` step 2 now includes the second unconditional override: `is_causal` is forced to `False` when `query.shape[2] <= 1` or `attention_mask is not None`.
- Fix 6 (lines ~151–152): `LlamaAttention.forward` step 5 now specifies that zero padding is prepended (`ttnn.concat([zero_pad, query_states], dim=2)`) and the post-SDPA slice takes the last `original_q_len` rows.

## New issues found

**Issue 1. [`transformers_attention_overview.md`], Section 7 RotarySetup table, line ~283 — decode `transformation_mat` shape wrong for TG topology**

The guide stated the decode `transformation_mat` shape as `[1, 1, batch * TILE_SIZE, TILE_SIZE]`.

Source (`rope.py`, lines 469–488):
```python
trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
    1, 1, self.batch_size_per_device_group, 1
)
```
`get_rot_transformation_mat` returns a `[1, 1, TILE_SIZE, TILE_SIZE]` tensor (32×32). After `.repeat(1, 1, self.batch_size_per_device_group, 1)`, the shape is `[1, 1, batch_size_per_device_group * TILE_SIZE, TILE_SIZE]`. On non-TG deployments `batch_size_per_device_group == batch_size`, so the guide's shape happens to be correct. On TG (32-device), `batch_size_per_device_group = max(batch_size // devices_in_shard_dim, 1)`, which is strictly smaller than `batch_size`. A reader targeting TG topology who allocates or expects a `[1, 1, batch * TILE_SIZE, TILE_SIZE]` matrix will produce a shape mismatch that causes `rotary_embedding_llama` to fail or silently produce wrong outputs.

Correct statement: The shape is `[1, 1, batch_size_per_device_group * TILE_SIZE, TILE_SIZE]`. On non-TG, `batch_size_per_device_group == batch_size`. On TG, it equals `batch_size // devices_in_shard_dim`.

Fix applied: Updated the table entry to read `[1, 1, batch_size_per_device_group * TILE_SIZE, TILE_SIZE]` with a note distinguishing the TG and non-TG cases.

---

## Verdict

Chapter approved after 1 new fix. All 3 Pass 2 fixes held. No further correctness issues found.

## Change Log — Pass 3 fixes applied

- Fix 7: `transformers_attention_overview.md` — Corrected the `RotarySetup` decode `transformation_mat` shape from `[1, 1, batch * TILE_SIZE, TILE_SIZE]` to `[1, 1, batch_size_per_device_group * TILE_SIZE, TILE_SIZE]`, with an explanatory note that on TG `batch_size_per_device_group` is divided by the number of devices in the shard dimension.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 4

## Re-check of Pass 3 fix

The Pass 3 fix is correctly applied. `transformers_attention_overview.md` line 283 now reads `[1, 1, batch_size_per_device_group * TILE_SIZE, TILE_SIZE]` with the accompanying note distinguishing TG (`batch_size_per_device_group = batch_size // devices_in_shard_dim`) from non-TG (`batch_size_per_device_group == batch_size`). The source (`rope.py`, line 469–475) confirms this.

## New issues found

**Issue 1. [`symbiote_attention_overview.md`], `LlamaAttention.forward` step 6 — `transpose_output=False` omitted, causing wrong tensor layout into `nlp_concat_heads`**

The guide stated: "Calls `self.sdpa` with `scaling=self.torch_layer.scaling` and `is_causal=self.torch_layer.is_causal`."

Source (`attention.py`, lines 989–999):
```python
attn_out = self.sdpa(
    self,
    query_states,
    key_states,
    value_states,
    None,
    dropout=0.0,
    scaling=self.torch_layer.scaling,
    is_causal=self.torch_layer.is_causal,
    transpose_output=False,
)
```

`TTNNSDPAAttention.forward` with `transpose_output=True` (the default) applies `ttnn.permute(attn_output, (0, 2, 1, 3))` before returning, producing `[batch, seq, heads, head_dim]`. With `transpose_output=False`, the output remains in the native SDPA layout `[batch, heads, seq, head_dim]`. The immediately following `ttnn.experimental.nlp_concat_heads` (step 7, source line 1000) expects heads at axis 1 — i.e., `[batch, heads, seq, head_dim]`. A reader who omits `transpose_output=False` and uses the default `True` would feed a permuted `[batch, seq, heads, head_dim]` tensor into `nlp_concat_heads`, producing wrong head-concatenation order and incorrect attention outputs.

Correct statement: `self.sdpa` is called with `transpose_output=False`; `nlp_concat_heads` is then applied to the `[batch, heads, seq, head_dim]` result.

Fix applied: Updated step 6 to include `transpose_output=False` with an explanation of why the non-default value is required, and clarified step 7's expected input layout.

---

## Verdict

Chapter approved after 1 new fix. All Pass 3 fixes held.

## Change Log — Pass 4 fixes applied

- Fix 8: `symbiote_attention_overview.md` — Added `transpose_output=False` to the `LlamaAttention.forward` step 6 description of the `self.sdpa` call, with an explanation that omitting this (leaving the default `True`) permutes the output to `[batch, seq, heads, head_dim]`, causing `nlp_concat_heads` in step 7 to receive the wrong layout and produce incorrect outputs.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 5

## Re-check of Pass 4 fix

The Pass 4 fix is correctly applied. `symbiote_attention_overview.md` step 6 now reads: "Calls `self.sdpa` with `scaling=self.torch_layer.scaling`, `is_causal=self.torch_layer.is_causal`, and **`transpose_output=False`**." with an explanation of why the non-default value is required. Source (`attention.py`, line 998) confirms `transpose_output=False`. The fix is present and accurate.

## New issues found

**Issue 1. [`transformers_attention_overview.md`], Section 3, line ~109 — `get_math_fidelity` described as returning a `MathFidelitySetting` enum; actually returns a compute kernel config object**

The guide stated: "These are returned by `decoders_optimizations.get_math_fidelity(decoder_id, op, configuration)` which returns one of the `MathFidelitySetting` enums."

Source (`model_config.py`, lines 4086–4095):
```python
def get_math_fidelity(self, decoder_id, op: OpGroup, configuration: ModelArgs):
    math_fidelity_setting_lookup = {
        MathFidelitySetting.LOFI: configuration.compute_kernel_config_lofi,
        MathFidelitySetting.HIFI2: configuration.compute_kernel_config_hifi2,
        MathFidelitySetting.HIFI2_NA: configuration.compute_kernel_config_hifi2_na,
        MathFidelitySetting.HIFI2_FP16: configuration.compute_kernel_config_hifi2_fp16,
        MathFidelitySetting.HIFI4: configuration.compute_kernel_config_hifi4,
        MathFidelitySetting.HIFI4_FP32: configuration.compute_kernel_config_hifi4_fp32,
    }
    return math_fidelity_setting_lookup[self.decoder_optimizations[decoder_id].op_fidelity_settings[op]]
```

The function looks up the `MathFidelitySetting` enum for the given op, then maps it to a **compute kernel config object** (e.g., `configuration.compute_kernel_config_hifi2`, which is a fully-constructed `ttnn.WormholeComputeKernelConfig`). The returned value is the config object, not the enum. A reader implementing integration code who expects a `MathFidelitySetting` enum back from this call would receive a kernel config object, causing type errors or incorrect dispatch logic when trying to pattern-match on the return value.

Correct statement: `get_math_fidelity` returns a compute kernel config object (e.g., `configuration.compute_kernel_config_hifi2`) that can be passed directly to a TTNN op. The `MathFidelitySetting` enum is an internal key used to look up the config; it is not the return value.

Fix applied: Updated the description at line ~109 to state that the function returns a compute kernel config object, not a `MathFidelitySetting` enum, and that callers receive a value suitable for direct use as a TTNN `compute_kernel_config` argument.

## Verdict

Chapter approved after 1 new fix. All Pass 4 fixes held. No further correctness issues found.

## Change Log — Pass 5 fixes applied

- Fix 9: `transformers_attention_overview.md` — Corrected the description of `decoders_optimizations.get_math_fidelity` to state that it returns a compute kernel config object (e.g., `configuration.compute_kernel_config_hifi2`), not a `MathFidelitySetting` enum. The enum is used as an internal lookup key; the return value is a fully-constructed TTNN kernel config passed directly to ops.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 6

## Re-check of Pass 5 fix

The Pass 5 fix is correctly applied. `transformers_attention_overview.md` line ~109 now reads: "which looks up the `MathFidelitySetting` enum stored for the given op and returns the corresponding **compute kernel config object** (e.g., `configuration.compute_kernel_config_hifi2`) — not the `MathFidelitySetting` enum itself. Callers receive a fully-constructed `ttnn.WormholeComputeKernelConfig` (or equivalent) that is passed directly to the TTNN op." Source (`model_config.py`, line 4086–4095) confirms `get_math_fidelity` performs a dictionary lookup and returns the kernel config object, not the enum. The fix is present and accurate.

## New issues found

**Issue 1. [`transformers_attention_overview.md`], Section 7 `RotarySetup` table, line 283 — decode `transformation_mat` shape wrong when `use_qk_fused=True`**

The guide stated: "on non-TG `batch_size_per_device_group == batch_size`" in the `transformation_mat` shape note.

Source (`rope.py`, lines 404–415 and 469–475):
```python
self.doubled_batch_size = self.original_batch_size * 2 if use_qk_fused else self.original_batch_size
# (TG)     batch_size_per_device_group = doubled_batch_size // devices_in_shard_dim
# (non-TG) batch_size_per_device_group = doubled_batch_size

trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
    1, 1, self.batch_size_per_device_group, 1,
)
```

When `use_qk_fused=True`, `doubled_batch_size = batch_size * 2`, so on non-TG `batch_size_per_device_group = batch_size * 2`, not `batch_size`. A reader allocating the decode transformation matrix from the guide with `use_qk_fused=True` would produce a matrix of half the required height, causing the height-sharded memory config (one tile per core) to have a wrong shard shape and the `rotary_embedding_llama_fused_qk` op to fail or silently produce wrong outputs.

Correct statement: `batch_size_per_device_group` is derived from `doubled_batch_size = batch_size * 2 if use_qk_fused else batch_size`. On non-TG, `batch_size_per_device_group == doubled_batch_size`. On TG, it equals `doubled_batch_size // devices_in_shard_dim`.

Fix applied: Updated the table note to describe the `use_qk_fused` doubling step and its effect on `batch_size_per_device_group` for both TG and non-TG topologies.

---

## Verdict

Chapter approved after 1 new fix. All Pass 5 fixes held. No further correctness issues found.

## Change Log — Pass 6 fixes applied

- Fix 10: `transformers_attention_overview.md` — Corrected the `RotarySetup` decode `transformation_mat` shape note to state that `batch_size_per_device_group` is derived from `doubled_batch_size = batch_size * 2` when `use_qk_fused=True` (otherwise `doubled_batch_size = batch_size`). On non-TG `batch_size_per_device_group == doubled_batch_size`; on TG it equals `doubled_batch_size // devices_in_shard_dim`. The previous note said `batch_size_per_device_group == batch_size` on non-TG, which is wrong when `use_qk_fused=True`.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 7

## Re-check of Pass 6 fix

The Pass 6 fix is correctly applied. `transformers_attention_overview.md` line 283 now reads:

> "Decode transformation matrix. `batch_size_per_device_group` is derived from `doubled_batch_size`, where `doubled_batch_size = batch_size * 2` when `use_qk_fused=True`, otherwise `doubled_batch_size = batch_size`. On non-TG, `batch_size_per_device_group == doubled_batch_size`. On TG, `batch_size_per_device_group = doubled_batch_size // devices_in_shard_dim`."

Source (`rope.py`, lines 404–415 and 469–475) confirms that `doubled_batch_size` is used as the base for `batch_size_per_device_group` and that the `use_qk_fused` doubling is correct. The fix is present and accurate.

## New issues found

**Issue 1. [`symbiote_attention_overview.md`], `LlamaAttention.forward` step 5 — post-padding slice described as occurring "after SDPA"; actual source applies it after `nlp_concat_heads` + `squeeze`, causing wrong dimension indexing if implemented early**

The guide stated in step 5: "After SDPA, the last `original_q_len` rows are sliced (`attn_out[:, -original_q_len:, :]`)."

Source (`attention.py`, lines 1000–1005):
```python
attn_out = ttnn.experimental.nlp_concat_heads(attn_out.to_ttnn)   # step 7
attn_out = ttnn.squeeze(attn_out, 1)                               # step 7
# Slice output if query was padded
if self.torch_layer.is_causal and original_q_len < kv_len:
    attn_out = attn_out[:, -original_q_len:, :]                   # AFTER squeeze
```

The slice `attn_out[:, -original_q_len:, :]` is a 3D slice that targets `dim=1` — the sequence dimension — only after `nlp_concat_heads` and `squeeze` have reduced the tensor from `[B, H, kv_len, D]` to `[B, kv_len, D]`. If a reader applies the slice to the 4D SDPA output directly (as the guide's step 5 description implied), `[:, -original_q_len:, :]` indexes dimension 1 of a `[B, H, kv_len, D]` tensor, which is the **heads dimension**, not the sequence dimension. This produces a wrong subset of heads, not a correct subset of sequence positions — a silent numerical error that passes shape checks but gives completely wrong attention outputs.

Correct statement: The post-padding slice is applied after `nlp_concat_heads` and `squeeze`, when the tensor is 3D `[B, kv_len, D]`. It must not be applied to the 4D SDPA output.

Fix applied: Moved the post-padding slice description from step 5 to step 7, explicitly stating the slice occurs after `squeeze` on the 3D tensor, and explaining why applying it to the 4D output would index the wrong dimension.

---

## Verdict

Chapter approved after 1 new fix. All Pass 6 fixes held. No further correctness issues found.

## Change Log — Pass 7 fixes applied

- Fix 11: `symbiote_attention_overview.md` — Moved the post-padding slice description from step 5 to step 7 of `LlamaAttention.forward`. Step 7 now specifies that `attn_out[:, -original_q_len:, :]` is applied after `nlp_concat_heads` and `squeeze` (when the tensor is 3D `[B, kv_len, D]`), and warns that applying the same slice expression to the 4D SDPA output `[B, H, kv_len, D]` would incorrectly index the heads dimension instead of the sequence dimension, producing wrong outputs.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 8

## Re-check of Pass 7 fix

The Pass 7 fix is correctly applied. `symbiote_attention_overview.md` step 7 now states that `attn_out[:, -original_q_len:, :]` is applied after `nlp_concat_heads` and `squeeze` (on the 3D tensor), and includes the warning that applying the same expression to the 4D SDPA output would index the heads dimension rather than the sequence dimension. Source (`attention.py`, lines 1000–1005) confirms the slice occurs after `squeeze`, when the tensor is 3D `[B, kv_len, D]`. The fix is present and accurate.

## New issues found

**Issue 1. [`symbiote_attention_overview.md`], Section 1.5 `TTNNWhisperAttention` — `out_proj` attribute omitted from `from_torch` inventory, causing `AttributeError` in `forward`**

The guide's description of `TTNNWhisperAttention.from_torch` listed four always-constructed attributes: `qkv_proj`, `q_proj_ttnn`, `k_proj_cross`, and `v_proj_cross`.

Source (`attention.py`, line 711):
```python
new_attn.out_proj = TTNNLinear.from_torch(whisper_attn.out_proj)
```

A fifth attribute, `out_proj`, is unconditionally constructed in `from_torch`. It is the final computation in `forward` (source line 830):
```python
return self.out_proj(attn_out), None, past_key_value
```

Both the self-attention path and the cross-attention path converge to `self.out_proj` at the end of `forward`. A reader implementing `from_torch` from the guide would omit `out_proj`, causing an `AttributeError: 'TTNNWhisperAttention' object has no attribute 'out_proj'` on the first forward call — this is a blocking implementation error.

Correct statement: `from_torch` always constructs five `TTNNLinear` attributes: `qkv_proj`, `q_proj_ttnn`, `k_proj_cross`, `v_proj_cross`, and `out_proj`.

Fix applied: Added `out_proj` to the attribute list in the `TTNNWhisperAttention.from_torch` description, noting it is used in both forward paths as the final output projection step.

---

## Verdict

Chapter approved after 1 new fix. All Pass 7 fixes held. No further correctness issues found.

## Change Log — Pass 8 fixes applied

- Fix 12: `symbiote_attention_overview.md` — Added the missing `out_proj` attribute to the `TTNNWhisperAttention.from_torch` inventory. The guide previously listed only four always-constructed attributes (`qkv_proj`, `q_proj_ttnn`, `k_proj_cross`, `v_proj_cross`); the source unconditionally constructs a fifth (`out_proj = TTNNLinear.from_torch(whisper_attn.out_proj)` at line 711), which is required for `forward` to complete without `AttributeError`.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 9

## Re-check of Pass 8 fix

The Pass 8 fix is correctly applied. `symbiote_attention_overview.md` section 1.5 (`TTNNWhisperAttention`) now lists `out_proj` as a fifth always-constructed attribute in `from_torch`, noting it is used in both the self-attention and cross-attention forward paths as the final output projection step (source line 830: `return self.out_proj(attn_out), None, past_key_value`). Source line 711 confirms `out_proj = TTNNLinear.from_torch(whisper_attn.out_proj)` is unconditional. The fix is present and accurate.

## New issues found

**Issue 1. [`symbiote_attention_overview.md`], Section 1.6 `LlamaAttention`, `from_torch` code block annotation — `init_parameters` listed as storing only `q_proj`, `k_proj`, `v_proj`; source also stores `o_proj`**

The guide's code block annotation on line 139 read:

```python
new_attn.init_parameters()            # stores q_proj, k_proj, v_proj separately
```

Source (`attention.py`, `init_parameters`, lines 909–913):

```python
def init_parameters(self):
    self.q_proj = TTNNLinear.from_torch(self.torch_layer.q_proj)
    self.k_proj = TTNNLinear.from_torch(self.torch_layer.k_proj)
    self.v_proj = TTNNLinear.from_torch(self.torch_layer.v_proj)
    self.o_proj = TTNNLinear.from_torch(self.torch_layer.o_proj)
```

`init_parameters` always stores `o_proj` in addition to `q_proj`, `k_proj`, and `v_proj`. The guide's annotation enumerated only the three Q/K/V projections, omitting the output projection. A reader implementing the non-fused GQA path from this code block would fail to store `o_proj`, causing an `AttributeError` when `forward` reaches `return self.o_proj(attn_out), None` (source line 1007).

Correct statement: `init_parameters` stores four `TTNNLinear` attributes: `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

Fix applied: Updated the annotation to read `# stores q_proj, k_proj, v_proj, and o_proj separately`.

---

## Verdict

1 new factual error found and fixed. All Pass 8 fixes held. After this fix, no remaining correctness issues found.

## Change Log — Pass 9 fixes applied

- Fix 13: `symbiote_attention_overview.md` — Corrected the `LlamaAttention.from_torch` code block annotation for the `init_parameters()` call from `# stores q_proj, k_proj, v_proj separately` to `# stores q_proj, k_proj, v_proj, and o_proj separately`. Source `init_parameters` (lines 909–913) stores all four projections; omitting `o_proj` would cause an `AttributeError` in `forward` at the `self.o_proj(attn_out)` call.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 10

## Re-check of Pass 9 fix

The Pass 9 fix is correctly applied. `symbiote_attention_overview.md` line 139 now reads:

```python
new_attn.init_parameters()            # stores q_proj, k_proj, v_proj, and o_proj separately
```

Source (`attention.py`, `init_parameters`, lines 909–913) confirms `o_proj = TTNNLinear.from_torch(self.torch_layer.o_proj)` is stored unconditionally alongside `q_proj`, `k_proj`, and `v_proj`. The fix is present and accurate.

## New issues found

**Issue 1. [`transformers_attention_overview.md`], Section 7 `RotarySetup` table, line ~284 — `transformation_mat_prefill` shape stated as `[1, 1, head_dim, head_dim]`; actual shape is always `[1, 1, 32, 32]`**

The guide stated the prefill transformation matrix shape as `[1, 1, head_dim, head_dim]`.

Source (`rope.py`, line 493):
```python
prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
```

This appears to pass `head_dim` as the dimension, but `get_rot_transformation_mat` in `models/tt_transformers/tt/common.py` (lines 473–477) hardcodes `dhead = 32` unconditionally regardless of the argument:

```python
def get_rot_transformation_mat(dhead=32):
    # ROPE op uses a single tile
    dhead = 32
    return get_rot_transformation_mat_v2(dhead)
```

`get_rot_transformation_mat_v2` (aliased from `models.common.tensor_utils.get_rot_transformation_mat`) returns a `[1, 1, dhead, dhead]` tensor. With `dhead` hardcoded to 32, the prefill transformation matrix always has shape `[1, 1, 32, 32]` — identical in size to the decode transformation matrix base tile — regardless of `head_dim`.

A reader who allocates or expects a `[1, 1, head_dim, head_dim]` prefill matrix for a model with `head_dim != 32` (e.g., a 128-dimensional head) would produce the wrong tensor, causing `rotary_embedding_llama` to fail on a shape mismatch or silently apply incorrect rotation coefficients.

Correct statement: `transformation_mat_prefill` always has shape `[1, 1, 32, 32]` (i.e., `[1, 1, TILE_SIZE, TILE_SIZE]`) because `get_rot_transformation_mat` ignores its `dhead` argument and hardcodes 32.

Fix applied: Updated the table entry for `transformation_mat_prefill` to state shape `[1, 1, TILE_SIZE, TILE_SIZE]` (i.e., `[1, 1, 32, 32]`) with a note that `get_rot_transformation_mat` hardcodes `dhead = 32` internally regardless of the `head_dim` argument.

## Verdict

1 new factual error found and fixed. The Pass 9 fix held. After this fix, no remaining correctness issues found.

## Change Log — Pass 10 fixes applied

- Fix 14: `transformers_attention_overview.md` — Corrected the `RotarySetup` table entry for `transformation_mat_prefill` shape from `[1, 1, head_dim, head_dim]` to `[1, 1, TILE_SIZE, TILE_SIZE]` (i.e., `[1, 1, 32, 32]`). Source shows `get_rot_transformation_mat(dhead=head_dim)` is called but the function body immediately overwrites the argument with `dhead = 32`, so the matrix is always 32×32 regardless of the model's actual `head_dim`.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 11

## Re-check of Pass 10 fix

The Pass 10 table fix is confirmed correct. `transformers_attention_overview.md` line 284 now reads `[1, 1, TILE_SIZE, TILE_SIZE]` (i.e., `[1, 1, 32, 32]`) for `transformation_mat_prefill`, with the note that `get_rot_transformation_mat` hardcodes `dhead = 32` internally. Source (`models/tt_transformers/tt/common.py`, lines 473–477) confirms:

```python
def get_rot_transformation_mat(dhead=32):
    # ROPE op uses a single tile
    dhead = 32
    return get_rot_transformation_mat_v2(dhead)
```

The table entry is accurate. However, the prose paragraph immediately following the table (line 286) was **not** updated by Pass 10 and still contains the claim that `transformation_mat_prefill` "uses the actual `head_dim`" — this is a residual factual error introduced by the partial Pass 10 fix and is addressed below.

## New issues found

**Issue 1. [`transformers_attention_overview.md`], line 286 — asymmetry prose contradicts the Pass 10 corrected table entry**

The paragraph below the `RotarySetup` allocation table read:

> "The decode `transformation_mat` uses `dhead=ttnn.TILE_SIZE` (32), while the prefill `transformation_mat_prefill` uses the actual `head_dim`. This asymmetry exists because decode operates on a single token per head (fitting in one tile row), while prefill processes full sequences."

Pass 10 corrected the table entry for `transformation_mat_prefill` to `[1, 1, 32, 32]` with the explicit note that `get_rot_transformation_mat` hardcodes `dhead = 32` regardless of the `head_dim` argument. The prose paragraph was not updated, leaving the guide asserting two contradictory facts in adjacent lines: the table says the prefill matrix is always 32×32, and the paragraph says it uses the actual `head_dim`.

Source (`models/tt_transformers/tt/common.py`, line 475: `dhead = 32`) confirms the hardcode. There is no `head_dim`-dependent asymmetry: both matrices use 32. A reader consulting the prose (rather than the table) would still believe the prefill matrix dimensions vary with `head_dim`, and would allocate or expect a wrong-sized tensor for any model with `head_dim != 32`.

Correct statement: Both matrices are `[1, 1, 32, 32]` in base tile shape. The decode matrix is additionally repeated to cover `batch_size_per_device_group` cores. The prefill matrix is not shaped by `head_dim`.

Fix applied: Replaced the asymmetry paragraph with an accurate statement that both matrices use `dhead = 32` due to the hardcode in `get_rot_transformation_mat`, and clarified that the decode matrix's repetition across cores is the only structural difference.

## Verdict

1 factual error found and fixed. The Pass 10 table fix held; its accompanying prose was the error. After this fix, no remaining correctness issues found in any of the three guide files.

## Change Log — Pass 11 fixes applied

- Fix 15: `transformers_attention_overview.md` — Replaced the contradictory "asymmetry" paragraph (line 286) with an accurate statement: both `transformation_mat` and `transformation_mat_prefill` have a base tile shape of `[1, 1, 32, 32]` because `get_rot_transformation_mat` hardcodes `dhead = 32` regardless of the `head_dim` argument passed by `rope.py`. The only structural difference is that the decode matrix is repeated `batch_size_per_device_group` times along dimension 2 to tile across cores.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 12

## Re-check of Pass 11 fix

The Pass 11 fix is correctly applied. `transformers_attention_overview.md` lines 286–287 now read:

> "Both the decode `transformation_mat` and the prefill `transformation_mat_prefill` are `[1, 1, 32, 32]` in their base tile shape. Although `rope.py` calls `get_rot_transformation_mat(dhead=head_dim)` for the prefill matrix, the function body in `models/tt_transformers/tt/common.py` immediately overwrites the argument with `dhead = 32`, so both matrices use 32 regardless of the model's `head_dim`. The decode matrix is additionally repeated along dimension 2 to tile it across `batch_size_per_device_group` cores."

Source (`models/tt_transformers/tt/common.py`, lines 473–477) confirms `dhead = 32` is hardcoded unconditionally. There is no stale prose claiming a `head_dim`-dependent asymmetry. The fix is present and accurate.

## New issues found

No feedback — chapter approved.

## Verdict

Pass 12: No factual errors found. All 15 prior fixes held across all three guide files. The chapter is approved as-is.

## Change Log — Pass 12 fixes applied

No fixes required.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 13

## New issues found

No feedback — chapter approved.

Both new table rows added by Agent C's compression pass were verified against source:

**Row: "Pre-RoPE bfloat16 typecast" — Decode `—`, Prefill `bfloat16`**

Source (`attention.py`, lines 854–866): `forward_prefill` contains an explicit conditional `ttnn.typecast(..., dtype=ttnn.bfloat16)` immediately before calling `rotary_embedding_llama` on Q and K. The decode path (`forward_decode`, lines 526–540) achieves bfloat16 implicitly — the `ttnn.sharded_to_interleaved(..., ttnn.bfloat16)` call (or `ttnn.matmul(..., dtype=ttnn.bfloat16)` in the TG case) produces bfloat16 output before `nlp_create_qkv_heads_decode`; no further typecast is issued immediately before the rotary op. The `—` for decode correctly denotes the absence of a dedicated pre-RoPE typecast call in that path. Both cells are accurate.

**Row: "Pre-cache kv_cache_dtype typecast" — Decode `—`, Prefill `K/V → kv_cache_dtype`**

Source (`attention.py`, lines 883–892): `forward_prefill` calls `ttnn.typecast(k_heads_1KSD, dtype=keys_BKSD.dtype)` and `ttnn.typecast(v_heads_1VSD, dtype=values_BKSD.dtype)` before writing to the cache. `keys_BKSD.dtype` equals `kv_cache_dtype` because `init_kv_cache` allocates the cache with `dtype=self.kv_cache_dtype` (line 408). The decode path (`forward_decode`, lines 599–609) writes K and V directly into the cache via `paged_update_cache` / `paged_fused_update_cache` with no preceding typecast. Both cells are accurate.

## Verdict

Pass 13: No factual errors found. All prior fixes held across all three guide files. The chapter is approved as-is.

## Change Log — Pass 13 fixes applied

No fixes required.

---

# Agent B Review — Chapter 3: Attention: RoPE, KV Cache, and SDPA — Pass 14

## New issues found

No feedback — chapter approved.

All three guide files were verified line-by-line against the five source files. The following areas received explicit cross-checks against source:

- `TTNNSDPAAttention.forward` `is_causal` override logic (source lines 356–357): the compound expression `query.shape[2] > 1 and attention_mask is None and is_causal` matches the guide's step 2 description.
- `TTNNFusedQKVSelfAttention.forward` `transpose_k_heads=False` and L1 move (source lines 524–531): matches guide section 1.3.
- `TTNNSelfAttention` `q_chunk_size=256`, `k_chunk_size=256`, HiFi4 kernel (source lines 573–587): matches guide section 1.2 and 1.4.
- `TTNNWhisperAttention.from_torch` five always-constructed attributes including `out_proj` (source lines 699–711): matches guide section 1.5.
- `LlamaAttention.from_torch` `qkv_same_shape` logic and `init_parameters` storing all four projections including `o_proj` (source lines 921–928, 909–913): matches guide section 1.6 annotation on line 139.
- `LlamaAttention.forward` prepend padding (`ttnn.concat([zero_pad, query_states], dim=2)`, source line 987), `transpose_output=False` (source line 998), post-squeeze slice at `attn_out[:, -original_q_len:, :]` (source lines 1000–1005): all match guide steps 5–7.
- `TTNNRotaryPositionEmbedding` partial-vs-full rotary branches (source lines 113–154): matches guide section 2.1 table.
- `TTNNDistributedRotaryPositionEmbedding` transform matrix loop `[True, False]` in `move_weights_to_device_impl` and hardcoded `is_decode_mode = False` in `forward` (source lines 179–247): matches guide section 2.2.
- `PagedAttentionConfig` defaults and derived properties (source lines 62–73): `max_seq_length = 2048 * 64 = 131072`, `blocks_per_sequence = 2048` confirmed correct.
- `TTNNPagedAttentionKVCache.to_device` K/V cache via `ttnn.zeros` with no mesh mapper; `ReplicateTensorToMesh` applied only to `_tt_page_table` (source lines 105–145): matches guide section 3.2.
- `RotarySetup` constructor allocation table including `transformation_mat_prefill` always `[1, 1, 32, 32]` due to `get_rot_transformation_mat` hardcoding `dhead = 32` (source `common.py` lines 473–477): matches guide line 258 and prose at line 260.
- `ModelOptimizations._default_settings` six op fidelity defaults (source lines 269–274): all match guide section 3 table.
- `shard_wo_dims = (2, 3)` when `use_fused_all_gather_matmul or TG`, else `(3, 2)` (source line 309): matches guide section 8.
- Agent C Pass 2 deletion: the `TTNNPagedAttentionKVCache.update` warning block now opens directly with the hardware-acceleration caveat; no missing context was introduced by the removal.
- Agent C Pass 3 deletion: the post-RotarySetup-table paragraph now opens with the decode-matrix repetition sentence; the removed `dhead=32` restatement was fully redundant with the table note at line 258 and its absence leaves the section consistent.

## Verdict

Pass 14: No factual errors found. All 15 prior fixes held. Agent C's two compression-pass deletions did not introduce any errors or leave any orphaned references. The chapter is approved as-is.

## Change Log — Pass 14 fixes applied

No fixes required.
