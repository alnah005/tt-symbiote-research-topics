# Compression Analysis: Chapter 5 — Built-In TTNN Modules — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~580 lines
- Estimated post-compression line count: ~525 lines
- Estimated reduction: ~10%

---

## CRUCIAL Suggestions

### [linear_layers.md] ~lines 46–48
**Issue:** "Both weight and bias are preprocessed into `bfloat16` tile layout on the host." — this sentence appears immediately after the `preprocess_weights_impl` code block, which already shows `dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT` for both weight and bias. Pure paraphrase of adjacent code, adds nothing.
**Suggestion:** Delete the trailing prose sentence after the `preprocess_weights_impl` code block.

### [linear_layers.md] ~line 95
**Issue:** "The `forward` body delegates entirely to `TTNNLinear.forward` after weight deallocation is set up by the decorator." — this restates what the one-liner `return super().forward(input_tensor)` in the adjacent code block already shows explicitly.
**Suggestion:** Delete this sentence.

### [linear_layers.md] ~lines 155–157
**Issue:** "The weight is sharded along `weight_dim` (`-2`) across the device mesh; the bias is sharded along `input_dim` (`-1`)." — immediately follows the `move_weights_to_device_impl` code block, which already shows `dim=self.weight_dim` and `dim=self.input_dim` directly. Pure paraphrase.
**Suggestion:** Delete the trailing prose sentence after the `move_weights_to_device_impl` code block for `TTNNLinearInputShardedWeightSharded`.

### [normalization_and_activation.md] ~lines 41–42
**Issue:** "Both weight and bias are converted to `bfloat16` tile tensors on the host." — immediately follows the `preprocess_weights_impl` code block for `TTNNLayerNorm`, which already shows `dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT` for both weight and bias. Pure paraphrase.
**Suggestion:** Delete this trailing prose sentence.

### [normalization_and_activation.md] ~lines 93–94
**Issue:** "The weight vector is unsqueezed and broadcast-expanded to shape `[32, hidden_size]` before conversion. No bias is stored; `DeepseekV2RMSNorm` has no bias parameter." — the first half restates the adjacent code (`self.torch_layer.weight.unsqueeze(0).expand(32, -1)`). The second sentence ("No bias is stored; `DeepseekV2RMSNorm` has no bias parameter") is load-bearing and must be kept.
**Suggestion:** Delete only the first sentence ("The weight vector is unsqueezed and broadcast-expanded to shape `[32, hidden_size]` before conversion.") since the code is self-evident. Retain the second sentence about no bias.

### [linear_layers.md] ~lines 258–259
**Issue:** "It calls the inner `linear_class` module, then applies the TTNN activation function to the result." — immediately follows the three-line `forward` code block for `TTNNLinearActivation`, which is fully readable. Pure paraphrase.
**Suggestion:** Delete this trailing prose sentence.

---

## MINOR Suggestions

### [linear_layers.md] ~line 236
**Issue:** "A fused linear + activation module. It is not typically instantiated directly; use `TTNNLinearGelu` or `TTNNLinearSilu` instead." — the "not typically instantiated directly" guidance is partially self-evident from the factory pattern above it.
**Suggestion:** The second sentence ("use `TTNNLinearGelu` or `TTNNLinearSilu` instead") is a useful cross-reference; keep it. The first sentence is borderline redundant with the heading and section context, but the clarification about factory usage is worth retaining.

### [linear_layers.md] ~line 283
**Issue:** "The default `linear_class` is `TTNNLinear` but any compatible class may be substituted." — this is partially inferrable from the `linear_class=TTNNLinear` default parameter in the code directly above.
**Suggestion:** Consider deleting this sentence since it restates the default argument.

### [linear_layers.md] ~line 362
**Issue:** "No program config is applied in decode mode." — inferrable from the absence of `compute_kernel_config` and `program_config` kwargs in the `decode_forward` code block above.
**Suggestion:** Delete or fold into the `prefill_forward` description as a contrast ("unlike `decode_forward`, `prefill_forward` applies...").

### [normalization_and_activation.md] ~lines 167–173
**Issue:** The intro prose block for the Activation Modules section ("All three activation modules follow a common pattern…") partially duplicates what the Summary Table at the bottom of the file captures in structured form.
**Suggestion:** Keep the intro block — it contextualizes the pattern before the per-class sections — but note the Summary Table at lines 229–238 also covers this. No critical redundancy requiring a CRUCIAL fix, but the two could be consolidated on a future pass.

---

## Load-Bearing Evidence

- `linear_layers.md` line ~31: "`from_parameters` accepts raw weight tensors directly and calls `preprocess_weights()` immediately, then deletes the raw `weight` and `bias` attributes." — load-bearing because the deletion of raw attributes after `preprocess_weights()` is a non-obvious behavioral difference from `from_torch`.
- `linear_layers.md` line ~174: "Calls `ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)` — note: **no bias** is passed here." — load-bearing because the bias omission during matmul is a non-obvious departure from `TTNNLinear.forward`.
- `linear_layers.md` line ~175: `reduce_scatter_minimal_async` parameters (`cluster_axis=1`, `ttnn.Topology.Ring`, `chunks_per_sync=10`, `num_workers_per_link=2`, `num_buffers_per_channel=2`) — load-bearing specific numeric constants not inferable from names.
- `linear_layers.md` line ~376: "For modules named `"lm_head"`, the program config is `None`." — load-bearing because this name-based special-case is non-obvious and the only place it is documented in Chapter 5.
- `linear_layers.md` line ~334–335: `compute_kernel_config` details (HiFi2, no math approx, no fp32 accumulator, L1 packer accumulation enabled) — load-bearing specific hardware configuration.
- `normalization_and_activation.md` line ~94: "No bias is stored; `DeepseekV2RMSNorm` has no bias parameter." — load-bearing because it documents an absence that a reader might otherwise assume exists.
- `normalization_and_activation.md` line ~126: "Note: the parameter type annotation in source is `"RMSNorm"` (a forward reference), not `DeepseekV2RMSNorm`." — load-bearing for developers who inspect the source signature.
- `normalization_and_activation.md` line ~206: "Unlike `TTNNSilu` and `TTNNGelu`, `TTNNReLU` preserves the input tensor's original memory config rather than defaulting to `DRAM_MEMORY_CONFIG`." — load-bearing behavioral difference.
- `attention_and_conv.md` line ~55: "The fallback PyTorch layers (`NHWCConvPytorch`, `NHWCMaxpoolPytorch`, `NHWCUpsamplePytorch`) handle the NCHW-to-NHWC permutation for CPU execution." — load-bearing; non-obvious that fallback handles layout conversion.
- `attention_and_conv.md` line ~83: "`TTNNAdd` calls `ensure_tile_layout` on both inputs before the add operation." — load-bearing; non-obvious pre-condition not shown in the table.

---

## VERDICT
- Crucial updates: yes

## Change Log — Pass 1 CRUCIAL fixes applied

- Change 1: `linear_layers.md` line ~48 — deleted trailing prose sentence "Both weight and bias are preprocessed into `bfloat16` tile layout on the host." after the `preprocess_weights_impl` code block for `TTNNLinear`.
- Change 2: `linear_layers.md` line ~95 — deleted trailing prose sentence "The `forward` body delegates entirely to `TTNNLinear.forward` after weight deallocation is set up by the decorator." after the `TTNNLinearLLama` forward code block.
- Change 3: `linear_layers.md` line ~157 — deleted trailing prose sentence "The weight is sharded along `weight_dim` (`-2`) across the device mesh; the bias is sharded along `input_dim` (`-1`)." after the `move_weights_to_device_impl` code block for `TTNNLinearInputShardedWeightSharded`.
- Change 4: `normalization_and_activation.md` line ~42 — deleted trailing prose sentence "Both weight and bias are converted to `bfloat16` tile tensors on the host." after the `preprocess_weights_impl` code block for `TTNNLayerNorm`.
- Change 5: `normalization_and_activation.md` line ~93 — deleted only the first sentence "The weight vector is unsqueezed and broadcast-expanded to shape `[32, hidden_size]` before conversion." (code-restating), retained the second sentence about no bias being stored.
- Change 6: `linear_layers.md` line ~259 — deleted trailing prose sentence "It calls the inner `linear_class` module, then applies the TTNN activation function to the result." after the `TTNNLinearActivation` forward code block.

---

# Compression Analysis: Chapter 5 — Built-In TTNN Modules — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~907 lines (after Pass 1)
- Estimated post-compression line count: ~905 lines
- Estimated reduction: ~0.2%

## Pass 1 Fix Verification

- Change 1 (`linear_layers.md` ~line 48 — trailing prose after `TTNNLinear.preprocess_weights_impl`): CORRECTLY APPLIED
- Change 2 (`linear_layers.md` ~line 95 — trailing prose after `TTNNLinearLLama.forward`): CORRECTLY APPLIED
- Change 3 (`linear_layers.md` ~line 157 — trailing prose after `TTNNLinearInputShardedWeightSharded.move_weights_to_device_impl`): CORRECTLY APPLIED
- Change 4 (`normalization_and_activation.md` ~line 42 — trailing prose after `TTNNLayerNorm.preprocess_weights_impl`): CORRECTLY APPLIED
- Change 5 (`normalization_and_activation.md` ~line 93 — first sentence deleted, second sentence about no bias retained): CORRECTLY APPLIED
- Change 6 (`linear_layers.md` ~line 259 — trailing prose after `TTNNLinearActivation.forward`): CORRECTLY APPLIED

## CRUCIAL Suggestions

### [normalization_and_activation.md] line 145
**Issue:** "The weight is reshaped to `[1, 1, dim//32, 32]` and sharded across the mesh's second dimension (column axis)." — immediately follows the `move_weights_to_device_impl` code block for `TTNNDistributedRMSNorm`, which already shows `.reshape([1, 1, dim // 32, 32])` and `dims=(None, 2)` explicitly. Pure paraphrase of adjacent code; same pattern as the 6 fixes in Pass 1.
**Suggestion:** Delete this trailing prose sentence.

### [linear_layers.md] line 221
**Issue:** "`forward` applies `@deallocate_weights_after` and delegates to `super().forward(input_tensor)`." — This sentence restates information already explicit in two places: the `**Decorators:**` header line at the top of the `TTNNLinearLLamaIColShardedWRowSharded` section (which lists `@deallocate_weights_after`), and the identical `super().forward` pattern documented with a code block for `TTNNLinearLLama` earlier in the file. No new information is added.
**Suggestion:** Delete this trailing prose sentence.

## MINOR Suggestions

### [linear_layers.md] line 275 (TTNNLinearGelu)
Previously flagged in Pass 1. "The default `linear_class` is `TTNNLinear` but any compatible class may be substituted." restates the `linear_class=TTNNLinear` default parameter in the code block directly above it. Low priority.

### [linear_layers.md] line 354 (SmartTTNNLinear — decode_forward)
Previously flagged in Pass 1. "No program config is applied in decode mode." is inferrable from the absence of `compute_kernel_config` and `program_config` in the `decode_forward` code block. Low priority; contrast with `prefill_forward` is useful context.

### [normalization_and_activation.md] lines 166–171 vs. lines 229–237 (Activation Modules intro vs. Summary Table)
Previously flagged in Pass 1. The intro prose block ("All three activation modules follow a common pattern…") partially duplicates the Summary Table. No change recommended at this pass.

### [index.md] vs. [attention_and_conv.md] — catalogue overlap
The Attention, RoPE, Tensor-Op, Conv, and MoE tables in `index.md` are near-verbatim copies of the same tables in `attention_and_conv.md`. This is structural (index serves navigation, sub-page serves depth) and intentional, so not a CRUCIAL issue. A future pass could shorten the `index.md` descriptions to one-word summaries and rely on the sub-pages for full text.

## Load-Bearing Evidence

All load-bearing items from Pass 1 remain valid and unchanged. No new load-bearing content identified in Pass 2 beyond what was already catalogued.

Previously identified load-bearing items (unchanged):
- `linear_layers.md` line ~31: `from_parameters` calls `preprocess_weights()` immediately and then deletes raw `weight`/`bias` attributes — non-obvious behavioral difference from `from_torch`.
- `linear_layers.md` line ~168: "no bias is passed" in `TTNNLinearIColShardedWRowSharded.forward` — non-obvious departure from `TTNNLinear`.
- `linear_layers.md` line ~169: `reduce_scatter_minimal_async` numeric constants (`chunks_per_sync=10`, `num_workers_per_link=2`, `num_buffers_per_channel=2`).
- `linear_layers.md` line ~368: "For modules named `"lm_head"`, the program config is `None`." — name-based special-case.
- `linear_layers.md` line ~326–327: `compute_kernel_config` details (HiFi2, no math approx, no fp32 accumulator, L1 packer accumulation enabled).
- `normalization_and_activation.md` line ~92: "No bias is stored; `DeepseekV2RMSNorm` has no bias parameter."
- `normalization_and_activation.md` line ~124: Forward reference type annotation `"RMSNorm"` in `TTNNDistributedRMSNorm.from_torch`.
- `normalization_and_activation.md` line ~204: `TTNNReLU` preserves original memory config rather than defaulting to `DRAM_MEMORY_CONFIG`.
- `attention_and_conv.md` line ~55: Fallback PyTorch layers handle NCHW-to-NHWC permutation for CPU execution.
- `attention_and_conv.md` line ~83: `TTNNAdd` calls `ensure_tile_layout` on both inputs before the add.

## VERDICT
- Crucial updates: yes

## Change Log — Pass 2 CRUCIAL fixes applied

- Change 7: `normalization_and_activation.md` line ~145 — deleted trailing prose sentence "The weight is reshaped to `[1, 1, dim//32, 32]` and sharded across the mesh's second dimension (column axis)." after the `move_weights_to_device_impl` code block for `TTNNDistributedRMSNorm`.
- Change 8: `linear_layers.md` line ~221 — deleted trailing prose sentence "`forward` applies `@deallocate_weights_after` and delegates to `super().forward(input_tensor)`." from the `TTNNLinearLLamaIColShardedWRowSharded` section.

---

# Compression Analysis: Chapter 5 — Built-In TTNN Modules — Pass 3

## Summary
- Current line count: 899 lines (after Passes 1+2; measured: index.md 137, linear_layers.md 410, normalization_and_activation.md 238, attention_and_conv.md 114)
- Estimated post-compression: ~899 lines
- Estimated reduction: ~0%

## Pass 1+2 Fix Verification

- Change 1 (`linear_layers.md` ~line 48 — trailing prose after `TTNNLinear.preprocess_weights_impl`): CORRECTLY APPLIED — code block ends at line 46, no trailing prose present.
- Change 2 (`linear_layers.md` ~line 95 — trailing prose after `TTNNLinearLLama.forward`): CORRECTLY APPLIED — code block ends at line 91, section separator follows directly.
- Change 3 (`linear_layers.md` ~line 157 — trailing prose after `TTNNLinearInputShardedWeightSharded.move_weights_to_device_impl`): CORRECTLY APPLIED — code block ends at line 151, section separator follows directly.
- Change 4 (`normalization_and_activation.md` ~line 42 — trailing prose after `TTNNLayerNorm.preprocess_weights_impl`): CORRECTLY APPLIED — code block ends at line 39, next heading follows.
- Change 5 (`normalization_and_activation.md` ~line 93 — first sentence deleted, second sentence about no bias retained): CORRECTLY APPLIED — line 92 reads only "No bias is stored; `DeepseekV2RMSNorm` has no bias parameter." with no preceding code-restating sentence.
- Change 6 (`linear_layers.md` ~line 259 — trailing prose after `TTNNLinearActivation.forward`): CORRECTLY APPLIED — code block ends at line 249, section separator follows directly.
- Change 7 (`normalization_and_activation.md` ~line 145 — trailing prose after `TTNNDistributedRMSNorm.move_weights_to_device_impl`): CORRECTLY APPLIED — code block ends at line 143, next heading `#### forward` follows directly.
- Change 8 (`linear_layers.md` ~line 221 — trailing prose from `TTNNLinearLLamaIColShardedWRowSharded`): CORRECTLY APPLIED — section ends with "...is the same as the parent class." and section separator follows directly.

## CRUCIAL Suggestions

None.

Detailed reasoning: All four files were re-read in full after Passes 1+2. No new verbatim or near-verbatim restatements of adjacent code were found. The remaining prose blocks (e.g., `SmartTTNNLinearLLama` description at lines 375–376, `SmartTTNNLinearLLamaBFloat16` at lines 383–384, and the three per-class "Fallback:" lines in `normalization_and_activation.md`) all add some marginal information (explicit override confirmation, self-contained section references) beyond what the Summary Tables alone convey, placing them in MINOR territory at most.

## MINOR Suggestions

All carried from Passes 1 and 2 — no new MINOR issues identified:

- `linear_layers.md` line 273 (`TTNNLinearGelu`): "The default `linear_class` is `TTNNLinear` but any compatible class may be substituted." restates the `linear_class=TTNNLinear` default argument visible in both factory methods directly above. (Flagged Pass 1.)
- `linear_layers.md` line 352 (`SmartTTNNLinear.decode_forward`): "No program config is applied in decode mode." is inferrable from the absence of `compute_kernel_config` and `program_config` in the `decode_forward` code block. Useful contrast with `prefill_forward`. (Flagged Pass 1.)
- `normalization_and_activation.md` lines 164–169 vs. lines 227–234 (Activation intro block vs. Summary Table): The intro prose ("All three activation modules follow a common pattern…") partially duplicates the Summary Table. Consolidation deferred. (Flagged Pass 1.)

## VERDICT
- Crucial updates: no
