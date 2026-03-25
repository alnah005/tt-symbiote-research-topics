# Agent B Review — Chapter 2: Weight Management and Precision — Pass 1

## Issues

**1. [symbiote_weight_pipeline.md], line ~167 — `TTNNLinearLLamaIColShardedWRowSharded` wrong parent class described**

The guide said this class combines `bfloat8_b` dtype "from the LLama family," implying it inherits from `TTNNLinearLLama`. The source (`linear.py` line 197) shows it inherits from `TTNNLinearIColShardedWRowSharded`, not from any LLama class. A developer reading the guide would construct the wrong class hierarchy when implementing or extending this variant. Fix: description updated to name the actual parent (`TTNNLinearIColShardedWRowSharded`) and to state that `bfloat8_b` is applied via an override in `move_weights_to_device_impl()`, not via inheritance from a LLama parent.

---

**2. [transformers_weight_pipeline.md], line ~36-42 — `ttnn.as_tensor` snippet hardcodes wrong `memory_config`**

The guide's representative `ttnn.as_tensor` call showed `memory_config=w1_w3_mem_config` unconditionally. The actual source (`mlp.py` lines 71-73) uses a conditional expression:

```python
memory_config=(
    ttnn.DRAM_MEMORY_CONFIG if args.is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config
),
```

A developer implementing weight loading from this snippet would use the wrong memory config for `w2` weights (non-galaxy) and for all weights on galaxy configurations. Fix: the snippet in the guide now matches the actual conditional from the source.

---

**3. [transformers_weight_pipeline.md], line ~181 — `ModelOptimizations.accuracy()` fidelity description is wrong for the common Llama/Mistral case**

The guide stated accuracy mode uses "HiFi4 fidelity for small models." The source (`model_config.py` lines 106-136) shows that for Llama 3, Mistral 7B, and Phi3-mini (the predominant small models), accuracy mode uses `HIFI2_FP16` for MLP fidelity, not HiFi4. HiFi4 is used only for models in neither the Llama/Mistral/Phi3-mini bucket nor the 70B+ bucket. A developer implementing a performance tuning configuration would apply the wrong fidelity setting for the most common models. Fix: description updated to correctly distinguish the three model-specific paths (Llama/Mistral/Phi3-mini → `HIFI2_FP16`; other small models → `HiFi4`; 70B+ → `LOFI`). Also added the `ModelOptimizations.performance()` exception for Qwen2.5-7B/VL-7B.

---

**4. [symbiote_weight_pipeline.md], line ~26 — `move_weights_to_device()` description omits the idempotency guard**

The guide said the base implementation "asserts that preprocessing has already run and that a device is set, then calls `move_weights_to_device_impl()`." The source (`module.py` lines 87-97) also includes an explicit `_weights_on_device` guard: it returns immediately if weights are already on the device, and only sets `_weights_on_device = True` and calls `move_weights_to_device_impl()` on the first call. Without knowing this guard exists, a developer implementing a new module phase or testing the lifecycle could incorrectly assume `move_weights_to_device_impl()` is called every time, leading to misunderstandings about double-move semantics. Fix: description updated to document the idempotency behavior.

---

**5. [transformers_weight_pipeline.md], line ~181 — `ModelOptimizations.performance()` description omits model-specific exception**

The guide described performance mode as using `bfloat4_b` for FF1/FF3 and `LOFI` math fidelity "by default" without mentioning any exceptions. The source (`model_config.py` lines 164-183) shows that Qwen2.5-7B and Qwen2.5-VL-7B use `bfloat8_b` MLP and `bfloat16` attention with `HiFi4` fidelity in performance mode because these models degrade under the standard high-performance settings. A developer relying on the description to select performance-mode dtype settings would get the wrong answer for these models. Fix: the performance-mode description now notes the Qwen2.5-7B/VL-7B exception.

---

## Change Log — Pass 1 fixes applied

- Fix 1 (`symbiote_weight_pipeline.md`): Corrected `TTNNLinearLLamaIColShardedWRowSharded` description to state that it inherits from `TTNNLinearIColShardedWRowSharded`, not from `TTNNLinearLLama`.
- Fix 2 (`transformers_weight_pipeline.md`): Updated the `ttnn.as_tensor` code snippet's `memory_config` argument from the hardcoded `w1_w3_mem_config` to the actual conditional expression from the source.
- Fix 3 (`transformers_weight_pipeline.md`): Rewrote the `ModelOptimizations.accuracy()` description to correctly specify `HIFI2_FP16` for Llama/Mistral/Phi3-mini, `HiFi4` for other smaller models, and `LOFI`/`bfloat4_b` for 70B+.
- Fix 4 (`symbiote_weight_pipeline.md`): Updated the `move_weights_to_device()` description to document the `_weights_on_device` idempotency guard.
- Fix 5 (`transformers_weight_pipeline.md`): Extended the `ModelOptimizations.performance()` description to note the Qwen2.5-7B/VL-7B exception.

---

# Agent B Review — Chapter 2: Weight Management and Precision — Pass 2

## Re-check of Pass 1 fixes

All 5 Pass 1 fixes were correctly applied and verified against the source files:

1. `TTNNLinearLLamaIColShardedWRowSharded` parent class description is now correct (`TTNNLinearIColShardedWRowSharded`), confirmed against `linear.py` line 197.
2. The `ttnn.as_tensor` `memory_config` snippet matches the actual conditional from `mlp.py` lines 71-73.
3. `ModelOptimizations.accuracy()` description now correctly names `HIFI2_FP16` for the Llama/Mistral/Phi3-mini branch, `HiFi4` for other small models, and `LOFI`/`bfloat4_b` for 70B+, confirmed against `model_config.py` lines 96-156.
4. The `move_weights_to_device()` idempotency guard (`_weights_on_device`) is now documented, confirmed against `module.py` lines 87-97.
5. The Qwen2.5-7B/VL-7B performance-mode exception is now mentioned, confirmed against `model_config.py` lines 164-183.

## New issues found

**1. [transformers_weight_pipeline.md], line ~183 — `ModelOptimizations.accuracy()` description omits `phi-4` from the `HIFI2_FP16` branch**

The post-Pass-1 description lists "Llama 3, Mistral 7B, and Phi3-mini" as the models using `bfloat8_b` attention and `HIFI2_FP16` MLP fidelity in accuracy mode. The source (`model_config.py` lines 107-112) also includes `phi-4` in this exact branch via `or base_model_name.startswith("phi-4")`. A developer porting phi-4 into Symbiote would read the guide, conclude that phi-4 falls in the "other smaller models → bfloat16 attention + HiFi4" bucket, and apply the wrong precision settings (bfloat16 instead of bfloat8_b for attention weights, HiFi4 instead of HIFI2_FP16 for MLP fidelity).

Fix: Add `phi-4` to the model list: "For Llama 3, Mistral 7B, Phi3-mini, and phi-4 models, uses `bfloat8_b` for attention/KV weights and `HIFI2_FP16` MLP fidelity."

---

**2. [symbiote_weight_pipeline.md], line ~11 (Phase 1 description) — base `preprocess_weights_impl()` described as calling `preprocess_weights_impl()` on children, but it actually calls `preprocess_weights()`**

The Phase 1 description stated: "The base implementation sets `_preprocessed_weight = True` and recursively calls `preprocess_weights_impl()` on child modules." The source (`module.py` lines 104-110) shows that the base `preprocess_weights_impl()` calls `child.preprocess_weights()` — the guarded public method — not `child.preprocess_weights_impl()` directly. This distinction matters: a developer writing a custom container module who follows the guide's description and calls `preprocess_weights_impl()` on children directly would bypass each child's idempotency guard, potentially causing double-conversion.

Fix: Change "recursively calls `preprocess_weights_impl()` on child modules" to "recursively calls `preprocess_weights()` on child modules (so each child's own idempotency guard is respected)."

## Verdict

Two new issues found and fixed. The chapter is approved after these corrections.

## Change Log — Pass 2 fixes applied

- Fix 1 (`transformers_weight_pipeline.md`): Added `phi-4` to the `ModelOptimizations.accuracy()` model list for the `bfloat8_b`/`HIFI2_FP16` branch, matching `model_config.py` lines 107-112.
- Fix 2 (`symbiote_weight_pipeline.md`): Corrected the Phase 1 description to state that the base `preprocess_weights_impl()` calls `child.preprocess_weights()` (with the guard), not `child.preprocess_weights_impl()` directly.

---

# Agent B Review — Chapter 2: Weight Management and Precision — Pass 3

## Re-check of Pass 2 fixes

Both Pass 2 fixes were correctly applied and verified against the source files:

1. (`transformers_weight_pipeline.md` line 183) `phi-4` is now listed alongside Llama 3, Mistral 7B, and Phi3-mini in the `ModelOptimizations.accuracy()` `bfloat8_b`/`HIFI2_FP16` branch, confirmed against `model_config.py` lines 107-112.
2. (`symbiote_weight_pipeline.md` line 11) The Phase 1 description now correctly states "recursively calls `preprocess_weights()` on child modules (so each child's own idempotency guard is respected)," confirmed against `module.py` lines 104-110.

## New issues found

**1. [symbiote_weight_pipeline.md], line ~197-198 — "practical rule" about `@trace_disabled` stated as absolute but is violated by `SmartTTNNLinearLLama` and `SmartTTNNLinearLLamaBFloat16`**

The guide states: "any module that calls `deallocate_weights()` inside or after `forward()` must be `@trace_disabled`." The source (`linear_intelligent.py` lines 82-104) shows that `SmartTTNNLinearLLama` and `SmartTTNNLinearLLamaBFloat16` both have `@deallocate_weights_after` on `forward()` but carry **no** `@trace_disabled` decorator. Neither is `SmartTTNNLinear` (the base) decorated. Because `SmartTTNNLinear` inherits from `TTNNLinear` (which is `@trace_enabled`), `is_trace_enabled()` returns `True` for these Smart variants — despite them deallocating weights after every forward pass.

A developer reading the guide's absolute rule would either (a) believe `SmartTTNNLinearLLama` is trace-disabled when it is actually trace-enabled per `is_trace_enabled()`, leading to incorrect assumptions about how these classes behave under `TracedRun`, or (b) when writing a new auto-deallocating Smart subclass, add `@trace_disabled` based on the rule and produce behavior inconsistent with the existing Smart variants.

Fix: Add a note to the practical rule that the current `SmartTTNNLinear*` variants are an exception: they carry `@deallocate_weights_after` but are not `@trace_disabled`, and are therefore treated as trace-enabled by `is_trace_enabled()`.

---

**2. [symbiote_weight_pipeline.md], line ~208 — `SmartTTNNLinear.__init__` described as attempting an unsafe device access at construction time**

The guide says `SmartTTNNLinear.__init__` "tries to access `self.device.compute_with_storage_grid_size()`" implying the constructor may fail if called before `to_device()`. The source (`linear_intelligent.py` line 14) shows an explicit guard: `self.grid_size = self.device.compute_with_storage_grid_size() if self.device else None`. At construction time `self.device` is `None`, so the guard short-circuits safely and `self.grid_size` is set to `None`. The constructor does not raise an error before `to_device()` is called.

A developer reading the guide might add unnecessary exception handling or believe the constructor is unsafe to call before device initialization, which is incorrect and would materially misrepresent when and how `SmartTTNNLinear` can be instantiated.

Fix: Reword to make clear that the access is guarded (`if self.device`), so construction before `to_device()` is safe by design.

## Verdict

Two new issues found and fixed. The chapter is approved after these corrections.

## Change Log — Pass 3 fixes applied

- Fix 1 (`symbiote_weight_pipeline.md`): Added a note to the "practical rule" paragraph that `SmartTTNNLinearLLama` and `SmartTTNNLinearLLamaBFloat16` are exceptions — they have `@deallocate_weights_after` but are not `@trace_disabled`, and are therefore trace-enabled per `is_trace_enabled()`.
- Fix 2 (`symbiote_weight_pipeline.md`): Rewrote the `SmartTTNNLinear.__init__` note to state that the device access is guarded with `if self.device`, making construction before `to_device()` safe by design.

---

# Agent B Review — Chapter 2: Weight Management and Precision — Pass 4

## Re-check of Pass 3 fixes

Both Pass 3 fixes were correctly applied and verified against the source files:

1. (`symbiote_weight_pipeline.md` line ~198) The exception note for `SmartTTNNLinearLLama` and `SmartTTNNLinearLLamaBFloat16` is present and accurate. Confirmed against `linear_intelligent.py` lines 82-104: both classes have `@deallocate_weights_after` on `forward()` and no `@trace_disabled` class decorator.
2. (`symbiote_weight_pipeline.md` line ~208) The `if self.device` guard note is present and accurately describes the construction-time behavior. Confirmed against `linear_intelligent.py` line 14.

## New issues found

**1. [reuse_vs_rewrite.md], line 25 — absolute rule for `@trace_disabled` contradicts the known `SmartTTNNLinear*` exception**

The guide states: "The only constraint is that the class must also be marked `@trace_disabled`." This is incorrect as an absolute rule. The source (`linear_intelligent.py` lines 82-104) shows that `SmartTTNNLinearLLama` and `SmartTTNNLinearLLamaBFloat16` both use `@deallocate_weights_after` without `@trace_disabled`, and are therefore trace-enabled per `is_trace_enabled()`. The Pass 3 fix in `symbiote_weight_pipeline.md` already documents this exception, but `reuse_vs_rewrite.md` was not updated in tandem, leaving conflicting guidance in the same chapter.

A developer reading only `reuse_vs_rewrite.md` as a quick-reference guide would add `@trace_disabled` to every new auto-deallocating class, including deliberate Smart-style subclasses where that is not the intended pattern, causing those modules to be excluded from trace capture.

Fix: Reworded to describe `@trace_disabled` as the typical case for auto-deallocating classes, while noting the `SmartTTNNLinear*` exception, consistent with the explanation already present in `symbiote_weight_pipeline.md`.

## Verdict

One new issue found and fixed. The chapter is approved after this correction.

## Change Log — Pass 4 fixes applied

- Fix 1 (`reuse_vs_rewrite.md`): Replaced the absolute rule "The only constraint is that the class must also be marked `@trace_disabled`" with qualified guidance that documents both the typical case and the `SmartTTNNLinearLLama`/`SmartTTNNLinearLLamaBFloat16` exception, consistent with `symbiote_weight_pipeline.md`.

---

# Agent B Review — Chapter 2: Weight Management and Precision — Pass 5

## Re-check of Pass 4 fix

The Pass 4 fix was correctly applied. `reuse_vs_rewrite.md` line 25 now reads: "In the typical case (as with `TTNNLinearLLama` and `TTNNLinearLLamaBFloat16`), a class that auto-deallocates weights should also be marked `@trace_disabled`..." followed by the explicit `SmartTTNNLinearLLama`/`SmartTTNNLinearLLamaBFloat16` exception note. This is consistent with `symbiote_weight_pipeline.md` and confirmed against `linear_intelligent.py` lines 82-104 (no `@trace_disabled` on either Smart class) and `linear.py` lines 179-193 (both `TTNNLinearLLama` and `TTNNLinearLLamaBFloat16` are `@trace_disabled`).

## New issues found

No feedback — chapter approved.

All material factual claims verified against source files for this pass:

- `symbiote_weight_pipeline.md` Phase 1 guard code snippet matches `module.py` lines 79-85 exactly.
- `TTNNLinear` bfloat16 for both weight and bias confirmed against `linear.py` lines 60-63.
- `TTNNLinearLLama` bfloat8_b, `@trace_disabled`, `@deallocate_weights_after` all confirmed against `linear.py` lines 179-193.
- `TTNNLinearLLamaIColShardedWRowSharded` parent class, bfloat8_b override, `@trace_disabled`, and `@deallocate_weights_after` confirmed against `linear.py` lines 196-222.
- `is_trace_enabled()` logic (instance-of enabled AND NOT instance-of disabled) confirmed against `run_config.py` line 900. The guide simplifies the cached-tuple implementation but is functionally equivalent.
- `SmartTTNNLinear` decode threshold `<= 32` confirmed against `linear_intelligent.py` line 32.
- `SmartTTNNLinear` `if self.device` guard for `grid_size` confirmed against `linear_intelligent.py` line 14.
- `SmartTTNNLinearLLama` bfloat8_b override and `@deallocate_weights_after` confirmed against `linear_intelligent.py` lines 85-95.
- `SmartTTNNLinearLLamaBFloat16` bfloat16 (inherited) and `@deallocate_weights_after` confirmed against `linear_intelligent.py` lines 98-104.
- `ttnn.as_tensor` `memory_config` conditional (`DRAM_MEMORY_CONFIG if is_galaxy else w2_mem_config if "w2" in name else w1_w3_mem_config`) confirmed against `mlp.py` line 72.
- `ModelOptimizations.accuracy()` Llama/Mistral/Phi3-mini/phi-4 branch (`bfloat8_b` WQKV/KV_CACHE/WO, `HIFI2_FP16` FF1_FF3 and FF2) confirmed against `model_config.py` lines 107-136.
- `ModelOptimizations.accuracy()` 70B+ branch (`bfloat4_b` FF1/FF3, `LOFI`) confirmed against `model_config.py` lines 96-105.
- `ModelOptimizations.performance()` Qwen2.5-7B/VL-7B exception (`bfloat16` attention, `HiFi4`) confirmed against `model_config.py` lines 164-184.
- `wqkv` sharding dims `(3, 2) if self.TG else (2, 3)` confirmed against `attention.py` line 250.
- `create_dram_sharded_mem_config` DRAM core counts (12 WH, 8 P150, 7 P100) confirmed against `model_config.py` line 2959.
- `deallocate_weights_after` decorator code snippet confirmed against `module.py` lines 238-247.
- `move_weights_to_device()` idempotency guard (`_weights_on_device`) confirmed against `module.py` lines 87-97.

## Verdict

Chapter approved. No new issues found in Pass 5.

## Change Log — Pass 5 fixes applied

No fixes required.

---

# Agent B Review — Chapter 2: Weight Management and Precision — Post-Compression Pass

## Post-compression check (3 changed areas)

**1. `@trace_disabled`/`deallocate_weights_after` rule collapsed to one location in `symbiote_weight_pipeline.md` with cross-references in other files**

Applied cleanly with no new errors. The canonical explanation lives at `symbiote_weight_pipeline.md` lines 173-198 and is complete: it includes the decorator definitions, the `is_trace_enabled()` logic, the practical rule, and the `SmartTTNN*` exception. The cross-reference in `reuse_vs_rewrite.md` line 25 ("see `symbiote_weight_pipeline.md` for the full rationale and the `SmartTTNN*` exception") is precise enough that a developer can locate the canonical section. The simplified `is_trace_enabled()` code snippet in the guide omits the tuple-caching optimization present in `run_config.py` lines 895-900, but is functionally equivalent; this was already accepted in Pass 5.

**2. "Dummy weights mode" standalone section removed from `transformers_weight_pipeline.md` — `cache_file_name=None` behavior**

Applied cleanly with no new errors. The `cache_file_name=None` behavior is still fully documented inline in `transformers_weight_pipeline.md` lines 53-60: the `args.dummy_weights` guard is shown, the `cache_name = lambda _: None` idiom is present, and the sentence explaining that `ttnn.as_tensor` with `cache_file_name=None` converts the tensor without reading or writing a cache file is retained. The concept is also summarized in `reuse_vs_rewrite.md` lines 27-29 under the Reuse section. No information loss.

**3. `ShardTensor2dMesh` dims-tuple code block removed from `reuse_vs_rewrite.md` — cross-reference clarity**

Applied cleanly with no new errors. `reuse_vs_rewrite.md` line 70 reads: "The full dimension-selection logic (`w1_dims`, `w2_dims`, and the `Attention` `wqkv` variants) is shown in `transformers_weight_pipeline.md` under `ShardTensor2dMesh`." The target section (`transformers_weight_pipeline.md` lines 82-98) contains all three code blocks — the `w1_dims`/`w2_dims` galaxy conditional and the `wqkv` dims snippet — and is reachable from the named anchor. A developer following the cross-reference will find the information.

## New issues found

No feedback — chapter approved.

All compression changes verified against source files for this pass:

- `@trace_disabled`/`deallocate_weights_after` canonical rule in `symbiote_weight_pipeline.md` line 198 confirmed against `linear.py` lines 179-222 (both `TTNNLinearLLama` and `TTNNLinearLLamaIColShardedWRowSharded` are `@trace_disabled` with `@deallocate_weights_after`) and `linear_intelligent.py` lines 82-104 (`SmartTTNNLinearLLama` and `SmartTTNNLinearLLamaBFloat16` have `@deallocate_weights_after` but no `@trace_disabled`, confirming the exception is accurate).
- `cache_file_name=None` behavior in `transformers_weight_pipeline.md` lines 53-60 consistent with TT Transformers idiom. No source file in this repo contradicts the description.
- `ShardTensor2dMesh` dims logic in `transformers_weight_pipeline.md` lines 82-98 (`w1_dims`, `w2_dims`, `wqkv` dims) verified as still present and correct.
- Cross-reference anchor name "ShardTensor2dMesh" in `reuse_vs_rewrite.md` matches the section heading at `transformers_weight_pipeline.md` line 70 exactly.

## Verdict

Chapter approved. No new issues introduced by the three compression changes.

## Change Log — Post-Compression fixes applied

No fixes required.
