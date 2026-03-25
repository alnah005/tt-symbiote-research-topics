# Integration Roadmap

This document proposes two integration paths for bringing TT Transformers' decode-loop
capabilities into TT Symbiote, based on what is actually present in the source code.

---

## Option A: Minimal — wiring trace capture inside `TTNNModule`

### What this means

Leave HF `generate()` as the generation loop. Improve decode efficiency by making
Symbiote's `TracedRun` mode cover more of the decode graph, and fix the constraints that
currently prevent tracing the LLaMA-specific linears.

### What would need to change in `TTNNModule` / `TTNNLinear`

**Problem 1 — `@deallocate_weights_after` blocks tracing.**
`TTNNLinearLLama`, `TTNNLinearLLamaBFloat16`, and
`TTNNLinearLLamaIColShardedWRowSharded` carry `@deallocate_weights_after`, making them
incompatible with trace capture (see `symbiote_inference_path.md` section 2 for the
mechanism). Two sub-options:

- Keep weights resident on device for the duration of a traced session and remove
  `@deallocate_weights_after` from the forward methods of these classes for traced sessions.
  This requires a per-module "weights-stay-resident" flag that `TracedRun.configure` enables.
- Alternatively, accept that LLaMA-specific linears remain `@trace_disabled` and rely only
  on tracing `TTNNDistributedRMSNorm` and `TTNNLinear` (bfloat16 path) — the two
  `@trace_enabled` normalization/linear classes per the decorator table in
  `symbiote_inference_path.md` section 2. Note: the replacement dict in `test_llama.py`
  registers `TTNNRMSNorm` (not `TTNNDistributedRMSNorm`); only models that use
  `TTNNDistributedRMSNorm` would benefit from normalization tracing. The benefit is smaller
  but requires no weight-management change.

**Problem 2 — per-module trace granularity.**
`TracedRun` in `run_config.py` captures one trace per `(module_name, input_shape)` pair.
For a 32-layer LLaMA model this means up to ~128 individual traces (one per linear per
layer). Each re-enters Python on every decode step. This is better than no tracing but
cannot match TT Transformers' single full-model trace.

**Problem 3 — `_TRACE_RUNNING` re-entrancy guard.**
`TracedRun.module_run` sets `_TRACE_RUNNING = True` before tracing and restores it in a
`finally` block. Any `TTNNModule` called from within another module's `forward` while
`_TRACE_RUNNING` is True falls through to plain `forward`. This is correct for capture, but
it means nested modules are not individually traced. The current architecture traces only the
outermost `@trace_enabled` module in any call stack — which may be the attention block or
MLP as a whole rather than the individual linears.

**What `run_on_devices` does not change.**
The `run_on_devices` decorator in `module.py` raises `RuntimeError` if the device arch does
not match. This does not interact with trace capture; it is purely a guard on `forward`.

### Constraints that trace-disabled modules impose

Because `@trace_disabled` modules (`TTNNLinearLLama` etc.) fall through to plain execution
even inside a `TracedRun` session, any module that contains a `@trace_disabled` child will
itself not be fully captured. A parent module's trace would include the child's device
address at capture time, and freeing that address breaks the trace. Therefore, Option A
provides meaningful benefit only for model configurations that use the `TTNNLinear`
(bfloat16) path and avoid the `@deallocate_weights_after` variants.

---

## Option B: Full port — Symbiote-native `Generator` class

### What this means

Write a class (call it `SymbioteGenerator`) that mirrors the structure of
`Generator` from `models/tt_transformers/tt/generator.py` but operates on Symbiote's
replaced `nn.Module` graph rather than on a `Transformer` instance.

### What it needs

**Prefill/decode phases.**
`SymbioteGenerator` must call `model.forward` differently for prefill (full-length input,
build KV cache) and decode (single token, read KV cache). HF's `generate()` does not
distinguish these; a custom generation loop does.

**Paged KV cache.**
The current Symbiote `LlamaAttention` module already references paged-attention patterns
(see `modules/attention.py`). A `SymbioteGenerator` would allocate a `page_table` tensor
before prefill and pass it through each decode step, matching the
`TTNNPagedAttentionKVCache` interface expected by the attention layers.

**Full-model trace capture.**
Rather than relying on `TracedRun.module_run` per module, `SymbioteGenerator` would call
`ttnn.begin_trace_capture` / `ttnn.end_trace_capture` around a single `model.forward` call
with fixed-size inputs. This requires that all weights used during that call are resident on
device and that no Python-side branching changes between capture and execution. The
`@trace_disabled` modules with `@deallocate_weights_after` must be converted to keep weights
resident (same Problem 1 as Option A).

**Sampling.**
The decode loop would hold the output logits on device and apply temperature/top-p sampling
before copying the token ID to host. `models/common/sampling.py` already provides
`SamplingParams`; `SymbioteGenerator` would consume the same interface.

**Warmup.**
`SymbioteGenerator.warmup_model_prefill` would sweep the supported prefill sequence lengths
using the same `get_warmup_prefill_supported_seq_lens()` pattern.

### What existing Symbiote pieces it can reuse

- `TTNNModule.preprocess_weights()` and `move_weights_to_device()` — weight loading pipeline
  is already in place.
- `DistributedConfig` / `DistributedTensorConfig` — tensor sharding configuration is
  already managed by `run_config.py`.
- `TorchTTNNTensor` — the tensor wrapper can still be used for input preparation even if the
  decode loop bypasses `__torch_dispatch__` for the hot path.
- `LlamaAttention` in `modules/attention.py` — already includes `page_table` handling.
- `DispatchManager` — timing infrastructure can instrument the new loop.

---

## Recommended path

**Start with a targeted Option A to unblock tracing, then progress toward Option B.**

The rationale:

1. `TT_SYMBIOTE_RUN_MODE=TRACED` already exists and is wired to `TracedRun`. The
   infrastructure is present; the blocker is `@trace_disabled` on the LLaMA linears.
   Addressing `@deallocate_weights_after` compatibility is a contained change that delivers
   immediate benefit without restructuring the generation loop.

2. Option B requires building a custom generation loop that replicates KV cache management,
   paged attention allocation, and warmup — a substantial scope. It is the right long-term
   target, but attempting it before Option A risks a long integration cycle with no
   intermediate milestones.

3. The two options are not exclusive: a successful Option A validates that tracing works
   end-to-end for Symbiote modules and surfaces any additional incompatibilities (e.g.,
   modules that hold Python-side state that breaks replay) before committing to Option B.

---

## Concrete milestones

### Milestone 1 — Make LLaMA linears traceable (Option A prerequisite)

- **Target**: `TTNNLinearLLama`, `TTNNLinearLLamaBFloat16` in
  `models/experimental/tt_symbiote/modules/linear.py`.
- **Change**: Introduce a `weights_resident` mode that suppresses `deallocate_weights_impl`
  during a `TracedRun` session. `TracedRun.configure` sets this flag before capture.
- **Validation**: Run `test_llama.py` with `TT_SYMBIOTE_RUN_MODE=TRACED` and verify
  `TracedRun._trace_cache` entries are created for linear layers on the first decode step
  and reused on subsequent steps.

### Milestone 2 — Measure per-module trace coverage

- Instrument `DispatchManager` to count traced vs. non-traced module calls per decode step.
- Identify which modules still fall through to plain `forward` after Milestone 1 (likely
  the LlamaAttention block if it contains any `@trace_disabled` child).
- Target: all layers in the non-attention MLP stack traced; establish a baseline latency
  number.

### Milestone 3 — Separate prefill and decode phases

- **Target**: `test_llama.py` or a new test file.
- Replace `model.generate()` with a two-phase loop: call `model.forward` once with the full
  prompt, store the resulting KV cache tensors, then loop calling `model.forward` with a
  single token. This is a pure Python change that does not require new Symbiote
  infrastructure but establishes the loop structure that Option B builds on.
- **Reference**: `decode_forward` in `generator.py` for the phase-switch pattern
  (`mode_switched` / `reset_batch`).

### Milestone 4 — Allocate a static KV cache and pass a `page_table`

- **Target**: `models/experimental/tt_symbiote/modules/attention.py`.
- Pre-allocate a static KV cache tensor (shape `[num_layers, 2, max_seq_len, ...]`) before
  inference.
- Build a `page_table` tensor for single-user inference (shape `[1, num_blocks]`) and
  thread it through `LlamaAttention.forward`.
- This is the prerequisite for full-model trace capture: the KV cache address must not
  change between capture and execution.

### Milestone 5 — Full-model trace capture for the decode step

- **Target**: new `SymbioteGenerator` class or a standalone function.
- With a static KV cache and resident weights from Milestones 1 and 4, wrap one decode
  `model.forward` call in `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`.
- Use `copy_host_to_device` to copy new token and position data into the captured input
  buffers each step, then call `ttnn.execute_trace`.
- **Reference**: `_capture_decode_trace_text` and `_decode_forward_trace_text` in
  `generator.py` for the exact buffer-copy and execute pattern.

### Milestone 6 — Warmup sweep

- Add `warmup_model_prefill`-equivalent logic: iterate over supported prefill lengths and
  call the prefill forward once per length before accepting live requests.
- **Reference**: `warmup_model_prefill` in `generator.py`; prefill seq-len list from
  `model_args[0].get_warmup_prefill_supported_seq_lens()` (requires a `ModelArgs` equivalent
  or a hard-coded list for the Symbiote use case).

### Milestone 7 — On-device sampling (optional)

- If `models/common/sampling.py` `SamplingParams` and the on-device `sampling` module are
  available, wire them into `SymbioteGenerator`'s decode loop following the split-sampling
  pattern in `_capture_decode_trace_text` (`capture_sampling_trace=True`,
  `sampling_module.capture_trace`).
- Gate behind `_supports_on_device_sampling`, matching the pattern in
  `prefill_forward_text`.
