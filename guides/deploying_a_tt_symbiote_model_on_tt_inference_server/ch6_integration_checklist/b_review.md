## Pass 1

**1. T3K mis-labelled as "galaxy" — factually wrong device description (checklist.md, item 18)**

The comment `# eight-chip T3K galaxy` conflates two distinct products. T3K is an 8-chip board; Galaxy is a separate 32-chip product. A reader selecting hardware based on this label may configure the wrong topology or miscount required chip indices. The `TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` line itself is correct for 8 chips, but the label is wrong and will mislead anyone mapping device names to physical hardware.

---

**2. Pitfall 2 — contradictory "Wrong" label and inline comment misleads implementers (worked_example.md, lines 462–466)**

The first code block under "Pitfall 2" is labelled `**Wrong:**` but contains the comment `# torch.Tensor is on CPU — this is fine`. This is directly contradictory: a reader learns simultaneously that the block is wrong and that its result is fine. The block shows `return ttnn.to_torch(tt_logits)` without `.to(torch.float32)`. The correct answer — whether omitting `.to(torch.float32)` is acceptable — is never stated clearly. A reader may implement `prefill_forward` without `.to(torch.float32)` and believe it is correct, which may produce dtype mismatches in vLLM's sampling code. The comment must be removed or the label must be corrected to reflect which specific thing is wrong (returning a raw `ttnn.Tensor`, not the `ttnn.to_torch()` conversion).

---

**3. ImplSpec `module_path` in checklist.md item 5 does not match the worked example's directory layout (checklist.md item 5 vs worked_example.md directory structure)**

`checklist.md` item 5 shows a concrete, copy-pasteable `ImplSpec` with `module_path="models.my_model.my_model"`. The worked example's canonical directory layout uses `models/my_symbiote_model/my_symbiote_model.py`, giving `module_path="models.my_symbiote_model.my_symbiote_model"`. A reader following the checklist `ImplSpec` block verbatim against the worked example's layout will get a `ModuleNotFoundError` at server startup. The checklist code block should either use a clearly generic placeholder (e.g., `"models.<your_module>.<your_module>"`) or match the worked example's path exactly.

## Pass 3

**1. `worked_example.md` line 70 — T3K labeled "eight-chip galaxy" in copy-pasteable code comment**

The comment `# T3K eight-chip galaxy configuration` in the `_T3K_SPEC` block of `worked_example.md` repeats the same factual error as Pass 1 item 1 (which cited `checklist.md` item 18). T3K and Galaxy are distinct products with different chip counts (8 vs 32). This comment appears inside a code block that readers are instructed to copy verbatim into `workflows/model_spec.py`. A reader who internalises this label will hold the wrong mental model when selecting hardware, counting chips, or setting `TT_VISIBLE_DEVICES` for a Galaxy deployment.

---

**2. `worked_example.md` `allocate_kv_cache` — TILE_LAYOUT applied to shape `(num_blocks, block_size, num_kv_heads, head_dim)` fails when `num_kv_heads` is not a multiple of 32**

`worked_example.md` Step 3d allocates KV cache tensors with `layout=ttnn.TILE_LAYOUT` and shape `(num_blocks, block_size, num_kv_heads, head_dim)`. TTNN's TILE_LAYOUT tiles over the last two dimensions in 32×32 blocks, so both `num_kv_heads` and `head_dim` must be multiples of 32. For the 7B GQA model described in the chapter, `num_kv_heads` is typically 8 — not a multiple of 32. Using TILE_LAYOUT on such a shape will either raise a TTNN shape error at allocation time or produce silently mis-shaped tensors, corrupting KV cache reads during inference. A reader following this example for any GQA model with fewer than 32 KV heads will implement `allocate_kv_cache` incorrectly. ROW_MAJOR_LAYOUT (or explicit padding to tile boundaries) is required for this shape.

---

**3. `worked_example.md` Step 4 smoke test omits `--dtype bfloat16` that is present in the N300 `DeviceModelSpec.vllm_args`**

The N300 `DeviceModelSpec` defined in Step 1 includes `"--dtype", "bfloat16"` in `vllm_args`. The smoke test command in Step 4 launches vLLM without `--dtype bfloat16`. A reader running the smoke test verbatim will exercise a different dtype configuration (vLLM default, typically `auto` or `float16`) than the actual deployment. On TT hardware the dtype controls the weight loading and activation precision path; a mismatch can cause the smoke test to pass while the real deployment fails, or vice versa. The smoke test command should include `--dtype bfloat16` to match the registered spec.

---

## Pass 2

**1. `register_tt_models()` import path in checklist.md item 8 does not match the worked example's directory layout (checklist.md item 8 vs worked_example.md Step 2)**

`checklist.md` item 8 (line 142) shows `from models.my_model.my_model import TTMySymbioteModel`. The worked example's directory structure and Step 2 both use `from models.my_symbiote_model.my_symbiote_model import TTMySymbioteModel`. A reader following the checklist item 8 code block verbatim will get a `ModuleNotFoundError` at plugin import time. This is a separate and distinct error from the Pass 1 item 3 finding (which concerned the `ImplSpec.module_path` field in item 5; that field was in fact already correct in the current file). The incorrect path is in the `register_tt_models()` import statement in item 8.

---

**2. Pitfall 2 inline comment factually mis-describes the return type of `ttnn.to_torch()` (worked_example.md, Pitfall 2, first "Wrong" block)**

The comment on the `return ttnn.to_torch(tt_logits)` line reads `# missing .to(torch.float32); returns bfloat16 ttnn-backed tensor`. This is factually wrong: `ttnn.to_torch()` returns a `torch.Tensor`, not a `ttnn.Tensor`. The distinction matters because the actual failure mode of this "Wrong" pattern (missing `.to(torch.float32)`) is a dtype mismatch in vLLM's sampling code — not a `ttnn.Tensor` being returned. A reader who internalises the comment as written will have the wrong mental model of what `ttnn.to_torch()` does and may misdiagnose dtype errors at runtime. The comment should state that the result is a `torch.Tensor` in `bfloat16`, not a `ttnn.Tensor`.

## Pass 5

No feedback — chapter approved.

## Pass 6

**1. `checklist.md` item 6 T3K spec omits `--tensor-parallel-size 8` that appears in the worked example, causing single-chip execution on an 8-chip board**

`checklist.md` item 6 defines `T3K_SPEC` with `vllm_args=["--block-size", "64", "--max-num-seqs", "64"]`. The worked example's `_T3K_SPEC` includes an additional `"--tensor-parallel-size", "8"` entry. T3K is an 8-chip board; without `--tensor-parallel-size 8`, vLLM launches with a single executor and the model either fails at startup (if the TTNN graph requires a mesh) or runs incorrectly on one chip. A reader who constructs their T3K `DeviceModelSpec` from the checklist template will omit this flag and produce a broken or single-chip deployment.

---

**2. `worked_example.md` Step 4 expected response shows `"completion_tokens": 8` for a request with `max_tokens: 5`, making the example unverifiable**

The smoke test curl command in `checklist.md` item 22 (referenced by Step 4) sends `"max_tokens": 5`. The expected response JSON in Step 4 shows `"completion_tokens": 8` and `"total_tokens": 15`. A response with 8 completion tokens is impossible when `max_tokens` is 5. A reader validating their smoke test output against this example will either conclude their working server is broken (8 ≠ 5) or, if they get the correct ≤5-token response, believe the example is wrong and distrust it. The `completion_tokens` value must not exceed `max_tokens`; the example should use self-consistent numbers (e.g., `max_tokens: 10` in the request, or `"completion_tokens": 5` in the response).

---

## Pass 4

**1. `worked_example.md` `prefill_forward` docstring contradicts the required return shape, causing implementers to return a wrong-shaped tensor (worked_example.md, Step 3b)**

The `prefill_forward` docstring states: "Only the logit at the last valid token position for each sequence is used by vLLM; returning the full sequence of logits is wasteful but not incorrect." This is wrong. The contract defined in both `checklist.md` item 10 and the method's own return-type annotation requires a tensor of shape `(batch, vocab_size)`. Returning the full sequence of logits would produce shape `(batch, seq_len, vocab_size)` — a 3-D tensor where a 2-D tensor is expected. vLLM's sampler will fail or silently mis-index logits when given the wrong shape. A reader who follows the docstring's permissive claim and returns a 3-D tensor will implement `prefill_forward` incorrectly. The docstring must be corrected to state that the caller requires exactly `(batch, vocab_size)` and that returning the full sequence is incorrect, not merely wasteful.

## Pass 7

No feedback — chapter approved.
