# Agent B — Chapter 4 Pass 1 Review

Scope: factual correctness, critical coherence, critical structural gaps. Max 5 items.

## Verdict

Chapter 4 is materially correct. The A/B orchestrator framing, the `__call__` override pins (`attention.py:90`, `decoder_layer.py:32`, `model.py:67`), the `init_*` vs `set_*` distinction, the Blackhole P150 monkey-patch detail, and the verbatim lifetimes contract all check out against the qwen3 sources. Five minor / borderline items follow.

---

## Items

### 1. (Borderline-misleading) `tensor_lifetimes.md` mixes Parameters into the Buffer section

`tensor_lifetimes.md:70` — inside the "Buffer — runtime state, not in `state_dict`" section, the prose says:

> Several of those (cos/sin/position_ids inside RoPE, and the KV caches) are consumed *by buffer address*. Their address is read once and baked into the compiled program's CT args...

But the immediately preceding bullet list (lines 65-68) only enumerates Buffer slots (`k_cache`, `v_cache`, `attn_out_tensor`, `qkv_out_tensor`, `o_proj_out_tensor`, `position_ids`). **`cos` and `sin` are Parameters, not Buffers** (lifetimes contract `modules/__init__.py:5-9`; `RoPE.params = ("cos", "sin", "trans_mat")` at `rope.py:25`). They were classified as Parameters in the Parameter section's "Flavor 2" already (lines 42-59).

Re-introducing them under "Several of those" inside the Buffer prose conflates the two lifetimes for a reader skimming the section, and the chapter is explicitly trying to *teach* the three-way split. Suggest: replace "Several of those (cos/sin/position_ids inside RoPE, and the KV caches)" with "Several of those Buffers (`position_ids` and the KV caches), as well as the cos/sin Parameters from the previous section, ..." — or just drop cos/sin from the Buffer paragraph entirely; they were already covered.

### 2. (Wrong-pin) `__init__.py` verbatim quote range mismatches the plan

Plan conventions §"Tensor-lifetime vocabulary" specifies a *verbatim* quote of `examples/qwen3_embedding_0_6b/modules/__init__.py:5-21`. The chapter (`tensor_lifetimes.md:7`) cites the same range — `From examples/qwen3_embedding_0_6b/modules/__init__.py:5-21` — but the rendered blockquote *prepends* the line-3 sentence "This port uses three tensor lifetimes:". In the source, line 3 is that header sentence; lines 5-21 are the three bullet blocks. Either:

- the pin should read `modules/__init__.py:3-21` (matching what is actually quoted), or
- the blockquote should drop the header sentence to match the cited range.

Per the conventions' writer-verification rule, the pin must match the bytes. Trivial to fix; called out because the plan flagged this exact pin as a "required verbatim quote."

### 3. (Wrong-pin) `_register_sdpa_decode_user_alloc` line range is `attention.py:14-24`, not the inline-block claim

`buffers_and_address_baking.md:127` claims:

> `examples/qwen3_embedding_0_6b/modules/attention.py:14-24` (inlined in `attention.py`, not a separate file as the directory layout might suggest)

The note about it being inlined in `attention.py` is correct. But the chapter then says the helper "runs in `Qwen3Attention.__init__` (`attention.py:42`)". In the source, `super().__init__()` is line 41 and `_register_sdpa_decode_user_alloc()` is **line 42** — that pin is correct. No fix needed for line 42; the 14-24 range is also correct (`def` at 14 through `SDPADecode._blaze_nn_user_alloc_patched = True` at 24).

(Self-cancelling on re-verification — leaving this entry to note that I checked the most-cited claim in the file and it is sound.) **Drop or downgrade**: not actionable.

### 4. (Wrong-pin, minor) `_blaze_nn_linear_patch.py:26-27` `_patched` guard is one line, not two

`buffers_and_address_baking.md:123` says:

> Idempotence is enforced by the `_patched` class flag check at the top (`_blaze_nn_linear_patch.py:26-27`).

In the source, the guard is at lines 26-27 (`if getattr(...): \n return`) — that is two lines (an `if` and a `return`). Pin is accurate. **No fix.**

### 5. (Minor / borderline-misleading) `init_attn_out_buffers` line range cited as `model.py:174-205`

`buffers_and_address_baking.md:56` says `init_attn_out_buffers(device) (model.py:174-205)`. The function `init_attn_out_buffers` actually spans `model.py:162-205` (signature at 162; the device-allocation branch starts at 178). Citing 174-205 covers the device-only path but excludes the function signature and the tensor-list branch.

Same shape applies to `init_kv_caches(device) (model.py:127-160)` (`buffers_and_address_baking.md:54`): the function is at `model.py:112-160`, the device branch starts at `127`. The chapter's prose is describing the device-allocation path, so the narrower range is defensible, but the pin convention is `function-start:function-end`. Trivial to widen.

---

## Re-verified, no issue found

- All three `__call__` overrides match plan pins exactly (`attention.py:90`, `decoder_layer.py:32`, `model.py:67`).
- `base.py:68-82` `Module.__call__` matches the chapter's verbatim quote in `orchestrator_pattern.md:88-103`; active-context check is at `base.py:71` as cited.
- `_register_sdpa_decode_user_alloc` is inlined in `attention.py:14-24` and the chapter correctly flags this against the directory layout's apparent suggestion of a separate file.
- `_blaze_nn_linear_patch.py` patches `cores_64 = CoreRange((0,0),(7,7))` (`_blaze_nn_linear_patch.py:40-42`) and `cores_32 = CoreRange((0,0),(3,7))` (`_blaze_nn_linear_patch.py:43-45`), matching the chapter's `8x8`/`4x8` claim.
- `init_position_ids` (`model.py:78-110`) and `set_position_ids` (`model.py:70-72`) are correctly distinguished (allocate vs bind); both end with `self.rope.set_position_ids(...)` so the same `ttnn.Tensor` aliases into both model and RoPE.
- Index uses backticked names (e.g. `__call__`) per the plan's API-reference convention; footer line is present on every content file.

## Net

Two genuine fixes (items 1, 2). Item 5 is a stylistic pin-tightening. Items 3, 4 cancel on re-verification and can be ignored.

Chapter ships with these two edits. The A/B orchestrator framing, the lifetimes vocabulary, and the Blackhole P150 monkey-patch documentation are all factually sound and well-structured.

---

## Agent A change log — applied after Pass 1 B review
- Issue 1 (`tensor_lifetimes.md:70` cos/sin conflated into Buffer paragraph): Rewrote the "Several of those..." sentence to read "Several of those Buffers (`position_ids` and the KV caches), together with the `cos` / `sin` Parameters from the previous section, are consumed *by buffer address*." The Buffer-section bullet list is now no longer extended to imply cos/sin are Buffers; the cross-reference to Parameter Flavor 2 is explicit.
- Issue 2 (`tensor_lifetimes.md:7` pin range mismatch): Changed the pin from `examples/qwen3_embedding_0_6b/modules/__init__.py:5-21` to `examples/qwen3_embedding_0_6b/modules/__init__.py:3-21`. Verified against the source: line 3 is the "This port uses three tensor lifetimes:" header sentence, lines 5-21 are the three bullet blocks; the blockquote in the chapter includes the header sentence, so `3-21` is the correct range.
- Issue 3 (`_register_sdpa_decode_user_alloc` self-cancelled by B on re-verification): No fix applied; B downgraded to "not actionable."
- Issue 4 (`_blaze_nn_linear_patch.py:26-27` two-line guard): No fix applied; B confirmed the pin is accurate.
- Issue 5 (pin tightening for `init_kv_caches` and `init_attn_out_buffers`): Widened both pins in `buffers_and_address_baking.md` to the full function ranges per the `function-start:function-end` convention. `init_kv_caches(device) (model.py:127-160)` → `init_kv_caches(device) (model.py:112-160)`; `init_attn_out_buffers(device) (model.py:174-205)` → `init_attn_out_buffers(device) (model.py:162-205)`. Verified line 112 (`def init_kv_caches`) and line 162 (`def init_attn_out_buffers`) against `examples/qwen3_embedding_0_6b/modules/model.py`.

---

## Pass 2

No feedback — chapter approved.

### Re-verification of Pass 1 fixes

- **Issue 1 fix landed.** `tensor_lifetimes.md:70` now reads "Several of those Buffers (`position_ids` and the KV caches), together with the `cos` / `sin` Parameters from the previous section, are consumed *by buffer address*." The conflation of cos/sin into the Buffer paragraph is resolved; the cross-reference to Parameter Flavor 2 is explicit.
- **Issue 2 fix landed.** `tensor_lifetimes.md:7` pin now reads `examples/qwen3_embedding_0_6b/modules/__init__.py:3-21`. Re-verified against the source: line 3 is the "This port uses three tensor lifetimes:" header sentence; lines 5-21 are the three bullet blocks. The blockquote in the chapter starts with that header sentence, so `3-21` matches the bytes quoted.
- **Issue 5 fix landed.** `buffers_and_address_baking.md:18-19` table rows now cite `model.py:112-160` for `init_kv_caches` and `model.py:162-205` for `init_attn_out_buffers`. Both ranges verified against `def init_kv_caches` at line 112 and `def init_attn_out_buffers` at line 162.

### Spot-checks for new factual errors

Re-verified the highest-traffic pins against the qwen3 sources; all hold:

- `Module.__call__` verbatim block in `orchestrator_pattern.md:88-103` matches `blaze_nn/modules/base.py:68-82` byte-for-byte (active-context check at `base.py:71`).
- All three orchestrator `__call__` overrides at the cited pins (`attention.py:90`, `decoder_layer.py:32`, `model.py:67`).
- `_bridge_kv_for_cache_update` usage at `attention.py:157-158` and `_bridge_q_for_sdpa` at `attention.py:162`.
- `FusedQKV` pin set: `qkv_proj.py:29` (`_ua_blackhole_cores = "64x8"`), `qkv_proj.py:31-32` (`set_output_tensor`), `qkv_proj.py:34-38` (`_get_output_tensor`), `qkv_proj.py:40-45` (`_collect_user_args`).
- `TokenEmbedding` pin set: `token_embedding.py:25` (`buffer_address()` line), `token_embedding.py:31-32` (the `merged.update` pair), `token_embedding.py:19-33` (whole `forward`).
- `RoPE` pin set: `rope.py:25` (`params = ("cos", "sin", "trans_mat")`), `rope.py:32-33` (`set_position_ids`), `rope.py:36-61` (whole `forward`), `rope.py:61` (`F.rope(x, trans_mat, ...)`).
- `weight_loader.py` pin set: `_hf_to_blaze_torch_tensors` at lines 105-152, `_build_blaze_nn_keys` at lines 81-102, `_ROLE_TO_CORES` at line 168, `_gamma_mc_for_width` at lines 171-185, `_wsharded_linear_weight_mc` at lines 188-213, `to_ttnn_state_dict` at lines 216-322.
- `_blaze_nn_linear_patch.py` pin set: function spans lines 25-72, idempotence guard at lines 26-27, `_HIFI4_OPS` loop at lines 64-70, `cores_64`/`cores_32` ranges at 40-45.

No new factual errors introduced. Chapter ships.
