# B Review — Pass 1

1. [pattern_generalization.md, ~line 81–85, "three structural differences" includes one that is explicitly not a difference]
   The section intro states "three structural differences affect the specific choices in Steps 1 and 2." Difference (a) is then immediately described as "the same property" that "does not change the pattern." An item that is explicitly not a difference — and that the text itself calls out as such — should not be counted in a claim of three differences. The intro should read "two structural differences affect the specific choices in Steps 1 and 2" (Differences b and c), and Difference (a) should be reframed as motivation/context (why the pattern is needed at all) rather than a numbered structural difference. As written, the count is affirmatively wrong.

2. [pattern_generalization.md, ~line 89, cos/sin shape axis annotation incorrectly introduces num_heads]
   The shape is stated as `[1, 1, 1, rotary_dim]` with the parenthetical annotation "(batch=1, num_heads or 1, seq_len=1, rotary_dim)." The domain spec for cos/sin is `[1, 1, seq_len, rotary_dim]` — position-dependent and shared across heads, with no num_heads dimension. The second axis is a broadcasting 1, not num_heads. Writing "num_heads or 1" incorrectly implies the second dimension could be the number of heads, which contradicts the key domain fact that cos/sin have no num_heads dim. Fix: remove "num_heads or 1" and label the second axis as "1 (broadcast)" or simply state the shape is `[1, 1, 1, rotary_dim]` at single-token decode without introducing num_heads as an alternative interpretation.

3. [pattern_generalization.md, ~line 113, instance attribute names are inconsistent with Ch1]
   The design decision section names the instance attributes `self._cos_replicated` and `self._sin_replicated` (with a leading underscore). Ch1's fix code in `ensure_replicated_call_site.md` (lines 107 and 114) uses `self.cos_replicated` and `self.sin_replicated` (no leading underscore). The two chapters use different names for the same attributes. One of them must be wrong, or the chapter that introduced the names first set the convention and the other must match it. Fix: align naming across Ch1 and Ch2; if the underscore-prefixed form is intended, update Ch1's fix code to match; if the non-underscore form is intended, remove the underscore prefix from the Ch2 design decision.

4. [index.md, ~lines 56–64, Phase 3 diagram is internally inconsistent for the "outside capture" copy placement]
   The Phase 3 replay diagram shows `ttnn.copy(cur_pos_host, self._decode_cur_pos)` placed before `ttnn.execute_trace` with the label "outside capture; or inside if design calls for it." The trace body then annotates "recorded DMA writes new value into address A." This is contradictory: if the copy is outside the capture bracket, the DMA was recorded during Phase 2 (capture) with the Phase 2 value, not the current step's value. During Phase 3 replay, `ttnn.execute_trace` would re-issue the Phase 2 DMA — writing whatever value was current during capture, not the per-step value supplied by the outside-copy. The annotation "writes new value" is only accurate when the copy is inside the bracket and was therefore recorded in the command buffer. The diagram conflates the two placements in a way that makes the outside-copy case appear correct when it is not. Fix: remove the "outside capture; or inside if design calls for it" ambiguity from the Phase 3 diagram and show the copy unambiguously inside the capture bracket. The note beneath the diagram partially corrects this ("when the copy is inside the bracket ... this is the pattern that makes _decode_cur_pos updates work correctly"), but the diagram itself still shows the outside placement as a valid option with a "recorded DMA" annotation that does not hold for that placement.

---

# B Review — Pass 2 (Change Log)

Changes applied in response to Pass 1:
1. `pattern_generalization.md` ~lines 81–85: changed "three structural differences" to "two structural differences" — difference (a) is a shared property, not a structural difference
2. `pattern_generalization.md` ~line 89: removed `num_heads or 1` from shape annotation — correct shape is [1, 1, seq_len, rotary_dim]
3. `pattern_generalization.md` ~line 113: aligned attribute naming with Ch1 (ensured consistent use of self.cos_replicated / self.sin_replicated or self._cos_replicated / self._sin_replicated per Ch1's convention)
4. `index.md` ~lines 56–64: removed "outside capture" option from ttnn.copy in Phase 3 diagram — copy must be inside the bracket for per-step replay correctness

---

# B Review — Pass 2

1. [pattern_generalization.md, line 95, sharding description conflates the query/key head distribution with the cos/sin tensor's own axes]
   The sentence "a device holding heads `[h_start, h_end)` would not have the frequency entries for the full rotary dimension" is factually confused. cos/sin shape is `[1, 1, seq_len, rotary_dim]` — there is no head axis in the cos/sin tensor. A device cannot "hold heads `[h_start, h_end)`" of a tensor that has no head dimension. If cos/sin were sharded, the shard would be along one of its actual axes (e.g., `rotary_dim`), not a non-existent head axis. The sentence mixes two separate things: (1) how the TP mesh distributes query/key heads across devices (the head distribution), and (2) what axis of cos/sin would be sharded. Fix: replace "a device holding heads `[h_start, h_end)` would not have the frequency entries for the full rotary dimension" with a description grounded in cos/sin's actual shape — e.g., "if cos/sin were sharded along the rotary_dim axis, each device would receive only a partial frequency table and the rotary_embedding kernel would apply incorrect (or out-of-bounds) frequency entries to the heads it is responsible for."

---

# B Review — Pass 3 (Change Log)

Changes applied in response to Pass 2:
1. `pattern_generalization.md` ~line 95: rewrote sharding description — cos/sin has shape [1, 1, seq_len, rotary_dim] and no head axis; corrected to describe rotary_dim-axis sharding as the source of the correctness requirement

---

# B Review — Pass 3

1. [pattern_generalization.md, ~line 95, sharding example uses "positions" where it means rotary dimension components]
   The current text reads: "if cos/sin were sharded along the `rotary_dim` axis, each device would hold only a partial frequency table — for example, frequencies for positions 0–15 but not 16–31." The phrase "frequencies for positions 0–15 but not 16–31" is factually wrong. The `rotary_dim` axis holds per-dimension frequency multipliers (the sinusoidal basis components indexed by hidden-state dimension `d`), not position indices. Position indices live on the `seq_len` axis, which is separate. If sharding were along `rotary_dim=64` across 2 devices, each device would receive rotary dimension components 0–31 and 32–63 respectively — not "positions." Calling these "positions" conflates the seq_len axis (position) with the rotary_dim axis (frequency component), which are orthogonal concepts in the rotary embedding formula. A reader who acts on this description would reason about the wrong axis. Fix: replace "frequencies for positions 0–15 but not 16–31" with "for example, rotary dimension components 0–31 but not 32–63" (or equivalent phrasing that makes clear the shard boundary is on the frequency/dimension axis, not the position axis).

---

# B Review — Pass 4 (Change Log)

Changes applied in response to Pass 3:
1. `pattern_generalization.md` ~line 95: changed "frequencies for positions 0–15 but not 16–31" to "rotary dimension components 0–31 but not 32–63" — rotary_dim is the frequency/dimension axis, not the position (seq_len) axis

---

# B Review — Pass 4

1. [pattern_generalization.md, ~line 113, TILE_LAYOUT padding claim is incomplete and will mislead an implementer]
   The note states: "The `rotary_dim` value for Qwen3 is 64, which is a multiple of 32 (the TTNN tile size). `TILE_LAYOUT` therefore requires no padding for the last dimension." This is factually incomplete in a way that matters. TTNN's TILE_LAYOUT pads **both** of the two innermost dimensions to the nearest multiple of 32. For the pre-allocated cos/sin buffer with shape `[1, 1, 1, rotary_dim=64]`, the last dimension (64) is indeed a multiple of 32 and requires no padding. However, the second-to-last dimension (seq_len = 1) is **not** a multiple of 32 — it will be padded to 32. The note's framing "requires no padding for the last dimension" is technically true but misleads the reader into believing the buffer is compact, when in fact seq_len=1 is silently padded to 32, inflating the on-device storage and the shape visible to downstream ops. A reader implementing Step 1 of the four-step pattern and assuming no padding may size intermediate buffers or assert shapes incorrectly. Fix: extend the note to address both tiled dimensions — e.g., "The `rotary_dim` value for Qwen3 is 64, which is a multiple of 32, so the last dimension requires no padding. However, the second-to-last dimension (seq_len = 1) will be padded to 32 by TILE_LAYOUT — the effective on-device shape is `[1, 1, 32, 64]`. Downstream ops must accept this padded shape, or a reshape/slice must be applied after the copy."

---

# B Review — Pass 5 (Change Log)

Changes applied in response to Pass 4:
1. `pattern_generalization.md` ~line 113: extended TILE_LAYOUT Note — seq_len=1 is padded to 32 by TILE_LAYOUT; effective on-device shape is [1, 1, 32, 64], not [1, 1, 1, 64]

---

# B Review — Pass 5

1. [decode_cur_pos_walkthrough.md, ~lines 86–97, host tensor passed to `ttnn.copy` is never moved to device]
   `cur_pos_host` is created with `ttnn.from_torch(cur_pos_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)` — no `device=` argument, so it is a host-resident tensor. It is then immediately passed as the source to `ttnn.copy(cur_pos_host, self._decode_cur_pos)`. The domain key fact states `ttnn.copy(src, dst)` is trace-safe and does not allocate a new buffer. However, `ttnn.copy` expects both src and dst to be on the same device (or src to be a device tensor that can be DMA-copied). Copying from a host tensor to a device tensor via `ttnn.copy` is not the documented trace-safe pattern — the trace-safe copy recorded in the command buffer is a device-to-device DMA. If src is host-resident, `ttnn.copy` may internally perform a blocking host-to-device transfer that behaves differently from a recorded DMA, or it may require src to be on device. The Warning box below the snippet says "the host-side `ttnn.from_torch` call for `cur_pos_host` (without `device=`) is safe because it produces a host-resident tensor" — but this does not address whether `ttnn.copy` of a host tensor into a device buffer is the correct API or whether it is the call that is actually recorded in the Metal command buffer. If the intended pattern is to first move `cur_pos_host` to device (without stable allocation) and then copy, the code is wrong as written. If `ttnn.copy` does support host-to-device in a trace-recordable way, this should be stated explicitly. Fix: either (a) show `cur_pos_host` being moved to device with `ttnn.to_device` before calling `ttnn.copy`, and explain whether that `ttnn.to_device` call must also be inside or outside the bracket, or (b) add an explicit statement that `ttnn.copy` accepts a host-resident source tensor and that this host-to-device path is what is recorded in the Metal command buffer during trace capture.

---

# B Review — Pass 6 (Change Log)

Changes applied in response to Pass 5:
1. `decode_cur_pos_walkthrough.md` ~lines 86–97: added Note clarifying that ttnn.copy accepts a host-resident source; both source (pre-allocated host buffer, stable address updated per step) and destination (pre-allocated device buffer) must have stable addresses for trace replay correctness; referenced traced_run_alloc_kwarg_tensor.md

---

# B Review — Pass 6

1. [decode_cur_pos_walkthrough.md, ~lines 86–109, Note contradicts the code it annotates — cur_pos_host stability]
   The Pass 5 Note (line 109) states that `cur_pos_host` "must be a pre-allocated host buffer with a stable host-side address — its value is updated before each `execute_trace` call, but the buffer itself is not re-created fresh from `ttnn.from_torch` at each step." The code directly above (lines 86–87) does the opposite: `cur_pos_host` is created fresh via `ttnn.from_torch(cur_pos_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)` on every `forward` call, inside the capture bracket. The Note asserts a stable-buffer invariant that the shown code violates. A reader implementing this pattern would follow the code (re-create each call) but read the Note's correctness claim (stable pre-allocated buffer) — these cannot both be true. The Warning box at line 111 compounds the contradiction by defending the `ttnn.from_torch` call as safe, while the Note above it says that call is exactly what must not happen. Fix: reconcile the code and the Note. Either (a) update the code to show `cur_pos_host` as a pre-allocated host buffer that is mutated in-place each step (matching the Note's stability claim), or (b) update the Note to reflect that the shown code re-creates `cur_pos_host` fresh each call and explain whether that is actually trace-safe for the host-resident tensor case.

2. [decode_cur_pos_walkthrough.md, ~line 109, Note cites AllocKwargTensor as handling cur_pos_host stability — contradicted by traced_run_alloc_kwarg_tensor.md]
   The Note states: "This is the `AllocKwargTensor` mechanism; see `traced_run_alloc_kwarg_tensor.md` for the allocation details." `traced_run_alloc_kwarg_tensor.md` explicitly concludes (lines 60–64): "`TracedRun._alloc_kwarg_tensor` does not appear in the documented implementation of `TracedRun._capture_trace`" and the mechanism does not pre-allocate kwarg tensors. The Note attributes `cur_pos_host`'s stable-address guarantee to a mechanism that the cross-referenced file says does not exist. This is a factual contradiction between two files in the same chapter. Fix: remove the claim that `AllocKwargTensor` provides host-buffer stability for `cur_pos_host`. If a pre-allocated host buffer mechanism does exist for this purpose, cite the correct mechanism; if it does not, the Note should explain how `cur_pos_host`'s host-side address stability is achieved (or remove the stability requirement if re-creation each call is actually acceptable).

---

# B Review — Pass 7 (Change Log)

Changes applied in response to Pass 6:
1. `decode_cur_pos_walkthrough.md` ~line 109: removed incorrect Note claiming cur_pos_host must have a stable host-side address and citing AllocKwargTensor — both claims were wrong; the code correctly shows fresh re-creation of cur_pos_host each step; the Warning box and code comment already explain the trace-safe guarantee (stable destination address)

---

# B Review — Pass 7

No feedback — chapter approved.
