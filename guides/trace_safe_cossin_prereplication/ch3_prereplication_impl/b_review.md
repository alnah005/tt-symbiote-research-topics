# B Review — Pass 1

Reviewed files:
- `index.md`
- `downstream_op_constraints.md`
- `replicated_mesh_mapping.md`
- `move_weights_impl_changes.md`

---

## Issues Found

### Issue 1 — `downstream_op_constraints.md`, Section 1, lines 11–13: inverted conditional logic

**What the text says:**

> If `TTNNRotaryPositionEmbedding.forward` calls `ttnn.unsqueeze` on cos/sin before passing them to `ttnn.experimental.rotary_embedding`, then the cos/sin argument arriving in `TTNNQwen3FullAttention.forward` has shape `[1, seq_len, rotary_dim]` (one fewer leading dimension), and the pre-allocated buffer must match that shape.

**Why it is wrong:**

The unsqueeze described here is an operation *internal* to `TTNNRotaryPositionEmbedding.forward` — it is applied to a local reference before that class calls `ttnn.experimental.rotary_embedding`. An internal op that RotaryPositionEmbedding applies to its own working copy before calling rotary_embedding does not change the shape of the tensor that RotaryPositionEmbedding *passes out* to `TTNNQwen3FullAttention.forward`. The shape arriving at FullAttention is determined solely by what RotaryPositionEmbedding forwards/returns, not by what it does internally before its own op calls.

As written, the premise ("RotaryPositionEmbedding calls unsqueeze before passing to rotary_embedding") and conclusion ("the argument *arriving in FullAttention.forward* has shape `[1, seq_len, rotary_dim]`") are not causally connected. The premise has no effect on the conclusion.

**Fix:**

Replace the conditional with one that is causally valid. The relevant question is what shape RotaryPositionEmbedding *produces and passes to FullAttention*, not what internal ops it applies before calling its own rotary_embedding. The corrected framing:

> If `TTNNRotaryPositionEmbedding` passes cos/sin to `TTNNQwen3FullAttention.forward` in 3D form (shape `[1, seq_len, rotary_dim]`) — for example because the unsqueeze is applied inside FullAttention just before the rotary_embedding call — then the pre-allocated buffer must use that 3D shape. If RotaryPositionEmbedding passes 4D tensors (shape `[1, 1, seq_len, rotary_dim]`), the pre-allocated buffer must use the 4D shape.

---

### Issue 2 — `downstream_op_constraints.md`, Section 1, line 13: TODO branches are backwards

**What the text says (TODO):**

> Confirm whether `TTNNRotaryPositionEmbedding.forward` calls `ttnn.unsqueeze` on cos/sin before the op call. If yes, the pre-allocated shape must be `[1, 1, rotary_dim]`; if no, it must be `[1, 1, 1, rotary_dim]` for decode.

**Why it is wrong:**

The "if yes" and "if no" branches are inverted relative to the actual shape consequence.

- If `ttnn.unsqueeze` is called on cos/sin (adding a dimension), the result has *more* dimensions than the input, not fewer. A 3D `[1, seq_len, rotary_dim]` tensor after unsqueeze becomes 4D `[1, 1, seq_len, rotary_dim]`. So "unsqueeze is called → shape arriving at FullAttention has an extra leading 1 → 4D → pre-allocated shape is `[1, 1, 1, rotary_dim]`."
- If `ttnn.unsqueeze` is *not* called, the cos/sin passed to FullAttention retain whatever base shape RotaryPositionEmbedding produces — if that is 3D `[1, seq_len, rotary_dim]`, then "no unsqueeze → 3D → pre-allocated shape is `[1, 1, rotary_dim]`."

The TODO as written has these exactly backwards: it maps "yes unsqueeze → 3D shape" and "no unsqueeze → 4D shape," which is the opposite of the correct inference.

The default value used throughout the chapter — `[1, 1, 1, rotary_dim]` (4D) — is consistent with the "unsqueeze has already been applied upstream" case, not the "no unsqueeze" case. The TODO's conditional should be corrected so that "yes" leads to the 4D shape and "no" leads to the 3D shape (or the TODO should be reframed in terms of what shape RotaryPositionEmbedding actually outputs, as described in Issue 1 above).

**Fix:**

> Confirm what shape cos/sin have when they arrive in `TTNNQwen3FullAttention.forward`. If they are 4D (shape `[1, 1, seq_len, rotary_dim]`), the pre-allocated buffer must be `[1, 1, 1, rotary_dim]` for decode — this is the default used in this chapter and is consistent with a 4D input. If they are 3D (shape `[1, seq_len, rotary_dim]`), the pre-allocated buffer must be `[1, 1, rotary_dim]`, and the Key Finding shape in Section 5 and the quick-reference table in `index.md` must be updated accordingly.

---

## Items Verified Correct

- Shape `[1, 1, 1, rotary_dim]` = `[1, 1, 1, 64]` for decode is stated consistently in `index.md` (quick-reference table and lifecycle diagram), `downstream_op_constraints.md` (Section 5 Key Finding), `replicated_mesh_mapping.md` (Section 1, 2, 5), and `move_weights_impl_changes.md` (Sections 1, 2, 4).
- dtype `ttnn.bfloat16` is stated consistently across all four files.
- Layout `ttnn.TILE_LAYOUT` is stated consistently; the rationale (required by `ttnn.experimental.rotary_embedding`, avoids trace-unsafe layout conversion) is correct.
- Memory config `ttnn.DRAM_MEMORY_CONFIG` is stated consistently; the rationale (persistent buffer, avoid L1 pressure) is correct.
- Mesh mapping `ReplicateTensorToMesh` is stated consistently; the rationale (each device needs the full 64-column table) is correct.
- Sharding arithmetic is correct: `rotary_dim / num_devices = 64 / 8 = 8` columns per device when sharded — matches the original crash description.
- Tile-padding arithmetic is correct: seq_len=1 padded to 32, giving effective shape `[1, 1, 32, 64]` and 4,096 bytes per device.
- Memory cost is correct: 128 bytes payload per device (64 × 2 bytes), consistent with domain fact.
- Trace-safety argument for `ttnn.copy` is correct: writes into the existing stable buffer at the destination's pre-allocated address; no new device buffer is allocated; the DMA command recorded during capture references the same stable address on every replay.
- Placement of `ttnn.copy` inside the trace bracket is correctly explained and the consequence of placing it outside (replay uses stale position-0 values) is correctly described in `move_weights_impl_changes.md` Section 3.
- Contrast between `_ensure_replicated` (trace-unsafe: calls `ttnn.from_torch` on sharded input, allocating a new device buffer) and `ttnn.copy` (trace-safe: writes into pre-allocated stable buffer) is accurate.
- `ttnn.clone` warning in `move_weights_impl_changes.md` Section 3 is correct: `ttnn.clone` allocates a new destination buffer and is trace-unsafe.
- `move_weights_to_device_impl` runs before trace capture; allocations there produce stable addresses — correctly stated in `index.md` lifecycle diagram Phase 1.
- The `ttnn.copy(src, dst)` argument order is used correctly in all code samples: `cos` is src, `self._cos_replicated` is dst.
- The Trace Invariant in `replicated_mesh_mapping.md` Section 2 (copying a sharded source into a replicated destination produces device-specific, non-identical values) is factually correct.

# B Review — Pass 2 (Change Log)

Changes applied in response to Pass 1:
1. `downstream_op_constraints.md` Section 1 (~lines 11–13): rewrote shape conditional and TODO — removed causally-broken claim (internal RotaryPositionEmbedding unsqueeze affecting FullAttention's argument shape); reframed TODO around what shape arrives at TTNNQwen3FullAttention.forward boundary; corrected direction: 4D arriving → 4D buffer (case a); 3D arriving with internal unsqueeze → 3D buffer (case b)

---

# B Review — Pass 2

Reviewed files (current state after Pass 1 fixes):
- `index.md`
- `downstream_op_constraints.md`
- `replicated_mesh_mapping.md`
- `move_weights_impl_changes.md`

---

## Issues Found

### Issue 1 — `downstream_op_constraints.md`, Section 1 TODO case (b) vs. Section 5 Key Finding: unanalyzed trace-unsafe op in the internal-unsqueeze scenario

**What the text says (Section 1 TODO, case b):**

> (b) `[1, 1, rotary_dim]` (3D) if the upstream omits one leading dimension and `TTNNQwen3FullAttention.forward` calls `ttnn.unsqueeze` internally before passing to the rotary op. The pre-allocated buffer must match whichever shape arrives — `[1, 1, rotary_dim]` for case (b).

**What the text says (Section 5 Key Finding):**

> If `_ensure_replicated` is the only transformation applied to cos/sin before they reach `ttnn.experimental.rotary_embedding`, then removing it and replacing it with `ttnn.copy` into a pre-allocated `TILE_LAYOUT` BF16 buffer satisfies all downstream constraints with no remaining trace-unsafe ops.

**Why this is a problem:**

Case (b) as described posits that `TTNNQwen3FullAttention.forward` calls `ttnn.unsqueeze` internally on cos/sin before passing them to `ttnn.experimental.rotary_embedding`. In that scenario:

1. `ttnn.copy(cos, self._cos_replicated)` writes the 3D incoming tensor into the 3D pre-allocated buffer. So far so good.
2. Inside `forward`, `ttnn.unsqueeze(self._cos_replicated)` (or equivalent) is then called to produce the 4D tensor that `ttnn.experimental.rotary_embedding` requires.
3. `ttnn.unsqueeze` on a TTNN device tensor allocates a new buffer — it does not reshape in-place the existing buffer at `self._cos_replicated`'s stable address.
4. That new buffer is allocated inside the trace bracket → trace-unsafe.

Section 5's Key Finding conditions its correctness on `_ensure_replicated` being "the only transformation." In case (b), `ttnn.unsqueeze` is an additional transformation inside the trace bracket and is trace-unsafe. The Key Finding's conditional is logically correct in isolation, but case (b) in Section 1 introduces a scenario that violates that condition without the guide flagging the implication: in case (b), the proposed fix is still incomplete because a trace-unsafe unsqueeze remains inside the bracket.

The guide presents case (b) as a valid scenario requiring only a different pre-allocated buffer shape, without noting that the shape fix alone does not resolve all trace-safety problems if an unsqueeze occurs inside `forward`.

**Fix:**

Extend the Section 1 TODO for case (b) to note the trace-safety consequence: if FullAttention calls `ttnn.unsqueeze` on the incoming cos/sin inside the trace bracket, that unsqueeze allocates a new buffer and is itself trace-unsafe. The fix in that scenario is not merely to use a 3D pre-allocated buffer — the unsqueeze must also be eliminated or moved outside the trace bracket (e.g., by pre-allocating the buffer at the already-unsqueezed 4D shape and pre-applying the unsqueeze once before trace capture). Add a cross-reference in the Section 5 Key Finding to clarify that the "only transformation" condition means no other allocating ops — including `ttnn.unsqueeze` — may appear between `ttnn.copy` and `ttnn.experimental.rotary_embedding`.

---

### Issue 2 — `downstream_op_constraints.md`, Section 5 Key Finding: hardcodes 4D shape as definitive while Section 1 leaves shape unconfirmed

**What the text says (Section 5 Key Finding):**

> The pre-allocated cos/sin buffer must be in `TILE_LAYOUT`, `DRAM_MEMORY_CONFIG`, `bfloat16`, with shape `[1, 1, 1, rotary_dim]` (decode; seq_len=1) and `ReplicateTensorToMesh` mapping.

**Why it is wrong:**

Section 1 of the same file contains an explicit TODO stating that the shape of cos/sin arriving at `TTNNQwen3FullAttention.forward` is unconfirmed, with two possible cases: 4D `[1, 1, 1, rotary_dim]` (case a) or 3D `[1, 1, rotary_dim]` (case b). The Key Finding in Section 5 presents the 4D shape as the definitive answer, which contradicts the open TODO in Section 1 of the same file. A reader who reads the Key Finding after the TODO is left with an unresolved contradiction: is the shape confirmed 4D or is it unconfirmed?

**Fix:**

Qualify the Section 5 Key Finding to reflect the conditional nature of the shape: state that the 4D shape `[1, 1, 1, rotary_dim]` applies if cos/sin arrive 4D at the `forward` boundary (case a from Section 1), and note that this assumption must be verified per the Section 1 TODO before the Key Finding can be treated as final. Alternatively, if the shape is in fact confirmed 4D, resolve the Section 1 TODO rather than leaving both an open TODO and a definitive Key Finding in the same file.

---

## Items Verified Correct

All items verified correct in Pass 1 remain correct in the current state. No new issues were found in `index.md`, `replicated_mesh_mapping.md`, or `move_weights_impl_changes.md`.

# B Review — Pass 3 (Change Log)

Changes applied in response to Pass 2:
1. `downstream_op_constraints.md` Section 1 TODO: added warning that case (b) with internal ttnn.unsqueeze is still trace-unsafe (unsqueeze allocates new buffer); correct design for case (b) is to pre-allocate 4D and use ttnn.reshape (view, no allocation) if shape adjustment is needed inside the trace bracket
2. `downstream_op_constraints.md` Section 5 Key Finding: qualified hardcoded [1,1,1,rotary_dim] shape to acknowledge the unresolved Section 1 TODO — shape is [1,1,1,rotary_dim] if no internal reallocating unsqueeze is present; see Section 1 TODO for the open shape question

---

## Pass 3

### Issues found: 0

None. Chapter is correct.

All four files were read in full and checked against the domain facts. The Pass 2 fixes are correctly applied:

- `downstream_op_constraints.md` Section 1 case (b) warning now correctly states that `ttnn.unsqueeze(self._cos_replicated)` inside the trace bracket allocates a new device buffer and is trace-unsafe, and correctly prescribes the alternative: pre-allocate the buffer as 4D `[1, 1, 1, rotary_dim]` and use `ttnn.reshape` (view, non-allocating) inside the trace if a shape adjustment is needed. This matches domain fact 10 exactly.

- `downstream_op_constraints.md` Section 5 Key Finding is now properly qualified: it states the shape is `[1, 1, 1, rotary_dim]` if cos/sin arrive as 4D, or if any internal `ttnn.unsqueeze` is replaced with a non-allocating alternative, and explicitly defers to the Section 1 TODO for the unresolved shape question. This matches domain facts 4 and 10.

- All claims verified correct in Pass 1 and Pass 2 remain correct: `ttnn.copy` trace-safety argument, `ReplicateTensorToMesh` rationale, DRAM_MEMORY_CONFIG rationale, TILE_LAYOUT requirement, dtype, tile-padding arithmetic, memory cost figures, placement of copy inside vs. outside the trace bracket, `ttnn.clone` warning.
