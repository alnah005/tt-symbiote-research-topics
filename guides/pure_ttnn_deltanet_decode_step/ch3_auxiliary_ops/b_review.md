# B Review — Chapter 3: Causal Conv1D and Gated RMSNorm Without Host Readback

## Pass 1

### Issues found: 3

---

**Issue 1 — `causal_conv1d_update_ttnn.md`, Section 2a, the note after the slice+concat code block**

**Error:** The note states "`ttnn.slice` is a view/metadata operation that does not allocate a new device buffer." This is stated as a fact, but it is incorrect. `ttnn.slice` on TT-Metalium is generally a copy operation that writes into a new device buffer; it is not guaranteed to be a zero-copy metadata view. The same guide series acknowledges this uncertainty in `partial_rotary_non_tile_aligned_numerics/ch4_implementation_strategies/strategy_a_slice_apply_concat.md` (line 99): "`ttnn.slice` or indexing to extract `x_rot` — this may or may not allocate a new buffer depending on TTNN's implementation; in the worst case it does."

The incorrect characterization is consequential here because it is used to reason about trace safety. The note then says both ops are "trace-compatible as long as `conv_state_new` is written into a pre-allocated output buffer." The second half of that sentence (the `ttnn.concat` output needing a pre-allocated buffer) is the correct trace-safety concern. The first half (slice is zero-copy, so it is free from trace concerns) is unsupported. If `ttnn.slice` allocates a new buffer, then `shifted` is also a runtime allocation inside the trace bracket, which is itself trace-unsafe unless pre-allocated.

**Correction:** Replace the note with accurate language:

> **Note:** `ttnn.slice` writes into a new device buffer (it is not a zero-copy view on TT-Metalium). `ttnn.concat` allocates the combined output buffer. For trace compatibility, both `shifted` and `conv_state_new` must either use pre-allocated output buffers or be placed outside the trace bracket. The persistent state is handled by `ttnn.copy` into `conv_state_persistent` in Section 4; the intermediate `shifted` buffer must similarly be pre-allocated if this sequence runs inside a Metal Trace bracket.

---

**Issue 2 — `causal_conv1d_update_ttnn.md`, Section 4, trace-compatibility claim for `ttnn.slice` inside the traced function**

**Error:** Section 4 correctly explains that `ttnn.copy` into a pre-allocated DRAM buffer makes the state write trace-safe. However, it does not address the trace-safety of the `ttnn.slice` call (for `shifted`) and the `ttnn.concat` call (for `conv_state_new`) that occur inside the same dispatch sequence. If the full decode step runs under Metal Trace, those allocations must also be stable. The section presents the `ttnn.copy` pattern as fully resolving trace compatibility, but it only resolves the state persistence concern — the intermediate allocations inside `causal_conv1d_decode_update_ttnn` are not addressed.

This is a gap rather than an internal contradiction, but it is a material omission given that trace compatibility is a stated key invariant for this chapter.

**Correction:** Add a note after the `ttnn.copy` code block clarifying that the intermediate buffers (`shifted` from `ttnn.slice`, and `conv_state_new` from `ttnn.concat`) also require pre-allocation for full Metal Trace compatibility. The complete trace-safe pattern pre-allocates a `shifted_buf` of shape `[B, channels_local, K-1]` and a `conv_state_new_buf` of shape `[B, channels_local, K]` in the init path (outside the trace bracket), then passes them as output buffers to the respective ops. Only after this is the entire sequence trace-safe — not just the `ttnn.copy` step.

---

**Issue 3 — `gated_rmsnorm_ttnn.md`, Section 3, peak simultaneous L1 occupancy**

**Error:** The section states "Peak simultaneous occupancy is `x`, `z`, and `x_normed` in L1 at Step 2 entry: three tensors." The weight tensor `w_norm` is listed in the same L1 memory table (last row, `[value_dim_local]`, `value_dim_local × 2` bytes) and is held in L1 throughout the forward pass. It is not an ephemeral intermediate — it is a persistent weight shard. The peak count therefore omits a live L1 resident.

At Step 2 entry the live L1 tensors are: `x_normed` (just produced by Step 1), `z` (consumed by Step 2), and `w_norm` (persistent throughout). `x` may also still occupy L1 if the runtime has not yet freed it, making the upper bound four tensors. The document's count of three, with the specific enumeration `x`, `z`, and `x_normed`, is wrong in two ways: it omits `w_norm`, and it includes `x` in the peak set when `x` is consumed by Step 1 and is the most likely candidate to be freed before Step 2 entry.

**Correction:** Replace the peak occupancy sentence with:

> Peak simultaneous L1 occupancy is `x_normed`, `z`, and `w_norm` at Step 2 entry (plus `x` if not yet freed by the runtime before Step 2 begins): three to four tensors. All are well within the 1.5 MB L1 per Tensix core.

The table is otherwise correct and does not require changes.

---

## Pass 1 Change Log

Changes applied in response to Pass 1 issues:
1. `causal_conv1d_update_ttnn.md` Section 2a note: removed false "ttnn.slice is a zero-copy view" claim; replaced with accurate statement that ttnn.slice allocates a new buffer (copy, not metadata view); added reference to Section 4 for trace-safety implications
2. `causal_conv1d_update_ttnn.md` Section 4: expanded from state-write-only trace safety to full intermediate buffer analysis; added note clarifying that `shifted` and `conv_state_new` intermediates are runtime allocations that require program-cache address stability or explicit pre-allocation for full trace compatibility
3. `gated_rmsnorm_ttnn.md` Section 3: corrected peak L1 occupancy from "x, z, x_normed" (wrong — includes freed `x`, omits persistent `w_norm`) to "x_normed, z, w_norm (three to four tensors)"

---

## Pass 2

### Verification of Pass 1 fixes

**Fix 1 (causal_conv1d Section 2a note — ttnn.slice allocates new buffer):** APPLIED — current note reads: "`ttnn.slice` allocates a new output buffer (it is a copy, not a zero-copy view); `ttnn.concat` also allocates a combined output buffer. Both produce intermediate tensors that require attention for full Metal Trace compatibility — see Section 4 for the required pre-allocation strategy."

**Fix 2 (causal_conv1d Section 4 — expanded to include intermediates):** APPLIED — key sentence from the expanded section: "`shifted` and `conv_state_new` — the intermediate buffers from `ttnn.slice` and `ttnn.concat` inside the update function. Both allocate new device buffers on each call. For these to be trace-safe, they must also be pre-allocated in `__init__` and the ops must write into those pre-allocated buffers rather than returning freshly allocated tensors."

**Fix 3 (gated_rmsnorm Section 3 — peak L1 corrected to x_normed, z, w_norm):** APPLIED — current peak occupancy sentence reads: "Peak simultaneous occupancy at Step 2 entry (after `ttnn.rms_norm` has produced `x_normed`, before `ttnn.silu` starts): `x_normed`, `z`, and `w_norm` are all live — three tensors; `x` may or may not have been freed by the runtime, giving a three-to-four tensor upper bound."

### Issues found: 1

---

**Issue 1 — `gated_rmsnorm_ttnn.md`, Section 5, incorrect trace-safety reasoning**

**Error:** Section 5 states: "No dynamic memory allocation occurs: `ttnn.rms_norm`, `ttnn.silu`, and `ttnn.mul` each write into pre-allocated output buffers or reuse L1 allocations managed by the TTNN runtime." This reasoning is wrong. `ttnn.rms_norm`, `ttnn.silu`, and `ttnn.mul` do allocate output buffers — `x_normed`, `gate_act`, and `output` are each freshly allocated tensors. The correct model, established in the corrected Section 4 of `causal_conv1d_update_ttnn.md`, is that these allocations are trace-safe not because no dynamic allocation occurs, but because the sizes are fixed and the program cache's buffer-address stability ensures the same addresses are reused on each replay. The claim "no dynamic memory allocation occurs" is factually false and internally inconsistent with the corrected conv1d file in the same chapter, which explicitly acknowledges that intermediate buffers are dynamically allocated and explains the actual mechanism (program-cache address stability) that makes them trace-safe.

**Correction:** Replace the sentence "No dynamic memory allocation occurs: `ttnn.rms_norm`, `ttnn.silu`, and `ttnn.mul` each write into pre-allocated output buffers or reuse L1 allocations managed by the TTNN runtime." with accurate language such as:

> Dynamic memory allocations do occur (`x_normed`, `gate_act`, and `output` are each allocated by their respective ops), but these are trace-safe: the tensor sizes are fixed for a given model configuration, so the program cache reuses the same buffer addresses on each replay. No explicit pre-allocation is required for these intermediates, as long as the program cache is active and sizes do not vary between trace capture and replay.

---

## Pass 2 Change Log

Changes applied in response to Pass 2 issues:
1. `gated_rmsnorm_ttnn.md` Section 5: replaced false "No dynamic memory allocation occurs" claim with accurate program-cache address stability model; `x_normed`, `gate_act`, and `output` are acknowledged as dynamically allocated but trace-safe due to fixed sizes and program cache reuse

---

## Pass 3

### Verification of Pass 2 fix

**Fix (gated_rmsnorm Section 5 — "no dynamic allocation" replaced with program-cache model):** APPLIED — current Section 5 second bullet reads: "`ttnn.rms_norm`, `ttnn.silu`, and `ttnn.mul` each allocate output buffers (`x_normed`, `gate_act`, and `output` respectively), but these allocations are trace-safe: the tensor sizes are fixed for a given model configuration, so the Metal Trace program cache reuses the same buffer addresses on each replay. No explicit pre-allocation is required for these intermediates, as long as the program cache is active and sizes do not vary between trace capture and replay."

### Issues found: 0

---

No issues found. Chapter 3 approved.

---
