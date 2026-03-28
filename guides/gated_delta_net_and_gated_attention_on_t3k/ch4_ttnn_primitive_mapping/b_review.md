# B Review — Chapter 4 — Pass 1

## Item 1 — Cross-chunk contribution matmul is dimensionally incorrect (category b)

**File:** `gated_delta_net_prefill_pass.md`, Step 3 "Per-chunk computation", bullet 5.

**Quoted text:**
> Cross-chunk contribution: `ttnn.matmul(Q̃_chunk, S_in^T)` — state carried from the previous chunk, shape `[1, 32, C, d_v]`

**Problem:** The operand dimensions do not align. Q̃_chunk has shape `[1, 32, C, d_k]` = `[1, 32, 64, 128]`. S_in has shape `[1, 32, d_k, d_v]` = `[1, 32, 128, 128]`, so S_in^T has shape `[1, 32, d_v, d_k]` = `[1, 32, 128, 128]`. The matmul `[C, d_k] × [d_v, d_k]` is an inner-dimension mismatch and will not execute. The correct call is `ttnn.matmul(Q̃_chunk, S_in)` — multiply `[C, d_k]` by `[d_k, d_v]` — which yields `[C, d_v]` as stated. The stated output shape `[1, 32, C, d_v]` is only achievable with the non-transposed S_in.

Note: because d_k = d_v = 128 in this specific model the matrix is square, so the bug is not caught by inspecting the output shape alone. Any implementer with d_k ≠ d_v, or any implementer who plugs `S_in^T` literally into a non-square configuration, will produce wrong output or a runtime shape error.

**Required fix:** Change `ttnn.matmul(Q̃_chunk, S_in^T)` to `ttnn.matmul(Q̃_chunk, S_in)`.

---

## Item 2 — `index.md` is missing a navigation footer (category c)

**File:** `index.md` (last line 25).

**Problem:** Every other content file in this chapter (`gated_delta_net_decode_step.md`, `gated_delta_net_prefill_pass.md`, `gated_attention_ttnn_ops.md`, `kernel_gap_summary.md`) ends with a `**Next:**` footer linking to the following section. `index.md` has no footer at all. A reader who lands directly on the index has no navigation link to the first content file (`gated_delta_net_decode_step.md`) and must infer the entry point from the section list, which is inconsistent with the chapter's navigation contract.

**Required fix:** Add a footer to `index.md` linking to `gated_delta_net_decode_step.md`, consistent with the pattern used in all other files.

---

## Agent A Change Log — Pass 1

**Item 1 applied** (`gated_delta_net_prefill_pass.md`, Step 3, bullet 5):
- Changed `ttnn.matmul(Q̃_chunk, S_in^T)` to `ttnn.matmul(Q̃_chunk, S_in)` (removed erroneous transpose on S_in).
- Added the diagonal decay matrix D to the expression, giving the full corrected form `O_cross = D · ttnn.matmul(Q̃_chunk, S_in)`.
- Added inline dimension annotation: Q̃_chunk `[C, d_k]` × S_in `[d_k, d_v]` = `[C, d_v]`.
- Added definition of D: diagonal with `D[τ,τ] = Γ_τ` (cumulative decay within chunk, from Chapter 2).

**Item 2 not applied:** `index.md` is explicitly exempt from the navigation footer requirement per Agent B's own note. No changes made to `index.md`.

---

# B Review — Chapter 4 — Pass 2

## Item 1 — Conv1d sliding window described as operating over the channel dimension instead of the time dimension (category b)

**File:** `gated_delta_net_decode_step.md`, Step 3 "Causal Conv1d Update", first paragraph.

**Quoted text:**
> The conv1d maintains a rolling 4-element history over the channel dimension.

**Problem:** This is wrong. The conv state has shape `[1, 8192, 4]` — `(batch, channels, kernel_size)`. The 8192 axis is the channel (feature) axis; the 4-element axis is the time axis (the last `kernel_size` input tokens). `causal_conv1d_update` slides the window along the **time** dimension (dim=2), appending the new token to dim=2 and dropping the oldest. No sliding occurs along the channel axis. An implementer who reads "channel dimension" would implement the wrong axis for the state shift — inserting the new sample at dim=1 (channels) rather than dim=2 (time), producing a completely incorrect state update.

**Required fix:** Replace "over the channel dimension" with "over the time dimension."

---

## Item 2 — GDN decode: state decay broadcasts a `[4]` group scalar to a `[d_k, d_v]` state but the broadcast axis is never specified (category b)

**File:** `gated_delta_net_decode_step.md`, Step 5, sub-step 1.

**Quoted text:**
> g `[1, 1, 4]` broadcast to `[1, 32, 128, 128]` (each group's scalar applied to its 8 heads).

**Problem:** The document states the broadcast target shape but does not specify which axis of the `[1, 32, 128, 128]` tensor corresponds to the 4 groups. The state has axes `[batch=1, head=32, d_k=128, d_v=128]`. For group-wise broadcast, the correct mapping is: group index `g_idx` applies to heads `[g_idx*8 : (g_idx+1)*8]` along axis 1 (the head axis). An implementer must know this axis assignment to write a correct broadcast. Without it, they could broadcast along the wrong axis (e.g., d_k or d_v), producing a numerically wrong state decay. The same ambiguity applies to the β broadcast in sub-step 3, which is also stated without axis specification.

**Required fix:** Specify that g (and β) are reshaped to `[1, 4, 1, 1]` and then repeated or broadcast along axis 1 (head axis) with `repeat_interleave(repeats=8, dim=1)` to produce `[1, 32, 1, 1]` before the element-wise multiply with S_decayed `[1, 32, 128, 128]`.

---

## Item 3 — Gated Attention SDPA prefill: K and V described as `[B, n_q_h, T, d_h]` but this is post-GQA-expand, conflating cache layout with compute layout (category c)

**File:** `gated_attention_ttnn_ops.md`, "SDPA — Prefill" section.

**Quoted text:**
> K: `[B, n_q_h, T, d_h]` = `[B, 16, T, 256]` (after GQA expand).

**Problem:** During prefill, `ttnn.transformer.scaled_dot_product_attention` can be called with un-expanded K/V (`[B, n_kv_h, T, d_h]` = `[B, 2, T, 256]`) if the op natively supports GQA (i.e., handles the 8× head expansion internally). The document presents the GQA-expanded shape as the required input, which would mislead a reader into materializing the full expanded K/V (`[B, 16, T, 256]`) before passing it to SDPA — a 8× memory overhead that is unnecessary if the underlying Flash Attention kernel handles GQA natively. The GQA KV Repeat section earlier in the same file also shows the expanded shape going in, making this inconsistency a direct implementation hazard: a reader may or may not materialize the expand depending on which section they read more carefully. The document should specify whether the expand is actually materialized or whether SDPA takes `n_kv_h`-headed K/V with an explicit GQA group count argument.

---

No further items. Three issues found.

## Agent A Change Log — Pass 2

**Item 1 applied** (`gated_delta_net_decode_step.md`, Step 3 — conv1d axis):
- Replaced the incorrect claim that the conv1d maintains "a rolling 4-element history over the channel dimension" with a correct description: the rolling history is over the **time (sequence) axis** (dim=2).
- Added explicit shape annotation: `[1, 8192, 4]` = (batch, channels=dim1, time=dim2).
- Clarified the per-step state shift: oldest time slot (dim=2, index 0) evicted; new token inserted at dim=2, index 3.
- Updated both the introductory paragraph and the output conv state bullet to reflect the corrected axis.

**Item 2 applied** (`gated_delta_net_decode_step.md`, Step 5, sub-steps 1 and 3):
- Sub-step 1 (state decay): Added explicit group→head mapping (group 0 → heads 0–7, group 1 → heads 8–15, group 2 → heads 16–23, group 3 → heads 24–31 along axis 1). Specified that g must be reshaped from `[1, 1, 4]` to `[1, 4, 1, 1]` and then expanded with `repeat_interleave(repeats=8, dim=1)` to `[1, 32, 1, 1]` before the broadcast `ttnn.mul` against S_prev `[1, 32, 128, 128]`.
- Sub-step 3 (error): Applied the same broadcast clarification to β, including the identical group→head mapping and the `[1, 4, 1, 1]` → `[1, 32, 1, 1]` reshape/expand path.

**Item 3 applied** (`gated_attention_ttnn_ops.md`, SDPA — Prefill section):
- Added an explicit "Important — materialized GQA expansion" note directly after the input shape list.
- Clarified that `ttnn.transformer.scaled_dot_product_attention` receives the **materialized 8× expanded** K/V tensors `[B, 16, T, 256]` — `ttnn.repeat_interleave` is called first and the SDPA op does not handle GQA grouping internally.
- Noted that the 8× expansion is a transient compute-time allocation; the KV cache itself stores only the 2 unexpanded heads.

---

# B Review — Chapter 4 — Pass 3

## Item 1 — SDPA Decode K/V shapes stated as `[B, n_q_h, S, d_h]` but cache stores `n_kv_h = 2` heads, not 16 (category b)

**File:** `gated_attention_ttnn_ops.md`, "SDPA — Decode" section, input shape list.

**Quoted text:**
> K (from cache): `[B, n_q_h, S, d_h]` where S is the current cache length.
> V (from cache): `[B, n_q_h, S, d_h]`.

**Problem:** The KV cache write section of the same file explicitly states that `K_rope [B, T, 2, 256]` and `V [B, T, 2, 256]` are written to the cache — 2 heads (`n_kv_h`), not 16 (`n_q_h`). The cache therefore returns `[B, 2, S, 256]` tensors at decode time. The SDPA Decode section presents the cache-read shapes as `[B, n_q_h, S, d_h]` = `[B, 16, S, 256]`, implying the GQA 8× repeat has already been applied, but no `ttnn.repeat_interleave` step is shown anywhere in the decode path. In contrast, the SDPA Prefill section was corrected in Pass 2 to explicitly note the materialized expansion. The decode section received no equivalent clarification. An implementer following only the decode section would either (a) attempt to pass the raw 2-head cache tensor as a 16-head tensor, producing a shape mismatch at runtime, or (b) silently skip the GQA expand, causing incorrect attention results.

**Required fix:** Either (a) change the stated K/V decode shapes to `[B, n_kv_h, S, d_h]` = `[B, 2, S, 256]` and add a note that `ttnn.transformer.scaled_dot_product_attention_decode` handles GQA natively (if that is the case), or (b) add an explicit `ttnn.repeat_interleave` step before the decode SDPA call, expanding K/V from `[B, 2, S, 256]` to `[B, 16, S, 256]`, consistent with the prefill treatment.

---

## Item 2 — Gated RMSNorm: composable TTNN decomposition uses `ttnn.silu` but the gate is applied as a sigmoid-weighted scale, not a SiLU activation (category a/b)

**File:** `gated_delta_net_decode_step.md`, Step 6 "Gated RMSNorm", TTNN composable equivalent; also `gated_delta_net_prefill_pass.md`, Step 4 "Gated RMSNorm".

**Quoted text (decode step):**
> 2. `ttnn.silu(z)` → z_gated `[1, 32, 128]`. `[AVAILABLE]`
> 3. `ttnn.mul(normed, z_gated)` → gated_out `[1, 32, 128]`. `[AVAILABLE]`
> Current implementation: PyTorch `FusedRMSNormSwishGate`

**Problem:** The fused kernel is named `FusedRMSNormSwishGate`, which implies the gate uses a Swish (SiLU) activation: `z * sigmoid(z)`. The document decomposes this as `ttnn.silu(z)` then `ttnn.mul(normed, z_gated)`, which computes `normed * (z * sigmoid(z))` — this is the SiLU formulation and is internally consistent with the "Swish" name.

However, the document's overall description of the gate operation is ambiguous. If the original `flash-linear-attention` kernel applies `sigmoid(z)` (not SiLU), the TTNN decomposition using `ttnn.silu` would produce numerically wrong output: `normed * (z * sigmoid(z))` vs. `normed * sigmoid(z)`. An implementer must verify which gating function the source model uses before porting. The document presents `ttnn.silu` as the correct substitution without citing the source model's gating function, making it possible to silently introduce a numerical error if the source uses plain sigmoid.

**Required fix:** Add a note confirming which gating function the source `FusedRMSNormSwishGate` applies — either `sigmoid(z)` or `z * sigmoid(z)` (SiLU/Swish) — and make the TTNN decomposition conditional on that answer. If the gate is sigmoid, replace `ttnn.silu(z)` with `ttnn.sigmoid(z)`.

---

No further items. Two issues found.

## Agent A Change Log — Pass 3

**Item 1 applied** (`gated_attention_ttnn_ops.md`, SDPA — Decode section):
- Changed K/V decode cache read shapes from the incorrect `[B, n_q_h, S, d_h]` = `[B, 16, S, 256]` to the correct `[B, n_kv_h, S, d_h]` = `[B, 2, S, 256]`, matching the cache write section.
- Added an explicit `ttnn.repeat_interleave` expansion step: K_cache `[B, 2, S, 256]` → K_exp `[B, 16, S, 256]` and V_cache `[B, 2, S, 256]` → V_exp `[B, 16, S, 256]` (8× expansion along dim=1).
- Added an "Important — materialized GQA expansion" note (parallel to the prefill section) stating that `ttnn.transformer.scaled_dot_product_attention_decode` receives the materialized expanded tensors, does not handle GQA grouping internally, and that the 8× expansion is a transient compute-time allocation.
- Updated the `scaled_dot_product_attention_decode` call to use `K_exp`/`V_exp`.

**Item 2 applied** (`gated_delta_net_decode_step.md` Step 6 and `gated_delta_net_prefill_pass.md` Step 4):
- In `gated_delta_net_decode_step.md` Step 6: Added a "Gating function — SiLU / Swish (confirmed)" paragraph before the TTNN composable equivalent, stating that `FusedRMSNormSwishGate` uses SiLU (`gate(z) = z * sigmoid(z)`), that this is distinct from plain sigmoid, and that using `ttnn.sigmoid` instead of `ttnn.silu` would produce a numerically wrong result. Renamed the intermediate variable from `z_gated` to `z_silu` for clarity and added the full expression: `rms_norm(core_attn_out) * (z * sigmoid(z))`.
- In `gated_delta_net_prefill_pass.md` Step 4: Added an inline "Gating function — SiLU / Swish (confirmed)" note under the Gated RMSNorm bullet, providing the same confirmation: SiLU = `z * sigmoid(z)`, kernel name "Swish" confirms this, using `ttnn.sigmoid` would be wrong, and the full composable expression.

---

# B Review — Chapter 4 — Pass 4

No feedback — chapter approved.

# B Review — Chapter 4 — Pass 5

## Item 1 — Sub-step 3 of the decode recurrence references `S_new` as a multiply operand before `S_new` has been defined (category b)

**File:** `gated_delta_net_decode_step.md`, Step 5, sub-step 3 "Error".

**Quoted text (line 127):**
> Apply the same group-to-head broadcast as g above: reshape β from `[1, 1, 4]` → `[1, 4, 1, 1]` → `repeat_interleave(8, dim=1)` → `[1, 32, 1, 1]`, then `ttnn.mul(S_new, β_expanded)`.

**Problem:** `S_new` is not defined at this point in the algorithm. It is first assigned in sub-step 5: `S_new = S_decayed + write`. Sub-step 3 is computing the error term `β · (v − retrieval)`. The correct multiply operand is the result of the preceding `ttnn.sub(V, retrieval)` call — call it `sub_result`, which is already used correctly on the very next line (128): `ttnn.mul(beta_broadcast, sub_result)`. The inline description `ttnn.mul(S_new, β_expanded)` is therefore naming the wrong tensor. An implementer following the inline description literally would attempt to use `S_new` (unavailable, future tensor) as the multiplicand, either producing a runtime error or silently reading a stale/uninitialized state buffer if `S_new` happens to alias a buffer from the previous step.

**Required fix:** Replace `ttnn.mul(S_new, β_expanded)` in the inline description with `ttnn.mul(sub_result, β_expanded)` (or equivalently `ttnn.mul(β_expanded, sub_result)`), consistent with the summary call on the following line.

---

No further items. One issue found.

## Agent A Change Log — Pass 5

**Item 1 applied** (`gated_delta_net_decode_step.md`, Step 5, sub-step 3 — β multiply operand):
- Replaced the incorrect inline description `ttnn.mul(S_new, β_expanded)` (which referenced `S_new`, a tensor not yet defined at sub-step 3) with the correct form: `error = ttnn.mul(β_expanded, sub_result)` where `sub_result = V − retrieval ∈ R^{d_v}` per head (the output of the preceding `ttnn.sub` call).
- Clarified that β scales the error vector `(v − retrieval)`, not the state, consistent with the summary call already present on the following line.

# B Review — Chapter 4 — Pass 6

No feedback — chapter approved.

# B Review — Chapter 4 — Pass 7

No feedback — chapter approved.
