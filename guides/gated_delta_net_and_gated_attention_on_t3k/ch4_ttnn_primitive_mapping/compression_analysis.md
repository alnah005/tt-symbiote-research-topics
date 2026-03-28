# Compression Analysis: Chapter 4 — TTNN Primitive Mapping: Decode and Prefill Forward Passes — Pass 1

## Summary
- Total files analyzed: 5
- Estimated current line count: ~545 lines
- Estimated post-compression line count: ~455 lines
- Estimated reduction: ~17%

---

## CRUCIAL Suggestions

### [gated_delta_net_decode_step.md] ~lines 116–128 (Step 5, sub-step 1 and sub-step 3)
**Issue:** The group-to-head mapping explanation (`group 0 → heads 0–7, group 1 → heads 8–15 …`) is stated in full twice — once inside the g broadcast description (lines 117–118) and again verbatim inside the β broadcast description (lines 127–128). Both paragraphs also re-explain the `repeat_interleave(repeats=8, dim=1)` reshape procedure identically.
**Suggestion:** In sub-step 3 (β error), replace the full re-explanation with a back-reference: "β must be reshaped and expanded identically to g (see sub-step 1 above)." Remove the duplicated sentence about groups mapping to heads along axis 1.

### [gated_attention_ttnn_ops.md] ~lines 134 and 153–158 (SDPA Prefill and SDPA Decode)
**Issue:** The "materialized GQA expansion" caveat — that `ttnn.repeat_interleave` must be called first, that the SDPA op does not accept a GQA group count argument, and that the 8× expansion is a transient compute-time allocation while the KV cache stores only 2 unexpanded heads — is copied almost word-for-word in both the prefill section (lines 134–135) and the decode section (lines 153–158). The decode block even ends with an extra sentence re-stating the 8× cost already stated in the same paragraph.
**Suggestion:** State the caveat once in full in the prefill section. In the decode section, open with: "The same materialized GQA expansion applies here (see SDPA — Prefill above)." Then give only the decode-specific details (the `dim=1` vs `dim=2` repeat, the `cur_pos` argument, the decode-optimized kernel name).

### [gated_delta_net_prefill_pass.md] ~lines 107–112 (Step 4, Gated RMSNorm)
**Issue:** The SiLU/Swish clarification block — "The kernel name `FusedRMSNormSwishGate` uses SiLU (also called Swish): `gate(z) = z * sigmoid(z)`. This is distinct from plain sigmoid — using `ttnn.sigmoid` instead of `ttnn.silu` would produce a numerically wrong result. The full composable expression is: `gated_out = rms_norm(core_attn_out) * silu(z)` = `rms_norm(core_attn_out) * (z * sigmoid(z))`. The 'Swish' in the kernel name confirms this is the correct activation." — is an almost verbatim repeat of the same explanation in `gated_delta_net_decode_step.md` lines 162–169.
**Suggestion:** In the prefill file, condense to a single cross-reference sentence: "The gating function is SiLU/Swish (not plain sigmoid) — see the decode step file for the full derivation. Composable TTNN: `ttnn.rms_norm` + `ttnn.silu` + `ttnn.mul`." The full reasoning lives once in the decode file.

---

## MINOR Suggestions

### [gated_delta_net_decode_step.md] ~lines 3–5 (opening paragraph)
**Issue:** "Decode is the latency-critical path: the sequence length is 1, the recurrent state `S` carries all context, and there is no parallelism over the time dimension." The phrase "the sequence length is 1" is immediately redundant given the section title already reads "B=1, T=1" and the symbols table on line 5 defines T=1 explicitly.
**Suggestion:** Drop "the sequence length is 1" from the sentence; replace with "T=1" inline reference only if needed, or rely on the title.

### [gated_delta_net_decode_step.md] ~lines 60–63 (Step 3, last paragraph)
**Issue:** "Porting requires either a custom Metalium kernel or expressing the update as a batched dot product … — the latter is functionally correct but incurs per-step launch overhead for a very small compute tile." This is a lighter restatement of the same porting recommendation already given with more detail in `kernel_gap_summary.md`.
**Suggestion:** Trim to one sentence: "Porting options are described in `kernel_gap_summary.md`." Alternatively, keep only the dot-product option sentence and remove the repetition.

### [kernel_gap_summary.md] ~lines 35–44 (Key Finding 1, bullet list)
**Issue:** The 6-op decomposition list (ttnn.mul → state decay, ttnn.matmul → retrieval, etc.) fully restates the numbered list in `gated_delta_net_decode_step.md` lines 116–141. The `kernel_gap_summary.md` audit table on line 18 also already labels these 6 ops. The list appears three times across the chapter in slightly different formats.
**Suggestion:** In Key Finding 1, replace the 6-item bullet list with: "All 6 sub-operations (state decay, retrieval, error, write, state update, output — detailed in `gated_delta_net_decode_step.md` Step 5) are individually available in TTNN." Keep only the dispatch-overhead argument that follows.

### [gated_attention_ttnn_ops.md] ~lines 53 and 117 (shape-check sentences)
**Issue:** Inline shape checks of the form "n_q_h × d_h = 16 × 256 = 4096 — consistent with the projection output" appear at the end of the Q Gating section (line 53) and the GQA KV Repeat section (line 117). These arithmetic checks repeat numbers already established in the Symbols line and in the projection sections immediately above.
**Suggestion:** Remove the parenthetical shape-check sentences; the arithmetic is recoverable from the symbol definitions without inline commentary.

### [gated_delta_net_prefill_pass.md] ~lines 13–35 (Step 1 projections)
**Issue:** The projection entries (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) each restate weight shapes and operation strings that are byte-for-byte identical to the decode step file, with only the tensor's T dimension changing from 1 to T. The note "All four projections are identical to the decode case with T>1 substituted for T=1" (line 13) already signals this — but then the full details are still written out.
**Suggestion:** Replace the full per-projection blocks with a compact table (weight, output shape, status) and a single note: "Operations are `ttnn.linear` with col-sharding for `in_proj_qkv`/`in_proj_z`; PyTorch `nn.Linear` (requires wiring) for `in_proj_a`/`in_proj_b` — identical to decode, T=1 → T." This removes ~15 lines of restatement.

---

## Load-Bearing Evidence

- `gated_delta_net_decode_step.md` line ~54: "The conv state shape is `[1, 8192, 4]` where 8192 is the channel dimension (dim=1) and 4 is the time/sequence axis (dim=2, equal to `kernel_size`)" — load-bearing because it precisely defines which axis is the time axis vs. the channel axis, correcting a common layout confusion that affects the porting approach.
- `gated_delta_net_decode_step.md` line ~162: "This is distinct from plain sigmoid — `sigmoid(z)` returns a value in (0, 1), while `silu(z) = z * sigmoid(z)` is unbounded above 0 for positive z." — load-bearing because the numerical distinction justifies the TTNN primitive choice (`ttnn.silu` not `ttnn.sigmoid`) for the gating path.
- `gated_attention_ttnn_ops.md` line ~134: "The `ttnn.repeat_interleave` call from the GQA KV Repeat section above is performed first; the SDPA op itself does not handle the GQA grouping internally and does not accept a GQA group count argument." — load-bearing because it documents a non-obvious API constraint that would otherwise cause a silent shape error.
- `gated_delta_net_prefill_pass.md` line ~81: "Here Q̃_chunk is `[C, d_k]` and S_in is `[d_k, d_v]`, so the product `Q̃_chunk @ S_in` yields `[C, d_v]` correctly (no transpose). D ∈ R^{C×C} is the diagonal cumulative-decay matrix with `D[τ,τ] = Γ_τ`" — load-bearing because it clarifies the transpose orientation that must match the WY decomposition in Chapter 2.
- `kernel_gap_summary.md` line ~46: "The state tensor `[1, 32, 128, 128]` = 32 × 128 × 128 × 2 bytes (bfloat16) = 1 MB fits comfortably in L1 SRAM on Wormhole, making a register-resident fused kernel the natural design." — load-bearing because it provides the hardware sizing argument that motivates the fused-kernel recommendation over the composable alternative.

---

## VERDICT
- Crucial updates: yes

---

## Agent A Change Log — Compression Pass 1

**Date:** 2026-03-27

**Items applied:** 3 CRUCIAL suggestions (no MINOR suggestions applied).

### Item 1 — `gated_delta_net_decode_step.md`, Step 5, sub-step 3 (β broadcast)
- Removed the verbatim re-explanation of the group-to-head mapping (`group 0 → heads 0–7 …`) and the `repeat_interleave` procedure that duplicated sub-step 1.
- Replaced with: "Apply the same group-to-head broadcast as g above: reshape β from `[1, 1, 4]` → `[1, 4, 1, 1]` → `repeat_interleave(8, dim=1)` → `[1, 32, 1, 1]`, then `ttnn.mul(S_new, β_expanded)`."
- Full explanation retained in sub-step 1 (g broadcast).

### Item 2 — `gated_attention_ttnn_ops.md`, SDPA Decode section
- Removed the near-verbatim restatement of the materialized GQA expansion caveat (including the duplicate "does not accept a GQA group count argument" sentence and the closing sentence restating the 8× cost).
- Replaced with a cross-reference: "The same GQA expansion applies: read K/V from cache at `[B, 2, S, 256]`, expand to `[B, 16, S, 256]` via `ttnn.repeat_interleave` (see Prefill section above for details)."
- Full caveat retained in SDPA Prefill section.

### Item 3 — `gated_delta_net_prefill_pass.md`, Step 4 (Gated RMSNorm)
- Removed the duplicated SiLU/sigmoid numerical distinction block ("The kernel name `FusedRMSNormSwishGate` uses SiLU … The 'Swish' in the kernel name confirms this is the correct activation.").
- Replaced with: "The gating function is SiLU (`z * sigmoid(z)`), not plain sigmoid — see decode Step 6 for the numerical justification."
- Full derivation and justification retained in `gated_delta_net_decode_step.md` Step 6.

---

# Compression Analysis: Chapter 4 — Pass 2

## Summary
- Total files analyzed: 5
- Estimated current line count: ~655 lines
- Estimated post-compression line count: ~640 lines
- Estimated reduction: ~2%

## CRUCIAL Suggestions

### [gated_delta_net_decode_step.md] line 127 — β reshape chain still duplicated (Pass 1 Item 1 partially unresolved)

Pass 1 intended to replace the β broadcast explanation with a pure back-reference to sub-step 1. The change log confirms this intent. However the live file at line 127 still spells out the full reshape chain: "reshape β from `[1, 1, 4]` → `[1, 4, 1, 1]` → `repeat_interleave(8, dim=1)` → `[1, 32, 1, 1]`" — identical to sub-step 1, line 117. Only the group-to-head mapping sentence (`group 0 → heads 0–7 …`) was removed; the reshape steps themselves were re-introduced verbatim in the replacement text.

**Suggestion:** Reduce line 127 to: "Apply the same group-to-head broadcast as g above (see sub-step 1), then `error = ttnn.mul(β_expanded, sub_result)` where `sub_result` is the output of the preceding `ttnn.sub` call." The reshape chain `[1, 1, 4]` → `[1, 4, 1, 1]` → `[1, 32, 1, 1]` is already fully described 10 lines earlier; repeating it adds no information.

## MINOR Suggestions

### [gated_delta_net_decode_step.md] line 142 — fused-kernel recommendation restated from kernel_gap_summary.md

The closing paragraph of Step 5 ("While each of the 6 sub-operations … A fused Metalium kernel … is the recommended path for decode latency (see `kernel_gap_summary.md`)") restates the recommendation already made at length in `kernel_gap_summary.md` Key Finding 1 (lines 33–46). The cross-reference at the end of the sentence acknowledges this but the prose before it still duplicates the core argument.

**Suggestion:** Trim to: "Dispatching these 6 ops separately incurs kernel-launch overhead on small tensors; a fused Metalium kernel is recommended (see `kernel_gap_summary.md` Key Finding 1)." — one sentence, cross-reference only.

### [gated_delta_net_prefill_pass.md] lines 3–5 (opening paragraph) — decode differences restated redundantly

The opening paragraph lists three differences from decode: "(1) all projections and conv1d operate over the full time dimension; (2) the chunkwise delta rule replaces the single-step recurrence; (3) the pass saves a `final_state` and `conv_state`." The "Key Difference: Prefill vs. Decode" table at lines 122–129 already captures every one of these distinctions in structured form. The opening prose restates them in a less scannable format before the reader has seen the table.

**Suggestion:** Shorten the opening paragraph to: "Prefill processes all T tokens of a prompt in a single forward call. Key differences from decode are summarized in the table at the end of this section." Let the table carry the comparison.

### [gated_delta_net_prefill_pass.md] lines 135–145 (TTNN Gaps table) — duplicates the audit table in kernel_gap_summary.md

The "TTNN Gaps for Prefill" table lists 5 rows that are a direct subset of the 19-row audit table in `kernel_gap_summary.md`. Three of the five rows (`causal_conv1d_fn`, `chunk_gated_delta_rule`, `FusedRMSNormSwishGate`) add no information beyond what is in the audit table, and the closing sentence ("Porting the three custom kernel gaps is the primary kernel development needed…") restates the conclusion of `kernel_gap_summary.md` Key Finding 3.

**Suggestion:** Replace the table with a single sentence: "The three custom-kernel gaps (`causal_conv1d_fn`, `chunk_gated_delta_rule`, `FusedRMSNormSwishGate`) and two wiring gaps (`in_proj_a`, `in_proj_b`) are consolidated in `kernel_gap_summary.md`." This removes ~12 lines.

### [kernel_gap_summary.md] lines 65–80 (Key Finding 3, priority table) — partial restatement of the audit table above

The priority table in Key Finding 3 (lines 72–76) lists four kernel gaps with priority rankings. All four appear in the main audit table on lines 9–29 with the same "Recommended Path" column. The only net-new information in Key Finding 3's table is the explicit priority numbers; the kernel names, affected phases, and notes are already present above.

**Suggestion:** Keep the priority numbers as a short inline list ("Priority 1: fused recurrent decode step; Priority 2: `causal_conv1d_update`; Priority 3: `chunk_gated_delta_rule`; Priority 4: `causal_conv1d_fn`") and drop the four-column table. Saves ~8 lines.

## Load-Bearing Evidence

- `gated_delta_net_decode_step.md` line 117: "The 4 groups map to consecutive blocks of 8 heads along the head axis (axis 1): group 0 → heads 0–7, group 1 → heads 8–15, group 2 → heads 16–23, group 3 → heads 24–31." — load-bearing; the only place the group-to-head mapping is fully defined after Pass 1.
- `gated_delta_net_decode_step.md` line 162: "This is distinct from plain sigmoid — `sigmoid(z)` returns a value in (0, 1), while `silu(z) = z * sigmoid(z)` is unbounded above 0 for positive z." — load-bearing; numerical justification for choosing `ttnn.silu` over `ttnn.sigmoid`.
- `gated_delta_net_prefill_pass.md` line 81: "Here Q̃_chunk is `[C, d_k]` and S_in is `[d_k, d_v]`, so the product `Q̃_chunk @ S_in` yields `[C, d_v]` correctly (no transpose). D ∈ R^{C×C} is the diagonal cumulative-decay matrix with `D[τ,τ] = Γ_τ`" — load-bearing; transpose orientation is non-obvious and directly affects correctness of the port.
- `gated_attention_ttnn_ops.md` line 134: "the SDPA op itself does not handle the GQA grouping internally and does not accept a GQA group count argument." — load-bearing; API constraint that prevents a silent shape error.
- `kernel_gap_summary.md` line 46: "The state tensor `[1, 32, 128, 128]` = 32 × 128 × 128 × 2 bytes (bfloat16) = 1 MB fits comfortably in L1 SRAM on Wormhole, making a register-resident fused kernel the natural design." — load-bearing; hardware sizing argument that motivates the fused kernel recommendation.

## VERDICT
- Crucial updates: yes

---

## Agent A Change Log — Compression Pass 2

**Date:** 2026-03-27

**Items applied:** 1 CRUCIAL suggestion (no MINOR suggestions applied).

### Item 1 — `gated_delta_net_decode_step.md`, Step 5, sub-step 3 (β broadcast) — reshape chain duplication

- Removed the verbatim reshape chain `[1, 1, 4]` → `[1, 4, 1, 1]` → `repeat_interleave(8, dim=1)` → `[1, 32, 1, 1]` from sub-step 3, which was identical to sub-step 1.
- Replaced with a cross-reference: "Reshape and expand β identically to g (see sub-step 1): `[1, 1, 4]` → `[1, 32, 1, 1]`."
- The `ttnn.mul(β_expanded, sub_result)` call and the `sub_result` definition are retained in sub-step 3 as they are specific to the β error computation.
- Full reshape procedure (including the intermediate `[1, 4, 1, 1]` step and `repeat_interleave`) remains only in sub-step 1 (g broadcast).

---

# Compression Analysis: Chapter 4 — Pass 3

## Summary
- Total files analyzed: 5
- Estimated current line count: ~650 lines
- Estimated post-compression line count: ~627 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions

None — all prior crucials resolved.

**Pass 2 CRUCIAL re-check (β reshape chain duplication):** Confirmed resolved. Live file at `gated_delta_net_decode_step.md` line 127 now reads: "Reshape and expand β identically to g (see sub-step 1): `[1, 1, 4]` → `[1, 32, 1, 1]`." The intermediate `[1, 4, 1, 1]` step and the `repeat_interleave(8, dim=1)` prose are absent from sub-step 3. Only the start/end shapes are shown; the full procedure lives exclusively in sub-step 1. No duplication remains.

## MINOR Suggestions

### [gated_delta_net_decode_step.md] line 142 — fused-kernel closing paragraph restates kernel_gap_summary.md Key Finding 1

The final paragraph of Step 5 ("While each of the 6 sub-operations is individually expressible in TTNN, dispatching them separately incurs kernel-launch overhead for very small tensors … A fused Metalium kernel that executes the full recurrence in a single dispatch is the recommended path for decode latency (see `kernel_gap_summary.md`).") restates an argument already made in full in `kernel_gap_summary.md` Key Finding 1 (lines 33–46), which explains the same launch-overhead problem and the same 1 MB L1-fit rationale. The cross-reference at the end of the sentence signals the duplication without eliminating it.

**Suggestion:** Trim to one sentence: "Dispatching these 6 ops separately incurs kernel-launch overhead on small tensors; a fused Metalium kernel is recommended — see `kernel_gap_summary.md` Key Finding 1." (~3 lines saved.)

### [gated_delta_net_prefill_pass.md] lines 3–5 — opening paragraph pre-states the prefill/decode comparison table

The opening paragraph lists three enumerated differences from decode: "(1) all projections and conv1d operate over the full time dimension; (2) the chunkwise delta rule replaces the single-step recurrence; (3) the pass saves a `final_state` and `conv_state`." The "Key Difference: Prefill vs. Decode" table at lines 122–129 covers every one of these items in structured, scannable form. The prose introduction is not wrong, but it duplicates the table before the reader reaches it.

**Suggestion:** Shorten the opening paragraph to: "Prefill processes all T tokens of a prompt in a single forward call. Key differences from decode are summarized in the comparison table at the end of this section." (~2 lines saved.)

### [gated_delta_net_prefill_pass.md] lines 135–145 — TTNN Gaps table duplicates the kernel_gap_summary.md audit table

The "TTNN Gaps for Prefill" table lists 5 rows (`causal_conv1d_fn`, `chunk_gated_delta_rule`, `FusedRMSNormSwishGate`, `in_proj_a`/`in_proj_b`, and "All other ops") that are a direct subset of the 19-row audit table in `kernel_gap_summary.md`. The closing sentence ("Porting the three custom kernel gaps is the primary kernel development needed to achieve full TTNN acceleration of Gated Delta Net prefill on T3K.") restates the conclusion of Key Finding 3. No row or sentence in this table adds information not already in `kernel_gap_summary.md`.

**Suggestion:** Replace the entire table and closing sentence with: "The three custom-kernel gaps (`causal_conv1d_fn`, `chunk_gated_delta_rule`, `FusedRMSNormSwishGate`) and two wiring gaps (`in_proj_a`, `in_proj_b`) are consolidated in `kernel_gap_summary.md`." (~12 lines saved.)

### [kernel_gap_summary.md] lines 71–76 — priority table in Key Finding 3 repeats columns already present in the audit table

The four-row priority table in Key Finding 3 (kernel name, affects, notes) reproduces information from the main audit table (lines 9–29) which already lists these four kernels with "Recommended Path" and "Gap" columns. The only net-new content in Key Finding 3's table is the explicit priority numbering (1–4). The kernel names, affected phases, and recommended paths are redundant.

**Suggestion:** Replace the table with an inline priority list: "Priority 1: fused recurrent decode step; Priority 2: `causal_conv1d_update`; Priority 3: `chunk_gated_delta_rule`; Priority 4: `causal_conv1d_fn`." (~6 lines saved.)

## Load-Bearing Evidence

- `gated_delta_net_decode_step.md` line 117: "The 4 groups map to consecutive blocks of 8 heads along the head axis (axis 1): group 0 → heads 0–7, group 1 → heads 8–15, group 2 → heads 16–23, group 3 → heads 24–31." — load-bearing; the only place the full group-to-head mapping is defined; removing it would make line 127's cross-reference ("identically to g") unresolvable.
- `gated_delta_net_decode_step.md` line 162: "This is distinct from plain sigmoid — `sigmoid(z)` returns a value in (0, 1), while `silu(z) = z * sigmoid(z)` is unbounded above 0 for positive z. Using `ttnn.sigmoid` instead of `ttnn.silu` here would produce a numerically wrong result." — load-bearing; numerical justification for the `ttnn.silu` primitive choice; referenced by cross-reference in prefill Step 4.
- `gated_delta_net_prefill_pass.md` line 81: "Here Q̃_chunk is `[C, d_k]` and S_in is `[d_k, d_v]`, so the product `Q̃_chunk @ S_in` yields `[C, d_v]` correctly (no transpose). D ∈ R^{C×C} is the diagonal cumulative-decay matrix with `D[τ,τ] = Γ_τ`" — load-bearing; transpose orientation and D definition are non-obvious and directly affect correctness of the port against Chapter 2's WY formulation.
- `gated_attention_ttnn_ops.md` line 134: "the SDPA op itself does not handle the GQA grouping internally and does not accept a GQA group count argument." — load-bearing; API constraint that would cause a silent shape error if missed; not derivable from the TTNN public docs without experimentation.
- `kernel_gap_summary.md` line 46: "The state tensor `[1, 32, 128, 128]` = 32 × 128 × 128 × 2 bytes (bfloat16) = 1 MB fits comfortably in L1 SRAM on Wormhole, making a register-resident fused kernel the natural design." — load-bearing; hardware sizing argument that drives the fused kernel recommendation over the composable alternative.

## VERDICT
- Crucial updates: no
