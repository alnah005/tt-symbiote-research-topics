# Compression Analysis: GatedDeltaNet — Pass 1

## Summary
- Total files analyzed: 5
- Estimated current line count: ~987 lines
- Estimated post-compression line count: ~830 lines
- Estimated reduction: ~16%

---

## CRUCIAL Suggestions

### [host_recurrence.md] ~lines 103–110
**Issue:** The per-token timing breakdown (54 ms, 18 ms, 14 ms, 86 ms total) is stated once in
prose at lines 94–110, then repeated verbatim as a near-identical table in `fused_kernel.md`
lines 210–219. The two tables are essentially identical (same four rows, same numbers, same
notes). One must be authoritative; the other is pure duplication.

**Suggestion:** Keep the full profiling table only in `host_recurrence.md` (where the host
path is the subject). In `fused_kernel.md` replace the duplicate table with a single line:
"See the profiling table in `host_recurrence.md` for the per-component breakdown." Save
~10 lines in `fused_kernel.md`.

---

### [host_recurrence.md] ~lines 26–31 AND [fused_kernel.md] ~lines 104–119
**Issue:** The Blackhole SrcB fp32 hang constraint is explained in full in `host_recurrence.md`
(the file whose entire purpose is that constraint). `fused_kernel.md` then re-explains the
same constraint in the "What the Kernel Computes" section (lines 105–119): it names the same
three workarounds (`init_sfpu`+`copy_tile`, SFPU binary path, `binary_dest_reuse_tiles`) and
restates that `matmul_tiles` has a separate unpack path. This is a duplicate explanation of
technical hardware detail across two files.

**Suggestion:** In `fused_kernel.md`, the parenthetical "(via SFPU binary path)" and the
final paragraph about `matmul_tiles` (lines 117–119) should be collapsed to a single
cross-reference sentence: "The fp32 workarounds are documented in `host_recurrence.md`."
Save ~5 lines.

---

### [projections_and_conv.md] ~lines 219–254 AND [host_recurrence.md] ~lines 153–161
**Issue:** The `_dev_state` initialization code block (allocating the float32 recurrent state)
appears twice, nearly word-for-word. `projections_and_conv.md` lines 233–240 show the inner
body of `initialize_states`; `host_recurrence.md` lines 154–161 show the identical
`ttnn.from_torch` call for `_dev_state` with an identical comment "# State initialized as
float32 on device DRAM." Both are inside sections explaining the same design decision (float32
precision for the state tensor).

**Suggestion:** Remove the duplicated code block from `host_recurrence.md` lines 153–161.
The `host_recurrence.md` section "Current Design: Float32 State on Device" can reference
`projections_and_conv.md` for the initialization code and focus only on the fused-kernel
call pattern (which is the new content in that section). Save ~10 lines.

---

### [index.md] ~lines 28–34
**Issue:** The "Reading Order" table (lines 21–26) already lists each file with a one-line
description. The prose paragraph immediately below it (lines 28–34) restates the same four
files in the same order with the same descriptions, adding no new information.

**Suggestion:** Delete the prose paragraph (lines 28–34) entirely. The table is sufficient.
Save ~7 lines.

---

## MINOR Suggestions

### [recurrence_math.md] ~lines 64–68
**Issue:** The code block preamble ("The following is a simplified excerpt of the Python
reference used in the test suite…") contains a parenthetical sentence explaining that
`if not output_final_state: last_recurrent_state = None` was omitted for clarity. The code
block already includes those lines (lines 107–108), making the parenthetical in the preamble
false and confusing.

**Suggestion:** Remove the parenthetical sentence starting "The actual source also contains…"
(lines 66–67). Save ~2 lines, and remove the false statement.

---

### [recurrence_math.md] ~lines 167–170
**Issue:** The explanation of L2 normalization's purpose (lines 167–170) restates what the
formula already shows. "Without it, the outer product $\mathbf{k}_t \otimes \boldsymbol{\delta}_t$
can have unbounded magnitude, causing the state matrix $S$ to grow without bound over many
token steps" is correct but could be compressed to one sentence.

**Suggestion:** Merge into: "L2 normalization is critical for numerical stability: without
it, $\mathbf{k}_t \otimes \boldsymbol{\delta}_t$ can grow without bound and corrupt $S$."
The sentence about `use_qk_l2norm_in_kernel=True` on line 170 is load-bearing; keep it.
Save ~2 lines.

---

### [projections_and_conv.md] ~lines 111–112
**Issue:** The explanation "Each slot has shape [1, 1, B_pad, conv_dim] in bfloat16.
`B_pad` is `tile_padded_batch_rows` (= 32, the minimum tile height), which pads a single-sample
batch to the tile boundary." is repeated or implied again in the Conv Weight Layout section
(lines 161–165 and 174–176). The `B_pad = 32` value and its reason are stated in three places
in the same file.

**Suggestion:** State `B_pad = tile_padded_batch_rows = 32` once (at its first use in the
Ring Buffer Layout section), and replace the two later restatements with just `B_pad`.
Save ~3 lines.

---

### [fused_kernel.md] ~lines 81–87
**Issue:** The comment block explaining the conditional permute (lines 81–87) is verbose.
The phrase "Note that `shape[2]` is not the outer batch dimension (`shape[0]`); it is the
tile-padded batch rows…" reads as a defensive clarification that over-explains what the
variable name `B_pad` and the earlier sections already make clear. The actual reason for
the permute (ttnn.reshape cannot flatten non-adjacent dims in tile layout) is the useful
content; the preceding two sentences are hedging.

**Suggestion:** Trim to: "The permute fires whenever `B_pad > 1` (always in production).
`ttnn.reshape` cannot flatten dim 1 into dim 3 in tile layout when the batch row dimension
intervenes, so `permute([0, 2, 1, 3])` moves heads adjacent to features first."
Save ~3 lines.

---

### [host_recurrence.md] ~lines 33–37
**Issue:** The sentence "This was confirmed empirically: running the recurrence in bfloat16
on device produces PCC values near 0 against the float32 reference after even a few tokens"
is a verbal summary of the quantitative test result already presented in full (with code) in
lines 128–143 of the same file under "Why bf16 Recurrence Is Incorrect." The upfront
assertion at lines 33–37 slightly duplicates the later, more rigorous evidence.

**Suggestion:** Remove the "This was confirmed empirically…" sentence from lines 35–37 and
let the full test evidence in the later section stand alone. Save ~2 lines.

---

### [fused_kernel.md] ~lines 185–195
**Issue:** The paragraph explaining why the existing host-recurrence path avoids certain
`from_torch` costs (lines 190–195) is speculative background comparison prose. It hedges
("comparable") and introduces the host path as a frame of reference in a section that is
supposed to be about the fused kernel's limitation. The comparison does not add precision;
it muddies the main point.

**Suggestion:** Delete lines 190–195 ("The existing host-recurrence path incurs…"). The
preceding paragraph (lines 185–189) already makes the core point cleanly. Save ~6 lines.

---

## Load-Bearing Evidence

- `recurrence_math.md` line ~31: `"> **Note:** The update cannot be written as the single compressed form…"` — load-bearing because it pre-empts a common misreading of the three-line recurrence; removing it would leave the compressed form $S \leftarrow S \cdot \exp(g) + \mathbf{k} \otimes \boldsymbol{\delta}$ appearing to be equivalent when it is not.

- `host_recurrence.md` line ~31: `"the three-line sequential form above is the correct reading order"` — load-bearing cross-reference that ties the abbreviated equations used in `host_recurrence.md` back to the canonical five-step form in `recurrence_math.md`; without it the abbreviated form would appear authoritative.

- `projections_and_conv.md` lines ~244–258: Key design decision list items 1–4 (float32 state, bfloat16 conv rows, `_oldest` init, fixed addresses) — load-bearing because these four design rationales are the architectural record for `initialize_states`; they do not appear together elsewhere and motivate constraints in both `host_recurrence.md` and `fused_kernel.md`.

- `fused_kernel.md` lines ~229–248: "Path to Production" steps 1–4 — load-bearing because these are the only place in the chapter where the future work sequence (Trace integration → kernel switch → multi-step verification → optional multi-CQ overlap) is enumerated concretely; removing any step would lose scope definition.

- `recurrence_math.md` lines ~115–138: Gate computation code block and the `_neg_A_exp_dev` constructor snippet — load-bearing because the precomputation of `-exp(A_log)` as a device constant is a non-obvious optimization that is referenced by name in both `projections_and_conv.md` and the kernel input table; the explanation cannot be shortened without losing why the device tensor exists.

---

## VERDICT
- Crucial updates: yes

---
## Change Log (Agent A — Pass 1 CRUCIAL fixes)
- fused_kernel.md: Removed duplicate profiling table; replaced with cross-reference to host_recurrence.md
- fused_kernel.md: Removed re-explanation of Blackhole fp32 workarounds; replaced with cross-reference to host_recurrence.md
- host_recurrence.md: Removed duplicate _dev_state initialization code block; replaced with cross-reference to projections_and_conv.md
- index.md: Removed redundant prose paragraph that restated the Reading Order table

---

# Compression Analysis: GatedDeltaNet — Pass 2

## Summary
- Total files analyzed: 5
- Estimated current line count: ~960 lines
- Estimated post-compression line count: ~940 lines
- Estimated reduction: ~2%

## CRUCIAL Suggestions

### [fused_kernel.md] lines 189–193 — Pass 1 MINOR not applied; now escalated
**Issue:** The comparison paragraph explaining the host-recurrence path's `from_torch` cost
model (lines 189–193) was flagged as MINOR in Pass 1 but not removed. On re-read it is the
strongest remaining cut. The paragraph hedges ("the Python dispatch overhead without Trace
is comparable") without stating a concrete number, introduces the host path as a confusing
frame of reference inside a section about the fused kernel's limitation, and contradicts the
preceding paragraph's cleaner argument. The preceding paragraph (lines 183–187) already
makes the core point: without Trace, `from_torch` overhead for 30 layers exceeds the savings.
Lines 189–193 add nothing quantifiable.

**Suggestion:** Delete lines 189–193 (the "The existing host-recurrence path incurs…"
paragraph in full). The section reads cleanly from line 188 directly into "With Metal Trace:"
at line 195. Save ~5 lines.

---

## MINOR Suggestions

### [recurrence_math.md] lines 66–67 — Pass 1 MINOR not applied
**Issue:** The code block preamble at lines 64–68 contains the sentence "The actual source
also contains `if not output_final_state: last_recurrent_state = None` before the return
statement, which is omitted here for clarity." The code block itself includes those exact
lines at lines 107–108, making the claim they were "omitted for clarity" false and
potentially confusing to a reader who looks at the code and finds them.

**Suggestion:** Delete the sentence "The actual source also contains…which is omitted here
for clarity" (lines 66–67). Save ~2 lines and remove the false statement.

---

### [host_recurrence.md] lines 35–37 — Pass 1 MINOR not applied
**Issue:** The sentence "This was confirmed empirically: running the recurrence in bfloat16
on device produces PCC values near 0 against the float32 reference after even a few tokens"
(lines 35–37) pre-announces a finding that is demonstrated in full with test code and
quantified thresholds later in the same file (lines 127–143, "Why bf16 Recurrence Is
Incorrect"). The early mention is weak verbal assertion; the later section is the rigorous
evidence. Keeping both means the reader gets a preview assertion followed by the same
conclusion again.

**Suggestion:** Remove the "This was confirmed empirically…" sentence from lines 35–37 and
let the dedicated section carry the argument alone. Save ~2 lines.

---

### [fused_kernel.md] line 206 — Inline number echo in cross-reference sentence
**Issue:** The cross-reference sentence added in Pass 1 reads: "See the full latency
breakdown in `host_recurrence.md` for the per-component profiling table (DeltaNet 54 ms /
Attention 18 ms / norm+LM head 14 ms / Total 86 ms)." The parenthetical re-lists all four
numbers from the table the reader is being directed to see. The cross-reference is useful;
the inline summary of the table's contents is redundant — the reader is about to see the
table.

**Suggestion:** Trim to: "See the full latency breakdown in `host_recurrence.md` for the
per-component profiling table." Save ~1 line (inline text reduction).

---

### [projections_and_conv.md] — B_pad definition repeated three times — Pass 1 MINOR not applied
**Issue:** `B_pad = tile_padded_batch_rows = 32` and the explanation "pads a single-sample
batch to the tile boundary" appear at lines 111–112 (ring buffer section), then the same
value and padding rationale recur implicitly at lines 161–165 and 174–176 (conv weight
layout section). The definition is stated or restated in three places within the same file.

**Suggestion:** State `B_pad = tile_padded_batch_rows = 32` fully once at its first use
(lines 111–112). In the Conv Weight Layout section, replace the second and third explanatory
uses with just `B_pad` with no redefinition. Save ~3 lines.

---

## Load-Bearing Evidence

- `recurrence_math.md` line 31 (inside the Note callout): "The update cannot be written as
  the single compressed form $S \leftarrow S \cdot \exp(g) + \mathbf{k} \otimes
  \boldsymbol{\delta}$ without loss of meaning…" — cannot be cut; it pre-empts the most
  natural but incorrect reading of the abbreviated recurrence used in host_recurrence.md.

- `host_recurrence.md` lines 50–74 (three-technique workaround list): `init_sfpu`+`copy_tile`,
  SFPU binary path, `binary_dest_reuse_tiles` — this is the only full explanation of the
  three Blackhole fp32 kernel workarounds in the chapter; fused_kernel.md now cross-references
  here, so none of these lines can be cut from host_recurrence.md.

- `projections_and_conv.md` lines 244–258 (design decision items 1–4): float32 state, bfloat16
  conv rows, `_oldest = 0` pointer, fixed tensor addresses — the only consolidated record of
  the four architectural constraints on `initialize_states`; referenced implicitly by both
  host_recurrence.md and fused_kernel.md.

- `fused_kernel.md` lines 218–235 (Path to Production steps 1–4): the only location in the
  chapter where the Trace integration sequence, kernel switch, multi-step verification, and
  optional multi-CQ overlap are enumerated as a concrete work plan.

- `recurrence_math.md` lines 131–138 (_neg_A_exp_dev constructor snippet): the precomputation
  of `-exp(A_log)` as a device constant is non-obvious and named by both projections_and_conv.md
  and the kernel input table; the explanation cannot be shortened without losing the motivation
  for the device tensor's existence.

---

## VERDICT
- Crucial updates: yes

---
## Change Log (Agent A — Pass 2 CRUCIAL fixes)
- fused_kernel.md: Removed hedging "existing host-recurrence path" comparison paragraph (~lines 189–193)

---

# Compression Analysis: GatedDeltaNet — Pass 3

## Summary
- Total files analyzed: 5
- Estimated current line count: ~951 lines (index.md: 32, recurrence_math.md: 242, projections_and_conv.md: 262, host_recurrence.md: 182, fused_kernel.md: 233)
- Estimated post-compression line count: ~943 lines
- Estimated reduction: ~1%

## CRUCIAL Suggestions
None — Pass 2 CRUCIAL confirmed applied.

**Verification:** The phrase "The existing host-recurrence path incurs" does not appear anywhere in the current `fused_kernel.md`. The section "Why the Fused Kernel Is Not in Production Today" (lines 176–204) now reads cleanly: the core argument (1–2 ms `from_torch` overhead × 30 layers > savings) is stated once in lines 183–187, then transitions directly to "With Metal Trace:" at line 189. The removal is confirmed.

**Fresh scan result:** No new crucial duplications or structural redundancies were found. The three remaining MINOR items from Pass 2 are still present and still do not rise to crucial severity.

---

## MINOR Suggestions

### [recurrence_math.md] lines 66–67 — Carry-over from Pass 1 and Pass 2; not yet applied
**Issue:** The code block preamble states "The actual source also contains `if not output_final_state: last_recurrent_state = None` before the return statement, which is omitted here for clarity." The code block at lines 107–108 includes those exact lines, making the claim of omission false.

**Suggestion:** Delete the sentence "The actual source also contains…which is omitted here for clarity" (lines 66–67). Save ~2 lines; removes a false statement.

---

### [host_recurrence.md] lines 35–37 — Carry-over from Pass 1 and Pass 2; not yet applied
**Issue:** "This was confirmed empirically: running the recurrence in bfloat16 on device produces PCC values near 0 against the float32 reference after even a few tokens." This pre-announces the finding demonstrated with test code and quantified thresholds later in the same file (lines 128–143). The early sentence is a weak verbal assertion; the later section is the rigorous evidence.

**Suggestion:** Remove the "This was confirmed empirically…" sentence (lines 35–37). Let the dedicated "Why bf16 Recurrence Is Incorrect" section carry the argument alone. Save ~2 lines.

---

### [fused_kernel.md] line 200 — Inline number echo in cross-reference sentence; carry-over from Pass 2
**Issue:** The cross-reference sentence reads: "See the full latency breakdown in `host_recurrence.md` for the per-component profiling table (DeltaNet 54 ms / Attention 18 ms / norm+LM head 14 ms / Total 86 ms)." The parenthetical restates all four numbers from the table the reader is being directed to see.

**Suggestion:** Trim to: "See the full latency breakdown in `host_recurrence.md` for the per-component profiling table." Save ~1 line of inline text.

---

## Load-Bearing Evidence

- `recurrence_math.md` line ~31: "> **Note:** The update cannot be written as the single compressed form $S \leftarrow S \cdot \exp(g) + \mathbf{k} \otimes \boldsymbol{\delta}$ without loss of meaning…" — load-bearing because it pre-empts the most natural but incorrect reading of the abbreviated three-line recurrence that appears in host_recurrence.md; without this note, the two forms appear equivalent when they are not.

- `host_recurrence.md` lines ~50–74: The three-technique workaround list (`init_sfpu`+`copy_tile`, SFPU binary path, `binary_dest_reuse_tiles`) — load-bearing as the sole full explanation of the Blackhole fp32 kernel workarounds in the chapter; fused_kernel.md now cross-references here, making this section the canonical source that cannot be trimmed.

- `projections_and_conv.md` lines ~244–258: Design decision items 1–4 (float32 state, bfloat16 conv rows, `_oldest = 0`, fixed addresses) — load-bearing because these four rationales are the only consolidated architectural record for `initialize_states`, referenced implicitly by both host_recurrence.md and fused_kernel.md.

- `fused_kernel.md` lines ~218–229: "Path to Production" steps 1–4 — load-bearing as the only location in the chapter where the Trace integration → kernel switch → multi-step verification → optional multi-CQ overlap sequence is enumerated as a concrete work plan.

- `recurrence_math.md` lines ~131–138: `_neg_A_exp_dev` constructor snippet — load-bearing because the precomputation of `-exp(A_log)` as a device constant is non-obvious and is named by both projections_and_conv.md and the kernel input table; the explanation cannot be shortened without losing the motivation for the device tensor's existence.

---

## VERDICT
- Crucial updates: no
