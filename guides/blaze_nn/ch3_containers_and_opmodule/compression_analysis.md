# Agent C (Compressor) — Chapter 3 analysis, Pass 1

## Summary

| File                                | Lines | Compressible region                                                | Est. lines saved |
|-------------------------------------|------:|--------------------------------------------------------------------|-----------------:|
| `index.md`                          |    12 | none (terse already)                                                |                0 |
| `sequential.md`                     |   100 | "What `Sequential` is not" partially overlaps the side-by-side table in next file | 4 |
| `modulelist_and_moduledict.md`      |   127 | per-container narrative restates info already in the side-by-side table (lines 105-114) | 6 |
| `opmodule_no_subclass.md`           |   138 | constructor numbered list (21-27) and the table (29-37) carry the same four facts; "What `OpModule(...)` is not" section restates premise of next file and `Sequential` content | 18 |
| `opmodule_subclass.md`              |   133 | "What the subclass form does not do" section (127-131) restates `m.to(device)` for the 3rd time in the chapter; "When to subclass" triage (lines 9-16) overlaps the table at 117-123 | 14 |
| `output_tensors.md`                 |    87 | concrete `Linear` example (27-50) overlaps the end-to-end pipeline in `prebuilt_modules.md` (66-108) — same test, same five lines | 0 (keep here — see CRUCIAL) |
| `prebuilt_modules.md`               |   118 | "End-to-end pipeline" section (66-108) duplicates `output_tensors.md`'s concrete example | 30 |

**Total chapter:** 741 lines (729 content). Compressible: roughly **70 lines (~10%)** without losing any source-of-truth pin or load-bearing claim.

## CRUCIAL Suggestions

### C1. `prebuilt_modules.md` lines 66-108 duplicate `output_tensors.md` lines 27-50.

Both walk the *exact same* `tests/test_pytorch_parity.py:test_linear_pipeline_matches_torch`. Both show: build `ttnn.Tensor` with `ttnn.from_torch(torch.zeros(...))`, `set_output_tensor`, `load_state_dict({"weight": ...})`, `to(device)`, call, comp_pcc. Both make the **identical** "order of the three pre-forward steps is independent" statement (`output_tensors.md:50` and `prebuilt_modules.md:108`).

The plan (line 17) explicitly says `output_tensors.md` should be "reduced to user-scope only (one example, one rule) per the v1 evaluator's recommendation, removing the v1 redundancy." The current Pass-1 draft re-introduces that redundancy at the *other* file.

**Recommendation:** in `prebuilt_modules.md`, collapse the "End-to-end pipeline" section (lines 66-108, ~40 lines) to ~10 lines: the 5-line blaze-nn block plus one sentence pointing at `output_tensors.md` for the order-independence rule and at the test for the full shard-spec setup. The PCC check fragment (lines 100-104) and the elaborate setup prose (lines 70-85) are not adding anything Chapter 3 needs to teach — the test reference is enough. Keep the four "facts" bullets in §`blaze_nn.Linear` (lines 14-19) since those *are* the new content this file owes the reader.

### C2. `opmodule_no_subclass.md` lines 21-37: numbered list + table cover the same four pieces of state.

Lines 21-27 enumerate the constructor's five steps; lines 29-37 then table four of the five state fields it set. The table headers `Slot / Source / Purpose` are *immediately* re-statements of items 3–5 of the numbered list. The current shape is "list, then table that paraphrases list" — a torch reader does not need both.

**Recommendation:** keep the numbered list (the action sequence is what matters for a reader following the lifecycle) and delete the table at 29-37. The four slot names (`_op_name`, `_param_slots`, `_op_kwargs`, `_required_output_names`) are already named in the prose; the table adds no new pin and no new vocabulary. Save ~10 lines.

## MINOR Suggestions

### M1. `opmodule_no_subclass.md` lines 130-136 — "What `OpModule(...)` is not" overlap.

Bullet 1 ("Not the only way to wrap one op... The subclass form is preferred when...") restates the opening sentences of `opmodule_subclass.md` (lines 3-4) and the "When to subclass" triage that immediately follows. Bullet 3 ("Not a `Sequential`-style composer") restates the `Sequential` scope already nailed down in `sequential.md` §"What `Sequential` is not". Keep bullet 2 (the `Unknown blaze op` error — that is the only fact this section actually owns). Trim ~6 lines.

### M2. `opmodule_subclass.md` lines 127-131 — "What the subclass form does not do" restates Ch2.

Bullet 3 ("It does not move parameters. `m.to(device)` records a `DeviceConfig`...") is the **third** time in this chapter the no-data-movement fact appears: it is in `opmodule_no_subclass.md:103` Warning, in `prebuilt_modules.md:106`, and again here. Ch2 `device_binding.md` is the source of truth. Either cut this bullet entirely or shorten to one line with a back-pointer.

### M3. `opmodule_subclass.md` lines 9-16 — "When to subclass" triage vs. lines 117-123 table.

Lines 9-16 give a 5-item numbered triage (when to subclass); lines 117-123 give a 5-row "When to subclass vs. instantiate" table that covers the same five cases with different wording. The two are 90 % overlapping. Pick one — the table is the more scannable form for a reference chapter; the numbered list can be dropped or shortened to a one-paragraph lead-in.

### M4. `modulelist_and_moduledict.md` — narrative + side-by-side table.

The three-row "containers side by side" table at 105-114 captures every load-bearing distinction (constructor shape, callable, key shape, state-dict prefix, mutation API, typical use) more crisply than the per-container narrative does. The narrative sections (lines 29-61 for `ModuleList`, 63-96 for `ModuleDict`) can each lose ~3 lines of "third thing to note: state-dict reads `<parent>.attn.weight`" / "Integer-stringified keys, same as `Sequential`" recaps that the table already states. Save ~6 lines.

### M5. `sequential.md` lines 85-92 — "What `Sequential` is not" partial duplication.

Bullet 1 ("Not a function composer... cannot pass `torch.nn.ReLU()`") and bullet 4 ("Not a place for non-`Module` callables") make the same point twice from two angles. Merge into one bullet.

### M6. `output_tensors.md` line 20-25 — "How to tell which modules need it" enumerates two paths but item 1 is just "read the docstring."

That bullet is filler — a reader does not need to be told to read a docstring. Cut item 1; keep item 2 (the `m._required_output_names` mechanical check) which is the actually-useful information.

### M7. `prebuilt_modules.md` line 116 contributor callout duplicates `output_tensors.md:85` callout.

Both forward-link to Ch7 `add_an_op_wrapper.md` for "add a new pre-built module." One contributor callout is enough; the one in `output_tensors.md` is in scope and the one at the end of `prebuilt_modules.md` repeats it. Trim the `prebuilt_modules.md` callout to one sentence or remove.

## Load-Bearing Evidence

- **`index.md`** — single-paragraph chapter intro plus 6-item ordered list of links; nothing compressible (each item is a file title, no prose elaboration).
- **`sequential.md`** — load-bearing: the `_register_indexed` snippet (lines 27-30) and the `forward` four-line loop (lines 54-58) are the only two code excerpts in the file, both <5 lines, both directly anchor the state-dict-keys-are-string-ints claim that the rest of the chapter inherits. Cannot be shortened without losing the pin.
- **`modulelist_and_moduledict.md`** — load-bearing: the `_NotCallableContainer` block (lines 12-22) is the source of the "raise on call" error message that the test anchors `match="not callable"` against; the `ModuleList` block (lines 34-44) shows `append` returning `self`, the chainability claim. The dict-shape table (lines 80-87) is the only place the `__iter__`-yields-keys asymmetry with `ModuleList` is captured. Keep all three.
- **`opmodule_no_subclass.md`** — load-bearing: the default `forward` block (lines 53-61) with its `F.<op>(*args, *params, **{op_kwargs, **call_kwargs})` shape is the canonical formulation the next file leans on; the `_collect_user_args` mention at 136 forward-links to Ch6 (the cross-chapter contract). Keep both. The constructor numbered list (21-27) is also load-bearing — it is the only place `_lookup_user_allocated_outputs` is named at user level before Ch6.
- **`opmodule_subclass.md`** — load-bearing: the `RMSNorm` source block (lines 22-33) and the `Linear` `define_fused_op` skeleton (lines 47-63) are the canonical small/complex pair the plan demands (plan line 142). The `_torch_init_specs` signature + `init_torch_params` device-required pitfall (lines 92-113) is the only place auto-init is documented at user level. Keep these three.
- **`output_tensors.md`** — load-bearing: the `RuntimeError` quote (lines 11-14) is the verbatim error message that pins the pre-forward check at `base.py:417-423`; the `_ua_blackhole_cores` line (78) is the only forward-link the chapter has to qwen3's `FusedQKV._ua_*` pattern (Ch4 `composing_submodules.md`). Both must stay.
- **`prebuilt_modules.md`** — load-bearing: the four `Linear` facts (lines 14-19) and the `RMSNorm` math identity (line 38) are the unique content this file owes the chapter. The `ops/` vs. `modules/` split table (lines 57-60) is the only place the directory convention is stated at user level. Keep all three.

## VERDICT

Crucial updates: **yes**.

Two CRUCIAL items — (1) `prebuilt_modules.md` end-to-end pipeline duplicates `output_tensors.md`'s concrete `Linear` walkthrough (same test, same 5-line sequence); (2) `opmodule_no_subclass.md` constructor numbered list (21-27) and the table at 29-37 paraphrase each other. Plus seven MINOR items, mostly recap/restatement trims. Net trimmable: ~70 lines (~10 % of chapter), no factual content lost.

---

## Agent A change log — applied after Pass 1 compression analysis
- `prebuilt_modules.md`: collapsed C1 — "End-to-end pipeline" section (orig lines 66-108, ~43 lines) reduced to a 10-line compressed form: 5-line blaze-nn snippet + one sentence on order-independence pointing at `output_tensors.md` + one sentence on the test's shard-spec/`comp_pcc` ownership + `RMSNorm` analogue pointer. Removed the abbreviated setup block, the PCC check fragment, and the "read out loud" recap. Footer intact, `Linear` 4-facts bullets and RMSNorm display equation preserved.
- `opmodule_no_subclass.md`: applied C2 — deleted the 9-line `Slot / Source / Purpose` table (orig lines 29-37) that paraphrased items 3–5 of the constructor numbered list immediately above. The four slot names (`_op_name`, `_param_slots`, `_op_kwargs`, `_required_output_names`) remain named in the prose and forward-link to `output_tensors.md` is preserved via the prose mention. Footer intact, default `forward` block and `_collect_user_args` forward-link preserved.

---

## Pass 2

Pass 2 reviewed the chapter after Agent A's Pass 1 compression. Both crucial fixes (C1, C2) landed cleanly; the seven Pass 1 MINOR items (M1–M7) were not applied. Pass 2 re-evaluates whether any surviving redundancy rises to CRUCIAL.

### Summary

| File                                | Lines | Compressible region                                                | Est. lines saved |
|-------------------------------------|------:|--------------------------------------------------------------------|-----------------:|
| `index.md`                          |    12 | none                                                                |                0 |
| `sequential.md`                     |   100 | "What `Sequential` is not" bullets 1 and 4 partially overlap (callable-vs-Module point) | 1 |
| `modulelist_and_moduledict.md`      |   127 | side-by-side table at 105-114 paraphrases narrative subsections; small "third thing to note" recaps | 4 |
| `opmodule_no_subclass.md`           |   129 | "What `OpModule(...)` is not" bullets 1 and 3 restate next-file intro / `Sequential` scope already covered | 5 |
| `opmodule_subclass.md`              |   133 | "When to subclass" triage (9-16) overlaps decision table (117-123); "does not move parameters" restated for 3rd time at 131 | 8 |
| `output_tensors.md`                 |    87 | "How to tell" item 1 (read the docstring) is light filler | 2 |
| `prebuilt_modules.md`               |    88 | contributor callout at 86 partially restates `output_tensors.md:85` cross-link | 2 |

**Total chapter:** 803 lines (incl. analysis/review). Compressible: roughly **22 lines (~3 %)** — small and all in the "minor recap" category.

### CRUCIAL Suggestions

None. The two CRUCIAL items from Pass 1 (C1, C2) were both applied correctly; no new chapter-level duplication has surfaced and no surviving overlap is load-bearing enough to qualify.

### MINOR Suggestions

#### N1. `opmodule_subclass.md:127-131` — "What the subclass form does *not* do" bullet 3 restates the no-data-movement fact a third time.

Bullet 3 ("It does not move parameters. `m.to(device)` records a `DeviceConfig` ...") repeats the same content already stated in `opmodule_no_subclass.md:94` Warning and (transitively) in Ch2 `device_binding.md`. Trim to one sentence with a back-pointer, or drop entirely. ~3 lines.

#### N2. `opmodule_subclass.md:9-16` vs. `:117-123` — triage list and decision table cover the same five cases.

Lines 9-16 (5-item numbered "When to subclass") and lines 117-123 (5-row "When to subclass vs. instantiate" table) are 90 % overlapping. The numbered list is the more discursive form, the table the more scannable. Either drop the list (and rely on the table later in the file) or replace the table with a one-line back-reference. ~5 lines.

#### N3. `opmodule_no_subclass.md:121-127` — "What `OpModule(...)` is not" partial overlap.

Bullet 1 ("Not the only way to wrap one op… The subclass form is preferred when…") restates the `opmodule_subclass.md` opening sentences (lines 3-4) and the "When to subclass" triage. Bullet 3 ("Not a `Sequential`-style composer") restates `sequential.md`'s scope. Keep bullet 2 (the `Unknown blaze op` error — the only fact this section owns). ~5 lines.

#### N4. `modulelist_and_moduledict.md` — narrative recap reiterated by the side-by-side table.

The three-row table at 105-114 captures every load-bearing distinction (constructor shape, callable, key shape, state-dict prefix, mutation API, typical use) more crisply than the per-container narrative. The "Three things matter" recap in `ModuleList` (47-51) and the "Two consequences" recap in `ModuleDict` (98-101) each repeat one or two table rows. Trim each by ~2 lines.

#### N5. `prebuilt_modules.md:86` contributor callout partially duplicates `output_tensors.md:85`.

Both forward-link to Ch7 `add_an_op_wrapper.md` / `add_a_fused_op.md` for "add a new pre-built module." The `output_tensors.md` callout is in scope at the chain-of-mechanisms position; the `prebuilt_modules.md` end-of-file callout is the duplicate. Trim to one sentence. ~2 lines.

#### N6. `output_tensors.md:18-23` — "How to tell" item 1 is light filler.

Item 1 ("Read the class docstring") is correct but not actionable beyond what any reader would do by default; item 2 (`m._required_output_names`) is the mechanical check that earns the section. Either shorten item 1 to a parenthetical or merge into item 2's lead. ~2 lines.

#### N7. `sequential.md:89, 92` — "Not a function composer" and "Not a place for non-`Module` callables" partial overlap.

Bullet 1 (no `torch.nn.ReLU()`/bare lambda) and bullet 4 (`__setattr__` only routes `Parameter`/`Module`) make the same point from two angles — both reduce to "children must be `Module` instances." Merge into one bullet. ~1 line.

### Load-Bearing Evidence

- **`index.md`** — unchanged at 12 lines; chapter intro + 6-item link list. Nothing compressible.
- **`sequential.md`** — load-bearing: the `_register_indexed` snippet (27-30) and `forward` loop (54-58) are the two minimal code excerpts and they anchor the string-cast-integer-keys claim that propagates through the rest of the chapter. Cannot be trimmed without losing the pin.
- **`modulelist_and_moduledict.md`** — load-bearing: the `_NotCallableContainer` block (12-22) is the source of the `match="not callable"` error message tests anchor; the `ModuleList` block (33-45) shows `append` returning `self`; the `ModuleDict` dict-shape table (80-87) is the only place the `__iter__`-yields-keys asymmetry is captured. All three must stay.
- **`opmodule_no_subclass.md`** — load-bearing: the default `forward` block (43-52) with its `F.<op>(*args, *params, **{op_kwargs, **call_kwargs})` shape is the canonical formulation the next file leans on; the constructor numbered list (21-27) is the only place all five `__init__` actions and four slot fields appear at user level; the `_collect_user_args` forward-link at 127 is the cross-chapter contract. Keep all three.
- **`opmodule_subclass.md`** — load-bearing: the `RMSNorm` source block (22-33) and the `Linear` `define_fused_op` skeleton (47-64) are the canonical small/complex pair the plan demands (plan line 142). The `_torch_init_specs` signature + `init_torch_params` device-required pitfall (92-113) is the only place auto-init is documented at user level. Keep these three.
- **`output_tensors.md`** — load-bearing: the `RuntimeError` quote (11-14) is the verbatim error pinned at `base.py:417-423`; the `_ua_blackhole_cores` example (78) is the only forward-link the chapter has to qwen3's `FusedQKV._ua_*` pattern (Ch4 `composing_submodules.md`). Both must stay.
- **`prebuilt_modules.md`** — load-bearing: the four `Linear` facts (14-19) and the `RMSNorm` math identity (38) are unique content this file owes the chapter; the `ops/` vs. `modules/` split table (57-60) is the only place the directory convention is stated at user level. The compressed end-to-end snippet (68-78) survived C1 collapse and now sits at the right granularity. Keep all four.

### VERDICT

Crucial updates: **no**.

Pass 1's C1/C2 fixes addressed the only chapter-scale duplications. Surviving overlaps are recap-level (~22 lines across seven minor items, ~3 % of chapter) and none cross a load-bearing boundary or restate factual content in a way that would mislead a reader. The chapter is in good shape.
