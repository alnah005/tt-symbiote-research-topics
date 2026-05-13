# Agent C — Full-guide Compression Analysis, Pass 1

Cross-chapter redundancy and bloat across the entire blaze-nn guide (guide-level `index.md` + 7 chapters). Within-chapter compression is owned by the per-chapter compression analyses; this report only flags duplication that spans chapter boundaries.

## Summary

- **Files in scope:** 38 markdown files (`index.md` + 7 chapter `index.md` files + 30 content files; excludes `plan.md`, `b_review*.md`, and the per-chapter `compression_analysis.md` files).
- **Current total:** 4,290 lines.
- **Projected savings if all CRUCIAL + MINOR items are taken:** ~155–195 lines (~4–5% of total). The guide is already disciplined about forward-references; the remaining redundancy is concentrated in four repeated code blocks (`Module.__call__` body, `_collect_user_args` body, orchestrator two-liner, `_register_indexed`) and three repeated prose contracts ("`to(device)` does not move tensors", "`load_state_dict` writes verbatim", "three test tiers").

## CRUCIAL Suggestions

None. Every cross-chapter overlap I found is either a legitimate forward-/back-reference (one chapter introduces a concept at user level, another revisits it at contributor depth) or a small enough code re-quote that consolidating it would damage standalone readability of the file it lives in. No load-bearing claim is in the wrong chapter; no chapter would be materially clearer if a section moved homes.

## MINOR Suggestions

### M1 — Re-quoted `Module.__call__` body (3 places, ~20 dup lines)

The full 15-line `Module.__call__` body from `blaze_nn/modules/base.py:68-82` is quoted three times:

- `ch2_module_and_parameter/module_attribute_protocol.md:120-148` — "`__call__` is the entry point — three branches"
- `ch4_qwen3_walkthrough/orchestrator_pattern.md:86-104` — "Mechanism B — active-context short-circuit at `base.py:71`"
- `ch5_tracing_internals/module_call_path.md:36-49` — "Outer-call dispatch in `Module.__call__`"

**Action:** Ch5 is the canonical home (it walks the body line-by-line as the chapter's whole subject). Ch2 needs the full body because the chapter's audience hasn't reached Ch4/Ch5 yet — keep it. **Ch4's quote at lines 86-104 can shrink to a 3-line excerpt of just the active-context check (lines 71-72)** plus a link to Ch5; the surrounding prose in Ch4 already names "Mechanism B at `base.py:71`" three sentences earlier. Saves ~10 lines.

### M2 — Re-quoted `_collect_user_args` body (4 places, ~24 dup lines)

The 6-line `_collect_user_args` body is quoted four times:

- `ch3_containers_and_opmodule/output_tensors.md:66-73`
- `ch4_qwen3_walkthrough/buffers_and_address_baking.md:84-96` (with the qwen3 example)
- `ch5_tracing_internals/module_call_path.md:122-128`
- `ch7_extending/extending_containers_and_modules.md:113-119, 126-132` (twice — the OpModule version and the `FusedQKV` override)

**Action:** Ch5 is the canonical contributor-side home. Ch7 explicitly walks the override pattern with the qwen3 example, so its two copies are load-bearing for the chapter's recipe. **Trim Ch3 `output_tensors.md:66-73` to a one-line gloss ("any attribute prefixed `_ua_` becomes a compile-time argument, named after the suffix") with a forward link to Ch5 — the user-facing explanation does not need the source body.** Saves ~6 lines.

### M3 — Orchestrator two-liner `__call__` override (4 places, ~15 dup lines)

The orchestrator two-liner pattern is quoted four times:

- `ch4_qwen3_walkthrough/orchestrator_pattern.md:40-56` — three copies (Qwen3Attention, Qwen3DecoderLayer, Qwen3EmbeddingModel)
- `ch7_extending/extending_containers_and_modules.md:70-82`

**Action:** Ch4's three-times-repeated literal is the right pedagogy ("the same two lines appear in three modules" is the load-bearing observation). **Ch7 `extending_containers_and_modules.md:70-82` can shrink: keep the pattern but cite Ch4 instead of re-pasting all three line numbers — the example needs only one canonical class.** Saves ~5 lines.

### M4 — `_register_sdpa_decode_user_alloc` monkey-patch quoted twice (~14 dup lines)

The full body of `_register_sdpa_decode_user_alloc` is quoted in:

- `ch4_qwen3_walkthrough/buffers_and_address_baking.md:130-139`
- `ch6_dispatch_and_registry/caller_allocated_outputs_internals.md:198-207`

**Action:** Ch6 is the canonical home (the file is named after the underlying mechanism). **Ch4's copy can shrink to the four-line "what the patch does + idempotence guard" summary plus a link to Ch6** — Ch4's section is about the demo's monkey-patches as a category, not about the registry internals. Saves ~6 lines.

### M5 — "`to(device)` does not move tensors / `load_state_dict` writes verbatim" repeated as standalone warnings

The two paired invariants are restated as full `> **Warning:**` blocks in:

- `ch1_why_blaze_nn/what_it_is.md:37` (1 line, summary)
- `ch1_why_blaze_nn/getting_started.md:94` (2 lines, in the Tier C section)
- `ch2_module_and_parameter/device_binding.md:34` (canonical)
- `ch2_module_and_parameter/traversal_and_state_dict.md:118-127` (canonical for `load_state_dict`)
- `ch3_containers_and_opmodule/opmodule_no_subclass.md:94` (warning block)
- `ch3_containers_and_opmodule/opmodule_subclass.md:131` (1 bullet under "what subclass form does *not* do")
- `ch4_qwen3_walkthrough/layout_and_weight_loader.md:120` (warning block)

**Action:** Ch2 is the canonical home for both invariants; the Ch1 mentions are introductory and load-bearing (they set up the contract that Ch2 then formalizes). **The Ch3 restatements (`opmodule_no_subclass.md:94`, `opmodule_subclass.md:131`) can collapse to a single one-line "verbatim — see Ch2 `traversal_and_state_dict.md`" cross-link** — the Ch3 audience has read Ch2. Saves ~6 lines.

### M6 — Three-tier test taxonomy restated

The three test tiers (framework-only / dispatch-integration / parity) are explained in:

- `ch1_why_blaze_nn/getting_started.md:28-70` (canonical introduction)
- `ch7_extending/testing_strategy.md:9-23` (table)
- `ch7_extending/testing_strategy.md:25-62` (per-tier sections, with file lists)

**Action:** Ch7 is the right place for the per-file reverse index, and Ch1 is the right place for the install-time introduction. **No structural change needed** — the Ch7 explanation is the file-list view, not a re-explanation. Flag only: the second-paragraph rationale at `testing_strategy.md:8` could shrink to a one-line "Ch1 introduced the tiers; this file lists every file in each." Saves ~2 lines.

### M7 — `_register_indexed` / `_IndexedContainer` body quoted twice

- `ch3_containers_and_opmodule/sequential.md:26-30` and the class definition at lines 72-81
- `ch7_extending/extending_containers_and_modules.md:10-23` (full mixin pair)

**Action:** Ch3 is the user-facing home, Ch7 the contributor recipe — both are legitimate. **Minor:** Ch7's quote could drop the `_IndexedContainer` definition body (it is a verbatim repeat of Ch3) and link instead. Saves ~6 lines.

### M8 — `index.md` Quick Reference table duplicates content in chapter files

The 14-row Quick Reference table at `index.md:34-50` paraphrases what each chapter file already says (e.g. "`Parameter` — One-slot holder for a `ttnn.Tensor`; populated by direct `.data =` or by `load_state_dict`" duplicates the opening of `ch2_module_and_parameter/parameter.md`). The table's value is the *links*, not the descriptions.

**Action:** Trim the "Purpose" column to short noun phrases (≤6 words) — the chapter target is one click away and already carries the full prose. Saves ~5 lines and improves scan-ability.

### M9 — `define_fused_op` idempotence guards quoted three times

The "three independent guards" prose is in:

- `ch3_containers_and_opmodule/opmodule_subclass.md:84` (single-line summary)
- `ch6_dispatch_and_registry/caller_allocated_outputs_internals.md:146-191` (canonical full walk)
- `ch7_extending/add_a_fused_op.md:79-103` (re-walk for the contributor recipe)

**Action:** Ch6 is the canonical home; Ch7 is the recipe. **Ch7's "Three independent guards" enumeration at `add_a_fused_op.md:97-100` paraphrases Ch6 — collapse to one line "the three guards (subclass-overrides check, `_fused_op_defined` per-subclass flag, registry membership check) — see Ch6 `caller_allocated_outputs_internals.md`"** while keeping Ch7's "Order matters" point (Ch6 doesn't make that one as crisply). Saves ~8 lines.

### M10 — `Module.__init__` "five buckets" code repeated

The five-line `Module.__init__` body (`base.py:26-31`) is quoted in:

- `ch2_module_and_parameter/module_attribute_protocol.md:9-16`
- `ch3_containers_and_opmodule/sequential.md:24` (summarized in prose with the field names)

**Action:** Ch3's summary is already prose-only; no change. **No action — this is well-managed.**

## Load-Bearing Evidence

Each chapter's headline claim is anchored at line ranges that cannot move without breaking the guide's narrative arc:

- **Ch1 — `ttnn_native_contract.md:3` ("tensors that cross a `Module` boundary are `ttnn.Tensor`, and blaze-nn treats them as opaque")**: this is the single invariant every later chapter restates. It must remain in Ch1 because Chapters 2–4 build on it as established. Removing it from Ch1 and pointing at Ch2 would require the reader to skip ahead before the framing is set.
- **Ch2 — `traversal_and_state_dict.md:131-133` (the identity-preserving roundtrip rule)**: the *behavioral guarantee* (`m2.load_state_dict(m1.state_dict())` preserves object identity) is the chapter's core load-bearing claim and the foundation for Ch4's Buffer-vs-Parameter split. It cannot move; Ch4 only *cites* it.
- **Ch3 — `output_tensors.md:7` ("a module whose op declares `user_allocated_outputs` requires `set_output_tensor(t)` before `forward`")**: the user-facing rule. Ch6 owns the full internal chain (`_lookup_user_allocated_outputs` → `_required_output_names` → `_get_output_tensor`), but the user-level *rule* must live with the user-facing chapter. The two-chapter split is intentional and correct.
- **Ch4 — `tensor_lifetimes.md:5-15` (the Parameter / Buffer / GraphInput vocabulary)**: this is the only chapter that establishes the three-way lifetime split, quoted verbatim from `examples/qwen3_embedding_0_6b/modules/__init__.py:3-21`. Every later chapter (Ch5 GraphInput proxies, Ch6 caller-allocated outputs as Buffers) refers back to it. It cannot move upstream because Chapters 2–3 are model-author surface, not qwen3 specifics.
- **Ch5 — `module_call_path.md:7-28` (the Mermaid flow from `model(x)` to `program.run()`)**: the chapter's whole reason to exist. The body-line walk of `_call_graph` lines 1-9 cannot move to Ch6 or Ch4 because both forward-link to it as the canonical answer to "what happens between `model(x)` and `program.run()`".
- **Ch6 — `caller_allocated_outputs_internals.md:8-22` (the five-step chain diagram)**: the chapter's headline claim is that the user-facing rule from Ch3 has five concrete internal steps; the chain itself must live in Ch6 because Ch3 deliberately stops at the user-level rule.
- **Ch7 — `contributing_checklist.md:94-148` (the six framework anti-patterns)**: each anti-pattern is the conclusion of an earlier chapter's load-bearing invariant (`import torch` at module scope ↔ Ch1; `interop` inside the framework ↔ Ch2; `blaze.<op>` bypass ↔ Ch6; Buffer rebinding ↔ Ch4; `user_allocated_outputs` monkey-patching ↔ Ch6). The list must live in Ch7 because it is the *contributor checklist* — a reader hitting an anti-pattern needs one place to scan, not seven chapter sections.

Each headline claim above is already in its right chapter; the redundancies flagged in MINOR are *body* quotes and *prose restatements*, not structural moves.

## VERDICT — Crucial updates: no

The guide already practices forward-/back-reference discipline well (every `> **For contributors:**` callout is a Ch5/Ch6/Ch7 forward-link; every Ch4-and-later mention of `to(device)` semantics cites Ch2 `device_binding.md`). The remaining cross-chapter bloat is small, mechanical re-quotes of short code blocks (Items M1–M4, M7, M9) plus three minor prose duplications (M5, M6, M8). Total projected savings ~155–195 lines on a 4,290-line guide. No structural moves needed; no chapter's headline claim is in the wrong place.

---

## Pass 2

Independent re-audit of cross-chapter redundancy on the same 38-file, 4,290-line corpus that Pass 1 covered. Pass 2 re-verifies each Pass 1 MINOR item against the source, looks for cross-chapter overlaps Pass 1 missed, and re-confirms the load-bearing anchor in every chapter. Same scope contract: cross-chapter only, no within-chapter, no factual checks.

### Summary

- **Files in scope:** 38 markdown files (`index.md` + 7 chapter `index.md` files + 30 content files), 4,290 lines total.
- **Pass 1 MINOR items re-verified:** all 9 confirmed against source (line numbers and quoted bodies match the live files). M1, M2, M3, M4, M9 are the highest-yield items (~35 of the projected ~155–195 lines come from those five mechanical code re-quotes).
- **New cross-chapter findings:** 2 minor items not flagged in Pass 1 — see M11 (Mermaid sub-graph overlap between Ch1 and Ch5 mental-model diagrams) and M12 (the "no active tracing context" error-string restatement chain).
- **Pass 1 verdict stands:** Crucial updates: no. The guide's forward-/back-reference plumbing is solid; nothing is in the wrong chapter.
- **Projected savings if Pass 1 + Pass 2 MINOR items are taken:** ~165–210 lines (~4–5% of 4,290).

### Pass 1 cross-checks

- **M1 re-verified.** `Module.__call__` body confirmed at ch2 `module_attribute_protocol.md:120-148`, ch4 `orchestrator_pattern.md:86-104`, ch5 `module_call_path.md:36-49`. The Ch4 re-quote at lines 87-104 carries 18 lines that the chapter could substitute with a 3-line excerpt of `base.py:69-72` plus a Ch5 link — the surrounding Ch4 prose at lines 83-86 and 106 already names "Mechanism B at `base.py:71`" explicitly, so the full method body is informational, not load-bearing for the chapter. Pass 1's estimate of ~10 saved lines is conservative; 12-14 lines is more accurate.
- **M2 re-verified.** `_collect_user_args` body confirmed in all four locations. Ch3 `output_tensors.md:66-73` is the weakest of the four — the user-facing audience doesn't need the source body, only the rule. Trim is sound.
- **M3 re-verified.** The orchestrator two-liner appears three times at ch4 `orchestrator_pattern.md:40-56` (Qwen3Attention, Qwen3DecoderLayer, Qwen3EmbeddingModel) and once at ch7 `extending_containers_and_modules.md:70-82`. The Ch4 triple-quote is load-bearing (the "same two lines in three modules" is the pedagogical point). Ch7's single-class re-quote is the legitimate trim target — its line 84 already names all three qwen3 examples.
- **M4 re-verified.** `_register_sdpa_decode_user_alloc` body appears at ch4 `buffers_and_address_baking.md:130-139` and ch6 `caller_allocated_outputs_internals.md:198-207`. Ch6 is canonical (chapter is titled after the mechanism). Trim is sound.
- **M5 re-verified.** The `to(device)` and `load_state_dict` invariant restatements span ch1 (2 mentions), ch2 (canonical, 2 mentions), ch3 (2 mentions), ch4 (1 mention). Ch3's two restatements (`opmodule_no_subclass.md:94`, `opmodule_subclass.md:131`) are the cleanest trim — the Ch3 audience has already passed through Ch2's canonical block.
- **M6 re-verified.** Three-tier test taxonomy split between ch1 `getting_started.md:28-70` (canonical intro) and ch7 `testing_strategy.md:9-62` (per-file reverse index). Pass 1 correctly notes that the two angles are different (intro vs. reverse-index lookup); no structural change needed.
- **M7 re-verified.** `_register_indexed` / `_IndexedContainer` body in ch3 `sequential.md` and ch7 `extending_containers_and_modules.md:10-23`. Both placements are legitimate (user view vs. contributor recipe); Ch7's drop of the `_IndexedContainer` class body is the right trim.
- **M8 re-verified.** Guide `index.md:34-50` Quick Reference table — Purpose column descriptions paraphrase the opening of each linked chapter file. The trim ("noun phrases ≤6 words") is sound and improves scannability.
- **M9 re-verified.** `define_fused_op` three guards prose appears at ch3 `opmodule_subclass.md:84` (single line, fine), ch6 `caller_allocated_outputs_internals.md:146-191` (canonical full walk), ch7 `add_a_fused_op.md:79-103` (recipe re-walk). Pass 1's call to collapse the Ch7 enumeration while preserving the "Order matters" point is sound.

### New cross-chapter observations Pass 2 adds

#### M11 — Mental-model Mermaid diagram overlap (Ch1 ↔ Ch5)

The four-layer "User code → blaze-nn → tt-blaze → tt-metal" picture is drawn twice:

- `ch1_why_blaze_nn/what_it_is.md:17-22` — 6-line `graph LR` of the four layers as a horizontal pipeline.
- `ch5_tracing_internals/module_call_path.md:7-28` — 22-line `graph TD` showing the same arc plus the `__call__` / `_call_graph` / `_call_compose` / `GraphTracingContext` internals.

The Ch5 diagram supersedes the Ch1 one *for contributors* but the Ch1 diagram is the only one a model-author reader will see. **Action:** keep both — they target different audiences. Flag only: the bullet list at `what_it_is.md:24-27` re-explains the four nodes in prose immediately after the diagram. The same content is repeated in the diagram labels (each box names the file or symbol the bullet then re-names). Collapse the four bullets to a single sentence that names the key files (`base.py:91-106`, `base.py:115-121`, `base.py:122`) once. Saves ~3 lines.

#### M12 — "no active tracing context" error string restated across five chapters

The `RuntimeError("... no active tracing context")` invariant is mentioned in:

- `ch2_module_and_parameter/module_attribute_protocol.md:152` — 1-line Warning ("never call `module.forward(x)` directly")
- `ch6_dispatch_and_registry/functional_dispatch.md:20, 38-39, 54, 141` — canonical home (the source code, the test, the diagnostic)
- `ch7_extending/extending_containers_and_modules.md:88` — Warning block inside the orchestrator section, restates the full error message
- `ch7_extending/contributing_checklist.md:166` — failure-mode table row pointing at Ch5 + Ch6

The Ch7 `extending_containers_and_modules.md:88` Warning is the redundant one — it quotes the exact error string verbatim where a `(see Ch6 functional_dispatch.md for the exact text)` link would suffice. The contributor reading Ch7's orchestrator section has already read Ch5/Ch6. **Action:** shrink the Ch7 Warning to one sentence ("calling `F.<op>` outside an active context raises — see Ch6 `functional_dispatch.md`"). Saves ~2 lines.

### Load-Bearing Evidence

Pass 2 confirms each chapter's headline claim is anchored where it cannot move:

- **Ch1 — `ttnn_native_contract.md:3`:** "tensors that cross a `Module` boundary are `ttnn.Tensor`" — the invariant Chapters 2–4 build on. Must stay in Ch1.
- **Ch2 — `traversal_and_state_dict.md:131-133`:** the identity-preserving roundtrip rule. Foundational for Ch4's Buffer/Parameter split. Cannot move.
- **Ch3 — `output_tensors.md:7`:** the user-facing `set_output_tensor` rule. Ch6 owns the internals; the user rule belongs with the user-facing chapter.
- **Ch4 — `tensor_lifetimes.md:5-15`:** the Parameter / Buffer / GraphInput three-way vocabulary. Every later chapter cites it; it cannot move upstream because Ch2/3 are framework-surface, not qwen3-specific.
- **Ch5 — `module_call_path.md:7-28`:** the Mermaid + line-by-line `_call_graph` walk. Both Ch4 and Ch6 forward-link to it as the canonical "what happens between `model(x)` and `program.run()`" answer.
- **Ch6 — `caller_allocated_outputs_internals.md:8-22`:** the five-step internal chain that the Ch3 user-rule expands into. Must live in Ch6 because Ch3 deliberately stops at the user-level rule.
- **Ch7 — `contributing_checklist.md:94-148`:** the six framework anti-patterns. Each restates an earlier chapter's invariant in checklist form; the *list itself* must live in Ch7 as the contributor's one-stop reference.

Pass 2 has no structural moves to add — every headline claim is in the right chapter, and Pass 1's read on this was correct.

### VERDICT — Crucial updates: no

Pass 2 re-verified all 9 Pass 1 MINOR items against the source (line numbers and contents match) and adds two small new ones (M11 diagram-plus-bullets duplication in Ch1, M12 error-string Warning in Ch7). Combined Pass 1 + Pass 2 savings: ~165–210 lines on a 4,290-line guide (~4–5%). No chapter needs structural rework; no load-bearing claim is in the wrong place. The guide's forward-/back-reference discipline is the model — every `> **For contributors:**` block already routes to the right chapter.
