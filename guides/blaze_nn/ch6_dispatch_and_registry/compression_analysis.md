# Chapter 6 — Pass 1 Compression Analysis (Agent C)

Scope: redundancy and bloat only. No factual checks.

---

## Verdict

**Crucial updates: no.** Three minor compression opportunities; chapter is on the verbose end but every block carries a distinct teaching beat.

---

## Load-Bearing Evidence

- `functional_dispatch.md` Mermaid diagram + four-observation list + "Walking `_dispatch`" snippet collectively triple-cover the same control flow, but each layer (visual / prose / code) targets a different reader pass — the diagram for skim, the list for the four contract rules, the code for line-level pin. Removing any one shrinks the teaching surface.
- `registry.md` "current entries" enumeration (lines 31-57) is restated by the field-lifecycle table at lines 22-26 plus the "Two aliases and four placement hints" prose, but the three views serve distinct purposes — table = which-flag-where invariant, snippet = literal source, prose = friendly summary. Compressing the prose to a one-liner is the only safe cut.
- `caller_allocated_outputs_internals.md` Step 5 has two registration guards (`_fused_op_defined` flag + inner `if "blaze_nn_linear" in BlazeOp._class_registry: return`); the chapter spends a full paragraph (lines 188-189) calling this "belt-and-suspenders" — that explanation is load-bearing because the inner guard's purpose isn't obvious and Pitfall 1 reuses the idempotence framing.
- `caller_allocated_outputs_internals.md` "Recap of the chain" (lines 243-254) is a deliberate six-step recap of the five preceding steps; this is not bloat — `plan.md:243-249` makes the full chain a single-file deliverable and the recap is the file's mental-model handoff. Keep.

## MINOR Suggestions

1. **`caller_allocated_outputs_internals.md` lines 5-6 vs lines 22-23.** The "this page walks the full internal chain..." sentence at the top (line 5) enumerates `_lookup_user_allocated_outputs`, `_required_output_names`, `set_output_tensor[s]`, `_get_output_tensor`, the pre-forward check, and `define_fused_op` — then lines 22-23 enumerate the same five files participating in the chain immediately after the Mermaid. The Mermaid already carries both lists visually. Cut line 22's "Five files participate" sentence down to a single clause ("All steps live in `blaze_nn/modules/base.py` and `blaze_nn/modules/linear.py`, with one tt-blaze registry lookup.") — ~3 lines saved.

2. **`functional_dispatch.md` lines 138-144 "Anchoring tests" section.** The two-bullet test list duplicates content already woven into each preceding section: the "no active tracing context" raise is named at line 54, the `bias` raise at line 75, the closure caching at line 110, the chained-ops edge at line 135, the "totally_made_up_op_name" case at line 142. The "Anchoring tests" recap is a convenience but every claim is already pinned inline. Either drop the section entirely (~7 lines) or compress to a one-line "All claims on this page are anchored by `tests/test_functional.py` (framework-only) and `tests/test_dispatch_integration.py` (gated by `pytest.importorskip("blaze")`)."

3. **`registry.md` lines 47-57 "Aliases" + "Placement flags on backend ops" bullet groups.** These two bullet groups restate the dict literal at lines 33-42 in prose form, then the trailing paragraph at lines 56-57 re-restates that `linear`/`sliced_matmul` carry no placement flags — which is also visible in the dict and re-stated by the warning at line 138. Either drop the two bullet groups (the dict literal + the field-lifecycle table at lines 22-26 already convey the same information), or drop the trailing "Note that..." paragraph. ~5-7 lines saved.

## Summary

| File | Current lines | Suggested cuts | After |
| --- | --- | --- | --- |
| `index.md` | 19 | 0 | 19 |
| `functional_dispatch.md` | 148 | ~7 | ~141 |
| `registry.md` | 152 | ~5 | ~147 |
| `caller_allocated_outputs_internals.md` | 258 | ~3 | ~255 |
| **Total** | **577** | **~15** | **~562** |

Net reduction: ~2.6%. The chapter is dense but the density is earned — three contributor-facing internals (dispatch, registry, caller-allocated outputs) packaged into one chapter per `plan.md:220-249` budget. Most apparent "redundancy" is intentional layering (diagram → prose → code → test pin) that mirrors the plan's teaching convention.

---

_Reviewer: Agent C · Pass 1 · 2026-05-13_

---

## Pass 2

Scope: redundancy and bloat only. No factual checks. Re-scan after Pass 1 edits.

### Verdict

**Crucial updates: no.** Pass 1's verbosity assessment still holds; the Pass-1-applied fix to Step 4 (orchestrator inversion) introduced one new redundancy worth flagging, plus two carry-over compressions from Pass 1 remain untouched and worth re-noting at minor priority.

### Load-Bearing Evidence

- `caller_allocated_outputs_internals.md` Step 4 bullet #1 was rewritten in Pass 1 to fix Issue 1. The new prose (line 142) packs two paths into one sentence ("the user's top-level `model(x)` call … **and** any submodule call inside an orchestrator's `forward`") **then** spends a sub-clause re-explaining that an orchestrator bypasses `Module.__call__` and **then** lists three submodule names (`self.sdpa`, `self.qkv`, `self.o_proj`) — the three-name enumeration is illustrative but the orchestrator-bypass mechanism is already covered by the preceding clause. This is the only place in the chapter where Pass 1 edits introduced new prose that overlaps within a single bullet; flag as a minor compression candidate.
- Pass 1 Suggestion #1 (the duplicate "five files participate" enumeration at `caller_allocated_outputs_internals.md` lines 5-6 vs line 22) is untouched by Pass 1 edits and remains a valid minor cut. The Mermaid still does the visual work and the intro sentence still lists the same five files prose-form.
- Pass 1 Suggestion #2 (the "Anchoring tests" recap at `functional_dispatch.md` lines 137-144) is untouched and remains a candidate cut — every claim it makes is already pinned inline in the body sections.
- Pass 1 Suggestion #3 (`registry.md` lines 47-57 alias/placement-flag bullet groups duplicating the dict literal) is untouched and remains a candidate cut. The field-lifecycle table at lines 22-26 + the dict at lines 33-42 already convey the same content the prose bullets restate.

### MINOR Suggestions

1. **`caller_allocated_outputs_internals.md` line 142 — Step 4 bullet #1 internal duplication (new since Pass 1).** The bullet now reads ~6 sentences. The clause "An orchestrator (e.g. `Qwen3Attention.__call__` at `examples/qwen3_embedding_0_6b/modules/attention.py:90-91`) overrides `__call__` to call `self.forward(...)` directly — it bypasses `Module.__call__` (`base.py:68-72`) and therefore never opens a tracing context" already pins the mechanism; the follow-up "So `self.sdpa(...)`, `self.qkv(...)`, `self.o_proj(...)` inside `Qwen3Attention.forward` all see `_get_active_context() is None` and re-run the pre-check" is illustrative but partly restates the same claim. Either drop the three-name enumeration (saves ~1 line, keeps the mechanism) or compress to "Every submodule call inside `Qwen3Attention.forward` therefore re-runs the pre-check." The "discussed in Step 1 and Pitfall 1" tail is a useful cross-ref — keep that.

2. **Carry-over from Pass 1 Suggestion #1.** `caller_allocated_outputs_internals.md` line 22's "Five files participate: …" sentence still duplicates line 5's enumeration plus the Mermaid. Compress to a single clause as in Pass 1. ~3 lines saved.

3. **Carry-over from Pass 1 Suggestion #2.** `functional_dispatch.md` lines 137-144 "Anchoring tests" section still duplicates inline-anchored claims (line 54 for the no-active-context raise, line 75 for the bias raise, line 110 for closure caching, line 142 for `totally_made_up_op_name`). Either drop entirely (~7 lines) or compress to one line. Same recommendation as Pass 1.

4. **Carry-over from Pass 1 Suggestion #3.** `registry.md` lines 47-57 alias + placement-flag prose bullets still restate the dict literal at lines 33-42. Drop the two bullet groups or drop the trailing "Note that …" paragraph at lines 56-57. ~5-7 lines saved.

### Summary

| File | Pass 1 lines | Suggested cuts (Pass 2) | After |
| --- | --- | --- | --- |
| `index.md` | 19 | 0 | 19 |
| `functional_dispatch.md` | 148 | ~7 | ~141 |
| `registry.md` | 152 | ~5 | ~147 |
| `caller_allocated_outputs_internals.md` | 258 | ~4 | ~254 |
| **Total** | **577** | **~16** | **~561** |

Net reduction: ~2.8%. Pass 1's verdict that the chapter is dense-but-earned still holds. Pass 1's accuracy fixes were applied cleanly; the Step 4 bullet grew by ~3 lines but the new prose is mostly load-bearing — only the three-submodule-name enumeration is trimmable without losing teaching value. The three Pass-1 minor suggestions remain unaddressed, suggesting Agent A judged them below the bar — that judgement is reasonable.

---

_Reviewer: Agent C · Pass 2 · 2026-05-13_
