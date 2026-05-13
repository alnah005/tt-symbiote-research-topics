# Agent B Full-Guide Review — Pass 1

## Issue 1 — Cross-chapter

**Location:** `ch4_qwen3_walkthrough/orchestrator_pattern.md` lines 149 and 155 vs. `ch5_tracing_internals/module_call_path.md` line 138 (and `ch4_qwen3_walkthrough/orchestrator_pattern.md` line 174 within the same Ch4 file).

**Problem:** Ch4 contradicts Ch5 — and itself — on `_compiled_cache` behavior. Ch4 `orchestrator_pattern.md:149` asserts as a present-tense fact: *"After that, the per-Module `_compiled_cache` lets subsequent calls skip the compile and re-run the same program."* Ch4:155 reinforces this: *"Each child module ... maintains its own `_compiled_cache` on the framework side. The orchestrator merely orchestrates calls; cache hits and misses happen per child."* But Ch5 `module_call_path.md:138` (and the plan.md spec at line 203) explicitly says: *"Nothing in the framework reads or writes this dict today ... Do not depend on its presence today, and do not populate it speculatively."* The contributor callout in Ch4:174 in the same file already concedes the cache is *"currently unused but reserved"* — internally inconsistent with Ch4:149 and Ch4:155 two paragraphs earlier. A user reading Ch4 in order will leave with the wrong mental model ("subsequent calls skip compile via `_compiled_cache`") and then be told the opposite when they reach Ch5.

**Required fix:** In `ch4_qwen3_walkthrough/orchestrator_pattern.md`, rewrite line 149 to remove the present-tense `_compiled_cache` claim (the diagram caption should say *"the first call compiles each `own graph` box; subsequent calls re-run the same compiled program — the per-Module compile-result cache hook (`_compiled_cache`) is currently unused, so today every call re-compiles; see Ch5"*). In line 155, rewrite the first bullet to match: each child *would* maintain its own `_compiled_cache` once the hook is wired up; today no caching happens at any level. Keep the Ch4:174 contributor callout phrasing ("currently unused but reserved") as the source of truth for both spots.

---

(No further items: cross-chapter link targets all resolve to existing files in the chapter directory listings; navigation Previous/Next chains across the six chapter boundaries — Ch1→Ch2, Ch2→Ch3, Ch3→Ch4, Ch4→Ch5, Ch5→Ch6, Ch6→Ch7 — all match correctly; the guide `index.md` and chapter `index.md` enumerations match the actual file set on disk; `_resolve_grid` priority and the three matmul-cores entries are stated identically in Ch5 `tracing_contexts.md` and Ch6 `registry.md`; the orchestrator Mechanism A / Mechanism B framing is consistent between Ch4 `orchestrator_pattern.md` and Ch5 `module_call_path.md`; `Parameter` behavior between Ch2 `parameter.md` and Ch4 `tensor_lifetimes.md` is consistent; the `init_position_ids` vs `set_position_ids` distinction is named the same way in `tensor_lifetimes.md` and `buffers_and_address_baking.md`; plan.md's chapter directory names and file names all match the on-disk layout.)

---

## Agent A change log — applied after Pass 1 full-guide B review
- Issue 1: Rewrote `ch4_qwen3_walkthrough/orchestrator_pattern.md:149` to remove the present-tense claim that `_compiled_cache` lets subsequent calls skip the compile; now states that each child sub-module call opens its own tracing context and re-compiles from scratch, and that `_compiled_cache` (`base.py:30`) is a dormant future-extension hook (pointer to Ch5 `module_call_path.md`). Rewrote `ch4_qwen3_walkthrough/orchestrator_pattern.md:155` first bullet to remove the "each child maintains its own `_compiled_cache`" claim; now states that no module — orchestrator or child — caches today, the dict is never read or written by the framework, and every child call re-compiles fresh. Both edits aligned with Ch5 `module_call_path.md:138` and Ch4's own contributor callout at line 174. Verified via `grep -rn _compiled_cache /home/ttuser/salnahari/blaze-nn/` — exactly one hit, the allocator at `blaze_nn/modules/base.py:30`, no readers or writers anywhere else in the codebase.

---

## Pass 2

Pass 1 verification: the `_compiled_cache` edits at `ch4_qwen3_walkthrough/orchestrator_pattern.md:149` and `:155` are in place and align with `ch5_tracing_internals/module_call_path.md:138`. Ch4:174 contributor callout ("currently unused but reserved") remains the third anchor, and all three now agree.

## Issue 1 — Cross-chapter (Pass 1 fix did not propagate to the same file's earlier paragraph)

**Location:** `ch4_qwen3_walkthrough/orchestrator_pattern.md:124` vs. its own corrected `:149` and `:155`, and vs. `ch5_tracing_internals/module_call_path.md:138`.

**Problem:** Line 124 still asserts as a positive property of the orchestrator pattern: *"This is exactly the property the orchestrator pattern is buying: every per-layer matmul / norm / RoPE / SDPA is a discrete, **independently-cached compile**."* The "independently-cached" claim is exactly the misconception Pass 1 corrected — 25 lines later at `:149`, Agent A's fix now reads "each child sub-module call opens its own tracing context and re-compiles from scratch; there is no per-Module compile-result cache wired up today," and `:155` says "no module — orchestrator or child — caches today, the dict is never read or written by the framework, and every child call re-compiles fresh." Ch5 `module_call_path.md:138` is the source of truth: "Nothing in the framework reads or writes this dict today." Line 124 lives inside the same Mechanism-B-fires-in-FusedQKV paragraph that Pass 1 left unmodified, so the stale wording was carried over silently. A user reading sequentially will hit "independently-cached compile" first (presented as the *value proposition* of the pattern), then 25 lines later hit "re-compiles from scratch; no cache," and conclude one of the two is wrong. Cross-chapter, line 124 also re-contradicts `module_call_path.md:138`.

**Required fix:** In `ch4_qwen3_walkthrough/orchestrator_pattern.md:124`, change the trailing clause "every per-layer matmul / norm / RoPE / SDPA is a discrete, independently-cached compile" to "every per-layer matmul / norm / RoPE / SDPA compiles into its own graph and runs as its own `program.run()` cycle — a discrete, independently-compiled per-child unit (no compile-result caching today; see `:149` and Ch5 `module_call_path.md` for the `_compiled_cache` story)." This keeps the load-bearing "independent" / "discrete" framing the paragraph needs (the orchestrator buys host-hop separability, not caching), removes the false "cached" claim, and explicitly forward-links to the two existing sources of truth so a reader who lands on `:124` first does not leave with the wrong mental model.

---

## Agent A change log — applied after Pass 2 full-guide B review
- Pass 2 Issue 1: Rewrote `ch4_qwen3_walkthrough/orchestrator_pattern.md:124` trailing clause from "every per-layer matmul / norm / RoPE / SDPA is a discrete, independently-cached compile" to "every per-layer matmul / norm / RoPE / SDPA compiles into its own graph and runs as its own `program.run()` cycle — a discrete, independently-compiled per-child unit (no compile-result caching today; see `:149` and Ch5 `module_call_path.md` for the `_compiled_cache` story)." Now consistent with the Pass 1 corrections at `:149` and `:155`, the `:174` contributor callout, and Ch5 `module_call_path.md:138`.

---

## Pass 3

Pass 2 verification: the rewrite at `ch4_qwen3_walkthrough/orchestrator_pattern.md:124` is in place — the trailing clause now reads "every per-layer matmul / norm / RoPE / SDPA compiles into its own graph and runs as its own `program.run()` cycle — a discrete, independently-compiled per-child unit (no compile-result caching today; see `:149` and Ch5 `module_call_path.md` for the `_compiled_cache` story)." The four `_compiled_cache` anchors in Ch4 (`:124`, `:149`, `:155`, `:174`) now agree with Ch5 `module_call_path.md:138` ("Nothing in the framework reads or writes this dict today"). A sequential reader of Ch4 encounters "independently-compiled per-child unit (no compile-result caching today)" at `:124`, the explicit "re-compiles from scratch; there is no per-Module compile-result cache wired up today" at `:149`, the "no module — orchestrator or child — caches today" at `:155`, and the "currently unused but reserved" callout at `:174` — one coherent story, no contradictions.

No feedback — guide approved.
