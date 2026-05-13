# Compression Analysis: Chapter 1 — Why blaze-nn and how it fits together — Pass 1

## Summary
- Total files analyzed: 4 (3 content + 1 index)
- Estimated current line count: ~272 lines (7 + 71 + 67 + 127)
- Estimated post-compression line count: ~205 lines
- Estimated reduction: ~25%

## CRUCIAL Suggestions

### [ttnn_native_contract.md] ~lines 7-10
**Issue:** The `blaze_nn/__init__.py:5-7` docstring ("The framework is ttnn-native: parameters, inputs, and outputs are `ttnn.Tensor`. No torch tensors flow through the framework code...") is quoted verbatim here AND in `what_it_is.md:11`. The two files are adjacent in the chapter; readers see the same blockquote twice within a few minutes.
**Suggestion:** Quote the docstring once in `what_it_is.md`. In `ttnn_native_contract.md`, replace lines 7-10 with a short callback such as "Restating the package docstring quoted in [What blaze-nn is](what_it_is.md#the-framework-in-its-own-words):" followed by a single short paraphrase, or simply reference the prior quote without re-quoting. Saves ~4 lines and removes a duplicate blockquote.

### [ttnn_native_contract.md] ~lines 21-37
**Issue:** The 12-line `Parameter.__init__` / `data` property code block is followed by prose ("No isinstance check, no shape declaration, no dtype constraint. The annotation is literally `Any`.") that says exactly what the code says. The same fact — `_tensor: Any`, no inspection, `object()` sentinels in tests — is also stated in `what_it_is.md:49` ("Not an autograd engine ... see `blaze_nn/parameter.py:16-26`") and `getting_started.md:38` ("Because the framework treats parameter tensors as opaque `Any` (`blaze_nn/parameter.py:18`)").
**Suggestion:** Either (a) drop the code block and keep the prose summary plus the `blaze_nn/parameter.py:16-26` pin, or (b) shrink the code block to just the two annotated lines (`_tensor: Any = None` and `def data(self) -> Any:`). Saves ~10 lines without losing information; the full source is one click away via the pin.

### [getting_started.md] ~lines 96-121
**Issue:** The 25-line "What 'running a model' actually looks like" section is forward example bloat — the plan (`plan.md:74-79`) scopes this file to install, environment, and the three test tiers. The snippet itself acknowledges its features "are each a section in Chapters 2–4", and `index.md` line 3 already names the user-code → device pipeline. Repeating a runnable example here previews Ch2/Ch3 content twice.
**Suggestion:** Either remove the section entirely and end the file at "Where to go next" pointing to Chapter 2, or cut the code block to a 4-5 line skeleton (Module subclass + `forward` + `model(x)` call) without `load_state_dict` / `to(device)` / `Parameter` details, which all belong to Ch2/Ch3. Saves ~20 lines.

### [what_it_is.md] ~lines 13-23
**Issue:** The README five-bullet quote (lines 15-21) is long and overlaps the package docstring quoted directly above it (lines 9-11). The summary sentence at line 23 ("Both statements pick the same three load-bearing words: PyTorch-style, ttnn-native, tracing") explicitly admits the two quotes carry the same payload. Additionally, the "Universal op dispatch" bullet (line 20) is re-quoted verbatim in `ttnn_native_contract.md:55`.
**Suggestion:** Keep one of the two quotes (prefer the package docstring, which is the canonical source) plus a short prose mention of the five selling-point names; drop the verbatim five-bullet block. Saves ~7-8 lines and eliminates the third copy of the "universal op dispatch" sentence.

### [what_it_is.md] ~lines 27-37 vs lines 39-45
**Issue:** The Mermaid diagram (27-32) shows four boxes; the bulleted list immediately after (34-37) restates each box in prose with a one-line gloss; then lines 39-43 add a "Each handoff point becomes a chapter" map that re-walks the same four-box structure a third time; then line 45 is a contributor callout that previews Ch5 with yet another walk of the same chain.
**Suggestion:** Keep the Mermaid diagram + the four-bullet gloss (lines 34-37, which carry the pins). Cut lines 39-43 (the chapter map can move into `index.md` if needed, but `index.md` already names the chain). The "For contributors" callout at 45 can survive as a one-liner. Saves ~6-8 lines and prevents three sequential passes over the same four-box picture.

### [what_it_is.md] ~lines 65-67
**Issue:** The vs. table row labeled "this framework" already says "Never call this framework 'blaze'" in the prose at line 65 ("A model author works exclusively with the blaze-nn column..."), and the Conventions section in `plan.md` already establishes the naming rule. The `> **Note:** Never call this framework "blaze"` callout at line 67 restates this a third time.
**Suggestion:** Drop the Note at line 67 — the table caption + the line-65 prose already cover it, and the rule belongs to the guide-wide Conventions, not a per-chapter callout. Saves ~2 lines.

## MINOR Suggestions

### [index.md] ~line 3
**Issue:** The "By the end you will have one mental model ... one invariant ... and an installed checkout with the three test tiers green." clause is a re-statement of the chapter title and the three filenames in the list below. The Conventions section (`plan.md`) says `index.md` files contain "only the chapter title, a one-paragraph summary, and an ordered list of links."
**Suggestion:** Compress the paragraph to one sentence: "This chapter positions blaze-nn against PyTorch (API shape), tt-blaze (compiler), and ttnn (tensor library), and gets you to a green test suite." Saves ~1-2 lines.

### [ttnn_native_contract.md] ~lines 13-15
**Issue:** "This is why `import blaze_nn` works on a machine without tt-blaze installed — useful for unit-testing the framework itself. See [Getting started](getting_started.md) for the three test tiers..." duplicates the same Note in `getting_started.md:26`. Same fact, same forward/back link.
**Suggestion:** Keep this as a single-sentence forward link without the "useful for unit-testing the framework itself" gloss (covered in `getting_started.md` Tier A). Saves ~1 line.

### [getting_started.md] ~lines 32-46
**Issue:** Tier A prose at line 38 ("Because the framework treats parameter tensors as opaque `Any` (`blaze_nn/parameter.py:18`), the test does not need a real tensor type to verify routing.") re-explains the contract that `ttnn_native_contract.md` is dedicated to. Line 46 ("This is the tier you run on every commit.") is filler.
**Suggestion:** Drop line 46. Shorten line 38 to "Uses `object()` sentinels in place of `ttnn.Tensor` (see [ttnn-native contract](ttnn_native_contract.md))." Saves ~2 lines.

## Load-Bearing Evidence
- `index.md` line ~5-7: the ordered list of three content links — load-bearing because the index is the only navigation entry to the chapter's three files; cannot be removed.
- `what_it_is.md` line ~49-53: the "What blaze-nn is **not**" bullets (no autograd, not a kernel library, no eager execution, no implicit device placement) — load-bearing because they are the *only* place in the chapter that names these scope limits, and the plan (`plan.md:66`) lists them as required content.
- `ttnn_native_contract.md` line ~19-43: the four-position enumeration (Parameters / forward args / forward return / state_dict values) — load-bearing because it is the only spot in Ch1 where the invariant is anchored to the four specific code locations (`base.py:92-93`, `base.py:122`, `parameter.py:16-26`, state_dict).
- `getting_started.md` line ~76-92: the tt-blaze environment recipe (`env.sh`, `TT_METAL_HOME`, `PYTHONPATH`) and the `python -c "import blaze, ttnn"` sanity check — load-bearing because this is the only chapter that owns environment setup, and Tiers B/C cannot run without it.

## VERDICT
- Crucial updates: yes

---

## Agent A change log — applied after Pass 1 compression analysis
- `what_it_is.md`: replaced the 9-line README five-bullet blockquote with a single-sentence prose mention of the five selling points, eliminating the duplicate "universal op dispatch" quote.
- `what_it_is.md`: cut the three-bullet "chapter map" after the Mermaid diagram and shrank the contributor callout to one line (no third pass over the four-box picture).
- `what_it_is.md`: removed the `> **Note:** Never call this framework "blaze"` callout (already covered by the table + line-49 prose + guide-wide Conventions).
- `ttnn_native_contract.md`: replaced the verbatim re-quote of the `blaze_nn/__init__.py:5-7` docstring with a one-sentence callback to `what_it_is.md`'s prior quote.
- `ttnn_native_contract.md`: collapsed the 12-line `Parameter.__init__` / `data` code block into a one-sentence prose summary plus the `blaze_nn/parameter.py:16-26` pin.
- `getting_started.md`: removed the entire "What 'running a model' actually looks like" section (Module subclass + load_state_dict + to(device) + call), which previewed Ch2/Ch3 content out of scope for this page.

---

# Compression Analysis: Chapter 1 — Why blaze-nn and how it fits together — Pass 2

## Summary
- Total files analyzed: 4 (3 content + 1 index)
- Estimated current line count: ~209 lines (7 + 53 + 49 + 100)
- Estimated post-compression line count: ~205 lines
- Estimated reduction: ~2%

## MINOR Suggestions

### [what_it_is.md] ~line 37
**Issue:** The "No implicit device placement" bullet ends with a parenthetical "(Restated as a `> **Warning:**` in Chapter 2's `device_binding.md`.)" — and `getting_started.md` line 94 carries the same warning a second time inside this very chapter, with another forward link to the same Ch2 file. Two forward links to the same Ch2 page within Ch1 is slightly redundant.
**Suggestion:** Drop the parenthetical at end of line 37 (the Ch2 forward link survives in `getting_started.md:94`). Saves ~1 line and removes one of the two "see Ch2 device_binding" pointers in Ch1.

### [getting_started.md] ~line 26
**Issue:** The "`import blaze_nn` is safe on a machine without tt-blaze installed ... `BlazeCompiler` is imported inside `_call_graph` at `blaze_nn/modules/base.py:98`, not at package load" note re-states the mechanical consequence already covered in `ttnn_native_contract.md:11-12` (which is the page that defines the no-torch-at-module-scope contract and even cites the same `base.py:98` pin). Tier A's framework-only premise can stand on its own.
**Suggestion:** Shorten the Note to one sentence cross-linking back: "`import blaze_nn` is safe without tt-blaze installed — see [ttnn-native contract](ttnn_native_contract.md#blaze-nn-never-imports-torch-at-module-scope) for why; this is what Tier A relies on." Saves ~2 lines and removes the duplicate `base.py:98` pin.

### [ttnn_native_contract.md] ~line 27
**Issue:** The sentence "The only `.shape` access in the framework's non-test code is the `Parameter.__repr__` heuristic at `blaze_nn/parameter.py:28-34`, which is used purely for debug printing and is guarded against `None`." is a precise but tangential aside — the surrounding paragraph's load-bearing claim is "blaze-nn does not inspect tensor contents." The `__repr__` exception belongs in Ch2 `parameter.md` (which the plan at line 89 already scopes to `__repr__` heuristics).
**Suggestion:** Either drop the sentence or shrink to "(One narrow exception lives in `Parameter.__repr__`; see Ch2 `parameter.md`.)" Saves ~1 line and moves a detail to its rightful chapter.

## Load-Bearing Evidence
- `index.md` line ~3: "By the end you will have one mental model (user code → blaze-nn tracing → tt-blaze graph → tt-metal kernels), one invariant (every tensor crossing a `Module` boundary is a `ttnn.Tensor`), and an installed checkout with the three test tiers green." — load-bearing because it is the only place the chapter's three deliverables (mental model, invariant, green tests) are enumerated together, mapping one-to-one to the three content files.
- `what_it_is.md` line ~33-37: the "What blaze-nn is **not**" bullets (no autograd, not a kernel library, not a torch-compatible tensor type, no eager execution, no implicit device placement) — load-bearing because they are the only place in the chapter that names these five scope limits, each with its own source-of-truth pin; the plan (`plan.md:66`) lists these as required content.
- `ttnn_native_contract.md` line ~17-25: the four-position enumeration (Parameters / forward args / forward return / state_dict values) — load-bearing because it is the only place in Ch1 where the invariant is anchored to the four specific code locations (`parameter.py:16-26`, `base.py:92-93`, `base.py:122`, state_dict); every later chapter references this enumeration.
- `getting_started.md` line ~72-92: the tt-blaze environment recipe (`env.sh`, `TT_METAL_HOME`, `PYTHONPATH`) and the `python -c "import blaze, ttnn"` sanity check — load-bearing because this is the only chapter that owns environment setup, and Tiers B/C cannot run without it; `README.md:30-44` is the cited source.

## VERDICT
- Crucial updates: no

