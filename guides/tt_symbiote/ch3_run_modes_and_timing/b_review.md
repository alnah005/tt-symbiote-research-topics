# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 1

## Issues

**1. `dispatch_manager.md`, line ~41 — `set_current_module_name` push happens after `preprocess_weights`, not before**

The guide's code sample showed `set_current_module_name(self.module_name)` (push) as the first operation in `module_run`, suggesting that `preprocess_weights()` runs with the module name already on the stack. The source (`run_config.py` lines 566–569) shows the opposite: `preprocess_weights()` is called and timed first, and the stack push happens only after. A developer reading the original guide would incorrectly conclude that ATen ops dispatched inside `preprocess_weights` are attributed to the current module; in reality they see the parent module's name (or `None`) as `current_module_name`.

Fix applied: rewrote the code example in `dispatch_manager.md` to show the actual execution order with inline comments.

---

**2. `dispatch_manager.md`, line ~19 — "head of the stack" incorrectly names the innermost module**

The guide said "The **head** of the stack is the innermost active module." In standard usage and in Python list semantics, the head (index 0) is the first-appended (outermost) element; the top (index `-1`, last element) is the most-recently-appended (innermost) element. Calling the innermost element "the head" contradicts both the English convention and how `_modules_in_progress[-1]` is used in the source. A reader implementing their own module stack inspection would read the wrong end of the list.

Fix applied: replaced "head" with "top (last element, `[-1]`)" in the `_modules_in_progress` table row.

---

**3. `run_modes.md`, line ~20 — warning about env var override incorrectly attributed to `set_run_mode`**

The guide said: "When the environment variable is also present it overrides the programmatic value, and a warning is printed." This implies the warning is emitted during the `set_run_mode()` call. The source shows `set_run_mode` (lines 1117–1125) only asserts idempotency and sets `_current_run_mode`; it never reads `TT_SYMBIOTE_RUN_MODE`. The override warning is actually printed inside `get_tensor_run_implementation()` (lines 1144–1147), which is called lazily each time a `TorchTTNNTensor` is constructed. A developer who calls `set_run_mode("NORMAL")` and also has `TT_SYMBIOTE_RUN_MODE=DPL` set would not see any warning at call-time and might believe the env var was silently ignored until operations start.

Fix applied: clarified in `run_modes.md` that the warning is printed by `get_tensor_run_implementation()`, not by `set_run_mode`.

---

## Change Log — Pass 1 fixes applied

- Fix 1: `dispatch_manager.md` — replaced misleading push-first code sample with accurate execution-order pseudocode showing `preprocess_weights` runs before the module-name stack push, with a note on attribution scope.
- Fix 2: `dispatch_manager.md` — changed "head of the stack" to "top (last element, `[-1]`)" in the `_modules_in_progress` attribute description.
- Fix 3: `run_modes.md` — corrected the description of where the env-var override warning is printed: `get_tensor_run_implementation()`, not `set_run_mode`.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 2

## Re-check of Pass 1 fixes

All three Pass 1 fixes are correctly applied in the current files:

1. **Push-order fix** (`dispatch_manager.md` lines 43–48): The pseudocode correctly shows `preprocess_weights()` executing and being timed before `set_current_module_name(self.module_name)` is called. Confirmed against source lines 566–569. ✓

2. **"head" → "top" fix** (`dispatch_manager.md` line 19): The `_modules_in_progress` table row now reads "top (last element, `[-1]`)", consistent with `_modules_in_progress[-1]` in the source. ✓

3. **Warning-location fix** (`run_modes.md` lines 20–22): The text now correctly states the env-var override warning is printed by `get_tensor_run_implementation()`, not by `set_run_mode`, and explains the lazy evaluation model. ✓

---

## New issues found

### Issue 1 — `dispatch_manager.md`, module-name stack section — SEL/DPL modes silently bypass the stack

**File:** `dispatch_manager.md`, the `set_current_module_name` section (lines 41–53)

**Problem:** The guide says "`NormalRun.module_run` (and `TracedRun.module_run`) bracket the forward pass with push/pop" — implying other run modes do the same or at minimum do not contradict it. In fact `SELRun.module_run`, `DPLRun.module_run`, and `DPLRunNoErrorProp.module_run` never call `set_current_module_name` at all (confirmed: none of the three methods in source lines 665–795 contain a `set_current_module_name` call). ATen ops dispatched during `self.forward(...)` in SEL/DPL modes will be attributed to whatever module name was already on the stack from a parent call — or to `""` at the top level. A developer relying on timing CSV attribution for SEL or DPL runs would get entirely wrong module-name fields for all ATen-level entries.

**Correct statement:** Only `NormalRun.module_run` and `TracedRun.module_run` (traced path) push/pop the module-name stack. SEL, DPL, and DPL_NO_ERROR_PROP module_run methods do not call `set_current_module_name`.

**Fix applied:** Added a callout block immediately before the pseudocode in the `set_current_module_name` section of `dispatch_manager.md` warning that SEL/DPL modes do not use the stack in their `module_run` implementations.

---

### Issue 2 — `dispatch_manager.md`, line ~131 — `TracedRun.module_run`'s non-trace-enabled path does not record a `TorchModules` entry

**File:** `dispatch_manager.md`, "Additional timing recorded by `NormalRun.module_run`" (line 131)

**Problem:** The guide parenthetically claims "`NormalRun.module_run` (and `TracedRun.module_run`) records three additional timing entries per module call" and lists `TorchModules` as one of the four. This is wrong for `TracedRun`. When a module is not trace-enabled (`not is_trace_enabled(self)`) or a trace is already running (`_TRACE_RUNNING`), `TracedRun.module_run` takes an early-return path (source lines 1047–1053) that records `_preprocess_weights`, `_move_weights_to_device`, and `_forward` entries but then returns immediately — `end_full` and the `TorchModules` entry are never recorded. Only the traced execution path (lines 1058–1086) reaches the `TorchModules` record call. A developer filtering timing output on `backend == "TorchModules"` would find entries missing for non-trace-enabled modules in `TRACED` mode and incorrectly conclude those modules had zero wall-clock cost.

**Correct statement:** `TracedRun.module_run` records `TorchModules` only when the module actually executes a trace (cache hit or cache miss path). Non-trace-enabled modules under `TRACED` mode do not produce a `TorchModules` entry.

**Fix applied:** Replaced the parenthetical in `dispatch_manager.md` with an accurate description specifying which path of `TracedRun.module_run` records `TorchModules`, and noting that SEL/DPL modes record no module-level timing entries at all.

---

### Issue 3 — `dispatch_manager.md`, Step 5 — interactive analysis example groups by `func_name`, which gives per-class totals, not per-instance totals

**File:** `dispatch_manager.md`, "Step 5: use `get_timing_entries_stats()` for interactive analysis" (lines 239–250)

**Problem:** The example code performs `.groupby("func_name")["duration"].sum()` filtered to `backend == "TorchModules"`. For `TorchModules` entries, `func_name` = the class name (e.g., `TransformerBlock`), while `module_name` = the instance path (e.g., `model.transformer.h.0`). A model with 24 transformer layers produces 24 `TorchModules` entries all sharing the same `func_name = "TransformerBlock"`. Grouping by `func_name` sums all 24 instances' durations into a single line — giving total class cost across all instances, not the cost of any individual slow instance. A developer using this example to diagnose which specific layer is the bottleneck would get a class-level aggregate that is useless for per-instance diagnosis.

**Correct statement:** To find the slowest individual module instance, group by `module_name`; to find the costliest class type across all instances, group by `func_name`.

**Fix applied:** Changed `.groupby("func_name")` to `.groupby("module_name")` in the Step 5 example, and added an explanatory comment clarifying the distinction between `func_name` (class name) and `module_name` (instance path) for `TorchModules` entries.

---

## Verdict

Three material issues were found and fixed. Two are architectural (SEL/DPL stack bypass, TracedRun TorchModules partial recording) that would cause developers to misread timing attribution data or search for entries that don't exist. One is an incorrect code example that produces per-class aggregation when per-instance diagnosis is intended. No further issues found; chapter is approved pending these three fixes.

---

## Change Log — Pass 2 fixes applied

- Fix 1: `dispatch_manager.md` — added callout to the `set_current_module_name` section explicitly stating that `SELRun`, `DPLRun`, and `DPLRunNoErrorProp` module_run methods do not push/pop the module-name stack, so ATen op attribution is unreliable for those modes.
- Fix 2: `dispatch_manager.md` — replaced the parenthetical "(and `TracedRun.module_run`)" with a precise description: `TracedRun` records all four timing entries only on the traced execution path; non-trace-enabled modules return early without a `TorchModules` entry; SEL/DPL write no module-level timing entries at all.
- Fix 3: `dispatch_manager.md` — changed the Step 5 interactive analysis example from `.groupby("func_name")` to `.groupby("module_name")` and added a comment explaining the `func_name` vs `module_name` distinction for `TorchModules` entries.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 3

## Re-check of Pass 2 fixes

All three Pass 2 fixes are correctly applied in the current files:

1. **SEL/DPL stack callout** (`dispatch_manager.md` lines 43–43): The `> **Important:**` callout explicitly names `SELRun`, `DPLRun`, and `DPLRunNoErrorProp` as non-stack-pushing. Confirmed against source (`SELRun.module_run` lines 664–685, `DPLRun.module_run` lines 706–735, `DPLRunNoErrorProp.module_run` lines 761–795 — none call `set_current_module_name`). ✓

2. **TracedRun TorchModules partial recording** (`dispatch_manager.md` line 133): The text now correctly distinguishes the non-trace-enabled early-return path (no `TorchModules` entry) from the traced execution path (records `TorchModules`). Confirmed against source lines 1046–1086. ✓

3. **groupby `module_name`** (`dispatch_manager.md` lines 247–254): Step 5 example now groups by `"module_name"` with explanatory comment. ✓

---

## New issues found

### Issue 1 — `dispatch_manager.md`, step 5 description of `save_stats_to_file` top-30 output — prints class names, not instance paths

**File:** `dispatch_manager.md`, line ~196 (step 5 of `save_stats_to_file`).

**Problem:** The guide said "prints the top 30 **module names** sorted by total duration in descending order." This implies the printed list is indexed by instance paths (e.g. `model.transformer.h.3`). The source (lines 304–312) builds `func_times` pivoted by `index=["func_name"]` and reads the `TorchModules` column from that table — so the printed index is `func_name` (class name, e.g. `TransformerBlock`), not `module_name`. A developer calling `save_stats_to_file()` to identify a slow specific layer instance would expect instance paths in the console output but get class totals.

**Correct statement:** `save_stats_to_file()` prints the top 30 **class names** (`func_name`) by total `TorchModules` duration; to find slow individual instances, use `get_timing_entries_stats()` grouped by `module_name`.

**Fix applied:** Replaced "prints the top 30 module names" with "prints the top 30 **class names** (`func_name`)" and added a note directing users to Step 5 for per-instance analysis.

---

### Issue 2 — `dispatch_manager.md`, Step 3 — "prefix before the first `.`" is wrong as a parsing rule for module name

**File:** `dispatch_manager.md`, line ~231 (Step 3: interpret the module name format).

**Problem:** The guide said "The prefix before the first `.` is the module name passed to `DispatchManager.set_current_module_name`." Module names contain dots (e.g. `model.transformer.h.0`). Splitting on the first `.` in `model.transformer.h.0.attn.TTNN::mm.default` yields `model`, not the full module name `model.transformer.h.0.attn`. A developer writing code to extract the contributing module from a CSV `module_name` value would parse the wrong string.

**Correct statement:** The module name is everything before the final `.<backend_prefix>::<op>` suffix. Since module names may contain dots, the correct parse is to split on the last occurrence of `.TTNN::` or `.aten::`.

**Fix applied:** Replaced the "prefix before the first `.`" sentence with an accurate description: the module name is everything before the final `.<backend_prefix>::<op>` suffix, and splitting on the first `.` is incorrect because module names themselves contain dots.

---

### Issue 3 — `dispatch_manager.md`, timing table omits the `_capture_trace` entry produced on `TracedRun` cache miss

**File:** `dispatch_manager.md`, "Additional timing recorded by `NormalRun.module_run`" (timing table, lines ~135–142).

**Problem:** The guide's timing table listed only four `func_name` entries per module call and said `TracedRun` records "the same four entries only on the traced execution path." The source (lines 1074–1076) shows that on a cache miss, `TracedRun.module_run` records a fifth entry: `{ClassName}_capture_trace` under `"TTNN"` backend, measuring the full trace capture duration (warm-up + begin/end capture). A developer querying the flat CSV for `func_name == "{ClassName}_forward"` to sum per-call forward times would get a correctly sized result, but a developer trying to account for all TTNN time from a cache-miss run would have an unexplained gap unless they know about `_capture_trace`.

**Correct statement:** On a `TracedRun` cache miss, a fifth timing entry `{ClassName}_capture_trace` / `"TTNN"` is also recorded, covering the full `_capture_trace` call (one warm-up forward plus trace capture overhead). This entry does not appear on cache hits.

**Fix applied:** Expanded the timing table to add the `{ClassName}_capture_trace` row with a "TracedRun cache-miss path only" annotation, and added a note reminding developers to filter this entry out when comparing per-call forward times across runs.

---

## Verdict

Three material issues were found and fixed. Issue 1 would cause a developer to misread `save_stats_to_file()` console output, interpreting class-level aggregates as instance-level data. Issue 2 would cause incorrect CSV parsing if anyone implements a module-name extractor from the `module_name` column. Issue 3 would leave an unaccounted timing entry in the flat CSV on cache-miss runs, causing confusion when reconciling total TTNN time. No further material issues found; chapter is approved pending these three fixes.

---

## Change Log — Pass 3 fixes applied

- Fix 1: `dispatch_manager.md` — corrected `save_stats_to_file` step 5 description: changed "top 30 module names" to "top 30 class names (`func_name`)" and added a note pointing to Step 5 for per-instance analysis.
- Fix 2: `dispatch_manager.md` — corrected Step 3 module-name parsing rule: replaced "prefix before the first `.`" with an accurate description that module names contain dots and must be extracted by splitting on the last `.TTNN::` or `.aten::` occurrence.
- Fix 3: `dispatch_manager.md` — added `{ClassName}_capture_trace` / `"TTNN"` row to the timing table, annotated as "TracedRun cache-miss path only", and added a note about filtering it when comparing per-call forward times.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 4

## Re-check of Pass 3 fixes

All three Pass 3 fixes are correctly applied in the current files:

1. **`save_stats_to_file` console output** (`dispatch_manager.md` line 197): Text now reads "top 30 **class names** (`func_name`)" with a note pointing to `get_timing_entries_stats()` for per-instance analysis. Confirmed against source lines 304–312 where the pivot is indexed by `func_name` only. ✓

2. **Module-name parsing rule** (`dispatch_manager.md` line 232): Text correctly explains that module names contain dots and the correct parse splits on the last `.TTNN::` or `.aten::` occurrence. ✓

3. **`_capture_trace` timing entry** (`dispatch_manager.md` timing table): The `{ClassName}_capture_trace` row is present with "TracedRun cache-miss path only" annotation. Confirmed against source lines 1071–1076. ✓

---

## New issues found

### Issue 1 — `dispatch_manager.md`, timing table — "All paths" incorrectly implies `NormalRunWithFallback` records module-level timing entries

**File:** `dispatch_manager.md`, timing table (previously `Paths recorded` column said "All paths").

**Problem:** The timing table's `Paths recorded` column previously read "All paths" for the three TTNN module-level entries (`_preprocess_weights`, `_move_weights_to_device`, `_forward`). `NormalRunWithFallback.module_run` (source lines 619–642) overrides `NormalRun.module_run` completely and contains zero `record_timing` calls, zero `set_current_module_name` calls, and no `TorchModules` entry. A developer using `NORMAL_WITH_FALLBACK` mode and then querying the flat CSV for `_preprocess_weights` or `TorchModules` entries would find nothing and incorrectly conclude a bug in their setup rather than understanding the mode's design.

**Correct statement:** The three TTNN module-level entries and the `TorchModules` entry are only recorded by `NormalRun.module_run` and `TracedRun.module_run`. `NormalRunWithFallback.module_run` writes no module-level timing entries whatsoever.

**Fix applied:** Added a callout block before the timing table explicitly noting that `NormalRunWithFallback.module_run` records no module-level timing entries. Updated the `Paths recorded` column to name specific classes instead of "All paths".

---

### Issue 2 — `dispatch_manager.md`, dispatch wrappers section — undocumented `can_dispatch_to_ttnn` timing entry inflates `"TTNN"` backend totals

**File:** `dispatch_manager.md`, dispatch wrappers section (after the `dispatch_to_torch_wrapper` table).

**Problem:** The guide documented only two categories of ATen-level timing entries: `"TTNN"` backend entries from `dispatch_to_ttnn_wrapper` and `"Torch"` backend entries from `dispatch_to_torch_wrapper`. It omitted the third entry: `NormalRun.torch_dispatch` (source lines 487–500) records a timing entry with `backend = "TTNN"` and `func_name = "can_dispatch_to_ttnn"` for **every single ATen op** dispatched in `NORMAL` or `NORMAL_WITH_FALLBACK` mode. A developer summing `df[df["backend"] == "TTNN"]["duration"].sum()` to estimate total TTNN kernel time in NORMAL mode would inadvertently include the overhead of all `can_dispatch_to_ttnn` checks, producing an inflated number. This entry is absent in `SELRun`, `DPLRun`, `DPLRunNoErrorProp`, `LightweightRun`, `CPU`, and `TracedRun`.

**Correct statement:** Under `NORMAL` and `NORMAL_WITH_FALLBACK` modes, every ATen-level dispatch also produces a `"TTNN"` / `"can_dispatch_to_ttnn"` timing entry. Filter `func_name != "can_dispatch_to_ttnn"` when summing TTNN kernel time.

**Fix applied:** Added a new subsection "Hidden per-op overhead entry: `can_dispatch_to_ttnn`" immediately before the "Additional timing recorded by `NormalRun.module_run`" section, documenting the entry's fields, which modes produce it, and how to filter it.

---

## Verdict

Two material issues were found and fixed. Issue 1 is architectural: a developer using `NORMAL_WITH_FALLBACK` mode would expect module-level timing data that never materializes. Issue 2 is numerical: summing `TTNN` backend duration without knowing about `can_dispatch_to_ttnn` entries would give inflated per-run TTNN totals under `NORMAL` mode. No further material issues found; chapter is approved pending these two fixes.

---

## Change Log — Pass 4 fixes applied

- Fix 1: `dispatch_manager.md` — added callout before the timing table stating that `NormalRunWithFallback.module_run` records no module-level timing entries; updated `Paths recorded` column from "All paths" to explicitly name `NormalRun` and `TracedRun`, and noted `NormalRunWithFallback` is excluded for the `TorchModules` row.
- Fix 2: `dispatch_manager.md` — added "Hidden per-op overhead entry: `can_dispatch_to_ttnn`" subsection documenting the additional `"TTNN"` backend entry recorded per ATen op by `NormalRun.torch_dispatch` and `NormalRunWithFallback.torch_dispatch`, with a filter note for developers summing TTNN kernel time.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 5

## Re-check of Pass 4 fixes

**Fix 1 — `NormalRunWithFallback` records no module-level timing entries:** Correctly applied. The callout before the timing table names `NormalRunWithFallback` explicitly and the `Paths recorded` column now lists `NormalRun` and `TracedRun` rather than "All paths". Confirmed against source lines 619–643, which contain zero `record_timing` calls. ✓

**Fix 2 — `can_dispatch_to_ttnn` hidden entry subsection:** The subsection was added and is structurally correct about the entry's fields and its absence from `SELRun`, `DPLRun`, etc. However, the subsection incorrectly states the entry is recorded by both `NormalRun.torch_dispatch` **and** `NormalRunWithFallback.torch_dispatch`. This is wrong — see Issue 1 below. The fix was partially correct but introduced a factual error in the scope clause.

---

## New issues found

### Issue 1 — `dispatch_manager.md`, `can_dispatch_to_ttnn` subsection — `NormalRunWithFallback` incorrectly listed as recording this entry

**File:** `dispatch_manager.md`, "Hidden per-op overhead entry: `can_dispatch_to_ttnn`" subsection (line ~133).

**Problem:** The subsection reads "`NormalRun.torch_dispatch` (and `NormalRunWithFallback.torch_dispatch`) records an additional timing entry…" and the closing sentence says "filter out rows where `func_name == 'can_dispatch_to_ttnn'`… under `NORMAL` or `NORMAL_WITH_FALLBACK`." Both claims are wrong. `NormalRunWithFallback` overrides `torch_dispatch` entirely (source lines 603–617). Its implementation calls `can_dispatch_to_ttnn` inline inside a bare `if` with no surrounding `time.time()` or `record_timing` call — there is no `can_dispatch_to_ttnn` timing entry produced under `NORMAL_WITH_FALLBACK`. A developer running `NORMAL_WITH_FALLBACK` and filtering `func_name != "can_dispatch_to_ttnn"` would be filtering rows that don't exist; more critically, a developer expecting to find these entries and not finding them would incorrectly diagnose a bug or under-count their `TTNN` totals.

**Correct statement:** Only `NormalRun.torch_dispatch` records the `can_dispatch_to_ttnn` entry. `NormalRunWithFallback.torch_dispatch` does not record it. The filter `func_name != "can_dispatch_to_ttnn"` is relevant only for `NORMAL` mode.

**Fix applied:** Replaced the opening sentence to name only `NormalRun.torch_dispatch`, added an explicit note that `NormalRunWithFallback` overrides `torch_dispatch` without timing instrumentation, and corrected the closing sentence to reference `NORMAL` mode only.

---

## Verdict

One material issue found and fixed. It is numerical: the Pass 4 subsection incorrectly listed `NormalRunWithFallback` as a producer of `can_dispatch_to_ttnn` timing entries. A developer summing or filtering TTNN entries under `NORMAL_WITH_FALLBACK` would search for entries that do not exist and incorrectly adjust their analysis. No further material issues found; chapter is approved pending this fix.

---

## Change Log — Pass 5 fixes applied

- Fix 1: `dispatch_manager.md` — corrected the "Hidden per-op overhead entry: `can_dispatch_to_ttnn`" subsection: removed `NormalRunWithFallback` from the list of modes that record this entry, added an explicit note that `NormalRunWithFallback.torch_dispatch` overrides the method without timing instrumentation, and corrected the filter guidance to reference `NORMAL` mode only (not `NORMAL_WITH_FALLBACK`).

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 6

## Re-check of Pass 5 fix

The Pass 5 fix is correctly applied. The "Hidden per-op overhead entry: `can_dispatch_to_ttnn`" subsection in `dispatch_manager.md` (lines 131–142) now names only `NormalRun.torch_dispatch` as the producer of this entry, explicitly notes that `NormalRunWithFallback` overrides `torch_dispatch` without timing instrumentation, and restricts the filter guidance to `NORMAL` mode only. Confirmed against source lines 603–617 (`NormalRunWithFallback.torch_dispatch`, no `record_timing` call) and lines 484–501 (`NormalRun.torch_dispatch`, records `can_dispatch_to_ttnn` entry). The Pass 5 fix is fully correct.

---

## New issues found

### Issue 1 — `dispatch_manager.md`, line 41 — `TracedRun.module_run` pushes/pops the module-name stack on ALL paths, not only the traced execution path

**File:** `dispatch_manager.md`, `set_current_module_name` section (line 41).

**Problem:** The guide stated "`NormalRun.module_run` (and `TracedRun.module_run` on its traced path) bracket the forward pass with push/pop." The qualifier "on its traced path" implies that on the non-trace-enabled early-return path (when `is_trace_enabled(self)` is False or `_TRACE_RUNNING` is already True), `TracedRun.module_run` does NOT push/pop the module-name stack. This is incorrect. In the source (lines 1073 and 1109), `set_current_module_name(self.module_name)` is called at line 1073, unconditionally before the `if not is_trace_enabled(self) or already_running:` check at line 1094. The pop at line 1109 is inside the early-return block. So ATen ops dispatched during `self.forward()` on the non-trace-enabled early-return path ARE attributed to this module's name — not the parent's. A developer who believed the push did not happen on the non-trace-enabled path would incorrectly expect those ATen ops to carry the parent module's `current_module_name` in the flat CSV, and build attribution filters accordingly.

**Correct statement:** Both `NormalRun.module_run` and `TracedRun.module_run` push/pop the module-name stack on all execution paths. In `TracedRun`, the push is unconditional (before the `is_trace_enabled` check), so the current module's name is active during `self.forward()` regardless of whether trace capture is used.

**Fix applied:** Replaced the phrase "on its traced path" with a precise statement that `TracedRun.module_run` pushes unconditionally before the `is_trace_enabled` check, so ATen ops on the non-trace-enabled path are attributed to this module's name.

---

### Issue 2 — `run_modes.md`, line 206 — `is_trace_enabled` uses `isinstance` (inheritance-aware), not exact class membership

**File:** `run_modes.md`, step 3 of `TracedRun.module_run` description (line 206).

**Problem:** The guide stated "`is_trace_enabled(self)` — returns `True` if the module's class is in `_TRACE_ENABLED_CLASSES` and not in `_TRACE_DISABLED_CLASSES`." The phrase "class is in `_TRACE_ENABLED_CLASSES`" implies direct set membership (i.e., `type(module) in _TRACE_ENABLED_CLASSES`). The source (line 900) uses `isinstance(module, _TRACE_ENABLED_TUPLE)`, which returns `True` for any instance of a class OR any of its subclasses. A developer who marks `Parent` with `@trace_enabled` would incorrectly believe that `Child(Parent)` (without `@trace_enabled`) is NOT trace-enabled and would add redundant decorators. More critically, a developer who marks `Child` with `@trace_enabled` expecting only `Child` instances (not `GrandChild(Child)` instances) to be traced would find `GrandChild` also traced — causing unexpected trace captures and cache growth.

**Correct statement:** `is_trace_enabled` uses `isinstance`, so trace-enablement is inherited by all subclasses. If `Parent` is `@trace_enabled`, then instances of `Child(Parent)` are also trace-enabled unless `@trace_disabled` is applied to `Child`.

**Fix applied:** Replaced the "class is in `_TRACE_ENABLED_CLASSES`" wording with an explicit statement that `isinstance` is used, and clarified that trace-enablement is inherited by subclasses.

---

### Issue 3 — `run_modes.md`, line 210 — `execute_trace` `blocking` flag incorrectly hardcoded as `False`

**File:** `run_modes.md`, step 7 of `TracedRun.module_run` description (line 210).

**Problem:** The guide stated the cache-hit path replays with `ttnn.execute_trace(..., blocking=False)`. The source (line 1124) calls `ttnn.execute_trace(entry.device, entry.trace_id, cq_id=TracedRun._cq_id, blocking=TracedRun._blocking)`. `TracedRun._blocking` defaults to `True` (source line 913) and is set via `configure(blocking=...)` (source line 928). The guide's hardcoded `blocking=False` is wrong in two ways: it misrepresents the default behavior (blocking by default), and it obscures that the flag is configurable. A developer reading the guide would not synchronize the device after a cache-hit replay, potentially consuming stale trace output from a previous execution.

**Correct statement:** `execute_trace` uses `TracedRun._blocking`, which defaults to `True`. Set `blocking=False` only by passing `blocking=False` to `TracedRun.configure()`, and if doing so, add an explicit `ttnn.synchronize_device` call before reading trace outputs.

**Fix applied:** Replaced `blocking=False` with `blocking=TracedRun._blocking` in the step 7 description, added a note that the default is `True` (controlled by `TracedRun.configure()`), and warned against assuming non-blocking execution.

---

## Verdict

Three material issues found and fixed. Issue 1 is architectural: the incorrect "traced path only" qualifier for `TracedRun`'s push/pop would cause developers to misread ATen op attribution for non-trace-enabled modules in TRACED mode. Issue 2 is an implementation error: using direct set-membership semantics instead of `isinstance` would cause developers to misunderstand which modules are trace-enabled, either missing expected traces or being surprised by inherited ones. Issue 3 is behavioral/numerical: hardcoding `blocking=False` would lead a developer to not synchronize after replaying a trace, potentially reading stale output. No further issues found; chapter is approved pending these three fixes.

---

## Change Log — Pass 6 fixes applied

- Fix 1: `dispatch_manager.md` — corrected line 41: replaced "on its traced path" qualifier with a precise statement that `TracedRun.module_run` calls `set_current_module_name(self.module_name)` unconditionally before the `is_trace_enabled` check, so the push/pop bracket the forward pass on all paths (traced, non-trace-enabled, and nested guard).
- Fix 2: `run_modes.md` — corrected step 3 of `TracedRun.module_run`: replaced "module's class is in `_TRACE_ENABLED_CLASSES`" with an explicit description that `isinstance` is used, making trace-enablement inherited by subclasses, with `@trace_disabled` required to opt out.
- Fix 3: `run_modes.md` — corrected step 7 of `TracedRun.module_run`: replaced hardcoded `blocking=False` with `blocking=TracedRun._blocking`, noted the default is `True` (set via `configure()`), and warned against assuming non-blocking replay.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 7

## Re-check of Pass 6 fixes

All three Pass 6 fixes are correctly applied in the current files:

1. **Unconditional push fix** (`dispatch_manager.md` line 41): The text now states that `TracedRun.module_run` calls `set_current_module_name(self.module_name)` unconditionally before the `is_trace_enabled` check, covering all execution paths. Confirmed against source lines 1073 and 1094. ✓

2. **`isinstance` fix** (`run_modes.md` line 206): Step 3 now explicitly states that `isinstance` is used (not `type()`), making trace-enablement inherited by subclasses, with `@trace_disabled` required to opt out. Confirmed against source line 900. ✓

3. **`blocking` flag fix** (`run_modes.md` line 210): Step 7 now reads `blocking=TracedRun._blocking` with a note that the default is `True` and a warning against assuming non-blocking replay. Confirmed against source lines 913 and 1124. ✓

---

## New issues found

### Issue 1 — `run_modes.md`, step 4 of `TracedRun.module_run` — `_TRACE_RUNNING` is set to `True` before the early-return path executes `forward()`, silently suppressing trace capture in nested trace-enabled submodules

**File:** `run_modes.md`, step 4 of `TracedRun.module_run` (line 207, before fix).

**Problem:** The guide said "If not trace-enabled, or if `_TRACE_RUNNING` is already `True` (nested trace guard), falls back to `self.forward(...)` normally." This implies the early-return path simply calls `forward()` without affecting the `_TRACE_RUNNING` flag. The source (lines 1089–1112) shows otherwise: before the `is_trace_enabled` check, the code acquires `_TRACE_RUNNING_LOCK` and unconditionally sets `_TRACE_RUNNING = True` if it was `False`. On the non-trace-enabled early-return path, `forward()` is called with `_TRACE_RUNNING = True`, and only after the return does the code reset it to `False`. Any nested `module_run` calls that occur during `self.forward()` will read `already_running = True` and also take their early-return path — regardless of whether they are `@trace_enabled`. A developer who decorates inner submodules with `@trace_enabled` but leaves the outer wrapper module undecorated would expect the inner modules to be traced; they will never be, because the outer non-trace-enabled `module_run` holds `_TRACE_RUNNING = True` for the entire duration of its `forward()` call.

**Correct statement:** Even on the non-trace-enabled early-return path, `TracedRun.module_run` sets `_TRACE_RUNNING = True` before calling `self.forward()`. All nested `module_run` calls during that forward pass will see `already_running = True` and take the early-return path, regardless of their own `@trace_enabled` status. To have inner modules captured, the outermost module in the call chain must also be `@trace_enabled`.

**Fix applied:** Replaced step 4 with a precise description of the `_TRACE_RUNNING` flag lifecycle, including the lock acquisition before the `is_trace_enabled` check, and added an explicit warning that non-trace-enabled wrapper modules block trace capture in all nested submodules.

---

### Issue 2 — `dispatch_manager.md`, timing table note — `_forward` duration on cache-miss runs is not comparable to cache-hit runs, even after filtering `_capture_trace`

**File:** `dispatch_manager.md`, timing note after the timing table (line 158, before fix).

**Problem:** The guide said "On a `TracedRun` cache miss, an additional `{ClassName}_capture_trace` entry appears in the flat CSV; filter it out when comparing per-call forward times across runs." This implies that filtering `_capture_trace` makes `_forward` durations comparable between cache-miss and cache-hit runs. This is incorrect. The `_forward` timing window in `TracedRun.module_run` uses `begin = time.time()` at source line 1086 (before the `_TRACE_RUNNING_LOCK` acquisition) and `end = time.time()` at source line 1141 (after the entire cache-miss branch, including the `_capture_trace` call at lines 1131–1136). The `_capture_trace` timing is a sub-interval of the `_forward` window. Filtering `_capture_trace` from the CSV does not remove the capture overhead from the `_forward` entry — it is already baked in. A developer comparing `_forward` durations between a warm-up run (cache miss) and a steady-state run (cache hit) would see the cache-miss `_forward` inflated by the full trace capture cost and incorrectly diagnose a regression or unexplained latency difference.

**Correct statement:** On a cache-miss run, `{ClassName}_forward` duration includes the full trace capture time because the `_forward` timing window encompasses `_capture_trace`. Filtering `_capture_trace` from the CSV is insufficient to make `_forward` comparable across runs. Use only cache-hit rows (runs where no `_capture_trace` entry exists for the same `(module_name, func_name)`) when comparing steady-state forward times.

**Fix applied:** Replaced the single-sentence note with an accurate description explaining that `_forward` on a cache miss already encompasses `_capture_trace`, and directed developers to compare only cache-hit rows for steady-state timing analysis.

---

## Verdict

Two material issues found and fixed. Issue 1 is architectural: a developer who places `@trace_enabled` only on inner submodules and not on the outermost wrapper module would find no traces ever captured, because the outer non-trace-enabled `module_run` sets `_TRACE_RUNNING = True` before its `forward()` call. Issue 2 is numerical: the guidance to "filter `_capture_trace` when comparing per-call forward times" is incorrect because `_forward` on cache-miss runs already includes the capture overhead inside its own timing window. No further material issues found; chapter is approved pending these two fixes.

---

## Change Log — Pass 7 fixes applied

- Fix 1: `run_modes.md` — replaced step 4 of `TracedRun.module_run` with a precise description of the `_TRACE_RUNNING` flag lifecycle: the flag is set to `True` before the `is_trace_enabled` check (not only on the traced execution path), so all nested `module_run` calls during a non-trace-enabled module's `forward()` also take the early-return path. Added explicit warning that outer non-trace-enabled modules suppress trace capture in all nested submodules.
- Fix 2: `dispatch_manager.md` — replaced the timing note with an accurate explanation that `{ClassName}_forward` on a cache-miss run already encompasses the `_capture_trace` duration, making cache-miss and cache-hit `_forward` durations non-comparable. Directed developers to use only cache-hit rows for steady-state timing comparisons.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 8

## Re-check of Pass 7 fixes

Both Pass 7 fixes are correctly applied in the current files:

1. **`_TRACE_RUNNING` flag lifecycle fix** (`run_modes.md` step 4 of `TracedRun.module_run`): The text now accurately describes the lock acquisition and unconditional `_TRACE_RUNNING = True` assignment before the `is_trace_enabled` check (source lines 1089–1092), the early-return path calling `forward()` with `_TRACE_RUNNING = True`, and the explicit warning that outer non-trace-enabled modules suppress trace capture in nested submodules. Confirmed against source. ✓

2. **`_forward` encompasses `_capture_trace` on cache-miss fix** (`dispatch_manager.md` timing note, line 158): The note now correctly states that the `_forward` timing window (`begin` at source line 1086, `end` at line 1141) already encompasses the full `_capture_trace` call on cache-miss runs, and directs developers to compare only cache-hit rows for steady-state timing. Confirmed against source lines 1086–1141. ✓

---

## New issues found

### Issue 1 — `run_modes.md`, helper transform table, `copy_to_ttnn` row — silent passthrough when `elem` is `None` not documented

**File:** `run_modes.md`, helper transform table, `copy_to_ttnn` row (line 276, before fix).

**Problem:** The guide described `copy_to_ttnn` as a function that "creates a fresh `ttnn.Tensor` from `elem.clone()`, preserving layout and moving to the original device." This description implies the function always creates a fresh copy for any `TorchTTNNTensor` input. The source (`run_config.py` lines 162–168) shows the fresh-copy branch executes only when `isinstance(e, TorchTTNNTensor) and e.elem is not None and e.ttnn_tensor is not None`. If `e.elem is None` — which occurs after `create_new_ttnn_tensors_using_torch_output` with `assign_ttnn_to_torch=False` (SEL mode) clears `elem` — or if `e.ttnn_tensor is None`, `copy_to_ttnn` returns the original tensor object unchanged without any copy. A developer using `DPLRunNoErrorProp`-style code on tensors flowing from a preceding SEL-mode forward pass (where `elem` was cleared) would believe error propagation is prevented, when in fact the function silently passed the original tensor through and the TTNN state is not reset.

**Correct statement:** `copy_to_ttnn` only creates a fresh TTNN tensor when both `e.elem is not None` and `e.ttnn_tensor is not None` are satisfied. For any other input (including tensors where `elem` has been cleared), the original value is returned unchanged and no isolation occurs.

**Fix applied:** Updated the `copy_to_ttnn` table entry in `run_modes.md` to document the dual-condition requirement and explicitly note that the function returns the original tensor unchanged when either `elem` or `ttnn_tensor` is absent.

---

## Verdict

One material issue found and fixed. It is an implementation correctness issue: the undocumented conditional in `copy_to_ttnn` means `DPLRunNoErrorProp`'s error-propagation prevention silently fails for tensors whose `elem` field has been cleared (e.g., by a preceding SEL-mode dispatch). A developer who receives no copy and no error would incorrectly believe the tensor was safely isolated. No other issues meeting the materiality threshold were found.

## Change Log — Pass 8 fixes applied

- Fix 1: `run_modes.md` — updated the `copy_to_ttnn` row in the helper transform table to document that the fresh-copy operation is conditional on both `e.elem is not None` and `e.ttnn_tensor is not None`; clarified that when either field is absent the original tensor is returned unchanged and TTNN error isolation does not occur.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 9

## Re-check of Pass 8 fix

The Pass 8 fix is correctly applied. `run_modes.md` line 276 (the `copy_to_ttnn` row in the helper transform table) now reads: "**Only operates when both `e.elem is not None` and `e.ttnn_tensor is not None`**; if either field is absent, the original tensor is returned unchanged and no fresh copy is made." This matches the dual-condition guard at `run_config.py` line 162 (`isinstance(e, TorchTTNNTensor) and e.elem is not None and e.ttnn_tensor is not None`). Fix correctly applied. ✓

---

## New issues found

### Issue 1 — `run_modes.md`, line 3 — `_RUN_MODE_REGISTRY` attributed to wrong line number

**File:** `run_modes.md`, opening paragraph (line 3).

**Problem:** The guide stated "`_RUN_MODE_REGISTRY` (defined at line 1103 of `core/run_config.py`)." Source line 1103 falls inside `TracedRun.module_run` (within the non-trace-enabled early-return branch). The actual definition of `_RUN_MODE_REGISTRY` is at source line 1167. A developer navigating to line 1103 to inspect the registry would land in the middle of unrelated `TracedRun` logic and either conclude the guide is wrong about the structure or examine the wrong code when reasoning about which classes are registered.

**Correct statement:** `_RUN_MODE_REGISTRY` is defined at line 1167 of `core/run_config.py`.

**Fix applied:** Changed "line 1103" to "line 1167" in `run_modes.md` line 3.

---

### Issue 2 — `run_modes.md`, step 5 of `TracedRun.module_run` — `_make_cache_key` omits `func_kwargs` from the call signature and description

**File:** `run_modes.md`, step 5 of `TracedRun.module_run` description (line 208, before fix).

**Problem:** The guide described the cache key as built via `_make_cache_key(module_name, func_args)`, and said it "hashes module name plus the shape/dtype/layout signature of each input tensor." The source (line 1117) calls `TracedRun._make_cache_key(self.module_name, func_args, func_kwargs)`, and `_make_cache_key` (lines 956–963) builds `kwargs_sig` from all tensor-valued keyword arguments, including them in the returned tuple. A developer who implements their own cache-key derivation, cache-hit prediction, or manual cache invalidation using only positional args would produce keys that differ from what `TracedRun` actually uses whenever tensor keyword arguments (e.g., `attention_mask`, KV-cache tensors) are present — causing spurious cache misses or stale hits.

**Correct statement:** `_make_cache_key(module_name, func_args, func_kwargs)` includes both positional and keyword tensor signatures in the cache key. Models that pass tensors as keyword arguments will have those shapes included in the key.

**Fix applied:** Updated step 5 in `run_modes.md` to show all three arguments and explicitly note that keyword tensor arguments are also hashed into the cache key.

---

## Verdict

Two material issues found and fixed. Issue 1 (wrong line number) would redirect a developer to unrelated source code when trying to locate the mode registry. Issue 2 (missing `func_kwargs` in `_make_cache_key` description) is an implementation-correctness issue: any developer building cache inspection or invalidation logic from the guide's description would produce incomplete cache keys, causing incorrect cache behavior for models that use tensor keyword arguments.

## Change Log — Pass 9 fixes applied

- Fix 1: `run_modes.md` line 3 — corrected `_RUN_MODE_REGISTRY` line reference from "line 1103" to "line 1167".
- Fix 2: `run_modes.md` step 5 of `TracedRun.module_run` — updated `_make_cache_key` call to include `func_kwargs` as the third argument and extended the description to note that keyword-argument tensor signatures are also included in the cache key.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 10

## Re-check of Pass 9 fixes

Both Pass 9 fixes are correctly applied in the current files:

1. **`_RUN_MODE_REGISTRY` line reference** (`run_modes.md` line 3): The text now reads "defined at line 1167 of `core/run_config.py`." Confirmed against source — `_RUN_MODE_REGISTRY = {...}` begins at line 1167. ✓

2. **`_make_cache_key` three-argument form** (`run_modes.md` step 5 of `TracedRun.module_run`): The text now shows `_make_cache_key(module_name, func_args, func_kwargs)` with all three arguments, and explicitly notes that keyword-argument tensor signatures are included in the cache key. Confirmed against source lines 956–963 and 1117. ✓

---

## New issues found

### Issue 1 — `run_modes.md`, helper transform table and `set_device_wrap` explanation — function described as handling only `ttnn.Tensor`, but it also handles `TorchTTNNTensor`

**File:** `run_modes.md`, helper transform table (`set_device_wrap` row, line 277) and the `set_device_wrap` vs `to_ttnn_wrap` comparison paragraph (line 286).

**Problem:** The table entry read: "`set_device_wrap(device)` … Returns a function that calls `ttnn.to_device` to move a `ttnn.Tensor` or the `ttnn_tensor` field of a `TorchTTNNTensor` onto `device`…" — this part of the table was partially accurate. However, the comparison paragraph at line 286 stated: "`set_device_wrap(device)` assumes the tensor is already a `ttnn.Tensor` and calls `ttnn.to_device`." This is incorrect. Source lines 344–356 show `set_device_wrap` has two explicit branches: one for raw `ttnn.Tensor` inputs (line 348) and a separate `elif` branch for `TorchTTNNTensor` inputs whose `ttnn_tensor` field is set (line 350–351) — it moves `e.ttnn_tensor` in-place for that case. A developer reading the comparison paragraph would believe `set_device_wrap` only works on unwrapped `ttnn.Tensor` values and would add an unnecessary intermediate unwrapping step (or incorrectly conclude that `TorchTTNNTensor` objects must be converted to raw `ttnn.Tensor` before the function is applied), leading to architecturally incorrect custom transform chains.

**Correct statement:** `set_device_wrap` handles both raw `ttnn.Tensor` objects (moves them directly) and `TorchTTNNTensor` objects with a non-`None` `ttnn_tensor` field (moves the `ttnn_tensor` field in-place). In the composed transform chain `wrap_to_torch_ttnn_tensor -> to_ttnn_wrap -> set_device_wrap`, `to_ttnn_wrap` produces a raw `ttnn.Tensor`, so `set_device_wrap` receives a `ttnn.Tensor` in that specific chain — but the function is equally capable when applied to `TorchTTNNTensor` inputs directly.

**Fix applied:** Updated the comparison paragraph in `run_modes.md` to correctly state that `set_device_wrap` handles both `ttnn.Tensor` and `TorchTTNNTensor` inputs, with a note explaining why only the `ttnn.Tensor` branch is exercised in the standard transform chain.

---

## Verdict

One material issue found and fixed. The incorrect "assumes the tensor is already a `ttnn.Tensor`" characterization of `set_device_wrap` would cause developers implementing custom transform chains to misunderstand the function's input contract, leading to unnecessary intermediate unwrapping steps or incorrect conclusions about which tensor types can be passed to it. No further issues meeting the materiality threshold were found.

## Change Log — Pass 10 fixes applied

- Fix 1: `run_modes.md` — corrected the `set_device_wrap` table entry to fully document both the `ttnn.Tensor` branch and the `TorchTTNNTensor` branch; corrected the comparison paragraph to remove the incorrect "assumes the tensor is already a `ttnn.Tensor`" claim and replace it with an accurate description of both handled types, with a note explaining the specific behavior within the standard `NormalRun` transform chain.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 11

## Re-check of Pass 10 fix

The Pass 10 fix is correctly applied. `run_modes.md` line 277 (the `set_device_wrap` table row) now correctly states that the function handles both raw `ttnn.Tensor` objects and `TorchTTNNTensor` objects with a non-`None` `ttnn_tensor` field. The comparison paragraph at line 286 no longer says "assumes the tensor is already a `ttnn.Tensor`"; instead it accurately describes both handled types and explains why only the `ttnn.Tensor` branch is exercised inside the standard `NormalRun` transform chain. Confirmed against source lines 344–356. Fix correctly applied.

---

## New issues found

### Issue 1 — `run_modes.md`, `NormalRun.module_run` step 1 and `TracedRun.module_run` step 1 — `past_key_value` kwargs are silently excluded from the input transform chain

**File:** `run_modes.md`, `NormalRun.module_run` step 1 (line 68) and `TracedRun.module_run` step 1 (line 204).

**Problem:** The guide describes step 1 of `NormalRun.module_run` as applying the transform chain `wrap_to_torch_ttnn_tensor -> to_ttnn_wrap -> set_device_wrap(device)` to all input tensors. The source (`run_config.py` lines 563–566) shows that keyword arguments whose key contains the substring `"past_key_value"` are explicitly excluded from transformation:

```python
other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
func_kwargs = tree_map(transform, other_kwargs)
func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})
```

These kwargs are passed through to `self.forward()` in their original form — not wrapped, not converted to TTNN, and not moved to device. The same exclusion is present in `TracedRun.module_run` (source lines 1066–1068). A developer implementing a transformer layer that passes KV-cache tensors as `past_key_value` kwargs would incorrectly assume those tensors are moved to the TTNN device before the forward pass; in reality they are passed raw, causing device-placement mismatches at runtime.

**Correct statement:** The transform chain is applied only to args and to kwargs whose keys do not contain `"past_key_value"`. Any kwarg whose key contains `"past_key_value"` bypasses all transformation and is forwarded unchanged. This is an acknowledged source-level limitation (the source has a `# TODO: fix kwds not being passed correctly` comment at line 563).

**Fix applied:** Added a warning sentence to step 1 of `NormalRun.module_run` in `run_modes.md` explicitly documenting the `"past_key_value"` exclusion and its implication.

---

### Issue 2 — `run_modes.md`, `TracedRun.configure()` description — "resets `_trace_cache`" understates the effect; GPU trace allocations are also released

**File:** `run_modes.md`, `TracedRun` class-level configuration section (line 200, before fix).

**Problem:** The guide stated "`configure()` also resets `_trace_cache`." This implies the method only discards the Python-side cache dictionary. The source (`run_config.py` lines 923–929) shows that `configure()` calls `cls.release_all()` as its first action. `release_all()` (lines 940–944) iterates over every cached `TraceEntry` and calls `ttnn.release_trace(entry.device, entry.trace_id)` on each — freeing the on-device trace allocation — before clearing the cache dict. Two categories of developer error follow from the incomplete description: (a) a developer who calls `configure()` mid-run expecting to change only settings (e.g., `cq_id`) without destroying cached traces would lose all captures and incur full re-capture overhead on the next forward pass without understanding why; (b) a developer who wants to free device trace memory before reinitializing the device would not know that `configure()` is sufficient and might not call it, leading to a resource leak if they only call `TracedRun._trace_cache.clear()` directly.

**Correct statement:** `configure()` calls `release_all()` as its first action, which calls `ttnn.release_trace` on every cached entry (freeing GPU-side trace memory) and then clears the Python cache dict. Any previously captured traces are fully destroyed before the new configuration parameters are stored.

**Fix applied:** Replaced the one-sentence description with an accurate explanation of the `release_all()` call chain and its consequences, including a note that existing captures are fully destroyed and re-capture will occur on the next forward pass.

---

## Verdict

Two material issues found and fixed. Issue 1 is an implementation-correctness issue: the undocumented `past_key_value` exclusion from the input transform chain would cause any developer implementing a model with KV-cache kwargs to encounter device-placement errors that the guide gives no warning of. Issue 2 is an architectural misunderstanding issue: calling `configure()` silently destroys all GPU-side trace allocations, which is significant enough that any developer reconfiguring `TracedRun` parameters mid-run or managing device memory must know about it.

## Change Log — Pass 11 fixes applied

- Fix 1: `run_modes.md` — added a warning to step 1 of `NormalRun.module_run` (and by reference `TracedRun.module_run` step 1) documenting that kwargs whose keys contain `"past_key_value"` are excluded from the `wrap_to_torch_ttnn_tensor -> to_ttnn_wrap -> set_device_wrap` transform and passed unchanged to `self.forward()`.
- Fix 2: `run_modes.md` — replaced the one-sentence "`configure()` also resets `_trace_cache`" with an accurate description: `configure()` calls `release_all()`, which calls `ttnn.release_trace` on every cached entry (freeing GPU-side allocations) and clears the Python cache, forcing full re-capture on the next forward pass.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 12

## Re-check of Pass 11 fixes

Both Pass 11 fixes are correctly applied in the current files:

1. **`past_key_value` exclusion documented** (`run_modes.md` line 68): Step 1 of `NormalRun.module_run` now includes an explicit warning that kwargs whose keys contain `"past_key_value"` are excluded from the transform chain and passed unchanged to `self.forward()`. The same exclusion is referenced by `TracedRun.module_run` step 1 ("Applies the same input transform chain as `NormalRun`"). Confirmed against source lines 563–566 and 1066–1068. Fix correctly applied. ✓

2. **`configure()` release_all description** (`run_modes.md` line 200): The description now accurately explains that `configure()` calls `release_all()` as its first action, which calls `ttnn.release_trace` on every cached entry (freeing GPU-side allocations) before clearing the Python cache. Confirmed against source lines 923–929 and 940–944. Fix correctly applied. ✓

---

## New issues found

### Issue 1 — `run_modes.md`, `NormalRunWithFallback.module_run` description — missing warning that ALL kwargs including `past_key_value` are transformed

**File:** `run_modes.md`, `NORMAL_WITH_FALLBACK` module-level behavior section (line 102–104, before fix).

**Problem:** The guide documents the `past_key_value` exclusion only for `NormalRun.module_run` (step 1, line 68) and says `TracedRun.module_run` applies "the same input transform chain as `NormalRun`". The description of `NormalRunWithFallback.module_run` says only "Adds a `try/except` around `self.forward(...)`" with no mention of its kwargs handling. The source (`run_config.py` lines 624–626) shows `NormalRunWithFallback.module_run` applies `func_kwargs = tree_map(transform, kwds)` with no `past_key_value` filtering — all kwargs are transformed, wrapped, and moved to device. This is materially different from `NormalRun.module_run` behavior. A developer who switches a KV-cache transformer from `NORMAL` to `NORMAL_WITH_FALLBACK` to get error-fallback behavior would find that `past_key_value` tensors are now device-wrapped and converted to TTNN, which may cause runtime failures or silently change the computation. The guide gives no warning of this behavioral difference.

**Correct statement:** `NormalRunWithFallback.module_run` applies the transform to ALL keyword arguments — including those whose keys contain `"past_key_value"` — because it does not apply the `past_key_value` exclusion present in `NormalRun.module_run` and `TracedRun.module_run`.

**Fix applied:** Added a paragraph before the `try/except` description in the `NormalRunWithFallback` module-level behavior section, documenting that all kwargs including `past_key_value` are transformed, and explicitly noting the difference from `NormalRun` and `TracedRun`.

---

### Issue 2 — `dispatch_manager.md`, line 3 — `DispatchManager` class definition line number is off by one

**File:** `dispatch_manager.md`, opening paragraph (line 3).

**Problem:** The guide stated "`DispatchManager` is defined in `core/run_config.py` (line 178)." The class definition `class DispatchManager:` is at source line 179. Line 178 is a blank line preceding the class. A developer navigating directly to line 178 lands on blank whitespace, not the class definition.

**Correct statement:** `DispatchManager` is defined at line 179 of `core/run_config.py`.

**Fix applied:** Changed "line 178" to "line 179" in `dispatch_manager.md` line 3.

---

## Verdict

Two issues found and fixed. Issue 1 is an implementation-correctness issue: `NormalRunWithFallback.module_run` transforms ALL kwargs including `past_key_value`, unlike `NormalRun` and `TracedRun`. A developer migrating a KV-cache transformer to `NORMAL_WITH_FALLBACK` would encounter unexpected device-placement or TTNN-conversion of KV-cache tensors with no guidance from the guide. Issue 2 is a minor navigation error (line number off by one) that would cause a developer to land on blank whitespace rather than the class definition. No further material issues found; chapter is approved pending these two fixes.

## Change Log — Pass 12 fixes applied

- Fix 1: `run_modes.md` — added a paragraph to `NormalRunWithFallback.module_run` description stating that ALL kwargs (including `past_key_value`) are passed through the transform chain, and explicitly noting this is different from `NormalRun.module_run` and `TracedRun.module_run`. Confirmed against source lines 624–626 vs 563–566.
- Fix 2: `dispatch_manager.md` line 3 — corrected `DispatchManager` class definition line reference from "line 178" to "line 179". Confirmed against source line 179.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Pass 13

## Re-check of Pass 12 fixes

Both Pass 12 fixes are correctly applied in the current files:

1. **`NormalRunWithFallback` transforms all kwargs including `past_key_value`** (`run_modes.md` lines 102–104): The module-level behavior section now contains an explicit paragraph stating that `NormalRunWithFallback.module_run` applies `tree_map(transform, kwds)` without `past_key_value` filtering, and notes the behavioral difference from `NormalRun.module_run` and `TracedRun.module_run`. Confirmed against source lines 624–626 (`func_kwargs = tree_map(transform, kwds)`, no filtering) vs 563–566 (NormalRun filters). Fix correctly applied. ✓

2. **`DispatchManager` line number corrected to 179** (`dispatch_manager.md` line 3): The opening paragraph now reads "defined in `core/run_config.py` (line 179)." Source confirms `class DispatchManager:` is at line 179. Fix correctly applied. ✓

---

## New issues found

No feedback — chapter approved.

A thorough independent pass was made against the source (`run_config.py` lines 101–1228). All material claims were verified:

- `NormalRun.torch_dispatch` `can_dispatch_to_ttnn` timing entry: only `NormalRun`, absent from all other modes including `NormalRunWithFallback`. ✓
- `SELRun.torch_dispatch` `create_new_ttnn_tensors_using_torch_output` default `assign_ttnn_to_torch=False` (source line 662, no kwarg passed). ✓
- `DPLRun.torch_dispatch` `assign_ttnn_to_torch=True` (source line 704). ✓
- `DPLRunNoErrorProp` ATen behavior: `copy_to_ttnn` applied to original `args`/`kwargs` (not the `copy_to_torch` clones), then `can_dispatch_to_ttnn` called on the fresh TTNN copies (source lines 748–753). ✓
- `TracedRun._blocking` defaults to `True` (source line 913); `configure()` calls `release_all()` first (source line 924), which iterates cache and calls `ttnn.release_trace` on each entry (source lines 940–944). ✓
- `_RUN_MODE_REGISTRY` at source line 1167. ✓
- `_make_cache_key` three-argument form including `func_kwargs` (source lines 956–963, 1117). ✓
- Module-name stack push/pop in `TracedRun.module_run` is unconditional before the `is_trace_enabled` check (source line 1073), covering all execution paths. ✓
- `TracedRun.module_run` non-trace-enabled early-return path records `_preprocess_weights`, `_move_weights_to_device`, and `_forward` but NOT `TorchModules` (source lines 1074–1109). ✓
- `NormalRunWithFallback.module_run` contains zero `record_timing` and zero `set_current_module_name` calls (source lines 619–643). ✓
- `copy_to_ttnn` conditional on both `e.elem is not None` and `e.ttnn_tensor is not None` (source line 162). ✓
- `set_device_wrap` handles both `ttnn.Tensor` and `TorchTTNNTensor` branches (source lines 347–354). ✓
- `past_key_value` exclusion in `NormalRun.module_run` and `TracedRun.module_run` (source lines 563–566, 1066–1068); absent in `NormalRunWithFallback.module_run` (source line 626). ✓
- `DispatchManager` class-level state, `set_current_module_name`, dispatch wrappers, `record_timing`, `save_stats_to_file` all match documented behavior. ✓

## Verdict

Chapter approved. No new material issues found. All 25 prior fixes from Passes 1–12 are correctly applied and verified against source.

## Change Log — Pass 13 fixes applied

No fixes required.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Post-Compression Pass

## Post-compression check (5 changed areas)

**1. `dispatch_manager.md`: `no_dispatch()` re-explanation replaced with cross-reference**

Applied cleanly with no new errors. `dispatch_manager.md` line 127 contains: "See [`no_dispatch()` in run_modes.md](run_modes.md#the-no_dispatch-context-manager) for why this prevents re-entry into `__torch_dispatch__`." The link uses a named anchor that matches the heading in `run_modes.md` (`## The \`no_dispatch()\` context manager`). The `run_modes.md` section (lines 253–266) gives a complete explanation including the implementation, the re-entry problem, and the solution. Any developer reading `dispatch_manager.md` first can follow the link and get full context. The cross-reference is sufficient. No issue.

**2. `dispatch_manager.md`: SEL/DPL not calling `set_current_module_name` replaced with cross-reference**

Applied cleanly. The `> **Important:**` callout at `dispatch_manager.md` lines 43–44 states the critical behavioral fact directly inline: "SEL, DPL, and DPL_NO_ERROR_PROP module_run methods do not call `set_current_module_name` at all. ATen ops dispatched during `self.forward(...)` in those modes will see `current_module_name` from whatever parent module is already on the stack (or `None` at the top level). Do not rely on `current_module_name` for op attribution when using SEL or DPL run modes." This fact is stated in the callout itself, not only via cross-reference. The information is findable without reading any other file. No issue.

**3. `dispatch_manager.md`: "(not `NormalRunWithFallback`)" parenthetical removed from timing table**

Applied cleanly. Two places cover this fact after removal: (a) the prose block at `dispatch_manager.md` line 148 reads "`NormalRunWithFallback` overrides `module_run` with its own implementation that contains no `record_timing` calls and no `set_current_module_name` calls. Modules running under `NORMAL_WITH_FALLBACK` mode produce **no module-level timing entries**…"; (b) the timing table `Paths recorded` column at line 156 reads "never for `NormalRunWithFallback`" for the `TorchModules` row. The prose block is immediately above the timing table, so any reader of the table would encounter it. No issue.

**4. `run_modes.md`: `isinstance`/inheritance explanation trimmed in step 3**

Applied cleanly. The remaining text at `run_modes.md` line 208 reads: "trace-enablement is inherited via `isinstance`, so a subclass decorated with `@trace_disabled` can opt out of a parent's `@trace_enabled`. See the [decorator documentation below](#trace_enabled-and-trace_disabled-decorators) for the full `isinstance`-based check." This correctly conveys (a) that `isinstance` is used (not exact type matching), (b) that inheritance propagates trace-enablement to subclasses automatically, and (c) that `@trace_disabled` is the mechanism to opt out. All three facts are necessary for correct implementation and all three remain. The link to the decorator section provides additional detail. No issue.

**5. `index.md`: Quick-reference table condensed to 3 rows**

Partially applied — issue found. See Issue 1 below.

---

## New issues found

### Issue 1 — `index.md`, line 14 — `NORMAL_WITH_FALLBACK` absent from quick-reference guidance

**File:** `index.md`, "Quick-reference: when to use which mode" section (line 14, before fix).

**Problem:** The condensed quick-reference explicitly named four modes as "self-describing" (`NORMAL`, `LIGHTWEIGHT`, `CPU`, `TRACED`) and listed three modes with "non-obvious distinctions" in the table (`SEL`, `DPL`, `DPL_NO_ERROR_PROP`). `NORMAL_WITH_FALLBACK` — the eighth mode — appeared nowhere in the quick-reference section. A reader using `index.md` as the entry point would see seven of the eight modes receive some guidance but `NORMAL_WITH_FALLBACK` would be invisible: not labeled self-describing, not in the table, and with no "when to use it" signal. The Contents table at line 9 mentions it parenthetically in a list of all eight mode names, but provides no decision guidance. A developer would not know when to choose `NORMAL_WITH_FALLBACK` over `NORMAL`, or that it exists as a distinct debugging tool.

**Correct statement:** `NORMAL_WITH_FALLBACK` is like `NORMAL` but wraps every TTNN dispatch in a `try/except` and falls back to PyTorch on error — the appropriate choice when some ops may not yet be supported in TTNN and silent per-op fallback is acceptable.

**Fix applied:** Added a sentence to `index.md` line 14, between the self-describing modes callout and the three-row table, describing when to use `NORMAL_WITH_FALLBACK`.

---

## Verdict

One material issue found and fixed. The compression that condensed the quick-reference table to 3 rows also removed all mention of `NORMAL_WITH_FALLBACK` from the quick-reference section, leaving a reader using `index.md` as entry point unable to discover when to choose that mode. All four other compression changes applied cleanly with no new errors introduced.

## Change Log — Post-Compression fixes applied

- Fix 1: `index.md` line 14 — inserted a sentence describing `NORMAL_WITH_FALLBACK` in the quick-reference section: it wraps every TTNN op dispatch in a `try/except` and falls back to PyTorch on error, suitable when some ops may not yet have TTNN support and silent per-op fallback is acceptable.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Post-Compression Pass 2

## Re-check of Post-Compression fix

The `NORMAL_WITH_FALLBACK` fix is correctly applied in `index.md`. Line 14 of `index.md` now reads: "`NORMAL_WITH_FALLBACK` is like `NORMAL` but wraps every TTNN op dispatch in a `try/except` and falls back to PyTorch on error — use it when some ops may not yet be supported in TTNN and silent per-op fallback is acceptable." This accurately reflects the source behavior (`NormalRunWithFallback.torch_dispatch`, `run_config.py` lines 603–617) and provides actionable selection guidance. Fix correctly applied.

---

## New issues found

### Issue 1 — `dispatch_manager.md`, class-level state table — `timings` described as having only one runtime key, but `record_timing` silently creates empty per-backend keys

**File:** `dispatch_manager.md`, Class-level state table, `timings` row (was line 18 before fix).

**Problem:** The guide stated "The only key used at runtime is `"TimingEntries"`". The source (`run_config.py` lines 249–252) shows that `record_timing` first checks `if backend not in DispatchManager.timings` and, when absent, creates `DispatchManager.timings[backend] = {}`. This runs on every unique backend value (`"TTNN"`, `"Torch"`, `"TorchModules"`). After a typical inference run, `DispatchManager.timings` has four top-level keys: `"TTNN"`, `"Torch"`, `"TorchModules"` (each an empty dict), plus `"TimingEntries"` (the flat list). A developer who inspects `DispatchManager.timings` and reads `DispatchManager.timings["TTNN"]` expecting to find timing records would receive `{}` (an empty dict), not an error — silently producing an empty analysis. A developer iterating `DispatchManager.timings.items()` to find all data would encounter three spurious empty-dict entries alongside the real `"TimingEntries"` list.

**Correct statement:** `record_timing` creates one empty sentinel dict per distinct `backend` value seen at runtime. After a run, `DispatchManager.timings` contains `{"TTNN": {}, "Torch": {}, "TorchModules": {}, "TimingEntries": [...]}`. All actual timing data lives exclusively under `"TimingEntries"`. Use `DispatchManager.timings["TimingEntries"]` or `get_timing_entries_stats()` to access timing records; never read `DispatchManager.timings[backend_name]` expecting records.

**Fix applied:** Updated the `timings` row in `dispatch_manager.md` to document that per-backend sentinel keys (`"TTNN"`, `"Torch"`, `"TorchModules"`) are created as empty dicts by `record_timing` but hold no data. Added explicit guidance not to read `DispatchManager.timings["TTNN"]` etc. for timing records.

---

## Verdict

One material issue found and fixed. A developer who reads `DispatchManager.timings["TTNN"]` (a natural first instinct given the structure) would receive an empty dict silently, producing a null analysis and no error message. The prior description ("only key used at runtime is `"TimingEntries"`") did not warn that the other keys exist but are empty, leaving this trap undocumented. No further issues meeting the materiality threshold were found. Chapter approved.

## Change Log

- Fix 1: `dispatch_manager.md` — updated the `timings` row in the class-level state table to document that `record_timing` creates empty sentinel dicts for each backend name seen (`"TTNN"`, `"Torch"`, `"TorchModules"`) and that all actual timing records live under `"TimingEntries"`. Added explicit guidance to use `timings["TimingEntries"]` or `get_timing_entries_stats()` and not to read `timings[backend_name]` directly.

---

# Agent B Review — Chapter 3: Run Modes and the Dispatch Manager — Post-Compression Pass 3

## Re-check of most recent fix

The `timings` dict per-backend sentinel keys fix (Post-Compression Pass 2, Fix 1) is correctly applied. `dispatch_manager.md` line 18 (the `timings` row in the class-level state table) now reads: "Top-level container. `record_timing` creates a key for each distinct `backend` value (e.g., `"TTNN"`, `"Torch"`, `"TorchModules"`) as an empty sentinel dict `{}`, but these per-backend keys are never populated with data. All actual timing data is stored under the `"TimingEntries"` key as a flat list of timing record dicts. Do not read `DispatchManager.timings["TTNN"]` expecting timing records; use `DispatchManager.timings["TimingEntries"]` or `get_timing_entries_stats()` instead."

Confirmed against source `run_config.py` lines 249–252: `record_timing` performs `if backend not in DispatchManager.timings: DispatchManager.timings[backend] = {}` on every call, creating empty per-backend sentinel dicts, with all actual records appended to `DispatchManager.timings["TimingEntries"]`. Fix correctly applied.

---

## New issues found

No feedback — chapter approved.

Full independent sweep against `run_config.py` lines 101–1228:

- `DispatchManager.record_timing` sentinel-key creation and `TimingEntries` flat list: matches `dispatch_manager.md` table row. ✓
- `NormalRun.module_run` execution order — transforms applied first (lines 561–566), `preprocess_weights()` called at line 568 before the stack push at line 570, `move_weights_to_device()` called at line 575 after push; both `_preprocess_weights` and `_move_weights_to_device` timing entries use explicit `self.module_name` (not `current_module_name`), which is identical in value after the push. Guide pseudocode correctly shows this order. ✓
- `past_key_value` exclusion in `NormalRun.module_run` (lines 564–566) and `TracedRun.module_run` (lines 1066–1068); absent in `NormalRunWithFallback.module_run` (line 626 — `func_kwargs = tree_map(transform, kwds)`, no filtering). All three documented correctly. ✓
- `TracedRun.module_run` — `_TRACE_RUNNING = True` set unconditionally at lines 1091–1092 before `is_trace_enabled` check at line 1094; early-return path (line 1110) resets to `False` only if `not already_running`; traced path resets in `finally` block at lines 1138–1140. Guide step 4 accurately describes all three paths including the suppression of nested trace capture. ✓
- `TracedRun._blocking` defaults to `True` (line 913); `execute_trace` uses `blocking=TracedRun._blocking` (line 1124). ✓
- `TracedRun.configure()` calls `cls.release_all()` first (line 924); `release_all()` calls `ttnn.release_trace` on each cached entry (lines 942–943) before clearing (line 944). ✓
- `disable_trace` sets `_TRACE_RUNNING = True` for function duration and restores `was_tracing` in `finally` (lines 1154–1161). Guide description ("set to `True` for its duration, preventing nested trace capture") is accurate. ✓
- `_RUN_MODE_REGISTRY` at line 1167; `_make_cache_key` three-argument form at lines 956–963, called with all three args at line 1117. ✓
- `SELRun`, `DPLRun`, `DPLRunNoErrorProp` module_run methods contain no `set_current_module_name` calls (lines 665–686, 707–736, 762–796). ✓
- `NormalRunWithFallback.module_run` contains no `record_timing` or `set_current_module_name` calls (lines 619–643). ✓
- `copy_to_ttnn` conditional at line 162: `isinstance(e, TorchTTNNTensor) and e.elem is not None and e.ttnn_tensor is not None`. ✓
- `set_device_wrap` handles both `ttnn.Tensor` (line 348) and `TorchTTNNTensor` with non-`None` `ttnn_tensor` (lines 350–351). ✓
- `TracedRun._capture_trace` records `_capture_trace` timing at lines 1134–1136 (cache-miss path only); `_forward` timing window (begin at line 1086, end at line 1141) encompasses the capture call. ✓
- `save_stats_to_file` pivot indexed by `["func_name", "module_name"]` (line 282); top-30 printout pivots by `index=["func_name"]` (line 306), printing class names not instance paths. ✓
- `get_timing_entries_stats` Step 5 example groups by `"module_name"` — confirmed in `dispatch_manager.md` lines 263–270. ✓

## Verdict

Chapter approved. The sentinel-key fix is correctly applied. No new material issues found. All documented claims across `index.md`, `run_modes.md`, and `dispatch_manager.md` match source behavior.

## Change Log

No fixes required.
