# B Review — Pass 1

## Verdict

One issue found. See Item 1 below. All other content is factually correct.

---

## Issue 1 — `audit_results.md` Section 2 implies all calls inside `TracedRun._capture_trace` are non-blocking (overly broad)

**File:** `audit_results.md`, Section 2, second bullet.

**What the text says:**

> Calls inside `TracedRun._capture_trace` during the warm-up compile run (before `begin_trace_capture`): … does not block trace capture — it only adds latency to warm-up.

**The problem:**

The qualifying phrase "before `begin_trace_capture`" correctly limits the non-blocking claim to the warm-up phase. However, the bullet is filed under the heading "Expected Non-Forward-Path Calls (Not Trace-Blocking)" and reads as a blanket dismissal of the entire method. An engineer scanning this section to quickly dismiss grep hits will see `_capture_trace` in the list and may dismiss a call that appears *after* `begin_trace_capture` inside the same method — that call *would* be inside the capture bracket and would be trace-blocking.

This contradicts `audit_methodology.md` Section 2, which correctly lists `TracedRun._capture_trace` and `TracedRun.execute` under "Forward-path calls that must be investigated" when those methods invoke module forward methods.

**Required correction:**

Rewrite the bullet so that the non-blocking claim is tightly scoped to the pre-capture warm-up phase, and explicitly warn that any call appearing after `begin_trace_capture` in the same method is in scope and must not be dismissed:

> **Calls inside `TracedRun._capture_trace` or `TracedRun.execute` only when they appear before `begin_trace_capture` (the warm-up compile run):** The warm-up run executes the full forward pass to compile ops before the trace bracket opens. A `synchronize_device` in this position does not block trace capture. However, any call that appears after `begin_trace_capture` in the same method — i.e., inside the actual capture bracket — is trace-blocking and must be flagged. Do not dismiss a hit in `_capture_trace` without first confirming which side of `begin_trace_capture` it falls on.

---

## Items Verified Correct

The following were checked against the provided domain facts and found to have no errors:

- **Search command pattern** (`synchronize_device\|synchronize_devices`): correctly catches both singular and plural forms, satisfying the domain requirement to flag both.
- **Exclusion filters** (`grep -v "test" | grep -v "benchmark" | grep -v "profil"`): correctly exclude test files, benchmark scripts, and profiling scripts from the raw grep output. The remaining non-forward-path categories (`__init__`, `move_weights_to_device_impl`, warmup variants, `compile_run`) are correctly handled via manual classification in Section 2 of the methodology rather than by filename filter, which is appropriate since those methods may appear in non-test source files.
- **`@trace_enabled` definition in methodology:** Correctly described as modules intended to run inside the trace bracket; their forward-path `synchronize_device` calls directly block trace capture.
- **Trace-blocking mechanism:** The description that host-blocking behavior is incompatible with trace capture is consistent with the domain fact (host-blocking is not recorded in the trace and causes capture failure).
- **Remedy Type A:** Correctly defined as delete-only when the preceding op is a synchronous TTNN op, consistent with the domain fact that CQ0 FIFO ordering makes the synchronize call redundant.
- **Remedy Type B:** Correctly defined as delete + TT_CCL cycling semaphores when the preceding op is or should be async CCL.
- **Remedy Type C:** Correctly defined as investigate-further when purpose is unclear.
- **Call 1 and Call 2 remedy classification:** The conditional "Type A if synchronous all_gather; Type B if all_gather_async" classification with explicit TODO-to-verify is appropriate given that the source was not accessible during writing. No premature classification was made.
- **`audit_results.md` Section 3 unknown call sites:** The flagging of `TTNNQwen3MoELayer`, the shared base class question, and `TTNNRotaryPositionEmbedding` as items requiring source verification is consistent with the audit scope and does not assert anything that contradicts the domain facts.
- **`compile_run` exclusion:** Both `index.md` and `audit_methodology.md` qualify the exclusion with "if it is a separate code path from `forward`", which is correct — the exclusion is conditional, not blanket.
- **Warm-up calls described as adding latency but not blocking trace capture:** Consistent with the domain fact that warm-up runs precede `begin_trace_capture`.

---

## Pass 1 Change Log

### Fix applied — 2026-04-23

**File changed:** `audit_results.md`, Section 2, second bullet.

**Change:** Tightened the non-blocking claim for `TracedRun._capture_trace` to the pre-`begin_trace_capture` (warm-up) phase only. Added an explicit warning that any call appearing after `begin_trace_capture` in the same method is inside the capture bracket and is trace-blocking, and that engineers must not dismiss a grep hit in `_capture_trace` without confirming which side of `begin_trace_capture` it falls on. No other content was modified.

---

## Pass 2

### Issues found: 1

**Issue 1:** `audit_methodology.md`, Section 3 ("Classifying Each Call"), Question 4, Type B definition — remedy description is incomplete for the sub-case where the synchronous `all_gather` should be replaced with an async variant.

**What the text says:**

> **Type B (delete + async CCL):** The preceding op is or should be `ttnn.experimental.all_gather_async` or another async CCL dispatch. Deleting `synchronize_device` alone would break ordering; the fix requires replacing the synchronize call with proper TT_CCL cycling semaphore management.

**The problem:**

The Type B trigger condition correctly covers two sub-cases via the phrase "is or should be":

- Sub-case B1: The preceding op is already `all_gather_async`. The all_gather is correct; only `synchronize_device` is wrong. The remedy is: delete `synchronize_device` and add TT_CCL cycling semaphore management. The description is accurate for this sub-case.
- Sub-case B2: The preceding op is currently the synchronous `ttnn.all_gather`, but it _should_ be `all_gather_async` (i.e., the author's intent was async dispatch but the synchronous form was used as a placeholder). The remedy requires: (1) replace the synchronous `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, AND (2) delete `synchronize_device` and add TT_CCL cycling semaphore management.

The remedy description — "the fix requires replacing the synchronize call with proper TT_CCL cycling semaphore management" — covers only sub-case B1. It omits the additional step required in sub-case B2: replacing the synchronous `all_gather` itself. An engineer applying the B2 fix who reads only the Type B definition will delete `synchronize_device` and add semaphores but leave a synchronous `ttnn.all_gather` in place, resulting in a functionally incorrect async dispatch pattern.

Domain fact 8 makes this explicit: "Remedy Type B: Replace `synchronize_device` + `all_gather` with `TT_CCL` async (async case)." The domain fact specifies that both the synchronize call and the all_gather are replaced in Type B. The methodology's description omits the all_gather replacement.

**Precise correction:**

Split the Type B description into two sub-cases, or add a parenthetical that covers the B2 scenario:

> **Type B (delete + async CCL):** The preceding op is or should be `ttnn.experimental.all_gather_async` or another async CCL dispatch. Two sub-cases apply:
>
> - **B1 — already async:** The preceding op is `all_gather_async`. Delete `synchronize_device` and replace it with TT_CCL cycling semaphore management. The all_gather call itself does not change.
> - **B2 — should be async:** The preceding op is the synchronous `ttnn.all_gather` but the intent is async dispatch. First replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, then delete `synchronize_device` and add TT_CCL cycling semaphore management. Deleting `synchronize_device` alone is insufficient in this sub-case.
>
> See Chapter 3 for the full semaphore pattern for both sub-cases.

---

## Pass 2 Change Log

### Fix applied — 2026-04-23

**File changed:** `audit_methodology.md`, Section 3 ("Classifying Each Call"), Question 4, Type B definition.

**Change:** Expanded the Type B remedy description to explicitly cover both sub-cases identified in the Pass 2 review:

- **B1 (already async):** The preceding op is already `all_gather_async`. Remedy is to delete `synchronize_device` and replace it with TT_CCL cycling semaphore management. The `all_gather` call itself does not change.
- **B2 (should be async):** The preceding op is the synchronous `ttnn.all_gather` but the intent is async dispatch. Remedy requires two steps: replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, then delete `synchronize_device` and add TT_CCL cycling semaphore management. The original description omitted this step, which would have left engineers applying the B2 fix with a synchronous `ttnn.all_gather` alongside new semaphore management — a broken async dispatch pattern.

No other content was modified.

---

## Pass 3

### Issues found: 2

**Issue 1:** `audit_methodology.md`, Section 3, Question 4, Type A definition — trigger condition is broader than the domain fact specifies.

**What the text says:**

> **Type A (delete only):** The preceding op is a synchronous TTNN op (including the synchronous variant of `ttnn.all_gather`). CQ0 ordering suffices to guarantee the result is ready before the next consumer. The `synchronize_device` call can be deleted without replacement.

**The problem:**

Domain fact 7 defines Type A with a specific trigger: "Delete `synchronize_device` only. Use when the preceding op is synchronous `ttnn.all_gather` and CQ0 ordering provides the needed guarantee." The domain fact names `ttnn.all_gather` specifically — it does not say "any synchronous TTNN op."

The methodology text frames `ttnn.all_gather` as merely one example of a broader class ("a synchronous TTNN op (including the synchronous variant of `ttnn.all_gather`)"). This extends Type A beyond what the domain fact authorizes. For a call where the preceding op is a compute op (e.g., `ttnn.matmul`, `ttnn.softmax`) with no CCL dependency, the audit methodology's own Question 2 guidance (lines 68–72) says the call "may be a debugging artifact or an overly conservative ordering guard" — which is the trigger for Type C (investigate further), not Type A (delete outright). An engineer applying the Type A definition as currently written could delete a `synchronize_device` after a compute op without investigation, when the correct action per the domain facts is to classify it Type C first.

Note: domain fact 5 does state that CQ0 ordering makes `synchronize_device` redundant after any synchronous op. However, domain fact 7 is the authoritative definition of remedy types, and it reserves Type A for the known-intent case where the preceding op is synchronous `ttnn.all_gather`. Other synchronous-op cases without a CCL dependency have unclear purpose and must route through Type C before any deletion is performed.

**Precise correction:**

Narrow the Type A trigger to match domain fact 7:

> **Type A (delete only):** The preceding op is the synchronous variant of `ttnn.all_gather`. CQ0 ordering suffices to guarantee the gather result is ready before the next consumer; the `synchronize_device` call can be deleted without replacement. If the preceding op is a compute op (e.g., `ttnn.matmul`, `ttnn.softmax`) with no CCL dependency, the purpose of the synchronize call is unclear — classify as Type C and investigate before deleting.

---

**Issue 2:** `audit_results.md`, Section 1, Call 1 and Call 2 remedy field — the two-branch conditional omits the B2 sub-case, creating an inconsistency with the updated `audit_methodology.md`.

**What the text says (Call 1 remedy field; Call 2 inherits the same classification):**

> **Remedy:** Type A (delete alone) if the all_gather variant is synchronous; Type B (delete + TT_CCL cycling semaphores) if it is `all_gather_async` — see Chapter 3 for the full verdict and the exact code change required

**The problem:**

After the Pass 2 fix, `audit_methodology.md` defines three distinct outcomes when examining a synchronize call site:

- B1: preceding op IS already `all_gather_async` — remedy is delete + TT_CCL semaphores only.
- B2: preceding op is synchronous `ttnn.all_gather` but the INTENT is async — remedy requires ALSO replacing `ttnn.all_gather` with `ttnn.experimental.all_gather_async`.
- Type A: preceding op is synchronous `ttnn.all_gather` and the intent is synchronous dispatch — remedy is delete only.

The current conditional in `audit_results.md` maps "synchronous → Type A" without qualification, which collapses B2 into Type A. B2 also has synchronous `ttnn.all_gather` as the preceding op, but its remedy is not Type A — deleting `synchronize_device` alone in a B2 context leaves the synchronous all_gather in place alongside the now-absent synchronize, producing a broken dispatch pattern that the methodology explicitly warns against.

An engineer who finds a synchronous `ttnn.all_gather` at Call 1, consults the remedy field, and sees "Type A if synchronous" will delete only the synchronize call. The conditional gives no signal that async intent is a possibility requiring a different remedy. The Chapter 3 deferral does not rescue this: Chapter 3 covers the mechanics of the fix, not the classification logic, and the engineer may not look there for classification guidance.

**Precise correction:**

Update the Remedy field for Call 1 (and equivalently for Call 2) to reflect the three-way branch introduced by the Pass 2 fix:

> **Remedy:** Depends on which all_gather variant is present and on the intended dispatch model — three cases: (1) synchronous `ttnn.all_gather` and intent is synchronous dispatch → Type A (delete `synchronize_device` only); (2) synchronous `ttnn.all_gather` but intent is async dispatch → Type B2 (first replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, then delete `synchronize_device` and add TT_CCL cycling semaphore management); (3) already `ttnn.experimental.all_gather_async` → Type B1 (delete `synchronize_device` and add TT_CCL cycling semaphore management; the all_gather call does not change). See Chapter 3 and `audit_methodology.md` Section 3 Question 4 for the full verdict and exact code changes.

---

## Pass 3 Change Log

### Fixes applied — 2026-04-23

**Fix 1 — `audit_methodology.md`, Section 3, Question 4, Type A definition.**

**Change:** Narrowed the Type A trigger from "a synchronous TTNN op (including the synchronous variant of `ttnn.all_gather`)" to "the synchronous variant of `ttnn.all_gather`" exclusively. Added an explicit redirect: if the preceding op is a compute op (e.g., `ttnn.matmul`, `ttnn.softmax`) with no CCL dependency, the call must be classified Type C and investigated before any deletion is performed. This aligns the definition with domain fact 7, which names `ttnn.all_gather` specifically as the Type A trigger, and prevents engineers from applying a delete-only remedy to synchronize calls whose purpose is unclear.

**Fix 2 — `audit_results.md`, Section 1, Call 1 and Call 2 remedy fields.**

**Change:** Replaced the two-branch "Type A if synchronous; Type B if async" conditional with a three-way branch that matches the updated methodology:
- (1) Synchronous `ttnn.all_gather` + synchronous dispatch intent → Type A (delete `synchronize_device` only).
- (2) Synchronous `ttnn.all_gather` + async dispatch intent → Type B2 (replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, then delete `synchronize_device` and add TT_CCL cycling semaphore management).
- (3) Already `ttnn.experimental.all_gather_async` → Type B1 (delete `synchronize_device` and add TT_CCL cycling semaphore management; the all_gather call does not change).

The original two-branch conditional collapsed B2 into Type A, which would have caused engineers finding a synchronous `ttnn.all_gather` to apply a delete-only remedy even when async dispatch was the intended design. Call 2's remedy field was updated in parallel with Call 1 to state the same three-way branch. No other content was modified.

---

## Pass 4

### Issues found: 1

**Issue 1:** `audit_results.md`, Section 1, Call 1 and Call 2 "Preceding op" fields — both present `ttnn.experimental.all_gather_async` as an equally plausible current variant, which contradicts domain fact 4.

**What the text says (Call 1, line 19):**

> **Preceding op:** `ttnn.all_gather` or `ttnn.experimental.all_gather_async` (TODO: verify which variant — see Chapter 3 for the full analysis of which variant is in use and what that implies for the remedy)

**What the text says (Call 2, line 33):**

> **Preceding op:** Same all_gather call (TODO: verify whether the method is shared via a base class or duplicated — see the note below)

**The problem:**

Domain fact 4 states: "Two confirmed call sites: TTNNQwen3FullAttention._maybe_all_gather and TTNNQwen3LinearAttention._maybe_all_gather — both have a synchronous `ttnn.all_gather` followed by `synchronize_device`."

The Call 1 "Preceding op" field frames the question as "which variant — `ttnn.all_gather` or `ttnn.experimental.all_gather_async`?" and defers with a TODO. But domain fact 4 does not leave this open: the preceding op IS `ttnn.all_gather` (synchronous), confirmed. Presenting `all_gather_async` as an equally likely current state contradicts this domain fact and will cause an engineer reading the field to invest time in verification that domain fact 4 has already resolved.

This error propagates into the Remedy field for both calls. Because the three-way branch correctly includes a B1 arm ("already `ttnn.experimental.all_gather_async`"), and because the Preceding op field leaves the current variant ambiguous, an engineer may treat all three branches as live options. But domain fact 4 rules out B1 for both Call 1 and Call 2 — the current op is confirmed synchronous, so B1 cannot apply to either call site as it currently exists. The live options are only Type A (synchronous intent) or Type B2 (async intent). Including B1 as a live option in the remedy field — when the current source is confirmed to have synchronous `ttnn.all_gather` — is misleading.

Note: The Remedy field's three-way branch is not wrong in isolation — it correctly describes the full classification space from the methodology. The error is that the Preceding op field fails to apply domain fact 4 to narrow that space for these specific call sites, and the Remedy field inherits that ambiguity rather than eliminating the inapplicable branch.

**Required correction:**

For Call 1, update "Preceding op" to reflect the confirmed fact:

> **Preceding op:** `ttnn.all_gather` (synchronous variant) — confirmed by domain context. `ttnn.experimental.all_gather_async` is not the current op at this call site.

For Call 2, update "Preceding op" similarly:

> **Preceding op:** `ttnn.all_gather` (synchronous variant) — same as Call 1 (TODO: verify whether the method is shared via a base class or duplicated).

For the Remedy field on both calls, the B1 arm should be marked inapplicable given the confirmed preceding op, and a note should clarify that the only open question is dispatch intent (Type A vs. Type B2):

> **Remedy:** The preceding op is confirmed synchronous `ttnn.all_gather`, so Type B1 does not apply. Two cases remain: (1) synchronous `ttnn.all_gather` and intent is synchronous dispatch → Type A (delete `synchronize_device` only); (2) synchronous `ttnn.all_gather` but intent is async dispatch → Type B2 (first replace `ttnn.all_gather` with `ttnn.experimental.all_gather_async`, then delete `synchronize_device` and add TT_CCL cycling semaphore management). See Chapter 3 and `audit_methodology.md` Section 3 Question 4 for the full verdict and exact code changes.

---

## Pass 4 Change Log

### Fix applied — 2026-04-23

**File changed:** `audit_results.md`, Section 1, Call 1 and Call 2 — "Preceding op" and "Remedy" fields.

**Change:** Applied two targeted corrections to each of the two confirmed call sites:

1. **Preceding op field (Call 1):** Replaced the ambiguous "ttnn.all_gather or ttnn.experimental.all_gather_async (TODO: verify which variant)" with a confirmed statement: synchronous `ttnn.all_gather` is the current op at this call site; `all_gather_async` is not. Removed the TODO deferral on the variant question.

2. **Preceding op field (Call 2):** Replaced "Same all_gather call (TODO: verify ...)" with the same confirmed statement as Call 1 (synchronous `ttnn.all_gather`, confirmed by domain context), retaining the separate TODO on whether the method is shared via a base class or duplicated (that question is still open).

3. **Remedy field (Call 1 and Call 2):** Replaced the three-way branch — which presented B1 ("already `all_gather_async`") as a live option alongside Type A and B2 — with a two-way branch that explicitly marks B1 inapplicable given the confirmed synchronous preceding op. The remedy now states: "Type B1 does not apply. Two cases remain: Type A (synchronous dispatch intent) or Type B2 (async dispatch intent)." The choice between A and B2 is noted as depending on dispatch intent, which remains a source-access TODO.

No other content in `audit_results.md` was modified.

---

## Pass 5

### Issues found: 1

**Issue 1:** `audit_methodology.md`, Section 3, Question 4, Type A definition — trigger condition is missing the "dispatch intent IS synchronous" qualifier required by domain fact 1.

**What the text says (line 84):**

> **Type A (delete only):** The preceding op is the synchronous variant of `ttnn.all_gather`. CQ0 ordering suffices to guarantee the gather result is ready before the next consumer; the `synchronize_device` call can be deleted without replacement. If the preceding op is a compute op (e.g., `ttnn.matmul`, `ttnn.softmax`) with no CCL dependency, the purpose of the synchronize call is unclear — classify as Type C and investigate before deleting.

**The problem:**

Domain fact 1 defines Type A with two conjunctive conditions: "Preceding op IS synchronous `ttnn.all_gather` AND dispatch intent IS synchronous → delete `synchronize_device` only."

The methodology's Type A trigger satisfies the first condition (preceding op is synchronous `ttnn.all_gather`) but is silent on the second condition (dispatch intent IS synchronous). As written, the Type A definition authorizes deletion whenever the preceding op is the synchronous `ttnn.all_gather`, regardless of dispatch intent. This is incorrect: when the preceding op is synchronous `ttnn.all_gather` but the dispatch intent is async, the correct remedy is Type B2 — not Type A. Type A and Type B2 share the same preceding-op state; it is dispatch intent alone that separates them.

The Pass 3 fix correctly narrowed the trigger from "any synchronous TTNN op" to "the synchronous variant of `ttnn.all_gather`", but it did not add the dispatch-intent gate. The omission is now directly observable: the Remedy fields in `audit_results.md` (updated in Pass 3 and Pass 4) correctly split the two cases by dispatch intent ("Type A if synchronous intent; Type B2 if async intent"), but the methodology's Type A definition still lacks that gate. An engineer who applies the methodology's Type A rule to a call site without consulting the Remedy fields could delete `synchronize_device` at a B2 site — leaving a synchronous `ttnn.all_gather` in place without the semaphore management that async dispatch requires.

**Precise correction:**

Add the dispatch-intent qualifier to the Type A trigger, and explicitly name Type B2 as the alternative when intent is async:

> **Type A (delete only):** The preceding op is the synchronous variant of `ttnn.all_gather` AND the intended dispatch model is synchronous. CQ0 ordering suffices to guarantee the gather result is ready before the next consumer; the `synchronize_device` call can be deleted without replacement. If the preceding op is the synchronous `ttnn.all_gather` but the intended dispatch model is async, do NOT apply Type A — apply Type B2 instead (see below). If the preceding op is a compute op (e.g., `ttnn.matmul`, `ttnn.softmax`) with no CCL dependency, the purpose of the synchronize call is unclear — classify as Type C and investigate before deleting.

---

## Pass 5 Change Log

### Fix applied — 2026-04-23

**File changed:** `audit_methodology.md`, Section 3 ("Classifying Each Call"), Question 4, Type A definition.

**Change:** Added the dispatch-intent qualifier as a second required condition for the Type A trigger, and added an explicit redirect to Type B2 when condition 1 is met but dispatch intent is async.

Specifically:
- **Condition 1 (existing):** The preceding op is the synchronous variant of `ttnn.all_gather`.
- **Condition 2 (added):** The intended dispatch model is synchronous.
- **Redirect (added):** If condition 1 is met but the intended dispatch model is async, do NOT apply Type A — apply Type B2 instead.

Before this fix, the Type A trigger required only condition 1. An engineer following the methodology could have applied Type A (delete `synchronize_device` only) to a call site where async dispatch was the intent, leaving a synchronous `ttnn.all_gather` in place without the TT_CCL cycling semaphore management that async dispatch requires — a silently broken async dispatch pattern. The added condition 2 and redirect ensure that when dispatch intent is async, engineers are explicitly redirected to Type B2 rather than erroneously applying a delete-only fix. No other content was modified.

---

## Pass 6

### Issues found: 0

None. Chapter is correct.

All seven checklist items verified against the domain facts:

1. **Type A trigger has both conditions** — `audit_methodology.md` line 84 reads "The preceding op is the synchronous variant of `ttnn.all_gather` AND the intended dispatch model is synchronous." Both required conditions are present.

2. **Type A has the B2 redirect when intent is async** — Same line continues: "If the preceding op is the synchronous `ttnn.all_gather` but the intended dispatch model is async, do NOT apply Type A — apply Type B2 instead (see below)." Redirect is explicit and correctly named.

3. **Type A excludes compute-op cases with redirect to Type C** — Same line continues: "If the preceding op is a compute op (e.g., `ttnn.matmul`, `ttnn.softmax`) with no CCL dependency, the purpose of the synchronize call is unclear — classify as Type C and investigate before deleting." Redirect present.

4. **Call 1 and Call 2 Preceding op fields confirm synchronous `ttnn.all_gather`** — Call 1 (`audit_results.md` line 19): "`ttnn.all_gather` (synchronous variant) — confirmed by domain context. `ttnn.experimental.all_gather_async` is not the current op at this call site." Call 2 (line 33): "`ttnn.all_gather` (synchronous variant) — same as Call 1, confirmed by domain context." Both confirmed without ambiguity.

5. **Call 1 and Call 2 Remedy fields show only A vs B2, with B1 marked inapplicable** — Both Remedy fields state: "The preceding op is confirmed synchronous `ttnn.all_gather`, so Type B1 does not apply. Two cases remain: (1) ... Type A ...; (2) ... Type B2 ..." B1 is explicitly excluded; only A and B2 are live.

6. **`_capture_trace` scoping correctly limited to pre-bracket with warning** — `audit_results.md` Section 2 second bullet correctly scopes the non-blocking claim to "only when they appear before `begin_trace_capture` (the warm-up compile run)" and includes the required warning: "any call that appears after `begin_trace_capture` in the same method — i.e., inside the actual capture bracket — is trace-blocking and must not be dismissed. Do not dismiss a grep hit in `_capture_trace` without first confirming which side of `begin_trace_capture` it falls on."

7. **No other errors** — `index.md` scope exclusions and summary table are consistent with the methodology and domain facts. `audit_methodology.md` Section 2 correctly lists `TracedRun._capture_trace` under calls that must be investigated (not blanket-dismissed). The Type B umbrella trigger's "is or should be" phrasing is imprecise but not factually incorrect, because the B1/B2 sub-case definitions below it are authoritative and correctly differentiate the two scenarios.
