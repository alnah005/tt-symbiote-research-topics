# Chapter 6 — End-to-End Integration and Test Strategy

This chapter consolidates the changes from Chapters 3 through 5 into a sequenced integration checklist and a concrete test plan. Chapter 3 defined the pre-allocation pattern; Chapter 4 verified that `ttnn.copy` is trace-safe; Chapter 5 confirmed that the warm-up guard remains effective. This chapter answers the practical question: in what order do you make these changes, how do you verify them step by step, and what tests give you confidence that the full decode trace path is correct before merging?

---

## Scope: Decode Trace Path Only

This chapter covers the decode trace path exclusively. Prefill has a different cos/sin lifecycle — sequence-length-varying tensors that cannot be pre-allocated to a fixed shape — and is out of scope here. See [`prefill_scope_note.md`](prefill_scope_note.md) for a brief explanation of why prefill is deferred and what the recommended interim approach is.

---

## Prerequisites

- Chapter 3 — Pre-allocation plan for `_cos_replicated` and `_sin_replicated`
- Chapter 4 — `ttnn.copy` trace safety analysis
- Chapter 5 — Warm-up guard preservation

---

## What's Next

Read the following files in order:

1. [`integration_checklist.md`](integration_checklist.md) — Pre-conditions to verify before starting, implementation steps in order, and post-implementation checks.
2. [`test_plan.md`](test_plan.md) — Five concrete tests covering correctness, trace safety, multi-step replay, guard preservation, and end-to-end smoke testing.
3. [`prefill_scope_note.md`](prefill_scope_note.md) — Why prefill cos/sin pre-allocation is deferred and what to do in the interim.
