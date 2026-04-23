# Compression Analysis — Chapters 5 and 6: Model Config Audit and Recommendations

## Pass 1

### Crucial issues found: 1

#### Issue 1 — Near-verbatim `assert head_dim % 64 == 0` block duplicated between `recommended_fix.md` and `precondition_policy.md`

**Files involved:**
- `ch6_recommendations/recommended_fix.md` (lines 137–142 and lines 32–36 in `__init__` code block)
- `ch6_recommendations/precondition_policy.md` (lines 23–29)

**Quoted near-verbatim block from `recommended_fix.md` (standalone precondition section):**
```python
assert head_dim % 64 == 0, (
    f"head_dim={head_dim} must be a multiple of 64. "
    "ttnn.experimental.rotary_embedding requires the input last dimension "
    "to be divisible by TILE_WIDTH * 2 = 64."
)
```

**Quoted near-verbatim block from `precondition_policy.md` (Hard Requirements section):**
```python
assert head_dim % 64 == 0, (
    f"head_dim={head_dim} must be a multiple of 64. "
    "ttnn.experimental.rotary_embedding requires the input last dimension "
    "to satisfy head_dim % (TILE_WIDTH * 2) == 0."
)
```

The two blocks are functionally identical (same assertion, same f-string structure, same rationale). They differ only in the final clause of the error message string ("to be divisible by TILE_WIDTH * 2 = 64." vs. "to satisfy head_dim % (TILE_WIDTH * 2) == 0."). Both files also contain the `assert rotary_dim % 2 == 0` and `assert rotary_dim <= head_dim` assertions written independently, compounding the duplication.

Additionally, the `rotary_dim % 32 != 0` warning block is written out in full in both files (6–8 lines each), with minor wording differences. `recommended_fix.md` already says "see `precondition_policy.md` for full rationale" in a comment — confirming that `precondition_policy.md` is the canonical location.

**Fix needed:** `recommended_fix.md` should not reproduce the full assertion and warning code. Replace the standalone "Precondition to Add" section in `recommended_fix.md` (the block starting with "Add the following assertion in `__init__`") with a one-line cross-reference: "Add the preconditions specified in [`precondition_policy.md`](./precondition_policy.md)." The `__init__` code listing inside `recommended_fix.md` may retain the assertions as inline illustration of the complete constructor, but the separate "Precondition to Add" section that restates them in isolation is the redundant element and should be collapsed to a reference.

---

### VERDICT

Crucial updates: yes

**What must be fixed and in which file:**

`ch6_recommendations/recommended_fix.md` — The standalone "Precondition to Add" section (the freestanding `assert head_dim % 64 == 0` block and surrounding prose that re-explains the same constraint) duplicates `precondition_policy.md` without adding new information. Collapse that section to a cross-reference to `precondition_policy.md`, which is already the designated canonical source for precondition policy. The full `__init__` code listing earlier in the file may keep the assertions as part of the complete implementation example, but should not re-derive or re-explain them in a separate prose section.

---

## Pass 2

### Verification of Pass 1 fix

**Fix (recommended_fix.md — "Precondition to Add" section collapsed to cross-reference):** APPLIED

Current "Precondition to Add" text (lines 132–134 of `recommended_fix.md`):

> See [`precondition_policy.md`](./precondition_policy.md) for the complete specification of required assertions and warnings. In brief: assert `head_dim % 64 == 0`, `rotary_dim % 2 == 0`, and `rotary_dim <= head_dim`; emit a warning (not an error) when `rotary_dim % 32 != 0`. Do NOT add an `assert rotary_dim % 32 == 0` or `assert rotary_dim % 64 == 0` guard — Strategy C makes such a guard unnecessary.

The standalone freestanding code block is gone. The section is now a single cross-reference sentence (with a brief inline summary that adds context without reproducing the full assertion blocks). The `__init__` code listing earlier in the file retains the assertions as part of the complete implementation example, which is appropriate.

### Crucial issues found: 0

No additional crucial issues were found. The remaining files checked (ch5 `index.md`, `is_this_dead_code.md`, ch6 `index.md`, `verification_checklist.md`) contain no 5+ near-verbatim line blocks duplicated from another file. The `verification_checklist.md` PyTorch reference function and test scaffolding are original to that file. The `is_this_dead_code.md` prose references Strategy C and Strategy B by name and links to the canonical sources rather than reproducing their content.

---

### VERDICT

Crucial updates: no

Chapters 5 and 6 approved.
