# Compression Analysis: Chapter 4 cur_pos Semantics

## Summary
- Files analyzed: `index.md`, `cur_pos_definition.md`, `per_user_vs_shared.md`, `cur_pos_in_paged_mode.md`
- Estimated current line count: 57 + 101 + 131 + 124 = 413 lines
- Estimated post-compression line count: ~395 lines
- Estimated reduction: ~4%

---

## CRUCIAL Suggestions

None.

No code block, table, derivation, or explanation appears verbatim or near-verbatim in two or more files. Each file is well-scoped to its own concern:

- `cur_pos_definition.md` owns the step-table, two passing modes, and the `-1` sentinel in full.
- `per_user_vs_shared.md` owns Common Mistakes 1–3 and the side-by-side correct/incorrect example.
- `cur_pos_in_paged_mode.md` owns `num_active_blocks`, block-boundary behavior, and Issue #30362.
- `index.md` is a navigation hub; its Quick Reference bullets are intentional one-liner pointers, not canonical definitions.

---

## MINOR Suggestions

### M-1: `index.md` Quick Reference partially restates `cur_pos_definition.md` opening

- **Files**: `index.md` lines 5–9, `cur_pos_definition.md` lines 5–10
- **What overlaps**: Both state that `cur_pos[i]` = length of valid prefix, `cur_pos[i] = 1` after first decode step, and `-1` skips computation.
- **Recommendation**: The Quick Reference is appropriate as a navigation summary. Add a trailing pointer "(full definition: `cur_pos_definition.md`)" to make the intent explicit and discourage future editors from expanding it into a true duplicate. No lines need to be deleted.
- **Estimated savings**: 0 lines deleted; clarity improved.

### M-2: `-1` sentinel mentioned in three files at different depths

- **Files**: `index.md` line 9 (one phrase), `cur_pos_definition.md` lines 76–100 (full section with code), `per_user_vs_shared.md` lines 10–17 (used as part of a per-user batch table example)
- **What overlaps**: All three note that `-1` means "skip this batch slot". The depths are very different (phrase vs. full section vs. illustrative use), so no content is redundant.
- **Recommendation**: No deletion needed. Consider adding a cross-reference in `per_user_vs_shared.md` line 12 area: "(See `cur_pos_definition.md` § The -1 Sentinel for full semantics.)" to reduce any future temptation to copy the full section there.
- **Estimated savings**: 0 lines deleted; clarity improved.

### M-3: Causal masking role mentioned briefly in three files

- **Files**: `index.md` lines 50–56 (why `cur_pos` matters), `cur_pos_definition.md` lines 9–10 (kernel enforces causal mask), `cur_pos_in_paged_mode.md` line 8 (causal masking bullet)
- **What overlaps**: Each mention is a single sentence or short paragraph providing local context for a different topic. There is no paragraph or code block that is duplicated.
- **Recommendation**: Leave as-is. Each mention is load-bearing context for its own section. Removing any of them would leave the surrounding explanation unmotivated.
- **Estimated savings**: 0 lines.

---

## Load-Bearing Evidence

The following facts must not be removed from the chapter under any future compression pass:

1. `cur_pos[i]` = length of valid KV prefix (not the token index being written) — `cur_pos_definition.md` lines 5–7.
2. After the first decode step: `cur_pos[i] = 1` (not 0) — `cur_pos_definition.md` lines 15–17 (step table).
3. Special value `-1`: skips all computation for that batch index; output at that index is undefined — `cur_pos_definition.md` lines 77–80.
4. Common Mistake 2: off-by-one using the write-index instead of the post-write cache length — `per_user_vs_shared.md` lines 40–77 (full section with code).
5. `num_active_blocks[i] = ceil(cur_pos[i] / block_size)` — `cur_pos_in_paged_mode.md` line 15.
6. Issue #30362: sporadic PCC failures; validate at dense `cur_pos` coverage — `cur_pos_in_paged_mode.md` lines 82–123.

---

## VERDICT: Crucial updates: no
