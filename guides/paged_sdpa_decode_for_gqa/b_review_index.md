# B Review — Guide Index — Pass 1

## Verdict
3 error(s) found.

---

### Error 1
- **File:** `index.md`
- **Line:** 113–114
- **Stated:** "`cur_pos[i]` is the number of valid tokens already in the KV cache for batch element `i` **after** the current `paged_update_cache` write."
- **Correct:** Per the authoritative facts, `cur_pos[i]` = 5 when 5 tokens are *already* cached — i.e., it is the token count *before* the current write (equivalently, the 0-indexed position of the next token to be written). The word "after" inverts the timing: `cur_pos` is an input to `paged_update_cache`, not a value computed from its output. The correct framing is: `cur_pos[i]` is the current context length / number of valid tokens already in the KV cache *before* writing the current decode-step token.

---

### Error 2
- **File:** `index.md`
- **Line:** 115–116
- **Stated:** "It is **not** the index of the token being written."
- **Correct:** Per the authoritative facts, `cur_pos[i]` is explicitly defined as "the 0-indexed position of the NEXT token to be written." That position *is* the index of the token being written in the current decode step. The negation on line 115–116 directly contradicts this. (The plan's `per_user_vs_shared.md` description of "Common mistake 2" refers to confusing the *pre-write* index with the *post-write* length, not to denying that `cur_pos` is the write index altogether.)

---

### Error 3
- **File:** `index.md`
- **Line:** 177
- **Stated:** "`cur_pos[i]` | Post-write KV length for batch element `i`"
- **Correct:** The authoritative facts define `cur_pos[i]` as the current context length — the number of valid tokens already cached *before* the current write step (equivalently, the index at which the next token will be written). Describing it as "post-write" is incorrect: `cur_pos=5` means 5 tokens are already in the cache *before* writing token 5, not after. The terminology table entry should read something like "current context length (number of valid tokens in the KV cache before the current decode write)" to match the authoritative definition.

## Agent A Change Log — B Feedback Pass 1
- index.md: Fixed cur_pos definition: removed "after paged_update_cache write"; cur_pos is pre-write current context length
- index.md: Fixed cur_pos description: it IS the 0-indexed next-write position, not "not the index of token being written"
- index.md terminology table: Fixed cur_pos[i] from "Post-write KV length" to "Current context length (0-indexed next-write position)"

# B Review — Guide Index — Pass 2

## Pass 1 Fix Verification

**Fix 1 — Verified.** Lines 113–117 now read: "`cur_pos[i]` is the current context length — the number of valid tokens already in the KV cache for batch element `i` before the current decode write." The phrase "after the current `paged_update_cache` write" has been removed. The text correctly frames `cur_pos` as an input to `paged_update_cache`, not a result of it.

**Fix 2 — Verified.** The sentence "It is **not** the index of the token being written" has been removed. The replacement text ("It is the 0-indexed position of the next token to be written: for a sequence with 5 tokens already cached, `cur_pos[i] = 5` and the next write goes at position 5") correctly identifies `cur_pos[i]` as the 0-indexed write position, consistent with the authoritative facts.

**Fix 3 — Verified.** The terminology table entry for `cur_pos[i]` (line 179) now reads "Current context length (0-indexed next-write position); for a seq with N tokens cached, cur_pos[i]=N". The incorrect "Post-write KV length" label has been removed.

## Verdict

1 remaining error found.

**Error — File:** `index.md`, **Line:** 52 (Chapter Navigation table, Ch4 row description)
- **Stated:** "Defines `cur_pos[i]` precisely (post-write KV length, not the write index)"
- **Correct:** Per the authoritative facts, `cur_pos[i]` is the *pre-write* current context length and *is* the 0-indexed write index (not "post-write KV length, not the write index"). This parenthetical is the same incorrect framing corrected in Fix 1 and Fix 3, but it was not updated in the Chapter Navigation table. It should read something like: "Defines `cur_pos[i]` precisely (pre-write current context length; equals the 0-indexed position of the next token to be written)".

## Agent A Change Log — B Feedback Pass 2
- index.md navigation table Ch4 row: Fixed cur_pos parenthetical from "post-write KV length, not the write index" to correct pre-write context length description

# B Review — Guide Index — Pass 3

## Pass 2 Fix Verification

**Fix verified.** Line 52, Ch4 row of the Chapter Navigation table now reads: "Defines `cur_pos[i]` precisely (pre-write current context length; equals the 0-indexed position of the next token to be written)". The incorrect parenthetical "post-write KV length, not the write index" has been removed. This matches the authoritative definition: `cur_pos[i]` is the number of valid tokens already cached before the current decode write, which is simultaneously the 0-indexed position at which the next token will be written. The fix is consistent with the corrections applied in Pass 1 (lines 113–117 body text and terminology table entry at line 179).

## Verdict

No feedback — guide index approved.
