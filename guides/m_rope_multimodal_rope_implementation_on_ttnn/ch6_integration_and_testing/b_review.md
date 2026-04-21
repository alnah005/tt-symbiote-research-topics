## B Feedback — Pass 1

Two crucial errors found:

1. **`integration_steps.md` `_gather_sections` — column slicing off by factor of 2**

   The cos/sin tables have shape `[max_seq_len, rotary_dim]` = `[max_seq_len, 64]` (established in Ch5: 8 MiB = 2 tables × 32768 × 64 × 2 bytes). The `mrope_section = [11, 11, 10]` values are *pair counts*, summing to 32 = `rotary_dim // 2`. Each section of `s_i` pairs spans `2*s_i` actual columns in the 64-column table (consistent with HuggingFace `apply_multimodal_rotary_pos_emb` which multiplies `mrope_section * 2` before splitting). The comment in `_mrope_forward` correctly states the boundaries:

   ```
   # Temporal: columns [0:2*s_t]
   # Height:   columns [2*s_t:2*(s_t+s_h)]
   # Width:    columns [2*(s_t+s_h):rotary_dim]
   ```

   But `_gather_sections` implements:
   ```python
   table_t = table[:, 0:s_t]           # [max_seq_len, 11]  ← WRONG, should be 22
   table_h = table[:, s_t:s_t+s_h]     # [max_seq_len, 11]  ← WRONG, should be 22
   table_w = table[:, s_t+s_h:]        # [max_seq_len, 10]  ← WRONG, should be 20
   ```

   The concatenated result has shape `[batch*seq_len, 32]`, not `[batch*seq_len, 64]`, causing a shape mismatch in the downstream `rotate_half` multiply. Fix:
   ```python
   table_t = table[:, 0:2*s_t]
   table_h = table[:, 2*s_t:2*(s_t+s_h)]
   table_w = table[:, 2*(s_t+s_h):]
   ```

2. **`correctness_validation.md` per-axis structural invariant checks — wrong column ranges (consequence of error 1)**

   All structural invariant column slice references assume the 32-column (incorrect) layout rather than the 64-column (correct) layout. Specific fixes:

   - Test Case 3 temporal: `cos_assembled[..., 0:11]` → `cos_assembled[..., 0:22]`
   - Test Case 3 height: `cos_assembled[..., 11:22]` → `cos_assembled[..., 22:44]`
   - Test Case 3 width: `cos_assembled[..., 22:32]` → `cos_assembled[..., 44:64]`
   - Test Case 4 temporal: `cos_assembled[..., 0:11]` → `cos_assembled[..., 0:22]`
   - Test Case 4 height+width cross-frame: `cos_assembled[..., 11:32]` → `cos_assembled[..., 22:64]`

## B Feedback Application Log — Pass 1

- Fix 1: Corrected `_gather_sections` column slicing from `[0:s_t, s_t:s_t+s_h, s_t+s_h:]` to `[0:2*s_t, 2*s_t:2*(s_t+s_h), 2*(s_t+s_h):]` in `integration_steps.md`; updated inline dimension comments to `[max_seq_len, 2*s_t]` etc. and final concat comment to `[batch*seq_len, rotary_dim=64]`.
- Fix 2: Corrected structural invariant column ranges in `correctness_validation.md`: temporal `0:11→0:22`, height `11:22→22:44`, width `22:32→44:64` (Test 3); temporal `0:11→0:22` (Test 4); height+width cross-frame `11:32→22:64` (Test 4); updated reshape dims to match (`11→22`, `10→20`).

---

## B Feedback — Pass 2

No feedback — chapter approved.

Verified after fixes:
- `_gather_sections` column boundaries: `[0:22, 22:44, 44:64]` for s_t=11, s_h=11, s_w=10; concatenated output is 22+22+20=64 columns = rotary_dim ✓
- `2*s_t + 2*s_h + 2*s_w = 22+22+20 = 64 = rotary_dim` — invariant holds ✓
- `sum(mrope_section) == rotary_dim//2` assertion: `11+11+10=32=64//2` ✓
- Test 3 structural invariants: temporal `[0:22]`, height `[22:44]`, width `[44:64]` — consistent with fixed code ✓
- Test 4 cross-frame height+width check: `[22:64]` covers both height and width sections ✓
- Step 5 CPU validation uses `assert_close(rtol=0, atol=0)` (exact identity on float32 CPU) — correct for this level ✓
- Step 6 PCC > 0.9999 threshold — justified by cross-layer error accumulation argument in `integration_steps.md` and `correctness_validation.md` ✓
- Metal Trace: `[3, batch, 1]` position tensor updated in-place before each decode replay — consistent with `cur_pos_tensor` pattern ✓
- Program cache hit rate 100% at decode (fixed shapes), +3 misses per new seq_len at prefill — consistent with Ch5 analysis ✓
