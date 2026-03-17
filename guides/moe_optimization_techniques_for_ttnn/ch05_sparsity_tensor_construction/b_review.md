# Agent B Review — Chapter 5: Sparsity Tensor Construction — Pass 1

1. **`index.md`, Notation table, rho column value — wrong numeric value**

   The table defines rho as "fraction of tile rows **skipped**" and lists the Qwen3.5-35B decode value as `≈ 0.875`. This is incorrect.

   For B=1, S=1 decode: k=8, E=256, N=8 devices, E_d=32 local experts. Each token selects k=8 experts globally. Distributed across N=8 devices, on average 1 expert per device is active, so 1 out of E_d=32 tile rows is active. The fraction of tile rows **skipped** is therefore 31/32 ≈ **0.969**, not 0.875.

   0.875 = 7/8 has no basis in the Qwen3.5-35B parameters (it would require E_d=8, not E_d=32).

   The tip immediately below the table correctly derives 1 − 1/32 = 0.969 but the table entry must be corrected to match.

   Correction: change `≈ 0.875` to `≈ 0.969` in the rho row of the notation table.

2. **`index.md`, Notation table, rho definition — inconsistency with key facts**

   The table defines rho as the **skip** fraction (`1 - active/total`), giving the Qwen3.5-35B decode value as ≈ 0.969 (after fix above). The key facts define rho as `active_experts / E_d` (the **active** fraction ≈ 0.031). The chapter's definition should be stated unambiguously and either definition is acceptable, but the numeric example in the table and the tip below it are written as the skip fraction. The definition column currently reads "fraction of tile rows skipped by `sparse_matmul`", which is self-consistent with the 0.969 corrected value — no additional change needed beyond fixing the numeric value in issue 1, provided the definition column is understood to mean skip fraction. If the intent is to match the key-facts convention (rho = active fraction), both the definition and the value must change to ≈ 0.031.

3. **`common_pitfalls.md`, P6 table, row B=32, S=1 — C value wrong**

   The P6 table lists B=32, S=1 as producing C=2, M_t=1. The correct computation is:

   C = ceil(k × B × S / E) = ceil(8 × 32 × 1 / 256) = ceil(256/256) = ceil(1) = **1**, not 2.

   With C=1, M_t = ceil(1/32) = 1, and the mask shape remains [32, 224] — consistent with B=1 and B=8 rows in the same table. The value C=2 is incorrect and contradicts the key facts (B=32, S=1: C=1).

   Correction: change the B=32, S=1 row from `C=2, M_t=1` to `C=1, M_t=1`.

---

# Agent A Fix — Chapter 5: Sparsity Tensor Construction — Pass 1 Fixes Applied

1. `index.md` notation table — Fixed ρ definition and value: changed skip-fraction convention (1 - active/total) to active-fraction convention (active_experts / E_d), consistent with Chapter 4. Updated example value from 0.875 to ~0.031 for B=1 decode.

2. `index.md` Tip — Updated to use active-fraction ρ ≈ 0.031; now reads "1 - ρ = 31/32 ≈ 0.969 of tile rows are skipped."

3. `common_pitfalls.md` P6 table, B=32 row — Corrected C from 2 to 1 (correct: ceil(8×32×1/256) = ceil(1) = 1). M_t remains 1. Prose "C > 1 when B > 32" was already correct and unchanged.

---

# Agent B Review — Chapter 5: Sparsity Tensor Construction — Pass 2

## Fix verification

All three Pass 1 fixes are correctly applied:

1. `index.md` notation table, ρ row — Definition now reads "fraction of expert slots (and tile rows) that are active: ρ = active_experts / E_d (consistent with Chapter 4 convention). Fraction of tile rows skipped = 1 − ρ." Value reads "≈ 0.031 for B=1 decode." Both are correct.

2. `index.md` Tip — Now reads "ρ ≈ 0.031: only 1 of the 32 local experts is active on average. That means 1 − ρ = 31/32 ≈ 0.969 of tile rows are skipped." Correct.

3. `common_pitfalls.md` P6 table, B=32, S=1 row — Now reads C=1, M_t=1, mask shape [32, 224]. Correct (ceil(8×32×1/256) = 1).

## Remaining issue

**`index.md`, Quick-Reference Checklist, item 1 — incorrect formula for `mask.shape[0]`**

The checklist currently reads:

> `mask.shape[0] == ceil(E_d * C / 32)`

This formula is mathematically wrong. The correct mask `shape[0]` is `E_d * M_t = E_d * ceil(C / 32)`.

`ceil(E_d * C / 32)` and `E_d * ceil(C / 32)` are not equal in general. Specifically, for the standard decode case (C=1, E_d=32):

- Formula as written: `ceil(32 × 1 / 32) = ceil(1) = 1`
- Correct value: `32 × ceil(1 / 32) = 32 × 1 = 32`

The formula gives 1 instead of 32 — a factor-of-32 error for the most common operating point. A developer using this checklist to validate their mask would incorrectly accept a mask of shape [1, K_t] when the actual required shape is [32, 224].

The two expressions are equal only when C is an exact multiple of 32 (e.g., C=256: `ceil(32×256/32)=256` vs `32×ceil(256/32)=32×8=256`). For any non-multiple-of-32 value of C (including C=1), they diverge.

**Correction:** Change the checklist item 1 check from:

```
mask.shape[0] == ceil(E_d * C / 32)
```

to:

```
mask.shape[0] == E_d * ceil(C / 32)
```

This is consistent with the notation table definition `M_t = ceil(C/32)` and the mask shape `[E_d × M_t, K_t]` stated throughout the chapter.

---

# Agent A Fix — Chapter 5: Sparsity Tensor Construction — Pass 2 Fixes Applied

4. `index.md` checklist item 1 — Fixed formula: `ceil(E_d * C / 32)` → `E_d * ceil(C / 32)`. For C=1, E_d=32: wrong formula gives ceil(32/32)=1; correct formula gives 32×1=32. The two only coincidentally agree when C is a multiple of 32.

---

# Agent B Review — Chapter 5: Sparsity Tensor Construction — Pass 3

## Fix 4 verification

`index.md`, Quick-Reference Checklist, item 1 now reads:

> `mask.shape[0] == E_d * ceil(C / 32)` and `mask.shape[1] == ceil(H / 32)`

This is correct. For C=1, E_d=32: `32 * ceil(1/32) = 32 * 1 = 32`. The fix matches the notation table definition `M_t = ceil(C/32)` and the mask shape `[E_d × M_t, K_t]` stated throughout the chapter.

## Full chapter review

All numeric values, formulas, and cross-references verified against the key facts (E=256, k=8, N=8, H=7168, E_d=32, K_t=224, C=ceil(k×B×S/E), M_t=ceil(C/32)):

- `index.md` notation table: M_t examples (1 for decode, 8 for prefill B=4,S=2048), K_t=224, C examples (1 for B=1,S=1; 256 for B=4,S=2048), ρ definition and ≈0.031 value, tip deriving 1−ρ≈0.969 — all correct.
- `sparsity_tensor_format.md` Section 4 tile counts: decode 1×7=7 tiles=7168 bytes≈7 KB; prefill ceil(256/32)×7=8×7=56 tiles=57344 bytes≈56 KB — all correct.
- `constructing_from_router_output.md` shape traces: decode `assignment_padded` [1024] = 32×1×32; prefill [8192] = 32×8×32 — both correct. Code logic uses `> 0` (not `== 32`) for tile-active check, consistent with P2 fix guidance.
- `sparsity_tensor_placement.md` Section 2 table: B=32,S=1 row shows C=1,M_t=1,7 tiles (consistent with P6 fix); B=4,S=2048 row shows C=256,M_t=8,56 tiles — all correct.
- `common_pitfalls.md` P6 table: B=32,S=1 row shows C=1,M_t=1,[32,224] (verified as ceil(8×32×1/256)=1) — correct.

No remaining issues found.

No feedback — chapter approved

---

# Agent A Fix — Chapter 5: Sparsity Tensor Construction — Compression Pass 1 Fixes Applied

C1. `constructing_from_router_output.md` lines 216–217 — Removed duplicate mask-size comments from inside the `build_sparsity_tensor` function (`# Decode size: 7 tiles × 1024 bytes = 7168 bytes in L1` and `# Prefill size: 56 tiles × 1024 bytes = 57344 bytes in L1`). The authoritative derivation remains in `sparsity_tensor_format.md` Section 4.

C2. `constructing_from_router_output.md` Step 5 — Replaced the 8-line standalone `ttnn.from_torch(...)` code block (lines 94–102) with a one-sentence inline reference pointing to `sparsity_tensor_placement.md`, Section 3. The canonical snippet in `sparsity_tensor_placement.md` Section 3 is preserved; the P4 fix snippet in `common_pitfalls.md` (with its distinguishing `# not ttnn.DRAM_MEMORY_CONFIG` comment) is preserved. The Tip following Step 5 is preserved unchanged.

---

# Agent B Review — Chapter 5: Sparsity Tensor Construction — Pass 4

## Verification of prior fixes (Passes 1–3 and Compression Pass 1)

**Fix 1** (`index.md` notation table, ρ definition and value): ρ is defined as `active_experts / E_d` with the note "Fraction of tile rows skipped = 1 − ρ", and the Qwen3.5-35B decode value reads "≈ 0.031 for B=1 decode". Correct.

**Fix 2** (`index.md` Tip): Reads "ρ ≈ 0.031: only 1 of the 32 local experts is active on average. That means 1 − ρ = 31/32 ≈ 0.969 of tile rows are skipped." Correct.

**Fix 3** (`common_pitfalls.md` P6 table, B=32, S=1 row): Shows C=1, M_t=1, mask shape [32, 224]. Correct (ceil(8×32×1/256) = 1).

**Fix 4** (`index.md` Quick-Reference Checklist item 1): Reads `mask.shape[0] == E_d * ceil(C / 32)` and `mask.shape[1] == ceil(H / 32)`. Correct.

**C1** (`constructing_from_router_output.md`): The two duplicate mask-size comments (`# Decode size: 7 tiles × 1024 bytes ...` and `# Prefill size: 56 tiles × 1024 bytes ...`) are absent from the file. Fix confirmed.

**C2** (`constructing_from_router_output.md` Step 5): Step 5 now contains a one-sentence inline reference to `sparsity_tensor_placement.md` Section 3 rather than a standalone `ttnn.from_torch` code block. The canonical snippet remains in `sparsity_tensor_placement.md` Section 3. The P4 snippet in `common_pitfalls.md` with the distinguishing comment is intact. The Tip is unchanged. Fix confirmed.

## Full chapter review

All numeric values, formulas, shape traces, and code logic were verified against the key facts (E=256, k=8, N=8, E_d=32, H=7168, K_t=224, C=ceil(k×B×S/E), M_t=ceil(C/32)):

- `index.md` checklist formula `mask.shape[0] == E_d * ceil(C / 32)` — correct.
- `sparsity_tensor_format.md` Section 1: mask shape `[M_t, K_t]` for single-expert, `[E_d × M_t, K_t]` for batched — correct. Section 4 tile counts: decode 1×7=7 tiles=7168 bytes≈7 KB; prefill ceil(256/32)×7=8×7=56 tiles=57344 bytes≈56 KB — correct. Section 7 partial-tile boundary formula verified as correct.
- `constructing_from_router_output.md` shape traces: decode `assignment [32,1]`, `assignment_padded [1024]` (=32×1×32), `tile_rows [32,32]`, `tile_active [32]`, `mask_torch [32,224]` — all correct. Prefill `assignment [32,256]`, `assignment_padded [8192]` (=32×8×32), `tile_rows [256,32]`, `tile_active [256]`, `mask_torch [256,224]` — all correct. Tile-active check uses `> 0`, consistent with P2. C computation examples (C=256 for B=4,S=2048; C=1 for B=1,S=1) — correct.
- `sparsity_tensor_placement.md` Section 2 table: all three rows (B=1,S=1: C=1,M_t=1,7 tiles,7 KB; B=32,S=1: C=1,M_t=1,7 tiles,7 KB; B=4,S=2048: C=256,M_t=8,56 tiles,56 KB) — correct. Section 8 placement table: decode mask shape [32,224], 7 KB; prefill [256,224], 56 KB — correct.
- `common_pitfalls.md` P6 table: B=32,S=1 row shows C=1,M_t=1,[32,224] — correct. Prose threshold "C > 1 when B > 32" verified: ceil(8×33/256)=ceil(1.03125)=2 — correct.

No remaining issues found.

No feedback — chapter approved
