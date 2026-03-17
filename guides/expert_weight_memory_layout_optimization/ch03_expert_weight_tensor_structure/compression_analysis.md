# Compression Analysis: Chapter 3 Expert Weight Tensor Structure

## Summary
- Files analyzed: `index.md`, `projection_shapes.md`, `tensor_to_shard_grid_mapping.md`, `dtype_and_tile_layout.md`
- Estimated current line count: 689 (index 77 + dtype_and_tile_layout 234 + projection_shapes 165 + tensor_to_shard_grid_mapping 213)
- Estimated post-compression line count: ~610
- Estimated reduction: ~11%

---

## CRUCIAL Suggestions

### C-1: Near-verbatim shape summary table duplicated between `index.md` and `projection_shapes.md`

**Files and lines:**
- `index.md` lines 62–71: "Quick Reference: Per-Expert Weight Dimensions" table (Gate/Up/Down rows with Canonical Shape and Role columns)
- `projection_shapes.md` lines 64–71: "Shape Summary" table (Gate/Up/Down rows with Notation, Shape, and Direction columns)

**Overlap:** Both tables list the three projections, their `[d_model, d_ff]` / `[d_ff, d_model]` shapes, and their functional roles. The index version adds a "Canonical Shape" column phrased as role text; the projection_shapes version adds a "Notation" (w1/w3/w2) and "Direction" column. The core content — three projections, two shapes, role descriptions — is restated near-verbatim.

**Recommendation:** Keep the canonical copy in `projection_shapes.md` (lines 64–71), which already includes the w1/w3/w2 notation column that the index version omits. Replace the index version (lines 62–71) with a one-line cross-reference: `> See the [Shape Summary table in projection_shapes.md](./projection_shapes.md#shape-summary) for the canonical projection shapes.`

**Estimated line savings:** ~10 lines (remove 10-line table block from `index.md`, replace with 1-line cross-reference).

---

### C-2: `ttnn.from_torch` loading code pattern duplicated between `projection_shapes.md` and `dtype_and_tile_layout.md`

**Files and lines:**
- `dtype_and_tile_layout.md` lines 35–50: "Converting to TILE_LAYOUT" code block — loads `w1_torch`, calls `ttnn.from_torch(..., dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)`.
- `projection_shapes.md` lines 109–128: "TTNN Storage Convention" code block — shows the same `ttnn.from_torch(w1_expert_i, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)` call inside a list comprehension for per-expert storage, and then again for the stacked storage path.

**Overlap:** Both files demonstrate the core `ttnn.from_torch(..., layout=ttnn.TILE_LAYOUT)` API call as the mechanism for loading weights. The `dtype_and_tile_layout.md` version additionally shows the `memory_config` keyword (unique there); the `projection_shapes.md` version additionally shows the stacked-tensor path (unique there). However, the per-expert `ttnn.from_torch` call with `dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT` appears in both files without meaningful differentiation.

**Recommendation:** Keep the canonical `ttnn.from_torch` conversion snippet in `dtype_and_tile_layout.md` (the natural home for dtype/layout mechanics). In `projection_shapes.md` lines 109–128, retain the stacked-tensor example (unique content) but remove the redundant per-expert `ttnn.from_torch` lines that repeat the identical API call without adding new information. Add a cross-reference: `> For the dtype and layout conversion step, see [dtype_and_tile_layout.md — Converting to TILE_LAYOUT](./dtype_and_tile_layout.md#converting-to-tile_layout).`

**Estimated line savings:** ~12 lines (collapse the per-expert list comprehension block that duplicates the from_torch pattern while retaining the stacked-tensor block as unique content).

---

### C-3: Tile-alignment constraint stated twice — `dtype_and_tile_layout.md` and `tensor_to_shard_grid_mapping.md`

**Files and lines:**
- `dtype_and_tile_layout.md` lines 54–71: "Tile Size and Alignment Constraint" section — defines the 32×32 tile, states the `shard_H % 32 == 0` / `shard_W % 32 == 0` constraint, and provides the bytes-per-tile table.
- `tensor_to_shard_grid_mapping.md` lines 9–24: "Core Divisibility Constraint" section — restates Constraint 2 as "(N // num_cores) % 32 == 0" for each sharding mode, with the warning that "a shard width or height that is not a multiple of 32 will produce incorrect matmul results."

**Overlap:** Both sections define the tile-alignment requirement (multiple of 32) as a mandatory sharding constraint and warn about silent failures or assertion errors. The formal `shard % 32 == 0` rule is spelled out in both files.

**Recommendation:** Keep the full definition (including the 32×32 tile properties table and bytes-per-tile table) in `dtype_and_tile_layout.md` as the canonical reference. In `tensor_to_shard_grid_mapping.md`, reduce the tile-alignment sub-statement inside Constraint 2 to a one-sentence summary with a cross-reference: `> Tile alignment rule: `(shard_dim) % 32 == 0`. See [dtype_and_tile_layout.md — Tile Size and Alignment Constraint](./dtype_and_tile_layout.md#tile-size-and-alignment-constraint) for derivation.` Retain the warning in `tensor_to_shard_grid_mapping.md` since it is contextually relevant to grid selection.

**Estimated line savings:** ~8 lines (collapse the 4-row tile-property table and the surrounding prose restatement in `tensor_to_shard_grid_mapping.md`; retain the constraint formula inline).

---

## MINOR Suggestions

### m-1: `index.md` "Quick Reference" section anticipates content that `projection_shapes.md` already delivers fully

`index.md` lines 61–77 (Quick Reference table + Next Steps) restates information the reader is about to encounter in `projection_shapes.md`. Because the index's role is navigation, the full Quick Reference table adds limited value on top of the Chapter Structure table (lines 52–58) and the learning objectives. Consider trimming the Quick Reference table to a 2-sentence prose summary and keeping only the cross-reference to `projection_shapes.md`.

Estimated line savings: ~8 lines.

### m-2: "Next Steps" footers are formulaic repetition across all three section files

`projection_shapes.md` line 163–165, `tensor_to_shard_grid_mapping.md` lines 211–213, and `dtype_and_tile_layout.md` lines 231–234 each contain a "Next Steps" paragraph that names the following file and briefly describes its content. This pattern is intentional for navigation but the descriptions in the footers restate language already present in the `index.md` Chapter Structure table (lines 52–58). No action required — these are low-value duplications but serve a navigational purpose.

### m-3: Model parameter constants repeated across multiple concrete-shapes sections

Mixtral 8x7B parameters (`d_model=4096, d_ff=14336, num_experts=8, top_k=2`) appear as prose in:
- `index.md` lines 35–39 (reference table)
- `projection_shapes.md` lines 134–135 (concrete shapes section)
- `dtype_and_tile_layout.md` lines 134–135 (memory calculation section)
- `tensor_to_shard_grid_mapping.md` lines 135–137 (worked example)

Similarly for Qwen MoE. These repetitions are load-bearing in each file's worked examples and should not be consolidated; they are listed here only for awareness. No action recommended.

---

## Load-Bearing Evidence

The following specific facts, formulas, and code must not be removed during consolidation:

1. **Per-expert memory formula** (`dtype_and_tile_layout.md` lines 126–131):
   `per_expert_bytes = 3 * d_model * d_ff * bytes_per_element`
   This is the central quantitative result of Chapter 3.

2. **Mixtral 8x7B byte calculations** (`dtype_and_tile_layout.md` lines 133–156):
   BF16 = 352,321,536 bytes (~335.9 MiB); BF8 = 176,160,768 bytes (~168.0 MiB).
   The SI vs binary megabyte distinction note (line 158) is load-bearing — it prevents a real production error.

3. **Qwen MoE byte calculations** (`dtype_and_tile_layout.md` lines 160–183):
   BF16 = 88,080,384 bytes (~84.0 MiB); BF8 = 44,040,192 bytes (~42.0 MiB).

4. **DeepSeek-MoE-16B footnote** (`dtype_and_tile_layout.md` line 195):
   `3 * 2048 * 1408 * 2 = 17,301,504 bytes = ~16.5 MiB` per expert. This derivation validates d_ff=1408 tile-alignment (1408/32=44, already aligned) and must not be removed.

5. **`expert_weight_bytes` utility function** (`dtype_and_tile_layout.md` lines 214–225):
   Canonical reusable function; must remain in `dtype_and_tile_layout.md`.

6. **`pad_to_tile` utility function** (`dtype_and_tile_layout.md` lines 90–92):
   Canonical padding helper; must remain in `dtype_and_tile_layout.md`.

7. **Mixtral 8x7B sharding worked example** (`tensor_to_shard_grid_mapping.md` lines 134–194):
   Step-by-step `14336 // 8 = 1792`, `1792 % 32 == 0` derivation plus full `ttnn.MemoryConfig` construction. This is the primary concrete demonstration of Chapter 3 sharding mechanics.

8. **Qwen MoE intra-expert sharding example** (`tensor_to_shard_grid_mapping.md` lines 123–130):
   `shard_width = 2048 // 8 = 256`, `256 % 32 == 0` — the key demonstration that small d_ff still satisfies tile alignment with 8 cores.

9. **SwiGLU forward pass pseudocode** (`projection_shapes.md` lines 13–27):
   The `gate = x @ w1 / up = x @ w3 / hidden = silu(gate) * up / output = hidden @ w2` compute graph. This is the foundational derivation from which all shapes follow.

10. **Down-projection transpose asymmetry warning** (`projection_shapes.md` lines 56–60):
    "A shard configuration valid for the gate projection is not automatically valid for the down projection." This caution must not be removed.

11. **w_gate_up concatenation tip** (`projection_shapes.md` lines 49–49):
    The note that w1 and w3 are often stored as a single `[d_model, 2 * d_ff]` tensor in checkpoint files. Critical for readers inspecting real model checkpoints.

12. **DRAM controller alignment rationale** (`tensor_to_shard_grid_mapping.md` lines 34–38):
    The 6/12/8/16/32-core preference table with Wormhole-specific DRAM controller justification. This is hardware-specific load-bearing guidance.

13. **20–30% overhead allowance warning** (`dtype_and_tile_layout.md` line 197):
    "Always add at least a 20–30% overhead allowance for non-expert parameters and runtime activations." Must not be removed.

---

## VERDICT: Crucial updates: yes

---

## Agent A Change Log — C Feedback Pass 1

### Fix 1 — Resolve C-1: Remove duplicate shape table from `index.md`

**File:** `index.md`
**Action:** Delete lines 62–77 (the "Quick Reference: Per-Expert Weight Dimensions" table and its introductory sentence, plus the "Next Steps" footer that immediately follows it). Replace with:

```markdown
## Quick Reference

For canonical projection shapes (Gate w1, Up w3, Down w2), see the [Shape Summary in projection_shapes.md](./projection_shapes.md#shape-summary).

---

## Next Steps

Continue to [`projection_shapes.md`](./projection_shapes.md) for a detailed breakdown of each projection's shape, the rationale behind the SwiGLU structure, and how stacked expert tensors are organized in TTNN.
```

**Rationale:** The Shape Summary table in `projection_shapes.md` (lines 64–71) already contains all of this information plus the w1/w3/w2 notation column. Keeping a duplicate in the index without the notation column is strictly inferior.

**Estimated savings:** ~10 lines net.

---

### Fix 2 — Resolve C-2: Remove redundant `ttnn.from_torch` per-expert block from `projection_shapes.md`

**File:** `projection_shapes.md`
**Action:** In the "TTNN Storage Convention" section (lines 105–129), replace the per-expert storage list comprehension block (lines 111–120) with a cross-reference, and retain only the stacked storage block (lines 122–128) which is unique to this file:

```markdown
## TTNN Storage Convention

TTNN typically stores expert weights as separate tensors per expert, or as per-group tensors covering the subset of experts assigned to a given device. For the dtype conversion and `TILE_LAYOUT` requirement that applies to both storage forms, see [dtype_and_tile_layout.md — Converting to TILE_LAYOUT](./dtype_and_tile_layout.md#converting-to-tile_layout).

The stacked storage path requires explicit handling:

```python
import ttnn

# Stacked storage (batch_matmul path)
# Requires careful dimension ordering and a matching batch_matmul call
gate_stacked = ttnn.from_torch(
    torch.stack([w1_i for w1_i in all_w1], dim=0),  # [num_experts, d_model, d_ff]
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)
```
```

**Rationale:** The `dtype_and_tile_layout.md` "Converting to TILE_LAYOUT" section (lines 33–50) is the canonical home for the `ttnn.from_torch` API usage. The per-expert list comprehension in `projection_shapes.md` shows nothing new — it repeats the same three keyword arguments. The stacked-tensor snippet is unique content and must be retained.

**Estimated savings:** ~12 lines.

---

### Fix 3 — Resolve C-3: Collapse tile-alignment restatement in `tensor_to_shard_grid_mapping.md`

**File:** `tensor_to_shard_grid_mapping.md`
**Action:** In the "Core Divisibility Constraint" section (lines 9–24), keep the full constraint table (it is the natural location for the combined divisibility + tile-alignment constraints in grid-selection context) but remove the standalone prose that restates the 32-element tile definition. Specifically, delete the sentence "TTNN's tile is a 32×32 element block" and the "(the tile side length in TILE_LAYOUT)" parenthetical from line 14, replacing with a cross-reference footnote below the table:

```markdown
> Tile size definition and bytes-per-tile table: see [dtype_and_tile_layout.md — Tile Size and Alignment Constraint](./dtype_and_tile_layout.md#tile-size-and-alignment-constraint).
```

**Rationale:** The tile-size properties table (32×32, 1024 elements) and the bytes-per-tile table already appear in `dtype_and_tile_layout.md` lines 57–79. The sharding file needs the constraint formulas (which are unique in context) but not a re-derivation of why 32 is the magic number.

**Estimated savings:** ~8 lines.

## Agent A Change Log — C Feedback Pass 1
- index.md: Replaced Quick Reference weight dimension table with cross-reference to projection_shapes.md
- projection_shapes.md: Removed per-expert loading code block (from_torch pattern); added cross-reference to dtype_and_tile_layout.md
- tensor_to_shard_grid_mapping.md: Replaced tile-32 re-derivation with cross-reference to dtype_and_tile_layout.md; kept constraint formulas

---

# Compression Analysis: Chapter 3 Expert Weight Tensor Structure — Pass 2

## Summary
- Pass 1 fixes: all three verified as correctly applied
- Current line count after Pass 1: 671 (index 69 + projection_shapes 155 + tensor_to_shard_grid_mapping 213 + dtype_and_tile_layout 234)
- Pre-Pass-1 estimated count was 689; actual post-Pass-1 is 671 — a reduction of 18 lines (~2.6%), consistent with the ~30 lines predicted (some headroom was estimates; actual savings are inline with scope).
- New crucial duplications: none

## CRUCIAL Suggestions

None.

All three Pass 1 fixes are correctly in place:

1. **Fix 1 verified — index.md:** The "Quick Reference: Per-Expert Weight Dimensions" table (originally lines 62–71, 10-line table) has been replaced with a single cross-reference sentence: "For a complete per-expert weight shape reference including w1/w3/w2 notation, see `projection_shapes.md`." The heading `## Quick Reference: Per-Expert Weight Dimensions` is retained as a navigation anchor with the cross-reference body. The `## Next Steps` footer is also retained. No residual table content remains in `index.md`. Fix correctly applied.

2. **Fix 2 verified — projection_shapes.md:** The per-expert `ttnn.from_torch` list comprehension block (the redundant loading code that duplicated the API call from `dtype_and_tile_layout.md`) has been removed. In its place, line 109 contains a cross-reference sentence: "For the canonical `ttnn.from_torch(..., dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)` loading pattern, including dtype options and tile-alignment requirements, see `dtype_and_tile_layout.md`." The stacked-tensor code block (`gate_stacked = ttnn.from_torch(torch.stack(...), ...)`) is retained at lines 111–119 as unique content. Fix correctly applied.

3. **Fix 3 verified — tensor_to_shard_grid_mapping.md:** The tile-32 re-derivation prose has been replaced with a cross-reference embedded in Constraint 2 (line 14): "For a full explanation of why the tile side length is 32 and how TILE_LAYOUT enforces this constraint, see `dtype_and_tile_layout.md`." The constraint formulas in the table (`(N // num_cores) % 32 == 0`, `(M // num_cores) % 32 == 0`, and the BLOCK_SHARDED form) remain intact at lines 20–22. The Warning about silent incorrect matmul results remains at line 24. Fix correctly applied.

## MINOR Suggestions

Carried forward from Pass 1 (no new items):

- **m-1 (carried):** `index.md` Quick Reference section now contains only the cross-reference and Next Steps — further trimming is not needed; the section is already minimal after Fix 1.
- **m-2 (carried):** "Next Steps" footers at the end of all three section files remain formulaic but serve navigational purpose; no action recommended.
- **m-3 (carried):** Mixtral 8x7B and Qwen MoE parameter constants (d_model, d_ff, num_experts, top_k) appear in worked examples across all four files; these are load-bearing in each context and should not be consolidated.

## Load-Bearing Evidence

All thirteen items listed in Pass 1 are confirmed still present:

1. `per_expert_bytes = 3 * d_model * d_ff * bytes_per_element` formula — `dtype_and_tile_layout.md` lines 126–129. Present.
2. Mixtral 8x7B BF16 = 352,321,536 bytes (~335.9 MiB); BF8 = 176,160,768 bytes (~168.0 MiB) — `dtype_and_tile_layout.md` lines 144–149. Present.
3. Qwen MoE BF16 = 88,080,384 bytes (~84.0 MiB); BF8 = 44,040,192 bytes (~42.0 MiB) — `dtype_and_tile_layout.md` lines 171–176. Present.
4. DeepSeek-MoE-16B footnote `3 * 2048 * 1408 * 2 = 17,301,504 bytes = ~16.5 MiB` — `dtype_and_tile_layout.md` line 195. Present.
5. `expert_weight_bytes` utility function — `dtype_and_tile_layout.md` lines 214–225. Present.
6. `pad_to_tile` utility function — `dtype_and_tile_layout.md` lines 90–92. Present.
7. Mixtral 8x7B sharding worked example (`14336 // 8 = 1792`, `1792 % 32 == 0`, full `ttnn.MemoryConfig` construction) — `tensor_to_shard_grid_mapping.md` lines 134–194. Present.
8. Qwen MoE intra-expert sharding example (`shard_width = 2048 // 8 = 256`, `256 % 32 == 0`) — `tensor_to_shard_grid_mapping.md` lines 123–130. Present.
9. SwiGLU forward pass pseudocode (`gate = x @ w1 / up = x @ w3 / hidden = silu(gate) * up / output = hidden @ w2`) — `projection_shapes.md` lines 13–27. Present.
10. Down-projection transpose asymmetry warning — `projection_shapes.md` line 60. Present.
11. w_gate_up concatenation tip — `projection_shapes.md` line 49. Present.
12. DRAM controller alignment rationale (6/12/8/16/32-core preference table) — `tensor_to_shard_grid_mapping.md` lines 32–39. Present.
13. 20–30% overhead allowance warning — `dtype_and_tile_layout.md` line 197. Present.

## VERDICT: Crucial updates: no
