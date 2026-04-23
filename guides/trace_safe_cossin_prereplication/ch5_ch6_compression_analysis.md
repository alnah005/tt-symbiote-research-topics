# Compression Analysis — Chapters 5 and 6: Warm-Up Guard Preservation and Integration

## Pass 1

### Crucial issues found: 1

**Issue 1 — Near-verbatim `ttnn.zeros` pre-allocation block duplicated across Chapter 5 and Chapter 6**

The canonical `ttnn.zeros` pre-allocation code for `_cos_replicated` appears as a near-verbatim 7-line block in two files:

- `ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md` (lines 22–33, "Tile-Aligned Case" section) — presents it as the canonical tile-aligned pattern for context, within a discussion of layout constraints.
- `ch6_integration_and_testing/integration_checklist.md` (lines 32–47, "Step 1" section) — repeats the same block (extended to include `_sin_replicated`) as the implementation instruction.

The seven near-verbatim lines shared between the two files are:

```python
self._cos_replicated = ttnn.zeros(
    shape=[1, 1, 1, self.rotary_dim],
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=self.mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
)
```

The duplication serves no purpose: a reader implementing Step 1 of `integration_checklist.md` is instructed by `ch6/index.md` to have already read Chapters 3–5, including `non_tile_aligned_rotary_dim_interaction.md`. The checklist re-stating the full block verbatim adds no new information; it only creates a maintenance hazard (any future change to the pattern must be made in two places).

**Fix:** In `integration_checklist.md` Step 1, replace the inline `ttnn.zeros` block for `_cos_replicated` with a cross-reference to the canonical pattern defined in `non_tile_aligned_rotary_dim_interaction.md`. The `_sin_replicated` allocation (which has no counterpart in ch5) may remain inline, or a single note may state that the same pattern is applied for both buffers. For example:

> Use the canonical `ttnn.zeros` pattern from Chapter 5 (`non_tile_aligned_rotary_dim_interaction.md`, "Tile-Aligned Case" section) for both `_cos_replicated` and `_sin_replicated`, substituting `self.rotary_dim` for the shape's last dimension.

---

### VERDICT

Crucial updates: yes

- `ch6_integration_and_testing/integration_checklist.md` Step 1: remove the verbatim `ttnn.zeros` block for `_cos_replicated` and replace it with a cross-reference to the canonical pattern in `ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md`. The `_sin_replicated` block may be kept inline (it has no ch5 counterpart) or collapsed into a single note covering both buffers.

---

## Pass 2

### Verification of Pass 1 fix

**Fix (integration_checklist.md — cross-reference added for _cos_replicated canonical pattern):** APPLIED

The current text of `integration_checklist.md` Step 1 (lines 23–33) reads:

> Add the two pre-allocations immediately after the `_decode_cur_pos` pre-allocation. Both buffers use the canonical `ttnn.zeros` pattern from Chapter 5 ([`../ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md`](../ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md), "Tile-Aligned Case") — same shape, dtype, layout, memory config, and mapper. If the parameters listed here ever diverge from that section, the Chapter 5 version is authoritative.

And the code block opens with:

```python
# Canonical source: ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md,
# "Tile-Aligned Case".
self._cos_replicated = ttnn.zeros(
    ...
)
self._sin_replicated = ttnn.zeros(
    shape=[1, 1, 1, self.rotary_dim],  # identical parameters to _cos_replicated above
    ...
)
```

Both the prose cross-reference (with authoritative-source note) and the in-code `# Canonical source:` comment are present. The `_sin_replicated` block carries `# identical parameters to _cos_replicated above`. All three elements of the fix are confirmed applied.

### Crucial issues found: 0

Two candidate duplications were examined and rejected:

**Candidate A — `if is_decode:` branching block in `integration_checklist.md` Step 5 vs. `prefill_scope_note.md`.**

Both files show the `if is_decode: ttnn.copy(...) / else: _ensure_replicated(...)` pattern. The blocks share the `if is_decode:` line and the two `ttnn.copy` lines, but the local variable names differ (`cos_out`/`sin_out` in the checklist vs. `cos_for_rotary`/`sin_for_rotary` in the prefill note), and the else-branch comment in the prefill note is explanatory prose not present in the checklist. The two files serve distinct purposes: the checklist gives the implementation instruction and the prefill note explains why prefill is deferred and shows the interim pattern. Neither should simply reference the other; readers of the prefill note need the pattern inline to follow the recommendation. Below the CRUCIAL threshold.

**Candidate B — post-allocation assertion loop in `guard_adequacy_after_change.md` vs. `integration_checklist.md` Step 2.**

Structurally similar (both use `ttnn.get_device_tensors` + `assert shape[-1] == rotary_dim`) but not near-verbatim: the checklist uses a combined outer loop over both buffers with a `name` variable, while `guard_adequacy_after_change.md` shows a single-buffer loop with different variable names and a different error message. Below the CRUCIAL threshold.

---

### VERDICT

Crucial updates: no

Chapters 5 and 6 approved.
