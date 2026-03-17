# Compression Analysis — Chapter 2: TTNN MeshDevice API — Pass 1

## Summary
- Files reviewed: `index.md`, `mesh_device_setup.md`, `tensor_distribution.md`, `collective_primitives.md`
- Current line count: ~995 lines (total across all files)
- Estimated post-compression line count: ~830 lines (~17% reduction)

---

## CRUCIAL Suggestions

**C1 — `cluster_axis=1` explanation repeated four times across files**

The same explanation — "`cluster_axis=1` traverses the column axis; for a `(1, 8)` T3K mesh this is the correct value because all eight devices are on a single row" — appears verbatim or near-verbatim in:

- `index.md` lines 25 and 85–86 (Prerequisites block and Key Terminology block)
- `mesh_device_setup.md` lines 42 and 163 (mesh_shape parameter section and Incorrect Mesh Shape pitfall)
- `collective_primitives.md` lines 51–53 (`cluster_axis` parameter section)

The full explanation belongs once in `collective_primitives.md` where `cluster_axis` is a formal parameter. Every other occurrence should be reduced to a one-sentence reminder with a cross-reference. Cutting three of the four full explanations saves approximately 10–12 lines.

**C2 — `ttnn.Topology.Linear` vs. `Topology.Ring` warning duplicated**

The rationale "T3K is a linear path, not a ring; device 7 and device 0 are not connected; do not use `Topology.Ring`" appears fully stated in:

- `index.md` line 27 (Prerequisites bullet)
- `mesh_device_setup.md` lines 42–43 (inside the mesh_shape parameter discussion)
- `collective_primitives.md` line 79 (`topology` parameter section for `all_to_all`)

The complete warning should live only in `collective_primitives.md` where `topology` is formally introduced. The `index.md` prerequisite bullet can keep a one-sentence pointer. The `mesh_device_setup.md` paragraph conflates mesh shape correctness with topology selection — these are distinct concerns and the topology sentence in `mesh_device_setup.md` should be cut entirely. Saves approximately 6–8 lines.

**C3 — `ttnn.to_device` / `ReplicateTensorToMesh` double-warning in `tensor_distribution.md`**

The warning "do not use `ttnn.to_device` for mesh-wide replication; use `ttnn.from_torch` with `ReplicateTensorToMesh` instead" is stated in full three times within the same file:

- Lines 222–234: inline code comment block (nine lines of comments)
- Lines 248–249: the `> **Warning**` block immediately after the code block
- Lines 250–251: the prose paragraph that follows the Warning block

All three make the same point. The inline comment should be reduced to two lines (the key "do not use" instruction and the correct alternative). The `> **Warning**` callout block is the right vehicle — keep it. The follow-on prose paragraph (lines 250–251) restates what the Warning block already says and should be deleted. Saves approximately 10–12 lines.

---

## Load-Bearing Evidence

- **`index.md`** — The "TTNN Multi-Device Concepts Introduced in This Chapter" table (lines 45–59) is load-bearing: it maps each concept to the file where it first appears AND to its first downstream use in later chapters. This cross-chapter forward reference map is not duplicated anywhere else and must not be shortened or merged with the simpler "Reading Order" list below it.

- **`mesh_device_setup.md`** — The five-step initialization sequence (lines 88–98) is load-bearing. Each step names a specific internal action and the failure mode it surfaces. This ordered walkthrough of what `open_mesh_device` does internally is the only place in the chapter where the device-opening internals are explained, and it is necessary for diagnosing initialization failures described in the Common Pitfalls section.

- **`tensor_distribution.md`** — The combined constraint explanation at lines 272–275 ("the divided dimension must be divisible by `num_devices × 32 = 256`") is load-bearing. The distinction between the naive `% 8 == 0` check and the correct `% 256 == 0` check is a non-obvious correctness point that is not stated anywhere else in the chapter and has direct consequences for padding logic.

- **`collective_primitives.md`** — The "Relationship to Reduce-Scatter" section (lines 233–235) is load-bearing. The nuance that `reduce_scatter` + `all_gather` is only beneficial if the all-gather can be *deferred or eliminated* — and that if it always follows immediately, `all_reduce` is simpler and equally efficient — is a decision-making insight not restated elsewhere and is necessary for the Chapter 5 composition discussion.

---

## MINOR Suggestions

**M1 — `index.md` "Relationship to Chapter 1" section (lines 33–38) is verbose**

The paragraph restates four facts from Chapter 1 that are already enumerated in the Prerequisites bullet list directly above it (lines 22–29). The two-paragraph "Relationship to Chapter 1" section can be reduced to two sentences: one stating the mapping intent and one naming the key translation (`(1, 8)` → constructor argument, `cluster_axis=1` → collective parameter, `num_links` → collective parameter). Saves approximately 5 lines.

**M2 — Over-long comment block in `collective_primitives.md` `all_to_all` example**

The code comment in lines 84–86 of `collective_primitives.md` states "Pre-condition: tokens have been sorted into groups by target device. dispatch_buffer has shape ... dispatch_buffer is a sharded mesh tensor on mesh_device." The shape and type of `dispatch_buffer` are already explicit in the variable name and in the formal parameter documentation immediately above. Reduce to one comment line: `# dispatch_buffer: (8, tokens_per_device, 4096), pre-sorted by target device.` Saves 2 lines.

**M3 — `mesh_device_setup.md` "Device Ordering Conventions" section repeats `device_ids` parameter prose**

The three bullet points in "Device Ordering Conventions" (lines 74–80) largely restate the same information as the `device_ids` parameter section directly above it (lines 44–52): both explain that position N maps to `(row=0, col=N)`, both warn about incorrect ordering, and both note the standard ordering aligns logical col distance with physical hop distance. The "Conventions" section should be collapsed into a short summary paragraph at the end of the `device_ids` parameter section, eliminating the standalone subsection header and cutting the repetition. Saves approximately 8 lines.

**M4 — Hedging qualifier in `collective_primitives.md` line 288**

"Note: the exact API for selecting queue IDs per operation may differ between TTNN versions. Consult the current TTNN documentation for the `queue_id` parameter status." — this sentence adds no actionable information beyond "things might change." If it is a known limitation worth flagging, a single inline parenthetical is sufficient: `(queue_id parameter availability is version-dependent; verify against current TTNN docs)`. The two-sentence paragraph can become one clause. Saves 1 line.

**M5 — `collective_primitives.md` "Default Behavior Without Explicit Synchronization" subsection (lines 304–307) restates the intro to the Synchronization section**

Lines 244–249 already establish the default async behavior and explain that sequential pipelines do not require explicit synchronization. The dedicated "Default Behavior Without Explicit Synchronization" subsection at the end of the synchronization block repeats these same points with no additional content. The subsection can be deleted; its one novel sentence ("The only time you must explicitly synchronize is when you need the CPU to observe a result") can be appended as a final sentence to the opening paragraph of the Synchronization section. Saves approximately 4 lines.

---

VERDICT: Crucial updates: yes

---

## Change Log — Pass 1 Fixes Applied

- C1 applied: `cluster_axis=1` full explanation in `index.md` (lines ~25, ~85-86) and `mesh_device_setup.md` (lines ~42, ~163) compressed to one-sentence reminder + cross-reference to `collective_primitives.md`. Authoritative explanation in `collective_primitives.md` unchanged.
- C2 applied: `Topology.Ring` warning removed from `mesh_device_setup.md` (topology is a separate concern from mesh shape). `index.md` prerequisite bullet shortened to one-sentence pointer. Authoritative warning in `collective_primitives.md` unchanged.
- C3 applied: Triple `ttnn.to_device` replication warning in `tensor_distribution.md` compressed: inline comments reduced to 2 lines, `> Warning` callout retained, redundant follow-on prose paragraph deleted.

---

# Compression Analysis — Chapter 2: TTNN MeshDevice API — Pass 2

## Summary
- Files reviewed: `index.md`, `mesh_device_setup.md`, `tensor_distribution.md`, `collective_primitives.md`
- Current line count: ~982 lines (total across all four files: 87 + 229 + 264 + 402)
- Estimated post-compression line count: ~960 lines (~2% reduction; major wins from Pass 1 already applied)

## Pass 1 Item Verification

**C1 — `cluster_axis=1` explanation compressed to one-sentence + cross-reference:** ADDRESSED. `index.md` line 25 and line 85 each read a single-sentence pointer to `collective_primitives.md`. `mesh_device_setup.md` line 42 and line 163 each carry a one-sentence reminder + pointer. The authoritative multi-sentence explanation in `collective_primitives.md` lines 51–53 is unchanged.

**C2 — `Topology.Ring` warning removed from `mesh_device_setup.md`; `index.md` shortened to pointer:** ADDRESSED. `mesh_device_setup.md` mesh_shape section (line 42) contains no topology warning. `index.md` line 27 is a one-sentence pointer to `collective_primitives.md`. The full rationale in `collective_primitives.md` line 79 is unchanged.

**C3 — Triple `ttnn.to_device` warning compressed:** ADDRESSED. `tensor_distribution.md` lines 222–223 contain a two-line inline comment. The `> Warning` callout at line 237 is retained. The redundant follow-on prose paragraph is gone; line 239 opens `### Retrieving Results with ttnn.from_device` directly after the Warning block.

## CRUCIAL Suggestions

**C1 — `index.md` "Relationship to Chapter 1" section (lines 33–38) restates Prerequisites block above it**

The two-paragraph "Relationship to Chapter 1" section (lines 35–37) restates four facts that are enumerated individually in the Prerequisites bullets at lines 22–29 in the same file: (1) eight-chip linear row, (2) `(1, 8)` mesh shape → constructor argument, (3) `cluster_axis=1` → collective parameter, (4) `num_links` → collective signature. Both blocks are on the same page. The "Relationship" section adds no new information; its only role is to describe the translation as a mapping narrative rather than a list. Replacing the two-paragraph section with two sentences — one stating the mapping intent and one naming the key translations — saves approximately 5 lines without losing any content not already present in the Prerequisites block.

## Load-Bearing Evidence

- **`index.md`** — The "TTNN Multi-Device Concepts Introduced in This Chapter" table (lines 45–59) is load-bearing: it maps each concept to both its introduction file and its first use outside Chapter 2. The forward references to specific chapters (Ch 3, Ch 4, Ch 5, Ch 6, Ch 7) are not duplicated anywhere else in the chapter and must not be shortened.

- **`mesh_device_setup.md`** — The five-step initialization sequence (lines 88–98) is load-bearing. Each step names a specific internal action and the failure mode that step can surface. This ordered walkthrough is the only place in the chapter where the internal behavior of `ttnn.open_mesh_device` is explained, and it is referenced directly by the "Insufficient L1" and "Stale Device State" pitfall sections that follow.

- **`tensor_distribution.md`** — The combined constraint paragraph at lines 260–262 is load-bearing. The distinction between the insufficient `% 8 == 0` check and the correct `% 256 == 0` check, plus the separate statement that non-divided dimensions must also be multiples of 32, are both non-obvious correctness requirements not restated anywhere else in the chapter.

- **`collective_primitives.md`** — The "Relationship to Reduce-Scatter" section (lines 231–235) is load-bearing. The conditional framing — that the two-phase approach is only beneficial when the all-gather can be deferred or eliminated, and that if always required immediately it provides no advantage over direct `all_reduce` — is a decision-making insight not restated elsewhere in the chapter.

## MINOR Suggestions

**M1 (from Pass 1, not yet applied) — `collective_primitives.md` example comment block at lines 84–86 still has three lines**

The three-comment block before the MoE dispatch example still reads:
```
# Pre-condition: tokens have been sorted into groups by target device.
# dispatch_buffer has shape (num_devices=8, tokens_per_device, hidden_dim=4096).
# dispatch_buffer is a sharded mesh tensor on mesh_device.
```
The third line ("dispatch_buffer is a sharded mesh tensor on mesh_device") is obvious from the function call on line 88 and from the formal parameter description immediately above. Collapsing to two lines saves 1 line and reduces noise in the example.

**M2 (from Pass 1, not yet applied) — `collective_primitives.md` hedging qualifier at line 288**

The two-sentence paragraph "Note: the exact API for selecting queue IDs per operation may differ between TTNN versions. Consult the current TTNN documentation for the `queue_id` parameter status." adds no actionable information beyond "things might change." A single inline parenthetical after the code block is sufficient. Saves 1 line.

**M3 (from Pass 1, not yet applied) — `collective_primitives.md` "Default Behavior Without Explicit Synchronization" subsection (lines 304–306) restates the Synchronization intro**

The subsection header plus single-paragraph body at lines 304–306 makes the same three points as lines 244–249 above (async dispatch, sequential dependencies handled via tensor handles, no explicit sync needed). The one novel clause — "The only time you must explicitly synchronize is when you need the CPU to observe a result" — could be appended as a final sentence to the opening paragraph of the Synchronization section. Removing the standalone subsection saves approximately 3 lines.

**M4 — `mesh_device_setup.md` "Device Ordering Conventions" subsection (lines 72–80) overlaps `device_ids` parameter section (lines 44–52)**

Both sections explain that position N in `device_ids` maps to `(row=0, col=N)`, that logical column distance equals physical hop distance, and that non-standard ordering requires downstream updates. Collapsing the Conventions subsection into a summary paragraph appended to the `device_ids` parameter section, eliminating the standalone `##` header, saves approximately 6–8 lines and eliminates the structural repetition.

VERDICT: Crucial updates: yes

---

# Compression Analysis — Chapter 2: TTNN MeshDevice API — Pass 3

## Summary
- Files reviewed: `index.md`, `mesh_device_setup.md`, `tensor_distribution.md`, `collective_primitives.md`
- Current line count: ~984 lines (total across all four files: 86 + 230 + 265 + 403)
- Estimated post-compression line count: ~967 lines (~2% reduction; all major wins captured in prior passes, remaining items are minor)

## Pass 2 Item Verification

**C1 — `index.md` "Relationship to Chapter 1" section compressed:** PARTIALLY ADDRESSED. The original two-paragraph section has been reduced to a single sentence: "This chapter applies the Chapter 1 topology knowledge directly to TTNN API patterns — everything in this chapter assumes the T3K hardware model from Chapter 1." Pass 2 called for two sentences. The content reduction is adequate, but the section header `## Relationship to Chapter 1` plus surrounding `---` horizontal rules persist as dedicated structural scaffolding for one sentence. That scaffolding consumes more space than the content it wraps. See C1 below for the residual treatment.

**M1 — `collective_primitives.md` example comment block (lines 84–86) still three lines:** NOT ADDRESSED. All three comment lines remain verbatim: `# Pre-condition: tokens have been sorted into groups by target device.`, `# dispatch_buffer has shape (num_devices=8, tokens_per_device, hidden_dim=4096).`, `# dispatch_buffer is a sharded mesh tensor on mesh_device.` The third line is still present.

**M2 — `collective_primitives.md` hedging qualifier at line 288:** NOT ADDRESSED. The two-sentence paragraph "Note: the exact API for selecting queue IDs per operation may differ between TTNN versions. Consult the current TTNN documentation for the `queue_id` parameter status." remains as a standalone paragraph.

**M3 — `collective_primitives.md` "Default Behavior Without Explicit Synchronization" subsection (lines 304–307):** NOT ADDRESSED. The subsection header `### Default Behavior Without Explicit Synchronization` and its single-paragraph body remain. The paragraph's points are already covered at lines 243–249 above; the one novel clause has not been moved.

**M4 — `mesh_device_setup.md` "Device Ordering Conventions" subsection (lines 72–82) overlaps `device_ids` parameter section:** NOT ADDRESSED. The standalone `## Device Ordering Conventions` section header and all three sub-bullets remain separate from the `device_ids` parameter section above. The content overlap is unchanged.

## CRUCIAL Suggestions

**C1 — `index.md` lines 33–37: section header + dividers now outweigh the single-sentence content**

The "Relationship to Chapter 1" section currently consists of one sentence of content (`index.md` line 35) wrapped in a `## Relationship to Chapter 1` header (line 33), a blank line, and a closing `---` divider (line 37). That is 5 lines of structural scaffolding for 1 sentence of content. The sentence itself ("This chapter applies the Chapter 1 topology knowledge directly to TTNN API patterns — everything in this chapter assumes the T3K hardware model from Chapter 1.") is fully implied by the Prerequisites block directly above it (lines 22–29), which already states "Read Chapter 1 before reading this chapter" and lists the five specific concepts required. The entire section can be deleted; if the mapping-intent sentence is considered valuable, it can be appended as a final sentence to the Prerequisites block before the closing "If any of those concepts are unfamiliar" sentence. Net saving: 5 lines. No load-bearing content is lost — the Prerequisites block already establishes the dependency.

## Load-Bearing Evidence

- **`index.md`** — The "TTNN Multi-Device Concepts Introduced in This Chapter" table (lines 43–58) is load-bearing. Each row maps a concept to both its introducing file and its first use outside Chapter 2, with specific chapter numbers (Ch 3 through Ch 7). These forward references are not duplicated elsewhere and establish the chapter's role in the guide's dependency chain.

- **`mesh_device_setup.md`** — The five-step initialization sequence (lines 88–98) is load-bearing. Each step names a specific internal action (`Device enumeration and validation`, `Device opening`, `Mesh firmware programming`, `Worker thread creation`, `Memory context initialization`) and the failure mode that step surfaces. This walkthrough is the only place in the chapter where `open_mesh_device` internals are explained and is referenced by the `Insufficient L1` and `Stale Device State` pitfall sections.

- **`tensor_distribution.md`** — The combined-constraint paragraph at lines 259–262 is load-bearing. The explicit derivation that the correct padding check is `dim % 256 == 0` (not `dim % 8 == 0`), plus the separate statement that non-divided dimensions must also be multiples of 32 for `TILE_LAYOUT`, are non-obvious correctness requirements with direct padding-logic consequences and are not restated anywhere else in the chapter.

- **`collective_primitives.md`** — The conditional framing in the "Relationship to Reduce-Scatter" section (lines 231–235) is load-bearing. The specific claim that `reduce_scatter` + `all_gather` is only beneficial when the all-gather can be deferred or eliminated — and that if the all-gather is always required immediately the two-phase approach has the same total communication cost and provides no memory advantage over direct `all_reduce` — is a decision-making distinction not restated elsewhere.

## MINOR Suggestions

**M1 (carry-forward from Pass 2, not yet applied) — `collective_primitives.md` line 86: third comment line is redundant**

Line 86 reads `# dispatch_buffer is a sharded mesh tensor on mesh_device.` This is evident from the call on line 88 (`ttnn.all_to_all(dispatch_buffer, ..., mesh_device=mesh_device, ...)`) and from the formal `input_tensor` parameter description immediately above the code block. Deleting line 86 saves 1 line.

**M2 (carry-forward from Pass 2, not yet applied) — `collective_primitives.md` line 288: hedging paragraph is two sentences where one clause suffices**

"Note: the exact API for selecting queue IDs per operation may differ between TTNN versions. Consult the current TTNN documentation for the `queue_id` parameter status." Replace with a single inline parenthetical appended to the sentence preceding the code block or to the code block's last comment: `(queue_id availability is version-dependent; verify against current TTNN docs)`. Saves 1 line.

**M3 (carry-forward from Pass 2, not yet applied) — `collective_primitives.md` lines 304–307: subsection restates lines 243–249**

The `### Default Behavior Without Explicit Synchronization` header and its three-sentence body repeat: (1) no need for `ttnn.synchronize_device` in sequential pipelines, (2) implicit dependency tracking via tensor handles. The one novel clause — "The only time you must explicitly synchronize is when you need the CPU to observe a result (triggering `ttnn.from_device` or printing a tensor value), or when you are orchestrating concurrent multi-queue execution" — should be appended as a final sentence to the bullet list at lines 247–249 in the Synchronization opening. The `### Default Behavior Without Explicit Synchronization` header and body can then be deleted. Saves approximately 4 lines.

**M4 (carry-forward from Pass 2, not yet applied) — `mesh_device_setup.md` lines 72–82: "Device Ordering Conventions" standalone section overlaps `device_ids` parameter section**

The "Logical coordinate assignment" bullet (line 76) and "Expert-to-device locality" bullet (line 80) restate the same content as `device_ids` parameter section lines 46–50 (position N → `(row=0, col=N)`, logical-physical hop-distance alignment, non-standard ordering consequences). Collapsing the Conventions section into a summary paragraph appended to the `device_ids` parameter section and removing the standalone `## Device Ordering Conventions` header saves approximately 6–8 lines. The "Linear traversal direction" bullet (line 78) contains one detail not present in the `device_ids` section — the physical PCB wiring direction alignment rationale — and that sentence should be retained in the merged paragraph.

VERDICT: Crucial updates: yes

---

## Change Log — Pass 3 Fixes Applied

- C1 applied: Deleted the "Relationship to Chapter 1" section from `index.md` (5 lines: section header, blank line, single sentence, blank line, horizontal rule). The single sentence was implied by the Prerequisites block immediately above it; no content lost.

---

# Compression Analysis — Chapter 2: TTNN MeshDevice API — Pass 4

## Summary
- Files reviewed: `index.md`, `mesh_device_setup.md`, `tensor_distribution.md`, `collective_primitives.md`
- Current line count: ~974 lines (total across all four files: 79 + 229 + 264 + 402)
- Estimated post-compression line count: ~958 lines (~2% reduction)

## Pass 3 Item Verification

**C1 — `index.md` "Relationship to Chapter 1" section deleted:** ADDRESSED. The section is absent from `index.md`. The file runs from the opening paragraph directly to `## What This Chapter Covers` (line 7), then `## Prerequisites` (line 20), then `## TTNN Multi-Device Concepts Introduced in This Chapter` (line 33). No header, sentence, or divider from the former section remains. The 5-line saving is confirmed by the drop from ~984 lines (Pass 3 measured) to 974 lines (current count).

## CRUCIAL Suggestions

None.

## Load-Bearing Evidence

- **`index.md`** — The "TTNN Multi-Device Concepts Introduced in This Chapter" table (lines 37–52) is load-bearing. Each row maps a concept to both its introducing file and its first downstream use in a specific later chapter (Ch 3 through Ch 7). These forward references are not duplicated anywhere else in the chapter and establish the chapter's role in the guide's dependency chain.

- **`mesh_device_setup.md`** — The five-step initialization sequence (lines 88–98) is load-bearing. Each step names a specific internal action and the failure mode it surfaces. This is the only place in the chapter where the internal behavior of `ttnn.open_mesh_device` is explained; the `Insufficient L1` and `Stale Device State` pitfall sections (lines 207–213) reference it directly.

- **`tensor_distribution.md`** — The combined-constraint paragraph at lines 259–262 is load-bearing. The explicit derivation that the correct padding check is `dim % 256 == 0` (not `dim % 8 == 0`), plus the separate statement that non-divided dimensions must also be multiples of 32 for `TILE_LAYOUT`, are non-obvious correctness requirements not restated anywhere else in the chapter.

- **`collective_primitives.md`** — The conditional framing in the "Relationship to Reduce-Scatter" section (lines 231–235) is load-bearing. The specific claim that `reduce_scatter` + `all_gather` is only beneficial when the all-gather can be deferred or eliminated — and that if the all-gather is always required immediately the two-phase approach has the same total communication cost as direct `all_reduce` — is a decision-making distinction not restated elsewhere.

## MINOR Suggestions

**M1 (carry-forward, not applied in any prior pass) — `collective_primitives.md` line 86: third comment line is redundant**

Line 86 reads `# dispatch_buffer is a sharded mesh tensor on mesh_device.` This is evident from the call on line 88 (`ttnn.all_to_all(dispatch_buffer, ..., mesh_device=mesh_device, ...)`) and from the formal `input_tensor` parameter description immediately above the code block. Deleting this line saves 1 line.

**M2 (carry-forward, not applied in any prior pass) — `collective_primitives.md` line 288: hedging paragraph is two sentences where one clause suffices**

"Note: the exact API for selecting queue IDs per operation may differ between TTNN versions. Consult the current TTNN documentation for the `queue_id` parameter status." Replace with a single inline parenthetical such as `(queue_id availability is version-dependent; verify against current TTNN docs)` appended to the preceding sentence. Saves 1 line.

**M3 (carry-forward, not applied in any prior pass) — `collective_primitives.md` lines 304–307: "Default Behavior Without Explicit Synchronization" subsection restates lines 243–249**

The `### Default Behavior Without Explicit Synchronization` header and its three-sentence body repeat the same points already established in the opening of the Synchronization section (async dispatch, sequential dependencies handled implicitly via tensor handles, no explicit sync needed for simple pipelines). The one novel clause — "The only time you must explicitly synchronize is when you need the CPU to observe a result (triggering `ttnn.from_device` or printing a tensor value), or when you are orchestrating concurrent multi-queue execution" — should be appended as a final sentence to the bullet list at lines 251–253, and the standalone subsection header and body deleted. Saves approximately 4 lines.

**M4 (carry-forward, not applied in any prior pass) — `mesh_device_setup.md` lines 72–82: "Device Ordering Conventions" standalone section overlaps `device_ids` parameter section (lines 44–52)**

Both sections explain that position N maps to `(row=0, col=N)`, that logical column distance equals physical hop distance with standard ordering, and that non-standard ordering requires downstream updates to expert placement analysis. Collapsing the Conventions section into a summary paragraph appended to the `device_ids` parameter section and removing the `## Device Ordering Conventions` header saves approximately 6–8 lines. The "Linear traversal direction" bullet (line 78) contains one detail not present in the `device_ids` section — the rationale that standard ordering prevents logical forward steps from corresponding to physical backward steps — and that sentence should be retained in the merged paragraph.

VERDICT: Crucial updates: no
