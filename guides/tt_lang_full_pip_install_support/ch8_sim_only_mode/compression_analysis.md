# Compression Analysis: Sim-Only Installation Mode — Pass 1

## Summary
- Total files analyzed: 2
- Estimated current line count: ~375 lines
- Estimated post-compression line count: ~280 lines
- Estimated reduction: ~25%

## CRUCIAL Suggestions

### C1: Runtime dependency list repeated three times
The exact same five dependencies (torch, greenlet, pydantic, numpy, PyYAML) with identical version specifiers appear in:
- `index.md` lines 73-78 (prose list)
- `design_options.md` lines 31-37 (Option A pyproject.toml)
- `design_options.md` lines 87-93 (Option B pyproject.toml)

**Fix:** Define the dependency list once in `index.md` "What the Simulator Actually Needs" section. In `design_options.md`, reference it or use a comment like `# sim-only deps (see index.md)` in the TOML examples instead of re-listing all five with version pins.

Estimated savings: ~15 lines

### C2: CMake prose and code block say the same thing twice in index.md
Lines 18-24 describe in prose exactly what the CMake block (lines 27-56) demonstrates. The four numbered bullet points (creates venv, installs requirements, generates activate script, returns early) are then re-stated line-by-line in the code's comments.

**Fix:** Keep the code block with its inline comments. Remove or collapse the numbered prose list into a single-sentence lead-in like "When `TTLANG_SIM_ONLY=ON`, the build creates a venv, installs runtime requirements, stubs out compiler variables, and returns before the LLVM/tt-metal build:".

Estimated savings: ~10 lines

### C3: Option B "thin base + extension package" variant is admitted redundancy
`design_options.md` lines 155-163 describe a variant of Option B that the text itself acknowledges "is essentially Option A with different naming." This section adds no new design insight — it exists only to show that Option B collapses into Option A under scrutiny.

**Fix:** Fold into a single sentence in Option B's Cons, e.g., "Refining this to split extensions into a separate `ttl-compiler` package effectively collapses back into Option A."

Estimated savings: ~10 lines

## MINOR Suggestions

### M1: "Pure Python / no compiled extensions" concept over-stated
The idea that the simulator is pure Python and needs no C++ compilation is expressed at least 8 times across both files:
- `index.md` lines 5-6, 69, 79, 83, 94
- `design_options.md` lines 3, 40, 44, 231

Each mention is contextually appropriate, but the cumulative effect is verbose. A few of the interior restatements (e.g., `design_options.md` line 3 repeating the chapter intro, and line 40 restating what the table just established) could be trimmed.

### M2: Comparison table partially redundant with pros/cons
The comparison summary table (`design_options.md` lines 213-221) recaps information already covered in each option's Pros/Cons subsections. This is a common documentation pattern (summarize after detail), so it is not strictly wasteful, but roughly half the table cells add no new information beyond what the preceding sections stated.

### M3: Hedging and elaborative phrasing
- `design_options.md` line 145: "Projects like `pandas[sql]`, `httpx[http2]`, and `celery[redis]` use this widely" — the point is made by the first example; three is over-supporting.
- `index.md` line 7: "This chapter examines what TT-Lang already provides for this use case and proposes packaging designs..." — could be "This chapter covers existing sim-only support and proposes lightweight packaging designs."
- `design_options.md` lines 233-235: The "Code duplication is solvable" rationale point restates the mitigation already given in Option A's section verbatim.

### M4: "Notably absent" list in index.md (line 79)
The list of things *not* needed (`nanobind`, `cmake`, `ninja`, `ml_dtypes`, LLVM/MLIR/tt-metal) is useful context but could be a parenthetical rather than a standalone paragraph.

## Load-Bearing Evidence

- **C1 evidence:** `index.md:73-78`, `design_options.md:31-37`, `design_options.md:87-93` — three identical dependency lists.
- **C2 evidence:** `index.md:18-24` (prose) vs `index.md:27-56` (code with matching comments) — same information in two formats.
- **C3 evidence:** `design_options.md:162` — "This is essentially Option A with different naming" — the text explicitly flags its own redundancy.

## VERDICT
- Crucial updates: yes

## Change Log

### Applied 2026-04-13

**C1 — Runtime dependency list repeated three times:** Replaced the full five-dependency lists in `design_options.md` Option A `pyproject.toml` (lines 31-37) and Option B dependency layout (lines 87-93) with a single-line comment referencing `index.md` "What the Simulator Actually Needs". The canonical list remains in `index.md` lines 73-78.

**C2 — CMake prose and code block say the same thing twice:** Collapsed the four numbered bullet points in `index.md` (lines 18-24) into a single-sentence lead-in: "When `TTLANG_SIM_ONLY=ON`, the build creates a venv, installs runtime requirements, stubs out compiler variables, and returns before the LLVM/tt-metal build:". The code block with inline comments is preserved.

**C3 — Option B "thin base + extension package" variant is admitted redundancy:** Removed the standalone subsection (heading, code block, and explanation at lines 153-163 of `design_options.md`) and folded the insight into the last Cons bullet as: "Refining this to split extensions into a separate `ttl-compiler` package effectively collapses back into Option A."

# Compression Analysis: Sim-Only Installation Mode — Pass 2

## Re-check of CRUCIAL Items

### C1 (dependency list triplication): VERIFIED FIXED
`design_options.md` Option A `pyproject.toml` (line 32) and Option B dependency layout (line 83) now contain a single comment `# sim-only deps — see index.md "What the Simulator Actually Needs"` instead of the full five-dependency lists. The canonical list remains solely in `index.md` lines 64-69.

### C2 (CMake prose/code duplication): VERIFIED FIXED
`index.md` line 18 now reads as a single-sentence lead-in: "When `TTLANG_SIM_ONLY=ON`, the build creates a venv, installs runtime requirements, stubs out compiler variables, and returns before the LLVM/tt-metal build:". The redundant numbered bullet points are gone. The code block with inline comments is preserved.

### C3 (Option B thin-base variant): VERIFIED FIXED
The standalone subsection is removed. The insight is now a single sentence in Option B's last Cons bullet (`design_options.md` line 142): "Refining this to split extensions into a separate `ttl-compiler` package effectively collapses back into Option A."

## Load-Bearing Evidence

- **`index.md`**: The CMake code block (lines 20-49) carries the implementation detail for the existing `TTLANG_SIM_ONLY` mechanism, the dependency table (lines 56-61) and runtime dependency list (lines 64-69) define the sim-only surface area, and the four limitations (lines 76-81) motivate the design options. All are unique content; no further compression targets among them.
- **`design_options.md`**: The three-option evaluation structure (A/B/C) with per-option pros/cons is the core analytical framework. The recommendation section (lines 205-245) synthesizes insights from all three options into a concrete plan with a three-tier dependency diagram. Removing any option's analysis would break the comparative argument that justifies Option A.

## MINOR Suggestions

### M5: Rationale point 3 ("Code duplication is solvable") restates Option A mitigation
`design_options.md` lines 213-215 in the Recommended Approach rationale re-explain the `package_dir` trick already covered in Option A's "Mitigation for code duplication" subsection (lines 53-69). The rationale bullet could be shortened to: "Code duplication is solvable via `package_dir` as shown in Option A's mitigation section." This saves roughly 2-3 lines without losing the argument.

### M6: Three examples where one suffices in Option B pros
`design_options.md` line 136 lists three well-known extras-based packages (`pandas[sql]`, `httpx[http2]`, `celery[redis]`). One example is sufficient to establish the pattern; the additional two add no analytical value.

## VERDICT
- Crucial updates: no
