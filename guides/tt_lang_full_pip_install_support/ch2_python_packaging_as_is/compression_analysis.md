# Compression Analysis: Python Packaging As-Is — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~545 lines
- Estimated post-compression line count: ~440 lines
- Estimated reduction: ~19%

## CRUCIAL Suggestions

### C1: Duplicate explanation of `cwd.parent / "build"` path assumption

**Files:** `cmake_build_class.md` (lines 57-84) and `why_pip_install_fails.md` (lines 47-65)

Both files explain that `cwd` is assumed to be `python/` and that `cwd.parent / "build"` resolves to the repo root's build directory. `cmake_build_class.md` states it as a "key assumption" with a bullet point; `why_pip_install_fails.md` section 3 restates it with a table of scenarios. The code snippet `cwd = pathlib.Path().absolute()` / `build_dir = cwd.parent / "build"` appears in both files.

**Recommendation:** In `cmake_build_class.md`, keep the code and one-liner noting the assumption, then add a forward reference: "See [why_pip_install_fails.md, section 3] for failure scenarios when this assumption breaks." Remove the explanatory bullets from cmake_build_class.md lines 76-84 that duplicate the failure analysis.

**Estimated saving:** ~10 lines

### C2: Duplicate CMake configure command shown in two files

**Files:** `cmake_build_class.md` (lines 56-73) and `why_pip_install_fails.md` (lines 9-15)

The CMake configure invocation (`cmake -G Ninja -B ...`) is shown as a full code block in both files. In `why_pip_install_fails.md` it is a simplified shell form; in `cmake_build_class.md` it is the Python form with the full `cmake_args` list.

**Recommendation:** In `why_pip_install_fails.md` section 1, replace the code block with a one-line reference: "The CMake configure step (see [cmake_build_class.md, Step 1]) runs..." and keep only the prose about missing LLVM/MLIR. The reader has already seen the command if reading sequentially, or can click through.

**Estimated saving:** ~8 lines

### C3: Pykernel's separate-target status explained twice in `cmake_build_class.md`

**File:** `cmake_build_class.md` (lines 131-139 AND lines 203-208)

The "Note on `pykernel`" paragraph at lines 131-139 explains that `PykernelPythonModules` is a separate CMake target not built by `TTLangPythonModules`. Lines 203-208 in "The package list" section restate this same fact. Both passages cite the same CMakeLists.txt lines (252-257).

**Recommendation:** Keep the detailed explanation in the Step 3 note (lines 131-139) and reduce lines 203-208 to a back-reference: "(`pykernel` is a separate CMake target -- see Step 3 note above)."

**Estimated saving:** ~6 lines

### C4: `setup()` call and package list duplicated between index.md and cmake_build_class.md

**Files:** `index.md` (lines 52-72) and `cmake_build_class.md` (lines 182-208)

`index.md` shows the full `setup()` call including `packages=[...]` and `package_dir={...}`. `cmake_build_class.md` then re-lists the same six packages with a table and explanatory paragraph. The package list and package_dir mapping appear in both files with no new information in the index.md version beyond what cmake_build_class.md covers.

**Recommendation:** In `index.md`, trim the `setup()` code block to show only `name`, `ext_modules`, and `cmdclass` (the structurally relevant fields). Add a note: "For the full package list and package_dir mapping, see [cmake_build_class.md]." This avoids the reader encountering the same six-package list twice.

**Estimated saving:** ~12 lines

## MINOR Suggestions

### M1: Verbose "How setuptools resolves the two files" section in index.md

`index.md` lines 79-98 walk through the 4-step PEP 517 resolution process and then explain the mismatch. This overlaps heavily with `why_pip_install_fails.md` section 3 (path assumptions). Consider condensing to 2-3 sentences with a forward reference to the failure analysis.

### M2: Hedging language in why_pip_install_fails.md

Line 143: "For pure Python files that are identical in both locations this is harmless, but..." -- the qualification is low-value since the important point is the generated-file case. Could be trimmed to start directly with the generated-file risk.

### M3: "Why the split exists" section in index.md is mostly obvious

`index.md` lines 100-119 explain that the repo root is the CMake source directory, python/ has Python files, and pyproject.toml must be at root per PEP 517. For an audience familiar with CMake and PEP 517 (the target readership), this is largely self-evident. Could be reduced to 3-4 lines.

### M4: Redundant navigation footers

All three files end with "**Next:** [link]" footers. The index.md already lists both sub-files under "Files covered." The footers add ~3 lines of pure navigation that duplicate the index.

## Load-Bearing Evidence

- **index.md** line 90: `"This is the first critical mismatch: setup.py lives in python/, not at the repo root."` -- restates what why_pip_install_fails.md section 3 covers in detail with a table.
- **cmake_build_class.md** line 131: `"Note on pykernel: PykernelPythonModules is a separate add_mlir_python_modules() target"` -- first of two occurrences of this fact in the same file (see also line 204).
- **why_pip_install_fails.md** line 12: `"cmake -G Ninja -B <repo_root>/build -S <repo_root>"` -- duplicates the configure command already shown in cmake_build_class.md lines 56-73.

## VERDICT
- Crucial updates: yes

---

# Compression Analysis — Change Log

## 2026-04-09 — Applied Agent B feedback (Pass 1, Issues 1 and 2)

**File modified:** `cmake_build_class.md`

### Issue 1: `sim` package falsely claimed to be routed through CMake

**What changed:** The paragraph after the package table previously stated that
all five pure-Python packages "are still routed through the CMake build system
via `declare_mlir_python_sources`." This was incorrect for `sim`, which does not
appear anywhere in `python/CMakeLists.txt`. The text now correctly distinguishes
three groups:

- `ttl._src` and `utils` — declared via `declare_mlir_python_sources`, built as
  part of `TTLangPythonModules`.
- `pykernel` and `pykernel._src` — declared via `declare_mlir_python_sources`,
  but under the separate `PykernelPythonModules` target.
- `sim` — handled exclusively by setuptools `build_py`.

**Verified against:** `tt-lang/python/CMakeLists.txt` (no mention of `sim`)
and `tt-lang/python/setup.py` (`sim` in `packages` and `package_dir` only).

### Issue 2: `pykernel` described as part of `TTLangPythonModules` output

**What changed:** The Step 3 (Install) section previously stated that the
install copies "`ttl/` and `pykernel/` as top-level directories" from
`python_packages/`. This was misleading because `PykernelPythonModules` is a
separate `add_mlir_python_modules()` target (CMakeLists.txt lines 252-257) that
is not a dependency of `TTLangPythonModules`. Building only `TTLangPythonModules`
does not produce `python_packages/pykernel/`. The text now explains that
`pykernel` files in the final wheel come from setuptools `build_py` via
`package_dir`, not from the CMake install step.

**Verified against:** `tt-lang/python/CMakeLists.txt` lines 252-257
(`PykernelPythonModules`) and lines 261-278 (`TTLangPythonModules` —
`PykernelPythonSources` is not listed in `DECLARED_SOURCES`).

## 2026-04-09 — Applied compression suggestions (Pass 1, CRUCIAL C1-C4 + MINOR M2)

**Files modified:** `index.md`, `cmake_build_class.md`, `why_pip_install_fails.md`

### C1: Duplicate `cwd.parent / "build"` explanation

Replaced the expanded "Key assumptions" bullet about `cwd` being `python/` in
`cmake_build_class.md` with a one-liner plus forward reference to
`why_pip_install_fails.md` section 3 for failure scenarios.

### C2: Duplicate CMake configure command

Replaced the full `cmake -G Ninja ...` code block in `why_pip_install_fails.md`
section 1 with a prose sentence referencing `cmake_build_class.md` Step 1.

### C3: Pykernel separate-target status explained twice

Reduced the second explanation in the "The package list" section of
`cmake_build_class.md` to a back-reference ("see Step 3 note above").

### C4: `setup()` call and package list duplicated

Trimmed the `setup()` code block in `index.md` to show only structurally
relevant fields (`ext_modules`, `cmdclass`) with placeholder comments for
`packages` and `package_dir`, plus a forward reference to `cmake_build_class.md`.

### M2: Hedging language in why_pip_install_fails.md

Removed the "For pure Python files that are identical in both locations this is
harmless, but" qualification, starting directly with the generated-file risk.

---

# Compression Analysis: Python Packaging As-Is — Pass 2

## Re-check of CRUCIAL items from Pass 1

### C1: Duplicate `cwd.parent / "build"` path explanation — RESOLVED

`cmake_build_class.md` lines 77-80 now contain a concise one-liner about the
`cwd` assumption ("**`cwd` is `python/`**, so `cwd.parent` is the repo root")
plus a forward reference to `why_pip_install_fails.md` section 3 for failure
scenarios. The code snippet (`cwd = pathlib.Path().absolute()` /
`build_dir = cwd.parent / "build"`) appears in both files, but this is
acceptable: `cmake_build_class.md` shows it as part of the full source listing,
while `why_pip_install_fails.md` quotes just the two lines to anchor its failure
analysis table. No remaining duplication of explanatory prose.

### C2: Duplicate CMake configure command — RESOLVED

`why_pip_install_fails.md` section 1 (lines 9-11) now opens with a prose
reference ("The `CMakeBuild.build_()` method runs the CMake configure step (see
[cmake_build_class.md, Step 1]...)") instead of reproducing the command. The
full `cmake_args` code block lives only in `cmake_build_class.md` lines 56-73.
No duplicate command block remains.

### C3: Pykernel separate-target stated twice in `cmake_build_class.md` — RESOLVED

The detailed "Note on `pykernel`" paragraph remains at lines 131-139 (Step 3).
The package-list section at lines 204-205 now reads: "`pykernel` and
`pykernel._src` are under a separate CMake target (`PykernelPythonModules` —
see Step 3 note above)." This is a back-reference, not a restatement. No
duplicate explanation remains.

### C4: `setup()` package list in `index.md` and `cmake_build_class.md` — RESOLVED

`index.md` lines 52-69 now show a trimmed `setup()` call with placeholder
comments (`packages=[...],  # 6 packages — see cmake_build_class.md` and
`package_dir={...},  # relative paths from python/`). Lines 66-69 provide a
forward reference to `cmake_build_class.md` for the full package list and
`package_dir` mapping. The six-package enumeration appears only once, in
`cmake_build_class.md` lines 184-198.

## Load-Bearing Evidence

- **index.md** lines 59-60: `packages=[...],  # 6 packages — see cmake_build_class.md` and `package_dir={...},  # relative paths from python/` — confirms the package list is no longer duplicated; only a placeholder with forward reference remains.
- **cmake_build_class.md** line 79: `"See [why_pip_install_fails.md, section 3] for failure scenarios when this assumption breaks."` — confirms the `cwd` path explanation now defers to the failure analysis file rather than restating it.
- **why_pip_install_fails.md** lines 9-11: `"The CMakeBuild.build_() method runs the CMake configure step (see [cmake_build_class.md, Step 1]...)"` — confirms the configure command is referenced, not reproduced.

## MINOR Suggestions

### M1: The three-row scenario table in `why_pip_install_fails.md` section 3 (lines 55-59) could be condensed

The table showing what `cwd.parent / "build"` resolves to under three scenarios
is clear but verbose. The middle column ("Working directory") is inferable from
the scenario name. Consider dropping that column and keeping only "Scenario" and
"Resolves to" to save ~3 lines of table width without losing information.

### M2: `cmake_build_class.md` CI mode section (lines 141-180) is self-contained but long

The CI mode section covers `env/activate` sourcing and install-dir adjustment.
Since the guide's focus is on making `pip install .` work (not CI), this section
could be trimmed to a single paragraph noting the two behavioral differences,
with a "see source" reference for the code blocks. This would save ~15 lines but
is non-critical since CI mode is a valid reference for understanding the build.

## VERDICT
- Crucial updates: no
