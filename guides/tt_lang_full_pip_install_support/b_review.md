# Cross-Chapter Consistency Review

## Item 1 — Dependency version inconsistency between Ch5/Ch8 and Ch6

Ch5 (`pyproject_toml_changes.md`, lines 35-36) and Ch8 (`index.md`, lines 66/69) specify `torch>=1.9.0` and `numpy>=1.20.0`. Ch6 (`index.md`, line 48; `main_wheel_design.md`, line 105-106) specifies `torch>=2.1` and `numpy>=1.24`. These are the same project's declared dependencies at different points in the guide. A reader following Ch5's proposed `pyproject.toml` and then Ch6's dependency graph will see conflicting minimum versions with no explanation of why they differ. Pick one set of version floors and use it consistently, or add a note in Ch6 explaining the intentional bump.

## Item 2 — Stale navigation footer in Ch6 `build_pipeline.md`

The last line of `ch6_two_phase_wheel_architecture/build_pipeline.md` (line 223) reads:

> **Next:** Chapter 7 -- Wheel Packaging and Platform Compliance (forthcoming)

Chapter 7 exists and is fully written. The "(forthcoming)" label is stale, and the text is not a link. It should be:

> **Next:** [Chapter 7 -- Wheel Packaging and Platform Compliance](../ch7_wheel_packaging/index.md)

## Item 3 — Guide index Key Concepts for Ch1 lists "five build phases" but Ch1 lists five phases with different names

The guide index (line 26) lists key concepts for Ch1 including "five build phases". Ch1's table names them: Configure, LLVM Build, tt-metal Build, tt-lang Build, Install/Finalize. The plan (line 26-27) calls them: configure, LLVM build, tt-metal build, tt-lang build, install/finalize. These are consistent with each other. No issue here after closer inspection — withdrawn.

**Revised count: 2 items.**

---

**No further issues found.** The terminology for "toolchain" (pre-built artifact directory) vs. "toolchain wheel" (`ttl-toolchain` pip package) is used consistently and distinctly across chapters. Cross-chapter references point to correct files. Concepts are introduced before use (CAPI library in Ch3 before Ch5/Ch6/Ch7 reference it; `TTLANG_USE_TOOLCHAIN` in Ch1 before Ch5 proposes changes). Chapter titles and descriptions in the guide index match the actual content.

## Final Pass

Verified the two cross-chapter fixes requested in the prior review:

**Fix 1 — Dependency version consistency (Item 1):** Partially fixed. The `pyproject.toml` snippets in `ch6/index.md` (lines 48-49) and `ch6/main_wheel_design.md` (lines 105-106) now show `torch>=1.9.0` and `numpy>=1.20.0`, consistent with Ch5 and Ch8. However, the **Wheel Metadata** block in `ch6/main_wheel_design.md` (lines 168-169) still reads `Requires-Dist: torch >=2.1` and `Requires-Dist: numpy >=1.24`. These should be updated to `torch >=1.9.0` and `numpy >=1.20.0` to match the authoritative `pyproject.toml` snippets earlier in the same file.

**Fix 2 — Stale navigation footer (Item 2):** Fixed. `ch6/build_pipeline.md` line 223 now reads `**Next:** [Chapter 7 — Wheel Packaging and Platform Compliance](../ch7_wheel_packaging/index.md)` — the "(forthcoming)" label is removed and the text is a working relative link.

**Guide index verification:** The guide index (`index.md`) lists all 8 chapters with correct relative links (`ch1_current_build_flow/index.md` through `ch8_sim_only_mode/index.md`). Titles and descriptions match actual chapter content.

**Verdict:** One residual inconsistency remains (Wheel Metadata block in `main_wheel_design.md` lines 168-169). After that single fix, no feedback — guide approved.
