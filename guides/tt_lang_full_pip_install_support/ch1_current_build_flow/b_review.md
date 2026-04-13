# Chapter 1 — Agent B (Critic) Review

1. **`cmake_architecture.md`, line 3 and lines 33-49 — Include order diagram places `TTLangCompilerSetup` after `TTLangPython`; actual order is reversed.** The diagram shows `TTLangPython` before `TTLangCompilerSetup`, but in the actual `CMakeLists.txt`, `TTLangCompilerSetup` is included at line 37, *before* the toolchain option declarations (lines 42-53) and `TTLangPython` (line 59). The guide also states (line 7) that `BuildLLVM.cmake` needs the Python venv from `TTLangPython.cmake` as justification for the ordering, which is correct — but the diagram itself misstates where `TTLangCompilerSetup` falls in the sequence. The diagram should show `TTLangCompilerSetup` immediately after `GetVersionFromGit` and before `TTLangPython`, matching the actual source. Additionally, the prose on line 3 says "six cmake modules under `cmake/modules/`" but there are seven (`TTLangUtils.cmake` is the seventh, and it is even shown first in the diagram). Fix: change "six" to "seven" and reorder the diagram to place `TTLangCompilerSetup` before `TTLangPython`.

2. **`cmake_architecture.md`, line 168 — LLK header count stated as 13; actual count is 14.** `BuildTTMLIRMinimal.cmake` lists 14 LLK headers (lines 109-123): tilize, untilize, pack_untilize, invoke_sfpi, dataflow_api, matmul, padding, coord_translation, fabric_topology_info, fabric_1d_routing, fabric_2d_routing, fabric_api, reg_api, and semaphore. The guide's description on line 168 says "13 LLK headers" and omits several of the fabric/routing headers from its prose summary ("tilize/untilize operations, SFPI invocation, dataflow API, matmul, padding, coordinate translation, fabric topology, and semaphores"). Fix: change "13" to "14" and update the prose to mention fabric routing and register API headers.

3. **`cmake_architecture.md`, lines 247-258 — Venv creation attributed to `TTLangPython.cmake`; actual creation happens in `BuildLLVM.cmake`.** The section titled "TTLangPython.cmake" (starting line 247) says the module handles "venv creation and discovery logic." In reality, `TTLangPython.cmake` only *discovers* existing venvs or *sets the path variable* for later creation — it never calls `python -m venv`. The actual venv creation (including `pip install --upgrade pip` and `requirements.txt` installation) happens in `BuildLLVM.cmake` lines 139-167. A reader implementing a new build path who relies on `TTLangPython.cmake` to create the venv would find it does not. Fix: clarify that `TTLangPython.cmake` performs discovery and path resolution only, and that venv creation is handled by `BuildLLVM.cmake` (and the `TTLANG_SIM_ONLY` block in `CMakeLists.txt`).

4. **`cmake_architecture.md`, line 175 — TTMetal dialect listed as "IR only" in the C++ libraries section, but TTMetal/Transforms is also built.** The C++ dialect libraries list says `MLIRTTMetalDialect -- TTMetal IR`, implying only IR is built. However, `BuildTTMLIRMinimal.cmake` runs `add_subdirectory` on both `TTMetal/IR` (line 75) and `TTMetal/Transforms` (line 79), and the warning-suppression target list (line 154) includes `obj.MLIRTTTransforms` which covers TTMetal transforms. A reader building a subset of these libraries would incorrectly omit the TTMetal transforms subdirectory. Fix: change line 175 to `MLIRTTMetalDialect -- TTMetal IR and transforms` (or list the transforms target separately if it produces a distinct library).

## Pass 2

All four items from Pass 1 have been addressed:

1. Include order diagram now correctly places `TTLangCompilerSetup` before `TTLangPython`, and the module count reads "seven." Verified against `CMakeLists.txt` lines 14-135.
2. LLK header count now reads "14" with prose covering all headers including fabric routing and register API. Verified against `BuildTTMLIRMinimal.cmake` lines 108-123.
3. `TTLangPython.cmake` section now correctly describes it as performing "discovery and path resolution" only, with venv creation attributed to `BuildLLVM.cmake` and the `TTLANG_SIM_ONLY` block. Verified against `TTLangPython.cmake` (no `python -m venv` call) and `BuildLLVM.cmake` lines 139-167.
4. `MLIRTTMetalDialect` entry now reads "TTMetal IR and transforms." Verified against `BuildTTMLIRMinimal.cmake` lines 74-79.

Navigation footers are present on all three content files. Chapter `index.md` uses clickable relative markdown links for both `cmake_architecture.md` and `environment_assumptions.md`.

**No feedback — chapter approved.**

## Pass 3

Verified all four Pass 1 corrections remain intact and accurate against the source code:

1. Include order diagram: `TTLangCompilerSetup` correctly precedes `TTLangPython`, module count says "seven." Confirmed against `CMakeLists.txt` lines 14-59.
2. LLK header count: reads "14" with complete prose coverage. Confirmed against `BuildTTMLIRMinimal.cmake` lines 108-123.
3. `TTLangPython.cmake` section: correctly describes discovery/path-resolution only, with venv creation attributed to `BuildLLVM.cmake` and the `TTLANG_SIM_ONLY` block. Confirmed no `python -m venv` call in `TTLangPython.cmake`.
4. `MLIRTTMetalDialect` entry: reads "TTMetal IR and transforms." Confirmed against `BuildTTMLIRMinimal.cmake` lines 74-79.

Navigation footers present on all three content files (`index.md`, `cmake_architecture.md`, `environment_assumptions.md`). All relative markdown links in `index.md` are clickable and point to correct targets. The `environment_assumptions.md` footer correctly links forward to Chapter 2.

**No feedback — chapter approved.**
