# Agent B Review -- Chapter 8

## Pass 1

**Issues found:** 5

---

### Issue 1: All cross-chapter links use non-existent `../chN_final/` directory paths
- **File:** `index.md`, `01_current_reset_mechanisms.md`, `02_reducing_reset_frequency_and_resilience.md`
- **Location:** Every cross-chapter link in all three content files plus the index (40+ occurrences total)
- **Category:** Cross-chapter inconsistency
- **Problem:** Every cross-chapter reference in Chapter 8 uses the path pattern `../chN_final/<filename>.md` (e.g., `../ch1_final/01_what_is_a_hang.md`, `../ch2_final/02_circular_buffer_deadlocks.md`, `../ch6_final/01_watcher_system.md`). The actual chapter directories are named `../ch01_anatomy_of_a_hang/`, `../ch02_kernel_and_noc_hangs/`, `../ch03_memory_related_hangs/`, `../ch04_dispatch_and_host_device_hangs/`, `../ch05_multi_chip_and_ccl_hangs/`, `../ch06_debugging_tools/`. No directory matching `*_final` exists anywhere in the guide.
- **Evidence:** Running `find` on the guide root shows directory names: `ch01_anatomy_of_a_hang`, `ch02_kernel_and_noc_hangs`, `ch03_memory_related_hangs`, `ch04_dispatch_and_host_device_hangs`, `ch05_multi_chip_and_ccl_hangs`, `ch06_debugging_tools`, `ch08_reset_reduction`. No `ch1_final`, `ch2_final`, etc. exist. This affects all three content files and the index -- every single prerequisite link, every inline reference in the prevention checklist (items 2-20 in `02_reducing_reset_frequency_and_resilience.md`), and every "Addresses:" reference.
- **Suggested fix:** Global find-and-replace across all Chapter 8 files:
  - `../ch1_final/` -> `../ch01_anatomy_of_a_hang/`
  - `../ch2_final/` -> `../ch02_kernel_and_noc_hangs/`
  - `../ch3_final/` -> `../ch03_memory_related_hangs/`
  - `../ch4_final/` -> `../ch04_dispatch_and_host_device_hangs/`
  - `../ch5_final/` -> `../ch05_multi_chip_and_ccl_hangs/`
  - `../ch6_final/` -> `../ch06_debugging_tools/`

---

### Issue 2: Chapter 7 reference points to a non-existent chapter
- **File:** `index.md`
- **Location:** Line 19
- **Category:** Cross-chapter inconsistency
- **Problem:** The index references "Chapter 7 ([`01_initial_triage.md`](../ch7_final/01_initial_triage.md))" as a prerequisite. No Chapter 7 directory exists in the guide structure at all -- the guide goes from `ch06_debugging_tools/` directly to `ch08_reset_reduction/` with no `ch07*` directory present. This is a dangling reference to content that either was never written or exists under a different name/number.
- **Evidence:** `ls` of the guide root shows directories ch01 through ch06 and ch08, with no ch07. The `ch7_final` path is doubly broken: wrong format and non-existent chapter.
- **Suggested fix:** Either (a) remove the Chapter 7 prerequisite entry entirely if that chapter does not exist yet, or (b) if there is a planned Chapter 7, update the reference to use the correct directory name once it is created. For now, the safest fix is to remove the entry and adjust the text that references Chapter 7 diagnostic workflows.

---

### Issue 3: "Level 0 (per-core reset)" should be "Level 1 (per-core reset)" in the Expected Impact table
- **File:** `03_future_tooling_proposals.md`
- **Location:** Lines 547, 549, 551 (the "Expected Impact on Reset Frequency" table)
- **Category:** Factual error
- **Problem:** The table describes the post-Phase 2 outcome for several scenarios as "Level 0 (per-core reset)." According to the hierarchy defined in `01_current_reset_mechanisms.md` Section 1, Level 0 is "Graceful Program Termination" (no hardware reset), and Level 1 is "Tensix Per-Core Soft Reset." Per-core reset is Level 1, not Level 0. Using "Level 0" here contradicts the chapter's own 5-level hierarchy and will confuse readers who learned that hierarchy from Section 01.
- **Evidence:** Section 01, line 22: "Level 0: Graceful Program Termination (~ms, no hardware reset)" and line 23: "Level 1: Tensix Per-Core Soft Reset (~us per core, resets individual RISCs)." The Impact table says:
  - CB deadlock after Phase 2: "Level 0 (per-core reset)" -- should be Level 1
  - Semaphore hang after Phase 2: "Level 0 (per-core reset if NOC is clean)" -- should be Level 1
  - Dispatch stall after Phase 2: "Level 0 (partial reset of dispatch core)" -- should be Level 1
- **Suggested fix:** Change all three entries from "Level 0" to "Level 1":
  - `Level 0 (per-core reset)` -> `Level 1 (per-core reset)`
  - `Level 0 (per-core reset if NOC is clean)` -> `Level 1 (per-core reset if NOC is clean)`
  - `Level 0 (partial reset of dispatch core)` -> `Level 1 (partial reset of dispatch core)`

---

### Issue 4: Proposal 1 classification table labels `HQW` as "Dispatch core" instead of "Prefetch kernel"
- **File:** `03_future_tooling_proposals.md`
- **Location:** Line 62 (Proposal 1 classification rules table)
- **Category:** Factual error
- **Problem:** The automatic hang classification table entry reads: "Dispatch core at `HQW` (waiting for host) | Dispatch stall (host not feeding)." Chapter 4 (`01_dispatch_architecture_and_hang_points.md`, Scenario 4.1.1) clearly documents `HQW` as the **prefetch kernel's** waypoint (set in `fetch_q_get_cmds()` in `cq_prefetch.cpp`), not the dispatch kernel's. The dispatch and prefetch kernels are distinct components of the fast dispatch pipeline running on separate cores. Using "Dispatch core" here conflates them and would cause an automatic classifier to misidentify the hung component, undermining the proposal's stated goal of eliminating manual log reading.
- **Evidence:** Chapter 4 `01_dispatch_architecture_and_hang_points.md`, Scenario 4.1.1 title: "Prefetch Kernel Stall on Host (Empty Fetch Queue)" with symptom: "The prefetch kernel is stuck at waypoint `HQW`." Chapter 4 index (line 87) lists `HQW` under prefetch-specific waypoints, separate from dispatch waypoints like `PWW`/`WCW`.
- **Suggested fix:** Change "Dispatch core at `HQW` (waiting for host)" to "Prefetch core at `HQW` (waiting for host)" and update the classification from "Dispatch stall (host not feeding)" to "Prefetch stall (host not feeding commands)" or "Dispatch pipeline stall (prefetch waiting for host)".

---

### Issue 5: Blackhole M3 reset description is internally contradictory
- **File:** `01_current_reset_mechanisms.md`
- **Location:** Lines 313 and 441
- **Category:** Factual error
- **Problem:** Section 4.3.3 (Blackhole Legacy Reset) explicitly states: 'The `reset_m3` flag has no effect on Blackhole ("Reset M3 flag doesn't influence Blackhole reset.").' However, Section 6 (Level 3: M3/DMC Board-Level Reset) states: "In the Blackhole path, the driver sends `ARC_MSG_TYPE_TRIGGER_RESET` with payload `3`." These two statements are contradictory -- either Blackhole supports a distinct M3 reset path (with a specific ARC message) or the M3 flag has no effect. The reader cannot determine the actual Blackhole behavior from these conflicting claims.
- **Evidence:** Line 313: "Note: The `reset_m3` flag has no effect on Blackhole." Line 441: "In the Blackhole path, the driver sends `ARC_MSG_TYPE_TRIGGER_RESET` with payload `3`."
- **Suggested fix:** Clarify that the two statements describe different code paths. Section 4.3.3 describes the UMD legacy path (`warm_reset_blackhole_legacy()`), where the M3 flag is indeed ignored. Section 6 describes the kernel driver path (`blackhole_reset()` in `blackhole.c`), where the ASIC_DMC_RESET IOCTL may have a different handler. Add a qualifying note in Section 6 explaining which code path the Blackhole behavior refers to, or note that Blackhole's DMC reset via the arch-agnostic KMD path may behave differently from the legacy UMD path. If M3 truly has no effect on Blackhole at any level, remove the Blackhole claim from Section 6.

---

## Format Compliance

The chapter's structure is well-organized and internally consistent across the three content files. Since this chapter is analysis/recommendation rather than hang scenarios, the absence of the X.Y.Z scenario format is appropriate and noted in the synthesis notes. The three content files follow a clear progression: mechanisms (Section 01), prevention and resilience (Section 02), and future proposals (Section 03). Tables, code blocks, and headers are formatted consistently. The prevention checklist in Section 02 and the proposal impact matrix in Section 03 are well-structured reference tables.

## What the Chapter Gets Right

- The 5-level reset hierarchy is clearly defined and well-documented with exact code paths, timing data, and state destruction consequences at each level.
- The kernel driver (tt-kmd) safety mechanisms section (reset_gen counter, reset_rwsem lock, reset window, PCIe hot reset, reset marker) is thorough and unique to this chapter -- no other chapter covers these critical driver-level details.
- The 12 future proposals are well-structured with consistent format (priority, effort, dependencies, gap, impact, implementation outline) and the dependency graph provides a clear engineering roadmap.
- The prevention checklist (20 items) in Section 02 provides actionable guidance tied to specific hang categories and chapter references.
- The quantitative impact estimates throughout the chapter, while necessarily approximate, provide useful guidance for prioritizing efforts.
- The TensixSoftResetOptions enum with correct bit positions (11, 12, 13, 14, 18) matches the source corrections documented in the synthesis notes.
- The multi-host distributed reset documentation (barrier-synchronized via NFS) is a valuable practical reference for Galaxy deployments.

---

**Verdict:** NEEDS REVISION

The cross-chapter link issue (Issue 1) is pervasive and affects all content files -- every single cross-chapter reference is broken. Combined with the non-existent Chapter 7 reference (Issue 2) and the reset level mislabeling (Issue 3), these issues would cause significant confusion for readers navigating the guide. The factual issues (Issues 3-5) are individually smaller but collectively undermine the chapter's precision, which is especially important for a reference chapter on reset mechanisms.
