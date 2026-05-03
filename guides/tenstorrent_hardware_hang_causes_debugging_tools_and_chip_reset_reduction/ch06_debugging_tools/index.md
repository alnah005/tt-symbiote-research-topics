# Chapter 6: Debugging Tools and Infrastructure

## Summary

This chapter provides a comprehensive, reference-card-style catalog of every debugging tool and infrastructure component available in the Tenstorrent tt-metal stack for detecting, diagnosing, and recovering from hardware hangs. Each section is organized around exhaustive configuration reference tables and decision trees designed for quick lookup during active debugging sessions, combined with practical recipes and progressive diagnosis workflows.

The tools covered span the full lifecycle of a hang investigation: from always-on monitoring (Watcher), to post-mortem analysis (watcher_dump), to device-side instrumentation (DPRINT/DEVICE_PRINT), to automated triage (tt-triage), to performance profiling (Tracy, NOC debug dump), and finally to timing perturbation and dispatch debugging for reproducing intermittent hangs.

## Prerequisites

- Familiarity with Chapters 1-5 of this guide (hang taxonomy, kernel/NOC hangs, memory hangs, dispatch hangs, multi-chip/CCL hangs)
- Access to a tt-metal source build environment (`build_metal.sh`)
- Understanding of Tenstorrent RISC-V processor types: BRISC, NCRISC, TRISC0/1/2, ERISC
- Basic knowledge of NOC architecture (NOC0, NOC1, unicast, multicast) and L1 memory layout
- Understanding of the dispatch system (command queues, prefetch, dispatch kernels)
- Python 3.x for tt-triage scripts; tt-exalens installed via `scripts/install_debugger.sh`
- Tracy profiler client (for visualization, Section 6.5)

## Tool Selection Decision Tree

Before diving into individual tools, use this decision tree to select the right starting point:

```
Is the device currently hung?
|
+-- YES: Is the process still alive?
|   |
|   +-- YES: Can you attach with GDB?
|   |   |
|   |   +-- YES --> Use GDB + Watcher dump (Section 6.1, GDB Integration)
|   |   +-- NO  --> Use tt-triage (Section 6.4)
|   |
|   +-- NO (process crashed/killed):
|       |
|       +-- Was Watcher enabled during the run?
|           |
|           +-- YES --> Read generated/watcher/watcher.log (Section 6.1)
|           +-- NO  --> Use watcher_dump standalone tool (Section 6.2)
|
+-- NO: Trying to reproduce or diagnose an intermittent hang?
    |
    +-- Suspect a race condition?
    |   --> Use Debug Delay / Timing Perturbation (Section 6.6)
    |
    +-- Suspect a specific kernel is stalling?
    |   --> Enable DPRINT on targeted cores (Section 6.3)
    |
    +-- Need performance timeline to find slow regions?
    |   --> Use Tracy Profiler (Section 6.5)
    |
    +-- Suspect missing NOC barriers?
        --> Use NOC Debug Dump (Section 6.5)
```

## Progressive Diagnosis Pipeline

The following pipeline describes the recommended order of tool engagement when investigating a hang. Each stage builds on the prior one.

```
Stage 0: Observation (no code change)
  tt-smi                    -- Is the device alive? Thermal? Power?
  TT_METAL_OPERATION_TIMEOUT_SECONDS  -- Did the host detect a timeout?

Stage 1: Passive Post-Mortem (no rebuild required)
  watcher_dump -w           -- Read watcher mailboxes from device memory
  tt-triage                 -- Automated callstacks, ARC checks, NOC checks
  Inspector logs            -- Operation sequence, kernel parameters

Stage 2: Active Monitoring (env vars, no code change)
  TT_METAL_WATCHER=120      -- Enable watcher polling thread (120 seconds)
  Watcher log analysis      -- Waypoints, NOC sanitize errors, assertions

Stage 3: Targeted Instrumentation (env vars + possible rebuild)
  TT_METAL_DPRINT_CORES=... -- Enable DPRINT on suspect cores
  TT_METAL_DEVICE_PROFILER=1 -- Tracy profiling for timing analysis
  TT_METAL_NOC_DEBUG_DUMP=1  -- Detect missing NOC barriers

Stage 4: Perturbation Testing (deliberate timing changes)
  Debug delay insertion     -- TT_METAL_WATCHER_DEBUG_DELAY + delay cores
  Null kernels              -- TT_METAL_NULL_KERNELS to isolate dispatch
  Timing perturbation       -- add_compute_nops<>() for compute races

Stage 5: Deep Analysis (source-level investigation)
  GDB + watcher dump        -- call tt::watcher::dump(stderr, true)
  Lightweight asserts       -- TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
  WATCHER_RING_BUFFER_PUSH  -- Custom kernel instrumentation
```

## Tool Selection Quick Reference

| Symptom | First Tool | Second Tool | Chapter Reference |
|---------|-----------|-------------|-------------------|
| Complete hang, no output | Watcher (`TT_METAL_WATCHER=1`) | `tt-triage` | Sections 6.1, 6.4 |
| Hang on specific core/RISC | Watcher waypoints + ring buffer | DPrint on target core | Sections 6.1, 6.3 |
| NOC address violation | Watcher NOC sanitization | NOC Debug Dump | Sections 6.1, 6.5 |
| Dispatch pipeline stall | Watcher + `watcher_dump -c` | Dispatch debug tools | Sections 6.1, 6.2, 6.6 |
| Multi-chip Ethernet hang | Watcher ETH link status | Fabric telemetry | Sections 6.1, 6.5 |
| Intermittent/timing-dependent hang | Debug delay | Timing perturbation | Section 6.6 |
| Post-mortem (chip already hung) | `watcher_dump` | `tt-triage` | Sections 6.2, 6.4 |
| Performance regression (not hang) | Tracy profiler | NOC event profiler | Section 6.5 |
| Assertion failure without watcher | Lightweight asserts | Watcher full asserts | Sections 6.1, 6.5 |
| Memory corruption suspected | Watcher CB sanitization | NOC Debug Dump | Sections 6.1, 6.5 |

## Chapter Files

| File | Title | Description |
|------|-------|-------------|
| `01_watcher_system.md` | Watcher System Architecture and Configuration | Complete reference for WatcherServer, WatcherDeviceReader, all monitoring features (waypoints, NOC sanitization, assertions, ring buffer, pause, stack usage, ETH link, CB sanitization, linked transactions), env var tables, log format, GDB integration, and performance impact |
| `02_watcher_dump_tool.md` | Standalone watcher_dump Post-Mortem Tool | The watcher_dump binary for post-mortem analysis without a running program, CQ dump, NOC logging dump, CLI reference, and limitations |
| `03_dprint_server.md` | DPRINT / DEVICE_PRINT Server | Device-side printf-style debugging via DPRINT macro and DPrintServer, targeting cores/RISCs/chips, TileSlice printing, hang risk from full buffers, interaction with watcher |
| `04_tt_triage_tool.md` | tt-triage Automated Diagnostic System | The tt-triage.py tool, all discoverable scripts (data providers, state checkers, dump scripts), CLI options, Inspector integration, custom script authoring |
| `05_profiler_tracy_and_noc_debug.md` | Profiler, Tracy, NOC Debug, and Lightweight Asserts | Tracy device profiler, NOC debug dump for missing barrier detection, fabric telemetry, tt-smi, lightweight asserts, LLK asserts |
| `06_debug_delay_and_timing_perturbation.md` | Debug Delay, Timing Perturbation, and Dispatch Debug | Debug delay insertion for NOC read/write/atomic operations, compute timing perturbation via NOP insertion, dispatch debug tools, progress heartbeat monitoring, dispatch timeout auto-trigger |

## Cross-References to Prior Chapters

| Chapter | Primary Diagnostic Tools | Secondary Tools |
|---------|------------------------|-----------------|
| Ch1 (Taxonomy) | Watcher waypoint codes (I/W/R/D/X) map directly to the 5-part hang scenario format | -- |
| Ch2 (Kernel/NOC Hangs) | Watcher (waypoints + NOC sanitize), DPRINT | tt-triage callstacks, debug delay |
| Ch3 (Memory Hangs) | Watcher (CB sanitize), NOC debug dump | Lightweight asserts, ring buffer |
| Ch4 (Dispatch Hangs) | Dispatch progress heartbeat, CQ dump | tt-triage fast dispatch, Inspector |
| Ch5 (Multi-Chip/CCL) | Watcher (ETH link), fabric telemetry | tt-triage NOC checks, debug delay |
