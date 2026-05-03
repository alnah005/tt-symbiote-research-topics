# Chapter 7: Diagnostic Workflows and End-to-End Triage

When a Tenstorrent device hangs, the developer faces a multi-dimensional diagnostic problem: the observable symptom is almost always "the host is blocked and the device is unresponsive," but the root cause could be any of the dozens of mechanisms documented in Chapters 2 through 5. The debugging tools cataloged in Chapter 6 are powerful individually, but their real value emerges when they are combined into coherent, step-by-step workflows that systematically narrow the problem from "something is hung" to a specific root cause and fix. This chapter provides those workflows. It is the integration chapter -- every tool, every hang category, and every diagnostic technique from the preceding six chapters is brought together into practical procedures a developer can follow with an active hang on their screen.

This chapter addresses research questions Q3 (debugging tools and workflows), Q4 (diagnostic procedures), and partially Q5 (automated debugging). The overarching goal is to provide systematic, repeatable diagnostic workflows that take a developer from "something is hung" to a specific root cause and fix.

Unlike Chapters 2-5, which are organized by hang mechanism and intended as reference material, this chapter is organized by **what you do** when you encounter a hang. The five files follow the natural debugging progression: initial triage (what to do in the first 60 seconds), category-specific diagnosis (once you know *what kind* of hang it is), narrowing and reproduction (when the root cause is not immediately obvious), reading tool output (how to interpret what the tools tell you), and distinguishing hardware from software bugs (the final frontier when everything else has been ruled out).

## Prerequisites

This chapter assumes familiarity with:

- **Chapter 1, [`01_what_is_a_hang.md`](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md)**: The RISC-V spin-loop model and the 5-part diagnostic format.
- **Chapter 1, [`02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md)**: The six primary blocking primitives and their waypoint codes (`CRBW`, `CWFW`, `NRBW`, `NWBW`, `NSW`, `NSMW`).
- **Chapter 1, [`03_hang_taxonomy.md`](../ch01_anatomy_of_a_hang/03_hang_taxonomy.md)**: The six-category classification (kernel, NOC, memory, dispatch, multi-chip, host-device).
- **Chapter 6**: All debugging tools (watcher, watcher_dump, DPrint, tt-triage, Tracy/NOC debug, debug delay). You do not need to have memorized Chapter 6 -- this chapter tells you when and how to use each tool.

Chapters 2-5 are referenced as needed for specific hang mechanisms. You can follow the workflows here and consult those chapters when the workflow directs you to a specific scenario.

## How to Use This Chapter

- **Active hang right now?** Start with [`01_initial_triage.md`](./01_initial_triage.md). Follow the decision tree to reach the correct procedure in [`02_diagnosing_by_hang_category.md`](./02_diagnosing_by_hang_category.md).
- **Active hang on a multi-chip system (T3K/Galaxy)?** Start with [`01_initial_triage.md`](./01_initial_triage.md), which includes multi-chip extensions at every step.
- **Know the hang category but cannot find the root cause?** Use the narrowing techniques in [`03_narrowing_and_reproducing.md`](./03_narrowing_and_reproducing.md).
- **Hang resolved but need to understand the root cause?** Read [`04_reading_watcher_and_triage_output.md`](./04_reading_watcher_and_triage_output.md) to interpret the diagnostic data you collected.
- **Suspect a hardware defect?** Follow the hardware vs. software discrimination procedure in [`05_distinguishing_hw_vs_sw_bugs.md`](./05_distinguishing_hw_vs_sw_bugs.md).
- **Intermittent hang?** Start with the intermittent reproduction strategies in [`03_narrowing_and_reproducing.md`](./03_narrowing_and_reproducing.md), then proceed through triage normally.
- **Have a specific error message on screen?** Use the error-message-to-action mapping tables in [`02_diagnosing_by_hang_category.md`](./02_diagnosing_by_hang_category.md).

## Chapter Contents

| # | File | Focus | Key Tools / Techniques |
|---|------|-------|------------------------|
| 1 | [`01_initial_triage.md`](./01_initial_triage.md) | First-response workflow: recognizing a hang, preserving state, automated triage, decision tree routing, multi-chip coordination | Watcher log, `tt-triage`, `watcher_dump`, DPrint, Inspector, `fabric_erisc_dumper.py` |
| 2 | [`02_diagnosing_by_hang_category.md`](./02_diagnosing_by_hang_category.md) | Category-specific diagnosis procedures with error-message-to-action mapping tables for kernel/CB, NOC, dispatch, memory, multi-chip, and semaphore hangs | Waypoint analysis, `NOC_DEBUG_DUMP`, `dump_fast_dispatch`, CB sanitization, `check_eth_status.py`, `fabric_erisc_dumper.py` |
| 3 | [`03_narrowing_and_reproducing.md`](./03_narrowing_and_reproducing.md) | Techniques for isolating and reproducing hangs: binary search, null kernels, slow dispatch, single-op isolation, multi-device mesh bisection, intermittent strategies | `Synchronize()` checkpoints, `null_kernels`, `kernels_early_return`, `TT_METAL_SLOW_DISPATCH_MODE`, trace/LightMetal replay, `hang_device` |
| 4 | [`04_reading_watcher_and_triage_output.md`](./04_reading_watcher_and_triage_output.md) | Interpreting diagnostic output: waypoint decoding, NOC violation fields, assert messages, ring buffer, kernel ID correlation, callstack PCs, cross-device correlation | `kernel_names.txt`, `kernel_elf_paths.txt`, `debug_sanitize_addr_msg_t`, `debug_assert_msg_t`, `riscv32-unknown-elf-objdump` |
| 5 | [`05_distinguishing_hw_vs_sw_bugs.md`](./05_distinguishing_hw_vs_sw_bugs.md) | Hardware vs. software bug identification: reproducibility signals, clean-state env vars, BH-specific options, kernel binary validation, Ethernet retrain, recovery without full reset | `TT_METAL_CLEAR_L1`, `TT_METAL_CLEAR_DRAM`, `enable_hw_cache_invalidation`, `disable_relaxed_memory_ordering`, `validate_kernel_binaries`, `skip_eth_cores_with_retrain` |

**Covers research questions:** Q3 (debugging tools and workflows), Q4 (diagnostic procedures), and partially Q5 (automated debugging).

## Workflow Pipeline

```
Hang detected --> 01 (triage) --> 02 (category diagnosis) --> 03 (narrowing) --> 04 (output reading) --> 05 (HW vs SW)
```

A developer may enter at any file depending on where they are in the debugging process.

## Key Environment Variables Quick Reference

The following environment variables appear throughout this chapter. They are organized by their role in the debugging workflow:

### Diagnostic Enablement

| Variable | Purpose |
|----------|---------|
| `TT_METAL_WATCHER=<N>` | Enable watcher with N-millisecond poll interval |
| `TT_METAL_DPRINT_CORES=<spec>` | Enable DPrint on specified cores |
| `TT_METAL_DPRINT_RISCVS=<spec>` | Which RISC-V harts to print (0=BRISC, 1=NCRISC, 2-4=TRISC0-2) |
| `TT_METAL_INSPECTOR=1` | Enable Inspector system (on by default) |
| `TT_METAL_RISCV_DEBUG_INFO=1` | Enable RISC-V debug info for callstacks |
| `TT_METAL_NOC_DEBUG_DUMP=1` | Track NOC debug state for missing barriers |
| `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` | Enable `ebreak`-based kernel asserts (low overhead, suitable for CI) |
| `TT_METAL_DISPATCH_PROGRESS_UPDATE_MS=<N>` | Periodic dispatch heartbeats (ms) |
| `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=<cmd>` | Auto-run command on dispatch timeout |
| `TT_METAL_OPERATION_TIMEOUT_SECONDS=<N>` | Host-side timeout threshold |

### Fault Isolation Controls

| Variable | Purpose |
|----------|---------|
| `TT_METAL_NULL_KERNELS=1` | Skip kernel execution entirely (env var shorthand) |
| `TT_METAL_KERNELS_EARLY_RETURN=1` | Kernels return immediately (same binary size, env var shorthand) |
| `TT_METAL_SLOW_DISPATCH_MODE=1` | Disable fast dispatch, use slow dispatch path |
| `TT_METAL_CLEAR_L1=1` | Zero-fill L1 on device init |
| `TT_METAL_CLEAR_DRAM=1` | Zero-fill DRAM on device init |
| `TT_METAL_VISIBLE_DEVICES=<list>` | Restrict which devices are visible for multi-device narrowing |

### Hardware-Specific Controls

| Variable | Purpose |
|----------|---------|
| `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1` | Skip ETH cores that have retrained |
| `TT_METAL_VALIDATE_PROGRAM_BINARIES=1` | Validate kernel binary integrity before execution |
| `TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1` | Enable HW cache invalidation (Blackhole) |
| `TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1` | Disable relaxed memory ordering (Blackhole) |
| `TT_METAL_ENABLE_GATHERING=1` | Enable instruction gathering (Blackhole) |

### Watcher Feature Granular Control

| Variable | Purpose |
|----------|---------|
| `TT_METAL_WATCHER_DUMP_ALL=1` | Dump all watcher data including unsafe state |
| `TT_METAL_WATCHER_NOINLINE=1` | Disable watcher inlining (reduces binary size, aids assert line-number accuracy) |
| `TT_METAL_WATCHER_DISABLE_ASSERT=1` | Disable watcher assert checking |
| `TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1` | Disable NOC address sanitization |
| `TT_METAL_WATCHER_DISABLE_WAYPOINT=1` | Disable waypoint tracking |
| `TT_METAL_WATCHER_DISABLE_RING_BUFFER=1` | Disable ring buffer |
| `TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION=1` | Enable linked transaction checking |
| `TT_METAL_READ_DEBUG_DELAY_CORES=<spec>` | Add read delays to specified cores for timing perturbation |
| `TT_METAL_WRITE_DEBUG_DELAY_CORES=<spec>` | Add write delays to specified cores |
| `TT_METAL_ATOMIC_DEBUG_DELAY_CORES=<spec>` | Add atomic operation delays to specified cores |

## Cross-References

- For detailed tool documentation, see [Chapter 6](../ch06_debugging_tools/)
- For architecture background needed to interpret results, see [Chapter 1](../ch01_anatomy_of_a_hang/)
- For kernel and NOC hang details, see [Chapter 2](../ch02_kernel_and_noc_hangs/)
- For memory-related hang details, see [Chapter 3](../ch03_memory_related_hangs/)
- For dispatch and host-device hang details, see [Chapter 4](../ch04_dispatch_and_host_device_hangs/)
- For multi-chip and CCL hang details, see [Chapter 5](../ch05_multi_chip_and_ccl_hangs/)

---

**Next:** [`01_initial_triage.md`](./01_initial_triage.md)
