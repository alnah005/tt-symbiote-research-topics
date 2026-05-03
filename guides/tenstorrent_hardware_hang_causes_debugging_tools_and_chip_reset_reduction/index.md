# Tenstorrent Hardware Hang Causes, Debugging Tools, and Chip Reset Reduction

A comprehensive reference guide covering every known hang mechanism in the Tenstorrent software/hardware stack, the tools available to diagnose them, systematic workflows for resolving them, and strategies for reducing the frequency of chip resets. Covers Grayskull, Wormhole, Blackhole, and Quasar architectures across single-chip, N300, T3K (8-chip), and Galaxy (32+ chip) configurations.

## Research Questions

| # | Question | Primary Chapters |
|---|----------|-----------------|
| Q1 | What are all the root causes of hardware hangs? | Ch 1, 2, 3, 4, 5 |
| Q2 | What are the multi-chip and CCL-specific hang causes? | Ch 5 |
| Q3 | What debugging tools and workflows exist? | Ch 6, 7 |
| Q4 | What are the systematic diagnostic procedures? | Ch 7 |
| Q5 | How can debugging be automated? | Ch 6, 7, 8 |
| Q6 | What are the architecture-specific differences (GS/WH/BH/Quasar)? | Ch 1, 2, 3 |
| Q7 | What strategies reduce chip reset frequency? | Ch 8 |
| Q8 | What future tooling improvements are proposed? | Ch 8 |
| Q9 | How do T3K and Galaxy configurations differ? | Ch 5 |

## Chapters

| # | Directory | Title | Content Files | Scenarios |
|---|-----------|-------|---------------|-----------|
| 1 | [`ch01_anatomy_of_a_hang/`](./ch01_anatomy_of_a_hang/index.md) | Anatomy of a Hang | 4 | Foundational taxonomy |
| 2 | [`ch02_kernel_and_noc_hangs/`](./ch02_kernel_and_noc_hangs/index.md) | Kernel and NOC Hangs | 4 | 44 scenarios |
| 3 | [`ch03_memory_related_hangs/`](./ch03_memory_related_hangs/index.md) | Memory-Related Hangs | 4 | 46 scenarios |
| 4 | [`ch04_dispatch_and_host_device_hangs/`](./ch04_dispatch_and_host_device_hangs/index.md) | Dispatch and Host-Device Hangs | 3 | 34 scenarios |
| 5 | [`ch05_multi_chip_and_ccl_hangs/`](./ch05_multi_chip_and_ccl_hangs/index.md) | Multi-Chip, CCL, and Fabric Hangs | 3 | 35 scenarios |
| 6 | [`ch06_debugging_tools/`](./ch06_debugging_tools/index.md) | Debugging Tools and Infrastructure | 6 | 22 scenarios |
| 7 | [`ch07_diagnostic_workflows/`](./ch07_diagnostic_workflows/index.md) | Diagnostic Workflows and End-to-End Triage | 5 | Workflow procedures |
| 8 | [`ch08_reset_reduction/`](./ch08_reset_reduction/index.md) | Reset Reduction, Resilience, and Future Improvements | 3 | 12 proposals |

**Total hang scenarios documented:** 180+ across Chapters 2-6.

## How to Use This Guide

- **Active hang on your screen:** Start with [Chapter 7](./ch07_diagnostic_workflows/index.md) for step-by-step triage workflows.
- **Know the hang category:** Go directly to the relevant chapter (Ch 2-5) for the specific scenario.
- **Setting up debugging tools:** Start with [Chapter 6](./ch06_debugging_tools/index.md) for tool configuration.
- **Reducing reset frequency:** Start with [Chapter 8](./ch08_reset_reduction/index.md) for prevention strategies.
- **New to Tenstorrent hang debugging:** Start with [Chapter 1](./ch01_anatomy_of_a_hang/index.md) for foundational concepts.

## Diagnostic Format

Every hang scenario in Chapters 2-6 follows the 5-part format:

1. **Symptom** -- Observable indicators (waypoints, watcher output, host behavior)
2. **Root Cause** -- Technical explanation of why the hang occurs
3. **Diagnosis Steps** -- Step-by-step procedure to confirm the root cause
4. **Fix** -- How to resolve the specific instance
5. **Prevention** -- How to avoid the hang in future code

## Key Environment Variables Quick Reference

| Variable | Purpose |
|----------|---------|
| `TT_METAL_WATCHER=1` | Enable watcher system (NOC sanitization, waypoints, assertions) |
| `TT_METAL_DPRINT_CORES` | Enable DPrint on specific cores |
| `TT_METAL_OPERATION_TIMEOUT_SECONDS` | Set dispatch operation timeout |
| `TT_METAL_SLOW_DISPATCH_MODE=1` | Use slow dispatch for isolation |
| `TT_METAL_CLEAR_L1=1` | Clear L1 between runs (detect stale state) |
| `TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1` | Route around unstable Ethernet links |
| `TT_METAL_ENABLE_REMOTE_CHIP=1` | Enable remote chip access |
