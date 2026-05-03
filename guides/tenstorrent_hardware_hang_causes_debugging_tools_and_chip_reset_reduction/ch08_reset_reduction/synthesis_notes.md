# Chapter 8 Synthesis Notes

## Source Versions and Scores

| Version | Evaluator Score | Lines | Primary Strength |
|---------|----------------|-------|-----------------|
| V1 | 7.6/10 | 1080 | Thorough UMD warm reset docs, all 3 architecture paths, TensixSoftResetOptions enum with correct bit positions |
| V2 | 8.6/10 (60/70) | 1224 | Most quantitative (impact percentages), best proposal analysis framework, strongest narrative arc |
| V3 | 8.2/10 | 1441 | Best engineering roadmap (P0-P3, dependency graph, effort breakdowns), 20-item prevention checklist |
| V4 | 7.6/10 | 1177 | Unique NVIDIA/AMD industry comparison, concrete proposed API signatures |
| V5 | 7.3/10 | 742 | Deepest kernel driver (tt-kmd) perspective: reset_gen, reset_rwsem, noc_cleanup ioctl, PCIe hot reset, reset marker |

## Synthesis Strategy Executed

### Primary Foundation: V2

V2 was used as the primary structural and narrative foundation across all three content files due to its highest overall score and best quantitative rigor. Key elements retained:

- **Section 01:** The 5-level hierarchy structure, decision matrix, timing tables, all three architecture-specific reset paths, UBB Galaxy reset, multi-host distributed reset, IPC notification system, reset granularity analysis
- **Section 02:** The prevention-practices-with-impact-estimates structure, watcher configuration table, timeout wrapper code pattern, aggregate impact summary table
- **Section 03:** The proposal structure (current gap, categories addressed, impact estimate, implementation outline), the impact matrix at the end, priority recommendation

### V1 Contributions Integrated

V1 provided the most detailed UMD-level code excerpts, which were integrated into Section 01:

- The full `TensixSoftResetOptions` enum with exact bit positions (bit 11 for BRISC, bit 12 for TRISC0, bit 13 for TRISC1, bit 14 for TRISC2, bit 18 for NCRISC) -- V1's bit positions were correct while V3's simplified bit numbering (0-4) was incorrect
- The `invert_selected_options()` utility function
- Pre-defined combinations table (`ALL_TENSIX_SOFT_RESET`, `TENSIX_ASSERT_SOFT_RESET`, etc.)
- The complete timeout constants table from `timeouts.hpp`
- The `DeviceManager::~DeviceManager()` destructor code
- The ARM platform `is_arm_platform()` constexpr guard documentation
- The Wormhole refclk counter verification mechanism

### V5 Contributions Integrated (Kernel Driver Perspective)

V5 was the only version covering tt-kmd internals. These were added as a new Section 5 ("Kernel Driver Reset Safety Mechanisms") in `01_current_reset_mechanisms.md`:

- `reset_gen` generation counter with code showing `atomic_long_inc` and the stale-fd check returning `ENODEV`
- `reset_rwsem` read-write semaphore for reset serialization
- The `needs_hw_init` window where only specific ioctls are permitted
- Architecture-specific driver reset paths: `wormhole_reset()` (ARC message protocol, M3 watchdog fallback) and `blackhole_reset()` (PCIe timer interrupt registers at offsets 0x930/0x934)
- `pcie_hot_reset_and_restore_state()` with the `safe_pci_restore_state()` vendor ID guard
- The reset marker mechanism using `PCI_COMMAND_PARITY` bit
- The firmware watchdog system (WH: message 0xBC, BH: message 0xC1)
- The NOC cleanup ioctl with `tenstorrent_set_noc_cleanup` struct definition
- The `tt_cdev_release()` cleanup on fd close

### V3 Contributions Integrated (Engineering Roadmap)

V3 provided the most actionable engineering framework:

- **20-item prevention checklist** (Section 02, subsection 6) -- retained in full with chapter references updated to use final paths
- **Priority labels** (P0-P3) for all 12 proposals
- **Effort estimates** with day-level breakdowns for each proposal
- **Dependency graph** (ASCII art) showing inter-proposal relationships
- **Phased implementation roadmap** (Phase 1-4 with timeline estimates)
- **Expected Impact on Reset Frequency table** mapping scenarios through phases
- **12th proposal** (Unified Diagnostic Dashboard) added to the synthesis
- **Sub-proposal structure** (6A/6B/6C, 8A/8B/8C, 10A/10B/10C) for granular implementation paths
- **Environment configuration patterns** (development/CI/production) for watcher setup

### V4 Contributions (Industry Comparison)

V4's industry comparison was the most controversial element across evaluations. The evaluator flagged specific inaccuracies:
- "CUDA_RESET_CONTEXT" is not a real NVIDIA API name (the actual mechanism is `cuCtxDestroy()`/`cudaDeviceReset()`)
- AMD ROCm claims were oversimplified
- NVIDIA per-context error containment was overstated

**Decision:** The industry comparison was NOT incorporated as a standalone section or table. The evaluator's concerns about fabricated API names and oversimplifications were too significant. Instead, the general strategic framing (TT's model is currently more binary than graduated, and the path forward is toward graduated recovery) was adopted in the index.md introduction, which is defensible without citing specific external APIs.

## Factual Corrections Applied

### V3 Soft Reset Bit Positions (FIXED)

V3 stated soft reset bits as "Bit 0 = BRISC, Bit 1 = NCRISC, Bits 2-4 = TRISC0/1/2." This is incorrect. The synthesis uses V1's correct values throughout:
- BRISC = bit 11 (0x00800)
- TRISC0 = bit 12 (0x01000)
- TRISC1 = bit 13 (0x02000)
- TRISC2 = bit 14 (0x04000)
- NCRISC = bit 18 (0x40000)

### V3 L1 State Preservation Contradiction (FIXED)

V3's state-destruction table claimed Level 1 "Core-local cleared" for L1 SRAM, contradicting its own text and V1/V5 which correctly state L1 contents survive soft reset. The synthesis clearly states: "L1 of the target core is also NOT cleared by soft reset -- only the RISC-V processor state (registers, PC) is reset."

### V1 Line Number Citations (REMOVED)

V1 cited specific line numbers like `device.cpp:453` and `device_manager.cpp:665`. These were removed as multiple evaluators flagged them as potentially fabricated or rapidly stale due to code churn.

### V3 ARM Platform Code Fragment (FIXED)

V3 showed ARM detection using `#if defined(__ARM_ARCH)`. V1 correctly identified that the actual implementation uses `is_arm_platform()` constexpr check. The synthesis uses V1's correct description.

## Structural Decisions

### State Destruction Table Added to Section 01

A comprehensive state-destruction table was added to the reset hierarchy overview, inspired by V3's approach but corrected for the L1 preservation issue. This provides an at-a-glance reference that no single version had in fully correct form.

### Kernel Driver Section Added as New Section 5

V5's kernel driver content was significant enough to warrant its own section in `01_current_reset_mechanisms.md` rather than being sprinkled throughout the existing reset level descriptions. This keeps the UMD-level and KMD-level perspectives clearly separated while ensuring completeness.

### Proposal Numbering Reordered

The synthesis reordered proposals to follow the P0-P3 priority structure from V3, which is more useful for engineering planning than V2's rough impact ordering. The mapping:

| Synthesis # | Based On | V2 Equivalent |
|-------------|----------|---------------|
| 1 | V2 P1 + V3 P1 | Proposal 1 (Auto Classification) |
| 2 | V2 P3 + V3 P2 | Proposal 3 (Pre-Reset Snapshots) |
| 3 | V2 P5 + V3 P3 | Proposal 5 (Error Propagation) |
| 4 | V2 P2 + V3 P4 | Proposal 2 (Heartbeat) |
| 5 | V2 P6 + V3 P5 | Proposal 6 (Partial Reset) |
| 6 | V2 P10 + V3 P6 | Proposal 10 (Enhanced NOC Debug) |
| 7 | V2 P7 + V3 P7 | Proposal 7 (Firmware Watchdog) |
| 8 | V2 P8 + V3 P8 | Proposal 8 (Static Analysis) |
| 9 | V2 P4 + V3 P9 | Proposal 4 (Deterministic Replay) |
| 10 | V2 P9 + V3 P10 | Proposal 9 (Resilient CCL) |
| 11 | V2 P11 + V3 P11 | Proposal 11 (Checkpoint/Restart) |
| 12 | V3 P12 (unique) | N/A (V3-only proposal) |

### Cross-Chapter References

All cross-chapter references were updated to use `../chN_final/` paths with specific file names, following V3's approach of explicit per-file references rather than V1/V2's more generic chapter-level references.

## Line Counts

| File | Lines |
|------|-------|
| `index.md` | ~55 |
| `01_current_reset_mechanisms.md` | ~470 |
| `02_reducing_reset_frequency_and_resilience.md` | ~430 |
| `03_future_tooling_proposals.md` | ~490 |
| `synthesis_notes.md` | This file |

All content files are within the ~400-500 line target range.

## What Was Excluded

1. **V4's industry comparison tables** -- Evaluator 4 identified fabricated API names (`CUDA_RESET_CONTEXT`) and oversimplified claims. The risk of propagating inaccuracies outweighed the value of the comparative context.
2. **V1's specific line number citations** (e.g., `device.cpp:453`) -- Flagged as likely stale by multiple evaluators.
3. **V3's "I2C or SPI interface" claim for M3 communication** -- Evaluator flagged as speculative.
4. **V1's "ARC common prefix" qualification** for ARC message codes -- Other versions do not corroborate this.
5. **5-part hang scenario format** -- Per the task requirements, this chapter is analysis/recommendation, not hang scenarios.
6. **V5's developer-journey organization** for Section 02 -- While conceptually appealing, the category-based organization (matching V2/V3) provides better reference value for looking up specific practices.
