# 7.5 Distinguishing Hardware vs. Software Bugs

[Previous: Reading Watcher and Triage Output](./04_reading_watcher_and_triage_output.md) | [Chapter Index](./index.md)

---

After exhausting the category-specific diagnosis (Section 7.2), narrowing techniques (Section 7.3), and watcher output interpretation (Section 7.4), a developer may face the final question: **is this a hardware defect or a software bug?** The answer determines the fix path -- a code change for software, an RMA or physical intervention for hardware. Misclassification in either direction wastes significant engineering time. On multi-chip systems (T3K, Galaxy), this distinction is especially critical because Ethernet link faults, individual chip defects, and backplane issues can produce symptoms identical to CCL protocol errors or kernel bugs.

This section provides a systematic 7-step discrimination procedure, clean-state isolation variables, architecture-specific diagnostics (Blackhole), Ethernet link health assessment, multi-chip hardware diagnosis, a recovery escalation ladder, and clear escalation criteria.

**Prerequisites:** [Section 7.1](./01_initial_triage.md) (all diagnostic data collected), [Section 7.2](./02_diagnosing_by_hang_category.md) (category-specific procedures attempted), [Section 7.3](./03_narrowing_and_reproducing.md) (minimal reproduction available), [Section 7.4](./04_reading_watcher_and_triage_output.md) (watcher output interpreted). [Chapter 1, `04_hang_causes_across_architectures.md`](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md) for architecture-specific failure modes.

---

## 7.5.1 Reproducibility as the Primary Signal

Reproducibility is the single most informative test for distinguishing hardware from software bugs. **Key principle:** If you can reproduce the hang deterministically (same inputs always produce the hang), the root cause is almost certainly software. Hardware defects typically manifest as intermittent failures because they depend on physical conditions (temperature, voltage, timing margins) that vary between runs.

| Behavior | Likely Cause | Reasoning |
|----------|-------------|-----------|
| **100% reproducible** on same device, same workload | Software bug | Hardware defects rarely produce perfectly deterministic failures |
| **100% reproducible** on ALL devices of the same type | Software bug (certain) | If every chip fails identically, the hardware is not at fault |
| **Intermittent** on one device, never on others | Hardware defect | The specific chip has a marginal component |
| **Intermittent** on all devices | Software race condition | Timing-dependent bug; use debug delays and timing perturbation ([Section 7.3](./03_narrowing_and_reproducing.md)) |
| **Reproducible** on one device only, different workloads | Hardware defect | If unrelated workloads fail on the same chip, the chip is suspect |
| **Reproducible** on one device only, one workload only | Unclear | Could be either; requires further investigation below |
| **Disappears after chip reset, returns next run** | Software bug | Likely stale state left from previous run |
| **Disappears after chip reset, does not return for many runs** | Possible HW fault | Transient thermal event or intermittent hardware |
| **Only under thermal stress or after long uptime** | Hardware defect | Thermal degradation, marginal silicon |
| **Only at specific device positions in multi-chip config** | Could be either | Topology-dependent SW bug or specific link/device HW issue |

### The Cross-Device Test

The most powerful hardware/software discriminator:

```
Single-Machine Procedure (multi-device):
  1. Take the minimal reproduction case from Section 7.3.
  2. Run on Device 0 only:  export TT_METAL_VISIBLE_DEVICES=0
  3. Run on Device 1 only:  export TT_METAL_VISIBLE_DEVICES=1
  4. Repeat for each device.
  5. Interpret:
     - Hangs on ALL devices:   SOFTWARE BUG.
     - Hangs on ONE device:    Likely HARDWARE DEFECT on that device.
     - Intermittent on all:    Try more iterations for statistical confidence.

Multi-Machine Procedure:
  1. Take the minimal reproduction case.
  2. Run on Machine A (the original). Confirm it hangs.
  3. Run on Machine B (same architecture, same software version).
     - Same architecture is critical: WH on WH, BH on BH.
     - Ideally same board revision and firmware version.
  4. Interpret:
     - Hangs on both machines:  SOFTWARE BUG.
     - Hangs only on Machine A: Likely HARDWARE DEFECT.
     - Intermittent on both:    Software race condition.
```

### Multi-Chip Reproducibility Tests

For multi-chip configurations (T3K, Galaxy), additional reproducibility tests are available:

1. **Same code, same mesh, same devices, same input:** If 100% reproducible, very likely software (unless a specific device always fails).
2. **Same code, different mesh layout:** If the hang moves when you remap devices (e.g., devices 0-7 vs. 8-15 on Galaxy), the bug follows the code (software).
3. **Same code, specific device always fails:** The device or its Ethernet links are suspect. Test that device in isolation.
4. **Swap two devices in the ring:** If the hang moves from position N to position M after swapping, the bug follows the device (hardware). If it stays at position N, the bug follows the code (software).

---

## 7.5.2 Clean-State Environment Variables

Software bugs can hide behind stale state from prior runs. Before concluding that a hang is a hardware issue, eliminate stale state as a possible cause.

### TT_METAL_CLEAR_L1

```bash
export TT_METAL_CLEAR_L1=1
```

**Effect:** Zeros all L1 SRAM on every core before each program execution.

**What it isolates:** Stale L1 data from prior programs -- corrupted CB metadata, stale semaphore values, leftover kernel data, corrupted runtime args from a killed process. If a hang disappears with `TT_METAL_CLEAR_L1=1`, the root cause is stale L1 state, likely from:
- A prior program that left dirty data in L1
- Missing initialization of semaphores or CB metadata
- Program cache serving a stale binary (see [Disabling Program Cache](#disabling-program-cache) below)

**Performance impact:** Significant -- zeroing all L1 adds milliseconds per program launch.

### TT_METAL_CLEAR_DRAM

```bash
export TT_METAL_CLEAR_DRAM=1
```

**Effect:** Zeros all DRAM before program execution.

**What it isolates:** Stale DRAM data -- corrupted buffer contents, stale trace replay data, leftover weight tensors, corrupted DRAM allocator metadata. If a hang disappears with this set, the root cause is DRAM state contamination.

**Performance impact:** Very significant -- DRAM is much larger than L1 and zeroing takes substantial time.

### TT_METAL_VALIDATE_PROGRAM_BINARIES

```bash
export TT_METAL_VALIDATE_PROGRAM_BINARIES=1
```

**Effect:** After loading kernel binaries into L1, reads them back and compares against the original ELF image. Reports any mismatches.

**What it isolates:** Corruption during kernel loading. Possible causes:
- **Software:** A bug in the dispatch pipeline that corrupts data during the PCIe-to-L1 transfer path, or a kernel that overwrites its own code.
- **Hardware:** A faulty NOC link, L1 SRAM bit flip, or PCIe data corruption.

**Interpreting binary validation failures:**

| Result | Interpretation |
|--------|---------------|
| All `.text` sections match | Binaries are intact; issue is in kernel logic or hardware |
| `.text` mismatch on specific core | Binary was corrupted on that core; likely SW (errant NOC write) or HW (PCIe DMA error) |
| `.text` mismatch on all cores | Systematic binary loading failure; likely dispatch/DMA bug |
| `.text` mismatch only on specific device | PCIe or device hardware issue on that device |
| Single-bit flip corruption pattern | Hardware likely (memory cell or signal integrity) |
| Block of wrong data (looks like another buffer) | Software likely (errant NOC write from another kernel) |

You can also use tt-triage for post-mortem validation:
```bash
python3 tools/tt-triage.py --run=check_binary_integrity
```

### Combined Clean-State Test

```bash
export TT_METAL_CLEAR_L1=1
export TT_METAL_CLEAR_DRAM=1
export TT_METAL_VALIDATE_PROGRAM_BINARIES=1
export TT_METAL_WATCHER=1
python3 your_model.py
```

If the hang **disappears**: software bug (stale state). Fix the initialization or cache management.
If the hang **persists**: not a stale-state issue. Proceed to architecture-specific checks.

---

## 7.5.3 Blackhole-Specific Options

Blackhole introduces hardware features (L1 data cache, relaxed memory ordering, instruction gathering) that can produce hangs not seen on Wormhole. **Key insight:** Both BH-specific variables below expose **software bugs**, not hardware defects. The hardware is working as designed; the software is not handling the hardware's relaxed semantics correctly.

### TT_METAL_ENABLE_HW_CACHE_INVALIDATION

```bash
export TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1
```

**What it does:** Forces explicit L1 data cache invalidation on every read. On Blackhole, the L1 data cache can serve stale data if a remote NOC write updated L1 after the local cache line was filled.

**What it isolates:** Missing cache invalidation bugs. If a hang disappears when cache invalidation is forced:
- The kernel reads L1 data that was updated by another core or by a NOC write
- Without invalidation, the local RISC reads stale cached data
- The stale data makes a spin-loop exit condition appear unsatisfied

**Example:** `cb_reserve_back` reads `pages_acked` which is updated by the consumer core. On BH without cache invalidation, the producer may read a stale value and spin forever.

**Fix:** Add explicit `invalidate_l1_cache()` calls before reads of data updated by other cores. The `cb_reserve_back` implementation already does this (see [Chapter 1, `02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md)), but custom kernel code may not.

### TT_METAL_DISABLE_RELAXED_MEM_ORDERING

```bash
export TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1
```

**What it does:** Disables Blackhole's relaxed memory ordering, forcing all memory operations to be strongly ordered.

**What it isolates:** Memory ordering bugs. Blackhole allows the hardware to reorder memory operations for performance. If kernel code depends on a specific write order being visible to other cores (e.g., writing data before incrementing a semaphore), relaxed ordering can make the semaphore visible before the data, causing the consumer to read uninitialized data.

**If the hang disappears:** The kernel has a memory ordering dependency that requires a memory barrier or a fence instruction. This is a software bug, but one that only manifests on BH.

### TT_METAL_ENABLE_GATHERING

```bash
export TT_METAL_ENABLE_GATHERING=1
```

**What it does:** Enables instruction gathering on Blackhole, which changes the instruction fetch behavior.

**What it isolates:** If behavior changes with gathering enabled/disabled, this points to a potential hardware issue with the instruction fetch path, or code that is sensitive to fetch timing.

### Inline Write Back-Pressure (Blackhole)

Blackhole's inline write to L1 uses all four memory ports and can cause NOC pipeline hangs under back-pressure. The known workaround is to write to stream registers (via `risc_attribs.h`) instead of inline L1 writes. If the hang occurs only on Blackhole, only with inline writes, and is fixed by the stream register workaround, this is a **known hardware limitation** (documented in [Chapter 1, `04_hang_causes_across_architectures.md`](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md)).

### Combined BH Diagnostic Matrix

Run the following test matrix for Blackhole-specific issues:

```bash
# Baseline: all defaults
python3 your_test.py

# Test 1: HW cache invalidation
TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1 python3 your_test.py

# Test 2: Strict memory ordering
TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1 python3 your_test.py

# Test 3: Both
TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1 TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1 python3 your_test.py
```

| Baseline | +Cache Inv | +Strict Order | +Both | Likely Cause |
|----------|-----------|---------------|-------|-------------|
| Hang | OK | Hang | OK | Cache coherency SW bug (missing invalidation) |
| Hang | Hang | OK | OK | Memory ordering SW bug (missing barrier) |
| Hang | OK | OK | OK | Either cache or ordering SW bug |
| Hang | Hang | Hang | Hang | Not cache/ordering related; proceed to next step |

---

## 7.5.4 Ethernet Link Diagnostics (Multi-Chip)

For multi-chip configurations (N300, T3K, Galaxy), Ethernet link instability is a common hardware-related issue that can cause hangs looking exactly like software bugs.

### Checking ETH Health

```bash
python3 tools/tt-triage.py --run=check_eth_status --dev=0
```

| Indicator | Value | Meaning |
|-----------|-------|---------|
| Heartbeat | False | ETH core RISC-V is not running (firmware crash or hardware failure) |
| Port Status | Down | Physical link is down |
| Retrain Count | > 0 | Link has retrained at least once (potential data loss) |
| RX Link Up | Down | Receive path is not active |
| Mailbox | 0xCA11xxxx | Pending message that was never processed |

### Retrain Count Interpretation

A non-zero retrain count is a **strong indicator of hardware issues** (cable, connector, or PHY). The retrain count is maintained by firmware and incremented each time the PHY retrains the link.

**Key facts about retrains:**
- A retrain does NOT guarantee data loss, but in-flight transactions may be lost
- A retrain during active data transfer will cause the software protocol to deadlock (sender waits for ack that was lost, receiver waits for data that was lost)
- Multiple retrains suggest an ongoing physical-layer problem

**Watcher retraining events:** The watcher tracks retraining events via `logical_core_to_eth_link_retraining_count`. Check the watcher log for:
```
ETH core (x,y): link_retraining_count=N
```
A non-zero count indicates the link has been retrained at least once since device initialization. The count is reported in the watcher destructor output:
```
Device 0 Ethernet Core (1,6) retraining events: 3
```

**Interpretation guidelines:**
- 0 events: Link is stable. Ethernet is not the issue.
- 1-5 events: Link is marginally stable. May be the root cause of intermittent hangs.
- >5 events: Link is unstable. Very likely the root cause.

### Using TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN

```bash
export TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1
```

This tells the runtime to exclude Ethernet cores that have experienced retraining from the active core set. If the hang disappears, the retrained link was the root cause.

**Important:** This is a **workaround**, not a fix. The physical link issue should be investigated:
- Check cable seating and condition
- Try a different cable
- Try the same cable on a different port pair
- If the problem follows the cable, replace it
- If the problem stays with the port, the device may need RMA
- Check for thermal issues near the ETH PHY

### Focused Link Testing

To test a specific Ethernet link in isolation:

1. Identify the two devices connected by the suspect link and their Ethernet core coordinates.
2. Create a 2-device submesh containing only those two devices.
3. Run a simple CCL operation (e.g., `all_gather` with a small tensor) on the 2-device submesh.
4. Repeat many times (1000+) with watcher enabled to detect intermittent failures.
5. If the 2-device test fails but each device passes single-device tests individually, the Ethernet link between them is the problem.

### When Ethernet Issues Are Software vs. Hardware

| Observation | Likely Cause |
|-------------|-------------|
| High retrain counts on a specific link regardless of workload | Hardware (cable or port) |
| High retrain counts only under heavy traffic | Could be either; high bandwidth may expose marginal links |
| Link down immediately after training | Hardware (link never came up properly) |
| Link goes down during CCL operation | Could be either; check if CCL code has a bug causing excessive traffic |

### Blackhole ETH-Specific Firmware Addresses

For manual ETH debugging on Blackhole, the key firmware addresses are:
- Port status: `0x7CC04` (1=Up, 2=Down, 3=Unused)
- Retrain count: `0x7CE00`
- RX link up: `0x7CE04`
- Heartbeat: `0x7CC70`
- Mailbox slots: `0x7D000` (4 slots, 4 bytes each)

Reference: [Chapter 5, `01_ethernet_and_fabric_fundamentals.md`](../ch05_multi_chip_and_ccl_hangs/01_ethernet_and_fabric_fundamentals.md), [Chapter 6, `01_watcher_system.md`](../ch06_debugging_tools/01_watcher_system.md).

---

## 7.5.5 Temperature, ECC, and Environmental Checks

### Temperature and Thermal Events

Thermal throttling can cause hangs by slowing down or halting chip operations mid-execution.

```bash
tt-smi   # Check temperature column for each device
```

| Architecture | Throttle Threshold (approx.) | Shutdown Threshold (approx.) |
|-------------|------------------------------|------------------------------|
| Wormhole B0 | ~90C | ~100C |
| Blackhole | ~85C | ~95C |

**Thermal diagnosis:**
1. Run `tt-smi` during the workload in a separate terminal. Record temperatures.
2. If any device reaches the throttle threshold during the hang period, thermal throttling may be the cause.
3. On Galaxy systems, check temperatures across all hosts. One host's cooling system may be inadequate, causing its 8 devices to throttle while the others continue at full speed. This creates a speed mismatch that can trigger CCL timeouts.

**Thermal vs. software:**
- Hang only after sustained high-power workloads (e.g., after 5 minutes of continuous inference) --> likely thermal.
- Hang occurs immediately on the first operation --> not thermal.
- Hang correlates with ambient temperature (more frequent in summer) --> thermal/hardware.

### ECC Errors

DRAM and SRAM ECC errors indicate hardware-level memory corruption.

```bash
tt-smi   # Check for ECC error counters per device
```

| Error Type | Meaning | Action |
|-----------|---------|--------|
| Correctable ECC (CE) | Single-bit error detected and corrected. A few CEs are normal over long periods. Many CEs (>100) suggest degrading DRAM. | Monitor; escalate if accumulating rapidly |
| Uncorrectable ECC (UE) | Multi-bit error that could not be corrected. Data is corrupted. | **Strong hardware failure signal.** Even a single UE warrants investigation. |

**ECC and hangs:** An uncorrectable ECC error in DRAM can corrupt kernel code (if ELF is DRAM-resident), weight data (corrupted weights used as pointers), or CB configuration (corrupted base addresses cause out-of-bounds NOC transactions).

### Power and PCIe Integrity

**Power:** If the board is not receiving adequate power (undersized PSU, failing voltage regulator), the chip may exhibit intermittent failures that look like SRAM or logic errors.

**PCIe Link Quality:**
```bash
dmesg | grep -i -E "pcie|aer|error|tenstorrent|tt_" | tail -50
```

Look for:
- **AER (Advanced Error Reporting) entries:** PCIe-level errors indicating link quality issues.
- **Device removal events:** The OS detected the device disappearing from the PCIe bus.
- **Driver error messages:** The TT kernel driver reported an error.
- **Link speed/width degradation:** `lspci -vv` -- check if the device is negotiated at the expected PCIe gen and lane width.

### ARC Processor Health

```bash
python3 tools/tt-triage.py --run=check_arc
```

| Indicator | Expected | Problem Signal |
|-----------|----------|---------------|
| ARC heartbeat rate | 9-11 Hz | Outside this range: ARC is malfunctioning |
| ARC postcode | `0xc0de____` | Different value: ARC did not boot correctly |

---

## 7.5.6 Disabling Program Cache

The program cache stores compiled kernel binaries to avoid recompilation. If the cache serves a stale or corrupted binary, hangs can occur that disappear when the workload is run for the first time (cache miss = fresh compile).

```python
# Disable program cache to force recompilation every time
device.set_program_cache_misses_allowed(True)
```

Or clear the cache entirely:
```bash
rm -rf ~/.cache/tt-metal-cache/
```

Or force fresh JIT compilation:
```bash
export TT_METAL_FORCE_JIT_COMPILE=1
```

**Interpretation:**
- Hang **disappears** when cache is cleared or disabled: stale cache entry. The program was recompiled or its layout changed but the cache served the old binary. This is a software bug in cache invalidation.
- Hang **persists** regardless of cache state: not cache-related.

Reference: The program cache stores ELFs at paths shown in `kernel_elf_paths.txt` (see [Section 7.4](./04_reading_watcher_and_triage_output.md)).

---

## 7.5.7 The Hardware vs. Software Discrimination Procedure

This is the master procedure. Follow it step by step when you cannot determine the cause from category-specific diagnosis alone.

```
SYSTEMATIC DISCRIMINATION PROCEDURE (7 Steps)
================================================

Step 1: Reproducibility Assessment
  +-- 100% reproducible on multiple devices? --> SOFTWARE. Fix the code.
  +-- Only on one device? --> Continue to Step 2.
  +-- Intermittent? --> Try timing perturbation (Section 7.3) first.
                        If still intermittent on one device only, continue to Step 2.

Step 2: Eliminate Stale State
  export TT_METAL_CLEAR_L1=1
  export TT_METAL_CLEAR_DRAM=1
  +-- Hang disappears? --> SOFTWARE (stale state). Fix initialization.
  +-- Hang persists? --> Continue to Step 3.

Step 3: Architecture-Specific Isolation (Blackhole only)
  TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1
  TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1
  +-- Hang disappears? --> SOFTWARE (BH-specific ordering or caching bug).
  +-- Hang persists? --> Continue to Step 4.

Step 4: Binary Integrity
  TT_METAL_VALIDATE_PROGRAM_BINARIES=1
  # Or: python3 tools/tt-triage.py --run=check_binary_integrity
  +-- Validation fails? --> Could be HW (SRAM/PCIe) or SW (loading bug).
  |                         Try on another device to distinguish.
  +-- Validation passes? --> Continue to Step 5.

Step 5: Eliminate Program Cache
  rm -rf ~/.cache/tt-metal-cache/
  # Or: TT_METAL_FORCE_JIT_COMPILE=1
  # Re-run
  +-- Hang disappears? --> SOFTWARE (stale cache). Fix cache invalidation.
  +-- Hang persists? --> Continue to Step 6.

Step 6: Multi-Chip Isolation (if applicable)
  TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1
  # Check watcher for link retraining counts
  # Check check_eth_status for all devices
  +-- Hang disappears? --> HARDWARE (unstable Ethernet link).
  +-- Hang persists? --> Continue to Step 7.

Step 7: Environmental Checks
  - tt-smi: ECC errors? Temperature? Power?
  - dmesg: PCIe AER errors? Device removal?
  - check_arc: ARC heartbeat and postcode normal?
  - Does the hang occur with DIFFERENT workloads on the SAME device?
  +-- Environmental anomaly found? --> HARDWARE.
  +-- Everything normal? --> The most likely remaining option is a
      subtle software bug not yet exposed by the debugging tools.
      Consider:
        - Static analysis of the kernel code
        - Peer review of the NOC transaction patterns
        - Filing a bug with all diagnostic data attached
```

### Multi-Chip Hardware Diagnosis Decision Tree

For multi-chip systems, use this supplementary decision tree after the main procedure reaches Step 6-7:

```
Is the hang reproducible on the same devices?
  |
  +-- YES (100% reproducible)
  |   |
  |   Does it reproduce on a DIFFERENT set of devices
  |   (same code, same input, different physical chips)?
  |     |
  |     +-- YES --> Software bug. The bug follows the code, not the hardware.
  |     |
  |     +-- NO (only fails on specific device set)
  |          |
  |          Does it reproduce on a single device in isolation?
  |            |
  |            +-- YES --> Software bug triggered by device-specific topology
  |            |           (e.g., harvested rows affecting NOC coordinates)
  |            |
  |            +-- NO (multi-device required)
  |                 |
  |                 Does it reproduce on a 2-device submesh containing
  |                 the suspected pair?
  |                   |
  |                   +-- YES
  |                   |   |
  |                   |   Check Ethernet link health (retraining counters).
  |                   |     |
  |                   |     +-- Retraining events > 0 --> Ethernet HW fault.
  |                   |     |                              Replace cable/connector.
  |                   |     |
  |                   |     +-- Retraining events = 0 --> Likely CCL software bug
  |                   |                                   triggered by this specific
  |                   |                                   device pair's configuration.
  |                   |
  |                   +-- NO (requires more than 2 devices)
  |                        |
  |                        Likely a ring-topology protocol bug or a Galaxy-level
  |                        routing issue. Use mesh bisection (Section 7.3)
  |                        to find the minimum device count that triggers the hang.
  |
  +-- NO (intermittent)
      |
      Does it occur more frequently on specific devices?
        |
        +-- YES --> Test those devices individually:
        |           - Check temperature (tt-smi)
        |           - Check ECC errors (tt-smi)
        |           - Check Ethernet retraining (watcher)
        |           - Run kernel binary validation
        |           If any hardware metric is abnormal: HW fault.
        |           If all metrics are clean: timing-sensitive SW bug
        |           amplified by that device's specific timing characteristics.
        |
        +-- NO (no device preference)
            |
            Timing-sensitive SW bug (race condition).
            Use debug delay (Section 7.3) to amplify
            the timing window and make it reproducible.
```

---

## 7.5.8 Known Software Bug Indicators (Always Software)

Before concluding a hardware defect, verify that none of these definitive software signals are present:

1. **Missing barrier asserts:** Watcher assert types `DebugAssertNCriscNOCReadsFlushedTripped` (code 4), `DebugAssertNCriscNOCNonpostedWritesSentTripped` (5), `DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped` (6), and `DebugAssertNCriscNOCPostedWritesSentTripped` (7) are **always** software bugs.
2. **CB protocol violations:** If watcher shows `CRBW`/`CWFW` stuck waypoints with mismatched tile counts, these are **always** software bugs (verify `num_tiles` divides CB size, producer/consumer use same tile size, cumulative tile counts are correct).
3. **Runtime argument out-of-bounds:** Assert types `DebugAssertRtaOutOfBounds` (8) or `DebugAssertCrtaOutOfBounds` (9) -- **always** software.
4. **NOC address computation errors:** Watcher NOC sanitize errors pointing to clearly invalid addresses (address 0, mailbox region, beyond L1 size) -- **always** software.
5. **Stale state from previous programs:** Hang only on second or subsequent iteration, fixed by `TT_METAL_CLEAR_L1=1` -- **always** software.
6. **NOC debug dump reports missing barrier:** `TT_METAL_NOC_DEBUG_DUMP=1` detects unflushed writes or missing barriers -- **always** software.
7. **Stack overflow detected:** Watcher stack overflow assertion -- **always** software.

---

## 7.5.9 Recovery Escalation Ladder

Before resorting to a full chip reset or host reboot, try these recovery steps in order. Each level preserves more diagnostic state than the next.

```
RECOVERY ESCALATION LADDER
============================

Level 1: Kill the host process
  - If the device is in a soft hang (firmware spin-loop), killing the host
    process allows the runtime cleanup to attempt graceful shutdown.
  - Use Device::close(skip_synchronize=True) to skip the synchronize call
    that would itself hang.
  - Preserves: all device state for post-mortem analysis.

Level 2: Tensix soft reset (per-core)
  - Reset individual cores using the soft reset register.
  - This is per-core and does not affect other cores.
  - The host API does not currently expose per-core reset, so this
    requires UMD-level access.
  - Preserves: state on non-reset cores.

Level 3: tt-smi warm reset
  - tt-smi --reset
  - Reinitializes the chip but preserves the PCIe link.
  - Most hangs are recoverable with a warm reset.
  - Preserves: PCIe link, host-side state.

Level 4: Full host reboot
  - Required only when the PCIe link is degraded, the ARC processor
    is unresponsive, or device removal events appear in dmesg.
  - Preserves: nothing (full power cycle).

TRACK: If warm resets > 1/day in production, investigate root
cause before it escalates to requiring full reboots.
```

---

## 7.5.10 War Story: The Hang That Looked Like Hardware But Was Software

**Symptom:** A T3000 training run hangs every 8-12 hours. Only happens on one specific machine (Machine 7 of 20 in a cluster).

**Initial Assessment:** Hardware suspect because it is machine-specific.

**Investigation:**
1. tt-triage shows core (5,3) on device 2 stuck with CB0 active. NOC write in-flight to DRAM.
2. `check_noc_status` shows 1 unacknowledged non-posted write.
3. Binary integrity check: PASS. ETH health: all OK.
4. Ran the same workload on Machine 1: no hang after 48 hours.
5. Swapped devices between Machine 7 and Machine 1 (physically moved the cards): **Machine 7 still hangs**, Machine 1 still fine. This seems to confirm hardware.

**The Plot Twist:** A closer look at Machine 7's configuration revealed that it had a slightly different kernel build cache from a prior failed software update. The kernel cache path includes a hash of build flags. Machine 7's cached kernels were compiled with an older compiler that generated slightly different NOC address calculations. The stale cache was not invalidated because the build system's hash did not include the compiler version at that time.

**Resolution:** `rm -rf ~/.cache/tt-metal-cache && rebuild`. The hang disappeared.

**Lesson:** Machine-specific does not automatically mean hardware. Configuration drift, stale build caches, different library versions, and other environmental differences between machines can make software bugs look unit-specific. Always check for configuration differences before concluding hardware defect.

---

## 7.5.11 War Story: The Hang That Looked Like Software But Was Hardware

**Symptom:** A single-chip model inference run hangs deterministically on the 5th forward pass when using a specific matmul configuration (2048x2048, TILE layout, on core grid 8x8).

**Initial Assessment:** Software suspect because it is deterministic and correlates with a specific operation.

**Investigation:**
1. The hang reproduces with watcher: no NOC sanitize error, no assert. Core (6,7) stuck at waypoint `RM` indefinitely.
2. `check_noc_status` shows core (6,7) has `noc_reads_num_issued = 48`, `NIU_MST_RD_RESP_RECEIVED = 47`. One read never got a response.
3. The read targets DRAM bank 3.
4. **Key test:** Ran with core grid 7x7 (excluding row 7): no hang. Ran with core grid 8x7 (excluding column 7): no hang. Only core (6,7) triggers the hang.
5. Ran a different kernel on core (6,7) that issues reads to the same DRAM bank: also hangs.
6. Ran the same matmul on a different device on the same machine: no hang.

**Root Cause:** Core (6,7) had a defective NOC router that intermittently failed to forward read responses from DRAM bank 3. The defect was position-specific (always the same physical core) and target-specific (always DRAM bank 3's routing path).

**Resolution:** RMA the device. Short-term workaround: exclude core (6,7) from the available core grid via `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`.

**Lesson:** When a hang consistently involves the same physical core regardless of which kernel runs on it, and particularly when the NOC transaction counters show a simple mismatch (one transaction never completed), suspect hardware. The deterministic nature was misleading -- it was deterministic because the same core and same routing path were used every time.

---

## 7.5.12 Clean State Reveals Stale Semaphore Initialization

**(1) Symptom:** Intermittent hang at `NSW` waypoint. The hang occurs on the second run of a model but never on the first run after device initialization.

**(2) Root Cause:** The kernel uses an L1 semaphore for inter-core synchronization but does not initialize it to 0 before use. On the first run, L1 is zero-initialized by the device initialization code. On the second run, the semaphore retains its final value from the first run (e.g., value=4). The kernel calls `noc_semaphore_wait(sem_addr, 4)`, but the semaphore is already 4 from the prior run, so the wait immediately passes. However, on a slightly different code path, the kernel calls `noc_semaphore_wait(sem_addr, 0)` expecting the semaphore to have been reset -- it reads 4 and waits forever for 0, which will never arrive because the signaling core only increments.

**(3) Diagnosis Steps:**
1. Set `TT_METAL_CLEAR_L1=1` and re-run. If the hang disappears, stale L1 is confirmed.
2. Check the kernel source for `noc_semaphore_set` calls -- is the semaphore explicitly initialized before use?
3. Check if the hang is second-run-only by restarting the device between runs.

**(4) Fix:** Add `noc_semaphore_set(sem_addr, 0)` at the beginning of the kernel, before any core begins using the semaphore. Alternatively, use the host-side `SetRuntimeArgs` to pass the initial semaphore value and `noc_semaphore_set` it at runtime.

**(5) Prevention:**
- Always explicitly initialize semaphores in kernel code, never assume L1 is zero.
- Use `TT_METAL_CLEAR_L1=1` in CI tests to catch initialization bugs early.
- Follow the semaphore protocol guidelines in [Chapter 2, `04_noc_barrier_and_semaphore_hangs.md`](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md).

---

## 7.5.13 Blackhole Relaxed Memory Ordering Exposes Missing Barrier

**(1) Symptom:** A kernel that works reliably on Wormhole hangs intermittently on Blackhole at `CWFW` waypoint (waiting for CB data). The hang is more frequent under high system load.

**(2) Root Cause:** The reader kernel writes tile data to a CB via NOC, then increments a semaphore on the compute core to signal data availability. On Wormhole, strong memory ordering ensures the CB data is visible before the semaphore increment. On Blackhole, relaxed ordering allows the semaphore increment to become visible before the CB data write completes. The compute core sees the semaphore, calls `cb_wait_front`, and reads incomplete/stale data from the CB -- or, in some cases, the CB metadata itself is not yet updated, causing `cb_wait_front` to see 0 available tiles and spin.

**(3) Diagnosis Steps:**
1. Run on Blackhole with `TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1`. If the hang disappears, relaxed ordering is confirmed as the trigger.
2. Check the kernel source for the pattern: `noc_async_write` (CB data) followed by `noc_semaphore_inc` (signal) **without** an intervening `noc_async_write_barrier`.
3. Verify the hang does not occur on Wormhole with the same binary.

**(4) Fix:** Insert `noc_async_write_barrier()` between the data write and the semaphore increment:

```c++
// WRONG: Semaphore may be visible before data on Blackhole
noc_async_write(src_l1_addr, dest_noc_addr, data_size);
noc_semaphore_inc(remote_sem_addr, 1);

// CORRECT: Barrier ensures data is committed before signal
noc_async_write(src_l1_addr, dest_noc_addr, data_size);
noc_async_write_barrier();  // <-- Forces data write to complete
noc_semaphore_inc(remote_sem_addr, 1);
```

**(5) Prevention:**
- On Blackhole, every signaling pattern (write data then signal via semaphore) requires a barrier between the data write and the signal.
- Use `TT_METAL_NOC_DEBUG_DUMP=1` to automatically detect missing barriers (see [Chapter 6, `05_profiler_tracy_and_noc_debug.md`](../ch06_debugging_tools/05_profiler_tracy_and_noc_debug.md)).
- Test all new kernels on Blackhole with `TT_METAL_DISABLE_RELAXED_MEM_ORDERING=0` (the default) to ensure correct barrier placement.
- Reference: [Chapter 1, `04_hang_causes_across_architectures.md`](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md) (Blackhole section).

---

## 7.5.14 When Both Are Possible: Software Bugs Triggered by Hardware Conditions

Some hangs are caused by a software bug that is only triggered by a specific hardware condition. In these cases, the root cause is software (the code should handle the hardware condition gracefully), but the trigger is hardware. The fix is a code change, but the diagnostic path must include hardware inspection to identify the trigger.

Examples:
- A kernel that works on non-harvested Wormhole chips but hangs on harvested chips due to incorrect NOC coordinate calculation.
- A CCL operation that works on stable Ethernet links but hangs when a link briefly retrains (the code does not handle the transient data loss).
- A memory-mapped region that happens to overlap with an ECC-uncorrectable DRAM address, causing corrupted data that triggers an invalid NOC transaction.

---

## Hardware Issue Escalation Checklist

Before filing a hardware issue, confirm all of the following:

- [ ] The hang reproduces on the same machine across multiple clean reboots
- [ ] The hang does NOT reproduce on a different machine with the same software
- [ ] All software-side clean-state variables have been tested (`CLEAR_L1`, `CLEAR_DRAM`, `VALIDATE_PROGRAM_BINARIES`)
- [ ] Program cache has been cleared and retested
- [ ] BH-specific options have been tested (if applicable): cache invalidation, relaxed ordering, gathering
- [ ] ETH link health has been checked (for multi-chip issues)
- [ ] The tt-triage output has been captured for the hardware team
- [ ] The watcher log has been captured (if watcher was enabled)
- [ ] `dmesg` output has been checked for PCIe errors or device removal events
- [ ] `check_arc` has been run for ARC processor health
- [ ] `tt-smi` output has been captured (temperature, power, ECC)

**Include in the bug report:**
- Chip serial number and board ID
- `tt-smi` output (temperature, power, ECC errors)
- `dmesg` PCIe/AER entries
- tt-triage full output
- Watcher log (if available)
- Reproduction steps and failure rate
- Environment: firmware version, driver version, tt-metal commit hash

---

## Printable Quick-Reference Checklist

```
HW vs SW DETERMINATION CHECKLIST
===================================

STEP 1: REPRODUCIBILITY ANALYSIS
[ ] Run test 100 times on same chip, same inputs
    100% fail rate? --> strong SW signal
    <20% fail rate? --> investigate both
[ ] Run test on different chip
    Fails on both?   --> likely SW
    Fails on one?    --> likely HW on that chip
[ ] Run test with different input data
    Data-dependent?  --> SW bug
    Data-independent? --> could be either

STEP 2: CLEAN-STATE ISOLATION
[ ] TT_METAL_CLEAR_L1=1
    Fixed? --> stale L1 state (SW)
[ ] TT_METAL_CLEAR_DRAM=1
    Fixed? --> stale DRAM state (SW)
[ ] TT_METAL_VALIDATE_PROGRAM_BINARIES=1
    Mismatch? --> loading corruption (investigate PCIe vs. SW)
[ ] Disable program cache / TT_METAL_FORCE_JIT_COMPILE=1
    Fixed? --> stale cache entry (SW)

STEP 3: BLACKHOLE-SPECIFIC (if BH chip)
[ ] TT_METAL_ENABLE_HW_CACHE_INVALIDATION=1
    Fixed? --> missing cache invalidation (SW)
[ ] TT_METAL_DISABLE_RELAXED_MEM_ORDERING=1
    Fixed? --> missing memory barrier (SW)
[ ] TT_METAL_ENABLE_GATHERING=1
    Behavior changes? --> instruction fetch issue (investigate)
[ ] Check for inline write back-pressure
    Fixed by stream register workaround? --> known HW limitation

STEP 4: MULTI-CHIP (if applicable)
[ ] Check Ethernet link retrain count
    Count > 0? --> unstable link (HW)
[ ] TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1
    Fixed? --> link issue (HW)
[ ] Run fabric deadlock stability tests
[ ] Test 2-device submesh for focused link testing

STEP 5: HARDWARE DIAGNOSTICS
[ ] tt-smi: ECC errors? (especially uncorrectable)
[ ] tt-smi: temperature within limits?
[ ] tt-smi: power readings normal?
[ ] dmesg: PCIe AER errors?
[ ] dmesg: device removal events?
[ ] check_arc: ARC heartbeat 9-11 Hz? Postcode 0xc0de____?
[ ] lspci -vv: PCIe link speed/width as expected?

DECISION:
  All SW indicators + no HW indicators --> SOFTWARE BUG
  Mixed indicators                     --> INVESTIGATE FURTHER (see Section 7.5.14)
  Strong HW indicator + no SW fix      --> FILE HW BUG (see escalation checklist)
```

---

## Summary: Software Bug Indicators vs. Hardware Defect Indicators

| Indicator | Points to Software | Points to Hardware |
|-----------|-------------------|-------------------|
| Reproducible on multiple devices | Yes | -- |
| Disappears with `CLEAR_L1`/`CLEAR_DRAM` | Yes | -- |
| Disappears with `DISABLE_RELAXED_MEM_ORDERING` (BH) | Yes | -- |
| Disappears with `ENABLE_HW_CACHE_INVALIDATION` (BH) | Yes | -- |
| Disappears when program cache is cleared | Yes | -- |
| Watcher NOC sanitize error (invalid address) | Yes | -- |
| Watcher assert (codes 3-9) | Yes | -- |
| NOC debug dump: missing barrier | Yes | -- |
| Stack overflow detected | Yes | -- |
| Fix by adding explicit barrier | Yes | -- |
| Same code works on other chips | -- | Yes |
| ECC errors in `tt-smi` (especially UE) | -- | Yes |
| ARC heartbeat out of 9-11 Hz range | -- | Yes |
| Temperature near throttling threshold | -- | Yes |
| PCIe AER errors in `dmesg` | -- | Yes |
| Intermittent across unrelated workloads on one device | -- | Yes |
| `validate_kernel_binaries` fails randomly (bit-flip pattern) | -- | Yes |
| Link retraining count increasing | -- | Yes (Ethernet PHY) |
| Core-specific hang regardless of kernel | -- | Yes (NOC router defect) |
| Hang appeared after physical handling of board | -- | Yes |
| No fix found after exhaustive debugging | -- | Suspect hardware |

---

**End of Chapter 7.** Return to [Chapter Index](./index.md).
