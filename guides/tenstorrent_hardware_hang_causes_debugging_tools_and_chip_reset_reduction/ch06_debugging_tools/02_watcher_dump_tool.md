# 6.2 Standalone watcher_dump Post-Mortem Tool

## Summary

The `watcher_dump` tool is a standalone binary that performs post-mortem analysis of a hung or crashed Tenstorrent system. Unlike the integrated watcher server (Section 6.1), which requires an active tt-metal session, `watcher_dump` performs minimal device initialization -- it does not clear L1 memory, does not overwrite existing mailbox state, and does not launch firmware. This makes it safe to run against devices in a hung state, allowing retrieval of watcher mailbox data, command queue contents, and NOC transfer logs from the last run.

## Prerequisites

- The `watcher_dump` binary must be built from `tt_metal/tools/watcher_dump/watcher_dump.cpp`
- Binary location: `build/tools/watcher_dump` or `build_Release/tools/watcher_dump`
- The original program should have been run with `TT_METAL_WATCHER=1` so that `kernel_names.txt` exists in `generated/watcher/`
- The hung device must still be powered and accessible (no chip reset has occurred)

## 6.2.1 When to Use watcher_dump vs. Alternatives

```
Process crashed or was killed, device may be hung:
|
+-- Was TT_METAL_WATCHER enabled during the run?
|   |
|   +-- YES: watcher.log likely has data up to last poll
|   |   |
|   |   +-- Is the log file sufficient?
|   |       +-- YES --> Just read generated/watcher/watcher.log
|   |       +-- NO (need current device state) --> Run watcher_dump -w
|   |
|   +-- NO: Mailbox data may be stale or zeroed
|       |
|       +-- Run watcher_dump -w anyway (kernel IDs and run messages still useful)
|       +-- Also run watcher_dump --dump-noc-transfer-data if NOC logging was enabled
|
+-- Need command queue state?
|   --> watcher_dump -c (currently disabled in code, see limitations)
|
+-- Need full triage with callstacks and firmware analysis?
    --> Use tt-triage instead (Section 6.4), which provides richer analysis
```

**watcher_dump is the right choice when:**
- You need a quick, lightweight post-mortem dump without installing tt-exalens
- You want to read the raw watcher mailbox state from a device that may have been running without Watcher enabled
- You need CQ dump or NOC transfer data specifically

**watcher_dump is NOT the right choice when:**
- The process is still alive and you can attach GDB (use GDB + watcher dump, Section 6.1.6)
- You need detailed callstacks, firmware checks, and structured analysis (use tt-triage, Section 6.4)
- The device has already been reset (mailbox data is gone)

---

## 6.2.2 Tool Architecture

**Source:** `tt_metal/tools/watcher_dump/watcher_dump.cpp`

### Phase 1: Minimal Device Initialization

```cpp
IDevice* device = tt::tt_metal::CreateDeviceMinimal(
    id, num_hw_cqs,
    DispatchCoreConfig{eth_dispatch ? DispatchCoreType::ETH : DispatchCoreType::WORKER}
);
```

Key point: `CreateDeviceMinimal()` initializes just enough of the device driver to read L1 memory, without clearing or resetting device state:
- `rtoptions.set_clear_l1(false)` -- preserves existing L1 contents
- `rtoptions.set_watcher_enabled(false)` -- prevents the Watcher server from starting and overwriting `kernel_names.txt`

### Phase 2: State Dump

For watcher dump mode (`-w`), the tool reads kernel name mappings from `generated/watcher/kernel_names.txt` (written by the original run), then calls `WatcherServer::isolated_dump()` which creates a `WatcherDeviceReader` and dumps mailbox state.

### Phase 3: Cleanup

The tool closes devices and exits. The device remains in its current state (the tool does not reset it).

---

## 6.2.3 CLI Reference Table

| Flag | Long Form | Argument | Default | Description |
|------|-----------|----------|---------|-------------|
| `-h` | `--help` | none | -- | Display usage message |
| `-d=LIST` | `--devices=LIST` | comma list or `all` | `all` | Device IDs to dump. E.g., `-d=0,2,3` |
| `-n=INT` | `--num-hw-cqs=INT` | integer | `1` | Number of hardware command queues. **Must match the original program** |
| `-w` | `--dump-watcher` | none | off | Dump watcher mailbox data from all specified devices |
| `-c` | `--dump-cqs` | none | off | Dump command queue data (currently disabled in source) |
| | `--dump-cqs-data` | none | off | Dump raw command queue byte data (currently disabled) |
| | `--dump-noc-transfer-data` | none | off | Dump NOC transfer data. Requires previous run built with `TT_METAL_RECORD_NOC_TRANSFER_DATA` |
| | `--eth-dispatch` | none | off | Assume ethernet dispatch mode. **Must match the original run** |

---

## 6.2.4 Usage Recipes

### Recipe: Basic Post-Mortem Watcher Dump

```bash
# Program has crashed or been killed. Device is still up.
./build/tools/watcher_dump -w
cat generated/watcher/watcher.log
```

### Recipe: Dump Specific Devices

```bash
./build/tools/watcher_dump -w -d=0,2
```

### Recipe: Dump with Ethernet Dispatch Matching

```bash
./build/tools/watcher_dump -w --eth-dispatch -n=2
```

### Recipe: Dump NOC Transfer Data

```bash
# Requires the original program to have been run with TT_METAL_RECORD_NOC_TRANSFER_DATA=1
./build/tools/watcher_dump --dump-noc-transfer-data -d=0
```

---

## 6.2.5 Output Interpretation

The output format is identical to the runtime watcher log (Section 6.1.5), since the same `WatcherDeviceReader::Dump()` code path is used. Key differences:

1. **Single snapshot**: Only one dump (not periodic polls), reflecting the final state at crash time.
2. **Kernel names**: If `generated/watcher/kernel_names.txt` exists from the original run, kernel IDs will be resolved. Otherwise, only numeric IDs appear.
3. **Staleness**: If watcher was not enabled during the original run, waypoints and ring buffer contain stale data. The `rmsg` and `k_ids` fields are more reliable since they are written by dispatch regardless of watcher state.

### What You Can Learn

- **Waypoint codes**: Which code point each RISC was at when execution stopped
- **NOC sanitization errors**: If a sanitization error triggered the hang, error details are in the mailbox
- **Assert status**: If a kernel assert was tripped, the line number and RISC are recorded
- **Ring buffer contents**: Any `WATCHER_RING_BUFFER_PUSH()` values from the last kernel execution
- **Run messages**: Dispatch state at crash time (running, waiting, completed)

### What You Cannot Learn

- **Historical progression**: Unlike the runtime watcher which produces periodic snapshots
- **Waypoints if watcher was disabled**: Mailboxes contain only the default `X` value
- **Command queue contents**: CQ dumping is currently disabled

---

## 6.2.6 Limitations

| Limitation | Detail |
|-----------|--------|
| **No kernel names without prior watcher run** | If `TT_METAL_WATCHER` was not set during the original run, `kernel_names.txt` will not exist |
| **CQ dump currently disabled** | The command queue dump functionality is disabled in current source |
| **Device must be accessible** | If the device is in an unrecoverable state (ARC hung, PCIe link down), reads may fail |
| **Must match dispatch configuration** | `--eth-dispatch` and `--num-hw-cqs` must match the original program |
| **No L1 clearing** | By design; may read stale data if device was reset between crash and dump |
| **Single-shot tool** | Runs once and exits; does not continuously monitor |

---

## 6.2.7 Workflow: Combining watcher_dump with tt-triage

When a program dies unexpectedly:

1. **First:** Run `watcher_dump -w` to capture raw mailbox state before anything else touches the device.
2. **Second:** Run `tt-triage` (Section 6.4) for automated analysis including callstacks, NOC status, and ARC health.
3. **Third:** Cross-reference kernel IDs from watcher_dump with the callstack data from tt-triage.

This order matters because tt-triage's initialization may modify some device state, while watcher_dump reads it non-destructively.

---

## 6.2.8 Hang Scenarios

### Scenario 6.2.1: Post-Mortem Dump Reveals Stuck Waypoints

**Symptom**: The original program hung and was killed. Running `watcher_dump -w` shows several cores with waypoint `NRW` (NOC Read Wait).

**Root Cause**: A NOC read operation never completed, likely due to an invalid target address or a hung remote core.

**Diagnosis Steps**:
1. Run `./build/tools/watcher_dump -w -d=all`
2. Open `generated/watcher/watcher.log`
3. Search for cores not in `W` (waiting) or `D` (done) state
4. For cores stuck in `NRW`, check the last NOC sanitize status
5. Cross-reference kernel IDs with `kernel_names.txt`

**Fix**: Investigate the kernel's NOC read operations. Check for invalid target coordinates or deadlocked remote cores (see Ch2 scenarios).

**Prevention**: Enable watcher during development with `TT_METAL_WATCHER=1` for real-time detection.

### Scenario 6.2.2: watcher_dump Cannot Identify Kernel Names

**Symptom**: watcher_dump output shows numeric kernel IDs but no kernel paths/names.

**Root Cause**: `generated/watcher/kernel_names.txt` does not exist or was overwritten by a subsequent run.

**Diagnosis Steps**:
1. Check if `generated/watcher/kernel_names.txt` exists
2. If overwritten, numeric IDs can sometimes be cross-referenced with `CreateKernel()` calls in host code
3. Use tt-triage which resolves kernel information from ELF files directly

**Fix**: For future runs, always enable `TT_METAL_WATCHER=1` to ensure kernel name files are generated.

**Prevention**: Use `TT_METAL_WATCHER_APPEND=1` to preserve logs across runs.

### Scenario 6.2.3: watcher_dump Hangs During Device Attach

**Symptom**: Running watcher_dump itself hangs during `CreateDeviceMinimal()`.

**Root Cause**: The device is in a state where even minimal initialization fails (hard NOC fabric hang or ARC processor failure), or another process still holds the device lock.

**Diagnosis Steps**:
1. Verify no other process is holding the device (`lsof /dev/tenstorrent*`)
2. Use `tt-smi` to check basic device health
3. If the device is truly hard-hung, a chip reset is required (`tt-smi -r 0`)
4. After chip reset, L1 state is lost -- focus on logs from the original run

**Fix**: Reset the device and rely on any host-side logs that were saved before the hang.

**Prevention**: Set `TT_METAL_WATCHER_APPEND=1` and `TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1` to maximize diagnostic data persistence.

---

**Cross-references:**
- Watcher log format: Section 6.1.5
- tt-triage for richer post-mortem analysis: Section 6.4
- Dispatch hang diagnosis: Chapter 4
