# 6.4 tt-triage Automated Diagnostic System

## Summary

`tt-triage` is a Python-based diagnostic framework that automatically runs a series of checks and data-gathering scripts against a Tenstorrent system (typically a hung or malfunctioning one). It leverages `ttexalens` for device access and the Inspector for runtime metadata, producing structured, visualized output tables for rapid diagnosis. The tool is extensible: developers can add new check or data provider scripts following a well-defined discovery protocol. tt-triage integrates with the Inspector RPC system for querying runtime state of a live or recently crashed tt-metal process.

## Prerequisites

- Python 3.x with `ttexalens` library installed (`scripts/install_debugger.sh`)
- `capnp` Python module (`pip install -r tools/triage/requirements.txt`)
- RISC-V ELF files from the last build (for callstack resolution)
- Device access (physical or via `ttexalens` remote)
- `TT_METAL_RISCV_DEBUG_INFO=1` recommended for best callstack quality
- Inspector data (optional but strongly recommended): `TT_METAL_INSPECTOR=1`

## 6.4.1 When to Use tt-triage vs. Alternatives

```
Device has hung or program has failed:
|
+-- Need quick, lightweight state dump?
|   --> watcher_dump (Section 6.2) -- faster, no Python deps
|
+-- Need detailed analysis with callstacks, firmware checks, structured output?
|   --> tt-triage (this section)
|
+-- Need to understand which OPERATION was running when the hang occurred?
|   --> tt-triage with Inspector integration
|
+-- Process is still alive and you can attach a debugger?
|   --> GDB + Watcher dump (Section 6.1.6)
|
+-- Need to automate triage on dispatch timeout in CI?
    --> TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=./tools/tt-triage.py
        (See Section 6.6)
```

---

## 6.4.2 Architecture

### Entry Point and Source Files

| File | Path | Role |
|------|------|------|
| Entry point | `tools/tt-triage.py` | Main script, adds `tools/triage/` to path, calls `triage.main()` |
| Framework | `tools/triage/__init__.py` | `ScriptConfig`, `triage_field`, `recurse_field`, `run_script`, `log_check` |
| Core engine | `tools/triage/triage.py` | Script discovery, dependency resolution, execution, output formatting |
| Session mgmt | `tools/triage/triage_session.py` | Session state management |
| HW utilities | `tools/triage/triage_hw_utils.py` | Hardware-level helper functions |

### Script Discovery Protocol

1. `tt-triage` scans all `.py` files in `tools/triage/`
2. Each file must define a global `script_config` of type `ScriptConfig` and a `run(args, context)` function
3. Files not meeting this signature are silently skipped
4. Dependency resolution ensures correct execution order (topological sort)

```python
@dataclass
class ScriptConfig:
    data_provider: bool = False    # True = provides data for other scripts
    disabled: bool = False         # True = skip execution
    depends: list[str] = []        # Script names this depends on
```

Scripts fail gracefully: if a data provider fails, all dependent scripts are marked as failed but other independent scripts continue.

---

## 6.4.3 CLI Reference

| Flag | Argument | Default | Description |
|------|----------|---------|-------------|
| `--run=SCRIPT` | script name | (all scripts) | Run only specified script (and its dependencies). Can be repeated |
| `-v` | none | level 0 | Increase verbosity. `-v` = level 1, `-vv` = level 2 |
| `--all-cores` | none | filter DONE | Show all cores including DONE (completed) cores |
| `--remote-exalens` | none | local | Use remote tt-exalens connection |
| `--remote-server` | host | -- | Remote server address |
| `--remote-port` | port | -- | Remote server port |
| `--disable-colors` | none | colors on | Disable Rich colors (for piping to files) |
| `--inspector-rpc-host` | host | localhost | Inspector RPC override host |
| `--inspector-rpc-port` | port | 50051 | Inspector RPC override port |

### Usage Examples

```bash
# Run all scripts
./tools/tt-triage.py

# Run specific scripts
./tools/tt-triage.py --run=dump_callstacks
./tools/tt-triage.py --run=dump_callstacks --run=check_noc_status

# Verbose output with all cores
./tools/tt-triage.py --run=dump_callstacks -vv --all-cores

# Remote connection
./tools/tt-triage.py --remote-exalens --remote-server=10.0.0.1 --remote-port=5555

# Standalone script execution
python3 tools/triage/dump_callstacks.py
```

---

## 6.4.4 Script Catalog

### Data Provider Scripts

| Script File | Provides | Description |
|------------|----------|-------------|
| `device_info.py` | Device information | Device enumeration, architecture, core topology |
| `device_telemetry.py` | Telemetry data | Runtime telemetry from devices |
| `dispatcher_data.py` | Dispatch state | Operations, kernels, firmware state |
| `callstack_provider.py` | Callstack data | Raw callstack data from RISC-V cores |
| `elfs_cache.py` | ELF cache | Cached ELF file parsing for symbol resolution |
| `firmware_versions.py` | FW versions | Firmware version info per device |
| `inspector_data.py` | Inspector state | Parsed Inspector runtime data |
| `inspector_capnp.py` | Cap'n Proto data | Inspector data in Cap'n Proto format |
| `metal_device_id_mapping.py` | Device ID map | Metal device IDs to physical device IDs |
| `operation_runtime_map.py` | Op runtime map | Runtime operation mapping |
| `system_info.py` | System info | Host OS, driver version |

### State Checker Scripts

| Script File | Checks | Description |
|------------|--------|-------------|
| `check_arc.py` | ARC processor health | Checks if ARC processors are responsive |
| `check_binary_integrity.py` | Binary integrity | Validates kernel binary checksums |
| `check_broken_components.py` | HW components | Detects non-functional hardware |
| `check_cb_inactive.py` | CB inactivity | Checks for inactive circular buffers |
| `check_core_magic.py` | L1 magic values | Validates L1[0] magic (firmware launch) |
| `check_eth_status.py` | ETH link status | Checks ethernet link state and retraining |
| `check_noc_locations.py` | NOC addresses | Validates NOC transaction address ranges |
| `check_noc_status.py` | NOC status | Compares NOC transaction counters vs. hardware registers |

### Dump Scripts

| Script File | Description | Key Verbosity Fields |
|------------|-------------|---------------------|
| `dump_callstacks.py` | Per-core RISC-V callstack resolution | v0: kernel, waypoint, PC, callstack; v1: +FW path, host ID; v2: +RD PTR, base |
| `dump_aggregated_callstacks.py` | Groups cores by identical callstack | Pattern detection across cores |
| `dump_configuration.py` | Runtime configuration and env vars | -- |
| `dump_fast_dispatch.py` | Fast dispatch state (CQs, dispatch cores) | -- |
| `dump_lightweight_asserts.py` | Lightweight assert trip information | Detects cores at `ebreak` |
| `dump_risc_debug_signals.py` | RISC-V debug bus signals | -- |
| `dump_running_operations.py` | Currently running operations from Inspector | -- |
| `dump_watcher_ringbuffer.py` | Watcher ring buffer contents from all cores | -- |

---

## 6.4.5 Inspector Integration

### Inspector Environment Variables

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| `TT_METAL_INSPECTOR` | flag | enabled | Enable/disable Inspector data collection |
| `TT_METAL_INSPECTOR_RPC` | flag | enabled | Enable/disable RPC server |
| `TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS` | `host:port` | `localhost:50051` | RPC server bind address |
| `TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT` | flag | `true` | Auto-serialize state on dispatch timeout |
| `TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT` | flag | `false` | Track init phase closely |
| `TT_METAL_INSPECTOR_WARN_ON_WRITE_EXCEPTIONS` | flag | `true` | Warn on write failures |
| `TT_METAL_INSPECTOR_CAPTURE_TENSOR_SPECS` | flag | `false` | Capture tensor specs on op dispatch |
| `TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES` | flag | `false` | Log to YAML (expensive) |
| `TT_METAL_RISCV_DEBUG_INFO` | flag | inherits Inspector | Enable debug info in ELFs for callstack resolution |

### Inspector Data Flow in tt-triage

```
tt-triage
  |
  +-- inspector_data.py (data provider)
      |
      +-- Try RPC connection to Inspector server
      |   (if tt-metal process is still alive)
      |
      +-- Fall back: load serialized RPC data from log directory
      |   (if process crashed but SERIALIZE_ON_DISPATCH_TIMEOUT was set)
      |
      +-- Fall back: parse Inspector text logs
          (last resort, less structured)
```

---

## 6.4.6 Key Diagnostic Workflows

### Workflow 1: Callstack Analysis

```bash
export TT_METAL_RISCV_DEBUG_INFO=1
export TT_METAL_INSPECTOR=1
./my_program &
# (program hangs)
./tools/tt-triage.py --run=dump_callstacks -vv
```

### Workflow 2: Hang Quick Reference

| Hang Type | Triage Scripts | What to Look For |
|-----------|---------------|-----------------|
| NOC hang | `check_noc_status`, `check_noc_locations`, `dump_callstacks` | Transaction counter mismatch, bad target coords |
| Dispatch hang | `dump_fast_dispatch`, `dump_running_operations` | Stuck CQ, missing completion signal |
| Compute hang | `dump_callstacks`, `dump_lightweight_asserts` | Assert trips, stuck waypoints |
| ETH hang | `check_eth_status`, `dump_callstacks` | Link failures, retraining events |
| ARC hang | `check_arc`, `device_telemetry` | Non-responsive ARC, thermal issues |

### Workflow 3: Combined watcher_dump + tt-triage

1. **First:** `./build/tools/watcher_dump -w` (non-destructive)
2. **Second:** `./tools/tt-triage.py` (may modify some state)
3. **Third:** Cross-reference kernel IDs with callstack data

---

## 6.4.7 Writing Custom tt-triage Scripts

### State Checker Template

```python
# tools/triage/check_my_feature.py
from triage import ScriptConfig, log_check

script_config = ScriptConfig(
    data_provider=False,
    depends=["device_info"],
)

def run(args, context):
    log_check("My feature check", passed=True, details="All good")
    return None

if __name__ == "__main__":
    from triage import run_script
    run_script()
```

### Data Provider Template

```python
# tools/triage/dump_my_data.py
from dataclasses import dataclass
from triage import ScriptConfig, triage_field

script_config = ScriptConfig(data_provider=True)

@dataclass
class MyData:
    core: str = triage_field("Core")
    status: str = triage_field("Status")
    value: int = triage_field("Value", verbose=1)

def run(args, context):
    return [
        MyData(core="(0,0)", status="OK", value=42),
        MyData(core="(1,1)", status="ERROR", value=0),
    ]

if __name__ == "__main__":
    from triage import run_script
    run_script()
```

---

## 6.4.8 Hang Scenarios

### Scenario 6.4.1: tt-triage Reveals Cores Stuck in NOC Read Wait

**Symptom**: `dump_callstacks` shows multiple worker cores with waypoint `NRW` and callstacks pointing to `noc_async_read_wait()`.

**Root Cause**: NOC read transactions issued to a target core that is itself hung, creating a chain of blocked reads.

**Diagnosis Steps**:
1. Run `./tools/tt-triage.py --run=dump_callstacks --run=check_noc_status`
2. Identify all cores with non-DONE waypoints
3. Use `dump_aggregated_callstacks` to find patterns (many cores at same callstack)
4. Check `check_noc_locations` for valid target addresses
5. Find the root cause core (the one that others are waiting on)

**Fix**: Fix the root cause on the blocking core (see Ch2 NOC hang scenarios).

**Prevention**: Use watcher with NOC sanitization (Section 6.1) to catch errors at point of issue.

### Scenario 6.4.2: Callstack Resolution Fails for Kernel

**Symptom**: `dump_callstacks` shows PC values but cannot resolve to source code ("unknown" or raw addresses).

**Root Cause**: ELF file not available, does not match loaded binary, or RISC-V debug info was not enabled.

**Diagnosis Steps**:
1. Check that `TT_METAL_RISCV_DEBUG_INFO=1` was set during build
2. Ensure the build directory matches the binary that was running
3. Do not `make clean` between the hang and running tt-triage

**Fix**: Enable `TT_METAL_RISCV_DEBUG_INFO=1` for future runs.

**Prevention**: Always build with debug info in development environments.

### Scenario 6.4.3: NOC Transaction Counter Mismatch Detected

**Symptom**: `check_noc_status` reports mismatches between firmware NOC transaction variables and hardware NOC status registers.

**Root Cause**: NOC transactions were issued but never completed -- a NOC-level hang where either a transaction is stuck in the fabric or the target core is not responding.

**Diagnosis Steps**:
1. The mismatch identifies which core(s) and NOC(s) have stuck transactions
2. Cross-reference with `dump_callstacks` to see the code path that initiated the transaction
3. Check target core state
4. See Chapter 2, Sections 2.1-2.3 for NOC hang root causes

**Fix**: Fix the condition that blocks the NOC transaction (missing barrier, bad target, etc.).

**Prevention**: Enable Watcher NOC sanitization for proactive detection.

### Scenario 6.4.4: tt-triage Cannot Connect to Device

**Symptom**: tt-triage fails during tt-exalens initialization with a device timeout or connection error.

**Root Cause**: Device is in a hard-hung state where even register reads time out (NOC fabric deadlock or ARC failure).

**Diagnosis Steps**:
1. Use `tt-smi` to check device responsiveness
2. If `tt-smi` also fails, a chip reset is required
3. After reset, L1 state is lost -- focus on host-side logs

**Fix**: Reset the device via `tt-smi -r <device>` and analyze available host-side logs.

**Prevention**: Set `TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1` to persist diagnostic data before device becomes unreachable.

---

**Cross-references:**
- watcher_dump for lightweight post-mortem: Section 6.2
- Inspector data for dispatch state: Section 6.4.5
- Dispatch timeout auto-trigger: Section 6.6
- NOC hang scenarios: Chapter 2
- Dispatch hang scenarios: Chapter 4
