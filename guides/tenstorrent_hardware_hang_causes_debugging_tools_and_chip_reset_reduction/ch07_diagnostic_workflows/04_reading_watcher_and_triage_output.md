# 7.4 Reading Watcher and Triage Output

[Previous: Narrowing and Reproducing](./03_narrowing_and_reproducing.md) | [Next: Distinguishing HW vs SW Bugs](./05_distinguishing_hw_vs_sw_bugs.md)

---

The debugging tools from Chapter 6 produce dense, encoded output that requires careful interpretation. This section is a practical decoding guide: how to read waypoint strings, decode NOC sanitize violations, interpret assert messages, correlate kernel IDs with source files, and map program counter values to source locations. Each subsection follows a "here is what you see / here is what it means" pattern.

**Prerequisites:** [Chapter 6, `01_watcher_system.md`](../ch06_debugging_tools/01_watcher_system.md) (watcher architecture and mailbox format), [Chapter 6, `04_tt_triage_tool.md`](../ch06_debugging_tools/04_tt_triage_tool.md) (tt-triage script catalog), [Chapter 1, `02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) (waypoint code reference).

---

## Watcher Log Anatomy

The watcher log consists of three sections:

1. **Header:** Session metadata (device type, build configuration, watcher settings).
2. **Periodic dumps:** One block per poll interval, containing per-core state for all cores.
3. **Kernel name mapping:** At the end, a listing of kernel IDs to source file paths.

### Single Watcher Line Decode

Each core produces one line per dump cycle:

```
Device 0 core(x=1,y=1) [phys(x=18,y=18)]:  rmsg:D0G|BNTTT  smsg:GGG  k_ids:5|5|5
  BRISC:CRBW  NCRISC:D  TRISC0:CWFW  TRISC1:R  TRISC2:R
```

Field-by-field breakdown:

| Field | Meaning |
|-------|---------|
| `Device 0` | Physical device ID |
| `core(x=1,y=1)` | Logical core coordinates |
| `[phys(x=18,y=18)]` | Physical core coordinates (if `TT_METAL_WATCHER_PHYS_COORDS` is set) |
| `rmsg:D0G\|BNTTT` | Run message: dispatch mode, NOC ID, run state, followed by per-RISC enable flags |
| `smsg:GGG` | Slave message: go signals for subordinate RISCs (NCRISC, TRISC0, TRISC1+TRISC2) |
| `k_ids:5\|5\|5` | Kernel IDs for BRISC, NCRISC, and TRISC (maps to `kernel_names.txt`) |
| `BRISC:CRBW` | BRISC waypoint: stuck at CB Reserve Back Wait |
| `NCRISC:D` | NCRISC waypoint: Done |
| `TRISC0:CWFW` | TRISC0 waypoint: stuck at CB Wait Front Wait |
| `TRISC1:R` | TRISC1 waypoint: Running |
| `TRISC2:R` | TRISC2 waypoint: Running |

### Run Message Decoding

The `rmsg` field encodes multiple pieces of state:

| Position | Value | Meaning |
|----------|-------|---------|
| 1st char | `D` or `H` | `D` = Device dispatch mode (fast dispatch), `H` = Host dispatch mode (slow dispatch) |
| 2nd char | `0` or `1` | NOC ID assigned to BRISC |
| 3rd char | `G`, `D`, `I`, `W`, `R` | Run state: `G`=Go, `D`=Done, `I`=Init, `W`=Waiting for reset, `R`=Reset read pointer |
| After `\|` | `BNTTT` | Per-RISC enable: B=BRISC, N=NCRISC, T=TRISC0, T=TRISC1, T=TRISC2. Uppercase = enabled. |

---

## Decoding Waypoint Strings

### The Waypoint Encoding

Each RISC-V core on every Tensix writes a 4-character waypoint code to a dedicated L1 mailbox location at every significant state transition. The `WAYPOINT(str)` macro (defined in `tt_metal/hw/inc/api/debug/waypoint.h`) compiles to a single 32-bit store when `WATCHER_ENABLED` is defined:

```
Byte layout (little-endian):
  Byte 0 (LSB): 1st character
  Byte 1:       2nd character
  Byte 2:       3rd character
  Byte 3 (MSB): 4th character (or 0 if < 4 chars)
```

### Complete Waypoint Code Reference

#### Kernel-Level Waypoints

| Code | Wait/Done | Full Name | Meaning | Blocking Primitive | Reference |
|------|-----------|-----------|---------|-------------------|-----------|
| `CRBW` | Wait | CB Reserve Back Wait | Producer waiting for CB free space | `cb_reserve_back` | [Ch1, 02](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) |
| `CRBD` | Done | CB Reserve Back Done | CB free space available | -- | -- |
| `CWFW` | Wait | CB Wait Front Wait | Consumer waiting for CB data | `cb_wait_front` | [Ch1, 02](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) |
| `CWFD` | Done | CB Wait Front Done | CB data available | -- | -- |
| `NRBW` | Wait | NOC Read Barrier Wait | Waiting for all NOC reads to complete | `noc_async_read_barrier` | [Ch1, 02](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) |
| `NRBD` | Done | NOC Read Barrier Done | All NOC reads completed | -- | -- |
| `NWBW` | Wait | NOC Write Barrier Wait | Waiting for all NOC writes to be acked | `noc_async_write_barrier` | [Ch1, 02](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) |
| `NWBD` | Done | NOC Write Barrier Done | All NOC writes acknowledged | -- | -- |
| `NSW` | Wait | NOC Semaphore Wait | Waiting for semaphore == target | `noc_semaphore_wait` | [Ch1, 02](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) |
| `NSD` | Done | NOC Semaphore Done | Semaphore reached target | -- | -- |
| `NSMW` | Wait | NOC Semaphore Min Wait | Waiting for semaphore >= target | `noc_semaphore_wait_min` | [Ch1, 02](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md) |
| `NSMD` | Done | NOC Semaphore Min Done | Semaphore at or above target | -- | -- |
| `RP2W` | Wait | Reg Poll 2 Wait | NOC command buffer not ready | `NOC_CMD_CTRL` polling | [Ch2, 04](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |

#### Firmware Synchronization Waypoints

| Code | Wait/Done | Meaning | Reference |
|------|-----------|---------|-----------|
| `NTW` | Wait | NCRISC/TRISC wait -- BRISC waiting for subordinate processors | [Ch2, 01](../ch02_kernel_and_noc_hangs/01_risc_synchronization_and_deadlocks.md) |
| `GW` | Wait | Go signal wait -- subordinate waiting for BRISC go message | [Ch2, 01](../ch02_kernel_and_noc_hangs/01_risc_synchronization_and_deadlocks.md) |
| `NABW` | Wait | NOC all barrier wait -- waiting for both NOC0 and NOC1 barriers | [Ch2, 04](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |
| `NKFW` | Wait | Next kernel FW wait -- firmware waiting for next kernel | [Ch2, 01](../ch02_kernel_and_noc_hangs/01_risc_synchronization_and_deadlocks.md) |
| `SEW` | Wait | Subordinate exit wait -- waiting for subordinate RISC to reach done | [Ch2, 01](../ch02_kernel_and_noc_hangs/01_risc_synchronization_and_deadlocks.md) |

#### Dispatch Waypoints

| Code | Wait/Done | Kernel | Meaning | Reference |
|------|-----------|--------|---------|-----------|
| `HQW` | Wait | Prefetch | Waiting for host to write fetch queue entries | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |
| `UAPW` | Wait | Prefetch | Upstream read (relay topology) | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |
| `CNSW` | Wait | Prefetch | Command-not-sent acknowledgment wait | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |
| `QRBW` | Wait | Prefetch/Dispatch | CB queue reserve back wait | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |
| `QRBD` | Done | Prefetch/Dispatch | CB queue reserve back done | -- |
| `PWW` | Wait | Dispatch | `process_wait`: waiting for worker completion semaphore | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |
| `PWD` | Done | Dispatch | `process_wait` done | -- |
| `WCW` | Wait | Dispatch | `write_and_check_completion_signal`: waiting for all workers | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |
| `WCD` | Done | Dispatch | Completion signal check done | -- |
| `DCW` | Wait | Dispatch_S | Waiting for dispatch master notification | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |
| `DAPW` | Wait | Dispatch | Data-mover progress wait | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |
| `DAPD` | Done | Dispatch | Data-mover progress done | -- |
| `CBRW` | Wait | Dispatch | CB page release stall | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |
| `CBRD` | Done | Dispatch | CB page release done | -- |
| `!CMD` | -- | Any dispatch | Invalid command received | [Ch4, 01](../ch04_dispatch_and_host_device_hangs/01_dispatch_architecture_and_hang_points.md) |

#### Transaction ID Barrier Waypoints

| Code | Wait/Done | Meaning | Reference |
|------|-----------|---------|-----------|
| `NBTW` | Wait | NOC barrier with transaction ID -- read barrier for specific TRID | [Ch2, 04](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |
| `NWTW` | Wait | NOC write barrier with transaction ID | [Ch2, 04](../ch02_kernel_and_noc_hangs/04_noc_barrier_and_semaphore_hangs.md) |

#### General State Waypoints

| Code | Meaning |
|------|---------|
| `I` | Init -- RISC is initializing |
| `W` | Wait -- RISC is in the firmware idle loop waiting for work |
| `R` | Run -- RISC is executing kernel code |
| `D` | Done -- RISC has completed kernel execution |

### Reading Multi-Core Patterns

When analyzing a hang, read the waypoints across all cores simultaneously. The pattern tells you the coordination failure:

```
Example: CB deadlock between reader and compute
  Core (1,1): BRISC:CRBW  NCRISC:D    TRISC0:CWFW  TRISC1:R    TRISC2:R
  
  Interpretation:
  - BRISC (reader): stuck waiting for CB free space (CRBW)
  - NCRISC (writer): finished (D)
  - TRISC0 (unpack): stuck waiting for CB data (CWFW)
  - TRISC1/2 (math/pack): still running (R) -- dependent on TRISC0
  
  Diagnosis: BRISC is producing data into a CB that TRISC0 consumes. But TRISC0
  is waiting for data in a DIFFERENT CB that some other producer should fill.
  Meanwhile, the CB that BRISC writes to is full because TRISC0 is not consuming.
  --> Classic CB deadlock. Check CB indices and producer/consumer assignments.
```

```
Example: Dispatch waiting for workers
  Dispatch core: BRISC:PWW
  Worker (0,0): BRISC:NSW   NCRISC:D
  Worker (0,1): BRISC:D     NCRISC:D
  Worker (1,0): BRISC:D     NCRISC:D
  Worker (1,1): BRISC:D     NCRISC:D

  Interpretation:
  - Dispatch is at PWW: waiting for all workers to signal completion
  - Worker (0,0) BRISC is stuck at NSW: waiting on a semaphore
  - All other workers are done (D)
  
  Diagnosis: Worker (0,0) is waiting for a semaphore that was never incremented.
  The dispatch hang is secondary -- fix the semaphore issue on (0,0).
```

---

## Interpreting NOC Sanitize Violations

When watcher NOC sanitization detects an illegal transaction, it writes a `debug_sanitize_addr_msg_t` structure to the watcher mailbox and then hangs the core. The watcher log decodes this structure.

### The `debug_sanitize_addr_msg_t` Structure

```c++
// tt_metal/hw/inc/internal/debug/sanitize.h
struct debug_sanitize_addr_msg_t {
    uint64_t noc_addr;        // The NOC address that was validated
    uint32_t l1_addr;         // The L1 address involved (for local ops)
    uint32_t len;             // Transfer length in bytes
    uint16_t which_risc;      // Which RISC-V core triggered this (0=BRISC, 1=NCRISC, etc.)
    uint8_t  is_multicast;    // 1 if this was a multicast transaction
    uint8_t  is_write;        // 1 if this was a write (0 = read)
    uint8_t  is_target;       // 1 if this core is the TARGET of the transaction (not the initiator)
    uint8_t  return_code;     // The specific violation type (see Section 02 mapping table)
    uint16_t pad;
};
```

### Decoding Example

```
Watcher log line:
  Core (2,3): SANITIZE: type=NocAddrOverflow noc_addr=0x000200001200F800 l1_addr=0x000F8000
              len=4096 risc=BRISC mcast=0 write=0 target=0

Decode:
  - Core (2,3) BRISC issued a NOC read (write=0, target=0 means this core is the initiator)
  - NOC address: 0x000200001200F800 -- the upper bits encode the target core coordinates
  - L1 address 0x000F8000 = 0xF8000 = 1,015,808 bytes into L1
  - Length: 4096 bytes
  - 0xF8000 + 4096 = 0xF9000 = 1,019,904 bytes
  - If MEM_L1_SIZE = 1,499,136 (WH): this is within range --> the overflow is on the TARGET
  - The target core's L1 at the NOC address may be smaller, or the target address is out of range
  
Action: Check the NOC address target coordinates and the target core's L1 size.
```

### NOC Address Decoding

For unicast transactions, the NOC address encoding is:
- Bits [35:0]: Local offset within the target's address space
- Bits [41:36]: Y coordinate
- Bits [47:42]: X coordinate

For multicast transactions, additional fields encode the range of target cores.

### Sentinel Values

| Sentinel | Value | Meaning |
|----------|-------|---------|
| `DEBUG_SANITIZE_SENTINEL_OK_64` | `0xbadabadabadabada` | No violation detected on this core |

If the sanitize mailbox contains this sentinel, the core has not triggered any NOC violations.

For the complete violation return code table with diagnosis steps, see [Section 02, NOC Error Message Mapping Table](./02_diagnosing_by_hang_category.md#noc-error-message-mapping-table).

---

## Interpreting Assert Messages

When a kernel hits an `ASSERT()` macro (with watcher enabled), it writes a `debug_assert_msg_t` to the mailbox:

### The `debug_assert_msg_t` Structure

```c++
// tt_metal/hw/inc/api/debug/assert.h
struct debug_assert_msg_t {
    uint16_t line_num;     // Source line number of the assertion
    uint8_t  tripped;      // Assert type (see below)
    uint8_t  which;        // Which RISC-V processor (0=BRISC, 1=NCRISC, 2-4=TRISC0-2)
};
```

### Assert Types

| `tripped` Value | Enum Name | Meaning |
|-----------------|-----------|---------|
| 2 | `DebugAssertOK` | No assert tripped (sentinel) |
| 3 | `DebugAssertTripped` | Generic `ASSERT(condition)` failed |
| 4 | `DebugAssertNCriscNOCReadsFlushedTripped` | Kernel completed with outstanding NOC reads (missing read barrier) |
| 5 | `DebugAssertNCriscNOCNonpostedWritesSentTripped` | Kernel completed with outstanding non-posted writes (missing write barrier) |
| 6 | `DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped` | Kernel completed with outstanding atomic operations (missing atomic barrier) |
| 7 | `DebugAssertNCriscNOCPostedWritesSentTripped` | Kernel completed with outstanding posted writes (missing posted-write barrier) |
| 8 | `DebugAssertRtaOutOfBounds` | Runtime argument index exceeds allocated count |
| 9 | `DebugAssertCrtaOutOfBounds` | Common runtime argument index exceeds allocated count |

**Note:** Assert types 4-7 (inter-kernel data race asserts) are among the most common assert types in practice. They always indicate a missing NOC barrier before kernel exit.

### Decoding Procedure

1. Note the `line_num` and `which` fields.
2. Look up the kernel ID in `kernel_names.txt` to find the source file.
3. Open the source file at the reported line number.
4. If the assert is `DebugAssertRtaOutOfBounds` (8) or `DebugAssertCrtaOutOfBounds` (9), the kernel is reading a runtime argument at an index beyond what was allocated -- check the host-side `SetRuntimeArgs` or `SetCommonRuntimeArgs` call.
5. For inter-kernel data race asserts (4-7), add the appropriate barrier before kernel exit.

### ERISC Assert Behavior

ERISC processors handle asserts differently from Tensix: instead of entering `while(1)`, an ERISC calls `erisc_exit()` to cleanly shut down the Ethernet core. The assert is still recorded in the watcher mailbox.

### Lightweight Kernel Asserts (ebreak)

When `TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1` is set, the `ASSERT()` macro compiles to an `ebreak` instruction instead of a watcher mailbox write. The ebreak causes the RISC-V core to halt immediately (not a spin-loop, but a hard stop).

To decode an ebreak-triggered assert:
```bash
./tools/tt-triage.py --run=dump_lightweight_asserts --dev=0
```

This reads the PC at which the ebreak occurred and maps it to a source location. Since ebreak does not write to the watcher mailbox, the watcher log will not show the assert -- only tt-triage can decode it.

---

## Reading Ring Buffer Data

The watcher ring buffer is a per-core circular debug log (31-element `uint32_t` buffer) that kernels can write to via `WATCHER_RING_BUFFER_PUSH(value)`. It is application-specific -- the meaning of the values depends entirely on what the kernel author chose to log.

### Ring Buffer Format in Watcher Log

```
Core (x,y) ring_buffer: [v0, v1, v2, ..., v30]
```

Values are 32-bit unsigned integers. The buffer wraps: the most recently pushed value overwrites the oldest. Look for a discontinuity (e.g., a sequence of zeros indicating unused slots) to find the write frontier.

### Common Patterns

| Value Pattern | Interpretation |
|--------------|---------------|
| Sequential integers (1, 2, 3, ...) | Progress markers -- the kernel pushed the iteration count |
| NOC addresses (large hex values) | The kernel logged addresses before/after transactions |
| Small values with a constant pattern | Enum values or state machine states |
| High byte = category, low bytes = value | Common encoding convention for structured debug data |
| All zeros | Ring buffer was not used (or only initialized) |

### Post-Mortem Retrieval

If watcher was not enabled, tt-triage can still read the ring buffer:

```bash
./tools/tt-triage.py --run=dump_watcher_ringbuffer --dev=0
```

---

## Correlating Kernel IDs from kernel_names.txt

The watcher assigns a sequential integer ID to each kernel as it is compiled. These IDs appear in the watcher log as `k_id:N`. The mapping is stored in `generated/watcher/kernel_names.txt`:

```
0: blank
1: tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
2: tt_metal/impl/dispatch/kernels/cq_dispatch.cpp
3: tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp
4: path/to/your/reader_kernel.cpp
5: path/to/your/compute_kernel.cpp
6: path/to/your/writer_kernel.cpp
```

### Key Observations

- IDs 0-3 are typically dispatch kernels (prefetch, dispatch, dispatch_subordinate).
- User kernels start at ID 4 or higher.
- The kernel name is the **source file path**, not a function name.
- If program caching is active, the same kernel may be compiled only once but reused across multiple programs -- the ID remains stable.

### Alternative: Inspector kernels.yaml

The Inspector `kernels.yaml` file provides an alternative mapping with additional metadata:

```yaml
- kernel:
    watcher_kernel_id: 5
    name: reader_matmul
    source: ttnn/cpp/ttnn/operations/matmul/device/kernels/reader_matmul.cpp
    program_id: 3
```

This is especially useful when `kernel_names.txt` is unavailable.

---

## Using kernel_elf_paths.txt for ELF Disassembly

For deeper analysis (e.g., mapping a program counter from tt-triage to a source line), the compiled kernel ELF binaries can be disassembled. The file `generated/watcher/kernel_elf_paths.txt` maps kernel IDs to ELF file paths:

```
0: blank
1: /home/user/.cache/tt-metal-cache/.../kernels/cq_prefetch/.../brisc/brisc.elf
2: /home/user/.cache/tt-metal-cache/.../kernels/cq_dispatch/.../brisc/brisc.elf
```

### Disassembly Procedure

```bash
# Get the ELF path for kernel ID 4
elf_path=$(sed -n '5p' generated/watcher/kernel_elf_paths.txt | cut -d' ' -f2)

# Disassemble with source annotations (if debug info is present)
riscv32-unknown-elf-objdump -d -S "$elf_path" > kernel_disasm.txt

# Search for a specific PC address
grep -A 10 "^0000abcd" kernel_disasm.txt
```

### Mapping PCs to Source Lines

When tt-triage `dump_callstacks.py` reports a program counter (e.g., `PC=0x0000abcd`):

1. Identify which RISC the PC belongs to (BRISC, NCRISC, or TRISC0-2).
2. Find the corresponding ELF from `kernel_elf_paths.txt`.
3. Use `addr2line` for quick resolution:
   ```bash
   riscv32-unknown-elf-addr2line -e "$elf_path" -f 0x00012345
   ```
4. Or use `objdump -d -S` for full disassembly context.

### Firmware vs. Kernel Code

- PCs in low memory ranges (near the start of the address space) are in **firmware** code (`brisc.cc`, `ncrisc.cc`, etc.).
- PCs in higher ranges are in **kernel** code (the user-written reader/compute/writer).
- The boundary depends on the firmware size, which varies by architecture and configuration.

---

## Reading tt-triage Callstacks

The `dump_callstacks.py` and `dump_aggregated_callstacks.py` scripts extract RISC-V program counters from all cores and optionally resolve them to function names.

### Callstack Output Format

```
Device 0, Core (1,2):
  BRISC: PC=0x00012345, stack: [0x00012340, 0x00011234, 0x00010ABC]
  NCRISC: PC=0x00009876
  TRISC0: PC=0x0000FEDC
```

### Aggregated Callstacks

The `dump_aggregated_callstacks.py` groups cores that are stuck at the same PC:

```
PC=0x00012345 (BRISC): 28 cores
  Cores: (0,0), (0,1), (0,2), ..., (3,6), (3,7)
PC=0x00009876 (NCRISC): 28 cores
  Cores: (0,0), (0,1), (0,2), ..., (3,6), (3,7)
```

**Key insight:** If all worker cores are stuck at the same PC, the hang is systematic (same kernel, same code path on every core). If only one or two cores are stuck at a unique PC, those cores have a specific issue -- they are likely the root cause, not the victims.

### Resolving PCs to Source

1. Get the ELF path from `kernel_elf_paths.txt`.
2. Use `addr2line` for quick resolution:
   ```bash
   riscv32-unknown-elf-addr2line -e "$elf_path" -f 0x00012345
   ```
3. Or use `objdump -d -S` for full disassembly context.

---

## Reading tt-triage Dispatch State

The `dump_fast_dispatch` script provides dispatch-specific state:

| Field | Meaning | Hang Implication |
|-------|---------|-----------------|
| `cmd_ptr` | Current command pointer | Not advancing = dispatch stuck |
| `cb_fence` | CB consumption fence | Low relative to production = dispatch not consuming |
| `last_wait_count` | Wait loop iterations | High = dispatch spinning |
| `last_wait_stream` | Stream being waited on | Identifies which resource is blocking |
| `sem_minus_local` | Extra CB pages | Zero = CB between prefetch and dispatch is full |
| `issue_q_rd` / `issue_q_wr` | Issue queue read/write pointers | If rd == wr, queue is empty/consumed |
| `completion_q_rd` / `completion_q_wr` | Completion queue pointers | If wr not advancing, device not signaling completion |

### Key Dispatch Command Table

| Command | Purpose | Hang Risk |
|---------|---------|-----------|
| `CQ_PREFETCH_CMD_RELAY_LINEAR` | Relay data from host to dispatch | Low |
| `CQ_PREFETCH_CMD_STALL` | Wait for dispatch to consume | Medium (can stall if dispatch is hung) |
| `CQ_DISPATCH_CMD_WRITE_LINEAR` | Write to L1 (kernel binary, CB config) | Low |
| `CQ_DISPATCH_CMD_WAIT` | Wait for workers to complete | High (most common dispatch hang point) |
| `CQ_DISPATCH_CMD_GO` | Send go signal to workers | Low |
| `CQ_DISPATCH_CMD_WRITE_LINEAR_HOST` | Write completion signal to host | Medium |

---

## Sync Register Inspection Caveats

The `DumpSyncRegs` function in tt-triage can read NOC synchronization registers from a hung core.

> **DANGER:** Reading registers while a core is running can itself cause hangs. The register read uses a NOC transaction to the target core; if the target core's NOC is already in a bad state, the read may not return, hanging the diagnostic tool. The tt-triage output includes the warning: "reading registers while running can cause hangs, only read if requested explicitly."

### When to Use

- **Safe:** The core is known to be hung (stuck waypoint). Reading its registers will not make things worse.
- **Unsafe:** The core is still running (waypoint is `R` or rapidly changing). Reading registers during active execution can corrupt the NOC state.
- **Use `--all-cores` cautiously:** This reads registers from all cores, including running ones.

### What the Registers Show

| Register | Meaning | Hang Implication |
|----------|---------|-----------------|
| `noc_reads_num_issued` | Count of NOC reads issued by this core | Compare with `NIU_MST_RD_RESP_RECEIVED` -- mismatch = stuck read |
| `noc_nonposted_writes_num_issued` | Count of non-posted writes issued | Compare with `noc_nonposted_writes_acked` -- mismatch = stuck write |
| `NOC_CMD_CTRL` | NOC command buffer status | If not `NOC_CTRL_STATUS_READY`, command buffer is full/stalled |

---

## Cross-Device Watcher Log Correlation (Multi-Chip)

On multi-chip systems, correlating watcher logs across devices reveals the initiator of a cascading failure:

### Procedure

1. **Collect watcher logs from all devices/hosts.**
2. **Align timestamps.** Each watcher dump includes a timestamp. Normalize across hosts if clocks differ.
3. **Find the earliest anomaly.** The device that shows the first stuck waypoint or first error message is the likely initiator.
4. **Trace the dependency chain.** If device A is stuck waiting for data from device B, check device B's Ethernet core state. If device B's ERISC is also stuck, trace to the next device in the chain.
5. **Check ERISC-specific output.** ERISC cores have unique watcher fields:
   - Ethernet link status (up/down)
   - Link retraining count
   - EDM flow control state

### ERISC-Specific Watcher Output

ERISC processors appear in watcher output with `erisc` as the processor type. Their run states include Ethernet-specific codes. Key indicators:
- ERISC at `RW` (router wait): blocked waiting on fabric flow control
- ERISC at `SEW` (semaphore/event wait): blocked on a synchronization primitive
- ERISC at `D` (done): either completed its task or exited due to link failure

---

## Putting It All Together: A Worked Example

Here is a complete example of reading and decoding watcher output for a CB deadlock:

```
=== Watcher Log ===
Core (1,1) [phys (18,18)]:  k_id:5  BRISC:CRBW  NCRISC:D  TRISC0:CWFW  TRISC1:R  TRISC2:CRBW

=== kernel_names.txt ===
5: ttnn/cpp/ttnn/operations/matmul/device/kernels/reader_matmul.cpp

=== Decode ===
1. Core (1,1) is running kernel ID 5: the matmul reader kernel.
2. BRISC (DM0) is at CRBW: waiting for free space in a CB (producer side).
   --> The CB is full. The consumer has not popped tiles.
3. TRISC0 (unpack) is at CWFW: waiting for data in a CB (consumer side).
   --> A CB that feeds unpack is empty. The producer has not pushed tiles.
4. TRISC2 (pack) is at CRBW: waiting for free space in its output CB.
   --> The output CB is full. The downstream consumer (writer/NCRISC) is done.
5. NCRISC (DM1, writer) is at D: finished.
   --> The writer has already completed and stopped consuming from TRISC2's output CB.

Root cause hypothesis:
  The writer (NCRISC) finished early -- it consumed fewer tiles than TRISC2 will produce.
  TRISC2 cannot push more output (CRBW) because NCRISC stopped popping.
  TRISC0 cannot unpack because TRISC2 is blocked (pipeline stall propagates backward).
  BRISC cannot push reader data because TRISC0 is blocked and the input CB is full.
  
  --> Loop count mismatch between writer and compute kernels.
  --> Check: does the writer kernel's loop count match the compute kernel's expected output tiles?
  
  Fix: Ensure writer iterates the same number of times as the compute kernel produces output tiles.
  Reference: [Ch2, Scenario 2.2.5] (Loop bound mismatch)
```

---

## Quick Reference: Common Triage Output Patterns

| Multi-Core Pattern | Diagnosis | Action |
|-------------------|-----------|--------|
| 28/28 cores at same BRISC PC | Systematic kernel bug | All cores hit same code path; debug the kernel |
| 1 core at unique PC, 27 at `D` | Single-core failure | That core is the root cause; check its waypoint and NOC state |
| All workers `D`, dispatch `PWW` | Dispatch missed completion | Check completion signal path; verify semaphore increment |
| All workers `NSW`, dispatch `PWW` | Semaphore protocol failure | Workers waiting for signal that dispatch never sent |
| Mix of `CRBW`/`CWFW`/`NRBW` | Cascading stall | Find the first core that stalled; others are secondary |

---

**Next:** [05_distinguishing_hw_vs_sw_bugs.md](./05_distinguishing_hw_vs_sw_bugs.md)
