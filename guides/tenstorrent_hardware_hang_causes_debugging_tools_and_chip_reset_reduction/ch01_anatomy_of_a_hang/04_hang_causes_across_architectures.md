# Hang Causes Across Architectures

Tenstorrent's hardware has evolved through multiple architecture generations, each with distinct characteristics that affect how and where hangs manifest. Some hang causes are universal -- they exist in every generation because they stem from the fundamental RISC-V spin-wait model. Others are architecture-specific, arising from unique hardware features, errata, or design decisions in a particular chip. This section documents the architecture-specific hang behaviors for Grayskull (GS), Wormhole (WH), Blackhole (BH), and Quasar (QA), as well as scale-dependent hang patterns that emerge in multi-chip configurations.

## Architecture Overview

| Feature | Grayskull (GS) | Wormhole (WH) | Blackhole (BH) | Quasar (QA) |
|---|---|---|---|---|
| Architecture family | tt-0xx | tt-1xx | tt-1xx | tt-2xx |
| Tensix cores | ~120 | ~80 | ~140 | TBD (large) |
| L1 SRAM per core | ~1 MB | 1464 KB (1.43 MB) | 1536 KB (1.5 MB) | 4096 KB (4 MB) |
| RISC-V cores per Tensix | 5 (BRISC, NCRISC, TRISC0-2) | 5 (BRISC, NCRISC, TRISC0-2) | 5 (BRISC, NCRISC, TRISC0-2) | DM cores + 4 Neo engines x 4 TRISCs |
| NOCs | 2 (NOC0, NOC1) | 2 (NOC0, NOC1) | 2 (NOC0, NOC1) | 2 (NOC0, NOC1) |
| NCRISC IRAM | No | Yes (executes from IRAM) | No (executes from L1) | N/A (unified DM) |
| Ethernet | No | Yes | Yes | Yes |
| Multi-chip | No | N300, T3K, Galaxy | Multi-BH mesh | Multi-QA mesh |
| Coordinate virtualization | No | Yes | Yes | Yes |
| Circular buffers | Up to 32 | Up to 32 (lower half only) | Up to 64 (full 64-bit mask) | Dataflow Buffers (DFBs) |
| Inline write to L1 | Safe | Safe | **Hangs under back-pressure** | TBD |
| L1 data cache | No | No | **Yes** (requires explicit invalidation) | TBD |
| Relaxed memory ordering | No | No | **Yes** (requires barriers) | TBD |
| NOC DRAM read alignment | -- | 32 bytes | 64 bytes | 64 bytes |
| NOC PCIe read alignment | -- | 32 bytes | 64 bytes | 64 bytes |
| Transaction IDs | No | No | Yes (max 255) | Yes (max 65535) |
| Dynamic NOC mode | -- | -- | Supported | Dedicated only (constexpr) |
| Heartbeat mechanism | -- | Yes (addr `0x1C`) | Not needed | TBD |
| Firmware source | `tt-1xx/brisc.cc` | `tt-1xx/brisc.cc` | `tt-1xx/brisc.cc` | `tt-2xx/dm.cc` |

## Universal Hang Causes

The blocking primitives documented in [02_blocking_primitives_taxonomy.md](./02_blocking_primitives_taxonomy.md) are universal across all architectures: CB deadlocks (CRBW/CWFW), NOC barrier stalls (NRBW/NWBW), semaphore overshoot (NSW), go-signal failure (GW), and NOC address errors all use the same spin-wait model and have the same fundamental failure modes on GS, WH, BH, and QA. The architecture-specific differences are:

- **CB count**: BH supports 64 circular buffers (full 64-bit mask) vs. 32 on WH, increasing the potential deadlock surface area.
- **NOC barrier implementation**: Register addresses and counter organizations differ per architecture (separate `noc_nonblocking_api.h` files), though the check is conceptually the same (compare hardware counter with software counter).
- **Alignment requirements**: DRAM and PCIe read alignment is 32 bytes on WH but 64 bytes on BH and QA -- a kernel that works on WH may hang on BH/QA when addresses are not 64-byte aligned.

---

## Grayskull: Baseline Architecture

Grayskull is the first-generation Tenstorrent architecture. It establishes the baseline behavior for all blocking primitives and hang categories.

### Single-Chip Only

Grayskull does not have Ethernet links. All hang categories involving multi-chip communication (Category 5) do not apply. This significantly reduces the hang surface area.

### Standard NCRISC Behavior

On GS, NCRISC deassert and reset follows the standard path. There is no special halt-and-reset sequence, eliminating the WH-specific NCRISC hang vector:

```c++
// GS and BH path (not WH)
#if !defined(ARCH_WORMHOLE)
    if (enables & (1u << ...TensixProcessorTypes::DM1...)) {
        subordinate_sync->dm1 = RUN_SYNC_MSG_GO;
    }
#endif
```

### GS-Specific Characteristics

- **No ethernet cores**: ERISC-related hang patterns do not apply
- **No IRAM for NCRISC**: NCRISC executes directly from L1, avoiding the Wormhole-specific IRAM hang modes
- **No coordinate virtualization**: Physical and logical coordinates are the same, eliminating one class of addressing errors
- **Simpler power management**: Uses `ex_setc16` instruction path rather than direct register writes; no `DEST_CG_CTRL` register to disable
- **No transaction ID support**: All barrier calls must wait for *all* outstanding transactions, making partial-barrier debugging impossible

GS is susceptible to Categories 1 (Kernel), 2 (NOC), 3 (Memory), 4 (Dispatch), and 6 (Host-Device). It has the simplest hang landscape of all architectures.

---

## Wormhole: Ethernet and NCRISC Complexity

Wormhole introduces Ethernet connectivity and a more complex NCRISC reset sequence, both of which create new hang surfaces.

### WH-1: NCRISC IRAM Reset Dance

This is the most distinctive Wormhole-specific hang pattern. The NCRISC on Wormhole executes kernels from a dedicated Instruction RAM (IRAM) rather than from L1. This creates a complex startup sequence:

1. NCRISC firmware copies the kernel from L1 to IRAM via TDMA DMA
2. NCRISC sets `ncrisc_halt.resume_addr` to the IRAM kernel address
3. NCRISC signals `RUN_SYNC_MSG_WAITING_FOR_RESET` to BRISC
4. BRISC detects this signal and responds:
   - Writes the IRAM address to the NCRISC reset PC register
   - Asserts the NCRISC reset line (`assert_just_ncrisc_reset()`)
   - Waits a calibrated delay (`riscv_wait(5)`) -- "chosen empirically"
   - Deasserts the reset (`deassert_all_reset()`)
5. NCRISC restarts execution from the IRAM address

The firmware comment explains: *"The NCRISC behaves badly if it jumps from L1 to IRAM, so instead halt it and then reset it to the IRAM address it provides."*

```c++
#if defined(ARCH_WORMHOLE)
inline void start_ncrisc_kernel_run(uint32_t enables) {
    if (enables & ...TensixProcessorTypes::DM1...) {
        while (subordinate_sync->dm1 != RUN_SYNC_MSG_WAITING_FOR_RESET);
        subordinate_sync->dm1 = RUN_SYNC_MSG_GO;

        volatile tt_reg_ptr uint32_t* cfg_regs = core.cfg_regs_base(0);
        cfg_regs[NCRISC_RESET_PC_PC_ADDR32] = mailboxes->ncrisc_halt.resume_addr;
        assert_just_ncrisc_reset();
        riscv_wait(5);  // Empirically determined wait
        deassert_all_reset();
    }
}
#endif
```

**Hang Modes:**
- If NCRISC never reaches `RUN_SYNC_MSG_WAITING_FOR_RESET` (e.g., its DMA copy hung), BRISC spins in the while loop forever. This spin loop has **no WAYPOINT marker**, making it harder to diagnose.
- If `riscv_wait(5)` is insufficient, the NCRISC may "continue where it left off" rather than resetting, executing stale code.
- After kernel completion, NCRISC performs a branch predictor flush (13 repetitions). If this is insufficient, instruction corruption during the next IRAM load can cause a hang.

### WH-2: L1 Hammering Mitigation

Wormhole is the only architecture where the firmware explicitly inserts NOP instructions in spin-wait loops:

```c
#if defined(ARCH_WORMHOLE)
    // Avoid hammering L1 while other cores are trying to work.
    asm volatile("nop; nop; nop; nop; nop");
#endif
```

This appears in both the BRISC subordinate wait loop and the NCRISC go-wait loop. The comment notes that this "seems not to be needed on Blackhole, probably because `invalidate_l1_cache` takes time." On WH, aggressive L1 polling can starve other cores of L1 bandwidth, potentially exacerbating performance degradation during a hang.

### WH-3: Heartbeat Mechanism

Wormhole uniquely uses a heartbeat mechanism where RISC-V cores periodically write an incrementing counter:

```c
inline void RISC_POST_HEARTBEAT(uint32_t& heartbeat) {
#if !defined(ARCH_BLACKHOLE)
    invalidate_l1_cache();
    volatile uint32_t* ptr = (volatile uint32_t*)(0x1C);
    heartbeat++;
    ptr[0] = 0xAABB0000 | (heartbeat & 0xFFFF);
#endif
}
```

If the watcher thread reads this address and the counter has stopped incrementing, it knows the core is stuck even if the waypoint is stale. Blackhole does not need this because it has other diagnostic capabilities.

### WH-4: Ethernet Core Hangs

Wormhole introduces Ethernet cores for multi-chip communication. The `eth_noc_semaphore_wait` primitive calls `run_routing()` during its spin loop to keep the Ethernet routing alive:

```c++
void eth_noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val,
                            uint32_t wait_min = 0) {
    uint32_t count = 0;
    while ((*sem_addr) != val) {
        invalidate_l1_cache();
        if (count == wait_min) {
            run_routing();
            count = 0;
        } else {
            count++;
        }
    }
}
```

The active ERISC firmware contains the `enter_reset` / `resume_from_reset` workaround for a known hang:

> "After running the base firmware, some core state (for erisc0) seems broken, so jumps into the kernel may occasionally hang. Resetting the core fixes the issue."

This is a hardware errata workaround: the ERISC core's microarchitectural state can become corrupted after running base firmware, and the only fix is a full core reset with GPR and local memory save/restore.

### WH-5: NCRISC Startup Ordering

On Wormhole, BRISC delays the NCRISC kernel start until after CB setup. On non-WH architectures, NCRISC starts early. This ordering difference means the critical path is longer on WH, and if CB setup hangs (e.g., at NABW), the NCRISC never receives its go signal.

---

## Blackhole: Inline Write Back-Pressure, L1 Cache, and Extended Addressing

Blackhole introduces the most significant architecture-specific hang vectors. It also extends the NOC address space, the number of supported circular buffers, and introduces dynamic NOC mode.

### BH-1: Inline Write Back-Pressure Hang

This is the most critical BH-specific hang mechanism. From `risc_attribs.h`:

```c++
// This enum is used to specify the dest location type for inline writes.
// It is needed because inline writes use all 4 memory ports and may hang
// on Blackhole when there is back-pressure.
// This hang only manifests when the inline writes are issued to a L1
// location. The workaround on BH is for inline writes to L1 to use
// noc async writes.
enum class InlineWriteDst : uint8_t { DEFAULT = 0, L1 = 1, REG = 2 };
```

The mechanism:
1. An inline write (via `noc_inline_dw_write`) places data directly in the NOC command
2. On Blackhole, inline writes use all four memory ports simultaneously
3. If L1 is under contention, back-pressure builds up
4. With all four ports engaged and back-pressure preventing completion, the NOC pipeline stalls
5. This stall prevents the command buffer from becoming ready, blocking all subsequent NOC operations
6. Any subsequent `noc_async_write_barrier` will hang at `NWBW`

The `InlineWriteDst` enum provides the workaround: when `dst_type == InlineWriteDst::L1`, the implementation falls back to `noc_async_write` instead of an inline write.

Two API functions carry explicit warnings:
> *"Note: On Blackhole, this API can only write to stream registers, writing to L1 will cause hangs!"* (from `noc_inline_dw_write_set_state` and `noc_inline_dw_write_with_state`)

### BH-2: L1 Data Cache Coherence

Blackhole introduces a small L1 data cache on the RISC-V cores. This cache must be explicitly invalidated (via `invalidate_l1_cache()`) when polling memory locations that are updated by the NOC hardware or other cores. All blocking primitives in `dataflow_api.h` include cache invalidation in their spin loops. However, **user-written spin loops that poll L1 memory without calling `invalidate_l1_cache()` will read stale cached values and hang** even though the actual L1 value has been updated. This is a new class of stale-read hangs that does not exist on GS or WH.

### BH-3: Relaxed Memory Ordering

Blackhole has relaxed memory ordering compared to earlier architectures. This means that NOC writes may complete in a different order than they were issued. Kernels that depend on ordering (e.g., writing data before writing a semaphore flag) must use explicit barriers. On GS/WH, the ordering was effectively sequential, so these patterns worked by accident. On BH, they require explicit `noc_async_write_barrier` calls to guarantee ordering, and omitting them creates a new category of data-dependent, intermittent hangs.

### BH-4: Fabric Router Counter Workaround

Blackhole requires an unconditional counter update for all write operations as a workaround for fabric router hangs:

```c++
#ifdef ARCH_BLACKHOLE
    // Issue https://github.com/tenstorrent/tt-metal/issues/28758: always update counter
    // for blackhole as a temporary workaround for avoiding hangs in fabric router
    constexpr bool update_counter_in_callee = true;
#else
    constexpr bool update_counter_in_callee = update_counter;
#endif
```

If a write operation on BH skips the counter update, the fabric router's internal counter check will find a mismatch and hang. The comment indicates this is temporary: "will remove this restriction once all inline write change to stream reg write."

### BH-5: Atomic Barrier During CB Setup

Blackhole is the only architecture that requires an atomic NOC barrier during remote CB interface setup:

```c++
#if defined(ARCH_BLACKHOLE)
inline void barrier_remote_cb_interface_setup(uint8_t noc_index,
                                              uint32_t noc_mode,
                                              uint32_t end_cb_index) {
    if (end_cb_index != NUM_CIRCULAR_BUFFERS) {
        WAYPOINT("NABW");
        if (noc_mode == DM_DYNAMIC_NOC) {
            do {
                invalidate_l1_cache();
            } while (!ncrisc_dynamic_noc_nonposted_atomics_flushed(noc_index));
        } else {
            while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
        }
        invalidate_l1_cache();
        WAYPOINT("NABD");
    }
}
#endif
```

A hang at NABW during firmware (not user kernel) execution indicates a NOC issue in the CB setup phase. The dispatch core is exempted because "`cq_dispatch` does not update noc transaction counts."

### BH-6: Dynamic NOC Mode and Stale State

Blackhole introduces dynamic NOC mode (`DM_DYNAMIC_NOC`), where BRISC and NCRISC share NOC resources dynamically. The post-kernel assertion check verifies that all NOC transactions are properly cleaned up:

```c
if (noc_mode == DM_DYNAMIC_NOC) {
    for (int noc = 0; noc < NUM_NOCS; noc++) {
        ASSERT(ncrisc_dynamic_noc_reads_flushed(noc));
        ASSERT(ncrisc_dynamic_noc_nonposted_writes_sent(noc));
        // ... (all five barrier types)
    }
}
```

If a kernel exits without properly barriering in dynamic NOC mode, residual transactions from one kernel can contaminate the next kernel's NOC state, causing **phantom hangs** that are extremely difficult to diagnose because the root cause is in a previously completed kernel.

### BH-7: NOC Re-initialization Per Kernel

Blackhole requires NOC local state re-initialization during kernel launch even in dedicated NOC mode:

```c++
#ifdef ARCH_BLACKHOLE
// Need to add this to allow adding barrier after setup_remote_cb_interfaces
noc_local_state_init(noc_index);
#endif
```

If this initialization is skipped, the NOC counters may be in an inconsistent state, causing subsequent barriers to hang.

### BH-8: Extended Circular Buffer Support

Blackhole supports 64 circular buffers (full 64-bit CB mask), compared to Wormhole's effective limit of 32:

```c++
#ifdef ARCH_BLACKHOLE
uint32_t local_cb_mask_upper = static_cast<uint32_t>(local_cb_mask >> 32);
setup_local_cb_read_write_interfaces<true, true, false, false>(cb_l1_base, 32, local_cb_mask_upper);
#endif
```

More CBs means more potential producer-consumer pairs that can deadlock.

### BH-9: Destination Clock Gating Disable

Blackhole disables destination register clock gating at startup:

```c++
#ifdef ARCH_BLACKHOLE
    *((volatile uint32_t*)RISCV_DEBUG_REG_DEST_CG_CTRL) = 0;
#endif
```

This prevents timing-related hangs where the destination register file's clock gating could cause the compute pipeline to miss updates.

### BH-10: Stricter DRAM/PCIe Alignment

Blackhole requires 64-byte alignment for DRAM and PCIe reads, compared to Wormhole's 32-byte requirement. **This is the single most common source of "works on WH, hangs on BH" regression bugs when porting kernels.** The sanitize checker in watcher will catch this if enabled, but in production builds the transaction silently fails and the read barrier hangs.

---

## Quasar: Fundamentally Different Core Topology

Quasar (tt-2xx) represents a significant architectural departure from the GS/WH/BH (tt-1xx) line. The firmware in `tt_metal/hw/firmware/src/tt-2xx/dm.cc` reveals the key differences.

### QA-1: Unified DM Core

Quasar replaces BRISC and NCRISC with a single DM (Data Mover) core with hardware threading. This eliminates the WH NCRISC IRAM issue and the BRISC-NCRISC synchronization overhead, but introduces new patterns:

- The DM core uses `thread_local` storage for CB interfaces and runtime arguments, meaning thread-local corruption can affect only one hardware thread
- The subordinate synchronization protocol differs: the DM core manages multiple hardware threads directly
- The firmware uses `do_thread_crt1` for thread-local initialization, adding a new failure point

### QA-2: Four TRISC Cores per Neo Engine

Quasar has four TRISC cores (TRISC0-TRISC3) per Neo engine, with four Neo engines per tile. This significantly expands the hang surface:

**More subordinates to wait on**: DM0 must wait for all DM subordinates AND all four Neo engines:

```c++
inline void wait_subordinates() {
    WAYPOINT("NTW");
    while (subordinate_sync->allDMs != RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE ||
           subordinate_sync->allNeo0 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE ||
           subordinate_sync->allNeo1 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE ||
           subordinate_sync->allNeo2 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE ||
           subordinate_sync->allNeo3 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE);
    WAYPOINT("NTD");
}
```

With 16 TRISCs (4 per Neo, 4 Neos) plus multiple DM cores, the number of potential subordinate hang sources is much larger than on tt-1xx architectures. A bug in a kernel running on Neo2's TRISC1 will hang Neo2 but not Neo0/1/3 -- though DM0 will still hang at `NTW`.

### QA-3: Dataflow Buffers (DFBs)

Quasar uses a different buffer abstraction (`LocalDFBInterface` / `g_dfb_interface`) instead of the traditional CB interface. The blocking primitive semantics may differ, creating new hang patterns.

### QA-4: No Dynamic NOC Mode

Quasar uses `constexpr uint8_t noc_mode = DM_DEDICATED_NOC;` unconditionally. This eliminates the dynamic NOC mode and its associated hang surfaces (counter synchronization between two cores sharing a NOC).

### QA-5: Larger L1 and Uncached Region

At 4 MB per core (vs. 1.5 MB on BH), Quasar's L1 provides more buffering capacity, which can mask CB sizing errors for longer. Quasar also introduces a cached/uncached L1 split (`MEM_L1_UNCACHED_BASE` at the 4 MB boundary), creating a potential failure mode where NOC transactions targeting the uncached region have different behavior.

### QA-6: Higher Transaction ID Capacity

With `NOC_MAX_TRANSACTION_ID_COUNT` of 65535 (vs. 255 on BH), Quasar reduces TRID exhaustion risk but may mask transaction tracking bugs that would surface earlier on BH.

### QA-7: Remapper API

Quasar introduces a `RemapperAPI` (`g_remapper_configurator`) that must be cleared between kernel launches. If the remapper state is corrupted, subsequent kernel launches may access wrong addresses.

---

## Scale-Dependent Hang Patterns

Hang behavior changes dramatically with system scale:

### Single Chip (GS, WH, BH, or Quasar)

- Categories 1-4 and 6 apply; Category 5 does not (no Ethernet, except on WH/BH/QA where Ethernet cores exist but are not used for cross-chip traffic)
- All hangs are self-contained and can be fully diagnosed from the single chip's state
- Recovery requires only a single chip reset

### N300 (2 Wormhole chips, Ethernet-connected)

- Category 5 becomes relevant: Ethernet link between L chip and R chip
- The R chip has no direct PCIe connection; host communication routes through the L chip
- A hang on the R chip requires the L chip's Ethernet core to relay diagnostic information
- Recovery may require resetting both chips

### T3K (8 Wormhole chips in mesh)

- Mesh topology means any chip can communicate with any other chip, creating N-to-N potential hang propagation paths
- EDM (Ethernet Data Movers) manage cross-chip traffic routing; EDM bugs can cause systemic hangs
- A single kernel bug on one chip can cascade through semaphore dependencies to hang all 8 chips

### Galaxy (32+ Wormhole chips, multi-host)

- Multi-host dispatch coordination: if one host's dispatch system fails, chips managed by that host may hang, cascading to other hosts through Ethernet
- At this scale, transient hangs become statistically likely even when per-chip reliability is high
- NOC congestion from cross-chip traffic can cause hangs on cores not involved in the cross-chip operation

### Multi-BH (Multiple Blackhole chips)

- All BH-specific hangs (inline-write back-pressure, fabric router counter workaround) apply to every chip
- Fabric routing uses the inline-write-prone code paths, making the counter workaround critical at scale
- A fabric router hang on one BH chip can block traffic for all chips that route through it

### Multi-QA (Multiple Quasar chips)

- Quasar's tt-2xx architecture is still emerging; multi-chip hang patterns are expected to differ
- The unified DM core simplifies some synchronization but changes the failure modes for cross-chip data movement
- Four TRISC cores per Neo engine increase the degree of parallelism and complexity of cross-chip pipeline coordination

---

## Architecture-Specific vs. Universal Hang Categories

### Universal (All Architectures)

| Hang Category | Root Cause | Affected Architectures |
|---|---|---|
| CB deadlock (CRBW/CWFW) | Mismatched push/pop counts | GS, WH, BH, Quasar |
| NOC invalid address (NRBW/NWBW) | Bad coordinates or address | GS, WH, BH, Quasar |
| Semaphore protocol error (NSW/NSMW) | Missing signal or value skip | GS, WH, BH, Quasar |
| Dispatch go-signal failure (GW) | Dispatch core hung | GS, WH, BH, Quasar |
| NOC command buffer stall (RP2W) | Hardware fault | GS, WH, BH, Quasar |

### Architecture-Specific

| Hang Category | Root Cause | Affected Architecture(s) |
|---|---|---|
| NCRISC halt-reset timeout | `riscv_wait(5)` insufficient | WH only |
| NCRISC `WAITING_FOR_RESET` never set | NCRISC firmware crash before signal | WH only |
| L1 hammering causing secondary stalls | Aggressive polling without NOP | WH only |
| ERISC core state corruption (errata) | Microarchitectural state broken after base FW | WH only |
| Inline write back-pressure | L1 port contention with 4-port inline write | BH only |
| Fabric router counter mismatch | `update_counter` not forced `true` | BH only |
| L1 data cache stale reads | Custom spin loop missing `invalidate_l1_cache()` | BH only |
| Relaxed memory ordering violations | Missing barriers for write ordering | BH only |
| Atomics flush barrier stall (NABW) | Remote CB setup atomics stuck | BH only |
| Dynamic NOC stale state contamination | Kernel exits without barriering | BH only |
| NOC re-initialization skipped | Missing `noc_local_state_init` | BH only |
| Stricter DRAM/PCIe alignment (64B) | 32B-aligned address not 64B-aligned | BH, Quasar |
| Extended subordinate wait (Neo0-3 + DMs) | Any of 16+ TRISCs hung | Quasar only |
| Dataflow Buffer (DFB) deadlock | DFB protocol error | Quasar only |
| Remapper state corruption | Stale remapper config across kernel launches | Quasar only |
| Ethernet data mover hang | Link failure or routing deadlock | WH, BH, Quasar |

### Recovery Requirements by Architecture

| Architecture | Kill Recovery | Chip Reset Required |
|---|---|---|
| GS | Most kernel/semaphore hangs | NOC hardware stalls, corrupted firmware state |
| WH | Kernel/semaphore hangs, some dispatch hangs | NOC stalls, NCRISC reset failures, Ethernet link failures |
| BH | Kernel/semaphore hangs | Inline-write back-pressure, fabric router hangs, NOC stalls |
| Quasar | Kernel/DFB hangs | NOC stalls, remapper corruption, subordinate sync corruption |

---

## Key Takeaways for Practitioners

1. **When porting WH kernels to BH**: Check all DRAM and PCIe read addresses for 64-byte alignment. Replace any direct inline writes to L1 with `noc_async_write` or use the `InlineWriteDst::L1` type parameter. Add explicit `invalidate_l1_cache()` calls to any custom spin loops that poll L1 memory. Ensure write ordering does not depend on implicit sequential semantics -- add explicit barriers.

2. **When debugging multi-chip hangs**: Start by isolating whether the hang is local (single chip) or cross-chip (Ethernet dependency). If cross-chip, check ERISC firmware state and Ethernet link health before tracing kernel-level semaphores.

3. **When scaling to larger core counts (BH, QA)**: Test synchronization protocols with the maximum number of participating cores. Semaphore protocol bugs that are statistically unlikely with 8 cores may be near-certain with 140 cores.

4. **When developing for Quasar**: Be aware that the unified DM core changes assumptions about read/write independence. Kernels that relied on NOC 0 for reads and NOC 1 for writes may need restructuring.

5. **Universal**: Enable watcher during development. The `assert_and_hang` and `debug_sanitize_post_addr_and_hang` mechanisms transform silent hangs into diagnosed failures with mailbox data. The performance overhead of watcher is small compared to the debugging time saved.

---

**Next:** [Chapter 2 -- Kernel-Level and NOC Hang Mechanisms](../ch02_kernel_and_noc_hangs/index.md)
