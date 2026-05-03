# Blocking Primitives Taxonomy

Every hang on a Tenstorrent device originates in a blocking primitive -- a device-side API function that spins in a tight loop until a hardware or software condition is met. This section catalogs every such primitive, documenting the exact spin-loop code from `dataflow_api.h`, the exit condition, the WAYPOINT markers, and the failure modes that can make the exit condition unsatisfiable.

Reference: `tt_metal/hw/inc/api/dataflow/dataflow_api.h`

## Understanding Waypoint Codes

The Watcher debug infrastructure writes a 4-character waypoint code to a per-core mailbox in L1 memory at every significant state transition. Each blocking primitive sets a "waiting" waypoint before entering its spin loop and a "done" waypoint after exiting. When a core hangs, it remains at the "waiting" waypoint, which directly identifies the blocking primitive causing the hang.

Waypoint codes are encoded as up to 4 ASCII characters packed into a 32-bit word, with the first character in the least-significant byte. The `WAYPOINT` macro (defined in `waypoint.h`) compiles to a single store instruction when `WATCHER_ENABLED` is defined, and compiles to nothing otherwise.

The convention is:
- The **waiting** waypoint ends with `W` (e.g., `CRBW`, `NRBW`)
- The **done** waypoint ends with `D` (e.g., `CRBD`, `NRBD`)

## Overview of Primary Blocking Primitives

| Abbreviation | Function | Wait Waypoint | Done Waypoint | Spins On | Core(s) |
|---|---|---|---|---|---|
| CRBW | `cb_reserve_back` | `CRBW` | `CRBD` | CB free space (producer side) | BRISC, NCRISC, TRISC2 (Pack) |
| CWFW | `cb_wait_front` | `CWFW` | `CWFD` | CB available tiles (consumer side) | BRISC, NCRISC, TRISC0 (Unpack) |
| NRBW | `noc_async_read_barrier` | `NRBW` | `NRBD` | NOC read responses received | BRISC, NCRISC |
| NWBW | `noc_async_write_barrier` | `NWBW` | `NWBD` | NOC write acks received | BRISC, NCRISC |
| NSW | `noc_semaphore_wait` | `NSW` | `NSD` | L1 semaphore equality | BRISC, NCRISC |
| NSMW | `noc_semaphore_wait_min` | `NSMW` | `NSMD` | L1 semaphore >= threshold | BRISC, NCRISC |

---

## CRBW: `cb_reserve_back` -- Circular Buffer Reserve Back Wait

### Purpose

`cb_reserve_back` is a blocking call used by the **producer** side of a circular buffer. It waits until the consumer has freed enough space in the CB for the producer to write `num_pages` new tiles.

### Exact Code

```c++
FORCE_INLINE
void cb_reserve_back(int32_t operand, int32_t num_pages) {
    uintptr_t pages_acked_ptr = (uintptr_t)get_cb_tiles_acked_ptr(operand);

    // While the producer is waiting, "tiles_pushed" (pages_received) is stable
    // because only the producer updates it via cb_push_back
    uint32_t pages_received = get_cb_tiles_received_ptr(operand)[0];

    int32_t free_space_pages;
    WAYPOINT("CRBW");
    do {
        invalidate_l1_cache();
        uint16_t pages_acked = (uint16_t)reg_read(pages_acked_ptr);
        uint16_t free_space_pages_wrap =
            get_local_cb_interface(operand).fifo_num_pages - (pages_received - pages_acked);
        free_space_pages = (int32_t)free_space_pages_wrap;
    } while (free_space_pages < num_pages);
    WAYPOINT("CRBD");
}
```

### Exit Condition

`free_space_pages >= num_pages`

Where:
- `free_space_pages = fifo_num_pages - (pages_received - pages_acked)`
- `pages_received` is the count of tiles the producer has pushed (frozen while waiting)
- `pages_acked` is the count of tiles the consumer has popped (incremented by `cb_pop_front` on the consumer side)
- `fifo_num_pages` is the total CB capacity in tiles

### Spin Mechanism

Each iteration:
1. Calls `invalidate_l1_cache()` to ensure fresh reads from L1 (which may have been updated by another RISC-V core or by a NOC write from another Tensix)
2. Reads `pages_acked` via `reg_read()` -- this is a `uint16_t` read because the Tensix Pack core updates this counter as a 16-bit value in the LLK `pop_tiles` path
3. Computes free space as the difference between capacity and occupied tiles
4. Compares against `num_pages`

### Waypoints

- **Wait marker**: `CRBW` (CB Reserve Back Wait)
- **Done marker**: `CRBD` (CB Reserve Back Done)

### Failure Modes

| Failure | Root Cause | Diagnosis |
|---|---|---|
| **Consumer never pops** | Consumer kernel terminates without calling `cb_pop_front`, or calls it with fewer tiles than expected | Watcher shows producer at `CRBW`; consumer at `D` (done) or `CWFW` (cascade) |
| **Consumer deadlock** | Consumer (typically TRISC0/Unpack via `cb_wait_front`) is itself blocked on a different CB, creating a circular dependency | Producer at `CRBW`, consumer at `CWFW` on a different CB |
| **Mismatched tile counts** | `num_pages` exceeds `fifo_num_pages`, or CB size is not evenly divisible by `num_pages` | Immediate hang on first call; check CB configuration |
| **Consumer hung at its own blocking primitive** | If the consumer is hung at `CWFW` waiting for input from yet another source, it will never pop tiles | Cascade pattern: trace the CB dependency chain |
| **uint16_t wrap-around** | Both counters are `uint16_t`; wrap is correct by design, but kernel bugs that corrupt these counters could trigger issues | Inspect CB counter values in L1 dump |

**Implementation detail:** The `pages_acked` value is read as `uint16_t` because TRISC cores update this value as a 16-bit quantity in `llk_pop_tiles`. The subtraction `pages_received - pages_acked` relies on unsigned wraparound semantics.

### Non-Blocking Alternative

```c++
bool cb_pages_reservable_at_back(int32_t operand, int32_t num_pages);
```

This performs the same check without spinning. Useful for debug instrumentation and for implementing custom timeout logic.

---

## CWFW: `cb_wait_front` -- Circular Buffer Wait Front

### Purpose

`cb_wait_front` is the **consumer** counterpart to `cb_reserve_back`. It waits until the producer has pushed at least `num_pages` tiles into the CB.

### Exact Code

```c++
FORCE_INLINE
void cb_wait_front(int32_t operand, int32_t num_pages) {
    uint32_t pages_acked = get_cb_tiles_acked_ptr(operand)[0];
    uintptr_t pages_received_ptr = (uintptr_t)get_cb_tiles_received_ptr(operand);

    uint16_t pages_received;

    WAYPOINT("CWFW");
    do {
        pages_received = ((uint16_t)reg_read(pages_received_ptr)) - pages_acked;
    } while (pages_received < num_pages);
    WAYPOINT("CWFD");
}
```

### Exit Condition

`pages_received >= num_pages`

Where:
- `pages_received = (uint16_t)reg_read(pages_received_ptr) - pages_acked`
- The `reg_read` fetches the producer's pushed count
- `pages_acked` is the consumer's own popped count (frozen while waiting)

### Spin Mechanism

Each iteration:
1. Reads `pages_received_ptr` via `reg_read()` to get the latest pushed count
2. Subtracts the consumer's own `pages_acked` to get the number of available tiles
3. Compares against `num_pages`

**Important asymmetry with CRBW:** Unlike `cb_reserve_back`, this loop does **not** call `invalidate_l1_cache()`. The `pages_received_ptr` points to a location that is updated by a different RISC-V core (typically BRISC or NCRISC updating via `cb_push_back`), and `reg_read()` bypasses the L1 cache.

### Waypoints

- **Wait marker**: `CWFW` (CB Wait Front Wait)
- **Done marker**: `CWFD` (CB Wait Front Done)

### Failure Modes

| Failure | Root Cause | Diagnosis |
|---|---|---|
| **Producer never pushes** | Reader kernel terminates without calling `cb_push_back`, or pushes fewer tiles than expected | Consumer at `CWFW`; producer at `D` (done) |
| **Producer hung at NOC barrier** | Reader kernel is stuck in `noc_async_read_barrier` waiting for data from DRAM or another core | Consumer at `CWFW`; producer at `NRBW` |
| **Cumulative count error** | The API explicitly warns: "in case multiple calls of `cb_wait_front(n)` are issued without a paired `cb_pop_front()` call, `n` is expected to be incremented by the user to be equal to a cumulative total of tiles." Issuing `cb_wait_front(8)` four times instead of `cb_wait_front(8)`, `cb_wait_front(16)`, `cb_wait_front(24)`, `cb_wait_front(32)` produces incorrect behavior. | Subtle hang partway through execution |
| **CB size not evenly divisible** | The API also warns: "CB total size must be an even multiple of the argument passed to this call." Violating this leads to wrap-around arithmetic errors. | Hang appears after specific iteration count |

### Non-Blocking Alternative

```c++
bool cb_pages_available_at_front(int32_t operand, int32_t num_pages);
```

---

## NRBW: `noc_async_read_barrier` -- NOC Read Barrier Wait

### Purpose

Waits until all outstanding `noc_async_read` operations on the current core have completed. After this call returns, all data requested by prior reads is guaranteed to be present in local L1 memory.

### Exact Code

```c++
void noc_async_read_barrier(uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::READ_BARRIER_START, false, noc);

    WAYPOINT("NRBW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_reads_flushed(noc));
    } else {
        while (!ncrisc_noc_reads_flushed(noc));
    }
    invalidate_l1_cache();
    WAYPOINT("NRBD");

    RECORD_NOC_EVENT(NocEventType::READ_BARRIER_END, false, noc);
}
```

### Exit Condition

**Dedicated NOC mode:**
```c++
NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == noc_reads_num_issued[noc]
```

The hardware counter of read responses received must equal the software counter of read requests issued.

**Dynamic NOC mode (Blackhole):**
```c++
NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) ==
    self_risc_acked + other_risc_acked
```

In dynamic mode, both RISC-V cores share the NOC, so the check must account for reads issued by both cores.

### Waypoints

- **Wait marker**: `NRBW` (NOC Read Barrier Wait)
- **Done marker**: `NRBD` (NOC Read Barrier Done)

### Failure Modes

1. **Read to invalid/unreachable address**: If `noc_async_read` targets a NOC address that does not correspond to any valid endpoint (wrong coordinates, address out of range), the read request may be dropped or routed incorrectly, and no response is ever generated.

2. **Target core in reset or powered down**: If the target core has been reset (e.g., during a partial chip reset), it cannot service the read request.

3. **NOC congestion causing packet loss**: Under extreme NOC load, packets may be delayed indefinitely, though this is rare in correctly functioning hardware.

4. **Counter mismatch**: If `noc_reads_num_issued[noc]` is corrupted (e.g., by a stray write to that memory location), the equality check can never succeed. In dynamic NOC mode, both BRISC and NCRISC maintain separate counters that are summed; corruption of either will cause a hang.

5. **Multi-chip read via Ethernet**: If the read crosses an Ethernet boundary and the remote chip is hung or the Ethernet link is down, the read response will never arrive.

6. **Incorrect NOC index**: Issuing reads on NOC 0 but waiting on NOC 1 (or vice versa) will cause a hang because the counters track different NOC instances.

---

## NWBW: `noc_async_write_barrier` -- NOC Write Barrier Wait

### Purpose

Waits until all outstanding non-posted `noc_async_write` operations on the current core have been **acknowledged** by their destinations.

### Exact Code

```c++
FORCE_INLINE
void noc_async_write_barrier(uint8_t noc = noc_index) {
    RECORD_NOC_EVENT(NocEventType::WRITE_BARRIER_START, false, noc);

    WAYPOINT("NWBW");
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        do {
            invalidate_l1_cache();
        } while (!ncrisc_dynamic_noc_nonposted_writes_flushed(noc));
    } else {
        while (!ncrisc_noc_nonposted_writes_flushed(noc));
    }
    invalidate_l1_cache();
    WAYPOINT("NWBD");

    RECORD_NOC_EVENT(NocEventType::WRITE_BARRIER_END, false, noc);
}
```

### Exit Condition

**Dedicated NOC mode:**
```c++
NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED) == noc_nonposted_writes_acked[noc]
```

**Dynamic NOC mode:**
```c++
NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED) ==
    self_risc_acked + other_risc_acked
```

### Waypoints

- **Wait marker**: `NWBW` (NOC Write Barrier Wait)
- **Done marker**: `NWBD` (NOC Write Barrier Done)

### Failure Modes

1. **Write to invalid/unreachable address**: No acknowledgment is generated for writes to nonexistent endpoints.

2. **Multicast write with incorrect `num_dests`**: The software counter `noc_nonposted_writes_acked[noc]` is incremented by `num_dests`. If `num_dests` does not match the actual number of cores in the multicast rectangle, the expected ack count will be wrong, and the barrier will either hang (too many expected) or complete prematurely (too few expected).

3. **Blackhole inline-write back-pressure**: On Blackhole, inline writes to L1 use all four memory ports and can hang when there is back-pressure. From `risc_attribs.h`: *"Inline writes use all 4 memory ports and may hang on Blackhole when there is back-pressure. This hang only manifests when the inline writes are issued to a L1 location. The workaround on BH is for inline writes to L1 to use noc async writes."*

4. **Fabric router counter mismatch (Blackhole)**: Known issue (#28758) where inline write counters must always be updated on BH as a workaround for fabric router hangs.

5. **Destination L1 address is invalid**: Writing to an address outside the valid L1 range on the destination core can cause the NOC hardware to silently discard the acknowledgement or stall the pipeline.

---

## NSW: `noc_semaphore_wait` -- NOC Semaphore Wait (Equality)

### Purpose

Waits until a local L1 semaphore equals a specific value. This is the primary inter-core synchronization primitive.

### Exact Code

```c++
FORCE_INLINE
void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    RECORD_NOC_EVENT(NocEventType::SEMAPHORE_WAIT, false, -1);

    WAYPOINT("NSW");
    do {
        invalidate_l1_cache();
    } while ((*sem_addr) != val);
    WAYPOINT("NSD");
}
```

### Exit Condition

`*sem_addr == val`

The L1 memory location pointed to by `sem_addr` must contain exactly `val`.

### Waypoints

- **Wait marker**: `NSW` (NOC Semaphore Wait)
- **Done marker**: `NSD` (NOC Semaphore Done)

### Failure Modes

1. **Signaling core never writes**: If the core that is supposed to set the semaphore is itself hung or has a bug that skips the `noc_semaphore_set`, the value never changes.

2. **Overshoot (the most dangerous failure mode)**: Because this waits for an **exact match** (`!=`), if the semaphore is incremented past the target value (e.g., two increments arrive before the check, jumping from 0 to 2, missing the expected value of 1), the condition will **never** be satisfied. In multi-producer scenarios, `noc_semaphore_wait_min` should be used instead.

3. **Wrong semaphore address**: If the signaling core writes to a different L1 address than the one being waited on, the semaphore is never updated at the expected location.

4. **NOC write delivering the semaphore value is lost**: If the `noc_inline_dw_write` or `noc_async_write` that delivers the semaphore value encounters a NOC error, the value never arrives.

5. **Cache coherence**: Without `invalidate_l1_cache()`, the RISC-V core would read a stale cached copy. The code correctly includes the invalidation, but custom semaphore wait loops that omit it will hang even if the semaphore has been updated in L1.

---

## NSMW: `noc_semaphore_wait_min` -- NOC Semaphore Wait Minimum

### Purpose

Waits until a local L1 semaphore is **greater than or equal to** a target value. This is the safer alternative to `noc_semaphore_wait` when the semaphore is incremented atomically and may be incremented multiple times before the check.

### Exact Code

```c++
FORCE_INLINE
void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    RECORD_NOC_EVENT(NocEventType::SEMAPHORE_WAIT, false, -1);

    WAYPOINT("NSMW");
    do {
        invalidate_l1_cache();
    } while ((*sem_addr) < val);
    WAYPOINT("NSMD");
}
```

### Exit Condition

`*sem_addr >= val`

### Waypoints

- **Wait marker**: `NSMW` (NOC Semaphore Min Wait)
- **Done marker**: `NSMD` (NOC Semaphore Min Done)

### Failure Modes

Same as `noc_semaphore_wait`, **except** failure mode #2 (overshoot) is eliminated by the `>=` comparison. This makes `noc_semaphore_wait_min` strictly more robust than `noc_semaphore_wait` for atomic-increment patterns.

---

## Additional Barrier Primitives

### NWFW -- `noc_async_writes_flushed` (NOC Write Flush Wait)

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `NWFW` (waiting) / `NWFD` (done) |
| **Condition** | All outstanding non-posted writes have departed from the local NIU (not necessarily received at destination) |
| **Failure Modes** | NOC command buffer full; NOC link-level flow control stall preventing writes from departing |

Less strict than `noc_async_write_barrier` (NWBW) -- waits for departure, not acknowledgment. Less likely to hang, but does not guarantee delivery.

### NPWW -- `noc_async_posted_writes_flushed` (NOC Posted Write Flush Wait)

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `NPWW` (waiting) / `NPWD` (done) |
| **Condition** | All outstanding posted writes have departed from the local NIU |

Posted writes do not require acknowledgment, so this only waits for local departure. Failure modes identical to NWFW.

### NABW -- `noc_async_atomic_barrier` (NOC Atomic Barrier Wait)

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `NABW` (waiting) / `NABD` (done) |
| **Condition** | All outstanding atomic operations (e.g., semaphore increments) have been acknowledged |

Same failure modes as NWBW. This waypoint also appears in the BRISC firmware itself during the `barrier_remote_cb_interface_setup` function on Blackhole -- a firmware-level synchronization point that ensures remote CB interface writes have landed before kernel execution begins. A hang at NABW during CB setup indicates a NOC issue during the firmware's own initialization, not a user kernel bug.

### NFBW through NFBD -- `noc_async_full_barrier` (NOC Full Barrier)

| Attribute | Detail |
|-----------|--------|
| **Waypoints** | `NFBW` / `NFCW` / `NFDW` / `NFEW` / `NFFW` / `NFBD` |
| **Condition** | All reads, non-posted writes, posted writes, and atomics have completed |

The full barrier is a sequence of five individual barriers executed in order:
1. `NFBW`: Waiting for reads to flush
2. `NFCW`: Waiting for non-posted writes to be sent
3. `NFDW`: Waiting for non-posted writes to be acknowledged
4. `NFEW`: Waiting for atomics to be acknowledged
5. `NFFW`: Waiting for posted writes to be sent
6. `NFBD`: All complete

The specific sub-waypoint where the core is stuck reveals which class of NOC transaction is incomplete.

---

## Firmware Synchronization Primitives

These are not user-facing APIs but appear in the firmware main loops and can cause hangs.

### GW -- Go-Wait

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `GW` (waiting) / `GD` (done) |
| **Location** | `brisc.cc` and `dm.cc` main loops |
| **Condition** | Waiting for `go_messages[go_message_index].signal == RUN_MSG_GO` or preload flag |

**Failure Modes:**
1. **Dispatcher never sends go signal.** Host-side bug, dispatch NOC write failure, or command queue corruption.
2. **Go signal written to wrong mailbox slot.** If `go_message_index` is out of sync between dispatcher and worker.

### NTW -- Subordinate Wait (NCRISC/TRISC Wait)

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `NTW` (waiting) / `NTD` (done) |
| **Location** | `brisc.cc` `wait_ncrisc_trisc()` function |
| **Condition** | `subordinate_sync->all == RUN_SYNC_MSG_ALL_SUBORDINATES_DONE` (all four bytes are `0x00`) |

Any subordinate core (NCRISC, TRISC0, TRISC1, TRISC2) hanging in its own kernel will cause BRISC to hang at NTW.

### W -- NCRISC Wait for BRISC

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `W` |
| **Location** | `ncrisc.cc` main loop, `wait_for_brisc_notification()` |
| **Condition** | `*ncrisc_run == RUN_SYNC_MSG_GO || *ncrisc_run == RUN_SYNC_MSG_LOAD` |

On Wormhole, NOP instructions are inserted into the spin loop to avoid hammering L1 while other cores are trying to work.

### SW -- Sync Register Wait

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `SW` (waiting) / `SD` (done) |
| **API Function** | `wait_for_sync_register_value(uintptr_t addr, int32_t val)` |
| **Condition** | Value at specified address equals `val` |

### SEW -- Subordinate ERISC Wait

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `SEW` (waiting) / `SED` (done) |
| **Location** | `active_erisc.cc` `wait_subordinate_eriscs()` |
| **Condition** | All subordinate ERISC cores have signaled completion |

### NKFW -- NOC Kernel Flush Wait (Post-Kernel Assertion Check)

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `NKFW` (waiting) / `NKFD` (done) |
| **Location** | `brisc.cc`, post-kernel assertion check in dynamic NOC mode |
| **Condition** | All NOC transaction counters are balanced (reads flushed, writes sent, writes flushed, atomics flushed, posted writes sent) |

This only fires when assertions are enabled. A failure here indicates a kernel bug: the kernel returned without properly barriering its NOC transactions. This is important on Blackhole with dynamic NOC mode, because residual transactions from one kernel can contaminate the next kernel's NOC state.

---

## NOC Command Buffer Wait Primitives

These appear inline during NOC transaction issuance, not as standalone barriers.

### NWPW / RP2W -- Command Buffer Ready Wait

| Attribute | Detail |
|-----------|--------|
| **Waypoints** | `NWPW` / `NWPD` for writes; `RP2W` / `RP2D` for reads |
| **Spin Mechanism** | `while (!noc_cmd_buf_ready(noc, cmd_buf));` |
| **Condition** | `NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY` |

```c++
inline bool noc_cmd_buf_ready(uint32_t noc, uint32_t cmd_buf) {
    return (NOC_CMD_BUF_READ_REG(noc, cmd_buf, NOC_CMD_CTRL) == NOC_CTRL_STATUS_READY);
}
```

This can hang if the NOC hardware is in a bad state (e.g., a previous transaction is stuck in the command buffer and never completes). This is a hardware-level hang that typically requires chip reset. It is distinct from the barrier waits: it occurs when *issuing* a transaction, not when waiting for completion.

---

## Transaction ID Barrier Primitives

For advanced NOC usage with transaction IDs (Blackhole and Quasar):

### NBTW -- Read Barrier with Transaction ID

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `NBTW` (waiting) / `NBTD` (done) |
| **API Function** | `noc_async_read_barrier_with_trid(uint32_t trid, uint8_t noc)` |
| **Condition** | `NIU_MST_REQS_OUTSTANDING_ID(trid) == 0` -- no outstanding requests with the given transaction ID |

```c++
void noc_async_read_barrier_with_trid(uint32_t trid, uint8_t noc = noc_index) {
    WAYPOINT("NBTW");
    while (!ncrisc_noc_read_with_transaction_id_flushed(noc, trid)) {
        continue;
    }
    invalidate_l1_cache();
    WAYPOINT("NBTD");
}
```

A hang here indicates that a specific transaction (identified by its ID) has not completed. This can occur if the TRID was never assigned to a transaction, if there is TRID reuse before the previous transaction completed, or if the TRID count limit is exceeded (max 255 on BH, 65535 on Quasar).

### NWTW -- Write Barrier with Transaction ID

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `NWTW` (waiting) / `NWTD` (done) |
| **API Function** | `noc_async_write_barrier_with_trid(uint32_t trid, uint8_t noc)` |
| **Condition** | Outstanding non-posted writes with the given transaction ID have been acknowledged |

### NFTW -- Write Flush with Transaction ID

| Attribute | Detail |
|-----------|--------|
| **Waypoint** | `NFTW` (waiting) / `NFTD` (done) |
| **API Function** | `noc_async_write_flush_with_trid(uint32_t trid, uint8_t noc)` |
| **Condition** | Outstanding non-posted writes with the given transaction ID have departed |

---

## The `assert_and_hang` Primitive

The `assert_and_hang` mechanism is documented in [01_what_is_a_hang.md](./01_what_is_a_hang.md#the-assert_and_hang-pattern-a-hang-by-design). It writes diagnostic data (line number, which RISC-V processor, assert type) to the assert mailbox and enters `while(1){}` on Tensix cores; ERISC cores instead record the assertion and exit back to base firmware, since they are not restarted between kernel launches. Note that `assert_and_hang` does not use the WAYPOINT mechanism -- it is identified by the assert mailbox contents, not by a waypoint code. During triage, it can look identical to a true hang if the watcher is not enabled -- always check the assert mailbox early in diagnosis.

---

## NOC Issuance Waypoints (Non-Blocking but Waypointed)

These are not blocking in the barrier sense, but they have associated waypoints because they may briefly spin on command buffer availability:

| Waypoint | Function | Description |
|----------|----------|-------------|
| `NAOW` / `NAOD` | `noc_async_read_one_packet` | Single-packet read issuance |
| `NARW` / `NARD` | `noc_async_read` (multi-packet) | Multi-packet read issuance |
| `NASW` / `NASD` | `noc_async_read_one_packet_set_state` | Read state setup |
| `NATW` / `NATD` | `noc_async_read_one_packet_with_state` | Read with pre-set state |
| `NAUW` / `NAUD` | `noc_async_read_set_state` | Read state setup (any length) |
| `NAVW` / `NAVD` | `noc_async_read_with_state` | Read with pre-set state (any length) |
| `NAWW` / `NAWD` | `noc_async_write` (multi-packet) | Multi-packet write issuance |
| `NMWW` / `NMWD` | `noc_async_write_multicast` (multi-packet) | Multicast write issuance |
| `NWIW` / `NWID` | `noc_inline_dw_write` | Inline 32-bit write |
| `NSIW` / `NSID` | `noc_semaphore_inc` | Semaphore increment issuance |
| `NIMW` / `NIMD` | `noc_inline_mcast_dw_write` | Inline multicast 32-bit write |
| `NRDW` / `NRDD` | `noc_rescale_dest_addr` | Destination address rescale |
| `NSTW` / `NSTD` | `noc_async_read_set_trid` | Set transaction ID for reads |
| `NWSW` / `NWSD` | `noc_async_write_set_trid` | Set transaction ID for writes |

A core stuck at one of these issuance waypoints indicates that the NOC command buffer is full (the issuance cannot proceed until a command buffer slot is freed). This is functionally equivalent to an `RP2W`/`NWPW` hang and usually requires chip reset.

---

## Ethernet-Specific Primitives

The ethernet dataflow API (`internal/ethernet/dataflow_api.h`) contains additional spin-wait patterns:

| Pattern | Condition |
|---------|-----------|
| `while (eth_txq_is_busy())` | Ethernet transmit queue is full |
| `while ((*sem_addr) != val)` inside `eth_noc_semaphore_wait` | Ethernet semaphore wait (calls `run_routing()` periodically) |
| `while (!ncrisc_noc_reads_flushed(noc_index))` | NOC read barrier for ethernet |
| `while (!ncrisc_noc_nonposted_writes_flushed(noc_index))` | NOC write barrier for ethernet |
| `while (erisc_info->channels[0].bytes_sent != 0)` | Waiting for ethernet channel to be ready |
| `while (!eth_is_receiver_channel_send_done(channel))` | Waiting for ethernet receive completion |

The `eth_noc_semaphore_wait` primitive is particularly noteworthy because it periodically calls `run_routing()` during its spin loop to keep the Ethernet routing firmware alive. If a hang occurs in this primitive, the Ethernet link may remain partially functional (routing continues) but the data mover is stuck. These primitives are highly hang-prone in multi-chip configurations because they depend on the remote chip's ethernet core being responsive.

---

## Summary Table: Blocking Primitives Quick Reference

| Waypoint | Function | Waits For | Most Common Hang Cause |
|----------|----------|-----------|----------------------|
| `CRBW` | `cb_reserve_back` | CB space freed by consumer | Consumer kernel hung/buggy |
| `CWFW` | `cb_wait_front` | CB tiles pushed by producer | Producer kernel hung/buggy |
| `NRBW` | `noc_async_read_barrier` | All reads completed | Remote core unreachable |
| `NWBW` | `noc_async_write_barrier` | All writes acknowledged | Destination address invalid |
| `NWFW` | `noc_async_writes_flushed` | All writes departed | NOC congestion |
| `NPWW` | `noc_async_posted_writes_flushed` | Posted writes departed | NOC congestion |
| `NABW` | `noc_async_atomic_barrier` | Atomics acknowledged | Remote core unreachable |
| `NFBW`-`NFFW` | `noc_async_full_barrier` | All NOC txns done | Any NOC hang cause |
| `NSW` | `noc_semaphore_wait` | Semaphore == val | Signaler hung; overshoot |
| `NSMW` | `noc_semaphore_wait_min` | Semaphore >= val | Signaler hung |
| `GW` | Go-wait (firmware) | Dispatcher go signal | Dispatch failure |
| `NTW` | Subordinate wait | All subordinates done | Subordinate kernel hung |
| `RP2W` | Read cmd buf ready | Cmd buf has capacity | NOC pipeline stalled |
| `NWPW` | Write cmd buf ready | Cmd buf has capacity | NOC pipeline stalled |
| `NBTW` | Read barrier (trid) | Reads with trid done | Remote unreachable |
| `NWTW` | Write barrier (trid) | Writes with trid acked | Destination invalid |
| `NFTW` | Write flush (trid) | Writes with trid departed | NOC congestion |

---

**Next:** [`03_hang_taxonomy.md`](./03_hang_taxonomy.md)
