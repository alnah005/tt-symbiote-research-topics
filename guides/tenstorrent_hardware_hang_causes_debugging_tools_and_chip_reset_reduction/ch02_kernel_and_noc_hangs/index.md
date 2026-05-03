# Chapter 2: Kernel-Level and NOC Hang Mechanisms

Chapter 1 established the vocabulary of hangs -- what they are, which blocking primitives produce them, and how they are classified across six categories. This chapter goes one level deeper: it examines the exact firmware and kernel code paths that produce hangs, with every hang cause documented in the **Symptom / Root Cause / Diagnosis Steps / Fix / Prevention** format. Each scenario is numbered with a 2.X.Y scheme for precise cross-referencing.

The four files in this chapter divide the hang surface into natural boundaries:

1. **RISC synchronization** -- the firmware-level protocols that coordinate the five (or more) RISC-V processors on each Tensix core, and how those protocols break down.
2. **Circular buffer deadlocks** -- the producer-consumer model that is the most common source of user-visible hangs, including the subtle API contract requirements that silently turn correct-looking code into deadlocks.
3. **NOC address sanitization and violations** -- the full validation pipeline that catches bad NOC addresses before they cause silent transaction failures, and what happens when validation is disabled.
4. **NOC barriers and semaphore hangs** -- the barrier and semaphore primitives that wait for NOC transactions to complete, the known hardware bugs that require software workarounds, and transaction ID barriers.

## Prerequisites

- **Chapter 1, [`01_what_is_a_hang.md`](../ch01_anatomy_of_a_hang/01_what_is_a_hang.md)**: Understanding of the RISC-V spin-loop model, the waypoint mechanism, and the distinction between hangs and other failure modes.
- **Chapter 1, [`02_blocking_primitives_taxonomy.md`](../ch01_anatomy_of_a_hang/02_blocking_primitives_taxonomy.md)**: Familiarity with the six primary blocking primitives (`cb_reserve_back`, `cb_wait_front`, `noc_async_read_barrier`, `noc_async_write_barrier`, `noc_semaphore_wait`, `noc_semaphore_wait_min`) and their waypoint codes.
- **Chapter 1, [`03_hang_taxonomy.md`](../ch01_anatomy_of_a_hang/03_hang_taxonomy.md)**: The six-category classification system, particularly categories 1 (kernel deadlocks) and 2 (NOC transaction failures).
- **Chapter 1, [`04_hang_causes_across_architectures.md`](../ch01_anatomy_of_a_hang/04_hang_causes_across_architectures.md)**: Architectural differences between Grayskull, Wormhole, Blackhole, and Quasar that affect synchronization and NOC behavior.

## Chapter Contents

| File | Focus | Key Waypoints |
|------|-------|---------------|
| [`01_risc_synchronization_and_deadlocks.md`](./01_risc_synchronization_and_deadlocks.md) | BRISC/NCRISC/TRISC synchronization protocols, subordinate mailbox, ERISC context switching, Quasar DM cores | `NTW`, `GW`, `NABW`, `NKFW`, `SEW` |
| [`02_circular_buffer_deadlocks.md`](./02_circular_buffer_deadlocks.md) | CB producer/consumer model, tile count mismatches, cumulative-total requirement, remote CBs | `CRBW`, `CWFW` |
| [`03_noc_address_sanitization_and_violations.md`](./03_noc_address_sanitization_and_violations.md) | NOC validation pipeline, all DebugSanitize return codes, deliberate hang mechanism, linked transactions | `while(1)` spin after violation |
| [`04_noc_barrier_and_semaphore_hangs.md`](./04_noc_barrier_and_semaphore_hangs.md) | Read/write barriers, mcast path reservation workaround, semaphore protocols, TRID barriers | `NRBW`, `NWBW`, `NSW`, `NSMW`, `NBTW`, `NWTW` |

**Covers research questions:** Q1 (kernel deadlocks, NOC deadlocks in detail), Q4 (CB overflow and L1/DRAM memory hangs via NOC sanitization).

---

**Next:** [`01_risc_synchronization_and_deadlocks.md`](./01_risc_synchronization_and_deadlocks.md)
