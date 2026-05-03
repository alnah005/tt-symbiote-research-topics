# Chapter 1: Anatomy of a Hang -- Core Concepts and Taxonomy

A hardware hang is the single most disruptive failure mode on Tenstorrent silicon. Unlike a crash that produces an error code or corrupted output that fails a test, a hang produces *nothing*: the chip goes silent, the host blocks indefinitely, and the only recourse is often a full board-level reset -- an operation that can take tens of seconds and disrupt every workload sharing that device. Understanding what hangs are, why they happen, and how they manifest is the foundation for every debugging technique in the chapters that follow.

This chapter establishes the foundational vocabulary and mental models needed to understand, classify, and ultimately resolve hardware hangs on Tenstorrent devices. We define what a hang is at the hardware level, catalog every blocking primitive that can produce one, present a systematic taxonomy of hang categories, and trace how hang behavior differs across the Grayskull, Wormhole, Blackhole, and Quasar architectures.

## Chapter Contents

1. [`01_what_is_a_hang.md`](./01_what_is_a_hang.md) -- Precise definition, observable symptoms, the fundamental RISC-V spin-loop model, hang lifecycle, the `assert_and_hang` and `debug_sanitize_post_addr_and_hang` deliberate-hang mechanisms, and the 5-part diagnostic format used throughout this guide.

2. [`02_blocking_primitives_taxonomy.md`](./02_blocking_primitives_taxonomy.md) -- Complete catalog of every device-side blocking call: the six primary API primitives (`cb_reserve_back`, `cb_wait_front`, `noc_async_read_barrier`, `noc_async_write_barrier`, `noc_semaphore_wait`, `noc_semaphore_wait_min`), firmware synchronization points, NOC command buffer waits, transaction-ID barriers, and ethernet-specific spin patterns. Each entry covers the exact spin-loop code, the exit condition, the WAYPOINT marker, and the failure modes.

3. [`03_hang_taxonomy.md`](./03_hang_taxonomy.md) -- Six-category classification system (kernel, NOC, memory, dispatch, multi-chip, host-device) with symptoms, RISC-V core involvement, compounding effects, a decision tree for rapid triage, a symptoms cross-reference matrix, and a hang frequency ranking.

4. [`04_hang_causes_across_architectures.md`](./04_hang_causes_across_architectures.md) -- How Grayskull, Wormhole, Blackhole, and Quasar differ in their susceptibility to hangs, architecture-specific failure modes such as Blackhole inline-write back-pressure, Wormhole NCRISC reset sequences, BH L1 data cache and relaxed memory ordering concerns, and scale-dependent hang categories from single-chip to Galaxy configurations.

---

**Next:** [`01_what_is_a_hang.md`](./01_what_is_a_hang.md)
