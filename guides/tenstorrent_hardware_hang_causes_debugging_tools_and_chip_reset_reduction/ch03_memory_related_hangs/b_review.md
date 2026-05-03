# Agent B Review -- Chapter 3

## Pass 1

**Issues found:** 2

### Issue 1: Incorrect NUM_CIRCULAR_BUFFERS values per architecture
- **File:** `03_alignment_and_tile_size_mismatches.md`, `04_allocation_failures_and_silent_oom.md`
- **Location:** File 03 line 711 ("32 on WH/BH, 64 on Quasar"), File 04 line 486 ("32 on WH/BH, 64 on Quasar")
- **Category:** Factual error
- **Problem:** The chapter states `NUM_CIRCULAR_BUFFERS` is 32 on WH/BH and 64 on Quasar. In the codebase, `NUM_CIRCULAR_BUFFERS` is 32 on Wormhole and 64 on Blackhole (and by extension Quasar). BH uses 64, not 32.
- **Evidence:** `tt_metal/api/tt-metalium/circular_buffer_constants.h` lines 33-39:
  ```c++
  #if defined(ARCH_WORMHOLE)
  constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
  #else
  // Blackhole device and HOST compilation (uses max for array sizing)
  constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 64;
  #endif
  ```
- **Suggested fix:** Change both occurrences of "32 on WH/BH, 64 on Quasar" to "32 on WH, 64 on BH/Quasar". In file 03 (Hang Cause 3.3.11), line 711: change `NUM_CIRCULAR_BUFFERS` (32 on WH/BH, 64 on Quasar)` to `NUM_CIRCULAR_BUFFERS` (32 on WH, 64 on BH/Quasar)`. In file 04 (Hang Cause 3.4.7), line 486: same change.

### Issue 2: Wrong intra-chapter cross-reference (scenario 3.1.3 should be 3.1.8)
- **File:** `04_allocation_failures_and_silent_oom.md`
- **Location:** Line 482
- **Category:** Cross-chapter inconsistency (intra-chapter cross-reference error)
- **Problem:** The text says `TT_METAL_WATCHER_DISABLE_SANITIZE_READ_ONLY_L1` disables "the `DebugSanitizeNocAddrMailbox` check from scenario 3.1.3". Scenario 3.1.3 is "Silent Corruption of Runtime Arguments" which discusses `DebugSanitizeNocTargetInvalidXY`, not `DebugSanitizeNocAddrMailbox`. The `DebugSanitizeNocAddrMailbox` check is covered in scenario 3.1.8 ("Mailbox Region Write Corruption").
- **Evidence:** File `01_l1_memory_corruption_and_overflow.md`: Hang Cause 3.1.3 title is "Silent Corruption of Runtime Arguments" (line 240) with `DebugSanitizeNocTargetInvalidXY` (line 244). Hang Cause 3.1.8 title is "Mailbox Region Write Corruption (Without Watcher)" (line 497) which discusses `DebugSanitizeNocAddrMailbox` (return code 12) (line 501).
- **Suggested fix:** Change "from scenario 3.1.3" to "from scenario 3.1.8" on line 482 of `04_allocation_failures_and_silent_oom.md`.

**Verdict:** NEEDS REVISION
