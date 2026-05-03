# Agent B Review -- Chapter 5

## Pass 1

**Issues found:** 5

---

### Issue 1: Deadlock avoidance direction requirements inverted for Torus topologies

- **File:** `01_ethernet_and_fabric_fundamentals.md`
- **Location:** Scenario 5.1.6, lines 280-287 (the pseudocode comment block describing when deadlock avoidance is needed)
- **Category:** Factual error
- **Problem:** The chapter states:
  > "Deadlock avoidance is required for:
  > - NORTH/SOUTH directions on Torus-X and Mesh topologies (turn channels)
  > - EAST/WEST directions on Torus-Y topologies (turn channels)
  > Not needed for fully linear topologies or directions without turns"

  This is inverted. The actual `need_deadlock_avoidance_support()` function in `tt_metal/fabric/fabric_context.cpp` (lines 344-362) returns `false` for the "torus_mismatch" case, which is defined as `(TORUS_X && NORTH/SOUTH) || (TORUS_Y && EAST/WEST)`. This means:
  - **TORUS_X** needs DA for **EAST/WEST** (the wrapped dimension), NOT NORTH/SOUTH
  - **TORUS_Y** needs DA for **NORTH/SOUTH** (the wrapped dimension), NOT EAST/WEST

  The chapter has the directions backwards for both Torus types. Additionally, the chapter claims "Mesh topologies" need DA, but `Topology::Mesh` falls through to `return false` in the actual code -- Mesh topologies do NOT require deadlock avoidance according to this function. Only `Ring` (always) and `Torus` (for wrapped dimensions) require it.

  A developer reading this chapter would enable deadlock avoidance on the wrong channels for Torus topologies, leaving the actually vulnerable channels unprotected while adding unnecessary overhead to safe channels.

- **Evidence:** `tt_metal/fabric/fabric_context.cpp` lines 344-362:
  ```cpp
  bool FabricContext::need_deadlock_avoidance_support(eth_chan_directions direction) const {
      if (topology_ == Topology::Ring) { return true; }
      if (topology_ == Topology::Torus) {
          const auto fabric_type = get_fabric_type(fabric_config_, is_ubb_galaxy_);
          const bool is_north_south = (direction == NORTH || direction == SOUTH);
          const bool is_east_west = (direction == EAST || direction == WEST);
          const bool torus_mismatch = (fabric_type == FabricType::TORUS_X && is_north_south) ||
                                      (fabric_type == FabricType::TORUS_Y && is_east_west);
          return !torus_mismatch;  // DA needed when NOT a mismatch
      }
      return false;  // Mesh, Linear, etc. do NOT need DA
  }
  ```
- **Suggested fix:** Replace the pseudocode comment block with:
  > "Deadlock avoidance is required for:
  > - All directions on Ring topology
  > - EAST/WEST directions on Torus-X topologies (the wrapped dimension that forms cycles)
  > - NORTH/SOUTH directions on Torus-Y topologies (the wrapped dimension that forms cycles)
  > - Both dimensions on full Torus (both X and Y wrapped)
  > Not needed for Mesh, Linear, or NeighborExchange topologies"

---

### Issue 2: `EriscDataMoverTerminationMode` enum values reversed

- **File:** `02_ccl_collective_operation_hangs.md`
- **Location:** Scenario 5.2.6, lines 244-249 (the enum definition code block)
- **Category:** Factual error
- **Problem:** The chapter presents the enum as:
  ```cpp
  enum class EriscDataMoverTerminationMode {
      WORKER_INITIATED,    // = 0 (implicit)
      MESSAGE_COUNT_REACHED // = 1 (implicit)
  };
  ```
  The actual enum in `ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp` line 36 is:
  ```cpp
  enum EriscDataMoverTerminationMode : uint32_t { MESSAGE_COUNT_REACHED = 0, WORKER_INITIATED = 1 };
  ```
  Two errors: (1) the values are swapped -- `MESSAGE_COUNT_REACHED` is 0 and `WORKER_INITIATED` is 1, not the other way around; (2) the actual enum is a plain `enum` with explicit `uint32_t` underlying type, not an `enum class`. A developer interpreting compile-time arguments or runtime argument values based on the chapter's enum ordering would misidentify the termination mode (thinking mode 0 is `WORKER_INITIATED` when it is actually `MESSAGE_COUNT_REACHED`).

- **Evidence:** `ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp` line 36:
  ```cpp
  enum EriscDataMoverTerminationMode : uint32_t { MESSAGE_COUNT_REACHED = 0, WORKER_INITIATED = 1 };
  ```
- **Suggested fix:** Replace the enum snippet with the actual definition:
  ```cpp
  enum EriscDataMoverTerminationMode : uint32_t {
      MESSAGE_COUNT_REACHED = 0,  // EDM terminates after processing N messages
      WORKER_INITIATED = 1        // Workers explicitly signal EDM to terminate
  };
  ```

---

### Issue 3: All cross-chapter prerequisite links use non-existent directory names

- **File:** `index.md`
- **Location:** Lines 5, 9-14 (Prerequisites section); also `03_topology_and_mesh_configuration_hangs.md` lines 3 and 504 (Next chapter link)
- **Category:** Cross-chapter inconsistency (broken links)
- **Problem:** All cross-chapter links in the Prerequisites section use directory names that do not exist:
  - `../ch1_final/` -- actual directory is `../ch01_anatomy_of_a_hang/`
  - `../ch2_final/` -- actual directory is `../ch02_kernel_and_noc_hangs/`
  - `../ch3_final/` -- actual directory is `../ch03_memory_related_hangs/`
  - `../ch4_final/` -- actual directory is `../ch04_dispatch_and_host_device_hangs/`
  - `../ch6_final/` -- no Chapter 6 directory exists yet

  There are 8 broken prerequisite links in `index.md` and 2 broken "Next chapter" links in `03_topology_and_mesh_configuration_hangs.md`. A reader following any of these links will get a 404.

- **Evidence:** `ls` of the parent directory shows: `ch01_anatomy_of_a_hang`, `ch02_kernel_and_noc_hangs`, `ch03_memory_related_hangs`, `ch04_dispatch_and_host_device_hangs`, `ch05_multi_chip_and_ccl_hangs`, `plan.md`. No directory matching `ch*_final` exists.
- **Suggested fix:** Replace all `../ch1_final/` with `../ch01_anatomy_of_a_hang/`, `../ch2_final/` with `../ch02_kernel_and_noc_hangs/`, `../ch3_final/` with `../ch03_memory_related_hangs/`, `../ch4_final/` with `../ch04_dispatch_and_host_device_hangs/`. For the Chapter 6 link, either create a placeholder or remove the link until Chapter 6 exists.

---

### Issue 4: Wrong filename in cross-chapter reference table

- **File:** `03_topology_and_mesh_configuration_hangs.md`
- **Location:** Line 498 (Cross-Chapter Reference Table, row for Scenario 5.2.10)
- **Category:** Cross-chapter inconsistency (incorrect filename)
- **Problem:** The table references Ch3 file `01_l1_and_buffer_corruption_hangs.md`, but the actual filename is `01_l1_memory_corruption_and_overflow.md`. A reader looking for the referenced file will not find it.
- **Evidence:** `ls ch03_memory_related_hangs/` shows: `01_l1_memory_corruption_and_overflow.md`, `02_dram_and_noc_backpressure.md`, `03_alignment_and_tile_size_mismatches.md`, `04_allocation_failures_and_silent_oom.md`.
- **Suggested fix:** Change `01_l1_and_buffer_corruption_hangs.md` to `01_l1_memory_corruption_and_overflow.md`.

---

### Issue 5: Chapter 5 index claims `index.md` navigation link to non-existent Scenario 5.1.3 anchor is about "link retraining" but the anchor text differs

- **File:** `index.md`
- **Location:** Lines 18 and 25 (the "How to Use This Chapter" section)
- **Category:** Cross-chapter inconsistency (internal navigation mismatch)
- **Problem:** Line 18 says: "If a watcher reports an Ethernet link went down... start with [Section 01, Scenario 5.1.1]" and points to anchor `#511-ethernet-link-down-during-active-operation`. This is correct and consistent.

  However, line 25 says: "If a hang occurs after a link retraining event: start with [Section 01, Scenario 5.1.3](./01_ethernet_and_fabric_fundamentals.md#513-link-retraining-during-active-data-transfer)." This anchor is consistent with the actual heading.

  The actual issue is at line 19: "If fabric telemetry shows `RouterState::INITIALIZING`... start with [Section 01, Scenario 5.1.3](./01_ethernet_and_fabric_fundamentals.md#513-edm-handshake-failure-during-initialization)." This points to Scenario 5.1.3 but the actual Scenario for EDM handshake failure is **5.1.4** (not 5.1.3). Scenario 5.1.3 is "Link Retraining During Active Data Transfer," not "EDM Handshake Failure During Initialization." A reader following this link would land on the wrong scenario.

- **Evidence:** In `01_ethernet_and_fabric_fundamentals.md`:
  - Line 157: `### 5.1.3 Link Retraining During Active Data Transfer`
  - Line 191: `### 5.1.4 EDM Handshake Failure During Initialization`
- **Suggested fix:** Change line 19 from `Scenario 5.1.3` to `Scenario 5.1.4` and update the anchor from `#513-edm-handshake-failure-during-initialization` to `#514-edm-handshake-failure-during-initialization`.

---

## Format Compliance

All 35 hang scenarios (12 + 12 + 11) consistently use the required 5-part format: **Symptom / Root Cause / Diagnosis Steps / Fix / Prevention**. Scenario numbering uses the `5.X.Y` format throughout with no gaps or duplicates. No format violations detected.

## What the Chapter Gets Right

The chapter is impressively thorough and largely accurate. Code snippets for `hang_on_down_link()`, `sender_side_handshake()`, `distance_behind()`, `check_if_send_socket()`, `ReceiverChannelCounterBasedResponseCreditSender`, `BidirectionalFabricSocket::create()`, and the `CoordinatedEriscContextSwitchState` enum all match the actual codebase. The `MAGIC_HANDSHAKE_VALUE = 0xAA`, `NUM_TRANSACTION_IDS` (8 with DA, 4 without), `super_speedy_mode` incompatibility with deadlock avoidance, termination master runtime arg offsets (1, 15, 16), and the `all_gather` reverse / `reduce_scatter` forward iteration order are all verified correct against the tt-metal source.

---

**Verdict:** NEEDS REVISION

Issues 1 and 2 are factual errors that would mislead developers. Issue 1 is the most critical: inverted deadlock avoidance directions could cause actual deadlocks in production Torus deployments. Issues 3-5 are broken links and cross-references that impair navigation. All five issues have straightforward fixes.
