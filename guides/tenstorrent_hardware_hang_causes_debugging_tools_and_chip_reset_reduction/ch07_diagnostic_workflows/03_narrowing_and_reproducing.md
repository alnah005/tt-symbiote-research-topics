# 7.3 Narrowing and Reproducing

[Previous: Diagnosing by Hang Category](./02_diagnosing_by_hang_category.md) | [Next: Reading Watcher and Triage Output](./04_reading_watcher_and_triage_output.md)

---

When the initial triage (Section 01) and category-specific diagnosis (Section 02) do not yield a clear root cause, the next step is to systematically narrow the problem: reduce the workload, isolate the failing component, and produce a minimal, reproducible test case. This section covers every narrowing technique available in tt-metal, from coarse (binary search over ops) to fine (single-core, single-iteration reproduction), along with strategies for the particularly challenging case of intermittent hangs.

**Prerequisites:** [Section 01, Initial Triage](./01_initial_triage.md), [Section 02, Diagnosing by Hang Category](./02_diagnosing_by_hang_category.md) (to have ruled out easily diagnosed cases). Familiarity with [Chapter 4, `03_trace_replay_and_lightmetal.md`](../ch04_dispatch_and_host_device_hangs/03_trace_replay_and_lightmetal.md) is helpful for the trace/LightMetal sections.

---

## Technique 1: Binary Search with Synchronize() Checkpoints

**When to use:** You have a model or workload with many ops, and you do not know which op hangs.

### Concept

Insert `ttnn.synchronize_device(device)` calls (or `Synchronize()` in C++) between ops. Each synchronize forces the host to wait for all preceding ops to complete before submitting the next one. The last synchronize that returns successfully identifies the operation *before* the hang; the first synchronize that does not return identifies the operation *after* the hang.

### Procedure

```python
# Step 1: Coarse bisection -- divide the model into halves
output = first_half_of_model(input)
ttnn.synchronize_device(device)  # <-- checkpoint A
output = second_half_of_model(output)
ttnn.synchronize_device(device)  # <-- checkpoint B

# If checkpoint A returns but B does not: the hang is in the second half.
# If checkpoint A does not return: the hang is in the first half.

# Step 2: Repeat, dividing the hanging half into quarters, eighths, etc.
# Continue until you identify the single op that hangs.
```

### Automation Example

```python
def find_hanging_op(operations, device):
    """Binary search for the first operation that causes a hang."""
    low, high = 0, len(operations) - 1
    while low < high:
        mid = (low + high) // 2
        for i in range(mid + 1):
            operations[i](device)
        try:
            ttnn.synchronize_device(device, timeout=30)  # 30s timeout
            low = mid + 1  # Sync succeeded, bug is after mid
        except:
            high = mid  # Sync failed/timeout, bug is at or before mid
        # Reset device for next iteration
    return low
```

### Important Caveats

- **Synchronize changes timing.** Some hangs are timing-dependent and may disappear when synchronize points are added. If this happens, the hang is likely a race condition. Use the intermittent strategies below.
- **Synchronize does not test dispatch in isolation.** If the hang is in the dispatch pipeline itself, synchronize calls may mask it by draining the pipeline between ops.
- **Order-dependent bugs:** The hang may require ops A, B, and C to all have been dispatched before C hangs (due to accumulated state). If you split between B and C, the Synchronize after B may change the behavior. Test without Synchronize to confirm the isolated set still reproduces.
- **For multi-device:** Synchronize each device individually to identify which device hangs.

---

## Technique 2: null_kernels Mode

**When to use:** You want to test whether the hang is in the dispatch/loading infrastructure or in the kernel execution itself.

### Concept

`null_kernels` mode replaces all kernel code with minimal no-op implementations. The dispatch pipeline still loads kernel binaries and sends go signals, but the kernels immediately return without executing any CB, NOC, or compute operations.

### Usage

```bash
# Using environment variable (preferred for quick testing):
export TT_METAL_NULL_KERNELS=1
python3 your_model.py

# Or via API in test code:
# program.set_null_kernels(true);
```

### Interpretation

| Still Hangs? | Diagnosis |
|-------------|-----------|
| Yes | Dispatch pipeline bug, configuration error, or L1 layout corruption. Focus on [Ch4](../ch04_dispatch_and_host_device_hangs/) and [Ch3](../ch03_memory_related_hangs/01_l1_memory_corruption_and_overflow.md). |
| No | Kernel execution bug. Focus on [Ch2](../ch02_kernel_and_noc_hangs/) (CB, NOC, semaphore issues) and the specific kernel code. |

### Caveats

- null_kernels will produce incorrect results (since kernels do not actually compute anything), so validation checks after the run will fail. This is expected.
- Some hangs are caused by the *interaction* between kernel execution and the dispatch system (e.g., kernels not signaling completion). null_kernels avoids this because the empty stubs exit immediately.

---

## Technique 3: kernels_early_return Mode

**When to use:** You want a middle ground between null_kernels (no kernel code at all) and full execution. `kernels_early_return` loads the full kernel binary but skips execution.

### Concept

Like null_kernels, but the actual kernel ELF is loaded into L1 (testing the full binary loading path). The kernel simply returns immediately without executing its body. This distinguishes between:

- Binary loading/configuration issues (will hang even in early_return mode)
- Kernel logic issues (will not hang in early_return mode)

### Usage

```bash
# Using environment variable:
export TT_METAL_KERNELS_EARLY_RETURN=1
python3 your_model.py
```

### Comparison Table

| Environment | Kernel Loaded? | Kernel Runs? | Binary Size | Useful For |
|-------------|---------------|-------------|-------------|------------|
| Normal | Yes | Yes | Normal | Baseline |
| `TT_METAL_NULL_KERNELS=1` | Yes (empty) | No | Minimal | Isolating dispatch vs kernel bugs |
| `TT_METAL_KERNELS_EARLY_RETURN=1` | Yes (normal) | No (returns) | Normal | Isolating binary loading vs logic bugs |

### Narrowing Ladder

```
Full execution:  HANGS
  |
  v
kernels_early_return:  ?
  |
  +-- HANGS --> Problem in binary loading, L1 layout, dispatch config.
  |             Not in kernel logic.
  |
  +-- OK --> Problem is in kernel execution.
       |
       v
null_kernels:  ?
  |
  +-- HANGS --> Problem in dispatch pipeline (not kernel-related at all).
  |
  +-- OK --> Problem is kernel loading (binary placement in L1, CB config).
             The full binary loads fine but early-return skips the buggy path.
```

---

## Technique 4: Slow Dispatch Mode

**When to use:** You suspect the hang is related to the fast dispatch pipeline itself (prefetch/dispatch kernel coordination, command queue management, go signal delivery).

### Concept

`TT_METAL_SLOW_DISPATCH_MODE=1` bypasses the entire fast dispatch infrastructure. Instead of writing commands to a hugepage for the prefetch kernel to consume, the host directly writes kernel binaries and configuration to L1 via PCIe MMIO and triggers RISC-V execution directly.

### Usage

```bash
export TT_METAL_SLOW_DISPATCH_MODE=1
python3 your_model.py
```

### Interpretation

| Fast Dispatch | Slow Dispatch | Diagnosis |
|--------------|---------------|-----------|
| HANGS | HANGS | Bug is in the kernel execution, not dispatch. Focus on [Ch2](../ch02_kernel_and_noc_hangs/) and [Ch3](../ch03_memory_related_hangs/). |
| HANGS | OK | Bug is in the fast dispatch pipeline. Focus on [Ch4](../ch04_dispatch_and_host_device_hangs/). |
| OK | HANGS | Extremely unlikely. Timing difference may be exposing a race. Investigate with timing perturbation. |

### Limitations

- Slow dispatch mode is **significantly slower** (10-100x for small operations).
- Some features are not available in slow dispatch (trace replay, certain multi-chip operations).
- Not all TTNN ops may work correctly in slow dispatch mode.
- Slow dispatch changes the NOC traffic pattern, so NOC-related bugs may not reproduce.

---

## Technique 5: Single-Op Isolation

**When to use:** Binary search (Technique 1) has identified a specific op that hangs. Now extract it into a standalone test to enable rapid iteration.

### Procedure

1. **Identify the op:** From binary search or `dump_running_operations`, you know (e.g.) `ttnn.matmul` with specific tensor shapes hangs.

2. **Create a minimal reproduction:**

```python
import ttnn

device = ttnn.open_device(device_id=0)

# Reproduce the exact tensor shapes and configuration
a = ttnn.from_torch(torch.randn(1, 1, 1024, 1024), device=device, layout=ttnn.TILE_LAYOUT)
b = ttnn.from_torch(torch.randn(1, 1, 1024, 1024), device=device, layout=ttnn.TILE_LAYOUT)

# The hanging op
result = ttnn.matmul(a, b)
ttnn.synchronize_device(device)

ttnn.close_device(device)
```

3. **Enable all diagnostics:**

```bash
export TT_METAL_WATCHER=120
export TT_METAL_DPRINT_CORES=0,0-7,7
export TT_METAL_DPRINT_RISCVS=0,1,2,3,4
export TT_METAL_NOC_DEBUG_DUMP=1
```

4. **Iterate:** Vary tensor shapes, data types, core grids, and memory configurations to find the minimal triggering condition.

### Multi-Chip Extension: Single-Op CCL Isolation

For CCL operations, the isolation test must use a MeshDevice:

```python
import ttnn

mesh_device = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 8),  # T3K: 1x8 mesh
    dispatch_core_type=ttnn.DispatchCoreType.ETH,
)

input_tensor = ttnn.from_torch(
    torch.randn(1, 1, 32, 1024),
    device=mesh_device,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dim=3),
)

output = ttnn.all_gather(input_tensor, dim=3, num_links=1)
mesh_device.synchronize()

ttnn.close_mesh_device(mesh_device)
```

Key parameters to match from the failing workload: tensor shape, dtype, shard/replicate mapping, topology (Ring vs. Linear), `num_links`, memory configuration.

### Tips for Effective Isolation

- Preserve the exact **data types, shapes, memory layouts, and shard specs** from the original.
- If the op uses program cache, test both with and without cache hits (first run vs subsequent runs).
- If the original model uses trace capture, test both with and without trace.
- If a standalone reproduction cannot reproduce the hang, the issue is likely in buffer lifecycle management or state accumulated from prior operations.

---

## Technique 6: Multi-Device Narrowing and Mesh Bisection

**When to use:** The hang occurs on a multi-chip system (Galaxy, T3K, N300) and may be scale-dependent.

### Scale Narrowing Ladder

```
Galaxy (32+ chips):  HANGS
  |
  v
T3K (8 chips):  ?
  |
  +-- HANGS --> Not Galaxy-specific. Continue narrowing.
  |
  +-- OK --> Galaxy topology or scale is involved. Focus on
             [Ch5, 03_topology_and_mesh_configuration_hangs.md].
  |
  v
N300 (2 chips):  ?
  |
  +-- HANGS --> Multi-chip but not T3K-specific.
  |
  +-- OK --> T3K-specific (ring topology, relay routing, etc.).
  |
  v
Single chip:  ?
  |
  +-- HANGS --> Not multi-chip-related.
  |
  +-- OK --> Multi-chip specific. Focus on [Ch5].
```

### Mesh Bisection (Multi-Chip Specific)

Instead of bisecting operations, bisect the number of devices:

1. **Reproduce on the full mesh first.**
2. **Reduce to half the mesh.** Create a submesh with half the devices:
   ```python
   mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))  # Instead of 1x8
   ```
3. **If yes:** Root cause is within this submesh. Reduce further.
4. **If no:** Try the other half (devices 4-7 instead of 0-3).
5. **If neither half hangs alone:** The bug requires interaction between the two halves.

### Submesh Selection Strategy

| Grouping | What It Tests |
|----------|--------------|
| Devices 0-3 vs. 4-7 | Left half vs. right half of T3K |
| Even devices vs. odd devices | Tests non-adjacent devices |
| Single device | Tests if bug is purely local |
| Two adjacent devices | Minimum multi-chip configuration |
| All devices except one | Removes one device at a time to find the "bad" one |

The "all except one" strategy is particularly effective: if removing device N makes the hang go away, device N is the root cause (or its Ethernet connections are faulty).

### Using TT_METAL_VISIBLE_DEVICES

```bash
export TT_METAL_VISIBLE_DEVICES=0        # Single device
export TT_METAL_VISIBLE_DEVICES=0,1      # Two devices
export TT_METAL_VISIBLE_DEVICES=0,1,2,3  # Four devices
python3 your_model.py
```

---

## Technique 7: Core Grid Reduction

**When to use:** A single-chip, single-op hang that involves many cores. Reducing the core grid helps isolate whether the hang is caused by inter-core coordination or is a per-core kernel bug.

### Procedure

1. **Reduce to a 1x1 core grid** (single core execution):
   ```python
   result = ttnn.matmul(a, b, core_grid=ttnn.CoreGrid(y=1, x=1))
   ```
2. If the hang disappears on a 1x1 grid, gradually increase: 1x2, 2x1, 2x2, etc.
3. The grid size at which the hang first appears reveals the inter-core dependency.

### What Different Results Mean

| Grid Result | Interpretation |
|-------------|---------------|
| 1x1 hangs | Bug is in the kernel itself, not inter-core communication |
| 1x1 works, NxN hangs | Bug is in multi-core coordination (CB flow, semaphores, mcast) |
| Only specific positions hang | Hardware issue (harvested core map) or addressing bug |
| Only large grids hang | Resource contention or NOC congestion issue |

---

## Technique 8: Intermittent Hang Strategies

Intermittent hangs are the hardest to debug because the root cause is often a race condition whose outcome depends on instruction-level timing. These strategies increase the probability of reproduction or help identify the racing operations.

### Strategy 8a: Stress Testing (Loop Reproduction)

```bash
export TT_METAL_WATCHER=120
export TT_METAL_WATCHER_APPEND=1  # Append to log instead of overwriting

for i in $(seq 1 1000); do
    echo "=== Run $i ==="
    python3 your_model.py
    if [ $? -ne 0 ]; then
        echo "Failed on run $i"
        break
    fi
done
```

If the hang occurs at iteration N, you have a reproduction case. Enable watcher and run again -- the hang may occur at a different iteration but will now be captured.

### Strategy 8b: Timing Perturbation with Debug Delays

Debug delays artificially slow down NOC transactions on specific cores, perturbing the timing to make race conditions either more or less likely:

```bash
# Add read delays to cores 0,0 through 3,3
export TT_METAL_READ_DEBUG_DELAY_CORES=0,0-3,3
# Add write delays
export TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0-3,3
# Add atomic operation delays
export TT_METAL_ATOMIC_DEBUG_DELAY_CORES=0,0-3,3
```

**How to use effectively:**
1. Run the workload with delays on all cores. Does the hang become more or less frequent?
2. If more frequent: the delay is widening the race window. Focus on the delayed operations as the race participants.
3. If less frequent: try reversing which cores get delays. The goal is to find the delay pattern that makes the hang deterministic.

### Strategy 8c: Compute Timing Perturbation

For race conditions in the compute pipeline (unpack/math/pack coordination):

```cpp
// tt_metal/hw/inc/api/debug/timing_perturbation.h
#include "debug/timing_perturbation.h"
```

This inserts configurable NOP sequences into the unpack, math, and pack pipelines, changing the relative timing of compute operations. See [Chapter 6, `06_debug_delay_and_timing_perturbation.md`](../ch06_debugging_tools/06_debug_delay_and_timing_perturbation.md).

### Strategy 8d: Memory Initialization

Stale memory from previous runs can mask or create intermittent issues:

```bash
export TT_METAL_CLEAR_L1=1
export TT_METAL_CLEAR_DRAM=1
python3 your_model.py
```

If the bug becomes deterministic with memory clearing, the root cause is a stale state dependency.

### Strategy 8e: Dispatch Progress Heartbeats

For intermittent dispatch-related hangs:

```bash
export TT_METAL_DISPATCH_PROGRESS_UPDATE_MS=5000
```

This configures dispatch kernels to write periodic heartbeat signals. If heartbeats stop, the host detects the stall within 5 seconds instead of waiting for the full timeout.

### Strategy 8f: Auto-Capture on Timeout

```bash
export TT_METAL_OPERATION_TIMEOUT_SECONDS=30
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="./tools/tt-triage.py --verbosity=4 > /tmp/triage_dump.txt 2>&1"
export TT_METAL_INSPECTOR_SERIALIZE_ON_DISPATCH_TIMEOUT=1
```

This auto-captures diagnostic data when a timeout fires, preserving the exact sequence of programs and kernels that were in flight.

---

## Technique 9: The hang_device Test Operation

**When to use:** You want to verify that your debugging tools and triage scripts correctly detect and report hangs before relying on them for real debugging.

### Concept

The `hang_device` operation is a test-only TTNN op that deliberately induces a hang on the device:

```python
# ttnn/cpp/ttnn/operations/experimental/test/hang_device/
import ttnn
ttnn.experimental.test.hang_device(device)
```

### Usage

1. Run `hang_device` with watcher enabled to verify waypoint capture.
2. Run `hang_device` and then `tt-triage` to verify all triage scripts produce output.
3. Verify that `TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE` fires correctly with `hang_device`.
4. Use `hang_device` as a CI health check: if triage cannot detect a deliberate hang, the tooling is broken.

---

## Technique 10: Trace Capture/Replay and LightMetal for Deterministic Reproduction

**When to use:** You need a deterministic reproduction of the exact command stream that caused the hang, especially for intermittent hangs or for sharing a reproduction with another developer.

### Trace Capture/Replay

```python
# Capture
tid = ttnn.begin_trace_capture(device, cq_id=0)
result = the_hanging_op(inputs)
ttnn.end_trace_capture(device, tid, cq_id=0)

# Replay (deterministic)
ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
```

If the hang reproduces during replay, you have a self-contained reproduction case. If it does not, the hang may depend on host-device timing that trace replay eliminates.

### Trace-Specific Failure Modes

If the hang only occurs during trace replay but not during non-traced execution, check for:
- `synchronize_device()` calls inside the traced region (not allowed)
- `ttnn.from_torch()` or other host-allocated buffers inside the trace (not allowed)
- Semaphore handle cycling baked into the trace at capture time that does not match replay state
- Tensor allocations/deallocations inside the trace that change L1 addresses between captures

### LightMetal Capture/Replay

LightMetal operates at a higher level: it records the entire sequence of Metal API calls into a file that can be replayed deterministically:

```bash
# Capture (host-side, minimal overhead)
export TT_METAL_LIGHTMETAL_CAPTURE=1
python3 your_model.py
# Produces a .lightmetal file

# Replay (deterministic, on the same or different machine)
./build/tools/lightmetal_replay the_capture.lightmetal
```

LightMetal replay is especially valuable for:
- Sharing reproduction cases across teams (the `.lightmetal` file is self-contained)
- Reproducing on a different device to test hardware vs. software theories
- Replaying in simulation environments

### Multi-Chip Trace Extension

Trace capture on multi-chip systems has additional failure modes:
- **Async CCL semaphore cycling:** The `TT_CCL.get_and_cycle_*` methods cycle through double-buffered semaphore handles. During trace capture, a specific handle is baked into the command buffer. On replay, the host counter continues cycling but the trace always uses the capture-time handle. Reset semaphore indices to match capture-time state before each replay.
- **Cross-device synchronization barriers baked into trace:** If a trace was captured with specific inter-device timing, replay assumes the same timing.

Reference: `tt_metal/impl/lightmetal/lightmetal_capture.hpp`, `tt_metal/impl/lightmetal/lightmetal_replay_impl.hpp` (see [Chapter 4, `03_trace_replay_and_lightmetal.md`](../ch04_dispatch_and_host_device_hangs/03_trace_replay_and_lightmetal.md)).

---

## Technique 11: Binary Validation and Clean State Initialization

**When to use:** You suspect kernel binary corruption or stale L1/DRAM state is causing the hang.

### Binary Validation

```bash
export TT_METAL_VALIDATE_PROGRAM_BINARIES=1
```

After loading kernel binaries into L1, this reads them back and compares against the original ELF. If validation fails, binary loading is corrupted -- either by a SW bug in the dispatch pipeline or a HW issue (PCIe, SRAM).

### Clean State Initialization

```bash
export TT_METAL_CLEAR_L1=1
export TT_METAL_CLEAR_DRAM=1
```

Zeros all memory before each program, eliminating stale state as a variable. If the hang disappears, the root cause is stale data from prior operations.

---

## Narrowing Decision Flowchart

```
HANG: root cause unknown
  |
  +-- Is the hang in a specific op? --> Technique 1 (Binary Search) to find it
  |     |
  |     v
  |   Isolated the op.
  |     |
  |     +-- Is it in kernel execution or dispatch?
  |     |     |
  |     |     +-- Technique 2 (null_kernels): still hangs? --> Dispatch issue
  |     |     |                                  no hang? --> Kernel issue
  |     |     |
  |     |     +-- Technique 3 (kernels_early_return): still hangs? --> Binary/config issue
  |     |     |                                         no hang? --> Kernel logic issue
  |     |     |
  |     |     +-- Technique 4 (slow dispatch): hangs in slow dispatch too? --> Kernel issue
  |     |                                      only in fast dispatch? --> Dispatch issue
  |     |
  |     +-- Is it a multi-chip issue?
  |     |     |
  |     |     +-- Technique 6 (scale reduction/mesh bisection):
  |     |           hangs on single chip? --> Not multi-chip
  |     |           only multi-chip? --> Ch5
  |     |
  |     +-- Is it a specific core?
  |     |     |
  |     |     +-- Technique 7 (grid reduction): 1x1 hangs? --> Per-core kernel bug
  |     |                                        only NxM? --> Inter-core coordination bug
  |     |
  |     +-- Is it stale state?
  |           |
  |           +-- Technique 11 (clean state): disappears with CLEAR_L1? --> Stale state bug
  |
  +-- Is the hang intermittent? --> Technique 8 (stress, delays, perturbation, auto-capture)
  |     |
  |     +-- Now reproducible? --> Continue with Techniques 1-7
  |     +-- Still intermittent? --> Technique 10 (LightMetal capture for deterministic replay)
  |
  +-- Need to verify tooling works? --> Technique 9 (hang_device)
```

---

## War Story: Binary Search Reveals a Stale Sharding Descriptor

**Symptom:** A 70B LLM decode hangs on the 3rd token generation. The first two tokens decode successfully.

**Narrowing:**
1. Binary search with Synchronize() pinpoints the hang to a specific `ttnn.all_gather` operation in the attention layer.
2. Single-op isolation with the same tensor shapes works fine when run standalone.
3. The key difference: the standalone test uses freshly allocated buffers, but in the full model, the buffers are reused from a buffer pool.

**Root Cause:** The all_gather kernel computes NOC addresses based on a sharding descriptor stored in L1. On the 3rd decode, a previous operation overwrites part of this descriptor (buffer aliasing bug in the memory allocator). The all_gather then computes wrong NOC target addresses.

**The Fix Clue:** Running with `TT_METAL_CLEAR_L1=1` made the hang deterministic on the 1st token (no stale data to mask the uninitialized descriptor). Running with `TT_METAL_WATCHER=1` caught the bad address immediately: `DebugSanitizeNocTargetInvalidXY`.

**Lesson:** When a standalone reproduction cannot reproduce the hang, the issue is likely in buffer lifecycle management or state accumulated from prior operations. Memory clearing and watcher sanitize are the key tools for these cases.

---

**Next:** [04_reading_watcher_and_triage_output.md](./04_reading_watcher_and_triage_output.md)
