# Tracy Profiling

Tracy is a frame profiler designed for low-overhead real-time tracing of both CPU and GPU workloads. In the context of T3K, the tt-metal runtime integrates Tracy to expose device-kernel execution timestamps, host-side dispatch spans, PCIe DMA windows, and Ethernet CCL bursts in a single unified timeline view. Where TTNN op timers (see [`ttnn_op_timers.md`](./ttnn_op_timers.md)) give a fast ranked list, Tracy gives a precise, correlated view of what every chip is actually executing at every microsecond of a decode step.

## 1. Tracy Setup for T3K

### 1.1 Build Flags

Tracy instrumentation is compiled into `tt-metal` using a CMake option. It is not enabled by default because the profiling hooks add a small but non-zero overhead to every kernel launch.

```bash
# From the tt-metal repository root:
cmake -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENABLE_TRACY=ON \
  -DENABLE_PROFILER=ON \
  -DENABLE_TRACY_ETH=ON \  # Required for Ethernet/CCL span capture
  -DTT_METAL_VERSIM_DISABLED=ON
cmake --build build --target tt_metal -j$(nproc)
```

> **Note:** Without `-DENABLE_TRACY_ETH=ON`, Ethernet spans will be silently absent from the capture. The build will succeed and Tracy will record CPU and device-kernel spans normally, but CCL Ethernet bursts (the `all_reduce` Ethernet phase visible in Section 4.4) will not appear in the timeline. If you are investigating CCL latency, this flag is required.

`RelWithDebInfo` is required so that Tracy's flame-graph view can resolve symbol names in stack traces. A pure `Release` build will work for timeline inspection but loses function-name annotations in the call-stack pane.

After building, verify that the Tracy hooks are present by checking the shared library:

```bash
nm -D build/lib/libtt_metal.so | grep Tracy | head -5
# Expected: several lines of ___tracy_emit_zone_* symbols
```

### 1.2 Tracy Server Startup

Tracy uses a client-server architecture. The profiled process (the T3K host running your Python script) is the client; the Tracy profiler GUI is the server. They communicate over a TCP socket. On a T3K host, the server and client typically run on the same machine:

```bash
# In a separate terminal on the T3K host:
./tracy/profiler/build/tracy-profiler &
# The GUI opens and listens on port 8086 by default.
```

If the Tracy GUI is not available on the T3K host (headless server), use the Tracy command-line capture tool instead:

```bash
# Headless capture to a .tracy file, then inspect on a workstation:
./tracy/capture/build/tracy-capture -o /tmp/decode_trace.tracy -s 5
# -s 5: capture for 5 seconds; adjust to cover your decode window.
```

Transfer the `.tracy` file to a workstation with a display and open it with the Tracy GUI there.

### 1.3 Connecting to a Multi-Device Host

On T3K, all 8 Wormhole chips are driven by a single host process. Tracy's client runs once per process, so all 8 chips appear within the same Tracy session. The timeline view will have:

- One **CPU thread lane** for each Python/C++ thread. tt-metal allocates one dispatch thread per device; on an 8-chip T3K MeshDevice this means up to 8 dispatch threads. Combined with the main host thread, expect up to 9 CPU lanes visible in Tracy.
- One **GPU lane** per chip (labeled `TT Device 0` through `TT Device 7`), showing device-kernel execution spans as horizontal colored bars.

To identify which device corresponds to which chip in the mesh, consult the device index printed at mesh initialization:

```python
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
# tt-metal logs: "Device 0 (row=0, col=0) ... Device 7 (row=0, col=7)"
```

Tracy device indices match the `ttnn.open_mesh_device` assignment order.

## 2. Annotating `TTNNBailingMoEAttention` Forward with Tracy Zones

Tracy zones are lightweight C++-level markers that appear as labeled spans in the timeline. For Python-side annotation, tt-metal exposes a thin Python binding:

```python
from tt_metal import tracy_scope  # Python wrapper around Tracy C API
```

Insert zone markers around the logical segments of the attention forward:

```python
def forward(self, hidden_states, position_ids, kv_cache):
    with tracy_scope("BailingAttn/QKV_projection"):
        qkv = self.qkv_proj(hidden_states)   # TTNNLinearIColShardedWAllReduced + all_reduce

    with tracy_scope("BailingAttn/to_replicated"):
        qkv_rep = _to_replicated(qkv, self.mesh)  # host round-trip (Ch3)

    with tracy_scope("BailingAttn/split_heads"):
        q, k, v = self._split_qkv(qkv_rep)

    with tracy_scope("BailingAttn/rope"):
        q, k = self.rope(q, k, position_ids)   # TTNNRotaryPositionEmbedding (Ch6)

    with tracy_scope("BailingAttn/qk_norm"):
        if self.use_qk_norm:
            q = self.q_norm(q)   # reshape + RMSNorm + reshape (Ch6)
            k = self.k_norm(k)

    with tracy_scope("BailingAttn/memory_configs"):
        q = ttnn.to_memory_config(q, self.sdpa_q_mem_cfg)    # Ch4 transitions
        k = ttnn.to_memory_config(k, self.sdpa_k_mem_cfg)
        v = ttnn.to_memory_config(v, self.sdpa_v_mem_cfg)

    with tracy_scope("BailingAttn/paged_sdpa"):
        attn_out = ttnn.paged_sdpa_decode(q, k, v, kv_cache, ...)  # Ch5

    with tracy_scope("BailingAttn/output_proj"):
        return self.o_proj(attn_out)
```

These zone names appear verbatim in the Tracy timeline as colored spans on the CPU thread lane. The device-side kernel bars run concurrently in the GPU lanes and can be correlated by timestamp to identify which host-side annotation triggered which device operation.

If you prefer not to modify the model source, tt-metal's built-in `OP_PROFILER` Tracy integration automatically emits one zone per TTNN op with the op name as the zone label, without any source annotation. This is sufficient for a first-pass timeline inspection.

## 3. Capturing a Single Decode Step Trace

### 3.1 Recommended Capture Window

A single Ling decode step takes approximately 600–1200 µs [ESTIMATE] of wall-clock time on T3K at batch=1. The Tracy capture should cover at minimum:

- 5 warm-up steps
- A brief idle gap (for a visual anchor in the timeline)
- 1–3 profiled decode steps

This is achievable with a 50–100 ms capture window. The Tracy capture command:

```bash
./tracy/capture/build/tracy-capture \
  -o /tmp/bailing_decode.tracy \
  -s 0.1   # capture for 100 ms
```

Start the capture, then immediately trigger your profiled decode step:

```python
import time, subprocess, threading

def run_profiled_step():
    time.sleep(0.01)   # brief delay to ensure Tracy has connected
    for _ in range(5):
        model(hidden_states, position_ids, kv_cache)   # warm-up
    ttnn.synchronize_devices(mesh)
    # Visual anchor: brief Python sleep creates a visible gap
    time.sleep(0.005)
    # Profiled step:
    model(hidden_states, position_ids, kv_cache)
    ttnn.synchronize_devices(mesh)

capture = subprocess.Popen([
    "./tracy/capture/build/tracy-capture",
    "-o", "/tmp/bailing_decode.tracy",
    "-s", "0.5"
])
t = threading.Thread(target=run_profiled_step)
t.start()
t.join()
capture.wait()
```

### 3.2 Isolating the Attention Layer from the Rest of the Decoder

In a full model run, the attention layer is surrounded by MLP layers, residual adds, layer norms, and MoE routing. To isolate the attention layer in the Tracy timeline:

1. Add a named Tracy zone around the full decoder loop iteration and a nested zone for just the attention call (as shown in Section 2).
2. In the Tracy GUI, right-click the outer zone and select **Find zones → this zone name** to jump to all occurrences.
3. Use the **Frame** view to zoom to the attention span within a single decode step.

Alternatively, run the attention module in isolation (without the surrounding MLP and MoE blocks) by constructing a test harness that calls only `TTNNBailingMoEAttention.forward` with synthetic inputs:

```python
# Minimal test harness for isolated attention profiling:
hidden_states = ttnn.from_torch(
    torch.randn(1, 1, 4096, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
for _ in range(5):
    out = attention_module(hidden_states, position_ids, kv_cache)
ttnn.synchronize_devices(mesh)
# Now start Tracy capture and run one more step.
```

This is the cleanest approach for bottleneck identification because it eliminates all MoE routing and MLP overhead from the timeline.

## 4. Reading the Tracy Timeline

### 4.1 Overview of the Timeline Layout

After loading a `.tracy` file, the main window shows:

- **Top section:** CPU thread lanes. Each horizontal lane is one OS thread. The main Python thread shows your `tracy_scope` zones as colored rectangles. The tt-metal device dispatch threads show TTNN op dispatch spans.
- **Middle section:** GPU device lanes, one per Wormhole chip (`TT Device 0`–`TT Device 7`). Each bar represents one kernel executing on that chip.
- **Bottom section:** Frame timing and statistics.

Zoom in with scroll-wheel. Pan with middle-mouse drag. Click any bar to see its name, duration, and call stack.

### 4.2 Correlating Device-Side Durations with Host-Side Zones

To determine which host-side zone caused a given device-side kernel:

1. Click the device-side bar to select it. Tracy highlights the host-side zone that issued the dispatch (connected by a dotted vertical line).
2. Note the **queue latency**: the gap between the right edge of the host dispatch span and the left edge of the device kernel bar. This is the time the op spent in the device command queue before executing. For T3K decode steps at batch=1, queue latency is typically 1–5 µs [ESTIMATE] per op.
3. For CCL all-reduce spans, the device bars on all 8 chips should start at nearly the same timestamp (within 2–3 µs [ESTIMATE]). A large skew between chip start times indicates Ethernet contention or an unbalanced mesh schedule.

### 4.3 Identifying Host-Side Dead Zones

The host round-trip (`_to_replicated`, see Chapter 3) appears in the Tracy timeline as a gap in all device GPU lanes — all 8 chips are idle — while the CPU main thread shows two spans:

1. A `ConcatMeshToTensor` span (PCIe reads from all 8 chips to host DRAM).
2. A `ttnn::from_torch` + `ReplicateTensorToMesh` span (PCIe writes from host DRAM to all 8 chips).

The combined host gap should be 6.8–25.2 µs [ESTIMATE] for the ROW_MAJOR QKV tensor at batch=1. If the measured gap exceeds this range, suspect PCIe bandwidth contention from another process on the host.

### 4.4 CCL and Data-Movement Zone Correlation

The all-reduce CCL operation (`ttnn::all_reduce`) following the QKV matmul appears as an Ethernet burst visible across all 8 device lanes simultaneously. In the Tracy timeline, look for:

- A set of 8 roughly synchronized device spans labeled `ccl::all_reduce` (or the specific CCL kernel name from the firmware build).
- A corresponding host span on the dispatch thread that covers the dispatch of all CCL sub-operations.
- Ethernet spans, if the Tracy build includes network interface counters (available with `ENABLE_TRACY_ETH=ON`).

The duration of the CCL span on the device lane is the actual Ethernet transfer + reduce time. The duration of the host dispatch span is the time to enqueue all CCL ops. For `num_links=1`, the device-side CCL span is expected to be longer than the dispatch span by a factor of 3–8× [ESTIMATE] at Ling's hidden size of 4096.

## 5. Common Pitfalls

### 5.1 Warm-Up Steps and JIT Compilation Artifacts

Verify that JIT compilation is complete before treating any Tracy capture as valid: check the tt-metal log for lines containing `Compiling program` — there should be none during the profiled step. In the Tracy timeline, discard any capture where the first N decode steps show spans significantly longer than steady-state; these represent compilation overhead, not runtime cost. If you need to profile the compilation cost itself (e.g., to measure model load time), run a separate capture with 0 warm-up steps and filter the timeline by the Tracy zone label `ttnn::CompileProgram`.


### 5.2 PCIe Transfer Inflation in First-Run Captures

The `_to_replicated` host round-trip transfers tensors over PCIe. On the very first execution, the IOMMU may need to set up DMA mappings, inflating the transfer time by 10–50× compared to steady-state [ESTIMATE]. This is distinct from JIT compilation — it affects the `ConcatMeshToTensor` and `ttnn::from_torch` ops specifically, and it resolves after the first call, not after 5 calls.

**Mitigation:** Run at minimum 2 warm decode steps before profiling. The IOMMU mapping is established on the first PCIe transfer and reused for all subsequent transfers to the same buffer.

### 5.3 Multi-Tenant Host Interference

T3K hosts running production workloads may have other processes performing PCIe DMA concurrently. This appears in the Tracy timeline as unexpected elongation of `ConcatMeshToTensor` spans without a corresponding increase in device-side kernel time. To check, use `iotop` or `perf stat -e dTLB-load-misses` on the host during the capture to detect PCIe contention.

### 5.4 Tracy Overhead on Critical Path

Tracy zones add approximately 30–80 ns per zone entry/exit on the host thread [ESTIMATE]. For decode steps where you have annotated every logical segment (8–12 zones as shown in Section 2), the total Tracy overhead is approximately 0.5–2 µs, which is negligible relative to the ~600–1200 µs decode step. However, if you add Tracy zones inside tight per-head loops (e.g., one zone per head), the overhead accumulates and distorts the timeline. Keep zones at the logical-block granularity shown in Section 2.

---

**Next:** [`bottleneck_decision_tree.md`](./bottleneck_decision_tree.md)
