# Tracy Profiling Setup

## Context

This file addresses **Q8**: What is the op-level latency breakdown for `TTNNMoE.forward` on T3K, and how do the key pipeline stages (all-gather, expert dispatch, sparse matmuls, reduce-scatter) relate on the critical path?

Source ranges: `moe.py:L1412–L1496` (`TTNNMoE.forward`), `moe.py:L1159–L1343` (`TTNNExperts.forward`).

---

## What Tracy Captures

Tracy is a C++ frame profiler that instruments annotated zones. In the TTNN context, it captures two interleaved timelines on a single display:

- **Host timeline:** Python dispatch calls and TTNN C++ API entry/exit points. Reflects host-side latency but not device execution time.
- **Device callback timeline:** TTNN injects Tracy zone markers from the device completion callbacks that fire after each kernel finishes on the device. These timestamps represent actual device-side kernel execution.

The critical path for `TTNNMoE.forward` at batch=1 decode is:

```
all_gather_async ──► gate linear ──► route_tokens_to_experts ──► TTNNExperts.forward ──► reduce_scatter_minimal_async
                                                                                    └──► shared_experts (potentially overlapped)
```

Tracy makes the overlap (or lack thereof) visible. If `shared_experts` and `reduce_scatter_minimal_async` run serially, there is a pipeline bubble to eliminate.

---

## Step 1: Enable Tracy in the TT-Metal Build

Tracy support is compile-time gated. Build with:

```bash
# From the tt-metal repository root
ENABLE_TRACY=1 cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ttnn -j$(nproc)
```

Verify the build enabled Tracy by checking for the shared object:

```bash
python -c "import ttnn; print(ttnn.__file__)"
# Confirm the .so was built after the ENABLE_TRACY=1 build
strings $(python -c "import ttnn; print(ttnn.__file__)") | grep -c "TracyZone"
# Should print a non-zero count
```

---

## Step 2: Environment Variables

Set the following before launching the Python process:

```bash
# Required: enable Tracy data collection
export TT_METAL_ENABLE_TRACY=1

# Required: open the TCP port Tracy Profiler connects to (default 8086)
export TRACY_PORT=8086

# Optional but recommended: increase the Tracy ring buffer to avoid dropped frames
# Default is 64 MB; at decode batch=1, 256 MB is more than sufficient
export TRACY_MEMORY=268435456

# Optional: enable TTNN op-level zone annotations in addition to the broader
# TT-Metal kernel zones. This is what produces per-op boxes in the timeline.
export TT_METAL_TRACY_ANNOTATIONS=1

# Suppress device profiler interference (it writes CSV; redundant when using Tracy)
export TT_METAL_DEVICE_PROFILER=0
```

---

## Step 3: Launch the Tracy GUI

Download the Tracy profiler binary matching your build's Tracy version from https://github.com/wolfpld/tracy/releases. The version must match the Tracy headers compiled into TT-Metal.

```bash
# Find the Tracy version compiled into TT-Metal
grep -r "TRACY_VERSION" /path/to/tt-metal/third_party/tracy/public/tracy/Tracy.hpp | head -1

# Launch the GUI (Linux)
./Tracy-linux-x86_64 &
```

In the Tracy GUI, click **Connect** and enter the host IP and port 8086. Leave the GUI open before running the Python script.

---

## Step 4: Run a Minimal Forward Pass Harness

Create an isolation harness that calls `TTNNMoE.forward` in a loop, warmups discarded, with a Tracy zone annotation around the full forward call.

> **Note:** Because `TTNNMoE.forward` is not externally decomposable at the Python level, the harness annotates only the full forward call; for per-stage zones, apply annotations directly inside `moe.py` as in Step 5.

```python
# profile_moe_forward.py
import os
import ttnn
import torch

# Tracy Python bindings (available when TT_METAL_ENABLE_TRACY=1)
try:
    import tracy_python as tracy
    TRACY_AVAILABLE = True
except ImportError:
    TRACY_AVAILABLE = False

def tracy_zone(name: str):
    """Context manager that emits a Tracy zone annotation from Python."""
    import contextlib
    @contextlib.contextmanager
    def _zone():
        if TRACY_AVAILABLE:
            tracy.begin_zone(name)
        try:
            yield
        finally:
            if TRACY_AVAILABLE:
                tracy.end_zone()
    return _zone()


def run_profiled_forward(model, x, mesh_device, n_warmup=3, n_profile=10):
    """
    Runs TTNNMoE.forward with Tracy zone annotations around each key region.
    n_warmup passes are discarded; n_profile passes are captured.
    """
    # Warmup: prime caches and JIT compilation
    for _ in range(n_warmup):
        _ = model(x)
        ttnn.synchronize_device(mesh_device)

    for i in range(n_profile):
        with tracy_zone(f"TTNNMoE.forward[{i}]"):

            # Full forward call — Tracy device callbacks fill in per-op timing
            with tracy_zone("forward_dispatch"):
                out = model(x)

            # Synchronize to ensure device kernels complete before zone ends
            ttnn.synchronize_device(mesh_device)

    return out
```

---

## Step 5: Annotating Key Regions Inside `TTNNMoE.forward`

Insert Tracy zones directly at the call sites in `moe.py`. The four key regions and their line numbers are:

```python
# moe.py:L1429–L1436 — all_gather_async
with tracy_zone("all_gather_async"):
    x, all_gather_future = ttnn.experimental.all_gather_async(
        x,
        dim=1,
        num_links=1,
        cluster_axis=0,
        mesh_device=self.mesh_device,
        topology=ttnn.Topology.Linear,
    )

# moe.py:L1445–L1455 — gate routing linear (HiFi4)
with tracy_zone("gate_linear_hifi4"):
    router_logits = ttnn.linear(
        x,
        self.gate_weight,
        compute_kernel_config=self._gate_compute_cfg,
        ...
    )

# moe.py:L1466 — route_tokens_to_experts
with tracy_zone("route_tokens_to_experts"):
    routing_output = self.router.route_tokens_to_experts(router_logits)

# moe.py:L1471 — TTNNExperts.forward (includes all expert stages)
with tracy_zone("experts_forward"):
    expert_out = self.experts(x, routing_output, ...)

# moe.py:L1477 — pre-normalization scale before reduce_scatter
with tracy_zone("pre_norm"):
    expert_out = ttnn.mul(expert_out, 1.0 / float(n_rs))

# moe.py:L1478–L1490 — reduce_scatter_minimal_async
with tracy_zone("reduce_scatter"):
    x, reduce_scatter_future = ttnn.experimental.reduce_scatter_minimal_async(
        expert_out,
        scatter_dim=1,
        cluster_axis=0,
        mesh_device=self.mesh_device,
        topology=ttnn.Topology.Ring,
    )

# moe.py:L1493–L1494 — shared experts + add (potentially overlapped with reduce_scatter)
with tracy_zone("shared_experts_and_add"):
    shared_out = self.shared_experts(residual)
    out = ttnn.add(x, shared_out)
```

And inside `TTNNExperts.forward` (`moe.py:L1159–L1343`), annotate the seven pipeline stages:

```python
# moe.py:L1191–L1212 — token padding
with tracy_zone("experts/token_padding"):
    ...

# moe.py:L1225–L1230 — all_to_all_dispatch
with tracy_zone("experts/all_to_all_dispatch"):
    ...

# moe.py:L1238–L1245 — moe_expert_token_remap
with tracy_zone("experts/token_remap"):
    ...

# moe.py:L1250–L1259 — w1 sparse_matmul (gate projection)
with tracy_zone("experts/w1_sparse_matmul"):
    ...

# moe.py:L1260–L1269 — w3 sparse_matmul (up projection)
with tracy_zone("experts/w3_sparse_matmul"):
    ...

# moe.py:L1271–L1275 — silu + elementwise mul
with tracy_zone("experts/silu_mul"):
    ...

# moe.py:L1280–L1289 — w2 sparse_matmul (down projection)
with tracy_zone("experts/w2_sparse_matmul"):
    ...

# moe.py:L1307–L1312 — all_to_all_combine
with tracy_zone("experts/all_to_all_combine"):
    ...

# moe.py:L1321–L1335 — weight application
with tracy_zone("experts/weight_application"):
    ...
```

---

## Step 6: Reading the Timeline

After a profiled run completes, Tracy displays a timeline. Key patterns to look for:

### Identifying the Critical Path

The critical path is the longest chain of non-overlapping zones from the start to the end of `TTNNMoE.forward[i]`. At batch=1 decode on T3K, the expected critical path is:

```
all_gather_async
    └── gate_linear_hifi4
            └── route_tokens_to_experts
                    └── experts_forward
                            ├── experts/all_to_all_dispatch
                            ├── experts/w1_sparse_matmul
                            ├── experts/w3_sparse_matmul
                            ├── experts/silu_mul
                            ├── experts/w2_sparse_matmul
                            └── experts/all_to_all_combine
```

`shared_experts_and_add` should run concurrently with `reduce_scatter` if the implementation overlaps them correctly (moe.py:L1493 is called before the scatter future is awaited).

### Signs of a Pipeline Stall

- `shared_experts_and_add` starts only after `reduce_scatter` completes → no overlap; the `ttnn.experimental.reduce_scatter_minimal_async` future is being awaited too early.
- A gap between `all_gather_async` zone end and `gate_linear_hifi4` zone start → the `all_gather_future` is being awaited before the linear dispatch; check whether the await can be deferred.
- `experts/token_padding` consumes a visible fraction of the expert stage time → padding is implemented in Python with slow host-side tensor construction; see Chapter 3 for diagnosis.

### Extracting Numeric Values

Tracy's **Statistics** panel (Ctrl+S) lists each zone by name with min, mean, and max durations in microseconds. Export to CSV with **File → Export → CSV** for use in spreadsheet comparison across configuration sweeps.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Tracy GUI shows "Waiting for connection" indefinitely | Process started without `TT_METAL_ENABLE_TRACY=1`, or port blocked | Verify env var is set in the same shell; check firewall for port 8086 |
| Timeline shows only one wide zone with no inner detail | `TT_METAL_TRACY_ANNOTATIONS=1` not set | Set the env var and rerun |
| Zone durations are implausibly small (< 1 µs) | Warmup passes not excluded; JIT compilation not yet done | Increase `n_warmup` to 5–10 |
| Tracy version mismatch error at connection | GUI binary version differs from compiled-in headers | Rebuild TT-Metal or download matching Tracy binary |
| Device callback zones appear as a flat bar with no nesting | Tracy ring buffer overflow due to small buffer | Increase `TRACY_MEMORY` to 512 MB or reduce `n_profile` |

---

Next: [`ttnn_op_timer_profiling.md`](./ttnn_op_timer_profiling.md)
