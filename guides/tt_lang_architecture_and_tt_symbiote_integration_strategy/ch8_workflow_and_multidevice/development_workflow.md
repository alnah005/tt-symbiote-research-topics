# Development Workflow: 7 Steps from Kernel to Production

This file walks through the complete development workflow for creating a TT-Lang fused kernel and deploying it inside the TT-Symbiote inference pipeline. Each step produces a testable artifact so problems surface early.

## Step 1: Write the Kernel

Start from the fusion target identified in [Chapter 7](../ch7_fusion_targets/index.md). The kernel follows the three-thread pattern described in [Chapter 1](../ch1_programming_model/index.md): one `@ttl.compute()` thread and two `@ttl.datamovement()` threads (reader and writer), coordinated through DataFlow Buffers (DFBs).

A minimal kernel skeleton:

```python
import ttl
import ttnn

TILE_SIZE = 32

@ttl.operation(grid="auto")
def my_fused_op(x_in: ttnn.Tensor, w_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = x_in.shape[0] // TILE_SIZE
    col_tiles = x_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    x_dfb = ttl.make_dataflow_buffer_like(x_in, shape=(1, 1), block_count=2)
    w_dfb = ttl.make_dataflow_buffer_like(w_in, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for lr in range(rows_per_node):
            row = node_row * rows_per_node + lr
            if row < row_tiles:
                for lc in range(cols_per_node):
                    col = node_col * cols_per_node + lc
                    if col < col_tiles:
                        with (
                            x_dfb.wait() as x_blk,
                            w_dfb.wait() as w_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            out_blk.store(x_blk + w_blk)  # Replace with actual fused op

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for lr in range(rows_per_node):
            row = node_row * rows_per_node + lr
            if row < row_tiles:
                for lc in range(cols_per_node):
                    col = node_col * cols_per_node + lc
                    if col < col_tiles:
                        with x_dfb.reserve() as x_blk, w_dfb.reserve() as w_blk:
                            tx_x = ttl.copy(x_in[row:row+1, col:col+1], x_blk)
                            tx_w = ttl.copy(w_in[row:row+1, col:col+1], w_blk)
                            tx_x.wait()
                            tx_w.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for lr in range(rows_per_node):
            row = node_row * rows_per_node + lr
            if row < row_tiles:
                for lc in range(cols_per_node):
                    col = node_col * cols_per_node + lc
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[row:row+1, col:col+1])
                            tx.wait()
```

Key constraints enforced by the compiler (from `ttl_api.py` `_compile_ttnn_kernel`):
- Exactly 3 kernels: 1 compute + 2 data movement. Each core has 2 NOCs, so more than 2 DM kernels causes NOC conflicts.
- All tensor arguments must be `ttnn.Tensor` in `TILE_LAYOUT` with L1 or DRAM memory space.
- No mixed tensor types (all ttnn or all torch, not both).

The `grid="auto"` option (resolved in `_resolve_grid`) queries the device's `compute_with_storage_grid_size()` to use the full compute grid. Alternatively, pass an explicit tuple like `grid=(8, 8)`.

## Step 2: Validate with the Functional Simulator

Before touching hardware, verify correctness using the functional simulator described in [Chapter 3](../ch3_functional_simulator/index.md). The simulator runs the same Python kernel code on CPU, executing DFB operations as regular memory copies:

```python
def test_my_fused_op():
    dim = 256
    x_torch = torch.rand((dim, dim), dtype=torch.bfloat16)
    w_torch = torch.rand((dim, dim), dtype=torch.bfloat16)
    out_torch = torch.zeros_like(x_torch)

    # Functional sim: pass torch tensors directly
    my_fused_op(x_torch, w_torch, out_torch)

    expected = x_torch + w_torch  # Replace with reference implementation
    assert torch.allclose(out_torch, expected, rtol=1e-2, atol=1e-2)
```

When torch tensors are passed (rather than `ttnn.Tensor`), the TT-Lang runtime detects this and runs the kernel through the simulation path. This validates the tile iteration logic, DFB synchronization, and compute correctness without any device.

## Step 3: On-Device Execution

Once the simulator validates correctness, move to hardware execution. The transition requires only converting tensors to `ttnn.Tensor`:

```python
device = ttnn.open_device(device_id=0)

x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
w = ttnn.from_torch(w_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.from_torch(torch.zeros_like(x_torch), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

my_fused_op(x, w, out)

result = ttnn.to_torch(out)
assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
ttnn.close_device(device)
```

The `@ttl.operation` decorator detects `ttnn.Tensor` inputs and triggers the full compilation pipeline: Python AST to MLIR, MLIR optimization passes, EmitC to C++ kernels, then execution via `ttnn.generic_op`. The compiled kernel is cached (keyed on tensor shapes, dtypes, memory spaces, mesh shape, and `CompilerOptions`) so subsequent calls with the same tensor profile skip recompilation.

Use `TTLANG_COMPILE_ONLY=1` to run compilation without executing, useful for checking that the kernel compiles cleanly:

```bash
TTLANG_COMPILE_ONLY=1 python my_kernel.py
```

Use `TTLANG_INITIAL_MLIR=/tmp/kernel.mlir` to dump the pre-optimization MLIR for inspection, and `TTLANG_DEBUG_LOCATIONS=1` to include source-location annotations in the MLIR output.

## Step 4: Profile

TT-Lang provides four profiling modes, each serving a different stage of optimization. All require `TT_METAL_DEVICE_PROFILER=1` to be set.

### 4a. Auto-Profile

The broadest view. Reads device profiler CSV data and produces a per-line cycle-count report attributed back to your Python source:

```bash
TT_METAL_DEVICE_PROFILER=1 TTLANG_AUTO_PROFILE=1 python my_kernel.py
```

Internally, `_run_profiling_pipeline` in `ttl_api.py` calls `ttnn.ReadDeviceProfiler(device)`, reads the CSV from `$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv`, and uses the CB flow graph to attribute DMA time to specific `ttl.copy` calls. This is the recommended starting point for identifying bottlenecks.

### 4b. Signpost Profiling

For targeted measurement of specific code regions using `with ttl.signpost("name"):` blocks:

```bash
TT_METAL_DEVICE_PROFILER=1 TTLANG_SIGNPOST_PROFILE=1 python my_kernel.py
```

Signpost zones are identified by the `ttl_` prefix in the profiler CSV (see `_src/signpost_profile.py`). This mode gives you precise cycle counts for the exact regions you care about, filtering out framework overhead.

### 4c. Perf Dump

The most detailed mode. Produces NOC profiler summaries, CB flow graphs, and pipe graphs:

```bash
TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
  TT_METAL_PROFILER_MID_RUN_DUMP=1 TTLANG_PERF_DUMP=1 python my_kernel.py
```

`_run_perf_dump` in `ttl_api.py` reads NOC traces from `$TT_METAL_HOME/generated/profiler/.logs/`, the CB flow graph from `/tmp/ttlang_cb_flow_graph.json` (written by the `ttl-dump-cb-flow-graph` compiler pass), and the pipe graph from `/tmp/ttlang_pipe_graph.json`. Use this when you need to understand NOC utilization and data movement patterns.

### 4d. Perfetto Trace Server

Converts profiler data to Chrome Trace Event format and serves it over HTTP for visualization in the Perfetto UI:

```bash
TTLANG_PERF_SERV=1 TTLANG_SIGNPOST_PROFILE=1 python my_kernel.py
```

The `serve_trace` function in `_src/perf_trace_server.py` starts a local HTTP server with an HTML landing page that pushes the trace into Perfetto via `postMessage`. This provides a timeline view of all BRISC, NCRISC, and TRISC threads across cores, making it easy to spot pipeline bubbles and synchronization stalls.

You can also run the trace server standalone:

```bash
python -m ttl._src.perf_trace_server --path /path/to/profiler/.logs/
```

## Step 5: Optimize with CompilerOptions

Based on profiling results, tune compilation using `CompilerOptions` (defined in `compiler_options.py`). The seven boolean flags and their effects:

| Flag | Default | CLI Flag | Effect |
|------|---------|----------|--------|
| `maximize_dst` | `True` | `--ttl-maximize-dst` | Enables DST maximization via subblock compute and operation scheduling. Increases register utilization to reduce spills. |
| `enable_fpu_binary_ops` | `True` | `--ttl-fpu-binary-ops` | Routes add/sub/mul through FPU instead of SFPU. Faster for these ops but uses FPU pipeline. |
| `use_block_matmul` | `True` | `--ttl-block-matmul` | Lowers matmul to block-level hardware intrinsics. Disable only for debugging. |
| `auto_sync` | `False` | `--ttl-auto-sync` | Lets the compiler insert and move DFB synchronization ops. Experimental --- can improve pipeline overlap but may change semantics. |
| `combine_pack_tiles` | `True` | `--ttl-combine-pack-tiles` | Combines consecutive `pack_tile` ops into `pack_tile_block`. Reduces instruction count. |
| `reduce_full_fp32` | `True` | `--ttl-reduce-full-fp32` | Enables FP32 accumulation for reduce operations. Improves numerical accuracy at slight cost. |
| `matmul_full_fp32` | `True` | `--ttl-matmul-full-fp32` | Enables FP32 accumulation for matmul operations. Critical for model accuracy. |

Options can be set at three levels with a clear priority order (highest wins):

```
sys.argv  >  TTLANG_COMPILER_OPTIONS env var  >  decorator options= string
```

Example --- disable DST maximization for debugging:

```bash
# Via CLI flag
python my_kernel.py --no-ttl-maximize-dst

# Via environment variable
TTLANG_COMPILER_OPTIONS="--no-ttl-maximize-dst" python my_kernel.py
```

The `merge()` method on `CompilerOptions` handles the layered override logic: only fields that were explicitly set in a higher-priority source override lower-priority defaults. Unmentioned flags fall through.

Use `--ttl-help` to print all available options:

```bash
python my_kernel.py --ttl-help
```

## Step 6: Integrate into TTNNModule

Once the kernel is profiled and optimized, wrap it in a `TTNNModule` subclass following the contract from [Chapter 6](../ch6_integration_strategy/index.md). The key change: replace `ttnn.*` op calls in `forward()` with the TT-Lang kernel call.

```python
from models.experimental.tt_symbiote.core.module import TTNNModule

class TTNNFusedOp(TTNNModule):
    @classmethod
    def from_torch(cls, torch_module):
        instance = cls()
        instance._fallback_torch_layer = torch_module
        return instance

    def preprocess_weights_impl(self):
        # Convert weights to ttnn format --- same as any TTNNModule
        self.tt_weight = ttnn.from_torch(
            self.torch_layer.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def move_weights_to_device_impl(self):
        # Move to device --- same as any TTNNModule
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Allocate output tensor
        out = ttnn.from_torch(
            torch.zeros(x.shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        # Call TT-Lang kernel instead of ttnn ops
        my_fused_op(x, self.tt_weight, out)
        return out
```

Important details:
- **Output tensor allocation is the caller's responsibility.** Unlike `ttnn.linear` which allocates internally, `CompiledTTNNKernel.__call__` expects a pre-allocated output tensor as the last argument (see [Chapter 6, interface contract](../ch6_integration_strategy/interface_contract.md)).
- **JIT compilation happens on the first `forward()` call.** The `_make_cache_key` function (in `ttl_api.py`) creates a cache key from tensor shapes, dtypes, memory spaces, mesh shape, compute config flags, and `CompilerOptions`. Subsequent calls with the same profile are free.
- **The weight lifecycle is unchanged.** `preprocess_weights_impl` and `move_weights_to_device_impl` run before `forward()`, so all weights are on-device in `TILE_LAYOUT` when the kernel executes.
- **`@trace_enabled` and `@run_on_devices` decorators** from `run_config.py` and `module.py` are fully compatible with TT-Lang kernel calls inside `forward()`.

## Step 7: Test with the TT-Symbiote Pipeline

The final validation runs the integrated module through TT-Symbiote's full inference pipeline to verify end-to-end correctness:

1. **Unit test the module in isolation.** Compare the `TTNNModule.forward()` output against the original PyTorch module's output using `compare_fn_outputs` from `core/utils.py`.

2. **Integration test with the model.** Swap the module into the full model (e.g., replace `TTNNRMSNorm` with a fused variant) and run inference on a known input. Compare against the unfused model's output to confirm numerical equivalence.

3. **Performance regression test.** Profile the full model inference with and without the fused kernel. The fused kernel should reduce DRAM traffic (fewer intermediate materializations) and wall-clock time. Use TT-Symbiote's `DispatchManager.timings` to get per-module timing breakdowns.

4. **Trace compatibility test.** If the module uses `@trace_enabled`, verify that the TT-Lang kernel works correctly under `ttnn.begin_trace_capture` / `ttnn.end_trace_capture`. The compiled kernel dispatches through `ttnn.generic_op`, which is trace-compatible.

## Quick-Reference: Environment Variables

| Variable | Values | Purpose |
|----------|--------|---------|
| `TTLANG_COMPILE_ONLY` | `0`/`1` | Compile without executing |
| `TTLANG_INITIAL_MLIR` | file path | Dump pre-optimization MLIR |
| `TTLANG_DEBUG_LOCATIONS` | `0`/`1` | Source locations in MLIR output |
| `TTLANG_AUTO_PROFILE` | `0`/`1` | Per-line cycle-count profiling |
| `TTLANG_SIGNPOST_PROFILE` | `0`/`1` | User-defined signpost zone profiling |
| `TTLANG_PERF_DUMP` | `0`/`1` | NOC traces, CB flow, pipe graph |
| `TTLANG_PERF_SERV` | `0`/`1` | Perfetto trace server |
| `TTLANG_COMPILER_OPTIONS` | option string | Compiler flags via env var |
| `TT_METAL_DEVICE_PROFILER` | `0`/`1` | Enable device profiler (required for all profiling) |
| `TT_METAL_DEVICE_PROFILER_NOC_EVENTS` | `0`/`1` | Enable NOC event tracing (required for perf dump) |
| `TT_METAL_PROFILER_MID_RUN_DUMP` | `0`/`1` | Flush profiler data mid-run |
| `TTLANG_PROFILE_CSV` | file path | Override default profiler CSV location |

---

**Next:** [`multidevice_simplification.md`](./multidevice_simplification.md)
