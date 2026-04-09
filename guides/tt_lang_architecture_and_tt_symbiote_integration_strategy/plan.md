# TT-Lang Architecture and TT-Symbiote Integration Strategy — Research Guide Plan

## Audience

This guide targets Tenstorrent software engineers who:

- Are already familiar with TTNN's op library and the `ttnn.Tensor` type.
- Have working knowledge of TT-Symbiote's `TTNNModule` base class and the `TorchTTNNTensor` dispatch mechanism.
- Understand Tenstorrent hardware concepts: Tensix cores, NOCs, L1/DRAM memory hierarchy, TILE_LAYOUT.
- Want to learn TT-Lang's programming model, compilation pipeline, and simulator so they can author custom fused kernels and integrate them into TT-Symbiote modules.

No prior MLIR knowledge is assumed; the guide introduces relevant MLIR concepts as needed.

---

## Chapter List

### Chapter 1 — TT-Lang Programming Model

**Directory:** `ch1_programming_model`

**Description:** Introduces TT-Lang's core abstractions — the decorator-based DSL, Dataflow Buffers, TensorBlocks, and the multi-node grid execution model — through annotated code examples.

**Files:**

- `index.md`
  - Overview of TT-Lang's position in the Tenstorrent software stack (between TTNN ops and TT-Metalium).
  - How `import ttl` exposes the unified namespace (`ttl/__init__.py` re-exports from `ttl/ttl.py`).

- `decorators_and_threads.md`
  - The three decorator tiers: `@ttl.operation(grid=...)`, `@ttl.compute()`, `@ttl.datamovement()`.
  - How `@ttl.operation` (in `ttl/ttl_api.py::pykernel_gen`) orchestrates kernel compilation: calls the user function, collects registered threads via `_thread_registry`, constructs a `Program`, and triggers the MLIR pipeline.
  - Thread registration mechanism: `_register_thread()` / `_get_registered_threads()` and the requirement for exactly 3 threads (1 compute + 2 DM) matching Tensix's TRISC + BRISC + NCRISC.
  - The simulator equivalents in `sim/decorators.py`: `ComputeTemplate` / `DMTemplate` with `BindableTemplate.bind()` protocol and `rebind_func_with_ctx()`.

- `dataflow_buffers.md`
  - `ttl.make_dataflow_buffer_like(tensor, shape, block_count)` — creating circular buffers tied to tensor shapes (`ttl/circular_buffer.py::CircularBuffer`).
  - DFB as a ring buffer: `DFBState` fields (`cap`, `head`, `visible`, `reserved`, `buf`) in `sim/dfbstate.py`.
  - Block acquisition: `dfb.wait()` (consumer) vs. `dfb.reserve()` (producer) context managers.
  - The `Block` class (`sim/dfb.py`) and its `BlockStateMachine` enforcing access patterns: `AccessState` enum (MW, MR, RW, ROR, NAW, OS), `BlockAcquisition` enum, and the full state transition table in `sim/blockstate.py`.
  - `block.store()` and `block.push()` / `block.pop()` operations.

- `tensor_blocks_and_grid.md`
  - `TensorBlock` (`ttl/operators.py`) as the compile-time proxy for tile data; operator overloading (`__add__`, `__mul__`, etc.) generating TTL MLIR ops.
  - `CopyTransferHandler` and `ttl.copy(src, dst)` for asynchronous DMA between tensors and DFBs.
  - Grid intrinsics: `ttl.node(dims=2)` returns `(col, row)` core coordinates; `ttl.grid_size(dims=2)` returns grid dimensions.
  - Walkthrough of `examples/eltwise_add.py`: grid-parallel elementwise add showing DFB double-buffering with `block_count=2`.

---

### Chapter 2 — TT-Lang Compilation Pipeline

**Directory:** `ch2_compilation_pipeline`

**Description:** Traces the full lowering path from Python DSL source through MLIR dialects to C++ codegen and JIT execution, detailing the optimization passes at each stage.

**Files:**

- `index.md`
  - End-to-end pipeline overview: Python AST -> TTL MLIR -> Compute dialect -> TTKernel dialect -> EmitC -> C++ source -> JIT compilation via `ttnn.generic_op`.
  - Entry point: `pykernel_gen` in `ttl/ttl_api.py` -> `_compile_kernel()` -> `_compile_ttnn_kernel()`.

- `python_to_mlir.md`
  - `TTLGenericCompiler` (`ttl/_src/ttl_ast.py`): Python AST visitor that builds TTL MLIR from decorated thread functions.
  - Capture collection: `_collect_captures()` extracting closure variables (ints, floats, `CircularBuffer`, `ttnn.Tensor`).
  - Tensor type construction: `_build_tensor_type()` creating `RankedTensorType` with `TTLLayoutAttr` encoding for grid, tile shape, and memory space (L1/DRAM).
  - Source location tracking: `_make_file_loc()` for MLIR debug locations, `_track_tensor_sources()` for error diagnostics.

- `mlir_passes.md`
  - The full pass pipeline assembled in `_compile_kernel()` (lines 1175-1233 of `ttl_api.py`):
    1. `convert-ttl-to-compute` — Lower TTL high-level ops to compute primitives (`lib/Dialect/TTL/Transforms/ConvertTTLToCompute.cpp`).
    2. `ttl-set-compute-kernel-config` — Configure FP32 dest accumulation, reduce/matmul precision (`TTLSetComputeKernelConfig.cpp`).
    3. `ttl-assign-dst` — Assign DST register slots; FPU binary ops flag (`TTLAssignDST.cpp`).
    4. `ttl-subblock-compute-for-dst` — (when `maximize_dst=True`) Subblock tiling for DST reuse (`TTLSubblockComputeForDST.cpp`).
    5. `ttl-lower-matmul-block` — (when `use_block_matmul=True`) Lower matmul to block-level hardware calls (`TTLLowerMatmulBlock.cpp`).
    6. `ttl-lower-to-loops` — Convert tile-level ops to SCF loops (`ConvertTTLComputeToSCF.cpp`).
    7. `ttl-schedule-operations` — Reorder ops for DST accumulation scheduling (`TTLScheduleOperations.cpp`).
    8. `ttl-annotate-cb-associations` — Tag ops with circular buffer IDs (`TTLAnnotateCBAssociations.cpp`).
    9. `ttl-dump-cb-flow-graph` — (perf modes) Dump CB flow graph to JSON (`TTLDumpCBFlowGraph.cpp`).
    10. `ttl-lower-dprint-to-emitc` / `ttl-lower-signpost-to-emitc` — Lower debug/profiler intrinsics.
    11. `convert-ttl-to-ttkernel` — Lower to TTKernel dialect (`ConvertTTLToTTKernel.cpp`).
    12. `ttkernel-insert-inits` — Insert hardware initialization calls (`TTKernelInsertInits.cpp`).
    13. `ttkernel-insert-l1-accumulation` — Insert L1 accumulation logic (`TTKernelInsertL1Accumulation.cpp`).
    14. `ttkernel-combine-pack-tiles` — Fuse consecutive pack_tile ops (`TTKernelCombinePackTiles.cpp`).
    15. Standard MLIR: `canonicalize`, `cse`, `lower-affine`.
    16. `convert-ttkernel-to-emitc` — Final EmitC lowering.
    17. `symbol-dce` — Dead code elimination.
  - `CompilerOptions` dataclass (`ttl/compiler_options.py`): `maximize_dst`, `enable_fpu_binary_ops`, `use_block_matmul`, `auto_sync`, `combine_pack_tiles`, `reduce_full_fp32`, `matmul_full_fp32`; priority: `sys.argv` > `TTLANG_COMPILER_OPTIONS` env > decorator `options=`.

- `codegen_and_execution.md`
  - C++ source generation: `ttkernel_to_cpp_by_name()` extracting EmitC output per kernel.
  - `_write_kernel_to_tmp()` writing kernels to `/tmp/{user}/ttlang_kernel_{name}_{hash}.cpp`.
  - `CompiledTTNNKernel` class: caches kernel paths, configs, arg specs, CB configs, `CoreRangeSet`; callable with `ttnn.Tensor` arguments.
  - `KernelSpec` and `run_kernel_on_device()` (`ttl/kernel_runner.py`): building `TensorAccessorArgs`, kernel descriptors, CB descriptors, and invoking `ttnn.generic_op`.
  - Compilation caching via `_make_cache_key()`: keyed on tensor shapes, dtypes, memory spaces, mesh shape, and `CompilerOptions`.

---

### Chapter 3 — Functional Simulator

**Directory:** `ch3_functional_simulator`

**Description:** Explains TT-Lang's Python-based functional simulator — how it validates kernel correctness without hardware by emulating DFB state machines, multi-core scheduling, and data movement.

**Files:**

- `index.md`
  - Purpose: validate kernel logic, DFB protocols, and data flow before on-device execution.
  - Sim entry point: `sim/operation.py::operation()` decorator vs. the compiler `ttl/ttl_api.py::pykernel_gen`.
  - When to use simulator vs. on-device: sim catches protocol violations and logic bugs but cannot model hardware timing or NOC contention.

- `dfb_state_machine.md`
  - The `BlockStateMachine` and full `STATE_TRANSITIONS` table in `sim/blockstate.py`.
  - `AccessState` lifecycle: MW (Must Write after reserve) -> MR (Must Read after store) -> RW (Read-Write) -> ROR (Read-Only while Reading during copy) -> OS (Out of Scope after push/pop).
  - `DFBContractError` raised on protocol violations (e.g., reading a MW block, double push).
  - Per-thread type enforcement: DM threads can `copy`, compute threads can `store`.

- `multicore_scheduling.md`
  - `GreenletScheduler` (`sim/greenlet_scheduler.py`): cooperative scheduler using greenlets.
  - Each thread (compute/DM) runs in its own greenlet; blocking on `wait()` / `reserve()` switches to scheduler.
  - Scheduling algorithms: `greedy` and `fair` (configurable via `set_scheduler_algorithm()`).
  - `Program` execution (`sim/program.py`): binds thread templates to per-core contexts via `BindableTemplate.bind()`, creates `GreenletScheduler`, runs across grid.

- `resource_limits.md`
  - `set_max_dfbs(limit)` and `set_max_l1_bytes(limit)` in `sim/program.py`.
  - Default L1 limit: 1336 KiB (Blackhole/Wormhole L1 minus reserved program space).
  - Warnings issued when kernel exceeds per-core DFB count or L1 capacity.
  - Simulator `DFBStats` snapshots for debugging buffer utilization.

---

### Chapter 4 — Performance Analysis Tools

**Directory:** `ch4_performance_tools`

**Description:** Covers TT-Lang's profiling and performance analysis infrastructure — environment-variable-driven instrumentation, Perfetto trace visualization, and signpost-based region profiling.

**Files:**

- `index.md`
  - Overview of the three profiling modes and their environment variables.
  - Prerequisite: `TT_METAL_DEVICE_PROFILER=1` and `TT_METAL_HOME` set.

- `auto_profile.md`
  - `TTLANG_AUTO_PROFILE=1`: automatic per-source-line cycle count instrumentation.
  - `SourceLineMapper` class (`ttl/_src/auto_profile.py`): maps signpost markers back to Python source lines.
  - `parse_device_profile_csv()` reading from `$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv`.
  - CB flow graph attribution: `build_cb_wait_to_dma_map()` / `build_dma_producer_to_cb_map()` linking DMA transfers to circular buffer operations.
  - `print_profile_report()` output format with ANSI-colored CB visualization.

- `signpost_profile.md`
  - `TTLANG_SIGNPOST_PROFILE=1` and `with ttl.signpost("name"):` user-defined profiling zones.
  - `ttl.signpost` operator (`ttl/operators.py`) lowered through `ttl-lower-signpost-to-emitc` pass.
  - `parse_signpost_zones()` (`ttl/_src/signpost_profile.py`): parsing zones with `ttl_` prefix from profiler CSV.
  - Per-region cycle count summary output.

- `perf_dump_and_perfetto.md`
  - `TTLANG_PERF_DUMP=1`: NOC profiler summary + CB flow graph + pipe graph dump.
  - `_run_perf_dump()` in `ttl_api.py`: reads NOC traces, prints CB flow graph from `/tmp/ttlang_cb_flow_graph.json`, pipe graph from `/tmp/ttlang_pipe_graph.json`.
  - `TTLANG_PERF_SERV=1`: Perfetto trace server (`ttl/_src/perf_trace_server.py`).
  - Conversion from device profiler CSV to Chrome Trace Event format; HTTP server with Perfetto UI integration via `postMessage`.

---

### Chapter 5 — TT-Symbiote Architecture and Pain Points

**Directory:** `ch5_symbiote_architecture`

**Description:** Analyzes TT-Symbiote's architecture in depth, identifying specific areas where TT-Lang custom kernels could reduce boilerplate, improve performance, and simplify the codebase.

**Files:**

- `index.md`
  - TT-Symbiote's mission: transparent PyTorch-to-TTNN acceleration via `TorchTTNNTensor.__torch_dispatch__`.
  - Architecture diagram: PyTorch model -> `TorchTTNNTensor` wrapping -> ATen dispatch -> TTNN ops on device.

- `ttnn_module_lifecycle.md`
  - `TTNNModule` base class (`core/module.py`): the 3-phase lifecycle — `preprocess_weights()` -> `move_weights_to_device()` -> `forward()` -> `deallocate_weights()`.
  - Boilerplate burden: every module must implement `preprocess_weights_impl()`, `move_weights_to_device_impl()`, `deallocate_weights_impl()`, and `forward()`.
  - Example: `TTNNLinear` (`modules/linear.py`) — 6 methods, explicit `preprocess_linear_weight()` / `preprocess_linear_bias()` calls, manual `ttnn.to_device()` / `ttnn.deallocate()`.
  - `DeviceArch` enum and `@run_on_devices()` decorator for architecture-specific forward paths.
  - `DistributedConfig` and `DistributedTensorConfig` (`core/run_config.py`): `ShardTensor2dMesh`, `ConcatMesh2dToTensor`, `CCLManagerConfig` with manual topology selection.

- `dispatch_system.md`
  - `TorchTTNNTensor` (`core/tensor.py`): PyTorch tensor subclass delegating `__torch_dispatch__` to pluggable dispatchers.
  - Dispatcher registry (`core/dispatchers/dispatcher_config.py`): `DEFAULT`, `CPU`, `DEBUG`, `TENSOR_OPS` dispatchers selected via `TT_SYMBIOTE_DISPATCHER` env var.
  - `default_dispatcher.py`: hand-written ATen-to-TTNN handlers — `handle_view`, `handle_reshape`, `handle_dropout`, `handle_broadcast_tensors`, `_prepare_binary_inputs()`, `_cleanup_tensors()`, etc.
  - Pain point: each new ATen op requires a manually written dispatch handler function.

- `module_catalog.md`
  - Inventory of existing TTNNModule subclasses in `modules/`:
    - `activation.py`: `TTNNSilu`, `TTNNReLU`, `TTNNGelu` — thin wrappers around `ttnn.silu/relu/gelu`.
    - `attention.py`: `TTNNPagedAttentionKVCache`, `PagedAttentionConfig`, multiple attention variants with distributed linear submodules.
    - `linear.py`: `TTNNLinear`, `TTNNLinearSilu`, `TTNNLinearLLamaIColShardedWRowSharded`, `TTNNLinearIColShardedWRowSharded`, `TTNNLinearIReplicatedWColSharded`, `TTNNLinearIColShardedWAllReduced`.
    - `moe.py`: MoE expert dispatch with `_make_sparse_matmul_program_config()`, `_safe_repeat()`, topk/sparsity helpers.
    - `normalization.py`: `TTNNLayerNorm`, `TTNNDistributedRMSNorm`, `DeepseekV2RMSNorm`.
    - `rope.py`: `TTNNRotaryPositionEmbedding`, `TTNNDistributedRotaryPositionEmbedding`.
    - `decoder_layer.py`, `embedding.py`, `conv.py`, `tensor.py`, `qwen_attention.py`, `qwen_moe.py`.

---

### Chapter 6 — Integration Strategy: TT-Lang Kernels in TT-Symbiote

**Directory:** `ch6_integration_strategy`

**Description:** Defines the concrete interface contract and code changes needed to use TT-Lang compiled kernels as drop-in replacements for TTNN ops inside TT-Symbiote modules.

**Files:**

- `index.md`
  - Integration philosophy: TT-Lang kernels are `CompiledTTNNKernel` objects that accept `ttnn.Tensor` in/out, making them callable from any `TTNNModule.forward()`.
  - Key constraint: all tensors must be TILE_LAYOUT, L1 or DRAM memory space.

- `interface_contract.md`
  - `CompiledTTNNKernel.__call__(*tensors)` API: validates tensor count, grid bounds against `device.compute_with_storage_grid_size()`, builds `KernelSpec` list, calls `run_kernel_on_device()`.
  - Tensor requirements: all `ttnn.Tensor`, same memory space, tilized layout.
  - Grid resolution: `_resolve_grid()` supporting static tuples, callables, and `"auto"` (queries device grid).
  - Compilation caching: `_make_cache_key()` on `(shape, dtype, memory_space, layout)` per tensor + mesh shape + `CompilerOptions`.

- `weight_pipeline_interaction.md`
  - How TT-Lang JIT compilation interacts with TT-Symbiote's `preprocess_weights()` -> `move_weights_to_device()` pipeline.
  - Weight tensors as kernel arguments: already on device after `move_weights_to_device_impl()`, compatible with `CompiledTTNNKernel` expectations.
  - Pattern: compile kernel lazily on first `forward()` call, cache `CompiledTTNNKernel` on the module instance.
  - Mesh tensor handling: `_is_mesh_tensor()` check, per-device shard shapes used for compilation.

- `forward_method_changes.md`
  - Current pattern: `TTNNModule.forward()` calls `ttnn.linear()`, `ttnn.silu()`, etc.
  - Proposed pattern: replace TTNN op calls with `self._compiled_kernel(input, weight, output)` in `forward()`.
  - Module-level kernel caching: store `CompiledTTNNKernel` as instance attribute after first compilation, invalidate on shape change via cache key comparison.
  - `@deallocate_weights_after` decorator compatibility.
  - Device placement: `TTNNModule.to_device()` sets `self._device`; kernel grid derived from same device.

---

### Chapter 7 — High-Value Fusion Targets

**Directory:** `ch7_fusion_targets`

**Description:** Identifies the highest-value TT-Symbiote operations for TT-Lang kernel fusion and sketches kernel designs for each.

**Files:**

- `index.md`
  - Selection criteria: operations with multiple sequential TTNN calls, high memory traffic, or complex dispatch logic.

- `moe_expert_pipeline.md`
  - Current MoE implementation in `modules/moe.py`: topk routing -> `_safe_repeat()` -> sparse matmul with `_make_sparse_matmul_program_config()` -> expert combine.
  - Fusion opportunity: fuse topk-dispatch + expert-matmul + combine into a single TT-Lang kernel, eliminating intermediate tensor allocations and repeated NOC transfers.
  - DFB design: input DFB for routed tokens, weight DFBs for expert parameters, output DFB for combined results.

- `fused_attention.md`
  - Current attention in `modules/attention.py`: separate Q/K/V projections -> RoPE -> scaled dot-product -> output projection.
  - Fusion candidates: QKV projection fusion (3 linear ops -> 1 kernel), fused softmax+value-multiply, fused RoPE application.
  - Integration with `TTNNPagedAttentionKVCache` and `PagedAttentionConfig.block_size`.

- `fused_activations.md`
  - Current activation modules (`modules/activation.py`): standalone `ttnn.silu`, `ttnn.relu`, `ttnn.gelu` calls.
  - Fusion with preceding linear: `TTNNLinearSilu` already fuses `linear + silu`; TT-Lang enables arbitrary activation fusion (e.g., SwiGLU = silu(x) * linear(x)).
  - Pattern: fuse activation into matmul compute kernel's post-processing, eliminating intermediate tensor write-back.

---

### Chapter 8 — Developer Workflow and Multi-Device Considerations

**Directory:** `ch8_workflow_and_multidevice`

**Description:** Provides the end-to-end developer workflow for writing, testing, and deploying TT-Lang kernels in TT-Symbiote, plus analysis of multi-device distribution simplification.

**Files:**

- `index.md`
  - The development lifecycle: design -> simulate -> profile -> integrate -> deploy.

- `development_workflow.md`
  - Step 1: Write kernel with `@ttl.operation` using simulator-compatible patterns.
  - Step 2: Validate with functional simulator — run kernel against torch reference, check DFB contract enforcement.
  - Step 3: On-device execution — set `TT_METAL_DEVICE_PROFILER=1`, run kernel, verify numerical correctness.
  - Step 4: Profile — use `TTLANG_AUTO_PROFILE=1` for per-line cycle counts, `TTLANG_SIGNPOST_PROFILE=1` for region timing, `TTLANG_PERF_DUMP=1` for NOC analysis, `TTLANG_PERF_SERV=1` for Perfetto visualization.
  - Step 5: Optimize — tune `CompilerOptions` flags (`--ttl-maximize-dst`, `--ttl-block-matmul`, etc.), adjust DFB `block_count` and `shape` for double-buffering.
  - Step 6: Integrate into TT-Symbiote `TTNNModule` — replace TTNN op calls in `forward()`, add kernel caching.
  - Step 7: Test with TT-Symbiote's model pipeline — ensure weight preprocessing, device placement, and distributed config compatibility.

- `multidevice_simplification.md`
  - Current multi-device code in TT-Symbiote: manual `ShardTensor2dMesh` configuration (`core/run_config.py::DistributedConfig.__post_init__`), `ConcatMesh2dToTensor` composers, `TT_CCL` for all-gather/reduce-scatter, ad-hoc `CCLManagerConfig` with topology selection.
  - TT-Lang's grid model: `ttl.node()` and `ttl.grid_size()` abstract over core identity; `ttl.copy()` with `TensorBlock` indexing handles data placement.
  - Potential: TT-Lang's grid could extend to multi-device grids where `node()` returns (device, core) coordinates, simplifying the `DistributedTensorConfig` + `CCLManager` boilerplate.
  - Limitations: TT-Lang currently targets single-device grids; multi-device would require extending the `CoreRangeSet` construction in `_compile_ttnn_kernel()` and the `_is_mesh_tensor()` path.
  - Near-term approach: use TT-Lang kernels within per-device shards, keep TT-Symbiote's existing CCL coordination for inter-device communication.

---

## Conventions

### Terminology

| Term | Definition |
|---|---|
| **DFB** | Dataflow Buffer — TT-Lang's circular buffer abstraction (`sim/dfb.py::DataflowBuffer`, `ttl/circular_buffer.py::CircularBuffer`). |
| **Block** | A logically contiguous window into a DFB's ring buffer, acquired via `wait()` or `reserve()`. |
| **TensorBlock** | Compile-time proxy for tile data in the MLIR compilation path (`ttl/operators.py::TensorBlock`). |
| **Grid** | 2D array of Tensix cores executing a kernel; specified as `(cols, rows)` matching tt-metal `CoreCoord` convention. |
| **Node** | A single core within the grid, identified by `ttl.node()` coordinates. |
| **TTNNModule** | TT-Symbiote's base class for TTNN-accelerated modules (`core/module.py`). |
| **Dispatcher** | TT-Symbiote's ATen-to-TTNN operation routing system (`core/dispatchers/`). |
| **CompiledTTNNKernel** | A cached, callable TT-Lang kernel ready for execution via `ttnn.generic_op`. |

### Notation

- File paths are always relative to their repository root: `/localdev/salnahari/testing_dir/tt-lang/` for TT-Lang, `/localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote/` for TT-Symbiote.
- MLIR pass names use their CLI form (e.g., `convert-ttl-to-compute`), corresponding C++ source files are noted in parentheses.
- Code snippets use Python unless otherwise noted; C++ snippets are labeled with `// C++`.
- Environment variables are written in `UPPER_SNAKE_CASE` with backtick formatting.

### Formatting Rules

- Each chapter's `index.md` begins with a one-paragraph summary and ends with a "Key Takeaways" bullet list.
- API references include the full module path (e.g., `ttl.ttl_api.CompiledTTNNKernel`).
- Diagrams use Mermaid syntax embedded in fenced code blocks.
- Cross-references to other chapters use relative links: `[Chapter N title](../chN_dir/file.md)`.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Concepts Referenced |
|---|---|---|
| Ch2 (Compilation Pipeline) | Ch1 (Programming Model) | `@ttl.operation`, `@ttl.compute`, `@ttl.datamovement`, DFB, `CircularBuffer`, `TensorBlock`, grid model |
| Ch3 (Functional Simulator) | Ch1 (Programming Model) | DFB, `Block`, `AccessState` lifecycle, `Program`, grid |
| Ch4 (Performance Tools) | Ch2 (Compilation Pipeline) | MLIR passes (`ttl-dump-cb-flow-graph`, `ttl-lower-signpost-to-emitc`), `CompiledTTNNKernel`, `CompilerOptions` |
| Ch5 (Symbiote Architecture) | None (self-contained) | — |
| Ch6 (Integration Strategy) | Ch1 (Programming Model), Ch2 (Compilation Pipeline), Ch5 (Symbiote Architecture) | `CompiledTTNNKernel`, `TTNNModule` lifecycle, `TorchTTNNTensor`, weight pipeline, compilation caching |
| Ch7 (Fusion Targets) | Ch1 (Programming Model), Ch5 (Symbiote Architecture), Ch6 (Integration Strategy) | DFB design patterns, `TTNNModule` subclasses (MoE, attention, activations), `CompiledTTNNKernel` integration |
| Ch8 (Workflow & Multi-Device) | Ch3 (Functional Simulator), Ch4 (Performance Tools), Ch6 (Integration Strategy) | Simulator validation, profiling workflow, `DistributedConfig`, mesh tensors, kernel caching |
