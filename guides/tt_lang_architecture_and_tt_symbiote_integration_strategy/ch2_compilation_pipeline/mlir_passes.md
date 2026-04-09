# MLIR Pass Pipeline

**Source:** `python/ttl/ttl_api.py` (`_compile_kernel`), `lib/Dialect/TTL/Transforms/`, `lib/Dialect/TTKernel/Transforms/`

The compilation pipeline applies a carefully ordered sequence of MLIR passes to lower the TTL dialect module into EmitC, which is then serialized to C++ source. The pass pipeline is assembled in `_compile_kernel()` and executed through a single `PassManager.run()` call.

## CompilerOptions

**Source:** `python/ttl/compiler_options.py`

The `CompilerOptions` dataclass controls which passes run and how they are configured:

```python
@dataclasses.dataclass(frozen=True)
class CompilerOptions:
    maximize_dst: bool = True          # DST subblock tiling + scheduling
    enable_fpu_binary_ops: bool = True # Use FPU for binary add/sub/mul
    use_block_matmul: bool = True      # Lower matmul to block-level HW calls
    auto_sync: bool = False            # Compiler-inserted DFB synchronization
    combine_pack_tiles: bool = True    # Merge consecutive pack_tile ops
    reduce_full_fp32: bool = True      # FP32 accumulation for reductions
    matmul_full_fp32: bool = True      # FP32 accumulation for matmul
```

The frozen dataclass is hashable and used directly as a cache key component. Options are resolved via a three-tier priority merge:

```
sys.argv  >  TTLANG_COMPILER_OPTIONS env var  >  decorator options= string
```

The merge uses an explicit-field tracking mechanism (`_explicit` frozenset) so that unmentioned flags fall through from the base:

```python
base = CompilerOptions.from_string(opts_str)      # decorator + env var
argv_overrides = CompilerOptions.from_argv()       # sys.argv
compiler_options = base.merge(argv_overrides)      # highest priority wins
```

## The Full Pass Pipeline

The pipeline is constructed as a comma-separated string and parsed by `PassManager.parse()`. Below is every pass in execution order, with its C++ source file, what it does, and whether it is conditional.

### Phase 1: TTL Dialect Lowering

| # | Pass | Source | Description | Conditional? |
|---|------|--------|-------------|-------------|
| 1 | `func.func(convert-ttl-to-compute)` | `ConvertTTLToCompute.cpp` | Rewrites high-level TTL ops (e.g., `ttl.add`, `ttl.mul`, `ttl.matmul`) into `ttl.compute` regions with explicit tensor operands and CB store/load patterns. Builds init tensors for output types. | Always |
| 2 | `func.func(ttl-set-compute-kernel-config)` | `TTLSetComputeKernelConfig.cpp` | Sets compute configuration attributes (`fp32_dest_acc_en`, `dst_full_sync_en`, `reduce_full_fp32`, `matmul_full_fp32`) on `ttl.compute` ops so downstream passes see stable, explicit settings. | Always |
| 3 | `func.func(ttl-assign-dst)` | `TTLAssignDST.cpp` | DST register allocation via interval-based linear scan. Three phases: (1) copy insertion for multi-consumer values, (2) live interval construction with in-place op merging (union-find), (3) linear scan allocation (Wimmer & Franz, CGO'10). Assigns `dst_idx` attributes to all tile compute ops. Controlled by `enable_fpu_binary_ops` flag. | Always |
| 4 | `func.func(ttl-subblock-compute-for-dst)` | `TTLSubblockComputeForDST.cpp` | Partitions `ttl.compute` regions into DST-sized subblocks via `TilingInterface`. Finds subblock sizes $[t_0, t_1, \ldots]$ where each $t_i$ divides the dimension size and $\prod t_i \leq \text{unroll\_factor}$. The `subblock-sync` option controls whether [DFB](../ch1_programming_model/index.md) synchronization ops are inserted at subblock boundaries. | `maximize_dst=True` |
| 5 | `func.func(ttl-lower-matmul-block)` | `TTLLowerMatmulBlock.cpp` | Replaces `ttl.compute` ops containing `tile_matmul_block` with a linear sequence: sync acquire, `matmul_block`, $M \times N$ tile stores, sync release. CB lifecycle ops (wait/pop, reserve/push) are NOT emitted here — they come from user DFB operations. | `use_block_matmul=True` |
| 6 | `func.func(ttl-lower-to-loops)` | `ConvertTTLComputeToSCF.cpp` | Converts `ttl.compute` regions into SCF loops. When `dst-accumulation=true` (tied to `maximize_dst`), generates accumulation-aware loop nests that respect DST register lifetimes. | Always |
| 7 | `func.func(ttl-schedule-operations)` | `TTLScheduleOperations.cpp` | Schedules tile operations within DST sync regions by `dst_idx`. Sorts operations to minimize register pressure and maximize hardware pipeline utilization. Uses init-affinity keys for deterministic sub-sorting within the same op type. | `maximize_dst=True` |
| 8 | `func.func(ttl-annotate-cb-associations)` | `TTLAnnotateCBAssociations.cpp` | Analysis pass that annotates [CircularBuffer](../ch1_programming_model/index.md) index associations on TTL ops (`ttl.cb_index.<N>`, `ttl.bcast_output_cb_index`). Enables subsequent conversion passes to find the correct CB without SSA tracing across multi-phase lowering. | Always |

### Phase 2: Optional Analysis (Profiling)

| # | Pass | Source | Description | Conditional? |
|---|------|--------|-------------|-------------|
| 9 | `ttl-dump-cb-flow-graph` | `TTLDumpCBFlowGraph.cpp` | Builds and dumps the CB producer/consumer flow graph to JSON. Enables the auto-profiler to correlate runtime barrier timings with source-level CB operations. | `TTLANG_PERF_DUMP=1` or `TTLANG_AUTO_PROFILE=1` |

### Phase 3: Cross-Dialect Lowering

| # | Pass | Source | Description | Conditional? |
|---|------|--------|-------------|-------------|
| 10 | `ttl-lower-dprint-to-emitc` | `LowerDPrintToEmitC.cpp` | Converts `ttl.dprint` operations to EmitC verbatim calls that emit `DPRINT` C++ macros. | Always |
| 11 | `convert-ttl-to-ttkernel` | `ConvertTTLToTTKernel.cpp` | Major dialect conversion: lowers all remaining TTL ops to the TTKernel dialect. Converts tensor types to TTKernel CB types, replaces TTL CB operations with TTKernel equivalents, and lowers [TensorBlock](../ch1_programming_model/index.md) addressing to compile-time/runtime arg lookups. Uses `TTLToTTKernelTypeConverter` and the `ttl.base_cta_index` / `ttl.crta_indices` function attributes set during thread compilation. Parameterized by `reduce-full-fp32`. | Always |

### Phase 4: TTKernel Optimization

| # | Pass | Source | Description | Conditional? |
|---|------|--------|-------------|-------------|
| 12 | `ttkernel-insert-inits` | `TTKernelInsertInits.cpp` | Inserts hardware initialization calls in two phases: (1) common inits (`init_sfpu`, `binary_op_init_common`) once per sync region, hoisted above enclosing loops; (2) per-op inits (`exp_tile_init`, `add_tiles_init`, etc.) in linear block order whenever the op type changes. Derives input/output CBs from compute and pack ops. | Always |
| 13 | `ttkernel-insert-l1-accumulation` | `TTKernelInsertL1Accumulation.cpp` | Inserts `pack_reconfig_l1_acc` guards inside reduction loops. From the second iteration onward, the packer switches to L1 accumulation mode so `pack_tile` adds to the existing L1 value instead of overwriting. | Always |
| 14 | `func.func(ttkernel-combine-pack-tiles)` | `TTKernelCombinePackTiles.cpp` | Combines consecutive `pack_tile` ops on the same dataflow buffer with contiguous DST and DFB tile indices into a single `pack_tile_block` call. Reduces instruction count. | `combine_pack_tiles=True` |

### Phase 5: Standard MLIR Cleanup + EmitC

| # | Pass | Source | Description | Conditional? |
|---|------|--------|-------------|-------------|
| 15 | `canonicalize` | MLIR upstream | Standard canonicalization patterns (constant folding, dead code elimination, algebraic simplifications). | Always |
| 16 | `cse` | MLIR upstream | Common subexpression elimination. | Always |
| 17 | `lower-affine` | MLIR upstream | Lowers affine operations to standard SCF/arith operations. | Always |
| 18 | `ttl-lower-signpost-to-emitc` | `LowerSignpostToEmitC.cpp` | Converts profiling signpost ops to EmitC verbatim calls. Skips cheap coordinate-lookup ops that should not trigger profiling scopes. | Always |
| 19 | `convert-ttkernel-to-emitc` | (ttmlir upstream) | Final conversion: all TTKernel ops become EmitC operations (function calls, verbatim C++, includes). | Always |
| 20 | `symbol-dce` | MLIR upstream | Dead symbol elimination. Removes unused function declarations left after conversion. | Always |

> **Note:** The numbered count exceeds 17 because the profiling pass (9) and cleanup passes (15-17, 20) are sometimes counted separately. The "17-pass" figure refers to the core compilation passes excluding profiling and standard cleanup.

## Pipeline Assembly

The pipeline is built as a Python list and joined:

```python
pipeline_passes = [
    "func.func(convert-ttl-to-compute)",
    set_compute_config_pass,                           # with fp32/reduce/matmul options
    f"func.func({assign_dst_pass})",                   # with enable-fpu-binary-ops flag
]
if compiler_options.maximize_dst:
    pipeline_passes.append("func.func(ttl-subblock-compute-for-dst{...})")
if compiler_options.use_block_matmul:
    pipeline_passes.append("func.func(ttl-lower-matmul-block)")
pipeline_passes.append("func.func(ttl-lower-to-loops{dst-accumulation=...})")
if compiler_options.maximize_dst:
    pipeline_passes.append("func.func(ttl-schedule-operations)")
pipeline_passes.append("func.func(ttl-annotate-cb-associations)")
# ... profiling passes (conditional) ...
pipeline_passes += [
    "ttl-lower-dprint-to-emitc",
    f"convert-ttl-to-ttkernel{{reduce-full-fp32={flag}}}",
    "ttkernel-insert-inits",
    "ttkernel-insert-l1-accumulation",
]
if compiler_options.combine_pack_tiles:
    pipeline_passes.append("func.func(ttkernel-combine-pack-tiles)")
pipeline_passes += [
    "canonicalize", "cse", "lower-affine",
    "ttl-lower-signpost-to-emitc",
    "convert-ttkernel-to-emitc",
    "symbol-dce",
]

pipeline_str = f"builtin.module({','.join(pipeline_passes)})"
pm = PassManager.parse(pipeline_str)
pm.enable_verifier(True)
pm.run(module.operation)
```

## Debugging the Pipeline

Several environment variables control pipeline introspection:

| Variable | Effect |
|----------|--------|
| `TTLANG_VERBOSE_PASSES=1` | Prints IR before and after every pass (disables multithreading) |
| `TTLANG_INITIAL_MLIR=path` | Saves pre-pipeline MLIR to file |
| `TTLANG_FINAL_MLIR=path` | Saves post-pipeline MLIR to file |
| `TTLANG_DEBUG_LOCATIONS=1` | Includes source locations in printed MLIR output |

## Error Handling

If any pass fails, the error handler in `_compile_kernel()` catches the exception and calls `format_mlir_error()` to map MLIR file locations back to the original Python source:

```python
try:
    pm.run(module.operation)
except Exception as e:
    formatted = format_mlir_error(str(e), source_lines, source_file)
    raise RuntimeError(formatted) from None
```

This produces error messages that point directly to the offending Python line, even though the failure occurred deep inside an MLIR pass.

---

**Next:** [`codegen_and_execution.md`](./codegen_and_execution.md)
