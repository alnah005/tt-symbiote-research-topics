# Codegen and Execution

**Source:** `python/ttl/ttl_api.py`, `python/ttl/kernel_runner.py`

After the MLIR pass pipeline completes, the module contains EmitC operations that map directly to C++ source. This stage extracts the C++ text, writes it to disk, builds execution descriptors, and dispatches the kernel to hardware.

## C++ Source Generation

The function `ttkernel_to_cpp_by_name()` (from `ttl.passes`) serializes a named function in the post-pipeline MLIR module to a C++ source string. Each thread function in the module produces one C++ file:

```python
for name, thread_type in kernel_info:
    cpp_source = ttkernel_to_cpp_by_name(module, name)
    kernel_path = _write_kernel_to_tmp(name, cpp_source)
```

## Kernel File Writing

`_write_kernel_to_tmp()` writes C++ source to a content-addressed path under `/tmp`:

```python
def _write_kernel_to_tmp(name: str, source: str) -> str:
    content_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    user = os.environ.get("USER", "default")
    path = f"/tmp/{user}/ttlang_kernel_{name}_{content_hash}.cpp"
    os.makedirs(f"/tmp/{user}", exist_ok=True)
    with open(path, "w") as f:
        f.write(source)
    return path
```

The MD5 content hash in the filename means identical C++ output reuses the same file path, providing implicit deduplication across compilations.

## _compile_ttnn_kernel: Building the Compiled Kernel

**Source:** `_compile_ttnn_kernel()` in `ttl_api.py`

This function bridges the MLIR compilation output and the runtime execution layer. It:

1. Extracts kernel names and thread types via `get_ttkernel_names(module)`.
2. Validates that exactly 3 kernels exist (1 compute + 2 data movement), matching the Tensix core's thread model.
3. Validates all tensors are the same type (all ttnn or all torch, no mixing).
4. Builds a `CoreRangeSet` from the grid dimensions.
5. For each kernel, generates the C++ file and creates a config descriptor:

| Thread Type | Config | RISC Thread |
|-------------|--------|-------------|
| `"compute"` | `ttnn.ComputeConfigDescriptor()` | TRISC_0, TRISC_1, TRISC_2 |
| `"noc"` (first) | `ttnn.ReaderConfigDescriptor()` | NCRISC |
| `"noc"` (second) | `ttnn.WriterConfigDescriptor()` | BRISC |

6. Auto-detects FP32 requirements:
   - If any input tensor is `float32`, enables `fp32_dest_acc_en`
   - If `reduce_full_fp32` is set and the kernel contains `reduce_tile`, enables FP32 accumulation
   - If `matmul_full_fp32` is set and the kernel contains `matmul_block` (but not `unary_bcast`), enables FP32 accumulation

7. Extracts runtime arg specs via `get_ttkernel_arg_spec()`.
8. Returns a `CompiledTTNNKernel` instance.

## CompiledTTNNKernel

**Source:** `CompiledTTNNKernel` class in `ttl_api.py`

This class stores all pre-compiled kernel artifacts and implements `__call__` for execution:

```python
class CompiledTTNNKernel:
    def __init__(self,
        kernel_paths,            # List of (path, thread_type) tuples
        kernel_configs,          # List of config descriptors
        kernel_arg_specs,        # List of runtime arg specs
        num_tensors,             # Number of input/output tensors
        core_ranges,             # ttnn.CoreRangeSet
        kernel_tensor_indices,   # Per-thread global tensor index lists
        cb_configs=None,         # CircularBuffer objects indexed by cb_index
        program_hash=None,       # Hash for tt-metal program cache
        ...
    ):
```

On invocation (`__call__`), it:

1. Validates tensor count matches `num_tensors`.
2. Validates the kernel grid fits within the device's compute grid.
3. Builds `KernelSpec` objects from stored paths, configs, and tensor indices.
4. Delegates to `run_kernel_on_device()`.

## KernelSpec

**Source:** `python/ttl/kernel_runner.py`

A lightweight dataclass describing a single kernel thread:

```python
@dataclass
class KernelSpec:
    path: str                  # Path to C++ source file
    thread_type: str           # "compute", "noc", or "ethernet"
    tensor_indices: List[int]  # Global tensor indices this kernel accesses
    config: Any                # ComputeConfigDescriptor, ReaderConfigDescriptor, etc.
```

The `tensor_indices` field maps the kernel's local tensor arguments to positions in the global tensor list, enabling `run_kernel_on_device()` to build per-kernel `common_runtime_args` (buffer addresses).

## run_kernel_on_device

**Source:** `python/ttl/kernel_runner.py`

The main execution entry point:

```python
def run_kernel_on_device(kernel_specs, tensors, cb_configs, core_ranges, program_hash=None):
```

### Step 1: Build Tensor Accessor Args

```python
tensor_accessor_args = build_tensor_accessor_args(tensors)
# Calls ttnn.TensorAccessorArgs(tensor).get_compile_time_args() for each tensor
# Returns flattened list of compile-time metadata (is_sharded, is_dram, etc.)
```

### Step 2: Build Kernel Descriptors

For each `KernelSpec`, `build_kernel_descriptors()` creates a `ttnn.KernelDescriptor`:

```python
# Per-kernel runtime args: buffer addresses for this kernel's tensors
common_runtime_args = [tensors[idx].buffer_address() for idx in spec.tensor_indices]

# Compile-time args differ by thread type:
if spec.thread_type == "compute":
    kernel_compile_time_args = cb_indices              # CB indices only
else:
    kernel_compile_time_args = cb_indices + tensor_accessor_args  # CB + TA metadata

kernel_desc = ttnn.KernelDescriptor(
    kernel_source=spec.path,
    core_ranges=core_ranges,
    compile_time_args=kernel_compile_time_args,
    common_runtime_args=common_runtime_args,
    config=spec.config,
)
```

### Step 3: Build CB Descriptors

For each [CircularBuffer](../ch1_programming_model/index.md) in `cb_configs`, `build_cb_descriptors()` creates a `ttnn.CBDescriptor`:

```python
page_size = tile_bytes_from_dtype(data_format)
num_tiles = cb.shape[0] * cb.shape[1] * cb.block_count
total_size = num_tiles * page_size

cb_format = ttnn.CBFormatDescriptor(
    buffer_index=i,
    data_format=data_format,
    page_size=page_size,
)
cb_desc = ttnn.CBDescriptor(
    total_size=total_size,
    core_ranges=core_ranges,
    format_descriptors=[cb_format],
)
```

### Step 4: Build and Execute Program

```python
program = ttnn.ProgramDescriptor(
    kernels=kernel_descriptors,
    cbs=cb_descriptors,
    semaphores=[],
)

# Ensure >= 2 tensors (ttnn.generic_op requirement)
io_tensors = list(tensors)
if len(io_tensors) < 2:
    io_tensors = [io_tensors[-1]] + io_tensors  # Duplicate output as dummy input

return ttnn.generic_op(io_tensors, program)
```

## Compilation Caching

The `pykernel_gen` decorator maintains a per-kernel cache:

```python
cache: Dict[tuple, CompiledTTNNKernel] = {}
```

The cache key is built by `_make_cache_key()`:

```python
def _make_cache_key(args, fp32_dest_acc_en, dst_full_sync_en, compiler_options):
    tensor_key = tuple(
        _get_tensor_cache_info(arg) for arg in args if is_ttnn_tensor(arg)
    )
    # _get_tensor_cache_info returns: (shape, dtype, memory_space, layout)
    mesh_key = None  # tuple(device.shape) for multi-device tensors
    return (tensor_key, mesh_key, fp32_dest_acc_en, dst_full_sync_en, compiler_options)
```

Because `CompilerOptions` is frozen and hashable, it participates directly in the cache key. This means changing a single compiler flag (e.g., `--no-ttl-maximize-dst`) triggers recompilation, while repeated calls with identical tensor metadata and options hit the cache.

## End-to-End Data Flow Summary

```
pykernel_gen.__call__(tensors)
  │
  ├─ cache hit? ──► CompiledTTNNKernel.__call__(tensors)
  │                      │
  │                      ├─ Build KernelSpec list
  │                      └─ run_kernel_on_device()
  │                           ├─ build_tensor_accessor_args()
  │                           ├─ build_kernel_descriptors()
  │                           ├─ build_cb_descriptors()
  │                           └─ ttnn.generic_op(io_tensors, program)
  │
  └─ cache miss ──► _compile_kernel()
                       ├─ Thread compilation (TTLGenericCompiler per thread)
                       ├─ Module merge
                       ├─ PassManager.run() (17+ passes)
                       ├─ _compile_ttnn_kernel()
                       │    ├─ ttkernel_to_cpp_by_name() → C++ strings
                       │    ├─ _write_kernel_to_tmp() → /tmp files
                       │    └─ CompiledTTNNKernel(...)
                       └─ Cache store
```

---

**Next:** [Chapter 3 — Functional Simulator](../ch3_functional_simulator/index.md)
