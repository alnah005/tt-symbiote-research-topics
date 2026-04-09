# TensorBlocks and Grid Execution

This section covers the operator-overloaded `TensorBlock` / `Block` types, asynchronous data transfers via `ttl.copy()` and `CopyTransferHandler`, the grid intrinsics `ttl.node()` and `ttl.grid_size()`, and ties everything together with an annotated walkthrough of `eltwise_add.py`.

## TensorBlock Operator Overloading

TT-Lang provides operator overloading on block types so that compute threads read like natural Python arithmetic.

### Compiler Path: `TensorBlock`

**Source:** `python/ttl/operators.py`

In the compiler path, `TensorBlock` is an MLIR-backed type decorated with `@syntax("!tensor")`. Each Python operator emits a corresponding TTL dialect operation:

```python
@syntax("!tensor")
class TensorBlock:
    def __add__(ast_self, rhs):
        return ttl.add(ast_self.type, ast_self, rhs)    # Emits ttl.add op

    def __sub__(ast_self, rhs):
        return ttl.sub(ast_self.type, ast_self, rhs)    # Emits ttl.sub op

    def __mul__(ast_self, rhs):
        return ttl.mul(ast_self.type, ast_self, rhs)    # Emits ttl.mul op

    def __truediv__(ast_self, rhs):
        return ttl.div(ast_self.type, ast_self, rhs)    # Emits ttl.div op

    def __matmul__(ast_self, rhs):
        # C[M,N] = A[M,K] @ B[K,N]
        result_shape = [lhs_shape[0], rhs_shape[1]]
        return ttl.matmul(result_type, ast_self, rhs)   # Emits ttl.matmul op
```

The `store()` method writes a result into a reserved block:

```python
def store(ast_self, rhs):
    reserve = _get_reserve_from_block(ast_self)
    ttl.store(rhs, reserve)    # Emits ttl.store op
```

### Simulator Path: `Block`

**Source:** `python/sim/dfb.py`

In the simulator, the `Block` class implements the same operators using PyTorch under the hood via a generic `_binary_op` method:

```python
# From sim/dfb.py
def _binary_op(self, other, op):
    # Validate shapes are compatible
    left_shape = self._shape
    right_shape = other._shape
    # ... broadcast handling ...
    return self._create_temporary_result(
        op(left_buf, right_buf), result_shape, other
    )

def __add__(self, other):   return self._binary_op(other, operator.add)
def __sub__(self, other):   return self._binary_op(other, operator.sub)
def __mul__(self, other):   return self._binary_op(other, operator.mul)
def __truediv__(self, other): return self._binary_op(other, operator.truediv)
```

Binary operations return a **temporary Block** — a Block with `is_temporary=True` that is not backed by any DFB. Temporary blocks have unrestricted access state (no state machine validation) and carry provenance tracking: `_source_blocks` records which `wait()` blocks contributed data, so the state machine can correctly validate the chain from `wait()` through arithmetic to `store()`.

**In-place accumulation** is supported for temporary blocks:

```python
# Allowed: y is temporary (from fill or expression)
y = ttl.math.fill(0)
y += a_blk @ b_blk  # Calls __iadd__, returns new temporary
```

## `CopyTransferHandler` and `ttl.copy()`

Asynchronous data transfer between device tensors and DFB blocks is the responsibility of `ttl.copy()`.

### Compiler Path

**Source:** `python/ttl/operators.py`

In the compiler path, `copy()` emits `ttl.copy` MLIR ops. It distinguishes reads (tensor-to-block) from writes (block-to-tensor) by inspecting which argument is a block:

```python
@syntax("copy")
def copy(src, dst) -> CopyTransferHandler:
    if dst_is_block and not src_is_block:
        # Read: device tensor slice -> CB
        xf_type = Type.parse("!ttl.transfer_handle<read>", ctx)
        return ttl.copy(xf_type, src, dst_cb)
    elif src_is_block and not dst_is_block:
        # Write: CB -> device tensor slice
        xf_type = Type.parse("!ttl.transfer_handle<write>", ctx)
        return ttl.copy(xf_type, src_cb, dst)
```

The returned `CopyTransferHandler` (decorated with `@syntax("!ttl.transfer_handle")`) has a single method:

```python
class CopyTransferHandler:
    def wait(ast_self):
        return ttl.wait(ast_self)    # Emits ttl.wait op
```

### Simulator Path

**Source:** `python/sim/copy.py`, `python/sim/copyhandlers.py`

In the simulator, `copy()` creates a `CopyTransaction` that uses a **registry-based handler pattern**. Each `(source_type, dest_type)` pair has a registered handler:

| Source | Destination | Handler Class |
|--------|-------------|---------------|
| `Tensor` | `Block` | `TensorToBlockHandler` |
| `Block` | `Tensor` | `BlockToTensorHandler` |
| `Block` | `Pipe` | `BlockToPipeHandler` |
| `Pipe` | `Block` | `PipeToBlockHandler` |

Handlers are registered at import time using a decorator:

```python
# From sim/copyhandlers.py
@register_copy_handler(Tensor, Block)
class TensorToBlockHandler:
    def validate(self, src, dst):
        # Check layout match and tile count match
        ...
    def transfer(self, src, dst):
        record_tensor_read(src)
        dst.copy_as_dest(src)
    def can_wait(self, src, dst):
        return True  # Tensor<->Block transfers are synchronous
```

The `CopyTransaction.wait()` method drives the state machine:

1. Calls `block_if_needed()` — yields the greenlet if the handler reports `can_wait() == False` (relevant for Pipe-based transfers).
2. Calls `handler.transfer()` — performs the actual data copy.
3. Calls `mark_tx_wait_complete()` on both source and destination blocks — transitions their state machines.

### `GroupTransfer` — Batched Transfers

For multicast patterns, `GroupTransfer` collects multiple `CopyTransaction` handles and waits on them together:

```python
gxf = GroupTransfer()
for dst in destinations:
    gxf.add(ttl.copy(src_blk, dst))
gxf.wait_all()  # Waits for all transfers sequentially
```

## Grid Intrinsics

### `ttl.node(dims=N)` — Current Core Coordinates

**Source:** `python/sim/corecontext.py`

Returns the coordinates of the current core within the grid. The `dims` parameter controls the dimensionality of the returned value:

```python
node_col, node_row = ttl.node(dims=2)  # Returns (x, y) tuple
linear_id = ttl.node(dims=1)           # Returns flattened index
```

Implementation walks the call stack to find the `_core` variable (injected by `Program` into each core's context) and decomposes it into coordinates:

```python
# From sim/corecontext.py
def node(dims=2):
    cid = _get_from_frame("_core", "...")
    grid = _get_from_frame("grid", "...")
    coords = []
    for s in reversed(grid):
        coords.append(cid % s)
        cid = cid // s
    coords.reverse()
    if dims == 1:
        return coords[0]
    return tuple(coords)
```

For a grid of shape $(C, R)$ and linear core ID $k$:

$$\text{node\_col} = \lfloor k / R \rfloor, \quad \text{node\_row} = k \bmod R$$

This follows from the decomposition loop in `corecontext.py`, which iterates over grid dimensions in reverse: the last dimension ($R$) is extracted first via modulo, and the first dimension ($C$) is extracted last via integer division.

### `ttl.grid_size(dims=N)` — Grid Dimensions

**Source:** `python/sim/corecontext.py`

Returns the grid dimensions, with flattening or padding as needed:

```python
grid_cols, grid_rows = ttl.grid_size(dims=2)  # Returns (C, R) tuple
total_cores = ttl.grid_size(dims=1)            # Returns C * R
```

Behavior with dimension mismatch:
- **`dims < grid_dims`:** Higher-rank dimensions are flattened. For a `(4, 8)` grid, `grid_size(dims=1)` returns 32.
- **`dims > grid_dims`:** Lower-rank dimensions are padded with 1. For a `(4, 8)` grid, `grid_size(dims=3)` returns `(4, 8, 1)`.

### Compiler Path Equivalents

In the compiler path (`ttl/operators.py`), `node(dims=2)` emits `(ttl.core_x(), ttl.core_y())` and `grid_size(dims=2)` returns the module-level `_current_grid` tuple directly. Currently only `dims=2` is supported in the compiler path.

## Walkthrough: `eltwise_add.py`

**Source:** `examples/eltwise_add.py`

This example performs elementwise addition of two tensors across a multi-node grid. It demonstrates every concept covered in this chapter.

### Constants and Kernel Signature

```python
TILE_SIZE = 32
GRANULARITY = 2   # Each block holds 2 tile-rows

@ttl.operation(grid="auto")
def eltwise_add(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
```

The `grid="auto"` resolves to the default grid (e.g., `(8, 8)` for a 64-core chip).

### Tiling Computation

```python
    row_tiles = a_in.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = a_in.shape[1] // TILE_SIZE
```

For a $256 \times 256$ input with `TILE_SIZE=32` and `GRANULARITY=2`:
- $\text{row\_tiles} = 256 / 32 / 2 = 4$
- $\text{col\_tiles} = 256 / 32 = 8$

### Grid Partitioning

```python
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)   # Ceiling division
    cols_per_node = -(-col_tiles // grid_cols)
```

Each core gets a ceil-divided share of the tile work. The `-(-a // b)` idiom computes $\lceil a/b \rceil$ in Python.

### Dataflow Buffer Creation

```python
    a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(GRANULARITY, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(GRANULARITY, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(GRANULARITY, 1), block_count=2)
```

Each DFB holds blocks of shape `(2, 1)` — two tile-rows by one tile-column. The `block_count=2` enables double buffering (see [`dataflow_buffers.md`](./dataflow_buffers.md#make_dataflow_buffer_like) for details).

### Shared Tiling Loop

All three threads iterate over the same tile grid, with each core processing its assigned partition. The common loop structure is:

```python
    node_col, node_row = ttl.node(dims=2)
    for local_row in range(rows_per_node):
        row = node_row * rows_per_node + local_row
        if row < row_tiles:
            for local_col in range(cols_per_node):
                col = node_col * cols_per_node + local_col
                if col < col_tiles:
                    # Thread-specific body (see below)
                    ...
```

The DM threads additionally compute tile-row extents inside the outer loop: `r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY`.

What differs between threads is only the inner body:

### Compute Thread

```python
    @ttl.compute()
    def compute():
        # ... shared tiling loop ...
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            out_blk.store(a_blk + b_blk)
```

The `with` block acquires input blocks via `wait()` and an output slot via `reserve()`; on exit, `pop()` and `push()` fire automatically (see [`dataflow_buffers.md`](./dataflow_buffers.md#context-manager-usage)).

### Read DM Thread (DM0)

```python
    @ttl.datamovement()
    def read():
        # ... shared tiling loop (with r0, r1 computation) ...
                        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                            tx_a = ttl.copy(a_in[r0:r1, col:col+1], a_blk)
                            tx_b = ttl.copy(b_in[r0:r1, col:col+1], b_blk)
                            tx_a.wait()
                            tx_b.wait()
```

Key observations:
- **`reserve()` in DM** — the read thread is the *producer* for `a_dfb` and `b_dfb`.
- **Tensor slicing** — `a_in[r0:r1, col:col+1]` selects a tile-coordinate range from the device tensor.
- **Two concurrent copies** — `tx_a` and `tx_b` are initiated before either `wait()` is called, enabling overlap.
- On `__exit__`, `a_blk.push()` and `b_blk.push()` make the data visible to the compute thread's `wait()`.

### Write DM Thread (DM1)

```python
    @ttl.datamovement()
    def write():
        # ... shared tiling loop (with r0, r1 computation) ...
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, col:col+1])
                            tx.wait()
```

This thread is the *consumer* of `out_dfb`: it `wait()`s for the compute thread to `push()` a result, then copies it out to the device tensor.

### Host Driver

```python
def main():
    device = ttnn.open_device(device_id=0)
    a = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.from_torch(torch.zeros_like(a_torch), ...)

    eltwise_add(a, b, out)   # Launches the kernel

    result = ttnn.to_torch(out)
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
```

The host code creates `TILE_LAYOUT` tensors on device, invokes the kernel as a regular Python function call, and verifies correctness.

### Data Flow Summary

```
  Device Tensor a_in ──┐                          ┌── Device Tensor out
  Device Tensor b_in ──┤                          │
                       ▼                          ▲
               ┌──────────────┐            ┌──────────────┐
   DM0 (read): │  a_dfb.reserve()         │  out_dfb.wait()  : DM1 (write)
               │  copy(a_in→a_blk)        │  copy(out_blk→out)
               │  copy(b_in→b_blk)        │
               │  push()                   │  pop()
               └──────┬───────┘            └───────┬──────┘
                      ▼                            ▲
               ┌──────────────────────────────────────┐
    Compute:   │  a_dfb.wait(), b_dfb.wait()          │
               │  out_dfb.reserve()                    │
               │  out_blk.store(a_blk + b_blk)         │
               │  pop(), pop(), push()                 │
               └──────────────────────────────────────┘
```

Each core independently processes its assigned tile range. The three threads synchronize exclusively through the DFBs' ring buffer counters — no explicit locks or barriers are needed.

---

**Next:** [Chapter 2 — Compilation Pipeline](../ch2_compilation_pipeline/index.md)
