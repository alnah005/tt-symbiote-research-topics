# TT Transformers Multi-Chip Parallelism

Source files: `models/tt_transformers/tt/ccl.py`, `models/tt_transformers/tt/mlp.py`, `models/tt_transformers/tt/attention.py`, `models/tt_transformers/tt/model.py`

---

## 1. TT_CCL Class

`TT_CCL` (`ccl.py` lines 59-133) is a thin state container that owns all pre-allocated global semaphores needed by asynchronous collective operations. It is constructed once in `Transformer.__init__` and passed down to every `TransformerBlock`, `MLP`, `Attention`, `LMHead`, and `DistributedNorm`.

```python
self.tt_ccl = TT_CCL(self.mesh_device)   # model.py line 49
```

### What TT_CCL wraps

`TT_CCL.__init__` allocates three families of semaphores, each double-buffered (two slots), for three cluster-axis indices (axis 0, axis 1, and no-axis = index 2):

| Semaphore family | Slot count per axis-index | Purpose |
|------------------|--------------------------|---------|
| `barrier_semaphore_handles` | 2 | Synchronisation barrier between sender and receiver in async collectives |
| `ag_semaphore_handles` | 2 x 2 (inner list of 2) | All-gather async coordination |
| `rs_semaphore_handles` | 2 x 3 (inner list of 3) | Reduce-scatter async coordination |

Each slot is a `ttnn.GlobalSemaphore` created over the full compute grid (`sub_device_crs` spans `(0,0)` to `(grid_x-1, grid_y-1)`).

The cycling helpers (`get_and_cycle_*`) advance an index modulo 2 on every call, implementing double-buffering so that successive collective operations do not race on the same semaphore.

### get_num_links

```python
def get_num_links(self, cluster_axis=None) -> int
```

Delegates to the module-level `get_num_links(mesh_device, cluster_axis)` function. Returns:
- `min(axis0_links, axis1_links)` when `cluster_axis=None`
- The axis-specific value when `cluster_axis` is 0 or 1

The per-device link counts are hard-coded in `link_dict`:

| Device | Axis-0 links | Axis-1 links |
|--------|-------------|-------------|
| N300   | 1           | 1           |
| T3K    | 1           | 1           |
| TG     | 4           | 3           |
| BHGLX  | 4           | 3           |

---

## 2. tt_all_gather

**Signature** (`ccl.py` lines 299-381):

```python
def tt_all_gather(
    input_tensor,
    mesh_device,
    tt_ccl,
    cluster_axis,      # required; 0 = vertical, 1 = horizontal, None = no-axis
    dim,               # required; dimension to gather along
    num_links=None,    # auto-detected from tt_ccl if None
    memory_config=None,
    sharded=False,
    topology=ttnn.Topology.Linear,
    dtype=ttnn.bfloat16,
    subdevice_id=None,
) -> ttnn.Tensor
```

**Behaviour:**

1. If `mesh_shape == [1, 1]` or `cluster_axis == 1` and there is only one device on that axis, returns the input unchanged (no-op guard).
2. If `not sharded`, moves input to `ttnn.DRAM_MEMORY_CONFIG` first.
3. If `input_tensor.dtype != dtype`, casts to `dtype` via `L1_MEMORY_CONFIG`; if `sharded` and `memory_config` is not None, re-shards.
4. Calls `ttnn.experimental.all_gather_async` with the `cluster_axis`-appropriate semaphore handles from `tt_ccl`.
5. Deallocates the input tensor before returning.

---

## 3. tt_all_reduce

**Signature** (`ccl.py` lines 136-296):

```python
def tt_all_reduce(
    input_tensor,
    mesh_device,
    tt_ccl,
    cluster_axis=0,
    dim=0,
    num_reduce_scatter_links=None,   # auto-detected if None
    num_all_gather_links=None,       # auto-detected if None
    topology=ttnn.Topology.Linear,
    memory_config=None,
    rs_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    sharded=False,
    dtype=ttnn.bfloat16,
    use_composite=False,
    chunks_per_sync=10,
    num_workers_per_link=2,
    subdevice_id=None,
) -> ttnn.Tensor
```

**Three code paths based on topology:**

### Path A — Single device or flat 1xN/Nx1 mesh (N300 / T3K)

Condition: `1 in list(mesh_device.shape)`

Uses `ttnn.experimental.reduce_scatter_minimal_async` only. This delivers each device its own partial reduction shard; the function returns that shard directly without a subsequent all-gather. Callers on these topologies therefore receive a **scatter-reduced** (not fully replicated) result.

Key parameters forwarded:
- `persistent_output_buffers=None`
- `multi_device_global_semaphore` from `tt_ccl.get_and_cycle_rs_semaphore_handles()`
- `barrier_semaphore` from `tt_ccl.get_and_cycle_barrier_semaphore_handle()`

### Path B — 2-D mesh, `use_composite=False` (TG default)

Condition: mesh is not flat and `use_composite=False`

1. `ttnn.experimental.all_gather_async` along `cluster_axis` — gathers partial activations from all devices.
2. `ttnn.experimental.fast_reduce_nc` — reduces along `dim` across the gathered copies.

For a full enumeration of the Galaxy-specific MLP CCL call sites (intermediate reduce-scatter, all-reduce, and all-gather around the element-wise multiply), see Section 6 below. Shard dim swaps are documented directly in Section 4.

### Path C — 2-D mesh, `use_composite=True` (TG large-dim variant)

Condition: mesh is not flat and `use_composite=True`

1. `reduce_scatter_minimal_async` along `cluster_axis` — scatter-reduces.
2. `all_gather_async` along `cluster_axis` — gathers the scatter results back.

The `use_composite=True` path is selected in `mlp.py` when `self.dim == 8192` (line 302).

---

## 4. Column-Parallel and Row-Parallel Linear Patterns

### MLP (`mlp.py`)

The MLP follows the SwiGLU pattern: two "up" projections (`w1` gate, `w3` up) followed by element-wise multiply and activation, then one "down" projection (`w2`).

**Column-parallel projections — w1 and w3:**

- Weight `w1` and `w3` are sharded along their output dimension (the hidden dimension).
- For T3K: `w1_dims = (-2, -1)` — the inner-most dimension maps to the mesh's column axis (`dim=-1` across `cluster_shape` axis 1).
- For Galaxy: `w1_dims = (-1, -2)` — the dimensions are swapped.
- No CCL is needed immediately after the `ttnn.linear` call.

**Row-parallel projection — w2:**

- Weight `w2` is sharded along its input dimension (hidden dimension).
- For T3K: `w2_dims = (-1, -2)`.
- For Galaxy: `w2_dims = (-2, -1)`.
- After the `w2` matmul, `tt_all_reduce` is called unconditionally (lines 290-311) with `cluster_axis=0`.

### Attention (`attention.py`)

**QKV projection — column-parallel:**

- The combined `wqkv` tensor is loaded with `ShardTensor2dMesh` using `dims=(3, 2)` for TG or `dims=(2, 3)` for T3K (`attention.py` lines 249-253).
- After the `ttnn.linear(x, self.wqkv, ...)` call, `tt_all_reduce` is called with `cluster_axis=1` to combine results across the column axis.

**Output projection `wo` — row-parallel:**

- `wo` is sharded with `dims=(2, 3)` (fused all-gather-matmul path or TG) or `dims=(3, 2)` (standard path).
- For the standard path (`use_fused_all_gather_matmul=False`, non-prefetcher T3K with Ring topology): attention heads are gathered via `all_gather_async`, then multiplied by `wo`.
- For the fused path (Ring topology, no prefetcher): `ttnn.experimental.all_gather_matmul_async` performs the gather and matmul in one fused op.

---

## 5. ShardTensor2dMesh and ShardTensorToMesh

### ShardTensor2dMesh

```python
ttnn.ShardTensor2dMesh(mesh_device, dims=(dim0, dim1), mesh_shape=cluster_shape)
```

Maps a 2-D grid of devices (shape `cluster_shape = (rows, cols)`) to two tensor dimensions simultaneously:
- `dim0` is split across mesh rows (axis 0).
- `dim1` is split across mesh columns (axis 1).
- Either dimension can be `None` (replication along that mesh axis).

Used in TT Transformers for all major weight tensors: `wqkv`, `wo`, `w1`, `w2`, `w3`, `current_pos`, `page_table`.

### ShardTensorToMesh

```python
ttnn.ShardTensorToMesh(mesh_device, dim=dim)
```

A 1-D sharding mapper: splits a single tensor dimension evenly across all devices in a flat order. Used in attention for the QKV bias tensors (`attention.py` line 182).

---

## 6. All-Reduce Placement in the MLP Forward Pass

The MLP forward (`mlp.py` lines 118-324) contains the following CCL call sites in order:

1. **(Galaxy only, large-dim or prefill)** `reduce_scatter_minimal_async` on `w1_out` and `w3_out` along `cluster_axis=1` (lines 181-211). Reduces the hidden-dimension partial outputs across the 8-chip column axis before the element-wise multiply.

2. **(Galaxy only, small-dim decode)** `tt_all_reduce` on `w1_out` and `w3_out` with `cluster_axis=1`, `num_all_gather_links=2` (lines 215-234). Comment in code: "hard codes to 2 links, so we do not get the dynamic link count from the CCL class to avoid any performance regressions."

3. **(Galaxy only, large-dim or prefill)** `all_gather_async` on `w2_in` along `cluster_axis=1` before the `w2` matmul (lines 252-266). Restores the full hidden dimension after the element-wise multiply.

4. **(All topologies)** `tt_all_reduce` on `w2_out` with `cluster_axis=0` (lines 290-311). This is the primary row-parallel all-reduce that combines partial `w2` outputs across all devices:
   - `dim = 0` when TG and `self.dim < 8192`, otherwise `dim = 3`
   - `sharded = (mode == Mode.DECODE)`
   - `use_composite = True` when `self.dim == 8192`
   - `dtype = self.args.ccl_dtype`

---

**Next:** [`symbiote_distributed_primitives.md`](./symbiote_distributed_primitives.md)
