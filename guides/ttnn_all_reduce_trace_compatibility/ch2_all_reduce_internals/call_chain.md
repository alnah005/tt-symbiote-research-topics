# Call Chain — Python Entry Point to C++ Collective

This file traces every function call from the Python `TTNNLinearIColShardedWAllReduced.forward`
method through the C++ layers that implement `ttnn.all_reduce`, ending at the branch
point that selects either the composite path (all-gather + local sum) or the
non-composite path (reduce-scatter + all-gather). Understanding this chain is essential
to determining which semaphore state is live during trace capture and replay.

---

## 1. Python entry point — `TTNNLinearIColShardedWAllReduced.forward`

File:
`models/experimental/tt_symbiote/modules/linear.py`

```python
class TTNNLinearIColShardedWAllReduced(TTNNLinearIColShardedWRowSharded):
    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        # ... shape checks ...
        input_tensor = ttnn.reshape(input_tensor, input_shape)   # [1, 1, 32, hidden_dim]

        # Matmul: partial sum on each device
        tt_output = ttnn.linear(input_tensor, self.tt_weight,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)

        tt_output = ttnn.all_reduce(
            tt_output,
            num_links=1,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ...
```

The call carries **no** `GlobalSemaphore` arguments. The caller passes only the
ordinary tuning knobs (`num_links`, `topology`, `cluster_axis`, `memory_config`).

---

## 2. C++ `ExecuteAllReduce::invoke` — `all_reduce.cpp`

File:
`ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp`

```cpp
ttnn::Tensor ExecuteAllReduce::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<uint32_t> num_links,
    std::optional<tt::tt_fabric::Topology> topology) {

    // Flat-mesh short-circuit omitted for T3K (1x8 is already a line topology)
    tt::tt_fabric::Topology topology_ =
        ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);

    return ::ttnn::experimental::all_reduce_async(
        input_tensor,
        cluster_axis,
        *mesh_device,
        std::nullopt,   // barrier_semaphores
        std::nullopt,   // rs_global_semaphores
        std::nullopt,   // ag_global_semaphores
        ttnn::operations::reduction::ReduceType::Sum,
        memory_config,
        topology_,
        num_links,
        subdevice_id);
}
```

Two points are critical:

- `get_usable_topology` and `convert_2d_to_1d_topology` resolve the topology string
  from the Python call site (`Ring`) to the concrete `tt::tt_fabric::Topology` value
  that C++ uses. On a 1×8 T3K mesh, `Ring` on cluster axis 1 resolves to
  `Topology::Ring` (1-D).
- All three semaphore optional arguments are `std::nullopt`. The synchronous
  `ExecuteAllReduce` layer has no mechanism to accept or forward semaphore handles; it
  simply never passes them.

---

## 3. Delegation to `ExecuteAllReduceAsync::invoke` (cluster\_axis overload)

File:
`ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp`

The `cluster_axis` overload of `ExecuteAllReduceAsync::invoke` (lines 258–399 in the
file) receives the three `std::optional<std::vector<GlobalSemaphore>>` arguments, all
`std::nullopt`. It then performs the following setup before the path-selection branch:

> **Implementation note — `barrier_semaphores` size constraint:** In the non-composite
> path, the async `all_gather` branch is guarded by
> `ag_global_semaphores.has_value() && barrier_semaphores.has_value()`. Immediately
> inside that guard (line 362), the code asserts:
> ```cpp
> TT_FATAL(barrier_semaphores.value().size() == 2, "Barrier semaphores must be of size 2");
> ```
> The `barrier_semaphores` vector must contain exactly **2** elements — `[0]` is
> forwarded to `reduce_scatter_minimal_async` and `[1]` is forwarded to
> `all_gather_async`. A caller supplying a vector of any other size will trigger a
> runtime assertion failure.

```cpp
tt::tt_fabric::Topology topology_ =
    ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());

uint32_t num_devices =
    ::ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
uint32_t dim = detail::finding_scatter_dim(input_tensor, num_devices);

auto composite_dim = (dim == padded_tensor.padded_shape().size()) ? 0 : dim;
bool composite_all_gather =
    composite_common::use_composite_all_gather(padded_tensor, composite_dim,
                                               out_memory_config);
bool composite_reduce_scatter =
    composite_common::use_composite_reduce_scatter(padded_tensor, composite_dim,
                                                   cluster_axis);
const bool composite_for_2d_mesh =
    tt::tt_fabric::GetFabricConfig() == tt::tt_fabric::FabricConfig::FABRIC_2D &&
    detail::is_true_2d_mesh(input_tensor, topology_);
```

---

## 4. Path-selection predicates

### 4.1 `detail::finding_scatter_dim`

Scans the tensor's padded shape from the innermost dimension outward and returns the
first dimension whose size (in tiles for TILE layout) is divisible by `num_devices`.
For a tensor of padded shape `[1, 1, 32, hidden_dim]` on an 8-device ring,
`hidden_dim / tile_width` must be divisible by 8. If `hidden_dim` is a typical
multiple of 8 tiles (e.g., 512, 1024), `dim` resolves to dimension 3 (the last
dimension). If no dimension divides evenly, `dim` is set to `rank` (past-the-end),
which forces `dim != composite_dim` and therefore triggers the composite path.

### 4.2 `composite_common::use_composite_all_gather`

File: `ttnn/cpp/ttnn/operations/experimental/ccl/composite_common.cpp`

Returns `true` (use composite) under these conditions:

- Active fabric config is `FABRIC_2D` and mesh is truly 2D (both axes $> 1$).
- Input layout is `ROW_MAJOR`.
- Input is tiled and the gather dimension is not tile-aligned (i.e.,
  `input_shape[-2] % tile_height != 0` for dim $= rank-2$, or
  `input_shape[-1] % tile_width != 0` for dim $= rank-1$).

For a DRAM-interleaved TILE-layout tensor with `hidden_dim` aligned to 32-element
tiles, this returns `false` on a 1-D fabric T3K.

### 4.3 `composite_common::use_composite_reduce_scatter`

Returns `false` (do NOT use composite) immediately when:

- The scatter dimension size is not evenly divisible by `num_devices`
  (`input_shape[scatter_dim] % num_devices != 0`). This is the leading early-exit;
  the predicate never proceeds to the tile-alignment check in this case.

If the dimension IS evenly divisible, the function then returns `true` (use
composite) when:

- The input is `ROW_MAJOR`, **or**
- The resulting per-device slice in that dimension is not tile-aligned (i.e.,
  `output_shape[scatter_dim] % tile_width != 0` for the innermost dimension, or
  `% tile_height != 0` for the second-to-last dimension).

Otherwise it returns `false`.

For typical decode shapes on T3K, this returns `false` as long as `hidden_dim / 8`
is tile-aligned.

### 4.4 `detail::is_true_2d_mesh`

Returns `true` only when both mesh axes have more than one device. A T3K (1×8) fails
this check because `mesh_shape[0] == 1`.

### 4.5 Combined condition

```cpp
if (composite_all_gather || composite_reduce_scatter
    || (dim != composite_dim) || composite_for_2d_mesh) {
    // composite path: all-gather + local reduce
    ...
}
// non-composite path: reduce-scatter + all-gather
```

For a typical T3K decode tensor `[1, 1, 32, H]` where $H$ is a multiple of
$8 \times 32 = 256$ elements (tile-aligned), on a 1-D fabric ring:

| Predicate | Value | Reason |
|---|---|---|
| `composite_all_gather` | `false` | tile-aligned, 1-D fabric |
| `composite_reduce_scatter` | `false` | per-device slice is tile-aligned |
| `dim != composite_dim` | `false` | scatter dim 3 == composite_dim 3 |
| `composite_for_2d_mesh` | `false` | mesh is 1×8 |

All four predicates are `false`, so execution falls through to the
**non-composite reduce-scatter + all-gather path**.

---

## 5. Path taken on T3K for typical decode tensors

On a T3K (1×8 mesh, cluster axis 1) with tensors of shape `[1, 1, 32, hidden_dim]`
in DRAM interleaved TILE layout:

- `hidden_dim` must be a multiple of $8 \times 32 = 256$ for the non-composite path
  to be selected. This is satisfied by all standard model hidden dimensions (e.g., 4096,
  8192, 14336).
- The non-composite path invokes `ttnn.reduce_scatter` followed by `ttnn.all_gather`
  (the synchronous forms), because `rs_global_semaphores` and `barrier_semaphores` are
  both `std::nullopt`.
- No persistent `GlobalSemaphore` objects are created or consulted.

> **Note:** If a model is run with a 2D fabric configuration, or if the hidden dimension
> is not tile-aligned, the composite path is selected instead. The composite path has
> different trace compatibility properties — see [`composite_path.md`](./composite_path.md).

---

**Next:** [`composite_path.md`](./composite_path.md)
