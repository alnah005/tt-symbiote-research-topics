# `get_cos_sin_for_decode`: Host Operations and Data Flow

## Call Site in `_forward_decode_paged`

Line 2696 of `attention.py`:

```python
cos_ttnn, sin_ttnn = self._rotary_setup.get_cos_sin_for_decode(cache_position_tensor)
```

`cache_position_tensor` at this point is already a host-resident `torch.Tensor` (int32, shape `[B]`). It was constructed at lines 2663–2675 — either directly as `torch.tensor([cur_pos])` (Path 1) or by calling `ttnn.to_torch` on a device tensor and slicing (Path 2). In both cases, by the time line 2696 executes, `cache_position_tensor` is a Python/PyTorch object sitting in CPU memory.

This matters because `BailingRotarySetup.get_cos_sin_for_decode` has a conditional branch at the top (lines 425–435 of `rope.py`) that checks whether `position_ids` is a `ttnn.Tensor` and, if so, calls `ttnn.to_torch` to bring it to the host. Since `cache_position_tensor` is already a `torch.Tensor`, this device→host path is skipped.

---

## Full Data Flow Inside `get_cos_sin_for_decode`

The method is at lines 420–472 of `rope.py`. Annotated step by step:

### Step 1 — Input type check (lines 425–435)

```python
if isinstance(position_ids, ttnn.Tensor):
    if self.is_mesh_device:
        pos_torch = ttnn.to_torch(
            position_ids,
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )
        pos_torch = pos_torch[: position_ids.shape[0]]
    else:
        pos_torch = ttnn.to_torch(position_ids)
else:
    pos_torch = position_ids
```

When called from `_forward_decode_paged`, `position_ids` is a `torch.Tensor` so the `else` branch executes: `pos_torch = position_ids`. **No device→host read occurs.**

### Step 2 — Shape normalization (lines 437–441)

```python
if len(pos_torch.shape) == 2:
    pos_torch = pos_torch.squeeze(0)

batch_size = pos_torch.shape[0]
pos_indices = pos_torch.reshape(1, batch_size).to(torch.int32)
```

Pure host-side Python and PyTorch operations on a tensor that is already in CPU memory. No device interaction.

### Step 3 — Upload position index to device (lines 442–451)

```python
mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None

pos_ttnn = ttnn.from_torch(
    pos_indices,
    device=self.device,
    dtype=ttnn.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=mesh_mapper,
)
```

This is a **host→device upload** that fires on every decode step. `pos_indices` has shape `[1, B]` and dtype uint32 (4 bytes × B elements). On a T3K mesh with `ReplicateTensorToMesh`, the tensor is uploaded to all N=8 devices. The cost structure is identical to the `cur_pos_tt` upload discussed in [`cur_pos_roundtrip.md`](cur_pos_roundtrip.md): buffer allocation × 8 devices + 8 PCIe write transactions + topology metadata setup.

### Step 4 — On-device embedding lookup (lines 453–464)

```python
cos = ttnn.embedding(
    pos_ttnn,
    self.cos_cache_row_major,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
sin = ttnn.embedding(
    pos_ttnn,
    self.sin_cache_row_major,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

`cos_cache_row_major` and `sin_cache_row_major` are device-resident DRAM tensors of shape `[max_seq_len, head_dim]` (shape `[max_seq_len, D]` with D=128), stored in ROW_MAJOR layout (lines 364–379 of `rope.py`). They are created once at `BailingRotarySetup.__init__` time and never recreated.

`ttnn.embedding` dispatches to the device asynchronously. It reads `pos_ttnn` (shape `[1, B]`) and looks up the corresponding rows from `cos_cache_row_major`, producing a result of shape `[1, B, D]`. This is fully on-device with no host interaction.

### Step 5 — Reshape and transpose (lines 466–471)

```python
cos = ttnn.unsqueeze_to_4D(cos)  # [1, B, D] -> [1, 1, B, D]
sin = ttnn.unsqueeze_to_4D(sin)
cos = ttnn.transpose(cos, 1, 2)  # [1, 1, B, D] -> [1, B, 1, D]
sin = ttnn.transpose(sin, 1, 2)
```

Purely on-device ops. Output shape is `[1, B, 1, D]` as required by `ttnn.experimental.rotary_embedding_llama` in decode mode.

---

## The `ttnn.Tensor` Input Path: Guarded but Present

When `get_cos_sin_for_decode` is called with a `ttnn.Tensor` input — for example, if a caller passes `cache_position` as a persistent device tensor directly to this method without first passing it through the `_forward_decode_paged` position-resolution block — the code at lines 425–433 calls `ttnn.to_torch`. This is a synchronous device→host read with the same sync-flush cost as the `cur_pos_tt` Path 2 case.

In the current `_forward_decode_paged` call chain, this does not happen: `cache_position_tensor` is always a `torch.Tensor` when passed to `get_cos_sin_for_decode`. However, any refactoring that makes `get_cos_sin_for_decode` accept a device tensor (e.g., from the persistent `cur_pos_tt` proposed in `cur_pos_roundtrip.md`) would inadvertently activate this path and add a round-trip. The guard must be preserved or the refactored caller must ensure a host tensor is passed.

---

## Redundancy with `cur_pos_tt` Upload

Across one decode step, `_forward_decode_paged` uploads a position-related integer tensor to device twice:

1. Line 2678–2685: `cur_pos_tt = ttnn.from_torch(cache_position_tensor, dtype=int32, ...)` — shape `[B]`, used by `paged_update_on_device` and `paged_sdpa_decode`.
2. Inside `get_cos_sin_for_decode`, line 444–451: `pos_ttnn = ttnn.from_torch(pos_indices, dtype=uint32, ...)` — shape `[1, B]`, used by `ttnn.embedding`.

Both uploads carry the same logical data (the current token position for each batch slot) in slightly different shapes and dtypes (`[B]` int32 vs `[1, B]` uint32). They are separate uploads because:

- `paged_update_on_device` / `paged_sdpa_decode` expect a 1-D int32 vector of length B.
- `ttnn.embedding` expects a 2-D uint32 index tensor.

These could in principle be unified: maintain a single device-resident position tensor in a shape and dtype compatible with all three ops, converting via on-device ops rather than re-uploading from host. In practice, the embedding op's uint32 requirement differs from the paged kernel's int32 requirement, so on-device typecast would be needed. The combined saving would be one fewer `ttnn.from_torch` call per step plus the elimination of one set of 8 PCIe write transactions.

---

## Proposed Optimizations

### Optimization 1: Reuse `cur_pos_tt` for the embedding lookup

The persistent on-device position tensor approach is described in [`cur_pos_roundtrip.md`](cur_pos_roundtrip.md). At this call site, reusing that tensor requires an additional on-device typecast from int32 to uint32 before the `ttnn.embedding` call, since the embedding op requires a uint32 index tensor.

### Optimization 2: Pre-index cos/sin on device

Rather than uploading a position index and performing `ttnn.embedding` each step, an alternative is to maintain a pointer into the pre-computed cos/sin cache and slice on-device using `ttnn.slice`. The device-resident `cos_cache` (`self.cos_cache`, TILE_LAYOUT, shape `[1, 1, max_seq_len, D]`) supports slicing: `cos_cache[:, :, pos:pos+1, :]` yields the cos row for position `pos`. This avoids the upload entirely — the position advance is expressed as an on-device slice with a device-resident position index. However, `ttnn.slice` with a dynamic start index derived from a device tensor is not a standard TTNN API as of the time of writing; this approach requires verification against the current TTNN slice API.

### Optimization 3: Eliminate the upload if `pos_ttnn` can be computed from `cur_pos_tt`

If `cur_pos_tt` is already on device, `ttnn.reshape(ttnn.typecast(cur_pos_tt, ttnn.uint32), (1, batch_size))` produces `pos_ttnn` entirely on-device with two dispatched ops and no PCIe transfer. The host→device upload in `get_cos_sin_for_decode` is then eliminated. This is the most direct fix and requires only changes to `get_cos_sin_for_decode`'s `ttnn.Tensor` input branch.

---

**Next:** [Chapter 5 — QK Normalization: Cost Analysis and Distributed Alternatives](../chapter_05_qk_normalization/index.md)
