# Downstream Op Constraints for the Pre-Allocated cos/sin Buffer

The cos/sin buffer's shape, dtype, layout, and memory config are not arbitrary — they are derived by working backwards from what `ttnn.experimental.rotary_embedding` requires, through any intermediate ops (`ttnn.unsqueeze`, etc.) in `TTNNQwen3FullAttention.forward`. By the end of this file you will understand why each attribute of the pre-allocated buffer is what it is, and what would happen if any attribute were chosen incorrectly.

---

## Section 1: Shape Requirement

`ttnn.experimental.rotary_embedding` expects cos and sin tensors shaped `[1, 1, seq_len, rotary_dim]`. At decode time, `seq_len=1`, so the expected shape at the op call site is `[1, 1, 1, rotary_dim]`.

The pre-allocated buffer must match the shape of cos/sin as they arrive in `TTNNQwen3FullAttention.forward`, because `ttnn.copy(cos, self._cos_replicated)` requires the source and destination to have identical shapes. The shape of the incoming cos/sin depends on how the upstream code (the caller of `TTNNQwen3FullAttention.forward`) prepares them.

> **TODO:** Confirm the shape of cos/sin as they arrive in `TTNNQwen3FullAttention.forward`. The possibilities are: (a) `[1, 1, 1, rotary_dim]` (4D) if the upstream already includes all leading dimensions; or (b) `[1, 1, rotary_dim]` (3D) if the upstream omits one leading dimension. In case (b), if `TTNNQwen3FullAttention.forward` calls `ttnn.unsqueeze(self._cos_replicated)` inside the trace bracket to promote the 3D buffer to 4D before the rotary op, that `ttnn.unsqueeze` call would allocate a new device buffer and be trace-unsafe. The correct design for case (b) is to pre-allocate the buffer in 4D (`[1, 1, 1, rotary_dim]`) and ensure `TTNNQwen3FullAttention.forward` does not call any reallocating op on the pre-allocated buffer inside the trace bracket — use `ttnn.reshape` (view, no allocation) if a shape adjustment is unavoidable, not `ttnn.unsqueeze`. Adjust the quick-reference table in [`index.md`](./index.md) accordingly.

For Qwen3, `rotary_dim = 64` (tile-aligned). The buffer for decode is effectively a single row of 64 BF16 values per device — 128 bytes of payload per device before tile padding.

---

## Section 2: Layout Requirement

`ttnn.experimental.rotary_embedding` is a compute op that requires `TILE_LAYOUT` for its inputs. If the pre-allocated buffer is in `ROW_MAJOR_LAYOUT`, TTNN will trigger an implicit layout conversion inside the traced region. This layout conversion allocates a new intermediate buffer — it is not trace-safe.

Therefore, the pre-allocated buffer must be in `TILE_LAYOUT` from the moment it is created in `move_weights_to_device_impl`.

> **Note:** `rotary_dim = 64` is already tile-aligned (64 ÷ 32 = 2 tiles), so `TILE_LAYOUT` does not require padding along the last dimension for the Qwen3 case. The second-to-last dimension (seq_len = 1) is not a multiple of 32 and will be padded to 32 on-device, making the effective stored shape `[1, 1, 32, 64]`. Downstream ops that receive this buffer must tolerate the padded shape — `ttnn.experimental.rotary_embedding` does. For non-tile-aligned `rotary_dim` values, see the cross-reference to the separate research topic in [`../ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md`](../ch5_warmup_guard/non_tile_aligned_rotary_dim_interaction.md).

---

## Section 3: Memory Config

Use `ttnn.DRAM_MEMORY_CONFIG`. The buffer lives for the entire decode session — hundreds to thousands of steps — so it must not consume L1, which is reserved for per-step intermediate activations. DRAM also accommodates the prefill case where `seq_len > 1` and the buffer would be larger, though this guide focuses on the decode trace path.

A decode-step cos/sin buffer of shape `[1, 1, 1, 64]` in BF16 occupies 128 bytes of payload per device. Even padded to a tile (`[1, 1, 32, 64]`), the on-device footprint is 4,096 bytes per device — well within DRAM capacity. L1 capacity on Wormhole is approximately 1.5 MB per core; while the buffer could technically fit in L1, occupying L1 for a persistent buffer reduces the L1 budget available to the compute kernels that run at every step, which is undesirable.

---

## Section 4: dtype

`TTNNQwen3FullAttention` operates in BF16 throughout its forward pass. `ttnn.experimental.rotary_embedding` expects BF16 cos/sin inputs. The pre-allocated buffer must use `dtype=ttnn.bfloat16`.

If the upstream source of cos/sin values (e.g., `TTNNRotaryPositionEmbedding`) produces float32 tensors, those tensors must be cast to BF16 before being passed to `TTNNQwen3FullAttention.forward`, or the cast must be applied to the incoming `cos` argument before the `ttnn.copy` call. Placing a dtype cast inside the trace bracket allocates a new intermediate buffer and is trace-unsafe. The cast must therefore happen before the trace bracket or the upstream source must already produce BF16.

---

## Section 5: Shape Transformations in `forward`

The key question is: what transformations (if any) are applied to cos/sin between the `forward` argument and the `ttnn.experimental.rotary_embedding` call site? Each such transformation must be analyzed for trace-safety:

- An in-place op on an existing buffer (e.g., `ttnn.reshape` that returns a view into the same storage) is trace-safe if no new buffer is allocated.
- A copy-on-write op that allocates a new buffer (e.g., `ttnn.to_layout`, `ttnn.typecast`, `ttnn.clone`) is trace-unsafe inside the trace bracket.

If `_ensure_replicated` is the only transformation applied to cos/sin before they reach `ttnn.experimental.rotary_embedding`, then removing it and replacing it with `ttnn.copy` into a pre-allocated `TILE_LAYOUT` BF16 buffer satisfies all downstream constraints with no remaining trace-unsafe ops.

> **Key Finding:** The pre-allocated cos/sin buffer must be in `TILE_LAYOUT`, `DRAM_MEMORY_CONFIG`, `bfloat16`, and `ReplicateTensorToMesh` mapping. The shape is `[1, 1, 1, rotary_dim]` (decode; seq_len=1) if cos/sin arrive at `TTNNQwen3FullAttention.forward` as 4D tensors, or if any internal `ttnn.unsqueeze` is replaced with a non-allocating alternative — see the Section 1 TODO for the unresolved shape question. These constraints follow directly from `ttnn.experimental.rotary_embedding`'s layout requirement, the persistent-buffer DRAM rule, and the TP replication requirement.
