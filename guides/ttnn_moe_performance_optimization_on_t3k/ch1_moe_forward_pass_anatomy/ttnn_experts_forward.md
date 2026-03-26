# TTNNExperts.forward — Step-by-Step Walkthrough

## Context

This file addresses:
- **Q1** — What is the per-op breakdown of a single MoE forward pass?
- **Q3** — How is sparsity expressed and exploited in the expert compute kernels?
- **Q4** — What are the shapes and memory layouts at each stage of expert dispatch and combine?

Source range: `moe.py:L1027–L1343`

---

## Class Overview

`TTNNExperts` owns the three expert weight projections (`tt_w1_proj`, `tt_w3_proj`, `tt_w2_proj`), the routing metadata tensors (`expert_mapping_tensors`, `remap_topk_mask`), and the two program configs that describe the sparse matmul tiling strategy. Two global constants govern data alignment throughout this class:

```python
TOPK_MIN_WIDTH = 64       # moe.py:L51
SPARSITY_BLOCK_SIZE = 32  # moe.py:L52
```

`SPARSITY_BLOCK_SIZE = 32` is the granularity of sparsity: tokens are grouped into blocks of 32, and the sparse matmul kernel decides on a per-block basis whether to compute or skip. `TOPK_MIN_WIDTH = 64` is a minimum column width constraint for the sparse output tiles, derived from the hardware tile size. Both values are architectural constants tied to the Wormhole SIMD tile dimensions (32×32 native tile, 64-wide minimum matmul output for efficient packing).

---

## Step 0: Input Shape and Geometry (moe.py:L1070–L1078)

```python
batch_size_per_device = x.shape[0]
seq_len = x.shape[2]
batch_size = batch_size_per_device * self.num_dispatch_devices
original_num_tokens = batch_size_per_device * seq_len
```

`x` arrives as `[batch_size_per_device, 1, seq_len, hidden_size]` — the `unsqueeze` inserted in `TTNNMoE.forward` places `seq_len` at dimension 2. `batch_size` is the global token count across all dispatch devices; `original_num_tokens` is the per-device token count and is saved to trim padding at the end.

---

## Step 1: Typecast Expert Indices (moe.py:L1080–L1085)

```python
if topk_experts_indices.dtype != ttnn.uint16:
    topk_experts_indices = ttnn.to_layout(topk_experts_indices, ttnn.TILE_LAYOUT, ...)
    topk_experts_indices = ttnn.typecast(topk_experts_indices, ttnn.uint16)
```

The routing indices from `TTNNMoE.forward` may arrive in a higher-precision integer format. The `all_to_all_dispatch` kernel requires `uint16` indices. The conditional avoids the layout conversion overhead when the indices are already the correct type.

---

## Step 2: Pad Tokens to SPARSITY_BLOCK_SIZE Boundary (moe.py:L1087–L1099)

```python
num_tokens = original_num_tokens
pad_amount = 0
if num_tokens % SPARSITY_BLOCK_SIZE != 0:
    pad_amount = SPARSITY_BLOCK_SIZE - (num_tokens % SPARSITY_BLOCK_SIZE)
    num_tokens += pad_amount
    x = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)
    topk_experts_indices = ttnn.pad(topk_experts_indices, padding=((0, pad_amount), (0, 0)), value=0)
    topk_experts_weights = ttnn.pad(topk_experts_weights, padding=((0, pad_amount), (0, 0)), value=0.0)
    seq_len = num_tokens // batch_size_per_device
```

`SPARSITY_BLOCK_SIZE = 32` is a hard alignment requirement for the sparse matmul kernel: it groups tokens into 32-token blocks and uses a per-block sparsity mask to skip blocks with no active expert assignment. If the token count is not a multiple of 32, zero padding is appended. The padding is zero-valued in both the data and weight tensors, so padded tokens contribute nothing to the output. The actual token count is recovered at the end via `ttnn.slice`.

Padding is applied consistently to all three input tensors — activations, indices, and weights — to keep their leading dimensions aligned.

---

## Step 3: Prepare ROW_MAJOR Tensors for Dispatch (moe.py:L1101–L1109)

```python
x = ttnn.typecast(x, ttnn.bfloat16)
x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, self.hidden_size))
topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
topk_experts_indices_rm = ttnn.reshape(topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, self.num_experts_per_tok))
```

`all_to_all_dispatch` requires `ROW_MAJOR_LAYOUT` inputs. The activations are also downcast to `bfloat16` at this point — they may have been held in `float32` from a previous op. The reshape confirms the 4-D layout contract: `[batch_per_device, 1, seq_len, feature_dim]`.

---

## Step 4: All-to-All Dispatch (moe.py:L1111–L1117)

```python
all_to_all_dispatch_output, all_to_all_dispatch_metadata = ttnn.all_to_all_dispatch(
    x_rm,
    topk_experts_indices_rm,
    self.expert_mapping_tensors,
    cluster_axis=1,
)
```

`all_to_all_dispatch` is the primary inter-device routing operation. It inspects `topk_experts_indices_rm` and routes each token to the device(s) responsible for its selected experts. `expert_mapping_tensors` is a precomputed mapping from expert index to device index; it is static for a given mesh configuration and is set up in `move_weights_to_device_impl`.

`cluster_axis=1` specifies that dispatch communicates across the second mesh dimension — the same axis used by the subsequent reduce-scatter. This keeps the routing and reduction on the same physical Ethernet links.

The function returns two values:
- `all_to_all_dispatch_output`: the reordered activation tensor, with tokens co-located with their assigned expert device.
- `all_to_all_dispatch_metadata`: a descriptor that `all_to_all_combine` uses at the end of the pipeline to reverse the routing and return results to originating devices.

---

## Step 5: Reshape for Expert Compute (moe.py:L1119–L1124)

```python
post_dispatch = ttnn.reshape(all_to_all_dispatch_output, shape=(1, 1, batch_size * seq_len, self.hidden_size))
post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)
num_tokens = batch_size * seq_len
```

The dispatched output is reshaped to flatten all tokens from all dispatch devices into a single sequence dimension: `[1, 1, batch_size * seq_len, hidden_size]`. The conversion to `TILE_LAYOUT` is required for the sparse matmul kernels.

---

## Step 6: Generate Sparsity Tensor (moe.py:L1126–L1136)

```python
remap_topk_mask_expanded = ttnn.repeat(self.remap_topk_mask, ttnn.Shape((1, batch_size_per_device, 1, 1)))
_, sparsity_t = ttnn.moe_expert_token_remap(
    remap_topk_mask_expanded,
    self.expert_mapping_tensors,
    all_to_all_dispatch_metadata,
    reduction_size=SPARSITY_BLOCK_SIZE,
)
```

`ttnn.moe_expert_token_remap` generates the sparsity tensor `sparsity_t` that the sparse matmul kernels consume. It uses the post-dispatch metadata and the `remap_topk_mask` (a boolean mask identifying which expert-token pairs are active after dispatch) to build a per-block activity map at `SPARSITY_BLOCK_SIZE=32` granularity.

`reduction_size=SPARSITY_BLOCK_SIZE` sets the block granularity; the resulting `sparsity_t` indicates which blocks contain active tokens, and the sparse matmul skips empty blocks entirely.

The `remap_topk_mask` is first expanded along the batch dimension via `ttnn.repeat` to match the replicated batch layout.

---

## Step 7: Reshape for Sparse Matmul (moe.py:L1138–L1140)

```python
num_sparse_blocks = num_tokens // SPARSITY_BLOCK_SIZE
x_sparse = ttnn.reshape(post_dispatch, shape=(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, self.hidden_size))
```

The token sequence is now explicitly segmented into `SPARSITY_BLOCK_SIZE`-token blocks. This 4-D layout `[1, num_sparse_blocks, 32, hidden_size]` directly maps to the sparse matmul's inner loop: the kernel iterates over `num_sparse_blocks` blocks and consults `sparsity_t` to decide whether to process each block.

---

## Step 8: Three Sparse Matmul Calls — Gate-Up and Down Projections (moe.py:L1142–L1172)

### Gate projection (w1) and up projection (w3) — parallel calls

```python
w1_out = ttnn.sparse_matmul(x_sparse, self.tt_w1_proj, sparsity=sparsity_t,
    output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
    program_config=self._gate_up_program_config,
    compute_kernel_config=self._expert_compute_cfg,
    is_input_a_sparse=False, is_input_b_sparse=True)

w3_out = ttnn.sparse_matmul(x_sparse, self.tt_w3_proj, sparsity=sparsity_t,
    output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
    program_config=self._gate_up_program_config,
    compute_kernel_config=self._expert_compute_cfg,
    is_input_a_sparse=False, is_input_b_sparse=True)
```

Both calls share the same `x_sparse` activations, `sparsity_t` mask, and `_gate_up_program_config`. The `is_input_a_sparse=False, is_input_b_sparse=True` flags tell the kernel that the activation tensor is dense (all blocks may be active) but the weight matrix is sparse (only columns corresponding to active experts are stored). The output tile size `[SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]` = `[32, 32]` matches the hardware's native tile dimensions.

`_gate_up_program_config` (computed in `move_weights_to_device_impl`):
- `in0_block_w = min(4, hidden_tiles)` — read up to 4 tiles of the input row at a time
- `per_core_M = 1` — each core processes one output tile row
- Uses `MatmulMultiCoreReuseMultiCast1DProgramConfig` which multicasts input tiles across a 1-D core grid

The compute kernel config for all three matmuls:

```python
self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

Note: the expert compute uses **HiFi2** (not HiFi4 as used for the gate linear). HiFi2 provides lower precision MACs and is faster. For expert projections, which are large weight matrices where accumulated error is less sensitive than for the routing decision, this is the correct precision-performance tradeoff.

### SiLU activation and element-wise multiply (moe.py:L1162–L1168)

```python
w1_activated = ttnn.silu(w1_out)
ttnn.deallocate(w1_out)
intermediate = ttnn.mul(w1_activated, w3_out)
ttnn.deallocate(w1_activated)
ttnn.deallocate(w3_out)
```

This is the gated linear unit (GLU) activation used in DeepSeek V3's expert FFN: `SiLU(w1 @ x) * (w3 @ x)`. The three `deallocate` calls are explicit DRAM reclamation — `w1_out`, `w1_activated`, and `w3_out` are large intermediate tensors that are not needed after the element-wise multiply completes.

```python
intermediate = ttnn.squeeze(intermediate, 0)
intermediate = ttnn.squeeze(intermediate, 1)
```

The two squeeze ops remove the leading `[1, num_sparse_blocks]` dimensions that were added for the block-sparse layout, leaving `intermediate` as `[SPARSITY_BLOCK_SIZE_total, intermediate_size]` for the down projection.

### Down projection (w2)

```python
expert_output = ttnn.sparse_matmul(intermediate, self.tt_w2_proj, sparsity=sparsity_t,
    output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
    program_config=self._down_program_config,
    compute_kernel_config=self._expert_compute_cfg,
    is_input_a_sparse=True, is_input_b_sparse=False)
ttnn.deallocate(intermediate)
```

The down projection differs in two ways from the gate/up projections:
1. `is_input_a_sparse=True, is_input_b_sparse=False` — now the activation (`intermediate`) is sparse (has empty blocks) and the weight matrix is dense. The sparsity mask skips blocks in the activation.
2. `_down_program_config` uses `in0_block_w=min(4, intermediate_tiles)` — `intermediate_size` may differ from `hidden_size` (for DeepSeek V3 it is larger), so the tile block width is recomputed independently.

---

## Step 9: Prepare for All-to-All Combine (moe.py:L1178–L1189)

```python
expert_output = ttnn.permute(expert_output, (1, 0, 2, 3))
expert_output = ttnn.reshape(expert_output, shape=(1, self.num_experts_per_device, num_tokens, self.hidden_size))
expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)
expert_output = ttnn.reshape(expert_output, shape=(self.num_experts_per_device, batch_size, seq_len, self.hidden_size))
```

After the down projection, the output tensor's layout reflects the expert-block organization. The `permute` and two `reshape` ops convert it to the layout expected by `all_to_all_combine`: `[num_experts_per_device, batch_size, seq_len, hidden_size]`. The `ROW_MAJOR_LAYOUT` conversion is required by the CCL operation, mirroring the dispatch-side conversion in Step 3.

---

## Step 10: All-to-All Combine (moe.py:L1191–L1198)

```python
combined_output = ttnn.all_to_all_combine(
    expert_output,
    all_to_all_dispatch_metadata,
    self.expert_mapping_tensors,
    cluster_axis=1,
)
```

`all_to_all_combine` is the inverse of `all_to_all_dispatch`. It uses the metadata saved from Step 4 to route each expert's computed output back to the device that originated the corresponding token. The result `combined_output` is shaped to have all top-k expert outputs for each token co-located on the token's originating device.

`cluster_axis=1` is the same axis used for dispatch, ensuring the combine traffic follows the same physical links.

---

## Step 11: Reshape Combined Output (moe.py:L1200–L1204)

```python
combined_output = ttnn.reshape(combined_output, shape=(self.num_experts_per_tok, 1, batch_size_per_device * seq_len, self.hidden_size))
combined_output = ttnn.to_layout(combined_output, ttnn.TILE_LAYOUT)
```

After combine, the output is reshaped to `[num_experts_per_tok, 1, tokens_per_device, hidden_size]`. The leading dimension now indexes over the top-k expert outputs for each token rather than over devices. The conversion to `TILE_LAYOUT` prepares for the subsequent element-wise multiply and sum.

---

## Step 12: Apply Expert Weights via Repeat + Permute (moe.py:L1206–L1220)

```python
topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 0)
topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 0)
topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, repeat_dims=(self.hidden_size, 1, 1, 1))
topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
topk_experts_weights_tile = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
ttnn.deallocate(topk_experts_weights_rm)

weighted_output = ttnn.mul(combined_output, topk_experts_weights_tile)
final_output = ttnn.sum(weighted_output, dim=0, keepdim=True)
```

The broadcast strategy:
1. Two `unsqueeze` calls add leading dimensions: `[1, 1, tokens, num_experts_per_tok]`
2. `ttnn.repeat(repeat_dims=(self.hidden_size, 1, 1, 1))` repeats along the first dimension, producing `[hidden_size, 1, tokens, num_experts_per_tok]`
3. `ttnn.permute((3, 1, 2, 0))` reorders to `[num_experts_per_tok, 1, tokens, hidden_size]`

The `ttnn.sum(..., dim=0)` then collapses the `num_experts_per_tok` dimension, summing the weighted contributions from all selected experts for each token. `keepdim=True` preserves the dimension as size 1, yielding `[1, 1, tokens, hidden_size]`.

The repeat-permute pattern is a shape-manipulation workaround for the absence of a broadcasting `outer_mul` op in TTNN. It is a known potential optimization target.

---

## Step 13: Remove Padding (moe.py:L1222–L1225)

```python
if pad_amount > 0:
    final_output = ttnn.slice(final_output, (0, 0, 0, 0), (1, 1, original_num_tokens, self.hidden_size))
```

If tokens were padded in Step 2, the padding tokens are trimmed here via `ttnn.slice`. Only the `original_num_tokens` real tokens are returned. The slice is a no-copy view when TTNN can express it as a stride offset.

---

## Summary: Op Sequence in TTNNExperts.forward

```
typecast indices (→ uint16)
[optional] pad to SPARSITY_BLOCK_SIZE=32 boundary
typecast activations (→ bfloat16)
to_layout → ROW_MAJOR (activations + indices)
all_to_all_dispatch          (cluster_axis=1)
reshape + to_layout TILE     (→ [1,1,total_tokens,hidden])
moe_expert_token_remap       (→ sparsity_t, reduction_size=32)
reshape                      (→ [1, num_blocks, 32, hidden])
sparse_matmul(x, w1_proj)    (is_b_sparse=True, HiFi2)
sparse_matmul(x, w3_proj)    (is_b_sparse=True, HiFi2)
silu(w1_out) * w3_out        (→ intermediate)
deallocate w1_out, w1_activated, w3_out
squeeze × 2
sparse_matmul(intermediate, w2_proj)  (is_a_sparse=True, HiFi2)
deallocate intermediate
permute + reshape × 2 + to_layout ROW_MAJOR
all_to_all_combine           (cluster_axis=1)
reshape + to_layout TILE
weights: unsqueeze × 2 + repeat + permute + to_layout TILE
ttnn.mul + ttnn.sum(dim=0)
[optional] ttnn.slice        (remove padding)
```

---

**Next:** [`cpu_fallback_paths.md`](./cpu_fallback_paths.md)
