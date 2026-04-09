# MoE Expert Pipeline Fusion

## Current Implementation

The MoE expert pipeline in TT-Symbiote spans two key classes in `modules/moe.py`:

- **`TTNNExperts`** (line 1031): the high-performance sparse-matmul path used for DeepSeek V3 / Qwen3 on T3K
- **`TTNNGlm4MoeExpertLayers`** (line 490): the per-expert loop path used for GLM-4

### TTNNExperts: Sparse-Matmul Path

The core compute in `TTNNExperts.forward()` (line 1163) executes the following sequence for each token batch:

```
# Step 1: All-to-all dispatch (tokens -> expert devices)
all_to_all_dispatch_output, metadata = ttnn.all_to_all_dispatch(x_rm, ...)

# Step 2: Generate sparsity tensor
_, sparsity_t = ttnn.moe_expert_token_remap(...)

# Step 3: Gate projection (sparse_matmul #1)
w1_out = ttnn.sparse_matmul(x_sparse, self.tt_w1_proj, sparsity=sparsity_t, ...)

# Step 4: Up projection (sparse_matmul #2)
w3_out = ttnn.sparse_matmul(x_sparse, self.tt_w3_proj, sparsity=sparsity_t, ...)

# Step 5: SiLU activation on gate output
w1_activated = ttnn.silu(w1_out)

# Step 6: Elementwise multiply gate * up
intermediate = ttnn.mul(w1_activated, w3_out)

# Step 7: Down projection (sparse_matmul #3)
expert_output = ttnn.sparse_matmul(intermediate, self.tt_w2_proj, sparsity=sparsity_t, ...)

# Step 8: All-to-all combine (expert devices -> tokens)
combined_output = ttnn.all_to_all_combine(expert_output, ...)

# Step 9: Weight and sum
weighted_output = ttnn.mul(combined_output, topk_experts_weights_tile)
final_output = ttnn.sum(weighted_output, dim=0, keepdim=True)
```

### TTNNGlm4MoeExpertLayers: Per-Expert Loop Path

The fallback path iterates over activated experts in Python (line 532):

```python
def forward(self, current_state, expert_idx):
    gate = self.gate_layers[expert_idx](current_state)    # TTNNLinearSilu
    up = self.up_layers[expert_idx](current_state)         # TTNNLinear
    current_hidden_states = gate * up                      # elementwise mul
    current_hidden_states = self.down_layers[expert_idx](current_hidden_states)  # TTNNLinear
    return current_hidden_states
```

This path is called from `Glm4MoeNaiveMoeHybrid.forward()` (line 588) inside a `for expert_idx in expert_hit` loop, which prevents trace capture entirely.

## Performance Bottleneck Analysis

### DRAM Traffic in the Sparse-Matmul Path

For a model with `hidden_size=4096`, `intermediate_size=1408`, and `num_tokens=1024` (after padding to SPARSITY_BLOCK_SIZE=32 boundaries):

| Step | Op | Output Size (BF16) | DRAM Write | DRAM Read (next op) |
|------|----|--------------------|-----------|-------------------|
| 3 | `sparse_matmul(w1)` | 1024 x 1408 = 2.75 MB | Yes | Yes (step 5) |
| 4 | `sparse_matmul(w3)` | 1024 x 1408 = 2.75 MB | Yes | Yes (step 6) |
| 5 | `silu(w1_out)` | 1024 x 1408 = 2.75 MB | Yes | Yes (step 6) |
| 6 | `mul(w1_act, w3)` | 1024 x 1408 = 2.75 MB | Yes | Yes (step 7) |
| 7 | `sparse_matmul(w2)` | 1024 x 4096 = 8.0 MB | Yes | Yes (step 8) |

**Total intermediate DRAM traffic: ~19 MB written + ~19 MB read = ~38 MB per expert batch.**

Steps 3 through 7 form a tight data-dependency chain. Each intermediate result is written to DRAM and immediately read back by the next op. This is pure overhead -- the data never needs to leave the compute cores.

### Host-Side Overhead in the Per-Expert Path

The `TTNNGlm4MoeExpertLayers` path adds:
- Python `for` loop over up to 128 experts (GLM-4-100B has 128 routed experts, 8 active per token)
- Per-expert `torch.where` to find token indices
- Host-device data transfer for each expert's input/output
- `@disable_trace` decorator prevents TTNN trace capture

## Fusion Opportunity: Fused Expert Compute Kernel

### What to Fuse

Fuse steps 3-7 into a single TT-Lang kernel: `fused_moe_expert_compute`.

```
Input:  x_sparse      [num_sparse_blocks, BLOCK_SIZE, hidden_size]
        sparsity_t     [num_experts_per_device, num_sparse_blocks]
        w1_proj        [num_experts_per_device, hidden_size, intermediate_size]  (gate)
        w3_proj        [num_experts_per_device, hidden_size, intermediate_size]  (up)
        w2_proj        [num_experts_per_device, intermediate_size, hidden_size]  (down)

Output: expert_output  [num_experts_per_device, num_tokens, hidden_size]
```

### TT-Lang DFB Design

Following the DFB patterns from [Chapter 1](../ch1_programming_model/index.md):

```python
@ttl.operation(grid="auto")
def fused_moe_expert_compute(
    x_sparse: ttnn.Tensor,       # [num_sparse_blocks, BLOCK_SIZE, hidden_size]
    w1_proj: ttnn.Tensor,        # [num_experts, hidden_size, intermediate_size]
    w3_proj: ttnn.Tensor,        # [num_experts, hidden_size, intermediate_size]
    w2_proj: ttnn.Tensor,        # [num_experts, intermediate_size, hidden_size]
    sparsity: ttnn.Tensor,       # sparsity mask
    out: ttnn.Tensor,            # output buffer
) -> None:

    BLOCK_SIZE = 32  # SPARSITY_BLOCK_SIZE
    num_sparse_blocks = x_sparse.shape[0]
    hidden_tiles = x_sparse.shape[2] // ttl.TILE_SHAPE[0]
    inter_tiles = w1_proj.shape[2] // ttl.TILE_SHAPE[0]

    # DFBs for streaming tiles through the pipeline
    x_dfb = ttl.make_dataflow_buffer_like(x_sparse, shape=(1, 1, 1), block_count=2)
    w1_dfb = ttl.make_dataflow_buffer_like(w1_proj, shape=(1, 1), block_count=2)
    w3_dfb = ttl.make_dataflow_buffer_like(w3_proj, shape=(1, 1), block_count=2)
    # Intermediate stays in L1 -- no DFB needed for DRAM
    w2_dfb = ttl.make_dataflow_buffer_like(w2_proj, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for block_idx in range(num_sparse_blocks):
            # Accumulate gate projection: x @ w1
            gate_acc = ttl.math.fill(0)
            for kt in range(hidden_tiles):
                with x_dfb.wait() as x_blk, w1_dfb.wait() as w1_blk:
                    gate_acc += x_blk @ w1_blk

            # Apply SiLU in-register (no DRAM write)
            gate_activated = ttl.math.silu(gate_acc)

            # Accumulate up projection: x @ w3
            up_acc = ttl.math.fill(0)
            for kt in range(hidden_tiles):
                with x_dfb.wait() as x_blk, w3_dfb.wait() as w3_blk:
                    up_acc += x_blk @ w3_blk

            # Elementwise multiply in-register
            intermediate = gate_activated * up_acc

            # Down projection: intermediate @ w2
            down_acc = ttl.math.fill(0)
            for it in range(inter_tiles):
                with w2_dfb.wait() as w2_blk:
                    down_acc += intermediate[it] @ w2_blk

            # Write final result
            with out_dfb.reserve() as out_blk:
                out_blk.store(down_acc)

    @ttl.datamovement()
    def dm_read():
        # Stream x tiles and weight tiles into DFBs
        # Sparsity mask controls which blocks are active
        for block_idx in range(num_sparse_blocks):
            # Read x block (reused for both w1 and w3 projections)
            for kt in range(hidden_tiles):
                with x_dfb.reserve() as x_blk:
                    tx = ttl.copy(x_sparse[block_idx, :, kt], x_blk)
                    tx.wait()
            # Read w1 tiles
            for kt in range(hidden_tiles):
                with w1_dfb.reserve() as w1_blk:
                    tx = ttl.copy(w1_proj[expert_for_block, kt, :], w1_blk)
                    tx.wait()
            # Read w3 tiles (same loop structure)
            # ...
            # Read w2 tiles
            # ...

    @ttl.datamovement()
    def dm_write():
        for block_idx in range(num_sparse_blocks):
            with out_dfb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[block_idx])
                tx.wait()
```

The key insight is that `gate_acc`, `gate_activated`, `up_acc`, and `intermediate` all live in L1/registers and never touch DRAM. Only the final `down_acc` result is written out.

### Integration with TT-Symbiote

Following the integration contract from [Chapter 6](../ch6_integration_strategy/index.md):

```python
class TTNNFusedMoEExperts(TTNNModule):
    """Drop-in replacement for TTNNExperts with fused expert compute."""

    def forward(self, x, topk_experts_indices, topk_experts_weights):
        # Steps 1-2 unchanged: all_to_all_dispatch + sparsity generation
        # ...

        # Step 3-7 FUSED: single TT-Lang kernel call
        expert_output = fused_moe_expert_compute(
            x_sparse, self.tt_w1_proj, self.tt_w3_proj, self.tt_w2_proj,
            sparsity_t, output_buffer
        )

        # Steps 8-9 unchanged: all_to_all_combine + weight
        # ...
```

## Expected Benefit

| Metric | Current (5 kernel launches) | Fused (1 kernel launch) | Improvement |
|--------|---------------------------|------------------------|-------------|
| DRAM intermediate traffic | ~38 MB per expert batch | ~0 MB (L1 only) | ~38 MB saved |
| Kernel launch overhead | 5 launches | 1 launch | 5x fewer launches |
| L1 utilization | Low (intermediates spill to DRAM) | High (intermediates stay in L1) | Significant |
| Trace compatibility | Compatible (sparse path) / Incompatible (per-expert loop) | Fully compatible | Enables trace on all paths |

The DRAM bandwidth saving is the dominant factor. At Tenstorrent's DRAM bandwidth of ~300 GB/s (Wormhole), saving 38 MB per expert batch translates to ~0.13 ms per layer. For a 46-layer model, that is ~6 ms per forward pass -- meaningful for decode latency.

### Additional Opportunity: Fused w1/w3 Projection

The Qwen variant in `qwen_moe.py` already stores `gate_up_proj` as a single `[num_experts, 2*intermediate_size, hidden_size]` tensor. A further optimization fuses the w1 and w3 `sparse_matmul` calls into a single call with double the output width, halving the number of weight-tile reads from DRAM.

---

**Next:** [`fused_attention.md`](./fused_attention.md)
