# Flash-Decode and Grouped Query Attention

This file covers the decode-phase attention kernel in tt-transformers. During decode, Q is a single new token — a vector of shape `[1, head_dim]` per head per batch element — while K and V are the full accumulated KV cache, growing by one position per step. The parallelization strategy is fundamentally different from prefill: there is no Q sequence to parallelize over, so the kernel parallelizes over the KV sequence dimension instead. This is the Flash-Decode algorithm. GQA (Grouped Query Attention) and MQA (Multi-Query Attention) are handled as a natural extension of this KV-parallel scheme.

---

## Flash-Decode: Parallelizing Over KV Sequence

### The Prefill Strategy Does Not Transfer to Decode

In prefill, Q has S rows. The FlashAttention-2 kernel parallelizes Q chunks across cores: each core handles a different subset of the query rows. With S=2048 and 64 cores, each core processes ~32 Q rows.

In decode, Q has 1 row (one new token per batch element). Parallelizing over Q rows assigns the same single row to all cores — every core does identical work and the results must be reduced. This gives no speedup and introduces unnecessary synchronization overhead.

### KV-Parallel Flash-Decode

Flash-Decode instead splits the KV sequence across cores:

- Each core is responsible for a contiguous slice of the KV cache: positions `[kv_start, kv_start + kv_chunk_size)`
- Each core computes a partial attention output and its local softmax statistics (row-max `m` and normalization sum `l`)
- A final reduction step combines partial outputs from all cores into the correct global attention output

The reduction uses the same online softmax update rule as prefill, applied across core outputs rather than across KV blocks:

```
# Per-core output:
(O_partial, m_partial, l_partial)

# Global reduction across P cores:
m_global = max(m_0, m_1, ..., m_{P-1})
l_global = sum(exp(m_i - m_global) * l_i  for i in 0..P-1)
O_global = sum(exp(m_i - m_global) * l_i * O_i  for i in 0..P-1) / l_global
```

This reduction is exact — it produces the same output as if a single core processed the entire KV sequence. The reduction overhead is O(P) rather than O(S), making it negligible for the P values used in practice (typically 8–32 cores per head).

### Input and Output Shapes

For a model with `n_heads` Q heads, `n_kv_heads` KV heads, batch size `B`, KV cache length `S`, and head dimension `D`:

| Tensor | Shape | Layout |
|---|---|---|
| Q | `[1, B, n_heads, D]` | TilizedLayout, L1-sharded or interleaved |
| K cache | `[B, n_kv_heads, S, D]` | TilizedLayout, DRAM interleaved |
| V cache | `[B, n_kv_heads, S, D]` | TilizedLayout, DRAM interleaved |
| Output | `[1, B, n_heads, D]` | TilizedLayout, L1-sharded |

The leading `1` in Q's shape represents the single decode token. K and V have shape `[B, n_kv_heads, S, D]` — the batch and heads dimensions are first, followed by the full sequence length S, making the cache a 4D tensor where S is the active KV dimension that grows each step.

---

## Grouped Query Attention (GQA) and Multi-Query Attention (MQA)

### What GQA/MQA Are

Standard multi-head attention (MHA) uses the same number of Q and KV heads: `n_heads == n_kv_heads`. This makes the KV cache large: at `n_kv_heads=32`, D=128, S=4096, BF16, the per-layer KV cache per batch element is 2 × 32 × 4096 × 128 × 2 bytes = **64 MB**.

GQA reduces `n_kv_heads` while keeping `n_heads` at its original value. Multiple Q heads share one KV head:

- **GQA**: `n_kv_heads = n_heads / G` for some group factor G (e.g., Llama 3 8B: 32 Q heads, 8 KV heads, G=4)
- **MQA**: `n_kv_heads = 1` — all Q heads share a single KV head (extreme case, G = n_heads)

This reduces the KV cache memory footprint by a factor of G. For Llama 3 8B with G=4, the KV cache is 4× smaller than MHA would require.

### GQA Mapping to Core Parallelization

In Flash-Decode, the core grid is partitioned over `(batch × n_kv_heads × kv_chunks)`. For GQA, all G Q heads within a group read the same KV head. The kernel handles this by broadcasting the shared KV head's output to all G Q heads within the group — no G-fold replication of KV data is needed.

The program config field `n_heads` controls the Q head count, and `n_kv_heads` controls the KV head count. The kernel infers the group size `G = n_heads / n_kv_heads` and assigns Q-head indexing accordingly:

```python
# Example: GQA decode for Llama 3 8B
# n_heads=32 Q heads, n_kv_heads=8 KV heads, G=4

config = ttnn.SDPAMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, 4),  # 32 cores
    q_chunk_size=1,                          # Single token per step (decode)
    kv_chunk_size=512,                       # KV slice per core
)
```

Each of the 8 KV heads is handled by a group of cores; within each group, the 4 Q heads that share that KV head run on different cores or are processed sequentially, depending on the grid assignment.

---

## `SDPAMultiCoreProgramConfig`: Configuration Reference

Decode attention uses `SDPAMultiCoreProgramConfig`, not the prefill `SDPAProgramConfig`. The key difference is that `kv_chunk_size` drives the primary parallelization axis:

```python
import ttnn

config = ttnn.SDPAMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, 4),  # Core grid for decode
    q_chunk_size=1,                          # Always 1 for single-token decode
    kv_chunk_size=512,                       # KV positions assigned per core
)

output = ttnn.transformer.scaled_dot_product_attention_decode(
    query,               # [1, batch, n_heads, head_dim]
    key,                 # [batch, n_kv_heads, seq, head_dim]
    value,               # [batch, n_kv_heads, seq, head_dim]
    cur_pos_tensor=cur_pos,  # [batch] int32 tensor of current positions
    scale=None,
    program_config=config,
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    ),
)
```

| Parameter | Type | Description |
|---|---|---|
| `compute_with_storage_grid_size` | `(int, int)` | Core grid. For decode, grid size is chosen to cover `batch × n_kv_heads × kv_chunks_per_head`. |
| `q_chunk_size` | `int` | Set to 1 for standard single-token decode. |
| `kv_chunk_size` | `int` | KV positions per core slice. Larger values reduce the number of partial outputs to reduce, but increase per-core DRAM traffic. |

---

## `cur_pos_tensor`: Per-Batch Position Tracking

Decode steps in batched serving rarely advance all sequences in the batch at the same rate. Some sequences may have finished generating (reached an end-of-sequence token) while others continue. The `cur_pos_tensor` parameter provides per-sequence position information to the kernel:

```python
# cur_pos_tensor: int32 tensor of shape [batch]
# Each element is the current KV cache position (0-indexed length) for that sequence
# pos = -1 signals that the sequence is finished; the kernel skips its attention computation

cur_pos = ttnn.from_torch(
    torch.tensor([512, 1024, -1, 256], dtype=torch.int32),
    dtype=ttnn.int32,
    device=device,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

When a sequence's position is `-1`, the kernel produces a zero output for that batch slot without reading from the KV cache. This avoids wasting DRAM bandwidth and FPU cycles on finished sequences during batched decode.

For contiguous KV caches, `cur_pos_tensor[b]` also determines how many KV positions are valid for batch element `b`. The kernel does not read beyond `cur_pos_tensor[b]` positions in the KV cache for that element, avoiding incorrect attention over uninitialized cache slots.

---

## Math Fidelity for Decode

Decode attention score computation (`Q @ K^T`) involves dot products of length `head_dim` (typically 64–128). Accumulation precision matters for correctness, but full FP32 accumulation (HiFi4) is rarely needed:

| Math Fidelity | Typical Use in Decode |
|---|---|
| LoFi | Not recommended for attention (score sensitivity to mantissa bits) |
| HiFi2 | Standard choice: good accuracy, ~2× throughput vs HiFi4 |
| HiFi3 | Occasionally used for D=256 or mixed-precision models |
| HiFi4 | Rarely needed; reserved for models where decode attention scores are empirically unstable at HiFi2 |

The online softmax reduction (computing `exp` and `rowmax` over partial scores) runs on the SFPU. Setting `math_approx_mode=True` uses polynomial approximations for `exp`, which is appropriate here: softmax is stable to small `exp` perturbations as long as the input shift `m_i - m_global` is accurately tracked, which depends on FPU accumulation quality (HiFi2) rather than SFPU approximation quality.

The recommendation for tt-transformers models is: **use HiFi2 with `math_approx_mode=True`** for decode attention unless ablation studies show measurable output degradation.

---

## Ring-Distributed Scaled Dot Product Attention

For multi-device inference (e.g., a TG board with multiple chips in a ring topology), the KV cache is distributed across devices: each device holds a slice of the KV sequence. Computing global attention requires each device to attend to the full KV sequence, which normally requires all-gather of the entire KV cache — expensive at long sequences.

`ttnn.transformer.ring_distributed_scaled_dot_product_attention` avoids this full all-gather by exploiting the structure of causal masking:

```python
output = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
    query,   # Local Q shard on this device
    key,     # Local KV shard on this device
    value,
    is_causal=True,
    cur_pos_tensor=cur_pos,
    program_config=config,
    compute_kernel_config=compute_cfg,
    # Ring topology is inferred from device mesh configuration
)
```

### How Ring-Distributed SDPA Works

In a ring of D devices, the attention computation proceeds in D phases:

1. Each device computes a partial attention output using its local KV shard: `(O_local, m_local, l_local)`
2. KV shards rotate around the ring: each device sends its KV shard to the next device and receives the previous device's shard
3. Each device accumulates the incoming partial output using the online softmax reduction
4. After D-1 rotations, every device has seen the full KV sequence

The key optimization: with a causal mask, devices holding earlier KV positions (lower in the sequence) have heavily masked interactions with devices holding later Q positions (higher in the sequence). The ring-distributed kernel detects and skips inter-device KV rotations that are entirely masked, reducing the effective number of active ring phases from D to roughly D/2 for causal prefill. This is the same block-skip logic used within a single device (fully-masked blocks skipped), applied at the device-shard granularity.

The result is that ring-distributed SDPA reduces both the communication volume (fewer rotations needed) and the compute (fewer partial outputs to accumulate) compared to naive all-gather followed by local SDPA.

---

## Key Takeaways

- Flash-Decode parallelizes over the KV sequence dimension rather than the Q dimension, because decode Q has only one row (one new token). Each core computes a partial output over a KV slice; a final online softmax reduction produces the exact global output.
- Q shape for decode is `[1, batch, n_heads, head_dim]`; KV cache shape is `[batch, n_kv_heads, seq, head_dim]`. The leading `1` in Q reflects the single decode token; `n_kv_heads < n_heads` for GQA/MQA models.
- GQA/MQA reduces the KV cache by a factor of G = n_heads/n_kv_heads. The kernel broadcasts shared KV head outputs to all Q heads in the group without replicating KV data.
- `cur_pos_tensor` enables selective decode: sequences at position -1 are skipped entirely, saving DRAM reads and FPU cycles for finished batch elements.
- HiFi2 with `math_approx_mode=True` is the standard decode compute config. HiFi4 is rarely needed; LoFi is not recommended for attention score accumulation.
- Ring-distributed SDPA avoids full all-gather by rotating KV shards around the device ring and accumulating partial outputs; causal mask skipping reduces active phases from D to ~D/2.

---

## Further Reading

- Dao et al., "FlashDecoding: Fast Large Language Model Inference on Long Sequences" (2023) — the KV-parallel Flash-Decode algorithm
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023) — GQA/MQA architecture motivation and training recipe
- TT-Metal SDPA decode kernel source: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/` — `SDPAMultiCoreProgramConfig` and per-batch position tracking
- [flash_attention_prefill.md](./flash_attention_prefill.md) — the prefill path this file contrasts against
- [paged_attention_kv_cache.md](./paged_attention_kv_cache.md) — extends the decode path with paged KV cache for variable-length batch serving

---

**Next:** [`paged_attention_kv_cache.md`](./paged_attention_kv_cache.md)
