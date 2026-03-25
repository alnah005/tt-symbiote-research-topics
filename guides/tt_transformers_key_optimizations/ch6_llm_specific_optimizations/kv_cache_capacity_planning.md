# KV Cache Capacity Planning

This file covers the practical engineering of KV cache memory allocation for LLM inference on Tenstorrent hardware: how to calculate the DRAM budget for a given model and context configuration, how to size the paged block pool, what constraints the page table imposes, and how increasing batch size for throughput trades against available KV cache capacity.

For the paged KV cache data structure and the attention kernel that reads it, see Ch2 (`paged_attention_kv_cache.md`). For the DRAM memory hierarchy and bandwidth characteristics, see Ch4 (`memory_hierarchy.md`). This file builds on those foundations; it does not re-derive the paging mechanism.

---

## KV Cache Shape and Storage Location

### Physical Layout

[confirmed] The KV cache for a single transformer layer has the following shape in tt-transformers:

```
key_cache:   [1, n_kv_heads, n_blocks * block_size, head_dim]
value_cache: [1, n_kv_heads, n_blocks * block_size, head_dim]
```

The leading dimension is `1` — not `batch`. The entire pool is a single flat tensor shared across all sequences in the batch. Individual sequences address their portion of the pool through the page table, not through a per-sequence slice of the first dimension.

[confirmed] KV cache tensors are stored in DRAM (not L1). L1 is the 120 KB per-core scratchpad used during active attention computation; the KV cache is far too large to reside in L1 and persists across decode steps, making DRAM the only viable storage location.

### Why the First Dimension Is 1, Not Batch

The `[1, ...]` shape reflects the flat-pool design: there is no per-batch partitioning at the tensor level. The page table (`[batch, max_pages]`) handles the mapping from logical sequence positions to physical blocks within the flat pool. This design means the KV cache tensor shape does not change when batch size changes — only the page table shape changes — which simplifies program caching (see the Page Table Constraints section).

---

## Per-Token KV Cache Footprint

### Reference Model: Llama 3.1 8B

The following parameters define the KV cache size for Llama 3.1 8B [confirmed]:

| Parameter | Value |
|---|---|
| `n_kv_heads` | 8 |
| `head_dim` | 128 |
| `n_layers` | 32 |
| KV data type | BF16 |
| Bytes per BF16 element | 2 |

**Per-token, per-head, per-layer (K only):**

```
128 elements × 2 bytes = 256 bytes
```

**Per-token, per-head, per-layer (K + V):**

```
256 bytes × 2 (K and V) = 512 bytes
```

**Per-token, all KV heads, per-layer:**

```
8 heads × 512 bytes = 4096 bytes = 4 KB
```

**Per-token, all layers:**

```
32 layers × 4 KB = 128 KB per token (across all layers, BF16)
```

[INFERRED from the confirmed per-layer calculation] For a 32K-token context with 32 layers in BF16:

```
32,768 tokens × 128 KB/token = 4,194,304 KB = 4 GB
```

This 4 GB figure is the DRAM requirement for the KV cache alone, not counting model weights. On a Wormhole device, this is a significant fraction of the available DRAM budget; it motivates KV cache quantization (see below).

### Effect of KV Cache Quantization

[confirmed] tt-transformers supports storing the KV cache in BFP8 (block floating point 8-bit). [INFERRED] BFP8 uses approximately half the memory of BF16 per element (7-bit mantissa plus a shared 8-bit exponent per 16-value block, vs 16-bit BF16); the specific internal bit layout is a format detail not independently confirmed in the key facts for this chapter.

| KV dtype | Bytes per element (approx) | Per-token (all layers, Llama 3.1 8B) | 32K-token KV budget |
|---|---|---|---|
| BF16 | 2 | 128 KB | ~4 GB |
| BFP8 | ~1 | ~64 KB | ~2 GB |

[INFERRED] BFP8 KV cache halves the DRAM requirement, enabling either twice the context length at the same memory budget or twice the batch size at the same context length. For more detail on BFP8 format and its accuracy implications, see Ch3 (`weight_layout_and_quantization.md`).

### Generalizing to Other Models

For an arbitrary transformer with `n_kv_heads`, `head_dim`, and `n_layers`, the per-token KV footprint in bytes is:

```
per_token_bytes = n_kv_heads × head_dim × 2 (K+V) × bytes_per_element × n_layers
```

For BF16 (`bytes_per_element = 2`):

```
per_token_bytes = n_kv_heads × head_dim × 4 × n_layers
```

---

## Block Pool Sizing

### Paged Block Pool Parameters

The paged KV cache pool is parameterized by two values [confirmed]:

- `n_blocks`: total number of physical blocks in the pool
- `block_size`: number of token positions per block

These determine the physical KV cache tensor size:

```
KV pool shape: [1, n_kv_heads, n_blocks * block_size, head_dim]
Pool DRAM bytes (per layer, BF16) = n_blocks * block_size * n_kv_heads * head_dim * 2 * 2 (K+V)
```

### Minimum Pool Size Formula

The minimum pool that can support `batch` sequences each growing up to `max_seq_len` tokens requires:

```python
import math

blocks_per_seq = math.ceil(max_seq_len / block_size)
n_blocks_minimum = batch * blocks_per_seq
```

In practice, `n_blocks` is set somewhat above this minimum to provide headroom for:

- Variable sequence lengths: not all sequences will reach `max_seq_len`, but the pool must handle the worst case where they all do simultaneously.
- Page table pre-allocation: the page table has shape `[batch, max_pages]` where `max_pages = blocks_per_seq`. This shape must remain constant across all decode steps (see Page Table Constraints). Pre-allocating `max_pages` per sequence means committing the page table entries up front, even if some sequences never use all their blocks.

### Worked Example: Llama 3.1 8B, Batch 32, 4K Context, BF16

```
block_size = 64 (a typical value; model-specific)
max_seq_len = 4096
batch = 32

blocks_per_seq = ceil(4096 / 64) = 64
n_blocks = 32 * 64 = 2048

# Per-layer pool size (BF16):
pool_elements = 2048 * 64 * 8 * 128 = 134,217,728 elements
pool_bytes_per_layer = 134,217,728 * 2 * 2 = 536,870,912 bytes = 512 MB (K + V combined)

# Total KV cache DRAM across all 32 layers:
total_kv_dram = 512 MB * 32 = 16 GB
```

This 16 GB total illustrates the dominant size of the KV cache for a large batch at long context in BF16. In practice, BFP8 storage cuts this to approximately 8 GB, and smaller context windows or smaller batch sizes reduce it further.

### Worked Example: Same Config with BFP8 KV Cache

```
# BFP8: approximately 1 byte per element (vs 2 for BF16)
pool_bytes_per_layer (BFP8) ≈ 512 MB / 2 = 256 MB
total_kv_dram (BFP8) ≈ 256 MB * 32 = 8 GB
```

[INFERRED] The 8 GB figure for BFP8 KV cache at batch 32 / 4K context leaves more DRAM headroom for model weights and activation buffers on the same device, enabling configurations that would OOM in BF16 KV mode.

---

## Page Table Constraints

### Shape and Data Type

[confirmed] The page table has shape `[batch, max_pages]` and stores `int32` physical block indices:

```python
page_table = ttnn.from_torch(
    torch.zeros(batch, max_pages, dtype=torch.int32),
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

where `max_pages = ceil(max_seq_len / block_size)`.

### Shape Must Remain Constant

[confirmed] The page table's shape (`batch` and `max_pages`) must remain constant across all decode steps. TTNN caches compiled kernel programs keyed on tensor shapes and dtypes. A shape change triggers recompilation, which takes hundreds of milliseconds and interrupts decode throughput.

Correct practice is to pre-allocate the page table at its maximum size before the first decode step and fill block index values in-place as sequences grow:

```python
# Pre-allocate page table at full size. Values are initially 0 (or invalid sentinels).
page_table_host = torch.zeros(batch, max_pages, dtype=torch.int32)

# As each sequence grows and needs a new block, update the value at the appropriate index:
def allocate_block_for_sequence(seq_idx: int, block_slot: int, physical_block_id: int):
    page_table_host[seq_idx, block_slot] = physical_block_id
    # Copy the updated row back to device in-place (no reshape).
    ttnn.copy_host_to_device_tensor(page_table_host, page_table)
```

Only value mutations (writing new block indices) are needed after initialization. Reshaping the page table — for example, by increasing `max_pages` because a sequence grew longer than anticipated — causes a cache miss and recompilation. The correct mitigation is to pre-set `max_pages` based on the maximum sequence length the server will ever support for this serving session, not the current longest sequence.

### Batch Dimension Is Also Fixed

The `batch` dimension of the page table is equally constrained. In continuous batching, batch slots are logically "recycled" (a completed sequence's slot is taken over by a new sequence), but the `batch` size of the page table tensor does not change. The new sequence's block indices overwrite the old sequence's entries in-place.

---

## Capacity vs Throughput Tradeoffs

### The Core Tradeoff

Increasing batch size improves decode throughput (tokens/second total) by amortizing the fixed DRAM weight-load cost across more simultaneous sequences. [confirmed] However, KV cache capacity is the primary bound on how large batch size can be: each additional sequence in the batch requires `blocks_per_seq` physical blocks, and the total pool `n_blocks` is finite.

This creates a direct tradeoff:

| Lever | Effect on throughput | Effect on KV capacity used |
|---|---|---|
| Increase batch size | Higher total T/S (more tokens per step) | More blocks consumed; pool exhausted sooner |
| Decrease `max_seq_len` | Fewer blocks per sequence | More sequences fit in same pool |
| Use BFP8 KV cache | Same block count at half the DRAM bytes | Doubles the pool DRAM budget at the same memory footprint |
| Increase `block_size` | Fewer blocks needed for the same seq len | Coarser granularity; last-block waste increases |
| Decrease `block_size` | Finer granularity; lower last-block waste | More blocks needed; larger page table |

### Throughput Scaling with Batch Size

As established in `prefill_decode_pipeline.md` (Decode Is Memory-Bandwidth-Bound), each decode step loads the same weight tiles from DRAM regardless of batch size, so per-token throughput scales approximately linearly with batch size until a secondary constraint binds:

```
Approximate T/S (total tokens/second) ≈ constant × batch
```

The constant is determined by the DRAM bandwidth, weight matrix sizes, and the number of layers. [INFERRED] In practice, this linear scaling holds until one of the following limits is reached:

1. **KV cache DRAM capacity**: adding another sequence requires blocks that push the pool beyond available DRAM.
2. **KV cache bandwidth**: at very large batch sizes and long contexts, reading the KV cache for attention becomes a significant DRAM access alongside weight reads, reducing the effective per-sequence bandwidth.
3. **CCL latency floor**: under tensor parallelism (Ch5), collective communication adds a fixed per-step latency that does not shrink as batch size grows; it sets a floor on per-step time regardless of batch.

### Sizing the Pool for a Target Configuration

The following procedure gives a starting point for block pool sizing:

1. **Choose `block_size`**: values of 32 or 64 are common. Larger `block_size` reduces page table size but increases waste in the last block.
2. **Determine `max_seq_len`**: the longest sequence the server needs to support.
3. **Compute `blocks_per_seq`**: `ceil(max_seq_len / block_size)`.
4. **Choose target batch size**: based on available DRAM after accounting for model weights.
5. **Compute `n_blocks`**: `batch * blocks_per_seq`, with a small headroom factor (e.g., 1.1x) to handle allocation overhead.
6. **Verify DRAM budget**: `total_kv_bytes = n_blocks * block_size * n_kv_heads * head_dim * bytes_per_element * 2 (K+V) * n_layers`.

```python
import math

def compute_kv_pool_bytes(
    n_blocks: int,
    block_size: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    bytes_per_element: float,  # 2 for BF16, ~1 for BFP8
) -> int:
    """Returns total DRAM bytes for the KV cache pool (all layers, K and V combined)."""
    elements_per_layer = n_blocks * block_size * n_kv_heads * head_dim
    bytes_per_layer = elements_per_layer * bytes_per_element * 2  # K + V
    return int(bytes_per_layer * n_layers)


# Llama 3.1 8B, batch=32, max_seq_len=4096, block_size=64, BF16:
n_blocks = 32 * math.ceil(4096 / 64)   # = 2048
total_bytes = compute_kv_pool_bytes(
    n_blocks=2048, block_size=64,
    n_kv_heads=8, head_dim=128,
    n_layers=32, bytes_per_element=2.0,
)
print(f"{total_bytes / 1e9:.1f} GB")   # ~16.0 GB (BF16)
```

### Practical Guidance: Small vs Large Context

| Configuration | Typical constraint | Recommended strategy |
|---|---|---|
| Short context (<=2K), large batch | KV pool fits easily; compute or CCL may dominate | Prioritize large batch for throughput; use BF16 KV for simplicity |
| Long context (32K–131K), any batch | KV pool DRAM is the binding constraint | Use BFP8 KV cache; set `max_seq_len` conservatively; consider smaller batch |
| Mixed workload (short + long sequences) | Pool fragmentation if longest sequence dominates allocation | Continuous batching with paged pool; set `max_pages` for the longest expected sequence; short sequences consume only their actual blocks |

---

## Interaction with Tensor Parallelism

Under tensor parallelism (Ch5), attention heads are distributed across devices: each device holds `n_kv_heads / TP` KV heads for the layers assigned to it. The per-device KV cache shape shrinks accordingly:

```
Per-device key_cache: [1, n_kv_heads / TP, n_blocks * block_size, head_dim]
```

[INFERRED] This means the per-device DRAM requirement for the KV cache scales inversely with TP degree (assuming TP <= n_kv_heads, the sharding condition discussed in Ch5). For Llama 3.1 8B at TP=8 on T3K, each device holds `8 / 8 = 1` KV head, so the per-device KV cache is 1/8 the single-device total. The page table shape remains `[batch, max_pages]` and is replicated on every device, since page table lookups must be available to every device's attention kernel.

The `n_blocks` value in the pool is per-device: the total physical block pool on each device must be sized to hold the sequences assigned to that device's KV heads. Because the page table maps logical sequence positions to physical blocks within the local pool, the addressing is local — no cross-device page table coordination is needed during attention.

---

## Key Takeaways

- The KV cache tensor shape is `[1, n_kv_heads, n_blocks * block_size, head_dim]` with a leading dimension of 1 (not batch). All sequences share the flat pool; the page table (`[batch, max_pages]`, int32) handles per-sequence addressing. KV cache is stored in DRAM. [confirmed]
- For Llama 3.1 8B in BF16: the per-token KV cost is 4 KB per layer, or 128 KB per token across all 32 layers. A 32K-context configuration requires approximately 4 GB of DRAM for the KV cache alone. [INFERRED from confirmed per-layer numbers]
- BFP8 KV cache halves the DRAM requirement (approximately 1 byte per element vs 2 for BF16), enabling twice the context length or batch size within the same DRAM budget. [INFERRED]
- The page table shape (`batch` and `max_pages`) must remain constant across all decode steps. Shape changes trigger TTNN kernel recompilation. Pre-allocate `max_pages = ceil(max_seq_len / block_size)` at session start and update values in-place. [confirmed]
- Increasing batch size improves total decode throughput (T/S) by amortizing fixed weight-loading costs, but is bounded by KV cache pool capacity (`n_blocks * block_size >= batch * max_seq_len`). KV cache DRAM is typically the binding constraint before compute or activation memory. [confirmed / INFERRED]

---

## Further Reading

- Ch2 `paged_attention_kv_cache.md` — physical block structure, page table addressing, `paged_update_cache` core sharding
- Ch3 `weight_layout_and_quantization.md` — BFP8 format definition, accuracy vs bandwidth tradeoff for KV quantization
- Ch4 `memory_hierarchy.md` — Wormhole DRAM bandwidth characteristics, interleaved vs sharded allocation
- Ch5 `tensor_parallelism.md` — KV head sharding under TP, per-device KV cache sizing when TP <= n_kv_heads
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — foundational paper on block-level KV memory management

---

**Next:** [Chapter 7 — Llama 3 End-to-End Walkthrough](../ch7_llama3_walkthrough/index.md)
