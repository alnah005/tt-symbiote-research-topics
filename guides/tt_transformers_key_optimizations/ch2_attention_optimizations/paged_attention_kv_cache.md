# Paged Attention and KV Cache Management

This file covers how tt-transformers manages KV cache memory for variable-length batch serving. The standard contiguous KV cache — a single pre-allocated tensor of shape `[batch, n_kv_heads, max_seq_len, head_dim]` — wastes memory when sequences in a batch vary significantly in length. Paged KV cache eliminates this waste by allocating KV memory in fixed-size blocks and addressing them through a page table, enabling efficient memory reuse across sequences of different lengths.

---

## The Memory Fragmentation Problem

### Contiguous KV Cache Waste

In a contiguous KV cache, every sequence in the batch is allocated `max_seq_len` positions up front. For a batch of 32 sequences with `max_seq_len=4096`, the KV cache reserves:

```
32 sequences × 4096 positions × n_kv_heads × head_dim × 2 bytes
= 32 × 4096 × 8 × 128 × 2 = 256 MB   (for Llama 3 8B parameters)
```

If actual sequence lengths in the batch are 128, 256, 512, and so on — typical in real serving workloads — the utilization of this 256 MB allocation may be as low as 10–20%. The remainder is wasted: allocated but unused DRAM that could otherwise hold another batch.

### Why Fragmentation Compounds at Scale

The problem worsens with heterogeneous request arrival patterns. As sequences complete and new shorter requests arrive, the freed "slots" in the contiguous layout cannot be reused by sequences of different lengths without copying or defragmentation. The result is DRAM fragmentation: large free regions that cannot be efficiently assigned to new requests.

Paged KV cache addresses this by decoupling logical sequence positions from physical memory locations.

---

## Paged KV Cache: Addressing via Page Table

### Block-Structured Memory

In a paged KV cache, the physical KV cache tensor is partitioned into fixed-size blocks:

```
KV cache shape: [1, n_kv_heads, n_blocks * block_size, head_dim]
```

The leading dimension is `1` — not `batch`. The entire physical KV cache is a single flat pool of `n_blocks` blocks, each holding `block_size` consecutive KV positions. This pool is shared across all sequences in the batch. There is no per-sequence allocation; instead, a **page table** maps each sequence's logical positions to physical block indices.

### Page Table Structure

The page table is an integer tensor of shape `[batch, max_num_blocks_per_seq]`:

```python
# page_table[b, i] = physical block index for sequence b, block slot i
# Logical position p in sequence b maps to:
#   physical_block = page_table[b, p // block_size]
#   within_block_offset = p % block_size
```

Physical block indices are assigned by a block allocator outside the attention kernel. When a sequence needs a new block (because its current last block is full), the allocator picks a free block index from the pool and writes it into the page table. When a sequence finishes, its block indices are returned to the free list — immediately available for other sequences.

This gives near-100% KV cache utilization: only the last partial block of each active sequence is underutilized, rather than the full `max_seq_len - actual_len` positions wasted in contiguous layout.

---

## TTNN API for Paged Decode

### `paged_scaled_dot_product_attention_decode`

```python
import ttnn

output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    query,          # [1, batch, n_heads, head_dim]
    key_cache,      # [1, n_kv_heads, n_blocks * block_size, head_dim]  (paged pool)
    value_cache,    # [1, n_kv_heads, n_blocks * block_size, head_dim]
    cur_pos_tensor=cur_pos,    # [batch] int32, current sequence length per batch element
    page_table=page_table,     # [batch, max_num_blocks_per_seq] int32
    scale=None,
    program_config=config,
    compute_kernel_config=compute_cfg,
)
```

The kernel uses `page_table` and `cur_pos_tensor` together to determine which physical blocks to read for each batch element. For batch element `b` at position `cur_pos[b]`, the kernel reads blocks `page_table[b, 0]` through `page_table[b, cur_pos[b] // block_size]` from the flat KV pool. Each DRAM read is keyed by the physical block index rather than a contiguous stride.

### `paged_flash_multi_latent_attention_decode`

For models using Multi-Latent Attention (see the MLA section below), a separate entry point handles the compressed KV projection:

```python
output = ttnn.transformer.paged_flash_multi_latent_attention_decode(
    query,
    key_cache,   # Compressed latent KV cache
    value_cache,
    cur_pos_tensor=cur_pos,
    page_table=page_table,
    scale=None,
    program_config=config,
    compute_kernel_config=compute_cfg,
)
```

---

## Page Table Tensor and DRAM Access Pattern

### How the Kernel Uses the Page Table

The page table participates directly in the attention kernel's DRAM access pattern. The kernel does not pre-gather KV blocks into a contiguous buffer; instead, it reads each block's physical address from the page table at kernel launch and issues per-block NoC DMA reads to those physical addresses:

```
For each KV block b_idx in [0, num_blocks_for_this_sequence):
    physical_block = page_table[batch_idx, b_idx]
    k_ptr = key_cache_base + physical_block * block_size * head_dim * sizeof(bfloat16)
    v_ptr = value_cache_base + physical_block * block_size * head_dim * sizeof(bfloat16)
    issue_noc_dma_read(k_ptr, block_size * head_dim elements → L1 CB slot)
    issue_noc_dma_read(v_ptr, block_size * head_dim elements → L1 CB slot)
```

Because physical blocks are not contiguous in DRAM (different sequences may have interleaved block allocations), each DMA read is a separate NoC transaction. The kernel issues these transactions asynchronously and double-buffers them in L1 circular buffers — the same overlap mechanism described in [flash_attention_prefill.md](./flash_attention_prefill.md). Provided DRAM bandwidth is not saturated, the scattered access pattern does not meaningfully degrade throughput compared to contiguous KV reads.

### Page Table Data Type

The page table tensor must be `ttnn.int32`, `ROW_MAJOR_LAYOUT`, stored in DRAM:

```python
page_table = ttnn.from_torch(
    torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32),
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

The `int32` type accommodates page indices up to 2³¹ − 1, which is sufficient for any practical KV pool size. The `ROW_MAJOR_LAYOUT` requirement means the page table is not tiled — it is read as a flat integer array by the kernel's BRISC reader, not processed by the FPU.

---

## Program Caching and Stable Page Table Shape

### Why Shape Stability Matters

TTNN compiles RISC-V kernel binaries at first use and caches them keyed on operation inputs, data types, shapes, and program configs. If any of these change between calls, TTNN recompiles — a process that takes hundreds of milliseconds and interrupts decode throughput.

For paged attention, the page table's shape `[batch, max_num_blocks_per_seq]` must remain **constant across all decode steps**. This means:

- `batch` is fixed (determined at serving session start)
- `max_num_blocks_per_seq` is fixed (determined by maximum supported sequence length / block_size)

The page table's *values* (the block index integers) change every time a new block is allocated for any sequence — but that is fine, because TTNN caches on shape and dtype, not values. Only a shape change triggers recompilation.

Violating shape stability — for example, by dynamically increasing `max_num_blocks_per_seq` as a sequence grows longer than originally anticipated — causes a cache miss and recompilation. The correct approach is to pre-allocate `max_num_blocks_per_seq` based on the maximum sequence length the server will support, and fill newly allocated block indices in-place without reshaping.

---

## `paged_update_cache`: Fused K and V Cache Writes

After each decode step, the newly computed K and V vectors for the current token must be written into the KV cache at the current position. For paged KV, the write target is determined by the page table (which physical block and which offset within that block).

TTNN provides `paged_update_cache` to perform this write efficiently:

```python
# After computing new K, V for the current decode step:
ttnn.transformer.paged_update_cache(
    cache=key_cache,       # [1, n_kv_heads, n_blocks * block_size, head_dim]
    input=new_key,         # [batch, n_kv_heads, 1, head_dim] — new token's K
    update_idxs=cur_pos,   # [batch] int32 — logical position to write to
    page_table=page_table, # [batch, max_blocks_per_seq] int32
)
ttnn.transformer.paged_update_cache(
    cache=value_cache,
    input=new_value,
    update_idxs=cur_pos,
    page_table=page_table,
)
```

### Core Sharding for Parallel K and V Writes

The `paged_update_cache` operation fuses K and V cache updates using a core sharding strategy that maximizes parallel DRAM write throughput:

- **Cores [0–7]**: write K cache updates — each core handles a subset of the `n_kv_heads` heads for K
- **Cores [8–15]**: write V cache updates — each core handles a subset of the `n_kv_heads` heads for V

K and V updates run concurrently on non-overlapping core subsets. This doubles the effective DRAM write bandwidth for cache updates compared to a sequential K-then-V approach, and avoids bank conflicts between the two writes because K and V typically occupy different DRAM regions in the flat block pool.

The page table lookup (translating `cur_pos[b]` to a physical block index and within-block offset) is performed by the BRISC reader on each core before issuing the NoC write, adding minimal overhead per batch element.

### Conditional Page Table Update

The page table itself only needs to be updated at block boundaries: when `cur_pos[b] % block_size == 0`, sequence `b` has consumed its current last block and needs a new physical block allocated and written into `page_table[b, cur_pos[b] // block_size]`. This allocation is performed on the host (or a control thread) and does not require a TTNN kernel call — it is a direct write to the page table tensor's DRAM buffer.

For `block_size=32` and a typical decode step rate of ~50 tokens/second per sequence, block boundary events occur roughly every 0.6 seconds per sequence — infrequent enough that the allocation cost is negligible relative to the decode kernel runtime.

---

## Multi-Latent Attention (MLA)

### Compressed KV Projection in DeepSeek Models

Multi-Latent Attention (MLA), introduced in DeepSeek-V2, reduces KV cache size by projecting K and V through a low-rank bottleneck before caching. Instead of caching the full `[n_kv_heads, head_dim]` vector per token, MLA caches a compressed latent vector of dimension `kv_lora_rank` (typically 512, vs a full KV size of `n_kv_heads × head_dim` which may be 2048+). K and V are reconstructed from the latent at attention time via a learned up-projection.

This reduces KV cache memory by a factor of `n_kv_heads × head_dim / kv_lora_rank`. For DeepSeek-V2 parameters, this is approximately a 4–8× reduction.

### TTNN MLA Entry Points

tt-transformers provides two MLA-specific TTNN operations:

```python
# Prefill: process a full prompt with compressed KV
output = ttnn.transformer.flash_mla_prefill(
    query,              # [batch, n_heads, seq, head_dim_q]
    kv_latent,          # [batch, seq, kv_lora_rank]  — compressed KV latent
    wkv_b,              # [n_kv_heads, kv_lora_rank, 2 * head_dim_kv]  — up-projection weight
    is_causal=True,
    scale=None,
    program_config=config,
    compute_kernel_config=compute_cfg,
)

# Prefill with chunking for very long sequences
output = ttnn.transformer.chunked_flash_mla_prefill(
    query,
    kv_latent,
    wkv_b,
    chunk_size=2048,    # Process kv_latent in chunks to fit within L1
    is_causal=True,
    scale=None,
    program_config=config,
    compute_kernel_config=compute_cfg,
)
```

`flash_mla_prefill` fuses the KV up-projection (`kv_latent @ wkv_b`) with the attention computation into a single kernel pass, avoiding the overhead of materializing the full `[batch, n_kv_heads, seq, head_dim_kv]` K and V tensors to DRAM. The up-projection matmul runs in L1 for each tile, and the result feeds directly into the FlashAttention-2 scoring loop.

`chunked_flash_mla_prefill` is used for long-context prefill where even the kv_latent tensor is too large to process in one pass. It partitions the KV latent sequence into chunks and accumulates online softmax statistics across chunks, using the same update rule as standard FlashAttention-2.

For decode with paged MLA KV cache, use `paged_flash_multi_latent_attention_decode` (covered in the TTNN API section above), which reads compressed latent blocks from the paged pool and performs the up-projection on-the-fly during the Flash-Decode KV-parallel loop.

---

## Key Takeaways

- Contiguous KV caches waste DRAM proportionally to `(max_seq_len - actual_len) / max_seq_len`; for heterogeneous batch workloads this can exceed 80% waste. Paged KV cache allocates memory in fixed-size blocks and wastes at most `block_size - 1` positions per sequence.
- The paged KV pool has shape `[1, n_kv_heads, n_blocks * block_size, head_dim]` — a single flat buffer shared across the entire batch. The page table (`[batch, max_blocks_per_seq]`, int32) maps logical positions to physical block indices; the attention kernel issues per-block NoC DMA reads keyed by these indices.
- Page table *shape* must remain constant across decode steps to avoid TTNN kernel recompilation. Only shape changes trigger recompilation; value changes (new block allocations) are transparent to the cache.
- `paged_update_cache` fuses K and V cache writes by assigning K updates to cores [0–7] and V updates to cores [8–15], enabling concurrent parallel DRAM writes. Page table entries are updated on the host only at block boundaries (`cur_pos % block_size == 0`), not every decode step.
- MLA (Multi-Latent Attention) compresses the KV cache by caching a low-rank latent vector rather than full K and V; `flash_mla_prefill` and `chunked_flash_mla_prefill` fuse the up-projection with attention to avoid materializing full K/V tensors to DRAM. `paged_flash_multi_latent_attention_decode` extends this to paged decode.

---

## Further Reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — the original vLLM paper introducing paged KV and block-level memory management
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024) — Multi-Latent Attention architecture and training details
- TT-Metal paged attention source: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/` — `paged_scaled_dot_product_attention_decode` kernel implementation
- TT-Metal `paged_update_cache` source: `ttnn/cpp/ttnn/operations/kv_cache/device/` — core sharding layout and physical block address resolution
- [flash_decode_and_gqa.md](./flash_decode_and_gqa.md) — the non-paged decode baseline that this file extends

---

**Next:** [Chapter 3 — MatMul Optimizations](../ch3_matmul_optimizations/index.md)
