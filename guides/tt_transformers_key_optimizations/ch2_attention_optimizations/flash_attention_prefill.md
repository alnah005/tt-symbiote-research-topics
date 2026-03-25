# FlashAttention-2 Prefill on Tensix

This file covers how tt-transformers implements prefill attention: the forward pass over a full prompt of S tokens. The implementation is FlashAttention-2, adapted to the Tensix core's L1/DRAM hierarchy and RISC-V kernel model. The goal throughout is to keep attention intermediates in L1 and never write the S×S score matrix to DRAM.

---

## Why Naive Attention Is Memory-Bandwidth-Bound

Standard attention computes:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(D)) @ V
```

For a sequence of length S and head dimension D, this requires materializing the full S×S score matrix. On hardware with slow DRAM access, that materialization dominates runtime:

1. Compute `Q @ K^T` → write S×S scores to DRAM (O(S²) writes)
2. Compute `softmax(scores)` → read S×S from DRAM, write S×S probabilities back to DRAM
3. Compute `probs @ V` → read S×S from DRAM again

At S=4096, D=128, BF16: the score matrix alone is 4096 × 4096 × 2 bytes = **32 MB per head**. For a model with 32 heads, that is 1 GB per layer — a per-step DRAM traffic figure that dwarfs the actual weight reads. The matrix engine (FPU) idles during these DRAM reads; the kernel becomes memory-bandwidth-bound regardless of FPU peak throughput.

FlashAttention-2 eliminates the S×S materialization by tiling the computation: it never writes attention scores to DRAM. All intermediates stay in L1 SRAM (120 KB per Tensix core). This turns the attention kernel from memory-bandwidth-bound to compute-bound.

---

## FlashAttention-2 Tiling on Tensix

### The Core Idea

FlashAttention-2 processes attention in tiles:

- Partition Q into chunks of `q_chunk_size` rows (parallelized across cores)
- For each Q chunk, iterate over KV blocks of `kv_chunk_size` rows
- Maintain running statistics (row-max `m` and normalization sum `l`) across KV blocks to compute online softmax
- Accumulate output `O` tile-by-tile; finalize `O` after all KV blocks are processed

No intermediate S×S matrix is ever formed. The only DRAM traffic is reading Q, K, V tiles once each and writing the final output O once.

### L1 Budget Per Core

With `q_chunk_size = Br` and `kv_chunk_size = Bc`, the L1 working set per core is:

| Buffer | Size (BF16) |
|---|---|
| Q chunk: `Br × D` | `Br × D × 2` bytes |
| K block: `Bc × D` | `Bc × D × 2` bytes |
| V block: `Bc × D` | `Bc × D × 2` bytes |
| Output accumulator O: `Br × D` | `Br × D × 2` bytes |
| Running stats (m, l): `Br` | negligible |

At D=128, Br=128, Bc=128: Q + K + V + O = 4 × (128 × 128 × 2) = 128 KB. This exceeds the 120 KB L1 limit, so production configs typically use Br=Bc=64 (D=128) or rely on double-buffering to overlap one block in transit with another in compute. At D=64, Br=Bc=128 fits comfortably. The `kv_chunk_size` and `q_chunk_size` fields in `SDPAProgramConfig` directly control these dimensions.

### Online Softmax

Standard softmax over a row requires two passes over the scores: one to find the row maximum and one to compute the normalized exponentials. FlashAttention-2 fuses these into a single pass using the online update rule:

Given current running stats `(m_old, l_old)` and a new KV block producing partial scores `S_new`:

```
m_new = max(m_old, rowmax(S_new))
l_new = exp(m_old - m_new) * l_old + rowsum(exp(S_new - m_new))
O_new = diag(exp(m_old - m_new)) * O_old + exp(S_new - m_new) @ V_new
```

After all KV blocks: `O_final = O_new / l_new`

This update is numerically stable and runs entirely on the SFPU (element-wise ops on the Dst register) without any DRAM access. The SFPU's `exp` and `rsqrt` instructions handle the per-row statistics; the FPU handles the tile matmuls (`S_new = Q_chunk @ K_block^T` and `exp(S_new) @ V_block`).

---

## Causality-Aware Load Balancing

### The Imbalance Problem

Causal language model prefill uses a lower-triangular attention mask: token i can only attend to tokens 0..i. This means:

- **Q_low** (early tokens, small row index): attend to a short prefix of K — light work
- **Q_high** (late tokens, large row index): attend to almost the full KV sequence — heavy work

If Q chunks are assigned naively (e.g., core 0 gets rows 0–Br, core 1 gets rows Br–2Br, etc.), the cores handling Q_high chunks process far more non-masked KV blocks than cores handling Q_low chunks. The result is severe load imbalance: some cores finish early and idle while high-index cores are still computing.

### The Pairing Strategy

The tt-transformers FlashAttention-2 kernel pairs Q_low and Q_high chunks on the same core:

- A core receives one chunk from the lower-left triangle region (Q_low) and one from the upper-right triangle region (Q_high) of the attention map
- Their compute loads are approximately complementary: Q_low (early tokens) has most KV blocks masked and few valid ones — it is compute-light; Q_high (late tokens) has the majority of KV blocks fully valid (below the causal diagonal) and only a few masked (above-diagonal) blocks — it is compute-heavy

This pairing achieves approximately **~1.6× speedup** over naive sequential Q chunk assignment across the core grid. The gain is purely from scheduling, not from algorithmic changes.

Concretely: for a grid of N cores and sequence length S, the kernel divides Q into 2N chunks and assigns chunk `i` and chunk `2N-1-i` to core `i`. This ensures each core handles one sparse (early-token) chunk and one dense (late-token) chunk.

---

## Sparse Causal Mask: Skipping Fully-Masked Blocks

Not all KV blocks require computation. In the causal case, a KV block `k` attending to Q chunk `q` is:

- **Fully below diagonal**: all K tokens are earlier than all Q tokens — no masking, full compute
- **On diagonal**: the KV block partially overlaps the causal boundary — masking applied element-wise
- **Fully above diagonal**: all K tokens are later than all Q tokens — entire block is masked to -inf

Fully-above-diagonal KV blocks contribute zero to the output (after softmax, all weights are zero). The kernel detects these blocks by comparing the KV block's position against the Q chunk's starting row index, and skips them entirely — no FPU compute, no L1 load for that block.

For a Q chunk covering rows [q_start, q_start + Br) and a KV block covering columns [kv_start, kv_start + Bc):

- Skip if `kv_start >= q_start + Br` (fully above diagonal — all Q tokens precede these K tokens)
- Apply element-wise mask if `kv_start + Bc > q_start` and `kv_start < q_start + Br` (diagonal block)
- Skip masking if `kv_start + Bc <= q_start` (fully below diagonal — all K tokens precede all Q tokens in this chunk)

This optimization reduces the number of FPU invocations from O(S²/Br/Bc) to O(S²/2Br/Bc) on average for uniform causal masks — a 2× reduction in compute for dense prefill.

---

## Double-Buffering with Circular Buffers

### RISC-V Kernel Roles

Each Tensix core runs five concurrent RISC-V processors. In the FlashAttention-2 kernel:

| Processor | Role | What It Does |
|---|---|---|
| BRISC | Reader kernel | Issues NoC DMA reads to fetch Q, K, V tiles from DRAM into L1 circular buffer slots |
| NCRISC | Writer kernel | Issues NoC DMA writes to push finalized output O tiles from L1 to DRAM |
| TRISC0 | Compute stage 0 | Orchestrates Unpacker: moves tiles from circular buffer slots into SrcA/SrcB registers |
| TRISC1 | Compute stage 1 | Runs FPU: performs `Q @ K^T`, scales, then `exp(scores) @ V` accumulation |
| TRISC2 | Compute stage 2 | Orchestrates Packer + SFPU: applies online softmax update, packs output tiles back to L1 |

All five processors operate concurrently within the core. The kernel is structured so that BRISC's DMA reads overlap with TRISC1/2's compute on the previous batch of tiles.

### Circular Buffer Double-Buffering

A circular buffer (CB) in TTNN is a ring-buffered region of L1 SRAM with a fixed number of slots. The Reader kernel fills slots; the Compute kernel drains them. When a CB has two slots (double-buffering), the following overlap is possible:

```
Cycle:    |--- Compute slot A ---|--- Compute slot B ---|--- Compute slot A ---|
Reader:   |  Fill slot B  |  Fill slot A  |  Fill slot B  |
```

While Compute processes the tile in slot A, Reader is simultaneously fetching the next tile into slot B over the NoC/DRAM. When Compute finishes slot A and marks it free, the slot B tile is already present — Compute proceeds immediately without stalling.

The FlashAttention-2 kernel uses double-buffered CBs for Q, K, and V tiles separately. This hides the DRAM read latency (tens to hundreds of nanoseconds) behind the FPU compute time for each tile-level matmul (`Q_chunk @ K_block^T`).

Without double-buffering, each KV block step would stall: Compute finishes the previous tile matmul, then waits for Reader to fill the next K and V tiles before resuming. With double-buffering, that stall is reduced to near zero for workloads where DRAM bandwidth is sufficient to supply tiles at the same rate Compute consumes them.

---

## Performance: Achieved Speedup

FlashAttention-2 on Tensix achieves the following speedups over a naive DRAM-materialized attention baseline:

| Sequence Length | Head Dim | Speedup |
|---|---|---|
| 512 | 64 | ~9× |
| 2048 | 128 | ~20× |
| 8192 | 128 | ~30× |
| 16384 | 128 | ~44× |
| 16384 | 256 | ~38× |

The speedup grows with sequence length because the DRAM traffic for the naive S×S baseline grows quadratically while FlashAttention-2's DRAM traffic grows linearly in S. The **~20× figure** (at S=2048, D=128) is the representative middle-case often cited in documentation. These numbers are sourced from tt-metal benchmark runs across the specified sequence lengths and head dimensions.

---

## `SDPAProgramConfig`: Configuration Reference

The `SDPAProgramConfig` object controls how the FlashAttention-2 kernel is mapped to the core grid:

```python
import ttnn

config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=(8, 8),  # Use an 8×8 core grid (64 cores)
    q_chunk_size=128,                        # Q rows per core chunk (must be multiple of 32)
    kv_chunk_size=128,                       # KV rows per block iteration (must be multiple of 32)
)

output = ttnn.transformer.scaled_dot_product_attention(
    query,           # [batch, n_heads, seq_q, head_dim], TilizedLayout
    key,             # [batch, n_kv_heads, seq_kv, head_dim]
    value,           # [batch, n_kv_heads, seq_kv, head_dim]
    is_causal=True,
    scale=None,      # Defaults to 1/sqrt(head_dim) if None
    program_config=config,
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,      # SFPU approximation for exp/rsqrt
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    ),
)
```

| Parameter | Type | Description |
|---|---|---|
| `compute_with_storage_grid_size` | `(int, int)` | Core grid dimensions `(cols, rows)`. Total cores = product. Controls how many Q chunks run in parallel. |
| `q_chunk_size` | `int` | Rows of Q assigned to each core per chunk. Must be a multiple of 32 (one TTNN tile row). Larger values reduce kernel launch overhead but must fit in L1 alongside K/V/O buffers. |
| `kv_chunk_size` | `int` | Rows of K/V per inner-loop iteration. Larger values amortize loop overhead but increase peak L1 usage. |

### Sizing Rule of Thumb

For a given head dimension D (in elements, BF16):

```
L1 used ≈ (q_chunk_size + kv_chunk_size + kv_chunk_size + q_chunk_size) × D × 2 bytes
         = (2 × q_chunk_size + 2 × kv_chunk_size) × D × 2 bytes
```

This must be kept below ~80–90 KB to leave room for kernel binary, stack, and circular buffer metadata. For D=128:

| Chunk Sizes | L1 Used (approx) |
|---|---|
| Br=64, Bc=64 | 64 KB |
| Br=128, Bc=64 | 96 KB (tight, often viable with double-buffering) |
| Br=128, Bc=128 | 128 KB (exceeds L1 — not valid without tighter buffering) |

---

## Sliding Window Attention

Some model architectures restrict each token's attention to a local window of the last N tokens rather than the full sequence. Examples include Mistral 7B's sliding window attention and various vision attention modules with spatial locality.

The tt-transformers FlashAttention-2 kernel supports this via the `sliding_window_size` parameter:

```python
output = ttnn.transformer.scaled_dot_product_attention(
    query, key, value,
    is_causal=True,
    scale=None,
    sliding_window_size=4096,   # Each token attends only to the last 4096 tokens
    program_config=config,
    compute_kernel_config=compute_cfg,
)
```

With `sliding_window_size=W`, KV blocks more than W positions before the current Q chunk's start are skipped entirely — the same block-skip logic used for fully-masked causal blocks, but with an additional lower-bound condition. For vision attention (non-causal, fixed patch grid), the window is typically set to match the local receptive field of the attention pattern.

This eliminates KV loads for out-of-window tokens and reduces both DRAM traffic and FPU compute proportionally to `W/S`. For W=4096, S=32768 (long-context models), this is a 8× reduction in effective KV footprint per attention call.

---

## Key Takeaways

- Naive attention writes an O(S²) score matrix to DRAM; FlashAttention-2 tiles the computation so all intermediates stay in L1 (120 KB per core), eliminating those writes entirely and turning the kernel from memory-bandwidth-bound to compute-bound.
- Causality-aware load balancing pairs Q_low (sparse, few valid KV blocks) and Q_high (dense, many valid KV blocks) chunks on the same core, achieving ~1.6× speedup over naive sequential Q assignment by equalizing per-core work.
- Fully-masked KV blocks (above the causal diagonal) are detected at the block level and skipped without issuing any FPU compute or DRAM loads for those blocks.
- Double-buffering via TTNN circular buffers (2-slot CBs for Q, K, V) overlaps BRISC's DMA prefetch of the next tile with TRISC1/2's compute on the current tile, hiding DRAM latency behind FPU throughput.
- Speedup over naive DRAM-based attention is ~20× at S=2048, D=128; the range is 9×–44× across S=512–16K and D=64/128/256, growing with S because FlashAttention-2 scales linearly in S while naive attention scales quadratically.
- `sliding_window_size` enables local-window attention with the same block-skip mechanism; for W≪S this produces proportional reductions in both DRAM traffic and FPU compute.

---

## Further Reading

- Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023) — the algorithmic foundation for the online softmax update and Q-parallel tiling
- TT-Metal SDPA kernel source: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/` — Reader, Writer, and Compute kernel implementations in C++ and Metalium
- [tensix_architecture.md](../ch1_hardware_and_ttnn_foundations/tensix_architecture.md) — RISC-V processor roles, circular buffer model, L1/DRAM capacity reference
- [flash_decode_and_gqa.md](./flash_decode_and_gqa.md) — how the single-token decode path differs from this prefill path in its parallelization axis

---

**Next:** [`flash_decode_and_gqa.md`](./flash_decode_and_gqa.md)
