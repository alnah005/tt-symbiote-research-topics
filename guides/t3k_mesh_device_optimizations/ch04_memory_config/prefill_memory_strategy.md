# Memory Placement Strategy: Prefill Phase

## Overview

The prefill phase processes the full prompt in a single forward pass. Its defining characteristics are:

- **Long sequences**: $S \in [512, 32768]$ tokens processed simultaneously, producing all KV cache entries for the prompt in one pass.
- **Large activation tensors**: shape $[B, S, H]$ scales linearly with sequence length. At $S=2048$, $H=7168$, $B=1$: $2048 \times 7168 \times 2 = 29.4\,\text{MB}$ — 24% of the entire chip's aggregate L1, just for a single activation buffer.
- **Throughput-critical but not as latency-sensitive as decode**: prefill happens once per prompt. Per-token generation throughput is more important than minimizing individual forward pass latency.
- **Memory-bandwidth intensive**: large matmuls over long sequences stress DRAM bandwidth. The goal is to maximize effective memory throughput while staying within L1 budget for the tensors that benefit from it.

The fundamental difference from decode: **most activation tensors are too large for L1 at long sequences**. DRAM becomes the natural placement for activations, and L1 is reserved for tensors that remain small (e.g., short-sequence attention intermediates in chunked prefill).

---

## Activation Tensors — DRAM

**Recommendation: `ttnn.DRAM_MEMORY_CONFIG` with `TensorMemoryLayout.INTERLEAVED` for $B \cdot S > 2{,}880$**

The full hidden state tensor shape is $[B, S, H]$. Viewed as a 2D matrix $[B \cdot S, H]$ for matmul operations:

| B | S | H | Total Size (BF16) | Total vs. Chip L1 |
|---|---|---|---|---|
| 1 | 512 | 7168 | 7.3 MB | 6.1% of 120 MB |
| 1 | 2048 | 7168 | 29.4 MB | 24.5% |
| 4 | 2048 | 7168 | 117.4 MB | 97.9% |
| 1 | 8192 | 7168 | 117.4 MB | 97.9% |
| 32 | 2048 | 7168 | 939.5 MB | 782.9% (does not fit) |

Even at $B=4$, $S=2048$, the activation tensor nearly exhausts the entire chip's aggregate L1. Multiple activation buffers must coexist during a layer's forward pass (input, output, residual), making the total L1 demand several multiples of the per-tensor size. DRAM is the only viable placement.

```python
import ttnn

# Prefill hidden state — DRAM for any practical sequence length
hidden_states = ttnn.from_torch(
    input_tensor,               # shape [B, S, H]
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,  # mandatory for B*S > 2880
    dtype=ttnn.bfloat16,
)
```

---

## Attention Intermediate Buffers (Q, K, V Projections) — Conditional

**Short sequences ($B \cdot S \leq 2{,}880$): L1 HEIGHT_SHARDED**
**Long sequences ($B \cdot S > 2{,}880$): DRAM interleaved**

The Q projection has shape $[B \cdot S, H_Q]$. For HEIGHT_SHARDED across 80 cores:

$$\text{per-core rows} = \left\lceil \frac{B \cdot S}{80} \right\rceil$$

$$\text{per-core bytes} = \left\lceil \frac{B \cdot S}{80} \right\rceil \times H_Q \times 2$$

The crossover point where the per-core shard fits in 1.5 MB:

$$\left\lceil \frac{B \cdot S}{80} \right\rceil \times H_Q \times 2 \leq 1{,}572{,}864$$

For $H_Q = H = 7168$:

$$\left\lceil \frac{B \cdot S}{80} \right\rceil \leq \frac{1{,}572{,}864}{7168 \times 2} = 109.7$$

So a single Q (or K or V) tensor shard fits as long as $B \cdot S \leq 80 \times 109 = 8{,}720$. However, Q, K, and V projections coexist in L1 simultaneously during attention, tripling the budget: $3 \times \lceil B \cdot S / 80 \rceil \times H_Q \times 2 \leq 1{,}572{,}864$ gives $B \cdot S \leq 80 \times 36 = 2{,}880$. The threshold $B \cdot S \leq 2{,}880$ accounts for Q+K+V occupying L1 concurrently.

For $B=1$, $S=2048$: $\lceil 2048/80 \rceil = 26$ rows/core per tensor; $3 \times 26 \times 7168 \times 2 = 1{,}118{,}208$ bytes = 1.07 MB/core — fits within 1.5 MB.

```python
import ttnn
import math

def attention_memory_config(
    B: int,
    S: int,
    H_Q: int,
    n_cores: int = 80,
) -> ttnn.MemoryConfig:
    """
    Select memory config for Q/K/V projection output.
    Returns L1 HEIGHT_SHARDED for short sequences, DRAM for long.
    """
    L1_PER_CORE = 1.5 * 1024 * 1024  # bytes
    rows_total = B * S
    rows_per_core = math.ceil(rows_total / n_cores)
    shard_bytes = rows_per_core * H_Q * 2  # BF16

    if 3 * shard_bytes <= L1_PER_CORE:  # Q, K, and V must all fit simultaneously
        # Build HEIGHT_SHARDED L1 config
        shard_spec = ttnn.ShardSpec(
            grid=ttnn.CoreRangeSet({
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 7))
            }),
            shape=[rows_per_core, H_Q],
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        return ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
            shard_spec=shard_spec,
        )
    else:
        return ttnn.DRAM_MEMORY_CONFIG


# Example: B=1, S=2048 -> B*S=2048 <= 2880 -> L1 HEIGHT_SHARDED (~364 KB/core single tensor: ceil(2048/80)=26 rows × 7168 × 2 = 372736 bytes)
# Example: B=4, S=2048 -> B*S=8192 > 2880 -> DRAM INTERLEAVED
q_memory_config = attention_memory_config(B=1, S=2048, H_Q=7168)
print(q_memory_config.buffer_type)  # L1 for B=1, S=2048

q_memory_config_long = attention_memory_config(B=4, S=2048, H_Q=7168)
print(q_memory_config_long.buffer_type)  # DRAM for B=4, S=2048
```

---

## All-to-All Dispatch and Combine Buffers — DRAM (Prefill)

**Recommendation: `ttnn.DRAM_MEMORY_CONFIG` — mandatory for any practical prefill configuration**

The all-to-all dispatch buffer volume scales with expert capacity $C$, which grows linearly with $B \cdot S$:

$$C = \left\lceil \frac{k \cdot B \cdot S}{E} \right\rceil$$

$$V_{\text{dispatch}} = C \times E_d \times H \times 2\,\text{bytes per device per direction}$$

### All-to-All Buffer Sizing Table

Parameters: $k=8$, $E=256$, $E_d=32$, $H=7168$, BF16 (2 bytes/element).

| B | S | $C = \lceil 8BS/256 \rceil$ | Buffer per Device per Direction |
|---|---|---|---|
| 1 | 2048 | $\lceil 64 \rceil = 64$ | $64 \times 32 \times 7168 \times 2 = 29.4\,\text{MB}$ |
| 4 | 2048 | $\lceil 256 \rceil = 256$ | $256 \times 32 \times 7168 \times 2 = 117.4\,\text{MB}$ |
| 32 | 2048 | $\lceil 2048 \rceil = 2048$ | $2048 \times 32 \times 7168 \times 2 = 939.5\,\text{MB}$ |

All of these exceed the chip's 120 MB aggregate L1 by large margins. The $B=4$, $S=2048$ case alone requires 117.4 MB per direction — nearly the full aggregate L1 budget just for one direction of one all-to-all operation, with no room for any other tensors.

```python
import ttnn
import math

# All-to-all buffer sizing formula
def alltoall_buffer_bytes(B: int, S: int, k: int = 8, E: int = 256,
                           E_d: int = 32, H: int = 7168) -> int:
    C = math.ceil(k * B * S / E)
    return C * E_d * H * 2  # BF16 bytes

# Prefill all-to-all — always DRAM for realistic S
# The L1 rule for decode (C=1) does NOT apply here
for B, S in [(1, 2048), (4, 2048), (32, 2048)]:
    vol_mb = alltoall_buffer_bytes(B, S) / 1_000_000
    C = math.ceil(8 * B * S / 256)
    print(f"B={B:2d} S={S}: C={C:5d}, buffer={vol_mb:.1f} MB/device/direction")

# Output:
# B= 1 S=2048: C=   64, buffer=29.4 MB/device/direction
# B= 4 S=2048: C=  256, buffer=117.4 MB/device/direction
# B=32 S=2048: C= 2048, buffer=939.5 MB/device/direction

# Prefill dispatch — mandatory DRAM
dispatch_output = ttnn.all_to_all(
    dispatch_input,
    cluster_axis=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,  # non-negotiable for prefill
)
```

> **Warning:** Do not attempt to use `ttnn.L1_MEMORY_CONFIG` for all-to-all buffers in the prefill phase. Even at $B=1$, $S=2048$, the buffer is 29.4 MB — 24.5% of aggregate L1. With both dispatch and combine buffers plus activation tensors, the chip would instantly exceed its L1 capacity.

---

## KV Cache Generation During Prefill — DRAM

**Recommendation: `ttnn.DRAM_MEMORY_CONFIG`**

During prefill, the KV cache is written incrementally layer by layer. The output shape per layer is $[B, n_{KV}, S, H_{KV}]$ for keys and the same for values. This is written once per layer and persists into the decode phase.

Since the KV cache must persist into decode (which also uses DRAM for KV cache), there is no reason to use L1 at any point. Write directly to DRAM during prefill, and the decode phase reads from the same buffer with no migration overhead.

```python
import ttnn

# KV cache written during prefill — DRAM, shared with decode
# No migration needed between phases if both use DRAM
k_cache = ttnn.matmul(
    hidden_states,      # [B, S, H], DRAM
    wk,                 # weight [H, H_KV], DRAM
    memory_config=ttnn.DRAM_MEMORY_CONFIG,  # write to DRAM; decode reads here
)
v_cache = ttnn.matmul(
    hidden_states,
    wv,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

---

## Chunked Prefill Strategy

For very long sequences ($S > 8192$) where DRAM bandwidth becomes a bottleneck for activation tensors, a **chunked prefill** strategy splits the sequence into smaller chunks, enabling L1-resident computation for each chunk.

### Strategy

Process the prompt as a sequence of $n_{\text{chunks}}$ chunks, each of $S_{\text{chunk}}$ tokens:

$$n_{\text{chunks}} = \left\lceil \frac{S}{S_{\text{chunk}}} \right\rceil$$

For each chunk, the activation tensor is $[B, S_{\text{chunk}}, H]$. Choose $S_{\text{chunk}}$ such that the per-core shard fits in L1:

$$S_{\text{chunk}} = n_{\text{cores}} \times \left\lfloor \frac{L1_{\text{per-core}}}{H \times 2 \times \text{tiles per row}} \right\rfloor$$

A practical formula for aligning to matmul tile requirements (tile size = 32):

$$S_{\text{chunk}} = n_{\text{cores}} \times M_{\text{per-core}} \times 32$$

where $M_{\text{per-core}}$ is the number of tile rows per core in the target matmul program config.

```python
import ttnn
import math

def chunked_prefill(
    input_ids: list,              # full prompt token IDs
    model,
    mesh_device,
    S_chunk: int = 256,           # chunk size (tune for L1 fit)
    H: int = 7168,
    n_cores: int = 80,
) -> ttnn.Tensor:
    """
    Process a long prompt in chunks to fit activations in L1 per chunk.
    KV cache is accumulated in DRAM across chunks.
    """
    S_total = len(input_ids)
    kv_caches = [None] * model.n_layers  # will hold DRAM KV cache tensors

    # Verify chunk fits in L1
    L1_PER_CORE = 1.5 * 1024 * 1024
    per_core_bytes = math.ceil(S_chunk / n_cores) * H * 2
    assert per_core_bytes <= L1_PER_CORE * 0.8, (
        f"Chunk too large: {per_core_bytes/1024:.0f} KB/core > 80% of {L1_PER_CORE/1024:.0f} KB"
    )

    for chunk_start in range(0, S_total, S_chunk):
        chunk_end = min(chunk_start + S_chunk, S_total)
        chunk_ids = input_ids[chunk_start:chunk_end]

        # Embed chunk — can use L1 HEIGHT_SHARDED since chunk is small
        chunk_hidden = model.embed(chunk_ids, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Process all transformer layers for this chunk
        # KV cache for positions [chunk_start:chunk_end] is written to DRAM
        chunk_hidden, kv_caches = model.forward_chunk(
            chunk_hidden,
            kv_caches=kv_caches,
            position_offset=chunk_start,
            activation_memory_config=ttnn.L1_MEMORY_CONFIG,   # small chunk fits
            kv_cache_memory_config=ttnn.DRAM_MEMORY_CONFIG,   # persistent
        )

    # Final hidden state for the last chunk token (used to generate first decode token)
    return chunk_hidden[:, -1:, :]  # [B, 1, H]
```

### Trade-offs of Chunked Prefill

| Aspect | Full-Sequence Prefill | Chunked Prefill |
|---|---|---|
| L1 usage per forward pass | High (often all DRAM) | Low (fits per chunk) |
| DRAM reads/writes | One pass per layer | $n_{\text{chunks}}$ passes per layer |
| Kernel launch overhead | Lower | Higher ($\times n_{\text{chunks}}$) |
| Causal attention correctness | Natural (full context per layer) | Requires careful KV cache accumulation |
| Throughput (long sequences) | Lower (DRAM bandwidth limited) | Higher (L1 compute paths) |
| Implementation complexity | Low | Medium |

> **Tip:** Chunked prefill is most beneficial when $S > 4096$ and the model is throughput-constrained (multiple requests in flight). For single-request, low-latency prefill of moderate-length prompts ($S \leq 2048$), full-sequence DRAM prefill is simpler and adequate.

---

## Prefill-to-Decode Memory Layout Transition

After prefill completes, the inference loop transitions to decode. No explicit tensor migration is needed for the KV cache if both phases use DRAM:

```
Prefill:  K = matmul(hidden, Wk, memory_config=DRAM_MEMORY_CONFIG)  → DRAM
Decode:   scores = attention(Q, K_from_cache)  → reads K from DRAM directly
```

Activation tensors are **not carried across the boundary** — they are recomputed each step (decode recomputes hidden state from scratch each step from the token embedding). The layout switch from DRAM activations (prefill) to L1 activations (decode) is automatic: call the same ops with different `memory_config` arguments depending on the phase.

```python
import ttnn

def get_memory_config(phase: str, tensor_type: str) -> ttnn.MemoryConfig:
    """
    Return the correct memory config for a tensor in the current phase.
    """
    table = {
        # (phase, tensor_type) -> MemoryConfig
        ("decode", "hidden_state"):    ttnn.L1_MEMORY_CONFIG,
        ("decode", "routing_scores"):  ttnn.L1_MEMORY_CONFIG,
        ("decode", "dispatch_buffer"): ttnn.L1_MEMORY_CONFIG,
        ("decode", "kv_cache"):        ttnn.DRAM_MEMORY_CONFIG,
        ("prefill", "hidden_state"):   ttnn.DRAM_MEMORY_CONFIG,
        ("prefill", "routing_scores"): ttnn.L1_MEMORY_CONFIG,   # still small
        ("prefill", "dispatch_buffer"): ttnn.DRAM_MEMORY_CONFIG,
        ("prefill", "kv_cache"):       ttnn.DRAM_MEMORY_CONFIG,
    }
    return table[(phase, tensor_type)]
```

---

## Summary: Prefill Memory Placement

| Tensor | Shape (B=1, S=2048) | Total Size (BF16) | Recommendation |
|---|---|---|---|
| Hidden state activations | `[B·S, H]` = `[2048, 7168]` | 29.4 MB | DRAM interleaved |
| Q/K/V projections (B·S≤2880) | e.g. `[2048, H]` at B=1 | 29.4 MB | L1 HEIGHT_SHARDED (~1.1 MB/core for Q+K+V) |
| Q/K/V projections (B·S>2880) | e.g. `[8192, H]` (B=4, S=2048) | 117.4 MB | DRAM interleaved |
| KV cache (per layer) | `[B, n_KV, S, H_KV]` | Model-dependent | DRAM always |
| All-to-all dispatch buffer | `[C·E_d, H]` (C=64 at B=1,S=2048) | 29.4 MB | DRAM always |
| All-to-all combine buffer | Same as dispatch | 29.4 MB | DRAM always |
| Routing scores | `[B·S, E]` = `[2048, 256]` | 1.0 MB | DRAM (or L1 if S small) |

For reference, see the all-to-all buffer sizing formula and table in the All-to-All Dispatch and Combine Buffers section above.

---

## References

- `wormhole_memory_hierarchy.md` — L1 capacity, DRAM interleaving, shard layout mechanics
- `memory_config_api.md` — `ttnn.MemoryConfig`, `TensorMemoryLayout` variants, `ShardSpec`
- `decode_memory_strategy.md` — Contrast with decode (L1-dominant) placement strategy
- Chapter 3 — `ch03_all_to_all_num_links/` — All-to-all bandwidth and buffer sizing background
- Tenstorrent technical note: chunked prefill performance on T3K (internal)
- TT-NN source: `ttnn/cpp/ttnn/operations/transformer/` — transformer op `memory_config` support

---

**Next:** [Chapter 5 — Expert Parallelism](../ch05_expert_parallelism/index.md)
