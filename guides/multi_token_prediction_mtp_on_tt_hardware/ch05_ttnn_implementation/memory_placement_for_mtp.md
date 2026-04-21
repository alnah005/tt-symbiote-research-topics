# Memory Placement for MTP Head Tensors

## MTP Head Weights

The MTP head weights total approximately 304.6 MiB in BF16 (see Chapter 2, `ch02_mtp_weights_and_memory/`). These cover the norm weights, attention projections, and dense FFN projections for the single MTP transformer block loaded under the `model.future_prediction[0].*` key prefix. The shared `lm_head` is excluded from this count as it is already allocated by the backbone.

**Wormhole L1 capacity**: Aggregate L1 SRAM across all Tensix cores on a Wormhole chip is approximately 108 MiB total. The 304.6 MiB MTP weight set does not fit in L1 by a factor of ~2.8×.

**Placement: DRAM interleaved**, consistent with backbone weight placement. All MTP block weight tensors are allocated with `ttnn.DRAM_MEMORY_CONFIG` (interleaved layout). No tiling or sharding across L1 is attempted for weights.

> **Key Finding:** The MTP head weights are ~0.4% of the backbone's ~70 GiB. Streaming them from DRAM adds approximately 304.6 MiB / 288 GB/s ≈ 1.06 ms per MTP head forward pass. Using the Chapter 4 (`ch04_speculative_decoding_with_mtp/`) P150 baseline of C_decode ≈ 243 ms: 1.06 ms / 243 ms ≈ 0.4% overhead — consistent with the Chapter 4 assessment that MTP head cost ≈ 0 in the BW-bound regime.

## Activation Tensors During MTP Head Forward

Activations produced during the MTP head forward pass are small and short-lived. All fit comfortably in L1.

### `backbone_hidden_state`

- Shape: `[1, 1, 4096]` at batch=1 (H = 4096 for Qwen3.6-35B-A3B)
- Size: 1 × 1 × 4096 × 2 bytes (BF16) = **8 KiB**
- Placement: **L1**
- This tensor is produced by the backbone's final layer norm in Step 1 and is passed directly to `TTNNMTPHead.forward()`. It is consumed once and discarded.

### `x_t1_embedding`

- Shape: `[1, 1, 4096]`
- Size: **8 KiB**
- Placement: **L1**
- Produced by `embed_tokens(x_t1)` immediately before the MTP head call. Short-lived.

### MTP Block Intermediate Buffers

During the single MTP transformer block forward:

- Attention K/V projections for the single input token: `[1, num_kv_heads, 1, head_dim]` = `[1, 8, 1, 128]` → 1 × 8 × 1 × 128 × 2 bytes = **2 KiB** each (K and V)
- Attention scores and weighted sum: similar KiB-range tensors
- FFN intermediate activations: `[1, 1, intermediate_size]` — intermediate_size for the dense FFN in Qwen3.6-35B-A3B is typically in the range 11008–14336, giving ~22–28 KiB per gate/up/down buffer

All intermediate activations fit in L1. Placement: **L1** for all transient buffers.

### `draft_logits`

- Shape: `[1, 1, 151936]` (vocab_size = 151936 for Qwen3.6-35B-A3B)
- Size: 1 × 1 × 151936 × 2 bytes = **~290 KiB**
- Placement: **L1 if L1 budget allows, otherwise DRAM**
- `draft_logits` is consumed immediately in Step 4 (acceptance check) and not retained across cycles. If L1 pressure from backbone activations is tight at the time of the MTP forward pass, it can be written to DRAM with negligible additional latency given the small size.

## KV Cache for the MTP Head

The single attention layer inside the MTP transformer block requires its own KV cache entry to maintain causally correct attention across the generation sequence. Without a KV cache, the MTP block would need to re-attend over the full sequence history on every call, which is prohibitively expensive.

**Sizing** (Qwen3.6-35B-A3B, `max_seq_len = 32768`):

```text
2 (K + V) × num_kv_heads × max_seq_len × head_dim × bytes_per_element
= 2 × 8 × 32768 × 128 × 2
= 134,217,728 bytes
≈ 128 MiB
```

**Placement: DRAM interleaved**. At 128 MiB, the MTP KV cache cannot fit in the 108 MiB total L1 budget. It must reside in DRAM.

**Allocation**: The MTP KV cache can share the same DRAM buffer pool used for backbone KV cache entries. Treat it as one additional "layer" slot in the existing KV cache allocator. The advance logic (1 or 2 positions per cycle) mirrors the backbone KV cache advance exactly, as described in `speculative_decode_loop_integration.md`.

> **Key Finding:** The MTP KV cache adds 128 MiB of DRAM at `max_seq_len = 32768`. This is equivalent to one backbone KV cache layer (the backbone has 94 layers; the total backbone KV cache is approximately 94 × 128 MiB ≈ 12 GiB at the same settings). The MTP KV cache is a ~0.8% increase in total KV cache DRAM. No new DRAM management infrastructure is required.

## Recommendation Table

| Tensor | Shape (batch=1) | Size | Recommended Placement | Rationale |
|---|---|---|---|---|
| MTP head weights (all) | various | 304.6 MiB | DRAM interleaved | Does not fit in L1 (108 MiB total) |
| `backbone_hidden_state` | `[1, 1, 4096]` | 8 KiB | L1 | Always fits; consumed once on hot path |
| `x_t1_embedding` | `[1, 1, 4096]` | 8 KiB | L1 | Always fits; consumed once on hot path |
| MTP block attention buffers | `[1, 8, 1, 128]` per K/V | ~2 KiB each | L1 | KiB-range; all transient |
| MTP block FFN buffers | `[1, 1, ~12288]` | ~22–28 KiB each | L1 | KiB-range; all transient |
| `draft_logits` | `[1, 1, 151936]` | ~290 KiB | L1 (if budget) or DRAM | Temporary; consumed immediately |
| MTP KV cache | `[1, 8, 32768, 128]` | ~128 MiB | DRAM interleaved | Too large for L1; one additional layer slot |

## L1 Pressure Note

Enabling MTP does **not** significantly increase L1 pressure:

- All MTP weights go to DRAM — no L1 weight pinning.
- All MTP activations are KiB-range and short-lived. Total peak L1 usage from MTP-specific tensors (excluding `draft_logits`) is well under 1 MiB.
- `draft_logits` at ~290 KiB is the largest transient activation. If L1 is already heavily occupied by backbone activations at the time of the MTP pass, a DRAM spill for `draft_logits` adds approximately 290 KiB / 288 GB/s ≈ 0.001 ms — negligible.

The MTP head forward pass is not L1-pressure-limited. The dominant memory cost is DRAM bandwidth for streaming 304.6 MiB of weights, which is already accounted for in the ~1 ms MTP head latency estimate.

## References

- Chapter 2: `ch02_mtp_weights_and_memory/` — MTP weight count, BF16 footprint, key prefix `model.future_prediction[0].*`
- Chapter 4: `ch04_speculative_decoding_with_mtp/` — BW-bound regime analysis, MTP head cost ≈ 0 claim, 288 GB/s bandwidth figure
- Chapter 5: `mtp_head_ttnn_module.md` — Weight loading and `lm_head` sharing
- Chapter 5: `speculative_decode_loop_integration.md` — KV cache advance logic (1 or 2 per cycle)
