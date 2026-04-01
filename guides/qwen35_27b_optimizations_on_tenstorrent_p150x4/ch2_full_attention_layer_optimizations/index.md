# Chapter 2: Full Attention Layer Optimizations

Qwen3.5-27B uses full multi-head attention in 16 of its 64 layers (every 4th layer), and those layers carry five architectural differences from standard transformer attention that require custom Tenstorrent-specific handling: partial RoPE over 64 of 256 head dimensions, QK L2 normalization with learned scales, sigmoid output gating, a fused Q+gate projection, and separate K/V projections.

## Files

| File | Description |
|------|-------------|
| [`attention_architecture.md`](./attention_architecture.md) | Five architectural differences from standard attention, partial RoPE setup, and QK L2 normalization |
| [`dram_sharded_decode.md`](./dram_sharded_decode.md) | DRAM-sharded matmul configuration for decode projections, the `_shard_linear` pattern, and KV cache updates |
| [`flash_attention_prefill.md`](./flash_attention_prefill.md) | 2D matmul for prefill projections, flash SDPA configuration, and the complete prefill forward pass |

## Process Files

The following files are internal pipeline artifacts and are not part of the guide narrative:

- `b_review.md` — reviewer notes and accuracy checks for this chapter
- `compression_analysis.md` — token-budget analysis used during content compression

---

**Next:** [`attention_architecture.md`](./attention_architecture.md)
