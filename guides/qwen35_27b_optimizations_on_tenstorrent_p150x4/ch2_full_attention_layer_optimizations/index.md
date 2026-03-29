# Chapter 2: Full Attention Layer Optimizations

Qwen3.5-27B uses full multi-head attention in 16 of its 64 layers (every 4th layer). These attention layers are not standard transformer attention -- they incorporate five architectural modifications that require custom handling on the Tenstorrent hardware: partial RoPE over 64 of 256 head dimensions, QK L2 normalization with learned scales, sigmoid output gating, a fused Q+gate projection, and separate K/V projections.

This chapter covers the `Qwen35Attention` class in `attention.py` and the supporting infrastructure in `rope.py` and `model_config.py`, explaining how each architectural feature is implemented and how the decode and prefill paths use different matmul strategies to maximize hardware utilization.

## Learning Objectives

After reading this chapter you will understand:

- The five ways Qwen3.5 attention differs from standard transformer attention and why each matters
- How partial RoPE is implemented via the `Qwen35PartialRopeSetup` class and the slice-rotate-concat pattern
- How DRAM-sharded matmuls achieve high bandwidth utilization for decode projections with M=1
- How 2D multicast matmuls and flash SDPA are configured for compute-bound prefill on an 8x8 grid
- The per-head KV cache update strategy using HEIGHT_SHARDED shard configs

## Files

| File | Description |
|------|-------------|
| [`attention_architecture.md`](./attention_architecture.md) | Five architectural differences from standard attention, partial RoPE setup, and QK L2 normalization |
| [`dram_sharded_decode.md`](./dram_sharded_decode.md) | DRAM-sharded matmul configuration for decode projections, the `_shard_linear` pattern, and KV cache updates |
| [`flash_attention_prefill.md`](./flash_attention_prefill.md) | 2D matmul for prefill projections, flash SDPA configuration, and the complete prefill forward pass |

See [`attention_architecture.md`](./attention_architecture.md) to begin.

---

**Next:** [`attention_architecture.md`](./attention_architecture.md)
