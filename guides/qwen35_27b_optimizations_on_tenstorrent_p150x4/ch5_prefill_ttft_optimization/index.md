# Chapter 5: Prefill TTFT Optimization

Prefill -- the phase where the model processes the full input prompt before generating tokens -- was the dominant contributor to time-to-first-token (TTFT) in the initial Qwen3.5-27B deployment. The baseline implementation processed each prompt token through every layer individually, dispatching per-token DRAM-sharded matmuls for projections and running the GDN recurrence one token at a time. For a 96-token prompt, this produced a TTFT of **498 ms/token** (47.8 seconds total).

The optimized prefill pipeline achieves **94 ms/token** (9.1 seconds for 96 tokens) -- a **5.3x speedup**. The key insight is separating the parallelizable parts of each layer from the inherently sequential parts. For both full attention and GDN layers, the weight projections (QKV, QKVZ, AB, output) are independent across tokens and can be computed in a single batched matmul over the entire sequence. Only the GDN recurrence -- where each token's state depends on the previous token's output -- must remain sequential.

This decomposition yields three categories of optimization:

1. **Batched projections** replace per-token DRAM-sharded matmuls with 2D multicast matmuls over the full sequence, shifting from bandwidth-bound to compute-bound execution.
2. **GDN prefill strategy** computes all projections once for the full sequence, then iterates per-token only for the conv1d shift register and fused recurrence kernel, using separate B=1 states.
3. **State replication** copies the B=1 prefill states (both GDN recurrence states and KV caches) to all B=32 decode slots after prefill completes, so batched decode can begin immediately.

## Performance Summary

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| TTFT per token | 498 ms | 94 ms | 5.3x |
| TTFT for 96-token prompt | 47.8 s | 9.1 s | 5.3x |
| Projection dispatch (per layer) | seq_len dispatches | 1 dispatch | seq_len x reduction |
| Output projection dispatch | seq_len dispatches | 1 dispatch | seq_len x reduction |

## Learning Objectives

After reading this chapter you will understand:

- How 2D multicast matmuls on an 8x8 compute grid replace per-token DRAM-sharded matmuls for prefill projections, and why this shifts the bottleneck from bandwidth to compute
- The hybrid GDN prefill strategy: batched projections followed by sequential per-token recurrence with B=1 states
- How `_init_prefill_states()` creates separate B=1 conv and recurrence state tensors that are independent of the B=32 decode states
- The post-prefill state replication mechanism that copies B=1 results into all B=32 decode slots for both GDN recurrence states and attention KV caches

## Files

| File | Description |
|------|-------------|
| [`batched_projections.md`](./batched_projections.md) | 2D matmul for compute-bound prefill projections in both attention and GDN layers |
| [`gdn_prefill_strategy.md`](./gdn_prefill_strategy.md) | Batched QKVZ + sequential per-token recurrence with B=1 states |
| [`state_replication.md`](./state_replication.md) | Post-prefill B=1 to B=32 state replication for KV cache and GDN states |

See [`batched_projections.md`](./batched_projections.md) to begin.

---

**Next:** [`batched_projections.md`](./batched_projections.md)
