# Chapter 3: GDN Layer Decode Pipeline

Gated DeltaNet (GDN) layers make up 48 of the 64 layers in Qwen3.5-27B. Unlike the full attention layers covered in Chapter 2, GDN layers replace the KV cache with a fixed-size recurrence state tensor and use a linear attention recurrence instead of softmax-based scaled dot-product attention. This makes them constant-memory per token during decode, but introduces a different set of implementation challenges on the Tenstorrent hardware: a causal conv1d that must be trace-compatible, a multi-step recurrence with strict numerical precision requirements, and a head expansion from 4 key heads to 12 value heads per device.

This chapter walks through the complete decode forward pass implemented in the `TtGatedDeltaNet` class (`gdn.py`), covering both the fused path (`_forward_decode_fused`) that dispatches a single custom kernel and the unfused fallback path (`_forward_decode_unfused`) that decomposes the recurrence into individual `ttnn` operations. The fused kernel internals are deferred to Chapter 4; this chapter focuses on the surrounding pipeline and the mathematical operations it performs.

## Learning Objectives

After reading this chapter you will understand:

- The end-to-end dataflow of a single GDN decode step, from input projection through all-reduce output
- How the 4-tap causal conv1d is implemented as a trace-compatible shift register using `ttnn.copy` chains
- The DeltaNet recurrence equations and how each maps to tensor operations on device
- How key heads are expanded to value heads via `repeat_interleave` with a repeat factor of 3
- The role of L2 normalization, sigmoid/softplus gating, and SiLU output gating in the pipeline
- Memory management discipline: when and why tensors are deallocated throughout the forward pass

## Files

| File | Description |
|------|-------------|
| [`gdn_decode_flow.md`](./gdn_decode_flow.md) | End-to-end decode dataflow for a single GDN layer, covering both fused and unfused paths |
| [`conv1d_shift_register.md`](./conv1d_shift_register.md) | The 4-tap causal conv1d implemented as a trace-compatible shift register |
| [`recurrence_math.md`](./recurrence_math.md) | DeltaNet recurrence equations and their mapping to tensor operations |

See [`gdn_decode_flow.md`](./gdn_decode_flow.md) to begin.

---

**Next:** [`gdn_decode_flow.md`](./gdn_decode_flow.md)
