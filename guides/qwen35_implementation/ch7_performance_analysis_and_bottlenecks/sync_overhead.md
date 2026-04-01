# Sync Overhead

A host-device synchronisation event is any operation that forces the host CPU to wait
for the Blackhole device to complete work before the host can continue. On P100A,
each such event adds roughly 1–2 ms of latency. At ~81 syncs per A3B decode step,
the cumulative sync cost is approximately 35 ms of the 86 ms total.

This file catalogues every sync source in the forward pass and its root cause.

## Sync sources by component

### 1. DeltaNet host recurrence (host-recurrence era)

**Ops:** `ttnn.to_torch` (read state → host) + `ttnn.from_torch` (write updated state → device)
**Count:** 1 pair per DeltaNet layer = 30 syncs total (A3B), 48 syncs (27B)
**Root cause:** The DeltaNet recurrence must run in float32 (Chapter 2, `host_recurrence.md`).
Blackhole's SrcB register is 19-bit TF32, so bf16 element-wise ops on fp32 circular buffers
hang the device. Running the recurrence on-device in bf16 introduces unacceptable numerical
error across 30+ layers. The only correct path is host float32, which requires a round-trip
per layer per token.

**Resolution:** The fused `ttnn.experimental.gated_delta_net` kernel maintains the recurrent
state as a device float32 tensor and updates it in-place via `ttnn.copy`. This eliminates
all 30 DeltaNet syncs. Key deployment facts:

- PCC = 0.999997 vs the reference host-recurrence implementation
- The kernel uses fp32 L1 SRAM accumulators, which are not affected by the SrcB TF32 hang
  that prevents standard bf16 element-wise ops on fp32 circular buffers
- In-place `ttnn.copy` preserves tensor device addresses — a prerequisite for Metal Trace
  replay without re-allocation (Chapter 2, `fused_kernel.md`)

### 2. Attention layer syncs

**Ops (host-RoPE era):** `partial_rope_fn` host round-trip; post-layer barrier
**Count:** ~50 syncs total (5 per attention layer × 10 layers)
**Root cause — `partial_rope_fn`:** Before the device-RoPE fix, `GatedAttention`
used a `custom_rope_fn` that applied partial rotation on the host:

```python
def partial_rope_fn(q_tt, k_tt, current_pos):
    pos = ttnn.to_torch(current_pos).int().item()  # sync 1: read position
    q   = ttnn.to_torch(q_tt).float()              # sync 2: read q tensor
    k   = ttnn.to_torch(k_tt).float()              # sync 3: read k tensor
    # ... rotation math on host ...
    q_out = ttnn.from_torch(q, ...)                # sync 4: write q back
    k_out = ttnn.from_torch(k, ...)                # sync 5: write k back
    return q_out, k_out
```

This is the source of 5 syncs per attention layer × 10 layers = 50 syncs. PERF.md labels
this "10+40" (10 post-layer barriers + 40 from RoPE round-trips, counting 4 blocking events
per layer since `from_torch` DMA may not stall the host).

**Root cause note — `HfRotarySetup.get_rot_mats`:** This method does NOT sync.
It returns the pre-cached `[cos_matrix, sin_matrix]` device tensors unchanged,
regardless of `position_idxs`. The position index is consumed internally by
`ttnn.experimental.rotary_embedding` on device.

**Resolution:** The `GatedAttention.__init__` was updated to use patched cos/sin matrices
with correct partial-rotation frequencies (see Chapter 3, `partial_rope.md`). The
`custom_rope_fn` path is no longer installed. This eliminates all 5 per-layer syncs,
reducing attention syncs from ~50 to ~10 (post-layer barriers only).

### 3. MoE router (per MoE layer)

**Ops:** `ttnn.to_torch(logits)` to bring routing logits to host
**Count:** 1 per MoE layer = 40 syncs (A3B; all 40 layers have MoE)
**Transferred data:** 256 × 2 bytes (bf16) = 512 bytes per sync
**Root cause:** `torch.topk` and `torch.softmax` on 256 values are not bottlenecked by
compute but by the lack of device-side topk with float32 accuracy guarantees. The 512-byte
DMA is negligible; the 1–2 ms sync latency is the actual cost.

These MoE syncs are already counted within the DeltaNet and Attention rows in the
profiling table because they occur inside the MoE MLP step of each decoder block.
The table counts per-layer-type syncs, not per-sublayer syncs.

### 4. norm + LM head

**Ops:** `ttnn.synchronize_device` once after `ttnn.linear(x, lm_weight_tt)`;
`ttnn.to_torch(logits_tt)` to retrieve logits
**Count:** 1 explicit sync + 1 implicit to_torch = effectively 1 counted sync
**Root cause:** The LM head produces logits over 248,320 vocabulary entries. Transferring
the full logit tensor to host for argmax is unavoidable; 248,320 × 2 bytes = ~484 KB per
step. The `ttnn.synchronize_device` call in the demo wraps this into a single measured
barrier.

## Sync cost summary

| Source | Syncs (profiling era) | Syncs (current: fused kernel + device RoPE) |
|---|---|---|
| DeltaNet recurrence | 30 | 0 |
| Attention layers (partial_rope_fn) | ~50 | ~10 (post-layer only) |
| norm + LM head | 1 | 1 |
| **Total** | **~81** | **~11** |

*PERF.md's Total row states "~70"; the individual row values sum to 30+50+1=81.*

With both the fused kernel and device-side partial RoPE deployed, the sync count drops
from ~81 to ~11 per token. The remaining syncs are 10 post-attention-layer barriers
and 1 LM-head sync, all of which require Metal Trace to eliminate (see `bottleneck_analysis.md`).

For Metal Trace's role in eliminating remaining syncs, see `bottleneck_analysis.md`
(Bottleneck 1, Fix — Metal Trace).

---

**Next:** [`bottleneck_analysis.md`](./bottleneck_analysis.md)
