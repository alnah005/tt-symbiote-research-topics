# Bottleneck Analysis

The 86 ms/token and 6.8% device efficiency are not from any single cause. Three
independent bottlenecks each consume roughly equal fractions of wall time, and each
requires a distinct remedy. This file analyses each bottleneck and the proposed fix.

## Bottleneck 1 — Python dispatch overhead (~26 ms)

**What it is:** Each TTNN op call (`ttnn.linear`, `ttnn.multiply`, `ttnn.slice`, …)
is a Python function that builds a command descriptor and enqueues it on the device
command queue. For a single A3B decode step, the Python interpreter executes
approximately:

- 40 layers × ~12 TTNN ops/layer (DeltaNet: projection, conv ops, fused kernel, out_proj;
  Attention: QKV, RoPE, SDPA, gate, out_proj; MoE: router fetch, gate+up, down, shared)
- 40 MoE router dispatches (topk/softmax on host, expert gather, scatter)
- RMSNorm and LM head ops

Total: roughly 500–600 Python-level TTNN calls per decode step. At ~40–50 µs overhead
per call (Python interpreter + command serialisation), this sums to ~25 ms.

**Why it dominates:** Device compute is fast (20 ms for all matrix math). Python dispatch
is serialised and single-threaded; it cannot be overlapped with device execution without
Metal Trace or asynchronous command queues.

**Fix — Metal Trace:** Metal Trace records all device commands during a "capture" pass
and replays the recorded binary stream on subsequent tokens. Python is bypassed entirely
for replay. The one-time capture cost (step 0 compile time) is paid once; all subsequent
tokens execute the trace with no Python overhead. This alone could reduce token latency
from ~86 ms to roughly ~50 ms (sync + compute, no dispatch).

## Bottleneck 2 — Host-device sync overhead (~35 ms)

See `sync_overhead.md` for the per-source breakdown. The summary:

- **Profiling era (host recurrence + host RoPE):** ~81 syncs/token ≈ 35 ms.
- **Current (fused kernel + device RoPE):** ~11 syncs/token (10 post-attention barriers + 1 LM head); sync overhead dramatically reduced but Python dispatch remains.

**Fix — Metal Trace (same as above):** With Trace, per-token sync is reduced to ~1
synchronisation event at the trace boundary (one `ttnn.synchronize_device` to read
the LM head output). The MoE router sync is absorbed into the trace replay.

**Fix — Multi-CQ overlap:** Without Trace, `ttnn.from_torch` writes (sending conv state,
position tensors, rot_mats) can be issued on command queue 1 (CQ1) while device compute
proceeds on CQ0. This hides DMA latency behind compute but does not eliminate sync
barriers at read-back points. Multi-CQ is complementary to Metal Trace.

## Bottleneck 3 — LM head on host (part of ~14 ms norm+head)

**What it is:** The vocabulary size is 248,320. The LM head weight matrix at bfp8 is:

```
248,320 × 5,120 × 1 byte ≈ 1.22 GiB
```

Placing this on device consumes ~1.2 GiB of the 28 GiB DRAM budget. The token embedding
table at float32 is 248,320 × 5,120 × 4 bytes ≈ 4.9 GiB — far too large for DRAM.
The current implementation keeps the embedding table on host CPU and performs the
embedding lookup in Python (one float32 vector copy per token, negligible compute).

The LM head is on device (`lm_weight_tt` in `demo_a3b.py`), but the argmax is performed
on host after `ttnn.to_torch(logits_tt)`. Transferring 248,320 × 2 bytes ≈ 484 KB per
step adds to the single sync cost counted in the norm+head row.

**Fix:** The LM head placement is already optimal — it is on device. The 14 ms
includes the norm (fast) and the LM head matmul (fast) plus the final sync that
reads out logits. Under Metal Trace the logit transfer is the only mandatory sync
and becomes the hard floor.

## Bottleneck 4 — DeltaNet recurrence (historical)

This was the dominant bottleneck before the fused kernel: each of the 30 DeltaNet layers
issued one `to_torch` (read ~2 MB fp32 state) + one `from_torch` (write updated state)
per token, contributing ~30 ms of the ~35 ms sync budget. The root cause, resolution,
and deployment details are in `sync_overhead.md` Section 1.

## Efficiency ceiling with current optimisations

| Scenario | Estimated latency | Estimated efficiency |
|---|---|---|
| Baseline (host recurrence + host RoPE) | 86 ms/token | 6.8% |
| Fused kernel + device RoPE (current) | ~50–60 ms/token | ~10–12% |
| Current + Metal Trace | ~20–25 ms/token | ~25–30% |
| Current + Trace + Multi-CQ | ~18–22 ms/token | ~27–33% |

The estimates for Trace assume Python dispatch (~26 ms) is eliminated entirely and sync
overhead drops from ~35 ms to ~2–3 ms (one sync for logit readout). Device compute
(~20 ms) is unchanged and becomes the dominant term.

## Why the fused kernel alone is insufficient

Even with zero DeltaNet syncs, the ~26 ms Python dispatch cost remains. The 40-layer
Python loop, MoE routing dispatches, and RoPE setup dispatches all execute in the Python
interpreter before each token. Metal Trace is required to bypass this path. This is why
PERF.md notes: *"Fused DeltaNet kernel ready (PCC 0.999997), blocked by from_torch overhead
without Metal Trace"* — the bottleneck shifts from DeltaNet syncs to dispatch overhead,
and Trace is the remedy for both simultaneously.

---

**Next:** Chapter 8 — [`../ch8_optimization_roadmap_and_testing/index.md`](../ch8_optimization_roadmap_and_testing/index.md)
