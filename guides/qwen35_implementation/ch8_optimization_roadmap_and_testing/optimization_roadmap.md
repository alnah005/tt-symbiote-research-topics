# Optimization Roadmap

## Overview

Chapter 7 established that the 86 ms/token baseline (6.8% efficiency) has three
independent cost categories: sync overhead (~35 ms profiling era, now ~11 syncs × ~1 ms),
Python dispatch (~26 ms), and device compute (~20 ms). The optimizations below target
each category in order of impact.

## Optimization 1 — Metal Trace (highest impact)

**Targets:** Python dispatch (~26 ms) and residual sync overhead (~10 ms)
**Expected outcome:** ~20–25 ms/token total, ~25–30% efficiency

**Mechanism:** Metal Trace records the device command sequence during a one-time "capture"
token. On all subsequent tokens, the driver replays the captured binary stream without
re-invoking any Python and without inserting inter-layer synchronisation barriers. The
trace replay executes the full 40-layer forward pass as a single atomic command stream
from the device's perspective.

**Why it is not yet enabled:** Metal Trace requires all tensor device addresses to remain
stable across tokens. If a tensor is reallocated between the capture and replay, the
recorded address is stale and the kernel reads garbage. All `from_torch` calls that allocate
new device tensors must either be removed from the traced region or replaced with in-place
writes to pre-allocated tensors.

**Current prerequisites (already met):**

- **DeltaNet recurrent state:** `gated_deltanet.py` uses `ttnn.copy(result[1], self._dev_state)`
  to write the updated state in-place. The `_dev_state` tensor is allocated once in
  `initialize_states` and never reallocated. Address stable. ✓
- **DeltaNet conv ring buffer:** `ttnn.copy(qkv_new, self._conv_rows[self._oldest])` writes
  the new conv slot in-place. All four `_conv_rows` tensors are pre-allocated. Address stable. ✓
- **RoPE matrices:** `HfRotarySetup.get_rot_mats()` returns the pre-patched `cos_matrix`
  and `sin_matrix` without new allocation. Address stable. ✓

**Remaining work before Trace enablement:**
- Verify that the MoE expert routing (the `ttnn.to_torch(router_logits)` sync) can be
  handled at the trace boundary rather than inside the trace. One option: move expert
  selection outside the trace region by pre-computing routing for a fixed batch.
- Confirm that `ttnn.from_torch` for position tensors and batch embedding can be replaced
  with in-place writes.

## Optimization 2 — Multi-CQ overlap

**Targets:** DMA latency hidden behind device compute
**Expected outcome:** Additional ~3–5 ms reduction when combined with Trace

**Mechanism:** TT-Metal supports two command queues per device. CQ0 handles the main
compute stream. CQ1 handles memory transfers. When the next-token embedding is known
(e.g., during decode step N, step N+1's input is available after argmax), the host can
issue the CQ1 `from_torch` for step N+1's input while CQ0 executes step N's layer loop.
This hides the embedding upload latency behind device compute.

**Current situation:** The demo loops `for layer in model.layers: x = layer(...)` on a single
implicit command queue. Adding CQ1 support requires restructuring the embedding/state
upload to use a separate queue and ensuring the CQ0 reads of those tensors are gated by
a CQ1-completion event.

**Relationship to Metal Trace:** Multi-CQ is complementary. Trace eliminates Python dispatch;
Multi-CQ hides upload latency within the trace replay window.

## Optimization 3 — Per-row MoE routing

**Targets:** Correctness for heterogeneous batch inputs (not a latency optimization)
**Expected outcome:** Enables non-trivial batch multiplexing with different prompts

**Current limitation:** `Qwen35MoE.forward()` reads the routing logits from batch row 0 only
and applies those expert selections to all batch rows:

```python
logits_cpu = ttnn.to_torch(router_logits).float()[0, 0, 0, :num_experts]
topk_ids = torch.topk(logits_cpu, self.num_experts_per_tok).indices
```

This is valid only when all rows carry the same token (same-prompt batch replication, the
default in `demo_a3b.py`). When rows carry different tokens (mixed-prompt batching), each
row needs its own topk, and experts must be dispatched per-row.

**Implementation path:**
1. Extend `ttnn.to_torch(router_logits)` to read all `B_pad` rows.
2. Per-row `torch.topk` to get `topk_ids[b]` for each batch row.
3. Group rows by selected expert sets to minimise the number of unique matmul calls.
4. Accumulate per-row expert outputs into the result tensor.

This is identified in the source as "future work" in a comment block in `qwen35_moe.py`.

## Optimization 4 — Fused DeltaNet kernel enabling path

**Status:** The fused `ttnn.experimental.gated_delta_net` kernel is **already deployed**.
This entry describes why it was previously blocked and what enabled it.

**Historical block:** Before the fused kernel, the note in PERF.md read:
*"Fused DeltaNet kernel ready (PCC 0.999997), blocked by from_torch overhead without Metal Trace."*
This meant: even with the fused kernel eliminating DeltaNet syncs, the ~26 ms Python dispatch
cost remained. The total latency would drop from ~86 ms to ~60 ms, but not to the target range.
Metal Trace is the remedy that makes the fused kernel's benefits fully realised.

**Current state:** The fused kernel is deployed and zero DeltaNet host syncs occur per token.
Python dispatch overhead (~26 ms) is the next bottleneck. Metal Trace will bring the fused
kernel's contribution from ~54 ms (host-recurrence era) to near-device-compute-bound.

## Expected latency trajectory

| Stage | Latency | Efficiency |
|---|---|---|
| Profiling era (host recurrence + host RoPE) | 86 ms/token | 6.8% |
| Current (fused kernel + device RoPE) | ~50–60 ms/token | ~10–12% |
| + Metal Trace | ~20–25 ms/token | ~25–30% |
| + Metal Trace + Multi-CQ | ~18–22 ms/token | ~27–33% |

---

**Next:** [`testing_infrastructure.md`](./testing_infrastructure.md)
