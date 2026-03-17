# 8.3 Troubleshooting Guide

This section documents six errors that occur frequently when bringing up MoE inference on T3K.
Each entry lists the observable symptom, the most likely root cause, and the corrective action.

---

## Error 1: `ttnn.exceptions.MemoryAllocationError` — L1 Budget Exceeded

### Symptom

Runtime raises `ttnn.exceptions.MemoryAllocationError` during a matmul or collective operation.
The error message typically references an L1 allocation failure on one or more Tensix cores.

### Root Cause

The `per_core_M` parameter in the program config controls how many output rows are assigned to
each core. A value of `per_core_M > 1` multiplies the per-core L1 working set linearly. At decode
time with $C < 32$, the correct value is:

$$\text{per\_core\_M} = \left\lceil \frac{C}{32} \right\rceil = 1$$

If `per_core_M` is set to a larger value (for example, copied from a prefill config without
adjustment), the matmul kernel will attempt to allocate more L1 than is available on Wormhole B0
(1.5 MB per core [UNVERIFIED]).

A secondary cause is placing large tensors in L1 when they should be in DRAM. Expert weight
tensors are approximately 205 MB per device [D UNVERIFIED], which exceeds the 120 MB aggregate L1
and must use `ttnn.DRAM_MEMORY_CONFIG`.

### Fix

1. Verify that `per_core_M = 1` in all decode-path program configs.
2. Confirm that all expert weight tensors are placed with `ttnn.DRAM_MEMORY_CONFIG`, not
   `ttnn.L1_MEMORY_CONFIG`.
3. If the error persists, profile L1 usage per core [VERIFY: TTNN profiler API] and identify which
   tensor is overflowing.

---

## Error 2: Low Throughput Despite Correct Topology

### Symptom

Token throughput is significantly below expectations even though the mesh topology is correctly
configured and all operations complete without errors. Profiling shows that the all-to-all
collective steps are consuming disproportionate wall time.

### Root Cause

The `num_links` parameter to `ttnn.all_to_all` [VERIFY] controls how many Ethernet links are used
for the collective. T3K provides 12.5 GB/s per Ethernet link on its $1 \times 8$ linear mesh
[UNVERIFIED]. If `num_links=1` is used for dispatch volumes that exceed approximately 1 MB, the
single link becomes the bottleneck.

For $B = 32$, the dispatch volume per device is:

$$7 \times 2 \times 32 \times 7168 \times 2 = 6{,}422{,}528 \text{ bytes} \approx 6.4 \text{ MB}$$

At one 12.5 GB/s link, this transfer takes approximately 0.51 ms. Increasing to two links halves
the effective transfer time for this volume.

### Fix

Set `num_links=2` in all `ttnn.all_to_all` [VERIFY] calls when the per-collective dispatch volume
exceeds 1 MB. As a rule of thumb, use `num_links=1` only for very small batch sizes ($B = 1$,
dispatch volume $< 1$ MB) where link contention is not a concern.

---

## Error 3: NaN or Inf Values in Output

### Symptom

Model output logits or hidden states contain NaN or Inf values. The issue may manifest as
degenerate token sampling (all tokens equally probable) or as an explicit NaN error from a
downstream operation.

### Root Cause

Two independent causes produce this symptom.

**Cause A: Expert weight placed in SRAM/L1 instead of DRAM.**
If an expert weight tensor is accidentally placed in L1, it will overflow into another tensor's
allocation and corrupt the data. Reads from this corrupted region produce NaN or garbage values.

**Cause B: Routing score normalization using maximum instead of sum.**
The routing scores $s_i = \sigma(g_i)$ for a token's $k$ selected experts must be normalized by
their sum:

$$\hat{s}_i = \frac{s_i}{\sum_{j=1}^{k} s_j}$$

Normalizing by the maximum value instead:

$$\hat{s}_i^{\text{wrong}} = \frac{s_i}{\max_j s_j}$$

does not produce a probability distribution (the weights do not sum to 1). For tokens where the
top-scoring expert has a much higher sigmoid output than the others, this normalization assigns
near-zero weight to all but the top expert, effectively discarding the contribution of $k-1$
experts and degrading output quality. In edge cases it can produce unbounded accumulated values
that overflow to Inf.

### Fix

For Cause A: audit all expert weight tensor creation calls and confirm that every call uses
`memory_config=ttnn.DRAM_MEMORY_CONFIG`.

For Cause B: audit the routing score normalization step. The denominator must be
`ttnn.sum(scores, dim=-1, keepdim=True)` [VERIFY], not `ttnn.max(scores, ...)`.

---

## Error 4: Token Drop Rate Exceeds 1%

### Symptom

Monitoring shows that more than 1% of tokens are being dropped (assigned to overflow slots or
discarded) during the dispatch step. Output quality degrades noticeably on inputs with uneven
expert load.

### Root Cause

Token dropping occurs when the number of tokens dispatched to a given expert exceeds its slot
capacity $C$. With capacity factor $\text{CF} = 1.25$, the expected capacity per expert is:

$$C = \left\lceil \frac{k \times B \times \text{CF}}{E} \right\rceil = \left\lceil \frac{B}{25.6} \right\rceil$$

This capacity is computed over the average load. If one or more experts are "hot" (consistently
receiving more tokens than average due to model behavior or input distribution), their actual load
exceeds $C$ and tokens spill.

### Fix

1. Instrument expert utilization per layer per step. Compute the per-expert token count histogram
   and identify which experts are hot.
2. Increase $\text{CF}$ (for example to 1.5 or 2.0) to raise capacity headroom. Note that
   increasing CF increases the dispatch volume and all-to-all latency proportionally.
3. For persistently hot experts, consider replicating them across multiple devices so that their
   effective capacity is multiplied by the replication factor. This requires modifying the dispatch
   logic to route tokens to one of several replica devices.

---

## Error 5: Router Outputs the Same Expert for All Tokens

### Symptom

Inspecting the expert index tensor after `ttnn.topk` [VERIFY] shows that most or all tokens are
being routed to a small number of identical experts (often expert 0 or the experts with the lowest
index). Expert utilization is extremely uneven and token drop (Error 4) appears immediately.

### Root Cause

This is a BF16 saturation artifact in the router logit computation. BF16 has 7 mantissa bits and
a machine epsilon of $2^{-7} \approx 0.0078$. The sigmoid function:

$$\sigma(g) = \frac{1}{1 + e^{-g}}$$

saturates to 1.0 for $g \gtrsim 15$ and to 0.0 for $g \lesssim -15$. In BF16, values near the
saturation boundary are represented with reduced precision, and multiple logits that should be
distinct can collapse to the same BF16 value (either 1.0 or 0.0). After saturation, `topk`
breaks ties arbitrarily, often by index order, causing the lowest-index experts to be
systematically selected.

The critical threshold to check is whether any router logit magnitude exceeds approximately 30.
Above this value, the sigmoid output in BF16 is indistinguishable from 1.0 and all affected
experts appear equally preferred.

### Fix

1. Log the per-token router logit range. If any logit satisfies $|g| > 30$, saturation is likely.
2. Inspect the model's $W_r$ weight norms. Unusually large weight norms (possibly from a failed
   or incomplete fine-tune) are the most common cause of logit explosion.
3. If the weights themselves are correct and the logit range is large due to the input distribution,
   consider applying a logit cap before the sigmoid:
   $g_{\text{capped}} = \operatorname{clip}(g, -15, 15)$. Verify that this does not degrade
   model accuracy on the target task before deploying.
4. As a diagnostic step, temporarily run the router in FP32 [VERIFY: TTNN dtype option] to
   determine whether the routing collapse disappears. If it does, the issue is confirmed as a
   BF16 precision artifact rather than a model weight problem.

---

## Error 6: Wrong Expert Results (Output Mismatch After Combine)

### Symptom

The model produces incorrect outputs that are reproducible and deterministic (not random NaN).
Unit tests comparing per-expert FFN outputs against a reference implementation show that the
correct computation was performed but the results were accumulated into the wrong token positions.
The effect is a permutation of expert contributions across tokens.

### Root Cause

The combine all-to-all (Step 5 of Section 8.2.2) returns processed tokens to their originating
devices. Each returned token must be accumulated into the correct position with its corresponding
routing score. This requires that the `dispatch_meta` structure (recording which token sent to
which expert at dispatch time) is used in the same order at combine time.

A mismatch occurs when:

1. The dispatch_meta records tokens in one order (for example, sorted by destination device then
   by expert index).
2. The combine step receives tokens in a different order (for example, sorted by source device
   then by expert index).
3. The accumulation loop iterates over the combine output using an index derived from one ordering
   while reading routing scores from dispatch_meta using the other.

The result is that token $i$'s expert output is added to token $j$'s accumulator with token $k$'s
routing score — producing a deterministic but incorrect output.

### Fix

1. Add a consistency check immediately after dispatch: verify that the expert indices embedded in
   the dispatch_meta exactly match the expert indices in the packed send buffer, in the same slot
   order.
2. Add a consistency check immediately after combine: verify that the token indices in the
   received combine buffer correspond to the token indices expected by the dispatch_meta, in the
   same slot order.
3. The safest implementation encodes a (token_id, expert_id) pair into each dispatched slot and
   verifies the pair at combine time before accumulating. This adds a small overhead but makes
   ordering bugs immediately detectable.
4. If using a custom packing kernel, ensure that the sort key used to fill the send buffer is
   identical to the sort key used to read the receive buffer at combine. Any difference in tie-
   breaking (for example, by token index vs. by sequence in the top-k result) is sufficient to
   produce a mismatch.
