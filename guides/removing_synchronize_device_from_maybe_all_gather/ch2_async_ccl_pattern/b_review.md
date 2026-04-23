# B Review — Pass 1

1. [cycling_semaphore_mechanics.md, ~line 83-86, Wrong axis-to-index mapping claim for `cluster_axis=0`]
   The inline comment in the reproduced `get_and_cycle_ag_semaphore_handles` snippet states:
   > "maps cluster_axis=None → index 2, cluster_axis=0 → index 0, cluster_axis=1 → index 1"

   This is factually wrong. The actual expression in both `get_and_cycle_ag_semaphore_handles` and `get_and_cycle_barrier_semaphore_handle` is:
   ```python
   semaphore_index = 2 if not cluster_axis else cluster_axis
   ```
   In Python, `not 0` evaluates to `True`, so `cluster_axis=0` maps to index `2`, not index `0`. Only `cluster_axis=1` maps to index `1`; both `cluster_axis=None` and `cluster_axis=0` map to index `2`. The same incorrect mapping comment appears again at `get_and_cycle_barrier_semaphore_handle` (~line 104).

   Fix: Replace the comment with the accurate mapping: `cluster_axis=None → index 2, cluster_axis=0 → index 2 (falsy), cluster_axis=1 → index 1`. Optionally flag the upstream code quirk: for models that use `cluster_axis=0` as a distinct axis, the falsy-zero test collapses the "axis-0" pool and the "no-axis" pool into the same slot, which is the actual source behavior and may be intentional (T3K only ever passes `cluster_axis=1` or `None` from attention.py).

2. [cycling_semaphore_mechanics.md, ~line 84, Same wrong mapping repeated in `get_and_cycle_barrier_semaphore_handle` snippet]
   The comment `# why: same axis-to-index mapping as above` at line ~104 defers to the wrong mapping described in issue 1 above. Both method descriptions carry the same error. Fix: update both comments together when correcting issue 1.

3. [all_gather_async_in_traced_attention.md, ~line 54-55, Variant B gating condition is imprecise]
   The chapter labels Variant B as:
   > "When `self.use_fused_all_gather_matmul` is `True` but `self.ccl_topology != ttnn.Topology.Ring`"

   The actual source structure is:
   ```python
   if self.use_fused_all_gather_matmul:          # outer gate (line 545)
       ...
       if self.ccl_topology == ttnn.Topology.Ring:   # inner gate → Variant A (line 551)
           ...
       else:                                          # inner else → Variant B (line 569)
           ...
   else:                                          # outer else → Variant C (line 597)
       ...
   ```
   The condition stated in the chapter is logically correct, but the description says this path applies "when `self.use_fused_all_gather_matmul` is `True` but topology != Ring." However the Variant C description immediately below it says the non-fused path is taken "when `self.use_fused_all_gather_matmul` is `False`." Together the three variants cover all cases correctly. This is a minor precision issue (the outer guard is `use_fused_all_gather_matmul == True` for both A and B), not a material error, but the repeated `use_fused_all_gather_matmul == True` qualifier on Variant B should be noted explicitly to avoid reader confusion about whether Variant B is reachable when `use_fused_all_gather_matmul` is False. No fix strictly required, but adding "outer gate: `use_fused_all_gather_matmul is True`; inner gate: `ccl_topology != Ring`" would be clearer.

4. [all_gather_async_in_traced_attention.md, ~line 97-99, Variant C description misstates what `tt_all_gather` does for the decode path]
   The chapter says:
   > "When `self.use_fused_all_gather_matmul` is `False`, the path calls `tt_all_gather` (defined in `models/tt_transformers/tt/ccl.py`), which in turn calls `ttnn.experimental.all_gather_async` with the same cycling semaphore pattern (see `ccl.py` lines ~239-252 for the `cluster_axis is not None` branch)."

   The actual code at line 598 of `attention.py` calls:
   ```python
   attn_output = tt_all_gather(
       attn_output_cat,
       self.mesh_device,
       self.tt_ccl,
       dim=2,
       cluster_axis=1,
       ...
   )
   ```
   Because `cluster_axis=1` (truthy), this call does reach the `else` branch of `tt_all_gather` at lines 238-252, so the line reference is accurate. However, the chapter omits that the non-fused path in `attention.py` does not return immediately after this all_gather — it then calls `ttnn.matmul` (line 622) and `tt_all_reduce` (line 635), not `ttnn.linear` as suggested by the surrounding Variant A/B narrative. The description is incomplete enough to mislead a reader into thinking the non-fused path's structure mirrors Variant B (all_gather + linear). This is an incompleteness/mischaracterization rather than a direct factual error, but it creates a false parallel.

   Fix: Add a note that in Variant C the all_gather is not followed by `ttnn.linear`; instead the decode path proceeds through `ttnn.matmul` + `tt_all_reduce` before returning.

5. [cycling_semaphore_mechanics.md, ~line 71, Total GlobalSemaphore count arithmetic is correct but the axis-pool sum is inconsistent with the mapping error in issue 1]
   The chapter states "3 × (2 + 4 + 6) = 36 objects." The arithmetic is correct: 3 axis-index entries × (2 barrier + 4 ag + 6 rs handles) = 36. However, given the mapping error documented in issue 1 (cluster_axis=0 and cluster_axis=None both map to index 2), index 0 of each pool is allocated but never selected by either `get_and_cycle_ag_semaphore_handles` or `get_and_cycle_barrier_semaphore_handle` in the code paths shown. The total allocation of 36 is accurate per the `__init__` code; the claim that the three indices correspond to three distinct axes is the inaccuracy. No fix needed to the count — but the "three axis variants" description should be corrected to reflect the actual mapping behavior.

6. [all_gather_async_in_traced_attention.md, ~line 105-107, `synchronize_device` absence claim overstates certainty for `ccl.py`]
   The chapter states:
   > "A search of the entire file confirms the same: the word `synchronize_device` does not appear anywhere in `attention.py` or in `ccl.py`."

   Verified: `synchronize_device` does not appear in the current source of either file. This claim is accurate as of the reviewed source. No fix needed.

7. [index.md, ~line 10-11, Finish token / host-side wait description is internally consistent with Ch1]
   The index states: "The Finish token is a CQ0 device command and IS recorded and replayed; the host-side blocking wait is NOT recorded." This matches Ch1 `synchronize_device_semantics.md` exactly. No error.

8. [persistent_output_buffer_contract.md, ~line 62, "buffer ... kept alive for the lifetime of the TT_CCL / model instance" — slightly imprecise attribution]
   The chapter attributes output buffer lifetime management to "the `TT_CCL` / model instance." In the actual mechanism, the buffer is held alive by the program cache entry (which is owned by the TTNN runtime, not by `TT_CCL` itself). `TT_CCL` manages semaphore handles, not output buffer lifetimes. Attributing address stability to `TT_CCL` is a category error: `TT_CCL` is unrelated to output buffer pinning. The correct owner is the TTNN program cache.

   Fix: Change "kept alive for the lifetime of the TT_CCL / model instance" to "kept alive by the TTNN program cache for the lifetime of the model instance (not freed between calls as long as the program cache entry is valid)."

# B Review — Pass 2 (Change Log)

Changes applied in response to Pass 1:
1+2. `cycling_semaphore_mechanics.md`: corrected axis-to-index mapping in both method snippets — cluster_axis=0 maps to index 2 (falsy), not index 0; added Note about T3K practical usage
3. `all_gather_async_in_traced_attention.md`: clarified Variant B dual-gate (outer: use_fused_all_gather_matmul; inner: topology != Ring)
4. `all_gather_async_in_traced_attention.md`: added Variant C note that post-all_gather path uses ttnn.matmul + tt_all_reduce, not ttnn.linear
5. `cycling_semaphore_mechanics.md`: corrected "three axis variants" language to reflect that index 0 is allocated but not selected by standard cycling convention
8. `persistent_output_buffer_contract.md`: corrected buffer lifetime attribution from TT_CCL to TTNN program cache

# B Review — Pass 2

1. [cycling_semaphore_mechanics.md, line 118, Wrong truthiness term — `not cluster_axis` described as "falsy" when it is "truthy"]
   The Note reads:
   > "But the mapping is worth noting: `not cluster_axis` is falsy for both `None` and `0`."

   This is factually wrong. In Python, `not None` evaluates to `True` and `not 0` evaluates to `True` — the expression `not cluster_axis` is **truthy** (i.e., it evaluates to `True`) for both `None` and `0`. The values `None` and `0` are themselves falsy, but the result of applying `not` to them is truthy. Using the word "falsy" here inverts the actual evaluation and would mislead any reader checking the mapping logic.

   Fix: Replace "is falsy for both `None` and `0`" with "evaluates to `True` for both `None` and `0`" (or: "`cluster_axis=None` and `cluster_axis=0` are both falsy values, so `not cluster_axis` is `True` for both — hence both collapse to index 2").

2. [cycling_semaphore_mechanics.md, line 120, Misattributed caller — `all_gather_async` described as "the caller" of the cycle methods]
   The Note reads:
   > "The caller (e.g., `all_gather_async`) is responsible for calling both methods in the same dispatch to obtain a matched pair of handles from the same slot."

   `ttnn.experimental.all_gather_async` is the op that *receives* the handles as arguments — it is the callee, not the caller. It is a device-side kernel and cannot call Python methods on `self.tt_ccl`. The actual Python-level caller that invokes both `get_and_cycle_ag_semaphore_handles()` and `get_and_cycle_barrier_semaphore_handle()` in the same dispatch is `Attention.forward_decode` (for Variants A and B) or `tt_all_gather` / `tt_all_reduce` (for the helper paths in `ccl.py`).

   Fix: Replace "The caller (e.g., `all_gather_async`)" with "The Python call site (e.g., `Attention.forward_decode` or `tt_all_gather`)".

# B Review — Pass 3 (Change Log)

Changes applied in response to Pass 2:
1. `cycling_semaphore_mechanics.md` ~line 118: changed "is falsy" to "evaluates to `True`" — None and 0 are the falsy values, making `not cluster_axis` truthy (True) for both
2. `cycling_semaphore_mechanics.md` ~line 120: replaced "The caller (e.g., `all_gather_async`)" with "The Python call site (e.g., `Attention.forward_decode` or `tt_all_gather`)" — all_gather_async is the device-side callee, not the Python caller

# B Review — Pass 3

No feedback — chapter approved.
