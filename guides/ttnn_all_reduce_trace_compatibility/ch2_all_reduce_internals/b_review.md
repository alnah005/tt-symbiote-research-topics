# B Review — Chapter 2: ttnn.all_reduce Internal Architecture — Pass 1

1. **`call_chain.md` §4.3 — `use_composite_reduce_scatter` predicate inversion (factual error, would cause wrong implementation)**

   The guide states `use_composite_reduce_scatter` returns `true` when "The scatter dimension size divided by `num_devices` is not evenly divisible." The actual code in `composite_common.cpp` (lines 46–48) does the opposite: `if (input_shape[scatter_dim] % num_devices != 0) { return false; }`. When the scatter dimension is not evenly divisible by `num_devices`, the function returns `false` (do NOT use composite). A reader implementing the path-selection logic from the guide's description would invert this branch and select the wrong path for non-divisible dimensions. The correct statement is: if the scatter dimension is not evenly divisible, the function returns `false`; the composite path is used only when the scatter dimension IS evenly divisible but the resulting per-device slice is not tile-aligned, or the tensor is row-major.

2. **`composite_path.md` §2 — `composite_all_gather` uses device-side `ttnn::concat`, not host-side (factual error)**

   The guide states "`concat` joins these tensors along the gather dimension on the host side." The actual `composite_all_gather` implementation (`composite_common.cpp` line 379) calls `ttnn::concat(broadcasted_tensors, gather_dim)`, which is a TTNN device op that executes on device. Describing it as a host-side operation is wrong and would cause a reader analysing trace compatibility to mischaracterise where the allocation and command recording occur.

3. **`call_chain.md` §3 — code snippet omits the `barrier_semaphores` size check that guards the async `all_gather` branch (coherence gap)**

   The guide's code snippet for the non-composite branch shows the `all_gather` async guard as `ag_global_semaphores.has_value() && barrier_semaphores.has_value()`. The actual code (line 362) adds a `TT_FATAL(barrier_semaphores.value().size() == 2, ...)` assertion immediately inside that guard, revealing that the `barrier_semaphores` vector must contain exactly 2 elements (one for reduce-scatter, one for all-gather). A reader implementing the async path from the guide's description alone would not know about this 2-element requirement and would produce a runtime assertion failure.

4. **`contrast_with_async_variants.md` §3 — `TT_CCL` init loop creates 3 `rs_semaphore_handles` per slot, but that detail is irrelevant to cycling; the more critical omission is the `barrier_semaphores` 2-element constraint (structural gap)**

   The guide correctly shows the double-buffered pool cycling. However, both the guide and the comparison table in §5 describe `ttnn.experimental.all_reduce_async` (the non-cluster-axis overload at lines 152–256) as requiring `vector<GlobalSemaphore>` for all three semaphore positions. The actual non-cluster-axis overload signature (line 152–162 of `all_reduce_async.cpp`) takes these as *non-optional* `const std::vector<GlobalSemaphore>&` — correct — but also always calls `reduce_scatter_minimal_async` and `prim::all_gather_async` unconditionally (no `has_value()` guard), unlike the cluster-axis overload which has the fallback to synchronous ops. A reader trying to understand when the synchronous fallback is available would incorrectly conclude it is available in both overloads; in fact, the fallback only exists in the cluster-axis overload. Chapter 3 or later text that depends on this distinction should be aware of the gap.

---

# B Review — Chapter 2 — Pass 3

1. **`composite_path.md` §4 — `gather_tensor` shape formula is factually wrong (wrong numerical answer)**

   Line 164–165 states that `gather_tensor` "is sized as `[1, num_devices * initial_shape[1], H, W]`". This is incorrect. In the cluster-axis overload (which is what `ttnn.all_reduce` actually invokes), the tensor is first reshaped to `[1, initial_shape[0]*initial_shape[1], H, W]` and then `composite_all_gather` gathers along `composite_dim = 0`. `composite_all_gather` calls `ttnn::prim::all_broadcast` to produce `num_devices` copies of the reshaped tensor, then `ttnn::concat` along dim 0. The resulting `gather_tensor` shape is therefore `[num_devices, initial_shape[0]*initial_shape[1], H, W]` — the first dimension becomes `num_devices`, not the second. For the typical decode input `[1, 1, 32, H]` the reshaped tensor is `[1, 1, 32, H]` and `gather_tensor` is `[8, 1, 32, H]`, not `[1, 8, 32, H]`. A reader computing or allocating a pre-allocated buffer for this intermediate would size it incorrectly.

2. **`contrast_with_async_variants.md` §1 — guard condition description omits `barrier_semaphores.has_value()` (factual error, causes incorrect implementation)**

   Lines 42–44 state that "the `std::nullopt` guards on `rs_global_semaphores.has_value()` and `ag_global_semaphores.has_value()` route execution to the synchronous fallbacks." The actual guard for the reduce-scatter branch (line 331 of `all_reduce_async.cpp`) is `rs_global_semaphores.has_value() && barrier_semaphores.has_value()`, and the all-gather branch guard (line 361) is `ag_global_semaphores.has_value() && barrier_semaphores.has_value()`. In both cases `barrier_semaphores` is a mandatory component of the guard. A developer reading this section and constructing a call site would believe they need only supply `rs_global_semaphores` (or `ag_global_semaphores`) to enter the async branch; omitting `barrier_semaphores` would silently fall through to the synchronous path, producing incorrect performance characteristics and subtle bugs when `barrier_semaphores` alone is missing while the other semaphore is present.

5. **`call_chain.md` §4.3 — `use_composite_reduce_scatter` condition list omits the early-return-false for non-divisible dimensions, leaving the predicate logic incomplete for a reader who wants to implement it**

   Closely related to item 1, but the additional issue is structural: the guide's bullet-point list for `use_composite_reduce_scatter` omits the leading early-exit condition entirely (`if (input_shape[scatter_dim] % num_devices != 0) return false`). Without this entry, a reader cannot reconstruct the full predicate from the guide. The function only proceeds to check tile-alignment after confirming divisibility; the guide presents tile-alignment as a top-level condition without noting that it is only evaluated when the dimension IS divisible.

---

# B Review — Chapter 2 — Pass 2

1. **`reduce_scatter_all_gather_path.md` §3 — incorrect claim that `scattered_tensor` has a stable DRAM address in trace replay (factual error, would cause wrong implementation)**

   The guide states: "Inside a trace capture: The TTNN trace machinery snapshots the device buffer address that was assigned during the capture pass. On every subsequent replay, the runtime replays the same command sequence… The allocator is not re-invoked during replay; instead, the buffer retains its capture-time address for the lifetime of the trace." This is mechanistically wrong for DRAM-interleaved tensors. TTNN trace captures metal-level command buffers; it does not pin dynamically-allocated DRAM buffer addresses at capture time. A DRAM buffer allocated inside an op during trace capture will be re-allocated by the standard allocator on each replay and may land at a different offset if the free-list state differs. The note later in §3 only weakens the claim ("less prone to instability") without correcting it. The composite-path analysis in `composite_path.md` §4 correctly identifies dynamic DRAM allocation as the source of trace incompatibility; the same argument applies, in full, to `scattered_tensor`. A reader implementing trace-safe collectives from this section would incorrectly conclude that the single-intermediate non-composite path is already safe without pre-allocating a persistent output buffer for `scattered_tensor`.

2. **`composite_path.md` §2 — `composite_all_gather` code snippet uses the non-cluster-axis overload's reshape, not the cluster-axis overload's (factual error for non-4D tensors)**

   The guide's code snippet for the composite path shows `ttnn::reshape(interleaved_tensor, ttnn::Shape({1, initial_shape[0] * initial_shape[1], initial_shape[2], initial_shape[3]}))`, which is the hard-coded 4-element reshape from the non-cluster-axis overload (lines 194–196 of `all_reduce_async.cpp`). The cluster-axis overload (which is what `ttnn.all_reduce` actually invokes) uses a dynamic `SmallVector` that copies all dimensions beyond index 1 from the original shape, producing `ttnn::Shape(ag_shape_vec)` (lines 300–305). For rank-4 tensors the results are identical, but for any other rank the expressions differ. Because the chapter explicitly targets the cluster-axis overload, presenting the non-cluster-axis reshape as the representative code will mislead any reader who applies the pattern to non-4D inputs.

3. **`reduce_scatter_all_gather_path.md` §1 — code comment labels the synchronous `reduce_scatter` branch as "taken when semaphore args are `std::nullopt`", but the actual guard checks both `rs_global_semaphores` AND `barrier_semaphores` (incomplete guard description)**

   The guide comment says "synchronous fallback — taken when semaphore args are `std::nullopt`". The actual guard (line 331 of `all_reduce_async.cpp`) is `if (rs_global_semaphores.has_value() && barrier_semaphores.has_value())`. The synchronous path is taken when EITHER argument is absent, not only when both are `std::nullopt`. A caller who supplies `rs_global_semaphores` but omits `barrier_semaphores` (or vice versa) would fall through to the synchronous path silently, which could cause confusion when debugging partial semaphore configurations. The guide's description implies the condition is symmetric and total (both must be present for async), which matches the code but the comment wording "semaphore args are `std::nullopt`" suggests "all are null" rather than "either is absent", which is marginally ambiguous. The identical asymmetry exists for the `all_gather` guard (line 361): `ag_global_semaphores.has_value() && barrier_semaphores.has_value()`. A precise description should read "taken when either `rs_global_semaphores` or `barrier_semaphores` is absent (`std::nullopt`)".

4. **`contrast_with_async_variants.md` §2 — comparison table column for `ttnn.experimental.all_reduce_async` describes only the non-cluster-axis overload, but `ttnn.all_reduce` delegates to the cluster-axis overload; the two overloads have different semaphore type signatures (coherence gap)**

   The comparison table in §5 lists `ttnn.experimental.all_reduce_async` as requiring `vector<GlobalSemaphore>` (non-optional). This correctly describes the non-cluster-axis overload (lines 152–162). However, `ttnn.all_reduce` itself calls the cluster-axis overload (lines 258–269), which takes `std::optional<std::vector<GlobalSemaphore>>`. A developer who looks up `ttnn.experimental.all_reduce_async` in the header to understand its interface will find the overload set includes the cluster-axis form with optional semaphores — calling it as though semaphores are required will fail to compile against the optional-accepting cluster-axis overload, and calling it with optional semaphores against the non-optional non-cluster-axis overload will also fail. The table should distinguish the two overloads (cluster-axis vs. non-cluster-axis) so a reader knows which semaphore type signature they will encounter depending on how they invoke the function.

5. **`composite_path.md` §3 and `reduce_scatter_all_gather_path.md` §4 — the summary verdict "compatible" for the non-composite path rests on the incorrect allocation-stability claim in §3, making the trace-compatibility conclusion unreliable (structural coherence)**

   `composite_path.md` §4 gives the correct reason for composite-path incompatibility: dynamic intermediate tensor allocation. `reduce_scatter_all_gather_path.md` §4 then declares the non-composite path "Compatible, subject to fabric config being 1-D" and attributes this to "address stable across replay" for `scattered_tensor`. Because the address-stability claim in §3 is wrong (see issue 1 above), the compatibility verdict inherits the error. The verdict may still be correct for other reasons (e.g., if `ttnn.reduce_scatter` internally pre-allocates its output via a persistent buffer mechanism), but the chapter does not establish this and cannot support the verdict as written. A downstream reader implementing or verifying trace-safe all-reduce on T3K would rely on this verdict and might skip the pre-allocation precaution that is actually required.

---

# B Review — Chapter 2 — Pass 3

1. **`composite_path.md` §4 — `gather_tensor` shape formula is factually wrong (wrong numerical answer)**

   Line 164–165 states that `gather_tensor` "is sized as `[1, num_devices * initial_shape[1], H, W]`". This is incorrect. In the cluster-axis overload (which is what `ttnn.all_reduce` actually invokes), the tensor is first reshaped to `[1, initial_shape[0]*initial_shape[1], H, W]` and then `composite_all_gather` gathers along `composite_dim = 0`. `composite_all_gather` calls `ttnn::prim::all_broadcast` to produce `num_devices` copies of the reshaped tensor, then `ttnn::concat` along dim 0. The resulting `gather_tensor` shape is therefore `[num_devices, initial_shape[0]*initial_shape[1], H, W]` — the first dimension becomes `num_devices`, not the second. For the typical decode input `[1, 1, 32, H]` the reshaped tensor is `[1, 1, 32, H]` and `gather_tensor` is `[8, 1, 32, H]`, not `[1, 8, 32, H]`. A reader computing or allocating a pre-allocated buffer for this intermediate would size it incorrectly.

2. **`contrast_with_async_variants.md` §1 — guard condition description omits `barrier_semaphores.has_value()` (factual error, causes incorrect implementation)**

   Lines 42–44 state that "the `std::nullopt` guards on `rs_global_semaphores.has_value()` and `ag_global_semaphores.has_value()` route execution to the synchronous fallbacks." The actual guard for the reduce-scatter branch (line 331 of `all_reduce_async.cpp`) is `rs_global_semaphores.has_value() && barrier_semaphores.has_value()`, and the all-gather branch guard (line 361) is `ag_global_semaphores.has_value() && barrier_semaphores.has_value()`. In both cases `barrier_semaphores` is a mandatory component of the guard. A developer reading this section and constructing a call site would believe they need only supply `rs_global_semaphores` (or `ag_global_semaphores`) to enter the async branch; omitting `barrier_semaphores` would silently fall through to the synchronous path, producing incorrect performance characteristics and subtle bugs when `barrier_semaphores` alone is missing while the other semaphore is present.

---

# B Review — Chapter 2 — Pass 4

1. **`composite_path.md` §2 — `local_sum` code snippet references undefined variable `input_tensor` (factual error, would cause wrong implementation)**

   The `local_sum` function signature declares its parameter as `const ttnn::Tensor& gathered_tensor`, but the body on line 115 calls `ttnn::moreh_sum(input_tensor, ...)`. `input_tensor` is not defined in this scope; the correct variable is `gathered_tensor`. A reader implementing `local_sum` from this snippet would produce a compilation error or, if correcting by inference, could not trust the accuracy of the surrounding documentation.

2. **`composite_path.md` §1 bullet 2 — `use_composite_reduce_scatter` is misattributed as the predicate that fires for non-divisible scatter dimensions (factual error, causes wrong conceptual model)**

   Line 22 states: "`composite_common::use_composite_reduce_scatter(...)` returns `true` — the tensor cannot be evenly scatter-reduced across devices in a tile-aligned manner." As established in `call_chain.md` §4.3, `use_composite_reduce_scatter` returns `false` (do NOT use composite) when the scatter dimension is not evenly divisible by `num_devices`. For non-divisible dimensions, composite-path selection is triggered by `dim != composite_dim` (bullet 3 of the same list), not by `use_composite_reduce_scatter`. The description in bullet 2 wrongly attributes that code path to this predicate. A reader building a mental model of when each predicate fires would assign composite-path responsibility to the wrong function, leading to incorrect path-selection logic when implementing or debugging the branch.

3. **`contrast_with_async_variants.md` §1 — trace compatibility claim omits the `scattered_tensor` pre-allocation requirement (critical structural gap)**

   Lines 62–64 state: "It is compatible with trace capture on a 1-D fabric T3K configuration when the tensor dimensions satisfy the non-composite predicates." This omits the pre-allocation requirement established in `reduce_scatter_all_gather_path.md` §4: the non-composite path is only trace-compatible when `scattered_tensor` is pre-allocated as a persistent buffer before `ttnn.begin_trace_capture` and passed as `optional_output_tensor` to `ttnn.reduce_scatter`. A reader consulting `contrast_with_async_variants.md` as their primary reference for `ttnn.all_reduce` trace compatibility would implement a trace loop without this pre-allocation and encounter silent data corruption or hardware faults on replay.

---

# B Review — Chapter 2 — Pass 5

No feedback — chapter approved.

---

# B Review — Chapter 2 — Pass 6

No feedback — chapter approved.
