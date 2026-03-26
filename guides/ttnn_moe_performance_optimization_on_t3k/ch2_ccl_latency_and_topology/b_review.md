# B Review — Chapter 2 — Pass 1

1. **`all_gather_async` call site is missing `cluster_axis=1` (`all_gather_linear_topology.md`, lines 15–22).** The ground truth specifies `cluster_axis=1` for both CCL operations. The documented call block omits this parameter entirely. A reader copying the snippet would produce a call that may default to a different axis, giving wrong output on a 1×8 mesh where the communication axis is axis 1.

2. **Ring topology mechanics: "opposite neighbor" is factually wrong (`ccl_sensitivity_analysis.md`, line 28).** The text states each ring step requires receiving "from the **opposite** neighbor." In a ring reduce-scatter, each device receives from its *adjacent* (left) neighbor, not the device diametrically across the ring. On an 8-device ring, "opposite" means 4 hops away. This mischaracterizes the algorithm and would cause a reader to implement or reason about it incorrectly.

3. **Last-hop message size is inconsistent with the latency formula (`all_gather_linear_topology.md`, lines 46 and 64–65).** The file correctly states that hop 7 carries 7 shards = 12 544 B, but the `T_linear` formula substitutes `M_full ≈ 14 KB` (14 336 B, all 8 shards) for the transfer term. The 8th shard is added locally on device 7 after receiving the 7-shard wire payload; it is never transferred. Using 14 336 B instead of 12 544 B overstates the final-hop transfer time by ~14%. The resulting ~29 µs estimate is inflated at the transfer term, which will mislead any reader checking the derivation.

4. **Pre-scatter normalization description conflates pre-scaling with the ring reduction (`reduce_scatter_ring_topology.md`, line 52).** The text states the ring sum "produces a mean rather than a sum." The ring operation always performs a *sum* reduction. The mean is an emergent property of having pre-scaled each device's contribution by 1/8 before the ring runs. Describing the ring sum itself as producing a mean is incorrect and would cause a reader to expect the CCL kernel to implement averaging logic that does not exist.

5. **Overlap feasibility rests on an unverified data-independence claim (`ccl_sensitivity_analysis.md`, lines 109 and 132).** The entire overlap argument depends on `shared_experts(residual)` being independent of `routed_output`. The document asserts that `residual` is "the original input tensor from before the all-gather," but does not cite where `residual` is bound in `moe.py`, nor does it verify this against the ground-truth forward-pass order (all_gather → gate → TTNNExperts → reduce_scatter → shared_experts → add). If `residual` is actually the post-all-gather tensor, the data-independence claim is wrong and the overlap is invalid. The missing reference to the binding site is a critical structural gap in the feasibility section.

# B Review — Chapter 2 — Pass 2

1. **`index.md` line 41 incorrectly states both CCL operations use `cluster_axis=1`.** The text reads: "Both operations communicate along `cluster_axis=1` (the 8-column axis)." However, per the verified source, `all_gather_async` does not accept a `cluster_axis` parameter at all — the parameter is absent from the call site. A reader would incorrectly expect `cluster_axis=1` to be present in the all-gather call, which contradicts the corrected call-site documentation in `all_gather_linear_topology.md`.

2. **All-gather transfer latency is numerically inconsistent across files.** `all_gather_linear_topology.md` (lines 66–67) computes `T_xfer = 12 544 B / 12 GB/s ≈ 1.04 µs` per hop, yielding `T_linear ≈ 7 × 4.04 µs ≈ 28 µs`. `ccl_sensitivity_analysis.md` (lines 33–35) uses `1.2 µs transfer` for the same hop, yielding `T_ag ≈ 7 × 4.2 µs ≈ 29 µs`. A reader cross-referencing the two files gets different numerical answers for the identical quantity from the same model inputs.

# B Review — Chapter 2 — Pass 3

1. **`ttnn.add` is annotated `moe.py:L1426` in the code block, but the same line number is claimed for `residual = x` in the prose (`ccl_sensitivity_analysis.md`, lines 106 and 109).** The inline comment `# moe.py:L1426` appears next to `output = ttnn.add(routed_output, shared_output.to_ttnn)`. Two lines later, the explanatory text states "`residual = x` is assigned at `moe.py:L1426`, before the `all_gather_async` call." A single source line cannot be both the `ttnn.add` call (post-reduce-scatter, late in the forward pass) and the `residual = x` assignment (pre-all-gather, early in the forward pass). One of these is wrong. Ground truth confirms `residual = x` is at L1426 before `all_gather_async`; the `ttnn.add` annotation is therefore incorrect, and a reader using the line number to locate either statement in `moe.py` will be sent to the wrong place.

2. **The cluster_axis-absence verification note cites the wrong source lines (`all_gather_linear_topology.md`, line 25).** The note reads "verified against `moe.py:L1429–1436`." The `all_gather_async` call site is documented throughout Chapter 2 as `moe.py:L1363–L1373`; L1429–1436 falls inside the reduce-scatter block. A reader who checks `moe.py:L1429–1436` to confirm the cluster_axis claim will be reading the wrong function entirely.

3. **`index.md` has no navigation footer, violating the chapter navigation requirement.** Every other content file in the chapter (`all_gather_linear_topology.md`, `reduce_scatter_ring_topology.md`, `ccl_sensitivity_analysis.md`) ends with a `**Next:**` link. `index.md` ends at line 62 with no footer. Per the ground-truth navigation requirement, content files must carry a navigation footer. A reader arriving at `index.md` has no forward link to follow into the chapter.

# B Review — Chapter 2 — Pass 4

1. **`all_gather_async` call site line range is wrong in multiple files.** `all_gather_linear_topology.md` (line 8) and `index.md` (line 37) both cite the call as `moe.py:L1363–L1373`. Ground truth places the `all_gather_async` call at `moe.py:L1429–1437`. A reader navigating to L1363–L1373 will find different code and draw incorrect conclusions about the call structure.

2. **`ccl_sensitivity_analysis.md` code block places `residual = x` (annotated `moe.py:L1426`) structurally before the reduce-scatter (`moe.py:L1410–L1423`), but L1426 > L1423 — the line numbers contradict the displayed order.** Ground truth confirms `residual = x` is at L1426, which is after the reduce-scatter block ends at L1423. The code block inversion misrepresents where in the forward pass `residual` is bound and is directly load-bearing for the data-independence argument used to justify overlap feasibility.

3. **`shared_experts` is annotated `moe.py:L1425` in the same code block, one line before `residual = x` at L1426, yet it is shown structurally after `residual = x`.** L1425 < L1426 means in the actual source `shared_experts` precedes `residual = x` — but the displayed code calls `self.shared_experts(residual)` after `residual` is assigned. A reader who checks the line numbers will find the shared-experts call at a line where `residual` does not yet exist, making the overlap feasibility argument unverifiable at its stated source locations.

# B Review — Chapter 2 — Pass 5

No feedback — chapter approved.

# B Review — Chapter 2 — Pass 6

No feedback — chapter approved.
