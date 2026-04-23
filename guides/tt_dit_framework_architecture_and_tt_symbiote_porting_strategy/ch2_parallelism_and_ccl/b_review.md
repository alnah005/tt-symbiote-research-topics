# Chapter 2 Review -- Agent B (Critic)

## Issue 1: TT-Symbiote async CCL capability materially understated (mapping_to_symbiote.md)

**Location:** mapping_to_symbiote.md, "Comparison: CCL Infrastructure" table and Gap #2 ("No Async CCL Operations").

**Problem:** The chapter states that TT-Symbiote has "No helper" for all-gather and reduce-scatter, and that it uses only "synchronous `ttnn.reduce_scatter` and `ttnn.all_gather`." This is incorrect. The `TT_CCL` class (in `models/tt_transformers/tt/ccl.py`) is accompanied by `tt_all_reduce` and `tt_all_gather` helper functions in the same file that call `ttnn.experimental.all_gather_async` and `ttnn.experimental.reduce_scatter_minimal_async` with semaphore management, hyperparameters (`chunks_per_sync`, `num_workers_per_link`, `num_buffers_per_channel`), and barrier semaphores. The `DistributedConfig.__post_init__` instantiates `TT_CCL`, so TT-Symbiote already has access to async CCL via these helpers.

The accurate characterization is: TT-Symbiote's *distributed linear modules* (`TTNNLinearIColShardedWRowSharded` etc.) call synchronous `ttnn.reduce_scatter`/`ttnn.all_gather` directly, but the framework's CCL infrastructure (`TT_CCL` + `tt_all_reduce`/`tt_all_gather`) already supports async operations. The gap is that the distributed linear modules do not use these helpers, not that the infrastructure is absent.

This mischaracterization inflates the "Async CCL" row in the Feature Parity table from "Not supported (sync only)" to what should be "Available via `tt_all_reduce`/`tt_all_gather` but not used by distributed linear modules."

**Severity:** Material misconception -- overstates porting effort for a critical infrastructure component.

## Issue 2: TT-Symbiote CCL comparison table has incorrect claims about helper methods and hyperparameters (mapping_to_symbiote.md)

**Location:** mapping_to_symbiote.md, "TT-DiT CCLManager vs. TT-Symbiote TT_CCL" table.

**Problem:** Several rows are factually wrong when checked against `models/tt_transformers/tt/ccl.py`:
- "All-gather helper: No helper -- callers use raw `ttnn.all_gather`" -- Incorrect. `tt_all_gather()` is a helper that wraps `ttnn.experimental.all_gather_async`.
- "Reduce-scatter helper: No helper -- callers use raw `ttnn.reduce_scatter`" -- Incorrect. `tt_all_reduce()` wraps `ttnn.experimental.reduce_scatter_minimal_async`.
- "Hyperparameter tuning: None" -- Incorrect. Both `tt_all_reduce` and `tt_all_gather` pass `chunks_per_sync=10`, `num_workers_per_link=2`, `num_buffers_per_channel=2` (though not shape-dependent like TT-DiT). The table should say "Fixed (not shape-dependent)" rather than "None."
- "Reset method: Not provided" -- This is accurate.
- "Persistent buffers: No" -- This is accurate.

**Severity:** Incorrect implementation description -- three table cells state the opposite of what the source code provides.

---

No other issues found in the remaining files.

- `index.md`: Factual claims about `DiTParallelConfig`, `ParallelFactor`, submesh creation, and the three parallelism axes all match the source code in `parallel/config.py` and `parallel/manager.py`. Navigation links are correct and clickable. The numerical example (2x4 and 4x8 Motif configs) is consistent with stated mesh shapes.
- `ccl_manager.md`: Semaphore counts (RS=3, AG=2, NP=1, SR=1, Barrier=1), ping-pong logic, persistent buffer caching, hyperparameter values, and VAE-specific CCL operations all match `parallel/manager.py` and `parallel/config.py` exactly.
- `parallel_linear_layers.md`: Weight sharding via `mesh_axes`, `_prepare_torch_state` transposition, SwiGLU interleaving, FSDP all-gather pattern, bias zero-padding in `RowParallelLinear`, `prepare_chunked_linear_output` reshaping, and `minimal_matmul` usage all verified against `layers/linear.py`.

---

## Pass 2 Review

**Verified that Pass 1 Issues 1 and 2 have been corrected.** The CCL comparison table now accurately reflects `tt_all_reduce()`/`tt_all_gather()` helpers and their hardcoded hyperparameters. The Feature Parity table's "Async CCL ops" row now correctly states the helpers exist but distributed linear modules bypass them.

### Re-verification against source code

All four chapter files were re-checked against the current source code:

- `index.md`: `DiTParallelConfig`, `ParallelFactor`, `EncoderParallelConfig`, `VAEParallelConfig`, `VaeHWParallelConfig`, `MochiVAEParallelConfig` all match `parallel/config.py`. Motif 2x4 config `(cfg=2 axis 0, sp=1 axis 0, tp=4 axis 1)` and 4x8 config `(cfg=2 axis 1, sp=4 axis 0, tp=4 axis 1)` verified against `pipelines/motif/pipeline_motif.py`. Flux1 configs (no CFG, 2x4/4x4/4x8 entries) verified against `pipelines/flux1/pipeline_flux1.py`. Submesh creation pattern matches pipeline source.
- `ccl_manager.md`: Semaphore counts, ping-pong logic, persistent buffer shapes, hyperparameter values, VAE CCL ops, and the `reset_global_semaphores` method (which correctly omits barrier semaphores, matching the source) all verified against `parallel/manager.py` and `parallel/config.py`.
- `parallel_linear_layers.md`: `ColParallelLinear` weight `mesh_axes=[fsdp_mesh_axis, mesh_axis]`, `RowParallelLinear` weight `mesh_axes=[mesh_axis, fsdp_mesh_axis]`, bias zero-padding, FSDP all-gather, SwiGLU interleaving, attention block pattern (both QKV and output use `ColParallelLinear` with all-gather, confirmed in `blocks/attention.py`), and feed-forward pattern (`ColParallelLinear` + `RowParallelLinear` pair) all verified against `layers/linear.py`, `blocks/attention.py`, and `layers/feedforward.py`.
- `mapping_to_symbiote.md`: `DistributedConfig`, `DistributedTensorConfig`, default batch+channel sharding, `TT_CCL` instantiation, distributed linear class names and their forward methods, CCL comparison table entries, and gap analysis all verified against `core/run_config.py`, `modules/linear.py`, and `tt_transformers/tt/ccl.py`.

**No feedback -- chapter approved.**
