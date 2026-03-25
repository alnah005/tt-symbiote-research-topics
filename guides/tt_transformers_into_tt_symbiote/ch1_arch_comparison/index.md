# Chapter 1 — Architectural Comparison: TT Symbiote vs TT Transformers

This chapter establishes the foundational vocabulary for the rest of the guide by examining both stacks side by side. TT Symbiote (`models/experimental/tt_symbiote/`) is a module-replacement framework designed for easy onboarding: a developer wraps a standard `nn.Module` hierarchy with `TTNNModule` subclasses, and the framework handles weight preprocessing, device placement, and operator dispatch automatically. TT Transformers (`models/tt_transformers/`) is a hand-written, production-grade LLM inference stack that bypasses PyTorch entirely in favor of `LightweightModule` and achieves peak throughput through per-layer dtype selection, DRAM-sharded weight loading, compute kernel tuning (`hifi2`/`hifi4`), paged KV-cache, and trace capture. The integration problem is straightforward to state but non-trivial to solve: Symbiote's framework lowers the barrier to hardware acceleration, but it does not yet expose the performance knobs that TT Transformers uses routinely. The goal of this guide is to identify exactly where the two stacks diverge and specify how TT Transformers optimizations can be brought into Symbiote's framework without sacrificing the ergonomic onboarding model.

## Reading order

1. [`tt_symbiote_internals.md`](./tt_symbiote_internals.md) — `TTNNModule`, `TorchTTNNTensor`, run modes, module replacement, and hardware gating
2. [`tt_transformers_internals.md`](./tt_transformers_internals.md) — `LightweightModule`, `Transformer`, `Attention`, `MLP`, `Generator`, and per-layer optimizations

## Summary table

| Feature | TT Symbiote | TT Transformers |
|---|---|---|
| **Abstraction level** | Framework: wraps existing `nn.Module` trees via class substitution; developer writes a `TTNNModule` subclass and declares weight lifecycle methods | Stack: every component (`Transformer`, `Attention`, `MLP`) is written from scratch as a `LightweightModule`; no PyTorch autograd overhead |
| **Model scope** | General-purpose; any `nn.Module` topology can be adapted | Specialized for autoregressive LLM inference (prefill + decode loop with trace capture) |
| **Weight management** | Declarative three-phase lifecycle: `preprocess_weights_impl()` on host → `move_weights_to_device_impl()` → optional auto-deallocation via `@deallocate_weights_after`; state tracked with `_preprocessed_weight` and `_weights_on_device` flags | Imperative: weights are loaded into TTNN tensors at `__init__` time using `ttnn.as_tensor` with DRAM-sharded memory configs (`create_dram_sharded_mem_config`); per-layer dtype is resolved at construction via `ModelOptimizations.get_tensor_dtype()` |
| **Device placement** | Centralized via `TTNNModule.to_device(device)` and `DistributedConfig`; the `TT_SYMBIOTE_RUN_MODE` environment variable selects the dispatch implementation at process startup | Per-module: each `Attention` and `MLP` instance receives a `mesh_device` and a `tt_ccl` object directly; topology is fixed at construction |
| **Debugging** | Rich run-mode infrastructure: `NORMAL`, `NORMAL_WITH_FALLBACK`, `SEL`, `DPL`, `DPL_NO_ERROR_PROP`, `CPU`, `TRACED`; the `DispatchManager` records per-operator timing breakdowns exportable to CSV | Minimal; debugging is done by running reference PyTorch models alongside and comparing PCC; no built-in fallback dispatch |

## The integration problem

TT Symbiote provides easy onboarding: a new module is three method overrides away from running on device, and the run-mode system makes it trivial to compare TTNN output against a reference PyTorch layer without changing application code. TT Transformers provides hand-tuned LLM performance: every matmul carries an explicit `compute_kernel_config` (`hifi2`/`hifi4`), weights are loaded into DRAM-sharded layouts at construction, prefill and decode paths diverge at the operation level, and the trace-capture loop eliminates host dispatch overhead entirely.

The gap between the two is a set of concrete design decisions: TT Symbiote's `TTNNModule` does not yet carry per-layer `ModelOptimizations`, does not distinguish prefill from decode mode, and does not expose the DRAM-sharded weight-loading pattern that TT Transformers' `MLP` and `Attention` rely on. Bridging that gap — bringing TT Transformers optimizations into Symbiote's framework — is the subject of this guide.

**What's next:** Read [`tt_symbiote_internals.md`](./tt_symbiote_internals.md) for a detailed walkthrough of every Symbiote primitive you will need to understand before tackling the integration.
