# Chapter 6: Weight Loading and Preprocessing

## Overview

Weight loading is the process of taking a trained PyTorch model's `state_dict` and converting it into TTNN tensors that reside on Tenstorrent hardware. Both TT-DiT and TT-Symbiote solve this problem, but they take fundamentally different architectural approaches. Understanding these two paradigms is essential for deciding what can be reused when porting models between frameworks and what must be rewritten.

This chapter examines both pipelines in detail, compares their design trade-offs, and identifies concrete reuse opportunities.

## The Two Paradigms at a Glance

| Aspect | TT-DiT | TT-Symbiote |
|---|---|---|
| **Entry point** | `Module.load_torch_state_dict(state_dict)` | `TTNNModule.from_torch(torch_layer)` |
| **Weight transformation** | `_prepare_torch_state()` hook per module | `preprocess_weights_impl()` override per module |
| **Device placement** | Immediate inside `Parameter.load_torch_tensor()` | Deferred to `move_weights_to_device_impl()` |
| **Mesh distribution** | Declarative via `Parameter(mesh_axes=...)` | Imperative via `ttnn.shard_tensor_to_mesh_mapper()` |
| **Serialization** | Native `.tensorbin` cache via `save()`/`load()` | No built-in serialization layer |
| **State dict manipulation** | Tree-recursive with `pop_substate()` helpers | Manual attribute assignment from `torch_layer` |
| **Lifecycle phases** | 1 phase (load = transform + place) | 3 phases (from_torch, preprocess, move_to_device) |

## Chapter Files

- **[`tt_dit_weight_pipeline.md`](./tt_dit_weight_pipeline.md)** -- The TT-DiT weight loading flow: HuggingFace `state_dict` through `load_torch_state_dict`, the `_prepare_torch_state` hook, `Parameter.load_torch_tensor`, tensor distribution via `mesh_axes`, and the `.tensorbin` serialization cache.

- **[`symbiote_weight_pipeline.md`](./symbiote_weight_pipeline.md)** -- The TT-Symbiote weight loading flow: `from_torch` factory methods, the three-phase lifecycle (`preprocess_weights_impl`, `move_weights_to_device_impl`, `deallocate_weights_impl`), and a comparative assessment identifying reuse opportunities and migration costs.

## Relationship to Other Chapters

- [Chapter 1](../ch1_architecture_overview/index.md) introduces the `Module`/`Parameter` class hierarchy in TT-DiT and the `TTNNModule` base class in TT-Symbiote. This chapter builds directly on those foundations.
- [Chapter 7](../ch7_tracing_and_performance/index.md) covers tracing and performance, where the weight loading strategy directly impacts trace compatibility -- weights must be stable on device before trace capture begins.

---

**Next:** [`tt_dit_weight_pipeline.md`](./tt_dit_weight_pipeline.md)
