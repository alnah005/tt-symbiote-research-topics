# blaze-nn Guide

A reader's guide to **blaze-nn**, the PyTorch-style `nn.Module` interface that traces `forward()` into a **tt-blaze** dataflow graph carrying **ttnn** tensors for Tenstorrent hardware. The guide is layered: Chapters 1–4 target model authors porting a network onto the framework's public API, while Chapters 5–7 take contributors through the tracing internals, the op-dispatch path, and the recipes for extending the framework safely.

## How to Use This Guide

| If you want to... | Go to |
|-------------------|-------|
| Get the one-page mental model and run the three test tiers | [Chapter 1 — Why blaze-nn and how it fits together](ch1_why_blaze_nn/index.md) |
| Use the `Module` / `Parameter` API and save/load weights | [Chapter 2 — Module, Parameter, and the device boundary](ch2_module_and_parameter/index.md) |
| Use containers, `OpModule`, and pre-built ops (`Linear`, `RMSNorm`) | [Chapter 3 — Containers, OpModule, and pre-built ops](ch3_containers_and_opmodule/index.md) |
| Port a model end-to-end (the `qwen3_embedding_0_6b` walkthrough) | [Chapter 4 — Authoring models: the Qwen3 walkthrough](ch4_qwen3_walkthrough/index.md) |
| Understand what happens between `model(x)` and `program.run()` | [Chapter 5 — Tracing internals: from `Module.__call__` to `program.run()`](ch5_tracing_internals/index.md) |
| Add an op alias, placement hint, or caller-allocated output | [Chapter 6 — Op dispatch, the registry, and caller-allocated outputs](ch6_dispatch_and_registry/index.md) |
| Extend blaze-nn safely (op wrappers, fused ops, tests, anti-patterns) | [Chapter 7 — Extending blaze-nn](ch7_extending/index.md) |
| Cross the torch ↔ ttnn boundary for `load_state_dict` | [Chapter 2 — Interop at the boundary](ch2_module_and_parameter/interop_at_the_boundary.md) |
| Troubleshoot a failure mode against the test suite | [Chapter 7 — Contributing checklist](ch7_extending/contributing_checklist.md) |
| Look up the test file that backs a specific guide section | [Chapter 7 — Testing strategy (reverse index)](ch7_extending/testing_strategy.md) |

## Chapter Index

| # | Title | Description | Key concepts |
|---|-------|-------------|--------------|
| 1 | [Why blaze-nn and how it fits together](ch1_why_blaze_nn/index.md) | Position blaze-nn against PyTorch, tt-blaze, and ttnn; install the package and run the three test tiers. | mental model, ttnn-native contract, three test tiers |
| 2 | [Module, Parameter, and the device boundary](ch2_module_and_parameter/index.md) | The two foundational classes, the attribute protocol, identity-preserving state-dict, `module.to(device)` semantics, and the torch ↔ ttnn `interop` helpers. | `Module`, `Parameter`, `state_dict`, `to(device)`, `interop` |
| 3 | [Containers, OpModule, and pre-built ops](ch3_containers_and_opmodule/index.md) | `Sequential` / `ModuleList` / `ModuleDict`, both `OpModule` forms, caller-allocated output tensors, and the pre-built `Linear` and `RMSNorm`. | containers, `OpModule`, `set_output_tensor`, `Linear`, `RMSNorm` |
| 4 | [Authoring models: the Qwen3 walkthrough](ch4_qwen3_walkthrough/index.md) | The only end-to-end model in the repo, used to compose the public API into a real port: HF → ttnn loader, three-way tensor vocabulary, orchestrators, and address baking. | `qwen3_embedding_0_6b`, Parameter / Buffer / GraphInput, orchestrator, address baking |
| 5 | [Tracing internals: from `Module.__call__` to `program.run()`](ch5_tracing_internals/index.md) | The private half of `Module.__call__`, the `_active_context` module-global, the three tracing context classes, and the `TensorProxy` handle. | `_call_graph`, `_call_compose`, `TracingContext`, `TensorProxy` |
| 6 | [Op dispatch, the registry, and caller-allocated outputs](ch6_dispatch_and_registry/index.md) | The `F.<any_op>` dispatch path, the `OpInfo` registry table (aliases, placement, sender), and the full `user_allocated_outputs` chain. | `_dispatch`, `__getattr__`, `_registry.py`, `user_allocated_outputs`, `define_fused_op` |
| 7 | [Extending blaze-nn](ch7_extending/index.md) | Concrete recipes: add an op wrapper, synthesize a fused op, extend containers, the test taxonomy reverse index, and the contributing checklist. | op wrappers, fused ops, test tiers, anti-patterns, compose-mode gap |

## Quick Reference

| API / Concept | Purpose | Where to learn more |
|---------------|---------|---------------------|
| `Module` | Base class — declare `Parameter`s in `__init__`, define `forward()` | [Ch2 — Module attribute protocol](ch2_module_and_parameter/module_attribute_protocol.md) |
| `Parameter` | One-slot holder for a `ttnn.Tensor`; populated by direct `.data =` or by `load_state_dict` | [Ch2 — Parameter](ch2_module_and_parameter/parameter.md) |
| `Sequential` | The one callable container — composes modules into a chain | [Ch3 — Sequential](ch3_containers_and_opmodule/sequential.md) |
| `ModuleList`, `ModuleDict` | Non-callable containers — index / look up submodules; you write the call order | [Ch3 — ModuleList and ModuleDict](ch3_containers_and_opmodule/modulelist_and_moduledict.md) |
| `OpModule(op=..., params=...)` | Single-op module built on the fly without subclassing | [Ch3 — OpModule without subclassing](ch3_containers_and_opmodule/opmodule_no_subclass.md) |
| `blaze_nn.Linear`, `blaze_nn.ops.RMSNorm` | Pre-built parameter-bearing ops every model reaches for first | [Ch3 — Pre-built modules](ch3_containers_and_opmodule/prebuilt_modules.md) |
| `blaze_nn.functional.F.<any_op>` | Universal op dispatch — resolves lazily against tt-blaze's op registry | [Ch6 — Functional dispatch](ch6_dispatch_and_registry/functional_dispatch.md) |
| `module.to(device)` | Records a `DeviceConfig` on every submodule; does **not** move tensors | [Ch2 — Device binding](ch2_module_and_parameter/device_binding.md) |
| `state_dict()` / `load_state_dict()` | Identity-preserving save / load — values are stored verbatim, not copied | [Ch2 — Traversal and state-dict](ch2_module_and_parameter/traversal_and_state_dict.md) |
| `blaze_nn.interop.to_device_tensor` / `to_torch` | The torch ↔ ttnn boundary helpers users call before `load_state_dict` | [Ch2 — Interop at the boundary](ch2_module_and_parameter/interop_at_the_boundary.md) |
| `set_output_tensor` / `set_output_tensors` | Bind caller-allocated output buffers for an `OpModule` | [Ch3 — User-allocated output tensors](ch3_containers_and_opmodule/output_tensors.md) / [Ch6 — Caller-allocated outputs internals](ch6_dispatch_and_registry/caller_allocated_outputs_internals.md) |
| `_ua_*` attribute prefix | Forwards kwargs through tracing to the compiler (e.g. `_ua_output`, `_ua_sender_core`) | [Ch3 — User-allocated output tensors](ch3_containers_and_opmodule/output_tensors.md) / [Ch4 — Buffers and address baking](ch4_qwen3_walkthrough/buffers_and_address_baking.md) |
| `@blaze_nn.compose` decorator | Switch a method from graph mode into compose mode | [Ch5 — The module call path](ch5_tracing_internals/module_call_path.md) |
| `define_fused_op` hook | Lazy `BlazeOp` synthesis — fuse a new op into the registry on first use | [Ch7 — Adding a fused op](ch7_extending/add_a_fused_op.md) |
| Parameter / Buffer / GraphInput vocabulary | The three-way tensor-lifetime split the qwen3 port uses | [Ch4 — Tensor lifetimes](ch4_qwen3_walkthrough/tensor_lifetimes.md) |

## Prerequisites

- Working familiarity with PyTorch's `nn.Module` / `nn.Parameter` idioms (Chapters 2–4 assume the reader recognizes `__setattr__` routing, `state_dict`, and `to(device)` by name from the torch side).
- Basic familiarity with `ttnn.Tensor`, `MemoryConfig`, and shard specs — blaze-nn carries these opaquely but every Chapter 4 example places one explicitly.
- For Chapters 5–7: know tt-blaze's `BlazeOp`, `FusedOp`, `BlazeGraph`, `BlazeCompiler`, and `BlazeOp._class_registry` by name — Chapter 5's opening blockquote restates this assumption.

## Source Code Location

- **Framework:** `blaze-nn/blaze_nn/` (Python import: `blaze_nn`)
- **Tests:** `blaze-nn/tests/` — three tiers: framework-only, dispatch-integration (`pytest.importorskip("blaze")`), and parity (device required)
- **Example model:** `blaze-nn/examples/qwen3_embedding_0_6b/` — the only end-to-end port in the repo, used throughout Chapter 4
- **Generated artifacts:** `blaze-nn/generated/`
- **Upstream — tt-blaze:** Python import `blaze`; provides `BlazeOp`, `blaze.fuse()`, `BlazeGraph`, `BlazeCompiler`, and the op registry blaze-nn dispatches against
- **Upstream — ttnn:** Python import `ttnn`; the only tensor type that crosses a `Module` boundary
- **Repo URL:** [github.com/tenstorrent/blaze-nn](https://github.com/tenstorrent/blaze-nn) — see `README.md` at the repo root for install and environment recipes
