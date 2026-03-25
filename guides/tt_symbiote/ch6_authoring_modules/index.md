# Chapter 6 — Authoring a New TTNN Module

## Overview

TT Symbiote accelerates PyTorch models by replacing selected `nn.Module` instances with
`TTNNModule` subclasses that run their forward pass on Tenstorrent hardware using the
TTNN library. Chapter 6 explains when and how to author such a module from scratch.

## When to write a custom module

**Compose existing modules when:**

- The computation you need is already covered by a built-in module (e.g., `TTNNLinear`,
  `TTNNLinearGelu`, `TTNNLinearSilu`). See the reference implementations in
  `models/experimental/tt_symbiote/modules/linear.py`.
- Your module can be expressed as a sequential chain of existing `TTNNModule` children.
  The `preprocess_weights_impl`, `move_weights_to_device_impl`, and
  `deallocate_weights_impl` methods on `TTNNModule` already recurse over
  `self.__dict__` children, so composing them requires no override of those methods.

**Write a custom module when:**

- The op-level computation requires a TTNN primitive not covered by an existing module.
- You need non-standard weight preprocessing (e.g., custom quantisation, custom layout
  conversion, per-device sharding strategies).
- The replacement target has a shape contract (e.g., mandatory 4-D reshape before
  `ttnn.linear`) that no existing module handles correctly.
- You need to guard a forward implementation to a specific set of device architectures
  with `@run_on_devices`.

## Chapter contents

| File | Description |
|---|---|
| [`implementation_guide.md`](implementation_guide.md) | Step-by-step authoring guide with a complete worked example |
| [`fallback_and_debugging.md`](fallback_and_debugging.md) | Debugging workflow, run modes, dispatcher logging, and common mistakes |

## Source files referenced in this chapter

| Path | Purpose |
|---|---|
| `models/experimental/tt_symbiote/core/module.py` | `TTNNModule` base class, lifecycle methods, `deallocate_weights_after`, `run_on_devices`, `DeviceArch` |
| `models/experimental/tt_symbiote/core/run_config.py` | Run-mode classes (`NormalRun`, `NormalRunWithFallback`, `SELRun`, `DPLRun`, `DPLRunNoErrorProp`), `DispatchManager` |
| `models/experimental/tt_symbiote/core/dispatcher.py` | Dispatcher interface; `TT_SYMBIOTE_DISPATCHER` env var |
| `models/experimental/tt_symbiote/utils/module_replacement.py` | `register_module_replacement_dict` |
| `models/experimental/tt_symbiote/modules/linear.py` | `TTNNLinear` — reference implementation |

---

**Navigation:**
[^ Chapter list](../plan.md) |
[Chapter 5 — Built-in Modules](../ch5_builtin_modules/) |
[Next: Implementation Guide](implementation_guide.md)
