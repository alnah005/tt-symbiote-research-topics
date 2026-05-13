# Device binding — what `module.to(device)` does and doesn't

`module.to(device)` is the second contract a PyTorch user expects to "just work" — and the second one where blaze-nn's ttnn-native invariant changes the meaning. This file walks what `Module.to` actually does in twelve lines of code, what it deliberately does not do, and how `DeviceConfig` exposes the device handle to the rest of the framework.

## What `to(device)` does

The full method (see `blaze_nn/modules/base.py:236-247`):

```python
def to(self, device: Any) -> Module:
    """Bind this module (and children) to a device.

    This only records the device for graph compilation. Parameters are
    expected to already be ttnn tensors on the appropriate device — the
    framework does not convert or move tensors during ``to``.
    """
    dc = DeviceConfig(device)
    object.__setattr__(self, "_device_config", dc)
    for module in self._modules.values():
        module.to(device)
    return self
```

Three operations, in order:

1. **Wrap the device handle in a `DeviceConfig`.** `DeviceConfig(device)` (see `blaze_nn/device.py:15-20`) stashes the device on `_device` and leaves `_grid_config = None` for lazy initialization.
2. **Stash it on `_device_config`.** Using `object.__setattr__` to bypass the type-routing `__setattr__` from [module_attribute_protocol.md](module_attribute_protocol.md) — a `DeviceConfig` is neither a `Parameter` nor a `Module`, so the fallthrough branch would handle it anyway, but the explicit form is consistent with the boot-strap in `__init__`.
3. **Recurse into children.** Every entry in `_modules` receives the same `device`. Each child constructs its own `DeviceConfig` — they share a device handle, not a `DeviceConfig` instance. This is intentional: it means a child's `_device_config` field can be inspected independently, and a future API that allows per-submodule overrides has room to grow without breaking the recursion.

The method returns `self`, matching torch's convention so callers can chain (`model = MyModel().to(device)`).

## What `to(device)` does NOT do

> **Warning:** `module.to(device)` does **not** move parameter tensors onto the device, does **not** change their layout, and does **not** promote their dtype. The docstring above states this directly. blaze-nn assumes that every `ttnn.Tensor` in `state_dict` is already on the device, in the layout, with the memory config, that subsequent ops expect. Mismatches surface at compile or runtime, not at `to()`.

The contrast with PyTorch is sharp. In PyTorch, `module.to("cuda")` walks every `Parameter`, asks each to move itself to CUDA, and (if requested) promotes dtypes. None of that happens here — `to` never touches `_parameters`, never calls a method on the wrapped `ttnn.Tensor`, never promotes dtype, never converts layout.

The corollary is the workflow rule: **construct your `ttnn.Tensor` with the correct device, layout, and memory config in your loader, then `load_state_dict`, then `to(device)`.** The order of the last two does not actually matter for parameter placement (because `to` does not touch parameters), but it does matter for the next section: trying to `forward` before binding raises.

## `forward` before `to(device)` raises

The companion private helper `_resolve_device_config` is what every `_call_graph` / `_call_compose` invocation reaches for first (see `blaze_nn/modules/base.py:249-254`):

```python
def _resolve_device_config(self) -> DeviceConfig:
    if self._device_config is not None:
        return self._device_config
    raise RuntimeError(
        f"{type(self).__name__} has no device. Call module.to(device) first."
    )
```

Calling a `Module` before binding produces a clear error pointing at the missing setup line. `tests/test_module.py:TestDeviceError.test_no_device_raises` pins the message text.

There is no "default device" fallback. Forgetting `to(device)` is always an explicit error — never a silent run on the wrong target. This is a deliberate departure from frameworks that pick a default (CPU, the first visible GPU, etc.); blaze-nn refuses to guess.

> **Note:** the recursion in `to` means binding the *outermost* `Module` is sufficient — every child gets the same device via the `for module in self._modules.values(): module.to(device)` loop. Calling `.to(device)` on a child individually is legal but not required.

## `DeviceConfig` from the user's perspective

`DeviceConfig` is the small wrapper around the device handle (see `blaze_nn/device.py:15-46`). From a model author's perspective there are only two things to know about it:

1. **You almost never construct one directly.** `Module.to(device)` constructs the `DeviceConfig` for you. The class is documented as "hidden from end users" in its module docstring (`blaze_nn/device.py:1-8`).
2. **The framework reads grid information through it lazily.** The `grid_config` property defers the import of `blaze.role_engine.GridConfig` until first access (see `blaze_nn/device.py:26-31`). This keeps `import blaze_nn` cheap and tt-blaze-free, consistent with the Chapter 1 ttnn-native contract (`ch1_why_blaze_nn/ttnn_native_contract.md`).

The exposed properties — `device`, `grid_config`, `sender_core`, `all_cores`, `matmul_cores`, `build_ab_grids` — are all things tracing and dispatch consult, not things the user reads. They appear in this chapter only so the reader recognizes the shape if it surfaces in a stack trace.

> **For contributors:** the `GridConfig`, sender/matmul core selection, and `build_ab_grids` machinery are how blaze-nn maps op placement onto the physical core grid. Chapter 5 `tracing_contexts.md` walks `_resolve_grid` (`blaze_nn/_tracing.py:82`), which reads `device_config.matmul_cores` for ops flagged `uses_matmul_cores` in the registry and `device_config.all_cores` otherwise. Chapter 6 `registry.md` documents the flags. The user-side view ends with: "`to(device)` makes the device handle available to the framework; everything else is internal."

See [Interop at the boundary](interop_at_the_boundary.md) for the recap of the four ttnn-native rules across the chapter. The next file is the loader-side bridge.

_Previous: [Traversal and state dict](traversal_and_state_dict.md) · Next: [Interop at the boundary](interop_at_the_boundary.md) · [Up](index.md)_
