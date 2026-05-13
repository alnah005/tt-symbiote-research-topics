# Parameter — the trivial-looking class

`Parameter` is the smallest non-trivial class in blaze-nn. The entire file is roughly thirty lines (see `blaze_nn/parameter.py`), and almost every interesting thing about it is something it deliberately does *not* do. Read it as a contract, not as a data structure.

## Two slots, no shape, no dtype

A `Parameter` has exactly two pieces of state (see `blaze_nn/parameter.py:16-18`):

```python
class Parameter:
    def __init__(self):
        self._name: str = ""
        self._tensor: Any = None
```

- `_tensor: Any` is the payload — the `ttnn.Tensor` that the user has bound to this slot. Its type is `Any` on purpose: the framework never inspects it, never calls a method on it inside `Parameter`, never asks what dtype or device it lives on. The Chapter 1 ttnn-native contract (`ch1_why_blaze_nn/ttnn_native_contract.md`) is enforced by *not* doing things here, not by a runtime check.
- `_name: str` is the attribute name under which the `Parameter` was assigned to its owning `Module`. It starts empty; `Module.__setattr__` fills it in the moment the assignment happens (see `blaze_nn/modules/base.py:36-37`, walked in the next file).

There is no `requires_grad`, no `device` field, no `dtype`, no shape declaration at construction. `Parameter()` takes no arguments. The shape, the layout, and the memory config are whatever the `ttnn.Tensor` the user assigns happens to carry — the `Parameter` is a typed slot, not a tensor descriptor.

## The `data` property is a passthrough

The `data` property is a one-line getter/setter pair on `_tensor` (see `blaze_nn/parameter.py:20-26`):

```python
@property
def data(self) -> Any:
    return self._tensor

@data.setter
def data(self, value: Any) -> None:
    self._tensor = value
```

No copy, no clone, no detach, no device check. `param.data = ttnn_tensor` binds the tensor by reference; later reads of `param.data` return the same object. The framework treats parameter values as opaque — `tests/test_parameter.py:test_data_property_roundtrip` exercises exactly this behavior with a plain `object()` sentinel.

## Two population paths

A `Parameter` slot becomes useful only once a tensor is in `_tensor`. There are exactly two sanctioned ways to put one there:

1. **Direct assignment.** `module.weight.data = ttnn_tensor`. Useful for one-off rebinds, debugging, and tests.
2. **Bulk via `Module.load_state_dict`.** `module.load_state_dict({"weight": ttnn_tensor, ...})`. This is the path real models use; it descends recursively through child modules and writes `param._tensor` for every matched key. The full contract is in [Traversal and state dict](traversal_and_state_dict.md).

A third path exists for `OpModule` subclasses that declare `_torch_init_specs` — `OpModule.init_torch_params()` builds tensors from `torch.randn` and routes them through `load_state_dict` (see `blaze_nn/modules/base.py:460-501`). That helper is the only place in the framework that mentions `torch`, and it imports torch lazily.

## `__repr__` is a shape-aware heuristic

`Parameter.__repr__` tries three things in order (see `blaze_nn/parameter.py:28-34`):

```python
def __repr__(self) -> str:
    if self._tensor is not None:
        shape = getattr(self._tensor, "shape", None)
        if shape is not None:
            return f"Parameter(shape={tuple(shape)})"
        return f"Parameter(tensor={self._tensor!r})"
    return "Parameter(uninitialized)"
```

- If `_tensor is None`, the repr is `Parameter(uninitialized)`.
- If `_tensor` has a `.shape` attribute (every real `ttnn.Tensor` does), the repr is `Parameter(shape=(...))`.
- Otherwise it falls back to `Parameter(tensor=<repr>)`. The fallback exists so the framework-only tests, which assign plain `object()` sentinels, still produce a readable repr — and so debugging output is never broken by an unexpected tensor type.

This heuristic is the only place in `Parameter` that touches `_tensor` at all, and it does so via `getattr` rather than an `isinstance` check. The class would still work if `_tensor` were a `numpy.ndarray`, a `bytes` blob, or `None`. That is by design.

## `_name` and why it matters later

`_name` looks like dead bookkeeping when you read `Parameter` in isolation, but it is the key that connects a `Parameter` to the rest of the system:

- `Module.__setattr__` writes `value._name = name` the instant a `Parameter` is bound (`blaze_nn/modules/base.py:36-38`). After `self.weight = Parameter()`, the parameter's `_name` is `"weight"`.
- During tracing, `_bind_parameters_to_context` calls `ctx.register_input(name, param._tensor)` (`blaze_nn/modules/base.py:148-151`). The name passed in is the dict key from `_parameters`, but the same string lives on the parameter itself — convenient when an op needs to refer back to its own port by name.
- The graph-input port name that the tt-blaze compiler sees is derived from this name. The `tensors` dict that `_call_graph` hands to `BlazeCompiler` is keyed by that string (see `blaze_nn/modules/base.py:106`).

> **For contributors:** Chapter 5 `tracing_contexts.md` walks the full path from `_name` to graph-input port. For Ch1–4 audiences it is enough to know that the name you write in your `Module.__init__` is the name that ends up on the wire.

## What `Parameter()` is not

To collect the negatives in one place — these are recurring user questions:

- **No autograd.** blaze-nn does not own a gradient engine. There is no `requires_grad` flag, no `.grad` field, no backward graph (Chapter 1 `what_it_is.md` lists this among the four "what blaze-nn is not" points).
- **No torch tensor storage.** A `Parameter` never holds a `torch.Tensor` directly. If you have torch weights, bridge through `blaze_nn.interop.to_device_tensor` ([Interop at the boundary](interop_at_the_boundary.md)) and then assign.
- **No shape contract at construction.** `Parameter()` doesn't know it will eventually hold a `(d_model, d_ffn)` tensor. The shape is whatever `ttnn.Tensor` arrives later. This is why `OpModule` subclasses that need a known shape express it through `_torch_init_specs` rather than through the `Parameter` constructor.
- **No device.** The parameter does not record which device its tensor lives on. The owning `Module` records a `DeviceConfig` via `to(device)` ([Device binding](device_binding.md)), but the parameter itself is a slot.
- **No identity beyond `id()`.** Two `Parameter()` instances are distinct Python objects even if both are uninitialized — there is no interning, no pooling.

The simplicity is the point. Every behavior that PyTorch's `nn.Parameter` carries — gradient tracking, device move, dtype promotion — has been deliberately *not* implemented here, because each one would either drag torch into the framework or contradict the ttnn-native invariant. `Parameter` is the narrowest possible interface for "a named slot that an owning `Module` knows about and that `state_dict` can find."

The next file walks how that ownership is wired up.

_Previous: [Chapter 1 — Why blaze-nn](../ch1_why_blaze_nn/getting_started.md) · Next: [Module attribute protocol](module_attribute_protocol.md) · [Up](index.md)_
