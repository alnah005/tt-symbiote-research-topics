# OpModule without subclassing

`OpModule` is the bridge between `Module` (Chapter 2) and the universal `F.<op>` dispatch surface. In its no-subclass form it lets a model author take any op the tt-blaze backend has registered, name it, declare its learnable parameter slots, and get back a fully-functional `Module` — `state_dict` keys, `to(device)`, the lot — without writing a class. The qwen3 example uses this form for nearly every "thin wrapper around one op" that does not need a custom `forward`.

The class is defined in `blaze_nn/modules/base.py:288-501`; the constructor and default `forward` together occupy under 100 lines.

## Construction signature

```python
from blaze_nn.modules import OpModule

rmsnorm = OpModule(op="rmsnorm", params=("gamma",), epsilon=1e-5)
```

Three keyword-only knobs and an arbitrary keyword tail:

- `op: str` — the tt-blaze op name. Whatever `F.<op>` resolves to (via the dispatch machinery in Chapter 6 `functional_dispatch.md`).
- `params: Iterable[str]` — the names of learnable `Parameter` slots, **in the positional order the op expects them in `forward`**.
- `**op_kwargs` — every other kwarg is captured as construction-time kwargs that get forwarded to the op call on every invocation.

The constructor body (`blaze_nn/modules/base.py:332-365`) does five things, in order:

1. Calls `super().__init__()` to lay down `Module`'s registries.
2. If a subclass overrode `define_fused_op` (only relevant for the subclass form — see next file) and the per-class `_fused_op_defined` flag is unset, runs the hook once. For direct `OpModule(op=..., ...)` use, the default no-op runs.
3. Stashes `_op_name`, `_param_slots`, and `_op_kwargs` on the instance via `object.__setattr__` so the `Module` attribute protocol does not try to route them as parameters or submodules.
4. Consults `_lookup_user_allocated_outputs(op_name)` to populate `_required_output_names` (covered in `output_tensors.md`).
5. Calls `setattr(self, slot, Parameter())` for each name in `params`. Because `Parameter` instances *do* trip `Module.__setattr__`'s special case (Chapter 2), each slot is registered in `_parameters` and the auto-named `_name = slot` is set.

Test anchor (`tests/test_op_module.py:13-19`):

```python
def test_construction_registers_params(self):
    m = OpModule(op="rmsnorm", params=("gamma",), epsilon=1e-5)
    assert "gamma" in m._parameters
    assert isinstance(m.gamma, Parameter)
    assert m._op_name == "rmsnorm"
    assert m._param_slots == ("gamma",)
    assert m._op_kwargs == {"epsilon": 1e-5}
```

## The default `forward`

```python
def forward(self, *args, **kwargs):
    if not self._op_name:
        raise NotImplementedError(...)
    from .. import functional as F

    params = [self._parameters[slot] for slot in self._param_slots]
    merged_kwargs = {**self._op_kwargs, **kwargs}
    return getattr(F, self._op_name)(*args, *params, **merged_kwargs)
```

(`blaze_nn/modules/base.py:431-441`)

Read the call site as:

```
F.<op>(*args, *params_in_declaration_order, **{op_kwargs, **call_kwargs})
```

The shape worth committing to memory:

- **Activations come first.** Anything the user passes positionally to the module call ends up as the leading positional args of `F.<op>`.
- **Parameters follow, in declaration order.** This is why `params=("gamma",)` matters — and why the qwen3 example always lists parameters in the order the op's signature accepts them.
- **Call-time kwargs override construction-time kwargs.** The merge is `{**self._op_kwargs, **kwargs}`, so a later `m(x, epsilon=1e-4)` overrides a constructor `epsilon=1e-5`.

> **Note:** If you forgot to pass `op`, the first `forward` raises `NotImplementedError("... must either set `op` (and optionally `params`) or override forward().")`. The test `tests/test_op_module.py:test_no_op_name_raises` pins this — the empty-class-attr default `op: str = ""` (`base.py:329`) is the sentinel.

> **For contributors:** the `getattr(F, self._op_name)` call relies on the universal `__getattr__` dispatch in `blaze_nn/functional.py:63`. Any tt-blaze op registered in `BlazeOp._class_registry` becomes accessible as `F.<name>` on first reference. Chapter 6's `functional_dispatch.md` walks the closure-caching mechanism.

## Lifecycle

The construct → load → bind → call sequence is identical to every other `Module`, but with one extra step (parameter slots) that is implicit:

```python
# 1. Construct — empty Parameter slot per declared name.
rmsnorm = OpModule(op="rmsnorm", params=("gamma",), epsilon=1e-5)

# 2. Populate gamma with a ttnn.Tensor (Chapter 2: state_dict & interop).
rmsnorm.load_state_dict({"gamma": gamma_ttnn_tensor})

# 3. Bind a device handle (Chapter 2: device_binding).
rmsnorm.to(device)

# 4. Call. The first call traces forward into a graph, compiles, and runs.
out = rmsnorm(x_ttnn_tensor)
```

The `state_dict` after step 2 contains a single key, `"gamma"`. The test anchor `tests/test_op_module.py:TestOpModuleNoSubclass::test_state_dict_roundtrip` confirms identity preservation through the round trip — values are passed through verbatim, no dtype coercion, no layout move, exactly as Chapter 2 `traversal_and_state_dict.md` describes.

The `_op_kwargs` dict was set at construction time and is *not* serialized — it lives only on the in-memory instance. If you persist an `OpModule` across processes you persist the `state_dict`, not the kwargs.

> **Warning:** Step 2 writes the `ttnn.Tensor` verbatim — no dtype coercion, no memory-config conversion, no layout change. If `gamma_ttnn_tensor` was built with the wrong memory config, you find out at compile time, not at `load_state_dict` time. Build the tensor with the layout the op expects first.

## Multiple parameters

The order of `params` is the order in which they enter the call:

```python
m = OpModule(op="some_op", params=("a", "b", "c"))
# Internally: F.some_op(x, m.a, m.b, m.c, **op_kwargs)
```

The test `tests/test_op_module.py:TestOpModuleNoSubclass::test_multiple_params_preserve_order` pins the registration order: `assert list(m._parameters.keys()) == ["a", "b", "c"]`. Reordering matters because the underlying op is positional in its parameter inputs.

## Real-world use: `OpModule(op="residual_add")`

The cleanest qwen3 use of the no-subclass form is the residual-add shim that appears in two modules — `Qwen3DecoderLayer` (`examples/qwen3_embedding_0_6b/modules/decoder_layer.py:30`) and `Qwen3Attention` (`examples/qwen3_embedding_0_6b/modules/attention.py:51`):

```python
self.residual_add = OpModule(op="residual_add")
# ...
return self.residual_add(post_attn, mlp_out)
```

Two activations, zero parameters, zero kwargs — a one-liner that turns `F.residual_add(post_attn, mlp_out)` into a properly-registered submodule. The reason to bother registering it (rather than just calling `F.residual_add` inline) is that the orchestrator's `forward` runs as plain Python (Chapter 4 `orchestrator_pattern.md`), so each call to `self.residual_add(...)` opens its own tracing context, compiles the op into its own little graph, and caches it on the instance.

> **For contributors:** The "register the slot so it gets its own compile" mechanism is in Chapter 5 `module_call_path.md`. The `_compiled_cache` field on `Module` is the cache.

## What `OpModule(...)` is not

- **Not the only way to wrap one op.** The subclass form (next file) is preferred when you want a custom `forward`, custom constructor signature, or class-level docs. Use the no-subclass form when the default `forward` is what you want.
- **Not a tt-blaze op declaration.** It does not register a new op with the backend; it consumes one that is already there. If `F.<op>` does not resolve, you will get a `ValueError("Unknown blaze op")` from the tracing context the first time `forward` runs (Chapter 5 `tracing_contexts.md`). The `define_fused_op` hook (`blaze_nn/modules/base.py:367-376`) is a classmethod that is only called when a subclass overrides it — the no-subclass form always assumes the op is already in `BlazeOp._class_registry`.
- **Not a `Sequential`-style composer.** It calls exactly one op, plus whatever positional/keyword arguments the op accepts. If you need a composition of ops, write a plain `Module` subclass with a hand-written `forward` (Chapter 4 has many examples).

> **For contributors:** The `_collect_user_args` method and the `_ua_*` attribute convention apply to *every* `OpModule`, including the no-subclass form. `output_tensors.md` introduces them at user-level; Chapter 6 `caller_allocated_outputs_internals.md` carries the full mechanism.

_Previous: [ModuleList and ModuleDict](modulelist_and_moduledict.md) · Next: [OpModule as a base class](opmodule_subclass.md) · [Up](index.md)_
