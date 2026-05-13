# Module attribute protocol — how `__setattr__` routes

If [parameter.md](parameter.md) is the "what is in a slot" file, this one is the "how does the slot get there" file. Every PyTorch user knows the idiom `self.weight = Parameter()` inside `__init__` — what makes that line do something useful is `Module`'s overridden `__setattr__`. This file walks the routing rules, the matching `__getattr__` and `__delattr__`, the abstract `forward` contract, and the entry-point `__call__` (mechanism only; the graph build is Chapter 5).

## `super().__init__()` lays down five buckets

Before any user assignment can be routed, `Module.__init__` builds the internal containers it will route into (see `blaze_nn/modules/base.py:26-31`):

```python
def __init__(self):
    object.__setattr__(self, "_parameters", OrderedDict())
    object.__setattr__(self, "_modules", OrderedDict())
    object.__setattr__(self, "_device_config", None)
    object.__setattr__(self, "_compiled_cache", {})
    object.__setattr__(self, "_state_loaded", False)
```

Each line uses `object.__setattr__` to bypass the very `__setattr__` we are about to install. If `Module.__init__` used plain `self._parameters = OrderedDict()`, the override below would inspect the right-hand side, see it is neither a `Parameter` nor a `Module`, and recurse into `object.__setattr__` anyway — but only by accident of the fallthrough branch. The explicit form makes the boot-strap step unambiguous and matches PyTorch's pattern.

The five buckets play these roles:

- `_parameters: OrderedDict[str, Parameter]` — every `Parameter` attribute, in declaration order.
- `_modules: OrderedDict[str, Module]` — every child `Module` attribute, in declaration order.
- `_device_config: DeviceConfig | None` — set by [`to(device)`](device_binding.md); starts `None`.
- `_compiled_cache: dict` — currently a hook with no occupants. Reserved for cached compiled programs.
- `_state_loaded: bool` — flipped to `True` by `load_state_dict`; consulted by `OpModule.__call__` to decide whether to auto-init parameters at the outer call boundary (see `blaze_nn/modules/base.py:427-428`).

> **For contributors:** the `_compiled_cache` and `_state_loaded` flags exist for the tracing pipeline, not the attribute protocol. Chapter 5 `module_call_path.md` documents the full state machine — when each flag is read, when it is written, and which compile path consumes which.

A subclass that forgets to call `super().__init__()` first will hit `AttributeError` the first time `self.weight = Parameter()` runs, because `__setattr__` reaches into a `_parameters` dict that does not exist yet. This is the one boilerplate line every `Module` subclass must include.

## `__setattr__` routes by type

The override is short enough to read top to bottom (see `blaze_nn/modules/base.py:35-42`):

```python
def __setattr__(self, name: str, value: Any) -> None:
    if isinstance(value, Parameter):
        value._name = name
        self._parameters[name] = value
    elif isinstance(value, Module):
        self._modules[name] = value
    else:
        object.__setattr__(self, name, value)
```

Three branches, evaluated in this order:

1. **`isinstance(value, Parameter)`** — stamp the attribute name onto the parameter and register it under `_parameters[name]`. After `self.weight = Parameter()` the parameter is reachable through `m._parameters["weight"]`, through `m.weight` (via the `__getattr__` below), and its `_name` attribute equals `"weight"`. `tests/test_module.py:TestModuleAttributes.test_parameter_name_set` pins exactly this.
2. **`isinstance(value, Module)`** — register the submodule under `_modules[name]`. No name is stamped onto the child — child modules do not need to know what they are called inside their parent, because traversal threads the name in via `prefix` arguments (see [Traversal and state dict](traversal_and_state_dict.md)).
3. **Anything else** — delegate to `object.__setattr__`, which writes the attribute into the instance `__dict__` as it normally would. Plain Python values (ints, strings, lists, custom helper objects) live on the instance the way they would on any class.

There is no fourth branch for `list[Module]` or `dict[Module]`. Assigning a list of child modules to an attribute stores them in `__dict__` like any other list — *they are not registered, and traversal will not find them*. This is the most common pitfall when porting from torch:

```python
# BUG: layers will not appear in m.modules() or m.state_dict().
self.layers = [Decoder() for _ in range(N)]

# Correct:
self.layers = blaze_nn.ModuleList([Decoder() for _ in range(N)])
```

`ModuleList` and `ModuleDict` are container `Module`s — assigning them goes through the `isinstance(value, Module)` branch and they then bring their children into traversal via their own `_modules` dict. Chapter 3 (`ch3_containers_and_opmodule/modulelist_and_moduledict.md`) covers the containers in full.

## `__getattr__` mirrors PyTorch's lookup order

The dual override (`blaze_nn/modules/base.py:44-51`) makes the registered parameters and submodules look like ordinary attributes:

```python
def __getattr__(self, name: str) -> Any:
    params = self.__dict__.get("_parameters")
    if params is not None and name in params:
        return params[name]
    modules = self.__dict__.get("_modules")
    if modules is not None and name in modules:
        return modules[name]
    raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
```

Python only calls `__getattr__` when normal lookup fails — so plain instance attributes (the third `__setattr__` branch above) shadow the parameter/module registries. The order is:

1. Instance `__dict__` (Python's default).
2. `_parameters[name]`.
3. `_modules[name]`.
4. `AttributeError`.

The `self.__dict__.get("_parameters")` guard handles the boot-strap case where `__getattr__` fires *before* `super().__init__()` has installed the buckets. Without it, the line `params = self._parameters` would itself recurse through `__getattr__` and stack-overflow.

`tests/test_module.py:test_getattr_parameter` and `test_getattr_module` exercise the success paths; `test_getattr_missing` exercises the `AttributeError`. The error message includes the class name to make typos easy to spot in stack traces.

## `__delattr__` keeps the registries in sync

The third member of the trio (`blaze_nn/modules/base.py:53-59`):

```python
def __delattr__(self, name: str) -> None:
    if name in self._parameters:
        del self._parameters[name]
    elif name in self._modules:
        del self._modules[name]
    else:
        object.__delattr__(self, name)
```

`del m.weight` removes the parameter from `_parameters`; `del m.layer1` removes the submodule from `_modules`; `del m.some_int` falls through to `object.__delattr__` for plain attributes. The symmetry matters because `state_dict()` and the traversal helpers iterate over the registries — leaving stale entries would produce ghost keys. `tests/test_module.py:test_delattr_parameter` and `test_delattr_module` pin the contract.

## `forward()` is abstract on the base

`Module.forward` is intentionally a stub that raises (see `blaze_nn/modules/base.py:63-66`):

```python
def forward(self, *args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(
        f"{type(self).__name__} must implement forward()"
    )
```

Concrete subclasses — every model author writes one — override this. The error message names the subclass to make the missing override obvious. `OpModule` provides a default `forward` that dispatches to `F.<op>(...)` (see `blaze_nn/modules/base.py:431-441`); plain `Module` subclasses must supply their own.

## `__call__` is the entry point — three branches

`Module.__call__` is what runs when user code writes `out = model(x)`. The full body (see `blaze_nn/modules/base.py:68-82`):

```python
def __call__(self, *args: Any, **kwargs: Any) -> Any:
    from .._tracing import _get_active_context

    if _get_active_context() is not None:
        return self.forward(*args, **kwargs)

    is_compose = getattr(
        getattr(type(self), "forward", None),
        "_blaze_nn_compose",
        False,
    )

    if is_compose:
        return self._call_compose(*args, **kwargs)
    return self._call_graph(*args, **kwargs)
```

Three distinct paths:

1. **Active-context short-circuit (`base.py:71`).** If a tracing context is already open — i.e. this `Module` was called from inside a parent module's `forward` that is currently being traced — `__call__` returns `self.forward(...)` immediately. No new context, no new compile. The parent's tracing context will record the ops invoked inside `self.forward` as part of the parent's graph. This is what makes a `Module` composable: an inner `Linear` called from inside an outer `forward` contributes its ops to the outer graph rather than building and running its own.
2. **Compose path (`base.py:80-81`).** If the subclass's `forward` was decorated with `@blaze_nn.compose`, dispatch to `_call_compose`. The decorator (defined in `blaze_nn/__init__.py:38-48`) is a one-bit flag — it simply sets `forward_fn._blaze_nn_compose = True` and returns the function unchanged. `__call__` reads that flag off the *class* attribute (the `getattr(type(self), "forward", None)` indirection), which means decorating an *instance* attribute would not flip the path — the decorator is intended at class definition time.
3. **Graph path (default, `base.py:82`).** Otherwise, dispatch to `_call_graph`. This is what every qwen3 sub-module uses.

> **For contributors:** the actual graph build, the `GraphTracingContext` / `ComposeTracingContext` machinery, the `wrap_input` / `_bind_parameters_to_context` calls, and the `BlazeCompiler.compile(...).run()` finale all live behind `_call_graph` (`base.py:86-122`) and `_call_compose` (`base.py:126-144`). Chapter 5 `module_call_path.md` walks them line by line. For this chapter the mechanism is: `__call__` decides the path; `_call_*` runs it.

The Chapter 4 `orchestrator_pattern.md` discussion of `__call__` *overrides* in `Qwen3Attention`, `Qwen3DecoderLayer`, and `Qwen3EmbeddingModel` builds on top of this default. An orchestrator's `__call__` is the two-liner `def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)`, which skips all three branches above and runs `forward` as plain Python — needed for host-side hops that cannot live in a single tt-blaze graph. That's a Chapter 4 concern; here the point is that the default `__call__` is one of three things, all of which eventually reach `forward`.

> **Warning:** never call `module.forward(x)` directly — it skips `__call__` entirely, so no tracing context opens and no compile happens. `F.<op>` calls inside that `forward` will raise `RuntimeError("... no active tracing context")`. The supported invocation is always `module(x)`.

## `OpModule.__call__` adds two pre-flight checks

`OpModule` overrides `__call__` (`blaze_nn/modules/base.py:413-429`) to insert two checks that only apply at the outer call boundary — i.e. when `_get_active_context()` is `None`:

1. **Missing user-allocated outputs raise early.** If the underlying tt-blaze op declares `user_allocated_outputs` and the caller never called `set_output_tensor`, a `RuntimeError("...has unset required output tensor(s): ...")` fires before `forward` runs (`base.py:417-423`).
2. **Lazy `init_torch_params` on first call.** If `load_state_dict` was never called (`_state_loaded == False`) and the subclass implements `_torch_init_specs`, parameters are auto-initialized from `torch.randn` onto the recorded device (`base.py:427-428`).

Both checks are skipped on nested (re-entry) calls — the parent module owns the lifecycle for those. Chapter 3 `opmodule_subclass.md` walks the auto-init mechanism; Chapter 3 `output_tensors.md` walks the output-tensor setup hooks.

With routing and the call entry point understood, the next file walks what traversal and `state_dict` actually do with the registries those routes populate.

_Previous: [Parameter](parameter.md) · Next: [Traversal and state dict](traversal_and_state_dict.md) · [Up](index.md)_
