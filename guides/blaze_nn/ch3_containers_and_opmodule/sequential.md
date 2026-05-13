# Sequential — the one callable container

Of the three blaze-nn containers, `Sequential` is the only one that is itself callable. It mirrors `torch.nn.Sequential`: pass a series of modules to the constructor, call the container with an input, and each child runs in declaration order with the previous child's output as the next child's input.

The whole class lives in 19 lines at `blaze_nn/containers.py:42-60` and inherits from a 16-line `_IndexedContainer` mixin (`blaze_nn/containers.py:8-22`). There is no other state, no other API.

## Construction

`Sequential(*modules)` takes a positional argument list, not a list literal — the same shape as torch:

```python
from blaze_nn import Sequential, Module, Parameter

class A(Module):
    def __init__(self):
        super().__init__()
        self.w = Parameter()
    def forward(self, x):
        return x  # placeholder

stack = Sequential(A(), A(), A())
```

Internally, `__init__` calls `super().__init__()` (the standard `Module.__init__`, which lays down `_parameters`, `_modules`, `_device_config`, `_compiled_cache`, `_state_loaded` — see Chapter 2's `module_attribute_protocol.md`) and then `_register_indexed(modules)` — which writes each child into `self._modules` under the **string-cast integer index** `str(idx)` (`blaze_nn/containers.py:11-13`):

```python
def _register_indexed(self, modules: Any) -> None:
    for idx, module in enumerate(modules):
        self._modules[str(idx)] = module
```

That string-cast integer index — `"0"`, `"1"`, `"2"`, … — is the load-bearing detail.

## State-dict keys nest as digits

Because children are stored under `"0"`, `"1"`, `"2"`, ... the state-dict produced by `Sequential` keys its children's parameters by integer prefix. The test anchor (`tests/test_containers.py:TestSequential.test_state_dict`) asserts exactly that:

```python
s = Sequential(DummyModule(), DummyModule())
sd = s.state_dict()
assert "0.w" in sd
assert "1.w" in sd
```

When the `Sequential` is itself a child of another `Module` (the usual case), the parent's `__setattr__` records the container under whatever attribute name the user chose — say `self.layers = Sequential(...)` — and the recursive `state_dict()` walk (Chapter 2 `traversal_and_state_dict.md`) prepends that name. The resulting keys read `layers.0.weight`, `layers.1.weight`, ... — the same convention `torch.nn.Sequential` uses, which means a weight-loader written against a torch reference model translates almost verbatim.

> **Note:** The string-keying-by-index detail is not just a coincidence with torch — it is how `Sequential` participates uniformly in `state_dict` / `load_state_dict` (Chapter 2). Because `_modules` is a plain `dict[str, Module]` and Python dicts preserve insertion order, the recursive walks in `parameters` / `named_parameters` / `named_modules` see children in declaration order.

## Calling: just walks `_modules` in order

`Sequential.forward(x)` is a four-line loop (`blaze_nn/containers.py:57-60`):

```python
def forward(self, x: Any) -> Any:
    for module in self._modules.values():
        x = module(x)
    return x
```

A few things follow:

1. **Single-tensor pipeline.** `Sequential` chains exactly one positional argument. Anything more complex — multi-input modules, conditionals, residuals — needs a plain `Module` with a hand-written `forward`. Chapter 4 shows several examples where a decoder block holds an explicit `Module` instead of a `Sequential` for precisely this reason.
2. **`module(x)` not `module.forward(x)`.** Each child is invoked through its `__call__`, not its `forward` — so each child opens its own tracing context, compiles its own graph, and runs. This is the same per-call compilation pattern Chapter 4 `orchestrator_pattern.md` documents at user-level and Chapter 5 `module_call_path.md` walks at internals depth.
3. **Inside an active tracing context**, the child's `__call__` short-circuits via the active-context check at `blaze_nn/modules/base.py:71` and re-uses the parent's context, so a `Sequential` nested inside another module's `forward` collapses into a single graph rather than three sub-graphs. Forward-link to Chapter 5 `module_call_path.md` for the full re-entry semantics.

> **For contributors:** the "each child compiles its own graph at the outer call boundary, then re-uses the parent's context when nested" claim is the active-context short-circuit at `blaze_nn/modules/base.py:71`. Chapter 5's `module_call_path.md` walks the full mechanics.

## Iteration, length, indexing

The three Pythonic accessors come from `_IndexedContainer` (`blaze_nn/containers.py:15-22`):

```python
def __len__(self) -> int:
    return len(self._modules)

def __iter__(self) -> Iterator[Module]:
    return iter(self._modules.values())

def __getitem__(self, idx: int) -> Module:
    return list(self._modules.values())[idx]
```

The test anchors (`tests/test_containers.py:TestSequential`) cover all three: `test_len`, `test_iter` (iteration yields children in registration order), `test_getitem` (positional indexing by integer). Note that `__getitem__` materializes the values list each call — access is **O(n), not O(1)** — fine for the layer counts blaze-nn cares about, but not a tight-loop primitive.

## What `Sequential` is not

It is worth naming the limits explicitly, since these are the points where a torch reader's instinct misleads.

- **Not a function composer.** It chains module calls, not arbitrary callables; you cannot pass `torch.nn.ReLU()` (no module of that type exists in `blaze_nn`) or a bare lambda.
- **No `OrderedDict` constructor overload.** Unlike `torch.nn.Sequential`, there is no `Sequential(OrderedDict([("a", ...), ("b", ...)]))` form. Children are keyed by string-cast integer only. If you need named children, use `ModuleDict` (next file) inside a hand-written `Module`.
- **No `add_module`, no `insert`, no slice-indexing.** The container is intentionally minimal; if you need to mutate it after construction, use `ModuleList`, which exposes `append`.
- **Not a place for non-`Module` callables.** `__setattr__` only routes `Parameter`/`Module` into the registries (Chapter 2); a bare function dropped into a `Sequential` would fail later inside `forward` with a `TypeError`, but the container does not pre-validate.

## When to reach for `Sequential`

In practice, `Sequential` is most useful at the top of a port — an MLP block, an embedding-plus-norm prelude, or any other strictly linear pipeline of tensor-in / tensor-out modules. Once the data flow forks (residuals, multi-tensor calls, head reshapes), the qwen3 example reaches for plain `Module` subclasses instead. Chapter 4 `composing_submodules.md` shows the cutover concretely: `Qwen3MLP` deliberately is *not* a `Sequential` because the SwiGLU activation needs both the gate and up projections side by side. The Qwen3 walkthrough uses **zero** `Sequential` containers for exactly this reason — every decoder layer takes extra kwargs (`cur_pos`, `cur_pos_tensor`) and threads residuals around its sub-blocks.

> **Warning:** Because `Sequential` is `_IndexedContainer` only (not `_NotCallableContainer`), calling it directly does what you expect. The other two containers in this chapter raise `RuntimeError` when called — that is a deliberate distinction, not a missing feature.

_Previous: [Chapter 2 — Module, Parameter, and the device boundary](../ch2_module_and_parameter/interop_at_the_boundary.md) · Next: [ModuleList and ModuleDict](modulelist_and_moduledict.md) · [Up](index.md)_
