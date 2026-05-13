# ModuleList and ModuleDict — the non-callable containers

A plain Python `list` or `dict` of `Module` instances does not auto-register: `Module.__setattr__` (Chapter 2 `module_attribute_protocol.md`) only routes when the assigned value is itself a `Module` or `Parameter`. Assign `self.layers = [DecoderBlock() for _ in range(n)]` and the children vanish from `state_dict`, `parameters`, `to(device)`. `ModuleList` and `ModuleDict` exist to plug exactly that gap.

Both classes — together about 60 lines in `blaze_nn/containers.py:63-119` — share a small `_NotCallableContainer` mixin that refuses to be called.

## The "not callable" mixin

`_NotCallableContainer` (`blaze_nn/containers.py:25-39`) overrides both `__call__` and `forward` to raise:

```python
class _NotCallableContainer(Module):
    _usage_hint: str = "Iterate over it in your forward() method."

    def _raise_not_callable(self) -> None:
        raise RuntimeError(
            f"{type(self).__name__} is not callable directly. "
            f"{self._usage_hint}"
        )

    def __call__(self, *args, **kwargs):  self._raise_not_callable()
    def forward(self, *args, **kwargs):   self._raise_not_callable()
```

The point is to fail loudly and with a usable hint the moment a user mistakes one of these containers for a `Sequential`. `ModuleList` keeps the default hint; `ModuleDict` overrides it to `"Access modules by key in your forward() method."` so the error message tells you which idiom is expected (`blaze_nn/containers.py:89`).

The test anchors `tests/test_containers.py:TestModuleList::test_not_callable` and `tests/test_containers.py:TestModuleDict::test_not_callable` pin the message: `pytest.raises(RuntimeError, match="not callable")`.

## `ModuleList`

`ModuleList` inherits from both `_IndexedContainer` and `_NotCallableContainer` (`blaze_nn/containers.py:63-79`):

```python
class ModuleList(_IndexedContainer, _NotCallableContainer):
    _usage_hint = "Iterate over it in your forward() method."

    def __init__(self, modules: list[Module] | None = None):
        super().__init__()
        if modules is not None:
            self._register_indexed(modules)

    def append(self, module: Module) -> ModuleList:
        self._modules[str(len(self._modules))] = module
        return self
```

Three things matter:

1. **Integer-stringified keys, same as `Sequential`.** Children land at `"0"`, `"1"`, ... so the state-dict reads `layers.0.weight`, `layers.1.weight`, ... — matching `torch.nn.ModuleList` and feeding straight into a torch-shaped weight loader.
2. **Constructor takes a `list`**, not a positional star-args. `ModuleList([A(), B()])`, not `ModuleList(A(), B())`. (`Sequential` is the opposite.) This is a small footgun — a quick `from blaze_nn import Sequential, ModuleList` followed by `ModuleList(A(), B())` will silently treat `B()` as the (nonexistent) `modules` kwarg and crash.
3. **`append` returns `self`** for chaining: `ml.append(A()).append(B())`. The append updates `_modules` directly under the next integer key. Anchor: `tests/test_containers.py:TestModuleList::test_append`.

The canonical idiom — hold layers in a `ModuleList`, iterate inside a hand-written `forward` — shows up in `Qwen3EmbeddingModel.forward` (Chapter 4 `orchestrator_pattern.md`):

```python
# Schematic — see examples/qwen3_embedding_0_6b/modules/model.py
for layer in self.layers:
    h = layer(h, cur_pos=cur_pos, cur_pos_tensor=cur_pos_tensor)
```

The reason `Sequential` does not work here is the extra kwargs (`cur_pos`, `cur_pos_tensor`); `Sequential`'s `forward(self, x)` only passes the single positional activation. `ModuleList`'s inherited `_IndexedContainer.__iter__` returns `iter(self._modules.values())` — children in registration order, by reference.

## `ModuleDict`

`ModuleDict` is the dict-shaped sibling — non-callable, keys by user-chosen strings rather than integers (`blaze_nn/containers.py:82-119`). Constructor takes an optional `dict[str, Module]`:

```python
class ModuleDict(_NotCallableContainer):
    _usage_hint = "Access modules by key in your forward() method."

    def __init__(self, modules: dict[str, Module] | None = None):
        super().__init__()
        if modules is not None:
            for name, module in modules.items():
                self._modules[name] = module
```

The full surface mirrors `dict`:

| Method | Behavior |
|---|---|
| `md["k"]` | `__getitem__` returns the module under key `"k"` |
| `md["k"] = m` | `__setitem__` registers `m` under `"k"` |
| `"k" in md` | `__contains__` membership check |
| `for k in md:` | `__iter__` yields **keys** (not modules — note the asymmetry with `ModuleList`) |
| `md.keys()` / `md.values()` / `md.items()` | iterator-returning, dict-shaped |
| `len(md)` | child count |

```python
from blaze_nn import ModuleDict
md = ModuleDict({"attn": AttentionLike(), "mlp": MLPLike()})
md["norm"] = NormLike()      # late-bind another submodule
assert "attn" in md
for name, sub in md.items():
    ...                       # name: str, sub: Module
```

Two consequences worth pointing out:

- **Keys are real strings**, so the state-dict reads `<parent>.attn.weight`, `<parent>.mlp.weight`. This is the only blaze-nn container that lets a model author choose its sub-keys.
- **Iteration yields keys**, not modules. Override is `__iter__(self) -> Iterator[str]: return iter(self._modules.keys())` (`blaze_nn/containers.py:100-101`). The test `tests/test_containers.py:TestModuleDict::test_iter` pins this. If you want modules, use `.values()`.

The intended uses are sparser than `ModuleList`: a small fixed set of named sub-blocks (e.g. `ModuleDict({"q": Linear(...), "k": Linear(...), "v": Linear(...)})`) or a switch table keyed by a runtime-known string. The Qwen3 example does not use `ModuleDict` — its dense decoder layout is more naturally an integer-indexed `ModuleList`.

## The three containers side by side

|                              | `Sequential`                          | `ModuleList`                          | `ModuleDict`                            |
|------------------------------|---------------------------------------|---------------------------------------|-----------------------------------------|
| Constructor shape            | `Sequential(*modules)`                | `ModuleList([modules])`               | `ModuleDict({name: module})`            |
| Callable directly?           | Yes — chains in order                 | No — raises `RuntimeError`            | No — raises `RuntimeError`              |
| Child key shape              | `"0"`, `"1"`, ...                     | `"0"`, `"1"`, ...                     | user-chosen strings                     |
| State-dict prefix            | `<name>.0.weight`                     | `<name>.0.weight`                     | `<name>.<key>.weight`                   |
| Mutation API                 | none (build-and-freeze)               | `append`                              | `__setitem__`                           |
| Typical use                  | strict tensor-in / tensor-out chain   | layer stacks iterated in `forward`    | named alternative branches              |

The decision tree a model author runs:

1. Is the sub-tree a strict pipe of single-tensor stages? → `Sequential`.
2. Otherwise, is it indexable by integer and iterated inside a hand-written `forward`? → `ModuleList`.
3. Otherwise, do you want to look up children by a name you choose? → `ModuleDict`.
4. Otherwise, you do not need a container — declare the children as plain attributes on a `Module` subclass.

> **Warning:** Assigning a bare Python `list` or `dict` of modules to `self.layers = [...]` *will not* register the children. `Module.__setattr__` (`blaze_nn/modules/base.py:35-42`) only routes `Parameter` and `Module` instances; a list-of-Module falls through to `object.__setattr__`, the children never enter `_modules`, and they will be missing from `state_dict` and from `to(device)`. The two containers above exist to make the "I want a list/dict of submodules" intent explicit.

> **For contributors:** The two-class mixin pattern (`_IndexedContainer` + `_NotCallableContainer`) is also the recommended entry point for hand-rolling a new container shape — see Chapter 7 `extending_containers_and_modules.md`.

_Previous: [Sequential](sequential.md) · Next: [OpModule without subclassing](opmodule_no_subclass.md) · [Up](index.md)_
