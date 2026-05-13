# Traversal and the state-dict contract

`Module`'s attribute protocol ([module_attribute_protocol.md](module_attribute_protocol.md)) populated two `OrderedDict`s: `_parameters` and `_modules`. This file walks the six helpers that consume them — `parameters`, `named_parameters`, `modules`, `named_modules`, `state_dict`, `load_state_dict` — and the identity-preserving contract that links the last two.

## Recursion rule: own first, then children

All four traversal helpers follow the same shape: yield this module's own contents first, then recurse into each registered child in declaration order. This matches PyTorch's behavior so users can transfer mental models 1:1. The dotted-name convention is also identical: `parent.child.weight`, with the dots inserted on descent.

### `parameters()` — flat iterator over `Parameter` objects

The plainest helper (see `blaze_nn/modules/base.py:163-167`):

```python
def parameters(self) -> Iterator[Parameter]:
    for param in self._parameters.values():
        yield param
    for module in self._modules.values():
        yield from module.parameters()
```

Yields `Parameter` instances, no names. Useful when the caller only needs the underlying tensors (typically `for p in m.parameters(): do_something(p.data)`). `tests/test_module.py:TestTraversal.test_parameters_nested` pins the count: own params first, then submodule params recursively.

### `named_parameters(prefix="")` — dotted-name iterator

The variant most code reaches for (see `blaze_nn/modules/base.py:169-175`):

```python
def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
    for name, param in self._parameters.items():
        full_name = f"{prefix}.{name}" if prefix else name
        yield full_name, param
    for mod_name, module in self._modules.items():
        sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
        yield from module.named_parameters(prefix=sub_prefix)
```

The `if prefix` guard avoids a leading dot at the root: `m.named_parameters()` yields `weight`, not `.weight`. Descending one level produces `layer1.weight`; two levels, `layers.0.weight`; and so on. `tests/test_module.py:test_named_parameters_nested` pins the dotted form.

### `modules()` and `named_modules(prefix="")` — over `Module` objects

Symmetric helpers for the submodule tree (see `blaze_nn/modules/base.py:177-186`). `modules()` yields `self` first, then each child's full subtree — so a root module with two children produces three modules total. `named_modules()` yields `("", self)` for the root, then `("child", child)`, and so on, with the same prefix-joining rule. `tests/test_module.py:test_modules` and `test_named_modules` cover both.

## `state_dict()` returns dotted keys to tensors

The save side of the contract (see `blaze_nn/modules/base.py:190-199`):

```python
def state_dict(self, prefix: str = "") -> OrderedDict:
    """Return an OrderedDict of name -> ttnn.Tensor (or None if unset)."""
    result = OrderedDict()
    for name, param in self._parameters.items():
        key = f"{prefix}.{name}" if prefix else name
        result[key] = param._tensor
    for mod_name, module in self._modules.items():
        sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
        result.update(module.state_dict(prefix=sub_prefix))
    return result
```

The key shape is identical to `named_parameters`. The value is `param._tensor` directly — no wrapping, no copy, no detach. The value type is **`ttnn.Tensor | None`**: parameters with nothing assigned yet show up with `None`, which is the correct sentinel for "not loaded." `tests/test_module.py:test_state_dict_empty` pins the `None` case; `test_state_dict_with_data` pins the populated case.

The return type is `OrderedDict`, and the order matches declaration order in both `_parameters` and `_modules` — the same order `named_parameters` would emit. This makes round-tripping a `state_dict` to JSON / pickle and back reproducible across processes.

> **Note:** the dict values can be heterogeneous in practice. A partially loaded model with some `Parameter`s bound and others still `None` is a valid in-memory state; the framework does not enforce "all or nothing." This makes incremental loading and per-layer weight surgery straightforward.

## `load_state_dict()` is strict, identity-preserving, verbatim

The load side (see `blaze_nn/modules/base.py:201-232`) is the most important contract in this chapter. The body, eliding the symmetric `KeyError` arm:

```python
def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    self._state_loaded = True
    own_params = set(self._parameters.keys())
    child_prefixes = set(self._modules.keys())

    for key, value in state_dict.items():
        parts = key.split(".", 1)
        if len(parts) == 1:
            name = parts[0]
            if name in own_params:
                self._parameters[name]._tensor = value
            else:
                raise KeyError(
                    f"Unexpected key '{name}' for {type(self).__name__}. "
                    f"Expected one of: {sorted(own_params)}"
                )
        else:
            mod_name, remainder = parts
            if mod_name in child_prefixes:
                self._modules[mod_name].load_state_dict({remainder: value})
            else:
                ...  # symmetric KeyError("Unexpected module prefix ...")
```

The elided arm at `base.py:227-231` raises `KeyError("Unexpected module prefix '<name>' for <ClassName>. Expected one of: [...]")` with the sorted list of expected child names.

Four behaviors fall out of those lines, each worth pinning explicitly.

### 1. Split on the *first* dot, then descend

`key.split(".", 1)` splits on the first dot only. A key like `layers.0.w1` becomes `("layers", "0.w1")`. The receiver looks `"layers"` up in its `_modules`; the child (`ModuleList` in this case) receives `{"0.w1": value}` and runs the same algorithm. Recursion terminates at a leaf with no dots in the key, where the value is assigned to a `Parameter` slot.

### 2. Strict on both sides

Unknown names raise — no silent skip:

- A leaf key not in `_parameters` raises `KeyError("Unexpected key '<name>' for <ClassName>. Expected one of: [...]")` (`base.py:219-222`).
- A dotted prefix not in `_modules` raises `KeyError("Unexpected module prefix '<name>' for <ClassName>. Expected one of: [...]")` (`base.py:227-231`).

Both messages include the sorted list of expected names — typos surface with their fix in the error text. `tests/test_module.py:test_load_state_dict_unknown_key` and `test_load_state_dict_unknown_module` pin both branches.

There is **no `strict=False` escape hatch** in the current API. Partial loads must be expressed by passing only the keys the caller wants to set; mismatches are errors, not warnings. (Contributors adding a `strict=False` mode should also extend `tests/test_state_dict.py`.)

### 3. `_state_loaded = True` is set unconditionally

The first line of the body flips the flag, even before any value has been written. The flag is consumed by `OpModule.__call__` to decide whether to call `init_torch_params()` as a fallback at the outer call boundary (`base.py:425-428`). The contract is: "the user has taken responsibility for the parameter tensors." A `KeyError` raised mid-load leaves the flag set — by design, because partial state was already written before the error.

### 4. Values are written verbatim — no conversion of any kind

The single line `self._parameters[name]._tensor = value` is the entire write. The framework does **not** call any method on `value`. In particular:

- **No dtype coercion.** A `ttnn.Tensor` whose dtype is `bfloat8_b` lands as `bfloat8_b`. The framework will not promote to `bfloat16` or to `float32`.
- **No device move.** A `ttnn.Tensor` that lives on host stays on host. A tensor on `mesh_device` stays on `mesh_device`. `module.to(device)` does *not* re-home parameters after load (`device_binding.md` covers this). The tensor must already be on the device you intend to compute on.
- **No layout conversion.** A `ROW_MAJOR_LAYOUT` tensor stays row-major; a `TILE_LAYOUT` tensor stays tiled. The compiler will fail at run time if the layout disagrees with what the op expects.
- **No memory-config rewrite.** Sharded vs. interleaved, DRAM vs. L1, shard spec — all are taken from the value as given.

The docstring at `blaze_nn/modules/base.py:202-207` states the contract: "Values are written verbatim; the framework does not interpret or convert them. Pass ttnn tensors that are already on the intended device with the desired memory config." If you need to re-shape, re-tile, or re-shard a weight, do it on the `ttnn.Tensor` *before* calling `load_state_dict`. The interop helpers in [Interop at the boundary](interop_at_the_boundary.md) accept a `memory_config` argument so the conversion happens once, in the user's loader.

## The identity-preserving roundtrip rule

The single most important behavioral guarantee in this chapter:

> **`m2.load_state_dict(m1.state_dict())` writes each value onto `m2`'s parameter slot by identity** — the `ttnn.Tensor` object stored on `m1.weight._tensor` is the same Python object stored on `m2.weight._tensor` after the call. No clone. **No dtype coercion. No device move. No layout conversion.**

The canonical test is `tests/test_state_dict.py:test_deep_model_roundtrip`:

```python
def test_deep_model_roundtrip(self):
    m1 = DeepModel()
    sd_in = { ... 'layers.0.w1': object(), ..., 'final': object() }
    m1.load_state_dict(sd_in)

    m2 = DeepModel()
    m2.load_state_dict(m1.state_dict())
    for key, value in sd_in.items():
        # ... each m2.<path>._tensor is the SAME object as sd_in[key]
```

The exact tensor *object* the user provided to `m1.load_state_dict` is what `m2.layers[0].w1._tensor` returns after the roundtrip. No copies are made anywhere along the chain. This matters for two reasons:

1. **Memory.** Loading a large model into a second `Module` instance does not duplicate parameter memory.
2. **Mutation.** A buffer-style parameter mutated in place (e.g. KV cache) updates visibly through every reference. This is leveraged in Chapter 4's `tensor_lifetimes.md`.

The flip side is: if you want two independent copies of a weight, you must clone the `ttnn.Tensor` yourself — `state_dict` will not do it for you.

> **Note:** the same identity rule applies to `None`. A `state_dict` containing `"weight": None` will *clear* that parameter — `_tensor` becomes `None`, and the next forward will see an uninitialized slot. This is rarely what you want, but it is consistent with the verbatim rule.

## Reaching for interop

In practice, model authors construct the dict they pass to `load_state_dict` by walking a HuggingFace `state_dict` (a `dict[str, torch.Tensor]`), remapping keys, and converting each torch tensor to a `ttnn.Tensor` with the appropriate `memory_config`. The `examples/qwen3_embedding_0_6b/weight_loader.py` does exactly this — Chapter 4 `layout_and_weight_loader.md` walks the full pipeline. The torch ↔ ttnn step uses `blaze_nn.interop.to_device_tensor`, covered next.

_Previous: [Module attribute protocol](module_attribute_protocol.md) · Next: [Device binding](device_binding.md) · [Up](index.md)_
