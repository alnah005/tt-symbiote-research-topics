# Module Replacement

Module replacement is the process of walking a PyTorch model graph and swapping each `nn.Module` instance whose class appears in a caller-supplied mapping with a freshly created `TTNNModule` instance.

**Source file:** `models/experimental/tt_symbiote/utils/module_replacement.py`

---

## Public Entry Point: `register_module_replacement_dict`

```python
def register_module_replacement_dict(
    model,
    old_class_to_new_class_dict,
    model_config=None,
    exclude_replacement: Optional[Set[str]] = None,
) -> Dict[str, TTNNModule]:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` or `TTNNModule` | Root model to traverse and modify in-place |
| `old_class_to_new_class_dict` | `dict[type, type]` | Maps each PyTorch class to its `TTNNModule` replacement class |
| `model_config` | `dict` or `None` | Arbitrary configuration dict forwarded to every created `TTNNModule` via `set_model_config`; `None` is treated as `{}` |
| `exclude_replacement` | `set[str]` or `None` | Set of module-name strings (from `model.named_modules()`) that must not be replaced even if their class is in the mapping; must contain only `str` values |

### Return value

A `dict` mapping each replaced module's `module_name` string (the `TTNNModule.module_name` property) to the newly created `TTNNModule` instance. Only modules that were actually replaced and are `TTNNModule` instances appear in this dict. The dict is the canonical input for the subsequent weight-preprocessing and weight-upload loops.

### What it does internally

1. Calls `model.named_modules()` to build a reverse map `{module_object: dotted_name}`. This snapshot is taken once before any replacement occurs.
2. Delegates immediately to `register_module_replacement_dict_with_module_names`, passing the pre-built name map.
3. Returns the accumulated `result` dict.

---

## Internal Recursive Traversal: `register_module_replacement_dict_with_module_names`

```python
def register_module_replacement_dict_with_module_names(
    model,
    old_class_to_new_class_dict,
    model_config,
    module_names,
    exclude_replacement: Optional[Set[str]] = None,
    result: Optional[Dict[str, TTNNModule]] = None,
):
```

This function handles three distinct container types depending on whether the current node is an `nn.Module` or a `TTNNModule`.

### `nn.Module` path

1. **`_modules` dict** — iterates `model._modules.items()`. For each child whose class is in the mapping, calls `initialize_module`. If a replacement is produced, assigns it back as `model._modules[name]` and records it in `result`. If the class is not in the mapping, recurses.

2. **Dict attributes** — iterates `dir(model)`, skipping names that start with `_`. For any attribute that is a `dict`, inspects each value: if the value is an `nn.Module` instance whose class is in the mapping, calls `initialize_module`; otherwise recurses. The dict is modified in-place (`value[k] = new_module`). Non-`nn.Module` values (including `TTNNModule` instances, which are not `nn.Module`) are passed to the recursive call but never replaced.

3. **List/tuple attributes** — for any attribute that is a `list` or `tuple`, collects elements into a temporary list; for each element that is an `nn.Module` instance whose class is in the mapping, calls `initialize_module`; otherwise recurses on the element. Writes the result back via `setattr(model, attr_name, type(value)(ls_value))`. This preserves the original container type (`list` vs `tuple`). As with dicts, non-`nn.Module` elements are never replaced.

### `TTNNModule` path

When the current node is a `TTNNModule`, the `_modules` step is skipped (TTNNModule does not have `_modules`). Only dict attributes and list/tuple attributes are scanned, using the same logic as the `nn.Module` path. The attribute scan additionally excludes `"torch_layer"` to avoid accidentally replacing the stored fallback layer.

---

## `initialize_module`

```python
def initialize_module(
    old_module,
    old_class_to_new_class_dict,
    module_names,
    model_config,
    exclude_replacement: Optional[Set[str]] = None,
) -> Optional[Union[TTNNModule, nn.Module]]:
```

This is the per-module factory. It is called only when `old_module.__class__` is confirmed to be in `old_class_to_new_class_dict`.

### Logic

1. **Exclusion check** — if `old_module` is found in `module_names` and its name is in `exclude_replacement`, returns `None` (signals: do not replace).
2. **Construction** — calls `old_class_to_new_class_dict[old_module.__class__].from_torch(old_module)`. The `from_torch` classmethod (defined on `TTNNModule`) creates a new instance and stores the original PyTorch module as `_fallback_torch_layer`.
3. **Name assignment and config injection** — if the result is a `TTNNModule`: (a) if `old_module` is in `module_names`, sets `new_module._unique_name = module_names[old_module]` and calls `new_module.override_children_module_names()` to propagate the name prefix into any children; (b) calls `new_module.set_model_config(model_config)`, which stores `model_config` (or `{}` if `None`) into `new_module._model_config`. Both (a) and (b) are skipped if the new module is not a `TTNNModule`.
4. Returns the new module, or `None` if the exclusion check fired.

### Name propagation via `override_children_module_names`

`override_children_module_names` calls `set_module_name_recursively(self, self.module_name)`, which walks the new module's `__dict__` and assigns dotted `_unique_name` values to all descendant `TTNNModule` and `nn.Module` instances. Names for dict-valued children are formatted as `prefix.attr[key]`; names for list/tuple-valued children are formatted as `prefix.attr[index]`.

---

## The `exclude_replacement` Parameter

### Discovering names before excluding them

Because names come from `model.named_modules()`, you can enumerate them before calling `register_module_replacement_dict`:

```python
# Print all module names in the model
for name, module in model.named_modules():
    print(name, type(module).__name__)

# Then exclude specific ones
module_dict = register_module_replacement_dict(
    model,
    nn_to_ttnn_dict,
    model_config=cfg,
    exclude_replacement={"model.layers.23.self_attn"},
)
```

`exclude_replacement` is matched against the name captured at the pre-replacement snapshot, so it is safe to build the set from `model.named_modules()` immediately before the call.

---

## Selective Replacement Patterns

### Replace all layers except one

```python
module_dict = register_module_replacement_dict(
    model,
    {TransformerBlock: TTTransformerBlock},
    model_config=cfg,
    exclude_replacement={"model.layers.0"},  # keep first block as PyTorch
)
```

After this call, `model.layers[0]` remains the original `TransformerBlock`. All other `TransformerBlock` instances are replaced.

### Mix PyTorch and TTNN layers in the same model

Because the mapping is class-based, only classes explicitly listed in `old_class_to_new_class_dict` are replaced. Any class not in the dict is left as-is, regardless of nesting depth.

```python
nn_to_ttnn_dict = {
    MyAttentionLayer: TTAttentionLayer,
    # MyMLPLayer is intentionally absent: keep it as PyTorch
}
module_dict = register_module_replacement_dict(model, nn_to_ttnn_dict, cfg)
```

This produces a model where attention layers run on TTNN and MLP layers run on PyTorch. Both types will receive forward-hook timing instrumentation when `set_device` is subsequently called.

---

**Next:** [`device_setup.md`](./device_setup.md)
