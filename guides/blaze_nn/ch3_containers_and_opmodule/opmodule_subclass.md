# OpModule as a base class

The no-subclass form of `OpModule` (previous file) is enough for one-off thin wrappers. The moment you need anything more — a torch-shaped constructor, extra documentation, a custom `forward`, auto-init helpers, or a synthesized fused op — you subclass.

`OpModule` is designed for both forms simultaneously. The two class attributes `op` and `params` (`blaze_nn/modules/base.py:329-330`) are picked up by the constructor at lines 351-354 *if* the matching keyword argument is not supplied, so a subclass that sets them at class scope gets the same wiring as the no-subclass form without restating it in every `__init__`.

## When to subclass

A short triage:

1. **Custom constructor signature.** You want `RMSNorm(normalized_shape, eps=1e-6)` rather than `OpModule(op="rmsnorm", params=("gamma",), epsilon=1e-6)`. The wrapper subclass exists to translate user-facing names into op-facing names.
2. **Custom `forward`.** The default `F.<op>(*args, *params, **kwargs)` shape is wrong — you need to read a buffer address, fan out to multiple ops, or massage kwargs. `blaze_nn.Linear`'s in-tree caller-allocated-output handling sits in the *base* class; a separate qwen3 example, `TokenEmbedding`, is the canonical custom-`forward` case (see Chapter 4 `composing_submodules.md`).
3. **Auto-init.** You want callers to be able to skip `load_state_dict` for quick experiments — override `_torch_init_specs` and the outer `__call__` will lazily fill in random weights on first call.
4. **Class-level documentation.** A torch-style docstring on the class with its constructor signature is plain easier to find than reading the constructor of every `OpModule(op="...")` call.
5. **Fused-op synthesis.** Your op is composed of upstream tt-blaze primitives but is not itself registered. Override `define_fused_op` (next section).

## The canonical small example: `RMSNorm`

The simplest "give the op a friendly name and a torch-shaped constructor" pattern is `blaze_nn/ops/rmsnorm/op.py` — 30 lines, no custom forward, no fused op:

```python
class RMSNorm(OpModule):
    op = "rmsnorm"
    params = ("gamma",)

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__(epsilon=eps, width=normalized_shape)
        self.normalized_shape = normalized_shape
        self.eps = eps

    def _torch_init_specs(self):
        return [("gamma", (1, self.normalized_shape), [1, 32])]
```

Walking the code:

- **`op = "rmsnorm"`, `params = ("gamma",)`** — class-level declarations replace the no-subclass form's keyword arguments.
- **`super().__init__(epsilon=eps, width=normalized_shape)`** — the parent constructor captures the two op-call kwargs in `_op_kwargs`. The default `forward` then calls `F.rmsnorm(x, gamma, epsilon=eps, width=normalized_shape)` on every invocation — no `forward` override needed.
- **`self.normalized_shape = normalized_shape` and `self.eps = eps`** — these go through `Module.__setattr__`. They are neither `Parameter` nor `Module`, so they fall through to `object.__setattr__` (Chapter 2 `module_attribute_protocol.md`) and live as plain Python attributes for the user's own benefit.
- **`_torch_init_specs`** — the opt-in for auto-init (see below).

## The canonical complex example: `Linear`

`blaze_nn/modules/linear.py:8-76` is the other end of the spectrum: torch-shaped constructor *and* a synthesized fused op *and* `_torch_init_specs`. The shape is worth studying because every fused-op subclass follows it:

```python
class Linear(OpModule):
    op = "blaze_nn_linear"
    params = ("weight",)

    @classmethod
    def define_fused_op(cls) -> None:
        import blaze
        from blaze.blaze_op import BlazeOp, FusedOp, Input, Output
        # ... see source for full body
        if "blaze_nn_linear" in BlazeOp._class_registry:
            return
        class BlazeNNLinear(FusedOp):
            name: str = "blaze_nn_linear"
            user_allocated_outputs: tuple[str, ...] = ("output",)
            # ... ports, compose() — see linear.py:34-55
        BlazeNNLinear.register()
        ...
```

The constructor adds two domain attributes and refuses bias:

```python
def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
    if bias:
        raise NotImplementedError(
            "Linear bias is not yet supported. Compose with F.residual_add."
        )
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

def _torch_init_specs(self):
    return [("weight", (self.in_features, self.out_features), [32, 32])]
```

Three things to note:

1. **`define_fused_op` is a classmethod called once.** The parent constructor (`blaze_nn/modules/base.py:345-349`) checks the `_fused_op_defined` per-class flag and runs the hook only on the first instantiation, guarded by both that flag and the `if name in BlazeOp._class_registry` short-circuit inside the body (`linear.py:23-59`). Repeated `Linear(...)` calls pay nothing; module re-import in a fresh process re-registers.
2. **`user_allocated_outputs = ("output",)`** on the synthesized `FusedOp` is what makes the user-facing `set_output_tensor(...)` mandatory. The constructor of `OpModule` reads this list via `_lookup_user_allocated_outputs` (`base.py:360`) and stores it as `_required_output_names`. The next file (`output_tensors.md`) covers what the user does with it; Chapter 6 `caller_allocated_outputs_internals.md` covers the full chain.
3. **`Linear` does not override `forward`.** The default `OpModule.forward(x)` → `F.blaze_nn_linear(x, self.weight, **op_kwargs)` is exactly right. The complexity is all in `define_fused_op`, not in the call shape.

> **For contributors:** The full `define_fused_op` recipe — when to use it, how the registration is idempotent across imports, and the relationship to `_lookup_user_allocated_outputs` — is in Chapter 7 `add_a_fused_op.md`.

## Auto-init: `_torch_init_specs` and `init_torch_params`

Both `RMSNorm` and `Linear` opt in to a small convenience by overriding `_torch_init_specs`. The signature is:

```python
def _torch_init_specs(self) -> list[tuple[str, tuple[int, ...], list[int]]]:
    return [(param_name, torch_shape, default_tile_dims), ...]
```

(default implementation at `blaze_nn/modules/base.py:452-458` returns `[]` — no auto-init).

There are two ways the spec is consumed:

1. **Explicit call**: `m.to(device); m.init_torch_params(seed=0)` builds each declared parameter from `torch.randn(shape, dtype=torch.bfloat16)`, packs it as a `ttnn.Tensor` with `TILE_LAYOUT`, the per-spec tile, and the recorded device, then calls `m.load_state_dict({...})` to install them (`blaze_nn/modules/base.py:460-501`).
2. **Implicit, from `__call__`**: if `_state_loaded` is `False` and `_torch_init_specs()` is non-empty, the outermost `__call__` runs `init_torch_params()` before forward (`blaze_nn/modules/base.py:425-428`). This is the "skip `load_state_dict` and just call the module for a quick experiment" path.

A few sharp edges:

- **`init_torch_params` requires a device.** It checks `self._device_config is None` and raises `RuntimeError("Call .to(device) before init_torch_params().")` (`base.py:483-484`).
- **`init_torch_params` imports torch *lazily*.** The import lives inside the function body (`base.py:480-481`), not at module scope, so the framework stays torch-free until the helper is actually used. Same defaults as everywhere else in the framework: `ttnn.bfloat16`, `TILE_LAYOUT` (`base.py:489-497`).
- **`init_torch_params` is a `load_state_dict` underneath.** After building the tensors it calls `self.load_state_dict(state)` (`base.py:501`), so `_state_loaded` flips to `True` and the implicit-init branch will not run again.
- **The tile dims must agree with the layout the op expects.** `RMSNorm`'s gamma is a 1-D scale broadcast along rows so the tile is `[1, 32]`; `Linear`'s weight is a 2-D matmul matrix so the tile is `[32, 32]`. Getting the tile wrong here surfaces as a layout error at first compile, not at init.

> **Warning:** Inside an active tracing context (an outer `forward` is already running), `OpModule.__call__` does *not* auto-init — only the outermost call boundary checks (`base.py:416`). This matters when a parent module's `forward` exercises a child for the first time: the child's params must already be loaded, because the parent's tracing context is open and the auto-init branch is skipped.

## When to subclass vs. instantiate

| Need | Form |
|---|---|
| One-off op with no kwargs, no custom forward | `OpModule(op=..., params=...)` inline |
| Reusable, importable, torch-shaped constructor | Subclass with class attrs |
| Custom `forward` (extra kwargs, buffer-address extraction) | Subclass and override `forward` |
| Op not yet in `BlazeOp._class_registry` | Subclass and override `define_fused_op` |
| Want `init_torch_params` to work | Subclass and override `_torch_init_specs` |

The Qwen3 walkthrough in Chapter 4 uses both forms intentionally: `OpModule(op="residual_add")` inline for parameterless one-offs, subclasses (`TokenEmbedding`, `RoPE`, `FusedQKV`) for anything that needs a custom forward or class-level docs. The buffer-address-extraction pattern in `TokenEmbedding` is covered in Chapter 4 `composing_submodules.md`.

## What the subclass form does *not* do

- **It does not change the dispatch path.** The default `forward` is still `F.<op>(*args, *params, **op_kwargs)`. If you need to read a buffer address, fan out, or unwrap, you write your own `forward`.
- **It does not register a name.** Subclassing `OpModule` and setting `op = "foo"` is a no-op unless `F.foo` resolves — either because the upstream tt-blaze registers it, or because your `define_fused_op` does. The class attribute is a lookup key, nothing more.
- **It does not move parameters.** `m.to(device)` records a `DeviceConfig` on the module, exactly as Chapter 2 `device_binding.md` describes — it never touches the `ttnn.Tensor` data. The user produces tensors with the desired memory config and shard spec *before* `load_state_dict` (or `init_torch_params`).

_Previous: [OpModule without subclassing](opmodule_no_subclass.md) · Next: [User-allocated output tensors](output_tensors.md) · [Up](index.md)_
