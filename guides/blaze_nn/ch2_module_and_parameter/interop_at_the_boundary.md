# Interop at the boundary — torch ↔ ttnn for `load_state_dict`

The three previous files in this chapter established that blaze-nn is ttnn-native: `Parameter._tensor` is opaque, `load_state_dict` writes verbatim, and `to(device)` does not move tensors. That contract leaves an obvious gap: where does the *first* `ttnn.Tensor` come from? Almost every real model starts from a HuggingFace checkpoint, which is a `dict[str, torch.Tensor]`. The `blaze_nn.interop` package is the sanctioned, narrow bridge that closes that gap.

## Two functions, sixteen lines of code

`blaze_nn/interop/__init__.py` is the entire surface. It exports two functions:

```python
def to_device_tensor(torch_tensor: Any, device: Any, memory_config: Any = None) -> Any:
    """Convert a torch.Tensor to a ttnn tensor on `device`."""
    import ttnn
    tt = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    if memory_config is not None:
        tt = ttnn.to_device(tt, device, memory_config=memory_config)
    else:
        tt = ttnn.to_device(tt, device)
    return tt


def to_torch(device_tensor: Any) -> Any:
    """Convert a ttnn tensor on device back to a torch.Tensor on CPU."""
    import ttnn
    return ttnn.to_torch(ttnn.from_device(device_tensor))
```

(See `blaze_nn/interop/__init__.py:22-47`.) The defaults to remember:

- **`to_device_tensor`** defaults to `dtype=ttnn.bfloat16` and `layout=ttnn.TILE_LAYOUT`. If `memory_config` is `None`, the tensor lands on the device with whatever placement `ttnn.to_device` picks by default; pass an explicit `MemoryConfig` to control sharding or DRAM/L1 placement.
- **`to_torch`** runs `ttnn.from_device` (pull to host) followed by `ttnn.to_torch` (convert to a `torch.Tensor`). Always returns a CPU torch tensor.

There is no third helper. If you need behavior beyond these defaults — a non-bf16 dtype, a row-major layout, a custom mesh mapper — call `ttnn.from_torch` and `ttnn.to_device` directly in your loader. The interop module is intentionally minimal.

## Lazy imports keep `blaze_nn` torch-free

`import ttnn` lives *inside* each function body, not at module scope. The same pattern repeats for `torch` — no `import torch` appears anywhere in `interop/__init__.py`; if `torch_tensor` happens to be a torch tensor, that's the *caller's* import, not the framework's. The module docstring (`blaze_nn/interop/__init__.py:1-15`) states the rule explicitly: "All blaze and ttnn imports are deferred so importing `blaze_nn.interop` does not pull either dependency unless a function is actually called."

The consequence is that `import blaze_nn.interop` works on a machine with neither tt-blaze nor ttnn installed — and so does the entire Tier A test suite ([Ch1 getting started](../ch1_why_blaze_nn/getting_started.md)). The framework only fails when a user calls `to_device_tensor` (or `to_torch`) without ttnn on `PYTHONPATH`, and that failure points cleanly at the user's environment rather than at a missed `import` deep in the framework.

## When to use these helpers

There are exactly three sanctioned use sites, all in the user's code:

1. **Data loading.** Building `ttnn.Tensor` inputs from disk-resident torch tensors. The Qwen3 example's `weight_loader.py` does this for every parameter — torch state dict in, ttnn tensors out.
2. **Building the `ttnn.Tensor` dict for `load_state_dict`.** The output of the user's loader is the dict that goes into `module.load_state_dict({...})`. Per [traversal_and_state_dict.md](traversal_and_state_dict.md), `load_state_dict` writes values verbatim — so the loader is the *only* place where the desired `memory_config` and layout can be applied. Pass it to `to_device_tensor`.
3. **Golden comparisons in parity tests.** Tier C tests pull the `ttnn.Tensor` output of a forward back to a torch tensor via `to_torch` and compare it against a torch reference using `comp_pcc` (the helper in `tests/torch_reference.py`). `tests/test_pytorch_parity.py` is the canonical caller.

A concrete loader sketch:

```python
import torch
import ttnn
import blaze_nn.interop as interop

torch_sd = torch.load("model.bin")  # dict[str, torch.Tensor]

mc_qkv = ttnn.MemoryConfig(...)  # the role-specific shard spec
state = {
    "qkv.weight": interop.to_device_tensor(torch_sd["qkv_weight"], device, mc_qkv),
    "out_proj.weight": interop.to_device_tensor(torch_sd["o_weight"], device, mc_o),
    # ...
}
model.load_state_dict(state)
model.to(device)
```

The `to(device)` call after `load_state_dict` records the device on the module (per [device_binding.md](device_binding.md)) — but the tensors are already on the device because `to_device_tensor` put them there.

## When NOT to use these helpers

The corresponding negative is just as important:

> **Warning:** never call `blaze_nn.interop.to_torch` or `to_device_tensor` from inside a `Module`'s `forward()`. Pulling to torch forces a host round-trip, which the tracing machinery cannot record into a tt-blaze graph — and even if it could, the resulting graph would not run on the device. `forward()` must operate exclusively on `ttnn.Tensor` arguments and return `ttnn.Tensor` results.

The "no torch in `forward`" rule is one of the three load-bearing invariants of the ttnn-native contract (the other two: parameters are opaque, and state-dict values are not converted). Breaking it produces one of two failure modes depending on where the call lands:

- If the call happens during tracing: `wrap_input` will receive a torch tensor instead of a `ttnn.Tensor`, the active context will fail to register it (no `name` for it in `_tensor_bindings`), and compilation raises with a confusing message about an unbound input.
- If the call happens outside tracing — e.g. inside an orchestrator's overridden `__call__` (Chapter 4 `orchestrator_pattern.md`) — the torch tensor will silently fail to flow into the next sub-module call, which expects a `ttnn.Tensor` argument.

Both failures are noisy enough to catch in development, but they are entirely avoidable by keeping all torch ↔ ttnn conversion at the user boundary: the loader at startup, the parity comparison at the test boundary.

## The contributor-side rule

> **For contributors:** the symmetric rule for framework code is **never call `blaze_nn.interop` from inside the `blaze_nn/` package itself**. The framework is ttnn-native; pulling into torch from inside `_tracing.py`, `functional.py`, or `modules/base.py` would silently leak a torch dependency into Tier A tests and break `import blaze_nn` on machines without torch. Chapter 7 `contributing_checklist.md` restates this as a hard anti-pattern, alongside "never `import torch` at module scope inside `blaze_nn/` (except inside `interop/` and the lazy `init_torch_params` helper at `blaze_nn/modules/base.py:480`)."

The one principled exception inside the framework is `OpModule.init_torch_params` (`blaze_nn/modules/base.py:460-501`), which imports `torch` and `ttnn` *inside* the function body to materialize random weights via `torch.randn` followed by `ttnn.from_torch`. That helper is opt-in: it only runs if a subclass declares non-empty `_torch_init_specs`. It is *not* called from `forward`; it is called from `__call__` before any tracing starts, when `_state_loaded` is `False` (see `blaze_nn/modules/base.py:427-428`). Even there, the principle holds — torch enters via a clearly marked, lazily imported helper, never silently.

## Recap of the boundary

The ttnn-native contract from Chapter 1 manifests in this chapter as four concrete rules:

1. `Parameter._tensor` is opaque — the framework never inspects it.
2. `load_state_dict` writes verbatim — no coercion.
3. `to(device)` binds, does not move.
4. The only sanctioned torch ↔ ttnn bridge is `blaze_nn.interop` — and only at the user's loader / test boundary.

These four rules together mean a model author's mental model is small: produce `ttnn.Tensor`s with the right placement, hand them to `load_state_dict`, bind the device, call `forward`. Everything else the framework owns. Chapter 3 takes the next step — containers, `OpModule`, and the pre-built ops that fill the slots this chapter set up.

_Previous: [Device binding](device_binding.md) · Next: [Chapter 3 — Containers, OpModule, and pre-built ops](../ch3_containers_and_opmodule/index.md) · [Up](index.md)_
