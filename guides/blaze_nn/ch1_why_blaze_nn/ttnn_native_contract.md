# The ttnn-native contract

Every later chapter leans on one invariant: **tensors that cross a `Module` boundary are `ttnn.Tensor`, and blaze-nn treats them as opaque**. This page states the invariant, points at the four places it shows up in code, and forward-links the two follow-up topics (torch interop, universal op dispatch).

## blaze-nn never imports torch at module scope

Restating the package docstring quoted in [What blaze-nn is](what_it_is.md#the-framework-in-its-own-words): the framework is ttnn-native, no torch tensors flow through framework code, and any torch interop is done by the user via `blaze_nn.interop` (`blaze_nn/__init__.py:5-7`).

A few mechanical consequences fall out of this:

- `import blaze_nn` does **not** import torch, ttnn, or tt-blaze. The optional `[dev]` extra adds torch only for the golden-reference test files.
- The framework's own dependencies are empty — see `pyproject.toml:11` (`dependencies = []`). The downstream tt-blaze / ttnn imports are lazy and happen only on the path where a `Module` is actually executed (e.g. `BlazeCompiler` is imported inside `_call_graph` at `blaze_nn/modules/base.py:98`).
- This is why `import blaze_nn` works on a machine without tt-blaze installed — useful for unit-testing the framework itself. See [Getting started](getting_started.md) for the three test tiers that depend on this.

## Where the contract lives in code

There are four positions that a `ttnn.Tensor` can occupy with respect to a `Module`, and the framework treats all four the same way — by reference, never by inspection.

1. **Parameters.** `Parameter._tensor: Any` holds whatever you put there — no isinstance check, no shape declaration, no dtype constraint (`blaze_nn/parameter.py:16-26`). Tests exploit this: `tests/test_parameter.py` and most of `tests/test_module.py` use `object()` sentinels rather than real `ttnn.Tensor`s to verify the framework's plumbing without needing a device.

2. **`forward` arguments.** What the user passes into `model(x, ...)` is a `ttnn.Tensor`; the framework wraps each one as a graph-input proxy on the way in (`blaze_nn/modules/base.py:92-93`) without inspecting its contents.

3. **`forward` return value.** Whatever falls out of `program.run()` (`blaze_nn/modules/base.py:122`) is a `ttnn.Tensor` (or a tuple of them); blaze-nn just hands it back.

4. **`state_dict` values.** `module.state_dict()` returns `OrderedDict[str, ttnn.Tensor | None]` keyed by dotted parameter paths; `load_state_dict` writes the values straight onto `param._tensor` with no coercion. This is covered in Chapter 2's `traversal_and_state_dict.md`.

In none of these positions does blaze-nn call `.shape`, `.dtype`, `.to(...)`, or `tensor.numpy()` on what you handed it. The only `.shape` access in the framework's non-test code is the `Parameter.__repr__` heuristic at `blaze_nn/parameter.py:28-34`, which is used purely for debug printing and is guarded against `None`.

## Why this matters: universal op dispatch becomes free

Because blaze-nn treats tensors as opaque tokens routed through tt-blaze, the set of operations available in `forward()` is not a hand-curated list inside this framework — it is **whatever tt-blaze's op registry has**, exposed lazily.

The `blaze_nn.functional` (`F`) module declares only two explicit shims (`F.linear`, `F.sliced_matmul`) at `blaze_nn/functional.py:46-60`. Every other op is resolved on first attribute access by the module-level `__getattr__` at `blaze_nn/functional.py:63-88`, which builds a closure that routes `F.<op>(*args, **kwargs)` through `_dispatch` (`blaze_nn/functional.py:24-43`) into the active tracing context. The closure is cached into `globals()` so the second `F.<op>` access is a normal attribute lookup.

The README states this directly (see `README.md:10`):

> **Universal op dispatch** — `F.<any_tt_blaze_op>(...)` resolves at attribute-access time against tt-blaze's op registry; no per-op wiring needed in blaze-nn.

In practice this means a new op added to tt-blaze becomes callable from `forward()` as `F.<that_op>(...)` immediately, with no edit to this framework — provided you don't need a friendlier name or special argument handling.

> **For contributors:** the lazy `__getattr__` mechanism, the explicit shim rule, and the alias registry (`linear → matmul`, `sliced_matmul → kn_sliced_matmul`) are walked end-to-end in Chapter 6 `functional_dispatch.md` and `registry.md`. Adding a new op normally means *no* edit to `functional.py`.

## The torch interop boundary

Model authors do need to materialize `ttnn.Tensor`s from somewhere — typically a Hugging Face checkpoint. The sanctioned bridge is `blaze_nn.interop`, an optional submodule that imports `ttnn` lazily inside each helper and is **not used by the core framework**.

Full coverage — public signatures, defaults (`bfloat16` + `TILE_LAYOUT`), and the "never call interop from inside `blaze_nn/`" anti-pattern — is in Chapter 2 `interop_at_the_boundary.md`.

_Previous: [What blaze-nn is](what_it_is.md) · Next: [Getting started](getting_started.md) · [Up](index.md)_
