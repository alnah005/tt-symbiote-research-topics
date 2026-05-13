# Contributing checklist — concrete recipes and anti-patterns

This is the contributor's last reference page. The earlier files in Chapter 7 walked the *how* of three extension paths (op wrapper, fused op, custom container/module). This file is the *what to actually edit* for the most common contributions, plus the hard rules the framework relies on you not violating.

## Recipe 1 — Add a new alias

You want `F.attention` to dispatch through tt-blaze's `sdpa` op without renaming anything upstream.

```text
1. Edit blaze_nn/_registry.py:
     "attention": OpInfo(backend="sdpa"),
2. Add a case to tests/test_dispatch_integration.py:
     - open a GraphTracingContext
     - call F.attention(x, k, v)
     - assert ctx.graph.nodes[0].spec.name == "sdpa"
3. Optionally document the alias in Ch6 registry.md.
```

That is the full diff. The universal `__getattr__` in `blaze_nn/functional.py` already routes any name through `resolve_alias` (see `blaze_nn/functional.py:24`), so no `functional.py` edit is needed unless the alias also requires non-trivial argument handling (in which case write a shim like `linear` or `sliced_matmul`).

Do **not** set `uses_matmul_cores` or `needs_sender_core` on the alias entry — those flags belong on the backend entry and are read **after** alias resolution (verify against `blaze_nn/_registry.py:22-32`).

## Recipe 2 — Add a fused op

You want a `BlazeNNMyFused` op composed of upstream tt-blaze primitives, with a caller-allocated output.

```text
1. Create blaze_nn/modules/my_fused.py with a class MyFused(OpModule):
     - op = "blaze_nn_my_fused"
     - params = (...)
     - @classmethod def define_fused_op(cls): ...
       (see Ch7 add_a_fused_op.md for the full body — register() + setattr)
2. Add a tests/test_dispatch_integration.py case:
     - instantiate MyFused(...)
     - open GraphTracingContext
     - run forward, assert node name appears
3. If you have a torch reference, add a parity test in tests/test_pytorch_parity.py.
4. If the op needs hardware-specific kwargs, document the _ua_* expectations.
```

Follow the idempotence rule (`_fused_op_defined` class flag + `if name in BlazeOp._class_registry: return` inside the method + the `hasattr(blaze, ...)` guard on the `setattr`). See [Adding a fused op](add_a_fused_op.md) for the canonical walkthrough.

## Recipe 3 — Add a placement hint

You added a new tt-blaze op that needs the matmul subgrid or the device's sender core.

```text
1. Edit blaze_nn/_registry.py — add an OpInfo entry on the backend op name:
     "my_op": OpInfo(uses_matmul_cores=True),     # for matmul-like ops
     # or
     "my_op": OpInfo(needs_sender_core=True),     # for mcast-like ops
2. Add a graph-construction test:
     - open GraphTracingContext(device_config=some_mock_config)
     - run F.my_op(...)
     - assert the resulting node was called with grid=device_config.matmul_cores
       (or that the sender= kwarg was injected)
3. The flags are read by _resolve_grid (blaze_nn/_tracing.py) AFTER alias
   resolution — set them on the backend name, not the alias.
```

Decision tree: if neither flag applies, do not add a registry entry at all — universal dispatch will handle the op.

## Recipe 4 — Add a new container

You need a container shape that is not `Sequential` / `ModuleList` / `ModuleDict`. See [Extending containers and modules](extending_containers_and_modules.md) for the mixin composition pattern. The contributing-side checklist:

```text
1. Inherit from _IndexedContainer / _NotCallableContainer (or both).
2. super().__init__() in __init__ (populates _modules, _parameters, etc.).
3. Children register via self._modules[key] = module — no auto-registration
   helpers exist beyond _register_indexed.
4. Add a TestNewContainer class in tests/test_containers.py — exercise
   iteration, indexing, len, state_dict roundtrip via the existing patterns.
```

## Recipe 5 — Add an `_ua_*` knob

You want a compile-time kwarg that flows from a user-facing `Module` attribute to `BlazeCompiler.compile(..., user_args=...)`.

```text
1. On the Module that is the *graph boundary* (the one a user calls
   directly, not its children), set:
     self._ua_my_knob = value
2. If your Module is a plain `Module`, copy the qkv_proj.py override:
     def _collect_user_args(self):
         return {k[4:]: getattr(self, k) for k in dir(self) if k.startswith("_ua_")}
   OpModule already does this — no override needed if you inherit from OpModule.
3. Read user_args["my_knob"] inside your FusedOp.compose classmethod (if
   you have one) or document where the consumer of the kwarg lives.
```

Pitfall: `_collect_user_args` is read once per compile. Mutating `_ua_*` after the first forward call does not retrigger anything.

## Anti-patterns — hard rules

These rules are framework invariants. Violating them does not always fail loudly, which is why they are listed here. Every one of them has been responsible for a real bug.

### 1. Never `import torch` at module scope inside `blaze_nn/`

`blaze_nn/__init__.py` documents: "The framework is ttnn-native; never imports torch at module scope." Two carve-outs exist:

- **`blaze_nn/interop/`** — this is the torch ↔ ttnn boundary by design. `interop/__init__.py` does its imports inside each function, but module-scope torch imports here are acceptable because nothing in the framework imports `blaze_nn.interop`.
- **`init_torch_params` in `blaze_nn/modules/base.py`** — imports `torch` and `ttnn` lazily *inside the method body* (`base.py:480-481`). The framework remains torch-free until `init_torch_params` is actually called.

Everything else must defer torch imports to inside functions, or not import torch at all. `import blaze_nn` must succeed in a torch-free environment.

> **Warning:** If you add a new top-level `import torch` anywhere under `blaze_nn/` outside those two locations, the test `import blaze_nn` (run by every CI smoke test, and by `test_functional.py`'s no-context tier) will start dragging in torch. This breaks the contract with downstream users who do not have torch installed.

### 2. Never call `blaze_nn.interop` from inside `blaze_nn/`

The `interop` package is for *users* converting torch state-dicts to ttnn tensors before `load_state_dict`. It is not a framework utility. Calling `blaze_nn.interop.to_device_tensor` from inside `blaze_nn/modules/`, `blaze_nn/_tracing.py`, `blaze_nn/functional.py`, etc. creates a dependency on torch that the rest of the framework explicitly avoids, and breaks the `forward()`-is-traced invariant: `to_torch(t)` returns a CPU torch tensor, which cannot be a `TensorProxy` in a tracing context — any subsequent `F.<op>(torch_tensor, ...)` will fail downstream when the backend op receives a torch tensor it cannot consume. `TracingContext._unwrap_args` (`blaze_nn/_tracing.py:70-80`) does not raise here: it passes non-`TensorProxy`/non-`Parameter` args through unchanged via its `else: out.append(a)` branch, so the failure surfaces one frame later, inside `op_handle(*unwrapped_args, ...)` at `_tracing.py:149` (or further downstream in the compiler).

If you find yourself wanting to call interop from framework code, you are doing the wrong thing — the user should have built `ttnn.Tensor`s before handing them to you.

### 3. Never bypass `F` to call `blaze.<op>` directly inside `forward`

`F` is not a convenience layer — it is the **dispatch boundary**. Every `F.<op>(...)` call goes through:

1. `_dispatch(op_name, *args)` — the entry in `blaze_nn/functional.py:24`.
2. `_get_active_context()` — the active-context lookup.
3. `resolve_alias(op_name)` — Ch6 `registry.md`.
4. `ctx.wrap_parameter(p)` for each `Parameter` arg.
5. `ctx.dispatch(op_name, *args, **kwargs)` — which finally consults `getattr(blaze, op_name)` (or `BlazeOp._class_registry[op_name]` in compose mode).

A `forward` method that calls `blaze.matmul(...)` directly skips every one of those steps: no alias resolution, no `_resolve_grid`, no `sender` injection, no `TensorProxy` wrapping, no proxy unwrap on the args. The node does not appear in the `BlazeGraph` because it was emitted outside the tracing context. The forward will appear to work in a unit test and then fail to compile in integration.

The only place `import blaze` is acceptable inside `blaze_nn/` is the body of `define_fused_op` (where the *class registration* is happening, not a *call*), and the module `_tracing.py` (where `getattr(blaze, op_name)` resolves an op handle once per dispatch).

### 4. Never reuse a Buffer's `ttnn.Tensor` across `to(device)` re-binds

This applies to model authors using the qwen3 Buffer pattern, but contributors who write similar setup hooks need to internalize it: a Buffer's `ttnn.Tensor` object has its `buffer_address()` baked into the compiled program at first compile. If you re-allocate the tensor (new `ttnn.from_torch` call, new buffer) without recompiling, the compiled program reads from the old address.

The rule: allocate Buffer tensors once, mutate in place via `ttnn.copy_host_to_device_tensor`, never reassign. If you must reallocate, clear `module._compiled_cache` (or just throw away the module instance and rebuild from scratch).

### 5. Never monkey-patch `user_allocated_outputs` non-idempotently

`OpModule.__init__` reads `user_allocated_outputs` once per instance, via `_lookup_user_allocated_outputs`. If you monkey-patch a tt-blaze op's `user_allocated_outputs` (the way qwen3 does for `SDPADecode`), the patch must:

- Run *before* any `OpModule(op="<that_op>")` is constructed.
- Be idempotent — repeated imports must not double-register or change the tuple's contents.

The qwen3 pattern (see `examples/qwen3_embedding_0_6b/modules/attention.py:_register_sdpa_decode_user_alloc`) is a guard-and-set: check whether the attribute is already the expected tuple, set it if not. Mirror that shape.

### 6. Never change `_required_output_names` after construction

`OpModule._required_output_names` is a snapshot of the tuple taken at `__init__` time. Mutating it later, or mutating `_output_tensors` keys, is undefined behavior. The supported API surface is `set_output_tensor(t)` and `set_output_tensors(name=t, ...)`, both of which only write *values*, never keys.

## Known gap — compose mode

**There is currently no end-to-end test that exercises compose mode.** The `_call_compose` path in `blaze_nn/modules/base.py:126-144` is unverified outside hand-testing. The `@blaze_nn.compose` decorator and the `ComposeTracingContext` walkthrough in [Chapter 5 — Tracing contexts](../ch5_tracing_internals/tracing_contexts.md) describe a real code path, but every test in the repo opens a `GraphTracingContext`.

If you take on a compose-mode contribution — adding a new backend, fixing dispatch, exposing a user-facing fused program — please add at least a dispatch-integration test that:

1. Defines a module with `@blaze_nn.compose` on `forward`.
2. Runs the module and asserts that `_call_compose` was taken (not `_call_graph`).
3. Asserts the resulting `FusedProgram` has the expected ops.

This is flagged in [Testing strategy — Known gap](testing_strategy.md#known-gap) and should be closed by the first compose-mode contributor.

## Where to look when something breaks

A short reference table for the most common failure modes:

| Error | Likely cause | Where to look |
|-------|--------------|---------------|
| `RuntimeError: ... no active tracing context` | `F.<op>(...)` called outside `forward()` (or inside an orchestrator's plain-Python `__call__`) | [Ch5 `module_call_path.md`](../ch5_tracing_internals/module_call_path.md), [Ch6 `functional_dispatch.md`](../ch6_dispatch_and_registry/functional_dispatch.md) |
| `ValueError: Unknown blaze op '<name>'` | Missing registry entry or missing `define_fused_op` | [`add_a_fused_op.md`](add_a_fused_op.md), [Ch6 `functional_dispatch.md`](../ch6_dispatch_and_registry/functional_dispatch.md) |
| `RuntimeError: ... has unset required output tensor(s): [...]` | `set_output_tensor` not called before forward | [Ch3 `output_tensors.md`](../ch3_containers_and_opmodule/output_tensors.md), [Ch6 `caller_allocated_outputs_internals.md`](../ch6_dispatch_and_registry/caller_allocated_outputs_internals.md) |
| `RuntimeError: ... has no device. Call module.to(device) first.` | `module.to(device)` skipped | [Ch2 `device_binding.md`](../ch2_module_and_parameter/device_binding.md) |
| `KeyError: 'Unexpected key ...'` | `state_dict` keys drifted from the model | [Ch2 `traversal_and_state_dict.md`](../ch2_module_and_parameter/traversal_and_state_dict.md) |
| `KeyError: 'Unexpected module prefix ...'` | submodule name in `state_dict` does not match the model | [Ch2 `traversal_and_state_dict.md`](../ch2_module_and_parameter/traversal_and_state_dict.md) |
| Silent stale-memory reads / faults after second forward | Buffer rebound after first compile | Anti-pattern 4 above; [Ch4 `tensor_lifetimes.md`](../ch4_qwen3_walkthrough/tensor_lifetimes.md) |
| Op runs but PCC drops far below tier threshold | Wrong placement hint, missing `_ua_*` propagation, or wrong memory_config on parameters | Anti-pattern 3 above; [Ch6 `registry.md`](../ch6_dispatch_and_registry/registry.md); [Ch4 `composing_submodules.md`](../ch4_qwen3_walkthrough/composing_submodules.md) |

Reach for the corresponding chapter section first; the test that backs it is named in [Testing strategy](testing_strategy.md).

## Final pre-flight before submitting a PR

```text
[ ] No `import torch` at module scope in blaze_nn/ (except interop/).
[ ] No `import blaze` or `import ttnn` at module scope in blaze_nn/.
[ ] All new ops use F.<op>(...) inside forward — never blaze.<op> directly.
[ ] Framework-only tests (Tier 1) added and passing.
[ ] Dispatch-integration test (Tier 2b) added if dispatch/registry/tracing
    touched.
[ ] Parity test (Tier 3) added if numerics changed.
[ ] If you added a fused op: define_fused_op is idempotent.
[ ] If you added a Buffer-shaped runtime tensor: documented "mutate in
    place; do not reassign" in the module docstring.
[ ] If you monkey-patched a tt-blaze op: did it at import-time, before
    any OpModule constructs against that op, and idempotently.
```

---

**End of guide.** Return to [Guide Index](../index.md)
