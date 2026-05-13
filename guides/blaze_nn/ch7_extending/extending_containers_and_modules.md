# Extending containers and modules — beyond the built-ins

`blaze_nn` ships four containers — `Sequential`, `ModuleList`, `ModuleDict`, and (implicitly) every `Module` that registers children via attribute assignment. It also ships `OpModule` and the two pre-built modules (`Linear`, `RMSNorm`). When a contributor needs something that does not fit any of these shapes — a new container, a custom orchestrator, or a Module that injects compile-time kwargs — this file shows the three reusable patterns and what to copy from existing in-tree examples.

## The two container mixins

`blaze_nn/containers.py` factors traversal and call-protection into two small mixins. Read the file end to end if you have not yet — it is 120 lines and they reuse cleanly:

```python
class _IndexedContainer(Module):
    def _register_indexed(self, modules):
        for idx, module in enumerate(modules):
            self._modules[str(idx)] = module
    def __len__(self): return len(self._modules)
    def __iter__(self): return iter(self._modules.values())
    def __getitem__(self, idx): return list(self._modules.values())[idx]

class _NotCallableContainer(Module):
    _usage_hint = "Iterate over it in your forward() method."
    def _raise_not_callable(self):
        raise RuntimeError(f"{type(self).__name__} is not callable directly. {self._usage_hint}")
    def __call__(self, *a, **k): self._raise_not_callable()
    def forward(self, *a, **k): self._raise_not_callable()
```

- `_IndexedContainer` is the right base when your container is keyed by integer index — children register as `str(idx)` keys, so `state_dict` produces `parent.0.weight`, `parent.1.weight`, ... in the standard PyTorch shape.
- `_NotCallableContainer` is the right base when your container should never be invoked directly — calling either `__call__` or `forward` raises with a usage hint. Override `_usage_hint` to customize the error message.

`Sequential` is `_IndexedContainer` only. `ModuleList` is both `_IndexedContainer` + `_NotCallableContainer`. `ModuleDict` is `_NotCallableContainer` only (with its own dict-shaped `__getitem__` / `__setitem__` / `__contains__` / `keys` / `values` / `items`).

## Recipe — a custom container

Suppose you need a `BranchedSequential` that takes a list of branches and runs all of them on the same input, returning a tuple. Compose the two mixins:

```python
from blaze_nn.containers import _IndexedContainer

class BranchedSequential(_IndexedContainer):
    def __init__(self, *branches):
        super().__init__()
        self._register_indexed(branches)
    def forward(self, x):
        return tuple(branch(x) for branch in self._modules.values())
```

State-dict keys nest as expected (`outer.branched.0.<param>`). Iteration and indexing come for free.

For a non-callable variant (must be iterated by name in `forward`), inherit from both mixins and either set `_usage_hint` or rely on the default:

```python
class GatedModuleDict(_NotCallableContainer):
    _usage_hint = "Use .gates() and .modules_by_gate() to iterate by gate name."
    def __init__(self, mapping):
        super().__init__()
        for name, module in mapping.items():
            self._modules[name] = module
    # ... dict-shaped accessors ...
```

> **Note:** Always call `super().__init__()` in your container — the base `Module.__init__` is what populates `_parameters`, `_modules`, `_device_config`, `_compiled_cache`, and `_state_loaded`. Skipping it breaks every traversal.

## Recipe — when to override `__call__`

The default `Module.__call__` (`blaze_nn/modules/base.py:68-82`) is what opens a `GraphTracingContext` or `ComposeTracingContext`, wraps inputs, runs `forward`, and compiles/runs the program. Two mechanisms can short-circuit it; both were introduced at user level in [Chapter 4 — Orchestrator pattern](../ch4_qwen3_walkthrough/orchestrator_pattern.md):

### Mechanism A — the orchestrator two-liner

When `forward` mixes blaze-nn module calls with host-side hops (`ttnn.kv_cache.update_cache_for_token_`, `nlp_create_qkv_heads_decode`, `ttnn.sharded_to_interleaved`) that cannot live inside a single tt-blaze graph, override `__call__` to bypass tracing entirely:

```python
class MyOrchestrator(Module):
    def __init__(self, ...):
        super().__init__()
        # ... children ...
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, x, ...):
        h = self.first_child(x)
        h = some_host_op(h)
        h = self.second_child(h)
        return h
```

Three qwen3 modules use this verbatim: `Qwen3Attention` (`examples/qwen3_embedding_0_6b/modules/attention.py:90`), `Qwen3DecoderLayer` (`examples/qwen3_embedding_0_6b/modules/decoder_layer.py:32`), and `Qwen3EmbeddingModel` (`examples/qwen3_embedding_0_6b/modules/model.py:67`). Each `self.first_child(x)` call inside `forward` re-enters `Module.__call__` normally and opens its own tracing context — the orchestrator is just the seam between independently-compiled subgraphs.

When NOT to use this pattern: if your whole `forward` can be a single graph (all `F.*` ops, all child `Module` calls, no host hops), the default `__call__` is correct and will give you a single fused program. Orchestrators trade compilation reuse for the ability to interleave host code.

> **Warning:** If you override `__call__` for an orchestrator, do **not** also expect `forward` to participate in tracing — the two patterns are mutually exclusive. Either the module is a graph (default behavior, `forward` is traced) or it is an orchestrator (overridden `__call__`, `forward` runs as plain Python). The framework enforces this loudly: if your orchestrator's plain-Python `forward` calls `F.<op>(...)` without an active tracing context, `_dispatch` (`blaze_nn/functional.py:24-32`) raises `RuntimeError("blaze_nn.F.<op>() must be called inside a Module.forward(). There is no active tracing context.")`. There is no silent-corruption failure mode — the invariant either holds or you see that exact error.

### Mechanism B — the active-context short-circuit (rare; no in-tree example)

The bare two-liner in Mechanism A is what every in-tree orchestrator uses, because every in-tree orchestrator is the top of its own call (verified across `Qwen3Attention`, `Qwen3DecoderLayer`, and `Qwen3EmbeddingModel`). If you have that shape, stop here.

In the hypothetical case where you override `__call__` *and* you also want the same module to be safely callable from inside another module's already-open tracing context, you could preserve the re-entry check the base class does:

```python
def __call__(self, *args, **kwargs):
    from blaze_nn._tracing import _get_active_context
    if _get_active_context() is not None:
        return self.forward(*args, **kwargs)
    # ... your custom path ...
```

The base `Module.__call__` does this at line 71 of `blaze_nn/modules/base.py`. There is no in-tree case that exercises this path — orchestrators are by definition the top of the call — so the simpler rule is: if you don't need to participate in tracing as a child, use the bare two-liner; if you do, don't override `__call__` at all (let the base class trace `forward` normally).

> **Warning:** Do not override `__call__` to add tracing logic — that lives in `_call_graph` / `_call_compose` and the `TracingContext` subclasses. Override only to bypass tracing entirely (orchestrator) or to add pre/post-call bookkeeping outside the trace.

## Recipe — `_collect_user_args` overrides

`_collect_user_args` is the hook that lets a `Module` inject compile-time kwargs into `BlazeCompiler.compile(..., user_args=...)`. The base `Module._collect_user_args` returns `{}`. `OpModule` overrides it to harvest every `_ua_*` attribute:

```python
# blaze_nn/modules/base.py:443-448 (OpModule)
def _collect_user_args(self) -> dict:
    args = {}
    for key in dir(self):
        if key.startswith("_ua_"):
            args[key[4:]] = getattr(self, key)
    return args
```

The qwen3 `FusedQKV` does the same on a plain `Module` because the graph boundary lives on `FusedQKV` (the compiler reads `FusedQKV._collect_user_args`, not `FusedQKV.linear._collect_user_args`):

```python
# examples/qwen3_embedding_0_6b/modules/qkv_proj.py:40-45
def _collect_user_args(self) -> dict:
    args = {}
    for key in dir(self):
        if key.startswith("_ua_"):
            args[key[4:]] = getattr(self, key)
    return args
```

With `self._ua_blackhole_cores = "64x8"` set in `FusedQKV.__init__` (line 29), the compiler receives `user_args = {"blackhole_cores": "64x8"}`. That key is read by the qwen3 monkey-patch `_blaze_nn_linear_patch.py`, which swaps `Linear.compose` to use the named subgrid.

### The four-step chain from attribute to kernel

Tracing `_ua_x = "v"` to the kernel:

1. `_ua_x = "v"` lives as an attribute on the module instance.
2. When `Module.__call__` opens the tracing context, it calls `self._collect_user_args()`. The default returns `{}`; `OpModule` and modules like `FusedQKV` return `{"x": "v"}` (prefix stripped).
3. The dict is threaded into `BlazeCompiler(dc.device).compile(graph, tensors, output_tensor, user_args=...).run()` — see [Chapter 5 — Module call path](../ch5_tracing_internals/module_call_path.md).
4. The compiler hands it to each fused op's `compose` classmethod as the final `user_args` argument. From there, op code reads `user_args["x"]` directly.

That four-step chain is the only path: there is no second mechanism for passing compile-time data into a `compose` recipe, which is why the `_ua_*` convention matters and why it must live on the boundary module.

### When to copy this pattern

- **Your module wraps a fused op that needs hardware-specific knobs** (qwen3 `FusedQKV` → `_ua_blackhole_cores`).
- **Your `Module` *is* the graph boundary** (it is the top-level module a user calls; the compiler reads `_collect_user_args` only on the boundary, not on child sub-modules whose `forward` runs inside an already-active tracing context).
- **You want the knob discoverable via `dir(self)`** rather than buried in a config dict — the `_ua_*` convention is grep-friendly and stays out of `state_dict`.

### When NOT to use it

- **For per-call kwargs.** Put those in `forward(self, x, **kwargs)` and pass through as op kwargs. `_collect_user_args` runs at compile time, before `forward`, and is cached on the compiled program.
- **For values that change between forward calls.** `_collect_user_args` is read once per compile; mutating `_ua_*` after the first call does not retrigger a recompile.
- **On a child module when the parent is the graph boundary.** They will be silently ignored — the compiler only reads `_collect_user_args` on the boundary. Walk up to the module whose `__call__` opens the tracing context and put them there.

## Putting it together — a custom Module checklist

For a new top-level `Module` (not an `OpModule` subclass, not a container):

```text
1. Subclass blaze_nn.Module.
2. super().__init__() in __init__.
3. Declare child Parameters / Modules as attributes (auto-registered).
4. Implement forward(*args, **kwargs) — use F.* for all op calls.
5. (Optional) Override __call__ if you are an orchestrator.
6. (Optional) Override _collect_user_args to inject compile-time kwargs.
7. (Optional) Override load_state_dict for key remapping (FusedQKV pattern).
8. (Optional) Override _get_output_tensor if you need caller-allocated outputs.
```

For a new container:

```text
1. Inherit from _IndexedContainer / _NotCallableContainer (or both).
2. super().__init__() in __init__.
3. Populate self._modules with str-keyed children.
4. Override forward() if callable; otherwise let _NotCallableContainer raise.
```

---

_Previous: [Adding a fused op — when the op does not exist upstream](add_a_fused_op.md) · Next: [Testing strategy — the test taxonomy (reverse index)](testing_strategy.md) · [Up](index.md)_
