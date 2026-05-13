# The op registry — aliases and placement hints

`blaze_nn/_registry.py` is 66 lines and holds a single dict. It is the smallest file in the framework that carries real semantic weight: it tells [`_dispatch`](functional_dispatch.md) which name maps to which tt-blaze op, and it tells `GraphTracingContext` which ops should run on the matmul subgrid and which should receive an auto-injected `sender` kwarg. Most ops need **no entry at all** — universal dispatch handles them.

This page walks the file, explains the three-flag semantics, lists the current entries, and walks a decision tree for adding a new op.

## `OpInfo` — the only data class

```python
@dataclass(frozen=True)
class OpInfo:
    """Per-op dispatch metadata."""
    backend: str | None = None
    uses_matmul_cores: bool = False
    needs_sender_core: bool = False
```

Three fields, all optional. `OpInfo()` with everything defaulted is `_DEFAULT_INFO` (the value returned for any op not in `_REGISTRY`). The dataclass is `frozen=True` — entries are intended to be immutable singletons.

The three fields have distinct lifecycles, which is the most important thing to understand about this file:

| Field | Set on which entries | Read when |
| --- | --- | --- |
| `backend` | **Alias entries** — the blaze-nn-facing name | Inside `_dispatch`, after the active-context check (`functional.py:34`) |
| `uses_matmul_cores` | **Backend entries** — the tt-blaze op name | Inside `TracingContext._resolve_grid` after alias resolution (`_tracing.py:88`) |
| `needs_sender_core` | **Backend entries** — the tt-blaze op name | Inside `GraphTracingContext.dispatch`, after alias resolution (`_tracing.py:142`) |

The split matters: placement flags must live on the **backend** entry, not the alias. If a future contributor moved `uses_matmul_cores=True` onto the `"linear"` alias entry, the flag would never fire — by the time `_resolve_grid` runs, the name has already been resolved to `"matmul"`. The current registry is wired correctly; preserve this invariant.

## The current entries

```python
_REGISTRY: dict[str, OpInfo] = {
    # Aliases: blaze_nn-facing name → tt-blaze backend op.
    "linear": OpInfo(backend="matmul"),
    "sliced_matmul": OpInfo(backend="kn_sliced_matmul"),
    # Backend ops with dispatch hints.
    "matmul": OpInfo(uses_matmul_cores=True),
    "kn_sliced_matmul": OpInfo(uses_matmul_cores=True),
    "residual_add": OpInfo(uses_matmul_cores=True),
    "mcast": OpInfo(needs_sender_core=True),
}
```

Six entries total: **two aliases** and **four placement hints**.

**Aliases.**

- `"linear" → "matmul"` — the friendlier, torch.nn.functional-compatible name maps to the tt-blaze matmul op.
- `"sliced_matmul" → "kn_sliced_matmul"` — short, idiomatic name for the KN-sliced variant; the underlying tt-blaze op carries its more descriptive internal name.

**Placement flags on backend ops.**

- `matmul`, `kn_sliced_matmul`, `residual_add` all set `uses_matmul_cores=True` — they should run on the device's matmul subgrid rather than the full all-cores grid.
- `mcast` sets `needs_sender_core=True` — the multicast op needs the device's sender core passed as the `sender` kwarg.

Note that `linear` and `sliced_matmul` themselves carry no placement flags. That is correct: `_resolve_grid` and the sender-injection branch in `GraphTracingContext.dispatch` see the resolved backend name (`"matmul"`, `"kn_sliced_matmul"`), and those backend entries carry the flags.

> **Note:** No entry sets both `uses_matmul_cores` and `needs_sender_core`. The data model permits it, but the current set of in-tree ops doesn't need it. If a future op does, the dispatcher handles each flag independently — there is no implicit precedence.

## The three public helpers

```python
def resolve_alias(name: str) -> str:
    """Return the tt-blaze op name for a blaze_nn functional name."""
    backend = _info(name).backend
    return backend if backend is not None else name

def uses_matmul_cores(backend_op: str) -> bool:
    return _info(backend_op).uses_matmul_cores

def needs_sender_core(backend_op: str) -> bool:
    return _info(backend_op).needs_sender_core
```

Three lookups, all `O(1)`. Behavior of each is exhaustively pinned by tests:

- `resolve_alias` — `tests/test_functional.py:TestAliasResolution::test_linear_resolves_to_matmul` asserts `resolve_alias("linear") == "matmul"`; `test_sliced_matmul_resolves_to_kn_sliced_matmul` asserts the second alias; `test_unknown_name_passes_through` asserts `resolve_alias("rmsnorm") == "rmsnorm"` and `resolve_alias("brand_new_op") == "brand_new_op"`. The pass-through behavior is what makes universal dispatch work — unmapped names route to themselves.
- `uses_matmul_cores` / `needs_sender_core` — invoked only from `_tracing.py`. No standalone test asserts return values directly; their effect is observed end-to-end in dispatch-integration tests (e.g. `tests/test_dispatch_integration.py:test_linear_alias_creates_matmul_node` exercises the full alias-plus-grid path).

The helpers are the **only** public surface of `_registry.py`. Internal callers should not import `_REGISTRY` or `_info`; they should go through these three names.

## How `_resolve_grid` consumes the flags

The matmul-cores hint is read in `TracingContext._resolve_grid` (`blaze_nn/_tracing.py:82`):

```python
def _resolve_grid(self, backend_op: str, explicit_grid: Any) -> Any:
    if explicit_grid is not None:
        return explicit_grid
    if self.device_config is None:
        return None
    if uses_matmul_cores(backend_op):
        return self.device_config.matmul_cores
    return self.device_config.all_cores
```

Three precedence rules in order:

1. An explicit `_grid=` kwarg passed by the user wins outright. Users should rarely set this — it exists for advanced placement debugging.
2. If there is no `DeviceConfig` (the framework-only test path with `device_config=None`), return `None` and let the backend pick.
3. Otherwise: matmul-cores ops get `device_config.matmul_cores`; everything else gets `device_config.all_cores`.

The sender-core hint is read in `GraphTracingContext.dispatch` (`blaze_nn/_tracing.py:138-144`):

```python
blaze_kwargs = dict(kwargs)
if (
    "sender" not in blaze_kwargs
    and self.device_config is not None
    and needs_sender_core(op_name)
):
    blaze_kwargs["sender"] = self.device_config.sender_core
```

Precedence again: an explicit `sender=` from the caller wins. With no device config, the branch is skipped. Otherwise: any op whose `OpInfo.needs_sender_core` is `True` gets `device_config.sender_core` auto-injected.

> **Note:** `_resolve_grid` runs on the **resolved backend name** because that is what `GraphTracingContext.dispatch` passes (`op_name` in the snippet above is the post-alias name — `_dispatch` resolved it before calling `ctx.dispatch`). This is the load-bearing reason placement flags must live on backend entries.

## Decision tree for adding a new op

When you add a new tt-blaze op and want it reachable from `blaze_nn`, walk this tree:

**1. Does the op need a friendlier or different blaze-nn-facing name than tt-blaze uses?**
   - **Yes:** Add an alias entry: `"my_friendly_name": OpInfo(backend="tt_blaze_internal_name")`. Add a `tests/test_functional.py:TestAliasResolution` case asserting the alias resolves correctly.
   - **No:** Skip this step.

**2. Should the op run on the matmul subgrid rather than the full device grid?**
   - **Yes:** Add (or extend) the backend entry: `"tt_blaze_internal_name": OpInfo(uses_matmul_cores=True)`. Verify with a dispatch-integration test that asserts the node's `kwargs.get("grid")` matches `device_config.matmul_cores`.
   - **No:** Skip this step. Most ops use the all-cores grid.

**3. Does the op need the device's sender core auto-injected as the `sender` kwarg?**
   - **Yes:** Add (or extend) the backend entry: `"tt_blaze_internal_name": OpInfo(needs_sender_core=True)`. Currently only `mcast` does this; any new multicast-shaped op is the candidate.
   - **No:** Skip this step.

**4. None of the above?** No registry entry is required at all. `blaze_nn.functional.__getattr__` will resolve `F.<your_op>(...)` to the tt-blaze op via universal dispatch the first time it is called. This is by far the most common case — `rmsnorm`, `rope`, `embedding`, `copy`, `untilize`, `swiglu`, `moe`, and most other ops live entirely in tt-blaze and need zero blaze-nn wiring.

> **Warning:** Do not add a placement flag to an alias entry. The flag will silently never fire because alias resolution happens before the placement-flag lookup. If you need both a friendlier name **and** a placement hint, add two entries: one alias entry with `backend="..."`, and a separate backend entry with the placement flags. The `linear/matmul` pair is the canonical example.

> **For contributors:** The placement-flag pattern is one of two hooks for changing op-level dispatch without forking the backend. The other is the `_ua_*` attribute pattern on `OpModule` (Ch5 `module_call_path.md`), which routes per-instance compile args (`_collect_user_args`) to `BlazeCompiler.compile(..., user_args=...)`. Use the registry when the rule is "every call to this op"; use `_ua_*` when the rule is "this specific module instance".

## What this file is **not**

- It is **not** a list of every op available in blaze-nn. Universal dispatch makes any tt-blaze op addressable as `F.<name>(...)` without an entry here.
- It is **not** a place to store op signatures, kwarg defaults, or documentation. Argument shims live in `blaze_nn/functional.py` (see [`functional_dispatch.md`](functional_dispatch.md)).
- It is **not** a place to register fused ops synthesized inside blaze-nn. Fused-op registration uses `OpModule.define_fused_op` and the tt-blaze `BlazeOp._class_registry` (see [`caller_allocated_outputs_internals.md`](caller_allocated_outputs_internals.md)).

Keeping `_registry.py` minimal is a deliberate design goal. Each entry must justify its existence against one of the three flag semantics; anything else belongs elsewhere.

---

_Previous: [Functional dispatch — `_dispatch` and the lazy `__getattr__`](functional_dispatch.md) · Next: [Caller-allocated outputs — internals](caller_allocated_outputs_internals.md) · [Up](index.md)_
