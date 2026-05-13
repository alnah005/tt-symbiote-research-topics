# Chapter 5 Pass 1 — Agent B review

Scope: factual correctness, critical coherence, structural gaps. Max 5 items.

## Verdict

Accept with required fixes (2 factual, 1 coherence). Pin checks pass; `_resolve_grid` priority and `_unwrap_args` Parameter-branch story are materially misleading.

## Findings

### 1. `_resolve_grid` priority description omits the `device_config is None` short-circuit (factual / misleading)

**Location:** `tracing_contexts.md`, "_resolve_grid: the grid-priority rule" section.

**Claim in chapter:** The chapter lists three priority steps — (1) explicit `_grid` wins, (2) otherwise consult the registry's `uses_matmul_cores`, (3) default to `all_cores`.

**Source of truth:** `blaze_nn/_tracing.py:82-90` has **four** branches in order:

```python
def _resolve_grid(self, backend_op: str, explicit_grid: Any) -> Any:
    if explicit_grid is not None:
        return explicit_grid
    if self.device_config is None:
        return None                       # ← missing from the chapter's narration
    if uses_matmul_cores(backend_op):
        return self.device_config.matmul_cores
    return self.device_config.all_cores
```

The chapter does quote the source verbatim immediately below its priority list, so the truth is on the page; but the prose-priority list says "Otherwise consult the registry" with no `device_config is None → None` rung. Contributors who read only the prose (or pattern-match the numbered list into a decision tree) will think the registry is always consulted when no explicit grid is given. The bullet at step 4 of `GraphTracingContext.dispatch` ("Inject `sender` — only when `device_config is not None` (so dispatch-integration tests with `device_config=None` don't trip)") acknowledges the `device_config=None` case exists in dispatch but never re-applies that logic to `_resolve_grid`. Add the missing rung to the numbered list — this is a one-line fix.

### 2. `_unwrap_args` Parameter-branch story misattributes when it fires (factual / misleading)

**Location:** `tracing_contexts.md`, "_unwrap_args: how F.* ops see backend handles"; reinforced in `tensor_proxy.md`, "The `_inner` invariant" section.

**Claim in chapter:** "The `Parameter` branch is the safety net that lets `F.matmul(x, self.weight)` work even when the author skipped `_bind_parameters_to_context`."

**Source of truth:** `_bind_parameters_to_context` does not wrap Parameters as proxies — it only writes `param._tensor` into `_tensor_bindings` (`base.py:148-151`). The actual `Parameter → TensorProxy` conversion on the live call path happens in `blaze_nn/functional.py:_dispatch` (lines 36-43), which pre-wraps any `Parameter` argument via `ctx.wrap_parameter` **before** calling `ctx.dispatch`. So by the time `_unwrap_args` runs inside `dispatch`, every arg is already a `TensorProxy` or a non-Parameter scalar — the Parameter branch in `_unwrap_args` (`_tracing.py:76-77`) is dead code on the `F.<op>` path.

The Parameter branch only fires if something calls `ctx.dispatch(op_name, some_raw_param, ...)` directly, bypassing `_dispatch`. That is genuinely a safety net, but the cause is "bypassing `_dispatch`," not "skipping `_bind_parameters_to_context`." The chapter's framing is wrong in both directions: skipping `_bind_parameters_to_context` would surface as a missing graph-input binding inside `BlazeCompiler.compile`, not a Parameter slipping through `_unwrap_args`.

Fix: replace the "even when the author skipped `_bind_parameters_to_context`" clause with "even when `dispatch` is called outside the `functional._dispatch` path (e.g. a contributor invoking `ctx.dispatch` from inside the framework)." Mirror the same change in `tensor_proxy.md`'s "_inner invariant" bullet that walks `_dispatch → ctx.dispatch → _unwrap_args`.

### 3. Mechanism A/B prose is correct but the "submodules bind their own" claim has the wrong mechanism (coherence)

**Location:** `module_call_path.md`, "_call_graph line by line", step 4.

**Claim in chapter:** "Iterates `self._parameters` (not children — submodules bind their own when their `forward` calls them via `F`)."

**Source of truth:** Inner-submodule Parameters are not "bound by submodules" — under Mechanism B the inner submodule's `__call__` short-circuits to `forward` directly (`base.py:71-72`), so neither `_bind_parameters_to_context` nor `_call_graph` ever runs on the inner submodule. Inner-submodule Parameters reach the parent context's `_tensor_bindings` via `functional._dispatch → ctx.wrap_parameter` on each `F.<op>(self.inner.weight, ...)` reference (the `if param._tensor is not None: self._tensor_bindings[name] = param._tensor` line at `_tracing.py:124-125`).

Combined with finding #2, the chapter is conflating two different paths into a single "submodules bind their own" story. Fix: state explicitly that inner-submodule Parameters reach the parent's `_tensor_bindings` via `wrap_parameter` triggered from `_dispatch`, not via any inner-submodule `_bind_parameters_to_context` call. This also matters for the Mechanism-B narrative in step 5 ("Nested `Module.__call__`s short-circuit through Mechanism B and emit into the same graph") — that sentence is correct, but the parameter-binding mechanism that supports it needs to be named.

### 4. `_dispatch` does not pass a `TensorProxy` to `ctx.dispatch` as `_unwrap_args` expects — but the chapter implies symmetry (minor factual)

**Location:** `tensor_proxy.md`, table under "The `_inner` invariant: who reads it, who doesn't", and the bullet immediately below ("`dispatch` calls `self._unwrap_args(args)`, which replaces each `TensorProxy` in the tuple with `a._inner` and each raw `Parameter` with the result of `wrap_parameter(a, a._name)._inner`").

**Source of truth:** `_dispatch` wraps Parameters but **does not unwrap** them — it passes the resulting `TensorProxy` to `ctx.dispatch`. So inside `_unwrap_args`, every Parameter that came through `_dispatch` has already become a `TensorProxy` and is unwrapped via the first `isinstance(a, TensorProxy)` branch. The second branch (`elif isinstance(a, Parameter)`) is the safety net described in finding #2.

The chapter's "each raw `Parameter` with the result of `wrap_parameter(a, a._name)._inner`" sentence reads as if `_unwrap_args` is the normal path for Parameter unwrapping. It is not — the normal path runs in `functional.py:_dispatch` BEFORE `_unwrap_args` ever sees the args. Tighten the bullet to: "TensorProxy → `_inner`; raw Parameter (only when something bypassed `functional._dispatch`) → wrap-then-extract via `wrap_parameter(a, a._name)._inner`."

### 5. No critical structural gap found

Three files (`module_call_path.md`, `tracing_contexts.md`, `tensor_proxy.md`) cover the plan's Ch5 scope: `_call_graph` / `_call_compose` line-by-line, the three context classes, `_resolve_grid` priority, the `TensorProxy` opacity rationale, and the compose-mode coverage gap. Cross-references to Ch4 Mechanism A/B and forward-references to Ch6 `registry.md` and `caller_allocated_outputs_internals.md` are all named. The "Known gap: no compose-mode test" claim is verified — `grep -i compose /home/ttuser/salnahari/blaze-nn/tests/` returns zero matches. No missing files; no missing diagrams (each file has at least one Mermaid). No critical coherence break between chapters or within the chapter beyond findings 1–4.

## Pins spot-checked (all correct)

- `base.py:68-82` — `__call__` body ✓
- `base.py:86-122` — `_call_graph` body ✓
- `base.py:106-112` — port-alias dual-keys loop ✓
- `base.py:126-144` — `_call_compose` body ✓
- `base.py:153-154` — `Module._collect_user_args` returns `{}` ✓
- `base.py:156-159` — `Module._get_output_tensor` aliases `inputs[0]` ✓
- `base.py:406-411` — `OpModule._get_output_tensor` override ✓
- `base.py:417-423` — unset-output-tensor pre-check ✓
- `base.py:443-448` — `OpModule._collect_user_args` (`dir(self)` + `_ua_` strip) ✓
- `_tracing.py:20-34` — module-level active-context helpers ✓
- `_tracing.py:37-90` — `TracingContext` base ✓
- `_tracing.py:70-80` — `_unwrap_args` ✓
- `_tracing.py:82-90` — `_resolve_grid` ✓ (but see finding #1)
- `_tracing.py:93-150` — `GraphTracingContext` ✓
- `_tracing.py:128-150` — `GraphTracingContext.dispatch` ✓
- `_tracing.py:153-196` — `ComposeTracingContext` ✓
- `_tensor_proxy.py:14-28` — class body ✓ (28 lines total)
- `_registry.py:40-42` — three `uses_matmul_cores=True` entries (`matmul`, `kn_sliced_matmul`, `residual_add`) ✓
- `_registry.py:43` — `mcast` is the only `needs_sender_core=True` entry ✓
- Orchestrator `__call__` overrides at `attention.py:90`, `decoder_layer.py:32`, `model.py:67` ✓
- `_compiled_cache` is referenced only at `base.py:30` ✓
- `grep -rn -i compose /home/ttuser/salnahari/blaze-nn/tests/` → zero matches ✓

---

## Agent A change log — applied after Pass 1 B review

- Issue 1 (`_resolve_grid` priority missing `device_config is None` rung): inserted a new step 2 in the priority list in `tracing_contexts.md` naming the `device_config is None → None` short-circuit and renumbered the registry / default rungs to 3 and 4. Verified against `_tracing.py:82-90`.
- Issue 2 (`_unwrap_args` Parameter-branch misattribution): rewrote the prose immediately under the `_unwrap_args` source block in `tracing_contexts.md` so the safety-net cause is "`dispatch` called outside `functional._dispatch`" rather than "the author skipped `_bind_parameters_to_context`"; named `functional._dispatch`'s pre-wrap step at `functional.py:36-43` as the reason the `Parameter` branch is unreachable on the live `F.<op>` path. Verified against `functional.py:24-43` and `_tracing.py:70-80`.
- Issue 3 ("submodules bind their own" mechanism in `module_call_path.md` step 4): replaced the parenthetical with an explicit two-sentence explanation that inner submodules short-circuit through Mechanism B (`base.py:71-72`) so `_bind_parameters_to_context` never runs on them, and that their Parameters reach the parent's `_tensor_bindings` via `functional._dispatch → ctx.wrap_parameter` writing the binding at `_tracing.py:124-125`. Verified against `base.py:68-82`, `base.py:148-151`, and `_tracing.py:121-126`.
- Issue 4 (`_inner` table bullet implying `_unwrap_args` is the normal Parameter path): rewrote the two bullets under the `_inner` table in `tensor_proxy.md` to (a) name `functional._dispatch`'s `Parameter`-pre-wrap step at `functional.py:36-43` and (b) tighten the `_unwrap_args` description to "TensorProxy → `_inner`; raw Parameter (only when something bypassed `functional._dispatch`) → wrap-then-extract". Verified against `functional.py:36-43` and `_tracing.py:70-80`.
- Issue 5 (no critical structural gap): no action required.
- Nav footers, index-as-nav-only, and all spot-checked pins preserved; no new pins introduced.

---

## Pass 2

### Verdict

Accept. All four Pass 1 factual fixes are correctly applied and verified against source; compression preserved every plan-required deliverable and every pin. No new findings.

### Pass 1 fix verification

- **Issue 1 (`_resolve_grid` `device_config is None` rung)** — `tracing_contexts.md:110-115` now lists four rungs with the `device_config is None → None` short-circuit as step 2. Matches `_tracing.py:82-90` verbatim. ✓
- **Issue 2 (`_unwrap_args` Parameter-branch safety net)** — `tracing_contexts.md:93` prose now correctly attributes the safety net to "`dispatch` called outside `functional._dispatch`" and names `functional.py:36-43` as the live-path pre-wrap site. Matches `functional.py:36-43`. ✓
- **Issue 3 (inner-submodule Parameter binding in `module_call_path.md` step 4)** — `module_call_path.md:62` now explicitly names Mechanism B short-circuit at `base.py:71-72` and the `_tracing.py:124-125` binding write. Matches `base.py:68-82`, `base.py:148-151`, and `_tracing.py:121-126`. ✓
- **Issue 4 (`_inner` table bullet in `tensor_proxy.md`)** — `tensor_proxy.md:67-68` correctly names `functional._dispatch`'s `functional.py:36-43` Parameter pre-wrap as the live path and reframes `_unwrap_args`'s `Parameter` branch as the bypass case. ✓

### Compression-preservation spot-check

- **Plan-required content kept:** Mermaid diagrams (3 total: 1 in `module_call_path.md`, 2 in `tracing_contexts.md`, 1 in `tensor_proxy.md`); `_call_graph` and `_call_compose` line-by-line walks; three context classes side-by-side; `_resolve_grid` priority list; `_compiled_cache` dormant-hook flag; `_collect_user_args` / `_get_output_tensor` extension points; compose-mode known-gap with grep evidence and contributor test recipe; `TensorProxy.__slots__` rationale; `_inner` invariant table.
- **Pins preserved:** all original spot-checked pins (`base.py:68-82`, `86-122`, `106-112`, `126-144`, `148-151`, `153-154`, `156-159`, `406-411`, `417-423`, `443-448`; `_tracing.py:20-34`, `37-90`, `70-80`, `82-90`, `93-150`, `128-150`, `153-196`; `_tensor_proxy.py:14-28`; `_registry.py:40-42`, `:43`; orchestrator `__call__` overrides at `attention.py:90`, `decoder_layer.py:32`, `model.py:67`) are still present in the compressed text.
- **Net change:** −47 lines across three content files (per Agent A change log); compression analysis's C1/C2/C3/C4 cuts verified in place; index unchanged.

### New findings

None. All Pass 1 findings closed; no new factual, implementation, or coherence issues surfaced on re-read.
