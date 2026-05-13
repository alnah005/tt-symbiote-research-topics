# Chapter 6 — Pass 1 Critic Review (Agent B)

Scope: flag only wrong answers, wrong implementations, or materially misleading claims. Max 5 items.

---

## Verdict

Three issues — one factually wrong, two pin/snippet mismatches. Otherwise the chapter accurately reflects the code.

---

## Issue 1 — Factually wrong: orchestrator submodule calls are NOT "inside an already-active context"

**File:** `caller_allocated_outputs_internals.md`, Step 4, the first bullet under "Two points worth pinning":

> "The check runs only at the outer call boundary. When `_get_active_context()` is `None` we are entering tracing fresh — this is the user's top-level `model(x)` call. **Inside an already-active context (an orchestrator's nested submodule call) the check is skipped**: if the outer module is going to compile, the buffers it inherits from upstream are the compiler's problem, not the enforcement's."

The parenthetical is inverted. An orchestrator (e.g. `Qwen3Attention.__call__` at `examples/qwen3_embedding_0_6b/modules/attention.py:90-91`) overrides `__call__` to call `self.forward(...)` directly — bypassing `Module.__call__` and never opening a tracing context. Every submodule call inside the orchestrator's `forward` (`self.qkv(...)`, `self.sdpa(...)`, `self.o_proj(...)`) therefore hits `OpModule.__call__` (`base.py:413`) with `_get_active_context() is None` — so the pre-check **fires**, not skips. This is precisely the path that enforces the SDPA caller-allocated output that the chapter spends Step 1 and Pitfall 1 explaining.

The check is skipped in the *opposite* case: a non-orchestrator parent that has opened a tracing context (its `Module.__call__` at `base.py:68-72` enters and stays in the active context), whose `forward` then calls child submodules — those nested calls see an active context and short-circuit via the re-entry path at `base.py:71`.

**Why it matters:** the chapter explicitly cross-references Ch4's orchestrator pattern (mermaid step "qwen3's monkey-patch path"), and a contributor reading Step 4 will form a wrong model of when SDPA's set_output_tensor enforcement runs. Fix by inverting: "Inside a parent module that is *itself* being traced (the non-orchestrator re-entry path at `base.py:71`), the check is skipped. Orchestrator-style nested calls run at the outer boundary just like the user's `model(x)` and so re-run the check on every submodule."

## Issue 2 — Pin mismatch: `linear.py:26-28` does not include `DownProj`

**File:** `caller_allocated_outputs_internals.md`, Step 5, inside the canonical `define_fused_op` snippet's comment line:

> `# ... imports: Mcast, Gather, DownProj — see linear.py:26-28`

Actual `blaze_nn/modules/linear.py:26-28` is:

```
26: from blaze.blaze_op import BlazeOp, FusedOp, Input, Output
27: from blaze.ops.mcast import Mcast
28: from blaze.ops.gather import Gather
```

`DownProj` is imported on line 29 (`from blaze.ops.down_proj.op import DownProj`). The correct range is `linear.py:27-29`. The mistake propagates to the Mermaid label/cross-reference and would cause a "writer must re-verify each pin" violation under the convention in `plan.md:38`.

## Issue 3 — Diagram cosmetic: `_dispatch` passes `resolved_args`, not `args`

**File:** `functional_dispatch.md`, end-to-end resolution Mermaid diagram, node `I`:

> `I["ctx.dispatch(backend, *args, **kwargs)<br/>_tracing.py:128"]`

The actual `_dispatch` (`functional.py:43`) passes `*resolved_args` to `ctx.dispatch`:

```python
return ctx.dispatch(backend, *resolved_args, **kwargs)
```

`resolved_args` is the post-Parameter-wrap list built in node `H`. Showing `*args` in node `I` undercuts the chapter's own observation #4 ("Only `Parameter` instances are wrapped … the active context's `_unwrap_args` handles `TensorProxy` unwrapping at dispatch time") by making it look like the wrap step has no downstream effect. Cosmetic but materially misleading in a teaching diagram.

---

_Reviewer: Agent B · Pass 1 · 2026-05-13_

---

## Agent A change log — applied after Pass 1 B review
- Issue 1: Inverted the parenthetical in Step 4 of `caller_allocated_outputs_internals.md`. The first "Two points worth pinning" bullet now explains that `_get_active_context() is None` covers two paths — the user's top-level `model(x)` call **and** orchestrator-style nested submodule calls (because orchestrators override `__call__` to call `self.forward(...)` directly and never open a tracing context, per `examples/qwen3_embedding_0_6b/modules/attention.py:90-91`). The "check is skipped" case now correctly describes the non-orchestrator re-entry path at `base.py:71`.
- Issue 2: Corrected pin in Step 5's canonical `define_fused_op` snippet from `linear.py:26-28` to `linear.py:27-29` so the range actually contains the `Mcast`, `Gather`, and `DownProj` imports (verified against `blaze_nn/modules/linear.py:27-29`).
- Issue 3: Updated Mermaid node `I` in the end-to-end resolution diagram (`functional_dispatch.md`) from `ctx.dispatch(backend, *args, **kwargs)` to `ctx.dispatch(backend, *resolved_args, **kwargs)`, matching `functional.py:43` and preserving the chapter's observation #4 about Parameter wrapping having a downstream effect.

---

## Pass 2

Scope: same as Pass 1 — flag only wrong answers, wrong implementations, or materially misleading claims. Max 5 items.

### Verification of Pass 1 fixes

- **Issue 1 (Step 4 orchestrator inversion)** — fix sticks. Step 4's first "Two points worth pinning" bullet now correctly says `_get_active_context() is None` covers (a) the user's top-level `model(x)` call **and** (b) orchestrator-style nested submodule calls, because the orchestrator's `__call__` override at `examples/qwen3_embedding_0_6b/modules/attention.py:90-91` calls `self.forward(...)` directly and never opens a tracing context. The "skipped" case is correctly attributed to the non-orchestrator re-entry path at `base.py:71`. Verified against `blaze_nn/modules/base.py:68-72` (re-entry short-circuit) and `examples/qwen3_embedding_0_6b/modules/attention.py:90-91` (orchestrator override).
- **Issue 2 (pin range)** — fix sticks. The canonical `define_fused_op` snippet in Step 5 now reads `linear.py:27-29`, and `blaze_nn/modules/linear.py:27-29` contains exactly `Mcast`, `Gather`, and `DownProj` imports.
- **Issue 3 (Mermaid `*args` vs `*resolved_args`)** — fix sticks. End-to-end resolution diagram node `I` now reads `ctx.dispatch(backend, *resolved_args, **kwargs)`, matching `blaze_nn/functional.py:43`.

### New issues found this pass

None. Spot-checked the remaining load-bearing pins — `_tracing.py:70` (`_unwrap_args` definition), `_tracing.py:82-90` (`_resolve_grid`), `_tracing.py:128-150` (`GraphTracingContext.dispatch`), `_registry.py` entries, the eight `pytest.raises(... match="no active tracing context")` ops in `tests/test_functional.py:TestFunctionalNoContext`, the test class names (`TestLinearBiasNotSupported`, `TestAliasResolution`, `TestDynamicDispatch`, etc.), and the qwen3 idempotency test at `examples/qwen3_embedding_0_6b/tests/test_l1_sdpa.py:test_sdpa_decode_user_alloc_monkey_patch_idempotent`. All anchor to the code.

One ignorable nit (not flagged as an issue): the Pitfall-1 text says the idempotency test "asserts the tuple identity is preserved on the second and third calls"; the test actually asserts equality (`==`) of the tuple value, not `is` identity. This is immaterial to the teaching point and does not mislead a contributor about behavior.

### Verdict

Pass 1 fixes all stuck; no new wrong answers, wrong implementations, or materially misleading claims. Chapter is accurate against the code.

_Reviewer: Agent B · Pass 2 · 2026-05-13_
