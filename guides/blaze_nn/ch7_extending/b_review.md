# Agent B (Critic) Review — Chapter 7 Pass 1

**Verdict:** Mostly accurate; flagging 4 materially misleading or wrong claims. None block; all are localized text fixes.

---

## Findings

### 1. `contributing_checklist.md:111` — wrong failure site for interop-in-forward

> "any subsequent `F.<op>(torch_tensor, ...)` will fail in `TracingContext._unwrap_args`."

**Wrong.** `TracingContext._unwrap_args` (`blaze_nn/_tracing.py:70-80`) explicitly handles non-`TensorProxy`/non-`Parameter` args by passing them through unchanged (the final `else: out.append(a)` branch). A torch tensor would flow through `_unwrap_args` silently and fail later — inside the backend op call (`op_handle(*unwrapped_args, ...)` at `_tracing.py:149`) or downstream in the compiler. The anti-pattern is still real, but the named failure site is wrong; this misleads contributors debugging the symptom.

**Fix:** Change to "will fail downstream when the backend op receives a torch tensor it cannot consume; `_unwrap_args` passes non-proxy args through unchanged."

---

### 2. `extending_containers_and_modules.md:88` — "silently corrupts the graph" is false

> "**Warning:** If you override `__call__` for an orchestrator, do **not** also override `forward` to participate in tracing — the two patterns are mutually exclusive. Either the module is a graph (default behavior, `forward` is traced) or it is an orchestrator (overridden `__call__`, `forward` runs as plain Python). Mixing them silently corrupts the graph."

**Wrong / materially misleading.** Mixing does not silently corrupt anything. If a contributor overrides `__call__` to bypass tracing and then writes a `forward` that calls `F.<op>(...)`, `_dispatch` (`blaze_nn/functional.py:24-32`) checks the active context and raises `RuntimeError("blaze_nn.F.<op>() must be called inside a Module.forward(). There is no active tracing context.")` — loudly. The framework's invariant is enforced; "silently corrupts" overstates the danger and tells contributors to fear a non-existent failure mode while training them to ignore the actual loud error they will see.

**Fix:** Drop "silently corrupts the graph"; the real failure is the no-active-context `RuntimeError`. State that directly.

---

### 3. `add_an_op_wrapper.md:73` — Warning hides the `define_fused_op` interaction

> "If you set `op` to a name that is **not** registered in `BlazeOp._class_registry`, the wrapper will construct fine — the op is not consulted at `__init__` except via `_lookup_user_allocated_outputs`, which returns `()` for unknown ops..."

**Materially misleading.** This is only true when the subclass does *not* override `define_fused_op`. If it does, `OpModule.__init__` runs `define_fused_op` at construction time (`blaze_nn/modules/base.py:346-349`) — *before* `_lookup_user_allocated_outputs` at line 360. So a subclass that overrides `define_fused_op` will fail at `__init__` if synthesis fails, contradicting "the wrapper will construct fine." The chapter is the canonical "ops/ wrapper" recipe (no `define_fused_op` override), so the claim is correct *in context* — but the warning is the very place a reader confusing the two paths would land. The "fused op" path is one click away in `add_a_fused_op.md` but the warning gives no hint.

**Fix:** Add one clause: "...the wrapper will construct fine *unless you also override `define_fused_op`* (see [Adding a fused op](add_a_fused_op.md)) — that hook runs at `__init__` time, before the lookup."

---

### 4. `extending_containers_and_modules.md:90-103` — Mechanism B advice is unmotivated

> "If you override `__call__` *and* you intend the module to also be usable as a child inside another module's tracing context, preserve the re-entry check from the base class:"

**Materially misleading guidance.** No qwen3 orchestrator does this (verified: `examples/qwen3_embedding_0_6b/modules/attention.py:90`, `decoder_layer.py:32`, `model.py:67` each use the bare `return self.forward(*args, **kwargs)` two-liner). And there is no in-tree case where an orchestrator is invoked from inside another tracing context — orchestrators are by definition the top of the call. Recommending a defensive re-entry check at the top of every custom `__call__` adds API surface that no real code exercises and contradicts the qwen3 pattern the same file points to as the canonical orchestrator example. The simpler advice — "if you don't need to participate in tracing as a child, the bare two-liner is sufficient; if you do, don't override `__call__` at all" — is what the codebase actually demonstrates.

**Fix:** Either delete Mechanism B, or reframe it as "very rare, no in-tree example" and lead with the two-liner.

---

## Minor pin drift (not flagged, FYI)

- `add_an_op_wrapper.md:118` cites `tests/test_dispatch_integration.py:21-30` for `test_linear_alias_creates_matmul_node`; actual `def test_linear_alias_creates_matmul_node` is at line 25. Within tolerance, but the pin-verification rule from `Conventions` requires re-verification before commit.
- `add_an_op_wrapper.md:69` cites `blaze_nn/modules/base.py:427-428` for the auto-init branch; verified, matches exactly.
- `add_a_fused_op.md:45` cites `blaze_nn/modules/base.py:345-349` for `define_fused_op` invocation; verified.
- `extending_containers_and_modules.md:111` cites `blaze_nn/modules/base.py:443-448` for `OpModule._collect_user_args`; verified.

## Scope-check

Per plan.md user→contributor boundary (Ch4↔Ch5), Ch7 contributor-only audience: respected throughout. No model-author callouts misplaced. Cross-references to Ch5/Ch6 are present and correct.

---

## Change log — Agent A pass

All four B issues applied; minor pin drift on `add_an_op_wrapper.md:118` also corrected. Files touched: `add_an_op_wrapper.md`, `extending_containers_and_modules.md`, `contributing_checklist.md`.

### Issue 1 — `contributing_checklist.md:111` (interop failure site)

Rewrote the second half of Anti-pattern 2's last sentence. Removed the false claim that `_unwrap_args` is the failure site; added the verified failure flow: torch tensors pass through `_unwrap_args` (`blaze_nn/_tracing.py:70-80`, the `else: out.append(a)` branch), and the failure surfaces at `op_handle(*unwrapped_args, ...)` in `_tracing.py:149` or further downstream in the compiler. Both pins re-verified against source.

### Issue 2 — `extending_containers_and_modules.md:88` ("silently corrupts" claim)

Replaced the "Mixing them silently corrupts the graph" line with the actual loud-failure mode: if an orchestrator's plain-Python `forward` calls `F.<op>(...)`, `_dispatch` (`blaze_nn/functional.py:24-32`) raises the no-active-context `RuntimeError`. Verbatim error text quoted from source (`functional.py:29-32`).

### Issue 3 — `add_an_op_wrapper.md:73` (`define_fused_op` warning gap)

Inserted the missing clause about the `define_fused_op` interaction. The warning now reads "the wrapper will construct fine *unless you also override `define_fused_op`*", links to `add_a_fused_op.md`, and adds the verified ordering note: `define_fused_op` runs at `blaze_nn/modules/base.py:345-349`, before the `_lookup_user_allocated_outputs` call at line 360 — so synthesis failures surface at construction. Confirmed against `base.py:340-362`.

### Issue 4 — `extending_containers_and_modules.md:90-103` (Mechanism B unmotivated)

Reframed Mechanism B from "preserve the re-entry check" prescription to a clearly-marked rare-case note ("no in-tree example"). Section header now reads "Mechanism B — the active-context short-circuit (rare; no in-tree example)". Lead-in restates the bare two-liner that the three qwen3 orchestrators actually use (`attention.py:90`, `decoder_layer.py:32`, `model.py:67`, all re-verified by grep). Closing sentence is the simpler advice B suggested: bare two-liner if you don't need child-of-tracing behavior; don't override `__call__` at all if you do.

### Minor pin (FYI in B's review)

`add_an_op_wrapper.md:118` updated from `tests/test_dispatch_integration.py:21-30` to `tests/test_dispatch_integration.py:25` to point at the exact `def test_linear_alias_creates_matmul_node` line (verified by grep against the source file).

### Verification

- Footers: all five content files retain their `_Previous · Next · [Up]_` footer (intact).
- `index.md`: nav-only, 20 lines, no content changes needed.
- Plan-bullet coverage: the chapter's five files still cover (1) op wrappers, (2) fused ops, (3) extending containers/modules, (4) test taxonomy, (5) contributing checklist + anti-patterns — unchanged.
- Pins verified against source: `_tracing.py:70-80`, `_tracing.py:149`, `functional.py:24-32`, `modules/base.py:345-349`, `modules/base.py:360`, `modules/base.py:71`, `tests/test_dispatch_integration.py:25`, qwen3 `__call__` overrides at `attention.py:90` / `decoder_layer.py:32` / `model.py:67`.

---

## Pass 2

**Verdict:** Clean. All four Pass 1 fixes stuck and verify against source; no new wrong-answer / wrong-implementation / materially-misleading findings.

### Pass 1 fix verification

1. **`contributing_checklist.md:111` (Issue 1 — interop failure site).** New text correctly states that `_unwrap_args` (`blaze_nn/_tracing.py:70-80`) passes non-`TensorProxy`/non-`Parameter` args through via the `else: out.append(a)` branch and that failure surfaces at `op_handle(*unwrapped_args, ...)` at `_tracing.py:149`. Both pins re-verified against source. The false "fail in `_unwrap_args`" claim is gone.
2. **`extending_containers_and_modules.md:88` (Issue 2 — "silently corrupts" claim).** New text quotes the exact `RuntimeError("blaze_nn.F.<op>() must be called inside a Module.forward(). There is no active tracing context.")` and pins `_dispatch` at `blaze_nn/functional.py:24-32`. Verified verbatim against `functional.py:29-32`. The false "silently corrupts" framing is gone.
3. **`add_an_op_wrapper.md:73` (Issue 3 — `define_fused_op` gap).** New text adds the "unless you also override `define_fused_op`" clause, links to `add_a_fused_op.md`, and pins the ordering: `define_fused_op` at `blaze_nn/modules/base.py:345-349` runs before `_lookup_user_allocated_outputs` at line 360. Verified — `cls.define_fused_op()` is at line 348, `required_outputs = _lookup_user_allocated_outputs(op_name)` is at line 360.
4. **`extending_containers_and_modules.md:90-104` (Issue 4 — Mechanism B reframe).** Section header now reads "Mechanism B — the active-context short-circuit (rare; no in-tree example)". Lead-in restates the bare two-liner used by all three qwen3 orchestrators (verified at `attention.py:90`, `decoder_layer.py:32`, `model.py:67`). Closing sentence delivers the simpler advice. The "preserve the re-entry check" prescription is gone; the codebase pattern is now what the section leads with.

### Other pin / claim spot-checks (Pass 2 fresh)

- `blaze_nn/modules/base.py:68-82` (default `__call__`): verified — opens at line 68, re-entry check at 71.
- `blaze_nn/modules/base.py:107-112` (port-alias dual-key population in `_call_graph`): verified — `tensor_to_ports` walk matches.
- `blaze_nn/modules/base.py:269-285` (`_lookup_user_allocated_outputs`): verified — returns `()` for unknown ops and for missing blaze import.
- `blaze_nn/modules/base.py:427-428` (auto-init branch): verified — `if not self._state_loaded and self._torch_init_specs():` at 427, `self.init_torch_params()` at 428.
- `blaze_nn/modules/base.py:440-441` (op kwargs merge in `OpModule.forward`): verified.
- `blaze_nn/modules/base.py:443-448` (`OpModule._collect_user_args`): verified.
- `blaze_nn/modules/linear.py:23-59` (fused `Linear` body): all six load-bearing pieces present and ordered as described. (Pin range starts at `@classmethod` on line 23; the surrounding `class Linear` declaration is at line 8 — minor presentational nit, not a wrong-implementation flag.)
- `blaze_nn/modules/linear.py:67-70` (bias `NotImplementedError`): verified.
- qwen3 `examples/qwen3_embedding_0_6b/modules/qkv_proj.py:29` (`self._ua_blackhole_cores = "64x8"`): verified.
- qwen3 `examples/qwen3_embedding_0_6b/modules/qkv_proj.py:40-45` (`_collect_user_args`): verified — `def` at line 40, body through 45.

No new (a), (b), or (c) flags. Pass 2 closes clean.
