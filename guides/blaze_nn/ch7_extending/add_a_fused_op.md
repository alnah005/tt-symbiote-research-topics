# Adding a fused op — when the op does not exist upstream

The wrapper pattern in [Adding an op wrapper](add_an_op_wrapper.md) assumes the op name you set on the class is already registered in `BlazeOp._class_registry`. When it is not — because the op is a composition of upstream tt-blaze primitives that no one has packaged as a single `FusedOp` yet — you need to synthesize the fused op from blaze-nn itself. The pattern is `BlazeNNLinear` in `blaze_nn/modules/linear.py`.

This is a contributor escape hatch, not a recommended general path. Composing fused ops belongs upstream in tt-blaze. Do it here only when the composition is specific to blaze-nn's ABI (caller-allocated output buffers, model-author-facing kwargs) or while you are prototyping an op that will eventually move upstream.

## When you reach for this

You are writing a wrapper class in `blaze_nn/modules/` and you discover:

- The op name you want to dispatch through is not in `BlazeOp._class_registry`.
- The op is composed of upstream primitives (e.g. `Mcast`, `DownProj.matmul`, `Gather` for `Linear`).
- The composition is stable enough to package as a single fused op with named `Input` / `Output` ports.
- Callers must pre-allocate one or more output buffers (otherwise this is just an alias and the dispatch shim in `functional.py` is enough).

If any of these is not true, prefer one of the alternatives: write the composition inline in `forward` using `F.*` (the `Qwen3MLP` pattern), add an alias entry in `_REGISTRY` (Chapter 6), or push the fused op upstream into tt-blaze.

## The canonical example, walked

`blaze_nn/modules/linear.py:23-59` is the only fused op blaze-nn synthesizes today. The relevant parts:

```python
class Linear(OpModule):
    op = "blaze_nn_linear"
    params = ("weight",)

    @classmethod
    def define_fused_op(cls) -> None:
        import blaze
        from blaze.blaze_op import BlazeOp, FusedOp, Input, Output
        from blaze.ops.mcast import Mcast
        from blaze.ops.gather import Gather
        from blaze.ops.down_proj.op import DownProj

        if "blaze_nn_linear" in BlazeOp._class_registry:
            return
        # ... class BlazeNNLinear(FusedOp): ...
        BlazeNNLinear.register()
        if not hasattr(blaze, BlazeNNLinear.name):
            setattr(blaze, BlazeNNLinear.name, blaze._OpHandle(BlazeNNLinear))
```

Six pieces are load-bearing:

1. **`define_fused_op` is a `@classmethod` on `OpModule`** — `OpModule.__init__` calls it exactly once per subclass via the `_fused_op_defined` class-level flag (see `blaze_nn/modules/base.py:345-349`). The default in `OpModule` is a no-op, so wrappers that do not need synthesis pay nothing.
2. **All tt-blaze imports are local to the method** — they cannot live at module scope inside `blaze_nn/modules/`, because `import blaze_nn` must succeed without tt-blaze installed. The `import blaze` and `from blaze.blaze_op import ...` lines are inside `define_fused_op`, executed only when an instance is constructed (which requires tt-blaze anyway).
3. **Membership check before defining** — `if "blaze_nn_linear" in BlazeOp._class_registry: return`. This is a second guard, redundant with `_fused_op_defined` for normal flows but necessary when the same op name has been registered by another path (qwen3's `_register_sdpa_decode_user_alloc` is the analogous pattern for monkey-patching `user_allocated_outputs` onto an existing op).
4. **`class BlazeNNLinear(FusedOp)`** declared inside the method — this is a tt-blaze concept (see tt-blaze's `blaze_op.py`), not a blaze-nn one. The class body declares ports and the `compose` recipe.
5. **`BlazeNNLinear.register()`** — registers the class into `BlazeOp._class_registry` under `BlazeNNLinear.name`.
6. **`setattr(blaze, name, blaze._OpHandle(BlazeNNLinear))`** — also publishes a handle on the `blaze` module so that `GraphTracingContext.dispatch` can resolve `getattr(blaze, op_name)` without `AttributeError`. Both registrations are required.

## The `FusedOp` body

The class body inside `define_fused_op` has four parts:

```python
class BlazeNNLinear(FusedOp):
    name: str = "blaze_nn_linear"
    math_fidelity: str = "LoFi"
    user_allocated_outputs: tuple[str, ...] = ("output",)

    input: Input = Input()
    weights: Input = Input()
    output: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        act = Mcast.emit(f, tensors["input"], prefix="mcast")
        mm = DownProj.matmul(f, act, tensors["weights"],
                             prefix="matmul", cores=f.matmul_cores)
        Gather.emit(f, mm, output_tensor=output, prefix="gather")
```

1. **Class metadata** — `name` is the op name (string-matches the `op` class attr on `Linear`); `math_fidelity` is a tt-blaze knob (`LoFi` is bf16 inputs with reduced internal precision; `HiFi*` variants exist for accumulation-heavy ops).
2. **`user_allocated_outputs`** — declared as a tuple of output-port names. This is the source of truth `OpModule.__init__` reads via `_lookup_user_allocated_outputs` (see [Chapter 6 — Caller-allocated outputs internals](../ch6_dispatch_and_registry/caller_allocated_outputs_internals.md)). The declared port names here must match the keys callers pass to `set_output_tensors(**kwargs)`.
3. **`Input` / `Output` port declarations** — each port name becomes a key in the `tensors` dict that `compose` receives. The compiler's port-alias dual-key population at `blaze_nn/modules/base.py:107-112` makes sure both the upstream "by port name" key and the blaze-nn "by `ExternalTensor` name" key resolve to the same backing tensor.
4. **`compose(cls, f, tensors, output, user_args)`** classmethod — the actual recipe. `f` is the `FusedProgram` being built. `tensors[<port>]` is a backend tensor handle. `output` is the caller-allocated buffer for the single output port (or a tuple for multi-output). `user_args` is the dict produced by `_collect_user_args` on the originating `Module` — note that `Linear` reads `user_args["blackhole_cores"]` here only after the qwen3 monkey-patch swaps the cores logic (`examples/qwen3_embedding_0_6b/modules/_blaze_nn_linear_patch.py`); the in-tree `compose` uses `f.matmul_cores`. Each `prefix=` tags the resulting node with a stable name in the graph — useful when inspecting `BlazeGraph` nodes from a dispatch-integration test.

## Registration sequence and idempotence

`OpModule.__init__` runs `define_fused_op` before `_lookup_user_allocated_outputs`:

```python
# blaze_nn/modules/base.py:345-349
cls = type(self)
if (cls.define_fused_op is not OpModule.define_fused_op
        and not cls.__dict__.get("_fused_op_defined", False)):
    cls.define_fused_op()
    cls._fused_op_defined = True
# ... then ...
required_outputs = _lookup_user_allocated_outputs(op_name)
```

Two things matter:

- **Order.** Synthesis must happen first; otherwise `_lookup_user_allocated_outputs` queries an empty registry and returns `()`, and `set_output_tensor` raises `ValueError` because the OpModule has no declared outputs.
- **Idempotence — three independent guards.** They protect against different races, and all three should remain in place:
  1. **`cls.define_fused_op is not OpModule.define_fused_op`** — only runs the hook for subclasses that actually override it. Pure `ops/` wrappers (e.g. `RMSNorm`) leave it as the base no-op and pay nothing.
  2. **`cls.__dict__.get("_fused_op_defined", False)`** — the per-subclass "done" flag. Critically, this is read from the subclass's own `__dict__`, **not** via attribute lookup. That means sibling subclasses each get their own flag and a subclass cannot accidentally inherit the parent's done-state — important if you ever subclass a class that already defined a fused op.
  3. **`if "blaze_nn_linear" in BlazeOp._class_registry: return`** — the inner registry check on tt-blaze's side. Catches the case where some other path (a different blaze-nn module, user code, a sibling test) already registered the same name. The accompanying `if not hasattr(blaze, ...)` guard on the `setattr` line below extends the same idempotence to the `blaze`-module handle publication.

Together these guards make registration safe under: repeated instantiations, repeated imports of the module, parallel imports from sibling packages, and re-imports inside test runners.

## Decision flow

```mermaid
graph TD
    A[New wrapper class] --> B{Is op name in<br/>BlazeOp._class_registry?}
    B -- Yes --> C[Use ops/ pattern.<br/>define_fused_op = no-op]
    B -- No --> D{Caller-allocates outputs?<br/>Hardware-specific compose?}
    D -- No --> E[Push fused op upstream<br/>to tt-blaze]
    D -- Yes --> F[Override define_fused_op<br/>in modules/]
    F --> G[Declare FusedOp class<br/>with Input/Output ports]
    G --> H[Set user_allocated_outputs<br/>declare math_fidelity]
    H --> I[Implement compose<br/>classmethod]
    I --> J[register() + setattr<br/>on blaze module]
```

## When the recipe is wrong for the job

Three smells that mean you should not be reaching for `define_fused_op`:

1. **You need a new low-level kernel.** That is a tt-blaze contribution, not a blaze-nn one. `define_fused_op` only composes kernels that already exist upstream.
2. **The op already exists upstream.** Use the `ops/` wrapper pattern instead. Synthesizing a duplicate of an upstream op creates two registry entries with the same role and produces confusing failures when imports race.
3. **You want runtime branching inside `compose`.** `compose` runs once per subclass at first instantiation and produces a fixed `FusedOp`; if the dataflow shape depends on a runtime value (a tensor's shape, a flag passed at call time), you want a regular `Module` that builds the graph through `F.*` calls in `forward()`, not a `FusedOp`.

## Tests to add

Synthesis is hardest to verify with only framework-only tests, so prefer the dispatch-integration tier:

1. **Dispatch-integration test** (`tests/test_dispatch_integration.py`, gated by `pytest.importorskip("blaze")`) — open a `GraphTracingContext`, instantiate your subclass, run `forward`, and assert the synthesized op name appears in `ctx.graph.nodes`. The synthesis happens during `__init__`, so a passing test also confirms that `BlazeOp._class_registry` was populated.
2. **`user_allocated_outputs` test** — also framework-importable: instantiate your class, assert `m._required_output_names == ("your_port",)`, then assert `m.set_output_tensor(object())` succeeds and `m._output_tensors["your_port"]` is the sentinel.
3. **Parity test (device-gated)** — if you have a torch reference, add a case in `tests/test_pytorch_parity.py` mirroring `test_linear_pipeline_matches_torch`.

> **Warning:** There is currently no end-to-end test that exercises a synthesized fused op outside `Linear`. If you add a second fused op, write the dispatch-integration test before the parity test — graph-construction failures are easier to debug than device-side failures.

---

_Previous: [Adding an op wrapper — the `blaze_nn/ops/<op>/` convention](add_an_op_wrapper.md) · Next: [Extending containers and modules — beyond the built-ins](extending_containers_and_modules.md) · [Up](index.md)_
