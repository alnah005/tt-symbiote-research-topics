# Chapter 5 — Tracing internals: from `Module.__call__` to `program.run()`

This chapter opens the user → contributor boundary. Chapters 1–4 covered the public surface a model author needs; from here on we look at the private half of `blaze_nn/modules/base.py`, the contents of `blaze_nn/_tracing.py`, and the `TensorProxy` handle in `blaze_nn/_tensor_proxy.py`. The goal is to make every claim made earlier — "the framework opens a tracing context", "every nested call is its own compile", "buffer-address kwargs bypass graph ports" — falsifiable against the source.

This is the first **contributor chapter**. The audience is engineers modifying blaze-nn itself. Public-API surface (`Module`, `Parameter`, `OpModule`, `F.<op>`) is assumed; private names (`_call_graph`, `_active_context`, `TensorProxy._inner`) are now fair game. Readers are assumed to have read Chapter 4 and to know tt-blaze's `BlazeOp`, `blaze.fuse()`, `BlazeGraph`, and `BlazeCompiler` by name.

## Contents

1. [The module call path: `model(x)` to `program.run()`](module_call_path.md) — annotated Mermaid flow plus a line-by-line trace through `Module.__call__`, `_call_graph`, and `_call_compose`, with the `_compiled_cache`, `_collect_user_args`, and `_get_output_tensor` extension points named.
2. [Tracing contexts: `TracingContext`, `GraphTracingContext`, `ComposeTracingContext`](tracing_contexts.md) — the `_active_context` module-global, the single-threaded assumption, the three context classes side-by-side, and the `_resolve_grid` priority that consumes the `_registry` flags from Chapter 6.
3. [`TensorProxy`: the opaque handle](tensor_proxy.md) — `__slots__` rationale, the `_inner` invariant, and how `_name` becomes the graph-input port name.

_Previous: [Chapter 4 — Authoring models: the Qwen3 walkthrough](../ch4_qwen3_walkthrough/buffers_and_address_baking.md) · Next: [The module call path: `model(x)` to `program.run()`](module_call_path.md) · [Up](index.md)_
