# Chapter 6 — Op dispatch, the registry, and caller-allocated outputs

This chapter opens the second half of the contributor view. Chapter 5 walked the `Module.__call__` → tracing-context → `program.run()` path; this chapter explains the three small but load-bearing pieces that sit alongside it:

1. The `blaze_nn.functional` (`F`) dispatch layer — how `F.<any_op>(*args)` resolves to a tt-blaze op call without per-op wiring, and the two narrow shims (`linear`, `sliced_matmul`) that override that default.
2. `blaze_nn/_registry.py` — the tiny `OpInfo` table that tells the tracing contexts which ops are aliases, which run on the matmul subgrid, and which need a `sender` core kwarg.
3. The full `OpModule` ↔ `BlazeOp.user_allocated_outputs` chain — how `_lookup_user_allocated_outputs`, `_required_output_names`, `set_output_tensor[s]`, `_get_output_tensor`, and `define_fused_op` interlock to make `Linear`'s caller-allocated output buffer work.

By the end of this chapter you should be able to add a new tt-blaze op wrapper, add a new alias or placement hint, and synthesize a fused op that requires a caller-allocated output — without reading code outside `blaze_nn/`.

## Files

1. [Functional dispatch — `_dispatch` and the lazy `__getattr__`](functional_dispatch.md)
2. [The op registry — aliases and placement hints](registry.md)
3. [Caller-allocated outputs — internals](caller_allocated_outputs_internals.md)

---

_Previous: [Chapter 5 — Tracing internals: from `Module.__call__` to `program.run()`](../ch5_tracing_internals/tensor_proxy.md) · Next: [Chapter 7 — Extending blaze-nn](../ch7_extending/index.md) · [Up](index.md)_
