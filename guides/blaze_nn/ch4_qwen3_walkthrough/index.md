# Chapter 4 — Authoring models: the Qwen3 walkthrough

This chapter closes the model-author layer by walking the only end-to-end model in the repo, `examples/qwen3_embedding_0_6b/`. We use it to show how the public API from Chapters 2 and 3 composes into a real port: how weights arrive from Hugging Face, how runtime state is held outside `state_dict`, how submodules nest, how non-graph host hops are accommodated by orchestrators that bypass the tracing machinery, and where buffer addresses get baked into compiled programs. No new framework concepts appear here — every mechanism was introduced earlier; Chapter 4 only composes.

1. [Layout and the weight loader](layout_and_weight_loader.md) — directory tour of `examples/qwen3_embedding_0_6b/`, the HF → ttnn bridge, and where torch enters the picture.
2. [Tensor lifetimes: Parameter / Buffer / GraphInput](tensor_lifetimes.md) — the three-way vocabulary the port uses, with the verbatim contract from `modules/__init__.py` and the buffer-address invariant.
3. [Composing submodules](composing_submodules.md) — `TokenEmbedding`, `FusedQKV`, `Qwen3MLP`, `RoPE`, `Qwen3DecoderLayer`, and how each maps onto a public-API choice from Chapter 3.
4. [The orchestrator pattern: two mechanisms](orchestrator_pattern.md) — the explicit `__call__` override (Mechanism A) and the active-context short-circuit (Mechanism B), shown side-by-side with named qwen3 modules.
5. [Buffers and address baking](buffers_and_address_baking.md) — `init_*` (allocate) vs `set_*` (bind) hooks, the host-side bridges in `Qwen3Attention`, the Blackhole P150 monkey-patches, and what each test file covers.
