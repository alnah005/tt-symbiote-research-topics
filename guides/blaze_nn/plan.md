# blaze-nn Guide — Final Plan

## Selection Rationale

**Base plan: v1, with three targeted borrowings from v2.** v1's overall structure is the stronger backbone: a clean Ch2 (Module + Parameter) → Ch3 (Containers + OpModule) split that paces a torch reader's mental model, an explicit `orchestrator_pattern.md` file in Ch4 (which the spot-check below confirms is the right framing), a dedicated `testing_strategy.md` reverse-index in the contributor capstone, and an `caller_allocated_outputs_internals.md` file that collects the `user_allocated_outputs` ↔ `_lookup_user_allocated_outputs` ↔ `define_fused_op` chain in one place. v1's conventions section is also more concrete (Parameter/Buffer/GraphInput vocabulary pre-registered, callout taxonomy enumerated, Mermaid convention named).

**Borrowed from v2:** (1) a dedicated `prebuilt_modules.md` file in Ch3 covering `blaze_nn.Linear(in_features, out_features, bias=False)` and `blaze_nn.ops.RMSNorm(normalized_shape, eps)` as the recommended starting point — this fixes v1's real user-facing gap where pre-built modules appear only as examples inside the subclass tutorial. (2) v2's tighter "source-of-truth pin" convention with a writer-verification rule (added to the Conventions section). (3) v2's audience-callout blockquote standard (`> **For contributors:** ...`), promoted to canonical.

**Resolution of evaluator-flagged issues, by spot-check:**

- *v1 issue #1 (no prebuilt_modules home)* — Resolved by adopting v2's dedicated `prebuilt_modules.md` in Ch3.
- *v1 issue #2 (orchestrator override claim)* — The evaluator was **factually wrong**. Spot-check at `examples/qwen3_embedding_0_6b/modules/attention.py:90`, `modules/decoder_layer.py:32`, `modules/model.py:67` confirms qwen3 orchestrators DO override `__call__` (each is the two-liner `def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)`). The active-context short-circuit at `blaze_nn/modules/base.py:71` is a *separate* mechanism that handles re-entry from non-orchestrator sub-modules called inside an already-tracing forward. v1's framing is correct; the file is retained, but the description is sharpened to explain both mechanisms side-by-side and name which qwen3 modules use which.
- *v1 issue (set_position_ids vs init_position_ids naming mix)* — Spot-check at `examples/qwen3_embedding_0_6b/modules/model.py:70,78` confirms BOTH exist (`init_position_ids` allocates the buffer; `set_position_ids` binds an existing tensor). The plan now distinguishes them explicitly.
- *v2 issue #1 (interop placement)* — Resolved by moving `interop` coverage into a small file at the END of Ch2 (`interop_at_the_boundary.md`), since model authors need `to_device_tensor` before `load_state_dict`. The contributor "no torch in core" rule remains in the Ch7 contributing checklist as a one-bullet anti-pattern.
- *v2 issue #2 (missing test taxonomy)* — Adopted v1's `testing_strategy.md` in Ch7.
- *v2 issue #3 (overloaded Ch2)* — Resolved by v1's two-chapter split.
- *v2 issue #4 (fragmented user_allocated_outputs)* — Resolved by v1's single-file `caller_allocated_outputs_internals.md` in Ch6; Ch3's user-level `output_tensors.md` is reduced to user-scope only (one example, one rule) per the v1 evaluator's recommendation, removing the v1 redundancy.

Final chapter count: **7**. User → contributor boundary: **between Ch4 and Ch5**, unchanged from both inputs.

## Audience

This guide serves two layered audiences in sequence.

1. **Model authors** (Chapters 1–4). Tenstorrent engineers porting a model — for example the in-tree `examples/qwen3_embedding_0_6b/` — who already know PyTorch's `nn.Module`/`Parameter` idioms and have basic familiarity with `ttnn.Tensor`, `MemoryConfig`, sharding, and what tt-blaze is. They need the public surface: `Module`, `Parameter`, the four containers, `OpModule` (both forms), `blaze_nn.Linear` / `blaze_nn.ops.RMSNorm`, `blaze_nn.functional` (`F`), `state_dict()` / `load_state_dict()`, `module.to(device)`, the torch ↔ ttnn `interop` helpers, and how all of this composes in the qwen3 walkthrough. They do **not** need to know tracing internals.

2. **Framework contributors** (Chapters 5–7). Engineers modifying blaze-nn itself — the `Module` base class, the tracing contexts (`_tracing.py`), the universal `F.__getattr__` dispatch (`functional.py`), the op alias / placement registry (`_registry.py`), the `TensorProxy` handle (`_tensor_proxy.py`), and the `ops/` extension surface. They have read Chapters 1–4 and now want to know what happens between `model(x)` and `program.run()`. They are expected to know tt-blaze's `BlazeOp`, `FusedOp`, `BlazeGraph`, `BlazeCompiler`, and `_class_registry` by name; cross-references to `tt-blaze` source land here, not earlier.

The user → contributor boundary is **between Chapter 4 and Chapter 5**: everything up to and including the qwen3 walkthrough is consumable using only `blaze_nn`'s public exports. Chapter 5 onward opens `_tracing.py`, `_tensor_proxy.py`, `_registry.py`, and the private half of `modules/base.py`.

## Conventions

- **Names.** Three distinct proper nouns appear constantly:
  - **blaze-nn** (lowercase, hyphenated) — this framework, the package distribution name. The Python import is `blaze_nn` (underscore). Never call the framework "blaze".
  - **tt-blaze** — the upstream dataflow compiler. Its Python import is `blaze`. Its op registry is `BlazeOp._class_registry`.
  - **ttnn** — Tenstorrent's tensor library. The only tensor type users place data into is `ttnn.Tensor`.
- **File references.** Always repo-relative inside backticks, e.g. `` `blaze_nn/_tracing.py:115` `` for a specific line, `` `blaze_nn/modules/base.py:68` `` for an entry-point. Directories use a trailing slash: `examples/qwen3_embedding_0_6b/modules/`.
- **Source-of-truth pins.** Every behavioral claim quoting code includes a parenthetical pin (`see blaze_nn/modules/base.py:68`). **The writer must re-verify each pin against the current file before final commit** — pins drift and bare-citation drift erodes the guide's authority.
- **API references.** Public names: plain backticks (`Module`, `Parameter`, `F.linear`, `Module.to`, `OpModule.set_output_tensor`). Private names: underscore prefix plus an inline note ("internal — not part of the public API").
- **Code blocks.** Always declare the language (` ```python `, ` ```bash `). Use `# ...` to elide irrelevant lines. Show imports the first time a symbol appears in a chapter; omit after that. Never paste more than ~25 lines — if a snippet would be longer, summarize in prose and link to the source file.
- **LaTeX rules.** Use inline math (`$...$`) only for op math (RMSNorm, RoPE, attention). Display equations: `$$...$$`, or fenced ```` ```math ```` blocks if the equation contains `\texttt` with underscores or `\!`. **Never use `\text{...}` with underscores** — wrap such names with backticks in prose instead. Avoid LaTeX for type signatures or tensor shapes (use backticks). Example acceptable display: $$ \hat{x} = x \cdot \mathrm{rsqrt}\left(\mathrm{mean}(x^2) + \epsilon\right) \cdot \gamma $$
- **Tensor-lifetime vocabulary.** When discussing the qwen3 example, use the three-way **Parameter / Buffer / GraphInput** vocabulary defined in `examples/qwen3_embedding_0_6b/modules/__init__.py:5-21` exactly. Parameter = frozen weight, in state_dict. Buffer = runtime tensor, mutated in place, NOT in state_dict, address baked at first compile. GraphInput = a `forward` argument, wrapped by `wrap_input` on each call.
- **Symbol formatting in math.** RMSNorm parameter `gamma` is referred to as $\gamma$ in math, ``\gamma`` in code, "gamma" in prose. Be consistent within a paragraph.
- **Diagrams.** Mermaid for graph topology and control flow (`graph LR; A --> B`). ASCII art only for directory trees. At least one diagram per chapter from Ch4 onward (graph topology for `_call_graph`, the user→tracing→compiler chain for `tracing_contexts.md`, the dispatch resolution path for `functional_dispatch.md`).
- **Callouts.** Three standardized blockquote forms:
  - `> **Note:**` — clarifying aside.
  - `> **Warning:**` — footgun (e.g. `module.to(device)` does NOT move tensors; buffer `ttnn.Tensor` objects must not be reallocated after first compile).
  - `> **For contributors:**` — used in Chapters 1–4 only to forward-reference contributor material in Chapters 5–7. Always names the target chapter.
- **Navigation footer.** Every content file ends with the footer line:

  `_Previous: [<prev-title>](<rel-path>) · Next: [<next-title>](<rel-path>) · [Up](index.md)_`

  Every `index.md` file contains only the chapter title, a one-paragraph summary, and an ordered list of links to that chapter's content files — no other content.

## Chapter list

### Chapter 1 — Why blaze-nn and how it fits together
**Description:** Position blaze-nn against PyTorch, tt-blaze, and ttnn; establish the one-page mental model and get the reader running tests.
**Audience layer:** model-author
**Files in `ch1_why_blaze_nn/`:**
- `index.md` — chapter navigation
- `what_it_is.md` — positioning and scope
  - One-paragraph answer: "PyTorch-style API that traces `forward()` into a tt-blaze graph for Tenstorrent hardware."
  - Three-way picture: user model code → blaze-nn (tracing) → tt-blaze graph → tt-metal kernels (Mermaid diagram).
  - Quote the framework's own one-line description from `blaze_nn/__init__.py` and `README.md:3-12`.
  - Clarify what blaze-nn is **not**: not an autograd engine, not a kernel library, not a torch-compatible tensor type, no eager execution, no implicit device placement.
  - Distinguish blaze-nn (PyTorch-style author surface) from tt-blaze (graph/composition APIs: `blaze.fuse()`, `FusedProgram`).
- `ttnn_native_contract.md` — the core invariant: tensors are `ttnn.Tensor`
  - blaze-nn never imports torch at module scope — quote the docstring at `blaze_nn/__init__.py:5-7`.
  - Parameters hold `ttnn.Tensor`, forward args/returns are `ttnn.Tensor`, state_dict values are `ttnn.Tensor` — the framework treats them as opaque.
  - Show `Parameter.__init__` (`blaze_nn/parameter.py:16-26`): `_tensor: Any`; the framework never inspects type. Tests use `object()` sentinels.
  - Tease universal dispatch: any op tt-blaze registers becomes `F.<name>(...)` immediately. `> **For contributors:**` forward link to Chapter 6 for the `__getattr__` mechanism.
  - Forward link to `interop_at_the_boundary.md` (Ch2) for users who need a torch ↔ ttnn bridge.
- `getting_started.md` — install, environment, and the three test tiers
  - `pip install -e ".[dev]"` from `README.md`.
  - Three test tiers introduced upfront: (a) framework-only tests run without tt-blaze or a device, using `object()` sentinels; (b) dispatch-integration tests need tt-blaze importable, gated by `pytest.importorskip("blaze")`; (c) parity tests need tt-blaze + ttnn + a Tenstorrent device.
  - tt-blaze environment setup: sourcing `env.sh` or the explicit `TT_METAL_HOME` / `PYTHONPATH` recipe from `README.md:30-44`.
  - Sanity check: `import blaze_nn; print(dir(blaze_nn))`.
  - Note that `import blaze_nn` is safe without tt-blaze installed — all blaze / ttnn imports are deferred.

### Chapter 2 — Module, Parameter, and the device boundary
**Description:** The two foundational classes (`Module`, `Parameter`), the state-dict save/load contract, the `to(device)` semantics, and the torch ↔ ttnn `interop` helpers that model authors use to feed `load_state_dict`.
**Audience layer:** model-author
**Files in `ch2_module_and_parameter/`:**
- `index.md` — chapter navigation
- `parameter.md` — the trivial-looking class
  - One slot (`_tensor: Any`) and one name (`_name: str`); both filled in by `Module.__setattr__` and `load_state_dict` (`blaze_nn/parameter.py`).
  - `data` property is a passthrough setter/getter on `_tensor` (`blaze_nn/parameter.py:21-27`).
  - `__repr__` heuristic: uses `.shape` if present, falls back to `tensor=<repr>`, says `uninitialized` when `_tensor is None`.
  - Two population paths: `param.data = ttnn_tensor` (direct) and `module.load_state_dict({...})` (bulk).
  - What `Parameter()` is **not**: no `requires_grad`, no autograd, no shape declaration at construction; shape is whatever `ttnn.Tensor` the user assigns.
  - `_name` auto-population is used as the graph-input port name during tracing (forward link to Ch5 `tracing_contexts.md`).
- `module_attribute_protocol.md` — how `__setattr__` routes
  - `super().__init__()` populates `_parameters`, `_modules`, `_device_config`, `_compiled_cache`, `_state_loaded` (`modules/base.py:26-31`).
  - `__setattr__` routes `Parameter` → `_parameters[name]` and sets `param._name = name`; routes `Module` → `_modules[name]`; everything else falls through to `object.__setattr__`.
  - `__getattr__` mirrors PyTorch: look up `_parameters` first, then `_modules`, else raise `AttributeError`.
  - `__delattr__` symmetry; anchor to `tests/test_module.py:TestModuleAttributes`.
  - `forward()` is abstract — `NotImplementedError` on the base; subclasses must override.
  - `__call__` overview: checks the active tracing context (re-entry short-circuit at `base.py:71`); otherwise dispatches to `_call_graph` (default) or `_call_compose` (when `@blaze_nn.compose` is on `forward`). The `@compose` decorator is a one-bit flag (`forward_fn._blaze_nn_compose = True`, `blaze_nn/__init__.py:38-48`). Defer the actual graph-build pipeline to Ch5.
  - Pitfall: assigning a `list` or `dict` of `Module`s does NOT auto-register — that's what `ModuleList` / `ModuleDict` are for (forward link to Ch3).
- `traversal_and_state_dict.md` — `parameters`, `named_parameters`, `modules`, `named_modules`, `state_dict`, `load_state_dict`
  - Recursive walks that mirror PyTorch's dotted naming convention (`parent.child.weight`); own params first, then submodules (matches torch).
  - `state_dict()` returns `OrderedDict[str, ttnn.Tensor | None]` keyed by dotted parameter paths (`modules/base.py:190`); values can be `None` for uninitialized parameters.
  - `load_state_dict()` is strict, identity-preserving: splits keys on the first `.`, descends into child modules, sets `_state_loaded = True`. Unknown top-level keys raise `KeyError("Unexpected key ...")`; unknown module prefixes raise `KeyError("Unexpected module prefix ...")`. No `strict=False` escape hatch yet (note this).
  - Identity-preserving roundtrip pattern from `tests/test_state_dict.py:test_deep_model_roundtrip`: `m2.load_state_dict(m1.state_dict())` preserves object identity.
  - Values are written verbatim — **no dtype coercion, no device move, no layout conversion**. The user must construct `ttnn.Tensor` with the desired memory_config / shard_spec **before** load. Forward link to `interop_at_the_boundary.md`.
- `device_binding.md` — what `module.to(device)` does and doesn't
  - Wraps the device handle in `DeviceConfig` and stashes it on `_device_config`; recurses into `_modules` (`modules/base.py:236-247`).
  - `> **Warning:**` Does NOT move parameters, does NOT change layout, does NOT promote dtypes. Quote the docstring.
  - Calling `forward` before `to(device)` raises `RuntimeError("...has no device. Call module.to(device) first.")` from `_resolve_device_config`.
  - `DeviceConfig` from a user's perspective: holds the device handle and lazily exposes `GridConfig` from `blaze.role_engine`. `> **For contributors:**` internals in Ch5.
- `interop_at_the_boundary.md` — torch ↔ ttnn helpers for `load_state_dict`
  - `blaze_nn.interop.to_device_tensor(torch_tensor, device, memory_config=None)` and `blaze_nn.interop.to_torch(device_tensor)`: the only sanctioned torch boundary; defaults are `bfloat16` + `TILE_LAYOUT`.
  - Lazy `import ttnn` inside each function (`blaze_nn/interop/__init__.py`) — keeps `import blaze_nn.interop` cheap and tt-blaze-free.
  - When to use: data loading, building the `ttnn.Tensor` dict for `load_state_dict`, golden comparisons in parity tests.
  - When NOT to use: inside `forward()` — pulling to torch breaks tracing.
  - `> **For contributors:**` the rule "never call `blaze_nn.interop` from inside `blaze_nn/` itself" is restated in Ch7 `contributing_checklist.md` as a hard anti-pattern.

### Chapter 3 — Containers, OpModule, and pre-built ops
**Description:** The three non-callable / callable containers, the workhorse `OpModule` (both forms), and the pre-built `Linear` / `RMSNorm` modules every model author reaches for first.
**Audience layer:** model-author
**Files in `ch3_containers_and_opmodule/`:**
- `index.md` — chapter navigation
- `sequential.md` — the one callable container
  - `Sequential(*modules)`: children registered by `str(idx)` keys, so `state_dict` produces `0.weight`, `1.weight`, ... (state-dict keys nest as `layers.0.weight` for an outer module).
  - `__call__` walks children in order, chaining `x`.
  - `__len__`, `__iter__`, `__getitem__(int)` — anchor to `tests/test_containers.py:TestSequential`.
- `modulelist_and_moduledict.md` — the non-callable containers
  - Both inherit from `_NotCallableContainer`: calling them raises `RuntimeError("not callable directly. Iterate ..." / "Access ... by key ...")` (see the two `_usage_hint`s in `blaze_nn/containers.py`).
  - `ModuleList(modules)` + `append(module)` (returns self for chaining).
  - `ModuleDict(mapping)` mirrors dict: `__setitem__`, `__contains__`, `keys/values/items`. Anchor to `tests/test_containers.py:TestModuleDict`.
  - Idiom: hold layers in `ModuleList`, iterate inside `forward()` — e.g. `Qwen3EmbeddingModel.forward` iterating `self.layers`.
- `opmodule_no_subclass.md` — `OpModule(op=..., params=..., **kwargs)`
  - Constructor records `_op_name`, `_param_slots`, `_op_kwargs`; auto-creates empty `Parameter()` for every name in `params` (`modules/base.py:332`).
  - Default `forward()`: `F.<op>(*args, *params_in_declaration_order, **{op_kwargs, **call_kwargs})`. Call-time `**kwargs` override construction-time `_op_kwargs`.
  - Lifecycle: instantiate → `load_state_dict({"gamma": ttnn_tensor})` → `m.to(device)` → `m(x)`.
  - Canonical example: `rmsnorm = OpModule(op="rmsnorm", params=("gamma",), epsilon=1e-5)` from `tests/test_op_module.py`.
  - Real-world use: `OpModule(op="residual_add")` instances in `Qwen3DecoderLayer` and `Qwen3Attention` — quick wrappers with no class needed.
- `opmodule_subclass.md` — declaring `op` and `params` as class attrs
  - When to subclass: custom `forward()` (extra kwargs, buffer-address plumbing) or class-level docs.
  - The pattern: class attrs `op = "<tt-blaze-op>"` and `params = ("a", "b")`; default constructor walks `params` to create the `Parameter` slots.
  - Reference `blaze_nn/ops/rmsnorm/op.py` as the canonical small example (no custom forward); reference `blaze_nn/modules/linear.py` as the canonical complex example (custom forward + `define_fused_op`).
  - `_torch_init_specs` (per-subclass shape + tile dims) and `init_torch_params` for random init on device — show the auto-init branch at `modules/base.py:425-428` and the bf16/TILE_LAYOUT defaults at `modules/base.py:460-501`. Pitfall: `init_torch_params` requires `.to(device)` first; the helper imports torch lazily so the framework stays torch-free until used.
- `output_tensors.md` — user-allocated outputs in plain English
  - The rule, scoped to user-level: **some modules require `set_output_tensor(t)` (or `set_output_tensors(name1=t1, ...)`) before `forward` is called**. `Linear` is the canonical case; SDPA decode after the qwen3 monkey-patch is the other.
  - How to know which: check the module class docs, or look for a constructor that says "caller must allocate output." Forgetting raises `RuntimeError("has unset required output tensor(s): ...")` before forward (`modules/base.py:417-423`).
  - One concrete example: `lin = Linear(D_in, D_out); out = ttnn.allocate(...); lin.set_output_tensor(out); lin(x)`.
  - Brief on user args: any attribute prefixed `_ua_` becomes a key in `user_args` passed to the compiler (one-line example pointing at `FusedQKV._ua_blackhole_cores = "64x8"` in qwen3 — `examples/qwen3_embedding_0_6b/modules/qkv_proj.py:29`).
  - `> **For contributors:**` the full `user_allocated_outputs` ↔ `_lookup_user_allocated_outputs` ↔ `define_fused_op` chain is in Ch6 `caller_allocated_outputs_internals.md`.
- `prebuilt_modules.md` — `blaze_nn.Linear` and `blaze_nn.ops.RMSNorm`
  - `Linear(in_features, out_features, bias=False)`: torch.nn.Linear-shaped constructor; `bias=True` raises; weight tile `[32, 32]`; uses the fused `blaze_nn_linear` op (`mcast → matmul → gather`); requires caller-allocated output via `set_output_tensor`.
  - `RMSNorm(normalized_shape, eps=1e-6)`: torch.nn.RMSNorm-shaped constructor; gamma tile `[1, 32]`; uses the tt-blaze `rmsnorm` op directly; no caller-allocated output.
  - The `blaze_nn/ops/` convention: one subpackage per op-with-an-init-shape, mirroring `blaze.ops.*`. `> **For contributors:**` the extension recipe is Ch7 `add_an_op_wrapper.md`.
  - End-to-end snippet following `tests/test_pytorch_parity.py:test_linear_pipeline_matches_torch`: build `Linear`, `set_output_tensor`, `load_state_dict`, `to(device)`, run, compare via `comp_pcc`.
  - Math sidebar: $$ \hat{x} = x \cdot \mathrm{rsqrt}\left(\mathrm{mean}(x^2) + \epsilon\right) \cdot \gamma $$ for RMSNorm, with prose explaining the per-row mean-square normalization.

### Chapter 4 — Authoring models: the Qwen3 walkthrough
**Description:** Walk the only end-to-end model in the repo to show how the public API composes — including the messy reality of caller-allocated buffers, buffer-address kwargs, host-side hops, and orchestrator `__call__` overrides.
**Audience layer:** model-author
**Files in `ch4_qwen3_walkthrough/`:**
- `index.md` — chapter navigation
- `layout_and_weight_loader.md` — directory tour and the HF → ttnn bridge
  - Walk `examples/qwen3_embedding_0_6b/`: `config.py`, `weight_loader.py`, `modules/`, `tests/`, `demo/`.
  - `Qwen3EmbeddingConfig`: frozen dataclass, derived properties (`effective_n_layers`, `n_kv_groups`, `qkv_out_dim`).
  - Weight loader pipeline: HF state_dict → torch tensors → key remap (fused `q_proj`/`k_proj`/`v_proj` → `qkv.weight`) → `ttnn.Tensor`s with role-specific memory configs.
  - The blaze-nn key set (`_build_blaze_nn_keys`): `embed_tokens`, per-layer `input_layernorm.gamma`, `self_attn.qkv.weight`, `self_attn.q_norm.gamma`, ..., `rope.cos/sin/trans_mat`, `norm.gamma`.
  - Per-role memory-config policy (`_wsharded_linear_weight_mc`, `_gamma_mc_for_width`): rows = qkv (8x8), o_proj (4x8), mlp (4x8); fall back to `DRAM_MEMORY_CONFIG` when shapes don't divide.
  - Note where torch enters the picture (and only there): the weight loader. Inside `modules/` torch only appears in `__init__`/buffer-setup helpers, never in `forward`.
- `tensor_lifetimes.md` — the three lifetimes (Parameter / Buffer / GraphInput)
  - Quote the lifetimes contract from `examples/qwen3_embedding_0_6b/modules/__init__.py:5-21` verbatim, then explain each lifetime with one Qwen3 example.
  - Parameter: frozen weight, lives in `state_dict`. Buffer: runtime tensor, mutated in place, NOT in state_dict, address baked at first compile (`> **Warning:**` must not be reallocated after first compile). GraphInput: a `forward` argument, wrapped by `wrap_input` each call.
  - Why some Parameters become buffer-address ints inside `forward()` rather than graph inputs: the op consumes them via DRAM read in the kernel; reading the address means baking it into the compiled program's CT args.
  - Mutation idiom: Buffer's `ttnn.Tensor` object stays put; mutate in place via `ttnn.copy_host_to_device_tensor`.
- `composing_submodules.md` — building decoders from primitives
  - `TokenEmbedding` (subclassed `OpModule(op="embedding")`): reads `weight.buffer_address()` inside `forward()` and passes it as `weight_buffer_address` kwarg (`modules/token_embedding.py:25`) — the embedding op consumes the weight by DRAM address, not as a graph port.
  - `FusedQKV` (`Module`) wraps `blaze_nn.Linear` and exposes a single `weight` key via `load_state_dict` remap (`weight` → `linear.weight`). Sets `_ua_blackhole_cores = "64x8"` (`modules/qkv_proj.py:29`).
  - `Qwen3MLP` (plain `Module`): manual three-step `F.matmul` chain with `F.gated_reduce(gate, up, activation="silu")` between them.
  - `RoPE` (`OpModule(op="rope")`): three Parameters — `cos`, `sin`, `trans_mat`. `trans_mat` flows as a graph input; `cos` and `sin` flow as buffer-address kwargs. `set_position_ids` binds an existing position-ids tensor by reference.
  - `Qwen3DecoderLayer` composition: norm → attn → norm → mlp + residuals, mixing plain modules with `OpModule(op="residual_add")` inline.
  - `q_norm` / `k_norm` use `blaze_nn.ops.RMSNorm`; `o_proj` uses `blaze_nn.Linear` with its own `_ua_blackhole_cores`.
- `orchestrator_pattern.md` — when forward() is plain Python (TWO mechanisms)
  - **Mechanism A — orchestrator `__call__` override.** `Qwen3Attention` (`modules/attention.py:90`), `Qwen3DecoderLayer` (`modules/decoder_layer.py:32`), and `Qwen3EmbeddingModel` (`modules/model.py:67`) each override `__call__` with the two-liner `def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)`. This bypasses `Module.__call__`'s tracing machinery entirely, so their `forward` runs as plain Python at the top level — needed because host-side hops (`nlp_create_qkv_heads_decode`, `ttnn.kv_cache.update_cache_for_token_`, `sharded_to_interleaved`) can't live inside a single tt-blaze graph.
  - **Mechanism B — active-context short-circuit.** Inside an orchestrator's `forward`, every nested module call (e.g. `self.input_layernorm(h)`, `self.qkv(x)`) hits `Module.__call__` (`base.py:68`), and the active-context check at `base.py:71` short-circuits to `forward` directly. But since the orchestrator never opened a tracing context, this short-circuit is irrelevant for orchestrator children — each child sub-module opens its own tracing context, compiles its own graph, and runs. The short-circuit only matters when an orchestrator's child is itself a non-orchestrator that *would* re-enter a tracing context if invoked inside a parent compile.
  - Show `Qwen3EmbeddingModel.forward` (the layer loop, 10-ish lines) and explicitly note: every `self.layers[i](h, ...)` is a separate compile that gets cached on the child's `_compiled_cache` after the first call.
  - `> **For contributors:**` Ch5 `module_call_path.md` shows the full re-entry semantics and `_compiled_cache` mechanics.
- `buffers_and_address_baking.md` — runtime state without state_dict membership
  - Setup hooks on `Qwen3EmbeddingModel`: `init_position_ids` (allocates buffer), `set_position_ids` (binds a tensor by reference), `init_kv_caches`, `init_attn_out_buffers`, `init_qkv_buffers`, `init_o_proj_buffers`, `make_input_ids_tensor`. None appear in `state_dict`. Naming: `init_*` allocates; `set_*` binds.
  - Why: Parameters are frozen weights, but KV caches, position ids, and per-layer SDPA output buffers are runtime tensors that mutate between forwards.
  - The `_bridge_kv_for_cache_update` and `_bridge_q_for_sdpa` host hops in `Qwen3Attention` (`modules/attention.py:93,106`) — head-split ops emit sharded shapes; `update_cache_for_token_` expects interleaved `(B, n_kv, 1, head_dim)`; bridges convert between the two.
  - The Blackhole P150 patches: `_blaze_nn_linear_patch.py` (idempotent monkey-patch swapping `Linear.compose` cores for `8x8` / `4x8` subgrids using `user_args["blackhole_cores"]`) and `_register_sdpa_decode_user_alloc` (declares `SDPADecode.user_allocated_outputs = ("output",)` so `OpModule.set_output_tensor` routes through the standard mechanism). Position: "model code can monkey-patch tt-blaze ops at startup when a hardware target needs a non-default compose; this is a last resort, not a recommended general pattern."
  - Demo entry: `demo/encode.py`. Explicit note: prefill (encode) is deferred to Phase B; the supported path is decode-shaped per-token forward.
  - Recap of which tests cover which slice: `tests/test_l0_*.py` (config/keys/RoPE math), `tests/test_l1_*.py` (per-module parity), `tests/test_layer_parity.py`, `tests/test_e2e_parity.py`. (Reverse-indexed in Ch7 `testing_strategy.md`.)

### Chapter 5 — Tracing internals: from `Module.__call__` to `program.run()`
**Description:** Open the user → contributor boundary. Walk `_call_graph`, `_call_compose`, the `TracingContext` hierarchy, `TensorProxy`, and the active-context re-entry contract.
**Audience layer:** contributor
**Files in `ch5_tracing_internals/`:**
- `index.md` — chapter navigation
- `module_call_path.md` — what happens between `model(x)` and `program.run()`
  - Annotated flow diagram (Mermaid): user call → `Module.__call__` → active-context check (`base.py:71`) → either return `forward(...)` (re-entry) or open new context → `wrap_input` each arg → `_bind_parameters_to_context` → run `forward` → exit context → compile (graph mode) / run (compose mode) → return `ttnn.Tensor`.
  - Step-by-step trace through `_call_graph` (`base.py:86-122`) line by line: `GraphTracingContext`, `wrap_input` for positional and keyword args, `_bind_parameters_to_context`, run user `forward`, exit context, fetch `graph`, build `tensors` dict (the port-alias dual-key at `base.py:107` so the compiler can resolve by port name or `ExternalTensor` name), `BlazeCompiler(dc.device).compile(graph, tensors, output_tensor, user_args).run()`.
  - Step-by-step trace through `_call_compose` (`base.py:126-`): `ComposeTracingContext`, instantiate `FusedProgram(kernel=None, device=...)`, same wrap/bind/run/exit pattern, then `ctx._fused_program.run()`.
  - `_compiled_cache` field on `Module`: currently unused — flag as a future-extension hook.
  - `_collect_user_args` (defaults `{}`) and the `OpModule` override that harvests every `_ua_*` attribute. Walk the full path: `OpModule._ua_x = "v"` → `_collect_user_args` returns `{"x": "v"}` → `BlazeCompiler.compile(..., user_args=...)`.
  - `_get_output_tensor` and the `OpModule` override that returns caller-allocated outputs (or aliases `inputs[0]` if none declared).
- `tracing_contexts.md` — `TracingContext`, `GraphTracingContext`, `ComposeTracingContext`
  - Module-level `_active_context` singleton and the `_get`/`_set`/`_clear` helpers in `blaze_nn/_tracing.py`. State the single-threaded assumption explicitly (mirrors blaze's own).
  - Walk the base `TracingContext`: `register_input`, `_tensor_bindings`, `_input_counter`, `_op_counter`, `_next_input_name`, `_next_prefix`, `_unwrap_args`, `_resolve_grid` (`_tracing.py:37-90`).
  - `_resolve_grid`: explicit `_grid` kwarg wins; else if op is in `uses_matmul_cores` set, use `device_config.matmul_cores`; else use `device_config.all_cores`. `> **For contributors:**` registry flags are covered in Ch6 `registry.md`.
  - `GraphTracingContext`: `__enter__` opens `blaze.fuse()` and sets active; `__exit__` clears and closes. `wrap_input` returns an `ExternalTensor(name)` proxy and registers the backing `ttnn.Tensor` in `_tensor_bindings`. `wrap_parameter` keys by the Parameter's attribute name. `dispatch` resolves the op handle via `getattr(blaze, op_name)` (raises `ValueError("Unknown blaze op")` if missing), unwraps args, resolves grid, injects `sender` kwarg if `needs_sender_core`, auto-assigns `ct_prefix` if absent, calls the op, wraps result as `TensorProxy`.
  - `ComposeTracingContext`: no `blaze.fuse()`; constructs a `FusedProgram(kernel=None, device=...)` ahead of time. `dispatch` looks up `BlazeOp._class_registry[op_name]` and calls `op_cls.emit(self._fused_program, ...)`. `wrap_input` returns the device tensor itself (not an `ExternalTensor`); `wrap_parameter` requires `_tensor is not None` and raises a clear error otherwise.
  - When graph vs. compose: graph is the default and what every qwen3 sub-module uses; compose is reserved for pre-fused programs where topology is fixed. **Known gap**: no test exercises compose mode end-to-end (verified via `grep -rn compose tests/`); flag as a contributor todo.
- `tensor_proxy.md` — the opaque handle
  - `TensorProxy(__slots__=("_inner","_name"))` wraps the backend object: `ExternalTensor` / `FusionResult` for graph mode, raw ttnn tensor for compose mode (`blaze_nn/_tensor_proxy.py` docstring).
  - Why `__slots__` matters here: tracing creates many short-lived proxies; the slot saves both memory and the per-instance dict.
  - Users never inspect `_inner`; ops unwrap it via `TracingContext._unwrap_args` (`_tracing.py:70`) — Proxy → `_inner`, Parameter → wrap-then-unwrap.
  - Connect `_name` to graph-input port names (`"__input_0"`, `"weight"`, op-prefix names like `"matmul_3"`).
  - Why users must not introspect or construct `TensorProxy` directly: it is the lingua franca between `F.*` ops and the active context; framework invariants depend on its opacity.

### Chapter 6 — Op dispatch, the registry, and caller-allocated outputs
**Description:** The other half of the contributor view — `_dispatch` + `__getattr__`, `_registry.py` aliases and placement hints, and the full `user_allocated_outputs` ↔ `define_fused_op` chain.
**Audience layer:** contributor
**Files in `ch6_dispatch_and_registry/`:**
- `index.md` — chapter navigation
- `functional_dispatch.md` — `_dispatch` and the lazy `__getattr__`
  - Mermaid diagram: `F.<op>(*args)` → `__getattr__` (first call only) → closure → `_dispatch(op_name, *args)` → active context check → `resolve_alias` → wrap `Parameter` args via `ctx.wrap_parameter` → `ctx.dispatch` → backend op handle → `TensorProxy` result.
  - Walk `_dispatch` (`functional.py:24`): fetch active context, run `resolve_alias`, wrap `Parameter` arguments via `ctx.wrap_parameter`, delegate to `ctx.dispatch`. Outside an active context raises `RuntimeError("... no active tracing context")`.
  - Walk module-level `__getattr__` (`functional.py:63`): underscore names re-raise `AttributeError`; everything else builds a `_op` closure that forwards to `_dispatch`, sets `__name__` / `__qualname__` / `__doc__`, then caches itself into `globals()` so subsequent lookups skip `__getattr__`.
  - Walk `__dir__` (`functional.py:91`): static names plus every key in `BlazeOp._class_registry`, with a graceful fallback when tt-blaze is not importable.
  - The explicit `linear` / `sliced_matmul` shims: `linear(bias=...)` raises `NotImplementedError`; `sliced_matmul` defaults `branch="gate"`. Rule for adding more: only when you need non-trivial arg handling or a friendlier name.
  - Anchor to `tests/test_functional.py:TestDynamicDispatch` (closure correctness) and `tests/test_dispatch_integration.py` (produces real `BlazeGraph` nodes).
- `registry.md` — `_registry.py` aliases and placement hints
  - Walk `OpInfo` (frozen dataclass) and the three fields: `backend` (alias target), `uses_matmul_cores`, `needs_sender_core`.
  - The three-flag semantics: `backend` set on alias entries (the blaze_nn-facing name); placement flags set on backend names and read AFTER alias resolution (verify against `blaze_nn/_registry.py:22-32`).
  - Current four backend-flag entries (`matmul`, `kn_sliced_matmul`, `residual_add` are matmul-grid; `mcast` needs sender) and two alias entries (`linear → matmul`, `sliced_matmul → kn_sliced_matmul`).
  - The three public helpers: `resolve_alias`, `uses_matmul_cores`, `needs_sender_core`.
  - Decision tree for adding a new op:
    1. Friendlier name needed? Add an `OpInfo(backend=...)` alias entry.
    2. Op should run on the matmul subgrid? Set `uses_matmul_cores=True` on the backend entry.
    3. Op needs `sender` kwarg auto-injected? Set `needs_sender_core=True`.
    4. None of the above? No registry entry needed at all — universal dispatch handles it.
  - Show how `_resolve_grid` (`_tracing.py:82`) consumes these flags.
- `caller_allocated_outputs_internals.md` — `OpModule` ↔ `BlazeOp.user_allocated_outputs`
  - `_lookup_user_allocated_outputs(op_name)`: opens `BlazeOp._class_registry[op_name]`, reads `user_allocated_outputs`, returns the tuple (empty if blaze missing or op missing).
  - How `OpModule.__init__` consumes it: declares `_required_output_names`, exposes `set_output_tensor` / `set_output_tensors`.
  - `OpModule._get_output_tensor` override: returns the registered output tensor(s) or aliases `inputs[0]` if none declared. Pre-forward check that raises `RuntimeError("has unset required output tensor(s): ...")` (`modules/base.py:417-423`).
  - `define_fused_op` hook: subclass-level method called once on first instantiation to register a synthesized `FusedOp` (e.g. `BlazeNNLinear` in `blaze_nn/modules/linear.py:23-58`); guarded by the `_fused_op_defined` class flag so registration is idempotent across imports.
  - Pitfall: monkey-patching `user_allocated_outputs` on a third-party op (qwen3's `_register_sdpa_decode_user_alloc`) is supported but must be idempotent — repeated imports must not double-register.
  - Pitfall: changing the tuple's contents after instantiation is undefined behavior; the OpModule reads it once at construction.

### Chapter 7 — Extending blaze-nn
**Description:** A capstone with concrete extension recipes plus the test taxonomy contributors will reach for repeatedly.
**Audience layer:** contributor
**Files in `ch7_extending/`:**
- `index.md` — chapter navigation
- `add_an_op_wrapper.md` — the `blaze_nn/ops/<op>/` convention
  - Walk `blaze_nn/ops/__init__.py` and `blaze_nn/ops/rmsnorm/{__init__.py, op.py}` as the canonical pattern.
  - Checklist: subclass `OpModule`; set class-level `op` and `params`; mirror the torch op's constructor signature (e.g. `RMSNorm(normalized_shape, eps=1e-6)`); optionally override `_torch_init_specs` for `init_torch_params` support; optionally implement `define_fused_op` if the op needs synthesis.
  - When to put a module in `ops/` vs. `modules/`: `modules/` is reserved for fused multi-op modules (`Linear` = mcast → matmul → gather); `ops/` is one op per subpackage.
  - Re-anchor to where each piece is wired: registry (`_registry.py` only if you need an alias or placement hint), dispatch (`functional.py` only if you need a non-trivial arg shim).
  - Tests to add: framework-only `OpModule` tests against `object()` sentinels (Ch7 `testing_strategy.md`), plus a dispatch-integration test gated by `pytest.importorskip("blaze")`.
- `add_a_fused_op.md` — when the op does not exist upstream
  - Walk `blaze_nn/modules/linear.py:Linear.define_fused_op` — the lazy `BlazeOp` synthesis hook called at most once per subclass, before `_lookup_user_allocated_outputs` consults the registry.
  - The parts: `class BlazeNNLinear(FusedOp)` with `name`, `math_fidelity`, `user_allocated_outputs`, declared `Input` / `Output` ports, and a `compose(cls, f, tensors, output, user_args)` classmethod.
  - Registration sequence: `BlazeNNLinear.register()` then `setattr(blaze, name, blaze._OpHandle(BlazeNNLinear))`.
  - When to do this in blaze-nn at all: the op is composed of upstream tt-blaze primitives but does not yet have its own registered fused op upstream.
  - Idempotence rule: guard via a class-level `_fused_op_defined` flag so re-imports don't double-register.
- `extending_containers_and_modules.md` — beyond the built-ins
  - The two `_IndexedContainer` / `_NotCallableContainer` mixins in `blaze_nn/containers.py`: how to compose a new container without rebuilding traversal.
  - When to introduce a custom `Module` subclass that overrides `__call__` (the qwen3 orchestrator pattern, Ch4) versus extending `OpModule`.
  - The `_collect_user_args` override pattern (qwen3 `FusedQKV`, `Qwen3MLP`) and where the kwargs end up (`BlazeCompiler.compile(..., user_args=...)`).
- `testing_strategy.md` — the test taxonomy (reverse index)
  - Framework-only (no tt-blaze, no device, uses `object()` sentinels): `test_module.py`, `test_parameter.py`, `test_containers.py`, `test_state_dict.py`, `test_op_module.py`, `test_functional.py`.
  - Torch-only reference sanity (`torch_reference.py` helpers): `test_integration.py`.
  - tt-blaze importable (no device, `pytest.importorskip("blaze")`): `test_dispatch_integration.py`.
  - tt-blaze + ttnn + Tenstorrent device (PCC comparison via `torch_reference.comp_pcc`, threshold 0.99): `test_pytorch_parity.py`.
  - qwen3 example test slices: `tests/test_l0_*.py` (config/keys/RoPE math), `tests/test_l1_*.py` (per-module parity), `tests/test_layer_parity.py`, `tests/test_e2e_parity.py`.
  - Recipe for a new framework feature: framework-only `object()`-sentinel test first → dispatch-integration test → optionally a parity test gated by device availability.
  - Reverse-index: each test bucket links back to the chapter section whose claims it backs.
- `contributing_checklist.md` — concrete extension recipes and anti-patterns
  - Adding a new alias: edit `_REGISTRY`, add a `test_functional` test, document in Ch6.
  - Adding a fused op: subclass `OpModule`, override `define_fused_op` returning a `FusedOp` subclass; add a `tests/test_dispatch_integration.py` case asserting node name appears in the graph.
  - Adding a placement hint: choose `uses_matmul_cores` or `needs_sender_core` in `_REGISTRY`; verify with a graph-construction test that the right `grid=` and `sender=` kwargs are passed.
  - Anti-patterns (hard rules): never `import torch` at module scope inside `blaze_nn/` (except in `interop/` and `init_torch_params`); never call `blaze_nn.interop` from inside `blaze_nn/` itself; never bypass `F` to call `blaze.matmul` directly inside `forward`; never reuse a Buffer's `ttnn.Tensor` object across `to(device)` re-binds (its address is baked in by the compiler at first compile).
  - Known gap: compose mode has no end-to-end test; contributors taking on a new compose-mode backend should add one.

## Cross-chapter dependencies

- **Chapter 1 → all.** The "mental model" three-way picture in `what_it_is.md` is referenced verbatim by Ch3 (`opmodule_no_subclass.md` lifecycle), Ch4 (`composing_submodules.md`), and Ch5 (`module_call_path.md`).
- **Chapter 2 → Chapters 3, 4, 5.** `Parameter._name` autopopulation feeds the graph-input port-name story in Ch5 `tracing_contexts.md`. `module.to(device)` is restated wherever buffers appear (especially Ch4 `tensor_lifetimes.md`). The `interop_at_the_boundary.md` rule (no interop inside `forward`) is recapped in Ch7 `contributing_checklist.md` as an anti-pattern.
- **Chapter 2 → Chapter 5 (single forward link).** `module_attribute_protocol.md` mentions `_call_graph` and `_call_compose` once with an explicit `> **For contributors:**` callout — no hard prerequisite from earlier chapters back-references this.
- **Chapter 3 → Chapter 4.** Every qwen3 module is one of (`Module`, `OpModule(op=..., params=...)`, `OpModule` subclass) or wraps `blaze_nn.Linear` / `blaze_nn.ops.RMSNorm`. Ch4 deliberately introduces no new framework concepts.
- **Chapter 3 `output_tensors.md` → Chapter 6 `caller_allocated_outputs_internals.md`.** The user-level introduction in Ch3 is intentionally thin (one example, one rule); Ch6 carries the full mechanism. The forward link is named in Ch3 via `> **For contributors:**`.
- **Chapter 4 → Chapters 5, 6 (forward references).** Each "this works because the framework does X internally" claim in Ch4 carries a callout pointing at the Ch5/Ch6 section. Concretely: `wrap_input` for `cur_pos_tensor` (→ Ch5 `tracing_contexts.md`); `_ua_*` reaching the compiler (→ Ch5 `module_call_path.md` `_collect_user_args` walk); the `linear → matmul` alias (→ Ch6 `registry.md`).
- **Chapter 4 `orchestrator_pattern.md` ↔ Chapter 5 `module_call_path.md`.** The two mechanisms (orchestrator `__call__` override vs. active-context short-circuit) are introduced at user level in Ch4 and carried to internals depth in Ch5. Ch4 names the user-visible signature; Ch5 walks the `base.py:68-71` lines.
- **Chapter 5 → Chapter 6.** `TracingContext._resolve_grid` and `GraphTracingContext.dispatch` consume the `_registry` helpers (`uses_matmul_cores`, `needs_sender_core`). Ch5 links forward; Ch6 owns the explanation.
- **Chapter 6 → Chapter 7.** The "add a new op" workflow in Ch7 `add_an_op_wrapper.md` references the registry decision-tree in Ch6 `registry.md` verbatim and the dispatch shim rule from Ch6 `functional_dispatch.md`. Ch7 `add_a_fused_op.md` references Ch6 `caller_allocated_outputs_internals.md` for the `define_fused_op` ↔ `_lookup_user_allocated_outputs` interplay.
- **Chapter 7 `testing_strategy.md` ← every prior chapter.** Calls back to every test file cited as evidence in earlier chapters — readers can use it as a reverse index from chapter section to test coverage. Ch1 `getting_started.md` introduces the three test tiers; Ch7 `testing_strategy.md` enumerates which file lives in which tier and which chapter section it backs.
