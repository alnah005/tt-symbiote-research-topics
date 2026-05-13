# Buffers and address baking

This section catalogs every Buffer in the qwen3 port, names the `init_*` vs `set_*` hook pattern, walks the two host-side bridges in `Qwen3Attention`, and documents the Blackhole P150 monkey-patches that the demo applies at startup. Each piece is here because it is runtime state that does *not* live in `state_dict` but does flow into the compiled programs as either a GraphInput or a baked address.

## The `init_*` / `set_*` naming convention

The port uses two related verbs deliberately:

- **`init_<name>(device, ...)` allocates** a new `ttnn.Tensor` for the Buffer slot. Called once at setup, before the first compile.
- **`set_<name>(tensor)` binds** an existing `ttnn.Tensor` (allocated elsewhere) by reference. Called by tests that supply pre-built tensors.

The model exposes both for several Buffers so that tests can inject fixtures while production code can let the model allocate. The hooks on `Qwen3EmbeddingModel` (`examples/qwen3_embedding_0_6b/modules/model.py:55-322`) are:

| Hook | Allocates | Binds | Location |
|---|---|---|---|
| `init_position_ids(device, batch_size=1)` | yes | — | `model.py:78-110` |
| `set_position_ids(position_ids_tensor)` | — | yes | `model.py:70-72` |
| `init_kv_caches(device)` or `init_kv_caches(k_caches, v_caches)` | yes (1-arg) | yes (2-arg) | `model.py:112-160` |
| `init_attn_out_buffers(device, max_cph=0)` or `init_attn_out_buffers(tensor_list)` | yes / yes | yes | `model.py:162-205` |
| `init_qkv_buffers(device)` or `init_qkv_buffers(tensor_list)` | yes / yes | yes | `model.py:207-240` |
| `init_o_proj_buffers(device)` or `init_o_proj_buffers(tensor_list)` | yes / yes | yes | `model.py:242-276` |
| `make_input_ids_tensor(device, token_ids)` | yes (returns) | — | `model.py:278-305` |

The `init_*` methods are idempotent on a non-`None` check (e.g. `model.py:79-80`: `if self.position_ids is not None: return`), so calling them twice is safe.

> **Note:** `make_input_ids_tensor` is the odd row out — it does **not** bind to a Buffer slot on the model. It allocates a fresh `ttnn.Tensor` on a single core at `(0, 0)` and returns it for the caller to pass as the next `input_ids` GraphInput. The single-core layout is a Blackhole P150 constraint (item B2 in the v4 notes). Use it once per token; do not stash the result on the model.

### `init_position_ids` vs `set_position_ids` — read both

`init_position_ids` (`model.py:78-110`) allocates a HEIGHT_SHARDED L1 int32 tensor sized to the full `(grid.x, grid.y)` core grid and assigns it to `self.position_ids`, then forwards the same tensor reference to `self.rope.set_position_ids(...)` (`model.py:110`). This is the production path.

`set_position_ids` (`model.py:70-72`) accepts a pre-built tensor and binds it to both `self.position_ids` and `self.rope.position_ids` (`model.py:71-72`). This is the test path — L1 tests pass in a fixture-built tensor of the exact shape they want.

```python
# modules/model.py
def set_position_ids(self, position_ids_tensor: Any) -> None:
    self.position_ids = position_ids_tensor
    self.rope.set_position_ids(position_ids_tensor)

def init_position_ids(self, device: Any, batch_size: int = 1) -> None:
    if self.position_ids is not None:
        return
    # ... allocate a HEIGHT_SHARDED int32 ttnn.Tensor on the full (grid.x, grid.y) ...
    self.position_ids = ttnn.from_torch(host_pos, ...)
    self.rope.set_position_ids(self.position_ids)
```

The pair is the canonical example of the convention: `init_*` covers the production allocation; `set_*` covers the by-reference binding. The two Buffers (model and RoPE) hold the same `ttnn.Tensor` object — that aliasing is intentional. Both end with the same `self.rope.set_position_ids(...)` call so the same Python `ttnn.Tensor` object is reachable from both the model's `.position_ids` and the RoPE submodule's `.position_ids`. RoPE bakes the buffer's address into its CT args at first compile; SDPA reads the same tensor as `cur_pos_tensor` on each call (as a GraphInput). Both views read the same DRAM allocation, so an in-place update via `ttnn.copy_host_to_device_tensor` advances the position for both.

> **Warning:** None of these `init_*` / `set_*` methods are safe to call after the first `program.run()` for any module that bakes the buffer's address into CT args (RoPE, embedding, kv-cache update). Doing so reallocates the underlying buffer, the address bakes go stale, and subsequent runs read invalid memory. The whole point of the in-place mutation idiom (`ttnn.copy_host_to_device_tensor`) is to avoid ever needing to. The `if self.position_ids is not None: return` guard in `init_position_ids` enforces this on the model side.

## The KV cache and the SDPA output buffer

`init_kv_caches(device)` (`model.py:112-160`) walks every layer and allocates `k_cache` and `v_cache` as DRAM-interleaved bf16 tensors of shape `(1, n_kv_heads, max_seq_len, head_dim)`. Each layer's `Qwen3Attention.init_kv_cache(k, v)` (`modules/attention.py:58-60`) binds them to `self.k_cache` / `self.v_cache`. The caches are mutated in place by `ttnn.kv_cache.update_cache_for_token_` once per layer per forward.

`init_attn_out_buffers(device)` (`model.py:162-205`) allocates one SDPA output tensor per layer using `SDPADecode.output_memory_config(...)` and binds it via `set_attn_out_tensor(t)` (`modules/attention.py:69-71`). That setter both stores the tensor on `Qwen3Attention.attn_out_tensor` and routes it into the inner SDPA `OpModule` via `self.sdpa.set_output_tensor(tensor)` — SDPA decode requires a caller-allocated output (see the monkey-patch below).

`init_qkv_buffers(device)` (`model.py:222-240`) and `init_o_proj_buffers(device)` (`model.py:258-276`) do the same for the QKV and o_proj Linear outputs. Each fans out to `set_qkv_out_tensor` / `set_o_proj_out_tensor` on each layer's attention, which in turn calls `set_output_tensor` on the inner `FusedQKV` / `Linear`.

## The two host-side bridges in `Qwen3Attention`

Two shape contracts disagree on the attention path: `nlp_create_qkv_heads_decode` emits HEIGHT_SHARDED `(1, 32, n_kv, head_dim)` k/v tiles, but `update_cache_for_token_` expects INTERLEAVED `(1, n_kv, 1, head_dim)`. Similarly, RoPE emits `q_roped` HEIGHT_SHARDED on 32 cores but SDPA decode wants its Q HEIGHT_SHARDED on 1 core at `(0, 0)`. Two private methods bridge the gap.

`_bridge_kv_for_cache_update` (`modules/attention.py:93-104`) — used at `attention.py:157-158`:

```python
def _bridge_kv_for_cache_update(self, kv_heads):
    import ttnn
    interleaved = ttnn.sharded_to_interleaved(
        kv_heads, memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    sliced = ttnn.slice(
        interleaved,
        [0, 0, 0, 0],
        [1, 1, self.cfg.n_kv_heads, self.cfg.head_dim],
    )
    return ttnn.permute(sliced, [0, 2, 1, 3])
```

`_bridge_q_for_sdpa` (`modules/attention.py:106-128`) — used at `attention.py:162`. It builds the SDPA-decode-expected memory config via `SDPADecode.q_memory_config(...)`, does `sharded_to_interleaved → slice → interleaved_to_sharded`, and returns a tensor SDPA can consume.

Both bridges run as plain Python inside the orchestrator's `forward`. They are not graph ops, do not appear in any compiled program, and execute via direct `ttnn.*` calls each step. They are the most visible cost of host hops, and the reason `Qwen3Attention` has to be an orchestrator. See the v4 conventions block in `modules/__init__.py:48-64` for the full reasoning.

## The `_ua_*` attribute as a user-arg channel

The `_ua_` prefix is how a model author pushes scalars into the compiler's `user_args` dict from a `Module`. The two qwen3 examples:

```python
# modules/qkv_proj.py:29
self._ua_blackhole_cores = "64x8"

# modules/attention.py:50
self.o_proj._ua_blackhole_cores = "32x8"
```

Inside each Module, `_collect_user_args` (overridden in `FusedQKV` and `Qwen3MLP`; built-in on `OpModule`) iterates `dir(self)`, picks every name starting with `_ua_`, strips the prefix, and returns `{key: value}`. The compiler reads this through `BlazeCompiler.compile(..., user_args=...)` on the framework's `_call_graph` path. `Linear`'s patched `compose` (next section) is what then reads `user_args["blackhole_cores"]`.

## The Blackhole P150 monkey-patches

Two startup-time monkey-patches adapt blaze-nn for the Blackhole P150. Both are idempotent and applied once by the demo before any forward runs. They are documented here as a "last resort, not a recommended general pattern" — they exist because the upstream defaults don't fit P150's grid math.

### `_blaze_nn_linear_patch.py` — mixed-grid `Linear.compose`

`examples/qwen3_embedding_0_6b/modules/_blaze_nn_linear_patch.py:25-72`. The upstream `blaze_nn.Linear.compose` hard-codes `cores=f.matmul_cores`, which on P150 is 92 cores — and 92 does not divide Qwen3's Linear out_features. The patch replaces `compose` with a function that reads `user_args["blackhole_cores"]` and picks an 8x8 sub-grid for qkv or a 4x8 sub-grid for o_proj / mlp:

```python
@classmethod
def compose_mixed(cls, f, tensors, output, user_args):
    spec = (user_args or {}).get("blackhole_cores", "32x8")
    if spec == "64x8":
        cores = cores_64    # CoreRange((0,0),(7,7))
    else:
        cores = cores_32    # CoreRange((0,0),(3,7))
    act = Mcast.emit(f, tensors["input"], prefix="mcast")
    mm = DownProj.matmul(f, act, tensors["weights"], prefix="matmul", cores=cores)
    Gather.emit(f, mm, output_tensor=output, prefix="gather")

cls.compose = compose_mixed
```

This is why every `Linear` in qwen3 carries `_ua_blackhole_cores = "64x8"` or `"32x8"` — those `_ua_*` attributes are harvested by the surrounding Module's `_collect_user_args`, passed into `BlazeCompiler.compile(..., user_args=...)`, and read by the patched `compose` when fused-op synthesis runs.

The same patch also flips the `preferred_math_fidelity` of `blaze_nn_linear`, `matmul`, `down_proj`, `rmsnorm`, and `gated_reduce` to `HiFi4` (`_blaze_nn_linear_patch.py:64-70`) so the bf16 PCC path gets the right compute precision. Idempotence is enforced by the `_patched` class flag check at the top (`_blaze_nn_linear_patch.py:26-27`).

### `_register_sdpa_decode_user_alloc` — opt-in caller-allocated output

`examples/qwen3_embedding_0_6b/modules/attention.py:14-24` (inlined in `attention.py`, not a separate file as the directory layout might suggest):

```python
def _register_sdpa_decode_user_alloc() -> None:
    try:
        from blaze.ops.sdpa.op import SDPADecode
    except ImportError:
        return
    if getattr(SDPADecode, "_blaze_nn_user_alloc_patched", False):
        return
    SDPADecode.user_allocated_outputs = ("output",)
    SDPADecode._blaze_nn_user_alloc_patched = True
```

This declares SDPA decode's `output` port as caller-allocated so that `OpModule.set_output_tensor(t)` routes through the standard `_lookup_user_allocated_outputs` mechanism (Ch3 `output_tensors.md`, Ch6 `caller_allocated_outputs_internals.md`). After the patch, the per-layer `attn_out_tensor` becomes the SDPA op's caller-allocated output; without it, `sdpa_decode` would allocate its own output every call, breaking the address baking. It runs in `Qwen3Attention.__init__` (`attention.py:42`) and is guarded by a flag so reimports don't double-register.

> **Note:** Both patches are **last-resort fixes**, not a recommended general pattern. They reach into tt-blaze internals (`BlazeOp._class_registry`, `Linear.compose`, `SDPADecode.user_allocated_outputs`) rather than asking the framework for the right hook. They exist because the upstream tt-blaze ops have shapes or output contracts that don't cover the P150 + Qwen3 case yet. The right long-term fix is to upstream both behaviours into tt-blaze; the patches sit in the example so the demo can run today. A future tt-blaze release that supports per-target grids natively will obviate them. New ports should always check upstream first.

The recommended path for new ports is: *do not* monkey-patch by default. If you find you must, mirror these patches' shape — idempotent guard at the top, single startup call, comment explaining what upstream API would replace it.

## The demo and prefill

`demo/encode.py` (`examples/qwen3_embedding_0_6b/demo/encode.py`) currently raises `NotImplementedError`:

```python
def encode(text, model_id="Qwen/Qwen3-Embedding-0.6B"):
    raise NotImplementedError(
        "Phase A: encode() requires the prefill host-loop (Step 10 / Phase B). "
        "See README for the supported decode-shaped forward path."
    )
```

> **Note:** Prefill (encoding a multi-token sequence in one shot) is deferred to Phase B. The supported path is the **decode-shaped per-token forward**: one token in, KV cache updated, one hidden state out. Embedding extraction then runs the decode loop over the input tokens and reads the final hidden state.

## Test coverage — reverse index

The qwen3 port has four test slices, each pinning a different layer of the walkthrough:

- `tests/test_l0_config.py` — `Qwen3EmbeddingConfig` shape integers and derived properties (frozen-dataclass-only; no device).
- `tests/test_l0_keys.py` — the exact `state_dict` key set produced by `_build_blaze_nn_keys` (no device).
- `tests/test_l0_rope.py` — `_precompute_rope_tables` math vs HF reference (torch only).
- `tests/test_l1_token_embed.py` — `TokenEmbedding` parity with HF (device required).
- `tests/test_l1_rmsnorm.py` — `RMSNorm` parity (device required).
- `tests/test_l1_qkv_heads.py` — `FusedQKV` + `nlp_create_qkv_heads_decode` parity (device required).
- `tests/test_l1_rope.py` — `RoPE` parity end-to-end (device required).
- `tests/test_l1_kv_cache.py` — KV cache update path (device required).
- `tests/test_l1_sdpa.py` — SDPA decode parity (device required).
- `tests/test_layer_parity.py` — one full decoder layer parity (device required).
- `tests/test_e2e_parity.py` — the whole `Qwen3EmbeddingModel` parity (device required).
- `tests/test_weight_loader.py` — torch ↔ ttnn weight loader sanity.

The L0 / L1 / layer / e2e tiering matches the three test tiers from Ch1 `getting_started.md`: L0 tests run framework-only (no tt-blaze, no device); L1 tests need the device; `test_layer_parity` and `test_e2e_parity` need the device and exercise the full forward path. Ch7 `testing_strategy.md` reverse-indexes which chapter section each file backs.

> **For contributors:** the registry mechanism that backs `user_allocated_outputs` is in Ch6 `caller_allocated_outputs_internals.md`; the dispatch path that takes `F.<op>` to the actual blaze op handle is in Ch6 `functional_dispatch.md`; the active-context machinery that drives Mechanism B short-circuits is in Ch5 `tracing_contexts.md`.

_Previous: [The orchestrator pattern: two mechanisms](orchestrator_pattern.md) · Next: [Chapter 5 — Tracing internals: from `Module.__call__` to `program.run()`](../ch5_tracing_internals/index.md) · [Up](index.md)_
