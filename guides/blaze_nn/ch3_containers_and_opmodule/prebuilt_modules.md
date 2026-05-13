# Pre-built modules: `blaze_nn.Linear` and `blaze_nn.ops.RMSNorm`

blaze-nn ships two ready-to-use modules that cover the workhorse ops every transformer port needs: the affine linear projection (`blaze_nn.Linear`) and root-mean-square normalization (`blaze_nn.ops.RMSNorm`). Both are `OpModule` subclasses, both have torch-shaped constructors so they are drop-in for a torch reader, and between them they exercise every public-API mechanism this chapter has introduced.

## `blaze_nn.Linear`

```python
from blaze_nn.modules import Linear
lin = Linear(in_features, out_features, bias=False)
```

Source: `blaze_nn/modules/linear.py:8-76`. The constructor mirrors `torch.nn.Linear` exactly — same positional argument order, same `bias` keyword — but bias is not yet supported on the hardware path and `bias=True` raises `NotImplementedError("Linear bias is not yet supported. Compose with F.residual_add.")` (`linear.py:67-70`).

Four facts that distinguish `Linear` from its torch counterpart:

1. **Fused as `mcast → matmul → gather`.** The op is a synthesized `FusedOp` called `BlazeNNLinear`, registered lazily on first instantiation via `define_fused_op` (`linear.py:23-59`). The `compose` classmethod stitches the three primitives in `blaze.ops.mcast`, `blaze.ops.down_proj`, and `blaze.ops.gather`. Once registered, `F.blaze_nn_linear` resolves and the default `OpModule.forward` routes through it.
2. **Weight tile is `[32, 32]`.** Auto-init via `_torch_init_specs` declares `[("weight", (in_features, out_features), [32, 32])]` (`linear.py:75-76`).
3. **Caller-allocated output.** `BlazeNNLinear.user_allocated_outputs = ("output",)` (`linear.py:41`) forces `lin.set_output_tensor(t)` before `forward`. The previous file (`output_tensors.md`) covers the rule; the snippet below shows where in the pipeline it lives.
4. **One graph input port for the weight.** The state-dict key is just `"weight"` — no nested prefix, because `Linear` is itself the module that owns the parameter. When the user wraps `Linear` in their own module (e.g., qwen3's `FusedQKV`, `examples/qwen3_embedding_0_6b/modules/qkv_proj.py:11-57`), the state-dict key becomes `linear.weight` and a small remap inside `load_state_dict` lets callers continue to use the bare `"weight"` key from a torch loader.

## `blaze_nn.ops.RMSNorm`

```python
from blaze_nn.ops.rmsnorm import RMSNorm
norm = RMSNorm(normalized_shape, eps=1e-6)
```

Source: `blaze_nn/ops/rmsnorm/op.py:8-29`. Same torch-shape rule: positional `normalized_shape`, default `eps=1e-6`. Behind the surface it is the simplest possible `OpModule` subclass — class attrs only, no custom `forward`, no `define_fused_op`. The constructor forwards `(eps, normalized_shape)` to the parent under the names the op expects (`epsilon`, `width`) and stashes both as plain attributes for the user.

- **Gamma tile is `[1, 32]`** — `_torch_init_specs` declares `[("gamma", (1, self.normalized_shape), [1, 32])]`.
- **No caller-allocated output.** RMSNorm allocates its own.
- **State-dict key is `"gamma"`** for the bare module; `<parent>.gamma` once nested.

### RMSNorm math

The op computes a per-row mean-square normalization scaled by a learnable per-channel gamma:

$$ \hat{x} = x \cdot \mathrm{rsqrt}\left(\mathrm{mean}(x^2) + \epsilon\right) \cdot \gamma $$

The reduction is over the trailing (channel) axis of width `normalized_shape`; $\epsilon$ is the `eps` constructor argument; $\gamma$ is the `gamma` parameter. The torch reference used in the parity test (`tests/torch_reference.py:rmsnorm_ref`) implements the same identity. Two practical consequences:

- **`gamma` defaults to ones** in torch's `nn.RMSNorm` initializer; in blaze-nn the user must populate it explicitly via `load_state_dict` or rely on `init_torch_params`, which produces a `torch.randn`-initialized gamma — for parity testing, build the gamma you want and pass it to `load_state_dict`.
- **The epsilon goes inside the square root**, not outside; the order matters for very small inputs. `RMSNorm(width, eps=1e-6)` forwards `epsilon=eps` into `_op_kwargs`, which the default `OpModule.forward` then passes to `F.rmsnorm(x, gamma, epsilon=1e-6, width=normalized_shape)`.

## The `blaze_nn/ops/` convention

`RMSNorm` lives at `blaze_nn/ops/rmsnorm/{__init__.py, op.py}`. The convention across `blaze_nn/ops/` is one subpackage per op-with-an-init-shape, mirroring the way `blaze.ops.*` is laid out. The `__init__.py` exists only to re-export the class so that `from blaze_nn.ops.rmsnorm import RMSNorm` works:

```python
# blaze_nn/ops/rmsnorm/__init__.py
from .op import RMSNorm
__all__ = ["RMSNorm"]
```

`Linear` is *not* in `ops/`. The split is intentional:

| Directory | Convention | Members |
|---|---|---|
| `blaze_nn/ops/<op>/` | one op per subpackage, op already in upstream registry | `RMSNorm` |
| `blaze_nn/modules/` | fused multi-op modules and base classes | `Linear`, plus `OpModule` itself |

When you have a single registered op and just want a torch-shaped constructor, the new wrapper goes in `ops/`; when you compose primitives, it goes in `modules/` with a `define_fused_op`. **Application-specific compositions** (qwen3's `TokenEmbedding`, `FusedQKV`, `RoPE`, `Qwen3MLP`) go in `examples/<model>/modules/`, not in `blaze_nn/`.

> **For contributors:** The end-to-end recipe for adding a new `ops/<op>/` subpackage — class skeleton, `_torch_init_specs`, optional `define_fused_op`, dispatch wiring, the test bucket to add the smoke test to — is Chapter 7 `add_an_op_wrapper.md`.

## End-to-end pipeline: `Linear` against a torch reference

The blaze-nn surface of `tests/test_pytorch_parity.py:test_linear_pipeline_matches_torch` is five lines:

```python
module = Linear(K, N_total)
module.set_output_tensor(ttnn_out)
module.load_state_dict({"weight": ttnn_b})
module.to(mesh_device)
output_torch = ttnn.to_torch(ttnn.from_device(module(ttnn_a)))
```

The three pre-`forward` steps are order-independent (see `output_tensors.md`); the test owns the shard-spec setup for `ttnn_a` / `ttnn_b` / `ttnn_out` and the `comp_pcc` check (threshold 0.99). `RMSNorm` is the same pattern minus `set_output_tensor` — see `tests/test_pytorch_parity.py:test_rmsnorm_matches_torch`.

## What is and is not in `blaze_nn` today

- **Linear and RMSNorm are the only two pre-built modules** in the public `blaze_nn.modules` / `blaze_nn.ops` surface. Everything else a model author needs comes from `OpModule(op="...")` directly or from a per-model hand-written `Module`.
- **No `Embedding` pre-built module.** The qwen3 example writes its own one-line `OpModule(op="embedding")` subclass in `examples/qwen3_embedding_0_6b/modules/token_embedding.py` — covered in Chapter 4 `composing_submodules.md`.
- **No activations, no dropout, no `LayerNorm`.** Activations live as op kwargs on the matmul-family ops (e.g. `F.gated_reduce(gate, up, activation="silu")`). Dropout is absent because no inference-time forward in the repo needs it. `LayerNorm` is absent because Qwen3 uses RMSNorm; if you need it, it lands in `blaze_nn/ops/layernorm/` as a copy of the RMSNorm wrapper with the right op name.

> **For contributors:** When you want to add a third pre-built module (LayerNorm, GeLU, etc.), the choice between "thin wrapper in `ops/`" and "fused composition in `modules/`" follows the same split: one registered op → `ops/`; multiple primitives stitched together → `modules/` with a `define_fused_op`. Chapter 7 `add_an_op_wrapper.md` walks the recipe end-to-end; Chapter 7 `add_a_fused_op.md` covers the fused case.

_Previous: [User-allocated output tensors](output_tensors.md) · Next: [Chapter 4 — Authoring models: the Qwen3 walkthrough](../ch4_qwen3_walkthrough/index.md) · [Up](index.md)_
