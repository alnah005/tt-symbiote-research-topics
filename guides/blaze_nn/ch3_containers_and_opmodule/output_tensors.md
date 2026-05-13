# User-allocated output tensors

Most blaze-nn ops follow a simple rule: hand them inputs, get back a `ttnn.Tensor` allocated by the op itself. A small but important minority — `blaze_nn.Linear` is the canonical case — declare that the *caller* must pre-allocate the output buffer and pass it in before `forward()` is allowed to run. This file states the user-facing rule and shows the one idiom you need to remember.

## The rule

> **A module whose op declares `user_allocated_outputs` requires `set_output_tensor(t)` (or `set_output_tensors(name1=t1, ...)`) to be called before `forward`.**

Forgetting it raises (`blaze_nn/modules/base.py:417-423`):

```
RuntimeError: <ClassName> has unset required output tensor(s): [...]. Call
set_output_tensor(...) or set_output_tensors(...) before forward().
```

The check fires from `OpModule.__call__` at the outer call boundary — *before* the tracing context opens, so the error happens on the user's stack frame rather than deep inside the compiler.

## How to tell which modules need it

Two ways, both pre-call:

1. **Read the class docstring** of the module you are using. `Linear` says it plainly: *"the gather step writes into a pre-allocated `ttnn.Tensor` ... Construct it with the desired memory config and pass it via `set_output_tensor(...)` before `forward()`"* (`blaze_nn/modules/linear.py:14-17`). `RMSNorm`'s docstring does not mention output tensors — because `RMSNorm` does not need one.
2. **Check `m._required_output_names`** if you have an instance. It is the tuple read off `BlazeOp.user_allocated_outputs` at construction. An empty tuple means no user allocation; a non-empty tuple means one port per name.

In practice, the model author memorizes two: `blaze_nn.Linear` always requires one (`user_allocated_outputs = ("output",)` declared on the fused `BlazeNNLinear` op at `linear.py:41`); the qwen3 SDPA-decode patch adds one (`output`) to `sdpa_decode` so `OpModule(op="sdpa_decode")` does too. Everything else in the qwen3 example — `rmsnorm`, `residual_add`, `embedding`, `rope`, the MLP matmuls — auto-allocates internally.

## The one concrete example

The `Linear` case end-to-end. The exact path that `tests/test_pytorch_parity.py:test_linear_pipeline_matches_torch` walks is:

```python
import torch, ttnn
from blaze_nn.modules import Linear

D_in, D_out = 512, 1024
lin = Linear(D_in, D_out)

# Caller-built ttnn.Tensor with the right memory_config / shard_spec.
# In practice you build it via ttnn.from_torch(torch.zeros(...), ...);
# see tests/test_pytorch_parity.py:test_linear_pipeline_matches_torch
# for the full shard spec (HEIGHT_SHARDED on the sender core).
out = ttnn.from_torch(torch.zeros(...), ..., device=device)
lin.set_output_tensor(out)
lin.load_state_dict({"weight": w_ttnn})
lin.to(device)

y = lin(x)                        # x is a ttnn.Tensor input
```

The order of the four pre-`forward` steps is flexible — `set_output_tensor`, `load_state_dict`, and `to(device)` are all independent — but every one of them must happen before the first `lin(x)`. The actual `out` tensor must be allocated to match the shape and placement the op expects; the `test_linear_pipeline_matches_torch` test shows the right shard spec end-to-end (covered in the next file).

The multi-output variant uses kwargs keyed by port name:

```python
m.set_output_tensors(out_main=t1, out_aux=t2)
```

`set_output_tensors` validates each kwarg name against `_required_output_names`; an unknown name raises `KeyError("... has no user-allocated output 'foo'. Expected one of: ...")` (`blaze_nn/modules/base.py:399-403`). Conversely, `set_output_tensor(t)` (singular) only validates the exactly-one-output case — it raises `ValueError` otherwise (`base.py:388-393`).

> **Note:** blaze-nn does **not** copy the result into a fresh tensor. `lin(x)` returns the same `ttnn.Tensor` object you registered with `set_output_tensor`. Repeated calls reuse the same buffer (subject to the buffer-address-baking warning in Chapter 4 `tensor_lifetimes.md`).

## `_ua_*`: compile-time hints in one line

There is a second, much thinner per-module knob worth knowing at user level. Any attribute on an `OpModule` whose name begins with `_ua_` is treated as a compile-time argument and passed to the compiler in a dict keyed by the post-prefix name. The collector is `_collect_user_args` (`blaze_nn/modules/base.py:443-448`):

```python
def _collect_user_args(self) -> dict:
    args = {}
    for key in dir(self):
        if key.startswith("_ua_"):
            args[key[4:]] = getattr(self, key)
    return args
```

The qwen3 example sets exactly one such hint on `FusedQKV` to pin the QKV matmul to a 64×8 Blackhole subgrid (`examples/qwen3_embedding_0_6b/modules/qkv_proj.py:29`):

```python
self._ua_blackhole_cores = "64x8"
```

That single attribute reaches the compiler as `user_args={"blackhole_cores": "64x8"}` on the `FusedQKV` graph's compile call. The startup-time `_blaze_nn_linear_patch.py` monkey-patch then reads `user_args["blackhole_cores"]` to pick the matmul compose for that grid (Chapter 4 `buffers_and_address_baking.md`).

For a user, the takeaway is just: **prefix an attribute with `_ua_` and it becomes a compile-time argument**, named after the suffix. No other registration is needed.

> **For contributors:** The full chain — `user_allocated_outputs` declared on the `BlazeOp`, picked up by `OpModule.__init__` via `_lookup_user_allocated_outputs`, enforced by the pre-`forward` check, threaded through `_get_output_tensor` into `BlazeCompiler.compile(..., output_tensor=...)`, plus the parallel `_collect_user_args` → `BlazeCompiler.compile(..., user_args=...)` path — is consolidated in Chapter 6 `caller_allocated_outputs_internals.md`. The `define_fused_op` ↔ `_lookup_user_allocated_outputs` ↔ `set_output_tensor` triple is also restated there with the registration timing rules.

_Previous: [OpModule as a base class](opmodule_subclass.md) · Next: [Pre-built modules](prebuilt_modules.md) · [Up](index.md)_
