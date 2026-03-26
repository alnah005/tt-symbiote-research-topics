# Plan: 5 Wrapper Overhead Optimizations for Ling-mini-2.0 Decode

**Date:** 2026-03-26
**Baseline:** ~942ms/token decode, ~50% wrapper/Python overhead
**Target:** ~490ms/token decode (~47% reduction from wrapper optimizations alone)

---

## Implementation Order

1. **P2: Gated Prints** -- trivial, safe, immediate ~150ms savings
2. **P5: Fast tree_map** -- trivial, safe, ~10ms savings
3. **P3: Fast-path compose_transforms** -- moderate, ~140ms savings
4. **P1: TTNNBailingMoEDecoderLayer** -- largest change, ~80ms savings
5. **P4: Flatten module dispatch nesting** -- requires P1, ~60ms savings

Each optimization is independently testable. P4 builds on P1 since once the decoder layer is a TTNNModule, child calls can be flattened within it.

---

## P1: New TTNNBailingMoEDecoderLayer Module

### Problem

The HuggingFace `BailingMoeV2DecoderLayer.forward()` performs residual adds in PyTorch:
```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _, present_key_value = self.attention(hidden_states=hidden_states, ...)
hidden_states = residual + hidden_states       # <-- aten::add on CPU, forces _unwrap_to_torch
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
if isinstance(hidden_states, tuple):
    hidden_states, router_logits = hidden_states
hidden_states = residual + hidden_states       # <-- aten::add on CPU, forces _unwrap_to_torch
```

Each `residual + hidden_states` triggers 2x `_unwrap_to_torch` calls (one for residual, one for attn/MoE output), forcing device synchronization. This happens 40 times per token (2 per layer x 20 layers), costing 76.6ms total plus 4.3ms for the CPU-side aten::add ops.

### Solution

Create a new `TTNNBailingMoEDecoderLayer` class that replaces `BailingMoeV2DecoderLayer` and performs residual adds on-device using `ttnn.add`.

### Exact Class to Replace

`BailingMoeV2DecoderLayer` -- this is `model.model.layers[i].__class__` (confirmed for all 20 layers).

### Files to Change

| File | Change |
|------|--------|
| `modules/attention.py` (or new file `modules/decoder_layer.py`) | Add `TTNNBailingMoEDecoderLayer` class |
| `tests/test_ling_mini_2_0.py` | Register `BailingMoeV2DecoderLayer -> TTNNBailingMoEDecoderLayer` in `nn_to_ttnn` |

### New Module Design

```python
class TTNNBailingMoEDecoderLayer(TTNNModule):
    """Replaces BailingMoeV2DecoderLayer to keep residual adds on-device."""

    def __init__(self):
        super().__init__()
        self.input_layernorm = None       # TTNNRMSNorm
        self.post_attention_layernorm = None  # TTNNRMSNorm
        self.attention = None             # TTNNBailingMoEAttention
        self.mlp = None                   # TTNNBailingMoE (layers 1-19) or BailingMoeV2MLP (layer 0)
        self._is_dense_layer = False      # True for layer 0 (MLP instead of MoE)

    @classmethod
    def from_torch(cls, torch_layer):
        """Create from BailingMoeV2DecoderLayer."""
        from modules.normalization import TTNNRMSNorm

        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer

        # Create child TTNNModules
        new_layer.input_layernorm = TTNNRMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNRMSNorm.from_torch(torch_layer.post_attention_layernorm)
        new_layer.attention = TTNNBailingMoEAttention.from_torch(torch_layer.attention)

        # Layer 0 has dense MLP, layers 1-19 have MoE
        config = torch_layer.attention.config
        is_dense = (config.num_experts is None or
                    torch_layer.attention.layer_idx < config.first_k_dense_replace)
        new_layer._is_dense_layer = is_dense

        if is_dense:
            # Layer 0: BailingMoeV2MLP -- keep as PyTorch, the nn.Linear and nn.SiLU
            # inside it are already replaced by nn_to_ttnn2 dict
            new_layer.mlp = torch_layer.mlp  # NOT a TTNNModule; stays as nn.Module
        else:
            new_layer.mlp = TTNNBailingMoE.from_torch(torch_layer.mlp)

        return new_layer

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False,
                output_router_logits=False, use_cache=False,
                position_embeddings=None, **kwargs):
        # Ensure we're working with TTNN tensors
        if hasattr(hidden_states, 'to_ttnn'):
            hs = hidden_states.to_ttnn
        else:
            hs = hidden_states

        # Save residual (stays on device as TTNN tensor)
        residual = hs

        # Input layernorm
        hs = self.input_layernorm(hidden_states)
        if hasattr(hs, 'to_ttnn'):
            hs = hs.to_ttnn

        # Attention
        attn_out, _, present_key_value = self.attention(
            hidden_states=hs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        if hasattr(attn_out, 'to_ttnn'):
            attn_out = attn_out.to_ttnn

        # RESIDUAL ADD ON DEVICE (replaces aten::add)
        hs = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)

        # Save new residual (stays on device)
        residual = hs

        # Post-attention layernorm
        hs_normed = self.post_attention_layernorm(hs)
        if hasattr(hs_normed, 'to_ttnn'):
            hs_normed = hs_normed.to_ttnn

        # MLP / MoE
        mlp_out = self.mlp(hs_normed)
        router_logits = None
        if isinstance(mlp_out, tuple):
            mlp_out, router_logits = mlp_out
        if hasattr(mlp_out, 'to_ttnn'):
            mlp_out = mlp_out.to_ttnn

        # RESIDUAL ADD ON DEVICE (replaces aten::add)
        hs = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)

        outputs = (hs,)
        if output_attentions:
            outputs += (None,)
        if output_router_logits:
            outputs += (router_logits,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
```

### Test Registration

In `test_ling_mini_2_0.py`, add to `nn_to_ttnn` dict:
```python
from models.experimental.tt_symbiote.modules.decoder_layer import TTNNBailingMoEDecoderLayer

nn_to_ttnn = {
    model.model.layers[0].__class__: TTNNBailingMoEDecoderLayer,  # BailingMoeV2DecoderLayer
    model.model.layers[1].mlp.__class__: TTNNBailingMoE,
    model.model.layers[0].attention.__class__: TTNNBailingMoEAttention,
}
```

**IMPORTANT ordering concern:** When `register_module_replacement_dict` replaces `BailingMoeV2DecoderLayer`, it calls `TTNNBailingMoEDecoderLayer.from_torch()` which internally creates `TTNNBailingMoEAttention` and `TTNNBailingMoE` children. This means the attention and MoE entries in `nn_to_ttnn` become redundant for modules that are children of the decoder layer. Two approaches:

- **Option A (recommended):** Remove `TTNNBailingMoEAttention` and `TTNNBailingMoE` from `nn_to_ttnn` since `from_torch` handles their creation internally. The decoder layer replacement must be registered first (before the general nn.Linear/SiLU replacements).
- **Option B:** Keep all entries but use `exclude_replacement` to skip modules that are children of already-replaced decoder layers.

Choose Option A for simplicity.

### Edge Cases

1. **Layer 0 has MLP, not MoE:** `first_k_dense_replace = 1` means layer 0 uses `BailingMoeV2MLP` (dense). The `from_torch` method detects this via `layer_idx < config.first_k_dense_replace` and keeps the MLP as a regular `nn.Module`. The `nn.Linear` and `nn.SiLU` inside it are still replaced by `nn_to_ttnn2`.

2. **position_embeddings as tuple:** The HF decoder layer passes `position_embeddings=(cos, sin)` to attention. When this goes through `compose_transforms`, the tuple elements get wrapped. The new decoder layer should pass `position_embeddings` through without wrapping since it is handled inside the attention module.

3. **Return format:** HF decoder layer returns a tuple `(hidden_states, ...)` with optional `self_attn_weights`, `router_logits`, `present_key_value`. The `TTNNBailingMoEDecoderLayer.forward()` must match this exactly for the HF model code that consumes the output.

4. **input_layernorm / post_attention_layernorm:** Currently NOT replaced as TTNNModules (not in `nn_to_ttnn`). The `from_torch` creates them as `TTNNRMSNorm` instances. This means they will now run as TTNN ops rather than through PyTorch dispatch. This is actually beneficial -- removes those 40 RMSNorm calls from the wrapper overhead.

5. **Distributed tensor config:** The decoder layer output must have the correct distributed tensor config set. The `set_output_tensors_config_impl` from TTNNModule base class handles this via `tree_map(set_distributed_tensor_config(...), output)`.

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Shape/dtype mismatch between residual and attn_out for ttnn.add | Medium | High (wrong output) | Add assert on shapes before ttnn.add; test with known-good reference |
| Layer 0 MLP output not compatible with ttnn.add | Medium | Medium | Layer 0 MLP runs through PyTorch dispatch; output is TorchTTNNTensor which needs .to_ttnn before ttnn.add |
| register_module_replacement_dict traversal order | Low | Medium | Verify decoder layer children are created in from_torch, not replaced again by subsequent passes |
| HF model code expects specific return tuple format | Low | High | Match BailingMoeV2DecoderLayer.forward return exactly |

### Estimated Savings

- Eliminates 84 `_unwrap_to_torch` calls: -76.6ms
- Eliminates 40 `aten::add.Tensor` calls: -4.3ms
- **Total: ~80ms**

---

## P2: Gated Prints for Lightweight Run Mode

### Problem

`NormalRun.module_run()` at line 557 of `run_config.py` prints a debug line for every TTNNModule call:
```python
print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
```

This runs 363 times per decode token, costing an estimated 100-200ms in terminal I/O.

### Solution

Gate behind a `NormalRun.verbose` class variable (which already exists at line 424 and defaults to `False`). **Do NOT remove the prints.**

### Files to Change

| File | Line | Change |
|------|------|--------|
| `core/run_config.py` | 557 | Gate print behind `NormalRun.verbose` |
| `core/run_config.py` | 620 | Gate print in `NormalRunWithFallback.module_run` too |
| `core/run_config.py` | 666 | Gate print in `SELRun.module_run` too |

### Exact Change

```python
# Line 557, NormalRun.module_run:
# BEFORE:
print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")

# AFTER:
if NormalRun.verbose:
    print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
```

Same pattern for lines 620 and 666.

### How the Test Enables/Disables

By default, `NormalRun.verbose = False` (already the case at line 424). No test change needed -- prints are already gated by the existing variable, but the `module_run` method at line 557 does NOT check it. The fix is simply to add the `if` guard.

For debugging, users can set `NormalRun.verbose = True` before running.

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Someone depends on print output for debugging | Low | Low | Verbose flag is easy to enable; add docstring |
| Performance regression from if-check | None | None | Single bool check is negligible |

### Estimated Savings

- Eliminates ~363 print calls/token at ~0.3-1ms each: **~100-200ms**

---

## P3: Fast-path compose_transforms

### Problem

`compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(device))` is called for every input argument to every TTNNModule (594 calls/token). Each call applies 3 transforms sequentially:

1. `wrap_to_torch_ttnn_tensor` -- isinstance check, wraps plain torch.Tensor
2. `to_ttnn_wrap` -- calls `.to_ttnn` property (calls `ttnn.from_torch()` if not cached)
3. `set_device_wrap` -- calls `ttnn.to_device()` if not on correct device

For tensors that are already `TorchTTNNTensor` with `.ttnn_tensor` set and on the correct device (the common case in steady-state decode), all three transforms are no-ops but still perform isinstance checks, attribute accesses, and Python function calls. Total cost: 158.9ms.

### Exact Location

`core/run_config.py`, line 560 in `NormalRun.module_run`:
```python
transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
func_args = tree_map(transform, args)
```

Also line 623-624 in `NormalRunWithFallback.module_run` and line 674 in `SELRun.module_run`.

### Solution

Add a fast-path directly in `compose_transforms` or create a new `fast_transform` function that short-circuits for already-prepared tensors:

```python
def make_fast_transform(device):
    """Create a fast-path transform that skips processing for already-prepared tensors."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    # Cache the full transform for fallback
    full_transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(device))
    full_transform.__name__ = "wrap_to_torch_ttnn_tensor_to_ttnn_wrap__set_device_wrap"

    def _fast_transform(e):
        # Fast path: already a TorchTTNNTensor with ttnn_tensor on the correct device
        if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
            # Check device match (handles both single and mesh devices)
            current_device = e.ttnn_tensor.device()
            if current_device is not None and current_device == device:
                return e
        # Fallback to full transform for non-tensor args (None, int, etc.)
        if not isinstance(e, (torch.Tensor, ttnn.Tensor)):
            return e
        return full_transform(e)

    _fast_transform.__name__ = full_transform.__name__
    return _fast_transform
```

### Where to Apply

Replace line 560:
```python
# BEFORE:
transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))

# AFTER:
transform = make_fast_transform(self.device)
```

Same for lines 623 and 674.

### Edge Cases

1. **None args:** The `compose_transforms` functions handle None gracefully (isinstance checks return False, so None passes through). The fast path must do the same -- the `not isinstance(e, (torch.Tensor, ttnn.Tensor))` check handles this.

2. **position_embeddings tuple:** `tree_map` recurses into tuples, so each element (cos, sin) is individually transformed. The fast path handles each tensor individually.

3. **First call (tensor not yet on device):** The fast path falls through to `full_transform`, which handles the initial wrapping and device transfer correctly.

4. **Tensor on wrong device (e.g., after device migration):** The device check `current_device == device` catches this and falls through to full transform.

5. **past_key_value filter:** Line 563-565 filters `past_key_value` out of kwargs before transforming. This must remain unchanged.

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fast path returns stale tensor (device changed) | Low | High | Explicit device equality check |
| Fast path misses edge case (e.g., ttnn_tensor set but elem also set) | Low | Medium | Full transform fallback handles all cases |
| Timing recording breaks (DispatchManager) | None | Low | Transform name preserved via __name__ |

### Estimated Savings

- 594 calls * 0.267ms -> 594 calls * ~0.03ms (single isinstance + attribute check)
- **Net: ~140ms**

---

## P4: Flatten Module Dispatch Nesting

### Problem

Composite modules like `TTNNBailingMoEAttention` call 6 child TTNNModules (q_proj, k_proj, v_proj, query_layernorm, key_layernorm, dense), each going through the full `module_run` cycle:

```
TTNNBailingMoEAttention.__call__()
  -> module_run(self, ...)           # wraps args, preprocess, timing
    -> self.forward(...)
      -> self.q_proj(hidden_states)  # child call
        -> module_run(q_proj, ...)   # wraps args AGAIN, preprocess (cached), timing
          -> q_proj.forward(...)     # actual TTNN matmul
```

Each child `module_run` adds:
- `compose_transforms` on args: ~0.3ms (even with P3 fast-path: ~0.03ms)
- `post_process_ttnn_module_output`: wraps output + sets distributed config: ~0.23ms
- `preprocess_weights` + `move_weights_to_device` (cached, near-zero but has Python overhead)
- `DispatchManager.record_timing` calls: ~0.02ms each
- `DispatchManager.set_current_module_name` push/pop: ~0.01ms

Per child: ~0.3ms overhead. 6 children * 20 layers = 120 child calls * 0.3ms = ~36ms. Plus the nesting overhead of the composite modules themselves.

### Solution

Inside the TTNNBailingMoEDecoderLayer (from P1) and TTNNBailingMoEAttention, call child TTNN ops directly instead of going through `__call__` (which triggers `module_run`). This means calling `child.forward()` directly instead of `child()`.

### Which Composite Modules to Flatten

1. **TTNNBailingMoEDecoderLayer** (from P1): calls input_layernorm, attention, post_attention_layernorm, mlp
2. **TTNNBailingMoEAttention**: calls q_proj, k_proj, v_proj, query_layernorm, key_layernorm, dense

The MoE module (TTNNBailingMoE/TTNNMoE) already mostly operates on raw TTNN tensors internally, so there is less to flatten there.

### How to Bypass module_run

Instead of:
```python
query_states = self.q_proj(hidden_states)  # goes through module_run
```

Use:
```python
# Ensure weights are preprocessed and on device (already done by parent's module_run)
q_out_ttnn = self.q_proj.forward(hidden_states_ttnn)  # direct TTNN call, no wrapping
```

Key requirement: The parent's `module_run` already called `preprocess_weights()` and `move_weights_to_device()` on all children (via the recursive base class implementation). So calling `child.forward()` directly is safe as long as:
1. Input is already a raw `ttnn.Tensor` (not TorchTTNNTensor)
2. Output is consumed as raw `ttnn.Tensor` (not wrapped)

### Files to Change

| File | Function | Change |
|------|----------|--------|
| `modules/attention.py` | `TTNNBailingMoEAttention._forward_decode_paged` | Call child `.forward()` directly instead of `child()` |
| `modules/attention.py` | `TTNNBailingMoEAttention._forward_prefill` | Same |
| `modules/decoder_layer.py` (from P1) | `TTNNBailingMoEDecoderLayer.forward` | Call `.forward()` on input_layernorm, post_attention_layernorm directly |

### Pseudocode for Attention Decode Flatten

```python
def _forward_decode_paged(self, hidden_states, ...):
    # hidden_states is already a raw ttnn.Tensor (parent ensures this)

    # Direct calls bypass module_run wrapping
    query_states = self.q_proj.forward(hidden_states)
    hidden_states_replicated = ttnn.all_gather(hidden_states, dim=-1, num_links=1)
    key_states = self.k_proj.forward(hidden_states_replicated)
    value_states = self.v_proj.forward(hidden_states_replicated)
    ttnn.deallocate(hidden_states_replicated)

    # ... rest of decode logic (already uses raw ttnn ops) ...

    # QK norm direct calls
    q_normed = self.query_layernorm.forward(q_reshaped)
    k_normed = self.key_layernorm.forward(k_reshaped)

    # Dense projection
    attn_output = self.dense.forward(attn_output)
    return attn_output, None, past_key_values
```

### Edge Cases

1. **Timing no longer recorded for child modules:** Since we bypass `module_run`, `DispatchManager` won't record per-child timings. Add optional manual timing if needed for profiling. For production decode, this is acceptable.

2. **preprocess_weights/move_weights_to_device:** These are idempotent and already called by the parent. But verify that the parent's `preprocess_weights_impl` and `move_weights_to_device_impl` recursively call all children (confirmed: the base TTNNModule class does this in `core/module.py` lines 107-118).

3. **set_output_tensors_config:** Skipped for child modules when calling `.forward()` directly. The parent module's post-processing handles the final output config. Intermediate tensors don't need distributed config since they stay on-device.

4. **Child modules that expect TorchTTNNTensor input:** Some child `.forward()` methods may call `hasattr(x, 'to_ttnn')` which fails on raw `ttnn.Tensor`. Check each child and ensure they accept raw `ttnn.Tensor`. Current TTNNBailingMoEAttention already handles both (`if hasattr(..., 'to_ttnn'): ... = ....to_ttnn`).

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Child forward() expects TorchTTNNTensor | Medium | High | Audit each child's forward(); ensure raw ttnn.Tensor works |
| Missing profiling data for children | Low | Low | Add optional manual timing behind verbose flag |
| Weight not preprocessed for child | Low | High | Verify parent recursion covers all children |
| SELRun / DPLRun comparison modes break | Medium | Medium | Only flatten in NormalRun path; SEL/DPL continue using module_run |

### Estimated Savings

- 120 child calls * ~0.3ms overhead each = **~36ms**
- 20 decoder layer children (norms) * ~0.3ms = **~6ms**
- Plus reduced Python function call overhead
- **Total: ~60ms**

---

## P5: Fast tree_map

### Problem

`tree_map` (in `core/utils.py` line 153) wraps `optimized_tree_map_with_only_dict_list` with timing instrumentation. The underlying function recursively traverses dicts, lists, and other sequences. For the common case of `module_run` args, the input is typically:
- `args` = `(hidden_states,)` -- a tuple with 1 tensor
- `kwargs` = `{'position_embeddings': (cos, sin), 'attention_mask': None, ...}` -- a dict with 3-5 entries

The recursive implementation creates intermediate lists, checks `isinstance(ds, Mapping)` and `isinstance(ds, Sequence)` at each level, imports `collections.abc`, and calls `time.time()` for timing. For 594 transform calls per token, this overhead adds up.

### Current Implementation Location

`core/utils.py`, lines 129-171:
- `optimized_tree_map_with_only_dict_list` (line 129): recursive traversal
- `tree_map` (line 153): timing wrapper around the above

### Solution

Add fast paths for the most common argument patterns:

```python
def tree_map(func, data_structure, *extra_args, **kwargs):
    import time
    from models.experimental.tt_symbiote.core.run_config import DispatchManager

    start_time = time.time()

    # Fast paths for common cases
    if not extra_args:  # Single data structure (the 99% case in module_run)
        if isinstance(data_structure, torch.Tensor):
            # Single tensor (most common for args)
            result = func(data_structure)
        elif isinstance(data_structure, tuple):
            n = len(data_structure)
            if n == 0:
                result = ()
            elif n == 1:
                result = (func(data_structure[0]),)
            elif n == 2:
                result = (func(data_structure[0]), func(data_structure[1]))
            elif n <= 4:
                result = tuple(func(x) for x in data_structure)
            else:
                result = optimized_tree_map_with_only_dict_list(func, data_structure, **kwargs)
        elif isinstance(data_structure, dict):
            # Shallow dict (kwargs case) -- apply func to values only, no recursion
            result = {k: func(v) if isinstance(v, torch.Tensor) else
                      (tuple(func(x) for x in v) if isinstance(v, tuple) else v)
                      for k, v in data_structure.items()}
        else:
            # Fallback: non-tensor leaf (None, int, etc.)
            result = func(data_structure)
    else:
        result = optimized_tree_map_with_only_dict_list(func, data_structure, *extra_args, **kwargs)

    end_time = time.time()
    DispatchManager.record_timing(
        "Torch",
        ("" if DispatchManager.current_module_name is None
         else DispatchManager.current_module_name + f".{func.__name__}"),
        func.__name__,
        {},
        end_time - start_time,
    )
    return result
```

### Edge Cases

1. **Nested dicts/lists:** The fast path only handles shallow structures. Deep nesting falls through to the recursive `optimized_tree_map_with_only_dict_list`. In practice, module_run args are never deeply nested.

2. **kwargs with nested tuples:** `position_embeddings` is `(cos, sin)` -- a tuple of 2 tensors inside a dict. The fast dict path handles this with the tuple branch: `tuple(func(x) for x in v)`.

3. **Non-tensor values in tuples:** Some tuples may contain `None` (e.g., position_embeddings might be None). The `func` (compose_transforms) handles None gracefully by returning it unchanged.

4. **extra_args (multi-data-structure tree_map):** Only used in SELRun for comparing outputs. Falls through to the recursive version.

5. **tree_map used in post_process_ttnn_module_output:** Called with `wrap_to_torch_ttnn_tensor` on the forward() result, which is typically a single tensor or a tuple of (tensor, None, cache). The fast tuple path handles this.

### Files to Change

| File | Function | Change |
|------|----------|--------|
| `core/utils.py` | `tree_map` (line 153) | Add fast paths before calling `optimized_tree_map_with_only_dict_list` |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fast path mishandles nested structure | Low | Medium | Fallback to recursive for len > 4 or deep nesting |
| Dict fast path misses non-tensor non-tuple values | Low | Low | Non-matching values returned as-is (func handles None) |
| Timing changes due to skipping recursive overhead | None | None | Timer still wraps the fast path |

### Estimated Savings

- Eliminates recursive function calls and isinstance checks for 594 + 297 = 891 tree_map calls
- **~10-15ms**

---

## Testing Strategy

### Per-Optimization Verification

Each optimization should be verified independently by running the existing test:

```bash
cd /home/ttuser/salnahari/tt-metal
MESH_DEVICE=T3K pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -xvs
```

**Pass criteria:**
1. Test completes without errors
2. Generated text is coherent (non-empty, readable English)
3. Timing CSV shows improvement in the targeted category

### Detailed Verification per Optimization

| Optimization | Verification |
|-------------|-------------|
| P2 (gated prints) | Run test, verify no prints to stdout. Set `NormalRun.verbose = True`, verify prints appear. |
| P5 (fast tree_map) | Run test, compare output text with baseline. Check timing CSV for reduced tree_map time. |
| P3 (fast compose_transforms) | Run test, check timing CSV: `wrap_to_torch_ttnn_tensor_to_ttnn_wrap__set_device_wrap` total time should drop from ~159ms to ~18ms. |
| P1 (decoder layer) | Run test, verify output matches. Check timing CSV: no `_unwrap_to_torch` entries (or drastically fewer). No `aten::add.Tensor` entries for residual adds. |
| P4 (flatten nesting) | Run test, verify output matches. Check timing CSV: fewer `TorchModules` entries (child modules no longer individually tracked). |

### Regression Testing

After all 5 optimizations are applied:
1. Run the full test with `max_new_tokens=128` and verify output text quality
2. Compare total decode time vs baseline (~942ms/token)
3. Verify the timing breakdown shows the expected reductions
4. Run prefill path as well (not just decode) to ensure no regression

### Numerical Accuracy Check

For P1 and P4, where computation moves from CPU to TTNN device:
- `ttnn.add` uses bfloat16 on device vs float32 on CPU. Verify the output still matches within acceptable tolerance
- Run a short generation (10 tokens) and compare output token-by-token with the CPU-only baseline

---

## Summary Table

| # | Optimization | Files | Estimated Savings | Risk | Depends On |
|---|-------------|-------|-------------------|------|-----------|
| P2 | Gated prints | `core/run_config.py` | ~150ms | Very Low | None |
| P5 | Fast tree_map | `core/utils.py` | ~10ms | Low | None |
| P3 | Fast compose_transforms | `core/run_config.py` | ~140ms | Low | None |
| P1 | TTNNBailingMoEDecoderLayer | `modules/decoder_layer.py`, `tests/test_ling_mini_2_0.py` | ~80ms | Medium | None |
| P4 | Flatten dispatch nesting | `modules/attention.py`, `modules/decoder_layer.py` | ~60ms | Medium | P1 |
| **Total** | | | **~440ms (47%)** | | |
