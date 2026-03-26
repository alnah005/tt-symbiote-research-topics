# CPU Fallback Paths — Glm4MoeNaiveMoeHybrid

## Context

This file addresses:
- **Q7** — Is the TTNN path actually active in a given deployment, or is the model silently running on CPU?
- **Q8** — What are the conditions that trigger fallback to the non-TTNN expert loop?

Source range: `moe.py:L559–L613`

---

## Overview

`Glm4MoeNaiveMoeHybrid` is a hybrid class that wraps the expert weights from an existing `Glm4MoeNaiveMoe` layer and is intended to support both a TTNN-accelerated path (`TTNNGlm4MoeExpertLayers`) and a pure-PyTorch fallback path (`Glm4MoeExpertLayersTorch`). At the time of writing, the TTNN path is **permanently disabled by a hardcoded flag**.

This class is the primary risk vector for accidentally running expert computation on CPU without any error, warning, or measurable TTNN trace. Profiling results collected while this class is active as `self.expert_layers` will not reflect the T3K kernel's performance.

---

## The Hardcoded Disable Flag (moe.py:L569–570)

```python
def __init__(self, old_layer, num_experts_off_chip: int = 20):
    super().__init__()
    self.num_experts = old_layer.num_experts
    self.hidden_dim = old_layer.hidden_dim
    self.intermediate_dim = old_layer.intermediate_dim

    # Create TTNN expert layers module
    ttnn = False   # <-- hardcoded: TTNN path is disabled
    if ttnn:
        self.expert_layers = TTNNGlm4MoeExpertLayers.from_parameters(
            old_layer.gate_up_proj, old_layer.down_proj, num_experts_off_chip=num_experts_off_chip
        )
        del old_layer.gate_up_proj
        del old_layer.down_proj
    else:
        self.expert_layers = Glm4MoeExpertLayersTorch(old_layer.gate_up_proj, old_layer.down_proj)

    assert old_layer.config.hidden_act == "silu", "Only SiLU activation is supported in naive MoE."
```

Because `ttnn = False` is a hardcoded literal, the `if ttnn:` branch is dead code; `Glm4MoeExpertLayersTorch` always runs, and the weight deletion in the `if` branch is also skipped, so those tensors remain allocated in `old_layer`.

---

## The CPU Expert Loop (moe.py:L584–L613)

```python
def forward(self, hidden_states, top_k_index, top_k_weights):
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        current_hidden_states = self.expert_layers(current_state, expert_idx.item())
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states
```

This `forward` method runs an expert-by-expert Python loop, distinct from the fully vectorized, hardware-parallel approach in `TTNNExperts.forward`. Key differences from the TTNN path:

| Property | `TTNNExperts.forward` | `Glm4MoeNaiveMoeHybrid.forward` |
|---|---|---|
| Execution device | T3K Tensix cores | CPU (PyTorch) |
| Expert dispatch | `all_to_all_dispatch` (hardware CCL) | `one_hot` + Python `for` loop |
| Expert compute | `sparse_matmul` (Wormhole hardware) | `Glm4MoeExpertLayersTorch` (CPU matmul) |
| Sparsity | Hardware block-sparse skip | Loop over only `expert_hit` experts |
| Weight application | `repeat + permute + mul + sum` (TTNN) | `index_add_` (CPU) |
| Parallelism | All experts in parallel across devices | Sequential per-expert |

The CPU loop does implement one optimization: `expert_hit` contains only the experts that received at least one token (`expert_mask.sum(...) > 0`), so experts with no assigned tokens are skipped entirely. This is semantically equivalent to the sparsity mask in the TTNN path but implemented in Python rather than hardware.

The `index_add_` at the end accumulates weighted expert outputs into `final_hidden_states`. The guard `if expert_idx == self.num_experts: continue` handles an off-by-one edge case that can arise if the top-k routing produces an out-of-range expert index.

---

## Checklist: Verifying the TTNN Path Is Active

Before collecting any performance measurements, confirm each item in this checklist.

### 1. Confirm the MoE class in use

Inspect the model's MoE layer class at runtime:

```python
for name, module in model.named_modules():
    if "moe" in type(module).__name__.lower():
        print(name, type(module).__name__)
```

The TTNN path requires `TTNNMoE` (or `TTNNBailingMoE`) as the outer module and `TTNNExperts` as `module.experts`. If you see `Glm4MoeNaiveMoeHybrid` or `Glm4MoeExpertLayersTorch`, the TTNN path is not active.

### 2. Confirm the `ttnn` flag is not hardcoded to False

Search the source:

```bash
grep -n "ttnn = False" moe.py
```

If `moe.py:L569–570` appears in the output and that file is the one loaded by the running process, the `Glm4MoeNaiveMoeHybrid` class will never use the TTNN path. The fix requires editing the source to set `ttnn = True` or replacing the flag with a proper configuration parameter.

### 3. Confirm the device is T3K and `@run_on_devices` is not short-circuiting

`TTNNMoE.forward` and `TTNNExperts.forward` are decorated with `@run_on_devices(DeviceArch.T3K)`. This decorator is a guard that may route execution to a fallback if the device does not match. Confirm the running device is T3K:

```python
print(model.device.arch())  # expected: DeviceArch.T3K
```

If the device is `DeviceArch.GRAYSKULL` or `DeviceArch.WORMHOLE_B0` (single-chip), the decorator may fall through to a default CPU path depending on its implementation.

### 4. Confirm weights are on device

`TTNNExperts` must have its weights moved to the T3K devices before forward is called. Check that `tt_w1_proj`, `tt_w3_proj`, and `tt_w2_proj` are not `None` and are `ttnn.Tensor` objects on the expected device:

```python
experts = model.layers[0].mlp.experts  # path varies
print(type(experts.tt_w1_proj))        # expected: ttnn.Tensor
print(experts.tt_w1_proj.device())     # expected: T3K mesh device
```

If these are `None` or are PyTorch tensors, `move_weights_to_device_impl` was not called.

### 5. Confirm CCL semaphores are initialized

`TTNNMoE.forward` calls `self.device_state.ccl_manager.get_and_cycle_ag_semaphore_handles(1)` on every forward pass. If `ccl_manager` is `None` or uninitialized, the forward call will raise an `AttributeError` rather than silently falling back. This means a clean TTNN run will always produce a semaphore-related error if `device_state` is misconfigured — it will not silently fall back to CPU.

### 6. Check for TTNN op trace in profiler output

If using the TTNN profiler or `ttnn.device.begin_trace_capture`:

```python
ttnn.device.begin_trace_capture(device, trace_buffer_size=...)
model(inputs)
ttnn.device.end_trace_capture(device, trace_id)
```

A TTNN trace for a MoE forward pass should contain ops named `all_gather_async`, `all_to_all_dispatch`, `sparse_matmul`, `all_to_all_combine`, and `reduce_scatter_minimal_async`. The absence of any of these ops confirms the corresponding path did not execute on Tensix cores.

---

## How to Enable the TTNN Path in Glm4MoeNaiveMoeHybrid

The minimum change required is to edit `moe.py:L569–570`:

```python
# Before:
ttnn = False

# After:
ttnn = True
```

However, enabling this requires that `TTNNGlm4MoeExpertLayers.from_parameters` succeeds, which in turn requires:
- The device is a T3K mesh with the expected number of devices.
- `num_experts_off_chip` is set appropriately for the available DRAM.
- Weight tensors in `old_layer.gate_up_proj` and `old_layer.down_proj` are compatible with `TTNNGlm4MoeExpertLayers.from_parameters`.

Until the flag is changed and the TTNN expert layers are validated, `Glm4MoeNaiveMoeHybrid` is solely a CPU reference implementation.

---

**Next:** [Chapter 2 — CCL Latency and Topology](../ch2_ccl_latency_and_topology/index.md)
