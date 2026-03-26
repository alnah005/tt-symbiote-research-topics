# GLM-4 CPU Path Audit

## Context

This file addresses **Q6**: quantifying the latency cost of the `TTNNGlm4MoeMoE` CPU expert path and providing a migration checklist to eliminate it.

Source ranges: `moe.py:L542–L556`, `moe.py:L559–L613`, `moe.py:L817–L852`.

---

## 1. Architecture Summary

`TTNNGlm4MoeMoE` (`moe.py:L817–L852`) is the outer MoE module used in the GLM-4 stack. Its `from_torch` factory constructs four sub-components:

| Sub-component | Class | Execution device |
|---|---|---|
| Router | `TTNNGlm4MoeTopkRouter` | T3K (TTNN) |
| Routed experts | `Glm4MoeNaiveMoeHybrid` | **CPU (PyTorch)** |
| Shared experts | `TTNNGlm4MoeMLP` | T3K (TTNN) |
| Token routing | `TTNNGlm4MoeRouteTokenToExperts` | T3K (TTNN) |

Three of the four components run on Tensix cores. The exception is `Glm4MoeNaiveMoeHybrid`, assigned to `self.experts`. In `TTNNGlm4MoeMoE.forward`, after the TTNN router produces `topk_indices` and `topk_weights`, execution returns to the host:

```python
hidden_states = self.experts(hidden_states, topk_indices.to(dtype=torch.int64), topk_weights)
```

`Glm4MoeNaiveMoeHybrid.forward` runs entirely on CPU, then `TTNNGlm4MoeMoE.forward` adds the TTNN shared-expert result on top. The router and shared experts appear in any TTNN trace; the expert computation does not.

The root cause is the hardcoded `ttnn = False` flag at `moe.py:L569–570`, which permanently selects `Glm4MoeExpertLayersTorch` over `TTNNGlm4MoeExpertLayers`. Ch1's `cpu_fallback_paths.md` covers the flag, the dead `if ttnn:` branch, and the checklist for confirming whether the TTNN path is active. This file does not repeat that analysis.

---

## 2. Latency Cost of the CPU Path

### Why the CPU path is expected to be 2–3 orders of magnitude slower

`Glm4MoeNaiveMoeHybrid.forward` (`moe.py:L584–L613`) runs a Python for-loop over the experts that received at least one token. At decode time (batch size = 1, top-k = 8), `expert_hit` contains 8 entries, so the loop executes 8 iterations.

Each iteration calls `Glm4MoeExpertLayersTorch.forward` (`moe.py:L551–L556`):

```python
gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
current_hidden_states = self.act_fn(gate) * up
current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
```

This performs two CPU matmuls per expert:

- First matmul: `[1, hidden_dim] @ [2*intermediate_dim, hidden_dim].T` → `[1, 2*intermediate_dim]`
- Second matmul: `[1, intermediate_dim] @ [hidden_dim, intermediate_dim].T` → `[1, hidden_dim]`

The combined cost per expert is two back-to-back small-matrix multiplications in single-token decode. These are among the worst-case shapes for CPU compute utilization: the matrices are tall and thin, limiting BLAS parallelism, and each call incurs Python dispatch overhead, kernel launch overhead, and tensor allocation.

The contributing factors to the expected 2–3 order-of-magnitude gap versus `TTNNExperts`:

1. **No inter-device parallelism.** All 8 T3K devices are idle during the expert loop. `TTNNExperts` dispatches expert computation across the full mesh using `all_to_all_dispatch` and hardware CCL.

2. **Sequential Python for-loop.** The 8 expert matmuls execute one at a time. `TTNNExperts` executes all experts in a single hardware-parallel `sparse_matmul` call.

3. **Weights in host DRAM.** `Glm4MoeExpertLayersTorch` holds `gate_up_proj` and `down_proj` as `nn.Parameter` tensors in system memory. Each matmul reads weight data over the CPU memory bus. `TTNNExperts` holds weights in device DRAM or L1, accessed by Tensix cores at device memory bandwidth.

4. **Python interpreter overhead per iteration.** Index selection (`self.gate_up_proj[expert_idx]`), `.chunk()`, tensor multiplication, `index_add_`, and the loop control itself each add non-trivial Python overhead that accumulates over the 8 iterations.

5. **No hardware SiLU fusion.** The activation function is an `nn.SiLU()` module applied as a separate Python op. On Tensix, gated SiLU is fused into the matmul pipeline.

### Measurement methodology

To quantify the gap on a specific model configuration, wrap both forward methods with `time.perf_counter()`:

```python
import time
import torch

def measure_forward(fn, *args, n_warmup=100, n_measure=1000):
    """
    Measure the wall-clock latency of fn(*args).
    Returns (mean_ms, min_ms, max_ms) over n_measure iterations.
    """
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)  # convert to ms

    times = sorted(times)
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "p50_ms": times[len(times) // 2],
        "p99_ms": times[int(len(times) * 0.99)],
    }

# Usage:
# cpu_stats = measure_forward(cpu_experts.forward, hidden_states, topk_indices, topk_weights)
# ttnn_stats = measure_forward(ttnn_moe.forward, hidden_states)
# print(f"CPU: {cpu_stats['mean_ms']:.3f} ms | TTNN: {ttnn_stats['mean_ms']:.3f} ms")
```

Run this on the same input tensors and the same hardware, with the T3K mesh initialized, to produce a fair comparison. The 100 warmup iterations allow PyTorch's JIT and any lazy initialization to settle before measurement begins. Report p99 in addition to mean, because the CPU path has high variance from OS scheduling and cache effects.

Note: do not express the ratio as a fixed multiplier until this measurement is run on the target model configuration. The 2–3 orders-of-magnitude estimate is based on the structural analysis above; the actual ratio depends on `hidden_dim`, `intermediate_dim`, `num_experts`, and the T3K mesh's realized `TTNNExperts` throughput.

---

## 3. Migration Checklist: CPU Path → TTNNGlm4MoeExpertLayers

This checklist is for engineers migrating `TTNNGlm4MoeMoE` from its current CPU expert path to the intended TTNN path. The immediate target is activating `TTNNGlm4MoeExpertLayers` (the class already referenced in the dead branch at `moe.py:L571–L576`), not a generic `TTNNExperts` swap. See the weight format note in Section 4 before beginning.

**Step 1: Confirm `TTNNGlm4MoeExpertLayers` is available and parameterizable for the target model.**

`TTNNGlm4MoeExpertLayers.from_parameters` expects `gate_up_proj` and `down_proj` tensors and a `num_experts_off_chip` argument. Confirm the class exists and its `from_parameters` signature is compatible with the tensors available in `Glm4MoeNaiveMoeHybrid.__init__`. Run a standalone construction test before modifying the live training or inference path:

```python
from moe import TTNNGlm4MoeExpertLayers

# Use dummy tensors matching the model's shapes
gate_up = torch.zeros(num_experts, 2 * intermediate_dim, hidden_dim)
down = torch.zeros(num_experts, hidden_dim, intermediate_dim)
layers = TTNNGlm4MoeExpertLayers.from_parameters(gate_up, down, num_experts_off_chip=32)
```

**Step 2: Verify weight format compatibility.**

`Glm4MoeNaiveMoeHybrid` holds a fused `gate_up_proj` of shape `[num_experts, 2*intermediate_dim, hidden_dim]`. The TTNN path calls `TTNNGlm4MoeExpertLayers.from_parameters(old_layer.gate_up_proj, old_layer.down_proj, ...)`, which passes the fused tensor directly. Confirm that `TTNNGlm4MoeExpertLayers.from_parameters` handles the fused format internally. If it expects split weights, apply the split described in Section 4 before calling `from_parameters`.

Also confirm that `old_layer.down_proj` has shape `[num_experts, hidden_dim, intermediate_dim]` — matching the convention in `Glm4MoeNaiveMoe` (`moe.py:L324–L363`) — rather than the transposed form.

**Step 3: Enable the TTNN branch by setting `ttnn = True` at `moe.py:L569`.**

```python
# moe.py:L569 — change:
ttnn = False
# to:
ttnn = True
```

This is the minimal source change. When `ttnn = True`, `__init__` selects `TTNNGlm4MoeExpertLayers.from_parameters` and deletes the original weight tensors from `old_layer`. Verify that the weight deletion does not break any other reference to `old_layer.gate_up_proj` or `old_layer.down_proj` in the calling code.

Consider replacing the boolean flag with a proper configuration parameter (e.g., a field on `old_layer.config`) so that the TTNN path can be enabled or disabled without source edits.

**Step 4: Validate numerically.**

Run both the CPU path and the TTNN path on the same input batch and compare outputs:

```python
import torch

def validate_expert_output(cpu_experts, ttnn_experts, hidden_states, topk_indices, topk_weights, tol_frac=0.005):
    """
    Compare CPU and TTNN expert outputs.
    tol_frac: maximum allowed max-absolute-error as a fraction of output norm.
    """
    with torch.no_grad():
        cpu_out = cpu_experts(hidden_states, topk_indices, topk_weights)
        ttnn_out = ttnn_experts(hidden_states, topk_indices, topk_weights)

    ttnn_out_cpu = ttnn_out.cpu() if hasattr(ttnn_out, 'cpu') else ttnn_out
    diff = (cpu_out - ttnn_out_cpu).abs().max().item()
    norm = cpu_out.norm().item()
    frac = diff / (norm + 1e-8)

    print(f"Max absolute error: {diff:.6f} | Output norm: {norm:.6f} | Relative: {frac:.4%}")
    assert frac < tol_frac, f"Numeric mismatch exceeds tolerance: {frac:.4%} > {tol_frac:.4%}"
    return frac
```

The tolerance of 0.5% of output norm is appropriate for HiFi2 math fidelity. At HiFi4, expect tighter agreement. Run validation on at least 10 distinct input batches, including a batch where all top-k experts are different and a batch where some tokens share experts.

**Step 5: Run the detection harness.**

After enabling `ttnn = True` and passing numeric validation, run the full detection harness from `fallback_detection_and_testing.md`:

- The module-level audit should return `findings["cpu"] == []`.
- The host-device transfer hook should record zero `ttnn.to_torch` / `ttnn.from_torch` calls during a decode-time forward pass (after warmup).
- A Tracy or TTNN profiler trace should contain `TTNNGlm4MoeExpertLayers`-related ops; the absence of any TTNN expert kernel ops is a regression indicator.

---

## 4. Weight Format Note

`Glm4MoeExpertLayersTorch.forward` (`moe.py:L551–L556`) uses a combined weight tensor where gate and up projections are concatenated along the first dimension:

```python
gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
```

For expert `e`, `self.gate_up_proj[e]` has shape `[2*intermediate_dim, hidden_dim]`. The `.chunk(2, dim=-1)` call splits the *output* dimension of the linear projection in half: the first `intermediate_dim` output features are the gate, and the second `intermediate_dim` output features are the up projection. Because `nn.functional.linear` computes `x @ W.T`, the weight matrix is stored as `[2*intermediate_dim, hidden_dim]`, where:

- Rows `0 : intermediate_dim` — gate projection weights (equivalent to `w1` in the three-weight format)
- Rows `intermediate_dim : 2*intermediate_dim` — up projection weights (equivalent to `w3` in the three-weight format)

If `TTNNGlm4MoeExpertLayers.from_parameters` or any downstream TTNN class requires separate gate and up weights, split as follows:

```python
w1 = gate_up_proj[:, :intermediate_dim, :]    # shape: [num_experts, intermediate_dim, hidden_dim]
w3 = gate_up_proj[:, intermediate_dim:, :]    # shape: [num_experts, intermediate_dim, hidden_dim]
w2 = down_proj                                 # shape: [num_experts, hidden_dim, intermediate_dim]
```

This naming matches the convention used in `TTNNExperts` (w1=gate, w3=up, w2=down). If `TTNNGlm4MoeExpertLayers.from_parameters` accepts the fused `gate_up_proj` tensor directly and handles the split internally, no pre-processing is required — confirm by inspecting its implementation before the migration.

---

**Previous:** [Chapter 5 Index](../ch5_profiling_methodology/index.md) | **Next:** [Fallback Detection and Testing](fallback_detection_and_testing.md)
