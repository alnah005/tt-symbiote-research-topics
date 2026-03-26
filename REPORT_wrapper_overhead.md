# TT-Symbiote Wrapper Overhead Report: Ling-mini-2.0 Decode Path

**Date:** 2026-03-26
**Model:** Ling-mini-2.0 (BailingMoeV2ForCausalLM), 20 layers, 64 experts, T3K (1x8 mesh)
**Data Source:** `ling_mini_2_0_paged_attention_timing_stats.csv` — iteration 10 (steady-state decode)
**Source Files:** `core/tensor.py`, `core/module.py`, `core/run_config.py`

---

## 1. Executive Summary

A single decode token through the Ling-mini-2.0 model takes **~942ms wall clock** (measured with DispatchManager host-level timing). This includes significant overhead from:

| Category | Time (ms) | % of Total | Description |
|----------|-----------|-----------|-------------|
| Wrapper/Torch ops | 309.1 | 32.8% | TorchTTNNTensor wrapping, unwrapping, device sync |
| Leaf TTNN forward (host dispatch) | 466.6 | 49.5% | Actual TTNN op submission to device |
| TTNN preprocess/move (cached, no-ops) | 1.5 | 0.2% | Already preprocessed; near-zero |
| Python overhead (prints, control flow) | 165.1 | 17.5% | 363 print statements + tree_map + isinstance checks |
| **Total** | **942.4** | **100%** | |

**Key Finding:** Over **50% of decode time is wrapper/Python overhead**, not actual TTNN op dispatch. The single largest contributor is `_unwrap_to_torch` (76.6ms, 8.1%), which forces **42 device synchronization points** per decode token where the host blocks waiting for device output. The `compose_transforms` input wrapping adds another 158.9ms (16.9%) across 594 calls.

**Note on print overhead:** The `NormalRun.module_run()` at line 557 of `run_config.py` prints a line for every TTNN module call (363 calls per token). This alone accounts for an estimated 100-200ms.

---

## 2. Per-Module Breakdown Table

### 2a. Wrapper Operations Per Decode Token

| Operation | Calls | Total (ms) | Avg (ms) | Purpose |
|-----------|-------|-----------|----------|---------|
| `wrap_to_torch_ttnn_tensor_to_ttnn_wrap__set_device_wrap` | 594 | 158.89 | 0.267 | Input arg wrapping: wrap torch tensor -> TorchTTNNTensor, convert to ttnn, ensure on device |
| `_unwrap_to_torch` | 84 | 76.64 | 0.912 | Output unwrapping: ttnn.to_torch (DEVICE SYNC - blocks until device output ready) |
| `wrap_to_torch_ttnn_tensor` | 297 | 60.59 | 0.204 | Output wrapping: wrap forward() return values as TorchTTNNTensor |
| `_set_distributed_config` | 297 | 8.57 | 0.029 | Set distributed tensor config on output tensors |
| `aten::add.Tensor` | 40 | 4.28 | 0.107 | Residual connections (runs on CPU via PyTorch) |
| `aten::slice.Tensor` | 1 | 0.09 | 0.086 | Input slicing |
| `aten::mul.Tensor` | 1 | 0.07 | 0.069 | MoE gate multiplication |
| **Total** | **1314** | **309.13** | | |

### 2b. TTNN Module Forward Times (Leaf Modules Only)

These are the actual TTNN op dispatch times — the "useful work" of submitting operations to the device.

| Module Type | Calls | Total (ms) | Avg (ms) | Description |
|-------------|-------|-----------|----------|-------------|
| TTNNMoERouterDecode | 19 | 136.15 | 7.166 | Router: gate matmul + 3-pass topk + sort (many small ops) |
| TTNNLinearIColShardedWRowSharded | 81 | 114.09 | 1.408 | Column-sharded linear (Q proj, gate proj, etc.) |
| TTNNExperts | 19 | 101.89 | 5.363 | Expert computation: all_to_all + sparse_matmul x3 + silu |
| TTNNLinearIReplicatedWColSharded | 60 | 55.91 | 0.932 | Replicated-input linear (K, V, O proj) |
| TTNNLinearActivation | 19 | 41.81 | 2.200 | Linear with SiLU activation (shared expert up proj) |
| TTNNRMSNorm | 40 | 16.34 | 0.409 | RMS normalization |
| TTNNSilu | 1 | 0.46 | 0.457 | SiLU activation (layer 0 only) |
| **Total** | **239** | **466.65** | | |

### 2c. Overhead Ratio by Module Type

| Module Type | Calls | TorchModules (ms) | Forward (ms) | Overhead (ms) | Overhead per call (ms) | Overhead % |
|-------------|-------|-------------------|-------------|--------------|----------------------|------------|
| TTNNBailingMoEAttention | 20 | 383.6 | 309.5 | 74.1 | 3.7 | 19.3% |
| TTNNBailingMoE | 19 | 453.4 | 436.3 | 17.1 | 0.9 | 3.8% |
| TTNNLinearIColShardedWRowSharded | 81 | 141.7 | 114.1 | 27.6 | 0.3 | 19.5% |
| TTNNLinearIReplicatedWColSharded | 60 | 77.8 | 55.9 | 21.9 | 0.4 | 28.2% |
| TTNNRMSNorm | 40 | 32.1 | 16.3 | 15.8 | 0.4 | 49.1% |
| TTNNMoERouterDecode | 19 | 143.9 | 136.1 | 7.8 | 0.4 | 5.4% |
| TTNNGlm4MoeMLP | 19 | 126.2 | 119.3 | 6.9 | 0.4 | 5.5% |
| TTNNExperts | 19 | 108.0 | 101.9 | 6.1 | 0.3 | 5.7% |
| TTNNLinearActivation | 19 | 47.0 | 41.8 | 5.2 | 0.3 | 11.0% |
| TTNNSilu | 1 | 0.6 | 0.5 | 0.2 | 0.2 | 27.5% |

Note: TTNNRMSNorm has the highest overhead percentage (49.1%) because its actual TTNN forward is very fast (0.4ms avg) but the wrapping/unwrapping cost is fixed per call.

---

## 3. Top 20 Worst Offenders

These are individual module instances with the highest per-call wrapper overhead (excluding container modules that have no own forward):

| # | Module Instance | Type | Total (ms) | Forward (ms) | Overhead (ms) |
|---|----------------|------|-----------|-------------|--------------|
| 1 | model.layers.0.attention | TTNNBailingMoEAttention | 34.1 | 14.9 | 19.2 |
| 2 | model.layers.8.attention.key_layernorm | TTNNRMSNorm | 4.0 | 0.5 | 3.6 |
| 3 | model.layers.14.attention | TTNNBailingMoEAttention | 20.8 | 17.5 | 3.2 |
| 4 | model.layers.19.attention | TTNNBailingMoEAttention | 17.5 | 14.4 | 3.1 |
| 5 | model.layers.5.attention | TTNNBailingMoEAttention | 20.3 | 17.3 | 3.0 |
| 6 | model.layers.18.attention | TTNNBailingMoEAttention | 17.9 | 14.9 | 3.0 |
| 7 | model.layers.13.attention | TTNNBailingMoEAttention | 17.1 | 14.2 | 3.0 |
| 8 | model.layers.8.attention | TTNNBailingMoEAttention | 20.3 | 17.4 | 3.0 |
| 9 | model.layers.3.attention | TTNNBailingMoEAttention | 16.9 | 14.0 | 3.0 |
| 10 | model.layers.12.attention | TTNNBailingMoEAttention | 17.2 | 14.2 | 3.0 |
| 11 | model.layers.10.attention | TTNNBailingMoEAttention | 16.9 | 14.0 | 2.9 |
| 12 | model.layers.2.attention | TTNNBailingMoEAttention | 16.9 | 14.0 | 2.9 |
| 13 | model.layers.6.attention | TTNNBailingMoEAttention | 16.9 | 14.0 | 2.9 |
| 14 | model.layers.1.attention | TTNNBailingMoEAttention | 17.2 | 14.4 | 2.8 |
| 15 | model.layers.4.attention | TTNNBailingMoEAttention | 16.8 | 13.9 | 2.8 |
| 16 | model.layers.9.attention | TTNNBailingMoEAttention | 17.1 | 14.3 | 2.8 |
| 17 | model.layers.7.attention | TTNNBailingMoEAttention | 16.8 | 14.0 | 2.8 |
| 18 | model.layers.11.attention | TTNNBailingMoEAttention | 28.9 | 26.1 | 2.7 |
| 19 | model.layers.16.attention | TTNNBailingMoEAttention | 16.3 | 13.6 | 2.7 |
| 20 | model.layers.15.attention | TTNNBailingMoEAttention | 16.4 | 13.8 | 2.7 |

TTNNBailingMoEAttention dominates the worst offender list because it is a composite module with 6 child TTNN modules (Q, K, V projections + q_norm, k_norm RMSNorms + O projection), each requiring its own wrapping/unwrapping cycle.

---

## 4. Root Cause Analysis

### 4.1. `compose_transforms` (wrap_to_torch_ttnn_tensor + to_ttnn_wrap + set_device_wrap)

**What it does:** In `NormalRun.module_run()` (line 560 of `run_config.py`), every TTNNModule call applies `compose_transforms` to ALL input arguments via `tree_map`. This performs three transforms per tensor:
1. **`wrap_to_torch_ttnn_tensor`**: If the input is a plain `torch.Tensor`, wraps it in a `TorchTTNNTensor` subclass. If it's already a `TorchTTNNTensor`, returns it unchanged.
2. **`to_ttnn_wrap`**: Calls `.to_ttnn` property which calls `ttnn.from_torch()` to convert to a TTNN tensor (if not already converted). This involves CPU-to-device data transfer.
3. **`set_device_wrap`**: Ensures the TTNN tensor is on the correct device via `ttnn.to_device()`.

**Why it's called 594 times:** Each of the 297+ TTNNModule calls per token has at least 2 input tensors (the hidden state + at least one kwarg). 297 modules * 2 args = 594 transform calls.

**Why it's expensive (avg 0.267ms/call):** The `to_ttnn_wrap` step calls `ttnn.from_torch()` which:
- Converts torch tensor to TTNN format
- For multi-device (T3K), applies `mesh_mapper` to shard/replicate across 8 devices
- Copies data from host CPU to device DRAM
- Each call involves Python->C++ FFI overhead

**Can it be avoided?** YES. Once a tensor is wrapped and on-device, passing it through child modules should NOT re-wrap it. Currently, every child module call re-applies `compose_transforms` even though the tensor is already a `TorchTTNNTensor` with `ttnn_tensor` set. The `wrap_to_torch_ttnn_tensor` becomes a no-op, but `to_ttnn_wrap` and `set_device_wrap` still involve isinstance checks and attribute accesses.

### 4.2. `_unwrap_to_torch` (Device Synchronization)

**What it does:** Converts a `TorchTTNNTensor` back to a plain torch tensor by calling `ttnn.to_torch()`. This is a **blocking synchronization point** — the host waits for ALL pending device operations to complete, then reads back the result tensor from device DRAM to host CPU.

**Why it's called 84 times:** There are 42 pairs of `_unwrap_to_torch` calls per decode token:
- 1 pair at the top level (ForCausalLM output)
- 20 pairs after attention output (hidden_states + residual for `aten::add.Tensor`)
- 19 pairs after MoE output (hidden_states + residual for `aten::add.Tensor`)
- 1 pair for layer 0 MLP special case
- 1 pair for final output

Each pair unwraps 2 tensors: the module output and the residual. This is needed because the residual add (`hidden_states = residual + attn_output`) is done in PyTorch (`aten::add.Tensor`) which requires plain torch tensors.

**Why it's expensive (avg 0.912ms/call, total 76.6ms):**
- `ttnn.to_torch()` forces device synchronization — the host blocks until all queued ops finish
- Data transfer from device DRAM back to host CPU over PCIe
- For multi-device (T3K), involves `mesh_composer` to concatenate shards from 8 devices
- The first call (14.49ms) is particularly expensive as it waits for ALL initial ops to complete

**Can it be avoided?** YES. The residual add should be done in TTNN (`ttnn.add`) on-device, eliminating the need to transfer data back to host. This is the **single most impactful optimization** because each pair forces a full pipeline stall.

### 4.3. `wrap_to_torch_ttnn_tensor` (Output Wrapping)

**What it does:** In `post_process_ttnn_module_output()` (line 398-402 of `run_config.py`), every TTNNModule's output is wrapped in `TorchTTNNTensor` and has its distributed tensor config set. This ensures the output can participate in `__torch_dispatch__` for subsequent operations.

**Why 297 calls:** One per TTNNModule call output tensor (some return multiple tensors).

**Why it's expensive (avg 0.204ms/call, total 60.59ms):** Creating a `TorchTTNNTensor` involves:
- `torch.Tensor._make_wrapper_subclass()` call with shape/stride/dtype extraction
- `get_default_distributed_tensor_config()` lookup
- Setting `ttnn_tensor`, `elem`, and distributed config attributes
- Python object creation overhead

### 4.4. Print Statement Overhead

**What it does:** `NormalRun.module_run()` line 557: `print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")`

**Impact:** 363 print calls per decode token. With terminal I/O, each print can take 0.3-1ms, adding an estimated 100-200ms of pure I/O overhead. This is captured in the "Python overhead" category (165.1ms).

### 4.5. `tree_map` and isinstance Overhead

**What it does:** `tree_map` recursively traverses argument trees (lists, tuples, dicts) applying transform functions. Each transform function performs multiple `isinstance` checks.

**Impact:** For 297 module calls with 2+ args each, the Python interpreter performs thousands of isinstance checks, function calls, and attribute lookups. This contributes to the unmeasured Python overhead.

---

## 5. Detailed Per-Layer Timing

Each decoder layer follows this pattern:
```
BailingMoeV2RMSNorm (input norm, ~0.3us Python)
  -> compose_transforms on args (2 calls, ~3us)
  -> TTNNBailingMoEAttention:
       compose_transforms (2 calls)
       6 child modules (Q, K, V, q_norm, k_norm, O proj), each with:
         compose_transforms (2 calls) + forward + output wrapping
       own forward code (~10.5ms avg)
  -> _unwrap_to_torch x2 (~1.4ms, DEVICE SYNC)
  -> aten::add.Tensor (~0.1ms, residual add on CPU)
BailingMoeV2RMSNorm (post norm, ~0.2us Python)
  -> compose_transforms on args (2 calls, ~0.8ms)
  -> TTNNBailingMoE:
       compose_transforms (2 calls)
       TTNNMoERouterDecode (gate + topk routing, ~7.2ms forward)
       TTNNExperts (dispatch + sparse_matmul + combine, ~5.4ms forward)
       TTNNGlm4MoeMLP (shared expert MLP, ~6.3ms forward)
       own forward code (~4.2ms avg)
  -> _unwrap_to_torch x2 (~1.4ms, DEVICE SYNC)
  -> aten::add.Tensor (~0.1ms, residual add on CPU)
```

### Per-Layer Breakdown (Layers 0-19)

| Layer | Total (ms) | Wrapper (ms) | Wrap Count | TTNN Fwd (ms) | Dispatch OH (ms) | aten (ms) |
|-------|-----------|-------------|-----------|--------------|-----------------|----------|
| 0 | 50.2 | 31.8 | 50 | 25.1 | 36.1 | 0.3 |
| 1 | 45.7 | 12.9 | 64 | 68.6 | 8.6 | 0.2 |
| 2 | 45.1 | 13.1 | 64 | 66.8 | 8.4 | 0.2 |
| 3 | 45.1 | 13.4 | 64 | 66.9 | 8.7 | 0.2 |
| 4 | 44.8 | 13.1 | 64 | 66.4 | 8.5 | 0.2 |
| 5 | 48.5 | 13.5 | 64 | 69.8 | 8.7 | 0.2 |
| 6 | 44.7 | 13.1 | 64 | 66.6 | 8.4 | 0.2 |
| 7 | 44.7 | 13.1 | 64 | 66.0 | 8.4 | 0.2 |
| 8 | 48.7 | 16.6 | 64 | 70.7 | 12.0 | 0.2 |
| 9 | 45.4 | 13.3 | 64 | 67.0 | 8.6 | 0.2 |
| 10 | 44.9 | 13.2 | 64 | 66.5 | 8.6 | 0.2 |
| 11 | 57.4 | 13.3 | 64 | 88.0 | 8.4 | 0.2 |
| 12 | 45.3 | 13.2 | 64 | 67.3 | 8.7 | 0.2 |
| 13 | 45.2 | 13.1 | 64 | 66.8 | 8.6 | 0.2 |
| 14 | 49.6 | 14.0 | 64 | 70.7 | 8.9 | 0.2 |
| 15 | 44.2 | 13.2 | 64 | 64.3 | 8.2 | 0.2 |
| 16 | 44.8 | 13.5 | 64 | 65.0 | 8.3 | 0.2 |
| 17 | 49.3 | 14.2 | 64 | 70.2 | 9.5 | 0.3 |
| 18 | 46.7 | 13.5 | 64 | 69.3 | 8.8 | 0.2 |
| 19 | 46.6 | 14.0 | 64 | 68.1 | 8.9 | 0.2 |
| **Total** | **936.8** | **289.1** | **1266** | **1330.1** | **203.4** | **4.3** |

Notes:
- "Total" is the DecoderLayer TorchModules time (includes all children)
- "Wrapper" is the sum of all Torch-backend wrapper ops within that layer
- "TTNN Fwd" is the sum of all TTNN _forward entries (includes nested composites, so will exceed "Total")
- Layer 0 has extra overhead (31.8ms wrapper vs ~13ms typical) due to first-call initialization
- Layer 11 has anomalously high TTNN Fwd (88.0ms vs ~67ms typical), likely a device scheduling variance

### Per-Layer Module Counts (Typical Layer)

| Item | Count per Layer | Count per Token (20 layers) |
|------|----------------|---------------------------|
| TTNNModule calls (TorchModules entries) | 18 | 360 |
| compose_transforms calls | 30 | 594 |
| wrap_to_torch_ttnn_tensor (output) | 15 | 297 |
| _set_distributed_config | 15 | 297 |
| _unwrap_to_torch | 4 | 84 |
| aten::add.Tensor | 2 | 40 |
| print statements | 18 | 363 |

### Device Synchronization Points (_unwrap_to_torch) Timing

| Sync Point | Count | Avg Time (ms) | Total (ms) | Description |
|-----------|-------|--------------|-----------|-------------|
| Initial (ForCausalLM output) | 2 | 7.26 | 14.51 | First sync, waits for pipeline to fill |
| After attention (per layer) | 40 | 0.73 | 29.05 | Residual add after attention |
| After MoE (per layer) | 38 | 0.74 | 28.17 | Residual add after MoE |
| Layer 0 MLP special | 4 | 1.22 | 4.88 | Layer 0 has BailingMoeV2MLP instead of MoE |
| **Total** | **84** | **0.91** | **76.61** | |

---

## 6. Recommendations

### Priority 1: Eliminate _unwrap_to_torch for Residual Adds (saves ~76ms)

**Problem:** 42 pairs of `_unwrap_to_torch` calls force device synchronization for CPU-side residual adds.

**Solution:** Perform residual adds in TTNN on-device:
```python
# BEFORE (in BailingMoeV2DecoderLayer):
hidden_states = residual + attn_output  # aten::add.Tensor on CPU

# AFTER:
hidden_states = ttnn.add(residual_ttnn, attn_output_ttnn)  # stays on device
```

This requires the BailingMoeV2DecoderLayer to be converted to a TTNNModule (or the residual add logic moved into the TTNN attention/MoE modules). This alone would eliminate 42 device synchronization points (76.6ms) and 40 aten::add calls (4.3ms).

**Estimated savings:** 80ms per decode token.

### Priority 2: Remove Print Statements (saves ~100-200ms)

**Problem:** `NormalRun.module_run()` prints a debug line for every module call (363 per token).

**Solution:** Gate prints behind a verbose flag or remove entirely:
```python
if NormalRun.verbose:
    print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
```

**Estimated savings:** 100-200ms per decode token.

### Priority 3: Skip compose_transforms for Already-Wrapped Tensors (saves ~100ms)

**Problem:** `compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap)` is applied to ALL input args for EVERY module call, even when tensors are already TorchTTNNTensors with ttnn_tensor set and on the correct device.

**Solution:** Add fast-path check:
```python
def compose_transforms_fast(device):
    def _fast_transform(e):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
        if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
            if e.ttnn_tensor.device() == device:
                return e  # Already wrapped, converted, and on device
        # Fall through to full transform
        return compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(device))(e)
    return _fast_transform
```

**Estimated savings:** Reduces 594 calls * 0.267ms = 158.9ms to ~594 * 0.03ms = ~18ms. Net savings ~140ms.

### Priority 4: Reduce Module Nesting / Flatten Dispatch (saves ~60ms)

**Problem:** Composite modules (TTNNBailingMoEAttention, TTNNBailingMoE) create a deep nesting of module_run calls. Each child module call triggers full wrapping/unwrapping even though parent already wrapped the inputs.

**Solution:** For the decode path, flatten composite modules into single TTNN forward calls that chain TTNN ops directly without wrapping/unwrapping intermediate results. The attention module could fuse Q/K/V projections + norms + SDPA into a single forward that avoids 6 child module_run cycles (saves 6 * (0.3ms wrap + 0.2ms output_wrap + 0.03ms set_config) = ~3.2ms per layer * 20 layers = 64ms).

### Priority 5: Optimize tree_map for Common Cases (saves ~10ms)

**Problem:** `tree_map` performs recursive traversal for every transform call, even when args are simple (single tensor or flat tuple).

**Solution:** Add fast path for common argument patterns:
```python
def tree_map_fast(fn, obj):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, tuple) and len(obj) <= 4:
        return tuple(fn(x) if isinstance(x, torch.Tensor) else x for x in obj)
    return tree_map(fn, obj)  # fallback to recursive version
```

### Priority 6: Batch Output Config Setting (saves ~8.5ms)

**Problem:** `_set_distributed_config` is called 297 times individually.

**Solution:** Set distributed config once at the TTNNModule output level, not per-tensor.

### Summary of Expected Savings

| Optimization | Savings (ms) | % of Current |
|-------------|-------------|-------------|
| P1: TTNN residual add | ~80 | 8.5% |
| P2: Remove prints | ~150 | 15.9% |
| P3: Fast-path compose_transforms | ~140 | 14.9% |
| P4: Flatten dispatch nesting | ~60 | 6.4% |
| P5: Fast tree_map | ~10 | 1.1% |
| P6: Batch config setting | ~8.5 | 0.9% |
| **Total** | **~450** | **~47%** |

Combined, these optimizations could reduce the host-side decode time from ~942ms to ~490ms. Further gains would come from trace capture (eliminating host dispatch overhead entirely for the TTNN ops) which could bring the total to the device execution time of the TTNN ops (~10-20ms based on previous Tracy profiling analysis).
