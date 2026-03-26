# Profiling Plan: TTNNMoE Module Performance

**Date:** 2026-03-26
**Target:** TTNNBailingMoE module used in `test_ling_mini_2_0.py`
**Hardware:** T3K (1x8 Wormhole mesh)

---

## 1. Problem Description

The TTNNMoE module (specifically TTNNBailingMoE for Ling-mini-2.0) is a Mixture of Experts implementation that replaces the model's MLP layers. It involves complex data movement across 8 devices (all-to-all dispatch/combine, all-gather, reduce-scatter) combined with expert computation (sparse matmuls). We need to:

1. **Measure per-op device kernel durations** to identify the slowest operations
2. **Quantify the split between compute, data movement, and host overhead**
3. **Identify optimization targets** (e.g., routing overhead, CCL op efficiency, sparse_matmul utilization)
4. **Establish a baseline** for future optimization work

The existing test (`test_ling_mini_2_0.py`) already includes DispatchManager host-level timing, but we need device-level profiling to see inside the TTNN ops.

---

## 2. TTNNMoE Architecture

### Class Hierarchy

```
TTNNBailingMoE (moe.py:1499) inherits from TTNNMoE (moe.py:1346)
```

TTNNBailingMoE adapts BailingMoeV2SparseMoeBlock (Ling-mini-2.0's native MoE) to the TTNNMoE interface by:
- Adapting config attributes (num_experts -> n_routed_experts, expert_bias -> e_score_correction_bias)
- Consolidating ModuleList experts into 3D weight tensors for TTNNExperts
- Adapting gate structure for TTNNGlm4MoeTopkRouter

### Forward Pass Op Sequence (TTNNMoE.forward, line 1412)

The forward pass executes the following sequence on T3K:

#### Phase 1: All-Gather (revert tensor parallelism)
- `ttnn.experimental.all_gather_async(x, dim=-1)` -- gather hidden states across 8 devices

#### Phase 2: Gate Routing
- `ttnn.to_layout(x, TILE_LAYOUT)` -- layout conversion if needed
- `ttnn.typecast(x, float32)` -- upcast for precision
- `ttnn.linear(x_f32, gate_weight)` -- router logits matmul (HiFi4, fp32 accumulation)
- `ttnn.typecast(router_logits_f32, bfloat16)` -- downcast for router
- `ttnn.reshape(router_logits)` -- reshape for router input

#### Phase 3: TTNNMoERouterDecode (route_tokens_to_experts, line 891)
This is a complex routing module with ~30+ TTNN ops per call:
- `ttnn.sigmoid(logits)` -- score computation
- `ttnn.typecast` (multiple) -- float32/bfloat16 conversions
- `ttnn.add(scores, bias)` -- bias correction
- **3-pass topk centering** (when n_group <= topk_group):
  - Pass 1: `ttnn.topk(k+1)` on bfloat16 -> coarse threshold
  - `ttnn.slice` -> extract threshold -> `ttnn.sub` -> center scores
  - Pass 2: `ttnn.topk(k+1)` on centered scores -> refined threshold
  - `ttnn.sub` -> double-center scores
  - Pass 3: `ttnn.topk(k)` -> final expert selection
- `ttnn.gather(scores, index=topk_idx)` -- gather original sigmoid scores
- `ttnn.sum` + `ttnn.div` -- normalize weights
- `ttnn.mul(weights, scale)` -- apply routing scale
- `ttnn.reshape` (multiple) -- output reshaping
- **Host round-trip**: `_to_torch_any` for sorting (lines 806-813) -- converts to torch, sorts, converts back

#### Phase 4: TTNNExperts (experts.forward, line 1160)
- `ttnn.typecast(topk_indices, uint16)` -- index type conversion
- `ttnn.pad` (if needed) -- pad to SPARSITY_BLOCK_SIZE=32 multiple
- `ttnn.to_layout(x, ROW_MAJOR)` -- prepare for all-to-all
- `ttnn.reshape` (multiple) -- dimension adjustments
- `ttnn.all_to_all_dispatch(x, indices, mapping)` -- distribute tokens to expert devices
- `ttnn.reshape` + `ttnn.to_layout(TILE_LAYOUT)` -- prepare for sparse matmul
- `ttnn.repeat(remap_topk_mask)` + `ttnn.moe_expert_token_remap` -- generate sparsity tensor
- `ttnn.sparse_matmul(x, w1_proj, sparsity)` -- gate projection (HiFi2, fp32 acc, packer_l1_acc)
- `ttnn.sparse_matmul(x, w3_proj, sparsity)` -- up projection
- `ttnn.silu(w1_out)` -- activation
- `ttnn.mul(w1_activated, w3_out)` -- element-wise multiply
- `ttnn.sparse_matmul(intermediate, w2_proj, sparsity)` -- down projection
- `ttnn.permute` + `ttnn.reshape` -- reshape expert output
- `ttnn.to_layout(ROW_MAJOR)` + `ttnn.reshape` -- prepare for combine
- `ttnn.all_to_all_combine(output, metadata, mapping)` -- gather results back
- `ttnn.reshape` + `ttnn.to_layout(TILE_LAYOUT)` -- post-combine layout
- Weight application: `ttnn.unsqueeze` + `ttnn.repeat` + `ttnn.permute` + `ttnn.to_layout` + `ttnn.mul` -- apply expert weights
- `ttnn.sum(weighted_output, dim=0)` -- sum over experts

#### Phase 5: Reduce-Scatter
- `ttnn.mul(routed_out, 1/n_rs)` -- pre-scale for reduce
- `ttnn.experimental.reduce_scatter_minimal_async(routed_output)` -- scatter final output

#### Phase 6: Shared Experts (TTNNGlm4MoeMLP)
- `gate_proj(residual)` -- TTNNLinearSilu (matmul + silu fused)
- `up_proj(residual)` -- TTNNLinearIColShardedWRowSharded (matmul)
- `ttnn.mul(gate_out, up_out)` -- element-wise multiply
- `down_proj(intermediate)` -- TTNNLinearIColShardedWRowSharded (matmul)

#### Phase 7: Final Addition
- `ttnn.add(routed_output, shared_output)` -- combine routed + shared
- `ttnn.squeeze(output, 1)` -- remove experts dimension

### Estimated Op Count Per Forward Pass

| Category | Approximate Op Count | Notes |
|----------|---------------------|-------|
| Router (TTNNMoERouterDecode) | 30-40 | Many small ops: topk, typecast, sigmoid, reshape, slice, gather |
| Expert compute (TTNNExperts) | 20-25 | sparse_matmul x3, all_to_all x2, reshapes, weight application |
| Gate matmul | 3-5 | linear + typecasts |
| CCL ops | 3 | all_gather, all_to_all_dispatch, all_to_all_combine, reduce_scatter |
| Shared experts | 6-8 | 3 matmuls + silu + mul |
| Miscellaneous | 5-10 | Layout conversions, reshapes, squeeze |
| **Total** | **~70-90 ops** | Per MoE layer forward pass |

### Model Configuration (Ling-mini-2.0)

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 (from BailingMoeV2 config) |
| moe_intermediate_size | 1408 |
| num_experts (n_routed_experts) | 64 |
| num_experts_per_tok | 8 |
| n_shared_experts | 1 |
| n_group | 4 |
| topk_group | 2 |
| num_hidden_layers | ~28 (MoE layers start after first_k_dense_replace) |

---

## 3. Available Profiling Methods

### Method A: DispatchManager Host-Level Timing (Built-in)

**Already in test_ling_mini_2_0.py** (lines 130-139):
```python
DispatchManager.clear_timings()
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, past_key_values=paged_cache)
DispatchManager.save_stats_to_file("ling_mini_2_0_paged_attention_timing_stats.csv")
```

**What it captures:**
- Per-module forward call durations (wall-clock, host-side)
- TTNN op durations within module wrappers
- Torch wrapper overhead

**Limitations:**
- No device kernel durations (only host-perceived latency)
- Cannot distinguish between compute-bound and memory-bound ops
- No per-RISC breakdown
- Cannot see inside individual TTNN ops

### Method B: Device-Level Profiling with TT_METAL_DEVICE_PROFILER

**Environment variables:**
```bash
TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_CPP_POST_PROCESS=1
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000
```

**What it captures:**
- `cpp_device_perf_report.csv` with per-op:
  - OP NAME (e.g., `ttnn::matmul`, `ttnn::all_gather`, `ttnn::sparse_matmul`)
  - DEVICE KERNEL DURATION [ns]
  - Per-RISC durations (BRISC, NCRISC, TRISC0/1/2, ERISC)
  - OP TO OP LATENCY [ns] (host dispatch gap)
  - CORE COUNT

**Why 4000 for buffer:** TTNNMoE has ~70-90 ops per layer. With ~28 MoE layers and 128 decode tokens, the total ops could be:
- 28 layers x 90 ops x 128 tokens x 5 RISCs x 2 markers = ~3.2M markers
- This WILL overflow. Must reduce `max_new_tokens`.

### Method C: Tracy GUI Capture

**For visual timeline inspection:**
- Start `tracy-capture` in one terminal
- Run test with `python3 -m tracy -v -r -p -m pytest`
- Produces `.tracy` file for GUI analysis

**Best for:**
- Visual identification of pipeline bubbles
- Understanding overlap between host dispatch and device execution
- Identifying synchronization points

### Method D: Combined (Recommended)

Use Method A (DispatchManager) for high-level module timing + Method B (device profiler) for device-level op breakdown. Run them in separate passes to avoid interference.

---

## 4. Step-by-Step Profiling Plan

### Step 1: Create a Standalone MoE Profiling Test

Create a dedicated profiling test that isolates the MoE module. This avoids profiling the attention layers and reduces buffer overflow risk.

**File:** `models/experimental/tt_symbiote/tests/test_profile_ttnn_moe.py`

**Design:**
```python
# Key parameters:
# - Load Ling-mini-2.0 model
# - Replace only MoE layers with TTNNBailingMoE (skip attention replacement)
# - Run warmup (3 iterations)
# - Profile 8 decode iterations
# - Save DispatchManager stats
```

**Why standalone:**
- The full model test profiles ALL layers (attention + MoE + embedding + LM head)
- A standalone test isolating just the MoE forward pass reduces the op count per iteration
- Fewer ops = less buffer pressure = complete profiling data

**Test structure:**
1. Load model, replace MoE layers only
2. Preprocess and move weights
3. Warmup run (2-3 tokens, not profiled)
4. Clear DispatchManager timings
5. Generate 8 tokens with profiling
6. Save DispatchManager stats to CSV
7. Print per-module timing summary

### Step 2: Run Host-Level Profiling (DispatchManager)

```bash
cd /home/ttuser/salnahari/tt-metal
source /home/ttuser/salnahari/tt_bashrc

pytest models/experimental/tt_symbiote/tests/test_profile_ttnn_moe.py -sv
```

**Output:** `ttnn_moe_timing_stats.csv` with per-module timing data.

**Analysis:**
- Identify which sub-module dominates: gate routing, experts, shared_experts, or CCL ops
- Calculate per-iteration wall-clock time
- Compare warmup vs steady-state

### Step 3: Run Device-Level Profiling

**Option 3a: Device profiler with C++ post-processing (recommended first)**

```bash
cd /home/ttuser/salnahari/tt-metal
source /home/ttuser/salnahari/tt_bashrc

TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 \
pytest models/experimental/tt_symbiote/tests/test_profile_ttnn_moe.py -sv
```

**Output:** `generated/profiler/.logs/cpp_device_perf_report.csv`

**Option 3b: Synchronous profiling for accurate per-op isolation**

```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
TT_METAL_PROFILER_SYNC=1 \
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 \
pytest models/experimental/tt_symbiote/tests/test_profile_ttnn_moe.py -sv
```

Note: `TT_METAL_PROFILER_SYNC=1` serializes execution so wall-clock is slower, but device kernel durations are measured without pipeline overlap.

**Option 3c: Full Tracy GUI capture**

Terminal 1:
```bash
cd /home/ttuser/salnahari/tt-metal
tt_metal/third_party/tracy/capture/build/unix/tracy-capture -o ttnn_moe_profile.tracy -f
```

Terminal 2:
```bash
cd /home/ttuser/salnahari/tt-metal
source /home/ttuser/salnahari/tt_bashrc

TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 \
python3 -m tracy -v -r -p -m pytest -- \
  models/experimental/tt_symbiote/tests/test_profile_ttnn_moe.py -sv
```

### Step 4: Run the Full Model Test with Profiling

After standalone MoE profiling, also profile the full model to understand MoE's contribution to end-to-end latency.

```bash
cd /home/ttuser/salnahari/tt-metal
source /home/ttuser/salnahari/tt_bashrc

# Modify test_ling_mini_2_0.py: change max_new_tokens from 128 to 8
# Then run:
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 \
pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py::test_ling_mini_2_0 -sv
```

**IMPORTANT:** Reduce `max_new_tokens` from 128 to 8 for profiling to avoid DRAM buffer overflow. With ~28 MoE layers + attention layers, each decode generates thousands of ops.

### Step 5: Analyze Results

#### 5a: DispatchManager CSV Analysis

```python
import pandas as pd
df = pd.read_csv("ttnn_moe_timing_stats.csv")

# Per-module breakdown
module_summary = df.groupby("func_name").agg({
    "duration_s": ["sum", "mean", "count", "max"]
}).sort_values(("duration_s", "sum"), ascending=False)

# Identify top bottlenecks
print(module_summary.head(20))
```

#### 5b: Device Profiler CSV Analysis

```python
import pandas as pd
df = pd.read_csv("generated/profiler/.logs/cpp_device_perf_report.csv")

# Per-op-type summary
op_summary = df.groupby("OP NAME").agg({
    "DEVICE KERNEL DURATION [ns]": ["sum", "mean", "count"],
    "OP TO OP LATENCY [ns]": ["sum", "mean"],
}).sort_values(("DEVICE KERNEL DURATION [ns]", "sum"), ascending=False)

print(op_summary.head(20))

# Compute vs memory bound classification
# If TRISC1 > NCRISC: compute-bound
# If NCRISC > TRISC1: memory/NOC-bound
df["bottleneck"] = df.apply(
    lambda row: "compute" if row.get("DEVICE TRISC1 KERNEL DURATION [ns]", 0) >
                             row.get("DEVICE NCRISC KERNEL DURATION [ns]", 0)
    else "memory", axis=1
)
```

#### 5c: Key Questions to Answer

1. **What fraction of MoE forward time is routing vs expert compute vs CCL?**
   - Group ops by phase: gate (linear), router (sigmoid, topk, gather), experts (sparse_matmul, all_to_all), shared (linear), CCL (all_gather, reduce_scatter)

2. **Is the 3-pass topk routing a bottleneck?**
   - The router has ~30 ops, many of which are small. Sum their device kernel durations and op-to-op latencies.

3. **How efficient are the sparse_matmul ops?**
   - Compare DEVICE KERNEL DURATION to PM IDEAL (if available)
   - Check FPU UTIL for the sparse_matmul ops
   - Compare TRISC1 (math) vs NCRISC (data movement) durations

4. **What is the all_to_all dispatch/combine overhead?**
   - These are multi-device CCL ops. Their latency depends on inter-device bandwidth and synchronization.

5. **How much host overhead exists?**
   - Sum OP TO OP LATENCY across all ops for total dispatch overhead
   - Compare total dispatch overhead to total device kernel time

6. **Is the router host round-trip (_to_torch_any sorting) significant?**
   - This is a device->host->device transfer in the router (lines 806-813)
   - Will appear as a gap in device activity (high op-to-op latency)

---

## 5. Success Criteria

A successful profiling run produces:

### Data Quality
- [ ] DispatchManager CSV with at least 5 steady-state decode iterations per MoE layer
- [ ] cpp_device_perf_report.csv with non-zero DEVICE KERNEL DURATION values
- [ ] No "profiler DRAM buffers were full" warnings in test output
- [ ] All expected op types present: sparse_matmul, all_to_all_dispatch, all_to_all_combine, all_gather_async, reduce_scatter_minimal_async, sigmoid, topk, linear/matmul

### Analysis Deliverables
- [ ] Per-phase time breakdown: gate routing / expert dispatch / expert compute / expert combine / weight application / reduce-scatter / shared experts
- [ ] Top-10 ops by total device kernel duration
- [ ] Compute-bound vs memory-bound classification for each op type
- [ ] Host dispatch overhead as fraction of total time
- [ ] Identification of the single largest optimization opportunity

### Performance Baseline
- [ ] Steady-state per-decode MoE forward time (wall-clock)
- [ ] Steady-state per-decode MoE forward time (device kernel sum)
- [ ] Ratio of device compute time to wall-clock time (efficiency metric)

---

## 6. Expected Bottlenecks (Hypotheses)

Based on architectural analysis, the likely bottleneck ranking is:

1. **Router overhead (TTNNMoERouterDecode):** ~30+ small TTNN ops with host round-trip for sorting. Each op has dispatch overhead. The 3-pass topk centering adds significant op count. The `_to_torch_any` host round-trip (lines 806-813) forces device->host->device data movement.

2. **CCL communication (all_to_all + all_gather + reduce_scatter):** Three separate multi-device communication ops per forward pass. Each synchronizes across 8 devices.

3. **Expert weight application:** The weight application sequence (unsqueeze, repeat, permute, mul, sum) involves ~6 ops that manipulate tensor dimensions and may not be efficiently fused.

4. **sparse_matmul efficiency:** With 64 experts distributed across 8 devices (8 experts/device), sparsity patterns may lead to unbalanced workloads across cores.

5. **Layout conversions:** Multiple `to_layout` and `reshape` ops between ROW_MAJOR and TILE_LAYOUT for CCL compatibility.

---

## 7. Risk Mitigation

### Buffer Overflow Risk
- **Mitigation:** Use `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000` (4x default)
- **Mitigation:** Limit decode tokens to 8 in profiling test
- **Mitigation:** Create standalone MoE test (fewer ops per iteration than full model)
- **Fallback:** If 4000 still overflows, use `TT_METAL_PROFILER_MID_RUN_DUMP=1`

### Model Loading Time
- **Issue:** Ling-mini-2.0 model download + weight preprocessing takes significant time
- **Mitigation:** Ensure model is cached locally before profiling runs
- **Mitigation:** Weight preprocessing is one-time per test session

### Multi-Device Profiling Complexity
- **Issue:** T3K has 8 devices; profiler data is per-device
- **Mitigation:** cpp_device_perf_report.csv includes DEVICE ID column for filtering
- **Analysis:** Focus on device 0 first, then compare across devices for load balance

---

## 8. Environment Setup Reference

```bash
# Full environment setup for profiling
cd /home/ttuser/salnahari/tt-metal
source /home/ttuser/salnahari/tt_bashrc

# This sets:
# - TT_METAL_HOME=/home/ttuser/salnahari/tt-metal/
# - PYTHONPATH includes TT_METAL_HOME
# - WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
# - ARCH_NAME=wormhole_b0
# - MESH_DEVICE=T3K
# - Activates Python venv at tt-metal/python_env/bin/activate

# Verify environment
echo "TT_METAL_HOME=$TT_METAL_HOME"
echo "MESH_DEVICE=$MESH_DEVICE"
python3 -c "import ttnn; print(f'TTNN version: {ttnn.__version__}')"
```
