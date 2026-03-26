# Plan: Profile TTNNMoE Module Performance

## 1. Problem Description

**What we are profiling:** The `TTNNMoE` module (and its subclass `TTNNBailingMoE`) used in the Ling-mini-2.0 model test at:
```
/home/ttuser/salnahari/tt-metal/models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py
```

**Why:** The MoE module is executed once per decoder layer per token and is the most compute-intensive module in the model. Understanding where time is spent -- across CCL ops, routing, expert compute, and shared experts -- is essential for identifying and prioritizing optimization targets. The module contains multiple potential bottlenecks: all_gather, gate routing (3-pass BF16 centering with CPU fallback sort), expert dispatch/combine via all_to_all, three sparse_matmul operations, reduce_scatter, and shared expert MLP.

**Implementation files:**
- `models/experimental/tt_symbiote/modules/moe.py` -- `TTNNMoE` (line 1346), `TTNNBailingMoE` (line 1499), `TTNNExperts` (line 1027), `TTNNMoERouterDecode` (line 855)

## 2. Environment Setup

### 2.1 Base Environment (tt_bashrc)

The environment must be sourced from `tt_bashrc` located at `/home/ttuser/salnahari/tt_bashrc`:

```bash
cd /home/ttuser/salnahari
source tt_bashrc
cd tt-metal
```

This sets:
- `TT_METAL_HOME` to the tt-metal directory
- `PYTHONPATH` to include tt-metal
- `ARCH_NAME=wormhole_b0`
- `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml`
- `MESH_DEVICE=T3K`
- Activates the Python virtual environment at `tt-metal/python_env/`
- Defines `pytest_full` alias: `python3 -m tracy -v -r -p -m pytest`

### 2.2 Known Blocker: T3K Hang

**CRITICAL:** The test currently hangs on T3K due to `ttnn.all_reduce` with 8 devices in `TTNNLinearIColShardedWAllReduced.forward()` (used by `TTNNBailingMoEAttention`). See `PLAN_fix_ling_mini_hang.md` for details and the fix. The fix must be applied before profiling can proceed.

The hang occurs in the attention module, NOT the MoE module. The MoE module uses async CCL variants (`all_gather_async`, `reduce_scatter_minimal_async`) with proper semaphore management and works correctly on T3K.

## 3. TTNNMoE Module Analysis

### 3.1 Architecture Overview

`TTNNBailingMoE` inherits from `TTNNMoE`. The forward pass (moe.py line 1412-1496) executes this pipeline:

```
Input (x)
    |
    v
[1] all_gather_async (Linear topology, num_links=1)    -- Revert tensor parallelism
    |
    v
[2] typecast to float32 + gate linear (HiFi4)          -- Router logit computation
    |
    v
[3] TTNNMoERouterDecode.forward()                      -- 3-pass BF16 centering topk
    |   (30+ TTNN ops including 3x topk, 6x typecast,
    |    3x sub/slice, CPU sort fallback)
    v
[4] TTNNExperts.forward()                               -- Expert pipeline
    |   [4a] Pad tokens to SPARSITY_BLOCK_SIZE=32
    |   [4b] all_to_all_dispatch
    |   [4c] moe_expert_token_remap (sparsity tensor)
    |   [4d] sparse_matmul w1 (gate projection)
    |   [4e] sparse_matmul w3 (up projection)
    |   [4f] silu + elementwise mul
    |   [4g] sparse_matmul w2 (down projection)
    |   [4h] all_to_all_combine
    |   [4i] Weight application (permute/unsqueeze/repeat/mul/sum)
    |
    v
[5] reduce_scatter_minimal_async (Ring topology)        -- Reduce output across devices
    |
    v
[6] shared_experts (3x matmul + silu)                   -- Shared expert MLP
    |
    v
[7] ttnn.add (routed + shared outputs)                  -- Final sum
```

### 3.2 Ling-mini-2.0 Model Parameters

| Parameter | Value |
|-----------|-------|
| hidden_size | 2048 |
| moe_intermediate_size | 1408 |
| n_routed_experts | 64 |
| num_experts_per_tok | 8 |
| n_group | 1 |
| topk_group | 1 |
| n_shared_experts | 1 |
| num_hidden_layers | 24 (23 MoE layers + 1 dense layer 0) |

Because `n_group=1` and `topk_group=1`, the router takes the `n_group <= topk_group` branch, which uses the 3-pass BF16 centering trick (not the group-based selection path).

### 3.3 Key Submodule Details

**TTNNMoERouterDecode (moe.py:855-1024):** The router performs:
1. typecast to float32, sigmoid
2. Add correction bias (repeat + to_layout + add)
3. Pass 1: typecast to bf16, topk(k+1), slice threshold, typecast back to f32, subtract
4. Pass 2: same as pass 1 on centered scores
5. Pass 3: typecast to bf16, topk(k) -- final expert selection
6. gather original sigmoid scores for weights
7. normalize weights (sum + div)
8. apply routing scale (repeat + to_layout + typecast + mul)
9. CPU fallback sort: `_to_torch_any()` to convert to CPU, `torch.sort()`, convert back to ttnn (lines 806-813)

The CPU fallback sort at step 9 is a known bottleneck that forces a device-to-host synchronization.

**TTNNExperts (moe.py:1027-1343):** Uses `sparse_matmul` with `MatmulMultiCoreReuseMultiCast1DProgramConfig`. Expert computation uses HiFi2 math fidelity with `fp32_dest_acc_en=True` and `packer_l1_acc=True`. Weight dimensions per device on T3K (8 devices, 64 experts total): 8 experts per device, w1/w3 shape `(8, 2048, 1408)`, w2 shape `(8, 1408, 2048)`.

**TTNNGlm4MoeMLP (shared experts, moe.py:664-682):** Standard SwiGLU MLP: gate_proj (with SiLU fused), up_proj, elementwise mul, down_proj. Uses `TTNNLinearIColShardedWRowSharded` for each projection.

## 4. Profiling Strategy

### 4.1 Method 1: DispatchManager Host-Level Timing (Built-in, No Build Changes)

The test already uses `DispatchManager` for host-side timing:
- `DispatchManager.clear_timings()` (line 130) -- resets before the main run
- `DispatchManager.save_stats_to_file("ling_mini_2_0_paged_attention_timing_stats.csv")` (line 139)

This produces a CSV with per-module, per-call timing entries showing wall-clock time for each `TTNNModule.forward()` invocation.

**How to run:**
```bash
cd /home/ttuser/salnahari && source tt_bashrc && cd tt-metal
unset TT_VISIBLE_DEVICES
python -m pytest models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py -v --timeout=0 -s 2>&1 | tee profile_output.log
```

**What it shows:** Wall-clock time per module call (includes host overhead, dispatch latency, and device execution). Output file: `ling_mini_2_0_paged_attention_timing_stats.csv`.

**Limitation:** This shows total time per module call but does NOT break down into individual TTNN ops within each module. Host overhead (wrapper, Python dispatch) is included in the timing.

### 4.2 Method 2: Device-Level Tracy Profiling (Requires Tracy Build)

Tracy profiling gives per-op device kernel timing with microsecond resolution.

**Prerequisites:** TT-Metal must be built with Tracy enabled:
```bash
cd /home/ttuser/salnahari/tt-metal
ENABLE_TRACY=1 cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ttnn -j$(nproc)
```

**Environment variables for profiling:**
```bash
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1

# Reduce token count to avoid DRAM buffer overflow.
# The profiler's DRAM circular buffer overflows with too many ops.
# Reduce max_new_tokens from 128 to 8-16 for profiling runs.
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000
```

**Running the profiler using the `pytest_full` alias (defined in tt_bashrc):**
```bash
cd /home/ttuser/salnahari && source tt_bashrc && cd tt-metal
unset TT_VISIBLE_DEVICES

# Option A: Use the pytest_full alias (wraps pytest with tracy)
# pytest_full = python3 -m tracy -v -r -p -m pytest
pytest_full models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py --timeout=0 -s 2>&1 | tee tracy_output.log

# Option B: Explicit tracy invocation
python3 -m tracy -v -r -p -m pytest \
    models/experimental/tt_symbiote/tests/test_ling_mini_2_0.py \
    --timeout=0 -s 2>&1 | tee tracy_output.log
```

**IMPORTANT: Reduce token count for profiling.** The test generates 128 decode tokens. With ~500+ ops per MoE layer per token, across 23 MoE layers, the profiler buffer will overflow. Temporarily modify the test or create a profiling wrapper:

```python
# Reduce max_new_tokens for profiling
# In test_ling_mini_2_0.py line 131, change:
#   outputs = model.generate(**inputs, max_new_tokens=128, ...)
# To:
#   outputs = model.generate(**inputs, max_new_tokens=8, ...)
```

**Output:** Tracy generates `ops_perf_results_*.csv` files (one per device) with columns:
- `OP TYPE`: TTNN op name (e.g., `ttnn.matmul`, `ttnn.topk`)
- `DEVICE KERNEL DURATION [ns]`: Device-side kernel execution time
- `BRISC KERNEL DURATION [ns]`, `NCRISC KERNEL DURATION [ns]`, `TRISC0/1/2 KERNEL DURATION [ns]`: Per-RISC timings
- `PM IDEAL [ns]`: Performance-model ideal time
- `INPUT_0_W`, `INPUT_0_Z`, etc.: Tensor shape information

### 4.3 Method 3: Manual Python Instrumentation (No Build Changes, Moderate Detail)

Add `time.perf_counter()` + `ttnn.synchronize_device()` wrappers around key sections of `TTNNMoE.forward()` to get wall-clock timing per stage without needing a Tracy build.

**Instrumentation points in moe.py TTNNMoE.forward() (line 1412):**

```python
import time

@run_on_devices(DeviceArch.T3K)
def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
    timings = {}

    # [1] All-gather
    ttnn.synchronize_device(self.device)
    t0 = time.perf_counter()
    x = ttnn.experimental.all_gather_async(...)
    ttnn.synchronize_device(self.device)
    timings["all_gather"] = (time.perf_counter() - t0) * 1e3  # ms

    # [2] Gate routing
    t0 = time.perf_counter()
    # ... typecast + linear ...
    ttnn.synchronize_device(self.device)
    timings["gate_linear"] = (time.perf_counter() - t0) * 1e3

    # [3] Router
    t0 = time.perf_counter()
    topk_experts_indices, topk_experts_weights = self.route_tokens_to_experts(router_logits)
    ttnn.synchronize_device(self.device)
    timings["router"] = (time.perf_counter() - t0) * 1e3

    # [4] Experts
    t0 = time.perf_counter()
    routed_output = self.experts(x, topk_experts_indices, topk_experts_weights)
    ttnn.synchronize_device(self.device)
    timings["experts"] = (time.perf_counter() - t0) * 1e3

    # [5] Reduce-scatter
    t0 = time.perf_counter()
    routed_output = ttnn.experimental.reduce_scatter_minimal_async(...)
    ttnn.synchronize_device(self.device)
    timings["reduce_scatter"] = (time.perf_counter() - t0) * 1e3

    # [6] Shared experts + add
    t0 = time.perf_counter()
    shared_output = self.shared_experts(residual)
    output = ttnn.add(routed_output, shared_output.to_ttnn)
    ttnn.synchronize_device(self.device)
    timings["shared_experts_add"] = (time.perf_counter() - t0) * 1e3

    print(f"MoE timings (ms): {timings}")
    return output
```

**Pros:** Works with any build, gives per-stage breakdown. **Cons:** `synchronize_device()` after every stage adds synchronization overhead (creates pipeline bubbles that would not exist in normal execution), so absolute numbers will be inflated. Relative proportions are still useful.

### 4.4 Recommended Profiling Sequence

Execute in this order:

1. **First: DispatchManager (Method 1)** -- Run the test as-is to get baseline per-module wall-clock timing. This requires no code changes and gives the initial picture of which modules dominate.

2. **Second: Manual instrumentation (Method 3)** -- Add timing annotations to `TTNNMoE.forward()` and optionally `TTNNExperts.forward()` to break down the MoE forward pass into stages. This identifies which stage (routing, expert compute, CCL) dominates without needing a Tracy build.

3. **Third: Tracy profiling (Method 2)** -- Once bottlenecks are identified at the stage level, use Tracy for per-op device kernel timing to understand WHY a stage is slow (e.g., is a sparse_matmul slow due to poor config, or is an all_to_all slow due to topology choice).

### 4.5 What to Profile Inside TTNNExperts.forward()

For deeper analysis of the expert pipeline (Method 3), instrument these stages separately:

| Stage | Lines | What to measure |
|-------|-------|-----------------|
| Token padding | 1191-1212 | Pad overhead (should be near-zero for batch=1) |
| Layout conversion | 1214-1223 | typecast + to_layout + reshape overhead |
| all_to_all_dispatch | 1225-1230 | CCL dispatch latency |
| moe_expert_token_remap | 1238-1245 | Sparsity tensor generation |
| w1 sparse_matmul | 1250-1259 | Gate projection |
| w3 sparse_matmul | 1260-1269 | Up projection |
| silu + mul | 1271-1275 | Activation |
| w2 sparse_matmul | 1280-1289 | Down projection |
| all_to_all_combine | 1307-1312 | CCL combine latency |
| Weight application | 1321-1335 | permute/unsqueeze/repeat/mul/sum |

### 4.6 Mapping Tracy CSV Ops to Forward-Pass Stages

When using Tracy profiling (Method 2), the ops in the CSV map to forward-pass stages as follows:

| CSV op name (partial match) | Source | Stage |
|---|---|---|
| `all_gather` | moe.py L1429-1436 | Step 1: all_gather_async |
| `matmul` (1st) | moe.py L1445-1455 | Step 2: gate linear (HiFi4) |
| `topk` (3 occurrences) | moe.py L891-1024 (router) | Step 3: 3-pass centering topk |
| `typecast` (6+ occurrences) | moe.py L891-1024 (router) | Step 3: f32<->bf16 conversions |
| `all_to_all` (1st) | moe.py L1225-1230 | Step 4b: all_to_all_dispatch |
| `sparse_matmul` (1st) | moe.py L1250-1259 | Step 4d: w1 gate projection |
| `sparse_matmul` (2nd) | moe.py L1260-1269 | Step 4e: w3 up projection |
| `silu` + `eltwise_mul` | moe.py L1271-1275 | Step 4f: activation |
| `sparse_matmul` (3rd) | moe.py L1280-1289 | Step 4g: w2 down projection |
| `all_to_all` (2nd) | moe.py L1307-1312 | Step 4h: all_to_all_combine |
| `repeat` + `permute` + `mul` + `sum` | moe.py L1321-1335 | Step 4i: weight application |
| `reduce_scatter` | moe.py L1478-1490 | Step 5: reduce_scatter |
| `matmul` (last 3) | moe.py L1493 | Step 6: shared experts |
| `eltwise_add` | moe.py L1494 | Step 7: final add |

## 5. Expected Output

### 5.1 DispatchManager CSV

The CSV will contain rows like:
```
module_name, call_index, wall_time_ms, backend
TTNNBailingMoE, 0, 15.2, ttnn
TTNNBailingMoE, 1, 14.8, ttnn
TTNNBailingMoEAttention, 0, 9.5, ttnn
...
```

The pivot table (auto-generated by `save_stats_to_file`) will show mean/std per module across all decode tokens.

### 5.2 Manual Instrumentation Output

Per-decode-token breakdown:
```
MoE timings (ms): {
  'all_gather': 0.8,
  'gate_linear': 0.3,
  'router': 2.5,         # Likely dominant due to CPU fallback sort
  'experts': 5.0,         # Expert pipeline
  'reduce_scatter': 0.6,
  'shared_experts_add': 1.2
}
```

### 5.3 Tracy ops_perf_results CSV

Per-op device kernel timing with columns:
```
OP TYPE, DEVICE KERNEL DURATION [ns], INPUT_0_W, INPUT_0_Z, ...
ttnn.matmul, 25000, 2048, 1, ...
ttnn.topk, 8000, 64, 1, ...
ttnn.sparse_matmul, 35000, 1408, 1, ...
```

### 5.4 Expected Bottleneck Profile (Based on Research Cache Findings)

From the completed research in `guides/ttnn_moe_performance_optimization_on_t3k/`:

| Stage | Expected % of total | Bottleneck type |
|-------|-------------------|-----------------|
| Router (3-pass topk + CPU sort) | 15-25% | Host overhead (CPU fallback, many small ops) |
| CCL ops (all_gather + all_to_all x2 + reduce_scatter) | 25-35% | Communication bandwidth |
| Expert sparse_matmul (w1 + w3 + w2) | 20-30% | Device compute (potentially config-limited) |
| Weight application (permute/repeat/mul/sum) | 5-10% | Memory bandwidth (reshape-heavy) |
| Shared experts | 10-15% | Device compute |
| Layout conversions + typecasts | 5-10% | Host dispatch overhead |

## 6. Success Criteria

Profiling is successful when all of the following are met:

1. **End-to-end timing captured:** DispatchManager CSV shows per-module timing for all 128 decode tokens (or reduced count for Tracy runs), with identifiable TTNNBailingMoE entries.

2. **Stage-level breakdown obtained:** Either through manual instrumentation or Tracy, the six stages of TTNNMoE.forward() each have measured latency with < 20% run-to-run variance (after warmup).

3. **Bottleneck identified:** The profiling data clearly shows which stage(s) account for > 50% of the total MoE forward time, enabling prioritization of optimization work.

4. **Op-level detail (Tracy only):** For the identified bottleneck stage, individual op durations are captured with device kernel timing (not just wall-clock), allowing differentiation between compute-bound ops and bandwidth-bound ops.

5. **Reproducible:** The profiling procedure can be re-run after code changes to measure the effect of optimizations, producing comparable results.

6. **CPU fallback quantified:** The cost of the CPU sort fallback in `TTNNMoERouterDecode.forward()` (lines 806-813) is isolated and measured as a specific fraction of the router stage time.

## 7. Prerequisites and Risks

| Risk | Mitigation |
|------|-----------|
| T3K hang blocks all profiling | Apply fix from PLAN_fix_ling_mini_hang.md first (replace ttnn.all_reduce with composite RS+AG in TTNNLinearIColShardedWAllReduced) |
| Tracy build not available | Use Method 1 + Method 3 (no Tracy required) for initial profiling |
| Profiler DRAM buffer overflow | Reduce max_new_tokens to 8-16; set TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 |
| synchronize_device in Method 3 inflates timings | Use for relative comparison only; get absolute numbers from Tracy |
| Model download required | Ling-mini-2.0 model must be available via HuggingFace (HF_TOKEN is set in tt_bashrc) |
