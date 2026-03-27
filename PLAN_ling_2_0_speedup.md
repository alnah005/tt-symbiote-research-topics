# PLAN: Ling-mini-2.0 (test_ling_mini_2_0.py) Speedup

**Date:** 2026-03-27
**Author:** Architect Agent
**Model:** inclusionAI/Ling-mini-2.0 (BailingMoeV2)
**Architecture:** 20 layers, hidden_size=2048, 16 Q heads, 4 KV heads, head_dim=128, 256 experts (top-k routing), moe_intermediate=512, first_k_dense_replace=1

---

## 1. Problem Description (Current Performance Profile)

### Test Flow
1. Load model + tokenizer from HuggingFace
2. `register_module_replacement_dict` x2: first replaces `BailingMoeV2DecoderLayer` with `TTNNBailingMoEDecoderLayer`, then replaces all `nn.Linear` with `TTNNLinearIColShardedWRowSharded` and `nn.SiLU` with `TTNNSilu`
3. `set_device(model, mesh_device)` -- propagates T3K mesh to all modules
4. Sequential `preprocess_weights()` + `move_weights_to_device()` for ALL TTNN modules
5. Create paged KV cache
6. Warmup: `model.generate(max_new_tokens=2)`
7. Reset KV cache
8. Main run: `model.generate(max_new_tokens=128)`
9. Save timing stats

### Key Bottleneck Areas

**A. Run mode is NORMAL, not TRACED:**
The env does not set `TT_SYMBIOTE_RUN_MODE=TRACED`. The default is `NORMAL` (`NormalRun`). This means every TTNN module forward call goes through `NormalRun.module_run`, which calls `preprocess_weights()` and `move_weights_to_device()` on every invocation (the methods are no-ops after first call, but the timing instrumentation still runs). More critically, **no trace capture/replay is happening** -- even though `device_params` allocates 50MB trace region and `TracedRun.release_all()` is called at the end.

`TTNNLinear` is decorated with `@trace_enabled`, and `TTNNLinearIColShardedWRowSharded` inherits from it. However, `TTNNBailingMoEDecoderLayer`, `TTNNBailingMoEAttention`, `TTNNBailingMoE`, and `TTNNMoERouterDecode` are NOT trace-enabled. Even in TRACED mode, the decoder layer runs normally and only inner linear layers would be traced -- but tracing individual linears is suboptimal compared to tracing the entire decoder layer.

**B. Host-device round-trips in decode path:**
- `TTNNBailingMoEAttention._forward_decode_paged` creates `cache_position_tensor` on host every decode step (line 2668-2678), then sends it to device via `ttnn.from_torch()` with `ReplicateTensorToMesh`
- The `_to_replicated` method (line 2292-2317) explicitly round-trips through host: `ttnn.to_torch()` then `ttnn.from_torch()` -- though this method does not appear to be called in the current decode path
- `TTNNMoERouterDecode.forward` does routing computations that involve host round-trips for topk (the router converts to torch for topk computation)

**C. Per-token decode overhead for 19 MoE layers:**
Each decode token processes 19 MoE layers (layers 1-19) and 1 dense layer (layer 0). Each MoE layer involves:
- All-gather (revert tensor parallelism)
- Gate routing (matmul + topk routing with host-side computation)
- Expert dispatch/compute/combine (all_to_all + sparse_matmul x3 + all_to_all)
- Reduce-scatter
- Shared expert computation (separate MLP)
- 2x reduce_scatter in attention (QKV + dense projections)

**D. Verbose print statements on every module_run:**
`NormalRun.module_run` (line 557) prints on every invocation: `f"{self.__class__.__name__}: {self.module_name} on device {self.device}"`. For 128 decode tokens x 20 layers x multiple modules per layer, this is thousands of print statements that add measurable host-side latency.

**E. Weight preprocessing loop is sequential:**
Lines 111-113 call `preprocess_weights()` + `move_weights_to_device()` sequentially for every module. For 256 experts per MoE layer x 19 layers, this is a massive amount of weight transfers that could be parallelized or pipelined.

---

## 2. Root Cause Analysis

### Bottleneck #1: No Traced Execution (HIGH IMPACT)
**What:** Run mode defaults to NORMAL; trace capture never fires.
**Why it matters:** Traced execution eliminates host dispatch overhead by recording device commands once and replaying them. For decode (which runs 128 times), trace replay avoids re-issuing every TTNN op from the host. This is the single biggest optimization opportunity.
**Evidence:** `TracedRun.module_run` has full trace capture logic (lines 1008-1086) but is only active when `TT_SYMBIOTE_RUN_MODE=TRACED`. The test does NOT set this env var, and `tt_bashrc` does not set it either.

### Bottleneck #2: Host Round-Trips Per Decode Token (HIGH IMPACT)
**What:** `cache_position_tensor` is created as a host torch tensor and transferred to device every decode step. Router topk also involves host computation.
**Why it matters:** Each host-device transfer synchronizes the device, stalling the pipeline. At 128 tokens, this is 128 * 20 layers * N transfers per layer.
**Evidence:** Lines 2666-2688 in attention.py create `cur_pos_tt` from torch tensor every forward call.

### Bottleneck #3: MoE Router Host-Side Topk (MEDIUM IMPACT)
**What:** `TTNNMoERouterDecode` converts router logits to torch to do topk on CPU.
**Why it matters:** Forces device synchronization before topk and host-to-device transfer after.

### Bottleneck #4: Print Statements (MEDIUM IMPACT)
**What:** Every `module_run` call prints class name, module name, and device.
**Why it matters:** stdout is synchronized. With thousands of calls per generate(), this adds measurable latency.

### Bottleneck #5: Sequential Weight Transfer (LOW IMPACT - ONE-TIME)
**What:** Weight preprocessing and device transfer is sequential.
**Why it matters:** One-time cost but can be significant for 256 experts x 19 layers. This does not affect per-token throughput.

### Bottleneck #6: Redundant Layout/Type Checks (LOW IMPACT)
**What:** Multiple redundant `if hs.layout != ttnn.TILE_LAYOUT` and `if hs.dtype != ttnn.bfloat16` checks throughout decoder_layer.py and attention.py.
**Why it matters:** These are cheap but add to dispatch overhead when multiplied across layers and tokens.

---

## 3. Step-by-Step Implementation Plan (Ordered by Impact)

### Step 1: Enable Traced Execution for Decode Phase (HIGHEST IMPACT)

**Estimated speedup:** 2-5x for decode throughput

**What to do:**
1. Set `TT_SYMBIOTE_RUN_MODE=TRACED` in the test (or export it before running)
2. Mark `TTNNBailingMoEDecoderLayer` with `@trace_enabled` decorator
3. The warmup run (max_new_tokens=2) will capture the trace; the main run will replay it

**Implementation:**

File: `test_ling_mini_2_0.py`
```python
import os
os.environ["TT_SYMBIOTE_RUN_MODE"] = "TRACED"
```

File: `modules/decoder_layer.py`
```python
from models.experimental.tt_symbiote.core.run_config import trace_enabled

@trace_enabled
class TTNNBailingMoEDecoderLayer(TTNNModule):
    ...
```

**Complications:**
- Trace capture requires ALL operations in the forward path to be device-only (no host round-trips). This conflicts with Bottleneck #2 (host cache_position creation) and Bottleneck #3 (host topk). These must be fixed first or trace will not capture correctly.
- MoE routing dynamically selects experts based on input content -- this makes the computation graph input-dependent, which trace replay cannot handle. The experts module uses `all_to_all_dispatch` with dynamic sparsity tensors.
- **Recommendation:** Trace the decoder layer forward EXCLUDING the MoE routing. Instead, trace attention + residual adds + norms separately, and keep MoE as non-traced.

**Alternative approach (more practical):**
Instead of tracing the entire decoder layer, trace individual components that have fixed computation graphs:
1. Mark `TTNNBailingMoEAttention` as `@trace_enabled` -- attention has a fixed graph for decode (same shapes every token)
2. Keep MoE as non-traced since routing is dynamic
3. This captures the attention path (QKV projection, RoPE, paged SDPA, dense projection) which runs 20 times per token

**BUT** there is a critical issue: `TTNNBailingMoEAttention.forward` creates host tensors (cache_position) mid-forward, which breaks trace capture. Fix Step 2 first.

### Step 2: Eliminate Host Round-Trips in Decode Attention (HIGH IMPACT)

**Estimated speedup:** 1.3-2x for decode throughput

**What to do:**
Pre-compute all `cur_pos_tt` tensors before generation and pass them as device tensors, OR compute position incrementally on device.

**Implementation option A: Pre-allocated position tensor**

File: `modules/attention.py`, method `_forward_decode_paged`

Replace lines 2666-2688 with:
```python
# Use a pre-allocated position tensor that gets updated on device
layer_idx = self._fallback_torch_layer.layer_idx
cur_pos = past_key_values.get_seq_length(layer_idx)
# Reuse or create the position tensor on device
if not hasattr(self, '_cur_pos_tt') or self._last_cur_pos != cur_pos:
    cache_position_tensor = torch.tensor([cur_pos], dtype=torch.int32)
    mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)
    self._cur_pos_tt = ttnn.from_torch(
        cache_position_tensor,
        device=self.device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    self._last_cur_pos = cur_pos
cur_pos_tt = self._cur_pos_tt
```

**Implementation option B: Move position tracking to device (preferred for tracing)**

Create a device-side position counter that is incremented each decode step. This eliminates ALL host synchronization from the attention decode path.

File: `modules/attention.py`
```python
# In move_weights_to_device_impl, pre-allocate:
self._decode_pos_buffer = ttnn.from_torch(
    torch.zeros(1, dtype=torch.int32),
    device=self.device,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

**Note:** Option A is simpler but still requires one host-device transfer per unique position. Option B is better for tracing but more complex. Start with Option A.

### Step 3: Move MoE Router Topk to Device (MEDIUM-HIGH IMPACT)

**Estimated speedup:** 1.2-1.5x for decode throughput

**What to do:**
The `TTNNMoERouterDecode.forward` currently converts router logits to torch for topk. Replace with `ttnn.topk` which runs on device.

**Files to modify:** `modules/moe.py`

Check whether `ttnn.topk` supports the required operation. If not, the 3-pass BF16 centering trick must be adapted.

**Note from research cache:** The router uses a 3-pass BF16 centering trick for precision. Simplifying to a single-pass device-side topk may lose some precision. Test output quality after this change.

### Step 4: Suppress Verbose Prints in NormalRun (MEDIUM IMPACT)

**Estimated speedup:** 1.1-1.2x

**What to do:**
Gate the print statements behind a debug flag or environment variable.

**Implementation:**

File: `core/run_config.py`

In `NormalRun.module_run` (line 557), add a gate:
```python
# Per the project instruction: "gate prints don't remove"
if os.environ.get("TT_SYMBIOTE_VERBOSE", "0") == "1":
    print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
```

Similarly gate the print in `TracedRun.module_run` lines 1038-1044, 1060, 1067-1068.

**Important:** Per project guidelines, "gate prints don't remove" -- we add a conditional gate, we do not delete the print statements.

### Step 5: Use bfloat8_b for Expert Weights (MEDIUM IMPACT)

**Estimated speedup:** 1.2-1.5x for MoE computation

**What to do:**
Expert matmuls use bfloat16 weights. Switching to bfloat8_b halves memory bandwidth for expert weight reads, which is the bottleneck for small-batch decode (compute is bandwidth-bound at batch=1).

**Implementation:**

File: `modules/moe.py`, class `TTNNExperts`

In `preprocess_weights_impl` or `move_weights_to_device_impl`, change weight dtype from `ttnn.bfloat16` to `ttnn.bfloat8_b`.

**Risk:** Accuracy degradation. Must verify output text quality. The research cache notes HiFi2 is already used for expert matmuls -- combining with bfloat8_b may compound errors. Test with HiFi4 + bfloat8_b as a safer option.

### Step 6: Fuse Shared Expert + Routed Expert Add (LOW IMPACT)

**Estimated speedup:** 1.05-1.1x

**What to do:**
In `TTNNMoE.forward` (moe.py line 1493-1494), the shared expert and routed expert outputs are computed then added:
```python
shared_output = self.shared_experts(residual)
output = ttnn.add(routed_output, shared_output.to_ttnn)
```

The shared expert computation is independent of routed experts. Currently they run sequentially. With async ops, shared_experts could overlap with expert routing/computation.

**Implementation:**
Start shared expert computation before the routing/dispatch, and synchronize only when adding:
```python
# Start shared experts early (before routing)
shared_output = self.shared_experts(residual)  # Move this BEFORE gate routing
# ... routing, experts ...
output = ttnn.add(routed_output, shared_output.to_ttnn)
```

This may already be happening via async dispatch. Profile to confirm.

### Step 7: Parallel Weight Transfer During Initialization (LOW IMPACT, ONE-TIME)

**Estimated speedup:** Reduces initialization time, not per-token throughput

**What to do:**
Batch weight transfers instead of sequential per-module loop.

**Implementation:**
```python
# Preprocess all weights first (CPU-only, can be parallel)
for k, v in tqdm(all_modules.items()):
    v.preprocess_weights()

# Then batch-transfer to device
for k, v in tqdm(all_modules.items()):
    v.move_weights_to_device()
```

This is already the current pattern. The improvement would be to use threading for CPU-side preprocessing.

### Step 8: Optimize RoPE Setup for Decode (LOW IMPACT)

**Estimated speedup:** 1.02-1.05x

**What to do:**
`BailingRotarySetup.get_cos_sin_for_decode` and `get_trans_mat_decode_sharded` are called every decode step. Pre-compute and cache cos/sin for all positions up to max_seq_length.

**Implementation:**
In `move_weights_to_device_impl`, pre-compute cos/sin for positions 0..max_seq_len and store on device. Index into the pre-computed buffer during decode.

---

## 4. Success Criteria

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Decode throughput (tokens/s) | TBD (measure with current test) | 2x+ baseline | DispatchManager timing stats |
| End-to-end time (128 tokens) | TBD | 50%+ reduction | Wall clock |
| Output correctness | Current output text | Identical or semantically equivalent | Compare decoded text |
| Warmup overhead | TBD | May increase (trace capture) | Wall clock for warmup |

**Correctness verification:**
1. Run test with current code, save output text as golden reference
2. Apply optimizations one at a time
3. After each optimization, compare output text -- must be identical (or within BF16 rounding tolerance for bfloat8_b change)
4. Final assertion: `len(decoded.strip()) > 0` must still pass

---

## 5. Risk Assessment

| Optimization | Risk Level | What Could Break | Mitigation |
|-------------|------------|------------------|------------|
| Step 1: Traced execution | HIGH | MoE dynamic routing breaks trace replay; attention host ops break capture | Trace only attention, not MoE; fix host ops first |
| Step 2: Eliminate host round-trips | MEDIUM | Position counter drift if cache reset logic changes | Unit test position tracking |
| Step 3: Device-side topk | MEDIUM | Precision loss in routing affects expert selection | Compare routing decisions before/after |
| Step 4: Gate prints | LOW | No functional risk | Gate, don't remove |
| Step 5: bfloat8_b experts | MEDIUM | Accuracy degradation in generated text | Test with golden reference; revert if quality drops |
| Step 6: Fuse shared+routed | LOW | Race condition if async dispatch reorders | Verify with device sync |
| Step 7: Parallel weight init | LOW | No functional risk | N/A |
| Step 8: RoPE cache | LOW | Memory increase for pre-computed tables | Minimal for small model |

### Critical Dependencies
- **Step 1 depends on Step 2:** Traced execution requires elimination of host round-trips in the traced forward path
- **Step 1 depends on understanding MoE traceability:** If MoE routing is fully device-side (after Step 3), the entire decoder layer could potentially be traced
- **Step 5 should be tested independently:** bfloat8_b change is orthogonal to tracing optimizations

### Recommended Implementation Order
1. **Step 4** (gate prints) -- zero risk, immediate benefit, quick to implement
2. **Step 2** (eliminate host round-trips) -- prerequisite for tracing, standalone benefit
3. **Step 3** (device-side topk) -- prerequisite for full tracing, standalone benefit
4. **Step 1** (traced execution) -- highest impact, requires Steps 2-3
5. **Step 5** (bfloat8_b weights) -- independent, test separately
6. **Steps 6-8** -- incremental improvements

---

## 6. T3K Hang Dependency

**IMPORTANT:** Per the research cache entry "Ling-mini-2.0 T3K Hang Root Cause", the test currently hangs on T3K due to a semaphore cycling race condition. The speedup plan assumes the hang is resolved first. The hang fix (documented in `FINDINGS_ling_mini_t3k_hang_deep_investigation.md`) should be applied before any speedup work begins.

If the hang is not yet resolved, speedup work can still proceed on N150 (single device) or N300 (2 devices) by temporarily adjusting the test's mesh device configuration. Note that CCL-related optimizations (Steps 2, 3) would not be testable on single-device configurations.
