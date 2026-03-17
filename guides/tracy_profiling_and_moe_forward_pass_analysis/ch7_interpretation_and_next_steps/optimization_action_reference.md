# Optimization Action Reference

This file is a concise reference for the TTNN optimization levers available when addressing
MoE forward pass performance gaps. For each lever the table and subsections below cover:
expected latency reduction, conditions under which it applies, and correctness caveats.

The final section describes the PCC validation procedure required after applying any
optimization to confirm that output correctness has not been degraded.

---

## Quick-Reference Table

| Lever | Pattern addressed | Expected reduction | Condition | Primary caveat |
|---|---|---|---|---|
| Remove `ttnn.synchronize_device` | B | ~14ms/step (constant) | Redundant barrier between expert matmul and combine | Must verify device command queue ordering is sufficient |
| `ttnn.record_event` / `ttnn.wait_for_event` | B | ~14ms/step (constant) | Ordering required but host blocking not needed | Correct event placement is critical; wrong placement causes silent corruption |
| `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace` | A, B | Eliminates per-op host dispatch overhead | Decode mode only (fixed shapes) | Tensors cannot be reallocated between capture and execute |
| `ttnn.enable_program_cache()` | D | 200–500ms first-call stall | Program cache not already enabled | Must be called before any TTNN op on the device |
| Sharded memory config for expert weights | C (secondary) | Reduces DRAM bandwidth pressure per matmul | Decode regime (batch_size × top_k ≤ 16 active tokens per chip); weights remain in DRAM shards, not L1 | Shard spec must match matmul grid; wrong spec silently falls back to DRAM |
| `ttnn.experimental.ccl.all_to_all_async` | C | Up to full CCL latency (~2ms at seq_len=1024) | Two consecutive MoE layers with non-aliased buffers | Buffer aliasing causes silent data corruption |
| Expert capacity padding to multiple of 32 | D (tile boundary) | Eliminates per-seq_len recompilation | Variable seq_len workloads | Padding wastes some compute on partial expert batches |

---

## Lever 1: Remove `ttnn.synchronize_device`

**What it does:** `ttnn.synchronize_device(device)` blocks the host thread until all
previously enqueued kernels have completed on the device. Removing it allows the host to
continue dispatching ops immediately after the expert matmul phase without waiting for the
device to drain.

**When it applies:** When a `ttnn.synchronize_device` call exists between the expert matmul
phase and the combine phase, and that barrier is redundant because both phases are enqueued
on the same command queue (cq_id=0). The device command queue serializes ops automatically —
the combine phase scatter will not begin executing until the matmuls it depends on are done.

**Expected reduction:** The full constant term from the Chapter 6 linear regression, which
in the worked example is approximately **13.8ms per forward pass step**.

**Correctness caveat:** Before removing the call, trace the data flow from the last expert
matmul op to the first combine op. Confirm that no output tensor from the matmul phase is
read on the host (e.g., via `ttnn.to_torch`) before the combine phase runs. If such a host
read exists, the synchronization may be necessary to ensure the host sees a consistent value.

```python
# Locate all synchronization calls in the MoE forward pass
import pathlib
for sf in pathlib.Path("models/").rglob("moe*.py"):
    for lineno, line in enumerate(sf.read_text().splitlines(), start=1):
        if "synchronize_device" in line:
            print(f"{sf}:{lineno}: {line.strip()}")
```

---

## Lever 2: `ttnn.record_event` / `ttnn.wait_for_event`

**What it does:** Records a lightweight event on the device command queue after the expert
matmuls complete, then inserts a device-side wait before the combine phase. The host does
not block — it dispatches the combine ops immediately, and the device enforces ordering via
the event.

**When it applies:** When ordering between the expert matmul phase and the combine phase
is required for correctness, but the combine ops can be enqueued speculatively on the host
without waiting for the matmuls to finish on the device.

**Expected reduction:** Same as removing the barrier outright — approximately **13.8ms per
step** — because the host is no longer stalled.

**Correctness caveat:** The `ttnn.wait_for_event` call must be placed on the device command
queue *before* the first combine op that reads matmul outputs. Placing it after that op will
not prevent the race condition. Test with a PCC check immediately after making this change.

```python
# Pattern: record after matmul, wait before combine, no host blocking
matmul_done = ttnn.record_event(device, cq_id=0)

# Device will wait for matmul_done before executing scatter
ttnn.wait_for_event(cq_id=0, event=matmul_done)

# Host dispatches combine ops to the command queue immediately after enqueuing the wait
ttnn.scatter(expert_outputs, gather_indices, output_tensor)
```

> **Warning:** If expert matmuls and combine ops are dispatched on different command queues
> (cq_id=0 and cq_id=1), you must record the event on the matmul queue and wait on the
> combine queue explicitly. The default single-queue path does not require this.

---

## Lever 3: Trace Capture (`ttnn.begin_trace_capture` / `ttnn.end_trace_capture` / `ttnn.execute_trace`)

**What it does:** Records the sequence of TTNN ops dispatched between `begin_trace_capture`
and `end_trace_capture` into a replayable trace. On subsequent calls, `execute_trace` replays
the recorded op sequence without re-dispatching each op from the host, eliminating per-op
host overhead.

**When it applies:** Decode mode only, where the op sequence and all tensor shapes are
identical across every inference step. In decode, `seq_len=1` per step, so the MoE forward
pass executes the same ops with the same shapes every time.

**Expected reduction:** Eliminates per-op host dispatch latency for the captured op sequence.
For a full MoE forward pass with ~13 ops, this can reduce host-side overhead by 0.5–2ms per
step depending on Python interpreter load. The device-side compute time is unchanged.

**Conditions:**

- All tensors used inside the trace must be pre-allocated before `begin_trace_capture` and
  must not be reallocated, reshaped, or re-created between trace capture and trace execute.
- The trace region must not contain Python control flow that depends on tensor values
  (e.g., `if ttnn.to_torch(topk_output)[0] > threshold`).

```python
# Allocate output tensors before capture
output_tensor = ttnn.zeros_like(hidden_states)

trace_id = ttnn.begin_trace_capture(device, cq_id=0)
# Dispatch the full MoE forward op sequence here
router_logits = ttnn.linear(hidden_states, router_weight)
topk_vals, topk_idx = ttnn.topk(router_logits, k=top_k)
flat_idx = ttnn.reshape(topk_idx, [seq_len * top_k])
sorted_order = ttnn.argsort(flat_idx)
gathered = ttnn.gather(hidden_states, sorted_order)
# ... expert matmul, combine ...
ttnn.end_trace_capture(device, trace_id, cq_id=0)

# Each decode step:
ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
```

> **Tip:** Capture the trace at the outermost scope possible (full MoE forward pass) to
> maximize the number of ops whose dispatch overhead is eliminated. Fine-grained traces
> covering only the dispatch sub-graph save less than capturing the entire forward pass.

---

## Lever 4: `ttnn.enable_program_cache()`

**What it does:** Enables the tt-metal program cache, which stores compiled TTNN kernels
keyed by op parameters (shapes, dtypes, memory configs, grid configs). With the cache
enabled, the second and subsequent calls to an op with the same parameters reuse the compiled
kernel without recompilation.

**When it applies:** Always. This should be called before the first TTNN op in any inference
process. It is the baseline configuration; the question is only whether it has been
accidentally disabled in a test harness.

**Expected reduction:** 200–500ms first-call reduction (eliminates cold-cache compilation).
Zero effect on subsequent calls with the same shapes when the cache is already warm.

```python
import ttnn

# Call once at process startup, before any inference
ttnn.enable_program_cache()

# Pre-warm the cache for all supported seq_len values
for seq_len in [64, 128, 256, 512, 1024, 2048, 4096]:
    dummy = ttnn.zeros([seq_len, d_model], dtype=ttnn.bfloat16,
                       layout=ttnn.TILE_LAYOUT,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG,
                       device=device)
    for _ in range(3):
        _ = moe_forward(dummy, router_weights)
```

---

## Lever 5: Sharded Memory Config for Expert Weight Tensors

**What it does:** Stores expert weight tensors in L1-sharded memory rather than DRAM. When
the expert matmul reads weights from L1, it avoids repeated DRAM bandwidth consumption for
the same weight data across multiple calls.

**When it applies:** When the same expert weights are read in every forward pass (which is
always true in inference) and the weight tensors fit within the L1 budget per core.

On Wormhole B0, each Tensix core has 1.5 MB of L1. The 80-core grid (8×10) provides
120 MB of total L1. A single expert's weight matrices (gate + up + down projection for
d_ff=2048, d_model=7168 in bfloat16) occupy approximately 3 × 7168 × 2048 × 2 ≈ 84 MB
per expert. Across 128 experts on T3K (8 chips), that is 128 × 84 MB / 8 chips = 1,344 MB
per chip in DRAM. Because a single expert's weights (~84 MB) far exceed the 120 MB total L1
available per chip, sharding individual expert weights into L1 is not feasible; this lever
is better applied to activation tensors or smaller projection sub-matrices.

**Expected reduction:** Reducing DRAM reads for expert weights can improve matmul throughput
by 10–30% when the op is DRAM-bandwidth bound. The effect on the gap specifically is
indirect: faster matmuls mean less time spent in `MoE/expert_matmul`, which increases the
opportunity for async CCL overlap.

**Correctness caveat:** The shard spec used when creating the weight tensor must exactly
match the shard spec expected by the `ttnn.matmul` op. Mismatched specs trigger a fallback
to DRAM layout at runtime, eliminating the benefit silently without an error.

```python
# Define a height-sharded memory config for expert weights
shard_spec = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))}),
    [d_model // 80, d_ff],          # shard shape: divide d_model across 80 cores
    ttnn.ShardOrientation.ROW_MAJOR,
    False,
)
expert_weight_mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    shard_spec,
)
expert_weight_sharded = ttnn.to_memory_config(expert_weight, expert_weight_mem_config)
```

---

## Lever 6: `ttnn.experimental.ccl.all_to_all_async`

**What it does:** Launches the CCL all-to-all collective asynchronously, allowing the host
to dispatch subsequent ops without waiting for the collective to complete. Combined with a
properly ordered command queue, this hides CCL latency behind other compute.

**When it applies:** When two consecutive MoE layers are processed in sequence and the
combine phase of layer N-1 is long enough to overlap with the all-to-all for layer N.

At d_model=7168, top_k=8, T3K effective bandwidth ~7 GB/s, the CCL latency at seq_len=1024
is approximately 2ms. The combine phase at the same seq_len takes approximately 0.8ms. The
overlap opportunity is therefore limited at seq_len=1024: the CCL is longer than the
available overlap window.

> **Tip:** Measure the combine phase duration at your target seq_len before investing in
> async CCL overlap. If `MoE/combine` is shorter than the CCL duration, overlap will only
> partially hide the CCL cost. At long prefill lengths (seq_len=4096), both the combine
> phase and the CCL scale linearly, maintaining a fixed overlap fraction.

**Correctness caveat:** The all_to_all_async output buffer must not be read until the async
handle signals completion. Accessing the output before completion produces silent data
corruption. Use the returned handle's completion check before dispatching the expert matmul.

---

## Lever 7: Expert Capacity Padding to Multiple of 32

**What it does:** Pads the number of tokens assigned to each expert (expert capacity) to the
nearest multiple of 32, ensuring that the expert matmul always operates on a whole number
of 32×32 tiles. This prevents tile-count discontinuities that would invalidate the program
cache when seq_len changes.

**When it applies:** Variable seq_len workloads (e.g., prefill at arbitrary lengths) where
`expert_capacity = seq_len × top_k / num_experts` does not always land on a tile boundary.

**Expected reduction:** Eliminates per-seq_len recompilation cost (200–500ms, Pattern D,
at each unique expert capacity value). Has no effect on hot-path latency once the cache is
warm.

**Correctness caveat:** Padding introduces dummy tokens in the expert batches. These tokens
must be masked out when computing the final weighted combine step. Verify that the masking
logic is applied correctly; otherwise padded tokens contribute to the output and reduce PCC.

```python
# Pad expert_capacity to a multiple of 32
raw_capacity = (seq_len * top_k + num_experts - 1) // num_experts  # ceiling
expert_capacity = ((raw_capacity + 31) // 32) * 32                 # tile-align
```

---

## PCC Validation Procedure

After applying any optimization, confirm that the output quality has not been degraded by
computing the Pearson Cross-Correlation (PCC) between the optimized output and a CPU
reference.

**Threshold:** PCC > 0.999 for bfloat16 outputs. This threshold accounts for bfloat16
rounding differences relative to float32 and is the standard correctness bar for TTNN MoE
output validation.

```python
import torch
import ttnn

def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson Cross-Correlation between two tensors (flattened)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    mean_a = a_flat.mean()
    mean_b = b_flat.mean()
    cov = ((a_flat - mean_a) * (b_flat - mean_b)).mean()
    std_a = a_flat.std(unbiased=False)
    std_b = b_flat.std(unbiased=False)
    if std_a == 0 or std_b == 0:
        return 1.0 if torch.allclose(a_flat, b_flat) else 0.0
    return (cov / (std_a * std_b)).item()


# Reference: CPU forward pass in float32
cpu_output = moe_forward_cpu(hidden_states_cpu, router_weights_cpu)

# Optimized: TTNN forward pass on device
device_output_tt = moe_forward_optimized(hidden_states_device, router_weights_device)
device_output_cpu = ttnn.to_torch(device_output_tt).float()

pcc = compute_pcc(cpu_output, device_output_cpu)
assert pcc > 0.999, f"PCC {pcc:.6f} below threshold 0.999 — optimization may have introduced a bug"
print(f"PCC: {pcc:.6f}  PASS")
```

> **Warning:** Run the PCC check at multiple seq_len values, not just the reference seq_len
> used during profiling. Some bugs introduced by shape canonicalization or capacity padding
> are seq_len-dependent and only manifest at certain sweep points.

> **Tip:** Keep the CPU reference implementation in a separate function that is never modified
> by optimization work. Use it as the ground truth for all PCC comparisons throughout the
> optimization cycle.

---

## Next Steps

This is the final file of the guide. You have now covered the full investigation loop:
collecting Tracy and device profiler data (Chapters 1–3), mapping the MoE op sequence and
expected performance (Chapter 4), attributing gaps to specific patterns (Chapter 5),
quantifying scaling behavior (Chapter 6), and translating findings into concrete optimization
actions with validation criteria (Chapter 7).

For related guides that build on this material, see:

- **Qwen 235B MoE Optimization Plan** — applies the gap analysis methodology to the specific
  16ms gap observed in the Qwen 235B-A22B model on T3K, with detailed implementation notes
  for the synchronization barrier and CCL overlap fixes.
- **T3K Expert Parallelism Configuration Guide** — covers `ep_degree` selection, CCL topology,
  and weight sharding strategies for large MoE models on the 8-chip Wormhole mesh.
- **TTNN Program Cache and Trace Capture Reference** — comprehensive API reference for
  `ttnn.enable_program_cache`, `ttnn.begin_trace_capture`, and related functions, including
  multi-device and multi-queue configurations not covered in this guide.
