# Common Gap Patterns

This file catalogs four gap patterns that appear repeatedly in MoE forward pass profiles on
Wormhole B0 and T3K hardware. Each pattern has a characteristic Tracy zone signature, a root
cause, and a set of diagnostic steps to confirm and eliminate it.

The pattern names (A–D) are referenced in Chapter 7 (`gap_to_action_mapping.md`), where each
pattern is mapped to a specific optimization action.

---

## Pattern A: Gap After `ttnn.topk` — Unannotated Index Construction

### Description

A gap of 0.5–5ms appears between the `MoE/dispatch/topk` zone and the `MoE/dispatch/gather`
zone. The gap is present every iteration and does not scale strongly with seq_len.

### Tracy Zone Signature

```
Timeline (dispatch thread):
  [MoE/dispatch/router ][MoE/dispatch/topk ]       [MoE/dispatch/gather ]
                                             ^^^^^^^
                                           gap here
                                        (no zone present)
```

### Root Cause

After `ttnn.topk` returns the top-K expert indices as a tensor on device, the model
implementation constructs an explicit token-to-expert mapping on the CPU before calling
`ttnn.gather`. This involves Python loops or NumPy operations that move data to the host,
build an assignment array, and transfer it back to the device. Because no Tracy zone wraps
this code, it appears as whitespace in the timeline. For the full description of this pattern
and the typical offending code, see
[Chapter 4, `dispatch_phase.md` — Index Tensor Construction](../ch4_moe_op_breakdown/dispatch_phase.md).

### Confirmation Steps

1. Add a Tracy zone around the index construction block:

```python
with tracy.zone("MoE/dispatch/index_construction"):
    expert_indices_cpu = ttnn.to_torch(expert_indices)
    token_assignments = []
    for token_idx in range(seq_len):
        for k in range(top_k):
            expert_id = expert_indices_cpu[token_idx, k].item()
            token_assignments.append((token_idx, expert_id))
    token_assignments.sort(key=lambda x: x[1])
    gather_indices = torch.tensor([ta[0] for ta in token_assignments])
    gather_indices_device = ttnn.from_torch(gather_indices, device=device)
```

2. Re-run the trace. If the gap is now entirely inside `MoE/dispatch/index_construction`,
   Pattern A is confirmed.

### Remedy

Replace the Python loop with a tensor-native implementation. The sort-and-index construction
can be done entirely on device:

```python
# Tensor-native index construction: no host round-trip
# expert_indices: [seq_len, top_k] on device, dtype=int32

# Flatten and argsort to get token order by expert assignment
flat_indices = ttnn.reshape(expert_indices, [seq_len * top_k])
sorted_order = ttnn.argsort(flat_indices)  # stable sort, ascending expert id
# sorted_order now gives the gather permutation without leaving the device
```

This eliminates the `ttnn.to_torch` / Python loop / `ttnn.from_torch` round-trip.

---

## Pattern B: Gap Between Last Expert Matmul and First Combine Op — Device Sync Barrier

### Description

A gap of 2–20ms appears between the end of the `MoE/expert_matmul` zone and the start of
the `MoE/combine` zone. The gap duration is highly consistent across iterations (CV < 5%)
and does not scale with seq_len.

### Tracy Zone Signature

```
Timeline (dispatch thread):
  [MoE/expert_matmul                  ][ttnn.synchronize_device][MoE/combine       ]
                                        ^^^^^^^^^^^^^^^^^^^^^^^^
                                           16ms barrier zone
```

If the synchronization call has no Tracy zone annotation, the signature instead shows:

```
Timeline (dispatch thread):
  [MoE/expert_matmul                  ]                        [MoE/combine       ]
                                        ^^^^^^^^^^^^^^^^^^^^^^^^
                                        gap with no zone present
```

### Root Cause

A `ttnn.synchronize_device(device)` call between the expert matmul phase and the combine
phase blocks the host thread until all previously enqueued kernels complete on the device.
This is sometimes inserted as a correctness guard: the combine phase reads expert matmul
outputs, and the developer wanted to ensure the matmuls are finished before dispatching the
scatter op.

On Wormhole B0, `ttnn.synchronize_device` typically drains the device queue in 1–5ms for
a small op graph and up to 20ms for a large prefill op graph. At seq_len=1024 with 128
experts and top_k=8, the expert matmul phase is large, making the synchronization expensive.

### Confirmation Steps

1. Search the MoE layer source code for `synchronize_device`:

```python
import pathlib

source_files = list(pathlib.Path("models/").rglob("moe*.py"))
for sf in source_files:
    text = sf.read_text()
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "synchronize_device" in line or "wait_for_event" in line:
            print(f"{sf}:{lineno}: {line.strip()}")
```

2. Wrap the call in a Tracy zone and re-profile:

```python
with tracy.zone("ttnn.synchronize_device"):
    ttnn.synchronize_device(device)
```

3. If the Tracy zone duration matches the gap, Pattern B is confirmed.

### Remedy

Evaluate whether the synchronization is required for correctness. In most MoE implementations,
the combine phase ops are enqueued after the matmul ops, and the device command queue enforces
op ordering — the scatter cannot begin executing until the matmuls complete. The explicit
`synchronize_device` call is therefore redundant.

If the synchronization was added as a defensive measure, replace it with a fine-grained
event:

```python
# Instead of synchronize_device, use an event to signal matmul completion
# without blocking the host dispatch of the combine ops
matmul_done_event = ttnn.record_event(device, cq_id=0)
# Host continues dispatching combine ops immediately
ttnn.scatter(expert_outputs, gather_indices, output_tensor)
# Device waits for the event before executing scatter (enforced in command queue)
ttnn.wait_for_event(cq_id=0, event=matmul_done_event)
```

This pattern is fully pipelined: the host dispatches all combine ops without stalling, and
the device enforces the ordering via the event mechanism.

---

## Pattern C: Gap Between Dispatch and Expert Compute Scaling With `num_active_tokens` — CCL All-to-All Latency

### Description

A gap appears between `MoE/dispatch/gather` and `MoE/expert_matmul`. The gap scales roughly
linearly with seq_len: at seq_len=256 the gap is ~4ms; at seq_len=1024 the gap is ~16ms.
The relationship is approximately `gap_ms ≈ k × seq_len` for some constant `k`.

### Tracy Zone Signature

```
Timeline (dispatch thread, seq_len=1024):
  [MoE/dispatch/gather ]                             [MoE/expert_matmul       ]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              ~16ms gap
                         (CCL all-to-all, unannotated)

Timeline (dispatch thread, seq_len=256):
  [MoE/dispatch/gather ]            [MoE/expert_matmul       ]
                         ^^^^^^^^^^^^
                           ~4ms gap
```

### Root Cause

On T3K, after `ttnn.gather` reorders tokens by expert, a CCL all-to-all op redistributes
the dispatched tokens to the chip that holds each expert shard. The latency of this collective
scales with the number of active tokens being redistributed:

```
message_bytes = num_active_tokens × d_model × bytes_per_element
num_active_tokens = seq_len × top_k  (before expert capacity clamping)
```

For DeepSeek-V3 at seq_len=1024: `1024 × 8 × 7168 × 2 = 117,440,512 bytes ≈ 112 MB`.

At an effective T3K ethernet throughput of ~7 GB/s per direction (typical for large all-to-all
with all 8 chips participating), the expected latency at seq_len=1024 is ~2.1 ms. For the
full CCL latency formula and parameter values, see
[Method 3 in `gap_attribution.md`](./gap_attribution.md#method-3-check-if-the-gap-aligns-with-a-ccl-collective).

If the observed gap is significantly larger than this estimate (e.g., 16ms vs. 2ms), the
CCL op is either running at well below peak bandwidth or there is additional blocking behavior
after the collective completes (see Pattern B stacking with Pattern C).

### Confirmation Steps

1. Add a Tracy zone around the CCL all-to-all call:

```python
with tracy.zone("MoE/dispatch/all_to_all"):
    redistributed_tokens = ttnn.experimental.all_to_all(
        gathered_tokens,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
```

2. Re-run the trace. Measure the `MoE/dispatch/all_to_all` zone duration as a function of
   seq_len.
3. If the zone duration grows linearly with seq_len, the gap is dominated by CCL latency
   (Pattern C confirmed).

### Remedy

See Chapter 7 (`gap_to_action_mapping.md`) for the full optimization plan. The primary lever
is `ttnn.experimental.ccl.all_to_all_async`, which allows CCL communication to overlap with
local computation from a preceding layer. A secondary lever is reducing the expert parallelism
degree (`ep_degree`) if hardware utilization permits running multiple experts per chip.

---

## Pattern D: Gap at the Beginning of the MoE Layer — Program Cache Miss

### Description

A gap of 50–500ms appears at the very start of the `MoE/forward` zone, before any dispatch
sub-zones begin. The gap is present on the first call and on any subsequent call where the
input tensor shapes differ from the previous call.

### Tracy Zone Signature

```
Timeline (dispatch thread, first call):
  [MoE/forward                                                        ]
  [          recompilation gap          ][MoE/dispatch/router ][...]
   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   200ms compilation stall (first call only)

Timeline (dispatch thread, subsequent calls with same shapes):
  [MoE/forward                         ]
  [MoE/dispatch/router ][MoE/dispatch/topk ][...]
  (no gap — program cache hit)
```

### Root Cause

tt-metal compiles TTNN op kernels on first use for a given set of op parameters (tensor
shapes, dtypes, memory layouts, grid configurations). The compiled program is stored in the
program cache. If the program cache is cold (first run, or cache invalidated by a shape
change), the compilation step blocks the host thread.

At the MoE layer, compilation is triggered when:

- The first forward pass after process startup (cold cache).
- The input `seq_len` changes between calls, producing new tile grid dimensions that require
  a new compiled program.
- A tensor's memory layout changes (e.g., a different `MemoryConfig` or shard spec).

The compilation latency is independent of seq_len; it depends on the number of unique ops
and the complexity of each kernel. For a full MoE forward pass with ~13 ops, cold compilation
may take 200–500ms.

### Confirmation Steps

1. Check whether the gap appears only on the first 1–3 calls and disappears on subsequent
   calls with identical shapes:

```python
import time

for i in range(6):
    t0 = time.perf_counter_ns()
    output = moe_forward(hidden_states, router_weights)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter_ns()
    print(f"Iteration {i}: {(t1 - t0) / 1e6:.1f} ms")

# Expected output:
# Iteration 0: 523.4 ms  (cold cache: compilation)
# Iteration 1:  18.2 ms  (warm cache)
# Iteration 2:  18.1 ms
# Iteration 3:  18.0 ms
# ...
```

2. Introduce a shape change and verify the gap reappears:

```python
# Warm up with seq_len=1024
for _ in range(3):
    moe_forward(hidden_states_1024, router_weights)

# Change seq_len → new shapes → cache miss
t0 = time.perf_counter_ns()
moe_forward(hidden_states_512, router_weights)
t1 = time.perf_counter_ns()
print(f"After shape change: {(t1 - t0) / 1e6:.1f} ms")  # should be high again
```

### Remedy

1. Call `ttnn.enable_program_cache()` before the first inference call. This is usually
   already set in production; verify it is not disabled in test harnesses.
2. Pre-warm the program cache by running 3 forward passes before recording timing.
3. Pad or canonicalize input tensor shapes to a fixed set. If the model runs at multiple
   seq_len values, run one warm-up pass at each supported seq_len to pre-compile all programs.

---

## Distinguishing Pattern C from Pattern D

Pattern C (CCL all-to-all) and Pattern D (program cache miss) can both produce a gap early
in the MoE forward pass and both may appear on the first call. The key distinction is scaling
behavior:

| Property | Pattern C (CCL latency) | Pattern D (recompilation) |
|---|---|---|
| **Location in trace** | Between `MoE/dispatch/gather` end and `MoE/expert_matmul` start | At the beginning of `MoE/forward`, before any child zones start |
| **First-call-only?** | No — present every iteration | Yes — absent after warm-up (unless shapes change) |
| **Scales with seq_len?** | Yes — linearly (message size = seq_len × top_k × d_model) | No — fixed latency per unique shape |
| **Scales with d_model?** | Yes — linearly | No |
| **Present on single-chip (no T3K)?** | No — CCL ops do not execute without multi-chip setup | Yes — compilation happens on every device configuration |
| **Device profiler evidence** | `OP TO OP LATENCY [ns]` between gather and first matmul is large; CCL kernel may appear in device profiler CSV | No kernel activity in device profiler during the gap (device is idle, host is compiling) |

The most reliable test: increase `d_model` by 2× (or equivalently, double `seq_len`) and
observe whether the gap duration also approximately doubles. If yes, it is Pattern C. If the
gap duration is unchanged, it is Pattern D.

```python
# Scaling test to distinguish Pattern C from Pattern D
import time

results = {}
for seq_len in [256, 512, 1024, 2048]:
    # Ensure program cache is warm for this seq_len (3 warm-up runs)
    for _ in range(3):
        moe_forward(make_hidden_states(seq_len), router_weights)

    # Measure gap (use Tracy CSV export for precision; wallclock here for illustration)
    gaps = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        moe_forward(make_hidden_states(seq_len), router_weights)
        t1 = time.perf_counter_ns()
        gaps.append((t1 - t0) / 1e6)

    import statistics
    results[seq_len] = statistics.median(gaps)
    print(f"seq_len={seq_len:5d}: {results[seq_len]:.2f} ms")

# Compute scaling ratios
seq_lens = sorted(results)
for i in range(1, len(seq_lens)):
    ratio = results[seq_lens[i]] / results[seq_lens[i - 1]]
    print(f"Ratio {seq_lens[i]}/{seq_lens[i-1]}: {ratio:.2f}x "
          f"({'linear CCL' if 1.7 < ratio < 2.3 else 'constant/recompile'})")
```

A doubling ratio between 1.7× and 2.3× per 2× seq_len increase is consistent with linear
CCL scaling (Pattern C). A ratio near 1.0× is consistent with a fixed-cost operation such
as recompilation (Pattern D) or a synchronization barrier (Pattern B).

---

---

**Next:** [Chapter 6 — Sequence Length Scaling Analysis](../ch6_sequence_length_scaling/index.md)
