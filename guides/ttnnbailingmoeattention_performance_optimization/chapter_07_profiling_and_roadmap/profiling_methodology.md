# Profiling Methodology

## Measuring Total Decode Step Latency

Wrap the full `_forward_decode_paged` call with `ttnn.synchronize_device` before and after to flush the command queue and get accurate wall-clock timing:

```python
import time

ttnn.synchronize_device(device)
t0 = time.perf_counter()
for _ in range(N_REPEATS):
    output, _ = module._forward_decode_paged(
        hidden_states, position_embeddings, attention_mask,
        past_key_values, cache_position
    )
ttnn.synchronize_device(device)
t1 = time.perf_counter()

total_ms = (t1 - t0) / N_REPEATS * 1000
print(f"Decode step latency: {total_ms:.3f} ms")
```

Use `N_REPEATS=100` to amortize Python dispatch overhead. Run at B=1 (latency-critical case) and B=32 (batch throughput case).

---

## Isolating Individual Op Latencies

To measure a specific op, insert `ttnn.synchronize_device` calls bracketing that op. This stalls the device pipeline and forces sequential execution, so use it only for profiling — not in production.

### Measuring the SDPA kernel

```python
# Before paged_sdpa_decode call (inside _forward_decode_paged)
ttnn.synchronize_device(device)
t0 = time.perf_counter()
attn_output = module.sdpa.paged_sdpa_decode(query_states, layer_idx, ...)
ttnn.synchronize_device(device)
t1 = time.perf_counter()
sdpa_ms = (t1 - t0) * 1000
```

### Measuring collective operations (reduce-scatter and all_gather)

Each collective in the decode path (lines 2624, 2626, 2631, 2632, 2633 of `attention.py`) can be timed with the same bracket pattern. The five collective positions in the decode path are:

1. Line 2624: `q_proj(hidden_states)` — contains the reduce-scatter inside `TTNNLinearIColShardedWRowSharded`
2. Line 2626: `ttnn.all_gather(hidden_states, dim=-1)` — explicit all_gather for K/V input
3. Line 2631: `_maybe_all_gather(query_states)` — conditional all_gather (no-op if already gathered)
4. Line 2632: `_maybe_all_gather(key_states)` — conditional all_gather (no-op if already gathered)
5. Line 2633: `_maybe_all_gather(value_states)` — conditional all_gather (no-op if already gathered)

### Measuring `to_memory_config` transitions

The nine `ttnn.to_memory_config` calls in the decode path (steps 8a, 8b, 12a, 12b, 12c, 12d, 16a, 16b, 20 per Chapter 1) can each be timed individually with the synchronize bracket. The two highest-value targets are:

- **Step 8a (line 2656)**: `ttnn.to_memory_config(query_states, L1_MEMORY_CONFIG)` — 131,072 bytes at B=32
- **Step 8b (line 2657)**: `ttnn.to_memory_config(key_states, L1_MEMORY_CONFIG)` — 32,768 bytes at B=32

### Measuring host-device round-trips

Host touches cannot be timed with `ttnn.synchronize_device` alone — they require wall-clock timing on the host side. The key transfers are:

- **Line 2674 (conditional)**: `ttnn.to_torch(cache_position)` — only if `cache_position` is a `ttnn.Tensor`
- **Lines 2678–2685**: `torch.tensor([cur_pos])` + `ttnn.from_torch(...)` for `cur_pos_tt`
- **`rope.py` line 444**: `ttnn.from_torch(...)` inside `get_cos_sin_for_decode`

Measure these with Python `time.perf_counter()` bracketing the relevant lines.

---

## Key Metrics to Collect

For each profiled operation:

| Metric | How to collect |
|---|---|
| Wall-clock kernel time (ms) | `ttnn.synchronize_device` bracket |
| Data movement bytes | Known from tensor shape and dtype (Chapter 3 baselines) |
| Relative % of total decode latency | `op_ms / total_ms * 100` |

---

## Tracy / Device Event Handler

For more detailed profiling (L1 occupancy, NoC utilization, per-core breakdown), `ttnn` supports Tracy integration and `ttnn.perf_device_event_handler`. These provide hardware performance counter data beyond what wall-clock timing can show. To collect tracy traces:

```bash
# Set environment variable before launching the Python script
TTNN_PROFILE_DEVICE=1 python my_decode_script.py
```

Tracy traces can be visualized in the tracy profiler and show per-op dispatch timeline, stalls, and data movement at the hardware level. This is the recommended tool for diagnosing scheduling gaps and NoC congestion that wall-clock timing cannot resolve.

---

## Interpreting Results

When analyzing profiling output, look for:

1. **Long stalls between back-to-back ops**: indicates a synchronization barrier (likely an `all_gather` completing before the next op can dispatch). The five collective ops in the decode path are the primary source.

2. **Op durations disproportionate to their data volume**: a `to_memory_config` that moves 32 KB should be much faster than one that moves 131 KB. If they take similar time, the bottleneck may be kernel dispatch overhead rather than bandwidth.

3. **PCIe transfer spikes**: the host-device round-trips for `cur_pos_tt` and `get_cos_sin_for_decode` show up as PCIe transfers that interrupt the device timeline. These should appear as brief but blocking events between other ops.

4. **SDPA kernel occupancy**: in the compute-bound regime (long KV cache), SDPA should dominate. If SDPA is much shorter than expected, it may be bandwidth-bound (short KV cache, waiting on DRAM reads) rather than compute-bound.

---

**Next:** [Bottleneck Ranking and Optimization Roadmap](bottleneck_ranking.md)
