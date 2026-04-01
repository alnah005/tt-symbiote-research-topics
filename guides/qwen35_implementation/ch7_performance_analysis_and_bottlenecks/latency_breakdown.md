# Latency Breakdown

## Measured decode performance

All numbers are from single-token steady-state decode on a single P100A Blackhole,
reported in `PERF.md`, measured on the 35B-A3B model (A3B hereafter).

| Model | Precision | Speed |
|---|---|---|
| Qwen3.5-35B-A3B | bfp4 experts, f32 recurrence | 11.7 tok/s |
| Qwen3.5-27B | bfp8 weights, f32 recurrence | 6.28 tok/s |

The A3B is the primary profiling target because it is faster (fewer and cheaper layers)
and representative of the MoE + hybrid-attention design.

## Per-component timing (A3B, 86 ms/token)

The forward pass is profiled at the component level:

| Component | Time | Syncs | Notes |
|---|---|---|---|
| DeltaNet (30 layers) | 54 ms | 30 | Host recurrence: 1 sync/layer (see note) |
| Attention (10 layers) | 18 ms | ~50 | Host partial RoPE era: 5 syncs/layer (3 to_torch + 2 from_torch) |
| norm + LM head | 14 ms | 1 | Host embedding lookup; 248 K vocab stays on host |
| **Total** | **86 ms** | **~81** | See `sync_overhead.md` for discrepancy with PERF.md's "~70" |

**Note on profiling era vs current code:** The table was captured with the historical
host-recurrence and host-RoPE implementations. Two subsequent fixes eliminated most
of these syncs:

1. `ttnn.experimental.gated_delta_net` fused kernel: replaces 30 DeltaNet host-recurrence
   syncs with zero (device-side fp32 state, in-place `ttnn.copy` update).
2. Device-side partial RoPE via patched cos/sin matrices (Chapter 3): replaces 5 host syncs
   per attention layer (3 `to_torch` + 2 `from_torch` in `partial_rope_fn`) with zero.
   `HfRotarySetup.get_rot_mats()` returns cached matrices without any host sync; position
   slicing is handled internally by `ttnn.experimental.rotary_embedding` on device.

## Three-way time split

The 86 ms is decomposed into three orthogonal cost categories:

| Category | Time | Source |
|---|---|---|
| Host-device synchronisation | ~35 ms | Blocking barriers waiting for device to drain |
| Python dispatch | ~26 ms | CPU time building and submitting TTNN ops |
| Device compute | ~20 ms | Actual tensor math executing on Blackhole |

These three categories are independent: reducing sync overhead does not automatically
reduce dispatch overhead, and vice versa. Each requires a different remedy.

## Theoretical peak and efficiency

P100A Blackhole specs give a theoretical maximum of approximately 172 tok/s for a model
of this size at decode batch = 1. The achieved efficiency is:

```
11.7 tok/s ÷ 172 tok/s ≈ 6.8%
```

Stated differently, for every millisecond the device spends on useful matrix math,
the system spends roughly 14 ms on synchronisation and Python overhead combined.

## Timing measurement in demo code

The demo (`demo_a3b.py`) measures per-step wall time after `ttnn.synchronize_device`:

```python
t0 = time.perf_counter()
for layer in model.layers:
    x = layer(x, current_pos=tt_pos, rot_mats_global=rot_mats, mode=Mode.DECODE)
x = model.norm(x, ...)
logits_tt = ttnn.linear(x, lm_weight_tt, ...)
ttnn.synchronize_device(device)
dt = time.perf_counter() - t0
```

`ttnn.synchronize_device` blocks until all previously submitted device commands complete,
so `dt` captures the entire forward pass including all Python dispatch and all
host-device sync overhead, not just device compute time.

Steps 0 and 1 are excluded from the average (step 0 = program compile, step 1 = program
cache warmup); steady-state timing starts at step 2.

---

**Next:** [`sync_overhead.md`](./sync_overhead.md)
