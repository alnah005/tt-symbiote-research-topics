# Rolling Window L1 State Design

The core idea behind L1 state management is straightforward: if GDN recurrence state lives in L1 instead of DRAM, the fused kernel no longer needs to issue NOC reads and writes for the 16 state tiles per pair. But Qwen3.5-27B has 48 GDN layers, each holding a recurrence state of shape `[B*Nv_TP, Dk, Dv]` = `[384, 128, 128]` per device. At bfloat16 precision that is exactly 12,582,912 bytes (12.0 MB) per layer. The total state footprint of 576 MB far exceeds the usable L1 per core, and even across all cores on a Blackhole chip the aggregate L1 cannot hold all 48 layers simultaneously.

## Profiler Breakdown

Non-traced decode profiling establishes the bottleneck clearly:

| Component | Time (ms) | Share |
|-----------|-----------|-------|
| GDN (48 layers) | 469.6 | 85% |
| Attention (16 layers) | 69.2 | 12% |
| Overhead | 15.7 | 3% |

Within each GDN layer the fused kernel issues 16 NOC tile reads and 16 NOC tile writes for the recurrence state per pair. With 384 pairs total, that is 12,288 NOC transactions per layer and roughly 590,000 state NOC transactions per full forward pass. The state round-trip across all 48 layers accounts for approximately 1.2 GB of DRAM bandwidth per decode step.

## Layer Pattern and Window Size

The solution exploits the model's repeating layer pattern: 3 GDN layers followed by 1 attention layer, repeating 16 times across the 64-layer model. At any point during the forward pass only one group of 3 consecutive GDN layers needs to be in L1. The attention layer that follows each group provides a natural swap point: the current group's states are saved to DRAM and the next group's states are loaded before continuing.

The window size `_l1_window = 3` (set at `model.py` line 227) matches this 3-GDN + 1-Attention structure directly.

## The `enable_l1_state()` Method

`enable_l1_state()` in the `Transformer` class (`model.py`, lines 220-249) initializes the rolling window in three steps:

**Step 1 — Build the GDN index list.** It scans `self.args.layer_types` for all `"linear_attention"` entries and records their model-layer indices in `self._gdn_indices`. For Qwen3.5-27B this produces a list of 48 indices.

**Step 2 — Preserve DRAM backup references.** For each GDN layer the method calls `gdn.reset_state()` if the layer has no state yet, then stores a reference to the existing DRAM-resident `rec_states` as `gdn._dram_state` (line 240). This pre-allocated buffer is reused throughout inference — states are copied back into it rather than allocating new DRAM tensors, which avoids memory fragmentation.

**Step 3 — Load the first window to L1.** The first 3 GDN layers (indices 0, 1, 2 in `_gdn_indices`) have their states moved to L1 via (lines 243-246):

```python
l1_state = ttnn.to_memory_config(gdn._dram_state, ttnn.L1_MEMORY_CONFIG)
gdn.rec_states = l1_state
```

The tracker `self._l1_current_start = 0` records which group is currently in L1 (0-based index into groups of 3).

## The `_swap_l1_state()` Method

`_swap_l1_state(old_start, new_start)` (`model.py`, lines 251-278) performs the bidirectional transfer between L1 and DRAM.

**Save phase — L1 to DRAM.** For each GDN layer in the old group the method inspects `gdn.rec_states.memory_config().buffer_type` (line 265). If the buffer type is `ttnn.BufferType.L1` it copies the data back to the pre-allocated DRAM buffer using the `output_tensor` parameter (line 267):

```python
ttnn.to_memory_config(gdn.rec_states, ttnn.DRAM_MEMORY_CONFIG, output_tensor=gdn._dram_state)
ttnn.deallocate(gdn.rec_states)
gdn.rec_states = gdn._dram_state
```

The `output_tensor` parameter writes into the existing `_dram_state` buffer rather than allocating a new one. After the copy the L1 tensor is explicitly deallocated, freeing L1 space for the next group.

**Load phase — DRAM to L1.** For each GDN layer in the new group the method allocates fresh L1 storage and copies the DRAM state into it (lines 271-278):

```python
l1_state = ttnn.to_memory_config(gdn._dram_state, ttnn.L1_MEMORY_CONFIG)
gdn.rec_states = l1_state
```

The bounds check `if gi >= len(self._gdn_indices): break` (lines 263, 274) handles the last group gracefully when the total GDN layer count is not evenly divisible by the window size.

## Forward Pass Hook Mechanism

The `forward()` method (`model.py`, lines 280-347) injects swap logic into the layer loop without modifying the parent `TTTransformer.forward()`. This is achieved by temporarily monkey-patching each GDN layer's `forward` method.

**Guard clause (line 291).** If `_l1_state_enabled` is `False` or the mode is not `Mode.DECODE`, the method delegates directly to `super().forward()`. L1 state management only applies to decode — prefill uses separate B=1 states.

**Ensure block 0 is loaded (lines 310-312).** Before the layer loop begins the method checks if `current_block_wrapper[0] != 0`. If a previous forward call left a different block in L1, it swaps back to block 0.

**Wrap each GDN layer's forward (lines 316-331).** For every index in `gdn_set`, the method saves the original `layer.forward` and replaces it with a closure. The `make_wrapped_forward` factory captures the GDN layer's sequential index `gdn_i` and computes the needed block at call time:

```python
def make_wrapped_forward(orig_fwd, layer_i, gdn_i):
    def wrapped_forward(*args, **kwargs):
        needed_block = gdn_i // W
        if needed_block != current_block_wrapper[0]:
            self._swap_l1_state(current_block_wrapper[0] * W, needed_block * W)
            current_block_wrapper[0] = needed_block
        return orig_fwd(*args, **kwargs)
    return wrapped_forward
```

The factory function is necessary to avoid Python's closure late-binding problem: without it, all closures would capture the same final value of `gdn_i` from the loop.

**try/finally (lines 334-345).** The parent `super().forward()` runs the full layer loop, norm, and LM head. The `finally` block restores original forwards and updates `self._l1_current_start`, even if an exception occurs.

## Swap Timing

The swap fires exactly once per block, triggered by the first GDN layer in each new block. The pattern for the 48-layer model:

| GDN index (`gdn_i`) | Block (`gdn_i // 3`) | Swap needed? |
|---|---|---|
| 0, 1, 2 | 0 | No (pre-loaded by `enable_l1_state()`) |
| 3, 4, 5 | 1 | Yes, before GDN index 3 |
| 6, 7, 8 | 2 | Yes, before GDN index 6 |
| ... | ... | ... |
| 45, 46, 47 | 15 | Yes, before GDN index 45 |

The attention layer between each group runs after the old group's states have already been saved to DRAM by the swap that precedes the next group.

## Memory Lifecycle per Decode Step

1. GDN layers 0-2 execute with `rec_states` pointing to L1 tensors. Fused kernel reads and writes state entirely in L1.
2. Before GDN index 3: `_swap_l1_state(0, 3)` saves layers 0-2 state L1 to DRAM, deallocates L1, loads layers 3-5 state DRAM to L1.
3. GDN layers 3-5 execute with `rec_states` in L1.
4. Pattern repeats for all 16 groups.
5. After the last layer `_l1_current_start` records the final block for the next decode step.

Pre-allocated `_dram_state` buffers ensure zero DRAM allocation overhead during swaps. Only L1 allocation and deallocation occur, which is fast because L1 is managed as a bump allocator within the ttnn runtime.

---

**Next:** [`height_sharded_kernel.md`](./height_sharded_kernel.md)
