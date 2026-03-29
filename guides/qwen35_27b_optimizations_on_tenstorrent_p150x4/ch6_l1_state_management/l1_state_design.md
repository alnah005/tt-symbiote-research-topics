# Rolling Window L1 State Design

The core idea behind L1 state management is straightforward: if GDN recurrence state lives in L1 instead of DRAM, the fused kernel no longer needs to issue NOC reads and writes for the 16 state tiles per pair. But Qwen3.5-27B has 48 GDN layers, each holding 12 MB of state per device. The total state footprint of 576 MB far exceeds the ~1.5 MB of usable L1 per core. Even across all cores on a Blackhole chip, the aggregate L1 capacity cannot hold all 48 layers simultaneously.

The solution exploits the model's repeating layer pattern: 3 GDN layers followed by 1 attention layer, repeating 16 times. At any point during the forward pass, only one group of 3 consecutive GDN layers needs to be in L1. The attention layer that follows each group provides a natural swap point where the current group's states can be saved back to DRAM and the next group's states can be loaded.

## The `enable_l1_state()` Method

The `enable_l1_state()` method in the `Transformer` class (`model.py`, lines 220-249) initializes the rolling window. It performs three steps:

1. **Build the GDN layer index list.** It scans `self.args.layer_types` for all `"linear_attention"` entries and records their indices in `self._gdn_indices`. For Qwen3.5-27B, this produces a list of 48 indices.

2. **Preserve DRAM backup references.** For each GDN layer, it stores a reference to the current DRAM-resident `rec_states` tensor as `gdn._dram_state`. This pre-allocated DRAM buffer will be reused throughout inference -- states are copied back into it (rather than allocating new DRAM tensors) to avoid memory fragmentation.

3. **Load the first window to L1.** The first 3 GDN layers (indices 0, 1, 2 in `self._gdn_indices`) have their states moved to L1 via `ttnn.to_memory_config(gdn._dram_state, ttnn.L1_MEMORY_CONFIG)`. The returned L1 tensor replaces `gdn.rec_states`, so the fused kernel will read from and write to L1 on the next forward pass.

The window size `self._l1_window = 3` matches the model's 3-GDN + 1-Attention repeating pattern. The tracker `self._l1_current_start = 0` records which group of GDN layers currently resides in L1 (0-based index into groups of 3).

## The `_swap_l1_state()` Method

`_swap_l1_state(old_start, new_start)` (`model.py`, lines 251-278) performs the bidirectional transfer between L1 and DRAM:

**Save phase (L1 to DRAM):** For each GDN layer in the old group (`old_start` to `old_start + W - 1`), it checks whether the state is currently in L1 by inspecting `gdn.rec_states.memory_config().buffer_type`. If the buffer type is `ttnn.BufferType.L1`, the method copies the data back to the pre-allocated DRAM buffer using:

```python
ttnn.to_memory_config(gdn.rec_states, ttnn.DRAM_MEMORY_CONFIG, output_tensor=gdn._dram_state)
```

The `output_tensor` parameter is critical: it writes into the existing `_dram_state` buffer rather than allocating a new one. After the copy, the L1 tensor is explicitly deallocated with `ttnn.deallocate(gdn.rec_states)`, freeing L1 space for the next group. The `rec_states` reference is then pointed back to the DRAM backup.

**Load phase (DRAM to L1):** For each GDN layer in the new group (`new_start` to `new_start + W - 1`), it allocates fresh L1 storage and copies the DRAM state into it:

```python
l1_state = ttnn.to_memory_config(gdn._dram_state, ttnn.L1_MEMORY_CONFIG)
gdn.rec_states = l1_state
```

The bounds check `if gi >= len(self._gdn_indices): break` handles the last group gracefully when the total GDN layer count is not evenly divisible by the window size.

## Forward Pass Hook Mechanism

The `forward()` method (`model.py`, lines 280-347) injects L1 swap logic into the layer loop without modifying the parent `TTTransformer.forward()`. This is achieved through a temporary monkey-patching mechanism:

1. **Guard clause.** If `_l1_state_enabled` is `False` or the mode is not `Mode.DECODE`, the method delegates directly to `super().forward()`. L1 state management only applies to decode -- prefill uses separate B=1 states (Chapter 5).

2. **Ensure block 0 is loaded.** Before the layer loop begins, the method checks if block 0 is currently in L1. If a previous forward call left a different block loaded, it swaps to block 0.

3. **Wrap each GDN layer's forward.** For every layer index in `gdn_set`, the method saves the original `layer.forward` and replaces it with a wrapped version. The wrapper closure captures the GDN layer's sequential index (`gdn_i`) and computes the needed block:

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

The `make_wrapped_forward` factory function is necessary to avoid the Python closure late-binding problem -- without it, all closures would capture the same `gdn_i` value.

4. **Call parent forward in try/finally.** The wrapped `super().forward()` runs the full layer loop, norm, and LM head. The `finally` block guarantees that original forwards are restored and `_l1_current_start` is updated, even if an exception occurs.

## Swap Timing and Layer Pattern

The swap pattern follows the model's 3+1 structure:

| GDN index (gdn_i) | Block (gdn_i // 3) | Layer indices | Swap needed? |
|---|---|---|---|
| 0, 1, 2 | 0 | layers 0, 1, 2 | No (pre-loaded) |
| 3, 4, 5 | 1 | layers 4, 5, 6 | Yes, before layer 4 |
| 6, 7, 8 | 2 | layers 8, 9, 10 | Yes, before layer 8 |
| ... | ... | ... | ... |
| 45, 46, 47 | 15 | layers 60, 61, 62 | Yes, before layer 60 |

Each swap occurs exactly once per group, triggered by the first GDN layer in the new block. The intervening attention layer (every 4th layer) runs between groups, during which time the old group's L1 state has already been saved back to DRAM by the swap that precedes the new group.

## Memory Lifecycle

The state tensor lifecycle during a single decode step:

1. **Initial state:** First 3 GDN layers' `rec_states` point to L1 tensors; remaining 45 layers' `rec_states` point to DRAM tensors (identical to `_dram_state`).
2. **GDN layers 0-2 execute:** Fused kernel reads/writes state from/to L1 in-place.
3. **Before GDN layer 3:** `_swap_l1_state(0, 3)` runs. Layers 0-2 states copied L1 to DRAM (into `_dram_state`), L1 deallocated. Layers 3-5 states copied DRAM to L1.
4. **GDN layers 3-5 execute:** Fused kernel reads/writes state from L1.
5. **Pattern repeats** through all 16 groups.
6. **After last layer:** `_l1_current_start` records the final block, so the next decode step knows where to resume.

The pre-allocated `_dram_state` buffers ensure zero DRAM allocation overhead during the swap. Only L1 allocation and deallocation occur, which is fast because L1 is managed as a simple bump allocator within the ttnn runtime.

---

**Previous:** [`index.md`](./index.md) | **Next:** [`height_sharded_kernel.md`](./height_sharded_kernel.md)
