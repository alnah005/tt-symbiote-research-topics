# Conv1d Shift Register: Trace-Compatible Causal Convolution

The GDN layer applies a 4-tap causal conv1d to the concatenated Q, K, V projections before they enter the recurrence. In a standard implementation, this would use a circular buffer with a write pointer that advances each step -- but circular buffers with dynamic indexing are incompatible with `ttnn` tracing, which requires a fixed operation graph that can be replayed without control-flow changes. The solution is a shift register: a fixed chain of `ttnn.copy` operations that shifts values through 4 state slots, followed by a weighted sum using the learned convolution taps.

## Shift Register State

The conv1d state consists of 4 tensors stored in `self.conv_states`, a Python list:

```python
self.conv_states = [
    _to_mesh(torch.zeros(1, B, self.qkv_dim_tp, dtype=torch.bfloat16))
    for _ in range(self.conv_kernel_size)  # conv_kernel_size = 4
]
```

Each tensor has shape `[1, B, qkv_dim_tp]` = `[1, 32, 2560]`, holding one time step's worth of QKV projections for all batch elements. The four slots represent the current and three previous time steps:

- `states[0]`: oldest value (t-3)
- `states[1]`: t-2
- `states[2]`: t-1
- `states[3]`: newest value (t), replaced each step

## The Shift Operation

Each decode step shifts the register by copying each slot to its predecessor, then writes the new input into the last slot:

```python
states = self.conv_states
ttnn.copy(states[1], states[0])   # states[0] = states[1]  (discard oldest)
ttnn.copy(states[2], states[1])   # states[1] = states[2]
ttnn.copy(states[3], states[2])   # states[2] = states[3]
ttnn.copy(qkv_tt, states[3])      # states[3] = new input
```

The `ttnn.copy(src, dst)` operation copies the data from `src` into `dst` in-place, preserving the tensor ID of the destination. This is the key property that makes the shift register trace-compatible: the tensor IDs in `self.conv_states` never change across decode steps. The trace system records these copy operations and can replay them identically on every subsequent step.

Note the copy order: oldest-first (0, 1, 2, 3). This avoids data loss -- if the newest slot were copied first, it would overwrite a value before it had been shifted down. Since `states[0]` is about to be overwritten, it is safe to write into it first.

## Weighted Sum

After shifting, the convolution output is computed as a weighted sum of all 4 slots using the learned tap weights:

```python
conv_acc = ttnn.multiply(states[0], tw["conv_taps"][0])
for j in range(1, self.conv_kernel_size):
    conv_acc = ttnn.mac(states[j], tw["conv_taps"][j], conv_acc)
conv_out = ttnn.silu(conv_acc)
```

The `tw["conv_taps"]` list contains 4 weight tensors, each of shape `[1, 1, qkv_dim_tp]` = `[1, 1, 2560]`, broadcast across the batch dimension. These weights are prepared during model loading by the `prepare_conv_taps` function in `model_config.py`, which extracts per-TP shards of the learned convolution kernel.

The computation sequence:
1. `conv_acc = states[0] * conv_taps[0]` -- element-wise multiply for the oldest tap
2. `conv_acc = states[1] * conv_taps[1] + conv_acc` -- multiply-accumulate via `ttnn.mac`
3. `conv_acc = states[2] * conv_taps[2] + conv_acc` -- multiply-accumulate
4. `conv_acc = states[3] * conv_taps[3] + conv_acc` -- multiply-accumulate (newest tap)

The `ttnn.mac(a, b, c)` operation computes `a * b + c` in a single kernel dispatch, avoiding an intermediate tensor allocation compared to separate `ttnn.multiply` and `ttnn.add` calls.

The result is a causal convolution: the output at time `t` depends only on inputs at times `t, t-1, t-2, t-3`, with no future information leaking in.

## SiLU Activation

The final `ttnn.silu(conv_acc)` applies the SiLU (Sigmoid Linear Unit) activation element-wise:

```
SiLU(x) = x * sigmoid(x)
```

This is the standard nonlinearity used in the Qwen3.5 GDN conv1d, matching the reference HuggingFace implementation. The activated output `conv_out` has the same shape `[1, B, qkv_dim_tp]` = `[1, 32, 2560]` and contains the convolved, activated Q/K/V values ready for the recurrence stage.

## Why the Shift Register is Trace-Compatible

The `ttnn` tracing system records a sequence of operations and their tensor operands (identified by tensor ID) to build a replayable execution graph. For trace compatibility, operations must satisfy two constraints:

1. **Fixed tensor IDs**: The same tensor IDs must appear in the same roles on every replay. The shift register satisfies this because `ttnn.copy` writes into existing tensors without creating new ones -- `states[0]` through `states[3]` keep their IDs across all decode steps.

2. **No dynamic control flow**: The operation sequence must be identical on every step. The shift register uses a fixed 4-copy chain followed by a fixed multiply-accumulate sequence -- no conditionals, no variable-length loops, no dynamic indexing.

A circular buffer alternative (e.g., maintaining a write pointer `idx = step % 4` and indexing `states[idx]`) would violate constraint 2: the Python-level index computation would produce different operation sequences depending on `step`, and the trace would not be replayable.

## State Reset

Two reset methods maintain trace compatibility:

- `reset_state()`: Creates fresh zero tensors, replacing `self.conv_states` entirely. Used during initialization.
- `reset_state_inplace()`: Zeros the existing tensors via `ttnn.copy` from a temporary zero tensor, preserving tensor IDs. Used between batches during traced execution.

```python
def reset_state_inplace(self):
    zeros_conv = _to_mesh(torch.zeros(1, B, self.qkv_dim_tp, dtype=torch.bfloat16))
    for cs in self.conv_states:
        ttnn.copy(zeros_conv, cs)
    ttnn.deallocate(zeros_conv)
```

The `reset_state_inplace` pattern is essential for traced decode loops: the trace captures the copy-from-zeros operations, and replaying them correctly resets the state without changing tensor IDs.

## Memory Footprint

Each conv state slot occupies `1 * 32 * 2560 * 2 = 160 KB` in bfloat16 (before tile padding). With 4 slots, the total conv state per layer is approximately `640 KB`. Across all 48 GDN layers, this is `~30 MB` per device -- a small fraction of the recurrence state memory (which is `~576 MB` per device for 48 layers at 12 MB each).

---

**Previous:** [`gdn_decode_flow.md`](./gdn_decode_flow.md) | **Next:** [`recurrence_math.md`](./recurrence_math.md)
