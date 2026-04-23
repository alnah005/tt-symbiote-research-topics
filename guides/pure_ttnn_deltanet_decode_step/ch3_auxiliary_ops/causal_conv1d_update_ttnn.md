# Causal Conv1D State Update in TTNN

This file derives a TTNN-native implementation of the causal conv1d state update for the decode path of `TTNNQwen3LinearAttention`. The current implementation calls the `causal_conv1d_update` C extension (a CUDA/CPU kernel), which forces a host crossing. The TTNN decomposition replaces this with a sequence of `ttnn.slice`, `ttnn.concat`, `ttnn.mul`, and `ttnn.sum` ops that run entirely on the Wormhole device.

---

## 1. What the Causal Conv1D Decode Update Computes

At decode time (T=1), the causal conv1d performs a sliding-window update on a persistent state buffer. The operation maintains a window of the last K input vectors and computes a weighted sum to produce the current output.

Let:
- `x` = new input vector, shape `[B, channels, 1]`
- `conv_state` = rolling state buffer, shape `[B, channels, K]` where `K=4`
- `conv_weight` = learned depthwise convolution weight, shape `[channels, K]`

The update rule is:

```
# Shift state left by 1: discard oldest, make room for new input
conv_state[:, :, 0:K-1]  =  conv_state[:, :, 1:K]   (shift)
conv_state[:, :, K-1]    =  x[:, :, 0]               (append new input)

# Compute output: weighted sum over the K-slot window
output = sum_{i=0}^{K-1} conv_weight[:, i] * conv_state[:, :, i]
       = sum(conv_weight_broadcast * conv_state, dim=-1)
```

The result `output` has shape `[B, channels, 1]`.

**Key values for Qwen3.6-35B-A3B under T3K head-parallel sharding:**
- `B = 1`
- `channels = mixed_dim = key_dim × 2 + value_dim = 2048 × 2 + 4096 = 8192` (full); per device = `8192 / 8 = 1024`
- `K = 4`

---

## 2. TTNN Decomposition

### 2a. State Shift — `ttnn.slice` + `ttnn.concat`

The state shift moves the last `K-1` slots of `conv_state` into positions `0` through `K-2`, then places the new input in slot `K-1`.

```python
# conv_state: [B, channels_local, K]  e.g. [1, 1024, 4]
# x_reshaped: [B, channels_local, 1]  e.g. [1, 1024, 1]

# Extract slots 1 through K-1 (the K-1 most recent past inputs)
shifted = ttnn.slice(conv_state, begins=[0, 0, 1], ends=[B, channels_local, K])
# shifted.shape: [1, 1024, 3]

# Append the new input as the newest slot
conv_state_new = ttnn.concat([shifted, x_reshaped], dim=2)
# conv_state_new.shape: [1, 1024, 4]
```

> **Note:** `ttnn.slice` allocates a new output buffer (it is a copy, not a zero-copy view); `ttnn.concat` also allocates a combined output buffer. Both produce intermediate tensors that require attention for full Metal Trace compatibility — see Section 4 for the required pre-allocation strategy.

### 2b. Depthwise Convolution Output — `ttnn.mul` + `ttnn.sum`

```python
# conv_weight: [channels_local, K]  e.g. [1024, 4]
# Broadcast over B dimension: reshape to [1, channels_local, K]
conv_weight_broadcast = ttnn.reshape(conv_weight, [1, channels_local, K])
# conv_weight_broadcast.shape: [1, 1024, 4]

# Element-wise multiply: applies the same weight at each position
weighted = ttnn.mul(conv_state_new, conv_weight_broadcast)
# weighted.shape: [1, 1024, 4]

# Sum over the K dimension to produce the output
output = ttnn.sum(weighted, dim=2, keepdim=True)
# output.shape: [1, 1024, 1]
```

The result `output` is the convolved output for the current decode step.

### 2c. Complete Step

```python
def causal_conv1d_decode_update_ttnn(
    x: ttnn.Tensor,             # [B, channels_local, 1]
    conv_state: ttnn.Tensor,    # [B, channels_local, K]  — persistent DRAM tensor
    conv_weight: ttnn.Tensor,   # [channels_local, K]  — weight tensor on device
    K: int = 4,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Returns (output, conv_state_new):
    - output: [B, channels_local, 1]  — conv output for this step
    - conv_state_new: [B, channels_local, K]  — updated state (to be written back)
    """
    B, channels_local, _ = x.shape

    # 1. Shift: extract the last K-1 slots
    shifted = ttnn.slice(conv_state, begins=[0, 0, 1], ends=[B, channels_local, K])

    # 2. Append new input
    x_reshaped = ttnn.reshape(x, [B, channels_local, 1])  # no-op if already [B, c, 1]
    conv_state_new = ttnn.concat([shifted, x_reshaped], dim=2)

    # 3. Depthwise weighted sum
    conv_weight_bcast = ttnn.reshape(conv_weight, [1, channels_local, K])
    output = ttnn.sum(ttnn.mul(conv_state_new, conv_weight_bcast), dim=2, keepdim=True)

    return output, conv_state_new
```

**Availability tags:**
- `ttnn.slice`: `[AVAILABLE]`
- `ttnn.concat`: `[AVAILABLE]`
- `ttnn.mul`: `[AVAILABLE]`
- `ttnn.sum`: `[AVAILABLE]`
- `ttnn.reshape`: `[AVAILABLE]`
- Full sequence: `[AVAILABLE — needs wiring]`

---

## 3. Memory Layout for `conv_state`

The `conv_state` tensor has shape `[B, channels_local, K] = [1, 1024, 4]`. The innermost dimension is `K=4`, which is not a multiple of the tile dimension 32.

Two layout options are available:

### Option A: `ROW_MAJOR` layout in DRAM

```python
conv_state_memory_config = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.DRAM,
)
# layout=ttnn.ROW_MAJOR_LAYOUT
```

- No padding required; stored as `1 × 1024 × 4 × 2 bytes = 8,192 bytes = 8 KB` per device per layer
- `ttnn.slice` and `ttnn.concat` on ROW_MAJOR tensors with non-tile-aligned dims are supported
- Slower per-element DMA (scalar reads rather than tile DMA) — but for 8 KB this is not a bottleneck

### Option B: `TILE` layout in DRAM with K-dimension padding

```python
# Pad K from 4 to 32 (nearest tile boundary):
# stored shape [1, 1024, 32]; only positions [0:4] are used
```

- Requires padding the state from K=4 to 32 columns: stored size = `1 × 1024 × 32 × 2 bytes = 65,536 bytes = 64 KB` per device per layer (8× overhead for K dimension)
- Enables tile-based DMA but adds complexity (slice must be index-aware of the padding)

**Recommendation:** Use ROW_MAJOR layout. The 8 KB size means DMA overhead is negligible, and ROW_MAJOR avoids the K-padding complexity. Consistent with the rule established in Chapter 2: tensors where either innermost dimension is not a multiple of 32 use ROW_MAJOR.

> **Note:** This is the same rule applied to `error_T` in Chapter 2 (`state_tensor_memory_config.md`): when the innermost dimension is not a multiple of 32, ROW_MAJOR DRAM is the correct layout choice.

---

## 4. Trace Compatibility Requirements

For full Metal Trace compatibility, every device buffer whose address is baked into the trace at capture time must be stable on every replay. This affects three buffers in the conv1d update sequence:

**`conv_state_persistent`** — the persistent DRAM state buffer. Must be pre-allocated before trace capture and updated via `ttnn.copy` rather than reassignment:

```python
# conv_state_persistent: pre-allocated DRAM tensor, address baked at trace capture
# At each decode step, write the updated state into the persistent buffer:
ttnn.copy(conv_state_new, conv_state_persistent)
# why: ttnn.copy is a DMA into an existing buffer — no new allocation, trace-safe
# see: trace_safe_cossin_prereplication/ch4_copy_trace_safety/what_copy_records.md
```

**`shifted` and `conv_state_new`** — the intermediate buffers from `ttnn.slice` and `ttnn.concat` inside the update function. Both allocate new device buffers on each call. For these to be trace-safe, they must also be pre-allocated in `__init__` and the ops must write into those pre-allocated buffers rather than returning freshly allocated tensors.

> **Note:** Whether `ttnn.slice` and `ttnn.concat` support writing into a caller-provided output buffer depends on the TTNN op implementation. If pre-allocated output buffers are not supported, these intermediates will be allocated at trace capture time and baked into the trace; subsequent replays will reuse the same addresses, which is acceptable as long as the sizes are fixed (which they are — `shifted` is always `[B, channels_local, K-1]` and `conv_state_new` is always `[B, channels_local, K]`). In that case, trace compatibility is preserved by the program cache's buffer-address stability, not by explicit pre-allocation.

The `conv_state_new` computed by `ttnn.concat` is an intermediate; `ttnn.copy` moves its contents into the fixed address of `conv_state_persistent`. This follows the same pattern as the pre-allocated cos/sin kwarg buffer described in the trace-safe cos/sin pre-replication guide.

---

## 5. Decode Path vs. Prefill Path

This file covers the **decode-time** update only (T=1, one new token per step). At decode time, the sliding-window update is a single shift + append + sum.

The **prefill-time** conv1d (`causal_conv1d_fn`) processes the full prompt sequence of length T and is covered in Chapter 6. The prefill operation is a standard depthwise convolution over T tokens; it is not on the critical path for decode trace compatibility.

---

## 6. Summary

| Property | Value |
|---|---|
| Channels per device | 1024 (= 8192 / 8 T3K devices) |
| Conv kernel size K | 4 |
| State shape per device | `[1, 1024, 4]` BF16 |
| State memory layout | ROW_MAJOR, DRAM |
| State size per device per layer | 8 KB |
| State size per device, 30 layers | 240 KB |
| TTNN ops required | `ttnn.slice`, `ttnn.concat`, `ttnn.mul`, `ttnn.sum`, `ttnn.reshape` |
| New kernel required? | No |
| Availability | `[AVAILABLE — needs wiring]` |

**Next:** [`gated_rmsnorm_ttnn.md`](./gated_rmsnorm_ttnn.md)
