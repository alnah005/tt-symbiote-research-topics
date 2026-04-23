# Device State Persistence

This file explains how the DeltaNet recurrent state matrix S and the causal conv1d state (`conv_state`) are currently managed between decode steps, identifies why that management strategy is the root cause of the step 4 host crossing, and specifies exactly what must change so that both state tensors live on the Wormhole device permanently. By the end of this file the reader understands the state lifecycle problem in full and can articulate the prerequisite cache refactor that Chapter 7 (Task 1) implements before any kernel changes begin.

---

## Current State Storage: Plain PyTorch Tensors

The DeltaNet state is managed by a cache object — referred to here as `TTNNQwenPagedAttentionKVCache` — that is responsible for storing per-layer recurrent and convolutional state between decode calls. The relevant fields are:

```python
class TTNNQwenPagedAttentionKVCache:
    # Recurrent state S per DeltaNet layer.
    # Key: layer index (int). Value: PyTorch CPU tensor.
    recurrent_states: dict[int, torch.Tensor]   # shape [B, num_v_heads, d_k, d_v]

    # Causal conv1d state per DeltaNet layer.
    # Key: layer index (int). Value: PyTorch CPU tensor.
    conv_states: dict[int, torch.Tensor]         # shape [B, mixed_dim, K]
```

Both dictionaries hold plain `torch.Tensor` objects. The tensors reside on CPU (or on a CUDA device if a GPU is available to the Python process). They are not `ttnn.Tensor` objects and are not allocated on the Wormhole mesh device.

At the start of each decode step, `TTNNQwen3LinearAttention.forward` reads from these fields to supply the previous state to `recurrent_gated_delta_rule` and `causal_conv1d_update`. At the end of each decode step, it writes updated tensors back into the dictionaries. The tensors never touch the Wormhole device until they are explicitly transferred via `ttnn.from_torch` at the beginning of the next decode call — and that transfer itself is the host crossing.

> **Key insight:** The root cause of the step 4 host crossing is not the `recurrent_gated_delta_rule` kernel call per se. The root cause is that S is a CPU tensor. Even if `recurrent_gated_delta_rule` were replaced with a TTNN op tomorrow, the TTNN op would still need to receive S as a `ttnn.Tensor` that is already on the device. If S lives in `recurrent_states` as a CPU `torch.Tensor`, the forward pass must call `ttnn.from_torch(S_prev)` to place it on the device before the TTNN op can use it — and `ttnn.from_torch` breaks trace just as badly as `ttnn.to_torch`.

---

## Current State Lifecycle (Per Decode Step)

The lifecycle of S across two consecutive decode steps illustrates the problem:

```
Decode step t-1:
  [Wormhole device]  → ttnn.to_torch(S_prev) → [CPU]
  [CPU] recurrent_gated_delta_rule(S_prev, ...) → S_new [CPU]
  S_new stored in recurrent_states[layer_idx]   [CPU]

Decode step t:
  S_prev = recurrent_states[layer_idx]           [CPU]  ← retrieved from CPU dict
  [CPU] → ttnn.from_torch(S_prev) → [Wormhole device]   ← FROM_TORCH: breaks trace
  [Wormhole device] → ttnn.to_torch(S_prev_tt)  [CPU]  ← TO_TORCH: breaks trace (if on-device)
  [CPU] recurrent_gated_delta_rule(S_prev, ...) → S_new [CPU]
  S_new stored in recurrent_states[layer_idx]   [CPU]
```

The same pattern applies to `conv_state`. Every decode step involves at minimum one `ttnn.to_torch` and one `ttnn.from_torch` for each of S and `conv_state`, per DeltaNet layer (30 layers in the 35B-A3B model).

---

## What Must Change

S and `conv_state` must be allocated as `ttnn.Tensor` objects on the Wormhole mesh device. They must persist in DRAM between decode calls, never moving to host. State updates must be performed in-place via TTNN ops only. The updated cache fields look like:

```python
class TTNNQwenPagedAttentionKVCache:
    # Recurrent state S per DeltaNet layer — now on-device.
    recurrent_states: dict[int, ttnn.Tensor]    # ttnn.Tensor on Wormhole device, DRAM, TILE layout

    # Causal conv1d state per DeltaNet layer — now on-device.
    conv_states: dict[int, ttnn.Tensor]          # ttnn.Tensor on Wormhole device, DRAM, TILE layout
```

**Allocation during model setup (before the first decode step):**

```python
# Recurrent state S: initialized to zeros on-device.
S_shape = [batch_size, num_v_heads, d_k, d_v]   # [1, 32, 128, 128]
S_memory_config = ttnn.DRAM_MEMORY_CONFIG
S_layout = ttnn.TILE_LAYOUT
S_dtype = ttnn.bfloat16

recurrent_states[layer_idx] = ttnn.zeros(
    S_shape,
    dtype=S_dtype,
    layout=S_layout,
    device=mesh_device,
    memory_config=S_memory_config,
)

# Conv state: initialized to zeros on-device.
conv_shape = [batch_size, mixed_dim, K]          # [1, 8192, 4]
# Note: K=4 is below the tile minimum of 32; see tile alignment discussion below.
conv_states[layer_idx] = ttnn.zeros(
    conv_shape,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

**State update in-place during the decode step:**

Once S and `conv_state` are on-device, the state update in `TTNNQwen3LinearAttention.forward` must write the new state back into the pre-allocated DRAM buffer without creating a new allocation:

```python
# In-place update of S (pseudocode — exact TTNN API TBD in Chapter 2):
S_new = <result of the 6-op TTNN recurrence>
ttnn.copy(S_new, recurrent_states[layer_idx])    # overwrite DRAM buffer in-place
# or: recurrent_states[layer_idx] = S_new        # rebind the Python handle; DRAM buffer persists
```

> **Note:** Metal Trace requires that all device buffer addresses be fixed at capture time. Writing to a pre-allocated DRAM buffer (via `ttnn.copy` or `ttnn.assign`) is trace-compatible because the buffer address does not change between trace capture and replay. Allocating a new buffer (via `ttnn.from_torch` or `ttnn.empty`) inside the trace bracket is not trace-compatible. The in-place write is the correct pattern.

---

## State Tensor Shape and Memory Configuration

### Recurrent State S

Full state tensor (before T3K head-parallel sharding):

$$S \in \mathbb{R}^{[B, \text{num\_v\_heads}, d_k, d_v]} = [1, 32, 128, 128] \text{ BF16}$$

Size: $1 \times 32 \times 128 \times 128 \times 2 \text{ bytes} = 1{,}048{,}576 \text{ bytes} \approx 1 \text{ MB per layer}$

Under head-parallel sharding on T3K (8 devices, 4 heads per device):

$$S_{\text{per device}} \in \mathbb{R}^{[1, 4, 128, 128]} \text{ BF16}$$

Size: $1 \times 4 \times 128 \times 128 \times 2 \text{ bytes} = 131{,}072 \text{ bytes} = 128 \text{ KB per device per layer}$

Memory configuration: `(DRAM, TILE)` — tensor stored in off-chip GDDR6 DRAM, in tile layout (32×32 BF16 tiles).

Tile alignment check: $d_k = d_v = 128$; tile size = 32; $128 / 32 = 4$ tiles per dimension. S is exactly $4 \times 4$ tiles per head — no padding required. This is a favorable alignment case.

Total DRAM for S across all 30 DeltaNet layers at B=1 on one device: $30 \times 128 \text{ KB} = 3{,}840 \text{ KB} = 3.75 \text{ MB}$. This is negligible within the 12 GB DRAM budget per Wormhole chip.

### Conv State

Full conv state tensor (before sharding):

$$\text{conv\_state} \in \mathbb{R}^{[B, \text{mixed\_dim}, K]} = [1, 8192, 4] \text{ BF16}$$

Size: $1 \times 8192 \times 4 \times 2 \text{ bytes} = 65{,}536 \text{ bytes} = 64 \text{ KB per layer}$

Under T3K channel sharding (8 devices, mixed_dim/8 = 1024 channels per device):

$$\text{conv\_state}_{\text{per device}} \in \mathbb{R}^{[1, 1024, 4]} \text{ BF16}$$

Size: $1 \times 1024 \times 4 \times 2 \text{ bytes} = 8{,}192 \text{ bytes} = 8 \text{ KB per device per layer}$

**Tile alignment for K=4:** The K dimension (= 4) is below the minimum tile dimension of 32. A TILE layout tensor with shape [1, 1024, 4] must be padded to [1, 1024, 32] — the K dimension is padded with zeros to the next tile boundary. Only the last 4 columns of each row hold valid data; the remaining 28 columns are zero-padded.

Storage overhead from padding: the padded size is $1 \times 1024 \times 32 \times 2 = 65{,}536 \text{ bytes} = 64 \text{ KB}$ per device per layer, compared to 8 KB unpadded. This is an 8x overhead in storage, but 64 KB per layer × 30 layers = 1,920 KB = 1.875 MB total — still negligible.

An alternative is to keep `conv_state` in ROW_MAJOR layout to avoid padding. ROW_MAJOR allows non-tile-aligned sizes but is slower for DMA reads (scalar vs. tiled). For K=4 and 1024 channels, the unpadded size is 8 KB per device — a single DMA transaction either way. The TILE layout is preferred for consistency with the state matrix and because Chapter 3's TTNN decomposition uses tile-based ops (`ttnn.mul`, `ttnn.sum`) that expect TILE layout.

> **Note:** The exact handling of K-dimension padding in the TTNN conv1d decomposition (Chapter 3) must pad the weight tensor and the slice/concat ops to match the 32-column TILE layout. The implementation is straightforward: always index into the last K=4 valid columns of the padded tensor.

Total DRAM for conv_state across all 30 layers at B=1 on one device: $30 \times 64 \text{ KB} = 1{,}920 \text{ KB} = 1.875 \text{ MB}$. Negligible.

---

## This Change Is a Prerequisite, Not a Complete Fix

Migrating S and `conv_state` to on-device TTNN tensors is a necessary prerequisite for trace compatibility, but it is not sufficient on its own:

1. After the cache refactor, `recurrent_states[layer_idx]` is a `ttnn.Tensor` on the Wormhole device. But `TTNNQwen3LinearAttention.forward` still calls `recurrent_gated_delta_rule` (a host kernel). That call still requires a PyTorch CPU tensor as input, so the forward pass still calls `ttnn.to_torch(S_prev)` to convert the on-device tensor for the host kernel. The host crossing at step 4 is unchanged.

2. Similarly, `conv_states[layer_idx]` being on-device does not help if `causal_conv1d_update` still calls `ttnn.to_torch(conv_state)`. The conv1d host crossing at step 2 is unchanged.

The cache refactor eliminates the `FROM_TORCH` allocations that occur when the previous state is loaded from CPU into device memory at the start of a decode step. It does not eliminate the `TO_TORCH` calls that feed the host kernels during the decode step. Both fixes are required together: state-on-device (this file) plus kernel replacement (Chapters 2 and 3).

The correct implementation order is:

1. Refactor the cache object to store S and `conv_state` as on-device `ttnn.Tensor` objects (this file; Chapter 7 Task 1).
2. Replace `recurrent_gated_delta_rule` with the TTNN 6-op decomposition that reads S directly from the on-device tensor (Chapter 2; Chapter 7 Task 5).
3. Replace `causal_conv1d_update` with the TTNN slice+concat+mul+sum composition that reads `conv_state` directly from the on-device tensor (Chapter 3; Chapter 7 Task 3).

The cache refactor (step 1) can be implemented and tested independently before the kernel replacements (steps 2 and 3) are ready. A useful intermediate test: after the refactor, verify that `recurrent_states[layer_idx]` has type `ttnn.Tensor`, shape [1, 32, 128, 128], memory config DRAM, and that its values are zeros after initialization and nonzero after the first decode step (which still uses the host kernel via `ttnn.to_torch` / `ttnn.from_torch` during the transition period).

> **Key insight:** The state persistence change is a prerequisite for — and independent of — the kernel implementation changes in Chapters 2 and 3. It establishes the on-device tensor handles that the TTNN ops will reference, and it eliminates the `FROM_TORCH` allocations that occur when loading state from CPU at the start of each decode step. The kernel changes eliminate the remaining `TO_TORCH` and `HOST_KERNEL_LAUNCH` events during the decode step. Both sets of changes together achieve full trace compatibility.

For the complete memory configuration specification for S, including L1 feasibility analysis during fused kernel execution, see Chapter 2, `state_tensor_memory_config.md`.

---

**Next:** [Chapter 2 — TTNN Decomposition of the Recurrent Delta Rule Step](../ch2_ttnn_decomposition/index.md)

---

