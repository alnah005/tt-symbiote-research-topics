# Shape Validation in `RotaryEmbeddingOperation::invoke`

This file traces the shape constraints enforced before any compute kernel runs. The C++ layer imposes two hard gates — one in `invoke` and one in `validate` — that together ensure `cos_cache.shape[-1]` and `sin_cache.shape[-1]` are exactly equal to `input.shape[-1]` (i.e., `head_dim`). Understanding these gates is the prerequisite for understanding why padding cos/sin to `nearest_32(rotary_dim)` fails when `rotary_dim != head_dim`.

---

## 1. The `invoke` Function (`rotary_embedding.cpp`)

The entry point for the device operation is `RotaryEmbeddingOperation::invoke` in:

```
ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/rotary_embedding.cpp
```

### 1a. Input last-dimension constraint

The first check establishes `X`, the canonical width value that all subsequent checks reference:

```cpp
TT_FATAL(
    input_tensor.padded_shape()[-1] % (tt::constants::TILE_WIDTH * 2) == 0,
    "Input tensor last dimension must be divisible by {} (two tile widths), but got {}",
    tt::constants::TILE_WIDTH * 2,
    input_tensor.padded_shape()[-1]);

uint32_t X = input_tensor.padded_shape()[-1];
```

`tt::constants::TILE_WIDTH` is 32, so the divisor is **64**. This means `head_dim` must be a multiple of 64. For `head_dim=128` this passes trivially; for `head_dim=96` it also passes; for `head_dim=48` it fails.

`X` is set to `input_tensor.padded_shape()[-1]`, the **full head dimension after tile-padding** — not `rotary_dim`.

### 1b. cos/sin last-dimension constraint

Immediately after establishing `X`, the function checks both cache tensors:

```cpp
TT_FATAL(
    cos_cache.padded_shape()[-1] == X,
    "cos_cache last dim must equal input last dim {}, but got shape {}",
    X,
    cos_cache.padded_shape());

TT_FATAL(
    sin_cache.padded_shape()[-1] == X,
    "sin_cache last dim must equal input last dim {}, but got shape {}",
    X,
    sin_cache.padded_shape());
```

> **Note:** Only the `% 64 == 0` and `padded_shape()[-1]` checks have been verified from the source. Additional dimension checks, if present, are not covered here.

> **Key Finding:** `cos_cache.padded_shape()[-1]` must equal `X = input_tensor.padded_shape()[-1] = head_dim`. The `rotary_dim` argument is accepted by the Python API but it does **not** change what shape cos/sin must have. Passing cos/sin with `shape[-1] == rotary_dim` (e.g., 48) or `shape[-1] == nearest_32(rotary_dim)` (e.g., 64) when `head_dim=128` will cause this `TT_FATAL` to fire.

---

## 2. The `validate` Function (`rotary_embedding_device_operation.cpp`)

After `invoke` completes its checks and constructs the device operation, control passes to:

```
ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_device_operation.cpp
```

The `validate` method repeats the shape gate independently:

```cpp
void RotaryEmbeddingDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& cos           = input_tensors.at(1);
    const auto& sin           = input_tensors.at(2);

    uint32_t X = input_tensor.padded_shape()[-1];

    TT_FATAL(
        cos.padded_shape()[-1] == X,
        "cos last dim {} must equal input last dim {}",
        cos.padded_shape()[-1], X);

    TT_FATAL(
        sin.padded_shape()[-1] == X,
        "sin last dim {} must equal input last dim {}",
        sin.padded_shape()[-1], X);
}
```

This is the same constraint expressed again at the device-operation level. Even if `invoke` were patched to skip its checks, `validate` would still fire.

---

## 3. The Autoformat Path

When `ttnn.experimental.rotary_embedding` is called from Python through the standard dispatch path, it goes through `run_with_autoformat`. The autoformat layer calls `AutoFormat::pad_to_tile_shape` on each input tensor — including cos and sin — before invoking the device operation.

The sequence is:

```
Python: ttnn.experimental.rotary_embedding(input, cos_cache, sin_cache, rotary_dim=48)
  └─► run_with_autoformat(cos_cache, ...)
        └─► AutoFormat::pad_to_tile_shape(cos_cache)
              cos_cache.shape[-1] padded to nearest_32(cos_cache.shape[-1])
              e.g., nearest_32(48) = 64  →  cos.padded_shape()[-1] = 64
        └─► RotaryEmbeddingOperation::invoke(input, cos_padded, sin_padded, ...)
              X = input.padded_shape()[-1] = 128
              TT_FATAL: cos.padded_shape()[-1] == X  →  64 == 128  →  FIRES
```

> **Warning:** The autoformat path does not widen cos/sin to match `head_dim`. It only pads each tensor to its own tile boundary. If the user supplies `cos_cache` with `shape[-1]=48`, autoformat produces `padded_shape()[-1]=64`, which still fails the `== X` check when `head_dim=128`. The caller is responsible for supplying cos/sin with `shape[-1] == head_dim` before calling the op.

---

## 4. Concrete Failure Scenario

To make the failure concrete, consider the following configuration:

| Parameter | Value |
|-----------|-------|
| `head_dim` | 128 |
| `rotary_dim` | 48 |
| `nearest_32(rotary_dim)` | 64 |

A caller that builds cos/sin by slicing a precomputed cache to `rotary_dim=48` and then pads to `nearest_32(48)=64` will produce:

```python
cos_cache.shape  # [1, 1, seq_len, 64]
```

When passed to `ttnn.experimental.rotary_embedding`:

- `X = input.padded_shape()[-1] = 128`
- `TT_FATAL: cos_cache.padded_shape()[-1] == X` evaluates as `64 == 128` — **fires**

The correct approach is to supply cos/sin with `shape[-1] == 128`. How to construct such tensors for partial RoPE (where only 48 positions carry meaningful values) is the subject of Chapter 4's Strategy C (identity-filled cos/sin).

> **Note:** The `rotary_dim` parameter accepted by the Python API is passed through to the kernel as a tile count (`rotary_dim / TILE_WIDTH`). It is used inside the kernel to bound which tiles receive rotation versus passthrough. However, this only works correctly when `rotary_dim` is tile-aligned. The shape of cos/sin must still be `head_dim` wide regardless of `rotary_dim`.

---

**Next:** [`kernel_rotate_half_pairing.md`](./kernel_rotate_half_pairing.md)
