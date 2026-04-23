# Step-by-Step Failure Trace for rotary_dim=48, head_dim=128

This file walks through the exact sequence of operations executed by `TTNNRotaryPositionEmbedding.forward` for the `rotary_dim=48, head_dim=128` configuration. It identifies where each of the two compounded errors is introduced, distinguishes the two failure paths (TT_FATAL vs. silent numerical corruption), and uses a concrete element-level example to show precisely which output positions are corrupted in the silent-failure path.

---

## 1. The Forward Sequence in `TTNNRotaryPositionEmbedding`

The following steps describe what `TTNNRotaryPositionEmbedding.forward` does when called with `rotary_dim=48` and an input tensor of shape `[B, H, S, 128]`. Source: `rope.py` in the tt-symbiote codebase; if the exact lines cannot be verified at read time, annotations are marked `# TODO: verify`.

### Step 1 — cos/sin table sliced at the current position

During initialization, a cos/sin table is precomputed for all sequence positions. Its shape is:

```python
cos_table.shape  # [1, 1, max_seq_len, 48]  — shape[-1] == rotary_dim, correct
sin_table.shape  # [1, 1, max_seq_len, 48]
```

At forward time the table is sliced to the current sequence position `cur_pos`:

```python
cos = cos_table[:, :, cur_pos : cur_pos + seq_len, :]  # [1, 1, seq_len, 48]
sin = sin_table[:, :, cur_pos : cur_pos + seq_len, :]  # [1, 1, seq_len, 48]
```

This step is correct: the slice still has `shape[-1] == 48 == rotary_dim`.

### Step 2 — `ttnn.pad` extends cos/sin to `nearest_32(48) = 64`

Because `48 % 32 != 0`, `TTNNRotaryPositionEmbedding` pads the cos/sin tables to the next tile boundary before calling the device op:

```python
cos = ttnn.pad(cos, padding=((0, 0), (0, 0), (0, 0), (0, 16)), value=0.0)
# why: 48 + 16 = 64 = nearest_32(48); now tile-aligned in the last dimension
sin = ttnn.pad(sin, padding=((0, 0), (0, 0), (0, 0), (0, 16)), value=0.0)

# After padding:
cos.shape  # [1, 1, seq_len, 64]  — positions 48–63 are 0.0
sin.shape  # [1, 1, seq_len, 64]  — positions 48–63 are 0.0
```

> **Warning:** This is the first error. The op requires `cos.shape[-1] == head_dim == 128`. The padding target of 64 is incorrect. The intent was to reach tile alignment; the actual requirement is to reach `head_dim`.

### Step 3 — `ttnn.experimental.rotary_embedding` is called

```python
output = ttnn.experimental.rotary_embedding(
    input,       # shape [B, H, S, 128]
    cos,         # shape [1, 1, seq_len, 64]  — WRONG: should be [1, 1, seq_len, 128]
    sin,         # shape [1, 1, seq_len, 64]  — WRONG
    token_idx=cur_pos,
)
# TODO: verify exact call signature in rope.py
```

At this point `X = input.padded_shape()[-1] = 128` and `cos.padded_shape()[-1] = 64`. The C++ gate fires:

```
TT_FATAL: cos.padded_shape()[-1] == X  →  64 == 128  →  FATAL
```

---

## 2. Path A — TT_FATAL (Expected Behavior)

In the standard execution path, `RotaryEmbeddingOperation::invoke` evaluates the shape constraint before any kernel is dispatched:

```
invoke:  X = input.padded_shape()[-1] = 128
         TT_FATAL(cos.padded_shape()[-1] == X)
         → 64 == 128 is false → execution halts with:
           "cos_cache last dim must equal input last dim 128, but got shape [..., 64]"
```

The same constraint is re-checked in `RotaryEmbeddingDeviceOperation::validate`. Both fire independently. The call never reaches the compute kernel.

Path A produces a crash with a clear message. It is not a silent failure and it does not produce PCC ~0.71. The PCC ~0.71 observation belongs to Path B.

---

## 3. Path B — Autoformat Pads cos/sin Further to 128 (Silent Failure)

Path B describes a hypothetical execution path in which some layer of the stack — an alternate autoformat route, an explicit second `ttnn.pad` call, or a future change to the padding logic — extends cos/sin all the way from 64 to 128 with zeros before the device op runs. Under this path the shape constraint is satisfied (`128 == 128`) and the kernel executes without error.

### 3a. What the cos/sin tensors contain in Path B

After the two-stage zero-fill (Step 2 fills 48–63, the hypothetical second fill fills 64–127):

```
cos[0:48]   = real cosine values for rotation frequencies 0–47
cos[48:64]  = 0.0  (from Step 2 padding)
cos[64:128] = 0.0  (from the hypothetical second padding)
sin[0:48]   = real sine values for rotation frequencies 0–47
sin[48:128] = 0.0  (from both padding steps)
```

> **Note:** If autoformat's `pad_to_tile_shape` is the second padding source, it would add zeros because `cos.shape[-1]=64` is already tile-aligned; `pad_to_tile_shape` would not add further padding. The second pad must come from some other mechanism. Path B is documented here to explain the PCC ~0.71 observation. Regardless of the source of the second pad, the numerical analysis below applies to any configuration that reaches the kernel with cos/sin fully zero-padded to `[..., 128]`.

### 3b. The kernel's computation in Path B

The kernel derives:

```
Wt      = head_dim / TILE_WIDTH = 128 / 32 = 4 tiles
half_Wt = Wt / 2 = 2 tiles  →  64 elements
```

For each output element `i`, the kernel computes:

```
output[i]        = input[i]       * cos[i]       + input[i + 64] * (-sin[i])   for i in [0, 64)
output[i + 64]   = input[i]       * sin[i]       + input[i + 64] *   cos[i]    for i in [0, 64)
```

This is the standard full-head rotate-half formula with pairing offset `head_dim/2 = 64`.

### 3c. Element-level trace

The table below shows what the kernel computes for five representative positions, compared to the correct partial RoPE output. Notation: `c_k = cos[k]` (real value), `s_k = sin[k]` (real value), `0` denotes a zero from padding.

| Position | Kernel output (Path B) | Correct partial RoPE output | Status |
|---|---|---|---|
| `output[0]` | `input[0]*c_0 + input[64]*(-s_0)` | `input[0]*c_0 + input[24]*(-s_0)` | Wrong pairing: uses `input[64]` instead of `input[24]` |
| `output[24]` | `input[24]*c_24 + input[88]*(-s_24)` | `input[0]*s_0 + input[24]*c_0` | Wrong pairing, wrong frequency, wrong sign: uses `input[88]*(-s_24)` where correct term is `input[0]*s_0`; also uses `input[24]*c_24` where correct coefficient is `c_0` |
| `output[48]` | `input[48]*cos[48] + input[112]*(-sin[48])` = `input[48]*0 + input[112]*0` = `0` | `input[48]` (passthrough) | Zeroed: cos=sin=0 at position 48 |
| `output[64]` | `input[0]*s_0 + input[64]*c_0` | `input[64]` (passthrough) | Corrupted: mixes `input[0]` rotation into passthrough |
| `output[127]` | `input[63]*cos[63] + input[127]*(-sin[63])` = `input[63]*0 + input[127]*0 = 0` | `input[127]` (passthrough) | Zeroed: cos=sin=0 at position 63 |

\* The correct formula for `output[24]` in the rotate-half convention is `input[0]*s_0 + input[24]*c_0` where `j=0` pairs `input[0]` (left half) with `input[24]` (right half) at frequency index 0. See [`correct_partial_rope_reference.md`](./correct_partial_rope_reference.md) for the full derivation.

### 3d. How many elements are corrupted

Classifying the 128 output positions:

- **Positions `[0, 24)`:** Kernel uses real `c_i` and `s_i`, but pairs `input[i]` with `input[i+64]` instead of `input[i+24]`. The real cos/sin values are applied but to the wrong input pairing. Result: incorrect rotation.
- **Positions `[24, 48)`:** Kernel computes these as first-half rotate-half outputs using `output[i] = x[i]*c_i + x[i+64]*(-s_i)`. But positions `[24, 48)` are the **right half** of the partial RoPE rotation: the correct output is `output[j+24] = x[j]*s_j + x[j+24]*c_j` for `j = i-24 in [0, 24)`. The error is not merely a wrong pairing offset — the input elements, trig subscripts, and combination rule are all different from what the kernel applies.
- **Positions `[48, 64)`:** `cos[48:64] = 0` and `sin[48:64] = 0` (from Step 2 zero-pad). Kernel computes `input[j]*0 + input[j+64]*0 = 0` for `j in [48,64)`. These 16 elements of the output are zeroed. Correct output: passthrough (equal to `input[48:64]`).
- **Positions `[64, 128)`:** These are the "second half" outputs. The kernel computes `input[i]*s_i + input[i+64]*c_i` for `i in [0,64)`. For `i in [48,64)` both `s_i=0` and `c_i=0`, so `output[i+64] = 0`. For `i in [0,48)`, real sin/cos values are applied to input positions `[0,48)` and `[64,112)`, corrupting the passthrough elements `[64,112)` (incorrect linear combinations mixing rotation values into passthrough) and zeroing the passthrough elements `[112,128)` (since `c_i=s_i=0` for `i in [48,64)`).

Summary: all 128 output elements are either zeroed or incorrectly computed. The 80-element passthrough region `[48, 128)` is entirely corrupted; the 48-element rotation region `[0, 48)` is entirely corrupted: positions `[0, 24)` receive wrong-paired rotations (pairing offset 64 instead of 24), and positions `[24, 48)` receive structurally wrong outputs (kernel applies first-half formula to right-half positions, using wrong input elements, wrong frequency indices, and wrong combination rule). PCC ~0.71 is consistent with this degree of corruption across the output distribution.

> **[SILENT FAILURE]:** In Path B, no `TT_FATAL` fires. The op completes successfully. The output tensor has the correct dtype, shape, and memory layout. Only a numerical comparison against the PyTorch reference reveals the corruption. If no PCC check is in place, this failure passes silently through training or inference.

---

## 4. Why Path A Is the Expected Outcome

Given the analysis above:

- The padding target in `TTNNRotaryPositionEmbedding` is `nearest_32(rotary_dim) = 64`.
- The op's shape gate requires `cos.shape[-1] == head_dim == 128`.
- `64 != 128`, so the `TT_FATAL` fires before any kernel dispatch.
- `AutoFormat::pad_to_tile_shape` does not widen cos/sin from 64 to 128; it only pads to tile boundaries, and 64 is already tile-aligned.
- Therefore, the normal execution path is Path A: a crash, not silent corruption.

The PCC ~0.71 observation implies that somewhere in the test setup or codebase version being evaluated, cos/sin were padded all the way to 128 by some means (possibly a manual `ttnn.pad` call, a second padding step, or a different invocation path). Path B documents what that execution looks like numerically. If you are observing PCC ~0.71 without a crash, verify that cos/sin are not being explicitly padded to 128 elsewhere in the forward path before this op is called.

---

**Next:** [`correct_partial_rope_reference.md`](./correct_partial_rope_reference.md)
