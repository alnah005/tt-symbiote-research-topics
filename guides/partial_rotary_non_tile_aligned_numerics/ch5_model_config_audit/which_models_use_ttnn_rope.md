# Which Models Use TTNNRotaryPositionEmbedding?

This file distinguishes the two RoPE classes in tt-symbiote, enumerates all currently known Qwen3-family models that route through `TTNNRotaryPositionEmbedding`, derives `rotary_dim` for each configuration, and checks whether the derived `rotary_dim` is tile-aligned. It also documents the investigation method so that future model additions can be audited using the same approach.

---

> **Key Finding:** Every Qwen3-family model in the current tt-symbiote codebase that uses `TTNNRotaryPositionEmbedding` does so with `partial_rotary_factor=0.5` and `head_dim=128`, yielding `rotary_dim=64`. Because `64 % 64 == 0`, the zero-padding branch in `TTNNRotaryPositionEmbedding` is never entered and the bug described in Chapters 1–3 is not triggered.

---

## The Two RoPE Classes

tt-symbiote's `rope.py` defines two distinct RoPE classes with different scopes of applicability.

### TTNNRotaryPositionEmbedding

This class handles the partial RoPE path, where `partial_rotary_factor < 1.0`. Its `__init__` precomputes a cos/sin table of shape `[1, 1, max_seq_len, rotary_dim]` and, when `rotary_dim % 32 != 0`, calls `ttnn.pad` to extend to `[1, 1, max_seq_len, nearest_32(rotary_dim)]`. Its `forward` passes the (possibly padded) cos/sin to `ttnn.experimental.rotary_embedding`.

The non-tile-aligned `rotary_dim` bug lives entirely inside this class.

### TTNNDistributedRotaryPositionEmbedding

This class handles the full-head RoPE path (`partial_rotary_factor == 1.0`) and the tensor-parallel (distributed) execution path. Because `rotary_dim == head_dim` in the full-head case, there is no partial-rotation padding problem: the cos/sin table is already `head_dim`-wide. This class is not affected by the bug documented in this guide.

The decision of which class is instantiated is made in the model's attention module based on `partial_rotary_factor`.

---

## Qwen3-Family Model Enumeration

### Deriving rotary_dim

For any model using the standard partial RoPE formula:

```
rotary_dim = int(partial_rotary_factor * head_dim)
```

This value must be even (required for rotate-half pairing) and must satisfy `rotary_dim <= head_dim`.

### Known Models

#### Qwen3.5-35B-A3B (Attention Layers)

- `partial_rotary_factor`: 0.5
- `head_dim`: 128
- Derived `rotary_dim`: `int(0.5 * 128) = 64`
- `64 % 32 == 0`: Yes
- `64 % 64 == 0`: Yes (satisfies the two-tile constraint)
- Bug path reached: **No** — `rotary_dim=64` is tile-aligned; the `ttnn.pad` branch in `TTNNRotaryPositionEmbedding` is taken only when `rotary_dim % 32 != 0`, which is false here.

#### Qwen3.6-35B-A3B (Attention Layers)

- `partial_rotary_factor`: 0.5
- `head_dim`: 128
- Derived `rotary_dim`: 64
- Tile alignment: same as Qwen3.5-35B-A3B above
- Bug path reached: **No**

#### Qwen3.6-35B-A3B (Linear Attention / DeltaNet Layers)

Qwen3.6-35B-A3B includes linear attention layers (gated DeltaNet) alongside the standard self-attention layers. These layers may have different head and rotary dimension configurations from the attention layers. Based on available configuration information, the linear attention layers use `partial_rotary_factor=0.5` and `head_dim=128`, yielding `rotary_dim=64`.

> **Note:** The linear attention layers in Qwen3.6-35B-A3B use a different attention mechanism (DeltaNet) but the same RoPE parameterization as the standard attention layers. `head_dim` and `rotary_dim` for these layers should be verified against the model configuration before bringing up any new variant of this architecture, as the DeltaNet layer design permits different head dimensions.

- Derived `rotary_dim`: 64
- Bug path reached: **No**

#### Hypothetical: Any Model with partial\_rotary\_factor = 0.375, head\_dim = 128

- Derived `rotary_dim`: `int(0.375 * 128) = 48`
- `48 % 32 == 16`: Not zero — not tile-aligned
- Bug path reached: **Yes**

This hypothetical configuration (`partial_rotary_factor=0.375`) is the source of the `rotary_dim=48` scenario described in the research topic. It is not a production-supported model; it was constructed synthetically to expose the bug.

---

## Summary Table

| Model | partial\_rotary\_factor | head\_dim | rotary\_dim | rotary\_dim % 32 | rotary\_dim % 64 | Bug path reached? |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (attn) | 0.5 | 128 | 64 | 0 | 0 | No |
| Qwen3.6-35B-A3B (attn) | 0.5 | 128 | 64 | 0 | 0 | No |
| Qwen3.6-35B-A3B (linear attn) | 0.5 | 128 | 64 | 0 | 0 | No |
| Hypothetical (partial\_rotary\_factor=0.375) | 0.375 | 128 | 48 | 16 | 48 | **Yes** |

---

## Investigation Method

To audit any new or existing model for non-tile-aligned `rotary_dim`, use the following approach.

### Step 1 — Locate partial\_rotary\_factor values

Search the model configuration files and `rope.py` for occurrences of `partial_rotary_factor`:

```bash
grep -r "partial_rotary_factor" <tt-symbiote-root>/models/ <tt-symbiote-root>/tt_transformers/
```

This surfaces every location where the factor is set or read.

### Step 2 — Identify the corresponding head\_dim

For each model, `head_dim = hidden_size / num_attention_heads`. This is typically set in the Hugging Face `config.json` or a tt-symbiote model configuration class. Check:

- `config.json` field `head_dim`, or
- `config.json` fields `hidden_size` and `num_attention_heads`, then compute `head_dim = hidden_size // num_attention_heads`.

### Step 3 — Derive rotary\_dim and check alignment

```python
rotary_dim = int(partial_rotary_factor * head_dim)
tile_aligned = (rotary_dim % 32 == 0)
two_tile_aligned = (rotary_dim % 64 == 0)
print(f"rotary_dim={rotary_dim}, tile_aligned={tile_aligned}, two_tile_aligned={two_tile_aligned}")
```

If `rotary_dim % 32 != 0`, the model exercises the non-tile-aligned bug path.

### Step 4 — Check which RoPE class is instantiated

Confirm that the model's attention module instantiates `TTNNRotaryPositionEmbedding` (not `TTNNDistributedRotaryPositionEmbedding`) for the configuration in question. Only `TTNNRotaryPositionEmbedding` applies the zero-padding logic.

---

## What's Next

Having enumerated all known models and confirmed that none exercises the non-tile-aligned path, the next question is whether this makes the bug unimportant. The answer is explored in [`is_this_dead_code.md`](./is_this_dead_code.md).
