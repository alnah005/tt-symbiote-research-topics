# Qwen3.6-35B-A3B: M-RoPE Configuration

## 1. Complete M-RoPE Configuration

The following fields from `config.json` fully specify the M-RoPE behavior of Qwen3.6-35B-A3B:

```json
{
  "head_dim": 128,
  "partial_rotary_factor": 0.5,
  "rope_theta": 1000000.0,
  "rope_scaling": {
    "type": "mrope",
    "mrope_section": [11, 11, 10]
  }
}
```

**Field explanations:**

- `rope_theta`: The base frequency for the sinusoidal position encoding. The same value (1,000,000) is used when building the frequency table for all three coordinate sections (temporal, height, width). There is no per-section theta.

- `partial_rotary_factor`: 0.5 means only the first 64 of the 128 head dimensions receive RoPE rotation. The remaining 64 dimensions pass through unchanged. This is identical to how partial RoPE works in standard 1D RoPE.

- `rope_scaling.type`: The string `"mrope"` activates the multimodal three-section RoPE code path in HuggingFace's `Qwen2_5_VLRotaryEmbedding` class. Without this field, the model would use standard 1D RoPE regardless of the `mrope_section` value.

- `rope_scaling.mrope_section`: `[11, 11, 10]` — the three section widths, in units of rotation pairs, assigned to the temporal ($s_t$), height ($s_h$), and width ($s_w$) coordinates respectively.

---

## 2. Rotary Dimension Derivation

The rotary dimension follows directly from `head_dim` and `partial_rotary_factor`:

```math
\text{rotary dim} = \lfloor \text{head dim} \times \text{partial rotary factor} \rfloor = \lfloor 128 \times 0.5 \rfloor = 64
```

Consistency check — the section widths must sum to half the rotary dimension:

```math
s_t + s_h + s_w = 11 + 11 + 10 = 32 = \frac{\text{rotary dim}}{2} = \frac{64}{2} \checkmark
```

The 64 rotary dimensions break down as follows:

| Head dimensions | Coordinate | Pairs | Section width |
|---|---|---|---|
| `[0, 11)` and `[32, 43)` | Temporal | 0–10 | $s_t = 11$ |
| `[11, 22)` and `[43, 54)` | Height | 11–21 | $s_h = 11$ |
| `[22, 32)` and `[54, 64)` | Width | 22–31 | $s_w = 10$ |

---

## 3. How HuggingFace Resolves `rotary_dim`

When `rope_scaling.type = "mrope"` is present, HuggingFace's `Qwen2_5_VLRotaryEmbedding` class determines `rotary_dim` from the section sum:

```
rotary_dim = sum(mrope_section) * 2 = 32 * 2 = 64
```

The `partial_rotary_factor` at the top level is a consistency hint for tooling and documentation; the section sum is the authoritative source inside the model code.

The constraint `sum(mrope_section) == rotary_dim // 2` must hold. HuggingFace does not validate this at runtime — a mismatch silently produces wrong output.

> **Key Finding:** The canonical source of `rotary_dim` for an M-RoPE model is `2 × sum(rope_scaling.mrope_section)`, not `partial_rotary_factor`. A TTNN implementation must read `rope_scaling.mrope_section` from the config, not rely on `partial_rotary_factor` alone.

---

## 4. The Frequency Table

A single shared frequency table of shape `[max_seq_len, rotary_dim // 2]` is used for all three coordinate sections. M-RoPE does not require separate tables per section. The section partition only governs which columns of the table are indexed for which head-dimension range.

The table is constructed identically to standard partial RoPE:

```python
# For i in [0, rotary_dim // 2):
inv_freq[i] = 1.0 / (rope_theta ** (2 * i / rotary_dim))
# cos_table[t, i] = cos(t * inv_freq[i])
# sin_table[t, i] = sin(t * inv_freq[i])
```

For Qwen3.6: `rotary_dim = 64`, so the table has shape `[max_seq_len, 32]`. Each row `t` holds the cosine (or sine) of `t` times each of the 32 inverse frequencies.

---

**Next:** [`hf_reference_implementation.md`](./hf_reference_implementation.md)
