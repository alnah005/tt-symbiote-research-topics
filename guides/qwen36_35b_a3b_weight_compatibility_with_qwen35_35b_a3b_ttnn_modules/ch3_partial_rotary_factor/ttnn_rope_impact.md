# TTNN RoPE Impact of `partial_rotary_factor` Promotion

## Section 1: How `TTNNRotaryPositionEmbedding` Reads Config

`TTNNRotaryPositionEmbedding.__init__` receives the model config (or a reduced `model_args` struct derived from it) and computes the rotary dimension:

```python
rotary_dim = int(head_dim * partial_rotary_factor)
```

If this line reads `config.partial_rotary_factor` via bare attribute access on a config object not initialized via `Qwen3_5MoeConfig.__init__` (e.g., a raw `PretrainedConfig` object or a custom `model_args` struct), it will fail for **Qwen3.5** config JSON — either raising `AttributeError` explicitly, or silently producing a wrong value if the exception is swallowed and a default substituted.

The safe access pattern that works for both checkpoints:

```python
prf = (
    getattr(config, "partial_rotary_factor", None)
    or (getattr(config, "rope_scaling", None) or {}).get("partial_rotary_factor", 1.0)
)
rotary_dim = int(config.head_dim * prf)
```

## Section 2: Computed `rotary_dim` for Both Checkpoints

| | Qwen3.5-35B-A3B | Qwen3.6-35B-A3B |
|---|---|---|
| `partial_rotary_factor` | `0.25` (JSON: `rope_scaling` only; loaded object: also top-level via `__init__`) | `0.25` (top-level AND inside `rope_scaling`) |
| `head_dim` | `128` | `128` |
| `rotary_dim = int(head_dim × factor)` | `int(128 × 0.25) = 32` | `int(128 × 0.25) = 32` |
| Cos/sin table shape | `[max_seq_len, 32]` | `[max_seq_len, 32]` |
| Rotated head dims | First 32 of 128 | First 32 of 128 |
| Pass-through head dims | Last 96 of 128 | Last 96 of 128 |

## Section 3: No TTNN Code Change Required

The cos/sin table shape is `[max_seq_len, 32]` for both checkpoints — identical. The partial RoPE application (rotate the first 32 dimensions of each head, pass through the last 96) is identical. No TTNN op configuration changes, no re-sharding, no dtype changes are needed.

`TTNNRotaryPositionEmbedding` constructed with a **Qwen3.6** config produces an embedding module identical in every respect to one constructed with a **Qwen3.5** config, provided the attribute-access guard is in place. The promotion of `partial_rotary_factor` to the top level is invisible to the TTNN layer at runtime.

## Section 4: The AttributeError Risk

Bare attribute access — `config.partial_rotary_factor` — raises `AttributeError` on **Qwen3.5** config objects that bypass `Qwen3_5MoeConfig.__init__` (see Section 1). The failure occurs during module construction, before any inference runs — it is loud, immediate, and easy to diagnose. However, if the exception is caught and a non-model-correct default substituted (e.g., `1.0` → `rotary_dim = 128` instead of 32), a silent numerical error can result — apply the guard in Section 1 to prevent this.

**Qwen3.6**'s top-level promotion eliminates this failure mode for downstream code that reads the top-level attribute without a guard. The guard (shown in Section 1) remains the correct long-term pattern for any code that must handle both checkpoints.

## Section 5: Verification

```python
from transformers import AutoConfig

config_35 = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B")
config_36 = AutoConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B")

# Safe access pattern — works for both checkpoints
def get_partial_rotary_factor(config):
    return (
        getattr(config, "partial_rotary_factor", None)
        or (getattr(config, "rope_scaling", None) or {}).get("partial_rotary_factor", 1.0)
    )

prf_35 = get_partial_rotary_factor(config_35)
prf_36 = get_partial_rotary_factor(config_36)
assert prf_35 == prf_36 == 0.25

rotary_dim_35 = int(config_35.head_dim * prf_35)
rotary_dim_36 = int(config_36.head_dim * prf_36)
assert rotary_dim_35 == rotary_dim_36 == 32  # identical for both checkpoints
```

> **Key Finding:** The `partial_rotary_factor` promotion does not change `rotary_dim` (it remains 32 for both **Qwen3.5** and **Qwen3.6**). The cos/sin table shapes and all TTNN RoPE module configurations are identical. The guard is the correct defensive pattern for any TT-Symbiote code that reads `config.partial_rotary_factor` from raw or non-standard config objects — without it, such code will raise `AttributeError` on **Qwen3.5** config objects that bypass `Qwen3_5MoeConfig.__init__`.

---
**Next:** [Chapter 4 — `bos_token_id` and Generation Loop Initialization](../ch4_bos_token_id/index.md)
