# HuggingFace Config Resolution for `partial_rotary_factor`

## Section 1: How AutoConfig Populates Config Attributes

`AutoConfig.from_pretrained` reads `config.json` from the checkpoint directory, resolves the model-specific config class (e.g., `Qwen3_5MoeConfig`), and passes every top-level key/value pair to that class's `__init__` as keyword arguments via `**kwargs`. The model-specific `__init__` handles each attribute explicitly — top-level keys appear in `kwargs` and are assigned, while nested values (such as those inside `rope_scaling`) are only set if the `__init__` explicitly reads and promotes them.

For **Qwen3.6**, `config.json` contains `"partial_rotary_factor": 0.25` at the top level. After loading:

```python
config.partial_rotary_factor  # → 0.25
```

For **Qwen3.5**, `partial_rotary_factor` appears only inside the `rope_scaling` dict — there is no top-level key in `config.json`. After loading via `AutoConfig.from_pretrained`, `Qwen3_5MoeConfig.__init__` reads the value from `rope_scaling` and sets it explicitly, so both access paths work:

```python
config.rope_scaling["partial_rotary_factor"]  # → 0.25
config.partial_rotary_factor                  # → 0.25 (set by Qwen3_5MoeConfig.__init__)
```

`AttributeError` occurs only in code that bypasses `Qwen3_5MoeConfig.__init__` — for example, working with a raw `PretrainedConfig` object populated directly from JSON, where model-specific attribute assignments are not performed.

## Section 2: `rope_scaling` Sub-object Handling

The `rope_scaling` value is stored as `config.rope_scaling` — a plain Python `dict`, not a nested config object. `AutoConfig` does not auto-promote sub-fields of `rope_scaling` to top-level attributes; the dict is stored verbatim.

Accessing `partial_rotary_factor` from the nested location requires explicit dict access:

```python
config.rope_scaling.get("partial_rotary_factor")     # safe, returns None if absent
config.rope_scaling["partial_rotary_factor"]          # raises KeyError if absent
```

Summary of access paths by checkpoint:

| Access path | Qwen3.5 | Qwen3.6 |
|---|---|---|
| `config.partial_rotary_factor` | `0.25` (via `__init__`) | `0.25` |
| `config.rope_scaling["partial_rotary_factor"]` | `0.25` | `0.25` |
| `config.rope_scaling.get("partial_rotary_factor")` | `0.25` | `0.25` |

## Section 3: Which Value Wins (Precedence)

Both locations carry the same value (`0.25`), so there is no true precedence conflict. `Qwen3_5MoeConfig` — the HuggingFace config class used for both checkpoints — computes `rotary_dim` via:

```python
int(self.head_dim * self.partial_rotary_factor)
```

This uses the top-level attribute. `Qwen3_5MoeConfig.__init__` always sets `self.partial_rotary_factor` explicitly — by reading it from the `rope_scaling` dict if no top-level key is present — before this line runs, so no `AttributeError` can occur inside `__init__` itself. The `AttributeError` risk exists only in external consumer code that reads `config.partial_rotary_factor` as a raw attribute without going through `__init__` — because the **Qwen3.5** `config.json` does not contain `partial_rotary_factor` as a top-level key; without `Qwen3_5MoeConfig.__init__`'s explicit promotion from `rope_scaling`, the attribute is never set on the object.

The numeric result is the same either way:

```python
int(128 * 0.25) == 32  # True for both checkpoints
```

## Section 4: `transformers` Version and Resolution Logic

In `transformers >= 4.51` (the version current for **Qwen3.6** bring-up), `Qwen3_5MoeConfig.__init__` resolves `partial_rotary_factor` with logic equivalent to (simplified illustration):

```python
self.partial_rotary_factor = kwargs.get(
    "partial_rotary_factor",
    rope_scaling_value  # fallback: value read from rope_scaling dict
)
```

Top-level takes precedence, but falls back to the nested `rope_scaling` value if the top-level key is absent. This makes config loading robust for both checkpoints:

- **Qwen3.5** — no top-level key → fallback resolves to `0.25` from `rope_scaling`
- **Qwen3.6** — top-level key present → reads `0.25` directly

Any consumer code that skips `Qwen3_5MoeConfig.__init__` and reads `config.partial_rotary_factor` via raw attribute access will fail on **Qwen3.5** config JSON — because `partial_rotary_factor` is not a top-level key in Qwen3.5's `config.json`, so it is only present if `Qwen3_5MoeConfig.__init__` explicitly promoted it from `rope_scaling`.

> **Key Finding:** `partial_rotary_factor` is promoted to the top-level config in **Qwen3.6** as a defensive measure. The value (`0.25`) is identical in both locations. Consumer code that reads `config.partial_rotary_factor` on a raw `PretrainedConfig` object (bypassing `Qwen3_5MoeConfig.__init__`) will raise `AttributeError` for **Qwen3.5** config JSON — use the guard pattern shown in `ttnn_rope_impact.md`. The guard's `1.0` fallback is never reached for Qwen3.5 or Qwen3.6 checkpoints — both always have the value in `rope_scaling`.

---
**Next:** [`ttnn_rope_impact.md`](./ttnn_rope_impact.md)
