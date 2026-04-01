# MoE Key Protection: Why Expert Tensors Must Be Extracted First

The `convert_hf_to_meta_qwen35` pipeline applies two general-purpose transforms to the state dict — `split_hf_keys` and `map_hf_to_meta_keys` — that were written for standard Llama-style architectures. Both transforms are safe for attention and DeltaNet weights. Neither is safe for Qwen3.5 MoE expert weights without modification. This document explains the two failure modes and the pop-protect-reinsert pattern that prevents them.

---

## The Two Failure Modes

### Failure Mode 1: `split_hf_keys` Corrupts 3D Expert Tensors

In a standard transformer, `gate_up_proj` refers to a fused 2D weight matrix of shape `[2 * intermediate, hidden]`. `split_hf_keys` identifies keys that contain `gate_proj` or `up_proj` (or their fused `gate_up_proj` form) and splits them along dimension 0, producing:

```
gate_up_proj [2 * intermediate, hidden]
    →  gate_proj [intermediate, hidden]
    +  up_proj   [intermediate, hidden]
```

In the A3B checkpoint, the key `mlp.experts.gate_up_proj` contains a **3D** packed tensor:

```
mlp.experts.gate_up_proj  shape: [256, 1024, 2048]
                                  ^    ^      ^
                              num_experts  2*intermediate  hidden
```

All 256 expert gate and up projections are packed into a single tensor along dimension 0 (the expert batch dimension). If `split_hf_keys` processes this key, it finds the `gate_up_proj` pattern and calls `torch.split(tensor, tensor.shape[0] // 2, dim=0)`. For this tensor, `tensor.shape[0] // 2 = 128`, so the split is on `dim=0` (the expert-batch axis), producing two tensors of `[128, 1024, 2048]`. This incorrectly splits what should be an indivisible 256-expert batch dimension in half, corrupting the expert layout irreversibly.

The same applies to `mlp.experts.down_proj`:

```
mlp.experts.down_proj  shape: [256, 2048, 512]
                               ^    ^      ^
                           num_experts  hidden  intermediate
```

`split_hf_keys` might not match `down_proj` directly, but any attempt to interpret the 3D tensor as a 2D projection would produce the wrong split.

### Failure Mode 2: `map_hf_to_meta_keys` Renames Inside Expert Paths

`map_hf_to_meta_keys` applies substring replacement rules across all key names. One standard rule renames `gate_proj` to `w1` (the meta-format name for the gate projection in a SwiGLU MLP). If applied to the expert keys, it would transform:

```
mlp.experts.gate_up_proj   →   mlp.experts.w1_up_proj   (partial rename, malformed)
```

or if the key were stored in split form after Step 1:

```
mlp.experts.gate_proj   →   mlp.experts.w1
```

The Qwen35MoE module's `__init__` method loads expert weights by looking up `experts.gate_up_proj` and `experts.down_proj` in the state dict after the `feed_forward.*` rename. A `w1`-renamed key would never be found, causing a `KeyError` at weight loading time or silently initializing the expert weights to zeros if the module falls back to a default.

This `gate_proj → w1` rename issue was a prior bug encountered during development: the initial version of the conversion pipeline did not protect MoE keys, and expert weights were either dropped (not found after rename) or loaded under wrong keys.

---

## The Protection Pattern

The solution is the pop-protect-reinsert pattern in `qwen35_utils.py`:

```python
def _is_moe_key(key):
    """Check if a key belongs to MoE-specific weights that need protection from transforms."""
    return any(pat in key for pat in ("mlp.experts", "mlp.gate.", "mlp.shared_expert"))

# Pop MoE keys before transforms to protect them
moe_keys = {k: v for k, v in state_dict.items() if _is_moe_key(k)}
if moe_keys:
    state_dict = {k: v for k, v in state_dict.items() if not _is_moe_key(k)}
```

The three patterns in `_is_moe_key` cover all weight categories that must not be transformed:

| Pattern | Matches | Why Protected |
|---------|---------|---------------|
| `mlp.experts` | `mlp.experts.gate_up_proj`, `mlp.experts.down_proj` | 3D tensors; would be corrupted by `split_hf_keys` |
| `mlp.gate.` | `mlp.gate.weight` (router) | Contains `gate` in path; could be renamed by `map_hf_to_meta_keys` |
| `mlp.shared_expert` | `mlp.shared_expert.gate_proj.weight`, `mlp.shared_expert.up_proj.weight`, `mlp.shared_expert.down_proj.weight`, `mlp.shared_expert_gate.weight` | Contains `gate_proj`; would be renamed to `w1` |

Note that `mlp.gate.` uses a trailing dot to match the router weight exactly (`mlp.gate.weight`) without also matching `mlp.gate_up_proj` (which begins `mlp.gate_up` not `mlp.gate.`). This specificity prevents false positives.

After Steps 1–3 run on the protected state dict, the MoE keys are re-inserted with only one transformation — the `mlp` to `feed_forward` prefix rename:

```python
for key, tensor in moe_keys.items():
    new_key = key.replace(".mlp.", ".feed_forward.")
    converted_weights[new_key] = tensor
```

This single rename aligns the MoE keys with the `feed_forward.*` namespace that `DeltaNetDecoderBlock` uses for the MLP component (regardless of whether MLP is a dense MLP or MoE). The rename is applied using `.mlp.` with surrounding dots to avoid partial matches — for instance, `mlp_intermediate` would not be affected.

---

## State Dict Before and After Conversion

For a single A3B MoE layer, the transformation looks like this:

**Before (`layer 0`, HF format):**

```
language_model.layers.0.mlp.experts.gate_up_proj          [256, 1024, 2048]
language_model.layers.0.mlp.experts.down_proj             [256, 2048, 512]
language_model.layers.0.mlp.gate.weight                   [256, 2048]
language_model.layers.0.mlp.shared_expert.gate_proj.weight [512, 2048]
language_model.layers.0.mlp.shared_expert.up_proj.weight   [512, 2048]
language_model.layers.0.mlp.shared_expert.down_proj.weight [2048, 512]
language_model.layers.0.mlp.shared_expert_gate.weight      [1, 2048]
```

**After (`layer 0`, meta format):**

```
language_model.layers.0.feed_forward.experts.gate_up_proj          [256, 1024, 2048]
language_model.layers.0.feed_forward.experts.down_proj             [256, 2048, 512]
language_model.layers.0.feed_forward.gate.weight                   [256, 2048]
language_model.layers.0.feed_forward.shared_expert.gate_proj.weight [512, 2048]
language_model.layers.0.feed_forward.shared_expert.up_proj.weight   [512, 2048]
language_model.layers.0.feed_forward.shared_expert.down_proj.weight [2048, 512]
language_model.layers.0.feed_forward.shared_expert_gate.weight      [1, 2048]
```

Every tensor shape is identical before and after. Only the path prefix changed from `.mlp.` to `.feed_forward.`. The attention and DeltaNet keys for the same layer undergo the full rename transformation (e.g., `self_attn.q_proj.weight` → `attention.wq`), but the expert keys are untouched by those transforms.

---

## Why Dense-Only Models Are Unaffected

For the 27B dense model, `_is_moe_key` matches nothing because there are no `mlp.experts`, `mlp.gate.`, or `mlp.shared_expert` keys in the checkpoint. The `if moe_keys:` guard means the state dict is not reconstructed, and `moe_keys` remains an empty dict. The re-insertion loop at the end is a no-op. The function behaves identically to a generic HF-to-meta conversion for the dense case.

---

**Next:** [Chapter 7 — Performance Analysis and Bottlenecks](../ch7_performance_analysis_and_bottlenecks/index.md)
