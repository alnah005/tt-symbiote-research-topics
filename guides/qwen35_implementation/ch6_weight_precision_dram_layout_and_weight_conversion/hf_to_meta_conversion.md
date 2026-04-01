# HuggingFace to Meta Weight Conversion

Qwen3.5 checkpoints are distributed in HuggingFace safetensors format with key names like `model.language_model.layers.0.self_attn.q_proj.weight`. The TTNN modules expect a different key namespace: `layers.0.attention.wq` (after stripping the `model.language_model.` prefix). They also expect the `q_proj` weight to be split into separate query and gate tensors, and MoE expert weights to live under `feed_forward.*` rather than `mlp.*`.

All of this is handled by `convert_hf_to_meta_qwen35` in `models/tt_transformers/tt/qwen35_utils.py`.

---

## Entry Point Signature

```python
def convert_hf_to_meta_qwen35(state_dict, head_dim, n_heads, n_kv_heads):
```

The caller passes the raw `state_dict` with the `model.language_model.` prefix already stripped (the stripping happens in the test harness and in `ModelArgs.load_state_dict`). The three integer parameters describe the full-attention geometry and are needed only for the `q_proj` gate split computation.

---

## The Five-Step Pipeline

### Step 0 — MoE Key Extraction (Pre-transform Protection)

Before any transforms run, all MoE-specific keys are popped out of `state_dict` and held aside:

```python
def _is_moe_key(key):
    """Check if a key belongs to MoE-specific weights that need protection from transforms."""
    return any(pat in key for pat in ("mlp.experts", "mlp.gate.", "mlp.shared_expert"))

moe_keys = {k: v for k, v in state_dict.items() if _is_moe_key(k)}
if moe_keys:
    state_dict = {k: v for k, v in state_dict.items() if not _is_moe_key(k)}
```

This step is listed here as "Step 0" because it is a prerequisite for Steps 1–3 rather than a transform in its own right. The reason expert keys must be protected from the subsequent steps is covered in detail in [`moe_key_protection.md`](./moe_key_protection.md).

---

### Step 1 — `split_hf_keys`: Splitting Fused Projections

```python
state_dict = split_hf_keys(state_dict, n_heads, n_kv_heads)
```

`split_hf_keys` (defined in `models/tt_transformers/tt/load_checkpoints.py`) handles HuggingFace models that ship fused weight matrices. For example, some models store the Q, K, and V projections as a single `qkv_proj` weight, or store the gate and up projections as `gate_up_proj`. `split_hf_keys` identifies these patterns by key name and splits the tensors along the output dimension into their component matrices.

For Qwen3.5, the full-attention layers store `q_proj`, `k_proj`, and `v_proj` as separate weights already, so `split_hf_keys` effectively passes them through. The important effect here is that `split_hf_keys` does **not** touch `linear_attn` keys — they contain no pattern it recognizes — so DeltaNet weights pass through this step unchanged.

---

### Step 2 — `q_proj` Gate Extraction (Full-Attention Layers)

Qwen3.5's GatedAttention uses an output gate applied to the attention result. The gate is produced by projecting the layer input through a dedicated weight matrix. In the HuggingFace checkpoint this gate weight is **packed inside `q_proj`**: the `q_proj` weight has twice the normal output dimension, with query and gate interleaved on a per-head basis.

**Layout of `q_proj.weight` in the HF checkpoint:**

```
shape: [n_heads * head_dim * 2, hidden_size]
        e.g. [24 * 256 * 2, 5120] = [12288, 5120]  for 27B
             [16 * 128 * 2, 2048] = [4096,  2048]  for A3B
```

The interleaving is per-head. For head $h$, rows $[h \cdot 2 \cdot \text{head dim},\; h \cdot 2 \cdot \text{head dim} + \text{head dim})$ are the query rows, and rows $[h \cdot 2 \cdot \text{head dim} + \text{head dim},\; (h+1) \cdot 2 \cdot \text{head dim})$ are the gate rows.

Schematically:

$$W_{q\_proj} = \begin{bmatrix} Q_{h=0} \\ G_{h=0} \\ Q_{h=1} \\ G_{h=1} \\ \vdots \\ Q_{h=n\_heads-1} \\ G_{h=n\_heads-1} \end{bmatrix}$$

where each $Q_h$ and $G_h$ block has `head_dim` rows.

**The split code:**

```python
elif "q_proj.weight" in key:
    q_size = n_heads * head_dim
    if tensor.shape[0] == q_size * 2:
        # Weight layout is interleaved per-head: [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
        # Split per-head into query and gate (NOT first/second half!)
        w = tensor.reshape(n_heads, head_dim * 2, -1)
        converted_weights[key] = w[:, :head_dim, :].reshape(q_size, -1).contiguous()
        gate_key = key.replace("q_proj.weight", "q_proj_gate.weight")
        converted_weights[gate_key] = w[:, head_dim:, :].reshape(q_size, -1).contiguous()
    else:
        converted_weights[key] = tensor
```

The reshape to `[n_heads, head_dim * 2, -1]` groups the rows by head. Slicing `[:, :head_dim, :]` picks the query half of each head; slicing `[:, head_dim:, :]` picks the gate half. Both halves are reshaped back to `[q_size, hidden_size]` = `[n_heads * head_dim, hidden_size]`.

The same interleaved split is applied to `q_proj.bias` when present:

```python
elif "q_proj.bias" in key:
    q_size = n_heads * head_dim
    if tensor.shape[0] == q_size * 2:
        # Same interleaved per-head split as weights
        b = tensor.reshape(n_heads, head_dim * 2)
        converted_weights[key] = b[:, :head_dim].reshape(-1).contiguous()
        gate_key = key.replace("q_proj.bias", "q_proj_gate.bias")
        converted_weights[gate_key] = b[:, head_dim:].reshape(-1).contiguous()
```

#### Why There Is No `reverse_permute`

Many Llama-style models require a `reverse_permute` transform on Q and K weights before loading into TTNN. The Llama HuggingFace checkpoint stores rotary frequencies in an interleaved complex-number format (pairs of $(r, i)$ at positions $0, 2, 4, \ldots$), whereas the TTNN RoPE kernel expects them in a split-half format (all real parts in $[0, \text{rotary dim}/2)$, all imaginary parts in $[\text{rotary dim}/2, \text{rotary dim})$). `reverse_permute` applies a permutation to undo this rearrangement before loading.

Qwen3.5 uses HuggingFace-style RoPE with a `partial_rotary_factor` parameter. The frequencies are computed as:

$$\theta_i = \frac{1}{\text{rope theta}^{2i / \text{rotary dim}}}, \quad i = 0, 1, \ldots, \frac{\text{rotary dim}}{2} - 1$$

where $\text{rotary dim} = \text{head dim} \times \text{partial rotary factor}$ (e.g., $256 \times 0.25 = 64$ for 27B). The weights are already stored in the format the HF-style RoPE kernel expects. No permutation transform is needed, and the code explicitly omits it:

```python
# NO reverse_permute: Qwen3.5 uses HF-style RoPE with partial_rotary_factor,
# so weights are already in the correct format.
```

K projection weights and the per-head RMSNorm weights (`q_norm.weight`, `k_norm.weight`) pass through Step 2 entirely unchanged:

```python
elif "k_proj.weight" in key:
    converted_weights[key] = tensor
elif "q_norm.weight" in key:
    converted_weights[key] = tensor  # No permute for HF-style RoPE
elif "k_norm.weight" in key:
    converted_weights[key] = tensor
```

DeltaNet keys (`"linear_attn" in key`) bypass Step 2 entirely:

```python
if "linear_attn" in key:
    # DeltaNet layer weights: pass through unchanged
    converted_weights[key] = tensor
```

---

### Step 3 — `map_hf_to_meta_keys`: Namespace Remapping

```python
converted_weights = map_hf_to_meta_keys(converted_weights)
```

`map_hf_to_meta_keys` applies a set of string replacement rules to convert HuggingFace key names to the meta-format names the TTNN modules use. The standard replacements include:

- `self_attn` → `attention`
- `q_proj` → `wq`
- `k_proj` → `wk`
- `v_proj` → `wv`
- `o_proj` → `wo`

Because DeltaNet keys use the prefix `linear_attn` (not `self_attn`) and contain names like `in_proj_qkv`, `in_proj_z`, `out_proj` (not `q_proj`, `k_proj`, etc.), none of the replacement patterns match DeltaNet keys. They pass through `map_hf_to_meta_keys` unchanged and arrive at the GatedDeltaNet module with their original HuggingFace names.

After this step, the extracted gate weight key (from Step 2) follows the same rename path as `q_proj`:

```
q_proj_gate.weight  →  wq_gate.weight   (inside attention.*)
```

This becomes `attention.wq_gate` in the final state dict, which is the key GatedAttention uses to load the gate projection.

---

### Step 4 — MoE Key Re-insertion

The MoE keys that were extracted in Step 0 are now re-inserted with a single rename applied:

```python
for key, tensor in moe_keys.items():
    new_key = key.replace(".mlp.", ".feed_forward.")
    converted_weights[new_key] = tensor
```

The only transform applied is renaming `.mlp.` to `.feed_forward.` in the key path, which aligns with the `feed_forward.*` namespace the Qwen35MoE module uses. No other renaming, splitting, or permutation is applied. The expert weight tensors retain their original 3D shapes:

```
feed_forward.experts.gate_up_proj  shape: [256, 1024, 2048]
feed_forward.experts.down_proj     shape: [256, 2048, 512]
feed_forward.shared_expert.gate_proj.weight   (standard 2D)
feed_forward.shared_expert.up_proj.weight
feed_forward.shared_expert.down_proj.weight
feed_forward.gate.weight
feed_forward.shared_expert_gate.weight
```

---

## Full Pipeline Summary

| Step | Function | What It Does | DeltaNet Keys | MoE Keys |
|------|----------|-------------|---------------|----------|
| 0 | `_is_moe_key` + pop | Extracts MoE keys before any transforms | Not affected | Popped into `moe_keys` |
| 1 | `split_hf_keys` | Splits fused projections (e.g., `gate_up_proj`) | Pass through (no match) | Not present (already popped) |
| 2 | Manual loop | Splits `q_proj` gate; skips `reverse_permute` | Pass through (`linear_attn` check) | Not present |
| 3 | `map_hf_to_meta_keys` | Renames `self_attn→attention`, `q_proj→wq`, etc. | Pass through (no match) | Not present |
| 4 | Re-insert loop | Applies only `.mlp.→.feed_forward.` rename | Not affected | Re-inserted with feed_forward prefix |

---

## Usage in Tests

The PCC test files demonstrate how the pipeline is invoked for a single layer. From `tests/test_pcc.py`:

```python
sd = {k.replace("model.", ""): v for k, v in layer_weights.items()}
sd = convert_hf_to_meta_qwen35(sd, HEAD_DIM, N_HEADS, N_KV_HEADS)
```

The `model.` prefix is stripped from the HF keys (the safetensors index uses `model.language_model.layers.N.*` but `model.language_model.` is retained as `language_model.` — the exact stripping depends on whether the caller uses the full path or a layer-relative prefix; in the test, `model.` is stripped leaving `language_model.layers.N.*` which the function handles through the standard `map_hf_to_meta_keys` chain).

---

**Next:** [`moe_key_protection.md`](./moe_key_protection.md)
