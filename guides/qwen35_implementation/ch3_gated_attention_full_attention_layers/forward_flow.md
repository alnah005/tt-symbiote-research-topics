# Forward Pass Through GatedAttention

## Overview

`GatedAttention.forward` is deliberately thin. It handles one Qwen3.5-specific concern — ensuring the gate input is saved before the parent consumes it — and then fully delegates to the standard `Attention.forward`. The gate itself fires inside the parent via the `pre_wo_hook` mechanism described in `output_gate.md`.

The complete call sequence for one decode step is:

```
GatedAttention.forward(x, current_pos, ...)
  1. _gate_input = ttnn.add(x, 0, DRAM)   ← save copy of x
  2. super().forward(x, ...)               ← delegate everything else
       a. QKV projections
       b. Q/K per-head RMSNorm
       c. RoPE (corrected partial rotation via patched cos/sin)
       d. KV cache update
       e. GQA expansion
       f. Scaled dot-product attention + softmax
       g. pre_wo_hook(attn_output)          ← _apply_gate fires here
            - gate = sigmoid(x_saved @ W_gate)
            - attn_output = attn_output * gate
       h. WO projection
  3. return output
```

## `GatedAttention.forward` Source

```python
def forward(
    self,
    x,
    current_pos,
    rot_mats=None,
    user_id=0,
    mode=Mode.DECODE,
    page_table=None,
    chunk_page_table=None,
    chunk_start_idx=None,
    kv_cache=None,
):
    # Save a copy of input for gate computation (the hook reads it later).
    # Attention.forward_decode deallocates x after the QKV matmul, so we
    # must ensure _gate_input is a separate buffer. ttnn.add(x, 0) always
    # creates a new tensor, unlike to_memory_config which may alias when
    # source and destination configs match.
    if self.gate_weight is not None:
        self._gate_input = ttnn.add(x, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Delegate to standard Attention forward
    return super().forward(
        x,
        current_pos,
        rot_mats=rot_mats,
        user_id=user_id,
        mode=mode,
        page_table=page_table,
        chunk_page_table=chunk_page_table,
        chunk_start_idx=chunk_start_idx,
        kv_cache=kv_cache,
    )
```

Every argument is passed through unmodified. `GatedAttention` does not inspect `rot_mats`, `page_table`, or `kv_cache` — those are entirely managed by the base class.

## Step-by-Step: What Happens Inside `Attention.forward`

### QKV Projections

The base class runs three linear projections:
- `q_proj`: input `[1, 1, B, hidden]` → `[1, 1, B, n_heads * head_dim]`
- `k_proj`: input `[1, 1, B, hidden]` → `[1, 1, B, n_kv_heads * head_dim]`
- `v_proj`: input `[1, 1, B, hidden]` → `[1, 1, B, n_kv_heads * head_dim]`

For the 27B model: `n_heads=24`, `n_kv_heads=4`, `head_dim=256`.
For the 35B-A3B model: `n_heads=16`, `n_kv_heads=2`, `head_dim=256`.

Note that `q_proj.weight` here is the **query-only** weight after the `q_proj_gate` split — half the size of the raw HF checkpoint weight.

### Per-Head Q/K RMSNorm

Qwen3.5 applies RMSNorm per attention head to both Q and K after projection. This is called the zero-centered RMSNorm pattern because the weights are initialized to zero and applied as $x \cdot (1 + w)$:

$$\text{qkNorm}(x, w) = \frac{x}{\sqrt{\frac{1}{\text{head dim}} \sum_j x_j^2 + \epsilon}} \cdot (1 + w)$$

where $w$ is `q_norm.weight` or `k_norm.weight`, and $\epsilon = \text{norm eps} = 10^{-6}$. The weights are initialized to zero, so the norm starts as standard RMSNorm (scale = 1.0) and the `(1 + w)` formulation ensures the effective scale is never exactly zero.

The Q norm uses `q_norm.weight` of shape `(head_dim,) = (256,)`. The K norm uses `k_norm.weight` of the same shape.

### Partial RoPE

After Q/K norms, the base class applies RoPE using the `rot_mats` tensors passed in. For Qwen3.5, these are the corrected cos/sin matrices from `HfRotarySetup` (A3B) or `RotarySetup` (27B), patched at build time to apply rotation only to dimensions `[0, 63]` and leave dimensions `[64, 255]` unchanged (cos=1, sin=0 in those positions).

The RoPE op sees a full `head_dim=256` tensor and processes it uniformly — but the all-ones cos and all-zeros sin in the pass-through range mean the identity transform is applied there. No explicit slicing or host roundtrip occurs during inference.

### KV Cache Update

The current step's K and V tensors are written into the KV cache before attention is computed. Qwen3.5 uses a paged KV cache when `paged_attention_config` is provided. The cache stores bfloat16 tensors.

KV cache dimensions:
- 27B: 16 attention layers, `n_kv_heads=4`, `head_dim=256`
- 35B-A3B: 10 attention layers, `n_kv_heads=2`, `head_dim=256`

Paged KV cache is supported via `PagedAttentionConfig` passed to `GatedAttention.__init__` (which forwards it to `Attention.__init__`). The `page_table` tensor provided at forward time maps logical cache positions to physical pages.

### GQA Expansion

Qwen3.5 uses Grouped Query Attention. The KV heads are expanded to match the Q head count via `repeat_interleave` inside the base class:

$$\text{GQA ratio} = \frac{n_\text{heads}}{n_\text{kv heads}}$$

For 27B: $24 / 4 = 6$. For 35B-A3B: $16 / 2 = 8$.

Each KV head is repeated this many times along the head dimension before the dot-product attention. From `test_pcc.py`:

```python
gqa = N_HEADS // N_KV_HEADS        # 6 for 27B
key   = key.repeat_interleave(gqa, dim=1)
value = value.float().repeat_interleave(gqa, dim=1)
```

### Scaled Dot-Product Attention

Standard attention with scale $1 / \sqrt{\text{head dim}}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_\text{head}}}\right) V$$

For `head_dim=256`, the scale is $1 / \sqrt{256} = 0.0625$.

For single-token decode (sequence length = 1), the attention output shape before reshape is `[1, n_heads, 1, head_dim]`.

### `pre_wo_hook` Fires: Gate Application

After the softmax attention output is computed and reshaped to `[1, 1, B, n_heads * head_dim]`, the base class calls `self.pre_wo_hook(attn_output)` if the hook is set. Control transfers to `GatedAttention._apply_gate`:

1. `gate = ttnn.linear(self._gate_input, self.gate_weight)` → `[1, 1, B, n_heads * head_dim]`
2. `gate = ttnn.sigmoid(gate)` → values in $(0, 1)$
3. `attn_output = ttnn.to_memory_config(attn_output, DRAM_MEMORY_CONFIG)` → ensure layout compatibility
4. `result = ttnn.mul(attn_output, gate)` → element-wise product
5. Deallocate `gate` and `_gate_input`
6. Return `result`

### WO Projection

The gated attention output (returned from the hook) is projected back to `hidden_size` by the WO linear:

$$\text{output} = \text{gated attn output} \, W_O^\top$$

For 27B: `[1, 1, B, 6144]` → `[1, 1, B, 5120]`.
For 35B-A3B: `[1, 1, B, 4096]` → `[1, 1, B, 2048]`.

## PCC Validation

Both test suites assert `pcc >= PCC_THRESHOLD` where `PCC_THRESHOLD = 0.99`.

**27B (`test_pcc.py`, `TestGatedAttentionPCC`)**

Tests the first `full_attention` layer using the hardcoded constant `ATTENTION_LAYER = 3` (with a comment noting layer 3 is the first full_attention layer in the 27B architecture). Input is `torch.randn(1, 1, HIDDEN_SIZE)` with `torch.manual_seed(42)`, position `0`. The reference runs the complete HF-equivalent forward on CPU float32:

```python
pcc = compute_pcc(ref_flat, tt_flat)
assert pcc >= PCC_THRESHOLD, f"GatedAttention PCC {pcc:.6f} < {PCC_THRESHOLD}"
```

`compute_pcc` is Pearson Correlation Coefficient (not cosine similarity):

```python
def compute_pcc(x, y):
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    x_c = x_flat - x_flat.mean()
    y_c = y_flat - y_flat.mean()
    num = (x_c * y_c).sum()
    den = torch.sqrt((x_c**2).sum() * (y_c**2).sum())
    return (num / den).item()
```

**A3B (`test_a3b_pcc.py`)**

The A3B test suite validates `GatedDeltaNet` and `Qwen35MoE` layers but does not include a dedicated `TestGatedAttentionPCC` class for A3B (the A3B attention layers use the same `GatedAttention` class). The fused kernel test `TestFusedKernelPCC` uses a tighter threshold of `pcc >= 0.998` for output PCC and `pcc >= 0.999` for state PCC.

**Reference test (`test_attention_pcc.py`)**

The standalone reference script determines its layer index dynamically via `LAYER = next(i for i, t in enumerate(layer_types) if t == "full_attention")`, reading `layer_types` from the model's `config.json` at runtime. It prints PCC computed as cosine similarity (not Pearson) for a quick sanity check:

```python
pcc = torch.nn.functional.cosine_similarity(
    ref_vec.unsqueeze(0), tt_result.unsqueeze(0)
).item()
print(f"PCC (cosine similarity): {pcc:.6f}")
```

This is used for interactive debugging rather than automated CI.

## Tensor Shapes Summary

| Stage | Shape (27B) | Shape (A3B) |
|-------|-------------|-------------|
| Input `x` | `[1, 1, B, 5120]` | `[1, 1, B, 2048]` |
| Q projection | `[1, 1, B, 6144]` | `[1, 1, B, 4096]` |
| K/V projection | `[1, 1, B, 1024]` | `[1, 1, B, 512]` |
| Q per-head (before RoPE) | `[1, 24, B, 256]` | `[1, 16, B, 256]` |
| K per-head (before RoPE) | `[1, 4, B, 256]` | `[1, 2, B, 256]` |
| K/V after GQA expand | `[1, 24, B, 256]` | `[1, 16, B, 256]` |
| Attention output (pre-gate) | `[1, 1, B, 6144]` | `[1, 1, B, 4096]` |
| Gate tensor | `[1, 1, B, 6144]` | `[1, 1, B, 4096]` |
| Gated output | `[1, 1, B, 6144]` | `[1, 1, B, 4096]` |
| WO output | `[1, 1, B, 5120]` | `[1, 1, B, 2048]` |

`B` is the tile-padded batch size (`tile_padded_batch_rows`, typically 32 for single-batch decode).

---

**Next:** [Chapter 4 — Decoder Block and Uniform Dispatch](../ch4_decoder_block_and_uniform_dispatch/index.md)
