# Testing Infrastructure

The test suite is split into two tiers: **reference scripts** (pure PyTorch, no device
required) and **PCC tests** (require a Blackhole device and, for most tests, a downloaded
HuggingFace checkpoint).

## Directory layout

```
models/demos/qwen35/
├── reference/
│   ├── test_deltanet_pcc.py      # pure PyTorch, single DeltaNet layer
│   ├── test_deltanet_multi.py    # pure PyTorch, 20 sequential tokens
│   └── test_attention_pcc.py     # pure PyTorch, single GatedAttention layer
└── tests/
    ├── test_pcc.py               # device test, 27B model
    └── test_a3b_pcc.py           # device test, A3B model + fused kernel
```

## Reference scripts (`reference/`)

Reference scripts run without a Blackhole device. They validate the PyTorch reference
implementation against the HuggingFace model output to confirm the reference functions
used in the PCC tests are correct. They also serve as standalone correctness baselines
during kernel development.

### `reference/test_deltanet_pcc.py`

Validates a single DeltaNet (GatedDeltaNet) layer end-to-end against the HuggingFace
reference implementation (`torch_recurrent_gated_delta_rule`).

- Loads raw safetensors for one DeltaNet layer (shard lookup via `model.safetensors.index.json`)
- Runs HF reference forward on host in float32
- Runs the TTNN implementation via `GatedDeltaNet` with `MinimalArgs` (no full `ModelArgs` needed)
- Compares outputs via Pearson Correlation Coefficient (PCC ≥ 0.99)
- Requires `HF_MODEL` env var pointing to a local checkpoint directory

### `reference/test_deltanet_multi.py`

Runs 20 sequential tokens through one DeltaNet layer to detect state divergence over time.

- Tracks per-step PCC and recurrent state norm for all 20 steps
- Validates that the circular conv buffer and recurrent state accumulate correctly
- Useful after any change to the recurrence implementation or conv ring buffer logic
- PCC threshold: 0.95 minimum across all 20 steps

### `reference/test_attention_pcc.py`

Validates a single GatedAttention layer including partial RoPE, per-head RMSNorm,
GQA expansion, and the output gate.

- Auto-detects the first `full_attention` layer from `config.json` layer_types field
  (i.e., `LAYER = next(i for i, t in enumerate(layer_types) if t == "full_attention")`)
- Compares TTNN output against HF reference at PCC ≥ 0.99

## PCC test suite (`tests/`)

These tests require both a Blackhole device and (except `TestFusedKernelPCC`) a downloaded
HuggingFace checkpoint. They are the authoritative correctness gate before merging changes.

### `tests/test_pcc.py` (27B model)

Two test classes:

**`TestDeltaNetPCC.test_deltanet_single_token`**
- Layer: `DELTANET_LAYER = 0` (hardcoded)
- Loads the single-layer safetensors shard for layer 0
- Builds `GatedDeltaNet` with `MinimalArgs` (27B constants: `HIDDEN_SIZE=5120`, `NUM_V_HEADS=48`,
  `NUM_K_HEADS=16`, `HEAD_K_DIM=128`, `HEAD_V_DIM=128`)
- Threshold: PCC ≥ 0.99

**`TestGatedAttentionPCC.test_gated_attention_single_token`**
- Layer: `ATTENTION_LAYER = 3` (hardcoded; first full-attention layer in 27B)
- Builds `GatedAttention` with `MinimalArgs` (27B constants: `N_HEADS=24`, `N_KV_HEADS=4`,
  `HEAD_DIM=256`, `PARTIAL_ROTARY_FACTOR=0.25`)
- Threshold: PCC ≥ 0.99

Both classes use `convert_hf_to_meta_qwen35` from `load_checkpoints` to transform the
per-layer state dict before constructing the TTNN module.

### `tests/test_a3b_pcc.py` (A3B model)

Three test classes:

**`TestDeltaNetPCC`**
- `test_single_step`: A3B DeltaNet single token, PCC ≥ 0.99
- `test_multi_step`: 10 sequential tokens, minimum PCC ≥ 0.95 across all steps
- Uses `ModelArgs` (full args object, not `MinimalArgs`) and `model_args.load_state_dict()`
- A3B constants: `HIDDEN_SIZE=2048`, `NUM_V_HEADS=32`, `NUM_K_HEADS=16`, `HEAD_K_DIM=128`

**`TestMoEPCC`**
- `test_single_layer`: Single MoE layer (layer 0), PCC ≥ 0.99
- Constructs `Qwen35MoE` with `ModelArgs` and full state dict
- Reference: `ref_moe_forward` runs PyTorch SwiGLU + topk routing with float32 weights
- Expert indexing: `gate_up[eid, :MOE_INTERMEDIATE, :]` for gate, `gate_up[eid, MOE_INTERMEDIATE:, :]`
  for up (slice the fused `[256, 1024, 2048]` tensor)

**`TestFusedKernelPCC`**
- `test_single_step`: **Does not require model download.** Uses synthetic random tensors.
- Constructs inputs matching the fused kernel interface:
  `conv_out [1,1,B,conv_dim]`, `z_flat [1,1,B,H*D]`, `ba_flat [1,1,B,2H]`,
  `dt_bias [1,H,1,1]`, `neg_A_exp [1,H,1,1]`, `state [1,H,D,D]`, `norm_w [1,H,1,D]`
- Calls `ttnn.experimental.gated_delta_net(conv_out, z_flat, ba_flat, dt_bias, neg_A_exp, state, norm_w, ...)`
- Compares output vs Python reference recurrence
- Thresholds: output PCC ≥ 0.998, state PCC ≥ 0.999

## PCC metric

All tests use the same Pearson Correlation Coefficient implementation:

```python
def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    x_c = x_flat - x_flat.mean()
    y_c = y_flat - y_flat.mean()
    num = (x_c * y_c).sum()
    den = torch.sqrt((x_c**2).sum() * (y_c**2).sum())
    return (num / den).item() if den > 0 else 0.0
```

PCC = 1.0 is exact match. PCC ≥ 0.99 reflects typical bfp8 quantization error.
The fused kernel's higher threshold (≥ 0.998 for output, ≥ 0.999 for state) reflects that
it operates in float32 internally and should closely track the float32 reference.

## Weight conversion in tests

Tests that load per-layer state dicts apply `convert_hf_to_meta_qwen35` to transform HF key
names to meta format before constructing the module:

```python
from models.tt_transformers.tt.qwen35_utils import convert_hf_to_meta_qwen35
sd = {k.replace("model.", ""): v for k, v in layer_weights.items()}
sd = convert_hf_to_meta_qwen35(sd, HEAD_DIM, N_HEADS, N_KV_HEADS)
```

Note: `test_pcc.py` imports this function from `models.tt_transformers.tt.load_checkpoints`,
but the function is defined in `qwen35_utils.py`. The correct import is from `qwen35_utils`.

The `replace("model.", "")` strips the HF `model.` prefix. For per-layer tests the
MoE protection step in `convert_hf_to_meta_qwen35` is a no-op because the per-layer
weight dict does not contain the 3D expert tensors by default in test_pcc.py (the MinimalArgs
path), though test_a3b_pcc.py uses the full ModelArgs path which loads the complete state dict.

---

**Next:** [`running_tests.md`](./running_tests.md)
