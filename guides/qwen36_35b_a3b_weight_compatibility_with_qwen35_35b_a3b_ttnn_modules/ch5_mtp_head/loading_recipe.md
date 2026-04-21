# MTP Weight Loading and TT-Symbiote Impact

## Loading Scenario A — `AutoModelForCausalLM.from_pretrained` (Default Settings)

Using the standard HuggingFace load path:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("path/to/qwen3.6-35b-a3b", torch_dtype="bfloat16")
```

**Result:** succeeds without error.

The MTP keys (`model.future_prediction[0].*`) are recognized as valid submodule parameters of `Qwen3_5MoeForConditionalGeneration`. The class instantiates `model.future_prediction` as part of its `__init__` whenever `mtp_num_hidden_layers >= 1`, regardless of inference mode. No `unexpected keys` or `missing keys` warnings are emitted.

Behavior after loading:

- `model.future_prediction[0]` is populated with BF16 weights in CPU RAM (~304.6 MiB)
- At inference — `model.eval()` followed by `model.generate()` — the MTP weights are loaded but never used
- The MTP gate (`labels is not None AND self.training is True`) is never satisfied during generation (see `mtp_architecture.md`)
- Output quality is identical to a hypothetical model with MTP weights removed

**Risk level for pure HuggingFace CPU/GPU inference:** **none**. No action required. The ~304.6 MiB overhead may be acceptable depending on available host RAM.

## Loading Scenario B — Direct Safetensors Load + Manual State Dict Filtering

TT-Symbiote's weight preprocessing pipeline typically reads weights directly from safetensors shards and passes them through TTNN module weight hooks for dtype casting, sharding, and key renaming. This is the relevant scenario for deploying Qwen3.6-35B-A3B on Tenstorrent hardware.

If the full Qwen3.6 state dict is passed to TT-Symbiote's preprocessing pipeline without filtering:

- MTP keys (`model.future_prediction[0].*`) will be present in the state dict
- These keys do **not** match any patterns consumed by the existing TTNN modules (`TTNNQwen3FullAttention`, `TTNNQwen3LinearAttention`, `TTNNQwen3MoE`)
- The MTP keys are unrecognized by all current TTNN backbone modules

**Risk:** **medium**. Two failure modes are possible:

1. If TT-Symbiote's weight preprocessing pipeline raises an error or emits a warning on unrecognized keys, MTP keys will trigger it and could abort the loading process.
2. If the pipeline silently ignores unrecognized keys, no error occurs but the MTP weights occupy ~304.6 MiB of CPU RAM unnecessarily — and the absence of an error provides no confirmation that the filter is working as expected.

**Recommendation:** Filter MTP keys before passing the state dict to TT-Symbiote.

## MTP Key Filter

```python
def filter_mtp_keys(state_dict: dict) -> dict:
    """Remove MTP head keys from state dict before passing to TT-Symbiote."""
    return {
        k: v for k, v in state_dict.items()
        if not k.startswith("model.future_prediction")
    }
```

Insert this filter between safetensors loading and TT-Symbiote weight preprocessing:

```python
from safetensors.torch import load_file

raw_state_dict = load_file("path/to/qwen3.6-35b-a3b.safetensors")
filtered_state_dict = filter_mtp_keys(raw_state_dict)
tt_symbiote_model.load_weights(filtered_state_dict)
```

The predicate `k.startswith("model.future_prediction")` is the complete filter. All 9 MTP key patterns (enumerated in `mtp_architecture.md`) share this prefix. No other Qwen3.6 keys share the prefix.

For multi-shard checkpoints, apply `filter_mtp_keys` after loading each shard or after merging shards, before the preprocessing step:

```python
from safetensors.torch import load_file
import glob

raw_state_dict = {}
for shard_path in sorted(glob.glob("path/to/qwen3.6-35b-a3b/*.safetensors")):
    raw_state_dict.update(load_file(shard_path))

filtered_state_dict = filter_mtp_keys(raw_state_dict)
tt_symbiote_model.load_weights(filtered_state_dict)
```

## Validation Step

After loading, confirm that no MTP-prefixed keys are present in the set of TTNN device tensors:

```python
loaded_keys = set(tt_symbiote_model.get_parameter_keys())
mtp_keys = [k for k in loaded_keys if "future_prediction" in k]
assert len(mtp_keys) == 0, f"MTP keys leaked into TTNN parameters: {mtp_keys}"
```

This assertion should always pass when `filter_mtp_keys` is applied correctly. If it fails, it indicates that either the filter was not applied or a new MTP key pattern has been introduced that does not match the `model.future_prediction` prefix.

## Overall TT-Symbiote Impact Summary

| Component | Impact | Action required |
|---|---|---|
| Backbone TTNN modules (`TTNNQwen3FullAttention`, `TTNNQwen3MoE`, etc.) | None — backbone weight shapes identical to Qwen3.5 | None |
| MTP weights in state dict | Extra keys not consumed by any TTNN module | Filter before passing to weight preprocessing pipeline |
| Inference output | None — MTP head is training-only; `model.generate()` never invokes it | None |
| CPU RAM during loading | ~304.6 MiB if state dict is not filtered before CPU load | Use filtered safetensors loading to avoid loading unused weights |

> **Key Finding:** The MTP head introduces extra state dict keys but requires no changes to the backbone TTNN modules. The only required TT-Symbiote change is a key filter applied before weight preprocessing. This is a one-line predicate on the key prefix `"model.future_prediction"`.
