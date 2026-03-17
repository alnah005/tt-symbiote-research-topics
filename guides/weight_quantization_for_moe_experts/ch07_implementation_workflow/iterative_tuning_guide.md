# Iterative Tuning Guide

## Purpose

Step 5 of the workflow applies the PCC measurements from Step 3 and the latency data
from Step 4 to a structured decision tree. The output is a locked final configuration —
a concrete `{dtype, MathFidelity}` tuple per projection — supported by calibration
perplexity evidence and protected by a regression test suite that runs after every
checkpoint update or TTNN version bump.

---

## Decision Tree for Tuning After Initial Validation

Start from the recommended Qwen starting point (bfloat8_b all + HiFi2) or from any
preliminary configuration, then apply the tree in order.

```
Initial configuration (e.g., bfloat8_b all + HiFi2)
│
├── Run per-layer PCC validation (per_layer_pcc_validation.md)
│
├── DOWN PROJECTION (w2) failing PCC < 0.975?
│   ├── YES → Option A: upgrade bfloat8_b → bfloat16 for w2
│   │         Option B: upgrade HiFi2 → HiFi4 for w2 (keeping bfloat8_b)
│   │         Re-run PCC. If still failing → use bfloat16 for w2.
│   └── NO  → w2 configuration is acceptable; continue.
│
├── GATE / UP PROJECTIONS (w1, w3) failing PCC < 0.96?
│   ├── YES → Option A: upgrade bfloat4_b → bfloat8_b for failing projection(s)
│   │         Option B: upgrade LoFi → HiFi2 for failing projection(s)
│   │         Re-run PCC. If still failing → use bfloat8_b + HiFi2 for that projection.
│   └── NO  → Gate/up configuration is acceptable; continue.
│
├── THROUGHPUT improvement insufficient?
│   ├── Is the matmul truly memory-bound? (check arithmetic intensity, Chapter 4)
│   │   ├── YES, memory-bound → consider bfloat4_b for gate/up if not already set
│   │   │   AND accuracy budget permits (PCC ≥ 0.96, perplexity delta ≤ 2.0 PPL)
│   │   └── NO, compute-bound at current batch → quantization alone will not close gap;
│   │       consider program config (tile sizes, grid) tuning instead
│   └── T3K: is communication latency masking FFN gains?
│       └── YES → throughput tuning is bounded by communication floor;
│           prioritize memory footprint reduction over compute latency
│
└── All checks pass → run calibration perplexity (see below) → lock in config
```

# Configs: see index.md (COMPUTE_KERNEL_CONFIG_LOFI, COMPUTE_KERNEL_CONFIG_HIFI2)

> **Warning:** HiFi4 is available as a fallback above LoFi and HiFi2, but it uses
> four-pass accumulation with FP32 destination and significantly reduces throughput. Use
> HiFi4 only as a last resort before reverting to bfloat16, and document the decision
> clearly in the model config dataclass.

---

## Calibration Perplexity Procedure

Per-layer PCC validates numerical similarity of individual forward passes but does not
directly measure model quality on natural language. Run a calibration perplexity sweep to
confirm that per-layer PCC thresholds translate to acceptable downstream model behaviour.

### Dataset and Tokenization

Use WikiText-2 validation set with 512-token segments. This is long enough to capture
multi-token dependencies (which expose cumulative quantization error) while keeping the
sweep fast enough to run during a validation pipeline.

```python
def compute_calibration_perplexity(model_fn, tokenizer, dataset_samples, max_tokens=512):
    """Compute perplexity on a list of text samples.

    Args:
        model_fn: Callable(input_ids) -> logits tensor (CPU, float32).
                  Should use the quantized model weights under test.
        tokenizer: HuggingFace tokenizer for the model.
        dataset_samples: List of strings (WikiText-2 validation segments).
        max_tokens: Maximum token length per sample (truncate or skip if longer).

    Returns:
        perplexity: float.
    """
    import math
    total_log_prob = 0.0
    total_tokens   = 0

    for text in dataset_samples:
        input_ids = tokenizer.encode(text, return_tensors="pt")
        if input_ids.shape[1] > max_tokens:
            input_ids = input_ids[:, :max_tokens]

        n_tokens = input_ids.shape[1] - 1  # predict all but the first token
        with torch.no_grad():
            logits = model_fn(input_ids)   # [1, seq_len, vocab_size]

        log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
        target_log_probs = log_probs[
            torch.arange(n_tokens), input_ids[0, 1:]
        ]
        total_log_prob += target_log_probs.sum().item()
        total_tokens   += n_tokens

    avg_nll    = -total_log_prob / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity
```

### Acceptance Thresholds

| Configuration | Perplexity delta from bfloat16 baseline | Accept? |
|---|---|---|
| `bfloat8_b` all + HiFi2 | ≤ 1.0 PPL | Yes |
| `bfloat8_b` all + HiFi2 | > 1.0 PPL | Investigate per-layer PCC; consider falling back one projection |
| Mixed: `bfloat4_b` gate/up + `bfloat8_b` down | ≤ 2.0 PPL | Yes |
| Mixed: `bfloat4_b` gate/up + `bfloat8_b` down | > 2.0 PPL | Fall back gate/up to `bfloat8_b`; re-measure |

For perplexity threshold guidance and visual sanity checks, see `../ch05_per_projection_strategy/qwen_adaptation_guide.md`.

---

## Locking In the Final Configuration

Once both per-layer PCC and calibration perplexity pass, record the final configuration
in the model config dataclass. Every field must be documented with the evidence that
justified it.

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ExpertQuantizationConfig:
    """Locked per-projection quantization configuration for one MoE model.

    Store this alongside the model's ModelConfig dataclass. All fields must be filled
    before committing a configuration to production.
    """

    # Gate projection (w1)
    gate_dtype: str = "bfloat4_b"           # ttnn dtype name as string
    gate_math_fidelity: str = "LoFi"
    gate_pcc_measured: float = 0.0          # fill from per-layer validation run
    gate_pcc_threshold: float = 0.96

    # Up projection (w3)
    up_dtype: str = "bfloat4_b"
    up_math_fidelity: str = "LoFi"
    up_pcc_measured: float = 0.0
    up_pcc_threshold: float = 0.96

    # Down projection (w2)
    down_dtype: str = "bfloat8_b"
    down_math_fidelity: str = "HiFi2"
    down_pcc_measured: float = 0.0
    down_pcc_threshold: float = 0.975

    # Full-layer validation
    full_layer_pcc_measured: float = 0.0
    full_layer_pcc_threshold: float = 0.97

    # Calibration perplexity
    baseline_perplexity: float = 0.0        # bfloat16 reference PPL
    quantized_perplexity: float = 0.0       # quantized config PPL
    perplexity_delta: float = 0.0           # computed: quantized - baseline
    perplexity_delta_threshold: float = 2.0 # 1.0 for bfloat8_b all; 2.0 for mixed

    # Validation metadata
    validation_date: str = ""               # ISO date string, e.g. "2026-03-17"
    ttnn_version: str = ""                  # ttnn.__version__
    calibration_dataset: str = "wikitext-2-v1"
    calibration_tokens: int = 512
    notes: str = ""                         # rationale for any non-default choices

    def assert_valid(self):
        """Raise AssertionError if any measured value falls below its threshold."""
        assert self.gate_pcc_measured  >= self.gate_pcc_threshold,  \
            f"Gate PCC {self.gate_pcc_measured:.4f} < {self.gate_pcc_threshold}"
        assert self.up_pcc_measured    >= self.up_pcc_threshold,    \
            f"Up PCC {self.up_pcc_measured:.4f} < {self.up_pcc_threshold}"
        assert self.down_pcc_measured  >= self.down_pcc_threshold,  \
            f"Down PCC {self.down_pcc_measured:.4f} < {self.down_pcc_threshold}"
        assert self.full_layer_pcc_measured >= self.full_layer_pcc_threshold, \
            f"Full layer PCC {self.full_layer_pcc_measured:.4f} < {self.full_layer_pcc_threshold}"
        assert self.perplexity_delta   <= self.perplexity_delta_threshold, \
            f"Perplexity delta {self.perplexity_delta:.2f} > {self.perplexity_delta_threshold}"
```

The `assert_valid` method serves as both documentation and a runtime guard: call it at
model initialization to catch stale or incomplete configurations before serving traffic.

---

## Regression Testing

Quantization PCC is sensitive to changes in TTNN kernel implementations and to changes in
the model checkpoint (e.g., a new fine-tuned checkpoint can shift weight distributions).
Add quantization-specific assertions to the model test suite so that any regression is
caught at the CI (continuous integration) level, not in production.

### Regression Test Template

```python
import pytest
import torch
import ttnn

# Reuse PCC harness from per_layer_pcc_validation.md
from .per_layer_pcc_validation import pcc_test_full_moe_layer

DECODE_TOKEN_COUNT  = 1
PREFILL_TOKEN_COUNT = 2048
D_MODEL = 7168  # Qwen 235B-A22B

@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    ttnn.device.enable_program_cache(d)
    yield d
    ttnn.close_device(d)


@pytest.mark.parametrize("token_count,regime", [
    (DECODE_TOKEN_COUNT,  "decode"),
    (PREFILL_TOKEN_COUNT, "prefill"),
])
def test_expert_quantization_pcc(token_count, regime, device, request):
    """Regression test: quantized expert PCC must meet thresholds in both regimes.

    Loads a fixed random weight seed so results are deterministic across runs.
    Replace torch.randn with actual checkpoint weights for production regression tests.
    """
    torch.manual_seed(0)
    w1_bf16 = torch.randn(2048, D_MODEL, dtype=torch.bfloat16)
    w3_bf16 = torch.randn(2048, D_MODEL, dtype=torch.bfloat16)
    w2_bf16 = torch.randn(D_MODEL, 2048, dtype=torch.bfloat16)

    x_torch = torch.randn(token_count, D_MODEL, dtype=torch.bfloat16)

    results = pcc_test_full_moe_layer(
        x_torch, w1_bf16, w3_bf16, w2_bf16, layer_idx=0, device=device
    )

    assert results["gate_pcc"] >= 0.96,  \
        f"[{regime}] Gate PCC {results['gate_pcc']:.4f} < 0.96"
    assert results["up_pcc"]   >= 0.96,  \
        f"[{regime}] Up PCC {results['up_pcc']:.4f} < 0.96"
    assert results["down_pcc"] >= 0.97,  \
        f"[{regime}] Full layer PCC {results['down_pcc']:.4f} < 0.97"
```

### When to Run Regression Tests

Run the regression test suite in these situations:

1. **After any TTNN version bump** — kernel implementations can change accumulation order
   or rounding behaviour, shifting PCC.
2. **After loading a new model checkpoint** — fine-tuned or continued-pretraining
   checkpoints can shift weight distributions enough to affect bfloat4_b outlier behaviour.
3. **After any change to the compute kernel config** — even a single field change
   (`math_approx_mode`, `packer_l1_acc`) can affect numerical output.
4. **After cache invalidation** — if the TTNN weight cache was deleted and re-generated,
   verify that the new cache produces the same PCC.

> **Warning:** Do not run only one of decode or prefill in regression tests. Both regimes
> must be covered because their arithmetic intensity and quantization noise profiles
> differ. A change that degrades decode PCC may not be visible in prefill results due to
> the averaging effect over 2048 tokens.

---

## Summary: Full Workflow Checklist

| Step | File | Artifact |
|---|---|---|
| 1 | `baseline_and_weight_conversion.md` | Baseline PCC > 0.999 confirmed |
| 2 | `baseline_and_weight_conversion.md` | Converted weights on device; conversion PCC verified |
| 3 | `per_layer_pcc_validation.md` | Per-layer PCC report for all layers in decode + prefill |
| 4 | `throughput_profiling.md` | Per-op latency table; decode improvement quantified |
| 5 (this file) | `iterative_tuning_guide.md` | Locked `ExpertQuantizationConfig`; passing perplexity; regression tests in CI |

---

**End of guide.** Return to [Guide Index](../index.md)
