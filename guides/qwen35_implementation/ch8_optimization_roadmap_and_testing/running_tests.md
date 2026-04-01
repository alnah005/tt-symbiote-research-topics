# Running Tests

## Environment setup

All tests that load model weights require the `HF_MODEL` environment variable to point
to a local HuggingFace checkpoint directory. If it points to a HF hub ID instead, the
loader will call `snapshot_download` to fetch the model automatically.

```bash
# Option A: point to a local directory
export HF_MODEL=/path/to/Qwen3.5-35B-A3B

# Option B: let snapshot_download fetch it (requires network + ~17 GB disk for A3B)
export HF_MODEL=Qwen/Qwen3.5-35B-A3B
```

Tests that do not require model weights (only `TestFusedKernelPCC`) ignore `HF_MODEL`.

## Quick test — no model download required

`TestFusedKernelPCC` constructs synthetic tensors and validates the fused DeltaNet kernel
on device without any checkpoint.

```bash
pytest models/demos/qwen35/tests/test_a3b_pcc.py::TestFusedKernelPCC -v -s
```

Expected output:
```
tests/test_a3b_pcc.py::TestFusedKernelPCC::test_single_step PASSED
  Fused kernel output PCC: 0.998xxx
  Fused kernel state PCC:  0.999xxx
```

## Full A3B PCC suite

Runs all three test classes for the 35B-A3B model. Requires device + checkpoint (~17 GB).

```bash
HF_MODEL=Qwen/Qwen3.5-35B-A3B pytest models/demos/qwen35/tests/test_a3b_pcc.py -v -s
```

For test class details, constants, and threshold rationale see `testing_infrastructure.md`.

## Full 27B PCC suite

Runs DeltaNet + GatedAttention tests for the 27B dense model. Requires device + checkpoint
(~50 GB).

```bash
HF_MODEL=Qwen/Qwen3.5-27B pytest models/demos/qwen35/tests/test_pcc.py -v -s
```

## Reference scripts (no device required)

Reference scripts run as standalone Python programs, not via pytest. They require `HF_MODEL`
pointing to a local checkpoint directory (no auto-download in reference scripts).

```bash
# Single DeltaNet layer PCC
HF_MODEL=/path/to/model python models/demos/qwen35/reference/test_deltanet_pcc.py

# 20-step sequential DeltaNet (state divergence check)
HF_MODEL=/path/to/model python models/demos/qwen35/reference/test_deltanet_multi.py

# Single GatedAttention layer PCC
HF_MODEL=/path/to/model python models/demos/qwen35/reference/test_attention_pcc.py
```

Reference scripts print per-step PCC and pass/fail summaries to stdout. They do not use
pytest fixtures so they can be run in environments without a device driver installed.

## End-to-end decode demo

To run full generation on A3B:

```bash
export HF_MODEL=Qwen/Qwen3.5-35B-A3B
python models/demos/qwen35/demo/demo_a3b.py \
  --prompt "Explain linear attention in one paragraph" \
  --max_tokens 100
```

For the 27B dense model:

```bash
export HF_MODEL=Qwen/Qwen3.5-27B
python models/demos/qwen35/demo/demo.py \
  --prompt "Explain linear attention in one paragraph" \
  --max_tokens 100
```

The demo reports compile time (step 0), warmup time (step 1), and steady-state decode
metrics (avg ms/token, tok/s) after step 2.

## Adding a new test

When adding a test for a new module or optimization:

1. **Single-layer test:** Follow the `MinimalArgs` pattern in `test_pcc.py`. Construct only
   the attributes the module reads — avoid importing `ModelArgs` unless you need the full
   weight-loading pipeline.

2. **Reference forward:** Add a `ref_<module>_forward(weights, hidden_states)` function that
   mirrors the HF implementation. Run it first to establish the float32 ground truth.

3. **PCC threshold:** Use 0.99 for bfp8/bf16 layer tests, 0.998 for fused kernel output,
   and 0.999 for state tensors. Lower thresholds (e.g., 0.95) are only appropriate for
   multi-step tests where quantisation error accumulates.

4. **No model download:** If the test can use synthetic data (e.g., testing a specific kernel
   interface), place it in a `TestXxxKernelPCC` class that constructs random tensors.
   This lets CI verify the kernel without fetching a 17 GB checkpoint.

---

**End of guide.** Return to [Guide Index](../index.md)
