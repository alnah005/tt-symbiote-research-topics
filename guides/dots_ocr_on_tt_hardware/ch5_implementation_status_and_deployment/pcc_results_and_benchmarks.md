# PCC Results and Benchmarks

## Overview

This file summarizes the PCC (Pearson Correlation Coefficient) results that have been confirmed by commit history, the targets stated in `IMPLEMENTATION_STEPS.md` that have not yet been independently confirmed, and the methodology for running the 14-test validation suite. Benchmark methodology for `perf/benchmark.py` is covered in the final section.

### PCC Results

| Component | PCC | Status | Source |
|-----------|-----|--------|--------|
| Text decoder prefill | > 0.98 | Confirmed | commit "prefill at 0.98" |
| Per-component (patch merger, vision blocks, decoder) | > 0.99 | Target | `IMPLEMENTATION_STEPS.md` |
| End-to-end (`test_e2e_pcc.py`) | > 0.99 | Target | `IMPLEMENTATION_STEPS.md` |

> **Note:** The stated target is PCC > 0.99 (per `IMPLEMENTATION_STEPS.md`); this figure has not been independently confirmed by commit history. The only PCC data point attributable to a specific commit is PCC > 0.98 for the text decoder prefill.

### Test Execution Order

Run the 14 test files in the order below to isolate failures at the earliest possible stage. Tests early in the sequence have no device dependency or minimal dependency; failures there indicate environment or weight-loading problems rather than numerical precision issues, which saves time when debugging on shared T3K hardware.

1. **`test_environment.py`** — Verifies hardware availability, software prerequisites (TTNN version, HuggingFace Transformers, trust_remote_code support). Run first; if this fails, no other test is meaningful.

2. **`test_weight_loading.py`** — Loads the dots.ocr checkpoint and checks tensor shapes against expected values (text decoder hidden size 1536, vision encoder hidden size 1536, vocab size 151936, 28 decoder layers, 42 vision layers). Catches checkpoint corruption or wrong model revision before any forward pass.

3. **`test_reference_embeddings.py`** — Runs the `reference/` CPU (PyTorch) model and validates that its embeddings match the HF `AutoModel` output. Establishes that the CPU oracle is correct before comparing against TTNN.

4. **`test_pcc_reference.py`** — Full reference model (`reference/`) vs. HF model PCC check. This is the baseline: if the CPU oracle does not match HF, TTNN PCC results are uninterpretable.

5. **`test_decoder_smoke.py`** — Fast smoke test for the TTNN text decoder (single forward pass, small sequence). Confirms that the decoder loads onto the device and produces output without crashing before the slower PCC tests run.

6. **`test_text_prefill_pcc.py`** — Text decoder prefill PCC test. This is the confirmed > 0.98 milestone from commit 3. A regression below 0.98 here indicates a weight-loading or RoPE alignment problem introduced after commit 3.

7. **`test_patch_merger_pcc.py`** — PCC test for `PatchMergerTT`, the TTNN spatial merge layer reused from the `qwen25_vl` demo. Isolates the merger from the full vision stack; a failure here is attributable to the merger rather than the 42-layer ViT.

8. **`test_vision_pcc.py`** — Per-component vision PCC: individual ViT blocks, attention layers, and MLP layers tested independently. Use this test to localize which layer is failing before running the full 42-layer tower test.

9. **`test_vision_tower_pcc.py`** — Full 42-layer vision tower PCC test. This is the primary gate for Step 4 (full TTNN vision stack). A pass here at PCC > 0.99 would be the first commit-independent confirmation of Step 4 completion on the target hardware.

10. **`test_fusion.py`** — Scatter fusion correctness test: validates that image token IDs (`image_token_id=151665`) are correctly replaced with patch embeddings during the fusion step between the vision encoder output and the text decoder input sequence.

11. **`test_e2e_pcc.py`** — End-to-end PCC test: full forward pass from raw image + text input to logits, comparing TTNN output against the HF reference. This is the key gate before production. A pass here at PCC > 0.99 covers Steps 3, 4, and 5 simultaneously under full TTNN operation.

12. **`test_mesh_topology.py`** — T3K submesh lifecycle test: open full mesh, carve 1x2 submesh, run a minimal forward pass, close submesh, close full mesh. Validates that `close_dots_mesh_device()` teardown order does not leave orphaned device handles.

13. **`test_demo_hf_torch_only.py`** — CPU-only demo test: runs the HF PyTorch demo path without any TTNN device. No T3K hardware required. Useful for verifying that the `reference/` model and `demo/reference_demo.py` are correct on any development machine.

> **Note:** The plan for the PCC validation framework listed 13 test files; the actual `tests/` directory contains 14. The list above reflects the 14 files present in the repository.

### Benchmark Methodology

The benchmark script is `perf/benchmark.py`. It measures three metrics:

- **TTFT (ms):** Time to first token — the latency from prompt submission to the first decoded output token. This includes all prefill chunks plus the first decode step.
- **Decode throughput (tokens/sec):** Sustained throughput after the first token, measured over a fixed number of decode steps.
- **Per-token decode latency (ms):** Inverse of throughput; useful for SLA calculations.

#### Required Environment Variables

Set these before any device initialization, before `perf/benchmark.py` is invoked:

```bash
export DOTS_T3K_TP=2
export DOTS_T3K_OPEN_FULL_MESH=1
export DOTS_LM_HEAD_MAX_COLUMNS_PER_DEVICE=2048
export DOTS_MAX_SEQ_LEN=<max_sequence_length>
export DOTS_MAX_SEQ_LEN_WH_LB=<chunked_prefill_window_lower_bound>
```

#### TTFT and Chunked Prefill

TTFT is directly affected by the chunked prefill configuration:

```
TTFT = (num_prefill_chunks × time_per_chunk) + decode_step_1_latency
```

Where `num_prefill_chunks = ceil(prompt_length / max_prefill_chunk_size)`.

`DOTS_MAX_SEQ_LEN_WH_LB` sets the lower bound on the window size used to compute chunk dimensions. Increasing it reduces `num_prefill_chunks` (lower TTFT for long prompts) but increases per-chunk L1 SRAM pressure (higher risk of L1 capacity errors for large images or long text sequences).

**Tuning guidance:** Benchmark both dimensions. Start with the default `DOTS_MAX_SEQ_LEN_WH_LB`, record TTFT and any OOM events, then increase the value incrementally and re-benchmark. The optimal setting minimizes TTFT while remaining below the L1 capacity limit for the document sizes expected in the OCR workload.

### Demo Usage

The primary demo is `demo/demo.py`. Two backends are supported:

```bash
# TTNN path — T3K device required; uses TTNN text decoder (hybrid or full TTNN vision)
python demo/demo.py --backend ttnn

# HF reference path — no device required; slower; uses pure HuggingFace PyTorch
python demo/demo.py --backend hf
```

Both paths require the same environment variables as the benchmark when `--backend ttnn` is selected. For `--backend hf`, no device env vars are needed.

**`demo/reference_demo.py`** is the pure HF demo, equivalent to `--backend hf` but as a standalone script. Use it for correctness comparisons: if output from `reference_demo.py` matches HF but `demo.py --backend ttnn` does not, the divergence is in the TTNN path.

**`demo/pyth.py`** is a sandbox/prototype script used during development. It is not a production demo entry point.

**`demo/sample_prompts/`** contains document type samples (cover letters and similar document formats). For OCR use cases, add custom prompt files in the same directory or pass prompts directly via the demo CLI.

**Next:** [TT-Symbiote Integration Gaps](tt_symbiote_integration_gaps.md)
