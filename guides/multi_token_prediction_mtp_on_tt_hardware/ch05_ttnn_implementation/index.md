# Chapter 5: TTNN Implementation Strategy for MTP-Based Speculative Decoding

## Prerequisites

This chapter assumes familiarity with the material covered in Chapters 1–4:

- **Chapter 1** (`ch01_mtp_foundations/`) — MTP architecture: what the MTP head is, how it relates to the backbone, and the Qwen3.6-35B-A3B configuration (`mtp_num_hidden_layers = 1`, dense FFN, shared `lm_head`)
- **Chapter 2** (`ch02_mtp_weights_and_memory/`) — MTP weight keys (`model.future_prediction[0].*`), parameter count (~160M), BF16 memory footprint (304.6 MiB), and contrast with the 70 GiB backbone
- **Chapter 3** (`ch03_mtp_in_huggingface/`) — MTP is training-only in standard HuggingFace; `model.generate()` never invokes the MTP head; weight loading via `AutoModelForCausalLM.from_pretrained` succeeds without interference
- **Chapter 4** (`ch04_speculative_decoding_with_mtp/`) — The K=1 speculative decoding algorithm, acceptance/rejection mechanics, the E[tokens/cycle] = 1 + α formula, and the cost analysis showing speedup > 1 requires batch size > 1 or K ≥ 3

## Scope

MTP is guarded as training-only in standard HuggingFace code (see Chapter 3, `ch03_mtp_in_huggingface/index.md`). This chapter details the **explicit modifications** required to activate MTP at inference time on TT hardware:

1. A new `TTNNMTPHead` module wrapping the single MTP transformer block
2. A modified generation loop that executes the three-pass speculative decode cycle (primary → draft → verify)
3. Memory placement decisions for MTP head weights and activations
4. Testing and validation procedures to confirm correctness before measuring throughput gains

This chapter does **not** cover training-time MTP loss computation, multi-token MTP heads (K > 1 draft tokens from a single MTP block), or changes to the backbone TTNN module itself. The backbone forward pass is unchanged.

> **Key Finding:** The implementation requires minimal new code — one new `TTNNMTPHead` module (reusing existing attention/FFN primitives already in tt-transformers) and a modified generation loop. No new TTNN kernel development is needed. The correctness of the throughput model depends critically on **not resampling position t+2 on rejection** (see Chapter 4, `ch04_speculative_decoding_with_mtp/`); this constraint must be explicitly preserved in the loop implementation. Violating it collapses the expected speedup to 1.0 regardless of the empirical acceptance rate α.

## Chapter Overview

| File | Topic |
|---|---|
| `index.md` | This file — prerequisites, scope, and navigation |
| `mtp_head_ttnn_module.md` | `TTNNMTPHead` module design: inputs, architecture, weight loading, toggle flag |
| `speculative_decode_loop_integration.md` | Modified generation loop: three-pass cycle, KV cache management, batch handling |
| `memory_placement_for_mtp.md` | Tensor placement decisions: weights and activations on DRAM vs. L1 |
| `testing_and_validation.md` | Correctness tests, non-regression, acceptance rate harness, throughput benchmarks |

## References

- Chapter 1: `ch01_mtp_foundations/`
- Chapter 2: `ch02_mtp_weights_and_memory/`
- Chapter 3: `ch03_mtp_in_huggingface/`
- Chapter 4: `ch04_speculative_decoding_with_mtp/`
