# Chapter 7: Worked Example — LLaMA 3.2-1B on Symbiote with TT Transformers Optimizations

## Overview

This chapter provides a concrete, end-to-end integration walkthrough using `meta-llama/Llama-3.2-1B-Instruct` as the target model. Chapters 1–6 established the conceptual framework for mapping HuggingFace modules into Symbiote and adopting TT Transformers precision and sharding patterns. Here those concepts are grounded in the actual source of `test_llama.py` and the modules it exercises.

The chapter is organized into three sequential steps:

| Step | File | What it covers |
|------|------|----------------|
| 1 | [step1_module_map.md](step1_module_map.md) | HF module hierarchy, the `LlamaMLP` wrapper, the complete class-level replacement table, `exclude_replacement`, and gaps |
| 2 | [step2_precision_config.md](step2_precision_config.md) | Dtype selection for each component, TT Transformers reference dtypes, recommended Symbiote classes per component, DPL validation |
| 3 | [step3_validation_and_benchmarking.md](step3_validation_and_benchmarking.md) | Running the tests, reading timing CSVs, interpreting DPL output, known limitations, N300 path |

## Hardware Targets

**N150 (single chip, Wormhole)** is the primary validation target for this chapter. All code paths shown compile and run on a single N150. The two test functions in `test_llama.py` — `test_llama` and `test_llama_intelligent` — are both written and validated for single-chip execution.

**N300 (two-chip, Wormhole)** represents the distributed path. The sharded linear variants (`TTNNLinearIColShardedWRowSharded`, `TTNNLinearLLamaIColShardedWRowSharded`) are annotated `@run_on_devices(DeviceArch.T3K)` and require a mesh device. Step 3 describes what this path would add without requiring N300 hardware for the walkthrough itself.

## Prerequisites

Readers should have completed Chapters 1–6 and be familiar with:

- The `TTNNModule` base class and the `from_torch` / `preprocess_weights` / `move_weights_to_device` lifecycle
- The `register_module_replacement_dict` API and how it walks the model tree
- The `DispatchManager` timing infrastructure and run-mode environment variables
- The `TorchTTNNTensor` wrapper and how it bridges PyTorch dispatch to TTNN ops

## What This Chapter Delivers

1. A concrete mapping table that ties each HuggingFace `LlamaForCausalLM` submodule class to its Symbiote replacement and its nearest TT Transformers equivalent.
2. Precision decisions grounded in both `test_llama.py` and the `MLP` / `Attention` constructors in `models/tt_transformers/tt/`.
3. Step-by-step validation commands, CSV column definitions, and a checklist for confirming per-layer accuracy before removing DPL mode.
4. An honest inventory of current limitations — trace capture gaps, LM head exclusion, memory deallocation trade-offs — so readers can assess production readiness.
