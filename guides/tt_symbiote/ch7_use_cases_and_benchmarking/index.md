# Chapter 7: End-to-End Use Cases and Performance Benchmarking

This chapter shows complete integration workflows for three model families and explains how to read the timing output that TT Symbiote produces. Each section is written as a self-contained reference: identify the model family closest to yours, follow its walkthrough, then use the timing interpretation guidance to evaluate where device time is actually being spent.

## What this chapter covers

- A full LLaMA-3.2-1B-Instruct workflow drawn from `tests/test_llama.py`, including the `LlamaMLP` wrapper pattern, both `TTNNLinear` and `SmartTTNNLinear` replacement strategies, the `exclude_replacement` mechanism, and the files produced by `DispatchManager.save_stats_to_file`.
- Vision model integration based on `tests/test_vit.py` (ViT-base) and `tests/test_resnet50.py` (ResNet-50), plus the built-in conv modules available in `modules/conv.py`.
- Speech model integration drawn from `tests/test_whisper3.py` (`distil-whisper/distil-large-v3`), together with the DPL debugging workflow sourced from `tests/test_dpl.py` and the run-mode environment variables documented in `core/run_config.py`.

## How to use this chapter

Each content file is independent. Read the section that matches your model family to understand the exact replacement dictionaries, wrapper classes, and setup sequence used in the test suite. The timing interpretation notes at the end of each file apply to any model family, so the guidance in `llm_acceleration.md` applies equally when you run vision or speech tests.

## Table of contents

| File | Contents |
|---|---|
| [llm_acceleration.md](./llm_acceleration.md) | LLaMA-3.2-1B-Instruct walkthrough; `TTNNLinear` vs `SmartTTNNLinear`; timing CSV interpretation |
| [vision_and_multimodal.md](./vision_and_multimodal.md) | ViT and ResNet-50 workflows; NHWC conv modules; conceptual replacement patterns |
| [speech_and_debugging.md](./speech_and_debugging.md) | Whisper integration; DPL debugging modes; `TT_SYMBIOTE_DISPATCHER=DEBUG`; pivot table interpretation |

## Cross-references to prior chapters

- **Chapter 2** covers `TTNNModule`, `preprocess_weights`, and `move_weights_to_device` — the base primitives used in every setup sequence in this chapter.
- **Chapter 3** covers `register_module_replacement_dict`, `nn_to_nn` pre-passes, and `exclude_replacement` — the replacement machinery called in every test file here.
- **Chapter 4** covers `TorchTTNNTensor` and the run-mode dispatch stack that underpins `DPL` and `DPL_NO_ERROR_PROP`.
- **Chapter 5** covers the built-in module library (`TTNNLinear`, `TTNNRMSNorm`, `LlamaAttention`, `TTNNConv2dBNActivationNHWC`, `TTNNBottleneck`, `TTNNViTEmbeddings`, `TTNNWhisperAttention`) referenced throughout this chapter.
- **Chapter 6** covers `DispatchManager` timing infrastructure and the `set_device` hook that instruments every `forward` call.
