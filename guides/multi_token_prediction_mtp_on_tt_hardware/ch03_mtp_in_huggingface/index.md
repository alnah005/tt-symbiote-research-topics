# Chapter 3: MTP in HuggingFace Transformers: Training-Only or Inference-Active?

## Framing Question

From the perspective of a TT-Symbiote bring-up engineer: does the MTP head need to be ported to TTNN, or can it be safely ignored for standard inference?

## Answer-First Summary

The MTP head is training-only in standard HuggingFace usage. `model.generate()` does not invoke the MTP head forward pass. The weights are loaded into the model object when using a Qwen3.6-aware model class, but they are not exercised during autoregressive generation in eval mode. For TT-Symbiote's current inference path, the MTP head weights can be safely discarded. Chapter 5 covers what is required to activate MTP for speculative decoding.

## Prerequisites

- **Chapter 1** — MTP architecture and objective
- **Chapter 2** — Weight inventory and loading

## Contents

- [`huggingface_mtp_forward_pass.md`](./huggingface_mtp_forward_pass.md) — How `Qwen3_5MoeForConditionalGeneration.forward()` gates the MTP head and why `model.generate()` never triggers it
- [`mtp_weight_loading_behavior.md`](./mtp_weight_loading_behavior.md) — The three weight-loading scenarios, which keys are required for inference, and a verification strategy
- [`mtp_inference_activation_scenarios.md`](./mtp_inference_activation_scenarios.md) — Decision table covering all scenarios from standard generation to TT-Symbiote speculative decoding, with implications for bring-up
