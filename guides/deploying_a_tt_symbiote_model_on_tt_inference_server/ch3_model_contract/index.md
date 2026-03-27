# Chapter 3 — The Model Implementation Contract

This chapter defines the exact interface your TT Symbiote model class must satisfy to be loaded and executed by `tt-inference-server`. The server is built on a customized vLLM runtime with a hardware-specific platform layer (`TTPlatform`) and a custom model loader (`TTModelLoader`). To hook into this runtime, your model class must fulfill two non-negotiable obligations: it must be registered in vLLM's `ModelRegistry` under a key that follows the `"TT<ArchName>"` naming convention, and it must implement a `@classmethod` named `initialize_vllm_model()` that constructs all TTNN tensors and model weights on the provided mesh device and returns a fully initialized instance ready to serve inference requests.

Everything else in this chapter — the forward-pass interface, the KV cache allocation contract, and the platform-level constraints — flows from those two requirements. Read the sections in order; each one builds on the previous.

## Reading Order

1. [class_registration.md](./class_registration.md) — How to register your model class with vLLM's `ModelRegistry` and why the `"TT"` prefix is required.
2. [initialize_vllm_model.md](./initialize_vllm_model.md) — Full signature and contract for the `initialize_vllm_model()` classmethod.
3. [forward_interface.md](./forward_interface.md) — The `prefill_forward()`, `decode_forward()`, and `allocate_kv_cache()` method signatures your class must implement.
4. [constraints.md](./constraints.md) — Platform-level constraints enforced by `TTPlatform` that every model implementation must respect.

## What's Next

Chapter 4 covers weight loading and tokenization: how the server locates and maps HuggingFace checkpoint shards onto TT hardware, and how the tokenizer is initialized and attached to the inference pipeline.

See [Chapter 4 — Weight Loading and Tokenization](../ch4_weights_and_tokenization/index.md).

---

**Next:** [class_registration.md](./class_registration.md)
