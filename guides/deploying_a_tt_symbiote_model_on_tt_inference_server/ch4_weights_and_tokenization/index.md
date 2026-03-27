# Chapter 4 — Weight Loading and Tokenization

## Overview

This chapter explains how tt-inference-server handles model weights and tokenization for TT Symbiote models. The design is deliberately minimal on the server side: **tt-inference-server delegates all weight management to the model itself**, and **tokenization is fully owned by vLLM via HuggingFace**.

### Key Insight

Unlike traditional inference servers that manage weight loading centrally, tt-inference-server takes a hands-off approach:

- **Weight loading** is entirely the responsibility of the model's own `initialize_vllm_model()` method. The server does not load, cache, or transform weights on the model's behalf.
- **Tokenization** is fully owned by vLLM, which loads an `AutoTokenizer` from the HuggingFace checkpoint directory. No tokenizer code is needed inside the model implementation. The model only ever receives integer token IDs.

This separation of concerns means that each TT Symbiote model can implement whatever weight loading strategy it needs — streaming from disk, converting on the fly, loading pre-tiled TTNN tensors from a side-channel cache — without any coupling to server internals.

## Reading Order

Work through the files in this order:

1. [`weight_discovery.md`](./weight_discovery.md) — How weights are downloaded, how the model receives the checkpoint path, supported checkpoint formats, pre-converted weight caches, and relevant `ModelSpec` flags.
2. [`tokenization.md`](./tokenization.md) — How vLLM handles tokenization, when and how to supply a custom tokenizer, and special token requirements.

## What's Next

Chapter 5 covers hardware initialization and device setup — how tt-inference-server acquires Tenstorrent devices, how TTNN mesh contexts are created, and what device state the model can expect to exist before `initialize_vllm_model()` is called.

See [Chapter 5 — Hardware Initialization and Device Ownership](../ch5_hardware_init/index.md).

---

**Next:** [`weight_discovery.md`](./weight_discovery.md)
