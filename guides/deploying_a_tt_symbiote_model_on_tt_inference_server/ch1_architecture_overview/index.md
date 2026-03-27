# Chapter 1 — tt-inference-server Architecture Overview

`tt-inference-server` is a workflow orchestrator and serving wrapper built around `vLLM` (or `SGLang`) as the core inference engine. It packages that engine inside Docker containers, enforces a `ModelSpec`-based configuration system that maps model names to device-specific parameters, and exposes an OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`) to callers. From the outside, `tt-inference-server` looks like any OpenAI-compatible endpoint; from the inside, each request flows through vLLM's scheduler and worker machinery until it reaches a Tenstorrent-native model class that runs on `ttnn` hardware.

## Chapter contents

| File | What you will learn |
|------|---------------------|
| [`serving_stack.md`](./serving_stack.md) | The three-layer serving stack (vLLM API server → `TTPlatform` plugin → model implementation), how vLLM is launched, what `TTPlatform` and `TTModelLoader` do, how device setup flows, and who owns tokenization |
| [`request_lifecycle.md`](./request_lifecycle.md) | End-to-end flow of a single completion request from HTTP POST through the vLLM scheduler and `TTWorker` to `prefill_forward()` / `decode_forward()` and back; KV cache block management; `TTModelInput` contents; on-device sampling |

## Reading order

Read [`serving_stack.md`](./serving_stack.md) first to understand the static architecture — which components exist and how they are wired together. Then read [`request_lifecycle.md`](./request_lifecycle.md) to understand the dynamic flow of a single inference request through that stack.

## The eight major subsystems

| Subsystem | Location | Role |
|-----------|----------|------|
| CLI entry point | `run.py` | Parses arguments and drives the top-level orchestration loop |
| `ModelSpec` registry | `workflows/model_spec.py` | Binds model names to device-specific configuration and vLLM arguments |
| Workflow orchestrator | `workflows/run_workflows.py` | Injects device-specific vLLM args and launches the API server |
| `tt-vllm-plugin` | Tenstorrent's vLLM platform plugin package | Registers TT-specific platform and loader with vLLM's plugin system |
| `TTPlatform` | `tt_vllm_plugin/platform/tt_platform.py` | Renames model architecture strings and enforces TT capability constraints |
| `TTModelLoader` | `tt_vllm_plugin/loader/tt_model_loader.py` | Validates the model class and calls `initialize_vllm_model()` |
| `TTWorker` | `tt_worker.py` | Opens mesh device, manages KV cache pool, dispatches forward calls via TTModelRunner |
| `TTModelRunner` | `tt_model_runner.py` (inside `TTWorker`) | Dispatches each step to `prefill_forward()` or `decode_forward()`; owns TTNN trace buffers |

## What's next

Chapter 2 covers the `ModelSpec` configuration schema in detail — how to register a new model, what fields are required, and how device-specific vLLM arguments are constructed and injected.

---

**Next:** [serving_stack.md](./serving_stack.md)
